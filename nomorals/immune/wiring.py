"""Wiring — the adaptive cancer gate, live in production.

Pass #28 fixed the gate's arithmetic; this module is what makes it
*run*. Before this wiring:

* nothing recorded live samples at runtime, so the baseline sat at 0
  samples forever and the gate could never arm — while ``/immune
  status`` printed "cancer gate: adaptive p95 baseline" as if live;
* the baseline was in-memory only (restart → 0 samples);
* nothing stopped an incident from teaching the baseline that a
  runaway is normal (600 recorded runaway samples pushed p95 to the
  runaway's own level, after which a runaway read "below thresholds").

What lives here (Layer 0, stdlib only — callers inject everything
deployment-shaped, exactly like ``police.py``):

* :class:`BaselineStore` — the on-disk sample store (JSON, atomic
  tmp+replace writes). Missing/corrupt/wrong-shaped content loads as
  an empty store: report-only, never a crash (fail-safe direction).
* :class:`ImmuneMonitor` — owns the shared :class:`AdaptiveBaseline`
  (recency-windowed, loaded from the store at construction, persisted
  on every change), the sampler tick, the gate consult path with the
  real enforcer, and the incident quarantine.
* :func:`shared_monitor` — the process-wide monitor per store path,
  so the scheduler sampler, the chat surface and the CLI all read and
  feed ONE baseline.
* :func:`ensure_immune_sampler_job` — registers the durable sampler
  job (``__immune_sampler__`` tool, every 1m) on the partner-runtime
  scheduler, idempotently, mirroring
  ``agents.modes.ensure_agent_modes_tick_job``.

The sampler tick is the live decision loop the baseline module was
built for::

    collect live sample → incident? quarantine it (never learn it)
                        → else consult the adaptive gate:
                            enforce → incident opens, sample quarantined
                            report  → sample learned, store persisted

**Incident quarantine is the anti-poisoning rule, by design:** while
a fever is active (any attached fever — the chat fevers register
themselves) or the gate's own incident is open, samples are counted
as quarantined but never learned, so an incident cannot move p95.
When a consulted sample comes back *below* thresholds the runaway
has stopped: the gate incident closes (and the gate-started fever
ends with it) — that closing sample is still quarantined, since it
was collected mid-incident; the next tick learns normally again.

The real enforcer for a host-population sample is the immune
system's own organism-level response: the incident opens, a bounded
:class:`~nomorals.immune.fever.Fever` starts (announced, never
silent), a receipt is kept, and learning quarantines until the
population falls back under its own baseline. A host-population
sample names no single process to arrest — arrest stays the manual
``/immune arrest <id> system`` path; nothing is fabricated here.
An extra ``enforce_fn`` may be injected (tests, future wiring); it
runs after the incident response is already in place.

**Reporting contract (Pass 3):** the sampler job's name carries the
scheduler's routine-tick marker, so the scheduler's generic per-run
success alert never fires for it — a tick that sampled nothing on
a platform that can sample (or learned routinely, or quarantined
under an already-announced incident) is a NON-event and reaches no
chat surface. A platform that cannot sample at all is different:
the tick records a declared failure (Pass #71) and the condition is
announced exactly once — see :meth:`ImmuneMonitor.tick`. Real events —
an enforcement, an announcement (fever start, incident close, the
baseline becoming ready), a tick error — are classified by
:func:`summarize_tick_event` and published once by the sampler tool
itself. Silent ticks still persist the store and are recorded as
the job's result: silence is a chat rule, not deleted evidence.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .baseline import (
    MIN_SAMPLES,
    AdaptiveBaseline,
    collect_process_sample,
    sample_over_thresholds,
    sampling_availability,
)
from .fever import Fever

__all__ = [
    "BASELINE_WINDOW",
    "STORE_ENV_VAR",
    "STORE_VERSION",
    "IMMUNE_SAMPLER_JOB",
    "LEGACY_IMMUNE_SAMPLER_JOB",
    "IMMUNE_SAMPLER_TOOL",
    "IMMUNE_SAMPLER_SPEC",
    "IMMUNE_SAMPLER_INTERVAL_S",
    "INCIDENT_FEVER_SECONDS",
    "BaselineStore",
    "ImmuneMonitor",
    "default_store_path",
    "shared_monitor",
    "reset_shared_monitors",
    "ensure_immune_sampler_job",
    "summarize_tick_event",
]

_log = logging.getLogger(__name__)

#: Rolling recency window for the live baseline. Baseline hygiene,
#: not a capability limit: 5000 one-minute samples ≈ 3.5 days of
#: behaviour — the baseline tracks how the system behaves *now*, and
#: the store cannot grow unbounded.
BASELINE_WINDOW = 5000

#: Env override for the baseline store path (mirrors the
#: ``NM_AGENT_MODES_STORE`` convention): absolute as-is, relative
#: resolved against the settings home.
STORE_ENV_VAR = "NM_IMMUNE_BASELINE_STORE"

#: On-disk store format version.
STORE_VERSION = 1

#: Durable sampler job: name, tool payload, cadence. One minute: the
#: process population is cheap to count on any platform, and a runaway is
#: visible within a minute of starting. The name carries the
#: scheduler's routine-tick marker ("tick" — the same convention as
#: "agent modes tick" / "improvement tick"), so the scheduler's
#: generic per-run success alert stays silent for this job: a
#: sampler tick is telemetry, not news. Real immune events are
#: published by the sampler tool itself (see
#: :func:`summarize_tick_event`); genuine job failures still alert
#: through the scheduler's failure-transition path.
IMMUNE_SAMPLER_JOB = "immune sampler tick"
#: The pre-Pass-3 job name. Deployments that registered the sampler
#: under it still carry that durable row; ``ensure_immune_sampler_job``
#: retires it so the old (alerting) job does not run alongside the new
#: one.
LEGACY_IMMUNE_SAMPLER_JOB = "immune baseline sampler"
IMMUNE_SAMPLER_TOOL = "__immune_sampler__"
IMMUNE_SAMPLER_SPEC = "every 1m"
IMMUNE_SAMPLER_INTERVAL_S = 60.0

#: Bounded fever started when the gate enforces — the throttle while
#: the runaway is looked at. A named bound like FEVER_MAX_SECONDS,
#: not a threshold: enforcement itself is decided by the p95 gate.
INCIDENT_FEVER_SECONDS = 300.0

#: A sampler whose last tick is older than this counts as stale.
_SAMPLER_STALE_AFTER_S = 2.5 * IMMUNE_SAMPLER_INTERVAL_S

#: Receipts/announcements kept in memory for status (disk keeps the
#: samples; receipts are a live view, not an archive).
_MAX_RECEIPTS = 100
_MAX_ANNOUNCEMENTS = 20


# ── store path ────────────────────────────────────────────────────

def _settings_home(settings: Any = None) -> Path:
    """The settings data dir, never the cwd (mirrors agents/modes)."""
    if settings is not None:
        try:
            return Path(settings.home_path)
        except Exception:  # noqa: BLE001 - duck-typed settings
            pass
    return Path.home() / ".nomorals"


def default_store_path(context: Any = None) -> Path:
    """Resolve the baseline store path.

    ``NM_IMMUNE_BASELINE_STORE`` wins when set (absolute as-is,
    relative resolved against the settings home); otherwise
    ``<settings home>/data/immune-baseline.json`` via the context's
    settings when one is supplied, else ``~/.nomorals/data/...`` —
    the same file from chat, CLI and scheduler.
    """
    settings = getattr(context, "settings", None) if context is not None \
        else None
    override = os.environ.get(STORE_ENV_VAR, "")
    if override:
        p = Path(os.path.expanduser(override))
        if p.is_absolute():
            return p
        return (_settings_home(settings) / p).resolve()
    if settings is not None:
        try:
            return Path(settings.resolve("data/immune-baseline.json"))
        except Exception:  # noqa: BLE001 - fall through to home default
            pass
    return _settings_home(settings) / "data" / "immune-baseline.json"


# ── persistent store ──────────────────────────────────────────────

def _empty_store() -> Dict[str, Any]:
    return {"samples": [], "quarantined": 0,
            "last_tick_at": 0.0, "last_sample_at": 0.0}


class BaselineStore:
    """JSON store for baseline samples + monitor counters.

    Atomic writes (pid/thread-unique tmp file in the same directory +
    ``os.replace``). Loads are fail-safe: a missing, unreadable,
    corrupt or wrong-shaped file yields an empty store — the baseline
    starts thin and report-only rather than crashing the sampler.
    """

    def __init__(self, path: Any) -> None:
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return _empty_store()
        try:
            data = json.loads(raw)
        except ValueError:
            _log.warning("immune baseline store %s is corrupt — "
                         "starting fresh (report-only)", self.path)
            return _empty_store()
        if not isinstance(data, dict):
            return _empty_store()
        samples = data.get("samples")
        if not isinstance(samples, list):
            samples = []
        samples = [s for s in samples if isinstance(s, dict)]
        out = _empty_store()
        out["samples"] = samples
        for key in ("quarantined",):
            try:
                out[key] = int(data.get(key) or 0)
            except (TypeError, ValueError):
                out[key] = 0
        for key in ("last_tick_at", "last_sample_at"):
            try:
                out[key] = float(data.get(key) or 0.0)
            except (TypeError, ValueError):
                out[key] = 0.0
        return out

    def save(self, *, samples: List[Dict[str, float]], quarantined: int,
             last_tick_at: float, last_sample_at: float) -> None:
        payload = {
            "version": STORE_VERSION,
            "samples": samples,
            "quarantined": int(quarantined),
            "last_tick_at": float(last_tick_at),
            "last_sample_at": float(last_sample_at),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(
            f"{self.path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)


# ── the live monitor ──────────────────────────────────────────────

class ImmuneMonitor:
    """The shared, persisted adaptive-baseline monitor.

    One monitor per store path (see :func:`shared_monitor`). Owns the
    baseline, the incident state and the counters; every public
    method is serialised on one lock, so the scheduler thread and
    chat threads can share it safely.
    """

    def __init__(self, path: Any = None, *,
                 window: Optional[int] = BASELINE_WINDOW,
                 enforce_fn: Optional[Callable[[Any], Any]] = None,
                 collect: Optional[Callable[..., Optional[Dict[str, float]]]] = None,
                 availability: Optional[Callable[[], Dict[str, Any]]] = None,
                 ) -> None:
        self.store = BaselineStore(
            path if path is not None else default_store_path())
        self.baseline = AdaptiveBaseline(window=window)
        data = self.store.load()
        for sample in data["samples"]:
            self.baseline.record(sample)
        self._quarantined = data["quarantined"]
        self._last_tick_at = data["last_tick_at"]
        self._last_sample_at = data["last_sample_at"]
        self._extra_enforce_fn = enforce_fn
        self._collect = collect
        self._availability_probe = availability
        #: The sampling-unavailable condition is announced exactly
        #: once per monitor lifetime (a standing platform fact is
        #: news once, not every minute) — see :meth:`tick`.
        self._sampling_unavailable_announced = False
        self._lock = threading.Lock()
        #: The organism fever the gate itself starts on enforcement.
        self.fever = Fever(announce_fn=self._announce)
        self._fevers: List[Fever] = [self.fever]
        self._gate_incident = False
        self._fever_started_by_gate = False
        self.announcements: List[str] = []
        #: Announcements not yet handed to a delivery surface. The
        #: sampler tool drains these once per tick and publishes the
        #: real events; without a drainer they simply wait here (the
        #: full history stays in ``announcements`` / the log).
        self._pending_announcements: List[str] = []
        self.receipts: List[Any] = []
        self.last_enforcement: Optional[Dict[str, Any]] = None

    # ── fevers / incidents ────────────────────────────────────────

    def attach_fever(self, fever: Fever) -> None:
        """Register another fever (e.g. a chat state's) for quarantine.

        While ANY attached fever is active, samples quarantine.
        Idempotent per fever object.
        """
        with self._lock:
            if all(f is not fever for f in self._fevers):
                self._fevers.append(fever)

    def fever_active(self, now: Optional[float] = None) -> bool:
        return any(f.is_active(now) for f in self._fevers)

    def incident_active(self, now: Optional[float] = None) -> bool:
        """True while samples must quarantine: a fever or the gate."""
        return self._gate_incident or self.fever_active(now)

    def _announce(self, message: str) -> None:
        self.announcements.append(message)
        del self.announcements[:-_MAX_ANNOUNCEMENTS]
        self._pending_announcements.append(message)
        del self._pending_announcements[:-_MAX_ANNOUNCEMENTS]
        _log.info("immune: %s", message)

    def drain_announcements(self) -> List[str]:
        """Announcements made since the last drain (and clear them).

        The delivery handoff for the monitor's news: fever starts,
        incident closes, the baseline becoming ready. Announcements
        also stay in :attr:`announcements` (history) and the log —
        draining decides *notification*, never the evidence.
        """
        with self._lock:
            pending = list(self._pending_announcements)
            self._pending_announcements.clear()
            return pending

    def _enforce(self, sample: Dict[str, Any]) -> None:
        """The real enforcer handed to the adaptive gate.

        Opens the gate incident and starts the bounded organism
        fever (announced). Runs inside the gate call, under the
        monitor lock when reached through this monitor.
        """
        self._gate_incident = True
        self.last_enforcement = {
            "ts": time.time(),
            "sample": dict(sample),
            "spawned_p95": self.baseline.thresholds().spawned_p95,
            "growth_p95": self.baseline.thresholds().growth_p95,
        }
        if not self.fever.is_active():
            self.fever.start(
                "adaptive cancer gate enforced on a system process "
                f"(spawned {float(sample.get('processes_spawned', 0) or 0):g}, "
                f"growth {float(sample.get('process_growth', sample.get('growth', 0)) or 0):g})",
                INCIDENT_FEVER_SECONDS,
            )
            self._fever_started_by_gate = True
        if self._extra_enforce_fn is not None:
            try:
                self._extra_enforce_fn(sample)
            except Exception:  # noqa: BLE001 - the incident response is
                _log.exception("immune: injected enforce_fn failed")  # already in place

    def _close_gate_incident(self, now: Optional[float] = None) -> None:
        """Close the gate incident: the runaway population stopped."""
        self._gate_incident = False
        if self._fever_started_by_gate:
            self._fever_started_by_gate = False
            if self.fever.active:
                report = self.fever.end(now=now, resolved=True)
                self._announce(
                    f"immune incident closed — {report.note}")

    def _remember_receipt(self, receipt: Any) -> None:
        self.receipts.append(receipt)
        del self.receipts[:-_MAX_RECEIPTS]

    # ── persistence ───────────────────────────────────────────────

    def _persist(self) -> None:
        try:
            self.store.save(
                samples=self.baseline.samples(),
                quarantined=self._quarantined,
                last_tick_at=self._last_tick_at,
                last_sample_at=self._last_sample_at)
        except OSError:
            _log.exception("immune: baseline store write failed (%s)",
                           self.store.path)

    def persist(self) -> None:
        """Flush baseline + counters to the store now."""
        with self._lock:
            self._persist()

    # ── sampling ──────────────────────────────────────────────────

    def collect_sample(self) -> Optional[Dict[str, float]]:
        """One live process sample (None where sampling cannot run)."""
        collect = self._collect or collect_process_sample
        return collect(self.baseline)

    def _probe_sampling(self) -> Dict[str, Any]:
        """Can this platform produce a process sample at all?

        The availability fact (:func:`sampling_availability`, or the
        injected probe): ``{"available", "backend", "platform",
        "reason"}``. A probe failure is itself an unavailable answer
        with the failure as the reason — never an exception, never a
        guessed "available".
        """
        probe = self._availability_probe or sampling_availability
        try:
            avail = probe()
        except Exception as exc:  # noqa: BLE001 - probe must not raise
            return {"available": False, "backend": "", "platform": "",
                    "reason": f"sampling availability probe failed: {exc}"}
        if not isinstance(avail, dict) or "available" not in avail:
            return {"available": False, "backend": "", "platform": "",
                    "reason": "sampling availability probe returned "
                              f"no verdict: {avail!r}"}
        return avail

    def tick(self, sample: Optional[Dict[str, Any]] = None, *,
             now: Optional[float] = None) -> Dict[str, Any]:
        """One sampler tick — the live decision loop (module docstring).

        ``sample`` may be injected (tests, manual probes); by default
        the real sample is collected live from the host. Returns a
        report dict: what was sampled, whether it was learned or
        quarantined, and whether the gate enforced.

        A tick that collects nothing splits in two, honestly: when
        the platform CAN sample but this tick got no sample, the
        report is the Pass-3 no-sample non-event (``ok``). When the
        platform cannot sample at all — a standing fact, probed via
        :meth:`_probe_sampling` — the report is a declared failure
        (``ok: False``, ``unavailable: True``) so the scheduler's run
        record tells the truth under the Pass #71 contract instead of
        logging an ok-success with "sampled: no" buried in it, and
        the condition is announced exactly once (the monitor's
        announcement, drained and published by the sampler tool) —
        a permanent platform fact must not page every minute.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._last_tick_at = now
            live = sample if sample is not None else self.collect_sample()
            if live is None:
                avail = self._probe_sampling()
                if not avail.get("available"):
                    reason = str(avail.get("reason") or
                                 "no backend could count processes")
                    if not self._sampling_unavailable_announced:
                        self._sampling_unavailable_announced = True
                        self._announce(
                            "immune process sampling unavailable on "
                            f"this platform — {reason}; no sample can "
                            "be collected here, so the adaptive "
                            "baseline cannot learn and the gate stays "
                            "report-only")
                    self._persist()
                    return {"ok": False, "unavailable": True,
                            "sampling_available": False,
                            "backend": str(avail.get("backend") or ""),
                            "error": "process sampling unavailable on "
                                     f"this platform: {reason}",
                            "sampled": False, "learned": False,
                            "enforced": False,
                            "quarantined": self._quarantined,
                            "samples": len(self.baseline),
                            "reason": "process sampling unavailable on "
                                      f"this platform: {reason}"}
                self._persist()
                return {"ok": True, "sampled": False, "learned": False,
                        "enforced": False, "quarantined": self._quarantined,
                        "samples": len(self.baseline),
                        "reason": "no live process sample available on "
                                  "this platform — nothing recorded"}
            self._last_sample_at = now
            if self.incident_active(now):
                # Quarantine. If this sample shows the runaway stopped
                # (below both thresholds), close the gate incident —
                # the sample itself still quarantines: it was
                # collected mid-incident.
                if self._gate_incident and not sample_over_thresholds(
                        live, self.baseline.thresholds()):
                    self._close_gate_incident(now)
                self._quarantined += 1
                self._persist()
                return {"ok": True, "sampled": True, "learned": False,
                        "enforced": False, "incident": True,
                        "quarantined": self._quarantined,
                        "samples": len(self.baseline),
                        "sample": dict(live),
                        "reason": "incident active (fever or gate "
                                  "enforcement) — sample quarantined, "
                                  "not learned"}
            result = self._consult_locked(
                live, is_system_process=True, owner_directed=False)
            if result["action"] == "enforce":
                self._quarantined += 1
                self._persist()
                return {"ok": True, "sampled": True, "learned": False,
                        "enforced": True, "incident": True,
                        "quarantined": self._quarantined,
                        "samples": len(self.baseline),
                        "sample": dict(live),
                        "reason": result["reason"]}
            was_ready = self.baseline.thresholds().status == "ready"
            self.baseline.record(live)
            if not was_ready and \
                    self.baseline.thresholds().status == "ready":
                # A discrete learning milestone — the baseline just
                # became able to enforce. Announced (so it is news
                # exactly once); routine samples into an already-warm
                # baseline announce nothing.
                th = self.baseline.thresholds()
                self._announce(
                    "immune baseline ready — "
                    f"{len(self.baseline)} samples learned "
                    f"(spawned p95 {th.spawned_p95:g}, "
                    f"growth p95 {th.growth_p95:g}); "
                    "the adaptive gate can now enforce")
            self._persist()
            return {"ok": True, "sampled": True, "learned": True,
                    "enforced": False, "incident": False,
                    "quarantined": self._quarantined,
                    "samples": len(self.baseline),
                    "sample": dict(live),
                    "reason": result["reason"]}

    # ── the gate ──────────────────────────────────────────────────

    def _consult_locked(self, sample: Dict[str, Any], *,
                        is_system_process: bool,
                        owner_directed: bool) -> Dict[str, Any]:
        from .baseline import adaptive_consult_then_enforce

        result = adaptive_consult_then_enforce(
            sample, self.baseline,
            is_system_process=is_system_process,
            owner_directed=owner_directed,
            enforce_fn=self._enforce)
        self._remember_receipt(result["receipt"])
        # A below-threshold consult while a gate incident is open is
        # the runaway stopping: close the incident. (The insufficient
        # receipt carries no "thresholds" detail — it cannot judge.)
        if (result["action"] == "report_only" and self._gate_incident
                and "thresholds" in (result.get("details") or {})):
            self._close_gate_incident()
        return result

    def consult(self, sample: Dict[str, Any], *,
                is_system_process: bool = False,
                owner_directed: bool = False) -> Dict[str, Any]:
        """Consult the adaptive gate on ``sample`` (never learns it).

        Learning is the sampler's job; this is the decision path —
        used by the chat ``/immune gate`` probe and any future caller
        policing a spawn. The enforcer is the monitor's real incident
        response (:meth:`_enforce`).
        """
        with self._lock:
            return self._consult_locked(
                sample, is_system_process=is_system_process,
                owner_directed=owner_directed)

    # ── status ────────────────────────────────────────────────────

    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        """The live gate state, for surfaces — every field measured."""
        now = time.time() if now is None else now
        with self._lock:
            th = self.baseline.thresholds()
            tick_age = (now - self._last_tick_at) \
                if self._last_tick_at else None
            avail = self._probe_sampling()
            return {
                "samples": len(self.baseline),
                "min_samples": MIN_SAMPLES,
                "baseline_status": th.status,
                "ready": th.status == "ready",
                "sampling_available": bool(avail.get("available")),
                "sampling_backend": str(avail.get("backend") or ""),
                "sampling_reason": str(avail.get("reason") or ""),
                "spawned_p95": th.spawned_p95,
                "growth_p95": th.growth_p95,
                "threshold_reason": th.reason,
                "quarantined": self._quarantined,
                "incident_active": self.incident_active(now),
                "fever_active": self.fever_active(now),
                "gate_incident": self._gate_incident,
                "last_tick_at": self._last_tick_at or None,
                "last_tick_age_s": tick_age,
                "last_sample_at": self._last_sample_at or None,
                "sampler_running": bool(
                    tick_age is not None and tick_age <= _SAMPLER_STALE_AFTER_S),
                "sampler_interval_s": IMMUNE_SAMPLER_INTERVAL_S,
                "store_path": str(self.store.path),
                "window": self.baseline.window,
                "receipts": len(self.receipts),
                "last_enforcement": self.last_enforcement,
            }


# ── the tick reporting contract ───────────────────────────────────

def _sample_desc(sample: Any) -> str:
    if not isinstance(sample, dict):
        return ""
    try:
        return (f"spawned {float(sample.get('processes_spawned', 0) or 0):g}, "
                f"growth {float(sample.get('process_growth', 0) or 0):g}")
    except (TypeError, ValueError):
        return ""


def summarize_tick_event(result: Any,
                         announcements: Any = None) -> Optional[Dict[str, Any]]:
    """Compose THE owner notification for one sampler tick, or None.

    The reporting contract: a tick is news only when something
    actually happened —

    * the tick **errored** (an error is an event),
    * the gate **enforced** (critical: an incident opened),
    * the monitor **announced** (a fever started, an incident
      closed, the baseline became ready).

    Everything else is a non-event and returns None: no live sample
    on a platform that can sample, a routine sample learned into an
    already-warm baseline, a sample quarantined under an
    already-announced incident. (A platform that cannot sample at
    all is a standing condition, not a per-tick event: it is
    announced exactly once by the monitor and classified here, and
    its ticks record as declared failures — see :meth:`tick`.)
    Non-events are still recorded — the tick result is the
    job's ``last_result`` and the baseline store keeps the tick and
    sample timestamps — silence is a chat-surface rule, not deleted
    evidence.

    Returns ``{"title": str, "body": str, "critical": bool}``.
    """
    if not isinstance(result, dict):
        return None
    notes = [str(a).strip() for a in (announcements or []) if str(a).strip()]
    context: List[str] = []
    desc = _sample_desc(result.get("sample"))
    if desc:
        context.append(f"sample: {desc}")
    context.append(
        f"quarantined: {result.get('quarantined', 0)} · "
        f"baseline samples: {result.get('samples', 0)}")
    if result.get("unavailable"):
        # A standing platform condition, not a per-tick event: the
        # monitor announces it exactly once, and that announcement
        # (drained into ``notes`` on the transition tick) is the one
        # notification. Later unavailable ticks return None — the
        # failed run record is their evidence, not a page a minute.
        if notes:
            return {"title": f"🛡️ {notes[0]}",
                    "body": "\n".join(notes[1:] + context),
                    "critical": False}
        return None
    if result.get("ok") is False or result.get("error"):
        return {"title": "❌ immune sampler error",
                "body": str(result.get("error") or "unknown error"),
                "critical": False}
    if result.get("enforced"):
        lines = [str(result.get("reason") or "the adaptive gate enforced")]
        lines += context
        lines += notes
        return {"title": "🛡️ immune gate ENFORCED",
                "body": "\n".join(lines), "critical": True}
    if notes:
        return {"title": f"🛡️ {notes[0]}",
                "body": "\n".join(notes[1:] + context),
                "critical": False}
    return None


# ── shared monitor ────────────────────────────────────────────────

_MONITORS: Dict[str, ImmuneMonitor] = {}
_MONITORS_LOCK = threading.Lock()


def shared_monitor(path: Any = None, **kwargs: Any) -> ImmuneMonitor:
    """The process-wide monitor for a store path (one per path).

    The scheduler sampler, the chat surface and the CLI resolve the
    same default path, so they share ONE monitor — one baseline, one
    incident state, one quarantine count. Extra ``kwargs`` apply only
    when the monitor is first created for the path.
    """
    resolved = Path(path) if path is not None else default_store_path()
    key = str(resolved)
    with _MONITORS_LOCK:
        monitor = _MONITORS.get(key)
        if monitor is None:
            monitor = ImmuneMonitor(resolved, **kwargs)
            _MONITORS[key] = monitor
        return monitor


def reset_shared_monitors() -> None:
    """Drop all shared monitors (tests)."""
    with _MONITORS_LOCK:
        _MONITORS.clear()


# ── scheduler registration ────────────────────────────────────────

_INTERVAL_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}


def _interval_seconds(spec: Any) -> Optional[float]:
    """``"every 1m"`` / ``"every 60s"`` / ``"60"`` → seconds, else None."""
    s = str(spec or "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"every\s+(\d+(?:\.\d+)?)\s*([a-z]+)", s, re.IGNORECASE)
    if not m:
        return None
    unit = m.group(2).lower()
    if unit not in _INTERVAL_UNITS:
        return None
    return float(m.group(1)) * _INTERVAL_UNITS[unit]


def _spec_matches(stored: Any, spec: str) -> bool:
    """True when a stored job spec is the requested one.

    ``every``-kind jobs may be stored as plain seconds or as
    ``"every 60s"`` by the scheduler's formatter, so those compare
    numerically; everything else compares verbatim. (Same rule as
    ``agents.modes`` — without it the idempotency check can never
    succeed and every boot re-registers the job.)
    """
    if str(stored or "") == spec:
        return True
    want = _interval_seconds(spec)
    have = _interval_seconds(stored)
    return want is not None and have is not None and want == have


def _job_field(job: Any, name: str) -> Any:
    if isinstance(job, dict):
        return job.get(name, "")
    return getattr(job, name, "")


def ensure_immune_sampler_job(scheduler: Any, *,
                              spec: str = IMMUNE_SAMPLER_SPEC) -> Any:
    """Register the durable immune-baseline sampler job. Idempotent.

    A job with the sampler's name already at the requested cadence is
    returned as-is; a stale one is replaced. A job still carrying the
    legacy (pre-Pass-3) name is retired first, so an upgraded
    deployment does not run the old alerting job alongside the new
    silent one. Best-effort: never raises (a missing scheduler just
    means no scheduled sampling — ``/immune gate`` still samples on
    demand).
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            if _job_field(job, "name") == LEGACY_IMMUNE_SAMPLER_JOB:
                try:
                    scheduler.remove(_job_field(job, "id"))
                except Exception:  # noqa: BLE001 - legacy cleanup is
                    pass  # best-effort; the new job still registers
        for job in (scheduler.list_jobs() or []):
            if _job_field(job, "name") == IMMUNE_SAMPLER_JOB:
                if _spec_matches(_job_field(job, "spec"), spec):
                    return job
                try:
                    scheduler.remove(_job_field(job, "id"))
                except Exception:  # noqa: BLE001 - stale job removal
                    pass
                break
        return scheduler.add(IMMUNE_SAMPLER_JOB, spec, "tool",
                             {"tool": IMMUNE_SAMPLER_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
