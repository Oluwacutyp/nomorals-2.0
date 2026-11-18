"""Outward-facing outage watch (Downdetector native build, 52-websites #45).

Downdetector detects outages by BASELINE DEVIATION: for each service
a baseline of typical report volume is computed, the status is a
three-state verdict — no problems / possible problems / problems —
and an incident is declared only when current volume sits
significantly above the baseline.

Devon already has the detection mathematics pointed at herself
(:mod:`nomorals.immune.baseline` — adaptive percentile baselines that
refuse to enforce on a thin baseline). What she lacked is the
outward-facing version: a watchlist of external services the owner
depends on (exchanges, brokers, MT5 bridges, Telegram, GitHub, his
ISP…), actively probed, with the baseline method applied to probe
results and public signals, and a verdict surface.

Her "crowd" is one user, so the report stream is replaced by (a) her
own active probes from the host, (b) public status-page feeds, and
(c) social chatter mined by her existing social/search organs
(open-sources law: those feeds are seeds; the organ discovers more —
the tool accepts injected ``signal_refs`` for that rung today, and no
automatic social miner is wired yet). The honest limit is stated on
every verdict: Downdetector's real asset is millions of human
reporters, and no native build replicates it. Devon's substitute
sensors detect "unreachable/broken from here + corroborating public
signal" — a different, narrower claim, labelled as such in every
verdict. Geo-distribution is likewise limited to her hosts.

Baseline idiom (from :mod:`nomorals.immune.baseline`, adapted, not
copied): ``MIN_SAMPLES = 30`` — below that the baseline reports
``"insufficient"`` and the verdict is ``"unknown"``, never a confident
call. At >= 30 samples the failure rate and the p95 latency are the
baseline. One deliberate difference from the immune class: its sample
keys are process-shaped (``processes_spawned`` / ``process_growth``)
and its thresholds are process thresholds. Forcing probe outcomes
into those keys would lie about what the numbers are, so the
watchtower carries its own probe-shaped baseline following the same
idiom — same constants, same honesty contract, honest keys.

Verdict rules (baseline = samples EXCLUDING the recent window, so
the current window is judged against history, not against itself):
- ``down``: 5 consecutive failures; or >= 3 consecutive failures
  with the recent failure rate above baseline + 0.25; or a recent
  failure rate >= 0.5 above baseline + 0.25; or a major public
  signal plus >= 2 recent failures.
- ``possible``: the recent failure rate above
  max(baseline * 2, baseline + 0.05); or the recent median latency
  above p95 * 1.5; or any corroborating public signal.
- ``ok``: everything else. A lone failure on a perfect baseline is
  ``possible`` — noise, never an incident.
"""

from __future__ import annotations

import contextlib
import json
import socket
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MIN_SAMPLES",
    "PERCENTILE",
    "RECENT_WINDOW",
    "BASELINE_WINDOW",
    "WATCHTOWER_SPEC",
    "WATCHTOWER_JOB",
    "WATCHTOWER_TOOL",
    "WATCHTOWER_FILE",
    "BaselineThresholds",
    "Incident",
    "ProbeBaseline",
    "ProbeSample",
    "Verdict",
    "WatchTarget",
    "WatchtowerStore",
    "check_target",
    "ensure_watchtower_job",
    "evaluate_verdict",
    "format_watchtower_reply",
    "probe_feed",
    "probe_http",
    "probe_target",
    "probe_tcp",
    "run_watchtower_tick_once",
]

#: Minimum samples before any baseline is trusted (immune idiom).
MIN_SAMPLES = 30
#: Percentile used for the latency baseline (immune idiom).
PERCENTILE = 95.0
#: The current window judged against the baseline.
RECENT_WINDOW = 10
#: Trailing cap on samples kept per target.
BASELINE_WINDOW = 500
#: Five straight failures is a total outage from here, whatever the
#: baseline says.
DOWN_CONSECUTIVE = 5
#: Three straight failures plus a real deviation is an incident.
DOWN_MIN_CONSECUTIVE = 3
#: "Significantly above baseline" for the down call.
DOWN_MARGIN = 0.25
#: Sustained failure rate that counts as down on its own.
DOWN_MIN_RATE = 0.5
#: "Above baseline" for the possible call: twice the baseline rate,
#: or baseline + 5 points, whichever is higher.
POSSIBLE_MARGIN = 0.05
#: Latency leg: recent median above p95 times this is "possible".
LATENCY_MULTIPLIER = 1.5
#: Scheduler beat: Downdetector re-evaluates every 4 minutes.
WATCHTOWER_SPEC = "*/4 * * * *"
WATCHTOWER_JOB = "watchtower tick"
WATCHTOWER_TOOL = "__watchtower_tick__"
WATCHTOWER_FILE = "watchtower.json"

#: Verdict states (Downdetector's three, plus the honest fourth).
STATES = ("ok", "possible", "down", "unknown")

#: The narrow-claim label carried by every verdict and reply.
VANTAGE = ("single-host probe + public status feed — not crowd "
           "reports (Downdetector's crowd is millions of reporters; "
           "this verdict is one host's view)")

#: Probe kinds.
KINDS = ("http", "tcp", "feed")

#: Feed-body markers that count as corroborating public signal.
OUTAGE_MARKERS = (
    "major outage",
    "major incident",
    "system outage",
    "partial outage",
    "degraded performance",
    "service disruption",
)
#: Markers that mean the service itself says it is down.
MAJOR_MARKERS = ("major outage", "major incident", "system outage")

#: Healthy status-page values (a feed's own status field).
FEED_HEALTHY = ("ok", "operational", "none", "all systems operational")
#: Status-page values that mean degraded but reachable.
FEED_DEGRADED = ("partial_outage", "degraded_performance",
                 "under_maintenance", "maintenance")


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over sorted ``values``."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _is_major_signal(ref: str) -> bool:
    lowered = ref.lower()
    return any(marker in lowered for marker in MAJOR_MARKERS)


def _normalize_sample(sample: Any) -> ProbeSample:
    if isinstance(sample, ProbeSample):
        return sample
    if isinstance(sample, dict):
        return ProbeSample.from_dict(sample)
    raise TypeError(f"not a probe sample: {type(sample)!r}")


@dataclass
class WatchTarget:
    """One external service under watch."""

    name: str
    url: str
    kind: str = "http"
    critical: bool = False
    added_at: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "url": self.url, "kind": self.kind,
                "critical": self.critical, "added_at": self.added_at,
                "note": self.note}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WatchTarget:
        return cls(name=str(raw.get("name", "")),
                   url=str(raw.get("url", "")),
                   kind=str(raw.get("kind", "http")),
                   critical=bool(raw.get("critical", False)),
                   added_at=float(raw.get("added_at", 0.0) or 0.0),
                   note=str(raw.get("note", "")))


@dataclass
class ProbeSample:
    """One probe outcome. Never invented: a failed probe records its
    error, and a latency that was never measured stays ``None``."""

    at: float
    ok: bool
    latency_ms: float | None = None
    status: int | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"at": self.at, "ok": self.ok,
                "latency_ms": self.latency_ms, "status": self.status,
                "error": self.error}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProbeSample:
        latency = raw.get("latency_ms")
        status = raw.get("status")
        return cls(at=float(raw.get("at", 0.0) or 0.0),
                   ok=bool(raw.get("ok", False)),
                   latency_ms=(None if latency is None
                               else float(latency)),
                   status=(None if status is None else int(status)),
                   error=str(raw.get("error", "")))


@dataclass
class BaselineThresholds:
    """Outcome of :meth:`ProbeBaseline.thresholds`."""

    status: str = "insufficient"  # insufficient | ready
    samples: int = 0
    min_samples: int = MIN_SAMPLES
    failure_rate: float = 0.0
    latency_p95: float | None = None
    latency_ready: bool = False
    reason: str = ""


@dataclass
class Verdict:
    """A three-state verdict over one target, with its evidence."""

    target: str
    state: str  # ok | possible | down | unknown
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    samples: int = 0
    at: float = 0.0
    vantage: str = VANTAGE

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "state": self.state,
                "reason": self.reason, "evidence": self.evidence,
                "samples": self.samples, "at": self.at,
                "vantage": self.vantage}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Verdict:
        return cls(target=str(raw.get("target", "")),
                   state=str(raw.get("state", "unknown")),
                   reason=str(raw.get("reason", "")),
                   evidence=dict(raw.get("evidence") or {}),
                   samples=int(raw.get("samples", 0) or 0),
                   at=float(raw.get("at", 0.0) or 0.0),
                   vantage=str(raw.get("vantage", VANTAGE)))


@dataclass
class Incident:
    """An open incident: state, since when, with what evidence."""

    target: str
    state: str  # possible | down
    since: float
    evidence: dict[str, Any] = field(default_factory=dict)
    alerted: bool = False
    resolved_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "state": self.state,
                "since": self.since, "evidence": self.evidence,
                "alerted": self.alerted, "resolved_at": self.resolved_at}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Incident:
        resolved = raw.get("resolved_at")
        return cls(target=str(raw.get("target", "")),
                   state=str(raw.get("state", "possible")),
                   since=float(raw.get("since", 0.0) or 0.0),
                   evidence=dict(raw.get("evidence") or {}),
                   alerted=bool(raw.get("alerted", False)),
                   resolved_at=(None if resolved is None
                                else float(resolved)))


class ProbeBaseline:
    """Accumulates probe samples; derives the baseline.

    The immune idiom, probe-shaped: below ``MIN_SAMPLES`` the
    baseline reports ``"insufficient"`` and any verdict built on it
    is ``"unknown"``. At >= 30 samples the failure rate and the p95
    latency are the baseline. A latency leg that never measured a
    real latency (p95 <= 0 or no latency at all) is flagged
    ``latency_ready=False`` and skipped by the verdict — the failure
    legs still judge, because a dead service must read "down", not
    "unknown", even when it never answered with a latency.
    """

    def __init__(self, window: int | None = BASELINE_WINDOW) -> None:
        if window is not None and int(window) < 1:
            raise ValueError("baseline window must be a positive count")
        self._samples: list[ProbeSample] = []
        self._window: int | None = int(window) if window else None

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def window(self) -> int | None:
        return self._window

    def samples(self) -> list[ProbeSample]:
        """A copy of the recorded samples, oldest first."""
        return list(self._samples)

    def record(self, sample: Any) -> int:
        """Record one sample. Returns the total sample count."""
        self._samples.append(_normalize_sample(sample))
        if self._window is not None and len(self._samples) > self._window:
            del self._samples[: len(self._samples) - self._window]
        return len(self._samples)

    def failure_rate(self) -> float:
        """Fraction of samples that failed (0.0 when empty)."""
        if not self._samples:
            return 0.0
        failed = sum(1 for s in self._samples if not s.ok)
        return failed / len(self._samples)

    def latency_p95(self) -> float | None:
        """p95 of measured latencies on ok samples (None when none)."""
        latencies = [s.latency_ms for s in self._samples
                     if s.ok and s.latency_ms is not None]
        if not latencies:
            return None
        return _percentile(latencies, PERCENTILE)

    def thresholds(self) -> BaselineThresholds:
        """Baseline readiness, or insufficient when the baseline is thin."""
        n = len(self._samples)
        if n < MIN_SAMPLES:
            return BaselineThresholds(
                status="insufficient", samples=n,
                failure_rate=self.failure_rate(),
                reason=(f"baseline from {n} samples, need "
                        f"{MIN_SAMPLES} — report-only, never a "
                        "confident verdict"),
            )
        p95 = self.latency_p95()
        latency_ready = p95 is not None and p95 > 0
        return BaselineThresholds(
            status="ready", samples=n,
            failure_rate=self.failure_rate(),
            latency_p95=p95 if latency_ready else None,
            latency_ready=latency_ready,
            reason=(f"baseline from {n} samples: failure rate "
                    f"{self.failure_rate():.3f}, latency p95 "
                    f"{p95 if p95 is not None else 'n/a'} ms"),
        )


def _consecutive_failures(samples: list[ProbeSample]) -> int:
    count = 0
    for sample in reversed(samples):
        if sample.ok:
            break
        count += 1
    return count


def evaluate_verdict(name: str, samples: list[Any], *,
                     signals: list[str] | None = None,
                     now: float | None = None) -> Verdict:
    """Judge one target's samples into a three-state verdict.

    The baseline is the samples EXCLUDING the recent window, so the
    current window is judged against history, not against itself.
    Never invents: below ``MIN_SAMPLES`` the verdict is ``"unknown"``
    with the shortfall named. ``signals`` are corroborating public
    signals (feed markers, injected social/search refs) — they can
    lift ``ok`` to ``possible`` and, with recent failures, to
    ``down``, but they never replace the probe evidence.
    """
    now_f = float(now if now is not None else time.time())
    clean = [_normalize_sample(s) for s in samples]
    n = len(clean)
    signal_refs = [str(s) for s in (signals or []) if str(s).strip()]
    if n < MIN_SAMPLES:
        return Verdict(
            target=name, state="unknown",
            reason=(f"insufficient baseline ({n}/{MIN_SAMPLES} "
                    "samples) — report-only, never a confident verdict"),
            evidence={"probe_failures": 0, "baseline_deviation": 0.0,
                      "signal_refs": signal_refs,
                      "consecutive_failures": 0, "baseline_samples": n,
                      "recent_window": min(n, RECENT_WINDOW)},
            samples=n, at=now_f)
    baseline_leg = (clean[:-RECENT_WINDOW]
                    if len(clean) > RECENT_WINDOW else clean)
    recent = clean[-RECENT_WINDOW:]
    base = ProbeBaseline()
    for sample in baseline_leg:
        base.record(sample)
    thresholds = base.thresholds()
    baseline_rate = thresholds.failure_rate
    recent_failures = sum(1 for s in recent if not s.ok)
    recent_rate = recent_failures / len(recent) if recent else 0.0
    consecutive = _consecutive_failures(clean)
    deviation = round(recent_rate - baseline_rate, 3)
    recent_latencies = [s.latency_ms for s in recent
                        if s.ok and s.latency_ms is not None]
    recent_median = (statistics.median(recent_latencies)
                     if recent_latencies else None)
    evidence = {"probe_failures": recent_failures,
                "baseline_deviation": deviation,
                "signal_refs": signal_refs,
                "consecutive_failures": consecutive,
                "baseline_samples": len(baseline_leg),
                "recent_window": len(recent)}
    major_signal = any(_is_major_signal(ref) for ref in signal_refs)
    rate_elevated = recent_rate > max(baseline_rate * 2,
                                      baseline_rate + POSSIBLE_MARGIN)
    latency_elevated = (thresholds.latency_ready
                        and recent_median is not None
                        and thresholds.latency_p95 is not None
                        and recent_median
                        > thresholds.latency_p95 * LATENCY_MULTIPLIER)
    state = "ok"
    reason = (f"within baseline: {recent_failures} failures in the "
              f"last {len(recent)} probes, baseline failure rate "
              f"{baseline_rate:.3f}")
    if (consecutive >= DOWN_CONSECUTIVE
            or (consecutive >= DOWN_MIN_CONSECUTIVE
                and recent_rate > baseline_rate + DOWN_MARGIN)
            or (recent_rate >= DOWN_MIN_RATE
                and recent_rate > baseline_rate + DOWN_MARGIN)
            or (major_signal and recent_failures >= 2)):
        state = "down"
        reason = (f"significantly above baseline: {recent_failures} "
                  f"failures in the last {len(recent)} probes "
                  f"(baseline failure rate {baseline_rate:.3f}, "
                  f"deviation {deviation:+.3f}), "
                  f"{consecutive} consecutive")
        if major_signal:
            reason += " + corroborating public signal"
    elif rate_elevated or latency_elevated or signal_refs:
        state = "possible"
        if signal_refs:
            reason = (f"corroborating public signal: "
                      f"{signal_refs[0]} (probes: {recent_failures} "
                      f"failures in the last {len(recent)})")
        elif latency_elevated:
            reason = (f"latency elevated: recent median "
                      f"{recent_median:.0f} ms vs baseline p95 "
                      f"{thresholds.latency_p95:.0f} ms")
        else:
            reason = (f"elevated failures: {recent_failures} in the "
                      f"last {len(recent)} probes (baseline failure "
                      f"rate {baseline_rate:.3f}, deviation "
                      f"{deviation:+.3f})")
    return Verdict(target=name, state=state, reason=reason,
                   evidence=evidence, samples=n, at=now_f)


# ── probes ────────────────────────────────────────────────────────────


def _default_fetcher(url: str, timeout: float,
                     ) -> tuple[int, bytes, float]:
    """One real HTTP GET. Returns (status, body, latency_ms)."""
    started = time.perf_counter()
    request = urllib.request.Request(
        url, headers={"User-Agent": "Devon-watchtower/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read()
            status = int(resp.status)
    except urllib.error.HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        status = int(exc.code)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return status, body, latency_ms


def _fetch_once(url: str, timeout: float, fetcher: Any,
                at: float) -> tuple[ProbeSample, bytes]:
    """One fetch, returning (sample, body). A failure is a sample
    with an empty body, never a raise."""
    fetch = fetcher or _default_fetcher
    try:
        status, body, latency_ms = fetch(url, timeout)
    except Exception as exc:  # noqa: BLE001 — a probe must never raise
        return (ProbeSample(at=at, ok=False, latency_ms=None,
                            error=str(exc)[:300]), b"")
    ok = 200 <= status <= 399
    sample = ProbeSample(at=at, ok=ok, latency_ms=float(latency_ms),
                         status=status,
                         error="" if ok else f"http {status}")
    return sample, body


def probe_http(url: str, *, timeout: float = 10.0,
               fetcher: Any = None, now: float | None = None) -> ProbeSample:
    """One HTTP probe. A failure is a sample, never a raise."""
    at = float(now if now is not None else time.time())
    sample, _body = _fetch_once(url, timeout, fetcher, at)
    return sample


def probe_tcp(url: str, *, timeout: float = 10.0,
              connector: Any = None,
              now: float | None = None) -> ProbeSample:
    """One TCP connect probe. ``url`` is ``host[:port]`` (port 443
    when absent). A failure is a sample, never a raise."""
    at = float(now if now is not None else time.time())
    host = url
    port = 443
    if ":" in url:
        host, _, port_s = url.rpartition(":")
        try:
            port = int(port_s)
        except ValueError:
            return ProbeSample(at=at, ok=False,
                               error=f"bad tcp target: {url!r}")
    if not host:
        return ProbeSample(at=at, ok=False,
                           error=f"bad tcp target: {url!r}")
    started = time.perf_counter()
    try:
        if connector is not None:
            latency_ms = float(connector(host, port, timeout))
        else:
            sock = socket.create_connection((host, port), timeout=timeout)
            with contextlib.suppress(Exception):
                sock.close()
            latency_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:  # noqa: BLE001 — a probe must never raise
        return ProbeSample(at=at, ok=False, latency_ms=None,
                           error=str(exc)[:300])
    return ProbeSample(at=at, ok=True, latency_ms=latency_ms, status=None)


def probe_feed(url: str, *, timeout: float = 10.0,
               fetcher: Any = None,
               now: float | None = None) -> tuple[ProbeSample, list[str]]:
    """One status-page feed probe.

    Returns (sample, signal_refs). The sample records reachability
    plus the feed's own verdict: a feed that reports a major outage
    is a failed sample (the service says it is down); a degraded
    feed is reachable, with the marker carried as a corroborating
    signal ref. A feed that cannot be read reports "unavailable" —
    never a verdict about the service.
    """
    at = float(now if now is not None else time.time())
    sample, body = _fetch_once(url, timeout, fetcher, at)
    if not sample.ok:
        return sample, []
    refs: list[str] = []
    text = ""
    with contextlib.suppress(Exception):
        text = body.decode("utf-8", "replace")
    status_value = ""
    with contextlib.suppress(Exception):
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            status_value = str(parsed.get("status", "")).strip().lower()
    if status_value:
        if status_value in FEED_HEALTHY:
            return sample, []
        if status_value in FEED_DEGRADED:
            refs.append(f"feed reports: {status_value}")
            return sample, refs
        refs.append(f"feed reports: {status_value}")
        return ProbeSample(at=at, ok=False,
                           latency_ms=sample.latency_ms,
                           status=sample.status,
                           error=f"feed reports: {status_value}"), refs
    lowered = text.lower()
    for marker in OUTAGE_MARKERS:
        if marker in lowered:
            refs.append(f"feed reports: {marker}")
    if any(_is_major_signal(ref) for ref in refs):
        return ProbeSample(at=at, ok=False,
                           latency_ms=sample.latency_ms,
                           status=sample.status,
                           error=f"feed reports: {refs[0]}"), refs
    return sample, refs


def probe_target(target: WatchTarget, *, fetcher: Any = None,
                 connector: Any = None,
                 now: float | None = None) -> tuple[ProbeSample, list[str]]:
    """Probe one target by its kind. Never raises."""
    at = float(now if now is not None else time.time())
    try:
        if target.kind == "http":
            return probe_http(target.url, fetcher=fetcher, now=at), []
        if target.kind == "tcp":
            return probe_tcp(target.url, connector=connector, now=at), []
        if target.kind == "feed":
            return probe_feed(target.url, fetcher=fetcher, now=at)
        return ProbeSample(at=at, ok=False,
                           error=f"unknown probe kind: {target.kind!r}"), []
    except Exception as exc:  # noqa: BLE001 — a probe must never raise
        return ProbeSample(at=at, ok=False,
                           error=str(exc)[:300]), []


# ── durable store ─────────────────────────────────────────────────────


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no store is an empty store
        return {}
    return raw if isinstance(raw, dict) else {}


class WatchtowerStore:
    """Durable watchlist + probe history + incidents.

    ``<home>/data/watchtower.json`` — targets, per-target probe
    samples (trailing ``BASELINE_WINDOW``), open/resolved incidents
    (last 50 resolved kept for the digest), last tick. Atomic writes;
    a corrupt store reads as empty, never a crash.
    """

    _NAME = WATCHTOWER_FILE
    _MAX_INCIDENTS = 50

    def __init__(self, home: Any) -> None:
        self._path = Path(home) / "data" / self._NAME
        self._data = _read_json(self._path)

    # -- persistence ---------------------------------------------------
    def _save(self) -> None:
        with contextlib.suppress(Exception):
            _atomic_json_write(self._path, self._data)

    def _targets(self) -> dict[str, Any]:
        raw = self._data.get("targets")
        return raw if isinstance(raw, dict) else {}

    def _samples(self) -> dict[str, Any]:
        raw = self._data.get("samples")
        return raw if isinstance(raw, dict) else {}

    def _incidents(self) -> list[dict[str, Any]]:
        raw = self._data.get("incidents")
        return list(raw) if isinstance(raw, list) else []

    # -- targets --------------------------------------------------------
    def add_target(self, target: WatchTarget) -> bool:
        """Add a target; False when the name is already watched."""
        targets = self._targets()
        if target.name in targets:
            return False
        targets[target.name] = target.to_dict()
        self._data["targets"] = targets
        self._save()
        return True

    def remove_target(self, name: str) -> bool:
        targets = self._targets()
        if name not in targets:
            return False
        del targets[name]
        self._data["targets"] = targets
        samples = self._samples()
        samples.pop(name, None)
        self._data["samples"] = samples
        self._save()
        return True

    def list_targets(self) -> list[WatchTarget]:
        return [WatchTarget.from_dict(raw)
                for raw in self._targets().values()]

    def get_target(self, name: str) -> WatchTarget | None:
        raw = self._targets().get(name)
        return WatchTarget.from_dict(raw) if raw else None

    # -- samples --------------------------------------------------------
    def record_sample(self, name: str, sample: Any) -> None:
        samples = self._samples()
        rows = list(samples.get(name) or [])
        rows.append(_normalize_sample(sample).to_dict())
        if len(rows) > BASELINE_WINDOW:
            del rows[: len(rows) - BASELINE_WINDOW]
        samples[name] = rows
        self._data["samples"] = samples
        self._save()

    def samples_for(self, name: str) -> list[ProbeSample]:
        rows = self._samples().get(name) or []
        out = []
        for raw in rows:
            with contextlib.suppress(Exception):
                out.append(ProbeSample.from_dict(raw))
        return out

    # -- incidents ------------------------------------------------------
    def active_incident(self, name: str) -> Incident | None:
        for raw in self._incidents():
            if raw.get("target") == name and not raw.get("resolved_at"):
                return Incident.from_dict(raw)
        return None

    def list_incidents(self, *, active_only: bool = False,
                       ) -> list[Incident]:
        out = [Incident.from_dict(raw) for raw in self._incidents()]
        if active_only:
            out = [i for i in out if i.resolved_at is None]
        return out

    def open_incident(self, name: str, state: str,
                      evidence: dict[str, Any],
                      *, now: float | None = None) -> Incident:
        """Open an incident, or update the active one in place.

        The ``since`` of the first opening is kept across
        possible→down escalation — the incident started when it
        started, not when it got worse.
        """
        now_f = float(now if now is not None else time.time())
        incidents = self._incidents()
        for index, raw in enumerate(incidents):
            if raw.get("target") == name and not raw.get("resolved_at"):
                incident = Incident.from_dict(raw)
                incident.state = state
                incident.evidence = evidence
                incidents[index] = incident.to_dict()
                self._data["incidents"] = incidents
                self._save()
                return incident
        incident = Incident(target=name, state=state, since=now_f,
                            evidence=evidence)
        incidents.append(incident.to_dict())
        self._data["incidents"] = incidents[-self._MAX_INCIDENTS:]
        self._save()
        return incident

    def resolve_incident(self, name: str, *,
                         now: float | None = None) -> Incident | None:
        now_f = float(now if now is not None else time.time())
        incidents = self._incidents()
        resolved = None
        for index, raw in enumerate(incidents):
            if raw.get("target") == name and not raw.get("resolved_at"):
                incident = Incident.from_dict(raw)
                incident.resolved_at = now_f
                incidents[index] = incident.to_dict()
                resolved = incident
        if resolved is not None:
            self._data["incidents"] = incidents
            self._save()
        return resolved

    def mark_alerted(self, name: str) -> None:
        incidents = self._incidents()
        for index, raw in enumerate(incidents):
            if raw.get("target") == name and not raw.get("resolved_at"):
                incident = Incident.from_dict(raw)
                incident.alerted = True
                incidents[index] = incident.to_dict()
        self._data["incidents"] = incidents
        self._save()

    # -- tick bookkeeping ----------------------------------------------
    def last_tick(self) -> float | None:
        value = self._data.get("last_tick")
        return None if value is None else float(value)

    def set_last_tick(self, now: float) -> None:
        self._data["last_tick"] = float(now)
        self._save()


# ── check + tick ──────────────────────────────────────────────────────


def _resolve_home(home: Any, context: Any) -> Path:
    if home is not None:
        return Path(home)
    settings = getattr(context, "settings", None) if context else None
    found = None
    if settings is not None:
        found = getattr(settings, "home_path", None) or getattr(
            settings, "home", None)
    if found is not None:
        return Path(found)
    return Path("~/.nomorals").expanduser()


def check_target(store: WatchtowerStore, target: WatchTarget, *,
                 fetcher: Any = None, connector: Any = None,
                 signals: list[str] | None = None,
                 now: float | None = None) -> dict[str, Any]:
    """Probe one target once, record the sample, judge the verdict,
    and maintain its incident. Never raises — a failed probe is a
    sample, and a failed check is a result with the reason."""
    now_f = float(now if now is not None else time.time())
    try:
        sample, feed_refs = probe_target(
            target, fetcher=fetcher, connector=connector, now=now_f)
        store.record_sample(target.name, sample)
        merged_signals = list(feed_refs) + [str(s) for s in (signals or [])
                                            if str(s).strip()]
        verdict = evaluate_verdict(target.name,
                                   store.samples_for(target.name),
                                   signals=merged_signals, now=now_f)
        incident = None
        if verdict.state in ("possible", "down"):
            incident = store.open_incident(
                target.name, verdict.state, verdict.evidence, now=now_f)
        elif verdict.state == "ok":
            store.resolve_incident(target.name, now=now_f)
        return {"ok": True, "target": target.name,
                "state": verdict.state, "reason": verdict.reason,
                "evidence": verdict.evidence,
                "incident": (incident.to_dict() if incident else None),
                "vantage": VANTAGE, "error": ""}
    except Exception as exc:  # noqa: BLE001 - a check must never raise
        return {"ok": False, "target": target.name, "state": "unknown",
                "reason": f"check failed: {exc}", "evidence": {},
                "incident": None, "vantage": VANTAGE,
                "error": f"check failed: {exc}"}


def run_watchtower_tick_once(*, context: Any = None,
                             notifier: Any = None, home: Any = None,
                             now: float | None = None,
                             fetcher: Any = None,
                             connector: Any = None) -> dict[str, Any]:
    """One watchtower tick: probe every watched target, page the
    owner once per escalating critical incident. Never raises.

    A quiet tick (no targets) makes no network call at all. Alert
    discipline: a target going ``down`` pages the owner only when
    it is marked critical (a target he trades through) and only once
    per incident — ``possible`` never pages, a non-critical ``down``
    surfaces in the briefing digest only, and a recovery digests
    silently, never pages.
    """
    now_f = float(now if now is not None else time.time())
    try:
        home_path = _resolve_home(home, context)
        store = WatchtowerStore(home_path)
        targets = store.list_targets()
    except Exception as exc:  # noqa: BLE001 - tick must never raise
        return {"ok": False, "checked": 0, "alerts": 0,
                "ran_at": now_f, "error": f"tick failed: {exc}"}
    if not targets:
        return {"ok": True, "checked": 0, "alerts": 0, "ran_at": now_f,
                "incidents": [], "error": ""}
    alerts = 0
    incidents: list[dict[str, Any]] = []
    for target in targets:
        result = check_target(store, target, fetcher=fetcher,
                              connector=connector, now=now_f)
        incident = result.get("incident")
        if incident is not None:
            incidents.append(incident)
        if (result.get("state") == "down" and target.critical
                and incident is not None and not incident.get("alerted")
                and notifier is not None):
            with contextlib.suppress(Exception):
                notifier.publish(
                    "watchtower",
                    title=f"{target.name} is down",
                    body=(f"{target.name} ({target.url}) is down: "
                          f"{result.get('reason', '')}. "
                          f"[{VANTAGE}]"),
                    critical=True)
                store.mark_alerted(target.name)
                alerts += 1
    with contextlib.suppress(Exception):
        store.set_last_tick(now_f)
    return {"ok": True, "checked": len(targets), "alerts": alerts,
            "ran_at": now_f, "incidents": incidents, "error": ""}


def ensure_watchtower_job(scheduler: Any, *,
                          spec: str = WATCHTOWER_SPEC) -> Any | None:
    """Register the 4-minute watchtower job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job
    that already matches is returned untouched. Best-effort — a
    scheduler that cannot store the job returns None rather than
    raising.
    """
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"),
                                          job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != WATCHTOWER_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            WATCHTOWER_JOB, spec, "tool",
            {"tool": WATCHTOWER_TOOL, "args": {}}, routine=True)
    except Exception:
        return None


# ── replies ───────────────────────────────────────────────────────────


_STATE_WORDS = {"ok": "no problems", "possible": "possible problems",
                "down": "problems", "unknown": "unknown — no verdict yet"}


def format_watchtower_reply(result: dict[str, Any]) -> str:
    """Plain-words reply for a watch/status result."""
    target = result.get("target", "")
    state = str(result.get("state", "unknown"))
    words = _STATE_WORDS.get(state, state)
    if not result.get("ok", False):
        return (f"Could not check {target or 'that service'}: "
                f"{result.get('error') or result.get('reason', 'unknown')}.")
    lines = [f"{target}: {words}."]
    reason = result.get("reason", "")
    if reason:
        lines.append(reason)
    evidence = result.get("evidence") or {}
    if evidence.get("probe_failures") is not None:
        lines.append(
            f"{evidence.get('probe_failures')} failures in the last "
            f"{evidence.get('recent_window', RECENT_WINDOW)} probes; "
            f"baseline deviation "
            f"{evidence.get('baseline_deviation', 0.0):+.3f}.")
    lines.append(f"[{result.get('vantage', VANTAGE)}]")
    return "\n".join(lines)
