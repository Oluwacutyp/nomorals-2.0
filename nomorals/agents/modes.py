"""Selectable agent modes: adaptive / autonomous / planning (combinable).

The owner asked for all three — not three labels on one path, but three
behaviours that can combine::

    /agent adaptive            learn from outcomes, adjust behaviour
    /agent autonomous          self-initiate work without being poked
    /agent planning            plan first, verify each step, replan on failure
    /agent adaptive+planning   any combination — modes are a SET, not a switch
    /agent +autonomous         add one mode · /agent -planning  remove one
    /agent off                 clear — baseline behaviour
    /agent status              what this chat is running

Natural language works too: "use planning mode", "go autonomous",
"adapt to me", "use autonomous and planning modes".

Each mode is built on systems that already exist — nothing parallel:

* **adaptive** — :class:`AdaptiveLearner` turns outcomes into behaviour:
  per-strategy success rates computed from REAL recorded results
  (Laplace-smoothed against the strategy's global rate), so rotation
  weights shift as evidence arrives; style hints come from the per-chat
  profile (``nomorals.partner.chat_profile`` shape) when one is supplied.
* **autonomous** — :class:`AutonomousEngine` is the self-initiation layer:
  registered sources propose initiatives (initiative lifecycle:
  proposed → running → verified → done/failed), rotation orders sources
  by learned weights (winners rise, failures sink), and proactive
  triggers are just sources — including a bridge over the existing
  ``ProactiveEngine`` suggestion generators.  This sits above the
  organ-level idle coordinator (``nomorals.autonomy``), which keeps
  doing organ maintenance; the engine is the *task* layer that sees
  what needs doing and does it.
* **planning** — :class:`PlanningRunner` is plan-first execution on top
  of the existing :class:`nomorals.agents.planner.AgentPlanner`:
  it reuses ``Plan`` / ``PlanStep`` / ``validate_plan`` /
  ``execution_batches`` (the dependency graph).  The plan is emitted
  via ``on_plan`` BEFORE any step runs; every step is verified; a
  failed verification replans with the failure as evidence.

Per-chat defaults live in :class:`ModeStore` (JSON, ``NM_AGENT_MODES_STORE``
or ``<settings-home>/data/agent-modes.json`` — resolved against the
settings data dir, never the working directory) — each chat remembers
its own mode set.

Approval flow: an initiative that needs approval is NOT executed on
tick. It is filed into the proposals queue (``proactive_log``,
kind ``"initiative"``) via the engine's ``proposal_sink`` and held as
``proposed``; ``/proposals`` lists it, ``/approve`` runs its action,
``/deny`` discards it.  When no approval is required the initiative
executes immediately, per the existing policy.  An initiative that
needs approval but has no sink fails closed — it is never executed
past a missing gate.

Scheduled tick: :func:`ensure_agent_modes_tick_job` registers one
durable ``agent modes tick`` job on the SAME scheduler the cognitive
loop / agency shift uses; the ``__agent_modes_tick__`` tool ticks
every chat with the autonomous mode.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.ids import new_id
from ..core.jobs import shared_registry
from ..core.logging_setup import get_logger
from .planner import AgentPlanner, Plan, PlanResult, PlanStep

_log = get_logger(__name__)

__all__ = [
    "MODES",
    "ModeStore",
    "AdaptiveLearner",
    "Initiative",
    "AutonomousEngine",
    "PlanningRunner",
    "AgentModes",
    "normalize_modes",
    "parse_agent_command",
    "parse_mode_phrase",
    "proactive_source",
    "ensure_agent_modes_tick_job",
    "AGENT_MODES_TICK_JOB",
    "AGENT_MODES_TICK_TOOL",
    "AGENT_MODES_TICK_SPEC",
]

#: The three modes.  A chat runs a SET of these (possibly empty = baseline).
MODES: tuple[str, ...] = ("adaptive", "autonomous", "planning")


# ── parsing ──────────────────────────────────────────────────────────────

def normalize_modes(raw: Any) -> frozenset[str]:
    """Coerce ``"adaptive+planning"`` / ``["adaptive"]`` / ``"adaptive"`` → set."""
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        parts = re.split(r"[\s,;+&]+", raw.strip().lower())
    elif isinstance(raw, (list, tuple, set, frozenset)):
        parts = [str(p).strip().lower() for p in raw]
    else:
        return frozenset()
    out = {p for p in parts if p in MODES}
    return frozenset(out)


#: Activation patterns per mode for natural-language switching.
#: A mode only matches on an *activation* phrase ("use planning mode",
#: "go autonomous", "adapt to me") — never on a passing mention
#: ("I like planning my day", "autonomous cars").
_MODE_PHRASES: dict[str, tuple[str, ...]] = {
    "adaptive": (
        r"\badapt\b", r"\badaptive mode\b", r"\bbe adaptive\b",
        r"\bswitch to adaptive\b", r"\bturn on adaptive\b",
        r"\badaptive\b.*\bmodes?\b", r"\buse adaptive\b",
    ),
    "autonomous": (
        r"\bgo autonomous\b", r"\bbe autonomous\b",
        r"\bautonomous mode\b", r"\bwork on your own\b",
        r"\bwork autonomously\b", r"\bself-?initiat\w*\b",
        r"\bautonomous\b.*\bmodes?\b", r"\buse autonomous\b",
        r"\bgo auto\b",
    ),
    "planning": (
        r"\bplanning mode\b", r"\bplan first\b",
        r"\bplan before (executing|acting|doing)\b",
        r"\buse planning\b", r"\bplanning\b.*\bmodes?\b",
        r"\bplan-?first\b",
    ),
}


def parse_mode_phrase(text: str) -> frozenset[str] | None:
    """Natural-language mode switch → mode set, or None (not a switch).

    >>> parse_mode_phrase("use planning mode")
    frozenset({'planning'})
    """
    t = (text or "").strip().lower().rstrip(".!? ")
    if not t:
        return None
    if re.search(r"\ball (three|3) modes?\b", t) or re.search(
            r"\ball modes\b", t):
        return frozenset(MODES)
    found = {mode for mode, pats in _MODE_PHRASES.items()
             if any(re.search(p, t) for p in pats)}
    return frozenset(found) if found else None


def parse_agent_command(arg: str) -> dict[str, Any]:
    """Parse ``/agent ...`` args → ``{"op": ..., "modes": ...}``.

    ops: ``set`` (replace the chat's set), ``add`` / ``remove`` (one mode),
    ``clear``, ``status``, ``tick`` (run one autonomous cycle), ``usage``.
    """
    text = (arg or "").strip().lower()
    if text in ("", "status", "show"):
        return {"op": "status", "modes": frozenset()}
    if text in ("off", "clear", "none", "baseline"):
        return {"op": "clear", "modes": frozenset()}
    if text == "tick":
        return {"op": "tick", "modes": frozenset()}
    if text.startswith("+"):
        modes = normalize_modes(text[1:])
        if modes:
            return {"op": "add", "modes": modes}
        return {"op": "usage", "modes": frozenset()}
    if text.startswith("-"):
        modes = normalize_modes(text[1:])
        if modes:
            return {"op": "remove", "modes": modes}
        return {"op": "usage", "modes": frozenset()}
    modes = normalize_modes(text)
    if modes and _covers(text, modes):
        return {"op": "set", "modes": modes}
    return {"op": "usage", "modes": frozenset()}


def _covers(text: str, modes: frozenset[str]) -> bool:
    """True when every token in ``text`` is a mode name / separator."""
    tokens = [t for t in re.split(r"[\s,;+&]+", text.strip()) if t]
    return bool(tokens) and all(t in MODES for t in tokens)


_USAGE = (
    "usage: /agent adaptive|autonomous|planning — set this chat's modes\n"
    "  /agent adaptive+planning — combine modes (a set, not a switch)\n"
    "  /agent +autonomous · /agent -planning — add / remove one mode\n"
    "  /agent status — what this chat runs · /agent off — baseline\n"
    "  /agent tick — run one autonomous cycle now"
)


# ── per-chat store ───────────────────────────────────────────────────────

def _settings_home(settings: Any = None) -> Path:
    """The settings data dir (``NM_HOME`` / ``home``), never the cwd.

    Same resolution the rest of the app uses for persistent state
    (``Settings.home_path``); ``~/.nomorals`` when settings cannot load.
    """
    try:
        if settings is not None:
            return Path(settings.home_path)
        from ..core.config import get_settings

        return Path(get_settings().home_path)
    except Exception:  # noqa: BLE001 - settings failure never breaks the store
        return Path.home() / ".nomorals"


def _default_store_path(settings: Any = None) -> Path:
    """Default store path, resolved against the settings data dir.

    ``NM_AGENT_MODES_STORE`` wins when set: absolute as-is, relative
    resolved against the settings home (like every other relative
    config path).  Otherwise ``<home>/data/agent-modes.json`` — the
    same file from every working directory, so chat and CLI share one
    store.
    """
    override = os.environ.get("NM_AGENT_MODES_STORE", "")
    if override:
        p = Path(os.path.expanduser(override))
        if p.is_absolute():
            return p
        return (_settings_home(settings) / p).resolve()
    try:
        if settings is not None:
            return settings.resolve("data/agent-modes.json")
        from ..core.config import get_settings

        return get_settings().resolve("data/agent-modes.json")
    except Exception:  # noqa: BLE001 - settings failure never breaks the store
        return Path.home() / ".nomorals" / "data" / "agent-modes.json"


class ModeStore:
    """Per-chat mode defaults + adaptive learning records, persisted as JSON.

    File shape: ``{"chats": {chat_key: [modes]}, "learning": {...}}``.
    Thread-safe, atomic writes (tmp + replace).  A corrupt file starts
    fresh rather than taking chat down.
    """

    def __init__(self, path: str | Path | None = None, *,
                 settings: Any = None) -> None:
        self.path = (Path(path) if path is not None
                     else _default_store_path(settings))
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"chats": {}, "learning": {}}
        self._load()

    def _load(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return
        try:
            data = json.loads(raw)
        except ValueError:
            _log.warning("agent-modes store is not valid JSON — starting fresh")
            return
        if isinstance(data, dict):
            if isinstance(data.get("chats"), dict):
                self._data["chats"] = data["chats"]
            if isinstance(data.get("learning"), dict):
                self._data["learning"] = data["learning"]

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:  # noqa: BLE001 — persistence is best-effort
            _log.warning("could not persist agent-modes store: %s", exc)

    # ── mode sets ──

    def get(self, chat_key: str) -> frozenset[str]:
        with self._lock:
            raw = self._data["chats"].get(chat_key or "", [])
        return normalize_modes(raw)

    def set(self, chat_key: str, modes: Any) -> frozenset[str]:
        wanted = normalize_modes(modes)
        with self._lock:
            self._data["chats"][chat_key or ""] = sorted(wanted)
            self._save()
        return wanted

    def add(self, chat_key: str, modes: Any) -> frozenset[str]:
        return self.set(chat_key, self.get(chat_key) | normalize_modes(modes))

    def remove(self, chat_key: str, modes: Any) -> frozenset[str]:
        return self.set(chat_key, self.get(chat_key) - normalize_modes(modes))

    def clear(self, chat_key: str) -> frozenset[str]:
        return self.set(chat_key, frozenset())

    def all_chats(self) -> dict[str, frozenset[str]]:
        with self._lock:
            return {k: normalize_modes(v)
                    for k, v in self._data["chats"].items()}

    # ── learning records ──

    def learning(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data["learning"]))

    def save_learning(self, learning: dict[str, Any]) -> None:
        with self._lock:
            self._data["learning"] = learning
            self._save()


# ── adaptive: outcome → knowledge → adjusted behaviour ───────────────────

class AdaptiveLearner:
    """Learns from outcomes.  Nothing here is a fixed default.

    ``record_outcome(context, strategy, ok)`` feeds real results in;
    ``success_rate`` / ``weights`` / ``choose`` read them back out.
    Rates are Laplace-smoothed against the strategy's *global* rate
    across contexts (prior strength 2), so a strategy that fails
    everywhere is avoided even in a brand-new context — and a fresh
    learner with no evidence has no preference at all (equal weights).
    """

    _PRIOR_STRENGTH = 2.0

    def __init__(self, store: ModeStore | None = None) -> None:
        self._store = store
        self._records: dict[str, dict[str, dict[str, int]]] = {}
        if store is not None:
            try:
                self._records = store.learning() or {}
            except Exception:  # noqa: BLE001 - a bad store never breaks learning
                self._records = {}

    def _persist(self) -> None:
        if self._store is not None:
            self._store.save_learning(self._records)

    def record_outcome(self, context_key: str, strategy: str, ok: bool,
                       *, weight: int = 1) -> None:
        """One real result: ``ok`` True = the strategy worked."""
        ctx = self._records.setdefault(context_key or "default", {})
        rec = ctx.setdefault(strategy, {"ok": 0, "fail": 0})
        rec["ok" if ok else "fail"] += max(1, int(weight))
        self._persist()

    def _counts(self, context_key: str, strategy: str) -> tuple[int, int]:
        rec = self._records.get(context_key or "default", {}).get(strategy, {})
        return int(rec.get("ok", 0)), int(rec.get("fail", 0))

    def _global_rate(self, strategy: str) -> float:
        ok = fail = 0
        for ctx in self._records.values():
            rec = ctx.get(strategy, {})
            ok += int(rec.get("ok", 0))
            fail += int(rec.get("fail", 0))
        if ok + fail == 0:
            return 0.5
        return ok / (ok + fail)

    def success_rate(self, strategy: str, context_key: str = "default") -> float:
        """Learned success rate for ``strategy`` in ``context_key``."""
        ok, fail = self._counts(context_key, strategy)
        prior = self._global_rate(strategy)
        prior_ok = self._PRIOR_STRENGTH * prior
        return (ok + prior_ok) / (ok + fail + self._PRIOR_STRENGTH)

    def weights(self, strategies: list[str],
                context_key: str = "default") -> dict[str, float]:
        """Rotation weights from real results — normalised, sum to 1."""
        rates = {s: self.success_rate(s, context_key) for s in strategies}
        total = sum(rates.values()) or 1.0
        return {s: r / total for s, r in rates.items()}

    def choose(self, strategies: list[str],
               context_key: str = "default") -> list[tuple[str, float]]:
        """Strategies ranked best-first by learned rate (stable on ties)."""
        ranked = [(s, self.success_rate(s, context_key)) for s in strategies]
        ranked.sort(key=lambda kv: -kv[1])
        return ranked

    def learned_summary(self) -> dict[str, Any]:
        total_ok = total_fail = 0
        per: dict[str, dict[str, int]] = {}
        for ctx in self._records.values():
            for strat, rec in ctx.items():
                ok, fail = int(rec.get("ok", 0)), int(rec.get("fail", 0))
                total_ok += ok
                total_fail += fail
                slot = per.setdefault(strat, {"ok": 0, "fail": 0})
                slot["ok"] += ok
                slot["fail"] += fail
        return {"outcomes": total_ok + total_fail, "ok": total_ok,
                "fail": total_fail, "strategies": per}

    @staticmethod
    def style_hints(profile: dict[str, Any]) -> dict[str, Any]:
        """Style hints derived from a per-chat profile (chat_profile shape).

        Only reads fields that are present — an empty profile yields no
        hints, never invented ones.
        """
        hints: dict[str, Any] = {}
        if not profile:
            return hints
        vibe = profile.get("vibe_label") or ""
        if vibe:
            hints["vibe"] = vibe
        activity = profile.get("activity_label") or ""
        if activity:
            hints["pace"] = activity
        topics = profile.get("topics") or []
        if topics:
            hints["topics"] = list(topics)[:5]
        purpose = profile.get("purpose") or ""
        if purpose:
            hints["purpose"] = purpose
        return hints


# ── autonomous: initiative lifecycle + rotation + proactive triggers ─────

@dataclass
class Initiative:
    """One self-initiated unit of work and where it is in its lifecycle."""

    id: str
    label: str
    source: str
    priority: float = 0.5
    # proposed → running → verified → done|failed.  "proposed" also
    # covers an initiative HELD for owner approval (its action has not
    # run yet); "denied" is the terminal state for a held initiative
    # the owner rejected.
    status: str = "proposed"
    created_at: float = field(default_factory=time.time)
    result: Any = None
    error: str = ""
    verified: bool = False
    #: ``proactive_log`` id when this initiative was filed for approval
    #: ("" when it was never held).
    proposal_id: str = ""
    #: Background-job handle when the action parked instead of running
    #: inline. Harvested (and verified) on a later tick.
    job_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "source": self.source,
                "priority": self.priority, "status": self.status,
                "result": str(self.result)[:400] if self.result is not None
                else None,
                "error": self.error, "verified": self.verified,
                "proposal_id": self.proposal_id, "job_id": self.job_id}


class AutonomousEngine:
    """Self-initiation: sees what needs doing, does it, verifies, follows up.

    Sources are callables ``fn(ctx) -> initiative-dict | list | None``.
    An initiative dict carries ``label`` + ``action`` (``action(ctx)`` does
    the work) and optionally ``verify(result)``.  Sources that return
    None propose nothing this tick.  ``tick(ctx)`` needs no prompt — it
    is the "shows up and works" entry point.

    Approval: when an initiative needs approval — the spec declares
    ``requires_approval=True``, or the engine-level ``requires_approval``
    callable says so — its action does NOT run on tick.  The initiative
    is filed into the proposals queue via ``proposal_sink`` and held as
    ``proposed``; ``approve_proposal`` runs the action,
    ``deny_proposal`` discards it.  When approval is not required the
    initiative executes immediately, per the existing policy.  When
    approval is required but no sink is configured, the initiative
    fails closed: it is never executed past a missing gate.

    Rotation: sources run in learned-weight order (see
    :class:`AdaptiveLearner`) — a source whose initiatives keep failing
    sinks below sources that deliver, so the rotation itself adapts.
    """

    def __init__(self, learner: AdaptiveLearner | None = None, *,
                 proposal_sink: Callable[[Initiative, dict[str, Any],
                                          dict[str, Any]], Any] | None = None,
                 requires_approval: Callable[[dict[str, Any],
                                              dict[str, Any]], bool] | None = None,
                 park_after_s: float = 5.0,
                 ) -> None:
        self.learner = learner or AdaptiveLearner()
        self.proposal_sink = proposal_sink
        self.requires_approval = requires_approval
        #: Grace period before a slow action parks as a background job.
        #: Fast actions complete inline (unchanged behavior); slower ones
        #: keep running with heartbeats while the tick moves on. A latency
        #: optimization, not a timeout — nothing is abandoned here.
        self.park_after_s = max(0.0, float(park_after_s))
        #: Background jobs for initiative actions — shared process-wide so
        #: parked initiatives stay visible for debugging.
        self._jobs = shared_registry()
        self._sources: list[dict[str, Any]] = []
        self._history: list[Initiative] = []
        self._ticks = 0
        #: proposal id → (initiative, spec, ctx, executor) for held work.
        #: In-memory only: the action itself cannot live in the DB row,
        #: so a restart orphans held rows — approving one then fails
        #: honestly instead of running something unrecorded.
        self._pending: dict[str, tuple[Initiative, dict[str, Any],
                                        dict[str, Any], Any]] = {}

    def add_source(self, name: str, fn: Callable[[dict[str, Any]], Any],
                   *, priority: float = 0.5) -> None:
        """Register an initiative source (a proactive trigger)."""
        self._sources = [s for s in self._sources if s["name"] != name]
        self._sources.append({"name": name, "fn": fn, "priority": float(priority)})

    def remove_source(self, name: str) -> None:
        self._sources = [s for s in self._sources if s["name"] != name]

    def source_names(self) -> list[str]:
        return [s["name"] for s in self._sources]

    def rotation_order(self, sources: list[str],
                       context_key: str = "autonomous") -> list[str]:
        """Sources ordered by learned weight (winners first)."""
        weights = self.learner.weights(list(sources), context_key)
        reg = {s["name"]: i for i, s in enumerate(self._sources)}
        return sorted(sources,
                      key=lambda s: (-weights.get(s, 0.0), reg.get(s, 1 << 30)))

    def pending(self) -> list[Initiative]:
        return [i for i in self._history
                if i.status in ("proposed", "running")]

    def pending_proposals(self) -> list[Initiative]:
        """Initiatives held for owner approval (action not yet run)."""
        return [init for init, _s, _c, _e in self._pending.values()
                if init.status == "proposed"]

    def approve_proposal(self, proposal_id: str) -> Initiative:
        """Owner approved a held initiative — run its action now.

        Raises :class:`KeyError` when nothing is pending under that id
        (unknown, already decided, or orphaned by a restart).
        """
        try:
            init, spec, ctx, executor = self._pending.pop(str(proposal_id))
        except KeyError:
            raise KeyError(
                f"no pending initiative for proposal {proposal_id!r}") from None
        if init.status != "proposed":
            raise KeyError(
                f"no pending initiative for proposal {proposal_id!r}")
        return self._execute(init, spec, ctx, executor)

    def deny_proposal(self, proposal_id: str) -> Initiative:
        """Owner denied a held initiative — discard it, never run it."""
        try:
            init, _spec, _ctx, _executor = self._pending.pop(
                str(proposal_id))
        except KeyError:
            raise KeyError(
                f"no pending initiative for proposal {proposal_id!r}") from None
        init.status = "denied"
        init.error = "denied by owner"
        self.learner.record_outcome("autonomous", init.source, False)
        return init

    def history(self, *, limit: int = 50) -> list[Initiative]:
        return list(self._history[-limit:])

    def tick(self, ctx: dict[str, Any] | None = None, *,
             executor: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None
             ) -> list[Initiative]:
        """One self-initiated cycle.  No prompt, no user message needed."""
        ctx = dict(ctx or {})
        self._ticks += 1
        # First: harvest parked initiatives whose background jobs finished
        # since the last tick — their results land here, exactly once.
        done: list[Initiative] = self._harvest()
        names = [s["name"] for s in self._sources]
        ordered = self.rotation_order(names, "autonomous")
        by_name = {s["name"]: s for s in self._sources}
        for name in ordered:
            source = by_name[name]
            try:
                raw = source["fn"](ctx)
            except Exception as exc:  # noqa: BLE001 - one bad source never kills the tick
                _log.warning("autonomous source %s raised: %s", name, exc)
                failed = Initiative(id=new_id("init"), label=name,
                                    source=name, priority=source["priority"],
                                    status="failed", error=str(exc)[:300])
                self._history.append(failed)
                done.append(failed)
                self.learner.record_outcome("autonomous", name, False)
                continue
            if raw is None:
                continue
            specs = raw if isinstance(raw, list) else [raw]
            for spec in specs:
                done.append(self._run_one(name, source["priority"], spec,
                                          ctx, executor))
        return done

    def _needs_approval(self, spec: dict[str, Any],
                        ctx: dict[str, Any]) -> bool:
        """True when this initiative must wait for owner approval.

        Precedence: an explicit ``requires_approval`` bool on the spec
        wins; otherwise the engine-level callable decides; default is
        no approval (immediate execution, as before).
        """
        declared = spec.get("requires_approval")
        if isinstance(declared, bool):
            return declared
        if self.requires_approval is not None:
            try:
                return bool(self.requires_approval(spec, ctx))
            except Exception:  # noqa: BLE001 - a broken policy fails closed
                _log.warning("requires_approval check failed — holding")
                return True
        return False

    def _run_one(self, source: str, priority: float, spec: dict[str, Any],
                 ctx: dict[str, Any], executor: Any) -> Initiative:
        init = Initiative(id=new_id("init"),
                          label=str(spec.get("label", source)),
                          source=source, priority=priority,
                          status="proposed")
        self._history.append(init)
        if self._needs_approval(spec, ctx):
            return self._hold_for_approval(init, spec, ctx, executor)
        return self._execute(init, spec, ctx, executor)

    def _hold_for_approval(self, init: Initiative, spec: dict[str, Any],
                           ctx: dict[str, Any], executor: Any) -> Initiative:
        """File the initiative into the proposals queue and hold it.

        The action has NOT run.  Fail closed: a missing sink or a sink
        that cannot file means the initiative fails, never executes.
        """
        if self.proposal_sink is None:
            init.status = "failed"
            init.error = ("initiative requires approval but no proposal "
                          "sink is configured")
            self.learner.record_outcome("autonomous", init.source, False)
            return init
        try:
            proposal_id = self.proposal_sink(init, spec, ctx)
        except Exception as exc:  # noqa: BLE001 - filing failure is data
            init.status = "failed"
            init.error = f"proposal sink failed: {exc}"[:300]
            self.learner.record_outcome("autonomous", init.source, False)
            return init
        if not proposal_id:
            init.status = "failed"
            init.error = "proposal sink did not file the initiative"
            self.learner.record_outcome("autonomous", init.source, False)
            return init
        init.proposal_id = str(proposal_id)
        init.status = "proposed"
        self._pending[init.proposal_id] = (init, spec, ctx, executor)
        return init

    def _run_action(self, spec: dict[str, Any], ctx: dict[str, Any],
                    executor: Any) -> Any:
        """Run one initiative's action synchronously. Exceptions propagate."""
        if executor is not None:
            return executor(spec, ctx)
        action = spec.get("action")
        if action is None:
            raise ValueError(
                f"initiative {spec.get('label', '?')!r} has no action")
        return action(ctx)

    def _execute(self, init: Initiative, spec: dict[str, Any],
                 ctx: dict[str, Any], executor: Any) -> Initiative:
        """Run one initiative's action now (immediate or approved path).

        Background-job execution (no blunt timeout): the action runs with
        heartbeats; a slow action parks — the initiative stays "running"
        with its job id and the tick moves on to other sources. Fast
        actions complete inline (unchanged behavior). Parked initiatives
        are harvested and verified by :meth:`_harvest` on a later tick.
        """
        init.status = "running"
        job = self._jobs.submit(
            lambda: self._run_action(spec, ctx, executor),
            tool=f"autonomous:{init.source}", args={"label": init.label},
            labels={"owner": "autonomous"})
        init.job_id = job.id
        finished = self._jobs.collect(job.id, timeout=self.park_after_s)
        if finished is not None and finished.done():
            return self._complete_initiative(init, spec, finished)
        return init  # parked — harvested on a later tick

    def _complete_initiative(self, init: Initiative, spec: dict[str, Any],
                             job) -> Initiative:
        """Verify a finished job and close the initiative honestly."""
        status = job.status
        if status == "failed":
            init.status = "failed"
            init.error = str(job.error)[:300]
            self.learner.record_outcome("autonomous", init.source, False)
            return init
        if status == "presumed-dead":
            # The dead-man's switch fired: honest UNKNOWN, never claimed
            # done. A late result is still recorded on the job itself.
            init.status = "failed"
            init.error = str(job.error)[:300]
            self.learner.record_outcome("autonomous", init.source, False)
            return init
        result = job.result
        init.result = result
        init.status = "verified"
        verify = spec.get("verify")
        try:
            if verify is not None:
                ok = bool(verify(result))
            elif isinstance(result, PlanResult):
                ok = bool(result.success)
            else:
                ok = True
        except Exception as exc:  # noqa: BLE001 - a broken verifier fails the initiative
            ok = False
            init.error = f"verify failed: {exc}"[:300]
        init.verified = ok
        init.status = "done" if ok else "failed"
        if not ok and not init.error:
            init.error = "verification failed"
        self.learner.record_outcome("autonomous", init.source, ok)
        return init

    def _harvest(self) -> list[Initiative]:
        """Collect parked initiatives whose jobs finished since last tick.

        Each job is harvested exactly once (collect_ready marks collected).
        Initiatives whose jobs are still running stay parked — visible via
        their job id, never silently dropped.
        """
        done: list[Initiative] = []
        by_job = {i.job_id: i for i in self._history
                  if i.job_id and i.status == "running"}
        for _pid, (init, _spec, _ctx, _exec) in self._pending.items():
            if init.job_id and init.status == "running":
                by_job[init.job_id] = init
        for job in self._jobs.collect_ready(labels={"owner": "autonomous"}):
            init = by_job.get(job.id)
            if init is None:
                continue
            spec = self._spec_for_harvest(init)
            done.append(self._complete_initiative(init, spec, job))
        return done

    def _spec_for_harvest(self, init: Initiative) -> dict[str, Any]:
        """Best-effort spec recovery for a harvested initiative.

        The verifier lives on the spec; for history initiatives the spec
        is not retained, so verification falls back to the PlanResult /
        truthy convention inside ``_complete_initiative``.
        """
        for _pid, (p_init, spec, _ctx, _exec) in self._pending.items():
            if p_init is init:
                return spec
        return {}


def proactive_source(proactive: Any, user_id: str, *,
                     min_confidence: float = 0.5,
                     limit: int = 5) -> Callable[[dict[str, Any]], Any]:
    """Bridge the existing ``ProactiveEngine`` into an autonomous source.

    Each suggestion becomes an initiative whose action returns the
    suggestion payload (delivery stays with the caller's proposal flow).
    Suggestions ALWAYS need approval: the spec carries
    ``requires_approval=True``, so under an approval policy the
    initiative is filed into the proposals queue instead of executing
    — a suggestion must never run past the owner unapproved.
    The engine is async, so the bridge runs it to completion — callers
    that already run an event loop should wrap it themselves.
    """
    def _source(ctx: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            coro = proactive.get_suggestions(
                user_id, limit=limit, min_confidence=min_confidence,
                context=dict(ctx))
            suggestions = asyncio.run(coro)
        except RuntimeError:
            # already inside a running loop — cannot block; skip this tick
            return []
        except Exception as exc:  # noqa: BLE001 - suggestions never break a tick
            _log.warning("proactive source failed: %s", exc)
            return []
        out = []
        for s in suggestions or []:
            payload = {"title": getattr(s, "title", str(s)),
                       "confidence": getattr(s, "confidence", 0.0)}
            out.append({"label": f"proactive: {payload['title']}",
                        "proposal": payload["title"],
                        "requires_approval": True,
                        "action": (lambda p: (lambda c: p))(payload)})
        return out

    return _source


# ── planning: plan first, verify each step, replan on failure ────────────

async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class PlanningRunner:
    """Plan-first execution over :class:`AgentPlanner`'s dependency graph.

    ``decompose(goal, evidence)`` produces step specs (``name``,
    ``description``, ``action``, ``parameters``, ``depends_on``);
    ``execute_step(step, ctx)`` does one step; ``verify_step(step, result)``
    checks it.  The plan is handed to ``on_plan`` BEFORE anything runs;
    a failed verification replans with the failure as evidence
    (``max_replans`` caps it).  An invalid plan (unknown dep, cycle,
    unknown action) raises :class:`ValueError` and NOTHING executes.

    Heartbeat + background jobs: every step (execute AND verify) runs
    as a background job with heartbeats — there is NO per-step timeout.
    A slow step keeps running while the run waits cancel-aware; only
    the dead-man's switch (heartbeat silence) stops a step, marking it
    ``presumed-dead`` with an honest UNKNOWN outcome — never claimed
    done — and the run stops with ``cancelled=True`` (presumed-dead is
    terminal, NOT a replan trigger, because re-running blindly could
    duplicate side effects).  ``cancel`` (a ``threading.Event`` or a
    callable returning bool) is honoured before the run and between
    steps: the run stops cleanly, finished steps keep their results,
    later steps never run.
    """

    def __init__(self, planner: AgentPlanner | None = None) -> None:
        self.planner = planner or AgentPlanner()
        #: Background jobs for plan steps — shared process-wide so parked
        #: steps stay visible for debugging.
        self._jobs = shared_registry()

    @staticmethod
    def _cancel_requested(cancel: Any) -> bool:
        """True when the cancel signal (Event or callable) has fired."""
        if cancel is None:
            return False
        try:
            if hasattr(cancel, "is_set"):
                return bool(cancel.is_set())
            if callable(cancel):
                return bool(cancel())
        except Exception:  # noqa: BLE001 - a broken cancel signal reads as "stop"
            return True
        return False

    @staticmethod
    @staticmethod
    def _run_step_sync(fn: Callable[..., Any], step: PlanStep,
                       ctx: dict[str, Any]) -> Any:
        """Run a step function to completion (job worker thread).

        Sync or async: an async function gets a fresh event loop in the
        worker thread; a sync function runs inline.  Exceptions
        propagate — the job records them as ``failed``.
        """
        value = fn(step, ctx)
        if inspect.isawaitable(value):
            async def _await_it() -> Any:
                return await value
            return asyncio.run(_await_it())
        return value

    def _submit_step_job(self, fn: Callable[..., Any], step: PlanStep,
                         ctx: dict[str, Any], phase: str):
        """Run a step phase (execute/verify) as a background job."""
        return self._jobs.submit(
            lambda: self._run_step_sync(fn, step, ctx),
            tool=f"plan:{step.name}:{phase}",
            args={"step_id": step.step_id},
            labels={"owner": "planning"})

    async def _collect_cancel_aware(self, job_id: str,
                                    cancel: Any):
        """Wait for a step job, honouring cancellation; never times out.

        Returns the job when done, or None when the run was cancelled.
        A finished job wins over a pending cancel (in-flight work is
        kept, mirroring the old between-steps cancel check). A cancelled
        wait leaves the job parked — it keeps running with heartbeats
        and stays visible in the registry.
        """
        while True:
            job = self._jobs.poll(job_id)
            if job is not None and job.done():
                return job
            if self._cancel_requested(cancel):
                return None
            await asyncio.sleep(0.1)

    @staticmethod
    def _job_failure(job) -> str:
        """Honest failure text for a finished step job ("" when done)."""
        if job.status == "done":
            return ""
        if job.status == "failed":
            return str(job.error)[:300] or "step failed"
        # presumed-dead: the dead-man's switch fired — honest UNKNOWN.
        return str(job.error)[:300] or "outcome unknown"

    async def run_step_job(self, fn: Callable[..., Any], step: PlanStep,
                           ctx: dict[str, Any], *, phase: str,
                           cancel: Any = None):
        """Run one step phase as a background job with heartbeats.

        No timeout — the wait is cancel-aware only. Returns the finished
        job (inspect ``status``/``result``/``error``), or None when the
        run was cancelled while waiting; the job then stays parked in
        the registry, still running and visible.
        """
        job = self._submit_step_job(fn, step, ctx, phase)
        return await self._collect_cancel_aware(job.id, cancel)

    async def plan(self, goal: str, decompose: Callable[..., Any], *,
                   evidence: dict[str, Any] | None = None) -> Plan:
        specs = await _maybe_await(decompose(goal, evidence))
        steps: list[PlanStep] = []
        for i, spec in enumerate(specs or []):
            if not isinstance(spec, dict):
                raise ValueError(f"decompose returned a non-dict step: {spec!r}")
            steps.append(PlanStep(
                step_id=str(spec.get("step_id", f"step_{i + 1}")),
                name=str(spec.get("name", f"step {i + 1}")),
                description=str(spec.get("description", "")),
                action=str(spec.get("action", "wait")),
                parameters=dict(spec.get("parameters") or {}),
                depends_on=list(spec.get("depends_on") or []),
            ))
        plan = Plan(plan_id=new_id("plan"), goal=goal, steps=steps)
        problems = self.planner.validate_plan(plan)
        if problems:
            raise ValueError("invalid plan: " + "; ".join(problems))
        return plan

    async def run(self, goal: str, *, decompose: Callable[..., Any],
                  execute_step: Callable[..., Any],
                  verify_step: Callable[..., Any] | None = None,
                  on_plan: Callable[[Plan], Any] | None = None,
                  max_replans: int = 2, account: str = "",
                  cancel: Any = None) -> PlanResult:
        started = time.time()
        if self._cancel_requested(cancel):
            return PlanResult(
                plan_id=new_id("plan"), success=False, cancelled=True,
                steps_completed=0, steps_total=0, results={},
                errors=["run cancelled before start"],
                duration_seconds=time.time() - started,
                summary="cancelled before start",
            )
        plan = await self.plan(goal, decompose)
        if on_plan is not None:
            await _maybe_await(on_plan(plan))
        results: dict[str, Any] = {}
        errors: list[str] = []
        completed = 0
        replans = 0
        halted = ""  # "presumed-dead" | "cancelled" — terminal, never a replan
        ctx: dict[str, Any] = {"goal": goal, "account": account,
                               "results": results}
        while True:
            replanned = False
            for batch in self.planner.execution_batches(plan):
                for step in batch:
                    if step.step_id in results:
                        continue  # already done under a previous plan version
                    if self._cancel_requested(cancel):
                        errors.append("run cancelled")
                        halted = "cancelled"
                        break
                    if not all(d in results for d in step.depends_on):
                        step.status = "skipped"
                        step.error = "dependency failed"
                        errors.append(f"{step.name}: dependency failed")
                        continue
                    step.status = "running"
                    step.started_at = time.time()
                    # Background job + heartbeats: no per-step timeout.
                    # The wait is cancel-aware only; a silent job ends
                    # via the dead-man's switch (honest UNKNOWN).
                    finished = await self.run_step_job(
                        execute_step, step, ctx, phase="execute",
                        cancel=cancel)
                    if finished is None:
                        errors.append("run cancelled")
                        halted = "cancelled"
                        break
                    failure = self._job_failure(finished)
                    if finished.status == "presumed-dead":
                        step.status = "presumed-dead"
                        step.error = failure
                        step.completed_at = time.time()
                        errors.append(f"{step.name}: {step.error}")
                        halted = "presumed-dead"
                        break
                    result = finished.result if not failure else None
                    if not failure and verify_step is not None:
                        vfinished = await self.run_step_job(
                            verify_step, step, ctx, phase="verify",
                            cancel=cancel)
                        if vfinished is None:
                            errors.append("run cancelled")
                            halted = "cancelled"
                            break
                        if vfinished.status == "presumed-dead":
                            step.status = "presumed-dead"
                            step.error = (
                                f"verify {self._job_failure(vfinished)}")
                            step.completed_at = time.time()
                            errors.append(f"{step.name}: {step.error}")
                            halted = "presumed-dead"
                            break
                        if vfinished.status == "failed":
                            ok, failure = False, (
                                f"verify raised: {vfinished.error}"[:300])
                        elif not vfinished.result:
                            failure = "verification failed"
                    if failure:
                        if replans < max_replans:
                            replans += 1
                            evidence = {"failed_step": step.name,
                                        "failed_step_id": step.step_id,
                                        "error": failure,
                                        "completed": sorted(results)}
                            plan = await self.plan(goal, decompose,
                                                   evidence=evidence)
                            if on_plan is not None:
                                await _maybe_await(on_plan(plan))
                            replanned = True
                            break
                        step.status = "failed"
                        step.error = failure
                        step.completed_at = time.time()
                        errors.append(f"{step.name}: {failure}")
                        continue
                    step.result = result
                    step.status = "completed"
                    step.completed_at = time.time()
                    results[step.step_id] = result
                    completed += 1
                if replanned or halted:
                    break
            if not replanned:
                break
        if halted:
            summary = (f"cancelled ({halted}): {completed}/"
                       f"{len(plan.steps)} steps, {replans} replan(s); "
                       f"errors: {errors[0]}" if errors else
                       f"cancelled ({halted})")
        else:
            summary = (f"{completed}/{len(plan.steps)} steps, "
                       f"{replans} replan(s)" + (f"; errors: {errors[0]}"
                                                 if errors else ""))
        return PlanResult(
            plan_id=plan.plan_id,
            success=not errors and not halted,
            cancelled=bool(halted),
            steps_completed=completed,
            steps_total=len(plan.steps),
            results=results,
            errors=errors,
            duration_seconds=time.time() - started,
            summary=summary,
        )

    def run_sync(self, goal: str, **kwargs: Any) -> PlanResult:
        """Synchronous wrapper for CLI / chat handlers (no running loop)."""
        return asyncio.run(self.run(goal, **kwargs))


# ── facade: one chat's mode set + the behaviours it switches on ──────────

class AgentModes:
    """Owns a chat's mode set and routes work through the modes it holds.

    * planning on → :meth:`handle_task` / autonomous initiatives go
      through :class:`PlanningRunner` (plan → verify → replan).
    * adaptive on → the strategy for a task is picked by the learner
      from recorded outcomes, and the outcome is recorded back.
    * autonomous on → :meth:`tick` runs the self-initiation engine.
    * no modes → baseline: direct single-step execution, no plan event,
      no learning, no self-initiation.
    """

    def __init__(self, store: ModeStore | None = None,
                 learner: AdaptiveLearner | None = None,
                 autonomous: AutonomousEngine | None = None,
                 planner_runner: PlanningRunner | None = None, *,
                 proposal_sink: Callable[[Initiative, dict[str, Any],
                                          dict[str, Any]], Any] | None = None,
                 requires_approval: Callable[[dict[str, Any],
                                              dict[str, Any]], bool] | None = None,
                 settings: Any = None) -> None:
        self.store = store or ModeStore(settings=settings)
        self.learner = learner or AdaptiveLearner(store=self.store)
        # proposal_sink / requires_approval configure the engine we
        # build; an explicitly passed engine already owns its own.
        self.autonomous = autonomous or AutonomousEngine(
            learner=self.learner, proposal_sink=proposal_sink,
            requires_approval=requires_approval)
        self.planner = planner_runner or PlanningRunner()

    # ── selection ──

    def modes_for(self, chat_key: str) -> frozenset[str]:
        return self.store.get(chat_key or "")

    def set_modes(self, chat_key: str, modes: Any) -> frozenset[str]:
        return self.store.set(chat_key or "", modes)

    def add_modes(self, chat_key: str, modes: Any) -> frozenset[str]:
        return self.store.add(chat_key or "", modes)

    def remove_modes(self, chat_key: str, modes: Any) -> frozenset[str]:
        return self.store.remove(chat_key or "", modes)

    def clear_modes(self, chat_key: str) -> frozenset[str]:
        return self.store.clear(chat_key or "")

    def describe(self, chat_key: str) -> str:
        modes = self.modes_for(chat_key)
        if modes:
            lines = [f"agent modes: {', '.join(sorted(modes))}"]
        else:
            lines = ["agent modes: none (baseline behaviour)"]
        if "adaptive" in modes:
            summary = self.learner.learned_summary()
            lines.append(f"adaptive: {summary['outcomes']} recorded outcome(s) "
                         f"({summary['ok']} ok / {summary['fail']} fail)")
        if "autonomous" in modes:
            done = [i for i in self.autonomous.history()
                    if i.status == "done"]
            failed = [i for i in self.autonomous.history()
                      if i.status == "failed"]
            pending = self.autonomous.pending_proposals()
            lines.append(f"autonomous: {len(done)} initiative(s) done, "
                         f"{len(failed)} failed, "
                         f"{len(pending)} awaiting approval, "
                         f"{len(self.autonomous.source_names())} source(s)")
        if "planning" in modes:
            lines.append("planning: plan first, verify each step, "
                         "replan on failure")
        return "\n".join(lines)

    def apply_command(self, arg: str, chat_key: str) -> str:
        """Handle ``/agent ...`` args for one chat.  Returns the reply."""
        parsed = parse_agent_command(arg)
        op, modes = parsed["op"], parsed["modes"]
        if op == "set":
            self.set_modes(chat_key, modes)
            return f"agent modes: {', '.join(sorted(modes))}"
        if op == "add":
            now = self.add_modes(chat_key, modes)
            return f"agent modes: {', '.join(sorted(now))}"
        if op == "remove":
            now = self.remove_modes(chat_key, modes)
            return (f"agent modes: {', '.join(sorted(now)) or 'none'}")
        if op == "clear":
            self.clear_modes(chat_key)
            return "agent modes cleared — baseline behaviour."
        if op == "tick":
            done = self.tick(chat_key, {})
            if not done:
                return "autonomous tick: nothing to do."
            lines = [f"autonomous tick: {len(done)} initiative(s)"]
            for init in done:
                mark = ("✓" if init.status == "done"
                        else "…" if init.status == "proposed"
                        else "✗")
                extra = (f" — awaiting approval: /proposals"
                         if init.status == "proposed" else "")
                lines.append(f"  {mark} {init.label} [{init.status}]{extra}")
            return "\n".join(lines)
        if op == "status":
            return self.describe(chat_key)
        return _USAGE

    # ── work ──

    async def handle_task(self, goal: str, chat_key: str, *,
                          decompose: Callable[..., Any] | None = None,
                          execute_step: Callable[..., Any],
                          verify_step: Callable[..., Any] | None = None,
                          on_plan: Callable[[Plan], Any] | None = None,
                          strategies: list[str] | None = None,
                          strategy_context: str = "default",
                          cancel: Any = None) -> PlanResult:
        """Run one task through this chat's mode set."""
        modes = self.modes_for(chat_key)
        strategy: str | None = None
        if "adaptive" in modes and strategies:
            strategy = self.learner.choose(list(strategies),
                                           strategy_context)[0][0]
        ctx = {"chat_key": chat_key, "modes": modes, "strategy": strategy,
               "goal": goal}
        if "planning" in modes:
            if decompose is None:
                raise ValueError(
                    "planning mode needs a decompose(goal, evidence) function")
            result = await self.planner.run(
                goal, decompose=decompose, execute_step=execute_step,
                verify_step=verify_step, on_plan=on_plan, cancel=cancel)
            if "adaptive" in modes and strategy is not None:
                self.learner.record_outcome(strategy_context, strategy,
                                            result.success)
            return result
        # baseline / adaptive-only: direct single-step execution — as a
        # background job with heartbeats (no blunt timeout; the wait is
        # cancel-aware only).
        step = PlanStep(step_id="step_1", name=goal, description=goal,
                        action="run")
        finished = await self.planner.run_step_job(
            execute_step, step, ctx, phase="direct", cancel=cancel)
        if finished is None:
            result_val, ok, failure = None, False, "run cancelled"
        else:
            failure = self.planner._job_failure(finished)
            result_val = finished.result if not failure else None
            ok = not failure
            if ok and verify_step is not None:
                # The direct path honors the documented contract
                # verify_step(step, result) — wrap so the job target
                # receives the result, not the ctx.
                def _verify_with_result(s, c, _v=verify_step,
                                        _r=result_val):
                    return _v(s, _r)

                vfinished = await self.planner.run_step_job(
                    _verify_with_result, step, ctx, phase="verify",
                    cancel=cancel)
                if vfinished is None:
                    result_val, ok, failure = None, False, "run cancelled"
                else:
                    vfailure = self.planner._job_failure(vfinished)
                    if vfailure:
                        ok, failure = False, vfailure
                    elif not vfinished.result:
                        ok, failure = False, "verification failed"
        if "adaptive" in modes and strategy is not None:
            self.learner.record_outcome(strategy_context, strategy, ok)
        return PlanResult(
            plan_id=new_id("plan"), success=ok,
            steps_completed=1 if ok else 0, steps_total=1,
            results={"step_1": result_val} if ok else {},
            errors=[] if ok else [f"{goal}: {failure}"],
            summary=f"direct: {'ok' if ok else 'failed'}",
        )

    def tick(self, chat_key: str, ctx: dict[str, Any] | None = None
             ) -> list[Initiative]:
        """One autonomous cycle for this chat — no prompt needed."""
        modes = self.modes_for(chat_key)
        if "autonomous" not in modes:
            return []
        base = dict(ctx or {})
        base.setdefault("chat_key", chat_key)
        if "planning" in modes:
            def _executor(spec: dict[str, Any], c: dict[str, Any]) -> Any:
                decompose = spec.get("decompose")
                if decompose is None:
                    label = str(spec.get("label", "initiative"))
                    decompose = (lambda g, e=None, _l=label: [
                        {"name": _l, "description": _l, "action": "run",
                         "parameters": {}, "depends_on": []}])
                execute_step = spec.get("execute_step")
                if execute_step is None:
                    action = spec.get("action")
                    if action is None:
                        raise ValueError("initiative has no action")
                    execute_step = (lambda s, cc, _a=action: _a(cc))
                return self.planner.run_sync(
                    str(spec.get("goal", spec.get("label", "initiative"))),
                    decompose=decompose, execute_step=execute_step,
                    verify_step=spec.get("verify_step"),
                    on_plan=spec.get("on_plan"))
            return self.autonomous.tick(base, executor=_executor)
        return self.autonomous.tick(base)

    # ── approval queue ──

    def pending_proposals(self) -> list[Initiative]:
        """Initiatives held for owner approval (action not yet run)."""
        return self.autonomous.pending_proposals()

    def approve_proposal(self, proposal_id: str) -> Initiative:
        """Owner approved a held initiative — run its action now."""
        return self.autonomous.approve_proposal(proposal_id)

    def deny_proposal(self, proposal_id: str) -> Initiative:
        """Owner denied a held initiative — discard it, never run it."""
        return self.autonomous.deny_proposal(proposal_id)

    def tick_all(self, ctx: dict[str, Any] | None = None
                 ) -> dict[str, list[Initiative]]:
        """One autonomous cycle for EVERY chat running autonomous mode.

        The scheduled-tick entry point: the ``__agent_modes_tick__``
        tool calls this, so chats that enabled autonomous mode keep
        getting cycles without anyone poking them.  Chats without the
        autonomous mode are skipped, never ticked.
        """
        out: dict[str, list[Initiative]] = {}
        for chat_key, modes in self.store.all_chats().items():
            if "autonomous" not in modes:
                continue
            out[chat_key] = self.tick(chat_key, ctx)
        return out


# ── scheduled autonomous tick ──────────────────────────────────────────────

#: Durable tick job: name, tool payload, default cadence.  Same scheduler
#: the cognitive loop / agency shift uses — never a parallel scheduler.
AGENT_MODES_TICK_JOB = "agent modes tick"
AGENT_MODES_TICK_TOOL = "__agent_modes_tick__"
AGENT_MODES_TICK_SPEC = "every 30m"

_INTERVAL_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}


def _interval_seconds(spec: str) -> float | None:
    """``"every 30m"`` / ``"every 1800s"`` / ``"1800"`` → seconds, else None."""
    s = str(spec or "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"every\s+(\d+(?:\.\d+)?)\s*([a-z]+)", s,
                     re.IGNORECASE)
    if not m:
        return None
    unit = m.group(2).lower()
    if unit not in _INTERVAL_UNITS:
        return None
    return float(m.group(1)) * _INTERVAL_UNITS[unit]


def _spec_matches(stored: Any, spec: str) -> bool:
    """True when a stored job spec is the requested one.

    ``every``-kind jobs store the interval as plain seconds or as
    ``"every 1800s"`` (via the scheduler's job formatter), so those
    compare numerically; every other kind compares verbatim.
    """
    if str(stored or "") == spec:
        return True
    want = _interval_seconds(spec)
    have = _interval_seconds(stored)
    return want is not None and have is not None and want == have


def ensure_agent_modes_tick_job(scheduler: Any, *,
                                spec: str = AGENT_MODES_TICK_SPEC) -> Any:
    """Register the durable agent-modes tick job. Idempotent by name.

    Mirrors :func:`nomorals.agency.wiring.ensure_agency_shift_job`: a
    job with the name already at the requested cadence is returned
    as-is; a stale one is replaced.  Best-effort: never raises (a
    missing scheduler just means no scheduled ticks — ``/agent tick``
    still works).
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = job.get("name", "") if isinstance(job, dict) \
                else getattr(job, "name", "")
            if name == AGENT_MODES_TICK_JOB:
                job_spec = job.get("spec", "") if isinstance(job, dict) \
                    else getattr(job, "spec", "")
                if _spec_matches(job_spec, spec):
                    return job
                job_id = job.get("id", "") if isinstance(job, dict) \
                    else getattr(job, "id", "")
                try:
                    scheduler.remove(job_id)
                except Exception:  # noqa: BLE001 - stale job removal
                    pass
                break
        return scheduler.add(AGENT_MODES_TICK_JOB, spec, "tool",
                             {"tool": AGENT_MODES_TICK_TOOL, "args": {}})
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
