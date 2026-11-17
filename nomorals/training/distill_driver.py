"""Daily driver for the periodic distillation loop (scout 17, A6).

``training/distillation.py`` was complete and test-pinned but had zero
production callers: no kept cloud turn ever became a training pair, no
scheduled distill ever fired, and no distilled adapter ever reached the
promotion gate. This module is the wiring that makes it run — it is not
a second training system:

* :func:`harvest_kept_turns` — the capture step. It reads the durable
  conversation log (the ``messages`` rows the runtime persists for every
  turn), pairs each user turn with the next assistant reply, and keeps
  only the turns the owner *kept*: a cloud teacher's answer the owner
  continued past (a later user message in the same conversation), PII
  scrubbed, command turns and too-short turns dropped as noise. A turn
  with no continuation yet is not rejected — it waits in the pending
  list and is re-checked on the next tick, so a turn is only ever
  harvested once its kept status is proven. The owner's own models
  (codebeast) never harvest: distilling her own output back into
  herself is circular. Pairs land in the durable
  :class:`~nomorals.training.distillation.DistillationStore`, which
  dedupes by pair hash.
* :func:`run_distill_once` — one daily tick: harvest -> train via
  :func:`~nomorals.training.distillation.periodic_distill` when enough
  pairs exist (``min_pairs=32``) -> the TrainingRegistry gate
  (``evaluate`` then ``promote``). A failed gate means the distilled
  candidate never lands; a failed train leaves the previous checkpoint
  file untouched. A promoted adapter notifies the owner once.
* ``ensure_distill_job`` + the ``__distill_tick__`` routine tool — a
  durable daily scheduler job, the continual-update / swarm-board /
  workflow pattern.

The first tick only looks back 30 days: the loop distills recent kept
turns, never the whole archive. The train itself is interval-gated
(every 24h unless forced) so a manual run and the scheduled tick never
double-train; the harvest runs on every tick regardless, so pairs
accrue daily.

Layering: L3 (``training``), same package as ``distillation.py`` /
``registry.py`` / ``collect.py``. It never imports the notifier — the
tick takes an injected one (the digest lane's lesson).
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from ..core.failures import record_failure
from ..core.logging_setup import get_logger
from .collect import scrub_pii
from .distillation import (
    DistillationStore,
    capture_owner_kept,
    periodic_distill,
)
from .preprocess import clean_text
from .registry import TrainingRegistry

_log = get_logger(__name__)

__all__ = [
    "DISTILL_JOB",
    "DISTILL_SPEC",
    "DISTILL_TOOL",
    "DistillState",
    "default_distill_out_dir",
    "default_distill_path",
    "default_distill_state_path",
    "ensure_distill_job",
    "harvest_kept_turns",
    "run_distill_once",
    "store_for_context",
]

DISTILL_JOB = "distill tick"
DISTILL_SPEC = "every 24h"
DISTILL_TOOL = "__distill_tick__"

#: The distillation gate: below this many pairs a tick is a no-op
#: ``deferred`` — a handful of pairs is noise, not a training set.
#: Mirrors ``periodic_distill``'s own default; the tick reports it.
MIN_PAIRS = 32
#: Minimum time between trains (the scheduled tick is daily anyway;
#: this stops a manual run + the tick double-training). ``force``
#: bypasses it.
DISTILL_INTERVAL_S = 24 * 3600
#: First tick only looks back this far — the loop distills recent kept
#: turns, never the whole archive.
FIRST_RUN_LOOKBACK_S = 30 * 24 * 3600
#: A pending turn the owner never comes back to stops being pending
#: after this long (it was never kept).
PENDING_TTL_S = 30 * 24 * 3600
#: Pending turns kept in the state (bounded — the state is a small JSON
#: file, not an archive).
MAX_PENDING = 500
#: Run history kept in the state (newest last).
MAX_RUNS = 50
#: The miner's noise floor (``training/collect.py``): shorter turns are
#: commands, acks, or fragments, not training data.
MIN_USER_CHARS = 8
MIN_ANSWER_CHARS = 24
#: Rows the brain writes that are not real model answers.
SYNTHETIC_MODELS = {"stt-failure", "tool-loop"}


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + os.replace)."""
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def default_distill_path(home: Path | str | None = None) -> Path:
    """The durable distillation pair store under a runtime's home."""
    base = Path(home) if home is not None else Path("~/.nomorals").expanduser()
    return base / "data" / "distillation.jsonl"


def default_distill_state_path(home: Path | str | None = None) -> Path:
    """The durable distill driver state under a runtime's home."""
    base = Path(home) if home is not None else Path("~/.nomorals").expanduser()
    return base / "data" / "distillation.json"


def default_distill_out_dir(home: Path | str | None = None) -> Path:
    """Where distilled checkpoints land under a home."""
    base = Path(home) if home is not None else Path("~/.nomorals").expanduser()
    return base / "training" / "distillation"


def _home_of(context: Any) -> Path | None:
    home = getattr(getattr(context, "settings", None), "home", None)
    if home is None:
        home = getattr(context, "home", None)
    return Path(home) if home is not None else None


def store_for_context(context: Any) -> DistillationStore:
    """The durable distillation store under a runtime's home (in-memory
    when the runtime has no home). Never raises."""
    try:
        home = _home_of(context)
        if home is None:
            return DistillationStore()
        return DistillationStore(path=default_distill_path(home))
    except Exception:  # noqa: BLE001 - durability is a bonus
        return DistillationStore()


class DistillState:
    """Durable distill driver state: harvest watermark, pending turns,
    run history.

    Shape follows WorkflowStore / ContinualStore: tmp+replace writes; a
    corrupt state is quarantined and degrades to empty with a named
    reason; bad rows are skipped; reads never raise. ``path=None``
    means in-memory only (tests, and runtimes with no home); the record
    shape is identical either way.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._last_harvest_at: float = 0.0
        self._last_run_at: float = 0.0
        self._pending: list[dict[str, Any]] = []
        self._runs: list[dict[str, Any]] = []
        self._degraded = ""
        if self.path is not None:
            self._load()

    @property
    def degraded_reason(self) -> str:
        """Why this state degraded, or \"\" when it is healthy."""
        return self._degraded

    @property
    def last_harvest_at(self) -> float:
        return self._last_harvest_at

    @last_harvest_at.setter
    def last_harvest_at(self, value: float) -> None:
        self._last_harvest_at = float(value or 0.0)

    @property
    def last_run_at(self) -> float:
        return self._last_run_at

    @last_run_at.setter
    def last_run_at(self, value: float) -> None:
        self._last_run_at = float(value or 0.0)

    @property
    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._pending)

    @pending.setter
    def pending(self, value: list[dict[str, Any]]) -> None:
        with self._lock:
            self._pending = [dict(row) for row in value or []]

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._quarantine(exc)
            return
        if not isinstance(raw, dict):
            self._quarantine(ValueError(
                "distill state must hold an object, found "
                f"{type(raw).__name__}"))
            return
        self._last_harvest_at = float(raw.get("last_harvest_at") or 0.0)
        self._last_run_at = float(raw.get("last_run_at") or 0.0)
        for row in raw.get("pending", []) or []:
            if not isinstance(row, dict):
                continue
            if not all(isinstance(row.get(k), str)
                       for k in ("conversation_id", "prompt", "answer",
                                 "teacher")):
                continue
            try:
                created = float(row.get("created_at") or 0.0)
            except (TypeError, ValueError):
                continue
            if not created:
                continue
            self._pending.append({
                "conversation_id": str(row["conversation_id"]),
                "prompt": str(row["prompt"]),
                "answer": str(row["answer"]),
                "teacher": str(row["teacher"]),
                "created_at": created,
            })
        runs = raw.get("runs")
        if isinstance(runs, list):
            self._runs = [r for r in runs if isinstance(r, dict)][-MAX_RUNS:]

    def _quarantine(self, exc: Exception) -> None:
        assert self.path is not None
        bad = self.path.with_name(
            f"{self.path.name}.corrupt-{int(time.time())}")
        try:
            os.replace(self.path, bad)
        except OSError:
            bad = self.path
        self._degraded = (
            f"distill state is corrupt — quarantined to {bad}: {exc}")

    def _to_dict(self) -> dict[str, Any]:
        return {
            "last_harvest_at": self._last_harvest_at,
            "last_run_at": self._last_run_at,
            "pending": copy.deepcopy(self._pending),
            "runs": copy.deepcopy(self._runs),
        }

    def save(self) -> None:
        """Persist the state (no-op for in-memory states). Never raises."""
        if self.path is None:
            return
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_text(
                    self.path,
                    json.dumps(self._to_dict(), ensure_ascii=False,
                               indent=1))
            except OSError as exc:  # noqa: BLE001 - durability is a bonus
                _log.warning("distill state save failed: %s", exc)

    def record_run(self, verdict: dict[str, Any]) -> None:
        """Append a tick verdict to the bounded run history."""
        with self._lock:
            self._runs.append(dict(verdict))
            if len(self._runs) > MAX_RUNS:
                del self._runs[:len(self._runs) - MAX_RUNS]

    def runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._runs)


def _is_owner_model(model: str) -> bool:
    """True when ``model`` names one of the operator's own models.

    Lazy import: ``nomorals.llm`` pulls in the whole model plane at
    package init. On import failure the check fails closed — the caller
    treats the turn as unharvestable rather than misclassifying it.
    """
    try:
        from ..llm.defaults import is_owner_model
    except Exception:  # noqa: BLE001 - fail closed, never misclassify
        return True
    try:
        return bool(is_owner_model(model))
    except Exception:  # noqa: BLE001 - fail closed
        return True


def _scrubbed(text: str) -> str:
    """PII-scrubbed, cleaned text — the miner's pair shape."""
    return clean_text(scrub_pii(str(text or "")))


def harvest_kept_turns(
    db: Any,
    *,
    since: float = 0.0,
    pending: list[dict[str, Any]] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Harvest kept cloud turns from the conversation log.

    Pairs each user turn with the next assistant reply in the same
    partner conversation and keeps only the turns the owner kept: a
    cloud teacher's answer the owner continued past. A turn with no
    continuation yet stays pending (re-checked next tick); a turn the
    owner never comes back to drops out after ``PENDING_TTL_S``. The
    store dedupes re-harvested pairs by hash, so a re-scan never
    double-counts. Never raises — a missing table or an unreadable log
    yields zero pairs with the reason named.
    """
    now = time.time() if now is None else now
    pending = [dict(row) for row in (pending or [])]
    stats = {"kept": 0, "pending": 0, "stale_dropped": 0,
             "owner_model": 0, "synthetic": 0, "noise_dropped": 0,
             "duplicates": 0}
    pairs: list[Any] = []
    seen_at = float(since or 0.0)
    try:
        from ..storage.conversations import ConversationLog

        log = ConversationLog(db)
        rows = log.messages_since(since)
    except Exception as exc:  # noqa: BLE001 - no log, no pairs
        if "no such table" in str(exc).lower():
            return {"pairs": [], "pending": pending, "seen_at": seen_at,
                    "stats": stats}
        _log.warning("distill harvest failed: %s", exc)
        return {"pairs": [], "pending": pending, "seen_at": seen_at,
                "stats": {**stats, "error": f"harvest failed: {exc}"}}
    try:
        titles = {str(c.get("id") or ""): str(c.get("title") or "")
                  for c in log.conversations()
                  if c.get("agent") == "partner"}
    except Exception:  # noqa: BLE001 - titles are context only
        titles = {}
    # Group by conversation, oldest first (messages_since already
    # orders by conversation, then time).
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            created = float(row.get("created_at") or 0.0)
        except (TypeError, ValueError):
            continue
        if created > seen_at:
            seen_at = created
        conv = str(row.get("conversation_id") or "")
        if conv not in titles:
            continue
        grouped.setdefault(conv, []).append(row)
    # Continuation lookup: the user timestamps per conversation, so a
    # turn is kept iff the owner spoke again after it.
    user_after: dict[str, list[float]] = {}
    for conv, turns in grouped.items():
        user_after[conv] = sorted(
            float(t.get("created_at") or 0.0)
            for t in turns if str(t.get("role") or "") == "user")
    pending_keys = {
        (str(p.get("conversation_id") or ""),
         float(p.get("created_at") or 0.0)) for p in pending}
    kept_pending: list[dict[str, Any]] = []

    def _consider(conv: str, prompt: str, answer: str, teacher: str,
                  created_at: float) -> None:
        """One candidate pair: kept now, pending, or dropped."""
        key = (conv, created_at)
        if key in pending_keys:
            return  # the pending list already holds this turn
        prompt = _scrubbed(prompt)
        answer = _scrubbed(answer)
        if (not prompt or not answer
                or len(prompt) < MIN_USER_CHARS
                or len(answer) < MIN_ANSWER_CHARS
                or prompt.startswith("/")):
            stats["noise_dropped"] += 1
            return
        later = [t for t in user_after.get(conv, [])
                 if t > created_at]
        if not later:
            kept_pending.append({
                "conversation_id": conv, "prompt": prompt,
                "answer": answer, "teacher": teacher,
                "created_at": created_at})
            stats["pending"] += 1
            return
        pair = capture_owner_kept(
            prompt, titles.get(conv, ""), answer, teacher=teacher)
        if pair is None:
            stats["noise_dropped"] += 1
            return
        pairs.append(pair)
        stats["kept"] += 1

    for conv, turns in grouped.items():
        for i, turn in enumerate(turns):
            if str(turn.get("role") or "") != "user":
                continue
            answer_row = next(
                (t for t in turns[i + 1:]
                 if str(t.get("role") or "") == "assistant"), None)
            if answer_row is None:
                continue
            teacher = str(answer_row.get("model") or "").strip()
            if not teacher:
                continue
            if teacher in SYNTHETIC_MODELS:
                stats["synthetic"] += 1
                continue
            if _is_owner_model(teacher):
                stats["owner_model"] += 1
                continue
            try:
                created = float(answer_row.get("created_at") or 0.0)
            except (TypeError, ValueError):
                continue
            _consider(conv, str(turn.get("content") or ""),
                      str(answer_row.get("content") or ""),
                      teacher, created)
    # Re-check pending turns against this tick's rows: a continuation
    # that arrived since the last tick proves the turn kept.
    still_pending: list[dict[str, Any]] = []
    for row in pending:
        conv = str(row.get("conversation_id") or "")
        try:
            created = float(row.get("created_at") or 0.0)
        except (TypeError, ValueError):
            continue
        if now - created > PENDING_TTL_S:
            stats["stale_dropped"] += 1
            continue
        later = [t for t in user_after.get(conv, []) if t > created]
        if not later:
            still_pending.append(row)
            stats["pending"] += 1
            continue
        pair = capture_owner_kept(
            str(row.get("prompt") or ""), titles.get(conv, ""),
            str(row.get("answer") or ""),
            teacher=str(row.get("teacher") or ""))
        if pair is not None:
            pairs.append(pair)
            stats["kept"] += 1
    kept_pending.extend(still_pending)
    if len(kept_pending) > MAX_PENDING:
        kept_pending = kept_pending[-MAX_PENDING:]
    return {"pairs": pairs, "pending": kept_pending, "seen_at": seen_at,
            "stats": stats}


def run_distill_once(context: Any, *, force: bool = False,
                     now: float | None = None,
                     notifier: Any = None,
                     store: DistillationStore | None = None,
                     registry: TrainingRegistry | None = None,
                     state: DistillState | None = None
                     ) -> dict[str, Any]:
    """Run one distillation tick. Never raises.

    Harvest -> train when enough pairs exist -> registry gate ->
    promote -> notify once on promotion. A not-due tick still
    harvests (pairs accrue daily); a deferred tick trains nothing.
    """
    now = time.time() if now is None else now
    try:
        home = _home_of(context)
        db = getattr(context, "db", None)
        state = state if state is not None else (
            DistillState(default_distill_state_path(home))
            if home is not None else DistillState())
        store = store if store is not None else store_for_context(context)
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("distill.tick.setup", exc)
        return {"ok": False, "status": "failed",
                "error": f"distill tick failed: {exc}"}
    try:
        return _run_once(context, now=now, force=force,
                         notifier=notifier, store=store,
                         registry=registry, state=state, home=home, db=db)
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("distill.tick", exc)
        return {"ok": False, "status": "failed",
                "error": f"distill tick failed: {exc}"}


def _run_once(context: Any, *, now: float, force: bool,
              notifier: Any, store: DistillationStore,
              registry: TrainingRegistry | None,
              state: DistillState, home: Path | None,
              db: Any) -> dict[str, Any]:
    # 1. Harvest: kept cloud turns since the watermark (first tick
    # looks back 30 days, never the whole archive).
    since = state.last_harvest_at
    if not since:
        since = now - FIRST_RUN_LOOKBACK_S
    harvested = 0
    harvest_stats: dict[str, Any] = {}
    if db is not None:
        result = harvest_kept_turns(
            db, since=since, pending=state.pending, now=now)
        for pair in result["pairs"]:
            if store.add(pair):
                harvested += 1
        state.pending = result["pending"]
        state.last_harvest_at = max(since, float(result["seen_at"] or 0.0))
        harvest_stats = dict(result["stats"])
    # 2. Interval gate: the train runs at most daily unless forced.
    # The harvest above already ran — pairs accrue even on a not-due
    # tick.
    if (not force and state.last_run_at
            and now - state.last_run_at < DISTILL_INTERVAL_S):
        verdict: dict[str, Any] = {
            "status": "not_due", "pairs": len(store),
            "harvested": harvested, "harvest": harvest_stats,
            "last_run_at": state.last_run_at,
            "interval_hours": DISTILL_INTERVAL_S / 3600,
        }
        state.record_run(verdict)
        state.save()
        return {"ok": True, **verdict}
    state.last_run_at = now
    # 3. The train: periodic_distill owns the min-pairs gate, the
    # NativeTrainer run, and the registry complete/evaluate/promote
    # sequence. A failed gate means the candidate never lands.
    out_dir = None
    if home is not None:
        out_dir = default_distill_out_dir(home) / f"run-{int(now)}"
    if registry is None and db is not None:
        try:
            registry = TrainingRegistry(db)
        except Exception as exc:  # noqa: BLE001 - train ungated
            _log.warning("distill registry unavailable: %s", exc)
            registry = None
    try:
        result = periodic_distill(
            store, min_pairs=MIN_PAIRS, registry=registry,
            run_name="distillation", out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001 - a failed train is a verdict
        verdict = {"status": "failed", "pairs": len(store),
                   "harvested": harvested, "harvest": harvest_stats,
                   "error": str(exc), "last_run_at": state.last_run_at}
        state.record_run(verdict)
        state.save()
        return {"ok": True, **verdict}
    verdict = {"status": str(result.get("status") or "failed"),
               "pairs": len(store), "harvested": harvested,
               "harvest": harvest_stats,
               "last_run_at": state.last_run_at}
    if verdict["status"] == "deferred":
        verdict["reason"] = (
            f"not enough distillation pairs yet "
            f"({len(store)} < {MIN_PAIRS})")
        verdict["min_pairs"] = MIN_PAIRS
    if verdict["status"] == "trained":
        verdict["gate_passed"] = bool(result.get("gate_passed", False))
        verdict["promoted"] = bool(result.get("promoted", False))
        verdict["checkpoint"] = str(result.get("checkpoint") or "")
        verdict["examples"] = int(result.get("examples") or 0)
    if verdict["status"] == "failed":
        verdict["error"] = str(result.get("error") or "")
    state.record_run(verdict)
    state.save()
    if verdict["status"] == "trained" and verdict.get("promoted") \
            and notifier is not None:
        with contextlib.suppress(Exception):
            notifier.publish(
                "training",
                "🏋️ distilled adapter promoted",
                f"distilled adapter trained from {verdict['pairs']} kept "
                f"cloud turns and passed the promotion gate\n"
                f"checkpoint: {verdict['checkpoint']}",
            )
    return {"ok": True, **verdict}


def ensure_distill_job(scheduler: Any, *,
                       spec: str = DISTILL_SPEC) -> Any:
    """Register the durable distill tick on the live scheduler.

    Mirror of the continual-update / swarm-board / digest / watch-loop
    ensures: a routine tool job, one per scheduler, replaced when its
    spec goes stale. Never raises — scheduling is best-effort; a
    missing or broken scheduler yields None.
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name != DISTILL_JOB:
                continue
            job_spec = (job.get("spec") if isinstance(job, dict)
                        else getattr(job, "spec", None))
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                job_id = (job.get("id") if isinstance(job, dict)
                          else getattr(job, "id", None))
                scheduler.remove(job_id)
            break
        return scheduler.add(DISTILL_JOB, spec, "tool",
                             {"tool": DISTILL_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
