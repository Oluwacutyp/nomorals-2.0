"""Daily driver for the confined continual-update loop (audit gap R2-32).

``training/continual.py`` was complete and test-pinned but had zero
production callers: no durable replay buffer ever accrued and no
scheduled update ever fired. This module is the wiring that makes it
run — it is not a second retrain system:

* :class:`ContinualStore` — durable replay buffer + frozen eval harness
  + run history, WorkflowStore shape (atomic tmp+replace, corrupt →
  quarantine + degrade to empty, bad rows skipped, reads never raise).
* :func:`run_continual_update_once` — one daily tick: harvest fresh
  preference pairs from the live representation ledger
  (``build_preference_pairs`` — owner corrections + counterfactuals,
  nothing fabricated), accrue them into the buffer, freeze the eval
  harness on the first run with enough pairs (chronological held-out
  split; the frozen set is never re-split), run
  ``ContinualUpdater.scheduled_update`` against the live preference
  adapter, and — only on accept — save the candidate as the live
  adapter and tell the owner once. A rejection leaves the previous
  adapter file untouched: the previous model stays in place.
* ``ensure_continual_update_job`` + the ``__continual_update_tick__``
  routine tool — a durable daily scheduler job, the swarm-board /
  workflow / drift pattern.

Confinement: the loop touches only its own store and the preference
adapter file. It never touches the full self-improvement pipeline's
artifacts (no dataset, no run registry row, no model dir, no mission).
The adapter is a ``PreferenceAdapter`` — a small token-weight delta;
base model weights are never opened, never rewritten.

Layering: L3 (``training``), same package as ``continual.py`` /
``preference.py`` / ``distillation.py``. It never imports the notifier
— the tick takes an injected one (the digest lane's lesson).
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
from .continual import (
    ContinualUpdater,
    FixedEvalHarness,
    ReplayBuffer,
)
from .preference import (
    MIN_HELD_OUT,
    PreferenceAdapter,
    PreferencePair,
    build_preference_pairs,
    split_held_out,
)

_log = get_logger(__name__)

__all__ = [
    "CONTINUAL_JOB",
    "CONTINUAL_SPEC",
    "CONTINUAL_TOOL",
    "ContinualStore",
    "default_continual_path",
    "default_preference_adapter_dir",
    "ensure_continual_update_job",
    "load_saved_adapter",
    "run_continual_update_once",
    "store_for_context",
]

CONTINUAL_JOB = "continual update tick"
CONTINUAL_SPEC = "every 24h"
CONTINUAL_TOOL = "__continual_update_tick__"

#: The continual loop only runs when the buffer holds at least this
#: many pairs. 15 is the smallest buffer whose chronological split
#: freezes a >= MIN_HELD_OUT (3) held-out set — the production gate's
#: own floor ("too little evidence is a reject, not a pass") — so the
#: first run that can fire can also freeze a real harness. 8..14 pairs
#: would split to 1..2 held-out pairs and defer forever.
MIN_BUFFER = 15
#: Replay buffer cap, matching ``ReplayBuffer.capacity``.
BUFFER_CAPACITY = 2000
#: Run history kept in the store (newest last).
MAX_RUNS = 50


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + os.replace)."""
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def default_continual_path(home: Path | str | None = None) -> Path:
    """The durable continual store under a runtime's home."""
    base = Path(home) if home is not None else Path("~/.nomorals").expanduser()
    return base / "data" / "continual.json"


def default_preference_adapter_dir(home: Path | str | None = None) -> Path:
    """Where the live preference adapter file lives under a home."""
    base = Path(home) if home is not None else Path("~/.nomorals").expanduser()
    return base / "training" / "preference-adapter"


def store_for_context(context: Any) -> ContinualStore:
    """The durable continual store under a runtime's home (in-memory
    when the runtime has no home). Never raises."""
    try:
        home = getattr(getattr(context, "settings", None), "home", None)
        if home is None:
            home = getattr(context, "home", None)
        if home is None:
            return ContinualStore()
        return ContinualStore(path=default_continual_path(home))
    except Exception:  # noqa: BLE001 - durability is a bonus
        return ContinualStore()


def load_saved_adapter(home: Path | str | None) -> PreferenceAdapter | None:
    """The saved live adapter for a home, or None (absent / unreadable).

    Never raises — a missing or corrupt adapter file means no tuned
    adapter is active, exactly as if the loop had never run.
    """
    if home is None:
        return None
    try:
        path = default_preference_adapter_dir(home) / "preference_adapter.json"
        if not path.is_file():
            return None
        return PreferenceAdapter.load(path)
    except Exception as exc:  # noqa: BLE001 - degrade to no adapter
        _log.warning("saved preference adapter unreadable: %s", exc)
        return None


class ContinualStore:
    """Durable replay buffer + frozen harness + run history.

    Shape follows WorkflowStore / SwarmBoardStore: tmp+replace writes;
    a corrupt store is quarantined and degrades to empty with a named
    reason; bad rows are skipped; reads never raise. ``path=None``
    means in-memory only (tests, and runtimes with no home); the
    record shape is identical either way.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._buffer: list[dict[str, Any]] = []
        self._last_run_at: float = 0.0
        self._harness: dict[str, Any] | None = None
        self._runs: list[dict[str, Any]] = []
        self._degraded = ""
        if self.path is not None:
            self._load()

    @property
    def degraded_reason(self) -> str:
        """Why this store degraded, or \"\" when it is healthy."""
        return self._degraded

    @property
    def last_run_at(self) -> float:
        return self._last_run_at

    @last_run_at.setter
    def last_run_at(self, value: float) -> None:
        self._last_run_at = float(value or 0.0)

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
                "continual store must hold an object, found "
                f"{type(raw).__name__}"))
            return
        for row in raw.get("buffer", []) or []:
            if not isinstance(row, dict):
                continue
            pair = row.get("pair")
            if not isinstance(pair, dict):
                continue
            try:
                PreferencePair.from_dict(pair)
            except (TypeError, ValueError):
                continue
            self._buffer.append({
                "pair": pair,
                "kind": str(row.get("kind") or "pair"),
                "added_at": float(row.get("added_at") or 0.0),
            })
        self._last_run_at = float(raw.get("last_run_at") or 0.0)
        harness = raw.get("harness")
        if isinstance(harness, dict) and isinstance(
                harness.get("pairs"), list):
            pairs: list[dict[str, Any]] = []
            for row in harness["pairs"]:
                if not isinstance(row, dict):
                    continue
                try:
                    PreferencePair.from_dict(row)
                except (TypeError, ValueError):
                    continue
                pairs.append(row)
            if pairs:
                self._harness = {
                    "pairs": pairs,
                    "fingerprint": str(harness.get("fingerprint") or ""),
                }
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
            f"continual store is corrupt — quarantined to {bad}: {exc}")

    def _to_dict(self) -> dict[str, Any]:
        return {
            "buffer": copy.deepcopy(self._buffer),
            "last_run_at": self._last_run_at,
            "harness": copy.deepcopy(self._harness),
            "runs": copy.deepcopy(self._runs),
        }

    def save(self) -> None:
        """Persist the store (no-op for in-memory stores). Never raises."""
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
                _log.warning("continual store save failed: %s", exc)

    def pairs(self) -> list[PreferencePair]:
        """The buffered pairs, oldest first."""
        with self._lock:
            out: list[PreferencePair] = []
            for row in self._buffer:
                try:
                    out.append(PreferencePair.from_dict(row["pair"]))
                except (TypeError, ValueError, KeyError):
                    continue
            return out

    def add_pairs(self, pairs: Any) -> int:
        """Append new pairs (dedupe by pair_id); returns how many landed.

        The store caps at ``BUFFER_CAPACITY`` by dropping the oldest
        entries — the buffer is a reservoir, not an archive.
        """
        seen = {row["pair"].get("pair_id")
                for row in self._buffer if isinstance(row, dict)}
        added = 0
        with self._lock:
            for pair in pairs or []:
                try:
                    data = pair.to_dict()
                except AttributeError:
                    continue
                if data.get("pair_id") in seen:
                    continue
                seen.add(data.get("pair_id"))
                self._buffer.append({
                    "pair": data, "kind": "pair",
                    "added_at": float(getattr(pair, "created_at", 0.0)
                                      or 0.0),
                })
                added += 1
            if len(self._buffer) > BUFFER_CAPACITY:
                del self._buffer[:len(self._buffer) - BUFFER_CAPACITY]
        return added

    def harness_pairs(self) -> list[PreferencePair]:
        """The frozen held-out pairs (empty until the harness freezes)."""
        if not self._harness:
            return []
        out: list[PreferencePair] = []
        for row in self._harness.get("pairs", []):
            try:
                out.append(PreferencePair.from_dict(row))
            except (TypeError, ValueError):
                continue
        return out

    def harness_fingerprint(self) -> str:
        return str((self._harness or {}).get("fingerprint") or "")

    def freeze_harness(self) -> str:
        """Freeze the held-out set from the current buffer (once).

        Chronological split: the most recent pairs are the held-out
        set. Returns the fingerprint, or \"\" when there are too few
        pairs to freeze a meaningful harness. Never re-freezes: once
        frozen, the harness only moves by explicit intent, never by
        a later tick.
        """
        if self._harness:
            return self.harness_fingerprint()
        pairs = self.pairs()
        _train, held_out = split_held_out(pairs)
        if len(held_out) < MIN_HELD_OUT:
            return ""
        harness = FixedEvalHarness(held_out)
        with self._lock:
            self._harness = {
                "pairs": [p.to_dict() for p in held_out],
                "fingerprint": harness.fingerprint,
            }
        return harness.fingerprint

    def record_run(self, verdict: dict[str, Any]) -> None:
        """Append a tick verdict to the bounded run history."""
        with self._lock:
            self._runs.append(dict(verdict))
            if len(self._runs) > MAX_RUNS:
                del self._runs[:len(self._runs) - MAX_RUNS]

    def runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._runs)


def _home_of(context: Any) -> Path | None:
    home = getattr(getattr(context, "settings", None), "home", None)
    if home is None:
        home = getattr(context, "home", None)
    return Path(home) if home is not None else None


def _set_live_brain_adapter(context: Any, adapter: Any) -> bool:
    """Hand an accepted adapter to the runtime's cached brain, if any.

    Best-effort: the durable file is the source of truth (fresh brains
    load it at construction); the live hand-off just makes the current
    process serve it without a restart.
    """
    try:
        brain = getattr(context, "_brain_for_ctx", None)
        setter = getattr(brain, "set_preference_adapter", None)
        if brain is None or setter is None:
            return False
        setter(adapter)
        return True
    except Exception:  # noqa: BLE001 - live hand-off is a bonus
        return False


def run_continual_update_once(context: Any, *, force: bool = False,
                              now: float | None = None,
                              notifier: Any = None,
                              store: ContinualStore | None = None
                              ) -> dict[str, Any]:
    """Run one continual-update tick. Never raises.

    Harvest → accrue → freeze harness (first run) → confined update →
    save-on-accept + notify. A rejected candidate never touches the
    live adapter file; a not-due or deferred tick notifies nobody.
    """
    now = time.time() if now is None else now
    try:
        store = store if store is not None else store_for_context(context)
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("continual.tick.setup", exc)
        return {"ok": False, "status": "failed",
                "error": f"continual tick failed: {exc}"}
    try:
        return _run_once(context, now=now, force=force,
                         notifier=notifier, store=store)
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("continual.tick", exc)
        return {"ok": False, "status": "failed",
                "error": f"continual tick failed: {exc}"}


def _run_once(context: Any, *, now: float, force: bool,
              notifier: Any, store: ContinualStore) -> dict[str, Any]:
    home = _home_of(context)
    # 1. Harvest fresh pairs from the live representation ledger.
    # Owner corrections + counterfactuals only; approved records are
    # counted, never fabricated into pairs (see build_preference_pairs).
    fresh: list[PreferencePair] = []
    try:
        from ..cognition.representation import RepresentationLedger

        ledger = RepresentationLedger(getattr(context, "db", None))
        fresh, _stats = build_preference_pairs(ledger)
    except Exception as exc:  # noqa: BLE001 - no pairs, not a crash
        _log.warning("continual harvest failed: %s", exc)
        fresh = []
    added = store.add_pairs(fresh)
    buffer = store.pairs()
    if len(buffer) < MIN_BUFFER:
        verdict = {"status": "deferred",
                   "reason": (f"not enough replay data yet "
                              f"({len(buffer)} < {MIN_BUFFER})"),
                   "buffer": len(buffer), "added": added,
                   "last_run_at": store.last_run_at}
        store.record_run(verdict)
        store.save()
        return {"ok": True, **verdict}
    # 2. Freeze the eval harness on the first run with enough pairs.
    # The frozen set is never re-split: later runs rebuild it from the
    # stored pairs and verify the fingerprint.
    fingerprint = store.freeze_harness()
    if not fingerprint:
        verdict = {"status": "deferred",
                   "reason": "not enough pairs to freeze an eval harness",
                   "buffer": len(buffer), "added": added,
                   "last_run_at": store.last_run_at}
        store.record_run(verdict)
        store.save()
        return {"ok": True, **verdict}
    harness = FixedEvalHarness(store.harness_pairs())
    if not harness.check_fingerprint(fingerprint):
        verdict = {"status": "failed",
                   "reason": ("harness fingerprint mismatch — the frozen "
                              "eval set changed, refusing to update"),
                   "buffer": len(buffer),
                   "fingerprint": fingerprint,
                   "last_run_at": store.last_run_at}
        store.record_run(verdict)
        store.save()
        return {"ok": True, **verdict}
    # 3. The confined update: adapter-only, fixed-harness gated.
    # The replay buffer is the TRAIN set only — the frozen harness
    # pairs are excluded by content, so the candidate never trains
    # on the pairs it is evaluated against (and near-duplicates of
    # them cannot leak in either).
    adapter = load_saved_adapter(home) or PreferenceAdapter.empty()
    held = {(p.prompt, p.chosen, p.rejected)
            for p in store.harness_pairs()}
    replay = ReplayBuffer()
    for pair in buffer:
        if (pair.prompt, pair.chosen, pair.rejected) not in held:
            replay.add(pair, kind="pair")
    updater = ContinualUpdater(harness, replay,
                               last_run_at=store.last_run_at)
    result = updater.scheduled_update(adapter, force=force, now=now)
    store.last_run_at = updater.last_run_at
    status = str(result.get("status") or "failed")
    verdict: dict[str, Any] = {
        "status": status,
        "buffer": len(buffer), "added": added,
        "fingerprint": fingerprint,
        "last_run_at": store.last_run_at,
    }
    if status in ("accepted", "rejected"):
        verdict["before"] = result.get("before")
        verdict["after"] = result.get("after")
    if status == "rejected":
        verdict["reason"] = str(result.get("reason") or "rejected")
    if status == "not_due":
        verdict["interval_hours"] = updater.interval_hours
    if status == "deferred":
        verdict["reason"] = str(result.get("reason") or "deferred")
        verdict["min_batch"] = result.get("min_batch")
    if status == "accepted":
        candidate = result["candidate"]
        live_applied = False
        if home is not None:
            candidate.save(default_preference_adapter_dir(home))
            live_applied = _set_live_brain_adapter(context, candidate)
        verdict["live_applied"] = live_applied
        verdict["pair_count"] = int(
            getattr(candidate, "pair_count", 0) or 0)
        if notifier is not None:
            before = result.get("before") or {}
            after = result.get("after") or {}
            with contextlib.suppress(Exception):
                notifier.publish(
                    "training",
                    "🏋️ continual update accepted",
                    f"preference adapter updated from {len(buffer)} "
                    f"replay pairs\nfixed harness: "
                    f"{before.get('accuracy')} → "
                    f"{after.get('accuracy')} "
                    f"(fingerprint {fingerprint})\n"
                    f"adapter now covers "
                    f"{verdict['pair_count']} pairs",
                )
    store.record_run(verdict)
    store.save()
    return {"ok": True, **verdict}


def ensure_continual_update_job(scheduler: Any, *,
                                spec: str = CONTINUAL_SPEC) -> Any:
    """Register the durable continual-update tick on the live scheduler.

    Mirror of the swarm-board / digest / watch-loop / drift / workflow
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
            if name != CONTINUAL_JOB:
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
        return scheduler.add(CONTINUAL_JOB, spec, "tool",
                             {"tool": CONTINUAL_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
