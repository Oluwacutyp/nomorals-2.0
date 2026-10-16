"""Unified scheduler — ONE scheduler for ambient + deliberate work.

Replaces the split-brain pattern where idle ticks, presence heartbeats,
and missions each ran on their own timing. Here there is one queue and
one ``run_due()``:

* **Ambient work** — organ ticks registered with
  :meth:`UnifiedScheduler.register_organ_tick`. A tick is an organ's
  periodic job (e.g. presence heartbeat every 30 min). Ticks are
  in-memory registrations: callables cannot survive a restart, so after
  a restart the owner re-registers them (the same way the old
  ``ensure_heartbeat_job`` re-registered on boot).
* **Deliberate work** — missions. ``Mission = organ tick with a plan``:
  a mission is a dict (or a :class:`nomorals.missions.mission.Mission`,
  duck-typed) carrying a ``plan`` — a list of steps. Each step names a
  handler (``{"name": ..., "handler": ...}``); handlers are callables
  registered with :meth:`register_handler` and re-registered after a
  restart. The *mission itself* — goal, plan, due time, status — is
  persisted, so it survives ``kill -9``.

Kill-9 durability: the mission queue lives in a JSON file written
**write-ahead** — the new queue is written to a temp file in the same
directory, fsynced, then atomically renamed over the old queue, then
the directory is fsynced. A crash mid-write leaves either the old or
the new queue intact, never a torn half-file. A mission is marked
``running`` (and persisted) *before* its steps execute, so a kill
mid-run leaves it ``running``; on reload, ``running`` missions are
treated as due again (at-least-once, the same contract as
``nomorals.missions.runner`` — a step may run twice, never zero times
without a record).

Optional collaborators (all duck-typed, all optional):

* ``rhythm`` — an :class:`OwnerRhythm`-like object with
  ``should_fire(now, priority=..., kind=...)``. Quiet-hour / busy /
  low-intensity deferrals happen here, before any work runs.
* ``budget`` — a :class:`BudgetGovernor`-like object with
  ``check(organ)`` returning a mapping with ``allowed``. Over-ceiling
  work is deferred, not executed.
* ``why_store`` — a :class:`WhyStore`-like object with ``record(...)``.
  Every firing (mission or tick) is explained there for ``nm why``.
* ``preemption`` — a :class:`PreemptionLog`-like object with
  ``log_preempted(...)``. Used when a firing is displaced by
  higher-priority work (currently: budget deferrals are reported as
  deferred, not as preemptions — preemption is reserved for one
  firing displacing another).

``run_due()`` never raises for one bad firing: a failing tick or
mission step is recorded (failed tick / failed mission) and the rest
of the due list still runs.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["UnifiedScheduler", "atomic_write_json", "normalize_mission"]

#: Priority order, highest first. Mirrors the coordinator ladder in
#: ``nomorals.autonomy.preemption`` without importing it (keeps this
#: module dependency-free so it can be imported standalone).
_PRIORITY_RANK = {
    "safety": 0,
    "owner-direct": 1,
    "owner_direct": 1,
    "scheduled": 2,
    "ambient": 3,
    "background": 4,
}


def _priority_rank(priority: str) -> int:
    return _PRIORITY_RANK.get(str(priority or "scheduled").lower(), 2)


def _new_id(prefix: str = "sched") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _to_ts(now: Any) -> float:
    """Accept None / epoch float / datetime for ``now``."""
    if now is None:
        return time.time()
    if isinstance(now, (int, float)):
        return float(now)
    to_ts = getattr(now, "timestamp", None)
    if callable(to_ts):
        return float(to_ts())
    return float(now)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as JSON atomically (write-ahead + rename).

    Temp file in the same directory → flush + fsync → ``os.replace``
    (atomic on POSIX) → fsync the directory. A crash at any point
    leaves either the old file or the complete new file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True,
                      default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    try:
        dir_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass  # directory fsync is best-effort on some filesystems


def normalize_mission(mission: Any, *, now: float | None = None) -> dict[str, Any]:
    """Normalize a mission (dict / Mission / duck-typed object) to a
    JSON-able queue record. Never mutates the caller's object."""
    ts = now if now is not None else time.time()
    if isinstance(mission, dict):
        data = dict(mission)
    elif hasattr(mission, "to_dict"):
        try:
            data = dict(mission.to_dict())
        except Exception:  # noqa: BLE001 - fall back to attributes
            data = {}
    else:
        data = {}
    if not data:
        for attr in ("id", "goal", "name", "status", "state", "metadata",
                     "priority", "tags"):
            value = getattr(mission, attr, None)
            if value is not None:
                data[attr] = value
        state = data.get("state") or {}
        if isinstance(state, dict) and state.get("plan"):
            data["plan"] = state["plan"]
    plan_raw = data.get("plan") or data.get("steps") or []
    plan: list[dict[str, Any]] = []
    for step in plan_raw if isinstance(plan_raw, list) else []:
        if isinstance(step, str):
            plan.append({"name": step, "handler": step})
        elif isinstance(step, dict):
            entry = dict(step)
            entry.setdefault("name", entry.get("handler", "step"))
            entry.setdefault("handler", entry.get("name", "step"))
            plan.append(entry)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    record = {
        "id": str(data.get("id") or _new_id("mission")),
        "goal": str(data.get("goal") or data.get("name") or "unspecified mission"),
        "name": str(data.get("name") or data.get("goal") or "mission"),
        "plan": plan,
        "status": str(data.get("status") or "pending"),
        "priority": str(data.get("priority") or metadata.get("priority")
                        or "scheduled").lower(),
        "organ": str(data.get("organ") or metadata.get("organ") or ""),
        "due_at": float(data.get("due_at") or data.get("run_at") or ts),
        "created_at": float(data.get("created_at") or ts),
        "attempts": int(data.get("attempts") or 0),
        "error": str(data.get("error") or ""),
        "metadata": metadata,
    }
    if record["status"] not in {"pending", "running", "done", "failed", "cancelled"}:
        record["status"] = "pending"
    return record


class UnifiedScheduler:
    """ONE scheduler: ambient organ ticks + deliberate missions."""

    def __init__(self, queue_path: str | Path, *,
                 why_store: Any = None,
                 rhythm: Any = None,
                 budget: Any = None,
                 preemption: Any = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.queue_path = Path(queue_path)
        self.why_store = why_store
        self.rhythm = rhythm
        self.budget = budget
        self.preemption = preemption
        self._clock = clock or time.time
        #: name -> {fn, interval, priority, organ, next_due}
        self._ticks: dict[str, dict[str, Any]] = {}
        #: handler name -> callable (mission plan steps)
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._missions: list[dict[str, Any]] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        if not self.queue_path.exists():
            self._missions = []
            return
        try:
            raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
            missions = raw.get("missions", []) if isinstance(raw, dict) else raw
            self._missions = [normalize_mission(m) for m in missions or []]
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            _log.warning("unified scheduler queue unreadable (%s) — "
                         "starting empty rather than inventing state", exc)
            self._missions = []

    def _persist(self) -> None:
        atomic_write_json(self.queue_path,
                          {"version": 1, "missions": self._missions})

    # ── registration ─────────────────────────────────────────────

    def register_organ_tick(self, name: str, fn: Callable[..., Any],
                            interval: float, *,
                            priority: str = "ambient",
                            organ: str = "",
                            start_now: float | None = None) -> str:
        """Register a periodic organ tick. ``interval`` in seconds."""
        if not callable(fn):
            raise ValueError(f"organ tick {name!r} needs a callable")
        if interval <= 0:
            raise ValueError(f"organ tick {name!r} needs interval > 0")
        now = self._clock()
        self._ticks[str(name)] = {
            "fn": fn,
            "interval": float(interval),
            "priority": str(priority or "ambient").lower(),
            "organ": str(organ or name),
            "next_due": float((start_now if start_now is not None
                              else now) + interval),
        }
        return str(name)

    def register_handler(self, name: str, fn: Callable[..., Any]) -> str:
        """Register a handler for mission plan steps named ``name``."""
        if not callable(fn):
            raise ValueError(f"handler {name!r} needs a callable")
        self._handlers[str(name)] = fn
        return str(name)

    def unregister_tick(self, name: str) -> bool:
        return self._ticks.pop(str(name), None) is not None

    # ── missions ─────────────────────────────────────────────────

    def schedule_mission(self, mission: Any) -> str:
        """Persist a mission (write-ahead) and return its id.

        The mission is durable *before* this returns — a kill-9
        immediately after scheduling loses nothing.
        """
        record = normalize_mission(mission, now=self._clock())
        record["status"] = "pending"
        self._missions.append(record)
        self._persist()
        _log.info("unified scheduler: mission %s scheduled (%s)",
                  record["id"], record["goal"][:60])
        return record["id"]

    def get_mission(self, mission_id: str) -> dict[str, Any] | None:
        for mission in self._missions:
            if mission["id"] == mission_id:
                return dict(mission)
        return None

    def pending_missions(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self._missions
                if m["status"] in ("pending", "running")]

    def all_missions(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self._missions]

    def cancel_mission(self, mission_id: str) -> bool:
        for mission in self._missions:
            if mission["id"] == mission_id and mission["status"] in (
                    "pending", "running"):
                mission["status"] = "cancelled"
                self._persist()
                return True
        return False

    # ── execution ────────────────────────────────────────────────

    def _gated(self, *, priority: str, organ: str, kind: str,
               now: float) -> tuple[bool, str]:
        """Rhythm + budget gates. Returns (allowed, reason)."""
        if self.rhythm is not None:
            try:
                if not self.rhythm.should_fire(
                        now, priority=priority, kind=kind):
                    return False, "rhythm: quiet hours / busy / low intensity"
            except Exception as exc:  # noqa: BLE001 - a broken gate defers, never crashes
                _log.warning("rhythm gate failed: %s", exc)
                return False, f"rhythm gate error: {exc}"
        if self.budget is not None:
            try:
                verdict = self.budget.check(organ or "scheduler")
                if isinstance(verdict, dict) and not verdict.get("allowed", True):
                    return False, str(verdict.get("reason") or "budget deferred")
                if verdict is False:
                    return False, "budget deferred"
            except Exception as exc:  # noqa: BLE001
                _log.warning("budget gate failed: %s", exc)
                return False, f"budget gate error: {exc}"
        return True, ""

    def _explain(self, *, what_fired: str, evidence: dict[str, Any],
                 did: str, happened: str, model: str = "") -> None:
        if self.why_store is None:
            return
        try:
            self.why_store.record(
                what_fired=what_fired, evidence=evidence, model=model,
                did=did, happened=happened)
        except Exception:  # noqa: BLE001 - explanation never breaks execution
            _log.debug("why record failed", exc_info=True)

    def _run_mission(self, mission: dict[str, Any],
                     now: float) -> dict[str, Any]:
        mission["status"] = "running"
        mission["attempts"] = int(mission.get("attempts") or 0) + 1
        self._persist()  # running is durable before any step executes
        step_results: list[dict[str, Any]] = []
        failure = ""
        for step in mission.get("plan") or []:
            handler_name = str(step.get("handler") or step.get("name") or "")
            fn = self._handlers.get(handler_name)
            if fn is None:
                step_results.append({
                    "step": step.get("name", handler_name),
                    "status": "skipped",
                    "reason": f"no handler registered for {handler_name!r} — "
                              "re-register handlers after restart",
                })
                continue
            try:
                outcome = fn(mission, step)
            except Exception as exc:  # noqa: BLE001 - one bad step fails the mission, not run_due
                failure = f"step {step.get('name')!r} raised {type(exc).__name__}: {exc}"
                step_results.append({"step": step.get("name", handler_name),
                                     "status": "failed", "error": failure[:300]})
                break
            step_results.append({"step": step.get("name", handler_name),
                                 "status": "ok",
                                 "outcome": outcome if isinstance(outcome, (str, int, float, bool)) else str(outcome)[:300]})
        if failure:
            mission["status"] = "failed"
            mission["error"] = failure[:500]
            happened = f"failed: {failure[:200]}"
        else:
            mission["status"] = "done"
            mission["error"] = ""
            happened = (f"completed {len(step_results)} plan step(s)"
                        if step_results else "completed (plan was empty)")
        self._persist()
        self._explain(
            what_fired=f"mission:{mission['id']}",
            evidence={"goal": mission["goal"], "name": mission["name"],
                      "priority": mission["priority"],
                      "plan": [s.get("name") for s in mission.get("plan") or []]},
            did=f"ran mission {mission['name']!r} "
                f"({len(step_results)} step(s): "
                f"{', '.join(s['step'] + ':' + s['status'] for s in step_results)})",
            happened=happened)
        return {"kind": "mission", "id": mission["id"],
                "status": mission["status"], "steps": step_results,
                "error": mission["error"]}

    def _run_tick(self, name: str, tick: dict[str, Any],
                  now: float) -> dict[str, Any]:
        try:
            outcome = tick["fn"]()
            tick["next_due"] = now + tick["interval"]
            self._explain(
                what_fired=f"organ-tick:{name}",
                evidence={"organ": tick["organ"], "priority": tick["priority"],
                          "interval_s": tick["interval"]},
                did=f"ran organ tick {name!r}",
                happened=f"ok: {str(outcome)[:200]}" if outcome is not None else "ok")
            return {"kind": "tick", "name": name, "status": "ok",
                    "outcome": str(outcome)[:300] if outcome is not None else ""}
        except Exception as exc:  # noqa: BLE001 - a failing tick never kills run_due
            tick["next_due"] = now + tick["interval"]
            self._explain(
                what_fired=f"organ-tick:{name}",
                evidence={"organ": tick["organ"], "priority": tick["priority"]},
                did=f"ran organ tick {name!r}",
                happened=f"failed: {type(exc).__name__}: {exc}")
            return {"kind": "tick", "name": name, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:300]}

    def run_due(self, now: Any = None) -> list[dict[str, Any]]:
        """Run every due mission and tick. Returns one result per firing
        (including deferred ones — deferral is a result, not silence)."""
        now_ts = _to_ts(now) if now is not None else self._clock()
        results: list[dict[str, Any]] = []

        due_missions = [m for m in self._missions
                        if m["status"] in ("pending", "running")
                        and float(m.get("due_at") or 0) <= now_ts]
        due_missions.sort(key=lambda m: (_priority_rank(m.get("priority")),
                                         float(m.get("due_at") or 0)))
        for mission in due_missions:
            allowed, reason = self._gated(
                priority=mission.get("priority", "scheduled"),
                organ=mission.get("organ") or mission.get("name", ""),
                kind="mission", now=now_ts)
            if not allowed:
                results.append({"kind": "mission", "id": mission["id"],
                                "status": "deferred", "reason": reason})
                continue
            results.append(self._run_mission(mission, now_ts))

        due_ticks = [(n, t) for n, t in self._ticks.items()
                     if float(t.get("next_due") or 0) <= now_ts]
        due_ticks.sort(key=lambda kv: _priority_rank(kv[1].get("priority")))
        for name, tick in due_ticks:
            allowed, reason = self._gated(
                priority=tick.get("priority", "ambient"),
                organ=tick.get("organ") or name,
                kind="tick", now=now_ts)
            if not allowed:
                results.append({"kind": "tick", "name": name,
                                "status": "deferred", "reason": reason})
                continue
            results.append(self._run_tick(name, tick, now_ts))
        return results
