"""The autonomous cascade — one heartbeat that keeps the whole intelligence
stack moving (wave 51).

``CognitiveLoop.tick()`` is a single supervised step through the stack:

  1. **goals** — every active goal advances one step. Goals with a linked
     project are driven *through* the project (plan → step → self-correct),
     and the project's progress is mirrored back onto the goal.
  2. **improve** — one closed-loop improvement cycle (only when the
     improvement mode is not ``off``; a mock/unmeasurable provider is
     skipped, never blocking).
  3. **train** — the personal-model fine-tune pipeline, run only when the
     retraining policy says data has accumulated (collect → curate → train →
     evaluate → promote, with the regression gate). A no-op when not due.

Everything is bounded (per-tick caps) and every stage degrades gracefully:
a failure in one stage is recorded in the summary and never stops the others.
The loop is the thing the scheduler calls on a cadence (``cognitive loop``
job), and it is directly callable from the main AI and sub-agents via the
``autonomy`` tool (``action=tick``) or ``nm autonomy tick``.

Off by default (``settings.autonomy.enabled``); power mode turns it on
automatically — the "autonomy dial".
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

__all__ = ["CognitiveLoop", "ModelBudget", "autonomy_enabled",
           "set_autonomy_enabled", "SelfImprovementStatus", "register"]

_log = get_logger(__name__)

#: Durable override so ``nm autonomy on`` survives process restarts. The
#: config/env value is the default; a kv flag (set by the CLI) wins.
_OVERRIDE_KEY = "autonomy.enabled"


def set_autonomy_enabled(context: Any, enabled: bool) -> bool:
    """Persistently enable/disable the autonomous cascade."""
    try:
        import json
        db = getattr(context, "db", None)
        if db is None:
            return False
        db.execute(
            "INSERT INTO kv_store (key, value, kind, updated_at) "
            "VALUES (?, ?, 'json', ?) ON CONFLICT(key) DO UPDATE SET "
            "value = excluded.value, updated_at = excluded.updated_at",
            (_OVERRIDE_KEY, json.dumps({"enabled": bool(enabled)}),
             time.time()),
        )
        # also flip the in-process settings so the current run picks it up
        try:
            from dataclasses import replace
            auto = context.settings.autonomy
            context.settings.autonomy = replace(auto, enabled=bool(enabled))
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("set_autonomy_enabled failed: %s", exc)
        return False


def _kv_override(context: Any) -> bool | None:
    """The durable on/off flag, or None when never set."""
    try:
        import json
        db = getattr(context, "db", None)
        if db is None:
            return None
        row = db.query_one("SELECT value FROM kv_store WHERE key=?",
                           (_OVERRIDE_KEY,))
        if not row:
            return None
        data = json.loads(row.get("value") or "null")
        return bool(data.get("enabled")) if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def autonomy_enabled(context: Any) -> bool:
    """True when the cognitive loop should run on a schedule.

    A durable kv override (``nm autonomy on/off``) wins; otherwise the
    config/env ``autonomy.enabled`` default applies.
    """
    try:
        override = _kv_override(context)
        if override is not None:
            return override
        auto = getattr(context.settings, "autonomy", None)
        return bool(getattr(auto, "enabled", False))
    except Exception:  # noqa: BLE001
        return False


class ModelBudget:
    """The autonomy budget governor (wave 64).

    The cognitive loop counts the model calls it makes (every provider in
    the router chain reports them, mock included) and keeps a durable
    daily ledger in ``kv_store``.  When the owner sets
    ``autonomy.daily_model_calls`` the governor does two things:

      * **stage guard** — before each model-heavy stage the loop asks
        ``allow(stage)``; once the daily cap is exhausted the stage is
        skipped until the day rolls over (the loop itself still ticks —
        local work like the fine-tune pipeline is never blocked);
      * **self-throttling** — at >= 80% pressure the adaptive cadence is
        floored at 2x the base interval, so the loop wakes up less
        often and the budget recovers between heartbeats.

    With a cap of 0 (the default) the governor is a pure accountant: it
    measures and reports every tick, it never restricts.
    """

    _KEY = "autonomy.budget"

    def __init__(self, context: Any) -> None:
        self.context = context

    # ── config ─────────────────────────────────────────────────────────────
    @property
    def cap(self) -> int:
        auto = getattr(self.context.settings, "autonomy", None)
        return max(0, int(getattr(auto, "daily_model_calls", 0) or 0))

    @staticmethod
    def _today(now: float | None = None) -> str:
        ts = now if now is not None else time.time()
        return time.strftime("%Y-%m-%d", time.localtime(ts))

    # ── the durable daily ledger ───────────────────────────────────────────
    def usage(self) -> dict[str, Any]:
        """Today's ledger ``{date, calls, tokens, ticks}`` (rolls over
        automatically at local midnight)."""
        today = self._today()
        try:
            row = self.context.db.query_one(
                "SELECT value FROM kv_store WHERE key=?", (self._KEY,))
            data = json.loads(row.get("value") or "{}") if row else {}
        except Exception:  # noqa: BLE001
            data = {}
        if not isinstance(data, dict) or data.get("date") != today:
            data = {"date": today, "calls": 0, "tokens": 0, "ticks": 0}
        return data

    def _save(self, data: dict[str, Any]) -> None:
        try:
            self.context.db.execute(
                "INSERT INTO kv_store (key, value, kind, updated_at) "
                "VALUES (?,?, 'json', ?) ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value, updated_at = excluded.updated_at",
                (self._KEY, json.dumps(data), time.time()))
        except Exception as exc:  # noqa: BLE001
            _log.debug("budget ledger write failed: %s", exc)

    def record(self, calls: int, tokens: int) -> dict[str, Any]:
        """Add one tick's spend to today's ledger (and persist it)."""
        data = self.usage()
        data["calls"] = int(data.get("calls", 0)) + max(0, int(calls))
        data["tokens"] = int(data.get("tokens", 0)) + max(0, int(tokens))
        data["ticks"] = int(data.get("ticks", 0)) + 1
        self._save(data)
        return data

    # ── governor ───────────────────────────────────────────────────────────
    @property
    def remaining(self) -> int:
        """Calls left today, or -1 when unlimited."""
        cap = self.cap
        if cap <= 0:
            return -1
        return max(0, cap - int(self.usage().get("calls", 0)))

    @property
    def pressure(self) -> float:
        """0.0 (fresh) .. 1.0 (cap hit); always 0.0 when unlimited."""
        cap = self.cap
        if cap <= 0:
            return 0.0
        return min(1.0, int(self.usage().get("calls", 0)) / cap)

    def allow(self, stage: str) -> tuple[bool, str]:
        """May ``stage`` use model calls right now? (reason when no)."""
        if self.cap <= 0:
            return True, ""
        if self.remaining > 0:
            return True, ""
        return False, (f"model budget exhausted "
                       f"({self.cap} calls/day used) — '{stage}' is "
                       f"skipped until the day rolls over")

    # ── predictive reservations (wave 66) ─────────────────────────────────
    # A build step that is ABOUT to spend N model calls (draft -> run ->
    # fix iterations) reserves them BEFORE starting.  If the daily cap
    # cannot cover the reservation, the work is suspended early with an
    # honest report instead of dying mid-build with the budget already
    # half-eaten.  Reservations live in a durable ledger and survive
    # restarts; committing/releasing is idempotent.

    _RES_KEY = "autonomy.budget.reservations"

    def _reservations_raw(self) -> list[dict[str, Any]]:
        try:
            row = self.context.db.query_one(
                "SELECT value FROM kv_store WHERE key=?", (self._RES_KEY,))
            data = json.loads(row.get("value") or "[]") if row else []
        except Exception:  # noqa: BLE001
            data = []
        return data if isinstance(data, list) else []

    def _save_reservations(self, rows: list[dict[str, Any]]) -> None:
        try:
            self.context.db.execute(
                "INSERT INTO kv_store (key, value, kind, updated_at) "
                "VALUES (?,?, 'json', ?) ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value, updated_at = excluded.updated_at",
                (self._RES_KEY, json.dumps(rows), time.time()))
        except Exception as exc:  # noqa: BLE001
            _log.debug("reservation ledger write failed: %s", exc)

    def held(self) -> int:
        """Calls currently held by open reservations."""
        return sum(int(r.get("calls", 0)) for r in self._reservations_raw()
                   if r.get("status") == "held")

    def remaining_available(self) -> int:
        """Calls left today AFTER open reservations; -1 when unlimited."""
        if self.cap <= 0:
            return -1
        return max(0, self.cap - int(self.usage().get("calls", 0))
                   - self.held())

    def affordable(self, calls: int) -> bool:
        """Stateless probe: could ``reserve(calls)`` succeed right now?

        Used by the heartbeat to decide whether a budget-suspended goal
        is ready to resume (after the daily ledger rolls over) WITHOUT
        actually taking a reservation.
        """
        remaining = self.remaining_available()
        return remaining < 0 or remaining >= int(calls or 0)

    def reserve(self, calls: int, *, task: str = "") -> dict[str, Any]:
        """Reserve ``calls`` model calls for an upcoming unit of work.

        Returns ``{"ok": True, "reservation_id", "held"}`` or
        ``{"ok": False, "reason"}`` when the daily cap cannot cover it.
        Unlimited budgets (cap 0) always reserve — the ledger still
        tracks who is holding what, so reports stay honest.
        """
        calls = max(0, int(calls))
        if calls <= 0:
            return {"ok": True, "reservation_id": "", "held": self.held()}
        if self.cap > 0:
            avail = self.remaining_available()
            if calls > avail:
                return {"ok": False, "reservation_id": "",
                        "held": self.held(),
                        "reason": (f"cannot reserve {calls} model calls "
                                   f"({avail} of {self.cap} daily calls "
                                   f"left after other reservations) — "
                                   f"suspended until the day rolls over")}
        rows = self._reservations_raw()
        rid = new_short_id("resv")
        rows.append({"id": rid, "calls": calls, "task": (task or "")[:120],
                     "status": "held", "created_at": time.time()})
        # cap the ledger's size: keep open ones + the 50 most recent closed
        open_rows = [r for r in rows if r.get("status") == "held"]
        closed = [r for r in rows if r.get("status") != "held"][-50:]
        self._save_reservations(open_rows + closed)
        return {"ok": True, "reservation_id": rid, "held": self.held()}

    def _resolve(self, rid: str, status: str) -> bool:
        if not rid:
            return False
        rows = self._reservations_raw()
        for r in rows:
            if r.get("id") == rid and r.get("status") == "held":
                r["status"] = status
                self._save_reservations(rows)
                return True
        return False

    def commit(self, rid: str) -> bool:
        """Unit of work finished — its reservation is spent (idempotent).
        The calls actually made are metered to the daily ledger by the
        caller via :meth:`record`."""
        return self._resolve(rid, "committed")

    def release(self, rid: str) -> bool:
        """Unit of work never ran (or was refused) — give the calls back."""
        return self._resolve(rid, "released")

    def reservations(self, limit: int = 20) -> list[dict[str, Any]]:
        """Open reservations first, then the most recent closed ones."""
        rows = self._reservations_raw()
        open_rows = [r for r in rows if r.get("status") == "held"]
        closed = [r for r in rows if r.get("status") != "held"]
        return (open_rows + closed)[-limit:]

    @staticmethod
    def usage_snapshot(context: Any) -> dict[str, int] | None:
        """Cumulative model usage across the router chain (None = no
        router, so nothing can be metered).  Shared by the cognitive
        loop's tick metering and the build executor's session metering,
        so out-of-tick work counts in the same daily ledger."""
        router = getattr(context, "router", None)
        if router is None:
            return None
        try:
            calls = 0
            tokens = 0
            for name in router.providers():
                provider = router.get(name)
                if provider is None:
                    continue
                stats = getattr(provider, "stats", {}) or {}
                calls += int(stats.get("calls", 0) or 0)
                tokens += (int(stats.get("prompt_tokens", 0) or 0)
                           + int(stats.get("completion_tokens", 0) or 0))
            return {"calls": calls, "tokens": tokens}
        except Exception:  # noqa: BLE001 — metering must never break work
            return None

    def usage_delta(self, snap: dict[str, int] | None) -> dict[str, int]:
        if not snap:
            return {"calls": 0, "tokens": 0}
        cur = self.usage_snapshot(self.context)
        if cur is None:
            return {"calls": 0, "tokens": 0}
        return {"calls": max(0, cur["calls"] - int(snap.get("calls", 0))),
                "tokens": max(0, cur["tokens"] - int(snap.get("tokens", 0)))}

    def meter_session(self, snap: dict[str, int] | None) -> dict[str, int]:
        """Meter one completed unit of work (e.g. a build session) into
        the daily ledger; returns the delta it added."""
        delta = self.usage_delta(snap)
        if delta["calls"] or delta["tokens"]:
            self.record(delta["calls"], delta["tokens"])
        return delta

    def report(self) -> dict[str, Any]:
        data = self.usage()
        cap = self.cap
        used = int(data.get("calls", 0))
        held = self.held()
        return {
            "date": data.get("date"), "cap": cap,
            "unlimited": cap <= 0,
            "used": used, "tokens": int(data.get("tokens", 0)),
            "ticks": int(data.get("ticks", 0)),
            "reserved": held,
            "remaining": -1 if cap <= 0 else max(0, cap - used - held),
            "pressure": round(self.pressure, 4),
            "throttled": (cap > 0) and self.pressure >= 0.8,
        }


class CognitiveLoop:
    """One heartbeat through goals, improvement, and personal training."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.settings = context.settings

    # ── the tick ───────────────────────────────────────────────────────────
    def tick(self, *, executor: Any = None,
             max_goals: int = 25) -> dict[str, Any]:
        """Run one bounded heartbeat. Returns a summary of every stage,
        including the model-call cost of this tick and the daily budget
        state (autonomy budget governor, wave 64)."""
        started = time.monotonic()
        summary: dict[str, Any] = {"stages": {}, "seconds": 0.0}
        auto = getattr(self.settings, "autonomy", None)
        do_goals = bool(getattr(auto, "tick_goals", True))
        do_improve = bool(getattr(auto, "tick_improvement", True))
        do_train = bool(getattr(auto, "tick_train", True))
        cap = int(getattr(auto, "max_project_steps_per_tick", 3) or 0)

        budget = ModelBudget(self.context)
        total = {"calls": 0, "tokens": 0}

        def _run(name: str, fn: Any, *, guard: bool) -> dict[str, Any]:
            """Run one stage, metering its model calls against the budget."""
            if guard:
                allowed, why = budget.allow(name)
                if not allowed:
                    return {"skipped": why}
            snap = self._usage_snapshot()
            out = fn()
            if not isinstance(out, dict):
                out = {"result": out}
            delta = self._usage_delta(snap)
            total["calls"] += delta["calls"]
            total["tokens"] += delta["tokens"]
            if delta["calls"] or delta["tokens"]:
                out = dict(out)
                out["model_calls"] = delta["calls"]
                out["model_tokens"] = delta["tokens"]
            return out

        # 1. goals (drive linked projects through them) — model-heavy
        if do_goals:
            try:
                summary["stages"]["goals"] = _run(
                    "goals", lambda: self._tick_goals(
                        executor=executor, max_goals=max_goals), guard=True)
            except Exception as exc:  # noqa: BLE001
                summary["stages"]["goals"] = {"error": str(exc)}
        else:
            summary["stages"]["goals"] = {"skipped": "tick_goals off"}

        # 2. improvement cycle — model-heavy
        if do_improve:
            try:
                summary["stages"]["improvement"] = _run(
                    "improvement", self._tick_improvement, guard=True)
            except Exception as exc:  # noqa: BLE001
                summary["stages"]["improvement"] = {"error": str(exc)}
        else:
            summary["stages"]["improvement"] = {"skipped": "tick_improvement off"}

        # 3. personal-model fine-tune (policy-gated) — local work: it is
        # metered, but the budget never blocks the owner's own training
        if do_train:
            try:
                summary["stages"]["training"] = _run(
                    "training", lambda: self._tick_train(cap=cap), guard=False)
            except Exception as exc:  # noqa: BLE001
                summary["stages"]["training"] = {"error": str(exc)}
        else:
            summary["stages"]["training"] = {"skipped": "tick_train off"}

        # 4. watchers sweep — model-free: check due watchers, route alerts.
        # Runs on the autonomy tick as well as the scheduler's sweeper job;
        # both are idempotent (per-watcher due timestamps), so overlap is
        # harmless. Not budget-guarded: no model calls involved.
        do_watch = bool(getattr(auto, "tick_watchers", True))
        if do_watch:
            try:
                summary["stages"]["watchers"] = _run(
                    "watchers", self._tick_watchers, guard=False)
            except Exception as exc:  # noqa: BLE001
                summary["stages"]["watchers"] = {"error": str(exc)}
        else:
            summary["stages"]["watchers"] = {"skipped": "tick_watchers off"}

        summary["seconds"] = round(time.monotonic() - started, 3)
        # settle the tick's spend in the daily ledger, then report it
        budget.record(total["calls"], total["tokens"])
        summary["budget"] = {"calls_this_tick": total["calls"],
                             "tokens_this_tick": total["tokens"],
                             **budget.report()}
        self._log_tick(summary)
        self._update_adaptive_cadence(summary)
        _log.info("cognitive loop tick: %s",
                  {k: v for k, v in summary["stages"].items()})
        return summary

    # ── model-usage metering (wave 64) ─────────────────────────────────────
    def _tick_watchers(self) -> dict[str, Any]:
        """One watchers sweep inside the autonomy tick (Prompt 03)."""
        from .watchers import WatcherAgent

        agent = WatcherAgent(self.context)
        return agent.tick()

    def _usage_snapshot(self) -> Optional[dict[str, int]]:
        """Cumulative model usage across the router chain (shared with the
        budget governor so tick and out-of-tick work meter alike)."""
        return ModelBudget.usage_snapshot(self.context)

    def _usage_delta(self, snap: Optional[dict[str, int]]) -> dict[str, int]:
        return ModelBudget(self.context).usage_delta(snap)

    def _ev_order(self, goal_ids: list[str]) -> dict[str, float]:
        """Risk-adjusted expected value per goal id (wave 64) — 0.0 when
        scoring is unavailable, so ordering degrades to priority."""
        if not goal_ids:
            return {}
        try:
            from .mission import MissionControl

            scores = MissionControl(self.context).ev_scores(goal_ids)
            return {gid: float(v.get("expected_value", 0.0))
                    for gid, v in scores.items()}
        except Exception as exc:  # noqa: BLE001
            _log.debug("ev ordering failed: %s", exc)
            return {}

    # ── cost-aware cadence (wave 63) ──────────────────────────────────────
    def effective_interval(self) -> float:
        """The interval the scheduler is actually using: the persisted
        adaptive value when the dial is on, else the configured base."""
        base = float(getattr(self.settings.autonomy, "interval_hours", 6.0)
                     or 6.0)
        if not getattr(self.settings.autonomy, "adaptive_cadence", True):
            return base
        try:
            import json as _json

            row = self.context.db.query_one(
                "SELECT value FROM kv_store "
                "WHERE key='autonomy.adaptive_interval'")
            if row:
                data = _json.loads(row.get("value") or "null")
                val = float((data or {}).get("interval_hours", base))
                return min(48.0, max(0.25, val))
        except Exception:  # noqa: BLE001
            pass
        return base

    def adaptive_interval(self, *, window: int = 10) -> float:
        """The next heartbeat interval, derived from the loop's OWN telemetry.

        The loop watches the last ``window`` heartbeats:
        * **busy** (>= half of them got work done — advanced or healed a
          goal, ran an improvement or training stage) → tighten to a
          quarter of the base (floor 15 min): more work means more ticks;
        * **idle** (none of them did) → relax to four times the base
          (cap 48 h): an empty loop is just cost;
        * mixed → the configured base.
        Always clamped to [15 min, 48 h].
        """
        base = float(getattr(self.settings.autonomy, "interval_hours", 6.0)
                     or 6.0)
        if not getattr(self.settings.autonomy, "adaptive_cadence", True):
            return base
        try:
            rows = self.context.db.query(
                "SELECT stages FROM cognition_log ORDER BY ts DESC LIMIT ?",
                (window,))
        except Exception:  # noqa: BLE001
            return base
        if not rows:
            return base
        import json as _json

        busy = 0
        for r in rows:
            try:
                stages = _json.loads(r.get("stages") or "{}")
            except (ValueError, TypeError):
                continue
            active = False
            g = stages.get("goals") or {}
            if isinstance(g, dict) and (g.get("advanced") or g.get("healed")
                                        or g.get("resumed")):
                active = True
            for name in ("improvement", "training"):
                st = stages.get(name) or {}
                if isinstance(st, dict) and st.get("ran"):
                    active = True
            if active:
                busy += 1
        ratio = busy / len(rows)
        if ratio >= 0.5:
            return max(0.25, min(base, base * 0.25))
        if busy == 0:
            # wave 67: an idle tick with WORK STILL PENDING (active or
            # budget-suspended goals with pending steps) is NOT idle —
            # it's waiting (on a rollover, on a dependency).  Sleeping
            # up to 48 h there is how a live portfolio looked "almost
            # always idle".  Keep the configured pace; only a genuinely
            # empty portfolio gets the 4x relax.
            if self._portfolio_has_pending_work():
                return base
            return min(48.0, max(base, base * 4))
        return base

    def _portfolio_has_pending_work(self) -> bool:
        """True when ANY goal (active or paused — including
        budget-suspended) still has pending steps.  Best-effort."""
        try:
            row = self.context.db.query_one(
                "SELECT COUNT(*) AS c FROM goal_steps s "
                "JOIN goals g ON g.id = s.goal_id "
                "WHERE s.status = 'pending' "
                "AND g.status NOT IN ('done', 'abandoned')")
            if row is None:
                return False
            return int(row.get("c", 0) or 0) > 0
        except Exception:  # noqa: BLE001 — cadence comfort, never fatal
            return False

    def _update_adaptive_cadence(self, summary: dict[str, Any]) -> None:
        """Persist the derived cadence and, when the scheduler is live,
        apply it without a restart. Best-effort: cadence is a comfort,
        never a failure mode."""
        try:
            if not getattr(self.settings.autonomy, "adaptive_cadence", True):
                return
            hours = self.adaptive_interval()
            import json as _json

            # budget pressure (wave 64): when the daily model budget is
            # running low, the heartbeat slows to let it recover — the
            # loop self-throttles instead of burning the owner's calls
            budget_throttled = False
            try:
                b = ModelBudget(self.context)
                if b.cap > 0 and b.pressure >= 0.8:
                    base = float(getattr(self.settings.autonomy,
                                         "interval_hours", 6.0) or 6.0)
                    hours = max(hours, min(48.0, base * 2.0))
                    budget_throttled = True
            except Exception:  # noqa: BLE001
                pass
            if budget_throttled and isinstance(summary.get("budget"), dict):
                summary["budget"]["throttled"] = True

            self.context.db.execute(
                "INSERT INTO kv_store (key, value, kind, updated_at) "
                "VALUES (?, ?, 'json', ?) ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value, updated_at = excluded.updated_at",
                ("autonomy.adaptive_interval",
                 _json.dumps({"interval_hours": hours, "ts": time.time()}),
                 time.time()))
            scheduler = getattr(self.context, "extras", {}).get("scheduler")
            if scheduler is not None and hasattr(scheduler, "set_interval"):
                try:
                    scheduler.set_interval("cognitive loop",
                                           int(hours * 3600))
                except Exception:  # noqa: BLE001
                    pass
            summary["next_interval_hours"] = round(hours, 3)
        except Exception as exc:  # noqa: BLE001
            _log.debug("adaptive cadence failed: %s", exc)

    # ── telemetry (wave 62) ────────────────────────────────────────────────
    def _log_tick(self, summary: dict[str, Any]) -> None:
        """Journal one heartbeat into cognition_log (best-effort)."""
        try:
            import json as _json

            from ..core.ids import new_short_id

            stages = summary.get("stages", {})
            note_bits = []
            goals = stages.get("goals") or {}
            if isinstance(goals, dict):
                if goals.get("advanced"):
                    note_bits.append(f"advanced {len(goals['advanced'])} goal(s)")
                if goals.get("healed"):
                    note_bits.append(f"healed {len(goals['healed'])} goal(s)")
            for name in ("improvement", "training"):
                st = stages.get(name) or {}
                if isinstance(st, dict) and st.get("error"):
                    note_bits.append(f"{name} error")
                elif isinstance(st, dict) and st.get("skipped"):
                    note_bits.append(f"{name} skipped")
            self.context.db.execute(
                "INSERT INTO cognition_log (id, ts, stages, seconds, note) "
                "VALUES (?,?,?,?,?)",
                (new_short_id("cog"), time.time(),
                 _json.dumps(stages, default=str)[:8000],
                 float(summary.get("seconds", 0.0)),
                 "; ".join(note_bits)[:500]))
        except Exception as exc:  # noqa: BLE001 — telemetry never kills a tick
            _log.debug("cognitive tick logging failed: %s", exc)

    def recent(self, *, limit: int = 10) -> list[dict[str, Any]]:
        """The last heartbeats, newest first."""
        import json as _json

        try:
            rows = self.context.db.query(
                "SELECT * FROM cognition_log ORDER BY ts DESC LIMIT ?",
                (limit,))
        except Exception:  # noqa: BLE001
            return []
        out = []
        for r in rows:
            try:
                stages = _json.loads(r.get("stages") or "{}")
            except (ValueError, TypeError):
                stages = {}
            out.append({
                "id": r["id"], "ts": float(r.get("ts", 0)),
                "seconds": float(r.get("seconds", 0.0)),
                "note": r.get("note", ""), "stages": stages,
            })
        return out

    def report(self, *, limit: int = 10) -> dict[str, Any]:
        """Autonomy telemetry: per-stage outcomes across all heartbeats."""
        import json as _json

        try:
            rows = self.context.db.query(
                "SELECT * FROM cognition_log ORDER BY ts ASC")
        except Exception:  # noqa: BLE001
            rows = []
        stage_counts: dict[str, dict[str, int]] = {}
        total_seconds = 0.0
        for r in rows:
            try:
                stages = _json.loads(r.get("stages") or "{}")
            except (ValueError, TypeError):
                stages = {}
            total_seconds += float(r.get("seconds", 0.0) or 0.0)
            for name, st in stages.items():
                bucket = stage_counts.setdefault(name, {})
                if isinstance(st, dict):
                    if st.get("error"):
                        key = "error"
                    elif st.get("skipped"):
                        key = "skipped"
                    else:
                        key = "ran"
                else:
                    key = "ran"
                bucket[key] = bucket.get(key, 0) + 1
        return {
            "ticks": len(rows),
            "total_seconds": round(total_seconds, 2),
            "avg_seconds": round(total_seconds / len(rows), 3) if rows else 0.0,
            "stages": stage_counts,
            "recent": self.recent(limit=limit),
        }

    # ── stages ─────────────────────────────────────────────────────────────
    def _tick_goals(self, *, executor: Any = None,
                    max_goals: int = 25) -> dict[str, Any]:
        from .goals import GoalSystem

        gs = GoalSystem(self.context)
        # mission control (wave 62) + portfolio risk scoring (wave 64):
        # work where the EXPECTED VALUE is highest — priority tempered by
        # dependency depth, heal history, and remaining size (degrades to
        # plain priority order when scoring is unavailable)
        active = gs.list(status="active", limit=max_goals)
        ev = self._ev_order([g.id for g in active])
        active.sort(key=lambda g: (-(ev.get(g.id, 0.0)), -g.priority,
                                   g.created_at))
        advanced: list[str] = []
        driven: list[str] = []
        for goal in active:
            if not any(s.status == "pending" for s in goal.steps):
                if any(s.status == "blocked" for s in goal.steps):
                    try:
                        gs.adapt(goal.id)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            if not gs.dependencies_met(goal):
                continue  # still waiting on an unfinished dependency
            before = goal.project_id
            gs.advance(goal.id, executor=executor)
            advanced.append(goal.id)
            if before:
                driven.append(goal.id)
        resumed = self._tick_resume_suspended(executor=executor)
        return {"active": len(active), "advanced": advanced,
                "project_driven": driven, "resumed": resumed,
                "healed": self._tick_heal(),
                "by_status": gs.status().get("by_status", {})}

    def _tick_resume_suspended(self, *, executor: Any = None) -> list[str]:
        """Wave 67: revive budget-suspended goals whose day has rolled
        over.

        A build that couldn't afford its reservation pauses the project
        (and the goal) instead of failing — but nobody was picking it up
        again.  This pass walks the paused goals, and for each one whose
        project reports a budget suspension, probes the budget: affordable
        now -> the project goes running and the goal takes its next step
        right here.  Not affordable -> leave it paused (the probe is
        stateless, so nothing leaks).
        """
        from .projects import ProjectManager

        resumed: list[str] = []
        try:
            rows = self.context.db.query(
                "SELECT id, project_id FROM goals "
                "WHERE status = 'paused' AND project_id != '' "
                "ORDER BY priority DESC, created_at ASC LIMIT 10")
            if not rows:
                return resumed
            mgr = ProjectManager(self.context)
            from .goals import GoalSystem

            gs = GoalSystem(self.context)
            for r in rows:
                goal_id = str(r.get("id") or "")
                project_id = str(r.get("project_id") or "")
                try:
                    project = mgr._load(project_id)
                    if project is None or project.status != "paused":
                        continue
                    if not str(project.report or "").startswith(
                            "Suspended on model budget"):
                        continue
                    cost = 3 if project.task_kind == "build" else 1
                    if not ModelBudget(self.context).affordable(cost):
                        continue  # still waiting on the rollover
                    # the goal un-pauses with its project
                    self.context.db.execute(
                        "UPDATE goals SET status='active', updated_at=? "
                        "WHERE goal_id=?", (time.time(), goal_id))
                    # advance() self-resumes the project and runs the step
                    gs.advance(goal_id, executor=executor)
                    resumed.append(goal_id)
                except Exception:  # noqa: BLE001 — one goal must not
                    _log.debug("resume of %s failed", goal_id, exc_info=True)
                    continue
        except Exception as exc:  # noqa: BLE001 — resuming never kills a tick
            _log.warning("resume-suspended stage failed: %s", exc)
        return resumed

    def _tick_heal(self) -> list[str]:
        """Self-healing (wave 62): revive paused goals whose linked project
        failed, rewording the exhausted steps with the failure analyzer's
        learned lessons.  Bounded by ``autonomy.max_project_heals`` per
        goal — after that the goal stays paused for the owner."""
        from .projects import ProjectManager

        max_heals = int(getattr(self.settings.autonomy, "max_project_heals",
                                3) or 0)
        if max_heals <= 0:
            return []
        healed: list[str] = []
        try:
            rows = self.context.db.query(
                "SELECT * FROM goals WHERE status='paused' AND project_id != '' "
                "ORDER BY priority DESC, created_at ASC LIMIT 10")
            mgr = ProjectManager(self.context)
            for r in rows:
                if int(r.get("heals", 0) or 0) >= max_heals:
                    continue
                project = mgr._load(str(r.get("project_id") or ""))
                if project is None or project.status != "failed":
                    continue
                result = mgr.heal(project.id)
                if result.get("ok"):
                    self.context.db.execute(
                        "UPDATE goals SET heals = heals + 1, updated_at=? "
                        "WHERE id=?", (time.time(), r["id"]))
                    healed.append(r["id"])
        except Exception as exc:  # noqa: BLE001 — healing never kills a tick
            _log.warning("self-heal stage failed: %s", exc)
        return healed

    def _tick_improvement(self) -> dict[str, Any]:
        from .improvement import ImprovementLoop

        loop = ImprovementLoop(self.context)
        mode = str(getattr(loop.settings, "mode", "off") or "off")
        if mode == "off":
            return {"skipped": "improvement mode off"}
        rec = loop.run_cycle()
        return {"action": rec.action, "status": rec.status,
                "dimension": rec.dimension}

    def _tick_train(self, *, cap: int = 0) -> dict[str, Any]:
        from ..self_improvement import SelfImprovementJob

        job = SelfImprovementJob(self.context)
        status = job.status()
        decision = status.get("decision", {})
        if not decision.get("data_available"):
            return {"skipped": "no training data", "decision": decision}
        if not decision.get("should_retrain"):
            return {"skipped": "policy not due",
                    "reasons": decision.get("reasons", [])}
        # due: run the durable pipeline (it re-checks the gate on promote)
        result = job.run(force=True)
        return {"status": result.status, "run_id": result.run_id,
                "dataset_id": result.dataset_id,
                "promoted": result.promoted, "skipped": result.skipped,
                "reason": result.reason}

    # ── status ─────────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        from .goals import GoalSystem
        from .improvement import ImprovementLoop

        auto = getattr(self.settings, "autonomy", None)
        gs = GoalSystem(self.context)
        loop = ImprovementLoop(self.context)
        try:
            train_status = SelfImprovementStatus(self.context)
        except Exception:  # noqa: BLE001
            train_status = {}
        return {
            "enabled": autonomy_enabled(self.context),
            "configured": bool(getattr(auto, "enabled", False)),
            "interval_hours": self.effective_interval(),
            "budget": ModelBudget(self.context).report(),
            "goals": gs.status(),
            "improvement": {
                "mode": str(getattr(loop.settings, "mode", "off")),
                "pending_approvals": len(loop.pending_approvals()),
                "recent": [r.to_dict() for r in loop.history(limit=3)],
            },
            "training": train_status,
        }


def SelfImprovementStatus(context: Any) -> dict[str, Any]:
    """Lightweight training-pipeline status (policy + last run)."""
    from ..self_improvement import SelfImprovementJob

    job = SelfImprovementJob(context)
    status = job.status()
    decision = status.get("decision", {})
    return {
        "should_retrain": bool(decision.get("should_retrain")),
        "data_available": bool(decision.get("data_available")),
        "reasons": decision.get("reasons", []),
        "runs": status.get("runs", {}),
    }


# ── tool registration ──────────────────────────────────────────────────────────

def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "autonomy",
        description=("The autonomous cascade: one heartbeat that ticks goals "
                     "(driving linked projects, self-healing failed ones), a "
                     "closed-loop improvement cycle, and the personal-model "
                     "fine-tune (when due). action=tick | status | report | "
                     "budget (the daily model-call budget the loop meters "
                     "against and self-throttles on)."),
        capability="model.call",
        parameters={
            "action": "str — tick | status | report | budget",
            "max_goals": "str — how many active goals to advance",
            "limit": "str — how many recent ticks in a report",
        },
    )
    def autonomy(action: str = "status", *, max_goals: str = "5",
                 limit: str = "10") -> dict[str, Any]:
        loop = CognitiveLoop(context)
        action = (action or "status").strip().lower()
        if action == "tick":
            try:
                n = int(max_goals or 5)
            except ValueError:
                n = 5
            return loop.tick(max_goals=n)
        if action == "report":
            try:
                n = int(limit or 10)
            except ValueError:
                n = 10
            return loop.report(limit=n)
        if action == "budget":
            return ModelBudget(context).report()
        if action == "status":
            return loop.status()
        return {"ok": False, "error": f"unknown action: {action}"}
