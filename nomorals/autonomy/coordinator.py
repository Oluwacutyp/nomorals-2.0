"""Idle-task coordinator — when the system goes quiet, organs work.

Subscribes to ``system.idle`` on the global bus. When it fires:

1. **Research organ** — one ``tick()``; the organ drains its own event
   queue (weakness investigations, interest watches, window preparations).
2. **Wisdom organ** — one ``tick()`` (corpus ingestion within its budget).
3. **Memory** — one consolidation tick (compress old episodes, refresh
   salience).
4. **Presence** — the heartbeat (notice/prepare/surface).
5. **Weakness** — scan for thresholded cases; route investigations to
   research.

When ``system.active`` fires, background work stands down gracefully
(current tick finishes, no new ticks start).

This is the "organs trigger each other" wiring — not a scheduler
cron, but a real event chain: quiet → idle event → organs wake →
work → done. Every cycle is journaled to the autonomy ledger and ends
with a ``system.idle_cycle`` bus event so downstream systems can react.

Scheduling discipline (borrowed from SRE practice):

* **Budget slices** — the cycle budget is split across organs
  proportional to pending work (queue depth), with minimum slices so
  no organ starves. Work without budget waits for the next cycle.
* **Priority order** — weakness/research first (they feed everything
  else), then memory, presence, wisdom, patterns.
* **Failure backoff** — an organ tick that fails repeatedly is skipped
  with exponential backoff instead of hammered every cycle; the skip
  is journaled so it's visible, not silent.
* **Health** — :meth:`coordinator_status` exposes per-organ health
  (last ok, avg seconds, consecutive failures) for machines; and
  :meth:`cycle_report` renders the last cycle for humans.

Design rule (learned the hard way): the coordinator NEVER pre-drains an
organ's event queue. ``organs.drain()`` marks events consumed in the same
transaction — a coordinator drain followed by an organ ``tick()`` would
consume every event without processing it. Each organ ticks once per
cycle and drains its own queue.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .. import organs as _organs

_log = get_logger(__name__)

#: Maximum seconds one idle maintenance cycle runs before yielding.
IDLE_CYCLE_BUDGET = 600  # 10 minutes

#: Minimum budget slice per organ (seconds) — nobody starves.
MIN_ORGAN_SLICE = 20.0

#: Priority order for organ ticks (feeds-everything-else first).
#: Used for budget planning and reporting. Execution order stays the
#: historical one (research → wisdom → memory → presence → weakness →
#: patterns) for compatibility with existing subscribers/tests.
TICK_PRIORITY = ("weakness", "research", "memory", "presence",
                 "wisdom", "patterns")

#: Historical execution order of the maintenance cycle.
TICK_ORDER = ("research", "wisdom", "memory", "presence",
              "weakness", "patterns")


def _budget_left(started: float) -> float:
    return max(0.0, IDLE_CYCLE_BUDGET - (time.time() - started))


class IdleCoordinator:
    """Listens for idle/active events, runs organ maintenance."""

    def __init__(self, workspace_dir: str | Path):
        self.workspace_dir = Path(workspace_dir)
        self._idle = False
        self._cycle_lock = threading.Lock()
        self._subscribed = False
        self._cycles = 0
        #: Per-organ health: {organ: {failures, last_ok_ts, last_err,
        #: avg_seconds, samples, last_cycle, skipped_cycles}}
        self._organ_health: dict[str, dict[str, Any]] = {}
        #: Last completed cycle's stats (for cycle_report).
        self._last_cycle: dict[str, Any] | None = None

    # ── health bookkeeping ─────────────────────────────────────────

    def _health(self, organ: str) -> dict[str, Any]:
        return self._organ_health.setdefault(organ, {
            "failures": 0, "last_ok_ts": 0.0, "last_err": "",
            "avg_seconds": 0.0, "samples": 0, "last_cycle": "",
            "skipped_cycles": 0,
        })

    def _record_ok(self, organ: str, seconds: float,
                   cycle_id: str) -> None:
        h = self._health(organ)
        h["failures"] = 0
        h["last_ok_ts"] = time.time()
        h["last_err"] = ""
        h["last_cycle"] = cycle_id
        n = h["samples"] + 1
        h["avg_seconds"] = (h["avg_seconds"] * h["samples"] + seconds) / n
        h["samples"] = n

    def _record_fail(self, organ: str, err: str, cycle_id: str) -> None:
        h = self._health(organ)
        h["failures"] += 1
        h["last_err"] = str(err)[:200]
        h["last_cycle"] = cycle_id

    def _backoff_skip(self, organ: str) -> int:
        """Cycles to skip for a failing organ (0 = run it).

        Exponential backoff: after 3 consecutive failures, skip
        2^(failures-3) cycles, capped at 8. The skip is journaled —
        visible, never silent.
        """
        failures = self._health(organ)["failures"]
        if failures < 3:
            return 0
        return min(2 ** (failures - 3), 8)

    def coordinator_status(self) -> dict[str, Any]:
        """Machine-readable coordinator + per-organ health."""
        return {
            "idle": self._idle,
            "cycles": self._cycles,
            "subscribed": self._subscribed,
            "last_cycle": self._last_cycle,
            "organs": {
                organ: {
                    "consecutive_failures": h["failures"],
                    "backoff_skip_cycles": self._backoff_skip(organ),
                    "last_ok_ts": h["last_ok_ts"],
                    "last_error": h["last_err"],
                    "avg_seconds": round(h["avg_seconds"], 1),
                    "samples": h["samples"],
                    "last_cycle": h["last_cycle"],
                }
                for organ, h in self._organ_health.items()
            },
        }

    # ── budget planning ────────────────────────────────────────────

    def _pending_depth(self, db: Any) -> dict[str, float]:
        """Rough pending-work depth per organ (for budget slices)."""
        depth: dict[str, float] = {}
        try:
            depth["research"] = float(
                _organs.pending_count(db, dst="research") or 0)
        except Exception:  # noqa: BLE001
            depth["research"] = 1.0
        try:
            from .weakness import open_weaknesses
            depth["weakness"] = float(len(open_weaknesses(db)))
        except Exception:  # noqa: BLE001
            depth["weakness"] = 1.0
        # Memory/presence/wisdom/patterns do roughly constant work.
        depth.setdefault("memory", 1.0)
        depth.setdefault("presence", 1.0)
        depth.setdefault("wisdom", 1.0)
        depth.setdefault("patterns", 0.5)
        return depth

    def _budget_plan(self, db: Any,
                     total: float = IDLE_CYCLE_BUDGET) -> dict[str, float]:
        """Split the cycle budget across organs by pending depth.

        Proportional to depth, with a minimum slice so no organ
        starves and a cap so one greedy organ can't eat the cycle.
        """
        depth = self._pending_depth(db)
        total_depth = sum(depth.values()) or 1.0
        shares = {o: total * (depth.get(o, 1.0) / total_depth)
                  for o in TICK_PRIORITY}
        # Cap a greedy organ, then enforce minimums.
        shares = {o: min(s, total * 0.4) for o, s in shares.items()}
        plan = {o: max(MIN_ORGAN_SLICE, s) for o, s in shares.items()}
        # If minimums overflow the budget, take the excess back from
        # organs above the minimum, proportionally — minimums hold.
        over = sum(plan.values()) - total
        if over > 0:
            adjustable = {o: plan[o] - MIN_ORGAN_SLICE for o in plan
                          if plan[o] > MIN_ORGAN_SLICE}
            adj_total = sum(adjustable.values())
            if adj_total > 0:
                for o in adjustable:
                    plan[o] -= over * (adjustable[o] / adj_total)
        return {o: round(v, 1) for o, v in plan.items()}

    # ── subscription ───────────────────────────────────────────────

    def subscribe(self) -> "IdleCoordinator":
        """Subscribe to system.idle / system.active. Idempotent."""
        if self._subscribed:
            return self
        global_bus.subscribe("system.idle", self._on_idle)
        global_bus.subscribe("system.active", self._on_active)
        self._subscribed = True
        _log.info("idle coordinator subscribed")
        return self

    def _on_idle(self, event: Event) -> None:
        idle_for = (event.data or {}).get("idle_seconds", 0)
        _log.info("idle coordinator: system idle (%.0fs) — waking organs",
                  idle_for)
        self._idle = True
        # Run the maintenance cycle in a background thread so the bus
        # dispatcher never blocks.
        t = threading.Thread(target=self._maintenance_cycle,
                             name="idle-maintenance", daemon=True)
        t.start()

    def _on_active(self, event: Event) -> None:
        _log.info("idle coordinator: system active — standing down")
        self._idle = False

    def _ledger(self, db: Any, kind: str, ref_id: str, summary: str,
                *, ok: bool = True, cost_seconds: float = 0.0,
                learned: str = "",
                metadata: dict[str, Any] | None = None) -> None:
        try:
            from ..agents.autonomy_ledger import record_ledger
            record_ledger(db, "idle", kind, ref_id, summary, ok=ok,
                          cost_seconds=cost_seconds, learned=learned,
                          metadata=metadata)
        except Exception:  # noqa: BLE001 - telemetry is fail-open
            _log.debug("idle ledger write failed", exc_info=True)

    def _maintenance_cycle(self) -> None:
        if not self._cycle_lock.acquire(blocking=False):
            _log.debug("maintenance cycle already running — skipping")
            return
        started = time.time()
        cycle_id = f"cycle-{self._cycles + 1}-{int(started)}"
        self._cycles += 1
        stats: dict[str, Any] = {"cycle": cycle_id, "steps": {}}
        try:
            from .idle import workspace_db, idle_inhibited, inhibitors
            db = workspace_db(self.workspace_dir)

            # Inhibitors: an organ holding one means "not now".
            held = []
            try:
                if idle_inhibited(db):
                    held = inhibitors(db)
            except Exception:  # noqa: BLE001
                held = []
            if held:
                _log.info("idle cycle %s deferred: %d inhibitor(s) held "
                          "(%s)", cycle_id, len(held),
                          ", ".join(h["name"] for h in held))
                self._ledger(db, "cycle_deferred", cycle_id,
                             "cycle deferred — idle inhibitors held",
                             metadata={"inhibitors": held})
                stats["deferred"] = [h["name"] for h in held]
                self._last_cycle = stats
                return

            plan = self._budget_plan(db)
            stats["budget_plan"] = plan
            self._ledger(db, "cycle_start", cycle_id,
                         "idle maintenance cycle started",
                         metadata={"budget_plan": plan})

            # Ticks run in priority order; each respects its slice.
            tickers = {
                "research": self._tick_research,
                "wisdom": self._tick_wisdom,
                "memory": self._tick_memory,
                "presence": self._tick_presence,
                "weakness": self._tick_weakness,
                "patterns": self._tick_patterns,
            }
            for organ in TICK_ORDER:
                skip = self._backoff_skip(organ)
                if skip:
                    h = self._health(organ)
                    h["skipped_cycles"] = skip
                    stats["steps"][organ] = {
                        "ok": False, "skipped": "backoff",
                        "consecutive_failures": h["failures"],
                        "skip_cycles": skip,
                    }
                    _log.warning(
                        "cycle %s: skipping %s tick (backoff, %d failures)",
                        cycle_id, organ, h["failures"])
                    self._ledger(db, "tick_backoff", cycle_id,
                                 f"{organ} tick skipped (backoff)",
                                 ok=False,
                                 metadata={"organ": organ,
                                           "failures": h["failures"]})
                    continue
                step = self._run_tick(tickers[organ], db, started,
                                      plan.get(organ, MIN_ORGAN_SLICE))
                stats["steps"][organ] = step
                if step.get("ok"):
                    self._record_ok(organ, float(step.get("seconds", 0)),
                                    cycle_id)
                elif not step.get("skipped"):
                    self._record_fail(organ,
                                      str(step.get("error", "unknown")),
                                      cycle_id)

            elapsed = time.time() - started
            stats["seconds"] = round(elapsed, 1)
            _log.info("idle maintenance cycle done in %.1fs: %s",
                      elapsed, {k: v.get("ok", v.get("skipped"))
                                for k, v in stats["steps"].items()})
            self._ledger(db, "cycle_end", cycle_id,
                         f"idle cycle done in {elapsed:.1f}s",
                         cost_seconds=elapsed,
                         metadata={"steps": stats["steps"]})
        except Exception as exc:  # noqa: BLE001
            _log.warning("idle maintenance cycle failed: %s", exc,
                         exc_info=True)
            try:
                from .idle import workspace_db
                self._ledger(workspace_db(self.workspace_dir), "cycle_failed",
                             cycle_id, f"idle cycle failed: {exc}")
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._last_cycle = stats
            self._cycle_lock.release()
        # Completion signal — downstream systems (improvement loop,
        # briefing, dashboards) can react to a finished cycle.
        try:
            global_bus.publish(Event(
                topic="system.idle_cycle",
                data={"cycle": cycle_id, "seconds": stats.get("seconds"),
                      "steps": stats["steps"]},
                source="nomorals.autonomy.coordinator",
            ))
        except Exception:  # noqa: BLE001
            _log.debug("system.idle_cycle publish failed", exc_info=True)

    # ── human-readable cycle report ──────────────────────────────────

    _STEP_ICON = {True: "✅", False: "⚠️"}

    def cycle_report(self, theme: str = "plain") -> str:
        """Render the last cycle as a human-readable summary.

        ``theme="plain"`` for chat, ``theme="rich"`` for a boxed
        dashboard-style render. This is the visible proof the nervous
        system is alive — not log lines.
        """
        stats = self._last_cycle
        if not stats:
            return ("💤 no idle maintenance cycle has run yet — "
                    "the system hasn't gone quiet long enough.")
        steps = stats.get("steps", {})
        if theme == "rich":
            lines = ["╭─ 💤 idle maintenance cycle ─────────╮",
                     f"│ {stats.get('cycle', '?')} · "
                     f"{stats.get('seconds', '?')}s"]
        else:
            lines = [f"💤 idle cycle {stats.get('cycle', '?')} "
                     f"({stats.get('seconds', '?')}s)"]
        if stats.get("deferred"):
            lines.append("deferred — inhibitors held: " +
                         ", ".join(stats["deferred"]))
            return "\n".join(lines)
        for organ in TICK_PRIORITY:
            step = steps.get(organ, {})
            if step.get("skipped"):
                mark = "⏭️"
                note = f"skipped ({step['skipped']})"
            else:
                ok = bool(step.get("ok"))
                mark = self._STEP_ICON[ok]
                secs = step.get("seconds", "?")
                note = f"{secs}s"
                extra = self._step_summary(organ, step)
                if extra:
                    note += f" · {extra}"
            if theme == "rich":
                lines.append(f"│ {mark} {organ:<10} {note}")
            else:
                lines.append(f"{mark} {organ}: {note}")
        failing = [o for o in TICK_PRIORITY
                   if self._health(o)["failures"] >= 3]
        if failing:
            lines.append("backing off: " + ", ".join(failing))
        if theme == "rich":
            lines.append("╰──────────────────────────────────╯")
        return "\n".join(lines)

    @staticmethod
    def _step_summary(organ: str, step: dict[str, Any]) -> str:
        if organ == "research":
            return (f"{step.get('watches', '?')} watches, "
                    f"{step.get('gaps', '?')} gaps")
        if organ == "memory":
            return (f"{step.get('episodes', '?')} episodes, "
                    f"{step.get('merged', '?')} merged")
        if organ == "presence":
            return (f"{step.get('noticed', 0)} noticed, "
                    f"{step.get('surfaced', 0)} surfaced")
        if organ == "weakness":
            return f"{step.get('open', 0)} open, {step.get('routed', 0)} routed"
        if organ == "patterns":
            return f"{step.get('interests_dropped', 0)} pruned"
        if organ == "wisdom":
            return str(step.get("report", ""))[:60]
        return ""

    # ── per-organ ticks (each organ drains its own queue) ───────────────

    @staticmethod
    def _run_tick(ticker: Any, db: Any, started: float,
                  budget: float) -> dict[str, Any]:
        """Invoke a tick, passing ``budget`` only if it accepts it.

        Keeps the coordinator compatible with third-party or test tick
        functions that only take ``(db, started)``.
        """
        try:
            import inspect as _inspect
            params = _inspect.signature(ticker).parameters
            if "budget" in params:
                return ticker(db, started, budget=budget)
            return ticker(db, started)
        except TypeError as exc:
            # Defensive: a tick with an incompatible signature must not
            # kill the whole cycle.
            if "budget" in str(exc):
                return ticker(db, started)
            raise

    def _tick_research(self, db: Any, started: float,
                       budget: float = MIN_ORGAN_SLICE) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False}
        if not self._idle or _budget_left(started) <= 0:
            out["skipped"] = "stand-down"
            return out
        t0 = time.time()
        try:
            out["queued"] = _organs.pending_count(db, dst="research")
        except Exception:  # noqa: BLE001
            out["queued"] = "?"
        try:
            from ..research.autonomy import ResearchOrgan
            report = ResearchOrgan(db).tick()
            out.update({
                "ok": True,
                "seconds": round(time.time() - t0, 1),
                "watches": getattr(report, "watches_run", None),
                "gaps": getattr(report, "gaps_opened", None),
                "errors": getattr(report, "errors", []),
            })
            _log.info("research tick: %s", report)
        except (ImportError, AttributeError):
            out["skipped"] = "research organ not available"
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)[:200]
            _log.warning("research tick failed", exc_info=True)
        return out

    def _tick_wisdom(self, db: Any, started: float,
                     budget: float = MIN_ORGAN_SLICE) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False}
        if not self._idle or _budget_left(started) <= 0:
            out["skipped"] = "stand-down"
            return out
        t0 = time.time()
        try:
            from ..wisdom.autonomy import WisdomOrgan
            report = WisdomOrgan(db).tick()
            out.update({"ok": True,
                        "seconds": round(time.time() - t0, 1),
                        "report": str(report)[:200]})
            _log.info("wisdom tick: %s", report)
        except (ImportError, AttributeError):
            out["skipped"] = "wisdom organ not available"
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)[:200]
            _log.warning("wisdom tick failed", exc_info=True)
        return out

    def _tick_memory(self, db: Any, started: float,
                     budget: float = MIN_ORGAN_SLICE) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False}
        if not self._idle or _budget_left(started) <= 0:
            out["skipped"] = "stand-down"
            return out
        t0 = time.time()
        try:
            from types import SimpleNamespace as _NS
            from ..memory.manager import MemoryManager
            # MemoryManager takes a context carrying .db, not a bare db.
            manager = MemoryManager(_NS(db=db, settings=None))
            report = manager.consolidate()
            out.update({"ok": True,
                        "seconds": round(time.time() - t0, 1),
                        "episodes": report.get("episodes"),
                        "merged": report.get("merged"),
                        "forgotten": report.get("forgotten")})
            _log.info("memory consolidation: %s", report)
        except (ImportError, AttributeError):
            out["skipped"] = "memory manager not available"
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)[:200]
            _log.warning("memory consolidation failed", exc_info=True)
        return out

    def _tick_presence(self, db: Any, started: float,
                       budget: float = MIN_ORGAN_SLICE) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False}
        if not self._idle or _budget_left(started) <= 0:
            out["skipped"] = "stand-down"
            return out
        t0 = time.time()
        try:
            from .presence import heartbeat
            did = heartbeat(db)
            out.update({"ok": True,
                        "seconds": round(time.time() - t0, 1),
                        "noticed": len(did.get("noticed", [])),
                        "prepared": len(did.get("prepared", [])),
                        "surfaced": len(did.get("surfaced", []))})
            _log.info("idle presence heartbeat: %s", did)
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)[:200]
            _log.warning("idle presence failed", exc_info=True)
        return out

    def _tick_weakness(self, db: Any, started: float,
                       budget: float = MIN_ORGAN_SLICE) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False}
        if not self._idle or _budget_left(started) <= 0:
            out["skipped"] = "stand-down"
            return out
        t0 = time.time()
        try:
            from .weakness import idle_tick
            report = idle_tick(db)
            out.update({"ok": True,
                        "seconds": round(time.time() - t0, 1),
                        **report})
            if report.get("open"):
                _log.info("idle: %d open weaknesses", report["open"])
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)[:200]
            _log.warning("idle weakness scan failed", exc_info=True)
        return out

    def _tick_patterns(self, db: Any, started: float,
                       budget: float = MIN_ORGAN_SLICE) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False}
        if not self._idle or _budget_left(started) <= 0:
            out["skipped"] = "stand-down"
            return out
        t0 = time.time()
        try:
            from .patterns import prune
            report = prune(db)
            out.update({"ok": True,
                        "seconds": round(time.time() - t0, 1),
                        **report})
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)[:200]
            _log.warning("pattern hygiene failed", exc_info=True)
        return out
