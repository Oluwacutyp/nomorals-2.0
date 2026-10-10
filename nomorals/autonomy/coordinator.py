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
            from .idle import workspace_db
            db = workspace_db(self.workspace_dir)
            self._ledger(db, "cycle_start", cycle_id,
                         "idle maintenance cycle started")

            # 1. Research organ — one tick (it drains its own queue).
            stats["steps"]["research"] = self._tick_research(db, started)

            # 2. Wisdom organ — one tick.
            stats["steps"]["wisdom"] = self._tick_wisdom(db, started)

            # 3. Memory consolidation.
            stats["steps"]["memory"] = self._tick_memory(db, started)

            # 4. Presence heartbeat — notice/prepare/surface.
            stats["steps"]["presence"] = self._tick_presence(db, started)

            # 5. Weakness scan — thresholded cases → research handoff.
            stats["steps"]["weakness"] = self._tick_weakness(db, started)

            # 6. Model hygiene — prune decayed interests/patterns.
            stats["steps"]["patterns"] = self._tick_patterns(db, started)

            elapsed = time.time() - started
            stats["seconds"] = round(elapsed, 1)
            _log.info("idle maintenance cycle done in %.1fs: %s",
                      elapsed, {k: v for k, v in stats["steps"].items()})
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

    # ── per-organ ticks (each organ drains its own queue) ───────────────

    def _tick_research(self, db: Any, started: float) -> dict[str, Any]:
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

    def _tick_wisdom(self, db: Any, started: float) -> dict[str, Any]:
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

    def _tick_memory(self, db: Any, started: float) -> dict[str, Any]:
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

    def _tick_presence(self, db: Any, started: float) -> dict[str, Any]:
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

    def _tick_weakness(self, db: Any, started: float) -> dict[str, Any]:
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

    def _tick_patterns(self, db: Any, started: float) -> dict[str, Any]:
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
