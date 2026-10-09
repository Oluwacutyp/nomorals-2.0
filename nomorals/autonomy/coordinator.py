"""Idle-task coordinator — when the system goes quiet, organs work.

Subscribes to ``system.idle`` on the global bus. When it fires:

1. **Research organ** — drains its organ-event queue (weakness
   investigations, interest watches, window preparations).
2. **Wisdom organ** — drains its queue (corpus ingestion from research
   findings).
3. **Memory** — consolidation tick (compress old episodes, refresh
   salience).
4. **Presence** — runs the heartbeat (notice/prepare/surface).

When ``system.active`` fires, background work stands down gracefully
(current tick finishes, no new ticks start).

This is the "organs trigger each other" wiring — not a scheduler
cron, but a real event chain: quiet → idle event → organs wake →
work → done.
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


class IdleCoordinator:
    """Listens for idle/active events, runs organ maintenance."""

    def __init__(self, workspace_dir: str | Path):
        self.workspace_dir = Path(workspace_dir)
        self._idle = False
        self._cycle_lock = threading.Lock()
        self._subscribed = False

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

    def _maintenance_cycle(self) -> None:
        if not self._cycle_lock.acquire(blocking=False):
            _log.debug("maintenance cycle already running — skipping")
            return
        started = time.time()
        try:
            from ..storage.db import Database
            db = Database(str(self.workspace_dir))

            # 1. Research organ — drain its event queue.
            self._drain_organ(db, "research", started)

            # 2. Wisdom organ — drain its event queue.
            self._drain_organ(db, "wisdom", started)

            # 3. Presence heartbeat — notice/prepare/surface.
            if self._idle and time.time() - started < IDLE_CYCLE_BUDGET:
                try:
                    from .presence import heartbeat
                    did = heartbeat(db)
                    _log.info("idle presence heartbeat: %s", did)
                except Exception:  # noqa: BLE001
                    _log.warning("idle presence failed", exc_info=True)

            # 4. Weakness scan — check for newly-thresholded cases.
            if self._idle and time.time() - started < IDLE_CYCLE_BUDGET:
                try:
                    from .weakness import open_weaknesses
                    open_w = open_weaknesses(db)
                    if open_w:
                        _log.info("idle: %d open weaknesses", len(open_w))
                except Exception:  # noqa: BLE001
                    _log.warning("idle weakness scan failed", exc_info=True)

            elapsed = time.time() - started
            _log.info("idle maintenance cycle done in %.1fs", elapsed)
        except Exception as exc:  # noqa: BLE001
            _log.warning("idle maintenance cycle failed: %s", exc,
                         exc_info=True)
        finally:
            self._cycle_lock.release()

    def _drain_organ(self, db: Any, organ: str, started: float) -> None:
        """Drain one organ's event queue, within budget."""
        if not self._idle:
            return
        if time.time() - started >= IDLE_CYCLE_BUDGET:
            return
        try:
            events = _organs.drain(db, dst=organ)
        except Exception:  # noqa: BLE001
            _log.warning("organ drain failed for %s", organ, exc_info=True)
            return
        if not events:
            return
        _log.info("idle: drained %d events for %s", len(events), organ)
        for ev in events:
            if not self._idle:
                break
            if time.time() - started >= IDLE_CYCLE_BUDGET:
                break
            try:
                self._handle_organ_event(db, organ, ev)
            except Exception:  # noqa: BLE001
                _log.warning("organ event failed (%s %s)", organ,
                             ev.get("kind"), exc_info=True)

    def _handle_organ_event(self, db: Any, organ: str,
                            ev: dict[str, Any]) -> None:
        kind = ev.get("kind", "")
        payload = ev.get("payload", {})
        _log.info("organ event: %s → %s (%s)", ev.get("src"), organ, kind)
        # Dispatch to organ-specific handlers. Each organ owns its
        # tick logic; the coordinator just routes.
        if organ == "research":
            self._handle_research_event(db, kind, payload)
        elif organ == "wisdom":
            self._handle_wisdom_event(db, kind, payload)

    def _handle_research_event(self, db: Any, kind: str,
                               payload: dict[str, Any]) -> None:
        # The research organ's autonomy class owns the tick.
        try:
            from ..research.autonomy import ResearchOrgan
            organ = ResearchOrgan(db)
            report = organ.tick()
            _log.info("research tick: %s", report)
        except (ImportError, AttributeError):
            _log.debug("research organ not available")
        except Exception:  # noqa: BLE001
            _log.warning("research tick failed", exc_info=True)

    def _handle_wisdom_event(self, db: Any, kind: str,
                             payload: dict[str, Any]) -> None:
        try:
            from ..wisdom.autonomy import WisdomOrgan
            organ = WisdomOrgan(db)
            report = organ.tick()
            _log.info("wisdom tick: %s", report)
        except (ImportError, AttributeError):
            _log.debug("wisdom organ not available")
        except Exception:  # noqa: BLE001
            _log.warning("wisdom tick failed", exc_info=True)
