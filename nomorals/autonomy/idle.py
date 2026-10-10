"""System-wide idle detection — the maintenance window trigger.

When Devon's been quiet (no inbound messages, no tool calls, no task
completions) for a threshold, the system is idle. Idle time is when
organs do their background work: research runs, wisdom ingests,
memory consolidates, weaknesses get investigated.

Design:
* Activity is recorded via :func:`note_activity` — called from the
  runtime's message path, tool loop, and scheduler completions.
* :class:`IdleMonitor` runs on a lightweight thread, checks every minute.
* When idle threshold is crossed, emits ``system.idle`` on the global
  bus with the idle duration. Organs subscribe and drain on their tick.
* When activity resumes, emits ``system.active`` so organs stand down.
* State persists in SQLite so restarts don't lose the idle clock.

Critical wiring rule: the monitor and every ``note_activity`` caller
must read and write the **same database file**. :func:`workspace_db`
resolves the canonical DB path for a workspace directory; both the
monitor and the call sites use it. (A past bug had the monitor opening
its own ``autonomy.db`` while writers wrote to the main DB — the idle
chain silently never fired.)
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: Default seconds of quiet before the system counts as idle.
DEFAULT_IDLE_SECONDS = 900  # 15 minutes

#: How often the monitor checks (seconds).
CHECK_INTERVAL = 60

#: Canonical DB filename inside a workspace directory.
WORKSPACE_DB_NAME = "nomorals.db"


def workspace_db(workspace_dir: str | Path) -> Any:
    """Open the canonical workspace database for autonomy state.

    ``workspace_dir`` may be a directory (the DB file is
    ``<dir>/nomorals.db``) or a direct path to a DB file. Every autonomy
    component — idle monitor, activity hooks, coordinator, presence,
    weakness — must use this, so they share one state store. This is the
    seam other modules should import instead of constructing their own
    ``Database(workspace_dir)``.
    """
    from ..storage.db import Database

    p = Path(workspace_dir)
    if p.is_dir() or not p.suffix:
        db_path = p / WORKSPACE_DB_NAME
    else:
        db_path = p
    return Database(str(db_path))


def ensure_schema(db: Any) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS autonomy_activity (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_activity_ts REAL NOT NULL DEFAULT 0,
            last_idle_emit_ts REAL NOT NULL DEFAULT 0,
            idle_state INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    db.execute(
        "INSERT OR IGNORE INTO autonomy_activity (id) VALUES (1)"
    )


def note_activity(db: Any, ts: float | None = None) -> None:
    """Record activity — resets the idle clock. Cheap, call liberally."""
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    db.execute(
        "UPDATE autonomy_activity SET last_activity_ts = ?, idle_state = 0 "
        "WHERE id = 1",
        (now,),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001 - best effort
        pass


def last_activity_ts(db: Any) -> float:
    ensure_schema(db)
    row = db.execute(
        "SELECT last_activity_ts FROM autonomy_activity WHERE id = 1"
    ).fetchone()
    return float(row[0]) if row else 0.0


def idle_state(db: Any) -> dict[str, Any]:
    """Current idle bookkeeping: last activity, idle flag, last emit."""
    ensure_schema(db)
    row = db.execute(
        "SELECT last_activity_ts, last_idle_emit_ts, idle_state "
        "FROM autonomy_activity WHERE id = 1"
    ).fetchone()
    now = time.time()
    last = float(row[0]) if row and row[0] else 0.0
    return {
        "last_activity_ts": last,
        "idle_seconds": max(0.0, now - last) if last > 0 else 0.0,
        "idle": bool(row[2]) if row else False,
        "last_idle_emit_ts": float(row[1]) if row and row[1] else 0.0,
    }


class IdleMonitor:
    """Background idle detector. Emits ``system.idle`` / ``system.active``.

    Usage::

        monitor = IdleMonitor(workspace_dir, idle_seconds=900)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(self, workspace_dir: str | Path,
                 idle_seconds: float = DEFAULT_IDLE_SECONDS,
                 on_idle: Callable[[float], None] | None = None,
                 on_active: Callable[[], None] | None = None):
        self.workspace_dir = Path(workspace_dir)
        self.idle_seconds = idle_seconds
        self.on_idle = on_idle
        self.on_active = on_active
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._was_idle = False

    def _db(self) -> Any:
        # MUST be the same store the note_activity call sites write to.
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_db(self.workspace_dir)

    def check_once(self) -> str:
        """Single idle check. Returns 'idle', 'active', or 'unchanged'."""
        db = self._db()
        try:
            last = last_activity_ts(db)
            now = time.time()
            idle_for = now - last if last > 0 else 0.0
            is_idle = last > 0 and idle_for >= self.idle_seconds

            if is_idle and not self._was_idle:
                self._was_idle = True
                db.execute(
                    "UPDATE autonomy_activity SET idle_state = 1, "
                    "last_idle_emit_ts = ? WHERE id = 1",
                    (now,),
                )
                try:
                    db.commit()
                except Exception:  # noqa: BLE001
                    pass
                _log.info("system idle for %.0fs — emitting system.idle",
                          idle_for)
                try:
                    global_bus.publish(Event(
                        topic="system.idle",
                        data={"idle_seconds": idle_for},
                        source="nomorals.autonomy.idle",
                    ))
                except Exception:  # noqa: BLE001
                    _log.debug("system.idle publish failed", exc_info=True)
                if self.on_idle:
                    try:
                        self.on_idle(idle_for)
                    except Exception:  # noqa: BLE001
                        _log.warning("on_idle callback failed", exc_info=True)
                return "idle"

            if not is_idle and self._was_idle:
                self._was_idle = False
                db.execute(
                    "UPDATE autonomy_activity SET idle_state = 0 WHERE id = 1"
                )
                try:
                    db.commit()
                except Exception:  # noqa: BLE001
                    pass
                _log.info("system active again — emitting system.active")
                try:
                    global_bus.publish(Event(
                        topic="system.active",
                        data={},
                        source="nomorals.autonomy.idle",
                    ))
                except Exception:  # noqa: BLE001
                    _log.debug("system.active publish failed", exc_info=True)
                if self.on_active:
                    try:
                        self.on_active()
                    except Exception:  # noqa: BLE001
                        _log.warning("on_active callback failed",
                                     exc_info=True)
                return "active"

            return "unchanged"
        finally:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass

    def start(self) -> "IdleMonitor":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()

        def _run() -> None:
            while not self._stop.wait(CHECK_INTERVAL):
                try:
                    self.check_once()
                except Exception:  # noqa: BLE001
                    _log.warning("idle check failed", exc_info=True)

        self._thread = threading.Thread(
            target=_run, name="idle-monitor", daemon=True)
        self._thread.start()
        _log.info("idle monitor started (threshold %ss)", self.idle_seconds)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
