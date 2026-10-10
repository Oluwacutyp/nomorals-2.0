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

Idle is graduated, not binary (inspired by logind/WICG idle research):

* **shallow** — threshold crossed; light maintenance is fine.
* **deep** — sustained quiet; heavier work (research ticks) unlocks.
* **night** — very long quiet; the full overnight pipeline may run.

Each stage has a minimum *residency* (cpuidle-style): a stage only
counts once the system has dwelled in it long enough for the work to
pay off — a 10-minute job must not start 30 seconds into a 35-second
idle window. Stage transitions emit ``system.idle.stage`` on the bus.

Organs doing long work can hold an **idle inhibitor**
(:func:`inhibit_idle` / :func:`idle_inhibited`) so the maintenance
cycle skips while they're busy — like a video player inhibiting the
screensaver.

Critical wiring rule: the monitor and every ``note_activity`` caller
must read and write the **same database file**. :func:`workspace_db`
resolves the canonical DB path for a workspace directory; both the
monitor and the call sites use it. (A past bug had the monitor opening
its own ``autonomy.db`` while writers wrote to the main DB — the idle
chain silently never fired.)
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: Default seconds of quiet before the system counts as idle.
DEFAULT_IDLE_SECONDS = 900  # 15 minutes

#: How often the monitor checks (seconds).
CHECK_INTERVAL = 60

#: Canonical DB filename inside a workspace directory.
WORKSPACE_DB_NAME = "nomorals.db"

#: Idle stages: (name, multiple of the idle threshold, min residency s).
#: A stage unlocks once idle_for >= threshold * multiple, and it only
#: *counts* (for scheduling heavy work) after dwelling RESIDENCY seconds.
IDLE_STAGES: tuple[tuple[str, float, float], ...] = (
    ("shallow", 1.0, 0.0),
    ("deep", 4.0, 120.0),
    ("night", 12.0, 600.0),
)

#: EMA alpha for the adaptive idle threshold (inter-activity gaps).
GAP_EMA_ALPHA = 0.15
#: Bounds for the learned threshold (seconds).
LEARNED_THRESHOLD_MIN = 300.0
LEARNED_THRESHOLD_MAX = 3600.0


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
    # Idle inhibitors: organs holding one block maintenance cycles.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS autonomy_inhibitors (
            name TEXT PRIMARY KEY,
            reason TEXT NOT NULL DEFAULT '',
            ts REAL NOT NULL DEFAULT 0
        )
        """
    )
    # Idle session history: each completed idle window's duration.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS autonomy_idle_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_ts REAL NOT NULL,
            ended_ts REAL NOT NULL,
            duration_s REAL NOT NULL,
            max_stage TEXT NOT NULL DEFAULT 'shallow'
        )
        """
    )
    # Adaptive threshold state: EMA of inter-activity gaps.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS autonomy_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL DEFAULT ''
        )
        """
    )
    db.execute(
        "INSERT OR IGNORE INTO autonomy_config (key, value) "
        "VALUES ('gap_ema', '0')"
    )


def _get_config(db: Any, key: str, default: str = "") -> str:
    ensure_schema(db)
    row = db.execute(
        "SELECT value FROM autonomy_config WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row and row[0] is not None else default


def _set_config(db: Any, key: str, value: str) -> None:
    ensure_schema(db)
    db.execute(
        "INSERT INTO autonomy_config (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass


def note_activity(db: Any, ts: float | None = None,
                  kind: str = "") -> None:
    """Record activity — resets the idle clock. Cheap, call liberally.

    ``kind`` optionally names the activity source
    (``message``/``tool_call``/``task``/``system``) for future analytics;
    it is stored on the session record, not on the hot row.
    """
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    prev = db.execute(
        "SELECT last_activity_ts FROM autonomy_activity WHERE id = 1"
    ).fetchone()
    prev_ts = float(prev[0]) if prev and prev[0] else 0.0
    db.execute(
        "UPDATE autonomy_activity SET last_activity_ts = ?, idle_state = 0 "
        "WHERE id = 1",
        (now,),
    )
    # Feed the adaptive threshold: EMA of inter-activity gaps.
    if prev_ts > 0 and now > prev_ts:
        gap = now - prev_ts
        try:
            ema = float(_get_config(db, "gap_ema", "0") or 0)
        except ValueError:
            ema = 0.0
        ema = gap if ema <= 0 else (
            GAP_EMA_ALPHA * gap + (1 - GAP_EMA_ALPHA) * ema)
        _set_config(db, "gap_ema", str(ema))
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


def learned_idle_threshold(
        db: Any, default: float = DEFAULT_IDLE_SECONDS) -> float:
    """Adaptive idle threshold from observed inter-activity gaps.

    The EMA of gaps between activities, doubled, clamped to a sane
    range. A owner who messages every 5 minutes gets a shorter fuse
    than one who goes quiet for hours. Falls back to ``default`` until
    enough history exists.
    """
    try:
        ema = float(_get_config(db, "gap_ema", "0") or 0)
    except ValueError:
        ema = 0.0
    if ema <= 0:
        return default
    learned = ema * 2.0
    return max(LEARNED_THRESHOLD_MIN,
               min(LEARNED_THRESHOLD_MAX, learned))


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


# ── idle stages ──────────────────────────────────────────────────────

def stage_for(idle_for: float,
              threshold: float = DEFAULT_IDLE_SECONDS) -> str:
    """Idle stage name for a given quiet duration.

    Returns ``'active'`` below the threshold, else the deepest stage
    whose multiple is reached.
    """
    if idle_for < threshold:
        return "active"
    stage = "shallow"
    for name, multiple, _residency in IDLE_STAGES:
        if idle_for >= threshold * multiple:
            stage = name
    return stage


def stage_residency_met(stage: str, stage_since: float,
                        ts: float | None = None) -> bool:
    """Has the system dwelled in ``stage`` long enough to trust it?

    cpuidle-style target residency: heavy work only starts after the
    minimum dwell, so a flicker of quiet doesn't launch a 10-minute job.
    """
    now = ts if ts is not None else time.time()
    for name, _multiple, residency in IDLE_STAGES:
        if name == stage:
            return (now - stage_since) >= residency
    return True


# ── idle inhibitors ──────────────────────────────────────────────────

def inhibit_idle(db: Any, name: str, reason: str = "") -> None:
    """Hold an idle inhibitor — maintenance cycles skip while held.

    Organs doing long work (a training run, a big ingestion) call this
    so the coordinator doesn't start competing maintenance. Name is
    unique per holder; re-inhibiting refreshes the timestamp.
    """
    ensure_schema(db)
    db.execute(
        "INSERT INTO autonomy_inhibitors (name, reason, ts) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT (name) DO UPDATE SET reason = excluded.reason, "
        "ts = excluded.ts",
        (str(name or "unnamed"), str(reason or ""), time.time()),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    _log.info("idle inhibited by %s (%s)", name, reason)


def release_idle(db: Any, name: str) -> bool:
    """Release a previously held idle inhibitor."""
    ensure_schema(db)
    cur = db.execute(
        "DELETE FROM autonomy_inhibitors WHERE name = ?",
        (str(name or "unnamed"),))
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    released = bool(cur.rowcount)
    if released:
        _log.info("idle inhibitor released: %s", name)
    return released


def inhibitors(db: Any) -> list[dict[str, Any]]:
    """Currently held idle inhibitors."""
    ensure_schema(db)
    rows = db.execute(
        "SELECT name, reason, ts FROM autonomy_inhibitors "
        "ORDER BY ts DESC").fetchall()
    return [{"name": r[0], "reason": r[1], "ts": r[2]} for r in rows]


def idle_inhibited(db: Any) -> bool:
    """True when any inhibitor is held (maintenance should stand down)."""
    return len(inhibitors(db)) > 0


@contextlib.contextmanager
def idle_inhibited_scope(db: Any, name: str,
                         reason: str = "") -> Iterator[None]:
    """Context manager: hold an inhibitor for the block's duration."""
    inhibit_idle(db, name, reason)
    try:
        yield
    finally:
        release_idle(db, name)


def prune_stale_inhibitors(db: Any, max_age: float = 7200.0) -> int:
    """Drop inhibitors older than ``max_age`` (crashed holders).

    Returns the number pruned. Called on monitor start so a crashed
    organ can't block maintenance forever.
    """
    ensure_schema(db)
    cutoff = time.time() - max_age
    cur = db.execute(
        "DELETE FROM autonomy_inhibitors WHERE ts < ?", (cutoff,))
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return cur.rowcount or 0


# ── idle session history ─────────────────────────────────────────────

def _record_idle_session(db: Any, started_ts: float, ended_ts: float,
                         max_stage: str) -> None:
    ensure_schema(db)
    db.execute(
        "INSERT INTO autonomy_idle_sessions "
        "(started_ts, ended_ts, duration_s, max_stage) "
        "VALUES (?, ?, ?, ?)",
        (started_ts, ended_ts, max(0.0, ended_ts - started_ts), max_stage),
    )
    # Bounded growth: keep the last 500 sessions.
    db.execute(
        "DELETE FROM autonomy_idle_sessions WHERE id NOT IN ("
        "SELECT id FROM autonomy_idle_sessions "
        "ORDER BY id DESC LIMIT 500)"
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass


def idle_history(db: Any, limit: int = 20) -> list[dict[str, Any]]:
    """Recent completed idle sessions, newest first."""
    ensure_schema(db)
    rows = db.execute(
        "SELECT started_ts, ended_ts, duration_s, max_stage FROM "
        "autonomy_idle_sessions ORDER BY id DESC LIMIT ?",
        (max(1, int(limit)),)).fetchall()
    return [{"started_ts": r[0], "ended_ts": r[1],
             "duration_s": r[2], "max_stage": r[3]} for r in rows]


def idle_stats(db: Any) -> dict[str, Any]:
    """Aggregate idle-session statistics for threshold tuning."""
    ensure_schema(db)
    rows = db.execute(
        "SELECT duration_s FROM autonomy_idle_sessions").fetchall()
    durations = [float(r[0]) for r in rows if r[0]]
    if not durations:
        return {"sessions": 0}
    durations.sort()
    n = len(durations)
    return {
        "sessions": n,
        "median_s": round(durations[n // 2], 1),
        "p90_s": round(durations[int(n * 0.9)], 1),
        "max_s": round(durations[-1], 1),
        "mean_s": round(sum(durations) / n, 1),
    }


class IdleMonitor:
    """Background idle detector. Emits ``system.idle`` / ``system.active``.

    Also emits ``system.idle.stage`` on stage transitions
    (shallow/deep/night) so organs can subscribe to the depth they care
    about, and honors idle inhibitors.

    Usage::

        monitor = IdleMonitor(workspace_dir, idle_seconds=900)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(self, workspace_dir: str | Path,
                 idle_seconds: float = DEFAULT_IDLE_SECONDS,
                 on_idle: Callable[[float], None] | None = None,
                 on_active: Callable[[], None] | None = None,
                 adaptive: bool = False):
        self.workspace_dir = Path(workspace_dir)
        self.idle_seconds = idle_seconds
        #: When True, the threshold is learned from inter-activity gaps.
        self.adaptive = adaptive
        self.on_idle = on_idle
        self.on_active = on_active
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._was_idle = False
        self._stage = "active"
        self._stage_since = 0.0
        self._idle_started = 0.0

    def _db(self) -> Any:
        # MUST be the same store the note_activity call sites write to.
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_db(self.workspace_dir)

    def _threshold(self, db: Any) -> float:
        if self.adaptive:
            return learned_idle_threshold(db, self.idle_seconds)
        return self.idle_seconds

    def _emit_stage(self, stage: str, idle_for: float) -> None:
        try:
            global_bus.publish(Event(
                topic="system.idle.stage",
                data={"stage": stage, "idle_seconds": idle_for,
                      "residency_met": stage_residency_met(
                          stage, self._stage_since)},
                source="nomorals.autonomy.idle",
            ))
        except Exception:  # noqa: BLE001
            _log.debug("system.idle.stage publish failed", exc_info=True)

    def _update_stage(self, idle_for: float, threshold: float) -> None:
        stage = stage_for(idle_for, threshold)
        if stage != self._stage:
            self._stage = stage
            self._stage_since = time.time()
            _log.info("idle stage → %s (%.0fs quiet)", stage, idle_for)
            self._emit_stage(stage, idle_for)

    def check_once(self) -> str:
        """Single idle check.

        Returns 'idle', 'active', 'stage:<name>', or 'unchanged'.
        """
        db = self._db()
        try:
            threshold = self._threshold(db)
            last = last_activity_ts(db)
            now = time.time()
            idle_for = now - last if last > 0 else 0.0
            is_idle = last > 0 and idle_for >= threshold

            if is_idle and not self._was_idle:
                self._was_idle = True
                self._idle_started = now - idle_for
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
                        data={"idle_seconds": idle_for,
                              "stage": stage_for(idle_for, threshold),
                              "threshold": threshold},
                        source="nomorals.autonomy.idle",
                    ))
                except Exception:  # noqa: BLE001
                    _log.debug("system.idle publish failed", exc_info=True)
                if self.on_idle:
                    try:
                        self.on_idle(idle_for)
                    except Exception:  # noqa: BLE001
                        _log.warning("on_idle callback failed", exc_info=True)
                self._update_stage(idle_for, threshold)
                return "idle"

            if is_idle and self._was_idle:
                # Already idle — watch for stage transitions.
                prev_stage = self._stage
                self._update_stage(idle_for, threshold)
                if self._stage != prev_stage:
                    return f"stage:{self._stage}"
                return "unchanged"

            if not is_idle and self._was_idle:
                self._was_idle = False
                db.execute(
                    "UPDATE autonomy_activity SET idle_state = 0 WHERE id = 1"
                )
                try:
                    db.commit()
                except Exception:  # noqa: BLE001
                    pass
                _record_idle_session(db, self._idle_started, now,
                                     self._stage)
                max_stage = self._stage
                self._stage = "active"
                _log.info("system active again — emitting system.active")
                try:
                    global_bus.publish(Event(
                        topic="system.active",
                        data={"was_idle_for": idle_for,
                              "max_stage": max_stage},
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
        # A crashed holder must not block maintenance forever.
        try:
            prune_stale_inhibitors(self._db())
        except Exception:  # noqa: BLE001
            _log.debug("inhibitor prune failed", exc_info=True)

        def _run() -> None:
            while not self._stop.wait(CHECK_INTERVAL):
                try:
                    self.check_once()
                except Exception:  # noqa: BLE001
                    _log.warning("idle check failed", exc_info=True)

        self._thread = threading.Thread(
            target=_run, name="idle-monitor", daemon=True)
        self._thread.start()
        _log.info("idle monitor started (threshold %ss, adaptive=%s)",
                  self.idle_seconds, self.adaptive)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
