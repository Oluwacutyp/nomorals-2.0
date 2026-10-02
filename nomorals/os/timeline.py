"""Persistent event timeline (Wave H2, L6).

Subscribes to the process :class:`~nomorals.core.events.EventBus` and
persists every matching event into an ``event_log`` SQLite table so the
history survives restarts. ``nm timeline`` reads this table back.

Consumes the H2 event topic contract:

* ``mission.transition`` — {mission_id, from_state, to_state, note}
* ``mission.verify``     — {mission_id, verdict, details}
* ``artifact.created``    — {artifact_id, uri, type, creator, mission_id, task_id}
* ``session.created``     — {session_id, frontend, principal}
* ``session.ended``       — {session_id}
* plus any other ``mission.*`` / ``artifact.*`` / ``session.*`` / ``task.*``
  topics the siblings emit.

The table is created with ``CREATE TABLE IF NOT EXISTS`` so this module
never needs a storage migration (and can never collide with one a sibling
lands). Recording never raises: a malformed event is persisted with
whatever fields could be extracted.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

__all__ = ["Timeline", "TIMELINE_PATTERNS"]

_log = logging.getLogger(__name__)

#: Bus topic patterns the timeline subscribes to (async delivery).
TIMELINE_PATTERNS: tuple[str, ...] = ("mission.*", "artifact.*", "session.*", "task.*")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_log (
    event_id   TEXT PRIMARY KEY,
    topic      TEXT NOT NULL,
    ts         REAL NOT NULL,
    source     TEXT NOT NULL DEFAULT '',
    session_id TEXT,
    project_id TEXT,
    mission_id TEXT,
    artifact_id TEXT,
    data_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_event_log_ts ON event_log (ts);
CREATE INDEX IF NOT EXISTS idx_event_log_topic ON event_log (topic);
CREATE INDEX IF NOT EXISTS idx_event_log_mission ON event_log (mission_id);
CREATE INDEX IF NOT EXISTS idx_event_log_session ON event_log (session_id);
"""

_ROW_COLUMNS = (
    "event_id",
    "topic",
    "ts",
    "source",
    "session_id",
    "project_id",
    "mission_id",
    "artifact_id",
    "data_json",
)


def _new_event_id() -> str:
    return "evt_" + uuid.uuid4().hex[:12]


def _coerce_ts(value: Any) -> float | None:
    """Accept epoch numbers, ISO-8601 strings, and datetimes."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        # Not a number — try ISO-8601 below.
        return _coerce_ts_iso(text)


def _coerce_ts_iso(text: str) -> float:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise ValueError(f"cannot parse timestamp {text!r}") from exc


def _glob_to_like(pattern: str) -> str:
    """Translate an fnmatch-style glob into a SQL LIKE pattern."""
    out: list[str] = []
    for ch in pattern:
        if ch == "*":
            out.append("%")
        elif ch == "?":
            out.append("_")
        elif ch in ("\\", "%", "_"):
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


class Timeline:
    """A durable, queryable log of bus events.

    Parameters
    ----------
    db_path:
        Path to the SQLite file. ``None`` keeps the log in memory (useful
        in tests). The ``event_log`` table is created on demand.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path) if db_path is not None else None
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            ":memory:" if self._path is None else str(self._path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._sub_ids: list[str] = []
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:  # noqa: BLE001 - best effort on teardown
                pass

    def __enter__(self) -> "Timeline":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- recording ------------------------------------------------------
    def attach(self, bus: Any, *, sync: bool = False) -> list[str]:
        """Subscribe (async by default) to the H2 topic families.

        Returns the subscription ids so the caller can :meth:`detach`.
        """
        ids: list[str] = []
        for pattern in TIMELINE_PATTERNS:
            sub_id = bus.subscribe(pattern, self._on_event, sync=sync)
            ids.append(sub_id)
        self._sub_ids.extend(ids)
        return ids

    def detach(self, bus: Any) -> None:
        """Remove every subscription created by :meth:`attach`."""
        for sub_id in self._sub_ids:
            try:
                bus.unsubscribe(sub_id)
            except Exception:  # noqa: BLE001 - unsubscribe is best effort
                _log.debug("timeline detach failed for %s", sub_id, exc_info=True)
        self._sub_ids.clear()

    def _on_event(self, event: Any) -> None:
        self.record(event)

    def record(self, event: Any) -> str | None:
        """Persist one event. Never raises — returns the event_id or None."""
        try:
            row = self._extract(event)
        except Exception:  # noqa: BLE001 - extraction must not kill the bus
            _log.debug("timeline: could not extract event", exc_info=True)
            return None
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO event_log "
                    "(event_id, topic, ts, source, session_id, project_id, "
                    " mission_id, artifact_id, data_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["event_id"],
                        row["topic"],
                        row["ts"],
                        row["source"],
                        row["session_id"],
                        row["project_id"],
                        row["mission_id"],
                        row["artifact_id"],
                        row["data_json"],
                    ),
                )
                self._conn.commit()
        except Exception:  # noqa: BLE001 - a full/corrupt db must not kill the bus
            _log.debug("timeline: could not persist event", exc_info=True)
            return None
        return row["event_id"]

    @staticmethod
    def _extract(event: Any) -> dict[str, Any]:
        data = getattr(event, "data", None)
        if isinstance(data, Mapping):
            payload: dict[str, Any] = dict(data)
        elif data is None:
            payload = {}
        else:  # malformed: data is not a mapping — keep it visible, keep going
            payload = {"_raw": str(data)}

        def _str(key: str) -> str | None:
            value = payload.get(key)
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        event_id = getattr(event, "event_id", None) or _new_event_id()
        topic = getattr(event, "topic", None) or ""
        try:
            ts = float(getattr(event, "ts", time.time()))
        except (TypeError, ValueError):
            ts = time.time()
        source = getattr(event, "source", None) or ""
        try:
            data_json = json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            data_json = json.dumps({"_unserializable": True})
        return {
            "event_id": str(event_id),
            "topic": str(topic),
            "ts": ts,
            "source": str(source),
            "session_id": _str("session_id"),
            "project_id": _str("project_id"),
            "mission_id": _str("mission_id"),
            "artifact_id": _str("artifact_id"),
            "data_json": data_json,
        }

    # -- querying -------------------------------------------------------
    def query(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        mission_id: str | None = None,
        artifact_id: str | None = None,
        topic: str | None = None,
        since: Any = None,
        until: Any = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Newest-first rows matching the given filters.

        ``topic`` is an fnmatch-style glob (``mission.*``). ``since`` /
        ``until`` accept epoch seconds, ISO-8601 strings, or datetimes.
        """
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("session_id", session_id),
            ("project_id", project_id),
            ("mission_id", mission_id),
            ("artifact_id", artifact_id),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if topic:
            clauses.append("topic LIKE ? ESCAPE '\\'")
            params.append(_glob_to_like(topic))
        since_ts = _coerce_ts(since)
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(since_ts)
        until_ts = _coerce_ts(until)
        if until_ts is not None:
            clauses.append("ts <= ?")
            params.append(until_ts)

        limit = int(limit)
        if limit < 0:
            limit = 200
        sql = (
            "SELECT " + ", ".join(_ROW_COLUMNS) + " FROM event_log"
            + (" WHERE " + " AND ".join(clauses) if clauses else "")
            + " ORDER BY ts DESC, rowid DESC LIMIT ?"
        )
        params.append(limit)
        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest-first convenience wrapper over :meth:`query`."""
        return self.query(limit=limit)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        out = {column: row[column] for column in _ROW_COLUMNS if column != "data_json"}
        raw = row["data_json"] or "{}"
        try:
            out["data"] = json.loads(raw)
        except (TypeError, ValueError):
            out["data"] = {"_raw": raw}
        return out
