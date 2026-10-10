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
* ``document.parsed``     — {doc_id, format, title, sections, tables, source}
* ``browser.session.opened`` / ``browser.session.closed`` — {session}
* ``browser.tab.navigated`` — {session, tab_id, url, title}
* ``browser.download.completed`` — {url, path, size, mime, artifact_uri,
  session, tab_id, mission_id}
* ``codews.workspace.opened`` — {root, mission_id}
* ``codews.patch.applied`` — {root, files, patched, failed}
* ``codews.tests.run``    — {root, runner, ok, passed, failed}
* ``codews.build.run``    — {root, target, ok}
* ``wisdom.practice.started`` / ``wisdom.practice.completed`` —
  {session_id, name, rounds, phases_done}
* ``connector.connected`` / ``connector.disconnected`` —
  {connector_id, name, account}
* ``datasci.dataset.loaded`` / ``datasci.dataset.dropped`` —
  {name, source, rows, columns}
* ``plugin.installed`` / ``plugin.enabled`` / ``plugin.disabled`` /
  ``plugin.removed`` — {name, version}
* ``mesh.task.dispatched`` / ``mesh.task.completed`` / ``mesh.task.failed`` —
  {job_id, task_type, origin_node, target_node}
* ``sync.completed`` — {peer_id, pushed, pulled, conflicts_resolved,
  duration_s}
* ``power.task.dispatched`` / ``power.task.completed`` /
  ``power.task.failed`` — {job_id, task_type, power_class}
* ``stream.started`` / ``stream.stopped`` — {host, port, url}
* ``search.performed`` — {query, sources, result_count}
* ``trigger.added`` / ``trigger.fired`` / ``trigger.removed`` /
  ``trigger.enabled`` / ``trigger.disabled`` — {trigger_id, name, action}
* plus any other ``mission.*`` / ``artifact.*`` / ``session.*`` /
  ``task.*`` / ``document.*`` / ``browser.*`` / ``codews.*`` /
  ``wisdom.*`` / ``connector.*`` / ``datasci.*`` / ``plugin.*`` /
  ``mesh.*`` / ``sync.*`` / ``power.*`` / ``stream.*`` / ``search.*`` /
  ``trigger.*`` topics the siblings emit.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = ["Timeline", "TIMELINE_PATTERNS"]

_log = logging.getLogger(__name__)

#: Bus topic patterns the timeline subscribes to (async delivery).
TIMELINE_PATTERNS: tuple[str, ...] = (
    "mission.*", "artifact.*", "session.*", "task.*",
    "document.*", "browser.*", "codews.*",
    "wisdom.*", "connector.*", "datasci.*", "plugin.*", "mesh.*",
    "sync.*", "power.*", "stream.*", "search.*", "trigger.*",
)

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
    causation_id TEXT,
    correlation_id TEXT,
    actor TEXT NOT NULL DEFAULT '',
    data_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_event_log_ts ON event_log (ts);
CREATE INDEX IF NOT EXISTS idx_event_log_topic ON event_log (topic);
CREATE INDEX IF NOT EXISTS idx_event_log_mission ON event_log (mission_id);
CREATE INDEX IF NOT EXISTS idx_event_log_session ON event_log (session_id);
CREATE INDEX IF NOT EXISTS idx_event_log_correlation ON event_log (correlation_id);
"""

#: Optional envelope columns backfilled onto older event_log tables.
_ENVELOPE_COLUMNS = (
    ("causation_id", "TEXT"),
    ("correlation_id", "TEXT"),
    ("actor", "TEXT NOT NULL DEFAULT ''"),
)

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS event_log_fts
    USING fts5(event_id UNINDEXED, topic, data_json,
               tokenize='unicode61');
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
    "causation_id",
    "correlation_id",
    "actor",
    "data_json",
)


def _ensure_envelope_columns(conn: "sqlite3.Connection") -> None:
    """ALTER older event_log tables up to the envelope schema."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(event_log)")}
    for column, ddl in _ENVELOPE_COLUMNS:
        if column not in cols:
            conn.execute(f"ALTER TABLE event_log ADD COLUMN {column} {ddl}")


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


def _fmt_ts(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(float(ts or 0), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%SZ")
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def _event_summary(topic: str, data: dict[str, Any]) -> str:
    """One-line human summary of an event payload."""
    for key in ("summary", "detail", "note", "message", "step", "task",
                "name", "status", "tool", "verdict"):
        value = data.get(key)
        if value:
            return f"{key}={value}"
    items = [f"{k}={v}" for k, v in list(data.items())[:3]]
    return " ".join(items) if items else topic


class _DictEvent:
    """Minimal event-like shim for re-importing exported rows."""

    def __init__(self, *, event_id: Any, topic: str, ts: Any, source: str,
                 data: dict[str, Any]) -> None:
        self.event_id = event_id
        self.topic = topic
        self.ts = ts
        self.source = source
        self.data = data


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
        self._fts_ok = False
        with self._lock:
            self._conn.executescript(_SCHEMA)
            _ensure_envelope_columns(self._conn)
            self._conn.commit()
            try:
                self._conn.executescript(_FTS_SCHEMA)
                self._fts_ok = True
            except sqlite3.Error:  # noqa: BLE001 — FTS5 may be unavailable
                _log.debug("timeline: FTS5 unavailable, search() disabled",
                           exc_info=True)

    @property
    def fts_available(self) -> bool:
        return self._fts_ok

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
                    " mission_id, artifact_id, causation_id, correlation_id,"
                    " actor, data_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["event_id"],
                        row["topic"],
                        row["ts"],
                        row["source"],
                        row["session_id"],
                        row["project_id"],
                        row["mission_id"],
                        row["artifact_id"],
                        row["causation_id"],
                        row["correlation_id"],
                        row["actor"],
                        row["data_json"],
                    ),
                )
                if self._fts_ok:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO event_log_fts"
                        " (event_id, topic, data_json) VALUES (?, ?, ?)",
                        (row["event_id"], row["topic"], row["data_json"]),
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
            "causation_id": _str("causation_id"),
            "correlation_id": _str("correlation_id"),
            "actor": _str("actor") or "",
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

    # -- full-text search -------------------------------------------------
    def search(self, query: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """Full-text search over topics and payloads (FTS5).

        ``query`` uses FTS5 syntax (``"exact phrase"``, ``a OR b``,
        ``prefix*``).  Returns newest-first rows with a ``rank`` field.
        Raises RuntimeError when FTS5 is unavailable.
        """
        if not self._fts_ok:
            raise RuntimeError("FTS5 full-text search is unavailable")
        sql = (
            "SELECT " + ", ".join(f"e.{c}" for c in _ROW_COLUMNS)
            + ", f.rank AS fts_rank FROM event_log_fts f"
            " JOIN event_log e ON e.event_id = f.event_id"
            " WHERE event_log_fts MATCH ?"
            " ORDER BY f.rank, e.ts DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (query, max(1, int(limit)))).fetchall()
        out = [self._row_to_dict(r) for r in rows]
        for row in out:
            row.pop("fts_rank", None)
        return out

    def trace(self, correlation_id: str, *,
              limit: int = 500) -> list[dict[str, Any]]:
        """All events sharing a correlation id — the causal chain, oldest
        first.  Event-sourcing style: follow one operation end to end."""
        rows = self.query(limit=limit)
        rows = [r for r in rows if r.get("correlation_id") == correlation_id]
        return sorted(rows, key=lambda r: (r.get("ts") or 0.0,
                                           str(r.get("event_id") or "")))

    # -- retention --------------------------------------------------------
    def prune_older_than(self, cutoff: Any) -> int:
        """Delete events at or before ``cutoff`` (epoch/ISO/datetime).

        Returns the number of rows deleted.  The FTS index is rebuilt for
        the removed ids.
        """
        cutoff_ts = _coerce_ts(cutoff)
        if cutoff_ts is None:
            raise ValueError(f"cannot parse cutoff {cutoff!r}")
        with self._lock:
            if self._fts_ok:
                self._conn.execute(
                    "DELETE FROM event_log_fts WHERE event_id IN"
                    " (SELECT event_id FROM event_log WHERE ts <= ?)",
                    (cutoff_ts,))
            cur = self._conn.execute("DELETE FROM event_log WHERE ts <= ?",
                                     (cutoff_ts,))
            deleted = cur.rowcount or 0
            self._conn.commit()
        return deleted

    def prune_keep_latest(self, keep: int) -> int:
        """Keep only the newest ``keep`` events.  Returns rows deleted."""
        keep = max(0, int(keep))
        with self._lock:
            cur = self._conn.execute(
                "SELECT event_id FROM event_log ORDER BY ts DESC, rowid DESC"
                " LIMIT -1 OFFSET ?", (keep,))
            victims = [r[0] for r in cur.fetchall()]
            if not victims:
                return 0
            placeholders = ",".join("?" for _ in victims)
            if self._fts_ok:
                self._conn.execute(
                    f"DELETE FROM event_log_fts WHERE event_id IN"
                    f" ({placeholders})", victims)
            cur = self._conn.execute(
                f"DELETE FROM event_log WHERE event_id IN ({placeholders})",
                victims)
            deleted = cur.rowcount or 0
            self._conn.commit()
        return deleted

    # -- export / import --------------------------------------------------
    def export_jsonl(self, path: str | Path) -> int:
        """Dump every event as one JSON object per line.  Returns the count."""
        path = Path(path)
        rows = self.query(limit=10**9)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str)
                         + "\n")
        return len(rows)

    def import_jsonl(self, path: str | Path) -> int:
        """Re-import an :meth:`export_jsonl` dump.  Returns rows imported."""
        imported = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                data = row.get("data") or {}
                event = _DictEvent(
                    event_id=row.get("event_id"),
                    topic=row.get("topic", ""),
                    ts=row.get("ts", time.time()),
                    source=row.get("source", ""),
                    data={
                        **(data if isinstance(data, dict) else {}),
                        "session_id": row.get("session_id"),
                        "project_id": row.get("project_id"),
                        "mission_id": row.get("mission_id"),
                        "artifact_id": row.get("artifact_id"),
                        "causation_id": row.get("causation_id"),
                        "correlation_id": row.get("correlation_id"),
                        "actor": row.get("actor"),
                    },
                )
                if self.record(event):
                    imported += 1
        return imported

    # -- stats / presentation ---------------------------------------------
    def topic_counts(self, *, since: Any = None) -> list[tuple[str, int]]:
        """(topic, count) pairs, most frequent first, optionally since."""
        clauses: list[str] = []
        params: list[Any] = []
        since_ts = _coerce_ts(since)
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(since_ts)
        sql = ("SELECT topic, COUNT(*) FROM event_log"
               + (" WHERE " + " AND ".join(clauses) if clauses else "")
               + " GROUP BY topic ORDER BY COUNT(*) DESC")
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Size, time span, topic count, FTS availability."""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*), MIN(ts), MAX(ts) FROM event_log").fetchone()
            topics = self._conn.execute(
                "SELECT COUNT(DISTINCT topic) FROM event_log").fetchone()
        return {
            "events": int(total[0] or 0),
            "topics": int(topics[0] or 0),
            "span_from": total[1],
            "span_to": total[2],
            "fts_available": self._fts_ok,
        }

    def render(self, limit: int = 20) -> str:
        """Plain-text event feed, newest first."""
        rows = self.recent(limit)
        lines = [f"timeline — {len(rows)} shown"]
        for row in rows:
            ts = _fmt_ts(row.get("ts"))
            topic = str(row.get("topic") or "?")
            data = row.get("data") or {}
            summary = _event_summary(topic, data)
            lines.append(f"  [{ts}] {topic} — {summary}")
        return "\n".join(lines)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        out = {column: row[column] for column in _ROW_COLUMNS if column != "data_json"}
        raw = row["data_json"] or "{}"
        try:
            out["data"] = json.loads(raw)
        except (TypeError, ValueError):
            out["data"] = {"_raw": raw}
        return out
