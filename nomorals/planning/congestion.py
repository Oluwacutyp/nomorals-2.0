"""Multi-agent congestion prediction (Symbotic pattern) — agent-fleet infrastructure.

At scale, the bottleneck is fleet coordination, not agent capability: N
agents hammering the same API/endpoint/file. This module tracks per-resource
queue depth (API keys, endpoints, file locks, rate limits) and predicts jams
BEFORE they happen: back off, switch to a different provider, or stagger.

Pairing (from the build map):
- #3 (code-first tools): fewer steps = less contention.
- #7 (router metering): the router consults this before fan-out.
- #6 (orchestration): ``pre_fanout_check()`` is the orchestrator seam.

Profile gating:
- termux/mobile: contention logic runs (acquire/release/depth/advise),
  prediction is a cheap queue-depth heuristic.
- laptop/workstation: full prediction model (arrival-rate vs service-rate
  projection with EMA smoothing).

Every function never raises: a dead monitor degrades to "proceed", never a
crash and never a fabricated jam.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator
from .failures import record_failure

_log = logging.getLogger("nomorals.planning.congestion")

CONGESTION_DISCLAIMER = (
    "🚦 congestion advice is a prediction from observed queue depth, "
    "not a guarantee — when in doubt, stagger."
)

#: How far ahead we project when predicting a jam (seconds).
_DEFAULT_HORIZON_S = 90.0
#: Queue depth (as a fraction of capacity) that counts as "congested".
_JAM_THRESHOLD = 0.85
#: Jam probability that triggers a "backoff" advisory.
_BACKOFF_PROB = 0.55
#: Jam probability that triggers a "stagger" advisory.
_STAGGER_PROB = 0.30
#: Default hold lease (seconds). A crashed agent's hold expires instead of
#: jamming the resource forever (mined: "a lock is a lease, not mutual
#: exclusion" — the TTL is the safety net against the dead owner).
_DEFAULT_HOLD_TTL_S = 600.0
#: Circuit-breaker tuning (mined: Resilience4j count-based window).
_CB_WINDOW = 10          # observations in the sliding window
_CB_MIN_CALLS = 5        # don't trip before this many observations
_CB_FAIL_THRESHOLD = 0.5  # open when >=50% of the window are jam signals
_CB_COOLDOWN_S = 60.0    # OPEN -> HALF-OPEN after this long
_CB_PROBES = 2           # successful probes to close from HALF-OPEN

#: The store's own DDL — the native path executes it and the seam
#: mapping is derived from it, so the two can never drift. The
#: in-file ALTER that adds expires_at to pre-lease files stays a
#: native-path concern; the seam DDL carries the final shape, and
#: imported pre-lease rows take expires_at from the column default.
_RESOURCES_DDL = """CREATE TABLE IF NOT EXISTS congestion_resources (
                       resource_id TEXT PRIMARY KEY, name TEXT, kind TEXT,
                       capacity INTEGER, alternates TEXT, created_at REAL)"""
_HOLDS_DDL = """CREATE TABLE IF NOT EXISTS congestion_holds (
                       hold_id TEXT PRIMARY KEY, resource_id TEXT,
                       agent TEXT, acquired_at REAL,
                       expires_at REAL DEFAULT 0)"""
_EVENTS_DDL = """CREATE TABLE IF NOT EXISTS congestion_events (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       resource_id TEXT, kind TEXT, ts REAL)"""
_BREAKERS_DDL = """CREATE TABLE IF NOT EXISTS congestion_breakers (
                       resource_id TEXT PRIMARY KEY, state TEXT,
                       window TEXT, consec_fail INTEGER DEFAULT 0,
                       probes_ok INTEGER DEFAULT 0,
                       opened_at REAL DEFAULT 0,
                       updated_at REAL DEFAULT 0)"""
_DDL = ";\n".join((_RESOURCES_DDL, _HOLDS_DDL, _EVENTS_DDL, _BREAKERS_DDL))
_TABLES = ("congestion_resources", "congestion_events",
           "congestion_holds", "congestion_breakers")


def _lease_live(expires_at: float | None, now: float) -> bool:
    """The native live-hold predicate, verbatim: a hold counts iff
    ``expires_at <= 0 OR expires_at > now`` — 0 (or NULL) means the
    lease never expires. Applied to the seam's data field
    (``sql_expires_at`` in the document body), never to the store's
    managed TTL."""
    exp = 0.0 if expires_at is None else float(expires_at)
    return exp <= 0 or exp > now


def _default_db() -> str:
    base = os.environ.get("NOMORALS_HOME", os.path.expanduser("~/.nomorals"))
    return os.path.join(base, "planning", "congestion.db")


def _profile_kind() -> str:
    """termux | laptop | workstation (cheap, never raises)."""
    try:
        from ..core.profiles import get_profile_kind
        return get_profile_kind() or "workstation"
    except Exception as exc:  # noqa: BLE001
        record_failure("congestion.profile_kind", exc)
        return os.environ.get("NM_PROFILE", "workstation") or "workstation"


def _full_model() -> bool:
    """Full prediction model runs on laptop/workstation; termux gets the heuristic."""
    return _profile_kind() not in ("termux", "mobile", "embedded")


@dataclass
class Resource:
    """One contended resource: an API key, endpoint, file lock, rate limit."""

    resource_id: str = ""
    name: str = ""
    kind: str = "endpoint"  # api_key | endpoint | lock | rate_limit
    capacity: int = 4       # how many agents may hold it concurrently
    alternates: list[str] = field(default_factory=list)  # provider-switch targets
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"resource_id": self.resource_id, "name": self.name,
                "kind": self.kind, "capacity": self.capacity,
                "alternates": list(self.alternates), "created_at": self.created_at}


@dataclass
class Hold:
    """One agent's current hold on a resource."""

    hold_id: str = ""
    resource_id: str = ""
    agent: str = ""
    acquired_at: float = 0.0


class ContentionMonitor:
    """Tracks per-resource queue depth and predicts jams. Never raises.

    Usage::

        mon = ContentionMonitor()
        mon.register("groq-key-1", kind="api_key", capacity=10)
        mon.acquire("groq-key-1", "agent-7")
        prob, eta = mon.predict_jam("groq-key-1")
        advice = mon.advise("groq-key-1", agents_waiting=6)
        mon.release("groq-key-1", "agent-7")
    """

    # When a document store is resolved (Mongo program P5 — an explicit
    # ``store`` or the boot ambient), the same API runs over the seam
    # instead (registered in storage.sidecars as planning-congestion):
    # resources, holds and breakers are documents in
    # ``planning-congestion__congestion_resources`` / ``__congestion_holds``
    # / ``__congestion_breakers`` and events ride allocated ids in
    # ``__congestion_events``. A hold's ``expires_at`` is domain data
    # on the seam, never the store's managed TTL: it rides in the
    # document body as ``sql_expires_at`` and every seam write passes
    # managed ``expires_at=None``, so 0 still means "no expiry" and
    # an expired hold stays stored until reap_stale counts it and
    # emits its expire event. Multi-document write sets write the
    # state document first and the event document second (the
    # composition order the registry entry records). The file's
    # existing rows are imported once, through the same mapping, at
    # first construction.

    def __init__(self, db_path: str = "", *, in_memory: bool = False,
                 store: Any = None) -> None:
        self._db: sqlite3.Connection | None = None
        self._port: Any = None
        self._events: dict[str, list[tuple[float, str]]] = {}  # resource_id → [(ts, kind)]
        try:
            path = ":memory:" if in_memory else (db_path or _default_db())
            if path != ":memory:":
                os.makedirs(os.path.dirname(path), exist_ok=True)
            from ..storage.sidecars import sidecar_port

            port = sidecar_port(
                "planning-congestion", _DDL, _TABLES,
                None if path == ":memory:" else path, store=store)
            if port is not None:
                self._import_into(port)
                self._port = port
            else:
                self._db = sqlite3.connect(path)
                self._db.row_factory = sqlite3.Row
                self._db.executescript(_DDL)
                # Migration for DBs created before hold leases existed.
                try:
                    cols = [r[1] for r in self._db.execute(
                        "PRAGMA table_info(congestion_holds)").fetchall()]
                    if "expires_at" not in cols:
                        self._db.execute(
                            "ALTER TABLE congestion_holds "
                            "ADD COLUMN expires_at REAL DEFAULT 0")
                except Exception:  # noqa: BLE001
                    pass
                self._db.commit()
        except Exception:  # noqa: BLE001 — a bad DB path is an empty monitor
            _log.warning("congestion: db unavailable, running empty", exc_info=True)
            self._db = None
            self._port = None

    def _import_into(self, port: Any) -> None:
        """One-time lazy import of congestion.db into the seam.
        Resources, holds and breakers port as-is through their
        RowTables; the AUTOINCREMENT events keep their file ids via
        put_raw, and the events counter is seeded from the imported
        maximum so seam-allocated event ids continue the file's
        sequence instead of colliding with it. Runs through
        ensure_imported: one-time, idempotent — a live store is
        adopted, never overwritten."""
        max_event = [0]

        def _make(table: Any) -> Any:
            def write(rows: list[dict[str, Any]]) -> int:
                for row in rows:
                    table.put(row)
                return len(rows)
            return write

        def write_events(rows: list[dict[str, Any]]) -> int:
            for row in rows:
                eid = int(row["id"])
                max_event[0] = max(max_event[0], eid)
                port.put_raw("congestion_events", str(eid), {
                    "id": eid, "resource_id": row["resource_id"],
                    "kind": row["kind"], "ts": row["ts"]})
            return len(rows)

        def seed_counter(_counts: dict[str, int]) -> None:
            doc = port.store.get(port.meta_collection, "counters") or {}
            if int(doc.get("events") or 0) < max_event[0]:
                port.store.put(port.meta_collection, "counters",
                               {**doc, "events": max_event[0]})

        writers = {name: _make(port.tables[name]) for name in
                   ("congestion_resources", "congestion_holds",
                    "congestion_breakers")}
        writers["congestion_events"] = write_events
        port.ensure_imported(writers, after_import=seed_counter)

    def _log_event(self, resource_id: str, kind: str, ts: float) -> None:
        """Append one event row. Seam: an allocated-id raw document,
        written after the state document it describes. Native: the
        INSERT joins the caller's transaction — call sites keep
        their single commit, exactly as before."""
        if self._port is not None:
            eid = self._port.next_id("events")
            self._port.put_raw("congestion_events", str(eid), {
                "id": eid, "resource_id": resource_id, "kind": kind,
                "ts": ts})
            return
        if self._db is None:
            return
        self._db.execute(
            "INSERT INTO congestion_events (resource_id, kind, ts) "
            "VALUES (?, ?, ?)", (resource_id, kind, ts))

    # ── resources ─────────────────────────────────────────────────────

    def register(self, name: str, *, kind: str = "endpoint",
                 capacity: int = 4,
                 alternates: list[str] | None = None) -> Resource | None:
        """Register (or update) a contended resource. Never raises."""
        try:
            name = (name or "").strip()[:120]
            if not name or (self._db is None and self._port is None):
                return None
            capacity = max(1, int(capacity or 4))
            kind = (kind or "endpoint").strip().lower()
            if kind not in ("api_key", "endpoint", "lock", "rate_limit"):
                kind = "endpoint"
            if self._port is not None:
                table = self._port.tables["congestion_resources"]
                rows = table.find({"name": name})
                if rows:
                    table.put({
                        "resource_id": rows[0]["resource_id"],
                        "name": name, "kind": kind, "capacity": capacity,
                        "alternates": ",".join(alternates or []),
                        "created_at": rows[0]["created_at"]})
                else:
                    table.put({
                        "resource_id": "res_" + uuid.uuid4().hex[:8],
                        "name": name, "kind": kind, "capacity": capacity,
                        "alternates": ",".join(alternates or []),
                        "created_at": time.time()})
                return self.get(name)
            rid = "res_" + uuid.uuid4().hex[:8]
            row = self._db.execute(
                "SELECT resource_id FROM congestion_resources WHERE name = ?",
                (name,)).fetchone()
            if row:
                rid = row["resource_id"]
                self._db.execute(
                    """UPDATE congestion_resources SET kind = ?, capacity = ?,
                       alternates = ? WHERE resource_id = ?""",
                    (kind, capacity, ",".join(alternates or []), rid))
            else:
                self._db.execute(
                    """INSERT INTO congestion_resources
                       (resource_id, name, kind, capacity, alternates, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (rid, name, kind, capacity, ",".join(alternates or []),
                     time.time()))
            self._db.commit()
            return self.get(name)
        except Exception:  # noqa: BLE001
            _log.debug("congestion register failed", exc_info=True)
            return None

    @staticmethod
    def _to_resource(row: Any) -> Resource:
        """One stored row (sqlite Row or seam dict) → Resource."""
        return Resource(
            resource_id=row["resource_id"], name=row["name"],
            kind=row["kind"], capacity=int(row["capacity"] or 1),
            alternates=[a for a in (row["alternates"] or "").split(",") if a],
            created_at=float(row["created_at"] or 0.0))

    def _resolve(self, name: str) -> Resource | None:
        """Name or id → Resource. Never raises."""
        try:
            if self._port is not None:
                table = self._port.tables["congestion_resources"]
                rows = table.find({"name": name})
                if rows:
                    return self._to_resource(rows[0])
                row = table.get(name)
                return self._to_resource(row) if row else None
            if self._db is None:
                return None
            row = self._db.execute(
                """SELECT * FROM congestion_resources
                   WHERE name = ? OR resource_id = ?""",
                (name, name)).fetchone()
            if row is None:
                return None
            return self._to_resource(row)
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.resolve", exc)
            return None

    def get(self, name: str) -> Resource | None:
        return self._resolve(name)

    def list_resources(self) -> list[Resource]:
        try:
            if self._port is not None:
                rows = self._port.tables["congestion_resources"].all()
                rows.sort(key=lambda r: r["name"] or "")
                return [self._to_resource(r) for r in rows]
            if self._db is None:
                return []
            rows = self._db.execute(
                "SELECT * FROM congestion_resources ORDER BY name").fetchall()
            return [self._to_resource(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.list_resources", exc)
            return []

    def register_alternate(self, name: str, alternate: str) -> bool:
        """When this resource is congested, route to the alternate. Never raises."""
        try:
            res = self._resolve(name)
            alt = self._resolve(alternate)
            if res is None or alt is None or (
                    self._db is None and self._port is None):
                return False
            alts = [a for a in res.alternates if a != alt.resource_id]
            alts.append(alt.resource_id)
            if self._port is not None:
                self._port.tables["congestion_resources"].update(
                    res.resource_id, {"alternates": ",".join(alts)})
                return True
            self._db.execute(
                "UPDATE congestion_resources SET alternates = ? WHERE resource_id = ?",
                (",".join(alts), res.resource_id))
            self._db.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.register_alternate", exc)
            return False

    # ── holds ─────────────────────────────────────────────────────────

    def acquire(self, name: str, agent: str,
                ttl_s: float = _DEFAULT_HOLD_TTL_S) -> Hold | None:
        """One agent takes the resource. Returns the hold (or None).

        Holds are *leases*: ``ttl_s`` bounds how long the hold counts even
        if the agent crashes without releasing (mined distributed-lock
        pattern). ``ttl_s <= 0`` means "no expiry" — only for holds you
        manage yourself. Never raises.
        """
        try:
            res = self._resolve(name)
            if res is None or (self._db is None and self._port is None):
                return None
            agent = (agent or "unknown").strip()[:80]
            now = time.time()
            try:
                ttl = float(ttl_s)
            except (TypeError, ValueError):
                ttl = _DEFAULT_HOLD_TTL_S
            expires = now + ttl if ttl > 0 else 0.0
            hold = Hold(hold_id="hold_" + uuid.uuid4().hex[:8],
                        resource_id=res.resource_id, agent=agent,
                        acquired_at=now)
            if self._port is not None:
                # State document first, event document second — the
                # composition order the registry entry records. The
                # lease rides as document data (sql_expires_at via
                # the carry rename); RowTable.put passes the managed
                # expires_at as None, so the store TTL never fires.
                self._port.tables["congestion_holds"].put({
                    "hold_id": hold.hold_id,
                    "resource_id": hold.resource_id,
                    "agent": hold.agent, "acquired_at": now,
                    "expires_at": expires})
                self._log_event(res.resource_id, "acquire", now)
                return hold
            self._db.execute(
                """INSERT INTO congestion_holds
                   (hold_id, resource_id, agent, acquired_at, expires_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (hold.hold_id, hold.resource_id, hold.agent, now, expires))
            self._db.execute(
                "INSERT INTO congestion_events (resource_id, kind, ts) VALUES (?, ?, ?)",
                (res.resource_id, "acquire", now))
            self._db.commit()
            return hold
        except Exception:  # noqa: BLE001
            _log.debug("congestion acquire failed", exc_info=True)
            return None

    def reap_stale(self) -> int:
        """Drop expired holds (dead owners). Returns the count reaped.

        Call it before fan-out, or rely on ``queue_depth()`` which ignores
        expired holds anyway. Never raises.
        """
        try:
            if self._db is None and self._port is None:
                return 0
            now = time.time()
            if self._port is not None:
                # The native reapable predicate (expires_at > 0 AND
                # expires_at <= now) evaluated in Python over the
                # data field — an unreaped expired hold stays stored.
                holds = self._port.tables["congestion_holds"]
                stale = [r for r in holds.all()
                         if (r["expires_at"] or 0) > 0
                         and (r["expires_at"] or 0) <= now]
                for r in stale:
                    holds.delete(r["hold_id"])
                    self._log_event(r["resource_id"], "expire", now)
                return len(stale)
            rows = self._db.execute(
                "SELECT hold_id, resource_id FROM congestion_holds "
                "WHERE expires_at > 0 AND expires_at <= ?",
                (now,)).fetchall()
            for r in rows:
                self._db.execute("DELETE FROM congestion_holds WHERE hold_id = ?",
                                 (r["hold_id"],))
                self._db.execute(
                    "INSERT INTO congestion_events (resource_id, kind, ts) "
                    "VALUES (?, ?, ?)", (r["resource_id"], "expire", now))
            self._db.commit()
            return len(rows)
        except Exception:  # noqa: BLE001
            _log.debug("reap_stale failed", exc_info=True)
            return 0

    def release(self, name: str, agent: str) -> bool:
        """One agent gives the resource back. Never raises."""
        try:
            res = self._resolve(name)
            if res is None or (self._db is None and self._port is None):
                return False
            if self._port is not None:
                holds = self._port.tables["congestion_holds"]
                agent_norm = (agent or "").strip()[:80]
                rows = [r for r in holds.find(
                    {"resource_id": res.resource_id})
                    if r["agent"] == agent_norm]
                if not rows:
                    return False
                rows.sort(key=lambda r: (r["acquired_at"] or 0,
                                         r["hold_id"]))
                holds.delete(rows[0]["hold_id"])
                self._log_event(res.resource_id, "release", time.time())
                return True
            row = self._db.execute(
                """SELECT hold_id FROM congestion_holds
                   WHERE resource_id = ? AND agent = ? LIMIT 1""",
                (res.resource_id, (agent or "").strip()[:80])).fetchone()
            if row is None:
                return False
            self._db.execute("DELETE FROM congestion_holds WHERE hold_id = ?",
                             (row["hold_id"],))
            self._db.execute(
                "INSERT INTO congestion_events (resource_id, kind, ts) VALUES (?, ?, ?)",
                (res.resource_id, "release", time.time()))
            self._db.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.release", exc)
            return False

    def queue_depth(self, name: str, *, now: float | None = None) -> int:
        """Current *live* holds on a resource — expired leases don't count.
        Never raises."""
        try:
            res = self._resolve(name)
            if res is None or (self._db is None and self._port is None):
                return 0
            now = now if now is not None else time.time()
            if self._port is not None:
                rows = self._port.tables["congestion_holds"].find(
                    {"resource_id": res.resource_id})
                return sum(1 for r in rows
                           if _lease_live(r["expires_at"], now))
            row = self._db.execute(
                """SELECT COUNT(*) AS n FROM congestion_holds
                   WHERE resource_id = ?
                   AND (expires_at <= 0 OR expires_at > ?)""",
                (res.resource_id, now)).fetchone()
            return int(row["n"] or 0)
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.queue_depth", exc)
            return 0

    def oldest_hold_age(self, name: str, *, now: float | None = None) -> float | None:
        """Age in seconds of the oldest live hold (stale-hold visibility).
        None when no live holds. Never raises."""
        try:
            res = self._resolve(name)
            if res is None or (self._db is None and self._port is None):
                return None
            now = now if now is not None else time.time()
            if self._port is not None:
                rows = self._port.tables["congestion_holds"].find(
                    {"resource_id": res.resource_id})
                live = [float(r["acquired_at"]) for r in rows
                        if r["acquired_at"] is not None
                        and _lease_live(r["expires_at"], now)]
                if live and min(live):
                    return max(0.0, now - min(live))
                return None
            row = self._db.execute(
                """SELECT MIN(acquired_at) AS m FROM congestion_holds
                   WHERE resource_id = ?
                   AND (expires_at <= 0 OR expires_at > ?)""",
                (res.resource_id, now)).fetchone()
            if row and row["m"]:
                return max(0.0, now - float(row["m"]))
            return None
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.oldest_hold_age", exc)
            return None

    def wait_estimate(self, name: str, *, now: float | None = None) -> float | None:
        """Little's law: expected seconds until a slot frees.

        depth / service_rate, where the service rate comes from release
        events. None when the resource is clear or the rate is unknown.
        Never raises.
        """
        try:
            res = self._resolve(name)
            if res is None:
                return None
            now = now if now is not None else time.time()
            depth = self.queue_depth(name, now=now)
            if depth < res.capacity:
                return 0.0
            _, service_per_min = self._rates(res.resource_id, now=now)
            if service_per_min <= 0:
                return None
            over = depth - res.capacity + 1  # slots ahead of a newcomer
            return round(over / (service_per_min / 60.0), 1)
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.wait_estimate", exc)
            return None

    def status(self) -> list[dict[str, Any]]:
        """Per-resource snapshot: depth, capacity, utilization, breaker,
        oldest hold. Never raises."""
        try:
            out: list[dict[str, Any]] = []
            for res in self.list_resources():
                depth = self.queue_depth(res.name)
                util = depth / max(1, res.capacity)
                out.append({"name": res.name, "kind": res.kind,
                            "depth": depth, "capacity": res.capacity,
                            "utilization": round(util, 2),
                            "congested": util >= _JAM_THRESHOLD,
                            "breaker": self.breaker_state(res.name),
                            "oldest_hold_s": self.oldest_hold_age(res.name)})
            return out
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.status", exc)
            return []

    # ── circuit breaker (mined: Resilience4j state machine) ──────────

    def _breaker_row(self, resource_id: str) -> dict[str, Any]:
        try:
            if self._port is not None:
                row = self._port.tables["congestion_breakers"].get(
                    resource_id)
            elif self._db is None:
                return {}
            else:
                row = self._db.execute(
                    "SELECT * FROM congestion_breakers WHERE resource_id = ?",
                    (resource_id,)).fetchone()
            if row is None:
                return {"state": "closed", "window": [],
                        "consec_fail": 0, "probes_ok": 0, "opened_at": 0.0}
            try:
                window = json.loads(row["window"] or "[]")
            except (TypeError, ValueError):
                window = []
            return {"state": row["state"] or "closed", "window": window,
                    "consec_fail": int(row["consec_fail"] or 0),
                    "probes_ok": int(row["probes_ok"] or 0),
                    "opened_at": float(row["opened_at"] or 0.0)}
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.breaker_row", exc)
            return {"state": "closed", "window": [], "consec_fail": 0,
                    "probes_ok": 0, "opened_at": 0.0}

    def _breaker_save(self, resource_id: str, row: dict[str, Any]) -> None:
        try:
            if self._port is not None:
                self._port.tables["congestion_breakers"].put({
                    "resource_id": resource_id, "state": row["state"],
                    "window": json.dumps(row["window"][-_CB_WINDOW:]),
                    "consec_fail": int(row["consec_fail"]),
                    "probes_ok": int(row["probes_ok"]),
                    "opened_at": float(row["opened_at"]),
                    "updated_at": time.time()})
                return
            if self._db is None:
                return
            self._db.execute(
                """INSERT INTO congestion_breakers
                   (resource_id, state, window, consec_fail, probes_ok,
                    opened_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(resource_id) DO UPDATE SET
                   state=excluded.state, window=excluded.window,
                   consec_fail=excluded.consec_fail,
                   probes_ok=excluded.probes_ok,
                   opened_at=excluded.opened_at,
                   updated_at=excluded.updated_at""",
                (resource_id, row["state"], json.dumps(row["window"][-_CB_WINDOW:]),
                 int(row["consec_fail"]), int(row["probes_ok"]),
                 float(row["opened_at"]), time.time()))
            self._db.commit()
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.breaker_save", exc)

    def breaker_state(self, name: str, *, now: float | None = None) -> str:
        """closed | open | half_open. Handles the OPEN→HALF-OPEN time
        transition (no direct OPEN→CLOSED, per the pattern). Never raises."""
        try:
            res = self._resolve(name)
            if res is None or (self._db is None and self._port is None):
                return "closed"
            now = now if now is not None else time.time()
            row = self._breaker_row(res.resource_id)
            if row["state"] == "open" and now - row["opened_at"] >= _CB_COOLDOWN_S:
                row["state"] = "half_open"
                row["probes_ok"] = 0
                self._breaker_save(res.resource_id, row)
            return row["state"]
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.breaker_state", exc)
            return "closed"

    def _breaker_observe(self, resource_id: str, jammed: bool,
                         *, now: float | None = None) -> str:
        """Feed one jam/healthy observation into the breaker. Returns the
        new state. Never raises."""
        try:
            now = now if now is not None else time.time()
            row = self._breaker_row(resource_id)
            state = row["state"]
            if state == "half_open":
                if jammed:
                    row.update(state="open", opened_at=now, probes_ok=0,
                               consec_fail=row["consec_fail"] + 1)
                else:
                    row["probes_ok"] += 1
                    if row["probes_ok"] >= _CB_PROBES:
                        row.update(state="closed", window=[],
                                   consec_fail=0, probes_ok=0)
                self._breaker_save(resource_id, row)
                return row["state"]
            if state == "open":
                return "open"  # cooldown handles the transition
            window = list(row["window"]) + [bool(jammed)]
            window = window[-_CB_WINDOW:]
            row["window"] = window
            row["consec_fail"] = row["consec_fail"] + 1 if jammed else 0
            if (len(window) >= _CB_MIN_CALLS
                    and sum(window) / len(window) >= _CB_FAIL_THRESHOLD):
                row.update(state="open", opened_at=now)
            self._breaker_save(resource_id, row)
            return row["state"]
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.breaker_observe", exc)
            return "closed"

    # ── prediction ────────────────────────────────────────────────────

    def _rates(self, resource_id: str, window_s: float = 300.0,
               now: float | None = None) -> tuple[float, float]:
        """(acquisitions/min, releases/min) over the window. Never raises."""
        try:
            now = now if now is not None else time.time()
            if self._port is not None:
                rows = self._port.find_raw(
                    "congestion_events", {"resource_id": resource_id})
                cutoff = now - window_s
                acq_n = sum(1 for r in rows if r["kind"] == "acquire"
                            and float(r["ts"] or 0) >= cutoff)
                rel_n = sum(1 for r in rows if r["kind"] == "release"
                            and float(r["ts"] or 0) >= cutoff)
                mins = max(window_s / 60.0, 1e-6)
                return (acq_n / mins, rel_n / mins)
            if self._db is None:
                return 0.0, 0.0
            cutoff = now - window_s
            acq = self._db.execute(
                """SELECT COUNT(*) AS n FROM congestion_events
                   WHERE resource_id = ? AND kind = 'acquire' AND ts >= ?""",
                (resource_id, cutoff)).fetchone()
            rel = self._db.execute(
                """SELECT COUNT(*) AS n FROM congestion_events
                   WHERE resource_id = ? AND kind = 'release' AND ts >= ?""",
                (resource_id, cutoff)).fetchone()
            mins = max(window_s / 60.0, 1e-6)
            return (float(acq["n"] or 0) / mins, float(rel["n"] or 0) / mins)
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.rates", exc)
            return 0.0, 0.0

    def predict_jam(self, name: str, horizon_s: float = _DEFAULT_HORIZON_S,
                    *, now: float | None = None) -> tuple[float, float | None]:
        """``(probability, eta_seconds)`` — will this resource jam within the horizon?

        Never raises. Unknown resource → (0.0, None). On termux this is a
        cheap queue-depth heuristic; on laptop/workstation it projects
        arrival-rate vs service-rate with EMA smoothing.
        """
        try:
            res = self._resolve(name)
            if res is None:
                return 0.0, None
            now = now if now is not None else time.time()
            depth = self.queue_depth(name)
            util = depth / max(1, res.capacity)

            if not _full_model():
                # Heuristic: congested now → likely to stay congested.
                prob = min(0.95, util) if util >= _JAM_THRESHOLD else max(0.0, (util - 0.5) * 0.8)
                return round(prob, 2), (0.0 if prob >= _BACKOFF_PROB else None)

            arrival, service = self._rates(res.resource_id, now=now)
            # Base probability from current utilization (logistic-ish).
            base = 1.0 / (1.0 + math.exp(-8.0 * (util - _JAM_THRESHOLD + 0.15)))
            # Trend: arrival outpacing service pushes probability up.
            trend = 0.0
            if arrival > service and (arrival + service) > 0:
                trend = min(0.35, 0.35 * (arrival - service) / (arrival + 1e-6))
            elif service > arrival:
                trend = -min(0.25, 0.25 * (service - arrival) / (service + 1e-6))
            prob = max(0.0, min(0.98, base + trend))

            eta: float | None = None
            if arrival > service and prob >= _STAGGER_PROB:
                # Time until the queue crosses the jam threshold.
                jam_depth = _JAM_THRESHOLD * res.capacity
                net_rate_per_s = (arrival - service) / 60.0
                if net_rate_per_s > 0:
                    eta = max(0.0, (jam_depth - depth) / net_rate_per_s)
                    if eta > horizon_s:
                        # Jam is real but beyond the horizon — downgrade.
                        prob = max(0.0, prob - 0.25)
                        eta = None
            return round(prob, 2), (round(eta, 1) if eta is not None else None)
        except Exception:  # noqa: BLE001
            _log.debug("predict_jam failed", exc_info=True)
            return 0.0, None

    # ── advice ────────────────────────────────────────────────────────

    def advise(self, name: str, agents_waiting: int = 0,
               *, now: float | None = None) -> dict[str, Any]:
        """What should the orchestrator do before fan-out?

        Returns ``{"action": "proceed"|"backoff"|"switch"|"stagger", "detail": str,
        "probability": float, "eta_s": float|None}``. Never raises.
        """
        try:
            res = self._resolve(name)
            if res is None:
                return {"action": "proceed", "detail": f"unknown resource '{name}' — no contention data.",
                        "probability": 0.0, "eta_s": None}
            now = now if now is not None else time.time()
            # Circuit breaker first: an OPEN breaker fails fast without
            # re-deriving anything; HALF-OPEN admits a single probe wave.
            bstate = self.breaker_state(name, now=now)
            if bstate == "open":
                return {"action": "backoff",
                        "detail": (f"'{name}' circuit is OPEN — it has been "
                                   f"jamming repeatedly. Failing fast: hold new "
                                   f"work ~{_CB_COOLDOWN_S:.0f}s or switch provider."),
                        "probability": 0.95, "eta_s": None,
                        "breaker": "open"}
            depth = self.queue_depth(name, now=now)
            prob, eta = self.predict_jam(name, now=now)
            # Waiting agents add pressure: effective probability rises.
            pressure = agents_waiting / max(1, res.capacity)
            eff = min(0.98, prob + 0.3 * min(pressure, 1.0))

            # Feed the observation into the breaker (jammed / healthy only;
            # the middle band is neutral signal).
            if eff >= _BACKOFF_PROB:
                self._breaker_observe(res.resource_id, True, now=now)
            elif eff < _STAGGER_PROB:
                self._breaker_observe(res.resource_id, False, now=now)
            if bstate == "half_open":
                return {"action": "stagger",
                        "detail": (f"'{name}' circuit is HALF-OPEN — probing "
                                   f"recovery with 1 agent. If it jams, the "
                                   f"circuit re-opens."),
                        "probability": round(eff, 2), "eta_s": eta,
                        "wave_size": 1, "wave_gap_s": 10,
                        "breaker": "half_open"}

            wait_s = self.wait_estimate(name, now=now)
            wait_note = (f" expect ~{wait_s:.0f}s per slot (Little's law)."
                         if wait_s else "")

            if eff >= _BACKOFF_PROB:
                # Prefer switching to an uncongested alternate.
                for alt_id in res.alternates:
                    alt = self._resolve(alt_id)
                    if alt is None:
                        continue
                    alt_prob, _ = self.predict_jam(alt.name, now=now)
                    if alt_prob < _STAGGER_PROB:
                        return {"action": "switch",
                                "detail": (f"'{name}' is congested ({depth}/{res.capacity} holds, "
                                           f"jam risk {int(eff * 100)}%) — routing to '{alt.name}' instead."),
                                "probability": round(eff, 2), "eta_s": eta,
                                "alternate": alt.name}
                return {"action": "backoff",
                        "detail": (f"'{name}' is congested ({depth}/{res.capacity} holds, "
                                   f"jam risk {int(eff * 100)}%) — hold new work for ~60s."
                                   f"{wait_note}"),
                        "probability": round(eff, 2), "eta_s": eta,
                        "breaker": self.breaker_state(name, now=now)}
            if eff >= _STAGGER_PROB:
                stagger_n = max(2, min(8, int(math.ceil(depth / max(1, res.capacity))) + 1))
                return {"action": "stagger",
                        "detail": (f"'{name}' is warming up ({depth}/{res.capacity} holds) — "
                                   f"launch agents in waves of {stagger_n}, 10s apart."
                                   f"{wait_note}"),
                        "probability": round(eff, 2), "eta_s": eta,
                        "wave_size": stagger_n, "wave_gap_s": 10,
                        "breaker": self.breaker_state(name, now=now)}
            return {"action": "proceed",
                    "detail": f"'{name}' is clear ({depth}/{res.capacity} holds).",
                    "probability": round(eff, 2), "eta_s": eta,
                    "breaker": self.breaker_state(name, now=now)}
        except Exception:  # noqa: BLE001
            _log.debug("advise failed", exc_info=True)
            return {"action": "proceed", "detail": "monitor unavailable — proceeding.",
                    "probability": 0.0, "eta_s": None}


@contextlib.contextmanager
def hold(monitor: ContentionMonitor | None, name: str,
         agent: str, ttl_s: float = _DEFAULT_HOLD_TTL_S) -> Iterator[Hold | None]:
    """Acquire a resource for the duration of a ``with`` block.

    Fan-out code no longer hand-rolls try/finally::

        with hold(mon, "groq-key-1", "agent-7"):
            ...  # the hold is always released

    A None monitor or a failed acquire yields None (fail-open "proceed").
    The lease TTL still applies inside the block as a dead-owner safety net.
    Never raises.
    """
    h: Hold | None = None
    try:
        if monitor is not None:
            h = monitor.acquire(name, agent, ttl_s=ttl_s)
    except Exception as exc:  # noqa: BLE001
        record_failure("congestion.hold_acquire", exc)
        h = None
    try:
        yield h
    finally:
        try:
            if monitor is not None and h is not None:
                monitor.release(name, agent)
        except Exception as exc:  # noqa: BLE001
            record_failure("congestion.hold_release", exc)


def route_around(monitor: ContentionMonitor | None,
                 names: list[str],
                 agents_waiting: int = 0) -> dict[str, dict[str, Any]]:
    """Route work around congestion: per-resource routed assignment.

    For each resource in ``names`` this returns
    ``{"target": <name or alternate>, "action": <advise action>,
    "reason": <detail>, "probability": float}`` — the fan-out target is
    the uncongested alternate when advise() says "switch", else the
    resource itself.  This is the planning-layer "route around
    congestion" answer: congested provider -> failover provider,
    warming provider -> staggered waves.  Never raises.
    """
    out: dict[str, dict[str, Any]] = {}
    try:
        if monitor is None:
            return out
        for name in names or []:
            try:
                advice = monitor.advise(name, agents_waiting=agents_waiting)
            except Exception as exc:  # noqa: BLE001
                record_failure("congestion.route_around_resource", exc)
                advice = {"action": "proceed", "detail": "unavailable",
                          "probability": 0.0}
            target = name
            if advice.get("action") == "switch" and advice.get("alternate"):
                target = str(advice["alternate"])
            out[name] = {
                "target": target,
                "action": str(advice.get("action", "proceed")),
                "reason": str(advice.get("detail", "")),
                "probability": float(advice.get("probability", 0.0) or 0.0),
            }
            eta = advice.get("eta_s")
            if eta is not None:
                out[name]["eta_s"] = eta
            if advice.get("wave_size"):
                out[name]["wave_size"] = advice["wave_size"]
    except Exception as exc:  # noqa: BLE001
        record_failure("congestion.route_around", exc)
    return out


def pre_fanout_check(monitor: ContentionMonitor | None,
                     resources: list[str],
                     agents_waiting: int = 0) -> dict[str, dict[str, Any]]:
    """The orchestrator seam: consult the monitor before fan-out.

    Returns {resource_name: advise(...)}. A None/empty monitor → all "proceed".
    Never raises.
    """
    try:
        out: dict[str, dict[str, Any]] = {}
        if monitor is None:
            return out
        for name in resources or []:
            out[name] = monitor.advise(name, agents_waiting=agents_waiting)
        return out
    except Exception as exc:  # noqa: BLE001
        record_failure("congestion.pre_fanout_check", exc)
        return {}


def format_status(monitor: ContentionMonitor) -> str:
    """/congestion status — per-resource queue depths. Never raises."""
    try:
        rows = monitor.status()
        if not rows:
            return "🚦 no resources registered yet. /congestion register <name> [kind] [capacity]"
        lines = ["🚦 contention monitor:"]
        for r in rows:
            flag = " 🔴" if r["congested"] else ""
            cb = r.get("breaker", "closed")
            cb_flag = {"open": " ⛔", "half_open": " 🟡"}.get(cb, "")
            stale = ""
            age = r.get("oldest_hold_s")
            if age is not None and age > 300:
                stale = f" · oldest hold {age / 60:.0f}m old"
            lines.append(f"  {r['name']} ({r['kind']}): {r['depth']}/{r['capacity']} holds"
                         f" — {int(r['utilization'] * 100)}%{flag}"
                         f" · circuit {cb}{cb_flag}{stale}")
        lines.append("")
        lines.append(CONGESTION_DISCLAIMER)
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        record_failure("congestion.format_status", exc)
        return "🚦 monitor unavailable."


def _usage() -> str:
    return ("usage: /congestion status — per-resource queue depths\n"
            "       /congestion register <name> [kind=endpoint|api_key|lock|rate_limit] [capacity=4]\n"
            "       /congestion advise <resource> [agents=N] — back off / switch / stagger\n"
            "       /congestion predict <resource> — jam probability + ETA\n"
            "       /congestion alternate <resource> <alternate> — provider switching\n"
            "       /congestion breaker <resource> — circuit-breaker state\n"
            "       /congestion reap — drop expired holds (dead owners)")


def control_congestion(tail: str, *, monitor: ContentionMonitor | None = None,
                       db_path: str = "") -> str:
    """/congestion — owner-only chat entry. Never raises."""
    try:
        mon = monitor or ContentionMonitor(db_path=db_path)
        rest = (tail or "").strip()
        if not rest or rest == "status":
            return format_status(mon)
        parts = rest.split()
        cmd = parts[0].lower()

        if cmd == "register" and len(parts) >= 2:
            name = parts[1]
            kind = parts[2] if len(parts) >= 3 else "endpoint"
            try:
                capacity = int(parts[3]) if len(parts) >= 4 else 4
            except (ValueError, TypeError):
                capacity = 4
            res = mon.register(name, kind=kind, capacity=capacity)
            if res is None:
                return "couldn't register that resource."
            return (f"🚦 registered '{res.name}' ({res.kind}, capacity {res.capacity}).")

        if cmd == "advise" and len(parts) >= 2:
            agents_waiting = 0
            for p in parts[2:]:
                if p.startswith("agents="):
                    try:
                        agents_waiting = max(0, int(p.split("=", 1)[1]))
                    except (ValueError, TypeError):
                        agents_waiting = 0
            a = mon.advise(parts[1], agents_waiting=agents_waiting)
            out = f"🚦 {a['action'].upper()}: {a['detail']}"
            if a.get("eta_s") is not None:
                out += f" (jam in ~{a['eta_s']:.0f}s)"
            out += f"\n\n{CONGESTION_DISCLAIMER}"
            return out

        if cmd == "predict" and len(parts) >= 2:
            prob, eta = mon.predict_jam(parts[1])
            line = f"🚦 '{parts[1]}': jam probability {int(prob * 100)}%"
            if eta is not None:
                line += f", ETA ~{eta:.0f}s"
            return line + f"\n\n{CONGESTION_DISCLAIMER}"

        if cmd == "alternate" and len(parts) >= 3:
            ok = mon.register_alternate(parts[1], parts[2])
            return ("✅ failover registered." if ok
                    else "couldn't register the alternate (both must exist).")

        if cmd == "breaker" and len(parts) >= 2:
            state = mon.breaker_state(parts[1])
            icon = {"open": "⛔", "half_open": "🟡"}.get(state, "🟢")
            return (f"{icon} circuit for '{parts[1]}': {state.upper()}.\n\n"
                    f"{CONGESTION_DISCLAIMER}")

        if cmd == "reap":
            n = mon.reap_stale()
            return (f"🧹 reaped {n} expired hold(s) (dead owners)."
                    if n else "🧹 no expired holds — queues are clean.")

        return _usage()
    except Exception as exc:  # noqa: BLE001
        record_failure("congestion.control_congestion", exc)
        return "🚦 congestion check failed safely — proceed with caution."
