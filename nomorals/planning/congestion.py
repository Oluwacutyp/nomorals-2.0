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

import logging
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

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


def _default_db() -> str:
    base = os.environ.get("NOMORALS_HOME", os.path.expanduser("~/.nomorals"))
    return os.path.join(base, "planning", "congestion.db")


def _profile_kind() -> str:
    """termux | laptop | workstation (cheap, never raises)."""
    try:
        from ...core.profiles import get_profile_kind
        return get_profile_kind() or "workstation"
    except Exception:  # noqa: BLE001
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

    def __init__(self, db_path: str = "", *, in_memory: bool = False) -> None:
        self._db: sqlite3.Connection | None = None
        self._events: dict[str, list[tuple[float, str]]] = {}  # resource_id → [(ts, kind)]
        try:
            if in_memory:
                self._db = sqlite3.connect(":memory:")
            else:
                path = db_path or _default_db()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS congestion_resources (
                       resource_id TEXT PRIMARY KEY, name TEXT, kind TEXT,
                       capacity INTEGER, alternates TEXT, created_at REAL)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS congestion_holds (
                       hold_id TEXT PRIMARY KEY, resource_id TEXT,
                       agent TEXT, acquired_at REAL)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS congestion_events (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       resource_id TEXT, kind TEXT, ts REAL)""")
            self._db.commit()
        except Exception:  # noqa: BLE001 — a bad DB path is an empty monitor
            _log.warning("congestion: db unavailable, running empty", exc_info=True)
            self._db = None

    # ── resources ─────────────────────────────────────────────────────

    def register(self, name: str, *, kind: str = "endpoint",
                 capacity: int = 4,
                 alternates: list[str] | None = None) -> Resource | None:
        """Register (or update) a contended resource. Never raises."""
        try:
            name = (name or "").strip()[:120]
            if not name or self._db is None:
                return None
            capacity = max(1, int(capacity or 4))
            kind = (kind or "endpoint").strip().lower()
            if kind not in ("api_key", "endpoint", "lock", "rate_limit"):
                kind = "endpoint"
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

    def _resolve(self, name: str) -> Resource | None:
        """Name or id → Resource. Never raises."""
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                """SELECT * FROM congestion_resources
                   WHERE name = ? OR resource_id = ?""",
                (name, name)).fetchone()
            if row is None:
                return None
            return Resource(
                resource_id=row["resource_id"], name=row["name"],
                kind=row["kind"], capacity=int(row["capacity"] or 1),
                alternates=[a for a in (row["alternates"] or "").split(",") if a],
                created_at=float(row["created_at"] or 0.0))
        except Exception:  # noqa: BLE001
            return None

    def get(self, name: str) -> Resource | None:
        return self._resolve(name)

    def list_resources(self) -> list[Resource]:
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                "SELECT * FROM congestion_resources ORDER BY name").fetchall()
            return [Resource(
                resource_id=r["resource_id"], name=r["name"], kind=r["kind"],
                capacity=int(r["capacity"] or 1),
                alternates=[a for a in (r["alternates"] or "").split(",") if a],
                created_at=float(r["created_at"] or 0.0)) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def register_alternate(self, name: str, alternate: str) -> bool:
        """When this resource is congested, route to the alternate. Never raises."""
        try:
            res = self._resolve(name)
            alt = self._resolve(alternate)
            if res is None or alt is None or self._db is None:
                return False
            alts = [a for a in res.alternates if a != alt.resource_id]
            alts.append(alt.resource_id)
            self._db.execute(
                "UPDATE congestion_resources SET alternates = ? WHERE resource_id = ?",
                (",".join(alts), res.resource_id))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ── holds ─────────────────────────────────────────────────────────

    def acquire(self, name: str, agent: str) -> Hold | None:
        """One agent takes the resource. Returns the hold (or None). Never raises."""
        try:
            res = self._resolve(name)
            if res is None or self._db is None:
                return None
            agent = (agent or "unknown").strip()[:80]
            hold = Hold(hold_id="hold_" + uuid.uuid4().hex[:8],
                        resource_id=res.resource_id, agent=agent,
                        acquired_at=time.time())
            self._db.execute(
                """INSERT INTO congestion_holds
                   (hold_id, resource_id, agent, acquired_at)
                   VALUES (?, ?, ?, ?)""",
                (hold.hold_id, hold.resource_id, hold.agent, hold.acquired_at))
            self._db.execute(
                "INSERT INTO congestion_events (resource_id, kind, ts) VALUES (?, ?, ?)",
                (res.resource_id, "acquire", hold.acquired_at))
            self._db.commit()
            return hold
        except Exception:  # noqa: BLE001
            _log.debug("congestion acquire failed", exc_info=True)
            return None

    def release(self, name: str, agent: str) -> bool:
        """One agent gives the resource back. Never raises."""
        try:
            res = self._resolve(name)
            if res is None or self._db is None:
                return False
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
        except Exception:  # noqa: BLE001
            return False

    def queue_depth(self, name: str) -> int:
        """Current in-flight holds on a resource. Never raises."""
        try:
            res = self._resolve(name)
            if res is None or self._db is None:
                return 0
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM congestion_holds WHERE resource_id = ?",
                (res.resource_id,)).fetchone()
            return int(row["n"] or 0)
        except Exception:  # noqa: BLE001
            return 0

    def status(self) -> list[dict[str, Any]]:
        """Per-resource snapshot: depth, capacity, utilization. Never raises."""
        try:
            out: list[dict[str, Any]] = []
            for res in self.list_resources():
                depth = self.queue_depth(res.name)
                util = depth / max(1, res.capacity)
                out.append({"name": res.name, "kind": res.kind,
                            "depth": depth, "capacity": res.capacity,
                            "utilization": round(util, 2),
                            "congested": util >= _JAM_THRESHOLD})
            return out
        except Exception:  # noqa: BLE001
            return []

    # ── prediction ────────────────────────────────────────────────────

    def _rates(self, resource_id: str, window_s: float = 300.0,
               now: float | None = None) -> tuple[float, float]:
        """(acquisitions/min, releases/min) over the window. Never raises."""
        try:
            now = now if now is not None else time.time()
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
        except Exception:  # noqa: BLE001
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
            depth = self.queue_depth(name)
            prob, eta = self.predict_jam(name, now=now)
            # Waiting agents add pressure: effective probability rises.
            pressure = agents_waiting / max(1, res.capacity)
            eff = min(0.98, prob + 0.3 * min(pressure, 1.0))

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
                                   f"jam risk {int(eff * 100)}%) — hold new work for ~60s."),
                        "probability": round(eff, 2), "eta_s": eta}
            if eff >= _STAGGER_PROB:
                stagger_n = max(2, min(8, int(math.ceil(depth / max(1, res.capacity))) + 1))
                return {"action": "stagger",
                        "detail": (f"'{name}' is warming up ({depth}/{res.capacity} holds) — "
                                   f"launch agents in waves of {stagger_n}, 10s apart."),
                        "probability": round(eff, 2), "eta_s": eta,
                        "wave_size": stagger_n, "wave_gap_s": 10}
            return {"action": "proceed",
                    "detail": f"'{name}' is clear ({depth}/{res.capacity} holds).",
                    "probability": round(eff, 2), "eta_s": eta}
        except Exception:  # noqa: BLE001
            _log.debug("advise failed", exc_info=True)
            return {"action": "proceed", "detail": "monitor unavailable — proceeding.",
                    "probability": 0.0, "eta_s": None}


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
    except Exception:  # noqa: BLE001
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
            lines.append(f"  {r['name']} ({r['kind']}): {r['depth']}/{r['capacity']} holds"
                         f" — {int(r['utilization'] * 100)}%{flag}")
        lines.append("")
        lines.append(CONGESTION_DISCLAIMER)
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return "🚦 monitor unavailable."


def _usage() -> str:
    return ("usage: /congestion status — per-resource queue depths\n"
            "       /congestion register <name> [kind=endpoint|api_key|lock|rate_limit] [capacity=4]\n"
            "       /congestion advise <resource> [agents=N] — back off / switch / stagger\n"
            "       /congestion predict <resource> — jam probability + ETA\n"
            "       /congestion alternate <resource> <alternate> — provider switching")


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

        return _usage()
    except Exception:  # noqa: BLE001
        return "🚦 congestion check failed safely — proceed with caution."
