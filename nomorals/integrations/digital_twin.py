"""Home Digital Twin — persistent queryable model of the home.

The endgame of Phase 15: every entity state change (from #61's
Home Assistant WebSocket stream and #63's MQTT bridge) lands here in
SQLite. Devon reasons over it: "is anything unusual?", "what changed
since I left?", learned rhythms, anomaly detection.

Anomaly detection is statistical (baselines + deviations over the
home's own history) — never claimed as "intelligent". Rhythms are
honest about needing history: nothing is claimed until enough days
of data exist.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, field

_log = logging.getLogger("nomorals.integrations.digital_twin")

DEFAULT_DB = os.path.expanduser("~/.nomorals/home/twin.db")

# Minimum days of history before rhythms are claimed.
MIN_RHYTHM_DAYS = 7
# Anomaly thresholds.
POWER_SPIKE_SIGMA = 3.0
OFFLINE_GRACE_S = 15 * 60          # 15 min before "unexpectedly offline"
OCCUPANCY_WINDOW_S = 30 * 60       # motion in last 30 min → occupied
DAY_S = 86400


def _is_num(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


@dataclass
class StateEvent:
    entity_id: str
    state: str
    ts: float
    attributes: dict = field(default_factory=dict)


@dataclass
class Anomaly:
    kind: str          # power_spike | unexpected_offline | unusual_open
    entity_id: str
    message: str
    ts: float


@dataclass
class Rhythm:
    entity_id: str
    description: str   # e.g. "usually on 18:00–23:00"
    confidence: float   # 0..1 — fraction of observed days matching


class HomeTwin:
    """Persistent digital twin of the home. Owner-scoped.

    ``ingest()`` is called by the #61 HA WebSocket stream and the #63
    MQTT bridge on every state change. Everything else is queries over
    the recorded history.
    """

    def __init__(self, db_path: str = DEFAULT_DB, *, community: bool = False):
        if community:
            raise PermissionError("the home twin is owner-scoped")
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        # WAL: readers never block the ingest stream.
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
        except Exception:  # noqa: BLE001 - WAL is a nicety
            pass
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS states ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " entity_id TEXT NOT NULL,"
            " state TEXT NOT NULL,"
            " attributes TEXT NOT NULL DEFAULT '{}',"
            " ts REAL NOT NULL)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_states_entity_ts"
            " ON states (entity_id, ts)"
        )
        # Ditto-style desired state: what Devon WANTS the device to be.
        # Reported state (states table) vs desired state is the twin's
        # core contract — commands flow toward desired.
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS desired ("
            " entity_id TEXT PRIMARY KEY,"
            " state TEXT NOT NULL,"
            " set_by TEXT NOT NULL DEFAULT 'devon',"
            " ts REAL NOT NULL)"
        )
        # Entity metadata registry: device_class, unit, area — enriched
        # from HA attributes on ingest, overridable by the owner.
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS entity_meta ("
            " entity_id TEXT PRIMARY KEY,"
            " device_class TEXT NOT NULL DEFAULT '',"
            " unit TEXT NOT NULL DEFAULT '',"
            " area TEXT NOT NULL DEFAULT '',"
            " friendly_name TEXT NOT NULL DEFAULT '',"
            " updated REAL NOT NULL)"
        )
        self._db.commit()

    # ── ingestion ────────────────────────────────────────────────

    def ingest(self, entity_id: str, state: Any, *,
               attributes: dict | None = None,
               ts: float | None = None) -> bool:
        """Record one state change. Returns True when recorded.

        Never raises. Dedupes: identical consecutive states are not
        stored again.
        """
        try:
            eid = str(entity_id or "").strip().lower()
            if not eid:
                return False
            st = str(state)
            ts = ts if ts is not None else time.time()
            attrs = json.dumps(dict(attributes or {}))
            with self._lock:
                row = self._db.execute(
                    "SELECT state FROM states WHERE entity_id = ?"
                    " ORDER BY ts DESC LIMIT 1", (eid,)).fetchone()
                if row is not None and row[0] == st:
                    return False  # no change — don't store duplicates
                self._db.execute(
                    "INSERT INTO states (entity_id, state, attributes, ts)"
                    " VALUES (?, ?, ?, ?)", (eid, st, attrs, ts))
                self._db.commit()
            # Enrich the metadata registry from HA attributes (best-effort).
            try:
                attr = dict(attributes or {})
                self.register_entity(
                    eid,
                    device_class=str(attr.get("device_class") or ""),
                    unit=str(attr.get("unit_of_measurement") or ""),
                    friendly_name=str(attr.get("friendly_name") or ""))
            except Exception:  # noqa: BLE001
                pass
            return True
        except Exception:  # noqa: BLE001 — twin ingestion never breaks streams
            _log.debug("twin ingest failed", exc_info=True)
            return False

    def register_entity(self, entity_id: str, *,
                        device_class: str = "", unit: str = "",
                        area: str = "", friendly_name: str = "") -> None:
        """Upsert entity metadata (registry enrichment). Never raises.

        Non-destructive: empty values never overwrite known values —
        later ingests without attributes can't wipe the registry.
        """
        try:
            eid = str(entity_id or "").strip().lower()
            if not eid:
                return
            with self._lock:
                self._db.execute(
                    "INSERT INTO entity_meta (entity_id, device_class,"
                    " unit, area, friendly_name, updated)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT (entity_id) DO UPDATE SET"
                    " device_class = CASE WHEN excluded.device_class = ''"
                    "   THEN entity_meta.device_class"
                    "   ELSE excluded.device_class END,"
                    " unit = CASE WHEN excluded.unit = ''"
                    "   THEN entity_meta.unit ELSE excluded.unit END,"
                    " area = CASE WHEN excluded.area = ''"
                    "   THEN entity_meta.area ELSE excluded.area END,"
                    " friendly_name = CASE WHEN excluded.friendly_name = ''"
                    "   THEN entity_meta.friendly_name"
                    "   ELSE excluded.friendly_name END,"
                    " updated = excluded.updated",
                    (eid, device_class, unit, area, friendly_name,
                     time.time()))
                self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("twin register_entity failed", exc_info=True)

    def entity_meta(self, entity_id: str) -> dict:
        """Metadata for one entity ({} when unknown)."""
        eid = str(entity_id or "").strip().lower()
        with self._lock:
            row = self._db.execute(
                "SELECT device_class, unit, area, friendly_name"
                " FROM entity_meta WHERE entity_id = ?",
                (eid,)).fetchone()
        if not row:
            return {}
        return {"device_class": row[0], "unit": row[1], "area": row[2],
                "friendly_name": row[3]}

    # ── desired state (Ditto-style twin contract) ────────────────────

    def set_desired(self, entity_id: str, state: Any, *,
                    set_by: str = "devon") -> bool:
        """Record what the device SHOULD be (command intent).

        Returns True when the desired state differs from the last
        reported state — i.e. a command is actually needed.
        """
        try:
            eid = str(entity_id or "").strip().lower()
            if not eid:
                return False
            st = str(state)
            with self._lock:
                self._db.execute(
                    "INSERT INTO desired (entity_id, state, set_by, ts)"
                    " VALUES (?, ?, ?, ?)"
                    " ON CONFLICT (entity_id) DO UPDATE SET"
                    " state = excluded.state, set_by = excluded.set_by,"
                    " ts = excluded.ts",
                    (eid, st, set_by, time.time()))
                self._db.commit()
                row = self._db.execute(
                    "SELECT state FROM states WHERE entity_id = ?"
                    " ORDER BY ts DESC LIMIT 1", (eid,)).fetchone()
            reported = row[0] if row else None
            return reported != st
        except Exception:  # noqa: BLE001
            _log.debug("twin set_desired failed", exc_info=True)
            return False

    def desired_state(self) -> dict[str, dict]:
        """All desired states: {entity_id: {state, set_by, ts}}."""
        with self._lock:
            rows = self._db.execute(
                "SELECT entity_id, state, set_by, ts FROM desired").fetchall()
        return {eid: {"state": st, "set_by": by, "ts": ts}
                for eid, st, by, ts in rows}

    def drift(self, *, now: float | None = None) -> list[dict]:
        """Desired-vs-reported mismatches — commands not yet honored.

        Each item: {entity_id, desired, reported, age_s}. The safety
        gate / command layer consumes this.
        """
        now = now if now is not None else time.time()
        current = self.current_state(now=now)
        out = []
        for eid, want in self.desired_state().items():
            reported = current.get(eid)
            if reported is not None and reported != want["state"]:
                out.append({"entity_id": eid, "desired": want["state"],
                            "reported": reported,
                            "age_s": now - want["ts"],
                            "set_by": want["set_by"]})
        return out

    def clear_desired(self, entity_id: str) -> None:
        """Drop a desired state (device reached it or command revoked)."""
        try:
            with self._lock:
                self._db.execute("DELETE FROM desired WHERE entity_id = ?",
                                 (str(entity_id or "").strip().lower(),))
                self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("twin clear_desired failed", exc_info=True)

    # ── queries ──────────────────────────────────────────────────

    def query(self, entity_id: str, since: float,
              *, limit: int = 500) -> list[StateEvent]:
        """State-change history for one entity since ``since``."""
        eid = str(entity_id or "").strip().lower()
        with self._lock:
            rows = self._db.execute(
                "SELECT state, attributes, ts FROM states"
                " WHERE entity_id = ? AND ts >= ?"
                " ORDER BY ts ASC LIMIT ?", (eid, since, limit)).fetchall()
        out = []
        for state, attrs, ts in rows:
            try:
                ad = json.loads(attrs)
            except Exception:  # noqa: BLE001
                ad = {}
            out.append(StateEvent(entity_id=eid, state=state, ts=ts,
                                  attributes=ad))
        return out

    def changes_since(self, since: float, *, limit: int = 200) -> list[StateEvent]:
        """Everything that changed across the home since ``since``."""
        with self._lock:
            rows = self._db.execute(
                "SELECT entity_id, state, attributes, ts FROM states"
                " WHERE ts >= ? ORDER BY ts ASC LIMIT ?",
                (since, limit)).fetchall()
        out = []
        for eid, state, attrs, ts in rows:
            try:
                ad = json.loads(attrs)
            except Exception:  # noqa: BLE001
                ad = {}
            out.append(StateEvent(entity_id=eid, state=state, ts=ts,
                                  attributes=ad))
        return out

    def current_state(self, *, now: float | None = None) -> dict[str, str]:
        """Latest known state per entity as of ``now`` — the home snapshot."""
        with self._lock:
            if now is None:
                rows = self._db.execute(
                    "SELECT entity_id, state, MAX(ts) FROM states"
                    " GROUP BY entity_id").fetchall()
            else:
                rows = self._db.execute(
                    "SELECT entity_id, state, MAX(ts) FROM states"
                    " WHERE ts <= ? GROUP BY entity_id", (now,)).fetchall()
        return {eid: state for eid, state, _ in rows}

    def entity_ids(self) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT entity_id FROM states").fetchall()
        return sorted(r[0] for r in rows)

    def history_days(self) -> float:
        """How many days of history exist (for rhythm honesty)."""
        with self._lock:
            row = self._db.execute(
                "SELECT MIN(ts), MAX(ts) FROM states").fetchone()
        if not row or row[0] is None:
            return 0.0
        return max(0.0, (row[1] - row[0]) / DAY_S)

    # ── derived signals ──────────────────────────────────────────

    def occupancy(self, *, now: float | None = None) -> dict:
        """Occupancy guess from motion/door patterns.

        Returns {"likely_home": bool, "basis": str}. Statistical only:
        recent motion/door activity suggests presence.
        """
        now = now if now is not None else time.time()
        window = now - OCCUPANCY_WINDOW_S
        with self._lock:
            rows = self._db.execute(
                "SELECT entity_id, state, ts FROM states"
                " WHERE ts >= ? ORDER BY ts DESC", (window,)).fetchall()
        for eid, state, ts in rows:
            if eid.startswith("binary_sensor.") and state in ("on", "motion"):
                return {"likely_home": True,
                        "basis": f"motion detected ({eid}) in the last "
                                 f"{OCCUPANCY_WINDOW_S // 60:.0f} minutes"}
            if eid.startswith(("binary_sensor.door", "binary_sensor.window",
                                "cover.", "lock.")) and state in (
                                    "on", "open", "unlocked", "opening"):
                return {"likely_home": True,
                        "basis": f"door/window activity ({eid}) in the last "
                                 f"{OCCUPANCY_WINDOW_S // 60:.0f} minutes"}
        return {"likely_home": False,
                "basis": "no motion or door activity in the last "
                         f"{OCCUPANCY_WINDOW_S // 60:.0f} minutes"}

    def energy_baseline(self, entity_id: str, *, days: int = 14,
                        now: float | None = None) -> dict | None:
        """Mean/std of numeric (power) readings — the baseline spikes
        are measured against. None when there's not enough numeric data."""
        now = now if now is not None else time.time()
        values = []
        for ev in self.query(entity_id, now - days * DAY_S, limit=5000):
            try:
                values.append(float(ev.state))
            except (TypeError, ValueError):
                continue
        if len(values) < 10:
            return None
        mean = statistics.fmean(values)
        stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
        return {"mean": mean, "stdev": stdev, "n": len(values),
                "unit": "as-reported"}

    def energy_today(self, *, now: float | None = None) -> dict:
        """Energy rollup for power/energy sensors: today's kWh estimate.

        Sums power-sensor readings (W) as average-W × elapsed hours.
        Statistical estimate — labeled as such, never a meter reading.
        """
        now = now if now is not None else time.time()
        day_start = now - (now % DAY_S)
        with self._lock:
            rows = self._db.execute(
                "SELECT entity_id, state, ts FROM states"
                " WHERE ts >= ? ORDER BY entity_id, ts",
                (day_start,)).fetchall()
        per_entity: dict[str, list[tuple[float, float]]] = {}
        for eid, state, ts in rows:
            try:
                per_entity.setdefault(eid, []).append((ts, float(state)))
            except (TypeError, ValueError):
                continue
        total_wh = 0.0
        contributors: list[dict] = []
        for eid, pts in per_entity.items():
            if len(pts) < 2:
                continue
            # average watts × elapsed hours
            watts = sum(v for _, v in pts) / len(pts)
            hours = (pts[-1][0] - pts[0][0]) / 3600
            if hours <= 0:
                continue
            wh = watts * hours
            total_wh += wh
            meta = self.entity_meta(eid)
            contributors.append({
                "entity_id": eid,
                "name": meta.get("friendly_name") or
                eid.split(".", 1)[-1].replace("_", " "),
                "wh": round(wh, 1),
                "avg_w": round(watts, 1)})
        contributors.sort(key=lambda c: c["wh"], reverse=True)
        return {"kwh": round(total_wh / 1000, 3),
                "estimated": True,
                "contributors": contributors[:10]}

    def predict(self, entity_id: str, horizon_min: float = 60,
                *, now: float | None = None) -> dict | None:
        """Honest linear extrapolation of a numeric sensor.

        Fits a least-squares line through today's readings and
        projects ``horizon_min`` ahead. Returns None with a reason
        when the data can't support it. Statistical only — labeled
        as a trend projection, never a prediction claim.
        """
        now = now if now is not None else time.time()
        pts = [(ev.ts, float(ev.state))
               for ev in self.query(entity_id, now - DAY_S, limit=2000)
               if _is_num(ev.state)]
        if len(pts) < 10:
            return None
        n = len(pts)
        sx = sum(t for t, _ in pts)
        sy = sum(v for _, v in pts)
        sxx = sum(t * t for t, _ in pts)
        sxy = sum(t * v for t, v in pts)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-9:
            return None
        slope = (n * sxy - sx * sy) / denom  # units per second
        intercept = (sy - slope * sx) / n
        future = now + horizon_min * 60
        return {"entity_id": entity_id,
                "in_min": horizon_min,
                "projected": round(slope * future + intercept, 2),
                "trend_per_hour": round(slope * 3600, 3),
                "n_points": n,
                "method": "linear-trend-projection"}

    # ── retention / export ───────────────────────────────────────────

    def prune(self, keep_days: float = 90) -> int:
        """Delete state history older than ``keep_days``. Returns rows
        removed. The twin keeps 90 days by default — rhythms need 7+."""
        cutoff = time.time() - keep_days * DAY_S
        with self._lock:
            cur = self._db.execute("DELETE FROM states WHERE ts < ?",
                                   (cutoff,))
            removed = cur.rowcount
            self._db.commit()
            try:
                self._db.execute("VACUUM")
                self._db.commit()
            except Exception:  # noqa: BLE001 - vacuum is a nicety
                pass
        _log.info("twin pruned %d rows older than %.0f days",
                  removed, keep_days)
        return removed

    def export_json(self) -> dict:
        """Full twin export: states count, desired, metadata, rhythms."""
        with self._lock:
            n_states = self._db.execute(
                "SELECT COUNT(*) FROM states").fetchone()[0]
        return {"exported_at": time.time(),
                "history_days": round(self.history_days(), 2),
                "state_rows": n_states,
                "entities": self.entity_ids(),
                "desired": self.desired_state(),
                "drift": self.drift(),
                "rhythms": [{"entity_id": r.entity_id,
                             "description": r.description,
                             "confidence": r.confidence}
                            for r in self.rhythms()],
                "energy_today": self.energy_today()}

    def rhythms(self, *, min_days: int = MIN_RHYTHM_DAYS,
                now: float | None = None) -> list[Rhythm]:
        """Learned on/off rhythms per entity by hour of day.

        Honest: returns [] until ``min_days`` of history exist.
        """
        now = now if now is not None else time.time()
        if self.history_days() < min_days:
            return []
        found: list[Rhythm] = []
        for eid in self.entity_ids():
            domain = eid.split(".", 1)[0]
            if domain not in ("light", "switch", "fan"):
                continue
            events = self.query(eid, now - min_days * DAY_S, limit=5000)
            if not events:
                continue
            # Bucket "on" hours per day.
            on_hours: dict[int, set[int]] = {}  # hour -> set of day-index
            days_seen: set[int] = set()
            for ev in events:
                day = int(ev.ts // DAY_S)
                days_seen.add(day)
                if ev.state == "on":
                    on_hours.setdefault(
                        time.localtime(ev.ts).tm_hour, set()).add(day)
            if len(days_seen) < min_days:
                continue
            # Hours where it's on on most days.
            strong = sorted(h for h, ds in on_hours.items()
                            if len(ds) >= 0.7 * len(days_seen))
            if not strong:
                continue
            # Merge contiguous hours into ranges.
            ranges = []
            start = prev = strong[0]
            for h in strong[1:]:
                if h == prev + 1:
                    prev = h
                else:
                    ranges.append((start, prev))
                    start = prev = h
            ranges.append((start, prev))
            desc = ", ".join(f"{s:02d}:00–{e + 1:02d}:00" for s, e in ranges)
            conf = sum(len(on_hours[h]) for h in strong) / (
                len(strong) * len(days_seen))
            found.append(Rhythm(entity_id=eid,
                                description=f"usually on {desc}",
                                confidence=round(conf, 2)))
        return found

    # ── anomalies ────────────────────────────────────────────────

    def anomalies(self, *, now: float | None = None) -> list[Anomaly]:
        """Statistical deviations from the home's own history.

        Power spike (>3σ vs 14-day baseline), device offline
        unexpectedly, door/cover open at an unusual hour.
        """
        now = now if now is not None else time.time()
        out: list[Anomaly] = []
        current = self.current_state(now=now)

        for eid, state in current.items():
            domain = eid.split(".", 1)[0]

            # — unexpected offline —
            if state in ("unavailable", "unknown", "offline"):
                last_ok = self._last_state_before(
                    eid, state, now - OFFLINE_GRACE_S)
                if last_ok is not None:
                    out.append(Anomaly(
                        kind="unexpected_offline", entity_id=eid, ts=now,
                        message=f"{eid} went offline — it was reporting "
                                f"normally {int((now - last_ok) / 60)} minutes "
                                "ago."))

            # — power spike —
            if domain == "sensor":
                try:
                    val = float(state)
                except (TypeError, ValueError):
                    continue
                base = self.energy_baseline(eid, now=now)
                if base and base["stdev"] > 0:
                    z = (val - base["mean"]) / base["stdev"]
                    if z >= POWER_SPIKE_SIGMA:
                        out.append(Anomaly(
                            kind="power_spike", entity_id=eid, ts=now,
                            message=f"{eid} reads {val:.0f} — "
                                    f"{z:.1f}× above its 14-day baseline "
                                    f"(avg {base['mean']:.0f})."))

            # — door/cover open at an unusual hour —
            if domain in ("cover", "lock", "binary_sensor") and state in (
                    "open", "on", "unlocked"):
                state_set_at = self._state_set_at(eid, state, now)
                if self._unusual_hour_for(eid, state, now,
                                          baseline_before=state_set_at):
                    hour = time.localtime(now).tm_hour
                    out.append(Anomaly(
                        kind="unusual_open", entity_id=eid, ts=now,
                        message=f"{eid} is {state} at {hour:02d}:00 — "
                                "unusual for this hour based on past weeks."))
        return out

    def _state_set_at(self, entity_id: str, state: str,
                      now: float) -> float:
        """When the entity last entered ``state`` (≤ now)."""
        with self._lock:
            row = self._db.execute(
                "SELECT MAX(ts) FROM states WHERE entity_id = ?"
                " AND state = ? AND ts <= ?",
                (entity_id, state, now)).fetchone()
        return row[0] if row and row[0] else now

    def _last_state_before(self, entity_id: str, exclude: str,
                           before: float) -> float | None:
        """Latest timestamp where the entity was NOT in ``exclude`` state."""
        with self._lock:
            row = self._db.execute(
                "SELECT MAX(ts) FROM states WHERE entity_id = ?"
                " AND state != ? AND ts <= ?",
                (entity_id, exclude, before)).fetchone()
        return row[0] if row else None

    def _unusual_hour_for(self, entity_id: str, state: str,
                         now: float, *,
                         baseline_before: float | None = None) -> bool:
        """True when ``state`` at this hour is unusual for the entity.

        Two ways: (a) the state has never been seen in 3 weeks of
        history; (b) the state occurs at this hour far less often than
        the entity's normal active hours for that state.

        ``baseline_before``: the baseline is history strictly before
        this timestamp — the event being evaluated must not count
        toward "normal".
        """
        if self.history_days() < MIN_RHYTHM_DAYS:
            return False
        hour = time.localtime(now).tm_hour
        active_hours: set[int] = set()
        ever_total = 0
        ever_matching = 0
        # Baseline is history BEFORE now — the current event must not
        # count toward "normal".
        cutoff = baseline_before if baseline_before is not None else now
        for ev in self.query(entity_id, now - 21 * DAY_S, limit=20000):
            if ev.ts >= cutoff:
                continue
            ever_total += 1
            if ev.state == state:
                ever_matching += 1
                active_hours.add(time.localtime(ev.ts).tm_hour)
        # (a) never seen in this state in 3 weeks → unusual.
        if ever_total >= 20 and ever_matching <= 1:
            return True
        # (b) this hour is outside the state's normal active window.
        if active_hours and hour not in active_hours:
            # Allow adjacent hours (e.g. 17:00 window, event at 18:00).
            if not any(abs(hour - h) <= 1 or abs(hour - h) >= 23
                       for h in active_hours):
                return True
        return False

    # ── summary ──────────────────────────────────────────────────

    def summary(self, *, now: float | None = None) -> str:
        """'Your home right now' in plain language."""
        now = now if now is not None else time.time()
        current = self.current_state(now=now)
        if not current:
            return ("no home data yet — connect Home Assistant or MQTT and "
                    "I'll start building your home's picture.")
        lines = ["🏠 your home right now:"]
        by_domain: dict[str, list[str]] = {}
        for eid, state in sorted(current.items()):
            domain = eid.split(".", 1)[0]
            by_domain.setdefault(domain, []).append((eid, state))
        for domain in ("light", "switch", "climate", "cover", "lock",
                       "sensor", "binary_sensor"):
            for eid, state in by_domain.get(domain, []):
                name = eid.split(".", 1)[1].replace("_", " ")
                lines.append(f"• {name}: {state}")
        occ = self.occupancy(now=now)
        presence = ("likely home" if occ["likely_home"]
                    else "no recent activity")
        lines.append(f"• presence: {presence} ({occ['basis']})")
        anomalies = self.anomalies(now=now)
        if anomalies:
            lines.append("⚠️ unusual:")
            for a in anomalies[:5]:
                lines.append(f"  — {a.message}")
        if self.history_days() < MIN_RHYTHM_DAYS:
            lines.append(f"(learning your home's rhythms — "
                         f"{self.history_days():.1f}/{MIN_RHYTHM_DAYS} days "
                         "of history so far)")
        return "\n".join(lines)

    def format_briefing(self, *, now: float | None = None) -> str:
        """God-tier home briefing: sections, severity-ordered anomalies,
        energy, rhythms, drift. The 'good morning' render."""
        now = now if now is not None else time.time()
        current = self.current_state(now=now)
        if not current:
            return ("🏠 **home briefing**\n_no home data yet — connect Home "
                    "Assistant or MQTT and I'll start building your home's "
                    "picture._")
        lines = ["🏠 **home briefing**"]
        # — climate first —
        climate = [(e, s) for e, s in sorted(current.items())
                   if e.startswith("climate.")]
        for eid, state in climate:
            meta = self.entity_meta(eid)
            name = meta.get("friendly_name") or eid.split(".", 1)[-1]
            lines.append(f"🌡️ {name}: {state}")
        # — lights / switches on —
        on_devs = sorted(e for e, s in current.items()
                         if e.split(".", 1)[0] in ("light", "switch",
                                                   "fan") and s == "on")
        if on_devs:
            names = [self.entity_meta(e).get("friendly_name") or
                     e.split(".", 1)[-1].replace("_", " ") for e in on_devs]
            lines.append(f"💡 **{len(on_devs)} on:** {', '.join(names)}")
        else:
            lines.append("💡 all lights off")
        # — open / unlocked —
        open_devs = sorted(e for e, s in current.items()
                           if s in ("open", "unlocked", "on")
                           and e.split(".", 1)[0] in (
                               "cover", "lock", "binary_sensor"))
        for eid in open_devs:
            name = (self.entity_meta(eid).get("friendly_name")
                    or eid.split(".", 1)[-1].replace("_", " "))
            lines.append(f"🚪 {name}: {current[eid]}")
        # — presence —
        occ = self.occupancy(now=now)
        lines.append(f"🧍 presence: "
                     f"{'likely home' if occ['likely_home'] else 'away/quiet'}"
                     f" — {occ['basis']}")
        # — energy —
        energy = self.energy_today(now=now)
        if energy["contributors"]:
            top = energy["contributors"][0]
            lines.append(f"⚡ ~{energy['kwh']} kWh today (est.) — "
                         f"biggest: {top['name']} ({top['wh']} Wh)")
        # — anomalies, severity-ordered —
        anomalies = self.anomalies(now=now)
        if anomalies:
            lines.append("\n⚠️ **needs attention:**")
            for a in anomalies[:5]:
                icon = {"power_spike": "⚡", "unexpected_offline": "📡",
                        "unusual_open": "🚪"}.get(a.kind, "•")
                lines.append(f"  {icon} {a.message}")
        # — desired-state drift —
        drift = self.drift(now=now)
        if drift:
            lines.append("\n🎯 **pending commands:**")
            for d in drift[:5]:
                lines.append(f"  → {d['entity_id']}: want "
                             f"{d['desired']}, still {d['reported']}")
        # — rhythms —
        rhythms = self.rhythms(now=now)
        if rhythms:
            lines.append("\n🔁 **rhythms I've learned:**")
            for r in rhythms[:5]:
                name = r.entity_id.split(".", 1)[-1].replace("_", " ")
                lines.append(f"  • {name} {r.description}"
                             f" ({r.confidence:.0%} of days)")
        elif self.history_days() < MIN_RHYTHM_DAYS:
            lines.append(f"\n_(learning rhythms — "
                         f"{self.history_days():.1f}/{MIN_RHYTHM_DAYS} days)_")
        return "\n".join(lines)

    def what_changed(self, since: float, *, limit: int = 30) -> str:
        """'What changed since I left?' in plain language."""
        events = self.changes_since(since, limit=limit)
        if not events:
            return "nothing changed since then."
        lines = [f"{len(events)} change(s) since then:"]
        for ev in events[-limit:]:
            t = time.strftime("%H:%M", time.localtime(ev.ts))
            name = ev.entity_id.split(".", 1)[-1].replace("_", " ")
            lines.append(f"• {t} — {name} → {ev.state}")
        return "\n".join(lines)

    def close(self) -> None:
        with self._lock:
            self._db.close()


# ── stream wiring ────────────────────────────────────────────────

def attach_twin_to_ha(stream: Any, twin: "HomeTwin") -> None:
    """Route every HA state_changed event into the twin."""
    def _on_event(entity_id: str, old: Any, new: Any,
                  attributes: dict | None = None) -> None:
        twin.ingest(entity_id, new, attributes=attributes)
    try:
        stream.subscribe_state_changed(callback=_on_event)
    except Exception:  # noqa: BLE001
        _log.debug("could not attach twin to HA stream", exc_info=True)


def attach_twin_to_mqtt(bridge: Any, twin: "HomeTwin") -> None:
    """Route every MQTT device update into the twin."""
    try:
        from .mqtt_client import _entity_id as _z2m_entity
    except Exception:  # noqa: BLE001
        _z2m_entity = None  # type: ignore[assignment]

    def _on_message(topic: str, payload: Any) -> None:
        try:
            device = topic.split("/")[1] if "/" in topic else topic
            eid = (_z2m_entity(device) if _z2m_entity
                   else f"zigbee2mqtt.{device}")
            if isinstance(payload, dict):
                state = payload.get("state", json.dumps(payload))
            else:
                state = str(payload)
            twin.ingest(eid, state)
        except Exception:  # noqa: BLE001
            _log.debug("twin mqtt ingest failed", exc_info=True)

    try:
        bridge.subscribe("zigbee2mqtt/+", _on_message)
    except Exception:  # noqa: BLE001
        _log.debug("could not attach twin to MQTT bridge", exc_info=True)


__all__ = ["HomeTwin", "StateEvent", "Anomaly", "Rhythm",
           "attach_twin_to_ha", "attach_twin_to_mqtt",
           "MIN_RHYTHM_DAYS"]
