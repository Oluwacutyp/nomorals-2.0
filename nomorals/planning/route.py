"""Predict → Build → Solve routing pipeline (scheduling science, not heuristics).

Build-map #90 — logistics & dispatch intelligence.  Any Devon sequencing
problem (errands, background-job batching, multi-stop plans) goes through
three stages:

1. **Predict** — ``CostModel``: learned travel-time/cost estimates per
   segment.  The mined insight: *cost-function accuracy dominates solver
   choice*, so the model learns from recorded actuals and starts honest
   when it knows nothing.
2. **Build** — ``GeoIndex``: H3-cell location work (Uber pattern).  Stops
   are indexed into cells; nearby lookups scan cells, never raw lat/lng
   pairs.  Uses the ``h3`` library when installed, otherwise a quantized
   grid fallback with the same API.
3. **Solve** — ``RouteSolver``: pluggable backends, profile-gated:
   greedy (termux default, always available) → OR-Tools (laptop,
   if installed) → gpu (selects the best available backend; there is no
   GPU-native TSP solver, and this module says so instead of pretending).

First use: "I need to do these 5 things this afternoon" → optimal order
with honest times (pairs with #91's prediction intervals).

Every public method never raises.  Honest times: estimates are labeled
``learned`` (from your actuals) or ``heuristic`` (distance/speed guess).
"""

from __future__ import annotations

import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Stop",
    "Leg",
    "RouteResult",
    "CostModel",
    "GeoIndex",
    "RouteSolver",
    "RoutePlanner",
    "control_route",
    "cell_for",
    "h3_available",
    "ROUTE_DISCLAIMER",
]

ROUTE_DISCLAIMER = (
    "Travel times are estimates from your recorded trips and distance "
    "heuristics, not live traffic. Verify consequential plans yourself."
)

# ── H3 cells (Uber pattern; never raw lat/lng scans) ────────────────────

try:  # pragma: no cover - environment dependent
    import h3 as _h3  # type: ignore
except Exception:  # noqa: BLE001 - h3 is an optional dependency
    _h3 = None


def h3_available() -> bool:
    """True when the real ``h3`` library is installed."""
    return _h3 is not None


def cell_for(lat: float, lng: float, resolution: int = 8) -> str:
    """Location → cell id.  h3 when available, quantized grid otherwise.

    Never raises; returns "" for bad input.
    """
    try:
        lat, lng = float(lat), float(lng)
        resolution = max(0, min(15, int(resolution)))
        if _h3 is not None:
            return str(_h3.geo_to_h3(lat, lng, resolution))
        # Grid fallback: quantize to ~1.1 km cells at res 8 (~0.01°).
        step = 0.01 * (2 ** max(0, 8 - resolution))
        return f"g{resolution}:{int(lat / step)}:{int(lng / step)}"
    except (TypeError, ValueError):
        return ""


def _cell_center(cell: str) -> tuple[float, float] | None:
    """Rough center of a fallback grid cell (h3 centers need the lib)."""
    try:
        parts = cell.split(":")
        if len(parts) != 3 or not parts[0].startswith("g"):
            return None
        res, ix, iy = int(parts[0][1:]), int(parts[1]), int(parts[2])
        step = 0.01 * (2 ** max(0, 8 - res))
        return (ix * step + step / 2, iy * step + step / 2)
    except (ValueError, IndexError):
        return None


def haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    """Great-circle distance in km. Never raises."""
    try:
        r = 6371.0
        dlat = math.radians(float(b_lat) - float(a_lat))
        dlng = math.radians(float(b_lng) - float(a_lng))
        s = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(float(a_lat)))
             * math.cos(math.radians(float(b_lat)))
             * math.sin(dlng / 2) ** 2)
        return 2 * r * math.asin(min(1.0, math.sqrt(s)))
    except (TypeError, ValueError):
        return 0.0


class GeoIndex:
    """Cell-based location index.  ``nearby()`` scans cells, never raw pairs."""

    def __init__(self) -> None:
        self._cells: dict[str, list[str]] = {}
        self._coords: dict[str, tuple[float, float]] = {}

    def add(self, stop_id: str, lat: float, lng: float,
            resolution: int = 8) -> str:
        """Index a stop; returns its cell id ("" on bad coords)."""
        try:
            cell = cell_for(lat, lng, resolution)
            if not cell:
                return ""
            sid = str(stop_id)
            self._cells.setdefault(cell, [])
            if sid not in self._cells[cell]:
                self._cells[cell].append(sid)
            self._coords[sid] = (float(lat), float(lng))
            return cell
        except Exception:  # noqa: BLE001 - never raises
            return ""

    def nearby(self, lat: float, lng: float,
               radius_km: float = 5.0) -> list[str]:
        """Stop ids within radius_km — scans only the local cell bucket(s)."""
        try:
            cell = cell_for(lat, lng)
            if not cell:
                return []
            # Cell-bucket scan first; exact distance check second.
            candidates: list[str] = []
            if _h3 is not None and not cell.startswith("g"):
                # Real h3: expand the query cell with k-rings so we only
                # touch buckets that can possibly be in range.
                try:
                    edge_km = {15: 0.0009, 14: 0.0018, 13: 0.0037,
                               12: 0.0074, 11: 0.0148, 10: 0.0296,
                               9: 0.059, 8: 0.12, 7: 0.24,
                               6: 0.48, 5: 0.96, 4: 1.9,
                               3: 3.8, 2: 7.7, 1: 15.0, 0: 30.0}.get(8, 0.12)
                    rings = min(12, max(0, int(float(radius_km) / max(edge_km, 1e-6)) + 1))
                    ring_cells = _h3.k_ring(cell, rings)
                    for c in ring_cells:
                        candidates.extend(self._cells.get(str(c), []))
                except Exception:  # noqa: BLE001 - fall through to full scan
                    candidates = []
            if not candidates:
                for c, ids in self._cells.items():
                    if self._same_area(cell, c):
                        candidates.extend(ids)
            out = []
            for sid in candidates:
                alat, alng = self._coords.get(sid, (None, None))
                if alat is None:
                    continue
                if haversine_km(lat, lng, alat, alng) <= float(radius_km):
                    out.append(sid)
            return out
        except Exception:  # noqa: BLE001 - never raises
            return []

    @staticmethod
    def _same_area(a: str, b: str) -> bool:
        """Coarse cell match: same cell or (grid fallback) adjacent cells."""
        if a == b:
            return True
        try:
            pa, pb = a.split(":"), b.split(":")
            if len(pa) == 3 and len(pb) == 3 and pa[0] == pb[0]:
                return abs(int(pa[1]) - int(pb[1])) <= 1 and \
                    abs(int(pa[2]) - int(pb[2])) <= 1
        except (ValueError, IndexError):
            pass
        return False


# ── stops / legs / results ───────────────────────────────────────────────


@dataclass
class Stop:
    """One thing to do/visit.  Coordinates optional — name-only stops work."""
    stop_id: str = ""
    label: str = ""
    lat: float | None = None
    lng: float | None = None
    dwell_minutes: float = 15.0  # how long you spend there
    window: tuple[float, float] | None = None  # (start, end) epoch; optional

    def has_coords(self) -> bool:
        return self.lat is not None and self.lng is not None


@dataclass
class Leg:
    from_id: str
    to_id: str
    minutes: float
    cost_kobo: int
    km: float = 0.0
    source: str = "heuristic"  # learned | heuristic


@dataclass
class RouteResult:
    order: list[Stop]
    legs: list[Leg]  # legs[i] gets you to order[i+1]; dwell folded into plan
    total_minutes: float
    total_cost_kobo: int
    backend: str
    arrivals: list[float] = field(default_factory=list)  # epoch per stop
    learned_legs: int = 0
    #: human-readable infeasibility notes, e.g. "bank: arrives 17:40 after
    #: window closes 17:00".  Empty when every window is respected.
    window_violations: list[str] = field(default_factory=list)


# ── stage 1: predict — the learned cost model ────────────────────────────

_DEFAULT_SPEED_KMH = 32.0      # Lagos urban default
_DEFAULT_COST_PER_KM_KOBO = 25000  # ~₦250/km (bolt/danfo blend), learned later
_DEFAULT_UNKNOWN_MINUTES = 30.0

_PAIR_RESOLUTION = 6  # coarse cells so pair-learning generalizes


class CostModel:
    """Learned travel-time/cost estimates.  Accuracy dominates solver choice.

    Learns from ``record_actual()``: per cell-pair EMAs of minutes and
    cost, plus global speed and cost-per-km.  Falls back to honest
    distance heuristics when nothing is known.
    """

    def __init__(self, db_path: str = "", *, profile: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = db_path or self._default_path(profile)
            if path != ":memory:":
                os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS route_actuals (
                       cell_a TEXT, cell_b TEXT, minutes REAL, cost_kobo REAL,
                       n INTEGER, recorded_at REAL DEFAULT 0,
                       PRIMARY KEY (cell_a, cell_b))""")
            # Migration for DBs created before recorded_at existed.
            try:
                cols = [r[1] for r in self._db.execute(
                    "PRAGMA table_info(route_actuals)").fetchall()]
                if "recorded_at" not in cols:
                    self._db.execute(
                        "ALTER TABLE route_actuals ADD COLUMN recorded_at REAL DEFAULT 0")
            except Exception:  # noqa: BLE001
                pass
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS route_globals (
                       key TEXT PRIMARY KEY, value REAL)""")
            self._db.commit()
        except Exception:  # noqa: BLE001 - a bad path is an empty model
            _log.warning("route: cost model db unavailable, memory-only",
                         exc_info=True)
            self._db = None
        self._speed_kmh = self._get_global("speed_kmh", _DEFAULT_SPEED_KMH)
        self._cost_per_km = self._get_global("cost_per_km_kobo",
                                            _DEFAULT_COST_PER_KM_KOBO)

    @staticmethod
    def _default_path(profile: str) -> str:
        termux = (profile == "termux") or (
            not profile and ("termux" in os.environ.get("PREFIX", "")
                             or "com.termux" in os.environ.get("HOME", "")))
        if termux:
            return ":memory:"
        return os.path.expanduser("~/.nomorals/planning/cost_model.db")

    # — learning —

    def record_actual(self, a: Stop, b: Stop, minutes: float,
                      cost_kobo: int = 0) -> bool:
        """Record a real trip.  Returns False on bad input; never raises."""
        try:
            minutes = float(minutes)
            if minutes <= 0 or not a or not b:
                return False
            ca = cell_for(a.lat, a.lng, _PAIR_RESOLUTION) if a.has_coords() else f"id:{a.stop_id}"
            cb = cell_for(b.lat, b.lng, _PAIR_RESOLUTION) if b.has_coords() else f"id:{b.stop_id}"
            if not ca or not cb:
                return False
            alpha = 0.35  # EMA weight for the new observation
            now = time.time()
            if self._db is not None:
                row = self._db.execute(
                    "SELECT minutes, cost_kobo, n, recorded_at FROM route_actuals "
                    "WHERE cell_a = ? AND cell_b = ?", (ca, cb)).fetchone()
                if row:
                    # Freshness decay: a pair untouched for 30+ days is mostly
                    # forgotten — the new observation dominates. (Mined: every
                    # serious cost model timestamps observations.)
                    try:
                        age_days = max(0.0, (now - float(row["recorded_at"] or now))
                                       / 86400.0)
                    except (TypeError, ValueError):
                        age_days = 0.0
                    w = min(0.85, alpha + age_days / 30.0 * 0.5)
                    new_min = row["minutes"] * (1 - w) + minutes * w
                    new_cost = row["cost_kobo"] * (1 - w) + float(cost_kobo) * w
                    self._db.execute(
                        "UPDATE route_actuals SET minutes = ?, cost_kobo = ?, "
                        "n = n + 1, recorded_at = ? "
                        "WHERE cell_a = ? AND cell_b = ?",
                        (new_min, new_cost, now, ca, cb))
                else:
                    self._db.execute(
                        "INSERT INTO route_actuals VALUES (?, ?, ?, ?, 1, ?)",
                        (ca, cb, minutes, float(cost_kobo), now))
                self._db.commit()
            # Learn global speed/cost-per-km when we know the distance.
            if a.has_coords() and b.has_coords():
                km = haversine_km(a.lat, a.lng, b.lat, b.lng)
                if km > 0.5 and minutes < 600:
                    self._speed_kmh = self._speed_kmh * (1 - alpha) + (km / (minutes / 60)) * alpha
                    self._set_global("speed_kmh", self._speed_kmh)
                    if cost_kobo > 0:
                        self._cost_per_km = self._cost_per_km * (1 - alpha) + (cost_kobo / km) * alpha
                        self._set_global("cost_per_km_kobo", self._cost_per_km)
            return True
        except Exception:  # noqa: BLE001 - never raises
            _log.debug("route: record_actual failed", exc_info=True)
            return False

    # — prediction —

    def estimate(self, a: Stop, b: Stop) -> tuple[float, int]:
        """``(minutes, cost_kobo)`` for a → b.  Never raises."""
        try:
            minutes, cost, _ = self.estimate_detail(a, b)
            return minutes, cost
        except Exception:  # noqa: BLE001
            return _DEFAULT_UNKNOWN_MINUTES, 0

    def estimate_detail(self, a: Stop, b: Stop) -> tuple[float, int, str]:
        """``(minutes, cost_kobo, source)`` — source is learned|heuristic."""
        try:
            learned = self._learned(a, b)
            if learned is not None:
                return learned[0], int(round(learned[1])), "learned"
            minutes, cost, km = self._heuristic(a, b)
            return minutes, cost, "heuristic"
        except Exception:  # noqa: BLE001 - never raises
            return _DEFAULT_UNKNOWN_MINUTES, 0, "heuristic"

    def _pair_key(self, a: Stop, b: Stop) -> tuple[str, str] | None:
        ca = cell_for(a.lat, a.lng, _PAIR_RESOLUTION) if a.has_coords() else f"id:{a.stop_id}"
        cb = cell_for(b.lat, b.lng, _PAIR_RESOLUTION) if b.has_coords() else f"id:{b.stop_id}"
        return (ca, cb) if ca and cb else None

    def _learned(self, a: Stop, b: Stop) -> tuple[float, float] | None:
        if self._db is None:
            return None
        key = self._pair_key(a, b)
        if key is None:
            return None
        row = self._db.execute(
            "SELECT minutes, cost_kobo FROM route_actuals "
            "WHERE cell_a = ? AND cell_b = ?", key).fetchone()
        if row:
            return float(row["minutes"]), float(row["cost_kobo"])
        return None

    def _heuristic(self, a: Stop, b: Stop) -> tuple[float, int, float]:
        km = 0.0
        if a.has_coords() and b.has_coords():
            km = haversine_km(a.lat, a.lng, b.lat, b.lng)
        if km > 0:
            minutes = km / max(5.0, self._speed_kmh) * 60.0 + 5.0  # +5 pickup overhead
            cost = int(round(km * self._cost_per_km))
        else:
            minutes, cost = _DEFAULT_UNKNOWN_MINUTES, 0
        return minutes, cost, km

    def _get_global(self, key: str, default: float) -> float:
        try:
            if self._db is None:
                return default
            row = self._db.execute(
                "SELECT value FROM route_globals WHERE key = ?", (key,)).fetchone()
            return float(row["value"]) if row else default
        except Exception:  # noqa: BLE001
            return default

    def _set_global(self, key: str, value: float) -> None:
        try:
            if self._db is None:
                return
            self._db.execute(
                "INSERT OR REPLACE INTO route_globals VALUES (?, ?)",
                (key, float(value)))
            self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> dict[str, Any]:
        """Learning summary: how much the model actually knows."""
        try:
            n = recent = 0
            if self._db is not None:
                row = self._db.execute(
                    "SELECT COUNT(*) AS c, SUM(n) AS s FROM route_actuals").fetchone()
                n = int(row["s"] or 0) if row else 0
                try:
                    cutoff = time.time() - 30 * 86400
                    row2 = self._db.execute(
                        "SELECT SUM(n) AS s FROM route_actuals "
                        "WHERE recorded_at >= ?", (cutoff,)).fetchone()
                    recent = int(row2["s"] or 0) if row2 else 0
                except Exception:  # noqa: BLE001
                    recent = n
            return {
                "recorded_trips": n,
                "recent_trips_30d": recent,
                "speed_kmh": round(self._speed_kmh, 1),
                "cost_per_km_kobo": int(round(self._cost_per_km)),
                "source": "learned" if n else "heuristic-defaults",
            }
        except Exception:  # noqa: BLE001
            return {"recorded_trips": 0, "source": "heuristic-defaults"}


# ── stage 3: solve — pluggable, profile-gated backends ───────────────────

def _ortools_available() -> bool:
    try:
        import ortools  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class RouteSolver:
    """TSP solver with pluggable backends.

    ``backend``: "auto" | "greedy" | "ortools" | "gpu".
    "auto" picks by profile: termux → greedy; otherwise ortools if
    installed, else greedy.  "gpu" selects the best available backend —
    there is no GPU-native TSP solver and this module is honest about it.
    The first stop in ``stops`` is the origin (where you start); the rest
    are sequenced.
    """

    def __init__(self, backend: str = "auto", *, profile: str = "") -> None:
        self.requested = (backend or "auto").lower()
        self.backend = self._pick_backend(self.requested, profile)

    @staticmethod
    def _pick_backend(requested: str, profile: str) -> str:
        if requested == "greedy":
            return "greedy"
        if requested in ("ortools", "gpu"):
            return "ortools" if _ortools_available() else "greedy"
        # auto
        termux = (profile == "termux") or (
            not profile and ("termux" in os.environ.get("PREFIX", "")
                             or "com.termux" in os.environ.get("HOME", "")))
        if termux:
            return "greedy"
        return "ortools" if _ortools_available() else "greedy"

    def solve(self, stops: list[Stop], cost: CostModel,
              *, start_time: float | None = None,
              return_to_origin: bool = False) -> RouteResult:
        """Sequence ``stops`` (first = origin).  Never raises."""
        try:
            stops = [s for s in (stops or []) if s is not None]
            if not stops:
                return RouteResult([], [], 0.0, 0, self.backend)
            if len(stops) == 1:
                return RouteResult(stops, [], 0.0, 0, self.backend,
                                   arrivals=[start_time or time.time()])
            t0 = start_time if start_time is not None else time.time()
            if self.backend == "ortools":
                order = self._solve_ortools(stops, cost, t0)
            else:
                order = self._solve_greedy(stops, cost, t0)
                order = self._two_opt(order, cost, t0=t0)
            return self._build_result(order, cost, start_time,
                                      return_to_origin=return_to_origin)
        except Exception:  # noqa: BLE001 - never raises
            _log.debug("route: solve failed", exc_info=True)
            return RouteResult(stops or [], [], 0.0, 0, self.backend)

    # — greedy: nearest-neighbor on learned minutes, window-aware —

    def _solve_greedy(self, stops: list[Stop], cost: CostModel,
                      t0: float) -> list[Stop]:
        origin, rest = stops[0], list(stops[1:])
        order = [origin]
        current = origin
        clock = t0
        while rest:
            # Window-aware choice: travel minutes + a heavy penalty for the
            # minutes we'd arrive after a stop's window closes. Arriving
            # early is free (we wait) — arriving late is what breaks plans.
            def _score(s: Stop) -> float:
                minutes = cost.estimate(current, s)[0]
                arr = clock + minutes * 60.0
                late = 0.0
                if s.window:
                    try:
                        end = float(s.window[1])
                        if end > 0 and arr > end:
                            late = (arr - end) / 60.0
                    except (TypeError, ValueError, IndexError):
                        pass
                return minutes + late * 1000.0

            nxt = min(rest, key=_score)
            order.append(nxt)
            rest.remove(nxt)
            travel = cost.estimate(current, nxt)[0]
            clock += travel * 60.0 + (nxt.dwell_minutes or 0) * 60.0
            current = nxt
        return order

    # — 2-opt local search (mined: NN ~25% over optimal, NN+2-opt ~5%) —

    def _two_opt(self, order: list[Stop], cost: CostModel,
                 max_passes: int = 25, t0: float | None = None) -> list[Stop]:
        """Uncross edge pairs until locally optimal. Pure Python, O(n²)
        per pass — sub-50ms for the errand-scale problems this module
        solves.

        Window-aware: when ``t0`` is given and any stop has a window, a
        swap is only accepted if it does not increase total lateness —
        the greedy pass's window respect is never undone. Never raises.
        """
        try:
            n = len(order)
            if n < 4 or n > 400:
                return order
            # Precompute the travel-minute matrix once.
            mat = [[cost.estimate(order[i], order[j])[0] for j in range(n)]
                   for i in range(n)]
            windows = [s.window for s in order]
            windowed = t0 is not None and any(windows)

            def tour_cost(tour: list[int]) -> float:
                total = 0.0
                for x, y in zip(tour, tour[1:]):
                    total += mat[x][y]
                if not windowed:
                    return total
                assert t0 is not None
                clock = t0
                late = 0.0
                for idx_pos, x in enumerate(tour):
                    if idx_pos:
                        clock += mat[tour[idx_pos - 1]][x] * 60.0
                    w = windows[x]
                    if w:
                        try:
                            end = float(w[1])
                            if end > 0 and clock > end:
                                late += (clock - end) / 60.0
                        except (TypeError, ValueError, IndexError):
                            pass
                    clock += (order[x].dwell_minutes or 0) * 60.0
                return total + late * 1000.0

            tour = list(range(n))
            best = tour_cost(tour)
            improved = True
            passes = 0
            while improved and passes < max_passes:
                improved = False
                passes += 1
                for i in range(1, n - 1):
                    for k in range(i + 1, n):
                        cand = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
                        c = tour_cost(cand)
                        if best - c > 1e-6:
                            tour, best = cand, c
                            improved = True
            return [order[i] for i in tour]
        except Exception:  # noqa: BLE001 - never raises
            _log.debug("route: 2-opt failed", exc_info=True)
            return order

    # — OR-Tools: exact-ish TSP on the learned cost matrix —

    def _solve_ortools(self, stops: list[Stop], cost: CostModel,
                       t0: float) -> list[Stop]:
        try:
            from ortools.constraint_solver import routing_enums_pb2, pywrapcp
            n = len(stops)
            matrix = [[int(round(cost.estimate(stops[i], stops[j])[0] * 60))
                       for j in range(n)] for i in range(n)]
            manager = pywrapcp.RoutingIndexManager(n, 1, 0)
            routing = pywrapcp.RoutingModel(manager)

            def _cb(i: int, j: int) -> int:
                return matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]

            transit = routing.RegisterTransitCallback(_cb)
            routing.SetArcCostEvaluatorOfAllVehicles(transit)

            # Time dimension (mined VRPTW pattern): when any stop has a
            # window, constrain arrivals — travel + dwell at the departing
            # node goes in the transit, waiting is the slack.
            has_windows = any(s.window for s in stops)
            if has_windows:
                try:
                    dwell_s = [int(round((s.dwell_minutes or 0) * 60))
                               for s in stops]

                    def _time_cb(i: int, j: int) -> int:
                        fi = manager.IndexToNode(i)
                        return matrix[fi][manager.IndexToNode(j)] + dwell_s[fi]

                    time_transit = routing.RegisterTransitCallback(_time_cb)
                    routing.AddDimension(
                        time_transit,
                        4 * 3600,   # slack: allow up to 4h waiting
                        24 * 3600,  # horizon: one day
                        False,      # don't force cumul start at zero
                        "Time")
                    time_dim = routing.GetDimensionOrDie("Time")
                    for loc, s in enumerate(stops):
                        if not s.window:
                            continue
                        try:
                            w0, w1 = float(s.window[0]), float(s.window[1])
                        except (TypeError, ValueError, IndexError):
                            continue
                        if w0 <= 0 or w1 <= 0 or w1 < w0:
                            continue
                        idx = manager.NodeToIndex(loc)
                        time_dim.CumulVar(idx).SetRange(
                            max(0, int(w0 - t0)), int(w1 - t0))
                    # Keep the depot/origin start honest: it leaves at t0.
                    time_dim.CumulVar(routing.Start(0)).SetRange(0, 0)
                except Exception:  # noqa: BLE001 - windows are best-effort
                    _log.debug("route: time dimension failed", exc_info=True)

            params = pywrapcp.DefaultRoutingSearchParameters()
            params.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            params.time_limit.seconds = 10
            solution = routing.SolveWithParameters(params)
            if solution is None:
                return self._two_opt(self._solve_greedy(stops, cost, t0), cost, t0=t0)
            order, idx = [], routing.Start(0)
            while not routing.IsEnd(idx):
                order.append(stops[manager.IndexToNode(idx)])
                idx = solution.Value(routing.NextVar(idx))
            return order or self._two_opt(self._solve_greedy(stops, cost, t0),
                                             cost, t0=t0)
        except Exception:  # noqa: BLE001 - fall back, never raise
            _log.debug("route: ortools failed, using greedy", exc_info=True)
            return self._two_opt(self._solve_greedy(stops, cost, t0), cost, t0=t0)

    # — result assembly with honest per-stop arrival times —

    def _build_result(self, order: list[Stop], cost: CostModel,
                      start_time: float | None,
                      return_to_origin: bool = False) -> RouteResult:
        legs: list[Leg] = []
        arrivals: list[float] = []
        t = start_time if start_time is not None else time.time()
        arrivals.append(t)
        total_min, total_cost, learned = 0.0, 0, 0
        seq = list(order)
        if return_to_origin and len(order) > 1:
            seq = list(order) + [order[0]]  # the closing leg home
        for a, b in zip(seq, seq[1:]):
            minutes, cost_kobo, source = cost.estimate_detail(a, b)
            km = haversine_km(a.lat, a.lng, b.lat, b.lng) \
                if a.has_coords() and b.has_coords() else 0.0
            legs.append(Leg(a.stop_id, b.stop_id, minutes, cost_kobo, km, source))
            if source == "learned":
                learned += 1
            t += minutes * 60.0
            arrivals.append(t)
            total_min += minutes
            total_cost += cost_kobo
            t += (b.dwell_minutes or 0) * 60.0  # time spent at the stop
            total_min += b.dwell_minutes or 0
        violations = self._window_violations(order, arrivals[:len(order)])
        return RouteResult(seq, legs, total_min, total_cost, self.backend,
                           arrivals, learned, violations)

    @staticmethod
    def _window_violations(order: list[Stop],
                           arrivals: list[float]) -> list[str]:
        """Which stops arrive after their window closes.

        Arriving *before* the window opens is fine (you wait); arriving
        after it closes is a miss.  ``Stop.window`` was previously ignored
        by the solver — this at least makes infeasibility visible instead
        of confidently promising the impossible.
        """
        out: list[str] = []
        try:
            for stop, arr in zip(order, arrivals):
                w = stop.window
                if not w:
                    continue
                try:
                    start, end = float(w[0]), float(w[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if end > 0 and arr > end:
                    late_min = (arr - end) / 60.0
                    out.append(
                        f"{stop.label}: arrives {_fmt_eta(arr)} — "
                        f"{late_min:.0f}m after window closes "
                        f"({_fmt_eta(end)})")
        except Exception:  # noqa: BLE001 - never raises
            pass
        return out


# ── RoutePlanner: the pipeline in one object ─────────────────────────────


class RoutePlanner:
    """Predict → Build → Solve in one place.  Owns stops + the cost model."""

    def __init__(self, db_path: str = "", *, profile: str = "",
                 backend: str = "auto") -> None:
        self.cost = CostModel(db_path=db_path, profile=profile)
        self.solver = RouteSolver(backend, profile=profile)
        self.geo = GeoIndex()
        self._stops: dict[str, Stop] = {}
        try:
            self._db_path = db_path or self.cost._default_path(profile)
            if self._db_path != ":memory:":
                db = sqlite3.connect(self._db_path)
                db.execute(
                    """CREATE TABLE IF NOT EXISTS route_stops (
                           stop_id TEXT PRIMARY KEY, label TEXT, lat REAL,
                           lng REAL, dwell_minutes REAL,
                           window_start REAL DEFAULT 0,
                           window_end REAL DEFAULT 0)""")
                try:
                    cols = [r[1] for r in db.execute(
                        "PRAGMA table_info(route_stops)").fetchall()]
                    for col in ("window_start", "window_end"):
                        if col not in cols:
                            db.execute(
                                f"ALTER TABLE route_stops ADD COLUMN {col} "
                                "REAL DEFAULT 0")
                    db.commit()
                except Exception:  # noqa: BLE001
                    pass
                for row in db.execute("SELECT * FROM route_stops"):
                    stop = Stop(row["stop_id"], row["label"], row["lat"],
                                row["lng"], row["dwell_minutes"] or 15.0)
                    try:
                        ws = float(row["window_start"] or 0)
                        we = float(row["window_end"] or 0)
                        if ws > 0 and we > ws:
                            stop.window = (ws, we)
                    except (TypeError, ValueError, IndexError, KeyError):
                        pass
                    self._stops[stop.stop_id] = stop
                    if stop.has_coords():
                        self.geo.add(stop.stop_id, stop.lat, stop.lng)
                db.close()
        except Exception:  # noqa: BLE001 - never raises
            _log.debug("route: stop registry unavailable", exc_info=True)

    # — stop registry —

    def add_stop(self, label: str, *, lat: float | None = None,
                 lng: float | None = None,
                 dwell_minutes: float = 15.0) -> Stop | None:
        """Save a named stop.  Never raises."""
        try:
            label = (label or "").strip()
            if not label:
                return None
            sid = "stop_" + uuid.uuid4().hex[:8]
            lat = float(lat) if lat is not None else None
            lng = float(lng) if lng is not None else None
            stop = Stop(sid, label, lat, lng, max(0.0, float(dwell_minutes or 0)))
            self._stops[sid] = stop
            if stop.has_coords():
                self.geo.add(sid, lat, lng)
            try:
                if self._db_path != ":memory:":
                    db = sqlite3.connect(self._db_path)
                    db.execute(
                        "INSERT OR REPLACE INTO route_stops VALUES (?, ?, ?, ?, ?)",
                        (sid, label, lat, lng, stop.dwell_minutes))
                    db.commit()
                    db.close()
            except Exception:  # noqa: BLE001
                pass
            return stop
        except Exception:  # noqa: BLE001 - never raises
            return None

    def set_window(self, label: str, start_hm: str, end_hm: str) -> bool:
        """Set a stop's time window ("16:00"-"17:30" today). Never raises."""
        try:
            stop = self.find_stop(label)
            if stop is None:
                return False
            import datetime as _dt
            today = _dt.date.today()
            w0 = _dt.datetime.combine(
                today, _dt.datetime.strptime(start_hm.strip(), "%H:%M").time())
            w1 = _dt.datetime.combine(
                today, _dt.datetime.strptime(end_hm.strip(), "%H:%M").time())
            if w1 <= w0:
                w1 += _dt.timedelta(days=1)
            stop.window = (w0.timestamp(), w1.timestamp())
            if self._db_path != ":memory:":
                db = sqlite3.connect(self._db_path)
                db.execute(
                    "UPDATE route_stops SET window_start = ?, window_end = ? "
                    "WHERE stop_id = ?",
                    (stop.window[0], stop.window[1], stop.stop_id))
                db.commit()
                db.close()
            return True
        except Exception:  # noqa: BLE001 - never raises
            return False

    def find_stop(self, text: str) -> Stop | None:
        """Fuzzy name lookup.  Never raises."""
        try:
            text = (text or "").strip().lower()
            if not text:
                return None
            for s in self._stops.values():
                if s.label.lower() == text:
                    return s
            for s in self._stops.values():
                if text in s.label.lower():
                    return s
            return None
        except Exception:  # noqa: BLE001
            return None

    def list_stops(self) -> list[Stop]:
        return list(self._stops.values())

    # — the pipeline —

    def plan(self, stops: list[Stop], *,
             start_time: float | None = None,
             return_to_origin: bool = False) -> RouteResult:
        """Predict → Build → Solve.  Never raises."""
        try:
            return self.solver.solve(stops, self.cost, start_time=start_time,
                                     return_to_origin=return_to_origin)
        except Exception:  # noqa: BLE001
            return RouteResult(stops or [], [], 0.0, 0, self.solver.backend)

    def record_trip(self, a: Stop, b: Stop, minutes: float,
                    cost_kobo: int = 0) -> bool:
        """Feed an actual trip back into the cost model."""
        return self.cost.record_actual(a, b, minutes, cost_kobo)


# ── chat: /route (owner-only at dispatch) ─────────────────────────────────


def _fmt_naira(kobo: int) -> str:
    try:
        return f"₦{int(kobo) // 100:,}"
    except (TypeError, ValueError):
        return "₦0"


def _fmt_eta(epoch: float) -> str:
    try:
        return time.strftime("%H:%M", time.localtime(float(epoch)))
    except (TypeError, ValueError):
        return "--:--"


def format_route(result: RouteResult) -> str:
    """Human-readable route with honest per-stop arrival times."""
    if not result.order:
        return "no stops to plan."
    closed = len(result.order) > 1 and result.order[-1] is result.order[0]
    solver_note = ("2-opt refined" if result.backend == "greedy"
                   else "OR-Tools optimized")
    lines = [f"🗺️ route ({result.backend} solver, {solver_note}, "
             f"{result.learned_legs}/{len(result.legs)} legs learned):"]
    for i, stop in enumerate(result.order):
        eta = _fmt_eta(result.arrivals[i]) if i < len(result.arrivals) else "--:--"
        if closed and i == len(result.order) - 1:
            tag = f"🏠 back home {_fmt_eta(result.arrivals[i])}"
        else:
            tag = "📍 start" if i == 0 else f"→ {eta}"
        win = ""
        if stop.window and not (closed and i == len(result.order) - 1):
            try:
                win = (f" 🕐 window {_fmt_eta(float(stop.window[0]))}–"
                       f"{_fmt_eta(float(stop.window[1]))}")
            except (TypeError, ValueError, IndexError):
                pass
        dwell = f" (~{stop.dwell_minutes:g}m there)" if i and stop.dwell_minutes else ""
        lines.append(f"  {i + 1}. {stop.label} {tag}{win}{dwell}")
    total_h = result.total_minutes / 60.0
    lines.append(f"⏱️ {total_h:.1f}h total (incl. dwell) · "
                 f"💰 {_fmt_naira(result.total_cost_kobo)} est. travel")
    if result.window_violations:
        lines.append("⚠️ window misses:")
        lines += [f"   • {v}" for v in result.window_violations]
    lines.append(ROUTE_DISCLAIMER)
    return "\n".join(lines)


def _usage() -> str:
    return (
        "usage:\n"
        "  /route plan <stop1>; <stop2>; ... [roundtrip] — optimal order + honest times\n"
        "  /route add <label> [lat,lng] — save a stop\n"
        "  /route window <label> <HH:MM>-<HH:MM> — set a stop's time window\n"
        "  /route stops — list saved stops\n"
        "  /route record <from> > <to> <minutes> [cost_kobo] — teach the model\n"
        "  /route stats — what the cost model has learned"
    )


def control_route(tail: str, context: Any = None, chat: Any = None,
                  planner: RoutePlanner | None = None) -> str:
    """/route chat control (owner-only at dispatch).  Never raises."""
    try:
        p = planner or RoutePlanner()
        parts = (tail or "").strip().split(None, 1)
        if not parts:
            return _usage()

        cmd, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        if cmd == "add":
            bits = rest.rsplit(None, 1)
            lat = lng = None
            label = rest
            if len(bits) == 2 and "," in bits[1]:
                try:
                    la, lo = bits[1].split(",", 1)
                    lat, lng = float(la), float(lo)
                    label = bits[0]
                except ValueError:
                    pass
            stop = p.add_stop(label, lat=lat, lng=lng)
            if stop is None:
                return "couldn't save that stop."
            coords = f" @ {lat},{lng}" if lat is not None else ""
            return f"✅ saved stop: {stop.label}{coords}"

        if cmd == "stops":
            stops = p.list_stops()
            if not stops:
                return "no saved stops yet. `/route add <label> [lat,lng]`"
            return "📍 saved stops:\n" + "\n".join(
                f"  • {s.label}" + (f" @ {s.lat},{s.lng}" if s.has_coords() else "")
                for s in stops)

        if cmd == "record":
            bits = rest.split()
            if len(bits) < 4 or ">" not in rest:
                return ("usage: /route record <from> > <to> <minutes> "
                        "[cost_kobo]")
            frm_txt, _, rest2 = rest.partition(">")
            nums = rest2.strip().split()
            a, b = p.find_stop(frm_txt.strip()), None
            minutes, cost_kobo = 0.0, 0
            if nums:
                try:
                    minutes = float(nums[-2]) if len(nums) >= 2 else float(nums[-1])
                except ValueError:
                    return "minutes must be a number."
                try:
                    cost_kobo = int(nums[-1]) if len(nums) >= 2 else 0
                except ValueError:
                    cost_kobo = 0
                b = p.find_stop(" ".join(nums[:-2] if len(nums) >= 2 else []))
            if a is None or b is None or minutes <= 0:
                return "couldn't match those stops — `/route stops` to see names."
            ok = p.record_trip(a, b, minutes, cost_kobo)
            return (f"✅ recorded: {a.label} → {b.label}, {minutes:g}m. "
                    f"the model learns." if ok else "couldn't record that.")

        if cmd == "stats":
            s = p.cost.stats()
            recent = (f" ({s.get('recent_trips_30d', 0)} in the last 30d)"
                      if s.get("recorded_trips") else "")
            return (f"📊 cost model: {s['recorded_trips']} recorded trips{recent} · "
                    f"avg speed {s['speed_kmh']} km/h · "
                    f"₦{s['cost_per_km_kobo'] // 100:,}/km "
                    f"({s['source']}, backend {p.solver.backend})")

        if cmd == "window":
            bits = rest.split()
            if len(bits) < 2 or "-" not in bits[-1]:
                return "usage: /route window <label> <HH:MM>-<HH:MM>"
            span = bits[-1]
            label = " ".join(bits[:-1])
            try:
                start_hm, end_hm = span.split("-", 1)
            except ValueError:
                return "usage: /route window <label> <HH:MM>-<HH:MM>"
            ok = p.set_window(label, start_hm, end_hm)
            return (f"🕐 window set: {label} {start_hm}–{end_hm}."
                    if ok else
                    f"couldn't set that window — `/route stops` to see names.")

        if cmd == "plan":
            roundtrip = rest.strip().lower().endswith("roundtrip")
            body = rest[: -len("roundtrip")].rstrip() if roundtrip else rest
            labels = [x.strip() for x in body.split(";") if x.strip()]
            if len(labels) < 2:
                return "give me at least 2 stops: `/route plan home; market; bank`"
            stops, missing = [], []
            for label in labels:
                s = p.find_stop(label)
                if s is None:
                    missing.append(label)
                else:
                    stops.append(s)
            if missing:
                return ("I don't know where these are: "
                        + ", ".join(missing)
                        + ". save them first: `/route add <label> [lat,lng]`")
            return format_route(p.plan(stops, return_to_origin=roundtrip))

        return _usage()
    except Exception:  # noqa: BLE001 - never raises
        _log.debug("route: control failed", exc_info=True)
        return "route planner hiccup — try again."
