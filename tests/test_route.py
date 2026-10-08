"""Tests for build-map #90: Predict → Build → Solve routing pipeline.

All offline.  The key invariant from the mine: cost-function accuracy
dominates solver choice — so the cost model learns from actuals, and the
solver backends stay honest (greedy always, ortools if installed, gpu
picks the best available and says so).
"""

import pytest

from nomorals.planning.route import (
    Stop,
    CostModel,
    GeoIndex,
    RouteSolver,
    RoutePlanner,
    control_route,
    format_route,
    cell_for,
    h3_available,
    haversine_km,
)


def _stop(label, lat=None, lng=None, dwell=15.0):
    return Stop(stop_id=f"s_{label}", label=label, lat=lat, lng=lng,
                dwell_minutes=dwell)


# ── H3 cells / geo index ─────────────────────────────────────────────────


def test_cell_for_grid_fallback():
    c1 = cell_for(6.5244, 3.3792)
    c2 = cell_for(6.5244, 3.3792)
    assert c1 and c1 == c2
    assert cell_for(6.5244, 3.3792) != cell_for(40.7, -74.0)
    assert cell_for("bad", None) == ""


def test_haversine_lagos_ikeja():
    # Lagos → Ikeja is ~15 km
    km = haversine_km(6.5244, 3.3792, 6.6018, 3.3515)
    assert 8 < km < 25


def test_geo_index_nearby_scans_cells():
    g = GeoIndex()
    g.add("home", 6.5244, 3.3792)
    g.add("far", 40.7, -74.0)
    near = g.nearby(6.53, 3.38, radius_km=5.0)
    assert "home" in near
    assert "far" not in near


# ── cost model: learning ─────────────────────────────────────────────────


def test_cost_model_heuristic_before_learning():
    m = CostModel(db_path=":memory:")
    a = _stop("A", 6.5244, 3.3792)
    b = _stop("B", 6.6018, 3.3515)
    minutes, cost = m.estimate(a, b)
    assert minutes > 0
    assert cost >= 0


def test_cost_model_learns_from_actuals():
    m = CostModel(db_path=":memory:")
    a = _stop("A", 6.5244, 3.3792)
    b = _stop("B", 6.6018, 3.3515)
    assert m.record_actual(a, b, 25.0, 500000)
    minutes, cost, source = m.estimate_detail(a, b)
    assert source == "learned"
    # EMA pulls toward the recorded actual
    assert 20 < minutes < 40
    assert cost > 0


def test_cost_model_learns_speed_and_cost_per_km():
    m = CostModel(db_path=":memory:")
    a = _stop("A", 6.5244, 3.3792)
    b = _stop("B", 6.6244, 3.3792)  # ~11 km north
    m.record_actual(a, b, 20.0, 300000)  # 33 km/h, ₦270/km
    stats = m.stats()
    assert stats["recorded_trips"] == 1
    assert stats["speed_kmh"] != 32.0  # moved from default


def test_cost_model_rejects_bad_input():
    m = CostModel(db_path=":memory:")
    a = _stop("A")
    assert m.record_actual(a, _stop("B"), -5) is False
    assert m.record_actual(None, _stop("B"), 10) is False


def test_cost_model_name_only_stops():
    m = CostModel(db_path=":memory:")
    a, b = _stop("bank"), _stop("market")
    minutes, cost = m.estimate(a, b)
    assert minutes == 30.0  # honest unknown default


# ── solver ───────────────────────────────────────────────────────────────


def _triangle():
    # home at origin; market 1km east; bank 10km north of home
    return [
        _stop("home", 6.5244, 3.3792),
        _stop("market", 6.5244, 3.3892),
        _stop("bank", 6.6144, 3.3792),
    ]


def test_greedy_solves_nearest_first():
    s = RouteSolver(backend="greedy")
    assert s.backend == "greedy"
    result = s.solve(_triangle(), CostModel(db_path=":memory:"))
    assert [x.label for x in result.order] == ["home", "market", "bank"]
    assert result.total_minutes > 0
    assert len(result.legs) == 2
    assert len(result.arrivals) == 3


def test_solver_keeps_first_stop_as_origin():
    s = RouteSolver(backend="greedy")
    stops = _triangle()
    result = s.solve(stops, CostModel(db_path=":memory:"))
    assert result.order[0].label == "home"


def test_solver_honest_arrival_times():
    s = RouteSolver(backend="greedy")
    result = s.solve(_triangle(), CostModel(db_path=":memory:"),
                     start_time=1_000_000.0)
    assert result.arrivals[0] == 1_000_000.0
    # arrivals increase (travel + dwell)
    assert result.arrivals[2] > result.arrivals[1] > result.arrivals[0]


def test_backend_selection_honest():
    # gpu has no native TSP backend — picks the best available, honestly
    s = RouteSolver(backend="gpu")
    assert s.backend in ("greedy", "ortools")
    s2 = RouteSolver(backend="auto", profile="termux")
    assert s2.backend == "greedy"


def test_solver_single_and_empty():
    s = RouteSolver(backend="greedy")
    m = CostModel(db_path=":memory:")
    assert s.solve([], m).order == []
    one = s.solve([_stop("home")], m)
    assert len(one.order) == 1


def test_learned_legs_counted():
    s = RouteSolver(backend="greedy")
    m = CostModel(db_path=":memory:")
    stops = _triangle()
    m.record_actual(stops[0], stops[1], 5.0, 100000)
    m.record_actual(stops[1], stops[2], 40.0, 800000)
    result = s.solve(stops, m)
    assert result.learned_legs >= 1


# ── planner ──────────────────────────────────────────────────────────────


def test_planner_stop_registry():
    p = RoutePlanner(db_path=":memory:")
    s = p.add_stop("Yaba Market", lat=6.51, lng=3.38)
    assert s is not None
    assert p.find_stop("yaba market") is s
    assert p.find_stop("yaba") is s  # fuzzy
    assert p.find_stop("nowhere xyz") is None
    assert p.add_stop("") is None


def test_planner_record_trip_teaches_model():
    p = RoutePlanner(db_path=":memory:")
    a = p.add_stop("home", lat=6.5244, lng=3.3792)
    b = p.add_stop("bank", lat=6.6144, lng=3.3792)
    assert p.record_trip(a, b, 35.0, 600000)
    _, _, source = p.cost.estimate_detail(a, b)
    assert source == "learned"


# ── chat ─────────────────────────────────────────────────────────────────


def test_control_route_usage():
    assert "plan" in control_route("")


def test_control_route_plan_flow():
    p = RoutePlanner(db_path=":memory:")
    p.add_stop("home", lat=6.5244, lng=3.3792)
    p.add_stop("market", lat=6.5244, lng=3.3892)
    p.add_stop("bank", lat=6.6144, lng=3.3792)
    out = control_route("plan home; market; bank", planner=p)
    assert "🗺️" in out
    assert "home" in out and "market" in out
    assert "learned" in out or "heuristic" in out


def test_control_route_unknown_stop_honest():
    p = RoutePlanner(db_path=":memory:")
    out = control_route("plan home; nowhere xyz", planner=p)
    assert "don't know" in out
    assert "nowhere xyz" in out


def test_control_route_stops_and_stats():
    p = RoutePlanner(db_path=":memory:")
    assert "no saved stops" in control_route("stops", planner=p)
    p.add_stop("home", lat=6.5244, lng=3.3792)
    assert "home" in control_route("stops", planner=p)
    stats = control_route("stats", planner=p)
    assert "cost model" in stats


def test_control_route_record():
    p = RoutePlanner(db_path=":memory:")
    p.add_stop("home", lat=6.5244, lng=3.3792)
    p.add_stop("bank", lat=6.6144, lng=3.3792)
    out = control_route("record home > bank 35 600000", planner=p)
    assert "recorded" in out
    _, _, source = p.cost.estimate_detail(
        p.find_stop("home"), p.find_stop("bank"))
    assert source == "learned"


def test_control_route_needs_two_stops():
    p = RoutePlanner(db_path=":memory:")
    assert "at least 2" in control_route("plan home", planner=p)


def test_control_route_never_raises():
    p = RoutePlanner(db_path=":memory:")
    for tail in ["", "plan", "add", "record", "stops extra junk", None]:
        out = control_route(tail, planner=p)
        assert isinstance(out, str) and out


def test_format_route_empty():
    from nomorals.planning.route import RouteResult
    assert "no stops" in format_route(RouteResult([], [], 0, 0, "greedy"))


def test_format_route_shows_times_and_cost():
    s = RouteSolver(backend="greedy")
    result = s.solve(_triangle(), CostModel(db_path=":memory:"),
                     start_time=1_700_000_000.0)
    out = format_route(result)
    assert "⏱️" in out and "💰" in out
    assert "→" in out  # per-stop arrival times
