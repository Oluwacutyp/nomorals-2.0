"""Doc 19/21 M13 — the corridor widening ladder (corridor-pad pass).

The census (``~/workspace/corridorpad-census``) proved on the
real pack that the shipped corridor — a pure function of the
endpoints — can starve a route whose roads bulge past its edge:
a Lagos hop served 11,702 m at ratio 4.00 while 9,370 m ran
just outside the corridor, and a Port Harcourt hop's 3.09 was
proven true geography (the pad ladder dead flat).  These pins
hold the ladder's contract:

* the first attempt is the shipped corridor, byte-identical —
  a healthy route never re-materializes (the materializer is
  called exactly once);
* a suspect first answer (ratio > ROUTE_RATIO_SUSPECT, or a
  no-route of a retryable class) climbs geometrically growing
  pads to the 0.08° ceiling, checking the tile cap BEFORE each
  step, stopping at the first non-suspect attempt;
* the final answer is the best measured across attempts — a
  worse widened result never replaces a better first;
* the ticket records the ladder (pad used, steps tried,
  per-step ratios) and a detour_class computed from the
  ladder's own measurements: corridor_bound when widening
  materially improved the answer, network_detour when the
  ladder exhausted and the ratio stands.

The graph materializer (``Router._graph``) is the seam under
test substitution — fixture graphs with scripted arc lengths
stand in for tile materialization; the ladder, the snap, the
Dijkstra and the ticket all run for real.  Real-pack numbers
live in the acceptance battery, not here.
"""

from __future__ import annotations

import pytest

from nomorals.location.primitives import LatLon, haversine_m
from nomorals.maps import routing
from nomorals.maps.coords import BBox
from nomorals.maps.manifest import FORMAT_VERSION, PackManifest
from nomorals.maps.routing import (
    CORRIDOR_PAD_MIN_DEG,
    DETOUR_CORRIDOR_BOUND,
    DETOUR_NETWORK_DETOUR,
    DETOUR_UNREMARKABLE,
    Router,
    _corridor,
    _Graph,
)

# The census legs, verbatim: the Lagos offset leg whose shipped
# route was corridor-bound, and the Port Harcourt offset leg
# whose detour is true geography.
LAGOS = (LatLon(6.4463, 3.3838), LatLon(6.4623, 3.4048))
PH = (LatLon(4.768, 7.0175), LatLon(4.784, 7.0385))
# A healthy leg (the battery's Ilawe-Ekiti → Ado Ekiti, 1.23).
HEALTHY = (LatLon(7.5989, 5.1057), LatLon(7.6239, 5.2226))


def _manifest() -> PackManifest:
    return PackManifest(
        format=FORMAT_VERSION, pack_id="ladder-fixture",
        name="Ladder Fixture",
        bbox=BBox(west=2.0, south=4.0, east=9.0, north=8.0),
        zoom_min=14, zoom_max=14, built_at="2026-11-10T00:00:00Z",
        attribution="fixture", tile_format="mvt",
        digest="ladder-fixture-digest")


def _graph_for(start: LatLon, end: LatLon, spec: tuple) -> _Graph:
    """One scripted corridor graph.  Specs: ("route", metres)
    joins the endpoints by one road of that length;
    ("disconnected",) leaves the two endpoint nodes unjoined;
    ("start-missing",) holds only the end node, so the start
    cannot snap."""
    nodes: dict[str, tuple[float, float]] = {"b": (end.lat, end.lon)}
    adj: dict[str, list] = {"b": []}
    if spec[0] != "start-missing":
        nodes["a"] = (start.lat, start.lon)
        adj["a"] = []
    if spec[0] == "route":
        arc = ("primary", "Fixture Road", False)
        adj["a"] = [("b", spec[1], *arc)]
        adj["b"] = [("a", spec[1], *arc)]
    return _Graph(nodes=nodes, adj=adj, tiles_present=9)


class _ScriptedGraphs:
    """Router._graph under script: one spec per attempt, the
    last repeating; every call (and corridor) is recorded."""

    def __init__(self, start: LatLon, end: LatLon, specs: list):
        self._start, self._end = start, end
        self._specs = list(specs)
        self.calls = 0
        self.corridors: list[BBox] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(_router, _pack_dir, _manifest, corridor, _zoom):
            spec = self._specs[min(self.calls, len(self._specs) - 1)]
            self.calls += 1
            self.corridors.append(corridor)
            return _graph_for(self._start, self._end, spec)

        monkeypatch.setattr(Router, "_graph", fake)


def _route(tmp_path, monkeypatch, pair, specs):
    scripted = _ScriptedGraphs(pair[0], pair[1], specs)
    scripted.install(monkeypatch)
    router = Router(tmp_path)
    result = router.route(tmp_path, _manifest(), pair[0], pair[1],
                          from_label="start", to_label="end")
    return result, scripted


class TestCorridorShape:
    """The shipped corridor itself does not move."""

    def test_default_corridor_is_the_shipped_formula(self):
        for pair in (LAGOS, PH, HEALTHY):
            assert _corridor(pair[0], pair[1]) == _corridor(
                pair[0], pair[1], pad_floor=CORRIDOR_PAD_MIN_DEG)

    def test_ladder_floors_are_the_census_ladder(self):
        assert Router._ladder_floors() == (0.01, 0.02, 0.04, 0.08)

    def test_wider_floor_grows_the_corridor(self):
        base = _corridor(LAGOS[0], LAGOS[1])
        wide = _corridor(LAGOS[0], LAGOS[1], pad_floor=0.04)
        assert wide.west < base.west and wide.east > base.east
        assert wide.south < base.south and wide.north > base.north


class TestFirstAttemptParity:
    """A healthy route is untouched by construction."""

    def test_good_route_never_rematerializes(self, tmp_path,
                                            monkeypatch):
        straight = haversine_m(*HEALTHY)
        distance = 1.23 * straight
        result, scripted = _route(
            tmp_path, monkeypatch, HEALTHY, [("route", distance)])
        assert scripted.calls == 1
        assert result.ok and result.distance_m == distance
        assert result.widened is False
        assert result.corridor_pad_deg == CORRIDOR_PAD_MIN_DEG
        assert result.ladder_stop == ""
        assert result.detour_class == DETOUR_UNREMARKABLE
        assert len(result.ladder_steps) == 1
        step = result.ladder_steps[0]
        assert step.pad_deg == CORRIDOR_PAD_MIN_DEG
        assert step.outcome == "route"
        assert step.distance_m == distance
        assert step.ratio == pytest.approx(1.23)
        assert step.tiles == len(
            _corridor(*HEALTHY).tiles(14))
        assert "corridor widening" not in result.detail


class TestWideningRecovery:
    """The Lagos shape: starvation at the shipped pad, recovered
    by widening, recorded on the ticket."""

    def test_starvation_recovers_at_the_third_pad(self, tmp_path,
                                                  monkeypatch):
        straight = haversine_m(*LAGOS)
        specs = [("route", 11702.0), ("route", 11702.0),
                 ("route", 9370.0), ("route", 9370.0)]
        result, scripted = _route(tmp_path, monkeypatch, LAGOS, specs)
        assert scripted.calls == 4
        assert result.ok and result.distance_m == 9370.0
        assert result.widened is True
        assert result.corridor_pad_deg == 0.04
        # 9,370 m still ratios 3.20 — suspect — so the ladder
        # honestly climbs to its ceiling and reports that.
        assert result.ladder_stop == "ceiling"
        assert result.detour_class == DETOUR_CORRIDOR_BOUND
        steps = result.ladder_steps
        assert [s.pad_deg for s in steps] == [0.01, 0.02, 0.04, 0.08]
        assert [s.distance_m for s in steps] == [
            11702.0, 11702.0, 9370.0, 9370.0]
        assert [s.ratio for s in steps] == pytest.approx(
            [11702.0 / straight, 11702.0 / straight,
             9370.0 / straight, 9370.0 / straight])
        assert all(s.outcome == "route" for s in steps)
        assert all(later.tiles >= earlier.tiles for earlier, later
                   in zip(steps, steps[1:], strict=False))
        assert "corridor widening" in result.detail
        assert "corridor_bound" in result.detail

    def test_disconnected_then_connected_by_widening(
            self, tmp_path, monkeypatch):
        straight = haversine_m(*LAGOS)
        specs = [("disconnected",), ("route", 1.3 * straight)]
        result, scripted = _route(tmp_path, monkeypatch, LAGOS, specs)
        assert scripted.calls == 2
        assert result.ok and result.distance_m == 1.3 * straight
        assert result.widened is True
        assert result.corridor_pad_deg == 0.02
        assert result.ladder_stop == "not-suspect"
        assert result.detour_class == DETOUR_CORRIDOR_BOUND
        assert result.ladder_steps[0].outcome == "disconnected"
        assert result.ladder_steps[0].distance_m is None
        assert result.ladder_steps[0].ratio is None

    def test_snap_failure_then_route_by_widening(self, tmp_path,
                                                 monkeypatch):
        straight = haversine_m(*LAGOS)
        specs = [("start-missing",), ("route", 1.4 * straight)]
        result, scripted = _route(tmp_path, monkeypatch, LAGOS, specs)
        assert scripted.calls == 2
        assert result.ok
        assert result.ladder_steps[0].outcome == "snap-from"
        assert result.detour_class == DETOUR_CORRIDOR_BOUND


class TestNetworkDetour:
    """The PH shape: the ladder climbs, nothing moves, and the
    ticket says the detour is the network's own."""

    def test_flat_suspect_route_is_unchanged_network_detour(
            self, tmp_path, monkeypatch):
        straight = haversine_m(*PH)
        specs = [("route", 9054.0)] * 4
        result, scripted = _route(tmp_path, monkeypatch, PH, specs)
        assert scripted.calls == 4
        assert result.ok and result.distance_m == 9054.0
        # The best route is the FIRST attempt's — ties never
        # move the winner to a dearer materialization.
        assert result.widened is False
        assert result.corridor_pad_deg == 0.01
        assert result.ladder_stop == "ceiling"
        assert result.detour_class == DETOUR_NETWORK_DETOUR
        assert [s.ratio for s in result.ladder_steps] == (
            pytest.approx([9054.0 / straight] * 4))
        assert "network_detour" in result.detail

    def test_no_route_at_any_pad_exhausts_the_ladder(
            self, tmp_path, monkeypatch):
        specs = [("disconnected",)] * 4
        result, scripted = _route(tmp_path, monkeypatch, PH, specs)
        assert scripted.calls == 4
        assert not result.ok
        assert result.reason == "disconnected"
        assert result.ladder_stop == "ceiling"
        assert result.detour_class == DETOUR_NETWORK_DETOUR
        assert [s.outcome for s in result.ladder_steps] == (
            ["disconnected"] * 4)
        assert all(s.distance_m is None and s.ratio is None
                   for s in result.ladder_steps)
        # The final answer carries the widest attempt's record.
        assert result.corridor_pad_deg == 0.08
        assert result.widened is True
        assert result.from_snap_m == 0.0
        assert "corridor widening" in result.detail


class TestBestAcrossAttempts:
    """The final answer is the best measured, never the latest."""

    def test_worse_widened_result_never_replaces_better_first(
            self, tmp_path, monkeypatch):
        specs = [("route", 8000.0), ("route", 9500.0),
                 ("route", 8200.0), ("route", 8100.0)]
        result, scripted = _route(tmp_path, monkeypatch, LAGOS, specs)
        assert scripted.calls == 4
        assert result.ok and result.distance_m == 8000.0
        assert result.widened is False
        assert result.corridor_pad_deg == 0.01
        assert result.detour_class == DETOUR_NETWORK_DETOUR
        assert result.ladder_steps[1].distance_m == 9500.0


class TestTileCap:
    """The cap is checked before a step, never after the spend."""

    def test_cap_refuses_the_step_before_attempting_it(
            self, tmp_path, monkeypatch):
        floors = Router._ladder_floors()
        counts = [len(_corridor(*LAGOS, pad_floor=floor).tiles(14))
                  for floor in floors]
        # Tile quantization: small pad growth can stay inside
        # the same tiles; the ladder's later floors do exceed.
        assert counts[-1] > counts[0]
        cap = counts[0]
        monkeypatch.setattr(routing, "MAX_CORRIDOR_TILES", cap)
        result, scripted = _route(
            tmp_path, monkeypatch, LAGOS, [("route", 11702.0)])
        attempted = sum(1 for count in counts if count <= cap)
        assert 0 < attempted < len(floors)
        # Every attempted step was under the cap, and the first
        # over-cap step was refused BEFORE materializing: the
        # materializer ran exactly once per recorded step.
        assert scripted.calls == attempted
        assert len(result.ladder_steps) == attempted
        assert all(step.tiles <= cap for step in result.ladder_steps)
        assert result.ladder_stop == "tile-cap"
        assert result.ok and result.distance_m == 11702.0
        assert result.detour_class == DETOUR_NETWORK_DETOUR

    def test_first_attempt_over_cap_is_the_shipped_refusal(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(routing, "MAX_CORRIDOR_TILES", 0)
        result, scripted = _route(
            tmp_path, monkeypatch, LAGOS, [("route", 1000.0)])
        assert scripted.calls == 0
        assert not result.ok
        assert result.reason == "corridor-too-large"
        assert "corridor bound" in result.detail
        # The ladder never engaged: no steps, no classification.
        assert result.ladder_steps == ()
        assert result.detour_class == ""
        assert result.ladder_stop == ""
