"""Live-routing federation proofs (census C3): OSRM, then Valhalla,
behind the local routing plane — local-first, honest degradation,
rung naming.

Every live call is stubbed at the urllib layer; the request and
response shapes below were verified against the real APIs before
this suite was written (a live OSRM route + NoSegment + NoRoute
smoke calls, and live Valhalla route + no-route smoke calls —
the Lagos route's distances/durations, the OSRM step fields,
the Valhalla maneuvers/summary and its real encoded shape are
the fixtures).  No test here touches the network.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.agents.map import MapAgent
from nomorals.location.primitives import LatLon
from nomorals.maps.definitions import _NIGERIA_BBOX, get_source
from nomorals.maps.manifest import build_manifest, write_manifest
from nomorals.maps.providers import (
    USER_AGENT,
    LiveRouteError,
    LiveRouteProvider,
    OSRMProvider,
    ValhallaProvider,
    _decode_polyline,
    live_route_provider,
)
from nomorals.maps.receipts import MapReceiptStore
from nomorals.maps.routing import REASON_DISCONNECTED, Router
from tests.test_d1921m5_routing import (
    A_LL,
    C_LL,
    E_LL,
    _network_pack,
)

#: The Lagos pair the live smoke calls routed (Osborne Road →
#: Glover Road, Lagos Island): OSRM 15,234.7 m / 895.7 s,
#: Valhalla 15,286 m / 2,183.5 s.
LAGOS_START = LatLon(6.5244, 3.3792)
LAGOS_END = LatLon(6.4531, 3.4356)

#: London → Paris, outside every installed pack (the
#: federation-reaches-it pair for the agent-level tests).
LONDON = LatLon(51.5074, -0.1278)
PARIS = LatLon(48.8566, 2.3522)

# ── fixtures: live-shaped payloads ──────────────────────────────

#: An OSRM route answer in the live shape (Lagos, 2026-11-12),
#: trimmed to the fields the provider reads: the route's
#: distance/duration, a 6-coordinate geometry (the live route's
#: first four + last two), three steps (the live route's first
#: three), and the live waypoints with their snap distances.
OSRM_OK = {
    "code": "Ok",
    "routes": [{
        "distance": 15234.7, "duration": 895.7,
        "geometry": {"type": "LineString", "coordinates": [
            [3.379291, 6.524442], [3.379384, 6.524241],
            [3.379676, 6.52437], [3.379746, 6.52422],
            [3.435494, 6.453075], [3.435581, 6.453045]]},
        "legs": [{
            "distance": 15234.7, "duration": 895.7,
            "steps": [
                {"name": "", "mode": "driving",
                 "distance": 24.5, "duration": 8.6},
                {"name": "", "mode": "driving",
                 "distance": 35.3, "duration": 7.0},
                {"name": "Eletu Odibo Street", "mode": "driving",
                 "distance": 322.7, "duration": 36.0},
            ]}],
    }],
    "waypoints": [
        {"name": "", "distance": 11.1,
         "location": [3.379291, 6.524442]},
        {"name": "Glover Road", "distance": 6.4,
         "location": [3.435581, 6.453045]},
    ],
}

#: The live OSRM bodies for the two honest empty answers
#: (Malabo → Douala, 2026-11-12): a disconnected pair answers
#: NoRoute; a point with no road inside the snap radius answers
#: NoSegment.  Both ride HTTP 400.
OSRM_NOROUTE = {"message": "Impossible route between points",
                "code": "NoRoute"}
OSRM_NOSEGMENT = {
    "message": "Could not find a matching segment for coordinate 0",
    "code": "NoSegment"}

#: A Valhalla route answer in the live shape (Lagos,
#: 2026-11-12): the trip summary, the leg's REAL encoded shape
#: (precision 6 — the decoder's proof artifact), and the live
#: leg's first three maneuvers, trimmed to the fields the
#: provider reads.
VALHALLA_OK = {
    "trip": {
        "status": 0, "status_message": "Found route between points",
        "units": "kilometers",
        "summary": {"time": 2183.488, "length": 15.286},
        "legs": [{
            "summary": {"time": 2183.488, "length": 15.286},
            "shape": "q`fmKudgmEnKwDaGiQjHkCbL{D`H{BjJuCjWqHl_@cKz]iLdXcIfLgCxHhUnQri@~Jz[rQbl@~Sxq@tEpQ|BxF~A`CzB|AzMbFzVfIlV`JxJrCfN`Edf@|NnBh@lBx@lBj@rYzIzc@zNfDwLlKw]bC_HrFwMbCsClDyB|DsA`JeDpMsDr]iK~AbExE~P|FbR`Mr_@w@`LA~B`F}A|S_G`SgFzNgE|McEpTcHnQ_FnPcFrSiG`MkDtNgEhNgE|VeHn[iJhQwEpOiF`UaH~UkHjNcE`UyGjKeC`C]vCA`E?hG^vJ|A~^xEbPbB`i@~GlRtBrO`BxG|@bVrCfXxDrXzCzHdAzWxCdRxBzDj@jALxNdBzTzCfTfChHhApENxCP|E@lFIxFa@pHcBlI}BjMiEp\\cK~c@_Ojb@oMfUuHnEsAfa@sLbW}Hh^qKdWmIzD}BdEmDzDcFpBcEbGoPnDcJnB_FnCkDbD_CbD}AlEcBdPsEbTmGl\\iJ`FsA`f@kMbb@iL`XoHz[{JhDcAvJoCd\\}Jp]gJp`@kJnH_BtGuA`Em@Y{@YgAOk@GUeDoI_Lk[q@sBoIoTgHqTeEeOwDuO{CgOkC{OmCsSsBmRwAuV[gVPsSz@}RzAmT~@uN`Gs`@nCaN|BaJ~EuQrFwOpCmIdDaHxGeNbH{MvGuLzHoLjHaJzFkGrZuY|]o[rn@qh@bVoNpYkSn]sVpa@cZ`ZqS~UaJxQmDhIcApKc@~TGbJv@xIlA~KnBhRjGpPxHz^nP~Al@xBv@fDlAzXfJrs@`\\zYjMlPdHlMrFtOhGzPrGtLdEhOnFpOdFnK`D`PxEnKdCvNnDdLxCjOfDtLlC`MhChLjCtOlCpMrBvJhAdLhBrZrDpu@vGzh@nD~Lf@bg@nAtu@n@vz@eAdo@mC`i@cDhz@qIpw@sKhi@oJjp@uOty@kVde@wPjo@}Wzc@sVve@cYbj@mb@tU_U^e@`\\k`@|a@ae@xx@aaAhg@iu@v\\{j@|T_a@tY{j@xCcKfHmXjEiPnGyXbFkWbGwb@`BuQ\\wDnBmXpBmi@nAic@dAsa@ZuKrAwYdBuSbBiMdBwKzB}KtCaLhJaYrIwPvImOnHiKrHsIpHmI~I_InMwJtG_FxIaGdUyJbLyDzG{@~AOtEa@tFCzDt@bDjAbF|EnAdBz@xBv@lC\\fDK`Fa@|CuA|DcB~CaDdCsE`CuDr@yDRmFYgBa@uBaAqBoAkBcA}HkIeDyI_EuK_F}LuG_MqGuKgVc`@eYye@kZeg@cIcNcIwMoLcUyp@}fAuPuXo^kn@cq@oiA_b@et@iVec@wD{GcCiFmNmZgJcSiEeJcEmJmMm]sFwOsG}RgGeRaHoUuDeOwDoNwCwMwDmRaDwQiCuPuAaNmAoM_A}Pg@qSYaPDgMDoMTiLv@oLv@eLdAyKnAyLlC{OpBeMxBiLrJka@dOyk@f[koAp[geA|HsYvAsEhFuPxIi\\zU_bAn^y|AfJs`@bBeD`D}E|CwCzA_ArCeAjDg@~DStFHpFx@bFnA~SzF~_AvVhhA`Z~eAhYrhAdYlBgHrFoR`T_w@bTqw@vKeb@lI_]jSgu@vFsSz@mD",
            "maneuvers": [
                {"type": 2, "instruction": "Drive southeast.",
                 "time": 7.968, "length": 0.024,
                 "begin_shape_index": 0, "end_shape_index": 1,
                 "travel_mode": "drive", "travel_type": "car"},
                {"type": 1, "instruction": "Turn left.",
                 "time": 16.729, "length": 0.035,
                 "begin_shape_index": 1, "end_shape_index": 2,
                 "travel_mode": "drive", "travel_type": "car"},
                {"type": 9,
                 "instruction": ("Turn right onto Eletu Odibo "
                                 "Street."),
                 "time": 83.373, "length": 0.324,
                 "begin_shape_index": 2, "end_shape_index": 11,
                 "travel_mode": "drive", "travel_type": "car"},
            ],
        }],
    },
}

#: The live Valhalla no-route bodies (2026-11-12): error 442
#: for a disconnected pair (Malabo → Douala), error 171 for
#: points with no road near them (mid-ocean).  Both ride HTTP 400.
VALHALLA_NO_PATH = {
    "error_code": 442, "error": "No path could be found for input",
    "status_code": 400, "status": "Bad Request"}
VALHALLA_NO_EDGES = {
    "error_code": 171, "error": "No suitable edges near location",
    "status_code": 400, "status": "Bad Request"}

#: A documented Valhalla failure that is NOT a no-route: error
#: 130, "Failed to parse location" (the route API reference's
#: error table).
VALHALLA_BAD_LOCATION = {
    "error_code": 130, "error": "Failed to parse location",
    "status_code": 400, "status": "Bad Request"}


# ── the HTTP stub ───────────────────────────────────────────────


class _FakeResponse:
    """A urlopen answer: canned bytes behind a context manager."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _HttpStub:
    """Records every (url, data, headers) call and answers from
    a queue of payloads-or-exceptions, in order.  An exhausted
    queue raises — a rung calling the network when the test
    says it must not is a failure, not a silent pass."""

    def __init__(self, answers) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, bytes | None, dict]] = []

    def __call__(self, request, timeout=None):
        url = request.full_url \
            if hasattr(request, "full_url") else str(request)
        data = getattr(request, "data", None)
        headers = {k.lower(): v for k, v
                   in request.header_items()}
        self.calls.append((url, data, headers))
        if not self.answers:
            raise AssertionError(
                "unexpected HTTP call: " + url)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return _FakeResponse(answer)


def _json(payload) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _http_error(code: int, payload=None) -> urllib.error.HTTPError:
    """An HTTPError carrying ``payload`` as its JSON body (the
    no-route codes ride HTTP 400 with a body the rung reads)."""
    body = _json(payload) if payload is not None else b""
    return urllib.error.HTTPError(
        "https://example.invalid/", code, "boom", {},
        io.BytesIO(body))


def _chain(stub) -> LiveRouteProvider:
    return LiveRouteProvider(
        osrm=OSRMProvider(), valhalla=ValhallaProvider())


class _ExplodingProvider:
    """A provider that must never be called (local-first proof)."""

    name = "exploding"

    def route(self, start, end):
        raise AssertionError("provider called on a local answer")


# ── OSRM: request shape + parsing ───────────────────────────────


class TestOSRMRequestShape:
    def test_route_url_params_ua_and_result(self):
        stub = _HttpStub([_json(OSRM_OK)])
        with mock.patch("urllib.request.urlopen", stub):
            found = OSRMProvider().route(LAGOS_START, LAGOS_END)
        assert len(stub.calls) == 1
        url, data, headers = stub.calls[0]
        assert data is None  # a GET
        assert headers["user-agent"] == USER_AGENT
        assert url.startswith(
            "https://router.project-osrm.org/route/v1/driving/")
        # lon,lat order, driving profile, full geometry + steps,
        # snap radius 500 m on both waypoints.
        assert "3.379200,6.524400;3.435600,6.453100" in url
        assert "overview=full" in url
        assert "geometries=geojson" in url
        assert "steps=true" in url
        assert "radiuses=500;500" in url
        assert found is not None and found.ok
        assert found.provider == "osrm"
        assert found.distance_m == 15234.7
        assert found.duration_s == 895.7
        # The geometry starts at the requested start and ends at
        # the requested end, the way the local plane's does.
        assert found.geometry[0] == (6.5244, 3.3792)
        assert found.geometry[-1] == (6.4531, 3.4356)
        assert len(found.geometry) == 8  # 6 coords + 2 endpoints
        assert found.from_snap_m == 11.1
        assert found.to_snap_m == 6.4
        assert found.pack_id == ""
        assert "OSRM demo server" in found.detail
        assert len(found.steps) == 3
        assert found.steps[0].name == "driving"
        assert found.steps[0].road_class == "driving"
        assert not found.steps[0].named
        assert found.steps[2].name == "Eletu Odibo Street"
        assert found.steps[2].named
        assert found.steps[2].length_m == 322.7

    def test_noroute_is_an_empty_answer(self):
        stub = _HttpStub(
            [_http_error(400, OSRM_NOROUTE)])
        with mock.patch("urllib.request.urlopen", stub):
            assert OSRMProvider().route(
                LAGOS_START, LAGOS_END) is None

    def test_nosegment_is_an_empty_answer(self):
        stub = _HttpStub(
            [_http_error(400, OSRM_NOSEGMENT)])
        with mock.patch("urllib.request.urlopen", stub):
            assert OSRMProvider().route(
                LAGOS_START, LAGOS_END) is None

    def test_other_code_is_a_rung_failure(self):
        stub = _HttpStub([_http_error(400, {
            "message": "Query string is not valid",
            "code": "InvalidQuery"})])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveRouteError):
            OSRMProvider().route(LAGOS_START, LAGOS_END)

    def test_http_status_is_a_rung_failure(self):
        stub = _HttpStub([_http_error(500)])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveRouteError):
            OSRMProvider().route(LAGOS_START, LAGOS_END)

    def test_ok_without_routes_is_a_rung_failure(self):
        stub = _HttpStub([_json({"code": "Ok", "routes": []})])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveRouteError):
            OSRMProvider().route(LAGOS_START, LAGOS_END)

    def test_bad_payload_is_a_rung_failure(self):
        stub = _HttpStub([b"not json {"])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveRouteError):
            OSRMProvider().route(LAGOS_START, LAGOS_END)


# ── Valhalla: request shape + parsing ───────────────────────────


class TestValhallaRequestShape:
    def test_route_posts_json_body_and_result(self):
        stub = _HttpStub([_json(VALHALLA_OK)])
        with mock.patch("urllib.request.urlopen", stub):
            found = ValhallaProvider().route(
                LAGOS_START, LAGOS_END)
        assert len(stub.calls) == 1
        url, data, headers = stub.calls[0]
        assert url == "https://valhalla1.openstreetmap.de/route"
        assert headers["user-agent"] == USER_AGENT
        assert headers["content-type"] == "application/json"
        body = json.loads(data.decode("utf-8"))
        assert body["costing"] == "auto"
        assert body["locations"] == [
            {"lat": 6.5244, "lon": 3.3792, "radius": 500},
            {"lat": 6.4531, "lon": 3.4356, "radius": 500}]
        assert found is not None and found.ok
        assert found.provider == "valhalla"
        assert found.distance_m == 15286.0
        assert found.duration_s == 2183.488
        assert found.geometry[0] == (6.5244, 3.3792)
        assert found.geometry[-1] == (6.4531, 3.4356)
        assert found.pack_id == ""
        assert "Valhalla demo server" in found.detail
        assert len(found.steps) == 3
        assert found.steps[0].name == "Drive southeast."
        assert found.steps[0].road_class == "drive"
        assert found.steps[0].length_m == 24.0
        assert found.steps[2].name == (
            "Turn right onto Eletu Odibo Street.")
        assert found.steps[2].length_m == 324.0
        # Snap distances come off the decoded shape's ends —
        # the same ~11 m / ~6 m the live OSRM waypoints report
        # for this pair.
        assert found.from_snap_m is not None
        assert 5.0 < found.from_snap_m < 20.0
        assert found.to_snap_m is not None
        assert 3.0 < found.to_snap_m < 12.0

    def test_no_path_code_is_an_empty_answer(self):
        stub = _HttpStub(
            [_http_error(400, VALHALLA_NO_PATH)])
        with mock.patch("urllib.request.urlopen", stub):
            assert ValhallaProvider().route(
                LAGOS_START, LAGOS_END) is None

    def test_no_edges_code_is_an_empty_answer(self):
        stub = _HttpStub(
            [_http_error(400, VALHALLA_NO_EDGES)])
        with mock.patch("urllib.request.urlopen", stub):
            assert ValhallaProvider().route(
                LAGOS_START, LAGOS_END) is None

    def test_other_error_code_is_a_rung_failure(self):
        stub = _HttpStub(
            [_http_error(400, VALHALLA_BAD_LOCATION)])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveRouteError):
            ValhallaProvider().route(LAGOS_START, LAGOS_END)

    def test_http_status_is_a_rung_failure(self):
        stub = _HttpStub([_http_error(500)])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveRouteError):
            ValhallaProvider().route(LAGOS_START, LAGOS_END)

    def test_200_with_nonzero_trip_status_is_a_rung_failure(self):
        stub = _HttpStub([_json({"trip": {
            "status": 400,
            "status_message": "No path could be found"}})])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveRouteError):
            ValhallaProvider().route(LAGOS_START, LAGOS_END)

    def test_leg_without_shape_is_a_rung_failure(self):
        payload = {"trip": {
            "status": 0,
            "summary": {"time": 10.0, "length": 1.0},
            "legs": [{"summary": {"time": 10.0, "length": 1.0},
                      "maneuvers": []}]}}
        stub = _HttpStub([_json(payload)])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveRouteError):
            ValhallaProvider().route(LAGOS_START, LAGOS_END)


# ── the polyline decoder ────────────────────────────────────────


class TestDecodePolyline:
    def test_live_shape_decodes_to_the_requested_ends(self):
        shape = VALHALLA_OK["trip"]["legs"][0]["shape"]
        points = _decode_polyline(shape)
        assert len(points) == 383
        first = points[0]
        assert abs(first[0] - 6.5244) < 0.001
        assert abs(first[1] - 3.3792) < 0.001
        last = points[-1]
        assert abs(last[0] - 6.4531) < 0.001
        assert abs(last[1] - 3.4356) < 0.001

    def test_empty_shape_decodes_to_nothing(self):
        assert _decode_polyline("") == ()

    def test_truncated_shape_is_a_rung_failure(self):
        # A lone continuation-bit char: the value never ends.
        with pytest.raises(LiveRouteError):
            _decode_polyline("~")


# ── the chain ───────────────────────────────────────────────────


class TestLiveRouteChain:
    def test_factory_returns_the_chain(self):
        assert live_route_provider().name == "osrm+valhalla"

    def test_chain_name_is_the_chain_before_any_answer(self):
        assert _chain(None).name == "osrm+valhalla"

    def test_osrm_hit_never_calls_valhalla(self):
        stub = _HttpStub([_json(OSRM_OK)])
        with mock.patch("urllib.request.urlopen", stub):
            found = _chain(stub).route(LAGOS_START, LAGOS_END)
        assert len(stub.calls) == 1
        assert "project-osrm.org" in stub.calls[0][0]
        assert found is not None and found.provider == "osrm"

    def test_chain_reports_the_rung_that_answered(self):
        stub = _HttpStub([_json(OSRM_OK)])
        chain = _chain(stub)
        with mock.patch("urllib.request.urlopen", stub):
            chain.route(LAGOS_START, LAGOS_END)
        assert chain.name == "osrm"

    def test_osrm_empty_falls_to_valhalla(self):
        stub = _HttpStub([
            _http_error(400, OSRM_NOROUTE),
            _json(VALHALLA_OK)])
        with mock.patch("urllib.request.urlopen", stub):
            found = _chain(stub).route(LAGOS_START, LAGOS_END)
        assert len(stub.calls) == 2
        assert found is not None and found.provider == "valhalla"

    def test_osrm_failure_falls_to_valhalla(self):
        stub = _HttpStub([
            _http_error(500),
            _json(VALHALLA_OK)])
        with mock.patch("urllib.request.urlopen", stub):
            found = _chain(stub).route(LAGOS_START, LAGOS_END)
        assert len(stub.calls) == 2
        assert found is not None and found.provider == "valhalla"

    def test_both_rungs_down_raises_naming_both(self):
        stub = _HttpStub([
            _http_error(500),
            _http_error(500)])
        with mock.patch("urllib.request.urlopen", stub), \
                pytest.raises(LiveRouteError) as excinfo:
            _chain(stub).route(LAGOS_START, LAGOS_END)
        message = str(excinfo.value)
        assert "osrm" in message and "valhalla" in message

    def test_osrm_empty_plus_valhalla_down_is_empty(self):
        stub = _HttpStub([
            _http_error(400, OSRM_NOROUTE),
            _http_error(500)])
        with mock.patch("urllib.request.urlopen", stub):
            assert _chain(stub).route(
                LAGOS_START, LAGOS_END) is None

    def test_both_empty_is_an_empty_answer(self):
        stub = _HttpStub([
            _http_error(400, OSRM_NOROUTE),
            _http_error(400, VALHALLA_NO_PATH)])
        with mock.patch("urllib.request.urlopen", stub):
            assert _chain(stub).route(
                LAGOS_START, LAGOS_END) is None


# ── the router seam ───────────────────────────────────────────────


class _RouterTmp(unittest.TestCase):
    """A Router over the m5 fixture grid (A—B—C primary, E—F an
    isolated lane), each test on its own tmp dir."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory(prefix="route-fed-")
        self.root = Path(self.tmp.name)
        self.packs = self.root / "packs"
        self.packs.mkdir(parents=True)
        self.store = MapReceiptStore(home=str(self.root / "home"))
        self.pack_dir = self.packs / "net"
        self.manifest = _network_pack(self.pack_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def _router(self, **kw):
        kw.setdefault("store", self.store)
        return Router(self.packs, **kw)

    def _route(self, start, end, **kw):
        return self._router(**kw).route(
            self.pack_dir, self.manifest,
            LatLon(*start), LatLon(*end))


class TestRouterLocalFirst(_RouterTmp):
    def test_local_route_never_calls_the_provider(self):
        res = self._route(
            A_LL, C_LL, provider=_ExplodingProvider(),
            federation=True)
        assert res.ok
        assert res.provider == "local"
        assert res.posture == "local"

    def test_local_no_route_falls_to_the_live_chain(self):
        stub = _HttpStub([_json(OSRM_OK)])
        # A→E is disconnected on the fixture grid (E sits on
        # an isolated lane), so the local answer is a no-route
        # and the chain answers.
        with mock.patch("urllib.request.urlopen", stub):
            res = self._route(
                A_LL, E_LL, provider=_chain(stub), federation=True)
        assert len(stub.calls) == 1
        assert res.ok
        assert res.provider == "osrm"
        assert res.posture == "federated"
        assert res.pack_id == ""
        assert res.reason == ""
        assert res.distance_m == 15234.7
        assert res.receipt is not None
        assert res.receipt.provider == "osrm"
        assert res.receipt.posture == "online"
        assert "answered by provider osrm" in res.detail

    def test_local_no_route_reason_stands_when_the_chain_is_empty(self):
        stub = _HttpStub([
            _http_error(400, OSRM_NOROUTE),
            _http_error(400, VALHALLA_NO_PATH)])
        with mock.patch("urllib.request.urlopen", stub):
            res = self._route(
                A_LL, E_LL, provider=_chain(stub), federation=True)
        assert not res.ok
        assert res.reason == REASON_DISCONNECTED
        assert "had no route either" in res.detail


# ── the agent surface ─────────────────────────────────────────────


def _vehicle_pack(packs_dir, *, pack_id="pack-a"):
    """An installed vector pack with no tiles: the federation
    vehicle — its corridor holds nothing, so the local attempt
    answers no-road-data and the Router falls to the provider."""
    pack_dir = Path(packs_dir) / pack_id
    pack_dir.mkdir(parents=True)
    write_manifest(
        pack_dir,
        build_manifest(
            pack_dir, pack_id=pack_id, name="Pack A",
            bbox=_NIGERIA_BBOX, zoom_min=0, zoom_max=14,
            attribution="© OpenStreetMap contributors",
            tile_format="mvt"))
    return pack_dir


def _agent_context(tmp_path):
    return SimpleNamespace(
        settings=SimpleNamespace(home_path=str(tmp_path)),
        db=None)


class TestAgentRouteFederation:
    def test_route_outside_the_packs_falls_to_osrm(
            self, tmp_path, monkeypatch):
        packs_dir = tmp_path / "packs"
        _vehicle_pack(packs_dir)
        monkeypatch.setenv("NM_MAPS_PACKS_DIR", str(packs_dir))
        monkeypatch.setenv("NM_MAPS_ROUTING_FEDERATION", "1")
        stub = _HttpStub([_json(OSRM_OK)])
        with mock.patch("urllib.request.urlopen", stub):
            ticket = MapAgent(_agent_context(tmp_path)).route(
                "51.5074,-0.1278", "48.8566,2.3522")
        assert len(stub.calls) == 1  # OSRM only, no Valhalla
        assert ticket["outcome"] == "ok"
        assert ticket["routed"] is True
        assert ticket["provider"] == "osrm"
        assert ticket["posture"] == "federated"
        assert ticket["pack_id"] == ""
        assert ticket["distance_m"] == 15234.7
        assert ticket["duration_s"] == 895.7
        assert len(ticket["steps"]) == 3
        assert ticket["geometry"][0] == [51.5074, -0.1278]
        assert ticket["geometry"][-1] == [48.8566, 2.3522]

    def test_route_outside_the_packs_federation_off_fetches_nothing(
            self, tmp_path, monkeypatch):
        packs_dir = tmp_path / "packs"
        _vehicle_pack(packs_dir)
        monkeypatch.setenv("NM_MAPS_PACKS_DIR", str(packs_dir))
        stub = _HttpStub([])
        with mock.patch("urllib.request.urlopen", stub):
            ticket = MapAgent(_agent_context(tmp_path)).route(
                "51.5074,-0.1278", "48.8566,2.3522")
        assert stub.calls == []
        assert ticket["outcome"] == "unavailable"
        assert ticket["routed"] is False
        assert "nothing was fetched from a provider" \
            in ticket["detail"]

    def test_local_route_never_calls_the_provider(
            self, tmp_path, monkeypatch):
        packs_dir = tmp_path / "packs"
        packs_dir.mkdir(parents=True)
        _network_pack(packs_dir / "net")
        monkeypatch.setenv("NM_MAPS_PACKS_DIR", str(packs_dir))
        monkeypatch.setenv("NM_MAPS_ROUTING_FEDERATION", "1")
        stub = _HttpStub([])
        with mock.patch("urllib.request.urlopen", stub):
            ticket = MapAgent(_agent_context(tmp_path)).route(
                f"{A_LL[0]},{A_LL[1]}", f"{C_LL[0]},{C_LL[1]}")
        assert stub.calls == []
        assert ticket["outcome"] == "ok"
        assert ticket["provider"] == "local"
        assert ticket["pack_id"] == "net-fixture"


# ── the registry ──────────────────────────────────────────────────


class TestRegistryRows:
    def test_live_router_rows_present_with_honest_notes(self):
        osrm = get_source("osrm-route")
        assert osrm.url == (
            "https://router.project-osrm.org/route/v1/driving/")
        assert osrm.attribution == "© OpenStreetMap contributors"
        assert osrm.license == "ODbL 1.0"
        assert "demo server" in osrm.notes
        assert "not production traffic" in osrm.notes
        assert "Fallback-only" in osrm.notes
        assert "NoSegment" in osrm.notes
        valhalla = get_source("valhalla-route")
        assert valhalla.url == (
            "https://valhalla1.openstreetmap.de/route")
        assert "costing auto" in valhalla.notes
        assert "14 s observed" in valhalla.notes
        assert "170/171/441/442" in valhalla.notes
        assert "Fallback-only" in valhalla.notes

    def test_sources_command_names_the_live_rows(self):
        from nomorals.cmdline.commands.maps import _sources

        class _Args:
            json = False

        seen: list[str] = []

        def _capture(_args, payload, text):
            seen.append(text)
            return 0

        with mock.patch("nomorals.cmdline.commands.maps._emit",
                        _capture):
            assert _sources(_Args(), None) == 0
        text = seen[0]
        assert "osrm-route — OSRM demo server" in text
        assert "valhalla-route — Valhalla demo instance" in text
        assert "live service — no artifact" in text
