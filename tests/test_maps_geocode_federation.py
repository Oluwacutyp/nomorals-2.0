"""Live-geocoder federation proofs (census B9): Nominatim, then
Overpass, behind the local gazetteer — local-first, honest
degradation, rung naming.

Every live call is stubbed at the urllib layer; the request and
response shapes below were verified against the real APIs before
this suite was written (Nominatim search + reverse docs and a
live search smoke call — its first two results are the search
fixture; the Overpass QL wiki and a live ``around`` response —
its shape is the reverse fixture).  No test here touches the
network.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.agents.map import MapAgent
from nomorals.maps.definitions import get_source
from nomorals.maps.gazetteer import Gazetteer, normalize_name
from nomorals.maps.manifest import BBox, build_manifest, write_manifest
from nomorals.maps.providers import (
    _NOMINATIM_LIMITER,
    USER_AGENT,
    LiveGeocodeError,
    LiveGeocodeProvider,
    NominatimProvider,
    OverpassProvider,
    _RateLimiter,
    live_geocode_provider,
)
from nomorals.maps.receipts import MapReceiptStore

# ── fixtures: live-shaped payloads ──────────────────────────────

#: The first two results of a live Nominatim search for
#: "Baker Street" (2026-11-12), trimmed to the fields the
#: provider reads.
NOMINATIM_SEARCH = [
    {
        "place_id": 262099875,
        "licence": "Data © OpenStreetMap contributors, ODbL 1.0.",
        "osm_type": "node", "osm_id": 261949787,
        "lat": "51.5225853", "lon": "-0.1567431",
        "category": "railway", "type": "station",
        "place_rank": 30, "importance": 0.645,
        "addresstype": "railway", "name": "Baker Street",
        "display_name": ("Baker Street, Marylebone Road, City of"
                         " Westminster, London, Greater London,"
                         " England, NW1 5LJ, United Kingdom"),
        "boundingbox": ["51.5225353", "51.5226353",
                        "-0.1567931", "-0.1566931"],
    },
    {
        "place_id": 259596780,
        "licence": "Data © OpenStreetMap contributors, ODbL 1.0.",
        "osm_type": "node", "osm_id": 260987987,
        "lat": "51.5231482", "lon": "-0.1585205",
        "category": "railway", "type": "station",
        "place_rank": 30, "importance": 0.594,
        "addresstype": "railway", "name": "Baker Street",
        "display_name": ("Baker Street, Baker Street, Marylebone"
                         " Road, City of Westminster, London,"
                         " Greater London, England, NW1 5LJ,"
                         " United Kingdom"),
        "boundingbox": ["51.5230982", "51.5231982",
                        "-0.1585705", "-0.1584705"],
    },
]

#: A Nominatim reverse answer in the documented jsonv2 shape
#: (single object; lat/lon as strings; name + display_name).
NOMINATIM_REVERSE = {
    "place_id": "134140761",
    "licence": "Data © OpenStreetMap contributors, ODbL 1.0.",
    "osm_type": "way", "osm_id": "280940520",
    "lat": "-34.4391708", "lon": "-58.7064573",
    "place_rank": "26", "category": "highway",
    "type": "motorway", "importance": "0.1",
    "addresstype": "road",
    "display_name": ("Autopista Pedro Eugenio Aramburu, El"
                     " Triángulo, Partido de Malvinas Argentinas,"
                     " Buenos Aires, 1.619, Argentina"),
    "name": "Autopista Pedro Eugenio Aramburu",
}

#: An Overpass forward-search answer in the documented shape:
#: a node, a way and a relation with centres, one nameless node
#: (skipped) and one centreless way (skipped).
OVERPASS_SEARCH = {
    "version": 0.6,
    "generator": "Overpass API 0.7.62.11 84e2e1a5",
    "osm3s": {"timestamp_osm_base": "2026-11-12T04:00:00Z",
              "copyright": "The data included in this document is"
                           " from www.openstreetmap.org. The data"
                           " is made available under ODbL."},
    "elements": [
        {"type": "node", "id": 261949787,
         "lat": 51.5225853, "lon": -0.1567431,
         "tags": {"name": "Baker Street",
                  "railway": "station"}},
        {"type": "way", "id": 123456,
         "center": {"lat": 51.5231482, "lon": -0.1585205},
         "tags": {"name": "Baker Street",
                  "highway": "residential"}},
        {"type": "relation", "id": 789,
         "center": {"lat": 51.53, "lon": -0.16},
         "tags": {"name": "Baker Street",
                  "type": "route", "route": "bus"}},
        {"type": "node", "id": 999,
         "lat": 51.5, "lon": -0.15,
         "tags": {"railway": "station"}},
        {"type": "way", "id": 555,
         "tags": {"name": "Baker Street",
                  "highway": "residential"}},
    ],
}

#: An Overpass ``around`` answer in the live shape (Oxford
#: Circus node + London node, 2026-11-12), trimmed.
OVERPASS_AROUND = {
    "version": 0.6,
    "generator": "Overpass API 0.7.62.11 84e2e1a5",
    "osm3s": {"timestamp_osm_base": "2026-11-12T04:00:00Z",
              "copyright": "The data included in this document is"
                           " from www.openstreetmap.org. The data"
                           " is made available under ODbL."},
    "elements": [
        {"type": "node", "id": 381987613,
         "lat": 51.5152312, "lon": -0.1418342,
         "tags": {"name": "Oxford Circus",
                  "railway": "station",
                  "station": "subway"}},
        {"type": "node", "id": 107775,
         "lat": 51.5073219, "lon": -0.1276474,
         "tags": {"name": "London",
                  "place": "city",
                  "capital": "yes"}},
    ],
}


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


def _nominatim(**kwargs) -> NominatimProvider:
    """A Nominatim rung that never waits (zero-interval limiter)."""
    return NominatimProvider(limiter=_RateLimiter(0.0), **kwargs)


def _chain(stub) -> LiveGeocodeProvider:
    return LiveGeocodeProvider(
        nominatim=_nominatim(), overpass=OverpassProvider())


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.invalid/", code, "boom", {}, None)


# ── Nominatim: request shape + parsing ──────────────────────────


class TestNominatimRequestShape:
    def test_search_url_params_ua_and_results(self):
        stub = _HttpStub([_json(NOMINATIM_SEARCH)])
        with mock.patch("urllib.request.urlopen", stub):
            found = _nominatim().search("Baker Street", limit=8)
        assert len(stub.calls) == 1
        url, data, headers = stub.calls[0]
        assert data is None  # a GET
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "nominatim.openstreetmap.org"
        assert parsed.path == "/search"
        params = urllib.parse.parse_qs(parsed.query)
        assert params == {
            "q": ["Baker Street"], "format": ["jsonv2"],
            "limit": ["8"], "addressdetails": ["0"]}
        assert headers["user-agent"] == USER_AGENT
        assert len(found) == 2
        first, second = found
        assert first.name == "Baker Street"
        assert first.kind == "poi"  # a station is a POI, not a place
        assert first.lat == pytest.approx(51.5225853)
        assert first.lon == pytest.approx(-0.1567431)
        assert first.pack_id == ""  # not from a pack
        assert first.matched_as == "provider"
        assert first.score == pytest.approx(400.0)
        assert second.score == pytest.approx(390.0)

    def test_search_empty_query_never_calls_http(self):
        stub = _HttpStub([])
        with mock.patch("urllib.request.urlopen", stub):
            assert _nominatim().search("   ") == ()
            assert _nominatim().search("") == ()
        assert stub.calls == []

    def test_search_http_status_is_a_rung_failure(self):
        for failure in (
                _http_error(429), _http_error(503),
                urllib.error.URLError("no route to host")):
            stub = _HttpStub([failure])
            with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveGeocodeError):
                _nominatim().search("Baker Street")

    def test_search_bad_payload_is_a_rung_failure(self):
        for payload in (b"not json {", _json({"error": "nope"}),
                        _json("a string")):
            stub = _HttpStub([payload])
            with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveGeocodeError):
                _nominatim().search("Baker Street")

    def test_search_skips_results_without_coordinates(self):
        payload = [{"name": "Nowhere", "category": "place",
                    "type": "x", "addresstype": "city"},
                   {"name": "Baker Street", "lat": "51.5",
                    "lon": "-0.15", "category": "railway",
                    "type": "station", "addresstype": "railway"}]
        stub = _HttpStub([_json(payload)])
        with mock.patch("urllib.request.urlopen", stub):
            found = _nominatim().search("Baker Street")
        assert [c.name for c in found] == ["Baker Street"]

    def test_reverse_url_params_and_result(self):
        stub = _HttpStub([_json(NOMINATIM_REVERSE)])
        with mock.patch("urllib.request.urlopen", stub):
            found = _nominatim().reverse(-34.4391708, -58.7064573)
        assert len(stub.calls) == 1
        url, _, headers = stub.calls[0]
        parsed = urllib.parse.urlparse(url)
        assert parsed.path == "/reverse"
        params = urllib.parse.parse_qs(parsed.query)
        assert params == {
            "lat": ["-34.439171"], "lon": ["-58.706457"],
            "format": ["jsonv2"], "zoom": ["18"]}
        assert headers["user-agent"] == USER_AGENT
        assert len(found) == 1
        assert found[0].name == "Autopista Pedro Eugenio Aramburu"
        assert found[0].kind == "place"  # a road is a place, not a POI
        assert found[0].distance_km == pytest.approx(0.0, abs=0.01)

    def test_reverse_documented_error_is_an_empty_answer(self):
        stub = _HttpStub([_json({"error": "Unable to geocode"})])
        with mock.patch("urllib.request.urlopen", stub):
            assert _nominatim().reverse(0.0, 0.0) == ()

    def test_reverse_non_object_is_a_rung_failure(self):
        stub = _HttpStub([_json([])])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveGeocodeError):
            _nominatim().reverse(0.0, 0.0)


# ── the 1-request-per-second throttle ───────────────────────────


class TestRateLimiter:
    def test_default_nominatim_limiter_is_one_per_second(self):
        assert _NOMINATIM_LIMITER.min_interval_s == 1.0
        assert NominatimProvider().limiter is None  # shares it

    def test_wait_spaces_calls_by_the_interval(self):
        limiter = _RateLimiter(0.05)
        started = time.monotonic()
        limiter.wait()
        limiter.wait()
        assert time.monotonic() - started >= 0.04

    def test_zero_interval_never_waits(self):
        limiter = _RateLimiter(0.0)
        started = time.monotonic()
        limiter.wait()
        limiter.wait()
        assert time.monotonic() - started < 0.05


# ── Overpass: QL shape + parsing ───────────────────────────────


class TestOverpassRequestShape:
    def test_search_posts_exact_name_ql(self):
        stub = _HttpStub([_json(OVERPASS_SEARCH)])
        with mock.patch("urllib.request.urlopen", stub):
            found = OverpassProvider().search("Baker Street",
                                              limit=8)
        assert len(stub.calls) == 1
        url, data, headers = stub.calls[0]
        assert url == "https://overpass-api.de/api/interpreter"
        assert headers["user-agent"] == USER_AGENT
        form = urllib.parse.parse_qs(data.decode("utf-8"))
        query = form["data"][0]
        assert query.startswith("[out:json][timeout:25];")
        # Exact-name, never the planet-scan regex form.
        assert 'nwr["name"="Baker Street",i];' in query
        assert "~" not in query
        assert query.endswith("out center 8;")
        assert [c.name for c in found] == [
            "Baker Street", "Baker Street", "Baker Street"]
        assert found[0].kind == "poi"  # railway station
        assert found[0].lat == pytest.approx(51.5225853)
        assert found[1].lat == pytest.approx(51.5231482)  # way centre
        assert found[2].lat == pytest.approx(51.53)  # relation centre
        assert all(c.pack_id == "" and c.matched_as == "provider"
                   for c in found)

    def test_search_escapes_quotes_in_the_query(self):
        stub = _HttpStub([_json({"elements": []})])
        with mock.patch("urllib.request.urlopen", stub):
            OverpassProvider().search('Say "hi"')
        form = urllib.parse.parse_qs(
            stub.calls[0][1].decode("utf-8"))
        assert 'nwr["name"="Say \\"hi\\"",i];' in form["data"][0]

    def test_search_empty_query_never_calls_http(self):
        stub = _HttpStub([])
        with mock.patch("urllib.request.urlopen", stub):
            assert OverpassProvider().search("  ") == ()
        assert stub.calls == []

    def test_search_capacity_502_is_a_rung_failure(self):
        stub = _HttpStub([_http_error(502)])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveGeocodeError):
            OverpassProvider().search("Baker Street")

    def test_search_bad_payload_is_a_rung_failure(self):
        stub = _HttpStub([_json({"version": 0.6})])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveGeocodeError):
            OverpassProvider().search("Baker Street")

    def test_reverse_steps_the_ladder_and_stops_at_first_hit(self):
        stub = _HttpStub([_json({"elements": []}),
                          _json(OVERPASS_AROUND)])
        with mock.patch("urllib.request.urlopen", stub):
            found = OverpassProvider().reverse(
                51.5225853, -0.1567431)
        assert len(stub.calls) == 2  # 1 km empty, 5 km hit
        first_ql = urllib.parse.parse_qs(
            stub.calls[0][1].decode("utf-8"))["data"][0]
        second_ql = urllib.parse.parse_qs(
            stub.calls[1][1].decode("utf-8"))["data"][0]
        assert "around:1000,51.522585,-0.156743" in first_ql
        assert "around:5000,51.522585,-0.156743" in second_ql
        assert '["name"]' in first_ql
        assert [c.name for c in found] == [
            "Oxford Circus", "London"]  # nearest first
        assert found[0].distance_km == pytest.approx(1.3, abs=0.2)
        assert found[1].distance_km == pytest.approx(2.6, abs=0.3)

    def test_reverse_drops_candidates_outside_the_rung(self):
        # 1.9 km away: outside the 1 km rung, inside the 5 km one.
        near_but_far = {"elements": [
            {"type": "node", "id": 1, "lat": 51.54, "lon": -0.1567,
             "tags": {"name": "Too Far", "place": "suburb"}}]}
        stub = _HttpStub([_json(near_but_far),
                          _json(OVERPASS_AROUND)])
        with mock.patch("urllib.request.urlopen", stub):
            found = OverpassProvider().reverse(
                51.5225853, -0.1567431)
        assert len(stub.calls) == 2
        assert "Too Far" not in [c.name for c in found]

    def test_reverse_all_rungs_empty_is_an_empty_answer(self):
        stub = _HttpStub([_json({"elements": []})] * 4)
        with mock.patch("urllib.request.urlopen", stub):
            assert OverpassProvider().reverse(0.0, 0.0) == ()
        assert len(stub.calls) == 4  # 1/5/25/100 km, all tried


# ── the chain ───────────────────────────────────────────────────


class TestLiveChain:
    def test_chain_name_is_the_chain_before_any_answer(self):
        assert live_geocode_provider().name == "nominatim+overpass"

    def test_nominatim_hit_never_calls_overpass(self):
        stub = _HttpStub([_json(NOMINATIM_SEARCH)])
        with mock.patch("urllib.request.urlopen", stub):
            found = _chain(stub).search("Baker Street")
        assert len(stub.calls) == 1  # Nominatim only
        assert found[0].name == "Baker Street"

    def test_chain_reports_the_rung_that_answered(self):
        stub = _HttpStub([_json(NOMINATIM_SEARCH)])
        with mock.patch("urllib.request.urlopen", stub):
            chain = _chain(stub)
            chain.search("Baker Street")
        assert chain.name == "nominatim"

    def test_nominatim_empty_falls_to_overpass(self):
        stub = _HttpStub([_json([]), _json(OVERPASS_SEARCH)])
        with mock.patch("urllib.request.urlopen", stub):
            chain = _chain(stub)
            found = chain.search("Baker Street")
        assert len(stub.calls) == 2
        assert [c.name for c in found] == ["Baker Street"] * 3
        assert chain.name == "overpass"

    def test_nominatim_failure_falls_to_overpass(self):
        stub = _HttpStub([_http_error(429), _json(OVERPASS_SEARCH)])
        with mock.patch("urllib.request.urlopen", stub):
            chain = _chain(stub)
            found = chain.search("Baker Street")
        assert len(found) == 3
        assert chain.name == "overpass"

    def test_both_rungs_down_raises_naming_both(self):
        stub = _HttpStub([_http_error(429), _http_error(502)])
        with mock.patch("urllib.request.urlopen", stub), pytest.raises(LiveGeocodeError) as excinfo:
            _chain(stub).search("Baker Street")
        message = str(excinfo.value)
        assert "nominatim" in message and "overpass" in message

    def test_nominatim_empty_plus_overpass_down_is_empty(self):
        # Nominatim's honest empty answer stands; the second
        # rung's failure is logged, never a fake answer.
        stub = _HttpStub([_json([]), _http_error(502)])
        with mock.patch("urllib.request.urlopen", stub):
            assert _chain(stub).search("Baker Street") == ()

    def test_both_empty_is_an_empty_answer(self):
        stub = _HttpStub([_json([]), _json({"elements": []})])
        with mock.patch("urllib.request.urlopen", stub):
            assert _chain(stub).search("Baker Street") == ()

    def test_reverse_chain_nominatim_first(self):
        stub = _HttpStub([_json(NOMINATIM_REVERSE)])
        with mock.patch("urllib.request.urlopen", stub):
            chain = _chain(stub)
            found = chain.reverse(-34.4391708, -58.7064573)
        assert len(stub.calls) == 1
        assert found[0].name == "Autopista Pedro Eugenio Aramburu"
        assert chain.name == "nominatim"

    def test_reverse_chain_falls_to_overpass_around(self):
        stub = _HttpStub([_json({"error": "Unable to geocode"}),
                          _json({"elements": []}),
                          _json(OVERPASS_AROUND)])
        with mock.patch("urllib.request.urlopen", stub):
            chain = _chain(stub)
            found = chain.reverse(51.5225853, -0.1567431)
        assert len(stub.calls) == 3  # nominatim + 2 around rungs
        assert found[0].name == "Oxford Circus"
        assert chain.name == "overpass"


# ── the gazetteer seam ──────────────────────────────────────────


def _seed_index(root, *, pack_id="pack-a", name="Lagos Island",
                kind="place", lat=6.45, lon=3.4):
    """A real Gazetteer index over ``root`` with one recorded
    pack + one entry — white-box rows, no MVT tile build."""
    gazetteer = Gazetteer(root)
    conn = gazetteer._connect(create=True)
    assert conn is not None
    try:
        conn.execute(
            "INSERT INTO gazetteer_packs (pack_id, digest, built_at,"
            " entries, tiles_scanned, corrupt_tiles)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (pack_id, "seed-digest", 0.0, 1, 0, 0))
        conn.execute(
            "INSERT INTO gazetteer_entries (pack_id, name,"
            " indexed_name, norm, kind, lat, lon, source_zoom,"
            " population) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pack_id, name, name, normalize_name(name), kind,
             lat, lon, 7, 0))
        conn.commit()
    finally:
        conn.close()
    return gazetteer


class _ExplodingProvider:
    """A provider that must never be called (local-first proof)."""

    name = "exploding"

    def search(self, query, *, limit=8):
        raise AssertionError("provider called on a local hit")

    def reverse(self, lat, lon, *, limit=8):
        raise AssertionError("provider called on a local hit")


class TestGazetteerFederationSeam:
    def test_federation_off_never_calls_the_provider(self, tmp_path):
        _seed_index(tmp_path)
        gazetteer = Gazetteer(
            tmp_path, provider=_ExplodingProvider(), federation=False)
        result = gazetteer.search("Baker Street")
        assert result.candidates == ()
        assert result.posture == "local"
        assert result.provider == "local"
        assert "federation off" in result.note

    def test_federation_on_local_hit_never_calls_the_provider(
            self, tmp_path):
        _seed_index(tmp_path)
        gazetteer = Gazetteer(
            tmp_path, provider=_ExplodingProvider(), federation=True)
        result = gazetteer.search("Lagos Island")
        assert [c.name for c in result.candidates] == ["Lagos Island"]
        assert result.posture == "local"
        assert result.provider == "local"

    def test_federation_on_local_miss_reaches_nominatim(self, tmp_path):
        _seed_index(tmp_path)
        stub = _HttpStub([_json(NOMINATIM_SEARCH)])
        gazetteer = Gazetteer(
            tmp_path, provider=_chain(stub), federation=True,
            store=MapReceiptStore(db=None, home=str(tmp_path)))
        with mock.patch("urllib.request.urlopen", stub):
            result = gazetteer.search("Baker Street")
        assert len(stub.calls) == 1
        assert result.posture == "federated"
        assert result.provider == "nominatim"
        assert result.candidates[0].name == "Baker Street"
        assert "answered by provider nominatim" in result.note
        assert result.receipt is not None
        assert result.receipt.provider == "nominatim"
        assert result.receipt.posture == "online"

    def test_federation_on_total_failure_names_it_honestly(
            self, tmp_path):
        _seed_index(tmp_path)
        stub = _HttpStub([_http_error(429), _http_error(502)])
        gazetteer = Gazetteer(
            tmp_path, provider=_chain(stub), federation=True)
        with mock.patch("urllib.request.urlopen", stub):
            result = gazetteer.search("Baker Street")
        assert result.candidates == ()
        assert result.posture == "federated"
        assert result.provider == "nominatim+overpass"
        assert "failed" in result.note
        assert "local answer stands" in result.note

    def test_federation_on_reverse_reaches_nominatim(self, tmp_path):
        _seed_index(tmp_path)
        stub = _HttpStub([_json(NOMINATIM_REVERSE)])
        gazetteer = Gazetteer(
            tmp_path, provider=_chain(stub), federation=True)
        with mock.patch("urllib.request.urlopen", stub):
            result = gazetteer.reverse(-34.4391708, -58.7064573)
        assert result.posture == "federated"
        assert result.provider == "nominatim"
        assert result.candidates[0].name == (
            "Autopista Pedro Eugenio Aramburu")

    def test_federation_on_reverse_local_hit_never_calls_provider(
            self, tmp_path):
        _seed_index(tmp_path)
        gazetteer = Gazetteer(
            tmp_path, provider=_ExplodingProvider(), federation=True)
        result = gazetteer.reverse(6.45, 3.4)
        assert [c.name for c in result.candidates] == ["Lagos Island"]
        assert result.posture == "local"


# ── the agent surface ───────────────────────────────────────────


class TestAgentFindFederation:
    def test_find_federates_and_names_the_rung(
            self, tmp_path, monkeypatch):
        packs_dir = tmp_path / "packs"
        pack_dir = packs_dir / "pack-a"
        pack_dir.mkdir(parents=True)
        (pack_dir / "tile-0-0-0.png").write_bytes(b"\x89PNG")
        write_manifest(
            pack_dir,
            build_manifest(
                pack_dir, pack_id="pack-a", name="Pack A",
                bbox=BBox(west=-180.0, south=-90.0, east=180.0,
                          north=90.0),
                zoom_min=0, zoom_max=0,
                attribution="© OpenStreetMap contributors"))
        monkeypatch.setenv("NM_MAPS_PACKS_DIR", str(packs_dir))
        monkeypatch.setenv("NM_MAPS_GEOCODE_FEDERATION", "1")
        context = SimpleNamespace(
            settings=SimpleNamespace(home_path=str(tmp_path)),
            db=None)
        stub = _HttpStub([_json(NOMINATIM_SEARCH)])
        with mock.patch("urllib.request.urlopen", stub):
            ticket = MapAgent(context).find("Baker Street")
        assert len(stub.calls) == 1  # Nominatim only, no Overpass
        assert ticket["outcome"] == "ok"
        assert ticket["posture"] == "federated"
        assert ticket["provider"] == "nominatim"
        assert ticket["candidates"][0]["name"] == "Baker Street"
        assert ticket["candidates"][0]["matched_as"] == "provider"


# ── the registry ────────────────────────────────────────────────


class TestRegistryRows:
    def test_live_geocoder_rows_present_with_honest_notes(self):
        nominatim = get_source("nominatim-search")
        assert nominatim.url == (
            "https://nominatim.openstreetmap.org/search")
        assert nominatim.attribution == "© OpenStreetMap contributors"
        assert nominatim.license == "ODbL 1.0"
        assert "1 request/second" in nominatim.notes
        assert "Fallback-only" in nominatim.notes
        overpass = get_source("overpass-search")
        assert overpass.url == (
            "https://overpass-api.de/api/interpreter")
        assert "exact-name" in overpass.notes
        assert "Fallback-only" in overpass.notes

    def test_pre_existing_rows_carry_no_notes(self):
        assert get_source("geofabrik-nigeria").notes == ""
        assert get_source("planetiler-jar").notes == ""

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
        assert "nominatim-search — Nominatim geocoder" in text
        assert "live service — no artifact" in text
        assert "overpass-search — Overpass API" in text
        assert "geofabrik-nigeria — Geofabrik Nigeria" in text
        assert "sha256 pinned" in text
