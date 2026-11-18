"""52-websites #29 (Radio.Garden): live radio stations indexed by place.

Proof the brief demands, each pinned here:
- seed-directory fixtures -> place queries return the planted stations;
- genre + place filters combine;
- a dead stream URL degrades via source health, never a silent hang;
- a place with no stations answers honestly.

The 24 seeds are real stations from the open radio-browser
directory, every stream URL verified live on 2026-11-18 (a real
GET returned audio bytes). City anchors are the municipality rows
of her airport table; station geo is the directory's own where it
carries any.
"""

from __future__ import annotations

import json
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.music_sources.radio import (
    SEED_STATIONS,
    Station,
    StationStore,
    capture_sample,
    check_all,
    check_station,
    check_stream,
    format_radio_reply,
    globe_dots,
    grow_from_directory,
    parse_radio_request,
    search_stations,
    station_by_name,
    stations_by_genre,
    stations_in_place,
    stations_near,
)


@pytest.fixture()
def home(tmp_path):
    return tmp_path


@pytest.fixture()
def store(home):
    return StationStore(home / "data" / "radio_stations.json")


class _FakeHeaders:
    def __init__(self, content_type: str = "audio/mpeg") -> None:
        self._content_type = content_type

    def get_content_type(self) -> str:
        return self._content_type


class _FakeStream:
    """A stream of audio-ish chunks, readable like urlopen's."""

    def __init__(self, chunks, content_type: str = "audio/mpeg",
                 delay: float = 0.0) -> None:
        self._chunks = list(chunks)
        self.headers = _FakeHeaders(content_type)
        self.reads = 0
        self.delay = delay

    def read(self, n: int = -1) -> bytes:
        self.reads += 1
        if self.delay:
            time.sleep(self.delay)
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_MP3_CHUNK = b"\xff\xfb\x90\x00" + b"\x00" * 100


def _ctx(home):
    return SimpleNamespace(settings=SimpleNamespace(home=str(home)))


def _tool(home):
    from nomorals.tools import radio as radio_mod
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry(context=_ctx(home))
    radio_mod.register(reg)
    return reg._tools["radio"].fn


# ── seeds ───────────────────────────────────────────────────────────


def test_seeds_load_24(store):
    assert store.count() == 24
    assert len(SEED_STATIONS) == 24
    for station in store.all():
        assert station.city and station.country
        assert station.stream_url.startswith(("http://", "https://"))
        assert -90.0 <= station.lat <= 90.0
        assert -180.0 <= station.lon <= 180.0
        assert station.health == "unchecked"


def test_seed_cities(store):
    cities = {s.city for s in store.all()}
    assert cities == {
        "Lagos", "Accra", "Tokyo", "London", "Paris", "New York",
        "New Orleans", "Berlin", "Nairobi", "Johannesburg", "Kingston",
        "São Paulo", "Mumbai", "Sydney", "Cairo", "Toronto", "Vienna",
        "Amsterdam", "San Francisco", "Eureka",
    }


def test_station_validation():
    with pytest.raises(ValueError):
        Station(name="X", city="Lagos", country="Nigeria",
                lat=91.0, lon=3.0, stream_url="http://example.com/x.mp3")
    with pytest.raises(ValueError):
        Station(name="X", city="Lagos", country="Nigeria",
                lat=6.0, lon=3.0, stream_url="ftp://example.com/x.mp3")
    with pytest.raises(ValueError):
        Station(name="", city="Lagos", country="Nigeria",
                lat=6.0, lon=3.0, stream_url="http://example.com/x.mp3")


# ── place-first browsing ────────────────────────────────────────────


def test_place_lagos(store):
    found = stations_in_place(store, "Lagos")
    assert [s.name for s in found] == ["METRO FM LAGOS", "Wazobia FM: Lagos"]


def test_place_case_and_accent_insensitive(store):
    for query in ("lagos", "LAGOS", "Lágoṣ"):
        assert len(stations_in_place(store, query)) == 2


def test_place_unknown_answers_honestly(store):
    assert stations_in_place(store, "Nowhereville") == []
    reply = format_radio_reply([], place="Nowhereville")
    assert "don't have" in reply
    assert "Nowhereville" in reply


def test_place_accra_two(store):
    assert {s.name for s in stations_in_place(store, "Accra")} == {
        "YFm Accra", "Sompa 106.5FM Accra"}


# ── genre + place combine ───────────────────────────────────────────


def test_genre_place_jazz_new_orleans(store):
    found = stations_in_place(store, "New Orleans", genre="jazz")
    assert [s.name for s in found] == ["WWOZ 90.7 New Orleans"]


def test_genre_place_jazz_london(store):
    found = stations_in_place(store, "London", genre="jazz")
    assert [s.name for s in found] == ["Jazz London Radio"]


def test_genre_place_jazz_lagos_empty(store):
    assert stations_in_place(store, "Lagos", genre="jazz") == []


def test_genre_substring_matches(store):
    # "smooth jazz" is a genre of Antena 1 São Paulo.
    found = stations_by_genre(store, "jazz")
    names = {s.name for s in found}
    assert {"WWOZ 90.7 New Orleans", "Jazz London Radio", "FIP Jazz",
            "Alpha Boys School Radio", "Antena 1 São Paulo"} <= names


def test_search_combined(store):
    found = search_stations(store, place="Paris", genre="jazz")
    assert [s.name for s in found] == ["FIP Jazz"]
    found = search_stations(store, genre="funk")
    assert [s.name for s in found] == ["Amsterdam Funk Channel"]


# ── distance ────────────────────────────────────────────────────────


def test_near_lagos(store):
    found = stations_near(store, 6.57737, 3.32116, radius_km=50.0)
    assert [s.name for s in found] == ["METRO FM LAGOS", "Wazobia FM: Lagos"]


def test_near_tokyo(store):
    found = stations_near(store, 35.549678, 139.786958, radius_km=50.0)
    assert [s.name for s in found] == ["AFN GO TOKYO"]


def test_near_null_island_empty(store):
    assert stations_near(store, 0.0, 0.0, radius_km=100.0) == []


def test_near_prefers_station_geo(store):
    # Sompa carries its own geo (5.6424482, -0.2435186); a query at
    # that exact point finds Accra's stations, nearest first.
    found = stations_near(store, 5.6424482, -0.2435186, radius_km=50.0)
    assert found[0].name == "Sompa 106.5FM Accra"


# ── the globe, as data ──────────────────────────────────────────────


def test_globe_dots(store):
    dots = globe_dots(store)
    assert len(dots) == 20
    assert sum(d["count"] for d in dots) == 24
    by_city = {d["city"]: d for d in dots}
    assert by_city["Lagos"]["count"] == 2
    assert by_city["Accra"]["count"] == 2
    assert by_city["Paris"]["count"] == 2
    assert by_city["London"]["count"] == 2
    assert by_city["Tokyo"]["count"] == 1
    assert by_city["Lagos"]["lat"] == pytest.approx(6.57737)


# ── store behaviour ─────────────────────────────────────────────────


def test_store_roundtrip(home):
    path = home / "data" / "radio_stations.json"
    first = StationStore(path)
    added = Station(name="Test FM", city="Lagos", country="Nigeria",
                    lat=6.57737, lon=3.32116,
                    stream_url="http://example.com/test.mp3")
    assert first.add(added) is True
    second = StationStore(path)
    assert second.get(added.id) is not None
    assert second.count() == 25
    assert second.remove(added.id) is True
    assert StationStore(path).count() == 24


def test_store_dedupe_by_stream_url(store):
    duplicate = Station(name="Copy FM", city="Lagos", country="Nigeria",
                        lat=6.57737, lon=3.32116,
                        stream_url="https://go.webgateready.com/metrofm/radio.mp3")
    assert store.add(duplicate) is False
    assert store.count() == 24


def test_corrupt_store_reseeds(home):
    path = home / "data" / "radio_stations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    reopened = StationStore(path)
    assert reopened.count() == 24
    quarantined = list(path.parent.glob("radio_stations.json.corrupt-*"))
    assert len(quarantined) == 1


# ── health: a dead stream degrades, never hangs ─────────────────────


def test_check_stream_dead(store):
    station = station_by_name(store, "METRO FM LAGOS")
    assert station is not None

    def _boom(url):
        raise urllib.error.URLError("connection refused")

    status, detail = check_stream(station, read_fn=_boom)
    assert status == "down"
    assert "URLError" in detail


def test_check_stream_up(store):
    station = station_by_name(store, "METRO FM LAGOS")
    assert station is not None
    status, detail = check_stream(
        station, read_fn=lambda url: (200, "audio/mpeg", 16384, 0.4))
    assert status == "up"
    assert "200" in detail


def test_check_stream_http_error_is_down(store):
    station = station_by_name(store, "METRO FM LAGOS")
    assert station is not None
    status, detail = check_stream(
        station, read_fn=lambda url: (404, "text/html", 0, 0.1))
    assert status == "down"
    assert "404" in detail


def test_check_station_persists(store):
    station = station_by_name(store, "METRO FM LAGOS")
    assert station is not None

    def _boom(url):
        raise urllib.error.URLError("connection refused")

    result = check_station(store, station.id, read_fn=_boom)
    assert result is not None
    assert result[0] == "down"
    assert store.get(station.id).health == "down"
    assert store.get(station.id).checked_at > 0


def test_check_all_counts(store):
    result = check_all(
        store, read_fn=lambda url: (200, "audio/mpeg", 16384, 0.1))
    assert result == {"up": 24, "down": 0, "failures": []}


# ── play: bounded live samples ──────────────────────────────────────


def test_capture_sample_byte_cap(store, home):
    station = station_by_name(store, "METRO FM LAGOS")
    assert station is not None
    chunks = [_MP3_CHUNK * 64] * 10  # ~640 KiB of fake audio

    def _read(url):
        return _FakeStream(chunks)

    sample = capture_sample(station, seconds=45, max_bytes=1000,
                            dest_dir=home / "samples", read_fn=_read)
    assert sample.ok is True
    assert sample.n_bytes == 1000
    assert Path(sample.path).stat().st_size == 1000
    assert sample.path.endswith(".mp3")


def test_capture_sample_time_cap(store, home):
    station = station_by_name(store, "METRO FM LAGOS")
    assert station is not None
    stream = _FakeStream([_MP3_CHUNK] * 50, delay=0.3)

    def _read(url):
        return stream

    sample = capture_sample(station, seconds=1.0,
                            dest_dir=home / "samples", read_fn=_read)
    assert sample.ok is True
    # A 1 s budget with 0.3 s per read stops after a few chunks —
    # the stream never runs to its end.
    assert stream.reads <= 5
    assert sample.n_bytes > 0


def test_capture_sample_empty_stream(store, home):
    station = station_by_name(store, "METRO FM LAGOS")
    assert station is not None
    sample = capture_sample(station, dest_dir=home / "samples",
                            read_fn=lambda url: _FakeStream([]))
    assert sample.ok is False
    assert "no bytes" in sample.error
    assert sample.path == ""


def test_capture_sample_dead_url(store, home):
    station = station_by_name(store, "METRO FM LAGOS")
    assert station is not None

    def _boom(url):
        raise urllib.error.URLError("connection refused")

    sample = capture_sample(station, dest_dir=home / "samples",
                            read_fn=_boom)
    assert sample.ok is False
    assert "URLError" in sample.error


def test_capture_sample_aac_extension(store, home):
    station = station_by_name(store, "FIP Jazz")
    assert station is not None
    aac = b"\xff\xf1\x50\x00" + b"\x00" * 100
    sample = capture_sample(
        station, dest_dir=home / "samples",
        read_fn=lambda url: _FakeStream([aac], content_type="audio/aac"))
    assert sample.ok is True
    assert sample.path.endswith(".aac")


# ── radio-browser directory ─────────────────────────────────────────


_YFM_ROW = {
    "name": "YFm Accra",
    "country": "Ghana",
    "countrycode": "GH",
    "state": "Accra",
    "geo_lat": None,
    "geo_long": None,
    "url_resolved": "https://atunwadigital.streamguys1.com/yfm1079accra",
    "url": "https://atunwadigital.streamguys1.com/yfm1079accra",
    "tags": "entertainment,music,news,variety",
    "votes": 28,
    "stationuuid": "00000000-0000-0000-0000-000000000001",
    "language": "English",
}


def test_row_mapping():
    from nomorals.music_sources.radio import RadioBrowserClient

    client = RadioBrowserClient(fetch_fn=lambda url: "[]")
    station = client.row_to_station(_YFM_ROW)
    assert station is not None
    assert station.name == "YFm Accra"
    assert station.city == "Accra"
    assert station.country == "Ghana"
    assert station.genres == ["entertainment", "music", "news", "variety"]
    assert station.votes == 28
    assert station.source == "radio-browser"
    # No coordinates in the row: the station has no anchor of its
    # own and must never be plotted at (0, 0).
    assert station.has_coords is False


def test_row_mapping_drops_unusable_rows():
    from nomorals.music_sources.radio import RadioBrowserClient

    client = RadioBrowserClient(fetch_fn=lambda url: "[]")
    assert client.row_to_station({**_YFM_ROW, "url": "",
                                  "url_resolved": ""}) is None
    assert client.row_to_station({**_YFM_ROW, "name": "  "}) is None
    # A row with no state AND no country has no place at all — it
    # is dropped, never built into a station with an empty city.
    assert client.row_to_station({**_YFM_ROW, "state": "",
                                  "country": ""}) is None


def test_row_mapping_country_fallback_city():
    from nomorals.music_sources.radio import RadioBrowserClient

    client = RadioBrowserClient(fetch_fn=lambda url: "[]")
    station = client.row_to_station({**_YFM_ROW, "state": ""})
    assert station is not None
    # No state in the row: the city is the finest place the
    # directory gives — the country — and nothing is invented.
    assert station.city == "Ghana"
    assert station.region == ""
    assert station.country == "Ghana"


def test_grow_adds_new_and_dedupes(store):
    from nomorals.music_sources.radio import RadioBrowserClient

    payload = json.dumps([
        {**_YFM_ROW, "name": "Brand New FM",
         "url_resolved": "http://example.com/brandnew.mp3",
         "url": "http://example.com/brandnew.mp3"},
        {**_YFM_ROW, "name": "YFm Accra copy",
         "url_resolved": "https://atunwadigital.streamguys1.com/yfm1079accra"},
    ])
    client = RadioBrowserClient(fetch_fn=lambda url: payload)
    result = grow_from_directory(store, client=client, limit=2)
    assert result == {"added": 1, "seen": 2, "error": ""}
    assert store.count() == 25


def test_grow_directory_down(store):
    from nomorals.music_sources.radio import RadioBrowserClient

    def _boom(url):
        raise urllib.error.URLError("directory unreachable")

    client = RadioBrowserClient(fetch_fn=_boom)
    result = grow_from_directory(store, client=client)
    assert result["added"] == 0
    assert "URLError" in result["error"]
    assert store.count() == 24


# ── natural-language parsing ────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ("play a radio station from Tokyo",
     {"intent": "play", "place": "Tokyo", "genre": "", "station_id": ""}),
    ("jazz stations in New Orleans",
     {"intent": "stations", "place": "New Orleans", "genre": "jazz",
      "station_id": ""}),
    ("what's on in Accra?",
     {"intent": "stations", "place": "Accra", "genre": "",
      "station_id": ""}),
    ("radio in Lagos",
     {"intent": "stations", "place": "Lagos", "genre": "",
      "station_id": ""}),
    ("play jazz radio from New Orleans",
     {"intent": "play", "place": "New Orleans", "genre": "jazz",
      "station_id": ""}),
])
def test_parse_shapes(store, text, expected):
    assert parse_radio_request(text, store.all()) == expected


def test_parse_play_genre_radio(store):
    parsed = parse_radio_request("play jazz radio", store.all())
    assert parsed is not None
    assert parsed["intent"] == "play"
    assert parsed["genre"] == "jazz"
    assert parsed["station_id"] != ""


def test_parse_play_station_by_name(store):
    parsed = parse_radio_request("play WWOZ 90.7 New Orleans", store.all())
    assert parsed is not None
    assert parsed["intent"] == "play"
    assert parsed["station_id"] != ""
    assert store.get(parsed["station_id"]).name == "WWOZ 90.7 New Orleans"


def test_parse_rejects_non_radio(store):
    assert parse_radio_request("play despacito", store.all()) is None
    assert parse_radio_request("play the radio", store.all()) is None
    assert parse_radio_request("what time is it", store.all()) is None
    assert parse_radio_request("", store.all()) is None


# ── the brain tool ──────────────────────────────────────────────────


def test_tool_stations(home):
    tool = _tool(home)
    result = tool(action="stations", place="Lagos")
    assert result["ok"] is True
    assert result["count"] == 2
    assert "METRO FM LAGOS" in result["reply"]


def test_tool_stations_unknown_place(home):
    tool = _tool(home)
    result = tool(action="stations", place="Nowhereville")
    assert result["ok"] is True
    assert result["count"] == 0
    assert "don't have" in result["reply"]


def test_tool_play_fake_sample(home, monkeypatch):
    import nomorals.tools.radio as radio_mod

    tool = _tool(home)

    def _fake_sample(station, **kwargs):
        path = Path(kwargs["dest_dir"]) / f"{station.id}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xfb" + b"\x00" * 100)
        from nomorals.music_sources.radio import Sample

        return Sample(ok=True, path=str(path), n_bytes=102,
                      seconds=1.0, content_type="audio/mpeg",
                      station_id=station.id)

    monkeypatch.setattr(radio_mod, "capture_sample", _fake_sample)
    result = tool(action="play", id="WWOZ 90.7 New Orleans")
    assert result["ok"] is True
    assert result["station"]["name"] == "WWOZ 90.7 New Orleans"
    assert Path(result["sample"]["path"]).is_file()
    assert result["stream_url"].startswith("http")


def test_tool_play_falls_through_dead_station(home, monkeypatch):
    import nomorals.tools.radio as radio_mod

    tool = _tool(home)

    def _fake_sample(station, **kwargs):
        from nomorals.music_sources.radio import Sample

        if station.name.startswith("METRO"):
            return Sample(ok=False, station_id=station.id,
                          error="URLError: connection refused")
        path = Path(kwargs["dest_dir"]) / f"{station.id}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xfb" + b"\x00" * 100)
        return Sample(ok=True, path=str(path), n_bytes=102,
                      seconds=1.0, content_type="audio/mpeg",
                      station_id=station.id)

    monkeypatch.setattr(radio_mod, "capture_sample", _fake_sample)
    result = tool(action="play", place="Lagos")
    assert result["ok"] is True
    assert result["station"]["name"] == "Wazobia FM: Lagos"


def test_tool_play_all_down(home, monkeypatch):
    import nomorals.tools.radio as radio_mod

    tool = _tool(home)

    def _dead_sample(station, **kwargs):
        from nomorals.music_sources.radio import Sample

        return Sample(ok=False, station_id=station.id,
                      error="URLError: connection refused")

    monkeypatch.setattr(radio_mod, "capture_sample", _dead_sample)
    result = tool(action="play", place="Lagos")
    assert result["ok"] is False
    assert "down" in result["error"]


def test_tool_status_and_globe(home):
    tool = _tool(home)
    status = tool(action="status")
    assert status["ok"] is True
    assert status["count"] == 24
    globe = tool(action="globe")
    assert globe["ok"] is True
    assert len(globe["dots"]) == 20


def test_tool_unknown_station(home):
    tool = _tool(home)
    assert tool(action="play", id="no such station")["ok"] is False
    assert tool(action="health", id="no such station")["ok"] is False
    assert tool(action="bogus")["ok"] is False


def test_tool_registered_by_builtins():
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register_builtins()
    assert "radio" in reg._tools


# ── the /music chat surface ─────────────────────────────────────────


def test_chat_radio_lagos(home):
    from nomorals.music_sources.chat import control_music_sources

    reply = control_music_sources("radio Lagos", context=_ctx(home))
    assert "METRO FM LAGOS" in reply
    assert "Wazobia FM: Lagos" in reply


def test_chat_radio_genre_place(home):
    from nomorals.music_sources.chat import control_music_sources

    reply = control_music_sources("radio jazz New Orleans",
                                  context=_ctx(home))
    assert "WWOZ 90.7 New Orleans" in reply


def test_chat_radio_unknown_place(home):
    from nomorals.music_sources.chat import control_music_sources

    reply = control_music_sources("radio Nowhereville",
                                  context=_ctx(home))
    assert "don't have" in reply


def test_chat_radio_play_fake(home, monkeypatch):
    import nomorals.music_sources.radio as radio_mod
    from nomorals.music_sources.chat import control_music_sources

    def _fake_sample(station, **kwargs):
        path = Path(kwargs["dest_dir"]) / f"{station.id}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xfb" + b"\x00" * 100)
        from nomorals.music_sources.radio import Sample

        return Sample(ok=True, path=str(path), n_bytes=102,
                      seconds=1.0, content_type="audio/mpeg",
                      station_id=station.id)

    monkeypatch.setattr(radio_mod, "capture_sample", _fake_sample)
    reply = control_music_sources("radio play WWOZ 90.7 New Orleans",
                                  context=_ctx(home))
    assert "WWOZ 90.7 New Orleans" in reply
    assert "saved on the server" in reply


def test_chat_radio_globe_and_status(home):
    from nomorals.music_sources.chat import control_music_sources

    globe = control_music_sources("radio globe", context=_ctx(home))
    assert "20 cities" in globe
    assert "Lagos" in globe
    status = control_music_sources("radio status", context=_ctx(home))
    assert "24 stations" in status


# ── registry wiring ─────────────────────────────────────────────────


def test_registry_row():
    from nomorals.music_sources.registry import by_id

    row = by_id("radio-browser")
    assert row is not None
    assert row.source_class == "open"
    assert row.health_check_hours == 6
    assert "radio-browser" in row.licence.lower() or "directory" in row.licence


def test_usage_lists_radio():
    from nomorals.music_sources.chat import MUSIC_SOURCES_USAGE
    from nomorals.music_sources.registry import MUSIC_REGISTRY

    usage = MUSIC_SOURCES_USAGE()
    assert "/music radio" in usage
    assert f"all {len(MUSIC_REGISTRY)} rows" in usage
