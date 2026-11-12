"""Openverse as a searchable real-photography pool (census C2).

picsum.photos is the pools' only real photography — and it is
seeded-random, not searchable.  Openverse (api.openverse.org)
is the verified keyless fit: its images endpoint answers a
query with CC-licensed photos from across the web, each row
carrying its direct image URL, creator, licence and a mature
flag.  Verified live 2026-11-12 from this box, anonymous
(no key): q=couple and q=portrait each returned 240 results,
a nonsense query returned 0 results honestly, and the live
response headers carry the anonymous limits (20/min burst,
200/day sustained).  The rows below are the real rows from
that live fetch (trimmed to the fields the extractor reads),
except the mature row, which is a real row's shape with the
flag flipped — the point of the test is the flag, not the
photo.  Network-free: every transport is a fixture.
"""

from __future__ import annotations

from typing import Any

import pytest

from nomorals.anime.sources import (
    IMAGE_POOLS,
    ImagePoolClient,
    SourceFailure,
    _openverse,
)
from nomorals.anime.transport import HttpResult, TransportError

# ── the live shape, trimmed ───────────────────────────────────

# Real rows from the 2026-11-12 live fetch of
# api.openverse.org/v1/images/?q=couple&page_size=3.
COUPLE_ROWS = [
    {"title": "Couple",
     "url": "https://live.staticflickr.com/8564/"
            "15285612973_2cf2573d8f_b.jpg",
     "creator": "kevin dooley", "license": "by",
     "foreign_landing_url": "https://www.flickr.com/photos/"
                           "12836528@N00/15285612973",
     "mature": False},
    {"title": "couple",
     "url": "https://live.staticflickr.com/3387/"
            "3334998087_824bac2101_b.jpg",
     "creator": "S13ven", "license": "by-nc-sa",
     "foreign_landing_url": "https://www.flickr.com/photos/"
                           "11868886@N04/3334998087",
     "mature": False},
    {"title": "Couple",
     "url": "https://live.staticflickr.com/3882/"
            "14870214192_47d774e1c3_b.jpg",
     "creator": "kevin dooley", "license": "by",
     "foreign_landing_url": "https://www.flickr.com/photos/"
                           "12836528@N00/14870214192",
     "mature": False},
]

# The same row shape with the mature flag flipped — the API
# serves mature rows the same way, and the pool's own
# re-check is what keeps them out.
MATURE_ROW = dict(COUPLE_ROWS[0], mature=True)


def _payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"result_count": len(rows), "page_count": 1,
            "page_size": 8, "page": 1, "results": rows}


class _Transport:
    """Routes by host substring; records every call's params.
    A route value is an HttpResult, an HTTP status int, or
    "down" (TransportError).  Hosts with no route answer 404."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def request(self, method: str, url: str, **kw: Any) -> HttpResult:
        self.calls.append((url, kw.get("params")))
        for host, resp in self.routes.items():
            if host in url:
                if resp == "down":
                    raise TransportError(f"{host} unreachable")
                if isinstance(resp, int):
                    return HttpResult(resp, url=url)
                if isinstance(resp, HttpResult) and not resp.url:
                    resp.url = url
                return resp
        return HttpResult(404, url=url)


def _json(data: Any) -> HttpResult:
    return HttpResult(200, data=data)


class _ScriptedRng:
    """choice takes the first eligible row (deterministic);
    random() replays a script for the weighted walk."""

    def __init__(self, script: list[float] | None = None) -> None:
        self._script = list(script or [])

    def random(self) -> float:
        return self._script.pop(0) if self._script else 0.5

    def randrange(self, *args: Any) -> int:
        return 424242

    def choice(self, seq: Any) -> Any:
        return seq[0]


def _one_source_pool(entries: list) -> dict[str, list]:
    return {"test": entries}


# ── the wiring ────────────────────────────────────────────────

def test_openverse_wired_only_into_charpic_and_couplepp():
    for kind, query in (("charpic", "portrait"),
                        ("couplepp", "couple")):
        entries = [e for e in IMAGE_POOLS[kind]
                   if e.shape == "openverse"]
        assert len(entries) == 1, (kind, entries)
        assert entries[0].name == "openverse"
        assert entries[0].query == query
        assert entries[0].source_class == "photography"
        assert entries[0].key == ""  # keyless, like picsum
        assert entries[0].weight == 2
    for kind in ("waifu", "neko", "kitsune", "husbando", "gif"):
        assert not [e for e in IMAGE_POOLS[kind]
                    if e.shape == "openverse"], kind


def test_charpic_and_couplepp_still_anime_first_by_weight():
    # The unified suite pins this for every kind: the anime
    # entries carry the top weight.  The new photography
    # rung (weight 2) must not outrank them (weight 3).
    for kind in ("charpic", "couplepp"):
        entries = IMAGE_POOLS[kind]
        anime = [e.weight for e in entries
                 if e.source_class == "anime"]
        other = [e.weight for e in entries
                 if e.source_class != "anime"]
        assert anime and max(anime) > max(other), kind


# ── the search answers ────────────────────────────────────────

def test_openverse_answers_a_search_with_real_photo_urls():
    transport = _Transport({
        "api.openverse.org": _json(_payload(COUPLE_ROWS)),
    })
    client = ImagePoolClient(
        transport,
        pools=_one_source_pool([_openverse("couple")]))
    hit = client.fetch_media("test", rng=_ScriptedRng())
    assert hit.source == "openverse"
    assert hit.source_class == "photography"
    assert hit.url == COUPLE_ROWS[0]["url"]
    assert hit.url.startswith("https://live.staticflickr.com/")


def test_openverse_query_reaches_the_api():
    transport = _Transport({
        "api.openverse.org": _json(_payload(COUPLE_ROWS)),
    })
    client = ImagePoolClient(
        transport,
        pools=_one_source_pool([_openverse("couple")]))
    client.fetch_media("test", rng=_ScriptedRng())
    assert len(transport.calls) == 1
    url, params = transport.calls[0]
    assert "api.openverse.org/v1/images/" in url
    assert params is not None
    assert params.get("q") == "couple"
    assert params.get("page_size") == 8
    # no key rides anywhere: the source is keyless by design
    assert "key" not in params and "api_key" not in params


def test_openverse_mature_rows_are_never_served():
    # A mature row on top, clean rows below: the walk serves
    # the first CLEAN row, never the mature one.
    transport = _Transport({
        "api.openverse.org": _json(
            _payload([MATURE_ROW] + COUPLE_ROWS[1:])),
    })
    client = ImagePoolClient(
        transport,
        pools=_one_source_pool([_openverse("couple")]))
    hit = client.fetch_media("test", rng=_ScriptedRng())
    assert hit.url == COUPLE_ROWS[1]["url"]


def test_openverse_only_mature_rows_is_no_answer():
    transport = _Transport({
        "api.openverse.org": _json(_payload([MATURE_ROW])),
    })
    client = ImagePoolClient(
        transport,
        pools=_one_source_pool([_openverse("couple")]))
    with pytest.raises(SourceFailure):
        client.fetch_media("test", rng=_ScriptedRng())


def test_openverse_rows_without_a_url_are_skipped():
    rows = [dict(COUPLE_ROWS[0], url=""),
            dict(COUPLE_ROWS[1], url="ftp://example.com/x.jpg"),
            COUPLE_ROWS[2]]
    transport = _Transport({
        "api.openverse.org": _json(_payload(rows)),
    })
    client = ImagePoolClient(
        transport,
        pools=_one_source_pool([_openverse("couple")]))
    hit = client.fetch_media("test", rng=_ScriptedRng())
    assert hit.url == COUPLE_ROWS[2]["url"]


def test_openverse_empty_results_fail_honestly():
    # The live nonsense-query answer: 0 results, results [].
    transport = _Transport({
        "api.openverse.org": _json(_payload([])),
    })
    client = ImagePoolClient(
        transport,
        pools=_one_source_pool([_openverse("couple")]))
    with pytest.raises(SourceFailure) as excinfo:
        client.fetch_media("test", rng=_ScriptedRng())
    assert "openverse" in str(excinfo.value)


def test_openverse_down_falls_through_without_fake_results():
    transport = _Transport({"api.openverse.org": "down"})
    client = ImagePoolClient(
        transport,
        pools=_one_source_pool([_openverse("couple")]))
    with pytest.raises(SourceFailure) as excinfo:
        client.fetch_media("test", rng=_ScriptedRng())
    assert "openverse" in str(excinfo.value)


def test_openverse_http_error_is_no_answer():
    transport = _Transport({"api.openverse.org": 500})
    client = ImagePoolClient(
        transport,
        pools=_one_source_pool([_openverse("couple")]))
    with pytest.raises(SourceFailure):
        client.fetch_media("test", rng=_ScriptedRng())


def test_openverse_falls_through_to_the_next_rung():
    # Openverse down, picsum up: the walk does not die on
    # the dead rung — the next source serves.
    from nomorals.anime.sources import _picsum

    jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 64
    transport = _Transport({
        "api.openverse.org": "down",
        "picsum.photos": HttpResult(
            200, content=jpeg, url="https://fastly.picsum/d.jpg"),
    })
    client = ImagePoolClient(
        transport,
        pools=_one_source_pool([_openverse("couple"), _picsum()]))
    hit = client.fetch_media("test", rng=_ScriptedRng([0.9, 0.1]))
    assert hit.source == "picsum"
    assert hit.source_class == "photography"
