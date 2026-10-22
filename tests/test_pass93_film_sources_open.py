"""Pass #93 — film sources go open (the owner's open-sources law).

Passes #86–#92 took downloads, music, books, web search, prices
and images open.  The remaining named CLOSED-SOURCE classes were
video and WisdomKeeper archives.  Survey truth this suite pins:

* The video class's real organ is the film search chain in
  ``media/film_sources.py`` (the ``film_search`` tool's roster —
  Internet Archive films, NetNaija movies, Nkiri, FzMovies): a
  FIXED-order chain whose ``search_films`` flattened every
  outcome to ``[]`` (transport failure ≡ bot-wall ≡ catalogue
  miss), with NO outcome recording, NO learned ordering, NO
  candidate intake, NO health coverage — and two of its four
  hosts (nkiri.com, fzmovies.net) not registered anywhere.
* ``media/video.py`` VideoFinder is NOT a closed class: it rides
  the web-search engines (open since #90) and pattern-ranks
  their results; it has no source list of its own.  The download
  leg of a found film already records outcomes on the download
  path (#86).  WisdomKeeper archives (``wisdom/ingestor.py``'s
  fixed ``SEARCH_SOURCES``) are still closed — the banked next
  class; this pass does not touch them.
* ARCHITECTURE DECISION, with evidence: the film class joins
  the SAME media registry as its own class — no mirrored
  machinery.  Every chain source is one host serving film
  search, so the host-keyed model fits without distortion; two
  hosts are ALREADY descriptors (``archive.org``, the #88
  music-search seed; ``thenetnaija.net``, the #76 download-blog
  seed) and both classes fetch the literal same hosts, so the
  film class shares those descriptors and extends their kinds
  (the #88 youtube.com / #92 Bing-Yandex precedent).  The new
  seeds declare NO extraction facts, so the #76/#87 extractor
  view is byte-identical (pinned below).
* Outcome taxonomy (the one vocabulary): results landed =
  served; a 200 page with no hits = weak evidence
  (``record_no_result``, #88 — an obscure film must never stand
  a source down); a bot-wall (HTTP 401/403, or a 200 page
  carrying challenge markers) is ``record_blocked`` (#89 —
  health ``bot-blocked``, never ``dead``); an HTTP 429 is
  ``record_rate_limited`` (#90); a transport failure / dead
  search route is a failure with an error class.  The taxonomy
  is readable because each source now keeps its last-fetch
  evidence (status / body) — before this pass ``_get``
  flattened an HTTP 403 and a timeout into the same "".
* Health loop: the ONE existing loop / job covers the class;
  film seeds carry their REAL search URL template and are
  probed through it with one bounded probe query ("Avatar" —
  the agency registry's film probe), because a root page
  proves nothing about search (#90 reasoning).
* Discovery is genuine on this path: film search results are
  post-page URLs on foreign hosts (mirrors, sister blogs), and
  those intake as untrusted candidates with ``kinds=("film",)``;
  health-promoted grown film sources serve through the generic
  WordPress film shape (``DiscoveredFilmSource``), exactly as
  grown book sources serve through the universal reader (#89).

All fetches are stubbed — no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media import film_sources
from nomorals.media.film_sources import (
    DiscoveredFilmSource, FilmCandidate, FilmSource)


# ── helpers ───────────────────────────────────────────────────────────


def _kv():
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return KVStore(db), db


def _seeded(store=None):
    from nomorals.media.source_registry import _seed

    return _seed(store=store)


def _ok_fetcher(url):
    return 200, b"<html><body>a real page with real content</body></html>"


class _FakeFilmSource(FilmSource):
    """A chain source with scripted results + fetch evidence."""

    def __init__(self, name, results=(), status=200, body="",
                 exc=None, order_log=None):
        self.name = name
        self._results = list(results)
        self._status = status
        self._body = body
        self._exc = exc
        self._order_log = order_log
        self.attempts = 0

    def search(self, query, limit=8):
        self.attempts += 1
        if self._order_log is not None:
            self._order_log.append(self.name)
        if self._exc is not None:
            raise self._exc
        self._last_status = self._status
        self._last_body = self._body
        self._last_error = ""
        return list(self._results)[:limit]


def _chain(monkeypatch, sources):
    monkeypatch.setattr(film_sources, "FILM_CHAIN", list(sources))
    return sources


def _cand(source, title="A Real Movie (2024)", url=""):
    return FilmCandidate(source=source, title=title,
                         url=url or f"https://{source}.example/post")


# ── (1) the film class joins the SAME registry ────────────────────────


def test_every_chain_source_is_a_registered_seed():
    reg = _seeded()
    origins = {s["domain"]: s["origin"] for s in reg.status()["sources"]}
    for name, domain in film_sources.FILM_SOURCE_DOMAINS.items():
        desc = reg.for_domain(domain)
        assert desc is not None, f"{name} ({domain}) not registered"
        assert origins[domain] == "seed"
        assert "film" in desc.kinds and "film-search" in desc.kinds
        assert desc.search_url_template, f"{domain} has no probe template"


def test_shared_hosts_carry_both_class_memberships():
    reg = _seeded()
    archive = reg.for_domain("archive.org")
    assert "music-search" in archive.kinds  # the #88 class, untouched
    assert "film-search" in archive.kinds   # extended, one descriptor
    netnaija = reg.for_domain("thenetnaija.net")
    assert "music" in netnaija.kinds and "film-search" in netnaija.kinds


def test_extractor_view_unchanged_by_film_seeds():
    from nomorals.media.download_kit import descriptors_for_registry

    reg = _seeded()
    for domain in ("nkiri.com", "fzmovies.net"):
        desc = reg.for_domain(domain)
        assert desc.extraction_rule == "" and desc.url_patterns == ()
    descs = descriptors_for_registry(reg)
    # the pristine tree derives exactly 13 extractor descriptors;
    # the film seeds add none and change none
    assert len(descs) == 13


def test_status_carries_the_film_class():
    reg = _seeded()
    film = reg.status()["film"]
    rows = {r["source"]: r for r in film["sources"]}
    assert set(rows) == set(film_sources.FILM_SEARCH_SOURCES)
    for name, row in rows.items():
        assert row["domain"] == film_sources.FILM_SOURCE_DOMAINS[name]
        assert row["origin"] == "seed"
        assert row["ok"] == row["fail"] == 0
    assert film["grown"] == [] and film["candidates_pending"] == 0


def test_render_status_text_shows_film_rows():
    from nomorals.media.source_registry import render_status_text

    text = render_status_text(_seeded().status())
    assert "🎬 film sources" in text
    assert "🎬 nkiri → nkiri.com" in text
    assert "🎬 fzmovies → fzmovies.net" in text


# ── (2) outcome taxonomy at the search seam ───────────────────────────


def test_served_search_credits_the_source(monkeypatch):
    reg = _seeded()
    hit = _cand("nkiri", url="https://nkiri.com/movie-one")
    _chain(monkeypatch, [_FakeFilmSource("nkiri", results=[hit])])
    out = film_sources.search_films("movie", registry=reg)
    assert [c.title for c in out] == ["A Real Movie (2024)"]
    st = reg.stats_for(reg.for_domain("nkiri.com"))
    assert st["ok"] == 1 and st["fail"] == 0


def test_no_result_is_weak_evidence_not_failure(monkeypatch):
    reg = _seeded()
    _chain(monkeypatch, [_FakeFilmSource(
        "nkiri", results=[], status=200, body="<html>no hits</html>")])
    assert film_sources.search_films("obscure", registry=reg) == []
    st = reg.stats_for(reg.for_domain("nkiri.com"))
    assert st["no_result"] == 1 and st["fail"] == 0 and st["ok"] == 0
    # weak evidence never stands a source down
    assert reg.is_active(reg.for_domain("nkiri.com"))


def test_botwall_403_records_blocked_not_dead(monkeypatch):
    reg = _seeded()
    _chain(monkeypatch, [_FakeFilmSource("nkiri", status=403)])
    film_sources.search_films("movie", registry=reg)
    desc = reg.for_domain("nkiri.com")
    st = reg.stats_for(desc)
    assert st["blocked"] == 1 and st["last_error_class"] == "bot_block"
    assert reg.health_for(desc) == "bot-blocked"


def test_botwall_200_challenge_page_records_blocked(monkeypatch):
    reg = _seeded()
    _chain(monkeypatch, [_FakeFilmSource(
        "fzmovies", status=200,
        body="<html>Just a moment... verify you are human</html>")])
    film_sources.search_films("movie", registry=reg)
    assert reg.stats_for(reg.for_domain("fzmovies.net"))["blocked"] == 1


def test_rate_limited_429_records_its_own_class(monkeypatch):
    reg = _seeded()
    _chain(monkeypatch, [_FakeFilmSource("nkiri", status=429)])
    film_sources.search_films("movie", registry=reg)
    desc = reg.for_domain("nkiri.com")
    st = reg.stats_for(desc)
    assert st["rate_limited"] == 1
    assert reg.health_for(desc) == "rate-limited"


def test_transport_failure_records_network_failure(monkeypatch):
    reg = _seeded()
    _chain(monkeypatch, [_FakeFilmSource("nkiri", status=0)])
    film_sources.search_films("movie", registry=reg)
    st = reg.stats_for(reg.for_domain("nkiri.com"))
    assert st["fail"] == 1 and st["last_error_class"] == "network_failure"


def test_dead_search_route_records_http_failure(monkeypatch):
    reg = _seeded()
    _chain(monkeypatch, [_FakeFilmSource("fzmovies", status=404)])
    film_sources.search_films("movie", registry=reg)
    st = reg.stats_for(reg.for_domain("fzmovies.net"))
    assert st["fail"] == 1 and st["last_error_class"] == "http_404"


def test_raising_search_records_search_failed(monkeypatch):
    reg = _seeded()
    _chain(monkeypatch, [_FakeFilmSource(
        "nkiri", exc=RuntimeError("boom"))])
    assert film_sources.search_films("movie", registry=reg) == []
    st = reg.stats_for(reg.for_domain("nkiri.com"))
    assert st["fail"] == 1 and st["last_error_class"] == "search_failed"


def test_search_results_intake_foreign_hosts_as_film_candidates(
        monkeypatch):
    reg = _seeded()
    hit = _cand("nkiri", url="https://sister-blog.example/movie-one")
    _chain(monkeypatch, [_FakeFilmSource("nkiri", results=[hit])])
    film_sources.search_films("movie", registry=reg)
    cands = {d.domain: d for d in reg.candidates()}
    assert "sister-blog.example" in cands
    assert cands["sister-blog.example"].kinds == ("film",)
    # a candidate serves nothing: it is not in the source pool
    assert reg.for_domain("sister-blog.example") is None


# ── (3) ordering: day-one pin + learned order ─────────────────────────


def test_day_one_chain_order_is_untouched(monkeypatch):
    reg = _seeded()
    order = []
    _chain(monkeypatch, [
        _FakeFilmSource("archiveorg", order_log=order),
        _FakeFilmSource("netnaija-movies", order_log=order),
        _FakeFilmSource("nkiri", order_log=order),
        _FakeFilmSource("fzmovies", order_log=order),
    ])
    film_sources.search_films("movie", registry=reg)
    assert order == ["archiveorg", "netnaija-movies", "nkiri", "fzmovies"]


def test_early_break_leaves_later_sources_unevidenced(monkeypatch):
    reg = _seeded()
    hits = [_cand("archiveorg", title=f"Film {i}") for i in range(3)]
    later = _FakeFilmSource("nkiri")
    _chain(monkeypatch, [
        _FakeFilmSource("archiveorg", results=hits), later])
    out = film_sources.search_films("movie", limit=3, registry=reg)
    assert len(out) == 3 and later.attempts == 0
    st = reg.stats_for(reg.for_domain("nkiri.com"))
    assert st["ok"] == st["fail"] == st["no_result"] == 0


def test_learned_order_demotes_stood_down_source(monkeypatch):
    reg = _seeded()
    nkiri = reg.for_domain("nkiri.com")
    for _ in range(3):  # STAND_DOWN_AFTER consecutive failures
        reg.record_outcome(nkiri, False, error_class="network_failure")
    assert not reg.is_active(nkiri)
    order = []
    _chain(monkeypatch, [
        _FakeFilmSource("archiveorg", order_log=order),
        _FakeFilmSource("netnaija-movies", order_log=order),
        _FakeFilmSource("nkiri", order_log=order),
        _FakeFilmSource("fzmovies", order_log=order),
    ])
    film_sources.search_films("movie", registry=reg)
    # stood-down sources are still tried — last, never silently dropped
    assert order[-1] == "nkiri"
    assert order[0] == "archiveorg"


def test_no_registry_keeps_pre_pass_behaviour(monkeypatch):
    order = []
    hit = _cand("nkiri")
    _chain(monkeypatch, [
        _FakeFilmSource("archiveorg", order_log=order),
        _FakeFilmSource("nkiri", results=[hit], order_log=order),
    ])
    out = film_sources.search_films("movie")  # no context, no registry
    assert [c.source for c in out] == ["nkiri"]
    assert order == ["archiveorg", "nkiri"]


def test_registry_resolution_discipline():
    # db-less contexts get NO registry (pre-#93 behaviour exactly)
    assert film_sources.film_registry_for_context(None) is None
    assert film_sources.film_registry_for_context(
        SimpleNamespace(db=None)) is None
    reg = _seeded()
    assert film_sources.film_registry_for_context(
        SimpleNamespace(db=object()), injected=reg) is reg
    _store, db = _kv()
    bound = film_sources.film_registry_for_context(
        SimpleNamespace(db=db))
    from nomorals.media.source_registry import DEFAULT_REGISTRY
    assert bound is DEFAULT_REGISTRY


# ── (4) growth: candidates promote, grown sources serve ───────────────


def test_candidate_promotes_and_grown_source_serves(monkeypatch):
    reg = _seeded()
    assert reg.note_candidate("https://newfilmblog.example",
                              via="test", kinds=("film",))
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["promoted"] >= 1
    desc = reg.for_domain("newfilmblog.example")
    assert desc is not None and reg.origin_for(desc) == "grown"
    assert reg.film_status()["grown"][0]["domain"] == \
        "newfilmblog.example"

    html = ('<h2><a href="https://newfilmblog.example/movie-one">'
            "Grown Movie (2025)</a></h2>")
    with mock.patch.object(DiscoveredFilmSource, "_get",
                           lambda self, url, timeout=10.0: html):
        _chain(monkeypatch, [_FakeFilmSource(
            "archiveorg", status=200, body="<html></html>")])
        out = film_sources.search_films("grown", registry=reg)
    grown_hits = [c for c in out
                  if c.source == "grown:newfilmblog.example"]
    assert len(grown_hits) == 1
    assert grown_hits[0].title == "Grown Movie (2025)"
    # the grown source earned its serve
    assert reg.stats_for(desc)["ok"] == 1


def test_unpromoted_candidate_serves_nothing(monkeypatch):
    reg = _seeded()
    reg.note_candidate("https://newfilmblog.example",
                       via="test", kinds=("film",))
    _chain(monkeypatch, [_FakeFilmSource(
        "archiveorg", status=200, body="<html></html>")])
    out = film_sources.search_films("grown", registry=reg)
    assert all(not c.source.startswith("grown:") for c in out)


# ── (5) the health loop probes film sources honestly ──────────────────


def test_health_loop_probes_film_search_endpoints():
    reg = _seeded()
    seen = []

    def fetcher(url):
        seen.append(url)
        return 200, b"<html><body>real content</body></html>"

    reg.verify_health(fetcher=fetcher)
    assert any("advancedsearch" in u and "Avatar" in u for u in seen)
    assert any("thenetnaija.net/?s=Avatar" in u for u in seen)
    assert any("nkiri.com/?s=Avatar" in u for u in seen)
    assert any("fzmovies.net/search.php?searchname=Avatar" in u
               for u in seen)


def test_health_loop_film_429_is_rate_limited_not_dead():
    reg = _seeded()

    def fetcher(url):
        if "fzmovies" in url:
            return 429, b""
        return 200, b"<html><body>real content</body></html>"

    reg.verify_health(fetcher=fetcher)
    assert reg.health_for(reg.for_domain("fzmovies.net")) == \
        "rate-limited"


def test_health_loop_film_403_is_bot_blocked():
    reg = _seeded()

    def fetcher(url):
        if "nkiri" in url:
            return 403, b""
        return 200, b"<html><body>real content</body></html>"

    reg.verify_health(fetcher=fetcher)
    assert reg.health_for(reg.for_domain("nkiri.com")) == "bot-blocked"


# ── (6) resolve seam records evidence too ─────────────────────────────


class _ResolveSource(FilmSource):
    """A chain source whose download_url fetches (or doesn't)."""

    def __init__(self, name, status=200, fetch=True):
        self.name = name
        self._status = status
        self._fetch = fetch

    def search(self, query, limit=8):
        return []

    def download_url(self, candidate):
        if candidate.direct_url:
            return candidate.direct_url
        if self._fetch:
            self._fetch_count = getattr(self, "_fetch_count", 0) + 1
            self._last_status = self._status
            self._last_body = "<html>post page</html>"
            return "https://files.example/movie.mp4"
        return candidate.url


def test_resolve_credits_a_serving_source(monkeypatch):
    reg = _seeded()
    _chain(monkeypatch, [_ResolveSource("nkiri", status=200)])
    cand = FilmCandidate(source="nkiri", url="https://nkiri.com/post")
    url = film_sources.resolve_film_url(cand, registry=reg)
    assert url == "https://files.example/movie.mp4"
    assert reg.stats_for(reg.for_domain("nkiri.com"))["ok"] == 1


def test_resolve_records_a_wall(monkeypatch):
    reg = _seeded()
    _chain(monkeypatch, [_ResolveSource("nkiri", status=403)])
    cand = FilmCandidate(source="nkiri", url="https://nkiri.com/post")
    film_sources.resolve_film_url(cand, registry=reg)
    assert reg.stats_for(reg.for_domain("nkiri.com"))["blocked"] == 1


def test_resolve_without_fetch_records_nothing(monkeypatch):
    reg = _seeded()
    _chain(monkeypatch, [_ResolveSource("nkiri")])
    cand = FilmCandidate(source="nkiri", url="https://nkiri.com/post",
                         direct_url="https://cdn.example/m.mp4")
    url = film_sources.resolve_film_url(cand, registry=reg)
    assert url == "https://cdn.example/m.mp4"
    st = reg.stats_for(reg.for_domain("nkiri.com"))
    assert st["ok"] == st["fail"] == st["blocked"] == 0


# ── (7) persistence: learned film state survives ──────────────────────


def test_film_outcomes_persist_through_the_store():
    store, _db = _kv()
    reg = _seeded(store=store)
    reg.record_outcome(reg.for_domain("nkiri.com"), True)
    reg.record_blocked(reg.for_domain("fzmovies.net"))
    reg2 = _seeded(store=store)
    assert reg2.stats_for(reg2.for_domain("nkiri.com"))["ok"] == 1
    assert reg2.stats_for(reg2.for_domain("fzmovies.net"))["blocked"] == 1
    assert reg2.film_status()["sources"][2]["ok"] == 1


# ── (8) the REAL sources' fetch evidence (urlopen stubbed) ────────────


class _FakeResp:
    def __init__(self, body, status=200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_real_source_captures_http_error_as_evidence(monkeypatch):
    import urllib.error
    import urllib.request

    from nomorals.media.film_sources import NkiriSource

    def _raise(req, timeout=0):
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    src = NkiriSource()
    assert src.search("movie") == []
    assert src._last_status == 403

    reg = _seeded()
    _chain(monkeypatch, [src])
    film_sources.search_films("movie", registry=reg)
    assert reg.stats_for(reg.for_domain("nkiri.com"))["blocked"] == 1


def test_real_source_parses_and_records_served(monkeypatch):
    import urllib.request

    from nomorals.media.film_sources import NkiriSource

    html = (b'<a href="https://nkiri.com/movie-one">'
            b"Some Great Movie (2024)</a>")

    def _resp(req, timeout=0):
        return _FakeResp(html)

    monkeypatch.setattr(urllib.request, "urlopen", _resp)
    reg = _seeded()
    _chain(monkeypatch, [NkiriSource()])
    out = film_sources.search_films("movie", registry=reg)
    assert [c.title for c in out] == ["Some Great Movie (2024)"]
    assert reg.stats_for(reg.for_domain("nkiri.com"))["ok"] == 1


# ── (9) the film_sources tool surface ─────────────────────────────────


class _ToolRegistry:
    def __init__(self, context):
        self.context = context
        self.tools = {}

    def register(self, name, description="", capability=None):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


def test_film_sources_tool_status_and_health(tmp_path):
    from nomorals.tools import sceneintel

    _store, db = _kv()
    context = SimpleNamespace(db=db, workspace=str(tmp_path))
    tools = _ToolRegistry(context)
    sceneintel.register(tools)
    assert "film_search" in tools.tools and "film_sources" in tools.tools
    status = tools.tools["film_sources"](action="status")
    assert {r["source"] for r in status["sources"]} == \
        set(film_sources.FILM_SEARCH_SOURCES)
    bad = tools.tools["film_sources"](action="bogus")
    assert "error" in bad
