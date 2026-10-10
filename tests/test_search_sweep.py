"""Sweep tests for the nomorals/search upgrade.

Covers the changed behavior only: weighted RRF + source_rank metadata,
field-weighted BM25/BM25+/k3/tokenizer seam, the fixed OSINT hit
construction, the new OSINT adapters (mocked HTTP), OSINT wiring in
build_adapters, federated timings + weighted fusion, SearXNG/ddgs query
surface, adaptive query helpers, and the new render styles.
"""

from __future__ import annotations

import sys
import types

import pytest

from nomorals.search import adaptive as adaptive_mod
from nomorals.search import federated as federated_mod
from nomorals.search import model as model_mod
from nomorals.search import osint as osint_mod
from nomorals.search import render as render_mod
from nomorals.search import rerank as rerank_mod
from nomorals.search import web as web_mod
from nomorals.search.adaptive import (
    adaptive_result_limit,
    detect_intent,
    freshness_intent,
    normalize_query,
)
from nomorals.search.errors import SearchError
from nomorals.search.model import SearchResponse, SearchResult, reciprocal_rank_fusion
from nomorals.search.osint import (
    CourtListenerAdapter,
    DisifyAdapter,
    EdgarAdapter,
    GleifAdapter,
    XposedOrNotAdapter,
    build_osint_hit,
)
from nomorals.search.render import (
    highlight_matches,
    render_search,
    source_badge,
)
from nomorals.search.rerank import bm25_field_scores, bm25_rerank, bm25_scores


def _hit(title, snippet="", source="books", type="book", score=0.5, query="q"):
    return SearchResult(
        query=query, title=title, snippet=snippet,
        source=source, type=type, score=score, raw_score=score,
        provenance={}, source_id=f"{source}:{title}",
    )


# ── weighted RRF ──────────────────────────────────────────────────────

def test_rrf_weights_trust_one_source_more():
    a1 = _hit("alpha doc", source="s1")
    b1 = _hit("beta doc", source="s2")
    plain, _ = reciprocal_rank_fusion([[a1], [b1]])
    assert plain[0].title == "alpha doc"  # tie-break on title
    a2 = _hit("alpha doc", source="s1")
    b2 = _hit("beta doc", source="s2")
    weighted, _ = reciprocal_rank_fusion([[a2], [b2]], weights=[0.1, 10.0])
    assert weighted[0].title == "beta doc"
    assert weighted[0].score > weighted[1].score


def test_rrf_default_weights_is_plain_rrf():
    a1 = _hit("x", source="s1")
    b1 = _hit("y", source="s2")
    r1, _ = reciprocal_rank_fusion([[a1], [b1]])
    a2 = _hit("x", source="s1")
    b2 = _hit("y", source="s2")
    r2, _ = reciprocal_rank_fusion([[a2], [b2]], weights=[1.0, 1.0])
    assert [h.title for h in r1] == [h.title for h in r2]
    assert r1[0].score == pytest.approx(r2[0].score)


def test_rrf_records_source_rank_in_provenance():
    h1 = _hit("one", source="s1")
    h2 = _hit("two", source="s1")
    fused, _ = reciprocal_rank_fusion([[h1, h2]])
    ranks = {h.title: h.provenance["source_rank"] for h in fused}
    assert ranks == {"one": 1, "two": 2}


def test_rrf_does_not_overwrite_existing_source_rank():
    h = _hit("one", source="s1")
    h.provenance["source_rank"] = 99
    reciprocal_rank_fusion([[h]])
    assert h.provenance["source_rank"] == 99


def test_rrf_bad_weights_raise():
    h = _hit("x", source="s1")
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[h]], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[h]], weights=[-1.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[h]], k=0)


# ── BM25 upgrades ─────────────────────────────────────────────────────

def test_bm25_field_scores_title_beats_snippet():
    terms = ["quantum"]
    titles = [["quantum"], ["nothing"]]
    snippets = [["nothing relevant here"], ["quantum quantum quantum"]]
    scores = bm25_field_scores(terms, titles, snippets, title_weight=2.0)
    assert scores[0] > scores[1]


def test_bm25_field_scores_weight_one_is_flat():
    terms = ["quantum"]
    titles = [["quantum"], ["nothing"]]
    snippets = [["nothing"], ["quantum"]]
    boosted = bm25_field_scores(terms, titles, snippets, title_weight=2.0)
    flat = bm25_field_scores(terms, titles, snippets, title_weight=1.0)
    assert boosted[0] - boosted[1] > flat[0] - flat[1]


def test_bm25_field_scores_misaligned_raises():
    with pytest.raises(ValueError):
        bm25_field_scores(["a"], [["a"]], [])


def test_bm25_plus_delta_lifts_long_docs():
    terms = ["cat"]
    docs = [["cat"], ["cat"] + ["filler"] * 200]
    plain = bm25_scores(terms, docs, delta=0.0)
    plus = bm25_scores(terms, docs, delta=1.0)
    assert plain[0] > plain[1]
    # BM25+ narrows the long-doc penalty instead of collapsing it
    assert plus[1] > plain[1]


def test_bm25_k3_saturates_repeated_query_terms():
    terms = ["cat", "cat", "cat"]
    docs = [["cat"], ["dog"]]
    no_k3 = bm25_scores(terms, docs)
    with_k3 = bm25_scores(terms, docs, k3=1.0)
    # without k3 the query term counts once; with k3=1, qf=3 the
    # multiplier is (k3+1)*qf/(k3+qf) = 1.5 — saturation, not linear growth
    assert with_k3[0] == pytest.approx(no_k3[0] * 1.5)
    assert with_k3[0] < no_k3[0] * 3.0  # not linear in query frequency


def test_bm25_rerank_uses_title_weight_and_tokenizer_seam():
    hits = [
        _hit("weather today", "quantum quantum physics"),  # 2 snippet hits
        _hit("quantum guide", "unrelated text"),  # 1 title hit
    ]
    # default title_weight=2.0: the title hit outranks two snippet hits
    ranked = bm25_rerank(hits, "quantum")
    assert ranked[0].title == "quantum guide"
    assert ranked[0].raw_score > 0
    # classic flat treatment (weight 1.0): raw frequency wins instead
    ranked_flat = bm25_rerank(hits, "quantum", title_weight=1.0)
    assert ranked_flat[0].title == "weather today"
    # custom tokenizer seam: drop everything (all scores zero → title order)
    ranked_none = bm25_rerank(hits, "quantum", tokenizer=lambda s: [])
    assert [h.title for h in ranked_none] == sorted(h.title for h in hits)
    assert all(h.raw_score == 0.0 for h in ranked_none)


# ── OSINT hit construction ────────────────────────────────────────────

def test_build_osint_hit_is_well_formed():
    adapter = XposedOrNotAdapter()
    hit = build_osint_hit(
        adapter, "a@b.com", "breach exposure: a@b.com",
        "https://xposedornot.com", "exposed in 2 breaches", 0.95,
        confidence="high",
    )
    assert isinstance(hit, SearchResult)
    assert hit.query == "a@b.com"
    assert hit.source == "osint_xon"
    assert hit.type == "breach"
    assert hit.provenance["url"] == "https://xposedornot.com"
    assert hit.provenance["confidence"] == "high"
    assert hit.source_id.startswith("osint_xon:")


def test_xposedornot_parses_nested_and_flat_shapes(monkeypatch):
    nested = (200, b'{"breaches": [["Adobe", "LinkedIn"]], "email": "a@b.com"}', "")
    monkeypatch.setattr(osint_mod, "_get", lambda url, timeout=15: nested)
    hits = XposedOrNotAdapter().search("a@b.com", limit=5)
    assert len(hits) == 1
    assert "Adobe" in hits[0].snippet and "LinkedIn" in hits[0].snippet
    assert hits[0].score == 0.95

    flat = (200, b'{"breaches": ["Adobe"], "email": "a@b.com"}', "")
    monkeypatch.setattr(osint_mod, "_get", lambda url, timeout=15: flat)
    hits = XposedOrNotAdapter().search("a@b.com", limit=5)
    assert "Adobe" in hits[0].snippet

    clean = (200, b'{"breaches": [], "email": "a@b.com"}', "")
    monkeypatch.setattr(osint_mod, "_get", lambda url, timeout=15: clean)
    hits = XposedOrNotAdapter().search("a@b.com", limit=5)
    assert "Not found" in hits[0].snippet
    assert hits[0].score == 0.5


def test_xposedornot_rate_limit_yields_no_hit_not_error(monkeypatch):
    monkeypatch.setattr(osint_mod, "_get", lambda url, timeout=15: (429, b"", ""))
    assert XposedOrNotAdapter().search("a@b.com", limit=5) == []


def test_xposedornot_rejects_non_email():
    assert XposedOrNotAdapter().search("not an email", limit=5) == []


def test_disify_parses_verdict(monkeypatch):
    payload = {"format": True, "domain": "b.com", "dns": True,
               "disposable": False, "whitelist": False}
    monkeypatch.setattr(osint_mod, "_get_json", lambda url, timeout=15: payload)
    hits = DisifyAdapter().search("a@b.com", limit=3)
    assert len(hits) == 1
    assert "valid" in hits[0].snippet
    assert hits[0].type == "email"


def test_edgar_resolves_ticker(monkeypatch):
    payload = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    monkeypatch.setattr(osint_mod, "_get_json", lambda url, timeout=15: payload)
    hits = EdgarAdapter().search("AAPL", limit=5)
    assert len(hits) == 1
    assert hits[0].provenance["cik"] == "0000320193"
    assert "sec.gov" in hits[0].provenance["url"]
    assert hits[0].type == "company"


def test_gleif_parses_lei_records(monkeypatch):
    payload = {"data": [{"attributes": {
        "lei": "5493006MHB84DD0ZWV18",
        "entity": {"legalName": {"name": "Test Corp"},
                   "legalAddress": {"city": "Lagos", "country": "NG"}},
        "registration": {"status": "ISSUED"}}}]}
    monkeypatch.setattr(osint_mod, "_get_json", lambda url, timeout=15: payload)
    hits = GleifAdapter().search("Test Corp", limit=5)
    assert len(hits) == 1
    assert "5493006MHB84DD0ZWV18" in hits[0].title
    assert "Lagos" in hits[0].snippet


def test_courtlistener_parses_opinions(monkeypatch):
    payload = {"results": [{
        "caseName": "Doe v. Roe", "court": "S.D.N.Y.",
        "dateFiled": "2024-05-01",
        "absolute_url": "/opinion/123/doe-v-roe/",
        "snippet": "the <b>court</b> held..."}]}
    monkeypatch.setattr(osint_mod, "_get_json", lambda url, timeout=15: payload)
    hits = CourtListenerAdapter().search("John Doe", limit=5)
    assert len(hits) == 1
    assert hits[0].title == "Doe v. Roe"
    assert "<b>" not in hits[0].snippet  # HTML stripped
    assert hits[0].type == "legal"


# ── OSINT wiring ──────────────────────────────────────────────────────

def test_build_adapters_keeps_classic_set_and_osint_builds_lazily():
    from types import SimpleNamespace
    from nomorals.search.sources import (
        build_adapters,
        build_osint_adapter,
        valid_source_names,
    )

    ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir="/tmp/ws"))
    adapters = build_adapters(ctx)
    # the context-bound set is unchanged (fleet contract)
    assert set(adapters) == {
        "memory", "wisdom", "books", "docs", "code", "timeline",
        "web_searxng", "web_ddgs", "web_tavily", "web_serper",
        "web_exa", "web_brave",
    }
    # OSINT adapters are context-free: built on demand by name
    for name in valid_source_names():
        if name.startswith("osint_"):
            adapter = build_osint_adapter(name)
            assert adapter is not None, f"no lazy adapter for {name}"
            assert adapter.name == name
    assert build_osint_adapter("memory") is None
    assert build_osint_adapter("nope") is None


def test_federated_reaches_osint_sources():
    ctx = types.SimpleNamespace(
        settings=types.SimpleNamespace(workspace_dir="/tmp/ws"))
    resp = federated_mod.federated_search(
        "a@b.com", context=ctx, sources=["osint_disify", "osint_xon"],
        limit=3)
    assert set(resp.sources_searched) == {"osint_disify", "osint_xon"}
    # and with prebuilt adapters that lack them (lazy fill, no crash)
    resp2 = federated_mod.federated_search(
        "a@b.com", sources=["osint_disify"], adapters={}, limit=3)
    assert resp2.sources_searched == ["osint_disify"]


def test_federated_still_raises_for_truly_unknown_adapter():
    with pytest.raises(SearchError):
        federated_mod.federated_search(
            "q", sources=["memory"], adapters={}, limit=3)


# ── federated timings + weighted fusion ───────────────────────────────

class _StubAdapter:
    def __init__(self, name, hits):
        self.name = name
        self._hits = hits

    def probe(self):
        return None

    def search(self, query, *, limit, since=None, before=None):
        return [h for h in self._hits][:limit]


def _stub_adapters():
    return {
        "memory": _StubAdapter("memory", [_hit("alpha memory", source="memory")]),
        "wisdom": _StubAdapter("wisdom", [_hit("beta wisdom", source="wisdom")]),
    }


def test_federated_records_timings_and_elapsed():
    resp = federated_mod.federated_search(
        "q", sources=["memory", "wisdom"], adapters=_stub_adapters(), limit=5)
    assert set(resp.timings) == {"memory", "wisdom"}
    assert all(t >= 0 for t in resp.timings.values())
    assert resp.elapsed >= 0
    d = resp.to_dict()
    assert "timings" in d and "elapsed" in d


def test_federated_weighted_rrf_dict_and_list():
    a = _stub_adapters()
    r_dict = federated_mod.federated_search(
        "q", sources=["memory", "wisdom"], adapters=a, limit=5,
        fusion="rrf", source_weights={"memory": 0.01, "wisdom": 100.0})
    assert r_dict.hits[0].source == "wisdom"

    a = _stub_adapters()
    r_list = federated_mod.federated_search(
        "q", sources=["memory", "wisdom"], adapters=a, limit=5,
        fusion="rrf", source_weights=[0.01, 100.0])
    assert [h.source for h in r_list.hits] == [h.source for h in r_dict.hits]


def test_federated_weights_rejected_for_legacy_and_bad_shapes():
    with pytest.raises(SearchError):
        federated_mod.federated_search(
            "q", sources=["memory"], adapters=_stub_adapters(), limit=5,
            fusion="legacy", source_weights={"memory": 2.0})
    with pytest.raises(SearchError):
        federated_mod.federated_search(
            "q", sources=["memory", "wisdom"], adapters=_stub_adapters(),
            limit=5, fusion="rrf", source_weights=[1.0])  # wrong length
    with pytest.raises(SearchError):
        federated_mod.federated_search(
            "q", sources=["memory", "wisdom"], adapters=_stub_adapters(),
            limit=5, fusion="rrf",
            source_weights={"nope": 1.0, "memory": 1.0})  # missing key


# ── web query surface ─────────────────────────────────────────────────

def test_searxng_sends_category_time_range_engines(monkeypatch):
    captured = {}

    def fake_http(method, url, *, params=None, json_body=None,
                  headers=None, timeout=10.0):
        captured["params"] = dict(params or {})
        return 200, b'{"results": []}'

    monkeypatch.setattr(web_mod, "_http", fake_http)
    monkeypatch.setenv("NM_SEARXNG_URL", "http://localhost:8888")
    monkeypatch.setenv("NM_SEARXNG_CATEGORY", "news")
    monkeypatch.setenv("NM_SEARXNG_TIME_RANGE", "week")
    monkeypatch.setenv("NM_SEARXNG_ENGINES", "google,bing")
    monkeypatch.setenv("NM_SEARXNG_LANGUAGE", "en")
    src = web_mod.SearXNGWebSource()
    assert src.probe() is None
    assert src._fetch("latest ai", 5) == []
    params = captured["params"]
    assert params["categories"] == "news"
    assert params["time_range"] == "week"
    assert params["engines"] == "google,bing"
    assert params["language"] == "en"


def test_searxng_invalid_category_falls_back_to_general(monkeypatch):
    monkeypatch.setenv("NM_SEARXNG_CATEGORY", "not-a-category")
    src = web_mod.SearXNGWebSource()
    assert src._category() == "general"
    monkeypatch.setenv("NM_SEARXNG_TIME_RANGE", "epoch")
    assert src._time_range() == ""


def test_ddgs_kind_verticals(monkeypatch):
    calls = []

    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, *a, **k):
            calls.append("text")
            return [{"title": "t", "href": "http://x", "body": "b"}]

        def news(self, *a, **k):
            calls.append("news")
            return [{"title": "n", "href": "http://x/n", "body": "nb"}]

        def images(self, *a, **k):
            calls.append("images")
            return [{"title": "i", "image": "http://x/i.jpg"}]

        def videos(self, *a, **k):
            calls.append("videos")
            return [{"title": "v", "content": "http://x/v.mp4"}]

    fake_ddgs = types.ModuleType("ddgs")
    fake_ddgs.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_ddgs)
    monkeypatch.setattr(web_mod.DdgsWebSource, "_ddgs_missing", False)

    src = web_mod.DdgsWebSource()
    assert src.probe() is None

    monkeypatch.setenv("NM_DDGS_KIND", "news")
    rows = src._fetch("elections", 5)
    assert calls[-1] == "news" and rows[0]["url"] == "http://x/n"

    monkeypatch.setenv("NM_DDGS_KIND", "images")
    rows = src._fetch("cats", 5)
    assert calls[-1] == "images" and rows[0]["url"] == "http://x/i.jpg"

    monkeypatch.setenv("NM_DDGS_KIND", "bogus-kind")
    src._fetch("cats", 5)
    assert calls[-1] == "text"  # invalid kind falls back to text


# ── adaptive helpers ──────────────────────────────────────────────────

def test_normalize_query():
    assert normalize_query("  hello   WORLD!!! ") == "hello WORLD!"
    assert normalize_query("what is @user's email??") == "what is @user's email?"
    assert normalize_query("") == ""
    assert normalize_query("a@b.com") == "a@b.com"


def test_freshness_intent():
    assert freshness_intent("latest AI news")
    assert freshness_intent("bitcoin price 2026")
    assert freshness_intent("breaking: earthquake today")
    assert not freshness_intent("how to bake bread")
    assert not freshness_intent("")


def test_detect_intent():
    assert detect_intent("someone@example.com") == "osint"
    assert detect_intent("8.8.8.8") == "osint"
    assert detect_intent("example.com") == "osint"
    assert detect_intent("somehandle") == "osint"
    assert detect_intent("+2348012345678") == "osint"
    assert detect_intent("buy cheap laptop") == "transactional"
    assert detect_intent("github login page") == "navigational"
    assert detect_intent("how do black holes work") == "informational"
    assert detect_intent("") == "informational"


def test_adaptive_limit_rewards_freshness():
    plain = adaptive_result_limit("AI developments")
    fresh = adaptive_result_limit("latest AI developments")
    assert fresh > plain


# ── render styles ─────────────────────────────────────────────────────

def _resp():
    return SearchResponse(
        query="quantum guide",
        hits=[
            _hit("quantum guide", "a guide to quantum computing",
                 source="books", score=0.9, query="quantum guide"),
            _hit("quantum news", "latest quantum breakthrough",
                 source="web_searxng", score=0.7, query="quantum guide"),
        ],
        sources_searched=["books", "web_searxng"],
        timings={"books": 0.01, "web_searxng": 0.2},
        elapsed=0.25,
    )


def test_render_plain_has_count_header_and_badges():
    out = render_search(_resp(), style="plain")
    assert "2 result(s) for 'quantum guide'" in out
    assert "[books]" in out and "[web_searxng]" in out


def test_render_markdown_structure():
    out = render_search(_resp(), style="markdown")
    assert out.startswith("## 2 result(s) for 'quantum guide'")
    assert "**" in out  # highlighted matches


def test_render_compact_one_line_per_hit():
    out = render_search(_resp(), style="compact")
    lines = [l for l in out.splitlines() if l and not l.startswith("2 result")]
    assert len(lines) >= 2
    assert all(l[0].isdigit() for l in lines[:2])


def test_render_unknown_style_raises():
    with pytest.raises(ValueError):
        render_search(_resp(), style="hologram")


def test_render_no_results_is_designed():
    resp = SearchResponse(
        query="zzz-nope", hits=[],
        sources_searched=["books"],
        sources_skipped={"web_searxng": "no SearXNG instance configured"},
        deduped=0,
    )
    out = render_search(resp, style="plain")
    assert "No matches for 'zzz-nope'" in out
    assert "web_searxng" in out  # skip notes surfaced
    assert "Try:" in out


def test_render_group_by_source_sections():
    out = render_search(_resp(), style="plain", group_by_source=True)
    assert out.index("[books]") < out.index("[web_searxng]")


def test_render_timings_footer():
    out = render_search(_resp(), style="plain")
    assert "250ms total" in out
    assert "books 10ms" in out


def test_highlight_matches_wraps_terms(monkeypatch):
    monkeypatch.setattr(render_mod, "_supports_color", lambda: True)
    out = highlight_matches("a guide to QUANTUM computing", "quantum guide")
    assert "\033[" in out and "QUANTUM" in out
    # no color support → text unchanged
    monkeypatch.setattr(render_mod, "_supports_color", lambda: False)
    assert highlight_matches("quantum", "quantum") == "quantum"
    # markdown flavor
    out = highlight_matches("quantum guide", "quantum", markdown=True)
    assert "**quantum**" in out


def test_source_badge_plain_has_no_ansi():
    assert source_badge("books", color=False) == "[books]"
