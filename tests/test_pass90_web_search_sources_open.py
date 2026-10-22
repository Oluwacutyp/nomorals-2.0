"""Pass #90 — web search sources go open (the owner's open-sources law).

Passes #86–#89 took downloads, music and books open through the ONE
registry (``media/source_registry.py``); the remaining named
CLOSED-SOURCE class was web search.  Survey truth this suite pins:

* Web search lives in TWO homes, and both are in scope because both
  are the web-search organ in production:
  (a) ``tools/web.py`` — the keyless HTML-scrape chain
  ``_SEARCH_ENGINES`` (DuckDuckGo HTML/lite, Bing, Mojeek, Ecosia,
  Yahoo, Startpage, Yandex, Brave HTML, Qwant), AGGREGATING (every
  engine is tried, results merged + deduped), driven by the
  ``web_search`` tool, which is what ``agents/search/engine.py``
  ``SearchEngine.search()``, the research pipeline, opportunities
  and watchers all call;
  (b) ``search/web.py`` — the federated backends ``WEB_SPECS``
  (SearXNG keyless metasearch, ddgs keyless package, then the keyed
  tiers Tavily / Serper / Exa / Brave API, dark without keys —
  probe() skips them, which is configuration, not evidence),
  consumed by ``search/federated.py`` behind ``nm search`` / the API.
  Neither recorded an outcome, held health state, or remembered
  anything, so a dead or rate-limited engine was retried in the
  same order forever.
* Architecture decision, with evidence: the registry's model FITS
  the fixed-host engines without distortion — every one is one host
  serving search queries (the #88 music precedent verbatim) — so
  they join the SAME registry as their own class (kinds
  ``search``/``web-search``), with NO extraction facts (the #76/#87
  extractor view is unchanged).  Two honest exceptions, pinned
  below: SearXNG's host is the OWNER-CONFIGURED instance, so its
  descriptor is registered per instance (origin ``registered``),
  and ddgs is a package aggregating other engines — it has no
  single host, gets NO descriptor, and earns no credit it cannot
  be given honestly (its component engines are recorded on the
  tools/web path, where they are hosts).
* Outcome taxonomy: served results credit the engine; an empty
  result set is weak evidence (``record_no_result``, #88 — an
  obscure query must never stand an engine down); a transport /
  HTTP failure records with an error class; a bot-wall (HTTP 403 on
  the scrape chain) is ``record_blocked`` (#89 — never ``dead``);
  an HTTP 429 is its OWN class (``record_rate_limited``, health
  ``rate-limited``) — recorded once, NEVER retried around: this
  suite counts the attempts and pins exactly one call per engine.
* Health probes are honest and quota-aware: keyless engines are
  probed through their real search URL template with one bounded
  probe query (the agency registry's ``Lagos Nigeria``); keyed
  APIs are NEVER probed with a search call (that would burn the
  owner's quota) — the loop only checks the API host root, and a
  non-live root verdict for a keyed engine is downgraded to
  ``unverifiable``, because the root route is not the search
  route; keyed engines are judged by real-search outcomes.
* Discovery verdict — NAMED ABSENCE, no invention: there is NO
  genuine discovery channel for new search engines in code.
  Search RESULT hosts are content sources, not engines (a result
  pointing at a blog does not make the blog a search engine);
  SearXNG results name upstream engines, but Devon has no adapter
  that can query those upstreams directly.  So this pass ships
  outcomes / health / persistence / ordering with NO candidate
  intake on the search path — a search must leave the candidate
  list untouched, pinned below.

All fetches are stubbed — no network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest


# ── helpers ───────────────────────────────────────────────────────────


def _kv(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return KVStore(db), db


def _seeded(store=None):
    from nomorals.media.source_registry import _seed

    return _seed(store=store)


def _ctx(tmp_path, db=None):
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=None)


def _ok_fetcher(url):
    return 200, b"<html><body>a real page with real content</body></html>"


def _dead_fetcher(url):
    return 404, b"gone"


from nomorals.tools import web as web_tools  # noqa: E402
from nomorals.search import web as search_web  # noqa: E402


class _Resp:
    def __init__(self, text="", status=200):
        self.text = text
        self.status = status
        self.ok = 200 <= status < 300


_BING_HTML = ('<li class="b_algo"><h2><a href="https://example.com/b">'
              'Bing title here</a></h2></li>')
_MOJEEK_HTML = ('<a class="title" href="https://example.com/m">'
                'Mojeek title here</a>')


class _ChainClient:
    """Fake search client routing by engine host fragment.

    ``behaviour`` maps an engine name to one of: ("results", html),
    ("empty",), ("raise", exc), ("http", status), ("badparse",).
    Unlisted engines answer an empty page (a no-result).
    """

    _HOSTS = {
        "html.duckduckgo.com": "duckduckgo",
        "lite.duckduckgo.com": "duckduckgo-lite",
        "bing.com": "bing",
        "mojeek.com": "mojeek",
        "ecosia.org": "ecosia",
        "search.yahoo.com": "yahoo",
        "startpage.com": "startpage",
        "yandex.com": "yandex",
        "search.brave.com": "brave",
        "qwant.com": "qwant",
    }

    def __init__(self, behaviour=None):
        self.behaviour = behaviour or {}
        self.calls = []

    def get(self, url, **kw):
        engine = next((name for frag, name in self._HOSTS.items()
                       if frag in url), "?")
        self.calls.append(engine)
        action = self.behaviour.get(engine, ("empty",))
        kind = action[0]
        if kind == "raise":
            raise action[1]
        if kind == "http":
            return _Resp("", status=action[1])
        if kind == "badparse":
            return _Resp(None, status=200)
        if kind == "results":
            return _Resp(action[1], status=200)
        return _Resp("<html><body>no links</body></html>", status=200)


# ── (1) the web-search class joins the SAME registry ──────────────────


def test_every_fixed_host_engine_is_a_registered_seed():
    reg = _seeded()
    origins = {s["domain"]: s["origin"] for s in reg.status()["sources"]}
    for engine, domain in web_tools.SEARCH_ENGINE_DOMAINS.items():
        assert domain, engine
        desc = reg.for_domain(domain)
        assert desc is not None, f"{engine} ({domain}) not registered"
        assert "web-search" in desc.kinds, (engine, desc.kinds)
        assert origins[domain] == "seed", (engine, origins[domain])
    for backend, domain in search_web.WEB_BACKEND_DOMAINS.items():
        if domain is None:
            continue  # searxng (instance host) / ddgs (no host)
        desc = reg.for_domain(domain)
        assert desc is not None, f"{backend} ({domain}) not registered"
        assert "web-search" in desc.kinds, (backend, desc.kinds)
        assert origins[domain] == "seed", (backend, origins[domain])
        assert desc.auth_needed is True, backend  # keyed tier


def test_membership_maps_cover_the_whole_rosters():
    assert set(web_tools.SEARCH_ENGINE_DOMAINS) == \
        set(web_tools._SEARCH_ENGINES)
    assert set(search_web.WEB_BACKEND_DOMAINS) == \
        set(search_web.web_source_names())
    # the honest exceptions: no fixed host exists for these two
    assert search_web.WEB_BACKEND_DOMAINS["web_searxng"] is None
    assert search_web.WEB_BACKEND_DOMAINS["web_ddgs"] is None
    # domains are the hosts actually fetched (evidence from
    # _engine_url / the backend _fetch URLs), not marketing domains
    assert web_tools.SEARCH_ENGINE_DOMAINS["duckduckgo"] == \
        "html.duckduckgo.com"
    assert web_tools.SEARCH_ENGINE_DOMAINS["brave"] == "search.brave.com"
    assert search_web.WEB_BACKEND_DOMAINS["web_brave"] == \
        "api.search.brave.com"  # the API host, a different service
    assert search_web.WEB_BACKEND_DOMAINS["web_tavily"] == "api.tavily.com"
    assert search_web.WEB_BACKEND_DOMAINS["web_serper"] == \
        "google.serper.dev"


def test_web_seeds_do_not_change_the_extractor_view():
    from nomorals.media.download_kit import descriptors_for_registry

    names = [d.name for d in descriptors_for_registry(_seeded())]
    assert names == [
        "netnaija", "audiomack", "boomplay", "notjustok", "tooxclusive",
        "youtube", "soundcloud", "spotify", "tiktok", "instagram",
        "x-twitter", "direct-media", "ytdlp-generic"], names


# ── (2) day-one pin: no learned state → today's order ─────────────────


def test_day_one_chain_order_is_byte_for_byte(tmp_path):
    reg = _seeded()
    # the pin itself: a pristine registry (zero learned state)
    # leaves the chain order untouched, byte-for-byte
    assert web_tools.ordered_search_engines(reg) == \
        list(web_tools._SEARCH_ENGINES)
    behaviour = {"bing": ("results", _BING_HTML),
                 "mojeek": ("results", _MOJEEK_HTML)}
    plain_client = _ChainClient(behaviour)
    plain = web_tools._run_engines(plain_client, "goodbye")
    learned_client = _ChainClient(behaviour)
    learned = web_tools._run_engines(
        learned_client, "goodbye", registry=reg)
    assert learned_client.calls == plain_client.calls == \
        list(web_tools._SEARCH_ENGINES)
    assert [r["url"] for r in learned[0]] == [r["url"] for r in plain[0]]
    assert learned[1] == plain[1] and learned[2] == plain[2]
    # (the run above recorded outcomes into reg — after evidence
    # exists the order may legitimately move; that is test (4)'s
    # subject, not the day-one pin's)


# ── (3) outcome recording on the real search paths ────────────────────


def test_chain_outcomes_are_recorded_per_engine(tmp_path):
    reg = _seeded()
    client = _ChainClient({
        "duckduckgo": ("raise", RuntimeError("ddg down")),
        "duckduckgo-lite": ("badparse",),
        "bing": ("results", _BING_HTML),
        "mojeek": ("http", 429),
        "ecosia": ("http", 403),
    })
    results, used, failed = web_tools._run_engines(
        client, "goodbye", registry=reg)
    assert "bing" in used and "mojeek" in failed and "ecosia" in failed
    # served results credit the serving engine
    st = reg.stats_for(reg.for_domain("bing.com"))
    assert st["ok"] == 1 and st["fail"] == 0, st
    # a transport failure records with an error class
    st = reg.stats_for(reg.for_domain("html.duckduckgo.com"))
    assert st["fail"] == 1 and st["last_error_class"] == \
        "network_failure", st
    # a parser failure is a malformed response, its own class
    st = reg.stats_for(reg.for_domain("lite.duckduckgo.com"))
    assert st["fail"] == 1 and st["last_error_class"] == \
        "malformed_response", st
    # an empty page is weak evidence only — never a breaker trip
    st = reg.stats_for(reg.for_domain("search.yahoo.com"))
    assert st["no_result"] == 1 and st["fail"] == 0, st
    assert reg.health_for(reg.for_domain("search.yahoo.com")) == "unknown"
    # 403 on the scrape chain is a bot wall — blocked, never dead
    desc = reg.for_domain("ecosia.org")
    st = reg.stats_for(desc)
    assert st["blocked"] == 1 and st["last_error_class"] == "bot_block", st
    assert reg.health_for(desc) == "bot-blocked"
    # exactly one attempt per engine: a failure is never retried around
    assert client.calls == list(web_tools._SEARCH_ENGINES), client.calls


def test_rate_limited_is_its_own_class_and_a_single_attempt(tmp_path):
    reg = _seeded()
    client = _ChainClient({"mojeek": ("http", 429)})
    _results, _used, failed = web_tools._run_engines(
        client, "goodbye", registry=reg)
    assert "mojeek" in failed
    assert client.calls.count("mojeek") == 1  # the 429 hard stop: no retry
    desc = reg.for_domain("mojeek.com")
    st = reg.stats_for(desc)
    assert st["rate_limited"] == 1 and st["fail"] == 1, st
    assert st["last_error_class"] == "rate_limited", st
    health = reg.health_for(desc)
    assert health == "rate-limited", health  # NOT dead, NOT disabled
    # repeated rate limits follow the registry's own breaker vocabulary
    reg.record_rate_limited(desc)
    reg.record_rate_limited(desc)
    assert reg.health_for(desc) == "stood-down"
    st = reg.stats_for(desc)
    assert st["rate_limited"] == 3, st


def test_backend_outcomes_are_recorded_per_engine(tmp_path, monkeypatch):
    kv, db = _kv(tmp_path)
    reg = _seeded(store=kv)
    ctx = _ctx(tmp_path, db=db)
    monkeypatch.setenv("NM_TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("NM_WEB_RERANK", "0")
    src = search_web.TavilyWebSource(ctx, source_registry=reg)
    payload = {"results": [{"title": "T", "url": "https://a.example/x",
                            "content": "snippet", "score": 0.9}]}
    calls = []

    def fake_http(*args, **kw):
        calls.append(args)
        return 200, json.dumps(payload).encode()

    with mock.patch.object(search_web, "_http", fake_http):
        hits = src.search("query", limit=3)
    assert hits and hits[0].provenance["provider"] == "web_tavily"
    st = reg.stats_for(reg.for_domain("api.tavily.com"))
    assert st["ok"] == 1 and st["fail"] == 0, st
    # an empty result set is weak evidence, not a failure
    with mock.patch.object(search_web, "_http",
                           lambda *a, **k: (200, b'{"results": []}')):
        assert src.search("obscure query xyz", limit=3) == []
    st = reg.stats_for(reg.for_domain("api.tavily.com"))
    assert st["no_result"] == 1 and st["fail"] == 0, st
    assert reg.health_for(reg.for_domain("api.tavily.com")) != "stood-down"


def test_backend_429_records_rate_limited_once_never_retried(
        tmp_path, monkeypatch):
    kv, db = _kv(tmp_path)
    reg = _seeded(store=kv)
    ctx = _ctx(tmp_path, db=db)
    monkeypatch.setenv("NM_TAVILY_API_KEY", "test-key")
    src = search_web.TavilyWebSource(ctx, source_registry=reg)
    calls = []

    def fake_http(*args, **kw):
        calls.append(args)
        return 429, b"rate limited"

    with mock.patch.object(search_web, "_http", fake_http):
        with pytest.raises(search_web.WebBackendError):
            src.search("query", limit=3)
    assert len(calls) == 1  # a provider 429 is a hard stop — no retry
    desc = reg.for_domain("api.tavily.com")
    st = reg.stats_for(desc)
    assert st["rate_limited"] == 1 and st["fail"] == 1, st
    assert st["last_error_class"] == "rate_limited", st
    assert reg.health_for(desc) == "rate-limited"


def test_backend_auth_failure_is_configuration_not_evidence(
        tmp_path, monkeypatch):
    kv, db = _kv(tmp_path)
    reg = _seeded(store=kv)
    ctx = _ctx(tmp_path, db=db)
    monkeypatch.setenv("NM_TAVILY_API_KEY", "wrong-key")
    src = search_web.TavilyWebSource(ctx, source_registry=reg)
    with mock.patch.object(search_web, "_http",
                           lambda *a, **k: (401, b"unauthorized")):
        with pytest.raises(search_web.WebBackendError):
            src.search("query", limit=3)
    st = reg.stats_for(reg.for_domain("api.tavily.com"))
    assert st["ok"] == 0 and st["fail"] == 0 and st["rate_limited"] == 0, st
    # a keyed engine dark without a key records nothing either:
    # probe() reports the configuration fact and search never runs
    monkeypatch.delenv("NM_TAVILY_API_KEY")
    assert src.probe() is not None
    st = reg.stats_for(reg.for_domain("api.tavily.com"))
    assert st["ok"] == 0 and st["fail"] == 0, st


def test_searxng_records_against_the_configured_instance(
        tmp_path, monkeypatch):
    kv, db = _kv(tmp_path)
    reg = _seeded(store=kv)
    ctx = _ctx(tmp_path, db=db)
    monkeypatch.setenv("NM_SEARXNG_URL", "https://searx.example.com")
    monkeypatch.setenv("NM_WEB_RERANK", "0")
    src = search_web.SearXNGWebSource(ctx, source_registry=reg)
    payload = {"results": [{"url": "https://a.example/x", "title": "A",
                            "content": "snip", "engine": "bing",
                            "score": 1.0}]}
    with mock.patch.object(
            search_web, "_http",
            lambda *a, **k: (200, json.dumps(payload).encode())):
        hits = src.search("query", limit=3)
    assert hits
    desc = reg.for_domain("searx.example.com")
    assert desc is not None  # the instance became a registered source
    assert reg.origin_for(desc) == "registered"
    assert "web-search" in desc.kinds
    st = reg.stats_for(desc)
    assert st["ok"] == 1, st
    # a 429 from the instance: recorded rate-limited, exactly once
    calls = []

    def fake429(*a, **k):
        calls.append(a)
        return 429, b"slow down"

    with mock.patch.object(search_web, "_http", fake429):
        with pytest.raises(search_web.WebBackendError):
            src.search("query", limit=3)
    assert len(calls) == 1
    st = reg.stats_for(desc)
    assert st["rate_limited"] == 1, st
    assert reg.health_for(desc) == "rate-limited"


def test_ddgs_has_no_descriptor_and_earns_no_credit(tmp_path):
    reg = _seeded()
    src = search_web.DdgsWebSource(_ctx(tmp_path), source_registry=reg)
    assert search_web.registry_desc_for_backend(reg, src) is None


def test_search_results_do_not_intake_candidates(tmp_path):
    # Discovery verdict: result hosts are content, not engines —
    # a real search leaves the candidate list untouched.
    reg = _seeded()
    client = _ChainClient({"bing": ("results", _BING_HTML)})
    web_tools._run_engines(client, "goodbye", registry=reg)
    assert reg.candidates() == []


# ── (4) ordering becomes outcome-aware ────────────────────────────────


def test_stood_down_engine_is_demoted_to_last_in_chain_order(tmp_path):
    reg = _seeded()
    desc = reg.for_domain("bing.com")
    for _ in range(3):
        reg.record_outcome(desc, False, error_class="network_failure")
    assert reg.health_for(desc) == "stood-down"
    order = web_tools.ordered_search_engines(reg)
    assert order[-1] == "bing", order  # demoted, still present
    assert [e for e in order if e != "bing"] == \
        [e for e in web_tools._SEARCH_ENGINES if e != "bing"]
    client = _ChainClient({"mojeek": ("results", _MOJEEK_HTML)})
    web_tools._run_engines(client, "goodbye", registry=reg)
    assert client.calls[-1] == "bing", client.calls


def test_federated_order_day_one_and_with_evidence(
        tmp_path, monkeypatch):
    from nomorals.search.federated import federated_search

    kv, db = _kv(tmp_path)
    reg = _seeded(store=kv)
    ctx = _ctx(tmp_path, db=db)
    monkeypatch.setenv("NM_TAVILY_API_KEY", "k")
    monkeypatch.setenv("NM_EXA_API_KEY", "k")
    monkeypatch.setenv("NM_WEB_RERANK", "0")
    payload = {"results": [{"title": "T", "url": "https://a.example/x",
                            "content": "c", "highlights": ["c"]}]}

    def fake_http(method, url, **kw):
        if "tavily" in url:
            return 200, json.dumps(
                {"results": payload["results"]}).encode()
        return 200, json.dumps({"results": [{
            "title": "T", "url": "https://b.example/y",
            "highlights": ["c"]}]}).encode()

    def _adapters():
        return {
            "web_tavily": search_web.TavilyWebSource(
                ctx, source_registry=reg),
            "web_exa": search_web.ExaWebSource(ctx, source_registry=reg),
        }

    with mock.patch.object(search_web, "_http", fake_http):
        resp = federated_search(
            "q", context=ctx, sources=["web_tavily", "web_exa"],
            adapters=_adapters())
    assert resp.sources_searched == ["web_tavily", "web_exa"], \
        resp.sources_searched
    # evidence: tavily stands down → exa is searched first, tavily last
    desc = reg.for_domain("api.tavily.com")
    for _ in range(3):
        reg.record_outcome(desc, False, error_class="search_failed")
    assert reg.health_for(desc) == "stood-down"
    with mock.patch.object(search_web, "_http", fake_http):
        resp = federated_search(
            "q", context=ctx, sources=["web_tavily", "web_exa"],
            adapters=_adapters())
    assert resp.sources_searched == ["web_exa", "web_tavily"], \
        resp.sources_searched


# ── (5) health loop + the existing scheduler job cover web search ─────


def test_health_loop_probes_keyless_with_a_real_query_only():
    reg = _seeded()
    probed = []

    def fetcher(url):
        probed.append(url)
        return 200, b"<html><body>real content</body></html>"

    summary = reg.verify_health(fetcher=fetcher)
    assert summary["checked"] >= len(reg)
    web = reg.status()["web_search"]
    assert web["engines"] and web["backends"]
    assert all(r["health"] == "verified-live" for r in web["engines"]), \
        web["engines"]
    # keyless engines are probed through their real search endpoint
    # (one bounded probe query — the agency registry's probe query)
    assert any("bing.com/search" in u and "Lagos" in u for u in probed), \
        probed
    # keyed APIs are NEVER probed with a search call (quota): the
    # only tavily URL the loop touched is the host root
    tavily_urls = [u for u in probed if "tavily" in u]
    assert tavily_urls == ["https://api.tavily.com/"], tavily_urls


def test_health_loop_never_marks_a_keyed_engine_dead_from_its_root():
    reg = _seeded()

    def fetcher(url):
        if "tavily" in url or "serper" in url or "exa.ai" in url \
                or "api.search.brave" in url:
            return 404, b"no root route"
        return 200, b"<html><body>real content</body></html>"

    reg.verify_health(fetcher=fetcher)
    web = reg.status()["web_search"]
    for row in web["backends"]:
        if row["domain"] is None:
            continue
        assert row["health"] == "unverifiable", row  # NOT dead
    # the dead-fetcher loop still stands the keyless class down by
    # the same vocabulary (3 consecutive failures → stood-down)
    reg2 = _seeded()
    for _ in range(3):
        reg2.verify_health(fetcher=_dead_fetcher)
    row = [r for r in reg2.status()["web_search"]["engines"]
           if r["domain"] == "bing.com"][0]
    assert row["health"] in ("stood-down", "disabled", "dead"), row


def _sched_ctx(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.tools.registry import ToolRegistry

    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=None)
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    ctx.tools = registry
    return ctx, registry


def test_existing_health_job_covers_web_search(tmp_path, monkeypatch):
    from nomorals.agents.scheduler import Scheduler
    from nomorals.media import source_registry as sr
    from nomorals.media.source_jobs import ensure_source_health_jobs

    monkeypatch.setattr(sr, "_default_health_fetch", _ok_fetcher)
    ctx, _registry = _sched_ctx(tmp_path)
    sched = Scheduler(ctx)
    sched.notifier = mock.MagicMock()
    ensure_source_health_jobs(sched)
    row = [j for j in sched.list_jobs()
           if j["name"] == "download source health"][0]
    payload = json.loads(ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE name = ?",
        ("download source health",))["payload"] or "{}")
    assert payload == {"tool": "download_sources",
                       "args": {"action": "health"}}
    out = sched.run_now(row["id"])
    assert out["ok"] is True, out
    from nomorals.media.downloader import MediaDownloader
    web = MediaDownloader(ctx).source_registry().status()["web_search"]
    assert web["engines"] and all(
        r["health"] == "verified-live" for r in web["engines"]), web


# ── (6) persistence + status surfaces ─────────────────────────────────


def test_web_outcomes_persist_across_restart(tmp_path):
    kv, _db = _kv(tmp_path)
    reg = _seeded(store=kv)
    client = _ChainClient({"bing": ("results", _BING_HTML),
                           "mojeek": ("http", 429)})
    web_tools._run_engines(client, "goodbye", registry=reg)
    from nomorals.media.source_registry import _PERSIST_KEY
    raw = kv.get(_PERSIST_KEY)
    payload = json.loads(raw) if isinstance(raw, str) else raw
    assert payload["version"] == 2  # no version bump (#88/#89 precedent)

    reg2 = _seeded(store=kv)  # restart
    st = reg2.stats_for(reg2.for_domain("bing.com"))
    assert st["ok"] == 1, st
    st = reg2.stats_for(reg2.for_domain("mojeek.com"))
    assert st["rate_limited"] == 1, st
    web = reg2.status()["web_search"]
    row = [r for r in web["engines"] if r["domain"] == "bing.com"]
    assert row and row[0]["ok"] == 1 and row[0]["origin"] == "seed", row


def test_status_surfaces_show_the_web_search_class(tmp_path):
    reg = _seeded()
    st = reg.status()
    web = st["web_search"]
    engine_names = {r["engine"] for r in web["engines"]}
    assert {"duckduckgo", "bing", "qwant"} <= engine_names
    backend_names = {r["backend"] for r in web["backends"]}
    assert {"web_tavily", "web_serper", "web_exa", "web_brave",
            "web_searxng", "web_ddgs"} <= backend_names
    from nomorals.media.source_registry import render_status_text
    text = render_status_text(st)
    assert "web search" in text.lower(), text
    assert "bing.com" in text, text
    # the shared download_sources tool carries the same class
    ctx, registry = _sched_ctx(tmp_path)
    outcome = registry.call("download_sources", action="status")
    assert outcome.ok, getattr(outcome, "error", outcome)
    payload = getattr(outcome, "value", None) or outcome.data
    assert "web_search" in payload and payload["web_search"]["engines"], \
        payload


def test_search_sources_tool_surface(tmp_path):
    ctx, registry = _sched_ctx(tmp_path)
    outcome = registry.call("search_sources", action="status")
    assert outcome.ok, getattr(outcome, "error", outcome)
    payload = getattr(outcome, "value", None) or outcome.data
    assert payload["engines"], payload
    assert any(r["engine"] == "bing" for r in payload["engines"])
    assert any(r["backend"] == "web_tavily" for r in payload["backends"])
