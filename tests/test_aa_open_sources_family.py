"""Assembly-audit closed-source family — AA-53 … AA-63.

The owner's OPEN-SOURCES LAW: provided sources are SEEDS, never a
closed list; every organ must discover new sources, health-check
them, grow into good ones and shed dead ones.  The assembly audit
(2026-11-05) found eleven organs where growth was still missing,
unwired, or unpersisted.  This suite proves each fix end-to-end
with fake fetchers/probers — offline, like every pass suite:

* AA-53  ta/feeds fetch_ohlcv falls over through market_data's
  health-ordered chain and notes results back into it.
* AA-56  market_data discovers catalog price providers, promotes
  verified ones into the chains, sheds dead candidates, persists.
* AA-54  web search intakes SearXNG directory instances as registry
  candidates; promoted engines join the scrape chain in learned
  order AND the federated backend's instance list.
* AA-55  news intakes publisher hosts from fetched items; promoted
  feeds join NewsAgent._feeds and their poll outcomes land back.
* AA-57  the legal music registry folds into the ONE media
  registry; play selection consults its persisted health and
  records real outcomes into it.
* AA-58  reverse image search gains a catalog/manual candidate
  channel; promoted engines join the scrape roster.
* AA-59  grown wisdom archives serve the hunt through the generic
  MediaWiki / advancedsearch shapes, after the seed backends.
* AA-60  proxylab's verified working pool bridges into the
  proxypool connector as tier=free, pool-health-tracked entries.
* AA-61  research's weak_sources computation is canonical in the
  pipeline, consumed by run_job ranking/fetch selection, and
  surfaced on run output.
* AA-62  health_plus's five descriptors are media-registry seeds;
  its SourceHealth delegates to the registry when bound.
* AA-63  owner book descriptors persist in the registry's KV store
  and are restored on first use after a restart.
"""

from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import pytest

# ── shared helpers ────────────────────────────────────────────────────


def _kv(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return KVStore(db), db


def _registry(tmp_path):
    from nomorals.media.source_registry import _seed

    store, _db = _kv(tmp_path)
    return _seed(store=store)


def _ok_fetcher(url):
    return 200, b"<html><body>a real page with real content</body></html>"


def _host_fetcher(live_hosts, body=None):
    """A health-loop fetcher: 200 for URLs on live_hosts, 404 else."""
    def fetch(url):
        for host in live_hosts:
            if host in url:
                return 200, body or b"<html><body>real content</body></html>"
        return 404, b"gone"
    return fetch


# ── AA-53: TA feeds fall over through the price health chain ──────────


@pytest.fixture(autouse=True)
def _clean_price_state():
    # getattr-guarded so this fixture itself never masks a missing
    # AA-56 structure on other trees — each test fails (or passes)
    # on its own evidence.
    from nomorals.integrations import market_data as md

    saved_health = copy.deepcopy(md._source_health)
    # grown providers REGISTER into these tables — snapshot them too
    saved_sources = copy.deepcopy(md.SOURCES)
    saved_tiers = dict(md.SOURCE_TIERS)
    saved_fetchers = dict(md._FETCHERS)
    saved_stale = dict(md._QUOTE_STALE_AFTER_S)
    grown = getattr(md, "_GROWN_PROVIDERS", None)
    cands = getattr(md, "_PRICE_CANDIDATES", None)
    shed = getattr(md, "_PRICE_SHED", None)
    saved_grown = dict(grown) if grown is not None else None
    saved_cands = copy.deepcopy(cands) if cands is not None else None
    saved_shed = copy.deepcopy(shed) if shed is not None else None
    md._source_health.clear()
    if grown is not None:
        grown.clear()
    if cands is not None:
        cands.clear()
    if shed is not None:
        shed.clear()
    md._price_store = None
    if hasattr(md, "_last_discovery_at"):
        md._last_discovery_at = 0.0
    yield
    md._source_health.clear()
    md._source_health.update(saved_health)
    md.SOURCES.clear()
    md.SOURCES.update(saved_sources)
    md.SOURCE_TIERS.clear()
    md.SOURCE_TIERS.update(saved_tiers)
    md._FETCHERS.clear()
    md._FETCHERS.update(saved_fetchers)
    md._QUOTE_STALE_AFTER_S.clear()
    md._QUOTE_STALE_AFTER_S.update(saved_stale)
    if grown is not None and saved_grown is not None:
        grown.clear()
        grown.update(saved_grown)
    if cands is not None and saved_cands is not None:
        cands.clear()
        cands.update(saved_cands)
    if shed is not None and saved_shed is not None:
        shed.clear()
        shed.extend(saved_shed)
    md._price_store = None
    if hasattr(md, "_last_discovery_at"):
        md._last_discovery_at = 0.0


def _frame_stub(rows=3):
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": [100.0] * rows, "high": [101.0] * rows,
         "low": [99.0] * rows, "close": [100.5] * rows,
         "volume": [10.0] * rows}, index=idx)


def test_aa53_dead_exchange_falls_over_and_notes_health(monkeypatch):
    from nomorals.integrations import market_data as md
    from nomorals.ta import feeds

    def dead_binance(*a, **kw):
        raise RuntimeError("exchange down")

    served = {}

    def good_coinbase(symbol, interval, limit, **kw):
        served["call"] = (symbol, interval, limit)
        return _frame_stub()

    monkeypatch.setattr(feeds, "fetch_binance", dead_binance)
    monkeypatch.setattr(feeds, "fetch_coinbase", good_coinbase)
    df = feeds.fetch_ohlcv("binance", "BTC/USDT", "1h", 10,
                           cache_ttl=0)
    assert df.attrs["source"] == "coinbase"
    assert df.attrs["requested_source"] == "binance"
    assert served["call"][0] == "BTC/USDT"
    # both attempts were noted into the price class's health store
    assert md.price_source_stats("binance")["fail"] >= 1
    assert md.price_source_stats("coinbase")["ok"] >= 1


def test_aa53_request_side_valueerror_fails_fast(monkeypatch):
    from nomorals.integrations import market_data as md
    from nomorals.ta import feeds

    calls = []

    def bad_request(*a, **kw):
        raise ValueError("unsupported interval")

    def fallback(symbol, interval, limit, **kw):
        calls.append(symbol)
        return _frame_stub()

    monkeypatch.setattr(feeds, "fetch_binance", bad_request)
    monkeypatch.setattr(feeds, "fetch_coinbase", fallback)
    with pytest.raises(ValueError, match="unsupported interval"):
        feeds.fetch_ohlcv("binance", "BTC/USDT", "7x", 10, cache_ttl=0)
    assert calls == []  # no fallback was attempted
    # and no health blame was recorded for a request-side error
    assert md.price_source_stats("binance")["fail"] == 0


def test_aa53_health_evidence_demotes_in_attempt_order():
    from nomorals.integrations import market_data as md
    from nomorals.ta import feeds

    assert feeds._ta_attempt_order("binance")[0] == "binance"
    for _ in range(3):  # STAND_DOWN_AFTER consecutive failures
        md.record_source_result("binance", False)
    order = feeds._ta_attempt_order("binance")
    assert order[-1] == "binance"  # demoted, never dropped
    assert set(order) == set(feeds.SOURCES)


def test_aa53_unknown_source_still_valueerror():
    from nomorals.ta import feeds

    with pytest.raises(ValueError, match="unknown OHLCV source"):
        feeds.fetch_ohlcv("nowhere", "BTC/USDT")


# ── AA-56: price provider catalog discovery ───────────────────────────


def _okx_klines():
    rows = []
    now_ms = int(time.time() * 1000)
    for i in range(5):
        o = now_ms - (5 - i) * 3600000
        px = 60000 + i * 100
        rows.append([str(o), str(px), str(px + 50), str(px - 50),
                     str(px + 10), "12.5", str(o + 3599999), "750000",
                     "1"])
    return json.dumps({"code": "0", "data": rows}).encode()


def test_aa56_discovery_promotes_verified_catalog_provider(monkeypatch):
    from nomorals.integrations import market_data as md

    def fake_fetch(url, params=None):
        if "okx.com" in url:
            return _okx_klines()
        raise RuntimeError("offline")

    monkeypatch.setattr(md, "_fetch", fake_fetch)
    summary = md.discover_price_providers()
    assert "okx" in summary["promoted"]
    status = md.price_source_status()
    assert "okx" in status["grown"]
    # the grown provider actually serves through the public chain:
    # every seed fetcher is dead (fake_fetch raises for their URLs),
    # so the OHLCV answer must come from discovery + the grown row
    df = md.get_ohlcv("BTC/USDT", "crypto", "1h", 5)
    assert df is not None and len(df) > 0
    assert md.price_source_stats("okx")["ok"] >= 1


def test_aa56_dead_candidates_are_shed_after_three_rounds(monkeypatch):
    from nomorals.integrations import market_data as md

    def dead_fetch(url, params=None):
        raise RuntimeError("offline")

    monkeypatch.setattr(md, "_fetch", dead_fetch)
    for _ in range(3):
        md.discover_price_providers()
    status = md.price_source_status()
    assert status["candidates_pending"] == 0
    summary = md.discover_price_providers()
    assert summary["checked"] == 0  # everything shed, nothing re-probed
    assert len(summary["skipped_shed"]) >= 1


def test_aa56_grown_providers_persist_across_rebind(tmp_path, monkeypatch):
    from nomorals.integrations import market_data as md
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    store = KVStore(db)
    md.bind_price_source_store(store)

    def fake_fetch(url, params=None):
        if "okx.com" in url:
            return _okx_klines()
        raise RuntimeError("offline")

    monkeypatch.setattr(md, "_fetch", fake_fetch)
    md.discover_price_providers()
    assert "okx" in md.price_source_status()["grown"]
    # simulate a restart: in-memory state wiped (the module's own
    # clear, with the store unbound first so the PERSISTED state
    # survives — a restart loses memory, not the store)
    md._price_store = None
    md.clear_price_source_state()
    assert "okx" not in md.price_source_status()["grown"]
    md.bind_price_source_store(KVStore(db))
    assert "okx" in md.price_source_status()["grown"]


# ── AA-54: web search engine discovery ────────────────────────────────


_DIRECTORY = json.dumps({"instances": {
    "https://searx.example.org": {"network_type": "normal"},
    "https://hidden.example.onion": {"network_type": "normal"},
    "http://plain.example.net": {"network_type": "normal"},
    "https://tor.example.org": {"network_type": "tor"},
}})


class _FakeWebClient:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        return SimpleNamespace(text=self._text, status_code=200)


def test_aa54_directory_intake_then_promotion_joins_chain(tmp_path):
    from nomorals.tools import web

    reg = _registry(tmp_path)
    summary = web.discover_search_engines(
        reg, client=_FakeWebClient(_DIRECTORY))
    assert summary["noted"] == 1  # only the https normal instance
    cand = [c for c in reg.candidates()
            if c.domain == "searx.example.org"]
    assert cand and "format=json" in cand[0].search_url_template
    assert web.grown_search_engines(reg) == []  # candidates serve nothing

    body = json.dumps({"results": [
        {"title": "A result", "url": "https://story.example/1",
         "content": "snippet"}]}).encode()
    reg.verify_health(fetcher=_host_fetcher(["searx.example.org"], body))
    grown = web.grown_search_engines(reg)
    assert grown == ["searxng:searx.example.org"]
    # the promoted engine joins the usable set, in learned order
    order = web.ordered_search_engines(reg)
    assert "searxng:searx.example.org" in order
    assert set(web._SEARCH_ENGINES) <= set(order)


def test_aa54_grown_engine_serves_a_real_search(tmp_path):
    from nomorals.tools import web

    reg = _registry(tmp_path)
    web.discover_search_engines(reg, client=_FakeWebClient(_DIRECTORY))
    body = json.dumps({"results": [
        {"title": "Grown result", "url": "https://story.example/9",
         "content": "from the grown engine"}]}).encode()
    reg.verify_health(fetcher=_host_fetcher(["searx.example.org"], body))

    class Client:
        def get(self, url, **kw):
            if "searx.example.org" in url:
                return SimpleNamespace(
                    text=body.decode(), status_code=200)
            raise RuntimeError("seed engines offline in this test")

    results, with_results, _failed = web._run_engines(
        Client(), "lagos", registry=reg,
        engines=["searxng:searx.example.org"])
    assert with_results == ["searxng:searx.example.org"]
    assert results[0]["url"] == "https://story.example/9"
    desc = reg.for_domain("searx.example.org")
    assert reg.stats_for(desc)["ok"] >= 1


def test_aa54_grown_instances_join_federated_failover(tmp_path,
                                                      monkeypatch):
    from nomorals.search.web import SearXNGWebSource
    from nomorals.tools import web

    reg = _registry(tmp_path)
    web.discover_search_engines(reg, client=_FakeWebClient(_DIRECTORY))
    body = json.dumps({"results": []}).encode()
    reg.verify_health(fetcher=_host_fetcher(["searx.example.org"], body))
    monkeypatch.setenv("NM_SEARXNG_URL", "https://mine.example")
    src = SearXNGWebSource()
    monkeypatch.setattr(src, "_registry", lambda: reg)
    instances = src._instances()
    assert instances[0] == "https://mine.example"  # owner's first
    assert "https://searx.example.org" in instances  # grown failover


# ── AA-55: news intake + grown feeds ──────────────────────────────────


def _promote(reg, host, kinds=("news",)):
    assert reg.note_candidate(host, via="test", kinds=kinds)
    reg.verify_health(fetcher=_host_fetcher([host]))


def test_aa55_item_publisher_hosts_intake_as_candidates(tmp_path):
    from nomorals.agents import news as news_mod

    reg = _registry(tmp_path)
    items = [
        {"source_url": "https://freshpub.example/story",
         "url": "https://agg.example/a"},
        {"source_url": "", "url": "https://wire.example/b"},
        {"source_url": "https://freshpub.example/other",
         "url": "https://agg.example/c"},
    ]
    noted = news_mod._note_news_hosts(reg, items, set())
    assert noted == 2  # freshpub once (deduped), wire once
    hosts = {c.domain for c in reg.candidates()}
    assert "freshpub.example" in hosts and "wire.example" in hosts
    assert all("news" in c.kinds for c in reg.candidates())


def test_aa55_promoted_feed_joins_rotation_and_poll_credits(tmp_path):
    from nomorals.agents.news import NewsAgent
    from nomorals.storage.db import Database

    reg = _registry(tmp_path)
    _promote(reg, "newpub.example")
    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(
            news=SimpleNamespace(feeds="")))
    agent = NewsAgent(ctx, registry=reg)
    feeds = dict(agent._feeds())
    assert feeds.get("newpub.example") == "https://newpub.example/feed/" \
        or "https://newpub.example/feed/" in feeds.values()

    rss = ("<?xml version='1.0'?><rss><channel>"
           "<item><title>Markets rally in Lagos</title>"
           "<link>https://agg.example/s1</link>"
           "<source url='https://freshpub.example/'>FreshPub</source>"
           "<description>Stocks climbed through the session.</description>"
           "</item></channel></rss>")

    class FakeResp:
        text = rss

    class FakeClient:
        def get(self, url, headers=None):
            if "newpub.example" in url:
                return FakeResp()
            raise RuntimeError("offline")

    agent._client = FakeClient()
    report = agent.poll()
    assert report["grown_feeds"] == 1
    assert report["candidates_noted"] >= 1  # freshpub.example intaked
    desc = reg.for_domain("newpub.example")
    assert reg.stats_for(desc)["ok"] >= 1  # the serve was credited


# ── AA-57: legal music registry folds into the ONE registry ───────────


def test_aa57_rows_fold_in_and_health_loop_judges_by_outcomes(tmp_path):
    from nomorals.music_sources import registry_bridge as bridge
    from nomorals.music_sources.registry import all_sources

    reg = _registry(tmp_path)
    added = bridge.ensure_music_sources(reg)
    assert added == len(all_sources())
    assert bridge.ensure_music_sources(reg) == 0  # idempotent
    first = all_sources()[0]
    desc = bridge.music_source_desc(reg, first.id)
    assert desc is not None and desc.domain == f"music-source:{first.id}"
    assert set(bridge.MUSIC_KINDS) <= set(desc.kinds)
    # the health loop must NOT probe a synthetic domain into a
    # false death — it is unverifiable by probe, judged by outcomes
    reg.verify_health(fetcher=lambda url: (404, b"gone"))
    assert reg.health_for(desc) == "unverifiable"
    assert reg.is_active(desc)


def test_aa57_play_selection_consults_registry_health(tmp_path):
    from nomorals.music_sources import play as play_mod
    from nomorals.music_sources import registry_bridge as bridge
    from nomorals.music_sources.downloader import DownloadResult
    from nomorals.music_sources.registry import by_class, by_id

    reg = _registry(tmp_path)
    bridge.ensure_music_sources(reg)
    metas = by_class("metadata")
    assert len(metas) >= 2
    dead = metas[0]
    for _ in range(5):  # recorded failures stand the source down
        bridge.record_play_outcome(reg, dead.id, False,
                                   error_class="download_failed")
    assert not bridge.is_source_usable(reg, dead.id)

    # an explicit pick of a stood-down source fails fast, honestly
    res = play_mod.play_via_registry(
        "some song", source_id=dead.id, registry=reg)
    assert res.status == "error" and "not usable" in res.detail

    # search selection skips it and the fallback lands on a usable
    # source; the real download outcome is recorded back
    def search_fn(q):
        return [{"source_id": dead.id, "title": "hit"}]

    def download_fn(source, q):
        return DownloadResult(source_id=source.id,
                              licence=source.licence, downloaded=True,
                              action="fetch", detail="ok")

    res = play_mod.play_via_registry(
        "some song", search_fn=search_fn, download_fn=download_fn,
        registry=reg)
    assert res.status == "ready"
    assert res.source_id != dead.id
    assert reg.stats_for(
        bridge.music_source_desc(reg, res.source_id))["ok"] >= 1
    # and without a registry the same explicit pick behaves as before
    res = play_mod.play_via_registry("some song", source_id=dead.id)
    assert res.status in ("plan_only", "checkout", "ready")
    assert by_id(dead.id) is not None


# ── AA-58: reverse image candidate channel ────────────────────────────


def test_aa58_catalog_intake_promotion_and_scrape(tmp_path):
    from nomorals.tools import imagedb

    reg = _registry(tmp_path)
    summary = imagedb.discover_reverse_image_engines(reg)
    assert summary["noted"] == len(imagedb.REVERSE_IMAGE_CATALOG)
    assert imagedb.grown_image_engines(reg) == {}  # candidates only

    reg.verify_health(fetcher=_host_fetcher(["iqdb.org"]))
    grown = imagedb.grown_image_engines(reg)
    assert list(grown) == ["grown:iqdb.org"]
    order = imagedb.ordered_image_engines(reg)
    assert set(order) == {"bing", "yandex", "grown:iqdb.org"}

    page = (b'<html><img src="https://cdn.other.example/match1.jpg">'
            b'<a href="https://iqdb.org/self.png">self</a>'
            b'<img src="https://iqdb.org/thumb.jpg"></html>')

    class Client:
        def get(self, url, **kw):
            return SimpleNamespace(status=200, body=page)

    urls, kind, _ec = imagedb._scrape_one(
        "grown:iqdb.org", Client(), "https://img.example/pic.jpg",
        registry=reg)
    assert kind == "served"
    assert "https://cdn.other.example/match1.jpg" in urls
    assert not any("iqdb.org" in u for u in urls)  # own host excluded

    results = imagedb._scrape_engines(
        Client(), "https://img.example/pic.jpg", registry=reg)
    assert results["grown:iqdb.org"] == urls
    desc = reg.for_domain("iqdb.org")
    assert reg.stats_for(desc)["ok"] >= 1  # the serve was credited


def test_aa58_manual_candidate_requires_template_shape(tmp_path):
    from nomorals.tools import imagedb

    reg = _registry(tmp_path)
    assert imagedb.note_reverse_image_candidate(
        reg, "newengine.example", "https://newengine.example/s?u={image_url}",
        via="manual") is True
    # a template without the placeholder cannot be served — refused
    assert imagedb.note_reverse_image_candidate(
        reg, "shapeless.example", "https://shapeless.example/s",
        via="manual") is False
    hosts = {c.domain for c in reg.candidates()}
    assert "newengine.example" in hosts
    assert "shapeless.example" not in hosts


# ── AA-59: grown wisdom archives join the hunt ────────────────────────


def _wisdom_ingestor(tmp_path, reg, handler):
    import tempfile
    from types import SimpleNamespace as NS

    from nomorals.wisdom.ingestor import ArchiveIngestor

    class FakeHttp:
        def __init__(self):
            self.calls = []

        def get(self, url, **kw):
            self.calls.append(url)
            return handler(url)

    ctx = NS(settings=NS(workspace_dir=tempfile.mkdtemp(
        prefix="aa59-wisdom-", dir=str(tmp_path))))
    ing = ArchiveIngestor(ctx, registry=reg)
    ing._http = FakeHttp()
    ing._robots = NS(allowed=lambda *a, **k: True)
    return ing


def test_aa59_grown_archive_serves_through_mediawiki_shape(tmp_path):
    from nomorals.core.errors import ProviderError
    from nomorals.core.http import HttpResponse

    reg = _registry(tmp_path)
    _promote(reg, "oldbooks.example", kinds=("wisdom-archive",))

    def handler(url):
        if "oldbooks.example" in url and "/w/api.php" in url:
            payload = {"query": {"search": [
                {"title": "Meditations",
                 "snippet": "Marcus <span>Aurelius</span> notes"}]}}
            return HttpResponse(status=200, body=json.dumps(
                payload).encode(), headers={}, url=url)
        # fail the way the real HttpClient fails (the backends
        # convert NoMoralsError into IngestError)
        raise ProviderError(f"offline: {url}",
                            details={"http_status": 503})

    ing = _wisdom_ingestor(tmp_path, reg, handler)
    found = ing.search("meditations", max_results=5)
    grown_hits = [c for c in found
                  if c.get("source") == "grown:oldbooks.example"]
    assert grown_hits, found
    assert grown_hits[0]["url"] == \
        "https://oldbooks.example/wiki/Meditations"
    desc = reg.for_domain("oldbooks.example")
    assert reg.stats_for(desc)["ok"] >= 1

    # an explicit seed-only order never touches the grown archive
    ing._http.calls.clear()
    try:
        ing.search("meditations", max_results=5, sources=["gutendex"])
    except Exception as exc:  # noqa: BLE001 - all seeds offline here
        assert exc is not None
    assert not any("oldbooks.example" in u for u in ing._http.calls)


def test_aa59_grown_archive_advancedsearch_fallback(tmp_path):
    from nomorals.core.http import HttpResponse
    from nomorals.wisdom.ingestor import DiscoveredArchiveSource

    def handler(url):
        if "/w/api.php" in url:
            return HttpResponse(status=200, body=b"<html>nope</html>",
                                headers={}, url=url)
        if "/advancedsearch.php" in url:
            payload = {"response": {"docs": [
                {"identifier": "tao-te-ching",
                 "title": "Tao Te Ching",
                 "description": "Laozi"}]}}
            return HttpResponse(status=200, body=json.dumps(
                payload).encode(), headers={}, url=url)
        raise RuntimeError("offline")

    class FakeHttp:
        def get(self, url, **kw):
            return handler(url)

    src = DiscoveredArchiveSource("oldbooks.example")
    found = src.search(FakeHttp(), "tao", "", 5)
    assert found and found[0]["url"] == \
        "https://oldbooks.example/details/tao-te-ching"


# ── AA-60: proxylab → proxypool bridge ────────────────────────────────


def test_aa60_import_lab_proxies_into_real_pool():
    from nomorals.accounts.vault import CredentialVault
    from nomorals.connectors.proxypool import ProxyPoolConnector, import_lab_proxies
    from nomorals.storage.db import Database
    from nomorals.tools.proxylab import Proxy

    vault = CredentialVault(Database(":memory:"),
                            master_passphrase="test")
    pool = ProxyPoolConnector(vault)
    rows = [
        Proxy(host="1.2.3.4", port=8080, scheme="http", alive=True,
              latency_ms=42, country="NG"),
        Proxy(host="5.6.7.8", port=1080, scheme="socks5", alive=True,
              latency_ms=90),
        Proxy(host="9.9.9.9", port=3128, scheme="http", alive=False),
    ]
    summary = import_lab_proxies(pool, rows)
    assert summary["imported"] == 2
    assert summary["skipped_dead"] == 1
    listed = {p["id"]: p for p in pool.list_proxies()}
    assert "http://1.2.3.4:8080" in listed
    entry = listed["http://1.2.3.4:8080"]
    assert entry["tier"] == "free"
    assert "proxylab" in entry["tags"]
    assert entry["health"]["status"] not in ("", "unknown") or \
        entry["health"]  # health machinery owns it now
    healthy = {p["id"] for p in pool.healthy_proxies()}
    assert "http://1.2.3.4:8080" in healthy  # recorded success serves


def test_aa60_lab_refresh_reports_pool_import(tmp_path, monkeypatch):
    from nomorals.tools.proxylab import Proxy, ProxyLab

    ctx = SimpleNamespace(settings=SimpleNamespace(
        home=str(tmp_path)))
    lab = ProxyLab(ctx)
    lab.store.save([], candidates=[
        Proxy(host="1.2.3.4", port=8080, scheme="http", alive=True)])

    class FakeTester:
        detect_country = False

        def test_many(self, proxies, limit=150):
            for p in proxies:
                p.alive = True
                p.latency_ms = 33
            return proxies

    lab._tester = FakeTester()
    imported = []

    class FakePool:
        def add_proxy(self, host, port, **kw):
            pid = f"{kw.get('protocol', 'http')}://{host}:{port}"
            imported.append(pid)
            return {"id": pid}

        def record_success(self, pid, latency_ms=None):
            return {"id": pid}

    monkeypatch.setattr(lab, "_pool_connector", lambda: FakePool())
    report = lab.refresh(scrape_if_empty=False)
    assert report["working"] == 1
    assert report["pool_imported"] == 1
    assert imported == ["http://1.2.3.4:8080"]


# ── AA-61: research weak sources consumed ─────────────────────────────


def _research_db(tmp_path):
    from nomorals.research.autonomy import ResearchOrgan
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    organ = ResearchOrgan(SimpleNamespace(db=db))  # ensures schema
    return db, organ


def test_aa61_canonical_weak_set_shared_by_report_and_pipeline(tmp_path):
    from nomorals.research import pipeline as P

    db, organ = _research_db(tmp_path)
    for _ in range(4):
        organ.record_source_quality("https://noise.example/a", False)
    for _ in range(4):
        organ.record_source_quality("https://good.example/a", True)
    weak = P.weak_domains(db)
    assert set(weak) == {"noise.example"}
    assert weak["noise.example"]["worth_rate"] == 0.0
    # the organ's report IS the same computation now
    assert organ.weak_sources() == list(weak.values())


def test_aa61_run_job_fetch_selection_skips_weak_domains(tmp_path):
    from nomorals.research import pipeline as P

    db, organ = _research_db(tmp_path)
    for _ in range(4):
        organ.record_source_quality("https://noise.example/a", False)

    class Ok:
        def __init__(self, value):
            self.ok, self.value, self.error = True, value, ""

    fetched = []

    class FakeRegistry:
        def call(self, name, actor="system", **kwargs):
            if name == "web_search":
                return Ok({"results": [
                    {"title": "weak hit", "url": "https://noise.example/x",
                     "snippet": "s"},
                    {"title": "strong hit",
                     "url": "https://strong.example/y", "snippet": "s"},
                ]})
            if name == "web_fetch":
                fetched.append(kwargs["url"])
                return Ok({"text": "full text"})
            return Ok({})

    rctx = P.ResearchContext(db=db, registry=FakeRegistry(),
                             memory=None, gateway=None)
    job = P.ResearchJob(id="j1", topic="t", queries=["q"],
                        fetch_top=1)
    findings = P.run_job(job, rctx)
    # the weak finding is still RETURNED (never filtered) …
    assert {f.url for f in findings} == {
        "https://noise.example/x", "https://strong.example/y"}
    # … but the single depth-fetch slot went to the strong domain
    assert fetched == ["https://strong.example/y"]


def test_aa61_run_output_surfaces_the_weak_set(tmp_path):
    from nomorals.research.autonomy import ResearchOrgan
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()

    class Ok:
        def __init__(self, value):
            self.ok, self.value, self.error = True, value, ""

    class FakeRegistry:
        def call(self, name, actor="system", **kwargs):
            if name == "web_search":
                return Ok({"results": [
                    {"title": "strong hit",
                     "url": "https://strong.example/y", "snippet": "s"}]})
            return Ok({"text": "full text"})

    ctx = SimpleNamespace(db=db, tools=FakeRegistry(), memory=None,
                          gateway=None)
    organ = ResearchOrgan(ctx)
    for _ in range(4):
        organ.record_source_quality("https://noise.example/a", False)
    watch_id = organ.add_watch("topic", ["q"])
    result = organ.run_watch(watch_id)
    # AA-61: the weak set the run consumed is ON the run's output
    assert "weak_sources" in result
    assert [w["domain"] for w in result["weak_sources"]] == [
        "noise.example"]


# ── AA-62: health_plus seeds + delegation ─────────────────────────────


def test_aa62_all_five_descriptors_are_registry_seeds():
    from nomorals.media.source_registry import _seed
    from nomorals.sources.health_plus import NEW_SOURCE_DESCRIPTORS

    reg = _seed()
    for row in NEW_SOURCE_DESCRIPTORS:
        desc = reg.for_domain(row["domain"])
        assert desc is not None, row["domain"]
        assert set(row["kinds"]) <= set(desc.kinds), row["domain"]


def test_aa62_source_health_delegates_to_registry(tmp_path):
    from nomorals.sources.health_plus import SourceHealth

    reg = _registry(tmp_path)
    sh = SourceHealth(registry=reg)
    desc = reg.for_domain("techcrunch.com")
    result = sh.probe("techcrunch.com", lambda s: (True, 0.05))
    assert result.ok
    assert reg.stats_for(desc)["ok"] == 1  # landed in the ONE registry
    assert sh.success_rate("techcrunch.com") == 1.0
    assert sh.is_healthy("techcrunch.com")
    for _ in range(10):
        sh.probe("techcrunch.com", lambda s: (False, 0.01))
    assert not reg.is_active(desc)  # registry stood it down/disabled it
    assert not sh.is_healthy("techcrunch.com")
    chain = sh.fallback_chain(["techcrunch.com", "espn.com"])
    assert chain[0] == "espn.com"  # the registry's standing orders it
    # unbound, the class keeps its in-memory behaviour exactly
    plain = SourceHealth()
    plain.probe("x.example", lambda s: (True, 0.01))
    assert plain.success_rate("x.example") == 1.0


# ── AA-63: owner book descriptors persist ─────────────────────────────


def test_aa63_owner_descriptor_survives_restart(tmp_path):
    from nomorals.books import descriptors as bd
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    store = KVStore(db)
    ctx = SimpleNamespace(db=db)
    saved_table = list(bd.DESCRIPTORS)
    saved_ids = set(bd._OWNER_IDS)
    try:
        desc = bd.SourceDescriptor(
            id="ownerlib", name="ownerlib",
            domains=("ownerlib.example",),
            licence="Author-posted (owner-added)",
            rights_class="author_posted")
        bd.register_descriptor(desc, context=ctx)
        rows = store.get(bd._OWNER_KV_KEY)
        assert rows and rows[0]["id"] == "ownerlib"

        # simulate a restart: in-memory table back to built-ins,
        # load-on-first-use restores from the store
        bd.DESCRIPTORS = [d for d in saved_table]
        bd._OWNER_IDS = set(saved_ids)
        assert bd.descriptor_by_id("ownerlib") is None
        assert bd.descriptor_by_id("ownerlib", context=ctx) is not None
        adapters = bd.descriptor_adapters(store=store)
        assert any(a.name == "ownerlib" for a in adapters)
        assert any(s["id"] == "ownerlib"
                   for s in bd.list_sources(context=ctx))
    finally:
        bd.DESCRIPTORS = saved_table
        bd._OWNER_IDS = saved_ids
