"""Pass #91 — price sources go open (the owner's open-sources law).

Passes #86–#90 took downloads, music, books and web search open
through the ONE media registry (``media/source_registry.py``); the
remaining named CLOSED-SOURCE class was prices.  Survey truth this
suite pins:

* The price layer is ``integrations/market_data.py``: a fixed roster
  of quote/OHLCV providers (binance, kraken, coinbase, coingecko,
  yahoo, yahoo_metals, stooq, frankfurter, gold_api, open_er_api,
  ecb + the keyed alphavantage / twelvedata / finnhub upgrades)
  with hardcoded per-market fallback chains, consumed by the
  finance alerts (#75), the finance overview / ``finance_price``
  tool, the morning/afternoon/evening pulses, the price watchers
  and triggers, ``FinancialExpert`` and ``sentinel_bridge``.
* ARCHITECTURE DECISION, with evidence: the price providers do NOT
  join the media registry — they gain mirrored machinery in their
  own home.  (1) Layering law: ``integrations/market_data`` is
  pinned L2 ("imports only core", tests/test_layering.py) while
  ``media`` is L4, so the module may not import the registry at
  all.  (2) Identity: the registry keys sources by
  (nation, language, domain), but a price source's identity is its
  provider id — ``yahoo`` and ``yahoo_metals`` share ONE host
  (query1.finance.yahoo.com) with different coverage, frankfurter
  is one host serving two markets, and nation/language are
  meaningless for a quote (the registry's nation-first ordering
  cannot express per-market chains).  (3) ``market_data`` already
  owns a routing health store (its circuit breaker); registry
  membership would not replace it (the registry knows nothing of
  per-market chains), so joining would CREATE the parallel stats
  store the law forbids.  What is shared is the vocabulary, the
  thresholds and the KV persistence discipline — one vocabulary,
  one discipline, the price class's one store in its own home.
* Outcome taxonomy: a fresh quote landed (``fresh``); a STALE
  quote landed — its provider timestamp older than the provider's
  own cadence (``stale``: its own evidence class, neither full
  success nor transport failure — acting on stale prices is how
  money gets lost); a quote with NO provider timestamp is
  ``served`` with freshness ``unknown`` — a freshness verdict is
  never fabricated; no data for the symbol is weak evidence
  (``no_result``, #88 precedent — it never stands a provider
  down); transport failure with an error class; an HTTP 429 is
  ``rate_limited`` (#90 class — recorded once, never retried
  around); a 403/451 bot/geo wall is ``blocked`` (never ``dead``);
  auth-config missing records NOTHING (#90 precedent).
* Staleness findings: until this pass ``quote()`` returned NO
  timestamp at all, so staleness was unobservable (not an ignored
  field — there was none).  Providers that DO supply a timestamp
  now surface it: binance ``closeTime``, gold-api ``updatedAt``,
  frankfurter / ECB fixing dates, open.er-api
  ``time_last_update_unix``, yahoo ``regularMarketTime``, stooq
  quote Date/Time.  CoinGecko's simple/price response carries
  none — its freshness stays honestly ``unknown``.  OHLCV bars
  always carry timestamps (the frame index), so bar staleness is
  computed against the timeframe cadence.  Consumer accept/reject
  policy is UNCHANGED: a stale quote is still returned (recorded
  and surfaced via ``freshness`` / ``quote_age_s``), never
  silently dropped.
* Health loop + persistence: keyless providers are probed by ONE
  bounded real quote fetch for a liquid symbol (the honest probe);
  keyed providers are NEVER probed (that would burn quota) — they
  are judged by real-fetch outcomes only.  Learned state persists
  through the KV discipline under ``finance.price_sources.v1``
  (version 1, new key — the pre-#91 state was memory-only, so
  there is nothing to migrate).  The status surface is the
  EXISTING ``source_status()`` (extended with ``price_sources``)
  plus a ``price_sources`` tool; the health job is a sibling
  ``ensure_price_source_health_jobs`` binding that real tool
  (#75 contract).
* Discovery verdict — NAMED ABSENCE, no invention: new price
  providers are APIs; no genuine discovery channel exists in code
  (no organ encounters candidate quote APIs on the real path —
  the binance/yahoo mirror hosts other organs call belong to the
  CONNECTOR layer's trading surface, not this keyless quote
  layer, and wiring them would be invention, not discovery).
  The class ships outcomes / health / ordering / persistence
  with NO candidate intake — ``candidates_pending`` is honestly 0.

All fetches are stubbed at ``market_data._fetch`` — no network.
"""

from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.integrations import market_data as md


# ── helpers ───────────────────────────────────────────────────────────


def _kv(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return KVStore(db), db


@pytest.fixture(autouse=True)
def _clean_price_state():
    saved = copy.deepcopy(md._source_health)
    md._source_health.clear()
    md._price_store = None
    yield
    md._source_health.clear()
    md._source_health.update(saved)
    md._price_store = None


def _binance_ticker(close_ms: int | None = None) -> bytes:
    payload = {"symbol": "BTCUSDT", "lastPrice": "61234.5",
               "priceChangePercent": "2.35", "bidPrice": "61230.0",
               "askPrice": "61240.0", "highPrice": "62000.0",
               "lowPrice": "60000.0", "quoteVolume": "1000000"}
    if close_ms is not None:
        payload["closeTime"] = close_ms
    return json.dumps(payload).encode()


_CG_PRICE = json.dumps(
    {"bitcoin": {"usd": 61000, "usd_24h_change": 1.2}}).encode()


def _klines_payload(last_ms: int, n: int = 3) -> bytes:
    rows = []
    for i in range(n):
        o = last_ms - (n - 1 - i) * 3600000
        px = 60000 + i * 100
        rows.append([o, str(px), str(px + 50), str(px - 50),
                     str(px + 10), "12.5", o + 3599999,
                     "750000", 100, "6.25", "375000", "0"])
    return json.dumps(rows).encode()


# ── (1) architecture: mirrored machinery, NOT the media registry ──────


def test_price_providers_are_not_in_the_media_registry():
    from nomorals.media.source_registry import _seed

    reg = _seed()
    for desc in reg.all():
        assert "price" not in desc.kinds, desc
    # and market_data (L2, imports only core/storage peers) never
    # imports the L4 media registry — the layering pin's own terms
    import ast
    from pathlib import Path

    tree = ast.parse(Path(md.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("source_registry" in m for m in imported), imported
    assert not any(m.startswith("nomorals.media")
                   or m == "media" or ".media." in m for m in imported), \
        imported


def test_roster_and_status_surface_cover_every_provider():
    st = md.source_status()
    assert "price_sources" in st  # the EXISTING surface, extended
    rows = {r["source"]: r for r in st["price_sources"]["sources"]}
    for src in md.SOURCES:
        assert src in rows, src
    for src in ("alphavantage", "twelvedata", "finnhub"):
        assert src in rows, src
        assert rows[src]["keyed"] is True
    assert rows["binance"]["tier"] == 1
    assert rows["gold_api"]["tier"] == 3
    # discovery verdict, surfaced honestly — no candidate intake
    assert st["price_sources"]["candidates_pending"] == 0
    assert "no candidate" in st["price_sources"]["discovery"]


# ── (2) day-one pin: zero learned state → today's order ───────────────


def test_day_one_chain_order_is_byte_for_byte():
    for market, chain in md._DEFAULT_CHAINS.items():
        assert md.ordered_price_chain(market) == list(chain), market
    assert md.ordered_price_chain("crypto", ["binance", "coingecko"]) == \
        ["binance", "coingecko"]


def test_day_one_quote_tries_binance_first():
    calls = []

    def fake(url, params=None):
        calls.append(url)
        if "binance" in url:
            return _binance_ticker(int(time.time() * 1000))
        raise AssertionError(f"unexpected {url}")

    with mock.patch.object(md, "_fetch", side_effect=fake):
        q = md.quote("BTC", "crypto")
    assert q["source"] == "binance"
    assert len(calls) == 1


# ── (3) outcome recording on the real fetch paths ─────────────────────


def test_fresh_quote_records_fresh_and_surfaces_timestamp():
    now_ms = int(time.time() * 1000)
    with mock.patch.object(md, "_fetch",
                           return_value=_binance_ticker(now_ms)):
        q = md.quote("BTC", "crypto")
    assert q["freshness"] == "fresh", q
    assert q["quote_ts"] == pytest.approx(now_ms / 1000, abs=1.0)
    assert q["quote_age_s"] < 5
    st = md.price_source_stats("binance")
    assert st["ok"] == 1 and st["stale"] == 0 and st["fail"] == 0, st
    assert md.price_source_health("binance") == "verified-live"


def test_quote_without_provider_timestamp_is_served_unknown():
    # the pre-#91 binance payload shape (no closeTime) and CoinGecko
    # simple/price carry NO timestamp: served, freshness unknown —
    # a verdict is never fabricated.
    with mock.patch.object(md, "_fetch", return_value=_binance_ticker()):
        q = md.quote("BTC", "crypto")
    assert q["freshness"] == "unknown", q
    assert q["quote_ts"] is None
    st = md.price_source_stats("binance")
    assert st["ok"] == 1 and st["stale"] == 0, st


def test_no_data_for_symbol_is_weak_evidence_never_a_failure():
    with mock.patch.object(md, "_fetch", return_value=b"[]"):
        with pytest.raises(md.MarketDataError):
            md.get_ohlcv("NOSUCHCOIN", "crypto", "1h", 5,
                         source="binance")
    st = md.price_source_stats("binance")
    assert st["no_result"] == 1 and st["fail"] == 0, st
    assert md.price_source_health("binance") == "unknown"
    # even repeated catalogue misses never stand the provider down:
    # per-symbol reality — binance not serving NOSUCHCOIN says
    # nothing about binance serving BTC.
    for _ in range(5):
        md.record_quote_outcome("binance", "no_result")
    assert md.price_source_health("binance") == "unknown"
    # weak evidence sinks the SCORE gradually (#88 semantics — the
    # chain may reorder), but binance is never DEMOTED for it: it
    # stays in the active set, not the stood-down tail.
    order = md.ordered_price_chain("crypto")
    assert "binance" in order
    assert md.price_source_health("binance") not in \
        md._INACTIVE_PRICE_HEALTH


def test_transport_failure_records_with_error_class():
    def fake(url, params=None):
        if "binance" in url:
            raise md.MarketDataError("fetch failed x: timed out")
        return _CG_PRICE

    with mock.patch.object(md, "_fetch", side_effect=fake):
        q = md.quote("BTC", "crypto")
    assert q["source"] == "coingecko"
    st = md.price_source_stats("binance")
    assert st["fail"] == 1 and st["last_error_class"] == "timeout", st


def test_rate_limited_is_its_own_class_and_a_single_attempt():
    calls = []

    def fake(url, params=None):
        calls.append(url)
        if "binance" in url:
            raise md.MarketDataError(
                "rate limited (429): https://api.binance.com/x")
        return _CG_PRICE

    with mock.patch.object(md, "_fetch", side_effect=fake):
        q = md.quote("BTC", "crypto")
    assert q["source"] == "coingecko"
    assert sum("binance" in u for u in calls) == 1  # never retried
    st = md.price_source_stats("binance")
    assert st["rate_limited"] == 1, st
    assert st["last_error_class"] == "rate_limited", st
    assert md.price_source_health("binance") == "rate-limited"  # not dead


def test_missing_api_key_records_nothing():
    import os

    env = {k: v for k, v in os.environ.items()
           if k != "ALPHA_VANTAGE_API_KEY"}
    with mock.patch.dict("os.environ", env, clear=True):
        with pytest.raises(md.MarketDataError):
            md.get_ohlcv("AAPL", "stocks", "1d", 5,
                         source="alphavantage")
    st = md.price_source_stats("alphavantage")
    assert st["ok"] == 0 and st["fail"] == 0 and st["no_result"] == 0 \
        and st["rate_limited"] == 0 and st["stale"] == 0, st


def test_ohlcv_outcomes_record_per_attempt():
    now_ms = int(time.time() * 1000)
    with mock.patch.object(md, "_fetch",
                           return_value=_klines_payload(now_ms)):
        df = md.get_ohlcv("BTC/USDT", "crypto", "1h", 3,
                          source="binance")
    assert df.attrs.get("freshness") == "fresh", df.attrs
    assert df.attrs.get("source") == "binance"
    st = md.price_source_stats("binance")
    assert st["ok"] == 1, st


# ── (4) staleness is first-class ──────────────────────────────────────


def test_stale_quote_is_its_own_class_and_still_returned():
    stale_ms = int((time.time() - 3600) * 1000)  # 1h old, cadence 5m
    with mock.patch.object(md, "_fetch",
                           return_value=_binance_ticker(stale_ms)):
        q = md.quote("BTC", "crypto")
    # consumer policy unchanged: the quote is returned, surfaced —
    # never silently dropped, never counted a transport failure.
    assert q["price"] == 61234.5
    assert q["freshness"] == "stale", q
    assert q["quote_age_s"] == pytest.approx(3600, abs=5)
    st = md.price_source_stats("binance")
    assert st["stale"] == 1 and st["ok"] == 0 and st["fail"] == 0, st
    assert st["last_error_class"] == "stale_quote", st
    assert md.price_source_health("binance") == "stale"


def test_stale_ohlcv_bars_are_recorded_stale():
    old_ms = int((time.time() - 30 * 86400) * 1000)  # 30d-old 1h bars
    with mock.patch.object(md, "_fetch",
                           return_value=_klines_payload(old_ms)):
        df = md.get_ohlcv("BTC/USDT", "crypto", "1h", 3,
                          source="binance")
    assert df.attrs.get("freshness") == "stale", df.attrs
    st = md.price_source_stats("binance")
    assert st["stale"] == 1 and st["ok"] == 0, st


def test_daily_fixing_cadence_is_not_flagged_stale_on_a_weekend():
    # frankfurter fixings are DAILY: a 2-day-old fixing (a Monday
    # morning reading Friday's fixing) is within cadence, fresh.
    import datetime as _dt

    day = (_dt.date.today() - _dt.timedelta(days=2)).isoformat()
    payload = json.dumps(
        {"date": day, "base": "EUR", "quote": "USD",
         "rate": 1.086}).encode()
    with mock.patch.object(md, "_fetch", return_value=payload):
        q = md.quote("EURUSD", "forex")
    assert q["source"] == "frankfurter"
    assert q["freshness"] == "fresh", q
    # …but a fixing from 30 days ago is a frozen feed: stale.
    day = (_dt.date.today() - _dt.timedelta(days=30)).isoformat()
    payload = json.dumps(
        {"date": day, "base": "EUR", "quote": "USD",
         "rate": 1.086}).encode()
    md._source_health.clear()
    with mock.patch.object(md, "_fetch", return_value=payload):
        q = md.quote("EURUSD", "forex")
    assert q["freshness"] == "stale", q
    assert md.price_source_stats("frankfurter")["stale"] == 1


def test_gold_api_stale_spot_is_classified_stale():
    from datetime import datetime, timezone

    old = datetime.fromtimestamp(time.time() - 7200, timezone.utc)
    payload = {"price": 4195.60, "currency": "USD", "symbol": "XAU",
               "updatedAt": old.strftime("%Y-%m-%dT%H:%M:%SZ")}

    def fake(url, params=None):
        return payload

    with mock.patch.object(md, "_json", side_effect=fake):
        q = md.quote("XAUUSD", market="forex")
    assert q["source"] == "gold_api"
    assert q["freshness"] == "stale", q
    assert md.price_source_stats("gold_api")["stale"] == 1


# ── (5) outcome-aware selection ───────────────────────────────────────


def test_fresh_serving_provider_is_preferred_with_evidence():
    for _ in range(3):
        md.record_quote_outcome("stooq", "fresh")
    md.record_quote_outcome("yahoo", "failure",
                            error_class="network_failure")
    assert md.ordered_price_chain("stocks") == ["stooq", "yahoo"]


def test_chronically_stale_provider_demotes_to_last():
    desc_order = list(md._DEFAULT_CHAINS["crypto"])
    for _ in range(3):
        md.record_quote_outcome("binance", "stale",
                                error_class="stale_quote")
    assert md.price_source_health("binance") == "stood-down"
    order = md.ordered_price_chain("crypto")
    assert order[-1] == "binance", order  # demoted, still present
    assert [s for s in order if s != "binance"] == \
        [s for s in desc_order if s != "binance"]


def test_demoted_provider_is_last_resort_and_visibly_so():
    for _ in range(3):
        md.record_quote_outcome("binance", "stale",
                                error_class="stale_quote")
    stale_ms = int((time.time() - 3600) * 1000)

    def fake(url, params=None):
        if "binance" in url:
            return _binance_ticker(stale_ms)
        raise md.MarketDataError("fetch failed: down")

    with mock.patch.object(md, "_fetch", side_effect=fake):
        q = md.quote("BTC", "crypto")
    # coingecko failed; the demoted binance is the only source left
    # and still serves — with the demotion VISIBLE on the result.
    assert q["source"] == "binance"
    assert q.get("source_demoted") is True, q


def test_legacy_breaker_vocabulary_still_works():
    src = f"cb91_{id(object())}"
    for _ in range(3):
        md.record_source_result(src, False)
    assert not md._source_healthy(src)
    md.record_source_result(src, True)
    assert md._source_healthy(src)
    assert md.price_source_stats(src)["ok"] == 1


# ── (6) health loop ───────────────────────────────────────────────────


def _fresh_router(url, params=None):
    """Serve every keyless probe a fresh, liquid-symbol payload."""
    now_ms = int(time.time() * 1000)
    if "binance" in url:
        if "klines" in url:
            return _klines_payload(now_ms)
        return _binance_ticker(now_ms)
    if "kraken" in url:
        return json.dumps({"error": [], "result": {
            "XXBTZUSD": [[now_ms // 1000, "60000", "60100", "59900",
                          "60050", "60020", "1.5", 10]],
            "last": now_ms // 1000}}).encode()
    if "coinbase" in url:
        return json.dumps(
            [[now_ms // 1000, 59900, 60100, 60000, 60050, 1.5]]).encode()
    if "coingecko" in url:
        return _CG_PRICE
    if "yahoo" in url:
        return json.dumps({"chart": {"result": [{
            "meta": {"regularMarketPrice": 329.51, "currency": "USD",
                     "regularMarketTime": now_ms // 1000},
            "timestamp": [now_ms // 1000],
            "indicators": {"quote": [{"open": [320.0], "high": [322.0],
                                      "low": [319.0], "close": [329.51],
                                      "volume": [1000]}]}}],
            "error": None}}).encode()
    if "stooq" in url:
        return (b"Date,Open,High,Low,Close,Volume\n"
                + time.strftime("%Y-%m-%d", time.gmtime()).encode()
                + b",230.1,232.5,229.8,231.9,45000000\n")
    if "frankfurter" in url:
        if "/v2/rates" in url:
            return json.dumps([{
                "date": time.strftime("%Y-%m-%d", time.gmtime()),
                "base": "EUR", "quote": "USD", "rate": 1.085}]).encode()
        return json.dumps({
            "date": time.strftime("%Y-%m-%d", time.gmtime()),
            "base": "EUR", "quote": "USD", "rate": 1.085}).encode()
    if "gold-api" in url:
        from datetime import datetime, timezone

        ts = datetime.fromtimestamp(time.time(), timezone.utc)
        return json.dumps({"price": 4195.6, "currency": "USD",
                           "symbol": "XAU",
                           "updatedAt": ts.strftime(
                               "%Y-%m-%dT%H:%M:%SZ")}).encode()
    if "open.er-api" in url:
        return json.dumps({"result": "success",
                           "time_last_update_unix": int(time.time()),
                           "rates": {"EUR": 0.92, "NGN": 1331.19}}).encode()
    if "ecb.europa" in url:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<gesmes:Envelope xmlns:gesmes="http://www.gesmes.org/xml/'
            '2002-08-01" xmlns="http://www.ecb.int/vocabulary/'
            '2002-08-01/eurofxref"><Cube><Cube time="' + day + '">'
            "<Cube currency='USD' rate='1.1206'/>"
            "</Cube></Cube></gesmes:Envelope>").encode()
    raise AssertionError(f"unexpected probe url {url}")


def test_health_loop_probes_keyless_with_real_fetches_only():
    with mock.patch.object(md, "_fetch", side_effect=_fresh_router):
        summary = md.verify_price_sources()
    assert summary["checked"] == len(md.SOURCES), summary
    assert summary["verified_live"] == len(md.SOURCES), summary
    # keyed providers are NEVER probed — quota is the owner's
    assert set(summary["skipped_keyed"]) == \
        {"alphavantage", "twelvedata", "finnhub"}, summary
    assert md.price_source_health("binance") == "verified-live"


def test_health_loop_dead_fetcher_stands_providers_down():
    def dead(url, params=None):
        raise md.MarketDataError("fetch failed: down")

    with mock.patch.object(md, "_fetch", side_effect=dead):
        for _ in range(3):
            md.verify_price_sources()
    assert md.price_source_health("binance") == "stood-down"


# ── (7) persistence + status + the tool/job surfaces ──────────────────


def test_price_outcomes_persist_across_restart(tmp_path):
    kv, _db = _kv(tmp_path)
    assert md.bind_price_source_store(kv) is True
    now_ms = int(time.time() * 1000)
    with mock.patch.object(md, "_fetch",
                           return_value=_binance_ticker(now_ms)):
        md.quote("BTC", "crypto")
    raw = kv.get(md.PRICE_PERSIST_KEY)
    payload = json.loads(raw) if isinstance(raw, str) else raw
    assert payload["version"] == 1, payload
    # restart: in-memory state wiped, the store rebinds and restores
    md._source_health.clear()
    md._price_store = None
    assert md.bind_price_source_store(kv) is True
    st = md.price_source_stats("binance")
    assert st["ok"] == 1, st
    assert md.source_status()["price_sources"]["persisted"] is True
    assert md.clear_price_source_state() is True
    md._source_health.clear()
    md._price_store = None
    md.bind_price_source_store(kv)
    assert md.price_source_stats("binance")["ok"] == 0


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


def test_price_sources_tool_surface(tmp_path):
    _ctx, registry = _sched_ctx(tmp_path)
    outcome = registry.call("price_sources", action="status")
    assert outcome.ok, getattr(outcome, "error", outcome)
    payload = getattr(outcome, "value", None) or outcome.data
    rows = {r["source"] for r in payload["sources"]}
    assert {"binance", "frankfurter", "gold_api"} <= rows, payload
    assert payload["candidates_pending"] == 0


def test_existing_health_job_covers_price_sources(tmp_path):
    from nomorals.agents.scheduler import Scheduler
    from nomorals.finance.price_jobs import (
        ensure_price_source_health_jobs)

    ctx, _registry = _sched_ctx(tmp_path)
    sched = Scheduler(ctx)
    sched.notifier = mock.MagicMock()
    out = ensure_price_source_health_jobs(sched)
    assert out and out[0]["scheduled"] is True, out
    row = [j for j in sched.list_jobs()
           if j["name"] == "price source health"][0]
    payload = json.loads(ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE name = ?",
        ("price source health",))["payload"] or "{}")
    assert payload == {"tool": "price_sources",
                       "args": {"action": "health"}}, payload
    # idempotent
    out2 = ensure_price_source_health_jobs(sched)
    assert out2[0].get("already_scheduled") is True, out2
    # and the job actually runs the real tool → the real health loop
    with mock.patch.object(md, "_fetch", side_effect=_fresh_router):
        result = sched.run_now(row["id"])
    assert result["ok"] is True, result
    assert md.price_source_health("binance") == "verified-live"
