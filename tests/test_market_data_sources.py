"""Tests for hardened finance data sources: commodities market,
forex redundancy, source health / circuit breaker, reliability tiers.

All network calls are mocked — these test logic, not live endpoints.
Live verification was done 2026-10-10 via curl (see commit message).
"""
from unittest.mock import patch

import pytest

from nomorals.integrations import market_data as md


# ── market auto-detection ─────────────────────────────────────────────
class TestMarketDetection:
    def test_xauusd_routes_to_commodities(self):
        assert md._detect_market("XAUUSD", "forex") == "commodities"

    def test_xau_routes_to_commodities(self):
        assert md._detect_market("XAU", "forex") == "commodities"

    def test_gold_routes_to_commodities(self):
        assert md._detect_market("GOLD", "forex") == "commodities"

    def test_eurusd_stays_forex(self):
        assert md._detect_market("EURUSD", "forex") == "forex"

    def test_btc_stays_crypto(self):
        assert md._detect_market("BTC", "crypto") == "crypto"


class TestNormalizeCommodities:
    def test_xauusd_splits(self):
        n = md.normalize_symbol("XAUUSD", "commodities")
        assert n["base"] == "XAU" and n["quote"] == "USD"

    def test_gold_aliases_to_xau(self):
        n = md.normalize_symbol("GOLD", "commodities")
        assert n["base"] == "XAU"

    def test_metal_yahoo_mapping(self):
        n = md.normalize_symbol("XAUUSD", "commodities")
        assert n["metal"] == "GC=F"
        n = md.normalize_symbol("XAGUSD", "commodities")
        assert n["metal"] == "SI=F"


# ── gold-api.com adapter ──────────────────────────────────────────────
class TestGoldApi:
    def _fake(self, url, params=None):
        if "gold-api.com/price/XAU" in url:
            return {"price": 4195.60, "currency": "USD", "symbol": "XAU"}
        raise AssertionError(f"unexpected {url}")

    def test_quote(self):
        with patch.object(md, "_json", side_effect=self._fake):
            q = md.quote("XAUUSD", market="forex")
        assert q["price"] == 4195.60
        assert q["currency"] == "USD"
        assert q["source"] == "gold_api"

    def test_unsupported_metal(self):
        with pytest.raises(md.MarketDataError):
            md._goldapi_quote({"base": "COPPER", "raw": "COPPER"})


# ── ECB direct adapter ────────────────────────────────────────────────
_ECB_XML = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<gesmes:Envelope xmlns:gesmes="http://www.gesmes.org/xml/2002-08-01"'
    b' xmlns="http://www.ecb.int/vocabulary/2002-08-01/eurofxref">'
    b"<Cube><Cube time='2026-10-09'>"
    b"<Cube currency='USD' rate='1.1206'/>"
    b"<Cube currency='GBP' rate='0.84763'/>"
    b"</Cube></Cube></gesmes:Envelope>"
)


class TestEcbDirect:
    def test_eur_usd(self):
        with patch.object(md, "_fetch", return_value=_ECB_XML):
            q = md._ecb_direct_quote(
                {"base": "EUR", "quote": "USD", "raw": "EURUSD"})
        assert abs(q["price"] - 1.1206) < 0.0001
        assert q["source"] == "ecb"

    def test_cross_rate(self):
        # GBP/USD = (EUR/USD) / (EUR/GBP)
        with patch.object(md, "_fetch", return_value=_ECB_XML):
            q = md._ecb_direct_quote(
                {"base": "GBP", "quote": "USD", "raw": "GBPUSD"})
        assert abs(q["price"] - 1.1206 / 0.84763) < 0.0001

    def test_missing_currency(self):
        with patch.object(md, "_fetch", return_value=_ECB_XML):
            with pytest.raises(md.MarketDataError):
                md._ecb_direct_quote(
                    {"base": "EUR", "quote": "XXX", "raw": "EURXXX"})


# ── open.er-api.com adapter ───────────────────────────────────────────
class TestOpenErApi:
    def test_quote(self):
        def fake(url, params=None):
            return {"rates": {"NGN": 1331.19}, "result": "success"}

        with patch.object(md, "_json", side_effect=fake):
            q = md._open_er_api_quote(
                {"base": "USD", "quote": "NGN", "raw": "USDNGN"})
        assert q["price"] == 1331.19
        assert q["source"] == "open_er_api"


# ── circuit breaker ───────────────────────────────────────────────────
class TestCircuitBreaker:
    def setup_method(self):
        # use a unique source name per test run to avoid cross-test state
        self.src = f"cb_test_{id(self)}"
        md._source_health.pop(self.src, None)

    def test_trips_after_threshold(self):
        for _ in range(3):
            md.record_source_result(self.src, False)
        assert not md._source_healthy(self.src)

    def test_recovers_on_success(self):
        for _ in range(3):
            md.record_source_result(self.src, False)
        assert not md._source_healthy(self.src)
        md.record_source_result(self.src, True)
        assert md._source_healthy(self.src)

    def test_healthy_chain_filters(self):
        for _ in range(3):
            md.record_source_result(self.src, False)
        chain = md._healthy_chain([self.src, "binance"])
        assert self.src not in chain
        assert "binance" in chain


# ── reliability tiers ─────────────────────────────────────────────────
class TestTiers:
    def test_infrastructure_is_tier1(self):
        for src in ("binance", "kraken", "coinbase", "ecb", "frankfurter"):
            assert md.SOURCE_TIERS[src] == 1, src

    def test_community_is_tier3(self):
        assert md.SOURCE_TIERS["gold_api"] == 3
        assert md.SOURCE_TIERS["open_er_api"] == 3

    def test_health_includes_tiers(self):
        h = md.source_health()
        assert h["binance"]["tier"] == 1
        assert h["gold_api"]["tier"] == 3
        assert "cooling_down" in h["binance"]


# ── chains ────────────────────────────────────────────────────────────
class TestChains:
    def test_commodities_chain(self):
        chain = md._DEFAULT_CHAINS["commodities"]
        assert "yahoo_metals" in chain
        assert "frankfurter" in chain

    def test_forex_has_redundancy(self):
        chain = md._DEFAULT_CHAINS["forex"]
        assert chain.index("frankfurter") < chain.index("open_er_api")
        assert chain.index("open_er_api") < chain.index("ecb")
