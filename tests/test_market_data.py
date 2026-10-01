"""Tests for the free keyless market-data adapters (Prompt 07 follow-up).

All HTTP is mocked at ``market_data._fetch`` — no network, no keys.
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.integrations import market_data as md


def _binance_klines_payload(n=5):
    base = 1700000000000
    rows = []
    for i in range(n):
        o = base + i * 3600000
        px = 60000 + i * 100
        rows.append([o, str(px), str(px + 50), str(px - 50),
                     str(px + 10), "12.5", o + 3599999,
                     "750000", 100, "6.25", "375000", "0"])
    return json.dumps(rows).encode()


_STOOQ_CSV = b"""Date,Open,High,Low,Close,Volume
2026-09-25,230.1,232.5,229.8,231.9,45000000
2026-09-26,231.9,233.0,230.5,232.4,42000000
2026-09-27,232.4,234.1,231.0,233.8,48000000
"""

_FRANKFURTER = json.dumps([
    {"date": "2026-09-25", "base": "EUR", "quote": "USD", "rate": 1.085},
    {"date": "2026-09-26", "base": "EUR", "quote": "USD", "rate": 1.087},
    {"date": "2026-09-27", "base": "EUR", "quote": "USD", "rate": 1.083},
]).encode()

_BINANCE_TICKER = json.dumps({
    "symbol": "BTCUSDT", "lastPrice": "61234.5",
    "priceChangePercent": "2.35"}).encode()

_CG_OHLC = json.dumps([
    [1700000000000, 60000, 60100, 59900, 60050],
    [1700001800000, 60050, 60200, 60000, 60150],
]).encode()

_STOOQ_QUOTE = b"Symbol,Date,Time,Open,High,Low,Close,Volume\nAAPL.US,2026-09-27,22:00:00,232.4,234.1,231.0,233.8,48000000\n"


class NormalizeTests(unittest.TestCase):
    def test_crypto_pair(self):
        s = md.normalize_symbol("BTC/USDT", "crypto")
        self.assertEqual(s["binance"], "BTCUSDT")
        self.assertEqual(s["coinbase"], "BTC-USDT")
        self.assertEqual(s["kraken"], "XXBTUSDT")

    def test_crypto_bare_defaults_usdt(self):
        s = md.normalize_symbol("ETH", "crypto")
        self.assertEqual(s["binance"], "ETHUSDT")

    def test_stocks(self):
        s = md.normalize_symbol("AAPL", "stocks")
        self.assertEqual(s["stooq"], "aapl.us")

    def test_forex(self):
        s = md.normalize_symbol("EURUSD", "forex")
        self.assertEqual(s["base"], "EUR")
        self.assertEqual(s["quote"], "USD")


class BinanceTests(unittest.TestCase):
    def test_klines_shape(self):
        with patch.object(md, "_fetch",
                          return_value=_binance_klines_payload(10)):
            df = md.get_ohlcv("BTC/USDT", "crypto", "1h", 10,
                              source="binance")
        self.assertEqual(list(df.columns),
                         ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 10)
        self.assertAlmostEqual(df["close"].iloc[-1], 60910.0)
        self.assertTrue((df["high"] >= df["low"]).all())

    def test_bad_timeframe(self):
        with self.assertRaises(md.MarketDataError):
            md.get_ohlcv("BTC/USDT", "crypto", "9h", 10, source="binance")


class FallbackTests(unittest.TestCase):
    def test_auto_falls_to_next_source(self):
        # binance/kraken/coinbase down → coingecko (last in chain) serves
        def fake2(url, params=None):
            if "binance" in url or "kraken" in url or "coinbase" in url:
                raise md.MarketDataError("down")
            return _CG_OHLC

        with patch.object(md, "_fetch", side_effect=fake2):
            df = md.get_ohlcv("BTC", "crypto", "1h", 2, source="auto")
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df["close"].iloc[0], 60050.0)

    def test_all_fail_lists_sources(self):
        with patch.object(md, "_fetch",
                          side_effect=md.MarketDataError("nope")):
            with self.assertRaises(md.MarketDataError) as cm:
                md.get_ohlcv("BTC", "crypto", "1h", 10, source="auto")
        msg = str(cm.exception)
        self.assertIn("binance", msg)
        self.assertIn("kraken", msg)


class StooqTests(unittest.TestCase):
    def test_daily_csv(self):
        with patch.object(md, "_fetch", return_value=_STOOQ_CSV):
            df = md.get_ohlcv("AAPL", "stocks", "1d", 3, source="stooq")
        self.assertEqual(len(df), 3)
        self.assertAlmostEqual(df["close"].iloc[-1], 233.8)
        self.assertEqual(df["volume"].iloc[0], 45000000.0)


class FrankfurterTests(unittest.TestCase):
    def test_daily_fixings(self):
        with patch.object(md, "_fetch", return_value=_FRANKFURTER):
            df = md.get_ohlcv("EURUSD", "forex", "1d", 3,
                              source="frankfurter")
        self.assertEqual(len(df), 3)
        # single daily fixing → O=H=L=C
        row = df.iloc[-1]
        self.assertAlmostEqual(row["open"], row["high"])
        self.assertAlmostEqual(row["high"], row["low"])
        self.assertAlmostEqual(row["low"], 1.083)


_YAHOO_CHART = json.dumps({
    "chart": {"result": [{
        "meta": {"regularMarketPrice": 329.51, "currency": "USD",
                 "symbol": "AAPL"},
        "timestamp": [1700000000, 1700086400, 1700172800],
        "indicators": {"quote": [{
            "open": [320.0, 325.0, 328.0],
            "high": [322.0, 327.0, 330.0],
            "low": [319.0, 324.0, 327.0],
            "close": [321.0, 326.0, 329.51],
            "volume": [1000, 1100, 1200]}]},
    }], "error": None},
}).encode()


class YahooTests(unittest.TestCase):
    def test_symbol_forms(self):
        self.assertEqual(
            md._yahoo_symbol(md.normalize_symbol("EURUSD", "forex"),
                             "forex"), "EURUSD=X")
        self.assertEqual(
            md._yahoo_symbol(md.normalize_symbol("AAPL", "stocks"),
                             "stocks"), "AAPL")
        self.assertEqual(
            md._yahoo_symbol(md.normalize_symbol("BTC/USDT", "crypto"),
                             "crypto"), "BTC-USD")

    def test_chart_bars(self):
        with patch.object(md, "_fetch", return_value=_YAHOO_CHART):
            df = md.get_ohlcv("AAPL", "stocks", "1d", 3, source="yahoo")
        self.assertEqual(len(df), 3)
        self.assertAlmostEqual(df["close"].iloc[-1], 329.51)
        self.assertEqual(list(df.columns),
                         ["open", "high", "low", "close", "volume"])

    def test_range_grows_with_bars(self):
        self.assertEqual(md._yahoo_range("1h", 100), "5d")
        self.assertEqual(md._yahoo_range("1d", 400), "1y")
        self.assertEqual(md._yahoo_range("1d", 1500), "5y")


class QuoteTests(unittest.TestCase):
    def test_crypto_quote_binance(self):
        with patch.object(md, "_fetch", return_value=_BINANCE_TICKER):
            q = md.quote("BTC", "crypto")
        self.assertAlmostEqual(q["price"], 61234.5)
        self.assertAlmostEqual(q["change_pct_24h"], 2.35)
        self.assertEqual(q["source"], "binance")

    def test_crypto_quote_falls_back_to_coingecko(self):
        cg = json.dumps(
            {"bitcoin": {"usd": 61000, "usd_24h_change": 1.2}}).encode()

        def fake(url, params=None):
            if "binance" in url:
                raise md.MarketDataError("blocked")
            return cg

        with patch.object(md, "_fetch", side_effect=fake):
            q = md.quote("BTC", "crypto")
        self.assertEqual(q["source"], "coingecko")
        self.assertAlmostEqual(q["price"], 61000.0)

    def test_stock_quote(self):
        with patch.object(md, "_fetch", return_value=_YAHOO_CHART):
            q = md.quote("AAPL", "stocks")
        self.assertAlmostEqual(q["price"], 329.51)
        self.assertEqual(q["source"], "yahoo")

    def test_stock_quote_falls_back_to_stooq(self):
        def fake(url, params=None):
            if "yahoo" in url:
                raise md.MarketDataError("blocked")
            return _STOOQ_QUOTE

        with patch.object(md, "_fetch", side_effect=fake):
            q = md.quote("AAPL", "stocks")
        self.assertEqual(q["source"], "stooq")
        self.assertAlmostEqual(q["price"], 233.8)

    def test_forex_quote(self):
        payload = json.dumps(
            {"date": "2026-09-27", "base": "EUR", "quote": "USD",
             "rate": 1.086}).encode()
        with patch.object(md, "_fetch", return_value=payload):
            q = md.quote("EURUSD", "forex")
        self.assertAlmostEqual(q["price"], 1.086)
        self.assertEqual(q["source"], "frankfurter")


class KeyedUpgradeTests(unittest.TestCase):
    def test_missing_key_is_helpful(self):
        import os
        env = {k: v for k, v in os.environ.items()
               if k not in ("ALPHA_VANTAGE_API_KEY", "TWELVEDATA_API_KEY",
                            "FINNHUB_API_KEY")}
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaises(md.MarketDataError) as cm:
                md.get_ohlcv("AAPL", "stocks", "1d", 5,
                             source="alphavantage")
            self.assertIn("ALPHA_VANTAGE_API_KEY", str(cm.exception))
            with self.assertRaises(md.MarketDataError) as cm2:
                md.get_ohlcv("AAPL", "stocks", "1d", 5,
                             source="twelvedata")
            self.assertIn("TWELVEDATA_API_KEY", str(cm2.exception))

    def test_keyed_joins_auto_chain_when_set(self):
        import os
        env = dict(os.environ, TWELVEDATA_API_KEY="demo")
        with patch.dict("os.environ", env, clear=True):
            chain = md._keyed_chain("stocks")
        self.assertIn("twelvedata", chain)


class ProviderTests(unittest.TestCase):
    def test_quote_and_movers(self):
        prov = md.SentinelMarketProvider()
        quotes = {"BTC": {"symbol": "BTC", "price": 60000.0,
                          "change_pct_24h": 2.0, "currency": "USDT",
                          "source": "binance"},
                  "ETH": {"symbol": "ETH", "price": 3000.0,
                          "change_pct_24h": -5.0, "currency": "USDT",
                          "source": "binance"}}
        with patch.object(md, "quote",
                          side_effect=lambda s, market="crypto": quotes[s]):
            movers = prov.overnight_movers(["BTC", "ETH"])
        self.assertEqual(movers[0]["symbol"], "ETH")  # |−5| > |2|
        self.assertEqual(movers[1]["symbol"], "BTC")

    def test_quote_failure_returns_none(self):
        prov = md.SentinelMarketProvider()
        with patch.object(md, "quote",
                          side_effect=md.MarketDataError("down")):
            self.assertIsNone(prov.quote("BTC"))

    def test_source_status(self):
        st = md.source_status()
        self.assertIn("binance", st["keyless"])
        self.assertIn("frankfurter", st["keyless"])
        self.assertIn("default_chains", st)


class LazyImportTests(unittest.TestCase):
    def test_no_pandas_at_import(self):
        import subprocess
        import sys
        from pathlib import Path
        code = (
            "import sys;"
            "import nomorals.integrations.market_data;"
            "import nomorals.agents.financial_expert;"
            "leaked=[m for m in ('pandas','numpy') if m in sys.modules];"
            "print('LEAKED:'+','.join(leaked) if leaked else 'CLEAN')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            timeout=120, cwd=str(Path(__file__).resolve().parents[1]))
        self.assertIn("CLEAN", out.stdout, out.stderr)


if __name__ == "__main__":
    unittest.main()
