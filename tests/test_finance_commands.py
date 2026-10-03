"""Tests for the deepened FinancialExpert: trade ideas, analysis, backtests,
price-alert wiring, the keyless-first bridge routing, and the /finance
chat command.

The native TA stack (nomorals.ta) runs for real here; only the market-data
layer is faked with deterministic synthetic bars — no network needed.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from nomorals.agents import financial_expert as fe
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.integrations import market_data
from nomorals.integrations import sentinel_bridge as bridge
from nomorals.integrations.market_data import MarketDataError
from nomorals.ta import data as ta_data


def _trend_bars(n=600, drift=0.002, seed=0):
    """Deterministic uptrend fixture the committee reads as LONG (default)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, n)))
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998,
         "close": close, "volume": 1000.0}, index=idx)


def _expert():
    return fe.FinancialExpert(SimpleNamespace(db=None, settings=None))


# The Sentinel.py submodule is private; CI and external checkouts cannot
# fetch it. Tests that need the real bridge skip gracefully when it is
# absent — run `git submodule update --init vendor/sentinel` for full
# coverage. Tests for the absent-submodule behavior itself always run.
requires_sentinel = unittest.skipUnless(
    bridge.sentinel_available(),
    "vendor/sentinel not checked out (private submodule)")


class TradeIdeaTests(unittest.TestCase):
    def test_long_idea_math(self):
        df = _trend_bars()
        with patch.object(market_data, "get_ohlcv", return_value=df):
            idea = _expert().trade_idea("BTC", "crypto", "default")
        self.assertEqual(idea.direction, "long")
        self.assertAlmostEqual(idea.entry, df["close"].iloc[-1])
        self.assertLess(idea.stop, idea.entry)
        self.assertLess(idea.entry, idea.target_1)
        self.assertLess(idea.target_1, idea.target_2)
        # stop = entry - 2*ATR, target_2 = entry + 4*ATR → exactly 1:2
        self.assertAlmostEqual(idea.risk_reward, 2.0)
        self.assertGreater(idea.confidence, 0)
        self.assertLessEqual(idea.confidence, 0.95)
        text = idea.summary_text()
        self.assertIn("LONG", text)
        self.assertIn("not financial advice", text)

    def test_conservative_gate_is_stricter(self):
        # same bars, but conservative demands unanimous committee agreement
        df = _trend_bars()
        with patch.object(market_data, "get_ohlcv", return_value=df):
            idea = _expert().trade_idea("BTC", "crypto", "conservative")
        self.assertEqual(idea.direction, "flat")

    def test_flat_idea_on_chop(self):
        df = ta_data.make_synthetic(600, seed=7)
        with patch.object(market_data, "get_ohlcv", return_value=df):
            idea = _expert().trade_idea("BTC", "crypto", "default")
        self.assertEqual(idea.direction, "flat")
        self.assertIn("FLAT", idea.summary_text())

    def test_bad_profile_rejected(self):
        with self.assertRaises(bridge.SentinelError):
            _expert().trade_idea("BTC", "crypto", "yolo")

    def test_data_unreachable_is_sentinel_error(self):
        with patch.object(market_data, "get_ohlcv",
                          side_effect=MarketDataError("no adapters")):
            with self.assertRaises(bridge.SentinelError):
                _expert().trade_idea("BTC", "crypto", "default")


class AnalyzeTests(unittest.TestCase):
    def test_analyze_report_shape(self):
        df = _trend_bars()
        with patch.object(market_data, "get_ohlcv", return_value=df):
            rep = _expert().analyze("BTC", "crypto")
        self.assertEqual(rep.symbol, "BTC")
        self.assertEqual(rep.regime_label, "TREND_UP")
        self.assertGreater(rep.bias, 0.5)
        self.assertEqual(rep.position_now, 1.0)
        self.assertEqual(rep.n_strategies, 9)
        self.assertEqual(rep.bars, 600)
        names = [s["name"] for s in rep.strategies]
        # ranked = committee members with >= 5 round-trips; the rest still vote
        self.assertTrue(0 < len(names) <= 9)
        self.assertEqual(sorted(names), sorted(set(names)))
        for n in names:
            self.assertIn(n, ["bollinger_squeeze", "breakout",
                              "ichimoku_trend", "mean_reversion", "momentum",
                              "rsi_divergence", "sar_reversal", "trend_follow",
                              "vwap_bounce"])
        # stop distance stored as a fraction — the summary must read ~1%, not ~100%
        self.assertLess(rep.stop_distance_pct, 1.0)
        text = rep.summary_text()
        self.assertIn("too few trades to rank", text)
        self.assertIn("TREND_UP", text)
        self.assertIn("not financial advice", text)
        self.assertNotIn("100.0%", text)

    def test_analyze_data_unreachable(self):
        with patch.object(market_data, "get_ohlcv",
                          side_effect=MarketDataError("no adapters")):
            with self.assertRaises(bridge.SentinelError):
                _expert().analyze("BTC", "crypto")


class BacktestExpertTests(unittest.TestCase):
    def test_strategy_backtest_runs(self):
        df = _trend_bars()
        with patch.object(market_data, "get_ohlcv", return_value=df):
            out = _expert().backtest("BTC", "crypto",
                                     strategy="trend_follow")
        self.assertIn("trend_follow", out.verdict)
        for key in ("total_return", "sharpe", "max_dd", "win_rate",
                    "trades", "final_equity"):
            self.assertIn(key, out.metrics, key)
        self.assertIn("PASSES the bar", out.verdict)

    def test_committee_backtest_label(self):
        df = _trend_bars()
        with patch.object(market_data, "get_ohlcv", return_value=df):
            out = _expert().backtest("BTC", "crypto")
        self.assertIn("committee", out.verdict)

    def test_unknown_strategy_rejected(self):
        df = _trend_bars()
        with patch.object(market_data, "get_ohlcv", return_value=df):
            with self.assertRaises(bridge.SentinelError):
                _expert().backtest("BTC", "crypto", strategy="nope")

    def test_bad_profile_rejected(self):
        df = _trend_bars()
        with patch.object(market_data, "get_ohlcv", return_value=df):
            with self.assertRaises(bridge.SentinelError):
                _expert().backtest("BTC", "crypto", profile="yolo")

    def test_strategies_registry(self):
        rows = _expert().strategies()
        self.assertEqual([r["name"] for r in rows],
                         ["bollinger_squeeze", "breakout", "ichimoku_trend",
                          "mean_reversion", "momentum", "rsi_divergence",
                          "sar_reversal", "trend_follow", "vwap_bounce"])
        for r in rows:
            self.assertTrue(r["blurb"])
            self.assertTrue(r["params"])


class WatchPriceTests(unittest.TestCase):
    def test_registers_market_watcher(self):
        seen = {}

        class _FakeAgent:
            def __init__(self, context):
                seen["context"] = context

            def add(self, text, **overrides):
                seen.update(overrides)
                seen["text"] = text
                return {"ok": True,
                        "watcher": {"id": "w1"},
                        "echo": "echo"}

        with patch("nomorals.agents.watchers.WatcherAgent", _FakeAgent):
            res = _expert().watch_price(
                "BTC", "crypto",
                condition={"op": "lt", "field": "value", "value": 60000})
        self.assertTrue(res["ok"])
        self.assertEqual(seen["kind"], "price")
        self.assertEqual(seen["target"]["source"], "market")
        self.assertEqual(seen["target"]["symbol"], "BTC")
        self.assertEqual(seen["target"]["market"], "crypto")
        self.assertEqual(seen["condition"]["op"], "lt")


class BridgeRoutingTests(unittest.TestCase):
    def test_auto_routes_to_free_adapters(self):
        from nomorals.integrations import market_data
        with patch.object(bridge, "_ensure_path", lambda: None), \
             patch.object(market_data, "get_ohlcv",
                          return_value="BARS") as m:
            out = bridge.load_data("BTC/USDT", "crypto", "1h", 500,
                                   source="auto")
        self.assertEqual(out, "BARS")
        m.assert_called_once_with("BTC/USDT", market="crypto",
                                  timeframe="1h", bars=500)

    @requires_sentinel
    def test_ccxt_path_needs_package(self):
        # ccxt is not installed in this env → helpful error, not ImportError.
        # (The real _ensure_path runs: vendor/sentinel is checked out.)
        with self.assertRaises(bridge.SentinelError) as cm:
            bridge.load_data("BTC/USDT", "crypto", source="ccxt")
        self.assertIn("pip install ccxt", str(cm.exception))

    def test_doctor_reports_market_data(self):
        with patch.object(bridge, "VENDOR_ROOT", __import__(
                "pathlib").Path("/nonexistent-xyz")):
            rep = bridge.doctor()
        self.assertFalse(rep.ok)  # submodule absent → not ok
        self.assertIn("git submodule update --init",
                      rep.summary_text())


class FinanceCommandTests(unittest.TestCase):
    def _call(self, tail):
        fake = SimpleNamespace(context=SimpleNamespace(db=None,
                                                       settings=None))
        fn = PartnerRuntime._control_finance.__get__(fake)
        return fn(tail, "chat1")

    def test_help(self):
        out = self._call("")
        self.assertIn("/finance quote", out)

    def test_unknown_verb(self):
        out = self._call("frobnicate")
        self.assertIn("unknown /finance verb", out)

    def test_quote(self):
        with patch.object(fe.FinancialExpert, "quote",
                          return_value={"symbol": "BTC", "price": 61234.5,
                                        "change_pct_24h": 2.35,
                                        "currency": "USDT",
                                        "source": "binance"}):
            out = self._call("quote BTC crypto")
        self.assertIn("61,234.5", out)
        self.assertIn("binance", out)

    def test_idea(self):
        idea = fe.TradeIdea(symbol="BTC", market="crypto",
                            profile="default", direction="long",
                            entry=61000.0, stop=59000.0,
                            target_1=63000.0, target_2=65000.0,
                            risk_reward=2.0, confidence=0.8,
                            rationale="test")
        with patch.object(fe.FinancialExpert, "trade_idea",
                          return_value=idea):
            out = self._call("idea BTC crypto --profile aggressive")
        self.assertIn("LONG", out)

    def test_watch(self):
        with patch.object(
                fe.FinancialExpert, "watch_price",
                return_value={"ok": True, "watcher": {"id": "w9"},
                              "echo": "will alert"}):
            out = self._call("watch BTC below 60000 crypto")
        self.assertIn("w9", out)
        self.assertIn("below", out)

    def test_watch_bad_syntax(self):
        out = self._call("watch BTC")
        self.assertIn("usage:", out)

    def test_doctor(self):
        rep = bridge.DoctorReport(ok=True, commit="abc1234",
                                  commit_matches=True, checks=[])
        with patch.object(bridge, "doctor", return_value=rep):
            out = self._call("doctor")
        self.assertIn("OK", out)

    def test_strategies_verb(self):
        out = self._call("strategies")
        self.assertIn("trend_follow", out)
        self.assertIn("mean_reversion", out)

    def test_backtest_strategy_flag(self):
        seen = {}

        def fake_backtest(self, symbol, market, strategy=None, profile="default"):
            seen.update(strategy=strategy, profile=profile)
            return fe.BacktestSummary(
                symbol=symbol, market=market, profile=profile,
                metrics={}, verdict="PASSES the bar")

        with patch.object(fe.FinancialExpert, "backtest", fake_backtest):
            out = self._call("backtest BTC crypto --strategy momentum "
                             "--profile aggressive")
        self.assertEqual(seen["strategy"], "momentum")
        self.assertEqual(seen["profile"], "aggressive")
        self.assertIn("PASSES the bar", out)

    def test_control_command_registered(self):
        from nomorals.social.chat.control import (COMMAND_DETAILS,
                                                  CONTROL_COMMANDS)
        self.assertIn("finance", CONTROL_COMMANDS)
        self.assertIn("finance", COMMAND_DETAILS)
        self.assertIn("/finance", COMMAND_DETAILS["finance"]["usage"])

    def test_provider_reexport(self):
        self.assertIs(fe.SentinelMarketProvider,
                      __import__("nomorals.integrations.market_data",
                                 fromlist=["SentinelMarketProvider"])
                      .SentinelMarketProvider)


if __name__ == "__main__":
    unittest.main()
