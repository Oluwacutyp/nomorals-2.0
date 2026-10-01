"""Tests for the deepened FinancialExpert: trade ideas, price-alert wiring,
the keyless-first bridge routing, and the /finance chat command.

Engine and HTTP layers are faked — no network, no submodule needed for
most of these (bridge._ensure_path is patched where load_data is hit).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import financial_expert as fe
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.integrations import sentinel_bridge as bridge


class _FakeReport:
    regime_label = "TREND-UP"
    bias = 0.5
    agreement = 0.8
    position_now = 1.0
    approved = True
    meta_proba = 0.7
    size_fraction = 0.02
    stop_distance = 0.02
    stops = {"stop": 59000.0, "breakeven_trigger": 61200.0,
             "target_1": 63000.0, "target_2": 65000.0}
    n_strategies = 5
    bars = 600


class _FakeEngine:
    def scan(self, df, symbol="SYM"):
        return _FakeReport()


class _FakeDF(dict):
    """Pretends to be a pandas frame with a close column."""

    def __init__(self, px):
        super().__init__()
        self._px = px

    def __getitem__(self, key):
        assert key == "close"
        return SimpleNamespace(iloc=[self._px])


def _expert():
    return fe.FinancialExpert(SimpleNamespace(db=None, settings=None))


class TradeIdeaTests(unittest.TestCase):
    def _run(self, pos=1.0):
        _FakeReport.position_now = pos
        try:
            with patch.object(bridge, "get_engine",
                              return_value=_FakeEngine()), \
                 patch.object(bridge, "load_data",
                              return_value=_FakeDF(61000.0)):
                return _expert().trade_idea("BTC", "crypto", "conservative")
        finally:
            _FakeReport.position_now = 1.0

    def test_long_idea_math(self):
        idea = self._run(pos=1.0)
        self.assertEqual(idea.direction, "long")
        self.assertEqual(idea.profile, "conservative")
        self.assertAlmostEqual(idea.entry, 61000.0)
        self.assertAlmostEqual(idea.stop, 59000.0)
        # risk 2000, reward 4000 → 1:2
        self.assertAlmostEqual(idea.risk_reward, 2.0)
        self.assertGreater(idea.confidence, 0)
        text = idea.summary_text()
        self.assertIn("LONG", text)
        self.assertIn("not financial advice", text)

    def test_flat_idea(self):
        idea = self._run(pos=0.0)
        self.assertEqual(idea.direction, "flat")
        self.assertIn("FLAT", idea.summary_text())

    def test_bad_profile_rejected(self):
        with self.assertRaises(bridge.SentinelError):
            _expert().trade_idea("BTC", "crypto", "yolo")


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
