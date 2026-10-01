"""Acceptance tests for Prompt 07 — FinancialExpert over the native TA stack.

Covers the spec's contract: submodule-absent doctor guidance; plain-language
analysis and backtest verdicts on deterministic synthetic feeds (the native
nomorals.ta pipeline runs for real; only market_data.get_ohlcv is faked);
paper session start/status/stop with restart persistence; live gates
(LiveTradingDisabled, 24h unlock expiry, daily-loss auto-kill, 3-error
auto-kill, vault-only keys, kill switch); and the lazy-import requirement
(no pandas at module import).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from nomorals.accounts.vault import CredentialVault
from nomorals.agents.financial_expert import FinancialExpert
from nomorals.integrations import market_data
from nomorals.integrations import sentinel_bridge as bridge
from nomorals.storage.db import Database
from nomorals.ta.data import make_synthetic
from nomorals.tools import trading as trading_tool


# ── fakes ────────────────────────────────────────────────────────────────────

class _FakeRow(dict):
    def get(self, k, d=None):  # noqa: D102
        return super().get(k, d)


class _FakeRanked:
    def __init__(self, rows):
        self._rows = rows

    def head(self, n):
        return _FakeRanked(self._rows[:n])

    def iterrows(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


class _FakeReport:
    regime_label = "TREND"
    bias = 0.42
    agreement = 0.71
    position_now = 1.0
    approved = True
    meta_proba = 0.62
    size_fraction = 0.02
    stop_distance = 0.015
    n_strategies = 5
    bars = 600
    ranked = _FakeRanked([
        ("StratAlpha", _FakeRow(score=0.91, sharpe=1.8, hit_rate=0.62,
                                turnover=0.4)),
        ("StratBeta", _FakeRow(score=0.84, sharpe=1.5, hit_rate=0.58,
                               turnover=0.3)),
        ("StratGamma", _FakeRow(score=0.77, sharpe=1.2, hit_rate=0.55,
                                turnover=0.5)),
    ])


class _FakeRisk:
    halted = False


def _fake_metrics(sharpe=1.5, max_dd=0.10, profit_factor=1.6):
    return {"final_equity": 135000.0, "total_return": 0.35, "cagr": 0.28,
            "sharpe": sharpe, "sortino": 1.9, "max_dd": max_dd,
            "profit_factor": profit_factor, "win_rate": 0.58, "trades": 120,
            "turnover": 2.1, "exposure": 0.65}


class _FakeEngine:
    def __init__(self, metrics=None):
        self.risk = _FakeRisk()
        self._metrics = metrics or _fake_metrics()

    def scan(self, df, symbol="SYM"):
        return _FakeReport()

    def backtest(self, df):
        return dict(self._metrics)


class _FakeClient:
    """Fake exchange client: records market() calls, never touches network."""

    def __init__(self):
        self.calls = []
        self._ex = None

    def market(self, delta_units, price, tag=""):
        self.calls.append({"delta": delta_units, "price": price, "tag": tag})
        return {"filled": float(delta_units), "ccxt_id": "X1"}


class _FakeRegistry:
    def __init__(self, context):
        self.context = context
        self.tools = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


def _ctx(db=None, live_enabled=False):
    db = db or Database(":memory:")
    db.migrate()
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(
            trading=SimpleNamespace(
                live_enabled=live_enabled,
                max_daily_loss_pct=3.0,
                min_sharpe=1.0,
                max_drawdown_pct=20.0,
                min_profit_factor=1.3,
                unlock_ttl_hours=24.0,
                followed_symbols=[])),
        router=None)


def _grant_unlock(ctx, expired=False):
    now = time.time()
    exp = now - 10 if expired else now + 3600
    ctx.db.execute(
        "INSERT INTO live_unlocks (id, granted_at, expires_at, note)"
        " VALUES (?,?,?,?)", ("u1", now - 100, exp, "test"))


def _store_keys(ctx):
    os.environ["NM_VAULT_PASSPHRASE"] = "test-pass"
    vault = CredentialVault(ctx.db, master_passphrase="test-pass")
    vault.store("exchange:binance", "api", "KEY123",
                credential_type="api_key",
                metadata={"secret": "SECRET456"})


# ── tests ────────────────────────────────────────────────────────────────────

class LazyImportTests(unittest.TestCase):
    def test_no_heavy_imports_at_module_level(self):
        code = (
            "import sys;"
            "import nomorals.tools.trading;"
            "import nomorals.agents.financial_expert;"
            "import nomorals.integrations.sentinel_bridge;"
            "leaked=[m for m in ('pandas','numpy') if m in sys.modules];"
            "print('LEAKED:'+','.join(leaked) if leaked else 'CLEAN')"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=120,
                             cwd=str(Path(__file__).resolve().parents[1]))
        self.assertIn("CLEAN", out.stdout, out.stderr)


class BridgeTests(unittest.TestCase):
    def test_doctor_absent_submodule_explains_recovery(self):
        with patch.object(bridge, "VENDOR_ROOT", Path("/nonexistent-xyz")):
            rep = bridge.doctor()
        self.assertFalse(rep.ok)
        text = rep.summary_text()
        self.assertIn("git submodule update --init", text)

    def test_doctor_present_passes_on_synthetic(self):
        rep = bridge.doctor()
        self.assertTrue(rep.ok, rep.summary_text())
        self.assertTrue(rep.commit_matches)
        self.assertEqual(rep.commit, bridge.TESTED_COMMIT[:7])

    def test_unknown_market_and_profile_rejected(self):
        with self.assertRaises(bridge.SentinelError):
            bridge.get_engine("nope")
        with self.assertRaises(bridge.SentinelError):
            bridge.get_engine("crypto", "yolo")

    def test_list_strategies_nonempty(self):
        names = bridge.list_strategies()
        self.assertGreater(len(names), 50)

    def test_load_data_csv_missing(self):
        with self.assertRaises(bridge.SentinelError):
            bridge.load_data("/nonexistent/data.csv", "crypto")


class FinancialExpertTests(unittest.TestCase):
    """Native-stack contract: deterministic synthetic bars in, verdicts out.

    Only market_data.get_ohlcv is faked; the nomorals.ta pipeline
    (regime -> committee -> fusion -> risk -> backtest) runs for real.
    """

    def setUp(self):
        self.ctx = _ctx()
        self._p = patch.object(market_data, "get_ohlcv")
        self._mock_bars = self._p.start()
        self.addCleanup(self._p.stop)

    @staticmethod
    def _trend(n=600, drift=0.002, seed=0):
        """Deterministic uptrend the committee reads as LONG (default)."""
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, n)))
        idx = pd.date_range("2023-01-01", periods=n, freq="h")
        return pd.DataFrame(
            {"open": close, "high": close * 1.002, "low": close * 0.998,
             "close": close, "volume": 1000.0}, index=idx)

    def test_analyze_plain_language(self):
        self._mock_bars.return_value = self._trend()
        rep = FinancialExpert(self.ctx).analyze("BTC/USDT")
        self.assertEqual(rep.symbol, "BTC/USDT")
        text = rep.summary_text()
        self.assertIn("trending", text)          # plain regime words
        self.assertIn(rep.strategies[0]["name"], text)  # native zoo names
        self.assertNotIn("StratAlpha", text)     # old codegen zoo is gone
        self.assertIn("not financial advice", text)
        d = rep.to_dict()
        self.assertEqual(d["position_now"], 1.0)

    def test_backtest_verdict_passes(self):
        self._mock_bars.return_value = self._trend(n=1500)
        out = FinancialExpert(self.ctx).backtest("XAUUSD", "forex",
                                                 strategy="trend_follow")
        self.assertTrue(out.passes)
        self.assertIn("PASSES the bar", out.verdict)
        self.assertIn("sharpe ≥ 1.0", out.verdict)

    def test_backtest_verdict_fails(self):
        # choppy synthetic: trend_follow cannot clear the bar
        self._mock_bars.return_value = make_synthetic(1500, seed=7)
        out = FinancialExpert(self.ctx).backtest("XAUUSD", "forex",
                                                 strategy="trend_follow")
        self.assertFalse(out.passes)
        self.assertIn("FAILS the bar", out.verdict)

    def test_signal_bullish_with_invalidation(self):
        self._mock_bars.return_value = self._trend()
        sig = FinancialExpert(self.ctx).signal("ETH/USDT")
        self.assertEqual(sig.direction, "bullish")
        self.assertIn("adverse move", sig.invalidation)
        self.assertIn("not financial advice", sig.summary_text())

    def test_compare_table(self):
        self._mock_bars.return_value = self._trend()
        comp = FinancialExpert(self.ctx).compare(["BTC/USDT", "ETH/USDT"])
        self.assertEqual(len(comp.rows), 2)
        self.assertEqual(comp.rows[0]["stance"], "LONG")
        self.assertIn("BTC/USDT", comp.summary_text())


class PaperSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "t.db")
        self.ctx = _ctx(Database(self.db_path))

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_feed(self):
        df = bridge.make_synthetic_bars(600, seed=11)
        p1 = patch.object(bridge, "load_data", return_value=df)
        p2 = patch.object(bridge, "get_engine",
                          return_value=_FakeEngine())
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)

    def test_paper_lifecycle_survives_restart(self):
        self._fake_feed()
        started = trading_tool.paper_start(self.ctx, "BTC/USDT",
                                           capital=10000.0)
        sid = started["session_id"]
        status = trading_tool.paper_status(self.ctx, session_id=sid)
        self.assertGreater(status["equity"], 0)
        self.assertIn("regime", status["last_action"])

        # restart: brand-new Database handle on the same file
        ctx2 = _ctx(Database(self.db_path))
        with patch.object(bridge, "load_data",
                          return_value=bridge.make_synthetic_bars(600,
                                                                  seed=11)), \
             patch.object(bridge, "get_engine",
                          return_value=_FakeEngine()):
            status2 = trading_tool.paper_status(ctx2, session_id=sid)
        self.assertEqual(status2["session_id"], sid)
        self.assertGreater(status2["equity"], 0)

        stopped = trading_tool.paper_stop(self.ctx, session_id=sid)
        self.assertEqual(stopped["session_id"], sid)
        row = self.ctx.db.query_one(
            "SELECT status FROM paper_sessions WHERE id=?", (sid,))
        self.assertEqual(row["status"], "closed")
        j = self.ctx.db.query_one(
            "SELECT * FROM trade_journal WHERE kind='paper' AND side='stop'")
        self.assertIsNotNone(j)

    def test_paper_start_idempotent(self):
        self._fake_feed()
        a = trading_tool.paper_start(self.ctx, "BTC/USDT")
        b = trading_tool.paper_start(self.ctx, "BTC/USDT")
        self.assertEqual(a["session_id"], b["session_id"])
        self.assertTrue(b["resumed"])


class LiveGateTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx(live_enabled=False)
        self.client = _FakeClient()
        self._p = patch.object(trading_tool, "_EXCHANGE_CLIENT_FACTORY",
                               return_value=self.client)
        self._p.start()
        self.addCleanup(self._p.stop)
        self.addCleanup(trading_tool.set_exchange_client_factory, None)
        os.environ["NM_VAULT_PASSPHRASE"] = "test-pass"

    def test_live_disabled_raises_and_never_touches_exchange(self):
        _store_keys(self.ctx)
        with self.assertRaises(bridge.LiveTradingDisabled):
            trading_tool.live_order(self.ctx, "BTC/USDT", "buy", 0.01)
        self.assertEqual(self.client.calls, [])

    def test_unlock_requires_exact_confirm(self):
        ctx = _ctx(live_enabled=True)
        with self.assertRaises(bridge.SentinelError):
            trading_tool.live_unlock(ctx, confirm="yes please")

    def test_unlock_disabled_in_settings(self):
        with self.assertRaises(bridge.LiveTradingDisabled):
            trading_tool.live_unlock(self.ctx, confirm="I understand")

    def test_expired_unlock_rejected(self):
        ctx = _ctx(live_enabled=True)
        _store_keys(ctx)
        _grant_unlock(ctx, expired=True)
        with self.assertRaises(bridge.LiveTradingDisabled) as cm:
            trading_tool.live_order(ctx, "BTC/USDT", "buy", 0.01)
        self.assertIn("unlock", str(cm.exception).lower())
        self.assertEqual(self.client.calls, [])

    def test_live_order_success_journaled(self):
        ctx = _ctx(live_enabled=True)
        _store_keys(ctx)
        _grant_unlock(ctx)
        out = trading_tool.live_order(ctx, "BTC/USDT", "buy", 0.01)
        self.assertTrue(out["ok"])
        self.assertEqual(len(self.client.calls), 1)
        self.assertAlmostEqual(self.client.calls[0]["delta"], 0.01)
        j = ctx.db.query_one(
            "SELECT * FROM trade_journal WHERE kind='live' AND side='buy'")
        self.assertIsNotNone(j)
        self.assertEqual(j["symbol"], "BTC/USDT")

    def test_keys_come_from_vault_only(self):
        ctx = _ctx(live_enabled=True)
        _grant_unlock(ctx)
        # no keys stored in the vault -> clear error, zero exchange calls
        with self.assertRaises(bridge.SentinelError) as cm:
            trading_tool.live_order(ctx, "BTC/USDT", "buy", 0.01)
        self.assertIn("vault", str(cm.exception).lower())
        self.assertEqual(self.client.calls, [])

    def test_daily_loss_auto_kill(self):
        ctx = _ctx(live_enabled=True)
        _store_keys(ctx)
        _grant_unlock(ctx)
        day_start = time.time() - (time.time() % 86400)
        for ts, eq in ((day_start - 3600, 100000.0),
                       (day_start + 3600, 96000.0)):  # -4% today
            ctx.db.execute(
                "INSERT INTO trade_journal (id, ts, kind, symbol, side,"
                " size, price, order_id, reason, meta_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"j{ts}", ts, "live", "BTC/USDT", "sell", 1, 50000,
                 "o1", "fill", json.dumps({"equity_after": eq})))
        with self.assertRaises(bridge.LiveTradingDisabled) as cm:
            trading_tool.live_order(ctx, "BTC/USDT", "buy", 0.01)
        self.assertIn("loss limit", str(cm.exception).lower())
        self.assertEqual(self.client.calls, [])
        # unlock revoked, kill journaled
        self.assertIsNone(trading_tool._valid_unlock(ctx.db))
        kill = ctx.db.query_one(
            "SELECT * FROM trade_journal WHERE side='kill'")
        self.assertIsNotNone(kill)

    def test_three_errors_auto_kill(self):
        ctx = _ctx(live_enabled=True)
        _store_keys(ctx)
        _grant_unlock(ctx)

        def boom(symbol, exchange, api_key, secret):
            raise RuntimeError("exchange exploded")

        with patch.object(trading_tool, "_EXCHANGE_CLIENT_FACTORY", boom):
            for _ in range(3):
                with self.assertRaises(bridge.SentinelError):
                    trading_tool.live_order(ctx, "BTC/USDT", "buy", 0.01)
        self.assertIsNone(trading_tool._valid_unlock(ctx.db))
        kill = ctx.db.query_one(
            "SELECT * FROM trade_journal WHERE side='kill'")
        self.assertIsNotNone(kill)
        self.assertIn("consecutive", kill["reason"])

    def test_kill_switch(self):
        ctx = _ctx(live_enabled=True)
        _grant_unlock(ctx)
        out = trading_tool.kill_switch(ctx, "test kill")
        self.assertTrue(out["ok"])
        self.assertIsNone(trading_tool._valid_unlock(ctx.db))
        kill = ctx.db.query_one(
            "SELECT * FROM trade_journal WHERE side='kill'")
        self.assertIsNotNone(kill)
        # live order now fails on the missing unlock
        _store_keys(ctx)
        with self.assertRaises(bridge.LiveTradingDisabled):
            trading_tool.live_order(ctx, "BTC/USDT", "buy", 0.01)


class ToolRegistrationTests(unittest.TestCase):
    def test_trading_tool_registers(self):
        ctx = _ctx()
        reg = _FakeRegistry(ctx)
        trading_tool.register(reg)
        self.assertIn("trading", reg.tools)
        fn = reg.tools["trading"]
        with patch.object(bridge, "VENDOR_ROOT", Path("/nonexistent-xyz")):
            out = fn(action="doctor")
        self.assertFalse(out["ok"])

    def test_tool_unknown_action(self):
        ctx = _ctx()
        reg = _FakeRegistry(ctx)
        trading_tool.register(reg)
        with self.assertRaises(bridge.SentinelError):
            reg.tools["trading"](action="moon")

    def test_tool_strategies(self):
        ctx = _ctx()
        reg = _FakeRegistry(ctx)
        trading_tool.register(reg)
        out = reg.tools["trading"](action="strategies")
        self.assertGreater(out["count"], 50)


class CliTests(unittest.TestCase):
    def test_cmd_trade_doctor_json(self):
        from nomorals.cli import _cmd_trade
        ctx = _ctx()
        args = SimpleNamespace(trade_action="doctor", json=True)
        self.assertEqual(_cmd_trade(args, ctx), 0)

    def test_cmd_trade_kill(self):
        from nomorals.cli import _cmd_trade
        ctx = _ctx(live_enabled=True)
        _grant_unlock(ctx)
        args = SimpleNamespace(trade_action="kill", json=True)
        self.assertEqual(_cmd_trade(args, ctx), 0)
        self.assertIsNone(trading_tool._valid_unlock(ctx.db))

    def test_cmd_trade_live_disabled_exit_code(self):
        from nomorals.cli import _cmd_trade
        ctx = _ctx(live_enabled=False)
        args = SimpleNamespace(trade_action="live", live_action="order",
                               symbol="BTC/USDT", side="buy", size=0.01,
                               exchange="binance", json=False)
        self.assertEqual(_cmd_trade(args, ctx), 3)


if __name__ == "__main__":
    unittest.main()
