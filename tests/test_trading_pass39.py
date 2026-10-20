"""Pass #39 — trading wiring defects.

Each test pins a defect reproduced on pristine HEAD a0fb7dd0:

A. NM_TRADING_ENABLED / NM_TRADING_ALLOW_LIVE / NM_TRADING_MODE were
   silently ignored (no settings fields, no env mapping) — the master
   switch the trading tool's own refusal messages cite did nothing.
B. The CCXT daily-loss kill switch was dead twice over: no production
   code ever wrote meta.equity_after (the gate's only input), and the
   reader took the OLDEST equity in its lookback window as the day's
   baseline instead of the last fill before today.
C. A failed live_order left no journal trace — an order the exchange
   may have accepted was invisible, and a retry could re-submit blind.
E. Exness checkpoint resume dropped close_all_positions' profitable
   filter and close_position's price — the resumed operation was wider
   than the payload the owner approved.
F. exness_size swallowed an explicit risk_pct=0 (`or` defaulting) and
   sized at 1% instead of refusing.
G. The Exness desk's live money actions ignored
   settings.trading.live_enabled entirely (the CCXT live_unlock in the
   same module refuses on exactly that gate).
"""

from __future__ import annotations

import base64
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.exness import ExnessConnector
from nomorals.core.config import load_settings
from nomorals.finance.trading_desk import size_position
from nomorals.integrations import sentinel_bridge as bridge
from nomorals.storage.db import Database
from nomorals.tools import trading as trading_tool


# ── helpers ──────────────────────────────────────────────────────────

def _ctx(db=None, *, enabled=True, live_enabled=False, mode="paper"):
    db = db or Database(":memory:")
    db.migrate()
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(trading=SimpleNamespace(
            enabled=enabled, live_enabled=live_enabled, mode=mode,
            max_daily_loss_pct=3.0, unlock_ttl_hours=24.0)))


def _store_keys(ctx, passphrase="pw"):
    CredentialVault(ctx.db, master_passphrase=passphrase).store(
        "exchange:binance", "api", "KEY123",
        metadata={"secret": "SECRET123"})


def _grant_unlock(ctx):
    now = time.time()
    ctx.db.execute(
        "INSERT INTO live_unlocks (id, granted_at, expires_at, note)"
        " VALUES (?,?,?,?)", ("u1", now, now + 3600, "test"))


def _seed_fill(db, ts, equity_after):
    db.execute(
        "INSERT INTO trade_journal (id, ts, kind, symbol, side, size,"
        " price, order_id, reason, meta_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (f"j{ts}", ts, "live", "BTC/USDT", "sell", 1, 50000, "o1", "fill",
         json.dumps({"equity_after": equity_after})))


class _FakeRegistry:
    def __init__(self, context):
        self.context = context
        self.tools = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


# ── A: env wiring ────────────────────────────────────────────────────

class TradingEnvWiringTests(unittest.TestCase):
    def test_master_switch_env_takes_effect(self):
        s = load_settings(env={"NM_TRADING_ENABLED": "0"},
                          use_env_file=False)
        self.assertFalse(s.trading.enabled)
        s = load_settings(env={}, use_env_file=False)
        self.assertTrue(s.trading.enabled)

    def test_allow_live_env_takes_effect(self):
        s = load_settings(env={"NM_TRADING_ALLOW_LIVE": "1"},
                          use_env_file=False)
        self.assertTrue(s.trading.live_enabled)

    def test_mode_env_takes_effect_and_defaults_paper(self):
        s = load_settings(env={"NM_TRADING_MODE": "live"},
                          use_env_file=False)
        self.assertEqual(s.trading.mode, "live")
        s = load_settings(env={}, use_env_file=False)
        self.assertEqual(s.trading.mode, "paper")

    def test_master_switch_refuses_tool_actions(self):
        ctx = _ctx(enabled=False)
        reg = _FakeRegistry(ctx)
        trading_tool.register(reg)
        out = reg.tools["trading"](action="paper_start",
                                   symbol="BTC/USDT")
        self.assertFalse(out["ok"])
        self.assertIn("disabled", out["error"])
        # kill (safety) still works while disabled
        out = reg.tools["trading"](action="kill", reason="test")
        self.assertTrue(out["ok"])

    def test_settings_mode_is_the_desk_default(self):
        seen = {}

        class StubDesk:
            def account(self):
                return {"account_state": {"balance": 1, "equity": 1,
                                          "currency": "USD"},
                        "positions": [], "orders": []}

        def fake_desk(context, mode="paper"):
            seen["mode"] = mode
            return StubDesk()

        ctx = _ctx(mode="live")
        with mock.patch.object(trading_tool, "_exness_desk", fake_desk):
            out = trading_tool.exness_action(ctx, "exness_snapshot")
        self.assertTrue(out["ok"])
        self.assertEqual(seen["mode"], "live")


# ── B: daily-loss gate ───────────────────────────────────────────────

class DailyLossGateTests(unittest.TestCase):
    def test_baseline_is_last_fill_before_today(self):
        db = Database(":memory:")
        db.migrate()
        day_start = time.time() - (time.time() % 86400)
        _seed_fill(db, day_start - 2 * 86400, 50000.0)   # 2 days ago
        _seed_fill(db, day_start - 3600, 100000.0)       # last before today
        _seed_fill(db, day_start + 3600, 96000.0)        # today: -4%
        self.assertAlmostEqual(trading_tool._daily_live_loss_pct(db),
                               0.04, places=6)

    def test_live_order_writes_equity_after(self):
        ctx = _ctx(live_enabled=True)
        _store_keys(ctx)
        _grant_unlock(ctx)

        class FakeEx:
            def fetch_ticker(self, symbol):
                return {"last": 50000.0}

            def fetch_balance(self):
                return {"total": {"USDT": 90000.0}}

        class FakeClient:
            _ex = FakeEx()

            def market(self, delta, price, tag=""):
                return {"filled": float(delta), "ccxt_id": "X1"}

        with mock.patch.dict(os.environ, {"NM_VAULT_PASSPHRASE": "pw"}), \
                mock.patch.object(trading_tool, "_EXCHANGE_CLIENT_FACTORY",
                                  lambda *a: FakeClient()):
            out = trading_tool.live_order(ctx, "BTC/USDT", "buy", 0.01)
        self.assertTrue(out["ok"])
        fill = ctx.db.query_one(
            "SELECT meta_json FROM trade_journal"
            " WHERE reason='manual live order'")
        meta = json.loads(fill["meta_json"])
        self.assertEqual(meta.get("equity_after"), 90000.0)

    def test_gate_trips_on_equities_recorded_by_real_fills(self):
        # Yesterday's fill at 100k, today's at 95.9k (-4.1% > 3% limit):
        # the next live_order must auto-kill before touching the exchange.
        ctx = _ctx(live_enabled=True)
        _store_keys(ctx)
        _grant_unlock(ctx)
        day_start = time.time() - (time.time() % 86400)
        _seed_fill(ctx.db, day_start - 3600, 100000.0)
        _seed_fill(ctx.db, day_start + 60, 95900.0)
        with mock.patch.dict(os.environ, {"NM_VAULT_PASSPHRASE": "pw"}):
            with self.assertRaises(bridge.LiveTradingDisabled) as cm:
                trading_tool.live_order(ctx, "BTC/USDT", "buy", 0.01)
        self.assertIn("loss limit", str(cm.exception).lower())
        self.assertIsNone(trading_tool._valid_unlock(ctx.db))


# ── C: submission journaling ─────────────────────────────────────────

class LiveOrderJournalTests(unittest.TestCase):
    def test_attempt_is_journaled_before_the_exchange_call(self):
        ctx = _ctx(live_enabled=True)
        _store_keys(ctx)
        _grant_unlock(ctx)

        class BoomClient:
            _ex = None

            def market(self, delta, price, tag=""):
                raise RuntimeError("network died after submit")

        with mock.patch.dict(os.environ, {"NM_VAULT_PASSPHRASE": "pw"}), \
                mock.patch.object(trading_tool, "_EXCHANGE_CLIENT_FACTORY",
                                  lambda *a: BoomClient()):
            with self.assertRaises(bridge.SentinelError):
                trading_tool.live_order(ctx, "BTC/USDT", "buy", 0.01)
        rows = ctx.db.query(
            "SELECT reason FROM trade_journal WHERE kind='live'")
        self.assertEqual(len(rows), 1)
        self.assertIn("outcome pending", rows[0]["reason"])

    def test_success_leaves_attempt_then_fill(self):
        ctx = _ctx(live_enabled=True)
        _store_keys(ctx)
        _grant_unlock(ctx)

        class OkClient:
            _ex = None

            def market(self, delta, price, tag=""):
                return {"filled": float(delta), "ccxt_id": "X9"}

        with mock.patch.dict(os.environ, {"NM_VAULT_PASSPHRASE": "pw"}), \
                mock.patch.object(trading_tool, "_EXCHANGE_CLIENT_FACTORY",
                                  lambda *a: OkClient()):
            trading_tool.live_order(ctx, "BTC/USDT", "sell", 0.02)
        reasons = [r["reason"] for r in ctx.db.query(
            "SELECT reason FROM trade_journal WHERE kind='live'"
            " ORDER BY ts, rowid")]
        self.assertEqual(len(reasons), 2)
        self.assertIn("outcome pending", reasons[0])
        self.assertEqual(reasons[1], "manual live order")


# ── E: checkpoint resume fidelity ────────────────────────────────────

class _FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status = status
        self._payload = payload
        self.headers = headers or {}
        self.url = ""

    @property
    def ok(self):
        return 200 <= self.status < 300

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self):
        self.calls = []
        self.routes = []

    def route(self, method, path, response):
        self.routes.append((method.upper(), path, response))

    def request(self, method, url, data=None, headers=None, **kw):
        self.calls.append((method.upper(), url, data, headers))
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return _FakeResponse(404, {"error_message": "not mocked",
                                   "code": 3})


_SEED_B64URL = base64.urlsafe_b64encode(
    bytes(range(32))).rstrip(b"=").decode()


def _connected_conn():
    http = _FakeHttp()
    vault = CredentialVault(Database(":memory:"),
                            master_passphrase="test")
    conn = ExnessConnector(vault, http=http)
    http.route("GET", "/v1/configuration/accounts/1042/account",
               _FakeResponse(200, {"account_id": "1042",
                                   "settings": {"currency": "USD"}}))
    conn.connect(api_key="<redacted>", private_key=_SEED_B64URL,
                 account_id="1042")
    http.route("GET", "/v1/trading/access-point",
               _FakeResponse(200,
                             {"access_point": "https://trade.exness.test"}))
    http.route("DELETE", "/v1/trading/accounts/1042/positions",
               _FakeResponse(200, {"operation_id": "op1"}))
    http.route("DELETE", "/v1/trading/accounts/1042/positions/P1",
               _FakeResponse(200, {"operation_id": "op2"}))
    return conn, http


class ResumeFidelityTests(unittest.TestCase):
    def test_resume_close_all_keeps_profitable_filter(self):
        conn, http = _connected_conn()
        conn.close_all_positions(profitable=False, confirmed=True)
        direct = http.calls[-1][1]
        cp = SimpleNamespace(resume_state={
            "op": "close_all_positions", "stage": "close_all_positions",
            "params": {"filter": "all", "instrument": "", "side": "",
                       "profitable": False, "unprofitable": True}})
        conn.resume_checkpoint(cp, db=object())
        resumed = http.calls[-1][1]
        self.assertEqual(direct, resumed)
        self.assertIn("profitable=False", resumed)

    def test_resume_close_position_keeps_price(self):
        conn, http = _connected_conn()
        conn.close_position("P1", price=1999.5, confirmed=True)
        direct = http.calls[-1][1]
        cp = SimpleNamespace(resume_state={
            "op": "close_position", "stage": "close_position",
            "position_id": "P1",
            "params": {"volume": None, "price": "1999.5"}})
        conn.resume_checkpoint(cp, db=object())
        resumed = http.calls[-1][1]
        self.assertEqual(direct, resumed)
        self.assertIn("price=1999.5", resumed)


# ── F + G: desk action wiring ────────────────────────────────────────

class _StubDesk:
    def __init__(self, demo=False):
        self._demo = demo

    def _is_demo_account(self):
        return self._demo

    def equity(self):
        return 10000.0

    def contract_specs(self, instrument):
        return {"volume_min": 0.01, "volume_max": 100.0,
                "volume_step": 0.01}

    def _contract_value(self, instrument):
        return 100.0

    def live_open(self, *a, **kw):
        return {"order": "placed"}

    def unlock_live(self, confirm=""):
        return {"ok": True, "mode": "live"}


class DeskActionWiringTests(unittest.TestCase):
    def test_explicit_zero_risk_pct_is_refused_not_defaulted(self):
        ctx = _ctx()
        with mock.patch.object(trading_tool, "_exness_desk",
                               lambda c, mode="paper": _StubDesk()):
            out = trading_tool.exness_action(
                ctx, "exness_size", symbol="XAUUSD", entry=2000.0,
                stop_loss=1990.0, risk_pct=0)
        self.assertFalse(out["ok"])
        self.assertIn("risk_pct", out["error"])

    def test_live_open_refused_while_live_disabled(self):
        ctx = _ctx(live_enabled=False)
        with mock.patch.object(trading_tool, "_exness_desk",
                               lambda c, mode="paper": _StubDesk()):
            out = trading_tool.exness_action(
                ctx, "exness_open", mode="live", symbol="XAUUSD",
                side="buy", volume=0.1, stop_loss=1990.0, confirmed=True)
        self.assertFalse(out["ok"])
        self.assertIn("live_enabled", out["error"])

    def test_live_open_allowed_when_live_enabled(self):
        ctx = _ctx(live_enabled=True)
        with mock.patch.object(trading_tool, "_exness_desk",
                               lambda c, mode="paper": _StubDesk()):
            out = trading_tool.exness_action(
                ctx, "exness_open", mode="live", symbol="XAUUSD",
                side="buy", volume=0.1, stop_loss=1990.0, confirmed=True)
        self.assertTrue(out["ok"])

    def test_live_open_allowed_on_demo_while_live_disabled(self):
        ctx = _ctx(live_enabled=False)
        with mock.patch.object(
                trading_tool, "_exness_desk",
                lambda c, mode="paper": _StubDesk(demo=True)):
            out = trading_tool.exness_action(
                ctx, "exness_open", mode="live", symbol="XAUUSD",
                side="buy", volume=0.1, stop_loss=1990.0, confirmed=True)
        self.assertTrue(out["ok"])

    def test_unlock_refused_while_live_disabled(self):
        ctx = _ctx(live_enabled=False)
        with mock.patch.object(trading_tool, "_exness_desk",
                               lambda c, mode="paper": _StubDesk()):
            out = trading_tool.exness_action(
                ctx, "exness_unlock", confirm="I understand the risks")
        self.assertFalse(out["ok"])
        self.assertIn("live_enabled", out["error"])


# ── H: finance tools' timeframe kwarg ────────────────────────────────

class FinanceToolTimeframeTests(unittest.TestCase):
    """finance_signal / finance_backtest passed timeframe= into expert
    methods that did not accept it — both tools died with TypeError on
    every call."""

    def _tools(self):
        from nomorals.tools import finance as finance_tools

        class Reg:
            def __init__(self):
                self.context = None
                self.tools = {}

            def register(self, name, **kw):
                def deco(fn):
                    self.tools[name] = fn
                    return fn
                return deco

        reg = Reg()
        finance_tools.register(reg)
        return reg.tools

    def _expert_ctx(self):
        return SimpleNamespace(
            db=Database(":memory:"),
            settings=SimpleNamespace(trading=SimpleNamespace(
                live_enabled=False, max_daily_loss_pct=3.0,
                min_sharpe=1.0, max_drawdown_pct=20.0,
                min_profit_factor=1.3, unlock_ttl_hours=24.0,
                followed_symbols=[])),
            router=None)

    def test_finance_signal_accepts_timeframe(self):
        from nomorals.agents.financial_expert import FinancialExpert
        from nomorals.ta.data import make_synthetic
        df = make_synthetic(700, seed=3)
        tools = self._tools()
        with mock.patch.object(FinancialExpert, "_load_bars",
                               lambda self, s, m, tf, b: df):
            out = tools["finance_signal"](self._expert_ctx(), "BTC/USDT",
                                          timeframe="4h")
        self.assertIn("direction", out)

    def test_finance_backtest_accepts_timeframe(self):
        from nomorals.agents.financial_expert import FinancialExpert
        from nomorals.ta.data import make_synthetic
        df = make_synthetic(1500, seed=5)
        tools = self._tools()
        with mock.patch.object(FinancialExpert, "_load_bars",
                               lambda self, s, m, tf, b: df):
            out = tools["finance_backtest"](self._expert_ctx(), "XAUUSD",
                                            timeframe="4h",
                                            strategy="trend_follow")
        self.assertIn("verdict", out)


# ── sizing round-trip (priority question 3) ──────────────────────────

class SizePositionRoundTripTests(unittest.TestCase):
    def test_sized_volume_never_exceeds_intended_risk(self):
        cases = [
            # equity, entry, stop, risk_pct, contract_value
            (10000.0, 2000.0, 1990.0, 1.0, 100.0),
            (120000.0, 4150.5, 4132.7, 0.5, 100.0),
            (250000.0, 1.0850, 1.0800, 2.0, 100000.0),
            (500.0, 67500.0, 67000.0, 1.0, 1.0),
        ]
        for equity, entry, sl, pct, cv in cases:
            out = size_position(equity=equity, entry=entry, stop_loss=sl,
                                risk_pct=pct, contract_value=cv)
            intended = equity * pct / 100.0
            self.assertLessEqual(out["actual_risk"], intended + 1e-6)
            # volume is an exact multiple of the 0.01 step
            self.assertAlmostEqual(out["volume"] / 0.01,
                                   round(out["volume"] / 0.01), places=6)


if __name__ == "__main__":
    unittest.main()
