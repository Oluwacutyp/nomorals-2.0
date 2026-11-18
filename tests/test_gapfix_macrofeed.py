"""Lane 75 (failing-first): live macro feed + journal equity into FinancialExpert.

Gap (register §8 L2005–2012): no live macro feed wired into
FinancialExpert (DX-Y.NYB / ^TNX) and no equity feed from the trading
journal into the pipeline — ``ta/pipeline.py::analyze`` already accepts
both (``macro=``, ``equity=``) but FinancialExpert passes neither, and
the only macro fetcher (``TrinityBias``) needs ``yfinance``, which is
not installed and not a declared dependency.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest import mock

import pytest

FAKE_TS = 1_780_000_000.0


def _fake_closes():
    """120 daily closes per symbol: DXY + US10Y both rising → bearish."""
    return {
        "GC=F": [2000.0 + i * 0.5 for i in range(120)],
        "DX-Y.NYB": [100.0 + i * 0.05 for i in range(120)],
        "^TNX": [4.0 + i * 0.005 for i in range(120)],
    }


def _fake_fetcher(closes=None):
    data = closes if closes is not None else _fake_closes()

    def fetch(symbol):
        return list(data[symbol]), FAKE_TS, "test-feed"

    return fetch


def _failing_fetcher(symbol):
    raise RuntimeError(f"test-feed down for {symbol}")


def _partial_fetcher(symbol):
    if symbol == "DX-Y.NYB":
        raise RuntimeError("test-feed down for DX-Y.NYB")
    return list(_fake_closes()[symbol]), FAKE_TS, "test-feed"


def _seed_equity(tmp_path, values):
    from nomorals.finance import trading_journal

    settings = SimpleNamespace(home_path=str(tmp_path))
    for i, value in enumerate(values):
        trading_journal.append_event(
            "equity_observation", settings=settings,
            equity=float(value), ts=FAKE_TS + i * 60.0)
    return settings


def _expert(tmp_path, fetcher=None):
    from nomorals.agents.financial_expert import FinancialExpert

    settings = SimpleNamespace(home_path=str(tmp_path))
    ctx = SimpleNamespace(db=None, settings=settings,
                          rooms=None, current_room=None)
    return FinancialExpert(ctx, macro_fetcher=fetcher)


def _synthetic_bars(n=600):
    from nomorals.ta.data import make_synthetic

    return make_synthetic(n)


# ── the keyless chain must keep Yahoo symbols intact ──────────────────

def test_normalize_keeps_yahoo_symbols():
    from nomorals.integrations import market_data as md

    assert md.normalize_symbol("DX-Y.NYB", "stocks")["base"] == "DX-Y.NYB"
    assert md.normalize_symbol("^TNX", "stocks")["base"] == "^TNX"
    assert md.normalize_symbol("GC=F", "stocks")["base"] == "GC=F"
    # Pairs still split exactly as before.
    assert md.normalize_symbol("BTC-USD", "crypto")["base"] == "BTC"
    assert md.normalize_symbol("BTC-USD", "crypto")["quote"] == "USD"
    assert md.normalize_symbol("EURUSD", "forex")["base"] == "EUR"
    assert md.normalize_symbol("AAPL", "stocks")["base"] == "AAPL"


def test_yahoo_symbol_for_macro_symbols():
    from nomorals.integrations import market_data as md

    for sym in ("DX-Y.NYB", "^TNX", "GC=F"):
        norm = md.normalize_symbol(sym, "stocks")
        assert md._yahoo_symbol(norm, "stocks") == sym


# ── TrinityBias with an injected fetcher ─────────────────────────────

def test_trinity_bias_injected_fetcher_live():
    from nomorals.ta.macro import TrinityBias, trinity_bias

    closes = _fake_closes()
    provider = TrinityBias(fetch_closes=_fake_fetcher(closes))
    provider.refresh()
    bias = provider.get()
    assert bias is not None
    assert bias == trinity_bias(closes["GC=F"], closes["DX-Y.NYB"],
                                closes["^TNX"]) | {"_ts": bias["_ts"],
                                                   "readings": bias["readings"]}
    assert bias["bias"] == "bearish"
    readings = provider.readings()
    assert readings is not None
    assert readings["dxy"]["close"] == closes["DX-Y.NYB"][-1]
    assert readings["dxy"]["ts"] == FAKE_TS
    assert readings["dxy"]["source"] == "test-feed"
    assert readings["tny"]["close"] == closes["^TNX"][-1]
    assert readings["xau"]["close"] == closes["GC=F"][-1]


def test_trinity_bias_fetch_failure_honest():
    from nomorals.ta.macro import TrinityBias

    provider = TrinityBias(fetch_closes=_failing_fetcher)
    provider.refresh()
    assert provider.get() is None
    # A failed fetch still records its readings — every leg named.
    readings = provider.readings()
    assert readings is not None
    assert all(r["status"] == "error" for r in readings.values())
    assert "GC=F" in readings["xau"]["error"]
    assert provider.last_error is not None
    assert "GC=F" in provider.last_error


def test_trinity_bias_partial_failure_names_the_symbol():
    from nomorals.ta.macro import TrinityBias

    provider = TrinityBias(fetch_closes=_partial_fetcher)
    provider.refresh()
    assert provider.get() is None  # no bias without all three legs
    readings = provider.readings()
    assert readings is not None
    assert readings["dxy"]["status"] == "error"
    assert "DX-Y.NYB" in readings["dxy"]["error"]
    assert readings["tny"]["status"] == "ok"
    assert readings["xau"]["status"] == "ok"


def test_trinity_bias_stale_after_max_age():
    from nomorals.ta.macro import TrinityBias

    provider = TrinityBias(max_age_hours=24.0,
                           fetch_closes=_fake_fetcher())
    provider.refresh()
    assert provider.get() is not None
    provider._cache["_ts"] = time.time() - 25 * 3600
    provider._readings_ts = time.time() - 25 * 3600
    assert provider.get() is None
    assert provider.readings() is None


# ── journal equity curve ──────────────────────────────────────────────

def test_equity_curve_from_journal(tmp_path):
    from nomorals.finance import trading_journal

    settings = _seed_equity(tmp_path, [10000.0, 10100.0, 9900.0])
    curve = trading_journal.equity_curve(settings=settings)
    assert curve["points"] == [10000.0, 10100.0, 9900.0]
    assert curve["last"] == 9900.0
    assert curve["count"] == 3
    assert curve["last_ts"] == FAKE_TS + 120.0


def test_equity_curve_missing_file_empty(tmp_path):
    from nomorals.finance import trading_journal

    settings = SimpleNamespace(home_path=str(tmp_path))
    curve = trading_journal.equity_curve(settings=settings)
    assert curve["points"] == []
    assert curve["last"] is None
    assert curve["count"] == 0


# ── FinancialExpert wiring ────────────────────────────────────────────

def test_expert_analyze_gets_macro_and_equity(tmp_path):
    _seed_equity(tmp_path, [10000.0, 10100.0, 10200.0])
    expert = _expert(tmp_path, fetcher=_fake_fetcher())
    expert._load_bars = lambda *a, **k: _synthetic_bars()
    report = expert.analyze("XAUUSD", market="forex", timeframe="1h",
                            bars=600)
    assert report.macro["status"] == "live"
    assert report.macro["bias"]["bias"] == "bearish"
    assert report.macro["readings"]["dxy"]["close"] == pytest.approx(105.95)
    assert report.macro["readings"]["tny"]["close"] == pytest.approx(4.595)
    assert report.equity["points"] == [10000.0, 10100.0, 10200.0]
    assert report.equity["last"] == 10200.0
    assert report.adaptive["macro"] is not None
    assert report.adaptive["throttle_factor"] == 1.0
    text = report.summary_text()
    assert "DXY" in text and "US10Y" in text
    assert "105.95" in text


def test_expert_analyze_drawdown_throttles(tmp_path):
    _seed_equity(tmp_path, [10000.0, 11000.0, 9900.0])
    expert = _expert(tmp_path, fetcher=_fake_fetcher())
    expert._load_bars = lambda *a, **k: _synthetic_bars()
    report = expert.analyze("XAUUSD", market="forex", timeframe="1h",
                            bars=600)
    assert 0.0 < report.adaptive["throttle_factor"] < 1.0
    assert report.adaptive["drawdown"] == pytest.approx(0.1)
    assert "throttle" in report.summary_text().lower()


def test_expert_analyze_macro_unavailable_honest(tmp_path):
    expert = _expert(tmp_path, fetcher=_failing_fetcher)
    expert._load_bars = lambda *a, **k: _synthetic_bars()
    report = expert.analyze("XAUUSD", market="forex", timeframe="1h",
                            bars=600)
    assert report.macro["status"] == "unavailable"
    assert "GC=F" in report.macro["reason"]
    assert report.equity["points"] == []
    assert "macro" in report.summary_text().lower()
    assert "unavailable" in report.summary_text().lower()


def test_expert_macro_report_text(tmp_path):
    _seed_equity(tmp_path, [10000.0, 10100.0])
    expert = _expert(tmp_path, fetcher=_fake_fetcher())
    rep = expert.macro_report()
    text = rep.summary_text()
    assert "DXY 105.95" in text
    assert "US10Y 4.595" in text
    assert "bearish" in text
    assert "test-feed" in text
    assert "equity" in text.lower()
    d = rep.to_dict()
    assert d["status"] == "live"
    assert d["readings"]["dxy"]["close"] == pytest.approx(105.95)


def test_expert_macro_report_unavailable(tmp_path):
    expert = _expert(tmp_path, fetcher=_failing_fetcher)
    rep = expert.macro_report()
    assert rep.to_dict()["status"] == "unavailable"
    assert "unavailable" in rep.summary_text().lower()


def test_expert_trade_idea_gets_feeds(tmp_path):
    _seed_equity(tmp_path, [10000.0, 10100.0])
    expert = _expert(tmp_path, fetcher=_fake_fetcher())
    expert._load_bars = lambda *a, **k: _synthetic_bars()
    idea = expert.trade_idea("XAUUSD", market="forex")
    assert idea.macro["status"] == "live"
    assert idea.equity["points"] == [10000.0, 10100.0]
    assert "DXY" in idea.summary_text() or "macro" in idea.summary_text().lower()


def test_finance_macro_tool(tmp_path):
    import pandas as pd

    from nomorals.tools import finance as finance_tools

    def fake_ohlcv(symbol, market="stocks", timeframe="1d", bars=200,
                   source="auto"):
        closes = _fake_closes()[symbol]
        idx = pd.date_range("2026-07-01", periods=len(closes), freq="D")
        frame = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes,
             "close": closes, "volume": [0] * len(closes)}, index=idx)
        frame.attrs["source"] = "yahoo"
        return frame

    seen = {}

    class _Registry:
        context = SimpleNamespace(
            db=None, settings=SimpleNamespace(home_path=str(tmp_path)),
            rooms=None, current_room=None)

        def register(self, name, fn=None, **meta):
            if fn is not None:
                seen[name] = fn
                return fn

            def deco(func):
                seen[name] = func
                return func

            return deco

    finance_tools.register(_Registry())
    assert "finance_macro" in seen
    with mock.patch("nomorals.integrations.market_data.get_ohlcv",
                    side_effect=fake_ohlcv):
        # Registered context-first tools are bound to the registry
        # context at registration time — dispatch calls with no args.
        out = seen["finance_macro"]()
    assert out["status"] == "live"
    assert out["readings"]["dxy"]["close"] == pytest.approx(105.95)
    assert out["readings"]["tny"]["source"] == "yahoo"
