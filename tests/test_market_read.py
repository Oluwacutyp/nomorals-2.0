"""Pass 1 of Devon's trading system: the standing whole-market read.

The read assembles D1·H4·H1·M30·M15·M5 from the market-data provider
seam (``integrations.market_data`` — faked here at the module seam,
the same one FinancialExpert uses) and the TA structure library.
Every test in this file is a NEW-BEHAVIOR test: on pristine 32448cc
the organ does not exist and the file fails at import. Preservation
of existing behavior is carried by the neighborhood suites.
"""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from nomorals.agents import market_read
from nomorals.finance import trading_journal
from nomorals.integrations import market_data
from nomorals.ta import structure

LAST_TS = 1762362000.0  # 2025-11-05 17:00 UTC
NOW = LAST_TS + 60.0
_STEP = {"1d": 86400, "4h": 14400, "1h": 3600,
         "30m": 1800, "15m": 900, "5m": 300}
# Distinct drift per timeframe: the hierarchy trends up while the
# execution timeframes trend down — if any block were filled from
# another timeframe's bars, its bias would betray it.
DRIFTS = {"1d": 0.9, "4h": 0.6, "1h": 0.35,
          "30m": 0.2, "15m": -0.3, "5m": -0.5}


def make_frame(*, n=220, base=2600.0, drift=0.35, tf="1h",
               source="fake_src", freshness="fresh"):
    closes = [base + drift * i + 6.0 * math.sin(i * 2 * math.pi / 24)
              for i in range(n)]
    rows, idx = [], []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c - drift
        hi = max(o, c) + 0.9 + 0.5 * ((i * 37) % 10) / 10.0
        lo = min(o, c) - 0.9 - 0.5 * ((i * 53) % 10) / 10.0
        rows.append((o, hi, lo, c, 100.0))
        idx.append(pd.Timestamp(LAST_TS - (n - 1 - i) * _STEP[tf],
                                unit="s", tz="UTC"))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                     "volume"], index=idx)
    df.attrs["source"] = source
    df.attrs["freshness"] = freshness
    df.attrs["last_bar_ts"] = LAST_TS
    return df


class FakeProviders:
    def __init__(self, *, stale=(), missing=(), short=(),
                 drifts=None, quote_error=None):
        self.stale = set(stale)
        self.missing = set(missing)
        self.short = set(short)
        self.drifts = dict(DRIFTS if drifts is None else drifts)
        self.quote_error = quote_error

    def ohlcv(self, symbol, market="crypto", timeframe="1h",
              bars=500, source="auto"):
        if timeframe in self.missing:
            raise market_data.MarketDataError(
                f"fake chain: {timeframe} unavailable")
        n = 12 if timeframe in self.short else bars
        return make_frame(n=n, drift=self.drifts[timeframe], tf=timeframe,
                          source=f"fake_{timeframe}",
                          freshness="stale" if timeframe in self.stale
                          else "fresh")

    def quote(self, symbol, market="crypto"):
        if self.quote_error:
            raise market_data.MarketDataError(self.quote_error)
        return {"symbol": symbol, "price": 2677.5,
                "change_pct_24h": 0.8, "currency": "USD",
                "source": "fake_quote", "quote_ts": LAST_TS,
                "quote_age_s": 12.0, "freshness": "fresh",
                "bid": 2677.4, "ask": 2677.6, "spread_bps": 0.7}


@pytest.fixture
def providers(monkeypatch):
    fake = FakeProviders()
    monkeypatch.setattr(market_data, "get_ohlcv", fake.ohlcv)
    monkeypatch.setattr(market_data, "quote", fake.quote)
    return fake


def _read(**kw):
    kw.setdefault("now", NOW)
    return market_read.whole_market_read("XAUUSD", market="forex", **kw)


def test_read_assembles_all_timeframes_with_provenance(providers):
    read = _read()
    assert read["kind"] == "market_read" and read["symbol"] == "XAUUSD"
    assert tuple(read["timeframes"]) == market_read.TIMEFRAMES
    for tf, blk in read["timeframes"].items():
        assert blk["status"] == "ok", tf
        assert blk["source"] == f"fake_{tf}", tf
        assert blk["freshness"] == "fresh", tf
        assert blk["bars"] == market_read.DEFAULT_BARS[tf], tf
        assert blk["last_bar_ts"] == LAST_TS, tf
        assert blk["bar_age_s"] == 60.0, tf
        assert blk["zones"] is not None, tf
        assert blk["premium_discount"] is not None, tf
    assert read["complete"] is True and read["degraded"] == []
    price = read["price"]
    assert price["price"] == 2677.5 and price["source"] == "fake_quote"
    assert price["freshness"] == "fresh"
    assert read["session"]["name"] == "new_york"


def test_each_timeframe_bias_comes_from_its_own_bars(providers):
    read = _read()
    labels = {tf: blk["bias"]["label"]
              for tf, blk in read["timeframes"].items()}
    assert labels["1d"] == "UPTREND" and labels["4h"] == "UPTREND"
    assert labels["1h"] == "UPTREND" and labels["30m"] == "UPTREND"
    assert labels["15m"] == "DOWNTREND" and labels["5m"] == "DOWNTREND"
    assert read["timeframes"]["5m"]["phase"] == "downtrend"
    hier = read["hierarchy"]
    assert hier["swing"]["bias_label"] == "UPTREND"
    assert hier["intraday"]["bias_label"] == "UPTREND"
    assert hier["scalp"]["bias_label"] == "UPTREND"
    assert read["alignment"] == "aligned_up"


def test_bias_is_the_structure_read_not_an_indicator(monkeypatch,
                                                      providers):
    seen = {}

    def spy_bias(pts, lookback=4):
        seen["rows"] = len(pts)
        return {"bias": -1, "label": "SENTINEL", "reason": "spy"}

    monkeypatch.setattr(structure, "market_bias", spy_bias)
    read = _read()
    assert seen["rows"] > 0  # the spy really was fed swing points
    blk = read["timeframes"]["1d"]
    assert blk["bias"]["label"] == "SENTINEL"
    assert blk["phase"] == "sentinel"
    assert read["hierarchy"]["swing"]["bias"] == -1


def test_flat_market_reads_range(providers):
    providers.drifts = {tf: 0.0 for tf in market_read.TIMEFRAMES}
    read = _read()
    for tf, blk in read["timeframes"].items():
        assert blk["bias"]["label"] == "RANGE", tf
        assert blk["bias"]["bias"] == 0, tf
    assert read["alignment"] == "flat"


def test_missing_timeframe_is_marked_never_filled(providers):
    providers.missing = {"15m"}
    read = _read()
    blk = read["timeframes"]["15m"]
    assert blk["status"] == "missing"
    assert blk["bias"] is None and blk["zones"] is None
    assert "15m" in blk["error"]
    assert read["degraded"] == ["15m"] and read["complete"] is False
    # Neighbours keep their own reads, untouched by the gap.
    assert read["timeframes"]["5m"]["bias"]["label"] == "DOWNTREND"
    assert read["timeframes"]["1h"]["bias"]["label"] == "UPTREND"


def test_stale_timeframe_is_marked_and_read_from_its_own_bars(providers):
    providers.stale = {"5m"}
    read = _read()
    blk = read["timeframes"]["5m"]
    assert blk["status"] == "stale" and blk["freshness"] == "stale"
    assert blk["bar_age_s"] == 60.0
    assert blk["bias"]["label"] == "DOWNTREND"  # its own bars, marked
    assert read["degraded"] == ["5m"] and read["complete"] is False


def test_short_timeframe_skips_structure_honestly(providers):
    providers.short = {"5m"}
    read = _read()
    blk = read["timeframes"]["5m"]
    assert blk["status"] == "ok" and blk["bars"] == 12
    assert blk["bias"] is None and "insufficient bars" in \
        blk["analysis_note"]


def test_quote_failure_still_reads_from_bars(providers):
    providers.quote_error = "fake chain: no quote"
    read = _read()
    assert read["price"]["price"] is None
    assert "no quote" in read["price"]["error"]
    assert read["timeframes"]["1d"]["bias"]["label"] == "UPTREND"
    assert read["timeframes"]["1d"]["zones"] is not None


def test_journal_roundtrip(providers, tmp_path):
    path = tmp_path / "trading_system_journal.jsonl"
    read = _read(journal=True, journal_path=path)
    events = trading_journal.read_events(event="market_read", path=path)
    assert len(events) == 1
    assert events[0]["snapshot"]["symbol"] == "XAUUSD"
    assert events[0]["snapshot"]["taken_at"] == read["taken_at"]
    snaps = market_read.recent_reads("XAUUSD", journal_path=path)
    assert len(snaps) == 1 and snaps[0]["kind"] == "market_read"
    assert market_read.recent_reads("ETHUSD", journal_path=path) == []


def test_journal_reader_tolerates_a_torn_line(tmp_path):
    path = tmp_path / "j.jsonl"
    trading_journal.append_event("market_read", path=path, symbol="X")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "market_read", "ts": ')  # torn append
    events = trading_journal.read_events(path=path)
    assert len(events) == 1 and events[0]["symbol"] == "X"


def test_render_names_provenance_stale_and_missing(providers):
    providers.stale = {"5m"}
    providers.missing = {"15m"}
    text = market_read.render_read(_read())
    assert "XAUUSD whole-market read" in text
    assert "fake_quote" in text and "UPTREND" in text
    assert "STALE" in text and "NO DATA" in text


def test_cmdline_read_action(providers, tmp_path, capsys):
    from nomorals.cmdline.commands.trade import _cmd_trade
    settings = SimpleNamespace(home_path=tmp_path)
    context = SimpleNamespace(settings=settings)
    args = argparse.Namespace(trade_action="read", symbol="XAUUSD",
                              market="forex", json=True)
    assert _cmd_trade(args, context) == 0
    out = capsys.readouterr().out
    assert "market_read" in out
    journal = tmp_path / "finance" / "trading_system_journal.jsonl"
    assert journal.exists()
    events = trading_journal.read_events(event="market_read",
                                         path=journal)
    assert len(events) == 1


def test_trading_tool_read_action(providers, tmp_path):
    from nomorals.tools import trading as trading_tool

    class FakeRegistry:
        def __init__(self, context):
            self.context = context
            self.tools = {}

        def register(self, name, **kwargs):
            def deco(fn):
                self.tools[name] = fn
                return fn
            return deco

    settings = SimpleNamespace(
        home_path=tmp_path,
        trading=SimpleNamespace(enabled=True))
    context = SimpleNamespace(settings=settings)
    reg = FakeRegistry(context)
    trading_tool.register(reg)
    out = reg.tools["trading"](action="read", symbol="XAUUSD",
                               market="forex")
    assert out["kind"] == "market_read"
    assert out["hierarchy"]["swing"]["bias_label"] == "UPTREND"
    assert "whole-market read" in out["rendered"]
