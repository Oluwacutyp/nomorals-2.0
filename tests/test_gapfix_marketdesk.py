"""Tests for the markets desk (nomorals/agents/marketdesk/).

BTC/stocks standing class (owner 2026-10-21): a MARKETS desk analyzing
BTCUSD + stocks -> analysis/signals. XAUUSD stays with GoldenSwing +
Devon trading; this desk is BTCUSD + US majors only.

The lab tests run on synthetic candle series with KNOWN ground truth:
a seeded random walk must come back NO_EDGE; a series with a planted
hour-of-day effect must come back EDGE_CANDIDATE with a callable hour
structure; a trend-only series may flag, but the signal engine must
refuse to call it — a timed alert needs a clock.

No network anywhere: fetches are injected at the collector seam.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

from nomorals.agents.marketdesk import lab
from nomorals.agents.marketdesk.collectors import (
    DEFAULT_SYMBOLS,
    SYMBOL_MARKETS,
    collect_all,
    collect_symbol,
)
from nomorals.agents.marketdesk.jobs import ensure_marketdesk_jobs
from nomorals.agents.marketdesk.signals import SignalEngine
from nomorals.agents.marketdesk.sources import keyed_feed_state
from nomorals.agents.marketdesk.store import Candle, CandleStore

# ── synthetic candle series ────────────────────────────────────────────

def _daily(n: int = 400, seed: int = 7, drift: float = 0.0,
           start: float = 1_700_000_000.0, price: float = 100.0,
           symbol: str = "BTCUSD"):
    """Daily candles: geometric random walk, optional daily drift."""
    rng = random.Random(seed)
    out = []
    p = price
    for i in range(n):
        ret = rng.gauss(drift, 0.02)
        o = p
        c = p * math.exp(ret)
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.005)))
        low = min(o, c) * (1 - abs(rng.gauss(0, 0.005)))
        out.append(Candle(symbol=symbol, timeframe="1d",
                          ts=start + i * 86400.0, open=o, high=h,
                          low=low, close=c,
                          volume=1000.0 + rng.random() * 500.0,
                          source="test"))
        p = c
    return out


def _hourly(n: int = 720, seed: int = 11, boost_hours: tuple = (),
            boost: float = 0.0, start: float = 1_700_000_000.0,
            price: float = 100.0, symbol: str = "BTCUSD"):
    """Hourly candles; ``boost_hours`` get a planted up-hour rate."""
    rng = random.Random(seed)
    out = []
    p = price
    for i in range(n):
        ts = start + i * 3600.0
        hour = int((ts // 3600) % 24)
        up = rng.random() < (0.5 + boost if hour in boost_hours else 0.5)
        ret = abs(rng.gauss(0, 0.01)) if up else -abs(rng.gauss(0, 0.01))
        o = p
        c = p * math.exp(ret)
        out.append(Candle(symbol=symbol, timeframe="1h", ts=ts,
                          open=o, high=max(o, c), low=min(o, c),
                          close=c, volume=500.0, source="test"))
        p = c
    return out


def _seed_store(tmp_path, symbol: str = "BTCUSD", daily_n: int = 400,
                hourly_n: int = 720, **kw):
    store = CandleStore(str(tmp_path))
    store.append(_daily(n=daily_n, symbol=symbol, **kw))
    store.append(_hourly(n=hourly_n, symbol=symbol, **kw))
    return store


# ── store ──────────────────────────────────────────────────────────────

def test_store_append_dedupes_and_counts(tmp_path):
    store = CandleStore(str(tmp_path))
    bars = _daily(n=50)
    assert store.append(bars)["added"] == 50
    assert store.append(bars)["added"] == 0
    assert store.append(bars[:10])["duplicates"] == 10
    assert store.count() == 50
    assert store.symbols() == ["BTCUSD"]


def test_store_roundtrip_fields(tmp_path):
    store = CandleStore(str(tmp_path))
    store.append([Candle("BTCUSD", "1d", 123.0, 100.0, 101.0, 99.0,
                         100.5, 42.0, source="test")])
    got = store.bars("BTCUSD", "1d")
    assert len(got) == 1 and got[0].close == 100.5
    assert got[0].volume == 42.0


# ── collectors ─────────────────────────────────────────────────────────

def _fake_fetch(rows):
    def fetch(symbol, market, timeframe, bars):
        return [dict(r) for r in rows if r["timeframe"] == timeframe]
    return fetch


def test_collect_symbol_lands_bars(tmp_path):
    store = CandleStore(str(tmp_path))
    rows = [{"symbol": "BTCUSD", "timeframe": "1d", "ts": 1_700_000_000.0,
             "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
             "volume": 10.0},
            {"symbol": "BTCUSD", "timeframe": "1h", "ts": 1_700_000_000.0,
             "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
             "volume": 5.0}]
    out = collect_symbol(store, "BTCUSD", fetch=_fake_fetch(rows))
    assert out["ok"] is True
    assert out["added"] == 2
    assert store.count() == 2
    assert out["source"] == "injected"


def test_collect_symbol_reports_failure_honestly(tmp_path):
    store = CandleStore(str(tmp_path))

    def failing(symbol, market, timeframe, bars):
        raise RuntimeError("yahoo: no data for BTCUSD")

    out = collect_symbol(store, "BTCUSD", fetch=failing)
    assert out["ok"] is False
    assert "yahoo" in out["error"]
    assert store.count() == 0


def test_collect_all_keeps_going_after_a_dead_leg(tmp_path):
    store = CandleStore(str(tmp_path))
    good = _daily(n=5) + _hourly(n=5)

    def fetch(symbol, market, timeframe, bars):
        if symbol == "AAPL":
            raise RuntimeError("stooq: no data for aapl.us")
        return [dict(symbol=symbol, timeframe=r.timeframe, ts=r.ts,
                     open=r.open, high=r.high, low=r.low, close=r.close,
                     volume=r.volume)
                for r in good if r.timeframe == timeframe]

    out = collect_all(store, symbols=("BTCUSD", "AAPL"), fetch=fetch)
    assert out["BTCUSD"]["ok"] is True
    assert out["AAPL"]["ok"] is False
    assert store.count() == 10


def test_default_symbols_are_btc_plus_us_majors():
    assert "BTCUSD" in DEFAULT_SYMBOLS
    assert "XAUUSD" not in DEFAULT_SYMBOLS
    assert all(SYMBOL_MARKETS[s] == "stocks"
               for s in DEFAULT_SYMBOLS if s != "BTCUSD")
    assert SYMBOL_MARKETS["BTCUSD"] == "crypto"


# ── lab ────────────────────────────────────────────────────────────────

def test_lab_random_walk_is_no_edge():
    v = lab.analyze_symbol(_daily(seed=7), _hourly(seed=11),
                           symbol="BTCUSD")
    assert v.verdict == "NO_EDGE"
    assert v.callable_structure == ""


def test_lab_insufficient_data_below_floor():
    v = lab.analyze_symbol(_daily(n=10), _hourly(n=10), symbol="BTCUSD")
    assert v.verdict == "INSUFFICIENT_DATA"
    assert "below" in v.evidence


def test_lab_planted_trend_flags():
    v = lab.analyze_symbol(_daily(seed=7, drift=0.004),
                           _hourly(seed=11), symbol="BTCUSD")
    names = [t.name for t in v.tests if t.ran]
    assert "trend" in names
    trend = next(t for t in v.tests if t.name == "trend")
    assert trend.p_value < 0.01


def test_lab_planted_hour_effect_is_callable():
    v = lab.analyze_symbol(_daily(seed=7),
                           _hourly(seed=11, boost_hours=(14, 15),
                                   boost=0.45),
                           symbol="BTCUSD")
    assert v.verdict == "EDGE_CANDIDATE"
    assert v.callable_structure == "hour_effect"
    # The desk names the hours that survive out-of-sample
    # confirmation — a non-empty subset of the planted hours,
    # never an hour that was not planted.
    assert v.best_hours
    assert set(v.best_hours) <= {14, 15}


def test_lab_skipped_test_says_so():
    # 60 daily bars: the level test needs 200 for MA200, so it skips.
    v = lab.analyze_symbol(_daily(n=60), _hourly(n=720), symbol="BTCUSD")
    level = next(t for t in v.tests if t.name == "level")
    assert not level.ran
    assert level.skipped


# ── signals ────────────────────────────────────────────────────────────

def _edge_verdict(now: float, hours=(14, 15)):
    return SimpleNamespace(
        symbol="BTCUSD", verdict="EDGE_CANDIDATE",
        callable_structure="hour_effect", best_hours=list(hours),
        hour_rates={14: 0.95, 15: 0.95}, base_success_rate=0.5,
        evidence="planted hour effect")


def test_signal_refuses_unconfirmed_structure(tmp_path):
    eng = SignalEngine(str(tmp_path))
    v = lab.analyze_symbol(_daily(seed=7), _hourly(seed=11),
                           symbol="BTCUSD")
    assert v.verdict == "NO_EDGE"
    decision = eng.evaluate(v)
    assert decision["call"] is None
    assert "NO_EDGE" in decision["reason"]


def test_signal_refuses_non_time_structure(tmp_path):
    eng = SignalEngine(str(tmp_path))
    v = SimpleNamespace(symbol="BTCUSD", verdict="EDGE_CANDIDATE",
                        callable_structure="", best_hours=[],
                        hour_rates={}, base_success_rate=0.5,
                        evidence="trend only")
    decision = eng.evaluate(v)
    assert decision["call"] is None
    assert "not time-based" in decision["reason"]


def test_signal_lead_floor_is_enforced(tmp_path):
    eng = SignalEngine(str(tmp_path))
    # Day floor 2023-11-15 00:00 UTC; now = 13:59 UTC.  The 14:00
    # boundary opens in 60s — inside the 120s lead floor — so the
    # engine must SKIP it and call the NEXT flagged window (15:00)
    # with proper lead, never the too-soon one.
    day = 1_700_006_400.0  # 2023-11-15 00:00:00 UTC
    now = day + 13 * 3600 + 59 * 60
    decision = eng.evaluate(_edge_verdict(now), now=now)
    assert decision["call"] is not None
    assert decision["call"].window_start == day + 15 * 3600
    assert decision["call"].lead_seconds == 3660
    assert decision["call"].lead_seconds >= 120
    # Same boundary from 13:00 -> 3600s lead: callable.
    decision = eng.evaluate(_edge_verdict(day + 13 * 3600),
                            now=day + 13 * 3600)
    assert decision["call"] is not None
    assert decision["call"].window_start == day + 14 * 3600
    assert decision["call"].lead_seconds == 3600


def test_signal_scores_against_landed_bars(tmp_path):
    eng = SignalEngine(str(tmp_path))
    store = CandleStore(str(tmp_path / "store"))
    now = (1_700_000_000.0 // 3600) * 3600 + 13 * 3600
    call = eng.evaluate(_edge_verdict(now), now=now)["call"]
    assert call is not None
    eng.publish(call, deliver=False)
    # An up hour lands inside the called window.
    start = call.window_start
    store.append([Candle("BTCUSD", "1h", start + 1800, 100.0, 101.0,
                         99.0, 100.8, 10.0, source="test")])
    scored = eng.score_due(store, now=call.window_end + 60)
    assert scored["calls_scored"] == 1
    assert scored["hits_total"] == 1
    assert scored["hit_rate"] == 1.0


# ── desk + library persistence ─────────────────────────────────────────

def test_library_verdicts_persist_across_restarts(tmp_path):
    from nomorals.agents.marketdesk import MarketsDesk
    desk = MarketsDesk(desk_dir=str(tmp_path))
    desk.store.append(_daily(symbol="BTCUSD"))
    desk.store.append(_hourly(symbol="BTCUSD", boost_hours=(14, 15),
                              boost=0.45))
    desk.analyze_all()
    lib = desk.library()
    assert lib["BTCUSD"]["verdict"] == "EDGE_CANDIDATE"
    desk2 = MarketsDesk(desk_dir=str(tmp_path))
    assert desk2.library()["BTCUSD"]["verdict"] == "EDGE_CANDIDATE"


def test_desk_status_names_verdicts_and_sources(tmp_path):
    from nomorals.agents.marketdesk import MarketsDesk
    desk = MarketsDesk(desk_dir=str(tmp_path))
    desk.store.append(_daily(n=100, symbol="BTCUSD"))
    st = desk.status()
    assert [s["symbol"] for s in st["symbols"]] == ["BTCUSD"]
    assert st["bars_total"] == 100
    assert "sources" in st


# ── keyed feeds ────────────────────────────────────────────────────────

def test_keyed_feed_without_key_reports_skipped(monkeypatch):
    for key in ("ALPHAVANTAGE_API_KEY", "TWELVEDATA_API_KEY",
                "FINNHUB_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    rows = keyed_feed_state()
    assert rows
    assert all(not r["key_present"] for r in rows)
    assert all("skipped" in r["status"] for r in rows)


# ── jobs + tool wiring ─────────────────────────────────────────────────

class _FakeScheduler:
    def __init__(self):
        self.jobs = []

    def list_jobs(self):
        return [{"name": j["name"]} for j in self.jobs]

    def add(self, name, spec, kind, payload, routine=False):
        self.jobs.append({"name": name, "spec": spec, "kind": kind,
                          "payload": payload, "routine": routine})
        return {"id": len(self.jobs)}


def test_ensure_marketdesk_jobs_idempotent():
    sched = _FakeScheduler()
    out = ensure_marketdesk_jobs(sched)
    assert len(out) == 2 and all(o.get("scheduled") for o in out)
    out2 = ensure_marketdesk_jobs(sched)
    assert all(o.get("already_scheduled") for o in out2)
    tools = {j["payload"]["tool"] for j in sched.jobs}
    assert tools == {"marketdesk"}
    assert all(j["routine"] for j in sched.jobs)


def test_tool_registered_in_agent_modules():
    from nomorals.tools.agents import AGENT_TOOL_MODULES
    assert "marketdesk" in AGENT_TOOL_MODULES


def test_tool_call_binds(tmp_path):
    from types import SimpleNamespace as NS

    from nomorals.tools.registry import ToolRegistry
    registry = ToolRegistry(NS(db=None, settings=None))
    registry.register_builtins()
    assert "marketdesk" in registry._tools
    out = registry.call("marketdesk", action="status")
    assert out.ok, getattr(out.error, "message", out.error)


def test_markets_command_is_listed_and_owner_only():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        PUBLIC_COMMANDS,
        parse_control,
    )
    assert "markets" in CONTROL_COMMANDS
    assert "markets" in COMMAND_DETAILS
    assert COMMAND_DETAILS["markets"].get("what")
    assert "markets" not in PUBLIC_COMMANDS
    listed = {c for _, v in LIST_GROUPS for c in v}
    assert "markets" in listed
    assert parse_control("/markets") is not None


def test_nm_markets_parses():
    from nomorals.cmdline.parser import _parser
    args = _parser().parse_args(["markets", "status"])
    assert args.markets_action == "status"
    args = _parser().parse_args(["markets", "signals", "--no-deliver"])
    assert args.markets_action == "signals" and args.no_deliver is True
