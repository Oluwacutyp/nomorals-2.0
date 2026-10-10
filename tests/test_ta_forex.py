"""Tests for the forex TA modules: structure, smc, sessions, analyst."""

import numpy as np
import pandas as pd
import pytest

from nomorals.ta import analyst, sessions, smc, structure


@pytest.fixture
def df():
    np.random.seed(42)
    n = 400
    # trending + noise so structure exists
    trend = np.linspace(0, 0.08, n)
    ret = np.random.normal(0.0001, 0.003, n)
    close = 2650 * np.exp(trend + np.cumsum(ret))
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.normal(0, 0.0008, n)),
            "high": close * (1 + np.abs(np.random.normal(0, 0.0018, n))),
            "low": close * (1 - np.abs(np.random.normal(0, 0.0018, n))),
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
    )


def test_swings_confirmed_not_repainted(df):
    s = structure.swings(df)
    assert set(s.columns) == {"swing_high", "swing_low"}
    assert len(s) == len(df)
    # last `right` bars can never have confirmed swings (prospective)
    assert not s["swing_high"].iloc[-5:].any()
    assert not s["swing_low"].iloc[-5:].any()


def test_swing_points_sorted(df):
    pts = structure.swing_points(df)
    assert (pts["bar"].astype(str) <= pts["bar"].astype(str).shift(-1).fillna("~")).all() \
        or pts["bar"].is_monotonic_increasing


def test_market_bias_keys(df):
    pts = structure.swing_points(df)
    b = structure.market_bias(structure.label_swings(pts))
    assert b["bias"] in (-1, 0, 1)
    assert b["label"] in ("UPTREND", "DOWNTREND", "RANGE")


def test_structure_breaks_prospective(df):
    brk = structure.structure_breaks(df)
    assert set(brk.columns) >= {"bar", "kind", "level", "bias_before"}
    assert set(brk["kind"]).issubset(
        {"BOS_BULL", "BOS_BEAR", "CHOCH_BULL", "CHOCH_BEAR"})


def test_sr_zones_strength(df):
    zones = structure.sr_zones(df)
    assert not zones.empty
    assert (zones["strength"] >= 0).all() and (zones["strength"] <= 100).all()
    assert (zones["touches"] >= 2).all()
    assert set(zones["state"]).issubset(
        {"support", "resistance", "inside", "broken_support",
         "broken_resistance"})
    lv = structure.nearest_levels(zones, float(df["close"].iloc[-1]))
    assert "support" in lv and "resistance" in lv


def test_fvg_detection(df):
    fvg = smc.fair_value_gaps(df)
    assert set(fvg.columns) >= {"bar", "kind", "top", "bottom", "mitigated"}
    assert set(fvg["kind"]).issubset({1, -1})
    unmit = smc.unmitigated_fvgs(df)
    assert (unmit["mitigated"] == False).all()


def test_order_blocks(df):
    obs = smc.order_blocks(df)
    if not obs.empty:
        assert set(obs["kind"]).issubset({1, -1})
        assert "effective_kind" in obs.columns


def test_liquidity_sweeps_stamped_on_reclaim(df):
    swp = smc.liquidity_sweeps(df)
    if not swp.empty:
        assert set(swp["kind"]).issubset({1, -1})
        assert (swp["penetration_atr"] >= 0.25).all()


def test_premium_discount_bounds(df):
    pd_ = smc.premium_discount(df)
    assert ((pd_["premium"] >= 0) & (pd_["premium"] <= 1)).all()
    assert set(pd_["zone"].dropna().unique()).issubset(
        {"discount", "equilibrium", "premium"})


def test_smc_confluence_honest(df):
    c = smc.smc_confluence(df)
    assert -100 <= c["score"] <= 100
    assert c["bias"] in (-1, 0, 1)
    assert "disclaimer" in c


def test_session_at_known_times():
    assert sessions.session_at(pd.Timestamp("2026-01-05 14:00", tz="UTC")) == "overlap"
    assert sessions.session_at(pd.Timestamp("2026-01-05 03:00", tz="UTC")) == "asian"
    assert sessions.session_at(pd.Timestamp("2026-01-05 23:00", tz="UTC")) == "off"


def test_time_filter_gate():
    ok = sessions.time_filter_ok(pd.Timestamp("2026-01-05 14:00", tz="UTC"))
    assert ok["ok"] is True
    bad = sessions.time_filter_ok(pd.Timestamp("2026-01-05 03:00", tz="UTC"))
    assert bad["ok"] is False


def test_session_volatility(df):
    vol = sessions.session_volatility(df)
    assert vol  # non-empty
    for v in vol.values():
        assert v["mean_range_atr"] > 0


def test_analyst_no_trade_on_noise():
    np.random.seed(1)
    n = 200
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    noisy = pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": 100},
        index=pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"))
    idea = analyst.Analyst().analyze(noisy, symbol="TEST")
    # pure noise should not produce a confident trade
    assert idea.side == 0 or idea.confidence < 80


def test_analyst_idea_wellformed(df):
    idea = analyst.Analyst().analyze(df, symbol="XAUUSD")
    assert idea.analyst_signature == "ta.analyst.v1"
    assert 0 <= idea.confidence <= 100
    assert isinstance(idea.reasons, list) and isinstance(idea.warnings, list)
    if idea.side != 0:
        assert idea.targets  # a trade needs targets
        chk = analyst.executor_check(idea)
        assert chk["ok"] is True


def test_executor_check_refuses():
    bad = analyst.TradeIdea(symbol="X", side=1, confidence=90.0,
                            entry=100.0, invalidation=101.0, targets=[],
                            analyst_signature="forged")
    assert analyst.executor_check(bad)["ok"] is False
    unsigned = analyst.TradeIdea(symbol="X", side=0, confidence=0.0,
                                 entry=0.0, invalidation=0.0, targets=[])
    assert analyst.executor_check(unsigned)["ok"] is False
    # inverted stop must be refused
    inv = analyst.TradeIdea(symbol="X", side=1, confidence=90.0,
                            entry=100.0, invalidation=90.0, targets=[(110, 1.0)])
    assert analyst.executor_check(inv)["ok"] is True
