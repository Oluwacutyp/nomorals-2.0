"""Gapfix R2-30: adaptive TA + macro bias wired into the real pipeline.

Failing-first: the wiring does not exist yet — ``pipeline.analyze``
takes no adaptive/equity/rate_tuner/macro kwargs, ``Analyst.analyze``
takes no macro kwarg, ``ta`` exports neither module, and no result
carries an ``adaptive`` record. These tests pin the contract the
implementation must meet.

Direction note (repo truth, flagged in the survey): VolatilityBreath
STRETCHES lookbacks when vol is high relative to its median and
SHRINKS them when vol is low — the module's documented design, not
the "volatile = faster" intuition.
"""

import unittest

import numpy as np
import pandas as pd

from nomorals.ta import analyst, pipeline
from nomorals.ta.adaptive import DrawdownThrottle, QuantileGate, SignalRateTuner, VolatilityBreath
from nomorals.ta.macro import trinity_bias

# ── fixtures ─────────────────────────────────────────────────────────

def realistic(trend_slope, seed, n=400, drift=0.0001, vol=0.003):
    """Bars with real structure (same shape as test_ta_forex's fixture)."""
    np.random.seed(seed)
    trend = np.linspace(0, trend_slope, n)
    ret = np.random.normal(drift, vol, n)
    close = 2650 * np.exp(trend + np.cumsum(ret))
    return pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.0008, n)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.0018, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.0018, n))),
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    }, index=pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"))


def regime_shift_bars(late_vol, seed=7, n_cal=300, n_late=100):
    """300 calm bars then 100 bars at ``late_vol`` — a vol regime shift."""
    rng = np.random.default_rng(seed)
    calm = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n_cal)))
    late = calm[-1] * np.exp(np.cumsum(rng.normal(0, late_vol, n_late)))
    closes = np.concatenate([calm, late])
    return pd.DataFrame({
        "open": closes, "high": closes * 1.001, "low": closes * 0.999,
        "close": closes, "volume": np.full(len(closes), 1000.0),
    }, index=pd.date_range("2026-01-01", periods=len(closes), freq="h",
                            tz="UTC"))


BEARISH_MACRO = {"bias": "bearish", "dxy_trend": 1.10, "y10y_trend": 0.17,
                 "corr": -0.50, "decoupled": False,
                 "reasons": ["DXY 20d +1.10%, US10Y 20d +0.17pts"]}
BULLISH_MACRO = {"bias": "bullish", "dxy_trend": -1.10, "y10y_trend": -0.17,
                 "corr": -0.50, "decoupled": False,
                 "reasons": ["DXY 20d -1.10%, US10Y 20d -0.17pts"]}


# ── 1. VolatilityBreath direction (the module's own design) ──────────

class VolatilityBreathTests(unittest.TestCase):
    def test_high_vol_stretches_lookback(self):
        vb = VolatilityBreath()
        volatile = regime_shift_bars(0.02)["close"]
        period = vb.dynamic_period(volatile, 14)
        assert int(period.iloc[-1]) > 14
        assert float(vb.scale(volatile).iloc[-1]) > 1.0

    def test_low_vol_shrinks_lookback(self):
        vb = VolatilityBreath()
        calm = regime_shift_bars(0.0005)["close"]
        period = vb.dynamic_period(calm, 14)
        assert int(period.iloc[-1]) < 14
        assert float(vb.scale(calm).iloc[-1]) < 1.0


# ── 2. Pipeline ATR adapts ───────────────────────────────────────────

class PipelineAdaptiveTests(unittest.TestCase):
    def test_pipeline_atr_period_adapts_to_volatility(self):
        hot = pipeline.analyze(regime_shift_bars(0.02))
        cold = pipeline.analyze(regime_shift_bars(0.0005))
        assert hot["adaptive"]["atr_period"] > 14
        assert hot["adaptive"]["vol_scale"] > 1.0
        assert cold["adaptive"]["atr_period"] < 14
        assert cold["adaptive"]["vol_scale"] < 1.0
        assert hot["adaptive"]["base_atr_period"] == 14

    def test_pipeline_disabled_adaptive_reproduces_old_behavior(self):
        df = realistic(0.15, 42)
        res = pipeline.analyze(df, adaptive=False)
        assert res["adaptive"]["enabled"] is False
        assert res["adaptive"]["atr_period"] == 14
        assert res["adaptive"]["adjustments"] == []
        assert res["adaptive"]["throttle_factor"] == 1.0


# ── 3. DrawdownThrottle sizing ───────────────────────────────────────

class DrawdownThrottleTests(unittest.TestCase):
    def test_throttle_reduces_sizing_in_drawdown_and_restores(self):
        df = realistic(0.15, 42)
        peak = np.linspace(100_000, 120_000, 100)
        drawn = np.concatenate(
            [peak, np.linspace(120_000, 108_000, 20)])  # 10% drawdown
        recovered = np.concatenate([drawn, np.linspace(108_000, 120_000, 20)])
        base = pipeline.analyze(df)
        under_dd = pipeline.analyze(df, equity=drawn)
        back = pipeline.analyze(df, equity=recovered)
        assert under_dd["adaptive"]["throttle_factor"] < 1.0
        assert under_dd["size_fraction"] < base["size_fraction"]
        assert back["adaptive"]["throttle_factor"] == 1.0
        assert back["size_fraction"] == base["size_fraction"]

    def test_throttle_unit_factor_shape(self):
        dt = DrawdownThrottle(soft_dd=0.05, hard_dd=0.15)
        assert dt.factor(np.linspace(100_000, 120_000, 100)) == 1.0
        dd10 = np.concatenate([np.linspace(100_000, 120_000, 100),
                               np.linspace(120_000, 108_000, 20)])
        assert 0.0 < dt.factor(dd10) < 1.0


# ── 4. Macro bias moves the analyst ──────────────────────────────────

class MacroAnalystTests(unittest.TestCase):
    def test_trinity_bias_shifts_analyst_confidence(self):
        df = realistic(0.15, 42)
        plain = analyst.Analyst().analyze(df, symbol="XAUUSD")
        bear = analyst.Analyst().analyze(df, symbol="XAUUSD",
                                         macro=BEARISH_MACRO)
        bull = analyst.Analyst().analyze(df, symbol="XAUUSD",
                                         macro=BULLISH_MACRO)
        # Opposing macro wind costs conviction; agreeing wind adds it.
        assert bull.confidence > plain.confidence > bear.confidence
        assert any("macro" in r for r in bear.reasons)
        assert bear.macro.get("bias") == "bearish"
        assert bull.macro.get("bias") == "bullish"
        assert plain.macro == {}

    def test_macro_none_reproduces_old_behavior(self):
        df = realistic(0.15, 42)
        idea = analyst.Analyst().analyze(df, symbol="XAUUSD")
        assert idea.macro == {}
        assert not any("macro" in r for r in idea.reasons)

    def test_trinity_bias_function_still_deterministic(self):
        xau = list(100.0 * np.exp(np.cumsum(
            np.random.default_rng(5).normal(0.001, 0.005, 60))))
        dxy = list(100.0 * np.exp(np.cumsum(
            np.random.default_rng(6).normal(0.002, 0.002, 60))))
        tny = list(np.linspace(4.0, 4.5, 60))
        out = trinity_bias(xau, dxy, tny)
        assert out is not None and out["bias"] in (
            "bullish", "bearish", "neutral")
        assert trinity_bias([1, 2, 3], [1, 2, 3], [1, 2, 3]) is None


# ── 5. Macro bias discounts the pipeline ─────────────────────────────

class MacroPipelineTests(unittest.TestCase):
    def test_pipeline_macro_discounts_opposing_bias(self):
        df = realistic(0.15, 42)
        plain = pipeline.analyze(df)
        assert plain["bias"] > 0  # fixture is a real uptrend
        opposed = pipeline.analyze(df, macro=BEARISH_MACRO)
        agreeing = pipeline.analyze(df, macro=BULLISH_MACRO)
        assert abs(opposed["bias"]) < abs(plain["bias"])
        assert opposed["adaptive"]["macro_discounted"] is True
        assert opposed["adaptive"]["macro"]["bias"] == "bearish"
        assert agreeing["bias"] == plain["bias"]
        assert agreeing["adaptive"]["macro_discounted"] is False


# ── 6. Adjustments are visible, not silent ───────────────────────────

class VisibilityTests(unittest.TestCase):
    def test_adaptive_adjustments_recorded_and_rendered(self):
        res = pipeline.analyze(regime_shift_bars(0.02))
        adj = res["adaptive"]["adjustments"]
        assert adj, "adaptive run must record what it changed"
        assert any("ATR" in a for a in adj)
        report = pipeline.render_report(res)
        assert "ADAPTIVE" in report


# ── 7. QuantileGate replaces the fixed RSI fade ──────────────────────

class QuantileGateTests(unittest.TestCase):
    def test_gate_position_direction(self):
        qg = QuantileGate()
        rising = pd.Series(np.linspace(40, 90, 200))
        assert float(qg.position(rising).iloc[-1]) > 0.5
        falling = pd.Series(np.linspace(90, 40, 200))
        assert float(qg.position(falling).iloc[-1]) < -0.5

    def test_gate_keeps_bullish_read_when_rsi_is_high_but_not_extreme(self):
        # 250 strong bars (RSI pinned ~85-100) then a shallow pullback:
        # RSI falls to ~71 — above the FIXED 70 fade — but low relative
        # to its own history, so the gate must NOT fade the long.
        rng = np.random.default_rng(11)
        rise = 100.0 * np.exp(np.cumsum(rng.normal(0.004, 0.002, 250)))
        pull = rise[-1] * np.exp(np.cumsum(rng.normal(-0.0005, 0.001, 20)))
        closes = np.concatenate([rise, pull])
        n = len(closes)
        rng2 = np.random.default_rng(3)
        df = pd.DataFrame({
            "open": closes * (1 + rng2.normal(0, 0.0008, n)),
            "high": closes * (1 + np.abs(rng2.normal(0, 0.0018, n))),
            "low": closes * (1 - np.abs(rng2.normal(0, 0.0018, n))),
            "close": closes,
            "volume": np.random.randint(100, 1000, n),
        }, index=pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"))
        idea = analyst.Analyst().analyze(df, symbol="XAUUSD")
        ema_line = [r for r in idea.reasons if "EMA20/50" in r]
        assert ema_line and "bullish" in ema_line[0]
        assert any("quantile" in r for r in idea.reasons)


# ── 8. SignalRateTuner gates approval ────────────────────────────────

class RateTunerTests(unittest.TestCase):
    def test_tuner_tightens_threshold_when_choppy(self):
        tuner = SignalRateTuner()
        choppy = realistic(0.0, 99, vol=0.01)  # sideways chop
        res = pipeline.analyze(choppy, rate_tuner=tuner)
        assert res["adaptive"]["rate_gate_mult"] > 1.0
        assert (res["adaptive"]["enter_threshold"]
                > res["adaptive"]["enter_threshold_raw"])

    def test_tuner_relaxes_threshold_when_trending(self):
        tuner = SignalRateTuner()
        trending = realistic(0.30, 5)
        res = pipeline.analyze(trending, rate_tuner=tuner)
        assert res["adaptive"]["rate_gate_mult"] < 1.0


# ── 9. Package exports ───────────────────────────────────────────────

class ExportsTests(unittest.TestCase):
    def test_ta_exports_adaptive_and_macro(self):
        from nomorals import ta
        for name in ("VolatilityBreath", "QuantileGate", "SignalRateTuner",
                     "DrawdownThrottle", "trinity_bias", "TrinityBias"):
            assert hasattr(ta, name), f"ta.{name} not exported"
        assert "adaptive" in ta.__all__ and "macro" in ta.__all__
