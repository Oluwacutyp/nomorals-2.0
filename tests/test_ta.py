"""Tests for Devon's native TA stack (nomorals.ta).

Covers: indicator math on synthetic candles (hand-computed expected values),
signal fusion determinism, the event backtester on known series, risk /
position-size math, and graceful degradation. No network, no submodule.
"""
from __future__ import annotations

import statistics
import unittest

import numpy as np
import pandas as pd

from nomorals.ta import backtest as bt
from nomorals.ta import data as ta_data
from nomorals.ta import indicators as ind
from nomorals.ta import math as tm
from nomorals.ta import pipeline as tp
from nomorals.ta import regime as rg
from nomorals.ta import risk as rk
from nomorals.ta import signals as sg
from nomorals.ta import strategies as st


def _bars(closes, highs=None, lows=None, volume=1000.0):
    n = len(closes)
    closes = [float(c) for c in closes]
    highs = [float(h) for h in (closes if highs is None else highs)]
    lows = [float(l) for l in (closes if lows is None else lows)]
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes,
         "volume": float(volume)},
        index=pd.RangeIndex(n),
    )


class IndicatorMathTests(unittest.TestCase):
    def test_rsi_hand_computed(self):
        # closes [10,11,12,11], period 2:
        # gains [1,1,0] -> avg [1,1,0.5]; losses [0,0,1] -> avg [0,0,0.5]
        # RSI: [100, 100, 50] (warmup backfilled to 100)
        out = tm.rsi(pd.Series([10.0, 11, 12, 11]), 2).tolist()
        self.assertEqual([round(v, 6) for v in out],
                         [100.0, 100.0, 100.0, 50.0])

    def test_rsi_boundaries(self):
        up = pd.Series(np.arange(1.0, 31.0))
        dn = pd.Series(np.arange(30.0, 0.0, -1.0))
        flat = pd.Series(np.full(30, 5.0))
        self.assertAlmostEqual(tm.rsi(up).iloc[-1], 100.0)
        self.assertAlmostEqual(tm.rsi(dn).iloc[-1], 0.0)
        self.assertAlmostEqual(tm.rsi(flat).iloc[-1], 50.0)

    def test_atr_hand_computed(self):
        df = _bars([10.5, 11.5, 11.0, 11.0],
                   highs=[11, 12, 12, 11.5], lows=[9, 10, 11, 10.5])
        # TR: [2, 2, 1, 1]; ATR(2) Wilder: [2, 2, 1.5, 1.25]
        out = tm.atr(df, 2).tolist()
        self.assertEqual([round(v, 6) for v in out], [2.0, 2.0, 1.5, 1.25])

    def test_stochastic_hand_computed(self):
        df = _bars([10.5, 11.5, 11.0, 11.0],
                   highs=[11, 12, 12, 11.5], lows=[9, 10, 11, 10.5])
        s = ind.stochastic(df, k=2, d=3)
        self.assertEqual([round(v, 4) for v in s["k"]],
                         [75.0, 83.3333, 50.0, 33.3333])
        self.assertEqual([round(v, 4) for v in s["d"]],
                         [75.0, 79.1667, 69.4444, 55.5556])

    def test_sma_head_usable(self):
        out = tm.sma(pd.Series([1.0, 2, 3, 4, 5]), 3).tolist()
        self.assertEqual(out, [1.0, 1.5, 2.0, 3.0, 4.0])

    def test_bollinger_against_stdlib(self):
        closes = [10.0, 11, 12, 11, 10, 9, 10, 12, 13, 12]
        df = _bars(closes)
        b = ind.bollinger(df, period=3, mult=2.0)
        i = 4
        window = closes[i - 2:i + 1]
        mean = statistics.fmean(window)
        sd = statistics.pstdev(window)
        self.assertAlmostEqual(b["mid"].iloc[i], mean)
        self.assertAlmostEqual(b["upper"].iloc[i], mean + 2 * sd)
        self.assertAlmostEqual(b["lower"].iloc[i], mean - 2 * sd)
        # %B at the top of the band window
        self.assertGreater(b["pct_b"].iloc[2], 0.5)

    def test_macd_matches_independent_ema(self):
        # Independent loop EMA (not pandas) as the reference.
        closes = [float(100 + i * 0.7 + (i % 5)) for i in range(60)]
        df = _bars(closes)

        def ema_loop(s, span):
            k = 2.0 / (span + 1)
            e = s[0]
            out = []
            for v in s:
                e = v * k + e * (1 - k)
                out.append(e)
            return out

        m = ind.macd(df)["macd"].tolist()
        ref = [a - b for a, b in zip(ema_loop(closes, 12),
                                     ema_loop(closes, 26))]
        for got, want in zip(m, ref):
            self.assertAlmostEqual(got, want, places=6)

    def test_adx_bounds_and_trend(self):
        n = 120
        close = 100 * np.exp(np.cumsum(np.full(n, 0.003)))
        df = _bars(close, highs=close * 1.002, lows=close * 0.998)
        a = ind.adx(df)
        self.assertTrue(((a["adx"] >= 0) & (a["adx"] <= 100)).all())
        # clean uptrend: +DI dominates, ADX strong
        self.assertGreater(a["plus_di"].iloc[-1], a["minus_di"].iloc[-1])
        self.assertGreater(a["adx"].iloc[-1], 25.0)

    def test_obv_sign(self):
        df = _bars([10.0, 11, 12, 11], volume=100.0)
        self.assertEqual(ind.obv(df).tolist(), [0.0, 100.0, 200.0, 100.0])

    def test_no_nans_after_warmup(self):
        df = ta_data.make_synthetic(300, seed=3)
        for name in ("rsi", "atr", "obv"):
            s = getattr(ind, name)(df)
            self.assertFalse(s.isna().any(), name)
        for name, fn in (("macd", ind.macd), ("bollinger", ind.bollinger),
                         ("stochastic", ind.stochastic), ("adx", ind.adx),
                         ("donchian", ind.donchian)):
            f = fn(df)
            self.assertFalse(f.isna().any().any(), name)


class FusionTests(unittest.TestCase):
    def _frames(self, idx):
        def fr(sig, conf=0.8, gate=1.0):
            return pd.DataFrame(
                {"signal": np.full(len(idx), sig),
                 "confidence": np.full(len(idx), conf),
                 "gate": np.full(len(idx), gate)}, index=idx)
        return {"a": fr(1.0), "b": fr(1.0), "c": fr(-1.0), "d": fr(0.0)}

    def test_fuse_deterministic(self):
        idx = pd.RangeIndex(50)
        frames = self._frames(idx)
        v1 = sg.fuse_all(frames)["blend"]["vote"]
        v2 = sg.fuse_all(frames)["blend"]["vote"]
        pd.testing.assert_series_equal(v1, v2)
        # 2 long (w=0.8) vs 1 short (w=0.8); the flat frame adds mass
        # but no direction: vote = 0.8 / 3.2 = 0.25
        self.assertAlmostEqual(v1.iloc[-1], 0.25)

    def test_hysteresis_hand_computed(self):
        v = pd.Series([0.0, 0.4, 0.1, -0.4, -0.1, 0.0])
        pos = sg.hysteresis_position(v, enter=0.3, exit=0.03)
        self.assertEqual(pos.tolist(), [0.0, 1.0, 1.0, -1.0, -1.0, 0.0])

    def test_cost_aware_threshold_scales_with_cost(self):
        lo = sg.cost_aware_threshold(5.0, 0.02)
        hi = sg.cost_aware_threshold(50.0, 0.02)
        self.assertGreater(hi, lo)
        self.assertGreaterEqual(lo, 0.02)

    def test_empty_frames_degrades(self):
        out = sg.fuse_all({})
        self.assertTrue(out["blend"].empty)
        self.assertEqual(out["n_strategies"], 0)


class StrategyTests(unittest.TestCase):
    def test_registry_and_unknown(self):
        names = st.list_strategies()
        self.assertEqual(names, ["breakout", "mean_reversion", "momentum",
                                "trend_follow"])
        self.assertEqual(st.list_strategies("trend"), ["trend_follow"])
        with self.assertRaises(KeyError):
            st.get_strategy("nope")

    def test_trend_follow_long_on_uptrend(self):
        n = 300
        rng = np.random.default_rng(1)
        close = 100 * np.exp(np.cumsum(0.002 + rng.normal(0, 0.003, n)))
        df = _bars(close, highs=close * 1.001, lows=close * 0.999)
        sig = st.get_strategy("trend_follow").generate_signals(df)
        self.assertEqual(set(sig.columns), {"signal", "confidence", "gate"})
        self.assertEqual(len(sig), n)
        self.assertEqual(sig["signal"].iloc[-1], 1.0)
        self.assertTrue(((sig["confidence"] >= 0)
                         & (sig["confidence"] <= 1)).all())

    def test_run_zoo_isolates_failures(self):
        df = ta_data.make_synthetic(200, seed=5)
        out = st.run_zoo(["trend_follow", "bogus_name"], df)
        self.assertEqual(list(out), ["trend_follow"])

    def test_rank_strategies_orders_best_first(self):
        df = ta_data.make_synthetic(600, seed=11)
        frames = st.run_zoo(st.list_strategies(), df)
        ranked = st.rank_strategies(frames, df["close"])
        self.assertGreater(len(ranked), 0)
        scores = ranked["score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))
        for col in ("sharpe", "hit_rate", "turnover", "trades", "score"):
            self.assertIn(col, ranked.columns)


class BacktestTests(unittest.TestCase):
    def _rising(self, n=10, ret=0.01):
        close = 100 * (1 + ret) ** np.arange(n)
        idx = pd.RangeIndex(n)
        df = pd.DataFrame(
            {"open": close, "high": close * 1.001, "low": close * 0.999,
             "close": close, "volume": 1000.0}, index=idx)
        return df, pd.Series(1.0, index=idx)

    def test_event_backtester_exact_profit(self):
        df, pos = self._rising()
        r = bt.EventBacktester(fee_bps=0, slippage_atr=0, spread_bps=0,
                               latency_bars=0).run(df, pos)
        self.assertAlmostEqual(r["total_return"], 1.01 ** 9 - 1)
        self.assertEqual(r["trades"], 1)
        self.assertAlmostEqual(r["final_equity"], 100_000 * 1.01 ** 9)

    def test_event_backtester_flat_is_zero(self):
        df, _ = self._rising()
        r = bt.EventBacktester(latency_bars=0).run(
            df, pd.Series(0.0, index=df.index))
        self.assertEqual(r["total_return"], 0.0)
        self.assertEqual(r["trades"], 0)

    def test_event_backtester_fees_drag(self):
        # flip long/short every bar on a flat market: costs must bleed
        n = 100
        close = np.full(n, 100.0)
        idx = pd.RangeIndex(n)
        df = pd.DataFrame(
            {"open": close, "high": close * 1.0005, "low": close * 0.9995,
             "close": close, "volume": 1000.0}, index=idx)
        pos = pd.Series(np.where(np.arange(n) % 2 == 0, 1.0, -1.0), index=idx)
        r = bt.EventBacktester(fee_bps=5, latency_bars=0).run(df, pos)
        self.assertLess(r["total_return"], -0.05)
        self.assertGreater(r["trades"], 50)

    def test_event_backtester_latency_delays_entry(self):
        df, pos = self._rising(n=6)
        r = bt.EventBacktester(fee_bps=0, slippage_atr=0, spread_bps=0,
                               latency_bars=2).run(df, pos)
        # entry delayed 2 bars: captures 1.01^3 instead of 1.01^5
        self.assertAlmostEqual(r["total_return"], 1.01 ** 3 - 1, places=6)

    def test_vector_backtester_matches_buy_hold(self):
        df, pos = self._rising()
        r = bt.VectorBacktester(fee_bps=0, latency_bars=0).run(df, pos)
        self.assertAlmostEqual(
            r["total_return"], df["close"].iloc[-1] / df["close"].iloc[0] - 1)

    def test_walk_forward_structure(self):
        df = ta_data.make_synthetic(900, seed=21)
        wf = bt.walk_forward(df, n_splits=3)
        self.assertEqual(wf["n_folds"], 3)
        self.assertEqual(len(wf["folds"]), 3)
        self.assertIn("mean_sharpe", wf)


class RiskTests(unittest.TestCase):
    def test_fixed_sizing_exact(self):
        s = rk.RiskManager().size_position(100_000, 100, 2.0)
        # risk 1% = 1000 over a 4.0 stop -> 250 shares -> 25% of equity
        self.assertAlmostEqual(s["fraction"], 0.25)
        self.assertAlmostEqual(s["shares"], 250.0)
        self.assertAlmostEqual(s["notional"], 25_000.0)
        self.assertAlmostEqual(s["stop_dist"], 4.0)

    def test_stop_ladder_exact(self):
        rm = rk.RiskManager()
        long = rm.stop_levels(1, 100, 2.0)
        self.assertEqual(long, {"stop": 96.0, "breakeven_trigger": 102.0,
                               "target_1": 104.0, "target_2": 108.0})
        short = rm.stop_levels(-1, 100, 2.0)
        self.assertEqual(short["stop"], 104.0)
        self.assertEqual(short["target_2"], 92.0)

    def test_trailing_stop_ratchets_only(self):
        self.assertEqual(rk.RiskManager.trailing_stop(1, 96.0, 103.0, 2.0),
                         99.0)
        # stop never loosens
        self.assertEqual(rk.RiskManager.trailing_stop(1, 99.0, 100.0, 2.0),
                         99.0)

    def test_kelly_fraction(self):
        self.assertAlmostEqual(rk.RiskManager.kelly_fraction(0.6, 1.5),
                               1.0 / 3.0)

    def test_governor_halts_on_drawdown(self):
        rm = rk.RiskManager()
        rm.update_equity(100_000)
        gov = rm.update_equity(80_000)  # -20% vs -15% max
        self.assertTrue(gov["halted"])
        self.assertFalse(rm.allow_trade(80_000, 0.0)["allow"])

    def test_position_size_profiles(self):
        for profile in ("default", "aggressive", "conservative"):
            s = rk.position_size(100_000, 100.0, 2.0, profile=profile)
            self.assertIn("stops", s)
            self.assertEqual(s["profile"], profile)
        sizes = [rk.position_size(100_000, 100.0, 2.0, p)["fraction"]
                 for p in ("conservative", "default", "aggressive")]
        self.assertEqual(sizes, sorted(sizes))

    def test_unknown_profile_falls_back_to_default(self):
        s = rk.position_size(100_000, 100.0, 2.0, profile="bogus")
        self.assertEqual(s["profile"], "bogus")
        self.assertAlmostEqual(s["fraction"], 0.25)


class PipelineTests(unittest.TestCase):
    def test_analyze_structure(self):
        df = ta_data.make_synthetic(600, seed=7)
        res = tp.analyze(df)
        for key in ("regime_label", "bias", "agreement", "position_now",
                    "approved", "size_fraction", "stop_distance_pct", "stops",
                    "entry", "ranked", "n_strategies", "bars"):
            self.assertIn(key, res, key)
        self.assertIn(res["regime_label"], rg.LABELS)
        self.assertTrue(-1.0 <= res["bias"] <= 1.0)
        self.assertTrue(0.0 <= res["agreement"] <= 1.0)
        self.assertIn(res["position_now"], (-1.0, 0.0, 1.0))
        self.assertEqual(res["n_strategies"], 4)
        self.assertEqual(res["bars"], 600)

    def test_analyze_deterministic(self):
        df = ta_data.make_synthetic(600, seed=7)
        a = tp.analyze(df)
        b = tp.analyze(df)
        for key in ("bias", "agreement", "position_now", "regime_label",
                    "approved", "size_fraction"):
            self.assertEqual(a[key], b[key], key)

    def test_analyze_uptrend_is_bullish(self):
        rng = np.random.default_rng(0)
        n = 600
        close = 100 * np.exp(np.cumsum(0.002 + rng.normal(0, 0.004, n)))
        df = _bars(close, highs=close * 1.002, lows=close * 0.998)
        res = tp.analyze(df)
        self.assertEqual(res["regime_label"], "TREND_UP")
        self.assertGreater(res["bias"], 0.5)
        self.assertEqual(res["position_now"], 1.0)

    def test_analyze_bad_profile(self):
        df = ta_data.make_synthetic(300, seed=1)
        with self.assertRaises(ValueError):
            tp.analyze(df, profile="yolo")

    def test_analyze_empty_raises(self):
        with self.assertRaises(ValueError):
            tp.analyze(pd.DataFrame())

    def test_analyze_too_few_bars_raises(self):
        df = ta_data.make_synthetic(300, seed=1).iloc[:20]
        with self.assertRaises(ValueError):
            tp.analyze(df)

    def test_committee_position_shape(self):
        df = ta_data.make_synthetic(300, seed=9)
        pos = tp.committee_position(df)
        self.assertEqual(len(pos), len(df))
        self.assertTrue(set(pos.unique()) <= {-1.0, 0.0, 1.0})


class DataTests(unittest.TestCase):
    def test_make_synthetic_deterministic(self):
        a = ta_data.make_synthetic(200, seed=42)
        b = ta_data.make_synthetic(200, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_split_embargo_no_overlap(self):
        df = ta_data.make_synthetic(500, seed=2)
        train, test = ta_data.split_embargo(df, test_frac=0.2,
                                            embargo_bars=20)
        self.assertTrue(train.index.max() < test.index.min())
        gap = (test.index[0] - train.index[-1]) / pd.Timedelta("1h")
        self.assertGreaterEqual(gap, 20)

    def test_clean_repairs_inversions(self):
        df = ta_data.make_synthetic(100, seed=3)
        df.loc[df.index[5], "high"] = 1.0  # broken bar
        cleaned = ta_data.clean_ohlcv(df)
        row = cleaned.iloc[5]
        self.assertGreaterEqual(row["high"],
                                max(row["open"], row["low"], row["close"]))


if __name__ == "__main__":
    unittest.main()
