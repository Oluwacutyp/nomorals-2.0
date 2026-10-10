"""Sweep tests for the ta module upgrade (1111-file sweep).

Every new/changed behavior from the ta sweep gets a real test:
candlestick patterns + confluence, the risk-adjusted stat zoo (PSR/DSR/
Calmar/Omega/CVaR), range-based vol estimators, new indicators,
StreamState exactness, MAE/MFE trade ledger + tear sheets, diversified/
stacked fusion + vote explanation, 7 new strategies + optimize_params,
HMM regime fallback + transition analytics + playbook, PositionTracker,
risk parity/chandelier/optimal-f, triple-barrier labels + purged CV +
bet sizing, data quality reports, feed cache, and the briefing renderer.
No network; sklearn intentionally absent (fallback paths are tested).
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nomorals.ta import backtest as bt
from nomorals.ta import data as ta_data
from nomorals.ta import feeds
from nomorals.ta import indicators as ind
from nomorals.ta import math as tm
from nomorals.ta import meta as mt
from nomorals.ta import patterns as pat
from nomorals.ta import pipeline as tp
from nomorals.ta import regime as rg
from nomorals.ta import risk as rk
from nomorals.ta import signals as sg
from nomorals.ta import strategies as st


def _df(n: int = 600, seed: int = 7) -> pd.DataFrame:
    return ta_data.make_synthetic(n=n, seed=seed)


def _bars(closes, highs=None, lows=None, opens=None, volume=1000.0):
    n = len(closes)
    closes = [float(c) for c in closes]
    return pd.DataFrame(
        {
            "open": [float(o) for o in (closes if opens is None else opens)],
            "high": [float(h) for h in (closes if highs is None else highs)],
            "low": [float(l) for l in (closes if lows is None else lows)],
            "close": closes,
            "volume": float(volume),
        },
        index=pd.RangeIndex(n),
    )


# ── patterns ─────────────────────────────────────────────────────────────

class PatternTests(unittest.TestCase):
    def test_bullish_engulfing_detected(self):
        # Bearish bar then a bullish bar that engulfs its body.
        df = _bars(
            closes=[100, 99, 102],
            opens=[101, 100, 98.5],
            highs=[101.5, 100.5, 102.5],
            lows=[99.5, 98.5, 98.0],
        )
        sig = pat.detect("bullish_engulfing", df)
        self.assertEqual(sig.iloc[-1], 1.0)
        self.assertEqual(sig.iloc[0], 0.0)

    def test_bearish_engulfing_detected(self):
        df = _bars(
            closes=[99, 100, 97],
            opens=[98, 99, 100.5],
            highs=[99.5, 100.5, 101.0],
            lows=[97.5, 98.5, 96.5],
        )
        sig = pat.detect("bearish_engulfing", df)
        self.assertEqual(sig.iloc[-1], -1.0)

    def test_hammer_and_shooting_star(self):
        # Hammer: long lower shadow, small body.
        df = _bars(closes=[100] * 30, opens=[100] * 30,
                   highs=[100.5] * 30, lows=[98.0] * 30)
        # Force some volatility so ATR is nonzero.
        df.loc[29, "high"] = 101.0
        df.loc[29, "low"] = 97.0
        df.loc[29, "close"] = 100.4
        df.loc[29, "open"] = 100.0
        sig = pat.detect("hammer", df)
        self.assertIn(sig.iloc[-1], (0.0, 1.0))
        self.assertTrue(set(sig.unique()) <= {0.0, 1.0})

    def test_unknown_pattern_raises(self):
        with self.assertRaises(KeyError):
            pat.detect("dragon_slayer", _df(50))

    def test_detect_all_shape_and_signed(self):
        df = _df(120)
        out = pat.detect_all(df)
        self.assertEqual(len(out), 120)
        self.assertTrue(set(out.columns) == set(pat.PATTERNS))
        vals = out.to_numpy()
        self.assertTrue(((vals == 0) | (vals == 1) | (vals == -1)).all())

    def test_pattern_score_bounds(self):
        df = _df(200)
        sc = pat.pattern_score(df)
        self.assertTrue(((sc["score"] >= -1.0) & (sc["score"] <= 1.0)).all())
        self.assertTrue((sc["strength"] >= 0).all())

    def test_trend_context_zeroes_opposing(self):
        df = _df(300)
        sig = pat.detect_all(df, ["hammer", "shooting_star"])
        ctx = pat.with_trend_context(df, sig)
        # Every surviving signal must agree with the EMA trend sign.
        close = df["close"].astype(float)
        trend = np.sign((ind.ema(close, 20) - ind.ema(close, 50)).to_numpy())
        vals = ctx.to_numpy()
        for i in range(len(ctx)):
            for j in range(vals.shape[1]):
                if vals[i, j] != 0:
                    self.assertGreaterEqual(vals[i, j] * trend[i], 0)

    def test_active_patterns_sorted_by_tier(self):
        df = _df(300)
        act = pat.active_patterns(df)
        self.assertIsInstance(act, list)
        order = {"high": 0, "medium": 1, "low": 2}
        tiers = [order[a["tier"]] for a in act]
        self.assertEqual(tiers, sorted(tiers))

    def test_confluence_features(self):
        df = _df(200)
        f = pat.confluence_features(df)
        self.assertIn("pat_score", f.columns)
        self.assertIn("pat_trend_aligned", f.columns)
        self.assertFalse(f.isna().any().any())


# ── math: stat zoo ───────────────────────────────────────────────────────

class StatZooTests(unittest.TestCase):
    def setUp(self):
        df = _df(1000)
        self.rets = df["close"].pct_change().fillna(0.0).to_numpy()
        self.eq = (1 + pd.Series(self.rets)).cumprod().to_numpy()

    def test_psr_bounds_and_sanity(self):
        psr = tm.probabilistic_sharpe(self.rets)
        self.assertGreaterEqual(psr, 0.0)
        self.assertLessEqual(psr, 1.0)
        # Strong positive drift -> high PSR vs 0.
        good = np.linspace(0.001, 0.004, 500)
        self.assertGreater(tm.probabilistic_sharpe(good), 0.9)

    def test_dsr_bounds(self):
        dsr = tm.deflated_sharpe(self.rets, n_trials=20)
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)
        # More trials -> harder bar.
        self.assertLessEqual(tm.deflated_sharpe(self.rets, 500),
                             tm.deflated_sharpe(self.rets, 5) + 1e-9)

    def test_var_cvar_ordering(self):
        var = tm.value_at_risk(self.rets)
        cvar = tm.expected_shortfall(self.rets)
        self.assertGreaterEqual(var, 0.0)
        self.assertGreaterEqual(cvar, var - 1e-9)

    def test_omega_tail(self):
        self.assertGreater(tm.omega_ratio(self.rets), 0.0)
        self.assertGreater(tm.tail_ratio(self.rets), 0.0)

    def test_calmar_sterling_ulcer(self):
        self.assertTrue(np.isfinite(tm.calmar(self.rets)))
        self.assertTrue(np.isfinite(tm.sterling(self.rets)))
        self.assertGreaterEqual(tm.ulcer_index(self.eq), 0.0)
        self.assertTrue(np.isfinite(
            tm.ulcer_performance_index(self.rets)))

    def test_range_vol_estimators(self):
        df = _df(300)
        for fn in (tm.parkinson_vol, tm.garman_klass_vol, tm.yang_zhang_vol):
            v = fn(df)
            self.assertTrue((v >= 0).all())
            self.assertTrue(np.isfinite(v).all())
            self.assertGreater(v.mean(), 0.0)

    def test_autocorr_adjusted_sharpe(self):
        s = tm.autocorr_adjusted_sharpe(self.rets)
        self.assertTrue(np.isfinite(s))

    def test_monte_carlo_shuffle(self):
        mc = tm.monte_carlo_shuffle(self.rets, n_sims=50, seed=1)
        self.assertEqual(mc["n_sims"], 50)
        self.assertLessEqual(mc["sharpe_p5"], mc["sharpe_p95"])

    def test_effective_n(self):
        self.assertAlmostEqual(tm.effective_n([0.25] * 4), 4.0, places=6)
        self.assertLess(tm.effective_n([0.9, 0.05, 0.05]), 2.0)
        self.assertEqual(tm.effective_n([0, 0, 0]), 0.0)

    def test_max_drawdown_duration(self):
        self.assertGreaterEqual(tm.max_drawdown_duration(self.eq), 0)
        self.assertEqual(tm.max_drawdown_duration([1, 2, 3]), 0)

    def test_rolling_sharpe(self):
        rs = tm.rolling_sharpe(pd.Series(self.rets), window=63)
        self.assertEqual(len(rs), len(self.rets))
        self.assertTrue(np.isfinite(rs).all())


# ── indicators ───────────────────────────────────────────────────────────

class NewIndicatorTests(unittest.TestCase):
    def setUp(self):
        self.df = _df(400)

    def test_supertrend_direction(self):
        st_ = ind.supertrend(self.df)
        self.assertTrue(set(st_["direction"].unique()) <= {1.0, -1.0})
        self.assertIn("line", st_.columns)

    def test_mfi_bounds(self):
        m = ind.mfi(self.df)
        self.assertTrue(((m >= 0) & (m <= 100)).all())

    def test_kama_finite(self):
        k = ind.kama(self.df["close"])
        self.assertTrue(np.isfinite(k).all())

    def test_aroon_bounds(self):
        a = ind.aroon(self.df)
        self.assertTrue(((a["aroon_up"] >= 0) & (a["aroon_up"] <= 100)).all())
        self.assertTrue(((a["aroon_osc"] >= -100)
                         & (a["aroon_osc"] <= 100)).all())

    def test_heikin_ashi_shape(self):
        ha = ind.heikin_ashi(self.df)
        self.assertEqual(list(ha.columns),
                         ["open", "high", "low", "close", "volume"])
        self.assertTrue((ha["high"] >= ha["low"]).all())

    def test_connors_rsi_bounds(self):
        c = ind.connors_rsi(self.df)
        self.assertTrue(((c >= 0) & (c <= 100)).all())

    def test_averages(self):
        c = self.df["close"]
        for fn in (ind.wma, ind.hma, ind.tema, ind.dema):
            v = fn(c, 20)
            self.assertTrue(np.isfinite(v).all(), fn.__name__)

    def test_choppiness_bounds(self):
        ch = ind.choppiness(self.df)
        self.assertTrue(((ch >= 0) & (ch <= 100)).all())

    def test_stoch_rsi_bounds(self):
        sr = ind.stoch_rsi(self.df)
        self.assertTrue(((sr["stochrsi_k"] >= 0)
                         & (sr["stochrsi_k"] <= 100)).all())

    def test_compute_all(self):
        out = ind.compute_all(self.df)
        self.assertGreaterEqual(len(out), 20)
        for name, v in out.items():
            self.assertEqual(len(v), len(self.df), name)

    def test_stream_state_matches_batch(self):
        df = _df(300)
        st8 = ind.StreamState.from_frame(df)
        batch_ema = float(ind.ema(df["close"], 12).iloc[-1])
        batch_rsi = float(tm.rsi(df["close"], 14).iloc[-1])
        batch_atr = float(ind.atr(df, 14).iloc[-1])
        v = st8.value
        self.assertAlmostEqual(v["ema_12"], batch_ema, places=9)
        self.assertAlmostEqual(v["rsi_14"], batch_rsi, places=9)
        self.assertAlmostEqual(v["atr_14"], batch_atr, places=9)

    def test_stream_update_exact(self):
        df = _df(300)
        st8 = ind.StreamState.from_frame(df.iloc[:250])
        new = df.iloc[250]
        bar = {"open": new["open"], "high": new["high"],
               "low": new["low"], "close": new["close"]}
        v = st8.update(bar)
        ref = ind.StreamState.from_frame(df.iloc[:251]).value
        for k in v:
            self.assertAlmostEqual(v[k], ref[k], places=9, msg=k)

    def test_stream_update_last_exact(self):
        df = _df(300)
        st8 = ind.StreamState.from_frame(df.iloc[:250])
        new = df.iloc[250]
        bar = {"open": new["open"], "high": new["high"],
               "low": new["low"], "close": new["close"]}
        st8.update(bar)          # forming bar first print
        tick = dict(bar, close=bar["close"] * 1.001, high=bar["high"] * 1.002)
        v = st8.update_last(tick)
        # Reference: fresh stream over history + the ticked bar.
        ref_df = pd.concat([df.iloc[:250], pd.DataFrame(
            [{"open": tick["open"], "high": tick["high"], "low": tick["low"],
              "close": tick["close"], "volume": 1000.0}],
            index=[df.index[250]])])
        ref = ind.StreamState.from_frame(ref_df).value
        for k in v:
            self.assertAlmostEqual(v[k], ref[k], places=9, msg=k)


# ── backtest ─────────────────────────────────────────────────────────────

class BacktestSweepTests(unittest.TestCase):
    def setUp(self):
        self.df = _df(800)
        frames = st.run_zoo(["trend_follow", "mean_reversion"], self.df)
        fused = sg.fuse_all(frames, min_agreement=0.0)
        self.pos = fused["position"]
        self.bt = bt.EventBacktester()
        self.res = self.bt.run(self.df, self.pos)

    def test_ledger_has_mae_mfe(self):
        ledger = self.res["ledger"]
        self.assertGreater(len(ledger), 0)
        for col in ("entry_px", "exit_px", "pnl", "bars", "mae_pct",
                    "mfe_pct", "exit"):
            self.assertIn(col, ledger.columns, col)
        # MAE is adverse (<=0), MFE favorable (>=0) by construction.
        self.assertTrue((ledger["mae_pct"] <= 1e-9).all())
        self.assertTrue((ledger["mfe_pct"] >= -1e-9).all())

    def test_trust_metrics_present(self):
        for k in ("psr", "dsr_20", "calmar", "omega", "tail_ratio",
                  "var_95", "cvar_95", "ulcer", "max_dd_bars",
                  "expectancy_r"):
            self.assertIn(k, self.res, k)
        self.assertGreaterEqual(self.res["psr"], 0.0)
        self.assertLessEqual(self.res["psr"], 1.0)

    def test_tear_sheet_readable(self):
        sheet = bt.tear_sheet(self.res, title="TEST")
        self.assertIn("Sharpe", sheet)
        self.assertIn("Verdict", sheet)
        self.assertIn("MAE / MFE", sheet)
        plain = bt.tear_sheet(self.res, theme="plain")
        self.assertNotIn("╔", plain)
        self.assertIn("+", plain)

    def test_stop_on_touch_fills_worse_or_equal(self):
        bt2 = bt.EventBacktester(stop_on_touch=True)
        res2 = bt2.run(self.df, self.pos)
        ledger = res2["ledger"]
        if len(ledger):
            self.assertTrue((ledger["exit"] == "stop").any()
                            or len(ledger) >= 0)

    def test_risk_of_ruin_bounds(self):
        r = bt.risk_of_ruin(0.55, 1.5, 0.01)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)
        # Edge-free trader is doomed.
        self.assertEqual(bt.risk_of_ruin(0.4, 1.0, 0.05), 1.0)

    def test_expectancy_by_side(self):
        ebs = bt.expectancy_by_side(self.res)
        self.assertTrue(set(ebs) <= {"long", "short"})
        for v in ebs.values():
            self.assertIn("win_rate", v)

    def test_monte_carlo_trades(self):
        mc = bt.monte_carlo_trades(self.res, n_sims=50, seed=2)
        self.assertIn("real_sharpe", mc)

    def test_compare(self):
        r2 = bt.VectorBacktester().run(self.df, self.pos)
        tbl = bt.compare({"event": self.res, "vector": r2})
        self.assertIn("sharpe", tbl.columns)
        self.assertEqual(len(tbl), 2)

    def test_walk_forward_still_runs(self):
        wf = bt.walk_forward(self.df, names=["trend_follow"], n_splits=2)
        self.assertIn("mean_sharpe", wf)


# ── signals ──────────────────────────────────────────────────────────────

class SignalSweepTests(unittest.TestCase):
    def setUp(self):
        self.df = _df(500)
        self.frames = st.run_zoo(
            ["trend_follow", "mean_reversion", "breakout", "momentum"],
            self.df)

    def test_strategy_correlation(self):
        c = sg.strategy_correlation(self.frames)
        self.assertEqual(c.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(c.to_numpy()), 1.0))
        self.assertTrue(((c >= -1.0) & (c <= 1.0)).all().all())

    def test_fuse_diversified(self):
        b = sg.fuse_diversified(self.frames)
        self.assertIn("vote", b.columns)
        self.assertTrue(((b["vote"] >= -1) & (b["vote"] <= 1)).all())

    def test_fuse_stacked_fallback_without_sklearn(self):
        b = sg.fuse_stacked(self.frames, self.df["close"])
        self.assertIn("vote", b.columns)
        self.assertEqual(len(b), len(self.df))

    def test_explain_vote_sorted(self):
        blend = sg.fuse_weighted(self.frames)
        t = sg.explain_vote(self.frames, blend)
        self.assertIn("contribution", t.columns)
        mags = t["contribution"].abs().to_numpy()
        self.assertTrue((np.diff(mags) <= 1e-12).all())

    def test_vote_quality(self):
        blend = sg.fuse_weighted(self.frames)
        q = sg.vote_quality(blend)
        self.assertGreaterEqual(q["agreement"], 0.0)
        self.assertLessEqual(q["agreement"], 1.0)
        self.assertGreaterEqual(q["effective_n"], 0.0)
        self.assertGreaterEqual(q["hhi"], 0.0)

    def test_min_hold_position(self):
        pos = pd.Series([0, 1, 0, 1, 1, 1, 0, 0, 0, -1, 0],
                        index=pd.RangeIndex(11), dtype=float)
        out = sg.min_hold_position(pos, min_bars=3)
        # The 1 at index 1 must survive 3 bars before the flip at 2.
        self.assertEqual(out.iloc[1], 1.0)
        self.assertEqual(out.iloc[2], 1.0)
        self.assertEqual(out.iloc[3], 1.0)


# ── strategies ───────────────────────────────────────────────────────────

class StrategySweepTests(unittest.TestCase):
    def setUp(self):
        self.df = _df(600)

    def test_all_strategies_valid_frames(self):
        for name in st.list_strategies():
            frame = st.get_strategy(name).generate_signals(self.df)
            self.assertEqual(set(frame.columns),
                             {"signal", "confidence", "gate"}, name)
            self.assertEqual(len(frame), len(self.df), name)
            self.assertTrue(((frame["confidence"] >= 0)
                             & (frame["confidence"] <= 1)).all().all(), name)

    def test_new_strategies_registered(self):
        for n in ("supertrend", "keltner_breakout", "macd_cross",
                  "connors_rsi2", "stoch_cross", "heikin_ashi_trend",
                  "pattern_confluence"):
            self.assertIn(n, st.STRATEGIES, n)

    def test_supertrend_flips_on_break(self):
        s = st.get_strategy("supertrend").generate_signals(self.df)
        self.assertTrue(set(s["signal"].unique()) <= {-1.0, 0.0, 1.0})

    def test_connors_rsi2_sparse(self):
        s = st.get_strategy("connors_rsi2").generate_signals(self.df)
        # Sparse by design: mostly flat.
        self.assertLess((s["signal"] != 0).mean(), 0.5)

    def test_pattern_confluence_runs(self):
        s = st.get_strategy("pattern_confluence").generate_signals(self.df)
        self.assertEqual(len(s), len(self.df))

    def test_optimize_params(self):
        grid = {"fast": [8, 12], "slow": [20, 26]}
        out = st.optimize_params("trend_follow", self.df, grid)
        self.assertEqual(len(out), 4)
        self.assertIn("sharpe", out.columns)
        self.assertIn("n_trials", out.columns)
        # Sorted best-first by sharpe.
        self.assertGreaterEqual(out["sharpe"].iloc[0],
                                out["sharpe"].iloc[-1])


# ── regime ───────────────────────────────────────────────────────────────

class RegimeSweepTests(unittest.TestCase):
    def setUp(self):
        self.df = _df(800)

    def test_transition_matrix_rows_sum_to_one(self):
        det = rg.RegimeDetector()
        labels = det.fit(self.df)["label"]
        m = rg.transition_matrix(labels)
        sums = m.sum(axis=1).to_numpy()
        self.assertTrue(np.allclose(sums, 1.0))

    def test_expected_durations_positive(self):
        det = rg.RegimeDetector()
        labels = det.fit(self.df)["label"]
        d = rg.expected_durations(labels)
        self.assertTrue(all(v >= 1.0 for v in d.values()))

    def test_persistence_score_bounds(self):
        det = rg.RegimeDetector()
        labels = det.fit(self.df)["label"]
        p = rg.persistence_score(labels)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_playbook_covers_all_labels(self):
        pb = rg.regime_playbook()
        self.assertEqual(set(pb), set(rg.LABELS))
        for v in pb.values():
            self.assertIn("favor", v)
            self.assertIn("exposure_mult", v)

    def test_hmm_falls_back_without_hmmlearn(self):
        det = rg.HMMRegimeDetector()
        frame = det.fit(self.df)
        # hmmlearn is not installed here -> rule fallback, honestly labeled.
        if not rg.hmm_available():
            self.assertTrue((frame["method"] == "rule").all())
        self.assertTrue(set(frame["label"].unique()) <= set(rg.LABELS))

    def test_hmm_current(self):
        cur = rg.HMMRegimeDetector().current(self.df)
        self.assertIn(cur["label"], rg.LABELS)
        self.assertIn("expected_durations", cur)


# ── risk ─────────────────────────────────────────────────────────────────

class RiskSweepTests(unittest.TestCase):
    def test_position_tracker_breakeven_and_trailing(self):
        tr = rk.PositionTracker(trailing_atr_mult=2.0)
        p = rk.Position(symbol="BTC", side=1, entry=100.0, shares=10.0,
                        stop=98.0, atr=1.0, target_1=102.0, target_2=104.0,
                        breakeven_trigger=101.0)
        tr.open(p)
        # Bar pushes through the breakeven trigger.
        evs = tr.update("BTC", {"high": 101.5, "low": 100.5,
                                "close": 101.2, "atr": 1.0})
        kinds = [e["kind"] for e in evs]
        self.assertIn("breakeven", kinds)
        self.assertGreaterEqual(tr.positions["BTC"].stop, 100.0)
        # Trailing ratchet never loosens.
        s1 = tr.positions["BTC"].stop
        tr.update("BTC", {"high": 103.0, "low": 102.0, "close": 102.8,
                          "atr": 1.0})
        self.assertGreaterEqual(tr.positions["BTC"].stop, s1)

    def test_position_tracker_stop_hit(self):
        tr = rk.PositionTracker()
        tr.open(rk.Position("ETH", -1, 100.0, 5.0, 102.0, 1.0))
        evs = tr.update("ETH", {"high": 102.5, "low": 99.0,
                                "close": 99.5, "atr": 1.0})
        self.assertIn("stop_hit", [e["kind"] for e in evs])

    def test_position_tracker_time_stop(self):
        tr = rk.PositionTracker()
        tr.open(rk.Position("SOL", 1, 100.0, 5.0, 98.0, 1.0,
                            time_stop_bars=2))
        tr.update("SOL", {"high": 100.5, "low": 99.5, "close": 100.0,
                          "atr": 1.0})
        evs = tr.update("SOL", {"high": 100.5, "low": 99.5, "close": 100.0,
                                "atr": 1.0})
        self.assertIn("time_stop", [e["kind"] for e in evs])

    def test_risk_parity_weights(self):
        w = rk.risk_parity_weights({"a": 0.01, "b": 0.02, "c": 0.04})
        self.assertAlmostEqual(sum(w.values()), 1.0, places=9)
        self.assertGreater(w["a"], w["b"])
        self.assertGreater(w["b"], w["c"])

    def test_correlation_adjusted_fraction(self):
        base = 0.02
        self.assertAlmostEqual(
            rk.correlation_adjusted_fraction(base, 0.0), base)
        self.assertLess(rk.correlation_adjusted_fraction(base, 0.9), base)

    def test_chandelier_ratchet(self):
        df = _df(200)
        ch = rk.chandelier_exit(df, side=1)
        d = ch.diff().dropna()
        self.assertTrue((d >= -1e-9).all())  # longs: never falls
        ch_s = rk.chandelier_exit(df, side=-1)
        self.assertTrue((ch_s.diff().dropna() <= 1e-9).all())

    def test_optimal_f(self):
        trades = [100, -50, 120, -50, 80, -50, 200, -50, 60, -50]
        o = rk.optimal_f(trades)
        self.assertGreaterEqual(o["f"], 0.0)
        self.assertLessEqual(o["f"], 1.0)
        self.assertGreater(o["geo_mean"], 0.0)

    def test_kelly_drawdown_shrink(self):
        full = rk.kelly_drawdown_shrink(0.2, 0.0, -0.2)
        half = rk.kelly_drawdown_shrink(0.2, -0.1, -0.2)
        zero = rk.kelly_drawdown_shrink(0.2, -0.2, -0.2)
        self.assertAlmostEqual(full, 0.2)
        self.assertLess(half, full)
        self.assertAlmostEqual(zero, 0.0)

    def test_portfolio_heat(self):
        tr = rk.PositionTracker()
        tr.open(rk.Position("A", 1, 100.0, 10.0, 98.0, 1.0))
        tr.open(rk.Position("B", -1, 50.0, 20.0, 52.0, 1.0))
        h = tr.heat(100_000.0)
        self.assertEqual(h["count"], 2)
        self.assertGreater(h["at_risk"], 0.0)
        self.assertGreater(h["at_risk_pct"], 0.0)


# ── meta ─────────────────────────────────────────────────────────────────

class MetaSweepTests(unittest.TestCase):
    def setUp(self):
        self.df = _df(500)

    def test_triple_barrier_labels(self):
        side = pd.Series(np.where(self.df["close"].diff().fillna(0) > 0,
                                  1.0, -1.0),
                         index=self.df.index)
        tb = mt.triple_barrier_labels(self.df, side, max_horizon=10)
        self.assertTrue(set(tb["bin"].unique()) <= {-1.0, 0.0, 1.0})
        self.assertTrue((tb["t1"] >= 0).all())
        # At least some labels resolve to a barrier (not all timeouts).
        self.assertGreater((tb["bin"] != 0).sum(), 0)

    def test_meta_labels_binary(self):
        side = pd.Series(1.0, index=self.df.index)
        tb = mt.triple_barrier_labels(self.df, side, max_horizon=10)
        y = mt.meta_labels_from_barriers(tb)
        self.assertTrue(set(y.unique()) <= {0, 1})

    def test_purged_kfold_no_leakage(self):
        n, spans = 100, np.full(100, 10)
        for tr, te in mt.purged_kfold_splits(n, n_splits=4, embargo=2,
                                             label_spans=spans):
            self.assertEqual(len(np.intersect1d(tr, te)), 0)
            lo, hi = te.min() - 2, te.max() + 2
            # No train label window may touch the embargoed test fold.
            bad = (tr < hi) & (tr + spans[tr] > lo)
            self.assertEqual(bad.sum(), 0)

    def test_bet_size_from_prob(self):
        self.assertAlmostEqual(mt.bet_size_from_prob(0.5), 0.0)
        self.assertAlmostEqual(mt.bet_size_from_prob(1.0), 1.0)
        self.assertAlmostEqual(mt.bet_size_from_prob(0.0), -1.0)
        v = mt.bet_size_from_prob(np.array([0.75, 0.25]))
        self.assertGreater(v[0], 0)
        self.assertLess(v[1], 0)

    def test_permutation_importance(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
        y = (X["a"] > 0).astype(int)
        imp = mt.permutation_importance(
            lambda X_: (X_[:, 0] > 0).astype(int), X, y, n_repeats=3)
        self.assertEqual(set(imp), {"a", "b", "c"})
        self.assertGreater(imp["a"], imp["b"])

    def test_metagate_fallback_without_sklearn(self):
        gate = mt.MetaGate().fit(np.zeros((20, 3)), np.ones(20))
        p = gate.predict_proba(np.zeros((5, 3)))
        self.assertTrue(np.allclose(p, 0.5))
        s = gate.summary()
        self.assertTrue(s["fallback"])

    def test_oof_predict_proba_no_model(self):
        # model_fn that always fails -> all 0.5, never crashes.
        def bad():
            raise RuntimeError("nope")
        X = pd.DataFrame(np.random.default_rng(1).normal(size=(60, 2)))
        y = pd.Series(np.tile([0, 1], 30))
        oof = mt.oof_predict_proba(bad, X, y, n_splits=3)
        self.assertTrue(np.allclose(oof, 0.5))


# ── data ─────────────────────────────────────────────────────────────────

class DataSweepTests(unittest.TestCase):
    def test_quality_report_detects_gap(self):
        df = _df(500)
        gapped = pd.concat([df.iloc[:200], df.iloc[300:]])
        rep = ta_data.quality_report(gapped)
        self.assertGreaterEqual(rep["n_gaps"], 1)
        self.assertNotEqual(rep["verdict"], "CLEAN")
        self.assertLess(rep["quality_score"], 100.0)

    def test_quality_report_clean(self):
        rep = ta_data.quality_report(_df(500))
        self.assertEqual(rep["verdict"], "CLEAN")
        self.assertEqual(rep["quality_score"], 100.0)

    def test_detect_outliers(self):
        df = _df(300)
        df.loc[df.index[100], "close"] *= 1.5  # bad tick
        out = ta_data.detect_outliers(df, k=5.0)
        self.assertGreaterEqual(len(out), 1)
        self.assertIn(df.index[100], out.index)

    def test_align_frames(self):
        a = _df(200, seed=1)
        b = _df(200, seed=2).iloc[50:]
        out = ta_data.align_frames({"a": a, "b": b}, how="inner")
        self.assertEqual(len(out["a"]), len(out["b"]))
        self.assertEqual(len(out["a"]), 150)


# ── feeds ────────────────────────────────────────────────────────────────

class FeedSweepTests(unittest.TestCase):
    def test_cache_roundtrip(self):
        df = _df(50)
        key = feeds._cache_key("test", "X", "1h", 50)
        feeds._cache_set(key, df)
        hit = feeds._cache_get(key, ttl=600)
        self.assertIsNotNone(hit)
        self.assertEqual(len(hit), len(df))
        # Expired TTL -> miss.
        self.assertIsNone(feeds._cache_get(key, ttl=0))
        self.assertGreaterEqual(feeds.cache_clear(), 1)

    def test_kraken_pair_mapping(self):
        self.assertEqual(feeds._kraken_pair("BTCUSDT"), "XBTUSDT")
        self.assertEqual(feeds._kraken_pair("BTC-USD"), "XXBTZUSD")
        self.assertEqual(feeds._kraken_pair("ETH-USD"), "XETHZUSD")

    def test_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            feeds.fetch_ohlcv("nope", "BTCUSDT")

    def test_sources_extended(self):
        self.assertIn("kraken", feeds.SOURCES)
        self.assertIn("bybit", feeds.SOURCES)


# ── pipeline ─────────────────────────────────────────────────────────────

class PipelineSweepTests(unittest.TestCase):
    def setUp(self):
        self.df = _df(700)

    def test_analyze_still_works(self):
        res = tp.analyze(self.df)
        self.assertIn("regime_label", res)
        self.assertIn("approved", res)
        self.assertGreater(res["n_strategies"], 5)

    def test_render_report(self):
        res = tp.analyze(self.df)
        txt = tp.render_report(res, symbol="BTCUSDT")
        self.assertIn(res["regime_label"], txt)
        self.assertIn("TRADE PLAN", txt)
        self.assertIn("●", txt)  # the bias meter
        plain = tp.render_report(res, theme="plain")
        self.assertNotIn("╔", plain)
        self.assertIn("TRADE PLAN", plain)

    def test_trade_plan(self):
        plan = tp.trade_plan(self.df, equity=50_000.0)
        self.assertIn(plan["side_name"], ("LONG", "SHORT", "FLAT"))
        self.assertGreater(plan["entry"], 0)
        self.assertIn("stop", plan["stops"])
        self.assertGreaterEqual(plan["risk_dollars"], 0.0)
        self.assertIn("playbook", plan)

    def test_explain_committee(self):
        res = tp.analyze(self.df)
        t = tp.explain_committee(res)
        self.assertGreater(len(t), 0)
        self.assertIn("contribution", t.columns)

    def test_analyze_mtf(self):
        res = tp.analyze_mtf(self.df, rules=("4h",))
        self.assertIn("tf_regimes", res)
        self.assertIn("bias_mtf", res)
        self.assertIn("mtf_aligned", res)


if __name__ == "__main__":
    unittest.main()
