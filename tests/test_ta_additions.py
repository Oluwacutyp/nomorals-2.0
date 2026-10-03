"""Tests for the finacc-1.0 TA additions (nomorals.ta).

Covers: the new indicators (VWAP, Ichimoku, Parabolic SAR, CCI,
Williams %R, Fibonacci, Keltner — plus the EMA/SMA reuse check), the five
new strategies (IchimokuTrend, VwapBounce, RsiDivergence,
BollingerSqueeze, SarReversal), and the connector-backed OHLCV feeds.
No network: connector tests use injected fakes.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nomorals.ta import data as ta_data
from nomorals.ta import feeds
from nomorals.ta import indicators as ind
from nomorals.ta import math as tm
from nomorals.ta import strategies as st


def _bars(closes, highs=None, lows=None, volume=1000.0, index=None):
    n = len(closes)
    closes = [float(c) for c in closes]
    highs = [float(h) for h in (closes if highs is None else highs)]
    lows = [float(l) for l in (closes if lows is None else lows)]
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes,
         "volume": volume if not hasattr(volume, "__len__") else volume},
        index=pd.RangeIndex(n) if index is None else index,
    )


class EmaSmaReuseTests(unittest.TestCase):
    def test_not_duplicated(self):
        # math.py owns the series math; indicators re-exports it.
        self.assertIs(ind.ema, tm.ema)
        self.assertIs(ind.sma, tm.sma)

    def test_ema_sma_values(self):
        s = pd.Series([1.0, 2, 3, 4, 5])
        self.assertAlmostEqual(ind.ema(s, 3).iloc[-1], tm.ema(s, 3).iloc[-1])
        self.assertAlmostEqual(ind.sma(s, 3).iloc[-1], 4.0)


class VwapTests(unittest.TestCase):
    def test_hand_computed(self):
        # tp: 9.5, 10.5, 34/3; vols 100, 200, 300
        df = _bars([9.5, 10.5, 11.0],
                   highs=[10, 11, 12], lows=[9, 10, 11],
                   volume=[100.0, 200.0, 300.0])
        out = ind.vwap(df)
        tp3 = (12 + 11 + 11.0) / 3.0
        expect = (9.5 * 100 + 10.5 * 200 + tp3 * 300) / 600.0
        self.assertAlmostEqual(out.iloc[-1], expect, places=9)
        self.assertFalse(out.isna().any())

    def test_day_anchor_resets(self):
        idx = pd.to_datetime(["2026-01-01 10:00", "2026-01-01 11:00",
                              "2026-01-02 10:00", "2026-01-02 11:00"])
        df = _bars([10, 10, 20, 20], highs=[11, 11, 21, 21],
                   lows=[9, 9, 19, 19], volume=100.0, index=idx)
        out = ind.vwap(df)
        # Day 2 anchors at its own typical price, not day 1's.
        self.assertAlmostEqual(out.iloc[2], 20.0, places=9)
        self.assertAlmostEqual(out.iloc[0], 10.0, places=9)

    def test_zero_volume_no_nan(self):
        df = _bars([10, 11, 12], volume=0.0)
        out = ind.vwap(df)
        self.assertFalse(out.isna().any().any())
        # Falls back to typical price.
        self.assertAlmostEqual(out.iloc[-1], 12.0)


class IchimokuTests(unittest.TestCase):
    def test_columns_and_tenkan_math(self):
        df = ta_data.make_synthetic(200, seed=11)
        out = ind.ichimoku(df)
        self.assertEqual(list(out.columns),
                         ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"])
        self.assertFalse(out.isna().any().any())
        i = 100
        expect = (df["high"].iloc[i - 8:i + 1].max()
                  + df["low"].iloc[i - 8:i + 1].min()) / 2.0
        self.assertAlmostEqual(out["tenkan"].iloc[i], expect, places=9)

    def test_displacement(self):
        df = ta_data.make_synthetic(200, seed=11)
        out = ind.ichimoku(df, displacement=26)
        i = 150
        unshifted = (out["tenkan"].iloc[i - 26] + out["kijun"].iloc[i - 26]) / 2
        self.assertAlmostEqual(out["senkou_a"].iloc[i], unshifted, places=6)
        self.assertAlmostEqual(out["chikou"].iloc[i],
                               df["close"].iloc[i + 26], places=9)

    def test_flat_series(self):
        df = _bars([5.0] * 60)
        out = ind.ichimoku(df)
        self.assertFalse(out.isna().any().any())
        self.assertTrue((out["tenkan"] == 5.0).all())


class PsarTests(unittest.TestCase):
    def test_uptrend_trails_below(self):
        df = _bars(list(np.arange(100.0, 130.0)),
                   highs=list(np.arange(100.0, 130.0) + 1),
                   lows=list(np.arange(100.0, 130.0) - 1))
        sar = ind.psar(df)
        self.assertTrue((sar.iloc[1:] <= df["low"].iloc[1:]).all())
        self.assertTrue((sar.diff().iloc[2:] > 0).all())

    def test_downtrend_flips(self):
        df = _bars(list(np.arange(130.0, 100.0, -1)),
                   highs=list(np.arange(130.0, 100.0, -1) + 1),
                   lows=list(np.arange(130.0, 100.0, -1) - 1))
        sar = ind.psar(df)
        cross = np.sign(df["close"].to_numpy() - sar.to_numpy())
        flip_at = int(np.argmax(cross < 0))
        self.assertLess(flip_at, 5)  # seed-long self-corrects fast
        self.assertTrue((sar.iloc[flip_at:] >= df["high"].iloc[flip_at:]).all())

    def test_v_shape_flips_twice(self):
        dn = list(np.arange(130.0, 115.0, -1))
        up = list(np.arange(115.0, 130.0))
        closes = dn + up
        df = _bars(closes,
                   highs=[c + 0.5 for c in closes],
                   lows=[c - 0.5 for c in closes])
        sar = ind.psar(df)
        cross = np.sign(df["close"].to_numpy() - sar.to_numpy())
        flips = int((np.diff(cross) != 0).sum())
        self.assertGreaterEqual(flips, 2)

    def test_synthetic_has_flips_and_no_nan(self):
        df = ta_data.make_synthetic(600, seed=7)
        sar = ind.psar(df)
        cross = np.sign(df["close"].to_numpy() - sar.to_numpy())
        self.assertGreater(int((np.diff(cross) != 0).sum()), 0)
        self.assertFalse(sar.isna().any())


class CciTests(unittest.TestCase):
    def test_hand_computed(self):
        # h=l=c -> tp=[10,11,12]; period 3 at last bar:
        # tp_sma series = [10, 10.5, 11]; |dev| series = [0, 0.5, 1];
        # md = mean(0, 0.5, 1) = 0.5; cci = (12-11)/(0.015*0.5) = 133.33
        df = _bars([10.0, 11.0, 12.0])
        out = ind.cci(df, 3)
        self.assertAlmostEqual(out.iloc[-1], 133.333333, places=4)

    def test_flat_is_zero(self):
        df = _bars([7.0] * 40)
        out = ind.cci(df)
        self.assertTrue((out == 0.0).all())


class WilliamsRTests(unittest.TestCase):
    def test_hand_computed(self):
        # hh=12, ll=10, close=11 -> -50
        df = _bars([10.0, 11.0, 12.0, 11.0],
                   highs=[10, 11, 12, 12], lows=[10, 10, 10, 10])
        out = ind.williams_r(df, 3)
        self.assertAlmostEqual(out.iloc[-1], -50.0, places=6)

    def test_bounds(self):
        df = ta_data.make_synthetic(400, seed=5)
        out = ind.williams_r(df)
        self.assertTrue(((out >= -100.0) & (out <= 0.0)).all())
        self.assertFalse(out.isna().any())


class FibonacciTests(unittest.TestCase):
    def test_levels(self):
        df = _bars(list(np.linspace(90.0, 110.0, 50)))
        lv = ind.fibonacci(df, period=50)
        self.assertAlmostEqual(lv["swing_high"], 110.0, places=6)
        self.assertAlmostEqual(lv["swing_low"], 90.0, places=6)
        self.assertAlmostEqual(lv["50.0%"], 100.0, places=6)
        self.assertAlmostEqual(lv["61.8%"], 90.0 + 0.618 * 20.0, places=6)
        self.assertAlmostEqual(lv["0.0%"], 90.0, places=9)
        self.assertAlmostEqual(lv["100.0%"], 110.0, places=9)
        prices = [lv[k] for k in ("0.0%", "23.6%", "38.2%", "50.0%",
                                  "61.8%", "78.6%", "100.0%")]
        self.assertEqual(prices, sorted(prices))

    def test_period_clamped(self):
        df = _bars([10.0, 12.0, 11.0])
        lv = ind.fibonacci(df, period=500)  # longer than the frame
        self.assertAlmostEqual(lv["swing_high"], 12.0)


class KeltnerTests(unittest.TestCase):
    def test_structure(self):
        df = ta_data.make_synthetic(300, seed=9)
        out = ind.keltner(df)
        self.assertEqual(list(out.columns), ["upper", "mid", "lower"])
        self.assertTrue((out["upper"] >= out["mid"]).all())
        self.assertTrue((out["mid"] >= out["lower"]).all())
        self.assertFalse(out.isna().any().any())

    def test_mid_is_ema_and_width_is_atr(self):
        df = ta_data.make_synthetic(300, seed=9)
        out = ind.keltner(df, period=20, atr_period=10, mult=2.0)
        mid_expect = tm.ema(df["close"].astype(float), 20)
        band_expect = 2.0 * tm.atr(df, 10)
        self.assertAlmostEqual(out["mid"].iloc[-1], mid_expect.iloc[-1],
                               places=9)
        self.assertAlmostEqual(out["upper"].iloc[-1] - out["mid"].iloc[-1],
                               band_expect.iloc[-1], places=9)


class NewStrategyFrameTests(unittest.TestCase):
    NAMES = ["ichimoku_trend", "vwap_bounce", "rsi_divergence",
             "bollinger_squeeze", "sar_reversal"]

    def test_registry(self):
        names = st.list_strategies()
        for n in self.NAMES:
            self.assertIn(n, names)
        self.assertEqual(len(names), 9)
        with self.assertRaises(KeyError):
            st.get_strategy("no_such_strategy")
        self.assertEqual(st.get_strategy("ICHIMOKU-TREND").name,
                         "IchimokuTrend")

    def test_frame_contract(self):
        df = ta_data.make_synthetic(800, seed=21)
        for n in self.NAMES:
            with self.subTest(strategy=n):
                f = st.get_strategy(n).generate_signals(df)
                self.assertEqual(set(f.columns),
                                 {"signal", "confidence", "gate"})
                self.assertEqual(len(f), len(df))
                self.assertTrue(f["signal"].isin([-1.0, 0.0, 1.0]).all())
                self.assertTrue(f["confidence"].between(0, 1).all())
                self.assertTrue(f["gate"].between(0, 1).all())
                self.assertFalse(f.isna().any().any())

    def test_short_data_zeros(self):
        df = ta_data.make_synthetic(5, seed=1, regimes=False)
        for n in self.NAMES:
            f = st.get_strategy(n).generate_signals(df)
            self.assertTrue((f["signal"] == 0).all(), n)

    def test_params_override(self):
        s = st.get_strategy("vwap_bounce", entry_z=3.0)
        self.assertEqual(s.params["entry_z"], 3.0)
        s2 = st.get_strategy("vwap_bounce", bogus=1)
        self.assertNotIn("bogus", s2.params)


class IchimokuTrendTests(unittest.TestCase):
    def test_uptrend_goes_long(self):
        closes = list(100 + np.cumsum(np.full(300, 0.15)
                                      + np.random.default_rng(4).normal(
                                          0, 0.2, 300)))
        df = _bars(closes, highs=[c + 0.4 for c in closes],
                   lows=[c - 0.4 for c in closes])
        f = st.get_strategy("ichimoku_trend").generate_signals(df)
        tail = f["signal"].iloc[-100:]
        self.assertGreater((tail > 0).mean(), 0.8)

    def test_downtrend_goes_short(self):
        closes = list(100 + np.cumsum(np.full(300, -0.15)
                                      + np.random.default_rng(5).normal(
                                          0, 0.2, 300)))
        df = _bars(closes, highs=[c + 0.4 for c in closes],
                   lows=[c - 0.4 for c in closes])
        f = st.get_strategy("ichimoku_trend").generate_signals(df)
        tail = f["signal"].iloc[-100:]
        self.assertGreater((tail < 0).mean(), 0.8)


class VwapBounceTests(unittest.TestCase):
    def test_fades_deviation_with_hysteresis(self):
        # Flat day at 100, then price pinned at 110: VWAP trails below,
        # z exceeds entry -> short until price returns near VWAP.
        n = 120
        closes = [100.0] * 60 + [110.0] * 60
        df = _bars(closes, volume=1000.0)
        f = st.get_strategy("vwap_bounce", entry_z=1.5,
                            exit_z=0.4).generate_signals(df)
        sig = f["signal"]
        self.assertEqual(sig.iloc[70], -1.0)  # entered the fade
        # Hysteresis: still short one bar later (z not yet back under exit).
        self.assertEqual(sig.iloc[71], -1.0)
        # Confidence is on while the position is on.
        self.assertGreater(f["confidence"].iloc[70], 0.0)

    def test_flat_market_stays_out(self):
        df = _bars([100.0] * 200, volume=1000.0)
        f = st.get_strategy("vwap_bounce").generate_signals(df)
        self.assertTrue((f["signal"] == 0).all())


class RsiDivergenceTests(unittest.TestCase):
    def _crafted(self):
        rng = np.random.default_rng(1)
        base = 100 + rng.normal(0, 0.15, 150)
        closes = np.concatenate([base, np.linspace(100, 90, 20),
                                 np.linspace(90, 95, 10),
                                 np.linspace(95, 88, 25)])
        return _bars(closes.tolist(),
                     highs=(closes + 0.3).tolist(),
                     lows=(closes - 0.3).tolist())

    def test_bullish_divergence_detected(self):
        df = self._crafted()
        f = st.get_strategy("rsi_divergence").generate_signals(df)
        # Second leg (bars 180-204) is the lower-low / higher-RSI-low.
        self.assertGreater((f["signal"].iloc[180:210] > 0).sum(), 0)
        # Flat noisy base has no qualifying divergence.
        self.assertEqual((f["signal"].iloc[60:150] != 0).sum(), 0)
        self.assertTrue(f["confidence"].iloc[180:210].max() > 0)

    def test_bearish_mirror(self):
        df = self._crafted()
        rev = df.iloc[::-1].reset_index(drop=True)
        f = st.get_strategy("rsi_divergence").generate_signals(rev)
        # Reversed series turns the bullish pattern into a bearish one;
        # detection needs a full lookback window, so it starts at bar 60.
        self.assertLess(f["signal"].iloc[60:120].min(), 0.0)
        self.assertEqual((f["signal"].iloc[:60] != 0).sum(), 0)


class BollingerSqueezeTests(unittest.TestCase):
    def test_squeeze_then_breakout_fires(self):
        # Volatile history -> calm compression (the squeeze) -> sharp break.
        rng = np.random.default_rng(8)
        volatile = 100 + np.cumsum(rng.normal(0, 0.5, 250))
        calm = volatile[-1] + rng.normal(0, 0.05, 120)
        ramp = calm[-1] + np.cumsum(np.full(20, 0.4))
        closes = np.concatenate([volatile, calm, ramp])
        df = _bars(closes.tolist(),
                   highs=(closes + 0.1).tolist(),
                   lows=(closes - 0.1).tolist())
        f = st.get_strategy("bollinger_squeeze").generate_signals(df)
        burst_at = 370
        # The squeeze-release breakout must be caught in the burst zone.
        self.assertGreater((f["signal"].iloc[burst_at:] > 0).sum(), 0)

    def test_trades_on_synthetic(self):
        df = ta_data.make_synthetic(2000, seed=3)
        f = st.get_strategy("bollinger_squeeze").generate_signals(df)
        score = st.quick_score(f, df["close"])
        self.assertGreaterEqual(score["trades"], 5)


class SarReversalTests(unittest.TestCase):
    def test_uptrend_long(self):
        closes = list(np.arange(100.0, 160.0))
        df = _bars(closes, highs=[c + 0.5 for c in closes],
                   lows=[c - 0.5 for c in closes])
        f = st.get_strategy("sar_reversal").generate_signals(df)
        self.assertTrue((f["signal"].iloc[-30:] == 1.0).all())

    def test_v_shape_flips_sign(self):
        dn = list(np.arange(130.0, 115.0, -0.5))
        up = list(np.arange(115.0, 130.0, 0.5))
        closes = dn + up
        df = _bars(closes, highs=[c + 0.3 for c in closes],
                   lows=[c - 0.3 for c in closes])
        f = st.get_strategy("sar_reversal").generate_signals(df)
        self.assertLess(f["signal"].iloc[10], 0.0)
        self.assertGreater(f["signal"].iloc[-1], 0.0)

    def test_runs_in_zoo(self):
        df = ta_data.make_synthetic(500, seed=12)
        frames = st.run_zoo(st.list_strategies(), df)
        self.assertEqual(len(frames), 9)
        ranked = st.rank_strategies(frames, df["close"])
        self.assertIn("score", ranked.columns)


# ── feeds ──────────────────────────────────────────────────────────────

class _FakeStatus:
    def __init__(self, connected):
        self.connected = connected


class _FakeBinance:
    id = "binance"

    def __init__(self, connected=True):
        self._connected = connected
        self.calls = []

    def status(self):
        return _FakeStatus(self._connected)

    def _public(self, method, path, params):
        self.calls.append((method, path, params))
        base = 1700000000000
        return [
            [base + i * 3600000, "100.0", "101.0", "99.0", "100.5",
             "12.5", base + i * 3600000 + 3599999, "0", 0, "0", "0", "0"]
            for i in range(params["limit"])
        ]


class _FakeCoinbase:
    id = "coinbase"

    def __init__(self, connected=True):
        self._connected = connected
        self.calls = []

    def status(self):
        return _FakeStatus(self._connected)

    def _api(self, method, path, *, params=None, **kw):
        self.calls.append((method, path, params))
        gran = params["granularity"]
        n = min(300, int((params["end"] - params["start"]) / gran))
        return {"candles": [
            {"start": params["end"] - i * gran, "low": "99.0",
             "high": "101.0", "open": "100.0", "close": "100.5",
             "volume": "7.5"}
            for i in range(n)
        ]}


class FeedsTests(unittest.TestCase):
    def test_unknown_source(self):
        with self.assertRaises(ValueError):
            feeds.fetch_ohlcv("kraken", "BTCUSDT")

    def test_needs_vault_or_connector(self):
        with self.assertRaises(ValueError):
            feeds.fetch_ohlcv("binance", "BTCUSDT")

    def test_unconnected_fails_fast(self):
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            feeds.fetch_binance("BTCUSDT",
                                connector=_FakeBinance(connected=False))
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            feeds.fetch_coinbase("BTC-USD",
                                 connector=_FakeCoinbase(connected=False))

    def test_binance_shapes_frame(self):
        fake = _FakeBinance()
        df = feeds.fetch_binance("BTCUSDT", "1h", 10, connector=fake)
        self.assertEqual(list(df.columns),
                         ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 10)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertAlmostEqual(df["close"].iloc[0], 100.5)
        self.assertAlmostEqual(df["volume"].iloc[-1], 12.5)
        method, path, params = fake.calls[0]
        self.assertEqual(path, "/api/v3/klines")
        self.assertEqual(params["symbol"], "BTCUSDT")
        self.assertEqual(params["interval"], "1h")

    def test_binance_bad_interval(self):
        with self.assertRaises(ValueError):
            feeds.fetch_binance("BTCUSDT", "9h", connector=_FakeBinance())

    def test_binance_empty_symbol(self):
        with self.assertRaises(ValueError):
            feeds.fetch_binance("", connector=_FakeBinance())

    def test_coinbase_shapes_frame_and_normalizes_symbol(self):
        fake = _FakeCoinbase()
        df = feeds.fetch_coinbase("BTCUSDT", "1h", 50, connector=fake)
        self.assertEqual(list(df.columns),
                         ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 50)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertTrue((df.index.to_series().diff().dropna()
                         > pd.Timedelta(0)).all())
        self.assertAlmostEqual(df["open"].iloc[0], 100.0)
        method, path, params = fake.calls[0]
        self.assertIn("/market/products/BTC-USD/candles", path)
        self.assertEqual(params["granularity"], 3600)

    def test_coinbase_paginates_past_300(self):
        fake = _FakeCoinbase()
        df = feeds.fetch_coinbase("BTC-USD", "1h", 400, connector=fake)
        self.assertEqual(len(df), 400)
        self.assertEqual(len(fake.calls), 2)

    def test_coinbase_bad_symbol(self):
        with self.assertRaises(ValueError):
            feeds.fetch_coinbase("???", connector=_FakeCoinbase())

    def test_coinbase_bad_interval(self):
        with self.assertRaises(ValueError):
            feeds.fetch_coinbase("BTC-USD", "2h",
                                 connector=_FakeCoinbase())

    def test_dispatch(self):
        df = feeds.fetch_ohlcv("binance", "BTCUSDT", limit=5,
                               connector=_FakeBinance())
        self.assertEqual(len(df), 5)
        df = feeds.fetch_ohlcv("coinbase", "BTC-USD", limit=5,
                               connector=_FakeCoinbase())
        self.assertEqual(len(df), 5)


if __name__ == "__main__":
    unittest.main()
