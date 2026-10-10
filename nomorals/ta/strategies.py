"""Canonical strategy zoo: a small hand-written set, honestly named.

Sentinel.py shipped ~120 auto-generated strategy permutations
(``StratRailAlpha`` … ``StratCipherSigma``) differing only in default
parameters — codegen filler with no evidence of edge. This module keeps the
valuable part (the ``signal`` / ``confidence`` / ``gate`` frame contract and
the scoring/ranking math from ``sentinel/strategies/base.py``) and replaces
the zoo with four canonical, independently understandable strategies:

- ``trend_follow`` — EMA-rail trend riding with ATR-normalized confidence
- ``mean_reversion`` — z-score fade with hysteresis and squeeze guard
- ``breakout`` — Donchian breakout with ADX confirmation gate
- ``momentum`` — RSI + MACD agreement, ADX-gated
- ``ichimoku_trend`` — ride price above/below the Ichimoku cloud
- ``vwap_bounce`` — fade ATR-normalized deviations from session VWAP
- ``rsi_divergence`` — RSI/price divergence reversals (swing confirmation)
- ``bollinger_squeeze`` — volatility-squeeze release breakouts
- ``sar_reversal`` — Parabolic SAR regime flips with ATR-scaled confidence

Each is deterministic, parameter-overridable, and scored the same way.
"""

from __future__ import annotations

import logging

from .indicators import adx as _adx
from .indicators import (bollinger, connors_rsi, donchian, heikin_ashi,
                         ichimoku, keltner)
from .indicators import macd as _macd
from .indicators import stoch_rsi as _stoch_rsi
from .indicators import supertrend as _supertrend


# ── lazy optional deps ──────────────────────────────────────────────────
# numpy/pandas are optional. The package imports without them; functions
# that need them raise TAError with a clear install hint.

class TAError(Exception):
    """Raised when a TA operation cannot be completed."""

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise TAError("numpy is required for this operation: pip install nomorals[ta]")


def _require_pandas() -> None:
    if not _HAS_PANDAS:
        raise TAError("pandas is required for this operation: pip install nomorals[ta]")


from .indicators import psar, rsi as _rsi, vwap
from .math import atr, ema, ensure_ohlcv, rolling_zscore, sharpe, sma


_log = logging.getLogger(__name__)

__all__ = [
    "BaseStrategy",
    "TrendFollow",
    "MeanReversion",
    "Breakout",
    "Momentum",
    "IchimokuTrend",
    "VwapBounce",
    "RsiDivergence",
    "BollingerSqueeze",
    "SarReversal",
    # ── sweep additions ──
    "SupertrendTrend",
    "KeltnerBreakout",
    "MacdCross",
    "ConnorsRsi2",
    "StochCross",
    "HeikinAshiTrend",
    "PatternConfluence",
    "STRATEGIES",
    "list_strategies",
    "get_strategy",
    "run_zoo",
    "quick_score",
    "rank_strategies",
    "optimize_params",
]


def _frame(signal: _np.ndarray, confidence: _np.ndarray, gate: _np.ndarray,
           index: _pd.Index) -> _pd.DataFrame:
    """Build a validated signal frame."""
    sig = _pd.Series(_np.sign(_np.asarray(signal, dtype=float)), index=index)
    conf = _pd.Series(_np.asarray(confidence, dtype=float), index=index
                     ).fillna(0.0).clip(0.0, 1.0)
    gt = _pd.Series(_np.asarray(gate, dtype=float), index=index
                   ).fillna(1.0).clip(0.0, 1.0)
    return _pd.DataFrame({"signal": sig, "confidence": conf, "gate": gt},
                        index=index)


class BaseStrategy:
    """Contract every strategy honors: params, generate_signals, describe."""

    name = "BaseStrategy"
    kind = "custom"
    default_params: dict = {}

    def __init__(self, **overrides):
        self.params = dict(self.default_params)
        for k, v in overrides.items():
            if k in self.params:
                self.params[k] = v

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "params": dict(self.params)}


class TrendFollow(BaseStrategy):
    """Ride EMA-rail trends; momentum must confirm the rail direction.

    Signal ±1 while the ATR-normalized rail exceeds threshold; 3-bar
    momentum scales confidence rather than vetoing the signal outright
    (a hard veto goes flat on any noisy bar). The gate throttles when
    price is overstretched from the slow EMA.
    """

    name = "TrendFollow"
    kind = "trend"
    default_params = {"fast": 12, "slow": 26, "mom": 3, "rail_thr": 0.10,
                      "stretch_cap": 5.0}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        close = df["close"].astype(float)
        fast = ema(close, int(p["fast"]))
        slow = ema(close, int(p["slow"]))
        a = atr(df, 14) + 1e-9
        rail = (fast - slow) / a
        mom = _np.sign(close.pct_change(int(p["mom"])).fillna(0.0))
        thr = float(p["rail_thr"])
        raw = _np.where(rail > thr, 1.0, _np.where(rail < -thr, -1.0, 0.0))
        signal = raw
        agree = (mom == raw) | (raw == 0.0)
        confidence = _np.clip(_np.abs(rail) / (thr * 4.0 + 1e-9), 0.0, 1.0)
        confidence = _np.where(agree, confidence, confidence * 0.5)
        stretch = (close - slow) / a
        gate = _np.clip(1.0 - _np.abs(stretch) / float(p["stretch_cap"]), 0.0, 1.0)
        return _frame(signal, confidence, gate, df.index)


class MeanReversion(BaseStrategy):
    """Fade z-score extremes; hold the fade until mean reversion completes.

    Signal opposes the z-score past ``entry_z`` and holds (hysteresis) until
    ``|z| < exit_z``. The gate suppresses fading inside volatility squeezes,
    where extremes tend to expand instead of revert.
    """

    name = "MeanReversion"
    kind = "meanrev"
    default_params = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        close = df["close"].astype(float)
        z = rolling_zscore(close, int(p["lookback"])).to_numpy(dtype=float)
        entry_z, exit_z = float(p["entry_z"]), float(p["exit_z"])
        signal = _np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            if state == 0.0 and abs(z[i]) >= entry_z:
                state = -float(_np.sign(z[i]))
            elif state != 0.0 and abs(z[i]) < exit_z:
                state = 0.0
            signal[i] = state
        confidence = _np.clip(_np.abs(z) / (entry_z * 1.5), 0.0, 1.0)
        width = bollinger(df, int(p["lookback"]))["width"]
        wq = width.rolling(200, min_periods=20).rank(pct=True).fillna(0.5)
        gate = _np.clip(wq.to_numpy(dtype=float) / 0.25, 0.0, 1.0)
        return _frame(signal, confidence, gate, df.index)


class Breakout(BaseStrategy):
    """Trade Donchian breakouts; exit on midline cross or opposite breakout.

    Enters ±1 on a close outside the channel, holds until price crosses the
    midline (or breaks the other way). Confidence scales with ATR-normalized
    penetration; the gate requires ADX trend strength so range chop is skipped.
    """

    name = "Breakout"
    kind = "breakout"
    default_params = {"period": 20, "adx_period": 14, "adx_min": 18.0}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        close = df["close"].astype(float)
        ch = donchian(df, int(p["period"]))
        a = atr(df, 14) + 1e-9
        upper = ch["upper"].to_numpy(dtype=float)
        lower = ch["lower"].to_numpy(dtype=float)
        mid = ch["mid"].to_numpy(dtype=float)
        px = close.to_numpy(dtype=float)
        signal = _np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            if state == 0.0:
                if px[i] > upper[i]:
                    state = 1.0
                elif px[i] < lower[i]:
                    state = -1.0
            elif state > 0 and (px[i] < mid[i] or px[i] < lower[i]):
                state = -1.0 if px[i] < lower[i] else 0.0
            elif state < 0 and (px[i] > mid[i] or px[i] > upper[i]):
                state = 1.0 if px[i] > upper[i] else 0.0
            signal[i] = state
        pen = _np.where(signal > 0, (px - upper) / a,
                       _np.where(signal < 0, (lower - px) / a, 0.0))
        confidence = _np.clip(_np.abs(pen) / 1.0, 0.15, 1.0) * (_np.abs(signal) > 0)
        adx_v = _adx(df, int(p["adx_period"]))["adx"].to_numpy(dtype=float)
        gate = _np.clip(adx_v / float(p["adx_min"]), 0.0, 1.0)
        return _frame(signal, confidence, gate, df.index)


class Momentum(BaseStrategy):
    """RSI + MACD agreement, ADX-gated.

    Long when RSI is bid (>55) and the MACD histogram is positive; short on
    the mirror. Both must agree — single-indicator whipsaws are ignored.
    Confidence blends RSI extremity with histogram strength.
    """

    name = "Momentum"
    kind = "momentum"
    default_params = {"rsi_period": 14, "rsi_hi": 55.0, "rsi_lo": 45.0,
                      "adx_period": 14, "adx_min": 20.0}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        rsi_v = _rsi(df, int(p["rsi_period"])).to_numpy(dtype=float)
        hist = _macd(df)["hist"].to_numpy(dtype=float)
        bull = (rsi_v > float(p["rsi_hi"])) & (hist > 0)
        bear = (rsi_v < float(p["rsi_lo"])) & (hist < 0)
        signal = _np.where(bull, 1.0, _np.where(bear, -1.0, 0.0))
        hist_vol = _pd.Series(_np.abs(hist)).rolling(50, min_periods=10).mean(
        ).bfill().fillna(1e-9).to_numpy(dtype=float)
        confidence = (
            0.6 * _np.clip(_np.abs(rsi_v - 50.0) / 50.0, 0.0, 1.0)
            + 0.4 * _np.clip(_np.abs(hist) / (hist_vol + 1e-12), 0.0, 1.0)
        ) * (_np.abs(signal) > 0)
        adx_v = _adx(df, int(p["adx_period"]))["adx"].to_numpy(dtype=float)
        gate = _np.clip(adx_v / float(p["adx_min"]), 0.0, 1.0)
        return _frame(signal, confidence, gate, df.index)


class IchimokuTrend(BaseStrategy):
    """Ride price above/below the Ichimoku cloud.

    Long while price sits above the cloud with tenkan above kijun; short
    on the mirror. Cloud distance, ATR-normalized, scales confidence so
    deep cloud breaks size up and thin whipsaws stay small.
    """

    name = "IchimokuTrend"
    kind = "trend"
    default_params = {"tenkan": 9, "kijun": 26, "senkou_b": 52,
                      "displacement": 26, "conf_scale": 3.0}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        ich = ichimoku(df, int(p["tenkan"]), int(p["kijun"]),
                       int(p["senkou_b"]), int(p["displacement"]))
        close = df["close"].astype(float)
        cloud_top = _np.maximum(ich["senkou_a"], ich["senkou_b"])
        cloud_bot = _np.minimum(ich["senkou_a"], ich["senkou_b"])
        tenkan_s = ich["tenkan"].to_numpy(dtype=float)
        kijun_s = ich["kijun"].to_numpy(dtype=float)
        px = close.to_numpy(dtype=float)
        bull = (px > cloud_top.to_numpy(dtype=float)) & (tenkan_s > kijun_s)
        bear = (px < cloud_bot.to_numpy(dtype=float)) & (tenkan_s < kijun_s)
        signal = _np.where(bull, 1.0, _np.where(bear, -1.0, 0.0))
        cloud_mid = ((cloud_top + cloud_bot) / 2.0).to_numpy(dtype=float)
        a = atr(df, 14).to_numpy(dtype=float) + 1e-9
        dist = _np.abs(px - cloud_mid) / a
        confidence = _np.clip(dist / float(p["conf_scale"]), 0.0, 1.0) \
            * (_np.abs(signal) > 0)
        gate = _np.ones(len(df))
        return _frame(signal, confidence, gate, df.index)


class VwapBounce(BaseStrategy):
    """Fade ATR-normalized deviations from session VWAP.

    Price tends to snap back toward the day's volume-weighted mean, so the
    strategy sells strength (|z| > entry_z) and buys weakness, holding the
    fade with hysteresis until price returns near VWAP (|z| < exit_z).
    Confidence scales with the extremity of the deviation; the gate
    throttles the fade when ADX says the market is strongly trending —
    fading a raging trend's VWAP deviation is how fades get run over.
    """

    name = "VwapBounce"
    kind = "meanrev"
    default_params = {"entry_z": 1.5, "exit_z": 0.4, "adx_period": 14,
                      "adx_lo": 20.0, "adx_hi": 35.0}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        v = vwap(df).to_numpy(dtype=float)
        px = df["close"].astype(float).to_numpy(dtype=float)
        a = atr(df, 14).to_numpy(dtype=float) + 1e-9
        z = (px - v) / a
        entry_z, exit_z = float(p["entry_z"]), float(p["exit_z"])
        signal = _np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            if state == 0.0 and abs(z[i]) >= entry_z:
                state = -float(_np.sign(z[i]))
            elif state != 0.0 and abs(z[i]) < exit_z:
                state = 0.0
            signal[i] = state
        confidence = _np.clip(_np.abs(z) / (entry_z * 1.5), 0.0, 1.0) \
            * (_np.abs(signal) > 0)
        adx_v = _adx(df, int(p["adx_period"]))["adx"].to_numpy(dtype=float)
        lo, hi = float(p["adx_lo"]), float(p["adx_hi"])
        gate = _np.clip((hi - adx_v) / (hi - lo + 1e-12), 0.15, 1.0)
        return _frame(signal, confidence, gate, df.index)


class RsiDivergence(BaseStrategy):
    """RSI/price divergence reversals.

    Scans each bar's trailing window for the textbook pattern: price makes
    a lower low (higher high) while RSI makes a higher low (lower high).
    Divergences only count at RSI extremes (oversold for bullish,
    overbought for bearish) — mid-range RSI wiggles in a trend are noise,
    not exhaustion. A confirmed divergence emits ±1 for ``hold`` bars.
    This is a sparse, high-conviction pattern — most bars carry no signal
    by design.
    """

    name = "RsiDivergence"
    kind = "reversal"
    default_params = {"rsi_period": 14, "lookback": 60, "hold": 5,
                      "min_gap": 10, "min_rsi_delta": 4.0,
                      "rsi_lo": 40.0, "rsi_hi": 60.0}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        lookback = max(20, int(p["lookback"]))
        min_gap = max(3, int(p["min_gap"]))
        hold = max(1, int(p["hold"]))
        min_delta = float(p["min_rsi_delta"])
        rsi_lo, rsi_hi = float(p["rsi_lo"]), float(p["rsi_hi"])
        px = df["close"].astype(float).to_numpy(dtype=float)
        rsi_v = _rsi(df, int(p["rsi_period"])).to_numpy(dtype=float)
        n = len(df)
        raw = _np.zeros(n)
        conf = _np.zeros(n)
        half = lookback // 2
        for i in range(lookback, n):
            w0, w1 = i - lookback, i
            mid = w0 + half
            p1_lo = w0 + int(_np.argmin(px[w0:mid]))
            p2_lo = mid + int(_np.argmin(px[mid:w1]))
            p1_hi = w0 + int(_np.argmax(px[w0:mid]))
            p2_hi = mid + int(_np.argmax(px[mid:w1]))
            bull = (p2_lo - p1_lo >= min_gap and px[p2_lo] < px[p1_lo]
                    and rsi_v[p2_lo] - rsi_v[p1_lo] >= min_delta
                    and rsi_v[p2_lo] <= rsi_lo)
            bear = (p2_hi - p1_hi >= min_gap and px[p2_hi] > px[p1_hi]
                    and rsi_v[p1_hi] - rsi_v[p2_hi] >= min_delta
                    and rsi_v[p2_hi] >= rsi_hi)
            if bull:
                raw[i] = 1.0
                conf[i] = _np.clip(
                    (rsi_v[p2_lo] - rsi_v[p1_lo]) / 10.0, 0.25, 1.0)
            elif bear:
                raw[i] = -1.0
                conf[i] = _np.clip(
                    (rsi_v[p1_hi] - rsi_v[p2_hi]) / 10.0, 0.25, 1.0)
        signal = _np.zeros(n)
        confidence = _np.zeros(n)
        for i in range(n):
            if raw[i] != 0.0:
                end = min(n, i + hold)
                signal[i:end] = raw[i]
                confidence[i:end] = conf[i]
        gate = _np.ones(n)
        return _frame(signal, confidence, gate, df.index)


class BollingerSqueeze(BaseStrategy):
    """Volatility-squeeze release breakouts.

    A squeeze is when bandwidth sits in the bottom ``squeeze_pct`` of its
    trailing range — energy compressing. When a squeeze was present within
    the last ``confirm`` bars and price breaks the band, follow the break;
    exit on a midline cross or after ``max_hold`` bars.
    """

    name = "BollingerSqueeze"
    kind = "breakout"
    default_params = {"period": 20, "mult": 2.0, "squeeze_pct": 0.20,
                      "rank_window": 200, "confirm": 10, "max_hold": 30}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        bb = bollinger(df, int(p["period"]), float(p["mult"]))
        width = bb["width"]
        rw = max(50, int(p["rank_window"]))
        # Smooth the width before ranking: raw bandwidth jitters with
        # micro-variation, which makes the percentile rank meaningless in
        # uniformly calm markets.
        smooth = width.rolling(5, min_periods=1).mean()
        rank = smooth.rolling(rw, min_periods=max(20, rw // 4)).rank(
            pct=True).fillna(0.5)
        squeezed = (rank <= float(p["squeeze_pct"])).to_numpy(dtype=float)
        confirm = max(1, int(p["confirm"]))
        recent_squeeze = _pd.Series(squeezed, index=df.index).rolling(
            confirm, min_periods=1).max().to_numpy(dtype=float) > 0
        px = df["close"].astype(float).to_numpy(dtype=float)
        upper = bb["upper"].to_numpy(dtype=float)
        lower = bb["lower"].to_numpy(dtype=float)
        mid = bb["mid"].to_numpy(dtype=float)
        signal = _np.zeros(len(df))
        state = 0.0
        held = 0
        for i in range(len(df)):
            if state == 0.0:
                if recent_squeeze[i] and px[i] > upper[i]:
                    state, held = 1.0, 0
                elif recent_squeeze[i] and px[i] < lower[i]:
                    state, held = -1.0, 0
            else:
                held += 1
                if held >= int(p["max_hold"]) \
                        or (state > 0 and px[i] < mid[i]) \
                        or (state < 0 and px[i] > mid[i]):
                    state, held = 0.0, 0
            signal[i] = state
        a = atr(df, 14).to_numpy(dtype=float) + 1e-9
        pen = _np.where(signal > 0, (px - mid) / a,
                       _np.where(signal < 0, (mid - px) / a, 0.0))
        confidence = _np.clip(_np.abs(pen) / 1.0, 0.15, 1.0) \
            * (_np.abs(signal) > 0)
        gate = _np.ones(len(df))
        return _frame(signal, confidence, gate, df.index)


class SarReversal(BaseStrategy):
    """Parabolic SAR regime flips: long above the SAR, short below.

    The SAR is a trailing stop by construction, so the flip is the trade:
    signal = sign(close - SAR) every bar. Confidence scales with the
    SAR distance in ATR units — wide separation means a strong regime.
    """

    name = "SarReversal"
    kind = "trend"
    default_params = {"accel": 0.02, "max_accel": 0.20}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        sar = psar(df, float(p["accel"]), float(p["max_accel"])
                   ).to_numpy(dtype=float)
        px = df["close"].astype(float).to_numpy(dtype=float)
        signal = _np.sign(px - sar)
        a = atr(df, 14).to_numpy(dtype=float) + 1e-9
        confidence = _np.clip(_np.abs(px - sar) / (2.0 * a), 0.0, 1.0)
        gate = _np.ones(len(df))
        return _frame(signal, confidence, gate, df.index)


class SupertrendTrend(BaseStrategy):
    """Ride the Supertrend line: long above, short below.

    The retail-standard ATR trailing system (Olivier Seban). Signal flips
    only when price closes through the line — no whipsaw on wicks.
    Confidence scales with ATR-normalized distance from the line; the
    gate throttles when ADX says there is no trend to ride.
    """

    name = "SupertrendTrend"
    kind = "trend"
    default_params = {"period": 10, "mult": 3.0, "adx_period": 14,
                      "adx_min": 15.0}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        st = _supertrend(df, int(p["period"]), float(p["mult"]))
        signal = st["direction"].to_numpy(dtype=float)
        px = df["close"].astype(float).to_numpy(dtype=float)
        line = st["line"].to_numpy(dtype=float)
        a = atr(df, 14).to_numpy(dtype=float) + 1e-9
        confidence = _np.clip(_np.abs(px - line) / (2.0 * a), 0.0, 1.0)
        adx_v = _adx(df, int(p["adx_period"]))["adx"].to_numpy(dtype=float)
        gate = _np.clip(adx_v / float(p["adx_min"]), 0.0, 1.0)
        return _frame(signal, confidence, gate, df.index)


class KeltnerBreakout(BaseStrategy):
    """Keltner channel breakouts with volume confirmation.

    Enters on a close outside the channel, exits on a midline cross.
    The gate requires volume above its rolling median — breakouts on
    thin volume are the ones that fail.
    """

    name = "KeltnerBreakout"
    kind = "breakout"
    default_params = {"period": 20, "atr_period": 10, "mult": 2.0,
                      "vol_window": 20}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        kc = keltner(df, int(p["period"]), int(p["atr_period"]),
                     float(p["mult"]))
        px = df["close"].astype(float).to_numpy(dtype=float)
        upper = kc["upper"].to_numpy(dtype=float)
        lower = kc["lower"].to_numpy(dtype=float)
        mid = kc["mid"].to_numpy(dtype=float)
        signal = _np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            if state == 0.0:
                if px[i] > upper[i]:
                    state = 1.0
                elif px[i] < lower[i]:
                    state = -1.0
            elif state > 0 and px[i] < mid[i]:
                state = 0.0
            elif state < 0 and px[i] > mid[i]:
                state = 0.0
            signal[i] = state
        a = atr(df, 14).to_numpy(dtype=float) + 1e-9
        pen = _np.where(signal > 0, (px - mid) / a,
                       _np.where(signal < 0, (mid - px) / a, 0.0))
        confidence = _np.clip(_np.abs(pen) / 1.0, 0.15, 1.0) \
            * (_np.abs(signal) > 0)
        vol = df["volume"].astype(float)
        vol_med = vol.rolling(int(p["vol_window"]),
                             min_periods=5).median().bfill()
        gate = _np.clip((vol / (vol_med + 1e-12)).to_numpy(dtype=float)
                        / 1.5, 0.0, 1.0)
        return _frame(signal, confidence, gate, df.index)


class MacdCross(BaseStrategy):
    """Classic MACD line/signal cross with histogram momentum filter.

    Long on bullish cross while the histogram confirms momentum; short
    on the mirror. Holds the position until the opposite cross — the
    textbook system, with confidence from cross strength.
    """

    name = "MacdCross"
    kind = "momentum"
    default_params = {"fast": 12, "slow": 26, "signal": 9}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        m = _macd(df, int(p["fast"]), int(p["slow"]), int(p["signal"]))
        line = m["macd"].to_numpy(dtype=float)
        sigl = m["signal"].to_numpy(dtype=float)
        hist = m["hist"].to_numpy(dtype=float)
        cross_up = (line > sigl) & (_np.roll(line, 1) <= _np.roll(sigl, 1))
        cross_dn = (line < sigl) & (_np.roll(line, 1) >= _np.roll(sigl, 1))
        cross_up[0] = cross_dn[0] = False
        signal = _np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            if cross_up[i]:
                state = 1.0
            elif cross_dn[i]:
                state = -1.0
            signal[i] = state
        hist_vol = _pd.Series(_np.abs(hist)).rolling(
            50, min_periods=10).mean().bfill().fillna(1e-9
                                                     ).to_numpy(dtype=float)
        confidence = _np.clip(_np.abs(hist) / (hist_vol + 1e-12), 0.0, 1.0) \
            * (_np.abs(signal) > 0)
        gate = _np.ones(len(df))
        return _frame(signal, confidence, gate, df.index)


class ConnorsRsi2(BaseStrategy):
    """Connors RSI(2)-style washed-out fade (Larry Connors).

    Buys when Connors RSI < ``oversold`` (default 10 — the classic
    washed-out print), exits when it recovers above ``exit``. Above the
    200-day SMA only for longs (the Connors trend filter); shorts are
    the mirror below it. Sparse by design.
    """

    name = "ConnorsRsi2"
    kind = "meanrev"
    default_params = {"oversold": 10.0, "overbought": 90.0, "exit": 50.0,
                      "trend_ma": 200}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        crsi = connors_rsi(df).to_numpy(dtype=float)
        px = df["close"].astype(float)
        trend = sma(px, int(p["trend_ma"])).to_numpy(dtype=float)
        pxv = px.to_numpy(dtype=float)
        os_, ob, ex = float(p["oversold"]), float(p["overbought"]), float(
            p["exit"])
        signal = _np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            if state == 0.0:
                if crsi[i] < os_ and pxv[i] > trend[i]:
                    state = 1.0
                elif crsi[i] > ob and pxv[i] < trend[i]:
                    state = -1.0
            elif state > 0 and crsi[i] > ex:
                state = 0.0
            elif state < 0 and crsi[i] < 100.0 - ex:
                state = 0.0
            signal[i] = state
        confidence = _np.clip(_np.where(
            signal > 0, (os_ - crsi) / os_, (crsi - ob) / (100.0 - ob)),
            0.0, 1.0) * (_np.abs(signal) > 0)
        gate = _np.ones(len(df))
        return _frame(signal, confidence, gate, df.index)


class StochCross(BaseStrategy):
    """Stochastic %K/%D cross in extreme zones.

    Long on %K crossing up through %D while both are oversold (<20);
    short on the mirror above 80. Mid-range crosses are chop — ignored.
    """

    name = "StochCross"
    kind = "momentum"
    default_params = {"k": 14, "d": 3, "oversold": 20.0, "overbought": 80.0}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        p = self.params
        sr = _stoch_rsi(df, k=int(p["k"]), d=int(p["d"]))
        k = sr["stochrsi_k"].to_numpy(dtype=float)
        d = sr["stochrsi_d"].to_numpy(dtype=float)
        os_, ob = float(p["oversold"]), float(p["overbought"])
        cross_up = (k > d) & (_np.roll(k, 1) <= _np.roll(d, 1)) & (k < os_)
        cross_dn = (k < d) & (_np.roll(k, 1) >= _np.roll(d, 1)) & (k > ob)
        cross_up[0] = cross_dn[0] = False
        signal = _np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            if cross_up[i]:
                state = 1.0
            elif cross_dn[i]:
                state = -1.0
            elif state > 0 and k[i] > 80:
                state = 0.0
            elif state < 0 and k[i] < 20:
                state = 0.0
            signal[i] = state
        confidence = _np.clip(_np.where(
            signal > 0, (os_ - k) / os_, (k - ob) / (100.0 - ob)),
            0.2, 1.0) * (_np.abs(signal) > 0)
        gate = _np.ones(len(df))
        return _frame(signal, confidence, gate, df.index)


class HeikinAshiTrend(BaseStrategy):
    """Trend-following on Heikin-Ashi candles.

    HA candles filter wick noise: long while HA candles are bullish
    without lower wicks (strong trend), exit on the first bearish HA
    candle. The gate requires consecutive same-color HA candles —
    single-candle flips are noise.
    """

    name = "HeikinAshiTrend"
    kind = "trend"
    default_params = {"min_run": 2}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        ha = heikin_ashi(df)
        o = ha["open"].to_numpy(dtype=float)
        c = ha["close"].to_numpy(dtype=float)
        h = ha["high"].to_numpy(dtype=float)
        l = ha["low"].to_numpy(dtype=float)
        bull = c > o
        no_lower_wick = (o - l) <= 0.1 * _np.maximum(h - l, 1e-9)
        no_upper_wick = (h - c) <= 0.1 * _np.maximum(h - l, 1e-9)
        strong_bull = bull & no_lower_wick
        strong_bear = (~bull) & no_upper_wick
        signal = _np.where(strong_bull, 1.0, _np.where(strong_bear, -1.0, 0.0))
        run = _np.zeros(len(df))
        cnt = 0
        for i in range(len(df)):
            cnt = cnt + 1 if signal[i] != 0 and (
                i == 0 or signal[i] == signal[i - 1]) else (
                1 if signal[i] != 0 else 0)
            run[i] = cnt
        gate = _np.clip(run / float(max(1, int(self.params["min_run"]))),
                        0.0, 1.0)
        a = atr(df, 14).to_numpy(dtype=float) + 1e-9
        confidence = _np.clip(_np.abs(c - o) / a, 0.0, 1.0) \
            * (_np.abs(signal) > 0)
        return _frame(signal, confidence, gate, df.index)


class PatternConfluence(BaseStrategy):
    """Candlestick patterns, but only with confluence (the honest version).

    Uses ``patterns.pattern_score`` filtered by trend context: a signal
    fires when the reliability-weighted pattern confluence exceeds
    ``min_score`` AND aligns with the EMA trend. Naked patterns are
    noise — this is the version with edge.
    """

    name = "PatternConfluence"
    kind = "confluence"
    default_params = {"min_score": 0.35, "hold": 3}

    def generate_signals(self, df: _pd.DataFrame) -> _pd.DataFrame:
        from .patterns import detect_all, with_trend_context

        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(_np.zeros(len(df)), _np.zeros(len(df)),
                          _np.ones(len(df)), df.index)
        from .patterns import PATTERN_TIERS

        p = self.params
        sig = detect_all(df)
        ctx = with_trend_context(df, sig)
        tier_w = {"high": 1.0, "medium": 0.6, "low": 0.3}
        w = _np.array([tier_w[PATTERN_TIERS.get(c, "low")]
                       for c in ctx.columns])
        vals = ctx.to_numpy(dtype=float)
        weighted = vals * w
        mass = _np.abs(weighted).sum(axis=1)
        score = (weighted.sum(axis=1) / (mass + 1e-12)
                 * _np.clip(mass, 0, 1))
        score = _np.nan_to_num(score)
        raw = _np.where(score >= float(p["min_score"]), 1.0,
                       _np.where(score <= -float(p["min_score"]), -1.0, 0.0))
        hold = max(1, int(p["hold"]))
        signal = _np.zeros(len(df))
        confidence = _np.zeros(len(df))
        for i in range(len(df)):
            if raw[i] != 0.0:
                end = min(len(df), i + hold)
                signal[i:end] = raw[i]
                confidence[i:end] = min(1.0, abs(float(score[i])))
        gate = _np.ones(len(df))
        return _frame(signal, confidence, gate, df.index)


STRATEGIES: dict[str, type[BaseStrategy]] = {
    "trend_follow": TrendFollow,
    "mean_reversion": MeanReversion,
    "breakout": Breakout,
    "momentum": Momentum,
    "ichimoku_trend": IchimokuTrend,
    "vwap_bounce": VwapBounce,
    "rsi_divergence": RsiDivergence,
    "bollinger_squeeze": BollingerSqueeze,
    "sar_reversal": SarReversal,
    "supertrend": SupertrendTrend,
    "keltner_breakout": KeltnerBreakout,
    "macd_cross": MacdCross,
    "connors_rsi2": ConnorsRsi2,
    "stoch_cross": StochCross,
    "heikin_ashi_trend": HeikinAshiTrend,
    "pattern_confluence": PatternConfluence,
}


def list_strategies(kind: str | None = None) -> list[str]:
    """Strategy names, optionally filtered by kind."""
    names = sorted(STRATEGIES)
    if kind is None:
        return names
    return [n for n in names
            if str(STRATEGIES[n].kind).lower() == kind.lower()]


def get_strategy(name: str, **overrides) -> BaseStrategy:
    """Instantiate a strategy by name (KeyError lists what's available)."""
    key = (name or "").strip().lower().replace("-", "_")
    if key not in STRATEGIES:
        raise KeyError(
            f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}")
    return STRATEGIES[key](**overrides)


def run_zoo(names: list[str], df: _pd.DataFrame,
            params: dict | None = None) -> dict[str, _pd.DataFrame]:
    """Run many strategies; failures are isolated per strategy.

    Returns ``{name: signal-frame}`` for the strategies that produced a
    well-formed frame. Unknown names and crashing strategies are skipped.
    """
    params = params or {}
    out: dict[str, _pd.DataFrame] = {}
    for n in names:
        try:
            strat = get_strategy(n, **params.get(n, {}))
            sig = strat.generate_signals(df)
            if set(sig.columns) == {"signal", "confidence", "gate"} \
                    and len(sig) == len(df):
                out[n] = sig
        except Exception as e:
            _log.debug("strategy %s failed: %s", n, e)
            continue
    return out


def quick_score(sig: _pd.DataFrame, close: _pd.Series,
                periods: int = 252) -> dict:
    """Strategy quality from sign returns: Sharpe, hit-rate, turnover."""
    px = close.astype(float)
    rets = px.pct_change().fillna(0.0).to_numpy()
    pos = _np.sign(sig["signal"].to_numpy(dtype=float))
    pos = _np.roll(pos, 1)
    pos[0] = 0.0
    strat_rets = pos * rets
    w = sig["confidence"].to_numpy(dtype=float) * sig["gate"].to_numpy(dtype=float)
    strat_rets = strat_rets * _np.clip(w, 0, 1)
    active = strat_rets[_np.abs(pos) > 0]
    return {
        "sharpe": sharpe(_pd.Series(strat_rets), periods),
        "hit_rate": float(_np.mean(active > 0)) if len(active) else 0.0,
        "turnover": float(_np.mean(_np.abs(_np.diff(pos)) > 0)) if len(pos) > 1 else 0.0,
        "exposure": float(_np.mean(_np.abs(pos))),
        "trades": int(_np.sum(_np.abs(_np.diff(pos)) > 0) if len(pos) > 1 else 0),
    }


def rank_strategies(frames: dict[str, _pd.DataFrame], close: _pd.Series,
                    periods: int = 252, min_trades: int = 5) -> _pd.DataFrame:
    """Rank strategy frames by Sharpe penalized for turnover.

    Returns a DataFrame indexed by strategy name with columns
    ``sharpe, hit_rate, turnover, exposure, trades, score`` (best first).
    Strategies below ``min_trades`` are excluded as unproven.
    """
    rows = []
    for name, sig in frames.items():
        try:
            s = quick_score(sig, close, periods)
            s["name"] = name
            rows.append(s)
        except Exception as e:
            _log.debug("strategy scoring failed for %s: %s", name, e)
            continue
    cols = ["name", "sharpe", "hit_rate", "turnover", "exposure", "trades"]
    if not rows:
        return _pd.DataFrame(columns=cols)
    df = _pd.DataFrame(rows).set_index("name")
    df = df[df["trades"] >= min_trades]
    if df.empty:
        return df
    df["score"] = df["sharpe"] - 2.0 * df["turnover"]
    return df.sort_values("score", ascending=False)


def optimize_params(name: str, df: _pd.DataFrame, grid: dict,
                    metric: str = "sharpe", fee_bps: float = 5.0) -> _pd.DataFrame:
    """TuneTA-lite: grid-search a strategy's params on the vector engine.

    ``grid`` maps param names to lists of values; every combination is
    run through the fast vector backtester and ranked by ``metric``
    (``sharpe`` | ``total_return`` | ``calmar`` | ``profit_factor``...).
    Returns a DataFrame with one row per combination, best first —
    plus ``n_trials`` so the caller can feed it to ``deflated_sharpe``.
    """
    from itertools import product

    from .backtest import VectorBacktester

    df = ensure_ohlcv(df)
    keys = list(grid)
    combos = list(product(*[grid[k] for k in keys]))
    bt = VectorBacktester(fee_bps=fee_bps)
    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            strat = get_strategy(name, **params)
            frame = strat.generate_signals(df)
            pos = _pd.Series(_np.sign(frame["signal"].to_numpy(dtype=float)),
                             index=df.index)
            res = bt.run(df, pos)
            row = {"n_trials": len(combos)}
            row.update({k: v for k, v in params.items()})
            row.update({
                "sharpe": res["sharpe"], "total_return": res["total_return"],
                "max_dd": res["max_dd"], "turnover": res["turnover"],
            })
            rows.append(row)
        except Exception as e:
            _log.debug("optimize %s %s failed: %s", name, params, e)
            continue
    if not rows:
        return _pd.DataFrame()
    out = _pd.DataFrame(rows)
    if metric in out.columns:
        out = out.sort_values(metric, ascending=False)
    return out.reset_index(drop=True)
