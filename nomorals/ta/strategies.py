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
import numpy as np
import pandas as pd

from .indicators import adx as _adx
from .indicators import bollinger, donchian, ichimoku, macd as _macd
from .indicators import psar, rsi as _rsi, vwap
from .math import atr, ema, ensure_ohlcv, rolling_zscore, sharpe


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
    "STRATEGIES",
    "list_strategies",
    "get_strategy",
    "run_zoo",
    "quick_score",
    "rank_strategies",
]


def _frame(signal: np.ndarray, confidence: np.ndarray, gate: np.ndarray,
           index: pd.Index) -> pd.DataFrame:
    """Build a validated signal frame."""
    sig = pd.Series(np.sign(np.asarray(signal, dtype=float)), index=index)
    conf = pd.Series(np.asarray(confidence, dtype=float), index=index
                     ).fillna(0.0).clip(0.0, 1.0)
    gt = pd.Series(np.asarray(gate, dtype=float), index=index
                   ).fillna(1.0).clip(0.0, 1.0)
    return pd.DataFrame({"signal": sig, "confidence": conf, "gate": gt},
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(np.zeros(len(df)), np.zeros(len(df)),
                          np.ones(len(df)), df.index)
        p = self.params
        close = df["close"].astype(float)
        fast = ema(close, int(p["fast"]))
        slow = ema(close, int(p["slow"]))
        a = atr(df, 14) + 1e-9
        rail = (fast - slow) / a
        mom = np.sign(close.pct_change(int(p["mom"])).fillna(0.0))
        thr = float(p["rail_thr"])
        raw = np.where(rail > thr, 1.0, np.where(rail < -thr, -1.0, 0.0))
        signal = raw
        agree = (mom == raw) | (raw == 0.0)
        confidence = np.clip(np.abs(rail) / (thr * 4.0 + 1e-9), 0.0, 1.0)
        confidence = np.where(agree, confidence, confidence * 0.5)
        stretch = (close - slow) / a
        gate = np.clip(1.0 - np.abs(stretch) / float(p["stretch_cap"]), 0.0, 1.0)
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(np.zeros(len(df)), np.zeros(len(df)),
                          np.ones(len(df)), df.index)
        p = self.params
        close = df["close"].astype(float)
        z = rolling_zscore(close, int(p["lookback"])).to_numpy(dtype=float)
        entry_z, exit_z = float(p["entry_z"]), float(p["exit_z"])
        signal = np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            if state == 0.0 and abs(z[i]) >= entry_z:
                state = -float(np.sign(z[i]))
            elif state != 0.0 and abs(z[i]) < exit_z:
                state = 0.0
            signal[i] = state
        confidence = np.clip(np.abs(z) / (entry_z * 1.5), 0.0, 1.0)
        width = bollinger(df, int(p["lookback"]))["width"]
        wq = width.rolling(200, min_periods=20).rank(pct=True).fillna(0.5)
        gate = np.clip(wq.to_numpy(dtype=float) / 0.25, 0.0, 1.0)
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(np.zeros(len(df)), np.zeros(len(df)),
                          np.ones(len(df)), df.index)
        p = self.params
        close = df["close"].astype(float)
        ch = donchian(df, int(p["period"]))
        a = atr(df, 14) + 1e-9
        upper = ch["upper"].to_numpy(dtype=float)
        lower = ch["lower"].to_numpy(dtype=float)
        mid = ch["mid"].to_numpy(dtype=float)
        px = close.to_numpy(dtype=float)
        signal = np.zeros(len(df))
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
        pen = np.where(signal > 0, (px - upper) / a,
                       np.where(signal < 0, (lower - px) / a, 0.0))
        confidence = np.clip(np.abs(pen) / 1.0, 0.15, 1.0) * (np.abs(signal) > 0)
        adx_v = _adx(df, int(p["adx_period"]))["adx"].to_numpy(dtype=float)
        gate = np.clip(adx_v / float(p["adx_min"]), 0.0, 1.0)
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(np.zeros(len(df)), np.zeros(len(df)),
                          np.ones(len(df)), df.index)
        p = self.params
        rsi_v = _rsi(df, int(p["rsi_period"])).to_numpy(dtype=float)
        hist = _macd(df)["hist"].to_numpy(dtype=float)
        bull = (rsi_v > float(p["rsi_hi"])) & (hist > 0)
        bear = (rsi_v < float(p["rsi_lo"])) & (hist < 0)
        signal = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))
        hist_vol = pd.Series(np.abs(hist)).rolling(50, min_periods=10).mean(
        ).bfill().fillna(1e-9).to_numpy(dtype=float)
        confidence = (
            0.6 * np.clip(np.abs(rsi_v - 50.0) / 50.0, 0.0, 1.0)
            + 0.4 * np.clip(np.abs(hist) / (hist_vol + 1e-12), 0.0, 1.0)
        ) * (np.abs(signal) > 0)
        adx_v = _adx(df, int(p["adx_period"]))["adx"].to_numpy(dtype=float)
        gate = np.clip(adx_v / float(p["adx_min"]), 0.0, 1.0)
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(np.zeros(len(df)), np.zeros(len(df)),
                          np.ones(len(df)), df.index)
        p = self.params
        ich = ichimoku(df, int(p["tenkan"]), int(p["kijun"]),
                       int(p["senkou_b"]), int(p["displacement"]))
        close = df["close"].astype(float)
        cloud_top = np.maximum(ich["senkou_a"], ich["senkou_b"])
        cloud_bot = np.minimum(ich["senkou_a"], ich["senkou_b"])
        tenkan_s = ich["tenkan"].to_numpy(dtype=float)
        kijun_s = ich["kijun"].to_numpy(dtype=float)
        px = close.to_numpy(dtype=float)
        bull = (px > cloud_top.to_numpy(dtype=float)) & (tenkan_s > kijun_s)
        bear = (px < cloud_bot.to_numpy(dtype=float)) & (tenkan_s < kijun_s)
        signal = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))
        cloud_mid = ((cloud_top + cloud_bot) / 2.0).to_numpy(dtype=float)
        a = atr(df, 14).to_numpy(dtype=float) + 1e-9
        dist = np.abs(px - cloud_mid) / a
        confidence = np.clip(dist / float(p["conf_scale"]), 0.0, 1.0) \
            * (np.abs(signal) > 0)
        gate = np.ones(len(df))
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(np.zeros(len(df)), np.zeros(len(df)),
                          np.ones(len(df)), df.index)
        p = self.params
        v = vwap(df).to_numpy(dtype=float)
        px = df["close"].astype(float).to_numpy(dtype=float)
        a = atr(df, 14).to_numpy(dtype=float) + 1e-9
        z = (px - v) / a
        entry_z, exit_z = float(p["entry_z"]), float(p["exit_z"])
        signal = np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            if state == 0.0 and abs(z[i]) >= entry_z:
                state = -float(np.sign(z[i]))
            elif state != 0.0 and abs(z[i]) < exit_z:
                state = 0.0
            signal[i] = state
        confidence = np.clip(np.abs(z) / (entry_z * 1.5), 0.0, 1.0) \
            * (np.abs(signal) > 0)
        adx_v = _adx(df, int(p["adx_period"]))["adx"].to_numpy(dtype=float)
        lo, hi = float(p["adx_lo"]), float(p["adx_hi"])
        gate = np.clip((hi - adx_v) / (hi - lo + 1e-12), 0.15, 1.0)
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(np.zeros(len(df)), np.zeros(len(df)),
                          np.ones(len(df)), df.index)
        p = self.params
        lookback = max(20, int(p["lookback"]))
        min_gap = max(3, int(p["min_gap"]))
        hold = max(1, int(p["hold"]))
        min_delta = float(p["min_rsi_delta"])
        rsi_lo, rsi_hi = float(p["rsi_lo"]), float(p["rsi_hi"])
        px = df["close"].astype(float).to_numpy(dtype=float)
        rsi_v = _rsi(df, int(p["rsi_period"])).to_numpy(dtype=float)
        n = len(df)
        raw = np.zeros(n)
        conf = np.zeros(n)
        half = lookback // 2
        for i in range(lookback, n):
            w0, w1 = i - lookback, i
            mid = w0 + half
            p1_lo = w0 + int(np.argmin(px[w0:mid]))
            p2_lo = mid + int(np.argmin(px[mid:w1]))
            p1_hi = w0 + int(np.argmax(px[w0:mid]))
            p2_hi = mid + int(np.argmax(px[mid:w1]))
            bull = (p2_lo - p1_lo >= min_gap and px[p2_lo] < px[p1_lo]
                    and rsi_v[p2_lo] - rsi_v[p1_lo] >= min_delta
                    and rsi_v[p2_lo] <= rsi_lo)
            bear = (p2_hi - p1_hi >= min_gap and px[p2_hi] > px[p1_hi]
                    and rsi_v[p1_hi] - rsi_v[p2_hi] >= min_delta
                    and rsi_v[p2_hi] >= rsi_hi)
            if bull:
                raw[i] = 1.0
                conf[i] = np.clip(
                    (rsi_v[p2_lo] - rsi_v[p1_lo]) / 10.0, 0.25, 1.0)
            elif bear:
                raw[i] = -1.0
                conf[i] = np.clip(
                    (rsi_v[p1_hi] - rsi_v[p2_hi]) / 10.0, 0.25, 1.0)
        signal = np.zeros(n)
        confidence = np.zeros(n)
        for i in range(n):
            if raw[i] != 0.0:
                end = min(n, i + hold)
                signal[i:end] = raw[i]
                confidence[i:end] = conf[i]
        gate = np.ones(n)
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(np.zeros(len(df)), np.zeros(len(df)),
                          np.ones(len(df)), df.index)
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
        recent_squeeze = pd.Series(squeezed, index=df.index).rolling(
            confirm, min_periods=1).max().to_numpy(dtype=float) > 0
        px = df["close"].astype(float).to_numpy(dtype=float)
        upper = bb["upper"].to_numpy(dtype=float)
        lower = bb["lower"].to_numpy(dtype=float)
        mid = bb["mid"].to_numpy(dtype=float)
        signal = np.zeros(len(df))
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
        pen = np.where(signal > 0, (px - mid) / a,
                       np.where(signal < 0, (mid - px) / a, 0.0))
        confidence = np.clip(np.abs(pen) / 1.0, 0.15, 1.0) \
            * (np.abs(signal) > 0)
        gate = np.ones(len(df))
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_ohlcv(df)
        if len(df) < 10:
            return _frame(np.zeros(len(df)), np.zeros(len(df)),
                          np.ones(len(df)), df.index)
        p = self.params
        sar = psar(df, float(p["accel"]), float(p["max_accel"])
                   ).to_numpy(dtype=float)
        px = df["close"].astype(float).to_numpy(dtype=float)
        signal = np.sign(px - sar)
        a = atr(df, 14).to_numpy(dtype=float) + 1e-9
        confidence = np.clip(np.abs(px - sar) / (2.0 * a), 0.0, 1.0)
        gate = np.ones(len(df))
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


def run_zoo(names: list[str], df: pd.DataFrame,
            params: dict | None = None) -> dict[str, pd.DataFrame]:
    """Run many strategies; failures are isolated per strategy.

    Returns ``{name: signal-frame}`` for the strategies that produced a
    well-formed frame. Unknown names and crashing strategies are skipped.
    """
    params = params or {}
    out: dict[str, pd.DataFrame] = {}
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


def quick_score(sig: pd.DataFrame, close: pd.Series,
                periods: int = 252) -> dict:
    """Strategy quality from sign returns: Sharpe, hit-rate, turnover."""
    px = close.astype(float)
    rets = px.pct_change().fillna(0.0).to_numpy()
    pos = np.sign(sig["signal"].to_numpy(dtype=float))
    pos = np.roll(pos, 1)
    pos[0] = 0.0
    strat_rets = pos * rets
    w = sig["confidence"].to_numpy(dtype=float) * sig["gate"].to_numpy(dtype=float)
    strat_rets = strat_rets * np.clip(w, 0, 1)
    active = strat_rets[np.abs(pos) > 0]
    return {
        "sharpe": sharpe(pd.Series(strat_rets), periods),
        "hit_rate": float(np.mean(active > 0)) if len(active) else 0.0,
        "turnover": float(np.mean(np.abs(np.diff(pos)) > 0)) if len(pos) > 1 else 0.0,
        "exposure": float(np.mean(np.abs(pos))),
        "trades": int(np.sum(np.abs(np.diff(pos)) > 0) if len(pos) > 1 else 0),
    }


def rank_strategies(frames: dict[str, pd.DataFrame], close: pd.Series,
                    periods: int = 252, min_trades: int = 5) -> pd.DataFrame:
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
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).set_index("name")
    df = df[df["trades"] >= min_trades]
    if df.empty:
        return df
    df["score"] = df["sharpe"] - 2.0 * df["turnover"]
    return df.sort_values("score", ascending=False)
