"""Adaptive market-regime detector.

Ported from ``sentinel/core/regime.py`` (user's own Sentinel.py bot). Five
states — TREND_UP / TREND_DOWN / RANGE / SQUEEZE / PANIC — with thresholds
calibrated from rolling quantiles of each feature's own history, so the
detector auto-tunes to any symbol or timeframe. Hysteresis prevents label
flicker.
"""

from __future__ import annotations


from .math import atr, ensure_ohlcv, rolling_quantile


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



__all__ = ["LABELS", "CODES", "RegimeDetector"]

LABELS = ("TREND_UP", "TREND_DOWN", "RANGE", "SQUEEZE", "PANIC")
CODES = {name: i for i, name in enumerate(LABELS)}


def _r2_trend(close: _pd.Series, window: int) -> _pd.Series:
    """Signed R² of a rolling log-price linear fit (trend strength × sign)."""
    x = _np.arange(window, dtype=float)

    def _r2(v):
        if len(v) < 8:
            return 0.0
        y = _np.asarray(v, dtype=float)
        if float(_np.std(y)) <= 1e-12:
            return 0.0
        xx = x[-len(y):]
        a, b = _np.polyfit(xx, y, 1)
        pred = a * xx + b
        ss_res = float(_np.sum((y - pred) ** 2))
        ss_tot = float(_np.sum((y - y.mean()) ** 2)) + 1e-12
        return float(max(0.0, 1.0 - ss_res / ss_tot)) * float(_np.sign(a) or 1.0)

    return (
        close.rolling(window, min_periods=max(8, window // 3))
        .apply(_r2, raw=True)
        .fillna(0.0)
    )


def _efficiency(close: _pd.Series, window: int) -> _pd.Series:
    """Kaufman-style efficiency ratio, signed by net direction."""
    net = close.diff(window).abs()
    path = close.diff().abs().rolling(window, min_periods=5).sum()
    er = (net / (path + 1e-12)).fillna(0.0)
    direction = _np.sign(close.diff(window).fillna(0.0))
    return er * direction


class RegimeDetector:
    """Five-state regime engine with quantile-calibrated thresholds."""

    def __init__(self, lookback: int = 200, feature_window: int = 60,
                 cal_window: int = 500, hysteresis: int = 3):
        self.lookback = lookback
        self.feature_window = feature_window
        self.cal_window = cal_window
        self.hysteresis = hysteresis

    def features(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        close = df["close"]
        logc = _np.log(close.clip(lower=1e-9))
        w = self.feature_window
        f = _pd.DataFrame(index=df.index)
        f["trend_r2"] = _r2_trend(logc, w)
        f["efficiency"] = _efficiency(close, w)
        rets = close.pct_change().fillna(0.0)
        f["vol"] = rets.rolling(w, min_periods=10).std().bfill().ffill().fillna(1e-9)
        f["vol_median"] = (
            f["vol"].rolling(self.lookback, min_periods=20).median().bfill().ffill()
        )
        f["vol_ratio"] = (f["vol"] / (f["vol_median"] + 1e-12)).fillna(1.0)
        a = atr(df, 14)
        f["atr_ratio"] = (
            a / (a.rolling(self.lookback, min_periods=20).mean().bfill().ffill() + 1e-12)
        ).fillna(1.0)
        hi = close.rolling(w, min_periods=5).max()
        lo = close.rolling(w, min_periods=5).min()
        f["range_pos"] = ((close - lo) / ((hi - lo) + 1e-12)).fillna(0.5)
        width = (hi - lo) / (close.rolling(w, min_periods=5).mean() + 1e-12)
        f["width"] = width.fillna(0.0)
        f["width_q"] = (
            f["width"].rolling(self.cal_window, min_periods=30).rank(pct=True).fillna(0.5)
        )
        f["vol_q"] = (
            f["vol_ratio"].rolling(self.cal_window, min_periods=30).rank(pct=True).fillna(0.5)
        )
        return f.fillna(0.0)

    def _raw_label(self, f: _pd.DataFrame, sq: float, pq: float,
                   tq: float) -> _pd.Series:
        tr = f["trend_r2"].to_numpy()
        er = f["efficiency"].to_numpy()
        wq = f["width_q"].to_numpy()
        vq = f["vol_q"].to_numpy()
        labels = _np.full(len(f), CODES["RANGE"], dtype=int)
        trend = (_np.abs(tr) > tq) & (_np.abs(er) > 0.25)
        labels[trend & (tr > 0)] = CODES["TREND_UP"]
        labels[trend & (tr < 0)] = CODES["TREND_DOWN"]
        labels[(wq < sq) & (~trend)] = CODES["SQUEEZE"]
        labels[vq > pq] = CODES["PANIC"]
        return _pd.Series(labels, index=f.index)

    def fit(self, df: _pd.DataFrame, squeeze_q: float = 0.15,
            panic_q: float = 0.95, trend_q: float = 0.45) -> _pd.DataFrame:
        """Full regime frame: features + hysteresis-smoothed label + probs."""
        f = self.features(df)
        raw = self._raw_label(f, squeeze_q, panic_q, trend_q)
        lab = raw.to_numpy().copy()
        h = max(1, self.hysteresis)
        for i in range(1, len(lab)):
            if lab[i] != lab[i - 1]:
                j = i
                while j < len(lab) and lab[j] == lab[i]:
                    j += 1
                if j - i < h:
                    lab[i:j] = lab[i - 1]
        out = f.copy()
        out["regime"] = lab
        out["label"] = [LABELS[int(c)] for c in lab]
        tr = _np.abs(f["trend_r2"].to_numpy())
        out["p_trend"] = _np.clip((tr - trend_q) / (1 - trend_q + 1e-9), 0, 1)
        out["p_squeeze"] = _np.clip(
            (squeeze_q - f["width_q"].to_numpy()) / (squeeze_q + 1e-9), 0, 1)
        out["p_panic"] = _np.clip(
            (f["vol_q"].to_numpy() - panic_q) / (1 - panic_q + 1e-9), 0, 1)
        out["p_range"] = _np.clip(
            1 - out[["p_trend", "p_squeeze", "p_panic"]].max(axis=1), 0, 1)
        return out

    def current(self, df: _pd.DataFrame, **kw) -> dict:
        """Regime snapshot at the last bar, with a lookback label mix."""
        frame = self.fit(df, **kw)
        last = frame.iloc[-1]
        counts = (
            frame["label"].iloc[-self.lookback:].value_counts(normalize=True).to_dict()
        )
        return {
            "label": str(last["label"]),
            "code": int(last["regime"]),
            "p_trend": float(last["p_trend"]),
            "p_range": float(last["p_range"]),
            "p_squeeze": float(last["p_squeeze"]),
            "p_panic": float(last["p_panic"]),
            "trend_r2": float(last["trend_r2"]),
            "vol_ratio": float(last["vol_ratio"]),
            "mix_lookback": {k: float(v) for k, v in counts.items()},
        }

    def transition_alert(self, df: _pd.DataFrame, **kw) -> dict:
        """Did the regime just change? Flags squeeze→expansion and panic onset."""
        frame = self.fit(df, **kw)
        if len(frame) < 10:
            return {"transition": False}
        prev = str(frame["label"].iloc[-6])
        now = str(frame["label"].iloc[-1])
        return {
            "transition": prev != now,
            "from": prev,
            "to": now,
            "squeeze_to_expansion": prev == "SQUEEZE"
            and now in ("TREND_UP", "TREND_DOWN", "PANIC"),
            "panic_onset": now == "PANIC" and prev != "PANIC",
        }
