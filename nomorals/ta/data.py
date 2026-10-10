"""Bar data utilities: synthetic generator, cleaning, embargoed splits.

Ported from ``sentinel/data/feed.py`` (user's own Sentinel.py bot) —
minus the keyed loaders (ccxt / yfinance), which Devon covers keylessly in
``nomorals.integrations.market_data``.
"""

from __future__ import annotations

import logging

from .math import ensure_ohlcv, resample_ohlcv


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




_log = logging.getLogger(__name__)

__all__ = [
    "make_synthetic",
    "clean_ohlcv",
    "split_embargo",
    "multi_timeframe",
    "resample",
    # ── sweep additions ──
    "quality_report",
    "detect_outliers",
    "bar_gaps",
    "align_frames",
]


def make_synthetic(n: int = 2000, seed: int = 42, start: str = "2022-01-01",
                   freq: str = "h", regimes: bool = True) -> _pd.DataFrame:
    """Regime-switching geometric random walk with OHLCV microstructure.

    Deterministic for a given ``seed`` — the standard fixture for tests and
    for exercising the pipeline with no network.
    """
    rng = _np.random.default_rng(seed)
    if regimes and n >= 300:
        cuts = _np.linspace(0, n, 7).astype(int)
        drifts = [0.0012, -0.0004, 0.0002, 0.0018, -0.0011, 0.0005]
        vols = [0.008, 0.016, 0.006, 0.011, 0.022, 0.009]
        rets = _np.zeros(n)
        for i in range(6):
            a, b = cuts[i], cuts[i + 1]
            rets[a:b] = drifts[i] + rng.normal(0, vols[i], b - a)
    else:
        rets = 0.0004 + rng.normal(0, 0.01, n)
    close = 100.0 * _np.exp(_np.cumsum(rets))
    noise_o = rng.normal(0, 0.0015, n)
    open_ = close * (1 + noise_o)
    spread = _np.abs(rng.normal(0.0, 0.004, n))
    high = _np.maximum(open_, close) * (1 + spread)
    low = _np.minimum(open_, close) * (1 - spread)
    base_vol = 5000 + 3000 * _np.abs(rets) / (_np.abs(rets).mean() + 1e-9)
    vol = _np.abs(base_vol + rng.normal(0, 800, n))
    idx = _pd.date_range(start, periods=n, freq=freq)
    return _pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": vol},
        index=idx,
    )


def clean_ohlcv(df: _pd.DataFrame, max_gap_bars: int = 5) -> _pd.DataFrame:
    """Repair bad bars: fix high/low inversions, fill small time gaps."""
    df = ensure_ohlcv(df).copy()
    df.loc[df["high"] < df[["open", "low", "close"]].max(axis=1), "high"] = \
        df[["open", "low", "close"]].max(axis=1)
    df.loc[df["low"] > df[["open", "high", "close"]].min(axis=1), "low"] = \
        df[["open", "high", "close"]].min(axis=1)
    if isinstance(df.index, _pd.DatetimeIndex):
        df = df.asfreq(_pd.infer_freq(df.index) or "h")
        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]].ffill(limit=max_gap_bars)
        df["volume"] = df["volume"].fillna(0.0)
        df = df.dropna(subset=["close"])
    return ensure_ohlcv(df)


def split_embargo(df: _pd.DataFrame, test_frac: float = 0.25,
                  embargo_bars: int = 20):
    """Chronological train/test split with an embargo gap (no leakage)."""
    df = ensure_ohlcv(df)
    n = len(df)
    cut = int(n * (1 - test_frac))
    train = df.iloc[:cut]
    test = df.iloc[cut + embargo_bars:]
    return train, test


def multi_timeframe(df: _pd.DataFrame, rules=("4h", "1D")) -> dict:
    """Base frame plus resampled higher-timeframe views."""
    out = {"base": ensure_ohlcv(df)}
    for r in rules:
        try:
            out[r] = resample_ohlcv(df, r)
        except Exception as e:
            _log.debug("resample to %s failed: %s", r, e)
            continue
    return out


def resample(df: _pd.DataFrame, rule: str) -> _pd.DataFrame:
    """Resample bars to a higher timeframe (e.g. '4h', '1D')."""
    return resample_ohlcv(df, rule)


# ── sweep additions: data integrity reporting ────────────────────────────
# Practitioner rule mined from the exchange adapters: never silently
# repair data — report what was found, then clean.

def bar_gaps(df: _pd.DataFrame) -> list[dict]:
    """Locate time gaps: where the bar spacing jumps beyond 1.5× median."""
    df = ensure_ohlcv(df)
    if not isinstance(df.index, _pd.DatetimeIndex) or len(df) < 3:
        return []
    diffs = df.index.to_series().diff().dropna()
    med = diffs.median()
    if med.total_seconds() <= 0:
        return []
    gaps = []
    for ts, d in diffs.items():
        if d > med * 1.5:
            missing = int(round(d / med)) - 1
            gaps.append({"at": ts, "gap": str(d), "missing_bars": missing})
    return gaps


def detect_outliers(df: _pd.DataFrame, k: float = 8.0) -> _pd.DataFrame:
    """Bars whose log return exceeds ``k``× rolling std — bad ticks.

    Returns the offending bars (not a mask) so the caller can inspect.
    """
    df = ensure_ohlcv(df)
    c = df["close"].astype(float)
    lr = _np.log(c / c.shift(1)).fillna(0.0)
    sd = lr.rolling(100, min_periods=20).std().bfill()
    flag = lr.abs() > float(k) * (sd + 1e-12)
    return df[flag.fillna(False)]


def quality_report(df: _pd.DataFrame) -> dict:
    """Full integrity report: gaps, outliers, stale bars, duplicates.

    Call before ``clean_ohlcv`` — it tells you what the cleaner is about
    to paper over.
    """
    raw = df
    df = ensure_ohlcv(df)
    gaps = bar_gaps(df)
    outliers = detect_outliers(df)
    stale = df[(df["high"] == df["low"]) & (df["volume"] == 0)]
    dupes = int(raw.index.duplicated().sum()) if hasattr(
        raw.index, "duplicated") else 0
    inversions = int(((df["high"] < df["low"])).sum())
    neg_vol = int((df["volume"] < 0).sum())
    n = len(df)
    score = 100.0
    gap_missing = sum(g["missing_bars"] for g in gaps)
    score -= min(45.0, len(gaps) * 3.0 + gap_missing / max(1, n) * 60.0)
    score -= min(25.0, len(outliers) * 5.0)
    score -= min(20.0, len(stale) / max(1, n) * 100.0)
    score -= min(15.0, dupes * 2.0)
    return {
        "bars": n,
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "gaps": gaps[:10],
        "n_gaps": len(gaps),
        "outliers": [str(t) for t in outliers.index[:10]],
        "n_outliers": len(outliers),
        "n_stale_bars": int(len(stale)),
        "n_duplicates": dupes,
        "n_inversions": inversions,
        "n_negative_volume": neg_vol,
        "quality_score": round(max(0.0, score), 1),
        "verdict": ("CLEAN" if score >= 90 else
                    "USABLE" if score >= 70 else
                    "DEGRADED" if score >= 40 else "TRASH"),
    }


def align_frames(frames: dict[str, _pd.DataFrame],
                 how: str = "inner") -> dict[str, _pd.DataFrame]:
    """Align multiple symbol frames onto a common index.

    ``how``: "inner" (default — only shared bars) or "outer" (union,
    forward-filled). Every frame is validated first.
    """
    cleaned = {k: ensure_ohlcv(v) for k, v in frames.items()}
    if not cleaned:
        return {}
    idx = None
    for v in cleaned.values():
        idx = v.index if idx is None else (
            idx.intersection(v.index) if how == "inner"
            else idx.union(v.index))
    out = {}
    for k, v in cleaned.items():
        vv = v.reindex(idx)
        if how == "outer":
            vv = vv.ffill()
        out[k] = vv.dropna(subset=["close"])
    return out
