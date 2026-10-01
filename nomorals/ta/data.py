"""Bar data utilities: synthetic generator, cleaning, embargoed splits.

Ported from ``sentinel/data/feed.py`` (user's own Sentinel.py bot) —
minus the keyed loaders (ccxt / yfinance), which Devon covers keylessly in
``nomorals.integrations.market_data``.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .math import ensure_ohlcv, resample_ohlcv


_log = logging.getLogger(__name__)

__all__ = [
    "make_synthetic",
    "clean_ohlcv",
    "split_embargo",
    "multi_timeframe",
    "resample",
]


def make_synthetic(n: int = 2000, seed: int = 42, start: str = "2022-01-01",
                   freq: str = "h", regimes: bool = True) -> pd.DataFrame:
    """Regime-switching geometric random walk with OHLCV microstructure.

    Deterministic for a given ``seed`` — the standard fixture for tests and
    for exercising the pipeline with no network.
    """
    rng = np.random.default_rng(seed)
    if regimes and n >= 300:
        cuts = np.linspace(0, n, 7).astype(int)
        drifts = [0.0012, -0.0004, 0.0002, 0.0018, -0.0011, 0.0005]
        vols = [0.008, 0.016, 0.006, 0.011, 0.022, 0.009]
        rets = np.zeros(n)
        for i in range(6):
            a, b = cuts[i], cuts[i + 1]
            rets[a:b] = drifts[i] + rng.normal(0, vols[i], b - a)
    else:
        rets = 0.0004 + rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    noise_o = rng.normal(0, 0.0015, n)
    open_ = close * (1 + noise_o)
    spread = np.abs(rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    base_vol = 5000 + 3000 * np.abs(rets) / (np.abs(rets).mean() + 1e-9)
    vol = np.abs(base_vol + rng.normal(0, 800, n))
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": vol},
        index=idx,
    )


def clean_ohlcv(df: pd.DataFrame, max_gap_bars: int = 5) -> pd.DataFrame:
    """Repair bad bars: fix high/low inversions, fill small time gaps."""
    df = ensure_ohlcv(df).copy()
    df.loc[df["high"] < df[["open", "low", "close"]].max(axis=1), "high"] = \
        df[["open", "low", "close"]].max(axis=1)
    df.loc[df["low"] > df[["open", "high", "close"]].min(axis=1), "low"] = \
        df[["open", "high", "close"]].min(axis=1)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.asfreq(pd.infer_freq(df.index) or "h")
        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]].ffill(limit=max_gap_bars)
        df["volume"] = df["volume"].fillna(0.0)
        df = df.dropna(subset=["close"])
    return ensure_ohlcv(df)


def split_embargo(df: pd.DataFrame, test_frac: float = 0.25,
                  embargo_bars: int = 20):
    """Chronological train/test split with an embargo gap (no leakage)."""
    df = ensure_ohlcv(df)
    n = len(df)
    cut = int(n * (1 - test_frac))
    train = df.iloc[:cut]
    test = df.iloc[cut + embargo_bars:]
    return train, test


def multi_timeframe(df: pd.DataFrame, rules=("4h", "1D")) -> dict:
    """Base frame plus resampled higher-timeframe views."""
    out = {"base": ensure_ohlcv(df)}
    for r in rules:
        try:
            out[r] = resample_ohlcv(df, r)
        except Exception as e:
            _log.debug("resample to %s failed: %s", r, e)
            continue
    return out


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample bars to a higher timeframe (e.g. '4h', '1D')."""
    return resample_ohlcv(df, rule)
