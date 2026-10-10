"""Candlestick pattern detection: the missing half of price action.

TA-Lib ships 61 canonical candlestick patterns as its recognition engine;
until now this kit had none. Every pattern here is a pure function,
``DataFrame in -> signed Series out`` (+1 bullish, -1 bearish, 0 none),
ATR-normalized so thresholds hold across volatility regimes (the mistake
the hardcoded 10%-of-range MQL5 gists make).

The mined lesson (yfinance-ta-patterns, Bulkowski): **naked patterns are
noise** — a hammer mid-consolidation is a coin flip. Edge comes from
confluence, so every signal ships with:

- a qualitative reliability tier per pattern (high/medium/low, grounded in
  Bulkowski's published reversal rankings — qualitative on purpose, no
  fabricated percentages),
- ``with_trend_context`` to keep only signals aligned with the local trend,
- ``pattern_score`` — reliability-weighted confluence across all patterns,
- ``confluence_features`` — ML-ready features for the meta model.
"""

from __future__ import annotations


from .math import atr as _atr
from .math import ema, ensure_ohlcv


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



__all__ = [
    "PATTERNS",
    "PATTERN_TIERS",
    "detect",
    "detect_all",
    "pattern_score",
    "with_trend_context",
    "confluence_features",
    "active_patterns",
]


def _bars(df: _pd.DataFrame):
    """Unpack OHLCV + ATR-normalized candle geometry."""
    df = ensure_ohlcv(df)
    o = df["open"].astype(float).to_numpy()
    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    a = _atr(df, 14).to_numpy() + 1e-12
    body = c - o
    abody = _np.abs(body)
    rng = (h - l) / a                     # range in ATR units
    upper = (h - _np.maximum(o, c)) / a    # upper shadow in ATR
    lower = (_np.minimum(o, c) - l) / a    # lower shadow in ATR
    b = abody / a                         # body in ATR units
    bull = body > 0
    bear = body < 0
    n = len(df)
    return {
        "o": o, "h": h, "l": l, "c": c, "a": a, "body": body,
        "abody": abody, "rng": rng, "upper": upper, "lower": lower,
        "b": b, "bull": bull, "bear": bear, "n": n, "index": df.index,
    }


def _out(sig: _np.ndarray, index) -> _pd.Series:
    return _pd.Series(sig.astype(float), index=index)


# ── single-candle patterns ─────────────────────────────────────────────

def doji(g) -> _np.ndarray:
    """Indecision: tiny body relative to range. Neutral (0) by itself."""
    return _np.zeros(g["n"])


def dragonfly_doji(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    is_doji = g["b"] <= 0.1 * _np.maximum(g["rng"], 0.3)
    sig[is_doji & (g["lower"] >= 2.0 * _np.maximum(g["b"], 0.05))
        & (g["upper"] <= 0.3)] = 1.0
    return sig


def gravestone_doji(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    is_doji = g["b"] <= 0.1 * _np.maximum(g["rng"], 0.3)
    sig[is_doji & (g["upper"] >= 2.0 * _np.maximum(g["b"], 0.05))
        & (g["lower"] <= 0.3)] = -1.0
    return sig


def hammer(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    sig[(g["lower"] >= 2.0 * _np.maximum(g["b"], 0.1))
        & (g["upper"] <= _np.maximum(g["b"], 0.2))
        & (g["rng"] >= 0.5)] = 1.0
    return sig


def inverted_hammer(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    sig[(g["upper"] >= 2.0 * _np.maximum(g["b"], 0.1))
        & (g["lower"] <= _np.maximum(g["b"], 0.2))
        & (g["rng"] >= 0.5)] = 1.0
    return sig


def shooting_star(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    sig[(g["upper"] >= 2.0 * _np.maximum(g["b"], 0.1))
        & (g["lower"] <= _np.maximum(g["b"], 0.2))
        & (g["rng"] >= 0.5)] = -1.0
    return sig


def hanging_man(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    sig[(g["lower"] >= 2.0 * _np.maximum(g["b"], 0.1))
        & (g["upper"] <= _np.maximum(g["b"], 0.2))
        & (g["rng"] >= 0.5)] = -1.0
    return sig


def marubozu_bull(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    sig[g["bull"] & (g["b"] >= 0.7) & (g["upper"] <= 0.1)
        & (g["lower"] <= 0.1)] = 1.0
    return sig


def marubozu_bear(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    sig[g["bear"] & (g["b"] >= 0.7) & (g["upper"] <= 0.1)
        & (g["lower"] <= 0.1)] = -1.0
    return sig


def spinning_top(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    small = g["b"] <= 0.3 * _np.maximum(g["rng"], 0.5)
    sig[small & (g["upper"] >= 0.4) & (g["lower"] >= 0.4)
        & (g["rng"] >= 0.8)] = _np.sign(g["body"][small & (g["upper"] >= 0.4)
                                                  & (g["lower"] >= 0.4)
                                                  & (g["rng"] >= 0.8)])
    return sig


def belt_hold_bull(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    sig[g["bull"] & (g["b"] >= 0.8) & (g["lower"] <= 0.1)
        & (g["upper"] >= 0.2)] = 1.0
    return sig


def belt_hold_bear(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    sig[g["bear"] & (g["b"] >= 0.8) & (g["upper"] <= 0.1)
        & (g["lower"] >= 0.2)] = -1.0
    return sig


# ── two-candle patterns ────────────────────────────────────────────────

def _prev(g, arr, k=1):
    out = _np.zeros_like(arr, dtype=float)
    out[k:] = arr[:-k]
    return out


def bullish_engulfing(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    pbear = _prev(g, g["bear"].astype(float))
    po, pc = _prev(g, g["o"]), _prev(g, g["c"])
    sig[(pbear > 0) & g["bull"] & (g["o"] <= pc) & (g["c"] >= po)
        & (g["b"] >= 0.5)] = 1.0
    return sig


def bearish_engulfing(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    pbull = _prev(g, g["bull"].astype(float))
    po, pc = _prev(g, g["o"]), _prev(g, g["c"])
    sig[(pbull > 0) & g["bear"] & (g["o"] >= pc) & (g["c"] <= po)
        & (g["b"] >= 0.5)] = -1.0
    return sig


def bullish_harami(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    pbear = _prev(g, g["bear"].astype(float))
    po, pc = _prev(g, g["o"]), _prev(g, g["c"])
    sig[(pbear > 0) & (g["o"] > pc) & (g["c"] < po)
        & (g["b"] <= 0.6 * _np.maximum(_prev(g, g["b"]), 0.1))] = 1.0
    return sig


def bearish_harami(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    pbull = _prev(g, g["bull"].astype(float))
    po, pc = _prev(g, g["o"]), _prev(g, g["c"])
    sig[(pbull > 0) & (g["o"] < pc) & (g["c"] > po)
        & (g["b"] <= 0.6 * _np.maximum(_prev(g, g["b"]), 0.1))] = -1.0
    return sig


def piercing(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    pbear = _prev(g, g["bear"].astype(float))
    po, pc = _prev(g, g["o"]), _prev(g, g["c"])
    mid = (po + pc) / 2.0
    sig[(pbear > 0) & g["bull"] & (g["o"] < pc) & (g["c"] > mid)
        & (g["c"] < po)] = 1.0
    return sig


def dark_cloud_cover(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    pbull = _prev(g, g["bull"].astype(float))
    po, pc = _prev(g, g["o"]), _prev(g, g["c"])
    mid = (po + pc) / 2.0
    sig[(pbull > 0) & g["bear"] & (g["o"] > pc) & (g["c"] < mid)
        & (g["c"] > po)] = -1.0
    return sig


def tweezer_bottom(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    pl = _prev(g, g["l"])
    sig[(_np.abs(g["l"] - pl) <= 0.1 * g["a"]) & (g["lower"] >= 0.4)
        & (_prev(g, g["bear"].astype(float)) > 0) & g["bull"]] = 1.0
    return sig


def tweezer_top(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    ph = _prev(g, g["h"])
    sig[(_np.abs(g["h"] - ph) <= 0.1 * g["a"]) & (g["upper"] >= 0.4)
        & (_prev(g, g["bull"].astype(float)) > 0) & g["bear"]] = -1.0
    return sig


def kicking_bull(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    pbear = _prev(g, g["bear"].astype(float))
    po = _prev(g, g["o"])
    sig[(pbear > 0) & g["bull"] & (g["o"] > po)
        & (g["b"] >= 0.8) & (_prev(g, g["b"]) >= 0.8)] = 1.0
    return sig


def kicking_bear(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    pbull = _prev(g, g["bull"].astype(float))
    po = _prev(g, g["o"])
    sig[(pbull > 0) & g["bear"] & (g["o"] < po)
        & (g["b"] >= 0.8) & (_prev(g, g["b"]) >= 0.8)] = -1.0
    return sig


# ── three-candle patterns ──────────────────────────────────────────────

def _prev2(g, arr):
    return _prev(g, _prev(g, arr))


def morning_star(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    c1bear = _prev2(g, g["bear"].astype(float))
    small2 = _prev(g, g["b"]) <= 0.4 * _np.maximum(_prev(g, g["rng"]), 0.4)
    o1, c1 = _prev2(g, g["o"]), _prev2(g, g["c"])
    sig[(c1bear > 0) & small2 & g["bull"]
        & (g["c"] > (o1 + c1) / 2.0) & (g["b"] >= 0.5)] = 1.0
    return sig


def evening_star(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    c1bull = _prev2(g, g["bull"].astype(float))
    small2 = _prev(g, g["b"]) <= 0.4 * _np.maximum(_prev(g, g["rng"]), 0.4)
    o1, c1 = _prev2(g, g["o"]), _prev2(g, g["c"])
    sig[(c1bull > 0) & small2 & g["bear"]
        & (g["c"] < (o1 + c1) / 2.0) & (g["b"] >= 0.5)] = -1.0
    return sig


def abandoned_baby_bull(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    c1bear = _prev2(g, g["bear"].astype(float))
    doji2 = _prev(g, g["b"]) <= 0.1 * _np.maximum(_prev(g, g["rng"]), 0.3)
    l1 = _prev2(g, g["l"])
    gap_down = _prev(g, g["h"]) < l1
    gap_up = g["l"] > _prev(g, g["h"])
    sig[(c1bear > 0) & doji2 & gap_down & gap_up & g["bull"]] = 1.0
    return sig


def abandoned_baby_bear(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    c1bull = _prev2(g, g["bull"].astype(float))
    doji2 = _prev(g, g["b"]) <= 0.1 * _np.maximum(_prev(g, g["rng"]), 0.3)
    h1 = _prev2(g, g["h"])
    gap_up = _prev(g, g["l"]) > h1
    gap_down = g["h"] < _prev(g, g["l"])
    sig[(c1bull > 0) & doji2 & gap_up & gap_down & g["bear"]] = -1.0
    return sig


def three_white_soldiers(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    b1 = _prev2(g, g["bull"].astype(float))
    b2 = _prev(g, g["bull"].astype(float))
    rising = (g["c"] > _prev(g, g["c"])) & (_prev(g, g["c"]) > _prev2(g, g["c"]))
    small_shadow = (g["upper"] <= 0.3) & (_prev(g, g["upper"]) <= 0.3)
    sig[(b1 > 0) & (b2 > 0) & g["bull"] & rising
        & (g["b"] >= 0.5) & small_shadow] = 1.0
    return sig


def three_black_crows(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    b1 = _prev2(g, g["bear"].astype(float))
    b2 = _prev(g, g["bear"].astype(float))
    falling = (g["c"] < _prev(g, g["c"])) & (_prev(g, g["c"]) < _prev2(g, g["c"]))
    small_shadow = (g["lower"] <= 0.3) & (_prev(g, g["lower"]) <= 0.3)
    sig[(b1 > 0) & (b2 > 0) & g["bear"] & falling
        & (g["b"] >= 0.5) & small_shadow] = -1.0
    return sig


def three_inside_up(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    har = bullish_harami(g)
    sig[(har > 0) & g["bull"] & (g["c"] > _prev2(g, g["o"]))] = 1.0
    return sig


def three_inside_down(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    har = bearish_harami(g)
    sig[(har < 0) & g["bear"] & (g["c"] < _prev2(g, g["o"]))] = -1.0
    return sig


def three_outside_up(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    eng = bullish_engulfing(_bars_shim(g, 1))
    sig[(eng > 0) & g["bull"] & (g["c"] > _prev(g, g["c"]))] = 1.0
    return sig


def three_outside_down(g) -> _np.ndarray:
    sig = _np.zeros(g["n"])
    eng = bearish_engulfing(_bars_shim(g, 1))
    sig[(eng < 0) & g["bear"] & (g["c"] < _prev(g, g["c"]))] = -1.0
    return sig


def _bars_shim(g, shift):
    """Shift the geometry dict back by ``shift`` bars for nested patterns."""
    n = g["n"]
    out = dict(g)
    for k in ("o", "h", "l", "c", "a", "body", "abody", "rng", "upper",
              "lower", "b"):
        v = g[k].copy()
        v[shift:] = v[:-shift]
        out[k] = v
    for k in ("bull", "bear"):
        v = g[k].copy()
        v[shift:] = v[:-shift]
        out[k] = v
    return out


#: name -> pattern function
PATTERNS = {
    "doji": doji,
    "dragonfly_doji": dragonfly_doji,
    "gravestone_doji": gravestone_doji,
    "hammer": hammer,
    "inverted_hammer": inverted_hammer,
    "shooting_star": shooting_star,
    "hanging_man": hanging_man,
    "marubozu_bull": marubozu_bull,
    "marubozu_bear": marubozu_bear,
    "spinning_top": spinning_top,
    "belt_hold_bull": belt_hold_bull,
    "belt_hold_bear": belt_hold_bear,
    "bullish_engulfing": bullish_engulfing,
    "bearish_engulfing": bearish_engulfing,
    "bullish_harami": bullish_harami,
    "bearish_harami": bearish_harami,
    "piercing": piercing,
    "dark_cloud_cover": dark_cloud_cover,
    "tweezer_bottom": tweezer_bottom,
    "tweezer_top": tweezer_top,
    "kicking_bull": kicking_bull,
    "kicking_bear": kicking_bear,
    "morning_star": morning_star,
    "evening_star": evening_star,
    "abandoned_baby_bull": abandoned_baby_bull,
    "abandoned_baby_bear": abandoned_baby_bear,
    "three_white_soldiers": three_white_soldiers,
    "three_black_crows": three_black_crows,
    "three_inside_up": three_inside_up,
    "three_inside_down": three_inside_down,
    "three_outside_up": three_outside_up,
    "three_outside_down": three_outside_down,
}

#: Qualitative reliability tiers, grounded in Bulkowski's published
#: reversal rankings (high = top-third performers, low = bottom-third).
#: Qualitative on purpose — exact percentages vary by market and sample.
PATTERN_TIERS = {
    "abandoned_baby_bull": "high", "abandoned_baby_bear": "high",
    "three_outside_up": "high", "three_outside_down": "high",
    "morning_star": "high", "evening_star": "high",
    "kicking_bull": "high", "kicking_bear": "high",
    "three_white_soldiers": "medium", "three_black_crows": "medium",
    "bullish_engulfing": "medium", "bearish_engulfing": "medium",
    "piercing": "medium", "dark_cloud_cover": "medium",
    "hammer": "medium", "shooting_star": "medium",
    "tweezer_bottom": "medium", "tweezer_top": "medium",
    "three_inside_up": "medium", "three_inside_down": "medium",
    "inverted_hammer": "medium",
    "belt_hold_bull": "low", "belt_hold_bear": "low",
    "bullish_harami": "low", "bearish_harami": "low",
    "hanging_man": "low", "marubozu_bull": "low", "marubozu_bear": "low",
    "dragonfly_doji": "low", "gravestone_doji": "low",
    "spinning_top": "low", "doji": "low",
}

_TIER_W = {"high": 1.0, "medium": 0.6, "low": 0.3}


def detect(name: str, df: _pd.DataFrame) -> _pd.Series:
    """Signed signal series for one named pattern."""
    key = (name or "").strip().lower().replace("-", "_")
    if key not in PATTERNS:
        raise KeyError(
            f"unknown pattern {name!r}; available: {sorted(PATTERNS)}")
    g = _bars(df)
    return _out(PATTERNS[key](g), g["index"]).rename(key)


def detect_all(df: _pd.DataFrame,
               names: list[str] | None = None) -> _pd.DataFrame:
    """Signed signal frame: one column per pattern."""
    g = _bars(df)
    names = names or sorted(PATTERNS)
    out = _pd.DataFrame(index=g["index"])
    for name in names:
        key = name.strip().lower().replace("-", "_")
        if key in PATTERNS:
            out[key] = PATTERNS[key](g)
    return out


def pattern_score(df: _pd.DataFrame,
                  names: list[str] | None = None) -> _pd.DataFrame:
    """Reliability-weighted confluence score per bar.

    Returns ``score`` in [-1, 1] (tier-weighted net signal), ``n_bull`` /
    ``n_bear`` counts, and ``strength`` (weighted mass). A lone low-tier
    pattern scores ~0.3; stacked high-tier agreement approaches ±1.
    """
    sig = detect_all(df, names)
    w = _np.array([_TIER_W[PATTERN_TIERS.get(c, "low")] for c in sig.columns])
    vals = sig.to_numpy(dtype=float)
    weighted = vals * w
    mass = _np.abs(weighted).sum(axis=1)
    score = weighted.sum(axis=1) / (mass + 1e-12) * _np.clip(mass, 0, 1)
    return _pd.DataFrame({
        "score": score,
        "n_bull": (vals > 0).sum(axis=1).astype(float),
        "n_bear": (vals < 0).sum(axis=1).astype(float),
        "strength": mass,
    }, index=sig.index)


def with_trend_context(df: _pd.DataFrame, signals: _pd.DataFrame,
                       fast: int = 20, slow: int = 50) -> _pd.DataFrame:
    """Keep only pattern signals aligned with the local EMA trend.

    Naked patterns are noise; a bullish pattern *in an uptrend* (or at a
    higher-timeframe support flip) is the confluence that carries edge.
    Signals opposing the EMA(fast/slow) trend are zeroed; with-trend
    signals pass through untouched.
    """
    df = ensure_ohlcv(df)
    close = df["close"].astype(float)
    trend = _np.sign((ema(close, fast) - ema(close, slow)).to_numpy())
    vals = signals.to_numpy(dtype=float)
    keep = (vals * trend[:, None]) >= 0
    out = vals * keep
    return _pd.DataFrame(out, index=signals.index, columns=signals.columns)


def active_patterns(df: _pd.DataFrame, at: int = -1,
                    names: list[str] | None = None) -> list[dict]:
    """Human-readable list of patterns firing at bar ``at``."""
    sig = detect_all(df, names)
    row = sig.iloc[at]
    out = []
    for name, v in row.items():
        if v != 0:
            out.append({
                "pattern": name,
                "direction": "bullish" if v > 0 else "bearish",
                "tier": PATTERN_TIERS.get(name, "low"),
            })
    # High-tier first — the ones worth paying attention to.
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(out, key=lambda d: order[d["tier"]])


def confluence_features(df: _pd.DataFrame) -> _pd.DataFrame:
    """ML-ready pattern features: score, strength, tier masses, counts."""
    sig = detect_all(df)
    sc = pattern_score(df)
    feats = _pd.DataFrame(index=sig.index)
    feats["pat_score"] = sc["score"]
    feats["pat_strength"] = sc["strength"]
    feats["pat_n_bull"] = sc["n_bull"]
    feats["pat_n_bear"] = sc["n_bear"]
    vals = sig.to_numpy(dtype=float)
    for tier in ("high", "medium", "low"):
        cols = [c for c in sig.columns
                if PATTERN_TIERS.get(c) == tier]
        if cols:
            w = _TIER_W[tier]
            feats[f"pat_{tier}"] = (sig[cols].to_numpy(dtype=float) * w
                                    ).sum(axis=1)
    ctx = with_trend_context(df, sig)
    feats["pat_trend_aligned"] = ctx.to_numpy(dtype=float).sum(axis=1)
    return feats.fillna(0.0)
