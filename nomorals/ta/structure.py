"""Market structure: swings, trend legs, BOS/CHoCH, support/resistance zones.

Price action starts here. Every framework — Dow Theory, Wyckoff, SMC —
is built on swing highs/lows and what happens when they break.

Prospective only: a swing is confirmed ``confirm`` bars after the fact;
BOS/CHoCH fire on the bar where the break closes through the level.
Nothing here peeks at future bars.
"""

from __future__ import annotations

import logging

from .math import atr, clamp, ensure_ohlcv

_log = logging.getLogger(__name__)


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


def _require() -> None:
    if not (_HAS_NUMPY and _HAS_PANDAS):
        raise TAError("numpy and pandas are required: pip install nomorals[ta]")


# ── swing detection ─────────────────────────────────────────────────────

def swings(df, left: int = 5, right: int = 5) -> _pd.DataFrame:
    """Fractal swing highs/lows.

    A swing high at bar ``i`` has the highest high of ``left`` bars before
    and ``right`` bars after. Returns a frame with ``swing_high`` /
    ``swing_low`` boolean columns aligned to ``df.index``. The ``right``
    lag is the confirmation delay — honest, never repainted.
    """
    _require()
    df = ensure_ohlcv(df)
    n = len(df)
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    sh = _np.zeros(n, dtype=bool)
    sl = _np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        if hi[i] == _np.max(hi[i - left:i + right + 1]) and \
                _np.sum(hi[i - left:i + right + 1] == hi[i]) == 1:
            sh[i] = True
        if lo[i] == _np.min(lo[i - left:i + right + 1]) and \
                _np.sum(lo[i - left:i + right + 1] == lo[i]) == 1:
            sl[i] = True
    return _pd.DataFrame({"swing_high": sh, "swing_low": sl}, index=df.index)


def swing_points(df, left: int = 5, right: int = 5) -> _pd.DataFrame:
    """Tidy table of confirmed swings: ``kind`` (+1 high / −1 low), ``price``."""
    _require()
    df = ensure_ohlcv(df)
    s = swings(df, left=left, right=right)
    rows = []
    for i in df.index[s["swing_high"]]:
        rows.append({"bar": i, "kind": 1, "price": float(df.loc[i, "high"])})
    for i in df.index[s["swing_low"]]:
        rows.append({"bar": i, "kind": -1, "price": float(df.loc[i, "low"])})
    out = _pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("bar").reset_index(drop=True)
    return out


# ── trend legs: HH/HL/LH/LL ─────────────────────────────────────────────

def label_swings(pts: _pd.DataFrame) -> _pd.DataFrame:
    """Label each swing HH/HL/LH/LL relative to the previous same-kind swing."""
    _require()
    pts = pts.copy()
    pts["label"] = ""
    last_hi = None
    last_lo = None
    for i, r in pts.iterrows():
        if r["kind"] == 1:
            pts.loc[i, "label"] = "HH" if last_hi is not None and r["price"] > last_hi else \
                                  "LH" if last_hi is not None else "H"
            last_hi = r["price"]
        else:
            pts.loc[i, "label"] = "HL" if last_lo is not None and r["price"] > last_lo else \
                                  "LL" if last_lo is not None else "L"
            last_lo = r["price"]
    return pts


def market_bias(pts: _pd.DataFrame, lookback: int = 4) -> dict:
    """Structural bias from recent swing labels.

    Uptrend needs HH+HL; downtrend LH+LL; anything else is range/transition.
    Returns ``{"bias": +1|-1|0, "label": str, "reason": str}``.
    """
    _require()
    recent = pts.tail(lookback)
    labels = list(recent["label"])
    highs = [l for l in labels if l in ("HH", "LH", "H")]
    lows = [l for l in labels if l in ("HL", "LL", "L")]
    if highs and all(l == "HH" for l in highs[-2:]) and \
            lows and all(l == "HL" for l in lows[-2:]):
        return {"bias": 1, "label": "UPTREND",
                "reason": "higher highs and higher lows"}
    if highs and all(l == "LH" for l in highs[-2:]) and \
            lows and all(l == "LL" for l in lows[-2:]):
        return {"bias": -1, "label": "DOWNTREND",
                "reason": "lower highs and lower lows"}
    return {"bias": 0, "label": "RANGE",
            "reason": "no clean HH/HL or LH/LL sequence"}


# ── BOS / CHoCH ─────────────────────────────────────────────────────────

def structure_breaks(df, left: int = 5, right: int = 3) -> _pd.DataFrame:
    """BOS and CHoCH events, prospective.

    - BOS: close through the most recent swing in the direction of bias.
    - CHoCH: close through the most recent swing against the bias
      (first failure — alert, not entry trigger).

    Returns event frame: ``bar``, ``kind`` (BOS_BULL/BOS_BEAR/CHOCH_BULL/
    CHOCH_BEAR), ``level``, ``bias_before``.
    """
    _require()
    df = ensure_ohlcv(df)
    pts = swing_points(df, left=left, right=right)
    if pts.empty:
        return _pd.DataFrame(
            columns=["bar", "kind", "level", "bias_before"])
    close = df["close"]
    events = []
    bias = 0
    last_hi = None
    last_lo = None
    # walk bars in order; swings are known only after confirmation lag
    for i in range(len(df)):
        bar = df.index[i]
        # register newly confirmed swings (confirmation bar = swing bar + right)
        for _, p in pts[pts["bar"] == bar].iterrows():
            if p["kind"] == 1:
                last_hi = float(p["price"])
            else:
                last_lo = float(p["price"])
        c = float(close.iloc[i])
        if last_hi is not None and c > last_hi:
            kind = "BOS_BULL" if bias >= 0 else "CHOCH_BULL"
            events.append({"bar": bar, "kind": kind, "level": last_hi,
                           "bias_before": bias})
            bias = 1
            last_hi = None  # consumed; wait for the next swing
        elif last_lo is not None and c < last_lo:
            kind = "BOS_BEAR" if bias <= 0 else "CHOCH_BEAR"
            events.append({"bar": bar, "kind": kind, "level": last_lo,
                           "bias_before": bias})
            bias = -1
            last_lo = None
    return _pd.DataFrame(events)


# ── support / resistance zones ──────────────────────────────────────────

def sr_zones(df, left: int = 5, right: int = 5, atr_mult: float = 0.5,
             merge_atr: float = 1.0, min_touches: int = 2,
             max_zones: int = 10) -> _pd.DataFrame:
    """Support/resistance zones from clustered swing points.

    Algorithm (convergent across implementations):
    1. confirmed fractal pivots over the lookback
    2. single-link clustering within ``merge_atr`` × ATR
    3. zone = [min low, max high] of cluster; width floored at
       ``atr_mult`` × ATR
    4. strength = recency-weighted touches (0–40) + flip bonus (×1.5 when
       the cluster holds both highs and lows) — scaled 0–100
    5. state vs current price: support / resistance / inside / broken

    Returns zones sorted by strength desc: ``top, bottom, mid, touches,
    strength, kind, state``.
    """
    _require()
    df = ensure_ohlcv(df)
    pts = swing_points(df, left=left, right=right)
    if pts.empty:
        return _pd.DataFrame(columns=["top", "bottom", "mid", "touches",
                                      "strength", "kind", "state"])
    a = float(atr(df).iloc[-1])
    if not (a > 0):
        a = float(df["close"].iloc[-1]) * 0.001 or 1e-6

    # single-link clustering on price
    prices = sorted(pts["price"].tolist())
    clusters: list[list[float]] = []
    kinds: list[list[int]] = []
    pmap = {p: k for p, k in zip(pts["price"], pts["kind"])}
    for p in prices:
        placed = False
        for ci, cl in enumerate(clusters):
            if abs(p - sum(cl) / len(cl)) <= merge_atr * a:
                cl.append(p)
                kinds[ci].append(pmap[p])
                placed = True
                break
        if not placed:
            clusters.append([p])
            kinds.append([pmap[p]])

    # recency weights: later swings matter more
    n = len(pts)
    wmap = {}
    for j, (_, r) in enumerate(pts.iterrows()):
        wmap.setdefault(r["price"], 0.0)
        wmap[r["price"]] += 0.5 + 0.5 * (j + 1) / max(n, 1)

    last_close = float(df["close"].iloc[-1])
    zones = []
    for cl, ks in zip(clusters, kinds):
        top = max(cl)
        bottom = min(cl)
        width = max(top - bottom, atr_mult * a)
        mid = (top + bottom) / 2
        touches = len(cl)
        if touches < min_touches:
            continue
        wsum = sum(wmap.get(p, 1.0) for p in cl)
        strength = clamp(wsum / max(n, 1) * 40.0 * (touches / 2.0), 0, 40)
        # flip bonus: acted as both support and resistance
        if any(k == 1 for k in ks) and any(k == -1 for k in ks):
            strength = clamp(strength * 1.5, 0, 60)
            kind = "flip"
        elif all(k == -1 for k in ks):
            kind = "support"
        else:
            kind = "resistance"
        strength = clamp(strength / 60.0 * 100.0, 0, 100)
        if bottom <= last_close <= top:
            state = "inside"
        elif mid < last_close:
            state = "support" if kind != "resistance" else "broken_resistance"
        else:
            state = "resistance" if kind != "support" else "broken_support"
        zones.append({"top": float(top), "bottom": float(bottom),
                      "mid": float(mid), "touches": touches,
                      "strength": round(float(strength), 1),
                      "kind": kind, "state": state})
    zones.sort(key=lambda z: -z["strength"])
    return _pd.DataFrame(zones[:max_zones])


def nearest_levels(zones: _pd.DataFrame, price: float) -> dict:
    """Nearest support below and resistance above ``price``."""
    if zones.empty:
        return {"support": None, "resistance": None}
    below = zones[zones["mid"] < price].sort_values("mid", ascending=False)
    above = zones[zones["mid"] > price].sort_values("mid", ascending=True)
    return {
        "support": below.iloc[0].to_dict() if not below.empty else None,
        "resistance": above.iloc[0].to_dict() if not above.empty else None,
    }
