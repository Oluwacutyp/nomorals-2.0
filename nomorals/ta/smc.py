"""Smart Money Concepts: order blocks, fair value gaps, liquidity sweeps.

HONEST FRAMING — read before using:
Every concept here is a renamed classical idea (see the Rosetta Stone in
FOREX_TA_MINING.md). Order blocks are supply/demand zones; fair value gaps
are price imbalances; liquidity sweeps are stop hunts. The SMC vocabulary
is useful because it is systematic, not because it reveals secret
institutional knowledge — a chart cannot prove who traded or why.

What SMC gets right: structure first, imbalances as magnets, stop-hunt
awareness, multi-timeframe location (premium/discount). What it gets wrong
in guru hands: hindsight labeling (drawing the OB after the reversal),
certainty language ("banks NEED this liquidity"), and ignoring that every
one of these patterns fails regularly.

Detection here is PROSPECTIVE: an event fires on the bar where all its
conditions are observable. No lookahead, no repainting.
"""

from __future__ import annotations

import logging

from .math import atr, clamp, ensure_ohlcv
from .structure import structure_breaks, swings

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


# ── fair value gaps ─────────────────────────────────────────────────────

def fair_value_gaps(df, min_atr: float = 0.1) -> _pd.DataFrame:
    """Fair value gaps (3-candle imbalance).

    Bullish FVG: low[i] > high[i-2] — the wicks of candle i-2 and i do not
    overlap, leaving a void on candle i-1. Bearish mirrors it.
    Gaps smaller than ``min_atr`` × ATR are noise, not signal.

    Returns: ``bar`` (the middle candle), ``kind`` (+1 bull / −1 bear),
    ``top``, ``bottom``, ``mid``, ``size_atr``, ``mitigated`` (price has
    since traded through the gap — filled gaps lose their edge).
    """
    _require()
    df = ensure_ohlcv(df)
    a = atr(df).replace(0, _np.nan).ffill()
    hi, lo, cl = df["high"], df["low"], df["close"]
    rows = []
    for i in range(2, len(df)):
        gap = None
        if lo.iloc[i] > hi.iloc[i - 2]:
            gap = {"kind": 1, "bottom": float(hi.iloc[i - 2]),
                   "top": float(lo.iloc[i])}
        elif hi.iloc[i] < lo.iloc[i - 2]:
            gap = {"kind": -1, "bottom": float(hi.iloc[i]),
                   "top": float(lo.iloc[i - 2])}
        if gap is None:
            continue
        size_atr = (gap["top"] - gap["bottom"]) / max(float(a.iloc[i]), 1e-12)
        if size_atr < min_atr:
            continue
        # mitigated? price traded fully through the gap after formation
        mitig = False
        if gap["kind"] == 1:
            mitig = bool((lo.iloc[i + 1:] <= gap["bottom"]).any()) \
                if i + 1 < len(df) else False
        else:
            mitig = bool((hi.iloc[i + 1:] >= gap["top"]).any()) \
                if i + 1 < len(df) else False
        rows.append({"bar": df.index[i - 1], "kind": gap["kind"],
                     "top": gap["top"], "bottom": gap["bottom"],
                     "mid": (gap["top"] + gap["bottom"]) / 2,
                     "size_atr": round(size_atr, 2), "mitigated": mitig})
    return _pd.DataFrame(rows)


def unmitigated_fvgs(df, **kw) -> _pd.DataFrame:
    """FVGs that price has not yet filled — the ones that still matter."""
    f = fair_value_gaps(df, **kw)
    if f.empty:
        return f
    return f[~f["mitigated"]].reset_index(drop=True)


# ── order blocks ────────────────────────────────────────────────────────

def order_blocks(df, left: int = 5, right: int = 3,
                 impulse_atr: float = 1.5) -> _pd.DataFrame:
    """Order blocks: last opposite candle before an impulsive BOS move.

    Classical equivalent: supply/demand base, Wyckoff accumulation,
    Rally-Base-Rally. Detection: for each BOS event, take the last
    opposite-direction candle before the displacement leg, provided the
    leg's range ≥ ``impulse_atr`` × ATR (impulse, not drift).

    Returns: ``bar``, ``kind`` (+1 bull / −1 bear), ``top``, ``bottom``,
    ``mid``, ``mitigated`` (close through the zone = broken → polarity
    flips; a broken bullish OB becomes bearish resistance).
    """
    _require()
    df = ensure_ohlcv(df)
    a = atr(df).replace(0, _np.nan).ffill()
    op, hi, lo, cl = df["open"], df["high"], df["low"], df["close"]
    brk = structure_breaks(df, left=left, right=right)
    rows = []
    for _, e in brk.iterrows():
        if not e["kind"].startswith("BOS"):
            continue
        direction = 1 if e["kind"] == "BOS_BULL" else -1
        bi = df.index.get_loc(e["bar"])
        # displacement leg: from the break bar, walk back to the impulse start
        # (first bar of the strong move). Simplify: last opposite candle
        # within the 10 bars before the break whose range contributed.
        start = max(0, bi - 10)
        ob_idx = None
        for j in range(bi - 1, start - 1, -1):
            is_down = cl.iloc[j] < op.iloc[j]
            if (direction == 1 and is_down) or (direction == -1 and not is_down):
                ob_idx = j
                break
        if ob_idx is None:
            continue
        leg_range = float(hi.iloc[ob_idx:bi + 1].max()
                          - lo.iloc[ob_idx:bi + 1].min())
        if leg_range < impulse_atr * float(a.iloc[bi]):
            continue  # drift, not displacement
        top = float(hi.iloc[ob_idx])
        bottom = float(lo.iloc[ob_idx])
        # mitigated: later close through the far side
        if direction == 1:
            mitig = bool((cl.iloc[bi + 1:] < bottom).any()) \
                if bi + 1 < len(df) else False
        else:
            mitig = bool((cl.iloc[bi + 1:] > top).any()) \
                if bi + 1 < len(df) else False
        rows.append({"bar": df.index[ob_idx], "kind": direction,
                     "top": top, "bottom": bottom,
                     "mid": (top + bottom) / 2, "mitigated": mitig,
                     "bos_bar": e["bar"]})
    out = _pd.DataFrame(rows)
    if not out.empty:
        # a mitigated bullish OB flips to bearish resistance (breaker block)
        out["effective_kind"] = out.apply(
            lambda r: -r["kind"] if r["mitigated"] else r["kind"], axis=1)
    return out


# ── liquidity sweeps ──────────────────────────────────────────────────

def liquidity_sweeps(df, left: int = 5, right: int = 5,
                     pen_atr: float = 0.25, reclaim_bars: int = 3) -> _pd.DataFrame:
    """Liquidity sweeps (stop hunts), prospective.

    A sweep = penetration of the most recent confirmed swing level by at
    least ``pen_atr`` × ATR, followed by a close back past that level
    within ``reclaim_bars`` bars. Stamped on the RECLAIM bar — every input
    is observable at trade time. No hindsight.

    Returns: ``bar`` (reclaim), ``kind`` (+1 = swept lows → bullish /
    −1 = swept highs → bearish), ``level``, ``penetration_atr``.
    """
    _require()
    df = ensure_ohlcv(df)
    from .structure import swing_points
    pts = swing_points(df, left=left, right=right)
    a = atr(df).replace(0, _np.nan).ffill()
    hi, lo, cl = df["high"], df["low"], df["close"]
    events = []
    for _, p in pts.iterrows():
        pi = df.index.get_loc(p["bar"])
        level = float(p["price"])
        pen = pen_atr * float(a.iloc[pi])
        if p["kind"] == -1:  # swing low → sweep of lows is bullish
            for j in range(pi + 1, min(pi + 1 + 20, len(df))):
                if float(lo.iloc[j]) < level - pen:
                    # penetrated; look for reclaim within reclaim_bars
                    for k in range(j + 1, min(j + 1 + reclaim_bars, len(df))):
                        if float(cl.iloc[k]) > level:
                            events.append({
                                "bar": df.index[k], "kind": 1,
                                "level": level,
                                "penetration_atr": round(
                                    (level - float(lo.iloc[j])) /
                                    max(float(a.iloc[j]), 1e-12), 2)})
                            break
                    break
        else:  # swing high → sweep of highs is bearish
            for j in range(pi + 1, min(pi + 1 + 20, len(df))):
                if float(hi.iloc[j]) > level + pen:
                    for k in range(j + 1, min(j + 1 + reclaim_bars, len(df))):
                        if float(cl.iloc[k]) < level:
                            events.append({
                                "bar": df.index[k], "kind": -1,
                                "level": level,
                                "penetration_atr": round(
                                    (float(hi.iloc[j]) - level) /
                                    max(float(a.iloc[j]), 1e-12), 2)})
                            break
                    break
    out = _pd.DataFrame(events)
    if not out.empty:
        out = out.sort_values("bar").reset_index(drop=True)
    return out


# ── premium / discount ────────────────────────────────────────────────

def premium_discount(df, lookback: int = 100) -> _pd.DataFrame:
    """Premium/discount position within the dealing range.

    Range = highest high / lowest low over ``lookback``. Above the 50%
    equilibrium = premium (sellers' territory); below = discount (buyers').
    Classical equivalent: overbought/oversold, above/below mean.
    Returns ``premium`` 0–1 (0.5 = equilibrium) and ``zone`` label.
    """
    _require()
    df = ensure_ohlcv(df)
    hh = df["high"].rolling(lookback, min_periods=1).max()
    ll = df["low"].rolling(lookback, min_periods=1).min()
    rng = (hh - ll).replace(0, _np.nan)
    prem = (df["close"] - ll) / rng
    prem = prem.fillna(0.5).clip(0, 1)
    zone = _pd.cut(prem, bins=[0, 0.35, 0.65, 1.0],
                   labels=["discount", "equilibrium", "premium"])
    return _pd.DataFrame({"premium": prem.round(3), "zone": zone,
                          "range_high": hh, "range_low": ll},
                         index=df.index)


# ── confluence scorer ─────────────────────────────────────────────────

def smc_confluence(df, **kw) -> dict:
    """One-dict SMC read on the latest bar.

    Scores −100…+100 from: structure bias (±30), unmitigated FVG proximity
    (±20), order-block proximity (±20), fresh liquidity sweep (±20),
    premium/discount location (±10, fades extremes). Honest about what it
    is: a systematic discretionary checklist, quantified — not a signal
    with a proven edge. Backtest before trusting.
    """
    _require()
    df = ensure_ohlcv(df)
    from .structure import label_swings, market_bias, swing_points
    close = float(df["close"].iloc[-1])
    a = float(atr(df).iloc[-1]) or 1e-9

    score = 0.0
    notes = []

    pts = swing_points(df)
    if not pts.empty:
        b = market_bias(label_swings(pts))
        score += 30 * b["bias"]
        notes.append(f"structure: {b['label']} ({b['reason']})")

    for f in unmitigated_fvgs(df).itertuples():
        d = abs(close - f.mid) / a
        if d < 1.0:
            s = 20 * f.kind * max(0.0, 1.0 - d)
            score += s
            notes.append(f"{'bullish' if f.kind > 0 else 'bearish'} FVG "
                         f"{d:.1f} ATR away ({s:+.0f})")

    for o in order_blocks(df).itertuples():
        if o.mitigated:
            continue
        d = abs(close - o.mid) / a
        if d < 1.5:
            k = o.kind
            s = 20 * k * max(0.0, 1.0 - d / 1.5)
            score += s
            notes.append(f"order block {d:.1f} ATR away ({s:+.0f})")

    sw = liquidity_sweeps(df)
    if not sw.empty:
        last = sw.iloc[-1]
        bars_ago = len(df) - 1 - df.index.get_loc(last["bar"])
        if bars_ago <= 5:
            s = 20 * last["kind"] * max(0.0, 1.0 - bars_ago / 5)
            score += s
            notes.append(f"liquidity sweep {bars_ago} bars ago ({s:+.0f})")

    pd_ = premium_discount(df).iloc[-1]
    if pd_["zone"] == "premium":
        score -= 10 * float(pd_["premium"])
        notes.append(f"in premium ({float(pd_['premium']):.0%}) — longs fade")
    elif pd_["zone"] == "discount":
        score += 10 * (1 - float(pd_["premium"]))
        notes.append(f"in discount ({float(pd_['premium']):.0%}) — shorts fade")

    score = clamp(score, -100, 100)
    return {"score": round(float(score), 1),
            "bias": 1 if score > 15 else -1 if score < -15 else 0,
            "notes": notes,
            "disclaimer": "SMC vocabulary systematizes discretion; it does "
                          "not confer edge. Backtest on this instrument."}
