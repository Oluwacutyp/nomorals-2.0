"""Macro bias for precious metals: the "Trinity" sentinel.

Ported from Sentinel.py's ``TrinitySentinel.compute_bias`` (user's own bot).
The insight: gold doesn't trade in a vacuum. DXY (dollar index) and US10Y
(10-year yield) trends + the XAU/DXY correlation tell you the macro wind
direction. This is a tie-breaker/veto layer, NOT an alpha source — the
backtest proved it moves results ±2pp, not ±20pp.

Honest framing from the Sentinel backtest report:
"Trinity voting + ADR floor trimmed losses slightly; still halted by the
20% emergency stop. It is a tie-breaker/veto layer, not an alpha source."
"""

from __future__ import annotations

from typing import Any


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


def _require_deps() -> None:
    if not (_HAS_NUMPY and _HAS_PANDAS):
        raise TAError("numpy and pandas required: pip install nomorals[ta]")


__all__ = ["trinity_bias", "TrinityBias"]


def trinity_bias(xau_closes, dxy_closes, tny_closes) -> dict[str, Any] | None:
    """Deterministic macro bias from 20d trends + 50d XAU/DXY correlation.

    Args:
        xau_closes: gold daily closes (list or Series, need 55+)
        dxy_closes: dollar index daily closes
        tny_closes: 10-year yield daily closes (in yield points)

    Returns:
        dict with bias ('bullish'/'bearish'/'neutral'), dxy_trend,
        y10y_trend, corr, decoupled flag, and reasons. None if
        insufficient data.

    Logic (from Sentinel, research-published thresholds):
    - DXY 20d < -0.2% AND US10Y 20d < -0.03pts → bullish gold
      (weak dollar + falling yields = gold tailwind)
    - DXY 20d > +0.2% AND US10Y 20d > +0.03pts → bearish gold
      (strong dollar + rising yields = gold headwind)
    - Otherwise neutral.
    - |corr50| < 0.3 → "decoupled" flag (classic negative XAU/DXY
      link is weakening — treat bias with extra skepticism).
    """
    _require_deps()
    xc = list(xau_closes)
    dc = list(dxy_closes)
    tc = list(tny_closes)
    if len(xc) < 55 or len(dc) < 55 or len(tc) < 55:
        return None

    dxy20 = (dc[-1] / dc[-21] - 1.0) * 100.0 if dc[-21] else 0.0
    tny20 = tc[-1] - tc[-21]  # yield points

    xr = _pd.Series(xc[-55:]).pct_change().dropna()
    dr = _pd.Series(dc[-55:]).pct_change().dropna()
    corr = float(xr.corr(dr)) if len(xr) >= 30 and len(dr) >= 30 else 0.0
    # NaN correlation (flat series) → treat as decoupled
    if corr != corr:  # NaN check
        corr = 0.0
    decoupled = abs(corr) < 0.3

    if dxy20 < -0.2 and tny20 < -0.03:
        bias = "bullish"
    elif dxy20 > 0.2 and tny20 > 0.03:
        bias = "bearish"
    else:
        bias = "neutral"

    reasons = [
        f"DXY 20d {dxy20:+.2f}%, US10Y 20d {tny20:+.2f}pts, corr50 {corr:+.2f}"
    ]
    if decoupled:
        reasons.append(
            "XAU/DXY decoupling — classic negative link weakening, "
            "treat bias with skepticism"
        )
    return {
        "dxy_trend": dxy20,
        "y10y_trend": tny20,
        "corr": corr,
        "decoupled": decoupled,
        "bias": bias,
        "reasons": reasons,
    }


class TrinityBias:
    """Cached macro-bias provider with background refresh.

    Mirrors Sentinel's TrinitySentinel: fetches GC=F / DX-Y.NYB / ^TNX
    daily closes (via yfinance when available), computes bias, caches
    for ``max_age_hours``. Thread-safe. Never raises on fetch failure —
    returns the last cached bias or None.
    """

    def __init__(self, max_age_hours: float = 24.0):
        import threading

        self.max_age_hours = max_age_hours
        self._cache: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._thread = None

    def get(self) -> dict[str, Any] | None:
        """Last computed bias, or None if stale/missing."""
        import time

        c = self._cache
        if c and (time.time() - c.get("_ts", 0)) < self.max_age_hours * 3600:
            return c
        return None

    def refresh_async(self) -> None:
        """Kick off a background refresh (no-op if one is running)."""
        import threading
        import time

        with self._lock:
            c = self._cache
            if c and (time.time() - c.get("_ts", 0)) < 900:  # 15 min min gap
                return
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._fetch, daemon=True)
            self._thread.start()

    def _fetch(self) -> None:
        import time

        try:
            import yfinance as yf
        except ImportError:
            return
        try:
            frames = {}
            for sym, key in (("GC=F", "xau"), ("DX-Y.NYB", "dxy"),
                             ("^TNX", "tny")):
                df = yf.download(sym, period="120d", interval="1d",
                                 progress=False)
                if df is None or df.empty:
                    return
                if hasattr(df.columns, "get_level_values"):
                    try:
                        df.columns = df.columns.get_level_values(0)
                    except Exception:
                        pass
                cols = {str(c).lower(): c for c in df.columns}
                if "close" not in cols:
                    return
                frames[key] = df[cols["close"]].dropna().tolist()
            n = min(len(v) for v in frames.values())
            bias = trinity_bias(frames["xau"][-n:], frames["dxy"][-n:],
                                frames["tny"][-n:])
            if bias:
                bias["_ts"] = time.time()
                with self._lock:
                    self._cache = bias
        except Exception:
            pass  # background fetch never raises
