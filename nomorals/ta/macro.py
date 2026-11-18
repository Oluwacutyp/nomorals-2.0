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

from ..core.logging_setup import get_logger

_log = get_logger(__name__)


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


__all__ = ["trinity_bias", "TrinityBias", "resolve_macro"]


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


def resolve_macro(macro) -> dict[str, Any] | None:
    """Resolve a macro input to a bias dict. Never raises.

    Accepts a bias dict (as returned by :func:`trinity_bias`), a
    :class:`TrinityBias` provider (consulted via its cached ``get()``),
    or None. Anything else — including a dict with an unknown bias
    word — resolves to None (no macro read).
    """
    if macro is None:
        return None
    if isinstance(macro, dict):
        bias = str(macro.get("bias") or "neutral").lower()
        if bias not in ("bullish", "bearish", "neutral"):
            return None
        return macro
    get = getattr(macro, "get", None)
    if callable(get):
        try:
            return resolve_macro(get())
        except Exception as exc:
            _log.debug("macro provider get() failed: %s", exc, exc_info=True)
            return None
    return None


class TrinityBias:
    """Cached macro-bias provider with background refresh.

    Mirrors Sentinel's TrinitySentinel: fetches GC=F / DX-Y.NYB / ^TNX
    daily closes, computes bias, caches for ``max_age_hours``.
    Thread-safe. Never raises on fetch failure — returns the last
    cached bias or None.

    The close fetcher is injectable: ``fetch_closes(symbol)`` returns
    ``(closes, last_bar_ts, source)``. The default fetcher is the
    legacy yfinance path (needs the ``yfinance`` package, which is NOT
    a declared dependency — boxes without it get no macro read).
    FinancialExpert injects the keyless market-data chain instead
    (Yahoo chart API → stooq, stdlib only), which is the live feed.
    """

    #: Yahoo symbols fetched, in trinity order (gold, dollar, yield).
    SYMBOLS = (("GC=F", "xau"), ("DX-Y.NYB", "dxy"), ("^TNX", "tny"))

    def __init__(self, max_age_hours: float = 24.0,
                 fetch_closes=None):
        import threading

        self.max_age_hours = max_age_hours
        self._fetch_closes = fetch_closes or self._yfinance_closes
        self._cache: dict[str, Any] | None = None
        self._readings: dict[str, Any] | None = None
        self._readings_ts: float = 0.0
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._thread = None

    def get(self) -> dict[str, Any] | None:
        """Last computed bias, or None if stale/missing."""
        import time

        c = self._cache
        if c and (time.time() - c.get("_ts", 0)) < self.max_age_hours * 3600:
            return c
        return None

    def readings(self) -> dict[str, Any] | None:
        """Per-symbol latest closes (``{key: {close, ts, source,
        status, error}}``), or None if stale/missing. A partial
        fetch still records its readings — the report can name
        exactly which leg failed instead of saying nothing."""
        import time

        if (self._readings is not None
                and (time.time() - self._readings_ts)
                < self.max_age_hours * 3600):
            return self._readings
        return None

    @property
    def fetched_at(self) -> float | None:
        """Epoch of the last fetch attempt, or None if never fetched."""
        return self._readings_ts or None

    def _should_fetch(self) -> bool:
        """True unless a fetch ran in the last 15 minutes."""
        import time

        now = time.time()
        c = self._cache
        recent_bias = c is not None and (now - c.get("_ts", 0)) < 900
        recent_readings = (self._readings is not None
                           and (now - self._readings_ts) < 900)
        return not (recent_bias or recent_readings)

    def refresh(self) -> None:
        """Synchronous fetch (bounded by the fetchers' own timeouts).

        No-op when a fetch ran in the last 15 minutes. Never raises —
        a failed fetch leaves ``last_error`` naming it.
        """
        with self._lock:
            if not self._should_fetch():
                return
        self._fetch()

    def refresh_async(self) -> None:
        """Kick off a background refresh (no-op if one is running)."""
        import threading

        with self._lock:
            if not self._should_fetch():
                return
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._fetch, daemon=True)
            self._thread.start()

    @staticmethod
    def _yfinance_closes(symbol: str):
        """Legacy fetcher: daily closes via yfinance (optional dep)."""
        import yfinance as yf

        df = yf.download(symbol, period="120d", interval="1d",
                         progress=False)
        if df is None or df.empty:
            raise TAError(f"yfinance: no data for {symbol}")
        if hasattr(df.columns, "get_level_values"):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception as exc:
                _log.debug("macro._fetch#2 failed", exc_info=True)
        cols = {str(c).lower(): c for c in df.columns}
        if "close" not in cols:
            raise TAError(f"yfinance: no close column for {symbol}")
        closes = df[cols["close"]].dropna().tolist()
        last_ts = None
        try:
            idx = df.index
            if len(idx):
                last = idx[-1]
                last_ts = (float(last.value // 10 ** 9)
                           if hasattr(last, "value") else float(last))
        except Exception as exc:  # noqa: BLE001 - ts is surface only
            _log.debug("macro yfinance ts failed: %s", exc)
        return [float(c) for c in closes], last_ts, "yfinance"

    def _fetch(self) -> None:
        import time

        readings: dict[str, Any] = {}
        closes: dict[str, list[float]] = {}
        errors: dict[str, str] = {}
        for sym, key in self.SYMBOLS:
            try:
                got = self._fetch_closes(sym)
                cls, last_ts, source = got
                cls = [float(c) for c in cls]
                if not cls:
                    raise TAError(f"no closes for {sym}")
                closes[key] = cls
                readings[key] = {"close": cls[-1], "ts": last_ts,
                                 "source": source or "unknown",
                                 "status": "ok", "error": None}
            except Exception as exc:  # noqa: BLE001 - one bad leg ≠ no read
                errors[key] = f"{sym}: {exc}"
                readings[key] = {"close": None, "ts": None, "source": None,
                                 "status": "error",
                                 "error": str(exc)[:160]}
        now = time.time()
        with self._lock:
            self._readings = readings
            self._readings_ts = now
            self.last_error = ("; ".join(errors.values())
                               if errors else None)
        if len(closes) != len(self.SYMBOLS):
            return
        try:
            n = min(len(v) for v in closes.values())
            bias = trinity_bias(closes["xau"][-n:], closes["dxy"][-n:],
                                closes["tny"][-n:])
            if bias:
                bias["_ts"] = now
                bias["readings"] = readings
                with self._lock:
                    self._cache = bias
        except Exception as exc:
            _log.debug("macro._fetch failed", exc_info=True)
            with self._lock:
                self.last_error = str(exc)[:160]
