"""Forex sessions, killzones, and time-of-day filters.

Timing is a first-class input for FX — especially XAUUSD, where the
London–NY overlap (13:00–17:00 UTC) carries extreme volatility and the
Asian session is mostly chop. ICT killzones name the same windows.

All times UTC. Session boundaries are conventional, not exchange-mandated
(FX is OTC); treat them as volatility priors, not laws.
"""

from __future__ import annotations

import logging
from datetime import time as _time

_log = logging.getLogger(__name__)


class TAError(Exception):
    """Raised when a TA operation cannot be completed."""


try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


def _require() -> None:
    if not _HAS_PANDAS:
        raise TAError("pandas is required: pip install nomorals[ta]")


# ── session definitions (UTC) ─────────────────────────────────────────
# Convergent across quantvps / analyticsinsight / scribehow / ICT sources.

SESSIONS = {
    "asian": {"start": _time(0, 0), "end": _time(8, 0),
              "volatility": "low",
              "note": "Tokyo/Sydney. Chop, fake breakouts. Range tactics only."},
    "london": {"start": _time(8, 0), "end": _time(13, 0),
               "volatility": "high",
               "note": "London open. Clean structure, breakout momentum."},
    "overlap": {"start": _time(13, 0), "end": _time(17, 0),
                "volatility": "extreme",
                "note": "London–NY overlap. Best XAUUSD window. Tight spreads, "
                        "directional bias."},
    "new_york": {"start": _time(17, 0), "end": _time(22, 0),
                 "volatility": "moderate",
                 "note": "Late NY. Momentum continuation, then fade. "
                         "Danger zone after 17:00 for scalps."},
    "off": {"start": _time(22, 0), "end": _time(0, 0),
            "volatility": "low",
            "note": "Dead hours. Wide spreads, no edge for intraday."},
}

# ICT killzones in ET (as taught); converted here to UTC for computation.
# Asian KZ 20:00–00:00 ET = 00:00–04:00 UTC (EST) — approximated below.
KILLZONES_UTC = {
    "asian_kz": (_time(0, 0), _time(4, 0)),
    "london_kz": (_time(7, 0), _time(10, 0)),
    "ny_kz": (_time(12, 0), _time(15, 0)),
}


def _in_window(t: _time, start: _time, end: _time) -> bool:
    if start <= end:
        return start <= t < end
    return t >= start or t < end  # wraps midnight


def session_at(ts) -> str:
    """Session name for a timestamp (UTC)."""
    _require()
    t = _pd.Timestamp(ts).tz_convert("UTC").time() \
        if _pd.Timestamp(ts).tzinfo else _pd.Timestamp(ts).time()
    for name, s in SESSIONS.items():
        if _in_window(t, s["start"], s["end"]):
            return name
    return "off"


def in_killzone(ts, killzone: str = "any") -> bool:
    """Is ``ts`` inside an ICT killzone (UTC approx)?"""
    _require()
    t = _pd.Timestamp(ts).tz_convert("UTC").time() \
        if _pd.Timestamp(ts).tzinfo else _pd.Timestamp(ts).time()
    zones = KILLZONES_UTC if killzone == "any" else \
        {killzone: KILLZONES_UTC[killzone]}
    return any(_in_window(t, s, e) for s, e in zones.values())


def session_profile(df) -> _pd.DataFrame:
    """Per-bar session + killzone labels for an OHLCV frame."""
    _require()
    idx = _pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return _pd.DataFrame(
        {"session": [session_at(t) for t in idx],
         "killzone": [in_killzone(t) for t in idx]},
        index=df.index)


def session_volatility(df, sessions: tuple = ("asian", "london", "overlap",
                                              "new_york")) -> dict:
    """Mean absolute bar range per session — the volatility prior, measured.

    Returns ``{session: {"mean_range_atr": x, "bars": n}}`` so the desk can
    size stops by session instead of guessing.
    """
    _require()
    from .math import atr, ensure_ohlcv
    df = ensure_ohlcv(df)
    prof = session_profile(df)
    a = atr(df).replace(0, float("nan"))
    rng = (df["high"] - df["low"]) / a
    out = {}
    for s in sessions:
        m = prof["session"] == s
        if m.sum() == 0:
            continue
        out[s] = {"mean_range_atr": round(float(rng[m].mean()), 3),
                  "bars": int(m.sum()),
                  "volatility": SESSIONS[s]["volatility"]}
    return out


def time_filter_ok(ts=None, allowed=("london", "overlap", "new_york")) -> dict:
    """Should Devon trade right now? Time-of-day gate for intraday systems.

    Returns ``{"ok": bool, "session": str, "killzone": bool, "reason": str}``.
    News blackouts (NFP/CPI/FOMC) are a separate feed — this is the session
    layer only.
    """
    _require()
    ts = _pd.Timestamp.now(tz="UTC") if ts is None else _pd.Timestamp(ts)
    sess = session_at(ts)
    kz = in_killzone(ts)
    if sess in allowed:
        return {"ok": True, "session": sess, "killzone": kz,
                "reason": f"{sess} session — volatility {SESSIONS[sess]['volatility']}"
                          + (", inside killzone" if kz else "")}
    return {"ok": False, "session": sess, "killzone": kz,
            "reason": f"{sess} session — {SESSIONS[sess]['note']}"}


def day_of_week_stats(df) -> dict:
    """Mean daily range by weekday — weekend-gap and weekday seasonality."""
    _require()
    from .math import ensure_ohlcv
    df = ensure_ohlcv(df)
    daily = df.resample("1D").agg({"high": "max", "low": "min"})
    daily = daily.dropna()
    if daily.empty:
        return {}
    rng = daily["high"] - daily["low"]
    out = {}
    for dow, grp in rng.groupby(rng.index.dayofweek):
        out[int(dow)] = {"mean_range": round(float(grp.mean()), 5),
                         "days": int(len(grp))}
    return out
