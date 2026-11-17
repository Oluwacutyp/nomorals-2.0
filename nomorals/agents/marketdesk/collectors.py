"""Market collectors — fetch OHLCV bars into the desk's ledger.

One collector shape for every symbol: the desk studies ONE candle
shape, feeds are data.  The fetch seam is
``fetch(symbol, market, timeframe, bars) -> [row dicts]`` — the
production fetch is :func:`integrations.market_data.get_ohlcv`
(keyless-first, health-ordered chain: crypto via binance/kraken/
coingecko, stocks via yahoo/stooq with keyed upgrades only when
their env keys are set).  A fetch that fails reports its failure in
the outcome; a dead leg never kills the cycle.

Symbols: BTCUSD + US majors.  NGX has no keyless feed in the chain
(stooq spells stocks ``<ticker>.us``; Yahoo's Lagos suffix is not a
seeded spelling anywhere in the price class), so stocks stay US
majors and the module says so instead of inventing a feed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from .failures import record_failure
from .store import Candle, CandleStore

__all__ = [
    "DEFAULT_SYMBOLS", "SYMBOL_MARKETS", "collect_symbol",
    "collect_all", "fetch_rows",
]

#: The desk's standing watchlist.  XAUUSD is deliberately absent —
#: gold stays with GoldenSwing + Devon trading.
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTCUSD", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
)

#: Price-class market per symbol (``integrations.market_data``).
SYMBOL_MARKETS: dict[str, str] = {
    "BTCUSD": "crypto",
    "AAPL": "stocks",
    "MSFT": "stocks",
    "NVDA": "stocks",
    "TSLA": "stocks",
    "AMZN": "stocks",
}

#: Bars fetched per timeframe per cycle (daily ~1.4y, hourly 30d).
DAILY_BARS = 500
HOURLY_BARS = 720

#: Timeframes the lab reads.  Daily bars carry structure; hourly bars
#: carry the time-of-day structure the signal engine can point at.
TIMEFRAMES = ("1d", "1h")


def _df_to_rows(symbol: str, timeframe: str, df: Any) -> list[dict[str, Any]]:
    """Convert a ``get_ohlcv`` DataFrame (DatetimeIndex, OHLCV cols)
    into the collector's row dicts.  Rows with no usable close are
    dropped, never stored."""
    rows: list[dict[str, Any]] = []
    try:
        records = df.itertuples()
    except AttributeError:
        return rows
    for rec in records:
        try:
            ts = float(rec.Index.timestamp())
            close = float(rec.close)
        except (AttributeError, TypeError, ValueError):
            continue
        if not (ts and math.isfinite(close) and close > 0):
            continue
        rows.append({
            "symbol": symbol, "timeframe": timeframe, "ts": ts,
            "open": float(rec.open), "high": float(rec.high),
            "low": float(rec.low), "close": close,
            "volume": float(rec.volume or 0.0),
        })
    return rows


def fetch_rows(symbol: str, market: str, timeframe: str,
               bars: int) -> list[dict[str, Any]]:
    """Production fetch: one timeframe of bars via the price class.

    Keyless-first by construction (``get_ohlcv`` tries the market's
    default chain in order; keyed upgrades only when their env keys
    are set).  Raises the price class's error, with every failure
    named, when the whole chain fails.
    """
    from ...integrations.market_data import get_ohlcv
    df = get_ohlcv(symbol, market=market, timeframe=timeframe,
                   bars=bars)
    return _df_to_rows(symbol, timeframe, df)


def collect_symbol(store: CandleStore, symbol: str, *,
                   fetch: Callable | None = None) -> dict[str, Any]:
    """Fetch one symbol's daily + hourly bars into the store.

    Returns one outcome dict: ``ok`` + per-timeframe counts, or
    ``ok=False`` + the error.  A failure is data, never an exception
    escaping into the cycle.
    """
    symbol = (symbol or "").strip().upper()
    market = SYMBOL_MARKETS.get(symbol, "crypto")
    fetch = fetch or fetch_rows
    out: dict[str, Any] = {"symbol": symbol, "market": market,
                           "ok": True, "added": 0, "timeframes": {},
                           "source": "injected" if fetch is not fetch_rows
                                     else "market_data.get_ohlcv"}
    try:
        for timeframe, bars in (("1d", DAILY_BARS), ("1h", HOURLY_BARS)):
            rows = fetch(symbol, market, timeframe, bars)
            candles = [Candle(symbol=r.get("symbol", symbol),
                              timeframe=r.get("timeframe", timeframe),
                              ts=float(r.get("ts") or 0.0),
                              open=float(r.get("open") or 0.0),
                              high=float(r.get("high") or 0.0),
                              low=float(r.get("low") or 0.0),
                              close=float(r.get("close") or 0.0),
                              volume=float(r.get("volume") or 0.0),
                              source=out["source"])
                       for r in (rows or [])]
            landed = store.append(candles)
            out["timeframes"][timeframe] = landed
            out["added"] += landed["added"]
    except Exception as exc:  # noqa: BLE001 - a leg reports, not raises
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        record_failure("collectors.collect_symbol", exc)
    return out


def collect_all(store: CandleStore,
                symbols: Iterable[str] | None = None, *,
                fetch: Callable | None = None) -> dict[str, Any]:
    """One collection cycle over every watched symbol.

    Every symbol reports its own outcome; a dead symbol never kills
    the cycle.
    """
    out: dict[str, Any] = {}
    for symbol in (symbols or DEFAULT_SYMBOLS):
        out[symbol] = collect_symbol(store, symbol, fetch=fetch)
    return out
