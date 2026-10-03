"""Live OHLCV feeds from the connector layer (Binance, Coinbase).

Pulls klines/candles through Devon's own connector adapters and returns
the exact DataFrame shape the indicators expect (open/high/low/close/
volume with a DatetimeIndex). Uses public market-data endpoints where the
exchange allows; fails fast with a plain-English message when the
connector is not connected.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from .math import ensure_ohlcv

_log = logging.getLogger(__name__)

__all__ = [
    "SOURCES",
    "fetch_ohlcv",
    "fetch_binance",
    "fetch_coinbase",
]

SOURCES = ("binance", "coinbase")

_BINANCE_INTERVALS = (
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
)
_COINBASE_GRANULARITY = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 21600, "1d": 86400,
}
_COINBASE_MAX_CANDLES = 300
_COINBASE_QUOTES = {"USDT": "USD", "USDC": "USD", "USD": "USD"}


def _get_connector(source: str, *, connector=None, vault=None,
                   http=None):
    """Instantiate (or accept) the connector; fail fast when unconnected.

    ``connector`` is an injection hook for tests — pass a real adapter in
    production by leaving it ``None`` and supplying ``vault``.
    """
    if connector is not None:
        conn = connector
    else:
        if vault is None:
            raise ValueError(
                f"fetching {source} OHLCV needs a CredentialVault "
                "(or pass a connected adapter via connector=)"
            )
        from importlib import import_module  # lazy: heavy tree, avoids L3->L5 import
        _registry = import_module("nomorals.connectors.registry")
        conn = _registry.create_connector(source, vault, http=http)
    try:
        status = conn.status()
    except Exception as exc:  # noqa: BLE001 - status must never wedge feeds
        raise RuntimeError(
            f"{source} connector status check failed: {exc}"
        ) from exc
    if not getattr(status, "connected", False):
        raise RuntimeError(
            f"{source} is not connected — run "
            f"`nm connectors connect --name {source}` first"
        )
    return conn


def _binance_interval(interval: str) -> str:
    iv = (interval or "").strip()
    if iv not in _BINANCE_INTERVALS:
        raise ValueError(
            f"unsupported binance interval {interval!r}; "
            f"choose from {', '.join(_BINANCE_INTERVALS)}"
        )
    return iv


def fetch_binance(symbol: str, interval: str = "1h", limit: int = 200, *,
                  vault=None, http=None, connector=None) -> pd.DataFrame:
    """Binance spot klines as an OHLCV frame.

    Uses the public ``/api/v3/klines`` endpoint (no signed key needed for
    market data) but still fails fast when the connector is not connected,
    per the connector policy.
    """
    conn = _get_connector("binance", connector=connector, vault=vault,
                          http=http)
    sym = (symbol or "").strip().upper().replace("-", "")
    if not sym:
        raise ValueError("symbol is required (e.g. 'BTCUSDT')")
    limit = max(1, min(int(limit), 1000))
    rows = conn._public("GET", "/api/v3/klines", {
        "symbol": sym, "interval": _binance_interval(interval),
        "limit": limit,
    })
    if not rows:
        raise RuntimeError(f"binance returned no klines for {sym}")
    df = pd.DataFrame([
        {"open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
         "close": float(r[4]), "volume": float(r[5])}
        for r in rows
    ], index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True))
    return ensure_ohlcv(df)


def _coinbase_product(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    if "-" in sym:
        return sym
    for quote, cb_quote in _COINBASE_QUOTES.items():
        if sym.endswith(quote) and len(sym) > len(quote):
            return f"{sym[:-len(quote)]}-{cb_quote}"
    raise ValueError(
        f"cannot map {symbol!r} to a Coinbase product id — pass "
        "exchange format like 'BTC-USD' or 'BTCUSDT'"
    )


def fetch_coinbase(symbol: str, interval: str = "1h", limit: int = 200, *,
                   vault=None, http=None, connector=None) -> pd.DataFrame:
    """Coinbase Advanced Trade candles as an OHLCV frame.

    The candles endpoint is paged at 300 candles per call; this walks
    backwards in chunks until ``limit`` bars are collected. Requires the
    coinbase connector to be connected (the API call is authenticated).
    """
    conn = _get_connector("coinbase", connector=connector, vault=vault,
                          http=http)
    product_id = _coinbase_product(symbol)
    iv = (interval or "").strip()
    if iv not in _COINBASE_GRANULARITY:
        raise ValueError(
            f"unsupported coinbase interval {interval!r}; choose from "
            f"{', '.join(_COINBASE_GRANULARITY)}"
        )
    gran = _COINBASE_GRANULARITY[iv]
    limit = max(1, min(int(limit), 1000))
    end = int(time.time())
    collected: list[dict[str, Any]] = []
    while len(collected) < limit:
        chunk_end = end - len(collected) * gran
        chunk_start = chunk_end - min(
            limit - len(collected), _COINBASE_MAX_CANDLES) * gran
        data = conn._api(
            "GET",
            f"/api/v3/brokerage/market/products/{product_id}/candles",
            params={"start": chunk_start, "end": chunk_end,
                    "granularity": gran},
        )
        candles = data.get("candles") or []
        if not candles:
            break
        collected.extend(candles)
    if not collected:
        raise RuntimeError(
            f"coinbase returned no candles for {product_id}")
    df = pd.DataFrame([
        {"open": float(c["open"]), "high": float(c["high"]),
         "low": float(c["low"]), "close": float(c["close"]),
         "volume": float(c["volume"])}
        for c in collected
    ], index=pd.to_datetime([c["start"] for c in collected], unit="s",
                             utc=True))
    df = df.sort_index()
    return ensure_ohlcv(df.tail(limit))


def fetch_ohlcv(source: str, symbol: str, interval: str = "1h",
                limit: int = 200, *, vault=None, http=None,
                connector=None) -> pd.DataFrame:
    """OHLCV from any supported connector source.

    ``source`` is one of ``SOURCES`` (``"binance"`` | ``"coinbase"``);
    ``symbol`` is the venue's format (``"BTCUSDT"`` / ``"BTC-USD"`` —
    the common forms are normalized). Raises ``ValueError`` on unknown
    sources and fails fast when the connector is not connected.
    """
    src = (source or "").strip().lower()
    if src == "binance":
        return fetch_binance(symbol, interval, limit, vault=vault,
                             http=http, connector=connector)
    if src == "coinbase":
        return fetch_coinbase(symbol, interval, limit, vault=vault,
                              http=http, connector=connector)
    raise ValueError(
        f"unknown OHLCV source {source!r}; supported: "
        f"{', '.join(SOURCES)}")
