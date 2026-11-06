"""Live OHLCV feeds from the connector layer (Binance, Coinbase).

Pulls klines/candles through Devon's own connector adapters and returns
the exact DataFrame shape the indicators expect (open/high/low/close/
volume with a DatetimeIndex). Uses public market-data endpoints where the
exchange allows; fails fast with a plain-English message when the
connector is not connected.

Open-sources law (AA-53): the four exchanges in :data:`SOURCES` are
SEEDS, not a closed dispatch.  :func:`fetch_ohlcv` walks them in the
price class's health-ordered chain (``integrations/market_data``'s
:func:`ordered_price_chain` over the TA seed set): a source-side
failure (exchange down, connector unconnected, empty response,
network error) records its outcome into the price class's shared
health store (:func:`record_source_result` — the same store
``verify_price_sources`` maintains) and the next healthy exchange is
tried, so a dead named source falls over instead of raising.
Request-side errors (unknown source, unsupported interval, bad
symbol, no vault/connector configured) are facts about the REQUEST,
not the source: they still fail fast, exactly as before, and are
never recorded as source ill-health.
"""

from __future__ import annotations

import logging
import time
from typing import Any


from .math import ensure_ohlcv


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
    "SOURCES",
    "fetch_ohlcv",
    "fetch_binance",
    "fetch_coinbase",
    # ── sweep additions: keyless public feeds + cache ──
    "fetch_kraken",
    "fetch_bybit",
    "cache_clear",
]

SOURCES = ("binance", "coinbase", "kraken", "bybit")

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
                  vault=None, http=None, connector=None) -> _pd.DataFrame:
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
    df = _pd.DataFrame([
        {"open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
         "close": float(r[4]), "volume": float(r[5])}
        for r in rows
    ], index=_pd.to_datetime([r[0] for r in rows], unit="ms", utc=True))
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
                   vault=None, http=None, connector=None) -> _pd.DataFrame:
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
    df = _pd.DataFrame([
        {"open": float(c["open"]), "high": float(c["high"]),
         "low": float(c["low"]), "close": float(c["close"]),
         "volume": float(c["volume"])}
        for c in collected
    ], index=_pd.to_datetime([c["start"] for c in collected], unit="s",
                             utc=True))
    # Chunk boundaries are inclusive on both requests, so the boundary
    # candle can arrive twice — dedupe before sorting.
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    return ensure_ohlcv(df.tail(limit))


def _note_ta_result(source: str, ok: bool) -> None:
    """Note one TA fetch attempt into the price class's health store
    (AA-53) — the same learned state ``market_data``'s chains order
    by and its health loop maintains, so TA outcomes improve the
    shared ordering and price outcomes improve TA's.  Lazy import
    (this module already lazy-imports the connector registry to
    respect layering); best-effort — bookkeeping never breaks a
    fetch."""
    try:
        from importlib import import_module

        md = import_module("nomorals.integrations.market_data")
        md.record_source_result(source, ok)
    except Exception:  # noqa: BLE001 - bookkeeping is never fatal
        pass


def _ta_attempt_order(requested: str) -> list[str]:
    """The order :func:`fetch_ohlcv` attempts the TA seed exchanges:
    the price class's health-ordered chain over the seed set with
    the requested source heading the pool (AA-53).  Zero learned
    state → the pool is returned untouched (requested first, then
    :data:`SOURCES` order); with evidence, stood-down / disabled
    exchanges demote to the end — demoted, never dropped, so a
    demoted exchange is still the fallback of last resort.  Never
    raises; a hiccup returns the plain pool."""
    pool = [requested] + [s for s in SOURCES if s != requested]
    try:
        from importlib import import_module

        md = import_module("nomorals.integrations.market_data")
        ordered = md.ordered_price_chain("crypto", pool)
        return list(ordered) or pool
    except Exception:  # noqa: BLE001 - ordering is advisory
        return pool


def _fetch_one(source: str, symbol: str, interval: str, limit: int, *,
               vault=None, http=None, connector=None,
               cache_ttl: int = 300) -> _pd.DataFrame:
    """One exchange's fetch, exactly the pre-AA-53 dispatch."""
    if source == "binance":
        return fetch_binance(symbol, interval, limit, vault=vault,
                             http=http, connector=connector)
    if source == "coinbase":
        return fetch_coinbase(symbol, interval, limit, vault=vault,
                              http=http, connector=connector)
    if source == "kraken":
        return _cached("kraken", symbol, interval, limit, cache_ttl,
                       fetch_kraken, symbol, interval, limit)
    if source == "bybit":
        return _cached("bybit", symbol, interval, limit, cache_ttl,
                       fetch_bybit, symbol, interval, limit)
    raise ValueError(
        f"unknown OHLCV source {source!r}; supported: "
        f"{', '.join(SOURCES)}")


def fetch_ohlcv(source: str, symbol: str, interval: str = "1h",
                limit: int = 200, *, vault=None, http=None,
                connector=None, cache_ttl: int = 300) -> _pd.DataFrame:
    """OHLCV from any supported source, with health-ordered fallback.

    ``source`` is one of ``SOURCES`` and is attempted first unless
    the price class's learned health has demoted it. ``binance`` /
    ``coinbase`` go through the connector layer; ``kraken``/``bybit``
    use keyless public endpoints (stdlib urllib — no credentials, no
    connector). Results are cached on disk for ``cache_ttl`` seconds
    (0 disables).

    A source-side failure (unconnected connector, exchange error,
    empty response, network failure) is noted into the price class's
    health store and the next exchange in the health-ordered chain
    is tried; when every exchange fails, the REQUESTED source's
    error is raised.  A request-side ``ValueError`` (unsupported
    interval, bad symbol, no vault/connector) fails fast with no
    fallback and no health note — it is not the source's fault.  An
    injected ``connector`` serves only its own source; fallback
    connector sources without a vault are skipped as unconfigured.
    """
    src = (source or "").strip().lower()
    if src not in SOURCES:
        raise ValueError(
            f"unknown OHLCV source {source!r}; supported: "
            f"{', '.join(SOURCES)}")
    first_error: Exception | None = None
    failures: list[str] = []
    for name in _ta_attempt_order(src):
        try:
            df = _fetch_one(
                name, symbol, interval, limit, vault=vault, http=http,
                connector=connector if name == src else None,
                cache_ttl=cache_ttl)
        except ValueError:
            if name == src:
                raise  # request-side error: fail fast, as before
            continue  # fallback source can't serve this request shape
        except Exception as exc:  # noqa: BLE001 - fall over to the next
            _note_ta_result(name, False)
            if name == src:
                first_error = exc
            failures.append(f"{name}: {exc}")
            _log.info("ta feeds: %s failed for %s %s (%s); "
                      "falling over", name, symbol, interval, exc)
            continue
        _note_ta_result(name, True)
        try:
            df.attrs["source"] = name
            if name != src:
                df.attrs["requested_source"] = src
        except Exception:  # noqa: BLE001 - attrs are surface only
            pass
        return df
    if first_error is not None:
        raise first_error
    raise RuntimeError(  # pragma: no cover - order always includes src
        f"no TA feed could serve {symbol}: " + "; ".join(failures))


# ── keyless public feeds + disk cache ────────────────────────────────────

def _cache_dir() -> "os.PathLike":
    import os

    d = os.path.join(os.path.expanduser("~"), ".cache", "devon", "ta_feeds")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_key(source: str, symbol: str, interval: str, limit: int) -> str:
    import hashlib

    raw = f"{source}|{symbol}|{interval}|{limit}".encode()
    return hashlib.sha256(raw).hexdigest() + ".pkl"


def _cache_get(key: str, ttl: int):
    import os
    import pickle

    if ttl <= 0:
        return None
    path = os.path.join(_cache_dir(), key)
    try:
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > ttl:
            return None
        with open(path, "rb") as fh:
            df = pickle.load(fh)
        _log.debug("feed cache hit: %s", key)
        return ensure_ohlcv(df)
    except Exception as e:
        _log.debug("feed cache miss (%s): %s", key, e)
        return None


def _cache_set(key: str, df: _pd.DataFrame) -> None:
    import os
    import pickle

    try:
        with open(os.path.join(_cache_dir(), key), "wb") as fh:
            pickle.dump(df, fh, protocol=4)
    except Exception as e:
        _log.debug("feed cache write failed: %s", e)


def _cached(source, symbol, interval, limit, ttl, fn, *args):
    key = _cache_key(source, symbol, interval, limit)
    hit = _cache_get(key, ttl)
    if hit is not None:
        return hit
    df = fn(*args)
    _cache_set(key, df)
    return df


def cache_clear() -> int:
    """Drop all cached feed frames. Returns files removed."""
    import os

    d = _cache_dir()
    n = 0
    for f in os.listdir(d):
        try:
            os.remove(os.path.join(d, f))
            n += 1
        except OSError:
            pass
    return n


def _http_get_json(url: str, params: dict, timeout: int = 15) -> dict:
    import json
    import urllib.parse
    import urllib.request

    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": "devon-ta/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


_KRAKEN_INTERVALS = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
                     "4h": 240, "1d": 1440, "1w": 10080}

def _kraken_pair(symbol: str) -> str:
    """Normalize to Kraken's pair format.

    Fiat quotes use the X/Z convention (BTC/USD → ``XXBTZUSD``);
    stablecoin quotes use the modern altname (BTC/USDT → ``XBTUSDT``).
    """
    s = (symbol or "").strip().upper().replace("-", "").replace("/", "")
    for quote in ("USDT", "USDC", "USD", "EUR"):
        if s.endswith(quote) and len(s) > len(quote):
            base = s[:-len(quote)]
            if quote in ("USD", "EUR"):
                b = {"BTC": "XXBT", "ETH": "XETH"}.get(base, base)
                return b + "Z" + quote
            b = {"BTC": "XBT"}.get(base, base)
            return b + quote
    return s


def fetch_kraken(symbol: str, interval: str = "1h",
                 limit: int = 200) -> _pd.DataFrame:
    """Kraken public OHLC — no key needed (stdlib urllib only).

    ``symbol`` like ``"BTCUSDT"`` / ``"BTC-USD"`` is normalized to
    Kraken's pair format. Returns up to 720 candles per call.
    """
    iv = (interval or "").strip()
    if iv not in _KRAKEN_INTERVALS:
        raise ValueError(
            f"unsupported kraken interval {interval!r}; choose from "
            f"{', '.join(_KRAKEN_INTERVALS)}")
    pair = _kraken_pair(symbol)
    data = _http_get_json("https://api.kraken.com/0/public/OHLC",
                          {"pair": pair, "interval": _KRAKEN_INTERVALS[iv]})
    if data.get("error"):
        raise RuntimeError(f"kraken error for {pair}: {data['error']}")
    result = data.get("result") or {}
    rows = None
    for k, v in result.items():
        if k != "last" and isinstance(v, list):
            rows = v
            break
    if not rows:
        raise RuntimeError(f"kraken returned no candles for {pair}")
    rows = rows[-max(1, min(int(limit), 720)):]
    df = _pd.DataFrame([
        {"open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
         "close": float(r[4]), "volume": float(r[6])}
        for r in rows
    ], index=_pd.to_datetime([r[0] for r in rows], unit="s", utc=True))
    return ensure_ohlcv(df)


_BYBIT_INTERVALS = {"1m": "1", "5m": "5", "15m": "15", "30m": "30",
                    "1h": "60", "4h": "240", "1d": "D", "1w": "W"}


def fetch_bybit(symbol: str, interval: str = "1h",
                limit: int = 200) -> _pd.DataFrame:
    """Bybit public klines (v5) — no key needed (stdlib urllib only).

    ``symbol`` like ``"BTCUSDT"``; ``category`` is spot by default.
    """
    iv = (interval or "").strip()
    if iv not in _BYBIT_INTERVALS:
        raise ValueError(
            f"unsupported bybit interval {interval!r}; choose from "
            f"{', '.join(_BYBIT_INTERVALS)}")
    sym = (symbol or "").strip().upper().replace("-", "").replace("/", "")
    if not sym:
        raise ValueError("symbol is required (e.g. 'BTCUSDT')")
    data = _http_get_json(
        "https://api.bybit.com/v5/market/kline",
        {"category": "spot", "symbol": sym,
         "interval": _BYBIT_INTERVALS[iv],
         "limit": max(1, min(int(limit), 1000))})
    if str(data.get("retCode")) != "0":
        raise RuntimeError(
            f"bybit error for {sym}: {data.get('retMsg')}")
    rows = (data.get("result") or {}).get("list") or []
    if not rows:
        raise RuntimeError(f"bybit returned no klines for {sym}")
    rows = sorted(rows, key=lambda r: int(r[0]))
    df = _pd.DataFrame([
        {"open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
         "close": float(r[4]), "volume": float(r[5])}
        for r in rows
    ], index=_pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True))
    return ensure_ohlcv(df)
