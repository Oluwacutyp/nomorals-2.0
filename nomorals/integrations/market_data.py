"""Free market-data adapters for Devon's finance brain (Prompt 07 follow-up).

Every source here is **keyless-first**: the default chain for each market
needs no signup, no API key, no package beyond the standard library and the
pandas/numpy the Sentinel engine already requires. Keyed free tiers
(Alpha Vantage, TwelveData, Finnhub) are supported as *optional upgrades*
via environment variables — never required.

Endpoint matrix (see ``docs/FREE_MARKET_DATA.md`` for the full table):

crypto  (default chain) : binance → coingecko   [quote(); OHLC also tries kraken/coinbase]
stocks  (default chain) : yahoo → stooq → [alphavantage|twelvedata|finnhub if keyed]
forex   (default chain) : frankfurter → yahoo → stooq → [alphavantage|twelvedata if keyed]

All OHLC fetchers return a pandas DataFrame with a DatetimeIndex and
``open/high/low/close/volume`` float columns — the exact shape Sentinel's
``ensure_ohlcv`` expects — so they plug straight into
:func:`sentinel_bridge.load_data`.

Importing this module must stay cheap: pandas is imported lazily inside the
OHLC functions only. The ``quote()`` path is stdlib-only (urllib + json/csv
through the proxy-aware :class:`nomorals.core.http.HttpClient`).
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..core.http import HttpClient
from ..core.logging_setup import get_logger

__all__ = [
    "SOURCES",
    "KEYED_SOURCES",
    "MarketDataError",
    "normalize_symbol",
    "get_ohlcv",
    "quote",
    "SentinelMarketProvider",
    "source_status",
]

_log = get_logger(__name__)

_UA = "nomorals-marketdata/1.0"


class MarketDataError(Exception):
    """All tried sources failed for this request."""


# ── source registry ───────────────────────────────────────────────────
# key: source id → dict(coverage, rate limit, needs key). The full matrix
# lives in docs/FREE_MARKET_DATA.md; this is the machine-readable summary.
SOURCES: dict[str, dict[str, str]] = {
    "binance": {
        "covers": "crypto OHLCV, 1m–1M, up to 1000 bars/request",
        "limits": "~1200 request-weight/min, no key",
        "key": "none",
    },
    "kraken": {
        "covers": "crypto OHLC, 1m–1w, up to 720 bars",
        "limits": "public tier, counter-decay, no key",
        "key": "none",
    },
    "coinbase": {
        "covers": "crypto candles, 1m–1d, up to 300 bars/request",
        "limits": "10 req/s public, no key",
        "key": "none",
    },
    "coingecko": {
        "covers": "crypto OHLC (no volume); 30m/4h/4d granularity by range",
        "limits": "~5–15 calls/min free, no key",
        "key": "none",
    },
    "stooq": {
        "covers": "stocks daily/weekly/monthly OHLCV; forex daily; crypto daily",
        "limits": "generous, no key (bot-walled from datacenters — fallback)",
        "key": "none",
    },
    "yahoo": {
        "covers": "stocks/forex/crypto intraday+daily (v8 chart API)",
        "limits": "unofficial API, no key (~2000 req/hr tolerated)",
        "key": "none",
    },
    "frankfurter": {
        "covers": "fiat FX daily fixings (v2 API, 104 central-bank sources)",
        "limits": "no quotas, no key",
        "key": "none",
    },
}

#: optional keyed upgrades — env var → source id
KEYED_SOURCES: dict[str, str] = {
    "ALPHA_VANTAGE_API_KEY": "alphavantage",   # 25 req/day free
    "TWELVEDATA_API_KEY": "twelvedata",         # 8 req/min, 800/day free
    "FINNHUB_API_KEY": "finnhub",              # 60 calls/min free
}


# ── HTTP ──────────────────────────────────────────────────────────────
def _client() -> HttpClient:
    return HttpClient(timeout=20.0, user_agent=_UA)


def _fetch(url: str, params: dict[str, Any] | None = None) -> bytes:
    """GET bytes or raise MarketDataError. Tests patch this function."""
    try:
        resp = _client().get(url, params=params or {})
        status = int(getattr(resp, "status", 0) or 0)
        if status == 429:
            raise MarketDataError(f"rate limited (429): {url}")
        if status >= 400:
            raise MarketDataError(f"HTTP {status}: {url}")
        return getattr(resp, "body", b"") or b""
    except MarketDataError:
        raise
    except Exception as exc:  # noqa: BLE001 - network errors
        raise MarketDataError(f"fetch failed {url}: {exc}") from exc


def _json(url: str, params: dict[str, Any] | None = None) -> Any:
    return json.loads(_fetch(url, params).decode("utf-8", "ignore"))


# ── symbol normalization ──────────────────────────────────────────────
def normalize_symbol(symbol: str, market: str = "crypto") -> dict[str, str]:
    """Split a user symbol into per-source spellings.

    Returns dict with ``base``/``quote`` plus source spellings. Never
    raises on odd input — garbage in, best-effort out.
    """
    raw = (symbol or "").strip().upper().replace("-", "/")
    market = (market or "crypto").strip().lower()
    if "/" in raw:
        base, quote_c = raw.split("/", 1)
    elif market == "forex" and len(raw) == 6 and raw.isalpha():
        base, quote_c = raw[:3], raw[3:]  # EURUSD → EUR/USD
    else:
        base, quote_c = raw, ("USDT" if market == "crypto"
                              else "USD" if market == "stocks" else "")
    base = base.strip() or raw
    quote_c = quote_c.strip()
    if market == "crypto" and not quote_c:
        quote_c = "USDT"
    out = {"base": base, "quote": quote_c, "raw": raw, "market": market}
    out["binance"] = base + quote_c
    out["kraken"] = _kraken_pair(base, quote_c)
    out["coinbase"] = f"{base}-{quote_c}"
    out["stooq"] = (f"{base.lower()}.us" if market == "stocks"
                    else f"{(base + quote_c).lower()}")
    return out


_KRAKEN_BASE = {"BTC": "XXBT", "ETH": "XETH", "DOGE": "XXDG",
                "LTC": "XLTC", "XRP": "XXRP"}
_KRAKEN_QUOTE = {"USD": "ZUSD", "EUR": "ZEUR", "GBP": "ZGBP",
                 "JPY": "ZJPY", "CAD": "ZCAD"}


def _kraken_pair(base: str, quote_c: str) -> str:
    return _KRAKEN_BASE.get(base, base) + _KRAKEN_QUOTE.get(quote_c, quote_c)


# ── timeframe maps ────────────────────────────────────────────────────
_BINANCE_TF = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
               "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h",
               "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1d",
               "3d": "3d", "1w": "1w", "1M": "1M"}
_KRAKEN_TF = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
              "4h": 240, "1d": 1440, "1w": 10080}
_COINBASE_TF = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600,
                "6h": 21600, "1d": 86400}


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:
        raise MarketDataError(
            "pandas is required for OHLC bars: pip install pandas") from exc
    return pd


def _frame(rows: list[list], columns: list[str]) -> Any:
    """Build the Sentinel-shaped OHLCV DataFrame from raw rows."""
    pd = _pd()
    import numpy as np  # noqa: F401  (ensure numeric dtypes)
    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].dropna(
        subset=["close"])


# ── keyless crypto adapters ───────────────────────────────────────────
def _binance_klines(sym: dict[str, str], timeframe: str,
                    bars: int) -> Any:
    iv = _BINANCE_TF.get(timeframe)
    if not iv:
        raise MarketDataError(f"binance: unsupported timeframe {timeframe!r}")
    rows: list[list] = []
    end_ms: int | None = None
    remaining = min(max(int(bars), 1), 3000)
    while remaining > 0:
        n = min(remaining, 1000)
        params: dict[str, Any] = {"symbol": sym["binance"], "interval": iv,
                                  "limit": n}
        if end_ms is not None:
            params["endTime"] = end_ms
        data = _json("https://api.binance.com/api/v3/klines", params)
        if not isinstance(data, list) or not data:
            break
        for k in data:
            rows.append([k[0], k[1], k[2], k[3], k[4], k[5]])
        end_ms = int(data[0][0]) - 1
        remaining -= len(data)
        if len(data) < n:
            break
    if not rows:
        raise MarketDataError(
            f"binance: no klines for {sym['binance']} {iv}")
    rows.sort(key=lambda r: r[0])
    return _frame(rows, ["timestamp", "open", "high", "low", "close",
                         "volume"])


def _kraken_ohlc(sym: dict[str, str], timeframe: str, bars: int) -> Any:
    minutes = _KRAKEN_TF.get(timeframe)
    if not minutes:
        raise MarketDataError(f"kraken: unsupported timeframe {timeframe!r}")
    data = _json("https://api.kraken.com/0/public/OHLC",
                 {"pair": sym["kraken"], "interval": minutes})
    if isinstance(data, dict) and data.get("error"):
        raise MarketDataError(f"kraken: {data['error']}")
    result = (data or {}).get("result") or {}
    key = next((k for k in result if k != "last"), "")
    rows = result.get(key) or []
    if not rows:
        raise MarketDataError(f"kraken: no OHLC for {sym['kraken']}")
    # kraken row: [time, open, high, low, close, vwap, volume, count]
    cut = [[r[0] * 1000, r[1], r[2], r[3], r[4], r[6]] for r in rows]
    return _frame(cut[-bars:], ["timestamp", "open", "high", "low",
                                "close", "volume"])


def _coinbase_candles(sym: dict[str, str], timeframe: str,
                      bars: int) -> Any:
    gran = _COINBASE_TF.get(timeframe)
    if not gran:
        raise MarketDataError(f"coinbase: unsupported timeframe {timeframe!r}")
    # coinbase row: [time, low, high, open, close, volume] — note the order
    data = _json(
        f"https://api.exchange.coinbase.com/products/"
        f"{sym['coinbase']}/candles",
        {"granularity": gran})
    if not isinstance(data, list) or not data:
        raise MarketDataError(f"coinbase: no candles for {sym['coinbase']}")
    rows = [[c[0] * 1000, c[3], c[2], c[1], c[4], c[5]] for c in data]
    rows.sort(key=lambda r: r[0])
    return _frame(rows[-bars:], ["timestamp", "open", "high", "low",
                                 "close", "volume"])


_CG_MAP = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
           "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
           "DOGE": "dogecoin", "TON": "the-open-network", "TRX": "tron",
           "USDT": "tether", "USDC": "usd-coin", "LINK": "chainlink",
           "AVAX": "avalanche-2", "DOT": "polkadot", "MATIC": "matic-network"}


def _coingecko_ohlc(sym: dict[str, str], timeframe: str,
                    bars: int) -> Any:
    coin = _CG_MAP.get(sym["base"], sym["base"].lower())
    # granularity is set by `days`: 1→30m, 7–30→4h, 31+→4d
    days = max(2, min(365, int(bars * {"1h": 1 / 24, "4h": 1 / 6,
                                       "1d": 1}.get(timeframe, 1))))
    data = _json(f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc",
                 {"vs_currency": "usd", "days": days})
    if not isinstance(data, list) or not data:
        raise MarketDataError(f"coingecko: no OHLC for {coin}")
    rows = [[r[0], r[1], r[2], r[3], r[4], 0.0] for r in data]
    return _frame(rows[-bars:], ["timestamp", "open", "high", "low",
                                 "close", "volume"])


# ── keyless stocks / forex adapters ───────────────────────────────────
_YAHOO_TF = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
             "1h": "60m", "2h": "60m", "4h": "60m", "1d": "1d",
             "1w": "1wk", "1M": "1mo"}


def _yahoo_symbol(sym: dict[str, str], market: str) -> str:
    if market == "forex":
        return f"{sym['base']}{sym['quote']}=X"  # EURUSD → EURUSD=X
    if market == "crypto":
        q = "USD" if sym["quote"] in ("USDT", "USDC", "DAI") else sym["quote"]
        return f"{sym['base']}-{q}"              # BTC-USDT → BTC-USD
    return sym["base"]


def _yahoo_range(timeframe: str, bars: int) -> str:
    minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
               "2h": 120, "4h": 240, "1d": 1440, "1w": 10080,
               "1M": 43200}.get(timeframe, 1440)
    need_min = minutes * max(int(bars), 1)
    for span_min, rng in ((2 * 1440, "2d"), (7 * 1440, "5d"),
                          (45 * 1440, "1mo"), (120 * 1440, "3mo"),
                          (300 * 1440, "6mo"), (540 * 1440, "1y"),
                          (1100 * 1440, "2y"), (2200 * 1440, "5y")):
        if need_min <= span_min:
            return rng
    return "max"


def _yahoo_chart(sym: dict[str, str], market: str, timeframe: str,
                 bars: int) -> Any:
    iv = _YAHOO_TF.get(timeframe)
    if not iv:
        raise MarketDataError(f"yahoo: unsupported timeframe {timeframe!r}")
    ysym = _yahoo_symbol(sym, market)
    data = _json(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}",
        {"interval": iv, "range": _yahoo_range(timeframe, bars)})
    try:
        result = (data or {})["chart"]["result"][0]
        ts = result["timestamp"]
        q = result["indicators"]["quote"][0]
    except (KeyError, TypeError, IndexError) as exc:
        raise MarketDataError(
            f"yahoo: no data for {ysym} ({str(data)[:120]})") from exc
    rows = [[t * 1000, o, h, l, c, v or 0.0]
            for t, o, h, l, c, v in zip(ts, q.get("open") or [],
                                        q.get("high") or [],
                                        q.get("low") or [],
                                        q.get("close") or [],
                                        q.get("volume") or [])]
    rows.sort(key=lambda r: r[0])
    if not rows:
        raise MarketDataError(f"yahoo: empty series for {ysym}")
    return _frame(rows[-bars:], ["timestamp", "open", "high", "low",
                                 "close", "volume"])


def _stooq_csv(sym: dict[str, str], timeframe: str, bars: int) -> Any:
    pd = _pd()
    iv = {"1d": "d", "1w": "w", "1M": "m"}.get(timeframe, "d")
    if timeframe not in ("1d", "1w", "1M") and timeframe.endswith("m"):
        iv = "5"  # intraday 5-minute bars (short history)
    raw = _fetch("https://stooq.com/q/d/l/",
                 {"s": sym["stooq"], "i": iv}).decode("utf-8", "ignore")
    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for r in reader:
        try:
            ts = pd.Timestamp(r.get("Date") or r.get("date"))
            rows.append([int(ts.value // 10**6),
                         r.get("Open"), r.get("High"), r.get("Low"),
                         r.get("Close"), r.get("Volume") or 0])
        except (ValueError, TypeError, KeyError):
            continue
    if not rows:
        raise MarketDataError(f"stooq: no data for {sym['stooq']}")
    return _frame(rows[-bars:], ["timestamp", "open", "high", "low",
                                 "close", "volume"])


def _frankfurter_daily(sym: dict[str, str], bars: int) -> Any:
    pd = _pd()
    base, quote_c = sym["base"], sym["quote"]
    if not base or not quote_c:
        raise MarketDataError("frankfurter: need a pair like EURUSD")
    from datetime import date, timedelta
    days = max(int(bars) + 40, 60)
    start = (date.today() - timedelta(days=days)).isoformat()
    # v2 time series → [{"date":..., "base":..., "quote":..., "rate":...}]
    data = _json("https://api.frankfurter.dev/v2/rates",
                 {"from": start, "base": base.lower(),
                  "quotes": quote_c.lower()})
    if not isinstance(data, list):
        raise MarketDataError(
            f"frankfurter: unexpected response ({str(data)[:120]})")
    rows = []
    for row in data:
        try:
            rate = float(row["rate"])
            ts = pd.Timestamp(row["date"])
        except (KeyError, TypeError, ValueError):
            continue
        # one fixing/day: O=H=L=C=fixing, no volume
        rows.append([int(ts.value // 10**6), rate, rate, rate, rate, 0.0])
    if not rows:
        raise MarketDataError(
            f"frankfurter: no rates for {base}/{quote_c}")
    return _frame(rows[-bars:], ["timestamp", "open", "high", "low",
                                 "close", "volume"])


# ── optional keyed upgrades (env vars, never required) ────────────────
def _key(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _alphavantage(sym: dict[str, str], market: str, timeframe: str,
                  bars: int) -> Any:
    key = _key("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise MarketDataError(
            "alphavantage: set ALPHA_VANTAGE_API_KEY to use this source")
    if market == "crypto":
        fn, av_sym = "DIGITAL_CURRENCY_DAILY", sym["base"]
        params = {"function": fn, "symbol": av_sym, "market": sym["quote"],
                  "apikey": key}
        tkey = "Time Series (Digital Currency Daily)"
        o, h, l, c, v = ("1a. open (USD)", "2a. high (USD)",
                         "3a. low (USD)", "4a. close (USD)",
                         "5. volume")
    elif market == "forex":
        params = {"function": "FX_DAILY",
                  "from_symbol": sym["base"], "to_symbol": sym["quote"],
                  "apikey": key}
        tkey = "Time Series FX (Daily)"
        o, h, l, c, v = "1. open", "2. high", "3. low", "4. close", None
    else:
        params = {"function": "TIME_SERIES_DAILY", "symbol": sym["base"],
                  "apikey": key, "outputsize": "full"}
        tkey = "Time Series (Daily)"
        o, h, l, c, v = "1. open", "2. high", "3. low", "4. close", "5. volume"
    pd = _pd()
    data = _json("https://www.alphavantage.co/query", params)
    series = (data or {}).get(tkey) or {}
    rows = []
    for day in sorted(series):
        r = series[day]
        ts = pd.Timestamp(day)
        rows.append([int(ts.value // 10**6), r.get(o), r.get(h), r.get(l),
                     r.get(c), r.get(v) if v else 0])
    if not rows:
        raise MarketDataError(
            f"alphavantage: no data ({str(data)[:120]})")
    return _frame(rows[-bars:], ["timestamp", "open", "high", "low",
                                 "close", "volume"])


def _twelvedata(sym: dict[str, str], market: str, timeframe: str,
                bars: int) -> Any:
    key = _key("TWELVEDATA_API_KEY")
    if not key:
        raise MarketDataError(
            "twelvedata: set TWELVEDATA_API_KEY to use this source")
    td_sym = (f"{sym['base']}/{sym['quote']}" if market == "crypto"
              else sym["base"])
    iv = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
          "1h": "1h", "4h": "4h", "1d": "1day", "1w": "1week"}.get(timeframe,
                                                                  "1day")
    pd = _pd()
    data = _json("https://api.twelvedata.com/time_series",
                 {"symbol": td_sym, "interval": iv,
                  "outputsize": min(max(int(bars), 1), 5000),
                  "apikey": key})
    vals = (data or {}).get("values") or []
    rows = []
    for r in vals:
        try:
            ts = pd.Timestamp(r["datetime"])
            rows.append([int(ts.value // 10**6), r["open"], r["high"],
                         r["low"], r["close"],
                         r.get("volume") or 0])
        except (KeyError, TypeError, ValueError):
            continue
    rows.sort(key=lambda r: r[0])
    if not rows:
        raise MarketDataError(f"twelvedata: no data ({str(data)[:120]})")
    return _frame(rows[-bars:], ["timestamp", "open", "high", "low",
                                 "close", "volume"])


def _finnhub(sym: dict[str, str], market: str, timeframe: str,
             bars: int) -> Any:
    key = _key("FINNHUB_API_KEY")
    if not key:
        raise MarketDataError(
            "finnhub: set FINNHUB_API_KEY to use this source")
    res = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60",
           "1d": "D", "1w": "W", "1M": "M"}.get(timeframe, "D")
    fh_sym = (f"BINANCE:{sym['binance']}" if market == "crypto"
              else sym["base"])
    now = int(time.time())
    span = {"1": 60, "5": 300, "15": 900, "30": 1800, "60": 3600,
            "D": 86400, "W": 604800, "M": 2592000}[res]
    data = _json("https://finnhub.io/api/v1/candle",
                 {"symbol": fh_sym, "resolution": res,
                  "from": now - span * int(bars), "to": now,
                  "token": key})
    if (data or {}).get("s") != "ok":
        raise MarketDataError(f"finnhub: {str(data)[:120]}")
    rows = [[t * 1000, o, h, l, c, v]
            for t, o, h, l, c, v in zip(data["t"], data["o"], data["h"],
                                        data["l"], data["c"], data["v"])]
    return _frame(rows[-bars:], ["timestamp", "open", "high", "low",
                                 "close", "volume"])


# ── routing ───────────────────────────────────────────────────────────
def _keyed_chain(market: str) -> list[str]:
    """Keyed sources whose env var is present, best-first per market."""
    out = []
    if _key("TWELVEDATA_API_KEY"):
        out.append("twelvedata")
    if _key("ALPHA_VANTAGE_API_KEY") and market in ("stocks", "forex",
                                                    "crypto"):
        out.append("alphavantage")
    if _key("FINNHUB_API_KEY"):
        out.append("finnhub")
    return out


_DEFAULT_CHAINS: dict[str, list[str]] = {
    "crypto": ["binance", "kraken", "coinbase", "coingecko"],
    "stocks": ["yahoo", "stooq"],
    "forex": ["frankfurter", "yahoo", "stooq"],
}

_FETCHERS: dict[str, Callable[..., Any]] = {
    "binance": lambda s, m, tf, b: _binance_klines(s, tf, b),
    "kraken": lambda s, m, tf, b: _kraken_ohlc(s, tf, b),
    "coinbase": lambda s, m, tf, b: _coinbase_candles(s, tf, b),
    "coingecko": lambda s, m, tf, b: _coingecko_ohlc(s, tf, b),
    "stooq": lambda s, m, tf, b: _stooq_csv(s, tf, b),
    "yahoo": _yahoo_chart,
    "frankfurter": lambda s, m, tf, b: _frankfurter_daily(s, b),
    "alphavantage": _alphavantage,
    "twelvedata": _twelvedata,
    "finnhub": _finnhub,
}


def get_ohlcv(symbol: str, market: str = "crypto",
              timeframe: str = "1h", bars: int = 500,
              source: str = "auto") -> Any:
    """OHLCV bars from the best free source. Keyless first, always.

    ``source="auto"`` tries the market's default chain in order (keyed
    upgrades first *only* when their env var is set), returning the first
    success. Pass an explicit source id (``"binance"``, ``"stooq"``…)
    to pin one. Raises :exc:`MarketDataError` listing every failure.
    """
    market = (market or "crypto").strip().lower()
    sym = normalize_symbol(symbol, market)
    if source == "auto":
        chain = _keyed_chain(market) + _DEFAULT_CHAINS.get(market,
                                                           ["binance"])
    else:
        chain = [source.strip().lower()]
    errors: list[str] = []
    for src in chain:
        fn = _FETCHERS.get(src)
        if fn is None:
            errors.append(f"{src}: unknown source")
            continue
        try:
            df = fn(sym, market, timeframe, bars)
            _log.info("market_data: %s %s %s via %s (%d bars)",
                      symbol, market, timeframe, src, len(df))
            return df
        except MarketDataError as exc:
            errors.append(f"{src}: {exc}")
        except Exception as exc:  # noqa: BLE001 - adapter bug, try next
            errors.append(f"{src}: unexpected {exc}")
            _log.warning("market_data adapter %s failed: %s", src, exc)
    raise MarketDataError(
        f"all sources failed for {symbol} [{market}]: "
        + "; ".join(errors))


# ── quotes (stdlib-only, no pandas) ───────────────────────────────────
def quote(symbol: str, market: str = "crypto") -> dict[str, Any]:
    """Latest price + 24h change for one symbol. Keyless, no pandas.

    Returns ``{"symbol","price","change_pct_24h","currency","source"}``.
    ``change_pct_24h`` may be None when the source has no change data.
    """
    market = (market or "crypto").strip().lower()
    sym = normalize_symbol(symbol, market)
    disp = (symbol or "").strip().upper()
    if market == "crypto":
        for src in ("binance", "coingecko"):
            try:
                if src == "binance":
                    d = _json(
                        "https://api.binance.com/api/v3/ticker/24hr",
                        {"symbol": sym["binance"]})
                    return {"symbol": disp,
                            "price": float(d["lastPrice"]),
                            "change_pct_24h": float(
                                d.get("priceChangePercent") or 0),
                            "currency": sym["quote"] or "USDT",
                            "source": "binance"}
                coin = _CG_MAP.get(sym["base"], sym["base"].lower())
                d = _json(
                    "https://api.coingecko.com/api/v3/simple/price",
                    {"ids": coin, "vs_currencies": "usd",
                     "include_24hr_change": "true"})
                row = (d or {}).get(coin) or {}
                if "usd" not in row:
                    continue
                return {"symbol": disp, "price": float(row["usd"]),
                        "change_pct_24h": row.get("usd_24h_change"),
                        "currency": "USD", "source": "coingecko"}
            except Exception:  # noqa: BLE001 - try next source
                continue
        raise MarketDataError(f"no crypto quote for {disp}")
    if market == "stocks":
        ysym = _yahoo_symbol(sym, market)
        try:
            d = _json(
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                f"{ysym}", {"interval": "1d", "range": "5d"})
            meta = (d or {}).get("chart", {}).get("result", [{}])[0].get(
                "meta", {})
            price = meta.get("regularMarketPrice")
            if price is not None:
                return {"symbol": disp, "price": float(price),
                        "change_pct_24h": None,
                        "currency": str(meta.get("currency") or "USD"),
                        "source": "yahoo"}
        except Exception:  # noqa: BLE001 - fall through to stooq
            pass
        raw = _fetch("https://stooq.com/q/l/",
                     {"s": sym["stooq"], "f": "sd2t2ohlcv",
                      "h": "", "e": "csv"}).decode("utf-8", "ignore")
        rows = list(csv.DictReader(io.StringIO(raw)))
        if rows:
            r = rows[0]
            try:
                return {"symbol": disp, "price": float(r["Close"]),
                        "change_pct_24h": None, "currency": "USD",
                        "source": "stooq"}
            except (KeyError, TypeError, ValueError):  # noqa: E103 - malformed stooq row; MarketDataError raised below
                pass
        raise MarketDataError(f"no stock quote for {disp}")
    if market == "forex":
        base, quote_c = sym["base"], sym["quote"]
        d = _json(f"https://api.frankfurter.dev/v2/rate/"
                  f"{base.lower()}/{quote_c.lower()}")
        try:
            rate = float((d or {})["rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError(
                f"no forex quote for {base}/{quote_c}") from exc
        return {"symbol": disp, "price": rate, "change_pct_24h": None,
                "currency": quote_c, "source": "frankfurter"}
    raise MarketDataError(f"unknown market {market!r}")


# ── MarketDataProvider (the briefing protocol) ─────────────────────────
class SentinelMarketProvider:
    """The richer Sentinel-backed provider the morning briefing's
    ``MarketDataProvider`` protocol anticipated.

    Crypto quotes via Binance 24h ticker (with real 24h change),
    stocks via Stooq, fiat FX via Frankfurter — all keyless. Implements
    the protocol structurally: ``quote(symbol, market)`` +
    ``overnight_movers(symbols, market)``.
    """

    def quote(self, symbol: str,
              market: str = "crypto") -> dict[str, Any] | None:
        try:
            return quote(symbol, market=market)
        except MarketDataError as exc:
            _log.debug("provider quote failed for %s: %s", symbol, exc)
            return None

    def overnight_movers(self, symbols: list[str],
                         market: str = "crypto") -> list[dict[str, Any]]:
        quotes = []
        for sym in symbols or []:
            q = self.quote(sym, market=market)
            if q:
                quotes.append(q)

        def _abs(q: dict[str, Any]) -> float:
            chg = q.get("change_pct_24h")
            return abs(chg) if isinstance(chg, (int, float)) else 0.0

        quotes.sort(key=_abs, reverse=True)
        return quotes


def source_status() -> dict[str, Any]:
    """Which sources are usable right now (keyless always; keyed iff env)."""
    keyed = {src: bool(_key(env)) for env, src in KEYED_SOURCES.items()}
    return {"keyless": list(SOURCES),
            "keyed": keyed,
            "default_chains": _DEFAULT_CHAINS}
