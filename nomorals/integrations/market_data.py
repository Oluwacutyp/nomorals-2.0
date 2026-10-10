"""Free market-data adapters for Devon's finance brain (Prompt 07 follow-up).

Every source here is **keyless-first**: the default chain for each market
needs no signup, no API key, no package beyond the standard library and the
pandas/numpy the Sentinel engine already requires. Keyed free tiers
(Alpha Vantage, TwelveData, Finnhub) are supported as *optional upgrades*
via environment variables — never required.

Endpoint matrix (see ``docs/FREE_MARKET_DATA.md`` for the full table):

crypto      (default chain) : binance → kraken → coinbase → coingecko
stocks      (default chain) : yahoo → stooq → [alphavantage|twelvedata|finnhub if keyed]
forex       (default chain) : frankfurter → open_er_api → ecb → yahoo → stooq
commodities (default chain) : gold_api → yahoo_futures → frankfurter

Source reliability tiers (infrastructure first):
  Tier 1 — Infrastructure (exchanges, central banks): binance, kraken,
           coinbase, ecb, frankfurter
  Tier 2 — Established aggregators: yahoo, coingecko, stooq
  Tier 3 — Community utilities (verified working, keyless): gold_api,
           open_er_api

Each source has automatic health tracking — after 3 consecutive failures
a source is skipped for 5 minutes (circuit breaker). See source_health().

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
    "SOURCE_TIERS",
    "MarketDataError",
    "normalize_symbol",
    "get_ohlcv",
    "quote",
    "SentinelMarketProvider",
    "source_status",
    "source_health",
    "record_source_result",
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
        "covers": "fiat FX + XAU/XAG daily fixings (v2 API, central-bank sources)",
        "limits": "no quotas, no key",
        "key": "none",
    },
    "gold_api": {
        "covers": "real-time XAU/XAG/XPT/XPD spot (gold-api.com, keyless /price/)",
        "limits": "no key, CORS-enabled; /history/ needs key",
        "key": "none",
    },
    "yahoo_metals": {
        "covers": "metals futures OHLC: GC=F (gold), SI=F (silver), PL=F, PA=F",
        "limits": "unofficial API, no key (~2000 req/hr tolerated)",
        "key": "none",
    },
    "open_er_api": {
        "covers": "fiat FX live rates (open.er-api.com, exchangerate-api.com free)",
        "limits": "no key, daily updates",
        "key": "none",
    },
    "ecb": {
        "covers": "ECB euro reference rates direct (eurofxref-daily.xml)",
        "limits": "no key, no rate limit, updated ~16:00 CET working days",
        "key": "none",
    },
}

#: Source reliability tiers — infrastructure first, community last.
#: Tier 1: exchanges and central banks (most likely to survive long-term).
#: Tier 2: established aggregators with track records.
#: Tier 3: verified-working community utilities (keyless, but smaller ops).
SOURCE_TIERS: dict[str, int] = {
    "binance": 1, "kraken": 1, "coinbase": 1,
    "ecb": 1, "frankfurter": 1,
    "yahoo": 2, "yahoo_metals": 2, "coingecko": 2, "stooq": 2,
    "gold_api": 3, "open_er_api": 3,
    "alphavantage": 2, "twelvedata": 2, "finnhub": 2,
}

#: optional keyed upgrades — env var → source id
KEYED_SOURCES: dict[str, str] = {
    "ALPHA_VANTAGE_API_KEY": "alphavantage",   # 25 req/day free
    "TWELVEDATA_API_KEY": "twelvedata",         # 8 req/min, 800/day free
    "FINNHUB_API_KEY": "finnhub",              # 60 calls/min free
}


# ── source health tracking (circuit breaker) ──────────────────────────
# After _HEALTH_FAIL_THRESHOLD consecutive failures, a source is skipped
# for _HEALTH_COOLDOWN_S seconds. This routes around dead sources
# automatically without manual intervention.
_HEALTH_FAIL_THRESHOLD = 3
_HEALTH_COOLDOWN_S = 300.0  # 5 minutes

_source_health: dict[str, dict[str, Any]] = {}
_health_lock = __import__("threading").Lock()


def record_source_result(source: str, ok: bool) -> None:
    """Record a source attempt. Called automatically by the fetchers."""
    now = time.time()
    with _health_lock:
        h = _source_health.get(source)
        if h is None:
            h = {"fails": 0, "ok": 0, "last_fail": 0.0,
                 "last_ok": 0.0, "cooling_until": 0.0}
            _source_health[source] = h
        if ok:
            h["fails"] = 0
            h["ok"] += 1
            h["last_ok"] = now
            h["cooling_until"] = 0.0
        else:
            h["fails"] += 1
            h["last_fail"] = now
            if h["fails"] >= _HEALTH_FAIL_THRESHOLD:
                h["cooling_until"] = now + _HEALTH_COOLDOWN_S
                _log.warning("market_data: %s cooling down for %ds "
                             "(%d consecutive failures)",
                             source, _HEALTH_COOLDOWN_S, h["fails"])


def _source_healthy(source: str) -> bool:
    """True if the source is not in cooldown."""
    with _health_lock:
        h = _source_health.get(source)
        if h is None:
            return True
        return time.time() >= h.get("cooling_until", 0.0)


def source_health() -> dict[str, dict[str, Any]]:
    """Per-source health: fails, successes, cooling status, tier."""
    now = time.time()
    with _health_lock:
        out = {}
        for src in SOURCES:
            h = _source_health.get(src, {})
            cooling = now < h.get("cooling_until", 0.0)
            out[src] = {
                "tier": SOURCE_TIERS.get(src, 9),
                "consecutive_fails": h.get("fails", 0),
                "total_ok": h.get("ok", 0),
                "cooling_down": cooling,
                "cooling_secs_left": max(
                    0.0, h.get("cooling_until", 0.0) - now) if cooling else 0.0,
                "last_ok": h.get("last_ok", 0.0),
                "last_fail": h.get("last_fail", 0.0),
            }
        return out


def _healthy_chain(chain: list[str]) -> list[str]:
    """Filter a source chain to healthy sources (circuit breaker)."""
    return [s for s in chain if _source_healthy(s)]


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
    elif market == "commodities":
        # XAUUSD → XAU/USD, XAGUSD → XAG/USD, GOLD → XAU
        noslash = raw.replace("/", "")
        base, quote_c = noslash, "USD"
        for metal in ("XAU", "XAG", "XPT", "XPD"):
            if noslash.startswith(metal):
                base, quote_c = metal, noslash[len(metal):] or "USD"
                break
        _ALIAS = {"GOLD": "XAU", "SILVER": "XAG", "PLATINUM": "XPT",
                  "PALLADIUM": "XPD"}
        base = _ALIAS.get(base, base)
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
    # commodities: metal code → yahoo futures symbol + gold-api path
    out["metal"] = _METAL_YAHOO.get(base, "")
    return out


#: spot metal code → Yahoo futures symbol (GC=F gold, SI=F silver, …).
#: XAUUSD=X is delisted on Yahoo — futures are the working path.
_METAL_YAHOO = {
    "XAU": "GC=F", "GOLD": "GC=F",
    "XAG": "SI=F", "SILVER": "SI=F",
    "XPT": "PL=F", "PLATINUM": "PL=F",
    "XPD": "PA=F", "PALLADIUM": "PA=F",
    "COPPER": "HG=F", "WTI": "CL=F", "BRENT": "BZ=F",
}

#: metal code → gold-api.com path segment (keyless /price/ endpoint).
_METAL_GOLDAPI = {"XAU", "XAG", "XPT", "XPD"}


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


# ── commodities adapters (gold-api.com + yahoo futures) ───────────────
def _goldapi_quote(sym: dict[str, str]) -> dict[str, Any]:
    """Real-time metal spot via gold-api.com (keyless /price/ endpoint).

    Verified working 2026-10-10: XAU/XAG/XPT/XPD, updates every few
    seconds. Only /price/ is keyless — /history/ needs a key.
    """
    base = sym["base"]
    if base not in _METAL_GOLDAPI:
        raise MarketDataError(
            f"gold_api: no keyless spot for {base} "
            f"(covered: {sorted(_METAL_GOLDAPI)})")
    d = _json(f"https://api.gold-api.com/price/{base}")
    try:
        price = float((d or {})["price"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MarketDataError(
            f"gold_api: bad response for {base} ({str(d)[:120]})") from exc
    record_source_result("gold_api", True)
    return {"symbol": sym["raw"], "price": price,
            "change_pct_24h": None, "currency": "USD",
            "source": "gold_api"}


def _yahoo_metal_ohlc(sym: dict[str, str], timeframe: str,
                      bars: int) -> Any:
    """Metals OHLC via Yahoo futures (GC=F, SI=F, PL=F, PA=F).

    XAUUSD=X is delisted on Yahoo — futures contracts are the working
    path for historical metal bars.
    """
    fsym = sym.get("metal") or _METAL_YAHOO.get(sym["base"], "")
    if not fsym:
        raise MarketDataError(
            f"yahoo_metals: no futures symbol for {sym['base']}")
    iv = _YAHOO_TF.get(timeframe)
    if not iv:
        raise MarketDataError(
            f"yahoo_metals: unsupported timeframe {timeframe!r}")
    data = _json(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{fsym}",
        {"interval": iv, "range": _yahoo_range(timeframe, bars)})
    try:
        result = (data or {})["chart"]["result"][0]
        ts = result["timestamp"]
        q = result["indicators"]["quote"][0]
    except (KeyError, TypeError, IndexError) as exc:
        raise MarketDataError(
            f"yahoo_metals: no data for {fsym}") from exc
    rows = [[t * 1000, o, h, l, c, v or 0.0]
            for t, o, h, l, c, v in zip(ts, q.get("open") or [],
                                        q.get("high") or [],
                                        q.get("low") or [],
                                        q.get("close") or [],
                                        q.get("volume") or [])]
    rows.sort(key=lambda r: r[0])
    if not rows:
        raise MarketDataError(f"yahoo_metals: empty series for {fsym}")
    record_source_result("yahoo_metals", True)
    return _frame(rows[-bars:], ["timestamp", "open", "high", "low",
                                 "close", "volume"])


def _frankfurter_metal_ohlc(sym: dict[str, str], bars: int) -> Any:
    """Daily metal fixings via frankfurter (XAU/XAG have history)."""
    base = sym["base"]
    if base not in ("XAU", "XAG"):
        raise MarketDataError(
            f"frankfurter: no metal history for {base} (XAU/XAG only)")
    return _frankfurter_daily(
        {"base": base, "quote": "USD"}, bars)


# ── forex redundancy adapters ─────────────────────────────────────────
def _open_er_api_quote(sym: dict[str, str]) -> dict[str, Any]:
    """Fiat FX quote via open.er-api.com (keyless, exchangerate-api.com)."""
    base, quote_c = sym["base"], sym["quote"]
    if not base or not quote_c:
        raise MarketDataError("open_er_api: need a pair like EURUSD")
    d = _json(f"https://open.er-api.com/v6/latest/{base}")
    rates = (d or {}).get("rates") or {}
    try:
        rate = float(rates[quote_c])
    except (KeyError, TypeError, ValueError) as exc:
        raise MarketDataError(
            f"open_er_api: no rate for {base}/{quote_c}") from exc
    record_source_result("open_er_api", True)
    return {"symbol": sym["raw"], "price": rate, "change_pct_24h": None,
            "currency": quote_c, "source": "open_er_api"}


def _ecb_direct_quote(sym: dict[str, str]) -> dict[str, Any]:
    """ECB euro reference rates, straight from the source (no middleman).

    Parses eurofxref-daily.xml directly. Rates are EUR-based; cross
    rates computed for non-EUR pairs. Updated ~16:00 CET on working days.
    """
    import xml.etree.ElementTree as ET
    base, quote_c = sym["base"], sym["quote"]
    if not base or not quote_c:
        raise MarketDataError("ecb: need a pair like EURUSD")
    raw = _fetch("https://www.ecb.europa.eu/stats/eurofxref/"
                 "eurofxref-daily.xml")
    try:
        root = ET.fromstring(raw)
        ns = {"e": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}
        cubes = root.findall(".//e:Cube[@currency]", ns)
        rates = {"EUR": 1.0}
        for c in cubes:
            rates[c.get("currency")] = float(c.get("rate"))
    except Exception as exc:
        raise MarketDataError(f"ecb: XML parse failed: {exc}") from exc
    if base not in rates or quote_c not in rates:
        raise MarketDataError(
            f"ecb: pair {base}/{quote_c} not in reference rates")
    # ECB quotes EUR/XXX; cross: base/quote = (EUR/quote) / (EUR/base)
    rate = rates[quote_c] / rates[base]
    record_source_result("ecb", True)
    return {"symbol": sym["raw"], "price": rate, "change_pct_24h": None,
            "currency": quote_c, "source": "ecb"}


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
    "forex": ["frankfurter", "open_er_api", "ecb", "yahoo", "stooq"],
    "commodities": ["yahoo_metals", "frankfurter"],
}

_FETCHERS: dict[str, Callable[..., Any]] = {
    "binance": lambda s, m, tf, b: _binance_klines(s, tf, b),
    "kraken": lambda s, m, tf, b: _kraken_ohlc(s, tf, b),
    "coinbase": lambda s, m, tf, b: _coinbase_candles(s, tf, b),
    "coingecko": lambda s, m, tf, b: _coingecko_ohlc(s, tf, b),
    "stooq": lambda s, m, tf, b: _stooq_csv(s, tf, b),
    "yahoo": _yahoo_chart,
    "yahoo_metals": lambda s, m, tf, b: _yahoo_metal_ohlc(s, tf, b),
    "frankfurter": lambda s, m, tf, b: (
        _frankfurter_metal_ohlc(s, b) if m == "commodities"
        else _frankfurter_daily(s, b)),
    "alphavantage": _alphavantage,
    "twelvedata": _twelvedata,
    "finnhub": _finnhub,
}


#: symbols that are metals, not forex pairs — auto-routed to commodities.
_METAL_SYMBOLS = {"XAUUSD", "XAU", "GOLD", "XAGUSD", "XAG", "SILVER",
                  "XPTUSD", "XPT", "XPDUSD", "XPD", "PLATINUM", "PALLADIUM"}


def _detect_market(symbol: str, market: str) -> str:
    """Auto-route metal symbols to the commodities market."""
    if market == "forex":
        raw = (symbol or "").strip().upper().replace("-", "/").replace(
            " ", "")
        if raw.replace("/", "") in _METAL_SYMBOLS:
            return "commodities"
    return market


def get_ohlcv(symbol: str, market: str = "crypto",
              timeframe: str = "1h", bars: int = 500,
              source: str = "auto") -> Any:
    """OHLCV bars from the best free source. Keyless first, always.

    ``source="auto"`` tries the market's default chain in order (keyed
    upgrades first *only* when their env var is set), returning the first
    success. Unhealthy sources (circuit breaker tripped) are skipped
    automatically. Pass an explicit source id (``"binance"``,
    ``"stooq"``…) to pin one. Raises :exc:`MarketDataError` listing
    every failure.

    Metal symbols (XAUUSD, XAG, …) passed with ``market="forex"`` are
    auto-routed to the ``commodities`` market.
    """
    market = _detect_market(symbol, (market or "crypto").strip().lower())
    sym = normalize_symbol(symbol, market)
    if source == "auto":
        chain = _keyed_chain(market) + _DEFAULT_CHAINS.get(market,
                                                           ["binance"])
        chain = _healthy_chain(chain)
        if not chain:
            # all sources cooling — try anyway, don't hard-fail
            chain = _keyed_chain(market) + _DEFAULT_CHAINS.get(
                market, ["binance"])
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
            record_source_result(src, True)
            _log.info("market_data: %s %s %s via %s (%d bars)",
                      symbol, market, timeframe, src, len(df))
            return df
        except MarketDataError as exc:
            record_source_result(src, False)
            errors.append(f"{src}: {exc}")
        except Exception as exc:  # noqa: BLE001 - adapter bug, try next
            record_source_result(src, False)
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

    Metal symbols (XAUUSD, XAG, …) are auto-routed to the commodities
    market even when ``market="forex"`` is passed.
    """
    market = _detect_market(symbol, (market or "crypto").strip().lower())
    sym = normalize_symbol(symbol, market)
    disp = (symbol or "").strip().upper()
    if market == "crypto":
        for src in _healthy_chain(["binance", "coingecko"]):
            try:
                if src == "binance":
                    d = _json(
                        "https://api.binance.com/api/v3/ticker/24hr",
                        {"symbol": sym["binance"]})
                    bid = float(d["bidPrice"]) if d.get("bidPrice") else None
                    ask = float(d["askPrice"]) if d.get("askPrice") else None
                    hi = float(d["highPrice"]) if d.get("highPrice") else None
                    lo = float(d["lowPrice"]) if d.get("lowPrice") else None
                    spread_bps = ((ask - bid) / float(d["lastPrice"])
                                  * 10000) if bid and ask else None
                    record_source_result("binance", True)
                    return {"symbol": disp,
                            "price": float(d["lastPrice"]),
                            "change_pct_24h": float(
                                d.get("priceChangePercent") or 0),
                            "currency": sym["quote"] or "USDT",
                            "bid": bid, "ask": ask,
                            "high": hi, "low": lo,
                            "spread_bps": spread_bps,
                            "volume": float(d.get("quoteVolume") or 0)
                            or None,
                            "source": "binance"}
                coin = _CG_MAP.get(sym["base"], sym["base"].lower())
                d = _json(
                    "https://api.coingecko.com/api/v3/simple/price",
                    {"ids": coin, "vs_currencies": "usd",
                     "include_24hr_change": "true"})
                row = (d or {}).get(coin) or {}
                if "usd" not in row:
                    record_source_result("coingecko", False)
                    continue
                record_source_result("coingecko", True)
                return {"symbol": disp, "price": float(row["usd"]),
                        "change_pct_24h": row.get("usd_24h_change"),
                        "currency": "USD", "source": "coingecko"}
            except Exception:  # noqa: BLE001 - try next source
                record_source_result(src, False)
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
        # Chain: frankfurter → open.er-api.com → ECB direct.
        # Each records health; cooling sources are skipped.
        forex_chain = _healthy_chain(
            ["frankfurter", "open_er_api", "ecb"])
        errors = []
        for src in forex_chain:
            try:
                if src == "frankfurter":
                    d = _json(f"https://api.frankfurter.dev/v2/rate/"
                              f"{base.lower()}/{quote_c.lower()}")
                    rate = float((d or {})["rate"])
                    record_source_result("frankfurter", True)
                    return {"symbol": disp, "price": rate,
                            "change_pct_24h": None,
                            "currency": quote_c, "source": "frankfurter"}
                elif src == "open_er_api":
                    return _open_er_api_quote(sym)
                elif src == "ecb":
                    return _ecb_direct_quote(sym)
            except Exception as exc:  # noqa: BLE001 - try next
                record_source_result(src, False)
                errors.append(f"{src}: {exc}")
                continue
        # yahoo as last resort for forex (v8 chart)
        try:
            ysym = _yahoo_symbol(sym, market)
            d = _json(
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                f"{ysym}", {"interval": "1d", "range": "5d"})
            meta = (d or {}).get("chart", {}).get("result", [{}])[0].get(
                "meta", {})
            price = meta.get("regularMarketPrice")
            if price is not None:
                record_source_result("yahoo", True)
                return {"symbol": disp, "price": float(price),
                        "change_pct_24h": None,
                        "currency": quote_c or "USD",
                        "source": "yahoo"}
        except Exception as exc:  # noqa: BLE001
            record_source_result("yahoo", False)
            errors.append(f"yahoo: {exc}")
        raise MarketDataError(
            f"no forex quote for {base}/{quote_c}: {'; '.join(errors)}")
    if market == "commodities":
        # Chain: gold-api.com (real-time) → frankfurter (daily fixing).
        errors = []
        for src in _healthy_chain(["gold_api", "frankfurter"]):
            try:
                if src == "gold_api":
                    return _goldapi_quote(sym)
                d = _json(f"https://api.frankfurter.dev/v2/rate/"
                          f"{sym['base'].lower()}/usd")
                rate = float((d or {})["rate"])
                record_source_result("frankfurter", True)
                return {"symbol": disp, "price": rate,
                        "change_pct_24h": None, "currency": "USD",
                        "source": "frankfurter"}
            except Exception as exc:  # noqa: BLE001 - try next
                record_source_result(src, False)
                errors.append(f"{src}: {exc}")
                continue
        raise MarketDataError(
            f"no commodity quote for {disp}: {'; '.join(errors)}")
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
    """Which sources are usable right now (keyless always; keyed iff env).

    Includes reliability tiers and live health (circuit-breaker state).
    """
    keyed = {src: bool(_key(env)) for env, src in KEYED_SOURCES.items()}
    return {"keyless": list(SOURCES),
            "keyed": keyed,
            "tiers": SOURCE_TIERS,
            "default_chains": _DEFAULT_CHAINS,
            "health": source_health()}


# ── TTL cache ─────────────────────────────────────────────────────────
# Every public quote/OHLCV call is cached: 15s for quotes, 60s for
# OHLCV. Same symbol twice in a briefing = one network call.
_CACHE: dict[tuple, tuple[float, Any]] = {}
_CACHE_LOCK = __import__("threading").Lock()
_QUOTE_TTL = 15.0
_OHLCV_TTL = 60.0


def _cached(key: tuple, ttl: float, loader: Callable[[], Any]) -> Any:
    now = time.time()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = loader()
    with _CACHE_LOCK:
        _CACHE[key] = (now, value)
        if len(_CACHE) > 512:  # bounded
            oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
            del _CACHE[oldest]
    return value


def clear_cache() -> None:
    """Drop all cached market data."""
    with _CACHE_LOCK:
        _CACHE.clear()


# ── ccxt unified adapter (optional, graceful) ──────────────────────────
# ccxt is THE standard: 100+ exchanges, one method shape. Used when
# installed; everything below falls back to the keyless fetchers.
_CCXT_EXCHANGES = ("binance", "kraken", "coinbase", "bybit", "okx",
                   "bitget", "gate")
_ccxt_markets: dict[str, Any] = {}


def _ccxt() -> Any:
    """Import ccxt or raise MarketDataError with the install recipe."""
    try:
        import ccxt  # type: ignore[import]
        return ccxt
    except ImportError as exc:
        raise MarketDataError(
            "ccxt not installed (pip install ccxt) — "
            "falling back to keyless fetchers") from exc


def ccxt_available() -> bool:
    try:
        _ccxt()
        return True
    except MarketDataError:
        return False


def _ccxt_exchange(exchange: str = "binance") -> Any:
    """One cached ccxt exchange instance w/ rate limiting on."""
    ccxt = _ccxt()
    key = (exchange or "binance").lower()
    if key not in _ccxt_markets:
        cls = getattr(ccxt, key, None)
        if cls is None:
            raise MarketDataError(f"ccxt has no exchange {key!r}")
        ex = cls({"enableRateLimit": True})
        _ccxt_markets[key] = ex
    return _ccxt_markets[key]


def ccxt_ohlcv(symbol: str, timeframe: str = "1h", bars: int = 500,
               exchange: str = "binance") -> Any:
    """Unified OHLCV via ccxt (paginates with `since` past one page)."""
    ex = _ccxt_exchange(exchange)
    market = ex.market(symbol) if "/" in symbol else None
    sym = symbol if market else symbol
    rows: list[list] = []
    since = None
    page_limit = 1000
    while len(rows) < bars:
        batch = ex.fetch_ohlcv(sym, timeframe=timeframe,
                               since=since, limit=page_limit)
        if not batch:
            break
        # drop the overlapping first candle when paginating
        if rows and batch[0][0] <= rows[-1][0]:
            batch = [b for b in batch if b[0] > rows[-1][0]]
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < page_limit:
            break
    rows = rows[-bars:]
    return _frame(rows, ["timestamp", "open", "high", "low", "close",
                         "volume"])


def ccxt_quote(symbol: str, exchange: str = "binance") -> dict[str, Any]:
    """Unified ticker via ccxt: bid/ask/high/low/volume included."""
    ex = _ccxt_exchange(exchange)
    t = ex.fetch_ticker(symbol)
    return {"symbol": symbol.upper(),
            "price": t.get("last"),
            "bid": t.get("bid"), "ask": t.get("ask"),
            "high": t.get("high"), "low": t.get("low"),
            "volume": t.get("quoteVolume") or t.get("baseVolume"),
            "change_pct_24h": t.get("percentage"),
            "currency": symbol.split("/")[-1] if "/" in symbol else "",
            "source": f"ccxt:{exchange}"}


def ccxt_order_book(symbol: str, depth: int = 20,
                    exchange: str = "binance") -> dict[str, Any]:
    """Order book + spread — the honest execution-cost view."""
    ex = _ccxt_exchange(exchange)
    book = ex.fetch_order_book(symbol, limit=depth)
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    spread = (best_ask - best_bid) if best_bid and best_ask else None
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
    return {"symbol": symbol.upper(), "bids": bids, "asks": asks,
            "best_bid": best_bid, "best_ask": best_ask,
            "spread": spread,
            "spread_bps": (spread / mid * 10000) if spread and mid else None,
            "timestamp": book.get("timestamp"),
            "source": f"ccxt:{exchange}"}


def ccxt_trades(symbol: str, limit: int = 50,
                exchange: str = "binance") -> list[dict[str, Any]]:
    """Recent trade tape via ccxt."""
    ex = _ccxt_exchange(exchange)
    out = []
    for tr in ex.fetch_trades(symbol, limit=limit) or []:
        out.append({"price": tr.get("price"), "amount": tr.get("amount"),
                    "side": tr.get("side"),
                    "timestamp": tr.get("timestamp")})
    return out


# ── keyless order book / trades (binance public) ──────────────────────

def order_book(symbol: str, market: str = "crypto",
               depth: int = 20) -> dict[str, Any]:
    """Best bid/ask + spread. ccxt when available, else Binance public."""
    if market == "crypto":
        if ccxt_available():
            sym = normalize_symbol(symbol, market)
            ccxt_sym = f"{sym['base']}/{sym['quote'] or 'USDT'}"
            return ccxt_order_book(ccxt_sym, depth=depth)
        sym = normalize_symbol(symbol, market)
        d = _json("https://api.binance.com/api/v3/depth",
                  {"symbol": sym["binance"], "limit": min(depth, 100)})
        bids = [[float(p), float(q)] for p, q in d.get("bids", [])]
        asks = [[float(p), float(q)] for p, q in d.get("asks", [])]
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        spread = (best_ask - best_bid) if best_bid and best_ask else None
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        return {"symbol": symbol.upper(), "bids": bids, "asks": asks,
                "best_bid": best_bid, "best_ask": best_ask, "spread": spread,
                "spread_bps": (spread / mid * 10000)
                if spread and mid else None,
                "source": "binance"}
    raise MarketDataError(f"order_book not supported for {market!r}")


def batch_quotes(symbols: list[str],
                 market: str = "crypto") -> list[dict[str, Any]]:
    """Quotes for many symbols; each failure degrades to None-safe skip."""
    out = []
    for sym in symbols or []:
        try:
            out.append(quote(sym, market=market))
        except MarketDataError as exc:
            _log.debug("batch quote failed for %s: %s", sym, exc)
    return out


async def stream_quotes(symbols: list[str], market: str = "crypto",
                        interval: float = 5.0):
    """Async generator of quote snapshots.

    ccxt.pro ``watch_ticker`` when installed (push, no REST burn);
    otherwise REST-poll ``quote()`` every ``interval`` seconds.
    """
    try:
        import ccxt.pro as ccxtpro  # type: ignore[import]
        have_pro = True
    except ImportError:
        have_pro = False
    if have_pro and market == "crypto":
        ex = ccxtpro.binance({"enableRateLimit": True})
        try:
            while True:
                ticks = await ex.watch_tickers(symbols)
                for sym in symbols:
                    t = ticks.get(sym) or {}
                    yield {"symbol": sym.upper(), "price": t.get("last"),
                           "bid": t.get("bid"), "ask": t.get("ask"),
                           "change_pct_24h": t.get("percentage"),
                           "source": "ccxt.pro:binance",
                           "ts": time.time()}
        finally:
            await ex.close()
        return
    # REST poll fallback
    while True:
        for q in batch_quotes(symbols, market=market):
            yield dict(q, ts=time.time())
        import asyncio
        await asyncio.sleep(max(1.0, interval))


# ── indicators (stdlib, pandas-optional) ─────────────────────────────

def _closes(bars: Any) -> list[float]:
    """Extract close prices from a DataFrame, list-of-lists, or
    list-of-dicts."""
    try:
        import pandas as pd  # type: ignore[import]
        if isinstance(bars, pd.DataFrame):
            col = "close" if "close" in bars.columns else bars.columns[4]
            return [float(x) for x in bars[col].tolist()]
    except ImportError:
        pass
    out = []
    for row in bars or []:
        if isinstance(row, dict):
            out.append(float(row.get("close", 0)))
        elif isinstance(row, (list, tuple)) and len(row) > 4:
            out.append(float(row[4]))
        else:
            out.append(float(row))
    return out


def _ema(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period or period < 1:
        return out
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    out[period - 1] = ema
    for i in range(period, len(values)):
        ema = values[i] * k + ema * (1 - k)
        out[i] = ema
    return out


def indicators(bars: Any, *,
               rsi_period: int = 14, ema_fast: int = 12, ema_slow: int = 26,
               macd_signal: int = 9, bb_period: int = 20,
               atr_period: int = 14) -> dict[str, Any]:
    """RSI / EMA / MACD / Bollinger / ATR — pure stdlib.

    Accepts a DataFrame, list-of-lists [ts,o,h,l,c,v], or list-of-dicts.
    Returns latest values + full series. No numpy needed.
    """
    closes = _closes(bars)
    n = len(closes)
    if n < 2:
        raise MarketDataError("not enough bars for indicators")
    # highs/lows for ATR
    highs, lows = [], []
    try:
        import pandas as pd  # type: ignore[import]
        if isinstance(bars, pd.DataFrame):
            highs = [float(x) for x in bars["high"].tolist()]
            lows = [float(x) for x in bars["low"].tolist()]
    except ImportError:
        pass
    if not highs:
        for row in bars or []:
            if isinstance(row, (list, tuple)) and len(row) > 4:
                highs.append(float(row[2])); lows.append(float(row[3]))
            else:
                highs.append(closes[min(len(closes) - 1, len(highs))])
                lows.append(highs[-1])

    # RSI (Wilder)
    rsi: list[float | None] = [None] * n
    if n > rsi_period:
        gains, losses = [], []
        for i in range(1, n):
            d = closes[i] - closes[i - 1]
            gains.append(max(d, 0)); losses.append(max(-d, 0))
        ag = sum(gains[:rsi_period]) / rsi_period
        al = sum(losses[:rsi_period]) / rsi_period
        rsi[rsi_period] = 100 - 100 / (1 + ag / al) if al else 100.0
        for i in range(rsi_period + 1, n):
            ag = (ag * (rsi_period - 1) + gains[i - 1]) / rsi_period
            al = (al * (rsi_period - 1) + losses[i - 1]) / rsi_period
            rsi[i] = 100 - 100 / (1 + ag / al) if al else 100.0

    ef, es = _ema(closes, ema_fast), _ema(closes, ema_slow)
    macd_line = [(a - b) if a is not None and b is not None else None
                 for a, b in zip(ef, es)]
    macd_vals = [x for x in macd_line if x is not None]
    sig = _ema(macd_vals, macd_signal) if macd_vals else []
    signal_line: list[float | None] = [None] * n
    hist: list[float | None] = [None] * n
    j = 0
    for i, m in enumerate(macd_line):
        if m is not None and j < len(sig) and sig[j] is not None:
            signal_line[i] = sig[j]
            hist[i] = m - sig[j]  # type: ignore[operator]
            j += 1
        elif m is not None:
            j += 1

    # Bollinger
    bb_up: list[float | None] = [None] * n
    bb_dn: list[float | None] = [None] * n
    bb_mid: list[float | None] = [None] * n
    for i in range(bb_period - 1, n):
        w = closes[i - bb_period + 1:i + 1]
        mu = sum(w) / bb_period
        sd = (sum((x - mu) ** 2 for x in w) / bb_period) ** 0.5
        bb_mid[i] = mu; bb_up[i] = mu + 2 * sd; bb_dn[i] = mu - 2 * sd

    # ATR (Wilder)
    atr: list[float | None] = [None] * n
    if n > atr_period:
        trs = []
        for i in range(1, n):
            trs.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
        a = sum(trs[:atr_period]) / atr_period
        atr[atr_period] = a
        for i in range(atr_period + 1, n):
            a = (a * (atr_period - 1) + trs[i - 1]) / atr_period
            atr[i] = a

    def _last(s: list) -> Any:
        for x in reversed(s):
            if x is not None:
                return round(float(x), 6)
        return None

    return {
        "rsi": _last(rsi), "rsi_series": rsi,
        "ema_fast": _last(ef), "ema_slow": _last(es),
        "macd": _last(macd_line), "macd_signal": _last(signal_line),
        "macd_hist": _last(hist),
        "bb_upper": _last(bb_up), "bb_mid": _last(bb_mid),
        "bb_lower": _last(bb_dn),
        "atr": _last(atr),
        "n_bars": n,
    }


# ── god-tier quote card ──────────────────────────────────────────────

def format_quote(q: dict[str, Any]) -> str:
    """📈 BTC/USDT — $67,432.10 (+2.4% 24h) · bid/ask · H/L · via binance."""
    sym = q.get("symbol", "?")
    price = q.get("price")
    chg = q.get("change_pct_24h")
    cur = q.get("currency", "")
    if price is None:
        return f"📈 {sym} — _no quote_"
    arrow = "🟢" if (chg or 0) >= 0 else "🔴"
    chg_s = f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "n/a"
    lines = [f"{arrow} **{sym}** — {price:,.4f} {cur} ({chg_s} 24h)"]
    bid, ask = q.get("bid"), q.get("ask")
    if bid and ask:
        lines.append(f"bid {bid:,.4f} / ask {ask:,.4f}")
        spread = q.get("spread_bps")
        if spread:
            lines.append(f"spread {spread:.1f} bps")
    hi, lo = q.get("high"), q.get("low")
    if hi and lo:
        lines.append(f"24h H {hi:,.4f} / L {lo:,.4f}")
    lines.append(f"_via {q.get('source', '?')}_")
    return "\n".join(lines)


__all__ += [
    "ccxt_available", "ccxt_ohlcv", "ccxt_quote", "ccxt_order_book",
    "ccxt_trades", "order_book", "batch_quotes", "stream_quotes",
    "indicators", "format_quote", "clear_cache",
]
