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

Pass #91 (owner's open-sources law — prices were named explicitly):
the price class is now outcome-aware, with the #86–#90 vocabulary and
discipline, in mirrored machinery HERE rather than in the media
source registry.  The registry was judged honestly and does not fit:
this module is layering-pinned L2 (imports only core/storage peers)
while ``media`` is L4; a price source's identity is its provider id,
not a (nation, language, domain) key — ``yahoo`` and ``yahoo_metals``
share one host with different coverage, and nation/language ordering
cannot express per-market chains; and this module already owns the
routing health store (the circuit breaker below), so registry
membership would have created a second, parallel stats store.
Instead the existing store learned the full vocabulary — fresh /
stale / served (no provider timestamp; freshness never fabricated) /
no-result (weak evidence) / failure with error class / rate-limited
(429: recorded once, never retried around) / bot-blocked — plus
stand-down/disable thresholds, outcome-aware chain ordering (day-one
pin: zero learned state → the hardcoded chains byte-for-byte),
KV persistence (``finance.price_sources.v1``), a real-fetch health
loop (:func:`verify_price_sources` — keyless providers probed with
one bounded liquid-symbol fetch; keyed providers never probed, they
are judged by real-fetch outcomes), and the extended
:func:`source_status` surface.  Discovery verdict: no genuine
discovery channel for new price APIs exists in code, so the class
ships with NO candidate intake — named absence, not invention.

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
    # Pass #91: the open price-source class
    "PRICE_PERSIST_KEY",
    "record_quote_outcome",
    "price_source_stats",
    "price_source_health",
    "price_source_status",
    "price_score",
    "ordered_price_chain",
    "verify_price_sources",
    "bind_price_source_store",
    "clear_price_source_state",
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


# ── source health tracking (circuit breaker) + learned state (#91) ────
# After _HEALTH_FAIL_THRESHOLD consecutive failures, a source is skipped
# for _HEALTH_COOLDOWN_S seconds. This routes around dead sources
# automatically without manual intervention.
_HEALTH_FAIL_THRESHOLD = 3
_HEALTH_COOLDOWN_S = 300.0  # 5 minutes

# Pass #91: the learned-state thresholds, mirroring the media source
# registry's vocabulary (``media/source_registry.py`` STAND_DOWN_AFTER
# / DISABLE_AFTER): STAND_DOWN_AFTER consecutive non-fresh outcomes
# stand a source down (demoted to last in its chains, still the
# fallback of last resort); DISABLE_AFTER disable it.  A fresh serve
# resets the streak and re-activates.
STAND_DOWN_AFTER = 3
DISABLE_AFTER = 10
#: Health states that remove a source from the ACTIVE ordering.
_INACTIVE_PRICE_HEALTH = frozenset({"stood-down", "disabled", "dead"})

#: KV key the price class's learned state persists under (Pass #91).
#: Version 1 is deliberate: pre-#91 state was memory-only, so there
#: is nothing to migrate — a bound store with no row is zero learned
#: state, and wiping the row returns exactly the seed behaviour.
PRICE_PERSIST_KEY = "finance.price_sources.v1"
_PRICE_STATE_VERSION = 1

_source_health: dict[str, dict[str, Any]] = {}
_health_lock = __import__("threading").Lock()
#: The bound KV store learned price state persists through (the same
#: KVStore-over-context-db discipline the media registry uses);
#: None = memory-only, exactly the pre-#91 behaviour.
_price_store: Any = None


def _blank_health() -> dict[str, Any]:
    return {"fails": 0, "ok": 0, "last_fail": 0.0, "last_ok": 0.0,
            "cooling_until": 0.0,
            # Pass #91 learned-state fields
            "stale": 0, "no_result": 0, "blocked": 0, "rate_limited": 0,
            "streak": 0, "status": "unknown", "last_error_class": "",
            "last_seen": 0.0, "last_quote_ts": None, "last_age_s": None}


def _health_entry(source: str) -> dict[str, Any]:
    """The mutable health entry for *source* (created blank).
    Caller must hold ``_health_lock``."""
    h = _source_health.get(source)
    if h is None:
        h = _blank_health()
        _source_health[source] = h
    else:
        for k, v in _blank_health().items():
            h.setdefault(k, v)
    return h


def _apply_breaker(h: dict[str, Any], source: str, now: float) -> None:
    """The pre-#91 circuit breaker, unchanged: _HEALTH_FAIL_THRESHOLD
    consecutive transport failures cool a source for 5 minutes."""
    if h["fails"] >= _HEALTH_FAIL_THRESHOLD:
        h["cooling_until"] = now + _HEALTH_COOLDOWN_S
        _log.warning("market_data: %s cooling down for %ds "
                     "(%d consecutive failures)",
                     source, _HEALTH_COOLDOWN_S, h["fails"])


def _apply_streak_status(h: dict[str, Any], kind: str) -> None:
    """Health-status transitions for one non-fresh outcome, mirroring
    the registry's: DISABLE_AFTER consecutive → disabled,
    STAND_DOWN_AFTER → stood-down, otherwise the kind's own status
    (``stale`` / ``rate-limited`` / ``bot-blocked``); a plain failure
    only marks an untouched source ``unverifiable``."""
    streak = int(h.get("streak") or 0)
    if streak >= DISABLE_AFTER:
        h["status"] = "disabled"
    elif streak >= STAND_DOWN_AFTER:
        h["status"] = "stood-down"
    elif kind == "stale":
        h["status"] = "stale"
    elif kind == "rate_limited":
        h["status"] = "rate-limited"
    elif kind == "blocked":
        h["status"] = "bot-blocked"
    elif h.get("status") in (None, "", "unknown"):
        h["status"] = "unverifiable"


def record_quote_outcome(source: str, kind: str, *,
                         error_class: str = "",
                         quote_ts: float | None = None,
                         age_s: float | None = None,
                         now: float = 0.0) -> None:
    """Record one real price-fetch attempt's outcome (Pass #91).

    ``kind`` is the open-sources vocabulary, price edition:

    * ``fresh`` — a quote/bars landed with a provider timestamp
      inside the provider's own cadence;
    * ``stale`` — a quote/bars landed but its timestamp is older
      than that cadence: its OWN evidence class, neither full
      success nor transport failure (acting on stale prices is
      how money gets lost);
    * ``served`` — a quote landed with NO provider timestamp, so
      no freshness verdict exists (never fabricated);
    * ``no_result`` — the provider answered but has no data for
      this symbol: weak evidence (#88 precedent) — it sinks the
      score's denominator but NEVER feeds the streak or the
      breaker, so a crypto-only provider is never punished for
      missing an FX symbol it was never chained for;
    * ``failure`` — transport/HTTP failure, with ``error_class``;
    * ``rate_limited`` — HTTP 429: its own class (#90), health
      ``rate-limited`` never ``dead``; recorded once — callers
      never retry or rotate around a 429 within one fetch;
    * ``blocked`` — a 403/451 bot/geo wall: health ``bot-blocked``,
      never ``dead`` (a wall is evidence about the wall).

    Auth-config missing is NOT recorded here at all (#90 precedent):
    a keyed source dark without its key is configuration, not
    evidence.  Never raises."""
    try:
        now = now or time.time()
        with _health_lock:
            h = _health_entry(source)
            h["last_seen"] = now
            if kind in ("fresh", "served"):
                h["ok"] += 1
                h["fails"] = 0
                h["streak"] = 0
                h["last_ok"] = now
                h["cooling_until"] = 0.0
                h["status"] = "verified-live"
                h["last_error_class"] = ""
                if quote_ts is not None:
                    h["last_quote_ts"] = quote_ts
                    h["last_age_s"] = age_s
            elif kind == "no_result":
                h["no_result"] += 1
            elif kind == "stale":
                h["stale"] += 1
                h["streak"] = int(h.get("streak") or 0) + 1
                h["last_fail"] = now
                h["last_error_class"] = "stale_quote"
                if quote_ts is not None:
                    h["last_quote_ts"] = quote_ts
                    h["last_age_s"] = age_s
                _apply_streak_status(h, kind)
            elif kind in ("failure", "rate_limited", "blocked"):
                if kind == "rate_limited":
                    h["rate_limited"] += 1
                elif kind == "blocked":
                    h["blocked"] += 1
                h["fails"] += 1
                h["streak"] = int(h.get("streak") or 0) + 1
                h["last_fail"] = now
                h["last_error_class"] = str(
                    error_class or kind)
                _apply_breaker(h, source, now)
                _apply_streak_status(h, kind)
            else:
                return
        _persist_price_state()
    except Exception:  # noqa: BLE001 - bookkeeping never breaks a fetch
        pass


def record_source_result(source: str, ok: bool) -> None:
    """Record a source attempt. Called automatically by the fetchers.

    The pre-#91 boolean seam, kept for its existing callers: a plain
    success is ``served`` (no freshness claim — the caller supplied
    no timestamp) and a plain failure is ``failure``; both feed the
    learned state above as well as the circuit breaker."""
    record_quote_outcome(source, "served" if ok else "failure",
                         error_class="" if ok else "fetch_failed")


def price_source_health(source: str) -> str:
    """Current learned health for one source ("unknown" until an
    outcome or the health loop says otherwise)."""
    with _health_lock:
        h = _source_health.get(source)
        return str((h or {}).get("status") or "unknown")


def price_source_stats(source: str) -> dict[str, Any]:
    """Recorded outcome statistics for one source (zeros when it has
    never served an attempt).  Never raises."""
    with _health_lock:
        h = _source_health.get(source) or {}
        ok = int(h.get("ok") or 0)
        fail = int(h.get("fails") or 0)
        stale = int(h.get("stale") or 0)
        no_result = int(h.get("no_result") or 0)
        total = ok + fail + stale + no_result
        return {
            "ok": ok, "fail": fail, "stale": stale,
            "blocked": int(h.get("blocked") or 0),
            "rate_limited": int(h.get("rate_limited") or 0),
            "no_result": no_result,
            "fresh_rate": round(ok / total, 3) if total else 0.5,
            "health": str(h.get("status") or "unknown"),
            "streak": int(h.get("streak") or 0),
            "last_ok": float(h.get("last_ok") or 0.0),
            "last_seen": float(h.get("last_seen") or 0.0),
            "last_error_class": str(h.get("last_error_class") or ""),
            "last_quote_ts": h.get("last_quote_ts"),
            "last_age_s": h.get("last_age_s"),
        }


#: Reliability-tier base weights for the learned price score.
_PRICE_TIER_WEIGHT = {1: 0.90, 2: 0.65, 3: 0.45}


def price_score(source: str) -> float:
    """Learned score 0..1 for one source: tier prior + observed
    fresh-serve rate.  Zero evidence → the neutral 0.5 rate, so the
    score only moves a chain once real outcomes exist."""
    st = price_source_stats(source)
    base = _PRICE_TIER_WEIGHT.get(SOURCE_TIERS.get(source, 9), 0.45)
    return round(max(0.0, min(1.0,
                              0.35 * base + 0.65 * st["fresh_rate"])), 3)


def ordered_price_chain(market: str,
                        chain: list[str] | None = None) -> list[str]:
    """Outcome-aware order for one market's source chain (Pass #91).

    * Zero learned state anywhere in the chain → the chain is
      returned untouched: day-one behaviour is the hardcoded
      ``_DEFAULT_CHAINS`` order (plus keyed upgrades first when
      their env var is set), byte-for-byte;
    * otherwise active sources sort by learned score (stable — ties
      keep chain order) and stood-down / disabled sources demote to
      the end, in chain order: demoted, never deleted — a demoted
      source is still the fallback of last resort when it is the
      only source for a symbol, and its demotion is visible in
      :func:`price_source_status` and on the served quote itself.
    Never raises; a hiccup returns the chain as-is."""
    pool = list(chain) if chain is not None else list(
        _DEFAULT_CHAINS.get(market, ["binance"]))
    try:
        evidence = False
        for src in pool:
            st = price_source_stats(src)
            if st["ok"] or st["fail"] or st["stale"] \
                    or st["no_result"] or st["blocked"] \
                    or st["rate_limited"] \
                    or st["health"] != "unknown":
                evidence = True
                break
        if not evidence:
            return pool
        active = [s for s in pool
                  if price_source_health(s) not in _INACTIVE_PRICE_HEALTH]
        demoted = [s for s in pool
                   if price_source_health(s) in _INACTIVE_PRICE_HEALTH]
        active.sort(key=lambda s: -price_score(s))
        return active + demoted
    except Exception:  # noqa: BLE001
        return pool


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
                # Pass #91: the learned vocabulary, on the same surface
                "status": str(h.get("status") or "unknown"),
                "stale": h.get("stale", 0),
                "no_result": h.get("no_result", 0),
                "blocked": h.get("blocked", 0),
                "rate_limited": h.get("rate_limited", 0),
                "last_error_class": h.get("last_error_class", ""),
            }
        return out


def _healthy_chain(chain: list[str]) -> list[str]:
    """Filter a source chain to healthy sources (circuit breaker)."""
    return [s for s in chain if _source_healthy(s)]


# ── price learned-state persistence (Pass #91) ────────────────────────
def bind_price_source_store(store: Any) -> bool:
    """Bind (or re-bind) the KV store price learned state persists
    through — the same injected-store discipline the media source
    registry uses.  Idempotent for the same store object; a new
    store's persisted state loads in (in-memory state, which is
    fresher, wins per source).  Returns True when the binding
    changed.  Never raises."""
    global _price_store
    try:
        if store is None or _price_store is store:
            return False
        _price_store = store
        _load_price_state()
        return True
    except Exception:  # noqa: BLE001
        return False


def _persist_price_state() -> None:
    """Write the learned price state to the bound store.  Never
    raises; a no-op when no store is bound (memory-only)."""
    if _price_store is None:
        return
    try:
        with _health_lock:
            payload = {"version": _PRICE_STATE_VERSION,
                       "sources": {src: dict(h) for src, h
                                   in _source_health.items()}}
        raw = json.dumps(payload, ensure_ascii=False)
        if hasattr(_price_store, "set_raw"):
            _price_store.set_raw(PRICE_PERSIST_KEY, raw, "json")
        else:
            _price_store.set(PRICE_PERSIST_KEY, payload)
    except Exception:  # noqa: BLE001
        _log.debug("price source persist failed", exc_info=True)


def _load_price_state() -> None:
    """Restore learned price state from the bound store, as
    recorded — a stood-down source stays stood-down until a fresh
    outcome or probe revives it; nothing resurrects as trusted.
    In-memory rows (fresher) win per source.  Never raises."""
    if _price_store is None:
        return
    try:
        data = _price_store.get(PRICE_PERSIST_KEY)
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            return
        rows = data.get("sources") or {}
        with _health_lock:
            for src, raw in rows.items():
                if not isinstance(raw, dict) or src in _source_health:
                    continue
                h = _blank_health()
                for k in h:
                    if k in raw:
                        h[k] = raw[k]
                _source_health[str(src)] = h
    except Exception:  # noqa: BLE001
        _log.debug("price source load failed", exc_info=True)


def clear_price_source_state() -> bool:
    """Drop the persisted learned price state (the seed chains are
    untouched).  Returns True when a store is bound and the delete
    was issued.  Never raises."""
    try:
        if _price_store is None:
            return False
        delete = getattr(_price_store, "delete", None)
        if callable(delete):
            delete(PRICE_PERSIST_KEY)
            return True
        return False
    except Exception:  # noqa: BLE001
        return False


# ── staleness + error classification (Pass #91) ───────────────────────
#: How old a quote's provider timestamp may be before the quote is
#: STALE, per source — the provider's own cadence plus grace:
#: continuous feeds (binance ticker, gold-api spot) get minutes;
#: daily fixings (frankfurter, ECB) get 4 days so a Friday fixing
#: read on Monday is fresh, while a week-frozen feed is stale;
#: open.er-api advertises daily updates (2 days); yahoo/stooq daily
#: closes get the same weekend-tolerant 4 days (this is what
#: catches a delisted symbol's frozen meta — cf. the delisted
#: XAUUSD=X note in _METAL_YAHOO).  None = the provider supplies
#: no timestamp on this path, so freshness stays unknown.
_QUOTE_STALE_AFTER_S: dict[str, float | None] = {
    "binance": 300.0, "gold_api": 300.0,
    "frankfurter": 4 * 86400.0, "ecb": 4 * 86400.0,
    "open_er_api": 2 * 86400.0,
    "yahoo": 4 * 86400.0, "yahoo_metals": 4 * 86400.0,
    "stooq": 4 * 86400.0,
    "coingecko": None, "kraken": None, "coinbase": None,
}

_TF_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
               "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
               "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400,
               "3d": 259200, "1w": 604800, "1M": 2592000}


def _quote_freshness(source: str, quote_ts: float | None,
                     now: float = 0.0) -> tuple[str, float | None]:
    """(freshness, age_s) for one quote: ``fresh`` / ``stale`` when
    the provider supplied a timestamp and a cadence is known;
    ``unknown`` otherwise — a verdict is never fabricated."""
    if quote_ts is None:
        return "unknown", None
    threshold = _QUOTE_STALE_AFTER_S.get(source)
    if threshold is None:
        return "unknown", None
    age = max(0.0, (now or time.time()) - float(quote_ts))
    return ("stale" if age > threshold else "fresh"), age


def _ohlcv_stale_after_s(timeframe: str) -> float:
    """How old the newest bar may be before an OHLCV frame is stale:
    3× the timeframe for intraday (weekend/gap tolerant), a flat
    multi-day grace for daily-and-above fixings."""
    tf = _TF_SECONDS.get(timeframe, 3600)
    if tf >= 86400:
        return max(5 * 86400.0, 1.5 * tf)
    return max(900.0, 3.0 * tf)


def _frame_last_ts(df: Any) -> float | None:
    """Epoch-seconds of a frame's newest bar, or None.  Uses the
    index's nanosecond value (UTC-exact), like sentinel_bridge."""
    try:
        idx = df.index
        if len(idx) == 0:
            return None
        last = idx[-1]
        if hasattr(last, "value"):
            return float(last.value // 10**9)
        return float(last)
    except Exception:  # noqa: BLE001
        return None


def _classify_fetch_error(exc: BaseException) -> tuple[str, str]:
    """(kind, error_class) for one failed price fetch (Pass #91).

    ``kind`` is the recording vocabulary: ``config`` means RECORD
    NOTHING (a keyed source dark without its key is configuration,
    #90 precedent); ``no_result`` is the provider answering "no
    data for this symbol" (weak evidence, per-symbol reality);
    ``rate_limited`` / ``blocked`` are their own classes; anything
    else is ``failure`` with a transport error class."""
    msg = str(exc or "").lower()
    if "api_key" in msg or "apikey" in msg:
        return "config", ""
    if "rate limited" in msg or "429" in msg \
            or "too many requests" in msg:
        return "rate_limited", "rate_limited"
    import re as _re
    m = _re.search(r"http (\d{3})", msg)
    if m:
        code = int(m.group(1))
        if code == 429:
            return "rate_limited", "rate_limited"
        if code in (403, 451):
            return "blocked", "bot_block"
        return "failure", f"http_{code}"
    _NO_DATA = ("no klines", "no ohlc", "no candles", "no data for",
                "no data (", "empty series", "empty", "no rate for",
                "not in reference rates", "unsupported timeframe",
                "no futures symbol", "no keyless spot",
                "no metal history", "unknown asset", "need a pair")
    if any(p in msg for p in _NO_DATA):
        return "no_result", ""
    if "bad response" in msg or "unexpected response" in msg \
            or "parse failed" in msg:
        return "failure", "malformed_response"
    if "timed out" in msg or "timeout" in msg:
        return "failure", "timeout"
    if "fetch failed" in msg:
        return "failure", "network_failure"
    return "failure", "site_error"


def _record_quote_result(source: str, result: dict[str, Any],
                         now: float = 0.0) -> dict[str, Any]:
    """Record one served quote's outcome and surface its freshness
    on the result itself (Pass #91): ``quote_ts`` (the provider's
    own timestamp, when it supplied one), ``quote_age_s`` and
    ``freshness`` (fresh | stale | unknown) become part of the
    quote dict every consumer already receives; ``source_demoted``
    marks a serve from a demoted source — the fallback of last
    resort, visible, never silent.  Consumer accept/reject policy
    is unchanged: a stale quote is returned, recorded, surfaced."""
    now = now or time.time()
    ts = result.get("quote_ts")
    freshness, age = _quote_freshness(source, ts, now)
    result["quote_ts"] = ts
    result["quote_age_s"] = round(age, 3) if age is not None else None
    result["freshness"] = freshness
    if price_source_health(source) in _INACTIVE_PRICE_HEALTH:
        result["source_demoted"] = True
    kind = {"fresh": "fresh", "stale": "stale"}.get(freshness, "served")
    record_quote_outcome(source, kind, quote_ts=ts, age_s=age, now=now)
    return result


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
    # Pass #91: gold-api stamps each spot (``updatedAt``, ISO-8601) —
    # surface it as quote_ts so staleness is computed, never guessed.
    # (Recording is centralised in quote()/get_ohlcv since #91: the
    # adapters used to self-record AND be recorded by their caller,
    # double-counting every serve.)
    out: dict[str, Any] = {"symbol": sym["raw"], "price": price,
                           "change_pct_24h": None, "currency": "USD",
                           "source": "gold_api"}
    ts = _parse_iso_ts((d or {}).get("updatedAt"))
    if ts is not None:
        out["quote_ts"] = ts
    return out


def _parse_iso_ts(value: Any) -> float | None:
    """ISO-8601 timestamp → epoch seconds, or None.  Never raises."""
    try:
        if not value:
            return None
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:  # noqa: BLE001
        return None


def _parse_date_ts(value: Any) -> float | None:
    """A ``YYYY-MM-DD`` fixing date → epoch seconds (UTC midnight),
    or None.  Never raises."""
    try:
        if not value:
            return None
        import calendar as _cal
        import time as _time
        return float(_cal.timegm(
            _time.strptime(str(value)[:10], "%Y-%m-%d")))
    except Exception:  # noqa: BLE001
        return None


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
    out: dict[str, Any] = {"symbol": sym["raw"], "price": rate,
                           "change_pct_24h": None,
                           "currency": quote_c, "source": "open_er_api"}
    # Pass #91: the payload stamps its own last update — surface it.
    try:
        updated = float((d or {}).get("time_last_update_unix") or 0)
        if updated:
            out["quote_ts"] = updated
    except (TypeError, ValueError):
        pass
    return out


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
        # Pass #91: the fixing's own date lives on the time Cube.
        timed = root.findall(".//e:Cube[@time]", ns)
        fixing_day = timed[0].get("time") if timed else None
    except Exception as exc:
        raise MarketDataError(f"ecb: XML parse failed: {exc}") from exc
    if base not in rates or quote_c not in rates:
        raise MarketDataError(
            f"ecb: pair {base}/{quote_c} not in reference rates")
    # ECB quotes EUR/XXX; cross: base/quote = (EUR/quote) / (EUR/base)
    rate = rates[quote_c] / rates[base]
    out: dict[str, Any] = {"symbol": sym["raw"], "price": rate,
                           "change_pct_24h": None,
                           "currency": quote_c, "source": "ecb"}
    ts = _parse_date_ts(fixing_day)
    if ts is not None:
        out["quote_ts"] = ts
    return out


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
        # Pass #91: the learned order (day-one pin: zero evidence →
        # the chain below byte-for-byte), then the circuit breaker;
        # when every source is cooling/demoted the learned order is
        # tried anyway — the demoted are the fallback of last
        # resort, never silently dropped.
        full = _keyed_chain(market) + _DEFAULT_CHAINS.get(market,
                                                          ["binance"])
        chain = _healthy_chain(ordered_price_chain(market, full))
        if not chain:
            # all sources cooling — try anyway, don't hard-fail
            chain = ordered_price_chain(market, full)
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
            # Pass #91: bars always carry timestamps, so bar
            # staleness is computed against the timeframe cadence
            # and recorded; the frame surfaces it via attrs.
            last_ts = _frame_last_ts(df)
            freshness, age = ("unknown", None) if last_ts is None \
                else (("stale" if (time.time() - last_ts)
                        > _ohlcv_stale_after_s(timeframe)
                        else "fresh"),
                      max(0.0, time.time() - last_ts))
            record_quote_outcome(
                src, {"fresh": "fresh", "stale": "stale"}.get(
                    freshness, "served"),
                quote_ts=last_ts, age_s=age)
            try:
                df.attrs["source"] = src
                df.attrs["freshness"] = freshness
                df.attrs["last_bar_ts"] = last_ts
                if price_source_health(src) in _INACTIVE_PRICE_HEALTH:
                    df.attrs["source_demoted"] = True
            except Exception:  # noqa: BLE001 - attrs are surface only
                pass
            _log.info("market_data: %s %s %s via %s (%d bars)",
                      symbol, market, timeframe, src, len(df))
            return df
        except MarketDataError as exc:
            kind, error_class = _classify_fetch_error(exc)
            if kind != "config":  # config records nothing (#90)
                record_quote_outcome(src, kind,
                                     error_class=error_class)
            errors.append(f"{src}: {exc}")
        except Exception as exc:  # noqa: BLE001 - adapter bug, try next
            record_quote_outcome(src, "failure",
                                 error_class="unexpected")
            errors.append(f"{src}: unexpected {exc}")
            _log.warning("market_data adapter %s failed: %s", src, exc)
    raise MarketDataError(
        f"all sources failed for {symbol} [{market}]: "
        + "; ".join(errors))


# ── quotes (stdlib-only, no pandas) ───────────────────────────────────
def _live_chain(market: str, chain: list[str]) -> list[str]:
    """The chain a quote fetch actually walks (Pass #91): the
    learned order (day-one pin: zero evidence → *chain* untouched),
    circuit-breaker filtered; when every source is cooling the
    learned order is walked anyway — demoted/cooling sources are
    the fallback of last resort, and a serve from one is marked
    ``source_demoted`` on the result by :func:`_record_quote_result`."""
    ordered = ordered_price_chain(market, chain)
    live = _healthy_chain(ordered)
    return live or ordered


def _record_failure(source: str, exc: BaseException) -> None:
    """Record one failed quote attempt with its classified kind —
    except ``config``, which records nothing (#90 precedent)."""
    kind, error_class = _classify_fetch_error(exc)
    if kind != "config":
        record_quote_outcome(source, kind, error_class=error_class)


def quote(symbol: str, market: str = "crypto") -> dict[str, Any]:
    """Latest price + 24h change for one symbol. Keyless, no pandas.

    Returns ``{"symbol","price","change_pct_24h","currency","source"}``
    plus, since Pass #91, the freshness surface: ``quote_ts`` (the
    provider's own timestamp when it supplied one, else None),
    ``quote_age_s`` and ``freshness`` (fresh | stale | unknown),
    and ``source_demoted`` when a demoted source served as the
    fallback of last resort.  ``change_pct_24h`` may be None when
    the source has no change data.  Every attempt records its
    outcome (see :func:`record_quote_outcome`); a stale quote is
    still returned — recorded and surfaced, never silently dropped.

    Metal symbols (XAUUSD, XAG, …) are auto-routed to the commodities
    market even when ``market="forex"`` is passed.
    """
    market = _detect_market(symbol, (market or "crypto").strip().lower())
    sym = normalize_symbol(symbol, market)
    disp = (symbol or "").strip().upper()
    if market == "crypto":
        for src in _live_chain(market, ["binance", "coingecko"]):
            try:
                if src == "binance":
                    d = _json(
                        "https://api.binance.com/api/v3/ticker/24hr",
                        {"symbol": sym["binance"]})
                    if not (d or {}).get("lastPrice"):
                        raise MarketDataError(
                            f"binance: bad response ({str(d)[:120]})")
                    bid = float(d["bidPrice"]) if d.get("bidPrice") else None
                    ask = float(d["askPrice"]) if d.get("askPrice") else None
                    hi = float(d["highPrice"]) if d.get("highPrice") else None
                    lo = float(d["lowPrice"]) if d.get("lowPrice") else None
                    spread_bps = ((ask - bid) / float(d["lastPrice"])
                                  * 10000) if bid and ask else None
                    result: dict[str, Any] = {
                        "symbol": disp,
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
                    # binance stamps the ticker (closeTime, ms) —
                    # surface it; absent → freshness stays unknown.
                    if d.get("closeTime"):
                        result["quote_ts"] = float(d["closeTime"]) / 1000.0
                    return _record_quote_result("binance", result)
                coin = _CG_MAP.get(sym["base"], sym["base"].lower())
                d = _json(
                    "https://api.coingecko.com/api/v3/simple/price",
                    {"ids": coin, "vs_currencies": "usd",
                     "include_24hr_change": "true"})
                row = (d or {}).get(coin) or {}
                if "usd" not in row:
                    raise MarketDataError(
                        f"coingecko: no data for {coin}")
                # CoinGecko's simple/price carries NO timestamp —
                # freshness stays honestly unknown (never fabricated).
                return _record_quote_result("coingecko", {
                    "symbol": disp, "price": float(row["usd"]),
                    "change_pct_24h": row.get("usd_24h_change"),
                    "currency": "USD", "source": "coingecko"})
            except Exception as exc:  # noqa: BLE001 - try next source
                _record_failure(src, exc)
                continue
        raise MarketDataError(f"no crypto quote for {disp}")
    if market == "stocks":
        # NOTE (survey, Pass #91): this path recorded NOTHING before
        # #91 — yahoo's success/failure and stooq's serve were the
        # only quote outcomes in the layer with no health record.
        for src in _live_chain(market, ["yahoo", "stooq"]):
            try:
                if src == "yahoo":
                    ysym = _yahoo_symbol(sym, market)
                    d = _json(
                        "https://query1.finance.yahoo.com/v8/finance/"
                        f"chart/{ysym}",
                        {"interval": "1d", "range": "5d"})
                    meta = (d or {}).get("chart", {}).get(
                        "result", [{}])[0].get("meta", {})
                    price = meta.get("regularMarketPrice")
                    if price is None:
                        raise MarketDataError(
                            f"yahoo: no data for {ysym}")
                    result = {"symbol": disp, "price": float(price),
                              "change_pct_24h": None,
                              "currency": str(
                                  meta.get("currency") or "USD"),
                              "source": "yahoo"}
                    if meta.get("regularMarketTime"):
                        result["quote_ts"] = float(
                            meta["regularMarketTime"])
                    return _record_quote_result("yahoo", result)
                raw = _fetch("https://stooq.com/q/l/",
                             {"s": sym["stooq"], "f": "sd2t2ohlcv",
                              "h": "", "e": "csv"}).decode(
                                  "utf-8", "ignore")
                rows = list(csv.DictReader(io.StringIO(raw)))
                if not rows:
                    raise MarketDataError(
                        f"stooq: no data for {sym['stooq']}")
                r = rows[0]
                try:
                    price = float(r["Close"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise MarketDataError(
                        f"stooq: bad response ({str(r)[:120]})") from exc
                result = {"symbol": disp, "price": price,
                          "change_pct_24h": None, "currency": "USD",
                          "source": "stooq"}
                ts = _parse_date_ts(r.get("Date"))
                if ts is not None and r.get("Time"):
                    try:
                        hh, mm, ss = (int(p) for p in
                                      str(r["Time"]).split(":"))
                        ts += hh * 3600 + mm * 60 + ss
                    except (TypeError, ValueError):
                        pass
                if ts is not None:
                    result["quote_ts"] = ts
                return _record_quote_result("stooq", result)
            except Exception as exc:  # noqa: BLE001 - try next source
                _record_failure(src, exc)
                continue
        raise MarketDataError(f"no stock quote for {disp}")
    if market == "forex":
        base, quote_c = sym["base"], sym["quote"]
        # Chain: frankfurter → open.er-api.com → ECB direct, in the
        # learned order; each attempt records its outcome.
        errors = []
        for src in _live_chain(market,
                               ["frankfurter", "open_er_api", "ecb"]):
            try:
                if src == "frankfurter":
                    d = _json(f"https://api.frankfurter.dev/v2/rate/"
                              f"{base.lower()}/{quote_c.lower()}")
                    try:
                        rate = float((d or {})["rate"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise MarketDataError(
                            f"frankfurter: bad response "
                            f"({str(d)[:120]})") from exc
                    result = {"symbol": disp, "price": rate,
                              "change_pct_24h": None,
                              "currency": quote_c,
                              "source": "frankfurter"}
                    ts = _parse_date_ts((d or {}).get("date"))
                    if ts is not None:
                        result["quote_ts"] = ts
                    return _record_quote_result("frankfurter", result)
                elif src == "open_er_api":
                    return _record_quote_result(
                        "open_er_api", _open_er_api_quote(sym))
                elif src == "ecb":
                    return _record_quote_result(
                        "ecb", _ecb_direct_quote(sym))
            except Exception as exc:  # noqa: BLE001 - try next
                _record_failure(src, exc)
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
                result = {"symbol": disp, "price": float(price),
                          "change_pct_24h": None,
                          "currency": quote_c or "USD",
                          "source": "yahoo"}
                if meta.get("regularMarketTime"):
                    result["quote_ts"] = float(meta["regularMarketTime"])
                return _record_quote_result("yahoo", result)
            raise MarketDataError(f"yahoo: no data for {ysym}")
        except Exception as exc:  # noqa: BLE001
            _record_failure("yahoo", exc)
            errors.append(f"yahoo: {exc}")
        raise MarketDataError(
            f"no forex quote for {base}/{quote_c}: {'; '.join(errors)}")
    if market == "commodities":
        # Chain: gold-api.com (real-time) → frankfurter (daily fixing).
        errors = []
        for src in _live_chain(market, ["gold_api", "frankfurter"]):
            try:
                if src == "gold_api":
                    return _record_quote_result(
                        "gold_api", _goldapi_quote(sym))
                d = _json(f"https://api.frankfurter.dev/v2/rate/"
                          f"{sym['base'].lower()}/usd")
                try:
                    rate = float((d or {})["rate"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise MarketDataError(
                        f"frankfurter: bad response "
                        f"({str(d)[:120]})") from exc
                result = {"symbol": disp, "price": rate,
                          "change_pct_24h": None, "currency": "USD",
                          "source": "frankfurter"}
                ts = _parse_date_ts((d or {}).get("date"))
                if ts is not None:
                    result["quote_ts"] = ts
                return _record_quote_result("frankfurter", result)
            except Exception as exc:  # noqa: BLE001 - try next
                _record_failure(src, exc)
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


# ── price class status + health loop (Pass #91) ───────────────────────
def price_source_status() -> dict[str, Any]:
    """The price class's learned state (Pass #91): one row per
    provider (keyless roster + keyed upgrades) with tier, keyed
    flag, learned health, outcome counts (fresh-served / failed /
    stale / blocked / rate-limited / no-result), score and last
    error class; the per-market chains in their current learned
    order; and the discovery verdict.  Never raises."""
    try:
        rows = []
        for src in list(SOURCES) + [s for s in
                                    ("alphavantage", "twelvedata",
                                     "finnhub") if s not in SOURCES]:
            st = price_source_stats(src)
            with _health_lock:
                h = _source_health.get(src) or {}
                cooling = time.time() < float(
                    h.get("cooling_until") or 0.0)
            rows.append({
                "source": src, "tier": SOURCE_TIERS.get(src, 9),
                "keyed": src in KEYED_SOURCES.values(),
                "covers": (SOURCES.get(src) or {}).get("covers", ""),
                "health": st["health"], "score": price_score(src),
                "ok": st["ok"], "fail": st["fail"],
                "stale": st["stale"], "blocked": st["blocked"],
                "rate_limited": st["rate_limited"],
                "no_result": st["no_result"],
                "last_error_class": st["last_error_class"],
                "last_quote_ts": st["last_quote_ts"],
                "last_age_s": st["last_age_s"],
                "cooling_down": cooling})
        chains = {market: ordered_price_chain(
            market, _keyed_chain(market) + list(chain))
            for market, chain in _DEFAULT_CHAINS.items()}
        return {
            "sources": rows, "chains": chains,
            "persisted": _price_store is not None,
            "candidates_pending": 0,
            "discovery": (
                "no candidate intake for the price class: new price "
                "providers are APIs and no genuine discovery channel "
                "exists in code (named absence, Pass #91) — outcomes, "
                "health, ordering and persistence only"),
        }
    except Exception:  # noqa: BLE001
        return {"sources": [], "chains": {}, "persisted": False,
                "candidates_pending": 0, "discovery": ""}


def _price_probes() -> dict[str, Callable[[], tuple[float | None, str]]]:
    """One bounded REAL fetch per keyless provider (Pass #91): the
    honest probe for a quote API is a quote — one liquid symbol,
    one call.  Each probe returns (provider_timestamp, timeframe)
    where timeframe is "" for quote probes (cadence judged by
    :func:`_quote_freshness`) or the bar timeframe for OHLC probes.
    Keyed providers are deliberately ABSENT: probing them would
    burn the owner's quota — they are judged by real-fetch
    outcomes only."""
    btc = normalize_symbol("BTC/USDT", "crypto")

    def _binance() -> tuple[float | None, str]:
        d = _json("https://api.binance.com/api/v3/ticker/24hr",
                  {"symbol": btc["binance"]})
        if not (d or {}).get("lastPrice"):
            raise MarketDataError("binance: bad response (probe)")
        ts = d.get("closeTime")
        return (float(ts) / 1000.0 if ts else None), ""

    def _kraken() -> tuple[float | None, str]:
        return _frame_last_ts(_kraken_ohlc(btc, "1h", 1)), "1h"

    def _coinbase() -> tuple[float | None, str]:
        return _frame_last_ts(_coinbase_candles(btc, "1h", 1)), "1h"

    def _coingecko() -> tuple[float | None, str]:
        d = _json("https://api.coingecko.com/api/v3/simple/price",
                  {"ids": "bitcoin", "vs_currencies": "usd"})
        if "usd" not in ((d or {}).get("bitcoin") or {}):
            raise MarketDataError("coingecko: no data for bitcoin")
        return None, ""  # no timestamp on this path — honestly unknown

    def _yahoo() -> tuple[float | None, str]:
        d = _json("https://query1.finance.yahoo.com/v8/finance/chart/"
                  "AAPL", {"interval": "1d", "range": "5d"})
        meta = (d or {}).get("chart", {}).get("result", [{}])[0].get(
            "meta", {})
        if meta.get("regularMarketPrice") is None:
            raise MarketDataError("yahoo: no data for AAPL")
        ts = meta.get("regularMarketTime")
        return (float(ts) if ts else None), ""

    def _stooq() -> tuple[float | None, str]:
        df = _stooq_csv(normalize_symbol("AAPL", "stocks"), "1d", 1)
        return _frame_last_ts(df), "1d"

    def _frankfurter() -> tuple[float | None, str]:
        d = _json("https://api.frankfurter.dev/v2/rate/eur/usd")
        if not (d or {}).get("rate"):
            raise MarketDataError("frankfurter: bad response (probe)")
        return _parse_date_ts(d.get("date")), ""

    def _gold_api() -> tuple[float | None, str]:
        q = _goldapi_quote(normalize_symbol("XAUUSD", "commodities"))
        return q.get("quote_ts"), ""

    def _yahoo_metals() -> tuple[float | None, str]:
        df = _yahoo_metal_ohlc(
            normalize_symbol("XAUUSD", "commodities"), "1d", 1)
        return _frame_last_ts(df), "1d"

    def _open_er_api() -> tuple[float | None, str]:
        q = _open_er_api_quote(normalize_symbol("USDEUR", "forex"))
        return q.get("quote_ts"), ""

    def _ecb() -> tuple[float | None, str]:
        q = _ecb_direct_quote(normalize_symbol("EURUSD", "forex"))
        return q.get("quote_ts"), ""

    return {"binance": _binance, "kraken": _kraken,
            "coinbase": _coinbase, "coingecko": _coingecko,
            "yahoo": _yahoo, "stooq": _stooq,
            "frankfurter": _frankfurter, "gold_api": _gold_api,
            "yahoo_metals": _yahoo_metals,
            "open_er_api": _open_er_api, "ecb": _ecb}


def verify_price_sources(now: float = 0.0) -> dict[str, Any]:
    """The price health loop (Pass #91): probe every KEYLESS
    provider with one bounded real fetch for a liquid symbol and
    record the outcome through the same vocabulary as real fetches
    (a probe IS a real fetch).  Keyed providers are never probed —
    probing would burn quota — they are reported ``skipped_keyed``
    and judged by real-fetch outcomes.  Persists via the recorder.
    Never raises."""
    now = now or time.time()
    summary: dict[str, Any] = {
        "checked": 0, "verified_live": 0, "stale": 0,
        "rate_limited": 0, "bot_blocked": 0, "unverifiable": 0,
        "stood_down": 0, "disabled": 0,
        "skipped_keyed": sorted(KEYED_SOURCES.values()),
        "changed": []}
    bucket = {"verified-live": "verified_live", "stale": "stale",
              "rate-limited": "rate_limited",
              "bot-blocked": "bot_blocked",
              "unverifiable": "unverifiable",
              "stood-down": "stood_down", "disabled": "disabled"}
    try:
        for src, probe in _price_probes().items():
            before = price_source_health(src)
            try:
                ts, timeframe = probe()
            except Exception as exc:  # noqa: BLE001 - record, move on
                kind, error_class = _classify_fetch_error(exc)
                if kind == "no_result":
                    # the probe symbol is liquid by construction:
                    # no data for it is a source problem, not a
                    # catalogue miss — record it unverifiable-grade
                    kind, error_class = "failure", "probe_no_data"
                if kind != "config":
                    record_quote_outcome(src, kind,
                                         error_class=error_class,
                                         now=now)
            else:
                if timeframe:
                    freshness = "unknown" if ts is None else (
                        "stale" if (now - ts)
                        > _ohlcv_stale_after_s(timeframe) else "fresh")
                    age = None if ts is None else max(0.0, now - ts)
                else:
                    freshness, age = _quote_freshness(src, ts, now)
                record_quote_outcome(
                    src, {"fresh": "fresh", "stale": "stale"}.get(
                        freshness, "served"),
                    quote_ts=ts, age_s=age, now=now)
            summary["checked"] += 1
            after = price_source_health(src)
            if bucket.get(after):
                summary[bucket[after]] += 1
            if before != after:
                summary["changed"].append(
                    {"source": src, "from": before, "to": after})
    except Exception:  # noqa: BLE001 - the loop reports, never raises
        _log.debug("price source health loop failed", exc_info=True)
    return summary


def source_status() -> dict[str, Any]:
    """Which sources are usable right now (keyless always; keyed iff env).

    Includes reliability tiers and live health (circuit-breaker state).
    Pass #91: also carries the price class's learned state under
    ``price_sources`` — this EXISTING surface is extended, not
    replaced, so sentinel_bridge's doctor and every other reader
    keep working.
    """
    keyed = {src: bool(_key(env)) for env, src in KEYED_SOURCES.items()}
    return {"keyless": list(SOURCES),
            "keyed": keyed,
            "tiers": SOURCE_TIERS,
            "default_chains": _DEFAULT_CHAINS,
            "health": source_health(),
            "price_sources": price_source_status()}


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
