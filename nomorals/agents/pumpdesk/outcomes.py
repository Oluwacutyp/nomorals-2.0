"""Outcome data for grading calls — DexScreener's free surface.

The design's prototype-first ruling (§4.3): grade calls against
DexScreener/Birdeye's free APIs before any keyed streaming tier
earns its budget.  DexScreener needs no key: pair data for a mint,
from which the grader reads the current native price (SOL), the
venue (a pumpswap pair existing at all IS graduation evidence),
and liquidity.  Historical peaks are NOT available here — peak /
excursion truth comes from the desk's own observation series
(calls.py says which source graded each call).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Optional

__all__ = ["DEX_API", "token_snapshot", "token_attention",
           "default_http_get"]

DEX_API = "https://api.dexscreener.com"


def default_http_get(url: str, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={
        "User-Agent": "nomorals-pumpdesk", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def token_snapshot(mint: str,
                   http_get: Callable = default_http_get) -> Optional[dict]:
    """Best Solana pair for a mint, or None when DexScreener has
    nothing (ungraduated / unindexed — a real answer, not an error).

    'Best' = a pumpswap/pump.fun pair first (the canonical venue),
    else the deepest-liquidity Solana pair."""
    status, text = http_get(f"{DEX_API}/latest/dex/tokens/{mint}", {})
    if status != 200:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    pairs = [p for p in (data.get("pairs") or [])
             if p.get("chainId") == "solana"]
    if not pairs:
        return None

    def _rank(p: dict) -> tuple:
        dex = (p.get("dexId") or "").lower()
        canonical = 1 if dex in ("pumpswap", "pumpfun") else 0
        liq = float(((p.get("liquidity") or {}).get("usd")) or 0)
        return (canonical, liq)

    best = max(pairs, key=_rank)
    try:
        price_native = float(best.get("priceNative") or 0)
    except (TypeError, ValueError):
        price_native = 0.0
    return {
        "dex": best.get("dexId", ""),
        "pair": best.get("pairAddress", ""),
        "price_native": price_native,          # in SOL for SOL pairs
        "market_cap_usd": best.get("marketCap"),
        "fdv_usd": best.get("fdv"),
        "liquidity_usd": (best.get("liquidity") or {}).get("usd"),
        "pair_created_at": best.get("pairCreatedAt"),
        "url": best.get("url", ""),
    }


def _best_pair(mint: str,
               http_get: Callable) -> Optional[dict]:
    """The raw best Solana pair for a mint (same ranking as
    token_snapshot), or None when DexScreener has nothing."""
    status, text = http_get(f"{DEX_API}/latest/dex/tokens/{mint}", {})
    if status != 200:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    pairs = [p for p in (data.get("pairs") or [])
             if p.get("chainId") == "solana"]
    if not pairs:
        return None

    def _rank(p: dict) -> tuple:
        dex = (p.get("dexId") or "").lower()
        canonical = 1 if dex in ("pumpswap", "pumpfun") else 0
        liq = float(((p.get("liquidity") or {}).get("usd")) or 0)
        return (canonical, liq)

    return max(pairs, key=_rank)


def token_attention(mint: str,
                    http_get: Callable = default_http_get
                    ) -> Optional[dict]:
    """The attention surface DexScreener exposes for a mint's best
    pair: declared social links, website count, and paid-boost
    state.  Returns None when DexScreener has no pair at all.
    Fields DexScreener does not populate for the pair stay None —
    the attention verifier records them as named gaps, never as
    zeros (a pair without an ``info`` block has UNKNOWN socials,
    not zero socials)."""
    best = _best_pair(mint, http_get)
    if best is None:
        return None
    info = best.get("info")
    socials = None
    websites = None
    if isinstance(info, dict):
        raw_socials = info.get("socials")
        if isinstance(raw_socials, list):
            socials = [{"type": str(s.get("type", "")),
                        "url": str(s.get("url", ""))}
                       for s in raw_socials if isinstance(s, dict)]
        raw_sites = info.get("websites")
        if isinstance(raw_sites, list):
            websites = len(raw_sites)
    boosts = best.get("boosts")
    boosts_active = None
    if isinstance(boosts, dict) and boosts.get("active") is not None:
        try:
            boosts_active = int(boosts["active"])
        except (TypeError, ValueError):
            boosts_active = None
    return {
        "dex": best.get("dexId", ""),
        "pair": best.get("pairAddress", ""),
        "socials": socials,
        "websites": websites,
        "boosts_active": boosts_active,
    }
