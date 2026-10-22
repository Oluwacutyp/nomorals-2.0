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

__all__ = ["DEX_API", "token_snapshot", "default_http_get"]

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
