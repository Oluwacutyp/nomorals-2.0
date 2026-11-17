"""Desk sources: markets join the shared source registry.

Two existing organs, extended — never paralleled:

* ``media/source_registry.SourceRegistry`` is the ONE source of truth
  every class grows through (downloads #86, music #88, books #89,
  betting #8x).  The desk registers its seed sources under kind
  ``markets`` the same way: seeds are the floor, outcomes move them,
  candidates enter untrusted.  Seed domains coincide with the
  collector's real fetch path (``integrations/market_data``'s
  keyless-first chain), so a fetch outcome and a registry health
  record are about the same host.

* The keyed upgrades in that chain (Alpha Vantage, Twelve Data,
  Finnhub) are env-keyed only: they have NO entries in the real
  self-provisioning registry (``accounts/provisioning``), so the
  desk does not invent provisioning flows for them.  Their state is
  reported honestly by :func:`keyed_feed_state` — a keyed feed
  without its key is a skipped feed with the reason, never a silent
  zero — and the keyless chain serves until a key exists.
"""

from __future__ import annotations

import os
from typing import Any

from .failures import record_failure

__all__ = [
    "MARKET_SEEDS", "KEYED_FEEDS", "ensure_market_sources",
    "market_status", "market_registry_for_context", "keyed_feed_state",
]


# ── source-registry seeds ──────────────────────────────────────────────

#: (nation, domain, name, auth_needed, notes)
MARKET_SEEDS: tuple = (
    ("GLOBAL", "api.binance.com", "Binance", False,
     "Public spot klines + 24h ticker; the crypto chain's first rung, "
     "keyless."),
    ("GLOBAL", "api.kraken.com", "Kraken", False,
     "Public OHLC; the crypto chain's keyless fallback."),
    ("GLOBAL", "api.bybit.com", "Bybit", False,
     "Public v5 klines; the crypto chain's keyless fallback."),
    ("GLOBAL", "api.coingecko.com", "CoinGecko", False,
     "Public market data; the crypto chain's keyless fallback."),
    ("GLOBAL", "query1.finance.yahoo.com", "Yahoo Finance", False,
     "v8 chart API (intraday + daily); the stocks chain's first "
     "rung, keyless."),
    ("GLOBAL", "stooq.com", "Stooq", False,
     "Daily CSV quotes; the stocks chain's keyless fallback."),
)

#: Keyed upgrades in the price class's chain: (service, env key).
#: Presence is reported, never provisioned — these services have no
#: entries in accounts/provisioning, so there is no drive to run.
KEYED_FEEDS: tuple = (
    ("alphavantage", "ALPHAVANTAGE_API_KEY"),
    ("twelvedata", "TWELVEDATA_API_KEY"),
    ("finnhub", "FINNHUB_API_KEY"),
)


def _seed_descriptor(nation: str, domain: str, name: str, auth: bool,
                     notes: str) -> Any:
    from ...media.source_registry import Licence, SourceDescriptor, TrustTier
    return SourceDescriptor(
        nation=nation, language="en", domain=domain, name=name,
        kinds=("markets",),
        licence=Licence.UNKNOWN,
        trust_tier=TrustTier.UNVERIFIED, auth_needed=auth,
        search_url_template="",
        notes=notes)


def ensure_market_sources(registry: Any) -> list[dict[str, Any]]:
    """Register the markets seed sources on the shared registry.

    Idempotent: a seed already present (by domain) with the current
    shape is reported, not duplicated.  Returns one status dict per
    seed.
    """
    out: list[dict[str, Any]] = []
    for nation, domain, name, auth, notes in MARKET_SEEDS:
        desc = _seed_descriptor(nation, domain, name, auth, notes)
        existing = registry.for_domain(domain)
        if existing is not None:
            if (tuple(existing.kinds) == tuple(desc.kinds)
                    and existing.auth_needed == desc.auth_needed):
                out.append({"domain": domain, "already_registered": True})
                continue
            registry._by_key[desc.key] = desc
            try:
                registry._persist()
            except Exception as e:  # persistence is best-effort
                record_failure("sources.ensure_market_sources.persist_update", e)
                pass
            out.append({"domain": domain, "updated": True})
            continue
        added = registry.register(desc)
        if added:
            try:
                registry._persist()
            except Exception as e:  # persistence is best-effort
                record_failure("sources.ensure_market_sources.persist_register", e)
                pass
        out.append({"domain": domain, "registered": bool(added)})
    return out


def market_registry_for_context(context: Any = None,
                                injected: Any = None) -> Any:
    """The source registry the desk's outcomes land in.

    Mirrors the betting desk resolver: an injected registry wins;
    otherwise the shared ``DEFAULT_REGISTRY`` bound to the context's
    KV store — the same persisted state the 6h health loop verifies
    — so desk-registered seeds are visible to the loop (and survive
    restarts).  A db-less context (CLI, tests) keeps the unbound
    singleton: a working registry, memory-only.  Never raises.
    """
    if injected is not None:
        return injected
    from ...media.source_registry import DEFAULT_REGISTRY
    reg = DEFAULT_REGISTRY
    try:
        db = getattr(context, "db", None) if context is not None else None
        if db is not None:
            from ...storage.kv import KVStore
            reg.bind_store(KVStore(db))
    except Exception as e:  # persistence is an enhancement;
        record_failure("sources.market_registry_for_context", e)
        pass  # its absence never sinks the desk
    return reg


def market_status(registry: Any) -> dict[str, Any]:
    """Registry view of the markets class: sources + health + origin."""
    rows = []
    for desc in registry.by_kind("markets"):
        rows.append({
            "domain": desc.domain, "name": desc.name,
            "health": registry.health_for(desc),
            "origin": registry.origin_for(desc),
            "active": registry.is_active(desc),
            "stats": registry.stats_for(desc),
        })
    return {"sources": rows, "count": len(rows)}


def keyed_feed_state() -> list[dict[str, Any]]:
    """Keyed upgrades in the price chain: presence only, never values.

    A feed without its env key is ``skipped — keyless chain only``:
    the desk reports the gap instead of silently serving fewer
    sources.  These services are NOT provisioned (no entries in
    accounts/provisioning); setting the env key is the owner's act.
    """
    out = []
    for service, env_key in KEYED_FEEDS:
        present = bool(os.environ.get(env_key, "").strip())
        out.append({
            "service": service, "env_key": env_key,
            "key_present": present,
            "status": ("present" if present
                       else "skipped — keyless chain only"),
        })
    return out
