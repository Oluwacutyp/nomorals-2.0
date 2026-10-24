"""Desk sources: pump.fun intelligence joins the shared registry.

One existing organ, extended — never paralleled (the betdesk
sources pattern):

* ``media/source_registry.SourceRegistry`` gains the desk's seed
  sources under kind ``memecoin``: PumpPortal's data socket, the
  public Solana RPC (the always-free floor), the pump.fun frontend
  API (backfill), DexScreener (outcome grading), Birdeye (keyed).
  Seeds are the floor — the ONE health loop (stand-down 3, disable
  10) grades them from real outcomes the firehose records, and the
  firehose fails over in that order.  Open-sources law: the desk
  grows into good feeds and sheds dead ones through the registry,
  not through a hard-coded list.
* ``accounts/provisioning`` gains the desk's one keyed tier worth
  having on day one — Helius (RPC + websockets without public
  rate limits) — as a service with a LIVE validator.  Provisioning
  is the owner's hand (his account, his key, vaulted); the desk
  runs keyless until he says otherwise.
"""

from __future__ import annotations

import os
from typing import Any
from .failures import record_failure

__all__ = [
    "MEMCOIN_SEEDS", "ensure_pump_sources", "pump_status",
    "pump_registry_for_context", "ensure_provisioning",
    "HELIUS_SERVICE", "PUMPPORTAL_SERVICE", "rpc_endpoint_from_env",
    "rpc_routes", "PUBLIC_RPC_ENDPOINTS",
    "desk_user_agent", "desk_source_budget_s", "pumpportal_api_key",
    "DEFAULT_USER_AGENT", "DEFAULT_SOURCE_BUDGET_S",
]

#: (nation, domain, name, auth_needed, notes)
MEMCOIN_SEEDS: tuple = (
    ("GLOBAL", "pumpportal.fun", "PumpPortal data API", False,
     "Free websocket subs for new-token + migration events; trade "
     "streams are keyed/metered since May 2026. The firehose's "
     "primary intake."),
    ("GLOBAL", "api.mainnet-beta.solana.com", "Solana public RPC",
     False,
     "The always-free floor: logsSubscribe on the pump program + "
     "account reads. Rate-limited; every keyed RPC tier replaces "
     "it through the same interface."),
    ("GLOBAL", "solana-rpc.publicnode.com", "Solana RPC (PublicNode)",
     False,
     "PublicNode's free public endpoint — a second keyless RPC "
     "route so holder reads are not hostage to one endpoint's "
     "per-method throttles. Health-graded like every source; "
     "stood down when it stops answering."),
    ("GLOBAL", "solana.drpc.org", "Solana RPC (dRPC public)",
     False,
     "dRPC's free public endpoint — a third keyless RPC route in "
     "the holder-read failover. Health-graded; shed when dead."),
    ("GLOBAL", "frontend-api-v3.pump.fun", "pump.fun frontend API",
     False,
     "Unofficial REST the pump.fun site itself reads: coin lists "
     "for backfill/candidate discovery. Schema drifts; the health "
     "loop is the containment."),
    ("GLOBAL", "api.dexscreener.com", "DexScreener API", False,
     "Free pair/token surface — the outcome grader for calls "
     "(post-graduation prices). No key, generous rate limits."),
    ("GLOBAL", "public-api.birdeye.so", "Birdeye API", True,
     "Keyed token analytics (holders, trades). Optional tier; the "
     "desk provisions it only on the owner's word."),
)

_SEED_EXTRAS: dict[str, dict[str, Any]] = {
    "public-api.birdeye.so": {"kinds": ("memecoin", "api")},
    "api.mainnet-beta.solana.com": {"kinds": ("memecoin", "api")},
    "solana-rpc.publicnode.com": {"kinds": ("memecoin", "api")},
    "solana.drpc.org": {"kinds": ("memecoin", "api")},
}


def _seed_descriptor(nation: str, domain: str, name: str, auth: bool,
                     notes: str) -> Any:
    from ...media.source_registry import (
        Licence, SourceDescriptor, TrustTier)
    extras = _SEED_EXTRAS.get(domain, {})
    return SourceDescriptor(
        nation=nation, language="en", domain=domain, name=name,
        kinds=tuple(extras.get("kinds", ("memecoin",))),
        licence=Licence.UNKNOWN,
        trust_tier=TrustTier.UNVERIFIED, auth_needed=auth,
        notes=notes)


def ensure_pump_sources(registry: Any) -> list[dict[str, Any]]:
    """Register the memecoin seed sources (idempotent; a persisted
    older shape is updated in place — the betdesk Pass-8 rule)."""
    out: list[dict[str, Any]] = []
    for nation, domain, name, auth, notes in MEMCOIN_SEEDS:
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
            except Exception as exc:  # noqa: BLE001
                record_failure("sources.ensure_pump_sources.update", exc)
            out.append({"domain": domain, "updated": True})
            continue
        added = registry.register(desc)
        if added:
            try:
                registry._persist()
            except Exception as exc:  # noqa: BLE001
                record_failure("sources.ensure_pump_sources.register", exc)
        out.append({"domain": domain, "registered": bool(added)})
    return out


def pump_registry_for_context(context: Any = None,
                              injected: Any = None) -> Any:
    """The registry the desk's outcomes land in: injected wins;
    else the shared DEFAULT_REGISTRY bound to the context KV store
    (the #94/#Pass-8 resolver shape).  Never raises."""
    if injected is not None:
        return injected
    from ...media.source_registry import DEFAULT_REGISTRY
    reg = DEFAULT_REGISTRY
    try:
        db = getattr(context, "db", None) if context is not None else None
        if db is not None:
            from ...storage.kv import KVStore
            reg.bind_store(KVStore(db))
    except Exception as exc:  # noqa: BLE001 - persistence is an enhancement
        record_failure("sources.pump_registry_for_context.bind", exc)
    return reg


def pump_status(registry: Any) -> dict[str, Any]:
    """Registry view of the memecoin class: sources + health."""
    rows = []
    for desc in registry.by_kind("memecoin"):
        rows.append({
            "domain": desc.domain, "name": desc.name,
            "health": registry.health_for(desc),
            "origin": registry.origin_for(desc),
            "active": registry.is_active(desc),
            "stats": registry.stats_for(desc),
        })
    return {"sources": rows, "count": len(rows)}


# ── desk network config (ONE surface) ────────────────────────────────
#
# Every desk fetch resolves its user-agent and its per-source time
# budget HERE — never as a literal scattered through the fetch
# modules.  Both read the process settings (``tools.user_agent`` /
# ``tools.pumpdesk_source_budget_s``) when a settings load is
# available, and degrade to the desk defaults when it is not
# (context-free CLI runs, tests): absence of config is a default,
# never an exception.

#: The desk's historical UA — the fallback when no configured UA
#: exists, kept in exactly one place.
DEFAULT_USER_AGENT = "nomorals-pumpdesk"

#: Fallback per-source, per-cycle hard budget (seconds) — mirrors
#: the ToolSettings default; see that field for the sizing note.
DEFAULT_SOURCE_BUDGET_S = 45.0


def _tools_settings() -> Any:
    try:
        from ...core.config import get_settings
        return get_settings().tools
    except Exception:  # noqa: BLE001 - config absence degrades
        return None


def desk_user_agent() -> str:
    """The ONE user-agent every desk fetch sends: the configured
    ``tools.user_agent`` when settings provide one, else the desk
    default.  Never raises."""
    tools = _tools_settings()
    ua = str(getattr(tools, "user_agent", "") or "").strip()
    return ua or DEFAULT_USER_AGENT


def desk_source_budget_s() -> float:
    """The hard per-source time budget for one desk cycle
    (seconds): ``tools.pumpdesk_source_budget_s`` when configured,
    else the desk default.  Never raises; a non-positive or
    unreadable value falls back to the default rather than
    disabling the budget."""
    tools = _tools_settings()
    try:
        value = float(getattr(tools, "pumpdesk_source_budget_s", 0)
                      or 0)
    except (TypeError, ValueError):
        value = 0.0
    return value if value > 0 else DEFAULT_SOURCE_BUDGET_S


def pumpportal_api_key() -> str:
    """The PumpPortal API key, read per call from the provisioning
    env bridge (``PUMPPORTAL_API_KEY`` — the vault→env bridge
    writes it when the owner provisions ``/pump provision
    pumpportal``, exactly the Helius pattern).  "" when the desk
    runs keyless: the free subscriptions (new-token + migration)
    need no key; the metered trade streams do (PumpPortal, since
    May 2026).  The value is never logged or rendered anywhere."""
    return os.environ.get("PUMPPORTAL_API_KEY", "").strip()


# ── RPC endpoint selection ────────────────────────────────────────────

#: Keyless public RPC routes, in seed preference order:
#: (endpoint, registry domain, human source label).  More than one
#: route is the point — the public floor throttles PER METHOD
#: (getTokenLargestAccounts 429s while account reads pass), so a
#: single endpoint makes whole dossier fields hostages.  Routes
#: join the memecoin registry kind above and are health-graded
#: from real outcomes; rpc_routes() orders by that health.
PUBLIC_RPC_ENDPOINTS: tuple = (
    ("https://api.mainnet-beta.solana.com",
     "api.mainnet-beta.solana.com", "solana public RPC"),
    ("https://solana-rpc.publicnode.com",
     "solana-rpc.publicnode.com", "PublicNode public RPC"),
    ("https://solana.drpc.org",
     "solana.drpc.org", "dRPC public RPC"),
)


def rpc_routes(registry: Any = None) -> list[dict[str, str]]:
    """The desk's ordered RPC routes: [{endpoint, domain, source}].

    A provisioned Helius key (``HELIUS_API_KEY`` in env — the
    provisioning bridge writes it when the owner provisions
    ``/pump provision helius``) is ALWAYS first: it is the tier
    the owner paid for, without public throttles.  The keyless
    public routes follow, ordered by the ONE registry's health
    when a registry is at hand — active sources before stood-
    down/disabled/dead ones, seed order preserved within a tier
    (a stable sort), so a route that keeps failing sinks without
    bespoke failover code.  Read per call: a freshly provisioned
    key and fresh health both take effect with no restart."""
    routes: list[dict[str, str]] = []
    key = os.environ.get("HELIUS_API_KEY", "").strip()
    if key:
        routes.append({
            "endpoint": f"https://mainnet.helius-rpc.com/?api-key={key}",
            "domain": "mainnet.helius-rpc.com",
            "source": "helius (provisioned)"})
    public = [{"endpoint": ep, "domain": dom, "source": label}
              for ep, dom, label in PUBLIC_RPC_ENDPOINTS]
    if registry is not None:
        def _tier(route: dict[str, str]) -> int:
            try:
                desc = registry.for_domain(route["domain"])
                if desc is not None and not registry.is_active(desc):
                    return 1
            except Exception:  # noqa: BLE001 - health is advisory here
                pass
            return 0
        public.sort(key=_tier)
    routes.extend(public)
    return routes


def rpc_endpoint_from_env() -> tuple[str, str]:
    """(endpoint, source) — the desk's PRIMARY route: a
    provisioned Helius key upgrades the public RPC; absent one,
    the public floor serves.  Read per call so a freshly
    provisioned key takes effect with no restart (the
    provisioning bridge writes the env).  Multi-route reads go
    through rpc_routes() + chain.RoutedRPC; this stays for the
    single-endpoint consumers (the firehose's log backend)."""
    routes = rpc_routes()
    first = routes[0]
    return (first["endpoint"], first["source"])


# ── self-provisioning: the keyed tiers ───────────────────────────────

HELIUS_SERVICE = "helius"
PUMPPORTAL_SERVICE = "pumpportal"


def _validate_helius(token: str, http_get: Any) -> Any:
    """Helius' REST surface answers a balances read 200 for a live
    key, 401 for a dead one — validated live before anything is
    vaulted (the provisioning HttpGet contract is GET-only)."""
    from ...accounts.provisioning import ValidationResult
    probe = ("https://api.helius.xyz/v0/addresses/"
             "So11111111111111111111111111111111111111112/balances"
             f"?api-key={token}")
    try:
        status, _payload = http_get(probe, {})
    except Exception as exc:  # noqa: BLE001 - transport is inconclusive
        return ValidationResult("inconclusive",
                                f"validation request failed: {exc}")
    if status == 200:
        return ValidationResult("ok", "helius key accepted (balances read)")
    if status in (401, 403):
        return ValidationResult("rejected",
                                f"helius answered HTTP {status}")
    return ValidationResult("inconclusive",
                            f"helius answered HTTP {status}")


def ensure_provisioning() -> list[str]:
    """Register the desk's keyed tiers with the real
    self-provisioning registry (idempotent).  The owner provisions
    each like any other service: his own account, vault, env
    bridge — the desk reads the bridged env keys per call
    (``HELIUS_API_KEY`` in :func:`rpc_routes`,
    ``PUMPPORTAL_API_KEY`` in :func:`pumpportal_api_key`).

    Helius carries a LIVE validator (its REST surface answers a
    key check by GET).  PumpPortal deliberately carries NONE:
    PumpPortal exposes no GET-only key check — its keyed surfaces
    are the trade POST API and the metered data socket — so the
    framework's answer for it is an honest ``inconclusive``, never
    a fabricated ok; the key's truth is proven by the socket's
    subscription acks on first use."""
    from ...accounts.provisioning import (
        ProvisionService, _VALIDATORS, register_provision_service)
    _VALIDATORS.setdefault("helius_key", _validate_helius)
    register_provision_service(ProvisionService(
        id=HELIUS_SERVICE,
        display="Helius",
        aliases=("helius-rpc",),
        vault_service="helius",
        env_keys=("HELIUS_API_KEY",),
        login_url="https://dashboard.helius.dev/login",
        token_url="https://dashboard.helius.dev/api-keys",
        signup_url="https://dashboard.helius.dev/signup",
        validator="helius_key",
        flow="",
    ))
    register_provision_service(ProvisionService(
        id=PUMPPORTAL_SERVICE,
        display="PumpPortal",
        aliases=("pumpportal-data",),
        vault_service="pumpportal",
        env_keys=("PUMPPORTAL_API_KEY",),
        login_url="https://pumpportal.fun",
        token_url="",
        signup_url="https://pumpportal.fun",
        validator="",
        flow="",
    ))
    return [HELIUS_SERVICE, PUMPPORTAL_SERVICE]
