"""Desk sources: betting joins the shared source registry, and the
desk's API-key needs join the real self-provisioning registry.

Two existing organs, extended — never paralleled:

* ``media/source_registry.SourceRegistry`` is the ONE source of truth
  every class grows through (downloads #86, music #88, books #89).
  The desk registers its seed sources under kind ``betting`` the
  same way: seeds are the floor, outcomes move them, candidates
  enter untrusted.  Book domains coincide with the collector
  providers in :mod:`.collectors`, so a fetch outcome and a registry
  health record are about the same host.
* ``accounts/provisioning.PROVISION_REGISTRY`` is how Devon gets
  herself API keys (drive signup/login on the owner's own account,
  validate, vault, bridge to env).  The desk's data keys — the Odds
  API (``ODDS_API_KEY``, already read by
  ``agents/sports_bet.TheOddsApiFetcher``) and football-data.org
  (``FOOTBALL_DATA_API_KEY``) — are registered here as services with
  real validators, so ``provision odds-api`` works through the same
  driver as every other service instead of the desk growing its own
  key plumbing.

Betting BOOK accounts are deliberately absent: books are the owner's
own accounts (the desk studies them through his sessions); they are
never provisioned, never opened under generated identities.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BETTING_SEEDS", "ensure_betting_sources", "betting_status",
    "ensure_provisioning", "ODDS_API_SERVICE", "FOOTBALL_DATA_SERVICE",
]


# ── source-registry seeds ──────────────────────────────────────────────

#: (nation, domain, name, auth_needed, notes)
BETTING_SEEDS: tuple = (
    ("US", "stake.com", "Stake", True,
     "Stake Originals crash family + sportsbook; history behind the "
     "owner's session."),
    ("NG", "sportybet.com", "SportyBet", True,
     "Spribe Aviator (NG). Public site bot-walls datacenter networks."),
    ("NG", "msport.com", "MSport", True,
     "Spribe Aviator (NG); session-shaped history."),
    ("NG", "football.com", "Football.com", True,
     "Aviator + crash family (NG); session-shaped history."),
    ("GLOBAL", "bustabit.com", "Bustabit", False,
     "The original crash game; provably-fair hash chain, published "
     "salts. History rides its app — exported logs import directly."),
    ("GLOBAL", "the-odds-api.com", "The Odds API", True,
     "Cross-book odds API (keyed, free tier) — the desk's odds feed; "
     "self-provisioned via accounts/provisioning."),
    ("GLOBAL", "football-data.org", "football-data.org", True,
     "Fixtures/results API (keyed, free tier) — the analyst's "
     "history feed; self-provisioned via accounts/provisioning."),
    ("GLOBAL", "site.api.espn.com", "ESPN scoreboard", False,
     "Keyless fixtures+odds feed the sports analyst already reads."),
)


def ensure_betting_sources(registry: Any) -> list[dict[str, Any]]:
    """Register the betting seed sources on the shared registry.

    Idempotent: a seed already present (by domain) is reported, not
    duplicated.  Returns one status dict per seed.
    """
    from ...media.source_registry import (
        Licence, SourceDescriptor, TrustTier)
    out: list[dict[str, Any]] = []
    for nation, domain, name, auth, notes in BETTING_SEEDS:
        existing = registry.for_domain(domain)
        if existing is not None:
            out.append({"domain": domain, "already_registered": True})
            continue
        desc = SourceDescriptor(
            nation=nation, language="en", domain=domain, name=name,
            kinds=("betting",), licence=Licence.UNKNOWN,
            trust_tier=TrustTier.UNVERIFIED, auth_needed=auth,
            notes=notes)
        added = registry.register(desc)
        if added:
            try:
                registry._persist()
            except Exception:  # noqa: BLE001 - persistence is best-effort
                pass
        out.append({"domain": domain, "registered": bool(added)})
    return out


def betting_status(registry: Any) -> dict[str, Any]:
    """Registry view of the betting class: sources + health + origin."""
    rows = []
    for desc in registry.by_kind("betting"):
        rows.append({
            "domain": desc.domain, "name": desc.name,
            "health": registry.health_for(desc),
            "origin": registry.origin_for(desc),
            "active": registry.is_active(desc),
            "stats": registry.stats_for(desc),
        })
    return {"sources": rows, "count": len(rows)}


# ── self-provisioning: the desk's API keys ─────────────────────────────

ODDS_API_SERVICE = "odds-api"
FOOTBALL_DATA_SERVICE = "football-data"


def _validate_odds_api(token: str, http_get: Any) -> Any:
    """The Odds API: 200 for a live key, 401 for a dead one."""
    from ...accounts.provisioning import ValidationResult
    try:
        status, _body = http_get(
            f"https://api.the-odds-api.com/v4/sports/?apiKey={token}", {})
    except Exception as exc:  # noqa: BLE001 - transport is inconclusive
        return ValidationResult("inconclusive",
                                f"validation request failed: {exc}")
    if status == 200:
        return ValidationResult("ok", "odds api key accepted")
    if status in (401, 403):
        return ValidationResult("rejected", f"odds api answered HTTP {status}")
    return ValidationResult("inconclusive",
                            f"odds api answered HTTP {status}")


def _validate_football_data(token: str, http_get: Any) -> Any:
    """football-data.org: 200 for a live key, 403 for a dead one."""
    from ...accounts.provisioning import ValidationResult
    try:
        status, _body = http_get(
            "https://api.football-data.org/v4/competitions/",
            {"X-Auth-Token": token})
    except Exception as exc:  # noqa: BLE001 - transport is inconclusive
        return ValidationResult("inconclusive",
                                f"validation request failed: {exc}")
    if status == 200:
        return ValidationResult("ok", "football-data key accepted")
    if status in (401, 403):
        return ValidationResult("rejected",
                                f"football-data answered HTTP {status}")
    return ValidationResult("inconclusive",
                            f"football-data answered HTTP {status}")


def ensure_provisioning() -> list[str]:
    """Register the desk's keyed data services with the REAL
    self-provisioning registry (idempotent — later registrations win,
    and re-registering the same entry is a no-op in effect).

    Returns the service ids ensured.  The driver can then provision
    them like any other service: signup/login on the owner's own
    account, key creation page, validation BEFORE the vault write
    counts, vault->env bridge (``ODDS_API_KEY`` /
    ``FOOTBALL_DATA_API_KEY`` are read per-call by the analyst's
    fetchers, so a provisioned key takes effect with no restart).
    """
    from ...accounts.provisioning import (
        ProvisionService, _VALIDATORS, register_provision_service)

    _VALIDATORS.setdefault("odds_api_key", _validate_odds_api)
    _VALIDATORS.setdefault("football_data_key", _validate_football_data)

    register_provision_service(ProvisionService(
        id=ODDS_API_SERVICE,
        display="The Odds API",
        aliases=("oddsapi", "the-odds-api"),
        vault_service="the-odds-api",
        env_keys=("ODDS_API_KEY",),
        login_url="https://the-odds-api.com/login/",
        token_url="https://the-odds-api.com/account/",
        signup_url="https://the-odds-api.com/#get-access",
        validator="odds_api_key",
        flow="",
    ))
    register_provision_service(ProvisionService(
        id=FOOTBALL_DATA_SERVICE,
        display="football-data.org",
        aliases=("footballdata",),
        vault_service="football-data",
        env_keys=("FOOTBALL_DATA_API_KEY",),
        login_url="https://www.football-data.org/client/login",
        token_url="https://www.football-data.org/client/home",
        signup_url="https://www.football-data.org/client/register",
        validator="football_data_key",
        flow="",
    ))
    return [ODDS_API_SERVICE, FOOTBALL_DATA_SERVICE]
