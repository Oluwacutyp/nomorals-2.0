"""Proxy trust tiers + the tier gate (PROXY_SOURCES).

A proxy is a stranger's machine sitting in the middle of the traffic:
it sees the destination, the timing, and — for plain-HTTP or a
TLS-intercepting proxy — the bytes.  That is an engineering fact, not
a policy opinion, and it is enforced here as code, not as a promise:

* ``free`` — scraped public proxies, no operator, no contract.  Usable
  ONLY for anonymous, unauthenticated, low-stakes fetches (public
  news/music reads, OSINT, price checks, catalogue reads).  NEVER
  eligible for any request carrying credentials, cookies, session
  tokens, or vault material: the operator can hijack the session, MITM
  the traffic, or run the endpoint as a honeypot.
* ``paid`` — a proxy the owner pays for under a provider contract
  (datacenter / residential / mobile plan).  Eligible for
  credential-carrying traffic.
* ``direct`` — no proxy at all.  Eligible for everything; nothing
  third-party sees the traffic.

The gate mirrors the :mod:`nomorals.core.policy` pattern: a
``TierDecision`` value (``allowed`` / ``reason`` / ``raise_if_denied``)
plus a typed ``TierGateRefusal`` (a ``CapabilityDenied``) that callers
raise.  It lives in ``core`` so both the L5 pool
(``connectors/proxypool.py``) and the L4 browser (``browser/``) can
enforce the same rule without a layering violation.

Trust order: ``direct`` > ``paid`` > ``free``.  A ``require_tier``
demand is satisfied by that tier or anything more trusted — e.g.
``require_tier="paid"`` accepts ``paid`` and ``direct`` (no proxy, no
third party), never ``free``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import CapabilityDenied
from .logging_setup import get_logger

__all__ = [
    "FREE",
    "PAID",
    "DIRECT",
    "TIERS",
    "TIER_TRUST",
    "ALLOWED_USES",
    "FREE_TIER_ALTERNATIVES",
    "PROXY_ETIQUETTE",
    "TierDecision",
    "TierGateRefusal",
    "TierGate",
    "normalize_tier",
    "tier_satisfies",
]

_log = get_logger(__name__)

FREE = "free"
PAID = "paid"
DIRECT = "direct"

TIERS: tuple[str, ...] = (FREE, PAID, DIRECT)

#: Higher = more trusted.  ``direct`` tops the order: no intermediary.
TIER_TRUST: dict[str, int] = {FREE: 0, PAID: 1, DIRECT: 2}

#: What each tier may carry.  Free = anonymous only, stated positively
#: so callers can explain a refusal without inventing policy.
ALLOWED_USES: dict[str, tuple[str, ...]] = {
    FREE: (
        "anonymous unauthenticated low-stakes fetches only",
        "public news / music reads, OSINT, price checks, catalogue reads",
    ),
    PAID: (
        "anonymous fetches",
        "credential-carrying traffic (logins, cookies, sessions, vault material)",
    ),
    DIRECT: (
        "everything — no third-party intermediary sees the traffic",
    ),
}

#: Free-tier alternatives ladder: when a free proxy is not good enough,
#: climb this instead of pushing credential traffic through it.  Ordered
#: cheapest / least-trust-needed first.
FREE_TIER_ALTERNATIVES: tuple[dict[str, str], ...] = (
    {
        "name": "Cloudflare WARP",
        "tier": DIRECT,
        "note": "One clean IP from a reputable operator; no proxy list, "
                "no stranger in the middle. Best first step off free proxies.",
    },
    {
        "name": "Tor (SOCKS 127.0.0.1:9050)",
        "tier": FREE,
        "note": "Slow, many sites block exit nodes, and exit operators see "
                "plaintext — NEVER logins. Anonymity tool, not a credential "
                "channel.",
    },
    {
        "name": "Provider free tiers",
        "tier": PAID,
        "note": "Small residential allowance from a real provider, "
                "geo-coherent, under a contract — the cheapest paid-shaped "
                "option.",
    },
    {
        "name": "Own VPS (~$5 datacenter)",
        "tier": PAID,
        "note": "Your own machine, your own keys. Flagged as datacenter IP "
                "by many sites, but nobody else reads the traffic.",
    },
    {
        "name": "Mobile IP via own device",
        "tier": PAID,
        "note": "Residential, one IP, yours. Best geo-coherence for "
                "account work when a provider plan is not available.",
    },
)

#: Etiquette is practical, not moral: free proxies are shared, fragile,
#: and run by strangers — hammering them burns the source for everyone
#: and gets the pool blocked.
PROXY_ETIQUETTE: tuple[str, ...] = (
    "rate-limit through free proxies — never relay aggressive bursts",
    "never relay aggressive traffic (floods, brute force, mass scraping)",
    "don't send anything you'd mind a stranger reading",
    "rotate provenance, not just proxies — one IP per identity/session",
)


def normalize_tier(tier: Any) -> str:
    """``free`` / ``paid`` / ``direct`` (case/whitespace tolerant).

    ``""`` / ``None`` means 'not specified' → ``free``: an unlabelled
    proxy is untrusted until proven otherwise.  Unknown values fail
    closed with ``ValueError`` — never silently treated as trusted.
    """
    if tier is None:
        return FREE
    value = str(tier).strip().lower()
    if not value:
        return FREE
    if value in TIER_TRUST:
        return value
    raise ValueError(
        f"unknown proxy tier {tier!r}: expected one of {', '.join(TIERS)}"
    )


def tier_satisfies(tier: str, require_tier: str | None) -> bool:
    """True when ``tier`` meets (or exceeds) the ``require_tier`` demand."""
    if require_tier is None:
        return True
    return TIER_TRUST[normalize_tier(tier)] >= TIER_TRUST[normalize_tier(require_tier)]


class TierGateRefusal(CapabilityDenied):
    """Typed refusal: this tier may not carry this request.

    A ``CapabilityDenied`` (``code = "policy.denied"``) so policy audit
    tooling sees it, with the tier + credential facts on the instance
    for callers that need to explain or retry with a better tier.
    """

    code = "proxy.tier.denied"

    def __init__(self, message: str, *, tier: str = "",
                 carries_credentials: bool = False,
                 require_tier: str | None = None,
                 **kw: Any) -> None:
        super().__init__(message, capability="proxy.use", **kw)
        self.tier = tier
        self.carries_credentials = carries_credentials
        self.require_tier = require_tier
        self.details.setdefault("tier", tier)
        self.details.setdefault("carries_credentials", carries_credentials)
        if require_tier is not None:
            self.details.setdefault("require_tier", require_tier)


@dataclass
class TierDecision:
    """Result of a tier-gate check — mirrors ``core.policy.PolicyDecision``."""

    allowed: bool
    reason: str = ""
    tier: str = ""
    capability: str = "proxy.use"
    carries_credentials: bool = False
    require_tier: str | None = None

    def __bool__(self) -> bool:
        return self.allowed

    def raise_if_denied(self) -> None:
        if not self.allowed:
            raise TierGateRefusal(
                self.reason or "proxy tier denied",
                tier=self.tier,
                carries_credentials=self.carries_credentials,
                require_tier=self.require_tier,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "tier": self.tier,
            "capability": self.capability,
            "carries_credentials": self.carries_credentials,
            "require_tier": self.require_tier,
        }


class TierGate:
    """The enforceable gate: tier + credential facts → allow / refuse."""

    @staticmethod
    def check(tier: str, *, carries_credentials: bool = False,
              require_tier: str | None = None) -> TierDecision:
        """Check without raising.  ``check`` never trusts by default."""
        tier = normalize_tier(tier)
        if carries_credentials and tier == FREE:
            reason = (
                "REFUSED: free proxy may not carry credentialed traffic "
                "(credentials, cookies, session tokens, vault material) — "
                "the free-proxy operator can hijack the session, MITM the "
                "traffic, or run the endpoint as a honeypot. Free proxies "
                "are anonymous unauthenticated low-stakes fetches ONLY. "
                "Use a paid proxy or direct traffic instead."
            )
            _log.warning("proxy tier gate refused: tier=%s credentials=True", tier)
            return TierDecision(
                allowed=False, reason=reason, tier=tier,
                carries_credentials=True, require_tier=require_tier,
            )
        if require_tier is not None and not tier_satisfies(tier, require_tier):
            reason = (
                f"REFUSED: proxy tier {tier!r} does not satisfy "
                f"require_tier={require_tier!r} "
                f"(trust order: direct > paid > free)"
            )
            _log.warning("proxy tier gate refused: tier=%s require_tier=%s",
                         tier, require_tier)
            return TierDecision(
                allowed=False, reason=reason, tier=tier,
                carries_credentials=carries_credentials,
                require_tier=require_tier,
            )
        return TierDecision(
            allowed=True, tier=tier,
            carries_credentials=carries_credentials,
            require_tier=require_tier,
        )

    @classmethod
    def require(cls, tier: str, *, carries_credentials: bool = False,
                require_tier: str | None = None) -> TierDecision:
        """Check and raise :class:`TierGateRefusal` on denial."""
        decision = cls.check(tier, carries_credentials=carries_credentials,
                             require_tier=require_tier)
        decision.raise_if_denied()
        return decision
