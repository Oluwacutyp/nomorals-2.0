"""BotWall: typed anti-bot errors + the proxy router for the browser path.

``BotWallError`` promotes ``browser/errors.py`` to a first-class error
carrying the vendor (cloudflare / datadome / akamai / perimeterx / ...).
It is raised explicitly and never silently degraded into a retry.

Proxy routing (identity continuity, NOT evasion):

* ONE persistent profile per site, ONE consistent fingerprint, proxy geo
  matching the profile locale, real history, human pacing.
* NO fingerprint rotation. Rotating fingerprints for authenticated work
  is forbidden — it is a liability, not a feature. The proxy pool's
  rotation call is deliberately never used here.
* A poisoned profile (failed challenge) is discarded one-time and binned,
  never reused.
* Escalation rungs, each typed + logged + attributed:
  direct -> real profile -> profile + residential proxy ->
  different browser engine -> owner handoff.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.proxy_tiers import TierGate, normalize_tier
from .errors import BrowserBotDetectedError

_log = get_logger(__name__)

__all__ = ["BotWallError", "ProxyRouter", "RUNG_DIRECT", "RUNG_PROFILE",
           "RUNG_PROFILE_PROXY", "RUNG_ENGINE", "RUNG_OWNER",
           "classify_botwall", "vendor_from_detection"]

RUNG_DIRECT = "direct"
RUNG_PROFILE = "real_profile"
RUNG_PROFILE_PROXY = "profile_residential_proxy"
RUNG_ENGINE = "different_engine"
RUNG_OWNER = "owner_handoff"

#: detection -> vendor mapping (mirrors errors.BrowserBotDetectedError).
_DETECTION_VENDOR: dict[str, str] = {
    "cloudflare-challenge": "cloudflare",
    "turnstile": "cloudflare",
    "perimeterx": "perimeterx",
    "datadome": "datadome",
    "akamai": "akamai",
    "imperva": "imperva",
    "kasada": "kasada",
    "f5-shape": "f5",
    "aws-waf": "aws_waf",
    "arkose": "arkose",
    "geetest": "geetest",
    "captcha-wall": "captcha_wall",
    "rate-limit": "rate_limit",
    "forbidden": "forbidden",
}


def vendor_from_detection(detection: str) -> str:
    return _DETECTION_VENDOR.get((detection or "").lower(), "unknown")


def classify_botwall(exc: Exception, *, url: str = "") -> "BotWallError":
    """Promote any bot-detection signal to a first-class BotWallError."""
    if isinstance(exc, BotWallError):
        return exc
    detection = getattr(exc, "detection", "") or ""
    vendor = vendor_from_detection(detection)
    return BotWallError(str(exc), vendor=vendor, detection=detection, url=url)


class BotWallError(BrowserBotDetectedError):
    """A bot wall, with the vendor named. Raised explicitly, never degraded."""

    kind = "bot_wall"

    def __init__(self, message: str, *, vendor: str = "unknown",
                 detection: str = "", url: str = "",
                 next_steps: list[str] | tuple[str, ...] = (),
                 evidence: dict[str, str] | None = None) -> None:
        super().__init__(message, next_steps=next_steps or (
            "one persistent profile, one consistent fingerprint — "
            "do not rotate",
            "escalate rung: direct -> real profile -> profile + "
            "residential proxy -> different engine -> owner handoff",
            "if the challenge persists, hand off to the owner",
        ), evidence=evidence, url=url)
        self.vendor = vendor
        self.detection = detection

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update({"vendor": self.vendor, "detection": self.detection})
        return payload


@dataclass
class ProxyRoute:
    rung: str
    profile_name: str
    proxy_url: str = ""
    geo: str = ""
    discarded_profile: str = ""
    #: trust tier of the route's proxy ("" = direct, no proxy).
    tier: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"rung": self.rung, "profile_name": self.profile_name,
                "proxy_url": ("***" if self.proxy_url else ""),
                "geo": self.geo, "discarded_profile": self.discarded_profile,
                "tier": self.tier or "direct"}


class ProxyRouter:
    """One persistent profile per site; escalation is by rung, not rotation.

    ``proxy_pool`` is duck-typed (the real connectors/proxypool.py is L5
    and cannot be imported from browser/ L4): it needs ``select`` /
    ``sticky_acquire`` / ``quarantine``-shaped callables, injected.
    ``proxy_for`` is the injected geo-matched proxy picker.
    ``tier_for`` maps a picked proxy URL to its trust tier (``free`` /
    ``paid`` / ``direct``) — the tier gate refuses ``free`` proxies for
    credential-carrying routes.  Without ``tier_for`` a picked proxy is
    treated as ``free`` (fail closed) when credentials are involved.
    """

    def __init__(self, *,
                 proxy_for: Callable[[str, str], str | None] | None = None,
                 quarantine: Callable[[str, str], None] | None = None,
                 tier_for: Callable[[str], str] | None = None) -> None:
        self._proxy_for = proxy_for
        self._quarantine = quarantine
        self._tier_for = tier_for
        self._profiles: dict[str, str] = {}  # site -> profile name (sticky)
        self._generations: dict[str, int] = {}  # site -> generation counter
        self._poisoned: set[str] = set()
        self._rung_log: list[dict[str, Any]] = []

    def profile_for(self, site: str) -> str:
        """Sticky profile name for a site — created once, never rotated.

        After a poisoned profile is binned, the replacement gets a new
        generation suffix, so the discarded name is never reused.
        """
        if site not in self._profiles:
            gen = self._generations.get(site, 0)
            suffix = f"-{gen}" if gen else ""
            self._profiles[site] = f"devon-{site}{suffix}"
        return self._profiles[site]

    def discard_poisoned(self, site: str, *, reason: str) -> str:
        """Bin a poisoned profile one-time. It is never reused."""
        old = self._profiles.pop(site, "")
        if old:
            self._poisoned.add(old)
            self._generations[site] = self._generations.get(site, 0) + 1
            if self._quarantine is not None:
                self._quarantine(site, reason)
        _log.warning("browser proxy router: discarded poisoned profile %r "
                     "for %s: %s", old, site, reason)
        return old

    def route(self, site: str, *, rung: str = RUNG_DIRECT,
              locale: str = "", carries_credentials: bool = False) -> ProxyRoute:
        """Pick the route for a rung. No fingerprint rotation at any rung.

        Tier gate: at the proxy rung, a picked proxy's tier is checked
        before the route is returned.  ``carries_credentials=True``
        (login flows, cookie-bearing sessions) refuses a ``free`` proxy
        with a typed ``TierGateRefusal`` — free tier is never on an
        authenticated path.  Anonymous routes may use a free proxy.
        """
        profile = self.profile_for(site)
        self._rung_log.append({"site": site, "rung": rung,
                               "profile": profile, "at": time.time()})
        if rung == RUNG_PROFILE_PROXY:
            proxy = self._proxy_for(site, locale) if self._proxy_for else None
            if not proxy:
                return ProxyRoute(rung=rung, profile_name=profile,
                                  proxy_url="", geo=locale, tier="direct")
            tier = (normalize_tier(self._tier_for(proxy))
                    if self._tier_for is not None else "free")
            TierGate.require(tier, carries_credentials=carries_credentials)
            return ProxyRoute(rung=rung, profile_name=profile,
                              proxy_url=proxy, geo=locale, tier=tier)
        return ProxyRoute(rung=rung, profile_name=profile, geo=locale,
                          tier="direct")

    def escalate(self, site: str, *, from_rung: str, locale: str = "",
                 reason: str = "", carries_credentials: bool = False) -> ProxyRoute:
        """Next rung after a failure. Poisoned profiles are binned, not retried."""
        order = [RUNG_DIRECT, RUNG_PROFILE, RUNG_PROFILE_PROXY,
                 RUNG_ENGINE, RUNG_OWNER]
        try:
            nxt = order[order.index(from_rung) + 1]
        except (ValueError, IndexError):
            nxt = RUNG_OWNER
        if from_rung in (RUNG_PROFILE, RUNG_PROFILE_PROXY) and reason:
            discarded = self.discard_poisoned(site, reason=reason)
            route = self.route(site, rung=nxt, locale=locale,
                               carries_credentials=carries_credentials)
            route.discarded_profile = discarded
            return route
        return self.route(site, rung=nxt, locale=locale,
                          carries_credentials=carries_credentials)

    @property
    def rung_log(self) -> list[dict[str, Any]]:
        return list(self._rung_log)

    @property
    def poisoned(self) -> set[str]:
        return set(self._poisoned)
