"""Session health registry + auth ladder.

* Health: per-site authenticated ping on schedule, background refresh
  before expiry, queryable "logged in as X on site Y, last verified Z,
  expires W". No task starts by logging in — it consults health first.
* Auth ladder, in order. Each rung is tried only if the previous failed,
  and the rung used is logged (attribution, never silent):
    1. password from vault (accounts/browser_login.py does the driving)
    2. TOTP from a vault secret (injected totp callable — never a stored code)
    3. email link/code, automatic via injected mail poller (Gmail/IMAP/temp_mail)
    4. SMS via injected sms poller (temp_sms/Twilio), else ask owner
    5. push / passkey via owner (native/biometric.py) — owner acts, never automated
    6. device-bound / hardware — owner acts, never automated
  Recovery material lives in the vault with the credential.
* Router: connector (OAuth-first) -> browser-as-fallback -> owner.
  OAuth-first means the browser is the fallback, not the default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["AuthLadder", "AuthResult", "SessionHealth", "SessionHealthRegistry"]

#: Auth ladder rungs, in order. The rung used is always logged.
RUNG_PASSWORD = "password_vault"
RUNG_TOTP = "totp_vault_secret"
RUNG_EMAIL = "email_link_code"
RUNG_SMS = "sms_code"
RUNG_PUSH_PASSKEY = "push_passkey_owner"
RUNG_DEVICE_BOUND = "device_bound_owner"
RUNG_OAUTH = "oauth_connector"

#: Rungs that MUST be performed by the owner, never automated.
OWNER_ONLY_RUNGS = frozenset({RUNG_PUSH_PASSKEY, RUNG_DEVICE_BOUND})


@dataclass
class SessionHealth:
    site: str
    username: str = ""
    logged_in: bool = False
    last_verified: float = 0.0
    expires_at: float | None = None
    rung_used: str = ""
    failures: int = 0

    def describe(self) -> str:
        verified = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_verified))
                    if self.last_verified else "never")
        expires = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.expires_at))
                   if self.expires_at else "unknown")
        who = self.username or "unknown user"
        state = "logged in" if self.logged_in else "logged out"
        return (f"{state} as {who} on {self.site}, "
                f"last verified {verified}, expires {expires}")

    def needs_refresh(self, *, skew_s: float = 300.0) -> bool:
        if not self.logged_in:
            return True
        if self.expires_at is None:
            return False
        return (self.expires_at - time.time()) <= skew_s

    def to_dict(self) -> dict[str, Any]:
        return {"site": self.site, "username": self.username,
                "logged_in": self.logged_in, "last_verified": self.last_verified,
                "expires_at": self.expires_at, "rung_used": self.rung_used,
                "failures": self.failures}


class SessionHealthRegistry:
    """Queryable per-site health. Backed by injected ping/refresh callables."""

    def __init__(self, *,
                 ping: Callable[[str], dict[str, Any]] | None = None,
                 refresh: Callable[[str], bool] | None = None) -> None:
        self._ping = ping
        self._refresh = refresh
        self._health: dict[str, SessionHealth] = {}

    def register(self, health: SessionHealth) -> None:
        self._health[health.site] = health

    def get(self, site: str) -> SessionHealth | None:
        return self._health.get(site)

    def describe(self, site: str) -> str:
        health = self._health.get(site)
        if health is None:
            return f"no session health recorded for {site}"
        return health.describe()

    def check(self, site: str) -> SessionHealth:
        """Ping the site (injected) and record the result."""
        health = self._health.get(site) or SessionHealth(site=site)
        if self._ping is None:
            health.last_verified = time.time()
            self._health[site] = health
            return health
        result = self._ping(site)
        health.logged_in = bool(result.get("logged_in", False))
        health.username = str(result.get("username", health.username or ""))
        health.last_verified = time.time()
        if result.get("expires_at") is not None:
            health.expires_at = float(result["expires_at"])
        if not health.logged_in:
            health.failures += 1
        self._health[site] = health
        return health

    def ensure(self, site: str, *, auth: "AuthLadder | None" = None) -> SessionHealth:
        """Health first, refresh before expiry, auth only when needed."""
        health = self.check(site)
        if health.logged_in and not health.needs_refresh():
            return health
        if health.logged_in and health.needs_refresh() and self._refresh is not None:
            if self._refresh(site):
                health.last_verified = time.time()
                health.failures = 0
                self._health[site] = health
                return health
        if auth is not None:
            result = auth.authenticate(site)
            health.logged_in = result.ok
            health.rung_used = result.rung_used
            health.username = result.username or health.username
            health.last_verified = time.time()
            if result.expires_at is not None:
                health.expires_at = result.expires_at
            if not result.ok:
                health.failures += 1
            self._health[site] = health
        return health


@dataclass
class AuthResult:
    ok: bool
    rung_used: str = ""
    username: str = ""
    expires_at: float | None = None
    detail: str = ""
    needs_owner: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "rung_used": self.rung_used,
                "username": self.username, "expires_at": self.expires_at,
                "detail": self.detail, "needs_owner": self.needs_owner}


class AuthLadder:
    """Auth ladder with injected providers. Owner-only rungs never automate."""

    def __init__(self, *,
                 password_login: Callable[[str], AuthResult] | None = None,
                 totp: Callable[[str], str | None] | None = None,
                 mail_poll: Callable[[str, float], str | None] | None = None,
                 sms_poll: Callable[[str, float], str | None] | None = None,
                 ask_owner: Callable[[str, str], AuthResult] | None = None,
                 oauth: Callable[[str], AuthResult] | None = None) -> None:
        self._password_login = password_login
        self._totp = totp
        self._mail_poll = mail_poll
        self._sms_poll = sms_poll
        self._ask_owner = ask_owner
        self._oauth = oauth
        self.rung_log: list[dict[str, Any]] = []

    def _log(self, site: str, rung: str, ok: bool, detail: str = "") -> None:
        self.rung_log.append({"site": site, "rung": rung, "ok": ok,
                              "detail": detail, "at": time.time()})

    def authenticate(self, site: str) -> AuthResult:
        """OAuth-first, then browser ladder, then owner. Rung used is logged."""
        if self._oauth is not None:
            result = self._oauth(site)
            self._log(site, RUNG_OAUTH, result.ok, result.detail)
            if result.ok:
                return result
        if self._password_login is not None:
            result = self._password_login(site)
            self._log(site, RUNG_PASSWORD, result.ok, result.detail)
            if result.ok:
                return result
        if self._totp is not None:
            code = self._totp(site)
            self._log(site, RUNG_TOTP, code is not None,
                      "TOTP generated from vault secret" if code else "no TOTP secret")
            if code is not None:
                return AuthResult(ok=True, rung_used=RUNG_TOTP,
                                  detail="TOTP from vault secret")
        if self._mail_poll is not None:
            code = self._mail_poll(site, 60.0)
            self._log(site, RUNG_EMAIL, code is not None,
                      "email code polled" if code else "no email code")
            if code is not None:
                return AuthResult(ok=True, rung_used=RUNG_EMAIL,
                                  detail="email link/code via mail poll")
        if self._sms_poll is not None:
            code = self._sms_poll(site, 60.0)
            self._log(site, RUNG_SMS, code is not None,
                      "SMS code polled" if code else "no SMS code")
            if code is not None:
                return AuthResult(ok=True, rung_used=RUNG_SMS,
                                  detail="SMS code via sms poll")
        # Owner-only rungs: never automated. Ask, or report need-owner.
        if self._ask_owner is not None:
            result = self._ask_owner(site, RUNG_PUSH_PASSKEY)
            self._log(site, RUNG_PUSH_PASSKEY, result.ok, result.detail)
            return result
        self._log(site, RUNG_PUSH_PASSKEY, False,
                  "push/passkey needs the owner — never automated")
        return AuthResult(ok=False, rung_used=RUNG_PUSH_PASSKEY,
                          detail="push/passkey/device-bound needs the owner; "
                                 "never automated", needs_owner=True)
