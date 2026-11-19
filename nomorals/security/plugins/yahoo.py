"""Yahoo — two-step form flow. Success is a redirect into mail/my."""
from __future__ import annotations

import re
import time

from .interface import AuthResult, EnumerateResult, LoginResult, SessionData, WebPlugin

_FIELD = re.compile(r'<input[^>]+name=["\']([^"\']+)["\'][^>]*value=["\']([^"\']*)', re.I)


class YahooPlugin(WebPlugin):
    name, domain = "yahoo", "login.yahoo.com"
    auth_cookies = ("A3", "A1")

    def _form(self, html: str) -> tuple[str, dict[str, str]]:
        action = (re.search(r'<form[^>]+action=["\']([^"\']+)', html, re.I) or [None, ""])[1]
        return action, {k: v for k, v in _FIELD.findall(html)}

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        page = client.get("https://login.yahoo.com/?display=login")
        if not page.ok:
            return done(AuthResult.ERROR, f"http {page.status}")
        action, data = self._form(page.text or "")
        if not action:
            return done(AuthResult.ERROR, "no login form")
        data["username"] = username
        step = client.post(action, data=data)
        body = step.text or ""
        if "password" not in str(step.url) and "challenge" not in body.lower():
            if "A3" in dict(client.cookie_dict()):
                return done(AuthResult.SUCCESS, "single step", self._session(client))
            return done(AuthResult.UNKNOWN, "no password step")
        action, data = self._form(body)
        if not action:
            return done(AuthResult.ERROR, "no password form")
        data["password"] = password
        final = client.post(action, data=data)
        text, url = (final.text or "").lower(), str(final.url)
        if "mail.yahoo.com" in url or "my.yahoo.com" in url:
            return done(AuthResult.SUCCESS, "login successful", self._session(client))
        if "invalid password" in text or "wrong password" in text:
            return done(AuthResult.FAILURE, "invalid password")
        if "account is locked" in text or "locked" in text:
            return done(AuthResult.LOCKOUT, "account locked")
        if "verification code" in text or "verify" in text:
            return done(AuthResult.MFA, "verification required")
        if "captcha" in text:
            return done(AuthResult.CAPTCHA, "captcha")
        return done(AuthResult.UNKNOWN, f"http {final.status}")

    def _session(self, client) -> SessionData:
        return SessionData(cookies=dict(client.cookie_dict()),
                           user_agent=client.user_agent)

    def enumerate(self, username, client):
        resp = client.post("https://login.yahoo.com/account/module/create",
                           data={"validate": "yid", "yid": username},
                           headers={"X-Requested-With": "XMLHttpRequest"})
        text = (resp.text or "").lower()
        if "already taken" in text or "already exists" in text:
            return EnumerateResult(username, True, 0.9, ["username taken"])
        if resp.ok and text.strip():
            return EnumerateResult(username, False, 0.6, ["validation answered free"])
        return EnumerateResult(username)
