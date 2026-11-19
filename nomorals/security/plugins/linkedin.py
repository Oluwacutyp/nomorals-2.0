"""LinkedIn — session_key/session_password; li_at cookie is the proof."""
from __future__ import annotations

import re
import time

from .interface import AuthResult, LoginResult, SessionData, WebPlugin

_CSRF = re.compile(r'name=["\']csrfToken["\'][^>]*value=["\']([^"\']*)', re.I)


class LinkedInPlugin(WebPlugin):
    name, domain = "linkedin", "linkedin.com"
    auth_cookies = ("li_at",)

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        page = client.get("https://www.linkedin.com/login")
        token = (_CSRF.search(page.text or "") or [None, ""])[1]
        resp = client.post("https://www.linkedin.com/checkpoint/lg/login-submit",
                           data={"session_key": username,
                                 "session_password": password,
                                 "csrfToken": token},
                           headers={"Content-Type":
                                    "application/x-www-form-urlencoded",
                                    "Referer": "https://www.linkedin.com/login"})
        body = (resp.text or "").lower()
        if "li_at" in {c.lower() for c in dict(client.cookie_dict())}:
            return done(AuthResult.SUCCESS, "login successful",
                        SessionData(cookies=dict(client.cookie_dict()),
                                    user_agent=client.user_agent))
        if "pin" in body or "verification code" in body:
            return done(AuthResult.MFA, "pin required")
        if "captcha" in body or "security check" in body:
            return done(AuthResult.CAPTCHA, "security check")
        if "wrong" in body or "couldn't find" in body:
            return done(AuthResult.FAILURE, "invalid credentials")
        return done(AuthResult.UNKNOWN, f"http {resp.status}")
