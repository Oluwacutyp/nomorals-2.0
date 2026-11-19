"""Reddit — /login with op; reddit_session cookie is the proof."""
from __future__ import annotations

import re
import time

from .interface import AuthResult, LoginResult, SessionData, WebPlugin

_UH = re.compile(r'name=["\']uh["\'][^>]*value=["\']([^"\']*)', re.I)


class RedditPlugin(WebPlugin):
    name, domain = "reddit", "reddit.com"
    auth_cookies = ("reddit_session",)

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        page = client.get("https://www.reddit.com/login/")
        uh = (_UH.search(page.text or "") or [None, ""])[1]
        resp = client.post("https://www.reddit.com/login",
                           data={"user": username, "passwd": password,
                                 "uh": uh, "op": "login-main",
                                 "dest": "https://www.reddit.com/"},
                           headers={"Content-Type":
                                    "application/x-www-form-urlencoded"})
        body = (resp.text or "").lower()
        if "reddit_session" in {c.lower() for c in dict(client.cookie_dict())}:
            return done(AuthResult.SUCCESS, "login successful",
                        SessionData(cookies=dict(client.cookie_dict()),
                                    user_agent=client.user_agent))
        if "incorrect" in body or "invalid" in body:
            return done(AuthResult.FAILURE, "invalid credentials")
        if "captcha" in body or "blocked" in body:
            return done(AuthResult.CAPTCHA, "captcha")
        if "2fa" in body or "verification" in body:
            return done(AuthResult.MFA, "2fa")
        return done(AuthResult.UNKNOWN, f"http {resp.status}")
