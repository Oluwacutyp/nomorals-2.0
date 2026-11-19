"""TikTok — web login; sessionid cookie is the proof."""
from __future__ import annotations

import json
import time

from .interface import AuthResult, LoginResult, SessionData, WebPlugin


class TikTokPlugin(WebPlugin):
    name, domain = "tiktok", "tiktok.com"
    auth_cookies = ("sessionid", "sid_tt")

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        client.get("https://www.tiktok.com/login/phone-or-email/email")
        resp = client.post(
            "https://www.tiktok.com/api/v1/auth/login/",
            data={"username": username, "password": password,
                  "accountParams": "email", "aid": "1988"},
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Referer": "https://www.tiktok.com/login/phone-or-email/email"})
        try:
            data = json.loads(resp.text or "{}")
        except ValueError:
            data = {}
        if "sessionid" in {c.lower() for c in dict(client.cookie_dict())}:
            return done(AuthResult.SUCCESS, "login successful",
                        SessionData(cookies=dict(client.cookie_dict()),
                                    user_agent=client.user_agent))
        desc = str((data.get("data") or {}).get("description") or "")
        if "verification" in desc.lower() or data.get("data", {}).get("captcha"):
            return done(AuthResult.CAPTCHA, "verification required")
        if desc:
            return done(AuthResult.FAILURE, desc[:80])
        return done(AuthResult.UNKNOWN, f"http {resp.status}")
