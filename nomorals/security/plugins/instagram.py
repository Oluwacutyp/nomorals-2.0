"""Instagram — web login; sessionid cookie is the proof."""
from __future__ import annotations

import json
import re
import time

from .interface import AuthResult, EnumerateResult, LoginResult, SessionData, WebPlugin

_CSRF = re.compile(r'"csrf_token":"([^"]+)"')


class InstagramPlugin(WebPlugin):
    name, domain = "instagram", "instagram.com"
    auth_cookies = ("sessionid",)

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        page = client.get("https://www.instagram.com/accounts/login/")
        token = (_CSRF.search(page.text or "") or [None, ""])[1]
        if not token:
            return done(AuthResult.ERROR, "no csrf token")
        resp = client.post(
            "https://www.instagram.com/api/v1/web/accounts/login/ajax/",
            data={"username": username, "enc_password": f"#PWD_INSTAGRAM_BROWSER:0:{int(time.time())}:{password}",
                  "queryParams": "{}", "optIntoOneTap": "false"},
            headers={"X-CSRFToken": token, "X-Requested-With": "XMLHttpRequest",
                     "Referer": "https://www.instagram.com/accounts/login/",
                     "Content-Type": "application/x-www-form-urlencoded"})
        try:
            data = json.loads(resp.text or "{}")
        except ValueError:
            return done(AuthResult.UNKNOWN, f"http {resp.status}")
        if data.get("authenticated"):
            return done(AuthResult.SUCCESS, "login successful",
                        SessionData(cookies=dict(client.cookie_dict()),
                                    user_agent=client.user_agent,
                                    tokens={"csrf": token}))
        if data.get("two_factor_required"):
            return done(AuthResult.MFA, "two factor required")
        if data.get("checkpoint_url"):
            return done(AuthResult.LOCKOUT, "checkpoint")
        msg = str(data.get("message") or "")
        if "password" in msg.lower() or data.get("user"):
            return done(AuthResult.FAILURE, msg or "invalid credentials")
        return done(AuthResult.UNKNOWN, msg or f"http {resp.status}")

    def enumerate(self, username, client):
        resp = client.get(f"https://www.instagram.com/{username}/",
                          headers={"X-IG-App-ID": "936619743392459"})
        if resp.status == 404:
            return EnumerateResult(username, False, 0.85, ["http 404"])
        if resp.ok and "og:title" in (resp.text or ""):
            return EnumerateResult(username, True, 0.8, ["profile rendered"])
        return EnumerateResult(username)
