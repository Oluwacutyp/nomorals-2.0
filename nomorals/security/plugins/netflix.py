"""Netflix — login form; NetflixId cookie is the proof."""
from __future__ import annotations

import json
import re
import time

from .interface import AuthResult, LoginResult, SessionData, WebPlugin

_AUTH = re.compile(r'"authURL"\s*:\s*"([^"]+)"')


class NetflixPlugin(WebPlugin):
    name, domain = "netflix", "netflix.com"
    auth_cookies = ("NetflixId",)

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        page = client.get("https://www.netflix.com/login")
        auth = (_AUTH.search(page.text or "") or [None, ""])[1]
        data = {"userLoginId": username, "password": password,
                "rememberMe": "true"}
        if auth:
            data["authURL"] = auth
        resp = client.post("https://www.netflix.com/login", data=data,
                           headers={"Content-Type":
                                    "application/x-www-form-urlencoded"})
        body = (resp.text or "").lower()
        if "netflixid" in {c.lower() for c in dict(client.cookie_dict())}:
            return done(AuthResult.SUCCESS, "login successful",
                        SessionData(cookies=dict(client.cookie_dict()),
                                    user_agent=client.user_agent))
        if '"errorcode"' in body:
            try:
                code = json.loads(resp.text or "{}").get("errorCode")
            except ValueError:
                code = None
            if code in ("INVALID_PASSWORD", "INVALID_LOGIN"):
                return done(AuthResult.FAILURE, str(code))
            if code in ("MEMBER_ACCOUNT_NOT_FOUND", "USER_SUSPENDED"):
                return done(AuthResult.LOCKOUT, str(code))
        if "captcha" in body:
            return done(AuthResult.CAPTCHA, "captcha")
        return done(AuthResult.UNKNOWN, f"http {resp.status}")
