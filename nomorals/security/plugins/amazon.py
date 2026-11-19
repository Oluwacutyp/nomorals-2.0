"""Amazon — /ap/signin with appActionToken. session-id is the proof."""
from __future__ import annotations

import re
import time

from .interface import AuthResult, LoginResult, SessionData, WebPlugin

_TOKEN = re.compile(r'name=["\'](appActionToken|openid\.return_to|prevRID|workflowState)["\'][^>]*value=["\']([^"\']*)', re.I)


class AmazonPlugin(WebPlugin):
    name, domain = "amazon", "amazon.com"
    auth_cookies = ("session-id", "at-main")

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        page = client.get("https://www.amazon.com/ap/signin"
                          "?openid.return_to=https://www.amazon.com/")
        html = page.text or ""
        fields = dict(_TOKEN.findall(html))
        if not fields.get("appActionToken"):
            return done(AuthResult.ERROR, "no appActionToken")
        data = {**fields, "email": username, "password": password,
                "create": "0", "metadata1": ""}
        resp = client.post("https://www.amazon.com/ap/signin", data=data)
        body = (resp.text or "").lower()
        if "session-id" in dict(client.cookie_dict()):
            return done(AuthResult.SUCCESS, "login successful",
                        SessionData(cookies=dict(client.cookie_dict()),
                                    user_agent=client.user_agent))
        if "otp" in body or "two-step" in body or "verification" in body:
            return done(AuthResult.MFA, "otp required")
        if "captcha" in body or "robot" in body:
            return done(AuthResult.CAPTCHA, "captcha")
        if "your password is incorrect" in body or "problem" in body:
            return done(AuthResult.FAILURE, "invalid credentials")
        return done(AuthResult.UNKNOWN, f"http {resp.status}")
