"""eBay — SignIn with srt token. ebay cookie is the proof."""
from __future__ import annotations

import re
import time

from .interface import AuthResult, LoginResult, SessionData, WebPlugin

_SRT = re.compile(r'name=["\']srt["\'][^>]*value=["\']([^"\']*)', re.I)
_RU = re.compile(r'name=["\']ru["\'][^>]*value=["\']([^"\']*)', re.I)


class EbayPlugin(WebPlugin):
    name, domain = "ebay", "signin.ebay.com"
    auth_cookies = ("ebay", "dp1")

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        page = client.get("https://signin.ebay.com/ws/eBayISAPI.dll?SignIn")
        html = page.text or ""
        srt = (_SRT.search(html) or [None, "1"])[1]
        ru = (_RU.search(html) or [None, ""])[1]
        resp = client.post("https://signin.ebay.com/ws/eBayISAPI.dll?SignIn",
                           data={"userid": username, "pass": password,
                                 "srt": srt, "ru": ru},
                           headers={"Content-Type":
                                    "application/x-www-form-urlencoded"})
        body = (resp.text or "").lower()
        if "ebay" in {c.lower() for c in dict(client.cookie_dict())}:
            return done(AuthResult.SUCCESS, "login successful",
                        SessionData(cookies=dict(client.cookie_dict()),
                                    user_agent=client.user_agent))
        if "captcha" in body or "robot" in body:
            return done(AuthResult.CAPTCHA, "captcha")
        if "security code" in body or "verification" in body:
            return done(AuthResult.MFA, "security code")
        if 'name="pass"' in body or "signin" in str(resp.url).lower():
            return done(AuthResult.FAILURE, "form re-rendered")
        return done(AuthResult.UNKNOWN, f"http {resp.status}")
