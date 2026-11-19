"""Spotify — OAuth password grant on the web client. sp_dc is the proof."""
from __future__ import annotations

import base64
import json
import time

from .interface import AuthResult, LoginResult, SessionData, WebPlugin

#: public web-player client id/secret, not per-user
_CLIENT = base64.b64encode(
    b"65b708073fc0480ea92a077233ca87bd:"
    b"6b1b0f2e0e9b4e9c9b7d3b0f7e1d4c1a").decode()


class SpotifyPlugin(WebPlugin):
    name, domain = "spotify", "spotify.com"
    auth_cookies = ("sp_dc",)

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        resp = client.post(
            "https://accounts.spotify.com/api/login",
            data={"username": username, "password": password,
                  "remember": "true"},
            headers={"Authorization": f"Basic {_CLIENT}",
                     "Content-Type": "application/x-www-form-urlencoded",
                     "Referer": "https://open.spotify.com/"})
        if resp.ok:
            try:
                data = json.loads(resp.text or "{}")
            except ValueError:
                data = {}
            if data.get("username") or data.get("id"):
                return done(AuthResult.SUCCESS, "login successful",
                            SessionData(cookies=dict(client.cookie_dict()),
                                        user_agent=client.user_agent,
                                        tokens={"access_token":
                                                data.get("accessToken", "")}))
        body = (resp.text or "").lower()
        if resp.status in (400, 401) or "error" in body:
            return done(AuthResult.FAILURE, f"http {resp.status}")
        if "captcha" in body:
            return done(AuthResult.CAPTCHA, "captcha")
        return done(AuthResult.UNKNOWN, f"http {resp.status}")
