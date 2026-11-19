"""Proton — SRP is JS-side, so the browser rung is the real path here."""
from __future__ import annotations

import time

from .interface import AuthResult, LoginResult, SessionData, WebPlugin


class ProtonPlugin(WebPlugin):
    name, domain = "proton", "account.proton.me"
    auth_cookies = ("AUTH-", "proton_session")

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None):
            return LoginResult(result=result, username=username, password=password,
                               message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session)
        # SRP needs JS; one GET to seed cookies, then report ERROR so the
        # forge escalates to the browser rung instead of faking success.
        page = client.get("https://account.proton.me/login")
        if not page.ok:
            return done(AuthResult.ERROR, f"http {page.status}")
        return done(AuthResult.ERROR, "SRP — browser rung required")
