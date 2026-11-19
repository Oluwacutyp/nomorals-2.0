"""Twitter/X — onboarding task.json flow with ct0 CSRF + att token."""
from __future__ import annotations

import json
import re
import time

from .interface import AuthResult, LoginResult, SessionData, WebPlugin

class TwitterPlugin(WebPlugin):
    name, domain = "twitter", "x.com"
    auth_cookies = ("auth_token",)

    def login(self, username, password, client):
        started = time.monotonic()
        def done(result, message="", session=None, meta=None):
            return LoginResult(result=result, username=username,
                               password=password, message=message,
                               response_time_ms=(time.monotonic() - started) * 1000,
                               session=session, metadata=meta or {})
        page = client.get("https://twitter.com/i/flow/login",
                          headers={"Referer": "https://twitter.com/"})
        if not page.ok:
            return done(AuthResult.ERROR, f"http {page.status}")
        text = page.text or ""
        ct0 = next((c for c in client.cookie_dict() if c[0] == "ct0"), ("", ""))[1]
        flow = (re.search(r'"flow_token":"([^"]+)"', text) or [None, ""])[1]
        headers = {"Content-Type": "application/json",
                   "X-CSRF-Token": ct0,
                   "Referer": "https://twitter.com/i/flow/login"}
        body = {"flow_token": flow, "subtask_inputs": [{
            "subtask_id": "LoginEnterUserIdentifier",
            "settings_list": {"setting_responses": [{
                "key": "user_identifier",
                "response_data": {"text_data": {"result": username}}}],
                "link": "next_link"}}]}
        step = client.post("https://api.twitter.com/1.1/onboarding/task.json",
                           data=json.dumps(body), headers=headers)
        data = step.json() if step.ok else {}
        if "LoginEnterPassword" not in json.dumps(data):
            return done(AuthResult.UNKNOWN, "no password subtask", meta={"step": data})
        body = {"flow_token": data.get("flow_token"),
                "subtask_inputs": [{"subtask_id": "LoginEnterPassword",
                                    "enter_password": {"password": password,
                                                       "link": "next_link"}}]}
        final = client.post("https://api.twitter.com/1.1/onboarding/task.json",
                            data=json.dumps(body), headers=headers)
        blob = final.text or ""
        if "LoginSuccessSubtask" in blob:
            return done(AuthResult.SUCCESS, "login successful", SessionData(
                cookies=dict(client.cookie_dict()), headers=headers,
                tokens={"ct0": ct0}, user_agent=client.user_agent))
        if "LoginEnterAlternateIdentifierSubtask" in blob:
            return done(AuthResult.MFA, "additional verification")
        errors = (final.json() if final.ok else {}).get("errors") or [{}]
        code = errors[0].get("code")
        if code == 32:
            return done(AuthResult.FAILURE, "invalid credentials")
        if code == 326:
            return done(AuthResult.LOCKOUT, "temporarily locked")
        return done(AuthResult.UNKNOWN, f"http {final.status}")
