"""Forge authenticated sessions and keep them.

Runs a plugin against a credential, and on success persists the cookie
jar + headers to `<home>/data/sessions/<domain>/<user>.json` (atomic,
0600) so later stages act authenticated instead of re-logging in.

Escalation ladder, cheapest first:
  1. the plugin's HTTP flow;
  2. one retry after a short pause;
  3. the browser flow (Playwright) when the plugin reports ERROR twice —
     some sites' login is JS-only and no HTTP flow will ever pass.

A MFA result is a stop, not a failure: the session is stored with
`verified: false` and the flag rides forward.

Sessions are verified before use and re-verified after; a stale cookie
jar is dropped rather than silently reused.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .plugins.interface import AuthResult, LoginResult

_log = get_logger(__name__)
__all__ = ["forge", "store", "load", "verify_session", "drop", "register"]

_SLUG = re.compile(r"[^a-z0-9._\-]+")


def _root(home: str | Path) -> Path:
    return Path(home) / "data" / "sessions"


def _path(root: Path, domain: str, username: str) -> Path:
    return root / _SLUG.sub("-", domain.lower()) / f"{_SLUG.sub('-', username.lower())}.json"


def store(root: Path, domain: str, result: LoginResult) -> Path:
    path = _path(root, domain, result.username)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "domain": domain, "username": result.username,
        "result": result.result.value, "message": result.message,
        "created": time.time(), "verified": False,
        "session": result.session.as_dict() if result.session else None,
    }, indent=1), encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(tmp, 0o600)
    tmp.replace(path)
    return path


def load(root: Path, domain: str, username: str) -> dict[str, Any] | None:
    path = _path(root, domain, username)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def drop(root: Path, domain: str, username: str) -> bool:
    path = _path(root, domain, username)
    if path.is_file():
        path.unlink()
        return True
    return False


def _client_from(stored: dict[str, Any], timeout: float,
                 proxy: str | None,
                 allow_private_ips: bool = False) -> HttpClient:
    client = HttpClient(timeout=timeout, proxy_url=proxy or "",
                        allow_private_ips=allow_private_ips)
    session = stored.get("session") or {}
    for name, value in (session.get("headers") or {}).items():
        client.headers[name] = value
    if session.get("user_agent"):
        client.user_agent = session["user_agent"]
    for name, value in (session.get("cookies") or {}).items():
        client.set_cookie(name, value)
    return client


def verify_session(root: Path, domain: str, username: str, probe_url: str,
                   *, timeout: float = 20.0,
                   marker: str = "",
                   allow_private_ips: bool = False) -> dict[str, Any]:
    """Replay the jar against a URL. `marker` is a string only a logged-in
    page carries; without one the result is UNKNOWN, never VALID."""
    stored = load(root, domain, username)
    if not stored or not stored.get("session"):
        return {"state": "missing"}
    client = _client_from(stored, timeout, None, allow_private_ips)
    try:
        resp = client.get(probe_url)
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "reason": str(exc)[:80]}
    body = resp.text or ""
    state = ("valid" if marker and marker.lower() in body.lower()
             else "unknown")
    if state == "valid":
        stored["verified"] = True
        stored["verified_at"] = time.time()
        _path(root, domain, username).write_text(
            json.dumps(stored, indent=1), encoding="utf-8")
    return {"state": state, "status": resp.status, "bytes": len(body)}


def _browser_escalation(domain: str, username: str, password: str,
                        login_url: str) -> LoginResult:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return LoginResult(AuthResult.ERROR, username, password,
                           "playwright not installed")
    started = time.monotonic()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(login_url, wait_until="domcontentloaded", timeout=45_000)
        for selector in ('input[type="email"]', 'input[name="username"]',
                         'input[name="session_key"]', "#username"):
            if page.locator(selector).count():
                page.fill(selector, username)
                break
        page.keyboard.press("Enter")
        page.wait_for_timeout(1_500)
        for selector in ('input[type="password"]', 'input[name="password"]'):
            if page.locator(selector).count():
                page.fill(selector, password)
                break
        page.keyboard.press("Enter")
        page.wait_for_timeout(4_000)
        cookies = {c["name"]: c["value"] for c in ctx.cookies()}
        browser.close()
    ok = bool(cookies)
    return LoginResult(
        AuthResult.SUCCESS if ok else AuthResult.UNKNOWN, username, password,
        "browser flow", (time.monotonic() - started) * 1000,
        session=type("S", (), {"as_dict": lambda self: {"cookies": cookies}})()
        if ok else None)


def forge(home: str | Path, plugin: Any, username: str, password: str, *,
          proxy: str | None = None, timeout: float = 30.0,
          escalate: bool = True, retries: int = 1,
          allow_private_ips: bool = False) -> LoginResult:
    root = _root(home)
    client = HttpClient(timeout=timeout, proxy_url=proxy or "",
                        allow_private_ips=allow_private_ips)
    result = plugin.login(username, password, client)
    for _ in range(retries):
        if result.result not in (AuthResult.ERROR, AuthResult.UNKNOWN):
            break
        time.sleep(1.5)
        client = HttpClient(timeout=timeout, proxy_url=proxy or "",
                            allow_private_ips=allow_private_ips)
        result = plugin.login(username, password, client)
    if result.result is AuthResult.ERROR and escalate:
        result = _browser_escalation(plugin.domain, username, password,
                                     f"https://{plugin.domain}/login")
    if result.result in (AuthResult.SUCCESS, AuthResult.MFA):
        store(root, plugin.domain, result)
    return result


def register(registry: Any) -> None:
    context = registry.context
    home = Path(getattr(context, "home", ".") or ".")

    @registry.register(
        "session_forge",
        description="Log in with a plugin and persist the cookie jar for "
                    "later authenticated stages. Escalates to a browser flow "
                    "when the HTTP flow fails.",
        capability=Capability.NET_OUT,
        parameters={"plugin": "str — plugin name",
                    "username": "str", "password": "str"},
    )
    def session_forge(plugin: str, username: str, password: str) -> dict[str, Any]:
        from .plugins import PLUGINS
        if plugin not in PLUGINS:
            raise ToolError(f"unknown plugin {plugin!r}")
        return forge(home, PLUGINS[plugin](), username, password).as_dict()

    @registry.register(
        "session_verify",
        description="Replay a stored session against a URL with a logged-in marker.",
        capability=Capability.NET_OUT,
        parameters={"domain": "str", "username": "str",
                    "url": "str", "marker": "str"},
    )
    def session_verify(domain: str, username: str, url: str,
                       marker: str = "") -> dict[str, Any]:
        return verify_session(_root(home), domain, username, url,
                              marker=marker)
