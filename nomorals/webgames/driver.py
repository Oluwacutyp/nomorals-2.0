"""Browser-as-mint: provision, stealth, clear the wall, hand back tokens.

The browser is used for three things only — clearing a bot wall,
registering, and capturing the save payload. Everything after that runs
over HTTP with the harvested cookies, which is 50x cheaper and does not
spend a browser process per account.

Wall handling is a detector plus a wait, never a sleep: it watches for the
interstitial to disappear by selector and title, and reports the wall kind
so the caller can rotate instead of retrying blind.

Provisioning is bounded and one-shot per process, mirroring
browser/provision.py.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)
__all__ = ["Wall", "MintedSession", "ensure_browser", "mint"]

_PROVISIONED = {"done": False}
_WALL_SELECTORS = (
    "#challenge-form", "#cf-challenge-running", ".cf-browser-verification",
    "iframe[src*='challenges.cloudflare.com']", "#challenge-stage",
)
_WALL_TITLE = ("just a moment", "attention required", "checking your browser",
               "verifying you are human")
_STEALTH = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
Object.defineProperty(navigator,'languages',
  {get:()=>['en-NG','en','en-US']});
window.chrome = window.chrome || {runtime:{}};
const orig = navigator.permissions.query;
navigator.permissions.query = p => p.name === 'notifications'
  ? Promise.resolve({state: Notification.permission}) : orig(p);
const gp = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function (p) {
  if (p === 37445) return 'Intel Inc.';
  if (p === 37446) return 'Intel Iris OpenGL Engine';
  return gp.call(this, p);
};
"""


class Wall(Exception):
    """A bot wall did not clear."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"bot wall: {kind}")
        self.kind = kind


@dataclass
class MintedSession:
    cookies: dict[str, str] = field(default_factory=dict)
    user_agent: str = ""
    storage_state: dict[str, Any] = field(default_factory=dict)
    wall_cleared: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"cookies": self.cookies, "user_agent": self.user_agent,
                "wall_cleared": self.wall_cleared}


def ensure_browser() -> None:
    """Provision Playwright + Chromium once per process. Raises if it can't."""
    if _PROVISIONED["done"]:
        return
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ToolError("playwright missing — `nm browser provision`") from None
    from ..core.shell import run  # noqa: PLC0415
    try:
        with sync_playwright() as pw:
            pw.chromium.launch(headless=True).close()
    except Exception:  # noqa: BLE001 - binary absent
        _log.info("chromium absent, provisioning once")
        run(["python", "-m", "playwright", "install", "chromium"],
            timeout=600)
        with sync_playwright() as pw:
            pw.chromium.launch(headless=True).close()
    _PROVISIONED["done"] = True


def _profile_dir(root: Path, slot: int) -> Path:
    path = root / f"slot-{slot}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _wall_kind(page: Any) -> str:
    title = (page.title() or "").lower()
    for marker in _WALL_TITLE:
        if marker in title:
            return "cloudflare-title"
    for selector in _WALL_SELECTORS:
        try:
            if page.locator(selector).count():
                return f"cloudflare:{selector}"
        except Exception:  # noqa: BLE001
            continue
    return ""


def _wait_wall(page: Any, *, timeout: float = 90.0) -> None:
    """Block until no wall is present. Raises Wall on timeout."""
    deadline = time.time() + timeout
    seen = ""
    while time.time() < deadline:
        kind = _wall_kind(page)
        if kind:
            seen = kind
            page.wait_for_timeout(700)
            continue
        if seen:
            _log.info("wall cleared after %s", seen)
        return
    raise Wall(seen or "unknown")


def mint(url: str, *, root: Path, slot: int, proxy: Optional[str] = None,
         headless: bool = True, timeout: float = 90.0) -> MintedSession:
    """Open a persistent context, clear the wall, return its tokens.

    A fresh profile is used per mint so canvas/storage fingerprints do not
    collide across slots.
    """
    ensure_browser()
    from playwright.sync_api import sync_playwright

    profile = _profile_dir(root, slot)
    with sync_playwright() as pw:
        kwargs: dict[str, Any] = {"headless": headless,
                                  "viewport": {"width": 393, "height": 852},
                                  "locale": "en-NG",
                                  "timezone_id": "Africa/Lagos",
                                  "user_agent": random.choice((
                                      "Mozilla/5.0 (Linux; Android 14; SM-S918B) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/126.0.0.0 Mobile Safari/537.36",
                                      "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/125.0.0.0 Mobile Safari/537.36"))}
        if proxy:
            kwargs["proxy"] = {"server": proxy}
        ctx = pw.chromium.launch_persistent_context(str(profile), **kwargs)
        try:
            ctx.add_init_script(_STEALTH)
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            _wait_wall(page, timeout=timeout)
            session = MintedSession(
                cookies={c["name"]: c["value"] for c in ctx.cookies()},
                user_agent=kwargs["user_agent"],
                storage_state=ctx.storage_state(), wall_cleared=True)
            return session
        finally:
            ctx.close()


def with_page(url: str, *, root: Path, slot: int, proxy: Optional[str] = None,
              headless: bool = True, timeout: float = 90.0):
    """Context manager yielding (page, ctx) with the wall already cleared."""
    ensure_browser()
    from contextlib import contextmanager
    from playwright.sync_api import sync_playwright

    @contextmanager
    def _cm():
        profile = _profile_dir(root, slot)
        with sync_playwright() as pw:
            kwargs: dict[str, Any] = {"headless": headless,
                                      "viewport": {"width": 393, "height": 852},
                                      "locale": "en-NG",
                                      "timezone_id": "Africa/Lagos"}
            if proxy:
                kwargs["proxy"] = {"server": proxy}
            ctx = pw.chromium.launch_persistent_context(str(profile), **kwargs)
            try:
                ctx.add_init_script(_STEALTH)
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                _wait_wall(page, timeout=timeout)
                yield page, ctx
            finally:
                ctx.close()

    return _cm()
