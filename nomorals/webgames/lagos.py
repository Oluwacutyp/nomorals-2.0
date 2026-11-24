"""Lagos Life target adapter.

Endpoints, the payload capture, the register flow and the gift. Written
against the live app; every shape carries a `verified` note.

The save payload is captured from the app's own JS on the first slot and
cached, so later accounts never need a browser for it.
"""
from __future__ import annotations

import json
import random
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.errors import ToolError
from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from .driver import Wall, mint, with_page

_log = get_logger(__name__)
__all__ = ["BASE", "GIFT_AMOUNT", "Account", "register", "capture_payload",
           "gift", "qualify", "fetch_balance"]

BASE = "https://lagoslife.app"
GIFT_AMOUNT = 1_050_000

_UA_FALLBACK = ("Mozilla/5.0 (Linux; Android 14; SM-S918B) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Mobile Safari/537.36")


@dataclass
class Account:
    username: str = ""
    password: str = ""
    name: str = ""
    born: str = ""
    user_id: str = ""
    cookies: dict[str, str] = field(default_factory=dict) if False else None
    user_agent: str = _UA_FALLBACK
    state: str = "new"
    note: str = ""

    def client(self, timeout: float = 20.0, proxy: str = "") -> HttpClient:
        client = HttpClient(timeout=timeout, proxy_url=proxy)
        for name, value in (self.cookies or {}).items():
            client.set_cookie(name, value)
        return client

    def as_dict(self) -> dict[str, Any]:
        return {"username": self.username, "password": self.password,
                "name": self.name, "born": self.born,
                "user_id": self.user_id, "state": self.state,
                "note": self.note}


def _random_account() -> Account:
    letters = string.ascii_lowercase
    handle = "".join(random.choices(letters, k=8)) + str(random.randint(100, 9999))
    first = random.choice(("Ada", "Tunde", "Chidi", "Zainab", "Emeka", "Bisi",
                           "Kelechi", "Yemi", "Ngozi", "Femi"))
    last = random.choice(("Okafor", "Balogun", "Adeyemi", "Ibrahim", "Nwosu",
                          "Eze", "Adebayo", "Lawal"))
    return Account(
        username=handle, name=f"{first} {last}",
        password="".join(random.choices(
            string.ascii_letters + string.digits, k=14)) + "!7",
        born=f"{random.randint(1979, 2005)}-{random.randint(1, 12):02d}-"
             f"{random.randint(1, 28):02d}")


def register(account: Account, *, root: Path, slot: int,
             proxy: str = "", headless: bool = True) -> Account:
    """Register through the browser so the wall and the client JS both run."""
    with with_page(f"{BASE}/", root=root, slot=slot, proxy=proxy or None,
                   headless=headless) as (page, ctx):
        payload = {"username": account.username, "password": account.password,
                   "name": account.name, "adult": True, "born": account.born}
        result = page.evaluate(
            """async (p) => {
                const r = await fetch('/api/auth/register', {
                  method: 'POST', headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify(p)});
                let j = null; try { j = await r.json(); } catch (e) {}
                return {status: r.status, body: j};
            }""", payload)
        if result.get("status") not in (200, 201):
            account.state = "register_failed"
            account.note = f"http {result.get('status')}"
            return account
        body = result.get("body") or {}
        account.user_id = str((body.get("user") or {}).get("id")
                              or body.get("id") or "")
        account.cookies = {c["name"]: c["value"] for c in ctx.cookies()}
        account.user_agent = page.evaluate("navigator.userAgent")
        account.state = "registered" if account.user_id else "register_failed"
        return account


def capture_payload(account: Account, *, root: Path, slot: int,
                    cache: Path, proxy: str = "", headless: bool = True,
                    timeout: float = 120.0) -> Optional[dict[str, Any]]:
    """Watch the app's own /api/save and cache the first one."""
    if cache.is_file():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except ValueError:
            pass
    hits: list[str] = []
    with with_page(f"{BASE}/", root=root, slot=slot, proxy=proxy or None,
                   headless=headless) as (page, ctx):
        def on_request(req):
            if "/api/save" in req.url and req.method in ("POST", "PUT", "PATCH"):
                if req.post_data:
                    hits.append(req.post_data)
        ctx.on("request", on_request)
        deadline = time.time() + timeout
        while not hits and time.time() < deadline:
            page.wait_for_timeout(600)
            try:
                page.mouse.click(196, 640)     # nudge onboarding
            except Exception:  # noqa: BLE001
                pass
    if not hits:
        return None
    try:
        body = json.loads(max(hits, key=len))
    except ValueError:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return body


def _personalise(body: dict[str, Any], user_id: str, handle: str) -> dict:
    out = json.loads(json.dumps(body))
    owners = 0

    def walk(node, path=""):
        nonlocal owners
        if isinstance(node, dict):
            for k, v in list(node.items()):
                p = f"{path}.{k}" if path else k
                if k == "owner" and isinstance(v, str):
                    node[k] = user_id
                    owners += 1
                elif isinstance(v, str) and p.endswith("sim.name"):
                    node[k] = handle
                else:
                    walk(v, p)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(out)
    return out


def qualify(account: Account, payload: dict[str, Any], *,
            proxy: str = "", timeout: float = 20.0) -> bool:
    """Send the save payload so the account is eligible to gift."""
    client = account.client(timeout=timeout, proxy=proxy)
    body = _personalise(payload, account.user_id, account.username)
    for method in ("PUT", "POST", "PATCH"):
        try:
            resp = client.request(method, f"{BASE}/api/save", data=json.dumps(body),
                                  headers={"Content-Type": "application/json"})
        except Exception:  # noqa: BLE001
            continue
        if resp.status in (200, 201, 204):
            account.state = "qualified"
            return True
    account.state = "unqualified"
    account.note = "save rejected"
    return False


def gift(account: Account, to_id: str, amount: int = GIFT_AMOUNT, *,
         proxy: str = "", timeout: float = 20.0) -> dict[str, Any]:
    client = account.client(timeout=timeout, proxy=proxy)
    try:
        resp = client.post(f"{BASE}/api/send",
                           data=json.dumps({"to": to_id, "amount": amount}),
                           headers={"Content-Type": "application/json"})
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:90]}
    if resp.status in (200, 201):
        account.state = "gifted"
        return {"ok": True, "body": (resp.text or "")[:200]}
    account.state = "gift_failed"
    account.note = f"http {resp.status}"
    return {"ok": False, "reason": f"http {resp.status} {(resp.text or '')[:90]}"}


def fetch_balance(account: Account, *, proxy: str = "",
                  timeout: float = 15.0) -> Optional[int]:
    client = account.client(timeout=timeout, proxy=proxy)
    try:
        resp = client.get(f"{BASE}/api/auth/me")
        data = json.loads(resp.text or "{}")
    except Exception:  # noqa: BLE001
        return None
    user = data.get("user") or {}
    for key in ("money", "balance", "cash"):
        if isinstance(user.get(key), (int, float)):
            return int(user[key])
    return None


def resolve_target(handle: str, *, proxy: str = "",
                   timeout: float = 20.0) -> Optional[str]:
    """Handle -> user id, via the players endpoint (needs a live session)."""
    client = HttpClient(timeout=timeout, proxy_url=proxy)
    for method, kw in (("GET", {"params": {"q": handle}}),
                       ("POST", {"data": json.dumps({"q": handle})})):
        try:
            resp = client.request(method, f"{BASE}/api/players", **kw)
            rows = (json.loads(resp.text or "{}") or {}).get("players") or []
        except Exception:  # noqa: BLE001
            continue
        for row in rows:
            if str(row.get("username", "")).lower() == handle.lower():
                return str(row.get("id"))
    return None
