"""Capture → mutate → replay, with drift bisection.

Three ideas stacked:

* DEVICE COHERENCE. Rotating an IP while keeping one UA and one canvas hash
  is one device teleporting. `mint_device()` produces a coherent set (UA,
  viewport, tz, locale, canvas noise) and a replay uses one set end to end.

* DECLARATIVE MUTATION. A spec says what to rotate and what to hold.
  Field kinds: identity (fresh), signature (re-signed), nonce, sequence,
  timestamp, static. Nothing is hardcoded per platform.

* DRIFT BISECTION. When `counted` collapses, the engine replays pairs with
  one field held static and the rest rotated. The field whose restoration
  brings acceptance back is the rotted one — reported by name. That turns
  a dead endpoint into a named fix instead of a mystery.

JS signatures are re-minted by evaluating the platform's own function in a
live page (`signer="browser"`). Python never pretends to compute X-Bogus.
"""
from __future__ import annotations

import base64
import json
import random
import secrets
import string
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.http import HttpClient
from ..core.errors import ToolError
from ..core.policy import Capability
from ..core.logging_setup import get_logger
from .scope import Governor

_log = get_logger(__name__)
__all__ = ["FIELD_KINDS", "Mutation", "Device", "mint_device", "mutate",
           "replay", "diagnose", "register"]

FIELD_KINDS = ("identity", "signature", "nonce", "sequence", "timestamp",
               "static")

_UAS = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Safari/605.1.15",
)
_VIEWPORTS = ((390, 844), (412, 915), (393, 873), (1920, 1080), (1512, 982))
_ZONES = ("Africa/Lagos", "Europe/London", "Europe/Berlin", "America/New_York")
_LOCALES = ("en-NG,en;q=0.9", "en-GB,en;q=0.9", "en-US,en;q=0.9",
            "de-DE,de;q=0.9,en;q=0.8")
_B36 = string.ascii_lowercase + string.digits


@dataclass
class Device:
    ua: str = ""
    width: int = 0
    height: int = 0
    zone: str = ""
    locale: str = ""
    canvas_seed: str = ""

    def headers(self) -> dict[str, str]:
        return {"User-Agent": self.ua,
                "Accept-Language": self.locale,
                "Sec-CH-UA-Platform": '"Android"' if "Android" in self.ua
                else '"iOS"' if "iPhone" in self.ua else '"Windows"',
                "Viewport-Width": str(self.width)}

    def as_dict(self) -> dict[str, Any]:
        return {"ua": self.ua[:60], "viewport": f"{self.width}x{self.height}",
                "zone": self.zone, "canvas_seed": self.canvas_seed}


def mint_device() -> Device:
    width, height = random.choice(_VIEWPORTS)
    return Device(ua=random.choice(_UAS), width=width, height=height,
                  zone=random.choice(_ZONES),
                  locale=random.choice(_LOCALES),
                  canvas_seed=secrets.token_hex(8))


@dataclass
class Mutation:
    """Which body fields rotate, and how."""
    kinds: dict[str, str] = field(default_factory=dict)
    keep: tuple[str, ...] = ()
    signer: str = "none"

    def kind_of(self, name: str) -> str:
        if name in self.keep:
            return "static"
        return self.kinds.get(name, "static")


def _nonce() -> str:
    return secrets.token_hex(8)


def _verify_fp() -> str:
    return "verify_" + "".join(random.choices(_B36, k=36))


def _ms_token() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(80)).decode().rstrip("=")[:107]


def _uuid() -> str:
    h = secrets.token_hex(16)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def mutate(body: dict[str, Any], spec: Mutation, *, device: Device,
           seq: int = 0) -> dict[str, Any]:
    """Apply the spec. `identity` fields get a fresh shape by name."""
    out = dict(body)
    for name, kind in spec.kinds.items():
        if kind == "static" or name in spec.keep:
            continue
        if kind == "timestamp":
            out[name] = str(int(time.time() * 1000))
        elif kind == "sequence":
            out[name] = str(seq)
        elif kind == "nonce":
            out[name] = _nonce()
        elif kind == "identity":
            low = name.lower()
            if "verifyfp" in low:
                out[name] = _verify_fp()
            elif "mstoken" in low:
                out[name] = _ms_token()
            elif "uuid" in low or "device" in low or "install" in low:
                out[name] = _uuid()
            elif "csrftoken" in low or "csrf" in low:
                out[name] = secrets.token_hex(16)
            else:
                out[name] = _nonce()
        elif kind == "signature":
            out[name] = ""        # filled by the signer
    return out


class _Signer:
    """Holds one live page per platform; re-signs via the page's own JS."""

    def __init__(self, url: str, *, headless: bool = True) -> None:
        self.url = url
        self.headless = headless
        self._pw = None
        self._browser = None
        self.page = None

    def __enter__(self) -> "_Signer":
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        ctx = self._browser.new_context()
        self.page = ctx.new_page()
        self.page.goto(self.url, wait_until="domcontentloaded", timeout=45_000)
        return self

    def sign(self, field_name: str, body: dict[str, Any]) -> str:
        """Call the page's own signer if it exposes one."""
        if self.page is None:
            return ""
        try:
            return str(self.page.evaluate(
                """([field, payload]) => {
                    const w = window;
                    const cands = [w.byted_acrawler, w._signature,
                                   w.sign, w.bytedance];
                    for (const c of cands) {
                      try {
                        if (c && typeof c.sign === 'function')
                          return c.sign(payload);
                        if (typeof c === 'function') return c(payload);
                      } catch (e) {}
                    }
                    return '';
                }""", [field_name, json.dumps(body)]))
        except Exception:  # noqa: BLE001
            return ""

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()


def _classify(status: int, text: str, *, success: str, dupe: str,
              id_field: str, seen: set[str]) -> tuple[str, str]:
    body = (text or "").lower()
    if status in (403, 429) or "captcha" in body:
        return "blocked", f"http {status}"
    if dupe and dupe.lower() in body:
        return "dupe", "dupe marker"
    rid = ""
    if id_field:
        try:
            rid = str(json.loads(text or "{}").get(id_field) or "")
        except ValueError:
            rid = ""
        if rid and rid in seen:
            return "dupe", f"id {rid}"
    if success and success.lower() in body:
        return "counted", rid or "success marker"
    if status in (200, 201) and not rid:
        return "counted", "http 200"
    return "error", f"http {status}"


def replay(url: str, body: dict[str, Any], spec: Mutation, *, count: int,
           governor: Governor, success: str = "", dupe: str = "",
           id_field: str = "", pool: Optional[Any] = None,
           method: str = "POST", stop_after_dupes: int = 5,
           delay: float = 1.4, timeout: float = 25.0,
           signer_url: str = "", progress: bool = True) -> dict[str, Any]:
    """Replay with per-attempt device + mutation. Same stop rule as the
    report engine: consecutive dupes end it."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    dupes = 0
    signer: Optional[_Signer] = None
    if spec.signer == "browser":
        if not signer_url:
            raise ToolError("signer='browser' needs signer_url")
        signer = _Signer(signer_url)
        signer.__enter__()
    try:
        for n in range(1, count + 1):
            device = mint_device()
            payload_body = mutate(body, spec, device=device, seq=n)
            headers = {**device.headers(),
                       "Content-Type": "application/json"}
            if signer is not None:
                for name, kind in spec.kinds.items():
                    if kind == "signature":
                        payload_body[name] = signer.sign(name, payload_body)
            proxy = pool.acquire() if pool is not None else None
            client = HttpClient(timeout=timeout, proxy_url=proxy or "")
            governor.spend(url)
            try:
                if method.upper() == "GET":
                    resp = client.get(url + "?" + urllib.parse.urlencode(
                        payload_body), headers=headers)
                else:
                    resp = client.post(url, data=json.dumps(payload_body),
                                       headers=headers)
                verdict, why = _classify(resp.status, resp.text or "",
                                         success=success, dupe=dupe,
                                         id_field=id_field, seen=seen)
            except Exception as exc:  # noqa: BLE001
                verdict, why = "error", type(exc).__name__
            if verdict == "counted" and why.startswith("id "):
                seen.add(why[3:])
            elif verdict == "counted" and id_field and why not in (
                    "http 200", "success marker"):
                seen.add(why)
            dupes = dupes + 1 if verdict == "dupe" else 0
            rows.append({"n": n, "verdict": verdict, "why": why,
                         "device": device.as_dict()})
            if progress:
                print(f"  {n:>3} {verdict:<8} {why[:44]}")
            if dupes >= stop_after_dupes:
                break
            time.sleep(delay + random.random() * 0.5)
    finally:
        if signer is not None:
            signer.__exit__(None, None, None)
    return {"url": url, "attempted": len(rows),
            "counted": sum(1 for r in rows if r["verdict"] == "counted"),
            "dupes": sum(1 for r in rows if r["verdict"] == "dupe"),
            "blocked": sum(1 for r in rows if r["verdict"] == "blocked"),
            "results": rows}


def diagnose(url: str, body: dict[str, Any], spec: Mutation, *,
             governor: Governor, probes: int = 3, **kw: Any) -> dict[str, Any]:
    """Bisect the rot: hold one field static, rotate the rest.

    The field whose restoration returns `counted` is the rotted one.
    """
    findings = []
    for name in list(spec.kinds):
        if spec.kinds[name] == "static":
            continue
        held = Mutation(kinds={k: v for k, v in spec.kinds.items() if k != name},
                        keep=spec.keep + (name,), signer=spec.signer)
        result = replay(url, body, held, count=probes, governor=governor,
                        progress=False, **kw)
        findings.append({"field": name, "counted": result["counted"],
                         "blocked": result["blocked"]})
    findings.sort(key=lambda f: -f["counted"])
    rotted = [f["field"] for f in findings if f["counted"] > 0]
    return {"url": url, "bisect": findings,
            "suspects": rotted[:3],
            "note": ("a held field that restores acceptance is the rotted one"
                     if rotted else "no single field explains the refusal")}


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "replay_run",
        description="Replay a captured request N times with coherent device "
                    "rotation and declarative field mutation.",
        capability=Capability.NET_OUT,
        parameters={"url": "str", "body": "dict", "count": "int",
                    "kinds": "dict — field -> kind",
                    "signer": "str (optional) — none|browser",
                    "signer_url": "str (optional)"})
    def replay_run(url: str, body: dict[str, Any], *, count: str = "20",
                   kinds: Optional[dict[str, str]] = None,
                   signer: str = "none", signer_url: str = "",
                   success: str = "", dupe: str = "",
                   id_field: str = "") -> dict[str, Any]:
        from .scope import Scope
        spec = Mutation(kinds=kinds or {}, signer=signer)
        gov = Governor(Scope(allow_all=True), max_total=int(count or 20) + 5,
                       max_per_host=3.0)
        pool = None
        try:
            from ..tools.proxy_rotate import pool_from_context
            pool = pool_from_context(context)
            pool.probe_all()
        except Exception:  # noqa: BLE001
            pool = None
        return replay(url, body, spec, count=int(count or 20), governor=gov,
                      success=success, dupe=dupe, id_field=id_field,
                      pool=pool, signer_url=signer_url)

    @registry.register(
        "replay_diagnose",
        description="Bisect which mutated field a target started rejecting.",
        capability=Capability.NET_OUT,
        parameters={"url": "str", "body": "dict",
                    "kinds": "dict", "signer": "str (optional)",
                    "signer_url": "str (optional)"})
    def replay_diagnose(url: str, body: dict[str, Any],
                        kinds: dict[str, str], *, signer: str = "none",
                        signer_url: str = "") -> dict[str, Any]:
        from .scope import Scope
        spec = Mutation(kinds=kinds, signer=signer)
        gov = Governor(Scope(allow_all=True), max_total=200,
                       max_per_host=3.0)
        return diagnose(url, body, spec, governor=gov,
                        signer_url=signer_url)
