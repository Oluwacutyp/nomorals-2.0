"""CORS misconfiguration testing and CSRF PoC generation.

CORS is decided by echoing an origin, so the test is an origin sweep with
a credentialed preflight — reflection of an arbitrary origin WITH
credentials is the finding. Wildcard without credentials is informational.

CSRF PoC is generated only for a form the caller already knows: the
module writes the HTML, it does not guess the target's token scheme.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.logging_setup import get_logger
from .scope import Governor, OutOfScope

_log = get_logger(__name__)
__all__ = ["ORIGINS", "CorsResult", "cors_probe", "csrf_poc", "register"]

ORIGINS = ("https://evil.example", "null", "https://target.evil.example",
           "https://evil.example:443")


@dataclass
class CorsResult:
    url: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = "clean"

    def as_dict(self) -> dict[str, Any]:
        return {"url": self.url, "verdict": self.verdict,
                "findings": self.findings}


def cors_probe(url: str, *, governor: Governor,
               client: Optional[HttpClient] = None,
               timeout: float = 20.0) -> CorsResult:
    client = client or HttpClient(timeout=timeout)
    out = CorsResult(url=url)
    # The scope gate sits OUTSIDE the per-origin try below: an
    # out-of-scope URL is a refusal, never a "clean" verdict with the
    # scope error buried as a per-origin error finding.
    if not governor.scope.allows(url):
        raise OutOfScope(f"{url} is not in scope")
    for origin in ORIGINS:
        try:
            governor.spend(url)
            resp = client.get(url, headers={"Origin": origin,
                                            "Referer": origin})
        except Exception as exc:  # noqa: BLE001
            out.findings.append({"origin": origin,
                                 "error": type(exc).__name__})
            continue
        allow = resp.headers.get("access-control-allow-origin", "")
        creds = (resp.headers.get("access-control-allow-credentials", "")
                 .lower() == "true")
        if not allow:
            continue
        reflected = allow == origin or allow == "*"
        severity = ("critical" if reflected and creds and allow != "*"
                    else "high" if reflected and creds
                    else "low" if allow == "*" else "info")
        out.findings.append({"origin": origin, "allow_origin": allow,
                             "credentials": creds, "severity": severity})
        if severity in ("critical", "high"):
            out.verdict = "exploitable"
    if not out.findings:
        out.verdict = "no_cors_headers"
    return out


_POC = """<!doctype html>
<meta charset="utf-8">
<title>CSRF PoC</title>
<body onload="document.forms[0].submit()">
<form action="{action}" method="{method}" enctype="text/plain">
{fields}
</form>
</body>"""


def csrf_poc(action: str, fields: dict[str, str],
             method: str = "POST") -> str:
    """Self-submitting form. `text/plain` enctype is deliberate: it is the
    only enctype a cross-origin page may send without a preflight, so it
    reaches endpoints that reject urlencoded."""
    rendered = "\n".join(
        f'  <input name="{html.escape(k)}" value="{html.escape(str(v))}">'
        for k, v in fields.items())
    return _POC.format(action=html.escape(action),
                       method=method.upper(), fields=rendered)


def register(registry: Any) -> None:
    from .case import governor_for, home_of, resolve_scope

    home = home_of(registry.context)

    @registry.register("cors_probe",
                       description="Origin sweep for CORS reflection with "
                                   "credentials. `scope` is a scope file; "
                                   "pass `case` instead to run under that "
                                   "case's scope.txt. Refuses without a "
                                   "scope.",
                       capability=Capability.NET_OUT,
                       parameters={"url": "str",
                                   "scope": "str — scope file (optional)",
                                   "case": "str — case name (optional)"})
    def cors_probe_tool(url: str, scope: str = "",
                        case: str = "") -> dict[str, Any]:
        gov = governor_for(resolve_scope(home, scope, case), max_total=10)
        return cors_probe(url, governor=gov).as_dict()

    @registry.register("csrf_poc",
                       description="Generate a self-submitting CSRF PoC form.",
                       capability=Capability.DB_READ,
                       parameters={"action": "str", "fields": "dict",
                                   "method": "str (optional, POST)",
                                   "out": "str (optional)"})
    def csrf_poc_tool(action: str, fields: dict[str, str],
                      *, method: str = "POST", out: str = "") -> dict[str, Any]:
        page = csrf_poc(action, fields, method)
        if out:
            with open(out, "w", encoding="utf-8") as f:
                f.write(page)
        return {"action": action, "bytes": len(page), "file": out or None,
                "html": page if not out else ""}
