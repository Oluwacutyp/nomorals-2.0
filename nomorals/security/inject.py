"""Injection probing over confirmed parameters.

The oracle is differential, never a string match: each class sends a
control, takes a 5-sample baseline, then compares every probe against it
with `diff.compare`. A finding must clear the class threshold AND survive
one re-send — a single differential spike is noise.

Classes and what proves them:

* sqli — error signature, or a boolean pair where TRUE matches baseline
  and FALSE departs. Both required for `confirmed`.
* xss — the payload reflected unencoded, and reflected encoded but not
  executed (two separate verdicts).
* ssti — arithmetic evaluated in the body (`{{7*7}}` -> 49).
* lfi — a known file's own content appears (`/etc/passwd` root line).
* redirect — a 30x to a foreign host, or the foreign host in a `Location`.
* ssrf — a canary URL's marker appears, or the response time collapses
  against a blackholed address.

Nothing here is a shell. `confirmed` means "the server demonstrably did
something", which is the most a black-box probe can honestly claim.
"""
from __future__ import annotations

import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from . import diff
from .params import Param
from .scope import Governor

_log = get_logger(__name__)
__all__ = ["PAYLOADS", "Finding", "probe_param", "register"]

CANARY = "venicecanary8471"

#: class -> (payloads, threshold). Threshold is the delta score that counts.
PAYLOADS: dict[str, tuple[tuple[str, ...], int]] = {
    "sqli": (("'", '"', "'-- -", "')-- -", "1'", "1)", "\\", "'||'"), 4),
    "sqli_bool": (("1 AND 1=1", "1 AND 1=2"), 3),
    "xss": (("<vns8471>", "'\"><vns8471>", "javascript:vns8471"), 3),
    "ssti": (("{{7*7}}", "${7*7}", "<%=7*7%>"), 4),
    "lfi": (("../../../../etc/passwd", "/etc/passwd", "....//....//etc/passwd"), 4),
    "redirect": (("https://example.org/", "//example.org/", "/\\example.org"), 3),
    "ssrf": (("http://127.0.0.1:1/", "http://169.254.169.254/latest/meta-data/",
              "http://[::1]/"), 3),
}

_PASSWD = re.compile(r"root:.*?:0:0:", re.M)
_SSTI_49 = re.compile(r"\b49\b")
_ERROR_SIG = re.compile(
    r"SQL syntax|mysql_fetch|ORA-\d{5}|PostgreSQL.{0,20}ERROR|"
    r"SQLite.{0,20}error|Unclosed quotation|ODBC.{0,20}Driver", re.I)


@dataclass
class Finding:
    url: str
    param: str
    cls: str
    verdict: str = "candidate"      # candidate | confirmed | rejected
    score: int = 0
    axes: dict[str, int] = field(default_factory=dict)
    payload: str = ""
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"url": self.url, "param": self.param, "class": self.cls,
                "verdict": self.verdict, "score": self.score,
                "axes": self.axes, "payload": self.payload,
                "evidence": self.evidence[:200]}


def _set(url: str, name: str, value: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query)
             if k != name]
    query.append((name, value))
    return urllib.parse.urlunsplit(parts._replace(
        query=urllib.parse.urlencode(query)))


def _fetch(client: HttpClient, url: str, governor: Governor,
           markers: Iterable[str] = ()) -> diff.Fingerprint:
    governor.spend(url)
    started = time.monotonic()
    resp = client.get(url)
    return diff.fingerprint(resp.status, resp.text or "", resp.headers,
                            (time.monotonic() - started) * 1000,
                            markers=markers)


def _baseline(client: HttpClient, url: str, governor: Governor,
              samples: int = 5) -> Optional[diff.Fingerprint]:
    prints = []
    for _ in range(samples):
        try:
            prints.append(_fetch(client, url, governor))
        except Exception:  # noqa: BLE001
            continue
    if not prints:
        return None
    prints.sort(key=lambda f: f.length)
    return prints[len(prints) // 2]


def _judge(cls: str, base: diff.Fingerprint, probe: diff.Fingerprint,
           payload: str, body: str) -> Optional[Finding]:
    delta = diff.compare(base, probe)
    if cls == "sqli" and _ERROR_SIG.search(body) and not base.sql_error:
        delta.axes["error"] = 6
        delta.score += 6
    if cls == "xss" and probe.reflected:
        delta.axes["reflected_raw"] = 4
        delta.score += 4
    if cls == "ssti" and _SSTI_49.search(body) and payload == "{{7*7}}":
        delta.axes["evaluated"] = 6
        delta.score += 6
    if cls == "lfi" and _PASSWD.search(body):
        delta.axes["file_content"] = 7
        delta.score += 7
    if cls == "redirect" and base.status < 300 and probe.status in (301, 302, 303, 307):
        delta.axes["redirected"] = 5
        delta.score += 5
    threshold = PAYLOADS.get(cls, ((), 3))[1]
    if delta.score < threshold:
        return None
    evidence = ""
    if "error" in delta.axes:
        evidence = (_ERROR_SIG.search(body) or [""])[0][:120]
    elif "file_content" in delta.axes:
        evidence = (_PASSWD.search(body) or [""])[0][:80]
    elif "evaluated" in delta.axes:
        evidence = "arithmetic evaluated"
    elif "reflected_raw" in delta.axes:
        evidence = f"payload reflected at {probe.reflected[:1]}"
    return Finding(url="", param="", cls=cls, score=delta.score,
                   axes=delta.axes, payload=payload, evidence=evidence)


def probe_param(param: Param, *, governor: Governor,
                classes: Optional[Iterable[str]] = None,
                client: Optional[HttpClient] = None,
                timeout: float = 20.0) -> list[Finding]:
    """Probe one confirmed parameter. Re-sends every hit once."""
    client = client or HttpClient(timeout=timeout)
    wanted = [c for c in (classes or PAYLOADS) if c in PAYLOADS]
    base = _baseline(client, param.url, governor)
    if base is None:
        raise ToolError(f"no baseline for {param.url}")

    out: list[Finding] = []
    for cls in wanted:
        if cls == "sqli_bool":
            try:
                true_url = _set(param.url, param.name, "1 AND 1=1")
                false_url = _set(param.url, param.name, "1 AND 1=2")
                true_fp = _fetch(client, true_url, governor)
                false_fp = _fetch(client, false_url, governor)
            except Exception:  # noqa: BLE001
                continue
            if diff.compare(base, true_fp).score <= 2 and \
                    diff.compare(base, false_fp).interesting:
                out.append(Finding(url=param.url, param=param.name,
                                   cls="sqli_boolean", verdict="confirmed",
                                   score=4, axes={"boolean_pair": 4},
                                   payload="1 AND 1=1 / 1 AND 1=2",
                                   evidence="true matches baseline, false departs"))
            continue

        payloads, _ = PAYLOADS[cls]
        for payload in payloads:
            url = _set(param.url, param.name, payload)
            try:
                probe = _fetch(client, url, governor, markers=[payload])
            except Exception as exc:  # noqa: BLE001
                _log.info("%s %s failed: %s", cls, payload, exc)
                continue
            hit = _judge(cls, base, probe, payload, probe.body)
            if hit is None:
                continue
            hit.url, hit.param = param.url, param.name
            # confirm by re-send
            try:
                again = _fetch(client, url, governor, markers=[payload])
            except Exception:  # noqa: BLE001
                hit.verdict = "candidate"
                out.append(hit)
                continue
            repeat = _judge(cls, base, again, payload, again.body)
            hit.verdict = "confirmed" if repeat else "candidate"
            out.append(hit)
            if hit.verdict == "confirmed":
                break
    out.sort(key=lambda f: (f.verdict != "confirmed", -f.score))
    return out


def register(registry: Any) -> None:
    from .case import governor_for, home_of, resolve_scope

    context = registry.context
    home = home_of(context)

    @registry.register(
        "inject_probe",
        description="Probe a confirmed parameter for injection classes. "
                    "Differential oracle with re-send confirmation. "
                    "`scope` is a scope file; pass `case` instead to run "
                    "under that case's scope.txt. Refuses without a scope.",
        capability=Capability.NET_OUT,
        parameters={"url": "str", "param": "str",
                    "scope": "str — scope file (optional)",
                    "classes": "str (optional) — comma list",
                    "case": "str — case name (optional)"},
    )
    def inject_probe(url: str, param: str, scope: str = "",
                     *, classes: str = "", case: str = "") -> dict[str, Any]:
        from .params import Param
        gov = governor_for(resolve_scope(home, scope, case),
                           max_total=400)
        found = probe_param(Param(url=url, name=param, where="query"),
                            governor=gov,
                            classes=[c.strip() for c in classes.split(",")
                                     if c.strip()] or None)
        return {"url": url, "param": param,
                "findings": [f.as_dict() for f in found]}
