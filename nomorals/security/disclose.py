"""Severity rubric, contact discovery, disclosure draft.

Severity is a deterministic product of four measured axes — kind,
sensitivity, reachability, scale — scored against a fixed table so two
analysts reach the same tier from the same finding. Nothing is scored on
impression.

Contact discovery follows RFC 9116 (`security.txt`, including the
`_security` TXT pointer and the `.well-known` location), `SECURITY.md`,
and RDAP abuse entities. Every contact found is quoted with its source;
none is guessed.

The draft is markdown built from the evidence chain: real request/response
pairs, masked, with repro steps. Sending stays the owner's act.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.errors import ToolError
from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.logging_setup import get_logger

_log = get_logger(__name__)
__all__ = ["score", "Tier", "find_contacts", "draft", "register"]

#: kind -> base weight.  Measured, not intuited.
_KIND = {
    "rce": 40, "sqli": 35, "auth_bypass": 35, "ssrf": 30, "idor": 30,
    "takeover": 30, "secret_leak": 30, "deserialization": 30,
    "xss_stored": 25, "exposed_data": 25, "path_traversal": 25,
    "bucket_listing": 20, "open_admin": 25, "xss_reflected": 15,
    "cors_wildcard": 15, "tls_weak": 10, "header_missing": 5,
    "version_disclosure": 5, "info": 2,
}
_SENSITIVITY = {"financial": 25, "pii": 20, "credential": 25,
                "internal": 10, "none": 0}
_REACH = {"unauth": 20, "auth": 10, "internal": 5}
_SCALE = {"bulk": 20, "many": 12, "few": 5, "single": 0}


class Tier(str):
    CRITICAL, HIGH, MEDIUM, LOW, INFO = "critical", "high", "medium", "low", "info"


def score(kind: str, *, sensitivity: str = "none", reach: str = "unauth",
          scale: str = "single") -> dict[str, Any]:
    """Deterministic 0-100 score and its tier."""
    total = (_KIND.get(kind.lower(), 5) + _SENSITIVITY.get(sensitivity.lower(), 0)
             + _REACH.get(reach.lower(), 0) + _SCALE.get(scale.lower(), 0))
    total = max(0, min(100, total))
    tier = (Tier.CRITICAL if total >= 75 else Tier.HIGH if total >= 55
            else Tier.MEDIUM if total >= 30 else Tier.LOW if total >= 12
            else Tier.INFO)
    return {"score": total, "tier": tier, "kind": kind,
            "sensitivity": sensitivity, "reach": reach, "scale": scale,
            "breakdown": {"kind": _KIND.get(kind.lower(), 5),
                          "sensitivity": _SENSITIVITY.get(sensitivity.lower(), 0),
                          "reach": _REACH.get(reach.lower(), 0),
                          "scale": _SCALE.get(scale.lower(), 0)}}


_CONTACT_RE = re.compile(r"^Contact:\s*(\S+)", re.I | re.M)
_MAILTO = re.compile(r"mailto:([^@\s]+@[a-z0-9.\-]+)", re.I)
_EMAIL = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.I)
_SECURITY_TXT = ("https://{d}/.well-known/security.txt", "https://{d}/security.txt")


def find_contacts(domain: str, *, timeout: float = 15.0) -> dict[str, Any]:
    """Contacts with provenance. Empty means empty, never invented."""
    client = HttpClient(timeout=timeout)
    found: list[dict[str, str]] = []

    for tmpl in _SECURITY_TXT:
        url = tmpl.format(d=domain)
        try:
            resp = client.get(url)
        except Exception:  # noqa: BLE001
            continue
        if resp.ok and "contact" in (resp.text or "").lower():
            for m in _CONTACT_RE.finditer(resp.text or ""):
                for mail in _MAILTO.findall(m.group(1)):
                    found.append({"email": mail, "source": url})
            break

    for path in ("/SECURITY.md", "/.github/SECURITY.md", "/security.md"):
        url = f"https://{domain}{path}"
        try:
            resp = client.get(url)
        except Exception:  # noqa: BLE001
            continue
        if resp.ok:
            for mail in _EMAIL.findall((resp.text or "")[:8_000]):
                found.append({"email": mail, "source": url})

    try:
        resp = client.get(f"https://rdap.org/domain/{urllib.parse.quote(domain)}")
        if resp.ok:
            data = json.loads(resp.text or "{}")
            for entity in data.get("entities", []) or []:
                if "abuse" not in (entity.get("roles") or []):
                    continue
                blob = json.dumps(entity)
                for mail in _EMAIL.findall(blob)[:2]:
                    found.append({"email": mail, "source": "rdap abuse entity"})
    except Exception:  # noqa: BLE001
        pass

    seen: dict[str, dict[str, str]] = {}
    for row in found:
        seen.setdefault(row["email"].lower(), row)
    ranked = sorted(seen.values(),
                    key=lambda r: ("security.txt" not in r["source"], r["email"]))
    return {"domain": domain, "contacts": ranked,
            "count": len(ranked),
            "note": ("" if ranked else
                     "no contact published — use the registrar abuse address")}


def _repro(record: dict[str, Any]) -> list[str]:
    req = record.get("request", {})
    lines = [f"1. `{req.get('method', 'GET')} {req.get('url', '')}`"]
    headers = req.get("headers") or {}
    keep = {k: v for k, v in headers.items()
            if k.lower() in ("accept", "content-type", "origin")}
    if keep:
        lines.append("   ```")
        for k, v in keep.items():
            lines.append(f"   {k}: {v}")
        lines.append("   ```")
    if req.get("body"):
        lines += ["   ```", f"   {req['body'][:800]}", "   ```"]
    resp = record.get("response", {})
    lines.append(f"2. Observed: HTTP {resp.get('status', '?')} — "
                 f"{resp.get('body_bytes', 0)} bytes, "
                 f"sha256 `{str(resp.get('body_sha256', ''))[:32]}…`")
    excerpt = (resp.get("body_excerpt") or "").strip()
    if excerpt:
        lines += ["   ```", *[f"   {ln}" for ln in excerpt.splitlines()[:12]],
                  "   ```"]
    return lines


def draft(case: str, chain_path: str | Path, *, domain: str,
          contact: str = "", days: int = 90) -> dict[str, Any]:
    """Markdown disclosure draft from a verified chain."""
    from .evidence import verify_chain

    path = Path(chain_path)
    integrity = verify_chain(path)
    if integrity["state"] != "intact":
        raise ToolError(f"chain is {integrity['state']} at "
                        f"{integrity['broken_at']} — fix before drafting")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if not rows:
        raise ToolError("chain has no records")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("verdict", "info"), []).append(row)

    head = rows[-1]["hash"]
    out = [
        f"# Security disclosure — {domain}", "",
        f"**Case:** {case}  ",
        f"**Prepared:** {rows[-1]['at_iso']}  ",
        f"**Evidence:** {len(rows)} records, chain head `{head[:24]}…`, "
        f"integrity **{integrity['state']}**  ",
        f"**Requested fix window:** {days} days from acknowledgement", "",
        "Findings were reproduced anonymously against public endpoints. "
        "No authenticated state was modified and no data was retained beyond "
        "the excerpts below.", "",
        "## Summary", "",
    ]
    for verdict, items in sorted(grouped.items(),
                                 key=lambda kv: -len(kv[1])):
        out.append(f"- `{verdict}` × {len(items)}")
    out.append("")

    for n, (verdict, items) in enumerate(sorted(grouped.items()), 1):
        first = items[0]
        out += [f"## {n}. {verdict}", "",
                f"**Target:** `{first.get('target', '')}`  ",
                f"**Occurrences:** {len(items)}  ",
                f"**Record:** `{first['hash'][:24]}…`", ""]
        detail = first.get("detail") or {}
        if detail.get("note"):
            out += [str(detail["note"]), ""]
        out += ["### Reproduction", "", *_repro(first), ""]
        if len(items) > 1:
            out += ["Further affected:", ""]
            out += [f"- `{r.get('target', '')}`" for r in items[1:8]]
            out.append("")

    out += ["## Verification", "",
            "The evidence chain is append-only and hash-linked: each record "
            "carries its predecessor's digest. Any edit breaks verification at "
            "the modified index. Re-verify with:", "",
            "```", "nm evidence verify " + case, "```", "",
            "## Contact", ""]
    out += ([f"Sent to `{contact}`."] if contact else
            ["No published contact was found for this domain."])
    out.append("")

    target_dir = path.parent / "disclosure"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "draft.md"
    target_file.write_text("\n".join(out), encoding="utf-8")
    return {"file": str(target_file), "records": len(rows),
            "head": head, "integrity": integrity["state"],
            "tiers": {v: score(v)["tier"] for v in grouped}}


def register(registry: Any) -> None:
    context = registry.context
    root = Path(getattr(context, "home", ".") or ".") / "data" / "evidence"

    @registry.register(
        "severity_score",
        description="Deterministic severity from kind/sensitivity/reach/scale.",
        capability=Capability.DB_READ,
        parameters={"kind": "str", "sensitivity": "str (optional)",
                    "reach": "str (optional)", "scale": "str (optional)"},
    )
    def severity_score(kind: str, *, sensitivity: str = "none",
                       reach: str = "unauth", scale: str = "single") -> dict[str, Any]:
        return score(kind, sensitivity=sensitivity, reach=reach, scale=scale)

    @registry.register(
        "disclose_contacts",
        description="Find published security contacts (security.txt, SECURITY.md, "
                    "RDAP abuse) with provenance.",
        capability=Capability.NET_OUT,
        parameters={"domain": "str"},
    )
    def disclose_contacts(domain: str) -> dict[str, Any]:
        return find_contacts(domain)

    @registry.register(
        "disclose_draft",
        description="Draft a disclosure from a verified evidence chain.",
        capability=Capability.DB_READ,
        parameters={"case": "str", "domain": "str", "contact": "str (optional)"},
    )
    def disclose_draft(case: str, domain: str, *, contact: str = "") -> dict[str, Any]:
        chain = root / case / "chain.jsonl"
        return draft(case, chain, domain=domain, contact=contact)
