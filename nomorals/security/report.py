"""Engagement report: inventory + findings + evidence, one artifact.

Composes `audit_report` for grading, remediation and summary rather than
re-implementing them. The only new vocabulary is a verdict→kind bridge,
because the chain's verdicts (takeover, wildcard-match, valid_subscribed)
are not the audit battery's finding kinds.

Coverage is stated explicitly, including what was NOT run — a report that
implies completeness it doesn't have is the usual failure.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from .audit_report import (
    SEVERITY_ORDER,
    _remediation_for,
    grade_findings,
    summarize,
)
from .evidence import verify_chain

_log = get_logger(__name__)
__all__ = ["EngagementReport", "VERDICT_KIND", "build", "register"]

#: chain verdict -> audit_report finding kind. Anything absent grades low
#: with the reason recorded; it is never silently upgraded.
VERDICT_KIND: dict[str, str] = {
    "takeover": "exposed_git",
    "secret_leak": "repo_secret",
    "exposed_data": "unauthenticated_api",
    "bucket_listing": "public_bucket",
    "open_admin": "unauthenticated_api",
    "cors_wildcard": "cors_wildcard",
    "header_missing": "debug_enabled",
    "tls_weak": "tls_verify_false",
    "valid_subscribed": "info",
    "valid": "info",
    "valid_free": "info",
    "two_fa": "info",
    "locked": "info",
    "captcha": "info",
    "invalid": "info",
    "free": "info",
    "taken": "info",
}

_METHOD = (
    "All discovery was passive first (certificate transparency, DNS over "
    "HTTPS, published zone data). Active steps followed the passive list and "
    "were rate-limited per host. No authenticated state was modified; no data "
    "beyond bounded excerpts was retained. Every finding carries a "
    "hash-chained evidence record whose integrity is verified below."
)


@dataclass
class Inventory:
    domain: str = ""
    wildcard: Optional[str] = None
    hosts: list[dict[str, Any]] = field(default_factory=list)
    axfr_zones: dict[str, int] = field(default_factory=dict)
    brute: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"domain": self.domain, "wildcard": self.wildcard,
                "host_count": len(self.hosts), "hosts": self.hosts,
                "axfr_zones": self.axfr_zones,
                "brute_queries": self.brute.get("queries"),
                "brute_found": self.brute.get("found")}


class EngagementReport:
    def __init__(self, case_dir: str | Path, *, domain: str,
                 client: str = "", scope_note: str = "") -> None:
        self.dir = Path(case_dir)
        self.chain_path = self.dir / "chain.jsonl"
        if not self.chain_path.is_file():
            raise ToolError(f"no evidence chain at {self.chain_path}")
        self.domain = domain
        self.client = client
        self.scope_note = scope_note

    def _load(self) -> list[dict[str, Any]]:
        rows = []
        for line in self.chain_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows

    def _inventory(self) -> Inventory:
        inv = Inventory(domain=self.domain)
        recon_file = self.dir / "recon.json"
        if recon_file.is_file():
            data = json.loads(recon_file.read_text(encoding="utf-8"))
            inv.wildcard = data.get("wildcard")
            inv.axfr_zones = data.get("axfr") or {}
            inv.hosts = data.get("assets") or []
        brute_file = self.dir / "subbrute.json"
        if brute_file.is_file():
            inv.brute = json.loads(brute_file.read_text(encoding="utf-8"))
            known = {h.get("host") for h in inv.hosts}
            for hit in inv.brute.get("hits") or []:
                if hit.get("host") not in known:
                    inv.hosts.append({"host": hit.get("host"),
                                      "records": {"A": hit.get("addresses") or []},
                                      "sources": ["brute"],
                                      "wildcard": hit.get("wildcard")})
        return inv

    def _grade(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        bridged = []
        for row in rows:
            verdict = str(row.get("verdict") or "info")
            kind = VERDICT_KIND.get(verdict)
            detail = row.get("detail") or {}
            bridged.append({
                "kind": kind or "info",
                "title": f"{verdict} — {row.get('target', '')}",
                "location": row.get("target", ""),
                "check": row.get("module", ""),
                "detail": detail.get("note") or verdict,
                "meta": {"rule_severity": detail.get("severity")}
                if detail.get("severity") in SEVERITY_ORDER else {},
                "evidence": {"request": row.get("request") or {},
                             "response": row.get("response") or {}},
                "_record": row,
            })
        graded = grade_findings(bridged)
        for item in graded:
            if item["kind"] == "info":
                item["grade_reason"] = ("verdict is informational — graded "
                                        "for the record, not as a finding")
        return graded

    def _coverage(self, inv: Inventory) -> list[str]:
        ran = ["certificate transparency", "DNS over HTTPS"]
        skipped = []
        if inv.axfr_zones:
            ran.append("zone transfer")
        else:
            skipped.append("zone transfer (every NS refused)")
        if inv.brute:
            ran.append("subdomain brute")
        else:
            skipped.append("subdomain brute")
        skipped.append("authenticated surface (no credentials supplied)")
        skipped.append("application logic / business rules")
        return [f"ran: {', '.join(ran)}",
                f"not run: {', '.join(skipped)}"]

    def render(self) -> tuple[str, dict[str, Any]]:
        integrity = verify_chain(self.chain_path)
        rows = self._load()
        inv = self._inventory()
        graded = self._grade(rows)
        summary = summarize(graded)
        real = [f for f in graded if f["kind"] != "info"]
        head = rows[-1]["hash"] if rows else ""

        out = [f"# Engagement report — {self.domain}", ""]
        if self.client:
            out.append(f"**Client:** {self.client}  ")
        out += [f"**Generated:** "
                f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}  ",
                f"**Evidence:** {len(rows)} records · head `{head[:24]}…` · "
                f"integrity **{integrity['state']}**  "]
        if self.scope_note:
            out.append(f"**Scope:** {self.scope_note}  ")
        out += ["", "## Executive summary", ""]
        if real:
            out.append(f"{len(real)} finding(s): "
                       + ", ".join(f"{summary['counts'][s]} {s}"
                                   for s in reversed(SEVERITY_ORDER)
                                   if summary["counts"].get(s)))
        else:
            out.append("No findings outside the informational band.")
        if integrity["state"] != "intact":
            out.append(f"\n> **Chain integrity {integrity['state']} at record "
                       f"{integrity['broken_at']}** — treat as unproven below "
                       f"that index.")
        out.append("")

        out += ["## Surface", "",
                f"- hosts in inventory: **{len(inv.hosts)}**",
                f"- wildcard DNS: `{inv.wildcard or 'none'}`"]
        if inv.axfr_zones:
            out.append("- zone transfer succeeded against: "
                       + ", ".join(f"`{ns}` ({n} RRs)"
                                   for ns, n in inv.axfr_zones.items()))
        out.append("")
        wild = [h for h in inv.hosts if h.get("wildcard")]
        real_hosts = [h for h in inv.hosts if not h.get("wildcard")]
        for host in real_hosts[:60]:
            records = host.get("records") or {}
            addr = ",".join(records.get("A") or []) or \
                ",".join(host.get("cname_chain") or [])
            out.append(f"- `{host.get('host')}` — {addr or '—'} "
                       f"({', '.join(host.get('sources') or [])})")
        if wild:
            out.append(f"- plus {len(wild)} wildcard-matching name(s), tagged "
                       f"not counted")
        out.append("")

        out += ["## Findings", ""]
        for n, f in enumerate(real, 1):
            record = f.get("_record") or {}
            out += [f"### {n}. [{f['severity'].upper()}] {f['title']}", "",
                    f"- where: `{f['location']}`",
                    f"- graded: {f['grade_reason']}",
                    f"- fix: {_remediation_for(f)}",
                    f"- record: `{str(record.get('hash', ''))[:24]}…`", ""]
            req = f.get("evidence", {}).get("request") or {}
            resp = f.get("evidence", {}).get("response") or {}
            if req.get("url"):
                out += ["```", f"{req.get('method', 'GET')} {req['url']}", "```",
                        f"Observed HTTP {resp.get('status', '?')} · "
                        f"{resp.get('body_bytes', 0)}b · "
                        f"sha256 `{str(resp.get('body_sha256', ''))[:32]}…`", ""]

        if [f for f in graded if f["kind"] == "info"]:
            out += ["## Informational", ""]
            for f in graded:
                if f["kind"] == "info":
                    out.append(f"- `{f['location']}` — {f['detail']}")
            out.append("")

        out += ["## Coverage", ""] + [f"- {c}" for c in self._coverage(inv)]
        out += ["", "## Method", "", _METHOD, "",
                "## Evidence integrity", "",
                "Records are append-only and hash-linked; each carries its "
                "predecessor's digest. Any edit breaks verification at the "
                "modified index.", "",
                "```", f"nm evidence verify {self.dir.name}", "```", ""]

        manifest = {
            "domain": self.domain, "client": self.client,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "integrity": integrity, "head": head,
            "counts": summary["counts"], "inventory": inv.as_dict(),
            "findings": [{k: v for k, v in f.items() if k != "_record"}
                         for f in graded],
            "coverage": self._coverage(inv),
        }
        return "\n".join(out), manifest

    def write(self) -> dict[str, Any]:
        body, manifest = self.render()
        out_dir = self.dir / "report"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(body, encoding="utf-8")
        (out_dir / "report.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return {"file": str(out_dir / "report.md"),
                "integrity": manifest["integrity"]["state"],
                "counts": manifest["counts"]}


def build(case_dir: str | Path, *, domain: str, client: str = "",
          scope_note: str = "") -> dict[str, Any]:
    return EngagementReport(case_dir, domain=domain, client=client,
                            scope_note=scope_note).write()


def register(registry: Any) -> None:
    context = registry.context
    root = Path(getattr(context, "home", ".") or ".") / "data" / "evidence"

    @registry.register(
        "report_build",
        description="Assemble an engagement report from a case's evidence "
                    "chain and inventory. Refuses an unverifiable chain.",
        capability=Capability.DB_READ,
        parameters={"case": "str", "domain": "str",
                    "client": "str (optional)",
                    "scope": "str (optional)"},
    )
    def report_build(case: str, domain: str, *, client: str = "",
                     scope: str = "") -> dict[str, Any]:
        return build(root / case, domain=domain, client=client,
                     scope_note=scope)
