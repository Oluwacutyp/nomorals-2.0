"""The security-audit service: one door over the three organs.

``run_web_scan`` / ``run_repo_audit`` each produce the same
shape of outcome: the graded findings, the markdown report
written to the book's out dir, the journal entry (evidence
packs masked at rest), and — when there are findings and a
public contact exists — a disclosure DRAFT saved beside the
report.  Drafts only: nothing is ever sent from here.

Own-asset resolution happens here, from the book's registry:
a target the owner has declared his reads at the own-asset
caps and its delivered report carries unmasked samples; the
journal twin is masked regardless.
"""

from __future__ import annotations

import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

from .audit_repo import audit_repo
from .audit_report import (
    AuditBook, discover_contacts_repo, discover_contacts_web,
    draft_disclosure, grade_findings, render_report_md, summarize)
from .audit_web import Fetcher, scan_url

__all__ = ["AuditService", "run_job"]


def _slug(target: str) -> str:
    host = urllib.parse.urlsplit(
        target if "://" in target else "").hostname
    base = host or Path(target).name or "target"
    return re.sub(r"[^A-Za-z0-9.-]+", "-", base).strip("-")[:60] \
        or "target"


class AuditService:
    def __init__(self, book: AuditBook, *,
                 allow_private: bool = False,
                 min_interval_s: float = 0.35,
                 osv_enabled: bool = True) -> None:
        self.book = book
        self.allow_private = allow_private
        self.min_interval_s = min_interval_s
        self.osv_enabled = osv_enabled

    # ── web ──────────────────────────────────────────────────
    def run_web_scan(self, url: str, *,
                     requester: str = "") -> dict[str, Any]:
        own = self.book.is_own_asset(url)
        result = scan_url(url, own_asset=own,
                          allow_private=self.allow_private,
                          min_interval_s=self.min_interval_s)
        return self._finish("web", result, own_asset=own,
                            requester=requester)

    # ── repo ─────────────────────────────────────────────────
    def run_repo_audit(self, target: str, *,
                       requester: str = "") -> dict[str, Any]:
        own = self.book.is_own_asset(target)
        result = audit_repo(target, own_asset=own,
                            osv_enabled=self.osv_enabled,
                            clone_root=self.book.root)
        return self._finish("repo", result, own_asset=own,
                            requester=requester)

    # ── shared tail ──────────────────────────────────────────
    def _finish(self, kind: str, result: dict[str, Any], *,
                own_asset: bool, requester: str) -> dict[str, Any]:
        graded = grade_findings(result.get("findings") or [])
        result = {**result, "findings": graded}
        summary = summarize(graded)
        stamp = time.strftime("%Y%m%d-%H%M%S",
                              time.gmtime(result.get("started_at")
                                          or time.time()))
        slug = _slug(str(result.get("target") or "target"))
        out_dir = self.book.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{kind}-{slug}-{stamp}.md"
        report_path.write_text(
            render_report_md(result, graded, own_asset=own_asset),
            encoding="utf-8")

        draft_path = ""
        contacts: list[str] = []
        if graded:
            contacts = self._contacts(kind, result)
            if contacts:
                draft = draft_disclosure(
                    target=str(result.get("target") or ""),
                    findings=graded, contacts=contacts,
                    report_name=report_path.name)
                draft_file = out_dir / f"disclosure-{slug}-{stamp}.md"
                draft_file.write_text(draft, encoding="utf-8")
                draft_path = str(draft_file)

        entry = self.book.record({
            "kind": kind, "target": result.get("target"),
            "own_asset": own_asset, "requester": requester,
            "checks_run": result.get("checks_run") or [],
            "notes": result.get("notes") or [],
            "findings": graded,
            "summary": summary,
            "report_path": str(report_path),
            "draft_path": draft_path,
            "contacts": contacts,
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
        })
        return {"audit_id": entry["id"], "kind": kind,
                "target": result.get("target"),
                "own_asset": own_asset, "findings": graded,
                "summary": summary,
                "report_path": str(report_path),
                "draft_path": draft_path, "contacts": contacts,
                "notes": result.get("notes") or [],
                "checks_run": result.get("checks_run") or []}

    def _contacts(self, kind: str,
                  result: dict[str, Any]) -> list[str]:
        target = str(result.get("target") or "")
        try:
            if kind == "web":
                fetcher = Fetcher(
                    allow_private=self.allow_private,
                    min_interval_s=0.0)
                parts = urllib.parse.urlsplit(target)
                base = f"{parts.scheme}://{parts.netloc}"
                return discover_contacts_web(fetcher.get, base)
            path = Path(target).expanduser()
            if path.is_dir():
                return discover_contacts_repo(path, target)
            return discover_contacts_repo(Path("."), target) \
                if "github.com" in target else []
        except Exception:  # noqa: BLE001 - contact discovery is
            return []             # best-effort, never fatal


def run_job(service: AuditService, job: dict[str, Any]) -> dict[str, Any]:
    """Execute one queued job; returns the outcome dict the
    book stores and the driver delivers."""
    if job.get("kind") == "web":
        return service.run_web_scan(str(job.get("target") or ""),
                                    requester=str(
                                        job.get("requester") or ""))
    return service.run_repo_audit(str(job.get("target") or ""),
                                  requester=str(
                                      job.get("requester") or ""))
