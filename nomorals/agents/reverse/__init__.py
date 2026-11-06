"""Reverse — Devon's reverse-engineering organ (Pass 1).

Mined from REA (morluto/rea) at code level, then built as Devon's
own organ under the standing outdo standard: REA's evidence
discipline is the floor, and Devon's version exceeds it where REA
stops — findings journal into her decision journal and memory,
residual unknowns are tracked records she can resume, and the
engines live behind her one tool registry so the brain's tool loop,
the ``/re`` chat surface, ``nm re`` and external agents over her MCP
server all reach the same organ.

Pass 1: the Evidence core (content-addressed Evidence records with
confidence × authority + limitations on the record, and the
residual-unknowns registry with a store-enforced close lifecycle)
plus the native static spine — built-in ELF / PE / Mach-O inspectors
emitting Evidence, never bare text. The ELF census generalizes
ApkForge's proven ELF parser; no external engines run in this pass
(the self-provisioned Ghidra rung is Pass 2).
"""

from __future__ import annotations

from typing import Any

from .engine import (
    ReEngine,
    capability_matrix,
    render_audit,
    render_evidence_card,
    render_inspect_card,
    render_status,
    render_unknowns,
)
from .evidence import AUTHORITIES, CONFIDENCES, Evidence, EvidenceStore, Unknown, UnknownCloseError

__all__ = [
    "ReEngine", "capability_matrix",
    "Evidence", "EvidenceStore", "Unknown", "UnknownCloseError",
    "AUTHORITIES", "CONFIDENCES",
    "render_status", "render_inspect_card", "render_evidence_card",
    "render_unknowns", "render_audit",
    "register",
]


def register(registry: Any) -> None:
    """Register the ``re`` read tool and the ``re_inspect`` run tool.

    ``re`` reads the evidence store (status, evidence records with
    their provenance chains, the unknowns registry, the unknowns
    audit). ``re_inspect`` performs inspections — each one records a
    chain of Evidence and registers the residual unknowns the parse
    raises — and closes unknowns under the store's lifecycle rules.
    """
    from ...core.policy import Capability

    context = registry.context

    @registry.register(
        "re",
        description=(
            "Reverse-engineering evidence store, read side. "
            "action=status (engine capability matrix, probe-executed "
            "for the built-ins, plus store counts) | evidence (one "
            "Evidence record by id or unique prefix, with its "
            "provenance chain; empty id lists the most recent "
            "records) | unknowns (the residual-unknown registry: "
            "status=open|resolved|contradicted, empty status lists "
            "all) | audit (unknowns audit: counts, the strongest "
            "evidence on record, and which open unknowns no recorded "
            "evidence could close yet). Evidence records are "
            "content-addressed (ev:<sha256>) and carry confidence, "
            "authority and limitations on the record itself."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — status|evidence|unknowns|audit",
            "evidence_id": "str — evidence id or unique prefix "
                           "(action=evidence)",
            "status": "str — unknowns filter: open|resolved|"
                      "contradicted (action=unknowns)",
            "limit": "str — max rows for lists (default 20)",
        },
    )
    def re_tool(*, action: str = "status", evidence_id: str = "",
                status: str = "", limit: str = "20") -> dict[str, Any]:
        act = (action or "status").strip().lower()
        engine = ReEngine(context)
        try:
            n = int(limit or 20)
        except (TypeError, ValueError):
            n = 20
        if act == "status":
            return engine.status()
        if act == "evidence":
            if (evidence_id or "").strip():
                return engine.evidence(evidence_id.strip())
            return engine.evidence_list(limit=n)
        if act == "unknowns":
            return engine.unknowns(status=(status or "open"), limit=n)
        if act == "audit":
            return engine.audit()
        return {"error": f"unknown re action {act!r} — "
                         "status|evidence|unknowns|audit"}

    @registry.register(
        "re_inspect",
        description=(
            "Reverse-engineering inspections + unknown closes, write "
            "side. action=inspect (path = a local binary or artifact "
            "file: the built-in static spine parses ELF, PE and "
            "Mach-O — headers, sections, symbols, imports, dylibs, "
            "hardening — and records the findings as a linked chain "
            "of Evidence records, registers the residual unknowns "
            "the parse raises, and journals the finding to the "
            "decision journal and memory when bound; static only, "
            "nothing is executed) | close (unknown_id + status="
            "resolved|contradicted: resolved takes a resolution "
            "text, contradicted takes at least one contradicting "
            "evidence id in evidence_ids, and the stated "
            "resolution_authority / resolution_confidence must meet "
            "what the unknown requires — the store refuses a weaker "
            "close and says what it takes)."
        ),
        capability=Capability.DB_WRITE,
        parameters={
            "action": "str — inspect|close",
            "path": "str — file to inspect (action=inspect)",
            "unknown_id": "str — unknown id or unique prefix "
                          "(action=close)",
            "status": "str — resolved|contradicted (action=close)",
            "resolution": "str — the answer, in words "
                          "(action=close, resolved)",
            "evidence_ids": "str — comma-separated evidence ids "
                            "backing the close (action=close)",
            "resolution_authority": "str — shipped-artifact|"
                "controlled-replay|historical-reference|"
                "external-service|analyst-inference",
            "resolution_confidence": "str — observed|derived|"
                                     "inferred",
        },
    )
    def re_inspect_tool(*, action: str = "inspect", path: str = "",
                        unknown_id: str = "", status: str = "",
                        resolution: str = "", evidence_ids: str = "",
                        resolution_authority: str = "",
                        resolution_confidence: str = ""
                        ) -> dict[str, Any]:
        act = (action or "inspect").strip().lower()
        engine = ReEngine(context)
        if act == "inspect":
            return engine.inspect_path(path)
        if act == "close":
            ids = tuple(x.strip() for x in (evidence_ids or "").split(",")
                        if x.strip())
            return engine.close_unknown(
                unknown_id, status=(status or "").strip().lower(),
                resolution=resolution,
                resolution_authority=(resolution_authority
                                      or "analyst-inference"),
                resolution_confidence=(resolution_confidence
                                       or "inferred"),
                evidence_ids=ids)
        return {"error": f"unknown re_inspect action {act!r} — "
                         "inspect|close"}
