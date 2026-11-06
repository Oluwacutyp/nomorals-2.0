"""ReEngine — Devon's reverse-engineering organ (Pass 1).

One facade, every surface a thin shell over it (the ApkForge/betdesk
idiom): the ``re`` / ``re_inspect`` registry tools, the ``/re`` chat
command and ``nm re`` all call these methods.

Pass 1 scope, per the mining brief: the Evidence core
(:mod:`.evidence`) plus the built-in static spine
(:mod:`.inspectors`) — ELF / PE / Mach-O parsed by Devon's own
parsers, no external engine. An inspection lands as a chain of
Evidence records (identify → structure → hardening → strings →
entropy) with each record linked to the one it rests on, registers
the residual unknowns the parse itself raises, and journals the
finding into the decision journal + memory when those are bound —
findings live in Devon's systems from day one, not in a side file.
Deep engines (the Ghidra rung and friends) are Pass 2; the
capability matrix says so, with the reason, instead of pretending.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

from .evidence import EvidenceStore, UnknownCloseError
from .failures import failure_count, record_failure
from .inspectors import PROVIDER_ID, PROVIDER_NAME, PROVIDER_VERSION, elf_info, inspect_bytes

__all__ = [
    "ReEngine", "capability_matrix",
    "render_status", "render_inspect_card", "render_evidence_card",
    "render_unknowns", "render_audit",
]

#: Refuse to slurp an unbounded file into memory for a static parse.
_MAX_INSPECT_BYTES = 256 * 1024 * 1024


def capability_matrix() -> list[dict[str, Any]]:
    """Per-engine availability, probe-executed where built-in.

    Built-in rows are proven by running the real parser over a
    minimal real header built here — an engine that cannot parse its
    own probe is reported unavailable, never assumed (the ApkForge
    toolchain ethos). External rows are Pass 2+ rungs: unavailable
    today, with the pass that lands them named on the row.
    """
    rows: list[dict[str, Any]] = []

    elf_probe = (
        b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\x00" * 8
        + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, 0, 0, 64,
                      0, 0, 0, 0, 0))
    rows.append({
        "engine": "elf", "kind": "built-in", "available":
        elf_info(elf_probe) is not None,
        "note": "ELF structure, symbols, needed libraries, hardening "
                "(generalizes ApkForge's census parser)"})

    pe_probe = bytearray(0x40 + 24)
    pe_probe[0:2] = b"MZ"
    struct.pack_into("<I", pe_probe, 0x3C, 0x40)
    pe_probe[0x40:0x44] = b"PE\x00\x00"
    pe_facts = inspect_bytes(bytes(pe_probe)).facts
    rows.append({
        "engine": "pe", "kind": "built-in",
        "available": pe_facts.get("pe_signature_found") is True,
        "note": "PE headers, sections, imports/exports, certificate "
                "table, hardening flags"})

    macho_probe = struct.pack("<IIIIIIII", 0xFEEDFACF, 0x01000007, 3,
                              2, 0, 0, 0x200000, 0)
    macho_facts = inspect_bytes(macho_probe).facts
    rows.append({
        "engine": "mach-o", "kind": "built-in",
        "available": macho_facts.get("macho_header_readable") is True,
        "note": "Mach-O + fat/universal: load commands, dylibs, "
                "symbols, UUID, code-signature presence"})

    rows.append({
        "engine": "strings+entropy", "kind": "built-in",
        "available": True,
        "note": "printable-string census (ASCII + UTF-16LE) and "
                "Shannon entropy, recorded as evidence on every "
                "inspection"})
    for engine, note in (
            ("ghidra-headless",
             "Pass 2 — self-provisioned deep decompile rung "
             "(pseudocode, call graph, xrefs); not built yet"),
            ("android-graph",
             "Pass 2 — class/method search + reference tracing over "
             "ApkForge dossiers; not built yet"),
            ("rizin-objdump",
             "Pass 2 — one-shot disassembly rung for light "
             "profiles; not built yet"),
            ("binwalk-unblob",
             "Pass 3 — firmware extraction + region analysis; "
             "not built yet"),
            ("tshark",
             "Pass 3 — packet-capture summaries + field "
             "inspection; not built yet"),
            ("frida-gdb",
             "dynamic rung — gated on the owner's sandbox/consent "
             "posture for executing targets; not built")):
        rows.append({"engine": engine, "kind": "external",
                     "available": False, "note": note})
    return rows


class ReEngine:
    """One RE organ instance.

    ``context`` is the runtime context when bound (its ``db``,
    ``settings`` and ``memory`` are the live seams); CLI and tests
    may pass ``db`` / ``journal`` / ``memory`` directly. With no
    database reachable the store falls back to an in-memory database
    and :meth:`status` says so — an inspection that evaporates at
    process exit is never dressed up as a durable record.
    """

    def __init__(self, context: Any = None, *, db: Any = None,
                 journal: Any = None, memory: Any = None) -> None:
        self.context = context
        self._db = db if db is not None else getattr(context, "db", None)
        self._ephemeral = False
        if self._db is None:
            from ...storage.db import Database

            self._db = Database(":memory:")
            self._ephemeral = True
        elif getattr(self._db, "path", None) in (None, "", ":memory:"):
            # a caller-supplied in-memory database (Database stores
            # path=None for ":memory:") is still ephemeral storage —
            # status reports the storage truth either way
            self._ephemeral = True
        self._store: EvidenceStore | None = None
        self._journal = journal
        self._journal_tried = journal is not None
        self._memory = (memory if memory is not None
                        else getattr(context, "memory", None))

    @property
    def store(self) -> EvidenceStore:
        if self._store is None:
            self._store = EvidenceStore(self._db)
        return self._store

    # ── wiring: decision journal + memory ────────────────────────────
    def _bound_journal(self) -> Any:
        if self._journal_tried:
            return self._journal
        self._journal_tried = True
        settings = getattr(self.context, "settings", None)
        home = getattr(settings, "home_path", None)
        if home is None:
            return None
        try:
            from ...decisions.journal import DecisionJournal

            self._journal = DecisionJournal(
                str(Path(home) / "data" / "decisions.json"))
        except Exception as exc:  # noqa: BLE001 - wiring is best-effort
            record_failure("reverse.engine.journal_bind", exc)
            self._journal = None
        return self._journal

    # ── status / capability ──────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        audit = self.store.audit()
        return {
            "engines": capability_matrix(),
            "evidence_records": audit["evidence_records"],
            "unknowns_open": audit["unknowns_open"],
            "unknowns_resolved": audit["unknowns_resolved"],
            "unknowns_contradicted": audit["unknowns_contradicted"],
            "store": ("ephemeral (in-memory — no context database "
                      "bound)" if self._ephemeral else "durable"),
            "failures_recorded": failure_count(),
        }

    # ── inspection ───────────────────────────────────────────────────
    def inspect_path(self, path: str) -> dict[str, Any]:
        """Static-inspect one file → Evidence chain + unknowns.

        Returns a structured dict; ``error`` is set (and nothing is
        recorded) when the target cannot be read at all.
        """
        target = Path(path or "").expanduser()
        if not target.is_file():
            return {"error": f"no readable file at {path!r}"}
        size = target.stat().st_size
        if size > _MAX_INSPECT_BYTES:
            return {"error": f"{target.name} is {size} bytes — the "
                             "built-in static spine reads at most "
                             f"{_MAX_INSPECT_BYTES} bytes in Pass 1"}
        try:
            data = target.read_bytes()
        except OSError as exc:
            return {"error": f"could not read {target}: {exc}"}
        sha = hashlib.sha256(data).hexdigest()
        insp = inspect_bytes(data)
        store = self.store
        common = {
            "subject_sha256": sha, "subject_name": target.name,
            "subject_size": size, "provider_id": PROVIDER_ID,
            "provider_name": PROVIDER_NAME,
            "provider_version": PROVIDER_VERSION,
        }
        identify = store.record(
            predicate_type="format.identify", operation="detect_format",
            parameters={"format": insp.format},
            result={"format": insp.format, "size": size,
                    "magic_hex": data[:16].hex()},
            limitations=(insp.limitations if insp.format == "unknown"
                         else []),
            locations=[{"type": "artifact-path",
                        "value": str(target)}],
            **common)
        evidence = [identify]
        structure = None
        if insp.format != "unknown":
            structure = store.record(
                predicate_type=f"{insp.format.split('-')[0]}.structure",
                operation="inspect",
                parameters={"format": insp.format},
                result=insp.facts, limitations=insp.limitations,
                locations=[{"type": "artifact-path",
                            "value": str(target)}],
                links=[identify.evidence_id], **common)
            evidence.append(structure)
        anchor = structure or identify
        hardening = insp.facts.get("hardening")
        if hardening is None and insp.format.startswith("macho"):
            hardening = {
                "pie": bool(insp.facts.get("pie")),
                "no_heap_execution": bool(
                    insp.facts.get("no_heap_execution")),
                "allow_stack_execution": bool(
                    insp.facts.get("allow_stack_execution")),
            }
        if hardening:
            evidence.append(store.record(
                predicate_type="binary.hardening", operation="derive",
                parameters={"format": insp.format}, result=hardening,
                confidence="derived", links=[anchor.evidence_id],
                limitations=["hardening verdicts are derived from "
                             "header flags alone; they describe the "
                             "artifact's declared posture, not a "
                             "runtime check"],
                **common))
        evidence.append(store.record(
            predicate_type="binary.strings", operation="census",
            parameters={"format": insp.format}, result=insp.strings,
            links=[identify.evidence_id], **common))
        evidence.append(store.record(
            predicate_type="binary.entropy", operation="measure",
            parameters={"format": insp.format}, result=insp.entropy,
            confidence="derived", links=[identify.evidence_id],
            **common))
        unknowns = [
            store.open_unknown(
                seed["question"], subject_sha256=sha,
                raised_by=anchor.evidence_id,
                supporting=[anchor.evidence_id],
                required_authority=seed["required_authority"],
                required_confidence=seed["required_confidence"])
            for seed in insp.unknowns
        ]
        journaled = self._journal_finding(
            target.name, insp.format, sha, evidence, unknowns)
        remembered = self._remember_finding(
            target.name, insp.format, sha, identify, evidence, unknowns)
        result: dict[str, Any] = {
            "subject": {"name": target.name, "path": str(target),
                        "sha256": sha, "size": size},
            "format": insp.format,
            "facts": insp.facts,
            "evidence_ids": [e.evidence_id for e in evidence],
            "evidence": [e.to_dict() for e in evidence],
            "unknowns": [u.to_dict() for u in unknowns],
            "journaled": journaled,
            "remembered": remembered,
        }
        result["card_text"] = render_inspect_card(result)
        return result

    def _journal_finding(self, name: str, fmt: str, sha: str,
                         evidence: list, unknowns: list) -> bool:
        journal = self._bound_journal()
        if journal is None:
            return False
        try:
            journal.record(
                decision=f"RE inspection: {name}",
                context=(f"format {fmt} · sha256 {sha} · evidence "
                         + ", ".join(e.evidence_id for e in evidence)
                         + f" · {len(unknowns)} residual unknown(s) "
                           "registered"),
                tags=["re", fmt])
            return True
        except Exception as exc:  # noqa: BLE001 - best-effort wiring
            record_failure("reverse.engine.journal_record", exc)
            return False

    def _remember_finding(self, name: str, fmt: str, sha: str,
                          identify: Any, evidence: list,
                          unknowns: list) -> bool:
        memory = self._memory
        if memory is None:
            return False
        try:
            memory.remember(
                f"Reverse-engineering inspection of {name} "
                f"({fmt}, sha256 {sha[:16]}…): {len(evidence)} "
                f"evidence records, {len(unknowns)} residual "
                f"unknown(s) open. Root evidence "
                f"{identify.evidence_id}.",
                kind="episode", source=f"re:{identify.evidence_id}")
            return True
        except Exception as exc:  # noqa: BLE001 - best-effort wiring
            record_failure("reverse.engine.memory_remember", exc)
            return False

    # ── evidence + unknowns queries ──────────────────────────────────
    def evidence(self, ref: str) -> dict[str, Any]:
        rec = self.store.get(ref)
        if rec is None:
            return {"error": f"no evidence matches {ref!r}"}
        chain = self.store.provenance_chain(rec.evidence_id)
        return {"record": rec.to_dict(),
                "chain": [c.to_dict() for c in chain]}

    def evidence_list(self, limit: int = 20) -> dict[str, Any]:
        return {"evidence": [r.to_dict()
                             for r in self.store.list_recent(limit)]}

    def unknowns(self, status: str | None = "open",
                 limit: int = 50) -> dict[str, Any]:
        rows = self.store.list_unknowns(
            status=status or None, limit=limit)
        return {"unknowns": [u.to_dict() for u in rows]}

    def close_unknown(self, unknown_id: str, *, status: str,
                      resolution: str = "",
                      resolution_authority: str = "analyst-inference",
                      resolution_confidence: str = "inferred",
                      evidence_ids: tuple[str, ...] | list[str] = ()
                      ) -> dict[str, Any]:
        try:
            rec = self.store.close_unknown(
                unknown_id, status=status, resolution=resolution,
                resolution_authority=resolution_authority,
                resolution_confidence=resolution_confidence,
                evidence_ids=evidence_ids)
        except UnknownCloseError as exc:
            return {"error": str(exc)}
        return {"unknown": rec.to_dict()}

    def audit(self) -> dict[str, Any]:
        return self.store.audit()


# ── renders (plain text; surfaces own the presentation layer) ────────

def render_status(status: dict[str, Any]) -> str:
    lines = ["reverse engineering — evidence core + built-in static "
             "spine (Pass 1)"]
    for row in status["engines"]:
        mark = "yes" if row["available"] else "no"
        lines.append(f"  [{mark}] {row['engine']} ({row['kind']}) — "
                     f"{row['note']}")
    lines.append(
        f"evidence: {status['evidence_records']} record(s) · "
        f"unknowns: {status['unknowns_open']} open, "
        f"{status['unknowns_resolved']} resolved, "
        f"{status['unknowns_contradicted']} contradicted")
    lines.append(f"store: {status['store']}")
    if status.get("failures_recorded"):
        lines.append(f"failures recorded: "
                     f"{status['failures_recorded']} (see the reverse "
                     "failure ledger)")
    return "\n".join(lines)


def render_inspect_card(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"⚠️ {result['error']}"
    subj = result["subject"]
    facts = result.get("facts") or {}
    lines = [
        f"🔍 {subj['name']} — {result['format']}",
        f"sha256 {subj['sha256'][:16]}… · {subj['size']} bytes",
    ]
    for key in ("machine", "cpu", "object_type", "pe_format",
                "filetype", "entry_point", "interpreter", "soname",
                "uuid"):
        if facts.get(key):
            lines.append(f"{key}: {facts[key]}")
    if facts.get("needed_libraries"):
        lines.append("needs: " + ", ".join(facts["needed_libraries"]))
    if facts.get("imported_dlls"):
        lines.append("imports from: "
                     + ", ".join(facts["imported_dlls"][:8]))
    if facts.get("hardening"):
        hard = ", ".join(f"{k}={v}" for k, v in
                         facts["hardening"].items())
        lines.append(f"hardening: {hard}")
    lines.append(f"evidence: {len(result['evidence_ids'])} record(s) — "
                 f"root {result['evidence_ids'][0]}")
    if result["unknowns"]:
        lines.append(f"residual unknowns: {len(result['unknowns'])} "
                     "registered")
        for unk in result["unknowns"]:
            lines.append(f"  ? {unk['question']} [{unk['unknown_id']}]")
    else:
        lines.append("residual unknowns: none raised by this parse")
    lines.append("journaled to decisions: "
                 + ("yes" if result["journaled"] else "no")
                 + " · remembered: "
                 + ("yes" if result["remembered"] else "no"))
    return "\n".join(lines)


def render_evidence_card(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        return f"⚠️ {payload['error']}"
    rec = payload["record"]
    lines = [
        f"evidence {rec['evidence_id']}",
        f"subject: {rec['subject_name']} "
        f"(sha256 {rec['subject_sha256'][:16]}…)",
        f"claim: {rec['predicate_type']} / {rec['operation']} "
        f"via {rec['provider_id']}",
        f"confidence: {rec['confidence']} · authority: "
        f"{rec['authority']}",
    ]
    if rec["limitations"]:
        lines.append("limitations:")
        lines.extend(f"  - {lim}" for lim in rec["limitations"])
    chain = payload.get("chain") or []
    if len(chain) > 1:
        lines.append("provenance chain:")
        lines.extend(f"  ← {c['evidence_id']} "
                     f"({c['predicate_type']})" for c in chain[1:])
    return "\n".join(lines)


def render_unknowns(unknowns: list[dict[str, Any]],
                    title: str = "residual unknowns") -> str:
    if not unknowns:
        return f"{title}: none"
    lines = [f"{title}: {len(unknowns)}"]
    for unk in unknowns:
        lines.append(
            f"  ? [{unk['status']}] {unk['question']} "
            f"[{unk['unknown_id']}] — closing takes authority "
            f"{unk['required_authority']} + confidence "
            f"{unk['required_confidence']}")
        if unk.get("resolution"):
            lines.append(f"    resolution: {unk['resolution']}")
    return "\n".join(lines)


def render_audit(audit: dict[str, Any]) -> str:
    lines = [
        "unknowns audit",
        f"evidence records: {audit['evidence_records']}",
        f"open: {audit['unknowns_open']} · resolved: "
        f"{audit['unknowns_resolved']} · contradicted: "
        f"{audit['unknowns_contradicted']}",
        f"strongest evidence on record: authority "
        f"{audit['strongest_evidence_authority']} · confidence "
        f"{audit['strongest_evidence_confidence']}",
    ]
    if audit["open_unclosable_on_current_evidence"]:
        lines.append("open and NOT closable on current evidence "
                     "(they name the next rung):")
        lines.extend(f"  ? {uid}" for uid in
                     audit["open_unclosable_on_current_evidence"])
    if audit["open_without_supporting_evidence"]:
        lines.append("open with no supporting evidence attached:")
        lines.extend(f"  ? {uid}" for uid in
                     audit["open_without_supporting_evidence"])
    if not (audit["open_unclosable_on_current_evidence"]
            or audit["open_without_supporting_evidence"]):
        lines.append("every open unknown is closable on current "
                     "evidence and carries support")
    return "\n".join(lines)
