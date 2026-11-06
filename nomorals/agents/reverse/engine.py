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

Pass 2 adds the self-provisioned deep providers behind the same
facade: the Ghidra headless rung (Devon provisions Ghidra + its
JDK herself — :mod:`.provision`, :mod:`.ghidra`), the binutils
light rung (:mod:`.lightbin`), and the Android reference graph
over ApkForge's jadx output (:mod:`.android_graph`). An inspection
now routes the target to the deepest rung that can answer and
says which rung answered; where no deep rung can, the gap is a
residual unknown naming the rung that closes it.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

from . import android_graph, ghidra, lightbin, provision
from .evidence import EvidenceStore, UnknownCloseError
from .failures import failure_count, record_failure
from .inspectors import PROVIDER_ID, PROVIDER_NAME, PROVIDER_VERSION, elf_info, inspect_bytes

__all__ = [
    "ReEngine", "capability_matrix",
    "render_status", "render_inspect_card", "render_evidence_card",
    "render_unknowns", "render_audit", "render_provision",
    "render_graph_answer",
]

#: Refuse to slurp an unbounded file into memory for a static parse.
_MAX_INSPECT_BYTES = 256 * 1024 * 1024


def capability_matrix(root: Any = None) -> list[dict[str, Any]]:
    """Per-engine availability, probe-executed where built-in.

    Built-in rows are proven by running the real parser over a
    minimal real header built here — an engine that cannot parse its
    own probe is reported unavailable, never assumed (the ApkForge
    toolchain ethos). Pass 2 rows carry REAL probe state from the
    provisioning ledger: ``state`` is ready (probed, usable now),
    provisionable (pinned, lands on demand — Devon's outdo over
    REA's bring-your-own), or unavailable with the reason on the
    row. ``available`` means ready *now*. ``root`` overrides the
    tools root (tests, alternate profiles).
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
    # ── Pass 2: the self-provisioned deep rungs, probe-true ────
    jdk = provision.jdk_state(root)
    rows.append({
        "engine": "temurin-jdk", "kind": "provisioned",
        "available": jdk["available"], "state": jdk["state"],
        "note": "the JDK the Ghidra rung stands on — " + jdk["detail"]})
    gh = provision.ghidra_state(root)
    rows.append({
        "engine": "ghidra-headless", "kind": "provisioned",
        "available": gh["available"], "state": gh["state"],
        "note": "self-provisioned deep decompile rung (functions, "
                "call graph, imports, pseudocode) — " + gh["detail"]})
    light = lightbin.probe()
    rows.append({
        "engine": "rizin-objdump", "kind": "one-shot",
        "available": light["available"],
        "state": "ready" if light["available"] else "unavailable",
        "note": "light one-shot triage rung (readelf-class: "
                "header, dynamic deps, dynamic symbols) — "
                + light["detail"]})
    rows.append({
        "engine": "android-graph", "kind": "built-in",
        "available": True, "state": "ready",
        "note": "class/method/reference graph over jadx output "
                "(who calls X, what Y touches) — consumes ApkForge "
                "jadx sources trees; an APK is decompiled into one "
                "through ApkForge's own provisioned jadx"})
    for engine, note in (
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
                 journal: Any = None, memory: Any = None,
                 tools_root: Any = None) -> None:
        self.context = context
        self._tools_root = (Path(tools_root) if tools_root
                            else None)
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
            "engines": capability_matrix(self._tools_root),
            "evidence_records": audit["evidence_records"],
            "unknowns_open": audit["unknowns_open"],
            "unknowns_resolved": audit["unknowns_resolved"],
            "unknowns_contradicted": audit["unknowns_contradicted"],
            "store": ("ephemeral (in-memory — no context database "
                      "bound)" if self._ephemeral else "durable"),
            "failures_recorded": failure_count(),
        }

    # ── inspection ───────────────────────────────────────────────────
    def inspect_path(self, path: str, deep: str = "auto"
                     ) -> dict[str, Any]:
        """Static-inspect one file → Evidence chain + unknowns.

        ``deep`` governs the Pass 2 rungs: ``auto`` routes the
        target to the deepest rung that can answer *now* (the
        provisioned Ghidra rung, else the binutils light rung for
        ELF, else the built-in spine alone); ``provision`` first
        runs the self-provisioning drive for any missing engine,
        then routes; ``off`` keeps the Pass 1 spine only. The
        result names the rung that answered (``rung``) — and when
        no deep rung could, a residual unknown names the rung that
        closes the gap. Returns a structured dict; ``error`` is set
        (and nothing is recorded) when the target cannot be read
        at all.
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
        routed = self._deep_route(
            target, insp.format, sha, size, identify, deep)
        for eid in routed["evidence_ids"]:
            rec = store.get(eid)
            if rec is not None:
                evidence.append(rec)
        unknowns.extend(routed["unknowns"])
        journaled = self._journal_finding(
            target.name, insp.format, sha, evidence, unknowns)
        remembered = self._remember_finding(
            target.name, insp.format, sha, identify, evidence, unknowns)
        result: dict[str, Any] = {
            "subject": {"name": target.name, "path": str(target),
                        "sha256": sha, "size": size},
            "format": insp.format,
            "facts": insp.facts,
            "rung": routed["rung"],
            "deep": routed["summary"],
            "evidence_ids": [e.evidence_id for e in evidence],
            "evidence": [e.to_dict() for e in evidence],
            "unknowns": [u.to_dict() for u in unknowns],
            "journaled": journaled,
            "remembered": remembered,
        }
        if routed.get("provision"):
            result["provision"] = routed["provision"]
        result["card_text"] = render_inspect_card(result)
        return result

    # ── Pass 2: deep-rung routing ────────────────────────────────────
    def _deep_route(self, target: Path, fmt: str, sha: str,
                    size: int, identify: Any, deep: str
                    ) -> dict[str, Any]:
        """Route one inspected binary to the deepest rung that can
        answer. Returns {rung, summary, evidence_ids, unknowns,
        provision?}. A rung that fails mid-run is recorded in the
        failure ledger and named in the summary — the route then
        degrades honestly to the next rung rather than swallowing
        the failure."""
        out: dict[str, Any] = {
            "rung": "built-in static spine", "summary": {},
            "evidence_ids": [], "unknowns": []}
        if deep == "off" or fmt not in ("elf", "pe", "macho",
                                        "macho-fat"):
            return out
        store = self.store
        common = {"subject_sha256": sha, "subject_name": target.name,
                  "subject_size": size, "subject_path": str(target),
                  "link_to": identify.evidence_id}
        gh_state = provision.ghidra_state(self._tools_root)
        if deep == "provision" and not gh_state["available"]:
            out["provision"] = self.provision_engines()
            gh_state = provision.ghidra_state(self._tools_root)
        if gh_state["available"]:
            try:
                export = ghidra.analyze(target,
                                        root=self._tools_root)
                folded = ghidra.fold_evidence(store, export=export,
                                              **common)
                out["rung"] = "ghidra-headless"
                out["summary"] = {
                    "engine": "ghidra-headless",
                    "version": gh_state["version"],
                    "functions": export.get("functions_total"),
                    "decompiled": export.get("decompiled_count"),
                }
                out["evidence_ids"] = folded["evidence_ids"]
                fetched = [store.get_unknown(u["unknown_id"])
                           for u in folded["unknowns"]]
                out["unknowns"] = [u for u in fetched
                                   if u is not None]
                return out
            except ghidra.GhidraError as exc:
                record_failure("reverse.engine.ghidra_route", exc)
                out["summary"] = {"engine": "ghidra-headless",
                                  "error": str(exc)}
        if fmt == "elf":
            light = lightbin.probe()
            if light["available"]:
                try:
                    facts = lightbin.triage(target)
                    folded = lightbin.fold_evidence(store, **common,
                                                    facts=facts)
                    summary = {
                        "engine": "binutils-triage",
                        "needed_libraries":
                            facts.get("needed_libraries"),
                        "defined_function_count":
                            facts.get("defined_function_count"),
                        "imported_symbol_count":
                            facts.get("imported_symbol_count"),
                    }
                    if out["summary"]:
                        summary["ghidra"] = out["summary"]
                    out["summary"] = summary
                    out["rung"] = "binutils-triage"
                    out["evidence_ids"] = folded["evidence_ids"]
                    return out
                except RuntimeError as exc:
                    record_failure("reverse.engine.light_route", exc)
                    out["summary"] = {
                        **out["summary"],
                        "light_error": str(exc)}
        # no deep rung could answer: the gap is a residual unknown
        # naming the rung that closes it (the audit's whole point)
        out["unknowns"] = [store.open_unknown(
            f"the deep behavior of {target.name} (function-level "
            "semantics, decompiled pseudocode) is unestablished — "
            "no deep rung answered on this device; the "
            "self-provisioned Ghidra rung closes this",
            subject_sha256=sha, raised_by=identify.evidence_id,
            supporting=[identify.evidence_id],
            required_authority="shipped-artifact",
            required_confidence="derived")]
        return out

    # ── Pass 2: provisioning drive ───────────────────────────────────
    def provision_engines(self) -> dict[str, Any]:
        """Run the self-provisioning drive for the deep rungs
        (JDK → Ghidra) and report each engine's landing state with
        its attempts. REA makes the human fetch engines; Devon
        fetches her own."""
        return provision.provision_all(root=self._tools_root)

    # ── Pass 2: Android reference graph ──────────────────────────────
    def android_sources(self, path: str) -> dict[str, Any]:
        """Resolve a path to a jadx sources root: a directory is
        taken as jadx output directly; an APK goes through
        ApkForge (add + jadx) on its own provisioned toolchain."""
        target = Path(path or "").expanduser()
        if target.is_dir():
            try:
                return {"sources": str(
                    android_graph.sources_root_of(target)),
                    "via": "jadx output tree"}
            except android_graph.GraphError as exc:
                return {"error": str(exc)}
        if target.is_file() and target.suffix.lower() in (
                ".apk", ".xapk", ".apks"):
            from ..apkforge import ApkForge

            forge = ApkForge(self.context)
            added = forge.add(str(target))
            if added.get("error"):
                return {"error": f"ApkForge intake failed: "
                                 f"{added['error']}"}
            ref = (added.get("project") or {}).get("id", "")
            jres = forge.jadx(ref)
            if jres.get("error"):
                return {"error": f"jadx decompile failed: "
                                 f"{jres['error']}"}
            try:
                sources = str(android_graph.sources_root_of(
                    jres["jadx"]))
            except android_graph.GraphError as exc:
                return {"error": str(exc)}
            return {"sources": sources,
                    "via": f"ApkForge jadx (project {ref})",
                    "project": ref}
        return {"error": f"{path!r} is neither a jadx output "
                         "directory nor an APK file"}

    def android_graph_build(self, path: str) -> dict[str, Any]:
        """Build the reference graph over an APK / jadx tree and
        record its census as Evidence."""
        resolved = self.android_sources(path)
        if resolved.get("error"):
            return resolved
        try:
            graph = android_graph.build_graph(resolved["sources"])
        except android_graph.GraphError as exc:
            return {"error": str(exc)}
        subject = Path(path).expanduser()
        name = subject.name
        if subject.is_file():
            data = subject.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            size = len(data)
        else:
            sha = hashlib.sha256(
                resolved["sources"].encode("utf-8")).hexdigest()
            size = 0
        folded = android_graph.fold_evidence(
            self.store, subject_sha256=sha, subject_name=name,
            graph=graph)
        return {"sources": resolved["sources"],
                "via": resolved["via"], "stats": graph["stats"],
                "collisions": graph["collisions"],
                "evidence_ids": folded["evidence_ids"],
                "graph_evidence_id": folded["graph_evidence_id"],
                "_sha": sha, "_name": name, "_size": size}

    def android_query(self, path: str, query: str,
                      target: str) -> dict[str, Any]:
        """Answer one graph question as a recorded Evidence record:
        callers | callees | class | search over the graph of an
        APK / jadx tree."""
        built = self.android_graph_build(path)
        if built.get("error"):
            return built
        try:
            graph = android_graph.build_graph(built["sources"])
            if query == "callers":
                answer = android_graph.callers(graph, target)
                operation = "graph.callers_of"
            elif query == "callees":
                answer = android_graph.callees(graph, target)
                operation = "graph.callees_of"
            elif query == "class":
                answer = android_graph.class_profile(graph, target)
                operation = "graph.class_profile"
            elif query == "search":
                answer = android_graph.search(graph, target)
                operation = "graph.search"
            else:
                return {"error": f"unknown graph query {query!r} — "
                                 "callers|callees|class|search"}
        except android_graph.GraphError as exc:
            return {"error": str(exc)}
        rec = self.store.record(
            predicate_type="android.references", operation=operation,
            provider_id=android_graph.PROVIDER_ID,
            provider_name=android_graph.PROVIDER_NAME,
            provider_version=android_graph.PROVIDER_VERSION,
            parameters={"query": query, "target": target,
                        "source": built["sources"]},
            result=answer,
            subject_sha256=built["_sha"],
            subject_name=built["_name"],
            subject_size=built["_size"],
            limitations=list(android_graph.LIMITATIONS),
            locations=[{"type": "artifact-path",
                        "value": built["sources"]}],
            links=[built["graph_evidence_id"]])
        return {"query": query, "target": target, "answer": answer,
                "evidence_id": rec.evidence_id,
                "graph_evidence_id": built["graph_evidence_id"],
                "sources": built["sources"]}

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
    lines = ["reverse engineering — evidence core + static spine "
             "+ self-provisioned deep rungs (Passes 1–2)"]
    for row in status["engines"]:
        mark = "yes" if row["available"] else "no"
        state = f" [{row['state']}]" if row.get("state") else ""
        lines.append(f"  [{mark}] {row['engine']} ({row['kind']})"
                     f"{state} — {row['note']}")
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
    if result.get("rung"):
        lines.append(f"deepest rung: {result['rung']}")
    deep = result.get("deep") or {}
    if deep.get("engine") == "ghidra-headless" and not deep.get("error"):
        lines.append(f"ghidra: {deep.get('functions')} function(s), "
                     f"{deep.get('decompiled')} decompiled")
    if deep.get("error"):
        lines.append(f"ghidra rung failed: {deep['error']}")
    if deep.get("engine") == "binutils-triage":
        lines.append(
            f"triage: {deep.get('defined_function_count')} defined "
            f"function(s), {deep.get('imported_symbol_count')} "
            "imported symbol(s)")
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


def render_provision(report: dict[str, Any]) -> str:
    lines = ["engine provisioning — Devon fetches her own:"]
    for name, row in (report.get("engines") or {}).items():
        mark = "ready" if row.get("available") else row.get(
            "state", "unavailable")
        lines.append(f"  {name}: {mark} — {row.get('detail', '')}")
        for attempt in row.get("attempts") or []:
            lines.append(f"    · {attempt}")
    lines.append("all engines provisioned"
                 if report.get("ok") else
                 "some engines did not land — the attempts above "
                 "say why, and the matrix reports the same truth")
    return "\n".join(lines)


def render_graph_answer(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        return f"⚠️ {payload['error']}"
    if "stats" in payload:  # a build result, not a query
        stats = payload["stats"]
        lines = [
            f"android reference graph — {payload['sources']}",
            f"via {payload['via']}",
            f"classes {stats['classes']} · methods "
            f"{stats['methods']} · edges {stats['edges']} "
            f"({stats['references']} references) over "
            f"{stats['files_scanned']} file(s)",
            f"graph evidence: {payload['graph_evidence_id']}",
        ]
        if payload.get("collisions"):
            lines.append("ambiguous simple names: "
                         + ", ".join(payload["collisions"][:8]))
        return "\n".join(lines)
    answer = payload["answer"]
    query, target = payload["query"], payload["target"]
    lines = [f"android graph · {query} · {target}",
             f"evidence: {payload['evidence_id']}"]
    if query == "callers":
        lines.append(f"{answer['caller_count']} caller edge(s), "
                     f"{answer['references']} reference(s)")
        for row in answer["callers"][:20]:
            lines.append(f"  ← {row['from_class']}."
                         f"{row['from_method']} ({row['kind']}"
                         f" ×{row['count']})")
    elif query == "callees":
        lines.append(f"{answer['callee_count']} callee edge(s), "
                     f"{answer['references']} reference(s)")
        for row in answer["callees"][:20]:
            lines.append(f"  → {row['to_class']}."
                         f"{row['to_method']} ({row['kind']}"
                         f" ×{row['count']})")
    elif query == "class":
        rec = answer["class"]
        lines.append(f"{rec['fqcn']} ({rec['kind']}) — "
                     f"{len(rec['methods'])} method(s), "
                     f"inbound {answer['inbound_count']}, "
                     f"outbound {answer['outbound_count']}")
        if rec.get("extends"):
            lines.append(f"extends {rec['extends']}")
        if rec.get("implements"):
            lines.append("implements "
                         + ", ".join(rec["implements"]))
        lines.extend(f"  · {name}" for name in
                     sorted(rec["methods"])[:20])
    elif query == "search":
        lines.append(f"{answer['class_count']} class hit(s), "
                     f"{answer['method_count']} method hit(s)")
        lines.extend(f"  {fq}" for fq in answer["classes"][:20])
        lines.extend(f"  {hit['class']}.{hit['method']}"
                     for hit in answer["methods"][:20])
    if answer.get("truncated"):
        lines.append("(sample truncated — the evidence record "
                     "carries the same bound)")
    return "\n".join(lines)
