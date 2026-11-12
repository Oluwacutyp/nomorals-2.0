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

Pass 3 adds the firmware and traffic families behind the same
routing (:mod:`.firmware`, :mod:`.traffic`): an inspection whose
bytes sniff as a firmware image goes to the firmware rung
(built-in signature census + the self-provisioned unblob
extraction engine + a bounded native extractor), and one that
sniffs as a capture (pcap / pcapng / HAR / mitmproxy flows /
Devon-addon JSONL) goes to the traffic rung (built-in readers +
a probed TShark second voice). Sniffing is by content, never by
file name.

Pass 4 adds the top layer (:mod:`.missions`, :mod:`.verify`):
brain-directed missions that decompose a question over the
capability matrix, work the rungs in dependency order under a
journaled plan, let the residual unknowns re-drive the plan, and
close with a reconstruction whose every claim is evidence-tied
and cross-rung verified — contradictions demoted to unknowns,
never smoothed.

The dynamic rung (:mod:`.dynamic`, Frida + GDB) dissolves the
Pass 4 gate: questions only running a target could answer are
answered by running it — as a copy, in a per-run workdir, under
the namespace sandbox with the network dropped. The sandbox is
part of the rung, not a posture question: with no namespace
backend the rung refuses rather than running unsandboxed.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from . import android_graph, dynamic, firmware, ghidra, lightbin, provision, traffic
from .evidence import EvidenceStore, UnknownCloseError
from .failures import failure_count, record_failure
from .inspectors import PROVIDER_ID, PROVIDER_NAME, PROVIDER_VERSION, elf_info, inspect_bytes

__all__ = [
    "ReEngine", "capability_matrix",
    "render_status", "render_inspect_card", "render_evidence_card",
    "render_unknowns", "render_audit", "render_provision",
    "render_graph_answer", "render_firmware_card",
    "render_traffic_card", "render_dynamic_card",
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
    # ── Pass 3: firmware + traffic, probe-true ────────────────
    fw_probe = (
        b"\x27\x05\x19\x56" + b"\x00" * 60)
    rows.append({
        "engine": "firmware-census", "kind": "built-in",
        "available": firmware.census(fw_probe)["is_firmware"],
        "state": "ready",
        "note": "signature census over firmware images (uImage "
                "with CRC checks, SquashFS superblock, cpio entry "
                "census, gzip-wrapper peeks) + native cpio/tar "
                "extraction — no engine needed"})
    ub = provision.unblob_state(root)
    rows.append({
        "engine": "unblob", "kind": "provisioned",
        "available": ub["available"], "state": ub["state"],
        "note": "self-provisioned firmware extraction engine "
                "(pinned via PyPI's published digests, "
                "--require-hashes venv) — " + ub["detail"]})
    bw = firmware.probe_binwalk()
    rows.append({
        "engine": "binwalk", "kind": "probe-only",
        "available": bw["available"],
        "state": "system" if bw["available"] else "probe-only",
        "note": bw["detail"]})
    pcap_probe = (
        struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 262144, 1)
        + struct.pack("<IIII", 0, 0, 42, 42)
        + bytes.fromhex("ffffffffffff123456789abc0806")
        + b"\x00" * 28)
    try:
        _probe_pkts, _ = traffic.read_packets(pcap_probe, "pcap")
        traffic_ok = len(_probe_pkts) == 1
    except traffic.TrafficError:
        traffic_ok = False
    rows.append({
        "engine": "traffic-native", "kind": "built-in",
        "available": traffic_ok, "state": "ready",
        "note": "built-in capture readers: pcap/pcapng dissect "
                "(DNS, TLS handshake facts, cleartext HTTP, "
                "conversations), HAR, mitmproxy flow files, "
                "Devon-addon JSONL"})
    ts = traffic.probe_tshark()
    rows.append({
        "engine": "tshark", "kind": "one-shot",
        "available": ts["available"],
        "state": "ready" if ts["available"] else "unavailable",
        "note": "second engine voice over captures (protocol "
                "hierarchy, conversations, field extraction) — "
                + ts["detail"]})
    addon_path = (Path(__file__).parent / "mitm_scripts"
                  / "devon_capture_addon.py")
    rows.append({
        "engine": "mitm-addon", "kind": "built-in",
        "available": addon_path.is_file(), "state": "ready",
        "note": "Devon's mitmproxy capture addon — a file the "
                "owner runs where mitmproxy lives (owner-invoked, "
                "local); its JSONL folds into evidence here"})
    rows.append({
        "engine": "re-missions", "kind": "built-in",
        "available": True, "state": "ready",
        "note": "brain-directed missions over the rungs (plan → "
                "rung steps → reconstruction verify), journaled "
                "end-to-end in re_missions/re_mission_steps; "
                "unknowns re-drive the plan"})
    # ── the dynamic rung (Frida + GDB), probe-true ─────────────
    dyn = dynamic.dynamic_state(root)
    rows.append({
        "engine": "frida", "kind": "provisioned",
        "available": dyn["frida"]["available"],
        "state": dyn["frida"]["state"],
        "note": "dynamic trace rung (spawn, intercept the main "
                "module's functions, report the calls) — "
                + dyn["frida"]["detail"]})
    rows.append({
        "engine": "gdb", "kind": "provisioned",
        "available": dyn["gdb"]["available"],
        "state": dyn["gdb"]["state"],
        "note": "dynamic session rung (breakpoints, frames, "
                "registers, exit code) — " + dyn["gdb"]["detail"]})
    rows.append({
        "engine": "dynamic-sandbox", "kind": "built-in",
        "available": dyn["sandbox"]["available"],
        "state": ("ready" if dyn["sandbox"]["available"]
                  else "unavailable"),
        "note": "the namespace sandbox dynamic runs stand on — "
                + dyn["sandbox"]["detail"]})
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
        from .missions import MissionStore

        audit = self.store.audit()
        counts = MissionStore(self._db).counts()
        return {
            "engines": capability_matrix(self._tools_root),
            "evidence_records": audit["evidence_records"],
            "unknowns_open": audit["unknowns_open"],
            "unknowns_resolved": audit["unknowns_resolved"],
            "unknowns_contradicted": audit["unknowns_contradicted"],
            "missions_open": counts.get("planned", 0)
            + counts.get("running", 0),
            "missions_closed": counts.get("closed", 0),
            "missions_failed": counts.get("failed", 0),
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
        # Pass 3 families first, by content: a capture or a
        # firmware image is not an "unknown binary" to the spine
        capture_kind = traffic.sniff(data)
        if capture_kind is None and traffic.sniff_addon(data):
            capture_kind = "mitm-addon-jsonl"
        if capture_kind is not None:
            return self._inspect_capture(
                target, data, sha, size, capture_kind, deep)
        fw_census = firmware.census(data)
        if fw_census["is_firmware"]:
            return self._inspect_firmware(
                target, data, sha, size, fw_census, deep)
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

    # ── Pass 3: capture routing ──────────────────────────────────────
    def _inspect_capture(self, target: Path, data: bytes, sha: str,
                         size: int, kind: str, deep: str
                         ) -> dict[str, Any]:
        """Route one capture file to the traffic rung: built-in
        readers first, TShark as the second voice when it probes
        present, everything folded as Evidence. An unreadable
        capture is an error result, never a dressed-up empty."""
        store = self.store
        common = {
            "subject_sha256": sha, "subject_name": target.name,
            "subject_size": size,
            "provider_id": traffic.PROVIDER_ID,
            "provider_name": traffic.PROVIDER_NAME,
            "provider_version": traffic.PROVIDER_VERSION,
        }
        tshark_facts: dict[str, Any] | None = None
        rung = "built-in traffic readers"
        try:
            if kind in ("pcap", "pcapng"):
                packets, fmt = traffic.read_packets(data, kind)
                facts = traffic.dissect(packets, fmt)
                if deep != "off" and traffic.probe_tshark()[
                        "available"]:
                    try:
                        tshark_facts = traffic.run_tshark(target)
                        rung = ("built-in traffic readers + tshark "
                                "second voice")
                    except traffic.TrafficError as exc:
                        record_failure(
                            "reverse.engine.tshark_route", exc)
            elif kind == "har":
                doc = json.loads(data.decode("utf-8"))
                facts = traffic.parse_har(doc)
            elif kind == "mitm-flows":
                facts = traffic.flows_facts(traffic.parse_flows(data))
            elif kind == "mitm-addon-jsonl":
                facts = traffic.parse_addon_jsonl(
                    data.decode("utf-8", "replace"))
            else:  # pragma: no cover - sniff() is a closed set
                return {"error": f"unroutable capture kind {kind!r}"}
        except (ValueError, struct.error, traffic.TrafficError) \
                as exc:
            record_failure("reverse.engine.capture_parse", exc)
            return {"error": f"{target.name} sniffs as {kind} but "
                             f"the reader refused it: {exc}"}
        identify = store.record(
            predicate_type="format.identify",
            operation="detect_format",
            parameters={"format": kind},
            result={"format": kind, "size": size,
                    "magic_hex": data[:16].hex()},
            locations=[{"type": "artifact-path",
                        "value": str(target)}],
            **common)
        if kind == "mitm-addon-jsonl":
            folded = traffic.fold_addon(
                store, subject_sha256=sha,
                subject_name=target.name, subject_size=size,
                subject_path=str(target), facts=facts,
                link_to=identify.evidence_id)
        else:
            folded = traffic.fold_evidence(
                store, subject_sha256=sha,
                subject_name=target.name, subject_size=size,
                subject_path=str(target), kind=kind, facts=facts,
                tshark_facts=tshark_facts,
                link_to=identify.evidence_id)
        evidence = [identify]
        for eid in folded["evidence_ids"]:
            rec = store.get(eid)
            if rec is not None:
                evidence.append(rec)
        unknowns = [store.get_unknown(u["unknown_id"])
                    for u in folded["unknowns"]]
        unknowns = [u for u in unknowns if u is not None]
        journaled = self._journal_finding(
            target.name, kind, sha, evidence, unknowns)
        remembered = self._remember_finding(
            target.name, kind, sha, identify, evidence, unknowns)
        result: dict[str, Any] = {
            "subject": {"name": target.name, "path": str(target),
                        "sha256": sha, "size": size},
            "format": kind,
            "facts": facts,
            "rung": rung,
            "evidence_ids": [e.evidence_id for e in evidence],
            "evidence": [e.to_dict() for e in evidence],
            "unknowns": [u.to_dict() for u in unknowns],
            "journaled": journaled,
            "remembered": remembered,
        }
        result["card_text"] = render_traffic_card(result)
        return result

    # ── Pass 3: firmware routing ─────────────────────────────────────
    def _inspect_firmware(self, target: Path, data: bytes, sha: str,
                          size: int, census_facts: dict[str, Any],
                          deep: str) -> dict[str, Any]:
        """Route one firmware image to the firmware rung: the
        built-in census always answers; extraction is unblob when
        it is ready (provisioned first under deep=provision), else
        the bounded native extractor for cpio/tar shapes, else a
        residual unknown naming the rung that closes the gap."""
        store = self.store
        common = {
            "subject_sha256": sha, "subject_name": target.name,
            "subject_size": size, "provider_id": PROVIDER_ID,
            "provider_name": PROVIDER_NAME,
            "provider_version": PROVIDER_VERSION,
        }
        identify = store.record(
            predicate_type="format.identify",
            operation="detect_format",
            parameters={"format": "firmware"},
            result={"format": "firmware",
                    "primary": census_facts.get("primary"),
                    "size": size, "magic_hex": data[:16].hex()},
            locations=[{"type": "artifact-path",
                        "value": str(target)}],
            **common)
        work = (provision.re_root(self._tools_root) / "extract"
                / sha[:16])
        provision_report: dict[str, Any] | None = None
        ub_state = provision.unblob_state(self._tools_root)
        if deep == "provision" and not ub_state["available"]:
            try:
                row = provision.provision_engine(
                    "unblob", root=self._tools_root)
                provision_report = {"engines": {"unblob": row},
                                    "ok": bool(row.get("available"))}
            except provision.ProvisionError as exc:
                provision_report = {
                    "engines": {"unblob": {
                        "engine": "unblob", "state": "unavailable",
                        "available": False,
                        "attempts": exc.attempts,
                        "detail": str(exc)}},
                    "ok": False}
                record_failure("reverse.engine.unblob_provision",
                               exc)
            ub_state = provision.unblob_state(self._tools_root)
        unblob_facts: dict[str, Any] | None = None
        inventory: dict[str, Any] | None = None
        extraction_source = ""
        rung = "built-in firmware census"
        if ub_state["available"]:
            try:
                unblob_facts = firmware.run_unblob(
                    target, work, root=self._tools_root)
                inventory = firmware.build_inventory(
                    unblob_facts["extract_root"])
                extraction_source = "unblob"
                rung = "unblob (self-provisioned)"
            except firmware.FirmwareError as exc:
                record_failure("reverse.engine.unblob_route", exc)
                unblob_facts = None
        if inventory is None:
            native = firmware.extract_native(data, work / "native")
            if native.get("extracted"):
                inventory = firmware.build_inventory(work / "native")
                extraction_source = "native"
                rung = ("built-in firmware census + native "
                        "extraction")
        folded = firmware.fold_evidence(
            store, subject_sha256=sha, subject_name=target.name,
            subject_size=size, subject_path=str(target),
            census_facts=census_facts, unblob_facts=unblob_facts,
            inventory=inventory,
            extraction_source=extraction_source,
            link_to=identify.evidence_id)
        evidence = [identify]
        for eid in folded["evidence_ids"]:
            rec = store.get(eid)
            if rec is not None:
                evidence.append(rec)
        unknowns = [store.get_unknown(u["unknown_id"])
                    for u in folded["unknowns"]]
        unknowns = [u for u in unknowns if u is not None]
        if inventory is None:
            gap = store.open_unknown(
                f"the contents of {target.name} beyond the "
                "signature census are unestablished — no "
                "extraction rung answered on this device; the "
                "self-provisioned unblob rung closes this",
                subject_sha256=sha, raised_by=identify.evidence_id,
                supporting=[identify.evidence_id],
                required_authority="shipped-artifact",
                required_confidence="derived")
            unknowns.append(gap)
        journaled = self._journal_finding(
            target.name, "firmware", sha, evidence, unknowns)
        remembered = self._remember_finding(
            target.name, "firmware", sha, identify, evidence,
            unknowns)
        result: dict[str, Any] = {
            "subject": {"name": target.name, "path": str(target),
                        "sha256": sha, "size": size},
            "format": "firmware",
            "facts": census_facts,
            "rung": rung,
            "extraction": extraction_source or "none",
            "evidence_ids": [e.evidence_id for e in evidence],
            "evidence": [e.to_dict() for e in evidence],
            "unknowns": [u.to_dict() for u in unknowns],
            "journaled": journaled,
            "remembered": remembered,
        }
        if provision_report is not None:
            result["provision"] = provision_report
        result["card_text"] = render_firmware_card(result)
        return result

    # ── Pass 2: provisioning drive ───────────────────────────────────
    def provision_engines(self) -> dict[str, Any]:
        """Run the self-provisioning drive for the provisioned
        rungs (JDK → Ghidra, then the firmware rung's unblob) and
        report each engine's landing state with its attempts.
        REA makes the human fetch engines; Devon fetches her own.
        """
        report = provision.provision_all(root=self._tools_root)
        try:
            row = provision.provision_engine(
                "unblob", root=self._tools_root)
        except provision.ProvisionError as exc:
            row = {"engine": "unblob", "state": "unavailable",
                   "available": False, "source": "",
                   "version":
                       provision.DESCRIPTORS["unblob"]["version"],
                   "path": "", "attempts": exc.attempts,
                   "detail": str(exc)}
            record_failure("reverse.provision.unblob", exc)
        report["engines"]["unblob"] = row
        if not row.get("available"):
            report["ok"] = False
        for engine in ("frida", "gdb"):
            try:
                row = provision.provision_engine(
                    engine, root=self._tools_root)
            except provision.ProvisionError as exc:
                row = {"engine": engine, "state": "unavailable",
                       "available": False, "source": "",
                       "version":
                           provision.DESCRIPTORS[engine]["version"],
                       "path": "", "attempts": exc.attempts,
                       "detail": str(exc)}
                record_failure(f"reverse.provision.{engine}", exc)
            report["engines"][engine] = row
            if not row.get("available"):
                report["ok"] = False
        return report

    # ── the dynamic rung (Frida + GDB, sandboxed) ─────────────────
    def run_dynamic(self, path: str, *, rung: str = "auto",
                    script: str | None = None,
                    args: tuple[str, ...] | list[str] = (),
                    timeout: float | None = None,
                    provision_missing: bool = False,
                    root: Any = None) -> dict[str, Any]:
        """Run one target under the dynamic rung and fold the run
        into Evidence.

        ``rung`` is ``auto`` (Frida when ready, else GDB), ``frida``
        or ``gdb``. The run executes a COPY of the target in a
        per-run workdir under the namespace sandbox, network
        dropped; with no namespace backend this returns an error
        naming the sandbox, never an unsandboxed run. The result
        carries the rung, the folded evidence ids, the residual
        unknowns, and the card text — or ``error`` naming what
        failed.
        """
        target = Path(path or "").expanduser()
        if not target.is_file():
            return {"error": f"no readable file at {path}"}
        effective_root = root if root is not None else self._tools_root
        try:
            result = dynamic.run_dynamic(
                target, root=effective_root, rung=rung, args=args,
                script=script,
                timeout=(timeout if timeout is not None
                         else dynamic.RUN_TIMEOUT_S),
                provision_missing=provision_missing)
        except dynamic.DynamicError as exc:
            return {"error": str(exc)}
        folded = dynamic.fold_evidence(
            self.store, subject_sha256=result["sha256"],
            subject_name=target.name,
            subject_size=result["size"], subject_path=str(target),
            result=result)
        out: dict[str, Any] = {
            "subject": {"name": target.name, "path": str(target),
                        "sha256": result["sha256"],
                        "size": result["size"]},
            "rung": result["rung"],
            "engine_version": result.get("engine_version") or "",
            "exit_code": result.get("exit_code"),
            "timed_out": bool(result.get("timed_out")),
            "duration_s": result.get("duration_s"),
            "sandbox": result.get("sandbox") or {},
            "evidence_ids": folded["evidence_ids"],
            "unknowns": folded["unknowns"],
            "result": result,
        }
        out["card_text"] = render_dynamic_card(out)
        return out

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

    # ── Pass 4: missions + reconstruction ────────────────────────────
    def run_mission(self, question: str, targets: list[str], *,
                    extra_questions: tuple[str, ...] | list[str] = (),
                    graph_target: str = "") -> dict[str, Any]:
        """Plan + work one RE mission (see :mod:`.missions`).

        The mission is journaled before it runs a single rung, so
        what comes back is the mission record — plan, per-step
        results, per-target verified reconstructions, parked
        unknowns with their reasons — not a console transcript.
        """
        from .missions import MissionRunner

        return MissionRunner(self).start(
            question, list(targets), extra_questions=extra_questions,
            graph_target=graph_target)

    def resume_mission(self, mission_id: str) -> dict[str, Any]:
        """Resume a mission from its journal: pending steps work,
        finished steps stand (evidence is content-addressed, so a
        re-walked step lands on the same records)."""
        from .missions import MissionRunner

        return MissionRunner(self).run(mission_id)

    def mission(self, mission_id: str) -> dict[str, Any]:
        """One mission's full journal record + its reconstructions."""
        from .missions import mission_status

        return mission_status(self, mission_id)

    def mission_list(self, limit: int = 20) -> dict[str, Any]:
        from .missions import list_missions

        return list_missions(self, limit=limit)

    def reconstruct(self, subject_sha256: str) -> dict[str, Any]:
        """Build + verify the reconstruction of one subject from
        its evidence chain (see :mod:`.verify`)."""
        from .verify import ReconstructionError, build_reconstruction

        try:
            return build_reconstruction(self.store, subject_sha256)
        except ReconstructionError as exc:
            return {"error": str(exc)}


# ── renders (plain text; surfaces own the presentation layer) ────────

def render_status(status: dict[str, Any]) -> str:
    lines = ["reverse engineering — evidence core + static spine "
             "+ self-provisioned deep rungs + firmware/traffic "
             "+ missions/reconstruction + dynamic rung (Frida/GDB)"]
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
    lines.append(
        f"missions: {status.get('missions_open', 0)} open, "
        f"{status.get('missions_closed', 0)} closed, "
        f"{status.get('missions_failed', 0)} failed")
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


def render_dynamic_card(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"⚠️ {result['error']}"
    subj = result["subject"]
    res = result.get("result") or {}
    lines = [
        f"▶️ {subj['name']} — dynamic run ({result['rung']})",
        f"sha256 {subj['sha256'][:16]}… · {subj['size']} bytes",
        f"exit code: {result.get('exit_code')}"
        + (" · timed out" if result.get("timed_out") else ""),
    ]
    if result["rung"] == "frida":
        lines.append(
            f"frida: {res.get('event_count', 0)} call(s) observed "
            f"across {len(res.get('hooked') or [])} intercepted "
            "function(s)")
        unique = sorted({str(e.get("function") or "")
                         for e in (res.get("events") or [])
                         if e.get("function")})
        if unique:
            lines.append("called: " + ", ".join(unique[:20]))
    else:
        lines.append(
            f"gdb: {res.get('stop_count', 0)} breakpoint stop(s)")
        for stop in (res.get("stops") or [])[:5]:
            frames = [f["function"] for f in stop.get("frames") or []]
            lines.append(f"  stop in {stop.get('function')}"
                         + (f" — frames: {', '.join(frames[:6])}"
                            if frames else ""))
        if res.get("signal"):
            lines.append(f"signal: {res['signal']}")
        if res.get("entry_fallback"):
            lines.append("main had no symbol — stopped at the entry "
                         "point instead")
    sb = result.get("sandbox") or {}
    lines.append(
        f"sandbox: {sb.get('backend')} · network dropped: "
        f"{'yes' if sb.get('network_enforced') else 'no'} · "
        "target ran as a copy in its own workdir")
    lines.append(f"evidence: {len(result['evidence_ids'])} record(s) — "
                 f"root {result['evidence_ids'][0]}")
    if result["unknowns"]:
        lines.append(f"residual unknowns: {len(result['unknowns'])}")
        for unk in result["unknowns"]:
            lines.append(f"  ? {unk['question']} [{unk['unknown_id']}]")
    else:
        lines.append("residual unknowns: none raised by this run")
    return "\n".join(lines)


def render_firmware_card(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"⚠️ {result['error']}"
    subj = result["subject"]
    facts = result.get("facts") or {}
    parsed = facts.get("parsed") or {}
    lines = [
        f"🔍 {subj['name']} — firmware image "
        f"({facts.get('primary') or 'unknown container'})",
        f"sha256 {subj['sha256'][:16]}… · {subj['size']} bytes",
        f"rung: {result.get('rung')} · extraction: "
        f"{result.get('extraction')}",
    ]
    if parsed.get("uimage"):
        u = parsed["uimage"]
        lines.append(
            f"uImage: {u.get('name')} · {u.get('os')}/"
            f"{u.get('arch')} · {u.get('image_type')} · "
            f"compression {u.get('compression')} · header CRC "
            f"{'ok' if u.get('header_crc_ok') else 'BAD'} · data "
            f"CRC {'ok' if u.get('data_crc_ok') else 'BAD'}")
    if parsed.get("squashfs"):
        s = parsed["squashfs"]
        lines.append(
            f"squashfs {s.get('version')}: {s.get('inodes')} "
            f"inode(s), compression {s.get('compression')}, "
            f"{s.get('bytes_used')} bytes used")
    if parsed.get("cpio"):
        c = parsed["cpio"]
        lines.append(f"cpio: {c.get('entries')} entr(ies) censused")
    kinds = [e["predicate_type"] for e in result["evidence"]]
    if "firmware.inventory" in kinds:
        inv = next(e["result"] for e in result["evidence"]
                   if e["predicate_type"] == "firmware.inventory")
        lines.append(
            f"inventory: {inv.get('files_total')} file(s), "
            f"{inv.get('bytes_total')} bytes extracted")
        if inv.get("notable_paths"):
            lines.append("notable: "
                         + ", ".join(inv["notable_paths"][:6]))
        if inv.get("carved_binaries"):
            lines.append(
                f"carved binaries: {len(inv['carved_binaries'])} "
                "(inspect one by path for the native rungs)")
    if "firmware.credential_flags" in kinds:
        flags = next(e["result"] for e in result["evidence"]
                     if e["predicate_type"]
                     == "firmware.credential_flags")
        lines.append(
            f"credential-shaped strings: {flags.get('flag_count')} "
            "flagged (observations, not proven secrets — see the "
            "evidence record)")
    lines.append(f"evidence: {len(result['evidence_ids'])} record(s) "
                 f"— root {result['evidence_ids'][0]}")
    if result["unknowns"]:
        lines.append(f"residual unknowns: {len(result['unknowns'])} "
                     "registered")
        for unk in result["unknowns"]:
            lines.append(f"  ? {unk['question']} [{unk['unknown_id']}]")
    lines.append("journaled to decisions: "
                 + ("yes" if result["journaled"] else "no")
                 + " · remembered: "
                 + ("yes" if result["remembered"] else "no"))
    return "\n".join(lines)


def render_traffic_card(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"⚠️ {result['error']}"
    subj = result["subject"]
    facts = result.get("facts") or {}
    lines = [
        f"🔍 {subj['name']} — capture ({result['format']})",
        f"sha256 {subj['sha256'][:16]}… · {subj['size']} bytes",
        f"rung: {result.get('rung')}",
    ]
    if facts.get("packets") is not None:
        lines.append(f"packets: {facts['packets']} · bytes: "
                     f"{facts.get('bytes')}")
        protos = facts.get("protocols") or {}
        if protos:
            census = ", ".join(
                f"{name} {row['frames']}" for name, row in
                sorted(protos.items(),
                       key=lambda kv: -kv[1]["frames"]))
            lines.append(f"protocols: {census}")
    if facts.get("entries") is not None:
        lines.append(f"HAR entries: {facts['entries']}")
    if facts.get("flows") is not None:
        lines.append(f"flows: {facts['flows']}")
    if facts.get("records") is not None:
        lines.append(f"addon records: {facts['records']}")
    dns = facts.get("dns") or {}
    names = [q["name"] for resp in dns.get("responses", [])
             for q in resp.get("questions", [])]
    asked = [q["name"] for query in dns.get("queries", [])
             for q in query.get("questions", [])]
    if asked or names:
        lines.append("DNS: " + ", ".join(
            dict.fromkeys(asked + names))[:160])
    tls = facts.get("tls") or {}
    for session in (tls.get("sessions") or [])[:4]:
        bits = []
        if session.get("sni"):
            bits.append(f"SNI {session['sni']}")
        if session.get("version"):
            bits.append(session["version"])
        for cert in (session.get("certificates") or [])[:1]:
            subject = (cert.get("subject") or {}).get("CN")
            if subject:
                bits.append(f"cert CN={subject}")
        if bits:
            lines.append("TLS: " + " · ".join(bits))
    exchanges = facts.get("exchanges") or (
        facts.get("http") or {}).get("exchanges") or []
    for ex in exchanges[:4]:
        lines.append(
            f"HTTP: {ex.get('method')} {ex.get('host') or ''}"
            f"{ex.get('path') or ex.get('uri') or ''} → "
            f"{ex.get('status')}")
    lines.append(f"evidence: {len(result['evidence_ids'])} record(s) "
                 f"— root {result['evidence_ids'][0]}")
    if result["unknowns"]:
        lines.append(f"residual unknowns: {len(result['unknowns'])} "
                     "registered")
        for unk in result["unknowns"]:
            lines.append(f"  ? {unk['question']} [{unk['unknown_id']}]")
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
