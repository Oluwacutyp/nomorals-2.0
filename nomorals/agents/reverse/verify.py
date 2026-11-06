"""Reconstruction verify — Pass 4's second half.

Passes 1–3 fill the evidence store with records from independent
rungs: the built-in spine, the binutils light rung, the Ghidra
deep rung, the firmware census/extraction pair, the built-in
traffic readers and their TShark second voice, the Android graph.
This module is where those voices are made to *check each other*.

A **reconstruction** is the answer to "so what IS this artifact,
and what does it do" built the only way Devon's discipline
allows: as a list of claims, where every claim names the evidence
records it rests on. The builder refuses — structurally, not by
convention — to emit a claim with no evidence, or one citing
evidence outside the subject's chain. A claim the record cannot
support is not a claim; it is an unknown, and the registry already
has a home for those.

On top of the claims sits **verification**: cross-rung agreement
checks over the same underlying facts (the dynamic section read
by two parsers; the import census read by two engines; the
strings a decompiler's pseudocode actually references; the
firmware census against the extraction it predicts; the capture
read by two dissectors). Agreement strengthens a claim to
``corroborated``. Contradiction does the opposite: the claim is
*demoted* out of the report, the conflict is recorded verbatim,
and a residual unknown is opened carrying both sides' evidence —
never smoothed over, never averaged away.

Everything here is static synthesis over records that already
exist. Nothing is executed; the reconstruction provider is
Devon's own inference over artifact evidence, and the records it
writes say exactly that (authority ``analyst-inference``,
confidence ``derived``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .failures import record_failure

__all__ = [
    "PROVIDER_ID", "Claim", "ReconstructionError",
    "build_report", "build_reconstruction", "render_reconstruction",
]

PROVIDER_ID = "devon.reconstruction"
PROVIDER_NAME = "Devon reconstruction verify (Pass 4)"
PROVIDER_VERSION = "1"

#: Behavior claims per family are capped in the report — the
#: reconstruction is the verified summary, not a second copy of
#: the evidence store; the records themselves carry the full sets.
_MAX_CALL_CLAIMS = 8
_MAX_STRING_REF_CLAIMS = 8
_MAX_CARVED_CLAIMS = 8
_MAX_CONVERSATION_CLAIMS = 6
_MAX_HTTP_CLAIMS = 6
_MAX_TLS_CLAIMS = 4

_REPORT_LIMITATIONS = (
    "a reconstruction claim is a synthesis over the evidence it "
    "cites — it is only as strong as the weakest record under it, "
    "and its limitations ride on the claim",
    "absence of a claim is not evidence of absence: coverage is "
    "bounded by the rungs that actually ran over this subject",
)


class ReconstructionError(ValueError):
    """A reconstruction the discipline refuses to emit. The
    message says which rule and what would satisfy it."""


def _claim_id(kind: str, text: str,
              evidence_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256(json.dumps(
        {"kind": kind, "text": text,
         "evidence": sorted(evidence_ids)},
        sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    return f"cl:{digest[:20]}"


@dataclass(frozen=True)
class Claim:
    """One reconstruction claim, evidence-tied by construction."""

    kind: str  # "identity" | "behavior"
    text: str
    evidence_ids: tuple[str, ...]
    confidence: str = "observed"
    check_keys: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    status: str = "supported"  # supported | corroborated
    checks: tuple[dict[str, Any], ...] = ()

    @property
    def claim_id(self) -> str:
        return _claim_id(self.kind, self.text, self.evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id, "kind": self.kind,
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "status": self.status,
            "checks": [dict(c) for c in self.checks],
            "limitations": list(self.limitations),
        }


@dataclass
class _Draft:
    """A claim under construction (adjudication mutates status)."""

    kind: str
    text: str
    evidence_ids: tuple[str, ...]
    confidence: str = "observed"
    check_keys: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    status: str = "supported"
    checks: list[dict[str, Any]] = field(default_factory=list)

    def freeze(self) -> Claim:
        return Claim(
            kind=self.kind, text=self.text,
            evidence_ids=self.evidence_ids,
            confidence=self.confidence, check_keys=self.check_keys,
            limitations=self.limitations, status=self.status,
            checks=tuple(self.checks))


# ── the structural builder ───────────────────────────────────────────

def build_report(claims: list[Claim], *, subject: dict[str, Any],
                 chain_ids: set[str] | frozenset[str],
                 checks: list[dict[str, Any]],
                 conflicts: list[dict[str, Any]]
                 ) -> dict[str, Any]:
    """Assemble a reconstruction report — and refuse a bad one.

    The refusal is the point: a claim with no evidence ids, or
    citing an id outside the subject's evidence chain, raises
    :class:`ReconstructionError` instead of landing in a report a
    reader would trust. Every surface that emits a reconstruction
    goes through here.
    """
    for claim in claims:
        if not claim.evidence_ids:
            raise ReconstructionError(
                f"claim {claim.text!r} cites no evidence — a "
                "reconstruction claim the record cannot support "
                "is an unknown, not a claim; register it in the "
                "unknowns registry instead")
        outside = [e for e in claim.evidence_ids
                   if e not in chain_ids]
        if outside:
            raise ReconstructionError(
                f"claim {claim.text!r} cites evidence outside "
                f"this subject's chain: {', '.join(outside)} — "
                "claims rest only on the chain they reconstruct")
    return {
        "subject": dict(subject),
        "claims": [c.to_dict() for c in claims],
        "checks": [dict(c) for c in checks],
        "conflicts": [dict(c) for c in conflicts],
        "counts": {
            "claims": len(claims),
            "corroborated": sum(1 for c in claims
                                if c.status == "corroborated"),
            "identity": sum(1 for c in claims
                            if c.kind == "identity"),
            "behavior": sum(1 for c in claims
                            if c.kind == "behavior"),
            "conflicts": len(conflicts),
        },
    }


# ── chain accessors ──────────────────────────────────────────────────

def _by_pred(records: list[Any], predicate: str) -> list[Any]:
    return [r for r in records if r.predicate_type == predicate]


def _first(records: list[Any], predicate: str) -> Any:
    rows = _by_pred(records, predicate)
    return rows[0] if rows else None


def _structure_record(records: list[Any]) -> Any:
    for pred in ("elf.structure", "pe.structure", "macho.structure",
                 "macho-fat.structure"):
        rec = _first(records, pred)
        if rec is not None:
            return rec
    return None


def _names_imports(rec: Any) -> set[str]:
    """Imported symbol names from one ghidra native.imports record."""
    out: set[str] = set()
    for row in (rec.result or {}).get("imports") or []:
        if isinstance(row, dict) and row.get("name"):
            out.add(str(row["name"]))
        elif isinstance(row, str):
            out.add(row)
    return out


def _string_values(rec: Any) -> set[str]:
    """String values from one ghidra native.strings record."""
    out: set[str] = set()
    for row in (rec.result or {}).get("strings") or []:
        if isinstance(row, dict) and row.get("value"):
            out.add(str(row["value"]))
        elif isinstance(row, str):
            out.add(row)
    return out


# ── claim derivation ─────────────────────────────────────────────────
# Every deriver reads ONLY records in the chain. A deriver that
# finds nothing emits nothing — silence in the report means the
# rung never spoke, and the coverage block says which rungs did.

def _derive_identity(records: list[Any], subject: dict[str, Any],
                     drafts: list[_Draft]) -> None:
    identify = _first(records, "format.identify")
    if identify is not None:
        fmt = (identify.result or {}).get("format") or "unknown"
        drafts.append(_Draft(
            "identity", f"is a {fmt} artifact "
            f"({subject.get('size', 0)} bytes, sha256 "
            f"{subject.get('sha256', '')[:16]}…)",
            (identify.evidence_id,), confidence=identify.confidence,
            limitations=tuple(identify.limitations)))
    structure = _structure_record(records)
    if structure is not None:
        facts = structure.result or {}
        for key in ("machine", "cpu"):
            if facts.get(key):
                drafts.append(_Draft(
                    "identity", f"targets machine {facts[key]}",
                    (structure.evidence_id,),
                    confidence=structure.confidence))
                break
        if facts.get("interpreter"):
            drafts.append(_Draft(
                "identity",
                f"dynamic loader: {facts['interpreter']}",
                (structure.evidence_id,),
                confidence=structure.confidence))
    program = _first(records, "native.program")
    if program is not None:
        res = program.result or {}
        bits = [str(res[k]) for k in ("format", "language",
                                      "compiler") if res.get(k)]
        if bits:
            drafts.append(_Draft(
                "identity",
                "the deep rung identifies it as " + " · ".join(bits),
                (program.evidence_id,), confidence=program.confidence,
                limitations=tuple(program.limitations)))
    census = _first(records, "firmware.census")
    if census is not None:
        res = census.result or {}
        if res.get("primary"):
            drafts.append(_Draft(
                "identity",
                f"firmware image: primary container "
                f"{res['primary']}",
                (census.evidence_id,), confidence=census.confidence,
                limitations=tuple(census.limitations)))
    graph = _first(records, "android.class_graph")
    if graph is not None:
        res = graph.result or {}
        stats = res.get("stats") or res
        if stats.get("classes") is not None:
            drafts.append(_Draft(
                "identity",
                f"the Android reference graph spans "
                f"{stats.get('classes')} classes, "
                f"{stats.get('methods')} methods, "
                f"{stats.get('edges')} reference edges",
                (graph.evidence_id,), confidence=graph.confidence,
                limitations=tuple(graph.limitations)))


def _derive_native_behavior(records: list[Any],
                            drafts: list[_Draft]) -> None:
    structure = _structure_record(records)
    triage = _first(records, "elf.triage")
    # — needed libraries, merged across the parsers that read them
    lib_sources: list[tuple[Any, set[str]]] = []
    if structure is not None:
        facts = structure.result or {}
        libs = set(facts.get("needed_libraries") or [])
        if not libs:
            libs = set(facts.get("imported_dlls") or [])
        if libs:
            lib_sources.append((structure, libs))
    if triage is not None:
        libs = set((triage.result or {}).get("needed_libraries")
                   or [])
        if libs:
            lib_sources.append((triage, libs))
    if lib_sources:
        union: set[str] = set()
        for _rec, libs in lib_sources:
            union |= libs
        drafts.append(_Draft(
            "behavior",
            "dynamically links against: "
            + ", ".join(sorted(union)),
            tuple(rec.evidence_id for rec, _l in lib_sources),
            confidence=lib_sources[0][0].confidence,
            check_keys=("needed_libraries",)))
    # — the import census, merged across engines
    imp_sources: list[tuple[Any, set[str]]] = []
    if triage is not None:
        names = set((triage.result or {}).get("imported_symbols")
                    or [])
        if names:
            imp_sources.append((triage, names))
    imports_rec = _first(records, "native.imports")
    if imports_rec is not None:
        names = _names_imports(imports_rec)
        if names:
            imp_sources.append((imports_rec, names))
    if structure is not None:
        facts = structure.result or {}
        pe_imports = facts.get("imports") or {}
        if isinstance(pe_imports, dict) and pe_imports:
            names = {str(n) for names in pe_imports.values()
                     for n in (names or [])}
            if names:
                imp_sources.append((structure, names))
    if imp_sources:
        union = set()
        for _rec, names in imp_sources:
            union |= names
        sample = ", ".join(sorted(union)[:10])
        drafts.append(_Draft(
            "behavior",
            f"imports {len(union)} distinct dynamic symbols "
            f"({sample}{'…' if len(union) > 10 else ''})",
            tuple(rec.evidence_id for rec, _n in imp_sources),
            confidence=imp_sources[0][0].confidence,
            check_keys=("imports",)))
    # — functions + decompile coverage
    functions_rec = _first(records, "native.functions")
    if functions_rec is not None:
        res = functions_rec.result or {}
        text = f"the deep census finds {res.get('count')} functions"
        if res.get("truncated"):
            text += " (the census hit its cap — the count is a floor)"
        drafts.append(_Draft(
            "behavior", text, (functions_rec.evidence_id,),
            confidence=functions_rec.confidence,
            limitations=tuple(functions_rec.limitations)))
        calls_made = 0
        for fn in res.get("functions") or []:
            calls = [c for c in (fn.get("calls") or []) if c]
            if not calls or fn.get("is_external"):
                continue
            if calls_made >= _MAX_CALL_CLAIMS:
                break
            drafts.append(_Draft(
                "behavior",
                f"{fn.get('name')} calls: "
                + ", ".join(str(c) for c in calls[:12]),
                (functions_rec.evidence_id,),
                confidence=functions_rec.confidence))
            calls_made += 1
    decompile_rec = _first(records, "native.decompile")
    if decompile_rec is not None:
        res = decompile_rec.result or {}
        text = (f"{res.get('decompiled', 0)} function(s) have "
                "decompiled pseudocode")
        if res.get("failed"):
            text += (f"; the decompiler failed on "
                     f"{len(res['failed'])}: "
                     + ", ".join(str(n) for n in res["failed"][:6]))
        drafts.append(_Draft(
            "behavior", text, (decompile_rec.evidence_id,),
            confidence=decompile_rec.confidence,
            limitations=tuple(decompile_rec.limitations)))
    # — strings census + decompile string references
    spine_strings = _first(records, "binary.strings")
    ghidra_strings = _first(records, "native.strings")
    values: set[str] = set()
    string_ids: list[str] = []
    if spine_strings is not None:
        res = spine_strings.result or {}
        values |= {str(s) for s in res.get("sample") or []}
        string_ids.append(spine_strings.evidence_id)
    if ghidra_strings is not None:
        values |= _string_values(ghidra_strings)
        string_ids.append(ghidra_strings.evidence_id)
    if spine_strings is not None:
        res = spine_strings.result or {}
        text = (f"carries {res.get('count')} printable strings "
                "in the built-in census")
        if res.get("sample_is_capped"):
            text += " (the record's sample is capped; the count is full)"
        drafts.append(_Draft(
            "behavior", text, (spine_strings.evidence_id,),
            confidence=spine_strings.confidence,
            check_keys=("strings",)))
    if decompile_rec is not None and values:
        refs_made = 0
        for snippet in (decompile_rec.result or {}).get(
                "snippets") or []:
            code = snippet.get("code") or ""
            hits = sorted(v for v in values
                          if len(v) >= 6 and v in code)
            if not hits:
                continue
            if refs_made >= _MAX_STRING_REF_CLAIMS:
                break
            drafts.append(_Draft(
                "behavior",
                f"the decompiled pseudocode of "
                f"{snippet.get('function')} references "
                + ", ".join(repr(h) for h in hits[:3]),
                tuple(dict.fromkeys(
                    [decompile_rec.evidence_id, *string_ids])),
                confidence="derived",
                check_keys=("string_reference",),
                limitations=tuple(decompile_rec.limitations)))
            refs_made += 1
    hardening = _first(records, "binary.hardening")
    if hardening is not None:
        res = hardening.result or {}
        posture = ", ".join(f"{k}={v}" for k, v in res.items())
        drafts.append(_Draft(
            "behavior",
            f"declared hardening posture: {posture}",
            (hardening.evidence_id,), confidence=hardening.confidence,
            limitations=tuple(hardening.limitations)))


def _derive_firmware_behavior(records: list[Any], store: Any,
                              drafts: list[_Draft]) -> None:
    inventory = _first(records, "firmware.inventory")
    if inventory is None:
        return
    res = inventory.result or {}
    census_kinds = res.get("type_census") or {}
    top = ", ".join(
        f"{k}×{v}" for k, v in sorted(
            census_kinds.items(), key=lambda kv: -kv[1])[:5])
    drafts.append(_Draft(
        "behavior",
        f"extraction yielded {res.get('files_total')} files "
        f"({res.get('bytes_total')} bytes)"
        + (f" — type census: {top}" if top else ""),
        (inventory.evidence_id,), confidence=inventory.confidence,
        check_keys=("firmware_inventory",),
        limitations=tuple(inventory.limitations)))
    if res.get("notable_paths"):
        drafts.append(_Draft(
            "behavior",
            "contains notable paths: "
            + ", ".join(str(p) for p in res["notable_paths"][:8]),
            (inventory.evidence_id,), confidence=inventory.confidence))
    for carved_idx, entry in enumerate(
            res.get("carved_binaries") or []):
        if carved_idx >= _MAX_CARVED_CLAIMS:
            break
        sha = str(entry.get("sha256") or "")
        ids = [inventory.evidence_id]
        if sha:
            landed = store.list_for_subject(sha)
            ids.extend(r.evidence_id for r in landed[:4])
        drafts.append(_Draft(
            "behavior",
            f"contains a carved {entry.get('kind')} binary at "
            f"{entry.get('path')}"
            + (f" (sha256 {sha[:12]}…)" if sha else "")
            + (" — that binary was itself inspected; its evidence "
               "is linked" if len(ids) > 1 else ""),
            tuple(ids), confidence=inventory.confidence,
            check_keys=("carved_binary",)))
    flags = _first(records, "firmware.credential_flags")
    if flags is not None:
        res = flags.result or {}
        drafts.append(_Draft(
            "behavior",
            f"carries {res.get('flag_count')} credential-shaped "
            f"strings across {res.get('scanned_files')} scanned "
            "files — observations of shape, not proven secrets",
            (flags.evidence_id,), confidence=flags.confidence,
            limitations=tuple(flags.limitations)))


def _derive_traffic_behavior(records: list[Any],
                             drafts: list[_Draft]) -> None:
    summary = _first(records, "traffic.summary")
    if summary is not None:
        res = summary.result or {}
        if res.get("packets") is not None:
            text = (f"the capture holds {res.get('packets')} "
                    f"packets ({res.get('bytes')} bytes)")
        elif res.get("entries") is not None:
            text = f"the HAR capture holds {res.get('entries')} entries"
        elif res.get("flows") is not None:
            text = f"the flow file holds {res.get('flows')} flows"
        else:
            text = ""
        if text:
            drafts.append(_Draft(
                "identity", text, (summary.evidence_id,),
                confidence=summary.confidence,
                check_keys=("capture_packets",),
                limitations=tuple(summary.limitations)))
        protos = res.get("protocols") or {}
        if protos:
            census = ", ".join(
                f"{name} {row.get('frames')}" for name, row in
                sorted(protos.items(),
                       key=lambda kv: -kv[1].get("frames", 0))[:8])
            drafts.append(_Draft(
                "behavior", f"protocol census: {census}",
                (summary.evidence_id,), confidence=summary.confidence,
                check_keys=("capture_protocols",)))
    conv = _first(records, "traffic.conversations")
    if conv is not None:
        rows = sorted(
            (conv.result or {}).get("conversations") or [],
            key=lambda c: -((c.get("bytes_ab") or 0)
                            + (c.get("bytes_ba") or 0)))
        for row in rows[:_MAX_CONVERSATION_CLAIMS]:
            frames = (row.get("frames_ab") or 0) + (
                row.get("frames_ba") or 0)
            drafts.append(_Draft(
                "behavior",
                f"observed conversation {row.get('a')} ↔ "
                f"{row.get('b')} ({row.get('protocol')}, "
                f"{frames} frames, "
                f"{(row.get('bytes_ab') or 0) + (row.get('bytes_ba') or 0)} "
                "bytes)",
                (conv.evidence_id,), confidence=conv.confidence))
    dns = _first(records, "traffic.dns")
    if dns is not None:
        res = dns.result or {}
        names: list[str] = []
        for side in ("queries", "responses"):
            for entry in res.get(side) or []:
                for q in entry.get("questions") or []:
                    if q.get("name") and q["name"] not in names:
                        names.append(str(q["name"]))
        if names:
            drafts.append(_Draft(
                "behavior",
                "DNS names in this capture: "
                + ", ".join(names[:12]),
                (dns.evidence_id,), confidence=dns.confidence))
    tls = _first(records, "traffic.tls")
    if tls is not None:
        for session in ((tls.result or {}).get("sessions")
                        or [])[:_MAX_TLS_CLAIMS]:
            bits = []
            if session.get("sni"):
                bits.append(f"SNI {session['sni']}")
            if session.get("version"):
                bits.append(str(session["version"]))
            for cert in (session.get("certificates") or [])[:1]:
                cn = (cert.get("subject") or {}).get("CN")
                if cn:
                    bits.append(f"cert CN={cn}")
            if bits:
                drafts.append(_Draft(
                    "behavior", "TLS session: " + " · ".join(bits),
                    (tls.evidence_id,), confidence=tls.confidence))
    for http in _by_pred(records, "traffic.http"):
        exchanges = (http.result or {}).get("exchanges") or []
        for ex in exchanges[:_MAX_HTTP_CLAIMS]:
            drafts.append(_Draft(
                "behavior",
                f"HTTP {ex.get('method')} "
                f"{ex.get('host') or ''}"
                f"{ex.get('path') or ex.get('uri') or ''} → "
                f"{ex.get('status')}",
                (http.evidence_id,), confidence=http.confidence,
                limitations=tuple(http.limitations)))
        if exchanges:
            break


# ── cross-rung agreement checks ──────────────────────────────────────
# A check reads the same underlying fact through two independent
# records. Verdicts: "agree" (the voices concur — claims resting
# on them are corroborated), "contradict" (they cannot both be
# right — the claim is demoted and the conflict becomes an
# unknown). A check with only one voice stays silent: a single
# source is support, not corroboration, and never a contradiction.

def _check_needed_libraries(records: list[Any]) -> list[dict[str, Any]]:
    sources: list[tuple[str, Any, set[str]]] = []
    structure = _structure_record(records)
    if structure is not None:
        facts = structure.result or {}
        libs = set(facts.get("needed_libraries") or []) or set(
            facts.get("imported_dlls") or [])
        if libs:
            sources.append(("built-in spine", structure, libs))
    triage = _first(records, "elf.triage")
    if triage is not None:
        libs = set((triage.result or {}).get("needed_libraries")
                   or [])
        if libs:
            sources.append(("binutils triage", triage, libs))
    if len(sources) < 2:
        return []
    first = sources[0][2]
    if all(libs == first for _n, _r, libs in sources[1:]):
        return [{
            "name": "needed-libraries agreement", "claim_key":
            "needed_libraries", "verdict": "agree",
            "detail": "both parsers read the same needed-library "
                      f"set ({len(first)} libraries)",
            "evidence_ids": [r.evidence_id for _n, r, _l in sources],
            "dissent_ids": []}]
    detail = "; ".join(f"{name} reads {sorted(libs)}"
                       for name, _r, libs in sources)
    return [{
        "name": "needed-libraries agreement", "claim_key":
        "needed_libraries", "verdict": "contradict",
        "detail": "the parsers disagree on the dynamic section: "
                  + detail,
        "evidence_ids": [r.evidence_id for _n, r, _l in sources],
        "dissent_ids": [r.evidence_id for _n, r, _l in sources]}]


def _check_import_overlap(records: list[Any]) -> list[dict[str, Any]]:
    sources: list[tuple[str, Any, set[str]]] = []
    triage = _first(records, "elf.triage")
    if triage is not None:
        names = set((triage.result or {}).get("imported_symbols")
                    or [])
        if names:
            sources.append(("binutils triage", triage, names))
    imports_rec = _first(records, "native.imports")
    if imports_rec is not None:
        names = _names_imports(imports_rec)
        if names:
            sources.append(("ghidra", imports_rec, names))
    structure = _structure_record(records)
    if structure is not None:
        names = set((structure.result or {}).get(
            "dynamic_symbols") or [])
        if names:
            sources.append(("built-in spine", structure, names))
    if len(sources) < 2:
        return []
    out: list[dict[str, Any]] = []
    agreed = False
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            n1, r1, s1 = sources[i]
            n2, r2, s2 = sources[j]
            overlap = s1 & s2
            if not overlap:
                out.append({
                    "name": "import-census overlap", "claim_key":
                    "imports", "verdict": "contradict",
                    "detail": f"{n1} and {n2} censused the same "
                              "symbol table and share zero imported "
                              "names — the two censuses cannot both "
                              "be of this artifact",
                    "evidence_ids": [r1.evidence_id, r2.evidence_id],
                    "dissent_ids": [r1.evidence_id, r2.evidence_id]})
            else:
                agreed = True
    if agreed and not out:
        out.append({
            "name": "import-census overlap", "claim_key": "imports",
            "verdict": "agree",
            "detail": "independent import censuses overlap on the "
                      "same symbol names",
            "evidence_ids": [r.evidence_id for _n, r, _s in sources],
            "dissent_ids": []})
    return out


def _check_string_references(records: list[Any]) -> list[dict[str, Any]]:
    decompile_rec = _first(records, "native.decompile")
    strings_rec = _first(records, "native.strings")
    if decompile_rec is None or strings_rec is None:
        return []
    values = _string_values(strings_rec)
    hits = 0
    for snippet in (decompile_rec.result or {}).get("snippets") or []:
        code = snippet.get("code") or ""
        if any(len(v) >= 6 and v in code for v in values):
            hits += 1
    if not hits:
        return []
    return [{
        "name": "strings ↔ decompile references", "claim_key":
        "string_reference", "verdict": "agree",
        "detail": f"{hits} decompiled function(s) literally "
                  "reference strings from the deep string census",
        "evidence_ids": [decompile_rec.evidence_id,
                         strings_rec.evidence_id],
        "dissent_ids": []}]


def _check_firmware_consistency(records: list[Any]
                                ) -> list[dict[str, Any]]:
    census = _first(records, "firmware.census")
    inventory = _first(records, "firmware.inventory")
    if census is None or inventory is None:
        return []
    parsed = (census.result or {}).get("parsed") or {}
    cpio = parsed.get("cpio") or {}
    entries = cpio.get("entries")
    files_total = (inventory.result or {}).get("files_total")
    if entries and files_total is not None:
        if files_total == 0:
            return [{
                "name": "firmware census ↔ extraction", "claim_key":
                "firmware_inventory", "verdict": "contradict",
                "detail": f"the census read {entries} entries in "
                          "the container but extraction yielded "
                          "zero files — the census and the "
                          "extraction cannot both describe this "
                          "image",
                "evidence_ids": [census.evidence_id,
                                 inventory.evidence_id],
                "dissent_ids": [census.evidence_id,
                                inventory.evidence_id]}]
        return [{
            "name": "firmware census ↔ extraction", "claim_key":
            "firmware_inventory", "verdict": "agree",
            "detail": f"the census read {entries} container "
                      f"entries and extraction yielded "
                      f"{files_total} files — consistent",
            "evidence_ids": [census.evidence_id,
                             inventory.evidence_id],
            "dissent_ids": []}]
    return []


def _check_capture_voices(records: list[Any]) -> list[dict[str, Any]]:
    summary = _first(records, "traffic.summary")
    tshark = _first(records, "traffic.tshark_summary")
    if summary is None or tshark is None:
        return []
    out: list[dict[str, Any]] = []
    capinfos = (tshark.result or {}).get("capinfos") or {}
    t_packets = capinfos.get("packets")
    b_packets = (summary.result or {}).get("packets")
    if t_packets is not None and b_packets is not None:
        if int(t_packets) == int(b_packets):
            out.append({
                "name": "capture packet count", "claim_key":
                "capture_packets", "verdict": "agree",
                "detail": f"both dissectors count {b_packets} "
                          "packets",
                "evidence_ids": [summary.evidence_id,
                                 tshark.evidence_id],
                "dissent_ids": []})
        else:
            out.append({
                "name": "capture packet count", "claim_key":
                "capture_packets", "verdict": "contradict",
                "detail": f"the built-in reader counts {b_packets} "
                          f"packets; tshark counts {t_packets} — "
                          "the two reads of this capture disagree",
                "evidence_ids": [summary.evidence_id,
                                 tshark.evidence_id],
                "dissent_ids": [summary.evidence_id,
                                tshark.evidence_id]})
    t_protos = {str(k).lower() for k in
                ((tshark.result or {}).get("protocols") or {})}
    b_protos = {str(k).lower() for k in
                ((summary.result or {}).get("protocols") or {})}
    if t_protos and b_protos:
        if t_protos & b_protos:
            out.append({
                "name": "capture protocol census", "claim_key":
                "capture_protocols", "verdict": "agree",
                "detail": "both dissectors see the same protocols: "
                          + ", ".join(sorted(t_protos & b_protos)[:8]),
                "evidence_ids": [summary.evidence_id,
                                 tshark.evidence_id],
                "dissent_ids": []})
        else:
            out.append({
                "name": "capture protocol census", "claim_key":
                "capture_protocols", "verdict": "contradict",
                "detail": "the two dissectors share zero protocols "
                          "on the same capture — the reads cannot "
                          "both be of this file",
                "evidence_ids": [summary.evidence_id,
                                 tshark.evidence_id],
                "dissent_ids": [summary.evidence_id,
                                tshark.evidence_id]})
    return out


_CHECKS = (
    _check_needed_libraries, _check_import_overlap,
    _check_string_references, _check_firmware_consistency,
    _check_capture_voices,
)


def _run_checks(records: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for check in _CHECKS:
        try:
            out.extend(check(records))
        except Exception as exc:  # noqa: BLE001 - a broken check is
            # recorded, never allowed to dress itself as a verdict
            record_failure(f"reverse.verify.{check.__name__}", exc)
    return out


# ── the reconstruction ───────────────────────────────────────────────

def build_reconstruction(store: Any, subject_sha256: str, *,
                         mission_id: str = "") -> dict[str, Any]:
    """Build + verify the reconstruction of one subject from its
    evidence chain, and record the report itself as Evidence.

    Claims are derived only from the chain; cross-rung checks
    adjudicate them; contradicted claims are demoted into the
    conflicts list and opened as residual unknowns carrying both
    sides. The recorded report record's content address includes
    a fingerprint of the chain, so a rebuilt report over a grown
    chain is a new record, not a silent overwrite.
    """
    # the chain is rung evidence only: prior reconstruction
    # reports over this subject are syntheses, not sources — a
    # report must never become its own input (the fingerprint
    # below stays stable, so a rebuild over an unchanged chain
    # lands on the same record)
    records = [r for r in store.list_for_subject(subject_sha256)
               if r.predicate_type != "reconstruction.report"]
    if not records:
        raise ReconstructionError(
            f"no evidence on record for subject {subject_sha256} — "
            "a reconstruction needs a chain; inspect the target "
            "first")
    identify = _first(records, "format.identify") or records[0]
    subject = {
        "sha256": subject_sha256,
        "name": identify.subject_name,
        "size": identify.subject_size,
    }
    drafts: list[_Draft] = []
    _derive_identity(records, subject, drafts)
    _derive_native_behavior(records, drafts)
    _derive_firmware_behavior(records, store, drafts)
    _derive_traffic_behavior(records, drafts)
    checks = _run_checks(records)
    chain_ids = {r.evidence_id for r in records}
    claims: list[Claim] = []
    conflicts: list[dict[str, Any]] = []
    for draft in drafts:
        applicable = [
            c for c in checks
            if c["claim_key"] in draft.check_keys
            and set(c["evidence_ids"]) & set(draft.evidence_ids)]
        draft.checks = [{
            "name": c["name"], "verdict": c["verdict"],
            "detail": c["detail"]} for c in applicable]
        bad = next((c for c in applicable
                    if c["verdict"] == "contradict"), None)
        if bad is not None:
            conflicts.append({
                "claim": draft.text, "check": bad["name"],
                "detail": bad["detail"],
                "evidence_ids": list(draft.evidence_ids),
                "check_evidence_ids": list(bad["evidence_ids"]),
                "dissent_ids": list(bad.get("dissent_ids") or []),
            })
            continue
        if any(c["verdict"] == "agree" for c in applicable):
            draft.status = "corroborated"
        claims.append(draft.freeze())
    # demoted claims become unknowns BEFORE the report is built,
    # so the report carries their ids — the conflict is the
    # question, both sides ride on the unknown record
    opened: list[dict[str, Any]] = []
    for conflict in conflicts:
        unk = store.open_unknown(
            "reconstruction conflict: " + conflict["detail"]
            + f" — the claim {conflict['claim']!r} is demoted "
            "until the disagreement is resolved",
            subject_sha256=subject_sha256,
            raised_by=(conflict["check_evidence_ids"] or [""])[0],
            supporting=[e for e in conflict["evidence_ids"]
                        if e not in conflict["dissent_ids"]],
            contradicting=list(conflict["dissent_ids"]),
            required_authority="shipped-artifact",
            required_confidence="derived")
        conflict["unknown_id"] = unk.unknown_id
        opened.append(unk.to_dict())
    report = build_report(
        claims, subject=subject, chain_ids=chain_ids,
        checks=checks, conflicts=conflicts)
    report["coverage"] = {
        "evidence_records": len(records),
        "predicates": sorted({r.predicate_type for r in records}),
        "providers": sorted({r.provider_id for r in records}),
    }
    fingerprint = hashlib.sha256(
        "|".join(sorted(chain_ids)).encode("utf-8")).hexdigest()[:16]
    rec = store.record(
        predicate_type="reconstruction.report",
        operation="verify.reconstruction",
        provider_id=PROVIDER_ID, provider_name=PROVIDER_NAME,
        provider_version=PROVIDER_VERSION,
        parameters={"chain": fingerprint, "mission": mission_id},
        result={"counts": report["counts"],
                "conflicts": [c["claim"] for c in conflicts],
                "checks": [{"name": c["name"],
                            "verdict": c["verdict"]}
                           for c in checks]},
        subject_sha256=subject_sha256,
        subject_name=subject["name"], subject_size=subject["size"],
        confidence="derived", authority="analyst-inference",
        limitations=list(_REPORT_LIMITATIONS),
        links=sorted(chain_ids)[:128])
    report["report_evidence_id"] = rec.evidence_id
    report["conflict_unknowns"] = opened
    report["unknowns_open"] = [
        u.to_dict() for u in store.list_unknowns(
            status="open", limit=100000)
        if u.subject_sha256 == subject_sha256]
    return report


# ── render ───────────────────────────────────────────────────────────

def render_reconstruction(report: dict[str, Any]) -> str:
    if report.get("error"):
        return f"⚠️ {report['error']}"
    subj = report["subject"]
    counts = report["counts"]
    lines = [
        f"🧩 reconstruction — {subj['name']}",
        f"sha256 {subj['sha256'][:16]}… · {subj['size']} bytes · "
        f"{counts['claims']} claim(s) "
        f"({counts['corroborated']} corroborated), "
        f"{counts['conflicts']} conflict(s)",
    ]
    for claim in report["claims"]:
        mark = "✓" if claim["status"] == "corroborated" else "·"
        lines.append(
            f"  {mark} [{claim['kind']}] {claim['text']} "
            f"[{claim['claim_id']}]")
        lines.append(
            "      evidence: "
            + ", ".join(e[:19] + "…" for e in claim["evidence_ids"]))
    for conflict in report["conflicts"]:
        lines.append(
            f"  ✗ DEMOTED: {conflict['claim']}")
        lines.append(f"      conflict ({conflict['check']}): "
                     f"{conflict['detail']}")
        if conflict.get("unknown_id"):
            lines.append(
                f"      opened as unknown [{conflict['unknown_id']}]")
    checks = report.get("checks") or []
    if checks:
        lines.append("cross-rung checks:")
        for check in checks:
            lines.append(f"  [{check['verdict']}] {check['name']} — "
                         f"{check['detail']}")
    unknowns = report.get("unknowns_open") or []
    lines.append(f"open unknowns over this subject: {len(unknowns)}")
    for unk in unknowns[:8]:
        lines.append(f"  ? {unk['question']} [{unk['unknown_id']}]")
    if report.get("report_evidence_id"):
        lines.append(
            f"report recorded as {report['report_evidence_id']}")
    return "\n".join(lines)
