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
ApkForge's proven ELF parser.

Pass 2: the self-provisioned deep providers. Where REA makes the
human fetch every engine, Devon provisions her own — pinned
Ghidra 12.2 + Temurin JDK 25, sha256-verified, probe-executed
(:mod:`.provision`) — and drives Ghidra headless for functions,
call graphs and decompiled pseudocode folded into Evidence
(:mod:`.ghidra`); a binutils one-shot light rung covers fast
triage (:mod:`.lightbin`); and a class/method/reference graph
over ApkForge's jadx output answers who-calls-what as Evidence
(:mod:`.android_graph`). Inspections route to the deepest rung
that can answer and say which one did.

Pass 3: firmware + traffic. Firmware images route (by content)
to a built-in signature census plus the self-provisioned unblob
extraction engine — pinned through PyPI's published digests —
with a bounded native cpio/tar extractor behind it and
extraction-bomb caps enforced (:mod:`.firmware`). Captures —
pcap/pcapng, HAR, mitmproxy flow files, and the JSONL of Devon's
own mitmproxy capture addon — route to built-in readers with a
probed TShark second voice (:mod:`.traffic`). Nothing extracted
or captured is ever executed; encrypted content is recorded as
an unknown, never guessed.
"""

from __future__ import annotations

from typing import Any

from . import android_graph, firmware, ghidra, lightbin, provision, traffic
from .engine import (
    ReEngine,
    capability_matrix,
    render_audit,
    render_evidence_card,
    render_firmware_card,
    render_graph_answer,
    render_inspect_card,
    render_provision,
    render_status,
    render_traffic_card,
    render_unknowns,
)
from .evidence import AUTHORITIES, CONFIDENCES, Evidence, EvidenceStore, Unknown, UnknownCloseError

__all__ = [
    "ReEngine", "capability_matrix",
    "Evidence", "EvidenceStore", "Unknown", "UnknownCloseError",
    "AUTHORITIES", "CONFIDENCES",
    "provision", "ghidra", "lightbin", "android_graph",
    "firmware", "traffic",
    "render_status", "render_inspect_card", "render_evidence_card",
    "render_unknowns", "render_audit", "render_provision",
    "render_graph_answer", "render_firmware_card",
    "render_traffic_card",
    "register",
]


def register(registry: Any) -> None:
    """Register the ``re`` read tool and the ``re_inspect`` run tool.

    ``re`` reads the evidence store (status, evidence records with
    their provenance chains, the unknowns registry, the unknowns
    audit) and answers Android graph queries as Evidence.
    ``re_inspect`` performs inspections — each one records a chain
    of Evidence and registers the residual unknowns the parse
    raises — runs the self-provisioning drive for the deep rungs,
    builds Android reference graphs, and closes unknowns under
    the store's lifecycle rules.
    """
    from ...core.policy import Capability

    context = registry.context

    @registry.register(
        "re",
        description=(
            "Reverse-engineering evidence store, read side. "
            "action=status (engine capability matrix — built-ins "
            "probe-executed, Pass 2 engines carry real provisioning "
            "state: ready | provisionable | unavailable-with-reason "
            "— plus store counts) | evidence (one Evidence record "
            "by id or unique prefix, with its provenance chain; "
            "empty id lists the most recent records) | unknowns "
            "(the residual-unknown registry: status=open|resolved|"
            "contradicted, empty status lists all) | audit "
            "(unknowns audit: counts, the strongest evidence on "
            "record, and which open unknowns no recorded evidence "
            "could close yet) | graph (one Android reference-graph "
            "answer as a recorded Evidence record: path = an APK "
            "or a jadx output directory, query = callers|callees|"
            "class|search, target = a class ('com.x.Foo' or 'Foo') "
            "or a method ('Foo.bar')). Evidence records are "
            "content-addressed (ev:<sha256>) and carry confidence, "
            "authority and limitations on the record itself."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — status|evidence|unknowns|audit|graph",
            "evidence_id": "str — evidence id or unique prefix "
                           "(action=evidence)",
            "status": "str — unknowns filter: open|resolved|"
                      "contradicted (action=unknowns)",
            "limit": "str — max rows for lists (default 20)",
            "path": "str — APK or jadx output dir (action=graph)",
            "query": "str — callers|callees|class|search "
                     "(action=graph)",
            "target": "str — class or Class.method (action=graph)",
        },
    )
    def re_tool(*, action: str = "status", evidence_id: str = "",
                status: str = "", limit: str = "20",
                path: str = "", query: str = "",
                target: str = "") -> dict[str, Any]:
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
        if act == "graph":
            return engine.android_query(
                path, (query or "callers").strip().lower(),
                target)
        return {"error": f"unknown re action {act!r} — "
                         "status|evidence|unknowns|audit|graph"}

    @registry.register(
        "re_inspect",
        description=(
            "Reverse-engineering inspections + provisioning + "
            "unknown closes, write side. action=inspect (path = a "
            "local binary, firmware image or capture file — "
            "routing is by content, never by name: the built-in "
            "static spine parses ELF, PE and Mach-O natively; a "
            "firmware image goes to the firmware rung (signature "
            "census + the self-provisioned unblob extraction "
            "engine + bounded native cpio/tar extraction, "
            "extraction-bomb caps enforced, nothing extracted is "
            "executed, credential-shaped strings flagged as "
            "observations with redacted values); a capture "
            "(pcap/pcapng, HAR, mitmproxy flow file, Devon-addon "
            "JSONL) goes to the traffic rung (built-in dissect: "
            "protocol census, conversations, DNS, TLS handshake "
            "facts incl. SNI and certificate subjects, cleartext "
            "HTTP; TShark as a probed second voice when present; "
            "encrypted payloads are recorded as unknowns, never "
            "guessed); binaries route on to the deepest Pass 2 "
            "rung that can answer — the self-provisioned Ghidra "
            "headless rung or the binutils light rung — and every "
            "route records a linked chain of Evidence, registers "
            "its residual unknowns, and names the rung that "
            "answered; deep = auto (deepest available now) | "
            "provision (provision missing engines first, then "
            "route) | off (built-in rungs only); static only, "
            "nothing is executed) | provision (run the "
            "self-provisioning drive — pinned Temurin JDK, "
            "pinned Ghidra, pinned unblob (PyPI digests, "
            "--require-hashes) — and report each engine's state "
            "with its attempts) | "
            "graph (path = an APK or jadx output dir: build the "
            "Android class/method/reference graph and record its "
            "census as Evidence) | close (unknown_id + status="
            "resolved|contradicted: resolved takes a resolution "
            "text, contradicted takes at least one contradicting "
            "evidence id in evidence_ids, and the stated "
            "resolution_authority / resolution_confidence must meet "
            "what the unknown requires — the store refuses a weaker "
            "close and says what it takes)."
        ),
        capability=Capability.DB_WRITE,
        parameters={
            "action": "str — inspect|provision|graph|close",
            "path": "str — file to inspect (action=inspect), or "
                    "APK / jadx output dir (action=graph)",
            "deep": "str — auto|provision|off (action=inspect, "
                    "default auto)",
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
                        deep: str = "auto",
                        unknown_id: str = "", status: str = "",
                        resolution: str = "", evidence_ids: str = "",
                        resolution_authority: str = "",
                        resolution_confidence: str = ""
                        ) -> dict[str, Any]:
        act = (action or "inspect").strip().lower()
        engine = ReEngine(context)
        if act == "inspect":
            mode = (deep or "auto").strip().lower()
            if mode not in ("auto", "provision", "off"):
                mode = "auto"
            return engine.inspect_path(path, deep=mode)
        if act == "provision":
            return engine.provision_engines()
        if act == "graph":
            return engine.android_graph_build(path)
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
                         "inspect|provision|graph|close"}
