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

Pass 4: missions + reconstruction verify. A question plus
targets becomes a mission (:mod:`.missions`): Devon decomposes
it over the capability matrix, works the rungs in dependency
order under a journal she can resume, lets the residual
unknowns re-drive the plan, and closes with a reconstruction
(:mod:`.verify`) — claims tied to evidence ids, cross-rung
checks that corroborate or demote, contradictions opened as
unknowns rather than smoothed.

The dynamic rung (:mod:`.dynamic`, Frida + GDB) dissolves the
Pass 4 gate: questions only running a target could answer are
answered by running it — as a copy, in a per-run workdir,
under the namespace sandbox with the network dropped. The
sandbox is part of the rung, not a posture question: with no
namespace backend the rung refuses rather than running
unsandboxed. The dynamic rung is built, and the mission planner
schedules dynamic steps when a dynamic rung is available;
when none is, the unknown names the missing engine.
"""

from __future__ import annotations

from typing import Any

from . import (
    android_graph,
    dynamic,
    firmware,
    ghidra,
    lightbin,
    missions,
    provision,
    traffic,
    verify,
)
from .engine import (
    ReEngine,
    capability_matrix,
    render_audit,
    render_dynamic_card,
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
from .missions import render_mission, render_mission_list, render_mission_status
from .verify import Claim, ReconstructionError, render_reconstruction

__all__ = [
    "ReEngine", "capability_matrix",
    "Evidence", "EvidenceStore", "Unknown", "UnknownCloseError",
    "AUTHORITIES", "CONFIDENCES",
    "provision", "ghidra", "lightbin", "android_graph",
    "firmware", "traffic", "missions", "verify", "dynamic",
    "Claim", "ReconstructionError",
    "render_status", "render_inspect_card", "render_evidence_card",
    "render_unknowns", "render_audit", "render_provision",
    "render_graph_answer", "render_firmware_card",
    "render_traffic_card", "render_mission", "render_mission_list",
    "render_mission_status", "render_reconstruction",
    "render_dynamic_card",
    "register",
]


def register(registry: Any) -> None:
    """Register the ``re`` read tool and the ``re_inspect`` run tool.

    ``re`` reads the evidence store (status, evidence records with
    their provenance chains, the unknowns registry, the unknowns
    audit), answers Android graph queries as Evidence, and reads
    the Pass 4 mission journal. ``re_inspect`` performs
    inspections — each one records a chain of Evidence and
    registers the residual unknowns the parse raises — runs the
    self-provisioning drive for the deep rungs, builds Android
    reference graphs, runs and resumes Pass 4 missions, and
    closes unknowns under the store's lifecycle rules.
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
            "or a method ('Foo.bar')) | mission (one Pass 4 "
            "mission by mission_id: its journaled plan, per-step "
            "results, per-target verified reconstructions and "
            "parked unknowns) | missions (the mission journal, "
            "most recent first). Evidence records are "
            "content-addressed (ev:<sha256>) and carry confidence, "
            "authority and limitations on the record itself."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — status|evidence|unknowns|audit|graph"
                      "|mission|missions",
            "evidence_id": "str — evidence id or unique prefix "
                           "(action=evidence)",
            "status": "str — unknowns filter: open|resolved|"
                      "contradicted (action=unknowns)",
            "limit": "str — max rows for lists (default 20)",
            "path": "str — APK or jadx output dir (action=graph)",
            "query": "str — callers|callees|class|search "
                     "(action=graph)",
            "target": "str — class or Class.method (action=graph)",
            "mission_id": "str — mission id or unique prefix "
                          "(action=mission)",
        },
    )
    def re_tool(*, action: str = "status", evidence_id: str = "",
                status: str = "", limit: str = "20",
                path: str = "", query: str = "",
                target: str = "",
                mission_id: str = "") -> dict[str, Any]:
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
        if act == "mission":
            if not (mission_id or "").strip():
                return {"error": "action=mission needs a mission_id "
                                 "(action=missions lists them)"}
            return engine.mission(mission_id.strip())
        if act == "missions":
            return engine.mission_list(limit=n)
        return {"error": f"unknown re action {act!r} — "
                         "status|evidence|unknowns|audit|graph|"
                         "mission|missions"}

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
            "route) | off (built-in rungs only)) | dynamic (path = "
            "a local binary to RUN under the dynamic rung — "
            "Frida trace or GDB session — as a copy in a per-run "
            "workdir under the namespace sandbox, network "
            "dropped; rung = auto (Frida when ready, else GDB) | "
            "frida | gdb; script = a Frida JS file for the trace; "
            "args = comma-separated target argv; timeout_s = the "
            "run budget; provision = true provisions a missing "
            "engine first. The run folds into Evidence with the "
            "rung named on the record; with no namespace sandbox "
            "the rung refuses rather than running unsandboxed) | "
            "provision (run the "
            "self-provisioning drive — pinned Temurin JDK, "
            "pinned Ghidra, pinned unblob (PyPI digests, "
            "--require-hashes), pinned Frida (PyPI digests, "
            "--require-hashes), GDB through apt — and report each "
            "engine's state with its attempts) | "
            "graph (path = an APK or jadx output dir: build the "
            "Android class/method/reference graph and record its "
            "census as Evidence) | mission (Pass 4: run a "
            "brain-directed RE mission — question + targets "
            "(comma-separated paths): Devon decomposes the "
            "question over the capability matrix, works the rungs "
            "in dependency order under a journaled plan, lets the "
            "residual unknowns re-drive the plan, and closes with "
            "a verified reconstruction per target — every claim "
            "evidence-tied, contradictions demoted to unknowns; "
            "sub_questions adds brain sub-questions through the "
            "same planner; graph_target names the class/method "
            "for a graph-intent query step; with mission_id set "
            "instead of a question, the mission RESUMES from its "
            "journal. A question only running a target could "
            "answer is answered by the dynamic rung (action="
            "dynamic) — the target runs as a copy under the "
            "namespace sandbox, network dropped — and when no "
            "dynamic rung can run, it opens a dynamic_required "
            "unknown naming the missing engine, not a gate) | close (unknown_id + status="
            "resolved|contradicted: resolved takes a resolution "
            "text, contradicted takes at least one contradicting "
            "evidence id in evidence_ids, and the stated "
            "resolution_authority / resolution_confidence must meet "
            "what the unknown requires — the store refuses a weaker "
            "close and says what it takes)."
        ),
        capability=Capability.DB_WRITE,
        parameters={
            "action": "str — inspect|provision|graph|mission|close|"
                      "dynamic",
            "path": "str — file to inspect (action=inspect), or "
                    "APK / jadx output dir (action=graph), or "
                    "binary to run (action=dynamic)",
            "rung": "str — auto|frida|gdb (action=dynamic, "
                    "default auto)",
            "script": "str — Frida JS file for the trace "
                      "(action=dynamic)",
            "args": "str — comma-separated target argv "
                    "(action=dynamic)",
            "timeout_s": "str — the run budget in seconds "
                         "(action=dynamic)",
            "provision": "str — true provisions a missing dynamic "
                         "engine first (action=dynamic)",
            "deep": "str — auto|provision|off (action=inspect, "
                    "default auto)",
            "question": "str — the mission question "
                        "(action=mission)",
            "targets": "str — comma-separated target paths "
                       "(action=mission)",
            "sub_questions": "str — comma-separated extra "
                             "sub-questions for the planner "
                             "(action=mission)",
            "graph_target": "str — class or Class.method for a "
                            "graph query step (action=mission)",
            "mission_id": "str — mission id or unique prefix to "
                          "resume (action=mission)",
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
                        question: str = "", targets: str = "",
                        sub_questions: str = "",
                        graph_target: str = "", mission_id: str = "",
                        unknown_id: str = "", status: str = "",
                        resolution: str = "", evidence_ids: str = "",
                        resolution_authority: str = "",
                        resolution_confidence: str = "",
                        rung: str = "auto", script: str = "",
                        args: str = "", timeout_s: str = "",
                        provision: str = ""
                        ) -> dict[str, Any]:
        act = (action or "inspect").strip().lower()
        engine = ReEngine(context)
        if act == "inspect":
            mode = (deep or "auto").strip().lower()
            if mode not in ("auto", "provision", "off"):
                mode = "auto"
            return engine.inspect_path(path, deep=mode)
        if act == "dynamic":
            argv = tuple(a.strip() for a in (args or "").split(",")
                         if a.strip())
            try:
                timeout = float(timeout_s) if timeout_s else None
            except ValueError:
                return {"error": f"timeout_s {timeout_s!r} is not "
                                 "a number"}
            return engine.run_dynamic(
                path, rung=(rung or "auto").strip().lower(),
                script=(script or "").strip() or None,
                args=argv, timeout=timeout,
                provision_missing=(provision or "").strip().lower()
                in ("true", "1", "yes"))
        if act == "provision":
            return engine.provision_engines()
        if act == "graph":
            return engine.android_graph_build(path)
        if act == "mission":
            if (mission_id or "").strip():
                return engine.resume_mission(mission_id.strip())
            paths = [p.strip() for p in (targets or "").split(",")
                     if p.strip()]
            subs = tuple(s.strip() for s in
                         (sub_questions or "").split(",") if s.strip())
            return engine.run_mission(
                question, paths, extra_questions=subs,
                graph_target=(graph_target or "").strip())
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
                         "inspect|provision|graph|mission|close|"
                         "dynamic"}
