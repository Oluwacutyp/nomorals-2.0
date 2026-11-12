"""``nm re`` — the reverse-engineering organ: evidence-led static
inspection of binaries, firmware images and captures.

status | inspect <path> [--deep] | provision | graph <path>
[query target] | evidence [id] | unknowns | audit |
close <unknown-id> <resolution…> | mission <question> -- <targets…>
| missions | mission-status <id> | mission-resume <id> — thin
shell over ReEngine, the same facade the ``re`` registry tools
and the ``/re`` chat command call. Pass 2: inspections route to
the deepest rung that can answer (the self-provisioned Ghidra
headless rung, the binutils light rung, the built-in spine),
``provision`` runs the self-provisioning drive for those rungs,
and ``graph`` answers Android reference questions over ApkForge's
jadx output. Pass 3: inspections also route by content to the
firmware rung (census + self-provisioned unblob extraction +
native extractor) and the traffic rung (pcap/pcapng/HAR/
mitmproxy-flow readers + a probed TShark second voice), and
``provision`` drives the unblob pin alongside the deep rungs.
Pass 4: ``mission`` plans + works a brain-directed mission over
the rungs and closes with a verified reconstruction per target;
``missions`` / ``mission-status`` / ``mission-resume`` read and
resume the mission journal. The dynamic rung: ``dynamic <path>``
runs one target under Frida or GDB — a copy in a per-run
workdir under the namespace sandbox, network dropped — and folds
the run into Evidence with the rung named.
"""

from __future__ import annotations

from typing import Any


def _cmd_re(args: Any, context: Any) -> int:
    """Route `nm re` subcommands."""
    from ...agents.reverse import (
        ReEngine,
        render_audit,
        render_evidence_card,
        render_graph_answer,
        render_mission,
        render_mission_list,
        render_mission_status,
        render_provision,
        render_status,
        render_unknowns,
    )

    engine = ReEngine(context)
    action = args.re_action

    if action == "status":
        print(render_status(engine.status()))
        return 0

    if action == "mission":
        targets = [t for t in (args.targets or []) if t != "--"]
        res = engine.run_mission(args.question, targets)
        if res.get("error"):
            print(res["error"])
            return 2
        print(render_mission(res))
        return 0

    if action == "missions":
        print(render_mission_list(engine.mission_list(limit=20)))
        return 0

    if action == "mission-status":
        payload = engine.mission(args.mission_id)
        if payload.get("error"):
            print(payload["error"])
            return 2
        print(render_mission_status(payload))
        return 0

    if action == "mission-resume":
        res = engine.resume_mission(args.mission_id)
        if res.get("error"):
            print(res["error"])
            return 2
        print(render_mission(res))
        return 0

    if action == "dynamic":
        timeout = getattr(args, "timeout", 0) or 0
        res = engine.run_dynamic(
            args.path,
            rung=getattr(args, "rung", "auto") or "auto",
            script=getattr(args, "script", "") or None,
            args=tuple(getattr(args, "args", []) or ()),
            timeout=(float(timeout) if timeout else None),
            provision_missing=bool(
                getattr(args, "provision", False)))
        if res.get("error"):
            print(res["error"])
            return 2
        print(res["card_text"])
        return 0

    if action == "inspect":
        res = engine.inspect_path(
            args.path,
            deep="provision" if getattr(args, "deep", False)
            else "auto")
        if res.get("error"):
            print(res["error"])
            return 2
        print(res["card_text"])
        return 0

    if action == "provision":
        print(render_provision(engine.provision_engines()))
        return 0

    if action == "graph":
        query = getattr(args, "query", "") or ""
        if query:
            payload = engine.android_query(
                args.path, query, getattr(args, "target", "") or "")
        else:
            payload = engine.android_graph_build(args.path)
        if payload.get("error"):
            print(payload["error"])
            return 2
        print(render_graph_answer(payload))
        return 0

    if action == "evidence":
        ref = getattr(args, "evidence_id", "") or ""
        if ref:
            payload = engine.evidence(ref)
            if payload.get("error"):
                print(payload["error"])
                return 2
            print(render_evidence_card(payload))
            return 0
        payload = engine.evidence_list(limit=20)
        rows = payload["evidence"]
        if not rows:
            print("no evidence recorded yet")
            return 0
        for row in rows:
            print(f"{row['evidence_id']} · {row['predicate_type']} · "
                  f"{row['subject_name']} · {row['confidence']}/"
                  f"{row['authority']}")
        return 0

    if action == "unknowns":
        payload = engine.unknowns(status=getattr(args, "status", "")
                                  or "open")
        print(render_unknowns(payload["unknowns"],
                              title="residual unknowns"))
        return 0

    if action == "audit":
        print(render_audit(engine.audit()))
        return 0

    if action == "close":
        out = engine.close_unknown(
            args.unknown_id, status=args.status_kind,
            resolution=" ".join(args.resolution or []),
            resolution_authority=args.authority,
            resolution_confidence=args.confidence)
        if out.get("error"):
            print(out["error"])
            return 2
        unk = out["unknown"]
        print(f"closed {unk['unknown_id']} as {unk['status']}: "
              f"{unk['resolution']}")
        return 0

    print(f"unknown nm re action {action!r}")
    return 2
