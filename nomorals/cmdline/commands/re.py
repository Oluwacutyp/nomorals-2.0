"""``nm re`` — the reverse-engineering organ: evidence-led static
inspection of binaries.

status | inspect <path> [--deep] | provision | graph <path>
[query target] | evidence [id] | unknowns | audit |
close <unknown-id> <resolution…> — thin shell over ReEngine, the
same facade the ``re`` registry tools and the ``/re`` chat command
call. Pass 2: inspections route to the deepest rung that can
answer (the self-provisioned Ghidra headless rung, the binutils
light rung, the built-in spine), ``provision`` runs the
self-provisioning drive for those rungs, and ``graph`` answers
Android reference questions over ApkForge's jadx output.
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
        render_provision,
        render_status,
        render_unknowns,
    )

    engine = ReEngine(context)
    action = args.re_action

    if action == "status":
        print(render_status(engine.status()))
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
