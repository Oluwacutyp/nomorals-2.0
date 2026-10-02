"""``nm improve`` — self-improvement runs."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from pathlib import Path



def _inbox_obj(context: Any) -> Any:
    """Build the drop-in Inbox for the CLI context's workspace."""
    from ...workspace.inbox import Inbox

    root = Path(context.settings.workspace_dir)
    # Prompt 09: wire the vision hook only when a router exists — otherwise
    # the image intent degrades to probe-only instead of parking forever.
    hook = None
    if getattr(context, "router", None) is not None:
        from ...tools.vision import make_inbox_vision_hook

        hook = make_inbox_vision_hook(context)
    return Inbox(root, db=context.db, vision=hook)


def _cmd_improve(args: argparse.Namespace, context: Any) -> int:
    """Route `nm improve` to status / lessons / skill / rollback."""
    as_json = getattr(args, "json", False)
    action = args.improve_action
    try:
        if action == "status":
            from ...agents.skill_evolution import SkillEvolutionLoop
            payload = SkillEvolutionLoop(context).status()
            if as_json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                print(f"mode: {payload['mode']}")
                cands = payload.get("candidates", [])
                print(f"candidates: {len(cands)}")
                for c in cands[:10]:
                    print(f"  {c['skill']}: {c['count']} failures")
                print("recent edits:")
                for e in payload.get("recent_edits", [])[:10]:
                    print(f"  {e['id']} {e['skill_name']} [{e['status']}] "
                          f"(mode={e['mode']})")
                print(f"lessons: {payload.get('lessons', {})}")
            return 0
        if action == "lessons":
            from ...agents.failure import FailureAnalyzer
            analyzer = FailureAnalyzer(context)
            if args.query:
                lessons = analyzer.rank(args.query, limit=args.limit)
            else:
                lessons = analyzer.recent(limit=args.limit)
            payload = [l.to_dict() for l in lessons]
            if as_json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                if not payload:
                    print("no lessons yet")
                for l in payload:
                    print(f"{l['id']} [{l['category']}] "
                          f"usefulness={l['usefulness']} "
                          f"surfaced={l['times_surfaced']} "
                          f"{'DEMOTED ' if l['demoted'] else ''}"
                          f"{(l['prevention'] or l['lesson'])[:100]}")
            return 0
        if action == "skill":
            from ...agents.skill_evolution import (
                SkillEvolutionLoop, SkillEvolutionError)
            loop = SkillEvolutionLoop(context)
            try:
                proposal = loop.propose(args.name)
            except SkillEvolutionError as exc:
                print(f"improve: {exc}", file=sys.stderr)
                return 1
            if args.propose:
                passed, results = loop.gate(proposal)
                payload = {
                    "skill_name": proposal["skill_name"],
                    "target": f"{proposal['target_kind']}:{proposal['target_ref']}",
                    "changed_lines": proposal["changed_lines"],
                    "before_hash": proposal["before_hash"],
                    "after_hash": proposal["after_hash"],
                    "fingerprint": proposal["fingerprint"],
                    "gate_passed": passed,
                    "gate": results,
                    "diff": proposal["diff"],
                }
                if as_json:
                    print(json.dumps(payload, indent=2, default=str))
                else:
                    print(f"proposal for {proposal['skill_name']}: "
                          f"{proposal['changed_lines']} changed lines, "
                          f"gate {'PASSED' if passed else 'FAILED'}")
                    for phase, r in results.items():
                        print(f"  {phase}: "
                              f"{'ok' if r.get('ok') else 'FAIL'} — "
                              f"{r.get('detail', '')[:120]}")
                    print("--- diff ---")
                    print(proposal["diff"][:3000])
                return 0
            print("nothing to do: pass --propose to draft a skill-edit "
                  "proposal", file=sys.stderr)
            return 2
        if action == "rollback":
            from ...agents.skill_canary import CanaryRollout
            out = CanaryRollout(context).restore_version(args.skill,
                                                        args.hash)
            if as_json:
                print(json.dumps(out, indent=2, default=str))
            elif out.get("ok"):
                print(f"{args.skill} restored to version {args.hash}")
            else:
                print(f"improve: {out.get('error')}", file=sys.stderr)
                return 1
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"improve: {exc}", file=sys.stderr)
        return 1
    print(f"unknown improve action: {action}", file=sys.stderr)
    return 2
