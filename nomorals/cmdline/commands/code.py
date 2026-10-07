"""``nm code`` — code run/review/test surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from pathlib import Path



def _cmd_code(args: argparse.Namespace, context: Any) -> int:
    """Route `nm code` to run / review / test."""
    words = list(args.task or [])
    if words and words[0] == "review":
        ref = words[1] if len(words) > 1 else None
        return _cmd_code_review(args, context, ref)
    if words and words[0] == "test":
        return _cmd_code_test(args, context)
    if words and words[0] == "approve":
        # nm code approve <plan-id> — approve a paused plan-mode plan
        # (created by an earlier `nm code` run) and execute it now.
        plan_id = words[1] if len(words) > 1 else ""
        if not plan_id:
            print("usage: nm code approve <plan-id>", file=sys.stderr)
            return 2
        return _cmd_code_approve(args, context, plan_id)
    task = " ".join(words).strip()
    if not task:
        print('usage: nm code "<task>" [--file F] [--accept CMD] [--max-iters N] [--root DIR]',
              file=sys.stderr)
        print('       nm code review [ref] [--root DIR]', file=sys.stderr)
        print('       nm code test [--changed] [--root DIR]', file=sys.stderr)
        return 2
    return _cmd_code_run(args, context, task)


def _cmd_code_approve(args: argparse.Namespace, context: Any,
                      plan_id: str) -> int:
    """Approve a paused plan-mode plan and execute it within its scope."""
    from ...agents.coding import CodingAgent

    root = str(Path(args.root).expanduser().resolve())
    agent = CodingAgent(context, root=root)
    result = agent.approve_and_execute(plan_id)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.ok else 1
    if result.ok:
        print(f"plan {plan_id} approved and executed — OK")
        print(f"files: {', '.join(result.files) or '(none)'}  "
              f"iterations: {result.iterations}  seconds: {result.seconds:.1f}")
    else:
        print(f"FAILED: {result.error}")
    return 0 if result.ok else 1


def _cmd_code_run(args: argparse.Namespace, context: Any, task: str) -> int:
    """Run a coding task directly through the CodingAgent (no orchestrator)."""
    from ...agents.coding import CodingAgent

    root = str(Path(args.root).expanduser().resolve())
    agent = CodingAgent(context, root=root)
    result = agent.run(
        task,
        filename=args.file,
        accept=args.accept,
        max_iterations=args.max_iters,
        plan_mode=args.plan_mode,
    )
    if result.needs_approval:
        # Plan mode paused the task BEFORE any file was touched. Show
        # the plan; nothing executes until the owner approves.
        print(result.plan_text)
        print(f"\nplan id: {result.plan_id}")
        print("approve with:  nm code approve <plan-id>")
        print("             (or re-run with --plan-mode never to skip the gate)")
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.ok else 1
    print(f"task: {task}")
    print(f"file: {args.file}  iterations: {result.iterations}  "
          f"seconds: {result.seconds:.1f}")
    if result.ok:
        print("OK — acceptance command passed")
    else:
        print(f"FAILED: {result.error}")
    # Always end the session with the diff on screen; committing is a
    # separate, explicit `nm` step — never automatic.
    from ...tools.git import git_diff
    try:
        diff = git_diff(None, [args.file], root)
        if diff["diff"].strip():
            print("\n--- diff ---")
            print(diff["diff"] if not diff["truncated"]
                  else diff["diff"] + "\n... [truncated]")
        else:
            print("\n(no changes)")
    except Exception as exc:  # noqa: BLE001 — diff is best-effort
        print(f"\n(could not render diff: {exc})")
    return 0 if result.ok else 1


def _critic_verdict(context: Any, diff_text: str) -> tuple[str, list[str]]:
    """Run the harsh reviewer over a diff.  Never raises — a failed review
    means 'no verdict', which is the conservative outcome."""
    if not diff_text.strip():
        return "no changes — nothing to review", []
    try:
        from ...agents.coding import CodingAgent, _DIFF_REVIEW_FOCUS

        flaws = CodingAgent(context)._review_flaws(
            "review the working tree diff", diff_text,
            focus=_DIFF_REVIEW_FOCUS)
    except Exception:  # noqa: BLE001 — review must never break the CLI
        return "critic unavailable", []
    if flaws:
        return "VERDICT: FAIL — do not commit as-is", flaws
    return "VERDICT: PASS", []


def _cmd_code_review(args: argparse.Namespace, context: Any, ref: str | None) -> int:
    """Render a readable diff of the working tree (or ref) for review.

    Phase C: the diff is ALWAYS shown with the critic's verdict before
    any commit is proposed, and the owner gets an explicit commit prompt.
    """
    from ...tools.git import git_diff, git_status

    root = str(Path(args.root).expanduser().resolve())
    try:
        status = git_status(root)
        diff = git_diff(ref, None, root)
    except Exception as exc:
        print(f"review failed: {exc}", file=sys.stderr)
        return 1
    body = diff["diff"]
    if args.json:
        verdict, flaws = _critic_verdict(context, body)
        print(json.dumps({"status": status, "diff": diff, "verdict": verdict,
                          "flaws": flaws}, indent=2, default=str))
        return 0
    print(f"repo: {root}  branch: {status['branch']}")
    dirty = status["staged"] + status["unstaged"] + status["untracked"]
    print(f"dirty files ({len(dirty)}): "
          + (", ".join(dirty[:20]) if dirty else "none"))
    if not body.strip():
        print("\n(no diff)")
        return 0
    print(f"\n--- diff{f' vs {ref}' if ref else ''} "
          f"({diff['bytes']} bytes{', truncated' if diff['truncated'] else ''}) ---")
    print(body if not diff["truncated"] else body + "\n... [truncated at 50KB]")
    # ── Phase C: critic verdict + explicit commit prompt ──
    verdict, flaws = _critic_verdict(context, body)
    print(f"\ncritic: {verdict}")
    for flaw in flaws:
        print(f"  - {flaw}")
    if dirty:
        print("\nCommit these changes?")
        print(f'  git -C "{root}" add -A && git -C "{root}" commit -m "<message>"')
        print("  (nothing is committed until you run it)")
    return 0


def _cmd_code_test(args: argparse.Namespace, context: Any) -> int:
    """Run the test suite through the pytest-aware runner (Phase B)."""
    from ...tools.pytest_runner import format_test_result, run_tests

    root = str(Path(args.root).expanduser().resolve())
    result = run_tests(changed_only=bool(args.changed), repo=root)
    print(format_test_result(result))
    if result.get("failed"):
        return 1
    return 0 if result.get("ok") else 1
