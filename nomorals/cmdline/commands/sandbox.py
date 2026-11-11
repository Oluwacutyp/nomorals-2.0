"""``nm sandbox proof`` — prove the sandbox backend enforces its claims.

Runs the hostile-probe battery in tools/sandbox_proof.py against the
real ``run_sandboxed`` and prints one line per probe plus a final
verdict line naming the backend and its flags. Exit 0 when every
probe passes, 1 when any probe fails or a flag lies about its
observation, 2 when the platform cannot run sandboxed probes at all.

Thin shell over the proof module — the same facade the test suite
calls. There is no chat verb for this: it is a host-diagnostic
command, and the chat catalog is not its surface.
"""

from __future__ import annotations

from typing import Any


def _cmd_sandbox(args: Any, context: Any) -> int:
    """Route `nm sandbox` subcommands."""
    from ...tools.sandbox_proof import render_proof, run_proof

    action = getattr(args, "sandbox_action", "proof")
    if action != "proof":
        print(f"unknown nm sandbox action {action!r}")
        return 2
    report = run_proof()
    print(render_proof(report))
    if report["verdict"] == "pass":
        return 0
    if report["verdict"] == "not-applicable":
        return 2
    return 1
