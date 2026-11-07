"""``nm braineval`` — the brain eval pack: grade the models Devon's
brain directs against the 110-case measuring stick (capability /
freedom / Devon tool fluency), through her own router machinery.

run [--rung groq|chain|local|mock] [--model M] [--base-url U]
    [--suites capability,refusal,tools] [--limit N]
    Execute the pack on one rung and journal the run. groq is the
    recorded baseline rung; local grades a llama.cpp server (the rung
    a locally served model runs through); mock is a machinery smoke
    only, never a baseline.
last            The most recent run's scorecard.
compare [A B]   Two runs side by side (default: the last two).
status          Pack integrity + store + last run at a glance.

Thin shell over BrainEvalEngine — the same facade the ``brain_eval``
registry tool calls.
"""

from __future__ import annotations

from typing import Any


def _cmd_braineval(args: Any, context: Any) -> int:
    """Route `nm braineval` subcommands."""
    from ...agents.braineval import BrainEvalEngine
    from ...agents.braineval.report import render_run

    engine = BrainEvalEngine(context)
    action = args.braineval_action

    if action == "status":
        print(engine.render_status())
        return 0

    if action == "run":
        suites = None
        if getattr(args, "suites", ""):
            suites = [s.strip() for s in args.suites.split(",") if s.strip()]
        run = engine.run(
            args.rung,
            model=getattr(args, "model", "") or "",
            base_url=getattr(args, "base_url", "") or "",
            suites=suites,
            limit=int(getattr(args, "limit", 0) or 0))
        if run.get("error"):
            print(run["error"])
            return 2
        print(render_run(run))
        return 0

    if action == "last":
        print(engine.render_last())
        return 0

    if action == "compare":
        print(engine.render_compare(
            getattr(args, "run_a", "") or "",
            getattr(args, "run_b", "") or ""))
        return 0

    print(f"unknown nm braineval action {action!r}")
    return 2
