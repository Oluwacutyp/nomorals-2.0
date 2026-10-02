"""``nm run`` / ``nm ask`` — agent runs and questions."""

from __future__ import annotations

import argparse
import json
from typing import Any



def _cmd_run(args: argparse.Namespace, context: Any) -> int:
    from ...agents.orchestrator import MasterOrchestrator

    # Warn if using mock provider
    active = context.router.active_model
    is_mock = not active or active == "mock"
    if is_mock:
        print("⚠️  No model configured. Using offline mock provider (outputs will be empty).")
        print("   Run `nm models` to see available models, or `nm setup` for guided setup.\n")
    
    goal = " ".join(args.goal)
    orchestrator = MasterOrchestrator(context, max_steps=args.max_steps)
    result = orchestrator.run(goal, reflect=not args.no_reflect)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.ok else 1
    print(f"plan ({len(result.plan.steps)} steps): {result.plan.rationale or 'n/a'}")
    for step in result.plan.steps:
        # Mark mock steps distinctly
        mock_tag = " [mock]" if is_mock else ""
        print(f"  - {step.name} [{step.role}/{step.kind.value}]{mock_tag} deps={step.depends_on}")
    print(f"\nreport: {result.report.to_dict()}")
    print(f"\n{result.answer}")
    if result.lessons:
        print("\nlessons:")
        for lesson in result.lessons:
            print(f"  - {lesson}")
    return 0 if result.ok else 1


def _cmd_ask(args: argparse.Namespace, context: Any) -> int:
    from ...llm.base import Message, SamplingParams

    if args.model:
        context.router.set_active(args.model)
    
    # Warn if using mock provider
    active = context.router.active_model
    if not active or active == "mock":
        print("⚠️  No model configured. Using offline mock provider (outputs will be gibberish).")
        print("   Run `nm models` to see available models, or `nm setup` for guided setup.\n")
    
    messages = ([Message.system(args.system)] if args.system else []) + [
        Message.user(" ".join(args.prompt))
    ]
    response = context.router.chat(
        messages, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    )
    if args.json:
        print(json.dumps(response.to_dict(), indent=2, default=str))
    else:
        print(response.text or f"[no output: {response.error}]")
    return 0 if response.ok else 1
