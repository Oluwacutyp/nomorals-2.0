"""``nm swarm`` — swarm orchestration."""

from __future__ import annotations

import argparse
import sys
from typing import Any
from ...llm.brain import brain_for
from ..emit import _emit

from ...core.logging_setup import get_logger

_log = get_logger(__name__)



def _cmd_swarm(args: argparse.Namespace, context: Any) -> int:
    """Route `nm swarm` to roles / run / debate."""
    from ...agents.debate import Debate, Issue, WorkArtifact
    from ...agents.fanout import fan_in, fan_out
    from ...agents.role_specs import default_registry

    action = args.swarm_action
    try:
        if action == "roles":
            registry = default_registry()
            payload = {"roles": {
                name: registry.resolve(name).to_dict()
                for name in registry.names()}}
            lines = []
            for name in registry.names():
                spec = registry.resolve(name)
                lines.append(
                    f"- {name}: {spec.description}\n"
                    f"  read_only={spec.read_only} "
                    f"budget={spec.budget.wall_seconds:.0f}s/"
                    f"{spec.budget.tokens}tok\n"
                    f"  tools: {', '.join(spec.tool_allowlist)}")
            _emit(args, payload, "\n".join(lines))
            return 0
        if action == "debate":
            # Standalone debate: the model (when available) plays both
            # sides; without a model we report honestly instead of
            # fabricating a transcript.
            router = getattr(context, "router", None)
            if router is None:
                _emit(args, {"verdict": "no_model", "work": args.work},
                      "swarm debate: no model available; nothing debated.")
                return 2
            from ...llm.base import Message, SamplingParams

            def coder_fn(brief, feedback, artifact):
                notes = ""
                if feedback:
                    notes = "\n".join(
                        f"- [{i.id}] {i.detail}" for i in feedback)
                resp = brain_for(context).chat(
                    [Message.system(
                        "You are a coder. Produce the work, then revise it "
                        "against the critic's issues, quoting each issue id "
                        "you fix. Reply with JSON: "
                        '{"content": "...", "summary": "...", '
                        '"addresses": ["ISSUE-1", ...]}'),
                     Message.user(f"Work: {brief}\n\nCritic issues:\n{notes or '(none)'}")],
                    SamplingParams(temperature=0.4, max_tokens=2048,
                                   json_mode=True), task_kind="chat")
                return _work_from_json(resp.text, brief)

            def critic_fn(artifact, rubric):
                resp = brain_for(context).chat(
                    [Message.system(
                        "You are an adversarial critic. Try to break the "
                        "work: counterexamples, edge cases, security holes. "
                        "Reply with JSON: {'verdict': "
                        "'approve|request_changes|reject', 'score': 0-100, "
                        "'issues': [{'id': 'ISSUE-n', 'severity': "
                        "'critical|major|minor', 'location': '...', "
                        "'detail': '...'}]}"),
                     Message.user(
                         f"Rubric: {', '.join(rubric)}\n\nWork:\n{artifact.content}")],
                    SamplingParams(temperature=0.3, max_tokens=2048,
                                   json_mode=True), task_kind="chat")
                return _critique_from_json(resp.text)

            debate = Debate(coder_fn=coder_fn, critic_fn=critic_fn,
                            max_rounds=args.rounds, context=context)
            result = debate.run(args.work)
            _emit(args, result.to_dict(),
                  f"swarm debate: {result.verdict} after {result.rounds} "
                  f"round(s) — {result.reason}")
            return 0 if result.approved else 1
        if action == "run":
            from ...agents.blackboard import Blackboard
            from ...agents.orchestrator import MasterOrchestrator, Plan, PlanStep

            roles = [r.strip() for r in args.roles.split(",") if r.strip()]
            registry = default_registry()
            board = Blackboard()
            angles = [f"angle {i + 1}: {r} perspective"
                      for i, r in enumerate(roles[:args.fanout])]
            fanout_res = fan_out(
                args.goal, angles or [f"research angle {i + 1}"
                                      for i in range(args.fanout)],
                role="researcher", blackboard=board, registry=registry,
                context=context)
            merged = fan_in(fanout_res.run_id, "concat_dedupe",
                            blackboard=board)
            steps = [PlanStep(name=f"research_{i}", goal=a, role="researcher")
                     for i, a in enumerate(fanout_res.angles)]
            steps.append(PlanStep(name="build", goal=args.goal, role="coder",
                                  depends_on=[s.name for s in steps]))
            steps.append(PlanStep(name="verify", goal=f"verify: {args.goal}",
                                  role="critic", depends_on=["build"]))
            plan = Plan(goal=args.goal, steps=steps,
                        rationale="swarm pipeline: fan-out research → "
                                  "coder builds → critic debates")
            orch = MasterOrchestrator(context, blackboard=board,
                                      roles=registry)
            result = orch.run(args.goal, plan=plan, reflect=False)
            payload = {"ok": result.ok, "answer": result.answer,
                       "research": merged.to_dict(),
                       "role_stats": orch.stats()["roles"]}
            _emit(args, payload,
                  f"swarm run: {'ok' if result.ok else 'failed'}\n"
                  f"{result.answer}")
            return 0 if result.ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f"swarm: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown swarm action: {action}", file=sys.stderr)
    return 2


def _work_from_json(text: str, brief: str) -> "WorkArtifact":
    from ...agents.debate import WorkArtifact

    import json as _json

    try:
        data = _json.loads(text)
    except Exception as e:
        _log.debug("could not parse work artifact JSON: %s", e)
        data = {}
    if not isinstance(data, dict):
        data = {}
    return WorkArtifact(content=data.get("content", text[:2000]),
                        summary=str(data.get("summary", ""))[:500],
                        addresses=list(data.get("addresses", []) or []))


def _critique_from_json(text: str) -> "Critique":
    from ...agents.debate import Critique, Issue

    import json as _json

    try:
        data = _json.loads(text)
    except Exception as e:
        _log.debug("could not parse critique JSON: %s", e)
        data = {}
    if not isinstance(data, dict):
        data = {}
    issues = []
    for i, raw in enumerate(data.get("issues", []) or []):
        if not isinstance(raw, dict):
            continue
        issues.append(Issue(
            id=str(raw.get("id", f"ISSUE-{i + 1}")),
            severity=str(raw.get("severity", "major")),
            location=str(raw.get("location", "")),
            detail=str(raw.get("detail", ""))[:1000]))
    return Critique(verdict=str(data.get("verdict", "request_changes")),
                    issues=issues,
                    score=float(data.get("score", 50) or 50))
