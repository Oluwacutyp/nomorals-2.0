"""The agent: plan a multi-step task, run it through the engine.

Port of the upstream agent loop (``crates/ai/src/agent.rs``): a model,
the engine's tools, consent, memory and the journal. Upstream's loop is
model-driven — the model picks tools turn by turn, with a turn budget
(``DEFAULT_MAX_TURNS`` / ``BUILDER_MAX_TURNS``) and a result budget
(``RESULT_BUDGET``). Devon has no model in this organ, so this agent is
deterministic: it plans from the goal text against the live tool
catalogue (tool names, action names, descriptions), orders the steps
by the goal's own order with producer-before-consumer for paths, and
runs each step through :class:`engine.CraftEngine` with the same
consent gate, verifying each step before the next. A goal that matches
no tool is an honest empty plan, never a fabricated one.

The planner's job is decomposition, not judgement: it finds the tools
the goal names, extracts the paths the goal carries, and threads the
document through the steps (an import step's output is the next step's
source). Anything it cannot match is a warning on the plan, and the
run stops at the first refusal with the reason named.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import effects as _effects
from .caller import Caller
from .engine import CallOptions, CraftEngine
from .specs import EFFECTS_TABLE

__all__ = ["PlanStep", "Plan", "CraftAgent"]

#: How many steps one plan may hold (upstream's turn budget, deterministic).
MAX_STEPS = 12
#: Document suffix each media tool writes, so the planner can thread the
#: document from an import/create step into the steps that follow it.
_DOC_SUFFIX = {
    "cadcraft": ".cadcraft",
    "materialcraft": ".mcraft",
    "texturecraft": ".tcraft",
    "anicraft": ".acraft",
    "figcraft": ".figcraft",
    "soundcraft": ".scraft",
    "designcraft": ".layout",
    "vectorcraft": ".vec",
}
#: File suffixes the planner recognises as paths in a goal.
_PATH_SUFFIXES = (
    ".dxf", ".dwg", ".cadcraft", ".mcraft", ".tcraft", ".acraft",
    ".figcraft", ".scraft", ".layout", ".vec", ".png", ".jpg", ".gif",
    ".mp4", ".webp", ".apng", ".svg", ".xfl", ".sbs", ".wav",
)


@dataclass
class PlanStep:
    """One step of a plan: a tool call with its rationale."""

    tool: str
    params: dict[str, Any]
    rationale: str
    effects: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """A decomposed goal: steps in run order, plus warnings."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CraftAgent:
    """Plans and runs multi-step tasks through a :class:`CraftEngine`.

    ``enabled_toolsets`` limits the catalogue the planner may use
    (upstream: agents see Core plus whatever toolsets they enabled).
    ``None`` means everything.
    """

    def __init__(self, engine: CraftEngine, *,
                 enabled_toolsets: set[str] | None = None) -> None:
        self.engine = engine
        self.enabled_toolsets = enabled_toolsets

    # ── planning ──────────────────────────────────────────────────────
    def _catalogue(self) -> list[dict[str, Any]]:
        tools = self.engine.list_tools()
        if self.enabled_toolsets is None:
            return tools
        wanted = set(self.enabled_toolsets) | {"core"}
        return [t for t in tools if t["toolset"] in wanted]

    @staticmethod
    def _paths_in(goal: str) -> list[str]:
        """Path-like tokens in the goal, in order, de-duplicated."""
        found: list[str] = []
        for token in re.split(r"[\s,;]+", goal):
            token = token.strip("\"'()")
            if not token:
                continue
            lowered = token.lower()
            if (any(lowered.endswith(suffix) for suffix in _PATH_SUFFIXES)
                    and token not in found):
                found.append(token)
        return found

    def plan(self, goal: str) -> Plan:
        """Decompose a goal into tool calls, in run order."""
        plan = Plan(goal=goal)
        text = goal.lower()
        paths = self._paths_in(goal)
        # A path token is not a tool mention: a directory named after a
        # tool ("/tmp/craftanything-proof/bracket.png") and a fixture
        # path (".../cadcraft/sample-bracket.dxf") must not invent
        # steps for tools the goal never named.
        for token in paths:
            text = text.replace(token.lower(), " ")
        catalogue = {t["name"]: t for t in self._catalogue()}
        # Actions the goal names, in the order it names them: an action
        # word next to (or anywhere with) a tool name is a step for
        # that tool. A bare tool name with no action is an inspect step.
        mentioned_tools = [name for name in catalogue
                           if re.search(rf"\b{re.escape(name)}\b", text)]
        if not mentioned_tools:
            plan.warnings.append(
                "no matching tool in the catalogue for this goal; "
                "nothing was planned")
            return plan
        for tool in mentioned_tools:
            actions = self._actions_for(tool)
            hits = [(m.start(), action) for action in actions
                    for m in re.finditer(
                        rf"\b{re.escape(action)}\b", text)]
            if not hits:
                plan.steps.append(PlanStep(
                    tool=tool, params={"action": "inspect"},
                    rationale=f"look at the {tool} document first",
                    effects=["idempotent"]))
                continue
            for _, action in sorted(hits):
                plan.steps.append(PlanStep(
                    tool=tool, params={"action": action},
                    rationale=f"{tool} {action} (named in the goal)",
                    effects=_effects.effect_names(
                        self._action_effects(tool, action))))
        self._fill_params(plan, paths)
        if len(plan.steps) > MAX_STEPS:
            plan.warnings.append(
                f"plan capped at {MAX_STEPS} steps")
            plan.steps = plan.steps[:MAX_STEPS]
        return plan

    @staticmethod
    def _actions_for(tool: str) -> list[str]:
        """The actions a tool offers, from the effects table if present."""
        entry = EFFECTS_TABLE.get(tool)
        if entry is not None:
            return sorted(entry["actions"])
        return []

    @staticmethod
    def _action_effects(tool: str, action: str) -> int:
        from .specs import action_effects
        mask = action_effects(tool, action)
        return mask if mask is not None else 0

    def _fill_params(self, plan: Plan, paths: list[str]) -> None:
        """Fill sources/output on each step and thread the document.

        Rules, in order: a step's ``sources`` is the first input path
        (a non-output suffix) unless an earlier step of the same tool
        already produced a document — then the produced document is the
        source. A step that writes a document gets an ``output`` next
        to the plan's final artifact (same directory, the tool's
        document suffix). A step that writes a rendered artifact gets
        the first output-suffixed path in the goal.
        """
        inputs = [p for p in paths
                  if not p.lower().endswith(
                      (".png", ".jpg", ".gif", ".mp4", ".webp", ".apng",
                       ".svg"))]
        outputs = [p for p in paths if p not in inputs]
        produced: dict[str, str] = {}
        final_dir = (str(Path(outputs[0]).parent) if outputs else None)
        for step in plan.steps:
            tool = step.tool
            action = step.params.get("action", "")
            if tool in produced and action not in ("import", "create", "open"):
                step.params["sources"] = [produced[tool]]
            elif inputs:
                step.params.setdefault("sources", [inputs[0]])
            if action in ("import", "create", "save"):
                suffix = _DOC_SUFFIX.get(tool, "")
                # The produced document sits next to the plan's
                # final artifact and shares its stem (bracket.png
                # -> bracket.cadcraft); with no final artifact it
                # keeps the input document's stem instead.
                stem = (Path(outputs[0]).stem if outputs else
                        Path(inputs[0]).stem if inputs else "")
                if suffix and final_dir and stem:
                    out = str(Path(final_dir) / (stem + suffix))
                    step.params["output"] = out
                    produced[tool] = out
            elif (action in ("solve", "cook", "retime")
                    and tool in produced):
                step.params["output"] = produced[tool]
            elif (action in ("render", "render_frame", "export", "publish")
                    and outputs):
                step.params["output"] = outputs[0]
                step.params.setdefault("sources",
                                       [produced.get(tool, inputs[0])]
                                       if inputs else [])
                if tool in produced:
                    step.params["sources"] = [produced[tool]]

    # ── running ───────────────────────────────────────────────────────
    def run(self, plan: Plan,
            opts: CallOptions | None = None) -> dict[str, Any]:
        """Run a plan step by step through the engine, verifying each.

        Stops at the first refusal or failure with the reason named;
        the report says what ran, what was refused, what it cost in
        wall time, and which outputs exist. A step that produces a file
        is verified by the file existing — a claimed write with no file
        is a failure, not a success.
        """
        opts = opts or CallOptions(caller=Caller.AGENT)
        started = time.perf_counter()
        ran = 0
        refused: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for step in plan.steps:
            result = self.engine.call(
                step.tool, dict(step.params), opts)
            if not result.get("ok"):
                # A consent refusal names the effect in the consent
                # decision's reason; report that, not the generic
                # message, so the refusal says what it needs.
                if result.get("error_code") == "consent_required":
                    reason = (result.get("consent") or {}).get(
                        "reason") or result.get("error", "call failed")
                else:
                    reason = result.get("error", "call failed")
                entry = {"tool": step.tool,
                         "action": step.params.get("action", ""),
                         "reason": reason,
                         "error_code": result.get("error_code", "")}
                if result.get("error_code") == "consent_required":
                    refused.append(entry)
                else:
                    failed.append(entry)
                break
            output = step.params.get("output")
            if (isinstance(output, str) and output
                    and not Path(output).is_file()):
                failed.append({
                    "tool": step.tool,
                    "action": step.params.get("action", ""),
                    "reason": f"claimed write with no file: {output}",
                    "error_code": "missing_output"})
                break
            ran += 1
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        ok = not refused and not failed and ran == len(plan.steps)
        return {
            "ok": ok,
            "goal": plan.goal,
            "ran": ran,
            "total": len(plan.steps),
            "refused": refused,
            "failed": failed,
            "duration_ms": duration_ms,
            "warnings": list(plan.warnings),
        }
