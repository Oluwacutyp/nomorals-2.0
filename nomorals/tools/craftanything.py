"""Craft-Anything: ToolSpec with effects + consent gates, as a tool.

A thin surface over :mod:`nomorals.agents.craftanything` — the organ
owns the Effects bitflags, the consent gate + journal, the toolsets,
the single Engine.call dispatch and the agent that plans multi-step
tasks; the tool only routes arguments and shapes the reply. The engine
binds the live builtin registry (built lazily once per process), so a
call through this surface exercises the same real organ code as a
direct call — no parallel registry.

Writes journal/evidence files, so FS_WRITE. Its own declared effects
are MUTATES_SESSION | WRITES_FILESYSTEM (it journals session state and
retains over-limit results); the effects of a dispatched call are the
dispatched tool's own, checked by the engine before it runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..agents.craftanything import effects as _effects
from ..agents.craftanything.agent import CraftAgent
from ..agents.craftanything.attribution import verify_attribution
from ..agents.craftanything.caller import Caller
from ..agents.craftanything.consent import Consent
from ..agents.craftanything.engine import CallOptions, CraftEngine
from ..agents.craftanything.parity import run_parity
from ..core.policy import Capability

__all__ = ["register"]

_ACTIONS = ("list_tools", "inspect", "call", "enable_toolset",
            "disable_toolset", "list_toolsets", "consent", "journal",
            "plan", "run_task", "parity", "attribution")

_ENGINE: CraftEngine | None = None


def _engine() -> CraftEngine:
    """The engine over the live builtin registry (one per process)."""
    global _ENGINE
    if _ENGINE is None:
        from .registry import ToolRegistry
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        _ENGINE = CraftEngine(registry)
    return _ENGINE


def _caller(name: str) -> Caller:
    try:
        return Caller(str(name or "ui").lower())
    except ValueError:
        raise ValueError(
            f"unknown caller {name!r}; use one of "
            f"{', '.join(c.value for c in Caller)}") from None


def _consent(name: str) -> Consent:
    value = str(name or "").lower()
    if value in ("", "not_asked"):
        return Consent.NOT_ASKED
    if value == "granted":
        return Consent.GRANTED
    raise ValueError(
        f"unknown consent {name!r}; use 'granted' or leave it empty")


def register(registry: Any) -> None:
    """Attach the craftanything family to a registry."""
    @registry.register(
        "craftanything",
        description=(
            "ToolSpec with effects + consent gates generalising the "
            "command registry — list tools by toolset, inspect a "
            "tool's declared effects and hints, call any tool through "
            "the single effect-checked, consent-gated dispatch, "
            "enable/disable toolsets, grant or deny standing consent "
            "rules, read the call + consent journals, plan a "
            "multi-step task and run it through the engine. "
            "Craft-Anything actions: list_tools=<list tool summaries, "
            "toolsets=[ids]>; inspect=<a tool's spec, tool=<name>>; "
            "call=<run a tool through the engine, tool=<name>, "
            "params=<dict>, caller=<ui|cli|mcp|agent>, "
            "consent=<granted>>; enable_toolset=<enable a toolset, "
            "toolset=<id>>; disable_toolset=<disable a toolset, "
            "toolset=<id>>; list_toolsets=<every toolset with counts>; "
            "consent=<standing rules, decision=<allow|deny|list>, "
            "tool=<name>, effects=[names]>; journal=<call + consent "
            "journals, limit=<n>>; plan=<decompose a goal, "
            "goal=<text>>; run_task=<plan and run a goal, goal=<text>, "
            "consent=<granted>>; parity=<run the machine-checked "
            "parity TOML>; attribution=<verify the SHA-256 attribution "
            "DB>. action=<list_tools|inspect|call|enable_toolset|"
            "disable_toolset|list_toolsets|consent|journal|plan|"
            "run_task|parity|attribution>; tool=<tool name>; "
            "params=<dict of tool parameters>."
        ),
        capability=Capability.FS_WRITE,
        effects=_effects.MUTATES_SESSION | _effects.WRITES_FILESYSTEM,
    )
    def craftanything_tool(action: str = "list_tools",
                           tool: str = "",
                           params: dict | None = None,
                           caller: str = "ui",
                           consent: str = "",
                           toolsets: list[str] | None = None,
                           toolset: str = "",
                           goal: str = "",
                           decision: str = "",
                           effects: list[str] | None = None,
                           limit: int = 20,
                           ) -> dict[str, Any]:
        action = (action or "list_tools").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown craftanything action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        engine = _engine()
        if action == "list_tools":
            tools = engine.list_tools(
                toolsets=set(toolsets) if toolsets else None)
            return {"tools": tools, "count": len(tools),
                    "reply": f"{len(tools)} tools listed."}
        if action == "inspect":
            if not tool:
                raise ValueError("craftanything inspect needs tool=<name>")
            info = engine.inspect(tool)
            if info is None:
                raise ValueError(f"No tool named {tool}")
            info["reply"] = (
                f"{tool}: {', '.join(info['effects']) or 'no effects'} "
                f"(toolset {info['toolset']}).")
            return info
        if action == "call":
            if not tool:
                raise ValueError("craftanything call needs tool=<name>")
            result = engine.call(
                tool, params or {},
                CallOptions(caller=_caller(caller),
                            consent=_consent(consent)))
            result["reply"] = (
                f"{tool}: {'ok' if result.get('ok') else 'refused'}"
                + (f" — {result.get('error', '')}"
                   if not result.get("ok") else "."))
            return result
        if action == "enable_toolset":
            if not toolset:
                raise ValueError(
                    "craftanything enable_toolset needs toolset=<id>")
            return engine.enable_toolset(toolset)
        if action == "disable_toolset":
            if not toolset:
                raise ValueError(
                    "craftanything disable_toolset needs toolset=<id>")
            return engine.disable_toolset(toolset)
        if action == "list_toolsets":
            sets = engine.toolsets()
            return {"toolsets": sets,
                    "reply": f"{len(sets)} toolsets in use."}
        if action == "consent":
            decision = (decision or "list").lower()
            if decision == "list":
                return {"standing_rules": engine.gate.standing_rules(),
                        "journal": engine.gate.journal[-limit:],
                        "reply": "standing consent rules listed."}
            if not tool:
                raise ValueError(
                    "craftanything consent needs tool=<name>")
            mask = (_effects.effects_from_names(effects)
                    if effects else None)
            if decision == "allow":
                engine.gate.allow_tool(tool, effects=mask)
            elif decision == "deny":
                engine.gate.deny_tool(tool, effects=mask)
            else:
                raise ValueError(
                    f"unknown consent decision {decision!r}; use "
                    f"allow, deny or list")
            return {"standing_rules": engine.gate.standing_rules(),
                    "reply": f"standing rule recorded: {decision} {tool}."}
        if action == "journal":
            return {"calls": engine.journal[-limit:],
                    "consent": engine.gate.journal[-limit:],
                    "reply": (f"{min(limit, len(engine.journal))} call "
                              f"entries, "
                              f"{min(limit, len(engine.gate.journal))} "
                              f"consent entries.")}
        if action == "plan":
            if not goal:
                raise ValueError("craftanything plan needs goal=<text>")
            agent = CraftAgent(engine)
            plan = agent.plan(goal)
            return {"goal": plan.goal,
                    "steps": [{"tool": s.tool, "params": s.params,
                               "rationale": s.rationale,
                               "effects": s.effects}
                              for s in plan.steps],
                    "warnings": plan.warnings,
                    "reply": (f"plan: {len(plan.steps)} steps"
                              + (f" ({'; '.join(plan.warnings)})"
                                 if plan.warnings else "."))}
        if action == "run_task":
            if not goal:
                raise ValueError("craftanything run_task needs goal=<text>")
            agent = CraftAgent(engine)
            plan = agent.plan(goal)
            report = agent.run(
                plan, CallOptions(caller=Caller.AGENT,
                                  consent=_consent(consent)))
            report["steps"] = [{"tool": s.tool, "params": s.params,
                                "rationale": s.rationale}
                               for s in plan.steps]
            report["reply"] = (
                f"task: {report['ran']}/{report['total']} steps ran"
                + (f", {len(report['refused'])} refused"
                   if report["refused"] else "")
                + (f", {len(report['failed'])} failed"
                   if report["failed"] else "")
                + ".")
            return report
        if action == "parity":
            result = run_parity()
            result["reply"] = (
                f"craftanything parity: {result['passed']} passed, "
                f"{len(result['failed'])} failed"
                + (f" ({'; '.join(result['failed'])})"
                   if result["failed"] else "") + ".")
            return result
        if action == "attribution":
            result = verify_attribution()
            result["reply"] = (
                f"craftanything attribution: {result['checked']} verified, "
                f"{len(result['failed'])} failed, "
                f"{len(result['skipped'])} skipped, "
                f"{result['noted']} noted.")
            return result
        raise ValueError(f"unhandled craftanything action {action!r}")
