"""Render a saved plan as a readable Implementation Plan.

Two shapes, one source of truth (the saved record):
:func:`render_plan_md` is the durable artifact stored under
``<home>/plans/<id>.md``; :func:`plan_display` is the short chat
render shown when the plan is created and listed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["render_plan_md", "plan_display"]

_MAX_DISPLAY_STEPS = 8


def _steps(plan: dict[str, Any]) -> list[dict[str, Any]]:
    steps = plan.get("steps") or []
    return [s for s in steps if isinstance(s, dict)]


def _is_gate(step: dict[str, Any]) -> bool:
    if str(step.get("role") or "").lower() == "critic":
        return True
    payload = step.get("payload") or {}
    return bool(isinstance(payload, dict) and payload.get("debate"))


def _step_line(step: dict[str, Any]) -> str:
    name = step.get("name") or step.get("role") or "step"
    goal = step.get("goal") or ""
    role = step.get("role") or ""
    writes = step.get("writes") or []
    line = f"- {name} ({role})" if role else f"- {name}"
    if goal:
        line += f" — {goal}"
    if writes:
        line += f" [writes: {', '.join(str(w) for w in writes)}]"
    return line


def render_plan_md(record: dict[str, Any]) -> str:
    """The full Implementation Plan artifact for a saved record."""
    plan = record.get("plan") or {}
    lines = [f"# Implementation Plan — {record.get('id', '')}", ""]
    lines.append(f"Task: {record.get('task', '')}")
    lines.append("")
    survey = record.get("survey") or {}
    files = survey.get("files") or []
    if files:
        lines.append("## Surveyed code")
        lines.append("")
        for entry in files:
            lines.append(f"- {entry.get('path')} "
                         f"({entry.get('hits', 0)} hits)")
        lines.append("")
    elif survey:
        lines.append("## Surveyed code")
        lines.append("")
        lines.append("No owning files found — treated as greenfield.")
        lines.append("")
    lines.append("## Approach")
    lines.append("")
    if plan.get("error"):
        lines.append(f"Planner fell back to a template: {plan['error']}")
        lines.append("")
    if plan.get("rationale"):
        lines.append(str(plan["rationale"]))
        lines.append("")
    steps = _steps(plan)
    for step in steps:
        lines.append(_step_line(step))
    if not steps:
        lines.append("No steps — the planner returned an empty plan.")
    lines.append("")
    gates = [s for s in steps if _is_gate(s)]
    lines.append("## Verification gates")
    lines.append("")
    if gates:
        for step in gates:
            lines.append(_step_line(step))
    else:
        lines.append("No critic or debate step — the plan has no "
                     "verification gate.")
    lines.append("")
    lines.append("## Risks")
    lines.append("")
    risks: list[str] = []
    if record.get("plan_error"):
        risks.append(f"template plan: {record['plan_error']}")
    for entry in record.get("unassigned") or []:
        if isinstance(entry, dict):
            role = entry.get("role")
            role_text = f" ({role})" if role else ""
            risks.append(
                f"unassigned step {entry.get('step')}{role_text}: "
                f"{entry.get('reason', 'no roster worker')}")
    if risks:
        lines.extend(f"- {risk}" for risk in risks)
    else:
        lines.append("None recorded.")
    lines.append("")
    runs = record.get("runs") or []
    lines.append("## Runs")
    lines.append("")
    if runs:
        for run in runs:
            if isinstance(run, dict):
                lines.append(
                    f"- {run.get('id')} ({run.get('role')}) — "
                    f"{run.get('status', 'unknown')}")
    else:
        lines.append("No runs spawned.")
    lines.append("")
    return "\n".join(lines)


def plan_display(record: dict[str, Any]) -> str:
    """The short chat render of a saved plan."""
    plan = record.get("plan") or {}
    steps = _steps(plan)
    lines = [f"Plan {record.get('id', '')} — {record.get('task', '')}", ""]
    if record.get("plan_error"):
        lines.append(f"Note: template plan — {record['plan_error']}")
        lines.append("")
    shown = steps[:_MAX_DISPLAY_STEPS]
    for step in shown:
        lines.append(_step_line(step))
    if len(steps) > _MAX_DISPLAY_STEPS:
        lines.append(f"… and {len(steps) - _MAX_DISPLAY_STEPS} more steps")
    if not steps:
        lines.append("No steps — the planner returned an empty plan.")
    gates = [s for s in steps if _is_gate(s)]
    if gates:
        lines.append("")
        lines.append("Verification gates: " + ", ".join(
            str(s.get("name") or s.get("role") or "step") for s in gates))
    unassigned = record.get("unassigned") or []
    if unassigned:
        lines.append("")
        lines.append("Unassigned (nothing ran for these):")
        for entry in unassigned:
            if isinstance(entry, dict):
                role = entry.get("role")
                role_text = f" ({role})" if role else ""
                lines.append(f"- {entry.get('step')}{role_text}: "
                             f"{entry.get('reason', 'no roster worker')}")
    lines.append("")
    lines.append(f"State: {record.get('state', 'running')} — "
                 f"runs are live; the plan runs now, no approval wait.")
    return "\n".join(lines)
