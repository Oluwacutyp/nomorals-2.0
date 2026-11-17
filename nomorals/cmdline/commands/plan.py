"""``nm plan`` — saved Implementation Plans from the console.

The store half of chat /plan (``agents/partner/runtime_plan.py:
_control_plan``): list/show/status over the same PlanStore at
``<home>/data/plans.json``. Show reads the saved ``.md`` artifact
when it exists, else renders the plan from the record — the same
two-step ``_plan_show_text`` uses. Status renders the stored record;
there is no delegation manager on the console, so unlike chat it
does not refresh the state from live runs (named in the survey).
Drafting from ``<task>`` needs the brain and stays chat-only: any
tail that is not a store verb is refused in words, pointing at chat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["_cmd_plan"]


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _store(context: Any) -> Any:
    from ...agents.plan.store import (
        PlanStore,
        default_plans_dir,
        default_plans_path,
    )

    home = _home_path(context)
    if home is None:
        raise ValueError("no home path on context")
    return PlanStore(path=default_plans_path(home),
                     plans_dir=default_plans_dir(home))


def _resolve_plan(store: Any, ref: str) -> tuple[Any | None, str]:
    """Exact id, else unique id prefix — as chat does.

    Kept local (not the mixin's ``_resolve_plan``) so the miss text
    names ``nm plan``, not ``/plan``.
    """
    row = store.resolve(ref)
    if row is None:
        return None, f"no plan '{ref}' — nm plan lists them."
    return row, ""


def _status_text(row: dict[str, Any]) -> str:
    """One plan's state — the stored record, as ``_plan_status_text``
    renders it without a live-run refresh."""
    lines = [f"plan {row['id']} [{row['state']}]",
             f"task: {row.get('task', '')}"]
    if row.get("plan_error"):
        lines.append(f"note: {row['plan_error']}")
    for run in row.get("runs") or []:
        if isinstance(run, dict):
            lines.append(f"• {run.get('id')} ({run.get('role')}) — "
                         f"{run.get('status', 'unknown')}")
    for entry in row.get("unassigned") or []:
        if isinstance(entry, dict):
            lines.append(f"• unassigned {entry.get('step')}: "
                         f"{entry.get('reason', 'no roster worker')}")
    return "\n".join(lines)


def _show_text(store: Any, row: dict[str, Any]) -> str:
    """The full Implementation Plan artifact — ``_plan_show_text``."""
    if store.plans_dir is not None:
        artifact = store.plans_dir / f"{row['id']}.md"
        try:
            return artifact.read_text(encoding="utf-8")
        except OSError:
            pass
    from ...agents.plan.render import render_plan_md

    return render_plan_md(row)


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _cmd_plan(args: argparse.Namespace, context: Any) -> int:
    """``nm plan [list|show <id>|status [id]]``."""
    from ...agents.plan import PlanError

    action = (getattr(args, "plan_action", None) or "list").lower()
    rest = list(getattr(args, "plan_rest", None) or [])
    as_json = bool(getattr(args, "json", False))
    try:
        store = _store(context)
    except Exception as exc:  # noqa: BLE001 - store path is the whole command
        return _usage_error(f"plan unavailable: {exc}")
    if store.degraded_reason:
        return _usage_error(
            "Plan store degraded: " + store.degraded_reason
            + " — saved plans are unreadable right now.")

    if action == "list":
        rows = store.list_plans()
        if as_json:
            print(json.dumps({"plans": rows}, indent=2))
            return 0
        if not rows:
            print("no plans yet — drafting needs the brain; "
                  "/plan <task> from chat writes one.")
            return 0
        lines = ["plans:"]
        for row in rows:
            lines.append(
                f"• {row['id']} [{row['state']}] — "
                f"{str(row.get('task', ''))[:60]}")
        print("\n".join(lines))
        return 0

    if action == "status":
        if not rest:
            rows = store.list_plans()
            if as_json:
                print(json.dumps({"plans": rows}, indent=2))
                return 0
            if not rows:
                print("no plans yet — drafting needs the brain; "
                      "/plan <task> from chat writes one.")
                return 0
            lines = ["plan status:"]
            for row in rows:
                line = f"• {row['id']} [{row['state']}]"
                if row.get("plan_error"):
                    line += f" — {row['plan_error'][:100]}"
                lines.append(line)
            print("\n".join(lines))
            return 0
        row, error = _resolve_plan(store, rest[0])
        if row is None:
            return _usage_error(error)
        if as_json:
            print(json.dumps(row, indent=2))
            return 0
        print(_status_text(row))
        return 0

    if action == "show":
        if not rest:
            return _usage_error(
                "usage: nm plan show <id> — nm plan lists them.")
        row, error = _resolve_plan(store, rest[0])
        if row is None:
            return _usage_error(error)
        try:
            text = _show_text(store, row)
        except PlanError as exc:
            return _usage_error(str(exc))
        if as_json:
            print(json.dumps(row, indent=2))
            return 0
        print(text)
        return 0

    return _usage_error(
        "drafting a plan needs the brain — run /plan <task> from "
        "chat; nm plan only manages saved plans "
        "(list · show <id> · status [id]).")
