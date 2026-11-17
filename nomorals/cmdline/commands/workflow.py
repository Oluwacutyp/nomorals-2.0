"""``nm workflow`` — saved workflow scripts from the console.

The store half of chat /workflow (``agents/partner/
runtime_workflow.py: _control_workflow``): list/show/status/run/
cancel/resume over the same WorkflowStore at
``<home>/data/workflows.json``. The single-workflow renderings are
the chat mixin's own staticmethods (``_workflow_status_text`` /
``_workflow_show_text``), called directly — one rendering, never a
second copy. Drafting from ``<task>`` needs the brain and stays
chat-only: any tail that is not a store verb is refused in words,
pointing at chat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["_cmd_workflow"]


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _store(context: Any) -> Any:
    from ...agents.workflow.store import (
        WorkflowStore,
        default_workflows_dir,
        default_workflows_path,
    )

    home = _home_path(context)
    if home is None:
        raise ValueError("no home path on context")
    return WorkflowStore(path=default_workflows_path(home),
                         scripts_dir=default_workflows_dir(home))


def _resolve_workflow(store: Any, ref: str) -> tuple[Any | None, str]:
    """Exact name/id, else unique name/id prefix — as chat does.

    Kept local (not the mixin's ``_resolve_workflow``) so the miss
    text names ``nm workflow``, not ``/workflow``.
    """
    ref = (ref or "").strip()
    if not ref:
        return None, "name a workflow — nm workflow lists them."
    rows = store.list_workflows()
    for row in rows:
        if row.get("name") == ref or row.get("id") == ref:
            return row, ""
    prefix = [row for row in rows
              if str(row.get("name", "")).startswith(ref)
              or str(row.get("id", "")).startswith(ref)]
    if len(prefix) == 1:
        return prefix[0], ""
    if len(prefix) > 1:
        names = ", ".join(sorted(r["name"] for r in prefix))
        return None, (f"{ref!r} matches several workflows "
                      f"({names}) — name one exactly.")
    return None, f"no workflow {ref!r} — nm workflow lists them."


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _cmd_workflow(args: argparse.Namespace, context: Any) -> int:
    """``nm workflow [list|show <name>|status [name]|run <name>|cancel <name>|resume <name>]``."""
    from ...agents.partner.runtime_workflow import RuntimeWorkflowMixin
    from ...agents.workflow.errors import WorkflowError

    action = (getattr(args, "workflow_action", None) or "list").lower()
    rest = list(getattr(args, "workflow_rest", None) or [])
    as_json = bool(getattr(args, "json", False))
    try:
        store = _store(context)
    except Exception as exc:  # noqa: BLE001 - store path is the whole command
        return _usage_error(f"workflow unavailable: {exc}")
    if store.degraded_reason:
        return _usage_error(
            "Workflow store degraded: " + store.degraded_reason
            + " — saved workflows are unreadable right now.")

    if action == "list":
        rows = store.list_workflows()
        if as_json:
            print(json.dumps({"workflows": rows}, indent=2))
            return 0
        if not rows:
            print("no workflows yet — drafting needs the brain; "
                  "/workflow <task> from chat writes one.")
            return 0
        lines = ["workflows:"]
        for row in rows:
            lines.append(
                f"• {row['name']} [{row['state']}] — "
                f"{str(row.get('task', ''))[:60]}")
        print("\n".join(lines))
        return 0

    if action == "status":
        if not rest:
            rows = store.list_workflows()
            if as_json:
                print(json.dumps({"workflows": rows}, indent=2))
                return 0
            if not rows:
                print("no workflows yet — drafting needs the brain; "
                      "/workflow <task> from chat writes one.")
                return 0
            lines = ["workflow status:"]
            for row in rows:
                line = f"• {row['name']} [{row['state']}]"
                if row.get("error"):
                    line += f" — {row['error'][:100]}"
                lines.append(line)
            print("\n".join(lines))
            return 0
        row, error = _resolve_workflow(store, rest[0])
        if row is None:
            return _usage_error(error)
        if as_json:
            print(json.dumps(store.get_workflow(row["id"]), indent=2))
            return 0
        print(RuntimeWorkflowMixin._workflow_status_text(store, row))
        return 0

    if action in ("show", "run", "cancel", "resume"):
        if not rest:
            return _usage_error(
                f"usage: nm workflow {action} <name> — "
                "nm workflow lists them.")
        row, error = _resolve_workflow(store, rest[0])
        if row is None:
            return _usage_error(error)
        try:
            if action == "show":
                if as_json:
                    print(json.dumps(store.get_workflow(row["id"]), indent=2))
                    return 0
                print(RuntimeWorkflowMixin._workflow_show_text(store, row))
                return 0
            if action == "run":
                store.requeue(row["id"], by="owner")
                print(f"{row['name']} queued again — running in the "
                      "background; the summary is delivered to chat "
                      "when it's done.")
                return 0
            if action == "cancel":
                store.cancel(row["id"], by="owner")
                print(f"{row['name']} cancelled — the script stays saved; "
                      f"nm workflow run {row['name']} starts it again.")
                return 0
            store.resume(row["id"], by="owner")
            print(f"{row['name']} queued again — running in the "
                  "background; the summary is delivered to chat "
                  "when it's done.")
            return 0
        except WorkflowError as exc:
            return _usage_error(str(exc))

    return _usage_error(
        "drafting a workflow needs the brain — run /workflow <task> "
        "from chat; nm workflow only manages saved workflows "
        "(list · show <name> · status [name] · run <name> · "
        "cancel <name> · resume <name>).")
