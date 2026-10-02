"""``nm project`` — project surfaces."""

from __future__ import annotations

import argparse
import sys
from typing import Any
from ..emit import _emit



def _cmd_project(args: argparse.Namespace, context: Any) -> int:
    """Projects CLI: create/list/status/heal/replan/run."""
    from ...agents.projects import ProjectManager

    pm = ProjectManager(context)
    action = getattr(args, "action", "list") or "list"
    pid = getattr(args, "id", "") or ""
    title = getattr(args, "title", "") or ""
    if action != "create" and not pid:
        pid = title  # positional reuse: `nm project heal <id>`
    if action == "create":
        objective = getattr(args, "description", "") or ""
        p = pm.create(title, objective=objective)
        _emit(args, p.to_dict(), f"created {p.id} — {p.title} [{p.status}]")
        return 0
    if action == "list":
        rows = pm.list_projects()
        lines = [f"{r.get('id', '')}  [{r.get('status', '')}]  "
                 f"{int(float(r.get('progress') or 0) * 100):3d}%  {r.get('title', '')}"
                 for r in rows]
        _emit(args, {"projects": rows},
              "\n".join(lines) if lines else "no projects")
        return 0
    if action == "heal":
        res = pm.heal(pid)
        n = int(res.get("healed_steps") or 0)
        text = (f"healed {n} step(s) on {pid}" if res.get("ok")
                else f"could not heal {pid}: {res.get('error', 'not healable')}")
        _emit(args, res, text)
        return 0
    if action == "replan":
        res = pm.replan(pid)
        _emit(args, res, f"replanned {pid}: {res.get('steps', '')}"
              if res.get("ok", True) else f"could not replan {pid}: "
              f"{res.get('error', '')}")
        return 0
    if action in ("status", "get"):
        st = pm.status(pid)
        if not st.get("ok", True):
            _emit(args, st, f"unknown project: {pid}")
            return 1
        human = (f"{pid} [{st.get('status', '?')}] "
                 f"{int(float(st.get('progress') or 0) * 100)}% — "
                 f"{st.get('title', '')}")
        _emit(args, st, human)
        return 0
    if action in ("plan", "run", "advance"):
        fn = getattr(pm, action if action != "advance" else "advance")
        res = fn(pid)
        payload = res.to_dict() if hasattr(res, "to_dict") else res
        _emit(args, payload, f"{action}: {pid}")
        return 0
    print(f"project: unknown action {action}", file=sys.stderr)
    return 2
