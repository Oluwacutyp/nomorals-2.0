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
    if action == "integrity":
        # `nm project integrity <id> [mode]`: show the mode, or set it.
        # The mode rides the second positional (the description slot).
        from ...agents.auditor import AuditorError, get_integrity_mode, set_integrity_mode
        mode = (getattr(args, "description", "") or "").strip()
        if mode:
            try:
                set_integrity_mode(pm.db, pid, mode)
            except AuditorError as exc:
                print(f"project: {exc}", file=sys.stderr)
                return 1
            _emit(args, {"project_id": pid, "integrity_mode": mode},
                  f"{pid} integrity mode: {mode}")
            return 0
        cur = get_integrity_mode(pm.db, pid)
        _emit(args, {"project_id": pid, "integrity_mode": cur},
              f"{pid} integrity mode: {cur}")
        return 0
    if action == "audit":
        # `nm project audit <id>`: re-run the project's evidence now
        # under its integrity mode and print the verdict.
        from ...agents.auditor import audit_project_step
        p = pm._load(pid)
        if p is None:
            _emit(args, {"ok": False}, f"unknown project: {pid}")
            return 1
        verdict = audit_project_step(pm, p)
        if verdict is None:
            _emit(args, {"project_id": pid, "audited": False},
                  f"{pid}: nothing to audit")
            return 0
        text = (f"{pid}: audit "
                f"{'PASSED' if verdict.passed else 'FAILED'} "
                f"[{verdict.mode}] — "
                + ("; ".join(verdict.reasons)
                   or "evidence re-ran clean"))
        _emit(args, verdict.to_dict(), text)
        return 0 if verdict.passed else 1
    if action in ("plan", "run", "advance"):
        fn = getattr(pm, action if action != "advance" else "advance")
        res = fn(pid)
        payload = res.to_dict() if hasattr(res, "to_dict") else res
        _emit(args, payload, f"{action}: {pid}")
        return 0
    if action in ("update", "delete"):
        # The parser advertises these two, but ProjectManager has no
        # update/delete operation at all (no method, no row-deletion
        # path) — there is nothing in the organ to wire to. Answer
        # honestly (exit 1, the command-failed code) instead of the
        # old "unknown action" with the usage-error code 2, which
        # claimed a parser-valid verb did not exist.
        print(f"project: {action} is not supported — ProjectManager "
              f"has no {action} operation", file=sys.stderr)
        return 1
    print(f"project: unknown action {action}", file=sys.stderr)
    return 2
