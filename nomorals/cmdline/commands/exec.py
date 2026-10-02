"""``nm exec`` / ``nm apps`` — execution surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from ..emit import _emit



def _cmd_exec(args: argparse.Namespace, context: Any) -> int:
    """Run code through the sandbox (exec tool)."""
    tools = context.tools
    code = getattr(args, "code", "") or ""
    if code.strip().lower() in {"languages", "langs"}:
        from ...execbox import CodeRunner

        langs = CodeRunner(context).languages()
        _emit(args, {k: v for k, v in langs.items()},
              "\n".join(f"  {k:<12} {'available' if getattr(v, 'available', True) else 'not installed'}"
                      for k, v in sorted(langs.items())))
        return 0
    if not code and not getattr(args, "file", ""):
        print("exec needs code — nm exec \"print(6*7)\" --lang python",
              file=sys.stderr)
        return 2
    out = tools.call("run_code", code=code,
                     language=getattr(args, "lang", "") or "",
                     timeout=float(getattr(args, "timeout", "30") or 30),
                     file=getattr(args, "file", "") or "")
    if not out.ok:
        print(f"exec: {out.error}", file=sys.stderr)
        return 1
    value = out.value
    _emit(args, value,
          (value.get("stdout") or "")
          + ("" if not value.get("stderr") else f"\n[stderr]\n{value['stderr']}"))
    return 0 if int(value.get("exit_code") or 0) == 0 else 1


def _cmd_apps(args: argparse.Namespace, context: Any) -> int:
    """Build / list / serve local apps through the build_app tool."""
    tools = context.tools
    action = getattr(args, "action", "list") or "list"
    name = getattr(args, "name", "") or ""
    kwargs: dict[str, Any] = {"action": action, "stack": getattr(args, "stack", "") or "static",
                              "features": getattr(args, "features", "") or "",
                              "title": getattr(args, "title", "") or ""}
    if name:
        kwargs["name"] = name
    if getattr(args, "port", ""):
        kwargs["port"] = int(args.port)
    out = tools.call("build_app", **kwargs)
    if not out.ok:
        print(f"apps: {out.error}", file=sys.stderr)
        return 1
    value = out.value
    if action == "build":
        v = value.get("validation") or {}
        _emit(args, value,
              f"built {name} [{kwargs['stack']}] — "
              f"{len(value.get('files') or [])} files, "
              f"validation: {'OK' if v.get('ok') else 'FAILED'}"
              + ("" if v.get("ok") else f" — {v.get('output', '')[:300]}"))
        return 0 if v.get("ok", True) else 1
    if action == "list":
        lines = [f"  {a.get('name')} [{a.get('stack')}] {a.get('path', '')}"
                 for a in (value.get("apps") or [])]
        _emit(args, value,
              "\n".join(lines) if lines else "no apps built yet — nm apps build <name> --stack flask")
        return 0
    _emit(args, value, json.dumps(value, indent=2, default=str))
    return 0
