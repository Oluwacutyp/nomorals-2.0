"""``nm vision`` — vision tool surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any



def _vision_tools(context: Any) -> Any:
    return context.tools.register_builtins()


def _vision_tool_call(context: Any, name: str, **kwargs: Any) -> Any:
    """Call a vision tool; return the value or raise a clean error."""
    outcome = _vision_tools(context).call(name, actor="cli", **kwargs)
    if not outcome.ok:
        err = outcome.error
        raise RuntimeError(getattr(err, "message", None) or str(err))
    return outcome.value


def _vision_source(file: str) -> dict[str, str]:
    if file.startswith(("http://", "https://")):
        return {"url": file}
    if re.match(r"^(inbox|room|attachment):", file):
        return {"reference": file}
    return {"path": file}


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:  # pragma: no cover - non-interactive stdin
        return False


def _print_vision_result(action: str, result: dict[str, Any]) -> None:
    if action == "read-text":
        print(result.get("text", ""))
        print(f"[{result.get('confidence_note', '')}]")
        return
    if action == "locate":
        if result.get("found"):
            print(f"found '{result.get('target')}': "
                  f"x={result['x']} y={result['y']} "
                  f"w={result['w']} h={result['h']} "
                  f"(confidence {result.get('confidence')})")
        else:
            print(f"not found: '{result.get('target')}'")
        print(result.get("disclaimer", ""))
        return
    # describe / screenshot
    if result.get("identity_note"):
        print(result["identity_note"])
        print()
    print(result.get("description", ""))
    model = result.get("model") or result.get("provider") or "?"
    print(f"[via {model}]")


def _cmd_vision(args: argparse.Namespace, context: Any) -> int:
    """Route `nm vision` to describe / read-text / locate / screenshot."""
    action = args.vision_action
    as_json = getattr(args, "json", False)
    try:
        if action == "describe":
            result = _vision_tool_call(
                context, "vision_describe",
                prompt=getattr(args, "question", "") or "",
                **_vision_source(args.file))
        elif action == "read-text":
            result = _vision_tool_call(context, "vision_read_text",
                                       **_vision_source(args.file))
        elif action == "locate":
            result = _vision_tool_call(context, "vision_locate",
                                       target=args.target,
                                       **_vision_source(args.file))
        elif action == "screenshot":
            # privileged: per-call user confirmation, on top of the tool's
            # own settings gate + confirm=True requirement
            if not _confirm("Capture the local display now? [y/N] "):
                print("screenshot cancelled — no capture taken", file=sys.stderr)
                return 2
            result = _vision_tool_call(
                context, "vision_screenshot", display=args.display,
                confirm=True, prompt=getattr(args, "prompt", "") or "")
        else:
            print(f"unknown vision action: {action}", file=sys.stderr)
            return 2
    except RuntimeError as exc:
        print(f"vision failed: {exc}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_vision_result(action, result)
    return 0
