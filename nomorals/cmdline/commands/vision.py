"""``nm vision`` — vision tool surfaces."""

from __future__ import annotations

import argparse
import json
import re
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
        print(f"[{result.get('method', '?')} — {result.get('confidence_note', '')}]")
        return
    if action == "locate":
        if result.get("found"):
            score = result.get("score", result.get("confidence", "?"))
            print(f"found '{result.get('target')}': "
                  f"x={result['x']} y={result['y']} "
                  f"w={result['w']} h={result['h']} "
                  f"(score {score}, {result.get('method', '?')})")
        else:
            print(f"not found: '{result.get('target')}' "
                  f"({result.get('method', '?')})")
        print(result.get("disclaimer", result.get("note", "")))
        return
    if action == "compare":
        native = result.get("native") or {}
        if native.get("identical"):
            print("native diff: pixel-identical")
        elif native:
            frac = (native.get("changed_fraction") or 0.0) * 100
            print(f"native diff: {frac:.1f}% of pixels changed "
                  f"(mean abs diff {native.get('mean_abs_diff')}/255)")
            bbox = native.get("changed_bbox_1000")
            if bbox:
                print(f"changed region: x={bbox['x']} y={bbox['y']} "
                      f"w={bbox['w']} h={bbox['h']} (0-1000)")
        desc = (result.get("description") or "").strip()
        if desc and result.get("method") != "native":
            print()
            print(desc)
        return
    if action == "analyze":
        _print_analyze(result)
        return
    if action == "layout":
        lines = result.get("lines") or []
        print(f"blocks={result.get('blocks')} lines={len(lines)} "
              f"words={result.get('words')} "
              f"(mean conf {result.get('mean_word_conf')})")
        for ln in lines[:20]:
            text = (ln.get("text") or "").strip()
            if text:
                print(f"  • {text} (conf {ln.get('conf')})")
        if len(lines) > 20:
            print(f"  … +{len(lines) - 20} more lines")
        return
    if action == "metadata":
        r = result
        print(f"{r.get('format', '?')} {r.get('width', '?')}x{r.get('height', '?')} "
              f"· {r.get('bytes', '?')} bytes · sha256 {str(r.get('sha256', ''))[:16]}…")
        return
    if action == "info":
        _print_capabilities(result)
        return
    if action == "extract":
        print(result.get("chat_summary", result.get("summary", "")))
        return
    # describe / screenshot
    if result.get("identity_note"):
        print(result["identity_note"])
        print()
    print(result.get("description", ""))
    model = result.get("model") or result.get("provider") or "?"
    print(f"[via {model}]")


def _print_analyze(result: dict[str, Any]) -> None:
    meta = result.get("metadata") or {}
    print(f"{meta.get('format', '?')} {meta.get('width', '?')}x{meta.get('height', '?')} "
          f"· {meta.get('bytes', '?')} bytes")
    exif = result.get("exif") or {}
    if exif.get("present"):
        cam = f"{exif.get('Make', '')} {exif.get('Model', '')}".strip()
        print(f"EXIF: {cam or 'present'}"
              + (f" · {exif.get('DateTimeOriginal', '')}"
                 if exif.get("DateTimeOriginal") else ""))
        gps = exif.get("GPS") or {}
        if isinstance(gps, dict) and gps.get("latitude") is not None:
            print(f"GPS: {gps['latitude']:.5f}, {gps['longitude']:.5f}")
    else:
        print("EXIF: none")
    colors = result.get("colors") or {}
    if colors.get("available", True):
        dom = " ".join(
            f"{c['hex']}({c['share'] * 100:.0f}%)"
            for c in (colors.get("dominant") or [])[:5])
        print(f"colors: {dom}")
        print(f"brightness {colors.get('brightness')}/255 · "
              f"contrast {colors.get('contrast')} · "
              f"saturation {colors.get('saturation_pct')}%")
    quality = result.get("quality") or {}
    if quality.get("available", True):
        print(f"sharpness {quality.get('sharpness')} "
              f"({quality.get('sharpness_label')}) · "
              f"entropy {quality.get('entropy_bits')} bits")
    hashes = result.get("hashes") or {}
    if hashes.get("available", True):
        print(f"dhash {hashes.get('dhash')}  ahash {hashes.get('ahash')}")
    faces = result.get("faces") or {}
    if faces.get("available", True):
        print(f"faces: {faces.get('count', 0)}")
    else:
        print(f"faces: unavailable ({faces.get('why', '?')})")
    qr = result.get("qr") or {}
    if qr.get("available", True):
        codes = qr.get("codes") or []
        print(f"QR/barcodes: {len(codes)}")
        for c in codes:
            print(f"  [{c.get('type')}] {c.get('data', '')[:120]}")
    else:
        print(f"QR: unavailable ({qr.get('why', '?')})")


def _print_capabilities(result: dict[str, Any]) -> None:
    print(f"vision capabilities (profile: {result.get('profile', '?')}):")
    print("native — offline, no key:")
    for name, info in (result.get("native") or {}).items():
        if info.get("available"):
            print(f"  ok   {name}: {info.get('what', '')}")
        else:
            hint = f" — {info.get('install')}" if info.get("install") else ""
            print(f"  miss {name}: {info.get('why', '')}{hint}")
    print("needs a vision model:")
    for name, why in (result.get("needs_model") or {}).items():
        print(f"  model {name}: {why}")
    print("model path:", "available"
          if result.get("router_vision") else "not configured")


def _cmd_vision(args: argparse.Namespace, context: Any) -> int:
    """Route `nm vision` to its subcommands."""
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
                                       strategy=getattr(args, "strategy",
                                                        "auto"),
                                       **_vision_source(args.file))
        elif action == "locate":
            kwargs: dict[str, Any] = {}
            if getattr(args, "template", ""):
                kwargs["template_path"] = args.template
            result = _vision_tool_call(context, "vision_locate",
                                       target=args.target,
                                       **kwargs,
                                       **_vision_source(args.file))
        elif action == "compare":
            result = _vision_tool_call(
                context, "vision_compare",
                **_compare_sources(args))
        elif action == "extract":
            result = _vision_tool_call(context, "vision_extract",
                                       prompt=getattr(args, "prompt", "") or "",
                                       **_vision_source(args.file))
        elif action == "analyze":
            result = _vision_tool_call(context, "vision_analyze",
                                       **_vision_source(args.file))
        elif action == "layout":
            result = _vision_tool_call(context, "vision_layout",
                                       **_vision_source(args.file))
        elif action == "metadata":
            result = _vision_tool_call(context, "vision_metadata",
                                       **_vision_source(args.file))
        elif action == "info":
            result = _vision_tool_call(context, "vision_capabilities")
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


def _compare_sources(args: argparse.Namespace) -> dict[str, Any]:
    """Build path_a=/url_a= + path_b=/url_b= kwargs for vision_compare."""
    out: dict[str, Any] = {}
    for attr, prefix in (("file_a", "a"), ("file_b", "b")):
        src = _vision_source(getattr(args, attr))
        for key, value in src.items():
            out[f"{key}_{prefix}"] = value
    prompt = getattr(args, "prompt", "") or ""
    if prompt:
        out["prompt"] = prompt
    return out
