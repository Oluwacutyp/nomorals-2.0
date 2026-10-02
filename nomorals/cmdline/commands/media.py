"""``nm media`` / ``nm studio`` — media and studio pipelines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
from pathlib import Path



_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv"}


def _media_tools(context: Any) -> Any:
    return context.tools.register_builtins()


def _media_call(context: Any, name: str, **kwargs: Any) -> Any:
    """Call a media tool; return the value or raise a clean error."""
    outcome = _media_tools(context).call(name, actor="cli", **kwargs)
    if not outcome.ok:
        err = outcome.error
        msg = getattr(err, "message", None) or str(err)
        raise RuntimeError(msg)
    return outcome.value


def _cmd_media(args: argparse.Namespace, context: Any) -> int:
    """Route `nm media` to edit / probe / jobs / convert."""
    action = args.media_action
    if action == "edit":
        return _cmd_media_edit(args, context)
    if action == "probe":
        return _cmd_media_probe(args, context)
    if action == "jobs":
        return _cmd_media_jobs(args, context)
    if action == "convert":
        return _cmd_media_convert(args, context)
    print(f"unknown media action: {action}", file=sys.stderr)
    return 2


def _cmd_studio(args: argparse.Namespace, context: Any) -> int:
    """Route `nm studio` to the EditStudio tools."""
    action = args.studio_action
    as_json = getattr(args, "json", False)
    try:
        if action == "presets":
            result = _media_call(context, "studio_presets")
            if as_json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print("filter presets:  " + ", ".join(result["filters"]))
                print("transitions:     " + ", ".join(result["transitions"]))
                print("export presets:  " + ", ".join(result["export_presets"]))
                print("templates:       " + ", ".join(result["templates"]))
                print("blend modes:     " + ", ".join(result["blend_modes"]))
            return 0
        if action == "filter":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "filter", "preset": args.preset,
                      "strength": args.strength}])
        elif action == "grade":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "grade", "temperature": args.temperature,
                      "tint": args.tint, "saturation": args.saturation,
                      "contrast": args.contrast, "vignette": args.vignette}])
        elif action == "text":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "text_layer", "text": args.text,
                      "position": args.position, "size": args.size,
                      "color": args.color}])
        elif action == "ai":
            gen_op: dict[str, Any] = {
                "op": "generative_edit", "instruction": args.instruction,
                "strength": args.strength}
            if args.seed is not None:
                gen_op["seed"] = args.seed
            if args.mask:
                gen_op["mask"] = [int(v) for v in args.mask.split(",")]
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[gen_op])
        elif action == "template":
            params: dict[str, Any] = {}
            for kv in args.param or []:
                if "=" not in kv:
                    print(f"bad --param {kv!r}; use k=v", file=sys.stderr)
                    return 2
                k, v = kv.split("=", 1)
                params[k.strip()] = v.strip()
            result = _media_call(context, "studio_template",
                                 template=args.name, params=params,
                                 wait=args.wait)
        elif action == "project":
            if args.action == "save":
                if not args.file or not args.project:
                    print("project save needs <file> <project>",
                          file=sys.stderr)
                    return 2
                result = _media_call(context, "studio_project",
                                     action="save", source=args.file,
                                     project_path=args.project,
                                     ops=json.loads(args.ops))
            elif args.action == "render":
                if not args.file:
                    print("project render needs <project>", file=sys.stderr)
                    return 2
                result = _media_call(context, "studio_project",
                                     action="render", project_path=args.file,
                                     wait=args.wait)
            else:
                if not args.file:
                    print("project describe needs <project>", file=sys.stderr)
                    return 2
                result = _media_call(context, "studio_project",
                                     action="describe",
                                     project_path=args.file)
                print(result["describe"])
                return 0
        elif action == "batch":
            result = _media_call(context, "studio_batch", src_dir=args.dir,
                                 project=args.project, pattern=args.pattern,
                                 out_dir=args.out_dir)
        elif action == "compare":
            result = _media_call(context, "studio_compare", source=args.file,
                                 ops=json.loads(args.ops), mode=args.mode)
        elif action == "gen-status":
            result = _media_call(context, "studio_gen_status")
            if as_json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"selected backend: {result['selected']}")
                print(f"huggingface_hub installed: {result['hf_installed']}")
                print(f"HF_TOKEN set: {result['hf_token_set']} "
                      f"(model: {result['hf_model']})")
                print(f"diffusers available: {result['diffusers_available']} "
                      f"(model: {result['diffusers_model']})")
                if not result['hf_installed'] and not result[
                        'diffusers_available']:
                    print("no generative backend ready: pip install "
                          "huggingface_hub + set HF_TOKEN, or install "
                          "diffusers+torch")
            return 0
        else:
            print(f"unknown studio action: {action}", file=sys.stderr)
            return 2
    except RuntimeError as exc:
        print(f"studio failed: {exc}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    elif result.get("job_id"):
        print(f"job {result['job_id']} queued")
        if args.wait:
            return _cmd_media_wait(args, context, result["job_id"], as_json)
    elif action == "batch":
        print(f"batch: {result['ok']}/{result['files']} ok")
    elif action == "compare":
        print(f"comparison → {result['output']}")
    elif action == "template":
        print(f"template {result.get('template')} → "
              f"{result.get('output') or result.get('job_id')}")
    elif action == "project":
        print(result.get("saved", result.get("output", result)))
    else:
        print(f"wrote {result['output']}")
        print(f"original untouched: {result['input']}")
    return 0


def _is_video_file(path: str, args: argparse.Namespace) -> bool:
    if getattr(args, "video", False):
        return True
    if getattr(args, "image", False):
        return False
    return Path(path).suffix.lower() in _VIDEO_EXTS


def _cmd_media_edit(args: argparse.Namespace, context: Any) -> int:
    as_json = getattr(args, "json", False)
    try:
        if _is_video_file(args.file, args):
            result = _media_call(context, "media_edit_video",
                                 video_path=args.file,
                                 instruction=args.instruction,
                                 dry_run=args.dry_run)
        else:
            result = _media_call(context, "media_edit",
                                 image_path=args.file,
                                 instruction=args.instruction,
                                 dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"media edit failed: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(result["plan"])
        return 0
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    elif result.get("job_id"):
        print(f"job {result['job_id']} queued: {result.get('summary', '')}")
        print(f"poll with: nm media jobs / media_job_status({result['job_id']})")
    else:
        print(f"wrote {result['output']}  ({result.get('summary', '')})")
        print(f"original untouched: {result['input']}")
    if result.get("job_id") and args.wait:
        return _cmd_media_wait(args, context, result["job_id"], as_json)
    return 0


def _cmd_media_wait(args: argparse.Namespace, context: Any,
                    job_id: str, as_json: bool) -> int:
    import time
    timeout = getattr(args, "timeout", 900.0)
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            info = _media_call(context, "media_job_status", job_id=job_id)
        except RuntimeError as exc:
            print(f"job poll failed: {exc}", file=sys.stderr)
            return 1
        status = info["status"]
        if status != last:
            prog = info.get("progress")
            extra = f" {prog:.0%}" if isinstance(prog, float) else ""
            print(f"job {job_id}: {status}{extra}")
            last = status
        if status in ("done", "failed"):
            if as_json:
                print(json.dumps(info, indent=2, default=str))
            elif status == "done":
                print(f"done → {info.get('output_ref')}")
            else:
                print(f"FAILED: {info.get('error')}", file=sys.stderr)
            return 0 if status == "done" else 1
        time.sleep(1.0)
    print(f"timed out after {timeout:.0f}s waiting for job {job_id}",
          file=sys.stderr)
    return 1


def _cmd_media_probe(args: argparse.Namespace, context: Any) -> int:
    try:
        info = _media_call(context, "media_edit_probe", path=args.file)
    except RuntimeError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(info, indent=2, default=str))
        return 0
    if info.get("kind") == "video":
        print(f"{info['path']}: {info.get('width')}x{info.get('height')} "
              f"{info.get('video_codec')} {info.get('fps') or '?'}fps "
              f"{(info.get('duration') or 0):.1f}s "
              f"({info['bytes'] / 1e6:.1f}MB)")
    else:
        print(f"{info['path']}: {info.get('width')}x{info.get('height')} "
              f"{info.get('format')} {info.get('mode')} "
              f"({info['bytes'] / 1024:.0f}KB)")
    return 0


def _cmd_media_jobs(args: argparse.Namespace, context: Any) -> int:
    try:
        jobs = _media_call(context, "media_jobs", limit=args.limit)
    except RuntimeError as exc:
        print(f"jobs failed: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(jobs, indent=2, default=str))
        return 0
    if not jobs:
        print("no media jobs yet")
        return 0
    for j in jobs:
        prog = j.get("progress")
        extra = f" {prog:.0%}" if isinstance(prog, float) else ""
        print(f"{j['id'][:13]}  {j['status']}{extra}  {j['kind']}  "
              f"{j.get('label', '')}  → {j.get('output_ref') or '-'}")
    return 0


def _cmd_media_convert(args: argparse.Namespace, context: Any) -> int:
    as_json = getattr(args, "json", False)
    try:
        result = _media_call(context, "media_convert", path=args.file,
                             format=args.fmt)
    except RuntimeError as exc:
        print(f"convert failed: {exc}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    elif result.get("job_id"):
        print(f"job {result['job_id']} queued: {result.get('summary', '')}")
    else:
        print(f"wrote {result['output']}")
        print(f"original untouched: {result['input']}")
    if result.get("job_id") and args.wait:
        return _cmd_media_wait(args, context, result["job_id"], as_json)
    return 0
