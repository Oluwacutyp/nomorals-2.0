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
            if args.negative_prompt:
                gen_op["negative_prompt"] = args.negative_prompt
            if args.steps is not None:
                gen_op["steps"] = args.steps
            if args.guidance is not None:
                gen_op["guidance_scale"] = args.guidance
            if args.backend:
                gen_op["backend"] = args.backend
            if args.width is not None:
                gen_op["width"] = args.width
            if args.height is not None:
                gen_op["height"] = args.height
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[gen_op])
        elif action == "generate":
            from ...media_edit.generate import op_txt2img
            # txt2img needs no source image; call the op directly.
            out = op_txt2img(
                args.prompt,
                seed=args.seed,
                backend=args.backend,
                negative_prompt=args.negative_prompt,
                steps=args.steps,
                guidance_scale=args.guidance,
                width=args.width,
                height=args.height,
                style=args.style,
                aspect=args.aspect,
                quality=args.quality,
                n=args.n or 1,
            )
            # Save PNGs next to the invocation dir (or --out path).
            import io as _io
            outs = out if isinstance(out, list) else [out]
            paths = []
            for i, img in enumerate(outs):
                if args.out and len(outs) == 1:
                    dest = Path(args.out)
                elif args.out:
                    dest = Path(args.out).with_stem(
                        f"{Path(args.out).stem}_{i}")
                else:
                    dest = Path(f"generated_{int(time.time())}_{i}.png")
                dest.parent.mkdir(parents=True, exist_ok=True)
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                dest.write_bytes(buf.getvalue())
                paths.append(str(dest))
            result = {"images": paths, "count": len(paths)}
        elif action == "upscale":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "upscale", "scale": args.scale}])
        elif action == "denoise":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "denoise", "strength": args.strength,
                      "method": args.method}])
        elif action == "edges":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "edge_detect", "low": args.low,
                      "high": args.high}])
        elif action == "inpaint-cv":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "inpaint_cv", "mask": args.mask,
                      "method": args.method}])
        elif action == "clone":
            clone_op: dict[str, Any] = {
                "op": "seamless_clone", "background": args.background}
            if args.x is not None and args.y is not None:
                clone_op["position"] = [args.x, args.y]
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[clone_op])
        elif action == "sharpen":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "sharpen", "amount": args.amount,
                      "sigma": args.sigma}])
        elif action == "cartoon":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "cartoonize"}])
        elif action == "sketch":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "pencil_sketch", "blur_sigma": args.sigma}])
        elif action == "perspective":
            persp_op: dict[str, Any] = {"op": "perspective"}
            if args.corners:
                persp_op["corners"] = args.corners
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[persp_op])
        elif action == "grabcut":
            grab_op: dict[str, Any] = {"op": "grabcut",
                                       "background": args.background}
            if args.rect:
                grab_op["rect"] = args.rect
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[grab_op])
        elif action == "rescale":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "rescale", "scale": args.scale,
                      "order": args.order}])
        elif action == "exposure":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "exposure", "mode": args.mode,
                      "gamma": args.gamma}])
        elif action == "match-hist":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "match_histogram",
                      "reference": args.reference}])
        elif action == "threshold":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "adaptive_threshold", "method": args.method}])
        elif action == "features":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "orb_features", "n": args.n}])
        elif action == "kmeans":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "kmeans", "k": args.k}])
        elif action == "deblur":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "deblur", "psf_size": args.psf,
                      "method": args.method}])
        elif action == "superpixels":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "superpixels", "n_segments": args.n}])
        elif action == "white-balance":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "white_balance", "method": args.method}])
        elif action == "autolevels":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "auto_levels", "cutoff": args.cutoff}])
        elif action == "color-transfer":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "color_transfer",
                      "reference": args.reference}])
        elif action == "panorama":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "panorama", "images": list(args.images),
                      "mode": args.mode}])
        elif action == "backends":
            from ...media_edit.cv_ops import backend_status
            result = backend_status()
            if as_json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"opencv:       {result['opencv']}")
                print(f"scikit-image: {result['scikit_image']}")
                print(f"numpy:        {result['numpy']}")
                print("selected backend per op:")
                for op, be in sorted(result["selected"].items()):
                    print(f"  {op:20s} {be}")
            return 0
        elif action == "retouch":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "retouch_smooth", "radius": args.radius,
                      "amount": args.amount}])
        elif action == "tonemap":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "tone_map", "method": args.method,
                      "saturation": args.saturation}])
        elif action == "detail":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "detail_enhance", "sigma_s": args.sigma_s,
                      "sigma_r": args.sigma_r}])
        elif action == "stylize":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "stylize", "sigma_s": args.sigma_s,
                      "sigma_r": args.sigma_r}])
        elif action == "clarity":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "clarity", "amount": args.amount,
                      "radius": args.radius}])
        elif action == "shadows":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "shadow_highlight", "shadows": args.shadows,
                      "highlights": args.highlights}])
        elif action == "dehaze":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "dehaze", "omega": args.omega}])
        elif action == "seamcarve":
            if args.width is None and args.height is None:
                print("seamcarve needs --width and/or --height",
                      file=sys.stderr)
                return 2
            sc_op: dict[str, Any] = {"op": "seam_carve"}
            if args.width is not None:
                sc_op["width"] = args.width
            if args.height is not None:
                sc_op["height"] = args.height
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[sc_op])
        elif action == "tiltshift":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "tilt_shift", "focus_center": args.center,
                      "focus_size": args.size, "blur": args.blur}])
        elif action == "selective":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "selective_color", "hue": args.hue,
                      "hue_width": args.width}])
        elif action == "split-tone":
            def _rgb3(spec: str, what: str) -> tuple[int, int, int]:
                try:
                    r, g, b = (int(v) for v in spec.split(","))
                except ValueError:
                    raise RuntimeError(
                        f"bad --{what} {spec!r}; use r,g,b")
                return (r, g, b)
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "split_tone",
                      "shadows": list(_rgb3(args.shadows, "shadows")),
                      "highlights": list(_rgb3(args.highlights,
                                               "highlights")),
                      "strength": args.strength}])
        elif action == "curves":
            pts = []
            for pair in args.points.split(";"):
                try:
                    x, y = (int(v) for v in pair.split(","))
                except ValueError:
                    print(f"bad --points {pair!r}; use in,out;...",
                          file=sys.stderr)
                    return 2
                pts.append([x, y])
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "curves", "points": pts,
                      "channel": args.channel}])
        elif action == "lut":
            result = _media_call(
                context, "studio_run", source=args.file,
                ops=[{"op": "cube_lut", "lut": args.lut}])
        elif action == "bg-replace":
            bgr_op: dict[str, Any] = {"op": "replace_background",
                                      "background": args.background,
                                      "feather": args.feather}
            if args.rect:
                bgr_op["rect"] = args.rect
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[bgr_op])
        elif action == "img2img":
            i2i_op: dict[str, Any] = {
                "op": "img2img", "prompt": args.prompt,
                "strength": args.strength}
            if args.seed is not None:
                i2i_op["seed"] = args.seed
            if args.style:
                i2i_op["style"] = args.style
            if args.negative_prompt:
                i2i_op["negative_prompt"] = args.negative_prompt
            if args.steps is not None:
                i2i_op["steps"] = args.steps
            if args.guidance is not None:
                i2i_op["guidance_scale"] = args.guidance
            if args.backend:
                i2i_op["backend"] = args.backend
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[i2i_op])
        elif action == "inpaint":
            inp_op: dict[str, Any] = {
                "op": "inpaint", "prompt": args.prompt,
                "mask": [int(v) for v in args.mask.split(",")]}
            if args.seed is not None:
                inp_op["seed"] = args.seed
            if args.negative_prompt:
                inp_op["negative_prompt"] = args.negative_prompt
            if args.steps is not None:
                inp_op["steps"] = args.steps
            if args.guidance is not None:
                inp_op["guidance_scale"] = args.guidance
            if args.backend:
                inp_op["backend"] = args.backend
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[inp_op])
        elif action == "outpaint":
            outp_op: dict[str, Any] = {
                "op": "outpaint", "prompt": args.prompt,
                "top": args.top, "right": args.right,
                "bottom": args.bottom, "left": args.left}
            if args.seed is not None:
                outp_op["seed"] = args.seed
            if args.negative_prompt:
                outp_op["negative_prompt"] = args.negative_prompt
            if args.steps is not None:
                outp_op["steps"] = args.steps
            if args.guidance is not None:
                outp_op["guidance_scale"] = args.guidance
            if args.backend:
                outp_op["backend"] = args.backend
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[outp_op])
        elif action == "bg-remove":
            bg_op: dict[str, Any] = {"op": "bg_remove", "mode": args.mode}
            if args.chroma_color:
                bg_op["chroma_color"] = args.chroma_color
            if args.tolerance is not None:
                bg_op["tolerance"] = args.tolerance
            result = _media_call(context, "studio_run", source=args.file,
                                 ops=[bg_op])
        elif action == "layer":
            return _cmd_studio_layer(args, as_json)
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
                for _pb, _info in (result.get("paid") or {}).items():
                    _state = ("ready — " + _info["reason"] if _info["ready"]
                              else "not ready — " + _info["reason"])
                    print(f"paid backend {_pb}: {_state}")
                if not result['hf_installed'] and not result[
                        'diffusers_available']:
                    print("no generative backend ready: pip install "
                          "huggingface_hub + set HF_TOKEN, or install "
                          "diffusers+torch")
            return 0
        elif action in ("vframes", "vfilter", "vtimelapse", "vstabilize",
                        "vhighlights", "vthumb", "vpreview", "vslowmo",
                        "vreverse", "vboomerang", "vblurry", "vheatmap",
                        "vsplit", "vkenburns", "vslideshow", "vchroma",
                        "vpip", "vfreeze", "vdenoise"):
            return _cmd_studio_cv_video(args, as_json)
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


def _cmd_studio_cv_video(args: argparse.Namespace, as_json: bool) -> int:
    """`nm studio v*` — OpenCV frame-level video ops.

    Calls nomorals.media_edit.cv_video / videos directly (no tool-registry
    round trip); these functions fail with plain, actionable errors.
    """
    from ...media_edit import cv_video
    from ...media_edit import videos as V
    action = args.studio_action
    backend = getattr(args, "backend", "auto") or "auto"
    try:
        if action == "vframes":
            ts = ([t.strip() for t in args.timestamps.split(",")]
                  if args.timestamps else None)
            result = cv_video.extract_frames(
                args.file, timestamps=ts, interval=args.interval,
                count=args.count, width=args.width, out_dir=args.out_dir,
                backend=backend)
        elif action == "vfilter":
            result = cv_video.apply_filter_to_video(
                args.file, filter_name=args.filter,
                strength=args.strength, out_dir=args.out_dir,
                backend=backend)
        elif action == "vtimelapse":
            result = cv_video.create_timelapse(
                args.file, factor=args.factor, out_dir=args.out_dir,
                backend=backend)
        elif action == "vstabilize":
            result = cv_video.stabilize_basic(
                args.file, smoothing_radius=args.smoothing,
                crop_borders=not args.no_crop, out_dir=args.out_dir,
                backend=backend)
        elif action == "vhighlights":
            result = cv_video.frame_diff_highlights(
                args.file, threshold=args.threshold, min_gap=args.min_gap,
                out_dir=args.out_dir, backend=backend)
        elif action == "vthumb":
            result = V.frame_accurate_thumbnail(
                args.file, timestamp=args.timestamp, width=args.width,
                out_dir=args.out_dir, backend=backend)
        elif action == "vpreview":
            result = V.make_preview_grid(
                args.file, cols=args.cols, rows=args.rows,
                cell_width=args.cell_width, out_dir=args.out_dir,
                backend=backend)
        elif action == "vslowmo":
            result = cv_video.slow_motion(
                args.file, factor=args.factor, method=args.method,
                out_dir=args.out_dir, backend=backend)
        elif action == "vreverse":
            result = cv_video.reverse_video(
                args.file, out_dir=args.out_dir, backend=backend)
        elif action == "vboomerang":
            result = cv_video.boomerang(
                args.file, start=args.start, duration=args.duration,
                out_dir=args.out_dir, backend=backend)
        elif action == "vblurry":
            result = cv_video.find_blurry_frames(
                args.file, threshold=args.threshold, out_dir=args.out_dir)
        elif action == "vheatmap":
            result = cv_video.motion_heatmap(
                args.file, out_dir=args.out_dir)
        elif action == "vsplit":
            result = cv_video.split_on_scenes(
                args.file, threshold=args.threshold, min_gap=args.min_gap,
                min_scene=args.min_scene, out_dir=args.out_dir,
                backend=backend)
        elif action == "vkenburns":
            result = cv_video.kenburns(
                args.file, duration=args.duration,
                zoom_from=args.zoom_from, zoom_to=args.zoom_to,
                pan=args.pan, fps=args.fps, out_dir=args.out_dir,
                backend=backend)
        elif action == "vslideshow":
            result = cv_video.slideshow(
                args.images, duration_each=args.duration_each,
                transition=args.transition,
                transition_duration=args.transition_duration,
                out_dir=args.out_dir, backend=backend)
        elif action == "vchroma":
            result = cv_video.chroma_key(
                args.file, args.background, color=args.color,
                similarity=args.similarity, blend=args.blend,
                out_dir=args.out_dir, backend=backend)
        elif action == "vpip":
            result = cv_video.pip(
                args.main, args.overlay, position=args.position,
                scale=args.scale, out_dir=args.out_dir, backend=backend)
        elif action == "vfreeze":
            result = cv_video.freeze_frame(
                args.file, timestamp=args.timestamp,
                duration=args.duration, out_dir=args.out_dir,
                backend=backend)
        elif action == "vdenoise":
            result = cv_video.denoise_video(
                args.file, strength=args.strength, out_dir=args.out_dir,
                backend=backend)
        else:  # pragma: no cover - argparse restricts choices
            print(f"unknown studio action: {action}", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001 - CLI surface: report, don't trace
        print(f"studio {action} failed: {exc}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    if result.get("backend"):
        print(f"backend: {result['backend']}")
    if action == "vframes":
        print(f"{result['count']} frames → {result['frames_dir']}")
    elif action in ("vhighlights", "vblurry"):
        kind = "highlights" if action == "vhighlights" else "blurry frames"
        print(f"{result['count']} {kind} → {result['thumbs_dir']}")
        for ev in result["events"][:10]:
            print(f"  t={ev['t']}s score={ev['score']} {ev['thumb']}")
        if result["count"] > 10:
            print(f"  ... and {result['count'] - 10} more")
    elif action == "vheatmap":
        print(f"heatmap → {result['heatmap']}")
        pk = result["peak_motion"]
        print(f"peak motion: t={pk['t']}s score={pk['score']}")
    elif action == "vsplit":
        print(f"{result['count']} scenes → {result['scenes_dir']}")
        for sc in result["scenes"]:
            print(f"  scene {sc['n']}: {sc['start']}s–{sc['end']}s "
                  f"{sc['output']}")
    else:
        print(f"wrote {result['output']}")
        print(f"original untouched: {result['input']}")
        if result.get("audio_dropped"):
            print("note: output is video-only (OpenCV can't mux audio); "
                  "use `nm media` mix_audio to re-attach sound")
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


def _load_stack(stack_file: str) -> Any:
    """Load a LayerStack from its JSON file (fail fast if missing/bad)."""
    from ...media_edit.layers import LayerStack
    p = Path(stack_file)
    if not p.exists():
        raise RuntimeError(f"no such layer stack: {stack_file} "
                           f"(create one with: nm studio layer new W H "
                           f"--stack {stack_file})")
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"bad layer stack {stack_file}: {exc}") from exc
    return LayerStack.from_dict(data)


def _save_stack(stack: Any, stack_file: str) -> None:
    p = Path(stack_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stack.to_dict(), indent=2))


def _cmd_studio_layer(args: argparse.Namespace, as_json: bool) -> int:
    """Route `nm studio layer <action>` to LayerStack."""
    from ...media_edit.layers import LayerStack
    action = args.layer_action
    try:
        if action == "new":
            stack = LayerStack((args.width, args.height), bg=args.bg)
            _save_stack(stack, args.stack)
            result: dict[str, Any] = {
                "stack": args.stack, "canvas": [args.width, args.height],
                "layers": 0}
        else:
            stack = _load_stack(args.stack)
            if action == "add-image":
                kw: dict[str, Any] = {}
                if args.name:
                    kw["name"] = args.name
                kw["opacity"] = args.opacity
                kw["blend"] = args.blend
                if args.position:
                    kw["position"] = tuple(
                        int(v) for v in args.position.split(","))
                if args.scale is not None:
                    kw["scale"] = args.scale
                lid = stack.add_image(args.image, **kw)
                _save_stack(stack, args.stack)
                result = {"stack": args.stack, "added": lid}
            elif action == "add-text":
                kw = {}
                if args.name:
                    kw["name"] = args.name
                lid = stack.add_text(
                    args.text, font_size=args.size, color=args.color,
                    position=args.position, opacity=args.opacity,
                    blend=args.blend, **kw)
                _save_stack(stack, args.stack)
                result = {"stack": args.stack, "added": lid}
            elif action == "add-shape":
                kw = {}
                if args.name:
                    kw["name"] = args.name
                if args.box:
                    kw["box"] = [int(v) for v in args.box.split(",")]
                lid = stack.add_shape(
                    args.shape, fill=args.fill, opacity=args.opacity,
                    blend=args.blend, **kw)
                _save_stack(stack, args.stack)
                result = {"stack": args.stack, "added": lid}
            elif action == "list":
                result = {"stack": args.stack,
                          "layers": stack.layer_info()}
            elif action == "set":
                lid = args.id
                if args.opacity is not None:
                    stack.set_opacity(lid, args.opacity)
                if args.blend:
                    stack.set_blend(lid, args.blend)
                if args.visible is not None:
                    v = args.visible.lower() not in ("0", "false", "no")
                    stack.set_visible(lid, v)
                if args.move:
                    x, y = (int(v) for v in args.move.split(","))
                    stack.move(lid, x, y)
                if args.rename:
                    stack.rename(lid, args.rename)
                _save_stack(stack, args.stack)
                result = {"stack": args.stack, "updated": lid}
            elif action == "remove":
                stack.remove(args.id)
                _save_stack(stack, args.stack)
                result = {"stack": args.stack, "removed": args.id}
            elif action == "flatten":
                flat = stack.flatten()
                dest = Path(args.out) if args.out else Path(
                    args.stack).with_name(
                        f"{Path(args.stack).stem}_flat.png")
                dest.parent.mkdir(parents=True, exist_ok=True)
                flat.convert("RGB").save(dest)
                result = {"stack": args.stack, "output": str(dest),
                          "size": list(flat.size)}
            else:
                print(f"unknown layer action: {action}", file=sys.stderr)
                return 2
    except RuntimeError as exc:
        print(f"layer failed: {exc}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    elif action == "list":
        for info in result["layers"]:
            print(f"{info['id']:12} {info['name'][:20]:20} "
                  f"{info['type']:8} op={info['opacity']:.2f} "
                  f"blend={info['blend']} "
                  f"{'visible' if info['visible'] else 'hidden'}")
    elif action == "flatten":
        print(f"wrote {result['output']} "
              f"({result['size'][0]}x{result['size'][1]})")
    else:
        print(json.dumps(result, default=str))
    return 0
