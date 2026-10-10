"""Builtin tools for universal media editing (``nomorals.media_edit``).

Image edits run synchronously; video edits enqueue background jobs.
Inputs are sandboxed to the workspace via :func:`filesystem.safe_path` —
absolute paths outside the workspace are rejected. Originals are never
overwritten: every edit writes a new file under an ``edited/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB default cap
MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB default cap

# Vision-locate hook: "circle the <thing>" needs a box for <thing>.
# Signature: locator(image_path, description) -> (l, t, r, b) | None.
# Tests inject a mock via register_locator(); a future vision tool can wire
# the real thing here.
_Locator = Callable[[str, str], tuple[int, int, int, int] | None]
_locator: _Locator | None = None


def register_locator(fn: _Locator | None) -> None:
    """Install (or clear) the vision-locate hook used by 'circle the X'."""
    global _locator
    _locator = fn


def _resolve_locate(ops: list[dict[str, Any]], image_path: str) -> list[dict[str, Any]]:
    resolved = []
    for op in ops:
        op = dict(op)
        thing = op.pop("locate", None)
        if thing is not None:
            if _locator is None:
                from ..media_edit.images import MediaEditError
                raise MediaEditError(
                    f"cannot locate {thing!r}: no vision locator is wired. "
                    "Pass an explicit box= instead, or register a locator with "
                    "nomorals.tools.media_edit.register_locator().")
            box = _locator(image_path, thing)
            if not box:
                from ..media_edit.images import MediaEditError
                raise MediaEditError(f"could not find {thing!r} in the image")
            op["box"] = tuple(int(v) for v in box)
        resolved.append(op)
    return resolved


def _sandbox(context: Any, path: str, *, must_exist: bool = True) -> Path:
    from .filesystem import safe_path
    return safe_path(context, path, must_exist=must_exist)


def _check_image_size(p: Path) -> None:
    size = p.stat().st_size
    if size > MAX_IMAGE_BYTES:
        from ..media_edit.images import MediaEditError
        raise MediaEditError(
            f"{p.name} is {size / 1e6:.1f}MB, over the "
            f"{MAX_IMAGE_BYTES / 1e6:.0f}MB image cap")


def _check_video_size(p: Path) -> None:
    size = p.stat().st_size
    if size > MAX_VIDEO_BYTES:
        from ..media_edit.images import MediaEditError
        raise MediaEditError(
            f"{p.name} is {size / 1e6:.0f}MB, over the "
            f"{MAX_VIDEO_BYTES / 1e6:.0f}MB video cap")


def _resolve_watermark_logos(context: Any,
                             ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for op in ops:
        op = dict(op)
        if op.get("op") == "watermark" and isinstance(op.get("logo"), str):
            op["logo"] = str(_sandbox(context, op["logo"]))
        out.append(op)
    return out


VIDEO_ACTIONS = {"trim", "concat", "transcode", "extract_frames",
                 "extract_audio", "make_gif", "burn_subtitles",
                 "speed", "fade", "overlay_text",
                 "transition", "effect", "speed_ramp",
                 "ducking", "mix_audio",
                 # sweep(media_edit) additions
                 "loudnorm", "watermark", "rotate_video",
                 "flip_video", "concat_normalized"}


def _dispatch_video(action: dict[str, Any], src: Path,
                    progress_cb: Callable[[float], None] | None = None,
                    out_dir: Path | None = None,
                    context: Any = None) -> dict[str, Any]:
    from ..media_edit import videos
    name = action.get("video_op")
    if name not in VIDEO_ACTIONS:
        from ..media_edit.images import MediaEditError
        raise MediaEditError(
            f"unknown video op {name!r}; allowed: {sorted(VIDEO_ACTIONS)}")
    kw = {k: v for k, v in action.items() if k != "video_op"}
    kw.setdefault("progress_cb", progress_cb)
    if out_dir is not None:
        kw.setdefault("out_dir", out_dir)
    if name == "trim":
        return videos.trim(src, kw.pop("start", 0), kw.pop("end", None), **kw)
    if name == "extract_audio":
        return videos.extract_audio(src, **kw)
    if name == "make_gif":
        return videos.make_gif(src, **kw)
    if name == "extract_frames":
        return videos.extract_frames(src, **kw)
    if name == "transcode":
        return videos.transcode(src, **kw)
    if name == "concat":
        sources = kw.pop("sources", None) or []
        # wave D: every extra source goes through the sandbox too — an
        # unsandboxed path here reached ffmpeg's concat demuxer raw.
        others = [str(_sandbox(context, s)) for s in sources]
        return videos.concat([src] + others, **kw)
    if name == "burn_subtitles":
        sub = kw.pop("subtitles", None)
        if not sub:
            from ..media_edit.images import MediaEditError
            raise MediaEditError("burn_subtitles needs a 'subtitles' file")
        return videos.burn_subtitles(src, _sandbox(context, sub), **kw)
    if name == "speed":
        return videos.speed(src, kw.pop("factor", 1.0), **kw)
    if name == "fade":
        return videos.fade(src, kw.pop("fade_in", 0.0),
                            kw.pop("fade_out", 0.0), **kw)
    if name == "overlay_text":
        return videos.overlay_text(src, kw.pop("text", ""), **kw)
    if name == "transition":
        other = kw.pop("other", None)
        if not other:
            from ..media_edit.images import MediaEditError
            raise MediaEditError(
                "transition needs an 'other' clip (second video)")
        return videos.transition(src, _sandbox(context, other), **kw)
    if name == "effect":
        return videos.effect(src, kw.pop("preset", "grayscale"), **kw)
    if name == "speed_ramp":
        segments = kw.pop("segments", None)
        if not segments:
            from ..media_edit.images import MediaEditError
            raise MediaEditError(
                "speed_ramp needs a 'segments' list of "
                "[start, end, factor] triples")
        return videos.speed_ramp(src, segments, **kw)
    if name == "ducking":
        music = kw.pop("music", None)
        if not music:
            from ..media_edit.images import MediaEditError
            raise MediaEditError("ducking needs a 'music' audio file")
        return videos.ducking(src, _sandbox(context, music), **kw)
    if name == "mix_audio":
        audio = kw.pop("audio", None)
        if not audio:
            from ..media_edit.images import MediaEditError
            raise MediaEditError("mix_audio needs an 'audio' file")
        return videos.mix_audio(src, _sandbox(context, audio), **kw)
    # sweep(media_edit) new ops
    if name == "loudnorm":
        return videos.loudnorm(src, **kw)
    if name == "watermark":
        logo = kw.pop("logo", None)
        if not logo:
            from ..media_edit.images import MediaEditError
            raise MediaEditError("watermark needs a 'logo' image file")
        return videos.watermark(src, _sandbox(context, logo), **kw)
    if name == "rotate_video":
        angle = kw.pop("angle", 90)
        return videos.rotate_video(src, angle, **kw)
    if name == "flip_video":
        return videos.flip_video(src, **kw)
    if name == "concat_normalized":
        sources = kw.pop("sources", None) or []
        others = [str(_sandbox(context, s)) for s in sources]
        return videos.concat_normalized([src] + others, **kw)
    raise AssertionError("unreachable")


def register(registry: Any) -> None:
    """Attach the media-editing tools to a registry."""
    context = registry.context

    @registry.register(
        "media_edit",
        description=("Edit an image from plain language "
                     "('make it square', 'rotate 90', 'circle the login button'). "
                     "Runs synchronously; writes a new file under edited/."),
        capability=Capability.FS_WRITE,
    )
    def media_edit(image_path: str, instruction: str = "",
                   *, dry_run: bool = False,
                   ops: list[dict[str, Any]] | None = None,
                   suffix: str = "edited") -> dict[str, Any]:
        """Edit an image via natural language or an explicit op chain."""
        from ..media_edit.images import edit_image, validate_ops, MediaEditError
        from ..media_edit.intent import parse_instruction, ParsedIntent
        src = _sandbox(context, image_path)
        _check_image_size(src)
        if ops is not None:
            plan = ParsedIntent(kind="image", ops=validate_ops(ops),
                                summary="explicit op chain")
        else:
            if not instruction.strip():
                raise MediaEditError("give an instruction or an ops chain")
            plan = parse_instruction(instruction, kind="image")
        plan.ops = _resolve_locate(plan.ops, str(src))
        plan.ops = _resolve_watermark_logos(context, plan.ops)
        if dry_run:
            return {"dry_run": True, "plan": plan.describe(),
                    "ops": plan.ops}
        result = edit_image(src, plan.ops, suffix=suffix)
        result["summary"] = plan.summary
        return result

    @registry.register(
        "media_edit_video",
        description=("Edit a video from plain language "
                     "('trim the first 30 seconds', 'extract the audio', "
                     "'make a gif'). Runs as a background job; poll with "
                     "media_job_status."),
        capability=Capability.FS_WRITE,
    )
    def media_edit_video(video_path: str, instruction: str = "",
                         *, dry_run: bool = False,
                         action: dict[str, Any] | None = None) -> dict[str, Any]:
        """Edit a video via natural language or an explicit action descriptor."""
        from ..media_edit.images import MediaEditError
        from ..media_edit.intent import parse_instruction, ParsedIntent
        from ..media_edit.jobs import get_manager
        src = _sandbox(context, video_path)
        _check_video_size(src)
        if action is not None:
            if action.get("video_op") not in VIDEO_ACTIONS:
                raise MediaEditError(
                    f"unknown video op {action.get('video_op')!r}")
            plan = ParsedIntent(kind="video", action=dict(action),
                                summary="explicit video action")
        else:
            if not instruction.strip():
                raise MediaEditError("give an instruction or an action")
            plan = parse_instruction(instruction, kind="video")
        if dry_run:
            return {"dry_run": True, "plan": plan.describe(),
                    "action": plan.action}
        manager = get_manager()
        act = dict(plan.action)

        def _run(progress_cb: Callable[[float], None]) -> dict[str, Any]:
            return _dispatch_video(act, src, progress_cb, context=context)

        job_id = manager.submit("video", plan.summary or "video edit",
                                _run, input_ref=str(src))
        return {"job_id": job_id, "status": "queued",
                "summary": plan.summary,
                "poll": "media_job_status"}

    @registry.register(
        "media_edit_probe",
        description=("Probe a local image or video file: dimensions, duration, "
                     "codec, format. (media_probe probes remote URLs.)"),
        capability=Capability.FS_READ,
    )
    def media_edit_probe(path: str) -> dict[str, Any]:
        """Return metadata for a local image or video file."""
        from ..media_edit.videos import media_probe_any
        src = _sandbox(context, path)
        return media_probe_any(src)

    @registry.register(
        "media_job_status",
        description="Poll a background media job: status, progress, output.",
        capability=Capability.FS_READ,
    )
    def media_job_status(job_id: str) -> dict[str, Any]:
        """Return the current state of a media job."""
        from ..media_edit.jobs import job_status
        return job_status(job_id)

    @registry.register(
        "media_jobs",
        description="List recent media jobs (newest first).",
        capability=Capability.FS_READ,
    )
    def media_jobs(limit: int = 20) -> list[dict[str, Any]]:
        """List recent media editing jobs."""
        from ..media_edit.jobs import get_manager
        return get_manager().list_jobs(limit=limit)

    @registry.register(
        "media_convert",
        description=("Convert an image or video to another format "
                     "('webp', 'mp4', ...). Writes a new file; original kept."),
        capability=Capability.FS_WRITE,
    )
    def media_convert(path: str, format: str) -> dict[str, Any]:
        """Convert a media file to another format, keeping the original."""
        from ..media_edit.images import edit_image, MediaEditError
        from ..media_edit.jobs import get_manager
        src = _sandbox(context, path)
        fmt = format.strip().lower().lstrip(".")
        image_fmts = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "avif"}
        video_fmts = {"mp4", "webm", "mov", "mkv"}
        if fmt in image_fmts:
            _check_image_size(src)
            return edit_image(src, [{"op": "convert",
                                    "format": fmt.upper().replace("JPG", "JPEG")}],
                              suffix=f"converted-{fmt}")
        if fmt in video_fmts:
            from ..media_edit.jobs import get_manager as _gm
            _check_video_size(src)
            manager = _gm()

            def _run(progress_cb: Callable[[float], None]) -> dict[str, Any]:
                return _dispatch_video({"video_op": "transcode",
                                        "ext": f".{fmt}"}, src, progress_cb,
                                       context=context)

            job_id = manager.submit("video", f"convert to {fmt}", _run,
                                    input_ref=str(src))
            return {"job_id": job_id, "status": "queued",
                    "summary": f"convert to {fmt}", "poll": "media_job_status"}
        raise MediaEditError(
            f"unsupported target format {format!r}; images: "
            f"{sorted(image_fmts)}; video: {sorted(video_fmts)}")

    register_studio(registry)


def _resolve_studio_paths(context: Any,
                          ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sandbox every file path referenced inside a studio op chain."""
    resolved = []
    for op in ops:
        op = dict(op)
        name = op.get("op", "")
        if name in ("v_clip", "v_image") and op.get("path"):
            op["path"] = str(_sandbox(context, op["path"]))
        elif name == "v_duck" and op.get("bgm"):
            op["bgm"] = str(_sandbox(context, op["bgm"]))
        elif name == "collage" and op.get("images"):
            op["images"] = [str(_sandbox(context, p))
                            for p in op["images"]]
        elif name == "composite" and op.get("layers"):
            layers = []
            for layer in op["layers"]:
                layer = dict(layer)
                if layer.get("path"):
                    layer["path"] = str(_sandbox(context, layer["path"]))
                layers.append(layer)
            op["layers"] = layers
        elif name == "generative_edit" and isinstance(op.get("mask"), str):
            op["mask"] = str(_sandbox(context, op["mask"]))
        elif name == "inpaint_cv" and isinstance(op.get("mask"), str):
            from ..media_edit.cv_ops import parse_box  # noqa: PLC0415
            if parse_box(op["mask"]) is None:  # box strings need no sandbox
                op["mask"] = str(_sandbox(context, op["mask"]))
        elif name == "seamless_clone":
            if isinstance(op.get("background"), str):
                op["background"] = str(_sandbox(context, op["background"]))
            if isinstance(op.get("mask"), str):
                op["mask"] = str(_sandbox(context, op["mask"]))
        elif name == "match_histogram" and isinstance(op.get("reference"),
                                                      str):
            op["reference"] = str(_sandbox(context, op["reference"]))
        elif name == "color_transfer" and isinstance(op.get("reference"),
                                                     str):
            op["reference"] = str(_sandbox(context, op["reference"]))
        elif name == "panorama" and op.get("images"):
            op["images"] = [str(_sandbox(context, p))
                            for p in op["images"]]
        elif name == "cube_lut" and isinstance(op.get("lut"), str):
            op["lut"] = str(_sandbox(context, op["lut"]))
        elif name == "replace_background":
            if isinstance(op.get("background"), str):
                op["background"] = str(_sandbox(context, op["background"]))
        resolved.append(op)
    return resolved


def _sandbox_template_params(context: Any,
                             params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    for key in ("source", "background", "image", "logo", "bgm"):
        if out.get(key):
            out[key] = str(_sandbox(context, out[key]))
    if out.get("images"):
        out["images"] = [str(_sandbox(context, p)) for p in out["images"]]
    return out


def register_studio(registry: Any) -> None:
    """Attach the EditStudio session tools to a registry."""
    context = registry.context

    @registry.register(
        "studio_run",
        description=("Run a pro edit session: an explicit op chain "
                     "(filter, grade, text_layer, composite, collage, "
                     "smart_crop, letterbox, generative_edit, v_* video ops) "
                     "replayed non-destructively from the source. Images "
                     "render synchronously; video renders as a background "
                     "job unless wait=true."),
        capability=Capability.FS_WRITE,
    )
    def studio_run(source: str, ops: list[dict[str, Any]],
                   *, name: str = "untitled", suffix: str = "studio",
                   wait: bool = False) -> dict[str, Any]:
        """Run an EditStudio op chain against a workspace file."""
        from ..media_edit.studio import EditStudio, MediaEditError
        src = _sandbox(context, source)
        _check_image_size(src) if src.suffix.lower() in (
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
            ".gif", ".avif") else _check_video_size(src)
        st = EditStudio(str(src), name=name)
        for op in _resolve_studio_paths(context, ops):
            op = dict(op)
            st.op(op.pop("op"), **op)
        try:
            return st.render(suffix=suffix, wait=wait)
        except MediaEditError:
            raise

    @registry.register(
        "studio_template",
        description=("Build + render a one-call studio template: "
                     "podcast-clip, quote-card, product-showcase, meme, "
                     "slideshow. Pass template params as a dict."),
        capability=Capability.FS_WRITE,
    )
    def studio_template(template: str, params: dict[str, Any],
                        *, wait: bool = False,
                        suffix: str = "studio") -> dict[str, Any]:
        """Build a template project and render it."""
        from ..media_edit.studio import build_template
        st = build_template(template,
                            **_sandbox_template_params(context, params))
        result = st.render(suffix=suffix, wait=wait)
        result["template"] = template
        return result

    @registry.register(
        "studio_project",
        description=("Save an EditStudio project (action='save'), load one "
                     "('load'), describe it ('describe'), or render a saved "
                     "project ('render'). Projects are JSON op stacks."),
        capability=Capability.FS_WRITE,
    )
    def studio_project(action: str, *,
                       source: str | None = None,
                       ops: list[dict[str, Any]] | None = None,
                       project_path: str | None = None,
                       name: str = "untitled",
                       wait: bool = False,
                       suffix: str = "studio") -> dict[str, Any]:
        """Save/load/describe/render EditStudio project files."""
        from ..media_edit.studio import EditStudio
        action = action.strip().lower()
        if action == "save":
            if not source or not project_path:
                from ..media_edit.images import MediaEditError
                raise MediaEditError("save needs source + project_path")
            src = _sandbox(context, source)
            st = EditStudio(str(src), name=name)
            for op in _resolve_studio_paths(context, ops or []):
                op = dict(op)
                st.op(op.pop("op"), **op)
            dest = _sandbox(context, project_path, must_exist=False)
            st.save_project(dest)
            return {"saved": str(dest), "ops": len(st.ops)}
        if action in ("load", "describe", "render"):
            if not project_path:
                from ..media_edit.images import MediaEditError
                raise MediaEditError(f"{action} needs project_path")
            proj = _sandbox(context, project_path)
            st = EditStudio.load_project(proj)
            if action == "describe":
                return {"project": str(proj), "describe": st.describe()}
            # re-sandbox paths referenced by the loaded project
            st.ops = _resolve_studio_paths(context, st.ops)
            if st.source:
                try:
                    st.source = str(_sandbox(context, st.source))
                except Exception:  # noqa: BLE001
                    pass  # slideshow templates legitimately have no source
            result = st.render(suffix=suffix, wait=wait)
            result["project"] = str(proj)
            return result
        from ..media_edit.images import MediaEditError
        raise MediaEditError(
            f"unknown studio_project action {action!r}; "
            "use save|load|describe|render")

    @registry.register(
        "studio_batch",
        description=("Apply a saved studio project to every file in a folder "
                     "(pattern like '*.jpg'). Returns per-file results."),
        capability=Capability.FS_WRITE,
    )
    def studio_batch(src_dir: str, project: str, *,
                     pattern: str = "*.jpg",
                     out_dir: str | None = None,
                     suffix: str = "studio") -> dict[str, Any]:
        """Batch-apply a studio project across a directory."""
        from ..media_edit.studio import EditStudio
        d = _sandbox(context, src_dir)
        proj = _sandbox(context, project)
        target = (_sandbox(context, out_dir, must_exist=False)
                  if out_dir else None)
        results = EditStudio.batch(d, proj, pattern=pattern,
                                   out_dir=target, suffix=suffix)
        ok = sum(1 for r in results if "error" not in r)
        return {"files": len(results), "ok": ok, "results": results}

    @registry.register(
        "studio_compare",
        description=("Render a studio op chain and export a before/after "
                     "comparison: 'side-by-side', 'split', 'stacked', or "
                     "'html' (interactive slider)."),
        capability=Capability.FS_WRITE,
    )
    def studio_compare(source: str, ops: list[dict[str, Any]],
                       *, mode: str = "side-by-side",
                       suffix: str = "studio") -> dict[str, Any]:
        """Render + export a before/after comparison."""
        from ..media_edit.studio import EditStudio
        src = _sandbox(context, source)
        _check_image_size(src)
        st = EditStudio(str(src), name="compare")
        for op in _resolve_studio_paths(context, ops):
            op = dict(op)
            st.op(op.pop("op"), **op)
        return st.compare(mode=mode)

    @registry.register(
        "studio_presets",
        description=("List everything the studio offers: filter presets, "
                     "transitions, export presets, templates, blend modes."),
        capability=Capability.FS_READ,
    )
    def studio_presets() -> dict[str, Any]:
        """List studio filters, transitions, exports, templates."""
        from ..media_edit.studio import studio_presets as _sp
        return _sp()

    @registry.register(
        "studio_gen_status",
        description=("Check generative-edit backend status: which AI backends "
                     "are usable (hf/diffusers), configured model, and the "
                     "env vars to set. Makes no model calls."),
        capability=Capability.FS_READ,
    )
    def studio_gen_status() -> dict[str, Any]:
        """Report generative backend availability (no model calls)."""
        from ..media_edit.generate import backend_status
        return backend_status()
