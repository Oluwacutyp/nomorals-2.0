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
                 "extract_audio", "make_gif", "burn_subtitles"}


def _dispatch_video(action: dict[str, Any], src: Path,
                    progress_cb: Callable[[float], None] | None = None,
                    out_dir: Path | None = None) -> dict[str, Any]:
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
        return videos.concat([src] + list(sources), **kw)
    if name == "burn_subtitles":
        sub = kw.pop("subtitles", None)
        if not sub:
            from ..media_edit.images import MediaEditError
            raise MediaEditError("burn_subtitles needs a 'subtitles' file")
        return videos.burn_subtitles(src, sub, **kw)
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
            return _dispatch_video(act, src, progress_cb)

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
                                        "ext": f".{fmt}"}, src, progress_cb)

            job_id = manager.submit("video", f"convert to {fmt}", _run,
                                    input_ref=str(src))
            return {"job_id": job_id, "status": "queued",
                    "summary": f"convert to {fmt}", "poll": "media_job_status"}
        raise MediaEditError(
            f"unsupported target format {format!r}; images: "
            f"{sorted(image_fmts)}; video: {sorted(video_fmts)}")
