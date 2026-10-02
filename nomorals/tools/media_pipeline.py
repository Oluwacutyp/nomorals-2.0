"""Download → compress → send pipeline: one coherent media flow.

Three disconnected tools already exist — ``media_download`` (fetch),
``compress_file`` (ffmpeg/zip), and ``file_send`` (gateway) — but using
them by hand means the agent juggles three calls, three paths, and three
failure modes. This module wires them into a single pipeline with a
per-stage status report:

    download (fetch the URL) → compress (shrink when it helps) → send (to chat)

Compression strategy per file kind:

* **image** (jpg/png/webp/bmp/tiff) — downscale to ``image_max_width`` and
  re-encode with Pillow (JPEG quality ``image_quality``, PNG optimized).
  Without Pillow the stage reports honestly that it skipped.
* **video** — delegated to :func:`nomorals.tools.compress.compress_file`,
  which re-encodes with ffmpeg when installed and says so when it isn't.
* **everything else** — no compression is attempted (zipping a PDF just to
  send it would be hostile); the stage reports the skip and the original
  goes out.

Failure policy is fail-fast, never silent: a download failure raises, a
send failure raises, and a compress miss is recorded in the stage report
with its reason instead of being swallowed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ..core.errors import MediaError, ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .filesystem import safe_path

_log = get_logger(__name__)

__all__ = ["run_pipeline", "compress_image_for_send", "register"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

#: what goes through the ffmpeg path in tools.compress
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _stage(name: str, started: float) -> dict[str, Any]:
    return {"stage": name, "seconds": round(time.perf_counter() - started, 2)}


def compress_image_for_send(
    path: str | Path,
    *,
    max_width: int = 1600,
    quality: int = 80,
) -> dict[str, Any]:
    """Downscale + re-encode an image so it is cheap to send.

    Returns a compress-style report: ``ok`` True with the new path when the
    result is actually smaller, otherwise ``ok`` False with a ``reason`` —
    the caller always knows whether compression happened and why not.
    """
    src = Path(path)
    if not src.is_file():
        return {"ok": False, "method": "none",
                "reason": f"no such file: {src}", "path": str(src)}
    original = src.stat().st_size
    try:
        from ..media_edit.images import load_image, save_image
    except ImportError as exc:
        return {
            "ok": False, "method": "none",
            "reason": f"Pillow not installed — cannot re-encode images ({exc})",
            "path": str(src), "original_bytes": original, "new_bytes": original,
            "ratio": 1.0,
        }
    try:
        img = load_image(src)
    except Exception as exc:  # noqa: BLE001 - corrupt/unsupported: say so, don't crash
        return {"ok": False, "method": "none",
                "reason": f"could not decode image: {exc}",
                "path": str(src), "original_bytes": original,
                "new_bytes": original, "ratio": 1.0}

    width, height = img.size
    resized = img
    note = ""
    if width > max_width:
        try:
            from PIL import Image as _PILImage
        except ImportError as exc:
            return {"ok": False, "method": "none",
                    "reason": f"Pillow not installed ({exc})",
                    "path": str(src), "original_bytes": original,
                    "new_bytes": original, "ratio": 1.0}
        new_height = max(1, round(height * max_width / width))
        resized = img.resize((max_width, new_height), _PILImage.LANCZOS)
        note = f"downscaled {width}x{height} → {max_width}x{new_height}; "

    dest = src.with_name(f"{src.stem}.send{src.suffix}")
    try:
        save_image(resized, dest, quality=quality)
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        return {"ok": False, "method": "none",
                "reason": f"re-encode failed: {exc}",
                "path": str(src), "original_bytes": original,
                "new_bytes": original, "ratio": 1.0}
    new = dest.stat().st_size
    if new >= original:
        dest.unlink(missing_ok=True)
        return {
            "ok": False, "method": "none",
            "reason": f"{note}re-encoded file ({new} bytes) is not smaller "
                      f"than the original ({original} bytes) — sending the original",
            "path": str(src), "original_bytes": original,
            "new_bytes": original, "ratio": 1.0,
        }
    return {
        "ok": True, "method": f"pillow-reencode ({note.rstrip('; ')})" if note
        else "pillow-reencode",
        "path": str(dest),
        "original_bytes": original, "new_bytes": new,
        "ratio": round(new / original, 3) if original else 1.0,
    }


def _compress_stage(src: Path, *, image_max_width: int, image_quality: int,
                    enabled: bool) -> dict[str, Any]:
    """Pick the compression strategy by file kind. Honest no-ops included."""
    if not enabled:
        return {"ok": False, "method": "none", "path": str(src),
                "reason": "compression disabled by caller (compress=False)",
                "original_bytes": src.stat().st_size,
                "new_bytes": src.stat().st_size, "ratio": 1.0}
    ext = src.suffix.lower()
    if ext in IMAGE_EXTS:
        return compress_image_for_send(
            src, max_width=image_max_width, quality=image_quality)
    if ext in VIDEO_EXTS:
        from .compress import compress_file

        report = compress_file(src)
        # compress_file is already honest about missing ffmpeg; normalize keys
        report.setdefault("path", str(src))
        return report
    size = src.stat().st_size
    return {
        "ok": False, "method": "none", "path": str(src),
        "reason": f"{ext or 'unknown type'} is not an image or video — "
                  "no compression attempted; the original will be sent",
        "original_bytes": size, "new_bytes": size, "ratio": 1.0,
    }


def run_pipeline(
    context: Any,
    url: str,
    platform: str,
    chat_id: str,
    *,
    audio_only: bool = False,
    format_spec: str = "bestvideo*+bestaudio/best",
    compress: bool = True,
    image_max_width: int = 1600,
    image_quality: int = 80,
    caption: str = "",
    download_timeout: float = 1800.0,
    downloader: Callable[..., dict[str, Any]] | None = None,
    sender: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch ``url``, optionally compress, then send to the active chat.

    Returns a report with one entry per stage — ``download``, ``compress``,
    ``send`` — each carrying ``ok`` and its details. The pipeline is
    fail-fast: a failed download raises :class:`MediaError`, a failed send
    raises :class:`ToolError`. A compress miss never fails the pipeline;
    it is recorded with its reason and the original file goes out.

    ``downloader`` / ``sender`` are injectable for tests; production code
    leaves them ``None`` and the real tools are used.
    """
    if not (url or "").strip():
        raise MediaError("pipeline needs a URL")
    if not (platform or "").strip():
        raise ToolError("pipeline needs a platform (telegram | whatsapp | …)")
    if not (chat_id or "").strip():
        raise ToolError("pipeline needs a chat_id")

    started_all = time.perf_counter()
    stages: list[dict[str, Any]] = []

    # ── stage 1: download ─────────────────────────────────────────────
    stage_start = time.perf_counter()
    from .media import download as _download
    from .filesend import send_file as _send_file

    fetch = downloader or _download
    dest_dir = safe_path(context, "media/pipeline")
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = fetch(url, dest_dir, format_spec=format_spec,
                           audio_only=audio_only, timeout=download_timeout)
    except TypeError:
        # injected fakes may not accept the full keyword surface
        downloaded = fetch(url, dest_dir)
    except Exception as exc:  # noqa: BLE001 - fail fast, but name the stage
        raise MediaError(f"pipeline download failed: {exc}") from exc
    path = Path(downloaded["path"])
    if not path.is_file() or path.stat().st_size == 0:
        raise MediaError(
            f"pipeline download produced no usable file at {downloaded.get('path')}")
    stage = _stage("download", stage_start)
    stage.update(ok=True, path=str(path), bytes=path.stat().st_size,
                 title=downloaded.get("title", ""),
                 extractor=downloaded.get("extractor", ""))
    stages.append(stage)
    _log.info("pipeline downloaded %s (%d bytes) via %s",
              path.name, path.stat().st_size, downloaded.get("extractor", "?"))

    # ── stage 2: compress (best-effort, reported honestly) ─────────────
    stage_start = time.perf_counter()
    report = _compress_stage(path, image_max_width=image_max_width,
                             image_quality=image_quality, enabled=compress)
    stage = _stage("compress", stage_start)
    stage.update(report)
    stages.append(stage)
    send_path = report.get("path") or str(path)
    if report.get("ok"):
        _log.info("pipeline compressed %s → %s (ratio %s)",
                  path.name, Path(send_path).name, report.get("ratio"))
    else:
        _log.info("pipeline compress skipped: %s", report.get("reason"))

    # ── stage 3: send ─────────────────────────────────────────────────
    stage_start = time.perf_counter()
    send = sender or _send_file
    try:
        sent = send(context, platform, chat_id, send_path, caption=caption)
    except TypeError:
        sent = send(context, platform, chat_id, send_path)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"pipeline send failed: {exc}") from exc
    if isinstance(sent, dict) and sent.get("sent") is False:
        raise ToolError(f"pipeline send reported failure: {sent}")
    stage = _stage("send", stage_start)
    stage.update(ok=True, platform=platform, chat=chat_id,
                 sent_path=send_path,
                 message_id=(sent or {}).get("message_id", "")
                 if isinstance(sent, dict) else "")
    stages.append(stage)

    return {
        "ok": True,
        "stages": stages,
        "downloaded_path": str(path),
        "sent_path": send_path,
        "bytes_sent": Path(send_path).stat().st_size,
        "compressed": bool(report.get("ok")),
        "seconds": round(time.perf_counter() - started_all, 2),
    }


def register(registry: Any) -> None:
    """Attach the pipeline tool to a registry."""
    context = registry.context

    @registry.register(
        "media_pipeline",
        description=(
            "Download → compress → send in one call: fetch a file/URL "
            "(media_download), shrink it when that helps (images: "
            "Pillow downscale+re-encode; video: ffmpeg via compress_file; "
            "both report honestly when tooling is missing), then send the "
            "result to a chat (file_send). Returns per-stage status."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "url": "str — media URL or direct file link",
            "platform": "str — telegram | whatsapp | discord | console | …",
            "chat_id": "str — the target chat id",
            "audio_only": "bool (optional, False) — extract audio only",
            "format_spec": "str (optional) — yt-dlp format selector",
            "compress": "bool (optional, True) — shrink before sending",
            "image_max_width": "int (optional, 1600)",
            "image_quality": "int (optional, 80) — JPEG quality",
            "caption": "str (optional)",
            "download_timeout": "float (optional, 1800)",
        },
    )
    def media_pipeline(url: str, platform: str, chat_id: str, *,
                       audio_only: bool = False,
                       format_spec: str = "bestvideo*+bestaudio/best",
                       compress: bool = True,
                       image_max_width: int = 1600,
                       image_quality: int = 80,
                       caption: str = "",
                       download_timeout: float = 1800.0) -> dict[str, Any]:
        return run_pipeline(
            context, url, platform, chat_id, audio_only=audio_only,
            format_spec=format_spec, compress=compress,
            image_max_width=image_max_width, image_quality=image_quality,
            caption=caption, download_timeout=download_timeout,
        )
