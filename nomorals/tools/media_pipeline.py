"""Download → compress → send pipeline: one coherent media flow.

Three disconnected tools already exist — ``media_download`` (fetch),
``compress_file`` (ffmpeg/zip), and ``file_send`` (gateway) — but using
them by hand means the agent juggles three calls, three paths, and three
failure modes. This module wires them into a single pipeline with a
per-stage status report:

    download (fetch the URL) → compress (shrink when it helps) → send (to chat)

Compression strategy per file kind (defaults; kwargs override env, env
overrides these):

* **image** (jpg/png/webp/bmp/tiff) — downscale to ``image_max_width``
  (default 1600, ``NM_MEDIA_IMAGE_MAX_WIDTH``) and re-encode with Pillow
  (JPEG quality ``image_quality``, default 80,
  ``NM_MEDIA_IMAGE_QUALITY``; PNG optimized). Without Pillow the stage
  reports honestly that it skipped.
* **video** — delegated to :func:`nomorals.tools.compress.compress_file`,
  which re-encodes with ffmpeg when installed (CRF ``video_crf``, default
  28 via ``NM_MEDIA_VIDEO_CRF``; preset ``video_preset``, default
  "veryfast"; capped at 1280px wide; audio at 96k) and says so when it
  isn't.
* **audio** (mp3/m4a/aac/ogg/opus/flac/wav) — ffmpeg re-encode to mp3 at
  ``audio_bitrate`` (default "128k", ``NM_MEDIA_AUDIO_BITRATE``), streamed
  so the whole file never sits in memory.
* **everything else** — no compression is attempted (zipping a PDF just to
  send it would be hostile); the stage reports the skip and the original
  goes out.

Size cap: files over ``max_mb`` MB (default 200, ``NM_MEDIA_MAX_MB``) get
one aggressive shrink attempt when the kind supports it, then the pipeline
refuses outright — naming the cap and the actual size — rather than
sending a multi-GB file or loading a huge file into memory and OOMing.

Failure policy is fail-fast, never silent: a download failure raises, a
send failure raises, and every raised error names its stage
(download|compress|send|zip|unzip). A compress miss is recorded in the
stage report with its reason instead of being swallowed. Compressed
intermediates are deleted after a successful send; the original download
is kept.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from ..core.errors import MediaError, ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .filesystem import safe_path

_log = get_logger(__name__)

__all__ = ["run_pipeline", "compress_image_for_send", "compress_audio_for_send",
           "register"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

#: what goes through the ffmpeg path in tools.compress
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

#: audio goes through ffmpeg re-encode (streamed, never fully in memory)
AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wav"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


#: default per-kind compress settings (env-overridable)
DEFAULT_MAX_MB = 200.0          # NM_MEDIA_MAX_MB — refuse inputs over this
DEFAULT_IMAGE_MAX_WIDTH = 1600  # NM_MEDIA_IMAGE_MAX_WIDTH
DEFAULT_IMAGE_QUALITY = 80      # NM_MEDIA_IMAGE_QUALITY (JPEG)
DEFAULT_VIDEO_CRF = 28          # NM_MEDIA_VIDEO_CRF
DEFAULT_VIDEO_PRESET = "veryfast"  # NM_MEDIA_VIDEO_PRESET
DEFAULT_AUDIO_BITRATE = "128k"  # NM_MEDIA_AUDIO_BITRATE (ffmpeg -b:a)


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


def compress_audio_for_send(
    path: str | Path,
    *,
    bitrate: str = DEFAULT_AUDIO_BITRATE,
) -> dict[str, Any]:
    """Re-encode audio to ``bitrate`` mp3 so it is cheap to send.

    ffmpeg streams the file — it is never loaded fully into memory, which
    is what makes this safe on low-memory hosts. Returns a compress-style
    report; ``ok`` False carries a ``reason`` (ffmpeg missing, re-encode
    failed, or the result was not smaller).
    """
    import shutil

    src = Path(path)
    if not src.is_file():
        return {"ok": False, "method": "none",
                "reason": f"no such file: {src}", "path": str(src)}
    original = src.stat().st_size
    if shutil.which("ffmpeg") is None:
        return {
            "ok": False, "method": "none",
            "reason": "ffmpeg not installed — cannot re-encode audio",
            "path": str(src), "original_bytes": original, "new_bytes": original,
            "ratio": 1.0,
        }
    dest = src.with_name(f"{src.stem}.send.mp3")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-c:a", "libmp3lame", "-b:a", bitrate, str(dest)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        return {"ok": False, "method": "none",
                "reason": f"audio re-encode failed: {exc}",
                "path": str(src), "original_bytes": original,
                "new_bytes": original, "ratio": 1.0}
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        detail = (proc.stderr or "").strip()[-300:]
        return {"ok": False, "method": "none",
                "reason": f"ffmpeg audio re-encode failed"
                          f"{(': ' + detail) if detail else ''}",
                "path": str(src), "original_bytes": original,
                "new_bytes": original, "ratio": 1.0}
    new = dest.stat().st_size
    if new >= original:
        dest.unlink(missing_ok=True)
        return {
            "ok": False, "method": "none",
            "reason": f"re-encoded audio ({new} bytes) is not smaller than "
                      f"the original ({original} bytes) — sending the original",
            "path": str(src), "original_bytes": original,
            "new_bytes": original, "ratio": 1.0,
        }
    return {
        "ok": True, "method": f"ffmpeg-audio ({bitrate})",
        "path": str(dest),
        "original_bytes": original, "new_bytes": new,
        "ratio": round(new / original, 3) if original else 1.0,
    }


def _human_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GiB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n} bytes"


def _compress_stage(src: Path, *, image_max_width: int, image_quality: int,
                    video_crf: int, video_preset: str, audio_bitrate: str,
                    enabled: bool) -> dict[str, Any]:
    """Pick the compression strategy by file kind. Honest no-ops included.

    Defaults per kind (all overridable via run_pipeline kwargs / env):

    * image — Pillow: downscale to ``image_max_width`` px (default 1600,
      ``NM_MEDIA_IMAGE_MAX_WIDTH``), re-encode JPEG at ``image_quality``
      (default 80, ``NM_MEDIA_IMAGE_QUALITY``), PNG optimized.
    * video — ffmpeg via :func:`nomorals.tools.compress.compress_file`:
      CRF ``video_crf`` (default 28, ``NM_MEDIA_VIDEO_CRF``), preset
      ``video_preset`` (default "veryfast"), capped at 1280px wide,
      audio at 96k.
    * audio — ffmpeg re-encode to mp3 at ``audio_bitrate`` (default "128k",
      ``NM_MEDIA_AUDIO_BITRATE``), streamed, never fully in memory.
    """
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

        report = compress_file(src, video_crf=video_crf,
                               video_preset=video_preset)
        # compress_file is already honest about missing ffmpeg; normalize keys
        report.setdefault("path", str(src))
        return report
    if ext in AUDIO_EXTS:
        return compress_audio_for_send(src, bitrate=audio_bitrate)
    size = src.stat().st_size
    return {
        "ok": False, "method": "none", "path": str(src),
        "reason": f"{ext or 'unknown type'} is not an image, video, or audio "
                  "file — no compression attempted; the original will be sent",
        "original_bytes": size, "new_bytes": size, "ratio": 1.0,
    }


def _size_gate(
    path: Path,
    *,
    cap_bytes: int,
    cap_mb: float,
    image_max_width: int,
    image_quality: int,
    video_crf: int,
    video_preset: str,
    audio_bitrate: str,
) -> tuple[Path, dict[str, Any]]:
    """Enforce the pipeline input-size cap on ``path``.

    Files at or under the cap pass through untouched. Over-cap files get
    exactly one aggressive shrink attempt when their kind supports it
    (image → downscale, video/audio → ffmpeg re-encode, both streamed or
    bounded so nothing huge is held in memory). If the file is still over
    the cap afterwards, this raises :class:`MediaError` naming the cap and
    the actual size — it never sends a multi-GB file and never OOMs trying.

    Returns ``(effective_path, gate_report)``.
    """
    size = path.stat().st_size
    gate: dict[str, Any] = {
        "cap_mb": cap_mb, "cap_bytes": cap_bytes,
        "input_bytes": size, "shrunk": False,
    }
    if size <= cap_bytes:
        return path, gate
    ext = path.suffix.lower()
    _log.warning("pipeline size gate: %s is %s, over the %s MB cap "
                 "(NM_MEDIA_MAX_MB) — attempting one shrink",
                 path.name, _human_bytes(size), cap_mb)
    report: dict[str, Any] | None = None
    try:
        if ext in IMAGE_EXTS:
            # aggressive single pass; Pillow decode of a very large image
            # can itself blow memory, so MemoryError is a refusal, not a crash
            report = compress_image_for_send(
                path, max_width=min(image_max_width, 800),
                quality=min(image_quality, 70))
        elif ext in VIDEO_EXTS:
            from .compress import compress_file
            report = compress_file(path, video_crf=video_crf,
                                   video_preset=video_preset)
        elif ext in AUDIO_EXTS:
            report = compress_audio_for_send(path, bitrate=audio_bitrate)
    except MemoryError as exc:
        raise MediaError(
            f"pipeline size gate: {path.name} is {_human_bytes(size)} — over "
            f"the {cap_mb} MB cap (NM_MEDIA_MAX_MB), and it is too large to "
            f"downscale in memory ({exc}); refusing to send") from exc
    except Exception as exc:  # noqa: BLE001 - shrink attempt failed: refuse
        report = {"ok": False, "reason": str(exc)}
    if report and report.get("ok"):
        new_path = Path(report["path"])
        new_size = new_path.stat().st_size
        gate.update(shrunk=True, method=report.get("method", ""),
                    shrunk_bytes=new_size)
        if new_size <= cap_bytes:
            _log.info("pipeline size gate: shrunk %s → %s",
                      _human_bytes(size), _human_bytes(new_size))
            return new_path, gate
        # shrink helped but not enough — clean up the intermediate
        new_path.unlink(missing_ok=True)
        size = new_size
    raise MediaError(
        f"pipeline size gate: {path.name} is {_human_bytes(size)} — over the "
        f"{cap_mb} MB cap (NM_MEDIA_MAX_MB); "
        f"{'even after one compression pass ' if gate['shrunk'] else ''}"
        f"refusing to send. Re-download a smaller format or raise the cap "
        f"with NM_MEDIA_MAX_MB.")


def run_pipeline(
    context: Any,
    url: str,
    platform: str,
    chat_id: str,
    *,
    audio_only: bool = False,
    format_spec: str = "bestvideo*+bestaudio/best",
    compress: bool = True,
    image_max_width: int = _env_int("NM_MEDIA_IMAGE_MAX_WIDTH",
                                   DEFAULT_IMAGE_MAX_WIDTH),
    image_quality: int = _env_int("NM_MEDIA_IMAGE_QUALITY",
                                  DEFAULT_IMAGE_QUALITY),
    video_crf: int = _env_int("NM_MEDIA_VIDEO_CRF", DEFAULT_VIDEO_CRF),
    video_preset: str = os.environ.get("NM_MEDIA_VIDEO_PRESET",
                                       DEFAULT_VIDEO_PRESET),
    audio_bitrate: str = os.environ.get("NM_MEDIA_AUDIO_BITRATE",
                                        DEFAULT_AUDIO_BITRATE),
    max_mb: float | None = None,
    caption: str = "",
    download_timeout: float = 1800.0,
    downloader: Callable[..., dict[str, Any]] | None = None,
    sender: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch ``url``, optionally compress, then send to the active chat.

    Returns a report with one entry per stage — ``download``, ``compress``,
    ``send`` — each carrying ``ok`` and its details. The pipeline is
    fail-fast: a failed download raises :class:`MediaError`, a failed send
    raises :class:`ToolError`, and every raised error names its stage.
    A compress miss never fails the pipeline; it is recorded with its
    reason and the original file goes out.

    Per-kind compress defaults (kwargs override env, env overrides these):

    * image — downscale to ``image_max_width`` px (default 1600,
      ``NM_MEDIA_IMAGE_MAX_WIDTH``), JPEG quality ``image_quality``
      (default 80, ``NM_MEDIA_IMAGE_QUALITY``);
    * video — ffmpeg CRF ``video_crf`` (default 28, ``NM_MEDIA_VIDEO_CRF``),
      preset ``video_preset`` (default "veryfast",
      ``NM_MEDIA_VIDEO_PRESET``), capped at 1280px wide, audio at 96k;
    * audio — ffmpeg re-encode to mp3 at ``audio_bitrate`` (default "128k",
      ``NM_MEDIA_AUDIO_BITRATE``), streamed so it never loads the whole
      file into memory.

    Size cap: downloads over ``max_mb`` MB (default 200,
    ``NM_MEDIA_MAX_MB``) get one aggressive shrink attempt when the kind
    supports it (images/video/audio); if the result is still over the cap
    the pipeline refuses with a :class:`MediaError` naming the cap and the
    actual size instead of sending a multi-GB file or OOMing on it.

    Temporary compressed intermediates are removed after a successful send
    (the original download is kept); if the send fails they are left in
    place and named in the error so a retry can reuse them.

    ``downloader`` / ``sender`` are injectable for tests; production code
    leaves them ``None`` and the real tools are used.
    """
    if not (url or "").strip():
        raise MediaError("pipeline needs a URL")
    if not (platform or "").strip():
        raise ToolError("pipeline needs a platform (telegram | whatsapp | …)")
    if not (chat_id or "").strip():
        raise ToolError("pipeline needs a chat_id")
    if max_mb is None:
        max_mb = _env_float("NM_MEDIA_MAX_MB", DEFAULT_MAX_MB)
    if max_mb <= 0:
        raise ToolError(f"pipeline max_mb must be positive, got {max_mb}")

    started_all = time.perf_counter()
    stages: list[dict[str, Any]] = []
    cap_bytes = int(max_mb * 1024 * 1024)

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

    # ── size gate: one shrink attempt for over-cap files, else refuse ──
    intermediates: list[Path] = []
    gate_path, gate = _size_gate(
        path, cap_bytes=cap_bytes, cap_mb=max_mb,
        image_max_width=image_max_width, image_quality=image_quality,
        video_crf=video_crf, video_preset=video_preset,
        audio_bitrate=audio_bitrate)
    if gate["shrunk"]:
        stage["size_gate"] = {
            "input_bytes": gate["input_bytes"],
            "shrunk_bytes": gate["shrunk_bytes"],
            "method": gate.get("method", ""),
            "cap_mb": max_mb,
        }
        intermediates.append(gate_path)
        path = gate_path
    _log.info("pipeline downloaded %s (%d bytes) via %s",
              path.name, path.stat().st_size, downloaded.get("extractor", "?"))
    stages.append(stage)
    download_path = path  # the file the download produced, kept on disk

    # ── stage 2: compress (best-effort, reported honestly) ─────────────
    stage_start = time.perf_counter()
    try:
        report = _compress_stage(
            path, image_max_width=image_max_width,
            image_quality=image_quality, video_crf=video_crf,
            video_preset=video_preset, audio_bitrate=audio_bitrate,
            enabled=compress)
    except Exception as exc:  # noqa: BLE001 - name the stage, never bare
        raise MediaError(f"pipeline compress failed: {exc}") from exc
    stage = _stage("compress", stage_start)
    stage.update(report)
    stages.append(stage)
    send_path = report.get("path") or str(path)
    if report.get("ok"):
        candidate = Path(send_path)
        if candidate != path and candidate not in intermediates:
            intermediates.append(candidate)
        _log.info("pipeline compressed %s → %s (ratio %s)",
                  path.name, Path(send_path).name, report.get("ratio"))
    else:
        _log.info("pipeline compress skipped: %s", report.get("reason"))

    # ── stage 3: send ─────────────────────────────────────────────────
    if Path(send_path).stat().st_size > cap_bytes:
        # unreachable after the size gate, but never send a huge file silently
        raise ToolError(
            f"pipeline send refused: {Path(send_path).name} is "
            f"{_human_bytes(Path(send_path).stat().st_size)} — over the "
            f"{max_mb} MB cap (NM_MEDIA_MAX_MB)")
    def _kept_note() -> str:
        if not intermediates:
            return ""
        return ("; compressed intermediate(s) kept at "
                + ", ".join(str(p) for p in intermediates) + " for a retry")

    stage_start = time.perf_counter()
    send = sender or _send_file
    try:
        sent = send(context, platform, chat_id, send_path, caption=caption)
    except TypeError:
        sent = send(context, platform, chat_id, send_path)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"pipeline send failed: {exc}{_kept_note()}") from exc
    if isinstance(sent, dict) and sent.get("sent") is False:
        raise ToolError(f"pipeline send reported failure: {sent}{_kept_note()}")
    stage = _stage("send", stage_start)
    stage.update(ok=True, platform=platform, chat=chat_id,
                 sent_path=send_path,
                 message_id=(sent or {}).get("message_id", "")
                 if isinstance(sent, dict) else "")
    stages.append(stage)

    report_out = {
        "ok": True,
        "stages": stages,
        "downloaded_path": str(download_path),
        "sent_path": send_path,
        "bytes_sent": Path(send_path).stat().st_size,
        "compressed": bool(report.get("ok")),
        "size_gate_shrunk": bool(gate["shrunk"]),
        "seconds": round(time.perf_counter() - started_all, 2),
    }

    # ── cleanup: compressed copies are regenerable; the download stays ──
    for intermediate in intermediates:
        try:
            intermediate.unlink()
            _log.debug("pipeline removed intermediate %s", intermediate.name)
        except OSError as exc:
            _log.warning("pipeline could not remove intermediate %s: %s",
                         intermediate, exc)

    return report_out


def register(registry: Any) -> None:
    """Attach the pipeline tool to a registry."""
    context = registry.context

    @registry.register(
        "media_pipeline",
        description=(
            "Download → compress → send in one call: fetch a file/URL "
            "(media_download), shrink it when that helps, then send the "
            "result to a chat (file_send). Per-kind compress defaults: "
            "images — Pillow downscale to 1600px (NM_MEDIA_IMAGE_MAX_WIDTH), "
            "JPEG quality 80 (NM_MEDIA_IMAGE_QUALITY); video — ffmpeg CRF 28 "
            "(NM_MEDIA_VIDEO_CRF), preset veryfast, 1280px cap, audio 96k; "
            "audio — ffmpeg re-encode to mp3 at 128k (NM_MEDIA_AUDIO_BITRATE), "
            "streamed. Files over 200 MB (NM_MEDIA_MAX_MB) are refused "
            "(after one shrink attempt for image/video/audio kinds). "
            "Each stage reports honestly when tooling is missing; every "
            "failure names its stage. Returns per-stage status."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "url": "str — media URL or direct file link",
            "platform": "str — telegram | whatsapp | discord | console | …",
            "chat_id": "str — the target chat id",
            "audio_only": "bool (optional, False) — extract audio only",
            "format_spec": "str (optional) — yt-dlp format selector",
            "compress": "bool (optional, True) — shrink before sending",
            "image_max_width": "int (optional, 1600; NM_MEDIA_IMAGE_MAX_WIDTH)",
            "image_quality": "int (optional, 80; NM_MEDIA_IMAGE_QUALITY) — JPEG quality",
            "video_crf": "int (optional, 28; NM_MEDIA_VIDEO_CRF) — ffmpeg CRF",
            "video_preset": "str (optional, 'veryfast'; NM_MEDIA_VIDEO_PRESET)",
            "audio_bitrate": "str (optional, '128k'; NM_MEDIA_AUDIO_BITRATE) — ffmpeg -b:a",
            "max_mb": "float (optional, 200; NM_MEDIA_MAX_MB) — refuse inputs over this",
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
                       video_crf: int = 28,
                       video_preset: str = "veryfast",
                       audio_bitrate: str = "128k",
                       max_mb: float | None = None,
                       caption: str = "",
                       download_timeout: float = 1800.0) -> dict[str, Any]:
        return run_pipeline(
            context, url, platform, chat_id, audio_only=audio_only,
            format_spec=format_spec, compress=compress,
            image_max_width=image_max_width, image_quality=image_quality,
            video_crf=video_crf, video_preset=video_preset,
            audio_bitrate=audio_bitrate, max_mb=max_mb,
            caption=caption, download_timeout=download_timeout,
        )
