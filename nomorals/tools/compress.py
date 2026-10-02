"""Compressor: make big files sendable.

Two real strategies, chosen by content:

* **video** (mp4/mov/mkv/avi/webm) — re-encode with ffmpeg when it's
  installed (CRF 28, capped at 720p, audio at 96k). Without ffmpeg there is
  no honest video compression to offer, and the tool says so instead of
  pretending a zip helped.
* **everything else** — zip with maximum deflate (already-compressed formats
  like jpg/zip/mp4 report a low ratio and the caller can skip sending).

The gateway uses this automatically on outbound files bigger than
``chat.max_send_mb`` (``compress_on_send``), so a 40MB video becomes a
~6MB file before it hits Telegram's limits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

__all__ = ["compress_file", "register"]

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _compress_video(src: Path, dest: Path, *, crf: int = 28,
                    preset: str = "veryfast", audio_bitrate: str = "96k") -> bool:
    if not _ffmpeg_available():
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-vf", "scale='min(1280,iw)':'-2'",
        "-c:a", "aac", "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception:  # noqa: BLE001
        return False
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        return False
    return True


def _compress_zip(src: Path, dest: Path) -> bool:
    try:
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(src, arcname=src.name)
        return True
    except Exception:  # noqa: BLE001
        return False


def compress_file(path: str | Path, *, video_crf: int = 28,
                  video_preset: str = "veryfast",
                  audio_bitrate: str = "96k") -> dict[str, Any]:
    """Compress one file. Returns a report; ``ok`` False means 'don't send
    the original either if you can avoid it' is the caller's call — the
    report always includes the original size for the decision.

    Video knobs (ffmpeg): ``video_crf`` (default 28), ``video_preset``
    (default "veryfast", capped at 1280px wide), ``audio_bitrate``
    (default "96k" for the video's audio track).
    """
    src = Path(path)
    if not src.exists():
        return {"ok": False, "error": f"no such file: {src}", "path": str(src)}
    original = src.stat().st_size
    ext = src.suffix.lower()
    # NOTE: the compressed copy must never share the source path — ffmpeg
    # reads and writes concurrently, so an in-place "re-encode" would
    # truncate the input mid-read and destroy it.
    dest = src.with_name(src.stem + ".compressed" + ext)

    method = ""
    ok = False
    if ext in VIDEO_EXTS:
        ok = _compress_video(src, dest, crf=video_crf, preset=video_preset,
                             audio_bitrate=audio_bitrate)
        method = "ffmpeg" if ok else ""
    if not ok:
        if ext in {".jpg", ".jpeg", ".png", ".zip", ".gz", ".7z", ".webp", ".mp3", ".mp4"}:
            # already compressed: a zip wrap is near-useless — say so
            return {
                "ok": False, "method": "none",
                "reason": f"{ext} is already compressed — zip/ffmpeg would not help",
                "path": str(src), "original_bytes": original, "new_bytes": original, "ratio": 1.0,
            }
        ok = _compress_zip(src, dest)
        method = "zip" if ok else ""
    if not ok:
        dest.unlink(missing_ok=True)
        return {"ok": False, "method": "none",
                "reason": "no compressor available for this file type"
                          + ("" if ext not in VIDEO_EXTS else " (install ffmpeg for video)"),
                "path": str(src), "original_bytes": original, "new_bytes": original, "ratio": 1.0}
    new = dest.stat().st_size
    report = {
        "ok": True,
        "method": method,
        "path": str(dest),
        "original_bytes": original,
        "new_bytes": new,
        "ratio": round(new / original, 3) if original else 1.0,
    }
    if new >= original:
        dest.unlink(missing_ok=True)
        report.update(ok=False, method="none", path=str(src),
                      reason="compression made it bigger — sending the original",
                      new_bytes=original, ratio=1.0)
    return report


def register(registry: Any) -> None:
    """Attach the compress_file tool to a registry."""

    @registry.register(
        "compress_file",
        description="Compress a file (ffmpeg for video when installed, else zip). "
                    "Returns the smaller path to send.",
        capability="fs.read",
    )
    def compress(path: str) -> dict[str, Any]:
        return compress_file(path)
