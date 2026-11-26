"""Clip validation — ported from the source's ``trendclip/validate.py``.

A rendered clip passes only when ffprobe confirms the structure
(1080x1920, h264, yuv420p, aac, sane fps and duration) AND the content
checks find no black frames, no frozen stretches and no silence. The
runner is injectable (``runner=``) so tests run offline with fake ffprobe
output.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = [
    "add_content_quality",
    "inspect_media_quality",
    "probe_video",
    "validate_clip",
    "write_validation",
]

EXPECTED_WIDTH = 1080
EXPECTED_HEIGHT = 1920


def _run(cmd: list[str], runner: Callable[..., Any] | None,
         timeout: int = 120) -> Any:
    run = runner or subprocess.run
    return run(cmd, capture_output=True, text=True, timeout=timeout)


def probe_video(path: str | Path, *,
                runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Structural probe via ffprobe: codec, size, fps, duration, audio."""
    proc = _run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(path)], runner)
    if getattr(proc, "returncode", 1) != 0:
        return {"ok": False,
                "error": "ffprobe failed: "
                + str(getattr(proc, "stderr", "") or "")[:200]}
    try:
        data = json.loads(getattr(proc, "stdout", "") or "{}")
    except ValueError:
        return {"ok": False, "error": "ffprobe returned unparseable JSON"}
    video = next((s for s in data.get("streams") or []
                  if s.get("codec_type") == "video"), {})
    audio = next((s for s in data.get("streams") or []
                  if s.get("codec_type") == "audio"), {})
    fps_raw = str(video.get("avg_frame_rate") or "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "ok": True,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "codec": str(video.get("codec_name") or ""),
        "pix_fmt": str(video.get("pix_fmt") or ""),
        "fps": round(fps, 2),
        "duration": float((data.get("format") or {}).get("duration") or 0.0),
        "has_audio": bool(audio),
        "audio_codec": str(audio.get("codec_name") or ""),
    }


def _detect(cmd: list[str], runner: Callable[..., Any] | None,
            label: str) -> dict[str, Any]:
    """Run one ffmpeg detect filter and count its hits."""
    proc = _run(cmd, runner, timeout=300)
    text = str(getattr(proc, "stderr", "") or "")
    hits = text.count(label)
    return {"hits": hits, "detail": text[-500:] if hits else ""}


def inspect_media_quality(path: str | Path, *,
                          runner: Callable[..., Any] | None = None
                          ) -> dict[str, Any]:
    """Content checks: blackdetect, freezedetect, silencedetect."""
    black = _detect([
        "ffmpeg", "-i", str(path), "-vf",
        "blackdetect=d=0.5:pic_th=0.98:pix_th=0.10",
        "-an", "-f", "null", "-"], runner, "black_start")
    frozen = _detect([
        "ffmpeg", "-i", str(path), "-vf",
        "freezedetect=n=0.003:d=1.0",
        "-an", "-f", "null", "-"], runner, "freeze_start")
    silence = _detect([
        "ffmpeg", "-i", str(path), "-af",
        "silencedetect=noise=-50dB:d=1.0",
        "-vn", "-f", "null", "-"], runner, "silence_start")
    return {
        "black_frames": black["hits"],
        "frozen_segments": frozen["hits"],
        "silent_segments": silence["hits"],
    }


def add_content_quality(report: dict[str, Any]) -> dict[str, Any]:
    """Attach the pass/fail verdict to a probe+quality report."""
    problems: list[str] = []
    probe = report.get("probe") or {}
    if probe.get("width") != EXPECTED_WIDTH \
            or probe.get("height") != EXPECTED_HEIGHT:
        problems.append(
            f"expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, got "
            f"{probe.get('width')}x{probe.get('height')}")
    if (probe.get("codec") or "") != "h264":
        problems.append(f"expected h264, got {probe.get('codec')!r}")
    if (probe.get("pix_fmt") or "") != "yuv420p":
        problems.append(f"expected yuv420p, got {probe.get('pix_fmt')!r}")
    if not probe.get("has_audio"):
        problems.append("no audio stream")
    quality = report.get("quality") or {}
    if int(quality.get("black_frames") or 0) > 0:
        problems.append(
            f"{quality['black_frames']} black-frame stretch(es)")
    if int(quality.get("frozen_segments") or 0) > 0:
        problems.append(
            f"{quality['frozen_segments']} frozen segment(s)")
    if int(quality.get("silent_segments") or 0) > 0:
        problems.append(
            f"{quality['silent_segments']} silent segment(s)")
    report["problems"] = problems
    report["passed"] = not problems
    return report


def validate_clip(path: str | Path, *,
                  runner: Callable[..., Any] | None = None
                  ) -> dict[str, Any]:
    """Probe + content-check one rendered clip; returns the full report."""
    probe = probe_video(path, runner=runner)
    if not probe.get("ok"):
        return {"ok": False, "passed": False,
                "problems": [str(probe.get("error") or "probe failed")],
                "probe": probe, "quality": {}}
    quality = inspect_media_quality(path, runner=runner)
    return add_content_quality(
        {"ok": True, "probe": probe, "quality": quality})


def write_validation(report: dict[str, Any], path: str | Path) -> Path:
    """Persist a validation report as JSON next to the clip."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
