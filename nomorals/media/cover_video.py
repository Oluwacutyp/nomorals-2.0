"""Procedural animated cover / visualizer for a delivered track
(Play & Downloads Phase 2 — doc 7 §5, doc 8 demonstrator 1).

No generation model anywhere in this path.  The pipeline is the
procedural-creation shape the docs prescribe:

    audio analysis → scene/timeline → render → encode with audio
    → inspect → report

* **Analysis**: the track is decoded with ffmpeg to mono PCM and a
  per-frame RMS envelope is measured — the bars in the video ARE
  the track's own loudness, frame by frame.  No invented BPM, no
  fake synchronization: if the audio cannot be decoded, the render
  fails honestly instead of drawing decorative noise.
* **Scene**: the track's embedded artwork when it carries any
  (extracted with ffmpeg), else a procedural backdrop whose palette
  is derived deterministically from the title/artist — plus the
  title/artist set in a real font (DejaVu where present, Pillow's
  bitmap font as the floor).
* **Render/encode**: Pillow draws the frames; ffmpeg encodes them
  with the track's own audio (a short clip) to an mp4.
* **Inspect**: the output is probed (ffprobe via
  :func:`nomorals.media.library.read_metadata`) — a render that
  does not probe as a real video is reported as failed, never
  delivered as one.

The report states what was measured (envelope source, artwork
source, frames, seconds) so callers never over-claim it.
"""

from __future__ import annotations

import array
import colorsys
import hashlib
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..core.logging_setup import get_logger
from .download_kit_failures import record_failure

_log = get_logger(__name__)

__all__ = ["CoverVideoReport", "render_cover_video", "audio_envelope"]

_SAMPLE_RATE = 8000


@dataclass
class CoverVideoReport:
    ok: bool = False
    path: str = ""
    error: str = ""
    frames: int = 0
    seconds: float = 0.0
    envelope: str = ""      # "measured" | "" when it could not be
    artwork: str = ""       # "supplied" | "embedded" | "procedural"


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or ""


def audio_envelope(path: str, *, seconds: float, fps: int,
                   ffmpeg: str = "") -> list[float] | None:
    """Per-frame RMS loudness of the track's first ``seconds``,
    normalized to 0..1 — measured from the decoded audio itself.
    ``None`` when the audio cannot be decoded (the caller must not
    fabricate a visualizer in that case).  Never raises."""
    try:
        ffmpeg = ffmpeg or _ffmpeg()
        if not ffmpeg:
            return None
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path),
             "-t", f"{seconds:.3f}", "-ac", "1", "-ar",
             str(_SAMPLE_RATE), "-f", "s16le", "-"],
            capture_output=True, timeout=120)
        if proc.returncode != 0 or not proc.stdout:
            return None
        samples = array.array("h")
        samples.frombytes(proc.stdout[: len(proc.stdout) // 2 * 2])
        per_frame = max(1, _SAMPLE_RATE // max(1, fps))
        frames = []
        for i in range(0, len(samples), per_frame):
            window = samples[i:i + per_frame]
            if not window:
                break
            rms = math.sqrt(
                sum(s * s for s in window) / len(window)) / 32768.0
            frames.append(min(1.0, rms * 1.6))
        if not frames:
            return None
        peak = max(frames)
        if peak > 0:
            frames = [f / peak for f in frames]
        return frames
    except Exception as exc:  # noqa: BLE001 - analysis is best-effort
        record_failure("cover_video.audio_envelope", exc)
        return None


def _embedded_artwork(audio_path: str, out_png: Path,
                      ffmpeg: str) -> bool:
    """Extract the track's embedded cover art, when it carries any.
    Returns True only when a real, openable image landed."""
    try:
        proc = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", str(audio_path),
             "-map", "0:v:0", "-frames:v", "1", str(out_png)],
            capture_output=True, timeout=60)
        if proc.returncode != 0 or not out_png.is_file() \
                or out_png.stat().st_size < 512:
            return False
        from PIL import Image

        with Image.open(str(out_png)) as img:
            img.verify()
        return True
    except Exception as exc:  # noqa: BLE001
        record_failure("cover_video._embedded_artwork", exc)
        return False


def _font(px: int):
    """The cover title's face through the ONE door: the
    typography service's ``cover`` context policy over the
    stocked registry.  The old private DejaVu path ladder is
    gone; should the registry ever report a total miss, the
    miss is recorded by name and Pillow's default carries
    the title (this surface never fails a render over a
    font)."""
    from PIL import ImageFont

    try:
        from .typography import get_service, pil_font_for
        return pil_font_for(get_service(), "cover", px,
                            weight=700)
    except Exception as exc:  # noqa: BLE001 - recorded miss
        record_failure("cover_video._font", exc)
        return ImageFont.load_default()


def _palette(title: str, artist: str) -> tuple[tuple, tuple, tuple]:
    digest = hashlib.sha256(
        f"{title}\n{artist}".encode("utf-8", "replace")).digest()
    h1 = digest[0] / 255.0
    h2 = (h1 + 0.12 + digest[1] / 2550.0) % 1.0

    def rgb(h: float, s: float, v: float) -> tuple:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    return rgb(h1, 0.55, 0.36), rgb(h2, 0.5, 0.16), rgb(h2, 0.35, 0.95)


def _base_image(size: int, artwork: str, title: str, artist: str):
    """The static backdrop: cover-fit artwork, darkened, or the
    procedural two-stop gradient for this track's identity."""
    from PIL import Image, ImageDraw, ImageEnhance

    top, bottom, _accent = _palette(title, artist)
    if artwork:
        try:
            with Image.open(artwork) as src:
                img = src.convert("RGB")
            scale = max(size / img.width, size / img.height)
            img = img.resize(
                (max(1, int(img.width * scale)),
                 max(1, int(img.height * scale))))
            x = (img.width - size) // 2
            y = (img.height - size) // 2
            img = img.crop((x, y, x + size, y + size))
            return ImageEnhance.Brightness(img).enhance(0.72)
        except Exception as exc:  # noqa: BLE001 - fall to procedural
            record_failure("cover_video._base_image.artwork", exc)
    img = Image.new("RGB", (size, size), bottom)
    draw = ImageDraw.Draw(img)
    for yy in range(size):
        t = yy / max(1, size - 1)
        draw.line([(0, yy), (size, yy)], fill=tuple(
            int(bottom[c] + (top[c] - bottom[c]) * t) for c in range(3)))
    return img


def render_cover_video(audio_path: str, out_path: str, *,
                       title: str = "", artist: str = "",
                       artwork_path: str = "",
                       seconds: float = 12.0, size: int = 720,
                       fps: int = 12) -> CoverVideoReport:
    """Render the animated cover for *audio_path* to *out_path*.

    Honest by construction: no ffmpeg/Pillow → failure report;
    undecodable audio → failure report (no fabricated envelope);
    an output that does not probe as video → failure report."""
    report = CoverVideoReport(path=str(out_path))
    try:
        ffmpeg = _ffmpeg()
        if not ffmpeg:
            report.error = "ffmpeg is not available on this host"
            return report
        try:
            import PIL  # noqa: F401
        except ImportError:
            report.error = "Pillow is not available on this host"
            return report
        if not audio_path or not os.path.isfile(audio_path):
            report.error = f"audio file missing: {audio_path}"
            return report
        seconds = max(1.0, min(30.0, float(seconds)))
        fps = max(4, min(24, int(fps)))
        size = max(128, min(1080, int(size)))
        envelope = audio_envelope(audio_path, seconds=seconds,
                                  fps=fps, ffmpeg=ffmpeg)
        if envelope is None:
            report.error = ("the audio could not be decoded for "
                            "analysis — no envelope, no visualizer")
            return report
        report.envelope = "measured"

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        import time

        workdir = out.parent / f".cover-{time.time_ns()}"
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            art = ""
            if artwork_path and os.path.isfile(artwork_path):
                art = artwork_path
                report.artwork = "supplied"
            elif _embedded_artwork(audio_path, workdir / "art.png",
                                   ffmpeg):
                art = str(workdir / "art.png")
                report.artwork = "embedded"
            else:
                report.artwork = "procedural"
            base = _base_image(size, art, title, artist)
            from PIL import ImageDraw

            _top, _bottom, accent = _palette(title, artist)
            title_font = _font(max(16, size // 13))
            artist_font = _font(max(12, size // 22))
            n_bars = 48
            bar_max = int(size * 0.34)
            bar_w = size / n_bars
            frames = len(envelope)
            for i, _level in enumerate(envelope):
                frame = base.copy()
                draw = ImageDraw.Draw(frame)
                for b in range(n_bars):
                    # Each bar reads the measured envelope at a
                    # small offset around this frame's moment — the
                    # shape is the track's own loudness neighbourhood,
                    # not a synthesized spectrum.
                    j = min(frames - 1, max(0, i + (b % 7) - 3))
                    h = max(2.0, envelope[j] * bar_max)
                    x0 = b * bar_w + 1
                    draw.rectangle(
                        [x0, size - h, x0 + bar_w - 2, size],
                        fill=accent)
                if title:
                    draw.text((size * 0.06, size * 0.06), title[:60],
                              fill=(255, 255, 255), font=title_font)
                if artist:
                    draw.text((size * 0.06, size * 0.06 + size // 11),
                              artist[:60], fill=(230, 230, 230),
                              font=artist_font)
                frame.save(str(workdir / f"frame{i:03d}.png"))
            report.frames = frames
            report.seconds = frames / fps
            proc = subprocess.run(
                [ffmpeg, "-y", "-v", "error",
                 "-framerate", str(fps),
                 "-i", str(workdir / "frame%03d.png"),
                 "-i", str(audio_path),
                 "-t", f"{report.seconds:.3f}",
                 "-map", "0:v", "-map", "1:a",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "128k", "-shortest",
                 str(out)],
                capture_output=True, timeout=300)
            if proc.returncode != 0 or not out.is_file() \
                    or out.stat().st_size == 0:
                report.error = (
                    "ffmpeg encode failed: "
                    + proc.stderr.decode("utf-8", "replace")[-300:])
                return report
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        # Inspect before claiming: the artifact must probe as video.
        from .library import read_metadata

        duration = float(read_metadata(str(out)).get("duration") or 0)
        if duration <= 0:
            report.error = "the rendered file does not probe as a video"
            report.ok = False
            return report
        report.ok = True
        return report
    except Exception as exc:  # noqa: BLE001 - a render reports, never raises
        record_failure("cover_video.render_cover_video", exc)
        report.ok = False
        report.error = f"cover render failed: {exc}"
        return report
