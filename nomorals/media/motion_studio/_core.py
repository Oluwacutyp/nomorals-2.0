"""Shared internals for Devon's motion studio.

Profile-aware defaults, easing curves, font discovery, and the numpy →
ffmpeg frame-pipe encoder that every renderer in this package uses.

Everything here is CPU-only by design: numpy + PIL + ffmpeg run on
termux, laptop and workstation alike.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.logging_setup import get_logger
from ...core.profiles import get_profile_kind
from ...media_edit.videos import MediaEditError, ffmpeg_path, run_ffmpeg, video_probe

_log = get_logger(__name__)

__all__ = [
    "MotionStudioError",
    "PROFILE_DEFAULTS",
    "profile_defaults",
    "EASINGS",
    "ease",
    "find_font",
    "text_size",
    "workdir",
    "new_render_path",
    "record_ledger",
    "read_ledger",
    "encode_frames",
    "render_sequence",
    "decode_audio_mono",
    "probe_duration",
    "rounded",
]

MEDIA_ROOT = Path.home() / "workspace" / "media" / "video"
LEDGER_PATH = MEDIA_ROOT / "ledger.json"


class MotionStudioError(MediaEditError):
    """Raised for every motion-studio failure (missing file, ffmpeg error…)."""


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

#: (width, height, fps) defaults per runtime profile. termux stays small on
#: purpose — 720p on a phone SoC is already heavy; laptop/workstation get
#: real canvases. Format overrides (9:16 etc.) rescale from these.
PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "termux": {
        "size": (480, 854),      # 9:16 default — shorts-first on the phone
        "landscape": (854, 480),
        "fps": 24,
        "crf": 23,
        "preset": "veryfast",
        "particles": 220,
        "supersample": 1,
    },
    "laptop": {
        "size": (1080, 1920),
        "landscape": (1920, 1080),
        "fps": 30,
        "crf": 20,
        "preset": "veryfast",
        "particles": 700,
        "supersample": 2,
    },
    "workstation": {
        "size": (1080, 1920),
        "landscape": (1920, 1080),
        "fps": 30,
        "crf": 18,
        "preset": "medium",
        "particles": 1500,
        "supersample": 2,
    },
}


def profile_defaults(kind: str = "") -> dict[str, object]:
    """Defaults for the current (or given) profile; always returns a copy."""
    key = (kind or "").strip().lower() or get_profile_kind()
    base = PROFILE_DEFAULTS.get(key, PROFILE_DEFAULTS["laptop"])
    return dict(base)


# ---------------------------------------------------------------------------
# easing
# ---------------------------------------------------------------------------

def _linear(t: float) -> float:
    return t


def _smooth(t: float) -> float:  # smoothstep — the default "cinematic" ease
    return t * t * (3.0 - 2.0 * t)


def _ease_in(t: float) -> float:
    return t * t * t


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _ease_in_out(t: float) -> float:
    return 4 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def _sine(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


EASINGS: dict[str, Callable[[float], float]] = {
    "linear": _linear,
    "smooth": _smooth,          # default
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
    "sine": _sine,
}


def ease(name: str, t: float) -> float:
    """Apply easing curve ``name`` to ``t`` in [0, 1].

    Consolidation (Phase 8A.8): routes through the directed easing
    library (motion_score.EASE_FUNCS) — the single source of truth —
    with the local table as fallback if directed is unavailable.
    """
    key = (name or "smooth").lower()
    try:
        from ..directed.motion_score import ease_value
        return ease_value(key, t)
    except Exception:  # noqa: BLE001 - stay importable no matter what
        fn = EASINGS.get(key, _smooth)
        t = max(0.0, min(1.0, t))
        return fn(t)


# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

#: bundled-candidate font names, best first
_FONT_CANDIDATES = (
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Bold.ttf",
    "LiberationSans-Regular.ttf",
)


def find_font(size: int, name: str = "") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Resolve a truetype font; falls back through candidates, then PIL default."""
    key = (name or "", size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = ([name] if name else []) + list(_FONT_CANDIDATES)
    for cand in candidates:
        try:
            font = ImageFont.truetype(cand, size)
            _FONT_CACHE[key] = font
            return font
        except Exception:  # noqa: BLE001 - try next candidate
            continue
    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def text_size(draw: ImageDraw.ImageDraw, text: str,
              font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> tuple[int, int]:
    """Width/height of ``text`` — works on every Pillow ≥ 8."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:  # noqa: BLE001
        return draw.textlength(text, font=font), getattr(font, "size", 24)


# ---------------------------------------------------------------------------
# paths + ledger
# ---------------------------------------------------------------------------

def workdir() -> Path:
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    return MEDIA_ROOT


def new_render_path(prefix: str, ext: str = "mp4") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return workdir() / f"{prefix}-{stamp}.mp4" if ext == "mp4" else \
        workdir() / f"{prefix}-{stamp}.{ext}"


def record_ledger(entry: dict) -> None:
    """Append a render record (best-effort; never breaks a render)."""
    try:
        ledger = read_ledger()
        ledger.append(entry)
        LEDGER_PATH.write_text(json.dumps(ledger[-500:], indent=1))
    except Exception:  # noqa: BLE001
        _log.debug("ledger write failed", exc_info=True)


def read_ledger() -> list[dict]:
    try:
        if LEDGER_PATH.exists():
            data = json.loads(LEDGER_PATH.read_text())
            return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        pass
    return []


# ---------------------------------------------------------------------------
# ffmpeg frame pipe
# ---------------------------------------------------------------------------

def encode_frames(frames: Iterable[np.ndarray], out_path: str | os.PathLike,
                  width: int, height: int, fps: float,
                  *, audio: str | os.PathLike | None = None,
                  crf: int = 20, preset: str = "veryfast",
                  pix_fmt: str = "yuv420p") -> Path:
    """Encode an iterable of H×W×3 uint8 frames to H.264 via a raw pipe.

    Returns the output path. Raises :class:`MotionStudioError` on failure.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_path(), "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", f"{fps:.3f}",
           "-i", "-"]
    if audio:
        cmd += ["-i", str(audio), "-c:a", "aac", "-b:a", "160k",
                "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", pix_fmt, "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        assert proc.stderr is not None
        n = 0
        for frame in frames:
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.shape != (height, width, 3):
                arr = np.asarray(
                    Image.fromarray(arr).resize((width, height)).convert("RGB"))
            try:
                proc.stdin.write(arr.tobytes())
            except BrokenPipeError:
                # ffmpeg died mid-stream — collect its stderr below
                break
            n += 1
        proc.stdin.close()
        err = proc.stderr.read()
        proc.wait(timeout=600)
        if proc.returncode != 0:
            raise MotionStudioError(
                f"ffmpeg encode failed ({n} frames): "
                f"{err.decode(errors='replace')[-800:]}")
        if n == 0:
            raise MotionStudioError("no frames rendered — nothing to encode")
    except MotionStudioError:
        raise
    except Exception as exc:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        raise MotionStudioError(f"frame pipe failed: {exc}") from exc
    return out


def render_sequence(n_frames: int, width: int, height: int, fps: float,
                    out_path: str | os.PathLike,
                    frame_fn: Callable[[int, float], np.ndarray],
                    **encode_kw) -> Path:
    """Render ``frame_fn(i, t)`` for ``n_frames`` and encode. ``t`` is seconds."""
    def _gen():
        for i in range(n_frames):
            yield frame_fn(i, i / fps)
    return encode_frames(_gen(), out_path, width, height, fps, **encode_kw)


# ---------------------------------------------------------------------------
# audio helpers
# ---------------------------------------------------------------------------

def decode_audio_mono(path: str | os.PathLike, sr: int = 22050) -> np.ndarray:
    """Any audio/video file → mono float32 waveform at ``sr`` Hz."""
    import wave as _wave
    p = Path(path)
    if not p.exists():
        raise MotionStudioError(f"no such audio file: {path}")
    fd, tmp = tempfile.mkstemp(prefix="motion-audio-", suffix=".wav")
    os.close(fd)
    try:
        run_ffmpeg(["-i", str(p), "-vn", "-ar", str(sr), "-ac", "1",
                    "-c:a", "pcm_s16le", tmp], timeout=180.0)
        with _wave.open(tmp, "rb") as w:
            raw = w.readframes(w.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return data
    except MotionStudioError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise MotionStudioError(f"could not decode audio {path}: {exc}") from exc
    finally:
        try:
            os.unlink(tmp)
        except OSError:  # noqa: BLE001
            pass


def probe_duration(path: str | os.PathLike) -> float:
    try:
        return float(video_probe(path).get("duration") or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def rounded(value: float, nd: int = 3) -> float:
    return round(float(value), nd)
