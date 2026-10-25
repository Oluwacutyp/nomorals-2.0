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

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

NUMPY_AVAILABLE = np is not None


def _require_numpy() -> None:
    if np is None:
        raise MotionStudioError(
            "nomorals.media.motion_studio._core is unavailable: numpy is not installed "
            "(pip install numpy)")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None; ImageDraw = None; ImageFont = None  # type: ignore[assignment]

PIL_AVAILABLE = Image is not None


def _require_pil() -> None:
    if Image is None:
        raise MotionStudioError(
            "nomorals.media.motion_studio._core is unavailable: Pillow is not installed "
            "(pip install Pillow)")


from ...core.logging_setup import get_logger
from ...core.profiles import get_profile_kind
from ...media_edit.videos import MediaEditError, ffmpeg_path, run_ffmpeg, video_probe
from ..failures import record_failure

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
    "ledger_status",
    "LEDGER_WINDOW_SIZE",
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
    _require_pil()
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


#: How many entries the live ``ledger.json`` window holds. This is a
#: *display/compat* window only — the full history lives in the
#: append-only archive (see below) and is never truncated.
LEDGER_WINDOW_SIZE = 500


def _archive_path() -> Path:
    return LEDGER_PATH.with_name(LEDGER_PATH.stem + ".archive.jsonl")


def _meta_path() -> Path:
    return LEDGER_PATH.with_name(LEDGER_PATH.stem + ".meta.json")


def _read_window_file() -> list[dict]:
    """The live window file exactly as stored (legacy read semantics)."""
    try:
        if LEDGER_PATH.exists():
            data = json.loads(LEDGER_PATH.read_text())
            return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        pass
    return []


def _read_archive() -> list[dict] | None:
    """All archived entries in record order; None when no archive exists."""
    archive = _archive_path()
    if not archive.exists():
        return None
    entries: list[dict] = []
    try:
        for line in archive.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 - skip a torn line, keep the rest
                continue
            if isinstance(row, dict):
                entries.append(row)
    except Exception:  # noqa: BLE001
        return None
    return entries


def _read_meta() -> dict:
    try:
        data = json.loads(_meta_path().read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _write_meta(meta: dict) -> None:
    _meta_path().write_text(json.dumps(meta, indent=1))


def _ensure_archive() -> bool:
    """Guarantee the archive holds every window entry (one-time migration).

    The pre-#82 ledger was ONLY the window file, rewritten as
    ``ledger[-500:]`` per record — history past 500 was silently dropped.
    On first contact the surviving window is seeded into the append-only
    archive (atomically, via a temp file + rename) so nothing that still
    exists is ever lost again. Returns True when appends may proceed.
    """
    archive = _archive_path()
    if archive.exists() and archive.stat().st_size > 0:
        return True
    window = _read_window_file()
    if not window:
        return True  # fresh install: the first append creates the archive
    tmp = archive.with_name(archive.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in window:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, archive)
    meta = _read_meta()
    meta.update({"migrated_from_legacy": True,
                 "migrated_entries": len(window)})
    _write_meta(meta)
    _log.info("motion ledger: migrated %d existing entries into the "
              "append-only archive %s", len(window), archive)
    return True


def _note_overflow(count: int) -> None:
    """Record (meta + log) that entries rotated out of the live window.

    Rotating out of the *window* is not data loss — the entries are in
    the archive — but it is never silent either: the event is counted in
    the ledger meta and surfaced by :func:`ledger_status`.
    """
    meta = _read_meta()
    total = int(meta.get("rotated_out_total") or 0) + count
    meta["rotated_out_total"] = total
    now = datetime.now(timezone.utc).isoformat()
    meta.setdefault("first_rotated_out_at", now)
    meta["last_rotated_out_at"] = now
    _write_meta(meta)
    _log.info("motion ledger: %d entr%s rotated out of the live window "
              "into the archive (%d archived total)",
              count, "y" if count == 1 else "ies", total)


def record_ledger(entry: dict) -> None:
    """Append a render record (best-effort; never breaks a render).

    Every entry is appended to the append-only archive
    (``ledger.archive.jsonl``, the download-kit IntegrityLedger pattern)
    and mirrored into the live window file, which keeps only the newest
    :data:`LEDGER_WINDOW_SIZE` entries for compatibility. Nothing is
    ever dropped from the archive; entries rotating out of the window
    are counted in the ledger meta (see :func:`ledger_status`).
    """
    try:
        stamped = dict(entry)
        stamped.setdefault("ts", datetime.now(timezone.utc).isoformat())
        archived = False
        try:
            archived = _ensure_archive()
        except Exception as exc:  # noqa: BLE001 - fall back to window-only below
            record_failure("motion_studio._core.record_ledger", exc)
            archived = False
        if archived:
            LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_archive_path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(stamped) + "\n")
        window = _read_window_file()
        window.append(stamped)
        overflow = len(window) - LEDGER_WINDOW_SIZE
        if overflow > 0:
            window = window[-LEDGER_WINDOW_SIZE:]
            if archived:
                _note_overflow(overflow)
        LEDGER_PATH.write_text(json.dumps(window, indent=1))
    except Exception:  # noqa: BLE001
        _log.debug("ledger write failed", exc_info=True)


def read_ledger(limit: int | None = None) -> list[dict]:
    """Ledger entries, oldest first.

    ``limit=None`` (the default) returns the FULL history from the
    archive; ``limit=N`` returns only the newest ``N`` entries. The
    recent window is a read choice here — never a storage limit.
    """
    entries: list[dict] = []
    try:
        try:
            _ensure_archive()
        except Exception:  # noqa: BLE001 - reads still serve the window
            pass
        archived = _read_archive()
        entries = archived if archived is not None else _read_window_file()
    except Exception:  # noqa: BLE001
        entries = _read_window_file()
    if limit is None:
        return entries
    if limit <= 0:
        return []
    return entries[-int(limit):]


def ledger_status() -> dict:
    """The ledger's whole truth: totals, archive state, oldest entry."""
    entries = read_ledger()
    window = _read_window_file()
    meta = _read_meta()
    total = len(entries)
    active = len(window)
    archived = max(0, total - active)
    oldest = entries[0] if entries else None
    newest = entries[-1] if entries else None
    return {
        "ledger_path": str(LEDGER_PATH),
        "archive_path": str(_archive_path()),
        "total_entries": total,
        "active_entries": active,
        "archived_entries": archived,
        "window_size": LEDGER_WINDOW_SIZE,
        "oldest_entry": oldest,
        "oldest_at": oldest.get("ts") if oldest else None,
        "newest_at": newest.get("ts") if newest else None,
        "rotated_out_total": int(meta.get("rotated_out_total") or archived),
        "first_rotated_out_at": meta.get("first_rotated_out_at"),
        "migrated_from_legacy": bool(meta.get("migrated_from_legacy")),
        "migrated_entries": int(meta.get("migrated_entries") or 0),
    }


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
    _require_pil()
    _require_numpy()
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
    _require_numpy()
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
