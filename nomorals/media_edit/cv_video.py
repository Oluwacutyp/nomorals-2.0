"""Frame-level video processing with multi-backend support.

Companion to :mod:`.videos` (the ffmpeg engine). Standing rule: **no
capability is limited to a single dependency** — every op below supports
every backend that can genuinely do the job, with ``backend="auto"`` (the
default) picking the best available one. The caller never needs to know
which backend ran; every result dict carries a ``"backend"`` key saying
which one won.

Backend preference (best FREE option first; all backends are free):

======================== ==================== ================================
op                       primary              fallback / notes
======================== ==================== ================================
``extract_frames``       OpenCV — exact       ffmpeg (``videos`` engine)
                         decoded frames,      — keyframe-approximate seeks
                         fast
``apply_filter_to_video`` per-filter:          the other backend;
                         ffmpeg for           sketch/cartoonize are
                         grayscale, blur,     OpenCV-first (ffmpeg only
                         sharpen, invert,     has approximations)
                         sepia, vignette,
                         edges, emboss,
                         pixelate, warm, cool
                         (SIMD-fast, keeps
                         audio); OpenCV for
                         sketch, cartoonize
``create_timelapse``     ffmpeg — ``setpts``  OpenCV frame sampling
                         + ``atempo`` keeps
                         the audio track
``stabilize_basic``      ffmpeg: vid.stab     OpenCV feature-tracking
                         two-pass when        stabilizer (always available
                         libvidstab is in     with the media-edit extra)
                         the ffmpeg build,
                         else built-in
                         ``deshake``
``frame_diff_highlights`` OpenCV — per-frame  pure-python fallback:
                         scores + thumbnails  stdlib diffs via rawvideo
                         need frame access    pipe, same events contract
``slow_motion``          ffmpeg               OpenCV: DIS-flow or blend;
                         ``minterpolate``     pure-python cross-dissolve
                         (mci, keeps audio)   (video-only)
``reverse_video``        ffmpeg               OpenCV backward index read
                         (reverse+areverse)   (video-only)
``boomerang``            ffmpeg trim+reverse  n/a: needs muxed concat
                         +concat (audio-aware)
``find_blurry_frames``   OpenCV (Laplacian    pure-python fallback:
                         variance)            gradient-energy scores,
                         relative threshold
``motion_heatmap``       OpenCV (accumulated  pure-python fallback:
                         diffs, JET, full     stdlib accumulation,
                         res)                 JET-like ramp, PNG writer
``split_on_scenes``      detect: OpenCV,     detect fallback: ffmpeg
                         cut: ffmpeg         ``select``; cuts always ffmpeg
``kenburns``             ffmpeg ``zoompan``   n/a: no better alternative
``slideshow``            ffmpeg xfade chain   n/a: needs muxed concat
``chroma_key``           ffmpeg chromakey +   n/a: needs muxed composite
                         overlay
``pip``                  ffmpeg overlay       n/a: needs muxed composite
``freeze_frame``         ffmpeg loop          frame pick: OpenCV exact /
                         (frame: OpenCV)      ffmpeg seek
``denoise_video``        ffmpeg ``hqdn3d``    OpenCV fastNlMeansDenoising
                         (keeps audio)        (video-only)
======================== ==================== ================================

``pip install nomorals[media-edit]`` ships OpenCV; ffmpeg is a system
binary (``sudo apt install ffmpeg`` / ``brew install ffmpeg`` /
``winget install ffmpeg``). ``backend="python"`` is the zero-mandatory-
dependency tier: stdlib-only pixel loops, with ffmpeg used just for
decode/encode IO — every frame-analysis op works with it when OpenCV is
absent. When a backend is missing, ``auto`` silently uses the next one —
the last-resort install hints only fire when NO backend can do the job.

Every op writes a NEW file — originals are never overwritten.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .images import MediaEditError, _unique_output

_log = get_logger(__name__)

MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB default cap, mirrors videos.py

_FFMPEG_INSTALL = (
    "ffmpeg is not installed. Install it — Debian/Ubuntu: "
    "`sudo apt install ffmpeg`; macOS: `brew install ffmpeg`; Windows: "
    "`winget install ffmpeg`; or https://ffmpeg.org/download.html."
)
_CV2_INSTALL = (
    "this needs the OpenCV backend: "
    "`pip install nomorals[media-edit]` (opencv-python-headless>=4.8)."
)

_BACKENDS = ("auto", "opencv", "ffmpeg", "python")


# ---------------------------------------------------------------------------
# backend machinery
# ---------------------------------------------------------------------------

def _cv2():
    """Direct cv2 import — last-resort hint only, never a gate on import."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(_CV2_INSTALL) from exc
    return cv2


def _np():
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(_CV2_INSTALL) from exc
    return numpy


def _have_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def cv2_available() -> bool:
    """True when the OpenCV backend can be used right now."""
    return _have_cv2()


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


_filter_caps: dict[str, bool] | None = None


def _ffmpeg_has_filter(*names: str) -> bool:
    """True when this ffmpeg build provides ALL named filters."""
    global _filter_caps
    exe = _ffmpeg_bin()
    if not exe:
        return False
    if _filter_caps is None:
        try:
            out = subprocess.run(
                [exe, "-hide_banner", "-filters"],
                capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            return False
        _filter_caps = {}
        for line in out.splitlines():
            parts = line.split()
            # lines look like: " .. minterpolate V->V Frame rate ..."
            if len(parts) >= 3:
                _filter_caps[parts[1]] = True
    return all(_filter_caps.get(n, False) for n in names)


def _resolve_backend(requested: str, want: tuple[str, ...]) -> str:
    """Pick the backend for an op.

    ``want`` lists backends best-first. ``"auto"`` takes the first
    available; an explicit name is honored or fails with a clear reason.
    """
    if requested not in _BACKENDS:
        raise MediaEditError(
            f"backend must be one of {_BACKENDS}, got {requested!r}")
    if requested == "auto":
        for b in want:
            if b == "opencv" and _have_cv2():
                return "opencv"
            if b == "ffmpeg" and _ffmpeg_bin():
                return "ffmpeg"
            if b == "python" and _ffmpeg_bin():
                # pure-python fallback: stdlib pixel loops, but decode/encode
                # still goes through the ffmpeg binary
                return "python"
        missing = []
        if "opencv" in want:
            missing.append(_CV2_INSTALL)
        if "ffmpeg" in want or "python" in want:
            missing.append(_FFMPEG_INSTALL)
        raise MediaEditError(
            "no backend available for this op. " + " ".join(missing))
    if requested == "opencv":
        _cv2()  # last-resort hint if truly missing
        return "opencv"
    if requested == "python":
        if not _ffmpeg_bin():
            raise MediaEditError(_FFMPEG_INSTALL)
        return "python"
    exe = _ffmpeg_bin()
    if not exe:
        raise MediaEditError(_FFMPEG_INSTALL)
    return "ffmpeg"


def _check_size(p: Path, cap: int = MAX_VIDEO_BYTES) -> None:
    if p.stat().st_size > cap:
        raise MediaEditError(
            f"video too large: {p.stat().st_size} bytes > {cap} byte cap")


# ---------------------------------------------------------------------------
# Pure-Python fallback machinery (stdlib only: no cv2, no numpy).
#
# Several ops need per-frame pixel access. When OpenCV is unavailable they
# fall back to ffmpeg rawvideo pipes + plain-Python pixel loops. Slower
# than OpenCV, but always working — decode/encode still goes through
# ffmpeg, so ``backend="python"`` still needs the ffmpeg binary.
# ---------------------------------------------------------------------------

def _pipe_frame_size(src: str | os.PathLike[str], width: int
                     ) -> tuple[int, int]:
    """Scaled (w, h) for rawvideo pipes, both even, aspect preserved."""
    from .videos import video_probe
    info = video_probe(src)
    w0 = int(info.get("width") or 0)
    h0 = int(info.get("height") or 0)
    if w0 <= 0 or h0 <= 0:  # pragma: no cover - defensive
        raise MediaEditError(f"could not probe dimensions of {src}")
    w = max(2, (width // 2) * 2)
    h = max(2, int(round(h0 * w / w0)) // 2 * 2)
    return w, h


def _iter_raw_frames(src: str | os.PathLike[str], *, width: int,
                     pix_fmt: str):
    """Yield raw frame bytes via an ffmpeg pipe. Stdlib only.

    ``pix_fmt`` is ``"gray8"`` (1 byte/px) or ``"rgb24"`` (3 bytes/px).
    """
    exe = _ffmpeg_bin()
    if not exe:
        raise MediaEditError(_FFMPEG_INSTALL)
    w, h = _pipe_frame_size(src, width)
    bpp = 1 if pix_fmt == "gray8" else 3
    size = w * h * bpp
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vf", f"scale={w}:{h}", "-pix_fmt", pix_fmt,
           "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield bytes(buf)
    finally:
        with contextlib.suppress(OSError):  # already-closed pipe
            proc.stdout.close()
        proc.wait()


def _py_absdiff_mean(a: bytes, b: bytes) -> float:
    """Mean |a-b| over two equal-length byte strings (stdlib only)."""
    n = len(a)
    if n == 0:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / n


def _py_gradient_energy(gray: bytes) -> float:
    """Mean horizontal |pixel - neighbor| — cheap sharpness proxy."""
    n = len(gray)
    if n < 2:
        return 0.0
    return sum(abs(x - y) for x, y in zip(gray[:-1], gray[1:], strict=True)) / (n - 1)


def _write_png(path: str | os.PathLike[str], w: int, h: int,
               rgb: bytes) -> None:
    """Minimal truecolor PNG writer — zlib+struct, stdlib only."""
    import struct
    import zlib

    def chunk(typ: bytes, data: bytes) -> bytes:
        head = struct.pack(">I", len(data)) + typ + data
        return head + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    if len(rgb) != w * h * 3:  # pragma: no cover - defensive
        raise MediaEditError("bad rgb buffer for PNG writer")
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + rgb[y * w * 3:(y + 1) * w * 3]
                    for y in range(h))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def _heat_color(t: float) -> tuple[int, int, int]:
    """JET-like colormap for heatmaps, t in [0, 1] -> (r, g, b)."""
    t = max(0.0, min(1.0, t))

    def band(c: float) -> int:
        return int(255 * max(0.0, min(1.0, 1.5 - abs(4.0 * t - c))))

    return band(3.0), band(2.0), band(1.0)


def _rawvideo_encoder(src_w: int, src_h: int, fps: float, out: Path,
                      audio_src: str | os.PathLike[str] | None = None):
    """Open ffmpeg reading rgb24 frames on stdin; returns the Popen.

    Callers write ``src_w*src_h*3``-byte frames to ``proc.stdin`` and
    close it when done.
    """
    from .videos import run_ffmpeg  # noqa: F401  (kept local, unused)
    exe = _ffmpeg_bin()
    if not exe:
        raise MediaEditError(_FFMPEG_INSTALL)
    cmd = [exe, "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{src_w}x{src_h}", "-r", f"{fps:.6f}",
           "-i", "pipe:0"]
    if audio_src:
        cmd += ["-i", str(audio_src), "-map", "0:v", "-map", "1:a?",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-y", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _probe_quick(src: str | os.PathLike[str]) -> dict[str, Any]:
    """fps/duration via OpenCV when present, else ffprobe/ffmpeg parsing."""
    if _have_cv2():
        cv2 = _cv2()
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return {"fps": float(fps),
                "duration": (frames / fps) if fps > 0 else 0.0}
    from .videos import video_probe
    info = video_probe(src)
    return {"fps": info.get("fps") or 0.0,
            "duration": info.get("duration") or 0.0}


def _open_capture(src: str | os.PathLike[str]):
    """Validate + open a video with OpenCV; returns (cv2, cap, info)."""
    cv2 = _cv2()
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise MediaEditError(f"could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    info = {"fps": float(fps), "width": width, "height": height,
            "frames": frames,
            "duration": (frames / fps) if fps > 0 and frames > 0 else 0.0}
    if info["frames"] <= 0 or info["width"] <= 0:
        cap.release()
        raise MediaEditError(f"video has no readable frames: {src}")
    return cv2, cap, info


def _frames_dir(src: Path, out_dir: str | os.PathLike[str] | None,
                suffix: str) -> Path:
    target = ((Path(out_dir) if out_dir else src.parent / "edited")
              / f"{src.stem}-{suffix}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _resize_keep_aspect(cv2, frame, width: int | None):
    if width and width > 0 and frame.shape[1] != width:
        h, w = frame.shape[:2]
        nh = max(1, round(h * width / w))
        return cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    return frame


def _target_frame_indices(*, n_frames: int, fps: float, duration: float,
                          timestamps, interval, count) -> list[int]:
    """Resolve exactly one of timestamps/interval/count to frame indices."""
    from .videos import parse_time
    chosen = sum(x is not None for x in (timestamps, interval, count))
    if chosen != 1:
        raise MediaEditError(
            "extract_frames needs exactly one of: timestamps, interval, count")
    if timestamps is not None:
        idx = []
        for t in timestamps:
            secs = parse_time(t)
            if duration and secs > duration:
                raise MediaEditError(
                    f"timestamp {t} is past the video end ({duration:.2f}s)")
            idx.append(min(max(0, int(secs * fps)), n_frames - 1))
        return sorted(set(idx))
    if count is not None:
        if count <= 0:
            raise MediaEditError("count must be positive")
        if n_frames <= 1 or count == 1:
            return [0]
        step = (n_frames - 1) / (count - 1)
        # count evenly spaced frames spanning first..last frame
        return sorted({min(int(round(i * step)), n_frames - 1)
                       for i in range(count)})
    gap = parse_time(interval)  # type: ignore[arg-type]
    if gap <= 0:
        raise MediaEditError("interval must be positive")
    every = max(1, int(round(gap * fps)))
    return list(range(0, n_frames, every))


# ---------------------------------------------------------------------------
# frame extraction — OpenCV primary (exact), ffmpeg fallback
# ---------------------------------------------------------------------------

def grab_frame_at(src: str | os.PathLike[str],
                  timestamp: str | float | int):
    """Grab the exact decoded frame at ``timestamp`` (BGR numpy array).

    Unlike ``ffmpeg -ss`` this seeks by frame index, so the result is the
    actual frame at that time — not the nearest keyframe. Returns
    ``(frame, actual_seconds)``. OpenCV-only: exactness needs frame access.
    """
    from .videos import parse_time
    cv2, cap, info = _open_capture(src)
    try:
        secs = parse_time(timestamp)
        if info["duration"] and secs > info["duration"]:
            raise MediaEditError(
                f"timestamp {timestamp} is past the video end "
                f"({info['duration']:.2f}s)")
        target = min(max(0, int(secs * info["fps"])), info["frames"] - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise MediaEditError(
                f"could not decode frame at {secs:.2f}s in {src}")
        actual = target / info["fps"] if info["fps"] else secs
        return frame, actual
    finally:
        cap.release()


def extract_frames(src: str | os.PathLike[str], *,
                   out_dir: str | os.PathLike[str] | None = None,
                   timestamps: list[str | float] | None = None,
                   interval: str | float | None = None,
                   count: int | None = None,
                   width: int | None = None,
                   fmt: str = "jpg",
                   backend: str = "auto",
                   progress_cb: Callable[[float], None] | None = None
                   ) -> dict[str, Any]:
    """Extract frames at explicit timestamps, every ``interval``, or
    ``count`` evenly spaced frames.

    Backend: OpenCV primary — frames are exact decoded frames, not
    keyframe-approximated. ffmpeg fallback (the :mod:`.videos` engine)
    when OpenCV is unavailable. ``"backend"`` in the result says which
    ran. Writes ``frame-<i>-t<secs>s.<fmt>`` files into ``<stem>-frames/``.
    """
    chosen = _resolve_backend(backend, ("opencv", "ffmpeg"))
    if chosen == "ffmpeg":
        from .videos import extract_frames as _ff_extract
        res = _ff_extract(src, out_dir=out_dir, timestamps=timestamps,
                          interval=interval, count=count,
                          width=width or 640)
        res["backend"] = "ffmpeg"
        res["exact_frames"] = False
        return res
    cv2, cap, info = _open_capture(src)
    try:
        if fmt.lower() not in ("jpg", "jpeg", "png"):
            raise MediaEditError(f"unsupported frame format: {fmt!r}")
        ext = ".jpg" if fmt.lower() in ("jpg", "jpeg") else ".png"
        want = set(_target_frame_indices(
            n_frames=info["frames"], fps=info["fps"],
            duration=info["duration"], timestamps=timestamps,
            interval=interval, count=count))
        if not want:
            raise MediaEditError("no frames selected")
        target = _frames_dir(Path(src), out_dir, "frames")
        files: list[str] = []
        t0 = time.monotonic()
        for i in range(info["frames"]):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if i in want:
                frame = _resize_keep_aspect(cv2, frame, width)
                secs = i / info["fps"] if info["fps"] else 0.0
                name = f"frame-{len(files):03d}-t{secs:.2f}s{ext}"
                dest = target / name
                if not cv2.imwrite(str(dest), frame):
                    raise MediaEditError(f"could not write frame {dest}")
                files.append(str(dest))
            if progress_cb and i % 60 == 0:
                progress_cb(i / info["frames"])
        if progress_cb:
            progress_cb(1.0)
        _log.info("extract_frames[%s]: %d frames from %s in %.1fs",
                  chosen, len(files), src, time.monotonic() - t0)
        return {"input": str(Path(src)), "frames_dir": str(target),
                "frames": files, "count": len(files),
                "fps": info["fps"], "duration": info["duration"],
                "backend": "opencv", "exact_frames": True}
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# per-frame filters — per-filter backend routing
# ---------------------------------------------------------------------------

def _filter_grayscale(cv2, frame, strength: float):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _filter_blur(cv2, frame, strength: float):
    k = max(3, int(round(3 + strength * 12)))
    k += 1 - k % 2  # odd
    return cv2.GaussianBlur(frame, (k, k), 0)


def _filter_sharpen(cv2, frame, strength: float):
    np = _np()
    kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(frame, -1, kernel)
    return cv2.addWeighted(frame, 1.0 - min(1.0, strength),
                           sharp, min(1.0, strength), 0)


def _filter_edges(cv2, frame, strength: float):
    lo = max(10, int(100 - strength * 60))
    hi = max(30, int(200 - strength * 100))
    edges = cv2.Canny(frame, lo, hi)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def _filter_cartoonize(cv2, frame, strength: float):
    d = max(5, int(5 + strength * 6))
    smooth = cv2.bilateralFilter(frame, d, 75, 75)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 7), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(smooth, edges)


def _filter_invert(cv2, frame, strength: float):
    return cv2.bitwise_not(frame)


def _filter_sepia(cv2, frame, strength: float):
    np = _np()
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    s = min(1.0, max(0.0, strength))
    return cv2.addWeighted(frame, 1.0 - s, sepia, s, 0)


def _filter_warm(cv2, frame, strength: float):
    np = _np()
    b, g, r = cv2.split(frame)
    s = 0.18 * max(0.0, strength)
    r = np.clip(r.astype(np.float32) * (1 + s), 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) * (1 - s), 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def _filter_cool(cv2, frame, strength: float):
    np = _np()
    b, g, r = cv2.split(frame)
    s = 0.18 * max(0.0, strength)
    b = np.clip(b.astype(np.float32) * (1 + s), 0, 255).astype(np.uint8)
    r = np.clip(r.astype(np.float32) * (1 - s), 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def _filter_emboss(cv2, frame, strength: float):
    np = _np()
    base = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
                    dtype=np.float32)
    embossed = cv2.filter2D(frame, -1, base * max(0.1, strength))
    return np.clip(embossed.astype(np.float32) + 128, 0, 255).astype(np.uint8)


def _filter_vignette(cv2, frame, strength: float):
    np = _np()
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dx = (xx - w / 2) / (w / 2)
    dy = (yy - h / 2) / (h / 2)
    dist = np.sqrt(dx * dx + dy * dy)
    mask = np.clip(1.0 - dist * 0.45 * max(0.0, strength), 0.0, 1.0)
    return (frame.astype(np.float32) * mask[..., None]).astype(np.uint8)


def _filter_pixelate(cv2, frame, strength: float):
    h, w = frame.shape[:2]
    s = max(2, int(4 + max(0.0, strength) * 12))
    small = cv2.resize(frame, (max(1, w // s), max(1, h // s)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _filter_sketch(cv2, frame, strength: float):
    if hasattr(cv2, "pencilSketch"):
        gray, _ = cv2.pencilSketch(
            frame, sigma_s=60, sigma_r=0.07,
            shade_factor=0.05 + 0.05 * max(0.0, strength))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # manual dodge-blend fallback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


FILTERS: dict[str, Callable] = {
    "grayscale": _filter_grayscale,
    "blur": _filter_blur,
    "sharpen": _filter_sharpen,
    "edges": _filter_edges,
    "cartoonize": _filter_cartoonize,
    "invert": _filter_invert,
    "sepia": _filter_sepia,
    "emboss": _filter_emboss,
    "vignette": _filter_vignette,
    "pixelate": _filter_pixelate,
    "sketch": _filter_sketch,
    "warm": _filter_warm,
    "cool": _filter_cool,
}

# Preferred backend first. ffmpeg wins where its filters are equivalent
# (SIMD-fast and audio-preserving); OpenCV is the reference-quality
# primary for sketch/cartoonize where ffmpeg only has approximations.
FILTER_BACKENDS: dict[str, tuple[str, ...]] = {
    "grayscale": ("ffmpeg", "opencv"),
    "blur": ("ffmpeg", "opencv"),
    "sharpen": ("ffmpeg", "opencv"),
    "invert": ("ffmpeg", "opencv"),
    "sepia": ("ffmpeg", "opencv"),
    "vignette": ("ffmpeg", "opencv"),
    "edges": ("ffmpeg", "opencv"),      # edgedetect=mode=canny ≈ Canny
    "emboss": ("ffmpeg", "opencv"),     # convolution, same kernel
    "pixelate": ("ffmpeg", "opencv"),   # neighbor scaling, identical
    "warm": ("ffmpeg", "opencv"),       # colorbalance grade
    "cool": ("ffmpeg", "opencv"),       # colorbalance grade
    "sketch": ("opencv", "ffmpeg"),     # ffmpeg: canny edges + negate
    "cartoonize": ("opencv", "ffmpeg"),  # ffmpeg: hqdn3d + colormix edges
}


def _ffmpeg_filter_for(filter_name: str, strength: float) -> str:
    if filter_name == "grayscale":
        return "hue=s=0,format=yuv420p"
    if filter_name == "blur":
        lr = 2.0 + max(0.0, strength) * 8.0
        return f"boxblur=luma_radius={lr:.1f}:luma_power=1"
    if filter_name == "sharpen":
        return f"unsharp=5:5:{0.5 + max(0.0, strength):.2f}"
    if filter_name == "invert":
        return "negate"
    if filter_name == "sepia":
        return ("colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:"
                ".272:.534:.131:0")
    if filter_name == "vignette":
        angle = max(2.0, 5.0 - max(0.0, strength))
        return f"vignette=a=PI/{angle:.2f}"
    s = max(0.0, strength)
    if filter_name == "edges":
        return "edgedetect=mode=canny,format=yuv420p"
    if filter_name == "emboss":
        m = "-2 -1 0 -1 1 1 0 1 2"
        return (f"convolution='0m={m}:1m={m}:2m={m}:3m={m}:"
                f"0bias=128:1bias=128:2bias=128:3bias=128',format=yuv420p")
    if filter_name == "pixelate":
        b = int(4 + s * 12)  # block size grows with strength
        return (f"scale='iw/{b}':'ih/{b}':flags=neighbor,"
                f"scale='iw*{b}':'ih*{b}':flags=neighbor,format=yuv420p")
    if filter_name == "sketch":
        # canny edges, inverted: dark pencil lines on white
        return "edgedetect=mode=canny,negate,format=yuv420p"
    if filter_name == "warm":
        return (f"colorbalance=rs={0.35 * s:.2f}:gs={0.08 * s:.2f}:"
                f"bs={-0.35 * s:.2f},format=yuv420p")
    if filter_name == "cool":
        return (f"colorbalance=rs={-0.35 * s:.2f}:gs={0.08 * s:.2f}:"
                f"bs={0.35 * s:.2f},format=yuv420p")
    if filter_name == "cartoonize":
        # smoothed colors with edges mixed back in — a decent cartoon
        # approximation of the OpenCV bilateral + adaptive-threshold look
        return (f"hqdn3d={4 + 6 * s:.1f}:{3 + 5 * s:.1f}:"
                f"{5 + 7 * s:.1f}:{4 + 5 * s:.1f},"
                "edgedetect=mode=colormix,format=yuv420p")
    raise MediaEditError(  # pragma: no cover - guarded by FILTER_BACKENDS
        f"no ffmpeg equivalent for filter {filter_name!r}; "
        f"use backend='opencv'")


def _new_writer(cv2, dest: Path, fps: float, size: tuple[int, int],
                codec: str = "mp4v"):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(dest), fourcc, fps, size)
    if not writer.isOpened():
        raise MediaEditError(
            f"could not open video writer for {dest} (codec {codec!r})")
    return writer


def _apply_filter_opencv(src: Path, *, filter_name: str, strength: float,
                         out_dir, suffix: str | None, codec: str,
                         progress_cb) -> dict[str, Any]:
    cv2, cap, info = _open_capture(src)
    fn = FILTERS[filter_name]
    out = _unique_output(src, Path(out_dir) if out_dir else src.parent / "edited",
                         suffix or f"cv-{filter_name}", ".mp4")
    writer = None
    t0 = time.monotonic()
    try:
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = _new_writer(cv2, out, info["fps"] or 30.0,
                                     (w, h), codec)
            writer.write(fn(cv2, frame, strength))
            written += 1
            if progress_cb and written % 60 == 0 and info["frames"]:
                progress_cb(written / info["frames"])
        if writer is None:
            raise MediaEditError(f"video has no readable frames: {src}")
        writer.release()
        if progress_cb:
            progress_cb(1.0)
        _log.info("apply_filter_to_video[opencv]: %s -> %s (%d frames, %.1fs)",
                  filter_name, out, written, time.monotonic() - t0)
        return {"input": str(src), "output": str(out),
                "filter": filter_name, "strength": strength,
                "frames": written, "bytes": out.stat().st_size,
                "audio_dropped": True, "fps": info["fps"],
                "duration": info["duration"], "backend": "opencv"}
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def _apply_filter_ffmpeg(src: Path, *, filter_name: str, strength: float,
                         out_dir, suffix: str | None,
                         progress_cb, duration: float) -> dict[str, Any]:
    from .videos import run_ffmpeg, FFMPEG_TIMEOUT
    vf = _ffmpeg_filter_for(filter_name, strength)
    out = _unique_output(src, Path(out_dir) if out_dir else src.parent / "edited",
                         suffix or f"ff-{filter_name}", ".mp4")
    t0 = time.monotonic()
    run_ffmpeg(["-i", str(src), "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", str(out)],
               timeout=FFMPEG_TIMEOUT, progress_cb=progress_cb,
               duration=duration or None)
    _log.info("apply_filter_to_video[ffmpeg]: %s -> %s (%.1fs)",
              filter_name, out, time.monotonic() - t0)
    return {"input": str(src), "output": str(out),
            "filter": filter_name, "strength": strength,
            "bytes": out.stat().st_size, "audio_dropped": False,
            "duration": duration, "backend": "ffmpeg"}


def apply_filter_to_video(src: str | os.PathLike[str], *,
                          filter_name: str,
                          strength: float = 1.0,
                          out_dir: str | os.PathLike[str] | None = None,
                          suffix: str | None = None,
                          codec: str = "mp4v",
                          backend: str = "auto",
                          progress_cb: Callable[[float], None] | None = None
                          ) -> dict[str, Any]:
    """Apply a per-frame artistic filter to a whole video.

    Available filters: grayscale, blur, sharpen, edges, cartoonize,
    invert, sepia, emboss, vignette, pixelate, sketch, warm, cool.
    ``strength`` tunes the effect (0..2, filter-dependent).

    Backend routing (see ``FILTER_BACKENDS``): ffmpeg is preferred for
    grayscale/blur/sharpen/invert/sepia/vignette/edges/emboss/pixelate/
    warm/cool — its filters are SIMD-fast, equivalent in quality, and the
    audio track survives (``audio_dropped: False``). OpenCV is the
    reference-quality primary for sketch/cartoonize, where ffmpeg only
    has approximations (edges+negate / hqdn3d+colormix); its writer is
    video-only, so audio is dropped there (re-attach with
    :func:`.videos.mix_audio`). ``"backend"`` in the result says which ran.
    """
    want = FILTER_BACKENDS.get(filter_name)
    if want is None:
        raise MediaEditError(
            f"unknown filter {filter_name!r}; "
            f"available: {', '.join(sorted(FILTERS))}")
    if not (0.0 <= strength <= 2.0):
        raise MediaEditError("strength must be within [0, 2]")
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    chosen = _resolve_backend(backend, want)
    if chosen == "ffmpeg":
        quick = _probe_quick(src)
        return _apply_filter_ffmpeg(p, filter_name=filter_name,
                                    strength=strength, out_dir=out_dir,
                                    suffix=suffix, progress_cb=progress_cb,
                                    duration=quick["duration"])
    return _apply_filter_opencv(p, filter_name=filter_name,
                                strength=strength, out_dir=out_dir,
                                suffix=suffix, codec=codec,
                                progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# timelapse — ffmpeg primary (keeps audio), OpenCV fallback
# ---------------------------------------------------------------------------

def _atempo_chain(factor: float) -> str:
    parts: list[str] = []
    rem = float(factor)
    while rem > 2.0:
        parts.append("atempo=2.0")
        rem /= 2.0
    parts.append(f"atempo={rem:.4f}")
    return ",".join(parts)


def _timelapse_ffmpeg(src: Path, *, factor: int, out_dir,
                      suffix: str | None, progress_cb,
                      duration: float, deflicker: bool) -> dict[str, Any]:
    from .videos import run_ffmpeg, FFMPEG_TIMEOUT
    out = _unique_output(src, Path(out_dir) if out_dir else src.parent / "edited",
                         suffix or f"timelapse-{factor}x", ".mp4")
    t0 = time.monotonic()
    vf = f"setpts={1.0 / factor:.6f}*PTS"
    if deflicker:
        vf += ",deflicker"
    run = run_ffmpeg(
        ["-i", str(src),
         "-vf", vf,
         "-af", _atempo_chain(factor),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", str(out)],
        timeout=FFMPEG_TIMEOUT, progress_cb=progress_cb,
        duration=duration or None)
    new_duration = (duration / factor) if duration else 0.0
    _log.info("create_timelapse[ffmpeg]: %dx -> %s (%.1fs)",
              factor, out, time.monotonic() - t0)
    return {"input": str(src), "output": str(out), "factor": factor,
            "duration_out": new_duration, "bytes": out.stat().st_size,
            "audio_dropped": False, "seconds": run["seconds"],
            "deflicker": deflicker, "backend": "ffmpeg"}


def _timelapse_opencv(src: Path, *, factor: int, out_dir,
                      suffix: str | None, codec: str,
                      progress_cb) -> dict[str, Any]:
    cv2, cap, info = _open_capture(src)
    out = _unique_output(src, Path(out_dir) if out_dir else src.parent / "edited",
                         suffix or f"timelapse-{factor}x", ".mp4")
    writer = None
    t0 = time.monotonic()
    try:
        idx = kept = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % factor == 0:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = _new_writer(cv2, out, info["fps"] or 30.0,
                                         (w, h), codec)
                writer.write(frame)
                kept += 1
            idx += 1
            if progress_cb and idx % 120 == 0 and info["frames"]:
                progress_cb(idx / info["frames"])
        if writer is None:
            raise MediaEditError(f"video has no readable frames: {src}")
        writer.release()
        if progress_cb:
            progress_cb(1.0)
        new_duration = kept / (info["fps"] or 30.0)
        _log.info("create_timelapse[opencv]: %dx -> %s (%d/%d frames, %.1fs)",
                  factor, out, kept, idx, time.monotonic() - t0)
        return {"input": str(src), "output": str(out), "factor": factor,
                "frames_in": idx, "frames_out": kept,
                "duration_out": new_duration, "bytes": out.stat().st_size,
                "audio_dropped": True, "backend": "opencv"}
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def create_timelapse(src: str | os.PathLike[str], *,
                     factor: int = 4,
                     out_dir: str | os.PathLike[str] | None = None,
                     suffix: str | None = None,
                     codec: str = "mp4v",
                     backend: str = "auto",
                     deflicker: bool = False,
                     progress_cb: Callable[[float], None] | None = None
                     ) -> dict[str, Any]:
    """Speed a video up by sampling every ``factor``-th frame.

    A 60s clip at factor=4 becomes a 15s timelapse at the original fps.
    ``deflicker=True`` (ffmpeg path) smooths the exposure flicker that
    plagues day-to-night timelapses.

    Backend: ffmpeg primary — ``setpts`` for video plus an ``atempo``
    chain for audio, so the soundtrack survives sped-up
    (``audio_dropped: False``). OpenCV fallback samples frames directly
    (video-only writer, audio dropped). ``"backend"`` says which ran.
    """
    if factor < 2:
        raise MediaEditError("timelapse factor must be >= 2")
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    chosen = _resolve_backend(backend, ("ffmpeg", "opencv"))
    if chosen == "ffmpeg":
        quick = _probe_quick(src)
        return _timelapse_ffmpeg(p, factor=factor, out_dir=out_dir,
                                 suffix=suffix, progress_cb=progress_cb,
                                 duration=quick["duration"],
                                 deflicker=deflicker)
    if deflicker:
        raise MediaEditError(
            "deflicker needs the ffmpeg backend; "
            "use backend='ffmpeg' or deflicker=False")
    return _timelapse_opencv(p, factor=factor, out_dir=out_dir,
                             suffix=suffix, codec=codec,
                             progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# stabilization — ffmpeg vid.stab primary when available, else OpenCV
# ---------------------------------------------------------------------------

def _stabilize_ffmpeg(src: Path, *, smoothing_radius: int, out_dir,
                      suffix: str, progress_cb,
                      duration: float) -> dict[str, Any]:
    """ffmpeg stabilization, best available method in this build.

    vid.stab two-pass when libvidstab is present (best quality), else the
    built-in single-pass ``deshake`` filter (always available). Both keep
    audio.
    """
    from .videos import run_ffmpeg, FFMPEG_TIMEOUT
    out = _unique_output(src, Path(out_dir) if out_dir else src.parent / "edited",
                         suffix, ".mp4")
    if _ffmpeg_has_filter("vidstabdetect", "vidstabtransform"):
        method = "vidstab-two-pass"
        work = Path(tempfile.mkdtemp(prefix="vidstab-"))
        trf = work / "transforms.trf"
        try:
            run_ffmpeg(["-i", str(src), "-vf",
                        f"vidstabdetect=shakiness=5:accuracy=15:"
                        f"result={trf}:show=0",
                        "-an", "-f", "null", "-"],
                       timeout=FFMPEG_TIMEOUT, duration=duration or None)
            run_ffmpeg(["-i", str(src), "-vf",
                        f"vidstabtransform=input={trf}:zoom=0:"
                        f"smoothing={max(1, smoothing_radius)}:"
                        f"optalgo=gauss:crop=keep",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-c:a", "aac", str(out)],
                       timeout=FFMPEG_TIMEOUT, progress_cb=progress_cb,
                       duration=duration or None)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    elif _ffmpeg_has_filter("deshake"):
        method = "deshake"
        run_ffmpeg(["-i", str(src), "-vf",
                    f"deshake=rx=64:ry=64:edge=mirror:"
                    f"blocksize={max(8, min(64, smoothing_radius * 2))}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", str(out)],
                   timeout=FFMPEG_TIMEOUT, progress_cb=progress_cb,
                   duration=duration or None)
    else:  # pragma: no cover - defensive
        raise MediaEditError(
            "this ffmpeg build has neither vid.stab nor deshake; "
            "use backend='opencv'")
    return {"input": str(src), "output": str(out),
            "smoothing_radius": smoothing_radius,
            "bytes": out.stat().st_size, "audio_dropped": False,
            "backend": "ffmpeg", "method": method}


def _affine_from_dx_dy_da(np, dx: float, dy: float, da: float):
    c, s = math.cos(da), math.sin(da)
    return np.array([[c, -s, dx], [s, c, dy]], dtype=np.float64)


def _stabilize_opencv(src: Path, *, smoothing_radius: int,
                      crop_borders: bool, out_dir, suffix: str,
                      codec: str, progress_cb) -> dict[str, Any]:
    cv2, cap, info = _open_capture(src)
    np = _np()

    # --- pass 1: estimate per-frame camera motion -------------------------
    ok, prev = cap.read()
    if not ok or prev is None:
        cap.release()
        raise MediaEditError(f"video has no readable frames: {src}")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    transforms: list[tuple[float, float, float]] = []
    t0 = time.monotonic()
    idx = 1
    while True:
        ok, curr = cap.read()
        if not ok or curr is None:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30,
            blockSize=3)
        dx = dy = da = 0.0
        if prev_pts is not None:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None)
            good_old = prev_pts[status.flatten() == 1]
            good_new = curr_pts[status.flatten() == 1]
            if len(good_old) >= 6:
                m, _ = cv2.estimateAffinePartial2D(good_old, good_new)
                if m is not None:
                    dx, dy = float(m[0, 2]), float(m[1, 2])
                    da = math.atan2(float(m[1, 0]), float(m[0, 0]))
        transforms.append((dx, dy, da))
        prev_gray = curr_gray
        idx += 1
        if progress_cb and info["frames"] and idx % 60 == 0:
            progress_cb(0.5 * idx / info["frames"])
    cap.release()
    n = len(transforms) + 1
    if n < 3:
        raise MediaEditError(
            f"not enough frames to stabilize ({n}); need >= 3")

    # --- smooth the trajectory (moving average over cumulative motion) -----
    traj = np.zeros((n, 3))
    for i, (dx, dy, da) in enumerate(transforms, start=1):
        traj[i] = traj[i - 1] + np.array([dx, dy, da])
    r = smoothing_radius
    kernel = np.ones(2 * r + 1) / (2 * r + 1)
    smooth = np.stack([np.convolve(traj[:, j], kernel, mode="same")
                       for j in range(3)], axis=1)
    residual = smooth - traj  # per-frame correction to apply

    # --- pass 2: warp frames by the correction ------------------------------
    cap = cv2.VideoCapture(str(src))
    out = _unique_output(src, Path(out_dir) if out_dir else src.parent / "edited",
                         suffix, ".mp4")
    max_dx = float(np.max(np.abs(residual[:, 0])))
    max_dy = float(np.max(np.abs(residual[:, 1])))
    if crop_borders:
        cx = min(w // 2 - 8, int(math.ceil(max_dx)))
        cy = min(h // 2 - 8, int(math.ceil(max_dy)))
        cx, cy = max(0, cx), max(0, cy)
    else:
        cx = cy = 0
    out_w, out_h = w - 2 * cx, h - 2 * cy
    if out_w <= 16 or out_h <= 16:
        cap.release()
        raise MediaEditError(
            "stabilization crop would destroy the frame; "
            "retry with crop_borders=False")
    writer = _new_writer(cv2, out, info["fps"] or 30.0, (out_w, out_h),
                         codec)
    try:
        for i in range(n):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            dx, dy, da = (float(v) for v in residual[i])
            m = _affine_from_dx_dy_da(np, dx, dy, da)
            warped = cv2.warpAffine(
                frame, m, (w, h),
                borderMode=cv2.BORDER_REPLICATE if not crop_borders
                else cv2.BORDER_CONSTANT)
            if crop_borders:
                warped = warped[cy:h - cy, cx:w - cx]
            writer.write(warped)
            if progress_cb and i % 60 == 0:
                progress_cb(0.5 + 0.5 * i / n)
        writer.release()
        if progress_cb:
            progress_cb(1.0)
        _log.info("stabilize_basic[opencv]: %s (%d frames, crop=%dx%d, %.1fs)",
                  out, n, cx, cy, time.monotonic() - t0)
        return {"input": str(src), "output": str(out), "frames": n,
                "smoothing_radius": smoothing_radius,
                "crop": {"x": cx, "y": cy, "w": out_w, "h": out_h},
                "max_correction_px": {"dx": max_dx, "dy": max_dy},
                "bytes": out.stat().st_size, "audio_dropped": True,
                "backend": "opencv", "method": "feature-tracking"}
    finally:
        cap.release()
        writer.release()


def stabilize_basic(src: str | os.PathLike[str], *,
                    smoothing_radius: int = 30,
                    crop_borders: bool = True,
                    out_dir: str | os.PathLike[str] | None = None,
                    suffix: str = "stabilized",
                    codec: str = "mp4v",
                    backend: str = "auto",
                    progress_cb: Callable[[float], None] | None = None
                    ) -> dict[str, Any]:
    """Basic video stabilization.

    Backend: ffmpeg primary — vid.stab two-pass WHEN the ffmpeg build
    ships libvidstab (best quality, audio preserved), else the built-in
    single-pass ``deshake`` filter (always in ffmpeg, still audio-safe).
    OpenCV fallback (``backend="opencv"``) — a built-in feature-tracking
    stabilizer: Shi-Tomasi corners + Lucas-Kanade flow estimate a
    per-frame rigid transform, the camera trajectory is smoothed with a
    moving average, and frames are warped by the residual. Black warp
    borders are cropped away unless ``crop_borders=False``.

    This tames handheld shake; it will not fix rolling-shutter wobble or
    parallax. ``"backend"``/``"method"`` in the result say what ran.
    """
    if smoothing_radius < 1:
        raise MediaEditError("smoothing_radius must be >= 1")
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    want: tuple[str, ...] = ("ffmpeg", "opencv")
    chosen = _resolve_backend(backend, want)
    if chosen == "ffmpeg":
        quick = _probe_quick(src)
        return _stabilize_ffmpeg(p, smoothing_radius=smoothing_radius,
                                 out_dir=out_dir, suffix=suffix,
                                 progress_cb=progress_cb,
                                 duration=quick["duration"])
    return _stabilize_opencv(p, smoothing_radius=smoothing_radius,
                             crop_borders=crop_borders, out_dir=out_dir,
                             suffix=suffix, codec=codec,
                             progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# scene-change / motion highlights
# ---------------------------------------------------------------------------

def _highlights_python(src: str | os.PathLike[str], *, threshold: float,
                       min_gap: float, out_dir,
                       analysis_width: int, progress_cb) -> dict[str, Any]:
    """Pure-Python highlights fallback: rawvideo pipe + stdlib diff loops.

    Same events contract as the OpenCV path; thumbnails are extracted by
    ffmpeg at the detected timestamps. Slower per-frame, but needs no
    cv2/numpy.
    """
    from .videos import video_probe
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    target = _frames_dir(p, out_dir, "highlights")
    info = video_probe(p)
    fps = float(info.get("fps") or 25.0)
    n_frames = int(info.get("frames") or 0)
    w, h = _pipe_frame_size(p, analysis_width)
    events: list[dict[str, Any]] = []
    times: list[float] = []
    scores: list[float] = []
    last_event_t = -min_gap
    t0 = time.monotonic()
    idx = 0
    prev = None
    for buf in _iter_raw_frames(p, width=analysis_width, pix_fmt="gray8"):
        if prev is not None:
            score = _py_absdiff_mean(prev, buf) / 255.0
            t = idx / fps if fps else 0.0
            if score >= threshold and (t - last_event_t) >= min_gap:
                events.append({"t": round(t, 2), "score": round(score, 4)})
                times.append(t)
                scores.append(score)
                last_event_t = t
        prev = buf
        idx += 1
        if progress_cb and n_frames and idx % 120 == 0:
            progress_cb(idx / n_frames)
    if progress_cb:
        progress_cb(1.0)
    # thumbnails via ffmpeg at the event timestamps (no cv2 needed)
    if times:
        got = extract_frames(
            p, timestamps=times, out_dir=target, fmt="jpg",
            backend="ffmpeg")
        for ev, th in zip(events, got["frames"], strict=True):
            ev["thumb"] = th
    _log.info("frame_diff_highlights[python]: %d events in %s (%.1fs)",
              len(events), src, time.monotonic() - t0)
    return {"input": str(p), "events": events, "count": len(events),
            "thumbs_dir": str(target), "threshold": threshold,
            "min_gap": min_gap, "frames_scanned": idx,
            "backend": "python"}


def frame_diff_highlights(src: str | os.PathLike[str], *,
                          threshold: float = 0.03,
                          min_gap: float = 1.0,
                          out_dir: str | os.PathLike[str] | None = None,
                          thumb_width: int = 320,
                          analysis_width: int = 320,
                          backend: str = "auto",
                          progress_cb: Callable[[float], None] | None = None
                          ) -> dict[str, Any]:
    """Detect scene changes and motion bursts via frame differencing.

    Compares consecutive grayscale frames (downscaled for speed); when the
    mean absolute difference exceeds ``threshold`` (fraction of the 0..1
    intensity range) and at least ``min_gap`` seconds passed since the
    last event, the moment is recorded as a highlight with a thumbnail.

    Backend: OpenCV primary — exact per-frame change scores and
    thumbnails. Pure-Python fallback (``backend="python"``): decodes
    through an ffmpeg rawvideo pipe and diffs frames with stdlib loops;
    same events contract, slower. ffmpeg's ``select='gt(scene,…)'`` filter
    can *find* scene cuts but cannot emit per-frame scores or thumbnails
    in the same pass, so there is no pure-ffmpeg backend for this op.

    Returns events: ``[{"t": seconds, "score": 0..1, "thumb": path}]``.
    """
    if not (0.0 < threshold < 1.0):
        raise MediaEditError("threshold must be within (0, 1)")
    if min_gap < 0:
        raise MediaEditError("min_gap must be >= 0")
    chosen = _resolve_backend(backend, ("opencv", "python"))
    if chosen == "ffmpeg":
        raise MediaEditError(
            "frame_diff_highlights needs per-frame change scores and "
            "thumbnails; ffmpeg's select filter cannot produce them. "
            "Use backend='opencv' or 'python'.")
    if chosen == "python":
        return _highlights_python(
            src, threshold=threshold, min_gap=min_gap, out_dir=out_dir,
            analysis_width=analysis_width, progress_cb=progress_cb)
    cv2, cap, info = _open_capture(src)
    p = Path(src)
    target = _frames_dir(p, out_dir, "highlights")
    events: list[dict[str, Any]] = []
    t0 = time.monotonic()
    try:
        ok, prev = cap.read()
        if not ok or prev is None:
            raise MediaEditError(f"video has no readable frames: {src}")
        prev_small = cv2.cvtColor(
            cv2.resize(prev, (analysis_width,
                              max(1, int(prev.shape[0] * analysis_width
                                         / prev.shape[1])))),
            cv2.COLOR_BGR2GRAY)
        last_event_t = -min_gap
        idx = 1
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            small = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (prev_small.shape[1], prev_small.shape[0]))
            diff = cv2.absdiff(prev_small, small)
            score = float(diff.mean()) / 255.0
            t = idx / info["fps"] if info["fps"] else 0.0
            if score >= threshold and (t - last_event_t) >= min_gap:
                thumb = _resize_keep_aspect(cv2, frame, thumb_width)
                tpath = target / f"highlight-{len(events):03d}-t{t:.2f}s.jpg"
                if not cv2.imwrite(str(tpath), thumb):
                    raise MediaEditError(f"could not write {tpath}")
                events.append({"t": round(t, 2), "score": round(score, 4),
                               "thumb": str(tpath)})
                last_event_t = t
            prev_small = small
            idx += 1
            if progress_cb and info["frames"] and idx % 120 == 0:
                progress_cb(idx / info["frames"])
        if progress_cb:
            progress_cb(1.0)
        _log.info("frame_diff_highlights[opencv]: %d events in %s (%.1fs)",
                  len(events), src, time.monotonic() - t0)
        return {"input": str(p), "events": events, "count": len(events),
                "thumbs_dir": str(target), "threshold": threshold,
                "min_gap": min_gap, "frames_scanned": idx,
                "backend": "opencv"}
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# slow motion / frame interpolation
# ---------------------------------------------------------------------------

def _slowmo_atempo_chain(factor: float) -> str:
    parts: list[str] = []
    rem = 1.0 / float(factor)
    while rem < 0.5:
        parts.append("atempo=0.5")
        rem /= 0.5
    parts.append(f"atempo={rem:.4f}")
    return ",".join(parts)


def _slow_motion_ffmpeg(src: Path, *, factor: int, out_dir,
                        suffix: str | None, progress_cb,
                        fps: float, duration: float) -> dict[str, Any]:
    from .videos import run_ffmpeg, FFMPEG_TIMEOUT
    if not _ffmpeg_has_filter("minterpolate"):
        raise MediaEditError(
            "this ffmpeg build lacks the minterpolate filter; "
            "use backend='opencv' or 'python'")
    out = _unique_output(src, Path(out_dir) if out_dir else src.parent / "edited",
                         suffix or f"slowmo-{factor}x", ".mp4")
    target_fps = (fps or 30.0) * factor
    run_ffmpeg(
        ["-i", str(src),
         "-vf", (f"minterpolate=fps={target_fps:.2f}:mi_mode=mci:"
                 f"mc_mode=aobmc:me_mode=bidir:vsbmc=1"),
         "-af", _slowmo_atempo_chain(factor),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", str(out)],
        timeout=FFMPEG_TIMEOUT, progress_cb=progress_cb,
        duration=duration or None)
    return {"input": str(src), "output": str(out), "factor": factor,
            "method": "minterpolate-mci",
            "duration_out": (duration * factor) if duration else 0.0,
            "bytes": out.stat().st_size, "audio_dropped": False,
            "backend": "ffmpeg"}


def _flow_interpolate(cv2, np, prev, curr, prev_gray, curr_gray,
                      alpha: float):
    """Interpolate one frame at position alpha in (0,1) between prev/curr
    using dense DIS optical flow, warping both frames toward the target."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_f = dis.calc(prev_gray, curr_gray, None)
    flow_b = dis.calc(curr_gray, prev_gray, None)
    h, w = prev_gray.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_fx = gx + flow_f[..., 0] * alpha
    map_fy = gy + flow_f[..., 1] * alpha
    map_bx = gx + flow_b[..., 0] * (1.0 - alpha)
    map_by = gy + flow_b[..., 1] * (1.0 - alpha)
    warped_f = cv2.remap(prev, map_fx, map_fy, cv2.INTER_LINEAR)
    warped_b = cv2.remap(curr, map_bx, map_by, cv2.INTER_LINEAR)
    return cv2.addWeighted(warped_f, 1.0 - alpha, warped_b, alpha, 0)


def _slow_motion_opencv(src: Path, *, factor: int, method: str, out_dir,
                        suffix: str | None, codec: str,
                        progress_cb) -> dict[str, Any]:
    cv2 = _cv2()
    np = _np()
    _, cap, info = _open_capture(src)
    out = _unique_output(src, Path(out_dir) if out_dir else src.parent / "edited",
                         suffix or f"slowmo-{factor}x", ".mp4")
    writer = None
    t0 = time.monotonic()
    try:
        ok, prev = cap.read()
        if not ok or prev is None:
            raise MediaEditError(f"video has no readable frames: {src}")
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        written = idx = 0
        while True:
            ok, curr = cap.read()
            if not ok or curr is None:
                break
            if writer is None:
                h, w = prev.shape[:2]
                writer = _new_writer(cv2, out, info["fps"] or 30.0,
                                     (w, h), codec)
            writer.write(prev)
            written += 1
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            for k in range(1, factor):
                alpha = k / factor
                if method == "flow":
                    mid = _flow_interpolate(cv2, np, prev, curr,
                                            prev_gray, curr_gray, alpha)
                else:
                    mid = cv2.addWeighted(prev, 1.0 - alpha,
                                          curr, alpha, 0)
                writer.write(mid)
                written += 1
            prev, prev_gray = curr, curr_gray
            idx += 1
            if progress_cb and info["frames"] and idx % 30 == 0:
                progress_cb(idx / info["frames"])
        # trailing frame
        if writer is None:
            raise MediaEditError(f"video has no readable frames: {src}")
        writer.write(prev)
        written += 1
        writer.release()
        if progress_cb:
            progress_cb(1.0)
        _log.info("slow_motion[opencv/%s]: %dx -> %s (%d frames, %.1fs)",
                  method, factor, out, written, time.monotonic() - t0)
        return {"input": str(src), "output": str(out), "factor": factor,
                "method": method, "frames_out": written,
                "duration_out": written / (info["fps"] or 30.0),
                "bytes": out.stat().st_size, "audio_dropped": True,
                "backend": "opencv"}
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def slow_motion(src: str | os.PathLike[str], *,
                factor: int = 2,
                method: str = "flow",
                out_dir: str | os.PathLike[str] | None = None,
                suffix: str | None = None,
                codec: str = "mp4v",
                backend: str = "auto",
                progress_cb: Callable[[float], None] | None = None
                ) -> dict[str, Any]:
    """Slow a video down by ``factor`` with motion-interpolated frames.

    Backend: ffmpeg primary — ``minterpolate`` with motion-compensated
    interpolation (``mci``/``aobmc``/``bidir``) plus a slowed audio track,
    so sound survives (``audio_dropped: False``). OpenCV fallback
    synthesizes the in-between frames itself: ``method="flow"`` warps
    both neighbors with dense DIS optical flow (best quality),
    ``method="blend"`` cross-dissolves (fast); the OpenCV writer is
    video-only so audio is dropped there. Pure-Python fallback
    (``backend="python"``): cross-dissolve blending through an
    ffmpeg rawvideo pipe — needs no cv2/numpy, slower per frame, audio
    dropped. ``"backend"``/``"method"`` say what ran.
    """
    if factor < 2:
        raise MediaEditError("slow-motion factor must be >= 2")
    if method not in ("flow", "blend"):
        raise MediaEditError("method must be 'flow' or 'blend'")
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    if _ffmpeg_has_filter("minterpolate"):
        want: tuple[str, ...] = ("ffmpeg", "opencv", "python")
    else:
        want = ("opencv", "python")
    chosen = _resolve_backend(backend, want)
    if chosen == "ffmpeg":
        quick = _probe_quick(src)
        return _slow_motion_ffmpeg(p, factor=factor, out_dir=out_dir,
                                   suffix=suffix, progress_cb=progress_cb,
                                   fps=quick["fps"],
                                   duration=quick["duration"])
    if chosen == "python":
        return _slow_motion_python(p, factor=factor, out_dir=out_dir,
                                   suffix=suffix, progress_cb=progress_cb)
    return _slow_motion_opencv(p, factor=factor, method=method,
                               out_dir=out_dir, suffix=suffix, codec=codec,
                               progress_cb=progress_cb)


def _slow_motion_python(src: Path, *, factor: int, out_dir,
                        suffix: str | None, progress_cb) -> dict[str, Any]:
    """Pure-Python slow motion: cross-dissolve blend via rawvideo pipes.

    Decodes rgb24 frames through ffmpeg, writes each frame plus
    ``factor - 1`` weighted blends between neighbors into an ffmpeg
    rawvideo encoder on stdin. Stdlib only — slower than OpenCV, but
    always works. Audio dropped (video-only pipeline).
    """
    from .videos import video_probe
    info = video_probe(src)
    fps = float(info.get("fps") or 25.0)
    w0 = int(info["width"])
    w, h = _pipe_frame_size(src, w0)
    out = _unique_output(src, Path(out_dir) if out_dir else src.parent / "edited",
                         suffix or f"slowmo-{factor}x-py", ".mp4")
    proc = _rawvideo_encoder(w, h, (fps or 25.0) * factor, out)
    t0 = time.monotonic()
    written = 0
    prev = None
    try:
        for buf in _iter_raw_frames(src, width=w0, pix_fmt="rgb24"):
            if prev is not None:
                for k in range(1, factor):
                    a = k / factor
                    blended = bytes(int(x + (y - x) * a)
                                    for x, y in zip(prev, buf, strict=True))
                    proc.stdin.write(blended)
                    written += 1
            proc.stdin.write(buf)
            written += 1
            prev = buf
            if progress_cb and written % 60 == 0:
                progress_cb(0.0)  # indeterminate total; keepalive
        proc.stdin.close()
    except BrokenPipeError as exc:  # pragma: no cover - defensive
        raise MediaEditError(f"ffmpeg encoder failed for {src}") from exc
    rc = proc.wait()
    if rc != 0:  # pragma: no cover - defensive
        raise MediaEditError(f"ffmpeg encoder exited with code {rc}")
    if progress_cb:
        progress_cb(1.0)
    _log.info("slow_motion[python]: %s -> %s (%d frames, %.1fs)",
              src, out, written, time.monotonic() - t0)
    return {"input": str(src), "output": str(out), "factor": factor,
            "method": "blend", "frames": written,
            "bytes": out.stat().st_size, "audio_dropped": True,
            "backend": "python"}


# ---------------------------------------------------------------------------
# reverse / boomerang
# ---------------------------------------------------------------------------

def reverse_video(src: str | os.PathLike[str], *,
                  out_dir: str | os.PathLike[str] | None = None,
                  suffix: str = "reversed",
                  codec: str = "mp4v",
                  backend: str = "auto",
                  progress_cb: Callable[[float], None] | None = None
                  ) -> dict[str, Any]:
    """Play a video backwards.

    Backend: ffmpeg primary — ``reverse`` + ``areverse`` filters, audio
    included (``audio_dropped: False``). OpenCV fallback reads frames back
    to front by index (video-only). ``"backend"`` says which ran.
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    chosen = _resolve_backend(backend, ("ffmpeg", "opencv"))
    if chosen == "ffmpeg":
        from .videos import run_ffmpeg, FFMPEG_TIMEOUT
        quick = _probe_quick(src)
        out = _unique_output(p, Path(out_dir) if out_dir else p.parent / "edited",
                             suffix, ".mp4")
        run_ffmpeg(["-i", str(p), "-vf", "reverse", "-af", "areverse",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", str(out)],
                   timeout=FFMPEG_TIMEOUT, progress_cb=progress_cb,
                   duration=quick["duration"] or None)
        return {"input": str(p), "output": str(out),
                "bytes": out.stat().st_size, "audio_dropped": False,
                "backend": "ffmpeg"}
    cv2, cap, info = _open_capture(src)
    out = _unique_output(p, Path(out_dir) if out_dir else p.parent / "edited",
                         suffix, ".mp4")
    writer = None
    try:
        n = info["frames"]
        for i in range(n - 1, -1, -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise MediaEditError(
                    f"could not decode frame {i} of {src}")
            if writer is None:
                h, w = frame.shape[:2]
                writer = _new_writer(cv2, out, info["fps"] or 30.0,
                                     (w, h), codec)
            writer.write(frame)
            if progress_cb and (n - 1 - i) % 60 == 0:
                progress_cb((n - 1 - i) / n)
        writer.release()
        if progress_cb:
            progress_cb(1.0)
        return {"input": str(p), "output": str(out), "frames": n,
                "bytes": out.stat().st_size, "audio_dropped": True,
                "backend": "opencv"}
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def boomerang(src: str | os.PathLike[str], *,
              start: str | float | int = 0,
              duration: str | float | int = 2,
              out_dir: str | os.PathLike[str] | None = None,
              suffix: str = "boomerang",
              backend: str = "auto",
              progress_cb: Callable[[float], None] | None = None
              ) -> dict[str, Any]:
    """Take a ``duration``-second slice at ``start`` and loop it
    forward-then-backward (the classic boomerang).

    ffmpeg-only under the hood (trim + reverse + concat in one filter
    graph); the audio slice is included forward+reversed when the source
    has an audio track. OpenCV has no muxer, so it cannot concat cleanly —
    ``backend="opencv"`` fails with that explanation instead of a broken
    file.
    """
    from .videos import parse_time, run_ffmpeg, FFMPEG_TIMEOUT, video_probe
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    s = parse_time(start)
    d = parse_time(duration)
    if d <= 0:
        raise MediaEditError("boomerang duration must be positive")
    chosen = _resolve_backend(backend, ("ffmpeg",))
    if chosen == "opencv":  # unreachable via want; explicit requests land here
        raise MediaEditError(  # pragma: no cover - defensive
            "boomerang needs muxed concat; OpenCV cannot mux. "
            "Use backend='ffmpeg'.")
    info = video_probe(p)
    has_audio = any(st.get("codec_type") == "audio"
                    for st in info.get("streams", []))
    e = s + d
    vgraph = (f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS,"
              f"split[f][fr];[fr]reverse[r];[f][r]concat=n=2:v=1:a=0[vout]")
    if has_audio:
        agraph = (f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS,"
                  f"asplit[a][ar];[ar]areverse[ra];"
                  f"[a][ra]concat=n=2:v=0:a=1[aout]")
        fc = f"{vgraph};{agraph}"
        maps = ["-map", "[vout]", "-map", "[aout]", "-c:a", "aac"]
    else:
        fc = vgraph
        maps = ["-map", "[vout]", "-an"]
    out = _unique_output(p, Path(out_dir) if out_dir else p.parent / "edited",
                         suffix, ".mp4")
    run_ffmpeg(["-i", str(p), "-filter_complex", fc, *maps,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                str(out)],
               timeout=FFMPEG_TIMEOUT, progress_cb=progress_cb,
               duration=d * 2 or None)
    return {"input": str(p), "output": str(out), "start": s,
            "duration": d, "loops": 2, "bytes": out.stat().st_size,
            "audio_dropped": not has_audio, "backend": "ffmpeg"}


# ---------------------------------------------------------------------------
# blurry-frame detection / motion heatmap (OpenCV-only, documented)
# ---------------------------------------------------------------------------

def _blurry_python(src: str | os.PathLike[str], *, threshold: float,
                   out_dir, thumb_width: int, analysis_width: int,
                   progress_cb) -> dict[str, Any]:
    """Pure-Python blurry-frame fallback: gradient-energy sharpness.

    Scores each frame by mean horizontal gradient energy (higher =
    sharper). ``threshold`` is interpreted relative to the median frame
    sharpness (default 100 = flag frames softer than the median); use
    smaller values to flag only the worst frames. Thumbnails come from
    ffmpeg at the flagged timestamps.
    """
    from .videos import video_probe
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    target = _frames_dir(p, out_dir, "blurry")
    info = video_probe(p)
    fps = float(info.get("fps") or 25.0)
    n_frames = int(info.get("frames") or 0)
    energies: list[float] = []
    idx = 0
    t0 = time.monotonic()
    for buf in _iter_raw_frames(p, width=analysis_width, pix_fmt="gray8"):
        energies.append(_py_gradient_energy(buf))
        idx += 1
        if progress_cb and n_frames and idx % 120 == 0:
            progress_cb(idx / n_frames)
    if not energies:
        raise MediaEditError(f"video has no readable frames: {src}")
    med = sorted(energies)[len(energies) // 2]
    cutoff = med * (threshold / 100.0)
    flagged = [i for i, e in enumerate(energies) if e < cutoff]
    worst_i = min(range(len(energies)), key=lambda i: energies[i])
    events: list[dict[str, Any]] = []
    worst = {"t": round(worst_i / fps, 2) if fps else 0.0,
             "score": round(energies[worst_i], 4), "thumb": ""}
    if flagged:
        times = [i / fps if fps else 0.0 for i in flagged]
        got = extract_frames(p, timestamps=times, out_dir=target,
                             fmt="jpg", backend="ffmpeg")
        for i, th in zip(flagged, got["frames"], strict=True):
            t = round(i / fps, 2) if fps else 0.0
            events.append({"t": t, "score": round(energies[i], 4),
                           "thumb": th})
            if i == worst_i:
                worst["thumb"] = th
    if progress_cb:
        progress_cb(1.0)
    _log.info("find_blurry_frames[python]: %d blurry in %s (%.1fs)",
              len(events), src, time.monotonic() - t0)
    return {"input": str(p), "events": events, "count": len(events),
            "thumbs_dir": str(target), "threshold": threshold,
            "frames_scanned": idx, "blurriest": worst,
            "backend": "python", "metric": "gradient-energy"}


def find_blurry_frames(src: str | os.PathLike[str], *,
                       threshold: float = 100.0,
                       out_dir: str | os.PathLike[str] | None = None,
                       thumb_width: int = 320,
                       analysis_width: int = 320,
                       backend: str = "auto",
                       progress_cb: Callable[[float], None] | None = None
                       ) -> dict[str, Any]:
    """Find blurry / out-of-focus frames via Laplacian variance.

    Each frame is scored with the variance of its Laplacian (higher =
    sharper); frames scoring below ``threshold`` are reported with a
    thumbnail. Useful for culling bad frames before a timelapse or
    highlight reel.

    Backend: OpenCV primary (Laplacian variance, ``threshold`` in those
    units). Pure-Python fallback (``backend="python"``): scores frames by
    mean gradient energy instead and interprets ``threshold`` relative to
    the median frame sharpness — same events contract, thumbnails via
    ffmpeg. ffmpeg has no per-frame sharpness reporting, so there is no
    pure-ffmpeg backend for this op.
    """
    if threshold <= 0:
        raise MediaEditError("threshold must be positive")
    chosen = _resolve_backend(backend, ("opencv", "python"))
    if chosen == "ffmpeg":
        raise MediaEditError(
            "find_blurry_frames needs per-frame sharpness scores; ffmpeg "
            "has no per-frame reporting (blurdetect only prints a stream "
            "summary). Use backend='opencv' or 'python'.")
    if chosen == "python":
        return _blurry_python(src, threshold=threshold, out_dir=out_dir,
                              thumb_width=thumb_width,
                              analysis_width=analysis_width,
                              progress_cb=progress_cb)
    cv2, cap, info = _open_capture(src)
    p = Path(src)
    target = _frames_dir(p, out_dir, "blurry")
    events: list[dict[str, Any]] = []
    worst = {"t": 0.0, "score": float("inf"), "thumb": ""}
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            small = frame
            if frame.shape[1] != analysis_width:
                small = cv2.resize(
                    frame, (analysis_width,
                            max(1, int(frame.shape[0] * analysis_width
                                       / frame.shape[1]))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            t = idx / info["fps"] if info["fps"] else 0.0
            if score < worst["score"]:
                worst = {"t": round(t, 2), "score": round(score, 2),
                         "thumb": ""}
            if score < threshold:
                thumb = _resize_keep_aspect(cv2, frame, thumb_width)
                tpath = target / f"blurry-{len(events):03d}-t{t:.2f}s.jpg"
                if not cv2.imwrite(str(tpath), thumb):
                    raise MediaEditError(f"could not write {tpath}")
                events.append({"t": round(t, 2), "score": round(score, 2),
                               "thumb": str(tpath)})
                if worst["thumb"] == "" and worst["t"] == round(t, 2):
                    worst["thumb"] = str(tpath)
            idx += 1
            if progress_cb and info["frames"] and idx % 120 == 0:
                progress_cb(idx / info["frames"])
        if progress_cb:
            progress_cb(1.0)
        return {"input": str(p), "events": events, "count": len(events),
                "thumbs_dir": str(target), "threshold": threshold,
                "frames_scanned": idx, "blurriest": worst,
                "backend": "opencv", "metric": "laplacian"}
    finally:
        cap.release()


def _heatmap_python(src: str | os.PathLike[str], *, out_dir,
                    decay: float, analysis_width: int,
                    progress_cb) -> dict[str, Any]:
    """Pure-Python motion heatmap: accumulate frame diffs via stdlib.

    Same accumulation semantics as the OpenCV path (exponential decay),
    color-mapped with a JET-like ramp and written as PNG by a minimal
    stdlib writer. Written at analysis resolution (no resampling without
    cv2/numpy).
    """
    from array import array
    from .videos import video_probe
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    info = video_probe(p)
    fps = float(info.get("fps") or 25.0)
    n_frames = int(info.get("frames") or 0)
    w, h = _pipe_frame_size(p, analysis_width)
    n = w * h
    acc = array("d", [0.0]) * n
    peak = {"t": 0.0, "score": 0.0}
    t0 = time.monotonic()
    prev = None
    idx = 0
    for buf in _iter_raw_frames(p, width=analysis_width, pix_fmt="gray8"):
        if prev is not None:
            total = 0.0
            for i, (x, y) in enumerate(zip(prev, buf, strict=True)):
                d = abs(x - y)
                acc[i] = acc[i] * decay + d
                total += d
            score = total / n
            if score > peak["score"]:
                peak = {"t": round(idx / fps, 2) if fps else 0.0,
                        "score": round(score, 4)}
        prev = buf
        idx += 1
        if progress_cb and n_frames and idx % 120 == 0:
            progress_cb(idx / n_frames)
    if idx < 2:
        raise MediaEditError(f"video has no readable frames: {src}")
    if progress_cb:
        progress_cb(1.0)
    mx = max(acc) or 1.0
    rgb = bytearray(n * 3)
    for i, v in enumerate(acc):
        r, g, b = _heat_color(v / mx)
        rgb[i * 3] = r
        rgb[i * 3 + 1] = g
        rgb[i * 3 + 2] = b
    out = _unique_output(p, Path(out_dir) if out_dir else p.parent / "edited",
                         "heatmap", ".png")
    _write_png(out, w, h, bytes(rgb))
    _log.info("motion_heatmap[python]: %s (%.1fs)", out,
              time.monotonic() - t0)
    return {"input": str(p), "heatmap": str(out),
            "peak_motion": peak, "frames_scanned": idx,
            "bytes": out.stat().st_size, "backend": "python"}


def motion_heatmap(src: str | os.PathLike[str], *,
                   out_dir: str | os.PathLike[str] | None = None,
                   decay: float = 0.95,
                   analysis_width: int = 320,
                   backend: str = "auto",
                   progress_cb: Callable[[float], None] | None = None
                   ) -> dict[str, Any]:
    """Build a heatmap of where motion happened across the whole video.

    Consecutive-frame differences are accumulated with exponential
    ``decay`` into a single image, color-mapped and saved as PNG —
    bright areas moved the most. Also reports the timestamp of peak
    motion.

    Backend: OpenCV primary (numpy accumulation, JET colormap, heatmap
    upscaled to source resolution). Pure-Python fallback
    (``backend="python"``): same accumulation in stdlib, JET-like ramp,
    PNG written by a minimal stdlib encoder at analysis resolution.
    ffmpeg cannot accumulate across frames, so there is no pure-ffmpeg
    backend for this op.
    """
    if not (0.0 < decay < 1.0):
        raise MediaEditError("decay must be within (0, 1)")
    chosen = _resolve_backend(backend, ("opencv", "python"))
    if chosen == "ffmpeg":
        raise MediaEditError(
            "motion_heatmap accumulates frame differences across time; "
            "ffmpeg cannot do that. Use backend='opencv' or 'python'.")
    if chosen == "python":
        return _heatmap_python(src, out_dir=out_dir, decay=decay,
                               analysis_width=analysis_width,
                               progress_cb=progress_cb)
    cv2 = _cv2()
    np = _np()
    _, cap, info = _open_capture(src)
    p = Path(src)
    try:
        ok, prev = cap.read()
        if not ok or prev is None:
            raise MediaEditError(f"video has no readable frames: {src}")
        h0, w0 = prev.shape[:2]
        aw = analysis_width
        ah = max(1, int(h0 * aw / w0))
        prev_small = cv2.cvtColor(cv2.resize(prev, (aw, ah)),
                                  cv2.COLOR_BGR2GRAY)
        acc = np.zeros((ah, aw), dtype=np.float32)
        peak = {"t": 0.0, "score": 0.0}
        idx = 1
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            small = cv2.cvtColor(cv2.resize(frame, (aw, ah)),
                                 cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_small, small).astype(np.float32)
            acc = acc * decay + diff
            score = float(diff.mean())
            if score > peak["score"]:
                t = idx / info["fps"] if info["fps"] else 0.0
                peak = {"t": round(t, 2), "score": round(score, 4)}
            prev_small = small
            idx += 1
            if progress_cb and info["frames"] and idx % 120 == 0:
                progress_cb(idx / info["frames"])
        if progress_cb:
            progress_cb(1.0)
        norm = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (w0, h0), interpolation=cv2.INTER_LINEAR)
        out = _unique_output(p, Path(out_dir) if out_dir else p.parent / "edited",
                             "heatmap", ".png")
        if not cv2.imwrite(str(out), heat):
            raise MediaEditError(f"could not write {out}")
        return {"input": str(p), "heatmap": str(out),
                "peak_motion": peak, "frames_scanned": idx,
                "bytes": out.stat().st_size, "backend": "opencv"}
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# scene splitting — detect (opencv/ffmpeg) + cut (ffmpeg)
# ---------------------------------------------------------------------------

def _scene_cuts_ffmpeg(src: Path, threshold: float) -> list[float]:
    """Cut timestamps via ffmpeg select='gt(scene,…)' + showinfo parsing."""
    exe = _ffmpeg_bin()
    if not exe:
        raise MediaEditError(_FFMPEG_INSTALL)
    proc = subprocess.run(
        [exe, "-hide_banner", "-i", str(src), "-vf",
         f"select='gt(scene,{threshold:.3f})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=600)
    cuts: list[float] = []
    import re
    # the select filter already dropped non-matching frames, so every
    # showinfo line here is a detected cut.
    for line in (proc.stderr or "").splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m and "showinfo" in line:
            with contextlib.suppress(ValueError):  # malformed pts_time
                cuts.append(float(m.group(1)))
    return sorted(cuts)


def split_on_scenes(src: str | os.PathLike[str], *,
                    threshold: float = 0.03,
                    min_gap: float = 0.5,
                    min_scene: float = 0.5,
                    out_dir: str | os.PathLike[str] | None = None,
                    backend: str = "auto",
                    progress_cb: Callable[[float], None] | None = None
                    ) -> dict[str, Any]:
    """Split a video into one file per scene at detected cuts.

    Detection backend: OpenCV primary (frame-diff scores, same engine as
    :func:`frame_diff_highlights`); ffmpeg ``select`` filter fallback when
    OpenCV is unavailable. Cutting is always ffmpeg (accurate re-encode;
    audio preserved per scene). ``min_scene`` merges cuts that would make
    shorter scenes. ``"backend"`` reports the detection backend.
    """
    if not (0.0 < threshold < 1.0):
        raise MediaEditError("threshold must be within (0, 1)")
    if min_scene < 0:
        raise MediaEditError("min_scene must be >= 0")
    from .videos import run_ffmpeg, FFMPEG_TIMEOUT
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    detect = _resolve_backend(backend, ("opencv", "ffmpeg"))
    if detect == "opencv":
        hl = frame_diff_highlights(src, threshold=threshold,
                                   min_gap=min_gap, out_dir=None,
                                   backend="opencv")
        cuts = [e["t"] for e in hl["events"]]
    else:
        cuts = _scene_cuts_ffmpeg(p, threshold)
    duration = _probe_quick(src)["duration"]
    bounds = [0.0] + [c for c in cuts if 0 < c < duration] + [duration]
    # merge scenes shorter than min_scene into the next one
    scenes: list[tuple[float, float]] = []
    i = 0
    while i < len(bounds) - 1:
        s, e = bounds[i], bounds[i + 1]
        while e - s < min_scene and i + 2 < len(bounds):
            i += 1
            e = bounds[i + 1]
        scenes.append((s, e))
        i += 1
    target = _frames_dir(p, out_dir, "scenes")
    outputs: list[dict[str, Any]] = []
    for n, (s, e) in enumerate(scenes):
        out = target / f"scene-{n:02d}-{s:.1f}s-{e:.1f}s.mp4"
        run_ffmpeg(["-ss", str(s), "-t", str(max(0.1, e - s)),
                    "-i", str(p),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", str(out)],
                   timeout=FFMPEG_TIMEOUT,
                   progress_cb=(lambda f, _n=n, _s=len(scenes):
                                progress_cb((_n + f) / _s)
                                if progress_cb else None),
                   duration=(e - s) or None)
        outputs.append({"n": n, "start": round(s, 2), "end": round(e, 2),
                        "output": str(out),
                        "bytes": out.stat().st_size})
    if progress_cb:
        progress_cb(1.0)
    return {"input": str(p), "scenes": outputs, "count": len(outputs),
            "scenes_dir": str(target), "threshold": threshold,
            "backend": detect}


# ---------------------------------------------------------------------------
# Ken Burns / slideshow / chroma key / PiP / freeze frame / denoise
# (ffmpeg-primary: these are filter-graph ops where ffmpeg is the best
# free tool; OpenCV fallbacks exist only where frame access adds value)
# ---------------------------------------------------------------------------

def _ff_encode_video(args: list, out: Path, duration: float | None,
                     progress_cb, has_audio_input: bool = True,
                     audio_args: list | None = None,
                     timeout: float | None = None) -> None:
    from .videos import run_ffmpeg, FFMPEG_TIMEOUT
    cmd = args + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                  "-pix_fmt", "yuv420p"]
    cmd += audio_args if audio_args is not None else (
        ["-c:a", "aac"] if has_audio_input else ["-an"])
    cmd.append(str(out))
    run_ffmpeg(cmd, timeout=timeout or FFMPEG_TIMEOUT,
               progress_cb=progress_cb, duration=duration or None)


def kenburns(image: str | os.PathLike[str], *,
             duration: str | float | int = 5,
             zoom_from: float = 1.0,
             zoom_to: float = 1.5,
             pan: str = "center",
             fps: int = 30,
             width: int = 1280,
             height: int = 720,
             out_dir: str | os.PathLike[str] | None = None,
             suffix: str = "kenburns",
             backend: str = "auto",
             progress_cb: Callable[[float], None] | None = None
             ) -> dict[str, Any]:
    """Animate a still image with a slow zoom/pan (Ken Burns effect).

    Backend: ffmpeg ``zoompan`` — the best free tool for this; there is
    no higher-quality frame-based alternative, so ``backend="opencv"``
    fails with that explanation rather than a worse result.
    ``pan``: center|left|right|up|down. zoom_from < zoom_to zooms in,
    the reverse zooms out.
    """
    from .videos import parse_time
    p = Path(image)
    if not p.exists():
        raise MediaEditError(f"no such image: {image}")
    d = parse_time(duration)
    if d <= 0:
        raise MediaEditError("kenburns duration must be positive")
    if fps < 1 or fps > 120:
        raise MediaEditError("fps must be within [1, 120]")
    if pan not in ("center", "left", "right", "up", "down"):
        raise MediaEditError("pan must be center|left|right|up|down")
    chosen = _resolve_backend(backend, ("ffmpeg",))
    if chosen == "opencv":  # explicit only; want has no opencv entry
        raise MediaEditError(  # pragma: no cover - defensive
            "kenburns needs the zoompan filter; OpenCV has no equivalent "
            "smoother than ffmpeg here. Use backend='ffmpeg'.")
    frames = max(2, int(round(d * fps)))
    zexpr = f"{zoom_from}+({zoom_to}-{zoom_from})*on/{frames}"
    if pan == "left":
        xexpr, yexpr = f"(iw-iw/zoom)*on/{frames}", "ih/2-(ih/zoom/2)"
    elif pan == "right":
        xexpr, yexpr = f"(iw-iw/zoom)*(1-on/{frames})", "ih/2-(ih/zoom/2)"
    elif pan == "up":
        xexpr, yexpr = "iw/2-(iw/zoom/2)", f"(ih-ih/zoom)*on/{frames}"
    elif pan == "down":
        xexpr, yexpr = "iw/2-(iw/zoom/2)", f"(ih-ih/zoom)*(1-on/{frames})"
    else:
        xexpr, yexpr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    vf = (f"scale=-2:{height * 2},zoompan=z='{zexpr}':x='{xexpr}':y='{yexpr}':"
          f"d={frames}:s={width}x{height}:fps={fps}")
    out = _unique_output(p, Path(out_dir) if out_dir else p.parent / "edited",
                         suffix, ".mp4")
    _ff_encode_video(["-loop", "1", "-i", str(p), "-vf", vf,
                      "-frames:v", str(frames)],
                     out, d, progress_cb, has_audio_input=False)
    return {"input": str(p), "output": str(out), "duration": d,
            "zoom": (zoom_from, zoom_to), "pan": pan, "fps": fps,
            "size": (width, height), "bytes": out.stat().st_size,
            "backend": "ffmpeg"}


def slideshow(images: list[str | os.PathLike[str]], *,
              duration_each: str | float | int = 3,
              transition: str = "fade",
              transition_duration: float = 0.5,
              fps: int = 30, width: int = 1280, height: int = 720,
              out_dir: str | os.PathLike[str] | None = None,
              suffix: str = "slideshow",
              backend: str = "auto",
              progress_cb: Callable[[float], None] | None = None
              ) -> dict[str, Any]:
    """Build a video slideshow from still images with transitions.

    Each image shows for ``duration_each`` seconds; consecutive clips are
    joined with an ``xfade`` ``transition`` (fade, wipeleft, slideup,
    circleopen, … — any ffmpeg xfade transition). Audio: none (stills
    have no soundtrack).

    Backend: ffmpeg only — xfade chains need a filter graph; OpenCV has
    no muxer.
    """
    from .videos import parse_time
    if len(images) < 1:
        raise MediaEditError("slideshow needs at least 1 image")
    for im in images:
        if not Path(im).exists():
            raise MediaEditError(f"no such image: {im}")
    d = parse_time(duration_each)
    if d <= 0:
        raise MediaEditError("duration_each must be positive")
    if not (0 <= transition_duration < d):
        raise MediaEditError(
            "transition_duration must be within [0, duration_each)")
    chosen = _resolve_backend(backend, ("ffmpeg",))
    if chosen == "opencv":  # explicit only
        raise MediaEditError(  # pragma: no cover - defensive
            "slideshow needs muxed xfade concat; OpenCV cannot mux. "
            "Use backend='ffmpeg'.")
    n = len(images)
    inputs: list[str] = []
    for im in images:
        inputs += ["-loop", "1", "-t", str(d), "-i", str(im)]
    # normalize all inputs, then xfade chain
    fc_parts = []
    for i in range(n):
        fc_parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio="
            f"decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,fps={fps},format=yuv420p[v{i}]")
    if n == 1:
        fc_parts.append("[v0]null[outv]")
    else:
        td = transition_duration
        prev = "[v0][v1]xfade=transition=" + transition + \
            f":duration={td}:offset={d - td}[x1]"
        fc_parts.append(prev)
        for k in range(2, n):
            off = k * d - k * td
            fc_parts.append(
                f"[x{k - 1}][v{k}]xfade=transition={transition}:"
                f"duration={td}:offset={off}[x{k}]")
        fc_parts.append(f"[x{n - 1}]null[outv]")
    total = n * d - (n - 1) * transition_duration
    src0 = Path(images[0])
    out = _unique_output(src0, Path(out_dir) if out_dir else src0.parent / "edited",
                         suffix, ".mp4")
    _ff_encode_video(inputs + ["-filter_complex", ";".join(fc_parts),
                               "-map", "[outv]"],
                     out, total, progress_cb, has_audio_input=False)
    return {"input": [str(Path(i)) for i in images], "output": str(out),
            "images": n, "duration_each": d, "transition": transition,
            "duration": total, "bytes": out.stat().st_size,
            "backend": "ffmpeg"}


def chroma_key(src: str | os.PathLike[str],
               background: str | os.PathLike[str], *,
               color: str = "green",
               similarity: float = 0.3,
               blend: float = 0.1,
               out_dir: str | os.PathLike[str] | None = None,
               suffix: str = "chroma",
               backend: str = "auto",
               progress_cb: Callable[[float], None] | None = None
               ) -> dict[str, Any]:
    """Green-screen: key out ``color`` from a video and composite over
    ``background`` (image or video).

    ``color``: "green"|"blue"|"red" or a hex like "00ff00".
    ``similarity`` (0..1): how close a pixel must be to key out;
    ``blend`` (0..1): edge softness.

    Backend: ffmpeg ``chromakey`` — the best free option; OpenCV has no
    muxer for the composite, so ``backend="opencv"`` fails honestly.
    """
    from .videos import video_probe
    p = Path(src)
    bg = Path(background)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    if not bg.exists():
        raise MediaEditError(f"no such background: {background}")
    _check_size(p)
    colors = {"green": "0x00FF00", "blue": "0x0000FF", "red": "0xFF0000"}
    key = colors.get(color.lower(), color)
    if not key.startswith("0x"):
        key = "0x" + key.lstrip("#")
    if not (0.0 <= similarity <= 1.0 and 0.0 <= blend <= 1.0):
        raise MediaEditError("similarity and blend must be within [0, 1]")
    chosen = _resolve_backend(backend, ("ffmpeg",))
    if chosen == "opencv":  # explicit only
        raise MediaEditError(  # pragma: no cover - defensive
            "chroma key compositing needs a muxer; OpenCV cannot mux. "
            "Use backend='ffmpeg'.")
    info = video_probe(p)
    w, h = info.get("width") or 1280, info.get("height") or 720
    duration = info.get("duration") or 0
    bg_is_video = bg.suffix.lower() in (
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
    if bg_is_video:
        bg_input = ["-i", str(bg)]
    elif duration > 0:
        # image bg: loop it, but bound to the foreground duration —
        # an unbounded -loop 1 makes overlay's EOF-repeat never end
        bg_input = ["-loop", "1", "-t", f"{duration:.3f}", "-i", str(bg)]
    else:  # pragma: no cover - defensive
        bg_input = ["-loop", "1", "-i", str(bg)]
    fc = (f"[1:v]scale={w}:{h},format=yuv420p[bg];"
          f"[0:v]chromakey={key}:{similarity:.3f}:{blend:.3f}[fg];"
          f"[bg][fg]overlay=0:0:format=yuv420[outv]")
    out = _unique_output(p, Path(out_dir) if out_dir else p.parent / "edited",
                         suffix, ".mp4")
    _ff_encode_video(["-i", str(p), *bg_input,
                      "-filter_complex", fc, "-map", "[outv]",
                      "-map", "0:a?", "-shortest"],
                     out, duration, progress_cb, has_audio_input=True,
                     audio_args=["-c:a", "aac"])
    return {"input": str(p), "background": str(bg), "output": str(out),
            "color": key, "similarity": similarity, "blend": blend,
            "bytes": out.stat().st_size, "backend": "ffmpeg"}


_PIP_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right",
                  "center")


def pip(main: str | os.PathLike[str], overlay: str | os.PathLike[str], *,
        position: str = "bottom-right", scale: float = 0.25,
        margin: int = 20,
        out_dir: str | os.PathLike[str] | None = None,
        suffix: str = "pip",
        backend: str = "auto",
        progress_cb: Callable[[float], None] | None = None
        ) -> dict[str, Any]:
    """Picture-in-picture: overlay a scaled video/image onto ``main``.

    ``position``: top-left|top-right|bottom-left|bottom-right|center.
    ``scale``: overlay width as a fraction of main width (0.05..0.9).

    Backend: ffmpeg ``overlay`` — best free tool; OpenCV cannot mux.
    """
    from .videos import video_probe
    a, b = Path(main), Path(overlay)
    if not a.exists():
        raise MediaEditError(f"no such video: {main}")
    if not b.exists():
        raise MediaEditError(f"no such overlay: {overlay}")
    _check_size(a)
    if position not in _PIP_POSITIONS:
        raise MediaEditError(f"position must be one of {_PIP_POSITIONS}")
    if not (0.05 <= scale <= 0.9):
        raise MediaEditError("scale must be within [0.05, 0.9]")
    chosen = _resolve_backend(backend, ("ffmpeg",))
    if chosen == "opencv":  # explicit only
        raise MediaEditError(  # pragma: no cover - defensive
            "picture-in-picture needs a muxer; OpenCV cannot mux. "
            "Use backend='ffmpeg'.")
    info = video_probe(a)
    duration = info.get("duration") or 0
    pos = {"top-left": f"{margin}:{margin}",
           "top-right": f"W-w-{margin}:{margin}",
           "bottom-left": f"{margin}:H-h-{margin}",
           "bottom-right": f"W-w-{margin}:H-h-{margin}",
           "center": "(W-w)/2:(H-h)/2"}[position]
    b_is_video = b.suffix.lower() in (
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
    ov_input = ["-i", str(b)] if b_is_video else ["-loop", "1", "-i", str(b)]
    fc = (f"[1:v]scale=iw*{scale:.3f}:-2,format=yuv420p[pip];"
          f"[0:v][pip]overlay={pos}:format=yuv420[outv]")
    out = _unique_output(a, Path(out_dir) if out_dir else a.parent / "edited",
                         suffix, ".mp4")
    _ff_encode_video(["-i", str(a), *ov_input, "-filter_complex", fc,
                      "-map", "[outv]", "-map", "0:a?",
                      "-shortest"],
                     out, duration, progress_cb, has_audio_input=True,
                     audio_args=["-c:a", "aac"])
    return {"input": str(a), "overlay": str(b), "output": str(out),
            "position": position, "scale": scale,
            "bytes": out.stat().st_size, "backend": "ffmpeg"}


def freeze_frame(src: str | os.PathLike[str], *,
                 timestamp: str | float | int = 0,
                 duration: str | float | int = 2,
                 out_dir: str | os.PathLike[str] | None = None,
                 suffix: str = "freeze",
                 backend: str = "auto",
                 progress_cb: Callable[[float], None] | None = None
                 ) -> dict[str, Any]:
    """Hold the frame at ``timestamp`` as a still for ``duration`` seconds.

    Backend: ffmpeg (extract frame → loop). Frame selection is exact via
    OpenCV when available, falling back to ffmpeg seeking.
    """
    from .videos import parse_time
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    d = parse_time(duration)
    if d <= 0:
        raise MediaEditError("freeze duration must be positive")
    chosen = _resolve_backend(backend, ("ffmpeg", "opencv"))
    work = Path(tempfile.mkdtemp(prefix="freeze-"))
    try:
        frame_png = work / "frame.png"
        if chosen == "opencv":
            cv2 = _cv2()
            frame, _ = grab_frame_at(src, timestamp)
            if not cv2.imwrite(str(frame_png), frame):
                raise MediaEditError("could not extract freeze frame")
        else:
            from .videos import run_ffmpeg, FFMPEG_TIMEOUT
            t = parse_time(timestamp)
            run_ffmpeg(["-ss", str(t), "-i", str(p), "-frames:v", "1",
                        str(frame_png)], timeout=min(FFMPEG_TIMEOUT, 120))
        out = _unique_output(p, Path(out_dir) if out_dir else p.parent / "edited",
                             suffix, ".mp4")
        _ff_encode_video(["-loop", "1", "-framerate", "30",
                          "-i", str(frame_png), "-t", str(d)],
                         out, d, progress_cb, has_audio_input=False)
        return {"input": str(p), "output": str(out),
                "timestamp": timestamp, "duration": d,
                "bytes": out.stat().st_size, "backend": chosen,
                "audio_dropped": True}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def denoise_video(src: str | os.PathLike[str], *,
                  strength: float = 1.0,
                  out_dir: str | os.PathLike[str] | None = None,
                  suffix: str | None = None,
                  codec: str = "mp4v",
                  backend: str = "auto",
                  progress_cb: Callable[[float], None] | None = None
                  ) -> dict[str, Any]:
    """Reduce sensor/grain noise across all frames.

    Backend: ffmpeg ``hqdn3d`` primary (fast, audio preserved).
    OpenCV fallback uses ``fastNlMeansDenoisingColored`` per frame
    (slower, video-only). ``strength`` 0..2.
    """
    if not (0.0 <= strength <= 2.0):
        raise MediaEditError("strength must be within [0, 2]")
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    chosen = _resolve_backend(backend, ("ffmpeg", "opencv"))
    if chosen == "ffmpeg":
        s = max(0.0, strength)
        vf = (f"hqdn3d={1 + 2 * s:.1f}:{1 + 1.5 * s:.1f}:"
              f"{2 + 3 * s:.1f}:{2 + 2 * s:.1f}")
        out = _unique_output(p, Path(out_dir) if out_dir else p.parent / "edited",
                             suffix or "denoised", ".mp4")
        quick = _probe_quick(src)
        _ff_encode_video(["-i", str(p), "-vf", vf], out,
                         quick["duration"], progress_cb,
                         has_audio_input=True)
        return {"input": str(p), "output": str(out), "strength": strength,
                "bytes": out.stat().st_size, "audio_dropped": False,
                "backend": "ffmpeg"}
    cv2, cap, info = _open_capture(src)
    out = _unique_output(p, Path(out_dir) if out_dir else p.parent / "edited",
                         suffix or "denoised", ".mp4")
    writer = None
    try:
        h0 = 3 + int(strength * 4)  # 3..11 filter window
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = _new_writer(cv2, out, info["fps"] or 30.0,
                                     (w, h), codec)
            clean = cv2.fastNlMeansDenoisingColored(frame, None, h0, h0,
                                                    7, 21)
            writer.write(clean if clean is not None else frame)
            written += 1
            if progress_cb and written % 30 == 0 and info["frames"]:
                progress_cb(written / info["frames"])
        if writer is None:
            raise MediaEditError(f"video has no readable frames: {src}")
        writer.release()
        if progress_cb:
            progress_cb(1.0)
        return {"input": str(p), "output": str(out), "strength": strength,
                "frames": written, "bytes": out.stat().st_size,
                "audio_dropped": True, "backend": "opencv"}
    finally:
        cap.release()
        if writer is not None:
            writer.release()


__all__ = [
    "cv2_available",
    "extract_frames",
    "grab_frame_at",
    "apply_filter_to_video",
    "create_timelapse",
    "stabilize_basic",
    "frame_diff_highlights",
    "slow_motion",
    "reverse_video",
    "boomerang",
    "find_blurry_frames",
    "motion_heatmap",
    "split_on_scenes",
    "kenburns",
    "slideshow",
    "chroma_key",
    "pip",
    "freeze_frame",
    "denoise_video",
    "FILTERS",
    "FILTER_BACKENDS",
]
