"""img2img filler — creates what isn't there.

- fill_frames(before, after, n): in-between frames for a gap.
  Backend routing is honest and best-available:
    1. RIFE neural interpolation when rife-ncnn-vulkan (or a pip
       ``rife`` package) is present — real optical-flow interpolation;
    2. motion-compensated CPU blend: global motion estimated with phase
       correlation, both frames warped toward the middle, then blended —
       beats plain crossfade on pans/zooms;
    3. eased cross-dissolve morph — real transitional content, honestly
       labeled (used only when numpy is unavailable).
- extend_background(image, target_w, target_h): wider canvas.
  Neural outpaint when the SD pipeline is available; mirror-pad +
  blur-blend photographic extension on CPU.
- fill_video_gap(video, t0, t1): repair a dropped segment by
  synthesizing bridge frames.

From EXPANSION_MINING.md §2.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path
from shutil import which

from PIL import Image, ImageFilter

from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "fill_frames",
    "extend_background",
    "fill_video_gap",
    "interpolation_backend",
]


def _ffmpeg():
    ff = which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found")
    return ff


def _ease(t: float) -> float:
    """Smoothstep — motion eases in/out instead of linear sliding."""
    return t * t * (3 - 2 * t)


def _numpy_available() -> bool:
    return importlib.util.find_spec("numpy") is not None


def _rife_available() -> bool:
    """RIFE neural frame interpolation present?

    Either the ``rife-ncnn-vulkan`` binary (nihui's ncnn build — the
    standard CPU/GPU route) or a pip ``rife`` package.
    """
    return bool(which("rife-ncnn-vulkan")) or \
        importlib.util.find_spec("rife") is not None


def interpolation_backend() -> str:
    """Which interpolation fill_frames() will use right now.

    "rife" (neural) > "motion-compensated" (numpy CPU) > "morph-blend"
    (PIL-only fallback). Always honest about what's actually installed.
    """
    if _rife_available():
        return "rife"
    if _numpy_available():
        return "motion-compensated"
    return "morph-blend"


def fill_frames(before: Image.Image, after: Image.Image, n: int, *,
                backend: str = "auto") -> list[Image.Image]:
    """Synthesize n in-between frames between two stills.

    ``backend``: "auto" | "rife" | "motion-compensated" | "morph-blend".
    Auto routes to the best installed backend — RIFE neural
    interpolation when available, otherwise the motion-compensated CPU
    blend (phase-correlation global motion + warp), otherwise the eased
    cross-dissolve morph. A failing backend degrades to the next one;
    the fallback is never mislabeled as optical flow.
    """
    if n <= 0:
        return []
    size = before.size
    a = before.resize(size).convert("RGB")
    b = after.resize(size).convert("RGB")
    want = (backend or "auto").lower()
    if want in ("auto", "rife"):
        if _rife_available() and n <= 8:
            try:
                return _rife_bridge(a, b, n)
            except Exception as exc:  # noqa: BLE001 - degrade honestly
                _log.info("filler: RIFE failed (%s) — falling back", exc)
        elif want == "rife" and n > 8:
            _log.info("filler: RIFE handles ≤8 in-betweens per pair — "
                      "using CPU for n=%d", n)
    if want in ("auto", "motion-compensated", "motion"):
        if _numpy_available():
            try:
                return _motion_compensated_frames(a, b, n)
            except Exception as exc:  # noqa: BLE001 - degrade honestly
                _log.info("filler: motion-compensated failed (%s) — "
                          "falling back", exc)
    return _morph_blend_frames(a, b, n)


def _rife_bridge(a: Image.Image, b: Image.Image, n: int) -> list[Image.Image]:
    """RIFE neural interpolation via rife-ncnn-vulkan.

    rife-ncnn-vulkan inserts 1/2/4/8 frames between each pair; we run
    one pass with the smallest step count covering ``n`` and sample
    evenly. Any failure raises — the caller falls back, it never
    pretends a RIFE frame was made.
    """
    size = a.size
    wd = Path(tempfile.mkdtemp(prefix="rife_"))
    ff = _ffmpeg()
    bin_path = which("rife-ncnn-vulkan")
    if not bin_path:
        raise RuntimeError("rife-ncnn-vulkan not installed")
    a.save(wd / "f0.png")
    b.save(wd / "f1.png")
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", "24", "-i", str(wd / "f%d.png"),
         "-pix_fmt", "yuv420p", str(wd / "pair.mp4")],
        check=True, capture_output=True, timeout=60)
    steps = next(s for s in (1, 2, 4, 8) if s >= n)
    subprocess.run(
        [bin_path, "-i", str(wd / "pair.mp4"), "-o", str(wd / "out.mp4"),
         "-n", str(steps), "-v"],
        check=True, capture_output=True, timeout=300)
    # out.mp4 = a, steps mids, b → extract the mids
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(wd / "out.mp4"), str(wd / "o_%03d.png")],
        check=True, capture_output=True, timeout=60)
    got = steps  # mids produced by this one pass
    frames = []
    for i in range(1, n + 1):
        idx = min(got, max(1, round(i * (got + 1) / (n + 1))))
        img = Image.open(wd / f"o_{idx:03d}.png").convert("RGB")
        if img.size != size:
            img = img.resize(size, Image.BICUBIC)
        frames.append(img)
    return frames


def _phase_shift(a: "np.ndarray", b: "np.ndarray") -> tuple[float, float]:
    """Global translation of ``b`` relative to ``a`` via phase
    correlation (numpy FFT, no OpenCV needed). Returns (dx, dy).

    Phase correlation wraps shifts larger than half the window
    (+40px reads as −24px on a 64px window), so the raw peak is
    validated against its wrapped candidates and the zero-shift by
    direct SSD — the winner is the actual scene motion. A spurious
    shift (worse than standing still) degrades to (0, 0), i.e. a
    plain blend.
    """
    import numpy as np
    fa = np.fft.fft2(a.astype(np.float64))
    fb = np.fft.fft2(b.astype(np.float64))
    cross = fa * np.conj(fb)
    cross /= (np.abs(cross) + 1e-8)
    corr = np.fft.ifft2(cross).real
    y, x = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    # Shift theorem (numpy FFT sign): b(x) = a(x - d) puts the peak at
    # p = -d (mod N), so d = -p wrapped into [-N/2, N/2).
    px, py = (-x) % w, (-y) % h
    dx = float(px if px <= w / 2 else px - w)
    dy = float(py if py <= h / 2 else py - h)

    def ssd(shift: tuple[float, float]) -> float:
        # Hypothesis "b = a shifted by (dx, dy)": warp b back by the
        # shift and compare with a. PIL transform does NOT wrap edges
        # (unlike np.roll), so a wrapped-equivalent shift scores badly
        # here — exactly the tie-break we need.
        sx, sy = shift
        moved = Image.fromarray(b).transform(
            (w, h), Image.AFFINE, (1, 0, sx, 0, 1, sy), Image.BICUBIC)
        diff = a.astype(np.float64) - np.asarray(moved, dtype=np.float64)
        return float((diff ** 2).mean())

    cands = [(0.0, 0.0), (dx, dy),
             (dx + w, dy), (dx - w, dy), (dx, dy + h), (dx, dy - h)]
    scored = sorted(((ssd(c), c) for c in cands), key=lambda t: t[0])
    best_ssd, best = scored[0]
    zero_ssd = ssd((0.0, 0.0))
    if best_ssd >= 0.85 * zero_ssd:
        return 0.0, 0.0
    return float(best[0]), float(best[1])


def _motion_compensated_frames(a: Image.Image, b: Image.Image,
                               n: int) -> list[Image.Image]:
    """CPU motion-compensated interpolation.

    Estimates the dominant global shift between the frames with phase
    correlation, warps each frame toward the middle position, then
    blends. On pans, tilts and zooms this keeps structure intact where
    a crossfade would ghost; on static frames it degenerates to a plain
    blend — exactly what it should.
    """
    import numpy as np
    size = a.size
    small = 64
    ga = np.asarray(a.convert("L").resize((small, small)))
    gb = np.asarray(b.convert("L").resize((small, small)))
    dx, dy = _phase_shift(ga, gb)
    sx = dx * size[0] / small
    sy = dy * size[1] / small
    out = []
    for i in range(1, n + 1):
        t = i / (n + 1)
        # warp a forward by t*shift, b backward by (1-t)*shift, blend
        wa = a.transform(size, Image.AFFINE, (1, 0, -sx * t, 0, 1, -sy * t),
                         Image.BICUBIC)
        wb = b.transform(size, Image.AFFINE,
                         (1, 0, sx * (1 - t), 0, 1, sy * (1 - t)),
                         Image.BICUBIC)
        out.append(Image.blend(wa, wb, t))
    return out


def _morph_blend_frames(a: Image.Image, b: Image.Image,
                        n: int) -> list[Image.Image]:
    """Eased cross-dissolve with a subtle scale drift (Ken Burns-ish).

    The honest baseline — labeled morph blend, not optical flow.
    """
    size = a.size
    out = []
    for i in range(1, n + 1):
        t = _ease(i / (n + 1))
        # scale drift: zoom 1.0 -> 1.04 across the bridge for motion feel
        z = 1.0 + 0.04 * t
        zw, zh = int(size[0] * z), int(size[1] * z)
        az = a.resize((zw, zh), Image.BICUBIC)
        bz = b.resize((zw, zh), Image.BICUBIC)
        cx, cy = (zw - size[0]) // 2, (zh - size[1]) // 2
        frame = Image.blend(az.crop((cx, cy, cx + size[0], cy + size[1])),
                            bz.crop((cx, cy, cx + size[0], cy + size[1])), t)
        out.append(frame)
    return out


def extend_background(image: Image.Image, target_w: int, target_h: int,
                      *, prompt: str = "", pipeline=None) -> tuple[Image.Image, str]:
    """Extend the canvas to target_w x target_h.

    Neural outpaint when ``pipeline`` (or a found SD checkpoint) is
    available; otherwise mirror-pad + blur-blend (standard photographic
    extension). Returns (image, backend_label).
    """
    w, h = image.size
    if target_w <= w and target_h <= h:
        return image.copy(), "noop"
    pipe = pipeline or _find_pipeline()
    if pipe is not None:
        try:
            from ..imggen.edit import outpaint as _neural_outpaint
            left = (target_w - w) // 2
            top = (target_h - h) // 2
            right = target_w - w - left
            bottom = target_h - h - top
            out = _neural_outpaint(
                pipe, image, prompt or "seamless extended background",
                left=left, right=right, top=top, bottom=bottom)[0]
            return out, "neural-outpaint"
        except Exception:
            pass
    # CPU: mirror pad then blur the seam
    ox, oy = (target_w - w) // 2, (target_h - h) // 2
    import numpy as np
    arr = np.array(image)
    pad_x = (ox, target_w - w - ox)
    pad_y = (oy, target_h - h - oy)
    big = (np.pad(arr, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1]), (0, 0)),
                  mode="reflect")
           if (any(pad_x) or any(pad_y)) else arr)
    canvas = Image.fromarray(big)
    soft = canvas.filter(ImageFilter.GaussianBlur(3))
    mask = Image.new("L", (target_w, target_h), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask).rectangle([ox, oy, ox + w, oy + h], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(12))
    return Image.composite(canvas, soft, mask), "mirror-pad"


def _find_pipeline():
    """Locate a usable SD pipeline: env var, then common paths."""
    import os
    from pathlib import Path as _P
    cand = os.environ.get("DEVON_SD_CKPT", "")
    paths = ([cand] if cand else []) + [
        str(_P.home() / ".devon-models" / "sd15" / "v1-5-pruned.safetensors"),
        str(_P.home() / ".devon-models" / "sd15" / "model.safetensors"),
    ]
    for p in paths:
        if p and os.path.exists(p):
            try:
                from ..imggen.pipeline import load_native_checkpoint
                return load_native_checkpoint(p)
            except Exception:
                return None
    return None


def fill_video_gap(video: str, t0: float, t1: float, fps: int = 24, *,
                   out_path: str | None = None) -> str:
    """Repair a dropped segment [t0, t1] with synthesized bridge frames."""
    ff = _ffmpeg()
    wd = Path(tempfile.mkdtemp(prefix="gap_"))
    # grab boundary frames
    for i, t in enumerate((t0, t1)):
        subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-y",
             "-ss", str(t), "-i", video, "-frames:v", "1",
             str(wd / f"b{i}.png")],
            check=True, capture_output=True, timeout=120)
    before = Image.open(wd / "b0.png")
    after = Image.open(wd / "b1.png")
    n = max(1, int((t1 - t0) * fps))
    bridge = fill_frames(before, after, n)
    for i, f in enumerate(bridge):
        f.save(wd / f"g_{i:04d}.png")
    out = out_path or str(wd / "gapfill.mp4")
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", str(fps), "-i", str(wd / "g_%04d.png"),
         "-pix_fmt", "yuv420p", out],
        check=True, capture_output=True, timeout=300)
    return out
