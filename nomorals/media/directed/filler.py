"""img2img filler — creates what isn't there.

- fill_frames(before, after, n): in-between frames for a gap (eased
  cross-dissolve morph — real transitional content, honestly labeled).
- extend_background(image, target_w, target_h): wider canvas.
  Neural outpaint when the SD pipeline is available; mirror-pad +
  blur-blend photographic extension on CPU.
- fill_video_gap(video, t0, t1): repair a dropped segment by
  synthesizing bridge frames.

From EXPANSION_MINING.md §2.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageFilter


def _ffmpeg():
    from shutil import which
    ff = which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found")
    return ff


def _ease(t: float) -> float:
    """Smoothstep — motion eases in/out instead of linear sliding."""
    return t * t * (3 - 2 * t)


def fill_frames(before: Image.Image, after: Image.Image, n: int) -> list[Image.Image]:
    """Synthesize n in-between frames between two stills.

    Eased cross-dissolve with a subtle scale drift (Ken Burns-ish) so
    the bridge has motion, not just fading. Honest label: morph blend,
    not optical flow.
    """
    if n <= 0:
        return []
    size = before.size
    a = before.resize(size).convert("RGB")
    b = after.resize(size).convert("RGB")
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
