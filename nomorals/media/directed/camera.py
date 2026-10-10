"""Camera emulation — the "shot on X" look as real post-process.

Applies to generated OR real footage:

- handheld shake: Perlin-noise 6DoF camera path (translation, roll,
  scale-breathing). Presets: tripod, handheld, walking, selfie, run.
- phone_look: front-cam FOV punch-in, over-sharpen, HDR-ish local
  contrast, chroma grain, rolling-shutter wobble. The selfie-video look.
- cctv / dashcam / cinema looks.

All PIL/numpy + ffmpeg. Zero model cost. Every look is a real
post-process, honestly labeled.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


# ── Perlin noise (1D, for camera paths) ──────────────────────────────
def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)


class Perlin1D:
    """Seeded 1D Perlin noise. fbm() sums octaves."""

    def __init__(self, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.p = np.arange(256, dtype=int)
        rng.shuffle(self.p)
        self.p = np.concatenate([self.p, self.p])

    def noise(self, x: float) -> float:
        xi = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = _fade(xf)
        a = self.p[xi] / 255.0 * 2 - 1
        b = self.p[xi + 1] / 255.0 * 2 - 1
        return a + (b - a) * u

    def fbm(self, x: float, octaves: int = 4) -> float:
        total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
        for _ in range(octaves):
            total += self.noise(x * freq) * amp
            norm += amp
            amp *= 0.5
            freq *= 2.03
        return total / norm


@dataclass
class ShakePreset:
    name: str
    # amplitude in px at 1080p, scaled by frame height
    amp_xy: float = 6.0
    amp_roll_deg: float = 0.4
    amp_zoom: float = 0.004
    freq: float = 1.2        # base Hz of the noise path
    octaves: int = 4


SHAKE_PRESETS: dict[str, ShakePreset] = {
    "tripod": ShakePreset("tripod", amp_xy=0.6, amp_roll_deg=0.03,
                          amp_zoom=0.0005, freq=0.4),
    "handheld": ShakePreset("handheld", amp_xy=6.0, amp_roll_deg=0.4,
                             amp_zoom=0.004, freq=1.2),
    "walking": ShakePreset("walking", amp_xy=14.0, amp_roll_deg=1.1,
                           amp_zoom=0.008, freq=1.8),
    "selfie": ShakePreset("selfie", amp_xy=4.0, amp_roll_deg=0.5,
                          amp_zoom=0.003, freq=2.6),
    "run": ShakePreset("run", amp_xy=26.0, amp_roll_deg=2.0,
                       amp_zoom=0.012, freq=2.8),
}


def camera_path(n_frames: int, fps: float, preset: str,
                seed: int = 0, W: int = 1920, H: int = 1080) -> np.ndarray:
    """Per-frame (dx, dy, roll_deg, zoom) camera offsets."""
    p = SHAKE_PRESETS.get(preset, SHAKE_PRESETS["handheld"])
    s = H / 1080.0
    per = [Perlin1D(seed + i) for i in range(4)]
    out = np.zeros((n_frames, 4))
    for i in range(n_frames):
        t = i / fps * p.freq
        out[i, 0] = per[0].fbm(t, p.octaves) * p.amp_xy * s
        out[i, 1] = per[1].fbm(t + 13.7, p.octaves) * p.amp_xy * s
        out[i, 2] = per[2].fbm(t + 41.2, p.octaves) * p.amp_roll_deg
        out[i, 3] = 1.0 + per[3].fbm(t + 77.9, 2) * p.amp_zoom
    return out


def apply_camera_shake(frames: list[Image.Image], preset: str = "handheld",
                       fps: float = 24.0, seed: int = 0,
                       overscan: float = 1.06) -> list[Image.Image]:
    """Apply a synthesized handheld path. Overscan-crops to hide edges."""
    if not frames:
        return frames
    W, H = frames[0].size
    path = camera_path(len(frames), fps, preset, seed, W, H)
    # render overscanned then crop
    ow, oh = int(W * overscan), int(H * overscan)
    out = []
    for img, (dx, dy, roll, zoom) in zip(frames, path):
        big = img.resize((ow, oh), Image.BICUBIC)
        # translate + roll about the shaken center
        cx, cy = ow / 2 + dx * overscan, oh / 2 + dy * overscan
        moved = big.rotate(-roll, resample=Image.BICUBIC, center=(cx, cy))
        # zoom about the same center, then crop back to W,H
        zw, zh = ow / zoom, oh / zoom
        zx0 = int(cx - zw / 2)
        zy0 = int(cy - zh / 2)
        crop = moved.crop((zx0, zy0, zx0 + int(zw), zy0 + int(zh)))
        crop = crop.resize((W, H), Image.BICUBIC)
        out.append(crop)
    return out


# ── rolling shutter ──────────────────────────────────────────────────
def rolling_shutter(frame: np.ndarray, vx: float,
                    strength: float = 1.0) -> np.ndarray:
    """Row-by-row horizontal shear proportional to horizontal velocity.

    frame: HxWxC uint8. vx: px/frame horizontal shake velocity.
    The phone signature: fast pans smear diagonally.
    """
    H = frame.shape[0]
    rows = np.arange(H)
    shift = (rows / H - 0.5) * vx * strength
    out = np.empty_like(frame)
    for y in range(H):
        out[y] = np.roll(frame[y], int(round(shift[y])), axis=0)
    return out


# ── phone ISP look ───────────────────────────────────────────────────
def phone_grade(img: Image.Image, *, sharpen: float = 120.0,
                grain: float = 6.0, hdr: float = 0.35,
                seed: int = 0) -> Image.Image:
    """The phone-camera look: over-sharpened, HDR-ish, grainy.

    sharpen: unsharp-mask percent. grain: gaussian noise sigma (0-255).
    hdr: 0..1 blend of lifted-shadows/compressed-highlights curve.
    """
    a = np.array(img).astype(np.float64)
    # HDR-ish tone curve: lift shadows, compress highlights
    if hdr > 0:
        norm = a / 255.0
        lifted = norm + (1 - norm) * norm * 0.25 * hdr        # shadows up
        comp = 1 - (1 - lifted) * (1 - 0.18 * hdr)            # highlights in
        a = np.clip(comp, 0, 1) * 255.0
    out = Image.fromarray(a.astype(np.uint8))
    if sharpen > 0:
        out = out.filter(ImageFilter.UnsharpMask(radius=2,
                                                 percent=int(sharpen),
                                                 threshold=2))
    if grain > 0:
        rng = np.random.RandomState(seed)
        g = rng.normal(0, grain, np.array(out).shape).astype(np.float64)
        out = Image.fromarray(
            np.clip(np.array(out).astype(np.float64) + g, 0, 255
                    ).astype(np.uint8))
    # slight saturation push (phone color science)
    out = ImageEnhance.Color(out).enhance(1.12)
    return out


def phone_selfie_look(img: Image.Image, W: int | None = None,
                      H: int | None = None) -> Image.Image:
    """Front-camera framing: vertical punch-in, softer detail, wider feel."""
    w, h = img.size
    # punch in ~12% (front cams are wide; selfies are close)
    m = 0.06
    box = (int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m)))
    crop = img.crop(box)
    if W and H:
        crop = crop.resize((W, H), Image.BICUBIC)
    # front cams are softer than rear: slight blur before sharpen
    crop = crop.filter(ImageFilter.GaussianBlur(0.6))
    return phone_grade(crop, sharpen=140.0, grain=7.0, hdr=0.4)


# ── other camera looks ───────────────────────────────────────────────
def cctv_look(img: Image.Image, timestamp: str = "") -> Image.Image:
    """CCTV: interlace, noise, crushed blacks, timestamp burn-in."""
    a = np.array(img).astype(np.float64)
    a[::2] *= 0.82  # interlace darkening
    rng = np.random.RandomState(42)
    a += rng.normal(0, 9, a.shape)
    a = np.clip(a * 0.9 - 8, 0, 255)  # crushed
    out = Image.fromarray(a.astype(np.uint8)).convert("L").convert("RGB")
    if timestamp:
        d = ImageDraw.Draw(out)
        d.text((12, 12), timestamp, fill=(255, 255, 255))
    return out


def dashcam_look(img: Image.Image, speed: str = "") -> Image.Image:
    """Dashcam: wide barrel-ish vignette, timestamp/speed overlay."""
    w, h = img.size
    # cheap barrel: edge stretch via resize-crop trick
    big = img.resize((int(w * 1.12), int(h * 1.12)), Image.BICUBIC)
    x0 = (big.width - w) // 2
    out = big.crop((x0, (big.height - h) // 2, x0 + w,
                    (big.height - h) // 2 + h))
    out = ImageEnhance.Contrast(out).enhance(1.08)
    if speed:
        d = ImageDraw.Draw(out)
        d.text((12, h - 28), speed, fill=(255, 255, 0))
    return out


def cinema_look(img: Image.Image) -> Image.Image:
    """Cinema: 2.39:1 letterbox, gentle S-curve, 24fps cadence handled upstream."""
    w, h = img.size
    target_h = int(w / 2.39)
    y0 = (h - target_h) // 2
    out = img.crop((0, y0, w, y0 + target_h)).resize((w, h), Image.BICUBIC)
    # letterbox bars
    bar = int(h * 0.06)
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, w, bar], fill=(0, 0, 0))
    d.rectangle([0, h - bar, w, h], fill=(0, 0, 0))
    # S-curve grade (numpy LUT)
    lut = (np.arange(256) / 255.0)
    lut = np.clip(1 / (1 + np.exp(-8 * (lut - 0.5))) * 1.06 - 0.03, 0, 1)
    lut = (lut * 255).astype(np.uint8)
    a = np.array(out)
    return Image.fromarray(lut[a])


LOOKS = ["phone_selfie", "phone", "handheld", "cctv", "dashcam", "cinema",
         "tripod"]


def apply_look(frames: list[Image.Image], look: str, *,
               fps: float = 24.0, seed: int = 0,
               overlay_text: str = "") -> list[Image.Image]:
    """Apply a full camera look to frames. Returns new frame list."""
    if look in ("tripod",):
        return list(frames)
    if look == "phone_selfie":
        W, H = frames[0].size
        out = [phone_selfie_look(f, W, H) for f in frames]
        return apply_camera_shake(out, "selfie", fps, seed)
    if look == "phone":
        out = [phone_grade(f, seed=seed + i) for i, f in enumerate(frames)]
        return apply_camera_shake(out, "handheld", fps, seed)
    if look == "handheld":
        return apply_camera_shake(frames, "handheld", fps, seed)
    if look == "cctv":
        import datetime as _dt
        ts = overlay_text or _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [cctv_look(f, ts) for f in frames]
    if look == "dashcam":
        return [dashcam_look(f, overlay_text or "62 km/h") for f in frames]
    if look == "cinema":
        return [cinema_look(f) for f in frames]
    raise ValueError(f"unknown look {look!r}; see LOOKS")


# ── video in/out ─────────────────────────────────────────────────────
def _ffmpeg() -> str | None:
    from shutil import which
    return which("ffmpeg")


def read_frames(video: str, fps: int = 24, max_frames: int = 0) -> \
        tuple[list[Image.Image], float]:
    """Extract frames with ffmpeg. Returns (frames, fps)."""
    import tempfile
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    tmp = Path(tempfile.mkdtemp(prefix="camlook_"))
    vf = f"fps={fps}"
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", video, "-vf", vf, str(tmp / "f_%04d.png")],
                   check=True, capture_output=True, timeout=600)
    paths = sorted(tmp.glob("f_*.png"))
    if max_frames:
        paths = paths[:max_frames]
    return [Image.open(p).convert("RGB") for p in paths], float(fps)


def write_frames(frames: list[Image.Image], out_path: str,
                 fps: float = 24.0, audio_src: str | None = None) -> str:
    """Write frames to mp4 (optionally carrying audio from a source)."""
    ff = _ffmpeg()
    tmp = Path(tempfile.mkdtemp(prefix="camwrite_"))
    for i, f in enumerate(frames):
        f.save(tmp / f"w_{i:04d}.png")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", str(fps), "-i", str(tmp / "w_%04d.png")]
    if audio_src:
        cmd += ["-i", audio_src, "-map", "0:v", "-map", "1:a?",
                "-c:a", "aac", "-shortest"]
    cmd += ["-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    return out_path


def camera_look_video(video: str, look: str, out_path: str, *,
                      fps: int = 24, seed: int = 0,
                      overlay_text: str = "") -> str:
    """Apply a camera look to a whole video file. Rolling shutter included
    for phone looks (driven by the shake path's horizontal velocity)."""
    frames, _ = read_frames(video, fps=fps)
    if look in ("phone_selfie", "phone"):
        W, H = frames[0].size
        path = camera_path(len(frames), fps,
                           "selfie" if look == "phone_selfie" else "handheld",
                           seed, W, H)
        vx = np.gradient(path[:, 0])  # horizontal velocity px/frame
        styled = apply_look(frames, look, fps=fps, seed=seed,
                            overlay_text=overlay_text)
        out = []
        for f, v in zip(styled, vx):
            a = np.array(f)
            if abs(v) > 0.5:
                a = rolling_shutter(a, v, strength=0.6)
            out.append(Image.fromarray(a))
        frames = out
    else:
        frames = apply_look(frames, look, fps=fps, seed=seed,
                            overlay_text=overlay_text)
    return write_frames(frames, out_path, fps=fps, audio_src=video)
