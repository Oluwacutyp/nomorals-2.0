"""Camera emulation — the "shot on X" look as real post-process.

Applies to generated OR real footage:

- handheld shake: Perlin-noise 6DoF camera path (translation, roll,
  scale-breathing). Presets: tripod, handheld, walking, selfie, run.
- phone_look: front-cam FOV punch-in, over-sharpen, HDR-ish local
  contrast, chroma grain, rolling-shutter wobble. The selfie-video look.
- cctv / dashcam / cinema looks.
- Phase 8A pack: drone, bodycam, webcam, vintage_film, anamorphic, gimbal —
  each a shake preset + ISP/grade signature, same pattern as the originals.

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
    bob_hz: float = 0.0      # vertical step-bounce Hz (bodycam); 0 = off
    bob_amp: float = 0.0     # step-bounce amplitude, px at 1080p


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
    # ── phase 8A pack ──
    "drone": ShakePreset("drone", amp_xy=2.2, amp_roll_deg=0.15,
                         amp_zoom=0.001, freq=0.35),
    "bodycam": ShakePreset("bodycam", amp_xy=9.0, amp_roll_deg=1.4,
                           amp_zoom=0.006, freq=1.9,
                           bob_hz=1.9, bob_amp=7.0),
    "webcam": ShakePreset("webcam", amp_xy=0.4, amp_roll_deg=0.02,
                          amp_zoom=0.0002, freq=0.3),
    "vintage": ShakePreset("vintage", amp_xy=3.0, amp_roll_deg=0.25,
                           amp_zoom=0.002, freq=0.9),
    "gimbal": ShakePreset("gimbal", amp_xy=1.2, amp_roll_deg=0.08,
                          amp_zoom=0.001, freq=0.5),
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
        if p.bob_hz > 0:
            # chest-mount step bounce: periodic vertical thump on top of noise
            secs = i / fps
            out[i, 1] += (math.sin(secs * 2 * math.pi * p.bob_hz)
                          * p.bob_amp * s)
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
    return _render_path(frames, path, overscan)


# ── camera language: natural language -> camera programs ──────────────
# "dolly in slowly, then orbit left" -> CameraProgram([dolly/in/slow,
# orbit/left/normal]) -> per-frame (dx, dy, roll, zoom) path rendered
# with the same overscan machinery as the shake presets. 2D post can't
# do true 3D moves; dolly == zoom, truck == lateral pan — documented,
# not faked.
import re as _re


@dataclass
class CameraMove:
    verb: str        # dolly|truck|pan|tilt|crane|orbit|roll|zoom|static|handheld
    direction: str = ""   # in|out|left|right|up|down (verb-dependent)
    speed: str = "normal"  # slow|normal|fast
    amount: float = 0.65   # 0..1 intensity
    clause: str = ""       # source clause, for debugging


@dataclass
class CameraProgram:
    moves: list[CameraMove]
    source: str = ""

    @property
    def empty(self) -> bool:
        return not self.moves

    def describe(self) -> str:
        return ", then ".join(_describe_move(m) for m in self.moves)


def _describe_move(m: CameraMove) -> str:
    bits = []
    if m.speed == "slow":
        bits.append("slow")
    elif m.speed == "fast":
        bits.append("quick")
    bits.append(m.verb)
    if m.direction:
        bits.append(m.direction)
    return " ".join(bits)


# (pattern, verb, direction-from-match or "")
_VERB_PATTERNS: list[tuple[str, str, str]] = [
    (r"dolly\s+(in|out|forward|back(?:ward)?)", "dolly", "@1"),
    (r"push\s+in|move\s+(in|closer)|dolly(?!\s+\w)", "dolly", "in"),
    (r"pull\s+(?:back|out|away)|move\s+(?:away|back)|zoom\s+out", "dolly", "out"),
    (r"zoom\s+(in|out)", "zoom", "@1"),
    (r"truck\s+(left|right)|slide\s+(left|right)", "truck", "@1"),
    (r"pan\s+(left|right)", "pan", "@1"),
    (r"whip\s+pan(?:\s+(left|right))?", "pan", "@1"),
    (r"tilt\s+(up|down)|pan\s+(up|down)", "tilt", "@1"),
    (r"crane\s+(up|down)|pedestal\s+(up|down)", "crane", "@1"),
    (r"\brise\b|\bascend\b", "crane", "up"),
    (r"\bdescend\b|\bdrop\s+down\b", "crane", "down"),
    (r"orbit\s*(left|right|around)?|arc\s+(?:shot\s+)?(left|right|around)|circle\s+around", "orbit", "@1"),
    (r"dutch(?:\s+angle)?|\broll\b|rotate\s+(?:the\s+)?camera", "roll", ""),
    (r"static|locked[\s-]?off|\bstill\b|no\s+camera\s+move", "static", ""),
    (r"handheld|shaky\s+cam|shakycam", "handheld", ""),
]

_SPEED_WORDS = (
    (r"slow(?:ly)?|gentle|gradual(?:ly)?|lingering", "slow"),
    (r"quick(?:ly)?|fast|rapid(?:ly)?|whip|snappy", "fast"),
)
_AMOUNT_WORDS = (
    (r"slight(?:ly)?|subtle|hint\s+of", 0.35),
    (r"dramatic(?:ally)?|extreme(?:ly)?|hard|pronounced", 1.0),
)


def parse_camera_language(text: str) -> CameraProgram:
    """Parse camera-direction language into a CameraProgram.

    Handles sequences: "dolly in slowly, then orbit left" and
    conjunctions: "pan left and tilt up". Unknown text -> empty program
    (never raises).
    """
    src = (text or "").strip()
    prog = CameraProgram(moves=[], source=src)
    if not src:
        return prog
    clauses = [c.strip() for c in
               _re.split(r"\bthen\b|;|,|\bwhile\b|\band\b", src.lower())
               if c.strip()]
    for clause in clauses:
        for pat, verb, dflt in _VERB_PATTERNS:
            m = _re.search(pat, clause)
            if not m:
                continue
            direction = dflt
            if dflt == "@1":
                g = next((g for g in m.groups() if g), "")
                direction = {"forward": "in", "backward": "out",
                             "back": "out"}.get(g, g)
            speed = "fast" if verb == "pan" and "whip" in clause else "normal"
            for spat, sval in _SPEED_WORDS:
                if _re.search(spat, clause):
                    speed = sval
                    break
            amount = 0.65
            for apat, aval in _AMOUNT_WORDS:
                if _re.search(apat, clause):
                    amount = aval
                    break
            prog.moves.append(CameraMove(verb=verb, direction=direction,
                                        speed=speed, amount=amount,
                                        clause=clause))
            break  # one move per clause
    return prog


def camera_program_path(program: CameraProgram, n_frames: int, fps: float,
                        W: int, H: int, seed: int = 0) -> np.ndarray:
    """Per-frame (dx, dy, roll_deg, zoom) for a camera program.

    Moves run sequentially, splitting the frames evenly; each move is
    smoothstep-eased over its window. Units match camera_path().
    """
    path = np.zeros((n_frames, 4))
    path[:, 3] = 1.0
    moves = program.moves
    if not moves or n_frames == 0:
        return path
    per = n_frames / len(moves)
    bounds = [min(n_frames, int(round(k * per))) for k in range(len(moves) + 1)]
    bounds[-1] = n_frames

    for k, mv in enumerate(moves):
        a = mv.amount
        i0, i1 = bounds[k], bounds[k + 1]
        span = max(1, i1 - i0 - 1)
        for i in range(i0, i1):
            ur = (i - i0) / span
            u = ur * ur * (3 - 2 * ur)  # smoothstep over the window
            v, d = mv.verb, mv.direction
            if v in ("dolly", "zoom"):
                sgn = -1.0 if d == "out" else 1.0
                path[i, 3] += sgn * (0.28 if d == "in" else 0.22) * a * u
            elif v in ("truck", "pan"):
                sgn = -1.0 if d == "left" else 1.0
                path[i, 0] += sgn * 0.10 * a * W * u
            elif v == "tilt":
                sgn = -1.0 if d == "up" else 1.0
                path[i, 1] += sgn * 0.10 * a * H * u
            elif v == "crane":
                # camera rises -> frame content sinks; slight widen
                sgn = 1.0 if d == "up" else -1.0
                path[i, 1] += sgn * 0.14 * a * H * u
                path[i, 3] += -sgn * 0.05 * a * u
            elif v == "orbit":
                sgn = -1.0 if d == "left" else 1.0
                path[i, 0] += sgn * 0.12 * a * W * math.sin(math.pi * u)
                path[i, 2] += sgn * 3.0 * a * math.sin(2 * math.pi * u)
                path[i, 3] += 0.05 * a * math.sin(math.pi * u)
            elif v == "roll":
                sgn = -1.0 if d in ("left", "counterclockwise") else 1.0
                path[i, 2] += sgn * 10.0 * a * u
            elif v == "handheld":
                hp = camera_path(i1 - i0, fps, "handheld", seed + k, W, H)
                path[i, 0] += hp[i - i0, 0] * a
                path[i, 1] += hp[i - i0, 1] * a
                path[i, 2] += hp[i - i0, 2] * a
            # static: zeros
    # carry forward: each move continues from where the previous left off
    for k in range(1, len(moves)):
        i0 = bounds[k]
        if i0 <= 0 or i0 >= n_frames:
            continue
        path[i0:, 0:3] += path[i0 - 1, 0:3] - path[i0, 0:3]
        path[i0:, 3] *= path[i0 - 1, 3] / max(1e-6, path[i0, 3])
    return path


def _render_path(frames: list[Image.Image], path: np.ndarray,
                 overscan: float = 1.06) -> list[Image.Image]:
    """Render frames through a (dx, dy, roll_deg, zoom) path."""
    if not frames:
        return frames
    W, H = frames[0].size
    ow, oh = int(W * overscan), int(H * overscan)
    out = []
    for img, (dx, dy, roll, zoom) in zip(frames, path):
        big = img.resize((ow, oh), Image.BICUBIC)
        cx, cy = ow / 2 + dx * overscan, oh / 2 + dy * overscan
        moved = big.rotate(-roll, resample=Image.BICUBIC, center=(cx, cy))
        zw, zh = ow / zoom, oh / zoom
        zx0, zy0 = int(cx - zw / 2), int(cy - zh / 2)
        crop = moved.crop((zx0, zy0, zx0 + int(zw), zy0 + int(zh)))
        out.append(crop.resize((W, H), Image.BICUBIC))
    return out


def apply_camera_program(frames: list[Image.Image], program: CameraProgram,
                         fps: float = 24.0, seed: int = 0,
                         overscan: float = 1.06) -> list[Image.Image]:
    """Apply a parsed camera program to frames. Empty program -> passthrough."""
    if not frames or program.empty:
        return list(frames)
    W, H = frames[0].size
    path = camera_program_path(program, len(frames), fps, W, H, seed)
    return _render_path(frames, path, overscan)


def direct_camera(text: str, frames: list[Image.Image], *,
                  fps: float = 24.0, seed: int = 0) -> list[Image.Image]:
    """Natural language -> camera move on frames. One call."""
    return apply_camera_program(frames, parse_camera_language(text),
                                fps=fps, seed=seed)


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


# ── phase 8A camera pack ─────────────────────────────────────────────
def _temporal_zoom(frames: list[Image.Image], z0: float, z1: float,
                   ease_fn=None) -> list[Image.Image]:
    """Progressive center zoom z0 -> z1 across the frame list (dolly feel)."""
    n = len(frames)
    if n == 0:
        return frames
    out = []
    for i, img in enumerate(frames):
        u = i / max(1, n - 1)
        k = ease_fn(u) if ease_fn else u * u * (3 - 2 * u)
        z = z0 + (z1 - z0) * k
        w, h = img.size
        zw, zh = int(w * z), int(h * z)
        big = img.resize((zw, zh), Image.BICUBIC)
        x0, y0 = (zw - w) // 2, (zh - h) // 2
        out.append(big.crop((x0, y0, x0 + w, y0 + h)))
    return out


def drone_look(img: Image.Image) -> Image.Image:
    """Drone: hazy aerial grade — lifted blacks, pulled saturation,
    faint cool cast. The slow forward drift is applied at the video
    level (drone shake preset + progressive push)."""
    a = np.array(img).astype(np.float64)
    a = a * 0.86 + 26.0                       # lifted, hazy blacks
    a[..., 2] = np.clip(a[..., 2] + 6.0, 0, 255)  # cool cast
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return ImageEnhance.Color(out).enhance(0.82)


def bodycam_look(img: Image.Image, timestamp: str = "") -> Image.Image:
    """Bodycam: chest-mount bounce (video level), crushed mids, heavy
    vignette, timestamp burn-in."""
    out = ImageEnhance.Color(img).enhance(0.78)
    out = ImageEnhance.Contrast(out).enhance(1.06)
    w, h = out.size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xs / w - 0.5) * 2.0
    dy = (ys / h - 0.5) * 2.0
    mask = np.clip(1.0 - (dx ** 2 + dy ** 2) * 0.55, 0.0, 1.0)
    a = np.asarray(out).astype(np.float32) * mask[..., None]
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if timestamp:
        d = ImageDraw.Draw(out)
        d.text((w - 200, 12), timestamp, fill=(255, 255, 255))
    return out


def webcam_look(img: Image.Image, *, pump: float = 0.0,
                seed: int = 0) -> Image.Image:
    """Webcam: fixed framing, soft upscale feel, sensor noise, and a
    slow auto-exposure pump (``pump`` in [-1, 1] drives brightness)."""
    out = img.filter(ImageFilter.GaussianBlur(0.7))
    a = np.array(out).astype(np.float64) * (1.0 + 0.05 * pump)
    rng = np.random.RandomState(seed)
    a += rng.normal(0, 4.5, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def vintage_film_look(img: Image.Image, *, seed: int = 0,
                      frame_idx: int = 0) -> Image.Image:
    """Vintage film: warm faded stock, heavy grain, scratches, gate
    weave + flicker. Deterministic per (seed, frame_idx)."""
    a = np.array(img).astype(np.float64)
    # faded warm stock: warm cast, lifted blacks, compressed highlights
    a[..., 0] = np.clip(a[..., 0] * 1.06 + 14.0, 0, 255)
    a[..., 1] = np.clip(a[..., 1] * 0.96 + 10.0, 0, 255)
    a[..., 2] = np.clip(a[..., 2] * 0.78 + 8.0, 0, 255)
    rng = np.random.RandomState(seed + frame_idx * 7919)
    a += rng.normal(0, 11.0, a.shape)                       # grain
    a *= 1.0 + (rng.uniform() - 0.5) * 0.10                 # flicker
    h, w = a.shape[:2]
    # scratches: 1-3 vertical hairlines, drifting with the frame
    for _ in range(int(rng.randint(1, 4))):
        x = int(rng.randint(0, w))
        a[:, max(0, x - 1):x + 1] *= 0.72
    # gate weave: whole-frame sub-pixel jitter
    jx, jy = int(rng.randint(-2, 3)), int(rng.randint(-2, 3))
    a = np.roll(np.roll(a, jy, axis=0), jx, axis=1)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    # rounded heavy vignette
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xs / w - 0.5) * 2.0) ** 2
                + ((ys / h - 0.5) * 2.0) ** 2)
    mask = np.clip(1.0 - np.clip(r - 0.72, 0, None) ** 1.6 * 1.9, 0, 1)
    return Image.fromarray(
        (np.asarray(out).astype(np.float32)
         * mask[..., None]).clip(0, 255).astype(np.uint8))


def anamorphic_look(img: Image.Image) -> Image.Image:
    """Anamorphic: 2.39:1 letterbox, warm grade, blue horizontal flare
    streaks blooming off the brightest highlights (the anamorphic
    signature)."""
    w, h = img.size
    target_h = int(w / 2.39)
    y0 = (h - target_h) // 2
    frame = img.crop((0, y0, w, y0 + target_h)).resize((w, h), Image.BICUBIC)
    a = np.array(frame).astype(np.float64)
    # warm grade
    a[..., 0] = np.clip(a[..., 0] * 1.05 + 6.0, 0, 255)
    a[..., 2] = np.clip(a[..., 2] * 0.94, 0, 255)
    luma = a.mean(axis=2)
    row_peak = luma.max(axis=1)
    hot_rows = np.where(row_peak > 215)[0]
    if len(hot_rows):
        # blue streak per hot row, centered on the brightest pixel,
        # gaussian falloff horizontally
        xs = np.arange(w, dtype=np.float64)
        for y in hot_rows:
            cx = int(np.argmax(luma[y]))
            fall = np.exp(-((xs - cx) ** 2) / (2 * (w * 0.09) ** 2))
            strength = (row_peak[y] - 215) / 40.0
            streak = fall * strength
            a[y, :, 0] = np.clip(a[y, :, 0] + streak * 40.0, 0, 255)
            a[y, :, 1] = np.clip(a[y, :, 1] + streak * 90.0, 0, 255)
            a[y, :, 2] = np.clip(a[y, :, 2] + streak * 220.0, 0, 255)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    bar = int(h * 0.06)
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, w, bar], fill=(0, 0, 0))
    d.rectangle([0, h - bar, w, h], fill=(0, 0, 0))
    return out


def gimbal_look(img: Image.Image) -> Image.Image:
    """Gimbal: buttery stabilized footage — gentle S-curve, clean
    saturation. The smoothness is the gimbal shake preset at video level."""
    lut = np.arange(256) / 255.0
    lut = np.clip(1 / (1 + np.exp(-6 * (lut - 0.5))) * 1.04 - 0.02, 0, 1)
    lut = (lut * 255).astype(np.uint8)
    a = np.array(img)
    out = Image.fromarray(lut[a])
    return ImageEnhance.Color(out).enhance(1.06)


LOOK_INFO: dict[str, str] = {
    "phone_selfie": "front-camera punch-in, over-sharpened, HDR-ish",
    "phone": "rear-camera ISP look: sharpen, grain, HDR tone curve",
    "handheld": "natural handheld shake, no grade",
    "cctv": "interlaced, noisy, crushed, timestamp",
    "dashcam": "wide vignette, speed overlay",
    "cinema": "2.39:1 letterbox, gentle S-curve",
    "tripod": "locked off, no processing",
    "drone": "aerial haze grade + slow forward drift",
    "bodycam": "chest-mount bounce, heavy vignette, timestamp",
    "webcam": "fixed, soft, sensor noise, exposure pump",
    "vintage_film": "faded warm stock, grain, scratches, flicker, gate weave",
    "anamorphic": "2.39:1, blue horizontal flare streaks, warm grade",
    "gimbal": "buttery stabilized, gentle S-curve",
}

#: every look apply_look() understands
LOOKS = list(LOOK_INFO)


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
    if look == "drone":
        graded = [drone_look(f) for f in frames]
        pushed = _temporal_zoom(graded, 1.00, 1.12)  # slow forward drift
        return apply_camera_shake(pushed, "drone", fps, seed)
    if look == "bodycam":
        import datetime as _dt
        ts = overlay_text or _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        graded = [bodycam_look(f, ts) for f in frames]
        return apply_camera_shake(graded, "bodycam", fps, seed)
    if look == "webcam":
        n = len(frames)
        graded = [webcam_look(f, pump=math.sin(2 * math.pi * i / max(1, n)),
                              seed=seed + i)
                  for i, f in enumerate(frames)]
        return apply_camera_shake(graded, "webcam", fps, seed)
    if look == "vintage_film":
        graded = [vintage_film_look(f, seed=seed, frame_idx=i)
                  for i, f in enumerate(frames)]
        return apply_camera_shake(graded, "vintage", fps, seed)
    if look == "anamorphic":
        graded = [anamorphic_look(f) for f in frames]
        return apply_camera_shake(graded, "gimbal", fps, seed)
    if look == "gimbal":
        graded = [gimbal_look(f) for f in frames]
        return apply_camera_shake(graded, "gimbal", fps, seed)
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
