"""Audio-reactive visualizer — FFT → particle fields / spectrum bars /
waveform tunnels, beat-cut on detected onsets.

Beat detection is reused from :mod:`nomorals.media.contentops.beats`
(librosa when present, deterministic numpy fallback otherwise) — never
duplicated here.

Styles and palettes are data (``VISUALIZER_STYLES`` / ``PALETTES`` /
``VISUAL_PRESETS``), so new looks are config, not code branches.

    from nomorals.media.motion_studio.visualizer import render_visualizer
    render_visualizer("song.mp3", "viz.mp4", style="bars", preset="phonk")
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ._core import (
    MotionStudioError,
    decode_audio_mono,
    new_render_path,
    profile_defaults,
    probe_duration,
    record_ledger,
    render_sequence,
)
from ..contentops.beats import detect_beats_full
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "PALETTES", "VISUALIZER_STYLES", "VISUAL_PRESETS",
    "analyze_audio", "render_visualizer", "list_styles", "list_presets",
]

_ANALYSIS_SR = 22050
_N_BANDS = 28


# ---------------------------------------------------------------------------
# style + palette data
# ---------------------------------------------------------------------------

#: palette name → list of hex colors (bg, primary, secondary, accent)
PALETTES: dict[str, list[str]] = {
    "neon":    ["#050510", "#00f0ff", "#ff2fd6", "#ffe14d"],
    "phonk":   ["#0a0a0f", "#b537f2", "#ff3860", "#f5f5f5"],
    "lofi":    ["#141824", "#7fb5a3", "#e8b04b", "#f2e8d5"],
    "sunset":  ["#12060f", "#ff5e3a", "#ff2a68", "#ffd23f"],
    "mono":    ["#000000", "#ffffff", "#8a8a8a", "#ffffff"],
    "ocean":   ["#020d1a", "#00c2ff", "#0077ff", "#aef1ff"],
    "ember":   ["#100604", "#ff7b00", "#ff2e00", "#ffd9a0"],
    "ambient": ["#060a12", "#4d7cff", "#9d4dff", "#cfe6ff"],
}

#: visual style name → engine + description (the engine dispatch is below)
VISUALIZER_STYLES: dict[str, dict[str, str]] = {
    "bars":      {"engine": "bars",      "label": "spectrum bars + mirrored sky"},
    "particles": {"engine": "particles", "label": "beat-reactive particle field"},
    "tunnel":    {"engine": "tunnel",    "label": "waveform tunnel flight"},
    "cover":     {"engine": "cover",     "label": "pulsing cover art + ring"},
}

#: preset name → {style, palette, vertical} — what users actually pick
VISUAL_PRESETS: dict[str, dict[str, object]] = {
    "lofi":    {"style": "particles", "palette": "lofi",    "vertical": False},
    "phonk":   {"style": "bars",      "palette": "phonk",   "vertical": True},
    "shorts":  {"style": "cover",     "palette": "neon",    "vertical": True},
    "ambient": {"style": "tunnel",    "palette": "ambient", "vertical": False},
    "club":    {"style": "bars",      "palette": "sunset",  "vertical": False},
    "minimal": {"style": "particles", "palette": "mono",    "vertical": True},
}


def list_styles() -> list[dict[str, str]]:
    return [{"name": n, **v} for n, v in VISUALIZER_STYLES.items()]


def list_presets() -> list[dict[str, object]]:
    return [{"name": n, **v} for n, v in VISUAL_PRESETS.items()]


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

@dataclass
class AudioAnalysis:
    sr: int
    fps: float
    n_frames: int
    duration: float
    bands: np.ndarray        # (n_frames, N_BANDS) 0..1 log-spaced spectrum
    energy: np.ndarray       # (n_frames,) 0..1 overall energy
    flux: np.ndarray         # (n_frames,) 0..1 onset strength
    wave: np.ndarray         # (n_frames, 256) waveform slice per frame -1..1
    beats: list[float]
    bpm: float
    backend: str


def analyze_audio(audio: str | os.PathLike, fps: float,
                  duration: float | None = None) -> AudioAnalysis:
    """Decode + per-frame spectral analysis. Deterministic, no heavy deps."""
    y = decode_audio_mono(audio, sr=_ANALYSIS_SR)
    full_dur = len(y) / _ANALYSIS_SR
    duration = min(duration or full_dur, full_dur)
    if duration <= 0:
        raise MotionStudioError(f"audio has no duration: {audio}")
    n_frames = max(1, int(round(duration * fps)))
    hop = _ANALYSIS_SR / fps
    win = 2048

    bands = np.zeros((n_frames, _N_BANDS), dtype=np.float32)
    energy = np.zeros(n_frames, dtype=np.float32)
    wave = np.zeros((n_frames, 256), dtype=np.float32)
    # log-spaced band edges over the FFT bins
    edges = np.logspace(np.log10(2), np.log10(win // 2), _N_BANDS + 1).astype(int)
    prev_mag = None
    flux = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        c = int(i * hop)
        s0 = max(0, c - win // 2)
        seg = y[s0:s0 + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        spec = np.abs(np.fft.rfft(seg * np.hanning(win)))
        mag = np.zeros(_N_BANDS)
        for b in range(_N_BANDS):
            mag[b] = spec[edges[b]:max(edges[b] + 1, edges[b + 1])].mean()
        mag = np.log1p(mag * 40.0)
        bands[i] = mag / (mag.max() + 1e-6)
        energy[i] = float(np.sqrt((seg ** 2).mean()))
        if prev_mag is not None:
            flux[i] = float(np.maximum(0, mag - prev_mag).sum())
        prev_mag = mag
        ws = y[c:c + int(hop)]
        if len(ws):
            idx = np.linspace(0, len(ws) - 1, 256).astype(int)
            wave[i] = np.clip(ws[idx], -1, 1)
    # normalize
    if energy.max() > 0:
        energy /= energy.max()
    if flux.max() > 0:
        flux /= flux.max()
    # smooth bands a touch so bars don't jitter
    if n_frames > 3:
        bands = 0.35 * bands + 0.65 * np.vstack(
            [bands[0:1], (bands[:-2] + bands[1:-1] + bands[2:]) / 3.0, bands[-1:]]) \
            if n_frames > 2 else bands

    try:
        info = detect_beats_full(audio)
        beats = [b for b in info.beats if b <= duration]
        bpm, backend = info.bpm, info.backend
    except Exception as exc:  # noqa: BLE001 - visuals never die on beat failure
        _log.warning("beat detection failed, visuals run beat-free: %s", exc)
        beats, bpm, backend = [], 0.0, "none"

    return AudioAnalysis(sr=_ANALYSIS_SR, fps=fps, n_frames=n_frames,
                         duration=duration, bands=bands, energy=energy,
                         flux=flux, wave=wave, beats=beats, bpm=bpm,
                         backend=backend)


def _beat_pulse(t: float, beats: list[float], decay: float = 0.35) -> float:
    """0..1 punch that spikes on a beat and decays — drives cuts/zooms."""
    if not beats:
        return 0.0
    # nearest beat at or before t
    prev = 0.0
    for b in beats:
        if b <= t:
            prev = b
        else:
            break
    dt = t - prev
    return float(math.exp(-dt / decay)) if dt < 1.5 else 0.0


# ---------------------------------------------------------------------------
# frame engines
# ---------------------------------------------------------------------------

class _Engine:
    def __init__(self, w: int, h: int, palette: list[str], seed: int,
                 analysis: AudioAnalysis, images: Sequence[str | os.PathLike],
                 n_particles: int):
        self.w, self.h = w, h
        self.bg, self.c1, self.c2, self.c3 = (_hex(c) for c in palette[:4])
        self.a = analysis
        self.rng = np.random.default_rng(seed)
        self.images = [self._load_cover(p) for p in images[:3]]
        # particle state
        n = n_particles
        self.px = self.rng.random(n) * w
        self.py = self.rng.random(n) * h
        ang = self.rng.random(n) * 2 * math.pi
        spd = 20 + self.rng.random(n) * 80
        self.pvx = np.cos(ang) * spd
        self.pvy = np.sin(ang) * spd
        self.psz = 1 + self.rng.random(n) * 3
        self.pcol = self.rng.integers(0, 3, n)

    def _load_cover(self, p) -> Image.Image | None:
        try:
            img = Image.open(p).convert("RGB")
            s = max(self.w / img.width, self.h / img.height)
            return img.resize((int(img.width * s) + 1, int(img.height * s) + 1),
                              Image.LANCZOS)
        except Exception:  # noqa: BLE001
            return None

    def _base(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (self.w, self.h), self.bg)
        return img, ImageDraw.Draw(img, "RGBA")

    def _band_slice(self, i: int, n: int) -> np.ndarray:
        b = self.a.bands[min(i, self.a.n_frames - 1)]
        idx = np.linspace(0, len(b) - 1, n).astype(int)
        return b[idx]


class _BarsEngine(_Engine):
    def frame(self, i: int, t: float) -> np.ndarray:
        img, d = self._base()
        n = 40
        vals = self._band_slice(i, n)
        pulse = _beat_pulse(t, self.a.beats)
        # mirrored sky: bars from center line
        bw = self.w / n
        mid = self.h * 0.52
        maxh = self.h * 0.42 * (1 + 0.25 * pulse)
        for k, v in enumerate(vals):
            x0 = k * bw + 1
            x1 = (k + 1) * bw - 1
            hh = max(2.0, float(v) * maxh)
            col = self.c1 if k % 3 == 0 else (self.c2 if k % 3 == 1 else self.c3)
            box = [x0, mid - hh, x1, mid + hh * 0.35]
            if hh > 12:
                d.rounded_rectangle(box, radius=3, fill=col + (230,))
            else:
                d.rectangle(box, fill=col + (230,))
        # beat flash
        if pulse > 0.55:
            d.rectangle([0, 0, self.w, self.h],
                        fill=self.c3 + (int(40 * pulse),))
        # energy glow line
        e = float(self.a.energy[min(i, self.a.n_frames - 1)])
        d.line([0, self.h - 4, self.w * e, self.h - 4], fill=self.c1, width=4)
        return np.asarray(img)


class _ParticlesEngine(_Engine):
    def frame(self, i: int, t: float) -> np.ndarray:
        img, d = self._base()
        dt = 1.0 / self.a.fps
        e = float(self.a.energy[min(i, self.a.n_frames - 1)])
        pulse = _beat_pulse(t, self.a.beats)
        spd = 1.0 + 3.0 * e + 4.0 * pulse
        self.px = (self.px + self.pvx * spd * dt) % self.w
        self.py = (self.py + self.pvy * spd * dt) % self.h
        cols = [self.c1, self.c2, self.c3]
        bright = int(120 + 135 * min(1.0, e + pulse))
        for k in range(len(self.px)):
            c = cols[int(self.pcol[k])]
            r = self.psz[k] * (1 + pulse)
            d.ellipse([self.px[k] - r, self.py[k] - r,
                       self.px[k] + r, self.py[k] + r],
                      fill=c + (bright,))
        # beat shockwave ring
        if pulse > 0.6:
            rr = (1 - pulse) * max(self.w, self.h) * 0.7 + 20
            d.ellipse([self.w / 2 - rr, self.h / 2 - rr,
                       self.w / 2 + rr, self.h / 2 + rr],
                      outline=self.c3 + (int(160 * pulse),), width=3)
        # faint trailing nebula from low bands
        low = float(self._band_slice(i, 8)[:3].mean())
        if low > 0.35:
            rad = int(min(self.w, self.h) * (0.2 + 0.3 * low))
            d.ellipse([self.w / 2 - rad, self.h * 0.7 - rad,
                       self.w / 2 + rad, self.h * 0.7 + rad],
                      fill=self.c1 + (int(30 * low),))
        return np.asarray(img)


class _TunnelEngine(_Engine):
    def frame(self, i: int, t: float) -> np.ndarray:
        img, d = self._base()
        cx, cy = self.w / 2, self.h / 2
        wv = self.a.wave[min(i, self.a.n_frames - 1)]
        pulse = _beat_pulse(t, self.a.beats)
        depth = 14
        cols = [self.c1, self.c2, self.c3]
        for r_ in range(depth, 0, -1):
            k = r_ / depth
            # each ring samples a different slice of the waveform → tunnel flight
            seg = wv[int((1 - k) * 200):int((1 - k) * 200) + 56]
            if len(seg) < 56:
                seg = np.pad(seg, (0, 56 - len(seg)))
            rad = (1 - k) ** 1.6 * min(self.w, self.h) * 0.48 + 8
            rad *= 1 + 0.18 * pulse
            pts = []
            for j, s in enumerate(seg):
                ang = j / len(seg) * 2 * math.pi + t * (0.4 + 0.6 * k)
                rr = rad * (1 + 0.35 * float(s))
                pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
            alpha = int(40 + 200 * k)
            d.line(pts + pts[:1], fill=cols[r_ % 3] + (alpha,), width=max(1, int(3 * k) + 1))
        # core flash on beats
        if pulse > 0.5:
            cr = int(30 + 60 * pulse)
            d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                      fill=self.c3 + (int(120 * pulse),))
        return np.asarray(img)


class _CoverEngine(_Engine):
    def frame(self, i: int, t: float) -> np.ndarray:
        img, d = self._base()
        pulse = _beat_pulse(t, self.a.beats)
        e = float(self.a.energy[min(i, self.a.n_frames - 1)])
        if self.images and self.images[0] is not None:
            cover = self.images[0]
            zoom = 1.0 + 0.10 * e + 0.10 * pulse
            cw, ch = int(self.w * zoom), int(self.h * zoom)
            bg = cover.resize((cw, ch), Image.LANCZOS).filter(
                ImageFilter.GaussianBlur(18))
            img.paste(bg.crop(((cw - self.w) // 2, (ch - self.h) // 2,
                               (cw + self.w) // 2, (ch + self.h) // 2)), (0, 0))
            # pulsing cover square
            side = int(min(self.w, self.h) * (0.52 + 0.06 * pulse))
            sq = cover.resize((side, side), Image.LANCZOS)
            img.paste(sq, ((self.w - side) // 2, int(self.h * 0.30)))
            d.rectangle([(self.w - side) // 2, int(self.h * 0.30),
                         (self.w + side) // 2, int(self.h * 0.30) + side],
                        outline=self.c1 + (255,), width=4)
        # spectrum ring around center
        cx, cy = self.w / 2, self.h * 0.30 + min(self.w, self.h) * 0.29
        vals = self._band_slice(i, 48)
        base_r = min(self.w, self.h) * 0.36
        pts = []
        for k, v in enumerate(vals):
            ang = k / len(vals) * 2 * math.pi
            rr = base_r * (1 + 0.45 * float(v))
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        d.line(pts + pts[:1], fill=self.c2 + (220,), width=3)
        # beat-cut flash frame
        if pulse > 0.75:
            d.rectangle([0, 0, self.w, self.h], outline=self.c3 + (int(200 * pulse),),
                        width=10)
        return np.asarray(img)


_ENGINES = {
    "bars": _BarsEngine,
    "particles": _ParticlesEngine,
    "tunnel": _TunnelEngine,
    "cover": _CoverEngine,
}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def render_visualizer(audio: str | os.PathLike,
                      out: str | os.PathLike | None = None, *,
                      style: str = "bars", preset: str = "",
                      palette: str = "neon",
                      images: Sequence[str | os.PathLike] = (),
                      duration: float | None = None,
                      size: tuple[int, int] | None = None,
                      fps: float | None = None,
                      seed: int = 11) -> str:
    """Render an audio-reactive visualizer and mux the audio. Returns path.

    ``preset`` (lofi/phonk/shorts/ambient/…) overrides style+palette when
    given. ``duration`` caps the render (default: full audio length).
    """
    if preset:
        p = VISUAL_PRESETS.get(preset.lower())
        if p is None:
            raise MotionStudioError(
                f"unknown preset {preset!r} — pick from: "
                f"{', '.join(sorted(VISUAL_PRESETS))}")
        style = str(p["style"])
        palette = str(p["palette"])
        if p.get("vertical"):
            size = size  # caller size wins; studio.py picks vertical
    engine_name = VISUALIZER_STYLES.get(style, {}).get("engine", "")
    if not engine_name:
        raise MotionStudioError(
            f"unknown visualizer style {style!r} — pick from: "
            f"{', '.join(sorted(VISUALIZER_STYLES))}")
    pal = PALETTES.get(palette.lower())
    if pal is None:
        raise MotionStudioError(
            f"unknown palette {palette!r} — pick from: {', '.join(sorted(PALETTES))}")

    defaults = profile_defaults()
    size = size or defaults["size"]
    fps = fps or float(defaults["fps"])
    analysis = analyze_audio(audio, fps, duration=duration)

    engine = _ENGINES[engine_name](size[0], size[1], pal, seed, analysis,
                                   images, int(defaults["particles"]))
    out_path = Path(out) if out else new_render_path(f"viz-{style}")
    render_sequence(analysis.n_frames, size[0], size[1], fps, out_path,
                    engine.frame, audio=audio,
                    crf=int(defaults["crf"]), preset=str(defaults["preset"]))
    record_ledger({"kind": "visualizer", "path": str(out_path),
                   "style": style, "palette": palette,
                   "bpm": analysis.bpm, "beats_backend": analysis.backend,
                   "duration": analysis.duration})
    return str(out_path)
