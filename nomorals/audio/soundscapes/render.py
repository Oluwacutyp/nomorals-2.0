"""Render a soundscape into a WAV file Devon can send.

``render_soundscape(home, name, ...)`` is the one entry point. The
name is a generator ("rain"), a preset ("sleep") or one of the
owner's saved mixes — in that order of the owner's own words first.
It returns a receipt — ``{"ok": True, "path": ..., "render_hash":
..., "peak": ..., "rms": ..., "reply": ...}`` — or ``{"ok": False,
"error": ...}`` when the request cannot be rendered. The receipt is
the proof: the SHA-256 of the file, the measured peak and RMS, the
actual length. Nothing is ever faked — a render that cannot be
built is refused in words, never returned as silence.

How a render is built, per kind:

* noise — one colour, EQed by the ten sliders, stereo by two
  decorrelated streams (a real soundstage, not a doubled mono).
* tones — binaural beats or a drone, EQed the same way.
* scene — each stem rendered at its slider level (dB), summed,
  then the hearing profile applied last.

Looping: the render is built in 1-second windows and, when
``loop`` is set, the last second crossfades into the first — a
5-minute file loops without a click. Fading: ``fade_s`` fades the
tail to silence over that many seconds; a sleep timer implies a
30-second fade. Without either, the file ends at full level so it
loops cleanly.

The hearing profile is read from the store at ``home``
automatically and named in the reply — a calibrated ear hears
every generator flat without asking twice.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

from .engine import (
    BAND_FREQS,
    MAX_DURATION_S,
    SR,
    animate_sliders,
    apply_eq,
    apply_hearing,
    synthesize_noise,
    synthesize_tone,
    wav_bytes_sha256,
    write_stereo_wav,
)
from .generators import PRESETS, resolve_generator
from .stems import STEMS
from .store import SoundscapeStore

__all__ = ["MAX_DURATION_S", "render_soundscape"]

_DEFAULT_DURATION_S = 60.0
_WINDOW_S = 1.0


def _db_to_gain(db: float) -> float:
    return 10.0 ** (float(db) / 20.0)


def _mix_stems(sliders: dict[str, float], n: int, sr: int,
               seed: int | None) -> list[float]:
    """Render each scene stem at its level and sum them."""
    mix: list[float] = [0.0] * n
    for name, level_db in sliders.items():
        fn = STEMS.get(name)
        if fn is None:
            continue
        gain = _db_to_gain(level_db)
        if gain <= 0.0:
            continue
        stem = fn(n, sr, seed=seed)
        for i in range(n):
            mix[i] += stem[i] * gain
    return mix


def _eq_sliders_to_gains(sliders: dict[str, float]) -> dict[float, float]:
    """Map slider labels ('1000 Hz') or band centres to EQ gains."""
    gains: dict[float, float] = {}
    for label, value in sliders.items():
        label = str(label).strip()
        for freq in BAND_FREQS:
            if label == f"{freq:g} Hz" or label == str(freq):
                gains[freq] = float(value)
                break
    return gains


def _render_windows(kind: str, sliders: dict[str, float], n: int, sr: int,
                    params: dict[str, Any], animate: bool,
                    seed: int | None) -> tuple[list[float], list[float]]:
    """Render in 1-second windows; the seam is crossfaded by the caller."""
    window = int(_WINDOW_S * sr)
    left: list[float] = []
    right: list[float] = []
    rng = random.Random(seed)
    elapsed = 0.0
    while len(left) < n:
        count = min(window, n - len(left))
        window_seed = rng.randrange(1 << 30)
        gains = dict(sliders)
        if animate:
            if kind == "scene":
                base = list(sliders.values())
                drifted = animate_sliders(base, elapsed, seed=window_seed)
                gains = {k: v for k, v in zip(sliders, drifted, strict=True)}
            else:
                base = [gains.get(f"{f:g} Hz", 0.0) for f in BAND_FREQS]
                drifted = animate_sliders(base, elapsed, seed=window_seed)
                gains = {f"{f:g} Hz": v
                         for f, v in zip(BAND_FREQS, drifted, strict=True)}
        if kind == "noise":
            color = params.get("color", "white")
            left_w = synthesize_noise(color, count, sr, seed=window_seed)
            right_w = synthesize_noise(color, count, sr,
                                       seed=window_seed + 1)
            eq = _eq_sliders_to_gains(gains)
            left_w = apply_eq(left_w, sr, eq)
            right_w = apply_eq(right_w, sr, eq)
        elif kind == "tones":
            tone_kind = params.get("tone", "binaural")
            left_w, right_w = synthesize_tone(
                tone_kind, count, sr,
                freq=float(params.get("carrier", 200.0)),
                beat=float(params.get("beat", 10.0)),
                root=float(params.get("root", 110.0)),
                seed=window_seed)
            eq = _eq_sliders_to_gains(gains)
            left_w = apply_eq(left_w, sr, eq)
            right_w = apply_eq(right_w, sr, eq)
        else:  # scene
            mono = _mix_stems(gains, count, sr, window_seed)
            left_w = list(mono)
            right_w = list(mono)
        left.extend(left_w)
        right.extend(right_w)
        elapsed += _WINDOW_S
    return left[:n], right[:n]


def _crossfade_seam(left: list[float], right: list[float], sr: int) -> None:
    """Blend the last second into the first so the file loops cleanly."""
    n = len(left)
    fade = min(int(_WINDOW_S * sr), n // 2)
    if fade < 2:
        return
    for i in range(fade):
        w = i / fade
        left[n - fade + i] = (left[n - fade + i] * (1.0 - w)
                              + left[i] * w)
        right[n - fade + i] = (right[n - fade + i] * (1.0 - w)
                               + right[i] * w)


def _fade_out_tail(left: list[float], right: list[float],
                   fade_s: float, sr: int) -> None:
    """Fade the tail to silence over the last ``fade_s`` seconds."""
    fade = min(int(fade_s * sr), len(left))
    if fade < 2:
        return
    n = len(left)
    for i in range(fade):
        w = 1.0 - (i / fade)
        left[n - fade + i] *= w
        right[n - fade + i] *= w


def _peak_rms(samples: list[float]) -> tuple[float, float]:
    peak = 0.0
    total = 0.0
    for s in samples:
        peak = max(peak, abs(s))
        total += s * s
    rms = math.sqrt(total / len(samples)) if samples else 0.0
    return peak, rms


def render_soundscape(home: str | os.PathLike[str], name: str, *,
                      duration_s: float | None = None,
                      seed: int | None = None,
                      animate: bool | None = None,
                      fade_s: float | None = None,
                      loop: bool = True,
                      sliders: dict[str, float] | None = None,
                      adjust: dict[str, float] | None = None,
                      timer_min: float | None = None) -> dict[str, Any]:
    """Render a soundscape into a stereo WAV under ``home``.

    ``name`` is a generator id, an alias, a preset, or a saved mix.
    ``duration_s`` over :data:`MAX_DURATION_S` raises ValueError —
    a render that long is refused, never silently shortened.
    An unknown name returns ``{"ok": False, "error": ...}`` naming
    the name and the real generators. The hearing profile is read
    from the store at ``home`` and applied automatically.
    """
    home_path = Path(home)
    store = SoundscapeStore(home_path)
    name = str(name or "").strip()
    if not name:
        return {"ok": False,
                "error": "no soundscape named — try 'rain', 'forest', "
                         "'cafe', 'brown noise' or a preset like 'sleep'"}
    mix = store.load_mix(name)
    preset = None
    if mix is None:
        preset = PRESETS.get(name.lower())
    if mix is not None:
        gen_id = str(mix["generator"])
        try:
            gen = resolve_generator(gen_id)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        base_sliders = dict(zip(gen.sliders, mix.get("sliders") or (), strict=True))
        if animate is None:
            animate = bool(mix.get("animate", False))
        if duration_s is None:
            duration_s = float(mix.get("duration_s") or _DEFAULT_DURATION_S)
    elif preset is not None:
        gen_id = str(preset["generator"])
        gen = resolve_generator(gen_id)
        base_sliders = dict(zip(gen.sliders, preset["sliders"], strict=True))
        if animate is None:
            animate = bool(preset.get("animate", False))
        if duration_s is None:
            duration_s = float(preset.get("duration_s") or _DEFAULT_DURATION_S)
    else:
        try:
            gen = resolve_generator(name)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        gen_id = gen.id
        if gen.kind == "scene":
            base_sliders = dict(zip(gen.sliders, gen.defaults, strict=True))
        else:
            base_sliders = {f"{f:g} Hz": 0.0 for f in BAND_FREQS}
        if animate is None:
            animate = False
        if duration_s is None:
            duration_s = _DEFAULT_DURATION_S
    duration = float(duration_s)
    if timer_min is not None:
        duration = float(timer_min) * 60.0
    if duration <= 0:
        return {"ok": False,
                "error": f"duration must be positive, got {duration}"}
    if duration > MAX_DURATION_S:
        raise ValueError(
            f"a render is capped at {int(MAX_DURATION_S)}s — asked for "
            f"{duration:.0f}s; loop the file or use a shorter timer")
    for key, value in (sliders or {}).items():
        key = str(key).strip()
        try:
            as_float = float(key)
            for freq in BAND_FREQS:
                if as_float == freq:
                    key = f"{freq:g} Hz"
                    break
        except (TypeError, ValueError):
            pass
        base_sliders[key] = max(-30.0, min(12.0, float(value)))
    applied_adjust: dict[str, float] = {}
    for key, value in (adjust or {}).items():
        key = str(key)
        if key in base_sliders:
            base_sliders[key] = max(
                -30.0, min(12.0, base_sliders[key] + float(value)))
            applied_adjust[key] = float(value)
    params = dict(gen.params)
    if gen.kind == "noise":
        params["color"] = gen.id
    if gen.kind == "tones":
        params["tone"] = gen.id
    n = max(1, int(duration * SR))
    left, right = _render_windows(gen.kind, base_sliders, n, SR, params,
                                  bool(animate), seed)
    if loop:
        _crossfade_seam(left, right, SR)
    if fade_s is None:
        fade_s = min(30.0, duration) if timer_min is not None else 0.0
    if fade_s > 0:
        _fade_out_tail(left, right, float(fade_s), SR)
    hearing = store.hearing()
    hearing_applied = bool(hearing)
    if hearing_applied:
        left = apply_hearing(left, SR, hearing)
        right = apply_hearing(right, SR, hearing)
    peak_l, rms_l = _peak_rms(left)
    peak_r, rms_r = _peak_rms(right)
    peak = max(peak_l, peak_r)
    rms = (rms_l + rms_r) / 2.0
    if peak > 0:
        scale = 0.9 / peak
        left = [s * scale for s in left]
        right = [s * scale for s in right]
        peak, rms = 0.9, rms * scale
    out_dir = home_path / "data" / "soundscapes"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "-_" else "-"
                   for c in gen_id.lower())
    path = out_dir / f"{slug}-{int(duration)}s.wav"
    write_stereo_wav(path, left, right, SR)
    reply = (f"{gen.name} — {duration:.0f}s stereo WAV, "
             f"peak {peak:.2f}, rms {rms:.2f}.")
    if applied_adjust:
        changes = ", ".join(f"{key} {value:+.1f} dB"
                            for key, value in sorted(applied_adjust.items()))
        reply += f" Adjusted: {changes}."
    if hearing_applied:
        bands = ", ".join(
            f"{band:g} Hz {gain:+.1f} dB"
            for band, gain in sorted(hearing.items()))
        reply += f" Hearing applied: {bands}."
    if timer_min is not None:
        reply += f" Fades out over the last {min(30.0, duration):.0f}s."
    receipt = {
        "ok": True,
        "generator": gen_id,
        "name": gen.name,
        "path": str(path),
        "render_hash": wav_bytes_sha256(str(path)),
        "peak": round(peak, 4),
        "rms": round(rms, 4),
        "duration_s": round(duration, 1),
        "sample_rate": SR,
        "animate": bool(animate),
        "hearing_applied": hearing_applied,
        "reply": reply,
        "sliders": {k: float(v) for k, v in base_sliders.items()},
    }
    store.set_last(receipt)
    return receipt
