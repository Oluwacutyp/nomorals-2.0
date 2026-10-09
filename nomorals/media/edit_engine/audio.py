"""Style-agnostic audio: layered mixing, ducking, loudness.

An :class:`AudioMix` is a list of :class:`AudioLayer` (each placed at a
timeline offset with volume/fades/loop) plus mix-wide ducking and a
loudness target. Two-pass loudnorm (single-pass fallback) is the same
proven pattern as the rest of Devon's audio code.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...media_edit.videos import (
    MediaEditError,
    ffmpeg_path,
    run_ffmpeg,
    video_probe,
)
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

TARGET_LUFS = -14.0
_TRUE_PEAK_DB = -1.5
_LRA = 11.0


@dataclass
class AudioLayer:
    """One audio source on the mix timeline.

    ``offset`` = source start (seconds). ``start`` = timeline position.
    ``duration`` = timeline length (None = rest of source). ``loop``
    repeats the source to fill ``duration``.
    """
    path: str
    start: float = 0.0
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    loop: bool = False
    offset: float = 0.0
    duration: float | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise MediaEditError("AudioLayer needs a path")
        self.start = max(0.0, float(self.start))
        self.volume = float(self.volume)
        self.fade_in = max(0.0, float(self.fade_in))
        self.fade_out = max(0.0, float(self.fade_out))
        self.offset = max(0.0, float(self.offset))
        if self.duration is not None:
            self.duration = float(self.duration)
            if self.duration <= 0:
                raise MediaEditError("AudioLayer duration must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "start": self.start,
                "volume": self.volume, "fade_in": self.fade_in,
                "fade_out": self.fade_out, "loop": self.loop,
                "offset": self.offset, "duration": self.duration}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> "AudioLayer":
        if isinstance(data, str):
            return cls(path=data)
        return cls(**{k: data[k] for k in (
            "path", "start", "volume", "fade_in", "fade_out", "loop",
            "offset", "duration") if k in data})


@dataclass
class AudioMix:
    """The full mix: layers + ducking + loudness target.

    ``ducking``: "none" | "sidechain" (compressor keyed on the
    ``duck_key`` layer) | "volume" (deterministic gain automation from
    voice activity on the key layer). ``duck_key`` = layer index whose
    presence ducks the others (typically the voiceover).
    """
    layers: list[AudioLayer] = field(default_factory=list)
    ducking: str = "none"
    duck_key: int = 0
    duck_level: float = 0.30
    target_lufs: float = TARGET_LUFS

    def __post_init__(self) -> None:
        self.layers = [l if isinstance(l, AudioLayer)
                       else AudioLayer.from_dict(dict(l))
                       for l in self.layers]
        if self.ducking not in ("none", "sidechain", "volume"):
            raise MediaEditError(
                f"unknown ducking {self.ducking!r}; "
                "use none/sidechain/volume")
        if not 0.0 < self.duck_level <= 1.0:
            raise MediaEditError("duck_level must be in (0, 1]")
        if self.layers and not (0 <= self.duck_key < len(self.layers)):
            raise MediaEditError(
                f"duck_key {self.duck_key} out of range "
                f"(0..{len(self.layers) - 1})")

    def to_dict(self) -> dict[str, Any]:
        return {"layers": [l.to_dict() for l in self.layers],
                "ducking": self.ducking, "duck_key": self.duck_key,
                "duck_level": self.duck_level,
                "target_lufs": self.target_lufs}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioMix":
        return cls(layers=[AudioLayer.from_dict(l)
                           for l in data.get("layers", [])],
                   ducking=str(data.get("ducking", "none")),
                   duck_key=int(data.get("duck_key", 0)),
                   duck_level=float(data.get("duck_level", 0.30)),
                   target_lufs=float(data.get("target_lufs", TARGET_LUFS)))


def _loudnorm_filter(target: float = TARGET_LUFS,
                     measured: dict[str, float] | None = None) -> str:
    base = f"loudnorm=I={target}:TP={_TRUE_PEAK_DB}:LRA={_LRA}"
    if measured:
        base += (f":measured_I={measured['input_i']}"
                 f":measured_TP={measured['input_tp']}"
                 f":measured_LRA={measured['input_lra']}"
                 f":measured_thresh={measured['input_thresh']}"
                 f":offset={measured['target_offset']}"
                 f":linear=true")
    return base


def loudness_measure(path: str | Path) -> dict[str, float]:
    """First loudnorm pass → measured values. Raises on parse failure."""
    cmd = [ffmpeg_path(), "-hide_banner", "-nostats", "-i", str(path),
           "-af", _loudnorm_filter() + ":print_format=json",
           "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaEditError(f"loudness measurement failed: {exc}") from exc
    m = re.search(r"\{[^}]*\"input_i\"[^}]*\}", proc.stderr, re.S)
    if not m:
        raise MediaEditError("loudnorm did not print measurement JSON")
    try:
        data = json.loads(m.group(0))
        return {k: float(data[k]) for k in
                ("input_i", "input_tp", "input_lra",
                 "input_thresh", "target_offset")}
    except (ValueError, KeyError) as exc:
        raise MediaEditError(
            f"could not parse loudnorm JSON: {exc}") from exc


def normalize_loudness(src: str | Path, *,
                       out: str | Path | None = None,
                       target_lufs: float = TARGET_LUFS,
                       suffix: str = "loud") -> dict[str, Any]:
    """Two-pass loudnorm of ``src`` → AAC (single-pass fallback)."""
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such audio: {src}")
    out_p = Path(out) if out else p.with_name(f"{p.stem}.{suffix}.m4a")
    try:
        measured = loudness_measure(p)
        filt, two_pass = _loudnorm_filter(target_lufs, measured), True
    except MediaEditError as exc:
        _log.warning("loudness measure failed (%s) — single-pass", exc)
        filt, measured, two_pass = _loudnorm_filter(target_lufs), {}, False
    run = run_ffmpeg(["-i", str(p), "-af", filt,
                      "-ar", "48000", "-ac", "2",
                      "-c:a", "aac", "-b:a", "192k", str(out_p)],
                     timeout=600.0)
    return {"input": str(p), "output": str(out_p),
            "bytes": out_p.stat().st_size, "seconds": run["seconds"],
            "target_lufs": target_lufs, "two_pass": two_pass,
            "measured": measured}


def _layer_chain(idx: int, layer: AudioLayer, total: float,
                 tag: str) -> str:
    """One layer → placed, faded, volume-set stereo stream ``[tag]``."""
    p = Path(layer.path)
    if not p.exists():
        raise MediaEditError(f"no such audio layer: {layer.path}")
    info = video_probe(p)
    src_dur = info.get("duration") or 0.0
    want = layer.duration or max(total - layer.start, 0.1)
    filt = ["aresample=48000,aformat=channel_layouts=stereo"]
    if layer.offset > 0:
        filt.append(f"atrim=start={layer.offset:.3f},asetpts=PTS-STARTPTS")
    if layer.loop and src_dur > 0:
        # loop enough copies to cover the wanted span
        reps = max(1, int(math.ceil(want / src_dur)) + 1)
        filt.append(f"aloop=loop={reps}:size={int(src_dur * 48000)}")
    filt.append(f"atrim=duration={want:.3f},asetpts=PTS-STARTPTS")
    if abs(layer.volume - 1.0) > 1e-6:
        filt.append(f"volume={layer.volume:.4f}")
    if layer.fade_in > 0:
        filt.append(f"afade=t=in:st=0:d={layer.fade_in:.3f}")
    if layer.fade_out > 0:
        st = max(want - layer.fade_out, 0.0)
        filt.append(f"afade=t=out:st={st:.3f}:d={layer.fade_out:.3f}")
    filt.append(f"adelay={int(layer.start * 1000)}|{int(layer.start * 1000)},"
                f"apad=whole_dur={total:.3f}")
    return f"[{idx}:a]{','.join(filt)}[{tag}]"


def mix_layers(mix: AudioMix, duration: float,
               out: str | Path) -> dict[str, Any]:
    """Mix ``mix.layers`` over ``duration`` seconds → loudness-normalised AAC.

    Never raises on ffmpeg quirks it can honestly route around; raises
    MediaEditError on bad inputs (missing files, empty layers).
    """
    if not mix.layers:
        raise MediaEditError("AudioMix needs at least one layer")
    if duration <= 0:
        raise MediaEditError("mix duration must be > 0")
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    tags: list[str] = []
    for i, layer in enumerate(mix.layers):
        tag = f"l{i}"
        parts.append(_layer_chain(i, layer, duration, tag))
        tags.append(f"[{tag}]")

    final = "mix"
    if mix.ducking == "sidechain" and len(tags) > 1:
        key = f"l{mix.duck_key}"
        # the key feeds two consumers → asplit it first
        parts.append(f"[{key}]asplit=2[{key}sc][{key}m]")
        others = "".join(t if t != f"[{key}]" else f"[{key}m]"
                         for t in tags)
        parts.append(f"{others}amix=inputs={len(tags)}:normalize=0[pre]")
        parts.append(
            f"[pre][{key}sc]sidechaincompress=threshold=0.06:ratio=10"
            f":attack=12:release=400:makeup=1.4,alimiter=limit=0.95[{final}]")
    else:
        parts.append(f"{''.join(tags)}amix=inputs={len(tags)}:normalize=0,"
                     f"alimiter=limit=0.95[{final}]")
        if mix.ducking == "volume" and len(tags) > 1:
            _log.warning("volume ducking needs voice-activity automation; "
                         "fell back to plain mix (use sidechain)")

    inputs: list[str] = []
    for layer in mix.layers:
        inputs += ["-i", str(layer.path)]
    with tempfile.TemporaryDirectory(prefix="engmix-") as tmpd:
        stage1 = str(Path(tmpd) / "stage1.m4a")
        run_ffmpeg(inputs + ["-filter_complex", ";".join(parts),
                             "-map", f"[{final}]",
                             "-t", f"{duration:.3f}",
                             "-c:a", "aac", "-b:a", "192k", stage1],
                   timeout=600.0)
        try:
            measured = loudness_measure(stage1)
            norm, two_pass = _loudnorm_filter(mix.target_lufs, measured), True
        except MediaEditError as exc:
            _log.warning("loudness measure failed (%s) — single-pass", exc)
            norm, measured, two_pass = _loudnorm_filter(mix.target_lufs), {}, False
        run = run_ffmpeg(["-i", stage1, "-af", norm,
                          "-c:a", "aac", "-b:a", "192k", str(out_p)],
                         timeout=600.0)
    return {"output": str(out_p), "bytes": out_p.stat().st_size,
            "seconds": run["seconds"], "duration": round(duration, 3),
            "target_lufs": mix.target_lufs, "ducking": mix.ducking,
            "two_pass": two_pass, "layers": len(mix.layers)}


def make_music_bed(duration_s: float, out_path: str | Path,
                   seed: int = 7) -> str:
    """Deterministic ambient pad bed (numpy → wav). Returns the path.

    A quiet (-18 dB) minor pad with a slow filter sweep and fade in/out —
    a *bed* to sit under a voiceover, not a track. Real music comes
    through :class:`AudioLayer`; this is the fallback when none is
    supplied.
    """
    import numpy as np

    sr = 22050
    n = max(1, int(float(duration_s) * sr))
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    t = np.arange(n) / sr
    # Am – F – C – G roots, 8 s each, detuned saw-ish stack
    roots = [110.0, 87.31, 130.81, 98.0]
    seg = 8.0
    y = np.zeros(n)
    for i in range(int(math.ceil(float(duration_s) / seg))):
        f0 = roots[i % len(roots)]
        a, b = int(i * seg * sr), min(n, int((i + 1) * seg * sr))
        if b <= a:
            continue
        tt = t[a:b] - t[a]
        env = np.minimum(1.0, tt / 2.0) * np.minimum(
            1.0, np.maximum((b - a) / sr - tt, 0.0) / 2.0)
        env = np.clip(env, 0, 1)
        det = 1.0 + (rng.rand() - 0.5) * 0.004
        tone = (np.sin(2 * np.pi * f0 * tt)
                + 0.6 * np.sin(2 * np.pi * f0 * det * tt)
                + 0.35 * np.sin(2 * np.pi * f0 * 2.0 * tt + 0.7)
                + 0.2 * np.sin(2 * np.pi * f0 * 3.0 * tt + 1.9))
        # one-pole lowpass with a slow sweep (breathing feel)
        cutoff = 600 + 500 * np.sin(2 * np.pi * tt / seg + i)
        alpha = np.clip(2 * np.pi * cutoff / sr, 0.001, 1.0)
        lp = np.zeros_like(tone)
        acc = 0.0
        for j in range(len(tone)):
            acc += alpha[j] * (tone[j] - acc)
            lp[j] = acc
        y[a:b] += lp * env
    # gentle fade in/out + bed level
    f = int(min(n, sr * 1.5))
    y[:f] *= np.linspace(0, 1, f)
    y[-f:] *= np.linspace(1, 0, f)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 10 ** (-18 / 20)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import wave as _wave
    with _wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return str(p)
