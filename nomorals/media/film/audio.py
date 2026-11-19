"""Film audio — the mix, the duck, the loudness.

FilmCraft's mixer: a mission (voice) track, a music bed ducked
under it, named markers on the timeline, and loudness meters that
tell the truth. This module is the same job with numpy + ffmpeg:

* :func:`mix_audio` — N sources at timeline offsets, each with a
  gain; the bed is multiplied by a ducking envelope that dips
  under speech and ramps back (attack/release, no clicks).
* :func:`measure_loudness` — EBU R128 integrated loudness via
  ffmpeg's ``loudnorm`` filter, honestly labelled; a failed
  measurement falls back to an RMS estimate and says so.
* :func:`normalize_loudness` — one gain stage to a target
  integrated loudness, clamped to ±12 dB, measured before and
  after.

All temporary files live in the caller's ``workdir`` — never
``/tmp`` (the box's tmpfs is 512MB and runtime restarts wipe it).
"""

from __future__ import annotations

import json
import math
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from ..motion_studio._core import ffmpeg_path

__all__ = [
    "ducking_envelope",
    "measure_loudness",
    "mix_audio",
    "normalize_loudness",
    "read_wav_mono",
    "write_wav",
]

DEFAULT_SR = 22050


def _run_ffmpeg(args: list[str], *, timeout: float = 180.0) -> str:
    proc = subprocess.run([ffmpeg_path(), *args],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): "
            f"{proc.stderr.strip()[-500:]}")
    return proc.stderr


def decode_to_wav(src: str | Path, out_wav: str | Path,
                  *, sr: int = DEFAULT_SR) -> Path:
    """Any audio/video file -> mono 16-bit PCM wav at ``sr`` Hz."""
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(["-y", "-i", str(src), "-vn", "-ar", str(sr), "-ac", "1",
                 "-c:a", "pcm_s16le", str(out)])
    return out


def read_wav_mono(path: str | Path, *, sr: int = DEFAULT_SR) -> np.ndarray:
    """A wav file as a float32 mono array at ``sr`` Hz (resampled
    through ffmpeg when the file's rate differs)."""
    path = Path(path)
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        n_channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    if n_channels == 1:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif n_channels == 2:
        stereo = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        data = stereo.reshape(-1, 2).mean(axis=1) / 32768.0
    else:
        raise RuntimeError(f"unsupported channel count {n_channels} in {path}")
    if rate != sr:
        tmp = path.with_suffix(".resample.wav")
        decode_to_wav(path, tmp, sr=sr)
        return read_wav_mono(tmp, sr=sr)
    return data


def write_wav(path: str | Path, samples: np.ndarray,
              *, sr: int = DEFAULT_SR) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return out


def ducking_envelope(duration_s: float,
                     speech: list[tuple[float, float]],
                     *, sr: int = DEFAULT_SR, duck_gain: float = 0.25,
                     attack_s: float = 0.15,
                     release_s: float = 0.4) -> np.ndarray:
    """A gain envelope (1.0 outside speech, ``duck_gain`` under it)
    with raised-cosine attack/release ramps — no clicks at the
    edges."""
    n = max(1, int(round(duration_s * sr)))
    env = np.ones(n, dtype=np.float32)
    attack = max(1, int(round(attack_s * sr)))
    release = max(1, int(round(release_s * sr)))
    for start, end in speech:
        s = max(0, int(round(start * sr)))
        e = min(n, int(round(end * sr)))
        if e <= s:
            continue
        env[s:e] = duck_gain
        # attack ramp before the dip
        a0 = max(0, s - attack)
        if s > a0:
            ramp = 0.5 * (1 + np.cos(np.linspace(math.pi, 0, s - a0)))
            env[a0:s] = 1.0 - (1.0 - duck_gain) * ramp
        # release ramp after the dip
        r1 = min(n, e + release)
        if r1 > e:
            ramp = 0.5 * (1 - np.cos(np.linspace(0, math.pi, r1 - e)))
            env[e:r1] = duck_gain + (1.0 - duck_gain) * ramp
    return env


def mix_audio(sources: list[dict[str, Any]], *, duration_s: float,
              sr: int = DEFAULT_SR, workdir: str | Path,
              ducking: dict[str, Any] | None = None) -> np.ndarray:
    """Mix ``sources`` (``{"path", "at", "gain"}``) into one mono
    float32 array of ``duration_s`` seconds. ``ducking`` names the
    bed index, the speech windows and the duck gain — the bed is
    multiplied by the envelope, everything else is left alone."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    n = max(1, int(round(duration_s * sr)))
    mix = np.zeros(n, dtype=np.float32)
    decoded: list[np.ndarray] = []
    for i, src in enumerate(sources):
        wav = work / f"src_{i}.wav"
        decode_to_wav(src["path"], wav, sr=sr)
        data = read_wav_mono(wav, sr=sr)
        decoded.append(data)
        at = int(round(float(src.get("at", 0.0)) * sr))
        gain = float(src.get("gain", 1.0))
        end = min(n, at + len(data))
        if end > at:
            mix[at:end] += data[:end - at] * gain
    if ducking:
        bed_index = int(ducking.get("bed_index", 1))
        speech = [(float(s), float(e))
                  for s, e in ducking.get("speech", [])]
        duck_gain = float(ducking.get("duck_gain", 0.25))
        if 0 <= bed_index < len(decoded):
            # re-mix the bed alone under the envelope: subtract the
            # unducked bed, add it back ducked
            src = sources[bed_index]
            at = int(round(float(src.get("at", 0.0)) * sr))
            gain = float(src.get("gain", 1.0))
            data = decoded[bed_index]
            end = min(n, at + len(data))
            if end > at:
                mix[at:end] -= data[:end - at] * gain
                env = ducking_envelope(duration_s, speech, sr=sr,
                                       duck_gain=duck_gain)
                mix[at:end] += data[:end - at] * gain * env[at:end]
    return np.clip(mix, -1.0, 1.0)


def _loudnorm_json(path: str | Path) -> dict[str, Any] | None:
    cmd = [ffmpeg_path(), "-i", str(path), "-vn",
           "-af", "loudnorm=print_format=json", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180.0)
    text = proc.stderr
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def measure_loudness(path: str | Path) -> dict[str, Any]:
    """EBU R128 loudness of ``path``. Returns the loudnorm report
    (``integrated`` in LUFS, ``true_peak`` in dBTP, ``lra``) with
    ``method: "loudnorm"``; when loudnorm cannot measure, an RMS
    estimate labelled ``method: "rms-estimate"`` — never a silent
    zero."""
    data = _loudnorm_json(path)
    if data is not None:
        try:
            return {
                "method": "loudnorm",
                "integrated": float(data["input_i"]),
                "true_peak": float(data["input_tp"]),
                "lra": float(data["input_lra"]),
                "threshold": float(data["input_thresh"]),
            }
        except (KeyError, TypeError, ValueError):
            pass
    # fallback: RMS estimate, honestly labelled
    wav = Path(path)
    if wav.suffix.lower() != ".wav":
        tmp = wav.with_suffix(".measure.wav")
        try:
            decode_to_wav(wav, tmp)
            wav = tmp
        except RuntimeError:
            return {"method": "rms-estimate", "integrated": float("-inf"),
                    "true_peak": float("-inf"), "lra": 0.0,
                    "reason": "could not decode audio"}
    try:
        samples = read_wav_mono(wav)
    except Exception as exc:  # noqa: BLE001
        return {"method": "rms-estimate", "integrated": float("-inf"),
                "true_peak": float("-inf"), "lra": 0.0,
                "reason": f"could not read audio: {exc}"}
    if len(samples) == 0:
        return {"method": "rms-estimate", "integrated": float("-inf"),
                "true_peak": float("-inf"), "lra": 0.0}
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(samples.astype(np.float64))))
    return {"method": "rms-estimate",
            "integrated": 20.0 * math.log10(rms) - 0.691 if rms > 0
            else float("-inf"),
            "true_peak": 20.0 * math.log10(peak) if peak > 0
            else float("-inf"),
            "lra": 0.0}


def normalize_loudness(path: str | Path, out_path: str | Path,
                       *, target_i: float = -14.0,
                       workdir: str | Path | None = None) -> dict[str, Any]:
    """Gain ``path`` toward ``target_i`` LUFS (clamped to ±12 dB)
    and write ``out_path``. Returns before/after measurements and
    the applied gain."""
    before = measure_loudness(path)
    integrated = before.get("integrated")
    if integrated is None or integrated == float("-inf"):
        raise RuntimeError(
            f"cannot normalize {path}: no measurable loudness "
            f"({before.get('reason', 'silent input')})")
    gain_db = max(-12.0, min(12.0, float(target_i) - float(integrated)))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # the gain stage, then a limiter at -1 dBTP so a hot peak never
    # clips after normalisation
    _run_ffmpeg(["-y", "-i", str(path), "-vn",
                 "-af", f"volume={gain_db:.2f}dB,"
                        "alimiter=limit=0.891:level=false",
                 "-c:a", "pcm_s16le", str(out)])
    after = measure_loudness(out)
    return {"before": before, "after": after, "gain_db": gain_db,
            "target_i": float(target_i), "path": str(out)}
