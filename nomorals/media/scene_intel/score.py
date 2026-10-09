"""Multi-modal highlight scoring — the "cool scene" detector.

Mined pattern (every working highlight system): no single signal is enough.
Combines:
- Audio energy (40%): RMS excitement peaks, adaptive rolling baseline
- Motion intensity (30%): frame-diff peaks (embarrassingly effective)
- Face/person presence (20%): more faces, longer = more important
- Dialogue density (10%): Whisper WPM spikes = quotable moments

Weights are the shipped heuristic. The trainable highlight_model/ in
nomorals/media/ learns better weights from data — same features in,
learned weights out. No "for later": the scaffold + training path ship now.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field


@dataclass
class ScoredSegment:
    index: int
    start: float
    end: float
    score: float              # 0-1 combined
    signals: dict = field(default_factory=dict)  # per-signal 0-1
    label: str = ""           # "action" | "emotional" | "dialogue" | "visual"


# Shipped heuristic weights (highlight_model/ learns better ones)
WEIGHTS = {"audio": 0.40, "motion": 0.30, "faces": 0.20, "dialogue": 0.10}


def _ffmpeg() -> str | None:
    from shutil import which
    return which("ffmpeg")


def _audio_energy_curve(src: str, sr: int = 22050) -> list[float]:
    """RMS energy at 10Hz, in dB relative to peak. Returns [] on failure."""
    ff = _ffmpeg()
    if not ff:
        return []
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav = tf.name
    try:
        subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", src,
             "-ac", "1", "-ar", str(sr), "-f", "wav", wav],
            check=True, capture_output=True, timeout=600)
        import wave, struct
        with wave.open(wav, "rb") as w:
            n = w.getnframes()
            raw = w.readframes(n)
        import array
        samples = array.array("h", raw)
        # 10Hz windows
        win = sr // 10
        curve: list[float] = []
        peak = 1e-9
        for i in range(0, len(samples), win):
            chunk = samples[i:i + win]
            if not chunk:
                break
            rms = math.sqrt(sum(s * s for s in chunk) / len(chunk)) / 32768.0
            curve.append(rms)
            peak = max(peak, rms)
        # dB relative to peak, clipped to [-60, 0], normalized 0-1
        return [max(0.0, 1.0 + (20 * math.log10(v / peak + 1e-9)) / 60.0)
                for v in curve]
    except Exception:
        return []
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


def _motion_curve(src: str, sample_fps: float = 2.0) -> list[float]:
    """Mean absolute frame difference at sample_fps, normalized 0-1."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(src_fps / sample_fps)))
    prev = None
    vals: list[float] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if prev is not None:
                diff = np.abs(gray.astype(float) - prev.astype(float)).mean() / 255.0
                vals.append(float(diff))
            prev = gray
        i += 1
    cap.release()
    if not vals:
        return []
    mx = max(vals) or 1e-9
    return [v / mx for v in vals]


def _face_curve(src: str, sample_fps: float = 1.0) -> list[float]:
    """Face count per sampled second, normalized. Empty without cv2."""
    try:
        import cv2
    except ImportError:
        return []
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade):
        return []
    face_cascade = cv2.CascadeClassifier(cascade)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(src_fps / sample_fps)))
    vals: list[float] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)
            vals.append(float(len(faces)))
        i += 1
    cap.release()
    if not vals:
        return []
    mx = max(vals) or 1e-9
    return [v / mx for v in vals]


def _dialogue_curve(src: str) -> list[float]:
    """Words-per-minute density via Whisper. Empty when STT unavailable."""
    try:
        from ...voice.stt import transcribe  # type: ignore
    except Exception:
        return []
    try:
        result = transcribe(str(src))
    except Exception:
        return []
    segments = result.get("segments", []) if isinstance(result, dict) else []
    if not segments:
        return []
    # Bin words into 10s windows
    bins: dict[int, int] = {}
    for seg in segments:
        start = float(seg.get("start", 0))
        words = len(str(seg.get("text", "")).split())
        bins[int(start // 10)] = bins.get(int(start // 10), 0) + words
    if not bins:
        return []
    mx = max(bins.values()) or 1
    last = max(bins)
    return [bins.get(i, 0) / mx for i in range(last + 1)]


def _resample(curve: list[float], n: int) -> list[float]:
    if not curve or n <= 0:
        return [0.0] * n
    if len(curve) == n:
        return curve
    out = []
    for i in range(n):
        pos = i * (len(curve) - 1) / max(1, n - 1)
        lo, hi = int(pos), min(len(curve) - 1, int(pos) + 1)
        frac = pos - lo
        out.append(curve[lo] * (1 - frac) + curve[hi] * frac)
    return out


def _adaptive_peaks(curve: list[float], min_rise: float = 0.25,
                    min_sustain: int = 2) -> list[int]:
    """Peak indices: rise above rolling median + sustain. (highlight-studio's
    rise & sustain filtering, simplified.)"""
    if len(curve) < 5:
        return []
    window = 9
    peaks = []
    for i in range(len(curve)):
        lo = max(0, i - window)
        baseline = sorted(curve[lo:i + 1])[len(curve[lo:i + 1]) // 2]
        if curve[i] - baseline >= min_rise:
            # sustain check
            if all(curve[min(len(curve) - 1, i + k)] >= baseline + min_rise * 0.5
                   for k in range(min_sustain)):
                peaks.append(i)
    return peaks


def score_segments(src: str, segments: list[tuple[float, float]],
                   *, weights: dict | None = None) -> list[ScoredSegment]:
    """Score segments 0-1 with multi-modal signals.

    Signals are computed per-second across the whole film, then each
    segment takes the max signal value within its window (a cool moment
    anywhere in the segment counts).
    """
    w = weights or WEIGHTS
    src = str(src)

    # Per-second signal curves across the film
    audio = _audio_energy_curve(src)          # 10Hz
    motion = _motion_curve(src)               # 0.5Hz
    faces = _face_curve(src)                  # 1Hz
    dialogue = _dialogue_curve(src)           # 0.1Hz

    # Normalize all to per-second
    dur_curves = []
    for curve, hz in ((audio, 10.0), (motion, 2.0), (faces, 1.0), (dialogue, 0.1)):
        if curve:
            n_sec = max(1, int(len(curve) / hz))
            dur_curves.append(_resample(curve, n_sec))
    n_sec = max((len(c) for c in dur_curves), default=0)
    sig = {
        "audio": _resample(audio, n_sec) if audio else [0.0] * n_sec,
        "motion": _resample(motion, n_sec) if motion else [0.0] * n_sec,
        "faces": _resample(faces, n_sec) if faces else [0.0] * n_sec,
        "dialogue": _resample(dialogue, n_sec) if dialogue else [0.0] * n_sec,
    }
    # Resample each to n_sec properly (audio was 10Hz etc.)
    for k, hz in (("audio", 10.0), ("motion", 2.0), ("faces", 1.0), ("dialogue", 0.1)):
        raw = {"audio": audio, "motion": motion,
               "faces": faces, "dialogue": dialogue}[k]
        sig[k] = _resample(raw, n_sec) if raw else [0.0] * n_sec

    out: list[ScoredSegment] = []
    for i, (s, e) in enumerate(segments):
        s_i, e_i = int(s), min(n_sec, int(e) + 1)
        if s_i >= n_sec:
            continue
        seg_sig = {k: max(v[s_i:e_i]) if v[s_i:e_i] else 0.0
                   for k, v in sig.items()}
        total_w = sum(w.get(k, 0) for k in seg_sig if seg_sig[k] > 0) or 1.0
        score = sum(seg_sig[k] * w.get(k, 0) for k in seg_sig) / sum(w.values())
        # Label by dominant signal
        dom = max(seg_sig, key=lambda k: seg_sig[k] * w.get(k, 0))
        label = {"audio": "emotional", "motion": "action",
                 "faces": "visual", "dialogue": "dialogue"}.get(dom, "")
        out.append(ScoredSegment(index=i, start=s, end=e,
                                 score=round(min(1.0, score), 3),
                                 signals={k: round(v, 3) for k, v in seg_sig.items()},
                                 label=label))
    out.sort(key=lambda s: s.score, reverse=True)
    return out
