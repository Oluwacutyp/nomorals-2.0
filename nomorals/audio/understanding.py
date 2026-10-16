"""Audio understanding, first-class: structure, energy, sections, mood.

``audio/fingerprint.py`` already hears the acoustics (tempo, key, chord,
energy, music-vs-speech).  What it does NOT do is UNDERSTAND the clip:
what is happening, how it is structured, where the sections are, what
mood it carries.  This module is the understanding layer on top of the
native analysis — no new model, no new provider:

- SECTIONING — split the clip into sections from per-window energy +
  spectral flux (onset envelope): quiet vs loud, sparse vs dense.
  Sections get labels (``intro`` / ``main`` / ``break`` / ``outro`` /
  ``silence``) from position + energy, not from a classifier.
- ENERGY CURVE — per-second RMS energy, normalized 0..1, with
  ``peak``, ``valley``, and overall ``dynamics`` (peak-to-median ratio).
- MOOD — a small, explicit heuristic from tempo + energy + spectral
  centroid + music/speech: ``calm`` / ``tense`` / ``energetic`` /
  ``melancholic`` / ``neutral`` / ``speech``.  The mapping is a table
  in the code, not a black box, and ``mood_evidence`` names the inputs.
- WHAT'S HAPPENING — a one-paragraph ``describe()`` that combines
  kind (music/speech/mixed), tempo, key, sections, energy arc, and
  mood into the answer to "what's happening in this clip?".

All inputs come from ``fingerprint.analyze`` + a lightweight
per-window pass over the same samples (``read_mono`` + ``_frames``);
nothing here re-implements the acoustic estimators.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "AudioSection",
    "AudioUnderstanding",
    "understand_audio",
    "describe_audio",
    "energy_curve",
    "detect_sections",
    "infer_mood",
]


@dataclass
class AudioSection:
    """One structural section of a clip."""

    start: float
    end: float
    label: str  # intro | main | break | outro | silence
    mean_energy: float = 0.0
    peak_energy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 2), "end": round(self.end, 2),
                "label": self.label,
                "mean_energy": round(self.mean_energy, 3),
                "peak_energy": round(self.peak_energy, 3),
                "seconds": round(self.end - self.start, 2)}


@dataclass
class AudioUnderstanding:
    """First-class understanding of one audio clip."""

    path: str = ""
    ok: bool = False
    reason: str = ""
    seconds: float = 0.0
    kind: str = ""  # music | speech | mixed
    tempo_bpm: float = 0.0
    key: str = ""
    energy_curve: list[float] = field(default_factory=list)
    energy_peak: float = 0.0
    energy_valley: float = 0.0
    dynamics: float = 0.0
    sections: list[AudioSection] = field(default_factory=list)
    mood: str = "neutral"
    mood_evidence: dict[str, Any] = field(default_factory=dict)
    acoustic: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "ok": self.ok, "reason": self.reason,
            "seconds": self.seconds, "kind": self.kind,
            "tempo_bpm": self.tempo_bpm, "key": self.key,
            "energy_curve": [round(v, 3) for v in self.energy_curve],
            "energy_peak": round(self.energy_peak, 3),
            "energy_valley": round(self.energy_valley, 3),
            "dynamics": round(self.dynamics, 3),
            "sections": [s.to_dict() for s in self.sections],
            "mood": self.mood, "mood_evidence": dict(self.mood_evidence),
            "acoustic": dict(self.acoustic),
        }

    def describe(self) -> str:
        """One paragraph: what's happening in this clip."""
        if not self.ok:
            return f"couldn't listen to that ({self.reason})"
        parts = [f"{self.seconds:.0f}s of {self.kind}"]
        if self.tempo_bpm:
            parts.append(f"~{self.tempo_bpm:.0f} BPM")
        if self.key:
            parts.append(f"key of {self.key}")
        parts.append(f"mood: {self.mood}")
        if self.sections:
            sec = ", ".join(
                f"{s.label} {s.start:.0f}–{s.end:.0f}s"
                for s in self.sections)
            parts.append(f"structure: {sec}")
        parts.append(
            f"energy arcs from {self.energy_valley:.2f} to "
            f"{self.energy_peak:.2f} (dynamics {self.dynamics:.1f}x)")
        return "; ".join(parts) + "."


def energy_curve(samples: Sequence[float], sr: int, *,
                 window_s: float = 1.0) -> list[float]:
    """Per-window RMS energy, normalized 0..1 (peak = 1.0)."""
    if sr <= 0 or not samples:
        return []
    win = max(1, int(sr * window_s))
    curve: list[float] = []
    for start in range(0, len(samples), win):
        chunk = samples[start:start + win]
        if not chunk:
            continue
        rms = math.sqrt(sum(float(s) ** 2 for s in chunk) / len(chunk))
        curve.append(rms)
    peak = max(curve) if curve else 0.0
    if peak <= 0:
        return [0.0] * len(curve)
    return [v / peak for v in curve]


def detect_sections(curve: Sequence[float], *,
                    window_s: float = 1.0,
                    silence_below: float = 0.08) -> list[AudioSection]:
    """Section the energy curve into intro / main / break / outro / silence.

    Runs of below-``silence_below`` windows are ``silence``.  Non-silent
    runs are labeled by position + energy: the loudest run is ``main``,
    a leading quiet run is ``intro``, a trailing run is ``outro``, and
    a low-energy run between louder ones is a ``break``.
    """
    if not curve:
        return []
    # Run-length encode into (label-ish, start_idx, end_idx, mean).
    runs: list[dict[str, Any]] = []
    current_silent = curve[0] < silence_below
    start = 0
    for i, value in enumerate(curve[1:], 1):
        silent = value < silence_below
        if silent != current_silent:
            chunk = curve[start:i]
            runs.append({"silent": current_silent, "start": start, "end": i,
                         "mean": sum(chunk) / len(chunk),
                         "peak": max(chunk)})
            start, current_silent = i, silent
    chunk = curve[start:]
    runs.append({"silent": current_silent, "start": start, "end": len(curve),
                 "mean": sum(chunk) / len(chunk), "peak": max(chunk)})
    non_silent = [r for r in runs if not r["silent"]]
    loudest_mean = max((r["mean"] for r in non_silent), default=0.0)
    sections: list[AudioSection] = []
    for idx, run in enumerate(runs):
        if run["silent"]:
            label = "silence"
        else:
            if run["mean"] >= 0.9 * loudest_mean and loudest_mean > 0:
                label = "main"
            elif idx == 0:
                label = "intro"
            elif idx == len(runs) - 1:
                label = "outro"
            elif run["mean"] < 0.5 * loudest_mean:
                label = "break"
            else:
                label = "main"
        sections.append(AudioSection(
            start=run["start"] * window_s, end=run["end"] * window_s,
            label=label, mean_energy=run["mean"],
            peak_energy=run["peak"]))
    return sections


def infer_mood(analysis: Any) -> tuple[str, dict[str, Any]]:
    """Mood from acoustic features — an explicit table, not a black box.

    Speech-dominant clips are ``speech`` (mood is a music concept).
    Music: fast + bright + high energy -> ``energetic``; slow + dark +
    low energy -> ``melancholic``; slow + low energy + soft ->
    ``calm``; fast + loud + high centroid -> ``tense``; else ``neutral``.
    """
    if isinstance(analysis, dict):
        music = float(analysis.get("music_score", 0.5) or 0.5)
        tempo = float(analysis.get("tempo_bpm", 0.0) or 0.0)
        centroid = float(analysis.get("spectral_centroid_hz", 0.0) or 0.0)
        rms_db = float(analysis.get("rms_db", -96.0) or -96.0)
        key = str(analysis.get("key", "") or "")
    else:
        music = float(getattr(analysis, "music_score", 0.5) or 0.5)
        tempo = float(getattr(analysis, "tempo_bpm", 0.0) or 0.0)
        centroid = float(getattr(analysis, "spectral_centroid_hz", 0.0) or 0.0)
        rms_db = float(getattr(analysis, "rms_db", -96.0) or -96.0)
        key = str(getattr(analysis, "key", "") or "")
    evidence = {"music_score": round(music, 2), "tempo_bpm": tempo,
                "spectral_centroid_hz": centroid, "rms_db": rms_db,
                "key": key}
    if music <= 0.4:
        return "speech", evidence
    loud = rms_db > -18.0
    bright = centroid > 2500.0
    fast = tempo >= 120.0
    slow = 0.0 < tempo < 80.0
    minor = key.endswith("m") or "minor" in key.lower()
    if fast and bright and loud:
        return "tense", evidence
    if fast and loud:
        return "energetic", evidence
    if slow and minor:
        return "melancholic", evidence
    if slow and not loud:
        return "calm", evidence
    return "neutral", evidence


def understand_audio(path: str, *,
                     analysis: Any = None,
                     max_seconds: float | None = None) -> AudioUnderstanding:
    """Understand one clip: acoustics + structure + energy + mood.

    ``analysis`` may be a precomputed ``AudioAnalysis`` (or its dict)
    to skip re-reading the file; otherwise ``fingerprint.analyze``
    runs.  Failures land in ``ok/reason`` — never raised, never faked.
    """
    from .fingerprint import analyze as _analyze, read_mono

    result = AudioUnderstanding(path=str(path))
    if analysis is None:
        kw: dict[str, Any] = {}
        if max_seconds is not None:
            kw["max_seconds"] = max_seconds
        analysis = _analyze(path, **kw)
    acoustic = analysis.to_dict() if hasattr(analysis, "to_dict") else dict(analysis or {})
    result.acoustic = acoustic
    if not acoustic.get("ok"):
        result.reason = str(acoustic.get("reason") or "analysis failed")
        return result
    result.ok = True
    result.seconds = float(acoustic.get("seconds") or 0.0)
    music = float(acoustic.get("music_score", 0.5) or 0.5)
    result.kind = ("music" if music >= 0.6 else
                   "speech" if music <= 0.4 else "mixed audio")
    result.tempo_bpm = float(acoustic.get("tempo_bpm") or 0.0)
    result.key = str(acoustic.get("key") or "")
    result.mood, result.mood_evidence = infer_mood(acoustic)
    try:
        samples, sr = read_mono(path, target_sr=8000,
                                max_seconds=max_seconds or 60.0)
        curve = energy_curve(samples, sr, window_s=1.0)
    except Exception as exc:  # noqa: BLE001 — no samples, no curve
        _log.debug("understanding: energy curve failed: %s", exc)
        curve = []
    result.energy_curve = curve
    if curve:
        result.energy_peak = max(curve)
        sorted_curve = sorted(curve)
        median = sorted_curve[len(sorted_curve) // 2]
        result.energy_valley = min(curve)
        result.dynamics = round(result.energy_peak / max(1e-6, median), 2)
        result.sections = detect_sections(curve, window_s=1.0)
    return result


def describe_audio(path: str, **kw: Any) -> str:
    """One honest human paragraph about what is in the audio."""
    return understand_audio(path, **kw).describe()
