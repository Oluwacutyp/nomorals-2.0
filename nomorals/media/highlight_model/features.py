"""8-feature extraction for the highlight model.

Same signals as score.py, plus the extra features the model wants.
Per-second feature vectors across the film → (n_seconds, 8).
"""

from __future__ import annotations


def extract_features(src: str) -> list[list[float]]:
    """Per-second 8-dim features. Zeros for unavailable signals (honest)."""
    from ..scene_intel.score import (
        _audio_energy_curve, _motion_curve, _face_curve, _dialogue_curve,
        _resample)

    src = str(src)
    audio = _audio_energy_curve(src)      # 10Hz
    motion = _motion_curve(src)           # 2Hz
    faces = _face_curve(src)              # 1Hz
    dialogue = _dialogue_curve(src)       # 0.1Hz

    # shot change rate from motion spikes
    shot_change = [0.0] * len(motion)
    for i in range(1, len(motion)):
        if motion[i] - motion[i - 1] > 0.4:  # hard cut spike
            shot_change[i] = 1.0

    # face size mean: reuse face curve as proxy (detailed size needs track.py)
    face_size = list(faces)

    # spectral flux proxy: audio derivative
    flux = [0.0] * len(audio)
    for i in range(1, len(audio)):
        flux[i] = max(0.0, audio[i] - audio[i - 1]) * 2.0

    # dialogue wpm: dialogue density scaled
    wpm = [min(1.0, d * 3.0) for d in dialogue]

    n_sec = 0
    for curve, hz in ((audio, 10.0), (motion, 2.0), (faces, 1.0),
                      (dialogue, 0.1), (shot_change, 2.0), (face_size, 1.0),
                      (flux, 10.0), (wpm, 0.1)):
        if curve:
            n_sec = max(n_sec, int(len(curve) / hz))
    if n_sec == 0:
        return []

    feats = []
    curves = [audio, motion, faces, dialogue, shot_change, face_size,
              flux, wpm]
    hzs = [10.0, 2.0, 1.0, 0.1, 2.0, 1.0, 10.0, 0.1]
    resampled = [_resample(c, n_sec) if c else [0.0] * n_sec
                 for c, hz in zip(curves, hzs)]
    # fix: _resample needs raw curve; resample each to n_sec
    resampled = []
    for c, hz in zip(curves, hzs):
        if c:
            # c is at hz; convert to per-second then pad/trim
            per_sec = _resample(c, max(1, int(len(c) / hz)))
            if len(per_sec) < n_sec:
                per_sec = per_sec + [per_sec[-1]] * (n_sec - len(per_sec))
            resampled.append(per_sec[:n_sec])
        else:
            resampled.append([0.0] * n_sec)
    for i in range(n_sec):
        feats.append([float(resampled[f][i]) for f in range(8)])
    return feats
