"""Voice biometrics — owner recognition from voice.

Uses the acoustic voice-print (no model, no API) to answer one question:
"is this the owner speaking?" Incoming voice notes get scored against the
registered owner print; the score feeds the perimeter gating layer.

Honest scope: this is a *signal*, not a lock. Prints are statistical —
good enough to bias decisions, not to replace cryptographic auth.
Thresholds are conservative by default.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

_log = logging.getLogger("nomorals.voice.biometrics")

# Distance below which we call it a likely match. Calibrated: same voice
# scores ~0.03, clearly different voices ~0.7+. Conservative band between.
MATCH_THRESHOLD = 0.4
# Distance above which we call it confidently NOT the owner.
REJECT_THRESHOLD = 0.6


def _store_path(voices_dir: str = "") -> str:
    if not voices_dir:
        from .catalogue import _default_voices_dir
        voices_dir = _default_voices_dir()
    return os.path.join(voices_dir, "owner_biometric.json")


def register_owner_voice(audio_path: str, *,
                         voices_dir: str = "") -> dict[str, Any]:
    """Enroll the owner's voice from a reference clip.

    ``audio_path`` may be a single file or a list of files — multiple
    samples are averaged into one mean print (the Resemblyzer pattern:
    5–30s of varied speech beats one clip).
    """
    from .tts import voice_print
    paths = [audio_path] if isinstance(audio_path, str) else list(
        audio_path or [])
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return {"ok": False, "reason": "reference audio not found"}
    prints = []
    warnings: list[str] = []
    for p in paths:
        vp = voice_print(p)
        if not vp.get("ok"):
            return {"ok": False,
                    "reason": f"print failed for {p}: {vp.get('verdict')}"}
        prints.append(vp)
        warnings.extend(vp.get("warnings", []))
    vp = _mean_print(prints) if len(prints) > 1 else prints[0]
    record = {"print": vp, "enrolled_at": time.time(),
              "source": [os.path.basename(p) for p in paths],
              "num_samples": len(paths)}
    try:
        with open(_store_path(voices_dir), "w", encoding="utf-8") as fh:
            json.dump(record, fh)
    except OSError as exc:
        return {"ok": False, "reason": f"store failed: {exc}"}
    return {"ok": True, "enrolled_at": record["enrolled_at"],
            "num_samples": len(paths),
            "warnings": sorted(set(warnings))}


def _mean_print(prints: list[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric print fields across enrollment samples.

    voice_print() nests the speaker-discriminative features under
    ``"features"`` — those are averaged too, since that's what
    voice_print_distance() actually compares.
    """
    def _avg_dict(dicts: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = dict(dicts[0])
        nums: dict[str, list[float]] = {}
        for d in dicts:
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    nums.setdefault(k, []).append(float(v))
        for k, vals in nums.items():
            merged[k] = sum(vals) / len(vals)
        return merged

    base = _avg_dict(prints)
    feats = [p.get("features") for p in prints
             if isinstance(p.get("features"), dict)]
    if feats:
        base["features"] = _avg_dict(feats)
    base["ok"] = True
    return base


def owner_enrolled(voices_dir: str = "") -> bool:
    try:
        with open(_store_path(voices_dir), encoding="utf-8") as fh:
            return bool(json.load(fh).get("print"))
    except (OSError, ValueError):
        return False


def verify_voice(audio_path: str, *,
                 voices_dir: str = "",
                 spoof_check: bool = True) -> dict[str, Any]:
    """Score an audio clip against the enrolled owner voice.

    Runs the countermeasure FIRST (cheap spoof heuristic short-circuits
    before the print comparison), then the four-outcome :func:`decide`.

    Returns ``{"ok", "decision", "match", "distance", "confidence",
    "verdict", "spoof"}``. ``match`` is kept for backward
    compatibility (True/False/None); ``decision`` is the richer
    IDENTIFIED/UNKNOWN/AMBIGUOUS/REJECTED outcome.
    """
    from .tts import voice_print, voice_print_distance
    if not owner_enrolled(voices_dir):
        return {"ok": True, "match": None, "distance": None,
                "confidence": 0.0, "decision": "UNKNOWN",
                "verdict": "no owner voice enrolled — enroll with "
                           "register_owner_voice first"}
    if not audio_path or not os.path.exists(audio_path):
        return {"ok": False, "reason": "audio not found"}
    spoof = spoof_score(audio_path) if spoof_check else {
        "score": 0.0, "verdict": "skipped", "reasons": []}
    try:
        with open(_store_path(voices_dir), encoding="utf-8") as fh:
            owner_print = json.load(fh)["print"]
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": f"enrollment unreadable: {exc}"}
    vp = voice_print(audio_path)
    if not vp.get("ok"):
        return {"ok": True, "match": None, "distance": None,
                "confidence": 0.0, "decision": "UNKNOWN", "spoof": spoof,
                "verdict": f"print failed: {vp.get('verdict')}"}
    dist = voice_print_distance(owner_print, vp)
    result = decide(dist, spoof=spoof["score"],
                    voiced_s=float(spoof.get("voiced_s", 99.0)))
    decision = result["decision"]
    match = {"IDENTIFIED": True, "UNKNOWN": False,
             "AMBIGUOUS": None, "REJECTED": False}[decision]
    return {"ok": True, "match": match, "decision": decision,
            "distance": round(dist, 3),
            "confidence": result["confidence"],
            "verdict": result["verdict"], "spoof": spoof,
            "warnings": vp.get("warnings", [])}


# ---------------------------------------------------------------------------
# decide() — four outcomes, countermeasure-first (sweep)
# ---------------------------------------------------------------------------

#: distance ≤ this → IDENTIFIED
IDENTIFY_THRESHOLD = MATCH_THRESHOLD
#: spoof score ≥ this → REJECTED before any embedding comparison
SPOOF_REJECT = 0.75
#: minimum voiced seconds for a trustworthy decision
MIN_SPEECH_S = 1.5


def spoof_score(audio_path: str) -> dict[str, Any]:
    """Cheap anti-spoof heuristic — the countermeasure runs FIRST.

    Mined from the voice-indentification pipeline: the countermeasure
    is 40× cheaper than the embedder and short-circuits rejection, and
    non-speech must never reach the embedding stage. Signals here:

    - flat energy contour (replay loops / gain-flattened fakes)
    - flat zero-crossing contour (over-clean synthetic speech)
    - too-short voiced content (nothing to judge)

    Returns ``{"score" 0..1, "verdict", "reasons"}`` — score high =
    likely spoof/replay. A *heuristic*, documented as such: it biases
    toward caution, it does not prove fakery.
    """
    import math as _math
    import wave as _wave
    out: dict[str, Any] = {"score": 0.0, "verdict": "clean",
                           "reasons": []}
    try:
        with _wave.open(audio_path, "rb") as w:
            n = w.getnframes()
            raw = w.readframes(n)
            sr = w.getframerate() or 16000
    except Exception as exc:  # noqa: BLE001
        out.update(score=1.0, verdict=f"unreadable: {exc}")
        return out
    from array import array as _array
    samples = _array("h", raw)
    if len(samples) < sr:  # < 1s — nothing to judge
        out["reasons"].append("clip under 1s")
        out["score"] = max(out["score"], 0.6)
    frame = max(64, int(sr * 0.03))
    energies, zcrs = [], []
    for start in range(0, len(samples) - frame, frame):
        seg = samples[start:start + frame]
        e = sum(float(s) * s for s in seg) / frame
        energies.append(e)
        zc = sum(1 for a, b in zip(seg, seg[1:])
                 if (a >= 0) != (b >= 0)) / frame
        zcrs.append(zc)
    if not energies:
        out.update(score=1.0, verdict="no frames")
        return out
    mean_e = sum(energies) / len(energies)
    if mean_e < 1e-6:
        out.update(score=1.0, verdict="digital silence")
        return out
    # voiced frames only — silence must not flatten the contour stats
    voiced_e = [e for e in energies if e > mean_e * 0.5]
    voiced_z = [z for z, e in zip(zcrs, energies) if e > mean_e * 0.5]
    out["voiced_s"] = round(len(voiced_e) * frame / sr, 2)
    if out["voiced_s"] < MIN_SPEECH_S:
        out["reasons"].append(
            f"only {out['voiced_s']}s voiced (< {MIN_SPEECH_S}s)")
        out["score"] = max(out["score"], 0.65)

    def _cv(vals: list[float]) -> float:
        m = sum(vals) / len(vals)
        if m <= 1e-9 or len(vals) < 4:
            return 0.0
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return _math.sqrt(var) / m

    ecv = _cv(voiced_e)
    zcv = _cv(voiced_z)
    # natural speech: energy CV ~0.5–1.5, ZCR CV ~0.3–1.0. Flat = suspect.
    if ecv < 0.25:
        out["reasons"].append(f"flat energy contour (CV {ecv:.2f})")
        out["score"] = max(out["score"], 0.7)
    if zcv < 0.15:
        out["reasons"].append(f"flat zero-crossing contour (CV {zcv:.2f})")
        out["score"] = max(out["score"], 0.55)
    out["score"] = round(min(1.0, out["score"]), 3)
    if out["score"] >= SPOOF_REJECT:
        out["verdict"] = "likely spoof/replay — rejected"
    elif out["score"] >= 0.4:
        out["verdict"] = "suspicious — treat with caution"
    return out


def decide(distance: float | None, *,
           spoof: float = 0.0,
           voiced_s: float = 99.0) -> dict[str, Any]:
    """Four-outcome decision — never just a name.

    - ``IDENTIFIED``: confident match (distance ≤ threshold).
    - ``UNKNOWN``: nobody above threshold — a *success*, bias hard
      toward it (the mined rule: unknown ≠ failure).
    - ``AMBIGUOUS``: too close to call (the uncertain band) — ask for
      more audio, don't guess.
    - ``REJECTED``: failed liveness/spoof or insufficient speech.

    ``distance`` None (no enrollment / print failed) → UNKNOWN.
    """
    if voiced_s < MIN_SPEECH_S:
        return {"decision": "REJECTED",
                "verdict": f"insufficient speech ({voiced_s}s)",
                "confidence": 1.0}
    if spoof >= SPOOF_REJECT:
        return {"decision": "REJECTED",
                "verdict": "spoof countermeasure triggered",
                "confidence": float(spoof)}
    if distance is None or distance == float("inf"):
        return {"decision": "UNKNOWN",
                "verdict": "no comparable print",
                "confidence": 0.0}
    if distance <= MATCH_THRESHOLD:
        conf = max(0.0, 1.0 - distance / MATCH_THRESHOLD)
        return {"decision": "IDENTIFIED",
                "verdict": "likely the owner",
                "confidence": round(conf, 3)}
    if distance >= REJECT_THRESHOLD:
        return {"decision": "UNKNOWN",
                "verdict": "does not match the owner",
                "confidence": round(min(1.0, distance - REJECT_THRESHOLD
                                        + 0.5), 3)}
    return {"decision": "AMBIGUOUS",
            "verdict": "too close to call — need more audio",
            "confidence": 0.5}


def liveness_challenge(num_digits: int = 4,
                       ttl_s: float = 120.0) -> dict[str, Any]:
    """Anti-replay challenge: speak-back random digits.

    Returns ``{"phrase", "digits", "expires_at"}`` — play ``phrase``
    via TTS, transcribe the reply, pass both to
    :func:`check_liveness_response`. A recording can't answer a
    challenge it never heard.
    """
    import random as _random
    import time as _time
    digits = [_random.randrange(10) for _ in range(max(1, num_digits))]
    words = " ".join(str(d) for d in digits)
    return {"phrase": f"Please say these numbers back: {words}",
            "digits": digits,
            "expires_at": _time.time() + ttl_s}


def check_liveness_response(transcript: str,
                            challenge: dict[str, Any]) -> dict[str, Any]:
    """Verify a spoken challenge response. Never raises."""
    import time as _time
    digits = challenge.get("digits", []) or []
    if _time.time() > float(challenge.get("expires_at", 0) or 0):
        return {"ok": False, "reason": "challenge expired"}
    heard = [int(d) for d in __import__("re").findall(r"\d",
                                                      transcript or "")]
    if not digits:
        return {"ok": False, "reason": "empty challenge"}
    # ordered subsequence match — the digits in order, extras allowed
    it = iter(heard)
    matched = all(any(h == d for h in it) for d in digits)
    return {"ok": bool(matched),
            "reason": "" if matched else
            f"expected {digits}, heard {heard}"}
