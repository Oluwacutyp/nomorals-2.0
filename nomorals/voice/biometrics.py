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
    """Enroll the owner's voice from a reference clip."""
    from .tts import voice_print
    if not audio_path or not os.path.exists(audio_path):
        return {"ok": False, "reason": "reference audio not found"}
    vp = voice_print(audio_path)
    if not vp.get("ok"):
        return {"ok": False,
                "reason": f"print failed: {vp.get('verdict')}"}
    record = {"print": vp, "enrolled_at": time.time(),
              "source": os.path.basename(audio_path)}
    try:
        with open(_store_path(voices_dir), "w", encoding="utf-8") as fh:
            json.dump(record, fh)
    except OSError as exc:
        return {"ok": False, "reason": f"store failed: {exc}"}
    return {"ok": True, "enrolled_at": record["enrolled_at"],
            "warnings": vp.get("warnings", [])}


def owner_enrolled(voices_dir: str = "") -> bool:
    try:
        with open(_store_path(voices_dir), encoding="utf-8") as fh:
            return bool(json.load(fh).get("print"))
    except (OSError, ValueError):
        return False


def verify_voice(audio_path: str, *,
                 voices_dir: str = "") -> dict[str, Any]:
    """Score an audio clip against the enrolled owner voice.

    Returns ``{"ok", "match": bool|None, "distance", "confidence",
    "verdict"}``. ``match`` is None when no owner voice is enrolled.
    """
    from .tts import voice_print, voice_print_distance
    if not owner_enrolled(voices_dir):
        return {"ok": True, "match": None, "distance": None,
                "confidence": 0.0,
                "verdict": "no owner voice enrolled — enroll with "
                           "register_owner_voice first"}
    if not audio_path or not os.path.exists(audio_path):
        return {"ok": False, "reason": "audio not found"}
    try:
        with open(_store_path(voices_dir), encoding="utf-8") as fh:
            owner_print = json.load(fh)["print"]
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": f"enrollment unreadable: {exc}"}
    vp = voice_print(audio_path)
    if not vp.get("ok"):
        return {"ok": True, "match": None, "distance": None,
                "confidence": 0.0,
                "verdict": f"print failed: {vp.get('verdict')}"}
    dist = voice_print_distance(owner_print, vp)
    if dist <= MATCH_THRESHOLD:
        match, confidence = True, max(0.0, 1.0 - dist / MATCH_THRESHOLD)
        verdict = "likely the owner"
    elif dist >= REJECT_THRESHOLD:
        match, confidence = False, min(1.0, (dist - REJECT_THRESHOLD + 1.0)
                                       / 2.0)
        verdict = "likely not the owner"
    else:
        match, confidence = None, 0.5
        verdict = "uncertain — too close to call"
    return {"ok": True, "match": match, "distance": round(dist, 3),
            "confidence": round(confidence, 3), "verdict": verdict,
            "warnings": vp.get("warnings", [])}
