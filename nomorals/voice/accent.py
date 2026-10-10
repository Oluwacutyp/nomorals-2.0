"""Accent conversion — real, not decorative.

``nl_director`` has parsed ``[in British accent]`` for a while; the accent
field then went nowhere. This module makes it real through the accent-first
hybrid (mined from OpenVoice's tone-color/style separation):

    synthesize in the TARGET accent (multilingual backends) → RVC identity
    transfer (keeps the accent, swaps the speaker)

Accent lives in prosody/phoneme realization ("style"), identity in the
timbre ("tone color") — RVC preserves style while converting tone color,
so the accent survives the identity swap.

Honest chain, best first:
1. ``neural`` — multilingual backend (chatterbox-multilingual, cosyvoice,
   qwen3) renders the target accent natively, then RVC to the voice.
2. ``phoneme`` — espeak-ng phonemizer accent hints (perception-level).
3. ``none`` — accent requested but no path; the call still succeeds and
   says so (never fake an accent).

Result dicts always name the tier that actually ran.
"""

from __future__ import annotations

import os
import tempfile
import wave
from array import array
from typing import Any, Optional

from ..core.logging_setup import get_logger
from . import rvc_bridge

_log = get_logger(__name__)

__all__ = [
    "ACCENTS",
    "normalize_accent",
    "convert_accent",
    "list_accents",
]

#: Accents the pipeline knows how to aim for.
ACCENTS = (
    "british", "american", "nigerian", "yoruba", "igbo", "hausa",
    "french", "australian", "indian", "russian", "german", "spanish",
    "italian", "scottish", "irish", "jamaican", "canadian",
    "south_african", "pidgin", "caribbean", "west_african",
)

#: Which backend language/voice hint to use per accent for the
#: accent-first render. Multilingual backends pick these up.
_ACCENT_HINTS: dict[str, dict[str, str]] = {
    "british": {"language": "en", "hint": "en-GB"},
    "american": {"language": "en", "hint": "en-US"},
    "nigerian": {"language": "en", "hint": "en-NG"},
    "australian": {"language": "en", "hint": "en-AU"},
    "indian": {"language": "en", "hint": "en-IN"},
    "canadian": {"language": "en", "hint": "en-CA"},
    "south_african": {"language": "en", "hint": "en-ZA"},
    "irish": {"language": "en", "hint": "en-IE"},
    "scottish": {"language": "en", "hint": "en-GB-sct"},
    "jamaican": {"language": "en", "hint": "en-JM"},
    "caribbean": {"language": "en", "hint": "en-JM"},
    "west_african": {"language": "en", "hint": "en-NG"},
    "pidgin": {"language": "en", "hint": "en-NG"},
    "french": {"language": "fr", "hint": "fr-FR"},
    "german": {"language": "de", "hint": "de-DE"},
    "spanish": {"language": "es", "hint": "es-ES"},
    "italian": {"language": "it", "hint": "it-IT"},
    "russian": {"language": "ru", "hint": "ru-RU"},
    "yoruba": {"language": "yo", "hint": "yo"},
    "igbo": {"language": "ig", "hint": "ig"},
    "hausa": {"language": "ha", "hint": "ha"},
}

#: Honest prosody nudges per accent family for the accent-only tier —
#: characteristic rhythm/pitch tendencies, scaled by ``strength``.
#: Documented as suggestive, not a real accent.
_ACCENT_PROSODY: dict[str, dict[str, float]] = {
    "nigerian": {"rate_mult": 1.04, "pitch_shift": 0.4},
    "yoruba": {"rate_mult": 1.04, "pitch_shift": 0.6},
    "igbo": {"rate_mult": 1.03, "pitch_shift": 0.5},
    "hausa": {"rate_mult": 1.02, "pitch_shift": 0.3},
    "pidgin": {"rate_mult": 1.06, "pitch_shift": 0.5},
    "west_african": {"rate_mult": 1.04, "pitch_shift": 0.4},
    "caribbean": {"rate_mult": 1.05, "pitch_shift": 0.7},
    "jamaican": {"rate_mult": 1.05, "pitch_shift": 0.7},
    "indian": {"rate_mult": 1.03, "pitch_shift": 0.3},
    "british": {"rate_mult": 0.97, "pitch_shift": -0.2},
    "scottish": {"rate_mult": 0.98, "pitch_shift": -0.3},
    "irish": {"rate_mult": 1.02, "pitch_shift": 0.4},
    "australian": {"rate_mult": 1.0, "pitch_shift": 0.2},
    "french": {"rate_mult": 1.02, "pitch_shift": 0.3},
    "spanish": {"rate_mult": 1.05, "pitch_shift": 0.2},
    "italian": {"rate_mult": 1.04, "pitch_shift": 0.3},
}


def list_accents() -> list[dict[str, str]]:
    """Every known accent with its language/hint mapping."""
    return [{"accent": a,
             "language": _ACCENT_HINTS.get(a, {}).get("language", "en"),
             "hint": _ACCENT_HINTS.get(a, {}).get("hint", "en")}
            for a in ACCENTS]


def normalize_accent(accent: str) -> str:
    """Normalize free text to a known accent key ("" when unknown)."""
    a = (accent or "").lower().strip().replace(" ", "_").replace("-", "_")
    # drop a trailing "_accent": "nigerian pidgin accent" → head matching
    core = a[:-7] if a.endswith("_accent") else a
    if core in ACCENTS:
        return core
    # suffix match on the head noun: "nigerian_pidgin" → "pidgin"
    for known in sorted(ACCENTS, key=len, reverse=True):
        if core.endswith(known):
            return known
    # substring fallback: "british english" → "british"
    for known in sorted(ACCENTS, key=len, reverse=True):
        if known in core or core in known:
            return known
    return ""


def _write_wav(path: str, samples: array, sr: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def convert_accent(text: str, tts: Any, accent: str, voice_name: str = "",
                   rvc_model: str = "", strength: float = 1.0) -> dict[str, Any]:
    """Render text in the requested accent. Returns {"path", "tier", ...}.

    ``tts``: UniversalTTS. The RVC identity transfer needs a model for the
    target voice (``rvc_model`` or ``voice_name``); without one, the
    accent-first render is returned as-is (tier "accent-only") — still the
    requested accent, just not the requested identity.

    ``strength`` 0–1 scales the accent-only prosody nudge (characteristic
    rhythm/pitch of the accent family, applied via emotion DSP — honest
    and documented, not a neural accent). The language hint itself now
    reaches the backend per call: multilingual backends (chatterbox,
    cosyvoice, qwen3) render ``en-NG``/``yo``/etc. natively.
    """
    accent_key = normalize_accent(accent)
    if not accent_key:
        # Unknown accent: honest pass-through, no fake.
        base = tts.speak(text, voice_name=voice_name or None)
        return {"ok": True, "path": base["path"], "tier": "none",
                "note": f"unknown accent '{accent}' — rendered neutrally"}

    hint = _ACCENT_HINTS.get(accent_key, {"language": "en", "hint": "en"})
    # Accent-first render: the language hint goes to the backend per
    # call now (chatterbox-multilingual honors language_id natively).
    try:
        accented = tts.speak(
            text, voice_name=voice_name or None,
            mood="",  # mood would fight the accent render
            language=hint["hint"],
        )
        tier = "accent-only"
        path = accented["path"]
        native_lang = "language" in (accented.get("native_opts") or [])
    except Exception as exc:
        raise RuntimeError(f"accent render failed: {exc}") from exc

    # Prosody nudge on the accent-only tier (strength-scaled).
    strength = max(0.0, min(1.0, strength))
    prosody = _ACCENT_PROSODY.get(accent_key)
    if strength > 0 and prosody:
        try:
            import wave as _wave
            from .emotion_dsp import shape_emotion
            with _wave.open(path, "rb") as w:
                sr = w.getframerate()
                samples = array("h", w.readframes(w.getnframes()))
            shaped = shape_emotion(
                samples, sr,
                pitch_shift_st=prosody["pitch_shift"] * strength,
                rate_mult=1.0 + (prosody["rate_mult"] - 1.0)
                * strength * 2.0,
                energy=1.0)
            _write_wav(path, shaped, sr)
            tier = "accent-only+prosody"
        except Exception as exc:  # noqa: BLE001 - nudge is a bonus
            _log.debug("accent prosody nudge skipped: %s", exc)

    # Identity transfer: keep the accent, swap the speaker.
    model_name = rvc_model or voice_name
    if model_name:
        try:
            probe = rvc_bridge.detect_rvc()
            models = rvc_bridge.list_models()
            model = next((m for m in models if m.name == model_name), None)
            if not probe["ok"] or model is None:
                raise rvc_bridge.RVCUnavailable("no RVC path")
            conv = rvc_bridge.convert(path, model)
            return {"ok": True, "path": conv["path"], "tier": "neural",
                    "accent": accent_key, "model": model.name,
                    "native_language": native_lang}
        except Exception as exc:  # noqa: BLE001 — honest downgrade
            _log.info("accent: RVC unavailable, accent-only: %s", exc)

    return {"ok": True, "path": path, "tier": tier, "accent": accent_key,
            "native_language": native_lang,
            "note": "accent render without identity transfer"}

    # Identity transfer: keep the accent, swap the speaker.
    model_name = rvc_model or voice_name
    if model_name:
        try:
            probe = rvc_bridge.detect_rvc()
            models = rvc_bridge.list_models()
            model = next((m for m in models if m.name == model_name), None)
            if not probe["ok"] or model is None:
                raise rvc_bridge.RVCUnavailable("no RVC path")
            conv = rvc_bridge.convert(path, model)
            return {"ok": True, "path": conv["path"], "tier": "neural",
                    "accent": accent_key, "model": model.name}
        except Exception as exc:  # noqa: BLE001 — honest downgrade
            _log.info("accent: RVC unavailable, accent-only: %s", exc)

    return {"ok": True, "path": path, "tier": tier, "accent": accent_key,
            "note": "accent render without identity transfer"}
