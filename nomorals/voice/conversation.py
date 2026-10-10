"""Voice conversation mode — talk, don't type.

Per-chat setting: when enabled, the chat becomes a voice conversation.
Incoming voice notes are transcribed into the brain, and her replies go
back out as voice notes automatically — no commands, no friction.

This is the chat-platform answer to real-time voice: WhatsApp/Telegram
can't do live duplex calls through the bot path, so the conversation
flows as low-latency voice-note turns. The pipeline is:

    voice note in → STT → brain (with voice context) → TTS → voice note out

Latency budget: STT and TTS run back-to-back; the brain generates text
once. No re-synthesis loops.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

_log = logging.getLogger("nomorals.voice.conversation")


def _store_path(voices_dir: str = "") -> str:
    if not voices_dir:
        from .catalogue import _default_voices_dir
        voices_dir = _default_voices_dir()
    return os.path.join(voices_dir, "voice_conversation.json")


def _load(voices_dir: str = "") -> dict[str, Any]:
    try:
        with open(_store_path(voices_dir), encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def _save(state: dict[str, Any], voices_dir: str = "") -> None:
    try:
        with open(_store_path(voices_dir), "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except OSError as exc:
        _log.warning("voice conversation state save failed: %s", exc)


def set_voice_mode(chat_key: str, enabled: bool,
                   voices_dir: str = "") -> bool:
    """Enable/disable voice conversation mode for a chat."""
    state = _load(voices_dir)
    chats = state.get("chats", {})
    if enabled:
        chats[chat_key] = {"enabled_at": time.time()}
    else:
        chats.pop(chat_key, None)
    state["chats"] = chats
    _save(state, voices_dir)
    return enabled


def voice_mode_on(chat_key: str, voices_dir: str = "") -> bool:
    return chat_key in _load(voices_dir).get("chats", {})


def list_voice_chats(voices_dir: str = "") -> list[str]:
    return sorted(_load(voices_dir).get("chats", {}))


def voice_turn(audio_path: str, *, chat_key: str = "",
               context: Any = None) -> dict[str, Any]:
    """One full voice-conversation turn.

    Transcribes the incoming note, generates a reply through the brain,
    synthesizes it, and returns ``{"ok", "heard", "reply_text",
    "audio_path", "voice_used", "latency_s", "timings"}`` — ``timings``
    breaks the turn into ``heard_s``/``thought_s``/``spoke_s`` so slow
    turns can be diagnosed. Delivery is the caller's job (it owns
    the chat reference); this function owns the pipeline.
    """
    t0 = time.time()
    timings: dict[str, float] = {}
    if not audio_path or not os.path.exists(audio_path):
        return {"ok": False, "reason": "audio not found"}

    # 1. hear
    heard = ""
    try:
        from ..tools.audio import stt as _stt
        res = _stt(audio_path, provider="auto", language="auto")
        heard = (res.get("text", "") or "").strip() if isinstance(
            res, dict) else str(res or "").strip()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"transcription failed: {exc}",
                "latency_s": round(time.time() - t0, 1)}
    timings["heard_s"] = round(time.time() - t0, 2)
    if not heard:
        return {"ok": False, "reason": "heard nothing intelligible",
                "latency_s": round(time.time() - t0, 1),
                "timings": timings}

    # 2. think — through the brain when available, else echo honestly
    reply_text = ""
    brain = None
    if context is not None:
        brain = getattr(context, "brain", None)
    if brain is not None:
        try:
            reply_text = brain.generate_voice_reply(heard, chat_key=chat_key)
        except AttributeError:
            try:
                reply_text = brain.generate(
                    f"[voice note] {heard}", chat_key=chat_key)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "reason": f"brain failed: {exc}",
                        "heard": heard,
                        "latency_s": round(time.time() - t0, 1)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"brain failed: {exc}",
                    "heard": heard, "latency_s": round(time.time() - t0, 1)}
    if not (reply_text or "").strip():
        return {"ok": False, "reason": "brain produced no reply",
                "heard": heard, "latency_s": round(time.time() - t0, 1),
                "timings": timings}
    timings["thought_s"] = round(time.time() - t0 - timings["heard_s"], 2)

    # 3. speak — short replies stay snappy; long ones still go through
    voice_used = ""
    try:
        from .catalogue import default_catalogue
        cat = default_catalogue()
        voice = cat.active_for_chat(chat_key)
        voice_used = voice.name if voice else ""
        out = cat.speak(reply_text.strip(), chat_key=chat_key)
        audio_out = out.get("path", "") if isinstance(out, dict) else ""
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"synthesis failed: {exc}",
                "heard": heard, "reply_text": reply_text.strip(),
                "latency_s": round(time.time() - t0, 1),
                "timings": timings}
    timings["spoke_s"] = round(
        time.time() - t0 - timings["heard_s"] - timings["thought_s"], 2)
    return {"ok": True, "heard": heard,
            "reply_text": reply_text.strip(),
            "audio_path": audio_out,
            "voice_used": voice_used,
            "latency_s": round(time.time() - t0, 1),
            "timings": timings}
