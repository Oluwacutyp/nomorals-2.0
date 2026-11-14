"""LiveKit voice-call tools — the brain starting/ending a live call.

``voice_live_start`` joins a LiveKit room on a daemon thread and
returns immediately with the session id; ``voice_live_status`` reports
readiness and the running session's state; ``voice_live_end`` leaves
the room and returns the call report. Capability is ``net.out``: a
call leaves the box to a LiveKit server.

The think path here is the brain itself (``brain_for`` + a rolling
in-session history) — NOT the PartnerRuntime: that runtime is L5 and
this module is L4, so it cannot be imported here. The CLI
(``nm voice live start``) wires the full PartnerRuntime instead, with
durable conversation memory; the tool path keeps session-scoped
history only, and says so. STT reuses the ``transcribe`` tool (the
voice-note path); TTS reuses ``UniversalTTS``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..voice.livekit_call import (
    LiveKitCallSession,
    LiveKitConfig,
    LiveKitUnavailable,
    end_live_call,
    live_call_status,
    start_live_call,
)

_log = get_logger(__name__)

__all__ = ["register"]

#: Rolling brain history per live session (tool path only): the brain's
#: chat is stateless per call, so the session carries its own turns.
_HISTORY: dict[str, list[Any]] = {}
_MAX_HISTORY_TURNS = 20


def _voice_data_dir(context: Any) -> str:
    try:
        settings = getattr(context, "settings", None)
        return str(settings.audio.voice_data_dir or "voice_data")
    except Exception:  # noqa: BLE001 — settings are best effort here
        return "voice_data"


def _build_think(context: Any, session_id: str
                 ) -> Any:
    """Brain-backed think: transcript -> reply, with session history."""
    from ..llm.base import Message, SamplingParams
    from ..llm.brain import brain_for

    brain = brain_for(context)
    history: list[Any] = _HISTORY.setdefault(session_id, [])

    def _think(transcript: str) -> str:
        history.append(Message.user(transcript))
        messages = history[-_MAX_HISTORY_TURNS * 2:]
        try:
            response = brain.chat(
                messages,
                SamplingParams(temperature=0.7, max_tokens=1024),
                task_kind="voice-live",
                timeout_s=120.0,
            )
        except Exception as exc:  # noqa: BLE001 — a dead brain is silence
            _log.warning("live call brain failed: %s", exc)
            return ""
        text = str(getattr(response, "text", "") or "").strip()
        if text:
            history.append(Message.assistant(text))
        return text

    return _think


def _build_stt(context: Any) -> Any:
    """STT callable reusing the exact transcription path as inbound
    voice notes (the ``transcribe`` tool). One-shot per utterance."""

    def _stt(wav_path: str) -> str:
        tools = getattr(context, "tools", None)
        if tools is None:
            return ""
        outcome = tools.call("transcribe", path=wav_path, provider="auto")
        if not outcome.ok:
            return ""
        value = outcome.value or {}
        return str(value.get("text", "") or "").strip()

    _stt.supports_partial = False  # type: ignore[attr-defined]
    return _stt


def _build_tts(context: Any, data_dir: str) -> Any:
    """TTS callable reusing ``UniversalTTS`` (the ``nm voice say``
    engine). Returns {"path": wav} per sentence."""
    from ..voice.tts import UniversalTTS

    try:
        backend = str(context.settings.audio.tts_engine or "auto")
    except Exception:  # noqa: BLE001
        backend = "auto"
    engine = UniversalTTS(backend=backend)

    def _tts(text: str, profile: str) -> dict[str, Any]:
        out_path = str(Path(data_dir) / "tmp" /
                       f"live-tts-{int(time.time() * 1000)}.wav")
        return engine.speak(text, voice_name=profile or None,
                            out_path=out_path)

    return _tts


def _build_session(registry: Any, room: str, *, profile: str = "",
                   max_wait_remote: float = 120.0) -> LiveKitCallSession:
    """Assemble a live-call session from the registry's context."""
    context = getattr(registry, "context", None)
    data_dir = _voice_data_dir(context)
    session = LiveKitCallSession(
        think=lambda text: "",  # replaced below, once the id exists
        stt=_build_stt(context),
        tts=_build_tts(context, data_dir),
        config=LiveKitConfig.from_env(),
        room=room,
        data_dir=data_dir,
        profile=profile,
        max_wait_remote_s=max_wait_remote,
        bus=None,
    )
    # the think closure keys history by the real session id
    session.think = _build_think(context, session.session_id)
    return session


def register(registry: Any) -> None:
    """Register the live-call tools on the spine."""
    from ..core.policy import Capability

    @registry.register(
        "voice_live_start",
        description=(
            "Start a live voice call in a LiveKit room: Devon joins the "
            "room as a participant, hears whoever is speaking, and "
            "answers with her own voice. Returns immediately with the "
            "session id; the call runs in the background. Needs a "
            "LiveKit server (LIVEKIT_URL plus LIVEKIT_TOKEN or "
            "LIVEKIT_API_KEY + LIVEKIT_API_SECRET) and the livekit SDK "
            "(`nm voice live provision`). One live call at a time."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "room": "str — the LiveKit room name to join",
            "max_turns": "int (optional) — stop after this many turns "
                         "(0 = until goodbye / hangup)",
            "max_wait_remote": "float (optional) — seconds to wait for "
                              "someone to join before giving up "
                              "(default 120)",
            "profile": "str (optional) — TTS voice profile name",
        },
    )
    def _voice_live_start(room: str, max_turns: int = 0,
                         max_wait_remote: float = 120.0,
                         profile: str = "") -> dict[str, Any]:
        # cheap pre-flight first: no brain/TTS is built for a call that
        # cannot start (the session re-checks the same config itself).
        config = LiveKitConfig.from_env()
        missing = config.missing()
        if missing:
            return {"ok": False,
                    "error": "LiveKit is not configured: "
                             + "; ".join(missing),
                    "missing": missing}
        if not (room or "").strip():
            return {"ok": False, "error": "no room named",
                    "missing": "a room name"}
        try:
            session = _build_session(registry, room, profile=profile,
                                     max_wait_remote=max_wait_remote)
            start_live_call(session, max_turns=max_turns)
        except LiveKitUnavailable as exc:
            return {"ok": False, "error": str(exc),
                    "missing": exc.missing}
        except Exception as exc:  # noqa: BLE001 — start never raises
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "session_id": session.session_id,
                "room": room, "state": session.state}

    @registry.register(
        "voice_live_status",
        description=(
            "Live-call readiness and state: whether the livekit SDK is "
            "installed, whether a LiveKit server is configured, and the "
            "running call's state and last report if any. No connections "
            "are made by this call."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "session_id": "str (optional) — a session id; empty means "
                          "the running call",
        },
    )
    def _voice_live_status(session_id: str = "") -> dict[str, Any]:
        return {"ok": True, "status": live_call_status(session_id)}

    @registry.register(
        "voice_live_end",
        description=(
            "End the running live voice call and return its report "
            "(turns, barge-ins, what was actually said, why it ended). "
            "Refuses honestly when no call is running."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "session_id": "str (optional) — a session id; empty means "
                          "the running call",
        },
    )
    def _voice_live_end(session_id: str = "") -> dict[str, Any]:
        report = end_live_call(session_id)
        if report is None:
            return {"ok": False, "error": "no live call is running"}
        return {"ok": True, "report": report.to_dict()}
