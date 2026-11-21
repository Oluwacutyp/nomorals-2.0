"""LiveKit voice-call tools — the brain starting/ending a live call.

``voice_live_start`` joins a LiveKit room on a daemon thread and
returns immediately with the session id; ``voice_live_status`` reports
readiness and the running session's state; ``voice_live_end`` leaves
the room and returns the call report. Capability is ``net.out``: a
call leaves the box to a LiveKit server.

The video rung lives here too (``video_live_*``): Devon publishes a
real video track in the room — a live title card, a still image, or
her own motion-studio scene rendered frame by frame — and watches the
room's video, saving one snapshot PNG per publisher per second.
``video_live_snapshot`` returns the latest frame she has seen, so the
brain (or the owner, through /videocall) can look at the room without
sitting in it. The video session joins as a second participant
(identity ``devon-video``), so it can run beside the voice call.

The think path here is the brain itself (``brain_for`` + a rolling
in-session history) — NOT the PartnerRuntime: the CLI
(``nm voice live start``) wires the full PartnerRuntime instead, with
durable conversation memory; the tool path keeps session-scoped
history only, and says so. STT reuses the ``transcribe`` tool (the
voice-note path); TTS reuses ``UniversalTTS``. This module is pinned
L5 in ``tests/test_layering.py`` (a tool surface over the L5
tool-loop organ) — that pin is what lets agent mode import the loop
below; the plain think path still imports nothing above L3.

Agent mode (``voice_live_start(..., agent=True)``) runs each turn
through the real ReAct tool loop (``agents.partner.tool_loop``) —
the brain sees the live registry and can DO things mid-call, not
just talk. This module is pinned L5 in ``tests/test_layering.py``
for exactly that composition (a tool surface over the L5 tool-loop
organ, the same shape as ``tools.braineval``). The loop is
voice-bounded (3 iterations, 60 s, 3 tool calls per turn); its
stall/budget degradation is the honest path when the model loops
or dies. Per-turn tools ride ``think.drain_tools_used()`` into the
call report — the session never imports the loop itself.

Face watch (``video_live_start(..., face_watch=True)``) runs inbound
room frames through the facemesh rung (landmarks → blendshapes +
head pose) at the snapshot cadence; mediapipe/model absent means
the report says ``unavailable`` and the video still runs.
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
    LiveKitVideoSession,
    end_live_call,
    end_video_session,
    get_video_session,
    image_frame_source,
    live_call_status,
    scene_frame_source,
    start_live_call,
    start_video_session,
    title_frame_source,
    video_session_status,
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


def _build_agent_think(context: Any, registry: Any, session_id: str,
                       tools_used: list[str]) -> Any:
    """Agent-mode think: the turn runs through the real ReAct loop.

    The brain is the loop's LLM, the live registry is its tools.
    Voice-bounded: 3 iterations, 60 s total, 3 tool calls per turn —
    a call is a conversation, not a batch job, and a slow tool must
    not hold the room silent. The loop's own stall/budget
    degradation is the honest path: think returns "" and the turn
    is skipped, never faked. Per-turn tools ride
    ``drain_tools_used()`` — the session's seam for the report.
    """
    from ..agents.partner.tool_loop import ToolCallingLoop
    from ..llm.base import Message
    from ..llm.brain import brain_for

    brain = brain_for(context)
    history: list[Any] = _HISTORY.setdefault(session_id, [])
    turn_tools: list[str] = []

    def _think(transcript: str) -> str:
        turn_tools.clear()
        loop = ToolCallingLoop(
            brain, registry,
            max_iterations=3, total_timeout_s=60.0,
            max_tool_calls_per_turn=3)
        try:
            result = loop.run(
                transcript, actor="owner",
                history=list(history[-_MAX_HISTORY_TURNS * 2:]),
                loop_ctx={"chat_kind": "dm", "platform": "livekit",
                          "chat_key": f"livekit:{session_id}"})
        except Exception as exc:  # noqa: BLE001 — a dead loop is silence
            _log.warning("live call agent loop failed: %s", exc)
            return ""
        for name in result.tools_used:
            turn_tools.append(name)
            if name not in tools_used:
                tools_used.append(name)
        if not result.ok:
            _log.warning("live call agent loop degraded: %s",
                         result.error or result.degrade_note)
            return ""
        text = (result.answer or "").strip()
        history.append(Message.user(transcript))
        if text:
            history.append(Message.assistant(text))
        return text

    def _drain() -> list[str]:
        out = list(turn_tools)
        turn_tools.clear()
        return out

    _think.drain_tools_used = _drain  # type: ignore[attr-defined]
    return _think


def _build_session(registry: Any, room: str, *, profile: str = "",
                   max_wait_remote: float = 120.0,
                   agent: bool = False) -> LiveKitCallSession:
    """Assemble a live-call session from the registry's context."""
    context = getattr(registry, "context", None)
    data_dir = _voice_data_dir(context)
    tools_used: list[str] = []
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
        tools_used=tools_used,
    )
    # the think closure keys history by the real session id
    if agent:
        session.think = _build_agent_think(
            context, registry, session.session_id, tools_used)
    else:
        session.think = _build_think(context, session.session_id)
    return session


def _build_video_session(registry: Any, room: str, *,
                         source: str = "title", image: str = "",
                         design: Any = None, duration: float = 0.0,
                         text: str = "", fps: float = 15.0,
                         width: int = 640, height: int = 360,
                         loop: bool = True,
                         face_watch: bool = False) -> LiveKitVideoSession:
    """Assemble a live-video session from the registry's context.

    ``source`` picks the picture: "title" (a live card with the room
    and the clock), "image" (a still file, letterboxed), or "scene"
    (a motion-studio design dict, rendered frame by frame, looping
    until the call ends unless ``loop`` is False). ``face_watch``
    runs inbound frames through the facemesh rung (see the video
    session — mediapipe absent means the report says so).
    """
    context = getattr(registry, "context", None)
    data_dir = _voice_data_dir(context)
    if source == "title":
        frame_source = title_frame_source(room, width=width,
                                          height=height)
        source_name = "title"
    elif source == "image":
        if not (image or "").strip():
            raise LiveKitUnavailable(
                "video source 'image' needs an image path",
                missing="an image path")
        frame_source = image_frame_source(image, width=width,
                                          height=height)
        source_name = f"image:{Path(image).name}"
    elif source == "scene":
        if not isinstance(design, dict) or not design:
            raise LiveKitUnavailable(
                "video source 'scene' needs a design dict",
                missing="a motion-studio design")
        if duration <= 0:
            duration = 30.0
        frame_source = scene_frame_source(
            design, width=width, height=height, duration=duration,
            text=text, loop=loop)
        source_name = f"scene:{design.get('kind', 'scene')}"
    else:
        raise LiveKitUnavailable(
            f"unknown video source {source!r} — title, image, or scene",
            missing="a video source (title, image, or scene)")
    return LiveKitVideoSession(
        frame_source=frame_source,
        config=LiveKitConfig.from_env(),
        room=room,
        data_dir=data_dir,
        width=width, height=height, fps=fps,
        source_name=source_name,
        bus=None,
        face_watch=face_watch,
    )


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
            "(`nm voice live provision`). One live call at a time. "
            "With agent=true, each turn runs through her real tool "
            "loop, so she can DO things in the call (look things up, "
            "run tools) instead of only talking; the report names "
            "every tool she used. Participants can also type in the "
            "room (data channel, topic 'devon-chat'): she hears the "
            "text as a turn and answers in voice and in text."
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
            "agent": "bool (optional) — agent mode: run each turn "
                     "through the tool loop so Devon can use tools "
                     "mid-call (default false)",
        },
    )
    def _voice_live_start(room: str, max_turns: int = 0,
                         max_wait_remote: float = 120.0,
                         profile: str = "",
                         agent: bool = False) -> dict[str, Any]:
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
                                     max_wait_remote=max_wait_remote,
                                     agent=agent)
            start_live_call(session, max_turns=max_turns)
        except LiveKitUnavailable as exc:
            return {"ok": False, "error": str(exc),
                    "missing": exc.missing}
        except Exception as exc:  # noqa: BLE001 — start never raises
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "session_id": session.session_id,
                "room": room, "state": session.state,
                "agent": agent}

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

    @registry.register(
        "video_live_start",
        description=(
            "Start a live video session in a LiveKit room: Devon joins "
            "the room as a second participant and publishes a real "
            "video track, while watching the room's video and saving "
            "one snapshot PNG per publisher per second. Returns "
            "immediately with the session id; the session runs in the "
            "background. Sources: 'title' (a live card with the room "
            "and the clock), 'image' (a still file), 'scene' (a "
            "motion-studio design dict, rendered frame by frame). "
            "With face_watch=true, inbound room frames also run "
            "through the facemesh rung (who is on camera, smiling, "
            "blinking, head pose) at the snapshot cadence; when "
            "mediapipe or the model is absent the report says so "
            "and the video still runs. "
            "Needs a LiveKit server (LIVEKIT_URL plus LIVEKIT_TOKEN or "
            "LIVEKIT_API_KEY + LIVEKIT_API_SECRET) and the livekit SDK "
            "(`nm voice live provision`). One live video session at a "
            "time; it can run beside the voice call."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "room": "str — the LiveKit room name to join",
            "source": "str (optional) — 'title' (default), 'image', "
                      "or 'scene'",
            "image": "str (optional) — still-image path, for "
                     "source='image'",
            "design": "dict (optional) — motion-studio design, for "
                      "source='scene'",
            "duration": "float (optional) — scene length in seconds "
                        "(default 30; loops until stopped)",
            "text": "str (optional) — caption text for a scene source",
            "fps": "float (optional) — publish rate (default 15)",
            "width": "int (optional) — publish width (default 640)",
            "height": "int (optional) — publish height (default 360)",
            "max_duration": "float (optional) — stop after this many "
                           "seconds (0 = until stopped)",
            "loop": "bool (optional) — loop a scene source (default "
                    "true; false ends the session at the scene end)",
            "face_watch": "bool (optional) — watch the room's video "
                          "with the facemesh rung (default false)",
        },
    )
    def _video_live_start(
            room: str, source: str = "title", image: str = "",
            design: Any = None, duration: float = 0.0,
            text: str = "", fps: float = 15.0, width: int = 640,
            height: int = 360, max_duration: float = 0.0,
            loop: bool = True, face_watch: bool = False
            ) -> dict[str, Any]:
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
            session = _build_video_session(
                registry, room, source=source, image=image,
                design=design, duration=duration, text=text, fps=fps,
                width=width, height=height, loop=loop,
                face_watch=face_watch)
            start_video_session(session, max_duration_s=max_duration)
        except LiveKitUnavailable as exc:
            return {"ok": False, "error": str(exc),
                    "missing": exc.missing}
        except Exception as exc:  # noqa: BLE001 — start never raises
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "session_id": session.session_id,
                "room": room, "state": session.state,
                "source": session.source_name,
                "face_watch": face_watch}

    @registry.register(
        "video_live_status",
        description=(
            "Live-video readiness and state: whether the livekit SDK "
            "is installed, whether a LiveKit server is configured, and "
            "the running video session's state, published frames, "
            "received frames, and snapshots if any. No connections "
            "are made by this call."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "session_id": "str (optional) — a session id; empty means "
                          "the running video session",
        },
    )
    def _video_live_status(session_id: str = "") -> dict[str, Any]:
        return {"ok": True, "status": video_session_status(session_id)}

    @registry.register(
        "video_live_end",
        description=(
            "End the running live video session and return its report "
            "(frames published, frames received per publisher, "
            "snapshots, why it ended). Refuses honestly when no video "
            "session is running."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "session_id": "str (optional) — a session id; empty means "
                          "the running video session",
        },
    )
    def _video_live_end(session_id: str = "") -> dict[str, Any]:
        report = end_video_session(session_id)
        if report is None:
            return {"ok": False,
                    "error": "no live video session is running"}
        return {"ok": True, "report": report.to_dict()}

    @registry.register(
        "video_live_snapshot",
        description=(
            "The latest snapshot of a room's video: the newest PNG "
            "Devon saved from a publisher's video track, with how "
            "many frames she has received from them and how old the "
            "snapshot is. Refuses honestly when no video session is "
            "running or nobody has sent a frame yet."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "session_id": "str (optional) — a session id; empty means "
                          "the running video session",
            "identity": "str (optional) — the publisher to look at; "
                        "empty means the only publisher",
        },
    )
    def _video_live_snapshot(session_id: str = "",
                             identity: str = "") -> dict[str, Any]:
        session = get_video_session(session_id)
        if session is None:
            return {"ok": False,
                    "error": "no live video session is running"}
        snap = session.snapshot(identity)
        if snap is None:
            who = f" from {identity!r}" if identity else ""
            return {"ok": False,
                    "error": "no video frame received yet" + who}
        return {"ok": True, "snapshot": snap}
