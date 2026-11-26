"""Tests for the LiveKit Agents lane (305 map #7): agent-mode think
(in-call function tools), data-channel text turns, room awareness,
and inbound-video face watch — all on the existing thin-SDK
participant (``livekit-agents`` NOT adopted; see SURVEY.md).

The fakes move REAL frames/text through the REAL session state
machines. No room, no network, no model — except the live-gated
join test in test_voice_livekit_live.py, which runs a real agent
session against a real server.
"""

from __future__ import annotations

import importlib.util
import math
import queue
import struct
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.voice.livekit_call import (  # noqa: E402
    AGENT_TEXT_TOPIC,
    FRAME_BYTES_48K,
    LiveKitCallSession,
    LiveKitConfig,
    LiveKitUnavailable,
    LiveKitVideoSession,
    RtcTransport,
)
from nomorals.voice.session import write_wav_bytes  # noqa: E402


def _tone48(seconds: float, *, freq: float = 440.0,
            amplitude: int = 12000) -> bytes:
    n = int(48000 * seconds)
    out = bytearray()
    for i in range(n):
        value = int(amplitude * math.sin(2.0 * math.pi * freq * i / 48000))
        out += struct.pack("<h", value)
    return bytes(out)


def _silence48(seconds: float) -> bytes:
    return b"\x00\x00" * int(48000 * seconds)


def _tone_rate(seconds: float, rate: int, *, freq: float = 440.0,
               amplitude: int = 12000) -> bytes:
    n = int(rate * seconds)
    out = bytearray()
    for i in range(n):
        value = int(amplitude * math.sin(2.0 * math.pi * freq * i / rate))
        out += struct.pack("<h", value)
    return bytes(out)


# ── the fake room (audio + text) ───────────────────────────────────────────


class AgentFakeTransport:
    """The fake room from test_voice_livekit.py, plus the data channel:
    inbound text is queued with :meth:`feed_text`, outbound text is
    recorded in :attr:`published_text`."""

    CHUNK = 1920  # 20 ms at 48 kHz mono int16

    def __init__(self, *, publishers=("owner",),
                 fail_text: bool = False):
        self._inbound: queue.Queue[bytes] = queue.Queue()
        self._inbound_text: queue.Queue = queue.Queue()
        self._publishers = set(publishers)
        self.published: list[bytes] = []
        self.published_text: list[str] = []
        self.fail_text = fail_text
        self.cleared = 0
        self.connected = False
        self.disconnect_called = False
        self.connect_args: tuple[str, str] | None = None

    def feed(self, pcm48: bytes) -> None:
        for i in range(0, len(pcm48), self.CHUNK):
            self._inbound.put(pcm48[i:i + self.CHUNK])

    def feed_silence(self, seconds: float) -> None:
        self.feed(_silence48(seconds))

    def feed_tone(self, seconds: float) -> None:
        self.feed(_tone48(seconds))

    def feed_text(self, text: str, identity: str = "owner",
                  topic: str = AGENT_TEXT_TOPIC) -> None:
        self._inbound_text.put((text, identity, topic))

    def drop_publishers(self) -> None:
        self._publishers.clear()

    def add_publisher(self, identity: str) -> None:
        self._publishers.add(identity)

    def connect(self, url: str, token: str, *, timeout: float = 30.0
                ) -> None:
        self.connect_args = (url, token)
        self.connected = True

    def poll_inbound(self, timeout: float = 0.05) -> bytes | None:
        try:
            return self._inbound.get(timeout=min(timeout, 0.02))
        except queue.Empty:
            return None

    def publish_frame(self, pcm48: bytes, *, timeout: float = 5.0) -> None:
        assert len(pcm48) == FRAME_BYTES_48K, len(pcm48)
        self.published.append(bytes(pcm48))
        time.sleep(0.01)

    def clear_outgoing(self) -> None:
        self.cleared += 1

    def publish_text(self, text: str, *, topic: str = AGENT_TEXT_TOPIC,
                     timeout: float = 5.0) -> None:
        if self.fail_text:
            raise RuntimeError("data channel down")
        self.published_text.append(text)

    def poll_text(self, timeout: float = 0.05):
        try:
            return self._inbound_text.get(timeout=min(timeout, 0.02))
        except queue.Empty:
            return None

    def remote_audio_publishers(self) -> list[str]:
        return sorted(self._publishers)

    def is_connected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.disconnect_called = True
        self.connected = False


class FakeVideoTransport:
    """The fake video room from test_voice_livekit_video.py."""

    def __init__(self, *, publishers=("owner",)):
        self._inbound: queue.Queue = queue.Queue()
        self._publishers = set(publishers)
        self.published: list[tuple[bytes, int, int]] = []
        self.connected = False
        self.disconnect_called = False

    def feed_frame(self, rgb: bytes, width: int, height: int,
                   identity: str = "owner") -> None:
        self._inbound.put((bytes(rgb), width, height, identity))

    def connect(self, url: str, token: str, *, timeout: float = 30.0
                ) -> None:
        self.connected = True

    def poll_video_frame(self, timeout: float = 0.05):
        try:
            return self._inbound.get(timeout=min(timeout, 0.02))
        except queue.Empty:
            return None

    def publish_video_frame(self, rgb: bytes, width: int, height: int,
                            *, timeout: float = 5.0) -> None:
        if len(rgb) != width * height * 3:
            raise ValueError("bad frame size")
        self.published.append((bytes(rgb), width, height))
        time.sleep(0.02)

    def remote_video_publishers(self) -> list[str]:
        return sorted(self._publishers)

    def is_connected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.disconnect_called = True
        self.connected = False


class FakeBus:
    """Captures journal events for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, event: object) -> None:
        self.events.append((event.topic, dict(event.data)))  # type: ignore[attr-defined]

    def topics(self) -> list[str]:
        return [t for t, _ in self.events]


def _fake_stt_factory(tmpdir: str, transcripts: list[str]):
    state = {"i": 0}

    def _stt(wav_path: str) -> str:
        idx = state["i"]
        state["i"] += 1
        return transcripts[idx] if idx < len(transcripts) else ""

    return _stt


def _fake_tts_factory(tmpdir: str, *, seconds: float = 1.0,
                      rate: int = 16000):
    state = {"n": 0}

    def _tts(text: str, profile: str) -> dict:
        path = str(Path(tmpdir) / f"tts-{state['n']}.wav")
        state["n"] += 1
        write_wav_bytes(path, _tone_rate(seconds, rate),
                        sample_rate=rate)
        return {"path": path}

    return _tts


# ── the fake brain + stub registry for the tool loop ───────────────────────


class _FakeLLMResponse:
    def __init__(self, *, ok: bool, text: str = "",
                 error: str = "", model: str = "fake"):
        self.ok = ok
        self.text = text
        self.error = error
        self.model = model


class _ScriptedBrain:
    """Brain.chat answers from a script, in order."""

    def __init__(self, replies: list[_FakeLLMResponse]):
        self._replies = list(replies)
        self.calls: list[list[Any]] = []

    def chat(self, messages: Any, *args: Any, **kwargs: Any
             ) -> _FakeLLMResponse:
        self.calls.append(list(messages))
        if not self._replies:
            return _FakeLLMResponse(ok=True, text="Done.")
        return self._replies.pop(0)


class _FakeOutcome:
    def __init__(self, ok: bool, value: Any = None, error: str = ""):
        self.ok = ok
        self.value = value
        self.error = error


class _StubToolRegistry:
    """The slice of ToolRegistry the ToolCallingLoop actually uses."""

    def __init__(self) -> None:
        self.context = None
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._tools: dict[str, Any] = {
            "echo_text": lambda text: {"echo": text},
        }

    def get(self, name: str) -> Any | None:
        if name in self._tools:
            return SimpleNamespace(name=name)
        return None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, name: str, *args: Any, **kwargs: Any) -> _FakeOutcome:
        fn = self._tools.get(name)
        if fn is None:
            return _FakeOutcome(False, error=f"unknown tool {name}")
        kwargs.pop("actor", None)
        kwargs.pop("capabilities", None)
        kwargs.pop("idempotency_key", None)
        self.calls.append((name, dict(kwargs)))
        try:
            return _FakeOutcome(True, value=fn(**kwargs))
        except Exception as exc:  # noqa: BLE001 — a stub tool failure is data
            return _FakeOutcome(False, error=str(exc))

    def ranked_listing(self, query: str, **kwargs: Any) -> str:
        return ""

    def grouped_listing(self, **kwargs: Any) -> str:
        return "- echo_text(text: str): echo the text back"

    def prompt_listing(self, **kwargs: Any) -> str:
        return "- echo_text(text: str): echo the text back"

    def schemas(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"name": "echo_text",
                 "description": "echo the text back",
                 "parameters": {"text": "str"}}]


_TOOL_CALL_REPLY = (
    "I will echo that.\n\n```tool\n"
    '{"tool": "echo_text", "args": {"text": "hello"}}\n```'
)


def _agent_think(brain: _ScriptedBrain, registry: _StubToolRegistry,
                 session_id: str, tools_used: list[str]) -> Any:
    """The production agent think, with a scripted brain."""
    from nomorals.tools import voice_live

    with mock.patch("nomorals.llm.brain.brain_for",
                    return_value=brain):
        return voice_live._build_agent_think(
            SimpleNamespace(), registry, session_id, tools_used)


def _make_agent_session(tmpdir: str, transport: AgentFakeTransport, *,
                        think: Any, transcripts: list[str],
                        bus: FakeBus | None = None,
                        tools_used: list[str] | None = None,
                        max_wait_remote_s: float = 5.0) -> LiveKitCallSession:
    return LiveKitCallSession(
        think=think,
        stt=_fake_stt_factory(tmpdir, transcripts),
        tts=_fake_tts_factory(tmpdir),
        transport=transport,
        config=LiveKitConfig(url="wss://fake.invalid",
                             token="fake-token"),
        room="test-room",
        data_dir=tmpdir,
        silence_ms=200,
        max_wait_remote_s=max_wait_remote_s,
        remote_silence_end_s=0.5,
        bus=bus if bus is not None else False,
        tools_used=tools_used,
    )


def _solid_rgb(width: int, height: int,
               rgb: tuple[int, int, int]) -> bytes:
    return bytes(rgb) * (width * height)


# ── agent mode: in-call tools ──────────────────────────────────────────────


class TestAgentThink(unittest.TestCase):
    def test_agent_turn_runs_a_tool_and_speaks_the_answer(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = AgentFakeTransport()
            bus = FakeBus()
            tools_used: list[str] = []
            brain = _ScriptedBrain([
                _FakeLLMResponse(ok=True, text=_TOOL_CALL_REPLY),
                _FakeLLMResponse(ok=True, text="The echo is hello."),
            ])
            registry = _StubToolRegistry()
            think = _agent_think(brain, registry, "agent-t1", tools_used)
            transport.feed_silence(0.6)
            transport.feed_tone(0.8)
            transport.feed_silence(1.5)
            session = _make_agent_session(
                tmpdir, transport, think=think,
                transcripts=["hello there"], bus=bus,
                tools_used=tools_used)
            report = session.start(max_turns=1)
            self.assertEqual(report.end_reason, "completed")
            self.assertEqual(report.turns, 1)
            self.assertEqual(report.tools_used, ["echo_text"])
            self.assertEqual(tools_used, ["echo_text"])
            turn = report.spoken[0]
            self.assertEqual(turn["source"], "voice")
            self.assertEqual(turn["tools"], ["echo_text"])
            self.assertEqual(turn["said"], ["The echo is hello."])
            self.assertTrue(transport.published,
                            "the reply was never published")
            self.assertEqual(registry.calls, [("echo_text",
                                              {"text": "hello"})])
            self.assertIn("voice.livekit.agent_tools", bus.topics())
            # the drain is spent: a second drain finds nothing
            self.assertEqual(think.drain_tools_used(), [])

    def test_agent_loop_failure_is_silence_not_a_crash(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = AgentFakeTransport()
            tools_used: list[str] = []
            brain = _ScriptedBrain([
                _FakeLLMResponse(ok=False, text="",
                                 error="model down"),
            ])
            think = _agent_think(brain, _StubToolRegistry(),
                                 "agent-t2", tools_used)
            transport.feed_silence(0.6)
            transport.feed_tone(0.8)
            transport.feed_silence(1.5)
            session = _make_agent_session(
                tmpdir, transport, think=think,
                transcripts=["hello there"], tools_used=tools_used)
            report = session.start(max_turns=1)
            self.assertEqual(report.end_reason, "completed")
            self.assertEqual(report.turns, 1)
            self.assertEqual(report.spoken[0]["said"], [])
            self.assertEqual(report.tools_used, [])

    def test_plain_think_uses_no_tools(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = AgentFakeTransport()
            tools_used: list[str] = []
            transport.feed_silence(0.6)
            transport.feed_tone(0.8)
            transport.feed_silence(1.5)
            session = _make_agent_session(
                tmpdir, transport, think=lambda text: "Hello there.",
                transcripts=["hello there"], tools_used=tools_used)
            report = session.start(max_turns=1)
            self.assertEqual(report.tools_used, [])
            self.assertEqual(report.spoken[0]["tools"], [])
            self.assertEqual(report.spoken[0]["source"], "voice")

    def test_agent_think_drains_per_turn(self):
        tools_used: list[str] = []
        brain = _ScriptedBrain([
            _FakeLLMResponse(ok=True, text=_TOOL_CALL_REPLY),
            _FakeLLMResponse(ok=True, text="The echo is hello."),
        ])
        think = _agent_think(brain, _StubToolRegistry(),
                             "agent-t3", tools_used)
        self.assertEqual(think("say hello"), "The echo is hello.")
        self.assertEqual(think.drain_tools_used(), ["echo_text"])
        self.assertEqual(think.drain_tools_used(), [])
        self.assertEqual(tools_used, ["echo_text"])


# ── data channel: text turns ───────────────────────────────────────────────


class TestDataChannelTurns(unittest.TestCase):
    def test_text_turn_answers_in_voice_and_data(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = AgentFakeTransport()
            bus = FakeBus()
            transport.feed_text("what time is it")
            session = _make_agent_session(
                tmpdir, transport, think=lambda text: "It is noon.",
                transcripts=[], bus=bus)
            report = session.start(max_turns=1)
            self.assertEqual(report.end_reason, "completed")
            self.assertEqual(report.turns, 1)
            self.assertEqual(report.text_turns, 1)
            turn = report.spoken[0]
            self.assertEqual(turn["source"], "text")
            self.assertEqual(turn["heard"], "what time is it")
            self.assertEqual(turn["said"], ["It is noon."])
            self.assertTrue(turn["text_reply_sent"])
            self.assertEqual(transport.published_text, ["It is noon."])
            self.assertTrue(transport.published,
                            "the text turn was never spoken")
            self.assertIn("voice.livekit.turn", bus.topics())

    def test_text_reply_publish_failure_does_not_kill_the_turn(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = AgentFakeTransport(fail_text=True)
            transport.feed_text("hello")
            session = _make_agent_session(
                tmpdir, transport, think=lambda text: "Hi.",
                transcripts=[])
            report = session.start(max_turns=1)
            self.assertEqual(report.end_reason, "completed")
            self.assertEqual(report.turns, 1)
            self.assertFalse(report.spoken[0]["text_reply_sent"])
            self.assertEqual(transport.published_text, [])

    def test_publish_text_without_connection_refuses(self):
        transport = RtcTransport()
        try:
            self.assertIsNone(transport.poll_text(timeout=0.0))
            with self.assertRaises(LiveKitUnavailable) as ctx:
                transport.publish_text("hello")
            self.assertIn("LiveKit room", str(ctx.exception))
            self.assertIn("LiveKit room", ctx.exception.missing)
        finally:
            transport.disconnect()

    def test_report_format_names_text_turns_and_tools(self):
        from nomorals.voice.livekit_call import LiveCallReport

        report = LiveCallReport(
            session_id="s1", room="r", turns=1, text_turns=1,
            tools_used=["echo_text"], participants_seen=["owner"],
            spoken=[{"turn": 1, "heard": "hi", "said": ["Hello."],
                     "barged": False, "source": "text",
                     "tools": ["echo_text"], "text_reply_sent": True}])
        data = report.to_dict()
        self.assertEqual(data["text_turns"], 1)
        self.assertEqual(data["tools_used"], ["echo_text"])
        self.assertEqual(data["participants_seen"], ["owner"])
        text = report.format()
        self.assertIn("text turns 1", text)
        self.assertIn("echo_text", text)
        self.assertIn("owner", text)


# ── room awareness ─────────────────────────────────────────────────────────


class TestRoomAwareness(unittest.TestCase):
    def test_participant_join_and_leave_journalled(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = AgentFakeTransport(publishers=())
            bus = FakeBus()
            session = _make_agent_session(
                tmpdir, transport, think=lambda text: "",
                transcripts=[], bus=bus)
            session._note_participants()
            self.assertEqual(bus.topics(), [])
            transport.add_publisher("owner")
            session._note_participants()
            self.assertIn("voice.livekit.participant_joined",
                          bus.topics())
            self.assertEqual(
                bus.events[-1][1]["identity"], "owner")
            transport.drop_publishers()
            session._note_participants()
            self.assertIn("voice.livekit.participant_left",
                          bus.topics())
            self.assertEqual(session._participants_seen, set())


# ── face watch ─────────────────────────────────────────────────────────────


def _stub_face(rgb: bytes, width: int, height: int) -> dict[str, Any]:
    return {
        "face": True,
        "landmarks": 468,
        "smile": 0.5,
        "jaw_open": 0.1,
        "blink_left": 0.0,
        "blink_right": 0.0,
        "head_deg": [0.0, 2.0, -1.0],
        "center": [0.5, 0.5],
        "size": 0.4,
    }


def _make_video_session(tmpdir: str, transport: FakeVideoTransport, *,
                        face_watch: bool, face_detector: Any = None,
                        bus: FakeBus | None = None) -> LiveKitVideoSession:
    width, height = 64, 36
    frame = (_solid_rgb(width, height, (16, 32, 48)), width, height)
    return LiveKitVideoSession(
        frame_source=lambda t: frame,
        transport=transport,
        config=LiveKitConfig(url="wss://fake.invalid",
                             token="fake-token"),
        room="test-room",
        data_dir=tmpdir,
        width=width, height=height, fps=15.0,
        source_name="solid",
        bus=bus if bus is not None else False,
        face_watch=face_watch,
        face_detector=face_detector,
    )


class TestFaceWatch(unittest.TestCase):
    def test_face_observations_recorded_with_stub_detector(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeVideoTransport()
            bus = FakeBus()
            transport.feed_frame(
                _solid_rgb(64, 36, (200, 180, 160)), 64, 36, "owner")
            session = _make_video_session(
                tmpdir, transport, face_watch=True,
                face_detector=_stub_face, bus=bus)
            report = session.start(max_duration_s=0.3)
            self.assertEqual(report.end_reason, "completed")
            self.assertEqual(report.face_watch, "on")
            self.assertTrue(report.frames_published >= 1)
            face = report.faces["owner"]
            self.assertEqual(face["frames_with_face"], 1)
            self.assertEqual(face["frames_seen"], 1)
            self.assertEqual(face["last"]["smile"], 0.5)
            self.assertEqual(face["last"]["head_deg"], [0.0, 2.0, -1.0])
            self.assertIn("voice.livekit.face_seen", bus.topics())
            self.assertTrue(report.frames_received.get("owner", 0) >= 1)

    @unittest.skipIf(
        importlib.util.find_spec("mediapipe") is not None,
        "mediapipe is installed here; the unavailable path is untestable")
    def test_face_watch_unavailable_without_mediapipe(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeVideoTransport()
            transport.feed_frame(
                _solid_rgb(64, 36, (200, 180, 160)), 64, 36, "owner")
            session = _make_video_session(
                tmpdir, transport, face_watch=True)
            report = session.start(max_duration_s=0.3)
            self.assertEqual(report.end_reason, "completed")
            self.assertTrue(
                report.face_watch.startswith("unavailable"),
                report.face_watch)
            self.assertIn("mediapipe", report.face_watch)
            self.assertTrue(report.frames_published >= 1,
                            "video must still run without face watch")
            # the frame was seen, but no observation exists: no face
            # was ever claimed, and nothing was invented
            self.assertEqual(report.faces["owner"]["frames_seen"], 1)
            self.assertEqual(
                report.faces["owner"]["frames_with_face"], 0)
            self.assertIsNone(report.faces["owner"]["last"])

    @unittest.skipIf(
        importlib.util.find_spec("mediapipe") is not None,
        "mediapipe is installed here; the unavailable path is untestable")
    def test_frame_face_detector_refuses_without_mediapipe(self):
        from nomorals.media.facemesh.capture import FrameFaceDetector
        from nomorals.media.facemesh.errors import FaceMeshError

        with self.assertRaises(FaceMeshError) as ctx:
            FrameFaceDetector(install=False, download=False)
        self.assertIn("mediapipe", str(ctx.exception))

    def test_face_watch_off_records_nothing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeVideoTransport()
            transport.feed_frame(
                _solid_rgb(64, 36, (200, 180, 160)), 64, 36, "owner")
            session = _make_video_session(
                tmpdir, transport, face_watch=False)
            report = session.start(max_duration_s=0.3)
            self.assertEqual(report.face_watch, "off")
            self.assertEqual(report.faces, {})


# ── tool wiring: no new tools, params on the existing ones ─────────────────


class TestAgentToolWiring(unittest.TestCase):
    def _registry(self) -> Any:
        from nomorals.tools import voice_live

        class _Reg:
            def __init__(self) -> None:
                self.context = SimpleNamespace()
                self.tools: dict[str, Any] = {}

            def register(self, name: str, **kwargs: Any):
                def _deco(fn: Any) -> Any:
                    self.tools[name] = (fn, kwargs)
                    return fn

                return _deco

        registry = _Reg()
        voice_live.register(registry)
        return registry

    def test_voice_live_start_takes_agent(self):
        registry = self._registry()
        fn, kwargs = registry.tools["voice_live_start"]
        self.assertIn("agent", kwargs["parameters"])
        import inspect

        self.assertIn("agent", inspect.signature(fn).parameters)

    def test_video_live_start_takes_face_watch(self):
        registry = self._registry()
        fn, kwargs = registry.tools["video_live_start"]
        self.assertIn("face_watch", kwargs["parameters"])
        import inspect

        self.assertIn("face_watch", inspect.signature(fn).parameters)

    def test_tool_count_still_808(self):
        """No new brain tool was added — agent mode and face watch
        are modes of the existing tools, so the count stays at 787
        after the swarm ship added build_swarm (was 785 at the
        lane's base; rebased onto c6b412fb)."""
        from types import SimpleNamespace as _NS

        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(_NS())
        registry.register_builtins()
        self.assertEqual(len(registry._tools), 808)  # trendclip adds 5 tools
        schemas = {s["name"]: s for s in registry.schemas()}
        self.assertIn("agent", str(schemas["voice_live_start"]))
        self.assertIn("face_watch", str(schemas["video_live_start"]))


if __name__ == "__main__":
    unittest.main()
