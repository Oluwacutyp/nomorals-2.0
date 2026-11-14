"""Tests for the LiveKit real-time voice rung (voice/livekit_call.py,
tools/voice_live.py, `nm voice live`).

The fake transport below moves REAL 48 kHz PCM frames through the REAL
session state machine — listen (VAD) → think → speak (publish) with
barge-in — against no room, no network, and no model. Nothing in the
fake claims success without doing the work: published frames are
counted and sized, inbound audio is really downsampled and really
VAD-gated, and the report records what was actually played.
"""

from __future__ import annotations

import json
import math
import os
import queue
import struct
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.voice.livekit_call import (  # noqa: E402
    FRAME_BYTES_48K,
    LiveCallReport,
    LiveKitCallSession,
    LiveKitConfig,
    LiveKitUnavailable,
    RtcTransport,
    end_live_call,
    get_live_call,
    live_call_status,
    livekit_status,
    provision_livekit,
    resample_pcm,
    sdk_available,
    start_live_call,
)
from nomorals.voice.session import write_wav_bytes  # noqa: E402

# ── PCM helpers ────────────────────────────────────────────────────────────


def _tone48(seconds: float, *, freq: float = 440.0,
            amplitude: int = 12000) -> bytes:
    """Loud 48 kHz mono tone — unmistakable speech to the energy VAD."""
    n = int(48000 * seconds)
    out = bytearray()
    for i in range(n):
        value = int(amplitude * math.sin(2.0 * math.pi * freq * i / 48000))
        out += struct.pack("<h", value)
    return bytes(out)


def _silence48(seconds: float) -> bytes:
    return b"\x00\x00" * int(48000 * seconds)


def _tone16(seconds: float, *, freq: float = 440.0,
            amplitude: int = 12000) -> bytes:
    return _tone_rate(seconds, 16000, freq=freq, amplitude=amplitude)


def _tone_rate(seconds: float, rate: int, *, freq: float = 440.0,
               amplitude: int = 12000) -> bytes:
    n = int(rate * seconds)
    out = bytearray()
    for i in range(n):
        value = int(amplitude * math.sin(2.0 * math.pi * freq * i / rate))
        out += struct.pack("<h", value)
    return bytes(out)


# ── the fake room ──────────────────────────────────────────────────────────


class FakeTransport:
    """A scripted LiveKit room: feeds 48 kHz PCM, records publishes.

    Inbound audio is fed with :meth:`feed`; :meth:`poll_inbound`
    serves it in 20 ms chunks (1920 bytes). Silence is fed the same
    way — the session cannot tell the fake from a quiet room except
    that the fake answers faster than a real network (polls sleep at
    most 0.02 s when empty).
    """

    CHUNK = 1920  # 20 ms at 48 kHz mono int16

    def __init__(self, *, publishers=("owner",)):
        self._inbound: queue.Queue[bytes] = queue.Queue()
        self._publishers = set(publishers)
        self.published: list[bytes] = []
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

    def drop_publishers(self) -> None:
        self._publishers.clear()

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
        # real-time backpressure, like AudioSource.capture_frame: a 10 ms
        # frame takes ~10 ms to leave. Without it the session would
        # publish a whole reply instantly and a barge-in could never
        # land mid-reply.
        time.sleep(0.01)

    def clear_outgoing(self) -> None:
        self.cleared += 1

    def remote_audio_publishers(self) -> list[str]:
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
    """STT fake: returns the next canned transcript per utterance."""
    state = {"i": 0}

    def _stt(wav_path: str) -> str:
        assert os.path.exists(wav_path), wav_path
        idx = state["i"]
        state["i"] += 1
        return transcripts[idx] if idx < len(transcripts) else ""

    return _stt


def _fake_tts_factory(tmpdir: str, *, seconds: float = 1.0,
                      rate: int = 16000):
    """TTS fake: writes a real wav of tone, returns {"path": ...}."""
    state = {"n": 0}

    def _tts(text: str, profile: str) -> dict:
        path = os.path.join(tmpdir, f"tts-{state['n']}.wav")
        state["n"] += 1
        pcm = _tone_rate(seconds, rate)
        write_wav_bytes(path, pcm, sample_rate=rate)
        return {"path": path}

    return _tts


def _make_session(tmpdir: str, transport: FakeTransport, *,
                  transcripts: list[str], reply: str = "Hello there.",
                  bus: FakeBus | None = None,
                  tts_seconds: float = 1.0, tts_rate: int = 16000,
                  silence_ms: int = 200, max_wait_remote_s: float = 5.0,
                  remote_silence_end_s: float = 0.5) -> LiveKitCallSession:
    return LiveKitCallSession(
        think=lambda text: reply,
        stt=_fake_stt_factory(tmpdir, transcripts),
        tts=_fake_tts_factory(tmpdir, seconds=tts_seconds,
                              rate=tts_rate),
        transport=transport,
        config=LiveKitConfig(url="wss://fake.invalid",
                             token="fake-token"),
        room="test-room",
        data_dir=tmpdir,
        silence_ms=silence_ms,
        max_wait_remote_s=max_wait_remote_s,
        remote_silence_end_s=remote_silence_end_s,
        bus=bus if bus is not None else False,
    )


# ── config ─────────────────────────────────────────────────────────────────


class TestLiveKitConfig(unittest.TestCase):
    def test_missing_names_every_absent_piece(self):
        cfg = LiveKitConfig.from_env(env={})
        missing = cfg.missing()
        self.assertIn("LIVEKIT_URL", missing[0])
        self.assertTrue(any("LIVEKIT_TOKEN" in m for m in missing))

    def test_token_only_is_ready(self):
        cfg = LiveKitConfig.from_env(env={
            "LIVEKIT_URL": "wss://room.example.test",
            "LIVEKIT_TOKEN": "pre-signed",
        })
        self.assertEqual(cfg.missing(), [])

    def test_key_plus_secret_is_ready(self):
        cfg = LiveKitConfig.from_env(env={
            "LIVEKIT_URL": "wss://room.example.test",
            "LIVEKIT_API_KEY": "key",
            "LIVEKIT_API_SECRET": "secret",
        })
        self.assertEqual(cfg.missing(), [])

    def test_presigned_token_used_verbatim_without_sdk(self):
        cfg = LiveKitConfig.from_env(env={
            "LIVEKIT_URL": "wss://room.example.test",
            "LIVEKIT_TOKEN": "verbatim-token",
        })
        self.assertEqual(cfg.token_for("room"), "verbatim-token")

    def test_token_for_without_anything_raises_naming_the_gap(self):
        cfg = LiveKitConfig.from_env(env={})
        with self.assertRaises(LiveKitUnavailable) as ctx:
            cfg.token_for("room")
        self.assertIn("LiveKit", str(ctx.exception))
        self.assertTrue(ctx.exception.missing)
        self.assertIn("LIVEKIT_URL", ctx.exception.missing)


class TestSdkPresence(unittest.TestCase):
    def test_sdk_available_returns_bool(self):
        self.assertIsInstance(sdk_available(), bool)

    def test_provision_refuses_under_pytest(self):
        result = provision_livekit()
        self.assertFalse(result["ok"])
        self.assertIn("pytest", result["error"])


# ── resampling ─────────────────────────────────────────────────────────────


class TestResamplePcm(unittest.TestCase):
    def test_identity(self):
        pcm = struct.pack("<4h", 1, -2, 3, -4)
        self.assertEqual(resample_pcm(pcm, 16000, 16000), pcm)

    def test_down_48k_to_16k_box_average(self):
        # groups of 3 averaged: (3000+3000+3000)/3, (0+6000+3000)/3 ...
        pcm = struct.pack("<6h", 3000, 3000, 3000, 0, 6000, 3000)
        self.assertEqual(resample_pcm(pcm, 48000, 16000),
                         struct.pack("<2h", 3000, 3000))

    def test_up_16k_to_48k_repeats(self):
        pcm = struct.pack("<2h", 1000, -1000)
        self.assertEqual(resample_pcm(pcm, 16000, 48000),
                         struct.pack("<6h", 1000, 1000, 1000,
                                     -1000, -1000, -1000))

    def test_odd_byte_dropped(self):
        pcm = struct.pack("<3h", 7, 8, 9) + b"\x01"
        self.assertEqual(resample_pcm(pcm, 48000, 16000),
                         struct.pack("<1h", 8))

    def test_empty(self):
        self.assertEqual(resample_pcm(b"", 48000, 16000), b"")

    def test_non_integer_ratio_interpolates(self):
        pcm = struct.pack("<4h", 0, 1000, 2000, 3000)
        out = resample_pcm(pcm, 16000, 22050)
        self.assertEqual(len(out) % 2, 0)
        self.assertGreater(len(out), len(pcm))

    def test_bad_rate_raises(self):
        with self.assertRaises(ValueError):
            resample_pcm(b"\x00\x00", 0, 16000)


# ── the session, end to end against the fake room ─────────────────────────


class TestLiveCallSession(unittest.TestCase):
    def test_one_turn_then_goodbye(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeTransport()
            bus = FakeBus()
            # calibration room tone, then TWO utterances: the greeting
            # and, after a pause, the goodbye that ends the call.
            transport.feed_silence(0.6)
            transport.feed_tone(0.8)
            transport.feed_silence(1.0)
            transport.feed_tone(0.8)
            transport.feed_silence(1.5)
            session = _make_session(
                tmpdir, transport,
                transcripts=["hello there", "goodbye"], bus=bus)
            report = session.start(max_turns=0)
            self.assertEqual(report.end_reason, "user-exit")
            self.assertEqual(report.turns, 1)
            self.assertEqual(report.barge_ins, 0)
            self.assertEqual(report.room, "test-room")
            self.assertTrue(transport.published,
                            "the reply was never published")
            for frame in transport.published:
                self.assertEqual(len(frame), FRAME_BYTES_48K)
            self.assertTrue(transport.disconnect_called)
            self.assertEqual(report.spoken[0]["heard"], "hello there")
            self.assertEqual(report.spoken[0]["said"], ["Hello there."])
            self.assertFalse(report.spoken[0]["barged"])
            self.assertIn("voice.livekit.session_started", bus.topics())
            self.assertIn("voice.livekit.turn", bus.topics())
            self.assertIn("voice.livekit.session_ended", bus.topics())
            self.assertTrue(report.latencies_ms,
                            "no ear-to-ear latency recorded")
            # the last-call record lands for `nm voice live status`
            last = json.loads(
                Path(tmpdir, "live-last.json").read_text())
            self.assertEqual(last["turns"], 1)
            self.assertEqual(last["end_reason"], "user-exit")

    def test_max_turns_completes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeTransport()
            transport.feed_silence(0.6)
            transport.feed_tone(0.8)
            transport.feed_silence(1.5)
            session = _make_session(
                tmpdir, transport,
                transcripts=["hello there", "and again"],
                reply="Okay.")
            report = session.start(max_turns=1)
            self.assertEqual(report.end_reason, "completed")
            self.assertEqual(report.turns, 1)

    def test_no_remote_ends_honestly(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeTransport(publishers=())
            session = _make_session(tmpdir, transport, transcripts=[],
                                    max_wait_remote_s=0.2)
            report = session.start()
            self.assertEqual(report.end_reason, "no-remote")
            self.assertIn("nobody joined", report.error)
            self.assertEqual(report.turns, 0)

    def test_remote_leaves_mid_call(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeTransport()
            transport.feed_silence(0.6)
            transport.feed_tone(0.8)
            transport.feed_silence(1.5)

            def _drop() -> None:
                time.sleep(1.5)
                transport.drop_publishers()

            threading.Thread(target=_drop, daemon=True).start()
            session = _make_session(
                tmpdir, transport,
                transcripts=["hello there", "still here"],
                reply="Okay.")
            report = session.start()
            self.assertEqual(report.end_reason, "remote-left")

    def test_barge_in_stops_the_reply(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeTransport()
            bus = FakeBus()
            transport.feed_silence(0.6)
            transport.feed_tone(0.8)
            transport.feed_silence(1.5)
            # a long reply (4 s of audio) so the interruption lands mid-way
            session = _make_session(
                tmpdir, transport,
                transcripts=["hello there", "goodbye"],
                reply="This is a very long reply. It keeps going on.",
                bus=bus, tts_seconds=4.0)

            def _interrupt() -> None:
                # wait for publishing to start, then talk over her
                deadline = time.time() + 5.0
                while not transport.published and time.time() < deadline:
                    time.sleep(0.02)
                time.sleep(0.3)
                transport.feed_tone(1.5)

            threading.Thread(target=_interrupt, daemon=True).start()
            report = session.start()
            self.assertEqual(report.barge_ins, 1)
            self.assertTrue(report.spoken[0]["barged"])
            self.assertGreaterEqual(transport.cleared, 1)
            self.assertIn("voice.livekit.barge_in", bus.topics())
            # the reply was cut short: far fewer frames than 4 s of audio
            self.assertLess(len(transport.published), 300,
                            "barge-in did not stop the publish")
            self.assertEqual(report.end_reason, "user-exit")

    def test_stop_from_outside(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeTransport()
            transport.feed_silence(0.6)
            session = _make_session(
                tmpdir, transport, transcripts=["hello"],
                max_wait_remote_s=30.0)
            box: dict[str, LiveCallReport] = {}

            def _run() -> None:
                box["report"] = session.start()

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            time.sleep(0.5)
            session.stop()
            thread.join(timeout=10.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(box["report"].end_reason, "stopped")

    def test_start_without_config_raises_naming_the_gap(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            session = LiveKitCallSession(
                think=lambda text: "", stt=lambda path: "",
                tts=lambda text, profile: {},
                transport=FakeTransport(),
                config=LiveKitConfig.from_env(env={}),
                room="test-room", data_dir=tmpdir, bus=False)
            with self.assertRaises(LiveKitUnavailable) as ctx:
                session.start()
            self.assertIn("LIVEKIT_URL", str(ctx.exception))

    def test_start_without_room_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            session = LiveKitCallSession(
                think=lambda text: "", stt=lambda path: "",
                tts=lambda text, profile: {},
                transport=FakeTransport(),
                config=LiveKitConfig(url="wss://fake.invalid",
                                     token="fake-token"),
                room="", data_dir=tmpdir, bus=False)
            with self.assertRaises(LiveKitUnavailable):
                session.start()

    def test_tts_at_22050_resamples_to_48k_frames(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeTransport()
            transport.feed_silence(0.6)
            transport.feed_tone(0.8)
            transport.feed_silence(1.0)
            transport.feed_tone(0.8)
            transport.feed_silence(1.5)
            session = _make_session(
                tmpdir, transport,
                transcripts=["hello", "goodbye"],
                tts_seconds=1.0, tts_rate=22050)
            report = session.start()
            self.assertEqual(report.turns, 1)
            # 1 s of 22050 Hz audio -> ~1 s of 48 kHz -> ~100 frames
            self.assertGreater(len(transport.published), 50)
            self.assertLess(len(transport.published), 200)

    def test_report_format_and_dict(self):
        report = LiveCallReport(session_id="s1", room="r",
                                turns=2, barge_ins=1,
                                latencies_ms=[100.0, 300.0],
                                end_reason="user-exit",
                                spoken=[{"turn": 1, "heard": "hi",
                                         "said": ["Hello."],
                                         "barged": False}])
        stats = report.latency_stats()
        self.assertEqual(stats["count"], 2)
        # same percentile rule as the local session: int(p * n)
        self.assertEqual(stats["p50"], 300.0)
        data = report.to_dict()
        self.assertEqual(data["turns"], 2)
        self.assertEqual(data["end_reason"], "user-exit")
        text = report.format()
        self.assertIn("turns 2", text)
        self.assertIn("barge-ins 1", text)


class TestRtcTransportWithoutSdk(unittest.TestCase):
    def test_connect_refuses_honestly_without_sdk(self):
        if sdk_available():
            self.skipTest("livekit SDK installed here; refusal untestable")
        transport = RtcTransport()
        try:
            with self.assertRaises(LiveKitUnavailable) as ctx:
                transport.connect("wss://fake.invalid", "token",
                                  timeout=5.0)
            self.assertIn("LiveKit", str(ctx.exception))
        finally:
            transport.disconnect()

    def test_publish_frame_validates_size(self):
        transport = RtcTransport()
        with self.assertRaises(ValueError):
            transport.publish_frame(b"\x00" * 100)


# ── registry helpers ───────────────────────────────────────────────────────


class TestSessionRegistry(unittest.TestCase):
    def _session(self, tmpdir: str) -> LiveKitCallSession:
        transport = FakeTransport(publishers=())
        return LiveKitCallSession(
            think=lambda text: "", stt=lambda path: "",
            tts=lambda text, profile: {},
            transport=transport,
            config=LiveKitConfig(url="wss://fake.invalid",
                                 token="fake-token"),
            room="test-room", data_dir=tmpdir,
            # long wait on purpose: the tests end the session with
            # end_live_call, and a 0.2 s wait would race that stop.
            max_wait_remote_s=5.0, bus=False)

    def test_second_concurrent_start_refuses(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            first = self._session(tmpdir)
            second = self._session(tmpdir)
            try:
                start_live_call(first)
                with self.assertRaises(LiveKitUnavailable):
                    start_live_call(second)
            finally:
                end_live_call()

    def test_end_returns_report_and_clears(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self._session(tmpdir)
            start_live_call(session)
            report = end_live_call()
            self.assertIsNotNone(report)
            assert report is not None
            self.assertEqual(report.end_reason, "stopped")
            self.assertIsNone(get_live_call())
            self.assertIsNone(end_live_call())

    def test_status_without_session(self):
        status = live_call_status()
        self.assertIn("sdk_installed", status)
        self.assertIn("configured", status)
        self.assertIn("missing", status)

    def test_livekit_status_reads_last_call(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "live-last.json").write_text(json.dumps({
                "session_id": "s", "room": "r", "turns": 3,
                "barge_ins": 0, "end_reason": "user-exit", "error": "",
                "ear_to_ear_ms_p50": 100.0, "ear_to_ear_ms_p95": 200.0,
                "spoken": [],
            }))
            status = livekit_status(tmpdir)
            self.assertEqual(status["last_call"]["turns"], 3)


# ── tools ──────────────────────────────────────────────────────────────────


class _FakeOutcome:
    def __init__(self, ok: bool, value: dict | None = None):
        self.ok = ok
        self.value = value or {}


class _FakeTools:
    def call(self, name: str, **kwargs: Any) -> _FakeOutcome:
        assert name == "transcribe"
        return _FakeOutcome(True, {"text": "hello from the tool"})


class _FakeContext:
    def __init__(self) -> None:
        self.tools = _FakeTools()
        self.settings = mock.Mock()
        self.settings.audio.voice_data_dir = ""
        self.settings.audio.tts_engine = "auto"


class _FakeRegistry:
    def __init__(self) -> None:
        self.context = _FakeContext()
        self.tools: dict[str, Any] = {}

    def register(self, name: str, **kwargs: Any):  # noqa: ARG002
        def _deco(fn: Any) -> Any:
            self.tools[name] = fn
            return fn

        return _deco


class TestVoiceLiveTools(unittest.TestCase):
    def _registry(self) -> _FakeRegistry:
        from nomorals.tools import voice_live

        registry = _FakeRegistry()
        voice_live.register(registry)
        return registry

    def test_tools_registered(self):
        registry = self._registry()
        self.assertEqual(
            sorted(registry.tools),
            ["voice_live_end", "voice_live_start", "voice_live_status"])

    def test_status_tool_never_connects(self):
        registry = self._registry()
        result = registry.tools["voice_live_status"]()
        self.assertTrue(result["ok"])
        self.assertIn("sdk_installed", result["status"])

    def test_end_tool_with_nothing_running(self):
        registry = self._registry()
        result = registry.tools["voice_live_end"]()
        self.assertFalse(result["ok"])
        self.assertIn("no live call", result["error"])

    def test_start_tool_without_config_refuses(self):
        registry = self._registry()
        with mock.patch.dict(os.environ, {}, clear=True):
            result = registry.tools["voice_live_start"](room="r")
        self.assertFalse(result["ok"])
        self.assertTrue(result.get("missing"))

    def test_start_and_end_through_tools(self):
        import tempfile

        from nomorals.tools import voice_live

        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeTransport(publishers=())

            def _fake_build(reg: Any, room: str, **kwargs: Any):
                return LiveKitCallSession(
                    think=lambda text: "", stt=lambda path: "",
                    tts=lambda text, profile: {},
                    transport=transport,
                    config=LiveKitConfig(
                        url="wss://fake.invalid", token="fake-token"),
                    room=room, data_dir=tmpdir,
                    # long wait on purpose: the test ends the session
                    # with voice_live_end, and a short wait would race it.
                    max_wait_remote_s=5.0, bus=False)

            with mock.patch.object(voice_live, "_build_session",
                                   _fake_build), \
                    mock.patch.dict(os.environ, {
                        "LIVEKIT_URL": "wss://fake.invalid",
                        "LIVEKIT_TOKEN": "fake-token",
                    }, clear=True):
                started = registry.tools["voice_live_start"](room="r")
                self.assertTrue(started["ok"], started)
                self.assertEqual(started["room"], "r")
                status = registry.tools["voice_live_status"]()
                self.assertIn("session", status["status"])
                ended = registry.tools["voice_live_end"]()
                self.assertTrue(ended["ok"], ended)
                self.assertEqual(ended["report"]["end_reason"],
                                 "stopped")


# ── CLI ────────────────────────────────────────────────────────────────────


class TestVoiceLiveCli(unittest.TestCase):
    def test_parser_renders_live_subcommands(self):
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        ns = parser.parse_args(
            ["voice", "live", "start", "--room", "test-room"])
        self.assertEqual(ns.voice_action, "live")
        self.assertEqual(ns.voice_live_action, "start")
        self.assertEqual(ns.room, "test-room")
        ns2 = parser.parse_args(["voice", "live", "status"])
        self.assertEqual(ns2.voice_live_action, "status")
        ns3 = parser.parse_args(["voice", "live", "provision"])
        self.assertEqual(ns3.voice_live_action, "provision")

    def test_voice_help_names_live(self):
        import contextlib
        import io

        from nomorals.cmdline.parser import _parser

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                self.assertRaises(SystemExit):
            _parser().parse_args(["voice", "live", "start", "--help"])
        text = buf.getvalue()
        self.assertIn("usage: nm voice live start", text)
        self.assertIn("--room ROOM", text)
        # the live level lists its three verbs
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2), \
                self.assertRaises(SystemExit):
            _parser().parse_args(["voice", "live", "--help"])
        live_text = buf2.getvalue()
        self.assertIn("start", live_text)
        self.assertIn("status", live_text)
        self.assertIn("provision", live_text)

    def test_live_status_command_runs(self):
        import argparse
        import io
        from contextlib import redirect_stdout

        from nomorals.cmdline.commands.voice import _cmd_voice_live_status

        context = mock.Mock()
        context.settings.audio.voice_data_dir = ""
        args = argparse.Namespace(json=True)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = _cmd_voice_live_status(args, context)
        self.assertEqual(rc, 0)
        data = json.loads(out.getvalue())
        self.assertIn("sdk_installed", data)
        self.assertIn("missing", data)

    def test_live_provision_command_refuses_under_pytest(self):
        import argparse
        import io
        from contextlib import redirect_stdout

        from nomorals.cmdline.commands.voice import _cmd_voice_live_provision

        context = mock.Mock()
        args = argparse.Namespace(json=True)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = _cmd_voice_live_provision(args, context)
        self.assertEqual(rc, 1)
        data = json.loads(out.getvalue())
        self.assertFalse(data["ok"])


if __name__ == "__main__":
    unittest.main()
