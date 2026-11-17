"""Tests for the LiveKit video rung (voice/livekit_call.py,
tools/voice_live.py, `nm voice live video`, /videocall).

The fake room below moves REAL RGB24 frames through the REAL video
session state machine — publish loop + inbound snapshot saving —
against no room, no network, and no model. Nothing in the fake claims
success without doing the work: published frames are counted and
sized, inbound frames are really received and really saved as PNGs,
and the report records what was actually published and seen.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from nomorals.voice.livekit_call import (  # noqa: E402
    LiveKitConfig,
    LiveKitUnavailable,
    LiveKitVideoSession,
    LiveVideoReport,
    RtcTransport,
    end_video_session,
    get_video_session,
    image_frame_source,
    live_video_status,
    scene_frame_source,
    start_video_session,
    title_frame_source,
    video_session_status,
)

DIAGRAM = {
    "engine": "diagram",
    "kind": "diagram",
    "title": "Lagos to Abuja",
    "nodes": [
        {"id": "lagos", "label": "Lagos", "x": 0.15, "y": 0.5,
         "shape": "circle", "step": 0},
        {"id": "abuja", "label": "Abuja", "x": 0.85, "y": 0.5,
         "shape": "circle", "step": 1},
    ],
    "edges": [{"from": "lagos", "to": "abuja", "label": "750 km",
               "step": 1}],
}


def _solid_rgb(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    return bytes(rgb) * (width * height)


# ── the fake room ──────────────────────────────────────────────────────────


class FakeVideoTransport:
    """A scripted LiveKit room for video: serves remote RGB frames,
    records published frames."""

    def __init__(self, *, publishers=("owner",)):
        self._inbound: queue.Queue = queue.Queue()
        self._publishers = set(publishers)
        self.published: list[tuple[bytes, int, int]] = []
        self.connected = False
        self.disconnect_called = False
        self.connect_args: tuple[str, str] | None = None

    def feed_frame(self, rgb: bytes, width: int, height: int,
                   identity: str = "owner") -> None:
        self._inbound.put((bytes(rgb), width, height, identity))

    def feed_color_frame(self, width: int, height: int,
                         rgb: tuple[int, int, int],
                         identity: str = "owner") -> None:
        self.feed_frame(_solid_rgb(width, height, rgb), width, height,
                        identity)

    def connect(self, url: str, token: str, *, timeout: float = 30.0
                ) -> None:
        self.connect_args = (url, token)
        self.connected = True

    def poll_video_frame(self, timeout: float = 0.05):
        try:
            return self._inbound.get(timeout=min(timeout, 0.02))
        except queue.Empty:
            return None

    def publish_video_frame(self, rgb: bytes, width: int, height: int,
                            *, timeout: float = 5.0) -> None:
        # like the real transport: a wrong size is a ValueError naming
        # the frame, never a silent accept
        if len(rgb) != width * height * 3:
            raise ValueError(
                f"publish_video_frame needs one RGB24 frame "
                f"({width * height * 3} bytes), got {len(rgb)}")
        self.published.append((bytes(rgb), width, height))
        # pacing like a real capture: a frame takes its slot in time
        time.sleep(0.02)

    def remote_video_publishers(self) -> list[str]:
        return sorted(self._publishers)

    def is_connected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.disconnect_called = True
        self.connected = False


def _solid_source(width: int, height: int,
                  rgb: tuple[int, int, int] = (16, 32, 48)):
    """A frame source that returns the same solid frame forever."""
    frame = (_solid_rgb(width, height, rgb), width, height)

    def _source(t: float):
        return frame

    return _source


def _counted_source(width: int, height: int, *, frames: int):
    """A frame source that ends after ``frames`` frames."""
    state = {"n": 0}

    def _source(t: float):
        if state["n"] >= frames:
            return None
        state["n"] += 1
        return (_solid_rgb(width, height, (200, 30, 30)), width, height)

    return _source


def _make_session(tmpdir: str, transport: FakeVideoTransport, *,
                  source=None, fps: float = 10.0,
                  width: int = 64, height: int = 36,
                  max_duration_s: float = 0.0) -> LiveKitVideoSession:
    return LiveKitVideoSession(
        frame_source=source if source is not None
        else _solid_source(width, height),
        transport=transport,
        config=LiveKitConfig(url="wss://fake.invalid",
                             token="fake-token"),
        room="test-room",
        data_dir=tmpdir,
        width=width, height=height, fps=fps,
        bus=False,
    )


# ── frame sources ──────────────────────────────────────────────────────────


class TestFrameSources(unittest.TestCase):
    def test_title_card_is_real_rgb_at_the_right_size(self):
        source = title_frame_source("my-room", width=320, height=180)
        frame = source(0.0)
        assert frame is not None
        rgb, w, h = frame
        self.assertEqual((w, h), (320, 180))
        self.assertEqual(len(rgb), 320 * 180 * 3)
        img = Image.frombytes("RGB", (w, h), rgb)
        self.assertEqual(img.size, (320, 180))

    def test_title_card_clock_moves_with_time(self):
        source = title_frame_source("my-room", width=160, height=90)
        first = source(0.0)
        second = source(5.0)
        assert first is not None and second is not None
        self.assertNotEqual(first[0], second[0])

    def test_title_card_names_the_room(self):
        # the room name is drawn on the card: two rooms differ
        a = title_frame_source("room-alpha", width=160, height=90)(0.0)
        b = title_frame_source("room-beta", width=160, height=90)(0.0)
        assert a is not None and b is not None
        self.assertNotEqual(a[0], b[0])

    def test_image_source_returns_the_picture(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "still.png")
            Image.new("RGB", (80, 45), (9, 99, 199)).save(path)
            source = image_frame_source(path, width=64, height=36)
            first = source(0.0)
            second = source(99.0)
            assert first is not None and second is not None
            self.assertEqual(first[1:], (64, 36))
            self.assertEqual(len(first[0]), 64 * 36 * 3)
            # a still is a still: the same bytes at any time
            self.assertEqual(first[0], second[0])
            img = Image.frombytes("RGB", (64, 36), first[0])
            self.assertEqual(img.size, (64, 36))

    def test_image_source_missing_file_refuses_naming_the_gap(self):
        with self.assertRaises(LiveKitUnavailable) as ctx:
            image_frame_source("/no/such/still.png", width=64, height=36)
        self.assertIn("still.png", str(ctx.exception))

    def test_scene_source_renders_the_design(self):
        source = scene_frame_source(dict(DIAGRAM), width=64, height=36,
                                    duration=2.0)
        frame = source(0.5)
        assert frame is not None
        rgb, w, h = frame
        self.assertEqual((w, h), (64, 36))
        self.assertEqual(len(rgb), 64 * 36 * 3)
        img = Image.frombytes("RGB", (w, h), rgb)
        self.assertEqual(img.size, (64, 36))

    def test_scene_source_ends_when_not_looping(self):
        source = scene_frame_source(dict(DIAGRAM), width=64, height=36,
                                    duration=2.0, loop=False)
        self.assertIsNotNone(source(1.0))
        self.assertIsNone(source(2.0))

    def test_scene_source_loops_by_default(self):
        source = scene_frame_source(dict(DIAGRAM), width=64, height=36,
                                    duration=2.0)
        self.assertIsNotNone(source(1.0))
        # past the end it wraps to the start instead of ending
        self.assertIsNotNone(source(2.5))

    def test_scene_source_rejects_a_bad_design(self):
        with self.assertRaises(LiveKitUnavailable):
            scene_frame_source({"kind": "nope"}, width=64, height=36,
                               duration=2.0)

    def test_scene_source_rejects_a_bad_duration(self):
        with self.assertRaises(LiveKitUnavailable):
            scene_frame_source(dict(DIAGRAM), width=64, height=36,
                               duration=0.0)


# ── the session, end to end against the fake room ─────────────────────────


class TestVideoSession(unittest.TestCase):
    def test_publish_loop_completes_on_max_duration(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeVideoTransport(publishers=())
            session = _make_session(tmpdir, transport)
            report = session.start(max_duration_s=0.5)
            self.assertEqual(report.end_reason, "completed")
            self.assertEqual(report.room, "test-room")
            self.assertGreaterEqual(report.frames_published, 3)
            self.assertTrue(transport.published,
                            "no video frame was ever published")
            for rgb, w, h in transport.published:
                self.assertEqual(len(rgb), w * h * 3)
                self.assertEqual((w, h), (64, 36))
            self.assertTrue(transport.disconnect_called)
            self.assertEqual(transport.connect_args[0],
                             "wss://fake.invalid")
            self.assertGreater(report.duration_s, 0.0)
            last = json.loads(
                Path(tmpdir, "live-video-last.json").read_text())
            self.assertEqual(last["frames_published"],
                             report.frames_published)
            self.assertEqual(last["end_reason"], "completed")

    def test_source_ending_ends_the_session(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeVideoTransport(publishers=())
            session = _make_session(
                tmpdir, transport,
                source=_counted_source(64, 36, frames=3))
            report = session.start()
            self.assertEqual(report.end_reason, "source-ended")
            self.assertEqual(report.frames_published, 3)
            self.assertEqual(len(transport.published), 3)

    def test_remote_frames_become_real_png_snapshots(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeVideoTransport()
            # the remote side is already on camera when we join
            transport.feed_color_frame(64, 36, (255, 0, 0), "owner")
            transport.feed_color_frame(64, 36, (0, 255, 0), "owner")
            transport.feed_color_frame(64, 36, (0, 0, 255), "guest")
            session = _make_session(tmpdir, transport)
            report = session.start(max_duration_s=0.5)
            self.assertEqual(report.end_reason, "completed")
            self.assertGreaterEqual(
                report.frames_received.get("owner", 0), 2)
            self.assertGreaterEqual(
                report.frames_received.get("guest", 0), 1)
            self.assertIn("owner", report.snapshots)
            self.assertIn("guest", report.snapshots)
            for identity, path in report.snapshots.items():
                self.assertTrue(Path(path).is_file(), path)
                img = Image.open(path)
                self.assertEqual(img.size, (64, 36))
                self.assertIn("live-video-", Path(path).name)
                self.assertIn(identity.replace(" ", "_"),
                              Path(path).name)

    def test_snapshot_returns_the_latest_frame(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeVideoTransport()
            transport.feed_color_frame(64, 36, (255, 0, 0), "owner")
            session = _make_session(tmpdir, transport)
            box: dict[str, LiveVideoReport] = {}

            def _run() -> None:
                box["report"] = session.start(max_duration_s=0.6)

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            time.sleep(0.3)
            snap = session.snapshot("owner")
            thread.join(timeout=10.0)
            self.assertIsNotNone(snap)
            assert snap is not None
            self.assertEqual(snap["identity"], "owner")
            self.assertTrue(Path(snap["path"]).is_file())
            self.assertGreaterEqual(snap["frames_received"], 1)
            self.assertIsNone(session.snapshot("nobody"))

    def test_start_without_config_raises_naming_the_gap(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            session = LiveKitVideoSession(
                frame_source=_solid_source(64, 36),
                transport=FakeVideoTransport(),
                config=LiveKitConfig.from_env(env={}),
                room="test-room", data_dir=tmpdir, bus=False)
            with self.assertRaises(LiveKitUnavailable) as ctx:
                session.start()
            self.assertIn("LIVEKIT_URL", str(ctx.exception))

    def test_start_without_room_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            session = LiveKitVideoSession(
                frame_source=_solid_source(64, 36),
                transport=FakeVideoTransport(),
                config=LiveKitConfig(url="wss://fake.invalid",
                                     token="fake-token"),
                room="", data_dir=tmpdir, bus=False)
            with self.assertRaises(LiveKitUnavailable):
                session.start()

    def test_start_without_frame_source_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            session = LiveKitVideoSession(
                frame_source=None,
                transport=FakeVideoTransport(),
                config=LiveKitConfig(url="wss://fake.invalid",
                                     token="fake-token"),
                room="test-room", data_dir=tmpdir, bus=False)
            with self.assertRaises(LiveKitUnavailable) as ctx:
                session.start()
            self.assertIn("frame source", str(ctx.exception))

    def test_bad_frame_size_ends_with_an_honest_error(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeVideoTransport(publishers=())

            def _bad_source(t: float):
                return (b"\x00" * 10, 64, 36)  # wrong byte count

            session = _make_session(tmpdir, transport,
                                    source=_bad_source)
            report = session.start()
            self.assertEqual(report.end_reason, "error")
            self.assertIn("frame", report.error)

    def test_stop_from_outside(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeVideoTransport(publishers=())
            session = _make_session(tmpdir, transport)
            box: dict[str, LiveVideoReport] = {}

            def _run() -> None:
                box["report"] = session.start()

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            time.sleep(0.5)
            session.stop()
            thread.join(timeout=10.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(box["report"].end_reason, "stopped")

    def test_report_format_and_dict(self):
        report = LiveVideoReport(
            session_id="s1", room="r", source="title",
            frames_published=150, frames_received={"owner": 90},
            snapshots={"owner": "/tmp/live-video-s1-owner.png"},
            duration_s=10.0, end_reason="completed")
        data = report.to_dict()
        self.assertEqual(data["frames_published"], 150)
        self.assertEqual(data["frames_received"], {"owner": 90})
        self.assertEqual(data["end_reason"], "completed")
        text = report.format()
        self.assertIn("frames published 150", text)
        self.assertIn("room r", text)
        self.assertIn("owner", text)


# ── the real transport without the SDK ─────────────────────────────────────


class TestRtcTransportVideoWithoutSdk(unittest.TestCase):
    def test_connect_refuses_honestly_without_sdk(self):
        from nomorals.voice.livekit_call import sdk_available

        if sdk_available():
            self.skipTest("livekit SDK installed here; refusal untestable")
        transport = RtcTransport(video=True)
        try:
            with self.assertRaises(LiveKitUnavailable) as ctx:
                transport.connect("wss://fake.invalid", "token",
                                  timeout=5.0)
            self.assertIn("LiveKit", str(ctx.exception))
        finally:
            transport.disconnect()

    def test_publish_video_frame_validates_size(self):
        transport = RtcTransport(video=True)
        with self.assertRaises(ValueError):
            transport.publish_video_frame(b"\x00" * 100, 64, 36)

    def test_publish_video_frame_validates_dimensions(self):
        transport = RtcTransport(video=True, video_width=64,
                                 video_height=36)
        # right byte count for 32x18, wrong dimensions for the source
        with self.assertRaises(ValueError):
            transport.publish_video_frame(b"\x00" * (32 * 18 * 3),
                                          32, 18)

    def test_video_track_ignored_when_video_off(self):
        transport = RtcTransport()  # video off, like the voice call
        pub = mock.Mock()
        pub.mime_type = "video/vp8"
        participant = mock.Mock()
        participant.identity = "owner"
        transport._on_track_subscribed(mock.Mock(), pub, participant)
        self.assertEqual(transport.remote_video_publishers(), [])

    def test_video_track_noted_when_video_on(self):
        transport = RtcTransport(video=True)
        pub = mock.Mock()
        pub.mime_type = "video/vp8"
        participant = mock.Mock()
        participant.identity = "owner"
        transport._on_track_subscribed(mock.Mock(), pub, participant)
        self.assertEqual(transport.remote_video_publishers(), ["owner"])
        transport._on_track_unsubscribed(mock.Mock(), pub, participant)
        self.assertEqual(transport.remote_video_publishers(), [])

    def test_audio_track_still_noted_when_video_on(self):
        transport = RtcTransport(video=True)
        pub = mock.Mock()
        pub.mime_type = "audio/opus"
        participant = mock.Mock()
        participant.identity = "owner"
        transport._on_track_subscribed(mock.Mock(), pub, participant)
        self.assertEqual(transport.remote_audio_publishers(), ["owner"])
        self.assertEqual(transport.remote_video_publishers(), [])


# ── registry helpers ───────────────────────────────────────────────────────


class TestVideoSessionRegistry(unittest.TestCase):
    def _session(self, tmpdir: str) -> LiveKitVideoSession:
        return LiveKitVideoSession(
            frame_source=_solid_source(64, 36),
            transport=FakeVideoTransport(publishers=()),
            config=LiveKitConfig(url="wss://fake.invalid",
                                 token="fake-token"),
            room="test-room", data_dir=tmpdir,
            width=64, height=36, fps=10.0, bus=False)

    def test_second_concurrent_start_refuses(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            first = self._session(tmpdir)
            second = self._session(tmpdir)
            try:
                start_video_session(first)
                with self.assertRaises(LiveKitUnavailable):
                    start_video_session(second)
            finally:
                end_video_session()

    def test_end_returns_report_and_clears(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self._session(tmpdir)
            start_video_session(session)
            report = end_video_session()
            self.assertIsNotNone(report)
            assert report is not None
            self.assertEqual(report.end_reason, "stopped")
            self.assertIsNone(get_video_session())
            self.assertIsNone(end_video_session())

    def test_status_without_session(self):
        status = video_session_status()
        self.assertIn("sdk_installed", status)
        self.assertIn("configured", status)
        self.assertIn("missing", status)

    def test_live_video_status_reads_last_session(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "live-video-last.json").write_text(json.dumps({
                "session_id": "s", "room": "r", "source": "title",
                "frames_published": 42, "frames_received": {},
                "snapshots": {}, "duration_s": 3.0,
                "end_reason": "completed", "error": "",
            }))
            status = live_video_status(tmpdir)
            self.assertEqual(status["last_video"]["frames_published"], 42)


# ── tools ──────────────────────────────────────────────────────────────────


class _FakeOutcome:
    def __init__(self, ok: bool, value: dict | None = None):
        self.ok = ok
        self.value = value or {}


class _FakeRegistry:
    def __init__(self) -> None:
        self.context = mock.Mock()
        self.context.settings.audio.voice_data_dir = ""
        self.tools: dict[str, Any] = {}

    def register(self, name: str, **kwargs: Any):  # noqa: ARG002
        def _deco(fn: Any) -> Any:
            self.tools[name] = fn
            return fn

        return _deco


class TestVideoLiveTools(unittest.TestCase):
    def _registry(self) -> _FakeRegistry:
        from nomorals.tools import voice_live

        registry = _FakeRegistry()
        voice_live.register(registry)
        return registry

    def test_video_tools_registered(self):
        registry = self._registry()
        for name in ("video_live_start", "video_live_status",
                     "video_live_end", "video_live_snapshot"):
            self.assertIn(name, registry.tools, name)

    def test_status_tool_never_connects(self):
        registry = self._registry()
        result = registry.tools["video_live_status"]()
        self.assertTrue(result["ok"])
        self.assertIn("sdk_installed", result["status"])

    def test_end_tool_with_nothing_running(self):
        registry = self._registry()
        result = registry.tools["video_live_end"]()
        self.assertFalse(result["ok"])
        self.assertIn("no live video", result["error"])

    def test_snapshot_tool_with_nothing_running(self):
        registry = self._registry()
        result = registry.tools["video_live_snapshot"]()
        self.assertFalse(result["ok"])
        self.assertIn("no live video", result["error"])

    def test_start_tool_without_config_refuses(self):
        registry = self._registry()
        with mock.patch.dict(os.environ, {}, clear=True):
            result = registry.tools["video_live_start"](room="r")
        self.assertFalse(result["ok"])
        self.assertTrue(result.get("missing"))

    def test_start_tool_rejects_an_unknown_source(self):
        registry = self._registry()
        with mock.patch.dict(os.environ, {
                "LIVEKIT_URL": "wss://fake.invalid",
                "LIVEKIT_TOKEN": "fake-token",
        }, clear=True):
            result = registry.tools["video_live_start"](
                room="r", source="webcam")
        self.assertFalse(result["ok"])
        self.assertIn("webcam", result["error"])

    def test_start_and_end_through_tools(self):
        import tempfile

        from nomorals.tools import voice_live

        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeVideoTransport(publishers=())

            def _fake_build(reg: Any, room: str, **kwargs: Any):
                return LiveKitVideoSession(
                    frame_source=_solid_source(64, 36),
                    transport=transport,
                    config=LiveKitConfig(
                        url="wss://fake.invalid", token="fake-token"),
                    room=room, data_dir=tmpdir,
                    width=64, height=36, fps=10.0, bus=False)

            with mock.patch.object(voice_live, "_build_video_session",
                                   _fake_build), \
                    mock.patch.dict(os.environ, {
                        "LIVEKIT_URL": "wss://fake.invalid",
                        "LIVEKIT_TOKEN": "fake-token",
                    }, clear=True):
                started = registry.tools["video_live_start"](room="r")
                self.assertTrue(started["ok"], started)
                self.assertEqual(started["room"], "r")
                status = registry.tools["video_live_status"]()
                self.assertIn("session", status["status"])
                ended = registry.tools["video_live_end"]()
                self.assertTrue(ended["ok"], ended)
                self.assertEqual(ended["report"]["end_reason"],
                                 "stopped")


# ── CLI ────────────────────────────────────────────────────────────────────


class TestVideoLiveCli(unittest.TestCase):
    def test_parser_renders_video_subcommands(self):
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        ns = parser.parse_args(
            ["voice", "live", "video", "start", "--room", "test-room"])
        self.assertEqual(ns.voice_live_action, "video")
        self.assertEqual(ns.voice_live_video_action, "start")
        self.assertEqual(ns.room, "test-room")
        self.assertEqual(ns.source, "title")
        ns2 = parser.parse_args(["voice", "live", "video", "status"])
        self.assertEqual(ns2.voice_live_video_action, "status")
        ns3 = parser.parse_args(["voice", "live", "video", "snapshot",
                                 "--identity", "owner"])
        self.assertEqual(ns3.voice_live_video_action, "snapshot")
        self.assertEqual(ns3.identity, "owner")
        ns4 = parser.parse_args(
            ["voice", "live", "video", "start", "--room", "r",
             "--source", "scene", "--duration", "10", "--fps", "12"])
        self.assertEqual(ns4.source, "scene")
        self.assertEqual(ns4.duration, 10.0)
        self.assertEqual(ns4.fps, 12.0)

    def test_voice_live_help_names_video(self):
        import contextlib
        import io

        from nomorals.cmdline.parser import _parser

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                self.assertRaises(SystemExit):
            _parser().parse_args(["voice", "live", "--help"])
        live_text = buf.getvalue()
        self.assertIn("video", live_text)
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2), \
                self.assertRaises(SystemExit):
            _parser().parse_args(["voice", "live", "video", "--help"])
        video_text = buf2.getvalue()
        self.assertIn("start", video_text)
        self.assertIn("status", video_text)
        self.assertIn("snapshot", video_text)

    def test_video_status_command_runs(self):
        import argparse
        import io
        from contextlib import redirect_stdout

        from nomorals.cmdline.commands.voice import _cmd_voice_live_video_status

        context = mock.Mock()
        context.settings.audio.voice_data_dir = ""
        args = argparse.Namespace(json=True)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = _cmd_voice_live_video_status(args, context)
        self.assertEqual(rc, 0)
        data = json.loads(out.getvalue())
        self.assertIn("sdk_installed", data)
        self.assertIn("missing", data)


if __name__ == "__main__":
    unittest.main()
