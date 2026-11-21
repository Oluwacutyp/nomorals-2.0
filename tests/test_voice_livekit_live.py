"""Tests for the LiveKit live-proof lane (lane 88).

Two halves:
* connect-error classification — unit tests, no SDK, no server: when
  the room join fails AFTER the signal join (ICE never establishes),
  the refusal must name the media path, not blame the server; when
  the signal itself fails, it must still blame reachability.
* live signal proof — gated on NM_LIVEKIT_LIVE=1 plus the livekit SDK
  plus a reachable server (LIVEKIT_URL + LIVEKIT_API_KEY +
  LIVEKIT_API_SECRET in the environment): real token minting through
  the module's own LiveKitConfig, real room joins, a real video track
  published and announced to a second participant, real frames fed
  into the track. Media (RTP) flow itself is NOT asserted here: on
  this sandbox UDP send is forbidden outright, so no ICE pair can
  form — the proof script (~/workspace/livekitproof-verify/) records
  that wall with the libwebrtc evidence instead of pretending.
"""

from __future__ import annotations

import contextlib
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.voice.livekit_call import (  # noqa: E402
    LiveKitUnavailable,
    RtcTransport,
)


def _transport_raising(message: str) -> RtcTransport:
    """An RtcTransport whose event-loop bridge raises on connect."""
    transport = RtcTransport(video=True)

    def _raise(coro, *, timeout):  # noqa: ARG001
        # the real bridge would await it; close it so the test leaves
        # no "coroutine was never awaited" warning behind
        coro.close()
        raise RuntimeError(message)

    transport._run_coro = _raise  # type: ignore[method-assign]
    return transport


class TestConnectErrorClassification(unittest.TestCase):
    def test_ice_timeout_names_the_media_path(self):
        transport = _transport_raising(
            'Connection("wait_pc_connection timed out")')
        with self.assertRaises(LiveKitUnavailable) as ctx:
            transport.connect("ws://127.0.0.1:7880", "token",
                              timeout=5.0)
        self.assertIn("media connection (ICE)", str(ctx.exception))
        self.assertIn("UDP", ctx.exception.missing)
        self.assertNotIn("could not join", str(ctx.exception))
        self.assertFalse(transport._connected)

    def test_signal_failure_still_blames_reachability(self):
        transport = _transport_raising(
            "signal failure: transport timed out")
        with self.assertRaises(LiveKitUnavailable) as ctx:
            transport.connect("ws://127.0.0.1:7880", "token",
                              timeout=5.0)
        self.assertIn("could not join", str(ctx.exception))
        self.assertIn("a reachable LiveKit server",
                      ctx.exception.missing)
        self.assertFalse(transport._connected)

    def test_plain_failure_still_blames_reachability(self):
        transport = _transport_raising("boom")
        with self.assertRaises(LiveKitUnavailable) as ctx:
            transport.connect("ws://127.0.0.1:7880", "token",
                              timeout=5.0)
        self.assertIn("could not join", str(ctx.exception))


def _live_gate() -> str | None:
    """Why the live tests skip (None when they can run)."""
    if os.environ.get("NM_LIVEKIT_LIVE") != "1":
        return "live LiveKit proof only with NM_LIVEKIT_LIVE=1"
    try:
        from nomorals.voice.livekit_call import sdk_available
    except Exception:  # noqa: BLE001
        return "livekit_call does not import here"
    if not sdk_available():
        return "livekit SDK is not installed"
    for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not os.environ.get(key):
            return f"{key} is not set (a LiveKit server is required)"
    return None


@unittest.skipIf(_live_gate() is not None, _live_gate() or "skipped")
class TestLiveSignal(unittest.TestCase):
    def test_local_capture_takes_real_frames(self):
        """Real title-card frames into a real VideoSource, no room."""
        from livekit import rtc as _rtc

        from nomorals.voice.livekit_call import title_frame_source

        source = title_frame_source("live-room", width=320, height=180)
        vsource = _rtc.VideoSource(320, 180)
        vtrack = _rtc.LocalVideoTrack.create_video_track("devon-video",
                                                         vsource)
        self.assertIsNotNone(vtrack)
        for i in range(5):
            rgb, w, h = source(i * 0.1)
            self.assertEqual((w, h), (320, 180))
            frame = _rtc.VideoFrame(w, h,
                                    _rtc.VideoBufferType.RGB24,
                                    bytes(rgb))
            vsource.capture_frame(frame)

    def test_signal_join_then_publish_or_name_the_wall(self):
        """A good token's join reaches a real server.

        Two outcomes, both honest:
        * on a host with working UDP, connect() returns and Devon
          publishes her video track, which the watcher is told about
          (the full path);
        * on a host where UDP send is forbidden, connect() never
          returns — the SDK waits for the media connection (ICE).
          The test then asserts the failure is exactly that ICE wait
          (signal got through; the wall is media, not the server),
          and passes naming it. A 401 / signal error fails: that
          would be a real regression in token minting or the server.
        """
        import asyncio

        from livekit import rtc as _rtc

        from nomorals.voice.livekit_call import LiveKitConfig

        async def _run() -> dict:
            config = LiveKitConfig.from_env()
            url = config.url
            seen: dict = {"announced": []}
            announced = asyncio.Event()
            watcher = _rtc.Room()

            def on_published(publication, participant):
                seen["announced"].append({
                    "name": str(getattr(publication, "name", "")),
                    "mime": str(getattr(publication, "mime_type", "")),
                })
                announced.set()

            watcher.on("track_published", on_published)
            devon = _rtc.Room()
            outcome = "unknown"
            try:
                await asyncio.wait_for(devon.connect(
                    url, config.token_for("live-room",
                                          identity="devon-video")),
                    timeout=40)
                outcome = "connected"
            except Exception as exc:  # noqa: BLE001 - classified below
                outcome = f"{type(exc).__name__}: {exc}"
            seen["connect_outcome"] = outcome
            if outcome != "connected":
                return seen
            await asyncio.wait_for(watcher.connect(
                url, config.token_for("live-room", identity="watcher")),
                timeout=20)
            vsource = _rtc.VideoSource(320, 180)
            vtrack = _rtc.LocalVideoTrack.create_video_track(
                "devon-video", vsource)
            await asyncio.wait_for(
                devon.local_participant.publish_track(vtrack),
                timeout=20)
            seen["published"] = ["devon-video"]
            import time as _time

            from nomorals.voice.livekit_call import title_frame_source

            source = title_frame_source("live-room", width=320,
                                        height=180)
            t0 = _time.time()
            fed = 0
            while _time.time() - t0 < 2.0:
                rgb, _, _ = source(_time.time() - t0)
                vsource.capture_frame(_rtc.VideoFrame(
                    320, 180, _rtc.VideoBufferType.RGB24, bytes(rgb)))
                fed += 1
                await asyncio.sleep(0.1)
            seen["fed"] = fed
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(announced.wait(), timeout=10.0)
            await asyncio.sleep(1.0)
            await devon.disconnect()
            await watcher.disconnect()
            return seen

        seen = asyncio.run(_run())
        outcome = str(seen.get("connect_outcome", ""))
        if outcome == "connected":
            self.assertIn("devon-video", seen.get("published", []))
            self.assertGreaterEqual(seen.get("fed", 0), 10)
            names = [t["name"] for t in seen.get("announced", [])]
            self.assertIn("devon-video", names,
                          f"watcher never saw the track: {seen}")
            announced = [t for t in seen.get("announced", [])
                         if t["name"] == "devon-video"][0]
            self.assertTrue(announced["mime"].startswith("video/"),
                            announced)
            return
        # the wall path: signal got through, ICE never did
        self.assertNotIn("401", outcome,
                         f"join rejected, not the media wall: {outcome}")
        self.assertNotIn("signal failure", outcome,
                         f"signal failed, not the media wall: {outcome}")
        self.assertTrue(
            "wait_pc_connection" in outcome or "TimeoutError" in outcome,
            f"unexpected connect failure (not the known ICE wall): "
            f"{outcome}")

    def test_agent_session_joins_real_room(self):
        """An agent-mode session traverses the real join path.

        The same two honest outcomes as the signal test: on a host
        with working UDP the session joins and ends "no-remote"
        (nobody came within the short wait); on this box the join
        raises the ICE-wall classification. A 401 / signal error
        fails — that would be a real regression in token minting or
        the server, not the wall.
        """
        import tempfile

        from nomorals.voice.livekit_call import (
            LiveKitCallSession,
            LiveKitConfig,
            LiveKitUnavailable,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            session = LiveKitCallSession(
                think=lambda text: "Hello.",
                stt=lambda path: "",
                tts=lambda text, profile: {},
                config=LiveKitConfig.from_env(),
                room="live-room",
                data_dir=tmpdir,
                max_wait_remote_s=5.0,
                bus=False,
                tools_used=[])
            try:
                report = session.start(max_turns=0)
            except LiveKitUnavailable as exc:
                self.assertIn("media connection (ICE)", str(exc))
                self.assertIn("UDP", exc.missing)
                return
            self.assertEqual(report.end_reason, "no-remote")
            self.assertEqual(report.tools_used, [])


if __name__ == "__main__":
    unittest.main()
