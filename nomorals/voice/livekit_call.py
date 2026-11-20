"""LiveKit real-time voice calls — Devon as a participant in a WebRTC room.

The local voice loop (``nomorals/voice/session.py``) talks through the
box's own mic and speakers. This rung puts her on the NETWORK: she joins
a LiveKit room as a participant, hears remote participants over WebRTC,
and answers with her own voice — the same brain, the same TTS/STT rungs,
the same turn-taking and barge-in, carried over a live call instead of
a local microphone.

Transport pick — the official ``livekit`` Python SDK (PyPI ``livekit``
1.1.21, room client) + ``livekit-api`` 1.2.3 (token minting), transport
ONLY. The pick is from the code actually read, not from memory:

* Own WebRTC is NOT built: DTLS-SRTP, ICE, Opus and congestion control
  in stdlib Python is a real wall, named as one — not a design choice.
* ``aiortc`` is NOT picked: it has no LiveKit room interop, so the
  LiveKit signalling protocol (protobuf over WebSocket) would still have
  to be hand-built, and it drags heavier native deps (av/ffmpeg) for
  less. It buys nothing the official SDK does not already do.
* ``livekit-agents`` (1.8.10) is NOT adopted: it bundles its own
  STT/LLM/TTS plugin pipeline and turn detector. Adopting it would fork
  a SECOND voice system beside Devon's own brain, TTS and STT rungs —
  the exact parallel system this rung must not build. Devon uses the
  thin room client (``livekit.rtc``) and keeps her own pipeline.
* The SDK is Apache-2.0, zero $ cost self-hosted: the LiveKit server
  itself is open-source (Apache-2.0, Go on Pion WebRTC); ``livekit-server
  --dev`` runs a local room on 127.0.0.1:7880 with the devkey/secret
  pair. LiveKit Cloud is the paid hosted alternative — never required.
* Wheels exist for the EC2 shape: ``livekit-1.1.21-py3-none-win_amd64``
  and ``livekit_api-1.2.3-py3-none-any`` were downloaded and verified
  against PyPI on 2026-11-14, so the provision path works on the
  Windows box. The SDK imports lazily: importing this module never
  fails when the SDK is absent — the session refuses honestly instead.

Audio rates: the room is natively 48 kHz Opus. Devon publishes at
48 kHz mono and subscribes at 48 kHz mono; the rung resamples to her
16 kHz STT/VAD rate on the way in and from the TTS wav rate on the way
out, in pure stdlib (integer-ratio box-average down, zero-order-hold
up, linear interpolation otherwise). No numpy, no new dependency.

Video has its own rung in this same module (``LiveKitVideoSession``,
``RtcTransport`` with ``video=True``): Devon publishes a real video
track — a live title card, a still image, or her own motion-studio
scene rendered frame by frame — and subscribes to the room's video,
saving one snapshot PNG per publisher per second. The video session
joins as a second participant (identity ``devon-video``) so it can run
beside the voice call in the same room; with a pre-signed
``LIVEKIT_TOKEN`` the identity is whatever the token says, so
simultaneous voice + video wants API key + secret minting.

Credentials: ``LIVEKIT_URL`` + either a pre-signed ``LIVEKIT_TOKEN``
or ``LIVEKIT_API_KEY`` + ``LIVEKIT_API_SECRET`` (token minted locally
with ``livekit.api.AccessToken``). Read from the environment at session
start, never from config or code, never logged, never journaled.

Session lifecycle: start / leave / status. A session that cannot start
(no SDK, no server configured, no network, bad token, nobody ever
joins) says so plainly with the reason and the missing piece — honest
unavailable, never a fake "connected". Sessions live in the process
that started them: the CLI runs one in the foreground (like
``nm voice call``); the tools start one on a daemon thread in the
agent runtime and end it with ``voice_live_end``. There is no CLI stop
because a CLI process cannot reach into another process's thread
without killing the whole agent — the module docstring says so rather
than shipping a stop that kills things.

Layering: this module is L4 (``voice``). It imports stdlib + core (L1)
+ the same-layer local session (VAD, speakify, wav helpers). The brain
is INJECTED as a ``think`` callable (the CLI wires it to
PartnerRuntime; the tools wire it to ``brain_for``) — never imported
from here, and never the L5 runtime.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..core.events import Event, global_bus
from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .failures import record_failure
from .session import (
    CHUNK_BYTES,
    SAMPLE_RATE,
    make_vad,
    read_wav_bytes,
    speakify,
    write_wav_bytes,
)

__all__ = [
    "LiveKitUnavailable",
    "LiveKitConfig",
    "LiveCallReport",
    "LiveKitCallSession",
    "LiveKitTransport",
    "LiveKitVideoTransport",
    "LiveVideoReport",
    "LiveKitVideoSession",
    "RtcTransport",
    "sdk_available",
    "sdk_version",
    "provision_livekit",
    "resample_pcm",
    "livekit_status",
    "start_live_call",
    "end_live_call",
    "get_live_call",
    "live_call_status",
    "title_frame_source",
    "image_frame_source",
    "scene_frame_source",
    "start_video_session",
    "end_video_session",
    "get_video_session",
    "video_session_status",
    "live_video_status",
    "LIVEKIT_SAMPLE_RATE",
    "VIDEO_IDENTITY",
    "VIDEO_DEFAULT_SIZE",
    "VIDEO_DEFAULT_FPS",
]

_log = get_logger(__name__)

#: Pinned SDK versions — verified against PyPI on 2026-11-14 (wheels
#: exist for Linux x86_64 and Windows amd64, py3-none).
LIVEKIT_SDK_PIN = "1.1.21"
LIVEKIT_API_PIN = "1.2.3"

#: The room's native rate: Opus at 48 kHz. Publish and subscribe here.
LIVEKIT_SAMPLE_RATE = 48000
#: One 10 ms frame at 48 kHz mono int16: 480 samples, 960 bytes.
FRAME_SAMPLES_48K = 480
FRAME_BYTES_48K = FRAME_SAMPLES_48K * 2

#: How long to wait for a remote participant to start publishing audio
#: before the call ends as "no-remote".
DEFAULT_MAX_WAIT_REMOTE_S = 120.0
#: How long a remote silence may last mid-call before the call ends as
#: "remote-left" (the last publisher left without a disconnect event).
REMOTE_SILENCE_END_S = 30.0

#: End words — the same set as the local session.
_END_WORDS = {"exit", "quit", "goodbye", "bye", "stop listening"}

#: The video session joins as a second participant, so it can run
#: beside the voice call in the same room.
VIDEO_IDENTITY = "devon-video"
#: Default publish size and rate for the video track.
VIDEO_DEFAULT_SIZE = (640, 360)
VIDEO_DEFAULT_FPS = 15.0
#: One snapshot PNG per publisher at most this often.
VIDEO_SNAPSHOT_EVERY_S = 1.0
#: Inbound video frames held at once; the pump drops the oldest when
#: the session is slower than the room (video never blocks the pump).
VIDEO_MAX_QUEUE_FRAMES = 8


class LiveKitUnavailable(Exception):
    """The call cannot start (or continue) — with the missing piece.

    ``missing`` names what is absent in plain words ("the livekit SDK",
    "LIVEKIT_URL", "a LiveKit server at that URL") so the caller can
    say it to the owner instead of a traceback.
    """

    def __init__(self, message: str, *, missing: str = "") -> None:
        super().__init__(message)
        self.missing = missing


# ── config (environment only — never config files, never code) ────────────


@dataclass
class LiveKitConfig:
    """Everything a call needs, read from the environment at start."""

    url: str = ""
    api_key: str = ""
    api_secret: str = ""
    token: str = ""

    @classmethod
    def from_env(cls, env: Any = None) -> LiveKitConfig:
        source = os.environ if env is None else env
        return cls(
            url=str(source.get("LIVEKIT_URL", "") or "").strip(),
            api_key=str(source.get("LIVEKIT_API_KEY", "") or "").strip(),
            api_secret=str(source.get("LIVEKIT_API_SECRET", "")
                           or "").strip(),
            token=str(source.get("LIVEKIT_TOKEN", "") or "").strip(),
        )

    def missing(self) -> list[str]:
        """What is absent, in plain words. Empty means ready to mint."""
        out: list[str] = []
        if not self.url:
            out.append("LIVEKIT_URL (the LiveKit server address)")
        if not self.token and not (self.api_key and self.api_secret):
            out.append("LIVEKIT_TOKEN (a pre-signed token) or "
                       "LIVEKIT_API_KEY + LIVEKIT_API_SECRET "
                       "(to mint one)")
        return out

    def describe_missing(self) -> str:
        return "; ".join(self.missing())

    def token_for(self, room: str, *, identity: str = "devon") -> str:
        """The access token for this room.

        A pre-signed ``LIVEKIT_TOKEN`` is used verbatim. Otherwise the
        token is minted locally with ``livekit.api.AccessToken`` — the
        secret never leaves the box.
        """
        if self.token:
            return self.token
        if not (self.api_key and self.api_secret):
            raise LiveKitUnavailable(
                "no LiveKit token and no API key + secret to mint one",
                missing=self.describe_missing())
        try:
            from livekit import api as _lkapi
        except ImportError as exc:
            raise LiveKitUnavailable(
                "the livekit SDK is not installed — run "
                "`nm voice live provision` first",
                missing="the livekit SDK") from exc
        token = (
            _lkapi.AccessToken(self.api_key, self.api_secret)
            .with_identity(identity)
            .with_grants(_lkapi.VideoGrants(
                room_join=True, room=room,
                can_publish=True, can_subscribe=True))
            .with_ttl(datetime.timedelta(hours=6))
            .to_jwt()
        )
        return str(token)


# ── SDK presence + provisioning ────────────────────────────────────────────


def sdk_available() -> bool:
    """True when both SDK packages import (rtc client + token minting)."""
    try:
        import importlib.util as _ilu

        return (_ilu.find_spec("livekit.rtc") is not None
                and _ilu.find_spec("livekit.api") is not None)
    except Exception:  # noqa: BLE001 — any import failure is "not here"
        return False


def sdk_version() -> str | None:
    """Installed ``livekit`` version, or None when it is not installed."""
    try:
        from importlib.metadata import version as _version

        return _version("livekit")
    except Exception:  # noqa: BLE001
        return None


def provision_livekit(*, timeout_s: int = 1200) -> dict[str, Any]:
    """Install the LiveKit SDK packages (``livekit`` + ``livekit-api``).

    Bounded pip install, never under pytest, never raises — the result
    dict says what happened. The SDK is the only third-party piece of
    this rung, and the only one that cannot be stdlib: WebRTC media
    transport is the wall named in the module docstring.
    """
    if "pytest" in sys.modules:
        return {"ok": False, "installed": False,
                "error": "refusing to pip install under pytest"}
    if sdk_available():
        return {"ok": True, "installed": False,
                "version": sdk_version(),
                "detail": "livekit SDK already installed"}
    steps: list[str] = []
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             f"livekit=={LIVEKIT_SDK_PIN}",
             f"livekit-api=={LIVEKIT_API_PIN}"],
            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "installed": False,
                "error": f"pip install timed out after {timeout_s}s"}
    steps.append(f"pip install: rc={proc.returncode}")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-500:]
        return {"ok": False, "installed": False,
                "error": f"pip install failed (rc={proc.returncode}): {tail}",
                "steps": steps}
    if not sdk_available():
        return {"ok": False, "installed": False,
                "error": "pip install reported success but livekit "
                         "still does not import",
                "steps": steps}
    return {"ok": True, "installed": True,
            "version": sdk_version(), "steps": steps}


# ── resampling (pure stdlib) ───────────────────────────────────────────────


def resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono int16 PCM between rates. Pure stdlib, no numpy.

    * same rate → returned unchanged.
    * integer down-ratio (48k → 16k) → box-average each group: the
      crude anti-alias filter, and the honest one for this job.
    * integer up-ratio (16k → 48k) → repeat each sample (zero-order
      hold).
    * anything else → linear interpolation.
    An odd trailing byte is dropped, never guessed at.
    """
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if src_rate == dst_rate:
        return pcm
    pcm = pcm[:len(pcm) // 2 * 2]
    n = len(pcm) // 2
    if n == 0:
        return b""
    samples = struct.unpack(f"<{n}h", pcm)
    if src_rate % dst_rate == 0:
        ratio = src_rate // dst_rate
        out = bytearray()
        for i in range(0, n - ratio + 1, ratio):
            out += struct.pack("<h", sum(samples[i:i + ratio]) // ratio)
        return bytes(out)
    if dst_rate % src_rate == 0:
        ratio = dst_rate // src_rate
        out = bytearray()
        for value in samples:
            out += struct.pack("<h", value) * ratio
        return bytes(out)
    total = int(round(n * dst_rate / src_rate))
    if total <= 0:
        return b""
    out = bytearray()
    for i in range(total):
        pos = i * (n - 1) / max(1, total - 1) if total > 1 else 0.0
        lo = int(pos)
        hi = min(n - 1, lo + 1)
        frac = pos - lo
        value = int(round(samples[lo] * (1.0 - frac)
                          + samples[hi] * frac))
        out += struct.pack("<h", max(-32768, min(32767, value)))
    return bytes(out)


def _pcm_chunks(pcm: bytes, chunk_bytes: int) -> list[bytes]:
    """Split PCM into whole chunks; a short tail is dropped, not padded."""
    return [pcm[i:i + chunk_bytes]
            for i in range(0, len(pcm) - chunk_bytes + 1, chunk_bytes)]


# ── transport seam (sync from the session's point of view) ─────────────────


class LiveKitTransport(Protocol):
    """What the session needs from a room, in sync calls.

    The real SDK is async; the transport owns its event loop on a
    daemon thread and bridges to sync here, exactly the way
    ``make_bridge_stt`` bridges the async bridge to the sync local
    session. A fake implementing this protocol drives the REAL session
    state machine in tests — the fake must move real frames, never
    claim success without doing the work.
    """

    def connect(self, url: str, token: str, *, timeout: float = 30.0
                ) -> None: ...

    def poll_inbound(self, timeout: float = 0.05) -> bytes | None:
        """One inbound 48 kHz PCM chunk, or None on timeout. Any length."""
        ...

    def publish_frame(self, pcm48: bytes, *, timeout: float = 5.0) -> None:
        """Publish one 10 ms frame (960 bytes at 48 kHz mono int16)."""
        ...

    def clear_outgoing(self) -> None:
        """Drop queued-but-unsent outgoing audio (barge-in)."""
        ...

    def remote_audio_publishers(self) -> list[str]:
        """Identities currently publishing an audio track."""
        ...

    def is_connected(self) -> bool: ...

    def disconnect(self) -> None: ...


class LiveKitVideoTransport(Protocol):
    """What the video session needs from a room, in sync calls.

    The same room connection as the voice transport — one
    ``RtcTransport`` implements both — with video opted in at
    construction. A fake implementing this protocol drives the REAL
    video session in tests: published frames are counted and sized,
    inbound frames are really received and really saved as snapshots.
    """

    def connect(self, url: str, token: str, *, timeout: float = 30.0
                ) -> None: ...

    def publish_video_frame(self, rgb: bytes, width: int, height: int,
                            *, timeout: float = 5.0) -> None:
        """Publish one RGB24 frame (``width * height * 3`` bytes).

        The capture itself is synchronous, so the timeout does not
        apply — it is kept for symmetry with the audio publish.
        """
        ...

    def poll_video_frame(self, timeout: float = 0.05
                         ) -> tuple[bytes, int, int, str] | None:
        """One inbound RGB24 frame (bytes, width, height, identity),
        or None on timeout."""
        ...

    def remote_video_publishers(self) -> list[str]:
        """Identities currently publishing a video track."""
        ...

    def is_connected(self) -> bool: ...

    def disconnect(self) -> None: ...


def _classify_connect_error(url: str, exc: Exception
                            ) -> tuple[str, str]:
    """(message, missing) for a failed room connect.

    The SDK's connect fails in two places with very different meanings,
    and the caller deserves the true one:
    * signal failure — the server was never reached (bad URL, down,
      proxy in the way, bad token): the missing piece is a reachable
      server.
    * the room joined but the media connection (ICE) never established
      — "wait_pc_connection timed out": the server is fine, the media
      path is not (UDP blocked on this host, no ICE pair forms).
      Saying "no reachable server" there sends the owner chasing a
      healthy server. Proven live 2026-11-18: signal joined in under
      a second while every STUN ping failed to send (UDP forbidden
      on the host) and the wait timed out.
    """
    text = str(exc)
    if "wait_pc_connection" in text or "pc_connection" in text:
        return (
            f"joined the LiveKit room at {url} but the media "
            f"connection (ICE) never established: "
            f"{type(exc).__name__}: {text}",
            "a working media (ICE/UDP) path to the LiveKit server — "
            "UDP is blocked on this host, or no ICE pair forms",
        )
    return (
        f"could not join the LiveKit room at {url}: "
        f"{type(exc).__name__}: {text}",
        f"a reachable LiveKit server at {url}",
    )


class RtcTransport:
    """The real transport: ``livekit.rtc`` Room + AudioSource/AudioStream.

    All ``livekit`` imports happen inside methods, so importing this
    module never requires the SDK — the session checks
    :func:`sdk_available` first and refuses honestly without it.
    """

    def __init__(self, *, video: bool = False,
                 video_width: int = VIDEO_DEFAULT_SIZE[0],
                 video_height: int = VIDEO_DEFAULT_SIZE[1]) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._inbound: queue.Queue[bytes] = queue.Queue(maxsize=512)
        self._room: Any = None
        self._source: Any = None
        self._streams: dict[str, Any] = {}
        self._publishers: set[str] = set()
        self._publishers_lock = threading.Lock()
        self._connected = False
        self._dropped_inbound = 0
        # -- video (opt-in: the voice call builds this transport with
        #    video off and its behavior is byte-identical) --------------
        self._video = video
        self._video_width = video_width
        self._video_height = video_height
        self._video_source: Any = None
        self._inbound_video: queue.Queue = queue.Queue(
            maxsize=VIDEO_MAX_QUEUE_FRAMES)
        self._video_publishers: set[str] = set()
        self._video_publishers_lock = threading.Lock()
        self._video_streams: dict[str, Any] = {}
        self._dropped_video = 0

    # -- loop plumbing ---------------------------------------------------
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and self._thread is not None:
            return self._loop
        loop = asyncio.new_event_loop()
        self._loop = loop

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True,
                                        name="livekit-rtc")
        self._thread.start()
        return loop

    def _run_coro(self, coro: Any, *, timeout: float) -> Any:
        loop = self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=timeout)

    def _stop_loop(self) -> None:
        loop, thread = self._loop, self._thread
        self._loop, self._thread = None, None
        if loop is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 — best effort
                loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5.0)

    # -- room events (run on the loop thread) -----------------------------
    def _note_publisher(self, identity: str, on: bool) -> None:
        with self._publishers_lock:
            if on:
                self._publishers.add(identity)
            else:
                self._publishers.discard(identity)

    def _note_video_publisher(self, identity: str, on: bool) -> None:
        with self._video_publishers_lock:
            if on:
                self._video_publishers.add(identity)
            else:
                self._video_publishers.discard(identity)

    def _on_track_subscribed(self, track: Any, publication: Any,
                             participant: Any) -> None:
        mime = str(getattr(publication, "mime_type", "") or "")
        identity = str(getattr(participant, "identity", "") or "remote")
        if mime.startswith("video/"):
            if not self._video:
                return  # video track — this transport is audio only
            self._note_video_publisher(identity, True)
            loop = self._loop
            if loop is None:
                return
            asyncio.run_coroutine_threadsafe(
                self._pump_video(track, identity), loop)
            return
        if mime and not mime.startswith("audio/"):
            return  # video / data track — this rung is audio only
        self._note_publisher(identity, True)
        loop = self._loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._pump_audio(track, identity), loop)

    def _on_track_unsubscribed(self, track: Any, publication: Any,
                               participant: Any) -> None:
        mime = str(getattr(publication, "mime_type", "") or "")
        identity = str(getattr(participant, "identity", "") or "remote")
        if mime.startswith("video/"):
            self._note_video_publisher(identity, False)
            return
        if mime and not mime.startswith("audio/"):
            return
        self._note_publisher(identity, False)

    def _on_participant_disconnected(self, participant: Any) -> None:
        identity = str(getattr(participant, "identity", "") or "remote")
        self._note_publisher(identity, False)
        self._note_video_publisher(identity, False)

    async def _pump_audio(self, track: Any, identity: str) -> None:
        """Drain one subscribed audio track into the inbound queue."""
        from livekit import rtc as _rtc

        try:
            stream = _rtc.AudioStream.from_track(
                track=track, sample_rate=LIVEKIT_SAMPLE_RATE,
                num_channels=1)
        except Exception as exc:  # noqa: BLE001 — a bad track ends the pump
            record_failure("livekit_call.RtcTransport._pump_audio", exc)
            return
        self._streams[identity] = stream
        try:
            async for frame in stream:
                try:
                    self._inbound.put_nowait(bytes(frame.data))
                except queue.Full:
                    self._dropped_inbound += 1
        except Exception as exc:  # noqa: BLE001 — stream end is normal
            record_failure("livekit_call.RtcTransport._pump_audio", exc)
        finally:
            self._streams.pop(identity, None)
            with contextlib.suppress(Exception):  # noqa: BLE001 — best effort
                await stream.aclose()

    async def _pump_video(self, track: Any, identity: str) -> None:
        """Drain one subscribed video track into the inbound queue.

        Frames arrive as RGB24 (the session's working format); the
        queue holds the latest few and drops the oldest when the
        session is slower than the room — video never blocks the pump.
        """
        from livekit import rtc as _rtc

        try:
            stream = _rtc.VideoStream.from_track(
                track=track, format=_rtc.VideoBufferType.RGB24)
        except Exception as exc:  # noqa: BLE001 — a bad track ends the pump
            record_failure("livekit_call.RtcTransport._pump_video", exc)
            return
        self._video_streams[identity] = stream
        try:
            async for event in stream:
                frame = event.frame
                try:
                    rgb = bytes(frame.data)
                    item = (rgb, int(frame.width), int(frame.height),
                            identity)
                except Exception:  # noqa: BLE001 — a bad frame is dropped
                    continue
                try:
                    self._inbound_video.put_nowait(item)
                except queue.Full:
                    self._dropped_video += 1
                    with contextlib.suppress(queue.Empty):
                        self._inbound_video.get_nowait()
                    with contextlib.suppress(queue.Full):
                        self._inbound_video.put_nowait(item)
        except Exception as exc:  # noqa: BLE001 — stream end is normal
            record_failure("livekit_call.RtcTransport._pump_video", exc)
        finally:
            self._video_streams.pop(identity, None)
            with contextlib.suppress(Exception):  # noqa: BLE001 — best effort
                await stream.aclose()

    async def _async_connect(self, url: str, token: str) -> None:
        from livekit import rtc as _rtc

        loop = asyncio.get_running_loop()
        room = _rtc.Room(loop=loop)
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("participant_disconnected",
                self._on_participant_disconnected)
        await room.connect(url, token)
        source = _rtc.AudioSource(LIVEKIT_SAMPLE_RATE, 1, loop=loop)
        track = _rtc.LocalAudioTrack.create_audio_track(
            "devon-voice", source)
        await room.local_participant.publish_track(track)
        self._room = room
        self._source = source
        if self._video:
            vsource = _rtc.VideoSource(self._video_width,
                                       self._video_height)
            vtrack = _rtc.LocalVideoTrack.create_video_track(
                "devon-video", vsource)
            await room.local_participant.publish_track(vtrack)
            self._video_source = vsource
        self._connected = True

    async def _async_disconnect(self) -> None:
        room, source = self._room, self._source
        vsource = self._video_source
        self._room, self._source = None, None
        self._video_source = None
        self._connected = False
        if source is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 — best effort
                await source.aclose()
        if vsource is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 — best effort
                await vsource.aclose()
        if room is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 — best effort
                await room.disconnect()

    # -- LiveKitTransport (sync) ------------------------------------------
    def connect(self, url: str, token: str, *, timeout: float = 30.0
                ) -> None:
        try:
            self._run_coro(self._async_connect(url, token),
                           timeout=timeout)
        except Exception as exc:
            self._connected = False
            message, missing = _classify_connect_error(url, exc)
            raise LiveKitUnavailable(message, missing=missing) from exc

    def poll_inbound(self, timeout: float = 0.05) -> bytes | None:
        try:
            return self._inbound.get(timeout=timeout)
        except queue.Empty:
            return None

    def publish_frame(self, pcm48: bytes, *, timeout: float = 5.0) -> None:
        if len(pcm48) != FRAME_BYTES_48K:
            raise ValueError(
                f"publish_frame needs one 10 ms frame "
                f"({FRAME_BYTES_48K} bytes), got {len(pcm48)}")
        if self._source is None:
            raise LiveKitUnavailable("not connected to a LiveKit room",
                                     missing="a LiveKit room connection")
        from livekit import rtc as _rtc

        frame = _rtc.AudioFrame(pcm48, LIVEKIT_SAMPLE_RATE, 1,
                                FRAME_SAMPLES_48K)
        self._run_coro(self._source.capture_frame(frame), timeout=timeout)

    def clear_outgoing(self) -> None:
        if self._source is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 — best effort
                self._source.clear_queue()

    def remote_audio_publishers(self) -> list[str]:
        with self._publishers_lock:
            return sorted(self._publishers)

    def publish_video_frame(self, rgb: bytes, width: int, height: int,
                            *, timeout: float = 5.0) -> None:
        """Publish one RGB24 frame on the video track.

        The capture itself is synchronous (``VideoSource.capture_frame``
        is a plain FFI call), so the timeout does not apply. Raises
        ValueError on a wrong size, LiveKitUnavailable when no video
        track is live.
        """
        if len(rgb) != width * height * 3:
            raise ValueError(
                f"publish_video_frame needs one RGB24 frame "
                f"({width * height * 3} bytes), got {len(rgb)}")
        if (width, height) != (self._video_width, self._video_height):
            raise ValueError(
                f"publish_video_frame size {(width, height)} does not "
                f"match the video source "
                f"{(self._video_width, self._video_height)}")
        if self._video_source is None:
            raise LiveKitUnavailable("not publishing video in a LiveKit "
                                     "room",
                                     missing="a LiveKit video connection")
        from livekit import rtc as _rtc

        frame = _rtc.VideoFrame(width, height,
                                _rtc.VideoBufferType.RGB24, bytes(rgb))
        try:
            self._video_source.capture_frame(frame)
        except Exception as exc:
            raise LiveKitUnavailable(
                f"could not publish the video frame: "
                f"{type(exc).__name__}: {exc}",
                missing="a live video track") from exc

    def poll_video_frame(self, timeout: float = 0.05
                         ) -> tuple[bytes, int, int, str] | None:
        try:
            return self._inbound_video.get(timeout=timeout)
        except queue.Empty:
            return None

    def remote_video_publishers(self) -> list[str]:
        with self._video_publishers_lock:
            return sorted(self._video_publishers)

    def is_connected(self) -> bool:
        if not self._connected or self._room is None:
            return False
        try:
            return bool(self._room.isconnected())
        except Exception:  # noqa: BLE001
            return False

    def disconnect(self) -> None:
        try:
            if self._room is not None:
                self._run_coro(self._async_disconnect(), timeout=10.0)
        except Exception:  # noqa: BLE001 — disconnect is best effort
            pass
        finally:
            self._stop_loop()


# ── the session ────────────────────────────────────────────────────────────


@dataclass
class LiveCallReport:
    session_id: str
    room: str = ""
    turns: int = 0
    barge_ins: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    end_reason: str = "completed"  # completed | user-exit | remote-left |
                                   # no-remote | stopped | error
    error: str = ""
    #: per turn: {"turn": n, "heard": str, "said": [sentences actually
    #: played], "barged": bool}. History records what was *played*, not
    #: what was generated — the local session's lesson, kept here.
    spoken: list[dict[str, Any]] = field(default_factory=list)

    def latency_stats(self) -> dict[str, float]:
        lats = sorted(self.latencies_ms)
        if not lats:
            return {"p50": 0.0, "p95": 0.0, "count": 0}

        def _pct(p: float) -> float:
            return lats[min(len(lats) - 1, int(p * len(lats)))]

        return {"p50": _pct(0.5), "p95": _pct(0.95),
                "count": len(lats)}

    def to_dict(self) -> dict[str, Any]:
        stats = self.latency_stats()
        return {
            "session_id": self.session_id,
            "room": self.room,
            "turns": self.turns,
            "barge_ins": self.barge_ins,
            "end_reason": self.end_reason,
            "error": self.error,
            "ear_to_ear_ms_p50": stats["p50"],
            "ear_to_ear_ms_p95": stats["p95"],
            "spoken": self.spoken,
        }

    def format(self) -> str:
        stats = self.latency_stats()
        bar = "─" * 46
        lines = [
            f"🎙 live call {self.session_id} (room {self.room or '?'})",
            bar,
            (f"  turns {self.turns}   barge-ins {self.barge_ins}   "
             f"ended: {self.end_reason}"),
            (f"  ear-to-ear latency  p50 {stats['p50']:.0f}ms  "
             f"p95 {stats['p95']:.0f}ms  (n={stats['count']})"),
        ]
        if self.error:
            lines.append(f"  ⚠ {self.error}")
        for turn in self.spoken:
            heard = (turn.get("heard") or "")[:70]
            said = turn.get("said") or []
            barged = "  ⏭ barged" if turn.get("barged") else ""
            lines.append(f"  ┌ turn {turn.get('turn')}{barged}")
            if heard:
                lines.append(f"  │ you: {heard}")
            for s in said[:4]:
                lines.append(f"  │ her: {s[:80]}")
            if len(said) > 4:
                lines.append(f"  │ … +{len(said) - 4} more")
            lines.append("  └")
        lines.append(bar)
        return "\n".join(lines)


class LiveKitCallSession:
    """One live voice call in a LiveKit room.

    States: ``idle → connecting → waiting → listening → thinking →
    speaking → listening …``, plus ``ended`` / ``error``. The shape is
    the local :class:`VoiceSession` loop carried over the network: the
    inbound audio track is the mic, the published audio track is the
    speaker, and barge-in works the same way — owner speech while Devon
    speaks stops her reply immediately, discards the unspoken remainder,
    and the interruption becomes the next turn's input.

    The constructor takes injected ``think`` / ``stt`` / ``tts``
    callables with the exact signatures the local session uses, so the
    whole loop is testable against a fake transport with no room, no
    network, and no model.
    """

    def __init__(
        self,
        *,
        think: Callable[[str], str],
        stt: Callable[[str], str],
        tts: Callable[[str, str], dict[str, Any]],
        transport: LiveKitTransport | None = None,
        config: LiveKitConfig | None = None,
        room: str = "",
        data_dir: str | Path = "",
        profile: str = "",
        silence_ms: int = 800,
        speech_cap_secs: int = 60,
        vad_kind: str = "auto",
        max_wait_remote_s: float = DEFAULT_MAX_WAIT_REMOTE_S,
        remote_silence_end_s: float = REMOTE_SILENCE_END_S,
        bus: Any = None,
    ) -> None:
        self.session_id = new_short_id()
        self.think = think
        self.stt = stt
        self.tts = tts
        self.transport = transport
        self.config = config or LiveKitConfig.from_env()
        self.room = (room or "").strip()
        self.data_dir = Path(data_dir) if data_dir else Path("voice_data")
        self.profile = profile
        self.silence_ms = silence_ms
        self.speech_cap_secs = speech_cap_secs
        self.vad_kind = vad_kind
        self.max_wait_remote_s = max_wait_remote_s
        self.remote_silence_end_s = remote_silence_end_s
        self._bus = global_bus if bus is None else bus
        self.state = "idle"
        self._stop = threading.Event()
        self._tmp = self.data_dir / "tmp"
        self._in16 = bytearray()   # downsampled inbound PCM buffer (FIFO:
                                   # pending audio, then barge audio, then
                                   # new arrivals — in arrival order)
        self._barge_in16 = bytearray()  # barge-in VAD buffer: accumulates
                                        # across drain calls so a 20 ms
                                        # network frame never has to make a
                                        # 30 ms VAD chunk on its own
        self._last_inbound_ts = 0.0

    # -- journal ---------------------------------------------------------
    def _journal(self, topic: str, data: dict[str, Any]) -> None:
        bus = self._bus
        if bus is False or bus is None:
            return
        try:
            bus.publish(Event(topic=topic, data={
                "session_id": self.session_id,
                "room": self.room,
                **data,
            }))
        except Exception as exc:  # noqa: BLE001 — journal never breaks a call
            _log.debug("livekit journal failed: %s", exc)

    def _set_state(self, state: str) -> None:
        _log.info("live call %s: %s → %s", self.session_id, self.state,
                  state)
        self.state = state

    def stop(self) -> None:
        """End the call from outside (tool end, owner gesture)."""
        self._stop.set()

    # -- main loop ---------------------------------------------------------
    def start(self, *, max_turns: int = 0) -> LiveCallReport:
        """Join the room and run the call. Raises LiveKitUnavailable
        when the call cannot start; returns a report for everything
        that ends after connecting."""
        report = LiveCallReport(session_id=self.session_id, room=self.room)
        if not self.room:
            raise LiveKitUnavailable(
                "no room named — pass --room <name>",
                missing="a room name")
        missing = self.config.missing()
        if missing:
            raise LiveKitUnavailable(
                "LiveKit is not configured: " + self.config.describe_missing(),
                missing=self.config.describe_missing())
        if self.transport is None:
            if not sdk_available():
                raise LiveKitUnavailable(
                    "the livekit SDK is not installed — run "
                    "`nm voice live provision` first",
                    missing="the livekit SDK")
            self.transport = RtcTransport()
        transport = self.transport
        token = self.config.token_for(self.room)

        self._set_state("connecting")
        try:
            transport.connect(self.config.url, token)
        except LiveKitUnavailable:
            self._set_state("error")
            raise
        self._journal("voice.livekit.session_started",
                      {"state": "connected"})
        self._tmp.mkdir(parents=True, exist_ok=True)

        try:
            self._set_state("waiting")
            waited = self._wait_for_remote()
            if waited == "stopped":
                report.end_reason = "stopped"
                return report
            if waited == "no-remote":
                report.end_reason = "no-remote"
                report.error = (
                    f"nobody joined room {self.room!r} within "
                    f"{self.max_wait_remote_s:.0f}s")
                return report

            vad = make_vad(self.vad_kind, silence_ms=self.silence_ms)
            self._calibrate(vad)
            while True:
                if self._stop.is_set():
                    report.end_reason = "stopped"
                    break
                if max_turns and report.turns >= max_turns:
                    report.end_reason = "completed"
                    break
                self._set_state("listening")
                heard = self._listen_utterance(vad)
                if heard is None:  # remote left
                    report.end_reason = "remote-left"
                    break
                pcm16, first_speech_ts = heard
                if not pcm16:
                    continue
                wav_path = self._store_utterance(pcm16, report.turns)

                self._set_state("thinking")
                transcript = (self.stt(wav_path) or "").strip()
                self._discard_utterance(wav_path)
                _log.info("live call %s heard: %r", self.session_id,
                          transcript[:120])
                if not transcript:
                    continue
                if transcript.lower().strip(" .!") in _END_WORDS:
                    report.end_reason = "user-exit"
                    break

                reply = self.think(transcript) or ""
                plan = speakify(reply, cap_secs=self.speech_cap_secs)

                self._set_state("speaking")
                outcome, played = self._speak(plan, first_speech_ts, report)
                report.turns += 1
                report.spoken.append({
                    "turn": report.turns,
                    "heard": transcript,
                    "said": played,
                    "barged": outcome != "done",
                })
                self._journal("voice.livekit.turn", {
                    "turn": report.turns,
                    "heard_chars": len(transcript),
                    "said_sentences": len(played),
                    "barged": outcome != "done",
                })
                if outcome == "barged":
                    report.barge_ins += 1
                    self._journal("voice.livekit.barge_in",
                                  {"turn": report.turns})
                    continue
        except KeyboardInterrupt:  # noqa: E106 - deliberate
            report.end_reason = "stopped"
        except Exception as exc:  # noqa: BLE001 - a call must end cleanly
            report.end_reason = "error"
            report.error = f"{type(exc).__name__}: {exc}"
            _log.exception("live call %s failed", self.session_id)
        finally:
            self._set_state("ended")
            try:
                transport.disconnect()
            except Exception as exc:  # noqa: BLE001 — best effort
                record_failure("livekit_call.LiveKitCallSession.start",
                               exc)
            self._journal("voice.livekit.session_ended", {
                "turns": report.turns,
                "barge_ins": report.barge_ins,
                "end_reason": report.end_reason,
                "error": report.error,
            })
            self._save_report(report)
        return report

    # -- wait + calibrate --------------------------------------------------
    def _wait_for_remote(self) -> str:
        """Wait for a remote participant to publish audio.

        Returns "joined" | "no-remote" | "stopped".
        """
        deadline = time.time() + max(0.0, self.max_wait_remote_s)
        while time.time() < deadline:
            if self._stop.is_set():
                return "stopped"
            if self.transport is None:
                return "no-remote"
            if self.transport.remote_audio_publishers():
                return "joined"
            if not self.transport.is_connected():
                raise LiveKitUnavailable(
                    "lost the LiveKit room while waiting for someone "
                    "to join",
                    missing="a LiveKit room connection")
            time.sleep(0.2)
        if self.transport is not None \
                and self.transport.remote_audio_publishers():
            return "joined"
        return "stopped" if self._stop.is_set() else "no-remote"

    def _calibrate(self, vad: Any) -> None:
        """Learn the remote room tone from ~0.5s of inbound audio."""
        if self.transport is None:
            return
        energies: list[float] = []
        deadline = time.time() + 2.0
        while len(energies) < 16 and time.time() < deadline:
            chunk = self.transport.poll_inbound(timeout=0.2)
            if chunk is None:
                continue
            self._last_inbound_ts = time.time()
            pcm16 = resample_pcm(chunk, LIVEKIT_SAMPLE_RATE, SAMPLE_RATE)
            self._in16.extend(pcm16)
            while len(self._in16) >= CHUNK_BYTES:
                head = bytes(self._in16[:CHUNK_BYTES])
                del self._in16[:CHUNK_BYTES]
                energies.append(_rms16(head))
        if energies:
            energies.sort()
            ambient = energies[len(energies) // 4]
            vad.noise_floor = max(200.0, ambient * 2.0)
            vad.reset()
            _log.info("live call %s: calibrated noise floor %.0f",
                      self.session_id, vad.noise_floor)

    # -- listen --------------------------------------------------------------
    def _next_16k_chunk(self, timeout: float = 0.2) -> bytes | None:
        """One 30 ms 16 kHz chunk from the inbound stream.

        None on timeout (no audio yet) — the caller distinguishes a
        quiet room from a dead one by the clock, not by the chunk.
        """
        if self.transport is None:
            return None
        while len(self._in16) < CHUNK_BYTES:
            chunk = self.transport.poll_inbound(timeout=timeout)
            if chunk is None:
                return None
            self._last_inbound_ts = time.time()
            self._in16.extend(
                resample_pcm(chunk, LIVEKIT_SAMPLE_RATE, SAMPLE_RATE))
        head = bytes(self._in16[:CHUNK_BYTES])
        del self._in16[:CHUNK_BYTES]
        return head

    def _listen_utterance(self, vad: Any
                          ) -> tuple[bytes, float] | None:
        """Capture one utterance: speech … then `silence_ms` of quiet.

        Returns (pcm16, first_speech_ts) or None when the remote side
        has gone (no audio publisher and no frames for
        ``remote_silence_end_s``). Audio that arrived during the
        previous reply — drained into the inbound FIFO by ``_speak`` —
        is consumed here in arrival order, so an interruption is never
        lost and never double-counted.
        """
        from collections import deque

        chunks: list[bytes] = []
        prebuffer: deque[bytes] = deque(maxlen=10)  # ~300 ms of onset
        first_speech_ts = 0.0
        in_speech = False
        idle_chunks = 0
        last_speech_ts = time.time()
        silence_s = max(0.1, self.silence_ms / 1000.0)
        while True:
            if self._stop.is_set():
                return b"", 0.0
            chunk = self._next_16k_chunk(timeout=0.2)
            if chunk is None:
                # No frames: quiet, or the remote side is gone. Silence
                # is measured in SECONDS here — a timeout is 0.2 s, not
                # a 30 ms chunk, so counting timeouts against the VAD's
                # chunk budget would end utterances ~7x late.
                if (self.transport is not None
                        and not self.transport.remote_audio_publishers()
                        and time.time() - self._last_inbound_ts
                        > self.remote_silence_end_s):
                    return None
                if in_speech and time.time() - last_speech_ts > silence_s:
                    return b"".join(chunks), first_speech_ts
                continue
            if not in_speech:
                prebuffer.append(chunk)
            status = vad.observe(chunk)
            if status == "speech":
                if not in_speech:
                    in_speech = True
                    first_speech_ts = time.time()
                    chunks.extend(prebuffer)
                    prebuffer.clear()
                idle_chunks = 0
                last_speech_ts = time.time()
                chunks.append(chunk)
            elif in_speech:
                chunks.append(chunk)
                idle_chunks += 1
                if idle_chunks >= vad.silence_chunks:
                    pcm = b"".join(chunks[:len(chunks) - idle_chunks])
                    return pcm, first_speech_ts
            # else: pre-speech silence — only the prebuffer keeps it

    # -- speak (with barge-in) -------------------------------------------------
    def _speak(self, plan: Any, first_speech_ts: float,
               report: LiveCallReport) -> tuple[str, list[str]]:
        """Publish the reply sentence by sentence at 48 kHz.

        Returns (outcome, played_sentences): 'done' | 'barged' |
        'stopped'. While publishing, inbound audio is watched with a
        dedicated VAD: owner speech stops the publish immediately
        (``clear_outgoing`` drops the queued remainder), the
        interruption audio is kept as the next turn's prefix, and the
        outcome is 'barged'.
        """
        try:
            from ..media.sentence import segment_text

            sentences = segment_text(plan.spoken)
        except Exception:
            # A live call never dies on a utility import: the old
            # regex split is the fallback, not the engine.
            sentences = [s.strip() for s in
                         re.split(r"(?<=[.!?…])\s+", plan.spoken)
                         if s.strip()]
        if not sentences:
            return "done", []
        # Audio that arrived BEFORE this reply is the next turn's input,
        # not an interruption of it: drain it into the inbound FIFO now,
        # so the barge-in VAD below only ever sees audio that arrives
        # WHILE Devon is speaking.
        self._drain_inbound()
        barge_vad = make_vad(self.vad_kind, silence_ms=10_000,
                             min_speech_ms=200)
        played: list[str] = []
        speech_started = False
        for sentence in sentences:
            if self._stop.is_set():
                return "stopped", played
            try:
                info = self.tts(sentence, self.profile)
                wav_path = info.get("path", "")
            except Exception as exc:  # noqa: BLE001 - TTS hiccup ≠ dead air
                _log.warning("live call TTS failed: %s", exc)
                continue
            if not wav_path or not Path(wav_path).exists():
                continue
            pcm, rate = read_wav_bytes(wav_path)
            pcm48 = resample_pcm(pcm, rate, LIVEKIT_SAMPLE_RATE)
            frames = _pcm_chunks(pcm48, FRAME_BYTES_48K)
            barged = False
            for frame in frames:
                if self._stop.is_set():
                    return "stopped", played
                if self._barge_check(barge_vad):
                    barged = True
                    break
                if self.transport is None:
                    return "stopped", played
                try:
                    self.transport.publish_frame(frame)
                except Exception as exc:  # noqa: BLE001 — publish fails end the turn
                    _log.warning("live call publish failed: %s", exc)
                    return "stopped", played
                if not speech_started:
                    speech_started = True
                    self._record_latency(first_speech_ts, report)
            if barged:
                # stop the audio NOW and drop the queued remainder: the
                # owner is talking, and the unspoken tail must never
                # leak out after the interruption.
                if self.transport is not None:
                    self.transport.clear_outgoing()
                return "barged", played
            played.append(sentence)
        return "done", played

    def _drain_inbound(self) -> None:
        """Move all currently-queued inbound audio into the 16 kHz FIFO,
        downsampled. Called at the start of a reply so stale audio is
        pending input, never a barge-in candidate."""
        if self.transport is None:
            return
        while True:
            chunk = self.transport.poll_inbound(timeout=0.0)
            if chunk is None:
                return
            self._last_inbound_ts = time.time()
            self._in16.extend(
                resample_pcm(chunk, LIVEKIT_SAMPLE_RATE, SAMPLE_RATE))

    def _barge_check(self, barge_vad: Any) -> bool:
        """Drain newly arrived inbound audio; True when the owner is
        speaking over Devon.

        Only audio that arrives during this call is VAD-observed: the
        bytes are appended to the inbound FIFO (the next turn's input)
        and to a persistent observation buffer, which is observed in
        whole 30 ms chunks as they complete — a 20 ms network frame
        can never make a VAD chunk on its own, so the remainder carries
        across calls instead of being silently dropped.
        """
        if self.transport is None:
            return False
        speaking = False
        while True:
            chunk = self.transport.poll_inbound(timeout=0.0)
            if chunk is None:
                break
            self._last_inbound_ts = time.time()
            pcm16 = resample_pcm(chunk, LIVEKIT_SAMPLE_RATE, SAMPLE_RATE)
            self._in16.extend(pcm16)
            self._barge_in16.extend(pcm16)
            while len(self._barge_in16) >= CHUNK_BYTES:
                head = bytes(self._barge_in16[:CHUNK_BYTES])
                del self._barge_in16[:CHUNK_BYTES]
                if barge_vad.observe(head) == "speech":
                    speaking = True
        return speaking

    # -- utterance files -----------------------------------------------------
    def _store_utterance(self, pcm16: bytes, turn: int) -> str:
        path = str(self._tmp / f"live-{self.session_id}-{turn}.wav")
        write_wav_bytes(path, pcm16, sample_rate=SAMPLE_RATE)
        return path

    def _discard_utterance(self, wav_path: str) -> None:
        with contextlib.suppress(Exception):  # noqa: BLE001 — best effort
            Path(wav_path).unlink(missing_ok=True)

    def _record_latency(self, first_speech_ts: float,
                        report: LiveCallReport) -> None:
        if first_speech_ts:
            report.latencies_ms.append(
                (time.time() - first_speech_ts) * 1000.0)

    def _save_report(self, report: LiveCallReport) -> None:
        """Last-call record, for `nm voice live status`."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / "live-last.json"
            path.write_text(json.dumps(report.to_dict(), indent=2))
        except Exception:  # noqa: BLE001 — a report must never break a call
            _log.debug("live call report write failed", exc_info=True)


def _rms16(chunk: bytes) -> float:
    """RMS of one int16 PCM chunk (stdlib)."""
    import math as _math

    n = len(chunk) // 2
    if n == 0:
        return 0.0
    total = 0
    for i in range(n):
        sample = int.from_bytes(chunk[i * 2:(i + 1) * 2], "little",
                                signed=True)
        total += sample * sample
    return _math.sqrt(total / n)


# ── readiness + the in-process session registry ────────────────────────────


def livekit_status(data_dir: str | Path = "") -> dict[str, Any]:
    """Readiness for a live call: SDK, config, last call. No connections."""
    config = LiveKitConfig.from_env()
    missing = config.missing()
    out: dict[str, Any] = {
        "sdk_installed": sdk_available(),
        "sdk_version": sdk_version(),
        "sdk_pin": LIVEKIT_SDK_PIN,
        "api_pin": LIVEKIT_API_PIN,
        "configured": not missing,
        "missing": missing,
        "url_configured": bool(config.url),
        "token_configured": bool(config.token),
        "api_key_configured": bool(config.api_key),
        "live_sessions": len(_ACTIVE),
    }
    if data_dir:
        last = Path(data_dir) / "live-last.json"
        try:
            if last.exists():
                out["last_call"] = json.loads(last.read_text())
        except Exception:  # noqa: BLE001 — a bad record is not a failure
            out["last_call"] = {"error": "unreadable last-call record"}
    return out


_ACTIVE: dict[str, LiveKitCallSession] = {}
_ACTIVE_THREADS: dict[str, threading.Thread] = {}
_ACTIVE_REPORTS: dict[str, LiveCallReport] = {}
_ACTIVE_LOCK = threading.Lock()


def start_live_call(session: LiveKitCallSession, *,
                    max_turns: int = 0) -> LiveKitCallSession:
    """Run a session on a daemon thread; returns immediately.

    One live call at a time — the brain cannot be in two calls at
    once, and a second start refuses honestly instead of splitting her
    voice. The caller ends it with :func:`end_live_call`.
    """
    with _ACTIVE_LOCK:
        if _ACTIVE:
            raise LiveKitUnavailable(
                "a live call is already running "
                f"(session {next(iter(_ACTIVE))}) — end it first",
                missing="a free voice")
        _ACTIVE[session.session_id] = session

    def _run() -> None:
        try:
            report = session.start(max_turns=max_turns)
        except LiveKitUnavailable as exc:
            report = LiveCallReport(session_id=session.session_id,
                                    room=session.room,
                                    end_reason="error",
                                    error=str(exc))
        except Exception as exc:  # noqa: BLE001 — a dead thread still reports
            report = LiveCallReport(session_id=session.session_id,
                                    room=session.room,
                                    end_reason="error",
                                    error=f"{type(exc).__name__}: {exc}")
        with _ACTIVE_LOCK:
            _ACTIVE.pop(session.session_id, None)
            _ACTIVE_THREADS.pop(session.session_id, None)
            _ACTIVE_REPORTS[session.session_id] = report

    thread = threading.Thread(target=_run, daemon=True,
                              name=f"livekit-call-{session.session_id}")
    with _ACTIVE_LOCK:
        _ACTIVE_THREADS[session.session_id] = thread
    thread.start()
    return session


def get_live_call(session_id: str = "") -> LiveKitCallSession | None:
    """The running session (the only one, or by id)."""
    with _ACTIVE_LOCK:
        if session_id:
            return _ACTIVE.get(session_id)
        return next(iter(_ACTIVE.values()), None)


def end_live_call(session_id: str = "") -> LiveCallReport | None:
    """Stop the running session and wait for its report (≤ 15 s).

    Returns None when no session is running. The report is the honest
    record of what happened — turns, barge-ins, what was actually
    played, and why it ended.
    """
    with _ACTIVE_LOCK:
        session = (_ACTIVE.get(session_id) if session_id
                   else next(iter(_ACTIVE.values()), None))
        thread = (_ACTIVE_THREADS.get(session.session_id)
                  if session is not None else None)
    if session is None:
        with _ACTIVE_LOCK:
            if session_id and session_id in _ACTIVE_REPORTS:
                return _ACTIVE_REPORTS.pop(session_id)
            if not session_id and _ACTIVE_REPORTS:
                _key = next(iter(_ACTIVE_REPORTS))
                return _ACTIVE_REPORTS.pop(_key)
        return None
    session.stop()
    if thread is not None:
        thread.join(timeout=15.0)
    with _ACTIVE_LOCK:
        return _ACTIVE_REPORTS.pop(session.session_id, None)


def live_call_status(session_id: str = "") -> dict[str, Any]:
    """Readiness + the running session's state, if any. No connections."""
    out = livekit_status()
    session = get_live_call(session_id)
    if session is not None:
        out["session"] = {
            "session_id": session.session_id,
            "room": session.room,
            "state": session.state,
        }
    else:
        with _ACTIVE_LOCK:
            if _ACTIVE_REPORTS:
                _key = (session_id if session_id in _ACTIVE_REPORTS
                        else next(iter(_ACTIVE_REPORTS)))
                out["last_report"] = _ACTIVE_REPORTS[_key].to_dict()
    return out


# ── video frame sources ────────────────────────────────────────────────────
#
# A frame source is ``(elapsed_seconds) -> (rgb24, width, height) | None``:
# one RGB24 frame at the session size, or None when the source is done
# (the session then ends as "source-ended"). The session owns the
# pacing; the source owns the pixels.


#: A frame source: elapsed seconds in -> one RGB24 frame or None.
FrameSource = Callable[[float], tuple[bytes, int, int] | None]


def title_frame_source(room: str, *, width: int = VIDEO_DEFAULT_SIZE[0],
                       height: int = VIDEO_DEFAULT_SIZE[1]) -> FrameSource:
    """A live title card: Devon's mark, the room, elapsed time.

    The card is drawn with PIL, once per whole second — the clock is
    the elapsed time since the source started, so the frame is a pure
    function of ``t`` (deterministic, testable) and still honest on a
    real call. The room name is on the card, so two rooms never show
    the same picture.
    """
    from PIL import Image, ImageDraw, ImageFont

    room = (room or "").strip() or "?"
    state: dict[str, Any] = {"second": -1, "frame": None}

    def _source(t: float) -> tuple[bytes, int, int] | None:
        whole = int(t)
        if whole == state["second"] and state["frame"] is not None:
            return state["frame"]
        img = Image.new("RGB", (width, height), (10, 10, 14))
        d = ImageDraw.Draw(img)
        font_big = ImageFont.load_default(size=max(24, height // 6))
        font = ImageFont.load_default(size=max(14, height // 12))
        d.rectangle([0, height - 6, width, height], fill=(180, 40, 40))
        d.text((width // 20, height // 8), "DEVON", font=font_big,
               fill=(235, 235, 240))
        d.text((width // 20, height // 8 + height // 5), room,
               font=font, fill=(160, 200, 255))
        mins, secs = divmod(whole, 60)
        hours, mins = divmod(mins, 60)
        clock = f"{hours:02d}:{mins:02d}:{secs:02d}"
        d.text((width // 20, height - height // 5), clock, font=font,
               fill=(235, 235, 240))
        d.text((width // 2, height - height // 5), "● LIVE — video",
               font=font, fill=(255, 120, 120))
        frame = (img.tobytes(), width, height)
        state["second"] = whole
        state["frame"] = frame
        return frame

    return _source


def image_frame_source(path: str | Path, *,
                       width: int = VIDEO_DEFAULT_SIZE[0],
                       height: int = VIDEO_DEFAULT_SIZE[1]) -> FrameSource:
    """A still image, letterboxed to the session size, held as video.

    The picture is loaded and resized ONCE — every call returns the
    same bytes. A missing or unreadable file refuses at build time,
    naming the path, instead of joining a room with no picture.
    """
    from PIL import Image, ImageOps

    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        raise LiveKitUnavailable(
            f"video image not readable: {path}: {exc}",
            missing=f"an image file at {path}") from exc
    frame = (ImageOps.pad(img, (width, height),
                          color=(0, 0, 0)).tobytes(), width, height)

    def _source(t: float) -> tuple[bytes, int, int] | None:  # noqa: ARG001
        return frame

    return _source


def scene_frame_source(design: dict[str, Any], *,
                       width: int = VIDEO_DEFAULT_SIZE[0],
                       height: int = VIDEO_DEFAULT_SIZE[1],
                       duration: float = 30.0, text: str = "",
                       loop: bool = True) -> FrameSource:
    """Devon's own scene, rendered live into the room.

    The motion-studio frame function seeks the design to ``t`` — the
    same deterministic frames the code-video rung stitches into MP4s,
    published here as a live track instead. By default the scene
    loops until the call ends; with ``loop=False`` it ends at
    ``duration`` and the session ends as "source-ended". An invalid
    design or duration refuses at build time, naming the problem.
    """
    from ..media.motion_studio._core import MotionStudioError
    from ..media.motion_studio.explainers import (
        frame_image,
        validate_design,
    )

    try:
        validate_design(design)
    except MotionStudioError as exc:
        raise LiveKitUnavailable(
            f"video scene is not valid: {exc}",
            missing="a valid motion-studio design") from exc
    if duration <= 0:
        raise LiveKitUnavailable(
            "video scene needs a positive duration",
            missing="a scene duration in seconds")

    def _source(t: float) -> tuple[bytes, int, int] | None:
        if t >= duration:
            if not loop:
                return None
            t = t % duration
        img = frame_image(design, (width, height), float(t),
                          float(duration), text=text)
        return (img.convert("RGB").tobytes(), width, height)

    return _source


def rig_frame_source(rig_path: str | Path | None,
                     take_path: str | Path, *, width: int = 640,
                     height: int = 360, text: str = "",
                     remap_dialect: str = "bvh",
                     loop: bool = False) -> FrameSource:
    """A VTuber rig, driven live into the room.

    The rig (``rig_path``, or the bundled default when empty) is
    driven by a BVH take (``take_path``) remapped through the named
    dialect, and the posed frames render as a 2D puppet — the same
    frames the rig renders to PNG offline, published here as a
    live track instead. With ``text``, the visemes follow the text
    on uniform timing. A missing rig or take, or a take with no
    frames, refuses at build time, naming the path, instead of
    joining a room with no picture.
    """
    from ..media.rig import (
        RigSession,
        default_remap,
        load_take_bvh,
    )
    from ..media.rig.errors import RigError
    from ..media.rig.rig import RigDefinition, default_rig

    try:
        rig = (RigDefinition.from_json(rig_path)
               if rig_path else default_rig())
        remap = default_remap(remap_dialect)
        take = load_take_bvh(take_path)
    except RigError as exc:
        raise LiveKitUnavailable(
            f"video rig is not usable: {exc}",
            missing=(f"a rig file at {rig_path} and a BVH take at "
                     f"{take_path}")) from exc
    if not take["frames"]:
        raise LiveKitUnavailable(
            f"video rig take at {take_path} has no frames",
            missing=f"a BVH take with frames at {take_path}")
    session = RigSession(rig, remap, width=width, height=height,
                         fps=15.0)
    return session.frame_source(
        take["skeleton"], take["frames"],
        text=(text or ""), loop=loop)


# ── the video session ──────────────────────────────────────────────────────


@dataclass
class LiveVideoReport:
    session_id: str
    room: str = ""
    source: str = ""
    frames_published: int = 0
    frames_received: dict[str, int] = field(default_factory=dict)
    snapshots: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0
    end_reason: str = "completed"  # completed | stopped | source-ended |
                                   # error
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "room": self.room,
            "source": self.source,
            "frames_published": self.frames_published,
            "frames_received": dict(self.frames_received),
            "snapshots": dict(self.snapshots),
            "duration_s": round(self.duration_s, 2),
            "end_reason": self.end_reason,
            "error": self.error,
        }

    def format(self) -> str:
        bar = "─" * 46
        lines = [
            f"📹 live video {self.session_id} (room {self.room or '?'})",
            bar,
            (f"  source {self.source or '?'}   frames published "
             f"{self.frames_published}   {self.duration_s:.1f}s   "
             f"ended: {self.end_reason}"),
        ]
        if self.error:
            lines.append(f"  ⚠ {self.error}")
        for identity, count in sorted(self.frames_received.items()):
            snap = self.snapshots.get(identity, "")
            lines.append(f"  👁 {identity}: {count} frames received"
                         + (f" — snapshot {snap}" if snap else ""))
        lines.append(bar)
        return "\n".join(lines)


class LiveKitVideoSession:
    """One live video session in a LiveKit room.

    States: ``idle → connecting → publishing → ended`` / ``error``.
    The loop is the video mirror of the voice call: the frame source
    is the camera, the published track is the picture, and the room's
    video is watched at the same time — every publisher's latest
    frame lands as a PNG snapshot (at most one per second each) so
    the owner can see who is on camera without sitting in the room.
    Nobody has to publish video back: the remote side may only watch,
    so the session never waits for a remote publisher — it publishes
    until stopped, the source ends, or the duration runs out.

    The constructor takes an injected ``frame_source`` with the exact
    ``(t) -> (rgb, w, h) | None`` contract above, so the whole loop
    is testable against a fake transport with no room, no network,
    and no camera.
    """

    def __init__(
        self,
        *,
        frame_source: FrameSource | None = None,
        transport: LiveKitVideoTransport | None = None,
        config: LiveKitConfig | None = None,
        room: str = "",
        data_dir: str | Path = "",
        width: int = VIDEO_DEFAULT_SIZE[0],
        height: int = VIDEO_DEFAULT_SIZE[1],
        fps: float = VIDEO_DEFAULT_FPS,
        snapshot_every_s: float = VIDEO_SNAPSHOT_EVERY_S,
        source_name: str = "",
        bus: Any = None,
    ) -> None:
        self.session_id = new_short_id()
        self.frame_source = frame_source
        self.transport = transport
        self.config = config or LiveKitConfig.from_env()
        self.room = (room or "").strip()
        self.data_dir = Path(data_dir) if data_dir else Path("voice_data")
        self.width = width
        self.height = height
        self.fps = fps
        self.snapshot_every_s = snapshot_every_s
        self.source_name = source_name
        self._bus = global_bus if bus is None else bus
        self.state = "idle"
        self._stop = threading.Event()
        self._frames_published = 0
        self._frames_received: dict[str, int] = {}
        self._snapshots: dict[str, str] = {}
        self._last_snapshot_ts: dict[str, float] = {}

    # -- journal ---------------------------------------------------------
    def _journal(self, topic: str, data: dict[str, Any]) -> None:
        bus = self._bus
        if bus is False or bus is None:
            return
        try:
            bus.publish(Event(topic=topic, data={
                "session_id": self.session_id,
                "room": self.room,
                **data,
            }))
        except Exception as exc:  # noqa: BLE001 — journal never breaks a call
            _log.debug("livekit video journal failed: %s", exc)

    def _set_state(self, state: str) -> None:
        _log.info("live video %s: %s → %s", self.session_id, self.state,
                  state)
        self.state = state

    def stop(self) -> None:
        """End the video session from outside (tool end, owner gesture)."""
        self._stop.set()

    def snapshot(self, identity: str = "") -> dict[str, Any] | None:
        """The latest snapshot for one publisher (or the only one).

        Returns {"identity", "path", "frames_received", "age_s"} or
        None when that publisher has sent no frame yet.
        """
        if identity:
            path = self._snapshots.get(identity)
            if path is None:
                return None
        elif len(self._snapshots) == 1:
            identity, path = next(iter(self._snapshots.items()))
        else:
            return None
        try:
            age_s = time.time() - Path(path).stat().st_mtime
        except OSError:
            age_s = -1.0
        return {
            "identity": identity,
            "path": path,
            "frames_received": self._frames_received.get(identity, 0),
            "age_s": round(age_s, 1),
        }

    # -- main loop ---------------------------------------------------------
    def start(self, *, max_duration_s: float = 0.0) -> LiveVideoReport:
        """Join the room and publish video. Raises LiveKitUnavailable
        when the session cannot start; returns a report for everything
        that ends after connecting."""
        report = LiveVideoReport(session_id=self.session_id,
                                 room=self.room,
                                 source=self.source_name)
        if not self.room:
            raise LiveKitUnavailable(
                "no room named — pass --room <name>",
                missing="a room name")
        missing = self.config.missing()
        if missing:
            raise LiveKitUnavailable(
                "LiveKit is not configured: "
                + self.config.describe_missing(),
                missing=self.config.describe_missing())
        if self.frame_source is None:
            raise LiveKitUnavailable(
                "no video frame source — pass a title, image, or scene",
                missing="a frame source")
        if self.transport is None:
            if not sdk_available():
                raise LiveKitUnavailable(
                    "the livekit SDK is not installed — run "
                    "`nm voice live provision` first",
                    missing="the livekit SDK")
            self.transport = RtcTransport(video=True,
                                          video_width=self.width,
                                          video_height=self.height)
        transport = self.transport
        token = self.config.token_for(self.room, identity=VIDEO_IDENTITY)

        self._set_state("connecting")
        try:
            transport.connect(self.config.url, token)
        except LiveKitUnavailable:
            self._set_state("error")
            raise
        self._journal("voice.livekit.video_started",
                      {"state": "connected",
                       "source": self.source_name})
        self.data_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        period = 1.0 / max(1.0, self.fps)
        next_t = t0
        try:
            self._set_state("publishing")
            while True:
                if self._stop.is_set():
                    report.end_reason = "stopped"
                    break
                elapsed = time.time() - t0
                if max_duration_s and elapsed >= max_duration_s:
                    report.end_reason = "completed"
                    break
                self._drain_inbound_video()
                frame = self.frame_source(elapsed)
                if frame is None:
                    report.end_reason = "source-ended"
                    break
                rgb, w, h = frame
                try:
                    transport.publish_video_frame(rgb, w, h)
                except Exception as exc:  # noqa: BLE001 — a bad frame ends it
                    report.end_reason = "error"
                    report.error = (f"video publish failed: "
                                    f"{type(exc).__name__}: {exc}")
                    break
                self._frames_published += 1
                next_t += period
                sleep_s = next_t - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    # the source is slower than the frame rate: never
                    # sleep a backlog, just re-time from now
                    next_t = time.time()
        except KeyboardInterrupt:  # noqa: E106 - deliberate
            report.end_reason = "stopped"
        except Exception as exc:  # noqa: BLE001 - a session must end cleanly
            report.end_reason = "error"
            report.error = f"{type(exc).__name__}: {exc}"
            _log.exception("live video %s failed", self.session_id)
        finally:
            report.duration_s = time.time() - t0
            report.frames_published = self._frames_published
            report.frames_received = dict(self._frames_received)
            report.snapshots = dict(self._snapshots)
            self._set_state("ended")
            try:
                transport.disconnect()
            except Exception as exc:  # noqa: BLE001 — best effort
                record_failure("livekit_call.LiveKitVideoSession.start",
                               exc)
            self._journal("voice.livekit.video_ended", {
                "frames_published": report.frames_published,
                "frames_received": report.frames_received,
                "end_reason": report.end_reason,
                "error": report.error,
            })
            self._save_report(report)
        return report

    # -- inbound video → snapshots -----------------------------------------
    def _drain_inbound_video(self) -> None:
        """Move queued inbound frames into counts + snapshots.

        At most one snapshot PNG per publisher per
        ``snapshot_every_s`` — the latest frame wins, the disk never
        fills with a frame per packet.
        """
        if self.transport is None:
            return
        now = time.time()
        while True:
            item = self.transport.poll_video_frame(timeout=0.0)
            if item is None:
                return
            rgb, w, h, identity = item
            self._frames_received[identity] = \
                self._frames_received.get(identity, 0) + 1
            last = self._last_snapshot_ts.get(identity, 0.0)
            if now - last < self.snapshot_every_s:
                continue
            self._last_snapshot_ts[identity] = now
            self._save_snapshot(identity, rgb, w, h)

    def _save_snapshot(self, identity: str, rgb: bytes, width: int,
                       height: int) -> None:
        """One inbound frame as a PNG the owner can open."""
        from PIL import Image

        safe = re.sub(r"[^A-Za-z0-9_-]", "_", identity) or "remote"
        path = (self.data_dir
                / f"live-video-{self.session_id}-{safe}.png")
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            Image.frombytes("RGB", (width, height), rgb).save(path)
        except Exception as exc:  # noqa: BLE001 — a bad frame never breaks a call
            record_failure("livekit_call.LiveKitVideoSession."
                           "_save_snapshot", exc)
            return
        self._snapshots[identity] = str(path)

    def _save_report(self, report: LiveVideoReport) -> None:
        """Last-video record, for `nm voice live video status`."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / "live-video-last.json"
            path.write_text(json.dumps(report.to_dict(), indent=2))
        except Exception:  # noqa: BLE001 — a report must never break a session
            _log.debug("live video report write failed", exc_info=True)


# ── readiness + the in-process video registry ──────────────────────────────


def live_video_status(data_dir: str | Path = "") -> dict[str, Any]:
    """Readiness for a live video session: SDK, config, last video.
    No connections."""
    config = LiveKitConfig.from_env()
    missing = config.missing()
    out: dict[str, Any] = {
        "sdk_installed": sdk_available(),
        "sdk_version": sdk_version(),
        "sdk_pin": LIVEKIT_SDK_PIN,
        "api_pin": LIVEKIT_API_PIN,
        "configured": not missing,
        "missing": missing,
        "url_configured": bool(config.url),
        "token_configured": bool(config.token),
        "api_key_configured": bool(config.api_key),
        "live_sessions": len(_ACTIVE_VIDEO),
    }
    if data_dir:
        last = Path(data_dir) / "live-video-last.json"
        try:
            if last.exists():
                out["last_video"] = json.loads(last.read_text())
        except Exception:  # noqa: BLE001 — a bad record is not a failure
            out["last_video"] = {"error": "unreadable last-video record"}
    return out


_ACTIVE_VIDEO: dict[str, LiveKitVideoSession] = {}
_ACTIVE_VIDEO_THREADS: dict[str, threading.Thread] = {}
_ACTIVE_VIDEO_REPORTS: dict[str, LiveVideoReport] = {}
_ACTIVE_VIDEO_LOCK = threading.Lock()


def start_video_session(session: LiveKitVideoSession, *,
                        max_duration_s: float = 0.0
                        ) -> LiveKitVideoSession:
    """Run a video session on a daemon thread; returns immediately.

    One live video session at a time — one video identity in the room,
    and a second start refuses honestly instead of splitting her
    picture. The caller ends it with :func:`end_video_session`.
    """
    with _ACTIVE_VIDEO_LOCK:
        if _ACTIVE_VIDEO:
            raise LiveKitUnavailable(
                "a live video session is already running "
                f"(session {next(iter(_ACTIVE_VIDEO))}) — end it first",
                missing="a free video")
        _ACTIVE_VIDEO[session.session_id] = session

    def _run() -> None:
        try:
            report = session.start(max_duration_s=max_duration_s)
        except LiveKitUnavailable as exc:
            report = LiveVideoReport(session_id=session.session_id,
                                     room=session.room,
                                     source=session.source_name,
                                     end_reason="error",
                                     error=str(exc))
        except Exception as exc:  # noqa: BLE001 — a dead thread still reports
            report = LiveVideoReport(session_id=session.session_id,
                                     room=session.room,
                                     source=session.source_name,
                                     end_reason="error",
                                     error=f"{type(exc).__name__}: {exc}")
        with _ACTIVE_VIDEO_LOCK:
            _ACTIVE_VIDEO.pop(session.session_id, None)
            _ACTIVE_VIDEO_THREADS.pop(session.session_id, None)
            _ACTIVE_VIDEO_REPORTS[session.session_id] = report

    thread = threading.Thread(target=_run, daemon=True,
                              name=f"livekit-video-{session.session_id}")
    with _ACTIVE_VIDEO_LOCK:
        _ACTIVE_VIDEO_THREADS[session.session_id] = thread
    thread.start()
    return session


def get_video_session(session_id: str = "") -> LiveKitVideoSession | None:
    """The running video session (the only one, or by id)."""
    with _ACTIVE_VIDEO_LOCK:
        if session_id:
            return _ACTIVE_VIDEO.get(session_id)
        return next(iter(_ACTIVE_VIDEO.values()), None)


def end_video_session(session_id: str = "") -> LiveVideoReport | None:
    """Stop the running video session and wait for its report (≤ 15 s).

    Returns None when no session is running. The report is the honest
    record of what happened — frames published, frames seen, snapshots,
    and why it ended.
    """
    with _ACTIVE_VIDEO_LOCK:
        session = (_ACTIVE_VIDEO.get(session_id) if session_id
                   else next(iter(_ACTIVE_VIDEO.values()), None))
        thread = (_ACTIVE_VIDEO_THREADS.get(session.session_id)
                  if session is not None else None)
    if session is None:
        with _ACTIVE_VIDEO_LOCK:
            if session_id and session_id in _ACTIVE_VIDEO_REPORTS:
                return _ACTIVE_VIDEO_REPORTS.pop(session_id)
            if not session_id and _ACTIVE_VIDEO_REPORTS:
                _key = next(iter(_ACTIVE_VIDEO_REPORTS))
                return _ACTIVE_VIDEO_REPORTS.pop(_key)
        return None
    session.stop()
    if thread is not None:
        thread.join(timeout=15.0)
    with _ACTIVE_VIDEO_LOCK:
        return _ACTIVE_VIDEO_REPORTS.pop(session.session_id, None)


def video_session_status(session_id: str = "") -> dict[str, Any]:
    """Readiness + the running video session's state, if any.
    No connections."""
    out = live_video_status()
    session = get_video_session(session_id)
    if session is not None:
        out["session"] = {
            "session_id": session.session_id,
            "room": session.room,
            "state": session.state,
            "source": session.source_name,
            "frames_published": session._frames_published,
            "frames_received": dict(session._frames_received),
            "snapshots": dict(session._snapshots),
        }
    else:
        with _ACTIVE_VIDEO_LOCK:
            if _ACTIVE_VIDEO_REPORTS:
                _key = (session_id if session_id in _ACTIVE_VIDEO_REPORTS
                        else next(iter(_ACTIVE_VIDEO_REPORTS)))
                out["last_report"] = _ACTIVE_VIDEO_REPORTS[_key].to_dict()
    return out
