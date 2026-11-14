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

Video is OUT of scope: the transport can publish video tracks, but
this rung publishes audio only. A video rung would be a separate build.

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
    "LIVEKIT_SAMPLE_RATE",
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


class RtcTransport:
    """The real transport: ``livekit.rtc`` Room + AudioSource/AudioStream.

    All ``livekit`` imports happen inside methods, so importing this
    module never requires the SDK — the session checks
    :func:`sdk_available` first and refuses honestly without it.
    """

    def __init__(self) -> None:
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

    def _on_track_subscribed(self, track: Any, publication: Any,
                             participant: Any) -> None:
        mime = str(getattr(publication, "mime_type", "") or "")
        if mime and not mime.startswith("audio/"):
            return  # video / data track — this rung is audio only
        identity = str(getattr(participant, "identity", "") or "remote")
        self._note_publisher(identity, True)
        loop = self._loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._pump_audio(track, identity), loop)

    def _on_track_unsubscribed(self, track: Any, publication: Any,
                               participant: Any) -> None:
        identity = str(getattr(participant, "identity", "") or "remote")
        self._note_publisher(identity, False)

    def _on_participant_disconnected(self, participant: Any) -> None:
        identity = str(getattr(participant, "identity", "") or "remote")
        self._note_publisher(identity, False)

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
        self._connected = True

    async def _async_disconnect(self) -> None:
        room, source = self._room, self._source
        self._room, self._source = None, None
        self._connected = False
        if source is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 — best effort
                await source.aclose()
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
            raise LiveKitUnavailable(
                f"could not join the LiveKit room at {url}: "
                f"{type(exc).__name__}: {exc}",
                missing=f"a reachable LiveKit server at {url}") from exc

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
        sentences = [s.strip() for s in
                     re.split(r"(?<=[.!?…])\s+", plan.spoken) if s.strip()]
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
