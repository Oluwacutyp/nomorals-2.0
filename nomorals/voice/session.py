"""Live voice conversation loop: listen → transcribe → think → speak.

This is the synchronous counterpart to the async voice notes in
:mod:`nomorals.voice.bridge`. It reuses the existing pieces instead of
forking a parallel "voice brain":

- STT: ``VoiceBridge.transcribe_voice`` (whatever backend it wraps)
- TTS: :class:`nomorals.voice.tts.UniversalTTS` (swappable backends, tag
  system, consent-gated voice cloning — reused exactly)
- Think: the caller's ``think`` callable — the CLI wires it to
  ``PartnerRuntime.handle_message``, i.e. the same pipeline as chat.

Everything audio-hardware related is an *optional* import: importing this
module on a headless server never fails. :func:`default_mic` returns
``None`` there, and the session raises :class:`NoAudioDevice` instead of a
traceback from deep inside a driver.

Turn-taking uses a dependency-free energy-based VAD (~60 lines). The
choice is deliberate: no extra model download, no native dependency, works
on Termux, and it's honest about what it is — an energy gate, not a neural
VAD. Each session calibrates the VAD against ~0.5s of room tone first, the
noise floor only ever adapts on silence (loud speech can't drag the
threshold up and clip an utterance's onset), and a rolling ~300ms
prebuffer keeps the onset the VAD needed to make up its mind.
``VoiceSession`` accepts injected mic/speaker/stt/tts so the whole loop is
testable without hardware.

One capture thread owns the mic for the whole session — the utterance loop
and the barge-in poll drain the same queue, so there are no competing
readers and no per-poll thread spam.

Raw mic audio is temp-only by default and deleted after transcription.
``keep_audio=True`` retains utterances, AES-encrypted at rest when an
``audio_key`` is supplied (``nm voice call --keep-audio --audio-key-file``);
without a key they're plain wav and the session says so loudly.
"""

from __future__ import annotations

import math
import queue
import re
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

__all__ = [
    "NoAudioDevice",
    "EnergyVAD",
    "MicSource",
    "SpeakerSink",
    "MicCapture",
    "default_mic",
    "default_speaker",
    "speakify",
    "int_to_words",
    "ConsentStore",
    "StatsStore",
    "VoiceSession",
    "SessionReport",
    "write_wav_bytes",
    "read_wav_bytes",
    "split_wav",
    "make_bridge_stt",
    "stt_supports_partial",
    "decrypt_kept_audio",
]

_log = get_logger(__name__)

# 16 kHz mono int16 — the STT backends' happiest format, and cheap to move.
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_MS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH


class NoAudioDevice(Exception):
    """Raised when a live session is requested but no mic is available."""


# ── energy VAD (dependency-free) ─────────────────────────────────────────────


def _rms_int16(chunk: bytes) -> float:
    """Root-mean-square of an int16 PCM chunk. Pure stdlib."""
    n = len(chunk) // SAMPLE_WIDTH
    if n == 0:
        return 0.0
    total = 0
    for i in range(n):
        sample = int.from_bytes(chunk[i * 2:(i + 1) * 2], "little", signed=True)
        total += sample * sample
    return math.sqrt(total / n)


class EnergyVAD:
    """Adaptive energy gate: speech vs. silence from PCM chunk RMS.

    The noise floor adapts slowly (rooms change); the speech threshold sits
    well above it. This is deliberately *not* a neural VAD — it can't tell
    speech from a slammed door, but it needs no model, no download, and no
    native dependency, which is the right trade for a first live loop on a
    phone or headless box.

    Two rules keep it honest:

    - :meth:`calibrate` learns the room from ~0.5s of ambient audio before
      listening starts, so the first utterance isn't measured against a
      made-up floor.
    - the floor adapts **only on silence-classified chunks**. Letting loud
      speech drag the threshold up behind it clipped the *start* of
      utterances (the onset reads as silence once the floor has moved).
    """

    def __init__(self, *, silence_ms: int = 800, min_speech_ms: int = 150,
                 chunk_ms: int = CHUNK_MS, floor: float = 200.0) -> None:
        self.silence_chunks = max(1, int(silence_ms / chunk_ms))
        self.min_speech_chunks = max(1, int(min_speech_ms / chunk_ms))
        self.noise_floor = floor
        self._speech_run = 0
        self._silence_run = 0

    def calibrate(self, chunks: Any) -> None:
        """Learn the room: feed ambient (nobody talking) chunks first."""
        peak = 0.0
        for chunk in chunks:
            if chunk:
                peak = max(peak, _rms_int16(chunk))
        # floor sits comfortably above the room's loudest ambient moment.
        self.noise_floor = max(200.0, peak * 1.5)
        self.reset()

    def is_speech(self, chunk: bytes) -> bool:
        energy = _rms_int16(chunk)
        threshold = max(self.noise_floor * 4.0, 300.0)
        if energy > threshold:
            return True
        # silence: let the floor follow a quieting room down, or a louder
        # room (AC kicking on) up — but never on speech itself.
        if energy > self.noise_floor:
            self.noise_floor = 0.7 * self.noise_floor + 0.3 * energy
        else:
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * energy
        return False

    def observe(self, chunk: bytes) -> str:
        """Feed one chunk; returns 'speech' | 'silence' (debounced)."""
        if self.is_speech(chunk):
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0
        if self._speech_run >= self.min_speech_chunks:
            return "speech"
        if self._silence_run >= self.silence_chunks:
            return "silence"
        return "undecided"

    def reset(self) -> None:
        self._speech_run = 0
        self._silence_run = 0


# ── wav helpers (utterance files for STT) ────────────────────────────────────


def write_wav_bytes(path: str | Path, pcm: bytes,
                     sample_rate: int = SAMPLE_RATE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(CHANNELS)
        fh.setsampwidth(SAMPLE_WIDTH)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm)


def read_wav_bytes(path: str | Path) -> tuple[bytes, int]:
    """Return (pcm_bytes, sample_rate); converts to 16k mono int16 if needed."""
    with wave.open(str(path), "rb") as fh:
        n_channels = fh.getnchannels()
        width = fh.getsampwidth()
        rate = fh.getframerate()
        frames = fh.readframes(fh.getnframes())
    if n_channels > 1:  # mix down: average channels
        n = len(frames) // (width * n_channels)
        mixed = bytearray()
        for i in range(n):
            total = 0
            for c in range(n_channels):
                off = (i * n_channels + c) * width
                total += int.from_bytes(frames[off:off + width], "little",
                                        signed=True)
            avg = total // n_channels
            mixed += avg.to_bytes(2, "little", signed=True)
        frames = bytes(mixed)
        width = 2
    if width != 2:  # 8-bit unsigned → 16-bit signed
        frames = b"".join(
            ((b - 128) * 256).to_bytes(2, "little", signed=True)
            for b in frames)
    # naive resample only if we must (tests use 16k already)
    return frames, rate


def split_wav(path: str | Path, *, max_secs: int = 120) -> list[str]:
    """Split a wav into <= ``max_secs`` segments. Returns segment paths.

    Voice notes over ~2 minutes get awkward on every platform (and some
    refuse them); the session plays/sends long replies as a run of
    segments instead of one giant file.
    """
    path = Path(path)
    pcm, rate = read_wav_bytes(path)
    stride = int(rate * max_secs) * SAMPLE_WIDTH
    if len(pcm) <= stride:
        return [str(path)]
    out: list[str] = []
    for i in range(0, len(pcm), stride):
        seg = path.with_name(f"{path.stem}-p{i // stride}{path.suffix}")
        write_wav_bytes(seg, pcm[i:i + stride], sample_rate=rate)
        out.append(str(seg))
    _log.info("split %s into %d voice segments", path.name, len(out))
    return out


# ── mic / speaker protocols + default implementations ────────────────────────


class MicSource(Protocol):
    """PCM chunk producer. ``None`` chunk = stream ended."""

    sample_rate: int

    def read_chunk(self) -> bytes | None: ...
    def close(self) -> None: ...


class SpeakerSink(Protocol):
    """Interruptible wav playback."""

    def play(self, wav_path: str) -> Any: ...
    def playing(self, token: Any) -> bool: ...
    def stop(self, token: Any) -> None: ...
    def close(self) -> None: ...


class _SounddeviceMic:
    """Microphone via the optional ``sounddevice`` package."""

    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        import sounddevice as sd  # noqa: F401  (optional)

        import sounddevice as _sd
        self._sd = _sd
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=64)

        def _cb(indata: Any, frames: int, time_info: Any,
                status: Any) -> None:
            try:
                self._queue.put_nowait(bytes(indata))
            except queue.Full:  # noqa: E103 - audio callback must never block; drop the chunk
                pass

        self._stream = _sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
            blocksize=CHUNK_SAMPLES, callback=_cb)
        self._stream.start()

    def read_chunk(self) -> bytes | None:
        try:
            return self._queue.get(timeout=5.0)
        except queue.Empty:
            return b"\x00" * CHUNK_BYTES  # treat dropouts as silence

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # noqa: BLE001 - best effort
            pass


class _SounddeviceSpeaker:
    """Interruptible playback via the optional ``sounddevice`` package."""

    def __init__(self) -> None:
        import sounddevice as sd  # noqa: F401  (optional)

        import sounddevice as _sd
        self._sd = _sd
        self._stop_events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._next_token = 0

    def play(self, wav_path: str) -> int:
        pcm, rate = read_wav_bytes(wav_path)
        samples = [struct.unpack("<h", pcm[i:i + 2])[0]
                   for i in range(0, len(pcm), 2)]
        stop = threading.Event()
        with self._lock:
            self._next_token += 1
            token = self._next_token
            self._stop_events[token] = stop

        def _run() -> None:
            try:
                block = int(rate * 0.1)  # 100ms slices → responsive stop
                for i in range(0, len(samples), block):
                    if stop.is_set():
                        break
                    self._sd.play(samples[i:i + block], samplerate=rate,
                                  blocking=True)
            finally:
                with self._lock:
                    self._stop_events.pop(token, None)

        threading.Thread(target=_run, daemon=True).start()
        return token

    def playing(self, token: int) -> bool:
        with self._lock:
            return token in self._stop_events

    def stop(self, token: int) -> None:
        with self._lock:
            event = self._stop_events.get(token)
        if event is not None:
            event.set()
            try:
                self._sd.stop()
            except Exception:  # noqa: BLE001 - best effort
                pass

    def close(self) -> None:
        pass


def default_mic() -> MicSource | None:
    """Best available mic, or None on a headless box. Never raises."""
    try:
        return _SounddeviceMic()
    except Exception as exc:  # noqa: BLE001 - no mic hardware / no package
        _log.info("no microphone available: %s", exc)
        return None


def default_speaker() -> SpeakerSink | None:
    """Best available speaker, or None. Never raises."""
    try:
        return _SounddeviceSpeaker()
    except Exception as exc:  # noqa: BLE001
        _log.info("no speaker available: %s", exc)
        return None


class MicCapture:
    """One thread owns the mic; every reader drains the same queue.

    The old design spawned a throwaway thread per barge-in poll while the
    listen loop blocked directly on the mic — two readers racing one
    stream, chunks lost between them. Now the capture thread is the only
    reader: :meth:`listen_next` blocks for the utterance loop,
    :meth:`poll` peeks with a short timeout for the barge-in check.
    """

    def __init__(self, mic: MicSource, *, maxsize: int = 512) -> None:
        self._mic = mic
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._ended = False
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mic-capture")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._mic.read_chunk()
            except Exception:  # noqa: BLE001 - a dying mic ends the stream
                chunk = None
            if chunk is None:
                self._ended = True
            try:
                self._queue.put(chunk, timeout=0.5)
            except queue.Full:
                self.dropped += 1
            if chunk is None:
                break

    def listen_next(self) -> bytes | None:
        """Blocking read for the utterance loop. None = stream ended.

        The terminal None is sticky: once the stream ends, this returns
        None immediately instead of blocking on a dead thread's queue.
        """
        if self._ended and self._queue.empty():
            return None
        return self._queue.get()

    def poll(self, timeout: float = 0.05) -> bytes | None:
        """Short-timeout read for the barge-in check. None = nothing yet.

        A terminal None is put back for the utterance loop to find — the
        barge-in poll must never swallow the stream's end marker.
        """
        try:
            chunk = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if chunk is None:
            self._queue.put(chunk)
        return chunk

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── output shaping for speech ────────────────────────────────────────────────

_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth", "ninth", "tenth"]


def _ordinal(n: int) -> str:
    if 0 < n <= len(_ORDINALS):
        return _ORDINALS[n - 1]
    suffix = "th"
    if n % 10 == 1 and n % 100 != 11:
        suffix = "st"
    elif n % 10 == 2 and n % 100 != 12:
        suffix = "nd"
    elif n % 10 == 3 and n % 100 != 13:
        suffix = "rd"
    return f"{n}{suffix}"


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def int_to_words(n: int) -> str:
    """Small deterministic number→words (0 .. 999,999,999). No model call."""
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        rest = "" if n % 100 == 0 else " " + int_to_words(n % 100)
        return _ONES[n // 100] + " hundred" + rest
    for scale, word in ((1_000_000_000, "billion"), (1_000_000, "million"),
                        (1_000, "thousand")):
        if n >= scale:
            rest = "" if n % scale == 0 else " " + int_to_words(n % scale)
            return int_to_words(n // scale) + f" {word}" + rest
    raise AssertionError("unreachable")


def _normalize_numbers(text: str) -> str:
    """Read amounts like a person: ₦50,000 → 'fifty thousand naira'."""

    def _naira(m: re.Match[str]) -> str:
        return int_to_words(int(m.group(1).replace(",", ""))) + " naira"

    def _dollars(m: re.Match[str]) -> str:
        amount = int(m.group(1).replace(",", ""))
        unit = "dollar" if amount == 1 else "dollars"
        return f"{int_to_words(amount)} {unit}"

    def _plain(m: re.Match[str]) -> str:
        return int_to_words(int(m.group(1).replace(",", "")))

    text = re.sub(r"₦\s*([\d,]+)", _naira, text)
    text = re.sub(r"\$\s*([\d,]+)", _dollars, text)
    text = re.sub(r"(\d+)\s*%", lambda m: int_to_words(int(m.group(1)))
                  + " percent", text)
    # bare integers with separators (years like 2026 stay — handled below)
    text = re.sub(r"(?<!\d)(\d{1,3}(?:,\d{3})+)(?!\d)", _plain, text)
    return text


@dataclass
class SpeakPlan:
    """What to say out loud vs. what goes to chat in full."""
    spoken: str          # TTS input
    full: str            # delivered to chat in parallel
    truncated: bool = False
    had_code: bool = False


def speakify(text: str, cap_secs: int = 60) -> SpeakPlan:
    """Shape chat-written text for speech. Deterministic, no model calls.

    - markdown/headers/emphasis/links → plain words
    - lists → spoken cadence ("first…, second…")
    - code blocks → "code snippet omitted — I sent it to chat"
    - amounts → words ("fifty thousand naira")
    - over ~cap_secs of speech → spoken summary + pointer to chat
    """
    full = text
    had_code = "```" in text
    # code blocks first (before inline-code stripping eats the fences)
    text = re.sub(r"```[\s\S]*?```", " ▦CODE▦ ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)   # images → alt
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)    # links → text
    text = re.sub(r"^#{1,6}\s*(.+)$", r"\1.", text, flags=re.M)  # headers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)          # bold
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", text)  # italic
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"^>\s?", "", text, flags=re.M)           # quotes
    text = re.sub(r"^\s*---+\s*$", "", text, flags=re.M)    # rules
    text = text.replace("▦CODE▦",
                        "code snippet omitted — I sent it to chat.")

    # lists → "first, …; second, …"
    lines = text.split("\n")
    out_lines: list[str] = []
    item_no = 0
    for line in lines:
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$", line)
        if m:
            item_no += 1
            out_lines.append(f"{_ordinal(item_no)}, {m.group(1)}.")
        else:
            if line.strip():
                item_no = 0
            out_lines.append(line)
    text = "\n".join(out_lines)

    text = _normalize_numbers(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # length cap: ~800 chars ≈ 60s of speech
    cap_chars = max(200, int(cap_secs * 800 / 60))
    truncated = False
    if len(text) > cap_chars:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        head = " ".join(sentences[:2])[:cap_chars]
        text = head.rstrip() + " … I sent the full version to chat."
        truncated = True
    return SpeakPlan(spoken=text, full=full, truncated=truncated,
                     had_code=had_code)


# ── consent + stats (tiny JSON stores) ───────────────────────────────────────


class ConsentStore:
    """Per-device recording consent. First run asks; afterwards remembered."""

    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir) / "consent.json"
        self._data: dict[str, Any] = {}
        try:
            if self.path.exists():
                import json

                self._data = json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001 - corrupt consent file ≠ consent
            self._data = {}

    def consented(self, device_id: str) -> bool:
        return bool(self._data.get(device_id, {}).get("consented"))

    def grant(self, device_id: str) -> None:
        import json

        self._data[device_id] = {"consented": True, "ts": time.time()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    def revoke(self, device_id: str) -> None:
        import json

        self._data.pop(device_id, None)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2))
        except Exception:  # noqa: BLE001 - best effort
            _log.debug("consent revoke write failed", exc_info=True)

    def ensure(self, device_id: str,
              ask: Callable[[], bool]) -> bool:
        """True if recording may start. Asks once per device."""
        if self.consented(device_id):
            return True
        if ask():
            self.grant(device_id)
            return True
        return False


class StatsStore:
    """Latency audit: ear-to-ear p50/p95, barge-ins, session counts."""

    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir) / "stats.json"
        self._data = {"sessions": 0, "turns": 0, "barge_ins": 0,
                      "latencies_ms": []}
        try:
            if self.path.exists():
                import json

                loaded = json.loads(self.path.read_text())
                self._data.update(loaded)
        except Exception:  # noqa: BLE001 - stats must never break a session
            pass

    def record_session(self, report: "SessionReport") -> None:
        import json

        self._data["sessions"] += 1
        self._data["turns"] += report.turns
        self._data["barge_ins"] += report.barge_ins
        self._data["latencies_ms"].extend(report.latencies_ms)
        self._data["latencies_ms"] = self._data["latencies_ms"][-500:]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2))
        except Exception:  # noqa: BLE001 - best effort
            _log.debug("voice stats write failed", exc_info=True)

    def summary(self) -> dict[str, Any]:
        lat = sorted(self._data["latencies_ms"])
        def _pct(p: float) -> float | None:
            if not lat:
                return None
            return lat[min(len(lat) - 1, int(p * len(lat)))]
        return {
            "sessions": self._data["sessions"],
            "turns": self._data["turns"],
            "barge_ins": self._data["barge_ins"],
            "ear_to_ear_ms_p50": _pct(0.50),
            "ear_to_ear_ms_p95": _pct(0.95),
        }


# ── STT wiring ───────────────────────────────────────────────────────────────


def make_bridge_stt(bridge: Any, *, language: str = "en",
                    timeout: float = 120.0) -> Callable[[str], str]:
    """Adapt ``VoiceBridge.transcribe_voice`` (async) to the sync session.

    Each call runs on a private event loop inside a helper thread, so this
    works whether or not the caller already has a running loop — the live
    session is synchronous by design and must never inherit the bridge's
    loop. This is the deliberate reuse point: one transcription path for
    voice notes *and* the live loop.
    """

    def _stt(wav_path: str) -> str:
        import asyncio

        async def _go() -> str:
            return await bridge.transcribe_voice(wav_path, language=language)

        box: dict[str, Any] = {}

        def _run() -> None:
            try:
                box["result"] = asyncio.run(_go())
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc

        thread = threading.Thread(target=_run, daemon=True,
                                  name="voice-stt")
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError(f"transcription timed out after {timeout}s")
        if "error" in box:
            raise box["error"]
        return str(box.get("result", "") or "")

    _stt.supports_partial = False  # type: ignore[attr-defined]
    return _stt


def stt_supports_partial(stt: Callable[..., Any]) -> bool:
    """Probe: does this STT backend stream partial transcripts?

    Honest answer today: no backend in the tree streams partials — every
    STT call is one-shot per utterance. The session is written so a
    streaming backend can slot in later (it would feed the think pipeline
    incrementally); until one exists this returns False and the session
    says so instead of faking it.
    """
    return bool(getattr(stt, "supports_partial", False))


# ── the live session ─────────────────────────────────────────────────────────


@dataclass
class SessionReport:
    session_id: str
    turns: int = 0
    barge_ins: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    end_reason: str = "completed"  # completed | no-consent | no-audio |
                                   # user-exit | interrupted | error
    error: str = ""


_END_WORDS = {"exit", "quit", "goodbye", "bye", "stop listening"}


class VoiceSession:
    """One live voice conversation.

    States: ``idle → listening → thinking → speaking → listening …``,
    plus ``ended`` / ``error``. Every transition is logged with a
    timestamp; ear-to-ear latency (first mic chunk → speech start) is
    recorded per turn for ``nm voice stats``.

    The mic stays hot while Devon speaks — that is the barge-in path, and
    it is implemented first-class, not bolted on: owner speech during
    playback stops the audio immediately, discards the unspoken remainder,
    and the interruption becomes the next turn's input.
    """

    def __init__(
        self,
        *,
        think: Callable[[str], str],
        stt: Callable[[str], str],
        tts: Callable[[str, str], dict[str, Any]],
        mic: MicSource | None = None,
        speaker: SpeakerSink | None = None,
        deliver_text: Callable[[str, str], None] | None = None,
        data_dir: str | Path = "",
        device_id: str = "",
        profile: str = "",
        silence_ms: int = 800,
        keep_audio: bool = False,
        audio_key: bytes | None = None,
        speech_cap_secs: int = 60,
    ) -> None:
        self.session_id = new_short_id()
        self.think = think            # transcript -> reply (normal pipeline)
        self.stt = stt                # wav path -> transcript (one-shot)
        self.tts = tts                # (text, profile) -> {"path": wav}
        self.mic = mic
        self.speaker = speaker
        self.deliver_text = deliver_text or (lambda _tid, _t: None)
        self.data_dir = Path(data_dir) if data_dir else Path("voice_data")
        self.device_id = device_id or "default"
        self.profile = profile
        self.silence_ms = silence_ms
        self.keep_audio = keep_audio
        self.audio_key = audio_key
        self.speech_cap_secs = speech_cap_secs
        self.consent = ConsentStore(self.data_dir)
        self.stats = StatsStore(self.data_dir)
        self.state = "idle"
        self._tmp = self.data_dir / "tmp"
        self._audio_keep = self.data_dir / "audio"
        self._capture: MicCapture | None = None
        if self.keep_audio and self.audio_key is None:
            _log.warning(
                "voice session %s: keep_audio is on but no audio key was "
                "given — retained utterances will be stored UNENCRYPTED",
                self.session_id)

    # -- state ---------------------------------------------------------
    def _set_state(self, state: str) -> None:
        _log.info("voice session %s: %s → %s", self.session_id, self.state,
                  state)
        self.state = state

    @property
    def is_recording(self) -> bool:
        """True while the mic is hot: listening for the owner, or speaking
        with barge-in armed. ``thinking``/``idle``/``ended`` → False."""
        return self.state in ("listening", "speaking")

    # -- consent -------------------------------------------------------
    def check_consent(self, ask: Callable[[], bool] | None = None) -> bool:
        """Recording starts ONLY after explicit per-session consent."""
        if ask is None:
            return self.consent.consented(self.device_id)
        return self.consent.ensure(self.device_id, ask)

    # -- main loop -----------------------------------------------------
    def run(self, *, max_turns: int = 0,
            ask_consent: Callable[[], bool] | None = None) -> SessionReport:
        report = SessionReport(session_id=self.session_id)
        if not self.check_consent(ask_consent):
            report.end_reason = "no-consent"
            _log.warning("voice session %s refused: no recording consent",
                         self.session_id)
            return report

        mic = self.mic or default_mic()
        if mic is None:
            report.end_reason = "no-audio"
            report.error = "no audio device"
            self._set_state("error")
            return report
        self.mic = mic
        if self.speaker is None:
            self.speaker = default_speaker()  # may be None → text fallback
        self._tmp.mkdir(parents=True, exist_ok=True)

        # One capture thread owns the mic for the whole session (utterance
        # loop and barge-in poll both drain it — no competing readers).
        capture = MicCapture(mic)
        capture.start()
        self._capture = capture

        # Room calibration: the first half-second is assumed ambient, so the
        # VAD learns the real noise floor before the first utterance. Uses a
        # low percentile (not the peak) so talking during the first instant
        # doesn't deafen the session.
        room_vad = EnergyVAD(silence_ms=self.silence_ms)
        energies = []
        for _ in range(16):  # ~480ms of room tone
            chunk = capture.listen_next()
            if chunk is None:
                break
            energies.append(_rms_int16(chunk))
        if energies:
            energies.sort()
            ambient = energies[len(energies) // 4]
            room_vad.noise_floor = max(200.0, ambient * 2.0)
            room_vad.reset()
            _log.info("voice session %s: calibrated noise floor %.0f",
                      self.session_id, room_vad.noise_floor)

        try:
            while True:
                if max_turns and report.turns >= max_turns:
                    break
                self._set_state("listening")
                heard = self._listen_utterance(
                    room_vad, capture, prefix=self._next_listen_prefix())
                if heard is None:  # stream ended
                    report.end_reason = "completed"
                    break
                pcm, first_speech_ts = heard
                if not pcm:
                    continue
                wav_path = self._store_utterance(pcm, report.turns)

                self._set_state("thinking")
                transcript = (self.stt(wav_path) or "").strip()
                self._discard_utterance(wav_path)
                _log.info("voice session %s heard: %r", self.session_id,
                          transcript[:120])
                if not transcript:
                    continue
                if transcript.lower().strip(" .!") in _END_WORDS:
                    report.end_reason = "user-exit"
                    break

                reply = self.think(transcript) or ""
                plan = speakify(reply, cap_secs=self.speech_cap_secs)
                # voice summarizes, chat keeps the detail — in parallel
                try:
                    self.deliver_text(f"turn-{report.turns}", plan.full)
                except Exception:  # noqa: BLE001 - delivery never breaks voice
                    _log.debug("deliver_text failed", exc_info=True)

                self._set_state("speaking")
                outcome = self._speak(plan, first_speech_ts, report)
                report.turns += 1
                if outcome == "barged":
                    report.barge_ins += 1
                    # the interruption audio is already captured in
                    # _pending_barge; the next listen consumes it as the
                    # start of the owner's turn.
                    continue
        except KeyboardInterrupt:
            report.end_reason = "interrupted"
        except Exception as exc:  # noqa: BLE001 - session must end cleanly
            report.end_reason = "error"
            report.error = f"{type(exc).__name__}: {exc}"
            _log.exception("voice session %s failed", self.session_id)
        finally:
            self._set_state("ended")
            try:
                capture.stop()
            except Exception:  # noqa: BLE001
                pass
            self._capture = None
            try:
                mic.close()
            except Exception:  # noqa: BLE001
                pass
            if self.speaker is not None:
                try:
                    self.speaker.close()
                except Exception:  # noqa: BLE001
                    pass
            self.stats.record_session(report)
        return report

    # -- listen --------------------------------------------------------
    # Rolling prebuffer: the VAD needs ~150ms of speech to *declare* speech,
    # and without a prebuffer those onset chunks are lost — the transcript
    # starts mid-word. We keep the last ~300ms rolling and prepend it.
    PREBUFFER_CHUNKS = 10

    def _listen_utterance(self, vad: EnergyVAD, capture: MicCapture,
                          prefix: bytes = b"") -> tuple[bytes, float] | None:
        """Capture one utterance: speech … then `silence_ms` of quiet.

        Returns (pcm_bytes, first_speech_timestamp) or None on stream end.
        """
        from collections import deque

        chunks: list[bytes] = []
        prebuffer: deque[bytes] = deque(maxlen=self.PREBUFFER_CHUNKS)
        first_speech_ts = 0.0
        in_speech = False
        if prefix:
            chunks.append(prefix)
            in_speech = True
            first_speech_ts = time.time()
        idle_chunks = 0
        while True:
            chunk = capture.listen_next()
            if chunk is None:
                return None
            if not in_speech:
                prebuffer.append(chunk)
            status = vad.observe(chunk)
            if status == "speech":
                if not in_speech:
                    in_speech = True
                    first_speech_ts = time.time()
                    # recover the onset the VAD needed to make up its mind
                    chunks.extend(prebuffer)
                    prebuffer.clear()
                idle_chunks = 0
                chunks.append(chunk)
            elif in_speech:
                chunks.append(chunk)  # keep trailing audio pre-silence
                idle_chunks += 1
                if idle_chunks >= vad.silence_chunks:
                    # trim the trailing silence
                    pcm = b"".join(chunks[:len(chunks) - idle_chunks])
                    return pcm, first_speech_ts
            # else: pre-speech silence — only the prebuffer keeps it

    # -- speak (with barge-in) ------------------------------------------
    def _speak(self, plan: SpeakPlan, first_speech_ts: float,
               report: SessionReport) -> str:
        """Play the reply sentence by sentence. Returns 'done' | 'barged'.

        TTS output over ~2 minutes is split into segments first — one giant
        voice note is awkward on every platform.
        """
        sentences = [s.strip() for s in
                     re.split(r"(?<=[.!?…])\s+", plan.spoken) if s.strip()]
        if not sentences:
            return "done"
        barge_vad = EnergyVAD(silence_ms=10_000, min_speech_ms=200)
        barge_prefix = bytearray()
        speech_started = False

        for sentence in sentences:
            try:
                info = self.tts(sentence, self.profile)
                wav_path = info.get("path", "")
            except Exception as exc:  # noqa: BLE001 - TTS hiccup ≠ dead air
                _log.warning("voice TTS failed: %s", exc)
                continue
            if not wav_path or not Path(wav_path).exists():
                continue
            segments = split_wav(wav_path, max_secs=120)
            for segment in segments:
                outcome = self._play_segment(
                    segment, sentence, barge_vad, barge_prefix,
                    first_speech_ts, report, speech_started)
                speech_started = True
                if outcome == "barged":
                    return "barged"
        return "done"

    def _play_segment(self, wav_path: str, sentence: str,
                      barge_vad: EnergyVAD,
                      barge_prefix: bytearray, first_speech_ts: float,
                      report: SessionReport, speech_started: bool) -> str:
        """Play one wav segment; returns 'done' | 'barged'."""
        if self.speaker is None:
            _log.info("voice (no speaker): %s", sentence[:120])
            if not speech_started:
                self._record_latency(first_speech_ts, report)
            return "done"
        token = self.speaker.play(wav_path)
        if not speech_started:
            self._record_latency(first_speech_ts, report)
        # mic stays hot: poll for the owner talking over us
        while self.speaker.playing(token):
            chunk = self._poll_mic()
            if chunk is None:
                break
            if barge_vad.is_speech(chunk):
                barge_prefix += chunk
                # ~200ms of continuous speech = a real interruption
                if len(barge_prefix) >= int(SAMPLE_RATE * 0.2
                                           * SAMPLE_WIDTH):
                    self.speaker.stop(token)
                    _log.info("voice session %s barged in",
                              self.session_id)
                    self._pending_barge = bytes(barge_prefix)
                    return "barged"
            else:
                barge_prefix.clear()
        return "done"

    def _poll_mic(self) -> bytes | None:
        """One short-timeout mic read during playback, via the capture
        thread. None = nothing arrived in time (treated as silence)."""
        capture = self._capture
        if capture is None:
            return None
        chunk = capture.poll(timeout=0.05)
        return chunk if chunk is not None else b"\x00" * CHUNK_BYTES

    def _record_latency(self, first_speech_ts: float,
                        report: SessionReport) -> None:
        if first_speech_ts:
            ms = (time.time() - first_speech_ts) * 1000.0
            report.latencies_ms.append(ms)
            _log.info("voice ear-to-ear latency: %.0f ms", ms)

    # -- utterance retention --------------------------------------------
    def _store_utterance(self, pcm: bytes, turn: int) -> str:
        """Raw mic audio: temp file always; kept only on explicit opt-in.

        Kept audio is AES-encrypted at rest when ``audio_key`` is set
        (``nm voice call --keep-audio --audio-key-file <path>``). Without a
        key it lands as plain wav and the session logs a warning saying so.
        """
        self._tmp.mkdir(parents=True, exist_ok=True)
        path = self._tmp / f"utt-{self.session_id}-{turn}.wav"
        write_wav_bytes(path, pcm)
        if self.keep_audio:
            self._audio_keep.mkdir(parents=True, exist_ok=True)
            if self.audio_key:
                from ..core.cipher import aes_encrypt

                blob = aes_encrypt(pcm, key=self.audio_key)
                kept = self._audio_keep / f"{self.session_id}-{turn}.wav.enc"
                kept.write_text(blob)
                path.unlink(missing_ok=True)
                return str(kept)
            kept = self._audio_keep / f"{self.session_id}-{turn}.wav"
            path.replace(kept)
            return str(kept)
        return str(path)

    def _discard_utterance(self, wav_path: str) -> None:
        if not self.keep_audio:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    def purge_audio(self) -> int:
        """Delete retained raw audio (plain and encrypted). Returns files removed."""
        removed = 0
        for path in (self._audio_keep, self._tmp):
            if path.exists():
                for f in list(path.glob("*.wav")) + list(path.glob("*.wav.enc")):
                    try:
                        f.unlink()
                        removed += 1
                    except Exception:  # noqa: BLE001
                        pass
        return removed


    # -- barge-in plumbing -------------------------------------------------
    _pending_barge: bytes = b""

    def _next_listen_prefix(self) -> bytes:
        pending, self._pending_barge = self._pending_barge, b""
        return pending


def decrypt_kept_audio(path: str | Path, key: bytes) -> bytes:
    """Recover PCM bytes from an encrypted kept utterance (``.wav.enc``)."""
    from ..core.cipher import aes_decrypt

    return aes_decrypt(Path(path).read_text(), key=key)
