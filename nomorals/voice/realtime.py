"""Realtime voice conversion: mic -> model -> speaker, live.

The realtime-voice-conversion pattern (census §3B.12), mined, not
ported — no upstream code or text lives here. The upstream is a Rust
shell that captures the microphone, ships proto-over-ZeroMQ to a
Python CycleGAN-VC sidecar, and plays the converted audio back over
the speaker (4.39s of delay on the author's desktop, drifting up to
6s). Devon's realisation, inside the voice organ:

- :class:`VocodeAudioRequest` / :class:`VocodeAudioResponse` /
  :class:`VocodeParams` — the ``audio.proto`` contract as data, with a
  pure-stdlib proto3 wire codec (packed ``float`` field 1, varint
  batch numbers, embedded params). The payload bytes are the same
  bytes the upstream's protobuf produces, so any peer speaking the
  proto can attach.
- :class:`AudioFrameQueue` — the Rust ``AudioQueue`` (locked deque of
  ``f32`` with ``drain_size``) in stdlib.
- :class:`SpeechGate` — the Rust sidecar's voice-activity gate,
  faithfully: mean-of-positive-samples > 0.006, activate after 3
  speech frames, deactivate after 20 silent ones, a 5-frame ring
  buffer that replays the activating edge. (One real fix: silence
  with no positive samples divided by zero upstream; here it is
  simply not speech.)
- :class:`Converter` — the Python sidecar's ``convert_partial``
  pipeline: pre-resample -> pad -> pitch map (the upstream's log-F0
  mean/std mapping, first order) -> spectral features -> model ->
  resynthesis with the original phase -> post-resample. WORLD/pyworld
  is replaced by STFT + iSTFT on scipy (this box's librosa 0.9.2 +
  numpy 1.26 breaks inside ``librosa.istft``): the same pipeline shape,
  the vocoder rung this box can actually run. ``skip_vocode`` and
  ``discard_vocoded_audio`` keep the upstream's exact semantics.
- :class:`TorchScriptExecutor` / :class:`RegistryExecutor` — the
  model seam. A TorchScript file by path, or a registered model
  service (``slug@version``) through the voice service registry
  (``ModelServer`` load/execute/unload) once that rung is on the
  build. :func:`export_probe_model` writes a real probe converter
  (a small residual spectral CNN, scripted, seed 0) for the rung to
  run against.
- :class:`ConversionServer` — the sidecar's ZeroMQ REP loop as a
  transport-free object: ``handle(payload) -> bytes`` accumulates to
  ``buffer_size_minimum`` (replying ``b"OK"`` while buffering, like
  the upstream), then converts and replies with the response.
- Transports: in-process (direct call), TCP (stdlib socket,
  length-prefixed REQ/REP framing), and ZeroMQ (pyzmq, the
  upstream-faithful backend — honest refusal naming ``pyzmq`` when
  it is absent).
- :class:`RealtimeConverter` — the live loop (the Rust ``main`` +
  sidecar thread): the organ's own ``MicCapture``/``default_mic``
  and ``SpeakerSink``/``default_speaker`` do the hardware; the loop
  gates, batches, ships, and plays converted audio back per batch.

Stdlib only at import time; torch, librosa, scipy and pyzmq are
lazy, and every rung that needs one names it when it is missing
(:class:`ConversionUnavailable`) instead of faking a conversion.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
from array import array
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "AudioFrameQueue",
    "ConversionParams",
    "ConversionServer",
    "ConversionUnavailable",
    "Converter",
    "InProcessClient",
    "RealtimeConverter",
    "RegistryExecutor",
    "SpeechGate",
    "TcpClient",
    "TorchScriptExecutor",
    "VocodeAudioRequest",
    "VocodeAudioResponse",
    "VocodeParams",
    "build_probe_converter",
    "control_convert",
    "conversion_status",
    "convert_file",
    "default_probe_path",
    "default_services_dir",
    "export_probe_model",
    "make_client",
    "resolve_executor",
    "serve_tcp",
    "serve_zeromq",
    "status_text",
]


class ConversionUnavailable(Exception):
    """A conversion rung is missing — the caller falls back honestly."""


#: The upstream's defaults, kept as the organ's defaults: the sidecar
#: waited for 70,000 buffered samples before converting (the value
#: that "sounds pretty good" in the Rust shell's comments), and the
#: model runs at 16 kHz ("probably don't want to change from 16000").
DEFAULT_MODEL_RATE = 16000
DEFAULT_BUFFER_SIZE_MINIMUM = 70000
#: The Rust shell drained the mic queue in 5,000-sample chunks.
SEND_SIZE = 5000


def default_services_dir() -> str:
    """The voice service store — the same rule as the service registry.

    Shared as a path convention (the registry module is not on every
    build yet): env override, else ``~/.config/nomorals/voice/services``.
    """
    override = os.environ.get("NOMORALS_VOICE_SERVICES_DIR", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".config",
                        "nomorals", "voice", "services")


def default_probe_path() -> str:
    """Where ``export_probe_model`` writes the probe converter."""
    return os.path.join(default_services_dir(),
                        "voice-convert-probe-0.1.0.pt")


# -- proto3 wire codec (audio.proto, pure stdlib) -------------------------

def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint fields here are never negative")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift >= 70:
            raise ValueError("varint too long")


def _tag(field: int, wire_type: int) -> bytes:
    return _encode_varint((field << 3) | wire_type)


def _skip_field(buf: bytes, pos: int, wire_type: int) -> int:
    if wire_type == 0:
        _, pos = _decode_varint(buf, pos)
        return pos
    if wire_type == 1:
        if pos + 8 > len(buf):
            raise ValueError("truncated 64-bit field")
        return pos + 8
    if wire_type == 2:
        length, pos = _decode_varint(buf, pos)
        if pos + length > len(buf):
            raise ValueError("truncated length-delimited field")
        return pos + length
    if wire_type == 5:
        if pos + 4 > len(buf):
            raise ValueError("truncated 32-bit field")
        return pos + 4
    raise ValueError(f"unsupported wire type {wire_type}")


@dataclass
class VocodeParams:
    """The per-batch conversion parameters (audio.proto field 10)."""

    initial_sample_rate: int = 0
    original_source_rate: int = 0
    original_source_save_file: bool = False
    pre_convert_resample: bool = False
    pre_convert_resample_rate: int = 0
    pre_convert_resample_save_file: bool = False
    model_hyperparameter_sampling_rate: int = DEFAULT_MODEL_RATE
    model_save_file: bool = False
    post_convert_resample: bool = False
    post_convert_resample_rate: int = 0
    post_convert_resample_save_file: bool = False

    def to_bytes(self) -> bytes:
        out = bytearray()
        if self.initial_sample_rate:
            out += _tag(1, 0) + _encode_varint(self.initial_sample_rate)
        if self.pre_convert_resample:
            out += _tag(2, 0) + b"\x01"
        if self.pre_convert_resample_rate:
            out += _tag(3, 0) + _encode_varint(
                self.pre_convert_resample_rate)
        if self.pre_convert_resample_save_file:
            out += _tag(4, 0) + b"\x01"
        if self.model_hyperparameter_sampling_rate:
            out += _tag(5, 0) + _encode_varint(
                self.model_hyperparameter_sampling_rate)
        if self.post_convert_resample:
            out += _tag(6, 0) + b"\x01"
        if self.post_convert_resample_rate:
            out += _tag(7, 0) + _encode_varint(
                self.post_convert_resample_rate)
        if self.post_convert_resample_save_file:
            out += _tag(8, 0) + b"\x01"
        if self.original_source_rate:
            out += _tag(9, 0) + _encode_varint(self.original_source_rate)
        if self.original_source_save_file:
            out += _tag(10, 0) + b"\x01"
        if self.model_save_file:
            out += _tag(11, 0) + b"\x01"
        return bytes(out)

    @classmethod
    def from_bytes(cls, buf: bytes) -> VocodeParams:
        params = cls()
        pos = 0
        while pos < len(buf):
            field, pos = _decode_varint(buf, pos)
            number, wire = field >> 3, field & 0x7
            if wire != 0:
                pos = _skip_field(buf, pos, wire)
                continue
            value, pos = _decode_varint(buf, pos)
            if number == 1:
                params.initial_sample_rate = value
            elif number == 2:
                params.pre_convert_resample = bool(value)
            elif number == 3:
                params.pre_convert_resample_rate = value
            elif number == 4:
                params.pre_convert_resample_save_file = bool(value)
            elif number == 5:
                params.model_hyperparameter_sampling_rate = value
            elif number == 6:
                params.post_convert_resample = bool(value)
            elif number == 7:
                params.post_convert_resample_rate = value
            elif number == 8:
                params.post_convert_resample_save_file = bool(value)
            elif number == 9:
                params.original_source_rate = value
            elif number == 10:
                params.original_source_save_file = bool(value)
            elif number == 11:
                params.model_save_file = bool(value)
        return params


@dataclass
class VocodeAudioRequest:
    """One mic batch for the sidecar (audio.proto, fields as numbered)."""

    float_audio: list[float] = field(default_factory=list)
    request_batch_number: int = 0
    vocode_params: VocodeParams = field(default_factory=VocodeParams)
    skip_vocode: bool = False
    buffer_size_minimum: int = 0
    discard_vocoded_audio: bool = False

    def to_bytes(self) -> bytes:
        out = bytearray()
        if self.float_audio:
            packed = struct.pack(f"<{len(self.float_audio)}f",
                                 *self.float_audio)
            out += _tag(1, 2) + _encode_varint(len(packed)) + packed
        if self.request_batch_number:
            out += _tag(2, 0) + _encode_varint(self.request_batch_number)
        if self.skip_vocode:
            out += _tag(4, 0) + b"\x01"
        if self.buffer_size_minimum:
            out += _tag(6, 0) + _encode_varint(self.buffer_size_minimum)
        if self.discard_vocoded_audio:
            out += _tag(8, 0) + b"\x01"
        encoded_params = self.vocode_params.to_bytes()
        out += _tag(10, 2) + _encode_varint(len(encoded_params))
        out += encoded_params
        return bytes(out)

    @classmethod
    def from_bytes(cls, buf: bytes) -> VocodeAudioRequest:
        request = cls()
        pos = 0
        while pos < len(buf):
            field, pos = _decode_varint(buf, pos)
            number, wire = field >> 3, field & 0x7
            if number == 1 and wire == 2:
                length, pos = _decode_varint(buf, pos)
                if pos + length > len(buf):
                    raise ValueError("truncated float_audio")
                raw = buf[pos:pos + length]
                pos += length
                if length % 4:
                    raise ValueError("float_audio length is not a "
                                     "multiple of 4")
                request.float_audio.extend(
                    struct.unpack(f"<{length // 4}f", raw))
            elif number == 1 and wire == 5:
                if pos + 4 > len(buf):
                    raise ValueError("truncated float_audio sample")
                request.float_audio.append(
                    struct.unpack_from("<f", buf, pos)[0])
                pos += 4
            elif number == 2 and wire == 0:
                request.request_batch_number, pos = _decode_varint(buf, pos)
            elif number == 4 and wire == 0:
                value, pos = _decode_varint(buf, pos)
                request.skip_vocode = bool(value)
            elif number == 6 and wire == 0:
                request.buffer_size_minimum, pos = _decode_varint(buf, pos)
            elif number == 8 and wire == 0:
                value, pos = _decode_varint(buf, pos)
                request.discard_vocoded_audio = bool(value)
            elif number == 10 and wire == 2:
                length, pos = _decode_varint(buf, pos)
                if pos + length > len(buf):
                    raise ValueError("truncated vocode_params")
                request.vocode_params = VocodeParams.from_bytes(
                    buf[pos:pos + length])
                pos += length
            else:
                pos = _skip_field(buf, pos, wire)
        return request


@dataclass
class VocodeAudioResponse:
    """One converted batch back from the sidecar."""

    float_audio: list[float] = field(default_factory=list)
    response_batch_number: int = 0

    def to_bytes(self) -> bytes:
        out = bytearray()
        if self.float_audio:
            packed = struct.pack(f"<{len(self.float_audio)}f",
                                 *self.float_audio)
            out += _tag(1, 2) + _encode_varint(len(packed)) + packed
        if self.response_batch_number:
            out += _tag(2, 0) + _encode_varint(self.response_batch_number)
        return bytes(out)

    @classmethod
    def from_bytes(cls, buf: bytes) -> VocodeAudioResponse:
        response = cls()
        pos = 0
        while pos < len(buf):
            field, pos = _decode_varint(buf, pos)
            number, wire = field >> 3, field & 0x7
            if number == 1 and wire == 2:
                length, pos = _decode_varint(buf, pos)
                if pos + length > len(buf):
                    raise ValueError("truncated float_audio")
                raw = buf[pos:pos + length]
                pos += length
                if length % 4:
                    raise ValueError("float_audio length is not a "
                                     "multiple of 4")
                response.float_audio.extend(
                    struct.unpack(f"<{length // 4}f", raw))
            elif number == 1 and wire == 5:
                if pos + 4 > len(buf):
                    raise ValueError("truncated float_audio sample")
                response.float_audio.append(
                    struct.unpack_from("<f", buf, pos)[0])
                pos += 4
            elif number == 2 and wire == 0:
                response.response_batch_number, pos = _decode_varint(
                    buf, pos)
            else:
                pos = _skip_field(buf, pos, wire)
        return response


# -- the frame queue (Rust AudioQueue) -------------------------------------

class AudioFrameQueue:
    """Thread-safe queue of f32 frames, with bounded drain.

    The Rust ``AudioQueue``: ``push_back`` / ``extend`` / ``drain`` /
    ``drain_size`` (``None`` when short) / ``len``. The bound drops
    the OLDEST frames first and counts them — a live loop that falls
    behind drops history, never the present.
    """

    def __init__(self, max_samples: int = 0) -> None:
        self.max_samples = max(0, int(max_samples))
        self._queue: deque[float] = deque()
        self._cond = threading.Condition(threading.Lock())
        self.pushed = 0
        self.dropped = 0

    def push(self, sample: float) -> None:
        self.extend([sample])

    def extend(self, samples: Any) -> None:
        items = [float(s) for s in samples]
        with self._cond:
            self._queue.extend(items)
            self.pushed += len(items)
            if self.max_samples and len(self._queue) > self.max_samples:
                drop = len(self._queue) - self.max_samples
                for _ in range(drop):
                    self._queue.popleft()
                self.dropped += drop
            self._cond.notify_all()

    def drain(self) -> list[float]:
        with self._cond:
            out = list(self._queue)
            self._queue.clear()
            return out

    def drain_size(self, size: int) -> list[float] | None:
        """Up to ``size`` oldest frames, or None when short."""
        with self._cond:
            if len(self._queue) < size:
                return None
            return [self._queue.popleft() for _ in range(size)]

    def wait_size(self, size: int, timeout: float) -> list[float] | None:
        """Blocking ``drain_size`` — None on timeout."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while len(self._queue) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
            return [self._queue.popleft() for _ in range(size)]

    def __len__(self) -> int:
        with self._cond:
            return len(self._queue)


# -- the voice-activity gate (Rust Sidecar::is_speaking) -------------------

class SpeechGate:
    """The sidecar's speech gate, faithfully ported.

    A frame is speech when the mean of its POSITIVE samples exceeds
    0.006. Three speech frames in a row activate the gate; twenty
    silent frames in a row deactivate it. The five most recent frames
    are kept in a ring buffer so the activating edge is replayed, not
    lost. ``feed`` returns the frames to send (empty while inactive).
    """

    THRESHOLD = 0.006
    ACTIVATE_FRAMES = 3
    DEACTIVATE_FRAMES = 20
    RING_FRAMES = 5

    def __init__(self) -> None:
        self.active = False
        self.activations = 0
        self._up = 0
        self._down = 0
        self._ring: deque[list[float]] = deque(maxlen=self.RING_FRAMES)

    @staticmethod
    def is_speaking(frame: Any) -> bool:
        positives = [float(x) for x in frame if float(x) > 0.0]
        if not positives:
            # Upstream divided by zero here (no positive samples at
            # all); silence with nothing positive is simply not speech.
            return False
        return sum(positives) / len(positives) > SpeechGate.THRESHOLD

    def feed(self, frame: Any) -> list[float]:
        frame = [float(x) for x in frame]
        self._ring.append(frame)
        if self.active:
            if self.is_speaking(frame):
                self._down = 0
            else:
                self._down += 1
                if self._down > self.DEACTIVATE_FRAMES:
                    self.active = False
                    self._up = 0
                    self._down = 0
                    self._ring.clear()
                    return []
        else:
            if self.is_speaking(frame):
                self._up += 1
            else:
                self._up = 0
            if self._up > self.ACTIVATE_FRAMES:
                self.active = True
                self.activations += 1
                self._up = 0
                self._down = 0
        if not self.active:
            return []
        out: list[float] = []
        for chunk in self._ring:
            out.extend(chunk)
        self._ring.clear()
        return out


# -- conversion parameters -------------------------------------------------

@dataclass
class ConversionParams:
    """One conversion's settings, as data (not buried constants)."""

    #: The rate the model runs at (upstream: 16000, do not change).
    model_rate: int = DEFAULT_MODEL_RATE
    #: Output rate; 0 = the input rate.
    output_rate: int = 0
    #: Samples buffered before a batch converts (upstream: 70000).
    buffer_size_minimum: int = DEFAULT_BUFFER_SIZE_MINIMUM
    #: A2B (source -> target) or B2A (target -> source).
    direction: str = "A2B"
    #: Explicit pitch shift in semitones; wins over the F0 stats.
    pitch_shift_semitones: float = 0.0
    #: Log-F0 normalisation stats of a trained pair (the upstream's
    #: logf0s_normalization.npz). The mean shift becomes semitones.
    f0_mean_log_src: float | None = None
    f0_std_log_src: float | None = None
    f0_mean_log_target: float | None = None
    f0_std_log_target: float | None = None
    #: Upstream flags, exact semantics (see Converter.convert).
    skip_vocode: bool = False
    discard_vocoded_audio: bool = False
    #: STFT geometry (upstream's wavs_to_specs used n_fft 1024).
    n_fft: int = 1024
    hop_length: int = 256
    #: 0 = resynthesize with the original phase; >0 = Griffin-Lim.
    griffin_iters: int = 0

    def pitch_shift(self) -> float:
        """The pitch shift in semitones for this conversion."""
        if self.pitch_shift_semitones:
            return float(self.pitch_shift_semitones)
        stats = (self.f0_mean_log_src, self.f0_std_log_src,
                 self.f0_mean_log_target, self.f0_std_log_target)
        if any(s is None for s in stats):
            return 0.0
        return float(12.0 * (self.f0_mean_log_target
                             - self.f0_mean_log_src) / math.log(2))

    def to_vocode_params(self, *, source_rate: int) -> VocodeParams:
        """The wire params for a batch from this source."""
        return VocodeParams(
            initial_sample_rate=int(source_rate),
            original_source_rate=int(source_rate),
            pre_convert_resample=bool(source_rate != self.model_rate),
            pre_convert_resample_rate=self.model_rate,
            model_hyperparameter_sampling_rate=self.model_rate,
            post_convert_resample=bool(self.output_rate
                                       and self.output_rate != self.model_rate),
            post_convert_resample_rate=int(self.output_rate or 0),
        )

    @classmethod
    def from_vocode_params(cls, params: VocodeParams,
                           request: VocodeAudioRequest,
                           *, base: ConversionParams | None = None
                           ) -> ConversionParams:
        """A request's params, as conversion settings.

        The upstream re-sent params with every batch and used the last
        batch's settings; the server does the same per batch.
        """
        base = base or ConversionParams()
        if params.pre_convert_resample and params.pre_convert_resample_rate:
            model_rate = params.pre_convert_resample_rate
        elif params.model_hyperparameter_sampling_rate:
            model_rate = params.model_hyperparameter_sampling_rate
        else:
            model_rate = base.model_rate
        if params.post_convert_resample and params.post_convert_resample_rate:
            output_rate = params.post_convert_resample_rate
        else:
            output_rate = 0
        return cls(
            model_rate=model_rate, output_rate=output_rate,
            buffer_size_minimum=(request.buffer_size_minimum
                                 or base.buffer_size_minimum),
            direction=base.direction,
            pitch_shift_semitones=base.pitch_shift_semitones,
            f0_mean_log_src=base.f0_mean_log_src,
            f0_std_log_src=base.f0_std_log_src,
            f0_mean_log_target=base.f0_mean_log_target,
            f0_std_log_target=base.f0_std_log_target,
            skip_vocode=request.skip_vocode,
            discard_vocoded_audio=request.discard_vocoded_audio,
            n_fft=base.n_fft, hop_length=base.hop_length,
            griffin_iters=base.griffin_iters,
        )


# -- resampling / pitch (librosa first, scipy second) ----------------------

def _librosa_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("librosa") is not None
    except Exception:  # noqa: BLE001 — availability probe, never raises
        return False


def _scipy_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("scipy.signal") is not None
    except Exception:  # noqa: BLE001 — availability probe, never raises
        return False


def _pyzmq_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("zmq") is not None
    except Exception:  # noqa: BLE001 — availability probe, never raises
        return False


def _resample(samples: Any, src_rate: int, dst_rate: int) -> list[float]:
    """Resample f32 audio. librosa first, scipy's polyphase second."""
    frames = [float(s) for s in samples]
    if not frames or src_rate == dst_rate:
        return frames
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"bad resample rates {src_rate} -> {dst_rate}")
    if _librosa_available():
        import librosa
        import numpy as np
        out = librosa.resample(
            np.asarray(frames, dtype=np.float32),
            orig_sr=int(src_rate), target_sr=int(dst_rate))
        return [float(x) for x in out]
    if _scipy_available():
        import numpy as np
        from scipy.signal import resample_poly
        common = math.gcd(int(src_rate), int(dst_rate))
        out = resample_poly(np.asarray(frames, dtype=np.float64),
                            int(dst_rate) // common, int(src_rate) // common)
        return [float(x) for x in out]
    raise ConversionUnavailable(
        "resampling needs librosa or scipy — neither is installed on "
        "this machine (pip install librosa)")


def _pitch_shift(samples: Any, rate: int, semitones: float) -> list[float]:
    """Shift pitch by semitones (librosa phase vocoder)."""
    frames = [float(s) for s in samples]
    if not frames or not semitones:
        return frames
    if not _librosa_available():
        raise ConversionUnavailable(
            "pitch shifting needs librosa — it is not installed on "
            "this machine (pip install librosa)")
    import librosa
    import numpy as np
    out = librosa.effects.pitch_shift(
        np.asarray(frames, dtype=np.float32), sr=int(rate),
        n_steps=float(semitones))
    return [float(x) for x in out]


def _require_scipy(what: str) -> Any:
    try:
        from scipy.signal import istft, stft
    except ImportError as exc:
        raise ConversionUnavailable(
            f"{what} needs scipy — it is not installed on this machine "
            "(pip install scipy)") from exc
    return stft, istft


def _stft(frames: Any, params: ConversionParams) -> tuple[Any, Any, int]:
    """Magnitude + unit phase of the STFT (the WORLD decompose rung).

    scipy's STFT, not librosa's: this box's librosa 0.9.2 + numpy
    1.26 breaks inside ``librosa.istft``/``griffinlim`` (``np.float``
    was removed in numpy 1.24), while scipy's pair round-trips to
    machine precision. Returns (magnitude, phase, sample count) —
    the count trims the padded resynthesis back to the input length.
    """
    stft, _ = _require_scipy("spectral features")
    import numpy as np
    signal = np.asarray(frames, dtype=np.float64)
    _, _, matrix = stft(signal, fs=params.model_rate, window="hann",
                        nperseg=params.n_fft,
                        noverlap=params.n_fft - params.hop_length,
                        boundary="zeros", padded=True)
    magnitude = np.abs(matrix).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = np.where(magnitude > 0, matrix / np.where(
            magnitude > 0, magnitude, 1), 1.0 + 0.0j)
    return magnitude, phase.astype(np.complex128), len(signal)


def _istft(magnitude: Any, phase: Any, params: ConversionParams,
           n_samples: int) -> list[float]:
    """Resynthesize magnitude with the original phase, trimmed."""
    _, istft = _require_scipy("resynthesis")
    import numpy as np
    matrix = np.asarray(magnitude, dtype=np.float64) * np.asarray(phase)
    _, out = istft(matrix, fs=params.model_rate, window="hann",
                   nperseg=params.n_fft,
                   noverlap=params.n_fft - params.hop_length,
                   boundary=True)
    return [float(x) for x in out[:n_samples]]


def _griffinlim(magnitude: Any, params: ConversionParams,
                n_samples: int) -> list[float]:
    """Resynthesize magnitude without phase (Griffin-Lim, seeded)."""
    stft, _ = _require_scipy("Griffin-Lim resynthesis")
    import numpy as np
    mag = np.asarray(magnitude, dtype=np.float64)
    rng = np.random.default_rng(0)
    angles = np.exp(2j * np.pi * rng.random(mag.shape))
    rebuilt: Any = None
    for _ in range(max(1, params.griffin_iters)):
        rebuilt = _istft(mag, angles, params, n_samples)
        _, _, matrix = stft(np.asarray(rebuilt, dtype=np.float64),
                            fs=params.model_rate, window="hann",
                            nperseg=params.n_fft,
                            noverlap=params.n_fft - params.hop_length,
                            boundary="zeros", padded=True)
        angles = np.exp(1j * np.angle(matrix))
    if rebuilt is None:
        rebuilt = _istft(mag, angles, params, n_samples)
    return [float(x) for x in rebuilt]


# -- the model seam ----------------------------------------------------------

class TorchScriptExecutor:
    """A TorchScript conversion model, by path.

    ``execute({"magnitude": mag})`` runs the module's ``forward`` on
    the named tensors (numpy float32, batch first) and returns the
    outputs by the same names. The module is loaded once, on first
    use — through the shared model-load cache
    (:mod:`nomorals.voice.cache`), so a repeat conversion with the
    same model file never re-runs ``torch.jit.load`` — and must
    return exactly one output per input.
    """

    def __init__(self, path: str, *, entry: str = "forward") -> None:
        self.path = path
        self.entry = entry

    def _load(self) -> Any:
        if not os.path.isfile(self.path):
            raise ConversionUnavailable(
                f"conversion model not found: {self.path} — "
                "nm voice convert --export-probe writes one")
        from .cache import shared_model_cache
        return shared_model_cache().load_torchscript(
            self.path, entry=self.entry)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from .vits._torch import require_torch
        torch = require_torch("running a TorchScript conversion model")
        import numpy as np
        module = self._load()
        entry = getattr(module, self.entry, None)
        if entry is None or not callable(entry):
            raise ValueError(
                f"model {self.path}: no callable entry {self.entry!r}")
        tensors = {name: torch.from_numpy(
            np.ascontiguousarray(value, dtype=np.float32))
            for name, value in inputs.items()}
        with torch.no_grad():
            outs = entry(*[tensors[name] for name in inputs])
        if not isinstance(outs, (tuple, list)):
            outs = (outs,)
        if len(outs) != len(inputs):
            raise ValueError(
                f"model {self.path}: entry {self.entry!r} returned "
                f"{len(outs)} outputs for {len(inputs)} inputs")
        return {name: out.detach().cpu().numpy()
                for name, out in zip(inputs, outs, strict=True)}


class RegistryExecutor:
    """A registered model service, through the voice service registry.

    The registry module (``nomorals.voice.services``) is imported
    lazily: on a build without it, the first execute names the missing
    rung instead of failing at import time. Inputs are numpy float32
    arrays; outputs come back the same way.
    """

    def __init__(self, ref: str, *, services_dir: str = "") -> None:
        self.ref = ref
        self.services_dir = services_dir
        self._server: Any = None

    def _server_or_raise(self) -> Any:
        if self._server is None:
            try:
                from .cache import shared_model_cache
                self._server = shared_model_cache().server(
                    self.services_dir)
            except ImportError as exc:
                raise ConversionUnavailable(
                    "the voice service registry is not on this build "
                    "yet — model services ship with the storyteller-ml "
                    "rung; use --model <torchscript file> meanwhile"
                    ) from exc
        return self._server

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        server = self._server_or_raise()
        payload = {}
        for name, value in inputs.items():
            arr = np.ascontiguousarray(value, dtype=np.float32)
            payload[name] = {
                "dtype": "float32",
                "shape": [int(d) for d in arr.shape],
                "values": arr.tolist(),
            }
        result = server.execute(self.ref, payload)
        out: dict[str, Any] = {}
        for name, part in result["outputs"].items():
            out[name] = np.array(part["values"], dtype=np.float32).reshape(
                part["shape"])
        return out


def resolve_executor(*, model_path: str = "", service_ref: str = "",
                     services_dir: str = "") -> Any:
    """The model for a conversion: explicit file, service, or probe."""
    if (model_path or "").strip():
        return TorchScriptExecutor((model_path or "").strip())
    if (service_ref or "").strip():
        return RegistryExecutor((service_ref or "").strip(),
                                services_dir=services_dir)
    probe = default_probe_path()
    if os.path.isfile(probe):
        return TorchScriptExecutor(probe)
    raise ConversionUnavailable(
        "no conversion model: pass --model <torchscript file>, "
        "--service <slug@version>, or export the probe first "
        "(nm voice convert --export-probe)")


def build_probe_converter(*, seed: int = 0) -> Any:
    """The probe conversion model: a small residual spectral CNN.

    Input ``magnitude`` is ``[1, F, T]`` (batch 1, frequency, time);
    the output has the same shape: the converted magnitude is the
    source magnitude plus the network's delta, so a random-weight
    probe stays close to its input instead of collapsing to noise.
    Scripted (shape-polymorphic), deterministic at ``seed``.
    """
    from .vits._torch import require_torch
    torch = require_torch("building the realtime probe converter")
    import torch.nn as nn

    class SpectralConverter(nn.Module):
        """Two conv layers over the magnitude spectrogram + residual."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 1, 3, padding=1)

        def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
            hidden = torch.relu(self.conv1(magnitude.unsqueeze(1)))
            delta = self.conv2(hidden)
            return delta.squeeze(1) + magnitude

    torch.manual_seed(seed)
    return SpectralConverter()


def export_probe_model(out_path: str = "", *, seed: int = 0) -> dict[str, Any]:
    """Script the probe converter to disk + a receipt, and prove it runs.

    The receipt next to the ``.pt`` carries the task, entry, tensor
    spec and sha256, so the service registry can adopt the file later
    without re-exporting it.
    """
    from .vits._torch import require_torch
    torch = require_torch("exporting the realtime probe converter")
    module = build_probe_converter(seed=seed)
    scripted = torch.jit.script(module)
    example = torch.randn(1, 513, 64)
    with torch.no_grad():
        scripted(example)
    out_path = out_path or default_probe_path()
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    torch.jit.save(scripted, out_path)
    sha256 = hashlib.sha256()
    with open(out_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha256.update(chunk)
    receipt = {
        "name": "voice-convert-probe",
        "version": "0.1.0",
        "task": "voice-conversion",
        "format": "torchscript",
        "path": os.path.abspath(out_path),
        "sha256": sha256.hexdigest(),
        "entry": "forward",
        "traced": False,
        "source": f"SpectralConverter random weights, seed {seed} — "
                  "probe, not a trained voice",
        "inputs": [{"name": "magnitude", "dtype": "float32",
                    "shape": [1, None, None]}],
        "outputs": [{"name": "magnitude", "dtype": "float32",
                     "shape": [1, None, None]}],
    }
    receipt_path = os.path.splitext(out_path)[0] + ".json"
    with open(receipt_path, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
    _log.info("realtime probe model exported: %s", out_path)
    return {"ok": True, "path": os.path.abspath(out_path),
            "receipt": receipt_path, "sha256": receipt["sha256"],
            "inputs": receipt["inputs"], "outputs": receipt["outputs"]}


# -- the converter (Python sidecar's convert_partial) ----------------------

class Converter:
    """One conversion: frames in, converted frames out.

    The pipeline is the sidecar's ``convert_partial`` with WORLD
    replaced by STFT + iSTFT (pyworld is not on this box): pre-resample
    to the model rate, pad to a whole number of 4 frames (the
    upstream's ``wav_padding``), pitch-map (the log-F0 mean/std
    mapping, first order), STFT magnitude through the model,
    resynthesis with the original phase, post-resample. ``skip_vocode``
    returns the pre-resampled audio unchanged and ``discard`` returns
    the model-rate input — exactly the upstream's semantics, not a
    shortcut: both flags exist in the proto and the sidecar honored
    them, so the organ honors them too.
    """

    def __init__(self, executor: Any = None,
                 params: ConversionParams | None = None) -> None:
        self.executor = executor
        self.params = params or ConversionParams()

    def convert(self, frames: Any, sample_rate: int, *,
                params: ConversionParams | None = None) -> list[float]:
        use = params or self.params
        audio = [float(s) for s in frames]
        if not audio:
            return []
        if sample_rate <= 0:
            raise ValueError(f"bad sample rate {sample_rate}")
        # 1. pre-resample to the model rate.
        audio = _resample(audio, sample_rate, use.model_rate)
        model_rate_in = list(audio)
        if use.skip_vocode:
            return audio
        if self.executor is None:
            raise ConversionUnavailable(
                "realtime conversion needs a model: pass --model "
                "<torchscript file> or --service <slug@version> "
                "(nm voice convert --export-probe writes a probe model)")
        # 2. pad to a whole number of 4 frames (5 ms frames).
        frame_samples = max(1, use.model_rate * 5 // 1000)
        block = 4 * frame_samples
        if len(audio) % block:
            audio = audio + [0.0] * (block - len(audio) % block)
        # 3. pitch map (log-F0 mean/std, first order).
        shift = use.pitch_shift()
        if shift:
            audio = _pitch_shift(audio, use.model_rate, shift)
        # 4. spectral features.
        magnitude, phase, n_samples = _stft(audio, use)
        # 5. the model, on the magnitude.
        import numpy as np
        batched = np.asarray(magnitude, dtype=np.float32)[np.newaxis, :, :]
        result = self.executor.execute({"magnitude": batched})
        if "magnitude" not in result:
            raise ValueError("conversion model returned no 'magnitude' "
                             "output")
        converted = np.asarray(result["magnitude"], dtype=np.float32)
        if converted.shape != batched.shape:
            raise ValueError(
                "conversion model changed the magnitude shape: "
                f"{list(converted.shape)} != {list(batched.shape)}")
        converted = np.clip(converted, 0.0, None)
        # 6. resynthesis with the original phase (or Griffin-Lim).
        if use.griffin_iters > 0:
            audio = _griffinlim(converted[0], use, n_samples)
        else:
            audio = _istft(converted[0], phase, use, n_samples)
        # 7. post-resample to the output rate.
        out_rate = use.output_rate or sample_rate
        audio = _resample(audio, use.model_rate, out_rate)
        if use.discard_vocoded_audio:
            return model_rate_in
        return audio


def _int16_bytes_to_floats(chunk: bytes) -> list[float]:
    samples = array("h")
    samples.frombytes(chunk[:len(chunk) // 2 * 2])
    if sys.byteorder != "little":
        samples.byteswap()
    return [s / 32768.0 for s in samples]


def _floats_to_int16_bytes(frames: Any) -> bytes:
    samples = array(
        "h",
        (max(-32768, min(32767, int(round(s * 32768.0))))
         for s in frames))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def convert_file(input_path: str, output_path: str, *,
                 executor: Any = None,
                 params: ConversionParams | None = None,
                 cache: Any = None) -> dict[str, Any]:
    """Convert one wav file end to end (stdlib wav I/O).

    Returns the conversion receipt: input/output paths, rates, sample
    counts, and the executor used. The file is mono-mixed by the
    reader (the sidecar was mono too).

    With ``cache`` (a :class:`~nomorals.voice.cache.ConversionCache`)
    the conversion is keyed by model + input bytes + settings: a
    repeat conversion copies the cached wav out instead of re-running
    the pipeline, and the receipt says ``cached: True``. A corrupt
    cached entry is refused by name, dropped, and the conversion
    re-runs — the receipt names it under ``cache_corrupt``.
    """
    from .session import read_wav_bytes, write_wav_bytes
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"no such audio file: {input_path}")
    pcm, rate = read_wav_bytes(input_path)
    frames = _int16_bytes_to_floats(pcm)
    use = params or ConversionParams()
    cache_key = ""
    cache_corrupt = ""
    if cache is not None:
        from .cache import CorruptCacheEntry, executor_identity
        with open(input_path, "rb") as fh:
            input_sha = hashlib.sha256(fh.read()).hexdigest()
        cache_key = cache.key_for(
            model_identity=executor_identity(executor),
            input_sha256=input_sha, input_rate=rate, params=use)
        try:
            hit = cache.lookup(cache_key)
        except CorruptCacheEntry as exc:
            # Refused by name and already dropped: re-run the
            # conversion and replace the entry below.
            hit = None
            cache_corrupt = cache_key
            _log.warning("convert_file: %s", exc)
        if hit is not None:
            shutil.copyfile(hit["path"], output_path)
            return {"ok": True, "input": input_path,
                    "output": output_path,
                    "input_rate": rate,
                    "output_rate": int(hit["output_rate"]),
                    "input_samples": len(frames),
                    "output_samples": int(hit["output_samples"]),
                    "executor": type(executor).__name__
                    if executor else "none",
                    "skip_vocode": bool(use.skip_vocode),
                    "cached": True, "cache_key": cache_key}
    converter = Converter(executor, use)
    out_frames = converter.convert(frames, rate)
    out_rate = use.output_rate or rate
    if use.discard_vocoded_audio:
        out_rate = use.model_rate
    write_wav_bytes(output_path, _floats_to_int16_bytes(out_frames),
                    sample_rate=out_rate)
    receipt = {"ok": True, "input": input_path, "output": output_path,
               "input_rate": rate, "output_rate": out_rate,
               "input_samples": len(frames),
               "output_samples": len(out_frames),
               "executor": type(executor).__name__ if executor else "none",
               "skip_vocode": bool(use.skip_vocode),
               "cached": False, "cache_key": cache_key}
    if cache_corrupt:
        receipt["cache_corrupt"] = cache_corrupt
    if cache is not None:
        from .cache import executor_identity
        try:
            cache.store(
                cache_key, output_path,
                model_identity=executor_identity(executor),
                input_sha256=input_sha, input_rate=rate, params=use,
                input_samples=len(frames),
                output_samples=len(out_frames), output_rate=out_rate)
        except OSError as exc:
            # A cache that cannot write never fails a conversion
            # that otherwise succeeded — the failure is named on
            # the receipt instead.
            receipt["cache_store_error"] = str(exc)
    return receipt


# -- the sidecar server (ZeroMQ REP loop, transport-free) -------------------

class ConversionServer:
    """The Python sidecar, without the socket.

    ``handle(payload)`` is the REP loop's body: accumulate the request's
    audio, reply ``b"OK"`` while the buffer is below the minimum (the
    upstream did exactly this — the client treats a <=2-byte reply as
    "still buffering"), and convert + reply with a
    :class:`VocodeAudioResponse` once the buffer is full. A conversion
    failure drops the batch, records the error in the stats and the
    voice failure ledger, and still replies ``b"OK"`` so the REQ/REP
    lockstep never wedges — the error is visible in :meth:`status`,
    not hidden in a dead socket.
    """

    def __init__(self, converter: Converter, *, default_rate: int = 16000
                 ) -> None:
        self.converter = converter
        self.default_rate = default_rate
        self._queue: list[float] = []
        self._lock = threading.Lock()
        self._request_batches = 0
        self._response_batches = 0
        self._samples_in = 0
        self._samples_out = 0
        self._errors = 0
        self._last_error = ""

    def handle(self, payload: bytes) -> bytes:
        try:
            request = VocodeAudioRequest.from_bytes(payload)
        except ValueError as exc:
            raise ValueError(f"bad VocodeAudioRequest: {exc}") from exc
        rate = request.vocode_params.original_source_rate or self.default_rate
        with self._lock:
            self._queue.extend(request.float_audio)
            self._samples_in += len(request.float_audio)
            self._request_batches += 1
            minimum = (request.buffer_size_minimum
                       or self.converter.params.buffer_size_minimum)
            if len(self._queue) < minimum:
                return b"OK"
            batch = self._queue
            self._queue = []
        params = ConversionParams.from_vocode_params(
            request.vocode_params, request, base=self.converter.params)
        try:
            results = self.converter.convert(batch, rate, params=params)
        except Exception as exc:  # noqa: BLE001 — recorded, lockstep kept
            with self._lock:
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            record_failure("realtime.server.convert", exc)
            return b"OK"
        with self._lock:
            self._response_batches += 1
            self._samples_out += len(results)
            number = self._response_batches
        response = VocodeAudioResponse(
            float_audio=[float(x) for x in results],
            response_batch_number=number)
        return response.to_bytes()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"queued_samples": len(self._queue),
                    "request_batches": self._request_batches,
                    "response_batches": self._response_batches,
                    "samples_in": self._samples_in,
                    "samples_out": self._samples_out,
                    "errors": self._errors,
                    "last_error": self._last_error}


# -- transports ---------------------------------------------------------------

class InProcessClient:
    """The sidecar in the same process — a direct call, no socket."""

    def __init__(self, server: ConversionServer) -> None:
        self._server = server

    def request(self, payload: bytes) -> bytes:
        return self._server.handle(payload)

    def close(self) -> None:
        pass


def _recvall(sock: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("transport closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


class TcpClient:
    """Length-prefixed REQ/REP over a stdlib TCP socket.

    The framing is Devon's own (4-byte big-endian length + payload);
    the payload bytes are the proto bytes, identical to the ZeroMQ
    backend's. One request in flight at a time, like REQ/REP.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, *,
                 timeout: float = 120.0) -> None:
        self.host = host
        self.port = int(port)
        self._lock = threading.Lock()
        self._sock = socket.create_connection((host, int(port)),
                                              timeout=timeout)
        self._sock.settimeout(timeout)

    def request(self, payload: bytes) -> bytes:
        with self._lock:
            self._sock.sendall(struct.pack(">I", len(payload)) + payload)
            header = _recvall(self._sock, 4)
            (size,) = struct.unpack(">I", header)
            if size > 64 * 1024 * 1024:
                raise ValueError(f"reply too large: {size} bytes")
            return _recvall(self._sock, size)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._sock.close()


def _serve_tcp_conn(conn: socket.socket, server: ConversionServer,
                    timeout: float) -> None:
    with conn:
        conn.settimeout(timeout)
        while True:
            try:
                header = _recvall(conn, 4)
            except (TimeoutError, ConnectionError, OSError):
                return
            (size,) = struct.unpack(">I", header)
            if size > 64 * 1024 * 1024:
                return
            try:
                payload = _recvall(conn, size)
            except (TimeoutError, ConnectionError, OSError):
                return
            try:
                reply = server.handle(payload)
            except Exception as exc:  # noqa: BLE001 — bad request, keep serving
                record_failure("realtime.serve_tcp", exc)
                reply = b"OK"
            try:
                conn.sendall(struct.pack(">I", len(reply)) + reply)
            except (TimeoutError, ConnectionError, OSError):
                return


def serve_tcp(server: ConversionServer, host: str = "127.0.0.1",
              port: int = 5555, *, stop_event: threading.Event | None = None,
              timeout: float = 120.0) -> None:
    """Serve the sidecar over TCP until ``stop_event`` is set."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))
    listener.listen(8)
    listener.settimeout(0.5)
    _log.info("realtime sidecar serving TCP on %s:%d", host, port)
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            thread = threading.Thread(
                target=_serve_tcp_conn, args=(conn, server, timeout),
                daemon=True, name="realtime-tcp-conn")
            thread.start()
    finally:
        listener.close()


def _require_zmq(what: str) -> Any:
    try:
        import zmq
    except ImportError as exc:
        raise ConversionUnavailable(
            f"{what} needs pyzmq, which is not installed on this "
            "machine — pip install pyzmq (the ZeroMQ transport is the "
            "upstream-faithful backend; TCP and in-process work "
            "without it)") from exc
    return zmq


class ZeroMqClient:
    """The upstream-faithful transport: proto bytes over ZeroMQ REQ."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555", *,
                 timeout_ms: int = 120000) -> None:
        zmq = _require_zmq("the ZeroMQ client")
        self.endpoint = endpoint
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._sock.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._sock.connect(endpoint)

    def request(self, payload: bytes) -> bytes:
        self._sock.send(payload)
        return bytes(self._sock.recv())

    def close(self) -> None:
        try:
            self._sock.close()
        finally:
            self._ctx.term()


def serve_zeromq(server: ConversionServer,
                 endpoint: str = "tcp://*:5555", *,
                 stop_event: threading.Event | None = None) -> None:
    """Serve the sidecar over ZeroMQ REP until ``stop_event`` is set."""
    zmq = _require_zmq("the ZeroMQ server")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(endpoint)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    _log.info("realtime sidecar serving ZeroMQ on %s", endpoint)
    try:
        while stop_event is None or not stop_event.is_set():
            if not poller.poll(timeout=500):
                continue
            payload = bytes(sock.recv())
            try:
                reply = server.handle(payload)
            except Exception as exc:  # noqa: BLE001 — bad request, keep serving
                record_failure("realtime.serve_zeromq", exc)
                reply = b"OK"
            sock.send(reply)
    finally:
        sock.close()
        ctx.term()


def make_client(transport: str, *, server: ConversionServer | None = None,
                host: str = "127.0.0.1", port: int = 5555,
                endpoint: str = "", timeout: float = 120.0) -> Any:
    """A request client for a transport: inproc | tcp | zeromq."""
    kind = (transport or "").strip().lower()
    if kind in ("inproc", "in-process", "direct"):
        if server is None:
            raise ValueError("the in-process transport needs a server")
        return InProcessClient(server)
    if kind == "tcp":
        return TcpClient(host, int(port), timeout=timeout)
    if kind in ("zeromq", "zmq"):
        return ZeroMqClient(endpoint or f"tcp://{host}:{port}")
    raise ValueError(f"unknown transport {transport!r} — "
                     "inproc, tcp, or zeromq")


# -- the live loop (Rust main + sidecar thread) -------------------------------

class RealtimeConverter:
    """The live loop: mic -> gate -> sidecar -> speaker.

    The Rust shell's shape, on the organ's own hardware rungs: the
    mic comes from ``default_mic`` (a ``MicCapture`` thread owns it),
    the speaker from ``default_speaker`` (a ``SpeakerSink`` that plays
    wav files). Converted batches are played per batch as wav files
    through the sink — the upstream played from a queue continuously,
    but the sink's contract is files, and a batch's latency (buffer /
    model rate, ~4.4s at the defaults) is the same shape the upstream
    measured. Mic and speaker are injected so tests can drive the
    whole loop against fakes at the hardware boundary.
    """

    def __init__(self, *, mic: Any, speaker: Any, client: Any,
                 params: ConversionParams | None = None,
                 data_dir: str = "", send_size: int = SEND_SIZE) -> None:
        self.mic = mic
        self.speaker = speaker
        self.client = client
        self.params = params or ConversionParams()
        self.data_dir = data_dir or "voice_data"
        self.send_size = max(1, int(send_size))

    def run(self, *, seconds: float = 0.0,
            stop_event: threading.Event | None = None) -> dict[str, Any]:
        from .session import MicCapture, write_wav_bytes
        if self.mic is None:
            raise ConversionUnavailable(
                "no microphone available on this device")
        if self.speaker is None:
            raise ConversionUnavailable(
                "no speaker available on this device")
        source_rate = int(getattr(self.mic, "sample_rate", 16000) or 16000)
        out_rate = self.params.output_rate or source_rate
        capture = MicCapture(self.mic)
        capture.start()
        gate = SpeechGate()
        incoming = AudioFrameQueue(
            max_samples=self.send_size * 64)
        outgoing = AudioFrameQueue(
            max_samples=self.params.buffer_size_minimum * 4)
        stats: dict[str, Any] = {
            "batches_sent": 0, "batches_received": 0,
            "samples_captured": 0, "samples_played": 0,
            "transport_errors": 0, "activations": 0,
            "dropped_incoming": 0, "dropped_outgoing": 0,
            "end_reason": "stopped",
        }
        deadline = (time.monotonic() + seconds) if seconds > 0 else None
        played: list[tuple[Any, str]] = []
        batch_no = 0
        play_no = 0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    stats["end_reason"] = "stopped"
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    stats["end_reason"] = "time"
                    break
                if capture.ended:
                    stats["end_reason"] = "mic-ended"
                    break
                chunk = capture.poll(timeout=0.05)
                if chunk is not None:
                    frames = _int16_bytes_to_floats(chunk)
                    incoming.extend(frames)
                    stats["samples_captured"] += len(frames)
                while True:
                    frame = incoming.drain_size(self.send_size)
                    if frame is None:
                        break
                    to_send = gate.feed(frame)
                    if not to_send:
                        continue
                    batch_no += 1
                    request = VocodeAudioRequest(
                        float_audio=to_send,
                        request_batch_number=batch_no,
                        vocode_params=self.params.to_vocode_params(
                            source_rate=source_rate),
                        skip_vocode=self.params.skip_vocode,
                        discard_vocoded_audio=(
                            self.params.discard_vocoded_audio),
                        buffer_size_minimum=self.params.buffer_size_minimum)
                    try:
                        reply = self.client.request(request.to_bytes())
                    except Exception as exc:  # noqa: BLE001 — recorded, loop continues
                        record_failure("realtime.live.request", exc)
                        stats["transport_errors"] += 1
                        continue
                    stats["batches_sent"] += 1
                    if len(reply) > 2:
                        response = VocodeAudioResponse.from_bytes(reply)
                        outgoing.extend(response.float_audio)
                        stats["batches_received"] += 1
                play_slice = int(out_rate * 1.0)
                while len(outgoing) >= play_slice:
                    frames = outgoing.drain_size(play_slice)
                    if frames is None:
                        break
                    play_no += 1
                    path = os.path.join(
                        self.data_dir, "tmp",
                        f"convert-live-{play_no}.wav")
                    write_wav_bytes(path, _floats_to_int16_bytes(frames),
                                    sample_rate=out_rate)
                    token = self.speaker.play(path)
                    played.append((token, path))
                    stats["samples_played"] += len(frames)
                still: list[tuple[Any, str]] = []
                for token, path in played:
                    if self.speaker.playing(token):
                        still.append((token, path))
                    else:
                        with contextlib.suppress(OSError):
                            os.unlink(path)
                played = still
        finally:
            capture.stop()
            with contextlib.suppress(Exception):  # noqa: BLE001 — best effort on shutdown
                self.mic.close()
            stats["activations"] = gate.activations
            stats["dropped_incoming"] = incoming.dropped
            stats["dropped_outgoing"] = outgoing.dropped
            # Play the tail: what converted but never reached a full
            # 1s slice still gets heard.
            rest = outgoing.drain()
            if rest:
                play_no += 1
                path = os.path.join(self.data_dir, "tmp",
                                    f"convert-live-{play_no}.wav")
                try:
                    write_wav_bytes(path, _floats_to_int16_bytes(rest),
                                    sample_rate=out_rate)
                    self.speaker.play(path)
                    stats["samples_played"] += len(rest)
                except Exception as exc:  # noqa: BLE001 — tail is best effort
                    record_failure("realtime.live.tail", exc)
        return stats


# -- file conversion, status, chat --------------------------------------------

def conversion_status(*, services_dir: str = "") -> dict[str, Any]:
    """What the conversion rungs can do on this machine, as data."""
    from .vits._torch import torch_available
    services: list[str] | None = None
    try:
        from .services import ServiceRegistry
        registry = (ServiceRegistry(services_dir)
                    if services_dir else ServiceRegistry())
        services = [d.ref for d in registry.list()
                    if d.task == "voice-conversion"]
    except ImportError:
        services = None
    cache_status: dict[str, Any] | None = None
    try:
        from .cache import default_conversion_cache
        cache_status = default_conversion_cache().status()
    except OSError:
        cache_status = None
    return {
        "torch": torch_available(),
        "librosa": _librosa_available(),
        "scipy": _scipy_available(),
        "pyzmq": _pyzmq_available(),
        "probe_model": os.path.isfile(default_probe_path()),
        "probe_path": default_probe_path(),
        "services": services,
        "cache": cache_status,
    }


def status_text(status: dict[str, Any] | None = None) -> str:
    """The conversion readiness report, as human-readable text."""
    status = status if status is not None else conversion_status()
    lines = ["realtime voice conversion:"]
    lines.append(f"  torch: {'yes' if status['torch'] else 'no'}")
    lines.append(f"  librosa: {'yes' if status['librosa'] else 'no'}")
    lines.append(f"  scipy: {'yes' if status['scipy'] else 'no'}")
    lines.append(f"  pyzmq: {'yes' if status['pyzmq'] else 'no'}")
    lines.append(f"  probe model: {'yes' if status['probe_model'] else 'no'}"
                 f" ({status['probe_path']})")
    if status["services"] is None:
        lines.append("  services: registry not on this build yet")
    elif not status["services"]:
        lines.append("  services: none registered for voice-conversion")
    else:
        lines.append("  services: " + ", ".join(status["services"]))
    cache = status.get("cache")
    if cache is None:
        lines.append("  cache: unavailable")
    else:
        lines.append(
            f"  cache: {cache['entries']} entries, "
            f"{cache['bytes']} bytes, {cache['hits']} hits / "
            f"{cache['misses']} misses, {cache['evictions']} evicted "
            f"({cache['dir']})")
    if not status["torch"] or not status["librosa"]:
        lines.append("  full conversion needs torch + librosa; "
                     "--skip-vocode converts nothing but exercises the "
                     "transport + resampling path")
    return "\n".join(lines)


def control_convert(tail: str, *, message: Any = None,
                    data_dir: str = "") -> dict[str, Any]:
    """``/voice convert`` — the chat verb for file conversion.

    ``/voice convert <path>`` converts an attached voice note or a
    file path and returns ``{"ok", "text", "path"}``; the caller
    delivers ``path`` as a voice note. ``/voice convert status``
    reports the rungs. Chat never silently passes audio through: with
    no model available it says so, naming what is missing.
    """
    parts = (tail or "").strip().split()
    path = ""
    service_ref = ""
    model_path = ""
    direction = "A2B"
    no_cache = False
    index = 0
    while index < len(parts):
        token = parts[index]
        if token == "--service" and index + 1 < len(parts):
            service_ref = parts[index + 1]
            index += 2
        elif token == "--model" and index + 1 < len(parts):
            model_path = parts[index + 1]
            index += 2
        elif token == "--direction" and index + 1 < len(parts):
            direction = parts[index + 1].upper()
            index += 2
        elif token == "--no-cache":
            no_cache = True
            index += 1
        elif token in ("status", "health") and not path:
            return {"ok": True, "text": status_text(), "path": ""}
        elif token.startswith("--"):
            index += 1
        elif not path:
            path = token
            index += 1
        else:
            index += 1
    if not path and message is not None:
        for media in getattr(message, "media", None) or []:
            if getattr(media, "kind", "") in ("audio", "voice"):
                path = getattr(media, "path", "") or ""
                break
    if not path:
        return {"ok": False,
                "text": ("usage: /voice convert <audio path> "
                         "[--service <slug@version> | --model <path>] "
                         "[--direction A2B|B2A] [--no-cache] — attach a "
                         "voice note to the message or pass a file path\n"
                         + status_text()),
                "path": ""}
    if not os.path.isfile(path):
        return {"ok": False, "text": f"no such audio file: {path}",
                "path": ""}
    if direction not in ("A2B", "B2A"):
        return {"ok": False,
                "text": f"unknown direction {direction!r} — A2B or B2A",
                "path": ""}
    try:
        executor = resolve_executor(model_path=model_path,
                                    service_ref=service_ref)
    except ConversionUnavailable as exc:
        return {"ok": False, "text": str(exc), "path": ""}
    dest_dir = data_dir or tempfile.mkdtemp(prefix="voice-convert-")
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(
        dest_dir, f"converted-{int(time.time() * 1000)}.wav")
    cache = None
    if not no_cache:
        from .cache import default_conversion_cache
        cache = default_conversion_cache()
    try:
        receipt = convert_file(
            path, out_path, executor=executor,
            params=ConversionParams(direction=direction), cache=cache)
    except (ConversionUnavailable, ValueError, OSError) as exc:
        return {"ok": False, "text": f"conversion failed: {exc}",
                "path": ""}
    return {"ok": True,
            "text": (f"converted {receipt['input_samples']} samples "
                     f"@{receipt['input_rate']} Hz -> "
                     f"{receipt['output_samples']} samples "
                     f"@{receipt['output_rate']} Hz "
                     f"({receipt['executor']}, {direction})"
                     + (" — from cache" if receipt.get("cached")
                        else "")),
            "path": out_path}
