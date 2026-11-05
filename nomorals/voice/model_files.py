"""Model-file validation on ingest — the trust boundary for voice files.

Donor lesson, two parts. FakeYou's ``vocodes_model_check`` validated
user-uploaded model files *before they entered the system* and
blocked invalid files (its README's example: video files) at the
door. ``model-serde`` treated a model file as a *format* — metadata
plus payload, fixture-tested. Devon's cloned voices are the owner's
own assets, so the licensing/policy machinery of model-serde is
explicitly NOT ported (§5 of the integration plan); the format
discipline is:

- **Validate before load, before copy.** A file is sniffed from its
  magic bytes (never its extension), size-checked, structurally
  parsed, and hashed *before* anything ingests it. Today
  ``VoiceLibrary.upload_voice`` copied any bytes into managed
  storage under a ``.wav`` name — a video renamed ``.wav`` entered
  the voice library silently. Now it is refused at the door.
- **Refusals are coded, never crashes or silent skips.**
  :class:`ModelFileError` carries ``code="voice.model_file"`` and a
  machine-readable ``details["reason"]`` from a closed taxonomy
  (``missing_file``, ``empty_file``, ``too_large``,
  ``unsupported_format``, ``corrupt_header``, ``corrupt_metadata``,
  ``truncated_payload``, ``duration_out_of_range``). It also derives
  from ``ValueError`` so existing callers that treat ingest failures
  as value errors keep their shape.
- **Policy as data.** :class:`IngestPolicy` holds the limits; a
  caller can tighten them per surface.
- **Success is measured, not assumed.** A passing file returns a
  :class:`ModelFileInfo` — format as sniffed, byte size, SHA-256
  integrity hash, and (for WAV) the parsed sample rate / channels /
  duration. For formats stdlib cannot parse (FLAC/OGG/MP3 reference
  clips), ``seconds`` is None — unmeasured, recorded as such, never
  reconstructed.

Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import wave
from dataclasses import dataclass
from typing import Any, Optional

from ..core.errors import NoMoralsError

__all__ = [
    "ModelFileError", "IngestPolicy", "ModelFileInfo",
    "REASON_MISSING_FILE", "REASON_EMPTY_FILE", "REASON_TOO_LARGE",
    "REASON_UNSUPPORTED_FORMAT", "REASON_CORRUPT_HEADER",
    "REASON_CORRUPT_METADATA", "REASON_TRUNCATED_PAYLOAD",
    "REASON_DURATION_OUT_OF_RANGE",
    "sniff_format", "validate_reference_audio", "validate_model_file",
]

REASON_MISSING_FILE = "missing_file"
REASON_EMPTY_FILE = "empty_file"
REASON_TOO_LARGE = "too_large"
REASON_UNSUPPORTED_FORMAT = "unsupported_format"
REASON_CORRUPT_HEADER = "corrupt_header"
REASON_CORRUPT_METADATA = "corrupt_metadata"
REASON_TRUNCATED_PAYLOAD = "truncated_payload"
REASON_DURATION_OUT_OF_RANGE = "duration_out_of_range"


class ModelFileError(NoMoralsError, ValueError):
    """A voice/model file refused at ingest, with a coded reason."""

    code = "voice.model_file"

    def __init__(self, reason: str, message: str, *,
                 path: str = "",
                 details: Optional[dict[str, Any]] = None) -> None:
        merged = {"reason": reason, "path": path}
        if details:
            merged.update(details)
        super().__init__(message, details=merged)
        self.reason = reason
        self.path = path


@dataclass(frozen=True)
class IngestPolicy:
    """Ingest limits as data.

    The byte ceilings are the gate's defaults. Duration bounds are
    opt-in (0 = unbounded): clip length is a *quality* axis in this
    codebase — ``probe_reference_audio`` warns and the voice-print
    scores it — while this gate is the *structural* boundary, so a
    caller that wants duration enforced sets the bounds here and
    gets ``duration_out_of_range`` refusals.
    """

    #: Reference clips (cloned voices) — generous but bounded.
    max_audio_bytes: int = 256 * 1024 * 1024
    min_seconds: float = 0.0
    max_seconds: float = 0.0
    #: Model files (ONNX voice models).
    max_model_bytes: int = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class ModelFileInfo:
    """What validation measured about an accepted file."""

    path: str
    kind: str                      # "reference_audio" | "onnx_model"
    format: str                    # sniffed: wav|flac|ogg|mp3|onnx
    size_bytes: int
    sha256: str
    sample_rate: int = 0
    channels: int = 0
    seconds: Optional[float] = None   # None = unmeasured for this format

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind,
                "format": self.format, "size_bytes": self.size_bytes,
                "sha256": self.sha256, "sample_rate": self.sample_rate,
                "channels": self.channels, "seconds": self.seconds}


def sniff_format(header: bytes, *, suffix: str = "") -> str:
    """Format from magic bytes, never from the file name.

    Returns ``wav`` | ``flac`` | ``ogg`` | ``mp3`` | ``onnx`` |
    ``video`` | ``unknown``. ONNX is protobuf: a ModelProto starts
    with field 1 (``ir_version``, varint) → first byte ``0x08``;
    because a single byte is weak evidence, the ``.onnx`` suffix is
    required to call it — a bare ``0x08`` blob is ``unknown``.
    """
    h = header or b""
    if len(h) >= 12 and h[0:4] == b"RIFF" and h[8:12] == b"WAVE":
        return "wav"
    if len(h) >= 12 and h[0:4] == b"RIFF":
        return "video"                      # AVI and friends
    if h[0:4] == b"fLaC":
        return "flac"
    if h[0:4] == b"OggS":
        return "ogg"
    if h[0:3] == b"ID3" or (len(h) >= 2 and h[0] == 0xFF
                             and (h[1] & 0xE0) == 0xE0):
        return "mp3"
    if len(h) >= 8 and h[4:8] == b"ftyp":
        return "video"                      # MP4 / MOV / M4A family
    if h[0:4] == b"\x1a\x45\xdf\xa3":
        return "video"                      # Matroska / WebM
    if h[0:1] == b"\x08" and suffix.lower() == ".onnx":
        return "onnx"
    return "unknown"


def _head(path: str, n: int = 16) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(n)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_present_and_sized(path: str, max_bytes: int, *,
                             kind: str) -> int:
    """Shared gate: existence, non-empty, size ceiling."""
    if not path or not os.path.isfile(path):
        raise ModelFileError(
            REASON_MISSING_FILE,
            f"{kind} file not found: {path!r}", path=str(path or ""))
    size = os.path.getsize(path)
    if size == 0:
        raise ModelFileError(
            REASON_EMPTY_FILE,
            f"{kind} file is empty: {path!r}", path=path)
    if size > max_bytes:
        raise ModelFileError(
            REASON_TOO_LARGE,
            f"{kind} file is {size} bytes, over the "
            f"{max_bytes}-byte ingest limit: {path!r}",
            path=path, details={"size_bytes": size,
                                "max_bytes": max_bytes})
    return size


def _wav_structure(path: str, size: int) -> tuple[int, int, float]:
    """Parse a WAV's chunk table; return (rate, channels, seconds).

    Structural, not trusting: the RIFF chunk walk verifies the
    ``fmt `` chunk exists and the ``data`` chunk's declared size
    fits inside the actual file — a header that promises more audio
    than the file holds is a truncated payload, refused by name.
    """
    try:
        with open(path, "rb") as fh:
            riff = fh.read(12)
            if len(riff) < 12 or riff[0:4] != b"RIFF" \
                    or riff[8:12] != b"WAVE":
                raise ModelFileError(
                    REASON_CORRUPT_HEADER,
                    f"reference clip has no RIFF/WAVE header: {path!r}",
                    path=path)
            have_fmt = False
            data_size: Optional[int] = None
            while True:
                hdr = fh.read(8)
                if len(hdr) < 8:
                    break
                chunk_id, chunk_size = hdr[0:4], struct.unpack(
                    "<I", hdr[4:8])[0]
                payload_at = fh.tell()
                if chunk_id == b"fmt ":
                    have_fmt = True
                elif chunk_id == b"data":
                    data_size = chunk_size
                    if payload_at + chunk_size > size:
                        raise ModelFileError(
                            REASON_TRUNCATED_PAYLOAD,
                            f"reference clip declares {chunk_size} "
                            f"audio bytes but the file holds only "
                            f"{size - payload_at}: {path!r}",
                            path=path,
                            details={"declared_bytes": chunk_size,
                                     "actual_bytes": size - payload_at})
                fh.seek(payload_at + chunk_size
                        + (chunk_size & 1))
            if not have_fmt or data_size is None:
                raise ModelFileError(
                    REASON_CORRUPT_HEADER,
                    f"reference clip is missing its fmt/data chunks: "
                    f"{path!r}", path=path)
    except ModelFileError:
        raise
    except OSError as exc:
        raise ModelFileError(
            REASON_CORRUPT_HEADER,
            f"reference clip unreadable: {exc}", path=path) from exc
    try:
        with wave.open(path, "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            frames = wav.getnframes()
    except (wave.Error, EOFError, OSError) as exc:
        raise ModelFileError(
            REASON_CORRUPT_HEADER,
            f"reference clip fails WAV parsing: {exc}",
            path=path) from exc
    seconds = (frames / rate) if rate else 0.0
    return rate, channels, round(seconds, 3)


def validate_reference_audio(path: str,
                             policy: IngestPolicy | None = None
                             ) -> ModelFileInfo:
    """Validate a cloned/reference voice clip before ingest.

    WAV clips are fully parsed (structure; duration bounds when the
    policy sets them — see :class:`IngestPolicy`); FLAC/OGG/MP3
    clips pass on sniff + size + integrity hash with
    ``seconds=None`` (stdlib cannot parse them — unmeasured, never
    reconstructed). Video files and unrecognized bytes are refused
    as ``unsupported_format`` — the donor's door rule.
    """
    policy = policy or IngestPolicy()
    size = _check_present_and_sized(path, policy.max_audio_bytes,
                                    kind="reference audio")
    fmt = sniff_format(_head(path), suffix=os.path.splitext(path)[1])
    if fmt == "video":
        raise ModelFileError(
            REASON_UNSUPPORTED_FORMAT,
            f"reference clip is a video file, not audio: {path!r}",
            path=path, details={"sniffed_format": fmt})
    if fmt not in ("wav", "flac", "ogg", "mp3"):
        raise ModelFileError(
            REASON_UNSUPPORTED_FORMAT,
            f"reference clip format not recognized as audio "
            f"(sniffed: {fmt}): {path!r}",
            path=path, details={"sniffed_format": fmt})
    rate = channels = 0
    seconds: Optional[float] = None
    if fmt == "wav":
        rate, channels, seconds = _wav_structure(path, size)
        if (policy.min_seconds > 0 and seconds < policy.min_seconds) \
                or (policy.max_seconds > 0
                    and seconds > policy.max_seconds):
            raise ModelFileError(
                REASON_DURATION_OUT_OF_RANGE,
                f"reference clip is {seconds}s; the ingest policy "
                f"accepts {policy.min_seconds}–{policy.max_seconds}s: "
                f"{path!r}",
                path=path, details={"seconds": seconds,
                                    "min_seconds": policy.min_seconds,
                                    "max_seconds": policy.max_seconds})
    return ModelFileInfo(path=path, kind="reference_audio", format=fmt,
                         size_bytes=size, sha256=_sha256(path),
                         sample_rate=rate, channels=channels,
                         seconds=seconds)


def validate_model_file(path: str,
                        policy: IngestPolicy | None = None
                        ) -> ModelFileInfo:
    """Validate an ONNX voice model before it is loaded or served.

    Checks the protobuf shape, the size ceiling, and — when the
    companion ``<file>.json`` config is present (Piper voices ship
    one) — that the config parses and its sample rate is sane. A
    missing companion is recorded, not refused: the sample rate is
    then read from the loaded voice, as before.
    """
    policy = policy or IngestPolicy()
    size = _check_present_and_sized(path, policy.max_model_bytes,
                                    kind="model")
    fmt = sniff_format(_head(path), suffix=os.path.splitext(path)[1])
    if fmt != "onnx":
        raise ModelFileError(
            REASON_UNSUPPORTED_FORMAT,
            f"model file is not an ONNX model (sniffed: {fmt}): "
            f"{path!r}",
            path=path, details={"sniffed_format": fmt})
    rate = 0
    companion = path + ".json"
    if os.path.isfile(companion):
        try:
            with open(companion, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ModelFileError(
                REASON_CORRUPT_METADATA,
                f"model config {companion!r} does not parse: {exc}",
                path=path) from exc
        raw_rate = None
        if isinstance(cfg, dict):
            audio = cfg.get("audio")
            if isinstance(audio, dict):
                raw_rate = audio.get("sample_rate")
            if raw_rate is None:
                raw_rate = cfg.get("sample_rate")
        if raw_rate is not None:
            if not isinstance(raw_rate, (int, float)) or raw_rate <= 0:
                raise ModelFileError(
                    REASON_CORRUPT_METADATA,
                    f"model config {companion!r} has an invalid "
                    f"sample_rate: {raw_rate!r}", path=path)
            rate = int(raw_rate)
    return ModelFileInfo(path=path, kind="onnx_model", format="onnx",
                         size_bytes=size, sha256=_sha256(path),
                         sample_rate=rate)
