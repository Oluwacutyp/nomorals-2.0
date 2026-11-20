"""ctypes bindings for libk4arecord — the recording half of k4a-sys.

The upstream echelon/k4a-sys crate (MIT, Copyright 2020 Amédée
d'Aboville) only binds ``k4a.h``: its ``wrapper.h`` is one line,
``#include <k4a/k4a.h>``, and bindgen never sees the vendored
``k4arecord/`` headers. The temp port (lane 122) inherited that
boundary. This module is the other half of the port: the same C
API (``k4arecord/record.h`` + ``playback.h`` + ``types.h``, SDK 1.3,
vendored under ``rs/k4a-sys/vendor/include/linux/k4arecord/``)
expressed as ctypes structs and function prototypes, with the same
names and the same field order.

Two libraries, two loads: ``libk4a`` owns devices, captures and
images; ``libk4arecord`` owns recording files and playback. A
``Playback`` still needs libk4a to decode the captures it reads,
so it loads that library lazily on the first capture read — tag,
track and configuration reads work with the record library alone.

Loading: ``libk4arecord.so.1.4`` first, then the unversioned
``libk4arecord.so``, then the 1.3 soname the vendored crate was
written against. ``K4A_RECORD_LIB`` names an explicit library path
and is authoritative when set. A missing library raises
:class:`NoSdkError` naming every soname tried — never a silent
skip, never fake success.
"""

from __future__ import annotations

import ctypes
import os
import re
from ctypes import (
    POINTER,
    byref,
    c_bool,
    c_char,
    c_char_p,
    c_int,
    c_int32,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import TracebackType

from .bindings import (
    Calibration as CCalibration,
)
from .bindings import (
    DeviceConfiguration,
    ImuSample,
    K4aLibrary,
    _DestroyCallback,
    load_library,
)
from .enums import (
    ColorResolution,
    DepthMode,
    Fps,
    ImageFormat,
    WiredSyncMode,
)
from .errors import (
    NoSdkError,
    PlaybackError,
    RecordError,
)
from .frames import Capture, Frame

__all__ = [
    "RECORD_SONAMES",
    "TRACK_COLOR",
    "TRACK_DEPTH",
    "TRACK_IMU",
    "TRACK_IR",
    "DataBlock",
    "K4aRecordLibrary",
    "Playback",
    "PlaybackSeekOrigin",
    "RecordConfiguration",
    "RecordSubtitleSettings",
    "RecordVideoSettings",
    "Recorder",
    "StreamResult",
    "load_record_library",
    "record_live",
    "validate_tag_name",
]

#: Sonames tried in order when loading libk4arecord.
RECORD_SONAMES = ("libk4arecord.so.1.4", "libk4arecord.so",
                  "libk4arecord.so.1.3")

#: Built-in track names (k4arecord/types.h).
TRACK_COLOR = "COLOR"
TRACK_DEPTH = "DEPTH"
TRACK_IR = "IR"
TRACK_IMU = "IMU"

#: Tag / track names are ALL CAPS and only A-Z, 0-9, '-' and '_'.
_TAG_NAME_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-_]*$")


def validate_tag_name(name: str) -> str:
    """Enforce the SDK's tag/track naming rule, failing fast."""
    if not isinstance(name, str) or not _TAG_NAME_RE.match(name):
        raise RecordError(
            "tag and track names must be ALL CAPS and only contain "
            f"A-Z, 0-9, '-' and '_', got {name!r}")
    return name


class StreamResult(IntEnum):
    """k4a_stream_result_t."""

    SUCCEEDED = 0
    FAILED = 1
    EOF = 2


class PlaybackSeekOrigin(IntEnum):
    """k4a_playback_seek_origin_t."""

    BEGIN = 0
    END = 1
    DEVICE_TIME = 2


class RecordConfiguration(ctypes.Structure):
    """k4a_record_configuration_t — the config a recording was made with."""

    _fields_ = [
        ("color_format", c_int),
        ("color_resolution", c_int),
        ("depth_mode", c_int),
        ("camera_fps", c_int),
        ("color_track_enabled", c_bool),
        ("depth_track_enabled", c_bool),
        ("ir_track_enabled", c_bool),
        ("imu_track_enabled", c_bool),
        ("depth_delay_off_color_usec", c_int32),
        ("wired_sync_mode", c_int),
        ("subordinate_delay_off_master_usec", c_uint32),
        ("start_timestamp_offset_usec", c_uint32),
    ]

    def as_dict(self) -> dict[str, object]:
        """The configuration as plain data, enums by name."""
        return {
            "color_format": ImageFormat(int(self.color_format)).name,
            "color_resolution": ColorResolution(
                int(self.color_resolution)).name,
            "depth_mode": DepthMode(int(self.depth_mode)).name,
            "camera_fps": Fps(int(self.camera_fps)).name,
            "color_track_enabled": bool(self.color_track_enabled),
            "depth_track_enabled": bool(self.depth_track_enabled),
            "ir_track_enabled": bool(self.ir_track_enabled),
            "imu_track_enabled": bool(self.imu_track_enabled),
            "depth_delay_off_color_usec": int(
                self.depth_delay_off_color_usec),
            "wired_sync_mode": WiredSyncMode(
                int(self.wired_sync_mode)).name,
            "subordinate_delay_off_master_usec": int(
                self.subordinate_delay_off_master_usec),
            "start_timestamp_offset_usec": int(
                self.start_timestamp_offset_usec),
        }


class RecordVideoSettings(ctypes.Structure):
    """k4a_record_video_settings_t — width/height/frame-rate of a track."""

    _fields_ = [
        ("width", c_uint64),
        ("height", c_uint64),
        ("frame_rate", c_uint64),
    ]

    def as_dict(self) -> dict[str, int]:
        """The settings as plain data."""
        return {"width": int(self.width), "height": int(self.height),
                "frame_rate": int(self.frame_rate)}


class RecordSubtitleSettings(ctypes.Structure):
    """k4a_record_subtitle_settings_t — batched timestamps or not."""

    _fields_ = [("high_freq_data", c_bool)]


#: (name, argtypes, restype) for every k4arecord export, in header order.
#: record.h first (12), then playback.h (28 — the 4 k4a_capture_*
#: re-declarations in playback.h belong to libk4a and are bound there).
_FUNCTIONS: tuple[tuple[str, list, object], ...] = (
    ("k4a_record_create",
     [c_char_p, c_void_p, DeviceConfiguration, POINTER(c_void_p)],
     c_int),
    ("k4a_record_add_tag", [c_void_p, c_char_p, c_char_p], c_int),
    ("k4a_record_add_imu_track", [c_void_p], c_int),
    ("k4a_record_add_attachment",
     [c_void_p, c_char_p, POINTER(c_uint8), c_size_t], c_int),
    ("k4a_record_add_custom_video_track",
     [c_void_p, c_char_p, c_char_p, POINTER(c_uint8), c_size_t,
      POINTER(RecordVideoSettings)], c_int),
    ("k4a_record_add_custom_subtitle_track",
     [c_void_p, c_char_p, c_char_p, POINTER(c_uint8), c_size_t,
      POINTER(RecordSubtitleSettings)], c_int),
    ("k4a_record_write_header", [c_void_p], c_int),
    ("k4a_record_write_capture", [c_void_p, c_void_p], c_int),
    ("k4a_record_write_imu_sample", [c_void_p, ImuSample], c_int),
    ("k4a_record_write_custom_track_data",
     [c_void_p, c_char_p, c_uint64, POINTER(c_uint8), c_size_t],
     c_int),
    ("k4a_record_flush", [c_void_p], c_int),
    ("k4a_record_close", [c_void_p], None),
    ("k4a_playback_open", [c_char_p, POINTER(c_void_p)], c_int),
    ("k4a_playback_get_raw_calibration",
     [c_void_p, POINTER(c_uint8), POINTER(c_size_t)], c_int),
    ("k4a_playback_get_calibration",
     [c_void_p, POINTER(CCalibration)], c_int),
    ("k4a_playback_get_record_configuration",
     [c_void_p, POINTER(RecordConfiguration)], c_int),
    ("k4a_playback_check_track_exists",
     [c_void_p, c_char_p], c_bool),
    ("k4a_playback_get_track_count", [c_void_p], c_size_t),
    ("k4a_playback_get_track_name",
     [c_void_p, c_size_t, c_char_p, POINTER(c_size_t)], c_int),
    ("k4a_playback_track_is_builtin",
     [c_void_p, c_char_p], c_bool),
    ("k4a_playback_track_get_video_settings",
     [c_void_p, c_char_p, POINTER(RecordVideoSettings)], c_int),
    ("k4a_playback_track_get_codec_id",
     [c_void_p, c_char_p, c_char_p, POINTER(c_size_t)], c_int),
    ("k4a_playback_track_get_codec_context",
     [c_void_p, c_char_p, POINTER(c_uint8), POINTER(c_size_t)],
     c_int),
    ("k4a_playback_get_tag",
     [c_void_p, c_char_p, c_char_p, POINTER(c_size_t)], c_int),
    ("k4a_playback_set_color_conversion", [c_void_p, c_int], c_int),
    ("k4a_playback_get_attachment",
     [c_void_p, c_char_p, POINTER(c_uint8), POINTER(c_size_t)],
     c_int),
    ("k4a_playback_get_next_capture",
     [c_void_p, POINTER(c_void_p)], c_int),
    ("k4a_playback_get_previous_capture",
     [c_void_p, POINTER(c_void_p)], c_int),
    ("k4a_playback_get_next_imu_sample",
     [c_void_p, POINTER(ImuSample)], c_int),
    ("k4a_playback_get_previous_imu_sample",
     [c_void_p, POINTER(ImuSample)], c_int),
    ("k4a_playback_get_next_data_block",
     [c_void_p, c_char_p, POINTER(c_void_p)], c_int),
    ("k4a_playback_get_previous_data_block",
     [c_void_p, c_char_p, POINTER(c_void_p)], c_int),
    ("k4a_playback_data_block_get_device_timestamp_usec",
     [c_void_p], c_uint64),
    ("k4a_playback_data_block_get_buffer_size",
     [c_void_p], c_size_t),
    ("k4a_playback_data_block_get_buffer",
     [c_void_p], POINTER(c_uint8)),
    ("k4a_playback_data_block_release", [c_void_p], None),
    ("k4a_playback_seek_timestamp",
     [c_void_p, c_int64, c_int], c_int),
    ("k4a_playback_get_recording_length_usec",
     [c_void_p], c_uint64),
    ("k4a_playback_get_last_timestamp_usec", [c_void_p], c_uint64),
    ("k4a_playback_close", [c_void_p], None),
)


class K4aRecordLibrary:
    """A loaded libk4arecord with every k4arecord export bound.

    Attribute access is the C function: ``lib.k4a_record_create(...)``.
    Missing symbols fail at load time, never at first call.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lib = ctypes.CDLL(path)
        self.functions: list[str] = []
        for name, argtypes, restype in _FUNCTIONS:
            func = getattr(self._lib, name, None)
            if func is None:
                raise NoSdkError(
                    f"{path} has no symbol {name}: not a k4a record "
                    "library")
            func.argtypes = argtypes
            func.restype = restype
            setattr(self, name, func)
            self.functions.append(name)


def load_record_library(sonames: tuple[str, ...] = RECORD_SONAMES
                        ) -> K4aRecordLibrary:
    """Load libk4arecord, trying each soname in order.

    ``K4A_RECORD_LIB`` names an explicit library path: when it is
    set it is authoritative — a path that does not exist fails
    loudly naming it, instead of silently falling back to a
    different library. Raises :class:`NoSdkError` naming every
    name tried when nothing loads.
    """
    tried: list[str] = []
    override_text = os.environ.get("K4A_RECORD_LIB", "")
    override = Path(override_text) if override_text else None
    if override is not None:
        if not override.exists():
            raise NoSdkError(
                "K4A_RECORD_LIB points at a missing library: "
                f"{override}")
        tried.append(str(override))
        return K4aRecordLibrary(str(override))
    for name in sonames:
        tried.append(name)
        try:
            return K4aRecordLibrary(name)
        except (OSError, NoSdkError):
            continue
    raise NoSdkError(
        "libk4arecord not found (tried: " + ", ".join(tried) + "). "
        "Install the Azure Kinect Sensor SDK (libk4arecord1.4) or "
        "set K4A_RECORD_LIB to the library path.")


def _check(result: int, what: str) -> None:
    if result != 0:  # K4A_RESULT_SUCCEEDED
        raise RecordError(f"{what} failed (k4a result {result})")


def _check_playback(result: int, what: str) -> None:
    if result != 0:  # K4A_RESULT_SUCCEEDED
        raise PlaybackError(f"{what} failed (k4a result {result})")


def _two_call_string(read, *, what: str) -> str | None:
    """A two-call string read: NULL first to size, then the buffer.

    Returns None when the SDK reports FAILED (the value does not
    exist); raises on any other failure.
    """
    size = c_size_t(0)
    result = read(None, byref(size))
    if result == 1:  # K4A_BUFFER_RESULT_FAILED — absent
        return None
    if result not in (0, 2):
        raise PlaybackError(
            f"{what} failed (k4a buffer result {result})")
    if size.value == 0:
        return ""
    buf = (c_char * size.value)()
    result = read(buf, byref(size))
    if result != 0:
        raise PlaybackError(
            f"{what} failed (k4a buffer result {result})")
    return buf.value.decode("utf-8", "replace")


def _two_call_bytes(read, *, what: str) -> bytes | None:
    """A two-call byte read: NULL first to size, then the buffer.

    Returns None when the SDK reports FAILED (the value does not
    exist); raises on any other failure.
    """
    size = c_size_t(0)
    result = read(None, byref(size))
    if result == 1:  # K4A_BUFFER_RESULT_FAILED — absent
        return None
    if result not in (0, 2):
        raise PlaybackError(
            f"{what} failed (k4a buffer result {result})")
    if size.value == 0:
        return b""
    buf = (c_uint8 * size.value)()
    result = read(buf, byref(size))
    if result != 0:
        raise PlaybackError(
            f"{what} failed (k4a buffer result {result})")
    return bytes(buf[:size.value])


def _imu_sample_from_dict(sample: dict) -> ImuSample:
    """A get_imu_sample-style dict -> the C struct."""
    from .bindings import Float3

    raw = ImuSample()
    raw.temperature = float(sample.get("temperature_c") or 0.0)
    acc = sample.get("acc") or (0.0, 0.0, 0.0)
    raw.acc_sample = Float3(float(acc[0]), float(acc[1]),
                            float(acc[2]))
    raw.acc_timestamp_usec = int(
        sample.get("acc_timestamp_usec") or 0)
    gyro = sample.get("gyro") or (0.0, 0.0, 0.0)
    raw.gyro_sample = Float3(float(gyro[0]), float(gyro[1]),
                             float(gyro[2]))
    raw.gyro_timestamp_usec = int(
        sample.get("gyro_timestamp_usec") or 0)
    return raw


def _imu_sample_to_dict(sample: ImuSample) -> dict[str, object]:
    """The C struct -> a get_imu_sample-style dict."""
    return {
        "temperature_c": float(sample.temperature),
        "acc": (float(sample.acc_sample.x),
                float(sample.acc_sample.y),
                float(sample.acc_sample.z)),
        "acc_timestamp_usec": int(sample.acc_timestamp_usec),
        "gyro": (float(sample.gyro_sample.x),
                 float(sample.gyro_sample.y),
                 float(sample.gyro_sample.z)),
        "gyro_timestamp_usec": int(sample.gyro_timestamp_usec),
    }


@dataclass(frozen=True)
class DataBlock:
    """One block of custom-track data read back from a recording."""

    track: str
    device_timestamp_usec: int
    data: bytes = b""


class Recorder:
    """A recording file open for writing (k4a_record_t).

    Create → add tags / tracks (before the header) → write header
    → write captures / IMU samples / custom data → flush → close.
    Captures must be written in increasing timestamp order and the
    header must be written first; both are enforced here with a
    named error instead of a bare SDK failure.
    """

    def __init__(self, lib: K4aRecordLibrary, path: str,
                 handle: int,
                 k4a_lib: K4aLibrary | None = None) -> None:
        self._lib = lib
        self._path = path
        self._handle = handle
        self._k4a_lib = k4a_lib
        self._closed = False
        self._header_written = False
        self._frames_written = 0
        self._imu_written = 0

    @classmethod
    def create(cls, path: str, config=None, *,
               lib: K4aRecordLibrary | None = None,
               device_handle: int | None = None,
               tags: dict[str, str] | None = None,
               k4a_lib: K4aLibrary | None = None) -> Recorder:
        """Open a new recording file for writing.

        ``device_handle`` is a live device's raw handle (or None to
        record user-generated data with no device — the recording
        then carries no device calibration or serial number).
        ``tags`` are written before the header, as the SDK requires.
        The file is created if missing, overwritten if present.
        """
        from .device import DeviceConfig

        record_lib = lib if lib is not None else load_record_library()
        cfg = (config or DeviceConfig()).to_ctypes()
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = c_void_p()
        result = record_lib.k4a_record_create(
            os.fsencode(str(target)), device_handle, cfg,
            byref(handle))
        if result != 0 or not handle.value:
            raise RecordError(
                f"could not create recording {target} "
                f"(k4a result {result})")
        recorder = cls(record_lib, str(target), handle.value,
                       k4a_lib=k4a_lib)
        for name, value in (tags or {}).items():
            recorder.add_tag(name, value)
        return recorder

    # -- properties ---------------------------------------------------

    @property
    def path(self) -> str:
        """The recording file path."""
        return self._path

    @property
    def is_open(self) -> bool:
        """True while the recording handle is held."""
        return not self._closed

    @property
    def header_written(self) -> bool:
        """True once the header (and metadata) is on disk."""
        return self._header_written

    @property
    def frames_written(self) -> int:
        """Captures written so far."""
        return self._frames_written

    @property
    def imu_samples_written(self) -> int:
        """IMU samples written so far."""
        return self._imu_written

    def _require_open(self) -> int:
        if self._closed:
            raise RecordError("recording is closed")
        return self._handle

    def _require_before_header(self, what: str) -> None:
        if self._header_written:
            raise RecordError(
                f"{what} must be added before the header is written")

    def _require_header(self, what: str) -> None:
        if not self._header_written:
            raise RecordError(
                f"{what} needs the header written first — call "
                "write_header()")

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Flush and close the recording (idempotent)."""
        if self._closed:
            return
        self._lib.k4a_record_close(self._handle)
        self._closed = True

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    # -- header contents -------------------------------------------------

    def add_tag(self, name: str, value: str) -> None:
        """Add a global tag (name must be ALL CAPS)."""
        validate_tag_name(name)
        self._require_before_header("tags")
        _check(self._lib.k4a_record_add_tag(
            self._require_open(), name.encode("utf-8"),
            str(value).encode("utf-8")), f"add tag {name}")

    def add_imu_track(self) -> None:
        """Add the track header for recording IMU samples."""
        self._require_before_header("the IMU track")
        _check(self._lib.k4a_record_add_imu_track(
            self._require_open()), "add IMU track")

    def add_attachment(self, name: str, data: bytes) -> None:
        """Add a file attachment (a filename with an extension)."""
        if not name:
            raise RecordError("attachment needs a file name")
        self._require_before_header("attachments")
        buf = (c_uint8 * len(data)).from_buffer_copy(bytes(data))
        _check(self._lib.k4a_record_add_attachment(
            self._require_open(), name.encode("utf-8"), buf,
            len(data)), f"add attachment {name}")

    def add_custom_video_track(
            self, name: str, codec_id: str,
            codec_context: bytes = b"", *, width: int, height: int,
            frame_rate: int) -> None:
        """Add a custom video track (name must be ALL CAPS)."""
        validate_tag_name(name)
        self._require_before_header("custom tracks")
        settings = RecordVideoSettings(
            width=int(width), height=int(height),
            frame_rate=int(frame_rate))
        ctx = (c_uint8 * len(codec_context)).from_buffer_copy(
            bytes(codec_context))
        _check(self._lib.k4a_record_add_custom_video_track(
            self._require_open(), name.encode("utf-8"),
            codec_id.encode("utf-8"), ctx, len(codec_context),
            byref(settings)), f"add video track {name}")

    def add_custom_subtitle_track(
            self, name: str, codec_id: str,
            codec_context: bytes = b"", *,
            high_freq_data: bool = False) -> None:
        """Add a custom subtitle track (name must be ALL CAPS)."""
        validate_tag_name(name)
        self._require_before_header("custom tracks")
        settings = RecordSubtitleSettings(high_freq_data=high_freq_data)
        ctx = (c_uint8 * len(codec_context)).from_buffer_copy(
            bytes(codec_context))
        _check(self._lib.k4a_record_add_custom_subtitle_track(
            self._require_open(), name.encode("utf-8"),
            codec_id.encode("utf-8"), ctx, len(codec_context),
            byref(settings)), f"add subtitle track {name}")

    def write_header(self) -> None:
        """Write the recording header and metadata to file."""
        _check(self._lib.k4a_record_write_header(
            self._require_open()), "write header")
        self._header_written = True

    # -- data --------------------------------------------------------------

    def write_capture_handle(self, capture_handle: int) -> None:
        """Write one camera capture, by its raw k4a_capture_t handle.

        The capture's images must be in the resolution and format
        of the device configuration the recording was created with,
        and captures must arrive in increasing timestamp order.
        """
        self._require_header("write_capture")
        _check(self._lib.k4a_record_write_capture(
            self._require_open(), int(capture_handle)),
            "write capture")
        self._frames_written += 1

    def write_capture(self, capture: Capture, *,
                      lib: K4aLibrary | None = None) -> None:
        """Write one decoded :class:`Capture` to the recording.

        The frames are rebuilt as k4a images from their buffers
        (with their timestamps carried over, so the SDK's
        increasing-timestamp rule holds) and released after the
        write. Needs libk4a — pass it, or it is loaded on demand.
        """
        self._require_header("write_capture")
        k4a = lib or self._k4a_lib or load_library()
        self._k4a_lib = k4a
        buffers: list = []
        image_handles: list[int] = []
        capture_handle = c_void_p()
        try:
            result = k4a.k4a_capture_create(byref(capture_handle))
            if result != 0 or not capture_handle.value:
                raise RecordError(
                    "could not build a capture to record "
                    f"(k4a result {result})")
            for frame, setter in (
                    (capture.color,
                     k4a.k4a_capture_set_color_image),
                    (capture.depth,
                     k4a.k4a_capture_set_depth_image),
                    (capture.ir,
                     k4a.k4a_capture_set_ir_image)):
                if frame is None:
                    continue
                handle = self._image_from_frame(k4a, frame, buffers)
                image_handles.append(handle)
                setter(capture_handle.value, handle)
            k4a.k4a_capture_set_temperature_c(
                capture_handle.value, float(capture.temperature_c))
            _check(self._lib.k4a_record_write_capture(
                self._require_open(), capture_handle.value),
                "write capture")
            self._frames_written += 1
        finally:
            if capture_handle.value:
                k4a.k4a_capture_release(capture_handle.value)
            for handle in image_handles:
                k4a.k4a_image_release(handle)

    def _image_from_frame(self, k4a: K4aLibrary, frame: Frame,
                          buffers: list) -> int:
        """A k4a_image_t sharing the frame's buffer, timestamps set.

        The destroy callback is NULL: the buffer lives in
        ``buffers`` until the image is released, so the SDK must
        not free it. ctypes rejects a bare None for a CFUNCTYPE
        argument, so the NULL is spelled as a cast.
        """
        data = (c_uint8 * len(frame.data)).from_buffer_copy(frame.data)
        buffers.append(data)
        handle = c_void_p()
        result = k4a.k4a_image_create_from_buffer(
            int(frame.fmt), frame.width, frame.height, frame.stride,
            data, len(frame.data),
            ctypes.cast(None, _DestroyCallback), None,
            byref(handle))
        if result != 0 or not handle.value:
            raise RecordError(
                "could not build a k4a image from the frame "
                f"(k4a result {result})")
        k4a.k4a_image_set_device_timestamp_usec(
            handle.value, int(frame.device_timestamp_usec))
        k4a.k4a_image_set_timestamp_usec(
            handle.value, int(frame.device_timestamp_usec))
        k4a.k4a_image_set_system_timestamp_nsec(
            handle.value, int(frame.system_timestamp_nsec))
        k4a.k4a_image_set_exposure_usec(
            handle.value, int(frame.exposure_usec))
        k4a.k4a_image_set_white_balance(
            handle.value, int(frame.white_balance))
        k4a.k4a_image_set_iso_speed(
            handle.value, int(frame.iso_speed))
        return handle.value

    def write_imu_sample(self, sample: dict) -> None:
        """Write one IMU sample (a get_imu_sample-style dict)."""
        self._require_header("write_imu_sample")
        _check(self._lib.k4a_record_write_imu_sample(
            self._require_open(),
            _imu_sample_from_dict(sample)), "write IMU sample")
        self._imu_written += 1

    def write_custom_track_data(self, track_name: str,
                                device_timestamp_usec: int,
                                data: bytes) -> None:
        """Write one block of custom-track data."""
        validate_tag_name(track_name)
        self._require_header("write_custom_track_data")
        buf = (c_uint8 * len(data)).from_buffer_copy(bytes(data))
        _check(self._lib.k4a_record_write_custom_track_data(
            self._require_open(), track_name.encode("utf-8"),
            int(device_timestamp_usec), buf, len(data)),
            f"write custom track data to {track_name}")

    def flush(self) -> None:
        """Flush all pending recording data to disk."""
        _check(self._lib.k4a_record_flush(self._require_open()),
               "flush recording")


class Playback:
    """A recording file open for reading (k4a_playback_t).

    Open → read the configuration, tracks, tags and attachments,
    then step captures / IMU samples / custom data blocks forward
    or backward, seeking by timestamp when needed. Captures come
    back as decoded :class:`Capture` objects; ``None`` means the
    end (or start) of the stream was reached. The record library
    alone is enough for everything but captures, which need libk4a
    to decode — it is loaded lazily on the first capture read.
    """

    def __init__(self, lib: K4aRecordLibrary, path: str, handle: int,
                 k4a_lib: K4aLibrary | None = None) -> None:
        self._lib = lib
        self._path = path
        self._handle = handle
        self._k4a_lib = k4a_lib
        self._closed = False

    @classmethod
    def open(cls, path: str, *,
             record_lib: K4aRecordLibrary | None = None,
             k4a_lib: K4aLibrary | None = None) -> Playback:
        """Open an existing recording file for reading."""
        lib = record_lib if record_lib is not None \
            else load_record_library()
        target = Path(path).expanduser()
        if not target.exists():
            raise FileNotFoundError(f"no recording at {target}")
        handle = c_void_p()
        result = lib.k4a_playback_open(
            os.fsencode(str(target)), byref(handle))
        if result != 0 or not handle.value:
            raise PlaybackError(
                f"could not open recording {target} "
                f"(k4a result {result})")
        return cls(lib, str(target), handle.value, k4a_lib=k4a_lib)

    # -- properties ---------------------------------------------------

    @property
    def path(self) -> str:
        """The recording file path."""
        return self._path

    @property
    def is_open(self) -> bool:
        """True while the playback handle is held."""
        return not self._closed

    def _require_open(self) -> int:
        if self._closed:
            raise PlaybackError("playback is closed")
        return self._handle

    def _k4a_library(self) -> K4aLibrary:
        """libk4a, loaded on first capture decode."""
        if self._k4a_lib is None:
            self._k4a_lib = load_library()
        return self._k4a_lib

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Close the playback handle (idempotent)."""
        if self._closed:
            return
        self._lib.k4a_playback_close(self._handle)
        self._closed = True

    def __enter__(self) -> Playback:
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    # -- file info ---------------------------------------------------------

    def info(self) -> dict[str, object]:
        """The recording at a glance: config, tracks, length."""
        return {
            "path": self._path,
            "configuration": self.record_configuration(),
            "track_count": self.track_count(),
            "track_names": self.track_names(),
            "recording_length_usec": self.recording_length_usec(),
            "last_timestamp_usec": self.last_timestamp_usec(),
        }

    def raw_calibration(self) -> str | None:
        """The raw calibration JSON, or None when the recording has none."""
        lib = self._lib
        handle = self._require_open()
        raw = _two_call_bytes(
            lambda buf, size: lib.k4a_playback_get_raw_calibration(
                handle, buf, size),
            what="get raw calibration")
        if raw is None:
            return None
        return bytes(raw).rstrip(b"\x00").decode("utf-8", "replace")

    def calibration(self):
        """The calibration of the device the recording was made with.

        The recording carries the calibration for the mode it was
        recorded in; there is nothing to ask for. Raises
        :class:`CalibrationError` when the recording has none (it
        was made with no device).
        """
        from .calibration import from_ctypes

        raw = CCalibration()
        _check_playback(self._lib.k4a_playback_get_calibration(
            self._require_open(), byref(raw)), "get calibration")
        return from_ctypes(raw)

    def record_configuration(self) -> dict[str, object]:
        """The device configuration used during recording."""
        config = RecordConfiguration()
        _check_playback(
            self._lib.k4a_playback_get_record_configuration(
                self._require_open(), byref(config)),
            "get record configuration")
        return config.as_dict()

    # -- tracks / tags / attachments ----------------------------------------

    def track_exists(self, name: str) -> bool:
        """True when a track with this name is in the file."""
        return bool(self._lib.k4a_playback_check_track_exists(
            self._require_open(), name.encode("utf-8")))

    def track_count(self) -> int:
        """The number of tracks in the file."""
        return int(self._lib.k4a_playback_get_track_count(
            self._require_open()))

    def track_names(self) -> list[str]:
        """Every track name, in file order."""
        names: list[str] = []
        for index in range(self.track_count()):
            name = _two_call_string(
                lambda buf, size, index=index:
                self._lib.k4a_playback_get_track_name(
                    self._require_open(), index, buf, size),
                what=f"get track {index} name")
            if name is None:
                raise PlaybackError(
                    f"track {index} has no name")
            names.append(name)
        return names

    def track_is_builtin(self, name: str) -> bool:
        """True for COLOR / DEPTH / IR / IMU, False for custom tracks."""
        return bool(self._lib.k4a_playback_track_is_builtin(
            self._require_open(), name.encode("utf-8")))

    def track_video_settings(self, name: str) -> dict[str, int]:
        """Width / height / frame-rate of a video track."""
        settings = RecordVideoSettings()
        _check_playback(
            self._lib.k4a_playback_track_get_video_settings(
                self._require_open(), name.encode("utf-8"),
                byref(settings)),
            f"get video settings for track {name}")
        return settings.as_dict()

    def track_codec_id(self, name: str) -> str | None:
        """The codec ID of a track, or None when it has none."""
        return _two_call_string(
            lambda buf, size: self._lib.k4a_playback_track_get_codec_id(
                self._require_open(), name.encode("utf-8"), buf, size),
            what=f"get codec id for track {name}")

    def track_codec_context(self, name: str) -> bytes | None:
        """The codec context of a track, or None when it has none."""
        return _two_call_bytes(
            lambda buf, size:
            self._lib.k4a_playback_track_get_codec_context(
                self._require_open(), name.encode("utf-8"), buf, size),
            what=f"get codec context for track {name}")

    def get_tag(self, name: str) -> str | None:
        """A tag's value, or None when the recording has no such tag."""
        validate_tag_name(name)
        return _two_call_string(
            lambda buf, size: self._lib.k4a_playback_get_tag(
                self._require_open(), name.encode("utf-8"), buf, size),
            what=f"get tag {name}")

    def set_color_conversion(self, fmt: ImageFormat) -> None:
        """Convert color captures to ``fmt`` on read."""
        _check_playback(
            self._lib.k4a_playback_set_color_conversion(
                self._require_open(), int(fmt)),
            "set color conversion")

    def get_attachment(self, name: str) -> bytes | None:
        """An attachment's bytes, or None when it is not in the file."""
        return _two_call_bytes(
            lambda buf, size: self._lib.k4a_playback_get_attachment(
                self._require_open(), name.encode("utf-8"), buf, size),
            what=f"get attachment {name}")

    # -- captures -----------------------------------------------------------

    def next_capture(self) -> Capture | None:
        """The next capture in sequence, or None at end of file."""
        return self._read_capture(previous=False)

    def previous_capture(self) -> Capture | None:
        """The previous capture in sequence, or None at start of file."""
        return self._read_capture(previous=True)

    def _read_capture(self, *, previous: bool) -> Capture | None:
        from .device import capture_from_handle

        lib = self._lib
        fn = lib.k4a_playback_get_previous_capture if previous \
            else lib.k4a_playback_get_next_capture
        handle = c_void_p()
        result = fn(self._require_open(), byref(handle))
        if result == 2:  # K4A_STREAM_RESULT_EOF
            return None
        if result != 0 or not handle.value:
            raise PlaybackError(
                "read capture failed "
                f"(k4a stream result {result})")
        k4a = self._k4a_library()
        try:
            return capture_from_handle(k4a, handle.value)
        finally:
            k4a.k4a_capture_release(handle.value)

    # -- IMU ------------------------------------------------------------------

    def next_imu_sample(self) -> dict[str, object] | None:
        """The next IMU sample, or None at end of file."""
        return self._read_imu_sample(previous=False)

    def previous_imu_sample(self) -> dict[str, object] | None:
        """The previous IMU sample, or None at start of file."""
        return self._read_imu_sample(previous=True)

    def _read_imu_sample(self, *, previous: bool
                         ) -> dict[str, object] | None:
        fn = self._lib.k4a_playback_get_previous_imu_sample \
            if previous else self._lib.k4a_playback_get_next_imu_sample
        sample = ImuSample()
        result = fn(self._require_open(), byref(sample))
        if result == 2:  # K4A_STREAM_RESULT_EOF
            return None
        if result != 0:
            raise PlaybackError(
                "read IMU sample failed "
                f"(k4a stream result {result})")
        return _imu_sample_to_dict(sample)

    # -- custom data blocks -----------------------------------------------------

    def next_data_block(self, track_name: str) -> DataBlock | None:
        """The next data block of a custom track, or None at its end."""
        return self._read_data_block(track_name, previous=False)

    def previous_data_block(self, track_name: str) -> DataBlock | None:
        """The previous data block of a custom track, or None at its start."""
        return self._read_data_block(track_name, previous=True)

    def _read_data_block(self, track_name: str, *,
                         previous: bool) -> DataBlock | None:
        validate_tag_name(track_name)
        fn = self._lib.k4a_playback_get_previous_data_block \
            if previous else self._lib.k4a_playback_get_next_data_block
        handle = c_void_p()
        result = fn(self._require_open(),
                    track_name.encode("utf-8"), byref(handle))
        if result == 2:  # K4A_STREAM_RESULT_EOF
            return None
        if result != 0 or not handle.value:
            raise PlaybackError(
                f"read data block from {track_name} failed "
                f"(k4a stream result {result})")
        try:
            lib = self._lib
            timestamp = int(
                lib.k4a_playback_data_block_get_device_timestamp_usec(
                    handle.value))
            size = int(
                lib.k4a_playback_data_block_get_buffer_size(
                    handle.value))
            buf = lib.k4a_playback_data_block_get_buffer(handle.value)
            data = ctypes.string_at(buf, size) if buf else b""
            return DataBlock(track=track_name,
                             device_timestamp_usec=timestamp,
                             data=data)
        finally:
            self._lib.k4a_playback_data_block_release(handle.value)

    # -- seeking / length ----------------------------------------------------------

    def seek(self, offset_usec: int,
             origin: PlaybackSeekOrigin = PlaybackSeekOrigin.BEGIN
             ) -> None:
        """Seek to a timestamp (begin / end / absolute device time)."""
        _check_playback(self._lib.k4a_playback_seek_timestamp(
            self._require_open(), int(offset_usec), int(origin)),
            "seek playback")

    def recording_length_usec(self) -> int:
        """The recording length: last timestamp minus first, in usec."""
        return int(
            self._lib.k4a_playback_get_recording_length_usec(
                self._require_open()))

    def last_timestamp_usec(self) -> int:
        """The last timestamp in the file, relative to its start.

        Deprecated in the SDK since 1.2.0 — prefer
        :meth:`recording_length_usec`, which returns the same value.
        """
        return int(
            self._lib.k4a_playback_get_last_timestamp_usec(
                self._require_open()))


def record_live(path: str, frames: int = 30, *,
                device_index: int = 0, config=None,
                tags: dict[str, str] | None = None,
                imu: bool = True, timeout_ms: int = 1000,
                lib: K4aLibrary | None = None,
                record_lib: K4aRecordLibrary | None = None
                ) -> dict[str, object]:
    """Record ``frames`` captures from a live device into ``path``.

    Opens the device, starts the cameras (and the IMU when asked),
    writes the header, then loops: one capture per frame, one IMU
    sample per frame when the IMU runs. Everything is closed on the
    way out — the device, the cameras, the IMU, the recording.
    Returns the path, the frames and IMU samples written, and the
    tags. A missing SDK or device raises by name; a timeout on one
    capture ends the run early and says so in the result.
    """
    from .device import DeviceConfig, KinectDevice

    if frames < 0:
        raise RecordError("frames must be 0 or more")
    record = record_lib if record_lib is not None \
        else load_record_library()
    k4a = lib if lib is not None else load_library()
    cfg = config or DeviceConfig()
    device = KinectDevice.open(device_index, lib=k4a)
    timed_out = False
    try:
        recorder = Recorder.create(
            path, cfg, lib=record, device_handle=device.handle,
            tags=tags)
        try:
            if imu:
                recorder.add_imu_track()
            recorder.write_header()
            device.start_cameras(cfg)
            try:
                if imu:
                    device.start_imu()
                try:
                    for _ in range(frames):
                        with device.raw_capture(timeout_ms) as handle:
                            if handle is None:
                                timed_out = True
                                break
                            recorder.write_capture_handle(handle)
                        if imu:
                            sample = device.get_imu_sample(timeout_ms)
                            if sample is not None:
                                recorder.write_imu_sample(sample)
                finally:
                    if imu:
                        device.stop_imu()
            finally:
                device.stop_cameras()
            recorder.flush()
        finally:
            recorder.close()
        return {
            "path": str(Path(path).expanduser()),
            "frames_written": recorder.frames_written,
            "imu_samples": recorder.imu_samples_written,
            "timed_out": timed_out,
            "tags": dict(tags or {}),
        }
    finally:
        device.close()
