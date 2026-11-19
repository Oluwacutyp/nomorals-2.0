"""Enum mirrors of the Azure Kinect C API (k4atypes.h, SDK 1.3).

Every value is pinned to the vendored header
``rs/k4a-sys/vendor/include/linux/k4a/k4atypes.h`` — the parity
runner re-parses that header and checks these numbers, so a
drift between the binding and the SDK it binds is caught by a
test, not by a device misbehaving at runtime.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "BufferResult",
    "CalibrationModelType",
    "CalibrationType",
    "ColorControlCommand",
    "ColorControlMode",
    "ColorResolution",
    "DepthMode",
    "FirmwareBuild",
    "FirmwareSignature",
    "Fps",
    "ImageFormat",
    "InterpolationType",
    "LogLevel",
    "Result",
    "WaitResult",
    "WiredSyncMode",
    "COLOR_RESOLUTION_DIMS",
    "DEPTH_MODE_DIMS",
    "bytes_per_pixel",
]


class Result(IntEnum):
    """k4a_result_t."""

    SUCCEEDED = 0
    FAILED = 1


class BufferResult(IntEnum):
    """k4a_buffer_result_t."""

    SUCCEEDED = 0
    FAILED = 1
    TOO_SMALL = 2


class WaitResult(IntEnum):
    """k4a_wait_result_t."""

    SUCCEEDED = 0
    FAILED = 1
    TIMEOUT = 2


class LogLevel(IntEnum):
    """k4a_log_level_t."""

    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    TRACE = 4
    OFF = 5


class DepthMode(IntEnum):
    """k4a_depth_mode_t."""

    OFF = 0
    NFOV_2X2BINNED = 1
    NFOV_UNBINNED = 2
    WFOV_2X2BINNED = 3
    WFOV_UNBINNED = 4
    PASSIVE_IR = 5


#: Depth / passive-IR capture size per depth mode, in pixels, as
#: documented on each enumerator in k4atypes.h.
DEPTH_MODE_DIMS: dict[DepthMode, tuple[int, int]] = {
    DepthMode.OFF: (0, 0),
    DepthMode.NFOV_2X2BINNED: (320, 288),
    DepthMode.NFOV_UNBINNED: (640, 576),
    DepthMode.WFOV_2X2BINNED: (512, 512),
    DepthMode.WFOV_UNBINNED: (1024, 1024),
    DepthMode.PASSIVE_IR: (1024, 1024),
}


class ColorResolution(IntEnum):
    """k4a_color_resolution_t."""

    OFF = 0
    RES_720P = 1
    RES_1080P = 2
    RES_1440P = 3
    RES_1536P = 4
    RES_2160P = 5
    RES_3072P = 6


#: Color capture size per resolution, in pixels, as documented on
#: each enumerator in k4atypes.h.
COLOR_RESOLUTION_DIMS: dict[ColorResolution, tuple[int, int]] = {
    ColorResolution.OFF: (0, 0),
    ColorResolution.RES_720P: (1280, 720),
    ColorResolution.RES_1080P: (1920, 1080),
    ColorResolution.RES_1440P: (2560, 1440),
    ColorResolution.RES_1536P: (2048, 1536),
    ColorResolution.RES_2160P: (3840, 2160),
    ColorResolution.RES_3072P: (4096, 3072),
}


class ImageFormat(IntEnum):
    """k4a_image_format_t."""

    COLOR_MJPG = 0
    COLOR_NV12 = 1
    COLOR_YUY2 = 2
    COLOR_BGRA32 = 3
    DEPTH16 = 4
    IR16 = 5
    CUSTOM8 = 6
    CUSTOM16 = 7
    CUSTOM = 8


def bytes_per_pixel(fmt: ImageFormat) -> int | None:
    """Bytes per pixel for the fixed-size formats, else None.

    MJPG is compressed (variable size) and CUSTOM is caller-defined,
    so neither has a fixed bytes-per-pixel. NV12's luma plane is
    1 byte/px with a half-height chroma plane after it.
    """
    return {
        ImageFormat.COLOR_NV12: 1,
        ImageFormat.COLOR_YUY2: 2,
        ImageFormat.COLOR_BGRA32: 4,
        ImageFormat.DEPTH16: 2,
        ImageFormat.IR16: 2,
        ImageFormat.CUSTOM8: 1,
        ImageFormat.CUSTOM16: 2,
    }.get(fmt)


class Fps(IntEnum):
    """k4a_fps_t."""

    FPS_5 = 0
    FPS_15 = 1
    FPS_30 = 2

    @property
    def hz(self) -> int:
        """Frames per second this enumerator means."""
        return {Fps.FPS_5: 5, Fps.FPS_15: 15, Fps.FPS_30: 30}[self]


class WiredSyncMode(IntEnum):
    """k4a_wired_sync_mode_t."""

    STANDALONE = 0
    MASTER = 1
    SUBORDINATE = 2


class CalibrationType(IntEnum):
    """k4a_calibration_type_t."""

    UNKNOWN = -1
    DEPTH = 0
    COLOR = 1
    GYRO = 2
    ACCEL = 3
    NUM = 4


class CalibrationModelType(IntEnum):
    """k4a_calibration_model_type_t."""

    UNKNOWN = 0
    THETA = 1
    POLYNOMIAL_3K = 2
    RATIONAL_6KT = 3
    BROWN_CONRADY = 4


class ColorControlCommand(IntEnum):
    """k4a_color_control_command_t."""

    EXPOSURE_TIME_ABSOLUTE = 0
    AUTO_EXPOSURE_PRIORITY = 1
    BRIGHTNESS = 2
    CONTRAST = 3
    SATURATION = 4
    SHARPNESS = 5
    WHITEBALANCE = 6
    BACKLIGHT_COMPENSATION = 7
    GAIN = 8
    POWERLINE_FREQUENCY = 9


class ColorControlMode(IntEnum):
    """k4a_color_control_mode_t."""

    AUTO = 0
    MANUAL = 1


class FirmwareBuild(IntEnum):
    """k4a_firmware_build_t."""

    RELEASE = 0
    DEBUG = 1


class FirmwareSignature(IntEnum):
    """k4a_firmware_signature_t."""

    MSFT = 0
    TEST = 1
    UNSIGNED = 2


class InterpolationType(IntEnum):
    """k4a_transformation_interpolation_type_t."""

    NEAREST = 0
    LINEAR = 1
