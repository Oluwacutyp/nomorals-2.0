"""ctypes bindings for libk4a — the Python half of the k4a-sys port.

This is the Python equivalent of what the vendored Rust crate does
with bindgen: the same C API (``k4a/k4a.h``, SDK 1.3, vendored under
``rs/k4a-sys/vendor/include/linux/k4a/``) expressed as ctypes
structs and function prototypes, with the same names and the same
field order. The struct layouts are machine-checked against the
real C compiler: the parity runner compares ``ctypes.sizeof`` and
field offsets against a gcc probe of the installed SDK headers.

Loading: ``libk4a.so.1.4`` first (the SDK this box has), then the
unversioned ``libk4a.so``, then the 1.3 soname the vendored crate
was written against. Nothing here is lazy-guessed: a missing
library raises :class:`NoSdkError` naming every soname tried.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_int32,
    c_size_t,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)
from pathlib import Path

from .errors import NoSdkError

__all__ = [
    "SONAMES",
    "Calibration",
    "CalibrationCamera",
    "CalibrationExtrinsics",
    "CalibrationIntrinsicParameters",
    "CalibrationIntrinsics",
    "DeviceConfiguration",
    "Float2",
    "Float3",
    "HardwareVersion",
    "ImuSample",
    "K4aLibrary",
    "Version",
    "load_library",
]

#: Sonames tried in order when loading libk4a.
SONAMES = ("libk4a.so.1.4", "libk4a.so", "libk4a.so.1.3")

#: The k4a-sys crate was written against SDK 1.3; the binding
#: surface it exposes is unchanged in 1.4.
MIN_SDK_MAJOR = 1
MIN_SDK_MINOR = 3


class Float2(ctypes.Structure):
    """k4a_float2_t — two floats, x then y."""

    _fields_ = [("x", c_float), ("y", c_float)]


class Float3(ctypes.Structure):
    """k4a_float3_t — three floats, x, y, z."""

    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float)]


class Version(ctypes.Structure):
    """k4a_version_t — major, minor, iteration."""

    _fields_ = [("major", c_uint32),
                ("minor", c_uint32),
                ("iteration", c_uint32)]


class HardwareVersion(ctypes.Structure):
    """k4a_hardware_version_t — four firmware versions + build info."""

    _fields_ = [("rgb", Version),
                ("depth", Version),
                ("audio", Version),
                ("depth_sensor", Version),
                ("firmware_build", c_int),
                ("firmware_signature", c_int)]


class ImuSample(ctypes.Structure):
    """k4a_imu_sample_t — temperature, accel, gyro + timestamps."""

    _fields_ = [("temperature", c_float),
                ("acc_sample", Float3),
                ("acc_timestamp_usec", c_uint64),
                ("gyro_sample", Float3),
                ("gyro_timestamp_usec", c_uint64)]


class CalibrationExtrinsics(ctypes.Structure):
    """k4a_calibration_extrinsics_t — 3x3 rotation + translation (mm)."""

    _fields_ = [("rotation", c_float * 9),
                ("translation", c_float * 3)]


class _IntrinsicParamStruct(ctypes.Structure):
    _fields_ = [("cx", c_float),
                ("cy", c_float),
                ("fx", c_float),
                ("fy", c_float),
                ("k1", c_float),
                ("k2", c_float),
                ("k3", c_float),
                ("k4", c_float),
                ("k5", c_float),
                ("k6", c_float),
                ("codx", c_float),
                ("cody", c_float),
                ("p2", c_float),
                ("p1", c_float),
                ("metric_radius", c_float)]


class CalibrationIntrinsicParameters(ctypes.Union):
    """k4a_calibration_intrinsic_parameters_t — params or float[15]."""

    _fields_ = [("param", _IntrinsicParamStruct),
                ("v", c_float * 15)]


class CalibrationIntrinsics(ctypes.Structure):
    """k4a_calibration_intrinsics_t — model type + parameters."""

    _fields_ = [("type", c_int),
                ("parameter_count", c_uint32),
                ("parameters", CalibrationIntrinsicParameters)]


class CalibrationCamera(ctypes.Structure):
    """k4a_calibration_camera_t — one camera's full calibration."""

    _fields_ = [("extrinsics", CalibrationExtrinsics),
                ("intrinsics", CalibrationIntrinsics),
                ("resolution_width", c_int),
                ("resolution_height", c_int),
                ("metric_radius", c_float)]


class Calibration(ctypes.Structure):
    """k4a_calibration_t — depth + color cameras + extrinsics grid."""

    _fields_ = [("depth_camera_calibration", CalibrationCamera),
                ("color_camera_calibration", CalibrationCamera),
                ("extrinsics",
                 CalibrationExtrinsics * 4 * 4),
                ("depth_mode", c_int),
                ("color_resolution", c_int)]


class DeviceConfiguration(ctypes.Structure):
    """k4a_device_configuration_t — the start_cameras() config."""

    _fields_ = [("color_format", c_int),
                ("color_resolution", c_int),
                ("depth_mode", c_int),
                ("camera_fps", c_int),
                ("synchronized_images_only", c_bool),
                ("depth_delay_off_color_usec", c_int32),
                ("wired_sync_mode", c_int),
                ("subordinate_delay_off_master_usec", c_uint32),
                ("disable_streaming_indicator", c_bool)]

    @classmethod
    def disable_all(cls) -> DeviceConfiguration:
        """K4A_DEVICE_CONFIG_INIT_DISABLE_ALL — every sensor off."""
        return cls(
            color_format=0,  # COLOR_MJPG
            color_resolution=0,  # OFF
            depth_mode=0,  # OFF
            camera_fps=2,  # 30 FPS
            synchronized_images_only=False,
            depth_delay_off_color_usec=0,
            wired_sync_mode=0,  # STANDALONE
            subordinate_delay_off_master_usec=0,
            disable_streaming_indicator=False,
        )


# Callback types (k4a.h): debug handler, allocator, buffer destroy.
_LogCallback = CFUNCTYPE(None, c_void_p, c_int, c_char_p, c_int, c_char_p)
_AllocCallback = CFUNCTYPE(c_void_p, c_size_t, c_void_p)
_FreeCallback = CFUNCTYPE(None, c_void_p, c_void_p)
_DestroyCallback = CFUNCTYPE(None, c_void_p, c_void_p)

#: (name, argtypes, restype) for every k4a.h export, in header order.
_FUNCTIONS: tuple[tuple[str, list, object], ...] = (
    ("k4a_device_get_installed_count", [], c_uint32),
    ("k4a_set_debug_message_handler",
     [_LogCallback, c_int, c_void_p], c_int),
    ("k4a_set_allocator", [_AllocCallback, _FreeCallback], c_int),
    ("k4a_device_open", [c_uint32, POINTER(c_void_p)], c_int),
    ("k4a_device_close", [c_void_p], None),
    ("k4a_device_get_capture",
     [c_void_p, POINTER(c_void_p), c_int32], c_int),
    ("k4a_device_get_imu_sample",
     [c_void_p, POINTER(ImuSample), c_int32], c_int),
    ("k4a_capture_create", [POINTER(c_void_p)], c_int),
    ("k4a_capture_release", [c_void_p], None),
    ("k4a_capture_reference", [c_void_p], None),
    ("k4a_capture_get_color_image", [c_void_p], c_void_p),
    ("k4a_capture_get_depth_image", [c_void_p], c_void_p),
    ("k4a_capture_get_ir_image", [c_void_p], c_void_p),
    ("k4a_capture_set_color_image", [c_void_p, c_void_p], None),
    ("k4a_capture_set_depth_image", [c_void_p, c_void_p], None),
    ("k4a_capture_set_ir_image", [c_void_p, c_void_p], None),
    ("k4a_capture_set_temperature_c", [c_void_p, c_float], None),
    ("k4a_capture_get_temperature_c", [c_void_p], c_float),
    ("k4a_image_create",
     [c_int, c_int, c_int, c_int, POINTER(c_void_p)], c_int),
    ("k4a_image_create_from_buffer",
     [c_int, c_int, c_int, c_int, c_void_p, c_size_t,
      _DestroyCallback, c_void_p, POINTER(c_void_p)], c_int),
    ("k4a_image_get_buffer", [c_void_p], POINTER(c_uint8)),
    ("k4a_image_get_size", [c_void_p], c_size_t),
    ("k4a_image_get_format", [c_void_p], c_int),
    ("k4a_image_get_width_pixels", [c_void_p], c_int),
    ("k4a_image_get_height_pixels", [c_void_p], c_int),
    ("k4a_image_get_stride_bytes", [c_void_p], c_int),
    ("k4a_image_get_timestamp_usec", [c_void_p], c_uint64),
    ("k4a_image_get_device_timestamp_usec", [c_void_p], c_uint64),
    ("k4a_image_get_system_timestamp_nsec", [c_void_p], c_uint64),
    ("k4a_image_get_exposure_usec", [c_void_p], c_uint64),
    ("k4a_image_get_white_balance", [c_void_p], c_uint32),
    ("k4a_image_get_iso_speed", [c_void_p], c_uint32),
    ("k4a_image_set_device_timestamp_usec", [c_void_p, c_uint64], None),
    ("k4a_image_set_timestamp_usec", [c_void_p, c_uint64], None),
    ("k4a_image_set_system_timestamp_nsec", [c_void_p, c_uint64], None),
    ("k4a_image_set_exposure_usec", [c_void_p, c_uint64], None),
    ("k4a_image_set_exposure_time_usec", [c_void_p, c_uint64], None),
    ("k4a_image_set_white_balance", [c_void_p, c_uint32], None),
    ("k4a_image_set_iso_speed", [c_void_p, c_uint32], None),
    ("k4a_image_reference", [c_void_p], None),
    ("k4a_image_release", [c_void_p], None),
    ("k4a_device_start_cameras",
     [c_void_p, POINTER(DeviceConfiguration)], c_int),
    ("k4a_device_stop_cameras", [c_void_p], None),
    ("k4a_device_start_imu", [c_void_p], c_int),
    ("k4a_device_stop_imu", [c_void_p], None),
    ("k4a_device_get_serialnum",
     [c_void_p, c_char_p, POINTER(c_size_t)], c_int),
    ("k4a_device_get_version",
     [c_void_p, POINTER(HardwareVersion)], c_int),
    ("k4a_device_get_color_control_capabilities",
     [c_void_p, c_int, POINTER(c_bool), POINTER(c_int32),
      POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
      POINTER(c_int)], c_int),
    ("k4a_device_get_color_control",
     [c_void_p, c_int, POINTER(c_int), POINTER(c_int32)], c_int),
    ("k4a_device_set_color_control",
     [c_void_p, c_int, c_int, c_int32], c_int),
    ("k4a_device_get_raw_calibration",
     [c_void_p, POINTER(c_uint8), POINTER(c_size_t)], c_int),
    ("k4a_device_get_calibration",
     [c_void_p, c_int, c_int, POINTER(Calibration)], c_int),
    ("k4a_device_get_sync_jack",
     [c_void_p, POINTER(c_bool), POINTER(c_bool)], c_int),
    ("k4a_calibration_get_from_raw",
     [c_char_p, c_size_t, c_int, c_int, POINTER(Calibration)], c_int),
    ("k4a_calibration_3d_to_3d",
     [POINTER(Calibration), POINTER(Float3), c_int, c_int,
      POINTER(Float3)], c_int),
    ("k4a_calibration_2d_to_3d",
     [POINTER(Calibration), POINTER(Float2), c_float, c_int, c_int,
      POINTER(Float3), POINTER(c_int)], c_int),
    ("k4a_calibration_3d_to_2d",
     [POINTER(Calibration), POINTER(Float3), c_int, c_int,
      POINTER(Float2), POINTER(c_int)], c_int),
    ("k4a_calibration_2d_to_2d",
     [POINTER(Calibration), POINTER(Float2), c_float, c_int, c_int,
      POINTER(Float2), POINTER(c_int)], c_int),
    ("k4a_calibration_color_2d_to_depth_2d",
     [POINTER(Calibration), POINTER(Float2), c_void_p,
      POINTER(Float2), POINTER(c_int)], c_int),
    ("k4a_transformation_create", [POINTER(Calibration)], c_void_p),
    ("k4a_transformation_destroy", [c_void_p], None),
    ("k4a_transformation_depth_image_to_color_camera",
     [c_void_p, c_void_p, c_void_p], c_int),
    ("k4a_transformation_depth_image_to_color_camera_custom",
     [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int,
      c_uint32], c_int),
    ("k4a_transformation_color_image_to_depth_camera",
     [c_void_p, c_void_p, c_void_p, c_void_p], c_int),
    ("k4a_transformation_depth_image_to_point_cloud",
     [c_void_p, c_void_p, c_int, c_void_p], c_int),
)


class K4aLibrary:
    """A loaded libk4a with every k4a.h export bound.

    Attribute access is the C function: ``lib.k4a_device_open(...)``.
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
                    f"{path} has no symbol {name}: not a k4a SDK "
                    "library")
            func.argtypes = argtypes
            func.restype = restype
            setattr(self, name, func)
            self.functions.append(name)

    def installed_count(self) -> int:
        """k4a_device_get_installed_count()."""
        return int(self.k4a_device_get_installed_count())


def load_library(sonames: tuple[str, ...] = SONAMES) -> K4aLibrary:
    """Load libk4a, trying each soname in order.

    ``K4A_LIB`` names an explicit library path: when it is set it
    is authoritative — a path that does not exist fails loudly
    naming it, instead of silently falling back to a different
    library. Raises :class:`NoSdkError` naming every name tried
    when nothing loads.
    """
    tried: list[str] = []
    override_text = os.environ.get("K4A_LIB", "")
    override = Path(override_text) if override_text else None
    if override is not None:
        if not override.exists():
            raise NoSdkError(
                f"K4A_LIB points at a missing library: {override}")
        tried.append(str(override))
        return K4aLibrary(str(override))
    for name in sonames:
        tried.append(name)
        try:
            return K4aLibrary(name)
        except (OSError, NoSdkError):
            continue
    raise NoSdkError(
        "libk4a not found (tried: " + ", ".join(tried) + "). "
        "Install the Azure Kinect Sensor SDK (libk4a1.4) or set "
        "K4A_LIB to the library path.")
