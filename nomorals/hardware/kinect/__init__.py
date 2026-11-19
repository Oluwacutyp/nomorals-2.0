"""Azure Kinect — the patched k4a-sys bindings, in Python.

The port of storytold/k4a-sys-temp (a Linux-building fork of
Amedee's k4a-sys, Azure Kinect Sensor SDK 1.3): the vendored Rust
crate lives under ``rs/k4a-sys/`` (byte-identical to the clone),
and this package is its Python half — ctypes bindings over
libk4a, the calibration geometry, frame decoding, the device,
and the probe. No device and no SDK on this box? ``probe()``
says so by name; a fixture-driven path proves the bindings
decode real frame shapes without hardware.
"""

from __future__ import annotations

from .attribution import verify_attribution
from .bindings import K4aLibrary, load_library
from .calibration import Calibration, parse_raw_calibration, point_cloud
from .device import DeviceConfig, KinectDevice, Transformation
from .enums import (
    CalibrationType,
    ColorResolution,
    DepthMode,
    Fps,
    ImageFormat,
)
from .errors import KinectError, NoDeviceError, NoSdkError
from .frames import Capture, Frame, FrameStats
from .parity import run_parity
from .probe import ProbeReport, probe

__all__ = [
    "Calibration",
    "CalibrationType",
    "Capture",
    "ColorResolution",
    "DepthMode",
    "DeviceConfig",
    "Fps",
    "Frame",
    "FrameStats",
    "ImageFormat",
    "K4aLibrary",
    "KinectDevice",
    "KinectError",
    "NoDeviceError",
    "NoSdkError",
    "ProbeReport",
    "Transformation",
    "load_library",
    "parse_raw_calibration",
    "point_cloud",
    "probe",
    "run_parity",
    "verify_attribution",
]
