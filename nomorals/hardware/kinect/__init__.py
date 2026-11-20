"""Azure Kinect — the k4a-sys bindings, in Python.

Two ports, one package:

* the k4a.h half: storytold/k4a-sys-temp (a Linux-building fork of
  Amedee's k4a-sys, Azure Kinect Sensor SDK 1.3) — the vendored Rust
  crate lives under ``rs/k4a-sys/`` (byte-identical to the clone),
  and this package is its Python half: ctypes bindings over
  libk4a, the calibration geometry, frame decoding, the device,
  and the probe.
* the k4arecord half: the upstream echelon/k4a-sys crate itself
  (0.2.2) — its vendored ``k4arecord/`` headers were never bound
  by either crate, so this package binds them too: recording
  ``.mkv`` captures and playing them back, over libk4arecord.

No device and no SDK on this box? ``probe()`` says so by name; a
fixture-driven path proves the bindings decode real frame shapes
without hardware.
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
from .errors import (
    KinectError,
    NoDeviceError,
    NoSdkError,
    PlaybackError,
    RecordError,
)
from .frames import Capture, Frame, FrameStats
from .parity import run_parity
from .probe import ProbeReport, probe
from .record import (
    K4aRecordLibrary,
    Playback,
    PlaybackSeekOrigin,
    RecordConfiguration,
    Recorder,
    StreamResult,
    load_record_library,
    record_live,
)

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
    "K4aRecordLibrary",
    "KinectDevice",
    "KinectError",
    "NoDeviceError",
    "NoSdkError",
    "Playback",
    "PlaybackError",
    "PlaybackSeekOrigin",
    "ProbeReport",
    "RecordConfiguration",
    "RecordError",
    "Recorder",
    "StreamResult",
    "Transformation",
    "load_library",
    "load_record_library",
    "parse_raw_calibration",
    "point_cloud",
    "probe",
    "record_live",
    "run_parity",
    "verify_attribution",
]
