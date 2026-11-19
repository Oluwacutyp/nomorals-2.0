"""Errors for the Kinect organ.

Every failure carries a machine-readable ``kind`` (and a human
sentence in the message) so callers — and the probe — can tell
"no SDK installed" from "no device plugged in" from "the device
refused to start" without parsing prose.
"""

from __future__ import annotations

__all__ = [
    "CalibrationError",
    "CaptureError",
    "DeviceError",
    "KinectError",
    "NoDeviceError",
    "NoSdkError",
    "SdkMismatchError",
]


class KinectError(Exception):
    """Base error for the Kinect organ."""

    def __init__(self, message: str, *, kind: str = "failed") -> None:
        super().__init__(message)
        self.kind = kind


class NoSdkError(KinectError):
    """libk4a could not be loaded: the SDK is not installed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="no_sdk")


class SdkMismatchError(KinectError):
    """The loaded libk4a is too old for this binding surface."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="sdk_mismatch")


class NoDeviceError(KinectError):
    """No Azure Kinect device is connected."""

    def __init__(self, message: str = "no Azure Kinect device found") -> None:
        super().__init__(message, kind="no_device")


class DeviceError(KinectError):
    """A device operation failed (open, start, control, version)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="device_failed")


class CaptureError(KinectError):
    """A capture failed or timed out."""

    def __init__(self, message: str, *, kind: str = "capture_failed") -> None:
        super().__init__(message, kind=kind)


class CalibrationError(KinectError):
    """Calibration could not be read or parsed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="calibration_failed")
