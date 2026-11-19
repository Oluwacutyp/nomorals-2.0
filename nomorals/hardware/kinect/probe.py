"""Probe — the end-to-end Kinect check the owner can run.

``probe()`` walks the whole path and reports each step: is the
SDK installed, how many devices are connected, and — for the
first device, when asked — open it, read its serial number and
firmware versions, fetch its calibration, and capture real
frames. Every step names its outcome; a missing SDK or a
missing device is a *report*, not an exception and not a
silent skip. Run it as a module::

    python -m nomorals.hardware.kinect.probe [--frames N] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field

from .bindings import load_library
from .device import DeviceConfig, KinectDevice
from .enums import ColorResolution, DepthMode
from .errors import KinectError

__all__ = ["ProbeReport", "probe"]


@dataclass
class ProbeReport:
    """The probe's structured result (JSON-serialisable)."""

    sdk_present: bool = False
    sdk_path: str | None = None
    sdk_error: str | None = None
    device_count: int = 0
    devices: list[dict] = field(default_factory=list)
    frames_captured: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        """The report as plain data."""
        return asdict(self)

    @property
    def ok(self) -> bool:
        """The SDK loaded (a missing device is not a failure)."""
        return self.sdk_present and self.error is None

    def summary(self) -> str:
        """One human line per step, in plain words."""
        lines = []
        if not self.sdk_present:
            lines.append(
                f"No Kinect SDK installed: {self.sdk_error}")
            return "\n".join(lines)
        lines.append(
            f"Kinect SDK found ({self.sdk_path}); "
            f"{self.device_count} device(s) connected.")
        if not self.devices:
            lines.append(
                "No Azure Kinect device connected — plug one in "
                "and run the probe again.")
            return "\n".join(lines)
        for device in self.devices:
            serial = device.get("serial_number") or "?"
            lines.append(f"Device {device['index']}: serial {serial}.")
            if device.get("calibration_ok"):
                lines.append("  Calibration read OK.")
            if self.frames_captured:
                lines.append(
                    f"  Captured {self.frames_captured} frame(s).")
            elif device.get("capture_error"):
                lines.append(
                    f"  Capture failed: {device['capture_error']}")
        if self.error:
            lines.append(f"Probe stopped early: {self.error}")
        return "\n".join(lines)


def probe(frames: int = 0, timeout_ms: int = 1000,
          config: DeviceConfig | None = None) -> ProbeReport:
    """Run the end-to-end probe.

    ``frames`` is how many real frames to capture from the first
    device (0 = don't start the cameras, just enumerate + identify).
    A missing SDK or device is reported, never raised.
    """
    report = ProbeReport()
    try:
        lib = load_library()
    except KinectError as exc:
        report.sdk_error = str(exc)
        return report
    report.sdk_present = True
    report.sdk_path = lib.path
    try:
        report.device_count = lib.installed_count()
    except Exception as exc:  # noqa: BLE001 — the probe never raises
        report.error = f"installed-count failed: {exc}"
        return report
    if report.device_count == 0:
        return report
    for index in range(report.device_count):
        entry: dict = {"index": index}
        try:
            with KinectDevice.open(index, lib=lib) as device:
                entry["serial_number"] = device.serial_number()
                entry["version"] = device.version()
                try:
                    device.calibration(
                        DepthMode.NFOV_UNBINNED,
                        ColorResolution.RES_720P)
                    entry["calibration_ok"] = True
                except KinectError as exc:
                    entry["calibration_ok"] = False
                    entry["calibration_error"] = str(exc)
                if frames > 0:
                    device.start_cameras(
                        config or DeviceConfig())
                    try:
                        got = 0
                        for _ in range(frames):
                            capture = device.get_capture(
                                timeout_ms=timeout_ms)
                            if capture is None:
                                entry["capture_error"] = (
                                    "timed out waiting for a frame")
                                break
                            got += 1
                            entry.setdefault("frames", []).append({
                                "color": (
                                    None if capture.color is None
                                    else {
                                        "format":
                                            capture.color.fmt.name,
                                        "width":
                                            capture.color.width,
                                        "height":
                                            capture.color.height}),
                                "depth": (
                                    None if capture.depth is None
                                    else {
                                        "format":
                                            capture.depth.fmt.name,
                                        "width":
                                            capture.depth.width,
                                        "height":
                                            capture.depth.height}),
                            })
                        report.frames_captured += got
                    finally:
                        device.stop_cameras()
        except KinectError as exc:
            entry["error"] = str(exc)
            entry["kind"] = exc.kind
        report.devices.append(entry)
    return report


def main(argv: list[str] | None = None) -> int:
    """The probe CLI: ``python -m nomorals.hardware.kinect.probe``."""
    parser = argparse.ArgumentParser(
        description="Probe for Azure Kinect devices (SDK + frames).")
    parser.add_argument("--frames", type=int, default=0,
                        help="frames to capture from the first device")
    parser.add_argument("--timeout-ms", type=int, default=1000,
                        help="milliseconds to wait per frame")
    parser.add_argument("--json", action="store_true",
                        help="print the report as JSON")
    args = parser.parse_args(argv)
    report = probe(frames=args.frames, timeout_ms=args.timeout_ms)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2,
                         default=str))
    else:
        print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
