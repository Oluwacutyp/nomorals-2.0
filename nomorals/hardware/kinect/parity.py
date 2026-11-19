"""Parity runner — the machine-checked parity TOML.

Each [[case]] names a real fixture (tests/fixtures/kinect/),
the operation applied to it, and the expected outcome — and the
runner EXECUTES it against the real code and reports pass/fail
per case with measured values. Parity is measured, never
asserted: a case whose expectation is wrong fails loudly, a
missing fixture fails (never a silent skip), and the SDK
cases compare this port's geometry against the *real* libk4a
binary on this box — same calibration, same points — so the
math is checked against the SDK, not against itself.

Schema (version 1):

    [schema]
    version = 1

    [[case]]
    id = "enum-depth-mode"
    op = "enum_values"
    [case.expect.values]
    "DepthMode.NFOV_UNBINNED" = 2

Ops: enum_values, resolutions, struct_layout, config_disable_all,
depth_decode, depth_stats, calibration_parse, geometry_roundtrip,
sdk_geometry, sdk_point_cloud, probe_report, jpeg_sniff,
pinhole_xyz. Expect keys are per-op; see _run_case.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import tomllib
from pathlib import Path
from typing import Any

from . import calibration as cal
from . import enums as k4a_enums
from .bindings import (
    Calibration as CCalibration,
)
from .bindings import (
    DeviceConfiguration,
    Float2,
    Float3,
    HardwareVersion,
    ImuSample,
    K4aLibrary,
    _LogCallback,
    load_library,
)
from .device import DeviceConfig
from .errors import KinectError
from .frames import (
    Frame,
    decode_depth16,
    depth_stats,
    frame_from_buffer,
    is_jpeg,
)
from .probe import probe as run_probe

#: The SDK's own env logger writes diagnostics to stderr unless
#: told not to; parity runs want silence (the cases report their
#: own failures). This must be set before the library first logs,
#: so it lives at import time — setdefault, so an explicit user
#: setting wins.
os.environ.setdefault("K4A_ENABLE_LOG_TO_STDOUT", "0")

__all__ = ["DEFAULT_PARITY_PATH", "ParitySkipped", "load_parity",
           "run_parity"]


class ParitySkipped(Exception):
    """A case the environment cannot run (never a silent pass).

    Only the SDK transform-engine case uses this: libdepthengine
    is GL-based and needs a display, so without one the case is
    skipped with the reason named — run under ``xvfb-run`` for
    the full green.
    """

DEFAULT_PARITY_PATH = Path(__file__).resolve().parent.parent \
    / "kinect-parity.toml"

_FIXTURES = Path(__file__).resolve().parents[3] \
    / "tests" / "fixtures" / "kinect"

_HEADER = Path(__file__).resolve().parent \
    / "rs" / "k4a-sys" / "vendor" / "include" / "linux" / "k4a" \
    / "k4atypes.h"

#: C enumerator name → (Python enum class, member name).
_MEMBER_NAMES = {
    "K4A_RESULT_SUCCEEDED": ("Result", "SUCCEEDED"),
    "K4A_RESULT_FAILED": ("Result", "FAILED"),
    "K4A_BUFFER_RESULT_SUCCEEDED": ("BufferResult", "SUCCEEDED"),
    "K4A_BUFFER_RESULT_FAILED": ("BufferResult", "FAILED"),
    "K4A_BUFFER_RESULT_TOO_SMALL": ("BufferResult", "TOO_SMALL"),
    "K4A_WAIT_RESULT_SUCCEEDED": ("WaitResult", "SUCCEEDED"),
    "K4A_WAIT_RESULT_FAILED": ("WaitResult", "FAILED"),
    "K4A_WAIT_RESULT_TIMEOUT": ("WaitResult", "TIMEOUT"),
    "K4A_LOG_LEVEL_CRITICAL": ("LogLevel", "CRITICAL"),
    "K4A_LOG_LEVEL_ERROR": ("LogLevel", "ERROR"),
    "K4A_LOG_LEVEL_WARNING": ("LogLevel", "WARNING"),
    "K4A_LOG_LEVEL_INFO": ("LogLevel", "INFO"),
    "K4A_LOG_LEVEL_TRACE": ("LogLevel", "TRACE"),
    "K4A_LOG_LEVEL_OFF": ("LogLevel", "OFF"),
    "K4A_DEPTH_MODE_OFF": ("DepthMode", "OFF"),
    "K4A_DEPTH_MODE_NFOV_2X2BINNED": ("DepthMode", "NFOV_2X2BINNED"),
    "K4A_DEPTH_MODE_NFOV_UNBINNED": ("DepthMode", "NFOV_UNBINNED"),
    "K4A_DEPTH_MODE_WFOV_2X2BINNED": ("DepthMode", "WFOV_2X2BINNED"),
    "K4A_DEPTH_MODE_WFOV_UNBINNED": ("DepthMode", "WFOV_UNBINNED"),
    "K4A_DEPTH_MODE_PASSIVE_IR": ("DepthMode", "PASSIVE_IR"),
    "K4A_COLOR_RESOLUTION_OFF": ("ColorResolution", "OFF"),
    "K4A_COLOR_RESOLUTION_720P": ("ColorResolution", "RES_720P"),
    "K4A_COLOR_RESOLUTION_1080P": ("ColorResolution", "RES_1080P"),
    "K4A_COLOR_RESOLUTION_1440P": ("ColorResolution", "RES_1440P"),
    "K4A_COLOR_RESOLUTION_1536P": ("ColorResolution", "RES_1536P"),
    "K4A_COLOR_RESOLUTION_2160P": ("ColorResolution", "RES_2160P"),
    "K4A_COLOR_RESOLUTION_3072P": ("ColorResolution", "RES_3072P"),
    "K4A_IMAGE_FORMAT_COLOR_MJPG": ("ImageFormat", "COLOR_MJPG"),
    "K4A_IMAGE_FORMAT_COLOR_NV12": ("ImageFormat", "COLOR_NV12"),
    "K4A_IMAGE_FORMAT_COLOR_YUY2": ("ImageFormat", "COLOR_YUY2"),
    "K4A_IMAGE_FORMAT_COLOR_BGRA32": ("ImageFormat", "COLOR_BGRA32"),
    "K4A_IMAGE_FORMAT_DEPTH16": ("ImageFormat", "DEPTH16"),
    "K4A_IMAGE_FORMAT_IR16": ("ImageFormat", "IR16"),
    "K4A_IMAGE_FORMAT_CUSTOM8": ("ImageFormat", "CUSTOM8"),
    "K4A_IMAGE_FORMAT_CUSTOM16": ("ImageFormat", "CUSTOM16"),
    "K4A_IMAGE_FORMAT_CUSTOM": ("ImageFormat", "CUSTOM"),
    "K4A_FRAMES_PER_SECOND_5": ("Fps", "FPS_5"),
    "K4A_FRAMES_PER_SECOND_15": ("Fps", "FPS_15"),
    "K4A_FRAMES_PER_SECOND_30": ("Fps", "FPS_30"),
    "K4A_WIRED_SYNC_MODE_STANDALONE": ("WiredSyncMode", "STANDALONE"),
    "K4A_WIRED_SYNC_MODE_MASTER": ("WiredSyncMode", "MASTER"),
    "K4A_WIRED_SYNC_MODE_SUBORDINATE": ("WiredSyncMode",
                                         "SUBORDINATE"),
    "K4A_CALIBRATION_TYPE_UNKNOWN": ("CalibrationType", "UNKNOWN"),
    "K4A_CALIBRATION_TYPE_DEPTH": ("CalibrationType", "DEPTH"),
    "K4A_CALIBRATION_TYPE_COLOR": ("CalibrationType", "COLOR"),
    "K4A_CALIBRATION_TYPE_GYRO": ("CalibrationType", "GYRO"),
    "K4A_CALIBRATION_TYPE_ACCEL": ("CalibrationType", "ACCEL"),
    "K4A_CALIBRATION_TYPE_NUM": ("CalibrationType", "NUM"),
    "K4A_CALIBRATION_LENS_DISTORTION_MODEL_UNKNOWN":
        ("CalibrationModelType", "UNKNOWN"),
    "K4A_CALIBRATION_LENS_DISTORTION_MODEL_THETA":
        ("CalibrationModelType", "THETA"),
    "K4A_CALIBRATION_LENS_DISTORTION_MODEL_POLYNOMIAL_3K":
        ("CalibrationModelType", "POLYNOMIAL_3K"),
    "K4A_CALIBRATION_LENS_DISTORTION_MODEL_RATIONAL_6KT":
        ("CalibrationModelType", "RATIONAL_6KT"),
    "K4A_CALIBRATION_LENS_DISTORTION_MODEL_BROWN_CONRADY":
        ("CalibrationModelType", "BROWN_CONRADY"),
    "K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE":
        ("ColorControlCommand", "EXPOSURE_TIME_ABSOLUTE"),
    "K4A_COLOR_CONTROL_AUTO_EXPOSURE_PRIORITY":
        ("ColorControlCommand", "AUTO_EXPOSURE_PRIORITY"),
    "K4A_COLOR_CONTROL_BRIGHTNESS": ("ColorControlCommand",
                                     "BRIGHTNESS"),
    "K4A_COLOR_CONTROL_CONTRAST": ("ColorControlCommand", "CONTRAST"),
    "K4A_COLOR_CONTROL_SATURATION": ("ColorControlCommand",
                                     "SATURATION"),
    "K4A_COLOR_CONTROL_SHARPNESS": ("ColorControlCommand",
                                    "SHARPNESS"),
    "K4A_COLOR_CONTROL_WHITEBALANCE": ("ColorControlCommand",
                                       "WHITEBALANCE"),
    "K4A_COLOR_CONTROL_BACKLIGHT_COMPENSATION":
        ("ColorControlCommand", "BACKLIGHT_COMPENSATION"),
    "K4A_COLOR_CONTROL_GAIN": ("ColorControlCommand", "GAIN"),
    "K4A_COLOR_CONTROL_POWERLINE_FREQUENCY":
        ("ColorControlCommand", "POWERLINE_FREQUENCY"),
    "K4A_COLOR_CONTROL_MODE_AUTO": ("ColorControlMode", "AUTO"),
    "K4A_COLOR_CONTROL_MODE_MANUAL": ("ColorControlMode", "MANUAL"),
    "K4A_FIRMWARE_BUILD_RELEASE": ("FirmwareBuild", "RELEASE"),
    "K4A_FIRMWARE_BUILD_DEBUG": ("FirmwareBuild", "DEBUG"),
    "K4A_FIRMWARE_SIGNATURE_MSFT": ("FirmwareSignature", "MSFT"),
    "K4A_FIRMWARE_SIGNATURE_TEST": ("FirmwareSignature", "TEST"),
    "K4A_FIRMWARE_SIGNATURE_UNSIGNED": ("FirmwareSignature",
                                        "UNSIGNED"),
    "K4A_TRANSFORMATION_INTERPOLATION_TYPE_NEAREST":
        ("InterpolationType", "NEAREST"),
    "K4A_TRANSFORMATION_INTERPOLATION_TYPE_LINEAR":
        ("InterpolationType", "LINEAR"),
}


def load_parity(path: str | Path | None = None) -> dict[str, Any]:
    """Load a parity TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_PARITY_PATH
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise KinectError(f"cannot read parity TOML {path}: {exc}",
                          kind="corrupt") from exc
    except tomllib.TOMLDecodeError as exc:
        raise KinectError(f"bad parity TOML {path}: {exc}",
                          kind="corrupt") from exc
    if not isinstance(data.get("schema"), dict) \
            or data["schema"].get("version") != 1:
        raise KinectError(f"parity TOML {path} is not schema "
                          "version 1", kind="unsupported")
    if not isinstance(data.get("case"), list):
        raise KinectError(f"parity TOML {path} has no [[case]] rows",
                          kind="corrupt")
    return data


def _fixture(name: str) -> Path:
    path = _FIXTURES / name
    if not path.exists():
        raise KinectError(f"missing fixture {path}",
                          kind="missing_fixture")
    return path


def _parse_c_enums(header: Path) -> dict[str, int]:
    """Enumerator → value for every k4a enum in k4atypes.h."""
    text = header.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    values: dict[str, int] = {}
    for block in re.finditer(
            r"typedef\s+enum\s*\{(.*?)\}\s*k4a_\w+;",
            text, flags=re.S):
        current = 0
        for item in block.group(1).split(","):
            item = item.strip()
            if not item:
                continue
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*"
                             r"(?:=\s*(-?\d+))?$", item)
            if not match:
                continue
            name, given = match.group(1), match.group(2)
            if given is not None:
                current = int(given)
            values[name] = current
            current += 1
    return values


def _close(actual: float, expected: float, tol: float,
           what: str) -> None:
    if not math.isclose(actual, expected, abs_tol=tol):
        raise KinectError(
            f"{what}: got {actual!r}, expected {expected!r} "
            f"(tol {tol})", kind="mismatch")


def _silence_sdk_log(lib: K4aLibrary) -> None:
    """Register a no-op SDK log callback (kept alive module-wide).

    The SDK otherwise prints its own diagnostics to stderr —
    useful on a live device, noise in a parity run. The callback
    is stored in a global so ctypes never frees it while the
    library is loaded.
    """
    global _SDK_LOG_SINK
    if _SDK_LOG_SINK is None:
        _SDK_LOG_SINK = _LogCallback(
            lambda context, level, file, line, message: None)
    # min_level 0 is K4A_LOG_LEVEL_CRITICAL: only criticals reach
    # the no-op callback. The SDK's env logger (stderr) is
    # silenced separately — K4A_ENABLE_LOG_TO_STDOUT=0 must be in
    # the environment before the library first logs; a min_level
    # of OFF (5) would report everything, not nothing.
    lib.k4a_set_debug_message_handler(_SDK_LOG_SINK, 0, None)


#: The no-op SDK log callback, kept alive for the process.
_SDK_LOG_SINK: object | None = None


def _sdk_calibration(lib: K4aLibrary, raw: bytes) -> CCalibration:
    """The SDK's mode-specific calibration for the fixture JSON.

    ``k4a_calibration_get_from_raw`` requires the JSON buffer to
    be NUL-terminated *within* the size (calibration.c checks
    ``strnlen(raw, size) < size``) — a plain ``len(raw)`` fails.
    """
    c_cal = CCalibration()
    nulled = raw + b"\x00"
    result = lib.k4a_calibration_get_from_raw(
        nulled, len(nulled), int(k4a_enums.DepthMode.NFOV_UNBINNED),
        int(k4a_enums.ColorResolution.RES_720P),
        ctypes.byref(c_cal))
    if result != 0:
        raise KinectError(
            f"SDK get_from_raw failed (result {result})",
            kind="sdk_failed")
    return c_cal


def _load_fixture_calibration() -> cal.Calibration:
    path = _fixture("calibration.json")
    return cal.parse_raw_calibration(path.read_text(encoding="utf-8"))


def _run_case(case: dict[str, Any]) -> None:
    op = str(case.get("op", ""))
    expect = case.get("expect", {})
    if not isinstance(expect, dict):
        raise KinectError("case expect is not a table",
                          kind="bad_case")

    if op == "enum_values":
        header_values = _parse_c_enums(_HEADER)
        for c_name, want in expect.get("values", {}).items():
            if c_name not in header_values:
                raise KinectError(
                    f"header has no enumerator {c_name}",
                    kind="missing_enumerator")
            if header_values[c_name] != int(want):
                raise KinectError(
                    f"header says {c_name}={header_values[c_name]}, "
                    f"TOML pins {want}", kind="mismatch")
            cls_name, member = _MEMBER_NAMES[c_name]
            got = int(getattr(getattr(k4a_enums, cls_name), member))
            if got != int(want):
                raise KinectError(
                    f"{cls_name}.{member}={got}, header says {want}",
                    kind="mismatch")
        return

    if op == "resolutions":
        for key, want in expect.get("depth", {}).items():
            mode = getattr(k4a_enums.DepthMode, key)
            got = k4a_enums.DEPTH_MODE_DIMS[mode]
            if list(got) != [int(want[0]), int(want[1])]:
                raise KinectError(
                    f"depth {key}: got {got}, want {want}",
                    kind="mismatch")
        for key, want in expect.get("color", {}).items():
            res = getattr(k4a_enums.ColorResolution, key)
            got = k4a_enums.COLOR_RESOLUTION_DIMS[res]
            if list(got) != [int(want[0]), int(want[1])]:
                raise KinectError(
                    f"color {key}: got {got}, want {want}",
                    kind="mismatch")
        return

    if op == "struct_layout":
        structs = {
            "DeviceConfiguration": DeviceConfiguration,
            "Calibration": CCalibration,
            "HardwareVersion": HardwareVersion,
            "ImuSample": ImuSample,
        }
        for name, want in expect.get("sizeof", {}).items():
            got = ctypes.sizeof(structs[name])
            if got != int(want):
                raise KinectError(
                    f"sizeof({name})={got}, C says {want}",
                    kind="mismatch")
        for dotted, want in expect.get("offsets", {}).items():
            struct_name, field_name = dotted.split(".", 1)
            field = getattr(structs[struct_name], field_name)
            if int(field.offset) != int(want):
                raise KinectError(
                    f"offsetof({dotted})={field.offset}, "
                    f"C says {want}", kind="mismatch")
        return

    if op == "config_disable_all":
        cfg = DeviceConfig.disable_all()
        for key, want in expect.items():
            got = getattr(cfg, key)
            if hasattr(got, "name"):
                got = got.name
            if got != want:
                raise KinectError(
                    f"disable_all.{key}={got!r}, want {want!r}",
                    kind="mismatch")
        return

    if op == "depth_decode":
        fixture = _fixture(str(case.get("fixture", "")))
        width, height = int(expect["width"]), int(expect["height"])
        stride = int(expect["stride"])
        rows = decode_depth16(fixture.read_bytes(), width, height,
                              stride)
        for where, want in expect.get("pixels", {}).items():
            y, x = (int(v) for v in str(where).split(","))
            if rows[y][x] != int(want):
                raise KinectError(
                    f"pixel ({x},{y})={rows[y][x]}, want {want}",
                    kind="mismatch")
        return

    if op == "depth_stats":
        fixture = _fixture(str(case.get("fixture", "")))
        width, height = int(expect["width"]), int(expect["height"])
        stride = int(expect["stride"])
        stats = depth_stats(decode_depth16(
            fixture.read_bytes(), width, height, stride))
        for key in ("valid_pixels", "zero_pixels", "min_mm",
                    "max_mm"):
            if getattr(stats, key) != int(expect[key]):
                raise KinectError(
                    f"stats.{key}={getattr(stats, key)}, "
                    f"want {expect[key]}", kind="mismatch")
        _close(stats.mean_mm, float(expect["mean_mm"]), 1e-6,
               "stats.mean_mm")
        _close(stats.valid_fraction,
               float(expect["valid_fraction"]), 1e-9,
               "stats.valid_fraction")
        return

    if op == "calibration_parse":
        calibration = _load_fixture_calibration()
        depth = calibration.depth
        if depth.resolution_width != int(expect["depth_width"]) \
                or depth.resolution_height != int(expect["depth_height"]):
            raise KinectError(
                "depth resolution mismatch", kind="mismatch")
        for key, want in expect.get("depth_intrinsics", {}).items():
            _close(float(getattr(depth.intrinsics, key)),
                   float(want), 1e-6, f"depth intrinsics.{key}")
        if depth.intrinsics.model.name != expect.get(
                "depth_model", "BROWN_CONRADY"):
            raise KinectError("depth model mismatch", kind="mismatch")
        if abs(depth.metric_radius
               - float(expect["depth_metric_radius"])) > 1e-6:
            raise KinectError("depth metric_radius mismatch",
                              kind="mismatch")
        color = calibration.color
        for key, want in expect.get("color_intrinsics", {}).items():
            _close(float(getattr(color.intrinsics, key)),
                   float(want), 1e-6, f"color intrinsics.{key}")
        gyro_t = calibration.gyro.translation
        for i, want in enumerate(expect.get("gyro_translation_mm",
                                            [])):
            _close(float(gyro_t[i]), float(want), 1e-6,
                   f"gyro translation[{i}]")
        return

    if op == "geometry_roundtrip":
        calibration = _load_fixture_calibration()
        depth_kind = k4a_enums.CalibrationType.DEPTH
        for spec in expect.get("points", []):
            pixel = (float(spec["u"]), float(spec["v"]))
            depth_mm = float(spec["depth_mm"])
            point, valid = calibration.unproject_2d_to_3d(
                pixel, depth_mm, depth_kind)
            if valid != bool(spec.get("valid", True)):
                raise KinectError(
                    f"unproject {pixel} valid={valid}, "
                    f"want {spec.get('valid', True)}",
                    kind="mismatch")
            for i, want in enumerate(spec["point_mm"]):
                _close(point[i], float(want), 0.05,
                       f"unproject {pixel} [{i}]")
            back, valid = calibration.project_3d_to_2d(
                point, depth_kind)
            if valid:
                _close(back[0], pixel[0], 0.01, "roundtrip u")
                _close(back[1], pixel[1], 0.01, "roundtrip v")
        return

    if op == "sdk_geometry":
        # The real SDK, same calibration + points, through ctypes.
        lib = load_library()
        _silence_sdk_log(lib)
        raw = _fixture("calibration.json").read_bytes()
        c_cal = _sdk_calibration(lib, raw)
        mine = cal.from_ctypes(c_cal)
        depth_kind = k4a_enums.CalibrationType.DEPTH
        tol = float(expect.get("tolerance_mm", 0.05))
        for spec in expect.get("points", []):
            pixel = (float(spec["u"]), float(spec["v"]))
            depth_mm = float(spec["depth_mm"])
            mine_point, mine_valid = mine.unproject_2d_to_3d(
                pixel, depth_mm, depth_kind)
            sdk_point = Float3()
            sdk_valid = ctypes.c_int(0)
            result = lib.k4a_calibration_2d_to_3d(
                ctypes.byref(c_cal), ctypes.byref(Float2(*pixel)),
                float(depth_mm), int(depth_kind), int(depth_kind),
                ctypes.byref(sdk_point), ctypes.byref(sdk_valid))
            if result != 0:
                raise KinectError(
                    f"SDK 2d_to_3d failed (result {result})",
                    kind="sdk_failed")
            if bool(sdk_valid.value) != mine_valid:
                raise KinectError(
                    f"SDK valid={bool(sdk_valid.value)} vs mine "
                    f"{mine_valid} at {pixel}", kind="mismatch")
            for i, mine_v in enumerate(mine_point):
                sdk_v = (sdk_point.x, sdk_point.y, sdk_point.z)[i]
                _close(mine_v, float(sdk_v), tol,
                       f"SDK 2d_to_3d {pixel} [{i}]")
            # And back: SDK 3d_to_2d of the same point, when the
            # unprojection was valid (an invalid point is (0,0,0)
            # and projects nowhere meaningful).
            if not mine_valid:
                continue
            sdk_pixel = Float2()
            result = lib.k4a_calibration_3d_to_2d(
                ctypes.byref(c_cal),
                ctypes.byref(Float3(*mine_point)),
                int(depth_kind), int(depth_kind),
                ctypes.byref(sdk_pixel), ctypes.byref(sdk_valid))
            if result != 0:
                raise KinectError(
                    f"SDK 3d_to_2d failed (result {result})",
                    kind="sdk_failed")
            _close(float(sdk_pixel.x), pixel[0], 0.05,
                   f"SDK 3d_to_2d {pixel} u")
            _close(float(sdk_pixel.y), pixel[1], 0.05,
                   f"SDK 3d_to_2d {pixel} v")
        # The SDK's own calibration must match the fixture's
        # parsed values field by field.
        sdk_depth = mine.depth
        for key, want in expect.get("sdk_intrinsics", {}).items():
            _close(float(getattr(sdk_depth.intrinsics, key)),
                   float(want), 1e-6, f"SDK intrinsics.{key}")
        return

    if op == "sdk_point_cloud":
        from .device import Transformation

        lib = load_library()
        _silence_sdk_log(lib)
        raw = _fixture("calibration.json").read_bytes()
        c_cal = _sdk_calibration(lib, raw)
        fixture_cal = _load_fixture_calibration()
        fixture = _fixture(str(case.get("fixture", "")))
        width = int(expect["width"])
        height = int(expect["height"])
        stride = int(expect["stride"])
        frame = Frame(fmt=k4a_enums.ImageFormat.DEPTH16,
                      width=width, height=height, stride=stride,
                      data=fixture.read_bytes())
        sdk_cal = cal.from_ctypes(c_cal)
        try:
            transformation = Transformation(lib, sdk_cal)
        except Exception as exc:
            # The SDK's transform engine is GL-based
            # (libdepthengine); with no display it cannot start,
            # and that is the box's wall, not the port's. A display
            # present but still failing is a real failure.
            if os.environ.get("DISPLAY"):
                raise
            raise ParitySkipped(
                "SDK transform engine needs a display "
                "(libdepthengine is GL-based); run under xvfb-run "
                f"for this case ({exc})") from exc
        sdk_points = transformation.depth_to_point_cloud(frame)
        mine_points = cal.point_cloud(
            fixture_cal, decode_depth16(
                fixture.read_bytes(), width, height, stride))
        if len(sdk_points) != len(mine_points):
            raise KinectError(
                f"point counts differ: SDK {len(sdk_points)} vs "
                f"mine {len(mine_points)}", kind="mismatch")
        tol = float(expect.get("tolerance_mm", 0.05))
        for (sx, sy, sz), (mx, my, mz) in zip(
                sdk_points, mine_points, strict=True):
            _close(mx, sx, tol, "point cloud x")
            _close(my, sy, tol, "point cloud y")
            _close(mz, sz, tol, "point cloud z")
        return

    if op == "probe_report":
        report = run_probe()
        if not isinstance(report.sdk_present, bool) \
                or not isinstance(report.device_count, int) \
                or report.device_count < 0:
            raise KinectError("probe report shape is wrong",
                              kind="mismatch")
        json.dumps(report.as_dict())  # must be JSON-serialisable
        if report.sdk_present != bool(expect.get("sdk_present",
                                                 True)):
            raise KinectError(
                "probe says the SDK is absent", kind="mismatch")
        return

    if op == "jpeg_sniff":
        fixture = _fixture(str(case.get("fixture", "")))
        data = fixture.read_bytes()
        if not is_jpeg(data):
            raise KinectError("fixture is not a JPEG payload",
                              kind="mismatch")
        frame = frame_from_buffer(
            k4a_enums.ImageFormat.COLOR_MJPG,
            int(expect["width"]), int(expect["height"]), 0, data)
        if frame.width != int(expect["width"]) \
                or frame.height != int(expect["height"]):
            raise KinectError("MJPG frame geometry mismatch",
                              kind="mismatch")
        return

    if op == "pinhole_xyz":
        from .frames import depth_to_xyz_mm

        for spec in expect.get("points", []):
            got = depth_to_xyz_mm(
                float(spec["depth_mm"]), int(spec["x"]),
                int(spec["y"]), float(spec["fx"]),
                float(spec["fy"]), float(spec["cx"]),
                float(spec["cy"]))
            for i, want in enumerate(spec["xyz_mm"]):
                _close(got[i], float(want), 1e-6,
                       f"pinhole xyz [{i}]")
        return

    raise KinectError(f"unknown parity op {op!r}", kind="bad_case")


def run_parity(source: str | Path | None = None) -> dict[str, Any]:
    """Run the parity cases; a missing TOML fails, never skips.

    Environment-gated cases (the SDK transform engine, which
    needs a display) land in ``skipped`` with the reason, never
    in ``failed`` and never counted as passed.
    """
    data = load_parity(source)
    passed: list[str] = []
    failures: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for case in data["case"]:
        case_id = str(case.get("id", "<unnamed>"))
        try:
            _run_case(case)
        except ParitySkipped as exc:
            skipped.append({"id": case_id, "reason": str(exc)})
        except Exception as exc:  # noqa: BLE001 — one case's failure
            failures.append({"id": case_id,
                             "reason": f"{type(exc).__name__}: {exc}"})
        else:
            passed.append(case_id)
    return {"passed": len(passed),
            "failed": [entry["id"] for entry in failures],
            "passed_ids": passed,
            "failures": failures,
            "total": len(data["case"]),
            "skipped": skipped}
