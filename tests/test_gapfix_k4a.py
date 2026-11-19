"""Kinect / k4a-sys port tests (lane 122, census 3A.30).

Two halves, like the port: the pure-Python half (enums, structs,
calibration parse, geometry, frame decode) runs anywhere; the
live-SDK half (libk4a calls, the probe, the parity SDK cases)
runs against the real Azure Kinect SDK 1.4.2 installed on this
box and reports honestly when there is no device — the probe
returns a report, it never raises for a missing device.

The transform-engine half (the SDK's depth→point-cloud, which
needs the GL-based libdepthengine and therefore a display)
skips cleanly without one and runs under ``xvfb-run``; the
parity runner records that case as skipped, never failed, when
there is no display.

Every pinned number comes from ``kinect-parity.toml``, measured
by running the code and cross-checked against the real libk4a
binary — same calibration, same points.
"""

import ctypes
import json
import os
from pathlib import Path

import pytest

from nomorals.hardware.kinect import attribution as kattrib
from nomorals.hardware.kinect import calibration as kcal
from nomorals.hardware.kinect import enums as kenums
from nomorals.hardware.kinect import parity as kparity
from nomorals.hardware.kinect.attribution import verify_attribution
from nomorals.hardware.kinect.bindings import (
    Calibration as CCalibration,
)
from nomorals.hardware.kinect.bindings import (
    DeviceConfiguration,
    Float2,
    Float3,
    HardwareVersion,
    ImuSample,
    load_library,
)
from nomorals.hardware.kinect.device import (
    DeviceConfig,
    KinectDevice,
    Transformation,
)
from nomorals.hardware.kinect.enums import (
    CalibrationModelType,
    CalibrationType,
    ColorResolution,
    DepthMode,
    ImageFormat,
)
from nomorals.hardware.kinect.errors import (
    CalibrationError,
    DeviceError,
    KinectError,
    NoDeviceError,
    NoSdkError,
)
from nomorals.hardware.kinect.frames import (
    Frame,
    decode_depth16,
    depth_stats,
    depth_to_xyz_mm,
    frame_from_buffer,
    is_jpeg,
)
from nomorals.hardware.kinect.parity import run_parity
from nomorals.hardware.kinect.probe import main as probe_main
from nomorals.hardware.kinect.probe import probe as probe_fn

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "kinect"


# -- enums -----------------------------------------------------------

def test_depth_mode_values_match_the_header():
    assert int(DepthMode.OFF) == 0
    assert int(DepthMode.NFOV_2X2BINNED) == 1
    assert int(DepthMode.NFOV_UNBINNED) == 2
    assert int(DepthMode.WFOV_2X2BINNED) == 3
    assert int(DepthMode.WFOV_UNBINNED) == 4
    assert int(DepthMode.PASSIVE_IR) == 5


def test_color_resolution_and_image_format_values():
    assert int(ColorResolution.OFF) == 0
    assert int(ColorResolution.RES_720P) == 1
    assert int(ColorResolution.RES_3072P) == 6
    assert int(ImageFormat.COLOR_MJPG) == 0
    assert int(ImageFormat.DEPTH16) == 4
    assert int(ImageFormat.CUSTOM) == 8


def test_calibration_model_type_has_no_negative_unknown():
    # The C enum starts UNKNOWN at 0 (k4atypes.h); there is no
    # -1 and no NUM enumerator for the model type.
    assert int(CalibrationModelType.UNKNOWN) == 0
    assert int(CalibrationModelType.BROWN_CONRADY) == 4
    assert int(CalibrationType.UNKNOWN) == -1
    assert int(CalibrationType.NUM) == 4


def test_resolution_tables_match_the_sdk_modes():
    assert kenums.DEPTH_MODE_DIMS[DepthMode.NFOV_UNBINNED] == (640, 576)
    assert kenums.DEPTH_MODE_DIMS[DepthMode.NFOV_2X2BINNED] == (320, 288)
    assert kenums.DEPTH_MODE_DIMS[DepthMode.WFOV_UNBINNED] == (1024, 1024)
    assert kenums.COLOR_RESOLUTION_DIMS[
        ColorResolution.RES_720P] == (1280, 720)
    assert kenums.COLOR_RESOLUTION_DIMS[
        ColorResolution.RES_3072P] == (4096, 3072)


def test_bytes_per_pixel_is_none_for_compressed_formats():
    assert kenums.bytes_per_pixel(ImageFormat.DEPTH16) == 2
    assert kenums.bytes_per_pixel(ImageFormat.COLOR_BGRA32) == 4
    assert kenums.bytes_per_pixel(ImageFormat.COLOR_MJPG) is None
    assert kenums.bytes_per_pixel(ImageFormat.CUSTOM) is None


# -- errors ----------------------------------------------------------

def test_each_error_carries_its_kind():
    for cls, kind in ((NoSdkError, "no_sdk"),
                      (NoDeviceError, "no_device"),
                      (DeviceError, "device_failed"),
                      (CalibrationError, "calibration_failed")):
        err = cls("boom")
        assert isinstance(err, KinectError)
        assert err.kind == kind
        assert "boom" in str(err)


def test_load_library_names_every_name_it_tried():
    with pytest.raises(NoSdkError) as caught:
        load_library(("libk4a-nope.so.9", "libk4a-also-nope.so"))
    assert "libk4a-nope.so.9" in str(caught.value)
    assert "libk4a-also-nope.so" in str(caught.value)


def test_load_library_honours_a_missing_k4a_lib_override(tmp_path,
                                                         monkeypatch):
    missing = tmp_path / "libk4a.so.1.4"
    monkeypatch.setenv("K4A_LIB", str(missing))
    with pytest.raises(NoSdkError):
        load_library()


# -- ctypes structs --------------------------------------------------

def test_struct_sizes_match_the_gcc_probe():
    assert ctypes.sizeof(DeviceConfiguration) == 36
    assert ctypes.sizeof(CCalibration) == 1032
    assert ctypes.sizeof(HardwareVersion) == 56
    assert ctypes.sizeof(ImuSample) == 48


def test_struct_field_offsets_match_c():
    assert DeviceConfiguration.color_format.offset == 0
    assert DeviceConfiguration.depth_mode.offset == 8
    assert DeviceConfiguration.disable_streaming_indicator.offset == 32
    assert CCalibration.depth_camera_calibration.offset == 0
    assert CCalibration.color_camera_calibration.offset == 128
    assert CCalibration.extrinsics.offset == 256
    assert CCalibration.depth_mode.offset == 1024
    assert HardwareVersion.rgb.offset == 0
    assert HardwareVersion.firmware_build.offset == 48
    assert ImuSample.acc_sample.offset == 4
    assert ImuSample.gyro_timestamp_usec.offset == 40


# -- device config ---------------------------------------------------

def test_device_config_defaults_are_the_streaming_example_shape():
    cfg = DeviceConfig()
    assert cfg.color_format is ImageFormat.COLOR_MJPG
    assert cfg.color_resolution is ColorResolution.RES_720P
    assert cfg.depth_mode is DepthMode.WFOV_2X2BINNED
    assert cfg.camera_fps.name == "FPS_30"
    assert cfg.wired_sync_mode.name == "STANDALONE"


def test_device_config_disable_all_matches_the_c_macro():
    cfg = DeviceConfig.disable_all()
    assert cfg.color_format is ImageFormat.COLOR_MJPG
    assert cfg.color_resolution is ColorResolution.OFF
    assert cfg.depth_mode is DepthMode.OFF
    assert cfg.camera_fps.name == "FPS_30"
    assert cfg.synchronized_images_only is False
    assert cfg.depth_delay_off_color_usec == 0
    assert cfg.wired_sync_mode.name == "STANDALONE"
    assert cfg.subordinate_delay_off_master_usec == 0
    assert cfg.disable_streaming_indicator is False


def test_device_config_to_ctypes_round_trips():
    raw = DeviceConfig.disable_all().to_ctypes()
    assert raw.color_format == 0  # MJPG
    assert raw.color_resolution == 0  # OFF
    assert raw.depth_mode == 0  # OFF


# -- calibration -----------------------------------------------------

def _fixture_calibration():
    return kcal.parse_raw_calibration(
        (FIXTURES / "calibration.json").read_text(encoding="utf-8"))


def test_parse_raw_calibration_mode_specific_values():
    cal = _fixture_calibration()
    assert cal.depth.resolution_width == 640
    assert cal.depth.resolution_height == 576
    assert cal.depth.intrinsics.model is CalibrationModelType.BROWN_CONRADY
    assert cal.depth.intrinsics.cx == pytest.approx(321.73828125, abs=1e-6)
    assert cal.depth.intrinsics.cy == pytest.approx(333.33843994140625,
                                                    abs=1e-6)
    assert cal.depth.intrinsics.fx == pytest.approx(504.6473693847656,
                                                    abs=1e-6)
    assert cal.depth.intrinsics.fy == pytest.approx(504.75738525390625,
                                                    abs=1e-6)
    assert cal.depth.metric_radius == pytest.approx(1.737323045730591,
                                                    abs=1e-6)
    # The color camera is a legacy 16:9 calibration (3840x2160);
    # the parse converts it to 4:3 first, then pixelizes for 720p.
    assert cal.color.resolution_width == 1280
    assert cal.color.resolution_height == 720
    assert cal.color.intrinsics.cx == pytest.approx(634.7147674560547,
                                                    abs=1e-6)
    assert cal.depth_mode == int(DepthMode.NFOV_UNBINNED)
    assert cal.color_resolution == int(ColorResolution.RES_720P)


def test_parse_raw_calibration_extrinsics_are_depth_centred():
    cal = _fixture_calibration()
    # Depth is the world: depth->depth is the identity.
    identity = cal.extrinsics(CalibrationType.DEPTH,
                              CalibrationType.DEPTH)
    assert identity.translation == (0.0, 0.0, 0.0)
    assert identity.rotation == (1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0)
    # The fixture's color Rt, metres -> mm (SDK x1000).
    depth_to_color = cal.extrinsics(CalibrationType.DEPTH,
                                    CalibrationType.COLOR)
    assert depth_to_color.translation[0] == pytest.approx(-31.942267,
                                                         abs=1e-4)
    assert depth_to_color.translation[1] == pytest.approx(-2.476229,
                                                         abs=1e-4)
    assert depth_to_color.translation[2] == pytest.approx(3.648853,
                                                         abs=1e-4)
    # IMU poses come from the InertialSensors section.
    assert cal.gyro.translation == (0.0, 0.0, 0.0)
    assert cal.accel.translation[0] == pytest.approx(-50.794426, abs=1e-4)


def test_parse_raw_calibration_rejects_bad_input():
    with pytest.raises(CalibrationError):
        kcal.parse_raw_calibration("{not json")
    with pytest.raises(CalibrationError):
        kcal.parse_raw_calibration("{}")
    with pytest.raises(CalibrationError):
        kcal.parse_raw_calibration(
            '{"CalibrationInformation": {"Cameras": []}}')
    with pytest.raises(CalibrationError):
        kcal.parse_raw_calibration(
            (FIXTURES / "calibration.json").read_text(
                encoding="utf-8"),
            depth_mode=DepthMode.OFF)


def test_parse_raw_calibration_accepts_friendly_location_names():
    raw = json.loads((FIXTURES / "calibration.json").read_text(
        encoding="utf-8"))
    for camera in raw["CalibrationInformation"]["Cameras"]:
        if camera["Location"] == "CALIBRATION_CameraLocationD0":
            camera["Location"] = "Depth"
        elif camera["Location"] == "CALIBRATION_CameraLocationPV0":
            camera["Location"] = "Color"
    cal = kcal.parse_raw_calibration(json.dumps(raw))
    assert cal.depth.resolution_width == 640
    assert cal.color.resolution_width == 1280


def test_metric_radius_floor_is_1_point_7():
    raw = json.loads((FIXTURES / "calibration.json").read_text(
        encoding="utf-8"))
    for camera in raw["CalibrationInformation"]["Cameras"]:
        camera["MetricRadius"] = 0.0
    cal = kcal.parse_raw_calibration(json.dumps(raw))
    assert cal.depth.metric_radius == pytest.approx(1.7, abs=1e-6)
    assert cal.color.metric_radius == pytest.approx(1.7, abs=1e-6)


def test_geometry_roundtrip_on_the_fixture():
    cal = _fixture_calibration()
    depth = CalibrationType.DEPTH
    for pixel, depth_mm in (((320.0, 288.0), 1500.0),
                            ((100.0, 100.0), 1500.0),
                            ((500.0, 400.0), 1500.0)):
        point, valid = cal.unproject_2d_to_3d(pixel, depth_mm, depth)
        assert valid
        back, valid = cal.project_3d_to_2d(point, depth)
        assert valid
        assert back[0] == pytest.approx(pixel[0], abs=0.01)
        assert back[1] == pytest.approx(pixel[1], abs=0.01)
    # Zero depth unprojects to the origin, valid, like the SDK.
    point, valid = cal.unproject_2d_to_3d((320.0, 288.0), 0.0, depth)
    assert valid and point == (0.0, 0.0, 0.0)
    # A point behind the camera does not project.
    _, valid = cal.project_3d_to_2d((0.0, 0.0, -100.0), depth)
    assert not valid


def test_point_cloud_on_a_small_grid():
    cal = _fixture_calibration()
    grid = [[1000, 1000, 1000],
            [1000, 0, 1000],
            [1000, 1000, 1000]]
    points = kcal.point_cloud(cal, grid)
    assert len(points) == 9
    assert points[4] == (0.0, 0.0, 0.0)  # the zero-depth pixel
    assert points[0][2] == 1000.0
    assert points[0][0] < 0.0  # left of centre is negative X


def test_calibration_to_ctypes_and_back():
    cal = _fixture_calibration()
    raw = kcal.to_ctypes(cal)
    assert raw.depth_mode == int(DepthMode.NFOV_UNBINNED)
    assert raw.color_resolution == int(ColorResolution.RES_720P)
    back = kcal.from_ctypes(raw)
    assert back.depth.intrinsics.cx == pytest.approx(
        cal.depth.intrinsics.cx, abs=1e-4)
    assert back.depth.intrinsics.fx == pytest.approx(
        cal.depth.intrinsics.fx, abs=1e-4)
    assert back.color.intrinsics.cy == pytest.approx(
        cal.color.intrinsics.cy, abs=1e-4)


# -- frames ----------------------------------------------------------

def test_decode_depth16_reads_the_fixture():
    data = (FIXTURES / "depth_nfov_640x576.raw").read_bytes()
    grid = decode_depth16(data, 640, 576, 1280)
    assert len(grid) == 576 and len(grid[0]) == 640
    assert grid[0][0] == 1000
    assert grid[0][99] == 1099
    assert grid[105][205] == 0  # the zero block
    assert grid[575][639] == 1064


def test_decode_depth16_rejects_a_short_buffer():
    with pytest.raises(KinectError, match="too small"):
        decode_depth16(b"\x00\x01", 640, 576, 1280)


def test_depth_stats_on_the_fixture():
    data = (FIXTURES / "depth_nfov_640x576.raw").read_bytes()
    stats = depth_stats(decode_depth16(data, 640, 576, 1280))
    assert stats.valid_pixels == 368540
    assert stats.zero_pixels == 100
    assert stats.min_mm == 1000
    assert stats.max_mm == 1148
    assert stats.mean_mm == pytest.approx(1071.6003147555218, abs=1e-6)
    assert stats.valid_fraction == pytest.approx(0.9997287326388888,
                                                 abs=1e-9)


def test_jpeg_sniff_on_the_fixture():
    data = (FIXTURES / "color_mjpg.jpg").read_bytes()
    assert is_jpeg(data)
    assert not is_jpeg(b"not a jpeg")
    frame = frame_from_buffer(ImageFormat.COLOR_MJPG, 64, 48, 0, data)
    assert frame.width == 64 and frame.height == 48
    assert frame.fmt is ImageFormat.COLOR_MJPG


def test_depth_to_xyz_mm_is_the_pinhole_model():
    x, y, z = depth_to_xyz_mm(
        1500.0, 320, 288, 504.6473693847656, 504.75738525390625,
        321.73828125, 333.33843994140625)
    assert (x, y, z) == pytest.approx(
        (-5.166819512363267, -134.73336279745513, 1500.0), abs=1e-6)


def test_frame_from_buffer_rejects_a_short_depth_buffer():
    with pytest.raises(KinectError, match="needs 737280 bytes"):
        frame_from_buffer(ImageFormat.DEPTH16, 640, 576, 1280, b"\x00")


# -- live SDK: probe + device ----------------------------------------

def test_library_loads_and_reports_no_device_on_this_box():
    lib = load_library()
    assert lib.installed_count() == 0


def test_open_without_a_device_names_the_count():
    with pytest.raises(NoDeviceError) as caught:
        KinectDevice.open(0)
    assert "0 device(s)" in str(caught.value)


def test_probe_reports_no_device_honestly():
    report = probe_fn()
    assert report.sdk_present is True
    assert report.device_count == 0
    assert report.devices == []
    assert "No Azure Kinect device connected" in report.summary()
    data = report.as_dict()
    json.dumps(data)  # the report is JSON-serialisable
    assert data["device_count"] == 0
    assert data["frames_captured"] == 0


def test_probe_cli_runs(capsys):
    # No device on this box, but the SDK loads — exit 0, JSON out.
    code = probe_main(["--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["device_count"] == 0
    assert data["sdk_present"] is True


def test_transformation_rejects_a_wrong_format_frame():
    if not os.environ.get("DISPLAY"):
        pytest.skip("SDK transform engine needs a display")
    lib = load_library()
    cal = _fixture_calibration()
    with Transformation(lib, cal) as transform:
        frame = Frame(fmt=ImageFormat.COLOR_MJPG, width=64, height=48,
                      stride=0, data=b"\xff\xd8\xff\xe0")
        with pytest.raises(KinectError):
            transform.depth_to_point_cloud(frame)


def test_transformation_point_cloud_matches_python_under_xvfb():
    if not os.environ.get("DISPLAY"):
        pytest.skip("SDK transform engine needs a display")
    lib = load_library()
    cal = _fixture_calibration()
    data = (FIXTURES / "depth_nfov_640x576.raw").read_bytes()
    frame = Frame(fmt=ImageFormat.DEPTH16, width=640, height=576,
                  stride=1280, data=data)
    with Transformation(lib, cal) as transform:
        sdk_points = transform.depth_to_point_cloud(frame)
    mine = kcal.point_cloud(cal, decode_depth16(data, 640, 576, 1280))
    assert len(sdk_points) == len(mine)
    # The SDK writes int16 millimetres: 1 mm covers the rounding.
    worst = 0.0
    for (sx, sy, sz), (mx, my, mz) in zip(sdk_points, mine, strict=True):
        worst = max(worst, abs(sx - mx), abs(sy - my), abs(sz - mz))
    assert worst <= 1.0


# -- parity + attribution --------------------------------------------

def test_parity_green_or_cleanly_skipped():
    result = run_parity()
    assert result["failed"] == []
    assert result["passed"] + len(result["skipped"]) == result["total"]
    assert result["total"] == 18
    # The only case the environment can gate is the SDK point
    # cloud, which needs a display for the GL depth engine.
    assert {entry["id"] for entry in result["skipped"]} <= {
        "sdk-point-cloud"}


def test_attribution_verifies_every_upstream_hash():
    result = verify_attribution()
    assert result["failed"] == []
    assert result["skipped"] == []
    # 30 upstream files + 1 SDK fixture + 16 Devon files.
    assert result["verified"] == 47


def test_attribution_toml_names_the_real_licence():
    data = kattrib.load_attribution()
    assert data["meta"]["license"] == "MIT"
    assert data["meta"]["commit"].startswith("ffc3815")
    assert len(data["upstream"]) == 30
    assert len(data["sdk"]) == 1
    assert len(data["devon"]) == 16


def test_parity_toml_is_versioned():
    data = kparity.load_parity()
    assert data["schema"]["version"] == 1
    assert len(data["case"]) == 18


def test_hardware_package_exports():
    import nomorals.hardware.kinect as kinect

    assert hasattr(kinect, "KinectDevice")
    assert hasattr(kinect, "probe")
    assert hasattr(kinect, "run_parity")
    assert hasattr(kinect, "verify_attribution")
    assert hasattr(kinect, "parse_raw_calibration")
    assert hasattr(kinect, "point_cloud")


def test_sdk_geometry_agrees_with_python_to_float32():
    # The parity runner's SDK case is the cross-check; here the
    # direct comparison pins the numbers it checks.
    lib = load_library()
    kparity._silence_sdk_log(lib)
    raw = (FIXTURES / "calibration.json").read_bytes()
    c_cal = kparity._sdk_calibration(lib, raw)
    sdk = kcal.from_ctypes(c_cal)
    mine = _fixture_calibration()
    assert sdk.depth.intrinsics.cx == pytest.approx(
        mine.depth.intrinsics.cx, abs=1e-4)
    assert sdk.depth.intrinsics.fx == pytest.approx(
        mine.depth.intrinsics.fx, abs=1e-4)
    assert sdk.color.intrinsics.cx == pytest.approx(
        mine.color.intrinsics.cx, abs=1e-4)
    assert sdk.color.intrinsics.fx == pytest.approx(
        mine.color.intrinsics.fx, abs=1e-4)
    point, valid = mine.unproject_2d_to_3d(
        (320.0, 288.0), 1500.0, CalibrationType.DEPTH)
    assert valid
    sdk_point = Float3()
    sdk_valid = ctypes.c_int(0)
    src_pixel = Float2(320.0, 288.0)
    result = lib.k4a_calibration_2d_to_3d(
        ctypes.byref(c_cal), ctypes.byref(src_pixel),
        1500.0, 0, 0, ctypes.byref(sdk_point),
        ctypes.byref(sdk_valid))
    assert result == 0 and bool(sdk_valid.value)
    for mine_v, sdk_v in zip(
            point, (sdk_point.x, sdk_point.y, sdk_point.z),
            strict=True):
        assert mine_v == pytest.approx(sdk_v, abs=0.05)
