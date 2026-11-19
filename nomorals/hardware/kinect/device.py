"""KinectDevice — the live device, over the ctypes bindings.

One object owns one open device handle: open → (optionally)
start cameras → get captures → stop → close. Every operation
that can fail names its reason (:class:`NoDeviceError`,
:class:`DeviceError`, :class:`CaptureError`) instead of
returning a bare false, and every handle the SDK hands out is
released — images after their buffers are copied, captures after
their images are read.

The device also owns the transformation handle: with a
calibration, ``transformation()`` yields a ``Transformation``
that turns depth frames into color-camera depth frames and
into XYZ point clouds through the real SDK, not a re-derive.
"""

from __future__ import annotations

import ctypes
import struct
from ctypes import byref, c_bool, c_char, c_int, c_int32, c_size_t, c_uint8
from dataclasses import dataclass
from types import TracebackType

from . import calibration as _cal
from .bindings import (
    Calibration as CCalibration,
)
from .bindings import (
    DeviceConfiguration,
    HardwareVersion,
    ImuSample,
    K4aLibrary,
    _DestroyCallback,
    load_library,
)
from .enums import (
    CalibrationType,
    ColorControlCommand,
    ColorControlMode,
    ColorResolution,
    DepthMode,
    Fps,
    ImageFormat,
    WiredSyncMode,
)
from .errors import (
    CaptureError,
    DeviceError,
    KinectError,
    NoDeviceError,
)
from .frames import Capture, Frame

__all__ = [
    "DeviceConfig",
    "KinectDevice",
    "Transformation",
]

#: How long get_capture waits for a frame by default (ms).
DEFAULT_TIMEOUT_MS = 1000


@dataclass(frozen=True)
class DeviceConfig:
    """The k4a_device_start_cameras() configuration.

    Defaults are the streaming example's shape at a modest
    resolution: MJPG color at 720p, WFOV 2x2-binned depth, 30 FPS.
    """

    color_format: ImageFormat = ImageFormat.COLOR_MJPG
    color_resolution: ColorResolution = ColorResolution.RES_720P
    depth_mode: DepthMode = DepthMode.WFOV_2X2BINNED
    camera_fps: Fps = Fps.FPS_30
    synchronized_images_only: bool = False
    depth_delay_off_color_usec: int = 0
    wired_sync_mode: WiredSyncMode = WiredSyncMode.STANDALONE
    subordinate_delay_off_master_usec: int = 0
    disable_streaming_indicator: bool = False

    @classmethod
    def disable_all(cls) -> DeviceConfig:
        """K4A_DEVICE_CONFIG_INIT_DISABLE_ALL — every sensor off."""
        return cls(
            color_format=ImageFormat.COLOR_MJPG,
            color_resolution=ColorResolution.OFF,
            depth_mode=DepthMode.OFF,
            camera_fps=Fps.FPS_30,
            synchronized_images_only=False,
            depth_delay_off_color_usec=0,
            wired_sync_mode=WiredSyncMode.STANDALONE,
            subordinate_delay_off_master_usec=0,
            disable_streaming_indicator=False,
        )

    def to_ctypes(self) -> DeviceConfiguration:
        """The C struct the SDK's start_cameras takes."""
        return DeviceConfiguration(
            color_format=int(self.color_format),
            color_resolution=int(self.color_resolution),
            depth_mode=int(self.depth_mode),
            camera_fps=int(self.camera_fps),
            synchronized_images_only=self.synchronized_images_only,
            depth_delay_off_color_usec=self.depth_delay_off_color_usec,
            wired_sync_mode=int(self.wired_sync_mode),
            subordinate_delay_off_master_usec=(
                self.subordinate_delay_off_master_usec),
            disable_streaming_indicator=self.disable_streaming_indicator,
        )


def _check(result: int, what: str) -> None:
    if result != 0:  # K4A_RESULT_SUCCEEDED
        raise DeviceError(f"{what} failed (k4a result {result})")


class KinectDevice:
    """One open Azure Kinect device."""

    def __init__(self, lib: K4aLibrary | None = None,
                 index: int = 0) -> None:
        self._lib = lib if lib is not None else load_library()
        self.index = index
        self._handle: int | None = None
        self._cameras_started = False
        self._imu_started = False

    # -- lifecycle -------------------------------------------------

    @classmethod
    def open(cls, index: int = 0,
             lib: K4aLibrary | None = None) -> KinectDevice:
        """Open device ``index`` (0 is the default device)."""
        device = cls(lib=lib, index=index)
        count = device._lib.installed_count()
        if index >= count:
            raise NoDeviceError(
                f"device {index} does not exist: {count} device(s) "
                "connected")
        handle = ctypes.c_void_p()
        result = device._lib.k4a_device_open(index, byref(handle))
        if result != 0 or not handle.value:
            raise DeviceError(
                f"could not open device {index} "
                f"(k4a result {result})")
        device._handle = handle.value
        return device

    def close(self) -> None:
        """Stop everything and close the handle (idempotent)."""
        if self._handle is None:
            return
        if self._cameras_started:
            self.stop_cameras()
        if self._imu_started:
            self.stop_imu()
        self._lib.k4a_device_close(self._handle)
        self._handle = None

    def __enter__(self) -> KinectDevice:
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        """True while the device handle is held."""
        return self._handle is not None

    def _require_open(self) -> int:
        if self._handle is None:
            raise DeviceError("device is closed")
        return self._handle

    # -- identity --------------------------------------------------

    @property
    def installed_count(self) -> int:
        """Devices the SDK sees right now."""
        return self._lib.installed_count()

    def serial_number(self) -> str:
        """The device serial number (two-call buffer pattern)."""
        handle = self._require_open()
        size = c_size_t(0)
        result = self._lib.k4a_device_get_serialnum(
            handle, None, byref(size))
        if result != 2:  # K4A_BUFFER_RESULT_TOO_SMALL
            raise DeviceError(
                "could not size the serial number "
                f"(k4a buffer result {result})")
        buf = (c_char * size.value)()
        result = self._lib.k4a_device_get_serialnum(
            handle, buf, byref(size))
        if result != 0:
            raise DeviceError(
                "could not read the serial number "
                f"(k4a buffer result {result})")
        return buf.value.decode("utf-8", "replace")

    def version(self) -> dict[str, object]:
        """Firmware versions + build / signature, as a dict."""
        handle = self._require_open()
        ver = HardwareVersion()
        _check(self._lib.k4a_device_get_version(handle, byref(ver)),
               "get version")

        def as_dict(v) -> dict[str, int]:
            return {"major": int(v.major), "minor": int(v.minor),
                    "iteration": int(v.iteration)}

        return {
            "rgb": as_dict(ver.rgb),
            "depth": as_dict(ver.depth),
            "audio": as_dict(ver.audio),
            "depth_sensor": as_dict(ver.depth_sensor),
            "firmware_build": int(ver.firmware_build),
            "firmware_signature": int(ver.firmware_signature),
        }

    def sync_jack(self) -> tuple[bool, bool]:
        """(sync-in connected, sync-out connected)."""
        handle = self._require_open()
        sync_in, sync_out = c_bool(False), c_bool(False)
        _check(self._lib.k4a_device_get_sync_jack(
            handle, byref(sync_in), byref(sync_out)), "get sync jack")
        return bool(sync_in.value), bool(sync_out.value)

    # -- calibration ------------------------------------------------

    def raw_calibration(self) -> str:
        """The device's raw calibration JSON (two-call pattern)."""
        handle = self._require_open()
        size = c_size_t(0)
        result = self._lib.k4a_device_get_raw_calibration(
            handle, None, byref(size))
        if result != 2:  # K4A_BUFFER_RESULT_TOO_SMALL
            raise DeviceError(
                "could not size the raw calibration "
                f"(k4a buffer result {result})")
        buf = (c_uint8 * size.value)()
        result = self._lib.k4a_device_get_raw_calibration(
            handle, buf, byref(size))
        if result != 0:
            raise DeviceError(
                "could not read the raw calibration "
                f"(k4a buffer result {result})")
        # The device's size counts its NUL terminator; strip it so
        # the string parses as JSON.
        return bytes(buf[:size.value]).rstrip(b"\x00").decode(
            "utf-8", "replace")

    def calibration(self, depth_mode: DepthMode,
                    color_resolution: ColorResolution
                    ) -> _cal.Calibration:
        """The mode-specific calibration for this device."""
        handle = self._require_open()
        raw = CCalibration()
        _check(self._lib.k4a_device_get_calibration(
            handle, int(depth_mode), int(color_resolution),
            byref(raw)), "get calibration")
        return _cal.from_ctypes(raw)

    # -- color controls ---------------------------------------------

    def color_control_capabilities(
            self, command: ColorControlCommand) -> dict[str, object]:
        """What a color control supports (auto? min/max/step/default)."""
        handle = self._require_open()
        auto = c_bool(False)
        minimum, maximum = c_int32(0), c_int32(0)
        step, default = c_int32(0), c_int32(0)
        default_mode = c_int(0)
        _check(self._lib.k4a_device_get_color_control_capabilities(
            handle, int(command), byref(auto), byref(minimum),
            byref(maximum), byref(step), byref(default),
            byref(default_mode)), "get color control capabilities")
        return {
            "supports_auto": bool(auto.value),
            "min": int(minimum.value),
            "max": int(maximum.value),
            "step": int(step.value),
            "default": int(default.value),
            "default_mode": ColorControlMode(int(default_mode.value)),
        }

    def color_control(self, command: ColorControlCommand
                      ) -> tuple[ColorControlMode, int]:
        """The current (mode, value) of a color control."""
        handle = self._require_open()
        mode, value = c_int(0), c_int32(0)
        _check(self._lib.k4a_device_get_color_control(
            handle, int(command), byref(mode), byref(value)),
            "get color control")
        return ColorControlMode(int(mode.value)), int(value.value)

    def set_color_control(self, command: ColorControlCommand,
                          mode: ColorControlMode, value: int) -> None:
        """Set a color control (mode + value)."""
        handle = self._require_open()
        _check(self._lib.k4a_device_set_color_control(
            handle, int(command), int(mode), int(value)),
            "set color control")

    # -- cameras -----------------------------------------------------

    def start_cameras(self, config: DeviceConfig | None = None) -> None:
        """Start the cameras with ``config`` (defaults if omitted)."""
        handle = self._require_open()
        cfg = (config or DeviceConfig()).to_ctypes()
        _check(self._lib.k4a_device_start_cameras(handle, byref(cfg)),
               "start cameras")
        self._cameras_started = True

    def stop_cameras(self) -> None:
        """Stop the cameras (idempotent)."""
        if self._handle is not None and self._cameras_started:
            self._lib.k4a_device_stop_cameras(self._handle)
        self._cameras_started = False

    def get_capture(self, timeout_ms: int = DEFAULT_TIMEOUT_MS
                    ) -> Capture | None:
        """Wait for one capture; None on timeout.

        Raises :class:`CaptureError` when the SDK reports failure.
        """
        handle = self._require_open()
        capture = ctypes.c_void_p()
        result = self._lib.k4a_device_get_capture(
            handle, byref(capture), int(timeout_ms))
        if result == 2:  # K4A_WAIT_RESULT_TIMEOUT
            return None
        if result != 0 or not capture.value:
            raise CaptureError(
                f"get capture failed (k4a wait result {result})")
        try:
            return Capture(
                color=self._read_image(
                    self._lib.k4a_capture_get_color_image(
                        capture.value)),
                depth=self._read_image(
                    self._lib.k4a_capture_get_depth_image(
                        capture.value)),
                ir=self._read_image(
                    self._lib.k4a_capture_get_ir_image(capture.value)),
                temperature_c=float(
                    self._lib.k4a_capture_get_temperature_c(
                        capture.value)),
            )
        finally:
            self._lib.k4a_capture_release(capture.value)

    def _read_image(self, image: int | None) -> Frame | None:
        if not image:
            return None
        try:
            lib = self._lib
            fmt = ImageFormat(lib.k4a_image_get_format(image))
            width = lib.k4a_image_get_width_pixels(image)
            height = lib.k4a_image_get_height_pixels(image)
            stride = lib.k4a_image_get_stride_bytes(image)
            size = lib.k4a_image_get_size(image)
            buf = lib.k4a_image_get_buffer(image)
            data = ctypes.string_at(buf, size) if buf else b""
            return Frame(
                fmt=fmt, width=int(width), height=int(height),
                stride=int(stride), data=data,
                device_timestamp_usec=int(
                    lib.k4a_image_get_device_timestamp_usec(image)),
                system_timestamp_nsec=int(
                    lib.k4a_image_get_system_timestamp_nsec(image)),
                exposure_usec=int(
                    lib.k4a_image_get_exposure_usec(image)),
                white_balance=int(
                    lib.k4a_image_get_white_balance(image)),
                iso_speed=int(lib.k4a_image_get_iso_speed(image)),
            )
        finally:
            self._lib.k4a_image_release(image)

    # -- IMU ----------------------------------------------------------

    def start_imu(self) -> None:
        """Start the IMU."""
        handle = self._require_open()
        _check(self._lib.k4a_device_start_imu(handle), "start IMU")
        self._imu_started = True

    def stop_imu(self) -> None:
        """Stop the IMU (idempotent)."""
        if self._handle is not None and self._imu_started:
            self._lib.k4a_device_stop_imu(self._handle)
        self._imu_started = False

    def get_imu_sample(self, timeout_ms: int = DEFAULT_TIMEOUT_MS
                       ) -> dict[str, object] | None:
        """Wait for one IMU sample; None on timeout."""
        handle = self._require_open()
        sample = ImuSample()
        result = self._lib.k4a_device_get_imu_sample(
            handle, byref(sample), int(timeout_ms))
        if result == 2:  # K4A_WAIT_RESULT_TIMEOUT
            return None
        if result != 0:
            raise CaptureError(
                f"get IMU sample failed (k4a wait result {result})")
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

    # -- transformation ------------------------------------------------

    def transformation(self, calibration: _cal.Calibration
                       ) -> Transformation:
        """A transformation handle bound to ``calibration``."""
        return Transformation(self._lib, calibration)


class Transformation:
    """The SDK's image transformation handle (context manager).

    Wraps ``k4a_transformation_create`` / ``destroy`` and the
    depth→color / depth→point-cloud transforms. Images are built
    from the caller's buffers with ``k4a_image_create_from_buffer``
    and released after the transform — the caller keeps plain
    :class:`Frame` objects.
    """

    def __init__(self, lib: K4aLibrary,
                 calibration: _cal.Calibration) -> None:
        self._lib = lib
        self.calibration = calibration
        self._handle: int | None = None
        self._live_buffers: list = []
        raw = _cal.to_ctypes(calibration)
        handle = lib.k4a_transformation_create(byref(raw))
        if not handle:
            raise DeviceError(
                "could not create a transformation handle")
        self._handle = handle

    def close(self) -> None:
        """Destroy the handle (idempotent)."""
        if self._handle is not None:
            self._lib.k4a_transformation_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> Transformation:
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    def _require_open(self) -> int:
        if self._handle is None:
            raise DeviceError("transformation is closed")
        return self._handle

    def _image_from_frame(self, frame: Frame) -> int:
        """A k4a_image_t sharing the frame's buffer (no copy).

        The destroy callback is NULL: the buffer lives in
        ``_live_buffers`` until the image is released, so the SDK
        must not free it. ctypes rejects a bare None for a
        CFUNCTYPE argument, so the NULL is spelled as a cast.
        """
        handle = ctypes.c_void_p()
        data = (c_uint8 * len(frame.data)).from_buffer_copy(frame.data)
        self._live_buffers.append(data)
        result = self._lib.k4a_image_create_from_buffer(
            int(frame.fmt), frame.width, frame.height, frame.stride,
            data, len(frame.data),
            ctypes.cast(None, _DestroyCallback), None,
            byref(handle))
        if result != 0 or not handle.value:
            raise DeviceError(
                "could not build a k4a image from the frame "
                f"(k4a result {result})")
        return handle.value

    def _new_image(self, fmt: ImageFormat, width: int, height: int,
                   stride: int) -> int:
        handle = ctypes.c_void_p()
        result = self._lib.k4a_image_create(
            int(fmt), width, height, stride, byref(handle))
        if result != 0 or not handle.value:
            raise DeviceError(
                "could not allocate a k4a image "
                f"(k4a result {result})")
        return handle.value

    def depth_to_point_cloud(
            self, depth: Frame,
            camera: CalibrationType = CalibrationType.DEPTH
            ) -> list[tuple[float, float, float]]:
        """Depth frame → XYZ point cloud (mm), via the real SDK.

        The SDK writes a CUSTOM image of 3 int16 values per pixel
        (6 bytes, X/Y/Z in millimetres — k4a.h documents this
        explicitly); the triplets are read back row by row.
        """
        if depth.fmt is not ImageFormat.DEPTH16:
            raise KinectError(
                "depth_to_point_cloud needs a DEPTH16 frame, got "
                f"{depth.fmt.name}", kind="wrong_format")
        self._live_buffers = []
        depth_handle = self._image_from_frame(depth)
        xyz_handle = self._new_image(
            ImageFormat.CUSTOM, depth.width, depth.height,
            depth.width * 6)
        try:
            _check(self._lib.k4a_transformation_depth_image_to_point_cloud(
                self._require_open(), depth_handle, int(camera),
                xyz_handle), "depth image to point cloud")
            lib = self._lib
            size = lib.k4a_image_get_size(xyz_handle)
            buf = lib.k4a_image_get_buffer(xyz_handle)
            raw = ctypes.string_at(buf, size) if buf else b""
            stride = lib.k4a_image_get_stride_bytes(xyz_handle)
            points: list[tuple[float, float, float]] = []
            for y in range(depth.height):
                base = y * stride
                for x in range(depth.width):
                    off = base + x * 6
                    px, py, pz = struct.unpack_from(
                        "<hhh", raw, off)
                    points.append((float(px), float(py), float(pz)))
            return points
        finally:
            self._lib.k4a_image_release(depth_handle)
            self._lib.k4a_image_release(xyz_handle)
            self._live_buffers = []

    def depth_to_color(self, depth: Frame, color_width: int,
                       color_height: int) -> Frame:
        """Depth frame reprojected into the color camera's geometry."""
        if depth.fmt is not ImageFormat.DEPTH16:
            raise KinectError(
                "depth_to_color needs a DEPTH16 frame, got "
                f"{depth.fmt.name}", kind="wrong_format")
        self._live_buffers = []
        depth_handle = self._image_from_frame(depth)
        out_handle = self._new_image(
            ImageFormat.DEPTH16, color_width, color_height,
            color_width * 2)
        try:
            _check(self._lib.k4a_transformation_depth_image_to_color_camera(
                self._require_open(), depth_handle, out_handle),
                "depth image to color camera")
            lib = self._lib
            size = lib.k4a_image_get_size(out_handle)
            buf = lib.k4a_image_get_buffer(out_handle)
            data = ctypes.string_at(buf, size) if buf else b""
            return Frame(
                fmt=ImageFormat.DEPTH16, width=color_width,
                height=color_height, stride=color_width * 2,
                data=data)
        finally:
            self._lib.k4a_image_release(depth_handle)
            self._lib.k4a_image_release(out_handle)
            self._live_buffers = []
