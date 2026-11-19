"""Calibration — parsing the SDK's raw calibration JSON and the
2D/3D geometry that goes with it.

The raw calibration JSON schema is the one the real SDK parses
(``src/calibration/calibration.c``, v1.4.1): a top-level
``CalibrationInformation`` object holding a ``Cameras`` array;
each camera carries ``Location``
(``"CALIBRATION_CameraLocationD0"`` / ``"CALIBRATION_CameraLocationPV0"``),
``Intrinsics`` (``ModelParameterCount``, ``ModelType``,
``ModelParameters``), ``Rt`` (``Rotation`` [9] + ``Translation``
[3], the translation in *metres* — the SDK multiplies by 1000
into millimetres), ``SensorHeight``, ``SensorWidth`` and
``MetricRadius`` (a radius at or below 0.0001 means the estimate
failed and the SDK substitutes 1.7, ~120 degrees of field of
view). Model type strings are
``CALIBRATION_LensDistortionModelBrownConrady`` (retail devices)
and its Theta / Polynomial3K / Rational6KT siblings.

The raw intrinsics are *unitized and 0-cornered* — cx/fx are
fractions of the image size — and the mode-specific calibration
(``src/transformation/mode_specific_calibration.c``) converts
them to pixelized, 0-centered values for the requested depth
mode and color resolution: cx/fx scale by the binned
resolution, the crop offset is subtracted, and the principal
point shifts half a pixel. A 16:9 color calibration is first
converted to 4:3, still unitized, then pixelized.

The projection math is the SDK's own
(``src/transformation/intrinsic_transformation.c``): the
Brown-Conrady rational model for 3D→2D, and the iterative
Newton unprojection (20 passes, Jacobian inverse) for 2D→3D.
The parity runner checks this module's numbers against the
*real* libk4a on this box — same calibration, same points,
through ``k4a_calibration_2d_to_3d`` / ``k4a_calibration_3d_to_2d``
— so the port's geometry is measured against the SDK binary,
not against itself.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .enums import CalibrationModelType, CalibrationType, ColorResolution, DepthMode
from .errors import CalibrationError

__all__ = [
    "MODEL_TYPE_STRINGS",
    "CameraCalibration",
    "Calibration",
    "Extrinsics",
    "Intrinsics",
    "from_ctypes",
    "parse_raw_calibration",
    "point_cloud",
    "to_ctypes",
]

#: Raw-calibration JSON model-type strings, from calibration.c.
MODEL_TYPE_STRINGS: dict[str, CalibrationModelType] = {
    "CALIBRATION_LensDistortionModelTheta":
        CalibrationModelType.THETA,
    "CALIBRATION_LensDistortionModelPolynomial3K":
        CalibrationModelType.POLYNOMIAL_3K,
    "CALIBRATION_LensDistortionModelRational6KT":
        CalibrationModelType.RATIONAL_6KT,
    "CALIBRATION_LensDistortionModelBrownConrady":
        CalibrationModelType.BROWN_CONRADY,
}

#: A metric radius at or below this means the estimate failed.
_METRIC_RADIUS_FLOOR = 0.0001
#: The SDK's substitute radius (~120 degrees of field of view).
_METRIC_RADIUS_DEFAULT = 1.7

#: Intrinsics parameter order: cx, cy, fx, fy, k1..k6, codx, cody,
#: p2, p1, metric_radius (k4atypes.h, _param struct).
_PARAM_NAMES = ("cx", "cy", "fx", "fy", "k1", "k2", "k3", "k4",
                "k5", "k6", "codx", "cody", "p2", "p1",
                "metric_radius")


@dataclass(frozen=True)
class Intrinsics:
    """One camera's intrinsic model (k4a_calibration_intrinsics_t)."""

    model: CalibrationModelType = CalibrationModelType.BROWN_CONRADY
    cx: float = 0.0
    cy: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    codx: float = 0.0
    cody: float = 0.0
    p2: float = 0.0
    p1: float = 0.0
    metric_radius: float = 0.0

    @property
    def parameter_count(self) -> int:
        """Parameters the SDK counts as valid (14 for Brown-Conrady)."""
        return 14

    def as_list(self) -> list[float]:
        """The 15 parameters in header order."""
        return [getattr(self, name) for name in _PARAM_NAMES]


@dataclass(frozen=True)
class Extrinsics:
    """Rotation (row-major 3x3) + translation in millimetres."""

    rotation: tuple[float, ...] = (1.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0,
                                   0.0, 0.0, 1.0)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def apply(self, point: tuple[float, float, float]
              ) -> tuple[float, float, float]:
        """Rotate + translate a point (row-major R, mm)."""
        r = self.rotation
        x, y, z = point
        return (r[0] * x + r[1] * y + r[2] * z + self.translation[0],
                r[3] * x + r[4] * y + r[5] * z + self.translation[1],
                r[6] * x + r[7] * y + r[8] * z + self.translation[2])

    def compose(self, other: Extrinsics) -> Extrinsics:
        """This transform after ``other`` (this ∘ other)."""
        r, s = self.rotation, other.rotation
        rotation = tuple(
            sum(r[3 * i + k] * s[3 * k + j] for k in range(3))
            for i in range(3) for j in range(3))
        moved = self.apply(other.translation)
        return Extrinsics(rotation=rotation, translation=moved)

    def invert(self) -> Extrinsics:
        """Inverse transform (rotation transpose, -R^T t)."""
        r = self.rotation
        rt = (r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8])
        t = self.translation
        inv_t = tuple(
            -(rt[3 * i] * t[0] + rt[3 * i + 1] * t[1]
              + rt[3 * i + 2] * t[2]) for i in range(3))
        return Extrinsics(rotation=rt, translation=inv_t)


@dataclass(frozen=True)
class CameraCalibration:
    """One camera's calibration (k4a_calibration_camera_t)."""

    extrinsics: Extrinsics = field(default_factory=Extrinsics)
    intrinsics: Intrinsics = field(default_factory=Intrinsics)
    resolution_width: int = 0
    resolution_height: int = 0
    metric_radius: float = 0.0


@dataclass(frozen=True)
class Calibration:
    """Device calibration (k4a_calibration_t, depth + color).

    Each sensor's ``extrinsics`` is its pose: the world→sensor
    transform, with the depth sensor as the world. The pairwise
    grid is derived, exactly as the SDK derives it
    (``transformation_get_extrinsic_transformation``):
    source→target is pose[target] ∘ pose[source]⁻¹.
    """

    depth: CameraCalibration = field(default_factory=CameraCalibration)
    color: CameraCalibration = field(default_factory=CameraCalibration)
    gyro: Extrinsics = field(default_factory=Extrinsics)
    accel: Extrinsics = field(default_factory=Extrinsics)
    depth_mode: int = 0
    color_resolution: int = 0

    def camera(self, kind: CalibrationType) -> CameraCalibration:
        """The calibration for a DEPTH or COLOR camera."""
        if kind is CalibrationType.DEPTH:
            return self.depth
        if kind is CalibrationType.COLOR:
            return self.color
        raise CalibrationError(
            f"no camera calibration for type {kind!r}")

    def pose(self, kind: CalibrationType) -> Extrinsics:
        """The world→sensor pose for a sensor type.

        Depth and color poses come from their camera calibrations;
        gyro / accel poses are the depth→IMU extrinsics the raw
        calibration carries in its InertialSensors section.
        """
        if kind is CalibrationType.DEPTH:
            return self.depth.extrinsics
        if kind is CalibrationType.COLOR:
            return self.color.extrinsics
        if kind is CalibrationType.GYRO:
            return self.gyro
        if kind is CalibrationType.ACCEL:
            return self.accel
        raise CalibrationError(
            f"no pose for calibration type {kind!r}")

    def extrinsics(self, source: CalibrationType,
                   target: CalibrationType) -> Extrinsics:
        """The source→target extrinsic transform.

        pose[target] ∘ pose[source]⁻¹ — the SDK's
        ``extrinsics[source][target]`` grid, built the same way
        (``transformation_get_extrinsic_transformation``).
        """
        return self.pose(target).compose(self.pose(source).invert())

    # -- geometry (SDK transformation.c / intrinsic_transformation.c) --

    def project_3d_to_2d(
            self, point_mm: tuple[float, float, float],
            kind: CalibrationType) -> tuple[tuple[float, float], bool]:
        """Project a 3D point (mm, camera space) to pixels.

        Returns (pixel, valid): valid is False when the point is
        behind the camera or its radius exceeds the metric radius.
        """
        x, y, z = point_mm
        if z <= 0.0:
            return (0.0, 0.0), False
        return _project(self.camera(kind), (x / z, y / z))

    def unproject_2d_to_3d(
            self, pixel: tuple[float, float], depth_mm: float,
            kind: CalibrationType) -> tuple[tuple[float, float, float],
                                           bool]:
        """Unproject a pixel + depth (mm) to a 3D point (mm).

        A zero depth returns the origin, valid, exactly as the SDK
        does; a failed model returns (0, 0, 0) invalid.
        """
        if depth_mm == 0.0:
            return (0.0, 0.0, 0.0), True
        xy, valid = _unproject(self.camera(kind), pixel)
        if not valid:
            return (0.0, 0.0, 0.0), False
        return (xy[0] * depth_mm, xy[1] * depth_mm, depth_mm), True

    def transform_3d_to_3d(
            self, point_mm: tuple[float, float, float],
            source: CalibrationType,
            target: CalibrationType) -> tuple[float, float, float]:
        """Move a 3D point from one camera's space to another's."""
        return self.extrinsics(source, target).apply(point_mm)

    def transform_2d_to_3d(
            self, pixel: tuple[float, float], depth_mm: float,
            source: CalibrationType,
            target: CalibrationType) -> tuple[tuple[float, float, float],
                                             bool]:
        """Unproject in ``source`` space, then move to ``target``."""
        point, valid = self.unproject_2d_to_3d(pixel, depth_mm, source)
        if not valid:
            return (0.0, 0.0, 0.0), False
        return self.transform_3d_to_3d(point, source, target), True

    def transform_3d_to_2d(
            self, point_mm: tuple[float, float, float],
            source: CalibrationType,
            target: CalibrationType) -> tuple[tuple[float, float], bool]:
        """Move a 3D point to ``target`` space, then project it."""
        moved = self.transform_3d_to_3d(point_mm, source, target)
        return self.project_3d_to_2d(moved, target)


def _project(camera: CameraCalibration, xy: tuple[float, float]
             ) -> tuple[tuple[float, float], bool]:
    """The SDK's project_internal (Brown-Conrady / Rational 6KT)."""
    intr = camera.intrinsics
    if intr.model not in (CalibrationModelType.RATIONAL_6KT,
                          CalibrationModelType.BROWN_CONRADY):
        raise CalibrationError(
            f"unsupported calibration model {intr.model!r}")
    if intr.fx <= 0.0 or intr.fy <= 0.0:
        raise CalibrationError(
            f"non-positive focal length fx={intr.fx} fy={intr.fy}")
    xp, yp = xy[0] - intr.codx, xy[1] - intr.cody
    xp2, yp2, xyp = xp * xp, yp * yp, xp * yp
    rs = xp2 + yp2
    if rs > camera.metric_radius * camera.metric_radius:
        return (0.0, 0.0), False
    rss, rsc = rs * rs, rs * rs * rs
    a = 1.0 + intr.k1 * rs + intr.k2 * rss + intr.k3 * rsc
    b = 1.0 + intr.k4 * rs + intr.k5 * rss + intr.k6 * rsc
    bi = 1.0 / b if b != 0.0 else 1.0
    d = a * bi
    xp_d, yp_d = xp * d, yp * d
    rs_2xp2, rs_2yp2 = rs + 2.0 * xp2, rs + 2.0 * yp2
    if intr.model is CalibrationModelType.RATIONAL_6KT:
        xp_d += rs_2xp2 * intr.p2 + xyp * intr.p1
        yp_d += rs_2yp2 * intr.p1 + xyp * intr.p2
    else:  # Brown-Conrady: the xyp terms carry a factor of 2.
        xp_d += rs_2xp2 * intr.p2 + 2.0 * xyp * intr.p1
        yp_d += rs_2yp2 * intr.p1 + 2.0 * xyp * intr.p2
    u = (xp_d + intr.codx) * intr.fx + intr.cx
    v = (yp_d + intr.cody) * intr.fy + intr.cy
    return (u, v), True


def _project_jacobian(camera: CameraCalibration,
                      xy: tuple[float, float]
                      ) -> tuple[tuple[float, float], bool,
                                 tuple[float, float, float, float]]:
    """Project + the 2x2 Jacobian (SDK project_internal, J branch)."""
    intr = camera.intrinsics
    uv, valid = _project(camera, xy)
    if not valid:
        return uv, False, (0.0, 0.0, 0.0, 0.0)
    xp, yp = xy[0] - intr.codx, xy[1] - intr.cody
    xp2, yp2 = xp * xp, yp * yp
    rs = xp2 + yp2
    rss, rsc = rs * rs, rs * rs * rs
    a = 1.0 + intr.k1 * rs + intr.k2 * rss + intr.k3 * rsc
    b = 1.0 + intr.k4 * rs + intr.k5 * rss + intr.k6 * rsc
    bi = 1.0 / b if b != 0.0 else 1.0
    d = a * bi
    dudrs = intr.k1 + 2.0 * intr.k2 * rs + 3.0 * intr.k3 * rss
    dvdrs = intr.k4 + 2.0 * intr.k5 * rs + 3.0 * intr.k6 * rss
    bis = bi * bi
    dddrs = (dudrs * b - a * dvdrs) * bis
    dddrs_2 = dddrs * 2.0
    xp_dddrs_2 = xp * dddrs_2
    yp_xp_dddrs_2 = yp * xp_dddrs_2
    if intr.model is CalibrationModelType.RATIONAL_6KT:
        j = (intr.fx * (d + xp * xp_dddrs_2 + 6.0 * xp * intr.p2
                        + yp * intr.p1),
             intr.fx * (yp_xp_dddrs_2 + 2.0 * yp * intr.p2
                        + xp * intr.p1),
             intr.fy * (yp_xp_dddrs_2 + 2.0 * xp * intr.p1
                        + yp * intr.p2),
             intr.fy * (d + yp * yp * dddrs_2 + 6.0 * yp * intr.p1
                        + xp * intr.p2))
    else:
        j = (intr.fx * (d + xp * xp_dddrs_2 + 6.0 * xp * intr.p2
                        + 2.0 * yp * intr.p1),
             intr.fx * (yp_xp_dddrs_2 + 2.0 * yp * intr.p2
                        + 2.0 * xp * intr.p1),
             intr.fy * (yp_xp_dddrs_2 + 2.0 * xp * intr.p1
                        + 2.0 * yp * intr.p2),
             intr.fy * (d + yp * yp * dddrs_2 + 6.0 * yp * intr.p1
                        + 2.0 * xp * intr.p2))
    return uv, True, j


def _unproject(camera: CameraCalibration, pixel: tuple[float, float]
               ) -> tuple[tuple[float, float], bool]:
    """The SDK's unproject_internal: analytic seed + 20 Newton passes."""
    intr = camera.intrinsics
    if intr.model not in (CalibrationModelType.RATIONAL_6KT,
                          CalibrationModelType.BROWN_CONRADY):
        raise CalibrationError(
            f"unsupported calibration model {intr.model!r}")
    if intr.fx <= 0.0 or intr.fy <= 0.0:
        raise CalibrationError(
            f"non-positive focal length fx={intr.fx} fy={intr.fy}")
    # Analytic seed: invert the radial model, then correct the
    # tangential terms approximately (SDK unproject_internal).
    xp_d = (pixel[0] - intr.cx) / intr.fx - intr.codx
    yp_d = (pixel[1] - intr.cy) / intr.fy - intr.cody
    rs = xp_d * xp_d + yp_d * yp_d
    rss, rsc = rs * rs, rs * rs * rs
    a = 1.0 + intr.k1 * rs + intr.k2 * rss + intr.k3 * rsc
    b = 1.0 + intr.k4 * rs + intr.k5 * rss + intr.k6 * rsc
    ai = 1.0 / a if a != 0.0 else 1.0
    di = ai * b
    xy = [xp_d * di, yp_d * di]
    two_xy = 2.0 * xy[0] * xy[1]
    xx, yy = xy[0] * xy[0], xy[1] * xy[1]
    xy[0] -= (yy + 3.0 * xx) * intr.p2 + two_xy * intr.p1
    xy[1] -= (xx + 3.0 * yy) * intr.p1 + two_xy * intr.p2
    xy[0] += intr.codx
    xy[1] += intr.cody
    # Newton refinement (SDK iterative_unproject, 20 passes).
    best_xy = [0.0, 0.0]
    best_err = math.inf
    valid = True
    for _ in range(20):
        p, valid, jac = _project_jacobian(camera, (xy[0], xy[1]))
        if not valid:
            return (0.0, 0.0), False
        err_x, err_y = pixel[0] - p[0], pixel[1] - p[1]
        err = err_x * err_x + err_y * err_y
        if err >= best_err:
            xy = best_xy
            break
        best_err = err
        best_xy = list(xy)
        if best_err < 1e-22:
            break
        det = jac[0] * jac[3] - jac[1] * jac[2]
        if det == 0.0:
            break
        inv_det = 1.0 / det
        jinv = (inv_det * jac[3], -inv_det * jac[1],
                -inv_det * jac[2], inv_det * jac[0])
        xy[0] += jinv[0] * err_x + jinv[1] * err_y
        xy[1] += jinv[2] * err_x + jinv[3] * err_y
    return (xy[0], xy[1]), valid


#: Raw-calibration camera locations the SDK looks up
#: (calibration.c ``calibration_create_from_raw``); friendly
#: aliases are accepted as well.
_DEPTH_LOCATIONS = ("CALIBRATION_CameraLocationD0", "Depth")
_COLOR_LOCATIONS = ("CALIBRATION_CameraLocationPV0", "Color")
_GYRO_TYPES = ("CALIBRATION_InertialSensorType_Gyro", "Gyro")
_ACCEL_TYPES = ("CALIBRATION_InertialSensorType_Accelerometer",
                "Accel", "Accelerometer")

#: Mode-specific conversion tables from the SDK's
#: mode_specific_calibration.c. Each row is
#: (binned resolution, crop offset, output resolution).
_DEPTH_MODE_INFO: dict[DepthMode, tuple[tuple[int, int],
                                        tuple[int, int],
                                        tuple[int, int]]] = {
    DepthMode.NFOV_2X2BINNED: ((512, 512), (96, 90), (320, 288)),
    DepthMode.NFOV_UNBINNED: ((1024, 1024), (192, 180), (640, 576)),
    DepthMode.WFOV_2X2BINNED: ((512, 512), (0, 0), (512, 512)),
    DepthMode.WFOV_UNBINNED: ((1024, 1024), (0, 0), (1024, 1024)),
    DepthMode.PASSIVE_IR: ((1024, 1024), (0, 0), (1024, 1024)),
}
_COLOR_RESOLUTION_INFO: dict[ColorResolution,
                             tuple[tuple[int, int], tuple[int, int],
                                   tuple[int, int]]] = {
    ColorResolution.RES_720P: ((1280, 960), (0, 120), (1280, 720)),
    ColorResolution.RES_1080P: ((1920, 1440), (0, 180), (1920, 1080)),
    ColorResolution.RES_1440P: ((2560, 1920), (0, 240), (2560, 1440)),
    ColorResolution.RES_1536P: ((2048, 1536), (0, 0), (2048, 1536)),
    ColorResolution.RES_2160P: ((3840, 2880), (0, 360), (3840, 2160)),
    ColorResolution.RES_3072P: ((4096, 3072), (0, 0), (4096, 3072)),
}
#: Legacy 16:9 color calibrations are first converted to 4:3,
#: unitized, with this table (mode_specific_calibration.c).
_LEGACY_COLOR_INFO = ((4096, 2304), (0, -384), (4096, 3072))


def _pixelized(camera: CameraCalibration,
               binned: tuple[int, int], crop: tuple[int, int],
               output: tuple[int, int]) -> CameraCalibration:
    """The SDK's mode-specific conversion, pixelized output.

    The raw calibration stores unitized, 0-cornered intrinsics
    (principal point and focal length divided by the image size);
    the mode-specific calibration is pixelized and 0-centered
    (OpenCV convention): cx/fx scale by the binned resolution,
    the crop offset is subtracted, and the principal point shifts
    half a pixel (``transformation_get_mode_specific_camera_calibration``,
    pixelized_zero_centered_output=true).
    """
    intr = camera.intrinsics
    cx = intr.cx * binned[0] - crop[0] - 0.5
    cy = intr.cy * binned[1] - crop[1] - 0.5
    fx = intr.fx * binned[0]
    fy = intr.fy * binned[1]
    return CameraCalibration(
        extrinsics=camera.extrinsics,
        intrinsics=Intrinsics(
            model=intr.model, cx=cx, cy=cy, fx=fx, fy=fy,
            k1=intr.k1, k2=intr.k2, k3=intr.k3, k4=intr.k4,
            k5=intr.k5, k6=intr.k6, codx=intr.codx, cody=intr.cody,
            p2=intr.p2, p1=intr.p1),
        resolution_width=output[0], resolution_height=output[1],
        metric_radius=camera.metric_radius)


def _unitized(camera: CameraCalibration,
              binned: tuple[int, int], crop: tuple[int, int],
              output: tuple[int, int]) -> CameraCalibration:
    """The same conversion, kept unitized and 0-cornered.

    Used for the legacy 16:9 → 4:3 color conversion, which the
    SDK performs before the pixelizing step
    (pixelized_zero_centered_output=false).
    """
    intr = camera.intrinsics
    cx = (intr.cx * binned[0] - crop[0]) / output[0]
    cy = (intr.cy * binned[1] - crop[1]) / output[1]
    fx = intr.fx * binned[0] / output[0]
    fy = intr.fy * binned[1] / output[1]
    return CameraCalibration(
        extrinsics=camera.extrinsics,
        intrinsics=Intrinsics(
            model=intr.model, cx=cx, cy=cy, fx=fx, fy=fy,
            k1=intr.k1, k2=intr.k2, k3=intr.k3, k4=intr.k4,
            k5=intr.k5, k6=intr.k6, codx=intr.codx, cody=intr.cody,
            p2=intr.p2, p1=intr.p1),
        resolution_width=output[0], resolution_height=output[1],
        metric_radius=camera.metric_radius)


def _mode_specific_depth(raw: CameraCalibration,
                         depth_mode: DepthMode) -> CameraCalibration:
    """Depth camera calibration for one depth mode."""
    if depth_mode is DepthMode.OFF:
        raise CalibrationError(
            "depth mode OFF has no depth calibration")
    info = _DEPTH_MODE_INFO.get(depth_mode)
    if info is None:
        raise CalibrationError(
            f"unknown depth mode {depth_mode!r}")
    if raw.resolution_width != 1024 \
            or raw.resolution_height != 1024:
        raise CalibrationError(
            "raw depth calibration must be 1024x1024, got "
            f"{raw.resolution_width}x{raw.resolution_height}")
    binned, crop, output = info
    return _pixelized(raw, binned, crop, output)


def _mode_specific_color(raw: CameraCalibration,
                         color_resolution: ColorResolution
                         ) -> CameraCalibration:
    """Color camera calibration for one color resolution."""
    if color_resolution is ColorResolution.OFF:
        raise CalibrationError(
            "color resolution OFF has no color calibration")
    info = _COLOR_RESOLUTION_INFO.get(color_resolution)
    if info is None:
        raise CalibrationError(
            f"unknown color resolution {color_resolution!r}")
    working = raw
    if raw.resolution_width * 9 // 16 == raw.resolution_height:
        binned, crop, output = _LEGACY_COLOR_INFO
        working = _unitized(raw, binned, crop, output)
    elif raw.resolution_width * 3 // 4 != raw.resolution_height:
        raise CalibrationError(
            "raw color calibration must be 16:9 or 4:3, got "
            f"{raw.resolution_width}x{raw.resolution_height}")
    binned, crop, output = info
    return _pixelized(working, binned, crop, output)


def parse_raw_calibration(raw: str | bytes, *,
                          depth_mode: DepthMode
                          = DepthMode.NFOV_UNBINNED,
                          color_resolution: ColorResolution
                          = ColorResolution.RES_720P) -> Calibration:
    """Parse the SDK's raw-calibration JSON into a Calibration.

    Mirrors ``calibration_create_from_raw`` (calibration.c) plus
    the mode-specific conversion
    (mode_specific_calibration.c): the JSON stores unitized,
    0-cornered intrinsics, and the returned cameras hold the
    pixelized, 0-centered values for the requested depth mode
    and color resolution — the same values
    ``k4a_calibration_get_from_raw`` returns for that config.
    The IMU sensors' richer model lives in
    ``k4a_calibration_imu_t``, which the public
    ``k4a_calibration_t`` does not carry; only their poses are
    kept here.
    """
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise CalibrationError(
            f"raw calibration is not JSON: {exc}") from exc
    info = data.get("CalibrationInformation") if isinstance(data, dict) \
        else None
    if not isinstance(info, dict):
        raise CalibrationError(
            "raw calibration has no CalibrationInformation object")
    cameras = info.get("Cameras")
    if not isinstance(cameras, list):
        raise CalibrationError(
            "raw calibration has no Cameras array")
    by_location: dict[str, dict] = {}
    for camera in cameras:
        if isinstance(camera, dict) and isinstance(
                camera.get("Location"), str):
            by_location[str(camera["Location"])] = camera
    depth_raw = _parse_camera(
        _find_camera(by_location, _DEPTH_LOCATIONS), "Depth")
    color_raw = _parse_camera(
        _find_camera(by_location, _COLOR_LOCATIONS), "Color")
    depth = _mode_specific_depth(depth_raw, depth_mode)
    color = _mode_specific_color(color_raw, color_resolution)
    sensors = info.get("InertialSensors")
    gyro = _parse_imu_pose(sensors, _GYRO_TYPES)
    accel = _parse_imu_pose(sensors, _ACCEL_TYPES)
    return Calibration(depth=depth, color=color,
                       gyro=gyro, accel=accel,
                       depth_mode=int(depth_mode),
                       color_resolution=int(color_resolution))


def _find_camera(by_location: dict[str, dict],
                 names: tuple[str, ...]) -> dict | None:
    for name in names:
        if name in by_location:
            return by_location[name]
    return None


def _parse_imu_pose(sensors: object,
                    sensor_types: tuple[str, ...]) -> Extrinsics:
    """The depth→IMU extrinsics for a Gyro / Accel sensor.

    The raw calibration carries them in ``CalibrationInformation``
    → ``InertialSensors`` (each with ``SensorType`` and ``Rt``);
    a missing entry leaves the identity pose — the SDK's own
    ``calibration_create_from_raw`` fails the whole parse when an
    IMU entry is absent, but a fixture that only exercises the
    cameras stays usable here, and the gap is named, not hidden.
    """
    if not isinstance(sensors, list):
        return Extrinsics()
    for sensor in sensors:
        if not isinstance(sensor, dict):
            continue
        if sensor.get("SensorType") not in sensor_types:
            continue
        rt = sensor.get("Rt")
        if not isinstance(rt, dict):
            return Extrinsics()
        rotation = rt.get("Rotation")
        translation = rt.get("Translation")
        if not isinstance(rotation, list) or len(rotation) != 9 \
                or not isinstance(translation, list) \
                or len(translation) != 3:
            return Extrinsics()
        return Extrinsics(
            rotation=tuple(float(v) for v in rotation),
            translation=tuple(float(v) * 1000.0
                              for v in translation))
    return Extrinsics()


def _parse_camera(camera: dict | None,
                  location: str) -> CameraCalibration:
    if camera is None:
        raise CalibrationError(
            f"raw calibration has no {location} camera")
    intrinsics = camera.get("Intrinsics")
    if not isinstance(intrinsics, dict):
        raise CalibrationError(
            f"{location} camera has no Intrinsics object")
    count = intrinsics.get("ModelParameterCount")
    if not isinstance(count, int):
        raise CalibrationError(
            f"{location} intrinsics lack ModelParameterCount")
    model_name = intrinsics.get("ModelType")
    if not isinstance(model_name, str) \
            or model_name not in MODEL_TYPE_STRINGS:
        raise CalibrationError(
            f"{location} intrinsics name an unknown ModelType "
            f"{model_name!r}")
    params = intrinsics.get("ModelParameters")
    if not isinstance(params, list) or len(params) < count:
        raise CalibrationError(
            f"{location} intrinsics lack {count} ModelParameters")
    if count > 15:
        raise CalibrationError(
            f"{location} intrinsics name {count} parameters, "
            "at most 15 fit")
    values = [float(v) for v in params[:15]]
    values += [0.0] * (15 - len(values))
    intr = Intrinsics(
        model=MODEL_TYPE_STRINGS[model_name],
        **{name: values[i] for i, name in enumerate(_PARAM_NAMES)})
    rt = camera.get("Rt")
    if not isinstance(rt, dict):
        raise CalibrationError(f"{location} camera has no Rt object")
    rotation = rt.get("Rotation")
    translation = rt.get("Translation")
    if not isinstance(rotation, list) or len(rotation) != 9 \
            or not isinstance(translation, list) \
            or len(translation) != 3:
        raise CalibrationError(
            f"{location} camera Rt lacks a 9-rotation / "
            "3-translation")
    # The SDK stores translations in metres; the struct holds mm.
    extr = Extrinsics(
        rotation=tuple(float(v) for v in rotation),
        translation=tuple(float(v) * 1000.0 for v in translation))
    width = camera.get("SensorWidth")
    height = camera.get("SensorHeight")
    radius = camera.get("MetricRadius")
    if not isinstance(width, int) or not isinstance(height, int) \
            or not isinstance(radius, (int, float)):
        raise CalibrationError(
            f"{location} camera lacks SensorWidth / SensorHeight / "
            "MetricRadius")
    if float(radius) <= _METRIC_RADIUS_FLOOR:
        radius = _METRIC_RADIUS_DEFAULT
    return CameraCalibration(
        extrinsics=extr, intrinsics=intr,
        resolution_width=int(width), resolution_height=int(height),
        metric_radius=float(radius))


def from_ctypes(calibration) -> Calibration:
    """Build a Calibration from a ctypes k4a_calibration_t."""
    from .bindings import Calibration as CCalibration

    if not isinstance(calibration, CCalibration):
        raise CalibrationError(
            "from_ctypes needs a ctypes k4a_calibration_t")

    def camera(cam) -> CameraCalibration:
        params = cam.intrinsics.parameters.param
        intr = Intrinsics(
            model=CalibrationModelType(int(cam.intrinsics.type)),
            **{name: float(getattr(params, name))
               for name in _PARAM_NAMES})
        extr = Extrinsics(
            rotation=tuple(float(v)
                           for v in cam.extrinsics.rotation),
            translation=tuple(float(v)
                              for v in cam.extrinsics.translation))
        return CameraCalibration(
            extrinsics=extr, intrinsics=intr,
            resolution_width=int(cam.resolution_width),
            resolution_height=int(cam.resolution_height),
            metric_radius=float(cam.metric_radius))

    depth_cam = camera(calibration.depth_camera_calibration)
    color_cam = camera(calibration.color_camera_calibration)
    grid = calibration.extrinsics
    depth_pose = depth_cam.extrinsics

    def pose_from_grid(index: int) -> Extrinsics:
        # extrinsics[DEPTH][i] is pose[i] ∘ pose[DEPTH]⁻¹, so
        # pose[i] is that grid entry composed with pose[DEPTH].
        entry = grid[0][index]
        pair = Extrinsics(
            rotation=tuple(float(v) for v in entry.rotation),
            translation=tuple(float(v) for v in entry.translation))
        return pair.compose(depth_pose)

    return Calibration(
        depth=depth_cam,
        color=color_cam,
        gyro=pose_from_grid(2),
        accel=pose_from_grid(3),
        depth_mode=int(calibration.depth_mode),
        color_resolution=int(calibration.color_resolution))


def to_ctypes(calibration: Calibration):
    """Build a ctypes k4a_calibration_t from a Calibration."""
    import ctypes

    from .bindings import Calibration as CCalibration
    from .bindings import CalibrationCamera as CCamera
    from .bindings import CalibrationExtrinsics as CExtrinsics
    from .bindings import (
        CalibrationIntrinsicParameters as CParams,
    )
    from .bindings import CalibrationIntrinsics as CIntrinsics

    def extrinsics_ctypes(pair: Extrinsics) -> CExtrinsics:
        return CExtrinsics(
            rotation=(ctypes.c_float * 9)(*pair.rotation),
            translation=(ctypes.c_float * 3)(*pair.translation))

    def camera(cam: CameraCalibration) -> CCamera:
        params = CParams()
        for i, value in enumerate(cam.intrinsics.as_list()):
            params.v[i] = float(value)
        return CCamera(
            extrinsics=extrinsics_ctypes(cam.extrinsics),
            intrinsics=CIntrinsics(
                type=int(cam.intrinsics.model),
                parameter_count=cam.intrinsics.parameter_count,
                parameters=params),
            resolution_width=int(cam.resolution_width),
            resolution_height=int(cam.resolution_height),
            metric_radius=float(cam.metric_radius))

    grid = ((CExtrinsics * 4) * 4)()
    for source in range(4):
        for target in range(4):
            grid[source][target] = extrinsics_ctypes(
                calibration.extrinsics(
                    CalibrationType(source), CalibrationType(target)))
    return CCalibration(
        depth_camera_calibration=camera(calibration.depth),
        color_camera_calibration=camera(calibration.color),
        extrinsics=grid,
        depth_mode=int(calibration.depth_mode),
        color_resolution=int(calibration.color_resolution))


def point_cloud(calibration: Calibration,
                depth_mm: list[list[float]],
                kind: CalibrationType = CalibrationType.DEPTH
                ) -> list[tuple[float, float, float]]:
    """Depth map → 3D points (mm) in the camera's own space.

    ``depth_mm`` is rows of depths in millimetres; a zero depth
    unprojects to the origin, exactly as the SDK's unproject does.
    Pixels are (x, y) = (column, row), origin top-left.
    """
    points: list[tuple[float, float, float]] = []
    for y, row in enumerate(depth_mm):
        for x, depth in enumerate(row):
            point, _valid = calibration.unproject_2d_to_3d(
                (float(x), float(y)), float(depth), kind)
            points.append(point)
    return points
