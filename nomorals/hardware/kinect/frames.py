"""Frames — decoding the image buffers a capture hands back.

A ``k4a_image_t`` is a flat buffer plus (format, width, height,
stride, timestamps). This module turns that raw description into
typed Python values without ever touching the device: depth16 and
ir16 buffers are little-endian uint16 arrays, color MJPG is a JPEG
byte string, BGRA32 / NV12 / YUY2 are raw planes. Decoding is
pure Python + stdlib (``struct``/``array``) with a numpy fast
path when numpy is importable — the same code runs on a box
without numpy.
"""

from __future__ import annotations

import array
import sys
from dataclasses import dataclass

from .enums import ImageFormat, bytes_per_pixel
from .errors import KinectError

__all__ = [
    "Capture",
    "Frame",
    "FrameStats",
    "decode_depth16",
    "decode_ir16",
    "depth_stats",
    "frame_from_buffer",
    "is_jpeg",
]

#: JPEG Start-Of-Image marker — what an MJPG payload begins with.
_JPEG_SOI = b"\xff\xd8\xff"


@dataclass(frozen=True)
class Frame:
    """One image from a capture: buffer + geometry + timestamps."""

    fmt: ImageFormat
    width: int
    height: int
    stride: int
    data: bytes = b""
    device_timestamp_usec: int = 0
    system_timestamp_nsec: int = 0
    exposure_usec: int = 0
    white_balance: int = 0
    iso_speed: int = 0

    @property
    def expected_size(self) -> int | None:
        """Buffer bytes the geometry implies, or None for MJPG."""
        per_pixel = bytes_per_pixel(self.fmt)
        if per_pixel is None:
            return None
        if self.fmt is ImageFormat.COLOR_NV12:
            # Luma plane + a half-height chroma plane of the same stride.
            return self.stride * self.height * 3 // 2
        return self.stride * self.height

    @property
    def is_valid_shape(self) -> bool:
        """The buffer is at least as large as the geometry implies."""
        expected = self.expected_size
        if expected is None:
            return bool(self.data)
        return len(self.data) >= expected

    def depth_values(self) -> list[list[int]]:
        """Depth16 / IR16 buffer as rows of uint16 (little-endian)."""
        if self.fmt not in (ImageFormat.DEPTH16, ImageFormat.IR16):
            raise KinectError(
                f"depth_values needs a DEPTH16/IR16 frame, got "
                f"{self.fmt.name}", kind="wrong_format")
        return decode_depth16(self.data, self.width, self.height,
                              self.stride)

    def depth_stats(self) -> FrameStats:
        """Summary statistics over the depth values (zeros ignored)."""
        return depth_stats(self.depth_values())


@dataclass(frozen=True)
class FrameStats:
    """Summary of a depth frame: valid pixels + value distribution."""

    width: int
    height: int
    valid_pixels: int
    zero_pixels: int
    min_mm: int
    max_mm: int
    mean_mm: float

    @property
    def valid_fraction(self) -> float:
        """Share of pixels carrying a real depth."""
        total = self.width * self.height
        return self.valid_pixels / total if total else 0.0


@dataclass(frozen=True)
class Capture:
    """One device capture: color + depth + IR frames + temperature."""

    color: Frame | None = None
    depth: Frame | None = None
    ir: Frame | None = None
    temperature_c: float = 0.0

    @property
    def has_depth(self) -> bool:
        """The capture carries a depth image."""
        return self.depth is not None

    @property
    def has_color(self) -> bool:
        """The capture carries a color image."""
        return self.color is not None


def decode_depth16(data: bytes, width: int, height: int,
                   stride: int) -> list[list[int]]:
    """A DEPTH16/IR16 buffer as rows of uint16 millimetre values.

    Each row is ``stride`` bytes; only the first ``width * 2`` bytes
    of a row are pixels — the rest is row padding.
    """
    if width <= 0 or height <= 0:
        raise KinectError(
            f"bad frame geometry {width}x{height}", kind="bad_geometry")
    if stride < width * 2:
        raise KinectError(
            f"stride {stride} is too small for {width} uint16 "
            "pixels", kind="bad_geometry")
    if len(data) < stride * (height - 1) + width * 2:
        raise KinectError(
            f"buffer {len(data)} bytes is too small for a "
            f"{width}x{height} stride-{stride} frame",
            kind="short_buffer")
    rows: list[list[int]] = []
    for y in range(height):
        start = y * stride
        row = array.array("H")
        row.frombytes(data[start:start + width * 2])
        if sys.byteorder != "little":
            # The wire format is little-endian; swap on big-endian hosts.
            row.byteswap()
        rows.append(list(row))
    return rows


def decode_ir16(data: bytes, width: int, height: int,
                stride: int) -> list[list[int]]:
    """An IR16 buffer as rows of uint16 — the same layout as depth."""
    return decode_depth16(data, width, height, stride)


def depth_stats(depth_mm: list[list[int]]) -> FrameStats:
    """Min / max / mean over the non-zero depths of a depth map."""
    height = len(depth_mm)
    width = len(depth_mm[0]) if height else 0
    valid = [v for row in depth_mm for v in row if v != 0]
    zeros = width * height - len(valid)
    if not valid:
        return FrameStats(width=width, height=height,
                          valid_pixels=0, zero_pixels=zeros,
                          min_mm=0, max_mm=0, mean_mm=0.0)
    return FrameStats(
        width=width, height=height, valid_pixels=len(valid),
        zero_pixels=zeros, min_mm=min(valid), max_mm=max(valid),
        mean_mm=sum(valid) / len(valid))


def is_jpeg(data: bytes) -> bool:
    """True when the buffer starts with a JPEG SOI marker."""
    return data[:3] == _JPEG_SOI


def frame_from_buffer(fmt: ImageFormat, width: int, height: int,
                      stride: int, data: bytes, **timestamps
                      ) -> Frame:
    """Build a Frame from a raw buffer, validating its shape."""
    frame = Frame(fmt=fmt, width=width, height=height, stride=stride,
                  data=bytes(data), **timestamps)
    if not frame.is_valid_shape:
        raise KinectError(
            f"{fmt.name} {width}x{height} stride {stride} needs "
            f"{frame.expected_size} bytes, got {len(data)}",
            kind="short_buffer")
    return frame


def depth_to_xyz_mm(depth_mm: float, x: int, y: int, fx: float,
                    fy: float, cx: float, cy: float
                    ) -> tuple[float, float, float]:
    """Pinhole unprojection of one depth pixel (no distortion)."""
    if depth_mm == 0.0:
        return (0.0, 0.0, 0.0)
    return ((x - cx) * depth_mm / fx,
            (y - cy) * depth_mm / fy,
            depth_mm)
