"""Float images — the material organ's pixel plane.

MaterialCraft port of upstream's ``crates/image``: an image is a
grayscale or RGBA plane of float32 values, shared by reference until
a cook writes (copy-on-write). Node cooks NEVER mutate an input
image in place — every op allocates a new array — so cached images
are safe to share between cooks, graphs and previews.

Sampling follows upstream: wrap or clamp at the edges, bilinear
interpolation, box/bilinear resampling, sRGB encode for display and
8-bit export through PIL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image as _PILImage


def _as_float_array(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] not in (1, 4):
        raise ValueError(f"image must be (h, w, 1|4), got shape {arr.shape}")
    return np.ascontiguousarray(arr)


@dataclass
class FloatImage:
    """A grayscale (1 channel) or RGBA (4 channel) float32 image.

    Values are linear-light floats, normally in [0, 1] for display
    maps; height/normal maps may carry values outside that range.
    The array is treated as immutable by every cook: ops allocate.
    """

    data: np.ndarray

    def __init__(self, data: np.ndarray) -> None:
        self.data = _as_float_array(data)

    @property
    def h(self) -> int:
        return int(self.data.shape[0])

    @property
    def w(self) -> int:
        return int(self.data.shape[1])

    @property
    def channels(self) -> int:
        return int(self.data.shape[2])

    @property
    def kind(self) -> str:
        return "gray" if self.channels == 1 else "color"

    @classmethod
    def gray(cls, value: float, w: int, h: int) -> FloatImage:
        return cls(np.full((h, w, 1), float(value), dtype=np.float32))

    @classmethod
    def color(cls, rgba: tuple[float, float, float, float],
              w: int, h: int) -> FloatImage:
        arr = np.empty((h, w, 4), dtype=np.float32)
        arr[:, :] = rgba
        return cls(arr)

    @classmethod
    def magenta(cls, w: int, h: int) -> FloatImage:
        """Upstream's failing-node image: magenta, error in the log."""
        return cls.color((1.0, 0.0, 1.0, 1.0), w, h)

    def copy(self) -> FloatImage:
        return FloatImage(self.data.copy())

    def to_gray(self) -> FloatImage:
        if self.channels == 1:
            return self.copy()
        rgb = self.data[:, :, :3].astype(np.float32)
        lum = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1]
               + 0.0722 * rgb[:, :, 2])
        return FloatImage(lum[:, :, None])

    def to_color(self) -> FloatImage:
        if self.channels == 4:
            return self.copy()
        g = self.data[:, :, 0]
        arr = np.stack([g, g, g, np.ones_like(g)], axis=2)
        return FloatImage(arr.astype(np.float32))

    def mean(self) -> list[float]:
        return [float(v) for v in self.data.mean(axis=(0, 1))]

    def sample(self, u: np.ndarray, v: np.ndarray,
               wrap: tuple[bool, bool] = (True, True)) -> np.ndarray:
        """Bilinear sample at normalised coords u, v (arrays, any shape).

        Returns an array of shape u.shape + (channels,).
        """
        h, w, c = self.data.shape
        x = np.asarray(u, dtype=np.float64) * w - 0.5
        y = np.asarray(v, dtype=np.float64) * h - 0.5
        x = np.mod(x, w) if wrap[0] else np.clip(x, 0.0, w - 1.0)
        y = np.mod(y, h) if wrap[1] else np.clip(y, 0.0, h - 1.0)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = (x - x0)[..., None]
        fy = (y - y0)[..., None]
        x1 = (x0 + 1) % w if wrap[0] else np.minimum(x0 + 1, w - 1)
        y1 = (y0 + 1) % h if wrap[1] else np.minimum(y0 + 1, h - 1)
        a = self.data[y0, x0]
        b = self.data[y0, x1]
        d = self.data[y1, x0]
        e = self.data[y1, x1]
        top = a + (b - a) * fx
        bottom = d + (e - d) * fx
        return (top + (bottom - top) * fy).astype(np.float32)

    def resample(self, w: int, h: int) -> FloatImage:
        """Box/bilinear resample to (w, h): bilinear sample at centres."""
        if w == self.w and h == self.h:
            return self.copy()
        xs = (np.arange(w, dtype=np.float64) + 0.5) / w
        ys = (np.arange(h, dtype=np.float64) + 0.5) / h
        uu, vv = np.meshgrid(xs, ys)
        return FloatImage(self.sample(uu, vv, wrap=(False, False)))

    def to_8bit(self) -> _PILImage.Image:
        """sRGB-encode to an 8-bit PIL image (L for gray, RGBA else)."""
        arr = np.clip(self.data, 0.0, 1.0)
        arr = np.where(arr <= 0.0031308, arr * 12.92,
                       1.055 * np.power(arr, 1.0 / 2.4) - 0.055)
        px = np.round(arr * 255.0).astype(np.uint8)
        if self.channels == 1:
            return _PILImage.fromarray(px[:, :, 0], mode="L")
        return _PILImage.fromarray(px, mode="RGBA")

    @classmethod
    def from_8bit(cls, pil: _PILImage.Image) -> FloatImage:
        """Decode an 8-bit PIL image to linear-light float planes."""
        img = pil.convert("RGBA") if pil.mode not in ("L", "LA") else pil.convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.where(arr <= 0.04045, arr / 12.92,
                       np.power((arr + 0.055) / 1.055, 2.4))
        return cls(arr.astype(np.float32))

    def key_bytes(self) -> bytes:
        """Bytes that identify this image's content for cache keys."""
        return self.data.tobytes()
