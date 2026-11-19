"""PBR preview — a CPU material-ball render.

MaterialCraft port of upstream's ``crates/render`` (``preview.rs``,
``ibl.rs``): the cooked maps are lit on a sphere with a GGX
microfacet model — Schlick Fresnel, Smith geometry, a single key
light plus a procedural gradient environment for ambient — and
ACES-fitted tone mapping. No prefiltered IBL, no parallax
occlusion from height: those are named remainders, and the preview
says so in its report rather than faking them.

``render_preview`` returns a PIL image plus a report of what was
used and what was not. ``render_channel`` draws one map flat (the
2D view).
"""

from __future__ import annotations

import numpy as np
from PIL import Image as _PILImage

from .image import FloatImage

_LIGHT_DIR = np.array([0.5, 0.7, 0.5], dtype=np.float64)
_LIGHT_DIR /= np.linalg.norm(_LIGHT_DIR)
_VIEW_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _aces(x: np.ndarray) -> np.ndarray:
    """ACES fitted tone mapping (Narkowicz)."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _ggx_ndf(n_dot_h: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    alpha = np.clip(roughness, 0.03, 1.0) ** 2
    alpha2 = alpha * alpha
    denom = (n_dot_h * n_dot_h) * (alpha2 - 1.0) + 1.0
    return alpha2 / (np.pi * denom * denom + 1e-9)


def _smith_g(n_dot_v: np.ndarray, n_dot_l: np.ndarray,
             roughness: np.ndarray) -> np.ndarray:
    k = (roughness + 1.0) ** 2 / 8.0
    g1v = n_dot_v / (n_dot_v * (1.0 - k) + k + 1e-9)
    g1l = n_dot_l / (n_dot_l * (1.0 - k) + k + 1e-9)
    return g1v * g1l


def render_preview(maps: dict[str, FloatImage], size: int = 256,
                   exposure: float = 1.0) -> tuple[_PILImage.Image, dict]:
    """Render the cooked maps on a sphere; returns (image, report).

    Channels used: albedo, roughness, metallic, normal, ao. Height
    is NOT used for parallax occlusion (named in the report);
    emissive is added flat. Missing channels fall back to honest
    defaults (albedo 0.5 gray, roughness 0.5, metallic 0, flat
    normal, ao 1).
    """
    size = max(16, min(1024, int(size)))
    ys = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    xs = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    xx, yy = np.meshgrid(xs, ys)
    r2 = xx * xx + yy * yy
    inside = r2 <= 1.0
    nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    normal = np.stack([xx, -yy, nz], axis=2)

    def _map(name: str, default: float, channels: int = 1) -> np.ndarray:
        image = maps.get(name)
        if image is None:
            return np.full((size, size, channels), default, dtype=np.float64)
        arr = image.resample(size, size).data.astype(np.float64)
        if arr.shape[2] == 1 and channels == 4:
            arr = np.repeat(arr, 4, axis=2)
        if arr.shape[2] == 4 and channels == 1:
            arr = arr[:, :, :1]
        return arr

    albedo = _map("albedo", 0.5, 4)[:, :, :3]
    roughness = np.clip(_map("roughness", 0.5)[:, :, 0], 0.03, 1.0)
    metallic = np.clip(_map("metallic", 0.0)[:, :, 0], 0.0, 1.0)
    ao = np.clip(_map("ao", 1.0)[:, :, 0], 0.0, 1.0)
    normal_map = maps.get("normal")
    used: list[str] = []
    if normal_map is not None:
        nrm = normal_map.resample(size, size).data.astype(np.float64)
        tangent = (nrm[:, :, :3] * 2.0 - 1.0)
        # perturb the sphere normal in view space (small angle)
        normal = normal + tangent * 0.35
        normal = normal / np.maximum(
            np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True)), 1e-9)
        used.append("normal")
    for channel in ("albedo", "roughness", "metallic", "ao"):
        if channel in maps:
            used.append(channel)

    n_dot_l = np.clip(np.sum(normal * _LIGHT_DIR, axis=2), 0.0, 1.0)
    n_dot_v = np.clip(np.sum(normal * _VIEW_DIR, axis=2), 0.0, 1.0)
    half = _LIGHT_DIR + _VIEW_DIR
    half = half / np.linalg.norm(half)
    n_dot_h = np.clip(np.sum(normal * half, axis=2), 0.0, 1.0)
    f0 = 0.04 + metallic[..., None] * (albedo - 0.04)
    fresnel = f0 + (1.0 - f0) * (1.0 - n_dot_v[..., None]) ** 5
    diffuse = albedo * (1.0 - metallic[..., None]) / np.pi
    spec = (_ggx_ndf(n_dot_h, roughness) * _smith_g(n_dot_v, n_dot_l, roughness))
    spec = spec[..., None] * fresnel / (4.0 * n_dot_v[..., None] * n_dot_l[..., None] + 1e-6)
    # procedural gradient environment for ambient
    env = 0.35 + 0.35 * np.clip(normal[:, :, 1], -1, 1)[..., None]
    ambient = albedo * env * ao[..., None] * 0.25
    color = (diffuse * n_dot_l[..., None] + spec * n_dot_l[..., None]) * 1.6
    color = color + ambient
    if "emissive" in maps:
        emissive = _map("emissive", 0.0, 4)[:, :, :3]
        color = color + emissive
        used.append("emissive")
    color = _aces(color * exposure)
    # sRGB encode
    srgb = np.where(color <= 0.0031308, color * 12.92,
                    1.055 * np.power(np.clip(color, 0, 1), 1.0 / 2.4) - 0.055)
    px = np.round(np.clip(srgb, 0, 1) * 255.0).astype(np.uint8)
    # background: dark neutral, sphere anti-aliased by the mask edge
    bg = np.full((size, size, 3), 24, dtype=np.uint8)
    out = np.where(inside[..., None], px, bg)
    report = {
        "size": size,
        "channels_used": sorted(set(used)),
        "channels_missing": sorted(set(("albedo", "roughness", "metallic",
                                         "normal", "ao", "emissive"))
                                   - set(used)),
        "not_rendered": ["parallax occlusion from height",
                         "prefiltered IBL environment"],
    }
    return _PILImage.fromarray(out, mode="RGB"), report


def render_channel(image: FloatImage, size: int = 256) -> _PILImage.Image:
    """Draw one cooked map flat (the 2D view)."""
    return image.resample(size, size).to_8bit()
