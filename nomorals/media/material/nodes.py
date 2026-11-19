"""Node library — definitions + CPU cooks.

MaterialCraft port of upstream's ``crates/nodes`` (``catalog.rs``,
``generators.rs``, ``patterns.rs``, ``filters.rs``, ``noise.rs``,
``material.rs``): every node is a catalog entry — typed ports,
parameters with defaults and ranges, a library category — and a
pure cook function. A cook is ``fn(ctx) -> FloatImage``; ``ctx``
carries the output size, the seed, tiling, the inputs (``None``
when unconnected) and the resolved parameters.

Upstream ships 227 nodes; this port ships the 26 that cover the
whole workflow — generators, patterns, filters and the PBR
utilities — each cooking real pixels. Pixel-based parameters scale
with ``w / 256`` (upstream's preview rule) through
``CookCtx.pixel_scale``.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image as _PILImage

from .errors import MaterialError, MaterialErrorKind
from .image import FloatImage
from .noise import fbm, hash2, warp_field, worley

_DEFAULT_GRADIENT: list = [[0.0, 0.0, 0.0, 0.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 1.0]]


@dataclass
class PortDef:
    id: str
    label: str
    kind: str  # "gray" | "color" | "any"


@dataclass
class ParamDef:
    id: str
    label: str
    kind: str  # float|int|bool|color|text|gradient|curve|enum
    default: object
    minimum: float | None = None
    maximum: float | None = None
    options: list[str] = field(default_factory=list)


@dataclass
class CookCtx:
    size: tuple[int, int]  # (w, h)
    seed: int
    tiling: tuple[bool, bool]
    inputs: list[FloatImage | None]
    params: dict

    @property
    def w(self) -> int:
        return self.size[0]

    @property
    def h(self) -> int:
        return self.size[1]

    @property
    def pixel_scale(self) -> float:
        return self.w / 256.0

    def f(self, param: str, default: float = 0.0) -> float:
        value = self.params.get(param, default)
        return float(value) if isinstance(value, (int, float)) else default

    def i(self, param: str, default: int = 0) -> int:
        value = self.params.get(param, default)
        return int(value) if isinstance(value, (int, float)) else default

    def b(self, param: str, default: bool = False) -> bool:
        return bool(self.params.get(param, default))

    def text(self, param: str, default: str = "") -> str:
        value = self.params.get(param, default)
        return str(value) if isinstance(value, str) else default

    def color(self, param: str,
              default: tuple[float, float, float, float] = (1, 1, 1, 1)
              ) -> tuple[float, float, float, float]:
        value = self.params.get(param)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            vals = [float(v) for v in value[:4]]
            while len(vals) < 4:
                vals.append(1.0)
            return (vals[0], vals[1], vals[2], vals[3])
        return default

    def gradient(self, param: str) -> list:
        value = self.params.get(param)
        if isinstance(value, list) and value:
            stops = []
            for stop in value:
                if isinstance(stop, (list, tuple)) and len(stop) >= 5:
                    stops.append([float(v) for v in stop[:5]])
            if stops:
                return sorted(stops, key=lambda s: s[0])
        return [list(s) for s in _DEFAULT_GRADIENT]

    def curve(self, param: str) -> list:
        value = self.params.get(param)
        if isinstance(value, list) and value:
            pts = []
            for pt in value:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pts.append([float(pt[0]), float(pt[1])])
            if pts:
                return sorted(pts, key=lambda p: p[0])
        return [[0.0, 0.0], [1.0, 1.0]]

    def main(self) -> FloatImage | None:
        return self.inputs[0] if self.inputs else None

    def gray_or(self, index: int, value: float) -> FloatImage:
        img = self.inputs[index] if index < len(self.inputs) else None
        if img is None:
            return FloatImage.gray(value, self.w, self.h)
        return img.to_gray()

    def any_or(self, index: int, value: tuple[float, float, float, float]
               ) -> FloatImage:
        img = self.inputs[index] if index < len(self.inputs) else None
        if img is None:
            return FloatImage.color(value, self.w, self.h)
        return img


@dataclass
class NodeDef:
    id: str
    label: str
    category: str
    description: str
    inputs: list[PortDef]
    outputs: list[PortDef]
    params: list[ParamDef]
    cook: object = None  # Callable[[CookCtx], FloatImage]

    def param(self, param_id: str) -> ParamDef | None:
        for param in self.params:
            if param.id == param_id:
                return param
        return None

    def defaults(self) -> dict:
        out: dict = {}
        for param in self.params:
            value = param.default
            if isinstance(value, list):
                value = [list(v) if isinstance(v, list) else v
                         for v in value]
            out[param.id] = value
        return out

    def input_index(self, port_id: str) -> int | None:
        for i, port in enumerate(self.inputs):
            if port.id == port_id:
                return i
        return None


def resolve_params(node_params: dict, definition: NodeDef) -> dict:
    """Defaults overridden by the node's stored params, clamped."""
    resolved = definition.defaults()
    for key, value in node_params.items():
        param = definition.param(key)
        if param is None:
            continue  # unknown params are ignored, never fatal
        if param.kind == "float" and isinstance(value, (int, float)):
            low = param.minimum if param.minimum is not None else value
            high = param.maximum if param.maximum is not None else value
            resolved[key] = float(min(max(float(value), low), high))
        elif param.kind == "int" and isinstance(value, (int, float)):
            low = int(param.minimum) if param.minimum is not None else int(value)
            high = int(param.maximum) if param.maximum is not None else int(value)
            resolved[key] = int(min(max(int(value), low), high))
        elif param.kind == "bool":
            resolved[key] = bool(value)
        elif param.kind == "enum" and isinstance(value, str):
            resolved[key] = value if value in param.options else param.default
        else:
            resolved[key] = value
    return resolved


def eval_gradient(stops: list, t: np.ndarray) -> np.ndarray:
    """Evaluate gradient stops at t (array in [0, 1]) → (…, 4) RGBA."""
    pos = np.array([s[0] for s in stops], dtype=np.float64)
    channels = []
    for c in range(1, 5):
        vals = np.array([s[c] for s in stops], dtype=np.float64)
        channels.append(np.interp(t, pos, vals))
    return np.stack(channels, axis=-1).astype(np.float32)


def eval_curve(points: list, t: np.ndarray) -> np.ndarray:
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    return np.interp(t, xs, ys).astype(np.float32)


# -- image ops shared by the cooks -------------------------------------

def _box_blur(data: np.ndarray, radius: int) -> np.ndarray:
    """Box blur via an integral image; radius 0 returns the input."""
    if radius <= 0:
        return data.copy()
    h, w = data.shape[:2]
    padded = np.pad(data, ((radius, radius), (radius, radius), (0, 0)),
                    mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1,
                         data.shape[2]), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(padded.astype(np.float64),
                                            axis=0), axis=1)
    k = 2 * radius + 1
    total = (integral[k:k + h, k:k + w] - integral[:h, k:k + w]
             - integral[k:k + h, :w] + integral[:h, :w])
    return (total / (k * k)).astype(np.float32)


def _gaussian_blur(data: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian blur approximated by three box blurs (upstream's
    cheap path; a true Gaussian is a remainder, named in the survey)."""
    if radius <= 0:
        return data.copy()
    out = data
    for _ in range(3):
        out = _box_blur(out, max(1, radius // 3))
    return out


def _blend(mode: str, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = np.clip(base, 0.0, 1.0)
    s = np.clip(blend, 0.0, 1.0)
    if mode == "multiply":
        return b * s
    if mode == "screen":
        return 1.0 - (1.0 - b) * (1.0 - s)
    if mode == "overlay":
        return np.where(b < 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))
    if mode == "soft_light":
        return (1.0 - 2.0 * s) * b * b + 2.0 * b * s
    if mode == "hard_light":
        return np.where(s < 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))
    if mode == "add":
        return np.clip(b + s, 0.0, 1.0)
    if mode == "subtract":
        return np.clip(b - s, 0.0, 1.0)
    if mode == "difference":
        return np.abs(b - s)
    if mode == "lighten":
        return np.maximum(b, s)
    if mode == "darken":
        return np.minimum(b, s)
    if mode == "divide":
        return np.clip(b / np.maximum(s, 1e-6), 0.0, 1.0)
    return s


def _rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    light = (mx + mn) / 2.0
    d = mx - mn
    s = np.where(light == 0.0, 0.0,
                 np.where(light < 0.5, d / np.maximum(mx + mn, 1e-9),
                          d / np.maximum(2.0 - mx - mn, 1e-9)))
    s = np.where(d == 0.0, 0.0, s)
    rc = np.where(d == 0.0, 0.0, (mx - r) / np.maximum(d, 1e-9))
    gc = np.where(d == 0.0, 0.0, (mx - g) / np.maximum(d, 1e-9))
    bc = np.where(d == 0.0, 0.0, (mx - b) / np.maximum(d, 1e-9))
    h = np.where(mx == r, bc - gc,
                 np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.mod(h / 6.0, 1.0)
    h = np.where(d == 0.0, 0.0, h)
    return h, s, light


def _hsl_to_rgb(h: np.ndarray, s: np.ndarray, light: np.ndarray) -> np.ndarray:
    def hue2rgb(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = np.mod(t, 1.0)
        return np.where(t < 1 / 6, p + (q - p) * 6.0 * t,
                        np.where(t < 1 / 2, q,
                                 np.where(t < 2 / 3,
                                          p + (q - p) * (2 / 3 - t) * 6.0, p)))
    q = np.where(light < 0.5, light * (1.0 + s),
                 light + s - light * s)
    p = 2.0 * light - q
    r = hue2rgb(p, q, h + 1 / 3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1 / 3)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# -- cooks ---------------------------------------------------------------

def _cook_uniform(ctx: CookCtx) -> FloatImage:
    return FloatImage.color(ctx.color("color"), ctx.w, ctx.h)


def _cook_input_color(ctx: CookCtx) -> FloatImage:
    return FloatImage.color(ctx.color("color"), ctx.w, ctx.h)


def _cook_input_grayscale(ctx: CookCtx) -> FloatImage:
    return FloatImage.gray(ctx.f("value", 0.5), ctx.w, ctx.h)


def _cook_bitmap(ctx: CookCtx) -> FloatImage:
    raw = ctx.text("data", "")
    if not raw:
        return FloatImage.gray(0.0, ctx.w, ctx.h)
    try:
        pil = _PILImage.open(io.BytesIO(base64.b64decode(raw)))
        img = FloatImage.from_8bit(pil)
    except Exception as exc:
        raise MaterialError(MaterialErrorKind.bad_format,
                            f"bitmap data is not a readable image: {exc}") from exc
    if img.w != ctx.w or img.h != ctx.h:
        img = img.resample(ctx.w, ctx.h)
    return img


def _cook_perlin(ctx: CookCtx) -> FloatImage:
    kind = ctx.text("kind", "perlin")
    data = fbm(ctx.w, ctx.h, ctx.f("scale", 4.0),
               ctx.i("octaves", 4), ctx.seed,
               lacunarity=ctx.f("lacunarity", 2.0),
               gain=ctx.f("gain", 0.5), kind=kind, tiling=ctx.tiling)
    return FloatImage(data[:, :, None])


def _cook_cellular(ctx: CookCtx) -> FloatImage:
    data = worley(ctx.w, ctx.h, ctx.f("scale", 4.0), ctx.seed,
                  mode=ctx.text("mode", "f2-f1"), tiling=ctx.tiling)
    return FloatImage(data[:, :, None])


def _cook_voronoi(ctx: CookCtx) -> FloatImage:
    data = worley(ctx.w, ctx.h, ctx.f("scale", 4.0), ctx.seed,
                  mode="cell", tiling=ctx.tiling)
    edge = worley(ctx.w, ctx.h, ctx.f("scale", 4.0), ctx.seed,
                  mode="f2-f1", tiling=ctx.tiling)
    width = max(0.0, min(0.5, ctx.f("edge_width", 0.1)))
    plates = np.clip(data, 0.0, 1.0)
    plates = np.where(edge < width, 0.0, plates)
    return FloatImage(plates[:, :, None].astype(np.float32))


def _cook_brick(ctx: CookCtx) -> FloatImage:
    scale = ctx.pixel_scale
    bw = max(1.0, ctx.f("brick_width", 32.0) * scale)
    bh = max(1.0, ctx.f("brick_height", 16.0) * scale)
    mortar = max(0.0, ctx.f("mortar", 2.0) * scale)
    offset = ctx.f("offset", 0.5)
    variation = ctx.f("variation", 0.3)
    color_a = np.array(ctx.color("color_a", (0.6, 0.25, 0.2, 1.0)),
                       dtype=np.float32)
    color_b = np.array(ctx.color("color_b", (0.55, 0.22, 0.18, 1.0)),
                       dtype=np.float32)
    mortar_color = np.array(ctx.color("mortar_color", (0.8, 0.8, 0.78, 1.0)),
                            dtype=np.float32)
    ys = (np.arange(ctx.h, dtype=np.float64) + 0.5)
    xs = (np.arange(ctx.w, dtype=np.float64) + 0.5)
    xx, yy = np.meshgrid(xs, ys)
    row = np.floor(yy / bh).astype(np.int64)
    shift = np.where(row % 2 == 0, 0.0, bw * offset)
    col = np.floor((xx + shift) / bw).astype(np.int64)
    fx = np.mod(xx + shift, bw)
    fy = np.mod(yy, bh)
    in_mortar = (fx < mortar) | (fx > bw - mortar) | (fy < mortar) | (fy > bh - mortar)
    vary = hash2(col, row, ctx.seed) * variation
    pick_a = hash2(col, row, ctx.seed + 7) < 0.5
    brick = np.where(pick_a[..., None], color_a, color_b)
    factor = (1.0 - variation / 2.0 + vary[..., None])
    brick_rgb = np.clip(brick[:, :, :3] * factor, 0, 1)
    brick = np.concatenate([brick_rgb, brick[:, :, 3:4]], axis=2)
    out = np.where(in_mortar[..., None], mortar_color, brick)
    return FloatImage(out.astype(np.float32))


def _cook_tile(ctx: CookCtx) -> FloatImage:
    rows = max(1, ctx.i("rows", 4))
    cols = max(1, ctx.i("cols", 4))
    gap = max(0.0, ctx.f("gap", 2.0) * ctx.pixel_scale)
    variation = ctx.f("variation", 0.2)
    color_a = np.array(ctx.color("color_a", (0.7, 0.7, 0.7, 1.0)), dtype=np.float32)
    color_b = np.array(ctx.color("color_b", (0.3, 0.3, 0.3, 1.0)), dtype=np.float32)
    gap_color = np.array(ctx.color("gap_color", (0.1, 0.1, 0.1, 1.0)), dtype=np.float32)
    ys = (np.arange(ctx.h, dtype=np.float64) + 0.5)
    xs = (np.arange(ctx.w, dtype=np.float64) + 0.5)
    xx, yy = np.meshgrid(xs, ys)
    row = np.minimum(np.floor(yy / ctx.h * rows).astype(np.int64), rows - 1)
    col = np.minimum(np.floor(xx / ctx.w * cols).astype(np.int64), cols - 1)
    cell_w = ctx.w / cols
    cell_h = ctx.h / rows
    fx = np.mod(xx, cell_w)
    fy = np.mod(yy, cell_h)
    in_gap = (fx < gap) | (fx > cell_w - gap) | (fy < gap) | (fy > cell_h - gap)
    vary = hash2(col, row, ctx.seed) * variation
    pick_a = ((row + col) % 2 == 0)
    tile = np.where(pick_a[..., None], color_a, color_b)
    factor = (1.0 - variation / 2.0 + vary[..., None])
    tile_rgb = np.clip(tile[:, :, :3] * factor, 0, 1)
    tile = np.concatenate([tile_rgb, tile[:, :, 3:4]], axis=2)
    out = np.where(in_gap[..., None], gap_color, tile)
    return FloatImage(out.astype(np.float32))


def _cook_checker(ctx: CookCtx) -> FloatImage:
    cells = max(1, ctx.i("cells", 8))
    color_a = np.array(ctx.color("color_a", (1, 1, 1, 1)), dtype=np.float32)
    color_b = np.array(ctx.color("color_b", (0, 0, 0, 1)), dtype=np.float32)
    ys = np.floor(np.arange(ctx.h)[:, None] / ctx.h * cells).astype(np.int64)
    xs = np.floor(np.arange(ctx.w)[None, :] / ctx.w * cells).astype(np.int64)
    pick = ((ys + xs) % 2 == 0)
    out = np.where(pick[..., None], color_a, color_b)
    return FloatImage(out.astype(np.float32))


def _cook_gradient_linear(ctx: CookCtx) -> FloatImage:
    angle = math.radians(ctx.f("angle", 0.0))
    stops = ctx.gradient("stops")
    ys = (np.arange(ctx.h, dtype=np.float64) + 0.5) / ctx.h - 0.5
    xs = (np.arange(ctx.w, dtype=np.float64) + 0.5) / ctx.w - 0.5
    xx, yy = np.meshgrid(xs, ys)
    t = np.clip((xx * math.cos(angle) + yy * math.sin(angle)) + 0.5, 0, 1)
    return FloatImage(eval_gradient(stops, t))


def _cook_gradient_radial(ctx: CookCtx) -> FloatImage:
    center = ctx.params.get("center", [0.5, 0.5])
    cx = float(center[0]) if isinstance(center, (list, tuple)) else 0.5
    cy = float(center[1]) if isinstance(center, (list, tuple)) else 0.5
    stops = ctx.gradient("stops")
    ys = (np.arange(ctx.h, dtype=np.float64) + 0.5) / ctx.h
    xs = (np.arange(ctx.w, dtype=np.float64) + 0.5) / ctx.w
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    t = np.clip(dist / max(np.sqrt(2.0), 1e-9) * np.sqrt(2.0), 0, 1)
    return FloatImage(eval_gradient(stops, t))


def _cook_shape(ctx: CookCtx) -> FloatImage:
    kind = ctx.text("kind", "circle")
    size = max(0.01, min(1.0, ctx.f("size", 0.4)))
    feather = max(0.0, ctx.f("feather", 0.02))
    color = np.array(ctx.color("color", (1, 1, 1, 1)), dtype=np.float32)
    bg = np.array(ctx.color("background", (0, 0, 0, 1)), dtype=np.float32)
    ys = (np.arange(ctx.h, dtype=np.float64) + 0.5) / ctx.h - 0.5
    xs = (np.arange(ctx.w, dtype=np.float64) + 0.5) / ctx.w - 0.5
    xx, yy = np.meshgrid(xs, ys)
    if kind == "square":
        dist = np.maximum(np.abs(xx), np.abs(yy)) / (size / 2.0)
    elif kind == "triangle":
        dist = (np.abs(xx) * 0.866 + yy * 0.5) / (size / 2.0)
        dist = np.maximum(dist, -yy / (size / 2.0))
    else:
        dist = np.sqrt(xx ** 2 + yy ** 2) / (size / 2.0)
    alpha = np.clip((1.0 - dist) / max(feather, 1e-6) + 0.5, 0, 1)
    alpha = np.where(dist <= 1.0 - feather, 1.0, alpha)
    alpha = np.where(dist > 1.0, 0.0, alpha)
    out = bg + (color - bg) * alpha[..., None]
    return FloatImage(out.astype(np.float32))


def _cook_gradient_map(ctx: CookCtx) -> FloatImage:
    src = ctx.gray_or(0, 0.5)
    stops = ctx.gradient("stops")
    t = np.clip(src.data[:, :, 0], 0, 1)
    return FloatImage(eval_gradient(stops, t))


def _cook_blend(ctx: CookCtx) -> FloatImage:
    base = ctx.any_or(0, (0.0, 0.0, 0.0, 1.0))
    top = ctx.any_or(1, (1.0, 1.0, 1.0, 1.0))
    if top.w != base.w or top.h != base.h:
        top = top.resample(base.w, base.h)
    # a gray side converts to color first (alpha 1): a gray image
    # has no alpha, so its value must never become the alpha.
    if base.channels != top.channels:
        if base.channels == 1:
            base = base.to_color()
        if top.channels == 1:
            top = top.to_color()
    mode = ctx.text("mode", "normal")
    opacity = ctx.f("opacity", 1.0)
    blended = _blend(mode, base.data, top.data)
    out = base.data + (blended - base.data) * opacity
    return FloatImage(np.clip(out, 0, 1).astype(np.float32))


def _cook_levels(ctx: CookCtx) -> FloatImage:
    src = ctx.main()
    if src is None:
        return FloatImage.gray(0.0, ctx.w, ctx.h)
    data = src.data.astype(np.float64)
    in_black = ctx.f("in_black", 0.0)
    in_white = max(ctx.f("in_white", 1.0), in_black + 1e-6)
    gamma = max(ctx.f("gamma", 1.0), 1e-6)
    out_black = ctx.f("out_black", 0.0)
    out_white = ctx.f("out_white", 1.0)
    norm = np.clip((data - in_black) / (in_white - in_black), 0, 1)
    norm = np.power(norm, 1.0 / gamma)
    out = out_black + norm * (out_white - out_black)
    return FloatImage(np.clip(out, 0, 1).astype(np.float32))


def _cook_blur(ctx: CookCtx) -> FloatImage:
    src = ctx.main()
    if src is None:
        return FloatImage.gray(0.0, ctx.w, ctx.h)
    radius = max(0, int(round(ctx.f("radius", 4.0) * ctx.pixel_scale)))
    mode = ctx.text("mode", "box")
    data = (_gaussian_blur(src.data, radius) if mode == "gaussian"
            else _box_blur(src.data, radius))
    return FloatImage(data)


def _cook_hsl(ctx: CookCtx) -> FloatImage:
    src = ctx.any_or(0, (0.5, 0.5, 0.5, 1.0)).to_color()
    rgb = src.data[:, :, :3].astype(np.float64)
    h, s, light = _rgb_to_hsl(rgb)
    h = np.mod(h + ctx.f("hue", 0.0) / 360.0, 1.0)
    s = np.clip(s + ctx.f("saturation", 0.0), 0, 1)
    light = np.clip(light + ctx.f("lightness", 0.0), 0, 1)
    out = _hsl_to_rgb(h, s, light)
    result = np.concatenate([out, src.data[:, :, 3:4]], axis=2)
    return FloatImage(result.astype(np.float32))


def _cook_invert(ctx: CookCtx) -> FloatImage:
    src = ctx.main()
    if src is None:
        return FloatImage.gray(1.0, ctx.w, ctx.h)
    return FloatImage((1.0 - src.data).astype(np.float32))


def _cook_transform(ctx: CookCtx) -> FloatImage:
    src = ctx.main()
    if src is None:
        return FloatImage.gray(0.0, ctx.w, ctx.h)
    translate = ctx.params.get("translate", [0.0, 0.0])
    tx = float(translate[0]) if isinstance(translate, (list, tuple)) else 0.0
    ty = float(translate[1]) if isinstance(translate, (list, tuple)) else 0.0
    angle = math.radians(ctx.f("rotate", 0.0))
    scale = max(ctx.f("scale", 1.0), 1e-6)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ys = (np.arange(ctx.h, dtype=np.float64) + 0.5) / ctx.h - 0.5
    xs = (np.arange(ctx.w, dtype=np.float64) + 0.5) / ctx.w - 0.5
    xx, yy = np.meshgrid(xs, ys)
    sx = (xx * cos_a - yy * sin_a) / scale + 0.5 + tx
    sy = (xx * sin_a + yy * cos_a) / scale + 0.5 + ty
    return FloatImage(src.sample(sx, sy, wrap=ctx.tiling))


def _cook_warp(ctx: CookCtx) -> FloatImage:
    src = ctx.main()
    if src is None:
        return FloatImage.gray(0.0, ctx.w, ctx.h)
    intensity = ctx.f("intensity", 0.1)
    dx, dy = warp_field(ctx.w, ctx.h, ctx.f("scale", 4.0), ctx.seed,
                        tiling=ctx.tiling)
    ys = (np.arange(ctx.h, dtype=np.float64) + 0.5) / ctx.h
    xs = (np.arange(ctx.w, dtype=np.float64) + 0.5) / ctx.w
    xx, yy = np.meshgrid(xs, ys)
    return FloatImage(src.sample(xx + dx * intensity, yy + dy * intensity,
                                 wrap=ctx.tiling))


def _cook_curve(ctx: CookCtx) -> FloatImage:
    src = ctx.main()
    if src is None:
        return FloatImage.gray(0.0, ctx.w, ctx.h)
    points = ctx.curve("points")
    out = eval_curve(points, np.clip(src.data, 0, 1))
    return FloatImage(np.clip(out, 0, 1).astype(np.float32))


def _cook_height_to_normal(ctx: CookCtx) -> FloatImage:
    src = ctx.gray_or(0, 0.5)
    strength = max(ctx.f("strength", 1.0), 0.0)
    data = src.data[:, :, 0].astype(np.float64)
    padded = np.pad(data, 1, mode="edge")
    dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) * strength * 0.5
    dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * strength * 0.5
    normal = np.stack([-dx, -dy, np.ones_like(dx)], axis=2)
    norm = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    normal = normal / np.maximum(norm, 1e-9)
    rgb = (normal * 0.5 + 0.5)
    out = np.concatenate([rgb, np.ones_like(rgb[:, :, :1])], axis=2)
    return FloatImage(out.astype(np.float32))


def _cook_roughness_from_height(ctx: CookCtx) -> FloatImage:
    src = ctx.gray_or(0, 0.5)
    base = ctx.f("base", 0.5)
    variation = ctx.f("variation", 0.5)
    out = np.clip(base + (src.data[:, :, 0] - 0.5) * variation, 0, 1)
    return FloatImage(out[:, :, None].astype(np.float32))


def _cook_ao_from_height(ctx: CookCtx) -> FloatImage:
    src = ctx.gray_or(0, 0.5)
    strength = ctx.f("strength", 0.5)
    blurred = _box_blur(src.data, max(1, int(4 * ctx.pixel_scale)))
    cavity = np.clip(src.data - blurred, -1, 1)
    out = np.clip(1.0 - np.abs(cavity) * strength * 2.0, 0, 1)
    return FloatImage(out.astype(np.float32))


def _cook_output(ctx: CookCtx) -> FloatImage:
    src = ctx.main()
    if src is None:
        return FloatImage.gray(0.0, ctx.w, ctx.h)
    return src.copy()


# -- catalog ---------------------------------------------------------------

def _float(param: str, label: str, default: float,
           low: float, high: float) -> ParamDef:
    return ParamDef(param, label, "float", default, low, high)


def _int(param: str, label: str, default: int, low: int, high: int) -> ParamDef:
    return ParamDef(param, label, "int", default, float(low), float(high))


def _enum(param: str, label: str, default: str, options: list[str]) -> ParamDef:
    return ParamDef(param, label, "enum", default, options=options)


def _color(param: str, label: str,
           default: tuple[float, float, float, float]) -> ParamDef:
    return ParamDef(param, label, "color", list(default))


def _gradient(param: str, label: str) -> ParamDef:
    return ParamDef(param, label, "gradient", [list(s) for s in _DEFAULT_GRADIENT])


def _def(node_id: str, label: str, category: str, description: str,
         inputs: list[PortDef], outputs: list[PortDef],
         params: list[ParamDef], cook) -> NodeDef:
    return NodeDef(node_id, label, category, description, inputs, outputs,
                   params, cook)


_G = PortDef("output", "Output", "gray")
_C = PortDef("output", "Output", "color")
_A = PortDef("output", "Output", "any")
_GIN = PortDef("input", "Input", "gray")
_CIN = PortDef("input", "Input", "color")
_AIN = PortDef("input", "Input", "any")


def catalog() -> list[NodeDef]:
    """The node library: 25 defs across generators, patterns,
    filters and PBR utilities."""
    return [
        _def("uniform_color", "Uniform Color", "Generator",
             "A flat color.", [], [_C],
             [_color("color", "Color", (1.0, 1.0, 1.0, 1.0))],
             _cook_uniform),
        _def("input_color", "Input Color", "Input",
             "A constant color input (graph input).", [], [_C],
             [_color("color", "Color", (1.0, 1.0, 1.0, 1.0))],
             _cook_input_color),
        _def("input_grayscale", "Input Grayscale", "Input",
             "A constant gray input (graph input).", [], [_G],
             [_float("value", "Value", 0.5, 0.0, 1.0)],
             _cook_input_grayscale),
        _def("bitmap", "Bitmap", "Input",
             "An embedded image, decoded from base64 PNG data.", [], [_C],
             [ParamDef("data", "Image data (base64 PNG)", "text", "")],
             _cook_bitmap),
        _def("perlin_noise", "Perlin Noise", "Generator",
             "Fractal Perlin gradient noise, tileable.",
             [], [_G],
             [_float("scale", "Scale", 4.0, 0.5, 64.0),
              _int("octaves", "Octaves", 4, 1, 8),
              _float("lacunarity", "Lacunarity", 2.0, 1.0, 4.0),
              _float("gain", "Gain", 0.5, 0.0, 1.0),
              _enum("kind", "Noise kind", "perlin", ["perlin", "value"])],
             _cook_perlin),
        _def("cellular_noise", "Cellular Noise", "Generator",
             "Worley/cellular noise: F1, F2, crack edges or cell id.",
             [], [_G],
             [_float("scale", "Scale", 4.0, 0.5, 64.0),
              _enum("mode", "Mode", "f2-f1", ["f1", "f2", "f2-f1", "cell"])],
             _cook_cellular),
        _def("voronoi", "Voronoi", "Generator",
             "Voronoi plates with dark edges.", [], [_G],
             [_float("scale", "Scale", 4.0, 0.5, 64.0),
              _float("edge_width", "Edge width", 0.1, 0.0, 0.5)],
             _cook_voronoi),
        _def("brick_generator", "Brick Generator", "Pattern",
             "Brick courses with mortar and per-brick variation.",
             [], [_C],
             [_float("brick_width", "Brick width (px)", 32.0, 2.0, 256.0),
              _float("brick_height", "Brick height (px)", 16.0, 2.0, 256.0),
              _float("mortar", "Mortar (px)", 2.0, 0.0, 32.0),
              _float("offset", "Row offset", 0.5, 0.0, 1.0),
              _float("variation", "Variation", 0.3, 0.0, 1.0),
              _color("color_a", "Brick A", (0.6, 0.25, 0.2, 1.0)),
              _color("color_b", "Brick B", (0.55, 0.22, 0.18, 1.0)),
              _color("mortar_color", "Mortar", (0.8, 0.8, 0.78, 1.0))],
             _cook_brick),
        _def("tile_generator", "Tile Generator", "Pattern",
             "A grid of tiles with gaps and per-tile variation.",
             [], [_C],
             [_int("rows", "Rows", 4, 1, 64),
              _int("cols", "Columns", 4, 1, 64),
              _float("gap", "Gap (px)", 2.0, 0.0, 32.0),
              _float("variation", "Variation", 0.2, 0.0, 1.0),
              _color("color_a", "Tile A", (0.7, 0.7, 0.7, 1.0)),
              _color("color_b", "Tile B", (0.3, 0.3, 0.3, 1.0)),
              _color("gap_color", "Gap", (0.1, 0.1, 0.1, 1.0))],
             _cook_tile),
        _def("checker", "Checker", "Pattern",
             "A checkerboard of two colors.", [], [_C],
             [_int("cells", "Cells", 8, 1, 64),
              _color("color_a", "Color A", (1.0, 1.0, 1.0, 1.0)),
              _color("color_b", "Color B", (0.0, 0.0, 0.0, 1.0))],
             _cook_checker),
        _def("gradient_linear", "Gradient Linear", "Generator",
             "A linear gradient between stops.", [], [_C],
             [_float("angle", "Angle (deg)", 0.0, -180.0, 180.0),
              _gradient("stops", "Stops")],
             _cook_gradient_linear),
        _def("gradient_radial", "Gradient Radial", "Generator",
             "A radial gradient between stops.", [], [_C],
             [ParamDef("center", "Center", "float", [0.5, 0.5]),
              _gradient("stops", "Stops")],
             _cook_gradient_radial),
        _def("shape", "Shape", "Generator",
             "A filled shape with a soft edge.", [], [_C],
             [_enum("kind", "Shape", "circle", ["circle", "square", "triangle"]),
              _float("size", "Size", 0.4, 0.01, 1.0),
              _float("feather", "Feather", 0.02, 0.0, 0.5),
              _color("color", "Color", (1.0, 1.0, 1.0, 1.0)),
              _color("background", "Background", (0.0, 0.0, 0.0, 1.0))],
             _cook_shape),
        _def("gradient_map", "Gradient Map", "Filter",
             "Remap a gray input through gradient stops.",
             [_GIN], [_C],
             [_gradient("stops", "Stops")],
             _cook_gradient_map),
        _def("blend", "Blend", "Filter",
             "W3C blend of two images with opacity.",
             [PortDef("base", "Base", "any"), PortDef("blend", "Blend", "any")],
             [_A],
             [_enum("mode", "Mode", "normal",
                    ["normal", "multiply", "screen", "overlay", "soft_light",
                     "hard_light", "add", "subtract", "difference",
                     "lighten", "darken", "divide"]),
              _float("opacity", "Opacity", 1.0, 0.0, 1.0)],
             _cook_blend),
        _def("levels", "Levels", "Filter",
             "Input/output black/white points and gamma.", [_AIN], [_A],
             [_float("in_black", "In black", 0.0, 0.0, 1.0),
              _float("in_white", "In white", 1.0, 0.0, 1.0),
              _float("gamma", "Gamma", 1.0, 0.1, 4.0),
              _float("out_black", "Out black", 0.0, 0.0, 1.0),
              _float("out_white", "Out white", 1.0, 0.0, 1.0)],
             _cook_levels),
        _def("blur", "Blur", "Filter",
             "Box or Gaussian blur.", [_AIN], [_A],
             [_float("radius", "Radius (px)", 4.0, 0.0, 64.0),
              _enum("mode", "Mode", "box", ["box", "gaussian"])],
             _cook_blur),
        _def("hsl", "HSL", "Filter",
             "Hue shift, saturation and lightness offsets.", [_CIN], [_C],
             [_float("hue", "Hue (deg)", 0.0, -180.0, 180.0),
              _float("saturation", "Saturation", 0.0, -1.0, 1.0),
              _float("lightness", "Lightness", 0.0, -1.0, 1.0)],
             _cook_hsl),
        _def("invert", "Invert", "Filter",
             "Invert every channel.", [_AIN], [_A],
             [], _cook_invert),
        _def("transform", "Transform", "Filter",
             "Translate, rotate and scale with tiling.", [_AIN], [_A],
             [ParamDef("translate", "Translate", "float", [0.0, 0.0]),
              _float("rotate", "Rotate (deg)", 0.0, -180.0, 180.0),
              _float("scale", "Scale", 1.0, 0.1, 8.0)],
             _cook_transform),
        _def("warp", "Warp", "Filter",
             "Displace an image with a Perlin field.", [_AIN], [_A],
             [_float("intensity", "Intensity", 0.1, 0.0, 1.0),
              _float("scale", "Scale", 4.0, 0.5, 64.0)],
             _cook_warp),
        _def("curve", "Curve", "Filter",
             "Remap values through a control-point curve.", [_AIN], [_A],
             [ParamDef("points", "Points", "curve",
                       [[0.0, 0.0], [1.0, 1.0]])],
             _cook_curve),
        _def("height_to_normal", "Height to Normal", "PBR",
             "A normal map from a height field (Sobel).", [_GIN], [_C],
             [_float("strength", "Strength", 1.0, 0.0, 8.0)],
             _cook_height_to_normal),
        _def("roughness_from_height", "Roughness from Height", "PBR",
             "Roughness varies with height around a base.", [_GIN], [_G],
             [_float("base", "Base", 0.5, 0.0, 1.0),
              _float("variation", "Variation", 0.5, 0.0, 1.0)],
             _cook_roughness_from_height),
        _def("ao_from_height", "AO from Height", "PBR",
             "Ambient occlusion from height cavities.", [_GIN], [_G],
             [_float("strength", "Strength", 0.5, 0.0, 1.0)],
             _cook_ao_from_height),
        _def("output", "Output", "Output",
             "Feed a cooked image into a PBR channel.", [_AIN], [],
             [_enum("channel", "Channel", "albedo",
                    ["albedo", "roughness", "metallic", "normal", "height",
                     "ao", "emissive"])],
             _cook_output),
    ]


_CATALOG: list[NodeDef] | None = None


def by_id(def_id: str) -> NodeDef | None:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = catalog()
    for definition in _CATALOG:
        if definition.id == def_id:
            return definition
    return None
