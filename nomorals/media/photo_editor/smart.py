"""Smart objects — ``SoLd``/``PlLd`` parsing, the smart-filter
stack (``filterFX``), the ``FEid`` unfiltered-pixel cache, and
the edit-path filter application.

The compositor does NOT re-apply smart filters: the stored
layer pixels ARE the filtered render (proven on the corpus —
lane-76 flatten already matches the composite exactly on
every filtered smart file). This module exists for the doc
model and the edit path: with the ``FEid`` unfiltered pixels
the editor can re-render the stack from the source pixels
when a filter is added, removed or changed.

``FEid`` is present on 28 of the 30 corpus smart files (all
except the two with no filters), in the additional layer
info section. Its layout (from photocraft's
``crates/psd/src/filter_effects.rs``): u32 version (1..=3),
then sections of u64 length + items padded to 4 bytes; an
item is a Pascal id, u32 version, u64 item length, rect
(top, left, bottom, right), u32 depth, u32 max_channels,
``max_channels + 2`` slots (u32 present flag + a plane:
u64 length, u16 compression, data), then a mask flag byte
and, when set, a mask rect + plane. Planes decode like
layer channels, except RLE uses four-byte row counts
(the Psb layout).
"""

from __future__ import annotations

import math
import struct
import zlib

import numpy as np
from PIL import Image, ImageFilter

from .blend import blend_rgb
from .descriptor import (
    Desc,
    desc_bool,
    desc_desc,
    desc_enum,
    desc_list,
    desc_num,
    desc_text,
    parse_versioned,
)

__all__ = [
    "parse_sold",
    "parse_plld",
    "parse_filter_stack",
    "parse_feid",
    "apply_filter_stack",
    "apply_smart_filter",
]

# Filter class -> the Photoshop menu name.
FILTER_NAMES = {
    "GsnB": "Gaussian Blur",
    "boxblur": "Box Blur",
    "MtnB": "Motion Blur",
    "UnsM": "Unsharp Mask",
    "AdNs": "Add Noise",
    "Mdn ": "Median",
    "HghP": "High Pass",
    "Mxm ": "Maximum",
    "Mnm ": "Minimum",
    "Msc ": "Mosaic",
    "Embs": "Emboss",
    "Lvls": "Levels",
    "Crvs": "Curves",
    "adaptCorrect": "Shadows/Highlights",
    "Blr ": "Blur",
    "BlrM": "Blur More",
    "Avrg": "Average",
    "Dspc": "Despeckle",
    "FndE": "Find Edges",
    "Fct ": "Facet",
    "Frgm": "Fragment",
    "surfaceBlur": "Surface Blur",
    "SmrB": "Smart Blur",
    "RdlB": "Radial Blur",
}

# Filters with no options at all (only the four-char
# filterID is stored).
NO_OPTIONS = {"Blr ", "BlrM", "Avrg", "Dspc", "FndE", "Fct ",
              "Frgm"}


def _px(desc: Desc | None, key: str) -> float | None:
    """A ``UntF`` pixel value -> float pixels, or None."""
    if desc is None:
        return None
    value = desc.get(key)
    if isinstance(value, tuple) and len(value) == 2 \
            and value[0] == "unit":
        code, amount = value[1]
        if code == "#Pxl":
            return float(amount)
    return None


def _int(desc: Desc | None, key: str) -> int | None:
    if desc is None:
        return None
    value = desc_num(desc, key)
    return None if value is None else int(value)


def _corners_affine(corners: list[float], width: float,
                    height: float) -> list[float] | None:
    """The affine [a, b, c, d, tx, ty] (x' = a*x + c*y + tx,
    y' = b*x + d*y + ty) taking the source rect [0, 0, w, h]
    onto ``corners`` (TL, TR, BR, BL as x, y pairs)."""
    if len(corners) != 8 or width <= 0 or height <= 0:
        return None
    if not all(math.isfinite(v) for v in corners):
        return None
    p = corners
    a = (p[2] - p[0]) / width
    b = (p[3] - p[1]) / width
    c = (p[6] - p[0]) / height
    d = (p[7] - p[1]) / height
    return [a, b, c, d, p[0], p[1]]


def _filter_params(cls: str, fltr: Desc | None,
                   item: Desc) -> dict:
    """The known parameters of one filter's ``Fltr``
    descriptor, as plain values. Unknown classes and
    descriptor-form Levels/Curves return an empty dict and
    are rendered as pass-through-named."""
    if fltr is None:
        return {}
    if cls in ("GsnB", "boxblur", "Mdn ", "HghP", "Mxm ",
               "Mnm "):
        radius = _px(fltr, "Rds ")
        return {} if radius is None else {"radius": radius}
    if cls == "MtnB":
        angle = _int(fltr, "Angl")
        distance = _px(fltr, "Dstn")
        params: dict = {}
        if angle is not None:
            params["angle"] = angle
        if distance is not None:
            params["distance"] = distance
        return params
    if cls == "UnsM":
        amount = _px(fltr, "Amnt")
        radius = _px(fltr, "Rds ")
        threshold = _int(fltr, "Thsh")
        params = {}
        if amount is not None:
            params["amount"] = amount
        if radius is not None:
            params["radius"] = radius
        if threshold is not None:
            params["threshold"] = threshold
        return params
    if cls == "AdNs":
        amount = _px(fltr, "Amnt")
        params = {}
        if amount is not None:
            params["amount"] = amount
        dist = desc_enum(fltr, "Dstr")
        if dist is not None:
            params["distribution"] = dist[1]
        return params
    if cls == "Msc ":
        size = _px(fltr, "ClSz")
        return {} if size is None else {"cell_size": size}
    if cls == "Embs":
        angle = _int(fltr, "Angl")
        height = _int(fltr, "Hght")
        amount = _int(fltr, "Amnt")
        params = {}
        if angle is not None:
            params["angle"] = angle
        if height is not None:
            params["height"] = height
        if amount is not None:
            params["amount"] = amount
        return params
    return {}


def parse_filter_stack(desc: Desc) -> dict:
    """A ``filterFX`` descriptor -> the filter stack:
    enabled, the filters in order (name, class, blend,
    opacity, visible, params), mask_enabled, mask_linked.
    Unknown filters keep their name and render as
    pass-through."""
    filters: list[dict] = []
    items = desc_list(desc, "filterFXList") or []
    for item in items:
        if not isinstance(item, Desc):
            continue
        name = desc_text(item, "Nm  ") or ""
        blend_desc = desc_desc(item, "blendOptions")
        opacity = 1.0
        blend = "norm"
        if blend_desc is not None:
            value = _px(blend_desc, "Opct")
            if value is not None:
                opacity = min(max(value / 100.0, 0.0), 1.0)
            mode = desc_enum(blend_desc, "Md  ")
            if mode is not None:
                blend = mode[1]
        fltr = desc_desc(item, "Fltr")
        cls = fltr.class_id if fltr is not None else ""
        if not cls:
            value = item.get("filterID")
            if isinstance(value, int):
                cls = value.to_bytes(4, "big",
                                     errors="ignore").decode(
                    "latin-1", errors="ignore")
        filters.append({
            "name": name or FILTER_NAMES.get(cls, cls),
            "class": cls,
            "blend": blend,
            "opacity": opacity,
            "visible": (desc_bool(item, "enab") if desc_bool(item, "enab") is not None else True),
            "params": _filter_params(cls, fltr, item),
        })
    return {
        "enabled": (desc_bool(desc, "enab")
                    if desc_bool(desc, "enab") is not None
                    else True),
        "filters": filters,
        "mask_enabled": (desc_bool(desc, "filterMaskEnable")
                         if desc_bool(desc, "filterMaskEnable")
                         is not None else True),
        "mask_linked": (desc_bool(desc, "filterMaskLinked")
                        if desc_bool(desc, "filterMaskLinked")
                        is not None else False),
    }


def parse_sold(data: bytes) -> dict | None:
    """A ``SoLd``/``SoLE`` block -> the placed record (idnt,
    placed id, source size, transform affine, filter stack),
    or None when it is not a ``soLD`` structure."""
    if len(data) < 8 or data[:4] != b"soLD":
        return None
    try:
        desc, _ = parse_versioned(data, 8)
    except Exception:
        return None
    idnt = desc_text(desc, "Idnt") or ""
    placed = desc_text(desc, "placed") or ""
    size: tuple[float, float] | None = None
    sz = desc_desc(desc, "Sz  ")
    if sz is not None:
        width = desc_num(sz, "Wdth")
        height = desc_num(sz, "Hght")
        if width is not None and height is not None \
                and width > 0 and height > 0:
            size = (width, height)
    transform: list[float] | None = None
    trnf = desc_list(desc, "Trnf")
    if trnf is not None and size is not None:
        corners = [float(v) for v in trnf
                   if isinstance(v, (int, float))]
        transform = _corners_affine(corners, size[0], size[1])
    stack: dict | None = None
    fx = desc_desc(desc, "filterFX")
    if fx is not None:
        stack = parse_filter_stack(fx)
    return {"idnt": idnt, "placed": placed, "size": size,
            "transform": transform, "stack": stack,
            "descriptor": desc}


def parse_plld(data: bytes) -> dict | None:
    """A ``PlLd`` block -> the linked record (id, the 8
    placed-corner doubles, warp style), or None when it is
    not a ``plcL`` record. The warp descriptor after the
    corners is parsed best-effort for its style only."""
    if len(data) < 9 or data[:4] != b"plcL":
        return None
    n = data[8]
    if len(data) < 9 + n + 16 + 64:
        return None
    try:
        ident = data[9:9 + n].decode("utf-8", errors="replace")
    except Exception:
        return None
    at = 9 + n + 16
    corners = [struct.unpack(">d", data[at + i * 8:at + i * 8 + 8])[0]
               for i in range(8)]
    if not all(math.isfinite(v) for v in corners):
        return None
    warp = "warpNone"
    rest = data[at + 64:]
    if rest and any(rest):
        try:
            desc, _ = parse_versioned(rest, 4)
            style = desc_enum(desc, "warpStyle")
            if style is not None:
                warp = style[1]
        except Exception:
            warp = "unreadable"
    return {"idnt": ident, "corners": corners,
            "warp": warp}


# ---------------------------------------------------------------------------
# FEid
# ---------------------------------------------------------------------------

def _read_pascal(data: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(data):
        raise ValueError("pascal string past the end")
    n = data[offset]
    end = offset + 1 + n
    if end > len(data):
        raise ValueError("pascal string is truncated")
    return (data[offset + 1:end].decode("utf-8",
                                        errors="replace"),
            end)


def _decode_plane(compression: int, data: bytes, width: int,
                  height: int, depth: int) -> bytes:
    """One FEid plane -> big-endian sample bytes. RLE uses
    four-byte row counts (the Psb layout)."""
    from .psd import _row_bytes, _unpredict, packbits_decode
    rb = _row_bytes(width, depth)
    total = rb * height
    if compression == 0:
        if len(data) < total:
            raise ValueError("raw FEid plane is short")
        return data[:total]
    if compression == 1:
        counts = struct.unpack(f">{height}I",
                               data[:height * 4])
        pos = height * 4
        out = bytearray()
        for count in counts:
            if pos + count > len(data):
                raise ValueError("RLE plane runs past its data")
            out.extend(packbits_decode(data[pos:pos + count],
                                       rb))
            pos += count
        if len(out) != total:
            raise ValueError("RLE plane decoded short")
        return bytes(out)
    if compression in (2, 3):
        raw = zlib.decompress(data)
        if len(raw) != total:
            raise ValueError("zlib plane decoded short")
        if compression == 3:
            buf = bytearray(raw)
            _unpredict(buf, width, depth)
            return bytes(buf)
        return raw
    raise ValueError(f"unsupported FEid compression "
                     f"{compression}")


def parse_feid(data: bytes, mode: str = "RGB",
               depth: int = 8) -> list[dict]:
    """An ``FEid`` block -> the items (placed id, rect,
    unfiltered RGBA float array, filter-mask array when
    present). Colour planes come from slot 0, transparency
    from the last slot; without a transparency plane the
    pixels are opaque."""
    if len(data) < 4:
        return []
    version = struct.unpack(">I", data[:4])[0]
    if version < 1 or version > 3:
        return []
    channels = {"RGB": 3, "CMYK": 4, "Lab": 3,
                "Grayscale": 1}.get(mode, 3)
    items: list[dict] = []
    pos = 4
    while pos + 8 <= len(data):
        section_len = struct.unpack(">Q", data[pos:pos + 8])[0]
        body = data[pos + 8:pos + 8 + section_len]
        if len(body) < section_len:
            break
        inner = 0
        while inner < len(body):
            try:
                ident, inner = _read_pascal(body, inner)
                item_version = struct.unpack(
                    ">I", body[inner:inner + 4])[0]
                inner += 4
                item_len = struct.unpack(
                    ">Q", body[inner:inner + 8])[0]
                inner += 8
                it = body[inner:inner + item_len]
                inner += item_len
                if len(it) < 24:
                    continue
                top, left, bottom, right = struct.unpack(
                    ">4i", it[:16])
                item_depth, max_channels = struct.unpack(
                    ">2I", it[16:24])
                width = right - left
                height = bottom - top
                if width <= 0 or height <= 0:
                    continue
                slots: list[bytes | None] = []
                p = 24
                for _ in range(max_channels + 2):
                    if p + 4 > len(it):
                        break
                    present = struct.unpack(
                        ">I", it[p:p + 4])[0]
                    p += 4
                    if not present:
                        slots.append(None)
                        continue
                    if p + 10 > len(it):
                        break
                    plane_len = struct.unpack(
                        ">Q", it[p:p + 8])[0]
                    comp = struct.unpack(
                        ">H", it[p + 8:p + 10])[0]
                    p += 10
                    raw = it[p:p + plane_len - 2]
                    p += plane_len - 2
                    try:
                        slots.append(_decode_plane(
                            comp, raw, width, height,
                            item_depth))
                    except Exception:
                        slots.append(None)
                mask: np.ndarray | None = None
                if inner < len(body) and body[inner] != 0:
                    inner += 1
                    if inner + 16 <= len(body):
                        mtop, mleft, mbot, mright = \
                            struct.unpack(">4i",
                                        body[inner:inner + 16])
                        inner += 16
                        if inner + 10 <= len(body):
                            plane_len = struct.unpack(
                                ">Q",
                                body[inner:inner + 8])[0]
                            comp = struct.unpack(
                                ">H",
                                body[inner + 8:inner + 10])[0]
                            inner += 10
                            raw = body[inner:inner + plane_len - 2]
                            inner += plane_len - 2
                            try:
                                decoded = _decode_plane(
                                    comp, raw, mright - mleft,
                                    mbot - mtop, item_depth)
                                arr = np.frombuffer(
                                    decoded,
                                    dtype=">u1" if item_depth == 8
                                    else ">u2")
                                scale = 255.0 if item_depth == 8 \
                                    else 65535.0
                                mask = (arr.astype(np.float64)
                                        / scale).reshape(
                                    (mbot - mtop, mright - mleft))
                            except Exception:
                                mask = None
                elif inner < len(body):
                    inner += 1
                planes: list[np.ndarray | None] = []
                for slot in range(channels):
                    raw = slots[slot] if slot < len(slots) \
                        else None
                    if raw is None:
                        planes.append(None)
                        continue
                    if item_depth == 8:
                        arr = np.frombuffer(raw, dtype=">u1")
                        planes.append(arr.astype(np.float64)
                                      / 255.0)
                    elif item_depth == 16:
                        arr = np.frombuffer(raw, dtype=">u2")
                        planes.append(arr.astype(np.float64)
                                      / 65535.0)
                    else:
                        arr = np.frombuffer(raw, dtype=">f4")
                        planes.append(arr.astype(np.float64))
                if any(p is None for p in planes):
                    continue
                rgba = np.zeros((height, width, 4),
                                dtype=np.float64)
                for c in range(channels):
                    rgba[..., c] = planes[c].reshape(
                        (height, width))
                alpha_raw = slots[-1] if slots else None
                if alpha_raw is not None:
                    if item_depth == 8:
                        arr = np.frombuffer(alpha_raw,
                                            dtype=">u1")
                        rgba[..., 3] = (arr.astype(np.float64)
                                        / 255.0).reshape(
                            (height, width))
                    elif item_depth == 16:
                        arr = np.frombuffer(alpha_raw,
                                            dtype=">u2")
                        rgba[..., 3] = (arr.astype(np.float64)
                                        / 65535.0).reshape(
                            (height, width))
                    else:
                        arr = np.frombuffer(alpha_raw,
                                            dtype=">f4")
                        rgba[..., 3] = arr.astype(
                            np.float64).reshape((height, width))
                else:
                    rgba[..., 3] = 1.0
                if mode == "CMYK":
                    rgba[..., :3] = 1.0 - rgba[..., :3]
                items.append({
                    "id": ident,
                    "version": item_version,
                    "rect": (top, left, bottom, right),
                    "depth": item_depth,
                    "rgba": rgba,
                    "mask": mask,
                    "mask_rect": ((mtop, mleft, mbot, mright)
                                  if mask is not None else None),
                })
            except Exception:
                break
        pos += 8 + section_len
        pos += (4 - section_len % 4) % 4
        if pos >= len(data):
            break
        if data[pos:] == b"\x00" * len(data[pos:]):
            break
    return items


# ---------------------------------------------------------------------------
# Edit-path filter application (never used by the compositor)
# ---------------------------------------------------------------------------

def _to_pil(rgba: np.ndarray) -> Image.Image:
    arr = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _from_pil(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"),
                      dtype=np.float64) / 255.0


def _motion_blur(rgba: np.ndarray, angle: float,
                 distance: float) -> np.ndarray:
    """A straight-line blur: the mean of shifted copies
    along ``angle`` degrees, over ``distance`` pixels."""
    steps = max(1, int(round(distance)))
    theta = math.radians(angle)
    dx = math.cos(theta)
    dy = math.sin(theta)
    acc = np.zeros_like(rgba)
    for i in range(steps):
        shift = (i - (steps - 1) / 2.0)
        ox = int(round(shift * dx))
        oy = int(round(shift * dy))
        shifted = np.roll(np.roll(rgba, oy, axis=0), ox,
                          axis=1)
        acc += shifted
    return acc / steps


def _premultiply(rgba: np.ndarray) -> np.ndarray:
    """Straight -> premultiplied alpha (Photoshop filters
    work in premultiplied space; transparent pixels carry
    white, as in its own cache)."""
    out = rgba.copy()
    out[..., :3] *= rgba[..., 3:4]
    return out


def _unpremultiply(pre: np.ndarray) -> np.ndarray:
    """Premultiplied -> straight alpha. Fully transparent
    pixels come back white, as in Photoshop's cache."""
    out = pre.copy()
    alpha = pre[..., 3:4]
    out[..., :3] = np.where(alpha > 1e-6,
                            pre[..., :3] / np.maximum(alpha, 1e-6),
                            1.0)
    return np.clip(out, 0.0, 1.0)


def _gaussian_blur(rgba: np.ndarray, radius: float
                   ) -> np.ndarray:
    """A true Gaussian blur in premultiplied space: scipy
    when importable, PIL's three-box approximation when
    not."""
    pre = _premultiply(rgba)
    try:
        from scipy.ndimage import gaussian_filter
        blurred = np.stack(
            [gaussian_filter(pre[..., c], radius)
             for c in range(4)], axis=-1)
    except ImportError:
        arr = np.clip(pre * 255.0, 0, 255).astype(np.uint8)
        blurred = np.asarray(
            Image.fromarray(arr, mode="RGBA").filter(
                ImageFilter.GaussianBlur(radius)),
            dtype=np.float64) / 255.0
    return _unpremultiply(blurred)


def _pil_filter(rgba: np.ndarray, filt) -> np.ndarray:
    """A PIL filter in premultiplied space."""
    pre = _premultiply(rgba)
    arr = np.clip(pre * 255.0, 0, 255).astype(np.uint8)
    out = np.asarray(Image.fromarray(arr, mode="RGBA").filter(
        filt), dtype=np.float64) / 255.0
    return _unpremultiply(out)


def apply_smart_filter(rgba: np.ndarray, filt: dict,
                       *, depth: int = 8) -> np.ndarray:
    """One smart filter over float RGBA (premultiplied
    straight alpha in, same out). Unknown or
    descriptor-form filters pass through unchanged — the
    name is kept, the pixels are not invented."""
    cls = filt.get("class", "")
    params = filt.get("params", {}) or {}
    if cls == "GsnB":
        radius = float(params.get("radius", 0.0))
        if radius <= 0:
            return rgba
        return _gaussian_blur(rgba, radius)
    if cls == "boxblur":
        radius = float(params.get("radius", 0.0))
        if radius <= 0:
            return rgba
        return _pil_filter(rgba, ImageFilter.BoxBlur(radius))
    if cls == "MtnB":
        distance = float(params.get("distance", 0.0))
        if distance <= 0:
            return rgba
        return _unpremultiply(_motion_blur(_premultiply(rgba),
                                           float(params.get(
                                               "angle", 0)),
                                           distance))
    if cls == "UnsM":
        amount = float(params.get("amount", 0.0))
        radius = float(params.get("radius", 0.0))
        threshold = int(params.get("threshold", 0))
        if amount <= 0 or radius <= 0:
            return rgba
        return _pil_filter(rgba, ImageFilter.UnsharpMask(
            radius=radius, percent=int(amount),
            threshold=threshold))
    if cls == "Mdn ":
        radius = float(params.get("radius", 0.0))
        if radius <= 0:
            return rgba
        size = max(3, int(round(radius)) * 2 + 1)
        return _pil_filter(rgba, ImageFilter.MedianFilter(
            size=size))
    if cls == "Mxm ":
        radius = float(params.get("radius", 0.0))
        if radius <= 0:
            return rgba
        size = max(3, int(round(radius)) * 2 + 1)
        return _pil_filter(rgba, ImageFilter.MaxFilter(size=size))
    if cls == "Mnm ":
        radius = float(params.get("radius", 0.0))
        if radius <= 0:
            return rgba
        size = max(3, int(round(radius)) * 2 + 1)
        return _pil_filter(rgba, ImageFilter.MinFilter(size=size))
    if cls == "Msc ":
        size = max(1, int(round(params.get("cell_size", 1.0))))
        if size <= 1:
            return rgba
        pre = _premultiply(rgba)
        img = Image.fromarray(
            np.clip(pre * 255.0, 0, 255).astype(np.uint8),
            mode="RGBA")
        small = img.resize((max(1, img.width // size),
                            max(1, img.height // size)),
                           Image.BOX)
        return _unpremultiply(_from_pil(
            small.resize(img.size, Image.NEAREST)))
    if cls == "HghP":
        radius = float(params.get("radius", 0.0))
        if radius <= 0:
            return rgba
        blurred = _gaussian_blur(rgba, radius)
        out = rgba - blurred + 0.5
        out[..., 3] = rgba[..., 3]
        return np.clip(out, 0.0, 1.0)
    if cls == "Embs":
        amount = float(params.get("amount", 100.0)) / 100.0
        if amount <= 0:
            return rgba
        embossed = _pil_filter(rgba, ImageFilter.EMBOSS)
        out = rgba * (1.0 - amount) + embossed * amount
        out[..., 3] = rgba[..., 3]
        return np.clip(out, 0.0, 1.0)
    if cls == "AdNs":
        amount = float(params.get("amount", 0.0)) / 100.0
        if amount <= 0:
            return rgba
        rng = np.random.default_rng(0)
        if params.get("distribution") == "Gaussian":
            noise = rng.standard_normal(rgba.shape[:2])
        else:
            noise = rng.uniform(-1.0, 1.0, rgba.shape[:2])
        out = rgba.copy()
        out[..., :3] = np.clip(
            rgba[..., :3] + noise[..., None] * amount, 0.0, 1.0)
        return out
    if cls == "Blr ":
        return _pil_filter(rgba, ImageFilter.BLUR)
    if cls == "BlrM":
        return _pil_filter(_pil_filter(rgba, ImageFilter.BLUR),
                           ImageFilter.BLUR)
    if cls == "Avrg":
        out = rgba.copy()
        opaque = rgba[..., 3] > 0
        if opaque.any():
            mean = rgba[opaque][..., :3].mean(axis=0)
            out[..., :3] = mean
        return out
    if cls == "Dspc":
        return _pil_filter(rgba, ImageFilter.MedianFilter(
            size=3))
    if cls == "FndE":
        return _pil_filter(rgba, ImageFilter.FIND_EDGES)
    if cls == "Lvls":
        return rgba
    if cls == "Crvs":
        return rgba
    return rgba


def apply_filter_stack(rgba: np.ndarray, stack: dict, *,
                       depth: int = 8,
                       mask: np.ndarray | None = None
                       ) -> np.ndarray:
    """A whole filter stack over float RGBA: each visible
    filter renders, then mixes with the input in
    premultiplied space with its own blend mode, opacity
    and mask. Disabled stacks and empty stacks pass
    through."""
    if not stack or not stack.get("enabled", True):
        return rgba
    current = rgba
    for filt in stack.get("filters", []):
        if not filt.get("visible", True):
            continue
        rendered = apply_smart_filter(current, filt,
                                      depth=depth)
        if rendered is current:
            continue
        blend = filt.get("blend", "norm")
        opacity = float(filt.get("opacity", 1.0))
        # The filter's blend mixes the filtered pixels with
        # the unfiltered input in premultiplied space — a
        # normal blend at 100% returns the filtered pixels
        # verbatim (a layer composite would double-count the
        # backdrop at semi-transparent edges).
        if blend != "norm" and blend != "normal":
            top = rendered.copy()
            top[..., :3] = blend_rgb(
                blend, current[..., :3], rendered[..., :3])
            rendered = top
        weight = opacity
        if mask is not None:
            m = mask if mask.ndim == 3 else mask[..., None]
            weight = m * opacity
        base_pre = _premultiply(current)
        top_pre = _premultiply(rendered)
        mixed = base_pre * (1.0 - weight) + top_pre * weight
        current = _unpremultiply(mixed)
    return current
