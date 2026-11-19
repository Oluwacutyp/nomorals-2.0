"""Fill layers — solid colour, gradient and pattern fills
(``SoCo`` / ``GdFl`` / ``PtFl``).

A fill layer's pixels are not stored: the layer carries a
versioned descriptor with the fill, and the compositor renders
it. The corpus has no fill layers at all, so this is built
from the documented layout and tested synthetically: a solid
colour is ``Clr `` in the descriptor, a gradient is ``Grad``
(a ``GrdF`` descriptor with ``Clrs`` / ``Trns`` lists, baked
like every other gradient), a pattern is ``Ptrn`` (name + id
— the pattern pixels themselves live in a ``Patt`` resource
the file may or may not carry, so a pattern fill renders only
when the document supplies its pixels, and says so otherwise).
"""

from __future__ import annotations

import numpy as np

from .descriptor import (
    Desc,
    color_from_desc,
    desc_bool,
    desc_desc,
    desc_enum,
    desc_list,
    desc_num,
    desc_text,
    parse_versioned,
)
from .gradient import (
    bake_opacity_stops,
    bake_stops,
    method_from_code,
    render_gradient,
)

__all__ = ["FILL_KEYS", "parse_fill_layer", "render_fill"]

#: Tagged-block keys that mark a fill layer (solid color,
#: gradient, pattern — no pixel channels of its own).
FILL_KEYS: dict[str, str] = {
    "SoCo": "solid color fill",
    "GdFl": "gradient fill",
    "PtFl": "pattern fill",
}


def _num(desc: Desc, key: str, default: float = 0.0) -> float:
    value = desc_num(desc, key)
    return default if value is None else value


def _pct(desc: Desc, key: str, default: float) -> float:
    return _num(desc, key, default * 100.0) / 100.0


def _bool(desc: Desc, key: str, default: bool = False) -> bool:
    value = desc_bool(desc, key)
    return default if value is None else value


def _text(desc: Desc, key: str) -> str:
    value = desc_text(desc, key)
    return value if value is not None else ""


def _enum(desc: Desc, key: str) -> str:
    pair = desc_enum(desc, key)
    return pair[1] if pair is not None else ""


def gradient_from_desc(desc: Desc) -> dict:
    """A gradient dict from a descriptor carrying a ``Grad``
    sub-descriptor (fill layers and gradient effects share it)."""
    grad = desc_desc(desc, "Grad")
    stops: list[tuple[float, tuple[float, float, float]]] = []
    mids: list[float] = []
    opacity: list[tuple[float, float]] = []
    if grad is not None:
        raw = desc_list(grad, "Clrs")
        if raw is not None:
            for item in raw:
                if not isinstance(item, Desc):
                    continue
                loc = _num(item, "Lctn", 0.0) / 4096.0
                colour = desc_desc(item, "Clr ")
                if colour is None:
                    continue
                rgb = color_from_desc(colour)
                if rgb is None:
                    continue
                stops.append((loc, rgb))
                mids.append(_num(item, "Mdpn", 50.0) / 100.0)
        raw = desc_list(grad, "Trns")
        if raw is not None:
            for item in raw:
                if not isinstance(item, Desc):
                    continue
                loc = _num(item, "Lctn", 0.0) / 4096.0
                opacity.append((loc, _num(item, "Opct", 100.0)
                                / 100.0))
    smooth = _num(grad, "Intr", 0.0) / 4096.0 if grad is not None \
        else 0.0
    method = method_from_code(
        _enum(desc, "gradientsInterpolationMethod")
        or _enum(desc, "gs99") or None)
    # The midpoint list belongs to the segments between the
    # sorted stops (midpoint k shapes the segment after stop k).
    ordered = sorted(range(len(stops)),
                     key=lambda i: stops[i][0])
    seg_mids = [mids[i] for i in ordered[1:]]
    baked = bake_stops(stops, seg_mids, smooth, method)
    baked_opacity = bake_opacity_stops(opacity, seg_mids, smooth,
                                       method)
    style = {"Rdl ": "radial", "Angl": "angle",
             "Rflc": "reflected", "Dmnd": "diamond"}.get(
                 _enum(desc, "Type"), "linear")
    offset_desc = desc_desc(desc, "Ofst")
    offset = (0.0, 0.0)
    if offset_desc is not None:
        offset = (_pct(offset_desc, "Hrzn", 0.0),
                  _pct(offset_desc, "Vrtc", 0.0))
    return {
        "stops": baked,
        "opacity_stops": baked_opacity,
        "style": style,
        "angle": _num(desc, "Angl", 90.0),
        "scale": _pct(desc, "Scl ", 1.0),
        "reverse": _bool(desc, "Rvrs"),
        "align": desc.get("Algn") is not False,
        "offset": offset,
    }


def parse_fill_layer(key: str, data: bytes) -> dict | None:
    """A fill layer's descriptor -> a fill spec dict, or None
    when the block cannot be parsed."""
    try:
        desc, _ = parse_versioned(data)
    except Exception:
        return None
    if key == "SoCo":
        colour = desc_desc(desc, "Clr ")
        rgb = color_from_desc(colour) if colour is not None \
            else None
        if rgb is None:
            return None
        return {"kind": "solid", "rgb": rgb}
    if key == "GdFl":
        gradient = gradient_from_desc(desc)
        if not gradient["stops"]:
            return None
        return {"kind": "gradient", "gradient": gradient}
    if key == "PtFl":
        pattern = desc_desc(desc, "Ptrn")
        offset_desc = desc_desc(desc, "Ofst")
        return {
            "kind": "pattern",
            "name": _text(pattern, "Nm  ") if pattern is not None
            else "",
            "id": _text(pattern, "Idnt") if pattern is not None
            else "",
            "scale": _pct(desc, "Scl ", 1.0),
            "linked": _bool(desc, "Lnkd"),
            "offset": (_pct(offset_desc, "Hrzn", 0.0)
                       if offset_desc is not None else 0.0,
                       _pct(offset_desc, "Vrtc", 0.0)
                       if offset_desc is not None else 0.0),
        }
    return None


def render_fill(spec: dict, width: int, height: int,
                *, pattern_pixels: dict | None = None
                ) -> np.ndarray | None:
    """A fill spec rendered as float RGBA over ``width``x
    ``height`` (``pattern_pixels`` maps a pattern id to a
    float RGBA tile for pattern fills)."""
    kind = spec.get("kind")
    if kind == "solid":
        rgb = spec["rgb"]
        out = np.zeros((height, width, 4), dtype=np.float64)
        out[..., 0] = rgb[0]
        out[..., 1] = rgb[1]
        out[..., 2] = rgb[2]
        out[..., 3] = 1.0
        return out
    if kind == "gradient":
        return render_gradient(spec["gradient"], width, height)
    if kind == "pattern":
        if not pattern_pixels:
            return None
        tile = pattern_pixels.get(spec.get("id", ""))
        if tile is None:
            return None
        scale = max(spec.get("scale", 1.0), 1e-3)
        th, tw = tile.shape[:2]
        tw = max(1, int(round(tw * scale)))
        th = max(1, int(round(th * scale)))
        # Nearest-neighbour scale of the tile, then tile it.
        ys = (np.arange(height) % th)
        xs = (np.arange(width) % tw)
        src_y = (ys * tile.shape[0] // th)
        src_x = (xs * tile.shape[1] // tw)
        return tile[np.ix_(src_y, src_x)]
    return None
