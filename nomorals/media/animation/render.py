"""Frame renderer — port of AniCraft ``crates/render/src/raster.rs``.

Renders a :class:`~nomorals.media.animation.eval.DisplayList` to a
PIL image: solid fills, linear/radial gradients (via numpy), strokes,
text (DejaVu), bitmap fills, mask clips (coverage mode: the mask's
fill area reveals; alpha/luminance modes are named in the report when
a clip uses them — this port implements coverage only, the others are
reported, not faked), and onion skinning (ghost frames around the
playhead at reduced alpha). Anything not drawn is named in
``report["unrendered"]``, never silently dropped.
"""

from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .eval import DisplayItem, DisplayList, evaluate
from .model import AnimationDocument, Rgba

__all__ = ["render_display_list", "render_frame"]


def _font(size: float) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            max(1, int(size)))
    except OSError:
        return ImageFont.load_default()


def _fill_image(size: tuple[int, int], fill: dict[str, Any],
                bounds: tuple[float, float, float, float]) -> Image.Image:
    """A fill painted into an image the size of the stage."""
    import numpy as np

    w, h = size
    kind = fill.get("kind", "solid")
    if kind == "solid":
        c = Rgba.from_dict(fill.get("color", {}))
        return Image.new("RGBA", (w, h), c.to_tuple255())
    if kind in ("linear", "radial"):
        stops = [(float(t), Rgba.from_dict(c))
                 for t, c in fill.get("stops", [])]
        if not stops:
            return Image.new("RGBA", (w, h), (0, 0, 0, 0))
        if len(stops) == 1:
            return Image.new("RGBA", (w, h), stops[0][1].to_tuple255())
        x0, y0, x1, y1 = bounds
        xs = np.linspace(0.0, 1.0, w)
        ys = np.linspace(0.0, 1.0, h)
        if kind == "linear":
            angle = math.radians(float(fill.get("angle", 0.0)))
            dx, dy = math.cos(angle), math.sin(angle)
            xx, yy = np.meshgrid(xs, ys)
            t = (xx * dx + yy * dy - min(0.0, dx) - min(0.0, dy)) / (
                abs(dx) + abs(dy) + 1e-9)
        else:
            cx, cy = 0.5 * (x0 + x1) / max(1, w), 0.5 * (y0 + y1) / max(1, h)
            xx, yy = np.meshgrid(xs, ys)
            t = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            t = t / max(t.max(), 1e-9)
        t = np.clip(t, 0.0, 1.0)
        out = np.zeros((h, w, 4), dtype=np.uint8)
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            mask = (t >= t0) & (t <= t1)
            if not mask.any():
                continue
            local = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
            for ch, v0, v1 in ((0, c0.r, c1.r), (1, c0.g, c1.g),
                               (2, c0.b, c1.b), (3, c0.a, c1.a)):
                out[..., ch] = np.where(
                    mask, (v0 + (v1 - v0) * local) * 255,
                    out[..., ch]).astype(np.uint8)
        # Regions before the first stop / after the last stop.
        first = stops[0][1].to_tuple255()
        last = stops[-1][1].to_tuple255()
        for ch in range(4):
            out[..., ch] = np.where(t < stops[0][0], first[ch],
                                    out[..., ch]).astype(np.uint8)
            out[..., ch] = np.where(t > stops[-1][0], last[ch],
                                    out[..., ch]).astype(np.uint8)
        return Image.fromarray(out, "RGBA")
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


def _draw_shape(base: Image.Image, item: DisplayItem,
                doc: AnimationDocument | None) -> str | None:
    """Draw one display item; returns an unrendered reason or None."""
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = max(0.0, min(1.0, item.opacity))
    if item.kind == "text":
        if not item.points or not item.points[0]:
            return None
        x, y = item.points[0][0]
        color = Rgba.from_hex(item.text_color)
        draw.text((x, y), item.text, font=_font(item.font_size),
                  fill=(round(color.r * 255), round(color.g * 255),
                        round(color.b * 255),
                        round(color.a * 255 * alpha)))
        base.alpha_composite(overlay)
        return None
    if item.kind == "bitmap":
        if doc is None or item.asset not in doc.assets:
            return (f"bitmap {item.asset!r}: asset not in the document, "
                    "not drawn")
        try:
            import io as _io

            from PIL import Image as _Image
            bmp = _Image.open(_io.BytesIO(doc.assets[item.asset]))
            bmp = bmp.convert("RGBA").resize(
                (max(1, int(item.width)), max(1, int(item.height))))
        except Exception as exc:
            return f"bitmap {item.asset!r}: cannot decode ({exc})"
        if not item.points or not item.points[0]:
            return None
        x, y = item.points[0][0]
        if alpha < 1.0:
            r, g, b, a = bmp.split()
            bmp.putalpha(a.point(lambda v: int(v * alpha)))
        base.alpha_composite(bmp, (int(x), int(y)))
        return None
    # shape
    if not item.points:
        return None
    bounds = (min(p[0] for c in item.points for p in c),
              min(p[1] for c in item.points for p in c),
              max(p[0] for c in item.points for p in c),
              max(p[1] for c in item.points for p in c))
    if item.fill is not None:
        kind = item.fill.get("kind", "solid")
        if kind == "bitmap":
            asset = item.fill.get("asset", "")
            if doc is None or asset not in doc.assets:
                return (f"bitmap fill {asset!r}: asset not in the "
                        "document, not drawn")
            try:
                import io as _io

                from PIL import Image as _Image
                tile = _Image.open(_io.BytesIO(doc.assets[asset]))
                tile = tile.convert("RGBA").resize(base.size)
            except Exception as exc:
                return f"bitmap fill {asset!r}: cannot decode ({exc})"
            fill_img = tile
        else:
            fill_img = _fill_image(base.size, item.fill, bounds)
        if alpha < 1.0:
            r, g, b, a = fill_img.split()
            fill_img.putalpha(a.point(lambda v: int(v * alpha)))
        mask = Image.new("L", base.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        for contour in item.points:
            mask_draw.polygon(contour, fill=255)
        base.paste(fill_img, (0, 0), mask)
    if item.stroke is not None:
        color = Rgba.from_dict(item.stroke.get("color", {}))
        width = max(1, int(item.stroke.get("width", 1.0)))
        stroke_col = (round(color.r * 255), round(color.g * 255),
                      round(color.b * 255),
                      round(color.a * 255 * alpha))
        for contour in item.points:
            pts = contour + [contour[0]] if len(contour) > 1 else contour
            draw.line(pts, fill=stroke_col, width=width, joint="curve")
        base.alpha_composite(overlay)
    return None


def render_display_list(doc: AnimationDocument | None, dl: DisplayList,
                        width: int, height: int,
                        background: Rgba | None = None) -> tuple[Image.Image, dict[str, Any]]:
    """Render an evaluated display list to a PIL image."""
    bg = background or Rgba(1.0, 1.0, 1.0, 1.0)
    base = Image.new("RGBA", (width, height), bg.to_tuple255())
    unrendered: list[str] = []
    # Group items by clip; unclipped items draw straight onto base.
    clipped: dict[int, list[DisplayItem]] = {}
    for item in dl.items:
        if item.clip is not None:
            clipped.setdefault(item.clip, []).append(item)
        else:
            reason = _draw_shape(base, item, doc)
            if reason:
                unrendered.append(reason)
    for clip_idx, clip_items in clipped.items():
        if clip_idx >= len(dl.clips):
            unrendered.append(f"clip {clip_idx}: no mask recorded")
            continue
        clip = dl.clips[clip_idx]
        if clip.get("mode") != "coverage":
            unrendered.append(
                f"clip {clip_idx}: mask mode {clip.get('mode')} is not "
                "rendered, only coverage is")
            continue
        layer_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for item in clip_items:
            reason = _draw_shape(layer_img, item, doc)
            if reason:
                unrendered.append(reason)
        mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        for contour in [c for shapes in clip["shapes"]
                        for c in shapes]:
            mask_draw.polygon(contour, fill=255)
        base.paste(layer_img, (0, 0), mask)
    return base.convert("RGB"), {"unrendered": unrendered,
                                 "items": len(dl.items),
                                 "clips": len(dl.clips)}


def render_frame(doc: AnimationDocument, frame: int,
                 scene: str | None = None, *,
                 onion: int = 0) -> tuple[Image.Image, dict[str, Any]]:
    """Render one frame; ``onion`` ghosts that many neighbours each side."""
    scene_obj = doc.scene(scene)
    dl = evaluate(doc, frame, scene_obj)
    img, report = render_display_list(
        doc, dl, scene_obj.stage.width, scene_obj.stage.height,
        background=scene_obj.stage.background)
    unrendered = list(report["unrendered"])
    if onion > 0:
        ghost = Image.new("RGBA", img.size, (0, 0, 0, 0))
        total = scene_obj.timeline.frame_count()
        for offset in range(1, onion + 1):
            for f in (frame - offset, frame + offset):
                if 0 <= f < max(1, total):
                    gdl = evaluate(doc, f, scene_obj)
                    gimg, greport = render_display_list(
                        doc, gdl, scene_obj.stage.width,
                        scene_obj.stage.height,
                        background=Rgba(0.0, 0.0, 0.0, 0.0))
                    gimg = gimg.convert("RGBA")
                    r, g, b, a = gimg.split()
                    strength = int(70 / offset)
                    gimg.putalpha(a.point(
                        lambda v, s=strength: min(255, v * s // 255)))
                    ghost.alpha_composite(gimg)
                    unrendered.extend(greport["unrendered"])
        base = img.convert("RGBA")
        base.alpha_composite(ghost)
        img = base.convert("RGB")
    report["unrendered"] = sorted(set(unrendered))
    report["notes"] = dl.notes
    report["frame"] = frame
    return img, report
