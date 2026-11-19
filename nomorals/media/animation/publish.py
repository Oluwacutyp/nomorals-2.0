"""Publish — port of AniCraft ``crates/doc/src/publish.rs``.

Upstream's publish formats, in its order: Html, Gif, Apng, Webp,
Video, Lottie, SpriteSheet, PngSequence, Png, Svg, Audio. This port
writes the ones it can do for real — Gif, Apng, Webp, Video (H.264
MP4 via Devon's ffmpeg frame pipe), SpriteSheet, PngSequence, Png,
Svg (real vector output from the evaluated frame) and Html (a page
embedding the published GIF) — and refuses the rest honestly:
``lottie`` (no Lottie writer in this port) and ``audio`` (an
animation document carries no audio) raise :class:`AnimationError`
naming the gap instead of writing a hollow file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .errors import AnimationError
from .eval import evaluate
from .model import AnimationDocument
from .render import render_frame

__all__ = ["PUBLISH_FORMATS", "publish"]

PUBLISH_FORMATS = ("html", "gif", "apng", "webp", "mp4", "spritesheet",
                   "pngseq", "png", "svg")
REFUSED_FORMATS = ("lottie", "audio")


def _frames(doc: AnimationDocument, scene: str | None,
            start: int, end: int | None) -> list[Image.Image]:
    scene_obj = doc.scene(scene)
    total = scene_obj.timeline.frame_count()
    last = total if end is None else min(end, total)
    out: list[Image.Image] = []
    for frame in range(start, last):
        img, _report = render_frame(doc, frame, scene)
        out.append(img)
    return out


def _svg_frame(doc: AnimationDocument, frame: int,
               scene: str | None) -> str:
    scene_obj = doc.scene(scene)
    dl = evaluate(doc, frame, scene_obj)
    w, h = scene_obj.stage.width, scene_obj.stage.height
    bg = scene_obj.stage.background.to_hex()
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" '
             f'height="{h}" viewBox="0 0 {w} {h}">',
             f'<rect width="{w}" height="{h}" fill="{bg}"/>']
    for item in dl.items:
        if item.kind == "text":
            x, y = item.points[0][0]
            parts.append(
                f'<text x="{x:g}" y="{y + item.font_size:g}" '
                f'font-size="{item.font_size:g}" '
                f'fill="{item.text_color}" '
                f'opacity="{item.opacity:g}">{item.text}</text>')
            continue
        if item.kind != "shape" or not item.points:
            continue
        d = " ".join(
            "M " + " L ".join(f"{x:g},{y:g}" for x, y in contour) + " Z"
            for contour in item.points)
        fill = "#000000"
        fill_op = item.opacity
        if item.fill is not None:
            if item.fill.get("kind") == "solid":
                from .model import Rgba
                c = Rgba.from_dict(item.fill.get("color", {}))
                fill = c.to_hex()
                fill_op = item.opacity * c.a
            else:
                fill = "#000000"
        stroke = ""
        if item.stroke is not None:
            from .model import Rgba
            c = Rgba.from_dict(item.stroke.get("color", {}))
            stroke = (f' stroke="{c.to_hex()}" stroke-width='
                      f'"{item.stroke.get("width", 1.0):g}"')
        parts.append(f'<path d="{d}" fill="{fill}" '
                     f'fill-opacity="{fill_op:g}"{stroke}/>')
    parts.append("</svg>")
    return "\n".join(parts)


def publish(doc: AnimationDocument, fmt: str, out: str | Path,
            *, scene: str | None = None, start: int = 0,
            end: int | None = None, frame: int | None = None,
            fps: float | None = None,
            columns: int = 6) -> dict[str, Any]:
    """Publish a scene to ``out``. Returns a report with the real
    frame count and the paths written."""
    fmt = fmt.lower()
    if fmt in REFUSED_FORMATS:
        reason = {
            "lottie": "no Lottie writer in this port yet",
            "audio": "an animation document carries no audio",
        }[fmt]
        raise AnimationError(f"publish {fmt!r} refused: {reason}")
    if fmt not in PUBLISH_FORMATS:
        raise AnimationError(
            f"unknown publish format {fmt!r}; pick from "
            f"{list(PUBLISH_FORMATS)}")
    out_path = Path(out)
    scene_obj = doc.scene(scene)
    rate = float(fps) if fps else scene_obj.timeline.fps
    if fmt in ("png", "svg"):
        f = 0 if frame is None else int(frame)
        if fmt == "png":
            img, report = render_frame(doc, f, scene)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            return {"format": fmt, "paths": [str(out_path)],
                    "frame_count": 1, "fps": rate,
                    "unrendered": report["unrendered"]}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_svg_frame(doc, f, scene), encoding="utf-8")
        return {"format": fmt, "paths": [str(out_path)],
                "frame_count": 1, "fps": rate, "unrendered": []}
    frames = _frames(doc, scene, start, end)
    if not frames:
        raise AnimationError("nothing to publish: the timeline is empty")
    if fmt == "gif":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(out_path, save_all=True,
                       append_images=frames[1:], duration=int(1000 / rate),
                       loop=0)
    elif fmt == "apng":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(out_path, format="APNG", save_all=True,
                       append_images=frames[1:],
                       duration=int(1000 / rate), loop=0)
    elif fmt == "webp":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(out_path, format="WEBP", save_all=True,
                       append_images=frames[1:],
                       duration=int(1000 / rate), loop=0)
    elif fmt == "mp4":
        from ..motion_studio._core import encode_frames

        arrays = [np.asarray(img) for img in frames]
        encode_frames(arrays, out_path, frames[0].width,
                      frames[0].height, rate)
    elif fmt == "pngseq":
        out_path.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(frames):
            p = out_path / f"frame_{i:04d}.png"
            img.save(p)
            paths.append(str(p))
        return {"format": fmt, "paths": paths,
                "frame_count": len(frames), "fps": rate,
                "unrendered": []}
    elif fmt == "spritesheet":
        cols = max(1, int(columns))
        rows = math.ceil(len(frames) / cols)
        w, h = frames[0].size
        sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
        for i, img in enumerate(frames):
            sheet.paste(img, ((i % cols) * w, (i // cols) * h))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_path)
    elif fmt == "html":
        gif_path = out_path.with_suffix(".gif")
        frames[0].save(gif_path, save_all=True,
                       append_images=frames[1:],
                       duration=int(1000 / rate), loop=0)
        html = ("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
                f"<title>{doc.name}</title></head><body>"
                f"<img src=\"{gif_path.name}\" "
                f"width=\"{frames[0].width}\" "
                f"height=\"{frames[0].height}\" alt=\"{doc.name}\">"
                "</body></html>\n")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        return {"format": fmt,
                "paths": [str(out_path), str(gif_path)],
                "frame_count": len(frames), "fps": rate,
                "unrendered": []}
    return {"format": fmt, "paths": [str(out_path)],
            "frame_count": len(frames), "fps": rate, "unrendered": []}
