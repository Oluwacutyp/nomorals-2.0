"""Render + export — what a staged point cloud becomes on disk.

Two real deliverables, no GPU needed:

* :func:`write_ply` — the frame as an ASCII PLY file (``x y z``
  plus ``red green blue alpha``), the format every 3D tool
  reads. The absolute path is returned, recorded at write time.
* :func:`render_projection` — the frame as a PNG scatter
  projection (top view by default, front on request), points
  depth-shaded by the dropped axis. Pillow is required and
  refused by name when it is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import CloudCamError
from .protocol import CloudCamPoint

__all__ = ["render_projection", "write_ply"]


def write_ply(points: list[CloudCamPoint], path: str | Path
              ) -> dict[str, Any]:
    """Write points as ASCII PLY 1.0. Returns the absolute path
    and the point count."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar alpha",
        "end_header",
    ]
    for p in points:
        lines.append(f"{p.x} {p.y} {p.z} {p.r} {p.g} {p.b} {p.a}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(target), "points": len(points)}


def render_projection(points: list[CloudCamPoint], path: str | Path,
                      *, width: int = 640, height: int = 360,
                      view: str = "top") -> dict[str, Any]:
    """Scatter-project the frame to a PNG. ``top`` plots x→right,
    y→down, shading by z; ``front`` plots x→right, z→up, shading
    by y. An empty frame renders the axes and says so — a blank
    image is never passed off as a render."""
    if view not in ("top", "front"):
        raise CloudCamError(
            f"projection view must be 'top' or 'front', got {view!r}",
            kind="bad_view")
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise CloudCamError(
            "point-cloud rendering needs Pillow — it is not "
            "installed on this machine (pip install Pillow)",
            kind="missing_pil") from exc
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (int(width), int(height)), (12, 12, 16))
    draw = ImageDraw.Draw(image)
    draw.line([(0, int(height) // 2), (int(width), int(height) // 2)],
              fill=(48, 48, 64))
    draw.line([(int(width) // 2, 0), (int(width) // 2, int(height))],
              fill=(48, 48, 64))
    plotted = 0
    if points:
        if view == "top":
            us = [p.x for p in points]
            vs = [p.y for p in points]
            ds = [p.z for p in points]
        else:
            us = [p.x for p in points]
            vs = [-p.z for p in points]
            ds = [p.y for p in points]
        u0, u1 = min(us), max(us)
        v0, v1 = min(vs), max(vs)
        d0, d1 = min(ds), max(ds)
        uspan = (u1 - u0) or 1.0
        vspan = (v1 - v0) or 1.0
        dspan = (d1 - d0) or 1.0
        pad = 16
        for _point, u, v, d in zip(points, us, vs, ds, strict=True):
            px = pad + (u - u0) / uspan * (int(width) - 2 * pad)
            py = pad + (v - v0) / vspan * (int(height) - 2 * pad)
            t = (d - d0) / dspan
            colour = (int(round(t * 255)), 64,
                      int(round((1.0 - t) * 255)))
            r = 2 if len(points) < 20000 else 1
            draw.ellipse([px - r, py - r, px + r, py + r],
                         fill=colour)
            plotted += 1
    image.save(target)
    return {"path": str(target), "view": view,
            "width": int(width), "height": int(height),
            "plotted": plotted, "points": len(points),
            "empty": len(points) == 0}
