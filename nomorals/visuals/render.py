"""Renderer ladder: one spec, three deliverables.

- ``live`` — a self-contained HTML page. Diagrams and overlays draw
  on an inline ``<canvas>`` with no server and no external URL
  (fully offline-capable); 3D scenes reuse the motion rungs'
  three.js page, the repo's real 3D rung.
- ``chat`` — a PNG snapshot for chat surfaces (Telegram photo).
  Diagrams render through the motion studio's deterministic
  ``frame_image`` at full build; overlays draw their markers on a
  neutral canvas. A 3D scene has no PNG rung — it refuses honestly
  and names the live page instead of faking a render.
- ``terminal`` — structured text for GodConsole.

Every file renderer writes a real file under the visuals output
dir: the PNG is what Devon sends as an image, the HTML page is
what she opens live or hands on, and the validated spec in the
record is what the code-video rung consumes next. Nothing here is
display-only.
"""

from __future__ import annotations

import html as _html
import json
import math
from pathlib import Path
from typing import Any

from ..core.failure_store import home_dir as _devon_home_dir
from ..core.ids import new_id
from .profile import pick_target
from .spec import SceneSpecError, validate_scene_spec

__all__ = [
    "default_output_dir",
    "render_live_html",
    "render_png",
    "render_scene",
    "render_text",
    "write_live_page",
]

_DEFAULT_SIZE = (1280, 720)
_BG = "#101018"


def default_output_dir() -> Path:
    """``<home>/visuals`` — where rendered scenes live by default."""
    return Path(_devon_home_dir()) / "visuals"


def _slug(title: str) -> str:
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "-" for ch in (title or ""))
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:40] or "scene"


def _file_name(title: str, ext: str) -> str:
    return f"{_slug(title)}-{new_id()[:8]}.{ext}"


# ── live HTML ────────────────────────────────────────────────────────


_CANVAS_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>body{{margin:0;background:#101018}}canvas{{display:block}}</style>
</head><body>
<canvas id="scene" width="{width}" height="{height}"></canvas>
<script>
const spec = {spec_json};
const cv = document.getElementById('scene');
const ctx = cv.getContext('2d');
const W = cv.width, H = cv.height;
ctx.fillStyle = '#101018'; ctx.fillRect(0, 0, W, H);
ctx.fillStyle = '#F8FAFC'; ctx.font = 'bold 28px sans-serif';
ctx.fillText(spec.title || 'scene', 24, 44);
function arrow(x1, y1, x2, y2, color) {{
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 14 * Math.cos(a - 0.4), y2 - 14 * Math.sin(a - 0.4));
  ctx.lineTo(x2 - 14 * Math.cos(a + 0.4), y2 - 14 * Math.sin(a + 0.4));
  ctx.closePath(); ctx.fill();
}}
if (spec.kind === 'diagram') {{
  const byId = {{}};
  for (const n of spec.nodes) byId[n.id] = n;
  for (const e of spec.edges) {{
    const a = byId[e.from], b = byId[e.to];
    if (!a || !b) continue;
    arrow(a.x * W, a.y * H, b.x * W, b.y * H, '#94A3B8');
    if (e.label) {{
      ctx.fillStyle = '#94A3B8'; ctx.font = '16px sans-serif';
      ctx.fillText(e.label, (a.x + b.x) / 2 * W + 6, (a.y + b.y) / 2 * H - 6);
    }}
  }}
  for (const n of spec.nodes) {{
    const x = n.x * W, y = n.y * H;
    ctx.strokeStyle = n.color; ctx.fillStyle = n.color + '33'; ctx.lineWidth = 3;
    if (n.shape === 'circle') {{
      ctx.beginPath(); ctx.arc(x, y, 52, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }} else {{
      ctx.beginPath(); ctx.roundRect(x - 90, y - 34, 180, 68, 10); ctx.fill(); ctx.stroke();
    }}
    ctx.fillStyle = '#F8FAFC'; ctx.font = 'bold 18px sans-serif';
    ctx.textAlign = 'center'; ctx.fillText(n.label, x, y + 6); ctx.textAlign = 'left';
  }}
}}
if (spec.kind === 'overlay') {{
  ctx.strokeStyle = '#38BDF8'; ctx.fillStyle = '#38BDF8'; ctx.lineWidth = 3;
  for (const an of spec.annotations) {{
    const x = an.x * W, y = an.y * H;
    if (an.kind === 'rect') {{
      const w = (an.w || 0.125) * W, h = (an.h || 0.139) * H;
      ctx.strokeRect(x - w / 2, y - h / 2, w, h);
    }} else if (an.kind === 'highlight') {{
      const w = (an.w || 0.156) * W, h = (an.h || 0.125) * H;
      ctx.fillStyle = '#FBBF2433'; ctx.fillRect(x - w / 2, y - h / 2, w, h);
      ctx.fillStyle = '#38BDF8';
    }} else if (an.kind === 'scribble' && an.points && an.points.length > 1) {{
      ctx.beginPath();
      ctx.moveTo(an.points[0][0] * W, an.points[0][1] * H);
      for (let i = 1; i < an.points.length; i++) ctx.lineTo(an.points[i][0] * W, an.points[i][1] * H);
      ctx.stroke();
    }} else if (an.kind === 'shape' && an.points && an.points.length > 1) {{
      const x2 = an.points[1][0] * W, y2 = an.points[1][1] * H;
      const a = Math.atan2(y2 - y, x2 - x);
      if (an.shape === 'curve') {{
        const mx = (x + x2) / 2, my = (y + y2) / 2 - 30;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.quadraticCurveTo(mx, my, x2, y2); ctx.stroke();
      }} else {{
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x2, y2); ctx.stroke();
      }}
      ctx.beginPath(); ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 14 * Math.cos(a - 0.4), y2 - 14 * Math.sin(a - 0.4));
      ctx.lineTo(x2 - 14 * Math.cos(a + 0.4), y2 - 14 * Math.sin(a + 0.4));
      ctx.closePath(); ctx.fill();
    }} else if (an.kind === 'caption') {{
      ctx.fillStyle = '#1E1E2E'; ctx.fillRect(x - 10, y - 34, 220, 40);
      ctx.fillStyle = '#F8FAFC'; ctx.font = '18px sans-serif';
      ctx.fillText(an.text, x, y - 8); continue;
    }} else {{
      ctx.beginPath(); ctx.arc(x, y, 16, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    }}
    if (an.text) {{
      ctx.fillStyle = '#F8FAFC'; ctx.font = '18px sans-serif';
      ctx.fillText(an.text, x + 24, y - 12);
    }}
  }}
}}
</script></body></html>
"""


def render_live_html(spec: dict[str, Any]) -> str:
    """A spec → a self-contained live HTML page (a string).

    Diagrams and overlays draw on an inline canvas with no server
    and no external URL. 3D scenes reuse the motion rungs' three.js
    page — the repo's real 3D rung, CDN pins verified live at build
    time by that module.
    """
    validated = validate_scene_spec(spec)
    if validated["kind"] == "scene":
        from ..media.motion_rungs import three_scene_page

        return three_scene_page(
            {"camera": validated["camera"],
             "background": validated["background"],
             "objects": validated["objects"]},
            title=validated["title"] or "Devon 3D scene")
    width, height = _DEFAULT_SIZE
    return _CANVAS_PAGE.format(
        title=_html.escape(validated["title"] or "scene"),
        width=width, height=height,
        spec_json=json.dumps(validated),
    )


def write_live_page(spec: dict[str, Any],
                    path: str | Path) -> str:
    """Write the live page to ``path``; return the path as a string."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_live_html(spec), encoding="utf-8")
    return str(target)


# ── PNG snapshot ─────────────────────────────────────────────────────


def _diagram_png(spec: dict[str, Any], path: Path,
                 size: tuple[int, int]) -> str:
    from ..media.motion_studio.explainers import frame_image

    design = {
        "engine": "diagram",
        "kind": "diagram",
        "title": spec["title"],
        "nodes": [
            {"id": n["id"], "label": n["label"], "x": n["x"], "y": n["y"],
             "shape": n["shape"], "color": n["color"], "step": 0}
            for n in spec["nodes"]
        ],
        "edges": [
            {"from": e["from"], "to": e["to"], "label": e["label"],
             "step": 0}
            for e in spec["edges"]
        ],
    }
    # t == duration: every build step is complete, the still shows the
    # finished diagram — the deterministic frame function, sequenced
    # to one frame.
    image = frame_image(design, size, t=1.0, duration=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return str(path)


def _overlay_png(spec: dict[str, Any], path: Path,
                 size: tuple[int, int]) -> str:
    from PIL import Image, ImageDraw

    width, height = size
    image = Image.new("RGB", size, _BG)
    draw = ImageDraw.Draw(image, "RGBA")
    for i in range(1, 8):
        x = width * i // 8
        draw.line([(x, 0), (x, height)], fill="#1E1E2E")
        y = height * i // 8
        draw.line([(0, y), (width, y)], fill="#1E1E2E")
    for ann in spec["annotations"]:
        x, y = int(ann["x"] * width), int(ann["y"] * height)
        kind = ann["kind"]
        if kind == "rect":
            w = int((ann.get("w") or 0.125) * width)
            h = int((ann.get("h") or 0.139) * height)
            draw.rectangle([x - w // 2, y - h // 2,
                            x + w // 2, y + h // 2],
                           outline="#38BDF8", width=3)
        elif kind == "highlight":
            w = int((ann.get("w") or 0.156) * width)
            h = int((ann.get("h") or 0.125) * height)
            draw.rectangle([x - w // 2, y - h // 2,
                            x + w // 2, y + h // 2],
                           fill=(251, 191, 36, 51))
        elif kind == "scribble" and len(ann.get("points") or []) > 1:
            pts = [(int(px * width), int(py * height))
                   for px, py in ann["points"]]
            draw.line(pts, fill="#38BDF8", width=3)
        elif kind == "shape" and len(ann.get("points") or []) > 1:
            x2 = int(ann["points"][1][0] * width)
            y2 = int(ann["points"][1][1] * height)
            draw.line([(x, y), (x2, y2)], fill="#38BDF8", width=3)
            angle = math.atan2(y2 - y, x2 - x)
            head = [(x2, y2),
                    (int(x2 - 14 * math.cos(angle - 0.4)),
                     int(y2 - 14 * math.sin(angle - 0.4))),
                    (int(x2 - 14 * math.cos(angle + 0.4)),
                     int(y2 - 14 * math.sin(angle + 0.4)))]
            draw.polygon(head, fill="#38BDF8")
        elif kind == "caption":
            draw.rectangle([x - 10, y - 34, x + 210, y + 6],
                           fill="#1E1E2E")
            draw.text((x, y - 26), ann["text"], fill="#F8FAFC")
            continue
        else:  # cursor
            draw.ellipse([x - 16, y - 16, x + 16, y + 16],
                         outline="#38BDF8", width=3)
            draw.ellipse([x - 4, y - 4, x + 4, y + 4],
                         fill="#38BDF8")
        if ann["text"]:
            draw.text((x + 24, y - 12), ann["text"], fill="#F8FAFC")
    if spec["title"]:
        draw.text((24, 20), spec["title"], fill="#F8FAFC")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return str(path)


def render_png(spec: dict[str, Any], path: str | Path, *,
               size: tuple[int, int] = _DEFAULT_SIZE) -> str:
    """A spec → a PNG snapshot at ``path``; return the path.

    Diagrams and overlays only. A 3D scene has no PNG rung — it
    raises :class:`SceneSpecError` naming the live page instead of
    faking a render.
    """
    validated = validate_scene_spec(spec)
    target = Path(path)
    if validated["kind"] == "diagram":
        return _diagram_png(validated, target, size)
    if validated["kind"] == "overlay":
        return _overlay_png(validated, target, size)
    raise SceneSpecError(
        "scene spec: 3D scenes have no PNG rung — render the live "
        "HTML page instead (target='live')")


# ── terminal text ────────────────────────────────────────────────────


def render_text(spec: dict[str, Any]) -> str:
    """A spec → structured text for GodConsole (the degradation rung)."""
    validated = validate_scene_spec(spec)
    kind = validated["kind"]
    title = validated["title"] or "scene"
    if kind == "diagram":
        lines = [f"Scene: {title} (diagram · "
                 f"{len(validated['nodes'])} nodes · "
                 f"{len(validated['edges'])} edges)"]
        for node in validated["nodes"]:
            lines.append(
                f"  [{node['id']}] {node['label']} @ "
                f"({node['x']:.2f}, {node['y']:.2f}) {node['shape']}")
        for edge in validated["edges"]:
            label = f" : {edge['label']}" if edge["label"] else ""
            lines.append(f"  {edge['from']} -> {edge['to']}{label}")
        return "\n".join(lines)
    if kind == "overlay":
        lines = [f"Scene: {title} (overlay · "
                 f"{len(validated['annotations'])} annotations)"]
        for ann in validated["annotations"]:
            text = f" — {ann['text']}" if ann["text"] else ""
            size = ""
            if ann.get("w") or ann.get("h"):
                size = f" {ann.get('w', 0):.2f}x{ann.get('h', 0):.2f}"
            lines.append(f"  ({ann['x']:.2f}, {ann['y']:.2f}) "
                         f"{ann['kind']}{size}{text}")
        return "\n".join(lines)
    lines = [f"Scene: {title} (scene · "
             f"{len(validated['objects'])} objects)"]
    for obj in validated["objects"]:
        lines.append(f"  {obj['type']} @ {obj['position']} "
                     f"size {obj['size']}")
    return "\n".join(lines)


# ── orchestrator ─────────────────────────────────────────────────────


def render_scene(spec: dict[str, Any], *,
                 target: str = "auto",
                 title: str = "",
                 out_dir: str | Path | None = None,
                 profile_kind: str | None = None,
                 size: tuple[int, int] = _DEFAULT_SIZE
                 ) -> dict[str, Any]:
    """Render one spec into a deliverable; return the record.

    ``target`` is ``auto`` (diagram/overlay → PNG, scene → live
    page), ``live``, ``chat`` or ``terminal``. The record carries the
    file path (or the text, for terminal), the counts, and the
    validated spec itself — the handoff to the next stage.
    """
    validated = validate_scene_spec(spec)
    if title:
        validated["title"] = title
    picked = pick_target(
        validated,
        surface=None if target == "auto" else target,
        profile_kind=profile_kind)
    out = Path(out_dir) if out_dir is not None else default_output_dir()
    counts = {
        "nodes": len(validated.get("nodes", [])),
        "edges": len(validated.get("edges", [])),
        "annotations": len(validated.get("annotations", [])),
        "objects": len(validated.get("objects", [])),
    }
    record: dict[str, Any] = {
        "engine": "visuals",
        "kind": validated["kind"],
        "title": validated["title"],
        "target": picked,
        "format": "",
        "path": None,
        "text": None,
        "spec": validated,
        "profile": profile_kind or "",
        **counts,
    }
    if picked == "live":
        path = write_live_page(
            validated, out / _file_name(validated["title"], "html"))
        record["format"] = "html"
        record["path"] = path
        record["note"] = f"live page written to {path}"
        return record
    if picked == "chat":
        path = render_png(
            validated, out / _file_name(validated["title"], "png"),
            size=size)
        record["format"] = "png"
        record["path"] = path
        record["note"] = f"PNG snapshot written to {path}"
        return record
    record["format"] = "text"
    record["text"] = render_text(validated)
    record["note"] = "terminal rendering (no file)"
    return record
