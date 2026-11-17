"""The interactive live page: the scene, the flight, the annotations.

Lane 1's renderer ladder writes a static live page (the scene drawn
once on a canvas). This module is the interaction half on top of it:
the same page, plus a transparent overlay canvas that draws the
board's sized annotations (rects, highlights, scribbles, arrows —
the tag vocabulary, with the sizes the brain authored) and animates
the buddy's cursor along the flight frames with
``requestAnimationFrame``, the label bubble popping in on arrival.

The flight is precomputed Python data (see :mod:`flight`); the page
only interpolates between adjacent frames, so the motion on the
page is exactly the motion the tests assert. The page is
self-contained — no server, no external URL — and offline-capable
like the base page it extends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .render import render_live_html
from .spec import validate_scene_spec

__all__ = [
    "render_interactive_page",
    "write_interactive_page",
]

#: The bubble holds 3s after arrival, then fades (farzaa: hold 3s,
#: fade 0.5s).
_BUBBLE_HOLD_MS = 3000

_FX_SCRIPT = """
<canvas id="fx" width="1280" height="720" style="position:fixed;left:0;top:0;pointer-events:none"></canvas>
<script>
const FLIGHT = {flight_json};
const ANNS = {anns_json};
const fx = document.getElementById('fx');
const fctx = fx.getContext('2d');
const W = fx.width, H = fx.height;
function drawAnn(an) {{
  const x = an.x * W, y = an.y * H;
  fctx.strokeStyle = '#38BDF8'; fctx.fillStyle = '#38BDF8'; fctx.lineWidth = 3;
  if (an.kind === 'rect') {{
    const w = (an.w || 0.125) * W, h = (an.h || 0.139) * H;
    fctx.strokeRect(x - w / 2, y - h / 2, w, h);
  }} else if (an.kind === 'highlight') {{
    const w = (an.w || 0.156) * W, h = (an.h || 0.125) * H;
    fctx.fillStyle = '#FBBF2433'; fctx.fillRect(x - w / 2, y - h / 2, w, h);
    fctx.fillStyle = '#38BDF8';
  }} else if (an.kind === 'scribble' && an.points && an.points.length > 1) {{
    fctx.beginPath();
    fctx.moveTo(an.points[0][0] * W, an.points[0][1] * H);
    for (let i = 1; i < an.points.length; i++) fctx.lineTo(an.points[i][0] * W, an.points[i][1] * H);
    fctx.stroke();
  }} else if (an.kind === 'shape' && an.points && an.points.length > 1) {{
    const x2 = an.points[1][0] * W, y2 = an.points[1][1] * H;
    const a = Math.atan2(y2 - y, x2 - x);
    if (an.shape === 'curve') {{
      const mx = (x + x2) / 2, my = (y + y2) / 2 - 30;
      fctx.beginPath(); fctx.moveTo(x, y); fctx.quadraticCurveTo(mx, my, x2, y2); fctx.stroke();
    }} else {{
      fctx.beginPath(); fctx.moveTo(x, y); fctx.lineTo(x2, y2); fctx.stroke();
    }}
    fctx.beginPath(); fctx.moveTo(x2, y2);
    fctx.lineTo(x2 - 14 * Math.cos(a - 0.4), y2 - 14 * Math.sin(a - 0.4));
    fctx.lineTo(x2 - 14 * Math.cos(a + 0.4), y2 - 14 * Math.sin(a + 0.4));
    fctx.closePath(); fctx.fill();
  }} else if (an.kind === 'caption') {{
    fctx.fillStyle = '#1E1E2E'; fctx.fillRect(x - 10, y - 34, 220, 40);
    fctx.fillStyle = '#F8FAFC'; fctx.font = '18px sans-serif';
    fctx.fillText(an.text || '', x, y - 8);
    return;
  }} else {{
    fctx.beginPath(); fctx.arc(x, y, 16, 0, Math.PI * 2); fctx.stroke();
    fctx.beginPath(); fctx.arc(x, y, 4, 0, Math.PI * 2); fctx.fill();
  }}
  if (an.text) {{
    fctx.fillStyle = '#F8FAFC'; fctx.font = '18px sans-serif';
    fctx.fillText(an.text, x + 24, y - 12);
  }}
}}
function drawCursor(frame, label) {{
  const x = frame.x * W, y = frame.y * H;
  const a = (frame.angle - 90) * Math.PI / 180;
  const s = frame.scale;
  fctx.save();
  fctx.translate(x, y);
  fctx.rotate(a);
  fctx.scale(s, s);
  fctx.fillStyle = '#38BDF8';
  fctx.beginPath();
  fctx.moveTo(0, -20); fctx.lineTo(14, 14); fctx.lineTo(0, 7); fctx.lineTo(-14, 14);
  fctx.closePath(); fctx.fill();
  fctx.restore();
  if (label) {{
    fctx.fillStyle = '#1E1E2E';
    const w = fctx.measureText(label).width + 24;
    fctx.fillRect(x + 18, y - 40, w, 30);
    fctx.fillStyle = '#F8FAFC'; fctx.font = '16px sans-serif';
    fctx.fillText(label, x + 30, y - 19);
  }}
}}
for (const an of ANNS) drawAnn(an);
if (FLIGHT.length > 1) {{
  const duration = FLIGHT[FLIGHT.length - 1].t * 1000;
  const label = (ANNS.length && ANNS[ANNS.length - 1].text) || '';
  const started = performance.now();
  let arrivedAt = null;
  function frame(now) {{
    const elapsed = now - started;
    const t = Math.min(elapsed / 1000, FLIGHT[FLIGHT.length - 1].t);
    let i = 0;
    while (i < FLIGHT.length - 2 && FLIGHT[i + 1].t < t) i++;
    const a = FLIGHT[i], b = FLIGHT[i + 1];
    const span = (b.t - a.t) || 1;
    const k = Math.min(1, Math.max(0, (t - a.t) / span));
    const cur = {{
      x: a.x + (b.x - a.x) * k,
      y: a.y + (b.y - a.y) * k,
      angle: a.angle + (b.angle - a.angle) * k,
      scale: a.scale + (b.scale - a.scale) * k,
    }};
    fctx.clearRect(0, 0, W, H);
    for (const an of ANNS) drawAnn(an);
    drawCursor(cur, elapsed >= duration ? label : '');
    if (elapsed < duration) {{
      requestAnimationFrame(frame);
    }} else {{
      if (arrivedAt === null) arrivedAt = now;
      if (now - arrivedAt < {bubble_hold}) {{
        requestAnimationFrame(frame);
      }} else {{
        fctx.clearRect(0, 0, W, H);
        for (const an of ANNS) drawAnn(an);
        drawCursor(FLIGHT[FLIGHT.length - 1], '');
      }}
    }}
  }}
  requestAnimationFrame(frame);
}}
</script>
"""


def _annotations_of(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if spec.get("kind") == "overlay":
        return list(spec.get("annotations", []))
    return []


def render_interactive_page(
        spec: dict[str, Any], *,
        flight: list[dict[str, float]] | None = None,
        annotations: list[dict[str, Any]] | None = None,
        title: str = "") -> str:
    """A spec -> the interactive live page (a string).

    ``flight`` is the precomputed frame list from
    :func:`flight.flight_frames` (or ``None`` for a still page);
    ``annotations`` are extra overlay annotations drawn on the
    effect canvas (the board's live ones when omitted and the spec
    is an overlay). The base static page comes from the organ's own
    renderer ladder — this module only adds the interaction layer.
    """
    validated = validate_scene_spec(spec)
    if title:
        validated["title"] = title
    if flight is None:
        flight = []
    extra = (list(annotations) if annotations is not None
             else _annotations_of(validated))
    base = render_live_html(validated)
    fx = _FX_SCRIPT.format(
        flight_json=json.dumps(flight),
        anns_json=json.dumps(extra),
        bubble_hold=_BUBBLE_HOLD_MS,
    )
    return base.replace("</body>", fx + "</body>", 1)


def write_interactive_page(
        spec: dict[str, Any], path: str | Path, *,
        flight: list[dict[str, float]] | None = None,
        annotations: list[dict[str, Any]] | None = None,
        title: str = "") -> str:
    """Write the interactive page to ``path``; return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_interactive_page(spec, flight=flight,
                                annotations=annotations, title=title),
        encoding="utf-8")
    return str(target)


def page_for_board(board: Any, *, title: str = "",
                  out_dir: str | Path | None = None) -> str | None:
    """The board's live page: overlay spec + last flight, written
    to ``out_dir``. ``None`` when the board holds nothing live."""
    from ..core.ids import new_id
    from .render import default_output_dir

    spec = board.to_overlay_spec(title or "scene")
    if spec is None:
        return None
    slug = "".join(ch.lower() if ch.isalnum() else "-"
                   for ch in (spec["title"] or ""))
    slug = "-".join(part for part in slug.split("-") if part)[:40]
    out = Path(out_dir) if out_dir is not None else default_output_dir()
    return write_interactive_page(
        spec, out / f"{slug or 'scene'}-{new_id()[:8]}.html",
        flight=board.last_flight)
