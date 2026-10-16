"""WhatsApp renderer: tables / diffs → PNG.

WhatsApp cannot render markdown tables or diffs — they arrive as a
mangled wall of monospace text (or worse, reflowed). This module
detects table-like and diff-like text and renders it to a PNG image
(Pillow), which WhatsApp *can* display faithfully. The caller sends
the PNG with the original text as caption/fallback.

* :func:`needs_image_render` — True for markdown tables (``| a | b |``
  rows + a ``|---|---|`` separator) and unified diffs (``+++``/``---``
  headers or ``+/-`` body lines).
* :func:`table_to_png` / :func:`diff_to_png` — real Pillow renders:
  measured column widths, header rule, green/red diff lines. The image
  is sized from the content (never a fixed canvas that clips).
* :func:`render_for_whatsapp` — ``("text", text)`` for plain text,
  ``("image", png_path)`` for tables/diffs. The adapter sends the PNG
  via ``send_media`` and keeps the text as the caption.

Invariant: a table/diff input always yields a real, non-empty PNG file
whose pixels contain the content (verified in tests by file size +
dimensions), never a text fallback pretending to be an image.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = [
    "needs_image_render",
    "is_table",
    "is_diff",
    "table_to_png",
    "diff_to_png",
    "render_for_whatsapp",
]

_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_DIFF_HEADER = re.compile(r"^(---|\+\+\+|diff --git|@@)", re.M)
_DIFF_BODY = re.compile(r"^([+-][^+-].*)$", re.M)

# Pillow rendering constants.
_FONT_SIZE = 15
_LINE_H = 20
_PAD_X = 14
_PAD_Y = 12
_BG = (255, 255, 255)
_FG = (20, 20, 20)
_HEADER_BG = (235, 238, 245)
_GRID = (200, 205, 215)
_ADD_BG = (232, 248, 232)
_ADD_FG = (20, 90, 30)
_DEL_BG = (250, 232, 232)
_DEL_FG = (140, 30, 30)
_HUNK_FG = (90, 60, 140)


def is_table(text: str) -> bool:
    """True iff the text contains a markdown table (header + separator)."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if _TABLE_ROW.match(line) and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            # A separator row must contain at least one run of dashes.
            if "-" in lines[i + 1]:
                return True
    return False


def is_diff(text: str) -> bool:
    """True iff the text looks like a unified diff."""
    t = text or ""
    if _DIFF_HEADER.search(t):
        return True
    hits = len(_DIFF_BODY.findall(t))
    return hits >= 3 and ("---" in t or "+++" in t)


def needs_image_render(text: str) -> bool:
    """True when WhatsApp would mangle this text — tables and diffs."""
    return is_table(text) or is_diff(text)


def _font() -> Any:
    from PIL import ImageFont
    for candidate in ("DejaVuSansMono.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, _FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def _parse_table(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in (text or "").splitlines():
        if not _TABLE_ROW.match(line):
            continue
        if _TABLE_SEP.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def table_to_png(text: str, path: str | Path) -> Path:
    """Render a markdown table to a PNG. Returns the written path."""
    from PIL import Image, ImageDraw

    rows = _parse_table(text)
    if not rows:
        raise ValueError("table_to_png: no table rows found")
    font = _font()
    probe = Image.new("RGB", (8, 8))
    measure = ImageDraw.Draw(probe)
    ncols = max(len(r) for r in rows)
    widths = [0] * ncols
    for row in rows:
        for j, cell in enumerate(row):
            w = int(measure.textlength(cell, font=font))
            widths[j] = max(widths[j], w)
    widths = [w + 16 for w in widths]
    total_w = sum(widths) + 2 * _PAD_X
    total_h = len(rows) * _LINE_H + 2 * _PAD_Y + 2
    img = Image.new("RGB", (total_w, total_h), _BG)
    draw = ImageDraw.Draw(img)
    y = _PAD_Y
    for i, row in enumerate(rows):
        x = _PAD_X
        if i == 0:
            draw.rectangle([_PAD_X - 2, y - 2, total_w - _PAD_X + 2, y + _LINE_H - 2],
                           fill=_HEADER_BG)
        for j in range(ncols):
            cell = row[j] if j < len(row) else ""
            draw.text((x + 8, y + 2), cell, font=font, fill=_FG)
            x += widths[j]
            draw.line([(x, _PAD_Y), (x, total_h - _PAD_Y)], fill=_GRID, width=1)
        draw.line([(_PAD_X, y + _LINE_H), (total_w - _PAD_X, y + _LINE_H)],
                  fill=_GRID, width=1)
        y += _LINE_H
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    return out


def diff_to_png(text: str, path: str | Path) -> Path:
    """Render a unified diff to a PNG (green adds, red dels)."""
    from PIL import Image, ImageDraw

    lines = (text or "").splitlines()
    if not lines:
        raise ValueError("diff_to_png: empty diff")
    font = _font()
    probe = Image.new("RGB", (8, 8))
    measure = ImageDraw.Draw(probe)
    max_w = max((int(measure.textlength(ln, font=font)) for ln in lines), default=0)
    total_w = max(320, max_w + 2 * _PAD_X)
    total_h = len(lines) * _LINE_H + 2 * _PAD_Y
    img = Image.new("RGB", (total_w, total_h), _BG)
    draw = ImageDraw.Draw(img)
    y = _PAD_Y
    for line in lines:
        bg, fg = _BG, _FG
        if line.startswith("+++") or line.startswith("---"):
            bg, fg = _HEADER_BG, _FG
        elif line.startswith("@@"):
            fg = _HUNK_FG
        elif line.startswith("+") and not line.startswith("++"):
            bg, fg = _ADD_BG, _ADD_FG
        elif line.startswith("-") and not line.startswith("--"):
            bg, fg = _DEL_BG, _DEL_FG
        if bg != _BG:
            draw.rectangle([_PAD_X - 2, y - 1, total_w - _PAD_X + 2, y + _LINE_H - 1], fill=bg)
        draw.text((_PAD_X, y + 2), line, font=font, fill=fg)
        y += _LINE_H
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    return out


def render_for_whatsapp(text: str, out_dir: str | Path,
                        *, stem: str = "whatsapp-render") -> tuple[str, str]:
    """``("text", text)`` or ``("image", png_path)`` for WhatsApp."""
    if not needs_image_render(text):
        return ("text", text or "")
    out = Path(out_dir) / f"{stem}.png"
    if is_table(text):
        table_to_png(text, out)
    else:
        diff_to_png(text, out)
    return ("image", str(out))
