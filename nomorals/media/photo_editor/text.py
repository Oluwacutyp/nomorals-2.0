"""Type layers — ``TySh`` parsing and the vector re-render
for layers with no stored raster.

A ``TySh`` block carries a 6-double transform matrix and a
versioned ``TxLr`` descriptor: the text itself (``Txt ``) and
the ``EngineData`` (a PostScript-like structure with the
font set, the style runs — font index, size in points, fill
colour — and the run lengths). The compositor keeps the
stored raster whenever the layer has one (Photoshop's
raster is better than any re-render); this module's render
path exists only for layers with no raster at all (text on
a path, empty text layers), where the alternative is nothing.

EngineData run extraction is best-effort: the first font,
size and colour of each style run in the StyleRun section,
paired with the run lengths in order. Fill colours are
Type 1, ``Values [1.0, R, G, B]`` — the last three are the
RGB colour (verified against the corpus: (230, 120, 30)
<-> [1.0 .90196 .47059 .11765], (40, 40, 140) <-> [1.0
.15686 .15686 .54902]).
"""

from __future__ import annotations

import glob
import math
import re
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .descriptor import desc_text, parse_versioned

__all__ = ["parse_tysh", "render_text"]

# PostScript name -> system font families (Liberation is
# metric-compatible with Arial / Times; DejaVu is the last
# resort). The owner's box has DejaVu + Liberation + Noto.
_FONT_FAMILIES = (
    ("arial", "LiberationSans"),
    ("helvetica", "LiberationSans"),
    ("myriadpro", "LiberationSans"),
    ("source", "LiberationSerif"),
    ("georgia", "LiberationSerif"),
    ("times", "LiberationSerif"),
    ("courier", "LiberationMono"),
    ("monaco", "DejaVuSansMono"),
    ("hiragino", "NotoSansCJK"),
    ("noto", "NotoSans"),
)

_FONT_DIRS = ("/usr/share/fonts", "/usr/local/share/fonts")


def _decode_utf16_name(raw: str) -> str:
    """A PostScript ``( ... )`` string holding UTF-16BE bytes
    (read as latin-1) -> the name."""
    data = raw.encode("latin-1", errors="ignore")
    if data[:2] == b"\xfe\xff":
        data = data[2:]
    try:
        return data.decode("utf-16-be",
                           errors="replace").strip("\x00")
    except UnicodeDecodeError:
        return raw


_FONT_CACHE: dict[str, str | None] = {}


def _find_font(postscript_name: str, bold: bool = False,
               italic: bool = False) -> str | None:
    """A PostScript font name -> a font file on this box, or
    None. Results are cached; the search is a fixed list of
    font directories, never a subprocess."""
    key = f"{postscript_name}|{bold}|{italic}"
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    name = postscript_name.lower().replace("-", "")
    family = "DejaVuSans"
    for prefix, mapped in _FONT_FAMILIES:
        if name.startswith(prefix):
            family = mapped
            break
    style = ""
    if bold and italic:
        style = "BoldItalic"
    elif bold:
        style = "Bold"
    elif italic:
        style = "Italic"
    candidates = []
    if style:
        candidates.append(f"{family}-{style}")
    candidates.append(f"{family}-Regular")
    candidates.append(family)
    found: str | None = None
    for candidate in candidates:
        for directory in _FONT_DIRS:
            paths = sorted(glob.glob(
                f"{directory}/**/{candidate}.ttf", recursive=True))
            paths += sorted(glob.glob(
                f"{directory}/**/{candidate}.otf", recursive=True))
            if paths:
                found = paths[0]
                break
        if found is not None:
            break
    if found is None:
        for directory in _FONT_DIRS:
            paths = sorted(glob.glob(
                f"{directory}/**/DejaVuSans.ttf", recursive=True))
            if paths:
                found = paths[0]
                break
    _FONT_CACHE[key] = found
    return found


def _engine_text(data: bytes) -> str:
    """The EngineData payload as latin-1 text (the structure
    is ASCII with UTF-16BE strings inside parentheses)."""
    return data.decode("latin-1", errors="replace")


def _font_set(engine: str) -> list[str]:
    """The FontSet names, in index order."""
    start = engine.find("/FontSet [")
    if start < 0:
        return []
    end = engine.find("]", start)
    section = engine[start:end if end > 0 else len(engine)]
    names = []
    for match in re.finditer(r"/Name \((.*?)\)", section,
                             re.S):
        names.append(_decode_utf16_name(match.group(1)))
    return names


def _style_runs(engine: str, fonts: list[str]
                ) -> list[dict]:
    """The StyleRun runs: (font, size in points, RGB colour,
    length in characters), in order. The StyleRun section runs
    from ``/StyleRun`` to ``/GridInfo``; each StyleSheetData
    block holds one ``/Font N``, one ``/FontSize`` and one
    ``/FillColor``."""
    start = engine.find("/StyleRun")
    if start < 0:
        return []
    end = engine.find("/GridInfo", start)
    section = engine[start:end if end > 0 else len(engine)]
    styles = []
    for match in re.finditer(
            r"/Font (\d+)\s*/FontSize ([\d.]+).*?"
            r"/FillColor\s*<<\s*/Type (\d+)\s*/Values "
            r"\[ ([^\]]*) \]",
            section, re.S):
        index = int(match.group(1))
        size = float(match.group(2))
        values = [float(v) for v in match.group(4).split()]
        if len(values) >= 4:
            rgb = tuple(values[1:4])
        elif len(values) == 3:
            rgb = tuple(values)
        else:
            rgb = (0.0, 0.0, 0.0)
        rgb = tuple(min(max(c, 0.0), 1.0) for c in rgb)
        name = fonts[index] if 0 <= index < len(fonts) else ""
        styles.append({"font": name, "size": size,
                       "color": rgb})
    lengths = re.findall(r"/RunLengthArray \[ ([^\]]*) \]",
                         section)
    flat_lengths: list[int] = []
    for group in lengths:
        flat_lengths.extend(int(v) for v in group.split())
    runs = []
    for i, style in enumerate(styles):
        length = flat_lengths[i] if i < len(flat_lengths) \
            else None
        runs.append({**style, "length": length})
    return runs


def parse_tysh(data: bytes) -> dict | None:
    """A ``TySh`` block -> a text dict (text, 6-double matrix,
    runs), or None when it cannot be parsed."""
    if len(data) < 56:
        return None
    matrix = [struct.unpack(">d", data[2 + i * 8:10 + i * 8])[0]
              for i in range(6)]
    if not all(math.isfinite(v) for v in matrix):
        return None
    try:
        desc, _ = parse_versioned(data, 52)
    except Exception:
        return None
    text = desc_text(desc, "Txt ") or ""
    runs: list[dict] = []
    engine_value = desc.get("EngineData")
    if isinstance(engine_value, tuple) \
            and len(engine_value) == 2:
        payload = engine_value[1]
        if isinstance(payload, tuple) and len(payload) == 2 \
                and isinstance(payload[1], (bytes, bytearray)):
            engine = _engine_text(bytes(payload[1]))
            fonts = _font_set(engine)
            runs = _style_runs(engine, fonts)
    if not runs:
        runs = [{"font": "", "size": 12.0,
                 "color": (0.0, 0.0, 0.0),
                 "length": len(text)}]
    return {"text": text, "matrix": matrix, "runs": runs,
            "descriptor": desc}


def _load_font(run: dict) -> ImageFont.FreeTypeFont | None:
    name = run.get("font", "")
    size = max(1, int(round(run.get("size", 12.0))))
    bold = "bold" in name.lower()
    italic = "italic" in name.lower() or "oblique" in name.lower()
    path = _find_font(name, bold=bold, italic=italic)
    if path is None:
        return None
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return None


def render_text(spec: dict, width: int, height: int, *,
                left: int = 0, top: int = 0) -> np.ndarray | None:
    """Render a parsed text spec as float RGBA over
    ``width``x``height`` at document offset (``left``,
    ``top``). The runs are laid out in local text space (the
    matrix origin is the baseline start) and then mapped
    through the TySh matrix. Returns None when no run has a
    usable font."""
    text = spec.get("text", "")
    matrix = spec.get("matrix", [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    runs = spec.get("runs", [])
    if not text:
        return None
    # Split the text into runs by length, then into lines.
    segments: list[tuple[str, dict]] = []
    pos = 0
    for run in runs:
        length = run.get("length")
        if length is None:
            length = len(text) - pos
        chunk = text[pos:pos + length]
        pos += length
        if chunk:
            segments.append((chunk, run))
    if pos < len(text):
        segments.append((text[pos:], runs[-1] if runs else {}))
    if not segments:
        return None
    lines: list[list[tuple[str, dict]]] = [[]]
    for chunk, run in segments:
        parts = chunk.split("\r")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append([])
            if part:
                lines[-1].append((part, run))
    # Measure in local space: one font per run, laid out left
    # to right, baseline at y = 0.
    draw_fonts: list[tuple[str, dict,
                           ImageFont.FreeTypeFont | None]] = []
    for line in lines:
        for chunk, run in line:
            font = _load_font(run)
            draw_fonts.append((chunk, run, font))
    if not any(font is not None for _, _, font in draw_fonts):
        return None
    line_height = 0.0
    line_widths: list[float] = []
    for line in lines:
        width_px = 0.0
        height_px = 0.0
        for chunk, run in line:
            font = _load_font(run)
            if font is None:
                continue
            bbox = font.getbbox(chunk)
            width_px += bbox[2] - bbox[0]
            height_px = max(height_px,
                            font.getmetrics()[0]
                            + font.getmetrics()[1])
        line_widths.append(width_px)
        line_height = max(line_height, height_px)
    if line_height <= 0.0:
        line_height = max((run.get("size", 12.0) for _, run in
                           segments), default=12.0) * 1.2
    local_w = int(math.ceil(max(line_widths, default=0.0))) + 4
    local_h = int(math.ceil(line_height * len(lines))) + 4
    if local_w <= 0 or local_h <= 0:
        return None
    # Draw with the baseline at y = ascent, so local (0, 0)
    # (the matrix origin) is the baseline start.
    ascent = 0
    for _chunk, _run, font in draw_fonts:
        if font is None:
            continue
        ascent = max(ascent, font.getmetrics()[0])
    local = Image.new("RGBA", (local_w, local_h),
                      (0, 0, 0, 0))
    draw = ImageDraw.Draw(local)
    y = 0
    for line in lines:
        x = 0.0
        for chunk, run in line:
            font = _load_font(run)
            if font is None:
                continue
            rgb = run.get("color", (0.0, 0.0, 0.0))
            fill = (int(rgb[0] * 255), int(rgb[1] * 255),
                    int(rgb[2] * 255), 255)
            draw.text((x, y), chunk, font=font, fill=fill,
                      anchor="la")
            bbox = font.getbbox(chunk)
            x += bbox[2] - bbox[0]
        y += line_height
    # Map local -> document through the TySh matrix: the
    # matrix maps (x, y) -> (xx*x + yx*y + tx, xy*x + yy*y +
    # ty); the local image's baseline sits ``ascent`` below
    # its top, so the input point is (x, y + ascent).
    xx, xy, yx, yy, tx, ty = matrix
    det = xx * yy - yx * xy
    if abs(det) < 1e-12:
        return None
    # Output (document) coords -> local image coords.
    inv = (yy / det, -yx / det, -xy / det, xx / det)
    # out = M . (in - (0, ascent)) + (left, top) offset, so
    # in = M^-1 . (out - (left, top)) + (0, ascent).
    a, b, c, d = inv
    e = (-(a * (tx - left) + b * (ty - top)))
    f = (-(c * (tx - left) + d * (ty - top))) + ascent
    transformed = local.transform(
        (width, height), Image.AFFINE, (a, b, e, c, d, f),
        resample=Image.BILINEAR)
    arr = np.asarray(transformed, dtype=np.float64) / 255.0
    return np.ascontiguousarray(arr)
