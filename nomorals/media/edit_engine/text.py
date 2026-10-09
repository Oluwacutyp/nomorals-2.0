"""Style-agnostic text layers → burned-in ASS subtitles.

A :class:`TextLayer` is presentation data: what text, when, where, how
it looks. Word-level timings enable karaoke highlighting — the timing
*data* lives here; beat-snapping policies live in style presets.

Rendering goes through Devon's caption engine
(:mod:`nomorals.media_edit.captions`) with the script resolution
rewritten to the output canvas so libass renders 1:1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...media_edit import captions as _captions
from ...media_edit.videos import MediaEditError, run_ffmpeg

#: position preset → ASS alignment number
_POSITIONS = {
    "bottom": 2, "top": 8, "center": 5, "middle": 5,
    "bottom-left": 1, "bottom-right": 3,
    "top-left": 7, "top-right": 9,
}

#: CSS-ish color names → ASS &HAABBGGRR
_NAMED_COLORS = {
    "white": "&H00FFFFFF", "black": "&H00000000",
    "yellow": "&H0000FFFF", "red": "&H000000FF",
    "green": "&H0000FF00", "blue": "&H00FF0000",
    "cyan": "&H00FFFF00", "magenta": "&H00FF00FF",
    "orange": "&H0000A5FF",
}


def ass_color(value: str) -> str:
    """CSS color (#rrggbb, #rrggbbaa, or name) → ASS &HAABBGGRR."""
    v = str(value or "white").strip().lower()
    if v in _NAMED_COLORS:
        return _NAMED_COLORS[v]
    m = re.fullmatch(r"#([0-9a-f]{6})([0-9a-f]{2})?", v)
    if not m:
        raise MediaEditError(f"cannot parse color {value!r}")
    rr, gg, bb = m.group(1)[0:2], m.group(1)[2:4], m.group(1)[4:6]
    aa = m.group(2) or "00"
    return f"&H{aa}{bb}{gg}{rr}".upper()


@dataclass
class TextLayer:
    """One text overlay.

    Provide ``text`` (static caption) and/or ``words`` (word timings
    for karaoke highlighting). ``position``: top|center|bottom (or a
    corner); explicit ``x``/``y`` override the preset. ``style`` is the
    caption *rendering* choice (karaoke/hormozi/minimal/mrbeast from
    Devon's caption engine) — picked by the caller, not the engine.
    """
    text: str | None = None
    words: list[dict[str, Any]] | None = None
    start: float = 0.0
    end: float | None = None
    font: str | None = None
    size: int = 72
    color: str = "white"
    stroke: int = 2
    stroke_color: str = "black"
    box: bool = False
    box_color: str = "#00000099"
    position: str = "bottom"
    x: int | None = None
    y: int | None = None
    style: str = "karaoke"
    margin_v: int = 120

    def __post_init__(self) -> None:
        if not self.text and not self.words:
            raise MediaEditError("TextLayer needs text= or words=")
        self.start = max(0.0, float(self.start))
        if self.end is not None:
            self.end = float(self.end)
            if self.end <= self.start:
                raise MediaEditError("TextLayer end must exceed start")
        pos = str(self.position).strip().lower()
        if pos not in _POSITIONS and self.x is None and self.y is None:
            raise MediaEditError(
                f"unknown text position {self.position!r}; use: "
                f"{sorted(_POSITIONS)} or explicit x/y")
        self.position = pos

    @property
    def alignment(self) -> int:
        return _POSITIONS.get(self.position, 2)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "words": self.words,
                "start": self.start, "end": self.end, "font": self.font,
                "size": self.size, "color": self.color,
                "stroke": self.stroke, "stroke_color": self.stroke_color,
                "box": self.box, "box_color": self.box_color,
                "position": self.position, "x": self.x, "y": self.y,
                "style": self.style, "margin_v": self.margin_v}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> "TextLayer":
        if isinstance(data, str):
            return cls(text=data) if data.strip() else cls(text=" ")
        return cls(**{k: data[k] for k in (
            "text", "words", "start", "end", "font", "size", "color",
            "stroke", "stroke_color", "box", "box_color", "position",
            "x", "y", "style", "margin_v") if k in data})


def as_words(words: list[Any]) -> list[_captions.Word]:
    """Coerce mappings / Word objects → caption-engine Words."""
    out: list[_captions.Word] = []
    for w in words:
        if isinstance(w, _captions.Word):
            out.append(w)
        else:
            out.append(_captions.Word.from_dict(dict(w)))
    return out


def words_to_ass(words: list[Any], *, style: str = "karaoke",
                 width: int = 1080, height: int = 1920) -> str:
    """Word timings → word-highlight .ass on the output canvas."""
    ws = as_words(words)
    if not ws:
        raise MediaEditError("no words — nothing to render")
    ass = _captions.words_to_ass(ws, style=style)
    ass = re.sub(r"^PlayResX:.*$", f"PlayResX: {width}", ass, flags=re.M)
    ass = re.sub(r"^PlayResY:.*$", f"PlayResY: {height}", ass, flags=re.M)
    return ass


def layer_to_ass(layer: TextLayer, *, width: int = 1080,
                 height: int = 1920) -> str:
    """A :class:`TextLayer` → .ass script text."""
    if layer.words:
        base = words_to_ass(layer.words, style=layer.style,
                            width=width, height=height)
        return base
    # static text → one dialogue event with the layer's look
    end = layer.end if layer.end is not None else layer.start + 4.0
    def ts(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"
    align = layer.alignment
    body = (str(layer.text or "").replace("\n", r"\N")
            .replace("{", r"\{").replace("}", r"\}"))
    style_line = (
        f"Style: Layer,Arial,{layer.size},"
        f"{ass_color(layer.color)},{ass_color(layer.color)},"
        f"{ass_color(layer.stroke_color)},{ass_color('#00000088')},"
        f"0,{-int(bool(layer.stroke))},0,0,100,100,0,0,1,"
        f"{layer.stroke},{layer.stroke},"
        f"{'3' if layer.box else '1'},"
        f"{layer.margin_v},{layer.margin_v},{layer.margin_v},1")
    # box → BorderStyle 3 needs a box color override
    if layer.box:
        style_line += f"\n# box color: {ass_color(layer.box_color)}"
    dialogue = (f"Dialogue: 0,{ts(layer.start)},{ts(end)},Layer,,0,0,0,,"
                f"{{\\an{align}}}"
                + (f"{{\\pos({layer.x},{layer.y})}}"
                   if layer.x is not None and layer.y is not None else "")
                + body)
    # strip the comment line (kept human-readable, harmless to libass)
    style_line = "\n".join(
        ln for ln in style_line.splitlines() if not ln.startswith("#"))
    return (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{style_line}\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
        f"{dialogue}\n"
    )


def estimate_word_timings(text: str, *, start: float = 0.0,
                          wpm: float = 150.0,
                          beats: list[float] | None = None
                          ) -> list[_captions.Word]:
    """APPROXIMATE word timings from plain text (even spacing at ``wpm``).

    An estimate for when no TTS alignment or transcription exists — it
    does not claim to be a real alignment. ``beats`` optionally snaps
    each word's start to the nearest grid time (any time grid, e.g.
    beats, chapter marks).
    """
    tokens = [t for t in (text or "").split() if t]
    if not tokens:
        raise MediaEditError("no text — nothing to time")
    if wpm <= 0:
        raise MediaEditError("wpm must be > 0")
    per = 60.0 / float(wpm)
    grid = sorted(float(b) for b in beats) if beats else []
    words: list[_captions.Word] = []
    t = max(0.0, float(start))
    for tok in tokens:
        if grid:
            t = min(grid, key=lambda g: abs(g - t))
            if words:
                t = max(t, words[-1].end + 0.01)
        words.append(_captions.Word(start=round(t, 3),
                                    end=round(t + per * 0.92, 3),
                                    text=tok))
        t += per
    return words


def build_captions(phrases: list[tuple[str, float, float]],
                   out_path: str | Path) -> str:
    """Write an SRT sidecar from ``[(text, start_s, end_s), …]``."""
    words = [_captions.Word(start=float(a), end=float(b), text=str(t))
             for t, a, b in phrases]
    if not words:
        raise MediaEditError("no caption phrases — nothing to write")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_captions.words_to_srt(words), encoding="utf-8")
    return str(p)


def burn_text_layers(video: str | Path, layers: list[TextLayer], *,
                     width: int = 1080, height: int = 1920,
                     out: str | Path | None = None,
                     suffix: str = "text",
                     workdir: str | Path | None = None) -> dict[str, Any]:
    """Burn text layers into ``video`` (one libass pass per layer)."""
    from pathlib import Path as _P
    p = _P(video)
    if not p.exists():
        raise MediaEditError(f"no such video: {video}")
    if not layers:
        raise MediaEditError("no text layers — nothing to burn")
    tmp = _P(workdir) if workdir else p.parent
    cur = p
    for i, layer in enumerate(layers):
        ass = layer_to_ass(layer, width=width, height=height)
        ass_path = tmp / f"{p.stem}.layer{i}.ass"
        ass_path.write_text(ass, encoding="utf-8")
        run = _captions.burn_captions(cur, ass_path, out_dir=tmp,
                                      suffix=f"{suffix}{i}")
        cur = _P(run["output"])
    if out and _P(cur).resolve() != _P(out).resolve():
        run_ffmpeg(["-i", str(cur), "-c", "copy", str(out)])
        cur = _P(out)
    return {"input": str(p), "output": str(cur), "layers": len(layers)}
