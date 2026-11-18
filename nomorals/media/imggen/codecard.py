"""Code cards — Carbon-style presentation images of code (52-websites #19).

Paste code, get a presentation image: real syntax highlighting from
real tokenisation (Pygments — the lexer's own token stream, never a
brain-drawn approximation), several themes, a window chrome with
traffic-light dots, padding, drop shadow, backgrounds, and the
owner's typography law stamped on presented surfaces
("Powered by peacethefirst").

The render substrate is Devon's own: the Studio's deterministic
image-document flow (the card lands as an :class:`ImageDocument`
whose source is the rendered PNG, so Studio effects continue from
it), and the typography service's stocked faces (JetBrains Mono
for code, Montserrat for the brand mark — the policy's ``mono``
and ``brandmark`` contexts, resolved through the service door,
never a host font lottery).

Determinism is the contract: the same card renders the same PNG
bytes on every replay (``render_hash`` — SHA-256 of the PNG),
proven by test. Theme breadth is polish, not system; the contract
is deterministic rendering from real tokenisation. Fonts on the
box bound the look — a face the service cannot stock is named in
the render notes, never silently substituted.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

from ...core.branding import BRAND_TEXT
from . import ImgGenError

__all__ = [
    "CodeCard",
    "CodeCardRender",
    "CodeCardError",
    "THEMES",
    "DEFAULT_THEME",
    "MAX_CODE_LINES",
    "MAX_WRAP_COLUMNS",
    "list_themes",
    "resolve_language",
    "tokenize",
    "render_code_image",
    "render_card",
    "PRESETS",
    "DEFAULT_PRESET",
    "list_presets",
]

#: Pygments is the highlighting rung: the lexer's own token stream
#: drives both the colours and the proof. It is a hard dependency
#: of this organ — without it there is no honest highlighting, so
#: the import failure is raised by name instead of a fake render.
try:
    from pygments import lex as _pyg_lex
    from pygments.lexers import TextLexer, get_lexer_by_name
    from pygments.styles import get_style_by_name
    from pygments.token import Token as _Token
    from pygments.util import ClassNotFound as _ClassNotFound

    _PYGMENTS_ERROR: str | None = None
except ImportError as _exc:  # pragma: no cover - pygments absent
    _pyg_lex = None  # type: ignore[assignment]
    get_lexer_by_name = None  # type: ignore[assignment]
    get_style_by_name = None  # type: ignore[assignment]
    TextLexer = None  # type: ignore[assignment]
    _Token = None  # type: ignore[assignment]
    _ClassNotFound = Exception  # type: ignore[assignment]
    _PYGMENTS_ERROR = str(_exc)


class CodeCardError(ImgGenError):
    """A code card that cannot be rendered — named, never silent."""


#: theme name -> (Pygments style, outer background, label).
#: The code pane's own background is the style's background_color;
#: ``outer`` is the backdrop the window floats on. Every style here
#: is a real Pygments style in the installed version — a name
#: Pygments does not ship (``github-light`` was tried and does not
#: exist in 2.21) never enters this table.
THEMES: dict[str, dict[str, str]] = {
    "dracula": {"style": "dracula", "outer": "#15161d",
                "label": "Dracula"},
    "monokai": {"style": "monokai", "outer": "#1b1c17",
                "label": "Monokai"},
    "github-dark": {"style": "github-dark", "outer": "#010409",
                    "label": "GitHub Dark"},
    "one-dark": {"style": "one-dark", "outer": "#1e2127",
                 "label": "One Dark"},
    "nord": {"style": "nord", "outer": "#242933", "label": "Nord"},
    "solarized-dark": {"style": "solarized-dark", "outer": "#073642",
                       "label": "Solarized Dark"},
    "gruvbox-dark": {"style": "gruvbox-dark", "outer": "#1d2021",
                     "label": "Gruvbox Dark"},
    "solarized-light": {"style": "solarized-light", "outer": "#eee8d5",
                        "label": "Solarized Light"},
}

DEFAULT_THEME = "dracula"

#: The Ray.so fold (52-websites #20): one system serves both
#: Carbon and Ray — the preset pack on this render surface.
#: ``carbon`` is the full knob set (every explicit parameter is
#: honoured as passed). A ``ray*`` preset is opinionated: fewer
#: knobs, stronger defaults — the preset fixes the theme, the
#: frame, the chrome, the watermark, the line numbers and the
#: font size, and paints a gradient backdrop (Ray's
#: differentiation: curated themes, gradient backgrounds,
#: dark/light frames, padding). An explicit parameter that
#: disagrees with a ray preset is overridden and the override
#: is named in the render notes — never silently dropped.
#: ``gradient`` is (top, bottom) hex; None means the theme's
#: solid ``outer``.
PRESETS: dict[str, dict[str, object]] = {
    "carbon": {"label": "Carbon — full knobs", "theme": None,
               "gradient": None, "chrome": None, "watermark": None,
               "line_numbers": None, "font_size": None},
    "ray": {"label": "Ray — dark frame", "theme": "dracula",
            "gradient": ("#2b2d5c", "#14152a"), "chrome": True,
            "watermark": True, "line_numbers": True, "font_size": 15},
    "ray-light": {"label": "Ray Light — light frame",
                  "theme": "solarized-light",
                  "gradient": ("#fdf6e3", "#e8e0c8"), "chrome": True,
                  "watermark": True, "line_numbers": True,
                  "font_size": 15},
    "ray-midnight": {"label": "Ray Midnight — dark frame",
                     "theme": "one-dark",
                     "gradient": ("#1b2a4a", "#0b1020"),
                     "chrome": True, "watermark": True,
                     "line_numbers": True, "font_size": 15},
}

DEFAULT_PRESET = "carbon"

#: A static image of more code than this is unreadable — the cap is
#: a budget, named in the refusal, not a silent truncation.
MAX_CODE_LINES = 500
#: Visual columns before a logical line wraps (continuation lines
#: indent, carry no line number, and keep their token colours).
MAX_WRAP_COLUMNS = 120

#: ANSI colour names a Pygments style may carry instead of hex —
#: mapped to the same hues the terminal would show.
_ANSI_HEX = {
    "black": "#000000", "red": "#aa0000", "green": "#00aa00",
    "yellow": "#aa5500", "blue": "#0000aa", "magenta": "#aa00aa",
    "cyan": "#00aaaa", "white": "#aaaaaa",
    "brightblack": "#555555", "brightred": "#ff5555",
    "brightgreen": "#55ff55", "brightyellow": "#ffff55",
    "brightblue": "#5555ff", "brightmagenta": "#ff55ff",
    "brightcyan": "#55ffff", "brightwhite": "#ffffff",
}

#: Traffic-light dots, Carbon's window chrome.
_DOTS = ("#ff5f57", "#febc2e", "#28c840")


@dataclass(frozen=True)
class CodeCard:
    """A code presentation image, as data — the Carbon shape.

    ``watermark`` follows the owner's typography law: the
    "Powered by peacethefirst" stamp rides on presented surfaces
    unless explicitly switched off. ``preset`` is the Ray.so
    fold: "carbon" honours every knob as set; a "ray*" preset
    fixes the look (theme, frame, chrome, watermark, line
    numbers, font size, gradient backdrop) and names any knob
    it overrode.
    """

    code: str
    language: str = ""
    theme: str = DEFAULT_THEME
    chrome: bool = True
    watermark: bool = True
    title: str = ""
    font_size: int = 15
    line_numbers: bool = True
    preset: str = DEFAULT_PRESET


@dataclass
class CodeCardRender:
    """A rendered card: the pixels plus the record of how they
    were made. ``render_hash`` is SHA-256 of the PNG bytes —
    the same card renders the same hash on every replay."""

    image: object  # PIL Image
    render_hash: str
    theme: str
    style: str
    language: str
    lexer: str
    font: str
    width: int
    height: int
    lines: int
    tokens: int
    preset: str = DEFAULT_PRESET
    gradient: tuple[str, str] | None = None
    notes: list = field(default_factory=list)


def list_themes() -> list[dict[str, str]]:
    """Every theme, with its Pygments style and backgrounds."""
    out = []
    for name, spec in THEMES.items():
        style = get_style_by_name(spec["style"])
        out.append({"name": name, "label": spec["label"],
                    "style": spec["style"],
                    "background": str(style.background_color),
                    "outer": spec["outer"]})
    return out


def list_presets() -> list[dict[str, object]]:
    """Every preset: the pack Ray.so folds into this surface."""
    return [{"name": name,
             "label": str(spec["label"]),
             "theme": spec["theme"],
             "gradient": spec["gradient"],
             "chrome": spec["chrome"],
             "watermark": spec["watermark"],
             "line_numbers": spec["line_numbers"],
             "font_size": spec["font_size"]}
            for name, spec in PRESETS.items()]


def resolve_language(language: str) -> tuple[object, str, bool, str]:
    """(lexer, canonical name, plain?, note).

    An unknown language never refuses the render: it falls back
    to the plain-text lexer and the note names the miss.
    """
    if _PYGMENTS_ERROR is not None:
        raise CodeCardError(
            f"code cards need Pygments (not installed: "
            f"{_PYGMENTS_ERROR})")
    want = (language or "").strip()
    if not want:
        return TextLexer(), "text", True, ""
    try:
        lexer = get_lexer_by_name(want)
    except _ClassNotFound:
        return (TextLexer(), "text", True,
                f"no lexer for {want!r} — rendered as plain text")
    return lexer, str(lexer.name), False, ""


def tokenize(code: str, language: str = "") -> list[tuple[str, str]]:
    """The lexer's own token stream: [(token-type name, text)].

    The proof surface: a fixture snippet's tokens here must match
    Pygments' own lexer output exactly — the highlighter is the
    authority, never a hand-rolled approximation.
    """
    lexer, _, _, _ = resolve_language(language)
    return [(str(ttype), text) for ttype, text in _pyg_lex(code, lexer)]


def _hex(color: str | None, ansicolor: str | None,
         default: str) -> str:
    """A style colour as ``#rrggbb`` — hex direct, ANSI mapped,
    missing falls back to the default."""
    if color:
        return "#" + color.lstrip("#")
    if ansicolor:
        return _ANSI_HEX.get(ansicolor.lower(), default)
    return default


def _opt_hex(color: str | None, ansicolor: str | None,
             ) -> str | None:
    """A style colour as ``#rrggbb``, or None when the style
    sets none — for token backgrounds (an Error token's red
    wash is real style data, painted, not dropped)."""
    if color:
        return "#" + color.lstrip("#")
    if ansicolor:
        return _ANSI_HEX.get(ansicolor.lower())
    return None


def _mono_fonts(size: int
                ) -> tuple[object, object, object, object, str, str]:
    """(regular, bold, italic, bold-italic, family name, note)
    through the typography service's ``mono`` context — the
    stocked JetBrains Mono faces. A face the service cannot stock
    falls back to the regular face and is named in the note (""
    when nothing fell back); a total miss falls back to Pillow's
    bitmap default, named as such.
    """
    from PIL import ImageFont

    from ..typography import get_service, pil_font_for

    service = get_service()
    note = ""
    try:
        regular = pil_font_for(service, "mono", size)
        family = str(regular.getname()[0])
    except Exception as exc:  # noqa: BLE001 - named fallback
        note = (f"mono font unavailable ({exc}) — Pillow default "
                "carries the card")
        default = ImageFont.load_default()
        return default, default, default, default, "Pillow default", note

    def _face(**kw: object) -> object:
        nonlocal note
        try:
            return pil_font_for(service, "mono", size, **kw)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - named fallback
            if not note:
                note = (f"mono face {kw} unavailable ({exc}) — "
                        "regular carries those tokens")
            return regular

    bold = _face(weight=700)
    italic = _face(italic=True)
    bold_italic = _face(weight=700, italic=True)
    return regular, bold, italic, bold_italic, family, note


def _brand_font(size: int) -> object:
    """The brand-mark face (Montserrat) for the footer stamp."""
    from PIL import ImageFont

    from ..typography import get_service, pil_font_for

    try:
        return pil_font_for(get_service(), "brandmark", size,
                            weight=700)
    except Exception:  # noqa: BLE001 - the stamp still lands
        return ImageFont.load_default()


def _styled_lines(code: str, lexer: object, style: object,
                  default_fg: str) -> list[list[tuple[str, str, bool,
                                                      bool, str | None]]]:
    """Token stream → per-logical-line segments:
    (text, fg hex, bold, italic, bg hex or None). Tabs expand
    to 4 spaces so the visual columns are exact; newlines
    split the lines."""
    lines: list[list[tuple[str, str, bool, bool, str | None]]] = [[]]
    for ttype, text in _pyg_lex(code, lexer):
        spec = style.style_for_token(ttype)
        fg = _hex(spec.get("color"), spec.get("ansicolor"), default_fg)
        bold = bool(spec.get("bold"))
        italic = bool(spec.get("italic"))
        bg = _opt_hex(spec.get("bgcolor"), spec.get("bgansicolor"))
        parts = text.replace("\t", "    ").split("\n")
        for i, part in enumerate(parts):
            if i:
                lines.append([])
            if part:
                lines[-1].append((part, fg, bold, italic, bg))
    while len(lines) > 1 and not lines[-1]:
        lines.pop()
    return lines


def _wrap(lines: list[list[tuple[str, str, bool, bool, str | None]]],
          width: int) -> list[tuple[int | None,
                                    list[tuple[str, str, bool,
                                               bool, str | None]]]]:
    """Wrap logical lines at ``width`` visual columns.
    Returns [(line number or None for continuations, segments)]."""
    out: list[tuple[int | None,
                    list[tuple[str, str, bool, bool, str | None]]]] = []
    for num, segments in enumerate(lines, 1):
        # Flatten to one char per style slot, then re-run.
        chars: list[tuple[str, str, bool, bool, str | None]] = []
        for text, fg, bold, italic, bg in segments:
            chars.extend((ch, fg, bold, italic, bg) for ch in text)
        if not chars:
            out.append((num, []))
            continue
        start = 0
        first = True
        while start < len(chars):
            # A continuation line spends 4 of its columns on the
            # indent, so it consumes fewer source chars — the
            # advance matches what was actually shown.
            take = width if first else width - 4
            chunk = chars[start:start + take]
            if not first:
                chunk = [(" ", "#000000", False, False, None)] * 4 \
                    + chunk
            segs: list[tuple[str, str, bool, bool, str | None]] = []
            for ch, fg, bold, italic, bg in chunk:
                if segs and segs[-1][1:] == (fg, bold, italic, bg):
                    segs[-1] = (segs[-1][0] + ch, fg, bold, italic,
                                bg)
                else:
                    segs.append((ch, fg, bold, italic, bg))
            out.append((num if first else None, segs))
            start += take
            first = False
    return out


def _luminance(hex_color: str) -> float:
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _muted(hex_color: str) -> str:
    """A muted version of a colour for gutters/titles: halfway
    toward the background's opposite pole, so it reads on both
    dark and light themes."""
    lum = _luminance(hex_color)
    target = 0.85 if lum < 0.5 else 0.15
    out = []
    for i in (1, 3, 5):
        v = int(hex_color[i:i + 2], 16) / 255.0
        out.append(round((v + target) / 2 * 255))
    return f"#{out[0]:02x}{out[1]:02x}{out[2]:02x}"


def render_code_image(code: str, *, language: str = "",
                      theme: str | None = None,
                      preset: str = DEFAULT_PRESET,
                      chrome: bool | None = None,
                      watermark: bool | None = None, title: str = "",
                      font_size: int | None = None,
                      line_numbers: bool | None = None
                      ) -> CodeCardRender:
    """Render a code card to a PIL image — deterministic.

    ``code`` is the exact text (newlines preserved, tabs expanded
    to 4 spaces). ``language`` is a Pygments lexer name or alias
    ("python", "py", "js", ...); unknown falls back to plain text
    and says so in the notes. ``theme`` is one of
    :func:`list_themes`; unknown falls back to the default and
    says so. ``preset`` is one of :func:`list_presets`:
    "carbon" honours every knob as passed (a knob left
    unspecified takes the Carbon default — dracula theme,
    chrome/watermark/line numbers on, 15pt); a "ray*" preset
    fixes the look — its theme, gradient backdrop, chrome,
    watermark, line numbers and font size win over any knob
    that disagrees, and the override is named in the notes.
    The PNG bytes hash to ``render_hash`` — the same card
    renders the same hash on every replay.
    """
    from PIL import Image, ImageDraw, ImageFilter

    if _PYGMENTS_ERROR is not None:
        raise CodeCardError(
            f"code cards need Pygments (not installed: "
            f"{_PYGMENTS_ERROR})")
    text = (code or "")
    if not text.strip():
        raise CodeCardError("code card needs code — empty input")
    preset_name = (preset or "").strip() or DEFAULT_PRESET
    pspec = PRESETS.get(preset_name)
    if pspec is None:
        raise CodeCardError(
            f"unknown code preset {preset_name!r}; use one of "
            f"{sorted(PRESETS)}")
    notes: list[str] = []
    # A ray preset is opinionated: its fixed knobs win, and an
    # explicit knob that disagrees is named, never silently
    # dropped. Carbon honours everything as passed.
    gradient: tuple[str, str] | None = None
    if preset_name != DEFAULT_PRESET:
        gradient = pspec["gradient"]  # type: ignore[assignment]
        for knob in ("theme", "chrome", "watermark",
                     "line_numbers", "font_size"):
            fixed = pspec[knob]
            given = {"theme": theme, "chrome": chrome,
                     "watermark": watermark,
                     "line_numbers": line_numbers,
                     "font_size": font_size}[knob]
            if given is not None and given != fixed:
                notes.append(
                    f"preset {preset_name!r} fixes {knob} to "
                    f"{fixed!r}; your {knob}={given!r} was not used")
        theme = pspec["theme"]  # type: ignore[assignment]
        chrome = pspec["chrome"]  # type: ignore[assignment]
        watermark = pspec["watermark"]  # type: ignore[assignment]
        line_numbers = pspec["line_numbers"]  # type: ignore[assignment]
        font_size = pspec["font_size"]  # type: ignore[assignment]
    if font_size is None:
        font_size = 15
    if int(font_size) < 10:
        raise CodeCardError(
            f"font_size must be >= 10, got {font_size}")
    if int(font_size) > 28:
        raise CodeCardError(
            f"font_size must be <= 28, got {font_size}")
    size = int(font_size)
    if chrome is None:
        chrome = True
    if watermark is None:
        watermark = True
    if line_numbers is None:
        line_numbers = True
    # A trailing newline ends the last line, it is not another
    # line: "a\nb\n" is two lines, not three. Trailing blank
    # lines carry no content for a card and are not rendered.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    n_lines = len(text.rstrip("\n").split("\n")) if text.strip() else 0
    if n_lines > MAX_CODE_LINES:
        raise CodeCardError(
            f"code card holds at most {MAX_CODE_LINES} lines, "
            f"got {n_lines} — split the snippet or render "
            "it in parts")

    theme_name = (theme or "").strip() or DEFAULT_THEME
    spec = THEMES.get(theme_name)
    if spec is None:
        notes.append(f"unknown theme {theme_name!r} — "
                     f"rendered as {DEFAULT_THEME!r}")
        theme_name = DEFAULT_THEME
        spec = THEMES[theme_name]
    style = get_style_by_name(spec["style"])
    bg = str(style.background_color)
    outer = spec["outer"]
    default_fg = _hex(style.style_for_token(_Token.Text).get("color"),
                      style.style_for_token(_Token.Text).get("ansicolor"),
                      "#f8f8f2" if _luminance(bg) < 0.5 else "#24292e")
    lexer, lexer_name, _plain, lang_note = resolve_language(language)
    if lang_note:
        notes.append(lang_note)

    regular, bold, italic, bold_italic, family, font_note = \
        _mono_fonts(size)
    fonts = {(False, False): regular, (True, False): bold,
             (False, True): italic, (True, True): bold_italic}
    if font_note:
        notes.append(font_note)

    styled = _styled_lines(text, lexer, style, default_fg)
    visual = _wrap(styled, MAX_WRAP_COLUMNS)
    n_tokens = sum(1 for _tt, _tx in _pyg_lex(text, lexer))

    # -- metrics (monospace: one advance for every face) ---------
    draw_probe = Image.new("RGB", (8, 8))
    probe = ImageDraw.Draw(draw_probe)
    char_w = max(
        int(probe.textlength("M", font=f)) for f in fonts.values())
    line_h = size + 6
    digits = len(str(max(1, n_lines)))
    gutter_w = (digits * char_w + 18) if line_numbers else 0
    code_w = max(
        (sum(len(seg[0]) for seg in segs) for _, segs in visual),
        default=1) * char_w
    pad_x, pad_top, pad_bottom = 20, 16, 16
    title_h = 40 if chrome else 0
    footer_h = 26 if watermark else 0
    outer_pad = 28
    win_w = pad_x * 2 + gutter_w + code_w
    win_h = title_h + pad_top + len(visual) * line_h + pad_bottom
    width = win_w + outer_pad * 2
    height = win_h + outer_pad * 2 + footer_h

    # -- canvas: outer backdrop, shadow, window ------------------
    # A ray preset paints a gradient backdrop (its
    # differentiation); carbon keeps the theme's solid outer.
    if gradient is not None:
        top, bottom = gradient
        mask = Image.linear_gradient("L").resize((width, height))
        img = Image.composite(
            Image.new("RGB", (width, height), bottom),
            Image.new("RGB", (width, height), top), mask)
    else:
        img = Image.new("RGB", (width, height), outer)
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sh = ImageDraw.Draw(shadow)
    sh.rounded_rectangle(
        [outer_pad, outer_pad + 10, outer_pad + win_w,
         outer_pad + win_h + 10], radius=12, fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    img.paste(Image.alpha_composite(
        img.convert("RGBA"), shadow).convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [outer_pad, outer_pad, outer_pad + win_w,
         outer_pad + win_h], radius=12, fill=bg)
    # Clip the code pane to the window's rounded corners: the
    # pane is the window itself, so text stays inside by layout.

    # -- chrome: traffic-light dots + title ----------------------
    if chrome:
        cy = outer_pad + title_h // 2
        for i, dot in enumerate(_DOTS):
            cx = outer_pad + 22 + i * 22
            draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=dot)
        if (title or "").strip():
            brand = _brand_font(13)
            draw.text((outer_pad + win_w // 2, cy),
                      (title or "").strip(), font=brand,
                      fill=_muted(default_fg), anchor="mm")
        draw.line([outer_pad, outer_pad + title_h,
                   outer_pad + win_w, outer_pad + title_h],
                  fill=_muted(bg), width=1)

    # -- code: gutter numbers + token runs -----------------------
    comment = style.style_for_token(_Token.Comment)
    gutter_fg = _hex(comment.get("color"), comment.get("ansicolor"),
                     _muted(default_fg))
    y = outer_pad + title_h + pad_top
    for num, segs in visual:
        if line_numbers and num is not None:
            draw.text((outer_pad + pad_x + gutter_w - 8, y),
                      str(num), font=regular, fill=gutter_fg,
                      anchor="ra")
        x = outer_pad + pad_x + gutter_w
        for text_run, fg, is_bold, is_italic, bg in segs:
            font = fonts[(is_bold, is_italic)]
            run_w = int(probe.textlength(text_run, font=font))
            if bg is not None:
                # A style-set token background (an Error token's
                # red wash) is real style data — painted behind
                # the run, never dropped.
                draw.rectangle([x, y, x + run_w, y + line_h],
                               fill=bg)
            draw.text((x, y), text_run, font=font, fill=fg)
            x += run_w
        y += line_h

    # -- footer stamp: the owner's typography law ----------------
    if watermark:
        brand = _brand_font(12)
        draw.text((width - outer_pad, height - 8), BRAND_TEXT,
                  font=brand, fill=_muted(default_fg), anchor="rb")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    return CodeCardRender(
        image=img, render_hash=digest, theme=theme_name,
        style=spec["style"], language=lexer_name,
        lexer=str(getattr(lexer, "name", lexer_name)),
        font=family, width=width, height=height,
        lines=n_lines, tokens=n_tokens, preset=preset_name,
        gradient=gradient, notes=notes)


def render_card(card: CodeCard) -> CodeCardRender:
    """Render a :class:`CodeCard` — the data shape's own entry."""
    return render_code_image(
        card.code, language=card.language, theme=card.theme,
        preset=card.preset, chrome=card.chrome,
        watermark=card.watermark, title=card.title,
        font_size=card.font_size,
        line_numbers=card.line_numbers)
