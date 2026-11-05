"""Typography organ (DESIGN §27): how Devon's group surfaces
LOOK — one shared layer every presented message renders through,
never ad-hoc strings per handler.

The owner rulings this encodes (2026-11-03):

- **Fonts are structure.** A reply is composed from semantic
  primitives — title, stat rows, money, flavor quote, footer
  note — and the primitives carry the hierarchy (bold titles,
  tabular mono amounts, blockquote flavor, italic footers) on
  every platform.  A flat wall of text is the defect this organ
  exists to kill.
- **NOT a hardcoded style.** The look is a THEME SYSTEM: named
  packs map each surface class to a preset; a group overrides
  the pack, a single class, the stamp text, or the stamp
  classes from its own settings at runtime.  Handlers name WHAT
  renders (a :class:`Card` of a surface class); resolution
  decides HOW.
- **The stamp.** Presented surfaces (economy panels, boards,
  briefings, announcements, game results, leaderboards) close
  with the owner's footer signature — default text
  ``"Powered by peacethefirst"`` — appended HERE, by the
  rendering path, exactly once.  Conversational replies are
  never stamped: a signature, never spam.
- **Games get lines.** The ``board`` preset: divider rules and
  aligned mono grids, growing the §12 emoji-board style.  Other
  classes default to the ``clean`` preset: whitespace and
  structure, no box-drawing frames.

The organ builds on :mod:`.present` (the Doc model + per-
transport renderers): a Card is shaped into a Doc under the
resolved preset, then rendered for the actual platform
(Telegram HTML / WhatsApp marks / Discord markdown / plain).
Surface-composed strings (welcome, rules, warnings) render the
same way from the group key's platform prefix; the gateway's
send path then passes them through untouched.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

from .present import Doc, KV, Rendered, Section
from .present import canonical_text as _canonical_text
from .present import render as _render_doc

__all__ = [
    "SURFACE_CLASSES", "PRESETS", "THEMES", "CLASS_ICONS",
    "DEFAULT_PRESENTATION", "DEFAULT_STAMP_TEXT",
    "STAMP_CLASSES", "ResolvedStyle", "resolve_style",
    "WATERMARK_MODES", "DEFAULT_WATERMARK_MODE",
    "MediaMark", "resolve_media_mark",
    "Card", "render_card", "render_card_framed", "card_text",
    "fmt_amount", "sparks", "moderation_warning", "notice_line",
]

#: The surface classes a handler may declare.  The class — not
#: the handler — decides preset and stamp treatment.
SURFACE_CLASSES = (
    "economy", "moderation", "game", "briefing", "announcement",
    "leaderboard", "settings", "chat", "menu",
)

#: Rendering presets (treatments).  ``clean``: hierarchy through
#: fonts and whitespace.  ``board``: divider rules + mono grids.
PRESETS = ("clean", "board")

#: One signpost icon per class (§27: the same icon in titles,
#: buttons and boards, so a member learns the map once).
CLASS_ICONS: dict[str, str] = {
    "economy": "🏦",
    "moderation": "🛡️",
    "game": "🎮",
    "briefing": "🗞️",
    "announcement": "📣",
    "leaderboard": "🏆",
    "settings": "⚙️",
    "chat": "💬",
    "menu": "📖",
}

#: The house footer signature (§19/§20's owner mark, message
#: form).  Groups may replace the text; the default is the
#: owner's name as he wrote it.
DEFAULT_STAMP_TEXT = "Powered by peacethefirst"

#: Classes stamped under the shipped theme packs: the PRESENTED
#: surfaces.  Moderation records, settings panels and chat are
#: not presentation — no stamp.
STAMP_CLASSES: tuple[str, ...] = (
    "economy", "game", "briefing", "announcement", "leaderboard",
    "menu",
)


@dataclass(frozen=True)
class ThemePack:
    """A named look: surface class → preset, plus the classes
    the stamp closes."""

    name: str
    class_presets: dict[str, str]
    stamp_classes: tuple[str, ...]


def _pack(name: str, game: str = "board", rest: str = "clean"
          ) -> ThemePack:
    presets = {cls: rest for cls in SURFACE_CLASSES}
    presets["game"] = game
    return ThemePack(name, presets, STAMP_CLASSES)


#: Shipped theme packs.  ``house`` is the default look (games
#: lined, everything else clean); ``clean`` drops the lines
#: everywhere; ``board`` lines everything.
THEMES: dict[str, ThemePack] = {
    "house": _pack("house"),
    "clean": _pack("clean", game="clean"),
    "board": _pack("board", game="board", rest="board"),
}

DEFAULT_THEME = "house"

#: The per-group settings block (group store ``presentation``).
#: ``classes`` overrides one class's preset; ``stamp_classes``
#: (when non-empty) replaces the pack's stamped classes;
#: ``stamp_text`` replaces the signature ("" = house default).
#: ``watermark`` is the §19 media-mark mode for the SAME
#: signature: off | text | media | both — "" means the §19
#: house default (``text``).  One settings home: the media
#: watermark is not a second system, it is this block's stamp
#: reaching images and files.
DEFAULT_PRESENTATION: dict[str, Any] = {
    "theme": DEFAULT_THEME,
    "classes": {},
    "stamp": True,
    "stamp_text": "",
    "stamp_classes": [],
    "watermark": "",
}

#: The §19 watermark modes.  ``text``: the signature rides the
#: caption only.  ``media``: generated media additionally
#: carries the image stamp.  ``both``: caption + image stamp.
#: ``off``: no media mark at all (the text-card stamp is
#: governed by ``stamp``/``stamp_classes`` as always).
WATERMARK_MODES: tuple[str, ...] = ("off", "text", "media",
                                    "both")
DEFAULT_WATERMARK_MODE = "text"

#: The house divider rule (the same vocabulary as style.py /
#: present.py's plain header rule — a rule line, never a frame).
DIVIDER = "━━━━━━━━━━━━━━━"

#: Flavor longer than this collapses into an expandable quote
#: on Telegram instead of walling the chat.
_EXPANDABLE_CHARS = 220


@dataclass(frozen=True)
class ResolvedStyle:
    """The resolved look for ONE card: theme pack, the preset
    its class maps to under the group's overrides, and whether
    the stamp closes it (and with what text)."""

    theme: str
    preset: str
    stamp: bool
    stamp_text: str
    surface_class: str


def _surface_class(value: Any) -> str:
    cls = str(value or "").strip().lower()
    return cls if cls in SURFACE_CLASSES else "chat"


def resolve_style(settings: dict[str, Any] | None,
                  surface_class: Any) -> ResolvedStyle:
    """Resolve a group's presentation settings to the concrete
    style for one surface class.  Settings are data: unknown
    theme/preset names fall back to the house defaults, never
    raise."""
    cls = _surface_class(surface_class)
    block = (settings or {}).get("presentation") or {}
    pack = THEMES.get(str(block.get("theme") or "").strip().lower())
    if pack is None:
        pack = THEMES[DEFAULT_THEME]
    preset = pack.class_presets.get(cls, "clean")
    override = (block.get("classes") or {}).get(cls)
    if override in PRESETS:
        preset = str(override)
    stamp_classes = block.get("stamp_classes") or pack.stamp_classes
    stamp = bool(block.get("stamp", True)) and cls in stamp_classes
    text = str(block.get("stamp_text") or "").strip() \
        or DEFAULT_STAMP_TEXT
    return ResolvedStyle(theme=pack.name, preset=preset,
                         stamp=stamp, stamp_text=text,
                         surface_class=cls)


@dataclass(frozen=True)
class MediaMark:
    """The §19 watermark decision for ONE piece of generated
    media: whether the image itself is stamped, and the
    signature text for the caption ("" = no caption line).
    ``image_text`` is the signature the image stamp draws —
    resolved even in ``media`` mode, where the caption stays
    bare.  ``mode`` is the resolved mode, kept for journaling.
    """

    image: bool
    text: str
    mode: str
    image_text: str = ""


def resolve_media_mark(settings: dict[str, Any] | None,
                       surface_class: Any, *,
                       mark_text: str = "") -> MediaMark:
    """Resolve the media watermark for one surface class from
    the group's presentation block — the same home as the text
    stamp.  Rules (§19): conversational classes are never
    marked; the master ``stamp`` switch governs the whole mark;
    only classes the pack stamps can carry it; the mode then
    splits caption signature vs image stamp.  ``mark_text`` is
    the owner's global signature (§19: set once by the owner,
    applies everywhere) — it fills in only when the group has
    not set its own ``stamp_text``.  Settings are data: an
    unknown mode falls back to the house default, never raises.
    """
    cls = _surface_class(surface_class)
    block = (settings or {}).get("presentation") or {}
    style = resolve_style(settings, cls)
    mode = str(block.get("watermark") or "").strip().lower()
    if mode not in WATERMARK_MODES:
        mode = DEFAULT_WATERMARK_MODE
    if not style.stamp or mode == "off":
        return MediaMark(image=False, text="", mode=mode)
    text = str(block.get("stamp_text") or "").strip() \
        or str(mark_text or "").strip() or DEFAULT_STAMP_TEXT
    image = mode in ("media", "both")
    return MediaMark(
        image=image,
        text=text if mode in ("text", "both") else "",
        mode=mode,
        image_text=text if image else "")


# ── amounts ──────────────────────────────────────────────────────

def fmt_amount(value: Any) -> str:
    """A grouped, tabular amount: ``12500`` → ``"12,500"``."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def sparks(value: Any) -> str:
    """An amount in the group purse's unit, grouped."""
    return f"{fmt_amount(value)} sparks"


# ── the Card ─────────────────────────────────────────────────────

class Card:
    """One presented message, composed semantically.

    A handler declares the surface class and stacks blocks —
    prose lines, stat rows, a preformatted board, flavor — then
    the rendering path shapes them under the resolved preset.
    Amounts belong in :meth:`stats` (tabular, never buried in a
    sentence); long fine print belongs in :meth:`flavor` (a
    quote, expandable when long); :meth:`note` is the quiet
    italic footer line.  The stamp is NOT the handler's job.
    """

    def __init__(self, surface_class: Any, *, title: str = "",
                 icon: str = "", tone: str = "info") -> None:
        self.surface_class = _surface_class(surface_class)
        self.title = str(title or "")
        self.icon = str(icon or "")
        self.tone = str(tone or "info")
        self._summary: list[str] = []
        self._blocks: list[tuple] = []
        self._note = ""

    # -- composition ------------------------------------------------
    def line(self, text: Any) -> "Card":
        """One prose line (house dialect, transcoded per
        platform by the present layer).  Lines before the first
        section block form the summary; later lines accumulate
        into one lines block until another block intervenes."""
        if self._blocks:
            kind, title, payload = self._blocks[-1]
            if kind == "lines":
                payload.append(str(text))
            else:
                self._blocks.append(("lines", "", [str(text)]))
        else:
            self._summary.append(str(text))
        return self

    def lines(self, texts: Any) -> "Card":
        for text in texts or ():
            self.line(text)
        return self

    def block_lines(self, texts: Any, *, title: str = "") -> "Card":
        """A forced block of prose lines (a list panel: shop
        shelf, quest board) — its own section even when it is
        the card's first content."""
        self._blocks.append(
            ("lines", str(title or ""),
             [str(t) for t in texts or ()]))
        return self

    def stats(self, rows: Any, *, title: str = "") -> "Card":
        """Label/value rows.  Each row is ``(label, value)`` or
        ``(label, value, mono)``; numeric values are tabular
        (mono) by default."""
        norm = []
        for row in rows or ():
            if len(row) == 3:
                label, value, mono = row
            else:
                label, value = row
                mono = isinstance(value, (int, float)) \
                    and not isinstance(value, bool)
            norm.append((str(label), value, bool(mono)))
        self._blocks.append(("stats", str(title or ""), norm))
        return self

    def board(self, text: Any, *, title: str = "") -> "Card":
        """A preformatted block (a game board, a grid, a literal
        list): monospace, literal-escaped, columns intact on
        every platform."""
        self._blocks.append(
            ("board", str(title or ""), str(text or "")))
        return self

    def flavor(self, text: Any) -> "Card":
        """Flavor / fine print: rendered as a quote block,
        expandable on Telegram when long."""
        self._blocks.append(("flavor", "", str(text or "")))
        return self

    def note(self, text: Any) -> "Card":
        """The quiet footer line (italic on rich platforms)."""
        self._note = str(text or "")
        return self

    # -- shaping ------------------------------------------------------
    def _grid(self, rows: list[tuple]) -> list[str]:
        """Stat rows as an aligned mono grid (board preset):
        labels padded, values right-aligned."""
        pairs = [(label, _value_text(value))
                 for label, value, _mono in rows]
        if not pairs:
            return []
        lw = max(len(label) for label, _ in pairs)
        vw = max(len(value) for _, value in pairs)
        return [f"{label.ljust(lw)}  {value.rjust(vw)}"
                for label, value in pairs]

    def to_doc(self, preset: str) -> Doc:
        sections: list[Section] = []
        board = preset == "board"

        def add(section: Section) -> None:
            # Board preset: a rule under the head and between
            # sections — the lined look games read as boards.
            if board:
                sections.append(Section(lines=(DIVIDER,)))
            sections.append(section)

        for kind, title, payload in self._blocks:
            if kind == "stats":
                if board:
                    add(Section(title=title,
                                pre=tuple(self._grid(payload))))
                else:
                    add(Section(
                        title=title,
                        rows=tuple(KV(label, value, mono=mono)
                                   for label, value, mono
                                   in payload)))
            elif kind == "board":
                add(Section(title=title,
                            pre=tuple(str(payload).splitlines())))
            elif kind == "flavor":
                text = str(payload)
                add(Section(
                    quote=(text,),
                    expandable=len(text) > _EXPANDABLE_CHARS))
            elif kind == "lines":
                add(Section(title=title, lines=tuple(payload)))
        return Doc(
            title=self.title,
            icon=self.icon or CLASS_ICONS[self.surface_class],
            tone=self.tone,
            summary="\n".join(self._summary),
            sections=tuple(sections),
            footer=self._note)


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value == int(value) \
            and abs(value) < 1e15:
        return str(int(value))
    return str(value)


# ── rendering ────────────────────────────────────────────────────

def _dialect(platform: Any) -> str:
    """Mirror of present._family's dialect names (kept local so
    the stamp wrapper matches the renderer's own output)."""
    p = str(platform or "").strip().lower()
    if p.startswith("telegram") or p in ("tg", "tgbot", "tgpersonal"):
        return "html"
    if p in ("whatsapp", "wa"):
        return "whatsapp"
    if p.startswith("discord") or p in ("web", "webapp"):
        return "markdown"
    return "plain"


def _stamp_line(stamp_text: str, dialect: str) -> str:
    if dialect == "html":
        return f"<i>{_html.escape(stamp_text)}</i>"
    if dialect in ("whatsapp", "markdown", "canonical"):
        return f"_{stamp_text}_"
    return stamp_text


def _append_stamp(text: str, stamp_text: str, dialect: str) -> str:
    """The signature, appended by the rendering path — exactly
    once: a text that already closes with the rendered stamp
    line (a re-render of an edited message) is returned
    untouched."""
    if not text or not stamp_text:
        return text
    line = _stamp_line(stamp_text, dialect)
    if text.rstrip().endswith(line):
        return text
    return f"{text.rstrip()}\n\n{line}"


def render_card(card: Card, *, platform: Any = "telegram-bot",
                settings: dict[str, Any] | None = None) -> Rendered:
    """Render ``card`` for ``platform`` under the group's
    presentation ``settings`` (the full group settings dict, or
    None for house defaults)."""
    style = resolve_style(settings, card.surface_class)
    rendered = _render_doc(card.to_doc(style.preset), platform)
    text = rendered.text
    if style.stamp:
        text = _append_stamp(text, style.stamp_text,
                             _dialect(platform))
    return Rendered(text, rendered.parse_mode, rendered.platform)


def render_card_framed(card: Card, *,
                       settings: dict[str, Any] | None = None
                       ) -> Rendered:
    """The card in the XMD frame grammar (owner order
    2026-11-05 — the boxed WhatsApp menu format): one box for the
    card's head, one box per section, in the WhatsApp dialect.

    Same Card, same content, same stamp resolution as
    :func:`render_card` — only the framing differs.  Section
    bodies come from present's own section renderer, so a framed
    line can never drift from the flat render's line; the section
    title rides the frame (small-capped, presentation only)
    instead of a bold line inside the box.  Pure divider sections
    (the board preset's rules) are the frame's job here and are
    skipped.  WhatsApp only — every other platform keeps
    :func:`render_card`."""
    # present's own internals, reused so framed content IS the
    # flat render's content (same package, same layer).
    from .present import _literal as _present_literal
    from .present import _prose as _present_prose
    from .present import _section_lines as _present_section_lines
    from .style import small_caps, xmd_frame_bottom, xmd_frame_line, xmd_frame_top

    style = resolve_style(settings, card.surface_class)
    doc = card.to_doc(style.preset)
    dialect = "whatsapp"
    boxes: list[str] = []
    if doc.title or doc.resolved_icon:
        title = _present_literal(doc.title, dialect)
        if doc.summary:
            head_title = (f"{doc.resolved_icon} {small_caps(title)}"
                          if title else doc.resolved_icon)
            head = [xmd_frame_top(head_title)]
            for ln in _present_prose(doc.summary, dialect) \
                    .split("\n"):
                head.append(xmd_frame_line(ln))
            head.append(xmd_frame_bottom())
            boxes.append("\n".join(head))
        else:
            # No summary: the title is a masthead line (the flat
            # render's own head form), not an empty box.
            boxes.append(f"{doc.resolved_icon} *{title}*"
                         if title else doc.resolved_icon)
    for section in doc.sections:
        body = _present_section_lines(section, dialect)
        if section.title and body:
            # the title rides the frame, not a bold line inside
            body = body[1:]
        flat = [ln for chunk in body
                for ln in str(chunk).split("\n")
                if ln.strip() and ln.strip() != DIVIDER]
        if not flat:
            continue
        title = _present_literal(section.title, dialect)
        box = [xmd_frame_top(small_caps(title) if title else "")]
        box += [xmd_frame_line(ln) for ln in flat]
        box.append(xmd_frame_bottom())
        boxes.append("\n".join(box))
    text = "\n\n".join(boxes)
    if doc.footer:
        foot = _present_prose(doc.footer, dialect)
        if foot:
            text = f"{text}\n\n_{foot}_" if text else f"_{foot}_"
    if style.stamp:
        text = _append_stamp(text, style.stamp_text, dialect)
    return Rendered(text, "", "whatsapp")


def card_text(card: Card, *,
              settings: dict[str, Any] | None = None) -> str:
    """The canonical house-dialect render (for surfaces that
    compose text without a live platform handle — the gateway's
    F1 presentation converts it per platform at send, exactly
    like any producer string)."""
    style = resolve_style(settings, card.surface_class)
    text = _canonical_text(card.to_doc(style.preset))
    if style.stamp:
        text = _append_stamp(text, style.stamp_text, "canonical")
    return text


# ── shared one-line compositions ─────────────────────────────────

def moderation_warning(mention: str, rule_text: str,
                       consequence: str) -> str:
    """The defense warning line (group_defense.render_warning's
    composition): ``⛔ {who} — {rule}. **{consequence}**`` — the
    member's name and the plain rule lead (the shape the defense
    tests pin), the consequence carries the emphasis.  House
    dialect: the send path renders the bold per platform.
    Moderation class — never stamped."""
    return (f"⛔ {mention} — {rule_text}. "
            f"**{consequence}**")


def notice_line(icon: str, text: str) -> str:
    """A group event notice line in house dialect (season /
    quest completions riding the surface's notice channel)."""
    return f"{icon} {text}"
