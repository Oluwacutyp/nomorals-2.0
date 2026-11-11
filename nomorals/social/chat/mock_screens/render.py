"""The Pillow renderer foundation for mock screens.

One layout engine for the chat screen class: status bar →
header → conversation → input bar → navigation, drawn from
a :class:`~.schema.Screen` through the platform design
registry.  HD scale: geometry is laid out in logical pixels
and rendered at :data:`SCALE`× so the export is a crisp
phone-resolution image.

The renderer returns the PNG bytes plus a drawn record —
size, chrome regions, and one entry per element with its
box, kind and state — the makers idiom, so tests assert
structure instead of OCR-ing pixels.

Typography and avatar machinery are reused from
``makers`` (same L4 package): fonts come from the
typography service through ``makers._font``, wrapping is
measured, avatars are masked discs or deterministic
initials placeholders.

Real media (the media hand-off, :mod:`.media`): an
image message whose ``media`` reference (or the render's
``media`` hand-off, keyed by message id) resolves has its
REAL pixels composited into the photo frame — cover-fit,
the frame height adapting to the image's own aspect —
and a voice message with real audio draws the waveform
MEASURED from its samples under its true duration.  A
reference that cannot be loaded or decoded fails the
render with the reason; the placeholder scene and the
synthetic waveform remain only for messages carrying no
media at all.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .. import makers as _makers
from .designs import CHAT_DESIGNS, DEVICES, GROUP_NAME_COLORS
from .media import (
    VOICE_BARS,
    MediaError,
    format_duration,
    load_image,
    probe_audio,
)
from .schema import Message, Participant, Screen

#: Logical → device pixel scale.  A 390-wide iPhone layout
#: renders 780 px wide — phone resolution, not a thumbnail.
SCALE = 2


class MockScreensError(Exception):
    """A screen that cannot be built or rendered; the
    message IS the user-fit reason."""


_EMOJI_FONT_PATHS = (
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "C:\\Windows\\Fonts\\seguiemj.ttf",
)
_emoji_font_cache: dict = {}


def _emoji_tile(char: str, px: int):
    """Render one emoji as a colour tile when a colour
    emoji font exists on the host; None otherwise (the
    caller falls back to a text glyph)."""
    key = (char, px)
    if key in _emoji_font_cache:
        return _emoji_font_cache[key]
    tile = None
    for path in _EMOJI_FONT_PATHS:
        try:
            font = ImageFont.truetype(path, 109)
        except Exception:  # noqa: BLE001 - next candidate
            continue
        try:
            img = Image.new("RGBA", (136, 136), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((8, 8), char, font=font, embedded_color=True)
            bbox = img.getbbox()
            if bbox is not None:
                tile = img.crop(bbox).resize(
                    (px, px), Image.LANCZOS)
            break
        except Exception:  # noqa: BLE001 - no colour rung
            tile = None
            break
    _emoji_font_cache[key] = tile
    return tile


def _shade(color, frac: float):
    """Blend an RGB tuple toward black by ``frac``."""
    return tuple(max(0, min(255, int(c * (1.0 - frac))))
                 for c in color)


def _participant_color(person: Participant, screen: Screen):
    """A participant's name colour: their explicit colour,
    else the design palette by deterministic hash of id."""
    raw = (person.color or "").strip()
    if raw.startswith("#") and len(raw) == 7:
        try:
            return tuple(int(raw[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            pass
    digest = hashlib.sha256(
        (person.id or person.name or "?").encode("utf-8")).digest()
    return GROUP_NAME_COLORS[digest[0] % len(GROUP_NAME_COLORS)]


class _Canvas:
    """Layout state for one render: the scratch measure
    draw, fonts at device scale, and the drawn record."""

    def __init__(self, screen: Screen) -> None:
        self.screen = screen
        self.design = CHAT_DESIGNS[screen.platform]
        self.pal = self.design["dark" if screen.dark else "light"]
        self.dev = DEVICES[screen.chrome.style]
        self.w = int(self.dev["width"])
        self.intent = self.design.get("intent", "text")
        self._measure = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        #: real-media hand-off for this render: message id
        #: → bytes / path / PIL image / MediaRef.  Wins over
        #: the message's own ``media`` reference.
        self.media_handoff: dict = {}
        self._images: dict = {}
        self._probes: dict = {}
        #: message id → (logical media box, source) for the
        #: drawn record: source is "real" or "placeholder".
        self.media_regions: dict = {}

    # ── real media ───────────────────────────────────────

    def media_source(self, msg: Message):
        """The media source for one message: the render's
        hand-off by id, else the message's own ``media``
        reference, else None (the explicit no-media case)."""
        if msg.id in self.media_handoff:
            return self.media_handoff[msg.id]
        return msg.media or None

    def message_image(self, msg: Message):
        """The real image for an image message, or None
        when the message carries no media (the placeholder
        case).  A supplied reference that cannot be loaded
        fails the render with the reason — never a silent
        placeholder swap."""
        if msg.id in self._images:
            return self._images[msg.id]
        source = self.media_source(msg)
        if source is None:
            return None
        try:
            img = load_image(source)
        except MediaError as exc:
            raise MockScreensError(
                "the photo on message " + repr(msg.id)
                + " could not be used: " + str(exc)) from exc
        self._images[msg.id] = img
        return img

    def voice_probe(self, msg: Message):
        """The measured probe (true duration + derived
        bars) for a voice message with real audio, or None
        when it carries no media.  Undecodable audio fails
        the render with the reason."""
        if msg.id in self._probes:
            return self._probes[msg.id]
        source = self.media_source(msg)
        if source is None:
            return None
        try:
            probe = probe_audio(source, bars=VOICE_BARS)
        except MediaError as exc:
            raise MockScreensError(
                "the voice note on message " + repr(msg.id)
                + " could not be used: " + str(exc)) from exc
        self._probes[msg.id] = probe
        return probe

    def note_media_region(self, msg: Message, box: tuple,
                          source: str) -> None:
        self.media_regions[msg.id] = (box, source)

    def dflag(self, name: str, default=None):
        """A behaviour flag from the design entry (the
        registry carries each platform's choreography as
        data: where the time lives, whether bubbles tail,
        which input bar the app wears…)."""
        return self.design.get(name, default)

    def run_avatar_applies(self) -> bool:
        """Incoming runs end with the sender's avatar at
        the left on this design (Telegram: groups only)."""
        return bool(self.dflag("run_avatar", False)) and not (
            self.dflag("run_avatar_group_only", False)
            and not self.screen.group)

    def run_avatar_size(self) -> float:
        return float(self.dflag("run_avatar_size", 24))

    def s(self, value: float) -> int:
        return int(round(value * SCALE))

    def font(self, size: float, *, bold: bool = False):
        return _makers._font(self.s(size), bold=bold,
                             intent=self.intent)

    def wrap(self, text: str, size: float, max_w: float,
             *, bold: bool = False) -> list:
        return _makers._wrap(self._measure, text,
                             self.font(size, bold=bold),
                             self.s(max_w))

    def text_w(self, text: str, size: float, *,
               bold: bool = False) -> float:
        return self._measure.textlength(
            text, font=self.font(size, bold=bold)) / SCALE


def _draw_ticks(draw: ImageDraw.ImageDraw, cv: _Canvas,
                x: float, y: float, state: str) -> float:
    """Draw the delivery choreography at logical (x, y);
    returns the width consumed.  sending = a small clock;
    sent = one check; delivered = two; read = two in the
    design's read colour."""
    pal = cv.pal
    if state == "sending":
        r = 4.5
        draw.ellipse([cv.s(x), cv.s(y), cv.s(x + 2 * r),
                      cv.s(y + 2 * r)],
                     outline=pal["ticks"], width=cv.s(1.2))
        draw.line([cv.s(x + r), cv.s(y + r), cv.s(x + r),
                   cv.s(y + 1.5)], fill=pal["ticks"],
                  width=cv.s(1.2))
        draw.line([cv.s(x + r), cv.s(y + r), cv.s(x + r + 2.5),
                   cv.s(y + r)], fill=pal["ticks"],
                  width=cv.s(1.2))
        return 2 * r + 2
    color = pal["ticks_read"] if state == "read" else pal["ticks"]

    def check(cx: float) -> None:
        pts = [(cx, y + 5.2), (cx + 2.6, y + 7.8),
               (cx + 7.2, y + 1.2)]
        draw.line([(cv.s(px), cv.s(py)) for px, py in pts],
                  fill=color, width=cv.s(1.6))

    check(x)
    if state in ("delivered", "read"):
        check(x + 5.0)
        return 14.5
    return 9.5


def _draw_status_bar(draw, cv: _Canvas, y0: float) -> float:
    pal, dev, chrome = cv.pal, cv.dev, cv.screen.chrome
    h = float(dev["status_h"])
    draw.rectangle([0, 0, cv.s(cv.w), cv.s(h)], fill=pal["status_bg"])
    fg = pal["status_fg"]
    font = cv.font(13, bold=True)
    draw.text((cv.s(22 if dev["notch"] else 12), cv.s(6 if dev["notch"] else 3)),
              chrome.clock, font=font, fill=fg)
    if dev["notch"]:
        pill_w, pill_h = 108, 24
        x = (cv.w - pill_w) / 2
        draw.rounded_rectangle(
            [cv.s(x), cv.s(8), cv.s(x + pill_w), cv.s(8 + pill_h)],
            radius=cv.s(pill_h / 2), fill=(10, 10, 10))
    # right cluster: network label, signal bars, battery
    bx = cv.w - 14
    lvl = chrome.battery / 100.0
    bw, bh = 24, 11
    by = (h - bh) / 2
    draw.rounded_rectangle(
        [cv.s(bx - bw), cv.s(by), cv.s(bx), cv.s(by + bh)],
        radius=cv.s(3), outline=fg, width=cv.s(1))
    draw.rectangle(
        [cv.s(bx - bw + 2), cv.s(by + 2),
         cv.s(bx - bw + 2 + (bw - 4) * lvl), cv.s(by + bh - 2)],
        fill=fg)
    draw.rectangle(
        [cv.s(bx + 1.5), cv.s(by + 3.5), cv.s(bx + 3.5),
         cv.s(by + bh - 3.5)], fill=fg)
    bx -= bw + 8
    net = chrome.network or ""
    if net:
        nf = cv.font(11, bold=True)
        tw = draw.textlength(net, font=nf) / SCALE
        draw.text((cv.s(bx - tw), cv.s((h - 13) / 2)), net,
                  font=nf, fill=fg)
        bx -= tw + 6
    for i in range(4):
        bar_h = 4 + i * 2.2
        x = bx - 16 + i * 4.4
        draw.rectangle(
            [cv.s(x), cv.s(h / 2 + 5 - bar_h), cv.s(x + 3),
             cv.s(h / 2 + 5)], fill=fg)
    return h


def _draw_avatar(canvas: Image.Image, draw, cv: _Canvas,
                 x: float, y: float, size: float,
                 person: Participant | None, fallback: str) -> None:
    """Avatar disc: the participant's image when the ref
    resolves, else the makers initials placeholder in a
    deterministic colour."""
    name = person.name if person is not None else fallback
    handed = getattr(cv, "avatars", {}).get(
        person.id if person is not None else "")
    if handed is not None and _makers._paste_avatar_shape(
            canvas, (cv.s(x), cv.s(y)), cv.s(size),
            handed, "circle"):
        return
    ref = (person.avatar if person is not None else "") or ""
    if ref:
        from pathlib import Path

        if Path(ref).exists() and _makers._paste_avatar_shape(
                canvas, (cv.s(x), cv.s(y)), cv.s(size),
                ref, "circle"):
            return
    color = (_participant_color(person, cv.screen)
             if person is not None
             else _makers._medallion_color(name))
    _makers._draw_placeholder(
        draw, (cv.s(x), cv.s(y)), cv.s(size), name, color,
        shape="circle")


def _draw_header_center(canvas, draw, cv: _Canvas,
                        y0: float) -> float:
    """The iMessage header: blue back arrow (+ unread
    count) at the left, the contact's avatar over their
    name centred, a video icon at the right — the iOS
    Messages composition, not a left-aligned title row."""
    pal, screen = cv.pal, cv.screen
    h = float(cv.design["header_h"])
    draw.rectangle([0, cv.s(y0), cv.s(cv.w), cv.s(y0 + h)],
                   fill=pal["header_bg"])
    accent = pal["accent"]
    mid = y0 + h / 2
    draw.line([(cv.s(19), cv.s(mid - 8)), (cv.s(12), cv.s(mid)),
               (cv.s(19), cv.s(mid + 8))], fill=accent,
              width=cv.s(2.4))
    tx = 24
    if screen.chrome.unread:
        label = str(screen.chrome.unread)
        draw.text((cv.s(tx), cv.s(mid - 9)), label,
                  font=cv.font(15), fill=accent)
    person = _header_person(screen)
    if screen.group and (person is None or not person.avatar) \
            and not getattr(cv, "avatars", {}).get(
                person.id if person is not None else ""):
        person = None
    av = 32
    _draw_avatar(canvas, draw, cv, (cv.w - av) / 2, y0 + 5, av,
                 person, screen.title)
    name = _fit_text(cv, screen.title, 11.5, cv.w - 140)
    tw = cv.text_w(name, 11.5)
    draw.text((cv.s((cv.w - tw) / 2), cv.s(y0 + 40)), name,
              font=cv.font(11.5), fill=pal["header_fg"])
    vx = cv.w - 52
    draw.rounded_rectangle(
        [cv.s(vx), cv.s(mid - 7), cv.s(vx + 17), cv.s(mid + 7)],
        radius=cv.s(3), fill=accent)
    draw.polygon([(cv.s(vx + 19), cv.s(mid - 3)),
                  (cv.s(vx + 26), cv.s(mid - 8)),
                  (cv.s(vx + 26), cv.s(mid + 8)),
                  (cv.s(vx + 19), cv.s(mid + 3))], fill=accent)
    return h


def _draw_header(canvas, draw, cv: _Canvas, y0: float) -> float:
    if cv.dflag("header_center"):
        return _draw_header_center(canvas, draw, cv, y0)
    pal, screen = cv.pal, cv.screen
    h = float(cv.design["header_h"])
    draw.rectangle([0, cv.s(y0), cv.s(cv.w), cv.s(y0 + h)],
                   fill=pal["header_bg"])
    fg = pal["header_fg"]
    mid = y0 + h / 2
    # back arrow (+ iPhone unread badge count)
    ax = 12
    draw.line([(cv.s(ax + 7), cv.s(mid - 7)), (cv.s(ax), cv.s(mid)),
               (cv.s(ax + 7), cv.s(mid + 7))], fill=fg,
              width=cv.s(2.2))
    tx = ax + 12
    if screen.chrome.style == "iphone" and screen.chrome.unread:
        label = str(screen.chrome.unread)
        f = cv.font(15)
        draw.text((cv.s(tx), cv.s(mid - 9)), label, font=f, fill=fg)
        tx += cv.text_w(label, 15) + 4
    av = 38
    person = _header_person(screen)
    if screen.group and (person is None or not person.avatar) \
            and not getattr(cv, "avatars", {}).get(
                person.id if person is not None else ""):
        # A group wears its own initials until a group photo
        # is supplied — not its first member's face.
        person = None
    _draw_avatar(canvas, draw, cv, tx + 2, mid - av / 2, av,
                 person, screen.title)
    tx += av + 12
    limit_x = cv.w - (150 if screen.platform == "instagram"
                      else 118)
    title_f = cv.font(16, bold=True)
    draw.text((cv.s(tx), cv.s(y0 + (9 if screen.subtitle else 17))),
              _fit_text(cv, screen.title, 16, limit_x - tx,
                        bold=True),
              font=title_f, fill=fg)
    if screen.subtitle:
        draw.text((cv.s(tx), cv.s(y0 + 31)),
                  _fit_text(cv, screen.subtitle, 12,
                            limit_x - tx),
                  font=cv.font(12), fill=pal["header_sub"])
    # right icons: video (vector) + call (glyph) — in the
    # accent on designs whose header actions are coloured
    # (Messenger), in the header foreground elsewhere
    icon_fg = pal["accent"] if cv.dflag(
        "header_icons_accent") else fg
    ix = cv.w - 30
    draw.text((cv.s(ix), cv.s(mid - 10)), "☎",
              font=cv.font(17), fill=icon_fg)
    vx = ix - 42
    draw.rounded_rectangle(
        [cv.s(vx), cv.s(mid - 7), cv.s(vx + 17), cv.s(mid + 7)],
        radius=cv.s(3), fill=icon_fg)
    draw.polygon([(cv.s(vx + 19), cv.s(mid - 3)),
                  (cv.s(vx + 26), cv.s(mid - 8)),
                  (cv.s(vx + 26), cv.s(mid + 8)),
                  (cv.s(vx + 19), cv.s(mid + 3))], fill=icon_fg)
    if screen.platform == "instagram":
        draw.ellipse([cv.s(vx - 34), cv.s(mid - 10),
                      cv.s(vx - 14), cv.s(mid + 10)],
                     outline=fg, width=cv.s(1.6))
        draw.text((cv.s(vx - 28.5), cv.s(mid - 8)), "i",
                  font=cv.font(13, bold=True), fill=fg)
    return h


def _fit_text(cv: _Canvas, text: str, size: float,
              max_w: float, *, bold: bool = False) -> str:
    """Truncate with an ellipsis to a measured width — a
    header title must never run under the header icons."""
    if cv.text_w(text, size, bold=bold) <= max_w:
        return text
    out = text
    while out and cv.text_w(out + "…", size, bold=bold) > max_w:
        out = out[:-1]
    return out + "…" if out else "…"


def _header_person(screen: Screen) -> Participant | None:
    """The header avatar: the non-me participant of a 1:1,
    else the first participant (the group's own photo
    stands as its first member's in this phase)."""
    for person in screen.participants:
        if person.id != "me":
            return person
    return screen.participants[0] if screen.participants else None


def _draw_mic(draw, cv: _Canvas, cx: float, cy: float,
              color, *, r: float = 4.0) -> None:
    """A microphone glyph as vectors (capsule + arc +
    stem), centred at logical (cx, cy)."""
    draw.rounded_rectangle(
        [cv.s(cx - r), cv.s(cy - 8), cv.s(cx + r),
         cv.s(cy + 4)], radius=cv.s(r), fill=color)
    draw.arc([cv.s(cx - r - 5), cv.s(cy - 4), cv.s(cx + r + 5),
              cv.s(cy + 12)], start=0, end=180, fill=color,
             width=cv.s(1.8))
    draw.line([cv.s(cx), cv.s(cy + 10), cv.s(cx), cv.s(cy + 15)],
              fill=color, width=cv.s(1.8))


def _draw_camera(draw, cv: _Canvas, cx: float, cy: float,
                 color, *, w: float = 20.0) -> None:
    """A camera glyph as vectors (body + lens), centred
    at logical (cx, cy)."""
    h = w * 0.72
    draw.rounded_rectangle(
        [cv.s(cx - w / 2), cv.s(cy - h / 2), cv.s(cx + w / 2),
         cv.s(cy + h / 2)], radius=cv.s(3), outline=color,
        width=cv.s(1.8))
    draw.ellipse([cv.s(cx - 4), cv.s(cy - 4), cv.s(cx + 4),
                  cv.s(cy + 4)], outline=color, width=cv.s(1.8))
    draw.rectangle([cv.s(cx - 5), cv.s(cy - h / 2 - 2.5),
                    cv.s(cx + 1), cv.s(cy - h / 2 + 1)],
                   fill=color)


def _draw_input_bar_styled(draw, cv: _Canvas, y0: float,
                           style: str) -> float:
    """The non-WhatsApp/Instagram input bars, one per
    app's own language.  All return the shared 56 height."""
    pal = cv.pal
    h = 56.0
    fy = y0 + 8
    fh = 40
    mid = fy + fh / 2
    if style == "telegram":
        draw.text((cv.s(16), cv.s(fy + 10)), "☺",
                  font=cv.font(19), fill=pal["meta"])
        draw.text((cv.s(46), cv.s(fy + 11)), "Message",
                  font=cv.font(14.5), fill=pal["input_fg"])
        _draw_mic(draw, cv, cv.w - 26, mid, pal["meta"])
        return h
    if style == "messenger":
        draw.ellipse([cv.s(10), cv.s(fy + 8), cv.s(34),
                      cv.s(fy + 32)], outline=pal["accent"],
                     width=cv.s(1.8))
        draw.text((cv.s(17.5), cv.s(fy + 9.5)), "+",
                  font=cv.font(16, bold=True), fill=pal["accent"])
        _draw_camera(draw, cv, 52, mid, pal["accent"], w=19)
        draw.rounded_rectangle(
            [cv.s(72), cv.s(fy + 3), cv.s(cv.w - 48),
             cv.s(fy + fh - 3)], radius=cv.s((fh - 6) / 2),
            fill=pal["input_field"])
        draw.text((cv.s(84), cv.s(fy + 11)), "Aa",
                  font=cv.font(14), fill=pal["input_fg"])
        tile = _emoji_tile("👍", cv.s(22))
        canvas = getattr(cv, "image", None)
        if tile is not None and canvas is not None:
            canvas.paste(tile, (cv.s(cv.w - 36),
                                cv.s(fy + 9)), tile)
        return h
    if style == "snapchat":
        draw.ellipse([cv.s(8), cv.s(fy + 2), cv.s(44),
                      cv.s(fy + 38)], outline=pal["meta"],
                     width=cv.s(1.6))
        _draw_camera(draw, cv, 26, mid, pal["header_fg"], w=18)
        draw.rounded_rectangle(
            [cv.s(52), cv.s(fy + 3), cv.s(cv.w - 44),
             cv.s(fy + fh - 3)], radius=cv.s((fh - 6) / 2),
            outline=pal["meta"], width=cv.s(1.2))
        draw.text((cv.s(64), cv.s(fy + 11)), "Send a chat",
                  font=cv.font(14), fill=pal["input_fg"])
        draw.text((cv.s(cv.w - 34), cv.s(fy + 10)), "☺",
                  font=cv.font(18), fill=pal["meta"])
        return h
    if style == "imessage":
        _draw_camera(draw, cv, 24, mid, pal["meta"], w=20)
        label = "Text Message" if cv.screen.variant == "sms" \
            else "iMessage"
        draw.rounded_rectangle(
            [cv.s(44), cv.s(fy + 3), cv.s(cv.w - 40),
             cv.s(fy + fh - 3)], radius=cv.s((fh - 6) / 2),
            outline=pal["meta"], width=cv.s(1.2))
        draw.text((cv.s(56), cv.s(fy + 11)), label,
                  font=cv.font(14), fill=pal["input_fg"])
        _draw_mic(draw, cv, cv.w - 22, mid, pal["meta"],
                  r=3.4)
        return h
    # sms — Google Messages: one filled pill, + inside at
    # the left, an accent send/mic disc at the right
    draw.rounded_rectangle(
        [cv.s(8), cv.s(fy + 2), cv.s(cv.w - 56), cv.s(fy + fh - 2)],
        radius=cv.s((fh - 4) / 2), fill=pal["input_field"])
    draw.text((cv.s(18), cv.s(fy + 10)), "+",
              font=cv.font(17, bold=True), fill=pal["accent"])
    draw.text((cv.s(40), cv.s(fy + 11)), "Text message",
              font=cv.font(14), fill=pal["input_fg"])
    mx = cv.w - 28
    draw.ellipse([cv.s(mx - 17), cv.s(fy + 3), cv.s(mx + 17),
                  cv.s(fy + fh - 3)], fill=pal["accent"])
    _draw_mic(draw, cv, mx, mid, (255, 255, 255), r=3.4)
    return h


def _draw_input_bar(draw, cv: _Canvas, y0: float) -> float:
    pal = cv.pal
    h = 56.0
    draw.rectangle([0, cv.s(y0), cv.s(cv.w), cv.s(y0 + h)],
                   fill=pal["input_bg"])
    fy = y0 + 8
    fh = 40
    style = str(cv.dflag("input_style", "whatsapp"))
    if style not in ("whatsapp", "instagram"):
        return _draw_input_bar_styled(draw, cv, y0, style)
    if style == "instagram":
        draw.rounded_rectangle(
            [cv.s(10), cv.s(fy), cv.s(cv.w - 10), cv.s(fy + fh)],
            radius=cv.s(fh / 2), outline=pal["meta"],
            width=cv.s(1))
        draw.ellipse([cv.s(16), cv.s(fy + 6), cv.s(16 + 28),
                      cv.s(fy + 6 + 28)], fill=pal["accent"])
        draw.rectangle([cv.s(24), cv.s(fy + 15), cv.s(36),
                        cv.s(fy + 25)], fill=(255, 255, 255))
        draw.ellipse([cv.s(27.5), cv.s(fy + 17.5),
                      cv.s(32.5), cv.s(fy + 22.5)],
                     fill=pal["accent"])
        draw.text((cv.s(54), cv.s(fy + 11)), "Message...",
                  font=cv.font(14), fill=pal["input_fg"])
        draw.text((cv.s(cv.w - 44), cv.s(fy + 9)), "☺",
                  font=cv.font(18), fill=pal["header_fg"])
        return h
    # WhatsApp: field + green mic button
    draw.rounded_rectangle(
        [cv.s(8), cv.s(fy), cv.s(cv.w - 60), cv.s(fy + fh)],
        radius=cv.s(fh / 2), fill=pal["input_field"])
    draw.text((cv.s(20), cv.s(fy + 10)), "☺",
              font=cv.font(18), fill=pal["meta"])
    draw.text((cv.s(48), cv.s(fy + 11)), "Message",
              font=cv.font(14), fill=pal["input_fg"])
    mx = cv.w - 32
    draw.ellipse([cv.s(mx - 20), cv.s(fy), cv.s(mx + 20),
                  cv.s(fy + fh)], fill=pal["accent"])
    draw.rounded_rectangle(
        [cv.s(mx - 4), cv.s(fy + 9), cv.s(mx + 4),
         cv.s(fy + 21)], radius=cv.s(4), fill=(255, 255, 255))
    draw.arc([cv.s(mx - 9), cv.s(fy + 13), cv.s(mx + 9),
              cv.s(fy + 29)], start=0, end=180,
             fill=(255, 255, 255), width=cv.s(1.8))
    draw.line([cv.s(mx), cv.s(fy + 27), cv.s(mx), cv.s(fy + 32)],
              fill=(255, 255, 255), width=cv.s(1.8))
    return h


def _draw_nav(draw, cv: _Canvas, y0: float) -> float:
    dev = cv.dev
    h = float(dev["nav_h"])
    if dev["nav"] == "home":
        bw = 134
        x = (cv.w - bw) / 2
        draw.rounded_rectangle(
            [cv.s(x), cv.s(y0 + h - 9), cv.s(x + bw),
             cv.s(y0 + h - 4)], radius=cv.s(2.5),
            fill=cv.pal["meta"])
        return h
    draw.rectangle([0, cv.s(y0), cv.s(cv.w), cv.s(y0 + h)],
                   fill=cv.pal["nav_bg"])
    fg = cv.pal["nav_fg"]
    mid = y0 + h / 2
    for i, cx in enumerate((cv.w * 0.25, cv.w * 0.5, cv.w * 0.75)):
        if i == 0:
            draw.polygon([(cv.s(cx - 8), cv.s(mid)),
                          (cv.s(cx + 6), cv.s(mid - 8)),
                          (cv.s(cx + 6), cv.s(mid + 8))],
                         outline=fg, width=cv.s(1.6))
        elif i == 1:
            draw.ellipse([cv.s(cx - 8), cv.s(mid - 8),
                          cv.s(cx + 8), cv.s(mid + 8)],
                         outline=fg, width=cv.s(1.6))
        else:
            draw.rectangle([cv.s(cx - 7), cv.s(mid - 7),
                            cv.s(cx + 7), cv.s(mid + 7)],
                           outline=fg, width=cv.s(1.6))
    return h


# ── message elements ─────────────────────────────────────

_PAD_X = 9.0
_PAD_Y = 6.0
_TEXT_SIZE = 14.5
_LINE_H = 19.0


def _time_row_w(cv: _Canvas, msg: Message, with_ticks: bool) -> float:
    w = 0.0
    if msg.time:
        w += cv.text_w(msg.time, 10.5) + 5
    if with_ticks and msg.ticks:
        w += 15.0
    return w


def _quote_block(cv: _Canvas, msg: Message):
    """(height, lines, name, color) of the reply quote, or
    None.  Quotes resolve against the screen's own
    messages; an unresolvable quote is simply not drawn
    (the schema keeps the reference for later phases)."""
    if not msg.reply_to:
        return None
    target = cv.screen.message_by_id(msg.reply_to)
    if target is None:
        return None
    author = cv.screen.participant(target.sender)
    name = "You" if target.sender == "me" else (
        author.name if author is not None else target.sender)
    color = (_participant_color(author, cv.screen)
             if author is not None else cv.pal["accent"])
    return name, color, (target.text or target.caption or "📷 Photo")


def _measure_bubble(cv: _Canvas, msg: Message) -> tuple:
    """(width, height, parts) for one bubble element."""
    design = cv.design
    max_w = cv.w * float(design["max_bubble_frac"])
    text_max = max_w - 2 * _PAD_X
    own = msg.sender == "me"
    parts: dict = {"own": own}
    if msg.kind == "image":
        media_w = min(248.0, text_max)
        media_h = 168.0
        img = cv.message_image(msg)
        if img is not None:
            # a real photo wears its own aspect: the frame
            # height follows it, clamped to the range the
            # apps themselves use for chat photos
            media_h = max(96.0, min(
                260.0, media_w * img.height
                / max(1, img.width)))
            parts["real_media"] = True
        h = 4 + media_h
        w = media_w + 8
        if msg.caption:
            lines = cv.wrap(msg.caption, _TEXT_SIZE, media_w - 4)
            parts["caption_lines"] = lines
            h += len(lines) * _LINE_H + 2
        # Designs whose time lives in the bubble (WhatsApp,
        # Telegram, Android Messages) carry a time row under
        # media; Instagram-style bubbles never show times.
        h += (15 if cv.dflag("time_in_bubble") else 4) + _PAD_Y
        parts["media"] = (media_w, media_h)
        return w, h, parts
    if msg.kind == "voice":
        return min(238.0, max_w), 52.0, parts
    # text
    quote = _quote_block(cv, msg)
    h = _PAD_Y
    if quote is not None:
        name, _color, qtext = quote
        qlines = cv.wrap(qtext, 11, text_max - 16)[:2]
        parts["quote"] = (name, qlines)
        h += 14 + len(qlines) * 14.5 + 7
    if cv.screen.group and not own:
        author = cv.screen.participant(msg.sender)
        parts["name_line"] = (
            author.name if author is not None else msg.sender)
        h += 15
    lines = cv.wrap(msg.text, _TEXT_SIZE, text_max)
    parts["lines"] = lines
    h += len(lines) * _LINE_H + 3 + 13 + _PAD_Y
    widths = [cv.text_w(line, _TEXT_SIZE) for line in lines]
    if "name_line" in parts:
        widths.append(cv.text_w(parts["name_line"], 12.5, bold=True))
    if quote is not None:
        widths.append(cv.text_w(quote[0], 11.5, bold=True) + 14)
        widths.extend(cv.text_w(q, 11) + 14 for q in parts["quote"][1])
    widths.append(_time_row_w(
        cv, msg, with_ticks=bool(cv.dflag("ticks_in_bubble"))))
    w = max(widths + [40.0]) + 2 * _PAD_X
    return min(w, max_w), h, parts


def _measure_flat(cv: _Canvas, msg: Message) -> tuple:
    """(width, height, parts) for a Snapchat row: a caps
    name line, then the content at full width — Snapchat
    chat has no bubbles.  Reply quotes are not a Snapchat
    chat idiom and are not drawn on rows."""
    parts: dict = {"own": msg.sender == "me", "flat": True}
    w = cv.w - 12.0
    h = 17.0  # the caps name line
    if msg.kind == "image":
        media_w, media_h = 168.0, 112.0
        img = cv.message_image(msg)
        if img is not None:
            media_h = max(80.0, min(
                220.0, media_w * img.height
                / max(1, img.width)))
            parts["real_media"] = True
        parts["media"] = (media_w, media_h)
        h += media_h + 4
        if msg.caption:
            lines = cv.wrap(msg.caption, _TEXT_SIZE, cv.w - 38)
            parts["caption_lines"] = lines
            h += len(lines) * _LINE_H + 2
        return w, h + 6, parts
    if msg.kind == "voice":
        return w, h + 46, parts
    lines = cv.wrap(msg.text, _TEXT_SIZE, cv.w - 38)
    parts["lines"] = lines
    h += len(lines) * _LINE_H + 6
    return w, h, parts


def _draw_media_placeholder(canvas, draw, cv: _Canvas,
                            x: float, y: float, w: float,
                            h: float, seed: str) -> None:
    """A photo-shaped stand-in: a two-tone scene (sky, sun,
    hills) tinted deterministically from the message id —
    clearly a picture, never a claim of a real photo."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    top = (0x8E + digest[0] % 40, 0xB8, 0xD8)
    bot = (0x4A, 0x7A + digest[1] % 30, 0x5A)
    tile = Image.new("RGB", (cv.s(w), cv.s(h)))
    td = ImageDraw.Draw(tile, "RGBA")
    for i in range(cv.s(h)):
        frac = i / max(1, cv.s(h))
        col = tuple(int(top[c] + (bot[c] - top[c]) * frac)
                    for c in range(3))
        td.line([(0, i), (cv.s(w), i)], fill=col)
    td.ellipse([cv.s(w * 0.68), cv.s(h * 0.14),
                cv.s(w * 0.68 + 26), cv.s(h * 0.14 + 26)],
               fill=(255, 236, 170, 255))
    td.polygon([(0, cv.s(h)), (cv.s(w * 0.42), cv.s(h * 0.45)),
                (cv.s(w * 0.72), cv.s(h))], fill=(38, 84, 62, 235))
    td.polygon([(cv.s(w * 0.4), cv.s(h)), (cv.s(w * 0.78),
                cv.s(h * 0.55)), (cv.s(w), cv.s(h))],
               fill=(28, 66, 48, 235))
    mask = Image.new("L", (cv.s(w), cv.s(h)), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, cv.s(w), cv.s(h)], radius=cv.s(6), fill=255)
    canvas.paste(tile, (cv.s(x), cv.s(y)), mask)


def _draw_media_image(canvas, draw, cv: _Canvas,
                      msg: Message, x: float, y: float,
                      w: float, h: float) -> None:
    """The photo in an image message's frame: the REAL
    image cover-fit into the frame when the message
    carries media (centre-cropped to the frame's aspect,
    the chat idiom), the placeholder scene only when it
    carries none.  Either way the media region lands in
    the canvas record with its source."""
    img = cv.message_image(msg)
    if img is None:
        _draw_media_placeholder(canvas, draw, cv, x, y, w, h,
                                msg.id)
        cv.note_media_region(msg, (x, y, w, h), "placeholder")
        return
    tile = ImageOps.fit(img.convert("RGB"),
                        (max(1, cv.s(w)), max(1, cv.s(h))),
                        Image.LANCZOS)
    mask = Image.new("L", tile.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, tile.width, tile.height], radius=cv.s(6),
        fill=255)
    canvas.paste(tile, (cv.s(x), cv.s(y)), mask)
    cv.note_media_region(msg, (x, y, w, h), "real")


def _draw_reactions(canvas, draw, cv: _Canvas, msg: Message,
                    bx: float, by: float, bw: float) -> None:
    """The reaction pill overlapping the bubble's bottom
    edge: colour emoji tiles where the host has a colour
    emoji font, a glyph fallback where it does not."""
    if not msg.reactions:
        return
    own = msg.sender == "me"
    size = 13.0
    gap = 3.0
    pill_w = len(msg.reactions) * (size + gap) + 9
    pill_h = 19.0
    px = (bx + bw - pill_w - 4
          if cv.screen.platform == "instagram" and not own
          else bx + 6)
    py = by - pill_h / 2
    draw.rounded_rectangle(
        [cv.s(px), cv.s(py), cv.s(px + pill_w), cv.s(py + pill_h)],
        radius=cv.s(pill_h / 2), fill=cv.pal["reaction_bg"],
        outline=_shade(cv.pal["reaction_bg"], 0.18),
        width=cv.s(1))
    cx = px + 5
    for glyph in msg.reactions:
        tile = _emoji_tile(glyph, cv.s(size))
        if tile is not None:
            canvas.paste(tile, (cv.s(cx), cv.s(py + 3)), tile)
        else:
            draw.text((cv.s(cx), cv.s(py + 2)), glyph,
                      font=cv.font(size), fill=cv.pal["text_in"])
        cx += size + gap


def _gradient_bubble(canvas, cv: _Canvas, x: float, y: float,
                     w: float, h: float, radius: float) -> None:
    top = cv.pal.get("bubble_out_top", cv.pal["bubble_out"])
    bot = cv.pal["bubble_out"]
    tile = Image.new("RGB", (max(1, cv.s(w)), max(1, cv.s(h))))
    td = ImageDraw.Draw(tile)
    for i in range(tile.height):
        frac = i / max(1, tile.height)
        col = tuple(int(top[c] + (bot[c] - top[c]) * frac)
                    for c in range(3))
        td.line([(0, i), (tile.width, i)], fill=col)
    mask = Image.new("L", tile.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, tile.width, tile.height], radius=cv.s(radius),
        fill=255)
    canvas.paste(tile, (cv.s(x), cv.s(y)), mask)


def _draw_bubble(canvas, draw, cv: _Canvas, msg: Message,
                 parts: dict, x: float, y: float, w: float,
                 h: float, *, tail: bool) -> None:
    pal = cv.pal
    own = parts["own"]
    radius = float(cv.design["bubble_radius"])
    fill = pal["bubble_out"] if own else pal["bubble_in"]
    if own and cv.screen.variant == "sms" and cv.dflag(
            "bubble_out_sms"):
        # the iMessage Apple-to-Android variant: green
        # bubbles, the census's green-SMS layout
        fill = cv.design["bubble_out_sms"]
    fg = pal["text_out"] if own else pal["text_in"]
    if cv.screen.platform == "instagram" and own:
        _gradient_bubble(canvas, cv, x, y, w, h, radius)
    else:
        draw.rounded_rectangle(
            [cv.s(x), cv.s(y), cv.s(x + w), cv.s(y + h)],
            radius=cv.s(radius), fill=fill)
    if tail and cv.dflag("tail", "none") == "top":
        if own:
            draw.polygon([(cv.s(x + w - 8), cv.s(y + 1)),
                          (cv.s(x + w + 6), cv.s(y)),
                          (cv.s(x + w - 8), cv.s(y + 11))],
                         fill=fill)
        else:
            draw.polygon([(cv.s(x + 8), cv.s(y + 1)),
                          (cv.s(x - 6), cv.s(y)),
                          (cv.s(x + 8), cv.s(y + 11))],
                         fill=fill)
    elif tail and cv.dflag("tail", "none") == "bottom":
        # the Telegram / iMessage tail: a small point at
        # the bottom corner of a run's last bubble
        if own:
            draw.polygon([(cv.s(x + w - 13), cv.s(y + h - 2)),
                          (cv.s(x + w + 3), cv.s(y + h + 2)),
                          (cv.s(x + w - 3), cv.s(y + h - 9))],
                         fill=fill)
        else:
            draw.polygon([(cv.s(x + 13), cv.s(y + h - 2)),
                          (cv.s(x - 3), cv.s(y + h + 2)),
                          (cv.s(x + 3), cv.s(y + h - 9))],
                         fill=fill)
    cx = x + _PAD_X
    cy = y + _PAD_Y
    if msg.kind == "image":
        media_w, media_h = parts["media"]
        if cv.screen.platform == "messenger":
            # a Messenger photo fills its bubble: no frame
            _draw_media_flush(canvas, draw, cv, msg, x, y, w,
                              media_h + 8, radius)
            cy = y + media_h + 8 + 2
        else:
            _draw_media_image(canvas, draw, cv, msg, x + 4,
                              y + 4, media_w, media_h)
            cy = y + 4 + media_h + 2
        for line in parts.get("caption_lines") or []:
            draw.text((cv.s(cx), cv.s(cy)), line,
                      font=cv.font(_TEXT_SIZE), fill=fg)
            cy += _LINE_H
        if cv.dflag("time_in_bubble"):
            _draw_time_row(draw, cv, msg, x, y, w, h, fg)
        return
    if msg.kind == "voice":
        _draw_voice(draw, cv, msg, x, y, w, h, fg)
        return
    quote = _quote_block(cv, msg)
    if quote is not None:
        name, color, _qtext = quote
        _n, qlines = parts["quote"]
        qh = 14 + len(qlines) * 14.5 + 5
        qbg = _shade(fill, 0.07) if not (
            cv.screen.platform == "instagram" and own) else (
            _shade(cv.pal["bubble_out"], 0.18))
        draw.rounded_rectangle(
            [cv.s(cx - 3), cv.s(cy - 2), cv.s(x + w - _PAD_X + 3),
             cv.s(cy + qh)], radius=cv.s(5), fill=qbg)
        draw.rectangle([cv.s(cx - 3), cv.s(cy - 2),
                        cv.s(cx), cv.s(cy + qh)], fill=color)
        draw.text((cv.s(cx + 5), cv.s(cy)), name,
                  font=cv.font(11.5, bold=True), fill=color)
        qy = cy + 14
        for line in qlines:
            draw.text((cv.s(cx + 5), cv.s(qy)), line,
                      font=cv.font(11), fill=pal["meta"])
            qy += 14.5
        cy += qh + 4
    if "name_line" in parts:
        author = cv.screen.participant(msg.sender)
        if cv.dflag("group_name_colored", True) \
                and author is not None:
            color = _participant_color(author, cv.screen)
        else:
            color = pal["meta"]
        draw.text((cv.s(cx), cv.s(cy - 1)), parts["name_line"],
                  font=cv.font(12.5, bold=True), fill=color)
        cy += 15
    for line in parts["lines"]:
        draw.text((cv.s(cx), cv.s(cy)), line,
                  font=cv.font(_TEXT_SIZE), fill=fg)
        cy += _LINE_H
    if cv.dflag("time_in_bubble"):
        _draw_time_row(draw, cv, msg, x, y, w, h, fg)


def _draw_media_flush(canvas, draw, cv: _Canvas, msg: Message,
                      x: float, y: float, w: float, h: float,
                      radius: float) -> None:
    """A photo that fills its bubble edge-to-edge (the
    Messenger idiom: no coloured frame around a picture).
    The real image is cover-fit edge-to-edge when the
    message carries media; the placeholder scene (drawn
    on a scratch canvas) only when it carries none.  Both
    are pasted through the bubble's rounded-corner mask,
    and the media region lands in the canvas record."""
    tmp = Image.new("RGB", (cv.s(w), cv.s(h)),
                    cv.pal["wallpaper"])
    tdraw = ImageDraw.Draw(tmp)
    img = cv.message_image(msg)
    if img is not None:
        tile = ImageOps.fit(img.convert("RGB"),
                            (max(1, cv.s(w)), max(1, cv.s(h))),
                            Image.LANCZOS)
        tmp.paste(tile, (0, 0))
        cv.note_media_region(msg, (x, y, w, h), "real")
    else:
        _draw_media_placeholder(tmp, tdraw, cv, 0, 0, w, h,
                                msg.id)
        cv.note_media_region(msg, (x, y, w, h), "placeholder")
    mask = Image.new("L", (cv.s(w), cv.s(h)), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle(
        [0, 0, cv.s(w), cv.s(h)], radius=cv.s(radius), fill=255)
    canvas.paste(tmp, (cv.s(x), cv.s(y)), mask)


def _draw_flat_row(canvas, draw, cv: _Canvas, msg: Message,
                   parts: dict, x: float, y: float, w: float,
                   h: float) -> None:
    """One Snapchat chat row: a coloured vertical bar
    (blue for chat, red for a snap), the author's name in
    caps, the content at full width, a hairline under it.
    ``x`` is the row's left edge; content starts at 24."""
    pal = cv.pal
    own = parts["own"]
    bar = pal.get("snap_red") or pal["accent"] \
        if msg.kind == "image" else pal["accent"]
    draw.rounded_rectangle(
        [cv.s(12), cv.s(y + 2), cv.s(15.5), cv.s(y + h - 4)],
        radius=cv.s(1.75), fill=bar)
    author = cv.screen.participant(msg.sender)
    name = "ME" if own else (
        author.name if author is not None else msg.sender)
    draw.text((cv.s(24), cv.s(y)), name.upper(),
              font=cv.font(11.5, bold=True),
              fill=pal["text_in"])
    cy = y + 17
    if msg.kind == "image":
        media_w, media_h = parts["media"]
        _draw_media_image(canvas, draw, cv, msg, 24, cy,
                          media_w, media_h)
        cy += media_h + 4
        for line in parts.get("caption_lines") or []:
            draw.text((cv.s(24), cv.s(cy)), line,
                      font=cv.font(_TEXT_SIZE),
                      fill=pal["text_in"])
            cy += _LINE_H
    elif msg.kind == "voice":
        _draw_voice(draw, cv, replace(msg, time="", ticks=""),
                    24, cy, 250, 40, pal["text_in"])
    else:
        for line in parts["lines"]:
            draw.text((cv.s(24), cv.s(cy)), line,
                      font=cv.font(_TEXT_SIZE),
                      fill=pal["text_in"])
            cy += _LINE_H
    draw.line([cv.s(24), cv.s(y + h - 1), cv.s(cv.w - 12),
               cv.s(y + h - 1)], fill=pal["chip_fg"],
              width=cv.s(0.75))


def _draw_time_row(draw, cv: _Canvas, msg: Message, x: float,
                   y: float, w: float, h: float, fg) -> None:
    pal = cv.pal
    ty = y + h - 15
    tx = x + w - _PAD_X
    if msg.sender == "me" and msg.ticks and cv.dflag(
            "ticks_in_bubble"):
        tw = _draw_ticks(draw, cv, tx - 15, ty + 1, msg.ticks)
        tx -= tw + 3
    if msg.time:
        label = msg.time
        tw = cv.text_w(label, 10.5)
        draw.text((cv.s(tx - tw), cv.s(ty)), label,
                  font=cv.font(10.5), fill=pal["meta"])


def _draw_voice(draw, cv: _Canvas, msg: Message, x: float,
                y: float, w: float, h: float, fg) -> None:
    pal = cv.pal
    own = msg.sender == "me"
    cx = x + 10
    cy = y + h / 2
    draw.ellipse([cv.s(cx), cv.s(cy - 15), cv.s(cx + 30),
                  cv.s(cy + 15)],
                 fill=pal["meta"] if own else pal["accent"])
    draw.polygon([(cv.s(cx + 12), cv.s(cy - 7)),
                  (cv.s(cx + 12), cv.s(cy + 7)),
                  (cv.s(cx + 22), cv.s(cy))],
                 fill=(255, 255, 255))
    bx = cx + 40
    probe = cv.voice_probe(msg)
    if probe is not None:
        # the measured waveform: bar heights ARE the
        # audio's own peak amplitudes, and the label is
        # its true duration read from the samples
        heights = [3.0 + peak * 13.0 for peak in probe.bars]
        label = format_duration(probe.seconds)
        cv.note_media_region(
            msg, (bx, y, len(heights) * 4.2, h), "real")
    else:
        digest = hashlib.sha256(msg.id.encode("utf-8")).digest()
        heights = [4 + (digest[i % len(digest)] % 13)
                   for i in range(VOICE_BARS)]
        label = msg.duration or "0:00"
    for i, bh in enumerate(heights):
        draw.rounded_rectangle(
            [cv.s(bx + i * 4.2), cv.s(cy - bh / 2 - 4),
             cv.s(bx + i * 4.2 + 2.2), cv.s(cy + bh / 2 - 4)],
            radius=cv.s(1.1), fill=fg)
    draw.text((cv.s(bx), cv.s(cy + 6)), label,
              font=cv.font(10.5), fill=pal["meta"])
    if own and msg.ticks and cv.dflag("ticks_in_bubble"):
        _draw_ticks(draw, cv, x + w - 26, y + h - 14, msg.ticks)
    if msg.time:
        tw = cv.text_w(msg.time, 10.5)
        draw.text((cv.s(x + w - 12 - tw), cv.s(y + 4)),
                  msg.time, font=cv.font(10.5), fill=pal["meta"])


def _chip_size(cv: _Canvas, text: str, *, system: bool) -> tuple:
    size = 11.0
    max_w = cv.w * (0.86 if system else 0.7)
    lines = cv.wrap(text, size, max_w - 20)
    w = max(cv.text_w(line, size) for line in lines) + 20
    h = len(lines) * 15 + 9
    return min(w, max_w), h, lines


def _draw_chip(draw, cv: _Canvas, msg: Message, y: float,
               *, system: bool) -> float:
    pal = cv.pal
    w, h, lines = _chip_size(cv, msg.text, system=system)
    x = (cv.w - w) / 2
    bg = pal["system_bg"] if system else pal["chip_bg"]
    fg = pal["system_fg"] if system else pal["chip_fg"]
    draw.rounded_rectangle(
        [cv.s(x), cv.s(y), cv.s(x + w), cv.s(y + h)],
        radius=cv.s(7), fill=bg)
    ly = y + 4.5
    for line in lines:
        tw = cv.text_w(line, 11.0)
        draw.text((cv.s(x + (w - tw) / 2), cv.s(ly)), line,
                  font=cv.font(11), fill=fg)
        ly += 15
    return h


def _attach_media_record(cv: _Canvas, entry: dict,
                         msg: Message) -> dict:
    """Fold one element's media region into its drawn
    record: the composited media box (device pixels) and
    its source for image/voice elements, plus the measured
    duration + bars when a voice note was probed — the
    record proves the media landed, and the animation
    phase reads the same facts."""
    region = cv.media_regions.get(msg.id)
    if region is not None:
        (mx, my, mw, mh), source = region
        entry["media_box"] = [cv.s(mx), cv.s(my), cv.s(mw),
                              cv.s(mh)]
        entry["media_source"] = source
    probe = cv._probes.get(msg.id)
    if probe is not None:
        entry["duration_s"] = round(probe.seconds, 3)
        entry["bars"] = [round(b, 3) for b in probe.bars]
    return entry


def render_screen(screen: Screen, *, avatars: dict | None = None,
                  media: dict | None = None):
    """Render one chat screen.  Returns ``(png_bytes,
    drawn)`` — the drawn record carries the output size,
    the chrome regions and one entry per element (id,
    kind, sender, box, ticks/seen state, and for media
    elements the composited media box + source).

    ``avatars`` maps a participant id to an avatar hand-off
    (bytes / path / PIL image) the live surface supplied;
    it wins over the participant's own ``avatar`` ref.
    ``media`` maps a message id to a real-media hand-off
    (bytes / path / PIL image / MediaRef) for that
    message's photo or voice note; it wins over the
    message's own ``media`` reference."""
    if screen.platform not in CHAT_DESIGNS:
        raise MockScreensError(
            "no chat design called " + repr(screen.platform)
            + " — designs: " + ", ".join(sorted(CHAT_DESIGNS)))
    cv = _Canvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})

    # ── measure ──────────────────────────────────────────
    items: list = []  # (msg, kind, w, h, parts, gap_before)
    prev_sender: str | None = None
    prev_time = ""
    flat = bool(cv.dflag("flat_rows"))
    status_mode = str(cv.dflag("status_line", "") or "")
    # iMessage / Snapchat put their read state under ONE
    # message: the last own message carrying ticks.
    last_own_ticks = -1
    for i, m in enumerate(screen.messages):
        if m.sender == "me" and m.ticks:
            last_own_ticks = i
    for mi, msg in enumerate(screen.messages):
        if msg.kind in ("divider", "system"):
            w, h, _lines = _chip_size(
                cv, msg.text, system=msg.kind == "system")
            items.append((msg, msg.kind, w, h, {}, 10.0))
            prev_sender = None
            prev_time = ""
            continue
        if (cv.dflag("time_centered") and msg.time
                and msg.time != prev_time):
            items.append((msg, "meta_time", 0.0, 15.0, {}, 8.0))
        if flat:
            w, h, parts = _measure_flat(cv, msg)
            gap = 6.0
        else:
            w, h, parts = _measure_bubble(cv, msg)
            gap = 3.0 if prev_sender == msg.sender else 10.0
        h_extra = 10.0 if msg.reactions else 0.0
        items.append((msg, "bubble", w, h, parts, gap))
        if h_extra:
            items[-1] = (msg, "bubble", w, h + h_extra, parts, gap)
        prev_sender = msg.sender
        prev_time = msg.time
        if (status_mode in ("seen", "seen_avatar")
                and msg.sender == "me" and msg.seen):
            h_seen = 15.0 if status_mode == "seen" else 17.0
            items.append((msg, status_mode, 0.0, h_seen, {}, 1.0))
            prev_sender = None
        elif (status_mode == "imessage" and mi == last_own_ticks
                and msg.ticks in ("delivered", "read")):
            label = "Delivered" if msg.ticks == "delivered" \
                else "Read " + (msg.time or "")
            parts_status = {"label": label}
            items.append((msg, "status", 0.0, 14.0,
                          parts_status, 1.0))
            prev_sender = None
        elif (flat and mi == last_own_ticks and msg.ticks):
            label = {"sending": "Sending…", "sent": "Delivered",
                     "delivered": "Delivered",
                     "read": "Opened"}.get(msg.ticks, "")
            if label:
                items.append((msg, "status", 0.0, 14.0,
                              {"label": label}, 1.0))
                prev_sender = None

    content_h = sum(it[3] + it[5] for it in items) + 14
    status_h = float(cv.dev["status_h"])
    header_h = float(cv.design["header_h"])
    input_h = 56.0
    nav_h = float(cv.dev["nav_h"])
    total_h = int(status_h + header_h + content_h
                  + input_h + nav_h)

    # ── draw ─────────────────────────────────────────────
    img = Image.new("RGB", (cv.s(cv.w), cv.s(total_h)),
                    cv.pal["wallpaper"])
    cv.image = img
    draw = ImageDraw.Draw(img)
    drawn: dict = {
        "platform": screen.platform,
        "style": screen.chrome.style,
        "group": screen.group,
        "dark": screen.dark,
        "size": [cv.s(cv.w), cv.s(total_h)],
        "regions": {},
        "participants": [p.id for p in screen.participants],
        "elements": [],
    }
    y = _draw_status_bar(draw, cv, 0.0)
    drawn["regions"]["status_bar"] = [0, 0, cv.s(cv.w), cv.s(y)]
    header_y = y
    y += _draw_header(img, draw, cv, y)
    drawn["regions"]["header"] = [
        0, cv.s(header_y), cv.s(cv.w), cv.s(y - header_y)]

    own_run_end = set()
    for idx, (msg, _k, _w, _h, _p, _g) in enumerate(items):
        if _k == "bubble" and msg.sender != "me":
            nxt = items[idx + 1][0] if idx + 1 < len(items) else None
            if nxt is None or nxt.sender != msg.sender:
                own_run_end.add(idx)

    cy = y + 6
    for idx, (msg, kind, w, h, parts, gap) in enumerate(items):
        cy += gap
        if kind in ("divider", "system"):
            _draw_chip(draw, cv, msg, cy, system=kind == "system")
            drawn["elements"].append({
                "id": msg.id, "kind": kind,
                "box": [cv.s((cv.w - w) / 2), cv.s(cy),
                        cv.s(w), cv.s(h)]})
            cy += h
            continue
        if kind == "seen":
            label = "Seen"
            tw = cv.text_w(label, 11)
            draw.text((cv.s(cv.w - 12 - tw), cv.s(cy)), label,
                      font=cv.font(11), fill=cv.pal["meta"])
            drawn["elements"].append({
                "id": msg.id, "kind": "seen",
                "box": [cv.s(cv.w - 12 - tw), cv.s(cy),
                        cv.s(tw), cv.s(13)]})
            cy += h
            continue
        if kind == "meta_time":
            # Messenger / iMessage carry the time centred
            # above a run, not inside the bubble
            tw = cv.text_w(msg.time, 11)
            draw.text((cv.s((cv.w - tw) / 2), cv.s(cy)), msg.time,
                      font=cv.font(11), fill=cv.pal["meta"])
            drawn["elements"].append({
                "id": msg.id, "kind": "meta_time",
                "text": msg.time,
                "box": [cv.s((cv.w - tw) / 2), cv.s(cy),
                        cv.s(tw), cv.s(13)]})
            cy += h
            continue
        if kind == "status":
            # the one read-state line (iMessage Delivered /
            # Read, Snapchat Delivered / Opened)
            label = parts.get("label", "")
            tw = cv.text_w(label, 10.5)
            if flat:
                lx, color = 24.0, cv.pal["accent"]
            else:
                lx, color = cv.w - 12 - tw, cv.pal["meta"]
            draw.text((cv.s(lx), cv.s(cy)), label,
                      font=cv.font(10.5), fill=color)
            drawn["elements"].append({
                "id": msg.id, "kind": "status", "text": label,
                "box": [cv.s(lx), cv.s(cy), cv.s(tw), cv.s(13)]})
            cy += h
            continue
        if kind == "seen_avatar":
            # Messenger's read state: the reader's little
            # avatar under the last seen message
            reader = next((p for p in screen.participants
                           if p.id != "me"), None)
            _draw_avatar(img, draw, cv, cv.w - 8 - 16, cy, 16,
                         reader, "seen")
            drawn["elements"].append({
                "id": msg.id, "kind": "seen_avatar",
                "box": [cv.s(cv.w - 8 - 16), cv.s(cy),
                        cv.s(16), cv.s(16)]})
            cy += h
            continue
        own = msg.sender == "me"
        bubble_h = h - (10.0 if msg.reactions else 0.0)
        if parts.get("flat"):
            bx = 6.0
            _draw_flat_row(img, draw, cv, msg, parts, bx, cy, w,
                           bubble_h)
            if msg.reactions:
                _draw_reactions(img, draw, cv, msg, 18,
                                cy + bubble_h, w)
            drawn["elements"].append(_attach_media_record(
                cv, {
                    "id": msg.id, "kind": msg.kind,
                    "sender": msg.sender,
                    "box": [cv.s(bx), cv.s(cy), cv.s(w),
                            cv.s(bubble_h)],
                    "ticks": msg.ticks, "seen": msg.seen,
                    "reactions": list(msg.reactions)}, msg))
            cy += h
            continue
        if own:
            bx = cv.w - 8 - w
            if cv.dflag("tail", "none") == "top":
                bx -= 6  # the tail lives outside the bubble box
        else:
            bx = 16 + cv.run_avatar_size() \
                if cv.run_avatar_applies() else 8
            if cv.dflag("tail", "none") == "top":
                bx += 6
        nxt = items[idx + 1][0] if idx + 1 < len(items) else None
        tail = nxt is None or nxt.sender != msg.sender
        _draw_bubble(img, draw, cv, msg, parts, bx, cy, w,
                     bubble_h, tail=tail)
        if (not own and cv.run_avatar_applies()
                and idx in own_run_end):
            author = screen.participant(msg.sender)
            size = cv.run_avatar_size()
            _draw_avatar(img, draw, cv, 8, cy + bubble_h - size,
                         size, author, msg.sender)
        if msg.reactions:
            _draw_reactions(img, draw, cv, msg, bx,
                            cy + bubble_h, w)
        drawn["elements"].append(_attach_media_record(
            cv, {
                "id": msg.id, "kind": msg.kind,
                "sender": msg.sender,
                "box": [cv.s(bx), cv.s(cy), cv.s(w),
                        cv.s(bubble_h)],
                "ticks": msg.ticks, "seen": msg.seen,
                "reactions": list(msg.reactions)}, msg))
        cy += h

    input_y = total_h - nav_h - input_h
    _draw_input_bar(draw, cv, input_y)
    drawn["regions"]["input_bar"] = [
        0, cv.s(input_y), cv.s(cv.w), cv.s(input_h)]
    _draw_nav(draw, cv, total_h - nav_h)
    drawn["regions"]["nav"] = [
        0, cv.s(total_h - nav_h), cv.s(cv.w), cv.s(nav_h)]

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), drawn
