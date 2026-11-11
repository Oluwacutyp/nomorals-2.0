"""The Phase 2b renderers: lists, status and calls.

Three more layout engines beside the chat engine
(:mod:`.render`), one per screen class the census's
Phase 2 names (§26), all reading platform designs held
as data (:mod:`.designs` — ``LIST_DESIGNS``,
``STATUS_DESIGNS``, ``CALL_DESIGNS``) and all returning
the makers contract: PNG bytes plus a drawn record whose
entries carry each element's box and STATE, so tests
prove structure — a missed call is its record kind, not
a shade of red (the census §26 exit proof).

Shared machinery is the chat engine's own: the canvas
(fonts through the typography policy door, measured
wrapping), the device status bar / navigation, the ticks.
Media follows the media hand-off contract
(:mod:`.media`): a SUPPLIED picture — a status photo, a
caller picture, an avatar — that cannot be loaded fails
the render with the reason, never a silent placeholder;
the placeholder stands in only where no media was
supplied at all (initials discs, the photo scene).
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from .. import makers as _makers
from .designs import (
    CALL_DESIGNS,
    LIST_DESIGNS,
    STATUS_BG_COLORS,
    STATUS_DESIGNS,
)
from .media import MediaError, load_image
from .render import (
    MockScreensError,
    _Canvas,
    _draw_media_placeholder,
    _draw_mic,
    _draw_nav,
    _draw_status_bar,
    _draw_ticks,
    _emoji_tile,
    _fit_text,
    _participant_color,
)
from .schema import (
    CallHistoryScreen,
    CallScreen,
    ListRow,
    ListScreen,
    Participant,
    StatusScreen,
    StatusViewsScreen,
)

__all__ = [
    "render_call_history",
    "render_call_screen",
    "render_list_screen",
    "render_status_screen",
    "render_status_views",
]


class _ListCanvas(_Canvas):
    DESIGNS = LIST_DESIGNS


class _StatusCanvas(_Canvas):
    DESIGNS = STATUS_DESIGNS


class _CallCanvas(_Canvas):
    DESIGNS = CALL_DESIGNS


def _require(platform: str, registry: dict, what: str) -> None:
    if platform not in registry:
        raise MockScreensError(
            "no " + what + " design called " + repr(platform)
            + " — designs: " + ", ".join(sorted(registry)))


def _finish(canvas: Image.Image) -> bytes:
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _strict_image(cv: _Canvas, source, what: str):
    """Load one supplied picture through the media door,
    translating its refusal into the organ's error: a
    supplied picture that cannot be used fails the
    render with the reason."""
    try:
        return load_image(source)
    except MediaError as exc:
        raise MockScreensError(
            what + " could not be used: " + str(exc)) from exc


def _avatar_source(cv: _Canvas, person: Participant | None):
    """The supplied avatar source for one person: the
    render's hand-off wins, else the participant's own
    reference; None when no avatar was supplied."""
    if person is None:
        return None
    handed = getattr(cv, "avatars", {}).get(person.id)
    if handed is not None:
        return handed
    return person.avatar or None


def _draw_avatar_strict(canvas, draw, cv: _Canvas, x: float,
                        y: float, size: float,
                        person: Participant | None,
                        fallback: str) -> str:
    """An avatar disc under the media hand-off contract:
    a supplied avatar that cannot be loaded fails the
    render with the reason; no avatar supplied wears the
    initials placeholder.  Returns the source worn
    (``"real"`` / ``"placeholder"``) for the record."""
    name = person.name if person is not None else fallback
    source = _avatar_source(cv, person)
    if source is not None:
        img = _strict_image(
            cv, source, "the avatar for " + repr(name))
        if _makers._paste_avatar_shape(
                canvas, (cv.s(x), cv.s(y)), cv.s(size), img,
                "circle"):
            return "real"
        raise MockScreensError(
            "the avatar for " + repr(name)
            + " could not be used: it did not paste")
    color = (_participant_color(person, cv.screen)
             if person is not None
             else _makers._medallion_color(name))
    _makers._draw_placeholder(
        draw, (cv.s(x), cv.s(y)), cv.s(size), name, color,
        shape="circle")
    return "placeholder"


def _draw_ring(draw, cv: _Canvas, x: float, y: float,
               size: float, ring: str, segments: int = 1
               ) -> None:
    """A story ring around an avatar: one arc per segment
    (gaps between), in the design's unseen/seen colour."""
    color = cv.pal["ring_unseen"] if ring == "unseen" \
        else cv.pal["ring_seen"]
    grow = 3.0
    box = [cv.s(x - grow), cv.s(y - grow),
           cv.s(x + size + grow), cv.s(y + size + grow)]
    if segments <= 1:
        draw.ellipse(box, outline=color, width=cv.s(2.2))
        return
    step = 360.0 / segments
    for i in range(segments):
        draw.arc(box, start=-90 + i * step + 5,
                 end=-90 + (i + 1) * step - 5, fill=color,
                 width=cv.s(2.2))


def _draw_badge(draw, cv: _Canvas, right: float, cy: float,
                count: int) -> float:
    """An unread-count pill ending at ``right``; returns
    the width it consumed."""
    label = str(count) if count < 100 else "99+"
    tw = cv.text_w(label, 10.5, bold=True)
    w = max(17.0, tw + 9)
    h = 17.0
    draw.rounded_rectangle(
        [cv.s(right - w), cv.s(cy - h / 2), cv.s(right),
         cv.s(cy + h / 2)], radius=cv.s(h / 2),
        fill=cv.pal["badge_bg"])
    draw.text((cv.s(right - w + (w - tw) / 2), cv.s(cy - 6.5)),
              label, font=cv.font(10.5, bold=True),
              fill=cv.pal["badge_fg"])
    return w


def _draw_online_dot(draw, cv: _Canvas, x: float, y: float,
                     size: float) -> None:
    r = max(4.5, size * 0.15)
    cx, cy = x + size - r, y + size - r
    draw.ellipse(
        [cv.s(cx - r), cv.s(cy - r), cv.s(cx + r),
         cv.s(cy + r)], fill=cv.pal["online"],
        outline=cv.pal["list_bg"], width=cv.s(1.6))


def _draw_search_icon(draw, cv: _Canvas, cx: float,
                      cy: float, color, r: float = 6.5) -> None:
    draw.ellipse([cv.s(cx - r), cv.s(cy - r), cv.s(cx + r),
                  cv.s(cy + r)], outline=color, width=cv.s(1.8))
    draw.line([cv.s(cx + r * 0.7), cv.s(cy + r * 0.7),
               cv.s(cx + r * 1.5), cv.s(cy + r * 1.5)],
              fill=color, width=cv.s(1.8))


def _draw_dots(draw, cv: _Canvas, cx: float, cy: float,
               color) -> None:
    for i in (-1, 0, 1):
        draw.ellipse(
            [cv.s(cx - 1.8), cv.s(cy + i * 7 - 1.8),
             cv.s(cx + 1.8), cv.s(cy + i * 7 + 1.8)],
            fill=color)


def _hex_rgb(value: str):
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def _darken(canvas: Image.Image, cv: _Canvas, y0: float,
            y1: float, factor: float = 0.55) -> None:
    """Darken a horizontal strip of the canvas in place
    (the scrim a photo status wears under its chrome)."""
    box = (0, max(0, cv.s(y0)), canvas.width,
           min(canvas.height, cv.s(y1)))
    if box[3] <= box[1]:
        return
    strip = canvas.crop(box)
    canvas.paste(
        ImageEnhance.Brightness(strip).enhance(factor), box[:2])


# ── lists (census §4.3–§4.8) ─────────────────────────────

#: Preview-kind → (glyph, default label) for the media /
#: event previews (census §4.4 + §4.6 vocabularies).  A
#: row's own preview text wins over the default label.
_PREVIEW_LABELS: dict[str, tuple[str, str]] = {
    "photo": ("📷", "Photo"),
    "video": ("🎥", "Video"),
    "audio": ("🎧", "Audio"),
    "voice": ("🎤", "Voice message"),
    "gif": ("", "GIF"),
    "sticker": ("", "Sticker"),
    "document": ("📄", "Document"),
    "location": ("📍", "Location"),
    "contact": ("👤", "Contact"),
    "disappearing": ("⏳", "Disappearing message"),
    "deleted": ("🚫", "This message was deleted"),
    "status_reaction": ("", "Reacted to your status"),
    "story_mention": ("", "Mentioned you in your story"),
    "story_reaction": ("", "Reacted to your story"),
    "story_reply": ("", "Replied to your story"),
    "reel_reaction": ("", "Reacted to your reel"),
    "typing": ("", "Typing…"),
}

#: Call preview kinds → (arrow, label, missed): direction
#: and medium ride the kind, so the record — and a reader
#: not relying on colour — can tell them apart.
_CALL_PREVIEWS: dict[str, tuple[str, str, bool]] = {
    "call_in_voice": ("↙", "Incoming voice call", False),
    "call_out_voice": ("↗", "Outgoing voice call", False),
    "call_missed_voice": ("↙", "Missed voice call", True),
    "call_in_video": ("↙", "Incoming video call", False),
    "call_out_video": ("↗", "Outgoing video call", False),
    "call_missed_video": ("↙", "Missed video call", True),
}


def _list_header(draw, cv: _ListCanvas, y0: float) -> float:
    pal, screen = cv.pal, cv.screen
    h = float(cv.design["header_h"])
    draw.rectangle([0, cv.s(y0), cv.s(cv.w), cv.s(y0 + h)],
                   fill=pal["header_bg"])
    mid = y0 + h / 2
    title = screen.title
    if screen.platform == "telegram" \
            and screen.chrome.style == "iphone":
        tw = cv.text_w(title, 16.5, bold=True)
        draw.text((cv.s((cv.w - tw) / 2), cv.s(mid - 11)),
                  title, font=cv.font(16.5, bold=True),
                  fill=pal["header_fg"])
    else:
        color = pal["accent"] if (
            screen.platform == "telegram"
            and screen.chrome.style == "android") \
            else pal["header_fg"]
        draw.text((cv.s(16), cv.s(mid - 11)), title,
                  font=cv.font(17, bold=True), fill=color)
    if screen.subtitle:
        draw.text((cv.s(16), cv.s(y0 + h - 19)),
                  _fit_text(cv, screen.subtitle, 11.5,
                            cv.w - 140),
                  font=cv.font(11.5), fill=pal["header_sub"])
    _draw_dots(draw, cv, cv.w - 16, mid, pal["header_fg"])
    _draw_search_icon(draw, cv, cv.w - 44, mid,
                      pal["header_fg"], r=6)
    return h


def _list_search(draw, cv: _ListCanvas, y0: float) -> float:
    pal = cv.pal
    h = 44.0
    draw.rounded_rectangle(
        [cv.s(12), cv.s(y0 + 5), cv.s(cv.w - 12),
         cv.s(y0 + h - 5)], radius=cv.s(17),
        fill=pal["search_bg"])
    _draw_search_icon(draw, cv, 32, y0 + h / 2,
                      pal["search_fg"], r=5.5)
    label = cv.screen.search or str(
        cv.design.get("search_hint", "Search"))
    draw.text((cv.s(48), cv.s(y0 + h / 2 - 8)), label,
              font=cv.font(13.5), fill=pal["search_fg"])
    return h


def _list_stories(canvas, draw, cv: _ListCanvas,
                  y0: float) -> tuple[float, list]:
    """The stories / notes row: an avatar per item (ring
    when the item wears one, a note bubble above it when
    the item carries a note — Instagram's notes row)."""
    screen = cv.screen
    size = 54.0
    slot = 74.0
    top = y0 + (30.0 if any(s.note for s in screen.stories)
                else 8.0)
    elements: list = []
    x = 8.0
    for item in screen.stories:
        person = screen.participant(item.participant_id)
        name = item.label or (
            person.name if person is not None else "")
        if item.note:
            note = _fit_text(cv, item.note, 10.5, slot - 8)
            nw = cv.text_w(note, 10.5) + 12
            bx = x + (slot - nw) / 2
            draw.rounded_rectangle(
                [cv.s(bx), cv.s(y0 + 2), cv.s(bx + nw),
                 cv.s(y0 + 24)], radius=cv.s(9),
                fill=cv.pal["tab_sel_bg"])
            tw = cv.text_w(note, 10.5)
            draw.text((cv.s(bx + (nw - tw) / 2),
                       cv.s(y0 + 6)), note,
                      font=cv.font(10.5),
                      fill=cv.pal["row_name"])
        source = _draw_avatar_strict(
            canvas, draw, cv, x + (slot - size) / 2, top, size,
            person, name)
        if item.ring:
            _draw_ring(draw, cv, x + (slot - size) / 2, top,
                       size, item.ring)
        label = _fit_text(cv, name.split(" ")[0], 11, slot - 4)
        tw = cv.text_w(label, 11)
        draw.text((cv.s(x + (slot - tw) / 2),
                   cv.s(top + size + 5)), label,
                  font=cv.font(11), fill=cv.pal["row_name"])
        elements.append({
            "id": item.participant_id, "kind": "story",
            "ring": item.ring, "note": item.note,
            "avatar_source": source,
            "box": [cv.s(x), cv.s(y0), cv.s(slot),
                    cv.s(top + size + 24 - y0)]})
        x += slot
    return top + size + 26 - y0, elements


def _list_tabs(draw, cv: _ListCanvas, y0: float) -> float:
    """The folder / filter tab strip: one pill per tab,
    the selected one in the design's selected colours,
    an unread count riding its label."""
    pal, screen = cv.pal, cv.screen
    h = 42.0
    x = 12.0
    cy = y0 + h / 2
    for i, tab in enumerate(screen.tabs):
        label = tab.label + (
            " " + str(tab.unread) if tab.unread else "")
        tw = cv.text_w(label, 12.5, bold=True)
        w = tw + 22
        selected = i == screen.tab_selected
        draw.rounded_rectangle(
            [cv.s(x), cv.s(cy - 13), cv.s(x + w),
             cv.s(cy + 13)], radius=cv.s(13),
            fill=pal["tab_sel_bg"] if selected
            else pal["tab_bg"],
            outline=None if selected else pal["divider"],
            width=cv.s(1))
        draw.text((cv.s(x + 11), cv.s(cy - 8)), label,
                  font=cv.font(12.5, bold=True),
                  fill=pal["tab_sel_fg"] if selected
                  else pal["tab_fg"])
        x += w + 8
    return h


def _preview_parts(cv: _ListCanvas, row) -> tuple[str, str]:
    """(text, colour-key) of a row's preview line: a
    draft wears its red prefix, typing the accent, a
    media/event kind its glyph + label (the row's own
    text winning over the default label), a call kind
    its arrow + label."""
    kind = row.preview_kind
    if row.draft or kind == "draft":
        return "Draft: " + row.preview, "draft_fg"
    if kind == "typing":
        return "Typing…", "accent"
    if kind in _CALL_PREVIEWS:
        arrow, label, missed = _CALL_PREVIEWS[kind]
        return arrow + " " + (row.preview or label), \
            "draft_fg" if missed else "row_preview"
    if kind in _PREVIEW_LABELS:
        glyph, label = _PREVIEW_LABELS[kind]
        text = row.preview or label
        return (glyph + " " + text) if glyph else text, \
            "row_preview"
    return row.preview, "row_preview"


def _list_row(canvas, draw, cv: _ListCanvas, row,
              y0: float) -> dict:
    pal, screen = cv.pal, cv.screen
    h = float(cv.design["row_h"])
    av = 52.0
    ax = 14.0
    ay = y0 + (h - av) / 2
    person = screen.participant(row.participant_id)
    entry: dict = {
        "id": row.id, "kind": "row",
        "row_type": row.row_type,
        "preview_kind": row.preview_kind,
        "unread": row.unread, "pinned": row.pinned,
        "muted": row.muted, "ring": row.ring,
        "ticks": row.ticks, "status": row.status,
        "draft": row.draft, "verified": row.verified,
        "online": row.online, "mention": row.mention,
        "sender": row.sender, "segments": row.segments,
        "section": row.section,
        "box": [cv.s(0), cv.s(y0), cv.s(cv.w), cv.s(h)]}
    if row.row_type == "archived":
        # the Archived Chats row: a muted archive box, no
        # person's face (census §4.7)
        draw.ellipse([cv.s(ax), cv.s(ay), cv.s(ax + av),
                      cv.s(ay + av)], fill=pal["tab_sel_bg"])
        bx, by = ax + av / 2, ay + av / 2
        draw.rectangle(
            [cv.s(bx - 11), cv.s(by - 8), cv.s(bx + 11),
             cv.s(by + 8)], outline=pal["row_preview"],
            width=cv.s(2))
        draw.line([cv.s(bx - 11), cv.s(by - 2),
                   cv.s(bx + 11), cv.s(by - 2)],
                  fill=pal["row_preview"], width=cv.s(2))
        draw.rectangle(
            [cv.s(bx - 4), cv.s(by + 2), cv.s(bx + 4),
             cv.s(by + 5)], fill=pal["row_preview"])
        entry["avatar_source"] = "none"
    else:
        entry["avatar_source"] = _draw_avatar_strict(
            canvas, draw, cv, ax, ay, av, person, row.name)
        if row.ring:
            _draw_ring(draw, cv, ax, ay, av, row.ring,
                       segments=row.segments
                       if screen.flavor == "status" else 1)
        if row.online:
            _draw_online_dot(draw, cv, ax, ay, av)
    tx = ax + av + 12
    right = cv.w - 14
    # time (top right)
    if row.time:
        tcol = pal["time_unread"] if row.unread \
            else pal["row_time"]
        tw = cv.text_w(row.time, 11.5)
        draw.text((cv.s(right - tw), cv.s(y0 + 13)), row.time,
                  font=cv.font(11.5), fill=tcol)
    # name line
    name = _fit_text(cv, row.name, 15.5, right - tx - 70,
                     bold=bool(row.unread))
    draw.text((cv.s(tx), cv.s(y0 + 11)), name,
              font=cv.font(15.5, bold=bool(row.unread)),
              fill=pal["row_name"])
    nx = tx + cv.text_w(name, 15.5, bold=bool(row.unread))
    if row.verified:
        vx, vy = nx + 6, y0 + 15
        draw.ellipse([cv.s(vx), cv.s(vy), cv.s(vx + 13),
                      cv.s(vy + 13)], fill=pal["accent"])
        draw.text((cv.s(vx + 2.5), cv.s(vy - 0.5)), "✓",
                  font=cv.font(9.5, bold=True),
                  fill=(255, 255, 255))
    # preview line
    py = y0 + 38
    px = tx
    if row.ticks:
        tw2 = _draw_ticks(draw, cv, px, py + 2, row.ticks)
        px += tw2 + 5
    if row.sender:
        draw.text((cv.s(px), cv.s(py)), row.sender + ": ",
                  font=cv.font(13), fill=pal["row_preview"])
        px += cv.text_w(row.sender + ": ", 13)
    ptext, pkey = _preview_parts(cv, row)
    if row.status and not ptext:
        ptext = {"sent": "Sent", "delivered": "Sent",
                 "seen": "Seen"}[row.status]
    draw.text(
        (cv.s(px), cv.s(py)),
        _fit_text(cv, ptext, 13.5,
                  right - px - (26 if (row.unread or row.mention)
                                else 0)),
        font=cv.font(13.5, bold=bool(
            row.unread and screen.platform == "instagram")),
        fill=pal[pkey])
    if screen.platform == "instagram" and row.unread:
        # the blue dot of an unseen chat (census §4.6)
        draw.ellipse([cv.s(right - 9), cv.s(py + 4),
                      cv.s(right), cv.s(py + 13)],
                     fill=pal["accent"])
    # bottom-right furniture: unread badge / mention / pin
    by = y0 + h - 24
    if row.unread:
        _draw_badge(draw, cv, right, by, row.unread)
    elif row.mention:
        draw.ellipse([cv.s(right - 17), cv.s(by - 8.5),
                      cv.s(right), cv.s(by + 8.5)],
                     fill=pal["badge_bg"])
        draw.text((cv.s(right - 13.5), cv.s(by - 6.5)), "@",
                  font=cv.font(10.5, bold=True),
                  fill=pal["badge_fg"])
    elif row.muted:
        draw.text((cv.s(right - 14), cv.s(by - 7)), "🔕",
                  font=cv.font(11), fill=pal["meta"])
    if row.pinned:
        draw.text((cv.s(right - (34 if row.unread else 14)),
                   cv.s(y0 + 32)), "📌",
                  font=cv.font(10), fill=pal["pinned_fg"])
    draw.line([cv.s(tx), cv.s(y0 + h - 0.5), cv.s(cv.w),
               cv.s(y0 + h - 0.5)], fill=pal["divider"],
              width=cv.s(0.75))
    return entry


def _list_call_row(canvas, draw, cv: _ListCanvas, row,
                   y0: float) -> dict:
    """A call-log row (census §4.5): avatar, name (the
    missed red rides the sub-line, the kind carries the
    state), direction arrow + time, and the medium icon
    at the right — voice or video, in the record too."""
    pal, screen = cv.pal, cv.screen
    h = float(cv.design["row_h"])
    av = 48.0
    ax = 14.0
    ay = y0 + (h - av) / 2
    person = screen.participant(row.participant_id)
    source = _draw_avatar_strict(
        canvas, draw, cv, ax, ay, av, person, row.name)
    arrow, label, missed = _CALL_PREVIEWS.get(
        row.preview_kind, ("↙", "Incoming voice call", False))
    call_type = row.call_type or (
        "video" if row.preview_kind.endswith("video")
        else "voice")
    entry = {
        "id": row.id, "kind": "call_row",
        "call": {"direction": row.preview_kind,
                 "type": call_type, "missed": missed},
        "avatar_source": source,
        "box": [cv.s(0), cv.s(y0), cv.s(cv.w), cv.s(h)]}
    tx = ax + av + 12
    draw.text((cv.s(tx), cv.s(y0 + 12)),
              _fit_text(cv, row.name, 15.5, cv.w - tx - 60),
              font=cv.font(15.5), fill=pal["row_name"])
    sub = arrow + " " + (row.preview or label)
    if row.time and row.preview:
        sub += " · " + row.time
    elif row.time and not row.preview:
        sub = arrow + " " + row.time
    draw.text((cv.s(tx), cv.s(y0 + 38)), sub,
              font=cv.font(13),
              fill=pal["draft_fg"] if missed
              else pal["row_preview"])
    ix = cv.w - 40
    mid = y0 + h / 2
    if call_type == "video":
        draw.rounded_rectangle(
            [cv.s(ix), cv.s(mid - 8), cv.s(ix + 17),
             cv.s(mid + 8)], radius=cv.s(3),
            fill=pal["accent"])
        draw.polygon([(cv.s(ix + 19), cv.s(mid - 3)),
                      (cv.s(ix + 26), cv.s(mid - 8)),
                      (cv.s(ix + 26), cv.s(mid + 8)),
                      (cv.s(ix + 19), cv.s(mid + 3))],
                     fill=pal["accent"])
    else:
        draw.text((cv.s(ix + 2), cv.s(mid - 10)), "☎",
                  font=cv.font(17), fill=pal["accent"])
    draw.line([cv.s(tx), cv.s(y0 + h - 0.5), cv.s(cv.w),
               cv.s(y0 + h - 0.5)], fill=pal["divider"],
              width=cv.s(0.75))
    return entry


def render_list_screen(screen: ListScreen, *,
                       avatars: dict | None = None,
                       media: dict | None = None):
    """Render one list screen (a chat inbox, a call log,
    or the status/Updates list).  Returns ``(png_bytes,
    drawn)`` in the makers contract."""
    _require(screen.platform, LIST_DESIGNS, "list")
    cv = _ListCanvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})
    pal = cv.pal
    design = cv.design

    status_h = float(cv.dev["status_h"])
    header_h = float(design["header_h"])
    search_h = 44.0 if design.get("search") else 0.0
    stories_h = 0.0
    if screen.stories:
        # _list_stories' own measure: the note offset (or
        # the plain one) + avatar + label + padding
        stories_h = (30.0 if any(
            s.note for s in screen.stories) else 8.0) + 80.0
    tabs_h = 42.0 if screen.tabs else 0.0
    row_h = float(design["row_h"])
    sections: list[str] = []
    for row in screen.rows:
        if row.section and row.section not in sections:
            sections.append(row.section)
    rows_h = len(screen.rows) * row_h \
        + len(sections) * 28.0
    archived_h = row_h if screen.archived >= 0 else 0.0
    nav_tabs = design.get("nav_tabs")
    bottom_h = 56.0 if nav_tabs else float(cv.dev["nav_h"])
    total_h = int(status_h + header_h + search_h + stories_h
                  + tabs_h + archived_h + rows_h + bottom_h
                  + 6)

    img = Image.new("RGB", (cv.s(cv.w), cv.s(total_h)),
                    pal["list_bg"])
    cv.image = img
    draw = ImageDraw.Draw(img)
    drawn: dict = {
        "platform": screen.platform,
        "style": screen.chrome.style,
        "flavor": screen.flavor,
        "dark": screen.dark,
        "size": [cv.s(cv.w), cv.s(total_h)],
        "regions": {},
        "participants": [p.id for p in screen.participants],
        "elements": [],
    }
    y = _draw_status_bar(draw, cv, 0.0)
    drawn["regions"]["status_bar"] = [0, 0, cv.s(cv.w),
                                      cv.s(y)]
    header_y = y
    y += _list_header(draw, cv, y)
    drawn["regions"]["header"] = [
        0, cv.s(header_y), cv.s(cv.w), cv.s(y - header_y)]
    if search_h:
        sy = y
        y += _list_search(draw, cv, y)
        drawn["regions"]["search"] = [
            0, cv.s(sy), cv.s(cv.w), cv.s(search_h)]
    if screen.stories:
        sy = y
        used, elements = _list_stories(img, draw, cv, y)
        drawn["elements"].extend(elements)
        drawn["regions"]["stories"] = [
            0, cv.s(sy), cv.s(cv.w), cv.s(used)]
        y += used
    if screen.tabs:
        ty = y
        y += _list_tabs(draw, cv, y)
        drawn["regions"]["tabs"] = [
            0, cv.s(ty), cv.s(cv.w), cv.s(tabs_h)]
    if screen.archived >= 0:
        arch = ListRow(
            id="archived", name="Archived Chats",
            row_type="archived", unread=screen.archived,
            preview=str(screen.archived) + " unread"
            if screen.archived else "")
        drawn["elements"].append(_list_row(img, draw, cv,
                                           arch, y))
        y += row_h
    seen_sections: set = set()
    for row in screen.rows:
        if row.section and row.section not in seen_sections:
            seen_sections.add(row.section)
            draw.text((cv.s(16), cv.s(y + 6)), row.section,
                      font=cv.font(13, bold=True),
                      fill=pal["section_fg"])
            drawn["elements"].append({
                "id": "section:" + row.section,
                "kind": "section", "text": row.section,
                "box": [cv.s(0), cv.s(y), cv.s(cv.w),
                        cv.s(28)]})
            y += 28.0
        if screen.flavor == "calls":
            drawn["elements"].append(
                _list_call_row(img, draw, cv, row, y))
        else:
            drawn["elements"].append(
                _list_row(img, draw, cv, row, y))
        y += row_h
    if nav_tabs:
        active = {"chats": "Chats", "status": "Updates",
                  "calls": "Calls"}[screen.flavor]
        by = total_h - bottom_h
        draw.rectangle(
            [0, cv.s(by), cv.s(cv.w), cv.s(total_h)],
            fill=pal["nav_bg"])
        draw.line([0, cv.s(by), cv.s(cv.w), cv.s(by)],
                  fill=pal["divider"], width=cv.s(0.75))
        slot = cv.w / len(nav_tabs)
        for i, label in enumerate(nav_tabs):
            cx = slot * i + slot / 2
            color = pal["nav_sel"] if label == active \
                else pal["nav_fg"]
            tw = cv.text_w(label, 11.5, bold=label == active)
            draw.text((cv.s(cx - tw / 2), cv.s(by + 26)),
                      label,
                      font=cv.font(11.5, bold=label == active),
                      fill=color)
            draw.ellipse(
                [cv.s(cx - 10), cv.s(by + 6), cv.s(cx + 10),
                 cv.s(by + 22)],
                outline=color, width=cv.s(1.6))
            if label == "Calls" and screen.nav_badge:
                _draw_badge(draw, cv, cx + 20, by + 8,
                            screen.nav_badge)
        drawn["regions"]["tab_bar"] = [
            0, cv.s(by), cv.s(cv.w), cv.s(bottom_h)]
    else:
        _draw_nav(draw, cv, total_h - bottom_h)
        drawn["regions"]["nav"] = [
            0, cv.s(total_h - bottom_h), cv.s(cv.w),
            cv.s(bottom_h)]
    return _finish(img), drawn


# ── status (census §4.2 + §4.1) ──────────────────────────


def _status_bg(cv: _StatusCanvas, screen: StatusScreen,
               img: Image.Image) -> tuple[str, tuple | None]:
    """Paint the status medium full-screen.  Returns
    (media_source, media_box): a photo status composes
    its REAL picture — cover-fit on ``fill``, contained
    over black or the photo's own colours on ``fit`` —
    and a supplied picture that cannot be loaded fails
    the render; no picture wears the placeholder scene.
    A text status paints its colour (a custom ``#rrggbb``
    or the design's stocked seven)."""
    w, h = cv.w, float(cv.dev["screen_h"])
    if screen.kind == "text":
        color = _hex_rgb(screen.bg) if screen.bg \
            else STATUS_BG_COLORS[0]
        ImageDraw.Draw(img).rectangle(
            [0, 0, cv.s(w), cv.s(h)], fill=color)
        return "none", None
    source = screen.media or None
    if source is None and "status" in cv.media_handoff:
        source = cv.media_handoff["status"]
    if source is None:
        _draw_media_placeholder(
            img, ImageDraw.Draw(img), cv, 0, 0, w, h,
            "status:" + screen.owner.name)
        return "placeholder", (0.0, 0.0, w, h)
    real = _strict_image(cv, source, "the status photo")
    if screen.fit == "fit":
        if screen.surround == "photo":
            backdrop = ImageOps.fit(
                real.convert("RGB"),
                (cv.s(w), cv.s(h)), Image.LANCZOS)
            backdrop = backdrop.resize(
                (max(1, cv.s(w) // 24), max(1, cv.s(h) // 24)))
            backdrop = backdrop.resize(
                (cv.s(w), cv.s(h)), Image.LANCZOS)
            img.paste(backdrop, (0, 0))
        contained = real.convert("RGB")
        contained.thumbnail((cv.s(w), cv.s(h)), Image.LANCZOS)
        img.paste(contained,
                  ((cv.s(w) - contained.width) // 2,
                   (cv.s(h) - contained.height) // 2))
        return "real", (0.0, 0.0, w, h)
    tile = ImageOps.fit(real.convert("RGB"),
                        (cv.s(w), cv.s(h)), Image.LANCZOS)
    img.paste(tile, (0, 0))
    return "real", (0.0, 0.0, w, h)


def render_status_screen(screen: StatusScreen, *,
                         avatars: dict | None = None,
                         media: dict | None = None):
    """Render one WhatsApp Status screen, full phone
    height: the medium, the segmented progress (segments
    / index / played fraction in the record), the owner's
    header line, and the reply bar with its quick
    reactions and like heart."""
    _require(screen.platform, STATUS_DESIGNS, "status")
    cv = _StatusCanvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})
    pal = cv.pal
    if screen.kind == "text":
        # the phone bar blends into the status colour
        color = _hex_rgb(screen.bg) if screen.bg \
            else STATUS_BG_COLORS[0]
        cv.pal = {**pal, "status_bg": color}
        pal = cv.pal
    w = cv.w
    total_h = float(cv.dev["screen_h"])
    img = Image.new("RGB", (cv.s(w), cv.s(total_h)),
                    pal["scrim"])
    cv.image = img
    draw = ImageDraw.Draw(img)
    media_source, media_box = _status_bg(cv, screen, img)
    draw = ImageDraw.Draw(img)
    drawn: dict = {
        "platform": screen.platform, "kind": screen.kind,
        "style": screen.chrome.style, "dark": screen.dark,
        "size": [cv.s(w), cv.s(total_h)],
        "regions": {}, "elements": [],
        "progress": {"segments": screen.segments,
                     "index": screen.index,
                     "played": screen.played},
        "liked": screen.liked,
    }
    if screen.kind == "photo":
        _darken(img, cv, 0, 130)
        if screen.caption:
            _darken(img, cv, total_h - 150, total_h)
        draw = ImageDraw.Draw(img)
    y = _draw_status_bar(draw, cv, 0.0)
    drawn["regions"]["status_bar"] = [0, 0, cv.s(w), cv.s(y)]
    # progress segments
    seg_y = y + 8
    n = screen.segments
    gap = 4.0
    seg_w = (w - 16 - (n - 1) * gap) / n
    for i in range(n):
        x = 8 + i * (seg_w + gap)
        draw.rounded_rectangle(
            [cv.s(x), cv.s(seg_y), cv.s(x + seg_w),
             cv.s(seg_y + 3)], radius=cv.s(1.5),
            fill=pal["track"])
        frac = 1.0 if i < screen.index else (
            screen.played if i == screen.index else 0.0)
        if frac > 0:
            draw.rounded_rectangle(
                [cv.s(x), cv.s(seg_y),
                 cv.s(x + seg_w * frac), cv.s(seg_y + 3)],
                radius=cv.s(1.5), fill=pal["fill"])
    drawn["regions"]["progress"] = [
        cv.s(8), cv.s(seg_y), cv.s(w - 16), cv.s(3)]
    # owner header line
    hy = seg_y + 14
    source = _draw_avatar_strict(
        img, draw, cv, 12, hy, 36, screen.owner,
        screen.owner.name)
    drawn["elements"].append({
        "id": "owner", "kind": "status_owner",
        "avatar_source": source,
        "box": [cv.s(12), cv.s(hy), cv.s(36), cv.s(36)]})
    draw.text((cv.s(58), cv.s(hy + 1)), screen.owner.name,
              font=cv.font(15, bold=True), fill=pal["name_fg"])
    when = screen.when + (
        "  •  " + screen.label if screen.label else "")
    draw.text((cv.s(58), cv.s(hy + 21)), when,
              font=cv.font(12), fill=pal["meta_fg"])
    _draw_dots(draw, cv, w - 16, hy + 16, pal["name_fg"])
    if screen.reshare:
        draw.text((cv.s(w - 44), cv.s(hy + 6)), "↗",
                  font=cv.font(16), fill=pal["name_fg"])
    drawn["regions"]["header"] = [
        0, cv.s(hy), cv.s(w), cv.s(38)]
    if media_box is not None:
        drawn["elements"].append({
            "id": "media", "kind": "status_media",
            "media_source": media_source,
            "fit": screen.fit, "surround": screen.surround,
            "box": [cv.s(media_box[0]), cv.s(media_box[1]),
                    cv.s(media_box[2]), cv.s(media_box[3])]})
    if screen.kind == "text" and screen.text:
        lines = cv.wrap(screen.text, screen.text_size,
                        w - 64)
        line_h = screen.text_size * 1.3
        ty = (total_h - len(lines) * line_h) / 2
        for line in lines:
            tw = cv.text_w(line, screen.text_size)
            draw.text((cv.s((w - tw) / 2), cv.s(ty)), line,
                      font=cv.font(screen.text_size),
                      fill=pal["text_fg"])
            ty += line_h
        drawn["elements"].append({
            "id": "text", "kind": "status_text",
            "text": screen.text,
            "box": [cv.s(32), cv.s((total_h
                                    - len(lines) * line_h) / 2),
                    cv.s(w - 64), cv.s(len(lines) * line_h)]})
    # reply bar
    nav_h = float(cv.dev["nav_h"])
    if screen.caption:
        cap_lines = cv.wrap(screen.caption, 14, w - 40)[:3]
        cy = total_h - nav_h - (64 if screen.reply_bar else 8) \
            - len(cap_lines) * 19 - 10
        for line in cap_lines:
            tw = cv.text_w(line, 14)
            draw.text((cv.s((w - tw) / 2), cv.s(cy)), line,
                      font=cv.font(14), fill=pal["caption_fg"])
            cy += 19
        drawn["elements"].append({
            "id": "caption", "kind": "status_caption",
            "text": screen.caption})
    if screen.reply_bar:
        by = total_h - nav_h - 56
        # quick reactions above the bar, right-aligned
        rx = w - 16
        for glyph in reversed(screen.reactions[:3]):
            tile = _emoji_tile(glyph, cv.s(26))
            if tile is not None:
                img.paste(tile, (cv.s(rx - 26),
                                 cv.s(by - 36)), tile)
            else:
                draw.text((cv.s(rx - 24), cv.s(by - 36)),
                          glyph, font=cv.font(20),
                          fill=pal["name_fg"])
            rx -= 34
        pill_w = w - 32 - (52 if screen.like else 0)
        draw.rounded_rectangle(
            [cv.s(16), cv.s(by), cv.s(16 + pill_w),
             cv.s(by + 44)], radius=cv.s(22),
            fill=pal["reply_bg"],
            outline=pal["name_fg"]
            if screen.reply_style == "glass" else None,
            width=cv.s(1.2))
        draw.text((cv.s(34), cv.s(by + 13)),
                  screen.reply_text or "Reply…",
                  font=cv.font(14), fill=pal["reply_hint"])
        if screen.like:
            hx = 16 + pill_w + 26
            draw.text((cv.s(hx - 11), cv.s(by + 8)), "♥",
                      font=cv.font(24),
                      fill=pal["heart_liked"] if screen.liked
                      else pal["heart"])
        drawn["regions"]["reply_bar"] = [
            0, cv.s(by), cv.s(w), cv.s(44)]
        drawn["elements"].append({
            "id": "reply", "kind": "status_reply_bar",
            "reactions": list(screen.reactions[:3]),
            "liked": screen.liked,
            "box": [cv.s(16), cv.s(by), cv.s(pill_w),
                    cv.s(44)]})
    _draw_nav(draw, cv, total_h - nav_h)
    drawn["regions"]["nav"] = [
        0, cv.s(total_h - nav_h), cv.s(w), cv.s(nav_h)]
    return _finish(img), drawn


def render_status_views(screen: StatusViewsScreen, *,
                        avatars: dict | None = None,
                        media: dict | None = None):
    """Render the Status Views screen (census §4.1): my
    status summary at the top, the views sheet under it —
    the view count and one row per viewer, each viewer's
    avatar under the media hand-off contract."""
    _require(screen.platform, STATUS_DESIGNS, "status")
    cv = _StatusCanvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})
    pal = cv.pal
    # the phone bar wears the sheet's own colours here
    cv.pal = {**pal, "status_bg": pal["sheet_bg"],
              "status_fg": pal["sheet_fg"]}
    pal = cv.pal
    w = cv.w
    status_h = float(cv.dev["status_h"])
    row_h = 64.0
    total_h = max(status_h + 64 + 46 + len(screen.viewers)
                  * row_h + 30,
                  float(cv.dev["screen_h"]) * 0.55)
    total_h = int(total_h)
    img = Image.new("RGB", (cv.s(w), cv.s(total_h)),
                    pal["sheet_bg"])
    cv.image = img
    draw = ImageDraw.Draw(img)
    drawn: dict = {
        "platform": screen.platform,
        "style": screen.chrome.style, "dark": screen.dark,
        "size": [cv.s(w), cv.s(total_h)],
        "regions": {},
        "view_count": len(screen.viewers),
        "elements": [],
    }
    y = _draw_status_bar(draw, cv, 0.0)
    drawn["regions"]["status_bar"] = [0, 0, cv.s(w), cv.s(y)]
    # my-status summary row
    source = _draw_avatar_strict(
        img, draw, cv, 14, y + 10, 44, screen.owner,
        screen.owner.name)
    draw.text((cv.s(70), cv.s(y + 13)), "My Status",
              font=cv.font(15.5, bold=True),
              fill=pal["sheet_fg"])
    draw.text((cv.s(70), cv.s(y + 34)), screen.when,
              font=cv.font(12.5), fill=pal["sheet_meta"])
    drawn["elements"].append({
        "id": "owner", "kind": "status_owner",
        "avatar_source": source,
        "box": [cv.s(14), cv.s(y + 10), cv.s(44), cv.s(44)]})
    y += 64
    # the sheet header: eye + count
    draw.line([0, cv.s(y), cv.s(w), cv.s(y)],
              fill=pal["divider"], width=cv.s(1))
    draw.ellipse([cv.s(16), cv.s(y + 15), cv.s(34),
                  cv.s(y + 29)], outline=pal["sheet_fg"],
                 width=cv.s(1.8))
    draw.ellipse([cv.s(21.5), cv.s(y + 19.5), cv.s(28.5),
                  cv.s(y + 26.5)], fill=pal["sheet_fg"])
    label = str(len(screen.viewers)) + " views"
    draw.text((cv.s(42), cv.s(y + 13)), label,
              font=cv.font(14.5, bold=True),
              fill=pal["sheet_fg"])
    drawn["regions"]["sheet_header"] = [
        0, cv.s(y), cv.s(w), cv.s(46)]
    y += 46
    for viewer in screen.viewers:
        person = screen.participant(viewer.participant_id)
        vsource = _draw_avatar_strict(
            img, draw, cv, 14, y + 10, 44, person, viewer.name)
        draw.text((cv.s(70), cv.s(y + 11)), viewer.name,
                  font=cv.font(15), fill=pal["sheet_fg"])
        draw.text((cv.s(70), cv.s(y + 32)), viewer.when,
                  font=cv.font(12.5), fill=pal["sheet_meta"])
        drawn["elements"].append({
            "id": viewer.participant_id or viewer.name,
            "kind": "viewer", "name": viewer.name,
            "when": viewer.when,
            "avatar_source": vsource,
            "box": [cv.s(0), cv.s(y), cv.s(w), cv.s(row_h)]})
        draw.line([cv.s(70), cv.s(y + row_h - 0.5),
                   cv.s(w), cv.s(y + row_h - 0.5)],
                  fill=pal["divider"], width=cv.s(0.75))
        y += row_h
    return _finish(img), drawn


# ── calls (census §6.1 + §6.2) ───────────────────────────

_CONTROL_LABELS: dict[str, str] = {
    "mute": "Mute", "keypad": "Keypad", "speaker": "Speaker",
    "video": "Video", "message": "Message", "add": "Add",
    "remind": "Remind me", "decline": "Decline",
    "accept": "Accept", "end": "",
}


def _control_glyph(draw, cv: _CallCanvas, name: str,
                   cx: float, cy: float, color) -> None:
    """One control's glyph, drawn as vectors (the call
    icon vocabulary the census §26 names as new)."""
    if name == "mute":
        _draw_mic(draw, cv, cx, cy - 2, color, r=5.0)
        draw.line([cv.s(cx - 11), cv.s(cy + 11),
                   cv.s(cx + 11), cv.s(cy - 13)], fill=color,
                  width=cv.s(2.2))
    elif name == "keypad":
        for r in range(3):
            for c in range(3):
                draw.ellipse(
                    [cv.s(cx - 10 + c * 10 - 2.2),
                     cv.s(cy - 10 + r * 10 - 2.2),
                     cv.s(cx - 10 + c * 10 + 2.2),
                     cv.s(cy - 10 + r * 10 + 2.2)], fill=color)
    elif name == "speaker":
        draw.polygon([(cv.s(cx - 12), cv.s(cy - 4)),
                      (cv.s(cx - 5), cv.s(cy - 4)),
                      (cv.s(cx + 2), cv.s(cy - 11)),
                      (cv.s(cx + 2), cv.s(cy + 11)),
                      (cv.s(cx - 5), cv.s(cy + 4)),
                      (cv.s(cx - 12), cv.s(cy + 4))],
                     fill=color)
        for rr in (8.0, 12.5):
            draw.arc([cv.s(cx + 2 - rr), cv.s(cy - rr),
                      cv.s(cx + 2 + rr), cv.s(cy + rr)],
                     start=-55, end=55, fill=color,
                     width=cv.s(1.8))
    elif name == "video":
        draw.rounded_rectangle(
            [cv.s(cx - 12), cv.s(cy - 8), cv.s(cx + 3),
             cv.s(cy + 8)], radius=cv.s(3), fill=color)
        draw.polygon([(cv.s(cx + 5), cv.s(cy - 3)),
                      (cv.s(cx + 13), cv.s(cy - 8)),
                      (cv.s(cx + 13), cv.s(cy + 8)),
                      (cv.s(cx + 5), cv.s(cy + 3))],
                     fill=color)
    elif name == "message":
        draw.rounded_rectangle(
            [cv.s(cx - 12), cv.s(cy - 9), cv.s(cx + 12),
             cv.s(cy + 6)], radius=cv.s(5), outline=color,
            width=cv.s(2))
        draw.polygon([(cv.s(cx - 5), cv.s(cy + 6)),
                      (cv.s(cx - 5), cv.s(cy + 13)),
                      (cv.s(cx + 3), cv.s(cy + 6))],
                     fill=color)
    elif name == "add":
        draw.text((cv.s(cx - 8), cv.s(cy - 13)), "+",
                  font=cv.font(24, bold=True), fill=color)
    elif name == "remind":
        draw.ellipse([cv.s(cx - 11), cv.s(cy - 11),
                      cv.s(cx + 11), cv.s(cy + 11)],
                     outline=color, width=cv.s(2))
        draw.line([cv.s(cx), cv.s(cy), cv.s(cx),
                   cv.s(cy - 7)], fill=color, width=cv.s(2))
        draw.line([cv.s(cx), cv.s(cy), cv.s(cx + 5),
                   cv.s(cy + 2)], fill=color, width=cv.s(2))
    else:  # accept / decline / end: the phone glyph
        glyph = "☎"
        tw = cv.text_w(glyph, 24)
        draw.text((cv.s(cx - tw / 2), cv.s(cy - 13)), glyph,
                  font=cv.font(24), fill=color)


def _control_disc(canvas, draw, cv: _CallCanvas, name: str,
                  cx: float, cy: float, size: float,
                  *, label: bool = True) -> dict:
    pal = cv.pal
    fill = pal["control_bg"]
    glyph_color = pal["control_fg"]
    if name == "accept":
        fill, glyph_color = pal["accept"], (255, 255, 255)
    elif name == "decline":
        fill, glyph_color = pal["decline"], (255, 255, 255)
    elif name == "end":
        fill, glyph_color = pal["end"], (255, 255, 255)
    draw.ellipse([cv.s(cx - size / 2), cv.s(cy - size / 2),
                  cv.s(cx + size / 2), cv.s(cy + size / 2)],
                 fill=fill)
    _control_glyph(draw, cv, name, cx, cy, glyph_color)
    if label and _CONTROL_LABELS.get(name):
        text = _CONTROL_LABELS[name]
        tw = cv.text_w(text, 11)
        draw.text((cv.s(cx - tw / 2), cv.s(cy + size / 2 + 4)),
                  text, font=cv.font(11), fill=pal["sub"])
    return {"kind": "control", "name": name,
            "box": [cv.s(cx - size / 2), cv.s(cy - size / 2),
                    cv.s(size), cv.s(size)]}


def render_call_screen(screen: CallScreen, *,
                       avatars: dict | None = None,
                       media: dict | None = None):
    """Render one call screen (census §6.2's anatomy on
    the named platforms' designs): the caller — picture
    or letter avatar — the state and status line, and
    the control cluster (the design's default for the
    state, or the screen's own).  A video call with a
    picture wears it as the full-screen feed."""
    _require(screen.platform, CALL_DESIGNS, "call")
    if screen.platform == "iphone_history":
        raise MockScreensError(
            "iphone_history is the call-history design — "
            "render it with render_call_history")
    cv = _CallCanvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})
    design = cv.design
    w = cv.w
    total_h = float(cv.dev["screen_h"])
    feed_source = screen.media or None
    if feed_source is None and "caller" in cv.media_handoff:
        feed_source = cv.media_handoff["caller"]
    feed = None
    if feed_source is not None:
        feed = _strict_image(cv, feed_source,
                             "the caller picture")
    full_feed = bool(screen.video and feed is not None)
    if full_feed:
        cv.pal = design["dark"]
    pal = cv.pal
    img = Image.new("RGB", (cv.s(w), cv.s(total_h)),
                    pal["bg"])
    cv.image = img
    if full_feed:
        tile = ImageOps.fit(feed.convert("RGB"),
                            (cv.s(w), cv.s(total_h)),
                            Image.LANCZOS)
        img.paste(tile, (0, 0))
        _darken(img, cv, 0, 150)
        _darken(img, cv, total_h - 240, total_h)
    draw = ImageDraw.Draw(img)
    drawn: dict = {
        "platform": screen.platform,
        "style": screen.chrome.style, "dark": screen.dark,
        "state": screen.state, "video": screen.video,
        "size": [cv.s(w), cv.s(total_h)],
        "regions": {}, "elements": [],
    }
    y = _draw_status_bar(draw, cv, 0.0)
    drawn["regions"]["status_bar"] = [0, 0, cv.s(w), cv.s(y)]
    name = screen.display_name()
    status = screen.status or (
        ("Incoming video call" if screen.video
         else "Incoming voice call")
        if screen.state == "incoming" else "00:00")
    drawn["status"] = status
    elements: list = drawn["elements"]
    if full_feed:
        draw.text((cv.s(18), cv.s(y + 14)), name,
                  font=cv.font(20, bold=True), fill=pal["fg"])
        draw.text((cv.s(18), cv.s(y + 42)), status,
                  font=cv.font(13.5), fill=pal["sub"])
        elements.append({
            "id": "caller", "kind": "caller_feed",
            "name": name, "media_source": "real",
            "box": [0, 0, cv.s(w), cv.s(total_h)]})
        controls_top = total_h - 210
    else:
        av = 118.0
        ax = (w - av) / 2
        ay = y + 54
        if feed is not None:
            if not _makers._paste_avatar_shape(
                    img, (cv.s(ax), cv.s(ay)), cv.s(av), feed,
                    "circle"):
                raise MockScreensError(
                    "the caller picture could not be used: "
                    "it did not paste")
            avatar_source = "real"
        else:
            avatar_source = _draw_avatar_strict(
                img, draw, cv, ax, ay, av, screen.caller, name)
        elements.append({
            "id": "caller", "kind": "caller",
            "name": name, "avatar_source": avatar_source,
            "media_source": "real" if feed is not None
            else "none",
            "box": [cv.s(ax), cv.s(ay), cv.s(av), cv.s(av)]})
        ny = ay + av + 16
        tw = cv.text_w(name, 22, bold=True)
        draw.text((cv.s((w - tw) / 2), cv.s(ny)), name,
                  font=cv.font(22, bold=True), fill=pal["fg"])
        sy = ny + 32
        if screen.caller.name and screen.number:
            tw = cv.text_w(screen.number, 14)
            draw.text((cv.s((w - tw) / 2), cv.s(sy)),
                      screen.number, font=cv.font(14),
                      fill=pal["sub"])
            sy += 22
        tw = cv.text_w(status, 14.5)
        draw.text((cv.s((w - tw) / 2), cv.s(sy)), status,
                  font=cv.font(14.5), fill=pal["sub"])
        if screen.sim:
            chip = screen.sim
            tw = cv.text_w(chip, 11, bold=True) + 16
            cx = (w - tw) / 2
            draw.rounded_rectangle(
                [cv.s(cx), cv.s(sy + 26), cv.s(cx + tw),
                 cv.s(sy + 46)], radius=cv.s(10),
                fill=pal["control_bg"])
            draw.text((cv.s(cx + 8), cv.s(sy + 30)), chip,
                      font=cv.font(11, bold=True),
                      fill=pal["control_fg"])
        controls_top = total_h - 250
    # the control cluster
    controls = list(screen.controls) or list(
        design["controls_incoming"]
        if screen.state == "incoming"
        else design["controls_ongoing"])
    drawn["controls"] = controls
    pair = [c for c in controls if c in ("decline", "accept")]
    grid = [c for c in controls
            if c not in ("decline", "accept", "end")]
    cy = controls_top
    if grid:
        per_row = 3
        for start in range(0, len(grid), per_row):
            row_items = grid[start:start + per_row]
            for i, control in enumerate(row_items):
                cx = w * (i + 1) / (len(row_items) + 1) \
                    if len(row_items) < per_row else \
                    w * (i + 0.5) / per_row
                elements.append(_control_disc(
                    img, draw, cv, control, cx, cy + 30, 58.0))
            cy += 88
    if "end" in controls:
        pw, ph = 128.0, 56.0
        cx = w / 2
        draw.rounded_rectangle(
            [cv.s(cx - pw / 2), cv.s(cy + 6),
             cv.s(cx + pw / 2), cv.s(cy + 6 + ph)],
            radius=cv.s(ph / 2), fill=pal["end"])
        _control_glyph(draw, cv, "end", cx, cy + 6 + ph / 2,
                       (255, 255, 255))
        elements.append({
            "kind": "control", "name": "end",
            "box": [cv.s(cx - pw / 2), cv.s(cy + 6),
                    cv.s(pw), cv.s(ph)]})
        cy += 6 + ph + 10
    if pair:
        py = total_h - float(cv.dev["nav_h"]) - 108
        for control in pair:
            cx = w * 0.28 if control == "decline" \
                else w * 0.72
            elements.append(_control_disc(
                img, draw, cv, control, cx, py + 34, 66.0))
    nav_h = float(cv.dev["nav_h"])
    _draw_nav(draw, cv, total_h - nav_h)
    drawn["regions"]["nav"] = [
        0, cv.s(total_h - nav_h), cv.s(w), cv.s(nav_h)]
    drawn["regions"]["controls"] = [
        0, cv.s(controls_top), cv.s(w),
        cv.s(total_h - nav_h - controls_top)]
    return _finish(img), drawn


# ── iPhone call history (census §6.3) ────────────────────


def _history_avatar(canvas, draw, cv: _CallCanvas, entry,
                    x: float, y: float, size: float,
                    person) -> str:
    """A Recents avatar in its census mode: a picture
    (the media hand-off contract), a letter disc, or the
    default silhouette."""
    pal = cv.pal
    if entry.avatar_mode == "picture":
        return _draw_avatar_strict(
            canvas, draw, cv, x, y, size, person, entry.name)
    if entry.avatar_mode == "letter":
        color = (_participant_color(person, cv.screen)
                 if person is not None
                 else _makers._medallion_color(entry.name))
        _makers._draw_placeholder(
            draw, (cv.s(x), cv.s(y)), cv.s(size), entry.name,
            color, shape="circle")
        return "placeholder"
    draw.ellipse([cv.s(x), cv.s(y), cv.s(x + size),
                  cv.s(y + size)], fill=pal["segment_bg"])
    cx, cy = x + size / 2, y + size / 2
    draw.ellipse([cv.s(cx - size * 0.13),
                  cv.s(cy - size * 0.24),
                  cv.s(cx + size * 0.13),
                  cv.s(cy + size * 0.02)],
                 fill=pal["tab_fg"])
    draw.pieslice([cv.s(cx - size * 0.26),
                   cv.s(cy + size * 0.02),
                   cv.s(cx + size * 0.26),
                   cv.s(cy + size * 0.5)], start=180, end=360,
                  fill=pal["tab_fg"])
    return "placeholder"


def render_call_history(screen: CallHistoryScreen, *,
                        avatars: dict | None = None,
                        media: dict | None = None):
    """Render the iPhone Call History (Recents) screen:
    the editable header texts, the All/Missed segment,
    the search field, the entries (type in the record —
    missed never relies on its red alone) and the footer
    tab bar with the voicemail badge."""
    _require(screen.platform, CALL_DESIGNS, "call")
    cv = _CallCanvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})
    pal = cv.pal
    w = cv.w
    status_h = float(cv.dev["status_h"])
    row_h = 58.0
    tab_h = 54.0
    head_h = 46.0 + 44.0 + 42.0
    content_h = status_h + head_h \
        + len(screen.entries) * row_h + tab_h
    total_h = int(max(content_h,
                      float(cv.dev["screen_h"]) * 0.7))
    img = Image.new("RGB", (cv.s(w), cv.s(total_h)),
                    pal["bg"])
    cv.image = img
    draw = ImageDraw.Draw(img)
    drawn: dict = {
        "platform": screen.platform,
        "style": screen.chrome.style, "dark": screen.dark,
        "size": [cv.s(w), cv.s(total_h)],
        "regions": {},
        "voicemail_count": screen.voicemail_count,
        "tabs": list(screen.tabs),
        "active_tab": screen.active_tab,
        "elements": [],
    }
    y = _draw_status_bar(draw, cv, 0.0)
    drawn["regions"]["status_bar"] = [0, 0, cv.s(w), cv.s(y)]
    # header: Edit · All|Missed segment · +
    accent = cv.design.get("accent", pal["fg"])
    draw.text((cv.s(16), cv.s(y + 12)), screen.edit_text,
              font=cv.font(15), fill=accent)
    seg_w = 150.0
    sx = (w - seg_w) / 2
    draw.rounded_rectangle(
        [cv.s(sx), cv.s(y + 10), cv.s(sx + seg_w),
         cv.s(y + 36)], radius=cv.s(8), fill=pal["segment_bg"])
    draw.rounded_rectangle(
        [cv.s(sx + 2), cv.s(y + 12), cv.s(sx + seg_w / 2),
         cv.s(y + 34)], radius=cv.s(6), fill=pal["segment_sel"])
    for i, label in enumerate((screen.all_text,
                               screen.missed_text)):
        tw = cv.text_w(label, 13, bold=True)
        cx = sx + seg_w * i / 2 + (seg_w / 2 - tw) / 2
        draw.text((cv.s(cx), cv.s(y + 15)), label,
                  font=cv.font(13, bold=True), fill=pal["fg"])
    draw.text((cv.s(w - 30), cv.s(y + 8)), "+",
              font=cv.font(20, bold=True), fill=accent)
    y += 46
    draw.text((cv.s(16), cv.s(y + 4)), screen.title_text,
              font=cv.font(28, bold=True), fill=pal["fg"])
    y += 44
    draw.rounded_rectangle(
        [cv.s(12), cv.s(y + 4), cv.s(w - 12), cv.s(y + 36)],
        radius=cv.s(10), fill=pal["search_bg"])
    _draw_search_icon(draw, cv, 32, y + 20,
                      pal["search_fg"], r=5.5)
    draw.text((cv.s(48), cv.s(y + 12)),
              screen.search_text or "Search",
              font=cv.font(14), fill=pal["search_fg"])
    y += 42
    drawn["regions"]["header"] = [
        0, cv.s(status_h), cv.s(w), cv.s(head_h)]
    for entry in screen.entries:
        person = screen.participant(entry.participant_id)
        source = _history_avatar(
            img, draw, cv, entry, 14, y + 9, 40, person)
        name_color = pal["missed_fg"] \
            if entry.type == "missed" else pal["fg"]
        label = entry.name + (
            " (" + str(entry.count) + ")"
            if entry.count > 1 else "")
        draw.text((cv.s(66), cv.s(y + 8)),
                  _fit_text(cv, label, 15, w - 170, bold=True),
                  font=cv.font(15, bold=True), fill=name_color)
        sub = entry.label
        if entry.sim:
            sub = (sub + " · " + entry.sim) if sub \
                else entry.sim
        if entry.type == "outgoing":
            sub = ("↗ " + sub) if sub else "↗"
        draw.text((cv.s(66), cv.s(y + 31)), sub,
                  font=cv.font(12), fill=pal["sub"])
        tw = cv.text_w(entry.when, 13)
        draw.text((cv.s(w - 40 - tw), cv.s(y + 10)),
                  entry.when, font=cv.font(13),
                  fill=pal["sub"])
        draw.ellipse([cv.s(w - 30), cv.s(y + 19),
                      cv.s(w - 14), cv.s(y + 35)],
                     outline=accent, width=cv.s(1.4))
        draw.text((cv.s(w - 26.5), cv.s(y + 20)), "i",
                  font=cv.font(11, bold=True), fill=accent)
        drawn["elements"].append({
            "id": entry.id, "kind": "history_entry",
            "type": entry.type,
            "avatar_mode": entry.avatar_mode,
            "avatar_source": source,
            "count": entry.count, "name": entry.name,
            "box": [cv.s(0), cv.s(y), cv.s(w), cv.s(row_h)]})
        draw.line([cv.s(66), cv.s(y + row_h - 0.5),
                   cv.s(w), cv.s(y + row_h - 0.5)],
                  fill=pal["divider"], width=cv.s(0.75))
        y += row_h
    # footer tab bar, pinned
    by = total_h - tab_h
    draw.rectangle([0, cv.s(by), cv.s(w), cv.s(total_h)],
                   fill=pal["segment_bg"])
    draw.line([0, cv.s(by), cv.s(w), cv.s(by)],
              fill=pal["divider"], width=cv.s(0.75))
    glyphs = {"Favourites": "★", "Calls": "☎",
              "Contacts": "👤", "Voicemail": "➿"}
    slot = w / max(1, len(screen.tabs))
    for i, label in enumerate(screen.tabs):
        cx = slot * i + slot / 2
        active = label == screen.active_tab
        color = accent if active else pal["tab_fg"]
        if label == "Keypad":
            for r in range(3):
                for c in range(3):
                    draw.ellipse(
                        [cv.s(cx - 8 + c * 8 - 1.8),
                         cv.s(by + 8 + r * 8 - 1.8),
                         cv.s(cx - 8 + c * 8 + 1.8),
                         cv.s(by + 8 + r * 8 + 1.8)],
                        fill=color)
        else:
            glyph = glyphs.get(label, "•")
            tw = cv.text_w(glyph, 17)
            draw.text((cv.s(cx - tw / 2), cv.s(by + 5)),
                      glyph, font=cv.font(17), fill=color)
        tw = cv.text_w(label, 10)
        draw.text((cv.s(cx - tw / 2), cv.s(by + 34)), label,
                  font=cv.font(10), fill=color)
        if label == "Voicemail" and screen.voicemail_count:
            _draw_badge(draw, cv, cx + 16, by + 10,
                        screen.voicemail_count)
    drawn["regions"]["tab_bar"] = [
        0, cv.s(by), cv.s(w), cv.s(tab_h)]
    return _finish(img), drawn
