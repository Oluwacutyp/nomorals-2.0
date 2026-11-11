"""The Phase 3 renderers: social posts, profiles, stories,
comments and video pages.

Five layout engines beside the chat and list engines
(:mod:`.render`, :mod:`.render_screens`), one per screen
class the census's Phase 3 names (§27), all reading the
platform designs held as data (:mod:`.designs` —
``SOCIAL_DESIGNS``) and all returning the makers
contract: PNG bytes plus a drawn record whose entries
carry each element's box and STATE, so tests prove
structure — a pinned video is its record flag, a creator
mark is a record field, a sticker's position is its
recorded fraction (the census §27 exit proofs).

Media follows the media hand-off contract (:mod:`.media`
through :mod:`.render_screens`' strict helpers): a
SUPPLIED picture — a post photo, a cover, a grid cover,
a story photo, a thumbnail — that cannot be loaded fails
the render with the reason, never a silent placeholder;
the placeholder scene stands in only where no media was
supplied at all.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageOps

from .designs import SOCIAL_DESIGNS
from .render import (
    _Canvas,
    _draw_media_placeholder,
    _draw_nav,
    _draw_status_bar,
)
from .render_screens import (
    _darken,
    _draw_avatar_strict,
    _finish,
    _require,
    _strict_image,
)
from .schema import (
    CommentScreen,
    PostScreen,
    ProfileScreen,
    StoryScreen,
    VideoScreen,
    shorten_count,
)

__all__ = [
    "render_comment_screen",
    "render_post_screen",
    "render_profile_screen",
    "render_story_screen",
    "render_video_screen",
]


class _SocialCanvas(_Canvas):
    DESIGNS = SOCIAL_DESIGNS


def _count(cv: _SocialCanvas, value: int, style: str = "") -> str:
    """Shorten a count in the screen's style ("" = the
    design's — YouTube's Indian shortening, the census
    §5.8, rides the design)."""
    return shorten_count(
        value, style or cv.design.get("count_style", "intl"))


def _verified(draw, cv: _SocialCanvas, x: float, y: float,
              r: float = 7.0) -> None:
    """The platform's verified tick: a filled disc with a
    white check."""
    color = cv.pal["verified"]
    draw.ellipse([cv.s(x), cv.s(y), cv.s(x + 2 * r),
                  cv.s(y + 2 * r)], fill=color)
    pts = [(x + r * 0.45, y + r * 1.05),
           (x + r * 0.85, y + r * 1.45),
           (x + r * 1.6, y + r * 0.6)]
    draw.line([(cv.s(px), cv.s(py)) for px, py in pts],
              fill=(0xFF, 0xFF, 0xFF), width=cv.s(1.6))


def _play(draw, cv: _SocialCanvas, cx: float, cy: float,
          r: float = 26.0) -> None:
    """The play button on a video preview / player."""
    draw.ellipse([cv.s(cx - r), cv.s(cy - r),
                  cv.s(cx + r), cv.s(cy + r)],
                 fill=(0x14, 0x14, 0x14))
    tri = [(cx - r * 0.35, cy - r * 0.55),
           (cx - r * 0.35, cy + r * 0.55),
           (cx + r * 0.6, cy)]
    draw.polygon([(cv.s(px), cv.s(py)) for px, py in tri],
                 fill=(0xFF, 0xFF, 0xFF))


def _media_tile(canvas, draw, cv: _SocialCanvas, source,
                x: float, y: float, w: float, h: float,
                seed: str, what: str) -> str:
    """One picture in a frame: the REAL image cover-fit
    when a source is supplied (a broken one fails the
    render with the reason), the placeholder scene only
    when none was.  Returns the source worn."""
    if source:
        img = _strict_image(cv, source, what)
        tile = ImageOps.fit(
            img.convert("RGB"),
            (max(1, cv.s(w)), max(1, cv.s(h))),
            Image.LANCZOS)
        canvas.paste(tile, (cv.s(x), cv.s(y)))
        return "real"
    _draw_media_placeholder(canvas, draw, cv, x, y, w, h,
                            seed)
    return "placeholder"


def _button(draw, cv: _SocialCanvas, x: float, y: float,
            w: float, h: float, label: str, *,
            primary: bool = False) -> None:
    """One action button (Follow / Message / Subscribe /
    Join): a rounded pill, primary in the accent."""
    pal = cv.pal
    bg = pal["primary_bg"] if primary else pal["button_bg"]
    fg = pal["primary_fg"] if primary else pal["button_fg"]
    draw.rounded_rectangle(
        [cv.s(x), cv.s(y), cv.s(x + w), cv.s(y + h)],
        radius=cv.s(h / 2), fill=bg)
    tw = cv.text_w(label, 13, bold=True)
    draw.text((cv.s(x + (w - tw) / 2), cv.s(y + h / 2 - 7)),
              label, font=cv.font(13, bold=True), fill=fg)


def _text_block(draw, cv: _SocialCanvas, text: str, x: float,
                y: float, max_w: float, size: float, color,
                *, bold: bool = False, leading: float = 1.35
                ) -> float:
    """Wrapped text; returns the height consumed."""
    lines = cv.wrap(text, size, max_w, bold=bold)
    for i, line in enumerate(lines):
        draw.text((cv.s(x), cv.s(y + i * size * leading)),
                  line, font=cv.font(size, bold=bold),
                  fill=color)
    return len(lines) * size * leading


def _base_drawn(screen, cv: _SocialCanvas, total_h: float,
                *, kind: str) -> dict:
    return {
        "platform": screen.platform,
        "kind": kind,
        "style": screen.chrome.style,
        "dark": screen.dark,
        "size": [cv.s(cv.w), cv.s(total_h)],
        "regions": {},
        "elements": [],
    }


def _start_canvas(cv: _SocialCanvas, total_h: float):
    """A fresh canvas + draw for a content-height screen,
    with the status bar drawn; returns (img, draw, y)."""
    pal = cv.pal
    img = Image.new("RGB", (cv.s(cv.w), cv.s(total_h)),
                    pal["bg"])
    cv.image = img
    draw = ImageDraw.Draw(img)
    y = _draw_status_bar(draw, cv, 0.0)
    return img, draw, y


# ── posts (census §5.2, §5.7, §5.10) ─────────────────────


def render_post_screen(screen: PostScreen, *,
                       avatars: dict | None = None,
                       media: dict | None = None):
    """Render one social post as a phone screen: the
    author row, the text, the photo (or the video-style
    preview with its play button) and the engagement
    row.  Returns ``(png_bytes, drawn)`` in the makers
    contract."""
    _require(screen.platform, SOCIAL_DESIGNS, "social")
    cv = _SocialCanvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})
    pal = cv.pal
    w = cv.w
    status_h = float(cv.dev["status_h"])
    bottom_h = float(cv.dev["nav_h"])

    author = screen.author
    name_w = cv.text_w(author.name, 15, bold=True) + 20
    sub = "@" + author.handle if author.handle else ""
    if screen.time:
        sub = (sub + " · " + screen.time) if sub \
            else screen.time
    if screen.visibility:
        sub = (sub + " · " + screen.visibility) if sub \
            else screen.visibility
    text_lines = cv.wrap(screen.text, 15, w - 24) \
        if screen.text else []
    text_h = len(text_lines) * 15 * 1.35 + 10
    media_h = 0.0
    if screen.media or screen.video:
        media_h = w * 0.75 + 12
    total_h = int(status_h + 64 + text_h + media_h + 52
                  + bottom_h + 8)

    img, draw, y = _start_canvas(cv, total_h)
    drawn = _base_drawn(screen, cv, total_h, kind="post")
    drawn["regions"]["status_bar"] = [0, 0, cv.s(w),
                                      cv.s(y)]
    # author row
    ay = y + 8
    source = _draw_avatar_strict(
        img, draw, cv, 12, ay, 40, author, author.name)
    draw.text((cv.s(60), cv.s(ay + 2)), author.name,
              font=cv.font(15, bold=True), fill=pal["fg"])
    if author.verified:
        _verified(draw, cv, 60 + name_w - 16, ay + 5)
    draw.text((cv.s(60), cv.s(ay + 22)), sub,
              font=cv.font(12.5), fill=pal["sub"])
    drawn["elements"].append({
        "id": "author", "kind": "post_author",
        "text": author.name, "handle": author.handle,
        "verified": bool(author.verified),
        "avatar_source": source,
        "box": [cv.s(12), cv.s(ay), cv.s(40), cv.s(40)]})
    y = ay + 52
    # text
    if text_lines:
        ty = y
        y += _text_block(draw, cv, screen.text, 12, y,
                         w - 24, 15, pal["fg"])
        drawn["elements"].append({
            "id": "text", "kind": "post_text",
            "text": screen.text,
            "box": [cv.s(12), cv.s(ty), cv.s(w - 24),
                    cv.s(y - ty)]})
        y += 10
    # photo / video preview
    if media_h:
        my = y
        src = screen.media or None
        if src is not None:
            src = cv.media_handoff.get("media", src)
        worn = _media_tile(
            img, draw, cv, src, 0, my, w, w * 0.75,
            "post:" + screen.platform, "the post photo")
        if screen.video:
            _play(draw, cv, w / 2, my + w * 0.375)
        drawn["elements"].append({
            "id": "media", "kind": "post_media",
            "media_source": worn, "video": bool(screen.video),
            "box": [0, cv.s(my), cv.s(w),
                    cv.s(w * 0.75)]})
        y = my + w * 0.75 + 12
    # engagement row
    ey = y + 6
    parts = [("♥", screen.likes), ("💬", screen.comments),
             ("↗", screen.shares)]
    ex = 12.0
    for glyph, count in parts:
        label = glyph + " " + _count(cv, count)
        tw = cv.text_w(label, 14)
        draw.text((cv.s(ex), cv.s(ey)), label,
                  font=cv.font(14), fill=pal["sub"])
        ex += tw + 26
    drawn["elements"].append({
        "id": "engagement", "kind": "post_engagement",
        "likes": screen.likes, "comments": screen.comments,
        "shares": screen.shares,
        "box": [cv.s(12), cv.s(ey), cv.s(ex), cv.s(20)]})
    _draw_nav(draw, cv, total_h - bottom_h)
    drawn["regions"]["nav"] = [
        0, cv.s(total_h - bottom_h), cv.s(w), cv.s(bottom_h)]
    return _finish(img), drawn


# ── profiles (census §5.3, §5.6, §5.8, §5.10) ────────────


def _profile_counts(draw, cv: _SocialCanvas, counts, x: float,
                    y: float, max_w: float, style: str,
                    *, size: float = 15) -> float:
    """The counts row (value over label, in the data's
    own order); returns the width consumed."""
    pal = cv.pal
    cx = x
    for count in counts:
        value = _count(cv, count.value,
                       count.style or style)
        vw = cv.text_w(value, size, bold=True)
        lw = cv.text_w(count.label, 11.5)
        col_w = max(vw, lw) + 18
        draw.text((cv.s(cx), cv.s(y)), value,
                  font=cv.font(size, bold=True),
                  fill=pal["fg"])
        draw.text((cv.s(cx), cv.s(y + size + 2)),
                  count.label, font=cv.font(11.5),
                  fill=pal["sub"])
        cx += col_w
    return cx - x


def _profile_grid_cell(canvas, draw, cv: _SocialCanvas, item,
                       x: float, y: float, size: float) -> dict:
    """One 3-column grid cell (Instagram / TikTok): the
    cover (real or placeholder), the play count on a
    video cover, the pinned chip, the cover caption."""
    src = cv.media_handoff.get(item.id, item.media or None)
    worn = _media_tile(
        canvas, draw, cv, src, x, y, size, size,
        "grid:" + item.id, "the cover for " + repr(item.id))
    el = {"id": item.id, "kind": "profile_item",
          "item_kind": item.kind, "media_source": worn,
          "pinned": bool(item.pinned),
          "box": [cv.s(x), cv.s(y), cv.s(size), cv.s(size)]}
    if item.kind in ("video", "short") and item.views >= 0:
        label = "▶ " + _count(cv, item.views)
        draw.text((cv.s(x + 6), cv.s(y + size - 20)), label,
                  font=cv.font(11, bold=True),
                  fill=(0xFF, 0xFF, 0xFF))
    if item.pinned:
        draw.rounded_rectangle(
            [cv.s(x + 4), cv.s(y + 4), cv.s(x + 44),
             cv.s(y + 18)], radius=cv.s(3),
            fill=(0x14, 0x14, 0x14))
        draw.text((cv.s(x + 8), cv.s(y + 6)), "PINNED",
                  font=cv.font(8, bold=True),
                  fill=(0xFF, 0xFF, 0xFF))
    if item.caption:
        draw.rectangle(
            [cv.s(x), cv.s(y + size - 20), cv.s(x + size),
             cv.s(y + size)], fill=(0x14, 0x14, 0x14))
        draw.text((cv.s(x + 6), cv.s(y + size - 17)),
                  item.caption, font=cv.font(10),
                  fill=(0xFF, 0xFF, 0xFF))
    if item.duration:
        label = item.duration
        tw = cv.text_w(label, 11, bold=True)
        draw.text((cv.s(x + size - tw - 6),
                   cv.s(y + size - 20)), label,
                  font=cv.font(11, bold=True),
                  fill=(0xFF, 0xFF, 0xFF))
    return el


def render_profile_screen(screen: ProfileScreen, *,
                          avatars: dict | None = None,
                          media: dict | None = None):
    """Render one profile page: the cover band where the
    platform wears one, the identity, the counts in the
    data's own order, the action buttons and tabs, and
    the grid (Instagram / TikTok, three columns) or the
    rows (YouTube videos + Shorts, Facebook / X posts).
    Returns ``(png_bytes, drawn)``."""
    _require(screen.platform, SOCIAL_DESIGNS, "social")
    cv = _SocialCanvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})
    pal = cv.pal
    design = cv.design
    w = cv.w
    status_h = float(cv.dev["status_h"])
    bottom_h = float(cv.dev["nav_h"])
    cover_h = float(design.get("cover_h", 0))
    platform = screen.platform
    ident = screen.identity
    style = screen.count_style or design.get(
        "count_style", "intl")
    grid_cols = int(design.get("grid_cols", 0))

    # measure: the identity block differs by platform —
    # each branch's head_h is exactly the height its
    # drawing code below consumes (counts ride inside
    # the header for instagram, below it elsewhere)
    name_h = 24.0
    bio_h = 0.0
    if screen.bio:
        bio_h = len(cv.wrap(screen.bio, 14, w - 24)) \
            * 14 * 1.35 + 6
    link_h = 20.0 if screen.link else 0.0
    handle_h = 22.0 if ident.handle else 0.0
    counts_h = 44.0 if screen.counts else 0.0
    buttons_h = 44.0 if screen.buttons else 0.0
    tabs_h = 40.0 if screen.tabs else 0.0
    if platform == "instagram":
        # avatar row (76 + margins) + name + bio + link;
        # the counts sit beside the avatar, inside the row
        head_h = 96.0 + name_h + bio_h + link_h
    elif platform == "tiktok":
        # name row + handle + counts + bio + link
        head_h = 40.0 + handle_h + counts_h + bio_h + link_h
    elif platform in ("threads", "x"):
        # avatar row (64 + margins) + counts + bio + link
        head_h = 84.0 + counts_h + bio_h + link_h
    else:  # facebook / youtube: cover + avatar overlap
        head_h = 46.0 + name_h + handle_h + counts_h \
            + bio_h + link_h
    # items block
    items_h = 0.0
    if screen.items:
        if grid_cols:
            cell = (w - 24 - (grid_cols - 1) * 2) / grid_cols
            rows_n = -(-len(screen.items) // grid_cols)
            items_h = rows_n * (cell + 2) + 12
        elif platform == "youtube":
            items_h = 12.0
            for item in screen.items:
                if item.kind == "short":
                    continue
                items_h += 96.0
            shorts = [i for i in screen.items
                      if i.kind == "short"]
            if shorts:
                items_h += 170.0
        else:
            items_h = 12.0
            for item in screen.items:
                lines = cv.wrap(item.title or item.caption,
                                14, w - 24)
                items_h += len(lines) * 14 * 1.35 + 34
    total_h = int(status_h + cover_h + head_h + buttons_h
                  + tabs_h + items_h + bottom_h + 8)

    img, draw, y = _start_canvas(cv, total_h)
    drawn = _base_drawn(screen, cv, total_h, kind="profile")
    drawn["regions"]["status_bar"] = [0, 0, cv.s(w),
                                      cv.s(y)]
    # cover band
    if cover_h:
        cy = y
        src = cv.media_handoff.get(
            "cover", screen.cover_media or None)
        worn = _media_tile(
            img, draw, cv, src, 0, cy, w, cover_h,
            "cover:" + platform, "the cover photo")
        drawn["regions"]["cover"] = [
            0, cv.s(cy), cv.s(w), cv.s(cover_h)]
        drawn["elements"].append({
            "id": "cover", "kind": "profile_cover",
            "media_source": worn,
            "box": [0, cv.s(cy), cv.s(w), cv.s(cover_h)]})
        y += cover_h
    # identity block
    avatar_size = 76.0
    if platform == "instagram":
        ay = y + 10
        source = _draw_avatar_strict(
            img, draw, cv, 12, ay, avatar_size, ident,
            ident.name)
        drawn["elements"].append({
            "id": "avatar", "kind": "profile_avatar",
            "avatar_source": source,
            "box": [cv.s(12), cv.s(ay), cv.s(avatar_size),
                    cv.s(avatar_size)]})
        _profile_counts(draw, cv, screen.counts, 116, ay + 18,
                        w - 128, style)
        y = ay + avatar_size + 8
        ny = y
        draw.text((cv.s(12), cv.s(ny)), ident.name,
                  font=cv.font(15, bold=True), fill=pal["fg"])
        if ident.verified:
            _verified(draw, cv,
                      12 + cv.text_w(ident.name, 15, bold=True)
                      + 4, ny + 3)
        y = ny + name_h
    elif platform == "tiktok":
        ay = y + 10
        # the census's TikTok shape: the name large on
        # the left, the picture on the right
        draw.text((cv.s(12), cv.s(ay)), ident.name,
                  font=cv.font(20, bold=True), fill=pal["fg"])
        if ident.verified:
            _verified(draw, cv,
                      12 + cv.text_w(ident.name, 20, bold=True)
                      + 4, ay + 6)
        source = _draw_avatar_strict(
            img, draw, cv, w - 88, ay, avatar_size, ident,
            ident.name)
        drawn["elements"].append({
            "id": "avatar", "kind": "profile_avatar",
            "avatar_source": source,
            "box": [cv.s(w - 88), cv.s(ay),
                    cv.s(avatar_size), cv.s(avatar_size)]})
        y = ay + 30
        if ident.handle:
            draw.text((cv.s(12), cv.s(y)), "@" + ident.handle,
                      font=cv.font(14), fill=pal["fg"])
            y += 22
        _profile_counts(draw, cv, screen.counts, 12, y, w - 24,
                        style)
        y += counts_h
    elif platform in ("threads", "x"):
        ay = y + 10
        source = _draw_avatar_strict(
            img, draw, cv, 12, ay, 64, ident, ident.name)
        drawn["elements"].append({
            "id": "avatar", "kind": "profile_avatar",
            "avatar_source": source,
            "box": [cv.s(12), cv.s(ay), cv.s(64), cv.s(64)]})
        draw.text((cv.s(84), cv.s(ay + 4)), ident.name,
                  font=cv.font(16, bold=True), fill=pal["fg"])
        if ident.verified:
            _verified(draw, cv,
                      84 + cv.text_w(ident.name, 16, bold=True)
                      + 4, ay + 7)
        if ident.handle:
            draw.text((cv.s(84), cv.s(ay + 26)),
                      "@" + ident.handle,
                      font=cv.font(13), fill=pal["sub"])
        y = ay + 74
        if screen.counts:
            _profile_counts(draw, cv, screen.counts, 12, y,
                            w - 24, style)
            y += counts_h
    else:  # facebook / youtube
        ay = y - 38
        source = _draw_avatar_strict(
            img, draw, cv, 12, ay, avatar_size, ident,
            ident.name)
        drawn["elements"].append({
            "id": "avatar", "kind": "profile_avatar",
            "avatar_source": source,
            "box": [cv.s(12), cv.s(ay), cv.s(avatar_size),
                    cv.s(avatar_size)]})
        y = ay + avatar_size + 8
        draw.text((cv.s(12), cv.s(y)), ident.name,
                  font=cv.font(18, bold=True), fill=pal["fg"])
        if ident.verified:
            _verified(draw, cv,
                      12 + cv.text_w(ident.name, 18, bold=True)
                      + 4, y + 5)
        y += name_h
        if ident.handle:
            draw.text((cv.s(12), cv.s(y)), "@" + ident.handle,
                      font=cv.font(13), fill=pal["sub"])
            y += 20
        if screen.counts:
            _profile_counts(draw, cv, screen.counts, 12, y,
                            w - 24, style)
            y += counts_h
    # bio + link
    if screen.bio:
        y += _text_block(draw, cv, screen.bio, 12, y, w - 24,
                         14, pal["fg"])
        y += 6
    if screen.link:
        draw.text((cv.s(12), cv.s(y)), screen.link,
                  font=cv.font(13), fill=pal["accent"])
        y += link_h
    # buttons
    if screen.buttons:
        bx = 12.0
        bw = (w - 24 - (len(screen.buttons) - 1) * 8) \
            / len(screen.buttons)
        for i, label in enumerate(screen.buttons):
            _button(draw, cv, bx, y + 6, bw, 32, label,
                    primary=(i == 0))
            bx += bw + 8
        drawn["elements"].append({
            "id": "buttons", "kind": "profile_buttons",
            "buttons": list(screen.buttons),
            "box": [cv.s(12), cv.s(y + 6), cv.s(w - 24),
                    cv.s(32)]})
        y += buttons_h
    # tabs
    if screen.tabs:
        tx = 12.0
        for tab in screen.tabs:
            tw = cv.text_w(tab, 13, bold=True) + 8
            draw.text((cv.s(tx), cv.s(y + 8)), tab,
                      font=cv.font(13, bold=True),
                      fill=pal["tab_sel"])
            tx += tw + 18
        draw.line([0, cv.s(y + 34), cv.s(w), cv.s(y + 34)],
                  fill=pal["divider"], width=cv.s(0.75))
        drawn["elements"].append({
            "id": "tabs", "kind": "profile_tabs",
            "tabs": list(screen.tabs),
            "box": [cv.s(0), cv.s(y), cv.s(w), cv.s(40)]})
        y += tabs_h
    # items
    if screen.items:
        y += 6
        if grid_cols:
            cell = (w - 24 - (grid_cols - 1) * 2) / grid_cols
            for idx, item in enumerate(screen.items):
                r, c = divmod(idx, grid_cols)
                ix = 12 + c * (cell + 2)
                iy = y + r * (cell + 2)
                drawn["elements"].append(_profile_grid_cell(
                    img, draw, cv, item, ix, iy, cell))
            rows_n = -(-len(screen.items) // grid_cols)
            y += rows_n * (cell + 2)
        elif platform == "youtube":
            shorts = [i for i in screen.items
                      if i.kind == "short"]
            videos = [i for i in screen.items
                      if i.kind != "short"]
            for item in videos:
                iy = y
                src = cv.media_handoff.get(
                    item.id, item.media or None)
                worn = _media_tile(
                    img, draw, cv, src, 12, iy, 150, 84,
                    "thumb:" + item.id,
                    "the thumbnail for " + repr(item.id))
                if item.duration:
                    tw = cv.text_w(item.duration, 10,
                                   bold=True)
                    draw.rectangle(
                        [cv.s(12 + 150 - tw - 10),
                         cv.s(iy + 84 - 18),
                         cv.s(12 + 150 - 4), cv.s(iy + 84 - 4)],
                        fill=(0x14, 0x14, 0x14))
                    draw.text(
                        (cv.s(12 + 150 - tw - 7),
                         cv.s(iy + 84 - 17)), item.duration,
                        font=cv.font(10, bold=True),
                        fill=(0xFF, 0xFF, 0xFF))
                draw.text((cv.s(172), cv.s(iy + 2)),
                          item.title or item.caption,
                          font=cv.font(13, bold=True),
                          fill=pal["fg"])
                meta = ""
                if item.views >= 0:
                    meta = _count(cv, item.views) + " views"
                draw.text((cv.s(172), cv.s(iy + 42)), meta,
                          font=cv.font(12), fill=pal["sub"])
                drawn["elements"].append({
                    "id": item.id, "kind": "profile_item",
                    "item_kind": item.kind,
                    "media_source": worn,
                    "pinned": bool(item.pinned),
                    "box": [cv.s(12), cv.s(iy), cv.s(w - 24),
                            cv.s(84)]})
                y += 96
            if shorts:
                draw.text((cv.s(12), cv.s(y + 4)), "Shorts",
                          font=cv.font(15, bold=True),
                          fill=pal["fg"])
                y += 28
                sx = 12.0
                for item in shorts:
                    src = cv.media_handoff.get(
                        item.id, item.media or None)
                    worn = _media_tile(
                        img, draw, cv, src, sx, y, 100, 150,
                        "short:" + item.id,
                        "the Short for " + repr(item.id))
                    drawn["elements"].append({
                        "id": item.id, "kind": "profile_item",
                        "item_kind": item.kind,
                        "media_source": worn,
                        "pinned": bool(item.pinned),
                        "box": [cv.s(sx), cv.s(y), cv.s(100),
                                cv.s(150)]})
                    sx += 108
                y += 150
        else:  # facebook / x / threads: post rows
            for item in screen.items:
                iy = y
                text = item.title or item.caption
                lines = cv.wrap(text, 14, w - 24)
                for li, line in enumerate(lines):
                    draw.text((cv.s(12), cv.s(iy + li * 19)),
                              line, font=cv.font(14),
                              fill=pal["fg"])
                used = len(lines) * 19 + 14
                draw.line([0, cv.s(iy + used), cv.s(w),
                           cv.s(iy + used)],
                          fill=pal["divider"], width=cv.s(0.75))
                drawn["elements"].append({
                    "id": item.id, "kind": "profile_item",
                    "item_kind": item.kind,
                    "media_source": "none",
                    "pinned": bool(item.pinned),
                    "box": [cv.s(12), cv.s(iy), cv.s(w - 24),
                            cv.s(used)]})
                y += used + 6
    _draw_nav(draw, cv, total_h - bottom_h)
    drawn["regions"]["nav"] = [
        0, cv.s(total_h - bottom_h), cv.s(w), cv.s(bottom_h)]
    return _finish(img), drawn


# ── stories (census §5.4) ────────────────────────────────


def _sticker_chip(draw, cv: _SocialCanvas, sticker, cx: float,
                  cy: float) -> tuple[float, float]:
    """One sticker at its fractional position: a pill
    carrying the kind's text (a poll/quiz wears its
    options under it).  Returns (w, h) consumed."""
    pal = cv.pal
    if sticker.kind == "mention":
        label = "@" + sticker.text
    elif sticker.kind == "hashtag":
        label = "#" + sticker.text
    elif sticker.kind == "location":
        label = "📍 " + sticker.text
    elif sticker.kind == "music":
        label = "♪ " + sticker.text
    else:
        label = sticker.text
    tw = cv.text_w(label, 14, bold=True)
    w = tw + 20
    h = 30.0
    x = cx - w / 2
    y = cy - h / 2
    draw.rounded_rectangle(
        [cv.s(x), cv.s(y), cv.s(x + w), cv.s(y + h)],
        radius=cv.s(15), fill=pal["card_bg"])
    draw.text((cv.s(x + 10), cv.s(y + 7)), label,
              font=cv.font(14, bold=True), fill=pal["fg"])
    if sticker.options and sticker.kind in ("poll", "quiz"):
        oy = y + h + 4
        for opt in sticker.options:
            ow = cv.text_w(opt, 12, bold=True) + 20
            draw.rounded_rectangle(
                [cv.s(cx - ow / 2), cv.s(oy),
                 cv.s(cx + ow / 2), cv.s(oy + 26)],
                radius=cv.s(13), fill=pal["card_bg"])
            draw.text((cv.s(cx - ow / 2 + 10), cv.s(oy + 5)),
                      opt, font=cv.font(12, bold=True),
                      fill=pal["fg"])
            oy += 30
        h = oy - y
    return w, h


def render_story_screen(screen: StoryScreen, *,
                        avatars: dict | None = None,
                        media: dict | None = None):
    """Render one Instagram Story, full phone height:
    the medium, the segmented progress, the viewer
    header, the positional stickers and the Send message
    bar.  Returns ``(png_bytes, drawn)`` — the record
    carries the segment model and every sticker's
    position as fractions (census §27 exit proof)."""
    _require(screen.platform, SOCIAL_DESIGNS, "social")
    cv = _SocialCanvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})
    pal = cv.pal
    w = cv.w
    total_h = float(cv.dev["screen_h"])
    img = Image.new("RGB", (cv.s(w), cv.s(total_h)),
                    pal["bg"])
    cv.image = img
    draw = ImageDraw.Draw(img)
    # the medium
    if screen.kind == "text":
        from .render_screens import _hex_rgb

        color = _hex_rgb(screen.bg) if screen.bg \
            else (0x76, 0x56, 0xC9)
        draw.rectangle([0, 0, cv.s(w), cv.s(total_h)],
                       fill=color)
        media_source = "none"
    else:
        src = cv.media_handoff.get("media",
                                   screen.media or None)
        media_source = _media_tile(
            img, draw, cv, src, 0, 0, w, total_h,
            "story:" + screen.platform, "the story photo")
        _darken(img, cv, 0, 130)
        _darken(img, cv, total_h - 150, total_h)
        draw = ImageDraw.Draw(img)
    drawn = _base_drawn(screen, cv, total_h, kind="story")
    drawn["progress"] = {"segments": screen.segments,
                         "index": screen.index,
                         "played": screen.played}
    drawn["media_source"] = media_source
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
            fill=(0xFF, 0xFF, 0xFF))
        frac = 1.0 if i < screen.index else (
            screen.played if i == screen.index else 0.0)
        if frac > 0:
            draw.rounded_rectangle(
                [cv.s(x), cv.s(seg_y),
                 cv.s(x + seg_w * frac), cv.s(seg_y + 3)],
                radius=cv.s(1.5), fill=pal["accent"])
    drawn["regions"]["progress"] = [
        cv.s(8), cv.s(seg_y), cv.s(w - 16), cv.s(3)]
    # viewer header
    hy = seg_y + 14
    owner = screen.owner
    source = _draw_avatar_strict(
        img, draw, cv, 12, hy, 36, owner, owner.name)
    name = "@" + owner.handle if owner.handle else owner.name
    draw.text((cv.s(56), cv.s(hy + 2)), name,
              font=cv.font(14, bold=True),
              fill=(0xFF, 0xFF, 0xFF))
    if screen.when:
        draw.text((cv.s(56), cv.s(hy + 20)), screen.when,
                  font=cv.font(12),
                  fill=(0xE0, 0xE0, 0xE0))
    draw.text((cv.s(w - 28), cv.s(hy)), "×",
              font=cv.font(20, bold=True),
              fill=(0xFF, 0xFF, 0xFF))
    drawn["elements"].append({
        "id": "owner", "kind": "story_owner",
        "text": name, "avatar_source": source,
        "box": [cv.s(12), cv.s(hy), cv.s(36), cv.s(36)]})
    # the story text, centred
    if screen.text:
        lines = cv.wrap(screen.text, 22, w - 48, bold=True)
        th = len(lines) * 22 * 1.35
        ty = total_h / 2 - th / 2
        for i, line in enumerate(lines):
            lx = (w - cv.text_w(line, 22, bold=True)) / 2
            draw.text((cv.s(lx + 2), cv.s(ty + i * 22 * 1.35
                                           + 2)), line,
                      font=cv.font(22, bold=True),
                      fill=(0x00, 0x00, 0x00))
            draw.text((cv.s(lx), cv.s(ty + i * 22 * 1.35)),
                      line, font=cv.font(22, bold=True),
                      fill=(0xFF, 0xFF, 0xFF))
        drawn["elements"].append({
            "id": "text", "kind": "story_text",
            "text": screen.text,
            "box": [cv.s(24), cv.s(ty), cv.s(w - 48),
                    cv.s(th)]})
    # stickers at their fractions
    for sticker in screen.stickers:
        cx = sticker.x * w
        cy = sticker.y * total_h
        sw, sh = _sticker_chip(draw, cv, sticker, cx, cy)
        drawn["elements"].append({
            "id": "sticker:" + sticker.kind + ":"
                  + sticker.text,
            "kind": "story_sticker",
            "sticker_kind": sticker.kind,
            "text": sticker.text,
            "options": list(sticker.options),
            "x": sticker.x, "y": sticker.y,
            "box": [cv.s(cx - sw / 2), cv.s(cy - sh / 2),
                    cv.s(sw), cv.s(sh)]})
    # reply bar
    if screen.reply_bar:
        by = total_h - 64
        draw.rounded_rectangle(
            [cv.s(12), cv.s(by), cv.s(w - 76),
             cv.s(by + 40)], radius=cv.s(20),
            fill=pal["card_bg"])
        draw.text((cv.s(28), cv.s(by + 11)), "Send message",
                  font=cv.font(14), fill=pal["sub"])
        draw.text((cv.s(w - 62), cv.s(by + 8)), "♥",
                  font=cv.font(18), fill=(0xFF, 0xFF, 0xFF))
        draw.text((cv.s(w - 34), cv.s(by + 8)), "↗",
                  font=cv.font(18), fill=(0xFF, 0xFF, 0xFF))
        drawn["regions"]["reply_bar"] = [
            cv.s(12), cv.s(by), cv.s(w - 12), cv.s(40)]
    return _finish(img), drawn


# ── comments (census §5.5) ───────────────────────────────


def _comment_block(img, draw, cv: _SocialCanvas, comment,
                   x: float, y: float, max_w: float, *,
                   reply: bool = False) -> tuple[float, dict]:
    """One comment (or an indented reply): avatar, name
    + verified + time, text, the Reply / heart line, the
    Creator label and the Liked-by-Creator mark.
    Returns (height used, record element)."""
    pal = cv.pal
    size = 26.0 if reply else 36.0
    source = _draw_avatar_strict(
        img, draw, cv, x, y, size, comment.author,
        comment.author.name)
    nx = x + size + 10
    name = comment.author.name
    draw.text((cv.s(nx), cv.s(y)), name,
              font=cv.font(13, bold=True), fill=pal["fg"])
    nw = cv.text_w(name, 13, bold=True)
    if comment.author.verified:
        _verified(draw, cv, nx + nw + 4, y + 2, r=6.0)
    if comment.creator:
        cx = nx + nw + (20 if comment.author.verified else 4)
        draw.rounded_rectangle(
            [cv.s(cx), cv.s(y + 1), cv.s(cx + 52),
             cv.s(y + 17)], radius=cv.s(3),
            fill=pal["button_bg"])
        draw.text((cv.s(cx + 5), cv.s(y + 3)), "Creator",
                  font=cv.font(10, bold=True),
                  fill=pal["sub"])
    if comment.time:
        tw = cv.text_w(comment.time, 12)
        draw.text((cv.s(x + max_w - tw), cv.s(y)),
                  comment.time, font=cv.font(12),
                  fill=pal["sub"])
    ty = y + 20
    th = _text_block(draw, cv, comment.text, nx, ty,
                     max_w - size - 10, 13.5, pal["fg"])
    ay = ty + th + 4
    draw.text((cv.s(nx), cv.s(ay)), "Reply",
              font=cv.font(12, bold=True), fill=pal["sub"])
    likes = "♥ " + _count(cv, comment.likes)
    draw.text((cv.s(nx + 52), cv.s(ay)), likes,
              font=cv.font(12), fill=pal["sub"])
    used = ay + 18 - y
    if comment.liked_by_creator:
        draw.text((cv.s(nx), cv.s(ay + 18)),
                  "♥ Liked by Creator",
                  font=cv.font(11.5), fill=(0xFE, 0x2C, 0x55))
        used += 20
    return used, {
        "id": comment.id, "kind": "comment",
        "text": comment.text, "creator": bool(comment.creator),
        "liked_by_creator": bool(comment.liked_by_creator),
        "likes": comment.likes, "avatar_source": source,
        "box": [cv.s(x), cv.s(y), cv.s(max_w), cv.s(used)]}


def render_comment_screen(screen: CommentScreen, *,
                          avatars: dict | None = None,
                          media: dict | None = None):
    """Render the TikTok comment sheet: the count at the
    top, up to 30 comments over the dimmed video frame,
    replies indented (at most ten shown, the rest behind
    a "View N replies" line), creator labels and
    Liked-by-Creator marks as data.  Returns
    ``(png_bytes, drawn)``."""
    _require(screen.platform, SOCIAL_DESIGNS, "social")
    cv = _SocialCanvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})
    pal = cv.pal
    w = cv.w
    status_h = float(cv.dev["status_h"])
    bottom_h = float(cv.dev["nav_h"])
    shown = screen.comments[:30]
    # measure
    body_h = 64.0
    for comment in shown:
        lines = cv.wrap(comment.text, 13.5, w - 24 - 46)
        body_h += len(lines) * 13.5 * 1.35 + 44
        if comment.liked_by_creator:
            body_h += 20
        replies = comment.replies[:10]
        for rep in replies:
            rlines = cv.wrap(rep.text, 13.5, w - 24 - 46 - 44)
            body_h += len(rlines) * 13.5 * 1.35 + 44
            if rep.liked_by_creator:
                body_h += 20
        if len(comment.replies) > 10:
            body_h += 22
    total_h = int(status_h + body_h + bottom_h + 8)
    img = Image.new("RGB", (cv.s(w), cv.s(total_h)),
                    pal["bg"])
    cv.image = img
    draw = ImageDraw.Draw(img)
    # the dimmed video behind the sheet
    src = cv.media_handoff.get(
        "backdrop", screen.backdrop_media or None)
    if src:
        _media_tile(img, draw, cv, src, 0, 0, w, total_h,
                    "comments:backdrop", "the video frame")
        _darken(img, cv, 0, total_h, factor=0.35)
        draw = ImageDraw.Draw(img)
    drawn = _base_drawn(screen, cv, total_h, kind="comment")
    y = _draw_status_bar(draw, cv, 0.0)
    drawn["regions"]["status_bar"] = [0, 0, cv.s(w), cv.s(y)]
    sheet_y = y + 24
    draw.rounded_rectangle(
        [0, cv.s(sheet_y), cv.s(w), cv.s(total_h)],
        radius=cv.s(16), fill=pal["card_bg"])
    draw.rounded_rectangle(
        [cv.s(w / 2 - 20), cv.s(sheet_y + 8), cv.s(w / 2 + 20),
         cv.s(sheet_y + 12)], radius=cv.s(2),
        fill=pal["divider"])
    hy = sheet_y + 22
    title = "Comments · " + _count(cv, screen.total)
    draw.text((cv.s(w / 2 - cv.text_w(title, 15, bold=True)
                    / 2), cv.s(hy)), title,
              font=cv.font(15, bold=True), fill=pal["fg"])
    drawn["elements"].append({
        "id": "header", "kind": "comment_header",
        "total": screen.total, "shown": len(shown),
        "box": [0, cv.s(sheet_y), cv.s(w), cv.s(48)]})
    y = hy + 30
    for comment in shown:
        used, el = _comment_block(
            img, draw, cv, comment, 12, y, w - 24)
        drawn["elements"].append(el)
        y += used + 10
        replies = comment.replies[:10]
        for rep in replies:
            rused, rel = _comment_block(
                img, draw, cv, rep, 56, y, w - 24 - 44,
                reply=True)
            rel["reply_to"] = comment.id
            drawn["elements"].append(rel)
            y += rused + 8
        if len(comment.replies) > 10:
            more = len(comment.replies) - 10
            label = "View " + str(more) + " more replies"
            draw.text((cv.s(56), cv.s(y)), label,
                      font=cv.font(12, bold=True),
                      fill=pal["sub"])
            drawn["elements"].append({
                "id": comment.id + ":more",
                "kind": "comment_more",
                "view_more": more,
                "box": [cv.s(56), cv.s(y), cv.s(w - 80),
                        cv.s(18)]})
            y += 22
    _draw_nav(draw, cv, total_h - bottom_h)
    drawn["regions"]["nav"] = [
        0, cv.s(total_h - bottom_h), cv.s(w), cv.s(bottom_h)]
    return _finish(img), drawn


# ── video pages (census §5.9) ────────────────────────────


def render_video_screen(screen: VideoScreen, *,
                        avatars: dict | None = None,
                        media: dict | None = None):
    """Render the YouTube video page: the player with the
    user thumbnail, the title and counts, the hashtags,
    the description (shortened with "…more" except on
    the description layout), the channel row with its
    Subscribe state, and — by layout — the up-next
    videos, the top-comment card, or the description
    panel.  Returns ``(png_bytes, drawn)``."""
    _require(screen.platform, SOCIAL_DESIGNS, "social")
    cv = _SocialCanvas(screen)
    cv.avatars = dict(avatars or {})
    cv.media_handoff = dict(media or {})
    pal = cv.pal
    w = cv.w
    status_h = float(cv.dev["status_h"])
    bottom_h = float(cv.dev["nav_h"])
    style = screen.count_style or cv.design.get(
        "count_style", "intl")
    player_h = w * 9 / 16
    title_lines = cv.wrap(screen.title, 16, w - 24,
                          bold=True)
    title_h = len(title_lines) * 16 * 1.35
    tags_h = 20.0 if screen.hashtags else 0.0
    if screen.layout == "description":
        desc_lines = cv.wrap(screen.description, 14, w - 24)
        desc_h = len(desc_lines) * 14 * 1.35 + 10
    else:
        desc_lines = cv.wrap(screen.description, 14, w - 24)
        desc_h = min(len(desc_lines), 2) * 14 * 1.35 + 10 \
            if desc_lines else 0.0
    channel_h = 64.0
    if screen.layout == "watch":
        tail_h = 30.0 + len(screen.up_next) * 96.0
    elif screen.layout == "comments":
        tail_h = 30.0
        if screen.top_comment is not None:
            clines = cv.wrap(screen.top_comment.text, 13.5,
                             w - 24 - 46)
            tail_h += len(clines) * 13.5 * 1.35 + 48
    else:
        tail_h = 0.0
    total_h = int(status_h + player_h + 12 + title_h + 24
                  + tags_h + desc_h + channel_h + tail_h
                  + bottom_h + 8)

    img, draw, y = _start_canvas(cv, total_h)
    drawn = _base_drawn(screen, cv, total_h, kind="video")
    drawn["regions"]["status_bar"] = [0, 0, cv.s(w), cv.s(y)]
    # player
    py = y
    src = cv.media_handoff.get(
        "thumbnail", screen.thumbnail or None)
    worn = _media_tile(
        img, draw, cv, src, 0, py, w, player_h,
        "video:player", "the video thumbnail")
    _play(draw, cv, w / 2, py + player_h / 2)
    drawn["regions"]["player"] = [
        0, cv.s(py), cv.s(w), cv.s(player_h)]
    drawn["elements"].append({
        "id": "player", "kind": "video_player",
        "media_source": worn,
        "box": [0, cv.s(py), cv.s(w), cv.s(player_h)]})
    y = py + player_h + 12
    # title + meta
    ty = y
    y += _text_block(draw, cv, screen.title, 12, y, w - 24,
                     16, pal["fg"], bold=True)
    drawn["elements"].append({
        "id": "title", "kind": "video_title",
        "text": screen.title,
        "box": [cv.s(12), cv.s(ty), cv.s(w - 24),
                cv.s(y - ty)]})
    meta = _count(cv, screen.views, style) + " views · " \
        + _count(cv, screen.likes, style) + " likes"
    if screen.date:
        meta += " · " + screen.date
    draw.text((cv.s(12), cv.s(y + 2)), meta,
              font=cv.font(12.5), fill=pal["sub"])
    y += 24
    # hashtags
    if screen.hashtags:
        tags = " ".join(screen.hashtags)
        draw.text((cv.s(12), cv.s(y)), tags,
                  font=cv.font(13, bold=True),
                  fill=pal["accent"])
        y += tags_h
    # description
    if screen.description:
        dy = y
        if screen.layout == "description":
            y += _text_block(draw, cv, screen.description,
                             12, y, w - 24, 14, pal["fg"])
        else:
            shown = desc_lines[:2]
            for i, line in enumerate(shown):
                text = line
                if i == 1 and len(desc_lines) > 2:
                    text = line + " …more"
                draw.text((cv.s(12),
                           cv.s(y + i * 14 * 1.35)), text,
                          font=cv.font(14), fill=pal["fg"])
            y += desc_h
        drawn["elements"].append({
            "id": "description", "kind": "video_description",
            "text": screen.description,
            "shortened": screen.layout != "description",
            "box": [cv.s(12), cv.s(dy), cv.s(w - 24),
                    cv.s(y - dy)]})
    # channel row
    cy = y + 8
    channel = screen.channel
    source = _draw_avatar_strict(
        img, draw, cv, 12, cy, 40, channel, channel.name)
    draw.text((cv.s(60), cv.s(cy + 2)), channel.name,
              font=cv.font(14, bold=True), fill=pal["fg"])
    if channel.verified:
        _verified(draw, cv,
                  60 + cv.text_w(channel.name, 14, bold=True)
                  + 4, cy + 5)
    sub = ""
    if channel.handle:
        sub = "@" + channel.handle
    sub = (sub + " · " if sub else "") \
        + _count(cv, screen.subscribers, style) \
        + " subscribers"
    draw.text((cv.s(60), cv.s(cy + 22)), sub,
              font=cv.font(12), fill=pal["sub"])
    label = "Subscribed" if screen.subscribed else "Subscribe"
    bw = cv.text_w(label, 13, bold=True) + 28
    _button(draw, cv, w - 12 - bw, cy + 4, bw, 32, label,
            primary=not screen.subscribed)
    drawn["elements"].append({
        "id": "channel", "kind": "video_channel",
        "text": channel.name,
        "subscribed": bool(screen.subscribed),
        "avatar_source": source,
        "box": [cv.s(12), cv.s(cy), cv.s(w - 24), cv.s(40)]})
    y = cy + channel_h
    # layout tail
    if screen.layout == "watch" and screen.up_next:
        draw.text((cv.s(12), cv.s(y + 4)), "Up next",
                  font=cv.font(15, bold=True), fill=pal["fg"])
        y += 30
        for item in screen.up_next:
            iy = y
            src = cv.media_handoff.get(
                item.id, item.media or None)
            worn = _media_tile(
                img, draw, cv, src, 12, iy, 150, 84,
                "thumb:" + item.id,
                "the thumbnail for " + repr(item.id))
            if item.duration:
                tw = cv.text_w(item.duration, 10, bold=True)
                draw.rectangle(
                    [cv.s(12 + 150 - tw - 10),
                     cv.s(iy + 84 - 18),
                     cv.s(12 + 150 - 4), cv.s(iy + 84 - 4)],
                    fill=(0x14, 0x14, 0x14))
                draw.text((cv.s(12 + 150 - tw - 7),
                           cv.s(iy + 84 - 17)), item.duration,
                          font=cv.font(10, bold=True),
                          fill=(0xFF, 0xFF, 0xFF))
            draw.text((cv.s(172), cv.s(iy + 2)), item.title,
                      font=cv.font(13, bold=True),
                      fill=pal["fg"])
            meta = item.channel
            if item.views >= 0:
                meta += (" · " if meta else "") \
                    + _count(cv, item.views, style) + " views"
            draw.text((cv.s(172), cv.s(iy + 42)), meta,
                      font=cv.font(12), fill=pal["sub"])
            drawn["elements"].append({
                "id": item.id, "kind": "video_up_next",
                "media_source": worn,
                "box": [cv.s(12), cv.s(iy), cv.s(w - 24),
                        cv.s(84)]})
            y += 96
    elif screen.layout == "comments" \
            and screen.top_comment is not None:
        draw.text((cv.s(12), cv.s(y + 4)), "Comments",
                  font=cv.font(15, bold=True), fill=pal["fg"])
        y += 30
        used, el = _comment_block(
            img, draw, cv, screen.top_comment, 12, y, w - 24)
        drawn["elements"].append(el)
        y += used
    _draw_nav(draw, cv, total_h - bottom_h)
    drawn["regions"]["nav"] = [
        0, cv.s(total_h - bottom_h), cv.s(w), cv.s(bottom_h)]
    return _finish(img), drawn
