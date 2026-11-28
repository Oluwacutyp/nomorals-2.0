"""Gapfix 150 — fake-message iPhone screens: every button that
should show an icon shows a REAL icon, not a tofu box.

The renderers used to draw UI icons as font glyphs (``draw.text
("☎")`` and friends) the stocked screen fonts do not carry, so
Pillow substituted the ``.notdef`` hollow box — the WhatsApp
header call button, the bottom-left plus button, the smiley,
the list/preview/call/history icons and the Facebook/TikTok
engagement icons all rendered as empty boxes (or, for the
plus, were never drawn at all).  The fix draws every one of
those icons as vectors in ``nomorals/social/chat/mock_screens/
icons.py``.

These tests prove the icons by their INK, not by grepping the
code: each assertion counts pixels in a small box that frames
one solid feature of the new vector icon (a handset earpiece,
a smiley eye, a blue plus, a heart lobe) and sits strictly
inside the hollow interior of the tofu box the old glyph left
behind — so the same box is empty on the pristine base.  Every
full-screen test below fails on pristine and passes on the
fixed tree.

Five sites could NOT be separated by ink on this box, because
the host fonts here (DejaVu, Inter) really do carry those
glyphs — the vectors replace them anyway, so the cards render
identically on hosts without those fonts, which is the actual
fix.  The five are: the list photo preview (a real 📷 emoji),
the call-screen end button (a real ☎), the profile grid play
(the ▶ glyph coincides with the vector triangle), and the
makers TikTok rail + X close (DejaVu ♥/↗/✕ at the same place
as the vectors).  Those shapes are proven instead by the tile
tests below (``_draw_camera`` is proven by the input-bar
camera and quote tests, which do discriminate), and the
wiring is proven by the render tests that do separate.
"""

import io

from PIL import Image, ImageDraw

from nomorals.social.chat.mock_screens.authoring import (
    demo_call_history,
    demo_comment_screen,
    demo_list_screen,
    demo_post_screen,
    demo_screen,
    demo_status_screen,
    demo_story_screen,
)
from nomorals.social.chat.mock_screens.render import (
    _quote_block,
    render_screen,
)
from nomorals.social.chat.mock_screens.render_screens import (
    _CallCanvas,
    _ListCanvas,
    render_call_history,
    render_list_screen,
    render_status_screen,
)
from nomorals.social.chat.mock_screens.render_social import (
    _count,
    _SocialCanvas,
    render_comment_screen,
    render_post_screen,
    render_story_screen,
)
from nomorals.social.chat.mock_screens.schema import (
    DeviceChrome,
    ListRow,
    ListScreen,
    Message,
    Participant,
    Screen,
)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# ── pixel helpers ──────────────────────────────────────────

def _open(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


def _ink(img: Image.Image, box, bg) -> int:
    """Pixels in ``box`` that differ from ``bg``."""
    x0, y0, x1, y1 = box
    px = img.load()
    return sum(1 for y in range(y0, y1) for x in range(x0, x1)
               if px[x, y] != bg)


def _near(img: Image.Image, box, target, tol: int = 14) -> int:
    """Pixels in ``box`` within ``tol`` of ``target`` on every
    channel — counts one colour's ink regardless of the
    background behind it."""
    x0, y0, x1, y1 = box
    px = img.load()
    tr, tg, tb = target
    return sum(
        1 for y in range(y0, y1) for x in range(x0, x1)
        if abs(px[x, y][0] - tr) <= tol
        and abs(px[x, y][1] - tg) <= tol
        and abs(px[x, y][2] - tb) <= tol)


def _blue(img: Image.Image, box) -> int:
    """Blue-dominant pixels (the WhatsApp plus is drawn in
    ``ticks_read`` blue, (0x34, 0xB7, 0xF1))."""
    x0, y0, x1, y1 = box
    px = img.load()
    return sum(
        1 for y in range(y0, y1) for x in range(x0, x1)
        if px[x, y][2] > 150 and px[x, y][2] - px[x, y][0] > 60
        and px[x, y][2] - px[x, y][1] > 40)


def _tile():
    """A blank white tile + a device-scale ``s`` for drawing
    one icon in isolation."""
    img = Image.new("RGB", (120, 120), WHITE)
    return img, ImageDraw.Draw(img), lambda v: int(round(v * 2))


def _icons():
    """The new vector-icon module, imported lazily so this file
    still collects on the pristine base (where the module does
    not exist yet and the tile tests error instead of passing)."""
    from nomorals.social.chat.mock_screens import icons
    return icons


def _el(drawn, kind, ident):
    for el in drawn["elements"]:
        if el.get("kind") == kind and el.get("id") == ident:
            return el
    raise AssertionError(f"no {kind} {ident} in drawn record")


# ── the icon vocabulary, drawn in isolation ────────────────

def test_icon_phone_has_two_earpieces_and_a_diagonal_bar():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_phone(draw, s, 30, 30, BLACK, r=10)
    # earpiece (top-left knob) and mouthpiece (bottom-right)
    assert _ink(img, (44, 44, 52, 52), WHITE) >= 8
    assert _ink(img, (68, 68, 76, 76), WHITE) >= 8
    # the curved bar's midpoint, bottom-left of centre
    assert _ink(img, (47, 64, 56, 73), WHITE) >= 8
    # the centre itself stays empty — a handset is not a blob
    assert _ink(img, (56, 56, 64, 64), WHITE) == 0


def test_icon_smiley_has_eyes_and_a_smile():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_smiley(draw, s, 30, 30, BLACK, r=10)
    assert _ink(img, (50, 52, 56, 58), WHITE) >= 3  # left eye
    assert _ink(img, (64, 52, 70, 58), WHITE) >= 3  # right eye
    assert _ink(img, (56, 72, 64, 78), WHITE) >= 3  # smile arc
    assert _ink(img, (56, 38, 64, 42), WHITE) >= 3  # ring top


def test_icon_plus_is_a_cross():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_plus(draw, s, 30, 30, BLACK, r=9)
    assert _ink(img, (58, 44, 62, 76), WHITE) >= 25
    assert _ink(img, (44, 58, 76, 62), WHITE) >= 25


def test_icon_heart_has_lobes_and_a_tip():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_heart(draw, s, 30, 30, BLACK, r=10)
    assert _ink(img, (52, 56, 58, 62), WHITE) >= 3  # left lobe
    assert _ink(img, (62, 56, 68, 62), WHITE) >= 3  # right lobe
    assert _ink(img, (56, 74, 64, 80), WHITE) >= 3  # bottom tip


def test_icon_check_is_a_tick():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_check(draw, s, 30, 30, BLACK, r=6)
    assert _ink(img, (52, 58, 58, 64), WHITE) >= 3  # down stroke
    assert _ink(img, (62, 52, 70, 58), WHITE) >= 3  # rising stroke


def test_icon_share_arrow_points_up_right():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_share_arrow(draw, s, 30, 30, BLACK, r=7)
    assert _ink(img, (64, 48, 72, 56), WHITE) >= 3  # arrowhead
    assert _ink(img, (50, 64, 56, 70), WHITE) >= 3  # tail


def test_icon_pin_has_body_and_tip():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_pin(draw, s, 30, 30, BLACK, r=6)
    assert _ink(img, (54, 56, 66, 62), WHITE) >= 5  # body
    assert _ink(img, (57, 68, 63, 74), WHITE) >= 3  # tip


def test_icon_bell_slash_has_a_slash():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_bell_slash(draw, s, 30, 30, BLACK, r=7)
    assert _ink(img, (50, 50, 70, 66), WHITE) >= 10  # bell body
    assert _ink(img, (64, 64, 70, 70), WHITE) >= 3  # slash


def test_icon_star_has_points():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_star(draw, s, 30, 30, BLACK, r=7)
    assert _ink(img, (55, 55, 65, 65), WHITE) >= 10  # centre
    assert _ink(img, (58, 42, 62, 48), WHITE) >= 2  # top point


def test_icon_video_camera_has_body_and_lens():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_video_camera(draw, s, 30, 30, BLACK, w=16)
    # the filled body
    assert _ink(img, (44, 51, 65, 69), WHITE) >= 100
    # the lens triangle jutting right
    assert _ink(img, (67, 50, 76, 70), WHITE) >= 40


def test_icon_play_is_a_right_pointing_triangle():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_play(draw, s, 30, 30, BLACK, r=8)
    # the tip, right of centre, and the flat left edge
    assert _ink(img, (66, 57, 72, 63), WHITE) >= 5
    assert _ink(img, (50, 54, 56, 66), WHITE) >= 5
    # the top-left corner stays empty — a triangle, not a box
    assert _ink(img, (80, 40, 90, 50), WHITE) == 0


def test_icon_close_is_two_crossing_strokes():
    img, draw, s = _tile()
    icons = _icons()
    icons._draw_close(draw, s, 30, 30, BLACK, r=8)
    # the crossing point and the top-left arm's end
    assert _ink(img, (57, 57, 63, 63), WHITE) >= 5
    assert _ink(img, (46, 46, 52, 52), WHITE) >= 2
    # a corner stays empty — two strokes, not a filled square
    assert _ink(img, (80, 40, 90, 50), WHITE) == 0


# ── WhatsApp iPhone chat: the owner's two reports ──────────

def test_whatsapp_iphone_header_call_button_is_a_handset():
    data, _drawn = render_screen(demo_screen("whatsapp",
                                             style="iphone"))
    img = _open(data)
    # the mouthpiece disc, bottom-right of the call icon:
    # the old tofu box is a narrow strip at the icon's left,
    # so this box is empty on pristine
    assert _ink(img, (746, 152, 756, 162), (0x07, 0x5E, 0x54)) >= 20


def test_whatsapp_iphone_input_bar_plus_is_blue():
    data, _drawn = render_screen(demo_screen("whatsapp",
                                             style="iphone"))
    img = _open(data)
    # the plus at the bottom left, in ticks_read blue; the
    # pristine bar has no blue pixel anywhere in this box
    assert _blue(img, (24, 1736, 64, 1776)) >= 50


def test_whatsapp_iphone_input_bar_camera_at_field_end():
    data, _drawn = render_screen(demo_screen("whatsapp",
                                             style="iphone"))
    img = _open(data)
    # the camera at the field's right end: the pristine
    # field is blank white here
    assert _ink(img, (597, 1749, 635, 1763), WHITE) >= 50


def test_telegram_iphone_input_bar_smiley_has_eyes():
    data, _drawn = render_screen(demo_screen("telegram",
                                             style="iphone"))
    img = _open(data)
    # the smiley's eyes: inside the hollow interior of the
    # old tofu smiley, so empty on pristine
    assert _ink(img, (40, 1324, 48, 1332), WHITE) >= 5
    assert _ink(img, (52, 1324, 60, 1332), WHITE) >= 5


def test_instagram_iphone_input_bar_smiley_has_eyes():
    # the Instagram bar used to crash here: its smiley
    # referenced ``mid`` before the assignment below the
    # branch (ruff F821 on the first pass) — the bar now
    # renders and the smiley has real eyes
    from nomorals.social.chat.mock_screens.render import _Canvas
    screen = demo_screen("instagram", style="iphone")
    data, _drawn = render_screen(screen)
    img = _open(data)
    fg = _Canvas(screen).pal["header_fg"]
    assert _near(img, (700, 1291, 708, 1299), fg) >= 5
    assert _near(img, (714, 1292, 720, 1298), fg) >= 4


def test_whatsapp_quote_photo_draws_camera_before_photo():
    screen = Screen(
        platform="whatsapp", title="Adaeze",
        chrome=DeviceChrome(style="iphone", clock="9:41",
                            battery=84, network="4G"),
        participants=[Participant("me", "You"),
                      Participant("adaeze", "Adaeze")],
        messages=[
            Message("m1", "adaeze", kind="image", time="9:41"),
            Message("m2", "me", text="Got it, thanks!",
                    time="9:42", reply_to="m1"),
        ])
    # the quote block names the vector icon for a text-less
    # image; the old fallback prefixed a "📷" glyph
    from nomorals.social.chat.mock_screens.render import _Canvas
    cv = _Canvas(screen)
    _name, _color, qtext, qicon = _quote_block(cv, screen.messages[1])
    assert (qtext, qicon) == ("Photo", "camera")
    meta = cv.pal["meta"]
    data, drawn = render_screen(screen)
    img = _open(data)
    el = _el(drawn, "text", "m2")
    x0, y0, _w, _h = el["box"]
    # the camera body sits at the quote's left; the word
    # "Photo" starts 18 logical px to its right on the fixed
    # tree, so the gap between them carries no meta ink —
    # on pristine the word started where the camera is now
    assert _near(img, (x0 + 28, y0 + 44, x0 + 56, y0 + 64),
                 meta) >= 40
    assert _near(img, (x0 + 58, y0 + 44, x0 + 72, y0 + 64),
                 meta) == 0


# ── WhatsApp list: previews, pin, mute, verified ───────────
#
# No photo-preview test here: the old "📷" renders as a real
# emoji on this box, so no ink box separates it from the new
# vector camera.  The camera shape itself is proven by the
# input-bar camera test and the quote test above, which do
# discriminate, and the preview calls the same _draw_camera.

def test_whatsapp_list_call_preview_draws_arrow():
    data, _drawn = render_list_screen(
        demo_list_screen("whatsapp-list", "iphone"))
    img = _open(data)
    # the missed-call arrow's shaft midpoint: inside the
    # hollow interior of the old "↙" tofu box
    assert _ink(img, (167, 1119, 173, 1125), WHITE) >= 3


def test_whatsapp_list_pinned_row_draws_pin():
    data, _drawn = render_list_screen(
        demo_list_screen("whatsapp-list", "iphone"))
    img = _open(data)
    # the pin's tip: inside the hollow interior of the old
    # "📌" tofu box
    assert _ink(img, (731, 522, 737, 528), WHITE) >= 3


def test_whatsapp_list_muted_row_draws_bell_slash():
    data, _drawn = render_list_screen(
        demo_list_screen("whatsapp-list", "iphone"))
    img = _open(data)
    # the bell-slash's diagonal: inside the hollow interior
    # of the old "🔕" tofu box
    assert _ink(img, (738, 1134, 744, 1140), WHITE) >= 3


def test_whatsapp_list_verified_row_draws_check():
    screen = ListScreen(
        platform="whatsapp", title="Chats", flavor="chats",
        chrome=DeviceChrome(style="iphone", clock="9:41",
                            battery=84, network="4G"),
        participants=[Participant("adaeze", "Adaeze")],
        rows=[ListRow(id="v1", name="Adaeze",
                      participant_id="adaeze",
                      preview="see you at dawn", time="9:41",
                      verified=True)])
    data, drawn = render_list_screen(screen)
    img = _open(data)
    el = _el(drawn, "row", "v1")
    _x0, y0, _w, _h = el["box"]
    # the verified disc sits right of the name; the white
    # check's rising stroke falls inside the hollow interior
    # of the old "✓" tofu box
    cv = _ListCanvas(screen)
    nx = 78 + cv.text_w("Adaeze", 15.5)
    cx = int(2 * nx + 24)
    cy = int(y0 + 40)
    # white check ink: the disc behind it is accent-coloured,
    # so only the check itself counts as white
    assert _near(img, (cx, cy, cx + 6, cy + 6), WHITE) >= 3


# ── calls: call log, call screen, iPhone history ───────────

def test_whatsapp_calls_voice_row_draws_handset():
    data, _drawn = render_list_screen(
        demo_list_screen("whatsapp-calls", "iphone"))
    img = _open(data)
    # the voice row's handset earpiece disc: the host font
    # renders "☎" as a real handset here, so no box is empty
    # on pristine — but the vector disc fills this box 30/30
    # while the glyph's earpiece only touches 9 of it
    assert _ink(img, (708, 354, 714, 359), WHITE) >= 20


# No call-screen end-button test here: the "☎" glyph renders
# as a real handset on this box with the same bounding box as
# the vector phone, so no ink box separates them.  The phone
# shape is proven by the tile test and the header test above.


def test_iphone_call_history_outgoing_draws_share_arrow():
    data, drawn = render_call_history(demo_call_history("iphone"))
    img = _open(data)
    # the outgoing row's share arrow sits LEFT of the sub
    # label, at (72, y+37) of its row — the old box measured
    # the info button at the row's right end instead.  The
    # arrowhead tip: empty on pristine (the "↗" is tofu in
    # the call font), solid on the fixed tree
    assert _ink(img, (147, 531, 155, 539), WHITE) >= 10


def test_iphone_call_history_voicemail_tab_draws_voicemail():
    screen = demo_call_history("iphone")
    data, drawn = render_call_history(screen)
    img = _open(data)
    # the voicemail tab is the fifth of five; its tape shell's
    # top edge, in tab grey — the "➿" glyph has a hollow
    # interior with a diamond in it, so only 8 of these
    # pixels are grey on pristine against 24 on the fixed tree
    tab_fg = _CallCanvas(screen).pal["tab_fg"]
    assert _near(img, (690, 1088, 698, 1096), tab_fg) >= 16


# ── Facebook / TikTok posts, profile, comments ─────────────

def _engagement_centres(screen, ey):
    """(heart, speech, share) icon centres of a post's
    engagement row, computed the way the renderer does."""
    cv = _SocialCanvas(screen)
    ex = 12.0
    centres = {}
    for icon, count in (("heart", screen.likes),
                        ("speech", screen.comments),
                        ("share", screen.shares)):
        centres[icon] = (ex + 7, ey + 7)
        ex += 16 + cv.text_w(_count(cv, count), 14) + 26
    return centres


def _assert_post_engagement(name, style):
    screen = demo_post_screen(name, style)
    data, drawn = render_post_screen(screen)
    img = _open(data)
    el = _el(drawn, "post_engagement", "engagement")
    ey = el["box"][1] / 2
    centres = _engagement_centres(screen, ey)
    sub = _SocialCanvas(screen).pal["sub"]
    hx, hy = centres["heart"]
    # heart lobes: inside the old "♥" tofu interior
    assert _near(img, (int(2 * hx - 8), int(2 * hy - 6),
                       int(2 * hx - 2), int(2 * hy)), sub) >= 3
    assert _near(img, (int(2 * hx + 2), int(2 * hy - 6),
                       int(2 * hx + 8), int(2 * hy)), sub) >= 3
    sx, sy = centres["speech"]
    # speech bubble's top edge: inside the old "💬" interior
    assert _near(img, (int(2 * sx - 5), int(2 * sy - 7),
                       int(2 * sx + 5), int(2 * sy - 5)), sub) >= 3
    ax, ay = centres["share"]
    # share arrowhead: inside the old "↗" tofu interior
    assert _near(img, (int(2 * ax + 4), int(2 * ay - 10),
                       int(2 * ax + 12), int(2 * ay - 2)), sub) >= 3


def test_facebook_post_engagement_draws_icons():
    _assert_post_engagement("facebook-post", "android")


def test_tiktok_post_engagement_draws_icons():
    _assert_post_engagement("tiktok-post", "android")


def test_tiktok_comment_like_draws_heart():
    screen = demo_comment_screen("android")
    data, drawn = render_comment_screen(screen)
    img = _open(data)
    el = _el(drawn, "comment", "c2")
    _x0, y0, _w, _h = el["box"]
    cv = _SocialCanvas(screen)
    comment = screen.comments[1]
    lines = cv.wrap(comment.text, 13.5, 400 - 24 - 46)
    ay = y0 / 2 + 20 + len(lines) * 13.5 * 1.35 + 4
    cy = int(2 * (ay + 6))
    sub = _SocialCanvas(screen).pal["sub"]
    # the like heart's lobes: inside the old "♥" interior
    assert _near(img, (224, cy - 6, 230, cy), sub) >= 3
    assert _near(img, (233, cy - 6, 239, cy), sub) >= 3


def test_tiktok_comment_liked_by_creator_draws_red_heart():
    screen = demo_comment_screen("android")
    data, drawn = render_comment_screen(screen)
    img = _open(data)
    el = _el(drawn, "comment", "c2")
    _x0, y0, _w, _h = el["box"]
    cv = _SocialCanvas(screen)
    comment = screen.comments[1]
    lines = cv.wrap(comment.text, 13.5, 400 - 24 - 46)
    ay = y0 / 2 + 20 + len(lines) * 13.5 * 1.35 + 4
    cy = int(2 * (ay + 24))
    # the Liked-by-Creator heart, in creator red, lobes
    # inside the old "♥" tofu interior
    assert _near(img, (120, cy - 6, 126, cy),
                 (0xFE, 0x2C, 0x55)) >= 3
    assert _near(img, (129, cy - 6, 135, cy),
                 (0xFE, 0x2C, 0x55)) >= 3


# ── stories and status ─────────────────────────────────────

def test_instagram_story_reply_bar_draws_heart_and_share():
    data, _drawn = render_story_screen(
        demo_story_screen("instagram-story", "android"))
    img = _open(data)
    # heart lobes and share arrowhead, each inside its old
    # tofu box's hollow interior; both icons are white on
    # the photo, so only white counts
    assert _near(img, (684, 1501, 690, 1507), WHITE) >= 3
    assert _near(img, (698, 1501, 704, 1507), WHITE) >= 3
    assert _near(img, (755, 1493, 765, 1503), WHITE) >= 3


def test_whatsapp_status_reply_bar_draws_heart():
    data, _drawn = render_status_screen(
        demo_status_screen("whatsapp-status", "iphone"))
    img = _open(data)
    # the liked heart's lobes, in WhatsApp green, inside
    # the old "♥" tofu interior
    assert _near(img, (685, 1564, 691, 1570),
                 (0x1E, 0xAA, 0x62)) >= 3
    assert _near(img, (701, 1564, 707, 1570),
                 (0x1E, 0xAA, 0x62)) >= 3


# No makers render tests here: the TikTok rail's ♥/↗ and the X
# card's ✕ render as real DejaVu glyphs on this box at the same
# place and size as the vectors, so no ink box separates them.
# The heart, share arrow and close shapes are proven by the
# tile tests above; the rail and the X header call the same
# helpers the render tests prove elsewhere.
