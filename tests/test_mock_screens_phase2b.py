"""Mock screens Phase 2b (FakeDetail port, census §26):
lists, status and calls — the WhatsApp / Instagram /
Telegram / Snapchat chat lists, the WhatsApp call list
and status (Updates) list, the WhatsApp Status screen
and Views sheet, the generic + named call screens, and
the iPhone Call History.

Network-free: renders are local Pillow work, media
fixtures are tiny generated files, and the delivery seam
is a recording stub — the phase-1 suite's idiom.
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_mock_screens import (
    RuntimeMockScreensMixin,
)
from nomorals.social.chat.mock_screens import (
    CALL_DESIGNS,
    LIST_DESIGNS,
    STATUS_DESIGNS,
    CallHistoryEntry,
    CallHistoryScreen,
    CallScreen,
    DeviceChrome,
    ListRow,
    ListScreen,
    MockScreensError,
    MockScreenService,
    Participant,
    StatusScreen,
    StatusViewsScreen,
)
from nomorals.social.chat.mock_screens.authoring import (
    demo_call_history,
    demo_call_screen,
    demo_list_screen,
    demo_screen,
    demo_status_screen,
    demo_status_views,
    parse_mock_tail,
)
from nomorals.social.chat.mock_screens.render import (
    render_screen,
)
from nomorals.social.chat.mock_screens.render_screens import (
    render_call_history,
    render_call_screen,
    render_list_screen,
    render_status_screen,
    render_status_views,
)

# ── fixtures ─────────────────────────────────────────────


def _png_bytes(color=(30, 120, 200), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _solid_png(path, color=(200, 30, 30), size=(300, 200)):
    Image.new("RGB", size, color).save(path)
    return path


class _Send:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, ref, caption=""):
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=self.ok, error="")


def _people():
    return [
        Participant("me", "You"),
        Participant("adaeze", "Adaeze Nwosu"),
        Participant("bola", "Bola"),
    ]


def _el(drawn, **want):
    for entry in drawn["elements"]:
        if all(entry.get(k) == v for k, v in want.items()):
            return entry
    raise AssertionError("no element " + repr(want)
                         + " in " + repr(drawn["elements"]))


# ── census accounting (census §26 build list) ────────────

#: every list/status/call tool the census's Phase 2
#: names, mapped to the registry that renders it
CENSUS_RENDERED = {
    "WhatsApp Status": "status:whatsapp",
    "WhatsApp Status Views": "views:whatsapp",
    "WhatsApp Status List": "list:whatsapp",
    "WhatsApp Chat List": "list:whatsapp",
    "WhatsApp Call List": "list:whatsapp",
    "Instagram Chat List": "list:instagram",
    "Snapchat Chat List": "list:snapchat",
    "Telegram Chat List": "list:telegram",
    "Generic Call Screen": "call:generic",
    "WhatsApp Call": "call:whatsapp",
    "Facebook Call": "call:facebook",
    "Instagram Call": "call:instagram",
    "Hangouts Call": "call:hangouts",
    "iPhone Call History": "call:iphone_history",
}

#: census-named scope held back, with its census reason —
#: never filled by guessing
CENSUS_REMAINDERS = {
    "Snapchat list row states / preview vocabulary":
        "census §4.8: verified elements are the friends/"
        "chats list + iPhone/Android styles only; its row "
        "states require a build-time inspection — the "
        "snapchat list design is the shared list anatomy, "
        "its states unguessed",
    "Facebook / Instagram / Hangouts call chrome":
        "census §6.1 names the tools; §6.2 verifies the "
        "generic call anatomy — the named designs ride "
        "that anatomy with their platform accent rather "
        "than guessed pixels",
    "The other six of the homepage's 13 list tools":
        "census §4.1: the category claims thirteen, the "
        "extraction exposes seven — the remainder must "
        "not be invented",
    "Status / call video export":
        "census §4.2/§4.7: animation is Phase 4 of the "
        "same organ, not this build",
}


def test_census_phase2_catalogue_is_accounted():
    registries = {"list": LIST_DESIGNS,
                  "status": STATUS_DESIGNS,
                  "views": STATUS_DESIGNS,
                  "call": CALL_DESIGNS}
    for target in CENSUS_RENDERED.values():
        kind, key = target.split(":")
        assert key in registries[kind], target
    assert set(LIST_DESIGNS) == {
        "whatsapp", "instagram", "telegram", "snapchat"}
    assert set(STATUS_DESIGNS) == {"whatsapp"}
    assert set(CALL_DESIGNS) == {
        "generic", "whatsapp", "facebook", "instagram",
        "hangouts", "iphone_history"}
    for reason in CENSUS_REMAINDERS.values():
        assert reason.startswith("census §")


def test_phase2b_palettes_are_complete_per_design():
    # a design can never half-exist: light and dark carry
    # the same keys, and list palettes share one key set
    for registry in (LIST_DESIGNS, STATUS_DESIGNS,
                     CALL_DESIGNS):
        for name, design in registry.items():
            assert set(design["light"]) \
                == set(design["dark"]), name
    keysets = {frozenset(d["light"])
               for d in LIST_DESIGNS.values()}
    assert len(keysets) == 1


def test_chat_canvas_seam_stays_a_no_op_for_chats():
    # render.py's only shared-code change: the canvas
    # reads its design from a class registry whose
    # default IS CHAT_DESIGNS, and a design-level font
    # context only when the design names one — no chat
    # design does, so chat renders cannot move
    from nomorals.social.chat.mock_screens.designs import (
        CHAT_DESIGNS,
    )
    from nomorals.social.chat.mock_screens.render import (
        _Canvas,
    )

    assert _Canvas.DESIGNS is CHAT_DESIGNS
    assert all("font_context" not in d
               for d in CHAT_DESIGNS.values())


# ── schema ───────────────────────────────────────────────


def test_phase2b_schema_round_trips_are_lossless():
    lst = demo_list_screen("telegram-list", "iphone")
    assert ListScreen.from_dict(lst.to_dict()).to_dict() \
        == lst.to_dict()
    status = demo_status_screen("whatsapp-status", "iphone")
    assert StatusScreen.from_dict(
        status.to_dict()).to_dict() == status.to_dict()
    views = demo_status_views("android")
    assert StatusViewsScreen.from_dict(
        views.to_dict()).to_dict() == views.to_dict()
    call = demo_call_screen("whatsapp-call", "android")
    assert CallScreen.from_dict(call.to_dict()).to_dict() \
        == call.to_dict()
    history = demo_call_history()
    assert CallHistoryScreen.from_dict(
        history.to_dict()).to_dict() == history.to_dict()


def test_phase2b_schema_rejects_states_the_apps_lack():
    with pytest.raises(ValueError):
        ListRow("x", "X", preview_kind="hologram")
    with pytest.raises(ValueError):
        ListRow("x", "X", ring="sparkly")
    with pytest.raises(ValueError):
        ListRow("x", "X", row_type="alien")
    with pytest.raises(ValueError):
        ListScreen("whatsapp", "T", flavor="wallpaper")
    with pytest.raises(ValueError):
        StatusScreen(Participant("a", "A"), kind="hologram")
    with pytest.raises(ValueError):
        StatusScreen(Participant("a", "A"), bg="purple")
    with pytest.raises(ValueError):
        CallScreen(state="ringing-back")
    with pytest.raises(ValueError):
        CallScreen(controls=["explode"])
    with pytest.raises(ValueError):
        CallHistoryEntry("x", "X", type="blocked")
    with pytest.raises(ValueError):
        CallHistoryScreen(
            chrome=DeviceChrome(style="android"))


def test_status_progress_model_clamps_to_the_census():
    screen = StatusScreen(
        Participant("a", "A"), kind="text", segments=3,
        index=9, played=4.5, text_size=200)
    assert screen.index == 2
    assert screen.played == 1.0
    assert screen.text_size == 60  # the census's 14–60 px


# ── list renders ─────────────────────────────────────────


@pytest.mark.parametrize("name,style,width", [
    ("whatsapp-list", "android", 800),
    ("whatsapp-list", "iphone", 780),
    ("whatsapp-calls", "android", 800),
    ("whatsapp-status-list", "iphone", 780),
    ("instagram-list", "iphone", 780),
    ("telegram-list", "android", 800),
    ("telegram-list", "iphone", 780),
    ("snapchat-list", "android", 800),
])
def test_list_demos_are_real_images(name, style, width):
    screen = demo_list_screen(name, style)
    data, drawn = render_list_screen(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(data))
    assert img.width == width == drawn["size"][0]
    assert img.height == drawn["size"][1] > 500
    for region in ("status_bar", "header"):
        assert region in drawn["regions"], region
    rows = [e for e in drawn["elements"]
            if e["kind"] in ("row", "call_row")]
    expected = len(screen.rows) + (
        1 if screen.archived >= 0 else 0)
    assert len(rows) == expected
    for entry in rows:
        x, y, w, h = entry["box"]
        assert x >= 0 and x + w <= drawn["size"][0]
        assert y >= 0 and y + h <= drawn["size"][1]


def test_list_render_is_deterministic():
    screen = demo_list_screen("whatsapp-list", "android")
    first, _ = render_list_screen(screen)
    second, _ = render_list_screen(screen)
    assert first == second


def test_whatsapp_list_states_are_in_the_record():
    _data, drawn = render_list_screen(
        demo_list_screen("whatsapp-list", "android"))
    adaeze = _el(drawn, id="w1")
    assert adaeze["unread"] == 2
    assert adaeze["ring"] == "unseen"
    crew = _el(drawn, id="w2")
    assert crew["pinned"] is True
    assert crew["preview_kind"] == "photo"
    assert crew["row_type"] == "group"
    mum = _el(drawn, id="w4")
    assert mum["draft"] is True
    channel = _el(drawn, id="w6")
    assert channel["muted"] is True
    assert channel["preview_kind"] == "call_missed_voice"
    typing = _el(drawn, id="w7")
    assert typing["preview_kind"] == "typing"
    # the bottom tab bar is a region, Calls badge rides it
    assert "tab_bar" in drawn["regions"]


def test_telegram_list_folders_archived_and_stories():
    _data, drawn = render_list_screen(
        demo_list_screen("telegram-list", "iphone"))
    assert "tabs" in drawn["regions"]
    assert "stories" in drawn["regions"]
    archived = _el(drawn, id="archived")
    assert archived["row_type"] == "archived"
    assert archived["unread"] == 2
    channel = _el(drawn, id="t2")
    assert channel["verified"] is True
    assert channel["mention"] is True
    saved = _el(drawn, id="t1")
    assert saved["row_type"] == "saved"
    stories = [e for e in drawn["elements"]
               if e["kind"] == "story"]
    assert [s["ring"] for s in stories] == ["unseen", "seen"]
    # the Android form carries no stories row (census
    # §4.7: the stories row is the iPhone form's)
    _d2, drawn_a = render_list_screen(
        demo_list_screen("telegram-list", "android"))
    assert "stories" not in drawn_a["regions"]


def test_instagram_list_notes_and_unseen_states():
    _data, drawn = render_list_screen(
        demo_list_screen("instagram-list", "iphone"))
    stories = [e for e in drawn["elements"]
               if e["kind"] == "story"]
    assert stories[0]["note"] == "studio day 🎬"
    unseen = _el(drawn, id="i1")
    assert unseen["unread"] == 1
    sent = _el(drawn, id="i3")
    assert sent["status"] == "sent"
    seen = _el(drawn, id="i4")
    assert seen["status"] == "seen"


def test_status_list_sections_and_ring_segments():
    _data, drawn = render_list_screen(
        demo_list_screen("whatsapp-status-list", "android"))
    sections = [e["text"] for e in drawn["elements"]
                if e["kind"] == "section"]
    assert sections == ["Recent updates", "Viewed updates",
                        "Muted updates", "Channels"]
    recent = _el(drawn, id="s1")
    assert recent["ring"] == "unseen"
    assert recent["segments"] == 3
    viewed = _el(drawn, id="s3")
    assert viewed["ring"] == "seen"
    muted = _el(drawn, id="s4")
    assert muted["muted"] is True


def test_call_list_distinguishes_without_colour():
    # census §26 exit proof: direction, medium and the
    # missed state are record facts, not shades
    _data, drawn = render_list_screen(
        demo_list_screen("whatsapp-calls", "android"))
    by_id = {e["id"]: e["call"] for e in drawn["elements"]
             if e["kind"] == "call_row"}
    assert by_id["c1"] == {"direction": "call_in_voice",
                           "type": "voice", "missed": False}
    assert by_id["c2"] == {"direction": "call_out_video",
                           "type": "video", "missed": False}
    assert by_id["c3"]["missed"] is True
    assert by_id["c4"] == {"direction": "call_missed_video",
                           "type": "video", "missed": True}


def test_list_dark_mode_and_style_switch():
    light, drawn_l = render_list_screen(
        demo_list_screen("whatsapp-list", "android"))
    dark, _dd = render_list_screen(
        demo_list_screen("whatsapp-list", "android",
                         dark=True))
    iphone, drawn_i = render_list_screen(
        demo_list_screen("whatsapp-list", "iphone"))
    assert dark != light
    assert drawn_i["size"][0] == 780
    assert drawn_l["size"][0] == 800


def test_unknown_list_design_is_an_honest_error():
    screen = demo_list_screen("whatsapp-list", "android")
    screen.platform = "myspace"
    with pytest.raises(MockScreensError):
        render_list_screen(screen)


# ── status renders ───────────────────────────────────────


@pytest.mark.parametrize("name,style,width,height", [
    ("whatsapp-status", "iphone", 780, 1688),
    ("whatsapp-status", "android", 800, 1600),
    ("whatsapp-status-text", "android", 800, 1600),
])
def test_status_demos_are_full_screen_images(
        name, style, width, height):
    screen = demo_status_screen(name, style)
    data, drawn = render_status_screen(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(data))
    # a status is a full phone screen, not content height
    assert (img.width, img.height) == (width, height)
    assert drawn["size"] == [width, height]
    for region in ("status_bar", "progress", "header",
                   "reply_bar", "nav"):
        assert region in drawn["regions"], region


def test_status_progress_at_boundary_values():
    base = demo_status_screen("whatsapp-status", "android")
    base.segments, base.index, base.played = 1, 0, 0.0
    _d, drawn = render_status_screen(base)
    assert drawn["progress"] == {
        "segments": 1, "index": 0, "played": 0.0}
    base.played = 1.0
    data_full, drawn_full = render_status_screen(base)
    assert drawn_full["progress"]["played"] == 1.0
    data_zero, _dz = render_status_screen(
        demo_status_screen("whatsapp-status", "android"))
    assert data_full != data_zero


def test_status_photo_placeholder_only_without_media():
    # the demo photo status carries no picture: the
    # placeholder scene stands in, recorded as such
    _data, drawn = render_status_screen(
        demo_status_screen("whatsapp-status", "android"))
    media = _el(drawn, kind="status_media")
    assert media["media_source"] == "placeholder"


def test_status_real_photo_lands_full_screen(tmp_path):
    img = _solid_png(tmp_path / "status.png", (200, 30, 30))
    screen = StatusScreen(
        owner=Participant("adaeze", "Adaeze Nwosu"),
        kind="photo", media=str(img), when="9:12",
        chrome=DeviceChrome(style="iphone"))
    data, drawn = render_status_screen(screen)
    media = _el(drawn, kind="status_media")
    assert media["media_source"] == "real"
    pixels = Image.open(io.BytesIO(data)).convert("RGB")
    assert pixels.getpixel((390, 800)) == (200, 30, 30)


def test_status_fit_and_surround_variants_render(tmp_path):
    img = _solid_png(tmp_path / "wide.png", (30, 60, 200),
                     size=(800, 100))
    for surround in ("black", "photo"):
        screen = StatusScreen(
            owner=Participant("adaeze", "Adaeze Nwosu"),
            kind="photo", media=str(img), fit="fit",
            surround=surround,
            chrome=DeviceChrome(style="iphone"))
        data, drawn = render_status_screen(screen)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        media = _el(drawn, kind="status_media")
        assert media["fit"] == "fit"
        assert media["surround"] == surround


def test_status_text_and_like_states_are_recorded():
    screen = demo_status_screen("whatsapp-status-text",
                                "android")
    data, drawn = render_status_screen(screen)
    text = _el(drawn, kind="status_text")
    assert text["text"] == screen.text
    photo = demo_status_screen("whatsapp-status", "iphone")
    _d2, drawn2 = render_status_screen(photo)
    assert drawn2["liked"] is True
    reply = _el(drawn2, kind="status_reply_bar")
    assert reply["reactions"] == ["❤️", "🔥", "😮"]
    assert reply["liked"] is True


def test_status_views_sheet_counts_and_rows():
    screen = demo_status_views("android")
    data, drawn = render_status_views(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert drawn["view_count"] == 3
    viewers = [e for e in drawn["elements"]
               if e["kind"] == "viewer"]
    assert [v["name"] for v in viewers] == [
        "Adaeze Nwosu", "Bola", "Chiamaka"]
    assert viewers[0]["when"] == "Today, 9:31 AM"
    again, _d = render_status_views(screen)
    assert again == data


# ── call renders ─────────────────────────────────────────


@pytest.mark.parametrize("name,style,width,height", [
    ("call", "android", 800, 1600),
    ("call-incoming", "iphone", 780, 1688),
    ("whatsapp-call", "android", 800, 1600),
    ("facebook-call", "iphone", 780, 1688),
    ("instagram-call", "android", 800, 1600),
    ("hangouts-call", "android", 800, 1600),
])
def test_call_demos_are_full_screen_images(
        name, style, width, height):
    screen = demo_call_screen(name, style)
    data, drawn = render_call_screen(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(data))
    assert (img.width, img.height) == (width, height)
    assert drawn["state"] == screen.state
    controls = [e["name"] for e in drawn["elements"]
                if e["kind"] == "control"]
    assert controls == drawn["controls"]


def test_call_incoming_and_ongoing_clusters():
    _d, incoming = render_call_screen(
        demo_call_screen("call-incoming", "iphone"))
    assert incoming["controls"] == [
        "message", "remind", "decline", "accept"]
    assert incoming["status"] == "Incoming voice call"
    _d2, ongoing = render_call_screen(
        demo_call_screen("call", "android"))
    assert ongoing["controls"] == [
        "mute", "keypad", "speaker", "video", "message",
        "end"]
    assert ongoing["status"] == "02:15"


def test_call_screen_custom_controls_win():
    screen = CallScreen(
        platform="generic",
        caller=Participant("adaeze", "Adaeze"),
        state="ongoing", controls=["mute", "end"],
        chrome=DeviceChrome(style="android"))
    _data, drawn = render_call_screen(screen)
    assert drawn["controls"] == ["mute", "end"]


def test_call_video_feed_composes_the_real_picture(
        tmp_path):
    img = _solid_png(tmp_path / "feed.png", (20, 160, 60))
    screen = CallScreen(
        platform="whatsapp",
        caller=Participant("adaeze", "Adaeze Nwosu"),
        state="ongoing", status="01:12", video=True,
        media=str(img), chrome=DeviceChrome(style="android"))
    data, drawn = render_call_screen(screen)
    feed = _el(drawn, kind="caller_feed")
    assert feed["media_source"] == "real"
    pixels = Image.open(io.BytesIO(data)).convert("RGB")
    assert pixels.getpixel((400, 700)) == (20, 160, 60)


def test_call_letter_avatar_when_no_picture():
    # census §6.2: a letter avatar in any colour when no
    # picture is used — the caller's own colour wins
    screen = demo_call_screen("call-incoming", "android")
    data, drawn = render_call_screen(screen)
    caller = _el(drawn, kind="caller")
    assert caller["avatar_source"] == "placeholder"
    assert caller["name"] == "+234 800 111 2222"
    again, _d = render_call_screen(screen)
    assert again == data


def test_call_history_states_are_in_the_record():
    screen = demo_call_history()
    data, drawn = render_call_history(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert drawn["size"][0] == 780  # iPhone chrome only
    assert drawn["voicemail_count"] == 2
    assert drawn["tabs"] == ["Favourites", "Calls",
                              "Contacts", "Keypad",
                              "Voicemail"]
    by_id = {e["id"]: e for e in drawn["elements"]}
    assert by_id["h1"]["type"] == "missed"
    assert by_id["h1"]["avatar_mode"] == "picture"
    assert by_id["h1"]["count"] == 2
    assert by_id["h2"]["type"] == "outgoing"
    assert by_id["h2"]["avatar_mode"] == "default"
    assert by_id["h3"]["avatar_mode"] == "letter"
    assert "tab_bar" in drawn["regions"]
    again, _d = render_call_history(screen)
    assert again == data


# ── media honesty (the hand-off contract) ────────────────


def test_broken_status_photo_fails_with_the_reason(
        tmp_path):
    junk = tmp_path / "status.png"
    junk.write_text("not an image")
    screen = StatusScreen(
        owner=Participant("adaeze", "Adaeze"),
        kind="photo", media=str(junk),
        chrome=DeviceChrome(style="android"))
    with pytest.raises(MockScreensError) as caught:
        render_status_screen(screen)
    assert "could not be used" in str(caught.value)
    service = MockScreenService(media_dir=tmp_path)
    out = service.render_status(screen)
    assert out["sent"] is False
    assert "could not be used" in out["reason"]


def test_broken_avatar_in_a_list_fails_the_render(
        tmp_path):
    junk = tmp_path / "face.png"
    junk.write_text("not an image")
    screen = ListScreen(
        platform="whatsapp", title="WhatsApp",
        participants=[Participant("adaeze", "Adaeze",
                                  avatar=str(junk))],
        rows=[ListRow("r1", "Adaeze",
                      participant_id="adaeze",
                      preview="hi", time="9:41 AM")],
        chrome=DeviceChrome(style="android"))
    with pytest.raises(MockScreensError) as caught:
        render_list_screen(screen)
    assert "could not be used" in str(caught.value)
    # and a missing file is a failure too, never a
    # placeholder dressed as the person's photo
    screen.participants[0].avatar = str(tmp_path / "gone.png")
    with pytest.raises(MockScreensError):
        render_list_screen(screen)


def test_real_avatar_in_a_list_is_composited(tmp_path):
    img = _solid_png(tmp_path / "face.png", (10, 180, 90),
                     size=(80, 80))
    screen = ListScreen(
        platform="whatsapp", title="WhatsApp",
        participants=[Participant("adaeze", "Adaeze",
                                  avatar=str(img))],
        rows=[ListRow("r1", "Adaeze",
                      participant_id="adaeze",
                      preview="hi", time="9:41 AM")],
        chrome=DeviceChrome(style="android"))
    data, drawn = render_list_screen(screen)
    row = _el(drawn, id="r1")
    assert row["avatar_source"] == "real"
    pixels = Image.open(io.BytesIO(data)).convert("RGB")
    _x, y, _w, h = row["box"]
    # the avatar disc's centre (logical 14 + 26, at the
    # row's mid-height) IS the source's colour
    assert pixels.getpixel((80, y + h // 2)) == (10, 180, 90)


def test_broken_caller_picture_fails_the_render(tmp_path):
    junk = tmp_path / "caller.png"
    junk.write_text("not an image")
    screen = CallScreen(
        platform="generic",
        caller=Participant("adaeze", "Adaeze"),
        state="ongoing", media=str(junk),
        chrome=DeviceChrome(style="android"))
    service = MockScreenService(media_dir=tmp_path)
    out = service.render_call(screen)
    assert out["sent"] is False
    assert "could not be used" in out["reason"]


def test_supplied_avatar_via_handoff_is_strict(tmp_path):
    screen = demo_list_screen("whatsapp-list", "android")
    out = MockScreenService(media_dir=tmp_path).render_list(
        screen, avatars={"adaeze": b"not an image"})
    assert out["sent"] is False
    assert "could not be used" in out["reason"]
    good = MockScreenService(media_dir=tmp_path).render_list(
        screen, avatars={"adaeze": _png_bytes((10, 180, 90))})
    row = _el(good["drawn"], id="w1")
    assert row["avatar_source"] == "real"


# ── service + command reachability ───────────────────────


def test_service_delivers_each_phase2b_class(tmp_path):
    service = MockScreenService(media_dir=tmp_path / "m")
    cases = [
        (service.render_list,
         demo_list_screen("whatsapp-list", "android")),
        (service.render_status,
         demo_status_screen("whatsapp-status", "android")),
        (service.render_status_views,
         demo_status_views("android")),
        (service.render_call,
         demo_call_screen("call", "android")),
        (service.render_call_history, demo_call_history()),
    ]
    for door, screen in cases:
        send = _Send()
        out = door(screen, audience="personal", send=send)
        assert out["sent"] is True
        ref, caption = send.calls[0]
        assert ref.kind == "image" and ref.mime == "image/png"
        assert Path(ref.path).exists()
        assert "mock screen" in caption


def test_parse_dispatch_modes():
    cases = {
        "demo whatsapp-list": "demo-list",
        "demo whatsapp-calls": "demo-list",
        "demo whatsapp-status": "demo-status",
        "demo whatsapp-status-views": "demo-views",
        "demo call": "demo-call",
        "demo iphone-call-history": "demo-history",
        "demo whatsapp": "demo",
    }
    for tail, mode in cases.items():
        _obj, got = parse_mock_tail(tail)
        assert got == mode, tail
    with pytest.raises(MockScreensError):
        parse_mock_tail("demo myspace-list")


class _Runtime(RuntimeMockScreensMixin):
    def __init__(self, tmp_path, operator=True):
        self._tmp = tmp_path
        self._operator = operator
        self.send = _Send()

    def _is_operator(self, message):
        return self._operator

    def _expression_root(self):
        return self._tmp

    def _makers_audience(self, message):
        return "personal"

    def _expression_send(self, message, ref, caption=""):
        return self.send(ref, caption)


def _message():
    return SimpleNamespace(
        chat=SimpleNamespace(kind="dm"), sender="owner")


@pytest.mark.parametrize("tail", [
    "demo whatsapp-list", "demo whatsapp-calls",
    "demo whatsapp-status-list", "demo instagram-list",
    "demo telegram-list", "demo snapchat-list",
    "demo whatsapp-status", "demo whatsapp-status-text",
    "demo whatsapp-status-views", "demo call",
    "demo call-incoming", "demo whatsapp-call",
    "demo facebook-call", "demo instagram-call",
    "demo hangouts-call", "demo iphone-call-history",
])
def test_command_delivers_each_phase2b_demo(tmp_path, tail):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock("mock", tail, _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1
    ref, caption = runtime.send.calls[0]
    assert ref.kind == "image"
    assert Path(ref.path).exists()
    assert "mock screen" in caption


def test_command_phase2b_stays_owner_gated(tmp_path):
    runtime = _Runtime(tmp_path, operator=False)
    reply = runtime._control_mock(
        "mock", "demo whatsapp-list", _message())
    assert reply == "that one's just for the owner."
    assert runtime.send.calls == []


def test_chat_renders_untouched_by_the_new_classes():
    # the phase-1 regression contract lives in the older
    # suite; here the seam fact: adding the classes did
    # not change the chat registry's contents
    from nomorals.social.chat.mock_screens import (
        CHAT_DESIGNS,
    )

    assert set(CHAT_DESIGNS) == {
        "whatsapp", "instagram", "telegram", "messenger",
        "snapchat", "imessage", "sms"}
    _data, drawn = render_screen(
        demo_screen("whatsapp", "android"))
    assert drawn["platform"] == "whatsapp"
