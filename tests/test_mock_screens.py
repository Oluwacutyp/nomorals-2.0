"""Mock screens (FakeDetail port, phases 0+1+2): the screen
schema, the platform-design registry, the Pillow renderer
foundation and the chat families — WhatsApp 1:1 + group,
Instagram DM + group, Telegram, Messenger, Snapchat,
iMessage and Android Messages — plus the /mock command
wiring.

Network-free: renders are local Pillow work, avatar
fixtures are tiny generated PNGs, and the delivery seam is
a recording stub.
"""
from __future__ import annotations

import io
import math
import shutil
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_mock_screens import (
    RuntimeMockScreensMixin,
)
from nomorals.social.chat.base import MediaRef
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    parse_control,
)
from nomorals.social.chat.expression import MEDIA_DELIVERED_KINDS
from nomorals.social.chat.mock_screens import (
    CHAT_DESIGNS,
    MOCK_COMMANDS,
    DeviceChrome,
    Message,
    MockScreensError,
    MockScreenService,
    Participant,
    Screen,
)
from nomorals.social.chat.mock_screens.authoring import (
    demo_screen,
    parse_mock_tail,
)
from nomorals.social.chat.mock_screens.media import (
    MediaError,
    format_duration,
    probe_audio,
)
from nomorals.social.chat.mock_screens.render import (
    render_screen,
)

# ── fixtures ─────────────────────────────────────────────


def _png_bytes(color=(30, 120, 200), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _Send:
    """The expression-send seam: records (ref, caption)."""

    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, ref, caption=""):
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=self.ok, error="")


def _rich_screen() -> Screen:
    return Screen(
        platform="whatsapp", group=True, title="Crew",
        subtitle="Adaeze, Bola, you",
        participants=[
            Participant("me", "You"),
            Participant("adaeze", "Adaeze", color="#9B59D0",
                        admin=True),
            Participant("bola", "Bola"),
        ],
        messages=[
            Message("s1", "system", kind="system",
                    text="Adaeze created the group"),
            Message("d1", "system", kind="divider",
                    text="Today"),
            Message("t1", "adaeze", text="standup in five",
                    time="9:41 AM", reactions=["🔥"]),
            Message("t2", "me", text="on my way",
                    time="9:42 AM", ticks="delivered"),
            Message("t3", "me", text="grabbing coffee first",
                    time="9:42 AM", ticks="read",
                    reply_to="t1"),
            Message("i1", "bola", kind="image",
                    caption="the board", time="9:43 AM"),
            Message("v1", "bola", kind="voice",
                    duration="0:07", time="9:44 AM"),
            Message("t4", "bola", text="seen it",
                    time="9:45 AM", seen=True),
        ],
        chrome=DeviceChrome(style="iphone", clock="9:41",
                            battery=62, network="5G", unread=2),
        dark=True)


# ── schema ───────────────────────────────────────────────


def test_schema_round_trip_is_lossless():
    screen = _rich_screen()
    again = Screen.from_dict(screen.to_dict())
    assert again.to_dict() == screen.to_dict()
    assert again.participant("adaeze").admin is True
    assert again.message_by_id("t3").reply_to == "t1"
    assert again.chrome.style == "iphone"


def test_schema_rejects_states_the_apps_do_not_have():
    with pytest.raises(ValueError):
        Message("x", "me", kind="hologram")
    with pytest.raises(ValueError):
        Message("x", "me", ticks="teleported")
    with pytest.raises(ValueError):
        DeviceChrome(style="blackberry")


# ── rendering ────────────────────────────────────────────


@pytest.mark.parametrize("name,style,width", [
    ("whatsapp", "android", 800),
    ("whatsapp", "iphone", 780),
    ("group", "android", 800),
    ("group", "iphone", 780),
    ("instagram", "iphone", 780),
    ("instagram", "android", 800),
])
def test_flagship_renders_are_real_images(name, style, width):
    screen = demo_screen(name, style)
    data, drawn = render_screen(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(data))
    assert img.width == width == drawn["size"][0]
    assert img.height == drawn["size"][1] > 900
    assert drawn["style"] == style
    for region in ("status_bar", "header", "input_bar", "nav"):
        assert region in drawn["regions"], region
    bubbles = [e for e in drawn["elements"]
               if e["kind"] in ("text", "image", "voice")]
    assert len(bubbles) == sum(
        1 for m in screen.messages
        if m.kind in ("text", "image", "voice"))
    for el in bubbles:
        x, y, w, h = el["box"]
        assert x >= 0 and x + w <= drawn["size"][0]
        assert y >= 0 and y + h <= drawn["size"][1]
        assert w > 40 and h > 20


def test_render_is_deterministic():
    screen = _rich_screen()
    first, _ = render_screen(screen)
    second, _ = render_screen(screen)
    assert first == second


def test_style_switch_changes_the_chrome():
    screen = demo_screen("whatsapp", "android")
    data_a, drawn_a = render_screen(screen)
    screen_iphone = demo_screen("whatsapp", "iphone")
    data_i, drawn_i = render_screen(screen_iphone)
    assert drawn_a["size"][0] != drawn_i["size"][0]
    assert data_a != data_i
    # the iPhone wears the notch pill and the unread count
    assert drawn_i["regions"]["status_bar"][3] > \
        drawn_a["regions"]["status_bar"][3]


def test_ticks_and_seen_states_are_in_the_record():
    _data, drawn = render_screen(_rich_screen())
    by_id = {e["id"]: e for e in drawn["elements"]}
    assert by_id["t2"]["ticks"] == "delivered"
    assert by_id["t3"]["ticks"] == "read"
    assert by_id["t1"]["reactions"] == ["🔥"]
    ig = demo_screen("instagram", "iphone")
    _d, ig_drawn = render_screen(ig)
    assert any(e["kind"] == "seen" for e in ig_drawn["elements"])


def test_group_members_get_distinct_name_colours():
    from nomorals.social.chat.mock_screens.render import (
        _participant_color,
    )

    screen = demo_screen("group", "android")
    colours = {
        _participant_color(p, screen)
        for p in screen.participants if p.id != "me"}
    assert len(colours) == 3


def test_supplied_avatar_renders_and_missing_falls_back():
    screen = demo_screen("whatsapp", "android")
    plain, _ = render_screen(screen)
    with_avatar, _ = render_screen(
        screen, avatars={"adaeze": _png_bytes()})
    assert with_avatar != plain
    broken, _ = render_screen(
        screen, avatars={"adaeze": b"not an image"})
    assert broken == plain


def test_unknown_platform_is_an_honest_error():
    screen = demo_screen("whatsapp", "android")
    screen.platform = "myspace"
    with pytest.raises(MockScreensError):
        render_screen(screen)


# ── authoring ────────────────────────────────────────────


def test_demo_screens_exist_per_family():
    for name in ("whatsapp", "group", "instagram"):
        screen = demo_screen(name, "android")
        assert screen.platform in CHAT_DESIGNS
        assert len(screen.messages) >= 5


def test_compact_form_builds_a_screen():
    screen, _mode = parse_mock_tail(
        "whatsapp iphone with Adaeze Nwosu "
        "Adaeze: we still on? ;; me: yes, by 11 [read] "
        ";; Adaeze: bring the ring light [react 🔥] "
        ";; me: ^ already packed")
    assert screen.platform == "whatsapp"
    assert screen.chrome.style == "iphone"
    assert screen.title == "Adaeze Nwosu"
    texts = [m for m in screen.messages if m.kind == "text"]
    assert texts[0].sender != "me"
    assert texts[1].sender == "me"
    assert texts[1].ticks == "read"
    assert texts[2].reactions == ["🔥"]
    assert texts[3].reply_to == texts[2].id
    assert screen.messages[0].kind == "divider"
    data, drawn = render_screen(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert set(drawn["participants"]) == {"me", "adaeze"}


def test_compact_form_group_and_dark():
    screen, _mode = parse_mock_tail(
        "group dark Bola: first ;; me: second [sent] "
        ";; ! Bola added Chiamaka")
    assert screen.group is True and screen.dark is True
    kinds = [m.kind for m in screen.messages]
    assert "system" in kinds


def test_compact_form_refuses_shapeless_input():
    with pytest.raises(MockScreensError):
        parse_mock_tail("")
    with pytest.raises(MockScreensError):
        parse_mock_tail("whatsapp just some words")


# ── service + command ────────────────────────────────────


def test_service_delivers_through_the_seam(tmp_path):
    service = MockScreenService(media_dir=tmp_path / "mock_media")
    send = _Send()
    out = service.render(demo_screen("whatsapp", "android"),
                         audience="personal", send=send)
    assert out["sent"] is True
    assert len(send.calls) == 1
    ref, caption = send.calls[0]
    assert ref.kind == "image" and ref.mime == "image/png"
    assert ref.audience == "personal"
    assert Path(ref.path).exists()
    assert "mock screen" in caption


def test_service_reports_a_failed_send_honestly(tmp_path):
    service = MockScreenService(media_dir=tmp_path / "mock_media")
    out = service.render(demo_screen("group", "android"),
                         send=_Send(ok=False))
    assert out["sent"] is False
    assert out["reason"]


class _Runtime(RuntimeMockScreensMixin):
    """The mixin over recording stubs for the live seams."""

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


def test_command_delivers_a_photo(tmp_path):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock(
        "mock", "demo whatsapp", _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1
    ref, _caption = runtime.send.calls[0]
    assert ref.kind == "image"
    assert Path(ref.path).exists()


def test_command_builds_from_the_compact_form(tmp_path):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock(
        "mock", "instagram Adaeze: hi ;; me: hello [read]",
        _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1


def test_command_is_owner_gated(tmp_path):
    runtime = _Runtime(tmp_path, operator=False)
    reply = runtime._control_mock(
        "mock", "demo whatsapp", _message())
    assert reply == "that one's just for the owner."
    assert runtime.send.calls == []


def test_command_answers_usage_without_a_tail(tmp_path):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock("mock", "", _message())
    assert reply.startswith("usage: /mock")


# ── registry ─────────────────────────────────────────────


def test_registry_presence_all_tables():
    assert MOCK_COMMANDS == ("mock",)
    assert "mock" in CONTROL_COMMANDS
    assert "mock" in COMMAND_DETAILS
    assert "mock" in LIST_ONELINERS
    flat = [k for _name, ks in LIST_GROUPS for k in ks]
    assert flat.count("mock") == 1
    assert "mock" in MEDIA_DELIVERED_KINDS
    # owner-only: never a public command
    assert "mock" not in PUBLIC_COMMANDS
    parsed = parse_control("/mock demo whatsapp")
    assert parsed is not None and parsed.kind == "mock"


# ── phase 2: the remaining chat families ─────────────────

#: every chat tool the FakeDetail census names (§3.1,
#: homepage + privacy page + first-party verified pages),
#: mapped to the design that renders it
CENSUS_RENDERED = {
    "WhatsApp Chat": "whatsapp",
    "WhatsApp Group Chat": "whatsapp",
    "Instagram DM": "instagram",
    "Instagram Group Chat": "instagram",
    "iPhone Text Message / iMessage": "imessage",
    "Android Text Message": "sms",
    "Messenger Chat": "messenger",
    "Facebook Chat": "messenger",
    "Facebook Group Chat": "messenger",
    "Telegram Chat": "telegram",
    "Snapchat Chat": "snapchat",
    "Snapchat Group Chat": "snapchat",
}

#: census-named chat tools NOT rendered in this phase,
#: each with its census reason (§3.8–§3.10: verified or
#: named there; a build-time pass captures each one's
#: exact chrome before implementation — never guessed)
CENSUS_REMAINDERS = {
    "Google Chat": "census §3.8 flagship-verified; "
                   "deferred past this phase's families",
    "LINE Chat": "census §3.9 flagship-verified; "
                 "deferred past this phase's families",
    "Twitter/X Chat": "census §3.10 named, not inspected "
                      "to flagship depth",
    "Skype Chat": "census §3.10 named, not inspected "
                  "to flagship depth",
    "WeChat Chat": "census §3.10 named, not inspected "
                   "to flagship depth",
    "LinkedIn Chat": "census §3.10 named, not inspected "
                     "to flagship depth",
    "Tumblr Chat": "census §3.10 named, not inspected "
                   "to flagship depth",
    "Hangouts Android Messenger": "census §3.10 named, "
                                  "not inspected to "
                                  "flagship depth",
    "Hangouts iPhone Messenger": "census §3.10 named, "
                                 "not inspected to "
                                 "flagship depth",
}


def test_census_chat_catalogue_is_fully_accounted():
    # the homepage's "24 chat/text tools" count (§3.1):
    # named evidence covers 21; the census warns the rest
    # of the count must not be filled by guessing — so the
    # accounted set is exactly the named evidence
    accounted = set(CENSUS_RENDERED) | set(CENSUS_REMAINDERS)
    assert len(accounted) == 21
    for design in CENSUS_RENDERED.values():
        assert design in CHAT_DESIGNS
    assert set(CHAT_DESIGNS) == set(CENSUS_RENDERED.values())
    for reason in CENSUS_REMAINDERS.values():
        assert reason.startswith("census §")


def test_every_design_has_matching_light_dark_palettes():
    for name, design in CHAT_DESIGNS.items():
        assert set(design["light"]) == set(design["dark"]), name


@pytest.mark.parametrize("name,style,width", [
    ("telegram", "android", 800),
    ("telegram", "iphone", 780),
    ("telegram-group", "android", 800),
    ("telegram-group", "iphone", 780),
    ("messenger", "iphone", 780),
    ("messenger", "android", 800),
    ("messenger-group", "android", 800),
    ("snapchat", "android", 800),
    ("snapchat", "iphone", 780),
    ("snapchat-group", "android", 800),
    ("imessage", "iphone", 780),
    ("imessage-sms", "iphone", 780),
    ("sms", "android", 800),
    ("sms-group", "android", 800),
    ("instagram-group", "iphone", 780),
])
def test_phase2_renders_are_real_images(name, style, width):
    screen = demo_screen(name, style)
    data, drawn = render_screen(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(data))
    assert img.width == width == drawn["size"][0]
    assert img.height == drawn["size"][1] > 700
    assert drawn["platform"] == screen.platform
    assert drawn["group"] is screen.group
    for region in ("status_bar", "header", "input_bar", "nav"):
        assert region in drawn["regions"], region
    rows = [e for e in drawn["elements"]
            if e["kind"] in ("text", "image", "voice")]
    assert len(rows) == sum(
        1 for m in screen.messages
        if m.kind in ("text", "image", "voice"))
    for el in rows:
        x, y, w, h = el["box"]
        assert x >= 0 and x + w <= drawn["size"][0]
        assert y >= 0 and y + h <= drawn["size"][1]
        assert w > 40 and h > 20


def test_phase2_demo_scripts_are_real_conversations():
    for name in ("telegram", "messenger", "snapchat",
                 "imessage", "sms"):
        assert len(demo_screen(name, "android").messages) >= 5
    for name in ("telegram-group", "messenger-group",
                 "snapchat-group", "imessage-group",
                 "sms-group", "instagram-group"):
        screen = demo_screen(name, "android")
        assert screen.group is True
        assert len(screen.messages) >= 3


def test_telegram_group_idioms_are_in_the_record():
    _data, drawn = render_screen(
        demo_screen("telegram-group", "android"))
    incoming = [e for e in drawn["elements"]
                if e.get("sender") not in (None, "me")
                and e["kind"] == "text"]
    # group runs sit right of the sender's avatar column
    assert incoming and all(e["box"][0] == 92
                            for e in incoming)
    ticks = {e["ticks"] for e in drawn["elements"]
             if e.get("ticks")}
    assert "read" in ticks


def test_messenger_idioms_are_in_the_record():
    _data, drawn = render_screen(
        demo_screen("messenger", "iphone"))
    kinds = [e["kind"] for e in drawn["elements"]]
    # centred times above runs + the reader's avatar as
    # the read state — Messenger's two signatures
    assert "meta_time" in kinds
    assert "seen_avatar" in kinds


def test_imessage_idioms_and_sms_variant():
    screen = demo_screen("imessage", "iphone")
    _data, drawn = render_screen(screen)
    kinds = [e["kind"] for e in drawn["elements"]]
    assert "meta_time" in kinds
    status = [e for e in drawn["elements"]
              if e["kind"] == "status"]
    assert status and status[0]["text"].startswith("Read")
    green = demo_screen("imessage-sms", "iphone")
    assert green.variant == "sms"
    green_data, _gd = render_screen(green)
    blue_data, _bd = render_screen(screen)
    assert green_data != blue_data
    # the variant survives the schema round-trip
    again = Screen.from_dict(green.to_dict())
    assert again.variant == "sms"


def test_snapchat_rows_and_opened_state():
    _data, drawn = render_screen(
        demo_screen("snapchat", "android"))
    rows = [e for e in drawn["elements"]
            if e["kind"] in ("text", "image")]
    # flat rows: every row spans the same full width
    assert rows and all(e["box"][2] == 776 for e in rows)
    status = [e for e in drawn["elements"]
              if e["kind"] == "status"]
    assert status and status[0]["text"] == "Opened"


def test_sms_has_no_ticks_or_status_lines():
    _data, drawn = render_screen(demo_screen("sms", "android"))
    kinds = [e["kind"] for e in drawn["elements"]]
    assert "status" not in kinds and "meta_time" not in kinds
    assert all(not e.get("ticks") for e in drawn["elements"])


@pytest.mark.parametrize("name", [
    "telegram", "messenger", "snapchat", "imessage"])
def test_phase2_style_and_dark_switches(name):
    light_a, drawn_a = render_screen(demo_screen(name, "android"))
    light_i, drawn_i = render_screen(demo_screen(name, "iphone"))
    assert drawn_a["size"][0] == 800
    assert drawn_i["size"][0] == 780
    assert light_a != light_i
    dark, _dd = render_screen(
        demo_screen(name, "android", dark=True))
    assert dark != light_a


@pytest.mark.parametrize("tail,platform", [
    ("telegram Adaeze: hi ;; me: yo [read]", "telegram"),
    ("messenger Adaeze: hi ;; me: yo", "messenger"),
    ("snapchat Adaeze: hi ;; me: yo [delivered]", "snapchat"),
    ("imessage Adaeze: hi ;; me: yo [read]", "imessage"),
    ("sms Adaeze: hi ;; me: yo", "sms"),
])
def test_compact_form_per_family(tail, platform):
    screen, _mode = parse_mock_tail(tail)
    assert screen.platform == platform
    data, drawn = render_screen(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert drawn["platform"] == platform


def test_compact_form_snapchat_group():
    screen, _mode = parse_mock_tail(
        "snapchat-group with Bola Bola: first ;; me: second")
    assert screen.platform == "snapchat"
    assert screen.group is True


@pytest.mark.parametrize("tail", [
    "demo telegram", "demo telegram-group",
    "demo messenger", "demo messenger-group",
    "demo snapchat", "demo snapchat-group",
    "demo imessage", "demo imessage-sms", "demo sms",
    "demo instagram-group",
])
def test_command_delivers_each_family(tmp_path, tail):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock("mock", tail, _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1
    ref, caption = runtime.send.calls[0]
    assert ref.kind == "image"
    assert Path(ref.path).exists()
    assert "mock screen" in caption


# ── the media hand-off: real images + real voice ─────────


def _solid_png(path, color=(200, 30, 30), size=(300, 200)):
    Image.new("RGB", size, color).save(path)
    return path


def _tone_wav(path, *, seconds=1.5, sr=8000,
              bursts=((0.1, 0.4),)):
    samples = array("h")
    for i in range(int(seconds * sr)):
        t = i / sr
        if any(a <= t < b for a, b in bursts):
            samples.append(int(
                0.8 * 32767 * math.sin(2 * math.pi * 440 * t)))
        else:
            samples.append(0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


def _media_screen(**msg_kwargs):
    return Screen(
        platform="whatsapp", title="Adaeze Nwosu",
        subtitle="online",
        participants=[Participant("me", "You"),
                      Participant("adaeze", "Adaeze Nwosu")],
        messages=[
            Message("d1", "system", kind="divider",
                    text="Today"),
            Message("m1", "adaeze", text="show me",
                    time="9:41 AM"),
            Message("x1", "me", time="9:42 AM", ticks="read",
                    **msg_kwargs),
        ],
        chrome=DeviceChrome(style="android"))


def _element(drawn, mid):
    return next(e for e in drawn["elements"] if e["id"] == mid)


def _sample(data, box):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x, y, w, h = box
    return img.getpixel((x + w // 2, y + h // 2))


def test_media_field_round_trips_losslessly(tmp_path):
    img = _solid_png(tmp_path / "a.png")
    screen = _media_screen(kind="image", media=str(img),
                           caption="the one")
    again = Screen.from_dict(screen.to_dict())
    assert again.messages[-1].media == str(img)
    assert again.to_dict() == screen.to_dict()
    # a pre-hand-off dict (no media key) still loads
    old = screen.to_dict()
    del old["messages"][-1]["media"]
    assert Screen.from_dict(old).messages[-1].media == ""


def test_real_image_lands_in_the_frame(tmp_path):
    img = _solid_png(tmp_path / "red.png", (200, 30, 30))
    data, drawn = render_screen(
        _media_screen(kind="image", media=str(img)))
    el = _element(drawn, "x1")
    assert el["media_source"] == "real"
    # the composited region's centre pixel IS the source's
    assert _sample(data, el["media_box"]) == (200, 30, 30)
    other = _solid_png(tmp_path / "blue.png", (30, 60, 200))
    data2, drawn2 = render_screen(
        _media_screen(kind="image", media=str(other)))
    el2 = _element(drawn2, "x1")
    assert _sample(data2, el2["media_box"]) == (30, 60, 200)


def test_real_image_via_service_handoff(tmp_path):
    # bytes hand-off, keyed by message id, through the
    # service — the generated-image seam's shape
    service = MockScreenService(media_dir=tmp_path)
    out = service.render(
        _media_screen(kind="image"),
        media={"x1": _png_bytes((10, 180, 90), (120, 90))})
    el = _element(out["drawn"], "x1")
    assert el["media_source"] == "real"
    assert _sample(out["data"], el["media_box"]) == (10, 180, 90)
    # and a MediaRef hand-off resolves by its path
    img = _solid_png(tmp_path / "ref.png", (90, 40, 160))
    out2 = service.render(
        _media_screen(kind="image"),
        media={"x1": MediaRef(path=str(img), kind="image")})
    el2 = _element(out2["drawn"], "x1")
    assert el2["media_source"] == "real"
    assert _sample(out2["data"], el2["media_box"]) \
        == (90, 40, 160)


def test_image_frame_height_follows_the_real_aspect(tmp_path):
    wide = _solid_png(tmp_path / "wide.png", size=(800, 100))
    tall = _solid_png(tmp_path / "tall.png", size=(100, 800))
    _d1, drawn_w = render_screen(
        _media_screen(kind="image", media=str(wide)))
    _d2, drawn_t = render_screen(
        _media_screen(kind="image", media=str(tall)))
    hw = _element(drawn_w, "x1")["media_box"][3]
    ht = _element(drawn_t, "x1")["media_box"][3]
    assert ht > hw
    # the no-media placeholder keeps the phase-1 geometry
    _d3, drawn_p = render_screen(_media_screen(kind="image"))
    elp = _element(drawn_p, "x1")
    assert elp["media_source"] == "placeholder"
    assert elp["media_box"][3] == 168 * 2


def test_voice_duration_is_read_from_the_audio(tmp_path):
    wav = _tone_wav(tmp_path / "note.wav", seconds=1.5)
    _data, drawn = render_screen(_media_screen(
        kind="voice", media=str(wav), duration="9:99"))
    el = _element(drawn, "x1")
    assert el["media_source"] == "real"
    # the true 1.5s wins over the authored display string
    assert el["duration_s"] == pytest.approx(1.5, abs=0.01)
    assert format_duration(el["duration_s"]) == "0:02"


def test_voice_waveform_is_derived_from_the_samples(tmp_path):
    silent = _tone_wav(tmp_path / "silent.wav", bursts=())
    probe = probe_audio(str(silent))
    assert probe.bars == tuple(0.0 for _ in range(26))
    early = _tone_wav(tmp_path / "early.wav",
                      bursts=((0.05, 0.35),))
    late = _tone_wav(tmp_path / "late.wav",
                     bursts=((1.1, 1.4),))
    pe = probe_audio(str(early))
    pl = probe_audio(str(late))
    assert len(pe.bars) == 26
    assert pe.bars != pl.bars
    # the energy sits where the burst is, not elsewhere
    assert max(pe.bars[:8]) == 1.0 and max(pe.bars[12:]) == 0.0
    assert max(pl.bars[16:]) == 1.0 and max(pl.bars[:10]) == 0.0
    # and the render record carries the same derivation
    _data, drawn = render_screen(
        _media_screen(kind="voice", media=str(early)))
    el = _element(drawn, "x1")
    assert el["bars"] == [round(b, 3) for b in pe.bars]


def test_voice_probe_reads_mp3_via_ffmpeg(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg on this host")
    wav = _tone_wav(tmp_path / "note.wav", seconds=1.5)
    mp3 = tmp_path / "note.mp3"
    import subprocess

    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav),
         str(mp3)], capture_output=True, timeout=60)
    assert proc.returncode == 0
    probe = probe_audio(str(mp3))
    assert probe.seconds == pytest.approx(1.5, abs=0.15)
    assert max(probe.bars) == 1.0


def test_undecodable_audio_fails_honestly(tmp_path):
    junk = tmp_path / "note.mp3"
    junk.write_text("this is not audio at all")
    with pytest.raises(MediaError):
        probe_audio(str(junk))
    bad_wav = tmp_path / "note.wav"
    bad_wav.write_text("also not a wav")
    with pytest.raises(MediaError):
        probe_audio(str(bad_wav))
    with pytest.raises(MediaError):
        probe_audio(str(tmp_path / "missing.ogg"))
    # at render, the failure is the service's reason —
    # never a synthetic waveform dressed as the note
    service = MockScreenService(media_dir=tmp_path)
    out = service.render(
        _media_screen(kind="voice", media=str(junk)))
    assert out["sent"] is False
    assert "could not be used" in out["reason"]


def test_unreadable_image_fails_honestly(tmp_path):
    junk = tmp_path / "photo.png"
    junk.write_text("not an image")
    service = MockScreenService(media_dir=tmp_path)
    out = service.render(
        _media_screen(kind="image", media=str(junk)))
    assert out["sent"] is False
    assert "could not be used" in out["reason"]
    out2 = service.render(_media_screen(
        kind="image", media=str(tmp_path / "gone.png")))
    assert out2["sent"] is False
    assert "could not be used" in out2["reason"]


def test_no_media_keeps_the_placeholder_contract():
    data, drawn = render_screen(_rich_screen())
    img_el = next(e for e in drawn["elements"]
                  if e["kind"] == "image")
    assert img_el["media_source"] == "placeholder"
    voice_el = next(e for e in drawn["elements"]
                    if e["kind"] == "voice")
    # no probe facts are claimed for a media-less note
    assert "duration_s" not in voice_el
    assert "bars" not in voice_el
    assert "media_box" not in voice_el
    # renders stay deterministic, phase-1/2 style
    again, _d = render_screen(_rich_screen())
    assert again == data


def test_compact_form_media_markers(tmp_path):
    img = _solid_png(tmp_path / "flyer.png")
    wav = _tone_wav(tmp_path / "note.wav")
    screen, _mode = parse_mock_tail(
        "whatsapp me: [photo " + str(img) + "] the flyer ;; "
        "Adaeze: [voice " + str(wav) + "] ;; me: [photo] ;; "
        "me: [photo last]")
    by_id = {m.id: m for m in screen.messages}
    photo = next(m for m in screen.messages
                 if m.kind == "image" and m.media == str(img))
    assert photo.caption == "the flyer"
    voice = next(m for m in screen.messages if m.kind == "voice")
    assert voice.media == str(wav)
    tokens = sorted(m.media for m in screen.messages
                    if m.media.startswith(("attach:", "last:")))
    assert tokens == ["attach:image", "last:image"]
    assert by_id  # ids assigned as usual
    with pytest.raises(MockScreensError):
        parse_mock_tail("whatsapp Adaeze: [voice] some words")


def _media_message(refs):
    return SimpleNamespace(
        chat=SimpleNamespace(kind="dm"), sender="owner",
        media=list(refs))


def test_command_binds_attached_media(tmp_path):
    img = _solid_png(tmp_path / "held.png", (20, 160, 60))
    wav = _tone_wav(tmp_path / "held.wav")
    runtime = _Runtime(tmp_path)
    message = _media_message([
        MediaRef(path=str(img), kind="image",
                 mime="image/png"),
        MediaRef(path=str(wav), kind="voice",
                 mime="audio/ogg"),
    ])
    reply = runtime._control_mock(
        "mock",
        "whatsapp me: [photo] mine ;; Adaeze: [voice] ;; "
        "me: done [read]",
        message)
    assert reply == ""
    assert len(runtime.send.calls) == 1
    sent = Path(runtime.send.calls[0][0].path).read_bytes()
    # the attached image's colour is in the delivered screen
    pixels = Image.open(io.BytesIO(sent)).convert("RGB")
    colours = pixels.getcolors(maxcolors=1 << 24)
    assert colours is not None
    assert any(colour == (20, 160, 60)
               for _n, colour in colours)


def test_command_attach_without_media_is_honest(tmp_path):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock(
        "mock", "whatsapp me: [photo]", _message())
    assert "no image was attached" in reply
    assert runtime.send.calls == []
    reply2 = runtime._control_mock(
        "mock", "whatsapp Adaeze: [voice]", _message())
    assert "no audio was attached" in reply2
    assert runtime.send.calls == []


class _RuntimeGen(_Runtime):
    """Adds the settings seam ``[photo last]`` resolves."""

    def __init__(self, tmp_path, gen_dir):
        super().__init__(tmp_path)
        self.context = SimpleNamespace(
            settings=SimpleNamespace(
                resolve=lambda _p: str(gen_dir)))


def test_command_photo_last_uses_newest_generated(tmp_path):
    gen = tmp_path / "generated"
    gen.mkdir()
    older = _solid_png(gen / "img_old.png", (10, 10, 220))
    newer = _solid_png(gen / "img_new.png", (220, 180, 10))
    import os

    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    runtime = _RuntimeGen(tmp_path, gen)
    reply = runtime._control_mock(
        "mock", "whatsapp me: [photo last] fresh", _message())
    assert reply == ""
    sent = Path(runtime.send.calls[0][0].path).read_bytes()
    pixels = Image.open(io.BytesIO(sent)).convert("RGB")
    colours = pixels.getcolors(maxcolors=1 << 24)
    assert colours is not None
    present = {colour for _n, colour in colours}
    assert (220, 180, 10) in present
    assert (10, 10, 220) not in present


def test_command_photo_last_without_generations_is_honest(
        tmp_path):
    gen = tmp_path / "generated"
    gen.mkdir()
    runtime = _RuntimeGen(tmp_path, gen)
    reply = runtime._control_mock(
        "mock", "whatsapp me: [photo last]", _message())
    assert "no generated image yet" in reply
    assert runtime.send.calls == []
