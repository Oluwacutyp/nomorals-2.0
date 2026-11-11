"""Mock screens (FakeDetail port, phases 0+1): the screen
schema, the platform-design registry, the Pillow renderer
foundation and the flagship chats — WhatsApp 1:1, WhatsApp
group, Instagram DM — plus the /mock command wiring.

Network-free: renders are local Pillow work, avatar
fixtures are tiny generated PNGs, and the delivery seam is
a recording stub.
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
