"""Tests for nomorals/social/render.py — platform-native group/channel output."""

from __future__ import annotations

from nomorals.social.render import (
    render_channel,
    render_community,
    render_group_card,
    render_group_list,
    render_members,
)

_INFO = {
    "subject": "Lagos Foodies",
    "desc": "Best food spots in Lagos",
    "size": 42,
    "admins": ["2348012345678", "2348098765432"],
    "announce": False,
    "restrict": True,
    "ephemeral_hours": 0,
    "creation": 1700000000.0,
    "invite": "https://chat.whatsapp.com/abc123",
}

_MEMBERS = [
    {"jid": "2348012345678@s.whatsapp.net", "name": "Ada", "role": "admin"},
    {"jid": "2348098765432@s.whatsapp.net", "name": "", "role": "superadmin"},
    {"jid": "2348077665544@s.whatsapp.net", "name": "Tunde", "role": ""},
]


def test_group_card_whatsapp():
    out = render_group_card(_INFO, _MEMBERS, "whatsapp")
    assert isinstance(out, str)
    assert "Lagos Foodies" in out
    assert "42 members" in out
    assert "2 admins" in out
    assert "🔒" in out  # restrict flag shown


def test_group_card_telegram_html():
    out = render_group_card(_INFO, _MEMBERS, "telegram")
    assert isinstance(out, str)
    assert "<b>Lagos Foodies</b>" in out
    assert "42 members" in out


def test_group_card_discord_embed():
    out = render_group_card(_INFO, _MEMBERS, "discord")
    assert isinstance(out, dict)
    assert "Lagos Foodies" in out["title"]
    assert any(f["name"] == "Members" for f in out["fields"])


def test_members_roles_shown():
    out = render_members(_MEMBERS, "whatsapp")
    assert "Ada" in out
    assert "👑" in out  # admin crown
    # empty name falls back to phone number, not invented
    assert "2348098765432" in out


def test_members_telegram():
    out = render_members(_MEMBERS, "telegram")
    assert "<b>Members (3)</b>" in out
    assert "👑" in out


def test_members_truncation():
    many = [{"jid": f"{i}@x", "name": f"U{i}", "role": ""} for i in range(100)]
    out = render_members(many, "whatsapp", limit=10)
    assert "…and 90 more" in out


def test_community_with_subgroups():
    subs = [{"subject": "Announcements", "size": 42},
            {"subject": "Recipes", "size": 18}]
    out = render_community(_INFO, subs, "whatsapp")
    assert "Subgroups (2)" in out
    assert "Announcements" in out


def test_channel():
    info = {"name": "Devon Updates", "description": "news", "followers": 1500}
    out = render_channel(info, "telegram")
    assert "Devon Updates" in out
    assert "1500 followers" in out
    wa = render_channel(info, "whatsapp")
    assert "*Devon Updates*" in wa


def test_group_list():
    groups = [{"subject": "A", "size": 10}, {"subject": "B", "size": 5}]
    out = render_group_list(groups, "whatsapp")
    assert "Groups (2)" in out
    assert "A (10 members)" in out
    empty = render_group_list([], "telegram")
    assert "no groups" in empty


def test_never_raises_on_garbage():
    assert render_group_card({}, None, "telegram")
    assert render_members([], "whatsapp")
    assert render_group_card({"subject": None}, None, "bogus-platform")
    assert render_channel({}, "telegram")
