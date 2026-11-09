"""Parity bundle #15 — Telegram bot chat-appearance + links.

Census §7 row 15, four TG-bot rows closed on TelegramBotAdapter
beside the existing admin_* family:

- set chat photo          → admin_set_photo (setChatPhoto, multipart)
- additional invite links → admin_create_invite_link /
                            admin_edit_invite_link /
                            admin_revoke_invite_link
                            (createChatInviteLink / editChatInviteLink /
                            revokeChatInviteLink)
- copy a message          → copy_message (copyMessage — the clean
                            re-post; forward_post is the attributed
                            sibling)
- forum topic create      → admin_forum_topic_create (createForumTopic;
                            the bot had edit but never create)

Exposure: tgbot_* tools in nomorals/tools/social.py, gated
SOCIAL_BULK like every sibling admin tool — the brain reaches the
same verbs from plain language.

Shapes are pinned against a recording fake Bot API (the house
pattern from bundles #9–#12): exact method + params on the wire,
refusals BEFORE traffic for local shape errors, and the API's own
refusal carried through when Telegram says no.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nomorals.social.chat.base import ChatKind, ChatRef

# ── recording fake Bot API ─────────────────────────────────

_LINK = {
    "invite_link": "https://t.me/+AbCdEfGh",
    "creator": {"id": 123456, "is_bot": True, "first_name": "Devon"},
    "creates_join_request": False,
    "is_primary": False,
    "is_revoked": False,
    "name": "",
    "expire_date": 0,
    "member_limit": 0,
    "pending_join_request_count": 0,
}


@pytest.fixture()
def bot_adapter(monkeypatch):
    from nomorals.social.chat.telegram import TelegramBotAdapter

    ad = TelegramBotAdapter(
        token="123456:TEST", chat_allow="",
        media_dir="data/media/telegram-bot-test")
    calls: list[tuple] = []
    uploads: list[tuple] = []

    def fake_api(method, **params):
        calls.append((method, params))
        if method in ("createChatInviteLink", "editChatInviteLink",
                      "revokeChatInviteLink"):
            link = dict(_LINK)
            if "name" in params:
                link["name"] = params["name"]
            if "expire_date" in params:
                link["expire_date"] = params["expire_date"]
            if "member_limit" in params:
                link["member_limit"] = params["member_limit"]
            if "creates_join_request" in params:
                link["creates_join_request"] = True
            if method == "revokeChatInviteLink":
                link["is_revoked"] = True
            return link
        if method == "copyMessage":
            return {"message_id": 777}
        if method == "createForumTopic":
            return {"message_thread_id": 42,
                    "name": params.get("name", ""),
                    "icon_color": params.get("icon_color", 0)}
        return True

    def fake_upload(method, fields, file_field, path, **kw):
        uploads.append((method, fields, file_field, path))
        return True

    monkeypatch.setattr(ad, "_api", fake_api)
    monkeypatch.setattr(ad, "_api_upload", fake_upload)
    ad.calls = calls          # type: ignore[attr-defined]
    ad.uploads = uploads      # type: ignore[attr-defined]
    return ad


def _tg_chat(chat_id="-1001234567890"):
    return ChatRef(platform="telegram-bot", chat_id=chat_id,
                   kind=ChatKind.GROUP, peer=chat_id)


# ── set chat photo ─────────────────────────────────────────

def test_set_photo_uploads_multipart(bot_adapter, tmp_path):
    photo = tmp_path / "group.jpg"
    photo.write_bytes(b"\xff\xd8\xff fake jpeg bytes")
    out = bot_adapter.admin_set_photo(_tg_chat(), str(photo))
    assert out == {"ok": True, "result": True}
    assert bot_adapter.uploads == [
        ("setChatPhoto", {"chat_id": -1001234567890}, "photo", photo)]
    assert bot_adapter.calls == []


def test_set_photo_refusals_before_traffic(bot_adapter, tmp_path):
    out = bot_adapter.admin_set_photo(_tg_chat(), "")
    assert not out["ok"] and "no photo file" in out["error"]
    out = bot_adapter.admin_set_photo(
        _tg_chat(), str(tmp_path / "gone.jpg"))
    assert not out["ok"] and "not found" in out["error"]
    assert bot_adapter.uploads == [] and bot_adapter.calls == []


def test_set_photo_api_refusal_is_the_result(bot_adapter, tmp_path,
                                             monkeypatch):
    photo = tmp_path / "group.png"
    photo.write_bytes(b"\x89PNG fake")

    def boom(method, fields, file_field, path, **kw):
        raise RuntimeError("not enough rights to change chat photo")

    monkeypatch.setattr(bot_adapter, "_api_upload", boom)
    out = bot_adapter.admin_set_photo(_tg_chat(), str(photo))
    assert not out["ok"] and "not enough rights" in out["error"]


# ── additional invite links ────────────────────────────────

def test_create_invite_link_bare(bot_adapter):
    out = bot_adapter.admin_create_invite_link(_tg_chat())
    assert bot_adapter.calls == [
        ("createChatInviteLink", {"chat_id": -1001234567890})]
    assert out["ok"] and out["invite"] == "https://t.me/+AbCdEfGh"
    assert out["name"] == "" and out["expire_date"] == 0
    assert out["member_limit"] == 0
    assert out["creates_join_request"] is False
    assert out["is_revoked"] is False and out["error"] == ""


def test_create_invite_link_full_options(bot_adapter):
    out = bot_adapter.admin_create_invite_link(
        _tg_chat(), name="Launch crew", expire_date=1893456000,
        member_limit=50)
    assert bot_adapter.calls == [("createChatInviteLink", {
        "chat_id": -1001234567890, "name": "Launch crew",
        "expire_date": 1893456000, "member_limit": 50})]
    assert out["ok"] and out["name"] == "Launch crew"
    assert out["expire_date"] == 1893456000
    assert out["member_limit"] == 50


def test_create_invite_link_name_capped_at_32(bot_adapter):
    bot_adapter.admin_create_invite_link(_tg_chat(), name="x" * 40)
    assert bot_adapter.calls[0][1]["name"] == "x" * 32


def test_join_request_link_cannot_carry_member_limit(bot_adapter):
    for fn, kwargs in (
            (bot_adapter.admin_create_invite_link, {}),
            (bot_adapter.admin_edit_invite_link,
             {"invite_link": "https://t.me/+AbCdEfGh"})):
        out = fn(_tg_chat(), member_limit=10,
                 creates_join_request=True, **kwargs)
        assert not out["ok"] and "join-request" in out["error"]
    assert bot_adapter.calls == []


def test_create_join_request_link(bot_adapter):
    out = bot_adapter.admin_create_invite_link(
        _tg_chat(), creates_join_request=True)
    assert bot_adapter.calls == [("createChatInviteLink", {
        "chat_id": -1001234567890, "creates_join_request": True})]
    assert out["ok"] and out["creates_join_request"] is True


def test_edit_invite_link(bot_adapter):
    out = bot_adapter.admin_edit_invite_link(
        _tg_chat(), "https://t.me/+AbCdEfGh", name="Renamed",
        member_limit=5)
    assert bot_adapter.calls == [("editChatInviteLink", {
        "chat_id": -1001234567890,
        "invite_link": "https://t.me/+AbCdEfGh",
        "name": "Renamed", "member_limit": 5})]
    assert out["ok"] and out["name"] == "Renamed"


def test_edit_invite_link_needs_the_link(bot_adapter):
    out = bot_adapter.admin_edit_invite_link(_tg_chat(), "  ")
    assert not out["ok"] and "no invite link" in out["error"]
    assert bot_adapter.calls == []


def test_revoke_invite_link(bot_adapter):
    out = bot_adapter.admin_revoke_invite_link(
        _tg_chat(), "https://t.me/+AbCdEfGh")
    assert bot_adapter.calls == [("revokeChatInviteLink", {
        "chat_id": -1001234567890,
        "invite_link": "https://t.me/+AbCdEfGh"})]
    assert out["ok"] and out["is_revoked"] is True
    out = bot_adapter.admin_revoke_invite_link(_tg_chat(), "")
    assert not out["ok"] and "no invite link" in out["error"]
    assert len(bot_adapter.calls) == 1


def test_invite_link_api_refusal_is_the_result(bot_adapter,
                                               monkeypatch):
    def boom(method, **params):
        raise RuntimeError("bot is not a member of the chat")

    monkeypatch.setattr(bot_adapter, "_api", boom)
    out = bot_adapter.admin_create_invite_link(_tg_chat())
    assert not out["ok"] and out["invite"] == ""
    assert "not a member" in out["error"]


# ── copy a message ─────────────────────────────────────────

def test_copy_message_payload(bot_adapter):
    res = bot_adapter.copy_message(
        _tg_chat("-100999"), _tg_chat(), "55")
    assert res.ok and res.message_id == "777"
    assert bot_adapter.calls == [("copyMessage", {
        "chat_id": -100999,
        "from_chat_id": -1001234567890,
        "message_id": 55})]


def test_copy_message_caption_replaces(bot_adapter):
    res = bot_adapter.copy_message(
        _tg_chat(), _tg_chat("-100999"), "7", caption="Fresh words")
    assert res.ok
    assert bot_adapter.calls[0][1]["caption"] == "Fresh words"
    # a long caption is capped at the Bot API's 1024
    res = bot_adapter.copy_message(
        _tg_chat(), _tg_chat(), "7", caption="y" * 2000)
    assert res.ok
    assert len(bot_adapter.calls[-1][1]["caption"]) == 1024


def test_copy_message_failures_are_honest(bot_adapter, monkeypatch):
    res = bot_adapter.copy_message(_tg_chat(), _tg_chat(), "abc")
    assert not res.ok and res.error  # int() refusal, never a crash

    def boom(method, **params):
        raise RuntimeError("message to copy not found")

    monkeypatch.setattr(bot_adapter, "_api", boom)
    res = bot_adapter.copy_message(_tg_chat(), _tg_chat(), "55")
    assert not res.ok and res.error == "message to copy not found"


# ── forum topic create ─────────────────────────────────────

def test_forum_topic_create_payload(bot_adapter):
    out = bot_adapter.admin_forum_topic_create(_tg_chat(), "Announcements")
    assert bot_adapter.calls == [("createForumTopic", {
        "chat_id": -1001234567890, "name": "Announcements"})]
    assert out == {"ok": True, "thread_id": "42",
                   "name": "Announcements"}


def test_forum_topic_create_icon_options(bot_adapter):
    bot_adapter.admin_forum_topic_create(
        _tg_chat(), "Art", icon_color=7322096,
        icon_custom_emoji_id="5312536423851630001")
    assert bot_adapter.calls == [("createForumTopic", {
        "chat_id": -1001234567890, "name": "Art",
        "icon_color": 7322096,
        "icon_custom_emoji_id": "5312536423851630001"})]


def test_forum_topic_create_refusals(bot_adapter, monkeypatch):
    out = bot_adapter.admin_forum_topic_create(_tg_chat(), "   ")
    assert not out["ok"] and "needs a name" in out["error"]
    assert bot_adapter.calls == []

    def boom(method, **params):
        raise RuntimeError("topics are not enabled in this chat")

    monkeypatch.setattr(bot_adapter, "_api", boom)
    out = bot_adapter.admin_forum_topic_create(_tg_chat(), "General")
    assert not out["ok"] and "not enabled" in out["error"]


# ── tool layer: the brain's surface ────────────────────────

class _FakeGateway:
    def __init__(self, adapters):
        self._adapters = adapters

    def _snapshot_adapters(self):
        return self._adapters


@pytest.fixture()
def tool_registry(bot_adapter):
    from nomorals.tools import social as soc_mod
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    soc_mod.register(reg)
    old = soc_mod.get_gateway()
    soc_mod.set_gateway(_FakeGateway({"telegram-bot": bot_adapter}))
    yield reg, soc_mod, bot_adapter
    soc_mod.set_gateway(old)


def _call(reg, tool_name, **kwargs):
    return reg._tools[tool_name].fn(**kwargs)


def test_tools_registered(tool_registry):
    reg, _, _ = tool_registry
    for name in ("tgbot_set_chat_photo", "tgbot_create_invite_link",
                 "tgbot_edit_invite_link", "tgbot_revoke_invite_link",
                 "tgbot_copy_message", "tgbot_forum_topic_create"):
        assert name in reg._tools, f"tool not registered: {name}"
        assert reg._tools[name].capability  # capability-gated, like siblings


def test_tool_create_invite_link_round_trip(tool_registry):
    reg, _, ad = tool_registry
    out = _call(reg, "tgbot_create_invite_link",
                chat_key="telegram-bot:-1001234567890",
                name="Via tool", member_limit=3)
    assert out["ok"] and out["invite"] == "https://t.me/+AbCdEfGh"
    assert ad.calls == [("createChatInviteLink", {
        "chat_id": -1001234567890, "name": "Via tool",
        "member_limit": 3})]


def test_tool_set_photo_round_trip(tool_registry, tmp_path):
    reg, _, ad = tool_registry
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    out = _call(reg, "tgbot_set_chat_photo",
                chat_key="telegram-bot:-1001234567890",
                photo_path=str(photo))
    assert out["ok"] is True
    assert ad.uploads and ad.uploads[0][0] == "setChatPhoto"


def test_tool_copy_message_round_trip(tool_registry):
    reg, _, ad = tool_registry
    out = _call(reg, "tgbot_copy_message",
                chat_key="telegram-bot:-100999",
                from_chat_key="telegram-bot:-1001234567890",
                message_id="55")
    assert out == {"ok": True, "message_id": "777", "error": ""}
    assert ad.calls[0][0] == "copyMessage"


def test_tool_forum_topic_create_round_trip(tool_registry):
    reg, _, ad = tool_registry
    out = _call(reg, "tgbot_forum_topic_create",
                chat_key="telegram-bot:-1001234567890", name="Mods")
    assert out["ok"] and out["thread_id"] == "42"


def test_tool_wrong_platform_key_refused(tool_registry):
    reg, _, ad = tool_registry
    out = _call(reg, "tgbot_create_invite_link",
                chat_key="telegram:-1001234567890")
    assert not out["ok"] and "telegram-bot" in out["error"]
    out = _call(reg, "tgbot_copy_message",
                chat_key="telegram-bot:-100999",
                from_chat_key="whatsapp:123", message_id="1")
    assert not out["ok"]
    assert ad.calls == []


def test_tool_no_bot_connected(tmp_path):
    from nomorals.tools import social as soc_mod
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    soc_mod.register(reg)
    old = soc_mod.get_gateway()
    soc_mod.set_gateway(_FakeGateway({}))
    try:
        out = reg._tools["tgbot_create_invite_link"].fn(
            chat_key="telegram-bot:-1001")
        assert not out["ok"] and "not connected" in out["error"]
    finally:
        soc_mod.set_gateway(old)


def test_new_verbs_are_bot_surface_only():
    """The personal (userbot) adapter does NOT grow these names:
    its own shapes already cover the family (admin_forum_topic
    creates over MTProto; admin_invite_link exports the primary
    link) — the census rows were bot rows, and a same-named
    userbot stub would misstate what MTProto can do."""
    from nomorals.social.chat.telegram import TelegramAdapter

    for name in ("admin_set_photo", "admin_create_invite_link",
                 "admin_edit_invite_link", "admin_revoke_invite_link",
                 "admin_forum_topic_create", "copy_message"):
        assert not hasattr(TelegramAdapter, name), name
    assert hasattr(TelegramAdapter, "admin_forum_topic")
    assert hasattr(TelegramAdapter, "admin_invite_link")


def test_photo_pathlib_anchor():
    # the adapter module resolves photo paths through pathlib,
    # the same Path the upload helper takes
    import nomorals.social.chat.telegram as tg

    assert tg.Path is Path
