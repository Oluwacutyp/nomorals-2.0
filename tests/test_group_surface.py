"""P1 group surface (DESIGN §1): the wiring that claims group chats.

Pins: persisted per-group settings + member ledger with identity
change detection; welcome/goodbye from the group's own settings;
the community Moderator running on live group traffic; adapter
service-event mapping (Bot API joins/leaves); the runtime turn —
service events consumed with a welcome, moderation actions
executed through the adapter's admin surface or audited as
unexecuted, never faked.
"""
from __future__ import annotations

import time

import pytest

from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupStoreError,
    GroupSurface,
)

KEY = "telegram:-100123"


def _surface(tmp_path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "groups.json").load())


def test_store_round_trip_and_corrupt_quarantine(tmp_path):
    path = tmp_path / "groups.json"
    surface = GroupSurface(GroupStore(path).load())
    surface.record_join(KEY, "7", username="ada", name="Ada")
    again = GroupSurface(GroupStore(path).load())
    member = again.member(KEY, "7")
    assert member is not None and member["username"] == "ada"

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(GroupStoreError):
        GroupStore(path).load()
    assert not path.exists()  # quarantined aside, not overwritten
    assert list(tmp_path.glob("groups.corrupt-*"))


def test_join_welcomes_with_group_rules(tmp_path):
    surface = _surface(tmp_path)
    surface.update_settings(KEY, rules=["No spam", "Be kind"])
    text = surface.record_join(KEY, "7", username="ada", name="Ada",
                               group_name=" builders")
    assert "Ada" in text and "No spam" in text
    surface.update_settings(KEY, welcome_enabled=False)
    assert surface.record_join(KEY, "8", name="Bo",
                               group_name="builders") == ""


def test_identity_changes_detected(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(KEY, "7", username="ada", name="Ada",
                           text="hi", ts=1000.0)
    out = surface.handle_message(KEY, "7", username="ada_ng",
                                 name="Ada N.", text="yo", ts=1001.0)
    kinds = {(c["kind"], c["old"], c["new"]) for c in out["changes"]}
    assert ("username_changed", "ada", "ada_ng") in kinds
    assert ("name_changed", "Ada", "Ada N.") in kinds
    member = surface.member(KEY, "7")
    assert member["message_count"] == 2
    assert member["username"] == "ada_ng"
    # unchanged identity: no events
    out = surface.handle_message(KEY, "7", username="ada_ng",
                                 name="Ada N.", text="ok", ts=1002.0)
    assert out["changes"] == []


def test_moderation_flood_repeat_and_links(tmp_path):
    surface = _surface(tmp_path)
    # flood: 7 messages inside the window (default max 6/10s)
    actions = []
    for i in range(7):
        out = surface.handle_message(
            KEY, "9", text=f"m{i}", ts=2000.0 + i * 0.5)
        actions = out["actions"]
    assert actions and actions[0]["kind"] == "mute"
    assert actions[0]["rule"] == "flood"
    # repeats: warn first, then delete
    first = surface.handle_message(KEY, "10", text="same",
                                   ts=3000.0)
    surface.handle_message(KEY, "10", text="same", ts=3001.0)
    third = surface.handle_message(KEY, "10", text="same",
                                   ts=3002.0)
    assert first["actions"] == []
    assert third["actions"][0]["kind"] in ("warn", "delete")
    # link allowlist: set one, post another domain
    surface.update_settings(
        KEY, moderation={"flood_max": 6, "flood_window_s": 10.0,
                         "repeat_threshold": 3,
                         "link_allowlist": ["example.com"]})
    surface.refresh_moderator(KEY)
    out = surface.handle_message(KEY, "11",
                                 text="see https://evil.example.net/x",
                                 ts=4000.0)
    assert out["actions"] and out["actions"][0]["kind"] == "delete"
    ok = surface.handle_message(KEY, "11",
                                text="see https://example.com/x",
                                ts=4001.0)
    assert ok["actions"] == []


def test_rules_text_reads_the_enforcers_config(tmp_path):
    surface = _surface(tmp_path)
    surface.update_settings(KEY, rules=["No spam"])
    text = surface.rules_text(KEY)
    assert "No spam" in text and "flood limit 6" in text


# ── runtime turn ──────────────────────────────────────────────────

class _Chat:
    platform = "telegram-bot"
    chat_id = "-100123"
    title = "Builders"
    key = KEY


class _Msg:
    def __init__(self, **kw):
        self.chat = _Chat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")


class _Adapter:
    def __init__(self):
        self.calls = []

    def admin_delete(self, chat, message_id):
        self.calls.append(("delete", message_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_restrict(self, chat, user_id, *, until=0, **kw):
        self.calls.append(("restrict", user_id, until))
        return {"ok": True}


class _Gateway:
    def __init__(self, adapter):
        self.adapters = {"telegram-bot": adapter}


from nomorals.agents.partner.runtime_group import RuntimeGroupMixin


class _Runtime(RuntimeGroupMixin):
    """RuntimeGroupMixin on a stub: send + gateway only."""

    def __init__(self, tmp_path, adapter):
        self.sent = []
        self.gateway = _Gateway(adapter)
        self.context = type("C", (), {
            "settings": type("S", (), {
                "workspace_dir": str(tmp_path)})()})()

    def _send(self, chat, text):
        self.sent.append(text)

    def turn(self, msg):
        return self._group_surface_turn(msg)


def test_runtime_join_consumed_with_welcome(tmp_path):
    rt = _Runtime(tmp_path, _Adapter())
    msg = _Msg(meta={"service": "join", "new_members": [
        {"id": "7", "username": "ada", "name": "Ada"}]})
    assert rt.turn(msg) is True
    assert rt.sent and "Ada" in rt.sent[0]


def test_runtime_link_delete_executes_and_consumes(tmp_path):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, adapter)
    # seed the allowlist through the surface the runtime will build
    from nomorals.social.chat.group_surface import default_surface

    surface = default_surface(str(tmp_path))
    surface.update_settings(
        KEY, moderation={"flood_max": 6, "flood_window_s": 10.0,
                         "repeat_threshold": 3,
                         "link_allowlist": ["example.com"]})
    msg = _Msg(text="https://evil.example.net/x", message_id="99")
    assert rt.turn(msg) is True
    assert ("delete", "99") in adapter.calls


def test_runtime_unexecutable_action_is_audited_not_faked(tmp_path):
    class _Bare:
        adapters = {}

    rt = _Runtime(tmp_path, None)
    rt.gateway = _Bare()
    from nomorals.social.chat.group_surface import default_surface

    surface = default_surface(str(tmp_path))
    surface.update_settings(
        KEY, moderation={"flood_max": 6, "flood_window_s": 10.0,
                         "repeat_threshold": 3,
                         "link_allowlist": ["example.com"]})
    msg = _Msg(text="https://evil.example.net/x")
    # no admin surface on the adapter: not consumed, but audited
    assert rt.turn(msg) is False
    surface2 = default_surface(str(tmp_path))
    kinds = [a["kind"] for a in surface2.audit(KEY)]
    assert "moderation_unexecuted" in kinds


def test_bot_api_convert_maps_join(tmp_path):
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = object.__new__(TelegramBotAdapter)
    adapter.name = "telegram-bot"
    adapter.owner_only = False
    adapter.chat_allow = set()
    adapter._bot_username = "DevonBot"
    adapter._last_inbound_was_command = {}
    adapter._last_command = {}
    adapter.partner_context = ""
    adapter.media_dir = str(tmp_path)
    update = {
        "update_id": 1,
        "message": {
            "message_id": 5, "date": int(time.time()),
            "chat": {"id": -100123, "type": "supergroup",
                     "title": "Builders"},
            "from": {"id": 7, "first_name": "Ada",
                     "username": "ada"},
            "new_chat_members": [
                {"id": 7, "first_name": "Ada", "username": "ada"}],
        },
    }
    msg = adapter._convert(update)
    assert msg is not None
    assert msg.meta.get("service") == "join"
    assert msg.meta["new_members"][0]["id"] == "7"
