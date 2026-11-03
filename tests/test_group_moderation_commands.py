"""Manual moderation commands (DESIGN §4) — the set the owner
hit live on 2026-11-03: he replied to a spammer with /mute and
Devon answered conversationally, because the commands did not
exist.  Pins: reply-targeted and @-targeted mute/ban/kick/warn/
unban/unmute, pin/purge on the replied message, the warn
escalation ladder, non-admin refusal, and platform refusals
reported as refusals (never as success).  Plus the adapter
change that makes reply-targeting possible at all: the Bot API
converter now surfaces the replied-to message's AUTHOR.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001"


class _Adapter:
    def __init__(self, *, admin: bool = True, refuse: bool = False):
        self.calls: list = []
        self._admin = admin
        self._refuse = refuse

    def _out(self, name, *args, **kw):
        self.calls.append((name, args, kw))
        return {"ok": not self._refuse,
                "error": "nope" if self._refuse else ""}

    def admin_member(self, chat, user_id):
        return {"status": "administrator" if self._admin
                else "member"}

    def admin_restrict(self, chat, user_id, *, minutes=60):
        return self._out("restrict", user_id, minutes=minutes)

    def admin_unrestrict(self, chat, user_id):
        return self._out("unrestrict", user_id)

    def admin_ban(self, chat, user_id):
        return self._out("ban", user_id)

    def admin_unban(self, chat, user_id):
        return self._out("unban", user_id)

    def admin_pin(self, chat, message_id):
        return self._out("pin", message_id)

    def admin_unpin(self, chat):
        return self._out("unpin")

    def admin_delete(self, chat, message_id):
        return self._out("delete", message_id)


def _msg(*, sender_id="admin1", tail_text="", reply_to="",
         reply_sender=None):
    meta = {}
    if reply_sender:
        meta["reply_to_sender"] = reply_sender
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform="telegram-bot",
                             kind="group", title="Builders"),
        sender_id=sender_id, sender="Admin", sender_username="boss",
        text=tail_text, meta=meta, reply_to=reply_to,
        message_id="900")


def _rig(tmp_path: Path, **adapter_kw):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    adapter = _Adapter(**adapter_kw)
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    return mixin, surface, adapter


def test_mute_by_reply_uses_replied_author(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    msg = _msg(reply_sender={"id": "555", "username": "spammy",
                             "name": "Spam"})
    out = mixin._control_mute("2h", msg)
    assert adapter.calls[0][0] == "restrict"
    assert adapter.calls[0][1][0] == "555"
    assert adapter.calls[0][2]["minutes"] == 120
    assert "@spammy" in out and "2h" in out
    audit = surface.audit(KEY, limit=5)
    assert any(e["kind"] == "admin_action" and e["action"] == "mute"
               for e in audit)


def test_mute_by_username_from_ledger(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    surface.handle_message(KEY, "777", username="loud", text="hi")
    out = mixin._control_mute("@loud 10m", _msg())
    assert adapter.calls[0][1][0] == "777"
    assert adapter.calls[0][2]["minutes"] == 10
    assert "10m" in out


def test_non_admin_is_refused_without_adapter_calls(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, admin=False)
    out = mixin._control_ban("@whoever", _msg())
    assert "admin" in out
    assert adapter.calls == []


def test_warn_ladder_mutes_at_threshold_and_resets(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    surface.handle_message(KEY, "777", username="loud", text="hi")
    msg = _msg()
    assert "1/3" in mixin._control_warn("@loud caps", msg)
    assert "2/3" in mixin._control_warn("@loud caps", msg)
    out = mixin._control_warn("@loud caps", msg)
    assert "muted for an hour" in out
    assert any(c[0] == "restrict" for c in adapter.calls)
    member = surface.member(KEY, "777")
    assert member["warn_count"] == 0


def test_kick_is_ban_then_unban(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    msg = _msg(reply_sender={"id": "555", "username": "spammy",
                             "name": "Spam"})
    out = mixin._control_kick("", msg)
    assert [c[0] for c in adapter.calls] == ["ban", "unban"]
    assert "can rejoin" in out


def test_purge_deletes_the_replied_message(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    surface.handle_message(KEY, "777", text="spam spam",
                           message_id="31337")
    out = mixin._control_purge("", _msg(reply_to="31337"))
    assert adapter.calls[0] == ("delete", ("31337",), {})
    assert "gone" in out


def test_platform_refusal_is_reported_as_refusal(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, refuse=True)
    msg = _msg(reply_sender={"id": "555", "username": "spammy",
                             "name": "Spam"})
    out = mixin._control_mute("", msg)
    assert "didn't go through" in out
    audit = surface.audit(KEY, limit=5)
    assert any(e["kind"] == "admin_action_failed" for e in audit)


def test_word_filter_deletes_and_regex_works(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    surface.update_settings(
        KEY, moderation={
            "flood_max": 6, "flood_window_s": 10.0,
            "repeat_threshold": 3, "link_allowlist": [],
            "word_filters": ["free money", r"/\b\ddays?\b/"],
            "filter_action": "delete"})
    hit = surface.handle_message(KEY, "u1", text="get FREE MONEY now")
    assert any(a["rule"] == "word_filter" and a["kind"] == "delete"
               for a in hit["actions"])
    hit2 = surface.handle_message(KEY, "u1", text="only 3days left")
    assert any(a["rule"] == "word_filter" for a in hit2["actions"])
    clean = surface.handle_message(KEY, "u1", text="hello friends")
    assert not any(a["rule"] == "word_filter"
                   for a in clean["actions"])


def test_report_files_into_the_held_log(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    msg = _msg(sender_id="member9",
               reply_sender={"id": "555", "username": "spammy",
                             "name": "Spam"})
    out = mixin._control_report("hiring spam", msg)
    assert "@spammy" in out
    entries = surface.audit(KEY, limit=5)
    assert any(e["kind"] == "report" and e["user_id"] == "555"
               and e["by"] == "member9" for e in entries)


def test_bot_api_convert_surfaces_replied_author(tmp_path):
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
        "update_id": 9,
        "message": {
            "message_id": 50, "date": int(time.time()),
            "chat": {"id": -100123, "type": "supergroup",
                     "title": "Builders"},
            "from": {"id": 1, "first_name": "Boss",
                     "username": "boss"},
            "text": "/mute",
            "reply_to_message": {
                "message_id": 49,
                "from": {"id": 555, "first_name": "Spam",
                         "username": "spammy"},
                "text": "Hiring now",
            },
        },
    }
    msg = adapter._convert(update)
    assert msg is not None
    assert msg.reply_to == "49"
    assert msg.meta["reply_to_sender"]["id"] == "555"
    assert msg.meta["reply_to_sender"]["username"] == "spammy"
