"""Member resolution (DESIGN §29) — the fix for the owner's
live failures of 2026-11-03: ``/mute @Shanks_371`` answered
"who should I mute?" (the @-form resolved nothing), and
``/ban`` on a message sent before Devon deployed dead-ended
the same way, because targeting only knew local records.

Pins: every resolution path (reply report, text-mention
entity, ledger, bot admin list, userbot participants,
userbot username registry, userbot message fetch, WhatsApp
roster, Discord member list), the fall-through ORDER, the
write-back that teaches the ledger, honest exhaustion (one
line naming what was tried — never the who-prompt), the
channel-post outcome, target-first duration grammar, and
clean degradation when the userbot is absent or broken.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.social.chat.member_resolution import MemberResolver

KEY = "telegram-bot:-1001"


# ── fakes ─────────────────────────────────────────────────────


class _BotAdapter:
    """The telegram-bot surface: admin calls + admin list."""

    name = "telegram-bot"

    def __init__(self, *, administrators=()):
        self.calls: list = []
        self._administrators = list(administrators)

    def _out(self, name, *args, **kw):
        self.calls.append((name, args, kw))
        return {"ok": True}

    def admin_member(self, chat, user_id):
        return {"status": "administrator"}

    def admin_administrators(self, chat):
        return {"ok": True, "administrators": self._administrators}

    def admin_restrict(self, chat, user_id, *, minutes=60):
        return self._out("restrict", user_id, minutes=minutes)

    def admin_unrestrict(self, chat, user_id):
        return self._out("unrestrict", user_id)

    def admin_ban(self, chat, user_id):
        return self._out("ban", user_id)

    def admin_unban(self, chat, user_id):
        return self._out("unban", user_id)

    def admin_delete(self, chat, message_id):
        return self._out("delete", message_id)


class _Userbot:
    """The telegram (MTProto) surface: participants, the
    username registry, and message fetch.  Records its calls
    so fall-through order is assertable."""

    name = "telegram"

    def __init__(self, *, participants=(), by_username=None,
                 senders=None, broken=False):
        self.calls: list = []
        self._participants = list(participants)
        self._by_username = dict(by_username or {})
        self._senders = dict(senders or {})
        self._broken = broken

    def admin_members(self, chat, limit=100):
        self.calls.append(("admin_members", limit))
        if self._broken:
            raise RuntimeError("userbot down")
        return {"ok": True, "members": self._participants,
                "count": len(self._participants)}

    def member_by_username(self, chat, username):
        self.calls.append(("member_by_username", username))
        if self._broken:
            raise RuntimeError("userbot down")
        return self._by_username.get(username)

    def message_sender(self, chat, message_id):
        self.calls.append(("message_sender", message_id))
        if self._broken:
            raise RuntimeError("userbot down")
        return self._senders.get(str(message_id))


class _WhatsApp:
    name = "whatsapp"

    def __init__(self, participants=()):
        self._participants = list(participants)

    def group_participants(self, chat):
        return self._participants


class _Discord:
    name = "discord"

    def __init__(self, members=()):
        self._members = list(members)

    def members(self, guild="", limit=50):
        return self._members


def _msg(*, sender_id="admin1", text="", reply_to="",
         reply_sender=None, reply_sender_chat=None, envelope=None,
         platform="telegram-bot", key=KEY):
    meta = {}
    if reply_sender:
        meta["reply_to_sender"] = reply_sender
    if reply_sender_chat:
        meta["reply_to_sender_chat"] = reply_sender_chat
    if envelope:
        meta["envelope"] = envelope
    chat_id = key.split(":", 1)[1] if ":" in key else ""
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind="group", title="Builders",
                             chat_id=chat_id),
        sender_id=sender_id, sender="Admin", sender_username="boss",
        text=text, meta=meta, reply_to=reply_to,
        message_id="900")


def _rig(tmp_path: Path, *, bot=None, userbot=None):
    from nomorals.agents.partner.runtime_group import \
        RuntimeGroupMixin

    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    bot = bot if bot is not None else _BotAdapter()
    adapters = {"telegram-bot": bot, "telegram": userbot}
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = (  # type: ignore[method-assign]
        lambda platform: adapters.get(platform))
    return mixin, surface, bot, userbot


# ── the live defects ──────────────────────────────────────────


def test_mute_at_username_via_userbot_participants(tmp_path):
    """Defect 1: /mute @Shanks_371 2h with an EMPTY ledger —
    the userbot's participant list names him, ID and all."""
    userbot = _Userbot(participants=[
        {"id": "4242", "username": "Shanks_371", "name": "Shanks",
         "bot": False}])
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    out = mixin._control_mute("@Shanks_371 2h", _msg())
    assert bot.calls[0][0] == "restrict"
    assert bot.calls[0][1][0] == "4242"
    assert bot.calls[0][2]["minutes"] == 120
    assert "@Shanks_371" in out


def test_ban_predeployment_reply_via_userbot_history(tmp_path):
    """Defect 2: /ban replying to a message from before Devon
    deployed — no reply payload, empty ledger; the userbot
    fetches the message and names its author."""
    userbot = _Userbot(senders={
        "49": {"id": "555", "username": "spammy", "name": "Spam",
               "kind": "user"}})
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    out = mixin._control_ban("", _msg(reply_to="49"))
    assert bot.calls[0] == ("ban", ("555",), {})
    assert "@spammy" in out
    # …and the ledger learned him (write-back, path 6).
    member = surface.member(KEY, "555")
    assert member is not None
    assert member["username"] == "spammy"
    assert member["resolved_via"] == "userbot history"


def test_writeback_teaches_the_ledger(tmp_path):
    userbot = _Userbot(participants=[
        {"id": "4242", "username": "Shanks_371", "name": "Shanks",
         "bot": False}])
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    mixin._control_mute("@Shanks_371 2h", _msg())
    member = surface.member(KEY, "4242")
    assert member is not None
    assert member["resolved_via"] == "userbot members"
    # Second time, the ledger answers — the userbot rests.
    userbot.calls.clear()
    mixin._control_mute("@Shanks_371 2h", _msg())
    assert userbot.calls == []


def test_registry_resolution_and_its_writeback(tmp_path):
    """A handle not in the participant page still resolves
    through the username registry (a pre-ban-able handle)."""
    userbot = _Userbot(by_username={
        "ghosty": {"id": "888", "username": "ghosty",
                   "name": "Ghost", "bot": False}})
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    out = mixin._control_ban("@ghosty", _msg())
    assert bot.calls[0] == ("ban", ("888",), {})
    member = surface.member(KEY, "888")
    assert member["resolved_via"] == "userbot resolve"


# ── fall-through order ────────────────────────────────────────


def test_ledger_beats_userbot(tmp_path):
    userbot = _Userbot(participants=[
        {"id": "9999", "username": "loud", "name": "Impostor",
         "bot": False}])
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    surface.handle_message(KEY, "777", username="loud", text="hi")
    mixin._control_mute("@loud 10m", _msg())
    assert bot.calls[0][1][0] == "777"
    assert userbot.calls == []


def test_bot_admin_list_beats_userbot(tmp_path):
    bot = _BotAdapter(administrators=[
        {"id": "900", "name": "Second Admin", "username": "second",
         "status": "administrator"}])
    userbot = _Userbot(participants=[
        {"id": "901", "username": "second", "name": "Not Admin",
         "bot": False}])
    mixin, surface, bot, _ = _rig(tmp_path, bot=bot,
                                  userbot=userbot)
    mixin._control_warn("@second spamming", _msg())
    audit = surface.audit(KEY, limit=5)
    assert any(e["kind"] == "admin_action" or
               e.get("user_id") == "900" for e in audit)
    member = surface.member(KEY, "900")
    assert member is not None
    assert member["resolved_via"] == "bot admins"
    assert userbot.calls == []


def test_reply_report_beats_everything(tmp_path):
    userbot = _Userbot(participants=[
        {"id": "9999", "username": "spammy", "name": "Impostor",
         "bot": False}])
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    msg = _msg(reply_sender={"id": "555", "username": "spammy",
                             "name": "Spam"})
    mixin._control_mute("2h", msg)
    assert bot.calls[0][1][0] == "555"
    assert userbot.calls == []


def test_text_mention_entity_resolves(tmp_path):
    """The picker path: the platform attached the user object
    to the command message itself (path 2)."""
    userbot = _Userbot()
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    text = "/mute Loud 2h"
    envelope = {"entities": [
        {"type": "text_mention", "offset": 6, "length": 4,
         "user_id": "313"}]}
    out = mixin._control_mute(
        "Loud 2h", _msg(text=text, envelope=envelope))
    assert bot.calls[0][0] == "restrict"
    assert bot.calls[0][1][0] == "313"
    assert bot.calls[0][2]["minutes"] == 120


# ── channel posts + anonymous ─────────────────────────────────


def test_reply_to_channel_post_names_the_fact(tmp_path):
    userbot = _Userbot(senders={
        "49": {"id": "-100777", "username": "", "name": "Gold News",
               "kind": "channel"}})
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    out = mixin._control_ban("", _msg(reply_to="49"))
    assert "posted as Gold News" in out
    assert "antichannel" in out
    assert bot.calls == []


def test_reply_sender_chat_meta_is_a_channel_post(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    msg = _msg(reply_sender_chat={"id": "-100777",
                                  "title": "Gold News"})
    out = mixin._control_mute("2h", msg)
    assert "posted as Gold News" in out
    assert bot.calls == []


def test_anonymous_admin_reply_is_not_a_member(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    msg = _msg(reply_sender={"id": "1087968824",
                             "username": "GroupAnonymousBot",
                             "name": "Group"})
    out = mixin._control_ban("", msg)
    assert "anonymous admin" in out
    assert bot.calls == []


# ── exhaustion honesty ────────────────────────────────────────


def test_exhaustion_names_what_was_tried(tmp_path):
    userbot = _Userbot()
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    out = mixin._control_mute("@ghost 2h", _msg())
    assert "I couldn't find @ghost" in out
    assert "member ledger" in out
    assert "member list (userbot)" in out
    assert "who should I" not in out
    assert bot.calls == []


def test_userbot_absent_still_answers_honestly(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=None)
    out = mixin._control_mute("@ghost 2h", _msg())
    assert "I couldn't find @ghost" in out
    assert "userbot" not in out  # never claims a path it lacks
    assert bot.calls == []


def test_userbot_broken_degrades_to_exhaustion(tmp_path):
    userbot = _Userbot(broken=True)
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    out = mixin._control_ban("@ghost", _msg())
    assert "I couldn't find @ghost" in out
    assert bot.calls == []


def test_duration_token_is_never_a_name(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    out = mixin._control_mute("2h", _msg())
    assert "who should I mute" in out
    assert "couldn't find" not in out


# ── target first, then duration ───────────────────────────────


def test_named_target_without_duration_asks_for_it(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    surface.handle_message(KEY, "777", username="loud", text="hi")
    out = mixin._control_mute("@loud", _msg())
    assert "for how long should I mute @loud" in out
    assert bot.calls == []


def test_group_default_mute_duration_applies(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    surface.handle_message(KEY, "777", username="loud", text="hi")
    surface.update_settings(KEY, mute_default_s=7200)
    mixin._control_mute("@loud", _msg())
    assert bot.calls[0][2]["minutes"] == 120


def test_gsettings_mute_default_grammar(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    out = mixin._control_gsettings("mute default 2h", _msg())
    assert "standing mute length 2h" in out
    assert surface.store.settings(KEY)["mute_default_s"] == 7200
    out = mixin._control_gsettings("mute default off", _msg())
    assert surface.store.settings(KEY)["mute_default_s"] == 0


def test_reply_target_without_duration_keeps_standing_hour(
        tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    msg = _msg(reply_sender={"id": "555", "username": "spammy",
                             "name": "Spam"})
    mixin._control_mute("", msg)
    assert bot.calls[0][2]["minutes"] == 60


# ── consumers ─────────────────────────────────────────────────


def test_pay_miss_uses_the_exhaustion_line(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    out = mixin._control_pay("@ghost 50", _msg())
    assert "I couldn't find @ghost" in out
    assert "ledger." not in out.split("—")[0]


def test_member_card_via_userbot_registry(tmp_path):
    userbot = _Userbot(by_username={
        "newface": {"id": "606", "username": "newface",
                    "name": "New Face", "bot": False}})
    mixin, surface, bot, _ = _rig(tmp_path, userbot=userbot)
    out = mixin._control_member("@newface", _msg())
    assert "New Face" in out
    assert surface.member(KEY, "606") is not None


def test_rob_reply_uses_the_real_author_id(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    msg = _msg(reply_sender={"id": "555", "username": "spammy",
                             "name": "Spam"})
    assert mixin._rob_target(surface, KEY, "", msg) == "555"


def test_raw_id_unban_without_writeback(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    out = mixin._control_unban("123456", _msg())
    assert bot.calls[0] == ("unban", ("123456",), {})
    assert surface.member(KEY, "123456") is None


def test_strikes_named_miss_is_the_exhaustion_line(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    out = mixin._control_strikes("@ghost", _msg())
    assert "I couldn't find @ghost" in out


def test_report_on_channel_post_files_the_fact(tmp_path):
    mixin, surface, bot, _ = _rig(tmp_path, userbot=_Userbot())
    msg = _msg(sender_id="member9",
               reply_sender_chat={"id": "-100777",
                                  "title": "Gold News"})
    out = mixin._control_report("spam", msg)
    assert "channel post" in out
    entries = surface.audit(KEY, limit=5)
    assert any(e["kind"] == "report"
               and e.get("target_label") == "Gold News"
               for e in entries)


# ── other platforms (organ level) ─────────────────────────────


def test_whatsapp_roster_path(tmp_path):
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    wa = _WhatsApp(participants=[
        {"jid": "2348012345678@s.whatsapp.net", "name": "Amina",
         "role": ""}])
    resolver = MemberResolver(
        surface, lambda p: wa if p == "whatsapp" else None)
    key = "whatsapp:120363001@g.us"
    msg = _msg(platform="whatsapp", key=key)
    res = resolver.resolve_token(key, msg, "Amina")
    assert res.found
    assert res.member.user_id == "2348012345678@s.whatsapp.net"
    assert res.member.source == "whatsapp roster"
    res2 = resolver.resolve_token(key, msg, "2348012345678")
    assert res2.found
    assert res2.member.user_id == "2348012345678@s.whatsapp.net"


def test_discord_member_path(tmp_path):
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    dc = _Discord(members=[
        {"id": "777001", "name": "Kai", "tag": "kai#0001",
         "joined": ""}])
    resolver = MemberResolver(
        surface, lambda p: dc if p == "discord" else None)
    key = "discord:555444"
    msg = _msg(platform="discord", key=key)
    res = resolver.resolve_token(key, msg, "Kai")
    assert res.found
    assert res.member.user_id == "777001"
    assert res.member.source == "discord members"


# ── the Bot API converter marks channel posts ─────────────────


def test_bot_convert_marks_replied_channel_post(tmp_path):
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
            "text": "/ban",
            "reply_to_message": {
                "message_id": 49,
                "sender_chat": {"id": -100777, "type": "channel",
                                "title": "Gold News"},
                "text": "Hiring now",
            },
        },
    }
    msg = adapter._convert(update)
    assert msg is not None
    assert "reply_to_sender" not in msg.meta
    assert msg.meta["reply_to_sender_chat"]["title"] == "Gold News"
