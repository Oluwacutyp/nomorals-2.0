"""The §16 P5 layer (census gap #9): member tags, sender-chat
ban, the join-request DM window + "queue" verdict, reaction
wipe, the forum-topic auto-rename duty, ephemeral answers,
per-member command scopes, boost intel — plus the
platform-truthful acceptance explainer.

Pins, per item:
* tags obey Telegram's law (0–16 chars, no emoji) before any
  platform call; the native set is the bot's setChatMemberTag;
  the ledger records what the platform accepted, and on the
  userbot surface the ledger carries it with the truth stated;
* a sender-chat ban hits the CHANNEL identity from the
  replied post — never the post's nominal sender, never a
  member;
* a held join request carrying a guard-bot query is answered
  "queue" (defer to other admins); /joinq ask writes through
  the 5-minute DM window and honestly reports it closed;
* the topic duty renames only auto-named topics, only when
  switched on, only in claimed rooms, and closes either way;
* ephemeral /balance goes to the asker only (adapter verb →
  DM → pointer), never silently public;
* scope sync publishes administrators + per-member scopes on
  the bot surface and reports partial application as partial;
* boost intel attributes exactly what Telegram's source
  named — unclaimed giveaways count for the room, for nobody;
* the P5 kinds are absent from WhatsApp's menu and get the
  platform truth when typed there.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_surface import GroupStore, GroupSurface

BOTKEY = "telegram-bot:-1002"
UBKEY = "telegram:-1001"
NOW = time.time()


# ── pure organ ───────────────────────────────────────────────

def test_validate_tag_law():
    from nomorals.social.chat.group_p5 import validate_tag

    assert validate_tag("night owl") == (True, "night owl", "")
    assert validate_tag("")[0:2] == (True, "")  # empty clears
    ok, _c, why = validate_tag("x" * 17)
    assert not ok and "16" in why
    ok, _c, why = validate_tag("party 🎉 crew")
    assert not ok and "emoji" in why
    ok, _c, why = validate_tag("@handle")
    assert not ok
    # CJK is letters, not emoji.
    assert validate_tag("夜猫子")[0]


def test_derive_topic_name():
    from nomorals.social.chat.group_p5 import derive_topic_name

    assert derive_topic_name(
        "Deploy window moves to Friday night, plan inside") \
        .startswith("Deploy window")
    assert derive_topic_name("https://example.com/only-a-link") == ""
    assert derive_topic_name("ok") == ""
    assert derive_topic_name("") == ""


def test_boost_source_kind():
    from nomorals.social.chat.group_p5 import boost_source_kind

    assert boost_source_kind({"source": "premium"}) == "premium"
    assert boost_source_kind({"source": "gift_code"}) == "gift_code"
    assert boost_source_kind({"source": "giveaway"}) == "giveaway"
    assert boost_source_kind({}) == "unknown"
    assert boost_source_kind(None) == "unknown"


# ── surface state ────────────────────────────────────────────

def _surface(tmp_path: Path, key=BOTKEY, *, designated=True):
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    if designated:
        surface.set_designated(key, True, by="owner")
    return surface


def test_surface_tag_state(tmp_path):
    surface = _surface(tmp_path)
    # A tag on a stranger is refused by shape.
    out = surface.member_set_tag(BOTKEY, "42", tag="owl")
    assert out["ok"] is False
    surface.note_join(BOTKEY, "42", name="Ada")
    out = surface.member_set_tag(BOTKEY, "42", tag="night owl",
                                 by="admin1")
    assert out["ok"] and out["tag"] == "night owl"
    assert surface.member(BOTKEY, "42")["tag"] == "night owl"
    out = surface.member_set_tag(BOTKEY, "42", tag="")
    assert out["ok"] and out["old"] == "night owl"
    assert surface.member(BOTKEY, "42")["tag"] == ""
    kinds = [a["kind"] for a in surface.audit(BOTKEY, limit=50)]
    assert kinds.count("member_tag") == 2


def test_surface_boost_ledger(tmp_path):
    surface = _surface(tmp_path)
    surface.note_boost(BOTKEY, boost_id="b1", user_id="42",
                       source_kind="premium", add_ts=NOW)
    surface.note_boost(BOTKEY, boost_id="b2", user_id="42",
                       source_kind="gift_code", add_ts=NOW)
    surface.note_boost(BOTKEY, boost_id="b3", user_id="",
                       source_kind="giveaway", add_ts=NOW)
    # A re-delivered update never double counts.
    surface.note_boost(BOTKEY, boost_id="b1", user_id="42",
                       source_kind="premium", add_ts=NOW)
    summary = surface.boost_summary(BOTKEY, "42")
    assert summary["count"] == 2
    assert summary["sources"] == {"premium": 1, "gift_code": 1}
    board = surface.boosters(BOTKEY)
    assert board["total_active"] == 3
    assert board["unattributed"] == 1
    assert [r["user_id"] for r in board["rows"]] == ["42"]
    surface.note_boost_removed(BOTKEY, boost_id="b1")
    assert surface.boost_summary(BOTKEY, "42")["count"] == 1
    assert surface.boosters(BOTKEY)["total_active"] == 2


def test_surface_topic_state(tmp_path):
    surface = _surface(tmp_path)
    surface.topic_note_created(BOTKEY, thread_id="7",
                               title="General", implicit=True)
    assert surface.topic_rename_candidate(BOTKEY, "7") is not None
    surface.topic_note_created(BOTKEY, thread_id="8",
                               title="My Own Name", implicit=False)
    assert surface.topic_rename_candidate(BOTKEY, "8") is None
    surface.topic_mark_renamed(BOTKEY, thread_id="7",
                               title="Deploy Plans")
    assert surface.topic_rename_candidate(BOTKEY, "7") is None


def test_p5_settings_defaults_and_merge(tmp_path):
    surface = _surface(tmp_path)
    p5 = surface.store.settings(BOTKEY)["p5"]
    assert p5 == {"topics_rename": False, "member_scopes": False,
                  "ephemeral": False}
    surface.update_settings(BOTKEY, p5={"topics_rename": True})
    p5 = surface.store.settings(BOTKEY)["p5"]
    assert p5["topics_rename"] is True
    assert p5["ephemeral"] is False  # merge keeps the rest


def test_queue_record_carries_p5_extras(tmp_path):
    surface = _surface(tmp_path)
    surface.update_settings(BOTKEY, acceptance={
        "mode": "request", "auto_approve": False,
        "criteria": {"username": True}})
    out = surface.acceptance_on_request(
        BOTKEY, {"user_id": "42", "username": "", "name": "Ada",
                 "query_id": "q-9", "user_chat_id": "555"})
    record = out["record"]
    assert record["query_id"] == "q-9"
    assert record["user_chat_id"] == "555"


# ── bot adapter verbs (fake _api) ────────────────────────────

def _bot_adapter():
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = TelegramBotAdapter(token="<redacted>", session=object())
    calls: list = []

    def fake_api(method, **params):
        calls.append((method, params))
        if method == "getMe":
            return {"id": 777}
        if method == "getChatMember":
            return {"status": "administrator",
                    "can_manage_tags": True,
                    "can_delete_messages": True}
        if method == "getUserChatBoosts":
            return {"total_count": 1, "boosts": [
                {"boost_id": "b1",
                 "source": {"source": "premium"}}]}
        return True

    adapter._api = fake_api  # type: ignore[method-assign]
    return adapter, calls


def _chat(key=BOTKEY):
    from nomorals.social.chat.base import ChatKind, ChatRef

    return ChatRef(platform="telegram-bot",
                   chat_id=key.split(":", 1)[1], kind=ChatKind.GROUP)


def test_bot_verbs_hit_the_right_api_methods():
    adapter, calls = _bot_adapter()
    chat = _chat()
    out = adapter.admin_set_member_tag(chat, "42", "night owl")
    assert out["ok"]
    method, params = calls[-1]
    assert method == "setChatMemberTag"
    assert params["user_id"] == 42 and params["tag"] == "night owl"

    out = adapter.admin_ban_sender_chat(chat, "-100999")
    assert out["ok"]
    method, params = calls[-1]
    assert method == "banChatSenderChat"
    assert params["sender_chat_id"] == -100999

    adapter.admin_unban_sender_chat(chat, "-100999")
    assert calls[-1][0] == "unbanChatSenderChat"

    adapter.admin_reaction_wipe(chat, "42")
    method, params = calls[-1]
    assert method == "deleteAllMessageReactions"
    assert params["user_id"] == 42

    out = adapter.admin_answer_join_query("q-9", "queue")
    assert out["ok"]
    method, params = calls[-1]
    assert method == "answerChatJoinRequestQuery"
    assert params["chat_join_request_query_id"] == "q-9"
    assert params["result"] == "queue"

    out = adapter.admin_dm_applicant("555", "one question…")
    assert out["ok"]
    method, params = calls[-1]
    assert method == "sendMessage" and params["chat_id"] == 555

    out = adapter.admin_user_boosts(chat, "42")
    assert out["ok"] and out["result"]["total_count"] == 1
    assert calls[-1][0] == "getUserChatBoosts"

    adapter.admin_forum_topic_edit(chat, "7", "Deploy Plans")
    method, params = calls[-1]
    assert method == "editForumTopic"
    assert params["message_thread_id"] == 7
    assert params["name"] == "Deploy Plans"


def test_bot_scope_verbs_and_ephemeral_params():
    import json

    adapter, calls = _bot_adapter()
    chat = _chat()
    adapter.admin_set_member_commands(chat, "42",
                                      [("tag", "set a tag")])
    method, params = calls[-1]
    assert method == "setMyCommands"
    scope = json.loads(params["scope"])
    assert scope == {"type": "chat_member",
                     "chat_id": -1002, "user_id": 42}
    cmds = json.loads(params["commands"])
    assert cmds == [{"command": "tag", "description": "set a tag"}]

    adapter.admin_set_admin_commands(chat, [("tag", "set a tag")])
    scope = json.loads(calls[-1][1]["scope"])
    assert scope["type"] == "chat_administrators"

    adapter.admin_clear_member_commands(chat, "42")
    assert calls[-1][0] == "deleteMyCommands"
    adapter.admin_clear_admin_commands(chat)
    assert calls[-1][0] == "deleteMyCommands"
    assert json.loads(calls[-1][1]["scope"])["type"] == \
        "chat_administrators"

    out = adapter.send_ephemeral(chat, "42", "your balance…")
    assert out["ok"]
    method, params = calls[-1]
    assert method == "sendMessage"
    eph = json.loads(params["ephemeral_message_parameters"])
    assert eph == {"receiver_user_id": 42}


def test_bot_admin_rights_include_manage_tags():
    adapter, _calls = _bot_adapter()
    rights = adapter.admin_rights(_chat())
    assert rights["ok"] and rights["can_manage_tags"] is True


def test_bot_convert_boost_attribution():
    adapter, _calls = _bot_adapter()
    update = {"chat_boost": {
        "chat": {"id": -1002, "type": "supergroup",
                 "title": "Builders"},
        "date": int(NOW),
        "boost": {"boost_id": "b1", "add_date": int(NOW),
                  "expiration_date": int(NOW) + 86400,
                  "source": {"source": "premium",
                             "user": {"id": 42,
                                      "first_name": "Ada",
                                      "username": "ada"}}}}}
    msg = adapter._convert_boost(update["chat_boost"],
                                 removed=False)
    assert msg.meta["service"] == "chat_boost"
    boost = msg.meta["boost"]
    assert boost["user_id"] == "42"
    assert boost["source_kind"] == "premium"
    # An unclaimed giveaway names nobody.
    update2 = {"removed_chat_boost": {
        "chat": {"id": -1002, "type": "supergroup"},
        "date": int(NOW),
        "boost": {"boost_id": "b2",
                  "source": {"source": "giveaway",
                             "giveaway_message_id": 5}}}}
    msg2 = adapter._convert_boost(update2["removed_chat_boost"],
                                  removed=True)
    assert msg2.meta["service"] == "chat_boost_removed"
    assert msg2.meta["boost"]["user_id"] == ""


def test_bot_convert_forum_topic_created():
    adapter, _calls = _bot_adapter()
    update = {"message": {
        "message_id": 7, "date": int(NOW),
        "chat": {"id": -1002, "type": "supergroup",
                 "title": "Builders"},
        "from": {"id": 42, "first_name": "Ada"},
        "message_thread_id": 7,
        "forum_topic_created": {"name": "General",
                                "icon_color": 1,
                                "is_name_implicit": True}}}
    msg = adapter._convert(update)
    assert msg.meta["service"] == "forum_topic_created"
    assert msg.meta["topic"]["implicit"] is True
    assert msg.meta["thread_id"] == "7"


def test_bot_join_request_captures_query_id():
    adapter, _calls = _bot_adapter()
    got: list = []
    adapter._handle_join_request(
        {"chat": {"id": -1002, "type": "supergroup",
                  "title": "Builders"},
         "from": {"id": 42, "first_name": "Ada"},
         "date": int(NOW), "user_chat_id": 555,
         "query_id": "q-9"},
        got.append)
    assert len(got) == 1
    assert got[0].meta["join_query_id"] == "q-9"
    assert got[0].meta["user_chat_id"] == "555"


def test_bot_dispatch_routes_boost_updates():
    adapter, _calls = _bot_adapter()
    got: list = []
    adapter._dispatch_update(
        {"update_id": 1, "chat_boost": {
            "chat": {"id": -1002, "type": "supergroup"},
            "date": int(NOW),
            "boost": {"boost_id": "b1",
                      "source": {"source": "premium",
                                 "user": {"id": 42}}}}},
        got.append)
    assert len(got) == 1
    assert got[0].meta["service"] == "chat_boost"


def test_userbot_botonly_verbs_are_honest():
    from nomorals.social.chat.telegram import TelegramAdapter

    bare = TelegramAdapter.__new__(TelegramAdapter)
    chat = _chat(UBKEY)
    for out in (
            bare.admin_set_member_tag(chat, "42", "owl"),
            bare.admin_reaction_wipe(chat, "42"),
            bare.admin_answer_join_query("q", "queue"),
            bare.admin_dm_applicant("555", "hi"),
            bare.admin_user_boosts(chat, "42"),
            bare.admin_set_member_commands(chat, "42", []),
            bare.send_ephemeral(chat, "42", "hi")):
        assert out["ok"] is False
        assert "unsupported" in out["error"]


# ── runtime commands (mixin rig) ─────────────────────────────

class _BotFake:
    """Bot-surface fake carrying the P5 verbs."""

    def __init__(self, *, tag_refuse=False):
        self.calls: list = []
        self.tag_refuse = tag_refuse
        self.dm_ok = True
        self.admins = [{"id": "admin1", "status": "administrator"},
                       {"id": "mod2", "status": "administrator"}]

    def admin_member(self, chat, user_id):
        return {"ok": True,
                "status": "administrator"
                if str(user_id) == "admin1" else "member"}

    def admin_administrators(self, chat):
        return {"ok": True, "administrators": list(self.admins)}

    def admin_set_member_tag(self, chat, user_id, tag):
        self.calls.append(("set_tag", str(user_id), tag))
        if self.tag_refuse:
            return {"ok": False,
                    "error": "not enough rights to manage tags"}
        return {"ok": True}

    def admin_ban_sender_chat(self, chat, sender_chat_id):
        self.calls.append(("ban_sender", str(sender_chat_id)))
        return {"ok": True}

    def admin_unban_sender_chat(self, chat, sender_chat_id):
        self.calls.append(("unban_sender", str(sender_chat_id)))
        return {"ok": True}

    def admin_reaction_wipe(self, chat, user_id):
        self.calls.append(("wipe", str(user_id)))
        return {"ok": True}

    def admin_user_boosts(self, chat, user_id):
        self.calls.append(("user_boosts", str(user_id)))
        return {"ok": True, "result": {"boosts": [{}, {}]}}

    def admin_set_member_commands(self, chat, user_id, commands):
        self.calls.append(("scope_member", str(user_id),
                           len(commands)))
        return {"ok": True}

    def admin_set_admin_commands(self, chat, commands):
        self.calls.append(("scope_admins", len(commands)))
        return {"ok": True}

    def admin_clear_member_commands(self, chat, user_id):
        self.calls.append(("scope_clear_member", str(user_id)))
        return {"ok": True}

    def admin_clear_admin_commands(self, chat):
        self.calls.append(("scope_clear_admins",))
        return {"ok": True}

    def admin_forum_topic_edit(self, chat, thread_id, name):
        self.calls.append(("topic_edit", str(thread_id), name))
        return {"ok": True}

    def admin_dm_applicant(self, user_chat_id, text):
        self.calls.append(("dm_applicant", str(user_chat_id),
                           text))
        return {"ok": self.dm_ok,
                "error": "" if self.dm_ok else "blocked"}

    def admin_answer_join_query(self, query_id, result):
        self.calls.append(("answer_query", str(query_id), result))
        return {"ok": True}

    def admin_join_request(self, chat, user_id, approve):
        self.calls.append(("join_req", str(user_id), approve))
        return {"ok": True}


def _msg(key=BOTKEY, platform="telegram-bot", *,
         sender_id="admin1", meta=None, text="", thread_id=""):
    chat = SimpleNamespace(
        key=key, platform=platform, kind="group", title="Builders",
        chat_id=key.split(":", 1)[1], thread_id=thread_id)
    return SimpleNamespace(
        chat=chat, sender_id=sender_id, sender="Admin",
        sender_username="boss", text=text, meta=meta or {},
        reply_to="", message_id="900")


def _rig(tmp_path: Path, adapter=None, *, key=BOTKEY,
         designated=True):
    from nomorals.agents.partner.runtime_group import \
        RuntimeGroupMixin

    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    if designated:
        surface.set_designated(key, True, by="owner")
    adapter = adapter if adapter is not None else _BotFake()
    sent: list = []
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore
    mixin._group_adapter = (  # type: ignore
        lambda platform: adapter)
    mixin.gateway = SimpleNamespace(  # type: ignore
        send=lambda *a, **k: sent.append((a, k)),
        adapters={})
    return mixin, surface, adapter, sent


def _seed_member(surface, key=BOTKEY, uid="42", name="Ada",
                 username="ada"):
    surface.note_join(key, uid, username=username, name=name,
                      ts=NOW)


def test_tag_command_sets_native_and_ledger(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    _seed_member(surface)
    out = mixin._control_tag("@ada night owl", _msg())
    assert "night owl" in out
    assert bot.calls == [("set_tag", "42", "night owl")]
    assert surface.member(BOTKEY, "42")["tag"] == "night owl"
    # The member card renders it.
    from nomorals.agents.partner.runtime_group import _member_card

    card = _member_card(surface, BOTKEY,
                        surface.member(BOTKEY, "42"))
    assert "night owl" in card
    # Clear path.
    out = mixin._control_tag("@ada off", _msg())
    assert "cleared" in out
    assert surface.member(BOTKEY, "42")["tag"] == ""


def test_tag_command_gates(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    _seed_member(surface)
    # Non-admin refused before any platform call.
    out = mixin._control_tag("@ada owl", _msg(sender_id="42"))
    assert "admin" in out and bot.calls == []
    # Invalid tags refused by law, no platform call.
    out = mixin._control_tag("@ada " + "x" * 17, _msg())
    assert "16" in out and bot.calls == []
    out = mixin._control_tag("@ada party 🎉", _msg())
    assert "emoji" in out and bot.calls == []


def test_tag_command_unclaimed_refused(tmp_path):
    # On the userbot surface a room is claimed only by the
    # owner's designation (§33); without it the §16 tools
    # refuse.  (A bot-platform room is claimed by the bot's
    # presence there — claim_state's own rule.)
    mixin, surface, bot, _sent = _rig(
        tmp_path, key=UBKEY, designated=False)
    surface.note_join(UBKEY, "42", username="ada", name="Ada",
                      ts=NOW)
    out = mixin._control_tag(
        "@ada owl", _msg(key=UBKEY, platform="telegram"))
    assert "switched on" in out and bot.calls == []


def test_tag_platform_refusal_not_recorded(tmp_path):
    bot = _BotFake(tag_refuse=True)
    mixin, surface, bot, _sent = _rig(tmp_path, adapter=bot)
    _seed_member(surface)
    out = mixin._control_tag("@ada owl", _msg())
    assert "refused" in out
    assert surface.member(BOTKEY, "42").get("tag") in (None, "")
    failed = [a for a in surface.audit(BOTKEY, limit=50)
              if a["kind"] == "admin_action_failed"]
    assert failed and failed[-1]["action"] == "tag"


def test_tag_userbot_surface_ledger_carries(tmp_path):
    class _UbFake(_BotFake):
        def admin_set_member_tag(self, chat, user_id, tag):
            self.calls.append(("set_tag", str(user_id), tag))
            return {"ok": False,
                    "error": "unsupported: member tags are set "
                             "by the bot surface"}

    mixin, surface, bot, _sent = _rig(
        tmp_path, adapter=_UbFake(), key=UBKEY)
    surface.note_join(UBKEY, "42", username="ada", name="Ada",
                      ts=NOW)
    out = mixin._control_tag("@ada owl",
                             _msg(key=UBKEY, platform="telegram"))
    assert "member book" in out
    assert surface.member(UBKEY, "42")["tag"] == "owl"


def test_bansender_hits_channel_identity(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    _seed_member(surface)  # the nominal sender is a member…
    msg = _msg(meta={"reply_to_sender_chat": {
        "id": "-100999", "title": "Spam Channel"}})
    out = mixin._control_bansender("", msg, unban=False)
    # …but the ban lands on the channel id, never on "42".
    assert bot.calls == [("ban_sender", "-100999")]
    assert "Spam Channel" in out
    audit = [a for a in surface.audit(BOTKEY, limit=50)
             if a["kind"] == "admin_action"]
    assert audit[-1]["action"] == "ban_sender_chat"
    assert audit[-1]["user_id"] == "-100999"
    # Unban by numeric id.
    out = mixin._control_bansender("-100999", _msg(), unban=True)
    assert bot.calls[-1] == ("unban_sender", "-100999")
    assert "again" in out


def test_bansender_without_channel_target_is_guidance(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    out = mixin._control_bansender("", _msg(), unban=False)
    assert "reply to a post" in out and bot.calls == []


def test_reactwipe_command(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    _seed_member(surface)
    out = mixin._control_reactwipe("@ada", _msg())
    assert bot.calls == [("wipe", "42")]
    assert "wiped" in out


def test_boosters_board_and_member(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    _seed_member(surface)
    surface.note_boost(BOTKEY, boost_id="b1", user_id="42",
                       source_kind="premium", add_ts=NOW)
    surface.note_boost(BOTKEY, boost_id="b2", user_id="",
                       source_kind="giveaway", add_ts=NOW)
    out = mixin._control_boosters("", _msg())
    assert "2 active" in out and "Ada" in out
    assert "name nobody" in out
    out = mixin._control_boosters("@ada", _msg())
    assert "×1" in out and "Premium" in out
    # The live lookup rode along on the bot surface.
    assert ("user_boosts", "42") in bot.calls
    assert "Telegram's own list: 2" in out


def test_boost_intake_through_the_turn(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    msg = _msg(meta={"service": "chat_boost", "boost": {
        "boost_id": "b7", "user_id": "42",
        "source_kind": "premium", "add_date": NOW,
        "expiration_date": NOW + 86400}})
    assert mixin._group_surface_turn(msg) is True
    assert surface.boost_summary(BOTKEY, "42")["count"] == 1
    msg2 = _msg(meta={"service": "chat_boost_removed", "boost": {
        "boost_id": "b7", "user_id": "42"}})
    assert mixin._group_surface_turn(msg2) is True
    assert surface.boost_summary(BOTKEY, "42")["count"] == 0


def test_topic_duty_renames_auto_named_topic(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    created = _msg(meta={"service": "forum_topic_created",
                         "thread_id": "7",
                         "topic": {"title": "General",
                                   "implicit": True}})
    assert mixin._group_surface_turn(created) is True
    assert surface.topic_rename_candidate(BOTKEY, "7")
    # Duty off by default: traffic changes nothing.
    talk = _msg(sender_id="42", thread_id="7",
                text="Deploy window moves to Friday night")
    mixin._group_surface_turn(talk)
    assert bot.calls == []
    # Duty on: the first nameable message renames, once.
    surface.update_settings(BOTKEY, p5={"topics_rename": True})
    mixin._group_surface_turn(talk)
    edits = [c for c in bot.calls if c[0] == "topic_edit"]
    assert len(edits) == 1
    assert edits[0][1] == "7"
    assert edits[0][2].startswith("Deploy window")
    assert surface.topic_rename_candidate(BOTKEY, "7") is None
    mixin._group_surface_turn(talk)
    assert len([c for c in bot.calls
                if c[0] == "topic_edit"]) == 1


def test_queue_verdict_answers_the_query(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    surface.update_settings(BOTKEY, acceptance={
        "mode": "request", "auto_approve": False,
        "criteria": {"username": True}})
    msg = _msg(meta={
        "service": "join_request",
        "requester": {"id": "42", "username": "", "name": "Ada",
                      "first_name": "Ada", "last_name": "",
                      "is_bot": False},
        "user_chat_id": "555", "join_query_id": "q-9"})
    assert mixin._group_surface_turn(msg) is True
    # Held → the query was answered "queue"; nothing decided.
    assert ("answer_query", "q-9", "queue") in bot.calls
    assert bot.calls.count(("join_req", "42", True)) == 0
    queue = surface.acceptance_queue(BOTKEY)
    assert len(queue) == 1
    assert queue[0]["query_id"] == "q-9"
    deferred = [a for a in surface.audit(BOTKEY, limit=50)
                if a["kind"] == "join_query_deferred"]
    assert deferred and deferred[-1]["ok"] is True


def test_joinq_ask_window(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    surface.update_settings(BOTKEY, acceptance={
        "mode": "request", "auto_approve": False,
        "criteria": {"username": True}})
    out = surface.acceptance_on_request(
        BOTKEY, {"user_id": "42", "username": "", "name": "Ada",
                 "user_chat_id": "555"})
    rid = out["record"]["id"]
    # Inside the window: the DM goes out.
    reply = mixin._control_joinq(
        f"ask {rid} where did you find us?", _msg())
    assert "sent to" in reply
    assert bot.calls[-1][0] == "dm_applicant"
    assert bot.calls[-1][1] == "555"
    # The record is still queued afterwards.
    assert len(surface.acceptance_queue(BOTKEY)) == 1
    # Past the window: honest refusal, no second DM.
    record = surface.acceptance_queue(BOTKEY)[0]
    record["ts"] = time.time() - 400
    surface.store.save()
    calls_before = len(bot.calls)
    reply = mixin._control_joinq(f"ask {rid} still there?",
                                 _msg())
    assert "window has closed" in reply
    assert len(bot.calls) == calls_before


def test_p5_grammar_topics_ephemeral_scopes(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    out = mixin._control_gsettings("p5 topics on", _msg())
    assert "auto-rename on" in out
    assert surface.store.settings(BOTKEY)["p5"]["topics_rename"]
    out = mixin._control_gsettings("ephemeral on", _msg())
    assert "ephemeral answers on" in out
    assert surface.store.settings(BOTKEY)["p5"]["ephemeral"]
    out = mixin._control_gsettings("scopes sync", _msg())
    assert "synced" in out
    assert surface.store.settings(BOTKEY)["p5"]["member_scopes"]
    kinds = [c[0] for c in bot.calls]
    assert "scope_admins" in kinds
    assert ("scope_member", "admin1") in [
        (c[0], c[1]) for c in bot.calls]
    assert ("scope_member", "mod2") in [
        (c[0], c[1]) for c in bot.calls]
    out = mixin._control_gsettings("scopes clear", _msg())
    assert "cleared" in out
    kinds = [c[0] for c in bot.calls]
    assert "scope_clear_admins" in kinds
    # The console shares the same grammar function.
    out = mixin._apply_p5_grammar(surface, BOTKEY,
                                  ["topics", "off"], by="admin1")
    assert "off" in out
    assert not surface.store.settings(BOTKEY)["p5"]["topics_rename"]


def test_acceptance_explainer_platform_truth(tmp_path):
    mixin, surface, bot, _sent = _rig(tmp_path)
    out = mixin._control_gsettings(
        "acceptance action restricted", _msg())
    assert "muted with a fix-it note" in out
    # WhatsApp: the same stored action, the platform's truth.
    wa_key = "whatsapp:123@g.us"
    surface.set_designated(wa_key, True, by="owner")
    wa_msg = _msg(key=wa_key, platform="whatsapp")
    out = mixin._control_gsettings(
        "acceptance action restricted", wa_msg)
    assert "queues for a human" in out
    assert "fix-it note" not in out
    out = mixin._control_gsettings(
        "acceptance action kick", wa_msg)
    assert "queues for a human" in out
    out = mixin._control_gsettings(
        "acceptance action decline", wa_msg)
    assert "refused" in out
