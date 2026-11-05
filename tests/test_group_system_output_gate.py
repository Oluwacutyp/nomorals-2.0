"""DESIGN §33 — the system-output gate (two hats).

The live bug: the owner's Telegram USERBOT posted a group
quest completion into a 3,688-member community his account
merely sits in.  Bot group systems enrolled the room from his
account's presence, accrued quest state, and announced the
completion through his personal session.

The law pinned here:

- Hat 1 — as-him conversation on personal accounts is NEVER
  gated: an ordinary message in an unclaimed room is not
  consumed by the group surface; it flows on to conversation,
  and the silent reads (roster observation, identity ledger)
  keep working.
- Hat 2 — bot-system OUTPUT (quest completions, economy
  announcements, game boards, welcomes, moderation notices)
  posts only in CLAIMED rooms: her bot operating there, or a
  room the owner designated.
- NO foreign accrual: no economy verification, no streak, no
  quest progress/completion, no game purse in unclaimed rooms.
- A claim that lapses (verified bot demotion, designation
  lifted) stops output at send time.
- Claimed rooms behave exactly as before.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupSurface,
)

BOT_KEY = "telegram-bot:-100123"
USER_KEY = "telegram:-100999"


def _surface(tmp_path) -> GroupSurface:
    return GroupSurface(
        GroupStore(tmp_path / "groups.json").load())


# ── surface: claim derivation ─────────────────────────────────


def test_claim_derivation_and_designation(tmp_path):
    surface = _surface(tmp_path)
    # a personal-session room starts unclaimed; a bot room is
    # claimed by the bot's own presence there
    assert surface.claim_state(USER_KEY)["claimed"] is False
    bot = surface.claim_state(BOT_KEY)
    assert bot["claimed"] is True and bot["source"] == "bot"
    assert surface.system_output_allowed(USER_KEY) is False
    assert surface.system_output_allowed(BOT_KEY) is True
    # the sending session can prove the bot's standing even
    # when the room's key lives on a personal platform
    assert surface.system_output_allowed(
        USER_KEY, via_platform="telegram-bot") is True
    # owner designation claims any room, by name
    state = surface.set_designated(USER_KEY, True, by="99")
    assert state["claimed"] is True
    assert state["source"] == "designated"
    assert surface.system_output_allowed(USER_KEY) is True
    # lifting the designation un-claims it again
    surface.set_designated(USER_KEY, False, by="99")
    assert surface.system_output_allowed(USER_KEY) is False
    kinds = [a["kind"] for a in surface.audit(USER_KEY)]
    assert kinds.count("system_claim") == 2


def test_verified_demotion_lapses_the_bot_claim(tmp_path):
    surface = _surface(tmp_path)
    surface.note_bot_admin(BOT_KEY, False)
    state = surface.claim_state(BOT_KEY)
    assert state["bot_admin"] is False
    assert state["claimed"] is False
    # the lapse holds on EVERY session, even a bot-platform one
    assert surface.system_output_allowed(
        BOT_KEY, via_platform="telegram-bot") is False
    # reinstatement restores it
    surface.note_bot_admin(BOT_KEY, True)
    assert surface.system_output_allowed(BOT_KEY) is True
    # ...but a designation stands through a demotion: the owner
    # named the room, only the owner lifts it
    surface.set_designated(BOT_KEY, True, by="99")
    surface.note_bot_admin(BOT_KEY, False)
    assert surface.system_output_allowed(BOT_KEY) is True


# ── one claim state per ROOM, across the Telegram family ──────
# The 2026-11-05 contradiction: the executor read the claim
# under telegram-bot:-100123 (claimed) while the §33 gate on the
# userbot read telegram:-100123 (empty record, "unclaimed") and
# posted the gate text beside the execution. The family keys
# name one room; the facts merge.

ROOM_BOT = "telegram-bot:-100123"
ROOM_USER = "telegram:-100123"
ROOM_PERSONAL_BOT = "telegram-personal:-100123"


def test_bot_standing_is_the_room_s_fact_on_every_surface(tmp_path):
    surface = _surface(tmp_path)
    # the runtime's probe stamps verified standing under the
    # BOT key only — every family surface reads the same fact
    surface.note_bot_admin(ROOM_BOT, True)
    for key in (ROOM_BOT, ROOM_USER, ROOM_PERSONAL_BOT):
        state = surface.claim_state(key)
        assert state["bot_admin"] is True, key
        assert state["claimed"] is True, key
        assert surface.system_output_allowed(key) is True, key


def test_designation_is_the_room_s_fact_on_every_surface(tmp_path):
    surface = _surface(tmp_path)
    # designated through the userbot's key (the DM console writes
    # under whichever surface the owner was driving) — the bot
    # surface reads the same room fact
    surface.set_designated(ROOM_USER, True, by="owner")
    for key in (ROOM_BOT, ROOM_USER, ROOM_PERSONAL_BOT):
        state = surface.claim_state(key)
        assert state["designated"] is True, key
        assert state["claimed"] is True, key
        assert state["source"] == "designated", key


def test_verified_demotion_lapses_across_the_family(tmp_path):
    surface = _surface(tmp_path)
    surface.note_bot_admin(ROOM_BOT, True)
    assert surface.claim_state(ROOM_USER)["claimed"] is True
    surface.note_bot_admin(ROOM_BOT, False)
    for key in (ROOM_BOT, ROOM_USER, ROOM_PERSONAL_BOT):
        state = surface.claim_state(key)
        assert state["bot_admin"] is False, key
        assert state["claimed"] is False, key
        assert surface.system_output_allowed(key) is False, key


def test_family_merge_never_crosses_rooms(tmp_path):
    surface = _surface(tmp_path)
    surface.note_bot_admin(ROOM_BOT, True)
    surface.set_designated(ROOM_USER, True, by="owner")
    # a different room under any family key stays unclaimed
    assert surface.claim_state("telegram:-100777")["claimed"] is False
    assert surface.claim_state(
        "telegram-bot:-100777")["bot_admin"] is None


# ── surface: no foreign accrual ───────────────────────────────


def test_no_economy_accrual_in_unclaimed_room(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(USER_KEY, "u1", username="ada",
                           name="Ada", text="hello", ts=1000.0)
    surface.handle_message(USER_KEY, "u1", username="ada",
                           name="Ada", text="again", ts=1060.0)
    econ = surface.economy
    # the roster read is a silent read — it still happened
    member = surface.store.group(USER_KEY)["members"]["u1"]
    assert member["message_count"] == 2
    # but NO economy state: never verified, no streak spark,
    # no activity on the books
    assert econ.is_verified(USER_KEY, "u1") is False
    assert econ.balance(USER_KEY, "u1") == 0
    assert econ.activity_count(USER_KEY, "u1", "message",
                               "daily", 1000.0) == 0
    # the same two messages in a claimed bot room accrue as
    # they always have (streak pays 5 once per day)
    surface.handle_message(BOT_KEY, "u1", username="ada",
                           name="Ada", text="hello", ts=1000.0)
    surface.handle_message(BOT_KEY, "u1", username="ada",
                           name="Ada", text="again", ts=1060.0)
    assert econ.is_verified(BOT_KEY, "u1") is True
    assert econ.balance(BOT_KEY, "u1") == 5
    assert econ.activity_count(BOT_KEY, "u1", "message",
                               "daily", 1000.0) == 2


def test_quest_completion_suppressed_until_claimed(tmp_path):
    surface = _surface(tmp_path)
    econ = surface.economy
    now = time.time()
    # legacy state, driven straight into the organ (as a room
    # enrolled before the gate would hold): a full Loud Week
    for _i in range(500):
        econ.record_activity(USER_KEY, "55", "message", now=now)
    extras = surface._extras(USER_KEY)
    extras["quest_check_at"] = 0
    # the cycle pays nothing and mints nothing abroad
    assert surface.group_quest_cycle(USER_KEY, now=now + 10) \
        == []
    assert econ.treasury_balance(USER_KEY) == 0
    out = surface.handle_message(
        USER_KEY, "55", username="bola", name="Bola",
        text="anyone here?", ts=now + 20)
    assert not any("quest complete" in n
                   for n in out["notices"])
    assert econ.treasury_balance(USER_KEY) == 0
    # claim the room and the very same state completes + pays
    surface.set_designated(USER_KEY, True, by="99")
    extras["quest_check_at"] = 0
    paid = surface.group_quest_cycle(USER_KEY, now=now + 30)
    assert "Loud Week" in {p["name"] for p in paid}
    assert econ.treasury_balance(USER_KEY) >= 200


def test_game_award_and_season_cycle_gated(tmp_path):
    surface = _surface(tmp_path)
    assert surface.game_award(USER_KEY, "u9", 40, "ttt",
                              now=1000.0) == 0
    assert surface.economy.balance(USER_KEY, "u9") == 0
    assert surface.game_award(BOT_KEY, "u9", 40, "ttt",
                              now=1000.0) == 40
    from nomorals.community.group_economy import season_key

    econ = surface.economy
    old = time.time() - 40 * 7 * 86400
    econ.award(USER_KEY, "55", 50, "old season work", now=old)
    extras = surface._extras(USER_KEY)
    extras["season_seen"] = season_key(old)
    # no season turns abroad — the state is frozen, not crowned
    assert surface.season_cycle(USER_KEY,
                                now=time.time()) is None
    assert extras["season_seen"] == season_key(old)


# ── runtime rig ───────────────────────────────────────────────


class _Chat:
    def __init__(self, platform, key, title="Builders"):
        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = title
        self.kind = ChatKind.GROUP


class _Msg:
    def __init__(self, chat, **kw):
        self.chat = chat
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")


class _Adapter:
    """The receiving session's adapter. ``_bot_id`` + a
    configurable bot status let the runtime's claim probe run;
    ``admin_ids`` are the human admins."""

    def __init__(self, admin_ids=(), bot_status="administrator"):
        self.calls = []
        self.admin_ids = set(admin_ids)
        self._bot_id = 4242
        self.bot_status = bot_status

    def admin_delete(self, chat, message_id):
        self.calls.append(("delete", message_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", user_id))
        return {"ok": True}

    def admin_restrict(self, chat, user_id, *, until=0, **kw):
        self.calls.append(("restrict", user_id, until))
        return {"ok": True}

    def admin_member(self, chat, user_id):
        if str(user_id) == str(self._bot_id):
            status = self.bot_status
        elif str(user_id) in self.admin_ids:
            status = "administrator"
        else:
            status = "member"
        return {"ok": True, "result": {"status": status}}


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, str(
            getattr(chat, "key", chat)), text))
        return SimpleNamespace(ok=True, message_id="1")


class _Runtime(RuntimeGroupMixin, RuntimeGamesMixin):
    def __init__(self, tmp_path, adapters, operators=()):
        self.sent = []
        self.gateway = _Gateway(adapters)
        self.operators = set(operators)
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))

    def _send(self, chat, text, **kw):
        self.sent.append(text)
        return SimpleNamespace(ok=True, message_id="1")

    def _is_operator(self, message):
        return str(message.sender_id or "") in self.operators

    def turn(self, msg):
        return self._group_surface_turn(msg)


def _rig(tmp_path, platform="telegram", key=USER_KEY, **kw):
    adapter = _Adapter(**kw)
    rt = _Runtime(tmp_path, {platform: adapter},
                  operators={"99"})
    chat = _Chat(platform, key)
    return rt, adapter, chat


def _join(chat, user_id="7", name="Ada", username="ada"):
    return _Msg(chat, meta={"service": "join", "new_members": [
        {"id": user_id, "username": username, "name": name}]},
        sender_id=user_id, sender=name, sender_username=username)


# ── runtime: the two hats ─────────────────────────────────────


def test_userbot_room_no_welcome_no_posts_but_chat_flows(
        tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    surface = rt._group_surface()
    # a join is recorded (roster read) but NO welcome posts
    assert rt.turn(_join(chat)) is True
    assert rt.sent == []
    member = surface.store.group(USER_KEY)["members"]["7"]
    assert member["username"] == "ada"
    # an ordinary message is NOT consumed: it flows on to the
    # as-him conversation (hat 1 — never gated), nothing posts,
    # and no economy state accrued behind it
    msg = _Msg(chat, text="hello room", message_id="56")
    assert rt.turn(msg) is False
    assert rt.sent == []
    assert surface.economy.balance(USER_KEY, "7") == 0
    assert surface.economy.is_verified(USER_KEY, "7") is False


def test_bot_room_behaves_exactly_as_before(tmp_path):
    rt, adapter, chat = _rig(tmp_path, platform="telegram-bot",
                             key=BOT_KEY)
    surface = rt._group_surface()
    assert rt.turn(_join(chat)) is True
    assert rt.sent and "Ada" in rt.sent[0]
    rt.turn(_Msg(chat, text="hello", message_id="56"))
    assert surface.economy.balance(BOT_KEY, "7") == 5


def test_designated_userbot_room_speaks_as_the_system(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    surface = rt._group_surface()
    surface.set_designated(USER_KEY, True, by="99")
    assert rt.turn(_join(chat)) is True
    assert rt.sent and "Ada" in rt.sent[0]
    rt.turn(_Msg(chat, text="hello", message_id="56"))
    assert surface.economy.balance(USER_KEY, "7") == 5


def test_moderation_notice_posts_only_in_claimed_rooms(
        tmp_path):
    # seed the link allowlist on both rooms
    from nomorals.social.chat.group_surface import default_surface

    surface = default_surface(str(tmp_path))
    mod = {"flood_max": 6, "flood_window_s": 10.0,
           "repeat_threshold": 3, "link_allowlist": ["example.com"]}
    surface.update_settings(USER_KEY, moderation=mod)
    surface.update_settings(BOT_KEY, moderation=mod)

    rt_u, adapter_u, chat_u = _rig(tmp_path)
    msg = _Msg(chat_u, text="see https://evil.example.net/x",
               message_id="99")
    assert rt_u.turn(msg) is True  # the delete executed
    assert ("delete", "99") in adapter_u.calls
    assert rt_u.sent == []  # ...but the warning never posted
    kinds = [a["kind"] for a in
             rt_u._group_surface().audit(USER_KEY)]
    assert "system_output_suppressed" in kinds

    rt_b, adapter_b, chat_b = _rig(tmp_path,
                                   platform="telegram-bot",
                                   key=BOT_KEY)
    msg = _Msg(chat_b, text="see https://evil.example.net/x",
               message_id="99")
    assert rt_b.turn(msg) is True
    assert ("delete", "99") in adapter_b.calls
    assert rt_b.sent  # the §31 notice posted, as before


def test_probe_records_demotion_and_output_stops(tmp_path):
    rt, adapter, chat = _rig(tmp_path, platform="telegram-bot",
                             key=BOT_KEY, bot_status="member")
    surface = rt._group_surface()
    # the first bot-platform turn probes the bot's own standing
    assert rt.turn(_join(chat)) is True
    assert surface.claim_state(BOT_KEY)["bot_admin"] is False
    assert rt.sent == []  # demoted bot: no welcome, no output
    # the bot is re-promoted; a fresh process (fresh probe
    # cache) sees it and the room speaks again
    adapter.bot_status = "administrator"
    rt2 = _Runtime(tmp_path, {"telegram-bot": adapter},
                   operators={"99"})
    assert rt2.turn(_join(chat, user_id="8", name="Bola",
                          username="bola")) is True
    assert rt2._group_surface().claim_state(
        BOT_KEY)["bot_admin"] is True
    assert rt2.sent and "Bola" in rt2.sent[0]


# ── runtime: designation grammar + console listing ────────────


def test_gsettings_system_output_is_the_owners_call(tmp_path):
    rt, adapter, chat = _rig(tmp_path, admin_ids={"7"})
    surface = rt._group_surface()
    # an operator designates the room in-group
    out = rt._control_gsettings(
        "system output on",
        _Msg(chat, sender_id="99", sender="Owner",
             sender_username="boss"))
    assert "designated" in str(out)
    assert surface.claim_state(USER_KEY)["designated"] is True
    # a group admin who is not the owner cannot flip it
    out = rt._control_gsettings(
        "system output off", _Msg(chat, sender_id="7"))
    assert "owner's call" in str(out)
    assert surface.claim_state(USER_KEY)["designated"] is True
    # the settings card reads the same state back
    card = rt._control_gsettings("", _Msg(chat, sender_id="99",
                                          sender="Owner"))
    text = getattr(card, "text", card)
    assert "System output" in text and "ON" in text


def test_console_listing_shows_claim_state(tmp_path):
    adapter = _Adapter(admin_ids={"99"})
    rt = _Runtime(tmp_path, {"telegram-bot": adapter,
                             "telegram": adapter},
                  operators={"99"})
    surface = rt._group_surface()
    surface.store.group(BOT_KEY)
    surface.store.group(USER_KEY)
    dm = _Msg(_Chat("telegram-bot", "telegram-bot:99",
                    title=""), sender_id="99", sender="Owner")
    dm.chat.kind = ChatKind.DM
    out = rt._control_console("", dm)
    text = getattr(out, "text", out)
    assert BOT_KEY in text and "system output on" in text
    assert USER_KEY in text and "system output off" in text


# ── games: boards and starts respect the gate ─────────────────


def test_relay_board_never_posts_into_unclaimed_group(tmp_path):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, {"telegram": adapter},
                  operators={"99"})
    surface = rt._group_surface()
    surface.store.group(USER_KEY)  # a room the surface knows
    assert rt._relay_send(USER_KEY, "♟️ board") is False
    assert rt.gateway.sent == []
    kinds = [a["kind"] for a in surface.audit(USER_KEY)]
    assert "system_output_suppressed" in kinds
    # a DM relay chat is not a group-system surface at all
    assert rt._relay_send("telegram:555", "♟️ board") is True
    assert rt.gateway.sent
    # claimed, the same board lands
    surface.set_designated(USER_KEY, True, by="99")
    assert rt._relay_send(USER_KEY, "♟️ board") is True


def test_group_game_start_refused_in_unclaimed_room(tmp_path):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, {"telegram": adapter},
                  operators={"99"})
    surface = rt._group_surface()
    surface.store.group(USER_KEY)
    out = rt._control_game("ttt", USER_KEY, kind="group")
    assert "aren't running in this group" in str(out)
    # the same call as a DM is untouched by the gate
    out_dm = rt._control_game("list", USER_KEY, kind="dm")
    assert "aren't running in this group" not in str(out_dm)
    # and a claimed room starts games as before
    surface.set_designated(USER_KEY, True, by="99")
    out2 = rt._control_game("list", USER_KEY, kind="group")
    assert "aren't running in this group" not in str(out2)
