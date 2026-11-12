"""Group duplication + game contradiction fix (owner screenshot
2026-11-12 ~02:32 AM, WhatsApp group "Devon").

The incident, replayed in the live shape: the bot says "no game
is live here" while the game keeps posting; the owner's own
account (userbot surface) and the bot BOTH post game messages in
the same group; the owner types "/game quit" and gets two replies
while the game never stops.

Proven mechanisms (audit + parent trace):

* dupfix2 is DM-only; WhatsApp has no peer-observation channel.
  Group rule (3) returns "down" only on a VERIFIED bot_admin,
  and the two runtimes hold SPLIT claim state for the group
  (userbot: room known, system_output_allowed False; bot: group
  unknown to its store at all) — so both handle group commands.
* The live wordchain room lives in the USERBOT's engine; the
  §33 gate is command-path-only, so the autonomous tick keeps
  playing in a blocked group.
* "/game quit" is a trap: the userbot's copy hits the §33 gate
  BEFORE verb dispatch (quit never runs); the bot's copy runs
  engine.quit on a runtime with no room ("no game is live
  here"). Two replies, game never stops.
* Repeats are unproven; the keying holes are real (reply_claims
  keys on inbound id only, edits unkeyed).
* Buttons: identity + form proven, delivery unproven at this rev
  (the gateway drops output_kind for WhatsApp) — instrumented
  here, and the derivation wired through the adapter's own
  declared params.

The law pinned here:

* A group game command is handled immediately ONLY by the
  runtime that OWNS the live room; every other surface defers
  (group window) and watches the room for the sibling's answer.
* A blocked group's autonomous tick emits nothing, and the room
  is reaped.
* An owner /game quit in a blocked group quits the owning
  runtime's room; the other surface stays silent.
* One platform message identity gets one conversational turn —
  and a redelivery with a fresh id, or an edit of an already
  answered message, is the same event, not a new one.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.partner.failures import clear_failures
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.games.engine import GameEngine
from nomorals.games.games.base import MultiGame
from nomorals.games.players import Player
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway
from nomorals.storage.db import Database

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OWNER_JID = "2348065574087@s.whatsapp.net"
OWNER_DIGITS = "2348065574087"
BOT_JID = "2348000000001@s.whatsapp.net"
GROUP_JID = "120363123456789@g.us"
GROUP_KEY = f"whatsapp:{GROUP_JID}"


class FakeTickGame(MultiGame):
    """A clock-driven game with no AI seats and no model calls:
    the timeout path is pure engine, so the blocked-gate test
    measures the gate, not the brain."""

    name = "faketick"
    move_timeout = 60.0
    stall_limit = 3


class Ctx:
    def __init__(self, db):
        self.db = db


class RecordingAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[tuple[str, Any]] = []

    def run(self, handler) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append((text, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


class WAAdapter(RecordingAdapter):
    """A WhatsApp surface: its own account JID plus a canned
    group roster (the bot is an admin; the owner is an admin)."""

    def __init__(self, name: str, user_jid: str) -> None:
        super().__init__(name)
        self.user_jid = user_jid
        self.user_lid = ""

    def group_participants(self, chat) -> list[dict[str, Any]]:
        return [
            {"jid": BOT_JID, "name": "Nomorals-2.0 Devon",
             "role": "admin"},
            {"jid": OWNER_JID, "name": "death", "role": "admin"},
        ]

    def admin_member(self, chat, user_id: str) -> dict[str, Any]:
        for row in self.group_participants(chat):
            if row["jid"] == user_id:
                return {"ok": True, "status": "administrator",
                        "role": row["role"], "jid": row["jid"],
                        "name": row["name"]}
        return {"ok": False, "error": "member not in the group roster"}


def _context(home: str):
    settings = load_settings(overrides={
        "home": home,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.enabled": "false",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    ws = Path(context.settings.workspace_dir)
    for marker in ("pulse_last_run.json",
                   "pulse_last_run_afternoon.json",
                   "pulse_last_run_evening.json",
                   "pulse_last_run_midnight.json"):
        (ws / marker).write_text(
            json.dumps({"ts": time.time(), "delivered_text": True}),
            encoding="utf-8")
    context.db.execute(
        "INSERT OR REPLACE INTO briefings (id, date, sections_json,"
        " generated_at) VALUES (?, ?, '[]', ?)",
        ("groupdup", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    return runtime


def _seed_owner(runtime) -> None:
    """The owner registry knows the owner's WhatsApp identity
    (the §15 pairing registry seeds it on the live box)."""
    runtime._owner_registry().seed(accounts=[
        SimpleNamespace(platform="whatsapp",
                        identity=OWNER_DIGITS),
    ])


@pytest.fixture()
def split_whatsapp():
    """Two runtimes, SEPARATE homes — the live shape. The bot
    runtime's store never knows the group; the userbot's store
    learns it when the test marks it."""
    clear_failures()
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-gd-a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-gd-b-",
                                        dir=_TEST_TMPDIR)
    ctx_a = _context(tmp_a.name)
    ctx_b = _context(tmp_b.name)
    ad_a = {"whatsapp": WAAdapter("whatsapp", BOT_JID)}
    ad_b = {"whatsapp": WAAdapter("whatsapp", OWNER_JID)}
    rt_a = _runtime(ctx_a, ad_a)
    rt_b = _runtime(ctx_b, ad_b)
    _seed_owner(rt_a)
    _seed_owner(rt_b)
    yield (rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b)
    for runtime, _adapters, context in ((rt_a, ad_a, ctx_a),
                                        (rt_b, ad_b, ctx_b)):
        with contextlib.suppress(Exception):  # teardown only
            runtime.stop()
        context.close()
    tmp_a.cleanup()
    tmp_b.cleanup()


def _texts(adapter):
    return [t for t, _b in adapter.sent]


def _group_chat() -> ChatRef:
    return ChatRef(platform="whatsapp", chat_id=GROUP_JID,
                   kind=ChatKind.GROUP, peer=GROUP_JID)


def _group_msg(text: str, message_id: str, *, sender_id=OWNER_JID,
               incoming=True, own=False) -> ChatMessage:
    meta: dict = {"is_owner": True}
    if own:
        meta["own_message"] = True
        meta["from_owner"] = True
    msg = ChatMessage(chat=_group_chat(), incoming=incoming,
                      text=text, sender="death",
                      message_id=message_id, meta=meta)
    msg.sender_id = sender_id
    return msg


def _mark_group_known_blocked(runtime) -> None:
    """The userbot's store learns the group with NO verified bot
    standing: known, unproven, system output NOT allowed —
    the incident's claim state."""
    surface = runtime._group_surface()
    surface.update_settings(GROUP_KEY, system_claim={})
    assert GROUP_KEY in surface.store.data
    assert surface.system_output_allowed(GROUP_KEY) is False


def _start_room(runtime, game: str = "wordchain") -> Any:
    """A live room in the group, built through the engine
    (setup, not the behaviour under test)."""
    engine = runtime._game_engine()
    host = Player.from_sender("whatsapp", OWNER_DIGITS, "death")
    room, _msgs = engine.start(GROUP_KEY, game, host, kind="group")
    assert engine.live(GROUP_KEY) is not None
    return room


# ── 1a. durable group claims ──────────────────────────────────

def test_durable_group_claim_store_round_trip():
    from nomorals.agents.partner.command_claims import DurableGroupClaimStore

    db = Database(":memory:")
    store = DurableGroupClaimStore(db)
    assert store.bot_admin(GROUP_KEY) is None
    assert store.record(GROUP_KEY) is None
    store.note_bot_admin(GROUP_KEY, True, bot_jid=BOT_JID)
    assert store.bot_admin(GROUP_KEY) is True
    rec = store.record(GROUP_KEY)
    assert rec is not None and rec.get("bot_jid") == BOT_JID
    store.note_bot_admin(GROUP_KEY, False)
    assert store.bot_admin(GROUP_KEY) is False
    # A demotion keeps the last known bot JID (the identity is
    # still the sibling's, even when its standing lapsed).
    assert store.record(GROUP_KEY).get("bot_jid") == BOT_JID


def test_group_surface_merges_durable_claim():
    from nomorals.agents.partner.command_claims import DurableGroupClaimStore
    from nomorals.social.chat.group_surface import GroupStore, GroupSurface

    tmp = tempfile.TemporaryDirectory(prefix="nm-gd-surface-",
                                      dir=_TEST_TMPDIR)
    try:
        db = Database(":memory:")
        durable = DurableGroupClaimStore(db)
        surface = GroupSurface(
            GroupStore(Path(tmp.name) / "groups.json").load())
        surface.durable_claims = durable
        # Local store knows nothing; the durable record is the
        # whole truth.
        durable.note_bot_admin(GROUP_KEY, True, bot_jid=BOT_JID)
        assert surface.claim_state(GROUP_KEY)["bot_admin"] is True
        # A local demotion is itself a verified observation —
        # the latest verified standing wins, so it lapses the
        # claim everywhere (the durable record follows it).
        surface.note_bot_admin(GROUP_KEY, False)
        assert surface.claim_state(GROUP_KEY)["bot_admin"] is False
        assert durable.bot_admin(GROUP_KEY) is False
        # A fresh durable verification re-establishes it.
        durable.note_bot_admin(GROUP_KEY, True, bot_jid=BOT_JID)
        assert surface.claim_state(GROUP_KEY)["bot_admin"] is True
        # The probe's stamp lands in BOTH stores.
        surface2 = GroupSurface(
            GroupStore(Path(tmp.name) / "groups.json").load())
        surface2.durable_claims = durable
        surface2.note_bot_admin(GROUP_KEY, True, bot_jid=BOT_JID)
        assert durable.bot_admin(GROUP_KEY) is True
        assert surface2.claim_state(GROUP_KEY)["bot_admin"] is True
    finally:
        tmp.cleanup()


# ── 1b. peer deferral: per-record window ──────────────────────

def test_peer_deferral_per_record_window():
    from nomorals.agents.partner.peer_deferral import PeerAnswerDeferral

    reg = PeerAnswerDeferral(window_s=20.0)
    base = 1000.0
    assert reg.register({"chat_key": "g1", "message_id": "m1"},
                        now=base)
    assert reg.register({"chat_key": "g2", "message_id": "m2",
                         "window_s": 3.0}, now=base)
    # At +4s the fast (group) record is due; the DM record waits.
    due = reg.expire(now=base + 4.0)
    assert [r["message_id"] for r in due] == ["m2"]
    assert reg.expire(now=base + 10.0) == []
    assert [r["message_id"] for r in reg.expire(now=base + 21.0)] \
        == ["m1"]


# ── 4. reply claims: edits + content key ─────────────────────

def _rc_msg(message_id, text, *, edited=False, sender_id=OWNER_JID):
    msg = SimpleNamespace(
        chat=SimpleNamespace(platform="whatsapp",
                             chat_id=GROUP_JID),
        message_id=message_id, sender_id=sender_id, text=text,
        meta={"edited": edited} if edited else {})
    return msg


def test_reply_claim_key_edits_are_the_same_event():
    from nomorals.agents.partner.reply_claims import reply_claim_key

    original = _rc_msg("m1", "microsoft loop")
    edited = _rc_msg("m1", "microsoft loop", edited=True)
    assert reply_claim_key(original) != ""
    # An edit of an already-answered message keys EXACTLY like
    # the original — it is the same event, not a new one.
    assert reply_claim_key(edited) == reply_claim_key(original)


def test_reply_content_key_catches_fresh_id_redelivery():
    from nomorals.agents.partner.reply_claims import ReplyClaimLedger, reply_content_key

    first = _rc_msg("m1", "microsoft loop")
    redelivered = _rc_msg("m2", "microsoft loop")
    different = _rc_msg("m3", "google chrome")
    assert reply_content_key(first) != ""
    assert reply_content_key(redelivered) \
        == reply_content_key(first)
    assert reply_content_key(different) \
        != reply_content_key(first)
    ledger = ReplyClaimLedger()
    assert ledger.claim(reply_content_key(first), now=1000.0,
                        window_s=30.0) is True
    # The same words from the same sender ten seconds later,
    # under a fresh id, is a redelivery — held.
    assert ledger.claim(reply_content_key(redelivered), now=1010.0,
                        window_s=30.0) is False
    # A genuinely different message passes.
    assert ledger.claim(reply_content_key(different), now=1010.0,
                        window_s=30.0) is True
    # Past the short window the same words are a new event.
    assert ledger.claim(reply_content_key(redelivered), now=1100.0,
                        window_s=30.0) is True


def test_reply_already_claimed_holds_fresh_id_redelivery(
        split_whatsapp):
    (_rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_whatsapp
    first = _rc_msg("m1", "microsoft loop")
    redelivered = _rc_msg("m2", "microsoft loop")
    assert rt_b._reply_already_claimed(first) is False
    assert rt_b._reply_already_claimed(redelivered) is True
    assert rt_b._reply_already_claimed(
        _rc_msg("m3", "google chrome")) is False


# ── 2. the §33 gate, unified: the tick emits nothing ─────────

def test_engine_blocked_tick_emits_nothing_and_reaps():
    db = Database(":memory:")
    db.migrate()
    sent: list[tuple[str, str]] = []
    engine = GameEngine(
        Ctx(db), send=lambda c, t: sent.append((c, t)),
        blocked=lambda key: key == GROUP_KEY)
    engine.register(FakeTickGame())
    host = Player.from_sender("whatsapp", OWNER_DIGITS, "death")
    room, _msgs = engine.start(GROUP_KEY, "faketick", host,
                               kind="group")
    try:
        assert engine.live(GROUP_KEY) is not None
        room.turn_started = 0.0  # the clock is long past due
        engine._sweep_timeouts()
        # The blocked tick emits NOTHING into the group …
        assert sent == [], sent
        # … and the room is reaped, not left ticking.
        assert engine.live(GROUP_KEY) is None
        assert engine._rooms.get(GROUP_KEY) is None
    finally:
        engine.shutdown()


def test_engine_unblocked_tick_still_emits():
    db = Database(":memory:")
    db.migrate()
    sent: list[tuple[str, str]] = []
    engine = GameEngine(
        Ctx(db), send=lambda c, t: sent.append((c, t)),
        blocked=lambda key: False)
    engine.register(FakeTickGame())
    host = Player.from_sender("whatsapp", OWNER_DIGITS, "death")
    room, _msgs = engine.start(GROUP_KEY, "faketick", host,
                               kind="group")
    try:
        room.turn_started = 0.0
        engine._sweep_timeouts()
        assert sent, "the unblocked tick must still speak"
        assert engine.live(GROUP_KEY) is not None
    finally:
        engine.shutdown()

# ── 1. arbitration verdicts: one handling, across surfaces ───

def _arbitrate(runtime, text, message_id, *, own=False):
    from nomorals.social.chat.control import parse_control

    command = parse_control(text)
    assert command is not None, text
    msg = _group_msg(text, message_id, own=own)
    return runtime._command_arbitration(msg, command)


def test_arbitration_owner_of_room_handles_quit(split_whatsapp):
    """The incident's first half: the runtime that OWNS the
    live room handles the owner's /game quit at once."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_whatsapp
    _start_room(rt_b)
    _mark_group_known_blocked(rt_b)
    assert _arbitrate(rt_b, "/game quit", "q1", own=True) == "handle"


def test_arbitration_no_room_defers_group_game(split_whatsapp):
    """The incident's second half: the runtime with NO room
    defers a group game command instead of answering "no game
    is live here" on top of the owner's answer."""
    (rt_a, _ad_a, _ctx_a), (_rt_b, _ad_b, _ctx_b) = split_whatsapp
    assert _arbitrate(rt_a, "/game quit", "q1") == "defer"
    assert _arbitrate(rt_a, "/game", "q2") == "defer"


def test_arbitration_room_owner_wins_over_known_claim(
        split_whatsapp):
    """The bot owns the room; the userbot's store knows the
    group with unproven standing: the bot handles, the userbot
    defers and watches."""
    (rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_whatsapp
    _start_room(rt_a)
    _mark_group_known_blocked(rt_b)
    assert _arbitrate(rt_a, "/game", "g1") == "handle"
    assert _arbitrate(rt_b, "/game", "g1") == "defer"


def test_arbitration_non_game_group_command(split_whatsapp):
    """Rule (3) does NOT defer non-game group commands on
    unproven standing: they fall through to rule (4) and run
    at once — the claimed-once contract pins immediate
    exactly-once handling, and a deferral here would mute
    every group command until a sweep. The cross-surface
    deferral is the game commands' rule (1.5) above; a
    non-game cross-runtime double is the shared-home-store
    follow-up, not a deferral."""
    (rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_whatsapp
    _mark_group_known_blocked(rt_b)
    assert _arbitrate(rt_b, "/play haza", "p1") == "handle"
    assert _arbitrate(rt_a, "/play haza", "p1") == "handle"


def test_arbitration_unknown_group_handles(split_whatsapp):
    """A group the surface never saw is not a deferral: no bot
    copy is expected there, so the command is handled at once."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_whatsapp
    assert _arbitrate(rt_b, "/play haza", "p1") == "handle"


# ── 3. the /game quit trap ────────────────────────────────────

def test_control_game_owner_quit_bypasses_gate(split_whatsapp):
    """An owner /game quit in a blocked group quits the live
    room in THAT runtime — the §33 gate no longer eats it."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_whatsapp
    _start_room(rt_b)
    _mark_group_known_blocked(rt_b)
    host = Player.from_sender("whatsapp", OWNER_DIGITS, "death")
    out = rt_b._control_game("quit", GROUP_KEY, player=host,
                             kind="group", is_owner=True)
    assert "table closed" in out, out
    assert "games aren't running" not in out
    assert rt_b._game_engine().live(GROUP_KEY) is None


def test_control_game_non_owner_quit_stays_gated(split_whatsapp):
    """The bypass is owner-only: anyone else's quit in a
    blocked group still meets the gate."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_whatsapp
    _start_room(rt_b)
    _mark_group_known_blocked(rt_b)
    host = Player.from_sender("whatsapp", "999", "stranger")
    out = rt_b._control_game("quit", GROUP_KEY, player=host,
                             kind="group", is_owner=False)
    assert "games aren't running in this group" in out
    assert rt_b._game_engine().live(GROUP_KEY) is not None


def test_handle_control_game_preserves_group_kind(split_whatsapp):
    """The failover path must not smuggle a game start into a
    blocked group through the console path's kind="dm"."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_whatsapp
    _mark_group_known_blocked(rt_b)
    msg = _group_msg("/game wordchain", "s1")
    out = rt_b.handle_control("/game wordchain", GROUP_KEY,
                              message=msg)
    assert "games aren't running in this group" in str(out), out
    assert rt_b._game_engine().live(GROUP_KEY) is None


# ── end to end: the incident, replayed ────────────────────────

def test_end_to_end_group_quit_exactly_one_reply(split_whatsapp):
    """The incident shape, whole: a live wordchain in a blocked
    group on the userbot; the owner types /game quit; exactly
    ONE reply lands across both surfaces, and the game stops."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_whatsapp
    _start_room(rt_b)
    _mark_group_known_blocked(rt_b)
    # The room start itself emits the live-table card into the
    # adapter; the pin measures what the QUIT adds, not setup.
    baseline_b = len(_texts(ad_b["whatsapp"]))
    baseline_a = len(_texts(ad_a["whatsapp"]))
    # The owner's typed command reaches BOTH runtimes under the
    # same per-chat message id (the WhatsApp shape: one message,
    # two deliveries — the bot's bridge copy and the userbot's
    # own-echo).
    rt_b._process_inner(_group_msg("/game quit", "q1", own=True))
    rt_a._process_inner(_group_msg("/game quit", "q1"))
    userbot_sends = _texts(ad_b["whatsapp"])[baseline_b:]
    # The quit answers once — the living-table card is
    # game-surface traffic (the finish renders the final
    # table), not a second answer to the command. The pin
    # counts the command's answer.
    userbot_replies = [t for t in userbot_sends
                       if "table closed" in t]
    assert len(userbot_replies) == 1, userbot_sends
    assert _texts(ad_a["whatsapp"])[baseline_a:] == []
    assert rt_b._game_engine().live(GROUP_KEY) is None
    # The userbot's answer lands in the room; the bot runtime
    # observes its own account's sibling answering and stands
    # down — its deferral resolves, nothing is sent, ever.
    rt_a._process_inner(_group_msg(userbot_replies[0], "q2"))
    assert _texts(ad_a["whatsapp"])[baseline_a:] == []
    assert rt_a._peer_deferral().pending_count() == 0
    rt_a._sweep_peer_deferrals(now=time.time() + 60)
    assert _texts(ad_a["whatsapp"])[baseline_a:] == []
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    assert len([t for t in _texts(ad_b["whatsapp"])[baseline_b:]
                if "table closed" in t]) == 1


def test_end_to_end_group_fresh_game_exactly_one_reply(
        split_whatsapp):
    """No room anywhere: a group /game defers on both sides;
    the stateless side fails over first (3s) and answers; the
    known side observes it and stays silent. Exactly one list."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_whatsapp
    _mark_group_known_blocked(rt_b)
    baseline_a = len(_texts(ad_a["whatsapp"]))
    baseline_b = len(_texts(ad_b["whatsapp"]))
    rt_b._process_inner(_group_msg("/game", "g1", own=True))
    rt_a._process_inner(_group_msg("/game", "g1"))
    assert _texts(ad_a["whatsapp"])[baseline_a:] == []
    assert _texts(ad_b["whatsapp"])[baseline_b:] == []
    # The bot's fast failover (unknown side, 3s) answers first.
    rt_a._sweep_peer_deferrals(now=time.time() + 4.0)
    bot_lists = _texts(ad_a["whatsapp"])[baseline_a:]
    assert len(bot_lists) == 1, bot_lists
    assert "/game <name>" in bot_lists[0]
    # Its answer, observed by the userbot runtime as the bot's
    # own post (roster-admin peer), resolves the known side.
    rt_b._process_inner(_group_msg(
        bot_lists[0], "g2", sender_id=BOT_JID))
    assert _texts(ad_b["whatsapp"])[baseline_b:] == []
    assert rt_b._peer_deferral().pending_count() == 0
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    assert _texts(ad_b["whatsapp"])[baseline_b:] == []

# ── 5. buttons: derivation through the adapter + journal ──────

def _wa_adapter(tmp: str, responder):
    """Lane-1 rig (test_whatsapp_button_forms): a connected
    adapter whose bridge is a recorder."""
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_dir=str(Path(tmp) / "outbox"))
    adapter.connected.set()
    calls: list = []

    def fake_send_cmd(cmd, timeout=25.0):
        calls.append(dict(cmd))
        return responder(cmd)

    adapter._send_cmd = fake_send_cmd  # type: ignore[method-assign]
    return adapter, calls


def _ok_responder(cmd) -> dict[str, Any]:
    return {"ok": True, "message_id": "M1",
            "buttons_form": "interactive"}


def _wa_chat_dm() -> ChatRef:
    return ChatRef(platform="whatsapp",
                   chat_id="2348012345678@c.us", kind=ChatKind.DM)


def _journal_rows(tmp: str) -> list[dict[str, Any]]:
    path = Path(tmp) / "outbox" / "whatsapp_buttons_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_whatsapp_send_game_kind_derives_keyboard_and_journals():
    """The screenshot's keyboard, derived: a send carrying
    output_kind="game" rides the adaptive game keyboard on
    WhatsApp — and the journal names what rode, how, and on
    which build."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-gd-wa-",
                                      dir=_TEST_TMPDIR)
    try:
        adapter, calls = _wa_adapter(tmp.name, _ok_responder)
        adapter.bridge_program = "gobridge/1.0.0"
        adapter.bridge_build = {"commit": "7a09b503",
                                "built": "2026-11-12T08:19:57Z"}
        result = adapter.send(
            _wa_chat_dm(), "no game is live here.",
            output_kind="game", output_meta={"game": "quit"})
        assert result.ok, result
        sent = [c for c in calls if c.get("cmd") == "send"]
        assert sent, calls
        payloads = [str(data) for row in (sent[0].get("buttons") or [])
                    for _label, data in row]
        assert "game rematch" in payloads, payloads
        assert "game leaderboard" in payloads, payloads
        rows = _journal_rows(tmp.name)
        assert len(rows) == 1, rows
        row = rows[0]
        assert row["adapter"] == "whatsapp"
        assert row["direction"] == "send"
        assert row["output_kind"] == "game"
        assert row["derived"] is True
        assert row["declared"] is False
        assert row["form"] == "interactive"
        assert row["bridge_build"]["commit"] == "7a09b503"
    finally:
        tmp.cleanup()


def test_whatsapp_send_plain_text_writes_no_journal_row():
    """No buttons, no row: the journal records button-carrying
    sends only, not every send."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-gd-wa-",
                                      dir=_TEST_TMPDIR)
    try:
        adapter, _calls = _wa_adapter(tmp.name, _ok_responder)
        result = adapter.send(_wa_chat_dm(), "hello there")
        assert result.ok, result
        assert _journal_rows(tmp.name) == []
    finally:
        tmp.cleanup()


def test_whatsapp_edit_message_journals_declared_buttons():
    tmp = tempfile.TemporaryDirectory(prefix="nm-gd-wa-",
                                      dir=_TEST_TMPDIR)
    try:
        adapter, _calls = _wa_adapter(tmp.name, _ok_responder)
        verdict = adapter.edit_message(
            _wa_chat_dm(), "M1", "live game: wordchain",
            buttons=[[("Rematch", "game rematch"),
                      ("Quit game", "game quit")]])
        assert verdict.get("ok"), verdict
        rows = _journal_rows(tmp.name)
        assert len(rows) == 1, rows
        assert rows[0]["direction"] == "edit"
        assert rows[0]["declared"] is True
        assert rows[0]["derived"] is False
    finally:
        tmp.cleanup()


def test_whatsapp_adapter_redelivery_dedupe():
    """A bridge redelivery of the same message id inside the
    window is dropped at the adapter — the runtime never sees
    it twice."""
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter()
    msgs: list = []
    event = {
        "type": "message",
        "id": "wamid.RED1",
        "chat": {"id": GROUP_JID, "kind": "group",
                 "title": "Devon"},
        "from": {"id": OWNER_JID, "name": "death"},
        "text": "microsoft loop",
        "ts": int(time.time() * 1000),
    }
    adapter._dispatch(dict(event), msgs.append)
    adapter._dispatch(dict(event), msgs.append)
    assert len(msgs) == 1, msgs
    # A genuinely new message still flows.
    event2 = dict(event, id="wamid.RED2", text="google chrome")
    adapter._dispatch(event2, msgs.append)
    assert len(msgs) == 2, msgs
