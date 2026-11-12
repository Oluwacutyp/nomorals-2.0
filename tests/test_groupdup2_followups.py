"""Group duplication follow-ups, round 2 (owner-reported LIVE).

Three named remainders from the groupdup ship (``2c8eebe1``) + the
ownmsg ship (``1b9dc0e6``), closed here:

1. **Shared home store for claims.** Rule (4) claims were per-home:
   the two runtimes (userbot + personal bot, separate homes on one
   EC2 host) each claimed in their own database, so a non-game
   cross-runtime double in groups was still possible. The durable
   claim tables now live in ONE shared file both runtimes derive
   with no new config (``~/.nomorals/shared/claims.db``, overridable
   by ``NM_SHARED_CLAIMS``) — the same tables, the same atomic
   insert-or-nothing / compare-and-swap logic, the same
   never-raises contract. When the shared file cannot be opened,
   the ledgers fall back to the home database and arbitration
   FAILS TO DEFER (never to double-handle) on unproven standing.
   Trace finding: the shared file alone was NOT sufficient — the
   claim contract is per-SURFACE, and both WhatsApp runtimes
   claimed as the bare platform ``"whatsapp"``, so the second
   runtime's claim read as its own. The runtime now claims as
   ``platform:own-account-id`` (``whatsapp:<bot-jid>`` vs
   ``whatsapp:<owner-jid>``); the same-surface re-check still
   owns its claim, a twin surface stands down.

2. **Both-unknown late-double.** When both runtimes saw the group
   claim state as unknown, both could answer a harmless "no game
   is live here." — one reply too many. Trace finding: the shipped
   8s/3s asymmetry was PROBE-DEPENDENT, not deterministic — with
   the owner registry seeded, the bot-claim probe stamps the bot
   side during the turn (3s) while the owner side stays
   known-blocked (8s); with the registry unseeded the probe stays
   silent on BOTH sides and BOTH take 8s, both sweepers answering
   together (proven in the rig: 1 list + 1 list at +9s). The
   windows are now a deterministic role stagger: the bot surface
   answers first (3s); the owner surface waits (8s) and observes
   the answer in the room. The role needs no registry: a runtime
   is the owner surface when its own account id is the owner
   (registry, when seeded) or IS the command's sender. On any
   role-check failure the side waits slow — fail to defer, never
   to double-handle. Residual case (a group MEMBER's command with
   an unseeded registry — neither side is the sender): both take
   the fast window, and the shared failover claim still leaves
   exactly one reply.

3. **Repeats candidate (c): send-side double dispatch after a
   lost ack.** The WhatsApp send-once guard only noted an
   unconfirmed dispatch on the ``ambiguous`` response branch. An
   exception escaping ``_send_cmd`` (a write that may have
   landed, reply lost) propagated with no guard noted, so the
   gateway's classified retry re-dispatched the same text.
   PROVEN by the rig below (the exception path re-sent; the
   ambiguous path already held). The fix: any outcome-unknown
   failure in a send notes the guard and reports UNCONFIRMED,
   so the retry reconciles-or-holds instead of re-sending.

The law pinned here:

* Group standing reads ONE truth across runtimes; a runtime
  restart keeps it (durable file), a stale/partial read degrades
  to defer, never to a double.
* Exactly ONE reply for a both-unknown group game command,
  across two runtimes with separate homes.
* A lost ack never re-sends — text or media, adapter retry or
  gateway retry. A fresh solicited reply still releases the
  guard (the deliberate 2026-11-04 release).
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
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    MediaRef,
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
        ("groupdup2", "2000-01-01", time.time()))
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


def _texts(adapter):
    return [t for t, _b in adapter.sent]


def _group_chat() -> ChatRef:
    return ChatRef(platform="whatsapp", chat_id=GROUP_JID,
                   kind=ChatKind.GROUP, peer=GROUP_JID)


def _group_msg(text: str, message_id: str, *, sender_id=OWNER_JID,
               sender="death", is_owner=True, incoming=True,
               own=False) -> ChatMessage:
    meta: dict = {"is_owner": is_owner}
    if own:
        meta["own_message"] = True
        meta["from_owner"] = True
    msg = ChatMessage(chat=_group_chat(), incoming=incoming,
                      text=text, sender=sender,
                      message_id=message_id, meta=meta)
    msg.sender_id = sender_id
    return msg


def _mark_group_known_blocked(runtime) -> None:
    """The userbot's store learns the group with NO verified bot
    standing: known, unproven, system output NOT allowed."""
    surface = runtime._group_surface()
    surface.update_settings(GROUP_KEY, system_claim={})
    assert GROUP_KEY in surface.store.data
    assert surface.system_output_allowed(GROUP_KEY) is False


def _arbitrate(runtime, text, message_id, *, own=False):
    from nomorals.social.chat.control import parse_control

    command = parse_control(text)
    assert command is not None, text
    msg = _group_msg(text, message_id, own=own)
    return runtime._command_arbitration(msg, command)


@pytest.fixture()
def shared_claims_file(monkeypatch):
    """One temp shared-claims file for the test, via the env
    override both runtimes read — the per-test isolation the
    shared store needs (a real default-home file would leak
    claims between tests)."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-gd2-shared-",
                                      dir=_TEST_TMPDIR)
    path = str(Path(tmp.name) / "claims.db")
    monkeypatch.setenv("NM_SHARED_CLAIMS", path)
    yield path
    tmp.cleanup()


@pytest.fixture()
def broken_shared_claims(monkeypatch):
    """A shared-claims path that can never open: its parent is a
    regular file, so the directory cannot be created."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-gd2-broken-",
                                      dir=_TEST_TMPDIR)
    blocker = Path(tmp.name) / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(
        "NM_SHARED_CLAIMS", str(blocker / "claims.db"))
    yield str(blocker / "claims.db")
    tmp.cleanup()


@pytest.fixture()
def split_whatsapp(shared_claims_file):
    """Two runtimes, SEPARATE homes, ONE shared claims file —
    the live shape with the shared store in use."""
    yield from _make_split(seed=True)


@pytest.fixture()
def split_whatsapp_unseeded(shared_claims_file):
    """Two runtimes, separate homes, one shared claims file,
    owner registry UNSEEDED — the probe cannot stamp either
    side, so role (not probe state) must decide who answers."""
    yield from _make_split(seed=False)


@pytest.fixture()
def split_whatsapp_broken(broken_shared_claims):
    """Two runtimes, SEPARATE homes, shared store UNAVAILABLE —
    the fallback shape: arbitration must fail to defer, never
    to double-handle."""
    yield from _make_split(seed=True)


@pytest.fixture()
def split_whatsapp_broken_unseeded(broken_shared_claims):
    """Two runtimes, separate homes, shared store unavailable,
    owner registry UNSEEDED — the both-unknown case in its
    purest form: neither side can prove anything about the
    room, so the deferral windows must diverge by role alone."""
    yield from _make_split(seed=False)


def _make_split(*, seed: bool):
    clear_failures()
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-gd2-a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-gd2-b-",
                                        dir=_TEST_TMPDIR)
    ctx_a = _context(tmp_a.name)
    ctx_b = _context(tmp_b.name)
    ad_a = {"whatsapp": WAAdapter("whatsapp", BOT_JID)}
    ad_b = {"whatsapp": WAAdapter("whatsapp", OWNER_JID)}
    rt_a = _runtime(ctx_a, ad_a)
    rt_b = _runtime(ctx_b, ad_b)
    if seed:
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


# ── 1. the shared home store ──────────────────────────────────

def test_shared_command_ledger_arbitrates_across_handles(
        shared_claims_file):
    """Two Database handles on ONE shared file arbitrate — but
    ONLY when the claimants are distinct surfaces: the claim
    contract is per-surface (a surface re-checking its OWN
    claim still owns it). On WhatsApp both runtimes used to
    claim as the bare platform ``"whatsapp"``, so the second
    runtime's claim read as its own — the shared file alone
    could never arbitrate them. The runtime now claims as
    ``platform:own-account-id`` (``whatsapp:<bot-jid>`` vs
    ``whatsapp:<owner-jid>``), and this is the primitive the
    cross-runtime rule (4) rests on."""
    from nomorals.agents.partner.command_claims import DurableCommandClaimLedger

    db_a = Database(shared_claims_file)
    db_b = Database(shared_claims_file)
    try:
        led_a = DurableCommandClaimLedger(db_a)
        led_b = DurableCommandClaimLedger(db_b)
        key = "whatsapp:120363123456789@g.us:m1"
        bot_surface = f"whatsapp:{BOT_JID}"
        owner_surface = f"whatsapp:{OWNER_JID}"
        assert led_a.claim(key, bot_surface) is True
        # The same surface re-checking its own claim owns it.
        assert led_a.claim(key, bot_surface) is True
        # A twin surface in another handle stands down.
        assert led_b.claim(key, owner_surface) is False
        assert led_b.claimed_by(key) == bot_surface
        # An expired claim is stealable by the other handle.
        assert led_a.claim("other", bot_surface, now=1000.0) is True
        assert led_b.claim("other", owner_surface,
                           now=1000.0 + 601.0) is True
    finally:
        db_a.close()
        db_b.close()


def test_shared_group_claim_store_reads_one_truth(
        shared_claims_file):
    """Two group-claim handles on ONE shared file: a stamp by
    one runtime is the other runtime's standing."""
    from nomorals.agents.partner.command_claims import DurableGroupClaimStore

    db_a = Database(shared_claims_file)
    db_b = Database(shared_claims_file)
    try:
        store_a = DurableGroupClaimStore(db_a)
        store_b = DurableGroupClaimStore(db_b)
        assert store_b.bot_admin(GROUP_KEY) is None
        store_a.note_bot_admin(GROUP_KEY, True, bot_jid=BOT_JID)
        assert store_b.bot_admin(GROUP_KEY) is True
        assert store_b.record(GROUP_KEY)["bot_jid"] == BOT_JID
    finally:
        db_a.close()
        db_b.close()


def test_non_game_group_command_shared_store_exactly_one_reply(
        split_whatsapp):
    """The item-1 acceptance: a non-game group command on both
    runtimes, separate homes, one shared file — the first
    handles, the second stands down (rule 4, shared claim).
    Exactly ONE reply across both surfaces."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_whatsapp
    assert rt_a._claims_are_shared() is True
    assert rt_b._claims_are_shared() is True
    assert _arbitrate(rt_b, "/ping", "p1") == "handle"
    assert _arbitrate(rt_a, "/ping", "p1") == "down"
    rt_b._process_inner(_group_msg("/ping", "p1", own=True))
    rt_a._process_inner(_group_msg("/ping", "p1"))
    assert len(_texts(ad_b["whatsapp"])) == 1, _texts(ad_b["whatsapp"])
    assert "pong" in _texts(ad_b["whatsapp"])[0]
    assert _texts(ad_a["whatsapp"]) == []


def test_shared_store_group_standing_merges_across_homes(
        split_whatsapp):
    """The bot runtime's probe stamp lands in the shared file;
    the userbot runtime reads it as the room's standing and
    stands its non-game group commands down (rule 3)."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_whatsapp
    assert rt_b._group_surface().claim_state(
        GROUP_KEY)["bot_admin"] is None
    rt_a._group_surface().note_bot_admin(
        GROUP_KEY, True, bot_jid=BOT_JID)
    assert rt_b._group_surface().claim_state(
        GROUP_KEY)["bot_admin"] is True
    assert _arbitrate(rt_b, "/ping", "p1") == "down"


def test_shared_store_unavailable_defers_on_unproven(
        split_whatsapp_broken):
    """Fallback law: the shared file cannot open, so no runtime
    can prove the sibling's standing — a non-game group command
    on unproven standing DEFERS (watch the room, fail over if
    no answer) instead of handling blind on a per-home claim."""
    (rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = \
        split_whatsapp_broken
    assert rt_a._claims_are_shared() is False
    assert rt_b._claims_are_shared() is False
    _mark_group_known_blocked(rt_b)
    assert _arbitrate(rt_b, "/ping", "p1") == "defer"
    assert _arbitrate(rt_a, "/ping", "p1") == "defer"


def test_coincident_windows_shared_store_exactly_one_reply(
        split_whatsapp_unseeded):
    """The residual coincidence, closed by the shared claim:
    a GROUP MEMBER (not the owner) types /game in a group
    neither runtime knows, registry unseeded — the role rule
    cannot separate the sides (the sender is neither runtime's
    own account), so BOTH take the fast window and BOTH sweep
    at +4s. The shared failover claim still leaves exactly
    ONE list: the first claim wins, the second stands down."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = \
        split_whatsapp_unseeded
    member_jid = "2348099999999@s.whatsapp.net"
    baseline_a = len(_texts(ad_a["whatsapp"]))
    baseline_b = len(_texts(ad_b["whatsapp"]))
    rt_b._process_inner(_group_msg(
        "/game", "g1", sender_id=member_jid, sender="member",
        is_owner=False))
    rt_a._process_inner(_group_msg(
        "/game", "g1", sender_id=member_jid, sender="member",
        is_owner=False))
    assert _texts(ad_a["whatsapp"])[baseline_a:] == []
    assert _texts(ad_b["whatsapp"])[baseline_b:] == []
    rt_a._sweep_peer_deferrals(now=time.time() + 4.0)
    rt_b._sweep_peer_deferrals(now=time.time() + 4.0)
    replies = (_texts(ad_a["whatsapp"])[baseline_a:]
               + _texts(ad_b["whatsapp"])[baseline_b:])
    assert len(replies) == 1, replies
    # Unseeded, so the probe stamped neither side and the room
    # is not a game room: the one reply is the "no game is live
    # here" refusal — the exact text of the owner's incident.
    assert "games aren't running in this group" in replies[0]


def test_shared_store_restart_keeps_standing(shared_claims_file):
    """Restart survival: a fresh handle on the same shared file
    reads the standing the pre-restart runtime stamped."""
    from nomorals.agents.partner.command_claims import DurableGroupClaimStore

    db_a = Database(shared_claims_file)
    store_a = DurableGroupClaimStore(db_a)
    store_a.note_bot_admin(GROUP_KEY, True, bot_jid=BOT_JID)
    db_a.close()
    db_b = Database(shared_claims_file)
    try:
        assert DurableGroupClaimStore(db_b).bot_admin(
            GROUP_KEY) is True
    finally:
        db_b.close()


# ── 2. both-unknown late-double: the role stagger ─────────────

def test_both_unknown_game_defers_with_role_stagger(
        split_whatsapp_broken_unseeded):
    """Neither runtime knows the group at all AND the owner
    registry is unseeded, so the bot-claim probe stays silent
    on both sides — the pre-fix shape was BOTH windows 8s
    (known-blocked on both) and both sweepers answering at
    +9s. The windows now diverge BY ROLE: the bot surface
    fails over fast (3s) and answers; the owner surface waits
    slow (8s) and observes. The owner side is slow because
    the command's sender IS its own account — no registry
    needed to tell the sides apart."""
    (rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = \
        split_whatsapp_broken_unseeded
    assert _arbitrate(rt_a, "/game", "g1") == "defer"
    assert _arbitrate(rt_b, "/game", "g1") == "defer"
    rt_a._process_inner(_group_msg("/game", "g1"))
    rt_b._process_inner(_group_msg("/game", "g1", own=True))
    assert rt_a._peer_deferral().pending_count() == 1
    assert rt_b._peer_deferral().pending_count() == 1
    # At +4s the bot side is due; the owner side is not.
    assert len(rt_a._peer_deferral().expire(
        now=time.time() + 4.0)) == 1
    assert rt_b._peer_deferral().expire(
        now=time.time() + 4.0) == []
    assert rt_b._peer_deferral().pending_count() == 1
    # At +9s the owner side is due (failover, bot down).
    assert len(rt_b._peer_deferral().expire(
        now=time.time() + 9.0)) == 1


def test_both_unknown_game_exactly_one_reply(
        split_whatsapp_broken_unseeded):
    """The item-2 acceptance, end to end, in the fallback shape
    (shared store unavailable AND registry unseeded — the
    stagger is load-bearing here): both runtimes defer a group
    /game neither knows; both sweepers fire at +4s; exactly ONE
    list lands, from the bot side. The owner side observes it
    and never speaks."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = \
        split_whatsapp_broken_unseeded
    baseline_a = len(_texts(ad_a["whatsapp"]))
    baseline_b = len(_texts(ad_b["whatsapp"]))
    rt_b._process_inner(_group_msg("/game", "g1", own=True))
    rt_a._process_inner(_group_msg("/game", "g1"))
    assert _texts(ad_a["whatsapp"])[baseline_a:] == []
    assert _texts(ad_b["whatsapp"])[baseline_b:] == []
    # Both sweepers fire at +4s — before the fix the owner side
    # failed over here too and answered a second list.
    rt_a._sweep_peer_deferrals(now=time.time() + 4.0)
    bot_replies = _texts(ad_a["whatsapp"])[baseline_a:]
    assert len(bot_replies) == 1, bot_replies
    # Unseeded, so the probe stamped neither side and the room
    # is not a game room: the one reply is the "no game is live
    # here" refusal — the exact text of the owner's incident.
    assert "games aren't running in this group" in bot_replies[0]
    rt_b._sweep_peer_deferrals(now=time.time() + 4.0)
    assert _texts(ad_b["whatsapp"])[baseline_b:] == []
    # The bot's answer, observed by the userbot runtime as the
    # bot's own post, resolves the owner side's deferral.
    rt_b._process_inner(_group_msg(
        bot_replies[0], "g2", sender_id=BOT_JID))
    assert _texts(ad_b["whatsapp"])[baseline_b:] == []
    assert rt_b._peer_deferral().pending_count() == 0
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    assert _texts(ad_b["whatsapp"])[baseline_b:] == []


# ── 3. repeats candidate (c): lost-ack re-send ────────────────

def _wa_adapter(tmp: str, responder):
    """Lane rig (test_groupdup_fix): a connected adapter whose
    bridge is a recorder."""
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
    if cmd.get("cmd") == "history":
        return {"ok": True, "messages": []}
    return {"ok": True, "message_id": "M1",
            "buttons_form": "interactive"}


def _wa_chat() -> ChatRef:
    return ChatRef(platform="whatsapp",
                   chat_id="2348012345678@c.us", kind=ChatKind.DM)


def _send_cmds(calls, cmd: str) -> list[dict[str, Any]]:
    return [c for c in calls if c.get("cmd") == cmd]


def test_lost_ack_ambiguous_holds_repeat():
    """Control (already green): an ambiguous dispatch is never
    re-dispatched — the guard holds the repeat, history finds
    nothing, the send stays held."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-gd2-wa-",
                                      dir=_TEST_TMPDIR)
    try:
        def responder(cmd):
            if cmd.get("cmd") == "history":
                return {"ok": True, "messages": []}
            return {"ok": False, "error": "bridge response timeout",
                    "ambiguous": True}

        adapter, calls = _wa_adapter(tmp.name, responder)
        first = adapter.send(_wa_chat(), "microsoft loop")
        assert first.ok is False
        assert "unconfirmed" in (first.error or "")
        second = adapter.send(_wa_chat(), "microsoft loop")
        assert second.ok is False
        assert "unconfirmed" in (second.error or "")
        assert len(_send_cmds(calls, "send")) == 1, calls
    finally:
        tmp.cleanup()


def test_lost_ack_exception_holds_repeat():
    """PROOF of candidate (c): an exception escaping _send_cmd —
    the write may have landed, the reply is lost — propagated
    with NO guard noted, so the next identical send dispatched
    again. After the fix the exception notes the guard and
    reports UNCONFIRMED, and the repeat is held."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-gd2-wa-",
                                      dir=_TEST_TMPDIR)
    try:
        def responder(cmd):
            if cmd.get("cmd") == "history":
                return {"ok": True, "messages": []}
            raise TimeoutError("timed out")

        adapter, calls = _wa_adapter(tmp.name, responder)
        first = adapter.send(_wa_chat(), "microsoft loop")
        assert first.ok is False
        assert "unconfirmed" in (first.error or ""), first
        second = adapter.send(_wa_chat(), "microsoft loop")
        assert second.ok is False
        assert "unconfirmed" in (second.error or ""), second
        assert len(_send_cmds(calls, "send")) == 1, calls
    finally:
        tmp.cleanup()


def test_lost_ack_exception_gateway_retry_sends_once():
    """End to end through the gateway's classified retry: the
    first dispatch raises, the gateway retries — the guard must
    absorb every retry, so the bridge sees ONE send attempt."""
    from nomorals.social.chat.send_retry import reset_retry_budgets, send_with_retry

    tmp = tempfile.TemporaryDirectory(prefix="nm-gd2-wa-",
                                      dir=_TEST_TMPDIR)
    try:
        def responder(cmd):
            if cmd.get("cmd") == "history":
                return {"ok": True, "messages": []}
            raise TimeoutError("timed out")

        adapter, calls = _wa_adapter(tmp.name, responder)
        reset_retry_budgets()
        result, attempts = send_with_retry(
            lambda: adapter.send(_wa_chat(), "microsoft loop"),
            platform="whatsapp", sleep=lambda _s: None)
        assert result.ok is False
        assert "unconfirmed" in (result.error or ""), result
        assert attempts == 3, attempts
        assert len(_send_cmds(calls, "send")) == 1, calls
    finally:
        tmp.cleanup()


def test_lost_ack_media_exception_holds_repeat():
    """The same hole on the media path: an exception escaping
    _send_cmd in send_media propagated with no guard noted.
    After the fix it reports UNCONFIRMED and the repeat holds."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-gd2-wa-",
                                      dir=_TEST_TMPDIR)
    try:
        media_path = str(Path(tmp.name) / "shot.jpg")
        Path(media_path).write_bytes(b"\xff\xd8\xff\xe0fake")

        def responder(cmd):
            if cmd.get("cmd") == "history":
                return {"ok": True, "messages": []}
            raise TimeoutError("timed out")

        adapter, calls = _wa_adapter(tmp.name, responder)
        chat = _wa_chat()
        media = MediaRef(kind="image", path=media_path)
        first = adapter.send_media(chat, media, caption="look")
        assert first.ok is False
        assert "unconfirmed" in (first.error or ""), first
        second = adapter.send_media(chat, media, caption="look")
        assert second.ok is False
        assert "unconfirmed" in (second.error or ""), second
        assert len(_send_cmds(calls, "send_media")) == 1, calls
    finally:
        tmp.cleanup()


def test_send_cmd_marks_mid_write_transport_failure_ambiguous():
    """The second hole in the same family, at the _send_cmd
    layer: a ConnectionError out of sendall (BrokenPipe, reset)
    used to arrive as a PLAIN error dict — the send paths read
    it as a clean failure and retried it. Only the pre-write
    "not connected" (no socket at all) is a known non-send;
    anything else from the write is ambiguous, so the send
    paths guard it."""
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    tmp = tempfile.TemporaryDirectory(prefix="nm-gd2-wa-",
                                      dir=_TEST_TMPDIR)
    try:
        adapter = WhatsAppAdapter(
            outbox_dir=str(Path(tmp.name) / "outbox"))

        class BrokenSock:
            def sendall(self, data):
                raise BrokenPipeError(32, "Broken pipe")

        adapter._sock = BrokenSock()
        failed = adapter._send_cmd({"cmd": "send", "chat": "x",
                                    "text": "hi"})
        assert failed.get("ok") is False
        assert failed.get("ambiguous") is True
        adapter._sock = None
        known = adapter._send_cmd({"cmd": "send", "chat": "x",
                                   "text": "hi"})
        assert known.get("ok") is False
        assert known.get("error") == "whatsapp bridge not connected"
        assert not known.get("ambiguous")
    finally:
        tmp.cleanup()


def test_fresh_solicited_reply_releases_guard():
    """The deliberate release is untouched: a FRESH solicited
    reply for a live turn releases a guarded identical send
    and dispatches (the 2026-11-04 flood-then-silence lesson)."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-gd2-wa-",
                                      dir=_TEST_TMPDIR)
    try:
        state = {"calls": 0}

        def responder(cmd):
            if cmd.get("cmd") == "history":
                return {"ok": True, "messages": []}
            state["calls"] += 1
            if state["calls"] == 1:
                return {"ok": False,
                        "error": "bridge response timeout",
                        "ambiguous": True}
            return {"ok": True, "message_id": "M2",
                    "buttons_form": "interactive"}

        adapter, calls = _wa_adapter(tmp.name, responder)
        first = adapter.send(_wa_chat(), "microsoft loop")
        assert first.ok is False
        assert "unconfirmed" in (first.error or "")
        fresh = adapter.send(_wa_chat(), "microsoft loop",
                             fresh=True)
        assert fresh.ok is True, fresh
        assert len(_send_cmds(calls, "send")) == 2, calls
    finally:
        tmp.cleanup()
