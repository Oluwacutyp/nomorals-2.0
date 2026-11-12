"""Telegram GROUP + DM-command duplication (dupfix4) — 2026-11-12.

DM plain text was fixed live by dupfix3; Telegram groups still
double, and DM slash commands still double (owner screenshot
2026-11-12 ~12:34: /play answered twice with the same pick list —
userbot green bubble + bot white bubble). The diagnosis
(groupdup3-diag-verify/DIAGNOSIS.md) proved the group half has
NONE of the DM fix's load-bearing parts, and the /play trace proved
the DM command half escapes through the async answer: the 20s
deferral expires before a ~2min /play answer lands, the ack/edit
observation never resolves it (empty companions + the edit dedupe),
and the sweep backstop fails when the keys do not collide or the
store is unshared.

The fixes pinned here, in order:
  (1) unconditional group peer recognition — a post from one of
      Devon's own surfaces is an observation for EVERY group post;
  (2) plain-text defer-and-observe for groups when the shared
      conversational claim is unavailable;
  (3) the bot registry re-syncs on group traffic;
  (4) group surface sends are gated through the shared claim;
  (5) the shared-claims open retries (bounded) instead of caching
      a transient failure for the runtime's life;
  (D1) DM deferral-context recognition — a pending deferral
      certifies the chat is a bot DM, so any non-owner sender
      while it is pending IS the primary's answer;
  (D2) a recognised bot EDIT bypasses the adapter dedupe, so the
      pick-list edit can be observed as a peer post;
  (D4) the async /play answer re-checks the command claim at SEND
      time, not only at intake.
"""
from __future__ import annotations

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
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888
GROUP_ID = "-100123456789"
CANNED = "what should I post? Send the text first, then say 'post this'."
MEMBER_ID = "777"

# Divergent Path.home() holder for the split-homes test: the shim
# below reads this dict, so the test can flip homes between intakes.
_DIVERGENT_HOME: dict[str, Any] = {"home": None}


class _DivergentHomePath:
    """pathlib.Path whose home() is holder-controlled.

    Only home() is faked: construction delegates to the real Path,
    which is all command_claims.py ever uses (Path.home() at the
    shared-path derivation)."""

    def __new__(cls, *args: Any, **kwargs: Any):
        return Path(*args, **kwargs)

    @classmethod
    def home(cls):
        return _DIVERGENT_HOME["home"]


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


class BotAdapter(RecordingAdapter):
    def __init__(self, name: str, bot_id: int, username: str) -> None:
        super().__init__(name)
        self._bot_id = bot_id
        self._bot_username = username


class BareUserbotAdapter(RecordingAdapter):
    """The split-hosting userbot: its own account identity, and NO
    companion bot ids — the documented live shape."""

    _me_username = "peacethefirst"
    _me_id = int(OWNER_ID)


class NoteTakingUserbotAdapter(BareUserbotAdapter):
    """Bare userbot that records registry pushes, like the real
    TelegramAdapter.note_companion_bot_ids."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pushed: list[set] = []

    def note_companion_bot_ids(self, ids) -> bool:
        self.pushed.append(set(ids))
        return True


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
        ("dupfix4", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    # The owner registry seeds from this attribute: the live shape
    # knows the owner's Telegram id, so the peer grounds that read
    # it (fix 1 ground (ii)) can fire in these tests.
    runtime._owner_sender_ids = (OWNER_ID,)
    return runtime


def _msg(platform, *, chat_id, text, incoming, message_id,
         sender_id=OWNER_ID, own=False, peer_is_bot=False,
         service=None, reply_to="", kind=ChatKind.DM):
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                   peer=chat_id)
    meta: dict = {"is_owner": True}
    if own:
        meta["own_message"] = True
        meta["from_owner"] = True
    if peer_is_bot:
        meta["peer_is_bot"] = True
    if service:
        meta["service"] = service
        meta["observe_only"] = True
    if reply_to:
        meta["reply_to"] = reply_to
    msg = ChatMessage(chat=chat, incoming=incoming, text=text,
                      sender="death", message_id=message_id, meta=meta,
                      reply_to=reply_to)
    msg.sender_id = sender_id
    return msg


def _bot_copy(text, message_id):
    """The Bot API view of the owner's message in the personal
    bot's DM."""
    return _msg("telegram-personal", chat_id=OWNER_ID, text=text,
                incoming=True, message_id=message_id)


def _echo_copy(text, message_id, *, stamped=True):
    """The MTProto own-echo view of the same message."""
    return _msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=text,
                incoming=False, message_id=message_id, own=True,
                peer_is_bot=stamped)


def _peer_post(text, message_id, *, reply_to="", service="peer_post",
               chat_id=str(PERSONAL_BOT_ID), kind=ChatKind.DM):
    """The bot's own post in that chat, as the userbot stream sees
    it — delivered as an observation."""
    return _msg("telegram", chat_id=chat_id, text=text,
                incoming=True, message_id=message_id,
                sender_id=str(PERSONAL_BOT_ID), service=service,
                reply_to=reply_to, kind=kind)


def _group_msg(platform, text, message_id, *, sender_id=OWNER_ID):
    """The same group message as one surface sees it (platform
    differs per surface; chat id, sender and message id agree)."""
    return _msg(platform, chat_id=GROUP_ID, text=text,
                incoming=True, message_id=message_id,
                sender_id=sender_id, kind=ChatKind.GROUP)


def _texts(adapter):
    return [t for t, _b in adapter.sent]


def _make_pair(monkeypatch, shared_path=None, note_taking=False):
    """Two runtimes, separate homes. shared_path=None leaves
    NM_SHARED_CLAIMS unset (the caller controls the store)."""
    clear_failures()
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-d4-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    if shared_path is None:
        monkeypatch.delenv("NM_SHARED_CLAIMS", raising=False)
    else:
        monkeypatch.setenv("NM_SHARED_CLAIMS", shared_path)
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-d4a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-d4b-",
                                        dir=_TEST_TMPDIR)
    ctx_a = _context(tmp_a.name)
    ctx_b = _context(tmp_b.name)
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    userbot_cls = (NoteTakingUserbotAdapter if note_taking
                   else BareUserbotAdapter)
    ad_b = {"telegram": userbot_cls("telegram")}
    rt_a = _runtime(ctx_a, ad_a)
    rt_b = _runtime(ctx_b, ad_b)
    return (rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b), (tmp_a, tmp_b,
                                                      home_tmp)


def _teardown_pair(pair):
    (rt_a, _ad_a, ctx_a), (rt_b, _ad_b, ctx_b) = pair
    for runtime, context in ((rt_a, ctx_a), (rt_b, ctx_b)):
        try:
            runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        context.close()


@pytest.fixture()
def split_homes(monkeypatch):
    """Two runtimes, SEPARATE homes, one shared claims file."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-d4-shared-",
                                      dir=_TEST_TMPDIR)
    path = str(Path(tmp.name) / "claims.db")
    (rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b), tmps = _make_pair(
        monkeypatch, shared_path=path)
    yield (rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b)
    _teardown_pair(((rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b)))
    for tmp_dir in tmps:
        tmp_dir.cleanup()
    tmp.cleanup()


@pytest.fixture()
def split_homes_broken_store(monkeypatch):
    """Same split homes, but the shared claims file cannot open:
    arbitration must fail to defer, never to double-handle. The
    blocker path is unique per fixture (no fixed-name clash)."""
    blocker_dir = tempfile.mkdtemp(prefix="nm-d4-blocker-",
                                   dir=_TEST_TMPDIR)
    blocker = os.path.join(blocker_dir, "claims-blocker")
    with open(blocker, "w", encoding="utf-8") as fh:
        fh.write("not a directory\n")
    path = os.path.join(blocker, "claims.db")
    (rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b), tmps = _make_pair(
        monkeypatch, shared_path=path)
    yield ((rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b),
           {"blocker": blocker, "blocker_dir": blocker_dir})
    _teardown_pair(((rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b)))
    for tmp_dir in tmps:
        tmp_dir.cleanup()


@pytest.fixture()
def split_homes_divergent(monkeypatch):
    """Two runtimes whose Path.home() DIVERGES and NO
    NM_SHARED_CLAIMS override — the live shape: each runtime's
    shared claims file is a different file, so the store never
    meets. The test flips holder["home"] between intakes."""
    home_a = Path(tempfile.mkdtemp(prefix="nm-d4-ha-",
                                   dir=_TEST_TMPDIR))
    home_b = Path(tempfile.mkdtemp(prefix="nm-d4-hb-",
                                   dir=_TEST_TMPDIR))
    _DIVERGENT_HOME["home"] = home_a
    import nomorals.agents.partner.command_claims as _cc

    monkeypatch.setattr(_cc, "Path", _DivergentHomePath)
    (rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b), tmps = _make_pair(
        monkeypatch, shared_path=None)
    holder = {"home_a": home_a, "home_b": home_b}
    yield ((rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b), holder)
    _teardown_pair(((rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b)))
    for tmp_dir in tmps:
        tmp_dir.cleanup()
    _DIVERGENT_HOME["home"] = None


# ── fix 1: unconditional group peer recognition ───────────────────────

def test_group_pingpong_owner_post_held_by_live_claim(split_homes):
    """Fix 1, ground (ii): the bot runtime answers a member's group
    turn (claiming it), and the userbot's answer — arriving as the
    owner's account posting the same words — is an OBSERVATION,
    never a second turn. One reply total on the bot runtime.

    Failing-first: before the fix the owner post flows to the
    override and the bot answers it again (2 replies)."""
    (rt_a, ad_a, _ctx_a), (_rt_b, _ad_b, _ctx_b) = split_homes
    rt_a._process_inner(_group_msg("telegram-personal", "post this",
                                   "800", sender_id=MEMBER_ID))
    assert len(_texts(ad_a["telegram-personal"])) == 1
    # The sibling's answer, as the bot runtime sees it: the owner's
    # account posting in the group (the loop's green copy carries
    # the owner's exact words).
    rt_a._process_inner(_group_msg("telegram-personal", "post this",
                                   "801", sender_id=OWNER_ID))
    assert len(_texts(ad_a["telegram-personal"])) == 1, \
        _texts(ad_a["telegram-personal"])


def test_group_admin_member_message_stays_a_turn(split_homes):
    """Fix 1 must NOT eat an admin member's message: the roster
    ground is deliberately NOT in the unconditional set — only a
    pending deferral may consult it. A member's group turn still
    gets exactly one answer."""
    (rt_a, ad_a, _ctx_a), (_rt_b, _ad_b, _ctx_b) = split_homes
    rt_a._process_inner(_group_msg("telegram-personal", "post this",
                                   "810", sender_id=MEMBER_ID))
    assert len(_texts(ad_a["telegram-personal"])) == 1


def test_group_fresh_owner_message_without_claim_stays_a_turn(
        split_homes):
    """Fix 1 ground (ii) needs a LIVE claim: the owner's fresh
    group message, with no turn claimed in the chat, is a turn —
    the runtime answers it. No over-blocking."""
    (rt_a, ad_a, _ctx_a), (_rt_b, _ad_b, _ctx_b) = split_homes
    rt_a._process_inner(_group_msg("telegram-personal", "post this",
                                   "820", sender_id=OWNER_ID))
    assert len(_texts(ad_a["telegram-personal"])) == 1


# ── fix 2: plain-text defer-and-observe for groups ───────────────────

def test_group_plain_text_defers_when_claim_unavailable(
        split_homes_broken_store):
    """Fix 2: with the shared store unreachable, a plain-text group
    turn DEFERS instead of answering blind — mirroring the command
    path. The sibling's answer in the room then resolves it.

    Failing-first: before the fix the turn answers at once
    (1 reply, 0 pending)."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b), _info = \
        split_homes_broken_store
    assert rt_b._claims_are_shared() is False
    rt_b._process_inner(_group_msg("telegram", "post this", "830",
                                   sender_id=MEMBER_ID))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 1
    rt_b._process_inner(_peer_post(
        CANNED, "831", reply_to="830", chat_id=GROUP_ID,
        kind=ChatKind.GROUP))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 0
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    assert _texts(ad_b["telegram"]) == []


def test_group_plain_text_failover_answers_once(
        split_homes_broken_store):
    """Fix 2, failover: no sibling answer lands, so the sweep
    re-queues the deferred text and it answers ONCE — the
    failover re-queue never re-defers.

    Failing-first: before the fix the turn answers at once, so
    there is already 1 reply before the sweep ever runs."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b), _info = \
        split_homes_broken_store
    rt_b._process_inner(_group_msg("telegram", "post this", "840",
                                   sender_id=MEMBER_ID))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 1
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    deadline = time.time() + 10.0
    while (len(_texts(ad_b["telegram"])) == 0
           and time.time() < deadline):
        time.sleep(0.05)
    assert len(_texts(ad_b["telegram"])) == 1, \
        _texts(ad_b["telegram"])
    assert "what should I post?" in _texts(ad_b["telegram"])[0]


# ── split homes, divergent Path.home(), no override ──────────────────

def test_split_homes_divergent_no_runaway(split_homes_divergent):
    """The live shape the old fixtures never exercised: divergent
    Path.home() and NO NM_SHARED_CLAIMS override, so each runtime's
    shared claims file is a different file and the store never
    meets. The first double is unavoidable there (each runtime
    claims in its own file) — but the loop TERMINATES: the bot
    runtime holds a live claim in its own ledger, so the sibling's
    answer is observed, never a third reply.

    Failing-first: before fix 1 the owner post is answered again
    (2 replies on the bot runtime)."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b), holder = \
        split_homes_divergent
    # rt_a claims in home_A's file.
    rt_a._process_inner(_group_msg("telegram-personal", "post this",
                                   "850", sender_id=MEMBER_ID))
    assert len(_texts(ad_a["telegram-personal"])) == 1
    # Flip homes: rt_b's store is a different file.
    _DIVERGENT_HOME["home"] = holder["home_b"]
    rt_b._process_inner(_group_msg("telegram", "post this", "850",
                                   sender_id=MEMBER_ID))
    assert len(_texts(ad_b["telegram"])) == 1, \
        _texts(ad_b["telegram"])
    # The sibling's answer arrives on rt_a as the owner's account
    # posting: observed, never a third reply.
    rt_a._process_inner(_group_msg("telegram-personal", "post this",
                                   "851", sender_id=OWNER_ID))
    assert len(_texts(ad_a["telegram-personal"])) == 1, \
        _texts(ad_a["telegram-personal"])
    assert len(_texts(ad_b["telegram"])) == 1


# ── fix 3: registry re-sync on group traffic ─────────────────────────

def test_group_registry_resync_on_group_traffic(monkeypatch):
    """Fix 3: the bot runtime registers its id, and the userbot
    runtime picks it up by processing ONE group message — the group
    intake path re-syncs the registry, so the companion set cannot
    stay empty for a runtime's life after a boot-order miss.

    Failing-first: before the fix the group intake never syncs, so
    the adapter receives no push at all."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-d4-shared-",
                                      dir=_TEST_TMPDIR)
    path = str(Path(tmp.name) / "claims.db")
    (rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b), tmps = _make_pair(
        monkeypatch, shared_path=path, note_taking=True)
    try:
        rt_a._sync_bot_registry()
        rt_b._process_inner(_group_msg("telegram", "post this", "860",
                                       sender_id=MEMBER_ID))
        assert ad_b["telegram"].pushed, \
            "no registry push on group traffic"
        pushed = set().union(*ad_b["telegram"].pushed)
        assert str(PERSONAL_BOT_ID) in {str(x) for x in pushed}
    finally:
        _teardown_pair(((rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b)))
        for tmp_dir in tmps:
            tmp_dir.cleanup()
        tmp.cleanup()


# ── D1: DM deferral-context recognition ─────────────────────────────

def test_dm_command_ack_observed_without_registry(
        split_homes_broken_store):
    """D1: the userbot defers a DM command — the deferral registers
    on the echo's peer stamp, while the shared store is unreachable
    so the runtime registry stays empty. The bot's ack then arrives
    as an ORDINARY incoming DM message (no peer_post stamp, no
    stamp, no registry, no shared store — the live gap). A pending
    deferral certifies the chat is a bot DM, so the non-owner
    sender IS the primary's answer: the deferral resolves, the
    userbot never fails over into a second copy.

    Failing-first: before the fix the ack becomes a turn (the
    deferral stays pending) and the sweep fails over into a
    second /chatmode answer."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b), _info = \
        split_homes_broken_store
    assert rt_b._claims_are_shared() is False
    text = "/chatmode games-only +2348164302080"
    # Stamped echo: the peer is identified at intake (the stamp),
    # so the deferral registers even with an empty registry.
    rt_b._process_inner(_echo_copy(text, "870", stamped=True))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 1
    # The bot's ack, as an ordinary incoming DM message from the
    # bot's id — no peer_post service, no stamp.
    ack = _msg("telegram", chat_id=str(PERSONAL_BOT_ID),
               text="whatsapp:2348164302080@s.whatsapp.net is now "
               "games-only", incoming=True, message_id="871",
               sender_id=str(PERSONAL_BOT_ID))
    rt_b._process_inner(ack)
    assert rt_b._peer_deferral().pending_count() == 0
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    assert _texts(ad_b["telegram"]) == []


def test_dm_command_sweep_stands_down_when_bot_claimed(split_homes):
    """D3 backstop (regression pin — passes before the fix too):
    the bot claimed the command key at intake, so the userbot's
    sweep stands down instead of failing over into a second copy."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_a._sync_bot_registry()
    rt_a._process_inner(_bot_copy("/ping", "880"))
    assert len(_texts(ad_a["telegram-personal"])) == 1
    rt_b._process_inner(_echo_copy("/ping", "880", stamped=False))
    assert rt_b._peer_deferral().pending_count() == 1
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    assert _texts(ad_b["telegram"]) == []


# ── fix 4: group surface sends gated through the shared claim ────────

def test_system_send_claim_gate(split_homes):
    """Fix 4: the same system notice, claimed by both runtimes —
    the first owns the send, the second stands down. A different
    text (or a different kind) is a different send.

    Failing-first: before the fix _claim_system_send does not
    exist at all."""
    (rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_homes
    chat = ChatRef(platform="telegram-personal", chat_id=GROUP_ID,
                   kind=ChatKind.GROUP)
    assert rt_a._claim_system_send(chat, "notice", "hello room") is True
    assert rt_b._claim_system_send(chat, "notice", "hello room") is False
    assert rt_b._claim_system_send(chat, "notice", "other text") is True
    assert rt_b._claim_system_send(chat, "ask", "hello room") is True


# ── fix 5: store-open retry ──────────────────────────────────────────

def test_shared_claims_db_retries_after_transient_failure(
        split_homes_broken_store):
    """Fix 5: a first open that fails does NOT poison the runtime
    for life. The retry is bounded (a second call inside the
    window does not re-attempt) and a late success re-opens the
    store — and invalidates the ledgers cached against the home
    store, so the next claim rebuilds against the shared file.

    Failing-first: before the fix the failed open is remembered
    forever and the store never opens, however long the runtime
    lives."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b), info = \
        split_homes_broken_store
    assert rt_b._shared_claims_db() is None
    assert rt_b._claims_are_shared() is False
    # Bounded: an immediate second call does not re-attempt (the
    # blocker file is still there, and no shared file appears).
    assert rt_b._shared_claims_db() is None
    # The home ledger is cached against the home store by now.
    rt_b._command_claims()
    assert rt_b._command_claims_obj is not None
    # The transient fault clears: the blocker file goes away and
    # its parent becomes a real directory, so the path opens.
    os.unlink(info["blocker"])
    os.makedirs(info["blocker"], exist_ok=True)
    rt_b._shared_claims_db_failed_at = time.time() - 61.0
    assert rt_b._shared_claims_db() is not None
    assert rt_b._claims_are_shared() is True
    assert rt_b._command_claims_obj is None


# ── D4: the async /play answer re-checks the claim at SEND time ──────

def test_command_claim_lost_across_surfaces(split_homes):
    """D4 predicate: a command key claimed by the bot runtime at
    intake reads as LOST on the userbot (another surface holds
    it) and as HELD on the bot runtime itself. No key, or no
    holder, reads as held (fail open).

    Failing-first: before the fix _command_claim_lost does not
    exist at all."""
    (rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_homes
    msg = _bot_copy("/ping", "890")
    key = rt_a._command_claim_key_for(msg)
    assert key, "the /ping message must be keyable"
    chat = msg.chat
    assert rt_b._command_claim_lost(key, chat) is False
    assert rt_a._command_claims().claim(
        key, rt_a._claim_surface("telegram-personal")) is True
    assert rt_b._command_claim_lost(key, chat) is True
    assert rt_a._command_claim_lost(key, chat) is False
    assert rt_b._command_claim_lost("", chat) is False


def test_play_download_bg_stands_down_when_sibling_claims(split_homes):
    """D4, end to end at the job: the bot runtime claimed the /play
    command key at intake; the userbot runtime's download job for
    the same command must NOT produce (send) — it stands down at
    send time. And the control: a key this runtime holds itself
    still produces.

    Failing-first: before the fix there is no send-time check, so
    the sibling-held job produces and sends the second copy."""
    from nomorals.core.jobs import shared_registry

    (rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    msg = _bot_copy("/ping", "900")
    key = rt_a._command_claim_key_for(msg)
    assert rt_a._command_claims().claim(
        key, rt_a._claim_surface("telegram-personal")) is True
    chat = ChatRef(platform="telegram", chat_id=OWNER_ID,
                   kind=ChatKind.DM)
    produced: list[str] = []

    def _produce():
        produced.append("ran")
        return ("", False)

    rt_b._play_download_bg(chat, "telegram:4242", "⏳ test track",
                           _produce, title="d4-skip-a",
                           claim_key=key)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        done = [s for s in shared_registry().list()
                if (s.get("labels") or {}).get("title") == "d4-skip-a"
                and s.get("status") == "done"]
        if done:
            break
        time.sleep(0.05)
    assert produced == [], "the sibling-held job must not produce"
    # Control: a key this runtime holds itself still produces.
    key_b = rt_b._command_claim_key_for(_bot_copy("/ping", "901"))
    assert rt_b._command_claims().claim(
        key_b, rt_b._claim_surface("telegram")) is True
    produced_b: list[str] = []

    def _produce_b():
        produced_b.append("ran")
        return ("", False)

    rt_b._play_download_bg(chat, "telegram:4242", "⏳ test track",
                           _produce_b, title="d4-skip-b",
                           claim_key=key_b)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        done = [s for s in shared_registry().list()
                if (s.get("labels") or {}).get("title") == "d4-skip-b"
                and s.get("status") == "done"]
        if done:
            break
        time.sleep(0.05)
    assert produced_b == ["ran"], \
        "a self-held key must still produce"


def test_play_dispatch_sets_claim_var_without_changing_call_shape(
        split_homes):
    """D4 wiring: the /play dispatch sets play_claim_var for the
    handler to read, WITHOUT changing _control_play's call shape —
    a two-argument replacement handler must keep working (the
    keyword-threaded first version broke
    test_split_instances_effect_runs_once exactly this way).

    Failing-first: before the fix there is no var at all, so the
    handler reads ""."""
    from nomorals.agents.partner.runtime_media import play_claim_var

    (rt_a, _ad_a, _ctx_a), (_rt_b, _ad_b, _ctx_b) = split_homes
    seen: list = []

    def fake_play(tail, chat_key=""):
        seen.append((tail, chat_key, play_claim_var.get("")))
        return "queued"

    rt_a._control_play = fake_play  # type: ignore[method-assign]
    rt_a._process_inner(_msg(
        "telegram-personal", chat_id=OWNER_ID, text="/play fear by nf",
        incoming=True, message_id="910"))
    assert len(seen) == 1, seen
    assert seen[0][2], "the dispatch must set the claim key var"
    assert play_claim_var.get("") == "", \
        "the var must be reset after dispatch"


# ── D2: a recognised bot EDIT is an observation, not a replay ────────

def _inbound_event(chat_id, sender_id, mid, text):
    return SimpleNamespace(
        chat_id=chat_id, sender_id=sender_id, sender=None,
        message=SimpleNamespace(id=mid, raw_text=text,
                                sender_id=sender_id))


def test_adapter_bot_edit_bypasses_dedupe(tmp_path):
    """D2: the bot's ack arrives (observed), then the pick-list
    EDIT of the same message id arrives. The edit is a new version,
    not a replay — a recognised bot edit is delivered as a peer_post
    observation instead of being swallowed by the at-most-once
    dedupe.

    Failing-first: before the fix the edit is dropped (1 delivery,
    never 2)."""
    import asyncio

    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=1, api_hash="x", session_path=str(tmp_path / "t.session"),
        media_dir=str(tmp_path / "media"),
        companion_bot_ids={PERSONAL_BOT_ID})
    delivered: list[ChatMessage] = []

    def handler(message):
        delivered.append(message)

    asyncio.run(adapter._handle_inbound(
        _inbound_event(GROUP_ID, PERSONAL_BOT_ID, 55, "pong"),
        None, handler, allow_check=True))
    assert len(delivered) == 1
    edit = _inbound_event(GROUP_ID, PERSONAL_BOT_ID, 55,
                          "pick:9f2ccbd3\n1. lifestyle (YA MAN)")
    edit._devon_edited = True
    asyncio.run(adapter._handle_inbound(
        edit, None, handler, allow_check=True))
    assert len(delivered) == 2, delivered
    assert (delivered[1].meta or {}).get("service") == "peer_post"


def test_adapter_human_edit_still_deduped(tmp_path):
    """D2 negative (behaviour preserved): a HUMAN's edit of an
    already-seen message is still dropped by the dedupe — the
    bypass is for recognised bot posts only, so an edited human
    message never becomes a second turn."""
    import asyncio

    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=1, api_hash="x", session_path=str(tmp_path / "t.session"),
        media_dir=str(tmp_path / "media"),
        companion_bot_ids={PERSONAL_BOT_ID})
    delivered: list[ChatMessage] = []

    def handler(message):
        delivered.append(message)

    asyncio.run(adapter._handle_inbound(
        _inbound_event(GROUP_ID, MEMBER_ID, 56, "hello"),
        None, handler, allow_check=True))
    assert len(delivered) == 1
    edit = _inbound_event(GROUP_ID, MEMBER_ID, 56, "hello!!")
    edit._devon_edited = True
    asyncio.run(adapter._handle_inbound(
        edit, None, handler, allow_check=True))
    assert len(delivered) == 1, delivered
