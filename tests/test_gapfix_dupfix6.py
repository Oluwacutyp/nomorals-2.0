"""Lane 131 (dupfix6) — failover only on provable absence.

The f624063a silence fix (cmdsilence) made a deferring surface fail
over whenever it could not PROVE the primary answered — and both
proof guards (the shared answer claim, the observed answer in the
room) can fail while the primary is alive and answering. The owner
screenshotted the result on 2026-11-19: /help twice in the DM, one
/play three times (the second delivery ~601s after the first — the
claim-expiry re-arm path), the same song twice in a group.

The repair law: a failover fires only on positive evidence the
primary is absent — never merely on unprovable absence. These
tests prove the repair in code:

1. primary answers -> the second surface stands down, NO failover
   at 20s (DM) / 8s (group owner side) / 3s (group bot side);
2. primary provably down -> failover fires within the window;
3. shared store unavailable -> NO failover, honest log + honest
   failure line naming the store;
4. claim live but answer unproven at the short window -> NO
   answer, it waits for claim expiry;
5. claim expiry +1s with no answer and no live claim -> failover;
6. the echo shape (userbot own-echo of a command) never arms a
   failover for a command already answered in the room;
7. the silence guarantee still holds (dead primary ends answered
   or honestly failed, never silent);
+ the stale-build rule (a pre-dupfix6 claim ends in an honest
failure naming the old build, never in a failover), the
perspective-split fallback keys (mode 4), and the ledger stamp
round-trip (including the ALTER onto a pre-existing table).

Rig copied from tests/test_cmdsilence_owner_commands.py: two
runtimes, separate homes, one shared claims file when the test
needs it. The owner types a command in one chat; the bot and the
userbot each see their own copy of it.
"""

import contextlib
import json
import logging
import os
import random
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.partner.command_claims import (
    CLAIM_PROTOCOL_VERSION,
    CommandClaimLedger,
    DurableCommandClaimLedger,
    claim_build_stamp,
    command_claim_key,
    command_claim_key_variants,
)
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
from nomorals.storage.db import Database

_TEST_TMPDIR = (os.environ.get("NM_TEST_TMPDIR")
                or os.environ.get("TMPDIR")
                or os.path.expanduser("~/workspace/dupfix6-verify"))

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888
MAIN_BOT_ID = "8133295902"
GROUP_ID = "-1001234567890"
MEMBER_ID = "777"


class RecordingAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list = []

    def run(self, handler) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons=None, reply_keyboard=None,
             parse_mode: str = "") -> SendResult:
        self.sent.append((text, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def reply(self, message, text: str, **kwargs) -> SendResult:
        self.sent.append((text, kwargs.get("buttons")))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


class BotAdapter(RecordingAdapter):
    def __init__(self, name: str, bot_id: int, username: str) -> None:
        super().__init__(name)
        self._bot_id = bot_id
        self._bot_username = username


class BareUserbotAdapter(RecordingAdapter):
    """The split-hosting userbot: its own account identity, and
    NO companion bot ids — the documented live shape."""

    _me_username = "peacethefirst"
    _me_id = int(OWNER_ID)


def _context(home: str):
    settings = load_settings(overrides={
        "home": home,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.enabled": "false",
        "web.bot_username": "Nomorals_bot",
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
        ("dupfix6", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    # Deterministic failover clock: the tests drive the sweep
    # explicitly with injected times; the sweeper thread must not
    # fire on real time behind them (same shape as the cmdsilence
    # S6 test).
    runtime._start_peer_deferral_sweeper = lambda: None
    return runtime


def _msg(platform, *, chat_id, text, incoming, message_id,
         sender_id=OWNER_ID, own=False, peer_is_bot=False,
         service=None, kind=ChatKind.DM, reply_to=""):
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
                      sender="death", message_id=message_id, meta=meta)
    msg.sender_id = sender_id
    return msg


def _bot_copy(text, message_id, *, platform="telegram-personal"):
    """The Bot API view of the owner's message in a bot DM."""
    return _msg(platform, chat_id=OWNER_ID, text=text, incoming=True,
                message_id=message_id)


def _echo_copy(text, message_id, *, chat_id=PERSONAL_BOT_ID):
    """The MTProto own-echo view of the same message (incoming=False)."""
    return _msg("telegram", chat_id=str(chat_id), text=text,
                incoming=False, message_id=message_id, own=True,
                peer_is_bot=True)


def _group_echo(text, message_id):
    """The owner's own message in a group, as the userbot sees it."""
    return _msg("telegram", chat_id=GROUP_ID, text=text, incoming=False,
                message_id=message_id, own=True, kind=ChatKind.GROUP)


def _group_msg(platform, text, message_id, *, sender_id=OWNER_ID):
    return _msg(platform, chat_id=GROUP_ID, text=text, incoming=True,
                message_id=message_id, sender_id=sender_id,
                kind=ChatKind.GROUP)


def _peer_post(text, message_id, *, reply_to="", chat_id=None,
               kind=ChatKind.DM, platform="telegram"):
    """The bot's own post in a chat, as the userbot stream sees it
    — delivered as an observation, quoting the command it
    answers (reply_to carries the command's message id)."""
    return _msg(platform, chat_id=str(chat_id or PERSONAL_BOT_ID),
                text=text, incoming=True, message_id=message_id,
                sender_id=str(PERSONAL_BOT_ID), service="peer_post",
                kind=kind, reply_to=reply_to)


def _texts(adapter):
    return [t for t, _b in adapter.sent]


def _pending(runtime):
    return runtime._peer_deferral().pending_count()


@pytest.fixture
def shared_claims_file(monkeypatch, tmp_path):
    path = tmp_path / "claims.db"
    monkeypatch.setenv("NM_SHARED_CLAIMS", str(path))
    return str(path)


@pytest.fixture
def broken_shared_claims(monkeypatch, tmp_path):
    """A shared-claims path that can never open: its parent is a
    regular file, so the directory cannot be created (the
    groupdup2 shape for a split/unavailable store)."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(
        "NM_SHARED_CLAIMS", str(blocker / "claims.db"))
    return str(blocker / "claims.db")


@pytest.fixture
def split_homes(monkeypatch):
    clear_failures()
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-d6-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-d6-a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-d6-b-",
                                        dir=_TEST_TMPDIR)
    yield tmp_a.name, tmp_b.name
    for tmp in (tmp_a, tmp_b, home_tmp):
        tmp.cleanup()


def _make_runtime(home, adapters):
    ctx = _context(home)
    return _runtime(ctx, adapters), ctx


def _stop(rt):
    with contextlib.suppress(Exception):  # noqa: BLE001 - teardown only
        rt.stop()


# ------------------------------------------------------------ (1) DM

def test_dm_primary_answers_no_failover_at_20s(
        split_homes, shared_claims_file):
    """(1) DM: the bot answers /ping and records the answer claim;
    the userbot's echo deferred first. The sweep must stand the
    failover down on the PROVEN answer — never a second pong —
    at the 20s window and past claim expiry alike. The answer is
    never observed (no peer post): only the answer claim can
    save this turn."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_echo_copy("/ping", "d1-1"))
        assert _texts(ad_b["telegram"]) == []
        assert _pending(rt_b) == 1, "the echo must defer"
        rt_a._process_inner(_bot_copy("/ping", "d1-1"))
        assert len(_texts(ad_a["telegram-personal"])) == 1
        rt_b._sweep_peer_deferrals(now=base + 25.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer a proven answer")
        assert _pending(rt_b) == 0
        rt_b._sweep_peer_deferrals(now=base + 602.0)
        assert _texts(ad_b["telegram"]) == []
        assert len(_texts(ad_a["telegram-personal"])) == 1
    finally:
        _stop(rt_a)
        _stop(rt_b)


# --------------------------------------------------------- (1) group

def test_group_owner_side_primary_answers_no_failover_at_8s(
        split_homes, shared_claims_file):
    """(1) Group, owner side (8s window): the bot answers the
    group /ping; the userbot's group echo deferred first. The
    sweep must stand down on the proven answer — no second pong
    — with the answer never observed."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_group_echo("/ping", "g1-1"))
        assert _texts(ad_b["telegram"]) == []
        assert _pending(rt_b) == 1, "the group echo must defer"
        rt_a._process_inner(_group_msg(
            "telegram-personal", "/ping", "g1-1"))
        assert len(_texts(ad_a["telegram-personal"])) == 1
        rt_b._sweep_peer_deferrals(now=base + 12.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer a proven group answer")
        assert _pending(rt_b) == 0
        assert len(_texts(ad_a["telegram-personal"])) == 1
    finally:
        _stop(rt_a)
        _stop(rt_b)


def test_group_bot_side_primary_answers_no_failover_at_3s(
        split_homes, shared_claims_file):
    """(1) Group, bot side (3s fast window): the userbot's copy
    DEFERS — it must never win the claim race for a group
    command. (dupfix7, 2026-11-20: the setup this test used to
    pin — "the userbot answers the group /ping first" — IS the
    defect the owner reported: the userbot replying while the
    public bot is in the group, prompt bot or slow. The no-double
    property below is unchanged; the roles invert — the bot
    answers, the userbot stands down on the proven answer.)"""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_group_msg("telegram", "/ping", "g2-1"))
        assert _texts(ad_b["telegram"]) == [], (
            "the userbot must defer, never answer first")
        assert _pending(rt_b) == 1, "the userbot copy must defer"
        rt_a._process_inner(_group_msg(
            "telegram-personal", "/ping", "g2-1"))
        assert len(_texts(ad_a["telegram-personal"])) == 1, (
            "the bot must answer once")
        rt_b._sweep_peer_deferrals(now=base + 12.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the userbot sweep must not double-answer")
        assert _pending(rt_b) == 0
        assert len(_texts(ad_a["telegram-personal"])) == 1
    finally:
        _stop(rt_a)
        _stop(rt_b)


# ------------------------------------------------------------ (2)

def test_absent_primary_fails_over_shared_store(
        split_homes, shared_claims_file):
    """(2) The primary never claims and never answers, and the
    shared store is open — so its absence IS provable: no claim,
    no answer, one shared truth. The sweep fails over at the
    short window and answers exactly once."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        assert rt._claims_are_shared() is True
        base = time.time()
        rt._process_inner(_echo_copy("/ping", "d2-1"))
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1
        rt._sweep_peer_deferrals(now=base + 25.0)
        texts = _texts(ad["telegram"])
        assert len(texts) == 1, f"the failover must answer once: {texts}"
        assert "pong" in texts[0]
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ------------------------------------------------------------ (4)+(5)

def test_stamped_claim_never_answering_fails_over_at_expiry(
        split_homes, shared_claims_file):
    """(4)+(5) A CURRENT-build primary claims the command and
    never answers — seeded 595s back, so its claim expires 5s
    into the test (the S2 shape: the ledger is read in real
    time, and expiry is crossed with a real sleep). At the short
    window the deferring surface must NOT answer — the claim is
    live, the primary may still be working — it re-arms to claim
    expiry. At claim expiry, with no answer and no live claim,
    it fails over and answers exactly once."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram"),
          "telegram-personal": BotAdapter(
              "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        base = time.time()
        echo = _echo_copy("/ping", "d3-1")
        key = command_claim_key(echo, own_account_id=OWNER_ID)
        assert rt._command_claims().claim(
            key, "telegram-personal:888", now=base - 595.0,
            build_stamp=claim_build_stamp())
        rt._process_inner(echo)
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1, "the echo must defer"
        # Short window: the claim is live in real time — wait,
        # never answer.
        rt._sweep_peer_deferrals(now=base + 25.0)
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1, "the deferral must survive to claim expiry"
        # The claim expires ~5s after the test started; wait it out.
        time.sleep(6.5)
        rt._sweep_peer_deferrals(now=base + 25.0)
        texts = _texts(ad["telegram"])
        assert len(texts) == 1, f"the failover must answer once: {texts}"
        assert "pong" in texts[0]
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ------------------------------------------------------------ (3)

def test_broken_store_no_failover_honest_failure_at_bound(
        split_homes, broken_shared_claims, caplog):
    """(3) The shared store is unavailable: this surface is
    BLIND — the primary's claim and answer are invisible to it,
    so no store evidence proves the primary is down. The sweep
    must NEVER fail over blindly: at the short window it waits,
    and at the claim window's bound it ends in ONE honest
    failure line naming the store — never a second answer."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        assert rt._claims_are_shared() is False
        base = time.time()
        rt._process_inner(_echo_copy("/ping", "d4-1"))
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1, "the echo must defer"
        with caplog.at_level(logging.WARNING,
                             logger="nomorals.agents.partner.runtime"):
            rt._sweep_peer_deferrals(now=base + 25.0)
        assert _texts(ad["telegram"]) == [], (
            "a blind sweep must not answer")
        assert _pending(rt) == 1, "the deferral must wait to the bound"
        assert "dupfix6 blind wait" in caplog.text
        rt._sweep_peer_deferrals(now=base + 605.0)
        texts = _texts(ad["telegram"])
        assert len(texts) == 1, f"exactly one honest line: {texts}"
        assert "pong" not in texts[0]
        assert "shared" in texts[0] and "store" in texts[0], texts[0]
        assert _pending(rt) == 0
        assert "dupfix6 honest failure" in caplog.text
    finally:
        _stop(rt)


def test_incoming_deferral_on_broken_store_fails_over(
        split_homes, broken_shared_claims):
    """Scope boundary: the blind rule is ECHO-only. An INCOMING
    deferral on a broken store still fails over at its short
    window — both surfaces saw the message directly and neither
    handled it, so the stagger (fast side fires, slow side
    observes) orders the race. This is the pinned groupdup2
    game stagger's contract, extended to an incoming command:
    the blind rule must never silence it. The echo is the shape
    that doubled live, and the echo is what the blind rule
    covers (the test above)."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        assert rt._claims_are_shared() is False
        base = time.time()
        # An incoming group /ping on a broken store: rule (3)
        # defers (no shared truth to fall through on).
        rt._process_inner(_group_msg("telegram", "/ping", "d10-1"))
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1, "the incoming copy must defer"
        rt._sweep_peer_deferrals(now=base + 12.0)
        texts = _texts(ad["telegram"])
        assert len(texts) == 1, f"the failover must answer once: {texts}"
        assert "pong" in texts[0]
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ------------------------------------------------------------ (6)

def test_echo_after_observed_answer_never_arms_failover(
        split_homes, broken_shared_claims):
    """(6) The fast-bot/late-echo race, with the store down so no
    answer claim can save the turn: the bot's answer lands in
    the room BEFORE the echo arrives. The echo must stand down
    on the observed answer WITHOUT registering a deferral — no
    failover is ever armed for a command already answered.
    Negative control: a peer post quoting a DIFFERENT message
    does not stand the echo down."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        assert rt._claims_are_shared() is False
        # The bot's answer, observed first (it quotes the command).
        rt._process_inner(_peer_post("pong", "102", reply_to="d5-1"))
        rt._process_inner(_echo_copy("/ping", "d5-1"))
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 0, (
            "the echo must not arm a failover for an answered command")
        rt._sweep_peer_deferrals(now=time.time() + 60.0)
        assert _texts(ad["telegram"]) == []
        # Negative control: a peer post quoting another message
        # is not this command's answer — the echo defers normally.
        rt._process_inner(_peer_post("pong", "104", reply_to="other-9"))
        rt._process_inner(_echo_copy("/ping", "d5-2"))
        assert _pending(rt) == 1, (
            "an unrelated peer post must not stand the echo down")
    finally:
        _stop(rt)


# ------------------------------------------------- stale build (c)

def test_stale_build_claim_ends_in_honest_failure(
        split_homes, shared_claims_file, caplog):
    """(c) Mixed builds: the claim on this command was written by
    a pre-dupfix6 build (no stamp — a build that never writes
    answer claims, so its missing answer proves nothing).
    Seeded 595s back, so its claim expires 5s into the test
    (the S2 shape). At the short window the deferring surface
    re-arms on the live claim, never answers. At claim expiry
    it must NOT fail over onto it: it ends in ONE honest
    failure naming the older build, never in a second answer."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram"),
          "telegram-personal": BotAdapter(
              "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        base = time.time()
        echo = _echo_copy("/ping", "d6-1")
        key = command_claim_key(echo, own_account_id=OWNER_ID)
        # An UNSTAMPED claim: a pre-dupfix6 primary.
        assert rt._command_claims().claim(
            key, "telegram-personal:888", now=base - 595.0)
        rt._process_inner(echo)
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1, "the echo must defer"
        with caplog.at_level(logging.WARNING,
                             logger="nomorals.agents.partner.runtime"):
            # Short window: the claim is live in real time —
            # re-arm, never answer.
            rt._sweep_peer_deferrals(now=base + 25.0)
            assert _texts(ad["telegram"]) == []
            assert _pending(rt) == 1
            # The claim expires ~5s after the test started; wait
            # it out.
            time.sleep(6.5)
            rt._sweep_peer_deferrals(now=base + 25.0)
        texts = _texts(ad["telegram"])
        assert len(texts) == 1, f"exactly one honest line: {texts}"
        assert "pong" not in texts[0]
        assert "older build" in texts[0], texts[0]
        assert _pending(rt) == 0
        assert "dupfix6 honest failure" in caplog.text
    finally:
        _stop(rt)


# ------------------------------------------------- mode 4: fallback keys

def test_fallback_key_claim_is_seen_not_doubled(
        split_homes, shared_claims_file):
    """(a) Mode 4: the primary claimed under the perspective-split
    FALLBACK key (its own account id was unknown at claim time),
    so the canonical key looks free. The echo's arbitration must
    still see the live foreign claim and defer — never handle at
    intake — and the sweep must re-arm on it, never answer."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram"),
          "telegram-personal": BotAdapter(
              "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        base = time.time()
        echo = _echo_copy("/ping", "d7-1")
        # The bot-side fallback key: the bot sees this DM as the
        # owner's id, so its fallback is telegram:<owner>:<mid>.
        fallback = f"telegram:{OWNER_ID}:d7-1"
        assert fallback in command_claim_key_variants(
            command_claim_key(echo, own_account_id=OWNER_ID))
        assert rt._command_claims().claim(
            fallback, "telegram-personal:888", now=base,
            build_stamp=claim_build_stamp())
        rt._process_inner(echo)
        assert _texts(ad["telegram"]) == []
        assert _texts(ad["telegram-personal"]) == []
        assert _pending(rt) == 1, (
            "the echo must defer on the fallback claim")
        rt._sweep_peer_deferrals(now=base + 25.0)
        assert _texts(ad["telegram"]) == [], (
            "the sweep must not answer over a live fallback claim")
        assert _pending(rt) == 1
    finally:
        _stop(rt)


def test_fallback_key_answer_stands_down_at_once(
        split_homes, shared_claims_file):
    """(a) Mode 4, answer side: the primary ANSWERED under the
    fallback key. The echo must stand down at once on the proven
    answer — no deferral, no failover."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        echo = _echo_copy("/ping", "d8-1")
        fallback = f"telegram:{OWNER_ID}:d8-1"
        rt._command_claims().record_answer(
            fallback, "telegram-personal:888",
            build_stamp=claim_build_stamp())
        rt._process_inner(echo)
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 0, (
            "the echo must stand down on the fallback answer")
        rt._sweep_peer_deferrals(now=time.time() + 60.0)
        assert _texts(ad["telegram"]) == []
    finally:
        _stop(rt)


# ------------------------------------------------------------ (7)

@pytest.mark.parametrize("store_state", ["shared", "broken"])
def test_silence_guarantee_both_store_states(
        split_homes, store_state, monkeypatch, tmp_path, caplog):
    """(7) The silence guarantee, both ways: a command whose
    primary is dead ends in exactly ONE outcome line — a
    failover answer when the store is open (absence is
    provable), an honest failure naming why when it is not.
    Never silent, never doubled."""
    if store_state == "shared":
        path = tmp_path / "claims.db"
        monkeypatch.setenv("NM_SHARED_CLAIMS", str(path))
    else:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setenv(
            "NM_SHARED_CLAIMS", str(blocker / "claims.db"))
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        base = time.time()
        rt._process_inner(_echo_copy("/ping", "d9-1"))
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1
        rt._sweep_peer_deferrals(now=base + 25.0)
        if store_state == "shared":
            texts = _texts(ad["telegram"])
            assert len(texts) == 1 and "pong" in texts[0], texts
        else:
            assert _texts(ad["telegram"]) == []
            assert _pending(rt) == 1
            rt._sweep_peer_deferrals(now=base + 605.0)
            texts = _texts(ad["telegram"])
            assert len(texts) == 1, f"exactly one honest line: {texts}"
            assert "pong" not in texts[0]
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ------------------------------------------------- ledger: stamps

def test_claim_and_answer_carry_the_build_stamp(tmp_path):
    """The stamp round-trips on both ledgers, and a missing stamp
    reads back as "" (a pre-dupfix6 row)."""
    mem = CommandClaimLedger()
    assert mem.claim("k", "a", build_stamp="dupfix6-p2") is True
    assert mem.claim_info("k")[2] == "dupfix6-p2"
    assert mem.last_claim_info("k")[2] == "dupfix6-p2"
    mem.record_answer("k", "a", build_stamp="dupfix6-p2")
    assert mem.answer_info("k")[2] == "dupfix6-p2"
    assert mem.claim("k2", "a") is True
    assert mem.claim_info("k2")[2] == ""
    db = Database(str(tmp_path / "claims.db"))
    dur = DurableCommandClaimLedger(db)
    assert dur.claim("k", "a", build_stamp="dupfix6-p2") is True
    assert dur.claim_info("k")[2] == "dupfix6-p2"
    assert dur.last_claim_info("k")[2] == "dupfix6-p2"
    dur.record_answer("k", "a", build_stamp="dupfix6-p2")
    assert dur.answer_info("k")[2] == "dupfix6-p2"
    assert dur.claim("k2", "a") is True
    assert dur.claim_info("k2")[2] == ""
    assert claim_build_stamp() == f"dupfix6-p{CLAIM_PROTOCOL_VERSION}"


def test_stamp_column_added_to_pre_existing_table(tmp_path):
    """A claims file created by an older build (tables WITHOUT
    build_stamp) gains the column on first open, and claims
    written before the upgrade keep working with stamp ""."""
    path = str(tmp_path / "claims.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE command_claims (claim_key TEXT PRIMARY KEY,"
        " surface TEXT NOT NULL DEFAULT '',"
        " claimed_at REAL NOT NULL DEFAULT 0);"
        "CREATE TABLE command_answers (answer_key TEXT PRIMARY KEY,"
        " surface TEXT NOT NULL DEFAULT '',"
        " answered_at REAL NOT NULL DEFAULT 0);"
        "INSERT INTO command_claims (claim_key, surface,"
        " claimed_at) VALUES ('old-k', 'telegram-bot:1', 1000.0);")
    conn.commit()
    conn.close()
    db = Database(path)
    ledger = DurableCommandClaimLedger(db)
    assert ledger.last_claim_info("old-k") == (
        "telegram-bot:1", 1000.0, "")
    assert ledger.claim("new-k", "a",
                        build_stamp=claim_build_stamp()) is True
    assert ledger.claim_info("new-k")[2] == claim_build_stamp()
    ledger.record_answer("new-k", "a",
                         build_stamp=claim_build_stamp())
    assert ledger.answer_info("new-k")[2] == claim_build_stamp()


def test_claim_key_variants_cover_both_fallbacks():
    """A canonical DM key expands to both perspective-split
    fallbacks; every other key stands alone."""
    assert command_claim_key_variants(
        "telegram:dm:4242:888:m1") == [
        "telegram:dm:4242:888:m1",
        "telegram:4242:m1",
        "telegram:888:m1",
    ]
    assert command_claim_key_variants(
        "telegram:-100123: m1".replace(" ", "")) == [
        "telegram:-100123:m1"]
    assert command_claim_key_variants("") == []
