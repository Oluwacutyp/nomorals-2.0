"""Duplicate control confirmations (owner screenshot, 2026-11-11):
one ``/chatmode`` in the personal bot's DM was answered TWICE —
once by the bot, once as-him.

The 2026-11-05 arbitration stopped this inside one runtime
instance: an in-memory claim ledger plus a bot-DM peer rule
keyed on adapter-carried companion ids. The screenshot's shape
crosses the seam both mechanisms stop at — the surfaces were
not arbitrated by one instance's memory (split hosting), the
userbot side carried no companion ids to feed the peer rule,
and the claim key itself was perspective-shaped (a bot DM keys
on the owner's id from the bot's side and on the bot's id from
the owner's side), so the two copies could never collide on
one key in ANY ledger.

The fix pinned here: claims live in the home store
(:class:`DurableCommandClaimLedger` — surfaces in different
instances/processes claim against the same table), and the key
is canonical (:func:`command_claim_key` — a Telegram DM keys
on the sorted pair of both participant ids, identical from
both sides). Single-surface and single-instance behaviour is
unchanged; the classic shapes stay pinned in
tests/test_surface_laws.py.
"""
from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.partner.command_claims import (
    WINDOW_S,
    CommandClaimLedger,
    DurableCommandClaimLedger,
    command_claim_key,
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

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888
GATED_BOT_ID = 777


# ── canonical key pins ────────────────────────────────────────

def _msg(platform: str, chat_id: str, kind: str, message_id: str,
         *, incoming: bool = True, own: bool = False) -> ChatMessage:
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                   peer=chat_id)
    meta: dict = {"is_owner": True}
    if own:
        meta["own_message"] = True
    msg = ChatMessage(chat=chat, incoming=incoming, text="/chatmode x",
                      sender="death", message_id=message_id, meta=meta)
    msg.sender_id = OWNER_ID
    return msg


def test_dm_key_is_the_same_from_both_sides():
    bot_copy = _msg("telegram-personal", OWNER_ID, ChatKind.DM, "100")
    echo_copy = _msg("telegram", str(PERSONAL_BOT_ID), ChatKind.DM,
                     "100", incoming=False, own=True)
    from_bot = command_claim_key(bot_copy, str(PERSONAL_BOT_ID))
    from_echo = command_claim_key(echo_copy, OWNER_ID)
    assert from_bot == from_echo == "telegram:dm:4242:888:100"


def test_group_key_keeps_the_original_shape():
    msg = _msg("telegram-bot", "-1001", ChatKind.GROUP, "55")
    assert command_claim_key(msg, str(GATED_BOT_ID)) \
        == "telegram:-1001:55"
    msg2 = _msg("whatsapp", "123@g.us", ChatKind.GROUP, "9")
    assert command_claim_key(msg2, "") == "whatsapp:123@g.us:9"


def test_dm_key_without_own_id_falls_back_to_original_shape():
    bot_copy = _msg("telegram-personal", OWNER_ID, ChatKind.DM, "100")
    assert command_claim_key(bot_copy, "") == "telegram:4242:100"


def test_self_chat_key_is_not_a_pair():
    msg = _msg("telegram", OWNER_ID, ChatKind.DM, "7", own=True)
    assert command_claim_key(msg, OWNER_ID) == "telegram:4242:7"


def test_unkeyable_message_has_no_key():
    msg = _msg("telegram", OWNER_ID, ChatKind.DM, "")
    assert command_claim_key(msg, OWNER_ID) == ""


# ── durable ledger pins ───────────────────────────────────────

@pytest.fixture()
def db_pair(tmp_path):
    path = str(tmp_path / "claims.db")
    first = Database(path)
    second = Database(path)
    yield first, second
    first.close()
    second.close()


def test_durable_claim_binds_across_instances(db_pair):
    first, second = db_pair
    ledger_a = DurableCommandClaimLedger(first)
    ledger_b = DurableCommandClaimLedger(second)
    key = "telegram:dm:4242:888:100"
    assert ledger_a.claim(key, "telegram-personal") is True
    # the SAME surface re-checking its own claim still owns it
    assert ledger_a.claim(key, "telegram-personal") is True
    # a twin surface in another instance stands down
    assert ledger_b.claim(key, "telegram") is False
    assert ledger_b.claimed_by(key) == "telegram-personal"


def test_durable_claim_expires_and_is_stealable(db_pair):
    first, second = db_pair
    ledger_a = DurableCommandClaimLedger(first)
    ledger_b = DurableCommandClaimLedger(second)
    key = "telegram:dm:4242:888:101"
    t0 = 1_000_000.0
    assert ledger_a.claim(key, "telegram-personal", now=t0) is True
    assert ledger_b.claim(key, "telegram", now=t0 + 10) is False
    # past the window the claim is dead: a late copy may handle —
    # a crashed first surface never mutes the chat forever
    assert ledger_b.claim(key, "telegram",
                          now=t0 + WINDOW_S + 1) is True
    assert ledger_b.claimed_by(key, now=t0 + WINDOW_S + 2) \
        == "telegram"


def test_durable_unkeyable_claims_pass(db_pair):
    first, _second = db_pair
    ledger = DurableCommandClaimLedger(first)
    assert ledger.claim("", "telegram") is True
    assert ledger.claim("k", "") is True
    assert ledger.claimed_by("") == ""


def test_memory_ledger_contract_unchanged():
    ledger = CommandClaimLedger()
    assert ledger.claim("k", "a") is True
    assert ledger.claim("k", "a") is True
    assert ledger.claim("k", "b") is False
    assert ledger.claimed_by("k") == "a"


# ── runtime pins: the incident topology ───────────────────────

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
    """The split-hosting userbot: its own account identity, and
    NO companion bot ids — the shape whose peer rule had no
    inputs in the incident."""

    _me_username = "peacethefirst"
    _me_id = int(OWNER_ID)


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
    import json as _json
    for marker in ("pulse_last_run.json",
                   "pulse_last_run_afternoon.json",
                   "pulse_last_run_evening.json",
                   "pulse_last_run_midnight.json"):
        (ws / marker).write_text(
            _json.dumps({"ts": time.time(), "delivered_text": True}),
            encoding="utf-8")
    context.db.execute(
        "INSERT OR REPLACE INTO briefings (id, date, sections_json,"
        " generated_at) VALUES (?, ?, '[]', ?)",
        ("dupfix", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    return runtime


def _owner_msg(platform, *, kind, chat_id, text, incoming,
               message_id, own=False):
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                   peer=chat_id)
    meta: dict = {"is_owner": True}
    if own:
        meta["own_message"] = True
        meta["from_owner"] = True
    msg = ChatMessage(chat=chat, incoming=incoming, text=text,
                      sender="death", message_id=message_id, meta=meta)
    msg.sender_id = OWNER_ID
    return msg


@pytest.fixture()
def split_home():
    clear_failures()
    tmp = tempfile.TemporaryDirectory(prefix="nm-dupfix-",
                                      dir=_TEST_TMPDIR)
    yield tmp.name
    tmp.cleanup()


def _split_runtimes(home: str):
    ctx_a = _context(home)
    adapters_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt_a = _runtime(ctx_a, adapters_a)
    ctx_b = _context(home)
    adapters_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_b = _runtime(ctx_b, adapters_b)
    return (rt_a, adapters_a, ctx_a), (rt_b, adapters_b, ctx_b)


def _teardown(*rigs):
    for runtime, _adapters, context in rigs:
        try:
            runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        context.close()


def test_split_instances_one_confirmation(split_home):
    """The incident, replayed: the bot copy on one instance, the
    as-him echo on another, one shared home store — exactly one
    confirmation is sent, and the claim row names the owner of
    the handling in the shared store (the stand-down is
    observable, never silent state)."""
    rig_a, rig_b = _split_runtimes(split_home)
    rt_a, adapters_a, _ctx_a = rig_a
    rt_b, adapters_b, ctx_b = rig_b
    try:
        text = "/chatmode games-only +2348164302080"
        rt_a._process_inner(_owner_msg(
            "telegram-personal", kind=ChatKind.DM, chat_id=OWNER_ID,
            text=text, incoming=True, message_id="100"))
        rt_b._process_inner(_owner_msg(
            "telegram", kind=ChatKind.DM,
            chat_id=str(PERSONAL_BOT_ID), text=text, incoming=False,
            message_id="100", own=True))
        sent_a = [t for t, _b in adapters_a["telegram-personal"].sent]
        sent_b = [t for t, _b in adapters_b["telegram"].sent]
        assert len(sent_a) == 1, sent_a
        assert "is now games-only" in sent_a[0]
        assert sent_b == [], sent_b
        row = ctx_b.db.query_one(
            "SELECT claim_key, surface FROM command_claims"
            " WHERE claim_key = ?",
            ("telegram:dm:4242:888:100",))
        assert row is not None
        assert row["surface"] == "telegram-personal"
    finally:
        _teardown(rig_a, rig_b)


def test_split_instances_effect_runs_once(split_home):
    """Not just the reply: a non-idempotent command (/play mints
    a job) executes exactly once across the two instances."""
    rig_a, rig_b = _split_runtimes(split_home)
    rt_a, adapters_a, _ctx_a = rig_a
    rt_b, adapters_b, _ctx_b = rig_b
    calls: list = []

    def fake_play(tail, chat_key=""):
        calls.append((tail, chat_key))
        return f"queued (job {len(calls)})"

    rt_a._control_play = fake_play  # type: ignore[method-assign]
    rt_b._control_play = fake_play  # type: ignore[method-assign]
    try:
        text = "/play fear by nf"
        rt_a._process_inner(_owner_msg(
            "telegram-personal", kind=ChatKind.DM, chat_id=OWNER_ID,
            text=text, incoming=True, message_id="110"))
        rt_b._process_inner(_owner_msg(
            "telegram", kind=ChatKind.DM,
            chat_id=str(PERSONAL_BOT_ID), text=text, incoming=False,
            message_id="110", own=True))
        assert len(calls) == 1, calls
        assert len(adapters_a["telegram-personal"].sent) == 1
        assert adapters_b["telegram"].sent == []
    finally:
        _teardown(rig_a, rig_b)


def test_split_instances_distinct_command_still_answers(split_home):
    """No over-suppression: a different command (new message id)
    in the same chat pair is handled normally by the echo
    surface when it is the only copy."""
    rig_a, rig_b = _split_runtimes(split_home)
    rt_a, adapters_a, _ctx_a = rig_a
    rt_b, adapters_b, _ctx_b = rig_b
    try:
        rt_a._process_inner(_owner_msg(
            "telegram-personal", kind=ChatKind.DM, chat_id=OWNER_ID,
            text="/chatmode games-only +2348164302080",
            incoming=True, message_id="100"))
        rt_b._process_inner(_owner_msg(
            "telegram", kind=ChatKind.DM,
            chat_id=str(PERSONAL_BOT_ID),
            text="/chatmode normal +2348164302080", incoming=False,
            message_id="101", own=True))
        sent_b = [t for t, _b in adapters_b["telegram"].sent]
        assert len(sent_b) == 1, sent_b
        assert "back to normal" in sent_b[0]
    finally:
        _teardown(rig_a, rig_b)


def test_split_instances_stale_claim_does_not_mute(split_home):
    """Crash TTL at the runtime seam: the bot claimed a command
    and died mid-handling (the row is a full window old); the
    echo copy must still be handled, not stood down forever."""
    rig_a, rig_b = _split_runtimes(split_home)
    rt_a, _adapters_a, ctx_a = rig_a
    rt_b, adapters_b, _ctx_b = rig_b
    try:
        # The crashed surface had run: its ledger exists (the
        # table is created with it) and its claim row is stale.
        rt_a._command_claims()
        ctx_a.db.execute(
            "INSERT INTO command_claims"
            " (claim_key, surface, claimed_at) VALUES (?, ?, ?)",
            ("telegram:dm:4242:888:102", "telegram-personal",
             time.time() - WINDOW_S - 500.0))
        rt_b._process_inner(_owner_msg(
            "telegram", kind=ChatKind.DM,
            chat_id=str(PERSONAL_BOT_ID),
            text="/chatmode normal +2348164302080", incoming=False,
            message_id="102", own=True))
        sent_b = [t for t, _b in adapters_b["telegram"].sent]
        assert len(sent_b) == 1, sent_b
        assert "back to normal" in sent_b[0]
    finally:
        _teardown(rig_a, rig_b)
