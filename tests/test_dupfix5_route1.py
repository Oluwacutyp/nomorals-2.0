"""Telegram duplication — Route 1 regression fix (dupfix5).

Owner order, twice: find how it was before it broke and restore
that. The diagnosis (dupfix5-diag-verify/DIAGNOSIS.md) proved the
breaking change is 4f0f3518 (2026-11-05 05:51, "Surface laws wave"):
before it, the Telethon userbot adapter dropped every outgoing event
not in the owner's self-chat, so the userbot runtime never saw the
owner's commands in the bot's DM or in groups — exactly-once held
trivially. That commit added the own-echo seam (_deliver_owner_echo)
for #28 owner-reply suppression, but nothing stops an echo from
reaching the operator command branch (runtime.py), which — unlike
the first slash branch — never checked message.incoming. From then
on the owner's /play in the bot's DM executed twice by construction:
once on the bot runtime, once on the userbot runtime via the echo.

The fix (Route 1): the operator branch requires an inbound turn.
An own-echo (incoming=False) is ledger bookkeeping only — it still
feeds the OwnerReplyLedger at the seam above, but it can never
execute as a command. Every live command turn is inbound (Bot API,
Telethon inbound, the self-chat path via _handle_inbound, the
WhatsApp owner-command turn, the local console), so no carve-out
is needed — and the first slash branch already requires incoming.

The live shape is replayed exactly, as in test_telegram_dm_loop.py:
two runtimes, SEPARATE homes, one shared claims file, the userbot
WITHOUT companion ids, no registry sync (the live gap shape).
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path
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
        ("dupfix5", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    return runtime


def _msg(platform, *, chat_id, text, incoming, message_id,
         sender_id=OWNER_ID, own=False, reply_to="",
         kind=ChatKind.DM):
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                   peer=chat_id)
    meta: dict = {"is_owner": True}
    if own:
        meta["own_message"] = True
        meta["from_owner"] = True
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
    """The MTProto own-echo view of the same message, exactly as
    _deliver_owner_echo builds it: incoming=False, own_message +
    from_owner meta, sender = the owner's id."""
    return _msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=text,
                incoming=False, message_id=message_id, own=True,
                reply_to="")
    # The live adapter also stamps peer_is_bot on the echo's meta;
    # the unstamped shape is used here — the echo must never
    # execute either way.


def _group_echo(text, message_id):
    """The owner's group command as the userbot stream reports it
    back: an own-echo, incoming=False, sender = the owner."""
    return _msg("telegram", chat_id=GROUP_ID, text=text,
                incoming=False, message_id=message_id, own=True,
                kind=ChatKind.GROUP)


def _group_msg(platform, text, message_id, *, sender_id=OWNER_ID):
    """The same group message as one surface sees it live
    (incoming): the bot's view of the owner's group command."""
    return _msg(platform, chat_id=GROUP_ID, text=text,
                incoming=True, message_id=message_id,
                sender_id=sender_id, kind=ChatKind.GROUP)


def _self_chat_command(text, message_id):
    """A self-chat (Saved Messages) command exactly as the live
    adapter delivers it: _handle_inbound builds incoming=True —
    the outgoing event is converted to an inbound turn, never an
    echo."""
    return _msg("telegram", chat_id=OWNER_ID, text=text,
                incoming=True, message_id=message_id)


def _texts(adapter):
    return [t for t, _b in adapter.sent]


@pytest.fixture()
def split_homes(monkeypatch):
    """Two runtimes, SEPARATE homes — the live shape. Claims live
    in the per-test shared file (groupdup2). HOME is pointed at a
    temp dir so DraftQueue's Path.home() path stays in tmp. No
    registry sync: the live userbot carries no companion ids."""
    clear_failures()
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-d5-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    tmp = tempfile.TemporaryDirectory(prefix="nm-d5-shared-",
                                      dir=_TEST_TMPDIR)
    path = str(Path(tmp.name) / "claims.db")
    monkeypatch.setenv("NM_SHARED_CLAIMS", path)
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-d5a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-d5b-",
                                        dir=_TEST_TMPDIR)
    ctx_a = _context(tmp_a.name)
    ctx_b = _context(tmp_b.name)
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a = _runtime(ctx_a, ad_a)
    rt_b = _runtime(ctx_b, ad_b)
    yield (rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b)
    for runtime, _adapters, context in ((rt_a, ad_a, ctx_a),
                                        (rt_b, ad_b, ctx_b)):
        try:
            runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        context.close()
    tmp_a.cleanup()
    tmp_b.cleanup()
    tmp.cleanup()
    home_tmp.cleanup()


# ── Route 1: the echo never executes ──────────────────────────────────

def test_dm_echo_slash_command_never_executes(split_homes):
    """The regression, pinned: the own-echo of a DM slash command
    arriving FIRST on the userbot runtime (the race the owner
    lived) must produce ZERO executions — no reply, no deferral.
    Before the fix the echo claims the key and answers (the
    userbot's "pong"); the bot's copy then stands down, so the
    owner still sees one reply — from the wrong surface, and the
    double appears whenever the bot copy lands first instead.
    After the fix only the bot answers, ever.

    Failing-first: before the fix rt_b sends "pong" and registers
    no deferral (it handles); the first two asserts fail.
    """
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_b._process_inner(_echo_copy("/ping", "1000"))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 0
    rt_a._process_inner(_bot_copy("/ping", "1000"))
    assert len(_texts(ad_a["telegram-personal"])) == 1
    assert "pong" in _texts(ad_a["telegram-personal"])[0]
    assert _texts(ad_b["telegram"]) == []


def test_group_echo_slash_command_never_executes(split_homes):
    """Same shape in a group: the owner's group command as the
    userbot stream echoes it back must never execute there. The
    bot's incoming copy answers once.

    Failing-first: before the fix the echo claims and answers
    ("pong" from the userbot); the first assert fails.
    """
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_b._process_inner(_group_echo("/ping", "1010"))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 0
    rt_a._process_inner(_group_msg("telegram-personal", "/ping",
                                   "1010"))
    assert len(_texts(ad_a["telegram-personal"])) == 1
    assert "pong" in _texts(ad_a["telegram-personal"])[0]
    assert _texts(ad_b["telegram"]) == []


def test_dm_echo_plain_text_never_answers(split_homes):
    """A plain-text own-echo is not a turn either: no reply from
    the userbot runtime, no deferral registered."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_b._process_inner(_echo_copy("say hi", "1020"))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 0


# ── constraints: what must keep working ───────────────────────────────

def test_self_chat_command_still_executes(split_homes):
    """Hard constraint (1): a command typed in Saved Messages
    still executes on the userbot runtime — it arrives as an
    INBOUND turn (_handle_inbound builds incoming=True), never
    as an echo, so the operator branch still takes it."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_b._process_inner(_self_chat_command("/ping", "1030"))
    assert len(_texts(ad_b["telegram"])) == 1
    assert "pong" in _texts(ad_b["telegram"])[0]


def test_bot_copy_still_executes_once(split_homes):
    """The bot runtime's copy still answers once — the fix takes
    nothing from the owning surface."""
    (rt_a, ad_a, _ctx_a), (_rt_b, _ad_b, _ctx_b) = split_homes
    rt_a._process_inner(_bot_copy("/ping", "1040"))
    assert len(_texts(ad_a["telegram-personal"])) == 1
    assert "pong" in _texts(ad_a["telegram-personal"])[0]


def test_slash_echo_still_feeds_owner_reply_ledger(split_homes):
    """Hard constraint (2): the echo still feeds the #28
    OwnerReplyLedger — the seam's legitimate purpose. A slash
    echo sent as a reply to message m-42 marks m-42 answered,
    even though it never executes as a command."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    echo = _echo_copy("/ping", "1050")
    echo.meta["reply_to"] = "m-42"
    echo.reply_to = "m-42"
    rt_b._process_inner(echo)
    assert _texts(ad_b["telegram"]) == []
    ledger = rt_b._owner_reply_ledger()
    assert ledger.owner_answered("telegram:888", "m-42") == (
        "owner-replied-target")


def test_group_echo_still_feeds_owner_reply_ledger(split_homes):
    """Same ledger feed for a group echo: the owner's reply in
    the group is recorded, the echo still never executes."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    echo = _group_echo("/ping", "1060")
    echo.meta["reply_to"] = "m-77"
    echo.reply_to = "m-77"
    rt_b._process_inner(echo)
    assert _texts(ad_b["telegram"]) == []
    ledger = rt_b._owner_reply_ledger()
    assert ledger.owner_answered(
        "telegram:-100123456789", "m-77") == "owner-replied-target"
