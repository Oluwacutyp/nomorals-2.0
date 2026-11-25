"""Lane 132 (helpdup) — group /help double reply: the public bot
answers, then the userbot answers the same /help.

Owner-verified 2026-11-24 on the live new box: /help in a Telegram
group gets TWO answers (bot first, userbot second); /play and
/status single-reply. The failover sweep fires for /help because
neither proof channel lands for it:

* the shared answer claim (the sweep's step-1 proof), and
* the in-room observation memory (dupfix6, keyed by reply_to).

Reproduction rig copied from tests/test_gapfix_dupfix6.py: two
runtimes, separate homes, one shared claims file when the test
needs it. The bot's answer post is delivered to the userbot stream
the way the real MTProto stream delivers it — an ordinary incoming
group message from the bot's id, quoting nothing (the runtime's
one control-reply channel never passes reply_to).
"""

import contextlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

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

_TEST_TMPDIR = (os.environ.get("NM_TEST_TMPDIR")
                or os.environ.get("TMPDIR")
                or os.path.expanduser("~/workspace/helpdup-verify/tmp"))

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888
MAIN_BOT_ID = "8133295902"
GROUP_ID = "-1001234567890"


class CaptureAdapter(ChatAdapter):
    """Records every send as (text, reply_to, buttons)."""

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list = []

    def run(self, handler) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons=None, reply_keyboard=None,
             parse_mode: str = "") -> SendResult:
        self.sent.append((text, reply_to, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def reply(self, message, text: str, **kwargs) -> SendResult:
        self.sent.append((text, kwargs.get("reply_to", ""), None))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


class BotAdapter(CaptureAdapter):
    def __init__(self, name: str, bot_id: int, username: str) -> None:
        super().__init__(name)
        self._bot_id = bot_id
        self._bot_username = username
        # Live-session edits (the real Bot API adapter exposes
        # edit_message, telegram.py:7363; the harness adapter
        # must carry the same verb or the edit shape can never
        # land in-harness and every menu render falls through
        # to a fresh send).
        self.edits: list = []

    def edit_message(self, chat, message_id, text, *, buttons=None,
                     parse_mode=""):
        self.edits.append((str(message_id), text, buttons))
        return {"ok": True}


class BareUserbotAdapter(CaptureAdapter):
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
        ("helpdup", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    # Deterministic failover clock: the tests drive the sweep
    # explicitly with injected times; the sweeper thread must not
    # fire on real time behind them.
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


def _group_echo(text, message_id):
    """The owner's own message in a group, as the userbot sees it."""
    return _msg("telegram", chat_id=GROUP_ID, text=text, incoming=False,
                message_id=message_id, own=True, kind=ChatKind.GROUP)


def _group_msg(platform, text, message_id, *, sender_id=OWNER_ID,
               reply_to=""):
    return _msg(platform, chat_id=GROUP_ID, text=text, incoming=True,
                message_id=message_id, sender_id=sender_id,
                kind=ChatKind.GROUP, reply_to=reply_to)


def _peer_post(text, message_id, *, reply_to="", chat_id=None,
               kind=ChatKind.DM, platform="telegram", sender_id=None):
    """The bot's own post in a chat, as the userbot stream sees it
    — delivered as an observation, quoting the command it answers
    (reply_to carries the command's message id)."""
    return _msg(platform, chat_id=str(chat_id or PERSONAL_BOT_ID),
                text=text, incoming=True, message_id=message_id,
                sender_id=str(sender_id or PERSONAL_BOT_ID),
                service="peer_post", kind=kind, reply_to=reply_to)


def _texts(adapter):
    return [t for t, _r, _b in adapter.sent]


def _reply_tos(adapter):
    return [r for _t, r, _b in adapter.sent]


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
    regular file, so the directory cannot be created."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(
        "NM_SHARED_CLAIMS", str(blocker / "claims.db"))
    return str(blocker / "claims.db")


@pytest.fixture
def split_homes(monkeypatch):
    clear_failures()
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-hd-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-hd-a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-hd-b-",
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


def _bot_answer_post(text, message_id, reply_to=""):
    """The bot's answer post as the userbot's MTProto stream
    delivers it: an ordinary incoming group message from the
    bot's account id. After the fix it quotes the command it
    answers (reply_to carries the command's message id) — the
    proof the sibling observes."""
    return _group_msg("telegram", text, message_id,
                      sender_id=str(PERSONAL_BOT_ID),
                      reply_to=reply_to)


def _dm_bot_copy(text, message_id):
    """The owner's message in the DM with the public bot, as the
    Bot API delivers it to the bot runtime."""
    return _msg("telegram-bot", chat_id=OWNER_ID, text=text,
                incoming=True, message_id=message_id, kind=ChatKind.DM)


def _dm_echo(text, message_id):
    """The same DM message as the userbot's own-echo (the owner's
    account is the userbot)."""
    return _msg("telegram", chat_id=MAIN_BOT_ID, text=text,
                incoming=False, message_id=message_id, own=True,
                kind=ChatKind.DM)


def _dm_bot_post(text, message_id, reply_to=""):
    """The bot's DM answer as the userbot's stream delivers it: an
    ordinary incoming DM message from the bot's account id. After
    the fix it quotes the command it answers."""
    return _msg("telegram", chat_id=MAIN_BOT_ID, text=text,
                incoming=True, message_id=message_id,
                sender_id=MAIN_BOT_ID, kind=ChatKind.DM,
                reply_to=reply_to)


def _dm_personal_bot_copy(text, message_id):
    """The owner's message in the DM with the PRIVATE bot, as the
    Bot API delivers it to the personal-bot runtime."""
    return _msg("telegram-personal", chat_id=OWNER_ID, text=text,
                incoming=True, message_id=message_id, kind=ChatKind.DM)


def _dm_personal_echo(text, message_id):
    """The same DM message as the userbot's own-echo (the owner's
    account is the userbot; the DM's peer is the personal bot)."""
    return _msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=text,
                incoming=False, message_id=message_id, own=True,
                kind=ChatKind.DM)


def _dm_personal_bot_post(text, message_id, reply_to=""):
    """The personal bot's DM answer as the userbot's stream
    delivers it: an ordinary incoming DM message from the
    personal bot's account id. After the fix it quotes the
    command it answers."""
    return _msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=text,
                incoming=True, message_id=message_id,
                sender_id=str(PERSONAL_BOT_ID), kind=ChatKind.DM,
                reply_to=reply_to)


def _deliver_bot_posts(rt, adapter, chat_post, base_id, command_id,
                       start=0):
    """Deliver every text the bot sent since ``start`` to the
    sibling runtime as ordinary incoming posts from the bot's
    account, each quoting the command it answers (the fixed
    send shape). Returns the new count of bot texts."""
    texts = _texts(adapter)
    for i in range(start, len(texts)):
        rt._process_inner(chat_post(texts[i], f"{base_id}-{i}",
                                    reply_to=command_id))
    return len(texts)


# ------------------------------------------------------------ repro

def test_help_echo_first_bot_answers_no_double(
        split_homes, shared_claims_file):
    """REPRODUCTION (live shape): the owner types /help in a group.
    The userbot's echo defers; the bot answers /help; the bot's
    answer lands on the userbot's stream as an ordinary group post
    (no quote — control replies never carry reply_to). The sweep
    must stand down: one /help answer in the room, never two."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_group_echo("/help", "h1"))
        assert _pending(rt_b) == 1, "the group echo must defer"
        rt_a._process_inner(_group_msg(
            "telegram-personal", "/help", "h1"))
        # The bot's /help answer is LONG: the gateway splits it
        # into chunks (the live screenshot's "(2/2)" tail).
        assert len(_texts(ad_a["telegram-personal"])) == 2, (
            "the bot must answer /help once (in 2 chunks)")
        # The send-side proof: the FIRST chunk quotes the command
        # it answers (the gateway quotes chunk one only — the
        # proof lands there, before the whole answer has been
        # delivered and its claim recorded).
        assert _reply_tos(ad_a["telegram-personal"])[0] == "h1", (
            "the first chunk of a control reply quotes the "
            "command it answers")
        # The bot's answer, observed on the userbot's stream.
        _deliver_bot_posts(rt_b, ad_a["telegram-personal"],
                           _bot_answer_post, "bot-h1", "h1")
        rt_b._sweep_peer_deferrals(now=base + 12.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer a proven group /help")
        assert _pending(rt_b) == 0
        assert len(_texts(ad_a["telegram-personal"])) == 2
    finally:
        _stop(rt_a)
        _stop(rt_b)


def test_help_late_echo_quoted_answer_stands_down(
        split_homes, broken_shared_claims):
    """FIXED (dupfix6 race): the bot's answer lands in the room
    BEFORE the echo arrives, and the store is down so no answer
    claim can save the turn. The answer quotes the command it
    answers, so the observation memory fills and the echo stands
    down without arming a failover for a command already
    answered. (On the pre-fix code this failed: the unquoted post
    never filled the memory and the echo armed a failover.)"""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        assert rt._claims_are_shared() is False
        # The bot's answer, observed first — quoting the command.
        rt._process_inner(_peer_post(
            "Devon — what I do here", "bot-h2", reply_to="h2",
            chat_id=GROUP_ID, kind=ChatKind.GROUP))
        rt._process_inner(_group_echo("/help", "h2"))
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 0, (
            "the echo must not arm a failover for an answered /help")
        rt._sweep_peer_deferrals(now=time.time() + 60.0)
        assert _texts(ad["telegram"]) == []
    finally:
        _stop(rt)


def test_ping_echo_first_bot_answers_no_double(
        split_homes, shared_claims_file):
    """CONTROL: /ping in the same shape stays single-reply — the
    shared answer claim proves the bot's answer and the sweep
    stands down. (The dupfix6 group test pins this for /ping;
    it is the contrast to the /help reproduction above.)"""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_group_echo("/ping", "p1"))
        assert _pending(rt_b) == 1, "the group echo must defer"
        rt_a._process_inner(_group_msg(
            "telegram-personal", "/ping", "p1"))
        assert len(_texts(ad_a["telegram-personal"])) == 1
        _deliver_bot_posts(rt_b, ad_a["telegram-personal"],
                           _bot_answer_post, "bot-p1", "p1")
        rt_b._sweep_peer_deferrals(now=base + 12.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer a proven group /ping")
        assert _pending(rt_b) == 0
    finally:
        _stop(rt_a)
        _stop(rt_b)


# ------------------------------------------------- DM /help (owner evidence 2)

def test_dm_help_echo_first_bot_answers_no_double(
        split_homes, shared_claims_file):
    """REPRODUCTION (DM shape): the owner types /help in his
    private DM with the public bot. The userbot's echo defers; the
    bot answers /help in the DM; the bot's answer lands on the
    userbot's stream as an ordinary incoming DM message from the
    bot, quoting the command. The sweep must stand down: one /help
    answer in the DM, never two."""
    home_a, home_b = split_homes
    ad_a = {"telegram-bot": BotAdapter(
        "telegram-bot", int(MAIN_BOT_ID), "Nomorals_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_dm_echo("/help", "dm1"))
        assert _pending(rt_b) == 1, "the DM echo must defer"
        rt_a._process_inner(_dm_bot_copy("/help", "dm1"))
        assert len(_texts(ad_a["telegram-bot"])) >= 1, (
            "the bot must answer /help in the DM")
        assert _reply_tos(ad_a["telegram-bot"]) == (
            ["dm1"] * len(_texts(ad_a["telegram-bot"]))), (
            "every control reply quotes the command it answers")
        _deliver_bot_posts(rt_b, ad_a["telegram-bot"],
                           _dm_bot_post, "bot-dm1", "dm1")
        rt_b._sweep_peer_deferrals(now=base + 25.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer a proven DM /help")
        assert _pending(rt_b) == 0
    finally:
        _stop(rt_a)
        _stop(rt_b)


def test_dm_help_late_echo_quoted_answer_stands_down(
        split_homes, broken_shared_claims):
    """FIXED (DM dupfix6 race): the bot's DM /help answer lands
    BEFORE the echo arrives, and the store is down so no answer
    claim can save the turn. The answer quotes the command, so
    the observation memory fills and the echo stands down
    without arming a failover. (On the pre-fix code this failed:
    the unquoted post never filled the memory and the echo armed
    a failover for a command already answered.)"""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        assert rt._claims_are_shared() is False
        rt._process_inner(_peer_post(
            "menu", "bot-dm2", reply_to="dm2",
            chat_id=MAIN_BOT_ID, kind=ChatKind.DM,
            sender_id=MAIN_BOT_ID))
        rt._process_inner(_dm_echo("/help", "dm2"))
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 0, (
            "the echo must not arm a failover for an answered /help")
        rt._sweep_peer_deferrals(now=time.time() + 60.0)
        assert _texts(ad["telegram"]) == []
    finally:
        _stop(rt)


# ------------------------------------------------- universal: status + play

def test_status_echo_first_bot_answers_no_double(
        split_homes, shared_claims_file):
    """UNIVERSAL shape: /status in a group — same echo-first
    shape as /help. One answer in the room, never two."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_group_echo("/status", "s1"))
        assert _pending(rt_b) == 1, "the group echo must defer"
        rt_a._process_inner(_group_msg(
            "telegram-personal", "/status", "s1"))
        assert len(_texts(ad_a["telegram-personal"])) >= 1, (
            "the bot must answer /status once")
        _deliver_bot_posts(rt_b, ad_a["telegram-personal"],
                           _bot_answer_post, "bot-s1", "s1")
        rt_b._sweep_peer_deferrals(now=base + 12.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer a proven /status")
        assert _pending(rt_b) == 0
    finally:
        _stop(rt_a)
        _stop(rt_b)


def test_play_second_invocation_no_double(
        split_homes, shared_claims_file):
    """UNIVERSAL shape (owner evidence 2): /play in a group,
    TWICE. The first invocation is single-reply; the second
    invocation must be single-reply too — whatever state the
    first leaves behind must not let the second one's echo arm
    a failover onto the bot's answer."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        for i, mid in enumerate(("pl1", "pl2")):
            rt_b._process_inner(_group_echo("/play", mid))
            assert _pending(rt_b) == 1, (
                f"invocation {i + 1}: the group echo must defer")
            before = len(_texts(ad_a["telegram-personal"]))
            rt_a._process_inner(_group_msg(
                "telegram-personal", "/play", mid))
            assert len(_texts(ad_a["telegram-personal"])) > before, (
                f"invocation {i + 1}: the bot must answer /play")
            _deliver_bot_posts(rt_b, ad_a["telegram-personal"],
                               _bot_answer_post, f"bot-{mid}", mid,
                               start=before)
            rt_b._sweep_peer_deferrals(now=base + 12.0 + i * 20.0)
            assert _texts(ad_b["telegram"]) == [], (
                f"invocation {i + 1}: the sweep must not "
                "double-answer /play")
            assert _pending(rt_b) == 0, (
                f"invocation {i + 1}: no deferral must be left")
    finally:
        _stop(rt_a)
        _stop(rt_b)


# ------------------------------------------------- private-bot DM
# (owner follow-up, 2026-11-24: "what about private bot that
# user bot is the one answering"). The lane's DM tests used the
# PUBLIC bot (telegram-bot) as the answerer; these use the
# PRIVATE bot (telegram-personal, @De_von_bot) — the DM the
# owner actually lives in. The machinery is shared (same
# runtime, one Telegram claim family, same DM deferral-context
# path), but "looks shared" is not proof: these shapes run it.
#
# One correction to the brief's premise, from repo truth: a
# TYPED repeat bare /help does NOT edit — the menu home opener
# always fresh-sends (menu_session_render home branch,
# session_edit=continuation, and continuation=tapped=False for
# a typed message). The true edit shapes are topic renders
# (/list <section>, session_edit=True always) and tapped
# continuations. The edit test below uses /list play after a
# first /help established the live menu message.


def test_private_dm_help_echo_first_bot_answers_no_double(
        split_homes, shared_claims_file):
    """PRIVATE-BOT DM /help, echo-first: the owner types /help
    in his DM with @De_von_bot. The userbot's echo defers; the
    personal bot answers /help in its DM, quoting the command;
    the answer lands on the userbot's stream as an ordinary
    incoming DM message from the personal bot. The sweep must
    stand down: one /help answer in the DM, never two."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_dm_personal_echo("/help", "pd1"))
        assert _pending(rt_b) == 1, "the private-DM echo must defer"
        rt_a._process_inner(_dm_personal_bot_copy("/help", "pd1"))
        assert len(_texts(ad_a["telegram-personal"])) >= 1, (
            "the personal bot must answer /help in its DM")
        assert _reply_tos(ad_a["telegram-personal"]) == (
            ["pd1"] * len(_texts(ad_a["telegram-personal"]))), (
            "every control reply quotes the command it answers")
        _deliver_bot_posts(rt_b, ad_a["telegram-personal"],
                           _dm_personal_bot_post, "bot-pd1", "pd1")
        rt_b._sweep_peer_deferrals(now=base + 25.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer a proven "
            "private-DM /help")
        assert _pending(rt_b) == 0
    finally:
        _stop(rt_a)
        _stop(rt_b)


def test_private_dm_list_topic_edits_no_double(
        split_homes, shared_claims_file):
    """EDIT shape (private-bot DM): the owner types /help, the
    personal bot posts the live menu; then /list play — a topic
    render (session_edit=True) — EDITS the live menu message
    instead of posting a follow-up. No new post exists to
    observe, so the deferral stands down on the shared answer
    claim alone (recorded on edit success). The sweep at +25s
    past the /list echo must send nothing."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_dm_personal_echo("/help", "pe1"))
        assert _pending(rt_b) == 1, "the /help echo must defer"
        rt_a._process_inner(_dm_personal_bot_copy("/help", "pe1"))
        assert len(_texts(ad_a["telegram-personal"])) >= 1, (
            "the personal bot must post the live menu")
        _deliver_bot_posts(rt_b, ad_a["telegram-personal"],
                           _dm_personal_bot_post, "bot-pe1", "pe1")
        rt_b._sweep_peer_deferrals(now=base + 25.0)
        assert _texts(ad_b["telegram"]) == []
        assert _pending(rt_b) == 0
        sent_before = len(_texts(ad_a["telegram-personal"]))
        rt_b._process_inner(_dm_personal_echo("/list play", "pe2"))
        assert _pending(rt_b) == 1, "the /list echo must defer"
        rt_a._process_inner(_dm_personal_bot_copy("/list play", "pe2"))
        assert len(_texts(ad_a["telegram-personal"])) == sent_before, (
            "the topic render must EDIT the live menu, not post")
        assert len(ad_a["telegram-personal"].edits) == 1, (
            "exactly one edit must land on the live menu message")
        # The harness names the first send "m1": the edit must
        # target the live menu message the first /help posted.
        assert ad_a["telegram-personal"].edits[0][0] == "m1", (
            "the edit must target the live menu message")
        # No post to observe — the answer claim alone must stand
        # the sweep down.
        rt_b._sweep_peer_deferrals(now=base + 50.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer an edit-proven /list")
        assert _pending(rt_b) == 0
    finally:
        _stop(rt_a)
        _stop(rt_b)


def test_private_dm_play_and_status_no_double(
        split_homes, shared_claims_file):
    """UNIVERSAL shape (private-bot DM): /play and /status in
    the owner's DM with @De_von_bot — the personal bot answers,
    the userbot's echo defers, the sweep stands down. One reply
    each, never two, and the first chunk of each answer quotes
    the command it answers."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        for i, (cmd, mid) in enumerate((
                ("/play", "pp1"), ("/status", "ps1"))):
            rt_b._process_inner(_dm_personal_echo(cmd, mid))
            assert _pending(rt_b) == 1, f"{cmd}: echo must defer"
            before = len(_texts(ad_a["telegram-personal"]))
            rt_a._process_inner(_dm_personal_bot_copy(cmd, mid))
            assert len(_texts(ad_a["telegram-personal"])) > before, (
                f"{cmd}: the personal bot must answer in its DM")
            assert _reply_tos(ad_a["telegram-personal"])[before] == mid, (
                f"{cmd}: the first chunk quotes the command")
            _deliver_bot_posts(rt_b, ad_a["telegram-personal"],
                               _dm_personal_bot_post, f"bot-{mid}", mid,
                               start=before)
            rt_b._sweep_peer_deferrals(now=base + 25.0 + i * 20.0)
            assert _texts(ad_b["telegram"]) == [], (
                f"{cmd}: the sweep must not double-answer")
            assert _pending(rt_b) == 0
    finally:
        _stop(rt_a)
        _stop(rt_b)


def test_private_dm_bot_down_userbot_answers_alone_once(
        split_homes, shared_claims_file):
    """ALONE shape (the failover working, not a double): the
    personal bot never answers the DM command — no claim, no
    answer, no post. The sweep fails over at step (4) and the
    userbot answers the command ITSELF, exactly once. One reply
    in the room, from the userbot alone — the designed cover
    for a down bot, not a defect. A second sweep adds nothing."""
    home_a, home_b = split_homes
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_dm_personal_echo("/help", "pd9"))
        assert _pending(rt_b) == 1, "the echo must defer"
        # The personal bot never answers: no copy delivered, no
        # claim, no post. The sweep fires past the 20s DM window.
        rt_b._sweep_peer_deferrals(now=base + 25.0)
        assert len(_texts(ad_b["telegram"])) == 1, (
            "the failover must answer exactly once when the "
            "personal bot never answers")
        assert _pending(rt_b) == 0
        rt_b._sweep_peer_deferrals(now=base + 60.0)
        assert len(_texts(ad_b["telegram"])) == 1, (
            "a second sweep must not answer again")
    finally:
        _stop(rt_b)
