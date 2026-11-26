"""Lane 138 (helpdup2) — the residual group double + the orchestrator
ack message.

Owner-verified 2026-11-25 on the live new box: one /help in the
Telegram group "Devon" produced FIVE messages — the bot's catalog
(2 chunks), the userbot's copy of it, a planning note ("multi-part
goal — the orchestrator is planning it:"), a bot ack
("acknowledged. working on devon do here everyone: rules group
rules report."), and a done note ("orchestrator: done (3 steps) …").

Q1 root cause: the bot's chunk-1 post was never recognised as a
peer post on the userbot side — _sender_is_bot failed (tier 3 dead
for bot senders: a bot's from_id is a PeerUser with no .bot
attribute; the guard rested on the companion set built at adapter
construction plus Telethon sender resolution). Timing is refuted:
the orchestrator fired on the help text, proving the post was
ingested as a normal turn.

Q2 root cause: the bot's catalog was ingested on the userbot as an
owner turn (CoreMind's gate is chat-based; the help text trips
_decide's multi trigger; no group suppression on _send_mind_reply /
_send_long), and on the bot side the userbot's failover answer fell
through both peer-post grounds (not a bot sender; slash commands
exempt from the conversational claim ground), so it reached the
brain, whose reply inherited the /help quote.

Rig copied from tests/test_gapfix_helpdup.py: two runtimes,
separate homes, one shared claims file when the test needs it.
"""

import contextlib
import json
import os
import tempfile
import time
import types
from pathlib import Path

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.coremind import CoreMind, Intent
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
from nomorals.social.chat.telegram import TelegramAdapter

_TEST_TMPDIR = (os.environ.get("NM_TEST_TMPDIR")
                or os.environ.get("TMPDIR")
                or os.path.expanduser("~/workspace/helpdup2fix-verify/tmp"))

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888
MAIN_BOT_ID = "8133295902"
GROUP_ID = "-1001234567890"

CATALOG_CHUNK_1 = (
    "🛠 Devon — what I do here\n\nfor everyone:\n"
    "/rules — the group rules\n"
    "/report — report a member\n"
    "/top — top sparks\n"
    "/treasury — the group treasury")
DONE_NOTE = (
    "orchestrator: done (3 steps)\n"
    "[✓] research\n[✓] execute\n[✓] verify\n"
    "⚠️ degraded plan: model returned no usable plan — "
    "using template plan")
MULTI_TEXT = "research gold and build a dashboard and download the data"


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


def _context(home: str, *, tokens: bool = False):
    overrides = {
        "home": home,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.enabled": "false",
        "web.bot_username": "Nomorals_bot",
    }
    if tokens:
        # The runtime's own settings carry both bot tokens — the
        # live shape on the new box. The ids are the token heads.
        overrides["chat.telegram_bot_token"] = f"{MAIN_BOT_ID}:secret"
        overrides["chat.telegram_personal_bot_token"] = \
            f"{PERSONAL_BOT_ID}:secret"
    settings = load_settings(overrides=overrides)
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
        ("helpdup2", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    import random
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


def _texts(adapter):
    return [t for t, _r, _b in adapter.sent]


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
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-hd2-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-hd2-a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-hd2-b-",
                                        dir=_TEST_TMPDIR)
    yield tmp_a.name, tmp_b.name
    for tmp in (tmp_a, tmp_b, home_tmp):
        tmp.cleanup()


def _make_runtime(home, adapters, *, tokens=False):
    ctx = _context(home, tokens=tokens)
    return _runtime(ctx, adapters), ctx


def _stop(rt):
    with contextlib.suppress(Exception):  # noqa: BLE001 - teardown only
        rt.stop()


# ------------------------------------------------------------ Q1-1


def test_settings_bot_tokens_seed_bot_registry(split_homes):
    """Q1-1: the bot ids straight from this runtime's own settings
    tokens — no adapter state, no companion set, no Telethon
    sender resolution, no shared registry. On the pre-fix code
    the registry is empty here (the userbot adapter carries no
    bot id and no companions), so peer recognition has nothing
    to match against."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad, tokens=True)
    try:
        ids = rt._bot_account_ids()
        assert MAIN_BOT_ID in ids, (
            "the main bot id from the settings token must seed "
            "the bot registry")
        assert str(PERSONAL_BOT_ID) in ids
        # And the numeric sender check built on it:
        bot_post = _group_msg("telegram", CATALOG_CHUNK_1, "bot-h1",
                              sender_id=MAIN_BOT_ID, reply_to="h1")
        assert rt._sender_is_bot_account(bot_post) is True
        owner_post = _group_msg("telegram", "/help", "h1",
                                sender_id=OWNER_ID)
        assert rt._sender_is_bot_account(owner_post) is False, (
            "the owner account id must never read as a bot account")
    finally:
        _stop(rt)


def test_bot_group_post_recognised_by_sender_id_without_companions(
        split_homes, shared_claims_file):
    """Q1-1 end to end (the live double, minus the companions):
    the userbot's stream carries NO companion bot ids and the
    shared registry holds nothing for the main bot — the only
    source of its id is this runtime's settings token. The
    owner's /help defers; the bot's catalog chunk lands as an
    ordinary incoming group post from the bot's id, quoting the
    command. The deferral must resolve on it: one /help answer
    in the room, never two. On the pre-fix code the post is
    ingested as a normal turn and the sweep fails over into a
    second answer."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad, tokens=True)
    try:
        base = time.time()
        rt._process_inner(_group_echo("/help", "h1"))
        assert _pending(rt) == 1, "the group echo must defer"
        rt._process_inner(_group_msg(
            "telegram", CATALOG_CHUNK_1, "bot-h1",
            sender_id=MAIN_BOT_ID, reply_to="h1"))
        assert _pending(rt) == 0, (
            "the bot's catalog post must resolve the deferral")
        assert _texts(ad["telegram"]) == [], (
            "nothing may be sent for a peer post")
        rt._sweep_peer_deferrals(now=base + 12.0)
        assert _texts(ad["telegram"]) == [], (
            "the sweep must not double-answer a proven group /help")
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ------------------------------------------------------------ Q1-2


def test_control_answer_self_observes_in_room(
        split_homes, broken_shared_claims):
    """Q1-2: when a surface sends a group command answer, the
    observation is recorded at send time — in the same memory
    the sweep reads — instead of waiting for the Telethon
    round-trip to deliver our own post back to us. The shared
    store is broken here, so no ledger claim can explain a
    stand-down: only the self-observation can. On the pre-fix
    code the memory is empty after the send."""
    home_a, _home_b = split_homes
    ad = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        assert rt._claims_are_shared() is False
        cmd = _group_msg("telegram-personal", "/help", "h1")
        rt._process_inner(cmd)
        assert len(_texts(ad["telegram-personal"])) >= 1, (
            "the bot must answer /help once")
        chat_key = str(cmd.chat.key)
        assert rt._already_answered_in_room(chat_key, "h1") is True, (
            "the send itself must record the in-room observation")
        assert rt._room_has_peer_answer(chat_key, "h1") is True, (
            "the send itself must journal the room proof too")
    finally:
        _stop(rt)


# ------------------------------------------------------------ Q1-3


def test_sender_is_bot_numeric_from_id():
    """Q1-3: _sender_is_bot tier 3 — the raw message's from_id
    carries the sender's NUMERIC id, compared against the
    companion set. The old ``from_id.bot`` read is dead for bot
    senders (a bot's from_id is a PeerUser, which carries no
    .bot attribute), so on the pre-fix code this returns False
    whenever Telethon never resolved the sender entity."""
    adapter = object.__new__(TelegramAdapter)
    adapter._companion_bot_ids = {PERSONAL_BOT_ID}
    adapter._companion_bot_id = PERSONAL_BOT_ID

    def _event_message(user_id):
        event = types.SimpleNamespace(
            sender_id=None, sender=None,
            message=types.SimpleNamespace(sender_id=None))
        # A PeerUser shape: user_id only, no .bot attribute.
        message = types.SimpleNamespace(
            sender_id=None,
            from_id=types.SimpleNamespace(user_id=user_id))
        return event, message

    event, message = _event_message(PERSONAL_BOT_ID)
    assert adapter._sender_is_bot(event, message) is True, (
        "a numeric from_id matching a companion must read as a bot")
    event, message = _event_message(999999)
    assert adapter._sender_is_bot(event, message) is False, (
        "an unknown id must not read as a bot")
    adapter._companion_bot_ids = set()
    adapter._companion_bot_id = None
    event, message = _event_message(PERSONAL_BOT_ID)
    assert adapter._sender_is_bot(event, message) is False, (
        "with no companion ids known, an id alone proves nothing")


# ------------------------------------------------------------ Q1-4


def test_room_journal_stands_down_sweep_when_recognition_missed(
        split_homes, shared_claims_file, monkeypatch):
    """Q1-4: the room journal is the sweep's second proof source.
    Recognition is patched to miss entirely (the live shape:
    empty companion set, unresolved sender, no claim) — the
    journal still fills for every incoming group post, and the
    sweep stands down on it. The shared store works here, but
    the bot never answered through a runtime, so no answer
    claim exists for the sweep's step (1) to find. On the
    pre-fix code there is no journal and the sweep fails over
    into a second /help answer."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad, tokens=True)
    try:
        # Total recognition miss, simulated: both peer-post
        # grounds return False for every post.
        monkeypatch.setattr(rt, "_sender_is_bot_account",
                            lambda m: False, raising=False)
        monkeypatch.setattr(rt, "_is_own_surface_group_post",
                            lambda m: False)
        base = time.time()
        rt._process_inner(_group_echo("/help", "h1"))
        assert _pending(rt) == 1, "the group echo must defer"
        # The bot's catalog post, as the stream delivers it:
        # an ordinary incoming group message from the bot's id,
        # quoting the command.
        rt._process_inner(_group_msg(
            "telegram", CATALOG_CHUNK_1, "bot-h1",
            sender_id=MAIN_BOT_ID, reply_to="h1"))
        sent_before = len(_texts(ad["telegram"]))
        assert _pending(rt) == 1, (
            "the journal proves at sweep time, not at delivery")
        rt._sweep_peer_deferrals(now=base + 12.0)
        assert _pending(rt) == 0, (
            "the sweep must stand down on the room journal")
        assert len(_texts(ad["telegram"])) == sent_before, (
            "the sweep must not double-answer a journal-proven "
            "group /help")
    finally:
        _stop(rt)


# ------------------------------------------------------------ Q2-2


def test_orchestrator_status_suppressed_in_group(split_homes):
    """Q2-2 (_send_mind_reply): an orchestrator status note never
    goes to a GROUP — the planning note and the done note are
    internal narration about Devon's own work, journal-only
    there. The turn is still consumed and the job still runs;
    only the post is suppressed. In a DM the same note still
    goes out (the suppression is group-only). On the pre-fix
    code the group post goes out."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        group_msg = _group_msg("telegram", "/help", "h1")
        rt._send_mind_reply(
            group_msg,
            "multi-part goal — the orchestrator is planning it:\n"
            "\"🛠 Devon — what I do here\"")
        assert _texts(ad["telegram"]) == [], (
            "the planning note must never reach the group")
        rt._send_mind_reply(group_msg, DONE_NOTE)
        assert _texts(ad["telegram"]) == [], (
            "the done note must never reach the group")
        # A normal mind reply in the same group still goes out.
        rt._send_mind_reply(group_msg, "here is what I found")
        assert _texts(ad["telegram"]) == ["here is what I found"]
        # And the status note in a DM still goes out.
        dm_msg = _msg("telegram", chat_id=MAIN_BOT_ID, text="/help",
                      incoming=True, message_id="dm1", kind=ChatKind.DM)
        rt._send_mind_reply(dm_msg, DONE_NOTE)
        assert DONE_NOTE in _texts(ad["telegram"]), (
            "the suppression is group-only")
    finally:
        _stop(rt)


def test_start_run_suppresses_orchestrator_done_in_group(split_homes):
    """Q2-2 (_start_run): the multi job's done summary is
    delivered from _start_run directly — it never passes through
    _send_mind_reply, so it needs its own gate. The staged gate
    (gapfix166's _is_group_key) reads the group surface's store,
    not the rebuilt ref — so the fake runtime carries a group
    surface holding the group key. (The lane's own gate read the
    ref's negative chat id; the staged tree keeps 166's form,
    which is the one that also sees WhatsApp group keys.)
    On the pre-fix code the done note goes to the group."""
    home_a, _home_b = split_homes
    ctx = _context(home_a)

    class _FakeRuntime:
        """Mirrors runtime._ref_from_key: kind is dropped, the
        ref carries platform + chat id only."""

        def __init__(self):
            self.sent = []

        def _ref_from_key(self, key):
            plat, _, cid = key.partition(":")
            return ChatRef(platform=plat or "local",
                           chat_id=cid or key)

        def _group_surface(self):
            # The staged gate reads the group surface's store: a
            # room the intake has seen is there under its own key.
            # The fake carries exactly the group key, so the gate
            # sees a group; the DM key is absent, so the DM
            # control below still sends.
            return types.SimpleNamespace(
                store=types.SimpleNamespace(
                    data={f"telegram:{GROUP_ID}": {}}))

        def _send_long(self, platform, chat, text, limit=3800):
            self.sent.append((str(chat.chat_id), text))
            return 1

    fake = _FakeRuntime()
    mind = CoreMind(ctx, runtime=fake)
    try:
        # _send_async is the real entry point: it acquires the
        # slot, then _start_run owns the worker thread. Calling
        # _start_run directly would release a slot that was never
        # acquired and the worker would die in the finally.
        job_id = mind._new_job(Intent(
            "multi", 0.8, target=MULTI_TEXT, route="orchestrator"))
        mind._send_async(f"telegram:{GROUP_ID}", lambda: DONE_NOTE,
                         job_id, "started", "multi")
        # _job_done marks the row done BEFORE the notify sends,
        # so poll for the row first, then give the notify a
        # moment to settle before asserting nothing was sent.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            row = mind._job_row(job_id)
            if row is not None and row.get("status") in ("done", "failed"):
                break
            time.sleep(0.05)
        time.sleep(0.5)
        assert fake.sent == [], (
            "the orchestrator done note must never reach the group")
        # Control: the same done note in a DM still goes out.
        job_id2 = mind._new_job(Intent(
            "multi", 0.8, target=MULTI_TEXT, route="orchestrator"))
        mind._send_async("telegram:4242", lambda: DONE_NOTE,
                         job_id2, "started", "multi")
        deadline = time.time() + 10.0
        while time.time() < deadline and len(fake.sent) < 1:
            time.sleep(0.05)
        assert len(fake.sent) == 1, (
            "the suppression is group-only — the DM still hears it")
        assert fake.sent[0][1] == DONE_NOTE
    finally:
        # Nothing to stop: the fake runtime holds no threads and
        # the worker threads are daemons.
        pass


# ------------------------------------------------------------ Q2-3


def test_multi_trigger_excludes_command_answer(split_homes):
    """Q2-3: a message that is itself a command answer — the
    catalog text, or a message quoting a command — never trips
    the multi trigger. Only the multi branch closes: the best
    single intent still fires, and chat still falls through.
    On the pre-fix code the catalog-headed text trips multi
    (it carries two strong signals and a joiner)."""
    home_a, _home_b = split_homes
    ctx = _context(home_a)
    mind = CoreMind(ctx, runtime=None)
    # Control: the bare text trips multi.
    assert mind.decide(MULTI_TEXT, allow_model=False).kind == "multi"
    # The catalog-headed version of the same text must not.
    catalog_text = (CATALOG_CHUNK_1 + "\n" + MULTI_TEXT)
    assert mind.decide(catalog_text, allow_model=False).kind != "multi", (
        "the catalog text must never trip the multi trigger")
    # A multi-tripping text that quotes a command must not either.
    quoting = _group_msg("telegram", MULTI_TEXT, "q1", reply_to="h1")
    assert mind.decide(MULTI_TEXT, allow_model=False,
                       message=quoting).kind != "multi", (
        "a message quoting a command must never trip multi")
    # And a multi-tripping text that quotes nothing still trips.
    plain = _group_msg("telegram", MULTI_TEXT, "q2")
    assert mind.decide(MULTI_TEXT, allow_model=False,
                       message=plain).kind == "multi"


# ------------------------------------------------------------ Q2-4


def test_group_command_answer_leaves_conversational_claim(
        split_homes, shared_claims_file):
    """Q2-4a: a group control answer leaves a CONVERSATIONAL claim
    in the chat, so the sibling's peer-post ground (ii) fires
    when its own failover answer arrives a few seconds later.
    On the pre-fix code no conversational claim exists after
    the answer."""
    home_a, _home_b = split_homes
    ad = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        rt._process_inner(_group_msg("telegram-personal", "/help", "h1"))
        assert len(_texts(ad["telegram-personal"])) >= 1, (
            "the bot must answer /help once")
        ledger = rt._shared_conversation_claims()
        assert ledger is not None
        assert ledger.has_live_claim_in_chat(
            "telegram", GROUP_ID,
            rt._claim_surface("telegram-personal")) is True, (
            "the group /help answer must leave a live "
            "conversational claim in the chat")
    finally:
        _stop(rt)


def test_sibling_answer_recognised_via_quoted_command(
        split_homes, shared_claims_file):
    """Q2-4b end to end (the live ack, message 4): the bot answers
    /help in the group; the userbot's failover answer — the
    owner's account posting the catalog, quoting /help — lands
    on the bot runtime. It must read as a sibling surface's
    answer (the shared command ledger carries the quoted
    command's claim/answer), never reach the brain, and never
    earn an ack. On the pre-fix code it falls through both
    peer-post grounds and the brain replies to it."""
    home_a, _home_b = split_homes
    ad = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        # The owner registry knows the owner on the live box;
        # seed it here so ground (ii)'s owner check can pass.
        rt._owner_registry().seed(telegram_ids=(OWNER_ID,))
        rt._process_inner(_group_msg("telegram-personal", "/help", "h1"))
        assert len(_texts(ad["telegram-personal"])) == 2, (
            "the bot must answer /help once (in 2 chunks)")
        # The userbot's failover answer: the owner's account,
        # the catalog, quoting /help.
        rt._process_inner(_group_msg(
            "telegram-personal", CATALOG_CHUNK_1, "ub-h1",
            sender_id=OWNER_ID, reply_to="h1"))
        assert len(_texts(ad["telegram-personal"])) == 2, (
            "the sibling's answer must earn no reply — no ack, "
            "no second catalog")
        assert _pending(rt) == 0
    finally:
        _stop(rt)
