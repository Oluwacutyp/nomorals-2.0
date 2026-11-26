"""Lane 166 (modnoreply) — the false "ran but produced no reply"
after a group moderation command, the double dispatch behind it,
and the orchestrator noise on group /help.

Owner-verified 2026-11-25 on the live new box (screen recording):
in the WhatsApp group "NOISY NEIGHBOR - Perfect Match", /mute,
/kick, /ban (target 6037101370) and /minactive each WORKED — the bot
posted the confirmation plus the theatre GIF — and after EACH one a
second message appeared from his personal account (the userbot):
"/mute ran but produced no reply — that's a bug on my end, not
something you did." The bug message is FALSE: the command ran and
produced a reply. The same video shows his group /help getting the
orchestrator planning note plus a done note carrying provider
cooling-down detail.

Root causes, all proven against the repo at 62a66a75:

1. The theatre send never marks a delivery. _admin_theatre fires
   the moment with the reply as its caption and returns "" on
   success, but never calls _note_control_delivery() — and the fire
   path never touches _send_long. handle_control's #32 law then
   converts the "" into the false bug text whenever the count did
   not rise during the dispatch.
2. A theatre answer leaves no proof for the failover sweep.
   _run_control_command skips _send_control_reply when the reply
   is "", so _note_control_answer never runs and no answer claim
   lands in the shared ledger; the GIF post carries no reply_to,
   so the in-room observation cannot link it either. The sweep
   therefore re-executes the moderation command on the sibling
   once the claim expires.
3. Orchestrator status notes (planning + done) reach groups
   through _send_mind_reply and _start_run's notify with no
   group suppression.

Rig copied from tests/test_gapfix_dupfix6.py (two runtimes,
separate homes, one shared claims file) + the group moderation
rig (seeded member ledger, admin adapter).
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
                or os.path.expanduser("~/workspace/modnoreply-verify/tmp"))

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888
MAIN_BOT_ID = "8133295902"
GROUP_ID = "-1001234567890"
GROUP_KEY = "telegram-bot:-1001234567890"

PLANNING_NOTE = (
    "multi-part goal — the orchestrator is planning it:\n"
    "“Devon — what do I do here for everyone: /rules — the group rules”")
DONE_NOTE = (
    "orchestrator: done (3 steps)\n"
    "[✓] research\n[✓] execute\n[✓] verify\n"
    "⚠ degraded plan: model call failed (All language models are "
    "unreachable right now — I can't generate a fresh answer. "
    "primary: RuntimeError: no chat provider available; openrouter "
    "(cooling down), groq (cooling down); all providers are cooling "
    "down) — using template plan")


class GroupAdapter(ChatAdapter):
    """Bot-surface adapter: admin verbs + media send, all recorded."""

    def __init__(self, name: str, bot_id: int) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self._bot_id = bot_id
        self._bot_username = "Nomorals_bot"
        self.sent: list = []
        self.media: list = []
        self.calls: list = []
        self.edits: list = []

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

    def send_media(self, chat, ref, *, caption: str = "",
                   buttons=None) -> SendResult:
        self.media.append((ref, caption, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"gif{len(self.media)}")

    def edit_message(self, chat, message_id, text, *, buttons=None,
                     parse_mode=""):
        self.edits.append((str(message_id), text, buttons))
        return {"ok": True}

    def admin_member(self, chat, user_id):
        return {"status": "administrator"}

    def admin_restrict(self, chat, user_id, *, minutes=60):
        self.calls.append(("restrict", user_id, minutes))
        return {"ok": True}

    def admin_unrestrict(self, chat, user_id):
        self.calls.append(("unrestrict", user_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", user_id))
        return {"ok": True}

    def admin_kick(self, chat, user_id):
        self.calls.append(("kick", user_id))
        return {"ok": True}


class BareUserbotAdapter(ChatAdapter):
    """The split-hosting userbot: its own account identity, and
    NO companion bot ids — the documented live shape."""

    _me_username = "peacethefirst"
    _me_id = int(OWNER_ID)

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list = []
        self.media: list = []
        self.calls: list = []

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

    def send_media(self, chat, ref, *, caption: str = "",
                   buttons=None) -> SendResult:
        self.media.append((ref, caption, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"gif{len(self.media)}")

    def admin_member(self, chat, user_id):
        return {"status": "administrator"}

    def admin_restrict(self, chat, user_id, *, minutes=60):
        self.calls.append(("restrict", user_id, minutes))
        return {"ok": True}

    def admin_unrestrict(self, chat, user_id):
        self.calls.append(("unrestrict", user_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", user_id))
        return {"ok": True}

    def admin_kick(self, chat, user_id):
        self.calls.append(("kick", user_id))
        return {"ok": True}


class FakeExpressionService:
    """Stands in for the media library only: fire() hands the
    send seam a canned MediaRef and reports the send's own
    outcome. The theatre, the system-output gate, the audit and
    the adapter send are all the real code under test."""

    def __init__(self, *, sent: bool = True) -> None:
        self._sent = sent
        self.fired: list = []

    def fire(self, moment, *, room="", platform="", settings=None,
             send):
        from nomorals.social.chat.base import MediaRef

        self.fired.append(str(moment))
        if not self._sent:
            return {"sent": False, "moment": str(moment),
                    "reason": "no media available — text only"}
        ref = MediaRef(path="/tmp/nm-test-media/mute.gif",
                       kind="animation", name=f"expression {moment}")
        result = send(ref)
        if not getattr(result, "ok", False):
            return {"sent": False, "moment": str(moment),
                    "reason": f"send failed: {getattr(result, 'error', '')}"}
        return {"sent": True, "moment": str(moment), "item_id": "fake"}


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
        ("modnoreply", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    runtime._start_peer_deferral_sweeper = lambda: None
    return runtime


def _make_runtime(home, adapters):
    return _runtime(_context(home), adapters)


def _group_msg(platform, text, message_id, *, sender_id=OWNER_ID):
    chat = ChatRef(platform=platform, chat_id=GROUP_ID,
                   kind=ChatKind.GROUP, peer=GROUP_ID)
    msg = ChatMessage(chat=chat, incoming=True, text=text,
                      sender="death", message_id=message_id,
                      meta={"is_owner": True})
    msg.sender_id = sender_id
    return msg


def _group_echo(text, message_id):
    """The owner's own message in a group, as the userbot sees it."""
    chat = ChatRef(platform="telegram", chat_id=GROUP_ID,
                   kind=ChatKind.GROUP, peer=GROUP_ID)
    msg = ChatMessage(chat=chat, incoming=False, text=text,
                      sender="death", message_id=message_id,
                      meta={"is_owner": True, "own_message": True,
                            "from_owner": True})
    msg.sender_id = OWNER_ID
    return msg


def _texts(adapter):
    return [t for t, _r, _b in adapter.sent]


def _pending(runtime):
    return runtime._peer_deferral().pending_count()


def _stop(rt):
    with contextlib.suppress(Exception):  # noqa: BLE001 - teardown only
        rt.stop()


@pytest.fixture
def shared_claims_file(monkeypatch, tmp_path):
    path = tmp_path / "claims.db"
    monkeypatch.setenv("NM_SHARED_CLAIMS", str(path))
    return str(path)


@pytest.fixture
def split_homes(monkeypatch):
    clear_failures()
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-mnr-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-mnr-a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-mnr-b-",
                                        dir=_TEST_TMPDIR)
    yield tmp_a.name, tmp_b.name
    for tmp in (tmp_a, tmp_b, home_tmp):
        tmp.cleanup()


def _designate(runtime):
    """Mark the group claimed in this runtime's surface store
    (designated + bot admin + present) — the live shape: the
    group IS claimed, so the theatre gate (system_output_allowed)
    passes on the userbot surface too. The bot surfaces pass the
    gate by platform alone; the userbot needs the claim."""
    surface = runtime._group_surface()
    if GROUP_KEY not in surface.store.data:
        surface.handle_message(GROUP_KEY, "777", username="loud",
                               text="hi")
    # system_claim lives under entry["settings"] — the store's
    # merged view reads it from there (peek_settings), not from
    # the entry top level.
    surface.store.data[GROUP_KEY].setdefault(
        "settings", {})["system_claim"] = {
        "designated": True, "bot_admin": True, "bot_present": True}


@pytest.fixture
def bot_runtime(tmp_path):
    """One bot runtime in a tmp home, member 'loud' (777) seeded
    in the group ledger, theatre wired to the fake service."""
    home = tempfile.TemporaryDirectory(prefix="nm-mnr-bot-",
                                       dir=_TEST_TMPDIR)
    adapter = GroupAdapter("telegram-bot", int(MAIN_BOT_ID))
    runtime = _make_runtime(home.name, {"telegram-bot": adapter})
    runtime._expression_service_obj = FakeExpressionService(sent=True)
    surface = runtime._group_surface()
    surface.handle_message(GROUP_KEY, "777", username="loud",
                           text="hi")
    yield runtime, adapter
    _stop(runtime)
    home.cleanup()


def _mute_texts(runtime, adapter, text, message_id):
    msg = _group_msg("telegram-bot", text, message_id)
    return runtime.handle_control(text, GROUP_KEY, message=msg)


# ------------------------------------------------------------ fix 1

def test_mute_theatre_send_is_not_a_bug(bot_runtime):
    runtime, adapter = bot_runtime
    out = _mute_texts(runtime, adapter, "/mute @loud 2h", "m1")
    assert out == "", f"theatre answer must stay '': {out!r}"
    assert "produced no reply" not in out
    assert len(adapter.media) == 1
    assert "muted for 2h" in adapter.media[0][1]
    assert adapter.calls[0][0] == "restrict"


def test_kick_theatre_send_is_not_a_bug(bot_runtime):
    runtime, adapter = bot_runtime
    out = _mute_texts(runtime, adapter, "/kick @loud", "m2")
    assert out == "", f"theatre answer must stay '': {out!r}"
    assert len(adapter.media) == 1
    assert "out" in adapter.media[0][1]


def test_ban_theatre_send_is_not_a_bug(bot_runtime):
    runtime, adapter = bot_runtime
    out = _mute_texts(runtime, adapter, "/ban @loud", "m3")
    assert out == "", f"theatre answer must stay '': {out!r}"
    assert len(adapter.media) == 1
    assert "banned" in adapter.media[0][1]


def test_unmute_theatre_send_is_not_a_bug(bot_runtime):
    runtime, adapter = bot_runtime
    out = _mute_texts(runtime, adapter, "/unmute @loud", "m4")
    assert out == "", f"theatre answer must stay '': {out!r}"
    assert len(adapter.media) == 1
    assert "can speak again" in adapter.media[0][1]


def test_theatre_send_marks_the_delivery(bot_runtime):
    """The mechanism, not just the outcome: the delivery count
    rises across a theatre dispatch, so the #32 guard sees it."""
    from nomorals.agents.partner.runtime import _control_deliveries

    runtime, adapter = bot_runtime
    before = _control_deliveries.get()
    out = _mute_texts(runtime, adapter, "/mute @loud 2h", "m5")
    assert out == ""
    assert _control_deliveries.get() > before


def test_theatre_failure_returns_text_not_bug(bot_runtime):
    """Mark on success ONLY: a failed fire returns the verdict
    text + reason (non-empty, no bug text) and marks nothing."""
    from nomorals.agents.partner.runtime import _control_deliveries

    runtime, adapter = bot_runtime
    runtime._expression_service_obj = FakeExpressionService(sent=False)
    before = _control_deliveries.get()
    out = _mute_texts(runtime, adapter, "/mute @loud 2h", "m6")
    assert "muted for 2h" in out
    assert "no mute visual here" in out
    assert "produced no reply" not in out
    assert _control_deliveries.get() == before


# ------------------------------------------------------------ fix 2

def test_theatre_answer_leaves_answer_claim(bot_runtime):
    """A send-during-dispatch answer is provable in the shared
    ledger — the sweep's step (1) must find it. The claim is
    recorded for the THEATRE answer itself: on the old code the
    dispatch returns the bug text instead, and the only claim
    that ever lands is for the false bug message."""
    runtime, adapter = bot_runtime
    msg = _group_msg("telegram-bot", "/mute @loud 2h", "m7")
    out = runtime.handle_control("/mute @loud 2h", GROUP_KEY, message=msg)
    assert out == "", f"theatre answer must stay '': {out!r}"
    proven, _stamp = runtime._answer_proof(msg)
    assert proven, "theatre answer left no answer claim"


def test_sibling_stands_down_after_theatre_answer(
        split_homes, shared_claims_file):
    """End to end: the bot answers /mute with theatre; the
    userbot's echo deferred first; the sweep stands down — no
    second GIF, no bug text, deferral resolved."""
    home_a, home_b = split_homes
    ad_a = {"telegram-bot": GroupAdapter("telegram-bot",
                                         int(MAIN_BOT_ID))}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a = _make_runtime(home_a, ad_a)
    rt_b = _make_runtime(home_b, ad_b)
    try:
        rt_a._expression_service_obj = FakeExpressionService(sent=True)
        rt_b._expression_service_obj = FakeExpressionService(sent=True)
        rt_a._group_surface().handle_message(
            GROUP_KEY, "777", username="loud", text="hi")
        base = time.time()
        rt_b._process_inner(_group_echo("/mute @loud 2h", "g-mute-1"))
        assert _texts(ad_b["telegram"]) == []
        assert _pending(rt_b) == 1, "the group echo must defer"
        rt_a._process_inner(_group_msg(
            "telegram-bot", "/mute @loud 2h", "g-mute-1"))
        assert len(ad_a["telegram-bot"].media) == 1, (
            "the bot must answer once, with theatre")
        assert _texts(ad_a["telegram-bot"]) == []
        rt_b._sweep_peer_deferrals(now=base + 12.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer a theatre answer")
        assert ad_b["telegram"].media == [], (
            "no second GIF from the userbot")
        assert _pending(rt_b) == 0
    finally:
        _stop(rt_a)
        _stop(rt_b)


def test_userbot_only_group_still_answers(
        split_homes, shared_claims_file):
    """The single-surface case must keep working: no bot anywhere,
    the userbot's echo defers, the sweep fails over and answers
    exactly once — with the theatre, never the bug text."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt = _make_runtime(home_a, ad)
    try:
        rt._expression_service_obj = FakeExpressionService(sent=True)
        # Seed the member under the key the echo carries
        # ("telegram:…") — the ledger resolves on the exact key,
        # while the claim merges across the Telegram family, so
        # the designation below (under GROUP_KEY) still applies.
        rt._group_surface().handle_message(
            "telegram:-1001234567890", "777", username="loud",
            text="hi")
        _designate(rt)
        assert rt._claims_are_shared() is True
        base = time.time()
        rt._process_inner(_group_echo("/mute @loud 2h", "g-mute-2"))
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1
        rt._sweep_peer_deferrals(now=base + 25.0)
        assert len(ad["telegram"].media) == 1, (
            "the failover must answer once, with theatre")
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ------------------------------------------------------------ fix 3

def _mind_msg(platform, kind, chat_id):
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                   peer=chat_id)
    return ChatMessage(chat=chat, incoming=True, text="",
                       sender="death", message_id="mind-1",
                       meta={"is_owner": True})


def test_mind_reply_orchestrator_notes_suppressed_in_group(bot_runtime):
    runtime, adapter = bot_runtime
    msg = _mind_msg("telegram-bot", ChatKind.GROUP, GROUP_ID)
    runtime._send_mind_reply(msg, PLANNING_NOTE)
    runtime._send_mind_reply(msg, DONE_NOTE)
    assert _texts(adapter) == [], (
        "orchestrator status notes must never reach a group")


def test_mind_reply_orchestrator_notes_still_send_in_dm(bot_runtime):
    runtime, adapter = bot_runtime
    msg = _mind_msg("telegram-bot", ChatKind.DM, OWNER_ID)
    runtime._send_mind_reply(msg, PLANNING_NOTE)
    runtime._send_mind_reply(msg, DONE_NOTE)
    assert len(_texts(adapter)) == 2, (
        "suppression is group-only; DMs still get the notes")


def test_start_run_notify_suppressed_in_group(tmp_path):
    """The done note's own delivery point: CoreMind._start_run's
    worker notifies via _send_long — a group orchestrator note
    is journal-only there too (provider detail included)."""
    import threading

    from nomorals.agents.coremind import (
        CoreMind,
        is_orchestrator_status_text,
    )

    assert is_orchestrator_status_text(PLANNING_NOTE) is True
    assert is_orchestrator_status_text(DONE_NOTE) is True
    assert is_orchestrator_status_text("hello there") is False

    home = tempfile.TemporaryDirectory(prefix="nm-mnr-mind-",
                                       dir=_TEST_TMPDIR)
    try:
        adapter = GroupAdapter("telegram-bot", int(MAIN_BOT_ID))
        runtime = _make_runtime(home.name, {"telegram-bot": adapter})
        mind = CoreMind(runtime.context)
        mind.runtime = runtime
        # The room must be known to the group surface: the
        # _start_run notify checks the chat KEY against the
        # surface store (a rebuilt ref drops kind).
        runtime._group_surface().handle_message(
            GROUP_KEY, "777", username="loud", text="hi")
        job_done = threading.Event()

        def job():
            job_done.set()
            return DONE_NOTE

        mind._send_async(GROUP_KEY, job, "test-job",
                         PLANNING_NOTE, kind="multi")
        assert job_done.wait(timeout=10.0), "the job itself must run"
        # The notify runs on the worker right after the job
        # returns; give it a moment to (not) land, then assert.
        time.sleep(1.0)
        assert _texts(adapter) == [], (
            "the done note must not reach the group")
        assert adapter.media == []
    finally:
        _stop(runtime)
        home.cleanup()
