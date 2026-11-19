"""Lane 62 — owner-command silence holes (cmdsilence).

Every test here is failing-first on the pristine base f073d80c: an
owner command that vanishes today with NO reply, and must end in
exactly one of: an answer, a bounded failover answer, or an honest
failure reply. Exactly-once still holds — a silence fix that
double-answers is worse than the silence.

Rig copied from tests/test_dupfix2_peer_answer.py (the split-homes
shape): two runtimes, separate homes, one shared claims file when
the test needs it. The owner types a command in one chat; the bot
and the userbot each see their own copy of it.
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
from nomorals.agents.partner.command_claims import (
    claim_build_stamp,
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

_TEST_TMPDIR = (os.environ.get("NM_TEST_TMPDIR")
                or os.environ.get("TMPDIR")
                or os.path.expanduser("~/workspace/cmdsilence-verify"))

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888
MAIN_BOT_ID = "8133295902"
GROUP_ID = "-1001234567890"


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
        ("cmdsilence", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    return runtime


def _msg(platform, *, chat_id, text, incoming, message_id,
         sender_id=OWNER_ID, own=False, peer_is_bot=False,
         service=None, kind=ChatKind.DM):
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


def _group_msg(text, message_id, *, sender_id=OWNER_ID):
    return _msg("telegram", chat_id=GROUP_ID, text=text, incoming=True,
                message_id=message_id, sender_id=sender_id,
                kind=ChatKind.GROUP)


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
def split_homes(monkeypatch):
    clear_failures()
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-cs-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-cs-a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-cs-b-",
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


def _stub_update(monkeypatch, runtime):
    """The /update dispatch, with the real controller replaced by a
    canned success: the update itself is out of scope here — the
    silence lives in arbitration/deferral, not in the handler."""

    class _StubController:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, kind):
            return {"message": "update ran (stub)",
                    "ok": True, "restart_requested": False,
                    "update_report": None}

    monkeypatch.setattr(
        "nomorals.social.chat.update_control.ChatUpdateController",
        _StubController)
    runtime._open_restart_card = lambda *a, **k: None


# ---------------------------------------------------------------- S1

def test_dm_command_addressed_to_other_bot_answers_here(split_homes):
    """S1 (rule 1): /ping@Nomorals_bot typed in the PERSONAL bot's
    DM. The main bot can never see this DM — standing the command
    down silences it forever. The runtime in front of the owner
    must answer it."""
    home_a, _home_b = split_homes
    ad = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        rt._process_inner(_bot_copy("/ping@Nomorals_bot", "s1-1"))
        texts = _texts(ad["telegram-personal"])
        assert len(texts) == 1, f"no reply for the addressed DM command: {texts}"
        assert "pong" in texts[0]
    finally:
        _stop(rt)


def test_group_command_addressed_to_absent_bot_fails_over(
        split_homes, monkeypatch):
    """S1 (group) + /update end-to-end: /update@Nomorals_bot in a
    group the main bot is not in. The userbot sees only the echo;
    the addressed bot never answers. The echo must defer, and the
    sweep must answer it — exactly once."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    _stub_update(monkeypatch, rt)
    try:
        base = time.time()
        rt._process_inner(_group_echo("/update@Nomorals_bot", "s1g-1"))
        # Route-1 holds: the echo never executes at once.
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1, "the addressed group echo must defer"
        # The group window is 8s from the record's creation, and
        # _process_inner takes ~1s before it creates the record
        # — sweeping at base+9.0 is only a hair past the deadline
        # and slips under load. +12.0 is safely past it.
        rt._sweep_peer_deferrals(now=base + 12.0)
        texts = _texts(ad["telegram"])
        assert len(texts) == 1, f"the sweep must answer the update: {texts}"
        assert "update ran (stub)" in texts[0]
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ---------------------------------------------------------------- S2

def test_peer_bot_dm_claim_without_answer_fails_over(
        split_homes, shared_claims_file):
    """S2 + S5: the owner types /ping in the personal bot's DM. The
    sibling bot's claim is live but it NEVER answers (seeded live,
    backdated so it expires 5s into the test). Today the echo is
    stood down on the claim and the command dies. After the fix the
    echo defers, the sweep re-arms at claim expiry, and the next
    sweep answers — exactly once."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram"),
          "telegram-personal": BotAdapter(
              "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        base = time.time()
        echo = _echo_copy("/ping", "s2-1")
        key = command_claim_key(echo, own_account_id=OWNER_ID)
        # The sibling bot claimed the command 595s ago and never
        # answered; its claim expires 5s from now. The seed is
        # STAMPED (dupfix6): it simulates a CURRENT primary, and
        # current builds stamp every claim — an unstamped seed
        # would simulate a pre-dupfix6 primary, whose missing
        # answer proves nothing and must end in an honest
        # failure, not a failover (see test_gapfix_dupfix6's
        # stale-build test).
        assert rt._command_claims().claim(
            key, "telegram-personal:888", now=base - 595.0,
            build_stamp=claim_build_stamp())
        rt._process_inner(echo)
        assert _texts(ad["telegram"]) == []
        assert _texts(ad["telegram-personal"]) == []
        assert _pending(rt) == 1, "the echo must defer, not stand down"
        # First sweep (past the 20s DM window): the holder's claim
        # is still live — stand down, but RE-ARM, never drop.
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


# ---------------------------------------------------------------- S3

def test_group_bot_admin_standing_without_answer_fails_over(
        split_homes, monkeypatch):
    """S3: the group surface says the bot is admin, but the bot
    never answers the owner's /ping in that group. Standing is not
    an answer: the echo must defer and the sweep must answer."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        surface = rt._group_surface()
        surface.note_bot_admin(f"telegram:{GROUP_ID}", True)
        assert surface.claim_state(
            f"telegram:{GROUP_ID}")["bot_admin"] is True
        base = time.time()
        rt._process_inner(_group_echo("/ping", "s3-1"))
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1, "the echo must defer, not stand down"
        # Group window is 8s from the record's creation, and
        # _process_inner takes ~1s before it creates the record —
        # +9.0 is only a hair past the deadline and slips under
        # load. +12.0 is safely past it.
        rt._sweep_peer_deferrals(now=base + 12.0)
        texts = _texts(ad["telegram"])
        assert len(texts) == 1, f"the sweep must answer once: {texts}"
        assert "pong" in texts[0]
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ---------------------------------------------------------------- S4

def test_claim_race_fails_over_at_expiry(split_homes,
                                        shared_claims_file):
    """S4: an incoming /ping in the personal bot's DM while ANOTHER
    surface holds a live claim on the same message id and never
    answers. The race must defer and fail over at claim expiry —
    exactly once."""
    home_a, _home_b = split_homes
    ad = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        base = time.time()
        msg = _bot_copy("/ping", "s4-1")
        key = command_claim_key(msg, own_account_id=str(PERSONAL_BOT_ID))
        # STAMPED (dupfix6): it simulates a CURRENT primary, and
        # current builds stamp every claim — an unstamped seed
        # would simulate a pre-dupfix6 primary, whose missing
        # answer proves nothing (see test_gapfix_dupfix6's
        # stale-build test).
        assert rt._command_claims().claim(
            key, f"telegram-bot:{MAIN_BOT_ID}", now=base - 595.0,
            build_stamp=claim_build_stamp())
        rt._process_inner(msg)
        assert _texts(ad["telegram-personal"]) == []
        assert _pending(rt) == 1, "the raced command must defer"
        rt._sweep_peer_deferrals(now=base + 25.0)
        assert _texts(ad["telegram-personal"]) == []
        assert _pending(rt) == 1, "the deferral must survive to claim expiry"
        time.sleep(6.5)
        rt._sweep_peer_deferrals(now=base + 25.0)
        texts = _texts(ad["telegram-personal"])
        assert len(texts) == 1, f"the failover must answer once: {texts}"
        assert "pong" in texts[0]
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ---------------------------------------------------------------- S6

def test_dead_sweeper_thread_cycle_sweep_fails_over(split_homes):
    """S6: the peer-deferral sweeper thread is GONE (patched to
    never start). A later message through _process_inner must
    drive the cycle sweep and fail the deferred command over —
    the failover clock cannot be one thread alone."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram"),
          "telegram-personal": BotAdapter(
              "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        # No sweeper thread, ever.
        rt._start_peer_deferral_sweeper = lambda: None
        rt._peer_answer_window_s = 0.3
        rt._process_inner(_echo_copy("/ping", "s6-1"))
        assert _pending(rt) == 1
        assert getattr(rt, "_peer_deferral_thread", None) is None
        assert _texts(ad["telegram"]) == []
        time.sleep(0.5)
        # A second, unrelated owner command drives the cycle sweep
        # BEFORE it is handled: failover pong + this ping's pong.
        rt._process_inner(_msg("telegram", chat_id=OWNER_ID,
                               text="/ping", incoming=True,
                               message_id="s6-2"))
        texts = _texts(ad["telegram"])
        assert len(texts) == 2, (
            "the cycle sweep must fail the deferred command over "
            f"before the new one is handled: {texts}")
        assert all("pong" in t for t in texts)
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ---------------------------------------------------------------- S7

def test_stale_owner_command_gets_honest_line(split_homes):
    """S7: a stale owner slash command must not execute — and must
    not vanish silently either. One honest line, no pong."""
    home_a, _home_b = split_homes
    ad = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        msg = _bot_copy("/ping", "s7-1")
        msg.meta["stale_replay"] = True
        rt._process_inner(msg)
        texts = _texts(ad["telegram-personal"])
        assert len(texts) == 1, f"a stale owner command must be named: {texts}"
        assert "pong" not in texts[0]
        assert "send it again" in texts[0]
    finally:
        _stop(rt)


# ---------------------------------------------------------------- S8

def test_bare_url_failure_gets_honest_reply(split_homes, monkeypatch):
    """S8: a bare media URL whose download raises must not vanish
    silently. The owner gets an honest failure naming the link."""
    home_a, _home_b = split_homes
    ad = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        def _boom(*args, **kwargs):
            raise RuntimeError("fetch boom")

        monkeypatch.setattr(rt, "_control_download", _boom)
        rt._process_inner(_bot_copy(
            "https://example.com/song.mp3", "s8-1"))
        texts = _texts(ad["telegram-personal"])
        assert len(texts) == 1, f"the failed download must be named: {texts}"
        assert "song.mp3" in texts[0] or "download" in texts[0].lower()
    finally:
        _stop(rt)


# ---------------------------------------------------------------- S9

def test_game_empty_reply_gets_honest_fallback(split_homes, monkeypatch):
    """S9: a game handler returning \"\" (real case: engine.quit
    returning no lines) must not vanish silently."""
    home_a, _home_b = split_homes
    ad = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        monkeypatch.setattr(rt, "_control_game", lambda *a, **k: "")
        rt._process_inner(_bot_copy("/game status", "s9-1"))
        texts = _texts(ad["telegram-personal"])
        assert len(texts) == 1, f"the empty game reply must be named: {texts}"
        assert "game" in texts[0].lower()
    finally:
        _stop(rt)


# ---------------------------------------------------------------- S10

def test_answered_claim_prevents_double_answer(
        split_homes, shared_claims_file):
    """S10 (exactly-once): the userbot's echo defers FIRST (the
    bot has not answered yet); the personal bot then ANSWERS /ping
    and records the answer claim, but the answer is never observed
    (no peer post). The sweep must stand the failover down on the
    PROVEN answer — never send a second pong. The bot's claim row
    is backdated past expiry (a slow primary whose claim lapsed
    before it answered), so the live-claim stand-down cannot save
    the test: only the answer claim can."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        echo = _echo_copy("/ping", "s10-1")
        # The userbot sees only the echo, before the bot answers.
        rt_b._process_inner(echo)
        assert _texts(ad_b["telegram"]) == []
        assert _pending(rt_b) == 1, "the echo must defer"
        # The bot answers its own incoming copy.
        rt_a._process_inner(_bot_copy("/ping", "s10-1"))
        assert len(_texts(ad_a["telegram-personal"])) == 1
        # Backdate the bot's claim row past expiry: the claim
        # lapsed before the slow primary answered; the ANSWER is
        # fresh. (A same-surface re-claim cannot move the stamp —
        # the ledger only updates it on a fresh insert or a steal —
        # so the row is aged directly.)
        key = command_claim_key(echo, own_account_id=OWNER_ID)
        import sqlite3

        conn = sqlite3.connect(shared_claims_file)
        conn.execute(
            "UPDATE command_claims SET claimed_at = ?"
            " WHERE claim_key = ?", (base - 601.0, key))
        conn.commit()
        conn.close()
        # The sweep must not double-answer a proven answer.
        rt_b._sweep_peer_deferrals(now=base + 25.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer a proven answer")
        assert _pending(rt_b) == 0
        assert len(_texts(ad_a["telegram-personal"])) == 1
    finally:
        _stop(rt_a)
        _stop(rt_b)
