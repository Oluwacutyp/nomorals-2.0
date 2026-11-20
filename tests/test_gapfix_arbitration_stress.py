"""Lane dupfix7 (arbitration under stress) — the userbot must never
answer a group command while the public bot is in the group.

Owner corrections, 2026-11-20:

1. the screenshot duplication was his own double-send after a slow
   first /rminactive — no double-send bug;
2. the defect is the userbot replying to a group command while the
   public bot is in the group too — the video-defects arbitration
   fix breaks under stress;
3. it is NOT only the slow case — the userbot answers while the bot
   is present, PERIOD, even when the bot is not slow.

The seams (see ~/workspace/dupfix7-verify/SURVEY.md for the full
trace with file:line evidence):

S1. `_command_arbitration` rule (3) fell through to a
    first-claim-wins race whenever the bot's standing was unproven
    — which, on the userbot, is always, because the bot-claim
    probe could never run there (no `_bot_id`, no `admin_member`).
    The userbot's MTProto copy routinely arrives before the bot's
    Bot API poll, claims first, and answers.
S2. `_sweep_peer_deferrals` step (4) failed over at claim expiry
    (600 s) onto a bot that was still working — the /rminactive
    scan is unbounded, so the bot can still be scanning past its
    claim. "Claim expired + no answer" is not proof the bot is down.
S3. Unkeyable commands fell through to "handle" (rule (4), and
    the two `_process_inner` deferral-refused fall-throughs).

The repair law: while the public bot is present in the group the
userbot does not answer group commands — slow or prompt. Fallback
only on positive proof the bot is down/absent (a probe stamp),
never on "cannot prove it answered".

Rig copied from tests/test_gapfix_dupfix6.py: two runtimes,
separate homes, one shared claims file when the test needs it.
Presence stamps are written straight into the group surface's
system_claim record — the same record the probe writes — so the
pins read the same fact on the base (where the field is ignored)
and on the fix.
"""

import contextlib
import json
import os
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
                or os.path.expanduser("~/workspace/dupfix7-verify"))

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


class ProbeCountingAdapter(RecordingAdapter):
    """A Bot API-shaped adapter for the scan-bound pin: the
    member probe counts its calls and answers at once."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.calls = 0

    def admin_member_profile(self, chat: ChatRef,
                             user_id: str) -> dict:
        self.calls += 1
        return {"ok": True, "status": "member",
                "user": {"id": str(user_id), "first_name": "M",
                         "last_name": "", "username": "",
                         "is_bot": False}}


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
        ("dupfix7", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    import random

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


def _group_echo(text, message_id):
    """The owner's own message in a group, as the userbot sees it."""
    return _msg("telegram", chat_id=GROUP_ID, text=text, incoming=False,
                message_id=message_id, own=True, kind=ChatKind.GROUP)


def _group_msg(platform, text, message_id, *, sender_id=OWNER_ID):
    return _msg(platform, chat_id=GROUP_ID, text=text, incoming=True,
                message_id=message_id, sender_id=sender_id,
                kind=ChatKind.GROUP)


def _texts(adapter):
    return [t for t, _b in adapter.sent]


def _pending(runtime):
    return runtime._peer_deferral().pending_count()


def _stamp_presence(runtime, group_key, present, admin=None):
    """Write the probe's standing straight into the group
    surface's system_claim record — the same record the probe
    writes. ``present`` is the membership fact; ``admin`` the
    admin half. Works on the base too (where the field is
    simply never read), so the pins fail for the right reason
    there instead of erroring on a missing method."""
    surface = runtime._group_surface()
    claim = dict(surface.store.settings(group_key).get(
        "system_claim") or {})
    claim["bot_present"] = bool(present)
    if admin is not None:
        claim["bot_admin"] = bool(admin)
    surface.update_settings(group_key, system_claim=claim)
    surface.store.save()


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
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-d7-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-d7-a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-d7-b-",
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


# ------------------------------------------------------- (S1) general

def test_group_member_command_unproven_standing_defers_not_races(
        split_homes, shared_claims_file):
    """(S1) The general case, prompt bot: an incoming group /ping
    on the userbot, standing unproven everywhere (no probe stamp
    — the live userbot shape), shared store open. The userbot
    must NOT win the claim race: it defers, the bot answers
    once, and the userbot's sweep stands down on the proven
    answer. On the base the userbot claims first and answers at
    once."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        base = time.time()
        rt_b._process_inner(_group_msg(
            "telegram", "/ping", "s1-1", sender_id=MEMBER_ID))
        assert _texts(ad_b["telegram"]) == [], (
            "the userbot must not race the bot for a group "
            "command — it must defer")
        assert _pending(rt_b) == 1, "the userbot must defer"
        rt_a._process_inner(_group_msg(
            "telegram-personal", "/ping", "s1-1"))
        assert len(_texts(ad_a["telegram-personal"])) == 1, (
            "the bot must answer once")
        rt_b._sweep_peer_deferrals(now=base + 12.0)
        assert _texts(ad_b["telegram"]) == [], (
            "the sweep must not double-answer a proven answer")
        assert _pending(rt_b) == 0
        assert len(_texts(ad_a["telegram-personal"])) == 1
    finally:
        _stop(rt_a)
        _stop(rt_b)


# ---------------------------------------------------------- (S2) slow

def test_group_slow_bot_presence_holds_deferral_past_claim_expiry(
        split_homes, shared_claims_file):
    """(S2) The slow case: the owner's group echo of /ping, the
    bot stamped PRESENT in the room (a completed probe), the bot
    claimed the command 595 s ago (a slow scan) and never
    answered. At the short window the sweep re-arms on the live
    claim; past claim expiry it must STILL send nothing — the
    bot is in the room and may still be working. On the base the
    sweep fails over at expiry and answers."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        echo = _group_echo("/ping", "s2-1")
        _stamp_presence(rt_b, echo.chat.key, True, admin=True)
        key = command_claim_key(echo, own_account_id=OWNER_ID)
        assert rt_b._command_claims().claim(
            key, "telegram-personal:888", now=time.time() - 595.0,
            build_stamp=claim_build_stamp())
        base = time.time()
        rt_b._process_inner(echo)
        assert _texts(ad_b["telegram"]) == []
        assert _pending(rt_b) == 1, "the echo must defer"
        rt_b._sweep_peer_deferrals(now=base + 12.0)
        assert _texts(ad_b["telegram"]) == []
        assert _pending(rt_b) == 1, (
            "the live claim must re-arm, never answer")
        # The claim expires ~5 s after the test started; wait it out.
        time.sleep(6.5)
        rt_b._sweep_peer_deferrals(now=base + 12.0)
        assert _texts(ad_b["telegram"]) == [], (
            "a present bot must never be failed over onto — "
            "claim expiry is not proof the bot is down")
        assert _pending(rt_b) == 1, (
            "the deferral must hold while the bot is in the room")
    finally:
        _stop(rt_a)
        _stop(rt_b)


# ------------------------------------------------------ (fallback)

def test_group_absent_bot_fails_over_even_with_store_down(
        split_homes, broken_shared_claims):
    """(fallback) The bot is PROVABLY absent — the probe stamped
    it not-a-member — and the shared store is DOWN, so no ledger
    evidence can prove anything. The userbot must still answer
    exactly once: the probe stamp is positive proof from a
    channel the store outage cannot touch. On the base the blind
    rule waits to the bound and ends in an honest failure line —
    no pong."""
    home_a, _home_b = split_homes
    ad = {"telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        assert rt._claims_are_shared() is False
        echo = _group_echo("/ping", "s3-1")
        _stamp_presence(rt, echo.chat.key, False)
        base = time.time()
        rt._process_inner(echo)
        assert _texts(ad["telegram"]) == []
        assert _pending(rt) == 1, "the echo must defer"
        rt._sweep_peer_deferrals(now=base + 12.0)
        texts = _texts(ad["telegram"])
        assert len(texts) == 1, (
            f"the failover must answer once on proven absence: "
            f"{texts}")
        assert "pong" in texts[0]
        assert _pending(rt) == 0
    finally:
        _stop(rt)


# ------------------------------------------------- (S3) unkeyable

def test_group_unkeyable_operator_command_bot_present_stands_down(
        split_homes, shared_claims_file):
    """(S3) An incoming operator group /ping with NO message id
    (unkeyable), the bot stamped PRESENT. The deferral cannot be
    registered, so the old code rewrote "defer" to "handle" and
    answered at once. The bot's own copy carries the id and will
    answer it — the userbot must stand down and send nothing,
    arming no failover."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        msg = _group_msg("telegram", "/ping", "")
        _stamp_presence(rt_b, msg.chat.key, True, admin=True)
        rt_b._process_inner(msg)
        assert _texts(ad_b["telegram"]) == [], (
            "an unkeyable group command must not be handled "
            "while the bot is verified in the room")
        assert _pending(rt_b) == 0, (
            "no failover may be armed for a command the bot owns")
    finally:
        _stop(rt_a)
        _stop(rt_b)


def test_group_unkeyable_member_help_bot_present_stands_down(
        split_homes, shared_claims_file):
    """(S3) An incoming member group /help with NO message id,
    the bot stamped PRESENT. Same fall-through, group branch:
    the old code answered with the catalog at once. The userbot
    must send nothing."""
    home_a, home_b = split_homes
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a, _ctx_a = _make_runtime(home_a, ad_a)
    rt_b, _ctx_b = _make_runtime(home_b, ad_b)
    try:
        msg = _group_msg("telegram", "/help", "",
                         sender_id=MEMBER_ID)
        _stamp_presence(rt_b, msg.chat.key, True, admin=True)
        rt_b._process_inner(msg)
        assert _texts(ad_b["telegram"]) == [], (
            "an unkeyable group /help must not produce a catalog "
            "while the bot is verified in the room")
        assert _pending(rt_b) == 0
    finally:
        _stop(rt_a)
        _stop(rt_b)


# -------------------------------------------------- (scan bound)

def test_cleanup_roster_probe_loop_is_bounded(
        split_homes, shared_claims_file):
    """(stressor) The Bot API path probes every ledger member
    once, unbounded — at up to 40 s a probe, a big-group scan
    outlives the 600 s claim window. The loop must stop at the
    cap: the roster reports partial, the probe count stays
    bounded, and the unprobed members keep their ledger-based
    rows. On the base the loop probes every member."""
    home_a, _home_b = split_homes
    stub = ProbeCountingAdapter("telegram-bot")
    ad = {"telegram-bot": stub,
          "telegram": BareUserbotAdapter("telegram")}
    rt, _ctx = _make_runtime(home_a, ad)
    try:
        surface = rt._group_surface()
        key = "telegram-bot:" + GROUP_ID
        entry = surface.store.group(key)
        for i in range(400):
            entry["members"][f"u{i}"] = {
                "user_id": f"u{i}", "name": f"Member {i}",
                "username": "", "is_bot": False}
        surface.store.save()
        msg = _group_msg("telegram-bot", "/cleanup", "s6-1")
        roster = rt._cleanup_roster(surface, key, msg)
        assert roster["probes_truncated"] is True, (
            "a 400-member ledger must exceed the probe budget")
        assert stub.calls <= rt._CLEANUP_PROBE_MAX, (
            f"probe calls must be capped: {stub.calls}")
        assert len(roster["rows"]) == 400, (
            "unprobed members keep their ledger-based rows")
        assert roster["probed"] == stub.calls
    finally:
        _stop(rt)
