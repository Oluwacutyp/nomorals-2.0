"""Group rhythm autonomy — the rhythm + estate cycle on its own
clock, and group system output from the BOT identity (owner live
reports 2026-11-16 ~08:06 and ~08:11).

Defect A (owner screenshot): the richest-balance record, the
Market Mile district event and the weekly gift proposal are all
evaluated inside handle_message only — in a quiet room they pile
up and land at once on the next command. The fix is a scheduler
job (``__group_rhythm_tick__``, same shape as the roster-diff
tick) that runs the rhythm + estate cycle on its own clock
across claimed groups.

Defect B (owner report): the richest-balance post went out from
the OWNER's own account (userbot) — both runtimes race the
first-come claim in _claim_system_send, and the userbot won it.
The fix is defer-and-observe in the claim gate: an owner surface
waits a bounded window for the bot's claim and stands down when
it appears; the bot is the deterministic sender, the userbot the
fallback when the bot is absent or demoted.

Failing-first: before the fix ``nomorals.agents.group_rhythm``
does not exist at all, and _claim_system_send has no
defer-and-observe path.
"""
from __future__ import annotations

import calendar
import contextlib
import json
import os
import random
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.group_rhythm import (
    RHYTHM_JOB,
    RHYTHM_SPEC,
    RHYTHM_TOOL,
    ensure_group_rhythm_job,
    render_proposal_announcement,
    run_group_rhythm_tick_once,
)
from nomorals.agents.partner.failures import clear_failures
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from tests.frozen_clock import MIDDAY_TS

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888
GROUP_ID = "-100123456789"
KEY = "telegram-bot:-1001234567"


def _ts(y, mo, d, h=0, mi=0):
    return float(calendar.timegm((y, mo, d, h, mi, 0, 0, 0, 0)))


DAY = _ts(2023, 11, 14, 12, 0)
NIGHT = _ts(2023, 11, 14, 22, 30)
MORNING = _ts(2023, 11, 15, 7, 30)


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _rhythm(surface: GroupSurface, **kw) -> None:
    block = dict(surface.store.settings(KEY)["rhythm"])
    block.update(kw)
    surface.update_settings(KEY, rhythm=block)


def _materialize(surface: GroupSurface) -> None:
    """Give the room a store entry so the tick enumerates it.
    (The economy ledger and the estate state live elsewhere;
    only a settings write materialises the group key.)"""
    surface.update_settings(KEY, rhythm={})


class _FakeLedger:
    """Shared claim ledger with the real contract
    (claim + claimed_by), thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claims: dict[str, tuple[str, float]] = {}

    def claim(self, key: str, surface: str, *,
              now: float | None = None) -> bool:
        with self._lock:
            held = self._claims.get(key)
            if held is not None and held[0] != surface:
                return False
            self._claims[key] = (surface, time.time())
            return True

    def claimed_by(self, key: str, *,
                   now: float | None = None) -> str:
        with self._lock:
            held = self._claims.get(key)
            return held[0] if held else ""


class _FakeSend:
    """Captures tick sends: (platform, chat_key, text, buttons)."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def __call__(self, platform: str, chat: Any, text: str, *,
                 buttons: Any = None, parse_mode: str = "") -> bool:
        self.sent.append({
            "platform": platform,
            "chat_key": getattr(chat, "key", str(chat)),
            "text": str(text),
            "buttons": buttons,
            "parse_mode": parse_mode,
        })
        return True

    def texts(self) -> list[str]:
        return [s["text"] for s in self.sent]


class _FakeScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []

    def list_jobs(self) -> list[dict[str, Any]]:
        return list(self.jobs)

    def add(self, name: str, spec: str, kind: str,
            payload: dict[str, Any], **kw) -> dict[str, Any]:
        job = {"id": f"job-{len(self.jobs)}", "name": name,
               "spec": spec, "kind": kind, "payload": payload}
        self.jobs.append(job)
        return job

    def remove(self, job_id: str) -> None:
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


class _FakeBotAdapter:
    _bot_id = 888


class _FakeOwnerAdapter:
    _me_id = 4242


# ── defect A: the tick fires without any message ───────────────────


def test_tick_posts_due_milestone_without_any_message(tmp_path):
    """A richest-balance record set with NO message in the room is
    announced by the tick itself — no handle_message anywhere."""
    surface = _surface(tmp_path)
    _materialize(surface)
    send = _FakeSend()
    ledger = _FakeLedger()
    # Seed pass: the book baselines silently.
    out = run_group_rhythm_tick_once(
        surface=surface, now=DAY, send=send, claim=ledger,
        adapters=[("telegram-bot", _FakeBotAdapter())])
    assert out["ok"] and send.texts() == [] and out["groups"] == 1
    # A record balance after the seed is news.
    surface.economy.treasury_bonus(KEY, 100, "seed", now=DAY)
    surface.grant(KEY, "7", 100, by="41", now=DAY + 10)
    out = run_group_rhythm_tick_once(
        surface=surface, now=DAY + 3600, send=send, claim=ledger,
        adapters=[("telegram-bot", _FakeBotAdapter())])
    assert out["posted"] >= 1, out
    assert any("richest balance" in t for t in send.texts())


def test_tick_posts_due_gift_proposal(tmp_path):
    """The weekly gift proposal is created AND posted by the tick:
    the proposal exists in governance and the card went out with
    its Approve/Deny buttons."""
    surface = _surface(tmp_path)
    _rhythm(surface, review={"mode": "auto"})
    surface.economy.treasury_bonus(KEY, 5000, "seed", now=DAY)
    extras = surface._extras(KEY)
    extras["contributions"] = [
        {"state": "counted", "ts": DAY + i * 60, "user_id": "7"}
        for i in range(3)
    ]
    surface.store.save()
    send = _FakeSend()
    out = run_group_rhythm_tick_once(
        surface=surface, now=DAY + 3600, send=send,
        claim=_FakeLedger(), is_owner_surface=False)
    assert out["posted"] >= 1, out
    gift = [t for t in send.texts() if "Gift proposal" in t]
    assert gift, send.texts()
    assert "250 sparks" in gift[0]
    proposals = surface.open_proposals(KEY)
    assert [p for p in proposals.values()
            if p.get("by") == "rhythm"], proposals
    buttons = [s["buttons"] for s in send.sent
               if "Gift proposal" in s["text"]][0]
    assert buttons, "the proposal must carry its decision buttons"


def test_tick_posts_estate_event(tmp_path):
    """A due district event (Market Mile) is announced by the tick
    — the estate tick is time-gated, so the message path calling
    it too cannot double-fire it."""
    surface = _surface(tmp_path)
    _materialize(surface)
    state = surface.estate._load_state(KEY)
    state["properties"] = {"p1": {"id": "p1", "owner": "7",
                                  "tier": "shack", "district": "market_mile",
                                  "anchor": DAY}}
    state["event_day"] = "2000-01-01"
    surface.estate._save_state(KEY, state)
    send = _FakeSend()
    out = run_group_rhythm_tick_once(
        surface=surface, now=DAY, send=send,
        claim=_FakeLedger(),
        adapters=[("telegram-bot", _FakeBotAdapter())])
    assert out["posted"] >= 1, out
    district = ("is booming", "is in a slump", "tenant trouble",
                "repair wave")
    assert any(any(p in t for p in district) for t in send.texts()), \
        send.texts()
    # A second tick the same day fires nothing new.
    send2 = _FakeSend()
    out2 = run_group_rhythm_tick_once(
        surface=surface, now=DAY + 3600, send=send2,
        claim=_FakeLedger(),
        adapters=[("telegram-bot", _FakeBotAdapter())])
    assert not any(any(p in t for p in district)
                   for t in send2.texts()), send2.texts()
    assert out2["posted"] == 0, out2


def test_tick_render_matches_turn_render(monkeypatch):
    """The tick's proposal card is byte-identical to the turn's
    render of the same item — so the shared claim gate dedupes
    the two paths instead of double-posting."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-rh-shared-",
                                      dir=_TEST_TMPDIR)
    path = str(Path(tmp.name) / "claims.db")
    pair = _make_pair(monkeypatch, shared_path=path)
    (rt, _ad, _ctx), (_rt_b, _ad_b, _ctx_b) = pair[0], pair[1]
    try:
        surface = rt._group_surface()
        surface.economy.treasury_bonus(KEY, 5000, "seed", now=DAY)
        item = {"kind": "gift_proposal", "proposal": "prop-1",
                "amount": 250, "recipient": "7",
                "label": "Ada", "count": 76}
        message = SimpleNamespace(
            chat=ChatRef(platform="telegram-bot",
                         chat_id="-1001234567", kind=ChatKind.GROUP))
        turn_text, turn_buttons = rt._proposal_feed_render(
            surface, KEY, message, item)
        tick_text, tick_buttons, _mode = render_proposal_announcement(
            surface, KEY, item)
        assert tick_text == turn_text
        assert tick_buttons == turn_buttons
    finally:
        _teardown_pair((pair[0], pair[1]))
        for tmp_dir in pair[2]:
            tmp_dir.cleanup()
        tmp.cleanup()


def test_tick_holds_notices_in_silent_hours(tmp_path):
    """Silent hours hold the tick's notices into the morning
    digest instead of sending them — and the digest releases
    them when the quiet ends."""
    surface = _surface(tmp_path)
    _rhythm(surface, silent_enabled=True, silent_start="22:00",
            silent_end="07:00", tz_offset_min=0)
    send = _FakeSend()
    ledger = _FakeLedger()
    run_group_rhythm_tick_once(
        surface=surface, now=NIGHT, send=send, claim=ledger,
        adapters=[("telegram-bot", _FakeBotAdapter())])
    surface.economy.treasury_bonus(KEY, 100, "seed", now=NIGHT)
    surface.grant(KEY, "7", 100, by="41", now=NIGHT + 10)
    run_group_rhythm_tick_once(
        surface=surface, now=NIGHT + 3600, send=send, claim=ledger,
        adapters=[("telegram-bot", _FakeBotAdapter())])
    assert send.texts() == [], "silent hours must hold, not send"
    held = list(surface._extras(KEY).get("rhythm_held") or [])
    assert held, "the notice must be held for the digest"
    run_group_rhythm_tick_once(
        surface=surface, now=MORNING, send=send, claim=ledger,
        adapters=[("telegram-bot", _FakeBotAdapter())])
    assert any("While the room was quiet" in t
               for t in send.texts()), send.texts()


def test_tick_marker_arbitrates_two_runtimes(tmp_path):
    """Two runtimes ticking the same room: the first runs it, the
    second skips the group on the shared tick marker."""
    surface = _surface(tmp_path)
    _materialize(surface)
    ledger = _FakeLedger()
    seed = run_group_rhythm_tick_once(
        surface=surface, now=DAY, send=_FakeSend(), claim=ledger,
        adapters=[("telegram-bot", _FakeBotAdapter())])
    assert seed["groups"] == 1
    surface.economy.treasury_bonus(KEY, 100, "seed", now=DAY)
    surface.grant(KEY, "7", 100, by="41", now=DAY + 10)
    send_a = _FakeSend()
    out_a = run_group_rhythm_tick_once(
        surface=surface, now=DAY + 3600, send=send_a, claim=ledger,
        adapters=[("telegram-bot", _FakeBotAdapter())])
    assert out_a["posted"] >= 1, out_a
    send_b = _FakeSend()
    out_b = run_group_rhythm_tick_once(
        surface=surface, now=DAY + 3600, send=send_b, claim=ledger,
        adapters=[("telegram-personal", _FakeOwnerAdapter())])
    assert send_b.texts() == [], "the second runtime must not re-post"
    assert out_b.get("skipped", 0) >= 1, out_b


def test_tick_owner_defers_to_bot_marker(tmp_path):
    """The owner runtime's tick stands down when the bot's tick
    marker is already live for the group."""
    surface = _surface(tmp_path)
    _materialize(surface)
    ledger = _FakeLedger()
    bucket = int((DAY + 3600) // 300)
    marker = f"system:rhythm-tick:{KEY}:{bucket}"
    assert ledger.claim(marker, "telegram-bot:888") is True
    send = _FakeSend()
    out = run_group_rhythm_tick_once(
        surface=surface, now=DAY + 3600, send=send, claim=ledger,
        adapters=[("telegram-personal", _FakeOwnerAdapter())],
        bot_wait_s=0.05)
    assert send.texts() == []
    assert out.get("skipped", 0) >= 1, out


def test_ensure_job_idempotent():
    sched = _FakeScheduler()
    first = ensure_group_rhythm_job(sched)
    second = ensure_group_rhythm_job(sched)
    assert len(sched.jobs) == 1
    assert first["name"] == RHYTHM_JOB == second["name"]
    assert first["spec"] == RHYTHM_SPEC == "every 5m"
    assert first["payload"]["tool"] == RHYTHM_TOOL
    assert first["payload"]["tool"] == "__group_rhythm_tick__"


# ── defect B: the bot is the deterministic sender ──────────────────


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
            json.dumps({"ts": MIDDAY_TS, "delivered_text": True}),
            encoding="utf-8")
    context.db.execute(
        "INSERT OR REPLACE INTO briefings (id, date, sections_json,"
        " generated_at) VALUES (?, ?, '[]', ?)",
        ("rhythm-wiring", "2000-01-01", MIDDAY_TS))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    runtime._owner_sender_ids = (OWNER_ID,)
    return runtime


def _make_pair(monkeypatch, shared_path=None):
    clear_failures()
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-rh-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    if shared_path is None:
        monkeypatch.delenv("NM_SHARED_CLAIMS", raising=False)
    else:
        monkeypatch.setenv("NM_SHARED_CLAIMS", shared_path)
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-rha-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-rhb-",
                                        dir=_TEST_TMPDIR)
    ctx_a = _context(tmp_a.name)
    ctx_b = _context(tmp_b.name)
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a = _runtime(ctx_a, ad_a)
    rt_b = _runtime(ctx_b, ad_b)
    return (rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b), (tmp_a, tmp_b,
                                                      home_tmp)


def _teardown_pair(pair):
    (rt_a, _ad_a, ctx_a), (rt_b, _ad_b, ctx_b) = pair
    for runtime, context in ((rt_a, ctx_a), (rt_b, ctx_b)):
        with contextlib.suppress(Exception):
            runtime.stop()
        context.close()


@pytest.fixture()
def split_homes(monkeypatch):
    """Two runtimes, SEPARATE homes, one shared claims file."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-rh-shared-",
                                      dir=_TEST_TMPDIR)
    path = str(Path(tmp.name) / "claims.db")
    pair = _make_pair(monkeypatch, shared_path=path)
    yield pair[0], pair[1]
    _teardown_pair((pair[0], pair[1]))
    for tmp_dir in pair[2]:
        tmp_dir.cleanup()
    tmp.cleanup()


def test_owner_stands_down_when_bot_claimed(split_homes):
    """The bot claims the notice first; the owner surface's claim
    then stands down — and it does so at once, not after the
    whole wait window."""
    (rt_bot, _ad_a, _ctx_a), (rt_owner, _ad_b, _ctx_b) = split_homes
    chat = ChatRef(platform="telegram-personal", chat_id=GROUP_ID,
                   kind=ChatKind.GROUP)
    assert rt_bot._claim_system_send(chat, "notice", "hello room") \
        is True
    started = time.monotonic()
    assert rt_owner._claim_system_send(
        chat, "notice", "hello room", bot_wait_s=2.0) is False
    assert time.monotonic() - started < 1.5, \
        "a live bot claim must short-circuit the wait"


def test_owner_stands_down_when_bot_claims_mid_wait(split_homes):
    """Defer-and-observe: the bot's claim lands DURING the owner
    surface's wait — the owner still stands down."""
    (rt_bot, _ad_a, _ctx_a), (rt_owner, _ad_b, _ctx_b) = split_homes
    chat = ChatRef(platform="telegram-personal", chat_id=GROUP_ID,
                   kind=ChatKind.GROUP)
    claimed: list[bool] = []

    def _late_claim():
        time.sleep(0.4)
        claimed.append(rt_bot._claim_system_send(
            chat, "notice", "hello room"))

    thread = threading.Thread(target=_late_claim, daemon=True)
    thread.start()
    assert rt_owner._claim_system_send(
        chat, "notice", "hello room", bot_wait_s=2.0) is False
    thread.join(timeout=3)
    assert claimed == [True]


def test_owner_sends_when_bot_absent(split_homes):
    """Fallback preserved: no bot claim appears in the window, so
    the owner surface claims and sends."""
    (_rt_bot, _ad_a, _ctx_a), (rt_owner, _ad_b, _ctx_b) = split_homes
    chat = ChatRef(platform="telegram-personal", chat_id=GROUP_ID,
                   kind=ChatKind.GROUP)
    assert rt_owner._claim_system_send(
        chat, "notice", "hello room", bot_wait_s=0.1) is True


def test_owner_sends_immediately_when_bot_demoted(split_homes, tmp_path):
    """A VERIFIED demotion (bot_admin False) means the bot is not
    the sender — the owner surface does not wait at all."""
    (_rt_bot, _ad_a, _ctx_a), (rt_owner, _ad_b, _ctx_b) = split_homes
    surface = _surface(tmp_path)
    surface.update_settings(KEY, system_claim={"bot_admin": False})
    chat = ChatRef(platform="telegram-personal", chat_id=GROUP_ID,
                   kind=ChatKind.GROUP)
    started = time.monotonic()
    assert rt_owner._claim_system_send(
        chat, "notice", "hello room", surface=surface, group_key=KEY,
        bot_wait_s=5.0) is True
    assert time.monotonic() - started < 1.0, \
        "a demoted bot must not cost the owner a wait"


def test_exactly_once_two_runtimes_same_notice(split_homes):
    """The gate still holds: one notice, two runtimes, one send.
    A different text (or kind) is a different send."""
    (rt_bot, _ad_a, _ctx_a), (rt_owner, _ad_b, _ctx_b) = split_homes
    chat = ChatRef(platform="telegram-personal", chat_id=GROUP_ID,
                   kind=ChatKind.GROUP)
    assert rt_bot._claim_system_send(chat, "notice", "hello room") \
        is True
    assert rt_owner._claim_system_send(
        chat, "notice", "hello room", bot_wait_s=0.1) is False
    assert rt_owner._claim_system_send(
        chat, "notice", "other text", bot_wait_s=0.1) is True
    assert rt_owner._claim_system_send(
        chat, "ask", "hello room", bot_wait_s=0.1) is True
