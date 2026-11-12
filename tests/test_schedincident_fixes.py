"""Scheduler-incident pins (2026-11-11 owner screen recording).

Two scheduled jobs hung to the 300s wall AND /help + a conversational
reply double-sent, live on the owner's EC2 box.  The fixes pinned
here:

* Scheduler watchdog: a guarded run samples the worker thread's
  stack while it waits, so a timeout NAMES the stuck call; the
  abandoned worker is RECORDED per job (count + prune), and the
  failure alert's policy line names the ONE failure mode that
  actually happened.
* Improvement cycle deadline: the benchmark aggregate + plan run
  under a cycle budget well inside the wall; the autonomous apply
  (1500s test gate) runs DETACHED, never inside the tick.
* Alert honesty: one stamp per alert (the policy line rides BEFORE
  a trailing stamp, never after it), and a caller-named failure
  reason beats the generic three-way fallback.
* Double-send: the Bot API consumer lock + update offset are
  home-anchored and durable (a restart from another CWD, or a
  failed stale-skip, can no longer double-deliver); the userbot
  deferral chain holds even when the echo carries no peer stamp
  (companion ids cover it); an owner echo never answers
  conversationally.

Every test redirects stores with the app's own NM_HOME override
(never the real home, never the repo tree).  Pins rule (#169): no
bare-except swallowing anywhere in this file — where a pin must
swallow, it binds ``as exc`` + ``assert isinstance(exc, Exception)``.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.agents import benchmark as bench_mod
from nomorals.agents import improvement as improv_mod
from nomorals.agents.alert_policy import AlertHints, AlertPolicy
from nomorals.agents.benchmark import DimensionScore, run_benchmark
from nomorals.agents.benchmark import DimensionScore as _DS
from nomorals.agents.context import build_context
from nomorals.agents.improvement import ImprovementLoop
from nomorals.agents.notifier import _insert_before_stamp
from nomorals.agents.partner.failures import clear_failures
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.agents.scheduler import Scheduler
from nomorals.core.config import ImprovementSettings
from nomorals.core.config import load_settings as _load_settings
from nomorals.social.chat.base import ChatAdapter, ChatKind, ChatMessage, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.single_consumer import SingleConsumerLock
from nomorals.social.chat.telegram import TelegramBotAdapter
from nomorals.social.chat.two_bots import TwoBotConfig, build_both
from nomorals.social.chat.typography import DEFAULT_STAMP_TEXT
from nomorals.storage.db import Database

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888

# ── scheduler watchdog ──────────────────────────────────────────────────



class _BoomDB:
    """DB whose query raises; execute/transaction are no-ops."""

    def query(self, *args, **kwargs):
        raise RuntimeError("db down")

    def query_one(self, *args, **kwargs):
        return None

    def execute(self, *args, **kwargs):
        return None

    def transaction(self):
        return contextlib.nullcontext()


def _sched(db=None, **kw):
    s = Scheduler.__new__(Scheduler)
    s.db = db
    s.context = SimpleNamespace()
    s.notifier = SimpleNamespace()
    s._lock = threading.Lock()
    s._active = set()
    s._deferrals = {}
    s._abandoned = {}
    s._last_guard_timeout = None
    s._tick_errors = 0
    s._last_tick_at = None
    s._started_at = None
    s._thread = None
    s.tick_seconds = 20.0
    s.redeliver_interval = 120.0
    s._last_redeliver_at = 1e18
    s._running_jobs = 0
    s.wall_seconds = 300.0
    s._rng = random.Random(0)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _hang_forever(*args, **kwargs):
    time.sleep(30.0)
    return "never", None, None


def _hang_briefly(*args, **kwargs):
    time.sleep(2.0)
    return "late", None, None


def _tool_row(job_id: str = "j1", name: str = "job1"):
    return {"id": job_id, "name": name, "payload_kind": "tool",
            "payload": json.dumps({"tool": "x", "args": {}})}


def test_hanging_tool_call_fails_fast_naming_the_stuck_call():
    """A tool call that hangs past its inner timeout fails fast
    with the stuck call NAMED — never the bare wall text."""
    s = _sched()
    s._run_tool_report = _hang_forever
    started = time.monotonic()
    ok, summary, timed_out, _declared, _doc, _silent = \
        s._run_payload_guarded(_tool_row(), 1.0)
    elapsed = time.monotonic() - started
    assert ok is False
    assert timed_out is True
    assert elapsed < 10.0, elapsed
    assert "timed out after 1s in " in summary, summary
    assert "_hang_forever" in summary, summary
    assert "test_schedincident_fixes.py" in summary, summary
    assert "abandoned worker(s) outstanding" in summary, summary


def test_abandoned_workers_are_counted_and_pruned():
    """Repeated timeouts of one job pile up in the open — and a
    worker that later finishes stops counting."""
    s = _sched()
    s._run_tool_report = _hang_briefly
    row = _tool_row()
    ok, _summary, timed_out, _d, _doc, _s = s._run_payload_guarded(row, 0.5)
    assert timed_out is True
    assert s.abandoned_count("j1") == 1
    # A second timeout while the first worker is still stuck: the
    # pile-up is visible, not silent.
    ok, _summary, timed_out, _d, _doc, _s = s._run_payload_guarded(row, 0.5)
    assert timed_out is True
    assert s.abandoned_count("j1") == 2
    # Both brief hangers finish (~2s): the count drains to zero.
    deadline = time.monotonic() + 10.0
    while s.abandoned_count("j1") and time.monotonic() < deadline:
        time.sleep(0.2)
    assert s.abandoned_count("j1") == 0


def test_execute_alert_names_the_stuck_call_and_abandoned_count():
    """The failure alert's policy reason names the stuck call and
    the outstanding abandoned count — the single failure mode."""
    s = _sched(_BoomDB())
    s.db = _BoomDB()
    published: dict[str, Any] = {}

    def _publish(*args, **kwargs):
        published["args"] = args
        published["kwargs"] = kwargs
        return {"delivered": True}

    s.notifier = SimpleNamespace(publish=_publish)

    def _guarded(row, t):
        # The real _run_payload_guarded sets this during the call,
        # after _execute cleared it just before.
        s._last_guard_timeout = {"stuck": "x.py:1:fn", "abandoned": 2,
                                 "timeout_s": 1.0}
        return (False, "job failed: timed out after 1s in x.py:1:fn "
                "(worker abandoned; 2 abandoned worker(s) outstanding "
                "for 'job1'; the stuck call may still be running)",
                True, False, None, None)

    s._run_payload_guarded = _guarded
    s._next_after_run = lambda row, now: (None, True)
    s._record_run = lambda *a, **k: None
    s._ledger = lambda *a, **k: None
    s._emit_bus = lambda *a, **k: None
    s._is_routine = lambda row: False
    out = s._execute({"id": "j1", "name": "job1",
                      "payload_kind": "tool"})
    assert out["ok"] is False
    hints = published["kwargs"]["alert_hints"]
    assert hints["failed"] is True
    assert hints["subject"] == "job1"
    assert hints["reason"].startswith(
        "timed out after 1s in x.py:1:fn"), hints["reason"]
    assert "2 abandoned worker(s) outstanding" in hints["reason"]


# ── alert policy: one named failure mode ────────────────────────────────



def test_failed_reason_names_the_single_failure_mode():
    """A caller-named failure reason IS the decision reason; the
    generic three-way fallback only applies when none is given."""
    policy = AlertPolicy()
    named = policy.decide(
        None, kind="schedule", subject="presence-heartbeat",
        hints=AlertHints(
            failed=True, subject="presence-heartbeat",
            reason="timed out after 300s in presence.py:629:heartbeat "
                   "— 1 abandoned worker(s) outstanding"))
    assert named.reason == (
        "timed out after 300s in presence.py:629:heartbeat "
        "— 1 abandoned worker(s) outstanding")
    assert "did not resolve" not in named.reason
    generic = policy.decide(
        None, kind="schedule", subject="presence-heartbeat",
        hints=AlertHints(failed=True, subject="presence-heartbeat"))
    assert "did not resolve" in generic.reason
    assert "containment failed" in generic.reason


def test_failed_reason_flows_through_dict_hints():
    """Scheduler-style dict hints keep the reason (coerce only
    keeps known fields — a dropped reason would silently fall
    back to the generic line)."""
    hints = AlertHints.coerce(
        {"failed": True, "subject": "improvement tick",
         "reason": "cycle budget exceeded during benchmark"})
    assert hints is not None
    assert hints.reason == "cycle budget exceeded during benchmark"


# ── notifier: one stamp per alert ───────────────────────────────────────



def test_insert_before_stamp_keeps_the_stamp_last():
    """A policy line inserted after a trailing stamp would make
    delivery stamp the text a second time — it rides BEFORE it."""
    stamped = _insert_before_stamp(
        DEFAULT_STAMP_TEXT, "alert policy: page — x (subject: y)")
    assert stamped.count(DEFAULT_STAMP_TEXT) == 1, stamped
    assert stamped.rstrip().endswith(DEFAULT_STAMP_TEXT), stamped
    assert "alert policy: page — x" in stamped
    plain = _insert_before_stamp("body text", "alert policy: page — x")
    assert plain == "body text\nalert policy: page — x", plain
    empty = _insert_before_stamp("", "alert policy: page — x")
    assert empty == "alert policy: page — x", empty


def test_failure_alert_publishes_with_exactly_one_stamp(tmp_path, monkeypatch):
    """The recorded failure alert, end to end: the delivered text
    carries the stamp EXACTLY once and the policy line names the
    single failure mode."""
    from nomorals.agents.context import build_context
    from nomorals.agents.job_reports import failure_doc
    from nomorals.agents.notifier import Notifier
    from nomorals.core.config import load_settings
    from nomorals.social.chat.base import ChatAdapter, SendResult
    from nomorals.social.chat.gateway import ChatGateway

    home = tmp_path / "home"

    class _Rec(ChatAdapter):
        def __init__(self, name):
            super().__init__(media_dir=str(tmp_path / "media"))
            self.name = name
            self.sent: list[str] = []

        def run(self, handler):
            pass

        def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
                 buttons=None, reply_keyboard=None,
                 parse_mode: str = "") -> SendResult:
            self.sent.append(text)
            return SendResult(ok=True, platform=self.name,
                              message_id=f"m{len(self.sent)}")

    settings = load_settings(overrides={
        "home": str(home),
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.enabled": "false",
        "partner.owner_chats": "telegram-personal:4242",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    adapter = _Rec("telegram-personal")
    gateway = ChatGateway({"telegram-personal": adapter}, db=context.db)
    notifier = Notifier(context, gateway=gateway)
    try:
        doc = failure_doc(
            "presence-heartbeat",
            "job failed: timed out after 300s in presence.py:629:heartbeat "
            "(worker abandoned; 1 abandoned worker(s) outstanding for "
            "'presence-heartbeat'; the stuck call may still be running)")
        result = notifier.publish(
            "schedule", "❌ scheduled: presence-heartbeat", "", doc=doc,
            output_kind="scheduled",
            output_meta={"job": "presence-heartbeat",
                         "attempt": "run:1.0"},
            alert_hints={"failed": True, "subject": "presence-heartbeat",
                         "reason": "timed out after 300s in "
                                   "presence.py:629:heartbeat — 1 "
                                   "abandoned worker(s) outstanding"})
        assert result.get("delivered"), result
        assert len(adapter.sent) == 1, adapter.sent
        text = adapter.sent[0]
        assert text.count(DEFAULT_STAMP_TEXT) == 1, text
        assert "timed out after 300s in presence.py:629:heartbeat" in text
        assert "did not resolve" not in text
    finally:
        context.close()


# ── benchmark deadline ──────────────────────────────────────────────────



def _slow_dimension(context, limit):
    time.sleep(5.0)
    return DimensionScore("slow", 0.9, 1, 1, details=[])


def test_benchmark_deadline_stops_a_hanging_dimension(monkeypatch):
    """A dimension whose tasks hang is abandoned at the deadline
    and scores unmeasured — the aggregate never eats the wall."""
    monkeypatch.setattr(bench_mod, "measurable", lambda ctx: True)
    monkeypatch.setattr(bench_mod, "_provider_name", lambda ctx: "test")
    monkeypatch.setattr(bench_mod, "_DIMENSIONS", {
        "fast": lambda ctx, limit: DimensionScore("fast", 0.8, 1, 1,
                                                  details=[]),
        "slow": _slow_dimension,
    })
    started = time.monotonic()
    report = run_benchmark(SimpleNamespace(),
                           deadline=time.monotonic() + 0.3)
    elapsed = time.monotonic() - started
    assert elapsed < 3.0, elapsed
    assert report.deadline_exceeded is True
    assert report.deadline_dimension == "slow"
    assert report.scores["fast"].score == 0.8
    assert report.scores["slow"].score is None


def test_benchmark_past_deadline_measures_nothing(monkeypatch):
    """A deadline already past measures nothing and returns at
    once — no dimension even starts."""
    monkeypatch.setattr(bench_mod, "measurable", lambda ctx: True)
    monkeypatch.setattr(bench_mod, "_provider_name", lambda ctx: "test")
    monkeypatch.setattr(bench_mod, "_DIMENSIONS", {
        "slow": _slow_dimension,
    })
    started = time.monotonic()
    report = run_benchmark(SimpleNamespace(),
                           deadline=time.monotonic() - 1.0)
    assert time.monotonic() - started < 1.0
    assert report.deadline_exceeded is True
    assert report.scores["slow"].score is None


# ── improvement cycle: deadline + detached apply ────────────────────────



def _migrated_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _imp_ctx(*, mode: str = "autonomous", target: float = 0.8,
             db: Database | None = None):
    settings = SimpleNamespace(
        improvement=ImprovementSettings(mode=mode, target=target))
    return SimpleNamespace(settings=settings,
                           db=db if db is not None else _migrated_db(),
                           router=None)


def _imp_report(scores, *, deadline_exceeded=False, dimension=""):
    return SimpleNamespace(scores=scores, provider="test",
                           deadline_exceeded=deadline_exceeded,
                           deadline_dimension=dimension)


def _imp_scores(**kwargs):
    out = {}
    for name, score in kwargs.items():
        out[name] = _DS(name, score, 0, 1, details=[
            {"pass": False, "task": "t1", "detail": "d1"}])
    return out


class _FakeProposal:
    id = "prop-1"
    edits = [{"path": "nomorals/agents/orchestrator.py"}]


class _FakeEvolver:
    def __init__(self, context):
        self.context = context
        self.applied: list[str] = []
        self.apply_started = threading.Event()
        self.apply_release = threading.Event()

    def plan(self, instruction, focus=None):
        return _FakeProposal()

    def apply(self, proposal_id):
        self.applied.append(proposal_id)
        self.apply_started.set()
        # The real gate takes minutes; the test holds the apply
        # open until the cycle has returned, proving detachment.
        self.apply_release.wait(timeout=15.0)
        return {"applied": True, "status": "applied"}


def test_cycle_budget_exceeded_during_benchmark():
    """The cycle records the deadline (naming the dimension) and
    stops — it never plans or applies on an unmeasured cycle."""
    ctx = _imp_ctx(mode="approval")
    loop = ImprovementLoop(ctx)
    report = _imp_report(_imp_scores(reasoning=None), deadline_exceeded=True,
                         dimension="reasoning")
    with mock.patch.object(improv_mod, "measurable", return_value=True), \
            mock.patch.object(improv_mod, "run_benchmark",
                              return_value=report):
        rec = loop.run_cycle(mode="approval", deadline_s=180.0)
    assert rec.action == "unmeasurable", rec.to_dict()
    assert "cycle budget exceeded during benchmark" in rec.rationale
    assert "reasoning" in rec.rationale


def test_autonomous_cycle_launches_detached_apply_and_returns(tmp_path):
    """Autonomous mode never waits on the 1500s gate inside the
    tick: the cycle returns at once with the proposal staged, and
    the detached worker applies it and records the follow-up."""
    # File-backed db: Database connections are thread-local, so an
    # in-memory db would be empty on the detached thread.
    file_db = Database(str(tmp_path / "imp.db"))
    file_db.migrate()
    ctx = _imp_ctx(mode="autonomous", db=file_db)
    loop = ImprovementLoop(ctx)
    evolver = _FakeEvolver(ctx)
    report = _imp_report(_imp_scores(reasoning=0.5, planning=0.9,
                                     tool_use=0.9, self_correction=0.9))
    with mock.patch.object(improv_mod, "measurable", return_value=True), \
            mock.patch.object(improv_mod, "run_benchmark",
                              return_value=report), \
            mock.patch("nomorals.agents.evolution.EvolutionAgent",
                       return_value=evolver):
        started = time.monotonic()
        rec = loop.run_cycle(mode="autonomous", deadline_s=180.0)
        elapsed = time.monotonic() - started
    # The cycle returned while the apply was still held open.
    assert elapsed < 5.0, elapsed
    assert rec.action == "applying", rec.to_dict()
    assert rec.status == "pending"
    assert rec.details.get("detached") is True
    # A duplicate launch for the same proposal is refused.
    assert loop._launch_detached_apply(
        "prop-1", "other-cycle", "reasoning", 0.5, "") is False
    # Release the gate; the worker records the follow-up.
    evolver.apply_release.set()
    deadline = time.monotonic() + 10.0
    follow = None
    while time.monotonic() < deadline:
        rows = [r for r in loop.history(limit=10)
                if r.id == f"{rec.id}-apply"]
        if rows:
            follow = rows[0]
            break
        time.sleep(0.1)
    assert follow is not None, "no detached follow-up record"
    assert follow.action == "applied", follow.to_dict()
    assert follow.status == "neutral"
    assert evolver.applied == ["prop-1"]


# ── presence heartbeat under a held write lock ──────────────────────────

def test_heartbeat_under_held_lock_returns_well_inside_the_wall(tmp_path):
    """The 4:30 PM presence-heartbeat hang is NOT a held-lock
    starve: with a writer holding BEGIN IMMEDIATE for a full
    minute, heartbeat() fails fast (tens of seconds) instead of
    hanging to the 300s wall.  The scheduler watchdog names the
    stuck call if it ever does hang again."""
    from nomorals.autonomy.idle import workspace_db
    from nomorals.autonomy.presence import ensure_schema, heartbeat

    ws = tmp_path / "ws"
    ws.mkdir()
    db_path = ws / "nomorals.db"
    seed = workspace_db(ws)
    ensure_schema(seed)
    seed.execute("CREATE TABLE IF NOT EXISTS repro_hold (id INTEGER)")
    seed.close()

    holder_ready = threading.Event()
    stop_hold = threading.Event()

    def _holder():
        try:
            conn = sqlite3.connect(str(db_path), timeout=30.0,
                                   isolation_level=None)
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO repro_hold (id) VALUES (1)")
            holder_ready.set()
            stop_hold.wait(timeout=60.0)
            conn.execute("ROLLBACK")
            conn.close()
        except Exception as exc:  # noqa: BLE001 - holder teardown
            assert isinstance(exc, Exception)

    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    assert holder_ready.wait(timeout=10), "holder never took the lock"
    try:
        db = workspace_db(ws)
        try:
            started = time.monotonic()
            result = heartbeat(db)
            elapsed = time.monotonic() - started
        finally:
            db.close()
        assert isinstance(result, dict), result
        assert elapsed < 120.0, elapsed
    finally:
        stop_hold.set()


# ── double-send: home-anchored lock + durable offset ───────────────────



def test_bot_lock_paths_are_home_anchored(tmp_path):
    """The consumer lock is the one adapter path that used to be
    CWD-relative: a restart launched from another directory held
    a DIFFERENT lock file and polled the same token beside the
    survivor.  Both builders now anchor it to the data home."""
    from nomorals.core.config import load_settings
    from nomorals.social.chat import build_adapter

    home = tmp_path / "home"
    settings = load_settings(overrides={
        "home": str(home),
        "chat.telegram_bot_enabled": "true",
        "chat.telegram_bot_token": "123:secret",
        "chat.telegram_personal_bot_enabled": "true",
        "chat.telegram_personal_bot_token": "456:secret",
    })
    gated = build_adapter(settings, "telegram-bot")
    personal = build_adapter(settings, "telegram-personal")
    assert gated is not None and personal is not None
    for adapter in (gated, personal):
        assert os.path.isabs(adapter._lock_path), adapter._lock_path
        assert str(adapter._lock_path).startswith(str(home)), \
            adapter._lock_path
        assert os.path.isabs(adapter._offset_path), adapter._offset_path
    assert "telegram-bot.lock" in gated._lock_path
    assert "telegram-personal.lock" in personal._lock_path
    # The two_bots construction site anchors the same way when it
    # is given the home.
    built = build_both(TwoBotConfig(
        gated_enabled=True, gated_token="123:secret",
        personal_enabled=True, personal_token="456:secret",
        home_dir=str(home)))
    assert str(built["telegram-bot"]._lock_path).startswith(str(home))
    assert str(built["telegram-personal"]._lock_path).startswith(
        str(home))


def test_second_consumer_on_home_lock_refuses(tmp_path):
    """Two processes of one home can never both poll: the second
    acquire on the same home-anchored lock file refuses."""
    lock = SingleConsumerLock(tmp_path / "data" / "telegram-bot.lock")
    assert lock.acquire() is True
    try:
        second = SingleConsumerLock(tmp_path / "data" / "telegram-bot.lock")
        assert second.acquire() is False
    finally:
        lock.release()


def test_update_offset_survives_restart(tmp_path):
    """The durable offset: a loop that restarts never re-handles
    an update a previous loop already answered."""
    lock = tmp_path / "telegram-bot.lock"
    offset = tmp_path / "telegram-bot.offset.json"
    a = TelegramBotAdapter(token="123:secret", lock_path=str(lock),
                           offset_path=str(offset))
    a._persist_offset(105)
    b = TelegramBotAdapter(token="123:secret", lock_path=str(lock),
                           offset_path=str(offset))
    assert b._load_offset() == 106
    assert b._offset == 0  # load is explicit, not constructor magic
    b._offset = b._load_offset()
    assert b._offset == 106


def test_dispatch_persists_offset_and_replay_is_dropped(tmp_path):
    """One update dispatched twice reaches the handler once, and
    the offset file already holds it after the first dispatch —
    so a restarted loop skips it even if its stale-skip fails."""
    lock = tmp_path / "telegram-bot.lock"
    offset = tmp_path / "telegram-bot.offset.json"
    adapter = TelegramBotAdapter(token="123:secret", lock_path=str(lock),
                                 offset_path=str(offset))
    seen: list[Any] = []
    adapter._deliver = lambda handler, msg: seen.append(msg)  # type: ignore[method-assign]
    update = {
        "update_id": 106,
        "message": {
            "message_id": 5,
            "date": 1700000000,
            "chat": {"id": 4242, "type": "private"},
            "from": {"id": 4242, "is_bot": False,
                     "first_name": "death"},
            "text": "/help",
        },
    }
    adapter._dispatch_update(update, lambda m: seen.append(m))
    adapter._dispatch_update(update, lambda m: seen.append(m))
    assert len(seen) == 1, seen
    assert adapter._offset == 107
    assert json.loads(offset.read_text(encoding="utf-8")) == {
        "update_id": 106}


# ── double-send: the two-runtime shape ──────────────────────────────────



class _RecordingAdapter(ChatAdapter):
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


class _BotAdapter(_RecordingAdapter):
    def __init__(self, name: str, bot_id: int, username: str) -> None:
        super().__init__(name)
        self._bot_id = bot_id
        self._bot_username = username


class _BareUserbotAdapter(_RecordingAdapter):
    """Split-hosting userbot: NO companion bot ids (the live
    shape) — the peer-is-bot stamp is what the runtime reads."""

    _me_username = "peacethefirst"
    _me_id = int(OWNER_ID)


class _UserbotWithCompanions(_RecordingAdapter):
    """Same userbot, built with both bot tokens."""

    _me_username = "peacethefirst"
    _me_id = int(OWNER_ID)
    companion_bot_ids = frozenset({PERSONAL_BOT_ID})


def _rt_context(home: str):
    settings = _load_settings(overrides={
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
        ("schedincident", "2000-01-01", time.time()))
    return context


def _rt_make(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    return runtime


def _rt_msg(platform, *, chat_id, text, incoming, message_id,
            sender_id=OWNER_ID, own=False, peer_is_bot=False,
            service=None, reply_to=""):
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM,
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


def _rt_bot_copy(text, message_id):
    return _rt_msg("telegram-personal", chat_id=OWNER_ID, text=text,
                   incoming=True, message_id=message_id)


def _rt_echo_copy(text, message_id, *, peer_is_bot=True):
    return _rt_msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=text,
                   incoming=False, message_id=message_id, own=True,
                   peer_is_bot=peer_is_bot)


def _rt_peer_post(text, message_id, *, reply_to="", service="peer_post"):
    return _rt_msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=text,
                   incoming=True, message_id=message_id,
                   sender_id=str(PERSONAL_BOT_ID), service=service,
                   reply_to=reply_to)


def _rt_texts(adapter):
    return [t for t, _b in adapter.sent]


@pytest.fixture()
def _split_homes():
    """Two runtimes, SEPARATE homes — the live shape."""
    clear_failures()
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-si-a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-si-b-",
                                        dir=_TEST_TMPDIR)
    ctx_a = _rt_context(tmp_a.name)
    ctx_b = _rt_context(tmp_b.name)
    ad_a = {"telegram-personal": _BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": _BareUserbotAdapter("telegram")}
    rt_a = _rt_make(ctx_a, ad_a)
    rt_b = _rt_make(ctx_b, ad_b)
    rt_b._peer_answer_window_s = 5.0
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


def test_help_answered_exactly_once_across_two_runtimes(_split_homes):
    """The 4:29 PM /help, replayed in the live shape: ONE menu
    with its category grid, from the primary — the echo defers,
    the primary's answer resolves it, the sweep stays silent."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = _split_homes
    rt_a._process_inner(_rt_bot_copy("/help", "300"))
    assert len(_rt_texts(ad_a["telegram-personal"])) == 1
    text, buttons = ad_a["telegram-personal"].sent[0]
    assert "Devon — command menu" in text, text[:200]
    assert buttons, "the one menu carries its category grid"
    rt_b._process_inner(_rt_echo_copy("/help", "300"))
    assert _rt_texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 1
    rt_b._process_inner(_rt_peer_post(text, "301", reply_to="300"))
    assert _rt_texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 0
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    assert _rt_texts(ad_b["telegram"]) == []
    total = (len(_rt_texts(ad_a["telegram-personal"]))
             + len(_rt_texts(ad_b["telegram"])))
    assert total == 1


def test_unstamped_echo_still_defers_when_companions_wired():
    """The gap this lane closes at the runtime: when Telethon
    never resolved the peer entity, the echo carries NO stamp —
    but the adapter's companion ids (wired from the tokens) let
    the peer rule fire anyway, so the echo still defers."""
    clear_failures()
    tmp = tempfile.TemporaryDirectory(prefix="nm-si-c-",
                                      dir=_TEST_TMPDIR)
    ctx = _rt_context(tmp.name)
    adapter = _UserbotWithCompanions("telegram")
    rt = _rt_make(ctx, {"telegram": adapter})
    rt._peer_answer_window_s = 5.0
    try:
        rt._process_inner(_rt_echo_copy("/help", "310",
                                        peer_is_bot=False))
        assert _rt_texts(adapter) == []
        assert rt._peer_deferral().pending_count() == 1
        rt._process_inner(_rt_peer_post("menu", "311", reply_to="310"))
        assert rt._peer_deferral().pending_count() == 0
        rt._sweep_peer_deferrals(now=time.time() + 60)
        assert _rt_texts(adapter) == []
    finally:
        try:
            rt.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        ctx.close()
        tmp.cleanup()


def test_owner_echo_stamps_peer_is_bot_from_companion_ids(tmp_path):
    """The gap this lane closes at the adapter: the own-echo's
    peer stamp must not depend on Telethon resolving the chat
    entity — when the peer id is a configured companion bot, the
    stamp is set from the ids alone."""
    from types import SimpleNamespace as _NS

    from nomorals.social.chat.telegram import TelegramAdapter

    session = tmp_path / "telegram.session"
    adapter = TelegramAdapter(
        api_id=1, api_hash="hash",
        session_path=str(session),
        media_dir=str(tmp_path / "media"),
        companion_bot_ids={PERSONAL_BOT_ID})
    received: list[Any] = []

    def _handler(message):
        received.append(message)

    # The outgoing event, with the chat entity UNRESOLVED
    # (event.chat is None — the live failure shape).
    message = _NS(id=77, action=None, message="/help",
                  text="/help", reply_to=None, date=None)
    event = _NS(chat=None, chat_id=str(PERSONAL_BOT_ID),
                message=message)
    adapter._deliver_owner_echo(event, _handler)
    assert len(received) == 1
    echo = received[0]
    assert (echo.meta or {}).get("own_message") is True
    assert (echo.meta or {}).get("peer_is_bot") is True
    assert echo.text == "/help"


def test_conversational_echo_never_answers_and_bot_answers_once(
        _split_homes):
    """The doubled bank-details reply, replayed: the owner's
    plain question reaches the userbot only as an own-echo
    (incoming=False), which the conversational path never
    answers; the primary answers it exactly once."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = _split_homes
    question = "what is Ada's bank details?"
    rt_b._process_inner(_rt_echo_copy(question, "320"))
    assert _rt_texts(ad_b["telegram"]) == []
    rt_a._process_inner(_rt_msg(
        "telegram-personal", chat_id=OWNER_ID, text=question,
        incoming=True, message_id="320"))
    sent = _rt_texts(ad_a["telegram-personal"])
    assert len(sent) == 1, sent
    assert sent[0].strip(), "the one reply is not empty"
    total = len(sent) + len(_rt_texts(ad_b["telegram"]))
    assert total == 1
