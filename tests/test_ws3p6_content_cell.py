"""WS3 P6 — the content cell, formalized.

Pins, per the pass brief:
(a) a weekly-pipeline run through the cell produces drafts whose
    receipt chain reads contract → draft ids in one record;
(b) the skeptic gate blocks a draft whose factual claim has no
    evidence and lets an evidenced one through;
(c) a shorts run through the cell lands its pieces with receipts;
(d) a series collects a draft + a short + a scheduled post and the
    calendar board renders all three kinds together;
(e) the _draft_post_fn remainder is closed — the poster is wired,
    a posterless fire fails loudly, and a scheduler without
    schedule_once is refused before any state moves;
(f) the /calendar command is catalogued owner-only (the P5 pin
    shape); P4/P5 suites run in the adjoining battery.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.agents.content_cell import (
    ReceiptStore,
    make_weekly_cell,
    receipt_chain,
    render_receipt,
)
from nomorals.agents.delegation import DelegationStore
from nomorals.media.contentops.pipeline import ContentCalendar
from nomorals.social import drafts as drafts_module
from nomorals.social.content_pipeline import (
    set_cell_factory,
    weekly_content,
)
from nomorals.social.drafts import (
    DraftQueue,
    post_scheduled_draft,
    schedule_post,
)
from nomorals.social.series import SeriesStore, render_calendar
from nomorals.storage.db import Database


@pytest.fixture()
def db():
    database = Database(":memory:")
    database.migrate()
    return database


@pytest.fixture()
def queue(tmp_path):
    q = DraftQueue(db_path=str(tmp_path / "drafts.db"))
    yield q
    q.close()


FINDINGS = [
    {"title": "Agent workflows grew 40% this quarter",
     "snippet": "Usage of agent workflows grew 40% across the teams"
                " surveyed.",
     "url": "https://example.com/growth"},
    {"title": "Local models get cheaper",
     "snippet": "Running costs keep falling for small local models.",
     "url": "https://example.com/local"},
]

CLEAN_COPY = ("Shipping agents that finish real work — here is what"
              " changed this week and why it matters.")


# ── (a) weekly run through the cell → receipt chain ───────────────────

def test_weekly_run_through_cell_lands_drafts_with_receipts(db, queue):
    cell = make_weekly_cell(db, "AI agents")
    result = weekly_content(
        "AI agents", findings=FINDINGS, queue=queue, db=db,
        llm_fn=lambda prompt: CLEAN_COPY, cell=cell)
    assert result.contract_id
    assert result.drafts, "the run must still produce its batch"
    assert not result.blocked
    # Receipts: one per queued (draft, platform) piece, all on the
    # run's contract.
    receipts = ReceiptStore(db).for_contract(result.contract_id)
    assert receipts
    draft_ids = {d_id for d in result.drafts
                 for d_id in (d.draft_ids or {}).values()}
    assert {r["draft_id"] for r in receipts} == draft_ids
    # The chain reads end-to-end in one record per piece.
    chain = receipt_chain(db, queue, receipts[0])
    assert chain["contract"]["status"] == "completed"
    assert chain["draft"] is not None
    text = render_receipt(chain)
    assert "contract completed" in text
    assert receipts[0]["draft_id"] not in text  # ids never print
    # Drafts sit in review exactly as before the cell existed.
    for d in result.drafts:
        for d_id in (d.draft_ids or {}).values():
            assert queue.get(d_id).status in (
                "pending_review", "scheduled")


def test_weekly_factory_seam_builds_the_cell(db, queue):
    set_cell_factory(make_weekly_cell)
    try:
        result = weekly_content(
            "AI agents", findings=FINDINGS, queue=queue, db=db,
            llm_fn=lambda prompt: CLEAN_COPY)
    finally:
        set_cell_factory(None)
    assert result.contract_id
    assert result.receipts


def test_weekly_without_cell_is_the_legacy_run(queue):
    result = weekly_content(
        "AI agents", findings=FINDINGS, queue=queue,
        llm_fn=lambda prompt: CLEAN_COPY)
    assert result.contract_id == ""
    assert result.receipts == []
    assert result.drafts


# ── (b) the skeptic gate ──────────────────────────────────────────────

def test_skeptic_blocks_untraced_numbers(db, queue):
    cell = make_weekly_cell(db, "AI agents")
    result = weekly_content(
        "AI agents", findings=FINDINGS, queue=queue, db=db,
        llm_fn=lambda prompt: "Agent workflows grew 900% this week!!",
        cell=cell)
    # The invented-figure drafts are blocked and recorded; the
    # pipeline's own template drafts (no untraced numbers) still
    # land — the gate is per-draft, not per-batch.
    assert result.blocked, "blocked drafts are recorded, not dropped"
    assert all("900" in b["reason"] for b in result.blocked)
    assert all("900" not in d.content for d in result.drafts)
    for d in queue.pending():
        assert "900" not in d.content
    # On the record: a challenge, resolved contradicted.
    store = DelegationStore(db)
    claims = store.claims(result.contract_id)
    resolutions = [c for c in claims if c.kind == "resolution"]
    assert any(c.verdict == "contradicted" for c in resolutions)


def test_skeptic_lets_evidenced_numbers_through(db, queue):
    cell = make_weekly_cell(db, "AI agents")
    result = weekly_content(
        "AI agents", findings=FINDINGS, queue=queue, db=db,
        llm_fn=lambda prompt: "Agent workflows grew 40% this quarter"
                              " — the survey numbers are in.",
        cell=cell)
    assert result.drafts
    assert result.blocked == []
    assert any("40%" in d.content for d in result.drafts)


# ── (c) shorts through the cell ───────────────────────────────────────

def _an_hour_ago() -> str:
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc)
            - timedelta(hours=1)).isoformat()


class _FakeCalendarPipeline:
    """The sweep's pipeline seam: a real ContentCalendar over a
    tmp root, plan/run faked at the render boundary."""

    def __init__(self, tmp_path, video: bytes = b"fake-mp4-bytes"):
        self.runs_root = tmp_path
        self.calendar = ContentCalendar(
            tmp_path, get_niche=lambda name: {"posts_per_day": 5})
        self._video = video

    def plan(self, niche, topic, *, platforms=None, style=""):
        return SimpleNamespace(
            id="job_fake1", run_id="run_fake1", topic=topic,
            niche=niche)

    def run(self, job):
        run_dir = self.runs_root / job.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        video = run_dir / "final.mp4"
        video.write_bytes(self._video)
        (run_dir / "post_copy.json").write_text(json.dumps({
            "title": "Three hooks that land",
            "description": "watch to the end",
            "hashtags": ["hooks"]}), encoding="utf-8")
        return SimpleNamespace(ok=True, final_path=str(video),
                               error=None, job_id=job.id,
                               run_id=job.run_id)


class _FakeLedger:
    def __init__(self):
        self.entries = []

    def record(self, **kwargs):
        entry = {"id": f"led_{len(self.entries) + 1}", **kwargs}
        self.entries.append(entry)
        return entry


class _FakeYouTube:
    def publish(self, video_path, title, **kwargs):
        ledger = kwargs.get("ledger")
        entry = ledger.record(
            platform="youtube", file=video_path, title=title,
            platform_video_id="vid123", status="uploaded")
        return {"video_id": "vid123", "ledger_id": entry["id"]}


def test_shorts_sweep_publishes_due_post_with_receipts(db, tmp_path):
    from nomorals.agents.shorts_publisher import publish_due_shorts

    pipe = _FakeCalendarPipeline(tmp_path)
    post = pipe.calendar.add(
        "motivation", "morning hooks that land",
        platforms=["youtube"], scheduled_for=_an_hour_ago())
    ledger = _FakeLedger()
    outcomes = publish_due_shorts(
        pipeline=pipe, db=db, ledger=ledger,
        publisher_for=lambda platform: _FakeYouTube())
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome["status"] == "posted"
    assert outcome["platforms"]["youtube"]["external_id"] == "vid123"
    assert pipe.calendar.get(post.id).status == "posted"
    assert [e["status"] for e in ledger.entries] == ["uploaded"]
    # The cell record: contract completed + a shorts receipt.
    store = DelegationStore(db)
    assert store.get_contract(outcome["contract_id"]).status == \
        "completed"
    receipts = ReceiptStore(db).for_contract(outcome["contract_id"])
    assert len(receipts) == 1
    assert receipts[0]["kind"] == "shorts"
    chain = receipt_chain(db, None, receipts[0])
    assert chain["post"]["external_id"] == "vid123"
    assert "vid123" in render_receipt(chain)


def test_shorts_sweep_records_an_honest_refusal(db, tmp_path):
    from nomorals.agents.shorts_publisher import publish_due_shorts
    from nomorals.media.contentops.publish import CapabilityUnavailable

    class _Refusing:
        def publish(self, *args, **kwargs):
            raise CapabilityUnavailable(
                "youtube", "no channel token configured",
                manual_step="connect the channel")

    pipe = _FakeCalendarPipeline(tmp_path)
    post = pipe.calendar.add(
        "motivation", "evening hooks", platforms=["youtube"],
        scheduled_for=_an_hour_ago())
    ledger = _FakeLedger()
    outcomes = publish_due_shorts(
        pipeline=pipe, db=db, ledger=ledger,
        publisher_for=lambda platform: _Refusing())
    outcome = outcomes[0]
    assert outcome["status"] == "failed"
    assert "no channel token" in outcome["platforms"]["youtube"]["error"]
    assert pipe.calendar.get(post.id).status == "failed"
    assert [e["status"] for e in ledger.entries] == ["failed"]


def test_shorts_sweep_names_a_post_with_no_platforms(db, tmp_path):
    from nomorals.agents.shorts_publisher import publish_due_shorts

    pipe = _FakeCalendarPipeline(tmp_path)
    post = pipe.calendar.add(
        "motivation", "platformless", platforms=[],
        scheduled_for=_an_hour_ago())
    outcomes = publish_due_shorts(
        pipeline=pipe, db=db, ledger=_FakeLedger(),
        publisher_for=lambda platform: None)
    assert outcomes[0]["status"] == "failed"
    assert "no platforms" in outcomes[0]["reason"]
    assert pipe.calendar.get(post.id).status == "failed"


# ── (d) series + the unified calendar ─────────────────────────────────

def test_series_collects_all_three_kinds_and_board_renders(db, queue,
                                                          tmp_path):
    store = SeriesStore(db)
    series = store.create_series("launch week",
                                 objective="first 100 users")
    draft = queue.create_draft("Launch day one: we shipped the thing",
                               platforms=["linkedin"])
    queue.propose(draft.id)
    store.attach(series["id"], "draft", draft.id)

    calendar = ContentCalendar(
        tmp_path, get_niche=lambda name: {"posts_per_day": 5})
    from datetime import datetime, timedelta, timezone

    soon = (datetime.now(timezone.utc)
            + timedelta(hours=2)).isoformat()
    short = calendar.add("motivation", "launch teaser short",
                         platforms=["youtube"], scheduled_for=soon)
    store.attach(series["id"], "short", short.id)

    db.execute(
        "INSERT INTO social_posts (platform, content, status,"
        " scheduled_at, created_at) VALUES (?, ?, 'scheduled', ?, ?)",
        ("x", "scheduled launch post", time.time() + 3600,
         time.time()))
    row = db.query_one("SELECT id FROM social_posts LIMIT 1")
    store.attach(series["id"], "post", str(row["id"]))

    members = store.members(series["id"])
    assert [m["kind"] for m in members] == ["draft", "short", "post"]

    board = render_calendar(db, queue, shorts=calendar.list(),
                            store=store)
    assert "awaiting your decision" in board
    assert "we shipped the thing" in board       # the draft
    assert "launch teaser short" in board        # the short (series)
    assert "scheduled launch post" in board      # the scheduled post
    assert "launch week" in board
    assert "first 100 users" in board            # campaign objective
    assert draft.id not in board                 # ids never print


def test_series_stamps_the_draft_itself(db, queue):
    from nomorals.social.series import stamp_draft

    store = SeriesStore(db)
    series = store.create_series("evergreen")
    draft = queue.create_draft("a piece of the run",
                               platforms=["x"])
    store.attach(series["id"], "draft", draft.id)
    stamp_draft(queue, draft, "evergreen")
    assert queue.get(draft.id).metadata["series"] == "evergreen"
    assert store.series_of("draft", draft.id)["name"] == "evergreen"


# ── (e) the _draft_post_fn remainder, closed ──────────────────────────

class _FakeUnifiedScheduler:
    """register_action + schedule_once, the unified scheduler's
    shape — schedule_post's real counterparty."""

    def __init__(self):
        self.actions = {}
        self.once = []

    def register_action(self, task_id, fn):
        self.actions[task_id] = fn

    def schedule_once(self, task_id, when_ts, action, params):
        self.once.append((task_id, when_ts, action, params))
        return None


class _FakeManager:
    def __init__(self):
        self.context = None
        self.published = []

    def publish(self, content, platforms=None, actor="user",
                confirmation=""):
        self.published.append((content, platforms))
        return SimpleNamespace(posted=["x"], failed=[], results=[])


@pytest.fixture()
def _no_module_poster():
    old = drafts_module._DEFAULT_DRAFT_POST_FN
    drafts_module._DEFAULT_DRAFT_POST_FN = None
    yield
    drafts_module._DEFAULT_DRAFT_POST_FN = old


def test_scheduled_fire_without_poster_fails_loudly(queue,
                                                    _no_module_poster):
    scheduler = _FakeUnifiedScheduler()
    draft = queue.create_draft("post me at six", platforms=["x"])
    queue.propose(draft.id)
    schedule_post(scheduler, queue, draft.id, time.time() + 3600)
    assert queue.get(draft.id).status == "scheduled"
    fire = scheduler.actions["social.post_draft"]
    result = fire({"draft_id": draft.id})
    assert result["ok"] is False
    assert "no draft poster wired" in result["error"]
    assert queue.get(draft.id).status == "scheduled"  # never faked


def test_scheduled_fire_posts_through_the_wired_poster(
        queue, _no_module_poster):
    manager = _FakeManager()
    drafts_module._DEFAULT_DRAFT_POST_FN = (
        lambda draft_id: post_scheduled_draft(
            draft_id, queue=queue, manager=manager))
    scheduler = _FakeUnifiedScheduler()
    draft = queue.create_draft("post me at six", platforms=["x"])
    queue.propose(draft.id)
    schedule_post(scheduler, queue, draft.id, time.time() + 3600)
    result = scheduler.actions["social.post_draft"](
        {"draft_id": draft.id})
    assert result["ok"] is True
    assert manager.published == [("post me at six", ["x"])]
    assert queue.get(draft.id).status == "posted"


def test_schedule_post_refuses_a_scheduler_without_schedule_once(queue):
    class _DurableShape:
        def add(self, *args, **kwargs):  # the durable scheduler's API
            raise AssertionError("must not be called")

    draft = queue.create_draft("stays in review", platforms=["x"])
    queue.propose(draft.id)
    with pytest.raises(RuntimeError, match="schedule_once"):
        schedule_post(_DurableShape(), queue, draft.id,
                      time.time() + 3600)
    # Refused BEFORE any state moved.
    assert queue.get(draft.id).status == "pending_review"


def test_post_scheduled_draft_names_its_states(queue):
    manager = _FakeManager()
    draft = queue.create_draft("not armed", platforms=["x"])
    queue.propose(draft.id)
    result = post_scheduled_draft(draft.id, queue=queue,
                                  manager=manager)
    assert result["ok"] is False
    assert "pending_review" in result["error"]
    assert manager.published == []


# ── (f) /calendar through the real runtime dispatch ──────────────────

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import (  # noqa: E402
    _FakeTools,
    _msg,
    _RecordingGateway,
)

from nomorals.agents.context import build_context  # noqa: E402
from nomorals.agents.partner_runtime import PartnerRuntime  # noqa: E402
from nomorals.core.config import load_settings  # noqa: E402
from nomorals.social import drafts as drafts_mod  # noqa: E402

_REAL_QUEUE = DraftQueue


class RuntimeCalendarTests(unittest.TestCase):
    """/calendar through the REAL partner-runtime dispatch: the
    board renders, a campaign is created, a draft attaches by its
    board number, and a non-owner is refused."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.db_path = str(Path(self.tmp.name) / "drafts.db")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        self._patcher = mock.patch.object(
            drafts_mod, "DraftQueue",
            lambda *a, **k: _REAL_QUEUE(db_path=self.db_path))
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str, *, owner: bool = True) -> str:
        reply = self.runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, chat_key="telegram:999", owner=owner))
        return reply.text if hasattr(reply, "text") else str(reply)

    def test_board_campaign_attach_and_owner_gate(self) -> None:
        board = self._run("/calendar")
        self.assertIn("content calendar", board)

        made = self._run("/calendar new launch week"
                         " | first 100 users")
        self.assertIn("campaign", made)
        self.assertIn("launch week", made)

        queue = _REAL_QUEUE(db_path=self.db_path)
        try:
            draft = queue.create_draft(
                "Launch day one: we shipped the thing",
                platforms=["linkedin"])
            queue.propose(draft.id)
        finally:
            queue.close()

        added = self._run("/calendar add launch week draft 1")
        self.assertIn("added draft #1", added)

        view = self._run("/calendar series launch week")
        self.assertIn("we shipped the thing", view)
        self.assertIn("first 100 users", view)

        refused = self._run("/calendar new hijack", owner=False)
        self.assertNotIn("started", refused)
        store = SeriesStore(self.context.db)
        self.assertIsNone(store.get_by_name("hijack"))


# ── (f) the /calendar catalog pins ────────────────────────────────────

def test_calendar_command_is_catalogued_owner_only():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
        detailed_help,
        help_text,
    )

    assert "calendar" in CONTROL_COMMANDS
    assert "calendar" in COMMAND_DETAILS
    grouped = [c for _, cs in LIST_GROUPS for c in cs]
    assert grouped.count("calendar") == 1
    assert "calendar" in LIST_ONELINERS
    assert "calendar" not in PUBLIC_COMMANDS
    assert "/calendar" in help_text()
    assert "/calendar" in detailed_help("calendar")
