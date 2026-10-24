"""Briefs <-> DraftQueue id-contract pins.

The contract, proven from the repo: once a briefs draft enters the queue,
the QUEUE owns the id. DraftQueue.create_draft assigns a ULID
(nomorals/social/drafts.py, via core.ids.new_id); queue.schedule resolves
only queue ids and raises KeyError otherwise. Every other schedule_post
caller passes a queue id (partner/runtime.py: pending draft.id or
new_draft.id; content_pipeline.py: draft.id from create_draft; drafts.py
handle_draft_callback: the queue id from the button callback). briefs.py
was the one caller passing its foreign bd_ id to schedule_post, so the
scheduler step recorded KeyError "unknown draft 'bd_...'" and booked no
timed task while schedule_draft still returned ok True.

These pins run the REAL DraftQueue end to end (no duck-typed queue):
brief -> schedule -> queue draft SCHEDULED + social.post_draft action
registered + timed task booked under the queue id, no KeyError.
"""

import os
import tempfile

import pytest

from nomorals.marketing import failures as mk
from nomorals.marketing.briefs import BriefStore
from nomorals.social.drafts import DraftQueue

MOCK_RESULTS = [
    {"title": "Lagos rents explained",
     "snippet": "How much is rent in Lagos? 2 bedroom flats in Yaba cost around 2m per year. What should you look for when renting? Always inspect before paying."},
]


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    mk.clear_failures()
    yield
    mk.clear_failures()


def _sites():
    return [f.site for f in mk.recent_failures(limit=50)]


class _FakeScheduler:
    """Recording-only scheduler: records, never raises."""

    def __init__(self):
        self.actions = []
        self.once = []

    def register_action(self, name, fn):
        self.actions.append(name)

    def schedule_once(self, task_id, when_ts, action, params):
        self.once.append((task_id, when_ts, action, params))
        return None


def _briefed_store(tmp_path):
    store = BriefStore(db_path=os.path.join(str(tmp_path), "briefs.db"))
    brief = store.build_brief("Lagos rents",
                              search_fn=lambda t: MOCK_RESULTS)
    draft = store.draft_from_brief(brief)
    assert draft is not None
    return store, draft


def test_schedule_draft_end_to_end_with_real_queue(tmp_path):
    """brief -> schedule -> queue SCHEDULED + action + timed task, no KeyError."""
    store, draft = _briefed_store(tmp_path)
    queue = DraftQueue(db_path=os.path.join(str(tmp_path), "queue.db"))
    scheduler = _FakeScheduler()
    when_ts = 9999999999.0
    res = store.schedule_draft(draft.draft_id, when_ts,
                               scheduler=scheduler, queue=queue)
    assert res["ok"] is True
    assert "briefs.BriefStore.schedule_draft.scheduler" not in _sites()
    assert "briefs.BriefStore.schedule_draft.queue" not in _sites()
    queued = queue.pending()
    assert len(queued) == 1
    qd = queued[0]
    # The mismatch class itself: the queue id is NOT the briefs bd_ id.
    assert qd.id != draft.draft_id
    assert draft.draft_id.startswith("bd_")
    # ...and the scheduled record is the queue draft, under its own id.
    assert qd.status == "scheduled"
    assert qd.scheduled_time == when_ts
    assert (qd.metadata or {}).get("brief_id") == draft.brief_id
    assert (qd.metadata or {}).get("pipeline") == "brief_first"
    assert scheduler.actions == ["social.post_draft"]
    assert scheduler.once == [
        (f"post-draft-{qd.id}", when_ts, "social.post_draft",
         {"draft_id": qd.id, "platforms": [draft.platform]})]
    assert store.get_draft(draft.draft_id).scheduled_for == when_ts


def test_schedule_draft_queue_only_still_proposes(tmp_path):
    """No scheduler: the queue step alone still lands the draft in review."""
    store, draft = _briefed_store(tmp_path)
    queue = DraftQueue(db_path=os.path.join(str(tmp_path), "queue.db"))
    res = store.schedule_draft(draft.draft_id, 9999999999.0, queue=queue)
    assert res["ok"] is True
    assert "briefs.BriefStore.schedule_draft.queue" not in _sites()
    queued = queue.pending()
    assert len(queued) == 1
    assert queued[0].status == "pending_review"
    assert queued[0].id != draft.draft_id


def test_schedule_draft_scheduler_without_queue_preserved(tmp_path):
    """Degenerate path preserved, not redesigned: scheduler with no queue
    cannot resolve any draft, so the scheduler failure is still recorded
    (as before) and the briefs record is still stamped."""
    store, draft = _briefed_store(tmp_path)
    res = store.schedule_draft(draft.draft_id, 9999999999.0,
                               scheduler=_FakeScheduler(), queue=None)
    assert res["ok"] is True
    assert "briefs.BriefStore.schedule_draft.scheduler" in _sites()
    assert store.get_draft(draft.draft_id).scheduled_for == 9999999999.0
