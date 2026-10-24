"""Briefs import-fix pins: the two lazy imports in briefs.py used three-dot
relatives (``from ...social...``), which from ``nomorals.marketing`` climb
beyond the top-level package and ALWAYS raise ImportError. Until this fix,
``score_draft`` silently fell back to a neutral brand_voice of 50.0 on every
call, and the scheduler step in ``schedule_draft`` never ran at all — both
recorded in the marketing ledger once pass #147 started recording them.

The voice import is a pure depth fix (``...`` -> ``..``). The scheduler
import is a depth fix PLUS a module correction, proven from the repo:
``nomorals/social/content_pipeline.py`` neither defines nor module-level
exports ``schedule_post`` (it only imports it function-locally inside
``weekly_content``), so even a two-dot import from content_pipeline still
raises ``ImportError: cannot import name 'schedule_post'``. The name lives
in ``nomorals/social/drafts.py`` (defined there; canonical in-repo user is
``agents/partner/runtime.py`` importing from ``social.drafts``), with
exactly the ``(scheduler, queue, draft_id, when_ts, *, platforms)``
signature the briefs call site uses.

Every test redirects stores with the app's own ``NM_HOME`` override (never
the real home, never the repo tree).
"""

import os
import tempfile

import pytest

from nomorals.marketing import failures as mk
from nomorals.marketing.briefs import BriefStore, ContentBrief
from nomorals.social.voice import virality_score

MOCK_RESULTS = [
    {"title": "Lagos rents explained",
     "snippet": "How much is rent in Lagos? 2 bedroom flats in Yaba cost around 2m per year. What should you look for when renting? Always inspect before paying."},
    {"title": "Yaba rental guide",
     "snippet": "Why are Yaba rents rising? The area is close to tech hubs. How do you negotiate with Lagos landlords? Get everything in writing."},
]


def _mock_search(topic):
    return MOCK_RESULTS


def _db():
    return os.path.join(tempfile.mkdtemp(), "briefs.db")


# Scores 68.0 through the real virality heuristic (strong hook + early
# question + CTA) — decisively not the neutral 50.0 fallback.
HOOK_TEXT = (
    "Stop overpaying for Lagos rent. Here is the one inspection trick "
    "landlords hope you skip — are you checking it? Save this and share "
    "it with a friend house-hunting in Yaba."
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    mk.clear_failures()
    yield
    mk.clear_failures()


def _sites():
    return [f.site for f in mk.recent_failures(limit=50)]


class _FakeQueue:
    """Duck-typed stand-in for DraftQueue: records, never raises."""

    def __init__(self):
        self.created = []
        self.proposed = []
        self.scheduled = []

    def create_draft(self, content, platforms=None, metadata=None):
        draft = type("QD", (), {"id": "qd_fake"})()
        self.created.append((content, list(platforms or []), metadata))
        return draft

    def propose(self, draft_id):
        self.proposed.append(draft_id)

    def schedule(self, draft_id, when_ts):
        self.scheduled.append((draft_id, when_ts))


class _FakeScheduler:
    """Duck-typed stand-in for the scheduler: records, never raises."""

    def __init__(self):
        self.actions = []
        self.once = []

    def register_action(self, name, fn):
        self.actions.append(name)

    def schedule_once(self, task_id, when_ts, action, params):
        self.once.append((task_id, when_ts, action, params))
        return None


def test_brand_voice_is_real_virality_score_not_fallback():
    """(a) score_draft returns the live virality_score, not 50.0."""
    store = BriefStore(db_path=_db())
    brief = ContentBrief(brief_id="b1", topic="Lagos rents",
                         keywords=["lagos", "rent", "yaba"])
    result = store.score_draft(HOOK_TEXT, brief, platform="x")
    expected = round(virality_score(HOOK_TEXT, platform="x").score, 1)
    assert expected != 50.0  # the pin text must stay off the fallback value
    assert result.brand_voice == expected
    assert "briefs.BriefStore.score_draft.virality" not in _sites()


def test_schedule_post_runs_through_briefs_path():
    """(b) the real drafts.schedule_post executes via schedule_draft."""
    store = BriefStore(db_path=_db())
    brief = store.build_brief("Lagos rents", search_fn=_mock_search)
    draft = store.draft_from_brief(brief)
    assert draft is not None
    queue = _FakeQueue()
    scheduler = _FakeScheduler()
    when_ts = 9999999999.0
    res = store.schedule_draft(draft.draft_id, when_ts,
                               scheduler=scheduler, queue=queue)
    assert res["ok"] is True
    # The real schedule_post ran: queue step, action registration, timed task.
    assert queue.scheduled == [(draft.draft_id, when_ts)]
    assert scheduler.actions == ["social.post_draft"]
    assert scheduler.once == [
        (f"post-draft-{draft.draft_id}", when_ts, "social.post_draft",
         {"draft_id": draft.draft_id, "platforms": [draft.platform]})]
    assert store.get_draft(draft.draft_id).scheduled_for == when_ts
    assert "briefs.BriefStore.schedule_draft.scheduler" not in _sites()


def test_old_importerror_path_is_gone():
    """(c) exercising both paths records no ImportError anywhere."""
    store = BriefStore(db_path=_db())
    brief = ContentBrief(brief_id="b1", topic="Lagos rents",
                         keywords=["lagos", "rent", "yaba"])
    store.score_draft(HOOK_TEXT, brief, platform="x")
    brief2 = store.build_brief("Lagos rents", search_fn=_mock_search)
    draft = store.draft_from_brief(brief2)
    store.schedule_draft(draft.draft_id, 9999999999.0,
                         scheduler=_FakeScheduler(), queue=_FakeQueue())
    for failure in mk.recent_failures(limit=50):
        assert failure.exc_type != "ImportError", failure
        assert "beyond top-level package" not in failure.message
        assert "cannot import name 'schedule_post'" not in failure.message
