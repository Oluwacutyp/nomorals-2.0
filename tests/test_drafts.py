"""Agent-drafts with explicit-override rule (build-map #43).

The user's directive: explicit "post this"/"send it"/"post at 6" →
execute immediately, no reconfirmation. Default → ask first.
"""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from nomorals.social.drafts import (
    DraftQueue,
    detect_explicit_post,
    execute_post,
    handle_draft_callback,
    parse_post_time,
    propose_post,
    schedule_post,
    DRAFT,
    PENDING_REVIEW,
    APPROVED,
    SCHEDULED,
    POSTED,
    DISCARDED,
)


@pytest.fixture()
def queue(tmp_path):
    q = DraftQueue(db_path=tmp_path / "drafts.db")
    yield q
    q.close()


# ── intent detection ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "post this",
    "Post This",
    "post it",
    "post that",
    "post now",
    "publish now",
    "send it",
    "reply now",
    "submit",
    "post it for engagement",
    "post for engagement",
])
def test_explicit_post_phrasings(text):
    got = detect_explicit_post(text)
    assert got is not None and got["action"] == "post", text


def test_schedule_intent_with_time():
    got = detect_explicit_post("post at 6 for engagement")
    assert got is not None and got["action"] == "schedule"
    assert got["at"] is not None


def test_schedule_this():
    assert detect_explicit_post("schedule this")["action"] == "schedule"


@pytest.mark.parametrize("text", [
    "read that post",
    "poster on the wall",
    "what do you think",
    "imposter syndrome",
    "",
    "   ",
])
def test_no_false_positives(text):
    assert detect_explicit_post(text) is None, text


def test_parse_post_time_bare_six():
    morning = datetime(2026, 10, 8, 9, 0)
    ts = parse_post_time("post at 6", now=morning)
    assert datetime.fromtimestamp(ts).strftime("%H:%M") == "18:00"
    evening = datetime(2026, 10, 8, 20, 0)
    ts2 = parse_post_time("post at 6", now=evening)
    dt = datetime.fromtimestamp(ts2)
    assert (dt.day, dt.strftime("%H:%M")) == (9, "06:00")


def test_parse_post_time_explicit_ampm():
    now = datetime(2026, 10, 8, 9, 0)
    assert datetime.fromtimestamp(
        parse_post_time("post at 6pm", now=now)).strftime("%H:%M") == "18:00"
    assert datetime.fromtimestamp(
        parse_post_time("post at 6:30am", now=now)).strftime("%H:%M") == "06:30"


def test_parse_post_time_none():
    assert parse_post_time("just post this") is None


# ── draft lifecycle ───────────────────────────────────────────────────────

def test_draft_lifecycle(queue):
    d = queue.create_draft("hello world", ["x", "threads"])
    assert d.status == DRAFT
    d = queue.propose(d.id)
    assert d.status == PENDING_REVIEW
    d = queue.approve(d.id)
    assert d.status == APPROVED
    d = queue.mark_posted(d.id)
    assert d.status == POSTED and d.posted_at is not None


def test_draft_schedule_flow(queue):
    d = queue.create_draft("timed post", ["x"])
    queue.propose(d.id)
    d = queue.schedule(d.id, 1791482400.0)
    assert d.status == SCHEDULED and d.scheduled_time == 1791482400.0


def test_draft_discard(queue):
    d = queue.create_draft("nope", ["x"])
    assert queue.discard(d.id).status == DISCARDED


def test_invalid_transition_rejected(queue):
    d = queue.create_draft("hello", ["x"])
    with pytest.raises(ValueError):
        queue.approve(d.id)  # draft → approved skips review


def test_empty_draft_rejected(queue):
    with pytest.raises(ValueError):
        queue.create_draft("   ", ["x"])


def test_unknown_draft(queue):
    assert queue.get("nope") is None
    with pytest.raises(KeyError):
        queue.approve("nope")


def test_pending_lists_open_drafts(queue):
    queue.create_draft("one", ["x"])
    d2 = queue.create_draft("two", ["x"])
    queue.discard(d2.id)
    assert len(queue.pending()) == 1


# ── review UI ─────────────────────────────────────────────────────────────

def test_propose_post_message(queue):
    d = queue.create_draft("draft content here", ["x"])
    text, buttons = propose_post(d)
    assert "Post directly or send for review?" in text
    assert "draft content here" in text
    labels = [b[0] for b in buttons]
    assert labels == ["📮 Post now", "⏰ Schedule", "✏️ Edit", "🗑 Discard"]
    assert all(b[1].startswith("draft:") and d.id in b[1] for b in buttons)


# ── the override: execute without reconfirming ────────────────────────────

def _manager():
    outcome = MagicMock()
    outcome.results = [MagicMock()]
    outcome.posted = [MagicMock()]
    mgr = MagicMock()
    mgr.publish.return_value = outcome
    policy = MagicMock()
    policy.mint_confirmation.return_value = "tok123"
    mgr.context.policy = policy
    return mgr, outcome


def test_execute_post_bypasses_review(queue):
    """Explicit intent → publish called directly, no confirmation step."""
    mgr, outcome = _manager()
    d = queue.create_draft("breaking news", ["x"])
    result = execute_post(mgr, d.content, platforms=["x"],
                          draft_id=d.id, queue=queue)
    assert result is outcome
    # publish was called exactly once, directly — no intermediate ask
    mgr.publish.assert_called_once()
    kwargs = mgr.publish.call_args.kwargs
    assert kwargs["confirmation"] == "tok123"  # real policy token minted
    assert kwargs["actor"] == "user"
    # draft marked posted
    assert queue.get(d.id).status == POSTED


def test_execute_post_without_policy_still_posts(queue):
    mgr = MagicMock()
    outcome = MagicMock()
    outcome.results = []
    outcome.posted = []
    mgr.publish.return_value = outcome
    mgr.context.policy = None
    execute_post(mgr, "hello", platforms=["x"])
    mgr.publish.assert_called_once()
    assert mgr.publish.call_args.kwargs["confirmation"] == ""


def test_no_explicit_intent_no_approval_never_posts(queue):
    """Default path: without explicit intent or approval, nothing posts."""
    mgr, _ = _manager()
    d = queue.create_draft("quiet draft", ["x"])
    queue.propose(d.id)  # asked, not approved
    mgr.publish.assert_not_called()
    assert queue.get(d.id).status == PENDING_REVIEW


def test_schedule_post_registers_action(queue):
    sched = MagicMock()
    d = queue.create_draft("timed", ["x"])
    queue.propose(d.id)
    schedule_post(sched, queue, d.id, 1791482400.0, platforms=["x"])
    sched.register_action.assert_called_once()
    assert sched.register_action.call_args.args[0] == "social.post_draft"
    assert queue.get(d.id).status == SCHEDULED


# ── callback routing ──────────────────────────────────────────────────────

def test_callback_post(queue):
    mgr, _ = _manager()
    d = queue.create_draft("cb post", ["x"])
    queue.propose(d.id)
    reply = handle_draft_callback(queue, f"draft:post:{d.id}", manager=mgr)
    assert "posted" in reply
    assert queue.get(d.id).status == POSTED
    mgr.publish.assert_called_once()


def test_callback_discard(queue):
    d = queue.create_draft("cb discard", ["x"])
    queue.propose(d.id)
    reply = handle_draft_callback(queue, f"draft:discard:{d.id}")
    assert "discarded" in reply
    assert queue.get(d.id).status == DISCARDED


def test_callback_schedule_asks_time(queue):
    d = queue.create_draft("cb schedule", ["x"])
    queue.propose(d.id)
    reply = handle_draft_callback(queue, f"draft:schedule:{d.id}")
    assert "when should I post it" in reply  # no time fn → asks


def test_callback_unknown_draft(queue):
    assert "gone" in handle_draft_callback(queue, "draft:post:nope")


def test_callback_bad_data(queue):
    assert handle_draft_callback(queue, "garbage") == ""
