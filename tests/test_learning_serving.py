"""Wave J: trajectory learning in the serving path — fully offline, tmp SQLite."""

import time

import pytest

from nomorals.cognition.trajectories import TrajectoryStore
from nomorals.llm.base import Message
from nomorals.llm.broker import ModelBroker, NoCandidate
from nomorals.llm.learning import LearningAttachment, attach_learning
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.router import LLMRouter


def _router() -> LLMRouter:
    """Two-provider router: ``a`` always answers, ``b`` always fails."""
    router = LLMRouter()
    router.add(MockProvider(model="mock-a"), primary=True, name="a")
    router.add(MockProvider(model="mock-b", failure_rate=1.0), name="b")
    return router


def _chat(router: LLMRouter):
    return router.chat([Message.user("hello")])


def test_attach_builds_broker_from_router(tmp_path):
    router = _router()
    assert router.broker is None
    att = attach_learning(router, db_path=tmp_path / "learn.db")
    try:
        assert not att.degraded
        broker = router.broker
        assert broker is not None
        # Cards mirror the router's providers; no operator override is set —
        # recorded history, not a pin, must decide selection.
        assert {c.id for c in broker.cards()} == {"a", "b"}
        assert broker.primary_model_id == ""
        resp = _chat(router)
        assert resp.ok
        assert router.active == "a"
    finally:
        att.detach()


def test_ranking_flips_after_recorded_history(tmp_path):
    router = _router()
    att = attach_learning(router, db_path=tmp_path / "learn.db")
    try:
        broker = router.broker
        # Take broker selection out of the loop while recording: otherwise
        # consult() would pre-empt the failover we are trying to observe.
        router.set_broker(None)
        _chat(router)  # served by a -> (a, success)
        router.set_active("b")
        resp = _chat(router)  # b fails, failover serves via a
        assert resp.ok and resp.degraded  # failover happened
        assert att.flush()
        router.set_broker(broker)  # selection back on
        ranked = [card.id for card, _ in broker.ranked("chat")]
        assert ranked[0] == "a", ranked
        assert broker.select("chat").id == "a"
        # The store itself agrees: a ~1.0, b ~0.0.
        assert att.trajectories.success_rate("chat", "chat", "a") == pytest.approx(1.0)
        assert att.trajectories.success_rate("chat", "chat", "b") == pytest.approx(0.0)
    finally:
        att.detach()


def test_require_raises_with_no_data():
    with pytest.raises(NoCandidate):
        ModelBroker().require("chat")


def test_corrupt_db_degrades_router_works_fine(tmp_path):
    bad = tmp_path / "junk.db"
    bad.write_bytes(b"this is not sqlite\x00\x01\x02")
    router = _router()
    att = attach_learning(router, db_path=bad)
    try:
        assert att.degraded
        assert att.degraded_reason
        # No broker was conjured from nothing, no hook installed.
        assert router.broker is None
        assert router.learning is None
        resp = _chat(router)
        assert resp.ok
        assert router.active == "a"
    finally:
        att.detach()


def test_live_benchmark_rows_recorded(tmp_path):
    router = _router()
    att = attach_learning(router, db_path=tmp_path / "learn.db")
    try:
        _chat(router)
        assert att.flush()
        rows = att.benchmarks.samples("a", "chat")
        assert rows, "expected live benchmark rows for provider a"
        assert all(r["source"] == "live" for r in rows)
        summary = att.benchmarks.summary("a", "chat")
        assert summary["sources"] == ["live"]
        assert summary["samples"] >= 1
        assert 0.0 <= summary["score"] <= 1.0
    finally:
        att.detach()


def test_attach_is_idempotent(tmp_path):
    router = _router()
    first = attach_learning(router, db_path=tmp_path / "learn.db")
    try:
        second = attach_learning(router, db_path=tmp_path / "learn.db")
        assert second is first
    finally:
        first.detach()


def test_routing_unchanged_by_attach_detach_cycle(tmp_path):
    expected = _chat(_router()).text
    router = _router()
    att = attach_learning(router, db_path=tmp_path / "learn.db")
    try:
        assert _chat(router).text == expected
        assert router.active == "a"  # broker consult did not move the primary
    finally:
        att.detach()
    assert router.learning is None
    assert _chat(router).text == expected


def test_existing_broker_gets_trajectories_not_clobbered(tmp_path):
    router = _router()
    broker = ModelBroker()
    broker.build_from_router(router)
    router.set_broker(broker)
    att = attach_learning(router, db_path=tmp_path / "learn.db")
    try:
        assert att.broker is broker
        assert broker.trajectories is att.trajectories
    finally:
        att.detach()

    # An operator-supplied store is never replaced.
    router2 = _router()
    broker2 = ModelBroker()
    sentinel = TrajectoryStore()
    broker2.trajectories = sentinel
    router2.set_broker(broker2)
    att2 = attach_learning(router2, db_path=tmp_path / "learn2.db")
    try:
        assert broker2.trajectories is sentinel
    finally:
        att2.detach()


def test_recording_never_blocks_dispatch(tmp_path):
    class SlowStore(TrajectoryStore):
        def record(self, **kwargs):
            time.sleep(2.0)

    router = _router()
    att = attach_learning(router, db_path=tmp_path / "learn.db")
    # Swap in a pathologically slow store *after* attach: the dispatch path
    # must still return immediately because writes happen off-thread.
    att.trajectories = SlowStore(str(tmp_path / "slow.db"))
    try:
        started = time.monotonic()
        resp = _chat(router)
        elapsed = time.monotonic() - started
        assert resp.ok
        assert elapsed < 1.0, f"dispatch blocked on learning for {elapsed:.2f}s"
    finally:
        att.detach()


def test_detach_is_idempotent_and_clears_marker(tmp_path):
    router = _router()
    att = attach_learning(router, db_path=tmp_path / "learn.db")
    assert isinstance(att, LearningAttachment)
    att.detach()
    att.detach()  # must not raise
    assert router.learning is None
    assert getattr(router, "_learning_attachment", None) is None
