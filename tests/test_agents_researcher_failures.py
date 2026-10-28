"""Tail-23 pins: researcher.py failure values + NOT-recording.

Verdict: 2 converts / 7 exclusions (parent-amended: site 465 converted
by the parent — see the 465 pin). Site 638 (approve/deny reported
"approved"/"denied" when the status write failed) is a convert: the
owner-facing string is now the failure string and the apply is not
started. Site 465 (research_log INSERT failure still notified) is the
second convert: the returned status is now "notified-unlogged" instead
of "notified" (the notification still publishes — no owner value
discarded — but the row is lost, so /research ideas never shows it
and approve-by-id cannot find it); the auto-apply gate also skips
row-less applies, 638 precedent. The no-DB configuration is unchanged
("notified" stays: no journal was attempted, none failed).

Every other handler is documented never-raises (or best-effort); each
pin drives the handler into its except path and pins the exact fallback
value plus the fact that the agents failures ledger is unchanged
(snapshot before/after, tail-8 method). researcher.py calls no record
function anywhere (grep-proven), so the ledger must not move.

The pins file itself contains no except handlers, so it adds zero
census sites.
"""

import threading
import time
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import failures as ag_fail
from nomorals.agents import researcher as R


def _ledger():
    return list(ag_fail.recent_failures())


class _FakeDB:
    """Configurable stand-in for the research_log DB."""

    def __init__(self, query_rows=None, query_one_row=None,
                 query_raises=False, query_one_raises=False,
                 execute_raises=False):
        self.query_rows = query_rows
        self.query_one_row = query_one_row
        self.query_raises = query_raises
        self.query_one_raises = query_one_raises
        self.execute_raises = execute_raises
        self.executed = []

    def query(self, sql, params=()):
        if self.query_raises:
            raise RuntimeError("db down (query)")
        return list(self.query_rows or [])

    def query_one(self, sql, params=()):
        if self.query_one_raises:
            raise RuntimeError("db down (query_one)")
        return self.query_one_row

    def execute(self, sql, params=()):
        self.executed.append(sql)
        if self.execute_raises:
            raise RuntimeError("db down (execute)")

    def transaction(self):
        db = self

        class _Tx:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        return _Tx()


def _ctx(db=None, settings=None):
    return SimpleNamespace(db=db, router=None, settings=settings,
                           memory=None, extras={})


def test_used_topics_db_failure_returns_empty():
    """376: query failure -> empty used set (control: rows -> topics)."""
    before = _ledger()
    agent = R.ResearchAgent(_ctx(db=_FakeDB(query_raises=True)))
    assert agent._used_topics("tech") == set()
    agent = R.ResearchAgent(_ctx(db=_FakeDB(
        query_rows=[{"topic": "alpha"}, {"topic": "beta"}])))
    assert agent._used_topics("tech") == {"alpha", "beta"}
    assert _ledger() == before


def test_model_suggestion_brain_failure_returns_empty():
    """392: brain failure -> "" (control: brain text -> stripped text)."""
    before = _ledger()
    agent = R.ResearchAgent(_ctx())
    agent.context.router = object()  # pass the router-None early return
    with mock.patch.object(R, "brain_for",
                           side_effect=RuntimeError("brain down")):
        assert agent._model_suggestion("p") == ""

    class _Resp:
        text = "  do the thing  "
        error = None

    class _Brain:
        def chat(self, msgs, task_kind=None):
            return _Resp()

    with mock.patch.object(R, "brain_for", return_value=_Brain()):
        assert agent._model_suggestion("p") == "do the thing"
    assert _ledger() == before


def test_reasoning_critique_failure_returns_empty():
    """514: critique is a bonus, never a gate -> "" on failure.

    Control: a real weakness verdict passes through; reasoning-off
    also yields "" (the documented no-critique value).
    """
    before = _ledger()
    import nomorals.agents.reasoning as reasoning_mod
    agent = R.ResearchAgent(_ctx())

    class _Raising:
        def __init__(self, context):
            pass

        def advise(self, *args, **kwargs):
            raise RuntimeError("reasoning down")

    with mock.patch.object(reasoning_mod, "reasoning_permitted",
                           return_value=True), \
            mock.patch.object(reasoning_mod, "ReasoningAgent", _Raising):
        assert agent._reasoning_critique("t", "s") == ""

    class _Weak:
        def __init__(self, context):
            pass

        def advise(self, *args, **kwargs):
            return "too vague to act on"

    with mock.patch.object(reasoning_mod, "reasoning_permitted",
                           return_value=True), \
            mock.patch.object(reasoning_mod, "ReasoningAgent", _Weak):
        assert agent._reasoning_critique("t", "s") == "too vague to act on"
    with mock.patch.object(reasoning_mod, "reasoning_permitted",
                           return_value=False):
        assert agent._reasoning_critique("t", "s") == ""
    assert _ledger() == before


def test_recent_topics_db_failure_returns_empty():
    """541: query failure -> [] (control: rows -> topics).

    Caveat recorded in SURVEY.md: [] is also the documented no-history
    value, so a down DB weakens the anti-repeat gate (novelty scores
    1.0). The suggestion itself stays real; a repeat notification is a
    quality wart, not a false statement — conservative-novelty redesign
    is a policy follow-up, not a lie fix.
    """
    before = _ledger()
    agent = R.ResearchAgent(_ctx(db=_FakeDB(query_raises=True)))
    assert agent._recent_topics() == []
    agent = R.ResearchAgent(_ctx(db=_FakeDB(
        query_rows=[{"topic": "old1"}, {"topic": "old2"}])))
    assert agent._recent_topics() == ["old1", "old2"]
    assert _ledger() == before


def test_approve_deny_write_failure_reported_not_confirmed():
    """638 CONVERT: failed status write -> failure string, no apply.

    Control: a working write still yields "approved"/"denied".
    """
    before = _ledger()
    row = {"id": "abc123", "status": "pending", "domain": "lifestyle",
           "topic": "test topic", "suggestion": "do x", "digest": "d",
           "score": 0.9}
    agent = R.ResearchAgent(_ctx(db=_FakeDB(query_one_row=row,
                                            execute_raises=True)))
    assert agent.approve("abc123") == \
        "couldn't update proposal status — try again"
    assert agent.deny("abc123") == \
        "couldn't update proposal status — try again"
    agent = R.ResearchAgent(_ctx(db=_FakeDB(query_one_row=row)))
    assert agent.approve("abc123").startswith("approved")
    agent = R.ResearchAgent(_ctx(db=_FakeDB(query_one_row=row)))
    assert agent.deny("abc123").startswith("denied")
    assert _ledger() == before


def test_background_apply_journal_failure_stays_quiet():
    """699: the final journal update failing loses the record, but the
    background thread must not die loudly — no thread exception, and
    both updates were attempted. Control: a working DB attempts both
    updates too. The stuck-"applying" status understates, never
    overstates: the owner never sees a success that did not happen.
    """
    before = _ledger()
    import nomorals.agents.coding as coding_mod

    class _OkAgent:
        def __init__(self, context):
            pass

        def run(self, *args, **kwargs):
            return SimpleNamespace(ok=True, output="done", error="")

    thread_errors = []
    old_hook = threading.excepthook
    threading.excepthook = lambda args: thread_errors.append(args)
    try:
        with mock.patch.object(coding_mod, "CodingAgent", _OkAgent):
            db = _FakeDB(execute_raises=True)
            agent = R.ResearchAgent(_ctx(db=db))
            agent._apply_in_background("rid12345", "task", "f.py")
            time.sleep(1.0)
            assert thread_errors == []
            assert len(db.executed) == 2
            db2 = _FakeDB()
            agent = R.ResearchAgent(_ctx(db=db2))
            agent._apply_in_background("rid12345", "task", "f.py")
            time.sleep(1.0)
            assert thread_errors == []
            assert len(db2.executed) == 2
    finally:
        threading.excepthook = old_hook
    assert _ledger() == before


def test_under_cap_db_failure_returns_true():
    """717: query failure -> True, the same value as the no-DB branch
    (cap > 0). Fail-open availability tradeoff, tail-17 site 190 shape:
    denying every notification on a transient backend error would be
    worse. Control: over cap -> False, under cap -> True.
    """
    before = _ledger()
    settings = SimpleNamespace(
        research=SimpleNamespace(daily_suggestion_cap=3))
    agent = R.ResearchAgent(_ctx(db=_FakeDB(query_one_raises=True),
                                 settings=settings))
    assert agent._under_cap() is True
    agent = R.ResearchAgent(_ctx(db=_FakeDB(query_one_row={"n": 5}),
                                 settings=settings))
    assert agent._under_cap() is False
    agent = R.ResearchAgent(_ctx(db=_FakeDB(query_one_row={"n": 1}),
                                 settings=settings))
    assert agent._under_cap() is True
    assert _ledger() == before


def test_loop_survives_raising_cycle():
    """732: the loop must never die — a raising cycle is caught and the
    thread stays alive for the next iteration. Control: a succeeding
    cycle also leaves the thread alive."""
    before = _ledger()
    import nomorals.agents.features as features_mod
    settings = SimpleNamespace(research=SimpleNamespace(
        enabled=True, interval_hours=12.0))
    with mock.patch.object(features_mod, "feature_enabled",
                           return_value=True):
        agent = R.ResearchAgent(_ctx(settings=settings))
        calls = []

        def _boom(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("cycle boom")

        agent.run_cycle = _boom
        assert agent.start_loop() is True
        time.sleep(0.7)
        assert agent._thread.is_alive()
        assert len(calls) >= 1
        agent.stop_loop()
        agent._thread.join(timeout=5)
        assert not agent._thread.is_alive()

        agent2 = R.ResearchAgent(_ctx(settings=settings))
        calls2 = []

        def _ok(*args, **kwargs):
            calls2.append(1)
            return {"ok": True}

        agent2.run_cycle = _ok
        assert agent2.start_loop() is True
        time.sleep(0.7)
        assert agent2._thread.is_alive()
        assert len(calls2) >= 1
        agent2.stop_loop()
        agent2._thread.join(timeout=5)
    assert _ledger() == before


def test_notify_journal_failure_reports_unlogged_not_notified():
    """465 CONVERT (parent): INSERT failure -> "notified-unlogged",
    notification still published (value kept), no row stored.

    Control: a working INSERT still yields "notified". The auto-apply
    gate (no apply without a journal row, 638 precedent) follows from
    the same `journaled` flag; the score here (0.748) sits below the
    0.8 auto-apply threshold, so the gate itself is code-read, not
    pin-proven — the pin guards the status honesty, which is the
    convert.
    """
    before = _ledger()
    import nomorals.agents.reasoning as reasoning_mod
    digest = ("Deploy 3 replicas for the cache tier today "
              + ("facts 42 and 99, " * 40))
    report = {"summary": digest, "pages_read": [1, 2, 3],
              "results": [{"url": "https://a.example/x"},
                          {"url": "https://b.example/y"},
                          {"url": "https://c.example/z"}]}

    class _FakeEngine:
        def __init__(self, context):
            pass

        def run(self, *args, **kwargs):
            return report

    class _Notifier:
        def __init__(self):
            self.published = []

        def publish(self, *args, **kwargs):
            self.published.append(args)
            return {"ok": True}

    with mock.patch.object(R, "SearchEngine", _FakeEngine), \
            mock.patch.object(reasoning_mod, "reasoning_permitted",
                              return_value=False):
        db = _FakeDB(query_raises=True, query_one_raises=True,
                     execute_raises=True)
        notifier = _Notifier()
        agent = R.ResearchAgent(_ctx(db=db), notifier=notifier)
        result = agent.run_cycle("tech")
        assert result["status"] == "notified-unlogged"
        assert len(notifier.published) == 1
        db2 = _FakeDB()
        notifier2 = _Notifier()
        agent2 = R.ResearchAgent(_ctx(db=db2), notifier=notifier2)
        result2 = agent2.run_cycle("tech")
        assert result2["status"] == "notified"
        assert len(notifier2.published) == 1
    assert _ledger() == before
