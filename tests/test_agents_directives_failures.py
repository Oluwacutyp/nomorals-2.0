"""Tail-64 pins for nomorals/agents/directives.py —
0 converts / 6 exclusions. Each test pins one site with a
control half proving the guarded operation was actually
ATTEMPTED via the double's own call log. The file contains
no exception handlers, so it adds zero census sites.

directives.py calls no failures-ledger record function
anywhere (grep-proven), so NOT-recording is pinned against
the agents package ledger (nomorals/agents/failures.py):
recent_failures() snapshot before/after every trigger.

Verdict notes (full grounds in the tail-64 SURVEY.md):
- 45: _as_brain's TypeError swallow IS the documented
  contract ("isinstance against a non-type raises TypeError,
  which we treat as 'not a Brain'") -> wrap in a Brain.
- 87 / 114 / 129: read paths degrade to the module's own
  no-DB contract values (db is None -> _get None, list [],
  run() 'no pending'); writes (add) fail loudly instead.
- 99: _set is void best-effort status journaling; db is
  None -> no-op by the same contract.
- 291: _notify is best-effort publication; run()'s own
  result dict is already decided before it is called.
"""

from types import SimpleNamespace
from unittest import mock

from nomorals.agents import failures as agent_fail
from nomorals.agents.directives import DirectivesAgent, _as_brain


def _snap():
    return list(agent_fail.recent_failures(200))


class Tx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeDB:
    def __init__(self, fail_on=""):
        self.fail_on = fail_on
        self.calls = []

    def transaction(self):
        return Tx()

    def execute(self, *a):
        self.calls.append("execute")
        if self.fail_on == "execute":
            raise RuntimeError("db execute dead")

    def query_one(self, *a):
        self.calls.append("query_one")
        if self.fail_on == "query_one":
            raise RuntimeError("db read dead")
        return None

    def query(self, *a):
        self.calls.append("query")
        if self.fail_on == "query":
            raise RuntimeError("db read dead")
        return []


def _agent(db):
    return DirectivesAgent(SimpleNamespace(db=db, router=None))


def test_45_as_brain_non_type_brain_wraps():
    before = _snap()
    calls = []

    def fake_brain(*, router=None):
        calls.append(router)
        return ("wrapped", router)

    with mock.patch("nomorals.llm.brain.Brain", fake_brain):
        out = _as_brain("ROUTER")
    assert out == ("wrapped", "ROUTER")
    assert calls == ["ROUTER"]
    assert _snap() == before


def test_87_get_db_failure_returns_none():
    before = _snap()
    db = FakeDB("query_one")
    assert _agent(db)._get("d1") is None
    assert db.calls == ["query_one"]
    assert _snap() == before


def test_99_set_db_failure_is_silent():
    before = _snap()
    db = FakeDB("execute")
    assert _agent(db)._set("d1", "done", result="r") is None
    assert db.calls == ["execute"]
    assert _snap() == before


def test_114_list_db_failure_returns_empty():
    before = _snap()
    db = FakeDB("query")
    assert _agent(db).list() == []
    assert db.calls == ["query"]
    assert _snap() == before


def test_129_run_pending_query_failure_reports_no_pending():
    before = _snap()
    db = FakeDB("query_one")
    out = _agent(db).run("")
    assert out["ok"] is False and "no pending" in out["error"]
    assert db.calls == ["query_one"]
    assert _snap() == before


def test_291_notify_publish_failure_is_silent():
    before = _snap()

    class BadNotifier:
        def __init__(self):
            self.calls = 0

        def publish(self, *a, **k):
            self.calls += 1
            raise RuntimeError("publish dead")

    notifier = BadNotifier()
    agent = DirectivesAgent(SimpleNamespace(db=None, router=None),
                            notifier=notifier)
    assert agent._notify("t", "b") is None
    assert notifier.calls == 1
    assert _snap() == before
