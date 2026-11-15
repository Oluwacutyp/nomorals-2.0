"""Pass #217 pins: agents/idea_development.py — 1 exclusion, 0 converts.

- 507: ``IdeaDevelopment.develop`` failure-report refusal. When a
  development run breaks mid-run, the record is marked failed, the
  failure detail is written into the record, and the record is
  SAVED — then the failure is also reported to the delegation store
  (``store.report_failure``). If the delegation store refuses that
  report (DelegationError — the contract is fenced or already
  closed), there is nothing more to do: the failure already lives
  in the saved development record, which is the primary store. The
  swallow is the correct behavior: the record carries the failure
  either way, and the function returns the failed record on a
  non-error path.

agents/idea_development.py calls no failures-ledger record function
anywhere, so NOT-recording is pinned with the agency snapshot
method: agency recent_failures() before/after each trigger must be
unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use a real Database and a
patched run/report, never a swallow.
"""

from nomorals.agents import failures as ag_fail
from nomorals.agents.delegation import DelegationError
from nomorals.agents.idea_development import IdeaDevelopment
from nomorals.goals import GoalTracker, IdeaTracker
from nomorals.goals.surface import run_sync
from nomorals.storage.db import Database

UID = "owner"


def _snap():
    return list(ag_fail.recent_failures(limit=200))


def _world():
    db = Database(":memory:")
    db.migrate()
    return db, IdeaDevelopment(db), IdeaTracker(db), GoalTracker(db)


def _idea(tracker):
    return run_sync(tracker.create(UID, "A testable idea",
                                   description="grows or fails"))


# ── L507: a refused failure report does not lose the record ─────────

def test_507_refused_failure_report_still_returns_failed_record(monkeypatch):
    _db, cell, tracker, _goals = _world()
    idea = _idea(tracker)

    def _boom_steps(*args, **kwargs):
        raise RuntimeError("mid-run boom")

    def _refuse_report(*args, **kwargs):
        raise DelegationError("contract fenced")

    monkeypatch.setattr(cell, "_run_steps", _boom_steps)
    monkeypatch.setattr(cell.store, "report_failure", _refuse_report)
    before = _snap()
    record = cell.develop(idea.idea_id)
    assert record["status"] == "failed"
    assert "development broke mid-run" in record["outcome"]["detail"]
    assert "RuntimeError" in record["outcome"]["detail"]
    # The record was saved with the failure before the report was
    # even attempted — the failure is not lost.
    saved = cell.record_for(record["dev_id"])
    assert saved is not None
    assert saved["status"] == "failed"
    assert saved["outcome"]["detail"] == record["outcome"]["detail"]
    assert _snap() == before


def test_507_failure_report_success_still_returns_failed_record(monkeypatch):
    # Control: when the delegation store ACCEPTS the report, the
    # same failed record comes back — the report is a notification,
    # not the record itself.
    _db, cell, tracker, _goals = _world()
    idea = _idea(tracker)
    reported = []

    def _boom_steps(*args, **kwargs):
        raise RuntimeError("mid-run boom")

    def _record_report(cid, **kwargs):
        reported.append((cid, kwargs.get("detail", "")))

    monkeypatch.setattr(cell, "_run_steps", _boom_steps)
    monkeypatch.setattr(cell.store, "report_failure", _record_report)
    record = cell.develop(idea.idea_id)
    assert record["status"] == "failed"
    assert len(reported) == 1
    assert reported[0][0] == record["contract_id"]
    assert "development broke mid-run" in reported[0][1]
