"""R14: automatic training runs notify the owner by DM, not just the log."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.cognition import CognitiveLoop
from nomorals.agents.context import build_context
from nomorals.storage.db import Database


def _ctx():
    return build_context(db=Database(":memory:"), with_tools=False,
                         with_router=False, with_memory=False)


def _due_job(run_id="run-1"):
    job = mock.Mock()
    job.status.return_value = {
        "decision": {
            "data_available": True,
            "should_retrain": True,
            "reasons": ["500 new examples since last run"],
        },
    }
    job.run.return_value = SimpleNamespace(
        status="ok", run_id=run_id, dataset_id="ds-1", promoted=False,
        skipped="", reason="",
    )
    return job


def _not_due_job():
    job = mock.Mock()
    job.status.return_value = {
        "decision": {
            "data_available": True,
            "should_retrain": False,
            "reasons": ["policy not due"],
        },
    }
    return job


def _training_notices(db):
    return db.query(
        "SELECT title, body, delivery_state FROM notifications "
        "WHERE kind = 'training' ORDER BY created_at DESC")


class TrainingNoticeTests(unittest.TestCase):
    def test_automatic_run_sends_owner_dm(self):
        ctx = _ctx()
        loop = CognitiveLoop(ctx)
        with mock.patch("nomorals.self_improvement.SelfImprovementJob",
                        return_value=_due_job()):
            result = loop._tick_train()
        self.assertEqual(result["status"], "ok")
        notices = _training_notices(ctx.db)
        self.assertEqual(len(notices), 1)
        self.assertIn("automatic training run", notices[0]["title"])
        self.assertIn("500 new examples", notices[0]["body"])

    def test_no_notice_when_not_due(self):
        ctx = _ctx()
        loop = CognitiveLoop(ctx)
        with mock.patch("nomorals.self_improvement.SelfImprovementJob",
                        return_value=_not_due_job()):
            result = loop._tick_train()
        self.assertEqual(result["skipped"], "policy not due")
        self.assertEqual(_training_notices(ctx.db), [])

    def test_no_notice_when_no_data(self):
        ctx = _ctx()
        job = mock.Mock()
        job.status.return_value = {
            "decision": {"data_available": False, "should_retrain": False,
                         "reasons": []},
        }
        loop = CognitiveLoop(ctx)
        with mock.patch("nomorals.self_improvement.SelfImprovementJob",
                        return_value=job):
            result = loop._tick_train()
        self.assertEqual(result["skipped"], "no training data")
        self.assertEqual(_training_notices(ctx.db), [])


if __name__ == "__main__":
    unittest.main()
