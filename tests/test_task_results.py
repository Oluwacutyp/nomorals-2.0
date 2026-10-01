"""Done semantics: completion is not correctness."""

import unittest

from nomorals.core.tasks import (
    AcceptanceCriterion,
    Task,
    TaskKind,
    TaskResult,
    TaskState,
)


def result_with(**kw):
    base = dict(
        status="done",
        artifacts=["artifact://abc123"],
        evidence={"approved_by": "owner"},
        assertions=[{"name": "schema-valid", "passed": True, "detail": "ok"}],
        tests={"unit": {"passed": True, "detail": "42 passed"}},
        metrics={"accuracy": 0.97, "latency_ms": 120},
    )
    base.update(kw)
    return TaskResult(**base)


class TestTaskResultVerify(unittest.TestCase):
    def test_metric_gte(self):
        r = result_with()
        out = r.verify([AcceptanceCriterion(name="acc", spec={"metric": "accuracy", "gte": 0.9})])
        self.assertTrue(out[0].passed)
        self.assertTrue(r.passed)

    def test_metric_gte_fails(self):
        r = result_with()
        out = r.verify([AcceptanceCriterion(name="acc", spec={"metric": "accuracy", "gte": 0.99})])
        self.assertFalse(out[0].passed)
        self.assertFalse(r.passed)

    def test_metric_missing_fails(self):
        r = result_with()
        out = r.verify([AcceptanceCriterion(name="x", spec={"metric": "nope", "gte": 1})])
        self.assertFalse(out[0].passed)

    def test_artifact_present(self):
        r = result_with()
        out = r.verify([AcceptanceCriterion(name="art", spec={"artifact": True})])
        self.assertTrue(out[0].passed)
        r2 = result_with(artifacts=[])
        out2 = r2.verify([AcceptanceCriterion(name="art", spec={"artifact": True})])
        self.assertFalse(out2[0].passed)

    def test_assertion_lookup(self):
        r = result_with()
        out = r.verify([AcceptanceCriterion(name="s", spec={"assertion": "schema-valid"})])
        self.assertTrue(out[0].passed)
        out2 = r.verify([AcceptanceCriterion(name="s", spec={"assertion": "missing"})])
        self.assertFalse(out2[0].passed)

    def test_test_suite(self):
        r = result_with()
        out = r.verify([AcceptanceCriterion(name="u", spec={"test_suite": "unit"})])
        self.assertTrue(out[0].passed)

    def test_manual_needs_approver(self):
        r = result_with()
        out = r.verify([AcceptanceCriterion(name="m", spec={"manual": True})])
        self.assertTrue(out[0].passed)
        r2 = result_with(evidence={})
        out2 = r2.verify([AcceptanceCriterion(name="m", spec={"manual": True})])
        self.assertFalse(out2[0].passed)

    def test_unknown_spec_fails_closed(self):
        r = result_with()
        out = r.verify([AcceptanceCriterion(name="?", spec={"bogus": 1})])
        self.assertFalse(out[0].passed)

    def test_non_required_failure_does_not_block(self):
        r = result_with()
        r.verify([
            AcceptanceCriterion(name="must", spec={"metric": "accuracy", "gte": 0.9}),
            AcceptanceCriterion(name="nice", spec={"metric": "accuracy", "gte": 0.999},
                                required=False),
        ])
        self.assertTrue(r.passed)

    def test_result_serialization_roundtrip(self):
        r = result_with()
        r.verify([AcceptanceCriterion(name="acc", spec={"metric": "accuracy", "gte": 0.9})])
        d = r.to_dict()
        r2 = TaskResult.from_dict(d)
        self.assertTrue(r2.passed)
        self.assertEqual(r2.metrics["accuracy"], 0.97)


class TestTaskVerified(unittest.TestCase):
    def _task(self, criteria):
        t = Task(name="t", kind=TaskKind.IO, acceptance=criteria)
        t.mark_running()
        return t

    def test_verified_when_criteria_pass(self):
        t = self._task([AcceptanceCriterion(name="acc", spec={"metric": "accuracy", "gte": 0.9})])
        t.mark_done(result_with())
        self.assertEqual(t.state, TaskState.DONE)
        self.assertTrue(t.verified)

    def test_not_verified_when_criteria_fail(self):
        t = self._task([AcceptanceCriterion(name="acc", spec={"metric": "accuracy", "gte": 0.999})])
        t.mark_done(result_with())
        self.assertEqual(t.state, TaskState.DONE)  # completion still recorded
        self.assertFalse(t.verified)               # ...but not correctness

    def test_no_criteria_verified_means_completed(self):
        t = self._task([])
        t.mark_done({"raw": "anything"})
        self.assertTrue(t.verified)

    def test_raw_result_with_criteria_not_verified(self):
        t = self._task([AcceptanceCriterion(name="a", spec={"artifact": True})])
        t.mark_done("just a string")
        self.assertFalse(t.verified)

    def test_to_dict_reports_verified(self):
        t = self._task([AcceptanceCriterion(name="acc", spec={"metric": "accuracy", "gte": 0.9})])
        t.mark_running()
        t.mark_done(result_with())
        d = t.to_dict(include_result=True)
        self.assertTrue(d["verified"])
        self.assertTrue(d["result"]["passed"])
        self.assertEqual(len(d["acceptance"]), 1)


if __name__ == "__main__":
    unittest.main()
