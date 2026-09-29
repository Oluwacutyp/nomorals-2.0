"""Offline tests for live collection, trigger policy, and the durable pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.self_improvement import SelfImprovementJob
from nomorals.training.collect import TrainingCollector, scrub_pii
from nomorals.training.policy import RetrainingPolicy


class SelfImprovementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = tempfile.mkdtemp(prefix="nm-loop-")
        self.context = build_context(
            Settings(home=self.home),
            with_executor=False,
            with_router=False,
            with_memory=False,
            with_tools=False,
        )
        self.context.__enter__()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)

    def test_scrubber_removes_direct_identifiers(self) -> None:
        clean = scrub_pii("email alice@example.com, token ghp_abcdefghijklmnop and 10.2.3.4")
        self.assertNotIn("alice@example.com", clean)
        self.assertNotIn("ghp_abcdefghijklmnop", clean)
        self.assertNotIn("10.2.3.4", clean)
        self.assertIn("[REDACTED_EMAIL]", clean)

    def test_collector_harvests_all_sources_and_dedupes_registered_data(self) -> None:
        db = self.context.db
        for index in range(3):
            db.insert(
                "memories",
                {
                    "id": f"memory-{index}",
                    "kind": "episode",
                    "content": f"A sufficiently useful conversation lesson number {index}.",
                    "created_at": float(index + 1),
                    "updated_at": float(index + 1),
                },
            )
        db.insert(
            "tool_calls",
            {"id": "tool-1", "tool": "shell_run", "status": "error", "error": "failed", "created_at": 4.0},
        )
        db.insert(
            "reflections",
            {
                "id": "reflection-1", "mission_id": "m", "score": 0.4,
                "summary": "Verification was missed.", "lessons": '["add a check"]', "created_at": 5.0,
            },
        )
        first = TrainingCollector(db).collect(
            output_dir=Path(self.home) / "data", name="collected", register=True
        )
        self.assertGreaterEqual(first.count, 4)
        self.assertEqual(first.stats.memories, 3)
        self.assertEqual(first.stats.tool_calls, 1)
        self.assertEqual(first.stats.reflections, 1)

        second = TrainingCollector(db).collect()
        self.assertEqual(second.count, 0)
        self.assertGreaterEqual(second.stats.duplicates, first.count)

    def test_policy_explains_growth_and_declining_reflections(self) -> None:
        policy = RetrainingPolicy(dataset_growth_threshold=3, interval_seconds=9999, reflection_window=4)
        decision = policy.decide(new_examples=3, elapsed_seconds=1, reflection_scores=[0.9, 0.8, 0.7, 0.6], data_available=True)
        self.assertTrue(decision.should_retrain)
        self.assertIn("dataset growth threshold reached", decision.reasons)
        self.assertIn("reflection scores trending down", decision.reasons)

    def test_job_resumes_an_interrupted_mission(self) -> None:
        db = self.context.db
        for index in range(6):
            db.insert(
                "memories",
                {
                    "id": f"crash-memory-{index}", "kind": "episode",
                    "content": f"Crash-safe training material with a useful independent lesson {index}.",
                    "created_at": float(index + 1), "updated_at": float(index + 1),
                },
            )

        class CrashOnce(TrainingCollector):
            crashed = False

            def collect(self, **kwargs):  # type: ignore[no-untyped-def]
                if not self.crashed:
                    self.crashed = True
                    raise KeyboardInterrupt
                return super().collect(**kwargs)

        self.context.settings.training.epochs = 1
        first = SelfImprovementJob(
            self.context,
            policy=RetrainingPolicy(dataset_growth_threshold=1, interval_seconds=0),
            collector=CrashOnce(db),
        )
        with self.assertRaises(KeyboardInterrupt):
            first.run(force=True)
        self.assertEqual(len(first.store.resumable()), 1)

        resumed = SelfImprovementJob(
            self.context,
            policy=RetrainingPolicy(dataset_growth_threshold=1, interval_seconds=0),
        ).run(force=True)
        self.assertTrue(resumed.ok)
        self.assertEqual(first.store.stats()["active"], 0)

    def test_job_runs_all_stages_and_promotes_a_gated_model(self) -> None:
        db = self.context.db
        for index in range(8):
            db.insert(
                "memories",
                {
                    "id": f"memory-{index}", "kind": "episode",
                    "content": f"Create a durable training example with a distinct useful topic number {index}.",
                    "created_at": float(index + 1), "updated_at": float(index + 1),
                },
            )
        self.context.settings.training.epochs = 1
        self.context.settings.training.eval_split = 0.2
        job = SelfImprovementJob(
            self.context,
            policy=RetrainingPolicy(dataset_growth_threshold=1, interval_seconds=0),
        )
        result = job.run(force=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "done")
        self.assertTrue(result.run_id)
        self.assertIn("promote", [step["step"] for step in result.steps])
        self.assertEqual(job.runs.stats()["promoted"], 1)


if __name__ == "__main__":
    unittest.main()
