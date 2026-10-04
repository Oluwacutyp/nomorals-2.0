"""Tests for the benchmark DB (nomorals.llm.benchmarks).

Rule under test: only real local measurements are ever stored — rows come
from actual provider calls (``seed_synthetic``) or explicit ``record()``
calls, never invented.
"""

from __future__ import annotations

import unittest

from nomorals.llm.base import LLMResponse, Message
from nomorals.llm.benchmarks import (
    BenchmarkDB,
    BenchmarkSample,
    benchmark_model,
    seed_synthetic,
)
from nomorals.llm.capabilities import Capability
from nomorals.llm.providers.mock import MockProvider


class FlakyProvider:
    """Fails every other call — deterministic, no network."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, params=None, **kw):
        self.calls += 1
        if self.calls % 2 == 0:
            raise RuntimeError("simulated outage")
        return LLMResponse(text="ok")


class BenchmarkDBTests(unittest.TestCase):
    def setUp(self):
        self.db = BenchmarkDB()

    def test_no_data_scores_neutral(self):
        self.assertEqual(self.db.score("ghost", "chat"), 0.5)

    def test_score_prefers_fast_and_reliable(self):
        for _ in range(6):
            self.db.record("fast", "chat", 0.05, True)
        for i in range(6):
            self.db.record("flaky", "chat", 2.5, success=(i % 2 == 0))
        fast, flaky = self.db.score("fast", "chat"), self.db.score("flaky", "chat")
        self.assertGreater(fast, flaky)
        self.assertGreater(fast, 0.9)
        self.assertLess(flaky, 0.7)

    def test_score_is_per_capability(self):
        self.db.record("m", "chat", 0.05, True)
        self.db.record("m", "vision", 5.0, False)
        self.assertGreater(self.db.score("m", "chat"), self.db.score("m", "vision"))

    def test_record_rejects_negative_latency(self):
        with self.assertRaises(ValueError):
            self.db.record("m", "chat", -1.0, True)

    def test_record_sample_dataclass(self):
        row_id = self.db.record_sample(BenchmarkSample(
            model_id="m", capability=Capability.CHAT, latency_s=0.1,
            success=True, source="synthetic"))
        self.assertGreater(row_id, 0)
        self.assertEqual(self.db.summary("m")["samples"], 1)

    def test_summary(self):
        self.db.record("m", "chat", 0.1, True, source="live")
        self.db.record("m", "chat", 0.3, True, source="live")
        self.db.record("m", "chat", 0.2, False, source="live")
        s = self.db.summary("m", "chat")
        self.assertEqual(s["samples"], 3)
        self.assertAlmostEqual(s["success_rate"], 2 / 3, places=3)
        self.assertAlmostEqual(s["median_latency_s"], 0.2, places=3)
        self.assertEqual(s["sources"], ["live"])

    def test_prune(self):
        self.db.record("m", "chat", 0.1, True)
        self.assertEqual(self.db.prune(older_than_days=-1), 1)
        self.assertEqual(self.db.summary("m")["samples"], 0)

    def test_models_lists_measured_models(self):
        self.db.record("a", "chat", 0.1, True)
        self.db.record("b", "chat", 0.1, True)
        self.assertEqual(sorted(self.db.models()), ["a", "b"])


class SyntheticSeedTests(unittest.TestCase):
    def test_seed_synthetic_measures_for_real(self):
        db = BenchmarkDB()
        provider = MockProvider(model="seed-me", latency_ms=5.0)
        result = seed_synthetic(db, "seed-me", "chat", provider,
                                ["probe one", "probe two", "probe three"])
        self.assertEqual(result["successes"], 3)
        self.assertEqual(len(result["rounds"]), 3)
        # Every stored row is a genuine measurement of an actual call.
        self.assertEqual(len(provider.calls), 3)
        summary = db.summary("seed-me", "chat")
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["sources"], ["synthetic"])
        self.assertGreaterEqual(summary["median_latency_s"], 0.0)
        latencies = [r["latency_s"] for r in db.samples("seed-me", "chat")]
        self.assertTrue(all(v >= 0 for v in latencies))

    def test_seed_synthetic_records_failures_honestly(self):
        db = BenchmarkDB()
        result = seed_synthetic(db, "flaky", "chat", FlakyProvider(),
                                ["a", "b", "c", "d"])
        self.assertEqual(result["successes"], 2)
        rows = db.samples("flaky", "chat", limit=10)
        self.assertEqual(sum(1 for r in rows if r["success"]), 2)
        self.assertEqual(sum(1 for r in rows if not r["success"]), 2)
        seed_synthetic(db, "perfect", "chat", MockProvider(model="perfect"),
                       ["a", "b", "c", "d"])
        self.assertLess(db.score("flaky", "chat"), db.score("perfect", "chat"))

    def test_benchmark_model_does_not_write(self):
        db = BenchmarkDB()
        result = benchmark_model("m", "chat", MockProvider(model="m"), ["hi"])
        self.assertEqual(result["rounds"][0]["success"], True)
        self.assertEqual(db.models(), [])  # benchmark_model never writes

    def test_embed_capability_probe(self):
        db = BenchmarkDB()
        result = seed_synthetic(db, "emb", Capability.EMBED,
                                MockProvider(model="emb"), ["vec me"])
        self.assertEqual(result["capability"], "embed")
        self.assertEqual(result["successes"], 1)


class PrefetchTests(unittest.TestCase):
    """R8 perf: one prefetch must equal the old per-model query path.

    ``select()`` used to issue three identical ``samples()`` queries per
    candidate; ``samples_many`` + the row-based scorers replace them with
    one query total.  These tests pin the equivalence.
    """

    def setUp(self):
        self.db = BenchmarkDB()
        for i in range(4):
            for j in range(25):
                self.db.record(f"m{i}", Capability.CHAT, 0.1 + 0.05 * j,
                               success=(j % 5 != 0))

    def test_samples_many_matches_samples(self):
        from nomorals.llm.benchmarks import SCORE_WINDOW
        got = self.db.samples_many(["m0", "m1", "m2", "m3"], Capability.CHAT)
        self.assertEqual(set(got), {"m0", "m1", "m2", "m3"})
        for mid, rows in got.items():
            expected = self.db.samples(mid, Capability.CHAT, limit=1000)
            self.assertEqual([r["latency_s"] for r in rows],
                             [r["latency_s"] for r in expected])
            self.assertLessEqual(len(rows), 1000)
        # limit is honored per model
        capped = self.db.samples_many(["m0"], Capability.CHAT, limit=7)
        self.assertEqual(len(capped["m0"]), 7)

    def test_samples_many_empty(self):
        self.assertEqual(self.db.samples_many([], Capability.CHAT), {})

    def test_score_rows_matches_score(self):
        from nomorals.llm.benchmarks import SCORE_WINDOW
        pref = self.db.samples_many(["m0", "m1", "m2", "m3"], Capability.CHAT)
        for mid, rows in pref.items():
            self.assertEqual(
                BenchmarkDB.score_rows(rows[:SCORE_WINDOW]),
                self.db.score(mid, Capability.CHAT))

    def test_score_rows_no_data_neutral(self):
        self.assertEqual(BenchmarkDB.score_rows([]), 0.5)

    def test_summary_rows_matches_summary(self):
        pref = self.db.samples_many(["m0", "m1", "m2", "m3"], Capability.CHAT)
        for mid, rows in pref.items():
            self.assertEqual(BenchmarkDB.summary_rows(mid, rows),
                             self.db.summary(mid, Capability.CHAT))

    def test_summary_is_single_query(self):
        calls = []
        inner = self.db.db.query

        def counting(sql, params=()):
            calls.append(sql)
            return inner(sql, params)

        self.db.db.query = counting
        try:
            self.db.summary("m0", Capability.CHAT)
        finally:
            self.db.db.query = inner
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
