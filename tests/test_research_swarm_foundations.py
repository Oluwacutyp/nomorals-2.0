"""Foundations (research-loop proposal #2) acceptance tests for ResearchSwarm.

G1 — failed angles get exactly one bounded retry (rephrased on
    "no search results", same angle on transient errors).
G2 — a researchable supervision pivot runs exactly one follow-up pass;
    a non-researchable pivot is never treated as an angle.
G3 — long-term memory is consulted BEFORE researching: prior swarm facts
    seed the report and dedupe new findings; memory failure never breaks
    a run; a context without memory behaves exactly as before.

All network is stubbed via ``_research_angle`` / ``_supervise`` /
fake ``context.memory`` — no network, no DB.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.research_swarm import ResearchSwarm, SwarmFinding


def _ctx(memory=None):
    ctx = SimpleNamespace(db=None, tools=None, router=None,
                          settings=None, extras={})
    if memory is not None:
        ctx.memory = memory
    return ctx


def _finding(angle, claim):
    return SwarmFinding(
        angle=angle, claim=claim,
        sources=[{"url": "https://example.com/x", "title": "example",
                  "trust": 0.7}],
        confidence=0.6, domain="")


def _ok_result(angle, claim):
    return {"angle": angle, "ok": True, "findings": [_finding(angle, claim)],
            "sources": [{"url": "https://example.com/x", "title": "example",
                         "trust": 0.7}],
            "pages_read": 0, "seconds": 0.01}


class _FakeMemory:
    """Duck-typed stand-in for MemoryManager.recall."""

    def __init__(self, records=None, explode=False):
        self._records = list(records or [])
        self.explode = explode
        self.calls: list[str] = []

    def recall(self, query, limit=5, **_kwargs):
        self.calls.append(query)
        if self.explode:
            raise RuntimeError("memory store down")
        return SimpleNamespace(records=list(self._records))


class RetryTests(unittest.TestCase):
    def test_failed_angle_retried_once_with_rephrasing(self):
        swarm = ResearchSwarm(_ctx(), workers=1)
        calls: list[str] = []

        def flaky(angle, worker_index, domain=""):
            calls.append(angle)
            if angle == "postgres":
                return {"angle": angle, "ok": False,
                        "error": "no search results",
                        "findings": [], "sources": []}
            # the rephrased retry
            return _ok_result("postgres", "postgres scales reads")

        swarm._research_angle = flaky  # noqa: SLF001 - stub the network
        report = swarm.run("postgres", angles=["postgres"])
        self.assertEqual(report.failed_angles, [])
        self.assertEqual(report.retried_angles, ["postgres"])
        self.assertEqual(report.recovered_angles, ["postgres"])
        self.assertEqual([f.claim for f in report.findings],
                         ["postgres scales reads"])
        # original angle + exactly one rephrased retry — bounded
        self.assertEqual(calls, ["postgres", "postgres overview"])
        # findings stay labelled under the ORIGINAL angle
        self.assertEqual(report.findings[0].angle, "postgres")

    def test_failed_angle_stays_failed_after_retry_exhausted(self):
        swarm = ResearchSwarm(_ctx(), workers=1)
        calls: list[str] = []

        def always_fail(angle, worker_index, domain=""):
            calls.append(angle)
            return {"angle": angle, "ok": False, "error": "timeout",
                    "findings": [], "sources": []}

        swarm._research_angle = always_fail  # noqa: SLF001 - stub the network
        report = swarm.run("postgres", angles=["postgres"])
        self.assertEqual(report.failed_angles, ["postgres"])
        self.assertEqual(report.retried_angles, ["postgres"])
        self.assertEqual(report.recovered_angles, [])
        self.assertEqual(report.findings, [])
        # exactly one retry — no infinite loop
        self.assertEqual(len(calls), 2)


class PivotTests(unittest.TestCase):
    def test_critique_pivot_runs_one_followup_pass(self):
        swarm = ResearchSwarm(_ctx(), workers=1)
        seen: list[str] = []

        def worker(angle, worker_index, domain=""):
            seen.append(angle)
            claim = ("Postgres uses MVCC for concurrency"
                     if angle == "postgres" else
                     "Benchmarks show Postgres beating MySQL on read-heavy workloads")
            return _ok_result(angle, claim)

        swarm._research_angle = worker  # noqa: SLF001 - stub the network
        swarm._supervise = lambda q, r: (  # noqa: SLF001 - stub supervision
            "weak: thin evidence", "postgres benchmarks vs mysql")
        report = swarm.run("postgres", angles=["postgres"])
        self.assertEqual(report.pivot_angles, ["postgres benchmarks vs mysql"])
        self.assertIn("postgres benchmarks vs mysql", seen)
        self.assertTrue(any(f.angle == "postgres benchmarks vs mysql"
                            for f in report.findings))
        self.assertEqual(report.critique, "weak: thin evidence")

    def test_non_researchable_pivot_is_not_an_angle(self):
        swarm = ResearchSwarm(_ctx(), workers=1)
        seen: list[str] = []

        def worker(angle, worker_index, domain=""):
            seen.append(angle)
            return _ok_result(angle, f"finding on {angle}")

        swarm._research_angle = worker  # noqa: SLF001 - stub the network
        swarm._supervise = lambda q, r: (  # noqa: SLF001 - stub supervision
            "weak: approach", "change approach: try a different tool")
        report = swarm.run("postgres", angles=["postgres"])
        self.assertEqual(report.pivot_angles, [])
        self.assertEqual(seen, ["postgres"])


class MemoryTests(unittest.TestCase):
    def test_memory_recall_seeds_prior_and_dedupes(self):
        prior = SimpleNamespace(
            content="[research: postgres] Postgres supports JSONB indexing "
                    "(source: https://example.com)",
            source="research_swarm:postgres")
        memory = _FakeMemory(records=[prior])
        swarm = ResearchSwarm(_ctx(memory=memory), workers=1)

        def worker(angle, worker_index, domain=""):
            # one already-known claim + one genuinely new claim
            return {"angle": angle, "ok": True,
                    "findings": [_finding(angle, "Postgres supports JSONB indexing"),
                                 _finding(angle, "Postgres has MVCC concurrency")],
                    "sources": [], "pages_read": 0, "seconds": 0.01}

        swarm._research_angle = worker  # noqa: SLF001 - stub the network
        report = swarm.run("postgres", angles=["postgres"])
        self.assertEqual(memory.calls, ["postgres"])
        self.assertEqual(len(report.prior_findings), 1)
        self.assertEqual(report.prior_findings[0].claim,
                         "Postgres supports JSONB indexing")
        self.assertEqual(report.prior_overlap, 1)
        # the already-known claim is NOT re-reported as a new finding
        self.assertEqual([f.claim for f in report.findings],
                         ["Postgres has MVCC concurrency"])

    def test_memory_failure_never_breaks_run(self):
        memory = _FakeMemory(explode=True)
        swarm = ResearchSwarm(_ctx(memory=memory), workers=1)
        swarm._research_angle = (  # noqa: SLF001 - stub the network
            lambda a, i, domain="": _ok_result(a, "finding on postgres"))
        report = swarm.run("postgres", angles=["postgres"])
        self.assertEqual(report.prior_findings, [])
        self.assertEqual(len(report.findings), 1)

    def test_no_memory_context_still_works(self):
        swarm = ResearchSwarm(_ctx(), workers=1)
        swarm._research_angle = (  # noqa: SLF001 - stub the network
            lambda a, i, domain="": _ok_result(a, "finding on postgres"))
        report = swarm.run("postgres", angles=["postgres"])
        self.assertEqual(report.prior_findings, [])
        self.assertEqual(len(report.findings), 1)


if __name__ == "__main__":
    unittest.main()
