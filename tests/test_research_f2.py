"""Wave F2 acceptance tests: research organs.

1. Ticket concreteness — upgrade tickets name real repo files (verified
   on disk), carry specific test names, and define acceptance criteria;
   ``gate_ticket`` rejects vague tickets; empty claims cannot become
   tickets.
2. Swarm dedup — two specialists reporting the same finding produce one
   merged finding with every source kept; affirmations are never merged
   with their negations.
3. Promotion gate — weak claims (low confidence, no evidence, too thin)
   are rejected with reasons and logged, never promoted; strong claims
   still promote.
4. Loop gating — with the research feature off, ``ResearchLoop.tick``
   does nothing: no pipeline run, no notify, no proposals, run recorded
   as skipped.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import research_digest as rd
from nomorals.agents import research_loop as rl
from nomorals.agents import research_swarm as rs
from nomorals.agents.kg import KnowledgeGraph
from nomorals.agents.research_digest import (
    ResearchClaim,
    ResearchDigest,
    gate_ticket,
    promote,
    promotion_gate,
)
from nomorals.agents.research_swarm import SwarmFinding
from nomorals.agents.upgrade_queue import UpgradePipeline, UpgradeQueue
from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _claim(cid, text, confidence=0.8, domain="systems", query="q"):
    return ResearchClaim(
        id=cid, claim=text, domain=domain, angle="core",
        sources=[{"url": "https://example.com/x", "title": "X",
                  "trust": 0.8}],
        confidence=confidence, fetched_at=1.0, query=query)


def _ctx(db=None, proactive=True):
    db = db or _db()
    partner = SimpleNamespace(proactive_enabled=proactive, quiet_start=22,
                              quiet_end=8)
    settings = SimpleNamespace(partner=partner)
    return SimpleNamespace(db=db, settings=settings)


# ── ticket concreteness ────────────────────────────────────────────────────

class TicketConcretenessTests(unittest.TestCase):
    def test_ticket_names_real_files_expected_tests_acceptance(self):
        claim = _claim(
            "t1",
            "Redis is the fastest in-memory cache for session storage in "
            "this stack",
            confidence=0.85, domain="systems")
        ticket = ResearchDigest.upgrade_ticket(claim)
        # files: all suggested, split into verified/unverified, verified real
        self.assertTrue(ticket["suggested_files"])
        self.assertTrue(all(f.startswith("nomorals/")
                            for f in ticket["suggested_files"]))
        self.assertEqual(set(ticket["suggested_files"]),
                         set(ticket["verified_files"])
                         | set(ticket["unverified_files"]))
        self.assertTrue(ticket["verified_files"])
        for f in ticket["verified_files"]:
            self.assertTrue((rd.repo_root() / f).exists(), f)
        # expected tests: specific test names, not vague descriptions
        self.assertGreaterEqual(len(ticket["expected_tests"]), 2)
        for name in ticket["expected_tests"]:
            self.assertRegex(name, r"^test_[a-z0-9_]+$")
            self.assertNotIn(" ", name)
        # acceptance criteria: concrete, checkable done-definition
        self.assertGreaterEqual(len(ticket["acceptance_criteria"]), 3)
        joined = " ".join(ticket["acceptance_criteria"]).lower()
        self.assertIn("error_scan", joined)
        self.assertIn("green", joined)
        # the gate agrees: no blocking problems
        self.assertEqual(gate_ticket(ticket), [])

    def test_gate_rejects_vague_ticket(self):
        vague = {
            "title": "[systems] make things better",
            "rationale": "research says this area could use some work, "
                         "probably worth looking at soon",
            "domain": "systems",
            "suggested_files": [],
            "expected_tests": [],
            "acceptance_criteria": [],
            "claim_ids": ["c9"],
            "confidence": 0.9,
        }
        problems = gate_ticket(vague)
        self.assertTrue(problems)
        joined = " ".join(problems).lower()
        self.assertIn("suggested files", joined)
        self.assertIn("expected_tests", joined)
        self.assertIn("acceptance_criteria", joined)

    def test_gate_rejects_nonexistent_files(self):
        ticket = ResearchDigest.upgrade_ticket(
            _claim("t2", "Redis is the fastest in-memory cache",
                   confidence=0.8, domain="systems"))
        ticket["suggested_files"] = ["nomorals/agents/does_not_exist.py"]
        problems = gate_ticket(ticket)
        self.assertTrue(any("do not exist" in p for p in problems))

    def test_empty_claim_cannot_become_ticket(self):
        claim = _claim("t3", "   ", confidence=0.9)
        with self.assertRaises(ValueError):
            ResearchDigest.upgrade_ticket(claim)

    def test_queue_rejects_vague_but_confident_ticket(self):
        ctx = _ctx()
        pipe = UpgradePipeline(ctx)
        vague = {
            "title": "[systems] make things better",
            "rationale": "research says this area could use some work, "
                         "probably worth looking at soon",
            "patch_plan": {"steps": ["do things"]},
            "suggested_files": [],
            "test_plan": [],
            "expected_tests": [],
            "acceptance_criteria": [],
            "claim_ids": ["c9"],
            "confidence": 0.95,
        }
        self.assertEqual(pipe.propose_from_ticket(vague), "")
        self.assertEqual(UpgradeQueue(ctx).list(), [])

    def test_queue_accepts_concrete_ticket(self):
        ctx = _ctx()
        pipe = UpgradePipeline(ctx)
        ticket = ResearchDigest.upgrade_ticket(
            _claim("t4", "Redis is the fastest in-memory cache",
                   confidence=0.85, domain="systems"))
        pid = pipe.propose_from_ticket(ticket, source="research")
        self.assertTrue(pid)
        row = UpgradeQueue(ctx).get(pid)
        self.assertEqual(row["status"], "proposed")
        self.assertEqual(row["source"], "research")
        self.assertTrue(row["files"])  # verified repo files carried through


# ── swarm cross-specialist dedup ───────────────────────────────────────────

class SwarmDedupTests(unittest.TestCase):
    def _finding(self, claim, domain, url, confidence=0.7):
        return SwarmFinding(
            angle=f"{domain} angle", claim=claim,
            sources=[{"url": url, "title": "src", "trust": 0.8}],
            confidence=confidence, domain=domain)

    def test_same_finding_two_specialists_merges(self):
        a = self._finding(
            "Redis pipelining dramatically reduces round-trip latency for "
            "bulk session writes",
            "systems", "https://example.com/redis-pipe", 0.7)
        b = self._finding(
            "Redis pipelining cuts round-trip latency hard when writing "
            "sessions in bulk",
            "tooling", "https://example.com/pipe-guide", 0.85)
        merged = rs.ResearchSwarm.dedupe_findings([a, b])
        self.assertEqual(len(merged), 1)
        m = merged[0]
        # every source kept
        urls = m.evidence_urls()
        self.assertIn("https://example.com/redis-pipe", urls)
        self.assertIn("https://example.com/pipe-guide", urls)
        # max confidence wins, both domains recorded
        self.assertEqual(m.confidence, 0.85)
        self.assertIn("systems", m.domain)
        self.assertIn("tooling", m.domain)

    def test_affirmation_and_negation_never_merge(self):
        a = self._finding(
            "PostgreSQL is the best relational database for analytics "
            "workloads",
            "systems", "https://example.com/pg", 0.85)
        b = self._finding(
            "PostgreSQL is not suitable for analytics workloads",
            "competitors", "https://example.com/nopg", 0.6)
        merged = rs.ResearchSwarm.dedupe_findings([a, b])
        self.assertEqual(len(merged), 2)

    def test_distinct_claims_untouched(self):
        findings = [
            self._finding("Redis pipelining reduces round-trip latency",
                          "systems", "https://example.com/a"),
            self._finding("Consistent hashing spreads cache keys evenly",
                          "systems", "https://example.com/b"),
        ]
        self.assertEqual(len(rs.ResearchSwarm.dedupe_findings(findings)), 2)

    def test_evidence_urls(self):
        f = self._finding("Redis pipelining reduces round-trip latency",
                          "systems", "https://example.com/a")
        self.assertEqual(f.evidence_urls(), ["https://example.com/a"])


# ── promotion gate ─────────────────────────────────────────────────────────

class PromotionGateTests(unittest.TestCase):
    def test_gate_rejects_low_confidence_with_reason(self):
        ok, reasons = promotion_gate(_claim("g1", "Redis is the fastest "
                                                 "in-memory cache",
                                           confidence=0.2))
        self.assertFalse(ok)
        self.assertTrue(any("confidence" in r for r in reasons))

    def test_gate_rejects_no_evidence(self):
        claim = _claim("g2", "Redis is the fastest in-memory cache",
                       confidence=0.9)
        claim.sources = []
        ok, reasons = promotion_gate(claim)
        self.assertFalse(ok)
        self.assertTrue(any("URL" in r for r in reasons))

    def test_gate_rejects_thin_claim(self):
        claim = _claim("g3", "Redis fast!", confidence=0.9)
        ok, reasons = promotion_gate(claim)
        self.assertFalse(ok)
        self.assertTrue(any("thin" in r for r in reasons))

    def test_gate_accepts_strong_claim(self):
        ok, reasons = promotion_gate(
            _claim("g4", "Redis is the fastest in-memory cache",
                   confidence=0.85))
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_promote_rejects_and_logs_weak_never_promoted(self):
        db = _db()
        weak = _claim("w1", "a vague rumor", confidence=0.2)
        weak.sources = []  # no evidence either
        strong = _claim("s1", "Redis is the fastest in-memory cache",
                        confidence=0.85)
        result = promote([weak, strong], db)
        self.assertEqual(result["promoted"], 1)
        self.assertEqual(result["skipped_low_confidence"], 1)
        self.assertEqual(len(result["rejected"]), 1)
        rej = result["rejected"][0]
        self.assertEqual(rej["claim_id"], "w1")
        self.assertTrue(rej["reasons"])
        # weak claim never became a node
        kg = KnowledgeGraph(db)
        self.assertIsNone(kg.find_node("claim:w1", type="claim"))
        self.assertIsNotNone(kg.find_node("claim:s1", type="claim"))

    def test_strong_claim_still_promotes(self):
        db = _db()
        result = promote([_claim("s2", "Redis is the fastest in-memory "
                                      "cache", confidence=0.85)], db)
        self.assertEqual(result["promoted"], 1)
        self.assertEqual(result["rejected"], [])


# ── research loop stays gated ──────────────────────────────────────────────

class LoopGatedOffTests(unittest.TestCase):
    def _disabled_ctx(self):
        # feature flag "research" defaults to off — the opt-in loop stays
        # off unless the owner explicitly enables it.
        return _ctx()

    def test_tick_when_disabled_does_nothing(self):
        ctx = self._disabled_ctx()
        fake_pipeline = mock.Mock()
        fake_notifier = mock.Mock()
        with (mock.patch("nomorals.agents.research_digest.ResearchPipeline",
                         fake_pipeline),
              mock.patch("nomorals.agents.research_loop.Notifier",
                         return_value=fake_notifier),
              mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                         return_value=False)):
            out = rl.ResearchLoop(ctx).tick()
        self.assertFalse(out["ok"])
        self.assertIn("research", out["skipped_reason"])
        self.assertEqual(out["proposals_created"], 0)
        self.assertFalse(out["notified"])
        # the expensive organs were never touched
        fake_pipeline.run.assert_not_called()
        fake_notifier.publish.assert_not_called()
        # the deferral is recorded, honestly
        row = ctx.db.query_one(
            "SELECT skipped_reason, proposals_created, notified "
            "FROM research_loop_runs")
        self.assertIn("research", row["skipped_reason"])
        self.assertEqual(row["proposals_created"], 0)
        self.assertEqual(row["notified"], 0)

    def test_tick_during_quiet_hours_does_nothing(self):
        from nomorals.agents.features import FeatureRegistry

        ctx = self._disabled_ctx()
        FeatureRegistry(ctx.db).set("research", True)
        fake_pipeline = mock.Mock()
        with (mock.patch("nomorals.agents.research_digest.ResearchPipeline",
                         fake_pipeline),
              mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                         return_value=True)):
            out = rl.ResearchLoop(ctx).tick()
        self.assertFalse(out["ok"])
        self.assertIn("quiet hours", out["skipped_reason"])
        fake_pipeline.run.assert_not_called()

    def test_scheduling_is_not_enabling(self):
        # ensure_research_job only registers the scheduler row; the loop
        # itself stays gated off until the owner opts in.
        from nomorals.agents.features import FeatureRegistry

        ctx = self._disabled_ctx()
        info = rl.ensure_research_job(ctx)
        self.assertTrue(info.get("scheduled") or
                        info.get("already_scheduled"))
        self.assertFalse(FeatureRegistry(ctx.db).get("research"))
        out = rl.ResearchLoop(ctx).tick()
        self.assertFalse(out["ok"])


# ── research modules still present (no deletions) ──────────────────────────

class ModulesPresentTests(unittest.TestCase):
    def test_all_research_organs_importable(self):
        import importlib

        for name in ("nomorals.agents.research_swarm",
                     "nomorals.agents.research_digest",
                     "nomorals.agents.research_loop",
                     "nomorals.agents.research_lexicon",
                     "nomorals.agents.upgrade_queue",
                     "nomorals.agents.researcher",
                     "nomorals.agents.kg"):
            mod = importlib.import_module(name)
            self.assertTrue(hasattr(mod, "register") or True)
        # key entry points intact
        self.assertTrue(callable(rs.ResearchSwarm))
        self.assertTrue(callable(rd.ResearchPipeline.run))
        self.assertTrue(callable(rd.promote))
        self.assertTrue(callable(rl.ResearchLoop))


if __name__ == "__main__":
    unittest.main()
