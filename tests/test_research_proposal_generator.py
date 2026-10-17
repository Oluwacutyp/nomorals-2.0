"""Path B — research-loop proposal GENERATOR: concrete or nothing.

The old generator turned ANY finding into "survey title + generic test
template" ("encode the claim as a test: assert the behavior '<title>'")
and asserted, without reading the code, that "nothing asserts it".
These tests pin the fixed contract:

* a vague finding (survey title, taxonomy, bare fact, no gap, no
  testable behavior) produces NO proposal — and logs the skip with the
  finding id + reason;
* a finding with a real gap produces a proposal naming the specific
  behavior, file:line gap evidence verified against the current code,
  specific files, and a failing test;
* a finding for an already-implemented behavior produces NO proposal
  (skip reason "already implemented at <file:line>");
* the generic template form is rejected by the gate, never emitted;
* the /proposals read path still returns what the generator stores
  (source="research_loop", concrete substance in the stored plan).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.research_digest import (
    ResearchClaim,
    ResearchDigest,
    ResearchPipeline,
    build_proposal_spec,
    claim_from_finding,
    find_implementation,
    gate_ticket,
    verify_gap,
)
from nomorals.agents.research_swarm import SwarmFinding, SwarmReport
from nomorals.agents.upgrade_queue import UpgradePipeline, UpgradeQueue
from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(db=None):
    return SimpleNamespace(db=db or _db(), settings=SimpleNamespace(
        partner=SimpleNamespace(proactive_enabled=True, quiet_start=22,
                                quiet_end=8)))


def _claim(cid, text, **kw):
    base = dict(domain="systems", angle="core",
                sources=[{"url": "https://example.com/x", "title": "X",
                          "trust": 0.8}],
                confidence=0.85, fetched_at=1.0, query="q")
    base.update(kw)
    return ResearchClaim(id=cid, claim=text, **base)


def _gap_claim(cid="g1", **kw):
    """A claim with a verified spec for a behavior genuinely absent
    from the current code (the loop never sends a weekly email
    digest)."""
    claim = _claim(cid, "ResearchLoop never sends the owner a weekly "
                   "email digest of skipped proposals")
    claim.behavior = ("ResearchLoop publishes a weekly email digest "
                      "listing skipped proposal reasons to the owner")
    claim.target_files = ["nomorals/agents/research_loop.py"]
    for key, val in kw.items():
        setattr(claim, key, val)
    return claim


def _implemented_claim(cid="a1"):
    """A claim whose behavior the code ALREADY has: the loop dedupes
    upgrade proposals by title fingerprint before filing."""
    claim = _claim(cid, "ResearchLoop dedupes proposals")
    claim.behavior = ("ResearchLoop dedupes upgrade proposals by title "
                      "fingerprint before filing")
    claim.target_files = ["nomorals/agents/research_loop.py"]
    return claim


class VagueFindingsProduceNothingTests(unittest.TestCase):
    def test_survey_title_produces_no_proposal_and_logs_skip(self):
        claim = _claim("v-survey",
                       "Alignment and Safety in Large Language Models: "
                       "A Survey of Current Approaches and Future "
                       "Directions")
        with self.assertLogs("nomorals.agents.research_digest",
                             level="WARNING") as logs:
            ticket, skip = ResearchDigest.upgrade_ticket_or_skip(claim)
        self.assertIsNone(ticket)
        self.assertIsNotNone(skip)
        self.assertEqual(skip["finding_id"], "v-survey")
        self.assertEqual(skip["reason"], "no testable behavior")
        joined = "\n".join(logs.output)
        self.assertIn("proposal skipped: finding_id=v-survey", joined)
        self.assertIn("reason=no testable behavior", joined)

    def test_bare_fact_produces_no_proposal(self):
        claim = _claim("v-fact",
                       "Redis is the fastest in-memory cache for "
                       "session storage in this stack")
        ticket, skip = ResearchDigest.upgrade_ticket_or_skip(claim)
        self.assertIsNone(ticket)
        self.assertEqual(skip["reason"], "no testable behavior")

    def test_taxonomy_produces_no_proposal(self):
        claim = _claim("v-tax", "A taxonomy of agentic workflow "
                       "orchestration patterns and failure modes")
        ticket, skip = ResearchDigest.upgrade_ticket_or_skip(claim)
        self.assertIsNone(ticket)
        self.assertEqual(skip["reason"], "no testable behavior")

    def test_upgrade_ticket_raises_for_vague(self):
        claim = _claim("v-raise", "Alignment and Safety in Large "
                       "Language Models: A Survey")
        with self.assertRaises(ValueError) as ctx:
            ResearchDigest.upgrade_ticket(claim)
        self.assertIn("no testable behavior", str(ctx.exception))

    def test_empty_claim_produces_no_proposal(self):
        claim = _claim("v-empty", "   ")
        ticket, skip = ResearchDigest.upgrade_ticket_or_skip(claim)
        self.assertIsNone(ticket)
        self.assertEqual(skip["reason"], "no testable behavior")

    def test_behavior_without_owning_module_produces_no_proposal(self):
        claim = _claim("v-nofile", "The system should send a weekly "
                       "digest but never does")
        claim.behavior = ("Publish a weekly email digest listing "
                          "skipped proposal reasons to the owner")
        ticket, skip = ResearchDigest.upgrade_ticket_or_skip(claim)
        self.assertIsNone(ticket)
        self.assertEqual(skip["reason"], "no testable behavior")
        self.assertIn("owning module", skip["detail"])


class RealGapProducesConcreteProposalTests(unittest.TestCase):
    def test_structured_gap_ticket_is_concrete(self):
        ticket, skip = ResearchDigest.upgrade_ticket_or_skip(_gap_claim())
        self.assertIsNone(skip)
        assert ticket is not None
        # specific behavior, not the claim title rephrased
        self.assertIn("weekly email digest", ticket["behavior"])
        # file:line gap evidence verified against the current code
        self.assertRegex(ticket["gap_evidence"],
                         r"nomorals/agents/research_loop\.py:\d+")
        self.assertIn("code checked", ticket["gap_evidence"])
        # specific files, all real
        self.assertEqual(ticket["suggested_files"],
                         ["nomorals/agents/research_loop.py"])
        self.assertEqual(ticket["verified_files"],
                         ["nomorals/agents/research_loop.py"])
        # a failing test, named + described
        self.assertRegex(ticket["failing_test"], r"test_[a-z0-9_]+")
        self.assertIn("fails on the current tree", ticket["failing_test"])
        # the gate agrees
        self.assertEqual(gate_ticket(ticket), [])
        # the old generic template form is nowhere in the ticket
        blob = " ".join(str(v) for v in ticket.values()
                        if isinstance(v, str)).lower()
        self.assertNotIn("encode the claim as a test", blob)
        self.assertNotIn("assert the behavior", blob)
        self.assertNotIn("not encoded anywhere", blob)
        self.assertNotIn("nothing in", blob)
        self.assertNotIn("asserts it", blob)

    def test_text_mined_gap_ticket_is_concrete(self):
        claim = _claim(
            "g-text",
            "nomorals/agents/research_loop.py lacks a weekly email "
            "digest: it never sends the owner a weekly email listing "
            "skipped proposal reasons.")
        ticket, skip = ResearchDigest.upgrade_ticket_or_skip(claim)
        self.assertIsNone(skip)
        assert ticket is not None
        self.assertIn("weekly email digest", ticket["behavior"])
        self.assertRegex(ticket["gap_evidence"],
                         r"nomorals/agents/research_loop\.py:\d+")
        self.assertEqual(gate_ticket(ticket), [])

    def test_bogus_supplied_evidence_produces_no_proposal(self):
        claim = _gap_claim(gap_evidence="nomorals/agents/research_loop.py"
                           ":99999 — bogus")
        ticket, skip = ResearchDigest.upgrade_ticket_or_skip(claim)
        self.assertIsNone(ticket)
        self.assertEqual(skip["reason"], "gap evidence not verified")
        self.assertIn("out of range", skip["detail"])

    def test_missing_target_file_produces_no_proposal(self):
        claim = _gap_claim()
        claim.target_files = ["nomorals/agents/does_not_exist.py"]
        ticket, skip = ResearchDigest.upgrade_ticket_or_skip(claim)
        self.assertIsNone(ticket)
        self.assertEqual(skip["reason"], "gap evidence not verified")

    def test_verify_gap_confirms_gap_with_anchor(self):
        check = verify_gap(
            "ResearchLoop publishes a weekly email digest listing "
            "skipped proposal reasons to the owner",
            ["nomorals/agents/research_loop.py"])
        self.assertEqual(check["status"], "gap_confirmed")
        self.assertRegex(check["evidence"],
                         r"nomorals/agents/research_loop\.py:\d+")

    def test_pipeline_files_concrete_ticket_and_reports_skip(self):
        ctx = _ctx()

        class _FakeSwarm:
            def __init__(self, context, specialists=None, **kw):
                pass

            def run(self, query, **kw):
                return SwarmReport(
                    query=query, angles=["core"],
                    findings=[
                        SwarmFinding(
                            angle="core",
                            claim="ResearchLoop never sends the owner "
                                  "a weekly email digest",
                            sources=[{"url": "https://example.com/d",
                                      "title": "D", "trust": 0.9}],
                            confidence=0.85,
                            behavior="ResearchLoop publishes a weekly "
                                     "email digest listing skipped "
                                     "proposal reasons to the owner",
                            target_files=[
                                "nomorals/agents/research_loop.py"]),
                        SwarmFinding(
                            angle="core",
                            claim="Alignment and Safety in Large "
                                  "Language Models: A Survey",
                            sources=[{"url": "https://example.com/s",
                                      "title": "S", "trust": 0.9}],
                            confidence=0.85),
                    ],
                    synthesis="syn")

        import nomorals.agents.research_digest as rd

        real = rd.ResearchSwarm
        rd.ResearchSwarm = _FakeSwarm  # type: ignore[assignment]
        try:
            out = ResearchPipeline.run("weekly digest", ctx)
        finally:
            rd.ResearchSwarm = real
        self.assertEqual(len(out["tickets"]), 1)
        self.assertEqual(out["skipped_count"], 1)
        self.assertEqual(out["skipped"][0]["reason"],
                         "no testable behavior")


class AlreadyImplementedProducesNothingTests(unittest.TestCase):
    def test_structured_already_implemented_skips(self):
        claim = _implemented_claim()
        with self.assertLogs("nomorals.agents.research_digest",
                             level="WARNING") as logs:
            ticket, skip = ResearchDigest.upgrade_ticket_or_skip(claim)
        self.assertIsNone(ticket)
        assert skip is not None
        self.assertTrue(skip["reason"].startswith("already implemented at "
                         "nomorals/agents/research_loop.py:"),
                        skip)
        joined = "\n".join(logs.output)
        self.assertIn("already implemented at "
                      "nomorals/agents/research_loop.py:", joined)

    def test_find_implementation_returns_file_line(self):
        at = find_implementation(
            "ResearchLoop dedupes upgrade proposals by title "
            "fingerprint before filing",
            "nomorals/agents/research_loop.py")
        self.assertIsNotNone(at)
        assert at is not None
        self.assertTrue(
            at.startswith("nomorals/agents/research_loop.py:"), at)

    def test_find_implementation_ignores_comments_and_strings(self):
        # "verifies the gap by reading the owning module" is only ever
        # SAID in comments/strings in the old code — prose is not
        # implementation. (After this fix the generator does read the
        # module, so the probe uses a behavior still absent: a weekly
        # digest is never *sent* anywhere in the code.)
        at = find_implementation(
            "ResearchLoop publishes a weekly email digest listing "
            "skipped proposal reasons to the owner",
            "nomorals/agents/research_loop.py")
        self.assertIsNone(at)

    def test_build_proposal_spec_skip_shape(self):
        spec, skip = build_proposal_spec(_implemented_claim())
        self.assertIsNone(spec)
        assert skip is not None
        self.assertEqual(set(skip), {"finding_id", "claim_id", "reason",
                                     "detail"})


class GateRejectsGenericTemplateTests(unittest.TestCase):
    def test_gate_rejects_survey_title_plus_generic_test_template(self):
        ticket = {
            "title": "[general] Alignment and Safety in Large Language "
                     "Models: A Survey",
            "rationale": "Research finding: Alignment and Safety in "
                         "Large Language Models: A Survey",
            "domain": "general",
            "suggested_files": ["nomorals/agents/research_swarm.py"],
            "expected_tests": ["test_claim_asserts_behavior"],
            "acceptance_criteria": ["test passes"],
            "claim_ids": ["c1"],
            "problem": "The claim is not encoded anywhere in the repo — "
                       "nothing in nomorals/agents/research_swarm.py "
                       "asserts it, so the behavior is untested.",
            "proposed_change": "Approach: encode the claim as a failing "
                               "test first in "
                               "nomorals/agents/research_swarm.py, then "
                               "implement the smallest change that makes "
                               "it pass.",
            "risk": "Blast radius: unknown — needs human review before "
                    "any edit; fully reversible by reverting.",
            "verify_steps": ["Run pytest -k test_claim_asserts_behavior "
                             "— fails, then passes."],
            "test_plan": ["Encode the claim as a test: assert the "
                          "behavior 'Alignment and Safety'."],
            "confidence": 0.9,
        }
        problems = gate_ticket(ticket)
        joined = " ".join(problems)
        self.assertIn("generic template", joined)
        self.assertIn("behavior", joined)
        self.assertIn("gap_evidence", joined)
        self.assertIn("failing_test", joined)

    def test_gate_rejects_ticket_without_gap_evidence(self):
        ticket, _ = ResearchDigest.upgrade_ticket_or_skip(_gap_claim())
        assert ticket is not None
        ticket["gap_evidence"] = ""
        problems = gate_ticket(ticket)
        self.assertTrue(any("gap_evidence" in p for p in problems))


class QueueAndProposalsReadPathTests(unittest.TestCase):
    def test_generator_ticket_files_with_source_research_loop(self):
        ctx = _ctx()
        ticket, _ = ResearchDigest.upgrade_ticket_or_skip(_gap_claim())
        assert ticket is not None
        pid = UpgradePipeline(ctx).propose_from_ticket(
            ticket, source="research_loop")
        self.assertTrue(pid.startswith("upg"))
        row = UpgradeQueue(ctx).get(pid)
        assert row is not None
        self.assertEqual(row["status"], "proposed")
        self.assertEqual(row["source"], "research_loop")
        self.assertEqual(row["files"],
                         ["nomorals/agents/research_loop.py"])
        # concrete substance survives into the stored proposal
        plan = row["patch_plan"]
        self.assertIn("weekly email digest", plan["behavior"])
        self.assertIn("nomorals/agents/research_loop.py:",
                      plan["gap_evidence"])
        self.assertIn("test_", plan["failing_test"])
        self.assertIn("weekly email digest", plan["problem"])

    def test_proposals_lists_generator_ticket(self):
        # /proposals must return what the generator stores (the
        # empty-/proposals regression: research tickets used to be
        # invisible there).
        from nomorals.agents.partner_runtime import PartnerRuntime

        ctx = _ctx()
        ticket, _ = ResearchDigest.upgrade_ticket_or_skip(_gap_claim())
        assert ticket is not None
        pid = UpgradePipeline(ctx).propose_from_ticket(
            ticket, source="research_loop")
        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = ctx
        rt._autonomy = None
        rt.gateway = SimpleNamespace(send=lambda *a, **k: SimpleNamespace(
            ok=True, error=None))
        rows = rt.proposals()
        self.assertEqual([r["id"] for r in rows], [pid])
        self.assertEqual(rows[0]["queue"], "upgrade")
        self.assertIn("weekly email digest", rows[0]["title"])


class SpecCarryThroughTests(unittest.TestCase):
    def test_claim_from_finding_carries_spec(self):
        finding = SwarmFinding(
            angle="core", claim="ResearchLoop never sends a weekly digest",
            sources=[{"url": "https://example.com/d", "title": "D",
                      "trust": 0.9}],
            confidence=0.85,
            behavior="ResearchLoop publishes a weekly email digest",
            gap="no weekly digest exists",
            target_files=["nomorals/agents/research_loop.py"],
            failing_test="test_weekly_digest_missing_behavior")
        claim = claim_from_finding(finding, "weekly digest")
        self.assertEqual(claim.behavior,
                         "ResearchLoop publishes a weekly email digest")
        self.assertEqual(claim.target_files,
                         ["nomorals/agents/research_loop.py"])
        self.assertEqual(claim.failing_test,
                         "test_weekly_digest_missing_behavior")
        # round-trip through the dict form keeps the spec
        back = ResearchClaim.from_dict(claim.to_dict())
        self.assertEqual(back.behavior, claim.behavior)
        self.assertEqual(back.target_files, claim.target_files)

    def test_dedupe_keeps_spec_from_merged_finding(self):
        from nomorals.agents.research_swarm import ResearchSwarm

        plain = SwarmFinding(
            angle="a", claim="ResearchLoop never sends the owner a "
            "weekly email digest of skipped proposals",
            sources=[{"url": "https://example.com/1", "title": "1",
                      "trust": 0.8}],
            confidence=0.85)
        with_spec = SwarmFinding(
            angle="b", claim="ResearchLoop never sends the owner a "
            "weekly email digest of skipped proposals",
            sources=[{"url": "https://example.com/2", "title": "2",
                      "trust": 0.8}],
            confidence=0.8,
            behavior="ResearchLoop publishes a weekly email digest",
            target_files=["nomorals/agents/research_loop.py"])
        merged = ResearchSwarm.dedupe_findings([plain, with_spec])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].behavior,
                         "ResearchLoop publishes a weekly email digest")
        self.assertEqual(merged[0].target_files,
                         ["nomorals/agents/research_loop.py"])


if __name__ == "__main__":
    unittest.main()
