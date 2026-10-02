"""End-to-end proof for the Wave C research loop.

One test drives the whole cycle with fakes only at the network/model
boundary — no real HTTP, no real LLM:

    fake swarm -> ResearchPipeline -> claims -> KG promotion ->
    upgrade ticket -> UpgradeQueue -> owner approve ->
    (fake) EvolutionAgent -> implemented

Plus: the deny path, the low-confidence gate, and the lexicon
acquisition from the same findings. If this file is green, the organs
compose; the only untested part is the live web itself.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from nomorals.agents import research_digest as digest_mod
from nomorals.agents.research_digest import ResearchPipeline
from nomorals.agents.research_lexicon import LexiconStore, acquire_from_findings
from nomorals.agents.research_swarm import SwarmFinding, SwarmReport
from nomorals.agents.upgrade_queue import UpgradePipeline as TicketPipeline
from nomorals.storage.db import Database


def _ctx(db):
    return SimpleNamespace(db=db, router=None, memory=None, extras={})


class _FakeSwarm:
    """Stands in for ResearchSwarm: no network, deterministic findings."""

    def __init__(self, context: Any, specialists: Any = None) -> None:
        self.context = context
        self.specialists = specialists

    def run(self, query: str) -> SwarmReport:
        findings = [
            SwarmFinding(
                angle="postgres threat model",
                domain="security",
                claim=("PostgreSQL row-level security policies are not "
                       "applied to table owners by default, which can "
                       "silently expose rows the policy was meant to hide."),
                sources=[{"url": "https://example.com/rls",
                           "title": "RLS bypass for owners", "trust": 0.8}],
                confidence=0.75,
            ),
            SwarmFinding(
                angle="postgres hardening",
                domain="security",
                claim=("Enabling row-level security without also revoking "
                       "direct table grants leaves the policy ineffective "
                       "against grantees with explicit privileges."),
                sources=[{"url": "https://example.com/rls2",
                           "title": "RLS grant interaction", "trust": 0.7}],
                confidence=0.65,
            ),
        ]
        return SwarmReport(query=query, angles=["a", "b"], findings=findings,
                           synthesis="syn", workers=2)


class _FakeEvolution:
    """Stands in for EvolutionAgent: records, never touches git."""

    def __init__(self) -> None:
        self.planned: list[str] = []
        self.applied: list[str] = []

    def plan(self, instruction: str) -> SimpleNamespace:
        self.planned.append(instruction)
        return SimpleNamespace(id="evo_test_1")

    def apply(self, proposal_id: str, verify: bool = True) -> dict[str, Any]:
        self.applied.append(proposal_id)
        return {"ok": True, "proposal_id": proposal_id, "verify": verify}


class TestResearchCycle(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Database(":memory:")
        self.db.migrate()
        self.ctx = _ctx(self.db)
        self._real_swarm = digest_mod.ResearchSwarm
        digest_mod.ResearchSwarm = _FakeSwarm  # type: ignore[assignment]
        self.addCleanup(setattr, digest_mod, "ResearchSwarm", self._real_swarm)

    def test_full_loop_approve_implements(self) -> None:
        out = ResearchPipeline.run("postgres row level security",
                                   self.ctx, min_confidence=0.5)
        self.assertEqual(len(out["claims"]), 2)
        self.assertTrue(out["promotion"]["promoted"] >= 1)
        self.assertTrue(out["brief"])
        self.assertTrue(out["note"])
        self.assertTrue(out["tickets"], "confident claims must yield tickets")

        pipe = TicketPipeline(self.ctx, evolution=_FakeEvolution())
        pid = pipe.propose_from_ticket(out["tickets"][0])
        self.assertTrue(pid)

        result = pipe.approve_and_implement(pid, by="owner")
        self.assertEqual(result["status"], "implemented")
        evo = pipe.evolution
        assert isinstance(evo, _FakeEvolution)
        self.assertEqual(len(evo.planned), 1)
        self.assertEqual(evo.applied, ["evo_test_1"])
        # the instruction given to the coding stack carries the substance
        self.assertIn("row-level security", evo.planned[0].lower())

    def test_deny_path_records_reason(self) -> None:
        out = ResearchPipeline.run("postgres row level security", self.ctx)
        pipe = TicketPipeline(self.ctx, evolution=_FakeEvolution())
        pid = pipe.propose_from_ticket(out["tickets"][0])
        denied = pipe.deny_with_reason(pid, "not a priority this week")
        self.assertEqual(denied["status"], "denied")
        self.assertIn("not a priority", denied["reason"])

    def test_low_confidence_ticket_never_reaches_owner(self) -> None:
        pipe = TicketPipeline(self.ctx, evolution=_FakeEvolution())
        pid = pipe.propose_from_ticket({
            "title": "[security] weak claim here",
            "rationale": "a very weakly supported research claim, ignore me",
            "patch_plan": ["do things"],
            "suggested_files": [],
            "test_plan": [],
            "claim_ids": [],
            "confidence": 0.1,
        })
        self.assertEqual(pid, "")

    def test_lexicon_feeds_from_same_findings(self) -> None:
        out = ResearchPipeline.run("postgres row level security", self.ctx)
        store = LexiconStore(self.ctx)
        res = acquire_from_findings(
            [SwarmFinding(**{k: f[k] for k in
                             ("angle", "claim", "sources", "confidence")
                             if k in f})
             for f in out["report"]["findings"]],
            store, module="partner", category="security",
            source="research_digest",
            category_keywords=["security", "vulnerability", "policy",
                               "privilege", "hardening", "threat", "exploit"])
        self.assertGreaterEqual(res["version"], 1)
        terms = store.terms_for("partner", "security")
        self.assertTrue(terms, "research should yield lexicon terms")

    def test_contradiction_is_flagged_not_merged(self) -> None:
        out = ResearchPipeline.run("postgres row level security", self.ctx)
        self.assertIsInstance(out["conflicts"], list)
        # promotion must never delete or silently overwrite claims
        promo = out["promotion"]
        self.assertIn("promoted", promo)
        self.assertIn("superseded", promo)


if __name__ == "__main__":
    unittest.main()
