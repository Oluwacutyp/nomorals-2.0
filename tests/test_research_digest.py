"""Stream 1 acceptance tests: structured research knowledge + digest formats.

Covers research_digest.py:
 1. classify_domain routes to all 7 domains (and honors priority).
 2. claim_from_finding converts a real SwarmFinding (fresh id, domain,
    timestamp, version 1, status active).
 3. check_conflicts flags a negating claim against a seeded active KG claim
    and stays silent on an affirming one (flag only, graph untouched).
 4. promote writes claim/domain/source nodes + links with provenance;
    low-confidence claims are skipped and counted; a higher-confidence
    contradiction supersedes the old node (marked, never deleted).
 5. operator_brief is hard-capped at 20 lines; empty input -> "".
 6. technical_note carries every section.
 7. upgrade_ticket has all keys and non-empty suggested_files per domain.
 8. ResearchPipeline.run works end to end with a FAKE swarm (no network).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import research_digest as rd
from nomorals.agents.kg import KnowledgeGraph
from nomorals.agents.research_digest import (
    ResearchClaim,
    ResearchDigest,
    ResearchPipeline,
    check_conflicts,
    claim_from_finding,
    classify_domain,
    promote,
)
from nomorals.agents.research_swarm import SwarmFinding, SwarmReport
from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _claim(cid, text, confidence=0.8, domain="general", angle="core",
           query="q"):
    return ResearchClaim(id=cid, claim=text, domain=domain, angle=angle,
                         sources=[{"url": "https://example.com/x",
                                   "title": "X", "trust": 0.8}],
                         confidence=confidence, fetched_at=1.0, query=query)


# ── domain classification ──────────────────────────────────────────────────

class ClassifyDomainTests(unittest.TestCase):
    def test_all_seven_domains(self):
        cases = {
            "security": "The new CVE allows remote code execution via a "
                        "buffer overflow exploit",
            "competitors": "Compared to Postgres, MySQL is the main "
                            "alternative; the vs comparison favors Postgres",
            "ml": "Fine-tuning the transformer on a larger dataset reduced "
                  "overfitting during training",
            "systems": "The distributed consensus protocol keeps replication "
                       "lag low across the cluster",
            "tooling": "The CI/CD pipeline lints, builds and deploys the CLI "
                       "in one workflow",
            "product": "Onboarding improvements lifted activation and "
                       "reduced churn in the pricing funnel",
            "general": "Cats are wonderful companions for quiet evenings",
        }
        for domain, text in cases.items():
            self.assertEqual(classify_domain(text), domain, text)

    def test_priority_security_over_competitors(self):
        self.assertEqual(
            classify_domain("The exploit was disclosed by a competitor's "
                            "security research team"),
            "security")

    def test_priority_competitors_over_tooling_and_product(self):
        self.assertEqual(
            classify_domain("CLI tool pricing comparison vs alternatives"),
            "competitors")

    def test_empty_is_general(self):
        self.assertEqual(classify_domain(""), "general")
        self.assertEqual(classify_domain("   "), "general")


# ── claim conversion ───────────────────────────────────────────────────────

class ClaimFromFindingTests(unittest.TestCase):
    def test_conversion(self):
        finding = SwarmFinding(
            angle="core",
            claim="PostgreSQL uses MVCC for concurrency control",
            sources=[{"url": "https://example.com/pg",
                      "title": "PG docs", "trust": 0.8}],
            confidence=0.75,
        )
        claim = claim_from_finding(finding, "postgres internals")
        self.assertTrue(claim.id.startswith("claim_"))
        self.assertEqual(claim.claim, finding.claim)
        self.assertEqual(claim.domain, classify_domain(finding.claim))
        self.assertEqual(claim.angle, "core")
        self.assertEqual(claim.confidence, 0.75)
        self.assertGreater(claim.fetched_at, 0)
        self.assertEqual(claim.query, "postgres internals")
        self.assertEqual(claim.status, "active")
        self.assertEqual(claim.supersedes, "")
        self.assertEqual(claim.version, 1)
        self.assertEqual(claim.sources[0]["url"], "https://example.com/pg")

    def test_round_trip(self):
        claim = _claim("claim_abc", "Some claim text", domain="ml")
        back = ResearchClaim.from_dict(claim.to_dict())
        self.assertEqual(back.to_dict(), claim.to_dict())


# ── conflict detection ─────────────────────────────────────────────────────

class CheckConflictsTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.kg = KnowledgeGraph(self.db)
        self.kg.upsert_node(
            "claim:seed_old", type="claim",
            properties={
                "claim_id": "seed_old",
                "claim": "PostgreSQL is the best relational database for "
                         "analytics workloads",
                "status": "active",
                "confidence": 0.8,
            })

    def test_affirming_claim_no_conflict(self):
        claim = _claim("x1", "PostgreSQL is the best relational database "
                             "for analytics workloads at scale",
                       confidence=0.7)
        self.assertEqual(check_conflicts([claim], self.kg), [])

    def test_negating_claim_flagged(self):
        claim = _claim("x2", "PostgreSQL is not suitable for analytics "
                             "workloads",
                       confidence=0.7)
        conflicts = check_conflicts([claim], self.kg)
        self.assertEqual(len(conflicts), 1)
        c = conflicts[0]
        self.assertEqual(c["kind"], "contradiction")
        self.assertEqual(c["claim_id"], "x2")
        self.assertIn("PostgreSQL", c["existing_claim"])
        self.assertEqual(c["existing_claim_id"], "seed_old")

    def test_flag_only_graph_untouched(self):
        claim = _claim("x2", "PostgreSQL is not suitable for analytics "
                             "workloads")
        before = self.kg.stats()
        check_conflicts([claim], self.kg)
        after = self.kg.stats()
        self.assertEqual(before, after)
        node = self.kg.find_node("claim:seed_old", type="claim")
        self.assertEqual(node.properties.get("status"), "active")


# ── promotion ──────────────────────────────────────────────────────────────

class PromoteTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.c1 = _claim("c1", "Redis is the fastest in-memory cache for "
                               "session storage",
                         confidence=0.85, domain="systems")
        self.c2 = _claim("c2", "Unimportant low-signal rumor",
                         confidence=0.2, domain="general")
        self.c3 = _claim("c3", "Redis is not suitable as a session store; "
                               "it loses data on restart",
                         confidence=0.9, domain="systems")

    def test_promote_nodes_links_provenance(self):
        result = promote([self.c1], self.db)
        self.assertEqual(result["promoted"], 1)
        self.assertEqual(result["skipped_low_confidence"], 0)
        kg = KnowledgeGraph(self.db)

        node = kg.find_node("claim:c1", type="claim")
        self.assertIsNotNone(node)
        self.assertEqual(node.type, "claim")  # not coerced to "entity"
        props = node.properties
        for key in ("claim", "domain", "angle", "confidence", "fetched_at",
                    "query", "sources", "version", "status"):
            self.assertIn(key, props, key)
        self.assertEqual(props["status"], "active")
        self.assertIn("redis", props["claim"].lower())

        domain_node = kg.find_node("domain:systems", type="domain")
        self.assertIsNotNone(domain_node)
        about = kg.query(subject="claim:c1", relation="about")
        self.assertEqual(len(about), 1)
        self.assertAlmostEqual(about[0].weight, 1.5)

        src_node = kg.find_node("source:https://example.com/x",
                                type="source")
        self.assertIsNotNone(src_node)
        cites = kg.query(subject="claim:c1", relation="cites")
        self.assertEqual(len(cites), 1)

    def test_low_confidence_skipped_and_counted(self):
        result = promote([self.c1, self.c2], self.db)
        self.assertEqual(result["promoted"], 1)
        self.assertEqual(result["skipped_low_confidence"], 1)
        kg = KnowledgeGraph(self.db)
        self.assertIsNone(kg.find_node("claim:c2", type="claim"))

    def test_supersede_marks_old_never_deletes(self):
        promote([self.c1], self.db)
        result = promote([self.c3], self.db)
        self.assertEqual(result["superseded"], 1)
        self.assertEqual(len(result["conflicts"]), 1)
        kg = KnowledgeGraph(self.db)

        old = kg.find_node("claim:c1", type="claim")
        self.assertIsNotNone(old)  # never deleted
        self.assertEqual(old.properties.get("status"), "superseded")
        self.assertEqual(old.properties.get("superseded_by"), "c3")

        new = kg.find_node("claim:c3", type="claim")
        self.assertEqual(new.properties.get("status"), "active")
        edges = kg.query(subject="claim:c3", relation="supersedes")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].dst, old.id)

    def test_weaker_contradiction_marks_new_contradicted(self):
        promote([self.c1], self.db)  # 0.85
        weak = _claim("c4", "Redis is not suitable as a session store; "
                            "it loses data on restart",
                      confidence=0.5, domain="systems")
        result = promote([weak], self.db)
        self.assertEqual(result["superseded"], 0)
        self.assertEqual(len(result["conflicts"]), 1)
        kg = KnowledgeGraph(self.db)
        node = kg.find_node("claim:c4", type="claim")
        self.assertEqual(node.properties.get("status"), "contradicted")
        old = kg.find_node("claim:c1", type="claim")
        self.assertEqual(old.properties.get("status"), "active")


# ── digests ────────────────────────────────────────────────────────────────

def _report(n_findings=12):
    findings = [
        SwarmFinding(
            angle=f"angle{i % 3}",
            claim=("Finding number %d: " % i) + "PostgreSQL improves "
                  "analytical query planning with parallel workers " * 3,
            sources=[{"url": f"https://example.com/{i}",
                      "title": f"Source {i}", "trust": 0.8}],
            confidence=0.9 - i * 0.02,
        )
        for i in range(n_findings)
    ]
    return SwarmReport(
        query="postgres analytics",
        angles=["a1", "a2", "a3"],
        findings=findings,
        synthesis="Line one of synthesis.\nLine two of synthesis.\n"
                  "Line three of synthesis.\nLine four of synthesis.",
        conflicts=["sources disagree on X vs Y"],
        failed_angles=["recent developments"],
        critique="solid but thin on primary sources",
    )


class OperatorBriefTests(unittest.TestCase):
    def test_hard_cap_20_lines(self):
        brief = ResearchDigest.operator_brief(_report(12))
        self.assertTrue(brief)
        self.assertLessEqual(len(brief.splitlines()), 20)

    def test_empty_inputs(self):
        self.assertEqual(ResearchDigest.operator_brief([]), "")
        self.assertEqual(
            ResearchDigest.operator_brief(SwarmReport(query="q", angles=[])),
            "")

    def test_claim_list_input(self):
        claims = [_claim("b1", "First claim about caching", confidence=0.9),
                  _claim("b2", "Second claim about caching", confidence=0.4)]
        brief = ResearchDigest.operator_brief(claims)
        self.assertTrue(brief)
        self.assertLessEqual(len(brief.splitlines()), 20)
        self.assertIn("90%", brief)


class TechnicalNoteTests(unittest.TestCase):
    def test_all_sections(self):
        note = ResearchDigest.technical_note(_report(3))
        for section in ("OBJECTIVE", "KEY FINDINGS", "CONFLICTS",
                        "GAPS / FAILED ANGLES", "CRITIQUE"):
            self.assertIn(section, note, section)
        self.assertIn("https://example.com/0", note)  # source URLs
        self.assertIn("postgres analytics", note)
        self.assertNotIn("|", note.split("KEY FINDINGS")[1].split(
            "CONFLICTS")[0].replace(" | sources:", ""))  # no markdown tables


class UpgradeTicketTests(unittest.TestCase):
    def test_all_keys_and_suggested_files(self):
        for domain in ("security", "competitors", "ml", "systems",
                       "tooling", "product", "general"):
            claim = _claim("t1", "A claim worth acting on " * 6,
                           domain=domain, confidence=0.82)
            ticket = ResearchDigest.upgrade_ticket(claim)
            for key in ("title", "rationale", "domain", "suggested_files",
                        "test_plan", "patch_plan", "claim_ids", "confidence"):
                self.assertIn(key, ticket, f"{domain}:{key}")
            self.assertEqual(ticket["domain"], domain)
            self.assertTrue(ticket["suggested_files"], domain)
            self.assertTrue(all(isinstance(f, str) and f.startswith(
                "nomorals/") for f in ticket["suggested_files"]), domain)
            self.assertTrue(ticket["test_plan"], domain)
            self.assertTrue(ticket["patch_plan"], domain)
            self.assertEqual(ticket["claim_ids"], ["t1"])
            self.assertEqual(ticket["confidence"], 0.82)

    def test_suggested_files_marked_heuristic(self):
        ticket = ResearchDigest.upgrade_ticket(
            _claim("t2", "Some security claim about exploits",
                   domain="security"))
        self.assertIn("heuristic", ticket["suggested_files_basis"].lower())


# ── pipeline (fake swarm, no network) ──────────────────────────────────────

class _FakeSwarm:
    """Stand-in for ResearchSwarm: deterministic report, no network."""

    def __init__(self, context, specialists=None, **kw):
        self.context = context
        self.specialists = specialists

    def run(self, query, **kw):
        return SwarmReport(
            query=query,
            angles=["core", "critical"],
            findings=[
                SwarmFinding(
                    angle="core",
                    claim="PostgreSQL is the best relational database for "
                          "analytics workloads",
                    sources=[{"url": "https://example.com/pg",
                              "title": "PG", "trust": 0.9}],
                    confidence=0.85),
                SwarmFinding(
                    angle="critical",
                    claim="PostgreSQL is not suitable for analytics "
                          "workloads",
                    sources=[{"url": "https://example.com/no",
                              "title": "NoPG", "trust": 0.7}],
                    confidence=0.6),
            ],
            synthesis="PostgreSQL dominates analytics chatter.",
            conflicts=[],
        )


class PipelineTests(unittest.TestCase):
    def _run(self, **kw):
        db = _db()
        ctx = SimpleNamespace(db=db, router=None, extras={})
        with mock.patch.object(rd, "ResearchSwarm", _FakeSwarm):
            return ResearchPipeline.run("postgres analytics", ctx, **kw), db

    def test_full_pipeline(self):
        result, db = self._run(specialists=["ml"])
        self.assertEqual(result["query"], "postgres analytics")
        self.assertEqual(len(result["claims"]), 2)
        self.assertEqual(len(result["report"]["findings"]), 2)

        promo = result["promotion"]
        self.assertEqual(promo["promoted"], 2)
        self.assertEqual(promo["skipped_low_confidence"], 0)
        self.assertEqual(promo["superseded"], 0)
        # same-batch conflict: weaker claim contradicts the stronger one
        self.assertEqual(len(promo["conflicts"]), 1)
        self.assertEqual(promo["conflicts"][0]["kind"], "contradiction")
        by_conf = sorted(result["claims"], key=lambda c: -c["confidence"])
        self.assertEqual(by_conf[1]["status"], "contradicted")

        # claims really landed in the KG
        kg = KnowledgeGraph(db)
        self.assertGreater(kg.stats()["nodes"], 0)

        # digests
        self.assertTrue(result["brief"])
        self.assertLessEqual(len(result["brief"].splitlines()), 20)
        self.assertIn("OBJECTIVE", result["note"])
        self.assertEqual(len(result["tickets"]), 2)
        self.assertFalse(result["notified"])

    def test_min_confidence_filters(self):
        result, _ = self._run(min_confidence=0.9)
        self.assertEqual(result["promotion"]["promoted"], 0)
        self.assertEqual(result["promotion"]["skipped_low_confidence"], 2)
        self.assertEqual(result["tickets"], [])

    def test_promote_claims_false(self):
        result, db = self._run(promote_claims=False)
        promo = result["promotion"]
        self.assertEqual(promo["promoted"], 0)
        self.assertIn("skipped_reason", promo)
        self.assertEqual(KnowledgeGraph(db).stats()["nodes"], 0)
        # conflicts still detected against an empty graph -> none
        self.assertEqual(result["conflicts"], [])


if __name__ == "__main__":
    unittest.main()
