"""Stream 4 acceptance tests: specialist research-swarm workers and the
morning-briefing research provider.

Swarm:
 1. ``specialists=["security", "ml"]`` decomposes ``postgres`` into
    domain-specific angles, capped at the worker count, in domain order.
 2. Unknown specialist domains raise ``ValueError`` (fail fast).
 3. No specialists -> the old generic template behavior is unchanged.
 4. ``_angle_domains`` stays aligned with the returned angles.
 5. Findings carry the domain through ``run()`` (stubbed network), and
    ``SwarmFinding.to_dict()`` includes it.

Briefing provider:
 6. Canned proposals -> a section naming the proposal titles.
 7. Empty queue (nothing pending, nothing recently decided) -> None.
 8. Broken queue import -> None (the briefing must never sink).
 9. ``ResearchProvider`` is registered on the composer with priority
    between RoomsProvider (60) and DevonSelfProvider (100).
"""

from __future__ import annotations

import sys
import time
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.research_swarm import (
    SPECIALIST_DOMAINS, ResearchSwarm, SwarmFinding,
)
from nomorals.agents.morning_briefing import (
    BriefingComposer, DevonSelfProvider, ResearchProvider, RoomsProvider,
)


def _ctx():
    return SimpleNamespace(db=None, tools=None, router=None,
                           settings=None, extras={})


# ── specialist angle generation ────────────────────────────────────────────

class SpecialistAngleTests(unittest.TestCase):
    def test_domains_table_shape(self):
        self.assertEqual(set(SPECIALIST_DOMAINS),
                         {"systems", "security", "product", "ml",
                          "tooling", "competitors"})
        for domain, entry in SPECIALIST_DOMAINS.items():
            self.assertIn(3 <= len(entry["angles"]) <= 4, (True,), domain)
            self.assertTrue(all("{q}" in t for t in entry["angles"]),
                            f"{domain}: every template needs the {{q}} placeholder")
            self.assertTrue(8 <= len(entry["keywords"]) <= 12,
                            f"{domain}: 8-12 keywords")

    def test_specialist_angles_are_domain_specific(self):
        swarm = ResearchSwarm(_ctx(), workers=4,
                             specialists=["security", "ml"])
        angles = swarm.angles_for("postgres")
        self.assertEqual(
            angles,
            ["postgres threat model vulnerabilities",
             "postgres security best practices hardening",
             "postgres CVE exploit mitigations",
             "postgres authentication authorization access control"])
        self.assertEqual(swarm._angle_domains, ["security"] * 4)

    def test_domain_order_and_worker_cap(self):
        swarm = ResearchSwarm(_ctx(), workers=6,
                             specialists=["security", "ml"])
        angles = swarm.angles_for("postgres")
        self.assertEqual(len(angles), 6)
        # security fills first (4 templates), then ml
        self.assertEqual(swarm._angle_domains,
                         ["security"] * 4 + ["ml"] * 2)
        self.assertTrue(angles[4].startswith("postgres "))

    def test_more_workers_than_templates_uses_all(self):
        swarm = ResearchSwarm(_ctx(), workers=6, specialists=["product"])
        angles = swarm.angles_for("postgres")
        self.assertEqual(len(angles), 4)  # product has 4 templates
        self.assertEqual(swarm._angle_domains, ["product"] * 4)

    def test_unknown_domain_raises(self):
        with self.assertRaises(ValueError):
            ResearchSwarm(_ctx(), specialists=["security", "astrology"])

    def test_no_specialists_keeps_old_templates(self):
        swarm = ResearchSwarm(_ctx(), workers=4)
        self.assertEqual(
            swarm.angles_for("postgres"),
            ["postgres",
             "postgres best practices how to",
             "postgres criticism problems limitations",
             "postgres recent developments"])
        self.assertEqual(swarm._angle_domains, [""] * 4)

    def test_empty_query_still_rejected(self):
        swarm = ResearchSwarm(_ctx(), specialists=["security"])
        with self.assertRaises(ValueError):
            swarm.angles_for("")


# ── domain tagging through run() ───────────────────────────────────────────

class DomainTaggingTests(unittest.TestCase):
    def _stubbed(self, **kwargs):
        swarm = ResearchSwarm(_ctx(), **kwargs)

        def fake_research(angle, worker_index, domain=""):
            return {
                "angle": angle, "ok": True,
                "findings": [SwarmFinding(
                    angle=angle, claim=f"finding on {angle}",
                    sources=[{"url": "https://example.com/x",
                              "title": "example", "trust": 0.7}],
                    confidence=0.6, domain=domain)],
                "sources": [{"url": "https://example.com/x",
                             "title": "example", "trust": 0.7}],
                "pages_read": 0, "seconds": 0.01,
            }

        swarm._research_angle = fake_research  # noqa: SLF001 - stub the network
        return swarm

    def test_run_tags_findings_with_domains(self):
        swarm = self._stubbed(workers=4, specialists=["security", "ml"])
        report = swarm.run("postgres")
        self.assertEqual(len(report.findings), 4)
        domains = [f.domain for f in report.findings]
        self.assertEqual(domains, ["security"] * 4)
        # the report payload carries domains too
        for f in report.to_dict()["findings"]:
            self.assertIn("domain", f)

    def test_run_without_specialists_tags_blank(self):
        swarm = self._stubbed(workers=2)
        report = swarm.run("postgres")
        self.assertTrue(all(f.domain == "" for f in report.findings))

    def test_explicit_angles_reset_domains(self):
        # caller-supplied angles have no known domain — never leak a
        # stale mapping from an earlier angles_for() call
        swarm = self._stubbed(workers=2, specialists=["security"])
        swarm.angles_for("postgres")
        self.assertTrue(swarm._angle_domains)
        report = swarm.run("postgres", angles=["a", "b"])
        self.assertEqual(swarm._angle_domains, ["", ""])
        self.assertTrue(all(f.domain == "" for f in report.findings))

    def test_finding_to_dict_includes_domain(self):
        d = SwarmFinding(angle="a", claim="c", domain="security").to_dict()
        self.assertEqual(d["domain"], "security")
        d2 = SwarmFinding(angle="a", claim="c").to_dict()
        self.assertEqual(d2["domain"], "")


# ── briefing provider ──────────────────────────────────────────────────────

def _queue_module(proposed=(), approved=(), denied=(), ctor=None):
    """A fake ``nomorals.agents.upgrade_queue`` module for import patching."""
    mod = types.ModuleType("nomorals.agents.upgrade_queue")

    class FakeQueue:
        def __init__(self, *args, **kwargs):
            if ctor == "raise":
                raise RuntimeError("cannot construct")
            if ctor == "type-error":
                raise TypeError("bad signature")

        def list(self, status="proposed", limit=50):
            data = {"proposed": proposed, "approved": approved,
                    "denied": denied}
            return [dict(r) for r in data.get(status, [])][:limit]

    mod.UpgradeQueue = FakeQueue
    return mod


class ResearchProviderTests(unittest.TestCase):
    def _collect(self, module, ctx=None):
        with mock.patch.dict(sys.modules,
                             {"nomorals.agents.upgrade_queue": module}):
            return ResearchProvider().collect(ctx or _ctx(), time.time())

    def test_proposals_become_a_section(self):
        mod = _queue_module(proposed=[
            {"id": "u1", "title": "Add GPU memory cap",
             "summary": "llama-server OOMs on 12GB phones"},
            {"id": "u2", "title": "Retry webhooks",
             "summary": "at-least-once delivery"},
        ])
        section = self._collect(mod)
        self.assertIsNotNone(section)
        self.assertEqual(section.name, "research")
        text = "\n".join(section.lines)
        self.assertIn("2 upgrade proposal(s) await your review", text)
        self.assertIn("Add GPU memory cap", text)
        self.assertIn("Retry webhooks", text)

    def test_recently_decided_counts(self):
        now = time.time()
        mod = _queue_module(approved=[
            {"id": "u3", "title": "Cache search results",
             "decided_at": now - 3600},
        ], denied=[
            {"id": "u4", "title": "Drop sqlite",
             "decided_at": now - 7200},
        ])
        section = self._collect(mod)
        self.assertIsNotNone(section)
        text = "\n".join(section.lines)
        self.assertIn("approved: 1 (implementing)", text)
        self.assertIn("denied: 1 (rejected)", text)
        self.assertIn("Cache search results", text)

    def test_stale_decisions_are_dropped(self):
        # decided 3 days ago -> older than the 24h window
        mod = _queue_module(approved=[
            {"id": "u5", "title": "Ancient approval",
             "decided_at": time.time() - 3 * 86400},
        ])
        self.assertIsNone(self._collect(mod))

    def test_empty_queue_returns_none(self):
        self.assertIsNone(self._collect(_queue_module()))

    def test_broken_import_returns_none(self):
        # module exists but has no UpgradeQueue -> ImportError path
        bare = types.ModuleType("nomorals.agents.upgrade_queue")
        self.assertIsNone(self._collect(bare))

    def test_ctor_failure_returns_none(self):
        self.assertIsNone(self._collect(_queue_module(ctor="raise")))

    def test_list_failure_returns_none(self):
        mod = types.ModuleType("nomorals.agents.upgrade_queue")

        class BadQueue:
            def __init__(self, *a, **k):
                pass

            def list(self, status="proposed", limit=50):
                raise RuntimeError("table missing")

        mod.UpgradeQueue = BadQueue
        self.assertIsNone(self._collect(mod))

    def test_registration_and_priority(self):
        composer = BriefingComposer()
        provider = next((p for p in composer.providers
                         if p.name == "research"), None)
        self.assertIsNotNone(provider, "research provider must be registered")
        self.assertIsInstance(provider, ResearchProvider)
        self.assertGreater(provider.priority, RoomsProvider.priority)
        self.assertLess(provider.priority, DevonSelfProvider.priority)

    def test_composer_still_composes_with_empty_queue(self):
        # end-to-end: with an empty queue the briefing builds and simply
        # has no research section
        with mock.patch.dict(sys.modules,
                             {"nomorals.agents.upgrade_queue":
                              _queue_module()}):
            briefing = BriefingComposer().compose(_ctx(), "2026-10-01",
                                                 since=0.0)
        names = [s.name for s in briefing.sections]
        self.assertNotIn("research", names)


if __name__ == "__main__":
    unittest.main()
