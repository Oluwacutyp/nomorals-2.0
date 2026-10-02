"""Tests for nomorals/skills/bench.py and nomorals/skills/repair.py."""

from __future__ import annotations

import unittest

from nomorals.skills.bench import SkillBench
from nomorals.skills.repair import (RepairTicket, RepairTicketStore,
                                    build_ticket, suggest_fix)
from nomorals.storage.db import Database


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.bench = SkillBench(self.db)

    def test_record_and_score(self):
        self.bench.record("s", "1.0.0", success=True, latency_ms=10.0)
        self.bench.record("s", "1.0.0", success=True, latency_ms=30.0)
        self.bench.record("s", "1.0.0", success=False, latency_ms=20.0)
        score = self.bench.score("s")
        self.assertEqual(score["skill"], "s")
        self.assertEqual(score["runs"], 3)
        self.assertAlmostEqual(score["success_rate"], 2 / 3, places=3)
        self.assertAlmostEqual(score["avg_latency_ms"], 20.0)
        self.assertAlmostEqual(score["min_latency_ms"], 10.0)
        self.assertAlmostEqual(score["max_latency_ms"], 30.0)
        self.assertIsNotNone(score["last_run_at"])

    def test_score_empty_history_is_zeroed(self):
        score = self.bench.score("never-ran")
        self.assertEqual(score["runs"], 0)
        self.assertEqual(score["success_rate"], 0.0)
        self.assertEqual(score["avg_latency_ms"], 0.0)
        self.assertIsNone(score["last_run_at"])

    def test_score_scopes_to_version(self):
        self.bench.record("s", "1.0.0", success=True, latency_ms=5.0)
        self.bench.record("s", "2.0.0", success=False, latency_ms=5.0)
        v1 = self.bench.score("s", version="1.0.0")
        v2 = self.bench.score("s", version="2.0.0")
        self.assertEqual(v1["success_rate"], 1.0)
        self.assertEqual(v2["success_rate"], 0.0)
        all_versions = self.bench.score("s")
        self.assertEqual(all_versions["runs"], 2)
        self.assertIn("1.0.0", all_versions["versions"])
        self.assertIn("2.0.0", all_versions["versions"])

    def test_recent_and_overview(self):
        self.bench.record("a", "1.0.0", success=True, latency_ms=1.0)
        self.bench.record("b", "1.0.0", success=False, latency_ms=2.0)
        recent = self.bench.recent(limit=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["skill"], "b")  # newest first
        overview = self.bench.overview()
        self.assertEqual({s["skill"] for s in overview}, {"a", "b"})

    def test_record_never_raises(self):
        # bench bookkeeping must not sink the run it measures
        self.bench.record("s", "", success=True, latency_ms=-5.0)
        self.assertEqual(self.bench.score("s")["runs"], 1)


class RepairTicketTests(unittest.TestCase):
    def test_ticket_carries_name_version_step_inputs_error(self):
        ticket = build_ticket(
            "chain", "1.0.0", step_index=1, tool="wrap",
            step_input={"shouted": "HI"}, error="boom happened")
        self.assertEqual(ticket.skill_name, "chain")
        self.assertEqual(ticket.version, "1.0.0")
        self.assertEqual(ticket.step_index, 1)
        self.assertEqual(ticket.tool, "wrap")
        self.assertEqual(ticket.inputs, {"shouted": "HI"})
        self.assertEqual(ticket.error, "boom happened")
        self.assertTrue(ticket.id)
        self.assertTrue(ticket.suggested_fix)

    def test_missing_key_fix_is_concrete(self):
        fix = suggest_fix(step_index=2, tool="render",
                          error="output: missing required key 'html'",
                          missing_keys=["html"])
        self.assertIn("step 2", fix)
        self.assertIn("'html'", fix)
        self.assertIn("tighten", fix)

    def test_unknown_tool_fix_names_close_match(self):
        fix = suggest_fix(step_index=0, tool="wrapp",
                          error="unknown tool 'wrapp'",
                          available_tools=["wrap", "shout"])
        self.assertIn("'wrapp'", fix)
        self.assertIn("wrap", fix)
        self.assertIn("manifest's tools list", fix)

    def test_wiring_failure_fix_names_the_mechanism(self):
        fix = suggest_fix(
            step_index=1, tool="wrap",
            error="could not resolve wiring: '$input.tag' at step 1: "
                  "key 'tag' not present in the referenced output")
        self.assertIn("$input.<key>", fix)
        self.assertIn("$<n>.<key>", fix)

    def test_disabled_fix_names_the_command(self):
        fix = suggest_fix(step_index=-2, tool="", error="",
                          disabled=True, skill_name="chain")
        self.assertIn("nm skill enable 'chain'", fix)

    def test_unknown_skill_fix_suggests_install(self):
        fix = suggest_fix(step_index=-2, tool="", error="",
                          unknown_skill=True, skill_name="chan",
                          available_tools=["chain", "other"])
        self.assertIn("no installed skill named 'chan'", fix)
        self.assertIn("chain", fix)  # close-match hint
        self.assertIn("nm skill install", fix)

    def test_generic_failure_fix_quotes_error(self):
        fix = suggest_fix(step_index=0, tool="shout",
                          error="connection reset by peer",
                          step_input={"text": "hi"}, skill_name="chain")
        self.assertIn("step 0 ('shout')", fix)
        self.assertIn("connection reset by peer", fix)


class RepairTicketStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.store = RepairTicketStore(self.db)

    def test_save_get_list_round_trip(self):
        ticket = build_ticket("chain", "1.0.0", step_index=0, tool="shout",
                              step_input={"text": "hi"}, error="kaput")
        self.store.save(ticket)
        fetched = self.store.get(ticket.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.to_dict(), ticket.to_dict())
        listed = self.store.list("chain")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].id, ticket.id)
        self.assertEqual(self.store.count("chain"), 1)
        self.assertEqual(self.store.count(), 1)

    def test_list_scopes_to_skill(self):
        self.store.save(build_ticket("a", "1.0.0", step_index=0, error="x"))
        self.store.save(build_ticket("b", "1.0.0", step_index=0, error="y"))
        self.assertEqual(len(self.store.list("a")), 1)
        self.assertEqual(len(self.store.list()), 2)

    def test_get_unknown_returns_none(self):
        self.assertIsNone(self.store.get("no-such-ticket"))

    def test_ticket_dataclass_serializes(self):
        ticket = RepairTicket(id="rt1", skill_name="s", version="1.0.0",
                              step_index=-1, tool="", inputs={"k": "v"},
                              error="e", suggested_fix="fix it")
        data = ticket.to_dict()
        self.assertEqual(data["inputs"], {"k": "v"})
        self.assertEqual(data["suggested_fix"], "fix it")


if __name__ == "__main__":
    unittest.main()
