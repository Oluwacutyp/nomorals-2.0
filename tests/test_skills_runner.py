"""Tests for nomorals/skills/runner.py — real dispatch, data threading."""

from __future__ import annotations

import unittest

from nomorals.skills.bench import SkillBench
from nomorals.skills.registry import SkillRegistry
from nomorals.skills.repair import RepairTicketStore
from nomorals.skills.runner import SkillRunner
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


def _shout(text):
    return {"shouted": text.upper()}


def _wrap(shouted, tag="x"):
    return {"wrapped": "<%s>%s</%s>" % (tag, shouted, tag)}


def _boom(text):
    raise RuntimeError("simulated tool explosion")


def _chain_manifest(name="chain", **overrides):
    data = {
        "name": name,
        "version": "1.0.0",
        "tools": ["shout", "wrap"],
        "input_schema": {"text": "str", "tag": "str?"},
        "output_schema": {"wrapped": "str"},
        "wiring": [{"text": "$input.text"},
                   {"shouted": "$0.shouted", "tag": "$input.tag"}],
    }
    data.update(overrides)
    return data


class RunnerHarness(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.registry = SkillRegistry(self.db)
        self.bench = SkillBench(self.db)
        self.tickets = RepairTicketStore(self.db)
        # Real ToolRegistry, real dispatch path (call → capability check →
        # audit).  enforce=False keeps this offline unit test hermetic.
        self.tools = ToolRegistry(enforce=False)
        self.tools.register("shout", _shout, capability="")
        self.tools.register("wrap", _wrap, capability="")
        self.runner = SkillRunner(self.registry, self.tools,
                                  bench=self.bench, tickets=self.tickets)

    def install(self, manifest):
        return self.registry.install(manifest)


class RunnerChainTests(RunnerHarness):
    def test_two_tool_chain_threads_data_through_real_dispatch(self):
        self.install(_chain_manifest())
        result = self.runner.run("chain", {"text": "hello", "tag": "b"})
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.steps), 2)
        # step 0 got exactly its wired input through the real dispatch
        self.assertEqual(result.steps[0].input, {"text": "hello"})
        self.assertEqual(result.steps[0].output, {"shouted": "HELLO"})
        # step 1 threaded step 0's output plus the run input
        self.assertEqual(result.steps[1].input,
                         {"shouted": "HELLO", "tag": "b"})
        self.assertEqual(result.outputs["1"],
                         {"wrapped": "<b>HELLO</b>"})
        self.assertIsNone(result.failed_step)
        self.assertGreaterEqual(result.latency_ms, 0.0)

    def test_audit_trail_records_both_calls(self):
        self.install(_chain_manifest())
        self.runner.run("chain", {"text": "hello", "tag": "b"})
        trail = self.tools.audit_trail(tool="shout")
        self.assertTrue(trail)
        self.assertEqual(trail[-1]["tool"], "shout")
        self.assertEqual(trail[-1]["status"], "ok")
        self.assertEqual(self.tools.audit_trail(tool="wrap")[-1]["status"],
                         "ok")

    def test_step_without_wiring_gets_raw_input(self):
        self.tools.register("echo", lambda **kw: {"got": kw}, capability="")
        self.install({"name": "passthru", "version": "1.0.0",
                      "tools": ["echo"],
                      "output_schema": {"got": "dict"}})
        result = self.runner.run("passthru", {"a": 1, "b": 2})
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.steps[0].input, {"a": 1, "b": 2})

    def test_last_reference_threads_previous_output(self):
        self.tools.register("again", lambda shouted: {"twice": shouted * 2},
                            capability="")
        self.install({"name": "lastref", "version": "1.0.0",
                      "tools": ["shout", "again"],
                      "input_schema": {"text": "str"},
                      "output_schema": {"twice": "str"},
                      "wiring": [{"text": "$input.text"},
                                 {"shouted": "$last.shouted"}]})
        result = self.runner.run("lastref", {"text": "yo"})
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.outputs["1"], {"twice": "YOYO"})

    def test_unknown_skill_fails_with_ticket(self):
        result = self.runner.run("ghost", {"text": "hi"})
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_step, -2)
        self.assertIn("unknown skill", result.error)
        self.assertIsNotNone(result.ticket)
        self.assertIn("ghost", result.ticket.suggested_fix)
        self.assertEqual(self.tickets.count("ghost"), 1)

    def test_disabled_skill_is_gated(self):
        self.install(_chain_manifest())
        self.registry.disable("chain")
        result = self.runner.run("chain", {"text": "hello", "tag": "b"})
        self.assertFalse(result.ok)
        self.assertIn("disabled", result.error)
        self.assertIsNotNone(result.ticket)
        self.assertIn("nm skill enable", result.ticket.suggested_fix)
        # no tool call happened: the gate is before dispatch
        self.assertEqual(self.tools.audit_trail(tool="shout"), [])
        # re-enable and it runs
        self.registry.enable("chain")
        rerun = self.runner.run("chain", {"text": "hello", "tag": "b"})
        self.assertTrue(rerun.ok, rerun.error)

    def test_bad_input_fails_before_any_step(self):
        self.install(_chain_manifest())
        result = self.runner.run("chain", {"text": 42})
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_step, -1)
        self.assertEqual(result.steps, [])
        self.assertIn("expected str, got int", result.error)
        self.assertIsNotNone(result.ticket)

    def test_tool_exception_names_failing_step_and_inputs(self):
        self.tools.register("boom", _boom, capability="")
        self.install({"name": "fragile", "version": "1.0.0",
                      "tools": ["shout", "boom"],
                      "input_schema": {"text": "str"},
                      "wiring": [{"text": "$input.text"},
                                 {"text": "$0.shouted"}]})
        result = self.runner.run("fragile", {"text": "hi"})
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_step, 1)
        self.assertFalse(result.steps[1].ok)
        self.assertIn("simulated tool explosion", result.steps[1].error)
        ticket = result.ticket
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket.step_index, 1)
        self.assertEqual(ticket.tool, "boom")
        self.assertEqual(ticket.inputs, {"text": "HI"})
        self.assertIn("simulated tool explosion", ticket.error)
        self.assertTrue(ticket.suggested_fix)
        # persisted
        self.assertEqual(self.tickets.count("fragile"), 1)

    def test_unknown_tool_suggests_close_match(self):
        self.install({"name": "typo", "version": "1.0.0",
                      "tools": ["shout", "wrapp"],
                      "wiring": [{"text": "$input.text"}, {}]})
        result = self.runner.run("typo", {"text": "hi"})
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_step, 1)
        self.assertIn("wrap", result.ticket.suggested_fix)

    def test_output_schema_violation_names_missing_key(self):
        self.install(_chain_manifest(
            name="strict",
            output_schema={"wrapped": "str", "bytes": "int"}))
        result = self.runner.run("strict", {"text": "hi", "tag": "b"})
        self.assertFalse(result.ok)
        self.assertIn("'bytes'", result.ticket.suggested_fix)

    def test_bench_records_success_and_failure(self):
        self.install(_chain_manifest())
        self.runner.run("chain", {"text": "ok", "tag": "b"})
        self.runner.run("chain", {"text": 42})
        score = self.bench.score("chain")
        self.assertEqual(score["runs"], 2)
        self.assertAlmostEqual(score["success_rate"], 0.5)
        self.assertGreaterEqual(score["avg_latency_ms"], 0.0)

    def test_version_pin_selects_chain(self):
        self.install(_chain_manifest())
        self.tools.register("quiet", lambda text: {"shouted": text.lower()},
                            capability="")
        v2 = dict(_chain_manifest(), version="2.0.0", tools=["quiet"],
                    wiring=[{"text": "$input.text"}])
        self.registry.install(v2)
        # pin still on 1.0.0 → shout
        first = self.runner.run("chain", {"text": "Hi", "tag": "b"})
        self.assertEqual(first.outputs["0"], {"shouted": "HI"})
        self.registry.pin("chain", "2.0.0")
        second = self.runner.run("chain", {"text": "Hi", "tag": "b"})
        self.assertEqual(second.outputs["0"], {"shouted": "hi"})


if __name__ == "__main__":
    unittest.main()
