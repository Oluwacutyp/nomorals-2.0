"""Breaker worker + named re-plan triggers (Antigravity Boost/Orchestrator).

Failing-first: the breaker phase, the re-plan trigger, and the planner
repair do not exist yet — these tests fail (ImportError / missing
attributes) until the build lands.
"""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from nomorals.agents import swarm as sw


def _board_module():
    from nomorals.agents import swarm_board

    return swarm_board


class _StubAgent(sw.SwarmAgent):
    """SwarmAgent with scripted legs (same shape as the swarm-board
    lane's stub). ``script(agent, subtask, index, kw)`` returns a leg
    dict; ``kw`` carries verifier=/breaker= flags."""

    def __init__(self, script):
        super().__init__(SimpleNamespace())
        self._script = script
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def _run_leg(self, subtask, deadline, *, index=0, run_id="",
                 on_progress=None, total_legs=0, **kw):
        with self._lock:
            self.calls.append({
                "subtask": subtask, "index": index,
                "verifier": bool(kw.get("verifier")),
                "breaker": bool(kw.get("breaker")),
                "verifier_for": kw.get("verifier_for", ()),
                "breaker_for": kw.get("breaker_for", ()),
            })
        return self._script(self, subtask, index, kw)

    def worker_calls(self, index):
        return [c for c in self.calls
                if c["index"] == index
                and not c["verifier"] and not c["breaker"]]


class BreakerTests(unittest.TestCase):
    def test_breaker_finding_is_recorded_in_result(self):
        def _script(agent, subtask, index, kw):
            if kw.get("verifier"):
                return {"subtask": subtask, "ok": True,
                        "digest": "OK @audit-parser: fine",
                        "planned_by": "test", "steps": 1, "seconds": 0.1}
            if kw.get("breaker"):
                target = agent.board.members[0]["name"]
                return {"subtask": subtask, "ok": True,
                        "digest": (f"BREAK @{target}: untested edge case "
                                   "— empty input crashes"),
                        "planned_by": "test", "steps": 1, "seconds": 0.1}
            return {"subtask": subtask, "ok": True,
                    "digest": f"done {index}", "planned_by": "test",
                    "steps": 1, "seconds": 0.1}

        agent = _StubAgent(_script)
        result = agent.run("audit the parser and the api", workers=2)
        self.assertTrue(result.ok)
        self.assertTrue(result.breaker_ran)
        self.assertGreaterEqual(len(result.breaker_findings), 1)
        finding = result.breaker_findings[0]
        self.assertIn("attack", finding)
        self.assertIn("untested edge case", finding["attack"])
        self.assertIn("worker", finding)
        # The finding reached the board, not just the result.
        kinds = [p["kind"] for p in agent.board.feed]
        self.assertIn("breaker", kinds)
        # The broken worker was re-run with the attack in its brief.
        reruns = agent.worker_calls(0)
        self.assertEqual(len(reruns), 2)
        self.assertTrue(any("untested edge case" in c["subtask"]
                            for c in reruns))

    def test_breaker_with_no_finding_is_recorded_not_silent(self):
        def _script(agent, subtask, index, kw):
            if kw.get("verifier"):
                return {"subtask": subtask, "ok": True,
                        "digest": "OK @audit-parser: fine",
                        "planned_by": "test", "steps": 1, "seconds": 0.1}
            if kw.get("breaker"):
                target = agent.board.members[0]["name"]
                return {"subtask": subtask, "ok": True,
                        "digest": (f"HOLD @{target}: holds — every edge "
                                   "case is covered"),
                        "planned_by": "test", "steps": 1, "seconds": 0.1}
            return {"subtask": subtask, "ok": True,
                    "digest": f"done {index}", "planned_by": "test",
                    "steps": 1, "seconds": 0.1}

        agent = _StubAgent(_script)
        result = agent.run("audit the parser and the api", workers=2)
        self.assertTrue(result.ok)
        # The breaker RAN and found nothing — that absence is a
        # recorded fact, not a silent green.
        self.assertTrue(result.breaker_ran)
        self.assertEqual(result.breaker_findings, [])
        kinds = [p["kind"] for p in agent.board.feed]
        self.assertIn("breaker", kinds)
        self.assertTrue(any("no findings" in str(p.get("text") or "")
                            for p in agent.board.feed
                            if p["kind"] == "breaker"))

    def test_breaker_phase_skipped_when_all_legs_failed(self):
        def _script(agent, subtask, index, kw):
            return {"subtask": subtask, "ok": False,
                    "digest": "leg crashed", "planned_by": "test",
                    "steps": 0, "seconds": 0.1}

        agent = _StubAgent(_script)
        result = agent.run("audit the parser and the api", workers=2)
        self.assertFalse(result.ok)
        # Nothing to break when everything failed.
        self.assertFalse(result.breaker_ran)
        self.assertEqual(result.breaker_findings, [])


class ReplanTests(unittest.TestCase):
    def test_double_failure_triggers_named_replan(self):
        counts: dict[int, int] = {}

        def _script(agent, subtask, index, kw):
            if kw.get("verifier") or kw.get("breaker"):
                return {"subtask": subtask, "ok": True,
                        "digest": "OK", "planned_by": "test",
                        "steps": 1, "seconds": 0.1}
            n = counts.get(index, 0) + 1
            counts[index] = n
            if index == 0 and n < 3:
                return {"subtask": subtask, "ok": False,
                        "digest": f"attempt {n} crashed",
                        "planned_by": "test", "steps": 0, "seconds": 0.1}
            return {"subtask": subtask, "ok": True,
                    "digest": f"done {index} attempt {n}",
                    "planned_by": "test", "steps": 1, "seconds": 0.1}

        agent = _StubAgent(_script)
        result = agent.run("audit the parser and the api", workers=2)
        self.assertTrue(result.ok)
        self.assertEqual(len(result.replans), 1)
        record = result.replans[0]
        self.assertEqual(record["trigger"], "verification_failed_twice")
        self.assertTrue(record["changes"])
        self.assertNotEqual(record["revised"], record["original"])
        # The retry ran the REVISED brief, not the old one.
        third = agent.worker_calls(0)[2]
        self.assertIn("Revised plan", third["subtask"])
        self.assertNotEqual(third["subtask"],
                            agent.worker_calls(0)[0]["subtask"])
        # The worker survived: the revised attempt worked.
        states = {m["name"]: m["state"] for m in agent.board.members}
        self.assertNotIn("killed", states.values())

    def test_replan_fires_exactly_once_then_third_failure_kills(self):
        def _script(agent, subtask, index, kw):
            if kw.get("verifier") or kw.get("breaker"):
                return {"subtask": subtask, "ok": True,
                        "digest": "OK", "planned_by": "test",
                        "steps": 1, "seconds": 0.1}
            if index == 0:
                return {"subtask": subtask, "ok": False,
                        "digest": "leg crashed", "planned_by": "test",
                        "steps": 0, "seconds": 0.1}
            return {"subtask": subtask, "ok": True,
                    "digest": "done", "planned_by": "test",
                    "steps": 1, "seconds": 0.1}

        agent = _StubAgent(_script)
        result = agent.run("audit the parser and the api", workers=2)
        self.assertEqual(len(result.replans), 1)
        # Three attempts, then killed — no re-plan loop.
        self.assertEqual(len(agent.worker_calls(0)), 3)
        states = {m["name"]: m["state"] for m in agent.board.members}
        self.assertIn("killed", states.values())
        # The history is attached to the replan record.
        record = result.replans[0]
        self.assertGreaterEqual(len(record["failures"]), 2)
        self.assertEqual(record["trigger"], "verification_failed_twice")

    def test_single_failure_does_not_replan(self):
        counts: dict[int, int] = {}

        def _script(agent, subtask, index, kw):
            if kw.get("verifier") or kw.get("breaker"):
                return {"subtask": subtask, "ok": True,
                        "digest": "OK", "planned_by": "test",
                        "steps": 1, "seconds": 0.1}
            n = counts.get(index, 0) + 1
            counts[index] = n
            if index == 0 and n == 1:
                return {"subtask": subtask, "ok": False,
                        "digest": "first attempt crashed",
                        "planned_by": "test", "steps": 0, "seconds": 0.1}
            return {"subtask": subtask, "ok": True,
                    "digest": "done", "planned_by": "test",
                    "steps": 1, "seconds": 0.1}

        agent = _StubAgent(_script)
        result = agent.run("audit the parser and the api", workers=2)
        self.assertTrue(result.ok)
        self.assertEqual(result.replans, [])
        self.assertEqual(len(agent.worker_calls(0)), 2)


class RepairStepTests(unittest.TestCase):
    def test_repair_step_template_names_changes(self):
        from nomorals.agents.orchestrator import MasterOrchestrator

        planner = MasterOrchestrator(SimpleNamespace())
        record = planner.repair_step(
            "audit the parser",
            "audit the parser for crashes",
            [{"attempt": 1, "digest": "leg crashed: null deref"},
             {"attempt": 2, "digest": "leg crashed: null deref again",
              "notes": ["verifier: missing null check"]}])
        self.assertEqual(record["trigger"], "verification_failed_twice")
        self.assertEqual(record["original"],
                         "audit the parser for crashes")
        self.assertNotEqual(record["revised"], record["original"])
        self.assertIn("null deref", record["revised"])
        self.assertTrue(record["changes"])
        self.assertEqual(len(record["failures"]), 2)

    def test_repair_step_needs_a_subtask_and_failures(self):
        from nomorals.agents.orchestrator import MasterOrchestrator

        planner = MasterOrchestrator(SimpleNamespace())
        with self.assertRaises(ValueError):
            planner.repair_step("goal", "", [{"attempt": 1}])
        with self.assertRaises(ValueError):
            planner.repair_step("goal", "do the thing", [])


class BreakerBoardTests(unittest.TestCase):
    def test_parse_breaker_verdict(self):
        mod = _board_module()
        found = mod.parse_breaker_verdict(
            "BREAK @audit-parser: untested edge case\n"
            "HOLD @api: holds — covered\n"
            "BREAK @api: silent assumption — auth never checked")
        self.assertEqual(set(found), {"audit-parser", "api"})
        self.assertIn("untested edge case", found["audit-parser"][0])
        self.assertIn("silent assumption", found["api"][0])
        self.assertEqual(mod.parse_breaker_verdict("all good"), {})

    def test_assign_breakers_one_per_two_to_four_workers(self):
        mod = _board_module()
        board = mod.SwarmBoard("swarm-test-1", goal="test")
        for i in range(5):
            board.add_member(f"worker-{i + 1}", "worker")
        breakers = board.assign_breakers()
        self.assertEqual(len(breakers), 2)
        self.assertEqual(board.breaker_group(breakers[0]),
                         [f"worker-{i + 1}" for i in range(4)])
        small = mod.SwarmBoard("swarm-test-2", goal="test")
        small.add_member("worker-1", "worker")
        self.assertEqual(len(small.assign_breakers()), 1)

    def test_breakers_group_mention_routes(self):
        mod = _board_module()
        board = mod.SwarmBoard("swarm-test-1", goal="test")
        board.add_member("worker-1", "worker")
        breaker = board.assign_breakers()[0]
        board.post("coordinator", "@breakers attack the parser now")
        hits = board.mentions_for(breaker)
        self.assertTrue(any("@breakers" in p["text"] for p in hits))


if __name__ == "__main__":
    unittest.main()
