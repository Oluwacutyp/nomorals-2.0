"""Doc 18 P6 / G8 — typed Plan/Stop schemas at the orchestrator seam.

Pins: (a) the legacy ``stop:`` text parses into StopDirective under
the HISTORICAL predicate exactly (stripped, case-insensitive
startswith("stop") — texts that fired before fire now, no others);
(b) text-born directives render back byte-identical via to_text()
(the graph cancel reason and reevaluation record are text consumers
and must see the same strings); (c) the revise_plan abort dict
parses at the same seam with its historical truthiness + fallback;
(d) a natively typed directive renders the canonical text form;
(e) the seam wiring: _course_correct_pivot returns the verbatim
pivot for a stop verdict and None otherwise; reevaluate aborts on a
revise_plan abort with the same reason text as before; (f) plan
round-trip: the persisted step shape rebuilds through
_plan_from_state with payload (writes/consequential/debate) intact;
(g) both step producers carry the G9 declarations into the payload:
the orchestrator's model-plan parser and the loop bridge.
"""

from __future__ import annotations

import unittest
from unittest import mock

from nomorals.agents.orchestrator import (
    MasterOrchestrator,
    Plan,
    PlanStep,
    StopDirective,
    _plan_from_state,
)
from nomorals.core.tasks import Task, TaskGraph, TaskState


class FromTextTests(unittest.TestCase):
    def test_colon_form(self):
        d = StopDirective.from_text("stop: the data is stale")
        self.assertIsNotNone(d)
        self.assertEqual(d.reason, "the data is stale")
        self.assertEqual(d.source, "course_correct")
        self.assertEqual(d.scope, "run")

    def test_verbatim_round_trip(self):
        for text in ("stop: the data is stale", "Stop", "STOP: halt now",
                     "stopping the rollout", "stop:  spaced",
                     "stopgap measure"):
            d = StopDirective.from_text(text)
            self.assertIsNotNone(d, text)
            self.assertEqual(d.to_text(), text.strip()[:300], text)

    def test_bare_stop_has_empty_reason(self):
        self.assertEqual(StopDirective.from_text("stop").reason, "")
        self.assertEqual(StopDirective.from_text("stop:").reason, "")

    def test_non_stop_texts_do_not_fire(self):
        for text in ("proceed", "", "a stop sign", " restart it", None):
            self.assertIsNone(StopDirective.from_text(text), text)

    def test_reason_truncates_at_300(self):
        d = StopDirective.from_text("stop: " + "x" * 400)
        self.assertEqual(len(d.reason), 300)


class FromMappingTests(unittest.TestCase):
    def test_abort_parses(self):
        d = StopDirective.from_mapping({"abort": "bad state"})
        self.assertEqual(d.reason, "bad state")
        self.assertEqual(d.source, "revise_plan")

    def test_no_abort_no_directive(self):
        self.assertIsNone(StopDirective.from_mapping({"drop": ["x"]}))
        self.assertIsNone(StopDirective.from_mapping({"abort": ""}))
        self.assertIsNone(StopDirective.from_mapping({"abort": 0}))
        self.assertIsNone(StopDirective.from_mapping(None))
        self.assertIsNone(StopDirective.from_mapping("abort"))

    def test_typed_directive_renders_canonical_text(self):
        self.assertEqual(StopDirective(reason="r").to_text(), "stop: r")
        self.assertEqual(StopDirective().to_text(), "stop")


class SeamWiringTests(unittest.TestCase):
    def _orch(self):
        from types import SimpleNamespace

        return MasterOrchestrator(SimpleNamespace(
            db=None, router=None, executor=None, blackboard=None,
            emit=lambda *a, **k: None))

    def test_course_correct_pivot_verbatim(self):
        orch = self._orch()
        verdict = {"should_pivot": True, "pivot": "Stop: approach is wrong"}
        agent = mock.Mock()
        agent.course_correct.return_value = verdict
        with mock.patch("nomorals.agents.reasoning.ReasoningAgent",
                        return_value=agent):
            self.assertEqual(
                orch._course_correct_pivot(Task(name="s")),
                "Stop: approach is wrong")

    def test_course_correct_non_stop_is_none(self):
        orch = self._orch()
        agent = mock.Mock()
        agent.course_correct.return_value = {
            "should_pivot": True, "pivot": "change approach: retry"}
        with mock.patch("nomorals.agents.reasoning.ReasoningAgent",
                        return_value=agent):
            self.assertIsNone(orch._course_correct_pivot(Task(name="s")))

    def test_reevaluate_abort_via_revise_plan(self):
        orch = self._orch()
        graph = TaskGraph(name="g")
        settled = Task(name="s1", payload={"goal": "one"})
        settled.state = TaskState.DONE
        settled.result = {"revise_plan": {"abort": "bad state"}}
        pending = Task(name="s2", payload={"goal": "two"})
        graph.add(settled)
        graph.add(pending)
        out = orch.reevaluate("goal", graph)
        self.assertEqual(out.action, "abort")
        self.assertIn("signalled abort: bad state", out.reason)
        self.assertEqual(pending.state, TaskState.CANCELLED)


class PlanRoundTripTests(unittest.TestCase):
    def test_persisted_shape_rebuilds_with_payload(self):
        raw = [{
            "name": "write", "goal": "write it", "role": "execution",
            "kind": "io", "depends_on": [],
            "payload": {"writes": ["out.txt"], "consequential": True,
                        "debate": True},
        }, {
            "name": "look", "goal": "read it", "role": "research",
            "kind": "io", "depends_on": ["write"], "payload": {},
        }]
        plan = _plan_from_state("g", raw)
        by_name = {s.name: s for s in plan.steps}
        self.assertEqual(by_name["write"].payload["writes"], ["out.txt"])
        self.assertIs(by_name["write"].payload["consequential"], True)
        self.assertIs(by_name["write"].payload["debate"], True)
        self.assertEqual(by_name["look"].depends_on, ["write"])

    def test_plan_to_dict_keeps_typed_kinds(self):
        plan = Plan(goal="g", steps=[PlanStep(name="a", goal="ga")])
        row = plan.to_dict()["steps"][0]
        self.assertEqual(row["kind"], "io")
        self.assertEqual(row["role"], "execution")


class ProducerCarryTests(unittest.TestCase):
    def test_parse_plan_carries_declarations(self):
        orch = SeamWiringTests()._orch()
        text = (
            '{"steps": [{"name": "w", "goal": "write it",'
            ' "role": "execution", "writes": ["a.txt", " b.txt "],'
            ' "consequential": true, "debate": true},'
            ' {"name": "r", "goal": "read", "role": "research",'
            ' "writes": "not-a-list"}]}')
        plan = orch._parse_plan("g", text)
        by_name = {s.name: s for s in plan.steps}
        self.assertEqual(by_name["w"].payload["writes"],
                         ["a.txt", "b.txt"])
        self.assertIs(by_name["w"].payload["consequential"], True)
        self.assertIs(by_name["w"].payload["debate"], True)
        self.assertNotIn("writes", by_name["r"].payload)

    def test_bridge_carries_declarations(self):
        from types import SimpleNamespace

        from nomorals.agents.orchestration.planner_bridge import (
            model_steps_to_plan,
        )

        tools = SimpleNamespace(has=lambda name: True)
        plan = model_steps_to_plan("g", [{
            "name": "w", "goal": "write it", "tool": "file_write",
            "role": "execution", "writes": ["a.txt"],
            "consequential": True,
        }], tools)
        self.assertEqual(plan.steps[0].payload["writes"], ["a.txt"])
        self.assertIs(plan.steps[0].payload["consequential"], True)
        self.assertEqual(plan.steps[0].payload["tool"], "file_write")


if __name__ == "__main__":
    unittest.main()
