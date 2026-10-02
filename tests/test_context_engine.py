"""Tests for nomorals.context.engine: section assembly + step-prompt wiring."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

from nomorals.context import (
    BuiltContext,
    ContextBudget,
    ContextEngine,
    SnapshotStore,
)
from nomorals.missions.mission import Mission


def _reference_step_prompt(mission, step) -> str:
    """Byte-exact copy of the pre-Wave-I missions/runner.py _step_prompt."""
    outputs = mission.state.get("outputs") or {}
    prior = "\n".join(
        f"- {name}: {str(value)[:300]}" for name, value in list(outputs.items())[-4:]
    )
    parts = [f"Mission goal: {mission.goal}", f"Current step: {step.goal or step.name}"]
    if prior:
        parts.append(f"Already produced:\n{prior}")
    return "\n\n".join(parts)


def _mission(**kwargs) -> Mission:
    kwargs.setdefault("goal", "Ship the frobnicate widget")
    return Mission(**kwargs)


def _mission_stub(**kwargs) -> SimpleNamespace:
    """Mission-shaped duck-type carrying acceptance_criteria etc.

    The real Mission dataclass has no acceptance_criteria field; the engine
    reads it structurally, so stubs exercise the same path.
    """
    kwargs.setdefault("goal", "Ship the frobnicate widget")
    kwargs.setdefault("id", "m-stub")
    kwargs.setdefault("status", "running")
    kwargs.setdefault("state", {})
    return SimpleNamespace(**kwargs)


def _step(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(goal=kwargs.get("goal", "Build it"),
                           name=kwargs.get("name", "build"))


class StepPromptCompatTest(unittest.TestCase):
    """The runner's _step_prompt wrapper must be byte-identical to before."""

    def test_identical_with_outputs(self) -> None:
        mission = _mission(state={"outputs": {
            "plan": "do the thing", "code": "x = 1" * 200, "notes": "n"}})
        step = _step(goal="Write the code", name="code")
        self.assertEqual(
            ContextEngine().build_step_prompt(mission, step),
            _reference_step_prompt(mission, step),
        )

    def test_identical_without_outputs(self) -> None:
        mission = _mission()
        step = _step(goal="", name="recon")
        self.assertEqual(
            ContextEngine().build_step_prompt(mission, step),
            _reference_step_prompt(mission, step),
        )

    def test_identical_many_outputs_keeps_last_four(self) -> None:
        mission = _mission(state={"outputs": {f"s{i}": f"v{i}" for i in range(9)}})
        step = _step()
        self.assertEqual(
            ContextEngine().build_step_prompt(mission, step),
            _reference_step_prompt(mission, step),
        )

    def test_runner_wrapper_delegates(self) -> None:
        from nomorals.missions import runner

        mission = _mission(state={"outputs": {"a": "b"}})
        step = _step()
        self.assertEqual(
            runner._step_prompt(mission, step),
            _reference_step_prompt(mission, step),
        )


class RichStepPromptTest(unittest.TestCase):
    def test_rich_mode_assembles_sections(self) -> None:
        engine = ContextEngine()
        mission = _mission_stub(
            acceptance_criteria=["widget frobnicates"],
            state={"outputs": {"plan": "p"}},
        )
        text = engine.build_step_prompt(mission, _step(), rich=True)
        self.assertIn("Ship the frobnicate widget", text)
        self.assertIn("widget frobnicates", text)
        self.assertIn("## System", text)
        self.assertIn("## Mission", text)


class BuildSectionsTest(unittest.TestCase):
    def _built(self, budget_total: int = 8000, **kwargs) -> BuiltContext:
        engine = ContextEngine(budget=ContextBudget(total=budget_total))
        mission = {
            "id": "m-1",
            "goal": "Ship it",
            "status": "running",
            "acceptance_criteria": ["it ships", "it works"],
            "required_artifacts": ["artifact://abc123"],
            "state": {"outputs": {"draft": "v1"}},
        }
        artifacts = [{
            "id": "abc123",
            "type": "text",
            "creator": "coder",
            "metadata": {"summary": "the draft"},
            "provenance": {"derived_from": ["artifact://seed"]},
        }]
        tools = [{"name": "web_search", "description": "search the web"}]
        return engine.build(
            mission=mission,
            artifacts=artifacts,
            tools=tools,
            project={"name": "p", "goal": "g"},
            user_profile={"name": "death"},
            history=[("user", "hi"), ("assistant", "hello")],
            **kwargs,
        )

    def test_all_sections_present(self) -> None:
        built = self._built()
        for name in ("system", "mission", "artifacts", "tools",
                     "project", "user_profile", "history"):
            section = built.section(name)
            self.assertIsNotNone(section, name)
            title = name.replace("_", " ").title()
            self.assertIn(f"## {title}", built.text)

    def test_artifact_uri_and_provenance_rendered(self) -> None:
        built = self._built()
        self.assertIn("artifact://abc123", built.text)
        self.assertIn("derived from", built.text)

    def test_tool_manifest_rendered(self) -> None:
        built = self._built()
        self.assertIn("web_search", built.text)

    def test_report_shape(self) -> None:
        report = self._built().report()
        self.assertIn("total_tokens", report)
        self.assertIn("sections", report)
        self.assertFalse(report["over_budget"])

    def test_load_bearing_survives_extreme_budget(self) -> None:
        # A 60-token total budget: acceptance criteria + required artifact
        # URIs must still be in the rendered text.
        built = self._built(budget_total=60)
        self.assertIn("it ships", built.text)
        self.assertIn("artifact://abc123", built.text)
        mission_section = built.section("mission")
        self.assertIsNotNone(mission_section)
        self.assertTrue(mission_section.load_bearing)
        self.assertTrue(built.over_budget)

    def test_tiny_budget_drops_history_first(self) -> None:
        engine = ContextEngine(budget=ContextBudget(total=400))
        history = [(f"user", f"filler message number {i} " * 10)
                   for i in range(60)]
        built = engine.build(
            mission={"goal": "g", "acceptance_criteria": ["a1"]},
            history=history,
        )
        self.assertIn("history", built.dropped)
        # Load-bearing mission survives even here.
        self.assertIn("a1", built.text)

    def test_dict_and_dataclass_missions_agree(self) -> None:
        engine = ContextEngine()
        dc = _mission_stub(acceptance_criteria=["x works"])
        d = {"goal": dc.goal, "acceptance_criteria": ["x works"]}
        a = engine.build(mission=dc).section("mission").content
        b = engine.build(mission=d).section("mission").content
        self.assertIn("x works", a)
        self.assertIn("x works", b)


class SnapshotRoundTripTest(unittest.TestCase):
    def test_save_load_round_trip(self) -> None:
        engine = ContextEngine()
        built = engine.build(
            mission={"goal": "g", "acceptance_criteria": ["a1"]},
            history=[("user", "hello world")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(tmp)
            store.save("rewind-point", built)
            self.assertEqual(store.list(), ["rewind-point"])
            restored = store.load("rewind-point")
        self.assertEqual(restored.text, built.text)
        self.assertEqual(restored.total_tokens, built.total_tokens)
        self.assertEqual(
            [s.name for s in restored.sections],
            [s.name for s in built.sections])
        self.assertEqual(restored.report()["sections"],
                         built.report()["sections"])

    def test_load_unknown_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                SnapshotStore(tmp).load("nope")

    def test_delete(self) -> None:
        engine = ContextEngine()
        built = engine.build(mission={"goal": "g"})
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(tmp)
            store.save("x", built)
            self.assertTrue(store.delete("x"))
            self.assertFalse(store.delete("x"))
            self.assertEqual(store.list(), [])


if __name__ == "__main__":
    unittest.main()
