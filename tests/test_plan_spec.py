"""Tests for spec-first plan mode (#4 extension). No network, no LLM."""

import os
import tempfile
import unittest

from nomorals.agents.plan_spec import (
    BACKEND_INCLUDED_BLOCK,
    PlanSpec,
    SpecStore,
    diff_specs,
    inject_backend_included,
    plan_to_spec,
    render_spec,
    should_backend_include,
)
from nomorals.agents.plan_mode import CodePlan


def _isolate_store(test):
    """Point the spec store at a temp dir for the test's duration."""
    tmp = tempfile.mkdtemp()
    old = os.environ.get("NM_SPEC_STORE")
    os.environ["NM_SPEC_STORE"] = os.path.join(tmp, "specs.json")
    SpecStore._specs = {}
    SpecStore._loaded = False
    def _restore():
        SpecStore._specs = {}
        SpecStore._loaded = False
        if old is None:
            os.environ.pop("NM_SPEC_STORE", None)
        else:
            os.environ["NM_SPEC_STORE"] = old
    test.addCleanup(_restore)


class BackendBlockTests(unittest.TestCase):
    def test_block_content(self):
        self.assertIn("SQLite", BACKEND_INCLUDED_BLOCK)
        self.assertIn("scheduled-job", BACKEND_INCLUDED_BLOCK)
        self.assertIn("never stub", BACKEND_INCLUDED_BLOCK.lower())

    def test_inject_appends_once(self):
        once = inject_backend_included("You are a coder.")
        self.assertIn("BACKEND INCLUDED", once)
        twice = inject_backend_included(once)
        self.assertEqual(once.count("BACKEND INCLUDED"), 1)
        self.assertEqual(twice.count("BACKEND INCLUDED"), 1)

    def test_should_backend_include(self):
        self.assertTrue(should_backend_include("nomorals/app/main.py"))
        self.assertTrue(should_backend_include("widget.py"))
        self.assertFalse(should_backend_include("tests/test_x.py"))
        self.assertFalse(should_backend_include("nomorals/tests/a.py"))
        self.assertFalse(should_backend_include("README.md"))
        self.assertFalse(should_backend_include(""))
        self.assertFalse(should_backend_include(None))

    def test_inject_never_raises(self):
        # None → block alone (never raises, always a string)
        self.assertIn("BACKEND INCLUDED", inject_backend_included(None))
        out = inject_backend_included(123)
        self.assertIn("123", out)
        self.assertIn("BACKEND INCLUDED", out)


class SpecStoreTests(unittest.TestCase):
    def setUp(self):
        _isolate_store(self)

    def test_new_get_roundtrip(self):
        spec = SpecStore.new("build a tracker", tasks=[{"path": "a.py"}])
        self.assertTrue(spec.spec_id)
        self.assertEqual(spec.version, 1)
        got = SpecStore.get(spec.spec_id)
        self.assertIsNotNone(got)
        self.assertEqual(got.goals, "build a tracker")
        self.assertEqual(len(got.tasks), 1)

    def test_get_versioned_key(self):
        spec = SpecStore.new("g")
        got = SpecStore.get(spec.key)
        self.assertEqual(got.version, 1)

    def test_get_unknown(self):
        self.assertIsNone(SpecStore.get("nope"))
        self.assertIsNone(SpecStore.get(""))
        self.assertIsNone(SpecStore.get(None))

    def test_bump_version(self):
        spec = SpecStore.new("v1 goals", architecture="flat")
        v2 = SpecStore.bump_version(spec.spec_id,
                                   {"goals": "v2 goals",
                                    "architecture": "layered"},
                                   note="rethink")
        self.assertEqual(v2.version, 2)
        self.assertEqual(v2.parent_spec_id, spec.key)
        self.assertEqual(v2.change_note, "rethink")
        # latest wins on bare id
        self.assertEqual(SpecStore.get(spec.spec_id).version, 2)
        # old version still addressable
        self.assertEqual(SpecStore.get(spec.key).goals, "v1 goals")
        self.assertEqual(SpecStore.get(v2.key).goals, "v2 goals")

    def test_bump_unknown(self):
        self.assertIsNone(SpecStore.bump_version("nope", {"goals": "x"}))

    def test_history(self):
        spec = SpecStore.new("g")
        SpecStore.bump_version(spec.spec_id, {"goals": "g2"})
        hist = SpecStore.history(spec.spec_id)
        self.assertEqual([s.version for s in hist], [1, 2])

    def test_persistence(self):
        spec = SpecStore.new("persisted goals")
        # simulate a fresh process
        SpecStore._specs = {}
        SpecStore._loaded = False
        got = SpecStore.get(spec.spec_id)
        self.assertIsNotNone(got)
        self.assertEqual(got.goals, "persisted goals")


class DiffRenderTests(unittest.TestCase):
    def setUp(self):
        _isolate_store(self)

    def test_diff_specs(self):
        a = SpecStore.new("goals a", architecture="flat")
        b = SpecStore.bump_version(a.spec_id, {"goals": "goals b"})
        d = diff_specs(a, b)
        self.assertIn("goals", d)
        self.assertIn("-goals a", d["goals"])
        self.assertIn("+goals b", d["goals"])
        self.assertNotIn("architecture", d)

    def test_diff_tasks(self):
        a = SpecStore.new("g", tasks=[{"path": "a.py"}])
        b = SpecStore.bump_version(a.spec_id,
                                  {"tasks": [{"path": "a.py"},
                                             {"path": "b.py"}]})
        d = diff_specs(a, b)
        self.assertIn("tasks", d)
        self.assertIn("b.py", d["tasks"])

    def test_diff_identical(self):
        a = SpecStore.new("g")
        self.assertEqual(diff_specs(a, a), {})

    def test_diff_never_raises(self):
        self.assertEqual(diff_specs(None, None), {})
        a = SpecStore.new("g")
        self.assertEqual(diff_specs(None, a), {})

    def test_render(self):
        spec = SpecStore.new("ship it", architecture="two tiers",
                             tasks=[{"path": "m.py", "what": "main",
                                     "new_file": True}])
        text = render_spec(spec)
        self.assertIn("SPEC", text)
        self.assertIn("## GOALS", text)
        self.assertIn("ship it", text)
        self.assertIn("## DATA MODEL", text)
        self.assertIn("[new] m.py", text)

    def test_render_revision_line(self):
        a = SpecStore.new("g")
        b = SpecStore.bump_version(a.spec_id, {"goals": "g2"},
                                   note="rethought")
        text = render_spec(b)
        self.assertIn(a.key, text)
        self.assertIn("rethought", text)


class PlanToSpecTests(unittest.TestCase):
    def setUp(self):
        _isolate_store(self)

    def test_from_code_plan(self):
        plan = CodePlan(
            id="p1", task="build the widget",
            files=[{"path": "w.py", "why": "main module", "new_file": True},
                   {"path": "w_store.py", "why": "storage", "new_file": True}])
        spec = plan_to_spec(plan)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.goals, "build the widget")
        self.assertEqual(len(spec.tasks), 2)
        self.assertEqual(spec.tasks[0]["path"], "w.py")
        self.assertTrue(spec.tasks[0]["new_file"])

    def test_goals_override(self):
        plan = CodePlan(id="p1", task="t", files=[])
        spec = plan_to_spec(plan, goals="custom goals")
        self.assertEqual(spec.goals, "custom goals")

    def test_none_never_raises(self):
        self.assertIsNone(plan_to_spec(None))


class CodingAgentSpecTests(unittest.TestCase):
    def _agent(self):
        from unittest.mock import MagicMock
        from nomorals.agents.coding import CodingAgent
        ctx = MagicMock()
        agent = CodingAgent(ctx, root=tempfile.mkdtemp())
        _isolate_store(self)
        return agent

    def test_draft_injects_backend_block(self):
        agent = self._agent()
        seen = {}

        class Resp:
            ok = True
            text = "```python\nprint(1)\n```"

        def fake_chat(phase, messages, params):
            seen["system"] = messages[0].content
            return Resp()

        agent._phase_chat = fake_chat
        code = agent._draft("make app", "app/main.py", "", "", 1)
        self.assertIn("BACKEND INCLUDED", seen["system"])
        self.assertIn("print(1)", code)

    def test_draft_opt_out(self):
        agent = self._agent()
        agent._backend_included = False
        seen = {}

        class Resp:
            ok = True
            text = "```python\nprint(1)\n```"

        def fake_chat(phase, messages, params):
            seen["system"] = messages[0].content
            return Resp()

        agent._phase_chat = fake_chat
        agent._draft("make app", "app/main.py", "", "", 1)
        self.assertNotIn("BACKEND INCLUDED", seen["system"])

    def test_draft_skips_tests(self):
        agent = self._agent()
        seen = {}

        class Resp:
            ok = True
            text = "```python\nprint(1)\n```"

        def fake_chat(phase, messages, params):
            seen["system"] = messages[0].content
            return Resp()

        agent._phase_chat = fake_chat
        agent._draft("make test", "tests/test_x.py", "", "", 1)
        self.assertNotIn("BACKEND INCLUDED", seen["system"])

    def test_plan_to_spec_method(self):
        from nomorals.agents.plan_mode import PlanStore
        agent = self._agent()
        plan = PlanStore.new("build x",
                             [{"path": "x.py", "why": "main",
                               "new_file": True}])
        spec = agent.plan_to_spec(plan.id)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.goals, "build x")
        self.assertIsNone(agent.plan_to_spec("nope"))

    def test_regenerate_from_spec(self):
        agent = self._agent()
        root = agent._root
        (root / "x.py").write_text("print('old')\n", encoding="utf-8")
        spec = SpecStore.new("regenerate goals",
                             tasks=[{"path": "x.py", "what": "rewrite it"}])

        class Resp:
            ok = True
            text = "```python\nprint('new')\n```"

        agent._phase_chat = lambda phase, messages, params: Resp()
        out = agent.regenerate_from_spec(spec.spec_id)
        self.assertTrue(out["ok"])
        self.assertEqual(out["spec"], spec.key)
        entry = out["files"]["x.py"]
        self.assertTrue(entry["ok"])
        self.assertTrue(entry["changed"])
        self.assertIn("print('new')", entry["after"])
        self.assertIn("print('old')", entry["diff"])

    def test_regenerate_unknown_spec(self):
        agent = self._agent()
        out = agent.regenerate_from_spec("nope")
        self.assertFalse(out["ok"])
        self.assertIn("unknown spec", out["reason"])

    def test_regenerate_never_raises(self):
        agent = self._agent()
        agent._resolve = lambda rel: (_ for _ in ()).throw(RuntimeError("boom"))
        spec = SpecStore.new("g", tasks=[{"path": "x.py"}])
        out = agent.regenerate_from_spec(spec.spec_id)
        self.assertTrue(out["ok"])
        self.assertIn("x.py", out["files"])


if __name__ == "__main__":
    unittest.main()
