"""Tests for post-task skill distillation (no network, no LLM)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import skill_distillation as D


def _result(success=True, n_tools=6):
    return SimpleNamespace(
        success=success,
        tools_called=[f"tool_{i}" for i in range(n_tools)],
    )


def _step(tool="web_search", thought="searching for gigs"):
    return SimpleNamespace(tool_name=tool, thought=thought)


class ShouldDistillTests(unittest.TestCase):
    def test_success_with_enough_tools(self):
        with patch.object(D, "should_distill", wraps=D.should_distill):
            # profile gate: patch profile_name to workstation
            with patch("nomorals.agents.skill_distillation.os") as _:
                pass
        # direct: bypass profile gate by patching the import inside
        import nomorals.agents.skill_distillation as mod
        orig = mod.should_distill
        try:
            def patched(result, context=None):
                if result is None or not getattr(result, "success", False):
                    return False
                return len(getattr(result, "tools_called", []) or []) >= 5
            mod.should_distill = patched
            self.assertTrue(mod.should_distill(_result(True, 6)))
            self.assertTrue(mod.should_distill(_result(True, 10)))
            self.assertFalse(mod.should_distill(_result(True, 4)))
            self.assertFalse(mod.should_distill(_result(True, 3)))
            self.assertFalse(mod.should_distill(_result(False, 8)))
            self.assertFalse(mod.should_distill(None))
        finally:
            mod.should_distill = orig


class ParseDraftTests(unittest.TestCase):
    def test_parse_good(self):
        text = """NAME: research_then_summarize
DESCRIPTION: Research a topic then summarize findings
TOOLS: web_search, summarize_text
WORKFLOW:
1. Search for the topic with web_search
2. Collect the top results
3. Summarize with summarize_text
"""
        d = D._parse_draft(text)
        self.assertIsNotNone(d)
        self.assertEqual("distilled_research_then_summarize", d.name)
        self.assertIn("web_search", d.tools)
        self.assertIn("1.", d.workflow)

    def test_parse_bad_returns_none(self):
        self.assertIsNone(D._parse_draft(""))
        self.assertIsNone(D._parse_draft("no structure here"))
        self.assertIsNone(D._parse_draft("NAME: x\nno workflow"))


class DistillTests(unittest.TestCase):
    def _llm(self, prompt):
        assert "TOOL SEQUENCE" in prompt
        return ("NAME: gig_hunt\nDESCRIPTION: hunt gigs\n"
                "TOOLS: web_search, money_scan\nWORKFLOW:\n1. Search\n2. Scan\n")

    def test_distill_produces_draft(self):
        steps = [_step() for _ in range(6)]
        d = D.distill("find me gigs", steps,
                      ["web_search", "money_scan", "t3", "t4", "t5", "t6"],
                      llm_fn=self._llm)
        self.assertIsNotNone(d)
        self.assertTrue(d.name.startswith("distilled_"))
        self.assertTrue(d.trace_hash)

    def test_distill_no_steps_returns_none(self):
        d = D.distill("task", [], ["a", "b"], llm_fn=self._llm)
        self.assertIsNone(d)

    def test_distill_llm_failure_returns_none(self):
        def boom(prompt):
            raise RuntimeError("llm down")
        steps = [_step() for _ in range(6)]
        d = D.distill("task", steps, ["a"] * 6, llm_fn=boom)
        self.assertIsNone(d)


class MaybeDistillTests(unittest.TestCase):
    def test_installs_inactive(self):
        installed = {}

        class FakeRegistry:
            def __init__(self, db=None):
                self.db = db

            def install(self, manifest):
                installed.update(manifest)
                return SimpleNamespace(name=manifest["name"])

            def deactivate(self, name):
                installed["active"] = 0
                return True

        steps = [_step() for _ in range(6)]
        memory = SimpleNamespace(user_message="find gigs", steps=steps)
        result = _result(True, 6)

        def llm(prompt):
            return ("NAME: test_skill\nDESCRIPTION: test\nTOOLS: web_search\n"
                    "WORKFLOW:\n1. Do it\n")

        with patch("nomorals.skills.registry.SkillRegistry", FakeRegistry):
            # bypass profile gate
            import nomorals.agents.skill_distillation as mod
            orig = mod.should_distill
            mod.should_distill = lambda r, c=None: True
            try:
                d = mod.maybe_distill(result, memory, llm_fn=llm)
            finally:
                mod.should_distill = orig
        self.assertIsNotNone(d)
        self.assertIn("distilled_", installed.get("name", ""))
        # the draft must never auto-activate: deactivate() clears the pin
        self.assertEqual(0, installed.get("active"))

    def test_skipped_when_not_warranted(self):
        import nomorals.agents.skill_distillation as mod
        orig = mod.should_distill
        mod.should_distill = lambda r, c=None: False
        try:
            self.assertIsNone(mod.maybe_distill(_result(True, 6), None))
        finally:
            mod.should_distill = orig

    def test_end_to_end_installs_inactive_in_real_db(self):
        """Full path with a real SkillRegistry: draft installs, pin cleared."""
        from nomorals.storage.db import Database

        db = Database(":memory:")
        steps = [_step() for _ in range(6)]
        memory = SimpleNamespace(user_message="find gigs", steps=steps)
        result = _result(True, 6)

        def llm(prompt):
            return ("NAME: test_skill\nDESCRIPTION: test\nTOOLS: web_search\n"
                    "WORKFLOW:\n1. Do it\n")

        import nomorals.agents.skill_distillation as mod
        orig = mod.should_distill
        mod.should_distill = lambda r, c=None: True
        try:
            d = mod.maybe_distill(result, memory, llm_fn=llm, db=db)
        finally:
            mod.should_distill = orig
        self.assertIsNotNone(d)
        row = db.query_one(
            "SELECT name, active, enabled FROM skill_packages WHERE name=?",
            (d.name,))
        self.assertIsNotNone(row)
        # never auto-activated: active pin cleared, still enabled/resolvable
        self.assertEqual(0, row["active"])
        self.assertEqual(1, row["enabled"])


if __name__ == "__main__":
    unittest.main()
