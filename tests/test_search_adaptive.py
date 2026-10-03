"""Tests for the dynamic-everything audit: adaptive search result counts
and adaptive morning-briefing section budgets (no more hardcoded 8s)."""
from __future__ import annotations

import unittest

from nomorals.search.adaptive import adaptive_result_limit


class AdaptiveLimitTests(unittest.TestCase):
    def test_empty_query_returns_base(self):
        self.assertEqual(adaptive_result_limit(""), 8)
        self.assertEqual(adaptive_result_limit("   "), 8)

    def test_short_factoid_gets_fewer_than_broad_research(self):
        factoid = adaptive_result_limit("bitcoin price")
        research = adaptive_result_limit(
            "compare the best paid survey sites that actually pay Nigerians")
        self.assertLess(factoid, research)

    def test_breadth_signals_increase_limit(self):
        plain = adaptive_result_limit("project management software")
        broad = adaptive_result_limit(
            "best project management software alternatives compared")
        self.assertLess(plain, broad)

    def test_question_shape_increases_limit(self):
        self.assertLess(
            adaptive_result_limit("lagos traffic"),
            adaptive_result_limit("why is lagos traffic so bad?"))

    def test_clamped_to_floor_and_ceiling(self):
        self.assertGreaterEqual(adaptive_result_limit("x"), 4)
        self.assertLessEqual(
            adaptive_result_limit("compare " + "the best options " * 40), 24)

    def test_custom_base_floor_ceiling(self):
        n = adaptive_result_limit("bitcoin price", base=10, floor=5,
                                  ceiling=12)
        self.assertTrue(5 <= n <= 12)


class _StubTools:
    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def call(self, name, **kw):
        self.last_kwargs = kw
        return _Ok({"results": []})


class _Ok:
    def __init__(self, value) -> None:
        self.ok = True
        self.value = value
        self.error = None

    def unwrap(self):
        return self.value


class _Ctx:
    settings = None
    tools = None


class EngineAdaptiveSearchTests(unittest.TestCase):
    def _engine(self):
        from nomorals.agents.search.engine import SearchEngine
        ctx = _Ctx()
        ctx.tools = _StubTools()
        return SearchEngine(ctx), ctx

    def test_default_is_adaptive_not_fixed_8(self):
        eng, ctx = self._engine()
        eng.search("bitcoin price")
        factoid_n = ctx.tools.last_kwargs["max_results"]
        eng.search("compare the best paid survey sites that actually pay")
        research_n = ctx.tools.last_kwargs["max_results"]
        self.assertLess(factoid_n, research_n)
        self.assertNotEqual(factoid_n, 8,
                            "a factoid must not get the old fixed default")

    def test_explicit_max_results_always_wins(self):
        eng, ctx = self._engine()
        eng.search("anything at all here", max_results=3)
        self.assertEqual(ctx.tools.last_kwargs["max_results"], 3)


class BriefingBudgetTests(unittest.TestCase):
    def test_single_section_gets_full_room(self):
        import nomorals.agents.morning_briefing as mb
        budgets = mb.adaptive_section_budget(object(), [10])
        self.assertEqual(budgets, [15])

    def test_more_sections_means_smaller_each(self):
        import nomorals.agents.morning_briefing as mb
        few = mb.adaptive_section_budget(object(), [10, 20])
        many = mb.adaptive_section_budget(object(), [10, 20, 30, 40, 50,
                                                     60, 70, 80])
        self.assertLess(sum(many) / len(many), sum(few) / len(few))

    def test_high_priority_section_gets_more_than_low(self):
        import nomorals.agents.morning_briefing as mb
        budgets = mb.adaptive_section_budget(object(), [10, 100])
        self.assertGreater(budgets[0], budgets[1])

    def test_owner_pref_overrides_adaptive(self):
        import nomorals.agents.morning_briefing as mb
        ctx = _Ctx()
        ctx.settings = type("S", (), {"workspace_dir": "/tmp",
                                      "briefing": None})()
        # settings.briefing namespace path with the pref set
        ns = type("N", (), {"max_items_per_section": 5})()
        ctx.settings = type("S2", (), {"briefing": ns,
                                       "workspace_dir": "/tmp"})()
        budgets = mb.adaptive_section_budget(ctx, [10, 20, 100])
        self.assertEqual(budgets, [5, 5, 5])

    def test_composer_clips_sections_to_budget(self):
        import nomorals.agents.morning_briefing as mb
        composer = mb.BriefingComposer.__new__(mb.BriefingComposer)
        secs = [mb.BriefingSection(name="a", title="A", priority=10,
                                   lines=[f"l{i}" for i in range(50)],
                                   items=[{"title": f"t{i}"}
                                          for i in range(50)])]
        composer._apply_section_budgets(object(), secs)
        self.assertEqual(len(secs[0].lines), 15)
        self.assertEqual(len(secs[0].items), 15)
        self.assertLess(len(secs[0].lines), 50)


if __name__ == "__main__":
    unittest.main()
