"""Tests for nomorals.context.budget: ContextBudget.fit semantics."""

from __future__ import annotations

import unittest

from nomorals.context import ContextBudget, Section


def _section(name: str, words: int, *, priority: float,
             load_bearing: bool = False) -> Section:
    return Section(
        name=name,
        content=" ".join(f"w{i}" for i in range(words)),
        priority=priority,
        load_bearing=load_bearing,
    )


class BudgetTrimOrderTest(unittest.TestCase):
    def test_trims_lowest_priority_first(self) -> None:
        budget = ContextBudget(total=200)
        history = _section("history", 120, priority=20.0)
        tools = _section("tools", 120, priority=60.0)
        mission = _section("mission", 60, priority=90.0, load_bearing=True)
        survivors = budget.fit([mission, tools, history])
        names = [s.name for s in survivors]
        # history (lowest priority) is sacrificed before tools.
        self.assertNotIn("history", names)
        self.assertIn("tools", names)
        self.assertIn("mission", names)
        self.assertTrue(history.dropped)
        self.assertFalse(tools.dropped)

    def test_survivors_keep_original_order(self) -> None:
        budget = ContextBudget(total=10_000)
        sections = [
            _section("history", 10, priority=20.0),
            _section("system", 10, priority=100.0, load_bearing=True),
            _section("tools", 10, priority=60.0),
        ]
        survivors = budget.fit(sections)
        self.assertEqual([s.name for s in survivors],
                         ["history", "system", "tools"])

    def test_zero_cost_sections_never_dropped(self) -> None:
        budget = ContextBudget(total=50)
        big = _section("history", 500, priority=20.0)
        empty = Section(name="user_profile", content="", priority=30.0)
        survivors = budget.fit([big, empty])
        self.assertIn(empty, survivors)
        self.assertFalse(empty.dropped)


class LoadBearingTest(unittest.TestCase):
    def test_load_bearing_survives_extreme_budget(self) -> None:
        # Even a budget smaller than the load-bearing content keeps it.
        mission = Section(
            name="mission",
            content="Acceptance criteria:\n- the widget must frobnicate",
            priority=90.0,
            load_bearing=True,
        )
        history = _section("history", 300, priority=20.0)
        budget = ContextBudget(total=5)
        survivors = budget.fit([mission, history])
        self.assertIn(mission, survivors)
        self.assertNotIn(history, survivors)
        self.assertIn("frobnicate", mission.content)

    def test_overrun_recorded_not_hidden(self) -> None:
        mission = _section("mission", 500, priority=90.0, load_bearing=True)
        budget = ContextBudget(total=100)
        budget.fit([mission])
        self.assertGreater(budget.last_overrun, 0)

    def test_no_overrun_when_it_fits(self) -> None:
        mission = _section("mission", 50, priority=90.0, load_bearing=True)
        budget = ContextBudget(total=10_000)
        budget.fit([mission])
        self.assertEqual(budget.last_overrun, 0)


class AllocationTest(unittest.TestCase):
    def test_default_allocations_cover_named_sections(self) -> None:
        budget = ContextBudget.default(total=8000)
        for name in ("system", "mission", "artifacts", "tools",
                     "project", "user_profile", "history"):
            self.assertGreater(budget.allocation_for(name), 0, name)

    def test_total_scales_allocations(self) -> None:
        small = ContextBudget.default(total=1000)
        big = ContextBudget.default(total=8000)
        self.assertLess(small.allocation_for("history"),
                        big.allocation_for("history"))

    def test_unknown_section_gets_floor(self) -> None:
        budget = ContextBudget.default(total=8000)
        self.assertGreaterEqual(budget.allocation_for("never_heard_of_it"), 16)

    def test_rejects_non_positive_total(self) -> None:
        with self.assertRaises(ValueError):
            ContextBudget(total=0)


if __name__ == "__main__":
    unittest.main()
