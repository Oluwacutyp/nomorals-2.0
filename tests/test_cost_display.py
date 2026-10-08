"""Build-map extension #7 — transparent cost display (Hercules steal).

"this answer cost $0.003" after every response, toggleable per user,
plus the budgeted-research NL ("research this with a $0.50 budget").

All offline.  Never raises.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.llm.cost_display import (
    CostDisplay,
    format_cost,
    get_display,
    maybe_cost_footer,
    parse_budget_nl,
    control_cost,
)


class FormatCostTests(unittest.TestCase):
    def test_zero_is_free(self):
        self.assertEqual(format_cost(0.0), "free")
        self.assertEqual(format_cost(-1.0), "free")

    def test_small(self):
        self.assertEqual(format_cost(0.003), "$0.003")
        self.assertEqual(format_cost(0.0008), "$0.0008")

    def test_normal(self):
        self.assertEqual(format_cost(0.50), "$0.50")
        self.assertEqual(format_cost(1.20), "$1.20")

    def test_large(self):
        self.assertEqual(format_cost(1234.5), "$1,234.50")

    def test_garbage(self):
        self.assertEqual(format_cost(None), "free")
        self.assertEqual(format_cost("junk"), "free")


class ParseBudgetTests(unittest.TestCase):
    def test_with_budget(self):
        self.assertEqual(parse_budget_nl("research this with a $0.50 budget"), 0.50)

    def test_budget_first(self):
        self.assertEqual(parse_budget_nl("budget $2 for this research"), 2.0)

    def test_spend(self):
        self.assertEqual(parse_budget_nl("spend $0.50 researching this"), 0.50)

    def test_no_budget(self):
        self.assertIsNone(parse_budget_nl("research this topic"))
        self.assertIsNone(parse_budget_nl(""))

    def test_garbage(self):
        self.assertIsNone(parse_budget_nl(None))
        self.assertIsNone(parse_budget_nl("budget $0"))


class CostDisplayTests(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.disp = CostDisplay(db_path=self.db)

    def test_toggle(self):
        self.assertFalse(self.disp.is_enabled())
        self.disp.set_enabled(True)
        self.assertTrue(self.disp.is_enabled())
        self.disp.set_enabled(False)
        self.assertFalse(self.disp.is_enabled())

    def test_toggle_per_user(self):
        self.disp.set_enabled(True, "owner")
        self.assertFalse(self.disp.is_enabled("someone_else"))

    def test_footer_disabled(self):
        self.assertEqual(self.disp.footer_for_turn(time.time()), "")

    def test_footer_enabled(self):
        self.disp.set_enabled(True)
        footer = self.disp.footer_for_turn(time.time() - 3600)
        self.assertIn("this answer cost", footer)

    def test_footer_with_spend(self):
        # Write a fake cost-log entry, then check the footer picks it up.
        cost_path = Path(tempfile.mktemp(suffix=".jsonl"))
        cost_path.write_text(json.dumps({
            "ts": time.time(), "provider": "groq", "model": "x",
            "cost_usd": 0.003, "operation": "chat",
        }) + "\n")
        self.disp.set_enabled(True)
        footer = self.disp.footer_for_turn(time.time() - 60, cost_path=cost_path)
        self.assertIn("$0.003", footer)

    def test_today_spend(self):
        cost_path = Path(tempfile.mktemp(suffix=".jsonl"))
        cost_path.write_text(json.dumps({
            "ts": time.time(), "provider": "groq",
            "cost_usd": 0.01, "operation": "chat",
        }) + "\n")
        spent = self.disp.today_spend(cost_path=cost_path)
        self.assertAlmostEqual(spent, 0.01, places=6)

    def test_today_spend_empty(self):
        self.assertEqual(self.disp.today_spend(cost_path="/nonexistent-xyz.jsonl"), 0.0)

    def test_never_raises(self):
        bad = CostDisplay(db_path="/nonexistent-dir-xyz/abc.db")
        self.assertFalse(bad.is_enabled())
        self.assertFalse(bad.set_enabled(True))
        self.assertEqual(bad.footer_for_turn(time.time()), "")
        # nonexistent cost log → 0.0
        self.assertEqual(bad.today_spend(cost_path="/nonexistent-xyz.jsonl"), 0.0)


class MaybeFooterTests(unittest.TestCase):
    def test_disabled_passthrough(self):
        disp = CostDisplay(db_path=tempfile.mktemp(suffix=".db"))
        out = maybe_cost_footer("hello", display=disp)
        self.assertEqual(out, "hello")

    def test_enabled_appends(self):
        disp = CostDisplay(db_path=tempfile.mktemp(suffix=".db"))
        disp.set_enabled(True)
        out = maybe_cost_footer("hello", since_ts=time.time() - 3600, display=disp)
        self.assertTrue(out.startswith("hello"))
        self.assertIn("this answer cost", out)

    def test_never_raises(self):
        import tempfile
        disp = CostDisplay(db_path=tempfile.mktemp(suffix=".db"))
        self.assertEqual(maybe_cost_footer(None, display=disp), "")


class ControlCostTests(unittest.TestCase):
    def setUp(self):
        self.disp = CostDisplay(db_path=tempfile.mktemp(suffix=".db"))

    def test_usage(self):
        out = control_cost("garbage xyz", display=self.disp)
        self.assertIn("/cost", out)

    def test_on_off(self):
        out = control_cost("on", display=self.disp)
        self.assertIn("on", out)
        self.assertTrue(self.disp.is_enabled())
        out = control_cost("off", display=self.disp)
        self.assertTrue(self.disp.is_enabled() is False)

    def test_status(self):
        out = control_cost("", display=self.disp)
        self.assertIn("spend today", out)
        self.assertIn("off", out)

    def test_budget(self):
        out = control_cost("budget $0.50", display=self.disp)
        self.assertIn("$0.50", out)

    def test_budget_bad(self):
        out = control_cost("budget xyz", display=self.disp)
        self.assertIn("usage", out.lower())

    def test_never_raises(self):
        self.assertIsInstance(control_cost(None, display=self.disp), str)
        self.assertIsInstance(control_cost("on", display=None), str)


class GetDisplayTests(unittest.TestCase):
    def test_singleton(self):
        self.assertIs(get_display(), get_display())


if __name__ == "__main__":
    unittest.main()
