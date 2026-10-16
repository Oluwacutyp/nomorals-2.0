"""Tests for nomorals.focus.interruption (focus × interruption wiring)."""

import unittest

from nomorals.autonomy.interruption import assess_worth  # noqa: F401 — import path check
from nomorals.focus.interruption import should_interrupt
from nomorals.focus.sessions import FocusTimer

LOW = {"priority": "normal", "urgency": 0.5, "importance": 0.5,
       "title": "routine ping"}  # policy: defer (score 0.45)
HIGH = {"priority": "safety", "urgency": 1.0, "importance": 1.0,
        "title": "safety alert"}  # policy: interrupt
DROP = {"priority": "background", "urgency": 0.1, "importance": 0.1,
        "title": "background noise"}  # policy: drop


class TestFocusInterruption(unittest.TestCase):
    def test_import_works(self):
        self.assertTrue(callable(should_interrupt))

    def test_high_worth_passes_during_focus(self):
        result = should_interrupt("deep work", HIGH, assess_fn=assess_worth)
        self.assertTrue(result["worth_it"])
        self.assertEqual(result["decision"], "interrupt")
        self.assertTrue(result["in_focus"])

    def test_low_worth_blocked_during_focus(self):
        result = should_interrupt("deep work", LOW, assess_fn=assess_worth)
        self.assertFalse(result["worth_it"])
        self.assertEqual(result["decision"], "defer")
        self.assertTrue(result["in_focus"])
        self.assertIn("focus", result["reason"])

    def test_low_worth_worth_surfacing_when_idle(self):
        # Same interruption, no session in progress → worth surfacing.
        result = should_interrupt("", LOW, assess_fn=assess_worth)
        self.assertTrue(result["worth_it"])
        self.assertEqual(result["decision"], "defer")
        self.assertFalse(result["in_focus"])

    def test_drop_never_worth_it(self):
        self.assertFalse(should_interrupt("deep work", DROP, assess_fn=assess_worth)["worth_it"])
        self.assertFalse(should_interrupt("", DROP, assess_fn=assess_worth)["worth_it"])

    def test_timer_public_api_drives_focus(self):
        timer = FocusTimer()  # in-memory, no path
        session = timer.start("deep work")
        try:
            blocked = should_interrupt("", LOW, timer=timer, assess_fn=assess_worth)
            self.assertTrue(blocked["in_focus"])
            self.assertFalse(blocked["worth_it"])
            passing = should_interrupt("", HIGH, timer=timer, assess_fn=assess_worth)
            self.assertTrue(passing["worth_it"])
        finally:
            timer.stop(session.id)
        idle = should_interrupt("", LOW, timer=timer, assess_fn=assess_worth)
        self.assertFalse(idle["in_focus"])
        self.assertTrue(idle["worth_it"])

    def test_explicit_focus_active_override(self):
        self.assertFalse(
            should_interrupt("", LOW, focus_active=True, assess_fn=assess_worth)["worth_it"])
        self.assertTrue(
            should_interrupt("deep work", LOW, focus_active=False, assess_fn=assess_worth)["worth_it"])

    def test_context_focus_flag(self):
        self.assertFalse(
            should_interrupt("", LOW, context={"focus_active": True}, assess_fn=assess_worth)["worth_it"])
        self.assertTrue(
            should_interrupt("", LOW, context={"focus_active": False}, assess_fn=assess_worth)["worth_it"])

    def test_delegates_to_assess_worth(self):
        result = should_interrupt("deep work", HIGH, assess_fn=assess_worth)
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["policy_reasons"])


if __name__ == "__main__":
    unittest.main()
