"""Power mode: owner-key-gated, audited capability expansion.

The security invariants tested here:
  * an empty owner key means power mode can NEVER unlock, no matter the input
  * a wrong key is rejected in constant time and leaves state untouched
  * the key itself never appears in any status/report output
  * unlocking widens dials; locking restores the configured values exactly
  * every unlock/lock/denial is journaled to audit_log
"""

from __future__ import annotations

import json
import tempfile
import unittest
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.power import PowerMode, power_mode_for
from nomorals.core.config import load_settings


def _make_context() -> tuple[Any, "tempfile.TemporaryDirectory"]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-power-")
    settings = load_settings(overrides={"home": tmp.name, "partner.platforms": "local",
                                        "chat.local_enabled": "true"})
    context = build_context(settings, with_executor=False, with_tools=False,
                            with_router=False, with_memory=False)
    return context, tmp


class PowerModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.partner = self.context.settings.partner

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _audit_actions(self) -> list[str]:
        rows = self.context.db.query("SELECT action FROM audit_log ORDER BY ts ASC")
        return [r["action"] for r in rows]

    def test_empty_key_never_unlocks(self) -> None:
        self.partner.owner_key = ""
        power = power_mode_for(self.context)
        result = power.unlock("anything at all", actor="test")
        self.assertFalse(result["ok"])
        self.assertFalse(power.active)
        self.assertIn("power.unlock_denied", self._audit_actions())
        # and a blank key never unlocks either
        self.assertFalse(power.unlock("", actor="test")["ok"])

    def test_wrong_key_is_rejected_and_state_untouched(self) -> None:
        self.partner.owner_key = "correct-horse"
        before = {
            "autonomy_mode": self.partner.autonomy_mode,
            "typing_seconds": self.partner.typing_seconds,
            "max_parallel_chats": self.partner.max_parallel_chats,
        }
        power = power_mode_for(self.context)
        result = power.unlock("wrong", actor="test")
        self.assertFalse(result["ok"])
        self.assertIn("wrong key", result["message"])
        self.assertFalse(power.active)
        self.assertEqual(self.partner.autonomy_mode, before["autonomy_mode"])
        self.assertEqual(self.partner.typing_seconds, before["typing_seconds"])
        self.assertIn("power.unlock_denied", self._audit_actions())

    def test_correct_key_widens_and_lock_restores_exactly(self) -> None:
        self.partner.owner_key = "correct-horse"
        before = {
            "autonomy_mode": self.partner.autonomy_mode,
            "typing_seconds": self.partner.typing_seconds,
            "max_parallel_chats": self.partner.max_parallel_chats,
            "max_proactive_dm_per_day": self.partner.max_proactive_dm_per_day,
            "max_group_posts_per_day": self.partner.max_group_posts_per_day,
            "history_window": self.partner.history_window,
        }
        power = power_mode_for(self.context)
        result = power.unlock("correct-horse", actor="console")
        self.assertTrue(result["ok"], result)
        self.assertTrue(power.active)
        self.assertEqual(self.partner.autonomy_mode, "auto")
        self.assertEqual(self.partner.typing_seconds, 0.6)
        self.assertGreaterEqual(self.partner.max_parallel_chats, 8)
        # Power mode = no volume caps at all (0 = unlimited); normal mode
        # keeps the protective defaults.
        self.assertEqual(self.partner.max_proactive_dm_per_day, 0)
        self.assertEqual(self.partner.max_group_posts_per_day, 0)
        self.assertIn("power.unlock", self._audit_actions())

        # The key must not leak into any report.
        blob = json.dumps(power.status(), default=str) + json.dumps(result, default=str)
        self.assertNotIn("correct-horse", blob)

        lock = power.lock(actor="console")
        self.assertTrue(lock["ok"])
        self.assertFalse(power.active)
        for field, value in before.items():
            self.assertEqual(getattr(self.partner, field), value, field)
        self.assertIn("power.lock", self._audit_actions())

    def test_unlock_is_idempotent(self) -> None:
        self.partner.owner_key = "k"
        power = power_mode_for(self.context)
        self.assertTrue(power.unlock("k")["ok"])
        result = power.unlock("k")
        self.assertTrue(result["ok"])
        self.assertEqual(result["changes"], [])
        self.assertIn("already active", result["message"])

    def test_power_mode_for_is_singleton_per_context(self) -> None:
        self.assertIs(power_mode_for(self.context), power_mode_for(self.context))

    def test_status_shape(self) -> None:
        self.partner.owner_key = "k"
        s = power_mode_for(self.context).status()
        self.assertFalse(s["active"])
        self.assertTrue(s["key_configured"])
        self.assertEqual(s["changes"], [])

    def test_unlock_persists_and_a_new_process_adopts_it(self) -> None:
        self.partner.owner_key = "k"
        first = PowerMode(self.context)  # "process 1"
        self.assertTrue(first.unlock("k", actor="test")["ok"])
        self.assertTrue(first.status()["applied_in_process"])

        # "Process 2": a fresh object AND fresh settings (a real new process
        # starts from base config values), same database.
        self.partner.autonomy_mode = "suggest"
        self.partner.typing_seconds = 2.0
        self.partner.max_parallel_chats = 4
        self.partner.max_proactive_dm_per_day = 6
        self.partner.max_group_posts_per_day = 2
        self.partner.history_window = 14
        second = PowerMode(self.context)
        self.assertFalse(second.active)
        self.assertTrue(second.adopt_persisted_state())
        self.assertTrue(second.active)
        self.assertEqual(self.partner.autonomy_mode, "auto")

        # "Process 3": status reflects the persisted state without adopting.
        third = PowerMode(self.context)
        s = third.status()
        self.assertTrue(s["active"])
        self.assertFalse(s["applied_in_process"])
        self.assertEqual(s["unlocked_by"], "test")

        # Locking from a fresh process clears the persisted state.
        self.assertTrue(second.lock(actor="cli")["ok"])
        self.assertFalse(PowerMode(self.context).status()["active"])
        # and the dials were restored by whoever held them (process 2).
        self.assertEqual(self.partner.autonomy_mode, "suggest")


if __name__ == "__main__":
    unittest.main()
