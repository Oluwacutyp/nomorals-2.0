"""Batch 3 (command inventory) + Batch 4 (style isolation) tests."""

import threading
import unittest

from nomorals.partner.style import (
    StyleProfile,
    format_reply,
    format_reply_detailed,
)
from nomorals.core.command_inventory import inventory


class TestCommandInventory(unittest.TestCase):
    def test_inventory_is_total_and_overlap_resolves(self):
        # The inventory lives in core (L1) and imports nothing, so the
        # command sets are injected by the caller (this test owns the
        # surfaces it audits).
        from nomorals.cmdline.dispatch import _COMMANDS
        from nomorals.social.chat.control import (
            CONTROL_COMMANDS, GAME_COMMANDS, PUBLIC_COMMANDS,
        )

        inv = inventory(
            cli_commands=set(_COMMANDS),
            chat_commands=(set(PUBLIC_COMMANDS) | set(GAME_COMMANDS)
                           | set(CONTROL_COMMANDS)))
        self.assertGreater(len(inv["cli_commands"]), 50)
        self.assertGreater(len(inv["chat_commands"]), 100)
        # Every chat command is classified: overlap, chat-only, or cli-only.
        classified = (set(inv["overlap"]) | set(inv["chat_only"])
                      | set(inv["cli_only"]))
        self.assertEqual(classified,
                         set(inv["chat_commands"]) | set(inv["cli_commands"]))
        # Overlap names resolve on BOTH surfaces (typed dispatch exists).
        self.assertIn("help", inv["overlap"])
        self.assertIn("search", inv["overlap"])
        # Unclassified drift fails loudly instead of silently diverging.
        self.assertEqual(inv["unclassified"], [])


class TestStyleIsolation(unittest.TestCase):
    def _profile(self, max_chars=400):
        from nomorals.partner.style import StyleMeasurement

        return StyleProfile(
            audience="owner", platform="telegram", max_chars=max_chars,
            target_chars=200, emoji_budget=1, allow_sections=True,
            allow_lists=True, allow_markup=True,
            measured=StyleMeasurement())

    def test_chat_theme_context_does_not_leak(self):
        from nomorals.social.chat import style as chat_style

        seen = {}

        def worker(name):
            with chat_style.theme_context(name):
                seen[threading.current_thread().name] = (
                    chat_style.current_theme())

        t1 = threading.Thread(target=worker, args=("plain",), name="t1")
        t2 = threading.Thread(target=worker, args=("terminal",), name="t2")
        t1.start()
        t2.start()
        t2.join()
        t1.join()
        self.assertEqual(seen["t1"], "plain")
        self.assertEqual(seen["t2"], "terminal")
        # Outside any context: the process default is untouched.
        self.assertEqual(chat_style.current_theme(), "rich")

    def test_core_theme_context_does_not_leak(self):
        from nomorals.core import style as core_style

        with core_style.theme_context("plain"):
            self.assertEqual(core_style.active_theme().name, "plain")
        self.assertEqual(core_style.active_theme().name, "ember")

    def test_format_reply_detailed_preserves_dropped_sections(self):
        profile = self._profile(max_chars=200)
        text = (
            "━━━━━━━━━━━━━━━\n**Section One**\n" + ("alpha. " * 60) + "\n"
            "━━━━━━━━━━━━━━━\n**Section Two**\n" + ("beta. " * 60) + "\n"
            "━━━━━━━━━━━━━━━\n**Section Three**\n" + ("gamma. " * 60) + "\n"
        )
        preview, overflow = format_reply_detailed(text, profile)
        self.assertLessEqual(len(preview), 200 + 120)  # marker slack
        self.assertIn("Section Three", overflow)
        # Nothing silently lost: preview + overflow cover every section.
        combined = preview + overflow
        for section in ("Section One", "Section Two", "Section Three"):
            self.assertIn(section, combined)
        # The legacy entrypoint still truncates (unchanged behaviour).
        self.assertNotIn("Section Three", format_reply(text, profile))


if __name__ == "__main__":
    unittest.main()
