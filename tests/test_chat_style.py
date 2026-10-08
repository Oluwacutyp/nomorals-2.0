"""Tests for Septorch-style output helpers and parse_mode threading."""

import unittest

from nomorals.social.chat.style import section, cmd, cta, bar, escape


class StyleTests(unittest.TestCase):
    def test_section(self):
        self.assertEqual("🎮 <b>ARENA</b>", section("🎮", "arena"))

    def test_cmd(self):
        self.assertEqual("<code>/game</code>", cmd("/game"))
        self.assertEqual("<code>/game</code>", cmd("game"))

    def test_cta(self):
        self.assertTrue(cta("go").startswith("🎯"))

    def test_bar(self):
        b = bar(0.6)
        self.assertIn("60%", b)
        self.assertIn("█", b)
        self.assertEqual("<code>░░░░░░░░░░</code> 0%", bar(0))
        self.assertEqual("<code>██████████</code> 100%", bar(1))

    def test_escape(self):
        self.assertEqual("&lt;tag&gt;", escape("<tag>"))
        # helpers escape their inputs
        self.assertNotIn("<script>", section("🎮", "<script>"))


class ParseModeTests(unittest.TestCase):
    def test_telegram_honors_parse_mode(self):
        import inspect
        from nomorals.social.chat.telegram import TelegramBotAdapter
        sig = inspect.signature(TelegramBotAdapter.send)
        self.assertIn("parse_mode", sig.parameters)

    def test_gateway_threads_parse_mode(self):
        import inspect
        from nomorals.social.chat.gateway import ChatGateway
        sig = inspect.signature(ChatGateway.send)
        self.assertIn("parse_mode", sig.parameters)

    def test_base_signature(self):
        import inspect
        from nomorals.social.chat.base import ChatAdapter
        sig = inspect.signature(ChatAdapter.send)
        self.assertIn("parse_mode", sig.parameters)


if __name__ == "__main__":
    unittest.main()
