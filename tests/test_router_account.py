"""Router account-intent + connector-awareness tests.

Regression tests for the live Termux findings (2026-10-03):
- "Create a sound cloud account and send me the logins" was routed to
  the CODE BUILDER instead of the account creator.
- "How do I link you to my Spotify" got "i don't have a way to link up
  with Spotify" — the model didn't know the Spotify connector exists.
"""
from __future__ import annotations

import unittest

from nomorals.agents.coremind import (
    CoreMind,
    _account_intent,
    understand,
)
from nomorals.partner.context import PartnerContextBuilder


class AccountIntentTests(unittest.TestCase):
    def test_create_service_account_routes_to_account(self):
        it = _account_intent("create a spotify account")
        self.assertIsNotNone(it)
        self.assertEqual(it.kind, "account")
        self.assertEqual(it.route, "account")
        self.assertEqual(it.meta.get("service"), "spotify")
        self.assertGreaterEqual(it.confidence, 0.8)

    def test_soundcloud_multiword_service(self):
        # the exact live failure: must not go to the code builder
        it = _account_intent(
            "Create a sound cloud account and send me the logins")
        self.assertIsNotNone(it)
        self.assertEqual(it.kind, "account")
        self.assertEqual(it.meta.get("service"), "soundcloud")

    def test_make_me_github_account(self):
        it = _account_intent("make me a github account")
        self.assertIsNotNone(it)
        self.assertEqual(it.meta.get("service"), "github")

    def test_signup_pattern(self):
        it = _account_intent("sign me up for twitter")
        self.assertIsNotNone(it)
        self.assertEqual(it.meta.get("service"), "twitter")

    def test_bare_account_asks_which_service(self):
        it = _account_intent("create an account")
        self.assertIsNotNone(it)
        self.assertEqual(it.kind, "account")
        self.assertEqual(it.action, "ask")

    def test_user_account_system_is_coding_not_account(self):
        # the app's own user system is a build task, not account creation
        self.assertIsNone(_account_intent("create a user account system"))
        self.assertIsNone(_account_intent("create a user accounts table"))

    def test_build_intent_untouched(self):
        self.assertIsNone(_account_intent("build a todo app"))
        self.assertIsNone(_account_intent("write me a story"))

    def test_account_beats_build_in_understand(self):
        cands = understand("Create a sound cloud account and send me the logins")
        self.assertTrue(cands)
        top = cands[0]
        self.assertEqual(top.kind, "account")
        self.assertEqual(top.route, "account")

    def test_spotify_account_beats_build(self):
        cands = understand("create a spotify account")
        self.assertTrue(cands)
        self.assertEqual(cands[0].kind, "account")

    def test_account_intent_bypasses_model_check(self):
        # >= 0.8 confidence goes straight to dispatch, no model round-trip
        it = _account_intent("create a spotify account")
        self.assertGreaterEqual(it.confidence, 0.8)


class AccountDispatchTests(unittest.TestCase):
    def test_dispatch_table_has_account_route(self):
        # the CoreMind._dispatch table must know the "account" kind
        import inspect
        src = inspect.getsource(CoreMind._dispatch)
        self.assertIn('"account"', src)

    def test_dispatch_account_method_exists(self):
        self.assertTrue(callable(getattr(CoreMind, "_dispatch_account", None)))

    def test_account_ask_without_service(self):
        import threading
        from unittest import mock
        mind = CoreMind.__new__(CoreMind)
        mind._lock = threading.Lock()
        mind.context = mock.Mock()
        mind.context.db = None
        # stub _job_done to avoid state-file writes
        mind._job_done = lambda *a, **k: None
        from nomorals.agents.coremind import Intent
        intent = Intent("account", 0.7, target="", action="ask",
                        route="account", why="test")
        # _dispatch_account asks which service without touching the vault
        reply = CoreMind._dispatch_account(mind, intent, "job1", "chat1", None)
        self.assertIn("which service", reply.lower())


class ConnectorAwarenessTests(unittest.TestCase):
    def test_capabilities_block_lists_spotify(self):
        block = PartnerContextBuilder._capabilities_block()
        self.assertIn("spotify", block.lower())

    def test_capabilities_block_lists_soundcloud(self):
        block = PartnerContextBuilder._capabilities_block()
        self.assertIn("soundcloud", block.lower())

    def test_capabilities_block_has_use_instruction(self):
        block = PartnerContextBuilder._capabilities_block()
        self.assertIn("never claim you can't", block)

    def test_capabilities_block_in_system_prompt(self):
        import inspect
        src = inspect.getsource(PartnerContextBuilder.build)
        self.assertIn("capabilities", src)


class SimpleScriptTests(unittest.TestCase):
    """Simple scripts should just run — no confusing "nothing ran" output."""

    def test_no_tests_dir_means_no_runner(self):
        import tempfile
        from pathlib import Path
        from nomorals.agents.coding import _workdir_has_tests
        with tempfile.TemporaryDirectory() as d:
            Path(d, "main.py").write_text("print('hi')")
            self.assertFalse(_workdir_has_tests(d))

    def test_tests_dir_means_runner(self):
        import os
        import tempfile
        from pathlib import Path
        from nomorals.agents.coding import _workdir_has_tests
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "tests"))
            self.assertTrue(_workdir_has_tests(d))

    def test_test_file_means_runner(self):
        import tempfile
        from pathlib import Path
        from nomorals.agents.coding import _workdir_has_tests
        with tempfile.TemporaryDirectory() as d:
            Path(d, "test_main.py").write_text("x = 1")
            self.assertTrue(_workdir_has_tests(d))


class MonitorNlTests(unittest.TestCase):
    """Natural-language /monitor parsing."""

    def test_btc_price_two_minutes(self):
        from nomorals.agents.partner.runtime_intel import parse_monitor_nl
        res = parse_monitor_nl("btc price movement in the next 2 minutes")
        self.assertIsNotNone(res)
        self.assertEqual(res["target"], "btc price")
        self.assertEqual(res["duration_s"], 120.0)
        self.assertGreaterEqual(res["interval_s"], 30.0)

    def test_watch_with_interval(self):
        from nomorals.agents.partner.runtime_intel import parse_monitor_nl
        res = parse_monitor_nl("watch eth every 30 seconds")
        self.assertIsNotNone(res)
        self.assertEqual(res["target"], "eth")
        self.assertEqual(res["interval_s"], 30.0)

    def test_monitor_for_duration(self):
        from nomorals.agents.partner.runtime_intel import parse_monitor_nl
        res = parse_monitor_nl("monitor bitcoin price for 1 hour")
        self.assertIsNotNone(res)
        self.assertEqual(res["target"], "bitcoin price")
        self.assertEqual(res["duration_s"], 3600.0)

    def test_plain_text_is_none(self):
        from nomorals.agents.partner.runtime_intel import parse_monitor_nl
        self.assertIsNone(parse_monitor_nl("hello world"))
        self.assertIsNone(parse_monitor_nl(""))
        self.assertIsNone(parse_monitor_nl("/monitor add x"))


if __name__ == "__main__":
    unittest.main()
