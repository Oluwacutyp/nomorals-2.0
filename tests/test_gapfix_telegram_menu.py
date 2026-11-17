"""R3-49: Telegram menu coverage — BotFather sync from the registry.

Failing-first: the gated bot has no registry-derived BotFather list, nothing
pushes either bot's list via setMyCommands, and no startup hook or CLI
command performs the sync. These tests pin the wired behaviour.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from unittest import mock

from nomorals.social.chat.control import CONTROL_COMMANDS, PUBLIC_COMMANDS
from nomorals.social.chat.personal_commands import (
    BOTFATHER_MAX_COMMANDS,
    BOTFATHER_MAX_DESCRIPTION,
    _public_priority,
    personal_botfather_commands,
)


class _FakeResp:
    def __init__(self, payload=None):
        self._payload = payload or {"ok": True, "result": True}

    def json(self):
        return self._payload


class _FakeSession:
    """Records Bot API calls; never touches the network."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def post_json(self, url, params, timeout=None):
        self.calls.append((url, dict(params)))
        method = url.rsplit("/", 1)[-1]
        if method == "getMe":
            return _FakeResp({"ok": True,
                              "result": {"id": 123456, "username": "TestBot"}})
        if method == "getUpdates":
            return _FakeResp({"ok": True, "result": []})
        return _FakeResp({"ok": True, "result": True})

    def get(self, url, timeout=None):
        self.calls.append((url, {}))
        return _FakeResp()


class _FakeAdapter:
    """Duck-typed stand-in for TelegramBotAdapter (sync function only)."""

    def __init__(self, *, owner_only=False, name="telegram-bot"):
        self.owner_only = owner_only
        self.name = name
        self.pushed: list[list[tuple[str, str]]] = []

    def set_default_commands(self, commands):
        self.pushed.append(list(commands))
        return {"ok": True, "result": True}


class GatedListTests(unittest.TestCase):
    def test_gated_list_is_first_100_public_priority(self):
        from nomorals.social.chat.personal_commands import (
            gated_botfather_commands,
        )

        entries = gated_botfather_commands()
        self.assertEqual(len(entries), BOTFATHER_MAX_COMMANDS)
        self.assertEqual([k for k, _ in entries], _public_priority()[:100])
        for kind, desc in entries:
            self.assertIn(kind, PUBLIC_COMMANDS)
            self.assertIn(kind, CONTROL_COMMANDS)
            self.assertTrue(desc)
            self.assertLessEqual(len(desc), BOTFATHER_MAX_DESCRIPTION)
        self.assertEqual(len({k for k, _ in entries}), len(entries))

    def test_gated_list_validates_clean(self):
        from nomorals.social.chat.personal_commands import (
            validate_gated_botfather_commands,
        )

        self.assertEqual(validate_gated_botfather_commands(), [])

    def test_personal_list_unchanged(self):
        from nomorals.social.chat.personal_commands import (
            validate_botfather_commands,
        )

        entries = personal_botfather_commands()
        self.assertEqual(len(entries), BOTFATHER_MAX_COMMANDS)
        self.assertEqual(validate_botfather_commands(), [])


class ValidateMenuDriftTests(unittest.TestCase):
    def test_validate_menu_names_a_missing_registered_command(self):
        # The menu is built from the registry, so a registry edit cannot
        # drift it — the gate is proved by feeding it a WRONG menu: drop
        # one kind from what build_telegram_menu returns and the validator
        # must name it.
        import nomorals.social.chat.telegram_menu as menu_mod
        from nomorals.social.chat.telegram_menu import validate_menu

        real_build = menu_mod.build_telegram_menu

        def _dropping(is_owner):
            built = real_build(is_owner)
            for kinds in built.values():
                if "help" in kinds:
                    kinds.remove("help")
            return built

        with mock.patch.object(menu_mod, "build_telegram_menu",
                               side_effect=_dropping):
            ok, problems = validate_menu()
        self.assertFalse(ok)
        self.assertTrue(any("/help" in p for p in problems))


class SyncFunctionTests(unittest.TestCase):
    def _sync(self):
        from nomorals.social.chat.personal_commands import sync_bot_commands

        return sync_bot_commands

    def test_sync_pushes_gated_list_to_default_scope(self):
        adapter = _FakeAdapter(owner_only=False)
        out = self._sync()(adapter)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["bot"], "telegram-bot")
        self.assertEqual(out["pushed"], 100)
        self.assertEqual(len(adapter.pushed), 1)
        from nomorals.social.chat.personal_commands import (
            gated_botfather_commands,
        )

        self.assertEqual(adapter.pushed[0], gated_botfather_commands())

    def test_sync_pushes_personal_list_to_personal_bot(self):
        adapter = _FakeAdapter(owner_only=True, name="telegram-personal")
        out = self._sync()(adapter)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["bot"], "telegram-personal")
        self.assertEqual(out["pushed"], 100)
        self.assertEqual(adapter.pushed[0], personal_botfather_commands())

    def test_sync_dry_run_pushes_nothing(self):
        adapter = _FakeAdapter(owner_only=False)
        out = self._sync()(adapter, dry_run=True)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["pushed"], 0)
        self.assertEqual(adapter.pushed, [])

    def test_sync_refuses_when_validation_fails(self):
        adapter = _FakeAdapter(owner_only=False)
        with mock.patch(
            "nomorals.social.chat.telegram_menu.validate_menu",
            return_value=(False, ["owner menu missing registered command /zz"]),
        ):
            out = self._sync()(adapter)
        self.assertFalse(out["ok"])
        self.assertTrue(any("zz" in p for p in out["problems"]))
        self.assertEqual(adapter.pushed, [])

    def test_sync_reports_the_cap_remainder(self):
        adapter = _FakeAdapter(owner_only=False)
        out = self._sync()(adapter)
        self.assertEqual(out["skipped"], len(_public_priority()) - 100)
        self.assertTrue(out["skipped"] > 0)


class AdapterMethodTests(unittest.TestCase):
    def _adapter(self, **kw):
        from nomorals.social.chat.telegram import TelegramBotAdapter

        kw.setdefault("token", "123456:test-token")
        kw["session"] = _FakeSession()
        return TelegramBotAdapter(**kw)

    def test_set_default_commands_calls_setmycommands(self):
        adapter = self._adapter()
        out = adapter.set_default_commands([("help", "this help."),
                                            ("ping", "pong.")])
        self.assertTrue(out["ok"], out)
        session = adapter._session
        calls = [(u, p) for u, p in session.calls if "setMyCommands" in u]
        self.assertEqual(len(calls), 1)
        params = calls[0][1]
        self.assertEqual(json.loads(params["scope"]), {"type": "default"})
        self.assertEqual(
            json.loads(params["commands"]),
            [{"command": "help", "description": "this help."},
             {"command": "ping", "description": "pong."}])

    def test_clear_default_commands_calls_deletemycommands(self):
        adapter = self._adapter()
        out = adapter.clear_default_commands()
        self.assertTrue(out["ok"], out)
        session = adapter._session
        calls = [(u, p) for u, p in session.calls if "deleteMyCommands" in u]
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0][1]["scope"]), {"type": "default"})


class StartupSyncTests(unittest.TestCase):
    def _adapter(self, tmp, **kw):
        from nomorals.social.chat.telegram import TelegramBotAdapter

        kw.setdefault("token", "123456:test-token")
        kw["session"] = _FakeSession()
        kw.setdefault("lock_path", tmp + "/telegram-bot.lock")
        return TelegramBotAdapter(**kw)

    def test_run_locked_syncs_once_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp)
            adapter._stopped.set()  # poll loop exits immediately
            adapter._run_locked(lambda *a, **k: None)
        session = adapter._session
        sync_calls = [(u, p) for u, p in session.calls
                      if "setMyCommands" in u]
        self.assertEqual(len(sync_calls), 1)
        self.assertEqual(json.loads(sync_calls[0][1]["scope"]),
                         {"type": "default"})
        pushed = json.loads(sync_calls[0][1]["commands"])
        self.assertEqual(len(pushed), 100)
        self.assertEqual(pushed[0]["command"], "help")

    def test_startup_sync_failure_does_not_kill_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp)
            adapter._stopped.set()

            orig_post = adapter._session.post_json

            def _failing(url, params, timeout=None):
                if "setMyCommands" in url:
                    raise RuntimeError("network down")
                return orig_post(url, params, timeout)

            adapter._session.post_json = _failing
            adapter._run_locked(lambda *a, **k: None)  # must not raise


class CliMenuSyncTests(unittest.TestCase):
    def _args(self, **kw):
        args = argparse.Namespace(
            task=["sync"], bot="both", dry_run=True, json=False)
        for key, value in kw.items():
            setattr(args, key, value)
        return args

    def _context(self, settings):
        ctx = mock.Mock()
        ctx.settings = settings
        return ctx

    def _settings(self, gated_token="111:aaa", personal_token="222:bbb"):
        chat = mock.Mock()
        chat.telegram_bot_token = gated_token
        chat.telegram_personal_bot_token = personal_token
        chat.telegram_bot_enabled = True
        chat.telegram_personal_bot_enabled = True
        settings = mock.Mock()
        settings.chat = chat
        return settings

    def test_cli_dry_run_pushes_nothing(self):
        from nomorals.cmdline.commands.menu import _cmd_menu

        def _fake_build(settings, name, **kw):
            return _FakeAdapter(owner_only=(name == "telegram-personal"),
                                name=name)

        with mock.patch("nomorals.social.chat.build_adapter",
                        side_effect=_fake_build):
            rc = _cmd_menu(self._args(), self._context(self._settings()))
        self.assertEqual(rc, 0)

    def test_cli_live_sync_pushes_both_bots(self):
        from nomorals.cmdline.commands.menu import _cmd_menu

        seen: list = []

        class _Recording(_FakeAdapter):
            def set_default_commands(self, commands):
                seen.append(self.name)
                return super().set_default_commands(commands)

        def _fake_build(settings, name, **kw):
            return _Recording(owner_only=(name == "telegram-personal"),
                              name=name)

        with mock.patch("nomorals.social.chat.build_adapter",
                        side_effect=_fake_build):
            rc = _cmd_menu(self._args(dry_run=False),
                           self._context(self._settings()))
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(seen), ["telegram-bot", "telegram-personal"])

    def test_cli_missing_bot_refuses_honestly(self):
        from nomorals.cmdline.commands.menu import _cmd_menu

        def _fake_build(settings, name, **kw):
            return None

        with mock.patch("nomorals.social.chat.build_adapter",
                        side_effect=_fake_build):
            rc = _cmd_menu(self._args(dry_run=False),
                           self._context(self._settings()))
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
