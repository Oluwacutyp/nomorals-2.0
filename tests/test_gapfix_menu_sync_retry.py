"""Menu sync TOO_MUCH retry: the live failure at EC2 boot 2026-11-17 06:56.

Both bots' startup syncs failed with HTTP 400 BOT_COMMANDS_TOO_MUCH
while pushing exactly 100 valid commands each. The payload is provably
correct (100 unique, charset-clean, description-clean entries, the same
transport shape as every working Bot API call in the adapter), so the
refusal is server-side counting — and the sync must LAND instead of
failing outright: on TOO_MUCH, retry with a smaller priority-head list
until Telegram accepts. Failing-first: every test here fails on the
pre-fix code (no retry, no charset rule, uncapped scope lists).
"""

from __future__ import annotations

import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

import nomorals.social.chat.personal_commands as pc
from nomorals.social.chat.personal_commands import (
    BOTFATHER_MAX_COMMANDS,
    PERSONAL_OWNER_COMMANDS,
    _public_priority,
    gated_botfather_commands,
    personal_botfather_commands,
    sync_bot_commands,
)


class _TooMuchAdapter:
    """Duck-typed adapter: setMyCommands fails TOO_MUCH above `ceiling`."""

    def __init__(self, *, owner_only=False, name="telegram-bot", ceiling=50):
        self.owner_only = owner_only
        self.name = name
        self.ceiling = ceiling
        self.pushed: list[list[tuple[str, str]]] = []

    def set_default_commands(self, commands):
        self.pushed.append(list(commands))
        if len(commands) > self.ceiling:
            return {
                "ok": False,
                "error": ("telegram bot api: setMyCommands failed: "
                          "Bad Request: BOT_COMMANDS_TOO_MUCH"),
            }
        return {"ok": True, "result": True}


class _RaisingTooMuchAdapter(_TooMuchAdapter):
    """Same refusal, but raised (the _set_command_scope error shape)."""

    def set_default_commands(self, commands):
        self.pushed.append(list(commands))
        if len(commands) > self.ceiling:
            raise RuntimeError(
                "telegram bot api: setMyCommands failed: "
                "Bad Request: BOT_COMMANDS_TOO_MUCH")
        return {"ok": True, "result": True}


class RetryTests(unittest.TestCase):
    def test_gated_retry_lands_50_on_too_much(self):
        adapter = _TooMuchAdapter(owner_only=False, ceiling=50)
        out = sync_bot_commands(adapter)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["pushed"], 50)
        self.assertEqual(out["attempts"], [100, 50])
        self.assertTrue(out["retried"])
        self.assertEqual(len(adapter.pushed), 2)
        self.assertEqual(len(adapter.pushed[0]), 100)
        self.assertEqual(adapter.pushed[1], gated_botfather_commands()[:50])
        self.assertEqual(out["commands"],
                         [k for k, _ in gated_botfather_commands()[:50]])
        self.assertEqual(out["skipped"], len(_public_priority()) - 50)

    def test_personal_retry_keeps_every_owner_command(self):
        adapter = _TooMuchAdapter(owner_only=True,
                                  name="telegram-personal", ceiling=50)
        out = sync_bot_commands(adapter)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["pushed"], 50)
        self.assertEqual(out["attempts"], [100, 50])
        kinds = out["commands"]
        # Retry priority head: all 48 owner commands first (they are
        # non-negotiable), then the public fill in priority order.
        self.assertEqual(kinds[:48], list(PERSONAL_OWNER_COMMANDS))
        self.assertEqual(kinds[48:], _public_priority()[:2])
        self.assertEqual(adapter.pushed[1],
                         [(k, d) for k, d in
                          personal_botfather_commands()
                          if k in PERSONAL_OWNER_COMMANDS][:48]
                         + [(k, d) for k, d in
                            gated_botfather_commands()][:2])
        self.assertEqual(out["skipped"], (len(_public_priority()) - 2)
                         + (len(PERSONAL_OWNER_COMMANDS) - 48))

    def test_raised_too_much_retries_the_same_way(self):
        adapter = _RaisingTooMuchAdapter(owner_only=False, ceiling=50)
        out = sync_bot_commands(adapter)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["pushed"], 50)
        self.assertEqual(out["attempts"], [100, 50])

    def test_non_too_much_failure_does_not_retry(self):
        adapter = _TooMuchAdapter(owner_only=False, ceiling=50)

        def _blocked(commands):
            adapter.pushed.append(list(commands))
            return {"ok": False,
                    "error": "telegram bot api: setMyCommands failed: "
                             "Bad Request: bot was blocked by the user"}

        adapter.set_default_commands = _blocked
        out = sync_bot_commands(adapter)
        self.assertFalse(out["ok"])
        self.assertEqual(len(adapter.pushed), 1)
        self.assertTrue(any("blocked" in p for p in out["problems"]))

    def test_floor_refusal_names_attempts(self):
        adapter = _TooMuchAdapter(owner_only=False, ceiling=10)
        out = sync_bot_commands(adapter)
        self.assertFalse(out["ok"])
        self.assertEqual(out["attempts"], [100, 50, 25])
        self.assertTrue(any("BOT_COMMANDS_TOO_MUCH" in p
                            for p in out["problems"]))
        self.assertEqual(out["pushed"], 0)


class ValidatorCharsetTests(unittest.TestCase):
    def test_validator_catches_hyphenated_name(self):
        bad = ([("bad-name", "a bad name."), ]
               + personal_botfather_commands()[1:])
        with mock.patch.object(pc, "personal_botfather_commands",
                               return_value=bad):
            problems = pc.validate_botfather_commands()
        self.assertTrue(any("bad-name" in p and "must match" in p
                            for p in problems))

    def test_validator_catches_33_char_name(self):
        long_name = "a" * 33
        bad = ([(long_name, "too long."), ]
               + gated_botfather_commands()[1:])
        with mock.patch.object(pc, "gated_botfather_commands",
                               return_value=bad):
            problems = pc.validate_gated_botfather_commands()
        self.assertTrue(any(long_name in p and "must match" in p
                            for p in problems))

    def test_sync_refuses_a_bad_name_without_an_api_round_trip(self):
        bad = ([("bad-name", "a bad name."), ]
               + personal_botfather_commands()[1:])
        adapter = _TooMuchAdapter(owner_only=True,
                                  name="telegram-personal")
        with mock.patch.object(pc, "personal_botfather_commands",
                               return_value=bad):
            out = sync_bot_commands(adapter)
        self.assertFalse(out["ok"])
        self.assertTrue(any("bad-name" in p for p in out["problems"]))
        self.assertEqual(adapter.pushed, [])


class ScopeCapTests(unittest.TestCase):
    def test_scope_commands_capped_at_100(self):
        from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

        kinds = tuple(f"kind{i:03d}" for i in range(150))
        with mock.patch.object(RuntimeGroupMixin, "_P5_SCOPE_KINDS",
                               kinds):
            obj = object.__new__(RuntimeGroupMixin)
            cmds = obj._p5_scope_commands()
        self.assertEqual(len(cmds), BOTFATHER_MAX_COMMANDS)
        self.assertEqual([k for k, _ in cmds], list(kinds[:100]))

    def test_set_command_scope_truncates_over_100(self):
        from nomorals.social.chat.telegram import TelegramBotAdapter

        class _FakeResp:
            def __init__(self, payload=None):
                self._payload = payload or {"ok": True, "result": True}

            def json(self):
                return self._payload

        class _FakeSession:
            def __init__(self):
                self.calls: list[tuple[str, dict]] = []

            def post_json(self, url, params, timeout=None):
                self.calls.append((url, dict(params)))
                return _FakeResp({"ok": True, "result": True})

        session = _FakeSession()
        adapter = TelegramBotAdapter(token="123456:test-token",
                                     session=session)
        entries = [(f"cmd{i:03d}", "desc.") for i in range(150)]
        out = adapter.set_default_commands(entries)
        self.assertTrue(out["ok"], out)
        calls = [(u, p) for u, p in session.calls if "setMyCommands" in u]
        self.assertEqual(len(calls), 1)
        sent = json.loads(calls[0][1]["commands"])
        self.assertEqual(len(sent), BOTFATHER_MAX_COMMANDS)
        self.assertEqual(sent[0]["command"], "cmd000")
        self.assertEqual(sent[-1]["command"], "cmd099")


class ScopeStateTests(unittest.TestCase):
    def _recording(self, live_count=100, fail=False):
        class _Adapter:
            name = "telegram-bot"
            owner_only = False
            live_calls = 0

            def get_default_commands(self):
                type(self).live_calls += 1
                if fail:
                    raise RuntimeError("network down")
                return [{"command": f"c{i:03d}", "description": "d."}
                        for i in range(live_count)]

        return _Adapter()

    def test_scope_state_reports_live_default_count(self):
        adapter = self._recording(live_count=100)
        state = pc.menu_scope_state(adapter, scope_size=34)
        self.assertEqual(state["live"]["default"], 100)
        self.assertEqual(state["builders"]["gated_default"], 100)
        self.assertEqual(state["builders"]["personal_default"], 100)
        self.assertEqual(state["builders"]["scope"], 34)

    def test_scope_state_never_raises_when_api_unreachable(self):
        adapter = self._recording(fail=True)
        state = pc.menu_scope_state(adapter, scope_size=34)
        self.assertIsNone(state["live"]["default"])
        self.assertIn("unreachable", state["live"]["note"])
        self.assertEqual(state["builders"]["gated_default"], 100)

    def test_scope_state_dry_run_never_calls_api(self):
        adapter = self._recording()
        state = pc.menu_scope_state(adapter, dry_run=True, scope_size=34)
        self.assertEqual(type(adapter).live_calls, 0)
        self.assertIsNone(state["live"]["default"])
        self.assertIn("dry run", state["live"]["note"])

    def test_scope_state_tolerates_adapter_without_live_reader(self):
        class _Bare:
            name = "telegram-bot"
            owner_only = False

        state = pc.menu_scope_state(_Bare(), scope_size=34)
        self.assertIsNone(state["live"]["default"])
        self.assertEqual(state["builders"]["gated_default"], 100)

    def test_scope_state_without_scope_size_reports_it_unknown(self):
        adapter = self._recording(live_count=100)
        state = pc.menu_scope_state(adapter)
        self.assertIsNone(state["builders"]["scope"])
        self.assertEqual(state["live"]["default"], 100)


class CliDiagnosticTests(unittest.TestCase):
    def _args(self, **kw):
        args = argparse.Namespace(
            task=["sync"], bot="both", dry_run=False, json=False)
        for key, value in kw.items():
            setattr(args, key, value)
        return args

    def _context(self):
        chat = mock.Mock()
        chat.telegram_bot_token = "1:test"
        chat.telegram_personal_bot_token = "2:test"
        chat.telegram_bot_enabled = True
        chat.telegram_personal_bot_enabled = True
        settings = mock.Mock()
        settings.chat = chat
        ctx = mock.Mock()
        ctx.settings = settings
        return ctx

    def test_cli_prints_menu_state_after_sync(self):
        from nomorals.cmdline.commands.menu import _cmd_menu

        class _Fake:
            def __init__(self, owner_only=False, name="telegram-bot"):
                self.owner_only = owner_only
                self.name = name
                self.pushed = []

            def set_default_commands(self, commands):
                self.pushed.append(list(commands))
                return {"ok": True, "result": True}

            def get_default_commands(self):
                return [{"command": "help", "description": "x."}] * 100

        def _fake_build(settings, name, **kw):
            return _Fake(owner_only=(name == "telegram-personal"),
                         name=name)

        buf = io.StringIO()
        with mock.patch("nomorals.social.chat.build_adapter",
                        side_effect=_fake_build), redirect_stdout(buf):
            rc = _cmd_menu(self._args(), self._context())
        self.assertEqual(rc, 0)
        self.assertIn("menu state", buf.getvalue())
        self.assertIn("live default scope: 100", buf.getvalue())

    def test_cli_dry_run_prints_state_without_live_query(self):
        from nomorals.cmdline.commands.menu import _cmd_menu

        class _Fake:
            def __init__(self, owner_only=False, name="telegram-bot"):
                self.owner_only = owner_only
                self.name = name
                self.live_calls = 0

            def set_default_commands(self, commands):
                return {"ok": True, "result": True}

            def get_default_commands(self):
                self.live_calls += 1
                return []

        made = {}

        def _fake_build(settings, name, **kw):
            made[name] = _Fake(owner_only=(name == "telegram-personal"),
                               name=name)
            return made[name]

        buf = io.StringIO()
        with mock.patch("nomorals.social.chat.build_adapter",
                        side_effect=_fake_build), redirect_stdout(buf):
            rc = _cmd_menu(self._args(dry_run=True), self._context())
        self.assertEqual(rc, 0)
        self.assertEqual(sum(f.live_calls for f in made.values()), 0)
        self.assertIn("menu state", buf.getvalue())
        self.assertIn("dry run", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
