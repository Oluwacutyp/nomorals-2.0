"""Gapfix R3-64: TUI command dispatch goes through TUIFacade.

The facade (nomorals/console/tui_facade.py) unifies the GodConsole and
the ``nm`` cmdline, but as shipped its cmdline half can never fire:
``TUIFacade`` takes no command table, ``cmdline_commands()`` with no
argument returns ``[]``, and both ``list_commands()`` and
``run_command()`` call it with no argument. These tests pin the wired
shape:

* a known cmdline command dispatched through the facade reaches the
  injected cmdline dispatch, not the console fallback;
* ``list_commands()`` is the union of both surfaces' commands;
* unknown commands fall through to the console handler;
* a leading ``nm`` prefix is stripped (the owner types ``nm status``);
* ``cmdline/dispatch.py::run_command_string`` turns one ``nm ...``
  line into the text it prints (for the GodConsole to render), and
  refuses terminal-taking commands instead of hanging the console.
"""

import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

from nomorals.console.tui_facade import TUIFacade


class _StubConsole:
    """GodConsole-shaped stub: _on_command handles everything."""

    def __init__(self):
        self.seen = []

    def _on_command(self, cmd):
        self.seen.append(cmd)
        return f"handled:{cmd}"


def _facade(cmdline_names=("doctor", "analytics")):
    seen_cmdline = []
    seen_console = []
    console = _StubConsole()

    def cmdline_dispatch(cmd):
        seen_cmdline.append(cmd)
        return f"cmdline:{cmd}"

    def console_dispatch(cmd):
        seen_console.append(cmd)
        return f"console:{cmd}"

    facade = TUIFacade(
        godconsole=console,
        cmdline_dispatch=cmdline_dispatch,
        cmdline_commands=list(cmdline_names),
        console_dispatch=console_dispatch,
    )
    return facade, console, seen_cmdline, seen_console


class FacadeRoutingTests(unittest.TestCase):
    def test_known_cmdline_command_goes_to_dispatch(self):
        facade, console, seen_cmdline, seen_console = _facade()
        self.assertEqual(facade.run_command("doctor"), "cmdline:doctor")
        self.assertEqual(seen_cmdline, ["doctor"])
        self.assertEqual(seen_console, [])
        self.assertEqual(console.seen, [])

    def test_list_commands_is_union_of_both_surfaces(self):
        facade, _console, _a, _b = _facade()
        names = facade.list_commands()
        self.assertIn("doctor", names)  # cmdline-only name
        self.assertIn("analytics", names)  # cmdline-only name
        self.assertIn("help", names)  # console native
        self.assertIn("status", names)  # console native

    def test_unknown_command_falls_through_to_console_dispatch(self):
        facade, console, seen_cmdline, seen_console = _facade()
        self.assertEqual(facade.run_command("dashboard"), "console:dashboard")
        self.assertEqual(seen_console, ["dashboard"])
        self.assertEqual(seen_cmdline, [])
        self.assertEqual(console.seen, [])

    def test_nm_prefix_stripped_before_routing(self):
        facade, _console, seen_cmdline, seen_console = _facade()
        self.assertEqual(facade.run_command("nm doctor"), "cmdline:doctor")
        self.assertEqual(seen_cmdline, ["doctor"])
        self.assertEqual(seen_console, [])

    def test_console_dispatch_wins_over_dynamic_lookup(self):
        # The facade's dynamic fallback (godconsole._on_command) must not
        # fire when an explicit console dispatch was injected -- otherwise
        # wiring the facade in as _on_command would recurse.
        facade, console, _a, seen_console = _facade()
        self.assertEqual(facade.run_command("dashboard"), "console:dashboard")
        self.assertEqual(seen_console, ["dashboard"])
        self.assertEqual(console.seen, [])

    def test_default_construction_unchanged(self):
        # The pre-existing contract (test_skiplist_build): no injection,
        # dynamic lookup, console natives only.
        console = _StubConsole()
        facade = TUIFacade(godconsole=console)
        self.assertEqual(facade.run_command("something-unknown"),
                         "handled:something-unknown")
        self.assertIn("status", facade.list_commands())
        self.assertNotIn("doctor", facade.list_commands())


class AdapterTests(unittest.TestCase):
    def _run(self, text):
        import nomorals.core.config as cfg
        from nomorals.cmdline import dispatch as d

        with mock.patch.object(cfg, "load_settings") as mock_load, \
                mock.patch.object(d, "_configure_log_file"), \
                redirect_stderr(io.StringIO()):
            mock_load.return_value = mock.Mock()
            return d.run_command_string(text)

    def test_adapter_runs_command_and_returns_its_output(self):
        # completion is a real no-context command whose handler ignores
        # settings entirely, so this runs the real parser + real dispatch.
        text = self._run("completion bash")
        self.assertIn("complete", text.lower())
        self.assertNotIn("exit code", text)

    def test_adapter_unknown_command_is_honest_refusal(self):
        # The real parser rejects an unknown name before _dispatch runs,
        # so the adapter returns argparse's invalid-choice text (the same
        # text a real terminal shows for `nm nosuchcmd123`) + exit code.
        text = self._run("nosuchcmd123")
        self.assertIn("invalid choice: 'nosuchcmd123'", text)
        self.assertIn("(exit code: 2)", text)

    def test_adapter_refuses_terminal_taking_commands(self):
        for cmd in ("tui", "serve", "run", "stream", "stream serve",
                    "update"):
            text = self._run(cmd)
            self.assertIn("cannot be run inside the console", text, cmd)
            self.assertIn(cmd.split()[0], text, cmd)

    def test_adapter_allows_safe_siblings(self):
        # stream status is one-shot; update --check-only never updates.
        from nomorals.cmdline.dispatch import _terminal_refusal

        self.assertIsNone(_terminal_refusal(["stream", "status"]))
        self.assertIsNone(_terminal_refusal(["update", "--check-only"]))
        self.assertIsNone(_terminal_refusal(["chat", "doctor"]))
        self.assertIsNone(_terminal_refusal(["watch", "list"]))

    def test_terminal_refusal_names_the_blocked_commands(self):
        from nomorals.cmdline.dispatch import _terminal_refusal

        for argv in (["tui"], ["serve"], ["run"], ["stream"],
                     ["stream", "serve"], ["update"]):
            refusal = _terminal_refusal(argv)
            self.assertIsNotNone(refusal, argv)
            self.assertIn(argv[0], refusal)

    def test_adapter_never_raises_on_garbage(self):
        text = self._run("!!! ???")
        self.assertIsInstance(text, str)
        self.assertTrue(text.strip())


class GodConsoleWiringTests(unittest.TestCase):
    def test_god_console_wired_through_facade(self):
        from nomorals.console.commands import ConsoleCommands

        seen = []

        def cmdline_dispatch(cmd):
            seen.append(cmd)
            return f"cmdline:{cmd}"

        commands = ConsoleCommands(
            lambda: {},
            cmdline_dispatch=cmdline_dispatch,
            cmdline_commands=["doctor"],
        )
        console = commands._build_god_console()
        # The input handler IS the facade.
        self.assertIsInstance(console._on_command.__self__, TUIFacade)
        self.assertEqual(console._on_command("doctor"), "cmdline:doctor")
        self.assertEqual(seen, ["doctor"])
        # A console-native name still reaches the console handler.
        result = console._on_command("dashboard")
        self.assertIsInstance(result, str)
        self.assertNotIn("cmdline:", result)

    def test_god_console_without_dispatch_unchanged(self):
        from nomorals.console.commands import ConsoleCommands

        commands = ConsoleCommands(lambda: {})
        console = commands._build_god_console()
        # No facade: the handler is the plain console handler.
        self.assertNotIsInstance(
            getattr(console._on_command, "__self__", None), TUIFacade)
        self.assertEqual(console._on_command("dashboard"),
                         commands._handle_console_command("dashboard"))


if __name__ == "__main__":
    unittest.main()
