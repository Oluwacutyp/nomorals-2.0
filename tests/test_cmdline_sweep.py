"""Sweep tests for the ``nomorals.cmdline`` upgrade.

Covers the new work only (existing CLI tests live in test_cli_*.py):

* dispatch registry: every parser command is registered exactly once, every
  handler is callable, settings-only commands don't take a context.
* ``register_command()`` plugin API: registers, rejects duplicates and junk.
* did-you-mean suggestions on unknown commands.
* ``nm completion``: all four shells generate, scripts contain every command
  and alias, the bash script is syntactically valid and functionally
  completes (top level, subcommands, nested options, aliases).
* style layer: plain output when piped/NO_COLOR/--no-color, styled on a TTY,
  table rendering, semantic helpers, spinner silence off-TTY.
* ``nm help cli`` overview stays stable when captured (non-TTY).
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from nomorals.cmdline import (
    CLI_ALIASES,
    _COMMANDS,
    _canonical_command,
    _cli_overview,
    _cmd_completion,
    _parser,
    _suggest_command,
    register_command,
)
from nomorals.cmdline import style
from nomorals.cmdline.commands.completion import _collect
from nomorals.cmdline.dispatch import _CommandSpec


def _parser_commands() -> dict[str, argparse.ArgumentParser]:
    """canonical name → subparser, deduped (aliases share the parser object)."""
    from nomorals.cmdline.commands.meta import _cli_subparsers

    sub = _cli_subparsers()
    assert sub is not None
    out: dict[str, argparse.ArgumentParser] = {}
    for name, target in sub.choices.items():
        out.setdefault(target.prog.split()[-1], target)
    return out


class DispatchRegistryTests(unittest.TestCase):
    def test_table_covers_every_parser_command_exactly(self):
        parser_cmds = set(_parser_commands())
        table_cmds = set(_COMMANDS)
        self.assertEqual(parser_cmds, table_cmds,
                         f"missing: {sorted(parser_cmds - table_cmds)}, "
                         f"extra: {sorted(table_cmds - parser_cmds)}")

    def test_every_handler_is_callable(self):
        bad = [k for k, spec in _COMMANDS.items()
               if not callable(spec.handler)]
        self.assertEqual(bad, [])

    def test_settings_only_commands(self):
        # These manage live state and must NOT hold the DB open.
        for name in ("doctor", "snapshot", "recover", "update", "config",
                     "completion"):
            self.assertFalse(_COMMANDS[name].needs_context, name)
        # Spot-check: a normal command does take a context.
        self.assertTrue(_COMMANDS["status"].needs_context)
        self.assertTrue(_COMMANDS["memory"].needs_context)

    def test_models_and_skill_keep_their_branches(self):
        # The two commands with internal dispatch still resolve via the table.
        self.assertIn("models", _COMMANDS)
        self.assertIn("skill", _COMMANDS)

    def test_canonical_command_still_resolves_aliases(self):
        self.assertEqual(_canonical_command("st"), "status")
        self.assertEqual(_canonical_command("status"), "status")
        self.assertEqual(_canonical_command("complete"), "completion")
        self.assertEqual(_canonical_command("nope"), "nope")


class RegisterCommandTests(unittest.TestCase):
    def test_register_and_unregister(self):
        def _dummy(args, context):
            return 0

        register_command("_sweep_dummy", _dummy, needs_context=False)
        try:
            self.assertIn("_sweep_dummy", _COMMANDS)
            spec = _COMMANDS["_sweep_dummy"]
            self.assertIsInstance(spec, _CommandSpec)
            self.assertFalse(spec.needs_context)
            self.assertEqual(spec.handler(args=None, context=None), 0)
        finally:
            del _COMMANDS["_sweep_dummy"]
        self.assertNotIn("_sweep_dummy", _COMMANDS)

    def test_register_rejects_duplicates(self):
        def _dummy(args, context):
            return 0

        with self.assertRaises(ValueError):
            register_command("status", _dummy)

    def test_register_rejects_junk(self):
        with self.assertRaises(ValueError):
            register_command("", lambda a, c: 0)
        with self.assertRaises(ValueError):
            register_command("_sweep_junk", "not-callable")  # type: ignore[arg-type]

    def test_register_works_as_decorator(self):
        @register_command("_sweep_deco")
        def _dummy(args, context):
            return 0

        try:
            self.assertIn("_sweep_deco", _COMMANDS)
        finally:
            del _COMMANDS["_sweep_deco"]


class SuggestCommandTests(unittest.TestCase):
    def test_typo_suggests_canonical(self):
        hits = _suggest_command("stauts")
        self.assertIn("status", hits)

    def test_prefix_suggests(self):
        hits = _suggest_command("memor")
        self.assertIn("memory", hits)

    def test_alias_can_be_suggested(self):
        hits = _suggest_command("complet")
        self.assertTrue("completion" in hits or "complete" in hits)

    def test_gibberish_suggests_nothing(self):
        self.assertEqual(_suggest_command("zzzzzz"), [])

    def test_no_duplicate_suggestions(self):
        hits = _suggest_command("stauts")
        self.assertEqual(len(hits), len(set(hits)))

    def test_unknown_command_message(self):
        # End-to-end through _dispatch with stubbed settings/config.
        # load_settings is imported inside _dispatch, so patch its source.
        from nomorals.cmdline import dispatch as d
        import nomorals.core.config as cfg

        args = argparse.Namespace(command="stauts", config=None,
                                  log_level="CRITICAL")
        with mock.patch.object(cfg, "load_settings") as mock_load, \
                mock.patch.object(d, "_configure_log_file"), \
                redirect_stderr(io.StringIO()) as err:
            mock_load.return_value = mock.Mock()
            rc = d._dispatch(args)
        self.assertEqual(rc, 2)
        text = err.getvalue()
        self.assertIn("unknown command: stauts", text)
        self.assertIn("did you mean", text)
        self.assertIn("status", text)
        self.assertIn("nm help cli", text)


class CompletionTests(unittest.TestCase):
    def test_collect_covers_all_commands(self):
        spec = _collect()
        self.assertEqual(set(spec["commands"]), set(_COMMANDS))

    def test_collect_finds_aliases_and_choices(self):
        spec = _collect()["commands"]
        self.assertIn("st", spec["status"]["aliases"])
        # autonomy's action positional has choices
        self.assertIn("status", spec["autonomy"]["pos"][0])
        # memory has nested subcommands with their own options
        self.assertIn("list", spec["memory"]["subs"])
        self.assertIn("--kind", spec["memory"]["subs"]["list"]["opts"])

    def test_all_shells_generate(self):
        from nomorals.cmdline.commands.completion import _GENERATORS

        spec = _collect()
        for shell, gen in _GENERATORS.items():
            script = gen(spec)
            self.assertTrue(script.strip(), shell)
            # every canonical command name appears in every script
            for name in _COMMANDS:
                self.assertIn(name, script, f"{shell} missing {name}")

    def test_bash_script_is_valid_and_functional(self):
        if not shutil_which("bash"):
            self.skipTest("bash not available")
        with tempfile.NamedTemporaryFile("w", suffix=".bash",
                                         delete=False) as fh:
            args = argparse.Namespace(shell="bash")
            with redirect_stdout(fh):
                rc = _cmd_completion(args)
            path = fh.name
        self.assertEqual(rc, 0)
        try:
            cp = subprocess.run(["bash", "-n", path], capture_output=True,
                                text=True, timeout=30)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            # functional: source it and drive _nm through fake COMP_WORDS
            probe = (
                f"source {path}\n"
                'COMP_WORDS=(nm ""); COMP_CWORD=1; _nm; echo "TOP:${COMPREPLY[*]}"\n'
                'COMP_WORDS=(nm mem); COMP_CWORD=1; _nm; echo "PRE:${COMPREPLY[*]}"\n'
                'COMP_WORDS=(nm memory ""); COMP_CWORD=2; _nm; echo "SUB:${COMPREPLY[*]}"\n'
                'COMP_WORDS=(nm st ""); COMP_CWORD=2; _nm; echo "ALIAS:${COMPREPLY[*]}"\n'
            )
            cp = subprocess.run(["bash", "-c", probe], capture_output=True,
                                text=True, timeout=30)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            out = cp.stdout
            top = _line(out, "TOP:")
            self.assertIn("status", top.split())
            self.assertIn("st", top.split())  # aliases complete too
            pre = _line(out, "PRE:")
            self.assertIn("memory", pre.split())
            sub = _line(out, "SUB:")
            self.assertIn("list", sub.split())
            self.assertIn("--query", sub.split())
            alias = _line(out, "ALIAS:")
            self.assertIn("--json", alias.split())  # st → status opts
        finally:
            Path(path).unlink(missing_ok=True)

    def test_completion_handler_rejects_bad_shell(self):
        args = argparse.Namespace(shell="tcsh-9")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(_cmd_completion(args), 2)

    def test_completion_output_is_byte_exact_script(self):
        # No styling may leak into the script — it gets redirected to a file.
        args = argparse.Namespace(shell="bash")
        buf = io.StringIO()
        with redirect_stdout(buf):
            _cmd_completion(args)
        script = buf.getvalue()
        self.assertNotIn("\033[", script)
        self.assertTrue(script.startswith("# nm completion for bash."))
        self.assertIn("complete -F _nm nm", script)


def _line(out: str, prefix: str) -> str:
    for line in out.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    return ""


def shutil_which(name: str):
    import shutil

    return shutil.which(name)


class StyleTests(unittest.TestCase):
    def setUp(self):
        self._old_force = style._force_no_color
        self._old_env = {k: os.environ.get(k) for k in ("NO_COLOR", "NM_NO_COLOR", "TERM")}

    def tearDown(self):
        style._force_no_color = self._old_force
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_plain_when_piped(self):
        # redirect_stdout gives a non-TTY StringIO → no ANSI, ever.
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertFalse(style.color_enabled())
            self.assertEqual(style.ok("done"), "[ok] done")
            self.assertEqual(style.err("boom"), "[!!] boom")
            self.assertEqual(style.stylize("x", "bold red"), "x")

    def test_no_color_env_kills_styling(self):
        os.environ["NO_COLOR"] = "1"
        with mock.patch.object(sys.stdout, "isatty", return_value=True):
            self.assertFalse(style.color_enabled())
            self.assertEqual(style.warn("careful"), "[!] careful")

    def test_nm_no_color_env_kills_styling(self):
        os.environ["NM_NO_COLOR"] = "yes"
        with mock.patch.object(sys.stdout, "isatty", return_value=True):
            self.assertFalse(style.color_enabled())

    def test_force_flag(self):
        style.set_no_color(True)
        with mock.patch.object(sys.stdout, "isatty", return_value=True):
            self.assertFalse(style.color_enabled())
            self.assertEqual(style.title("T"), "T")
        style.set_no_color(False)

    def test_styled_on_tty(self):
        with mock.patch.object(sys.stdout, "isatty", return_value=True):
            self.assertTrue(style.color_enabled())
            out = style.ok("done")
            self.assertIn("\033[", out)
            self.assertIn("✓", out)
            self.assertIn("done", out)

    def test_table_renders_plain_when_captured(self):
        t = style.Table(["command", "aliases"], [("status", "st"), ("memory", "mem")])
        rendered = t.render()
        self.assertNotIn("\033[", rendered)
        self.assertIn("status", rendered)
        self.assertIn("st", rendered)
        # columns align
        lines = rendered.splitlines()
        self.assertEqual(len(lines), 3)  # header + 2 rows
        self.assertTrue(lines[1].startswith("status"))

    def test_table_styled_header_on_tty(self):
        t = style.Table(["a"], [["b"]])
        with mock.patch.object(sys.stdout, "isatty", return_value=True):
            self.assertIn("\033[", t.render())

    def test_table_truncates_long_cells(self):
        t = style.Table(["name", "desc"], [("x", "y" * 500)], max_width=40)
        for line in t.render().splitlines():
            self.assertLessEqual(len(line), 41)  # width + ellipsis

    def test_kv_panel(self):
        out = style.kv({"version": "1.2.3", "profile": "dev"})
        self.assertIn("version", out)
        self.assertIn("1.2.3", out)
        self.assertNotIn("\033[", out)  # captured → plain

    def test_icons_have_ascii_fallbacks(self):
        self.assertEqual(style.icon("ok"), "[ok]")
        self.assertEqual(style.icon("arrow"), "->")

    def test_spinner_silent_off_tty(self):
        buf = io.StringIO()
        with mock.patch.object(sys, "stderr", buf):
            with style.spinner("working"):
                pass
            self.assertEqual(buf.getvalue(), "")


class OverviewTests(unittest.TestCase):
    def test_overview_lists_every_command(self):
        page = _cli_overview()
        for name in _COMMANDS:
            self.assertIn(name, page)

    def test_overview_mentions_completion(self):
        self.assertIn("completion", _cli_overview())

    def test_overview_stable_when_captured(self):
        # Non-TTY → the classic flat format, no ANSI.
        page = _cli_overview()
        self.assertNotIn("\033[", page)
        self.assertIn("nm help <command>", page)

    def test_overview_table_on_tty(self):
        with mock.patch.object(sys.stdout, "isatty", return_value=True):
            page = _cli_overview()
            self.assertIn("\033[", page)
            self.assertIn("completion", page)


class ParserSurfaceTests(unittest.TestCase):
    def test_no_color_flag_parses(self):
        args = _parser().parse_args(["--no-color", "status"])
        self.assertTrue(args.no_color)
        self.assertEqual(args.command, "status")

    def test_completion_parses(self):
        args = _parser().parse_args(["completion", "zsh"])
        self.assertEqual(args.command, "completion")
        self.assertEqual(args.shell, "zsh")
        args = _parser().parse_args(["complete"])  # alias
        self.assertEqual(args.command, "complete")
        self.assertEqual(_canonical_command(args.command), "completion")

    def test_completion_rejects_bad_shell(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                _parser().parse_args(["completion", "tcsh"])
        self.assertEqual(cm.exception.code, 2)

    def test_help_epilog_documents_exit_codes(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as cm:
                _parser().parse_args(["--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("exit codes", buf.getvalue())

    def test_main_applies_no_color(self):
        with mock.patch("nomorals.cmdline.dispatch._parser") as mock_parser, \
                mock.patch("nomorals.cmdline.dispatch.setup_logging"), \
                mock.patch("nomorals.cmdline.dispatch._dispatch") as mock_dispatch:
            ns = argparse.Namespace(no_color=True, log_level="INFO")
            mock_parser.return_value.parse_args.return_value = ns
            mock_dispatch.return_value = 0
            from nomorals.cmdline.dispatch import main

            # reset the flag first so the test is hermetic
            style.set_no_color(False)
            main([])
            self.assertFalse(style.color_enabled())
            style.set_no_color(False)


if __name__ == "__main__":
    unittest.main()
