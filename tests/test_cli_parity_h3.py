"""Wave H3/cli — façade parity for the ``nomorals.cli`` → ``nomorals.cmdline`` split.

``nomorals/cli.py`` is now a compatibility façade over the new
``nomorals.cmdline`` package.  This pins the contract:

* every name the old monolith exported is still importable from
  ``nomorals.cli``,
* each is the *identical* object (``is``, not a copy) as the one in the
  new package — so ``mock.patch``-style attribute access, ``__main__``,
  and the ``nm``/``nmctl`` entry points keep working,
* the parser still builds and ``main(["--help"])`` exits 0.
"""

from __future__ import annotations

import importlib
import unittest


# Names external importers actually use (grep over tests/, __main__,
# vendor/).  A subset of the full surface, pinned explicitly so a future
# refactor cannot silently drop one.
IMPORTER_USED = [
    "CLI_ALIASES",
    "main",
    "_parser",
    "_dispatch",
    "_emit",
    "_canonical_command",
    "_attach_cli_session",
    "_configure_log_file",
    "_render_config",
    "_cli_overview",
    "_cli_subparsers",
    "_cli_command_help",
    "_cmd_data",
    "_cmd_native",
    "_cmd_timeline",
    "_cmd_doctor",
    "_cmd_models",
    "_cmd_models_doctor",
    "_cmd_setup",
    "_cmd_tools",
    "_cmd_owner",
    "_cmd_power",
    "_cmd_memory",
    "_cmd_memory_action",
    "_cmd_code",
    "_cmd_code_run",
    "_cmd_code_review",
    "_cmd_code_test",
    "_cmd_media",
    "_cmd_studio",
    "_cmd_vision",
    "_cmd_captcha",
    "_cmd_weather",
    "_cmd_bet",
    "_cmd_voice",
    "_cmd_improve",
    "_cmd_trade",
    "_cmd_swarm",
    "_cmd_room",
    "_cmd_benchmark",
    "_cmd_briefing",
    "_cmd_inbox",
    "_cmd_run",
    "_cmd_ask",
    "_cmd_backup",
    "_cmd_missions",
    "_cmd_mission",
    "_cmd_tui",
    "_cmd_serve",
    "_cmd_stream",
    "_cmd_queue",
    "_cmd_status",
    "_cmd_mind",
    "_cmd_hub",
    "_cmd_book",
    "_cmd_arena",
    "_cmd_trial",
    "_cmd_skill",
    "_cmd_simulate",
    "_cmd_research_loop",
    "_cmd_kg",
    "_cmd_cookies",
    "_cmd_structure",
    "_cmd_reason",
    "_cmd_workspace",
    "_cmd_partner_ask",
    "_cmd_train",
    "_cmd_crack",
    "_cmd_decode",
    "_cmd_money",
    "_cmd_osint",
    "_cmd_cipher",
    "_cmd_monitor",
    "_cmd_watch",
    "_cmd_commands",
    "_cmd_deliver",
    "_cmd_zip",
    "_cmd_connectors",
    "_cmd_help",
    "_cmd_finance",
    "_cmd_cards",
    "_cmd_autonomy",
    "_cmd_music",
    "_cmd_exec",
    "_cmd_apps",
    "_cmd_project",
    "_cmd_goal",
]


class FacadeParityTests(unittest.TestCase):
    def test_importer_used_names_importable_from_cli(self):
        cli = importlib.import_module("nomorals.cli")
        for name in IMPORTER_USED:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(cli, name),
                    f"nomorals.cli no longer exports {name!r}",
                )

    def test_every_cli_name_is_identical_object_in_cmdline(self):
        cli = importlib.import_module("nomorals.cli")
        cmdline = importlib.import_module("nomorals.cmdline")
        self.assertTrue(hasattr(cli, "__all__"), "façade lost __all__")
        self.assertGreater(len(cli.__all__), 100, "façade __all__ looks truncated")
        for name in cli.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(cmdline, name),
                                f"nomorals.cmdline missing {name!r}")
                self.assertIs(
                    getattr(cli, name), getattr(cmdline, name),
                    f"nomorals.cli.{name} is not the same object as "
                    f"nomorals.cmdline.{name}",
                )

    def test_no_name_lost_from_all(self):
        cli = importlib.import_module("nomorals.cli")
        for name in IMPORTER_USED:
            self.assertIn(name, cli.__all__)


class ParserSmokeTests(unittest.TestCase):
    def test_parser_builds_with_every_registered_command(self):
        from nomorals.cli import CLI_ALIASES, _cli_subparsers, _parser
        parser = _parser()
        sub = _cli_subparsers()
        self.assertIsNotNone(sub)
        for canonical in CLI_ALIASES:
            with self.subTest(command=canonical):
                self.assertIn(canonical, sub.choices,
                              f"command {canonical!r} fell out of the parser")

    def test_main_help_exits_zero(self):
        from nomorals.cli import main
        self.assertEqual(main(["--help"]), 0)

    def test_entry_point_module_still_resolves(self):
        # python -m nomorals -> nomorals.__main__ -> nomorals.cli.main
        mod = importlib.import_module("nomorals.__main__")
        self.assertTrue(callable(mod.main))
        from nomorals.cli import main as cli_main
        self.assertTrue(callable(cli_main))


if __name__ == "__main__":
    unittest.main()
