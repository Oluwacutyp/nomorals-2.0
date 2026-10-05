"""Console UI: palette, dashboard, commands, and the local-adapter hook."""

from __future__ import annotations

import io
import logging
import unittest

from nomorals.console import ConsoleCommands, render_dashboard, supports_color
from nomorals.console.dashboard import _fmt_uptime, render_status_line
from nomorals.console.palette import (
    MAGENTA,
    paint,
    strip_ansi,
)
from nomorals.core.logging_setup import _ConsoleFormatter


def _snap(**over):
    base = {
        "uptime_s": 3723.0,
        "adapters": {
            "local": {"running": True, "received": 3, "sent": 2},
            "telegram": {"running": True, "received": 12, "sent": 9},
        },
        "traffic": {"messages": 40, "replies": 32, "errors": 0, "controls": 1},
        "scheduler": {
            "running": True,
            "jobs": [
                {"name": "briefing", "spec": "daily 08:00", "enabled": True,
                 "next_run": 1760000000.0},
            ],
        },
        "games": {"players": 7},
        "extras": {"autonomy": "on", "arena": "off"},
    }
    base.update(over)
    return base


class PaletteTests(unittest.TestCase):
    def test_no_red_or_black_in_palette(self):
        import nomorals.console.palette as p

        for name in dir(p):
            if name.isupper():
                val = getattr(p, name)
                if isinstance(val, str) and val.startswith("\033["):
                    self.assertNotIn("31m", val, f"{name} contains red")
                    self.assertNotIn("40m", val, f"{name} contains black bg")
                    self.assertNotIn("41m", val, f"{name} contains red bg")

    def test_paint_noop_without_color(self):
        self.assertEqual(paint("hi", MAGENTA, color=False), "hi")

    def test_paint_wraps_with_color(self):
        out = paint("hi", MAGENTA, color=True)
        self.assertTrue(out.startswith(MAGENTA))
        self.assertTrue(out.endswith("\033[0m"))

    def test_strip_ansi(self):
        self.assertEqual(strip_ansi(paint("hi", MAGENTA, color=True)), "hi")

    def test_supports_color_returns_bool(self):
        self.assertIsInstance(supports_color(), bool)


class LogColorTests(unittest.TestCase):
    def test_error_is_not_red(self):
        colors = _ConsoleFormatter.COLORS
        for level in ("ERROR", "CRITICAL"):
            self.assertNotIn("31m", colors[level], f"{level} must not be red")
            self.assertNotIn("41m", colors[level], f"{level} must not use red bg")

    def test_formatter_renders_all_levels(self):
        fmt = _ConsoleFormatter(color=False)
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            rec = logging.LogRecord("nomorals.test", getattr(logging, level),
                                    __file__, 1, "msg %s", ("x",), None)
            out = fmt.format(rec)
            self.assertIn("msg x", out)
            self.assertIn(level[:4], out)

    def test_formatter_strips_nomorals_prefix(self):
        fmt = _ConsoleFormatter(color=False)
        rec = logging.LogRecord("nomorals.social.chat.telegram", logging.INFO,
                                __file__, 1, "hello", (), None)
        out = fmt.format(rec)
        self.assertIn("social.chat.telegram", out)
        self.assertNotIn("nomorals.social", out)


class DashboardTests(unittest.TestCase):
    def test_renders_all_sections(self):
        out = strip_ansi(render_dashboard(_snap(), color=True))
        for needle in ("uptime", "adapters", "traffic", "scheduler",
                       "players", "briefing", "autonomy"):
            self.assertIn(needle, out)

    def test_empty_snapshot_does_not_crash(self):
        out = strip_ansi(render_dashboard({}, color=True))
        self.assertIn("DEVON", out)

    def test_none_snapshot_does_not_crash(self):
        out = strip_ansi(render_dashboard(None, color=True))
        self.assertIn("DEVON", out)

    def test_errors_highlighted(self):
        snap = _snap(traffic={"messages": 1, "replies": 0, "errors": 3, "controls": 0})
        out = strip_ansi(render_dashboard(snap, color=True))
        self.assertIn("3 errors", out)

    def test_color_false_strips_ansi(self):
        out = render_dashboard(_snap(), color=False)
        self.assertNotIn("\033[", out)

    def test_fmt_uptime(self):
        self.assertEqual(_fmt_uptime(45), "45s")
        self.assertEqual(_fmt_uptime(125), "2m 5s")
        self.assertEqual(_fmt_uptime(3723), "1h 2m 3s")
        self.assertEqual(_fmt_uptime(90061), "1d 1h 1m")

    def test_status_line(self):
        out = strip_ansi(render_status_line(_snap()))
        self.assertIn("up", out)
        self.assertIn("adapters", out)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.cmd = ConsoleCommands(lambda: _snap())

    def test_dashboard_command(self):
        out = self.cmd.handle("dashboard")
        self.assertIsNotNone(out)
        self.assertIn("DEVON", strip_ansi(out))

    def test_status_command(self):
        out = self.cmd.handle("status")
        self.assertIsNotNone(out)

    def test_jobs_command(self):
        out = self.cmd.handle("jobs")
        self.assertIsNotNone(out)
        self.assertIn("briefing", strip_ansi(out))

    def test_jobs_command_empty(self):
        cmd = ConsoleCommands(lambda: {})
        out = cmd.handle("jobs")
        self.assertIn("no jobs", strip_ansi(out))

    def test_clear_command(self):
        out = self.cmd.handle("clear")
        self.assertIn("\033[2J", out)

    def test_help_command(self):
        out = self.cmd.handle("help")
        self.assertIn("dashboard", strip_ansi(out))

    def test_unknown_text_passes_through(self):
        self.assertIsNone(self.cmd.handle("hello devon how are you"))
        self.assertIsNone(self.cmd.handle("/game stats"))

    def test_broken_snapshot_provider_does_not_crash(self):
        def _boom():
            raise RuntimeError("db gone")

        cmd = ConsoleCommands(_boom)
        out = cmd.handle("dashboard")
        self.assertIsNotNone(out)


class LocalHookTests(unittest.TestCase):
    def test_hook_response_printed_and_not_dispatched(self):
        from nomorals.social.chat.local import LocalAdapter

        out = io.StringIO()
        seen = []
        adapter = LocalAdapter(out=out, command_hook=lambda t: "HOOKED" if t == "dashboard" else None)
        # Drive run() with fake stdin.
        import sys
        old = sys.stdin
        sys.stdin = io.StringIO("dashboard\nhello\nexit\n")
        try:
            adapter.run(seen.append)
        finally:
            sys.stdin = old
        self.assertEqual([m.text for m in seen], ["hello"])
        self.assertIn("HOOKED", out.getvalue())

    def test_no_hook_normal_flow(self):
        from nomorals.social.chat.local import LocalAdapter

        out = io.StringIO()
        adapter = LocalAdapter(out=out)
        seen = []
        import sys
        old = sys.stdin
        sys.stdin = io.StringIO("hi there\nexit\n")
        try:
            adapter.run(seen.append)
        finally:
            sys.stdin = old
        self.assertEqual([m.text for m in seen], ["hi there"])

    def test_broken_hook_does_not_kill_input(self):
        from nomorals.social.chat.local import LocalAdapter

        def _boom(text):
            raise RuntimeError("hook exploded")

        out = io.StringIO()
        adapter = LocalAdapter(out=out, command_hook=_boom)
        seen = []
        import sys
        old = sys.stdin
        sys.stdin = io.StringIO("still works\nexit\n")
        try:
            adapter.run(seen.append)
        finally:
            sys.stdin = old
        self.assertEqual([m.text for m in seen], ["still works"])


class ThemeTests(unittest.TestCase):
    def test_three_themes_listed(self):
        from nomorals.console.themes import list_themes

        themes = list_themes()
        self.assertIn("ocean", themes)
        self.assertIn("violet", themes)
        self.assertIn("sunrise", themes)

    def test_no_banned_codes_in_any_theme(self):
        from nomorals.console.themes import assert_no_banned_codes

        self.assertEqual(assert_no_banned_codes(), [])

    def test_unknown_theme_falls_back(self):
        from nomorals.console.themes import DEFAULT_THEME, get_theme

        self.assertEqual(get_theme("nope-not-a-theme"), get_theme(DEFAULT_THEME))


class BannerTests(unittest.TestCase):
    def test_banner_has_logo_and_adapters(self):
        from nomorals.console.banner import render_banner

        out = render_banner(["local", "telegram"], color=False, tip_seed=1)
        self.assertIn("████", out)
        self.assertIn("local", out)
        self.assertIn("telegram", out)

    def test_tip_of_the_day_deterministic(self):
        from nomorals.console.banner import tip_of_the_day

        self.assertEqual(tip_of_the_day(seed=42), tip_of_the_day(seed=42))
        self.assertTrue(tip_of_the_day(seed=7))


class WidgetTests(unittest.TestCase):
    def test_sparkline_shape(self):
        from nomorals.console.widgets import sparkline

        out = sparkline([1, 2, 3, 4], width=4, color=False)
        self.assertEqual(len(out), 4)

    def test_sparkline_empty(self):
        from nomorals.console.widgets import sparkline

        out = sparkline([], width=8, color=False)
        self.assertEqual(out, "─" * 8)

    def test_progress_bar_completes(self):
        import io as _io

        from nomorals.console.widgets import ProgressBar

        out = _io.StringIO()
        bar = ProgressBar("test", total=10, out=out, color=False, show_eta=False)
        for i in range(1, 11):
            bar.update(i)
        bar.done()
        text = out.getvalue()
        self.assertIn("100.0%", text)
        self.assertIn("done", text)

    def test_message_card_has_parts(self):
        from nomorals.console.widgets import format_message_card

        card = format_message_card(
            platform="telegram", sender="Mary", text="/game stats",
            chat_title="xauusd", timestamp=0, color=False,
        )
        self.assertIn("telegram", card)
        self.assertIn("Mary", card)
        self.assertIn("/game stats", card)

    def test_live_screen_draw_no_color(self):
        import io as _io

        from nomorals.console.widgets import LiveScreen

        out = _io.StringIO()
        screen = LiveScreen(interval=0.01, color=False, out=out)
        with screen:
            screen.draw("hello")
            screen.stop()
        self.assertIn("hello", out.getvalue())


class WatchCommandTests(unittest.TestCase):
    def _cmds(self):
        from nomorals.console import ConsoleCommands

        return ConsoleCommands(lambda: {"uptime_s": 5})

    def test_watch_without_tty_returns_hint(self):
        import unittest.mock as mock

        cmds = self._cmds()
        with mock.patch("nomorals.console.commands.supports_color", return_value=False):
            out = cmds.handle("dashboard --watch")
        self.assertIsNotNone(out)
        self.assertIn("dashboard", out)

    def test_theme_command_lists(self):
        cmds = self._cmds()
        out = cmds.handle("theme")
        self.assertIsNotNone(out)
        self.assertIn("ocean", out)

    def test_theme_command_switches(self):
        import os

        cmds = self._cmds()
        old = os.environ.get("NM_CONSOLE_THEME")
        try:
            out = cmds.handle("theme violet")
            self.assertIsNotNone(out)
            self.assertEqual(os.environ.get("NM_CONSOLE_THEME"), "violet")
            self.assertIn("violet", out)
        finally:
            if old is None:
                os.environ.pop("NM_CONSOLE_THEME", None)
            else:
                os.environ["NM_CONSOLE_THEME"] = old

    def test_theme_command_rejects_unknown(self):
        cmds = self._cmds()
        out = cmds.handle("theme neon-nope")
        self.assertIsNotNone(out)
        self.assertIn("unknown theme", out)

    def test_tip_command(self):
        cmds = self._cmds()
        out = cmds.handle("tip")
        self.assertIsNotNone(out)
        self.assertTrue(len(out) > 5)

    def test_dashboard_renders_llm_and_history(self):
        from nomorals.console import render_dashboard

        snap = {
            "uptime_s": 60,
            "llm": {"active": "groq", "chain": ["groq", "hf"], "health": {}},
            "history": [0, 1, 3, 2, 5],
            "traffic": {"messages": 11, "replies": 9, "errors": 0, "controls": 0},
        }
        out = render_dashboard(snap, color=False)
        self.assertIn("groq", out)
        # sparkline chars present
        self.assertTrue(any(c in out for c in "▁▂▃▄▅▆▇█"))

    def test_gateway_mirror_hook_exists(self):
        from nomorals.social.chat.gateway import ChatGateway

        self.assertTrue(hasattr(ChatGateway, "__init__"))
        import inspect

        src = inspect.getsource(ChatGateway.__init__)
        self.assertIn("console_mirror", src)


if __name__ == "__main__":
    unittest.main()
