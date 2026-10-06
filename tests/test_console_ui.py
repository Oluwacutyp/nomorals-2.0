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
        import io as _io
        import unittest.mock as mock

        cmds = self._cmds()
        # GodScreen with color disabled returns the hint instead of blocking.
        with mock.patch(
            "nomorals.console.widgets.supports_color", return_value=False
        ):
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


class GodTierTests(unittest.TestCase):
    def test_message_feed_push_and_recent(self):
        from nomorals.console.widgets import MessageEvent, MessageFeed

        feed = MessageFeed(capacity=10)
        feed.push(MessageEvent(platform="telegram", sender="Mary", text="hi"))
        feed.push(MessageEvent(platform="telegram", sender="Devon", text="yo",
                               incoming=False))
        recent = feed.recent(5)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0].sender, "Mary")
        self.assertEqual(feed.unread, 2)
        self.assertEqual(feed.mark_read(), 2)
        self.assertEqual(feed.unread, 0)

    def test_message_feed_capacity(self):
        from nomorals.console.widgets import MessageEvent, MessageFeed

        feed = MessageFeed(capacity=10)
        for i in range(25):
            feed.push(MessageEvent(sender=f"u{i}", text="x"))
        self.assertEqual(len(feed), 10)
        self.assertEqual(feed.recent(1)[0].sender, "u24")

    def test_watch_hub_active_flag(self):
        from nomorals.console.widgets import WatchHub

        old = WatchHub.is_active()
        try:
            WatchHub.set_active(True)
            self.assertTrue(WatchHub.is_active())
            WatchHub.set_active(False)
            self.assertFalse(WatchHub.is_active())
        finally:
            WatchHub.set_active(old)

    def test_watch_hub_feed_shared(self):
        from nomorals.console.widgets import MessageEvent, WatchHub

        WatchHub.feed().push(MessageEvent(sender="t", text="shared"))
        self.assertTrue(len(WatchHub.feed()) >= 1)

    def test_barchart_shape(self):
        from nomorals.console.widgets import barchart

        lines = barchart([("mafia", 12), ("c4", 4)], color=False)
        self.assertEqual(len(lines), 2)
        self.assertIn("mafia", lines[0])
        self.assertIn("12", lines[0])
        # longer bar for the bigger value
        self.assertGreater(lines[0].count("█"), lines[1].count("█"))

    def test_barchart_empty(self):
        from nomorals.console.widgets import barchart

        lines = barchart([], color=False)
        self.assertEqual(len(lines), 1)

    def test_gradient_no_banned_colors(self):
        from nomorals.console.widgets import gradient_text

        out = gradient_text("DEVON", 51, 201, color=True)
        self.assertNotIn("31m", out)  # no red
        self.assertIn("38;5;", out)   # 256-color codes used
        plain = gradient_text("DEVON", 51, 201, color=False)
        self.assertEqual(plain, "DEVON")

    def test_feed_line_format(self):
        from nomorals.console.widgets import MessageEvent, format_feed_line

        ev = MessageEvent(platform="telegram", sender="Mary",
                          text="/game stats", chat_title="grp",
                          timestamp=1760000000.0)
        line = format_feed_line(ev, color=False)
        self.assertIn("Mary", line)
        self.assertIn("/game stats", line)
        self.assertIn("grp", line)

    def test_render_view_dispatch(self):
        from nomorals.console import render_view
        from nomorals.console.dashboard import strip_ansi as _s

        for view in ("status", "games", "jobs", "brain"):
            out = _s(render_view({"uptime_s": 60}, view, color=True))
            self.assertTrue(len(out) > 20, view)

    def test_render_games_view(self):
        from nomorals.console import render_games_view
        from nomorals.console.dashboard import strip_ansi as _s

        snap = {"games": {"players": 5, "active_tables": 2,
                          "activity": {"mafia": 10, "c4": 3},
                          "top_players": [{"name": "Mary", "score": 99}]}}
        out = _s(render_games_view(snap, color=True))
        self.assertIn("mafia", out)
        self.assertIn("Mary", out)

    def test_render_scheduler_view(self):
        from nomorals.console import render_scheduler_view
        from nomorals.console.dashboard import strip_ansi as _s

        snap = {"scheduler": {"running": True, "jobs": [
            {"name": "briefing", "spec": "daily 08:00", "enabled": True,
             "next_run": 1760000000.0},
            {"name": "off-job", "spec": "daily", "enabled": False},
        ]}}
        out = _s(render_scheduler_view(snap, color=True))
        self.assertIn("briefing", out)
        self.assertIn("1/2 enabled", out)

    def test_render_llm_view(self):
        from nomorals.console import render_llm_view
        from nomorals.console.dashboard import strip_ansi as _s

        snap = {"llm": {"active": "groq", "chain": ["groq", "hf"],
                        "health": {"hf": {"cooldown_until": 9999999999}},
                        "stats": {"groq": {"calls": 42}}}}
        out = _s(render_llm_view(snap, color=True))
        self.assertIn("groq", out)
        self.assertIn("cooling down", out)

    def test_render_statusbar(self):
        from nomorals.console import render_statusbar
        from nomorals.console.dashboard import strip_ansi as _s

        snap = {"uptime_s": 3723, "traffic": {"messages": 40, "errors": 0},
                "games": {"active_tables": 2}, "llm": {"active": "groq"}}
        out = _s(render_statusbar(snap, "status", unread=3, color=True))
        self.assertIn("1h 2m 3s", out)
        self.assertIn("groq", out)
        self.assertIn("3 new", out)

    def test_godscreen_no_color_returns_hint(self):
        import io as _io

        from nomorals.console.widgets import GodScreen

        screen = GodScreen(interval=0.1, snapshot=lambda: {},
                           color=False, out=_io.StringIO())
        out = screen.run()
        self.assertIn("dashboard", out)

    def test_godscreen_view_switch_keys(self):
        from nomorals.console.widgets import WATCH_VIEW_KEYS, WATCH_VIEWS

        self.assertEqual(len(WATCH_VIEWS), 5)
        self.assertEqual(WATCH_VIEW_KEYS["1"], "status")
        self.assertEqual(WATCH_VIEW_KEYS["4"], "brain")
        self.assertEqual(WATCH_VIEW_KEYS["d"], "debug")
        self.assertIn("debug", WATCH_VIEWS)


class TruncateVisibleTests(unittest.TestCase):
    def test_plain_truncation(self):
        from nomorals.console.palette import truncate_visible, visible_width

        t = truncate_visible("hello world", 5)
        self.assertLessEqual(visible_width(t), 5)
        self.assertIn("…", t)

    def test_no_truncation_when_fits(self):
        from nomorals.console.palette import truncate_visible

        self.assertEqual(truncate_visible("hi", 10), "hi")

    def test_ansi_width_ignored_and_reset_added(self):
        from nomorals.console.palette import (
            RESET,
            CYAN,
            paint,
            truncate_visible,
            visible_width,
        )

        colored = paint("hello world", CYAN)
        t = truncate_visible(colored, 5)
        self.assertLessEqual(visible_width(t), 5)
        self.assertTrue(t.endswith(RESET))

    def test_wide_chars_count_double(self):
        from nomorals.console.palette import truncate_visible, visible_width

        self.assertGreaterEqual(visible_width("✈️"), 2)
        t = truncate_visible("✈️abcdefghij", 6)
        self.assertLessEqual(visible_width(t), 6)

    def test_gradient_sequence_survives(self):
        from nomorals.console.palette import truncate_visible, visible_width
        from nomorals.console.widgets import gradient_text

        g = gradient_text("DEVON LIVE", 51, 201, color=True)
        t = truncate_visible(g, 5)
        self.assertLessEqual(visible_width(t), 5)
        self.assertIn("38;5;", t)  # 256-color escapes preserved


class AlternateScreenTests(unittest.TestCase):
    def _run_one_frame(self, screen):
        import unittest.mock as mock

        # One frame renders, then quit.
        with mock.patch.object(
            type(screen), "_wait_key", side_effect=[True, False]
        ):
            return screen.run()

    def test_godscreen_uses_alternate_screen(self):
        import io as _io

        from nomorals.console.widgets import (
            _ALT_SCREEN_OFF,
            _ALT_SCREEN_ON,
            GodScreen,
        )

        out = _io.StringIO()
        screen = GodScreen(interval=0.01, snapshot=lambda: {"uptime_s": 5},
                           color=True, out=out)
        msg = self._run_one_frame(screen)
        self.assertIn("exited live dashboard", msg)
        data = out.getvalue()
        self.assertIn(_ALT_SCREEN_ON, data)
        self.assertIn(_ALT_SCREEN_OFF, data)
        # Enter comes before exit.
        self.assertLess(data.index(_ALT_SCREEN_ON),
                        data.index(_ALT_SCREEN_OFF))

    def test_livescreen_uses_alternate_screen(self):
        import io as _io

        from nomorals.console.widgets import (
            _ALT_SCREEN_OFF,
            _ALT_SCREEN_ON,
            LiveScreen,
        )

        out = _io.StringIO()
        screen = LiveScreen(interval=0.01, color=True, out=out)
        with screen:
            screen.draw("hello")
            screen.stop()
        data = out.getvalue()
        self.assertIn("hello", data)
        self.assertIn(_ALT_SCREEN_ON, data)
        self.assertIn(_ALT_SCREEN_OFF, data)
        self.assertLess(data.index(_ALT_SCREEN_ON),
                        data.index(_ALT_SCREEN_OFF))

    def test_godscreen_frame_fits_terminal(self):
        import io as _io
        import unittest.mock as mock

        from nomorals.console.palette import visible_width
        from nomorals.console.widgets import GodScreen, MessageEvent, WatchHub

        # Long feed lines must not wrap: every rendered line fits 80 cols.
        WatchHub.feed().push(MessageEvent(
            platform="telegram", sender="Spammer",
            text="x" * 500, chat_title="y" * 100))
        out = _io.StringIO()
        screen = GodScreen(interval=0.01,
                           snapshot=lambda: {"uptime_s": 3723,
                                             "traffic": {"messages": 40}},
                           color=True, out=out)
        with mock.patch("shutil.get_terminal_size", return_value=(80, 24)):
            screen._render_frame()
        for line in out.getvalue().split("\n"):
            self.assertLessEqual(visible_width(line), 80,
                                 f"line wraps: {line[:60]!r}")
        WatchHub.feed().mark_read()


class DebugHubTests(unittest.TestCase):
    def setUp(self):
        from nomorals.console.debug import DebugHub

        DebugHub.reset()
        DebugHub.install()
        self.addCleanup(DebugHub.uninstall)
        self.addCleanup(DebugHub.reset)

    def test_install_is_idempotent(self):
        import logging

        from nomorals.console.debug import DebugHub

        root = logging.getLogger()
        before = len(root.handlers)
        DebugHub.install()
        DebugHub.install()
        self.assertEqual(len(root.handlers), before)
        self.assertTrue(DebugHub.installed())

    def test_log_capture_and_level_counts(self):
        import logging

        from nomorals.console.debug import DebugHub

        log = logging.getLogger("test.debughub")
        log.setLevel(logging.DEBUG)
        log.info("hello world")
        log.warning("a warning")
        log.error("an error")
        counts = DebugHub.level_counts()
        self.assertGreaterEqual(counts.get("INFO", 0), 1)
        self.assertGreaterEqual(counts.get("WARNING", 0), 1)
        self.assertGreaterEqual(counts.get("ERROR", 0), 1)
        msgs = [m for _, _, _, m in DebugHub.recent(10)]
        self.assertTrue(any("hello world" in m for m in msgs))

    def test_slow_op_mined_from_log(self):
        import logging

        from nomorals.console.debug import DebugHub

        log = logging.getLogger("test.debughub.slow")
        log.setLevel(logging.DEBUG)
        log.debug("telegram: slow get_entity('x') took 1.23s")
        slow = DebugHub.slow_ops()
        self.assertTrue(slow)
        self.assertGreaterEqual(slow[0]["duration_s"], 1.0)
        self.assertIn("get_entity", slow[0]["name"])

    def test_fast_ops_ignored_from_log(self):
        import logging

        from nomorals.console.debug import DebugHub

        log = logging.getLogger("test.debughub.fast")
        log.setLevel(logging.DEBUG)
        log.debug("cache lookup took 0.01s")
        self.assertEqual(DebugHub.slow_ops(), [])

    def test_record_llm_and_timed(self):
        import time

        from nomorals.console.debug import DebugHub

        DebugHub.record_llm(operation="chat", provider_name="groq",
                            success=True, latency_s=1.5)
        DebugHub.record_llm(operation="chat", provider_name="groq",
                            success=False, latency_s=0.2, error="boom")
        calls = DebugHub.llm_calls(5)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["provider"], "groq")
        self.assertFalse(calls[1]["success"])
        with DebugHub.timed("unit op"):
            time.sleep(0.01)
        self.assertTrue(any(e["name"] == "unit op" for e in DebugHub.slow_ops()))

    def test_install_llm_hook_chains_existing(self):
        from nomorals.console.debug import DebugHub

        seen = []

        class FakeRouter:
            def __init__(self):
                self._hook = None

            def set_learning(self, hook):
                self._hook = hook

            def learning(self):
                return self._hook

        router = FakeRouter()
        router.set_learning(lambda **kw: seen.append(kw["provider_name"]))
        self.assertTrue(DebugHub.install_llm_hook(router))
        router._hook(operation="chat", provider_name="p1",
                     success=True, latency_s=0.5, error="")
        self.assertEqual(seen, ["p1"])  # existing hook still ran
        self.assertEqual(len(DebugHub.llm_calls(5)), 1)

    def test_handler_never_breaks_logging(self):
        import logging

        from nomorals.console.debug import DebugHub

        DebugHub.reset()
        DebugHub.install()
        log = logging.getLogger("test.debughub.broken")
        log.setLevel(logging.DEBUG)
        # A record whose getMessage raises must not propagate.
        rec = logging.LogRecord("x", logging.INFO, __file__, 1, "%s", ("ok",),
                                None)
        logging.getLogger().handle(rec)  # goes through all handlers
        log.info("still works")


class DebugViewTests(unittest.TestCase):
    def setUp(self):
        from nomorals.console.debug import DebugHub

        DebugHub.reset()
        DebugHub.install()
        self.addCleanup(DebugHub.uninstall)
        self.addCleanup(DebugHub.reset)

    def test_render_debug_view_sections(self):
        import logging

        from nomorals.console import render_debug_view, render_view
        from nomorals.console.palette import strip_ansi

        log = logging.getLogger("test.view")
        log.setLevel(logging.DEBUG)
        log.error("something broke")
        out = strip_ansi(render_debug_view({}, color=False))
        for section in ("recent logs", "slowest ops", "LLM calls", "errors"):
            self.assertIn(section, out)
        self.assertIn("something broke", out)

    def test_render_view_dispatches_debug(self):
        from nomorals.console import render_view
        from nomorals.console.palette import strip_ansi

        out = strip_ansi(render_view({}, "debug", color=False))
        self.assertIn("telemetry", out)

    def test_render_debug_view_no_red(self):
        from nomorals.console import render_debug_view

        out = render_debug_view({}, color=True)
        self.assertNotIn("\033[31m", out)
        self.assertNotIn("\033[41m", out)

    def test_statusbar_shows_debug_hint(self):
        from nomorals.console.dashboard import render_statusbar
        from nomorals.console.palette import strip_ansi

        out = strip_ansi(render_statusbar({}, "status", color=False))
        self.assertIn("d debug", out)

    def test_godscreen_header_has_mark_and_debug_tab(self):
        import io as _io

        from nomorals.console.palette import strip_ansi
        from nomorals.console.widgets import AVATAR, GodScreen

        self.assertEqual(AVATAR, "🥷")
        out_buf = _io.StringIO()
        screen = GodScreen(interval=0.1, snapshot=lambda: {},
                           color=True, out=out_buf)
        screen._render_frame()
        frame = strip_ansi(out_buf.getvalue())
        # Header: ninja mark + live + compact debug tab.
        self.assertIn("DEVON", frame)
        self.assertIn("live", frame)
        self.assertIn("﹝◉﹞", frame)
        self.assertIn("[d]", frame)


class NinjaAvatarTests(unittest.TestCase):
    def test_dimensions(self):
        from nomorals.console.avatar import (
            NINJA_HEIGHT,
            NINJA_MINI_HEIGHT,
            NINJA_MINI_WIDTH,
            NINJA_WIDTH,
            render_ninja,
            render_ninja_mini,
        )

        full = render_ninja(color=False).splitlines()
        self.assertEqual(len(full), NINJA_HEIGHT)
        self.assertEqual(NINJA_HEIGHT, 13)
        mini = render_ninja_mini(color=False).splitlines()
        self.assertEqual(len(mini), NINJA_MINI_HEIGHT)
        self.assertEqual(NINJA_MINI_HEIGHT, 7)
        for ln in full:
            self.assertLessEqual(len(ln), NINJA_WIDTH)
        for ln in mini:
            self.assertLessEqual(len(ln), NINJA_MINI_WIDTH)

    def test_eye_and_no_banned_colors(self):
        from nomorals.console.avatar import render_ninja, render_ninja_mini

        for fn in (render_ninja, render_ninja_mini):
            colored = fn(color=True)
            self.assertIn("◉", colored)  # the amber eye
            self.assertNotIn("\033[31m", colored)  # no red text
            self.assertNotIn("\033[40m", colored)  # no black background
            plain = fn(color=False)
            self.assertNotIn("\033[", plain)


class KeyReaderTests(unittest.TestCase):
    def test_canonical_fallback_reads_line(self):
        import io as _io
        import unittest.mock as mock

        from nomorals.console.widgets import _KeyReader

        reader = _KeyReader()
        reader._fd = None  # force canonical path
        fake_stdin = _io.StringIO("2\n")
        with mock.patch("sys.stdin", fake_stdin):
            with mock.patch("select.select", return_value=([fake_stdin], [], [])):
                self.assertEqual(reader.get(1.0), "2")

    def test_canonical_timeout_returns_none(self):
        import unittest.mock as mock

        from nomorals.console.widgets import _KeyReader

        reader = _KeyReader()
        reader._fd = None
        with mock.patch("select.select", return_value=([], [], [])):
            self.assertIsNone(reader.get(0.01))

    def test_raw_mode_parses_escape_as_esc(self):
        import unittest.mock as mock

        from nomorals.console.widgets import _KeyReader

        reader = _KeyReader()
        reader._fd = 99  # fake fd; os.read/select mocked
        with mock.patch("select.select", return_value=([99], [], [])), \
             mock.patch("os.read", return_value=b"\x1b[A"):
            self.assertEqual(reader._get_raw(1.0), "esc")
        with mock.patch("select.select", return_value=([99], [], [])), \
             mock.patch("os.read", return_value=b"d"):
            self.assertEqual(reader._get_raw(1.0), "d")
        with mock.patch("select.select", return_value=([99], [], [])), \
             mock.patch("os.read", return_value=b"\x03"):
            self.assertEqual(reader._get_raw(1.0), "q")  # Ctrl-C → quit


class ScreenGuardTests(unittest.TestCase):
    def test_mutes_logging_but_keeps_debug_capture(self):
        import io as _io
        import logging

        from nomorals.console.debug import DebugHub
        from nomorals.console.widgets import WatchHub, _ScreenGuard

        stream = _io.StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            DebugHub.install()
            DebugHub.reset()
            WatchHub.set_active(True)
            with _ScreenGuard(isolate=False, out=_io.StringIO()):
                logging.getLogger("test.guard").warning("muted line")
            WatchHub.set_active(False)
            # Nothing reached the terminal handler…
            self.assertNotIn("muted line", stream.getvalue())
            # …but DebugHub still captured it.
            recent = DebugHub.recent(5)
            self.assertTrue(any("muted line" in r[3] for r in recent))
            # After exit, logging works again.
            logging.getLogger("test.guard").warning("loud line")
            self.assertIn("loud line", stream.getvalue())
        finally:
            root.removeHandler(handler)
            WatchHub.set_active(False)
            DebugHub.uninstall()

    def test_debug_capture_handler_never_muted(self):
        import logging

        from nomorals.console.debug import DebugHub
        from nomorals.console.widgets import WatchHub, _ScreenGuard

        DebugHub.install()
        try:
            WatchHub.set_active(True)
            with _ScreenGuard(isolate=False):
                dh = DebugHub._handler
                self.assertTrue(getattr(dh, "_devon_debug_capture", False))
                self.assertEqual(
                    [f for f in dh.filters
                     if type(f).__name__ == "_WatchMuteFilter"], [])
        finally:
            WatchHub.set_active(False)
            DebugHub.uninstall()


class GodScreenLayoutTests(unittest.TestCase):
    def test_frame_pinned_to_terminal_height(self):
        import io as _io
        import shutil
        import unittest.mock as mock

        from nomorals.console.palette import strip_ansi
        from nomorals.console.widgets import GodScreen

        out = _io.StringIO()
        screen = GodScreen(interval=0.1, snapshot=lambda: {"uptime_s": 5},
                           color=True, out=out)
        with mock.patch.object(shutil, "get_terminal_size",
                               return_value=(100, 30)):
            screen._render_frame()
        frame = strip_ansi(out.getvalue())
        content = frame.split("\x1b[H", 1)[-1].split("\x1b[J")[0]
        self.assertEqual(len(content.splitlines()), 30)

    def test_pane_borders_and_status_bar(self):
        import io as _io

        from nomorals.console.palette import strip_ansi
        from nomorals.console.widgets import GodScreen

        out = _io.StringIO()
        screen = GodScreen(interval=0.1, snapshot=lambda: {"uptime_s": 5},
                           color=True, out=out)
        screen._render_frame()
        frame = strip_ansi(out.getvalue())
        # Grok layout discipline: box-drawing frame, one status block.
        self.assertIn("┌─ DEVON", frame)   # header corner
        self.assertIn("├─ messages", frame)  # feed separator
        self.assertIn("└─", frame)           # footer corner
        self.assertIn("q quit", frame)
        # status block drawn exactly once
        self.assertEqual(frame.count("├─ messages"), 1)

    def test_view_switch_changes_pane_title(self):
        import io as _io

        from nomorals.console.palette import strip_ansi
        from nomorals.console.widgets import GodScreen, WATCH_VIEW_KEYS

        for key, view in WATCH_VIEW_KEYS.items():
            out = _io.StringIO()
            screen = GodScreen(interval=0.1, snapshot=lambda: {},
                               color=True, out=out)
            screen._view = view
            screen._render_frame()
            frame = strip_ansi(out.getvalue())
            self.assertIn(view, frame)

    def test_ninja_mark_in_header(self):
        """Devon's face is one ANSI mark (﹝◉﹞), not multi-row art."""
        import io as _io
        import shutil
        import unittest.mock as mock

        from nomorals.console.palette import strip_ansi
        from nomorals.console.widgets import GodScreen

        out = _io.StringIO()
        screen = GodScreen(interval=0.1, snapshot=lambda: {},
                           color=False, out=out)
        with mock.patch.object(shutil, "get_terminal_size",
                               return_value=(80, 30)):
            screen._render_frame()
        frame = strip_ansi(out.getvalue())
        self.assertIn("﹝◉﹞", frame)
        self.assertIn("DEVON", frame)
        # the old 7-row figure is gone from the header
        self.assertNotIn("/ || \\", frame)

    def test_ninja_mark_shown_on_narrow_terminal(self):
        """The one-glyph mark fits even on narrow terminals."""
        import io as _io
        import shutil
        import unittest.mock as mock

        from nomorals.console.palette import strip_ansi
        from nomorals.console.widgets import GodScreen

        out = _io.StringIO()
        screen = GodScreen(interval=0.1, snapshot=lambda: {},
                           color=False, out=out)
        with mock.patch.object(shutil, "get_terminal_size",
                               return_value=(50, 24)):
            screen._render_frame()
        frame = strip_ansi(out.getvalue())
        self.assertIn("◉", frame)
        self.assertIn("DEVON", frame)
        # compact traffic row on narrow terminals
        self.assertIn("tg●", frame)

    def test_stealth_palette_in_frame(self):
        """Cyber-Stealth truecolor codes present; banned codes absent."""
        import io as _io
        import shutil
        import unittest.mock as mock

        from nomorals.console.palette import (
            STEALTH_AMBER,
            STEALTH_CYAN,
            STEALTH_TEXT,
            visible_width,
        )
        from nomorals.console.widgets import GodScreen

        snap = {"uptime_s": 60, "traffic": {"messages": 1, "errors": 2},
                "adapters": {"telegram": {"running": True,
                                          "received": 1, "sent": 0}}}
        out = _io.StringIO()
        screen = GodScreen(interval=0.1, snapshot=lambda: snap,
                           color=True, out=out)
        with mock.patch.object(shutil, "get_terminal_size",
                               return_value=(80, 24)):
            screen._render_frame()
        raw = out.getvalue()
        self.assertIn(STEALTH_CYAN, raw)   # borders, tabs, eye
        self.assertIn(STEALTH_AMBER, raw)  # err > 0 warning
        self.assertIn(STEALTH_TEXT, raw)   # body text
        for bad in ("[31m", "[40m", "[41m"):
            self.assertNotIn(bad, raw)
        for line in raw.split("\n"):
            self.assertLessEqual(visible_width(line), 80)

    def test_traffic_and_scheduler_rows(self):
        """Status view shows the one traffic row and next-jobs row."""
        import io as _io

        from nomorals.console.palette import strip_ansi
        from nomorals.console.widgets import GodScreen

        snap = {
            "adapters": {"telegram": {"running": True,
                                       "received": 68, "sent": 0}},
            "traffic": {"messages": 70, "errors": 0},
            "scheduler": {"jobs": [
                {"name": "rooms tick", "enabled": True,
                 "next_run": 1760000000}]},
            "extras": {"autonomy": "on", "arena": "off"},
        }
        out = _io.StringIO()
        screen = GodScreen(interval=0.1, snapshot=lambda: snap,
                           color=False, out=out)
        screen._render_frame()
        frame = strip_ansi(out.getvalue())
        self.assertIn("tg", frame)
        self.assertIn("68/0", frame)
        self.assertIn("autonomy ON", frame)
        self.assertIn("next:", frame)
        self.assertIn("rooms tick", frame)


class DashboardCommandWiringTests(unittest.TestCase):
    """`dashboard` / `dashboard --watch` reach the right handlers."""

    def _commands(self):
        from nomorals.console.commands import ConsoleCommands

        return ConsoleCommands(lambda: {"uptime_s": 5})

    def test_dashboard_static(self):
        cc = self._commands()
        out = cc.handle("dashboard")
        self.assertIn("live status", out)

    def test_dashboard_watch_invokes_godscreen(self):
        import unittest.mock as mock

        cc = self._commands()
        with mock.patch("nomorals.console.widgets.GodScreen") as MockGS:
            MockGS.return_value.run.return_value = "exited live dashboard"
            out = cc.handle("dashboard --watch")
            MockGS.assert_called_once()
            MockGS.return_value.run.assert_called_once()
            self.assertEqual(out, "exited live dashboard")
            # interval parsed, snapshot wired
            kwargs = MockGS.call_args.kwargs
            self.assertEqual(kwargs.get("interval"), 2.0)
            self.assertTrue(callable(kwargs.get("snapshot")))

    def test_dashboard_watch_custom_interval(self):
        import unittest.mock as mock

        cc = self._commands()
        with mock.patch("nomorals.console.widgets.GodScreen") as MockGS:
            MockGS.return_value.run.return_value = "x"
            cc.handle("dashboard --watch 5")
            self.assertEqual(MockGS.call_args.kwargs.get("interval"), 5.0)

    def test_dashboard_watch_failure_is_visible(self):
        import unittest.mock as mock

        from nomorals.console.palette import strip_ansi

        cc = self._commands()
        with mock.patch("nomorals.console.widgets.GodScreen",
                        side_effect=RuntimeError("boom")):
            out = cc.handle("dashboard --watch")
            plain = strip_ansi(out)
            self.assertIn("watch mode failed", plain)
            self.assertIn("live status", plain)  # static fallback included

    def test_non_command_passes_through(self):
        cc = self._commands()
        self.assertIsNone(cc.handle("hello there"))
        self.assertIsNone(cc.handle(""))

    def test_runtime_wires_command_hook(self):
        import unittest.mock as mock

        with mock.patch("nomorals.agents.partner.runtime.PartnerBrain"), \
             mock.patch("nomorals.agents.coremind.CoreMind"):
            from nomorals.agents.partner.runtime import PartnerRuntime

            ctx = mock.MagicMock()
            ctx.settings.partner.max_parallel_chats = 2
            ctx.settings.partner.platforms = ""
            ctx.settings.chat.local_enabled = False
            ctx.extras = {}
            rt = PartnerRuntime(ctx)
            local = rt.gateway.adapters.get("local")
            self.assertIsNotNone(local)
            self.assertIsNotNone(getattr(local, "command_hook", None))
            # the wired hook serves the static dashboard
            out = local.command_hook("dashboard")
            self.assertIn("live status", out)

    def test_console_snapshot_shape(self):
        import unittest.mock as mock

        with mock.patch("nomorals.agents.partner.runtime.PartnerBrain"), \
             mock.patch("nomorals.agents.coremind.CoreMind"):
            from nomorals.agents.partner.runtime import PartnerRuntime

            ctx = mock.MagicMock()
            ctx.settings.partner.max_parallel_chats = 2
            ctx.settings.partner.platforms = ""
            ctx.settings.chat.local_enabled = False
            ctx.extras = {}
            ctx.db.query_one.return_value = {"n": 7}
            rt = PartnerRuntime(ctx)
            snap = rt.console_snapshot()
            for key in ("uptime_s", "adapters", "traffic", "scheduler",
                        "games", "llm", "theme", "extras"):
                self.assertIn(key, snap)
            self.assertEqual(snap["games"], {"players": 7})


if __name__ == "__main__":
    unittest.main()
