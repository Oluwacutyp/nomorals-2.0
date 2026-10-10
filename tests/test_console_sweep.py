"""Console sweep tests: new themes, banner/avatar styles, widgets, commands."""

from __future__ import annotations

import io
import logging
import os
import unittest


def _snap(**over):
    base = {
        "uptime_s": 3723.0,
        "adapters": {
            "local": {"running": True, "received": 3, "sent": 2},
            "telegram": {"running": True, "received": 12, "sent": 9},
        },
        "traffic": {"messages": 40, "replies": 32, "errors": 1, "controls": 1},
        "scheduler": {
            "running": True,
            "jobs": [
                {"name": "briefing", "spec": "daily 08:00", "enabled": True,
                 "next_run": 1760000000.0},
            ],
        },
        "games": {"players": 7},
        "extras": {"autonomy": "on", "arena": "off"},
        "llm": {"active": "groq", "chain": ["groq", "hf"], "health": {}},
    }
    base.update(over)
    return base


class PaletteSweepTests(unittest.TestCase):
    def test_hex_color(self):
        from nomorals.console.palette import hex_color, strip_ansi

        code = hex_color("#00F0FF")
        self.assertTrue(code.startswith("\033[38;2;0;240;255m"))
        self.assertEqual(hex_color("fff"), hex_color("#ffffff"))
        with self.assertRaises(ValueError):
            hex_color("zzzzzz")

    def test_rgb_clamps(self):
        from nomorals.console.palette import rgb

        self.assertEqual(rgb(300, -5, 128), "\033[38;2;255;0;128m")

    def test_no_banned_codes_in_new_constants(self):
        import nomorals.console.palette as p

        for name in dir(p):
            if name.isupper():
                val = getattr(p, name)
                if isinstance(val, str) and val.startswith("\033["):
                    self.assertNotIn("31m", val, name)
                    self.assertNotIn("40m", val, name)
                    self.assertNotIn("41m", val, name)

    def test_paint_sem(self):
        from nomorals.console.palette import SEMANTIC, paint_sem

        self.assertIn("type", SEMANTIC)
        out = paint_sem("Server", "type", color=True)
        self.assertIn("Server", out)
        self.assertTrue(out.startswith("\033["))
        self.assertEqual(paint_sem("x", "nope", color=False), "x")

    def test_pad_visible(self):
        from nomorals.console.palette import pad_visible, visible_width

        self.assertEqual(pad_visible("ab", 5), "ab   ")
        self.assertEqual(pad_visible("ab", 5, "right"), "   ab")
        self.assertEqual(pad_visible("ab", 5, "center"), " ab  ")
        self.assertEqual(visible_width(pad_visible("ab", 5)), 5)

    def test_wrap_visible(self):
        from nomorals.console.palette import visible_width, wrap_visible

        lines = wrap_visible("aa bb cc dd", 5)
        self.assertEqual(lines, ["aa bb", "cc dd"])
        for ln in lines:
            self.assertLessEqual(visible_width(ln), 5)

    def test_box_glyph_sets(self):
        from nomorals.console.palette import (
            BOX_DOUBLE, BOX_HEAVY, BOX_ROUNDED, BOX_SINGLE,
        )

        for box in (BOX_SINGLE, BOX_DOUBLE, BOX_ROUNDED, BOX_HEAVY):
            for key in ("tl", "tr", "bl", "br", "h", "v"):
                self.assertIn(key, box)


class ThemeSweepTests(unittest.TestCase):
    def test_seven_themes(self):
        from nomorals.console.themes import list_themes

        names = list_themes()
        for want in ("ocean", "violet", "sunrise", "tokyo-night", "nord",
                     "dracula", "catppuccin"):
            self.assertIn(want, names)

    def test_all_themes_validate(self):
        from nomorals.console.themes import (
            assert_no_banned_codes, get_theme, list_themes, validate_theme,
        )

        self.assertEqual(assert_no_banned_codes(), [])
        for name in list_themes():
            self.assertEqual(validate_theme(get_theme(name)), [], name)

    def test_roles_contract(self):
        from nomorals.console.themes import ROLES, get_theme

        for role in ("title", "accent", "info", "ok", "warn", "err", "crit",
                     "subtle", "bright", "dim", "bold", "border", "chart"):
            self.assertIn(role, ROLES)
        theme = get_theme("ocean")
        for role in ROLES:
            self.assertIn(role, theme, role)

    def test_unknown_theme_falls_back(self):
        from nomorals.console.themes import get_theme

        self.assertEqual(get_theme("nope"), get_theme("ocean"))

    def test_theme_preview(self):
        from nomorals.console.themes import theme_preview

        out = theme_preview("dracula", color=False)
        self.assertIn("dracula", out)
        self.assertIn("title", out)
        self.assertIn("chart", out)

    def test_describe_theme(self):
        from nomorals.console.themes import describe_theme

        d = describe_theme("nord")
        self.assertEqual(d["name"], "nord")
        self.assertIn("role.title", d)


class BannerSweepTests(unittest.TestCase):
    def test_styles(self):
        from nomorals.console.banner import BANNER_STYLES, list_banner_styles

        self.assertEqual(list_banner_styles(), list(BANNER_STYLES))
        self.assertIn("block", BANNER_STYLES)
        self.assertIn("slant", BANNER_STYLES)

    def test_each_style_renders(self):
        from nomorals.console.banner import list_banner_styles, render_banner

        for style in list_banner_styles():
            out = render_banner(["local"], style=style, color=False, tip_seed=7)
            self.assertIn("local", out)
            self.assertGreater(len(out.splitlines()), 5, style)

    def test_default_style_unchanged(self):
        from nomorals.console.banner import render_banner

        out = render_banner(["local"], color=False, tip_seed=7)
        self.assertIn("██████╗", out)  # the classic block logo

    def test_gradient_and_frame(self):
        from nomorals.console.banner import render_banner

        g = render_banner(["local"], gradient=True, color=True, tip_seed=7)
        self.assertIn("\033[", g)
        f = render_banner(["local"], frame=True, color=False, tip_seed=7)
        self.assertIn("╭", f)
        self.assertIn("╯", f)

    def test_unknown_style_falls_back(self):
        from nomorals.console.banner import render_banner

        out = render_banner(["local"], style="nope", color=False, tip_seed=7)
        self.assertIn("██████╗", out)


class AvatarSweepTests(unittest.TestCase):
    def test_styles(self):
        from nomorals.console.avatar import AVATAR_STYLES, list_avatar_styles

        self.assertEqual(list_avatar_styles(), list(AVATAR_STYLES))
        for s in ("mini", "full", "braille", "pixel", "wide"):
            self.assertIn(s, AVATAR_STYLES)

    def test_render_each_style(self):
        from nomorals.console.avatar import list_avatar_styles, render_avatar

        for style in list_avatar_styles():
            out = render_avatar(style, color=False)
            self.assertGreater(len(out.splitlines()), 3, style)

    def test_original_art_untouched(self):
        from nomorals.console.avatar import (
            NINJA_HEIGHT, NINJA_MINI_HEIGHT, render_avatar, render_ninja,
            render_ninja_mini,
        )

        self.assertEqual(render_avatar("mini", color=False), render_ninja_mini(color=False))
        self.assertEqual(render_avatar("full", color=False), render_ninja(color=False))
        self.assertEqual(NINJA_HEIGHT, 13)
        self.assertEqual(NINJA_MINI_HEIGHT, 7)

    def test_framed_avatar(self):
        from nomorals.console.avatar import render_avatar

        out = render_avatar("mini", color=False, frame=True)
        self.assertIn("╭", out)
        self.assertIn("╯", out)

    def test_avatar_size(self):
        from nomorals.console.avatar import avatar_size

        w, h = avatar_size("wide")
        self.assertGreater(w, 20)
        self.assertGreater(h, 10)


class CommandSweepTests(unittest.TestCase):
    def _cmd(self):
        from nomorals.console.commands import ConsoleCommands

        return ConsoleCommands(lambda: _snap())

    def test_complete(self):
        from nomorals.console.commands import complete

        self.assertIn("dashboard", complete("dash"))
        self.assertIn("theme", complete("t"))
        self.assertEqual(complete("zzz"), [])

    def test_new_commands_dispatch(self):
        cmd = self._cmd()
        for line in ("banner", "banner slant", "palette", "uptime", "errors",
                     "slow", "llm", "log", "log error 5", "history",
                     "help", "help theme", "theme", "tip"):
            out = cmd.handle(line)
            self.assertIsInstance(out, str, line)
            self.assertTrue(out.strip(), line)

    def test_banner_unknown_style(self):
        cmd = self._cmd()
        out = cmd.handle("banner nope")
        self.assertIn("unknown style", out)

    def test_export(self):
        import tempfile

        cmd = self._cmd()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "dash.txt")
            out = cmd.handle(f"export {path}")
            self.assertIn("exported", out)
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn("devon", content.lower())
            self.assertIn("raw snapshot", content)
        self.assertIn("usage", cmd.handle("export"))

    def test_history_records(self):
        cmd = self._cmd()
        cmd.handle("status")
        cmd.handle("theme")
        out = cmd.handle("history")
        self.assertIn("status", out)
        self.assertIn("theme", out)

    def test_help_per_command(self):
        cmd = self._cmd()
        out = cmd.handle("help dashboard")
        self.assertIn("dashboard", out)
        self.assertIn("unknown command", cmd.handle("help zzz"))

    def test_highlight(self):
        cmd = self._cmd()
        out = cmd.highlight("dashboard --watch")
        self.assertIn("dashboard", out)

    def test_unknown_still_flows_to_brain(self):
        cmd = self._cmd()
        self.assertIsNone(cmd.handle("tell me a joke"))


class DashboardSweepTests(unittest.TestCase):
    def test_new_views(self):
        from nomorals.console.dashboard import (
            render_adapters_view, render_health_view, render_summary,
            render_top_view,
        )

        snap = _snap()
        for fn in (render_adapters_view, render_health_view, render_top_view):
            out = fn(snap, color=False)
            self.assertTrue(out.strip(), fn.__name__)
        self.assertIn("adapters", render_adapters_view(snap, color=False).lower())

    def test_render_view_dispatch(self):
        from nomorals.console.dashboard import render_view

        snap = _snap()
        for view in ("status", "games", "jobs", "brain", "debug", "adapters",
                     "health", "top"):
            out = render_view(snap, view, color=False)
            self.assertTrue(out.strip(), view)

    def test_render_summary(self):
        from nomorals.console.dashboard import render_summary

        out = render_summary(_snap(), color=False)
        self.assertIn("devon", out.lower())
        self.assertIn("uptime", out)

    def test_dashboard_width_param(self):
        from nomorals.console.dashboard import render_dashboard

        out = render_dashboard(_snap(), color=False, width=100)
        self.assertIn("DEVON", out)

    def test_adapters_empty(self):
        from nomorals.console.dashboard import render_adapters_view

        out = render_adapters_view({}, color=False)
        self.assertIn("no adapters", out)


class DebugSweepTests(unittest.TestCase):
    def setUp(self):
        from nomorals.console.debug import DebugHub

        DebugHub.reset()

    def tearDown(self):
        from nomorals.console.debug import DebugHub

        DebugHub.reset()

    def _log(self, level, msg, name="test.mod", exc_info=None):
        record = logging.LogRecord(name, level, __file__, 1, msg, (), exc_info)
        from nomorals.console.debug import DebugHub

        DebugHub._ingest(record)

    def test_exception_capture(self):
        from nomorals.console.debug import DebugHub

        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            self._log(logging.ERROR, "it broke", exc_info=sys.exc_info())
        excs = DebugHub.exceptions(5)
        self.assertEqual(len(excs), 1)
        self.assertIn("ValueError", excs[0]["traceback"])
        self.assertIn("boom", excs[0]["traceback"])

    def test_record_exception_direct(self):
        from nomorals.console.debug import DebugHub

        DebugHub.record_exception(RuntimeError("rt"), logger_name="x")
        excs = DebugHub.exceptions(5)
        self.assertTrue(any("RuntimeError" in e["traceback"] for e in excs))

    def test_search(self):
        from nomorals.console.debug import DebugHub

        self._log(logging.INFO, "hello world marker-xyz")
        self._log(logging.INFO, "something else")
        hits = DebugHub.search("marker-xyz")
        self.assertEqual(len(hits), 1)
        self.assertEqual(DebugHub.search("([invalid"), [])

    def test_recent_filters(self):
        from nomorals.console.debug import DebugHub

        self._log(logging.ERROR, "bad thing", name="a.one")
        self._log(logging.INFO, "fine thing", name="b.two")
        self.assertEqual(len(DebugHub.recent(10, level="ERROR")), 1)
        self.assertEqual(len(DebugHub.recent(10, logger="b.two")), 1)
        self.assertEqual(len(DebugHub.recent(10, pattern="bad")), 1)

    def test_histogram(self):
        from nomorals.console.debug import DebugHub

        self._log(logging.ERROR, "e1")
        self._log(logging.WARNING, "w1")
        h = DebugHub.histogram(n=4, bucket_s=60)
        self.assertEqual(len(h["buckets"]), 4)
        self.assertEqual(sum(h["series"]["ERROR"]), 1)
        self.assertEqual(sum(h["series"]["WARNING"]), 1)

    def test_log_rate_and_errors_since(self):
        import time

        from nomorals.console.debug import DebugHub

        before = time.time()
        self._log(logging.ERROR, "e1")
        self._log(logging.INFO, "i1")
        self.assertGreaterEqual(DebugHub.errors_since(before - 1), 1)
        self.assertGreater(DebugHub.log_rate(60), 0)

    def test_stats_extended(self):
        from nomorals.console.debug import DebugHub

        s = DebugHub.stats()
        self.assertIn("exceptions", s)
        self.assertIn("log_rate_s", s)


class WidgetSweepTests(unittest.TestCase):
    def test_progress_bar_basic(self):
        import io

        from nomorals.console.widgets import ProgressBar

        out = io.StringIO()
        bar = ProgressBar("dl", total=10, out=out, color=False, min_interval=0)
        bar.update(5)
        bar.advance(5)
        bar.done()
        text = out.getvalue()
        self.assertIn("100.0%", text)
        self.assertIn("done", text)

    def test_progress_bar_postfix_and_format(self):
        import io

        from nomorals.console.widgets import ProgressBar

        out = io.StringIO()
        bar = ProgressBar("train", total=4, out=out, color=False,
                          min_interval=0, bar_format="{desc} {bar} {pct}{postfix}")
        bar.set_postfix(loss="0.02")
        bar.update(4)
        bar.done()
        self.assertIn("loss=0.02", out.getvalue())

    def test_progress_bar_indeterminate(self):
        import io

        from nomorals.console.widgets import ProgressBar

        out = io.StringIO()
        bar = ProgressBar("spin", total=None, out=out, color=False, min_interval=0)
        bar.update(3)
        bar.done("finished")
        self.assertIn("finished", out.getvalue())

    def test_progress_bar_track(self):
        import io

        from nomorals.console.widgets import ProgressBar

        out = io.StringIO()
        seen = [x for x in ProgressBar.track([1, 2, 3], "t", out=out,
                                             color=False, min_interval=0)]
        self.assertEqual(seen, [1, 2, 3])

    def test_progress_bar_context_manager(self):
        import io

        from nomorals.console.widgets import ProgressBar

        out = io.StringIO()
        with ProgressBar("cm", total=2, out=out, color=False,
                         min_interval=0) as bar:
            bar.update(2)
        self.assertIn("100.0%", out.getvalue())

    def test_progress_bar_unit_scale(self):
        import io

        from nomorals.console.widgets import ProgressBar, _fmt_num

        self.assertEqual(_fmt_num(1500), "1.5k")
        self.assertEqual(_fmt_num(2_500_000), "2.5M")
        out = io.StringIO()
        bar = ProgressBar("bytes", total=2000, out=out, color=False,
                          min_interval=0, unit="B", unit_scale=True)
        bar.update(2000)
        bar.done()
        self.assertIn("2.0kB", out.getvalue())

    def test_sparkline_options(self):
        from nomorals.console.widgets import sparkline

        s = sparkline([1, 2, 3, 4], min_label=True, color=False)
        self.assertIn("1/4", s)
        s2 = sparkline([1, 2, 3, 10], warn_at=5, crit_at=9, color=True)
        self.assertIn("\033[", s2)
        self.assertEqual(sparkline([], color=False), "─" * 24)
        # baseline: flat data renders the lowest block
        flat = sparkline([5, 5, 5], baseline=True, color=False, width=3)
        self.assertEqual(flat, "▁▁▁")

    def test_gauge(self):
        from nomorals.console.widgets import gauge

        g = gauge(0.5, color=False)
        self.assertIn("50.0%", g)
        g0 = gauge(0, color=False)
        self.assertIn("0.0%", g0)
        self.assertIn("100.0%", gauge(2.0, color=False))  # clamped
        # sub-precision: half of one cell renders a partial block
        g = gauge(0.5, width=1, color=False)
        self.assertIn("▌", g)

    def test_columns_chart(self):
        from nomorals.console.widgets import columns_chart

        rows = columns_chart([1, 2, 3, 4], height=4, color=False)
        self.assertEqual(len(rows), 4)
        self.assertIn("█", rows[0])
        rows = columns_chart([1, 2], height=2, color=False, labels=["a", "b"])
        self.assertEqual(len(rows), 3)
        self.assertIn("ab", rows[-1])

    def test_braille_chart(self):
        from nomorals.console.widgets import braille_chart

        rows = braille_chart([1, 2, 3, 2, 1], width=10, height=3, color=False)
        self.assertEqual(len(rows), 4)  # 3 rows + min/max label
        self.assertIn("min 1", rows[-1])
        # braille chars are in the U+2800 block
        self.assertTrue(any("\u2800" <= ch <= "\u28ff" for ch in rows[0]))
        self.assertEqual(braille_chart([], color=False), ["(no data)"])

    def test_heatmap(self):
        from nomorals.console.widgets import heatmap

        rows = heatmap([[1, 2, 3], [3, 2, 1]], color=False,
                       labels=["a", "b"])
        self.assertEqual(len(rows), 2)
        self.assertIn("a", rows[0])

    def test_table(self):
        from nomorals.console.widgets import table

        lines = table(["name", "n"], [["a", "1"], ["bb", "22"]], color=False)
        self.assertIn("╭", lines[0])
        self.assertIn("╯", lines[-1])
        self.assertIn("name", lines[1])
        self.assertIn("bb", lines[4])
        dbl = table(["x"], [["y"]], color=False, box="double")
        self.assertIn("╔", dbl[0])

    def test_panel(self):
        from nomorals.console.widgets import panel

        lines = panel(["hello", "world"], title="t", color=False)
        self.assertIn("t", lines[0])
        self.assertIn("hello", lines[1])
        self.assertIn("╰", lines[-1])

    def test_rule_caption(self):
        from nomorals.console.widgets import rule_caption

        r = rule_caption("lbl", width=20, color=False)
        self.assertIn("┤ lbl ├", r)
        self.assertEqual(len(r), 20)

    def test_spinner_context(self):
        import io

        from nomorals.console.widgets import Spinner, status

        out = io.StringIO()
        with Spinner("working", out=out, color=False):
            pass
        with status("s2", out=out, color=False):
            pass
        self.assertIn("working", out.getvalue())

    def test_message_feed_search(self):
        from nomorals.console.widgets import MessageEvent, MessageFeed

        feed = MessageFeed()
        feed.push(MessageEvent(platform="telegram", sender="amy",
                               text="hello needle here"))
        feed.push(MessageEvent(platform="discord", sender="bob", text="nope"))
        hits = feed.search("needle")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].sender, "amy")
        self.assertEqual(len(feed.by_platform("discord")), 1)
        self.assertEqual(feed.by_platform("telegram")[0].sender, "amy")
        feed.clear()
        self.assertEqual(len(feed), 0)
        self.assertEqual(feed.unread, 0)

    def test_watchhub_stats(self):
        from nomorals.console.widgets import WatchHub

        s = WatchHub.stats()
        self.assertIn("buffered", s)
        self.assertIn("platforms", s)

    def test_godscreen_command_mode(self):
        from nomorals.console.widgets import GodScreen

        screen = GodScreen(snapshot=lambda: {}, color=False, out=io.StringIO())
        self.assertTrue(screen._run_god_command("view games"))
        self.assertEqual(screen._view, "games")
        self.assertTrue(screen._run_god_command("view nope"))
        self.assertIn("unknown view", screen._notice)
        self.assertTrue(screen._run_god_command("interval 5"))
        self.assertEqual(screen.interval, 5.0)
        self.assertTrue(screen._run_god_command("theme ocean"))
        self.assertEqual(os.environ.get("NM_CONSOLE_THEME"), "ocean")
        self.assertTrue(screen._run_god_command("theme nope"))
        self.assertIn("unknown theme", screen._notice)
        self.assertTrue(screen._run_god_command("help"))
        self.assertTrue(screen._help)

    def test_godscreen_nav_keys(self):
        from nomorals.console.widgets import GodScreen

        screen = GodScreen(snapshot=lambda: {}, color=False, out=io.StringIO())
        self.assertTrue(screen._handle_nav_key("2"))
        self.assertEqual(screen._view, "games")
        self.assertFalse(screen._handle_nav_key("q"))
        screen2 = GodScreen(snapshot=lambda: {}, color=False, out=io.StringIO())
        self.assertTrue(screen2._handle_nav_key("?"))
        self.assertTrue(screen2._help)
        self.assertTrue(screen2._handle_nav_key(":"))
        self.assertEqual(screen2._cmd_buf, "")
        self.assertTrue(screen2._handle_nav_key("/"))
        self.assertEqual(screen2._filter_buf, "")
        screen3 = GodScreen(snapshot=lambda: {}, color=False, out=io.StringIO())
        screen3._handle_nav_key("+")
        self.assertGreater(screen3.interval, 2.0)
        screen3._handle_nav_key("j")
        self.assertEqual(screen3._feed_offset, 3)
        screen3._handle_nav_key("G")
        self.assertEqual(screen3._feed_offset, 0)

    def test_godscreen_theme_cycle(self):
        from nomorals.console.widgets import GodScreen

        screen = GodScreen(snapshot=lambda: {}, color=False, out=io.StringIO())
        before = os.environ.get("NM_CONSOLE_THEME", "ocean")
        screen._cycle_theme()
        after = os.environ.get("NM_CONSOLE_THEME")
        self.assertNotEqual(after, None)
        os.environ["NM_CONSOLE_THEME"] = before

    def test_godscreen_help_overlay_renders(self):
        from unittest.mock import patch

        from nomorals.console.palette import strip_ansi
        from nomorals.console.widgets import GodScreen

        out = io.StringIO()
        screen = GodScreen(snapshot=lambda: {"uptime_s": 5}, color=True, out=out)
        screen._help = True
        with patch("shutil.get_terminal_size", return_value=(100, 30)):
            screen._render_frame()
        frame = strip_ansi(out.getvalue())
        self.assertIn("keys", frame)
        self.assertIn(": opens command mode", frame)

    def test_godscreen_extra_views(self):
        from nomorals.console.widgets import WATCH_EXTRA_VIEWS, GodScreen

        self.assertIn("adapters", WATCH_EXTRA_VIEWS)
        screen = GodScreen(snapshot=lambda: {"uptime_s": 5}, color=False,
                           out=io.StringIO())
        self.assertTrue(screen._run_god_command("view adapters"))
        self.assertEqual(screen._view, "adapters")


class InputLineTests(unittest.TestCase):
    def test_typing_and_submit(self):
        from nomorals.console.godconsole import InputLine

        il = InputLine()
        for ch in "hello":
            self.assertIsNone(il.key(ch))
        self.assertEqual(il.buf, "hello")
        self.assertEqual(il.key("\r"), "submit")
        self.assertEqual(il.take_submit(), "hello")
        self.assertEqual(il.buf, "")

    def test_history(self):
        from nomorals.console.godconsole import InputLine

        il = InputLine()
        for cmd in ("one", "two"):
            for ch in cmd:
                il.key(ch)
            il.key("\r")
        il.key("\x1b[A")
        self.assertEqual(il.buf, "two")
        il.key("\x1b[A")
        self.assertEqual(il.buf, "one")
        il.key("\x1b[B")
        self.assertEqual(il.buf, "two")
        il.key("\x1b[B")
        self.assertEqual(il.buf, "")

    def test_emacs_keys(self):
        from nomorals.console.godconsole import InputLine

        il = InputLine()
        for ch in "hello world":
            il.key(ch)
        il.key("\x01")  # Ctrl-A
        self.assertEqual(il.cursor, 0)
        il.key("\x05")  # Ctrl-E
        self.assertEqual(il.cursor, 11)
        il.key("\x01")
        for _ in range(6):
            il.key("\x1b[C")  # right 6
        il.key("\x0b")  # Ctrl-K kills to end
        self.assertEqual(il.buf, "hello ")
        il.key("\x15")  # Ctrl-U kills line
        self.assertEqual(il.buf, "")
        for ch in "ab cd":
            il.key(ch)
        il.key("\x17")  # Ctrl-W deletes word
        self.assertEqual(il.buf, "ab ")

    def test_ctrl_c_clears_not_quits(self):
        from nomorals.console.godconsole import InputLine

        il = InputLine()
        for ch in "xyz":
            il.key(ch)
        self.assertIsNone(il.key("\x03"))
        self.assertEqual(il.buf, "")

    def test_arrows_home_end_delete(self):
        from nomorals.console.godconsole import InputLine

        il = InputLine()
        for ch in "abcd":
            il.key(ch)
        il.key("\x1b[D")
        il.key("\x1b[D")
        self.assertEqual(il.cursor, 2)
        il.key("\x1b[3~")  # Delete
        self.assertEqual(il.buf, "abd")
        il.key("\x1b[H")
        self.assertEqual(il.cursor, 0)
        il.key("\x1b[F")
        self.assertEqual(il.cursor, 3)

    def test_completion(self):
        from nomorals.console.godconsole import InputLine

        il = InputLine(completer=lambda p: ["dashboard", "dash", "debug"])
        for ch in "dash":
            il.key(ch)
        # multiple candidates → popup + first pick applied
        self.assertEqual(il.key("\t"), "complete")
        self.assertTrue(len(il.popup) > 1)
        il2 = InputLine(completer=lambda p: ["dashboard"])
        for ch in "dash":
            il2.key(ch)
        self.assertIsNone(il2.key("\t"))
        self.assertEqual(il2.buf, "dashboard")

    def test_bracketed_paste(self):
        from nomorals.console.godconsole import InputLine

        il = InputLine()
        token = "\x1b[200~line1\nline2\x1b[201~"
        self.assertIsNone(il.key(token))
        self.assertEqual(il.buf, "line1 line2")

    def test_esc_cancels(self):
        from nomorals.console.godconsole import InputLine

        il = InputLine()
        for ch in "abc":
            il.key(ch)
        self.assertEqual(il.key("\x1b"), "cancel")
        self.assertEqual(il.buf, "")

    def test_render_rows(self):
        from nomorals.console.godconsole import InputLine

        il = InputLine(completer=lambda p: ["a", "ab", "abc"])
        for ch in "a":
            il.key(ch)
        rows = il.render("❯ ", color=False, width=40)
        self.assertEqual(len(rows), 1)  # no popup yet
        il.key("\t")
        rows = il.render("❯ ", color=False, width=40)
        self.assertEqual(len(rows), 2)  # popup row
        self.assertIn("↳", rows[1])

    def test_history_file_roundtrip(self):
        import tempfile

        from nomorals.console.godconsole import InputLine

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "hist")
            il = InputLine(history_path=path)
            for ch in "cmd1":
                il.key(ch)
            il.key("\r")
            il.save_history()
            il2 = InputLine(history_path=path)
            self.assertIn("cmd1", il2.history)


class GodConsoleSweepTests(unittest.TestCase):
    def test_handle_key_view_switch_needs_empty_line(self):
        from nomorals.console.godconsole import GodConsole

        gc = GodConsole(snapshot=lambda: {}, on_command=lambda c: c)
        self.assertTrue(gc._handle_key("2"))
        self.assertEqual(gc._view, "games")
        gc._input.key("1")  # typing "1" as a command…
        self.assertTrue(gc._handle_key("2"))
        self.assertEqual(gc._view, "games")  # …must not switch views
        self.assertIn("1", gc._input.buf)

    def test_submit_shows_output(self):
        from nomorals.console.godconsole import GodConsole

        gc = GodConsole(snapshot=lambda: {}, on_command=lambda c: f"out:{c}")
        for ch in "status":
            gc._handle_key(ch)
        gc._handle_key("\r")
        self.assertTrue(gc._show_output)
        self.assertIn("out:status", gc._output_lines[-1])
        gc._handle_key("1")  # view key dismisses output — but input nonempty?
        # input was cleared on submit, so "1" switches view + hides output
        self.assertFalse(gc._show_output)

    def test_quit_on_empty_q(self):
        from nomorals.console.godconsole import GodConsole

        gc = GodConsole(snapshot=lambda: {}, on_command=lambda c: c)
        self.assertFalse(gc._handle_key("q"))
        gc2 = GodConsole(snapshot=lambda: {}, on_command=lambda c: c)
        gc2._input.key("q")
        self.assertTrue(gc2._handle_key("q"))  # typing q ≠ quitting

    def test_godconsole_wired_completion(self):
        from nomorals.console.commands import ConsoleCommands

        cmd = ConsoleCommands(lambda: _snap())
        from nomorals.console.godconsole import GodConsole

        gc = GodConsole(snapshot=lambda: {}, on_command=lambda c: c,
                        completer=cmd.highlight and __import__(
                            "nomorals.console.commands",
                            fromlist=["complete"]).complete,
                        highlighter=cmd.highlight)
        self.assertIsNotNone(gc._input.completer)
        self.assertIsNotNone(gc._input.highlighter)

    def test_render_frame_positions(self):
        from unittest.mock import patch

        from nomorals.console.godconsole import GodConsole
        from nomorals.console.palette import strip_ansi

        gc = GodConsole(snapshot=lambda: {"uptime_s": 60},
                        on_command=lambda c: c)
        gc._color = True
        gc._width, gc._height = 100, 30
        frame = strip_ansi(gc._render_frame())
        self.assertIn("DEVON", frame)
        self.assertIn("❯", frame)


if __name__ == "__main__":
    unittest.main()
