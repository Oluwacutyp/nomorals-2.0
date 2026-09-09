"""L7 — TUI: state, layout, and key handling.

The curses driver is thin and untested on purpose; everything with logic in it
lives in model.py and is exercised here, without a terminal.
"""

from __future__ import annotations

import unittest

from nomorals.tui import KeyAction, Line, Panel, TuiApp, TuiState, action_for, render


class LineTests(unittest.TestCase):
    def test_short_text_is_one_row(self):
        self.assertEqual(Line("hello").render(40), ["hello"])

    def test_long_text_wraps_at_a_word_boundary(self):
        rows = Line("alpha beta gamma delta").render(12)
        self.assertTrue(all(len(r) <= 12 for r in rows))
        self.assertEqual("".join(r.replace(" ", "") for r in rows), "alphabetagammadelta")

    def test_a_word_longer_than_the_width_is_force_split(self):
        rows = Line("aaaaaaaaaaaaaaaaaaaa").render(5)
        self.assertTrue(all(len(r) <= 5 for r in rows))
        self.assertEqual("".join(rows), "aaaaaaaaaaaaaaaaaaaa")

    def test_newlines_become_separate_rows(self):
        self.assertEqual(Line("one\ntwo").render(40), ["one", "two"])

    def test_empty_text_is_one_blank_row(self):
        self.assertEqual(Line("").render(40), [""])


class EditingTests(unittest.TestCase):
    def setUp(self):
        self.state = TuiState()

    def test_insert_appends_at_the_cursor(self):
        self.state.insert("abc")
        self.assertEqual(self.state.buffer, "abc")
        self.assertEqual(self.state.cursor, 3)

    def test_insert_in_the_middle_does_not_clobber(self):
        self.state.insert("ac")
        self.state.move(-1)
        self.state.insert("b")
        self.assertEqual(self.state.buffer, "abc")
        self.assertEqual(self.state.cursor, 2)

    def test_cursor_cannot_leave_the_buffer(self):
        self.state.insert("ab")
        self.state.move(-99)
        self.assertEqual(self.state.cursor, 0)
        self.state.move(99)
        self.assertEqual(self.state.cursor, 2)

    def test_backspace_deletes_before_the_cursor(self):
        self.state.insert("abc")
        self.state.backspace()
        self.assertEqual(self.state.buffer, "ab")
        self.state.home()
        self.state.backspace()
        self.assertEqual(self.state.buffer, "ab", "backspace at position 0 must be a no-op")

    def test_delete_removes_after_the_cursor(self):
        self.state.insert("abc")
        self.state.home()
        self.state.delete_char()
        self.assertEqual(self.state.buffer, "bc")

    def test_home_and_end(self):
        self.state.insert("abcd")
        self.state.home()
        self.assertEqual(self.state.cursor, 0)
        self.state.end()
        self.assertEqual(self.state.cursor, 4)

    def test_empty_insert_is_a_noop(self):
        self.state.insert("")
        self.assertEqual(self.state.buffer, "")


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.state = TuiState()

    def test_submit_records_history_and_clears(self):
        self.state.insert("first")
        self.assertEqual(self.state.submit(), "first")
        self.assertEqual(self.state.buffer, "")
        self.assertEqual(self.state.history, ["first"])

    def test_whitespace_only_input_is_not_recorded(self):
        self.state.insert("   ")
        self.assertEqual(self.state.submit(), "")
        self.assertEqual(self.state.history, [])

    def test_consecutive_duplicates_are_collapsed(self):
        for _ in range(3):
            self.state.insert("same")
            self.state.submit()
        self.assertEqual(self.state.history, ["same"])

    def test_history_navigation_walks_backwards_then_forwards(self):
        for text in ("one", "two", "three"):
            self.state.insert(text)
            self.state.submit()
        self.state.history_prev()
        self.assertEqual(self.state.buffer, "three")
        self.state.history_prev()
        self.assertEqual(self.state.buffer, "two")
        self.state.history_next()
        self.assertEqual(self.state.buffer, "three")
        self.state.history_next()
        self.assertEqual(self.state.buffer, "", "stepping past the newest clears the buffer")

    def test_history_prev_on_empty_history_is_a_noop(self):
        self.state.history_prev()
        self.assertEqual(self.state.buffer, "")

    def test_history_is_bounded(self):
        self.state.max_history = 3
        for index in range(6):
            self.state.insert(f"c{index}")
            self.state.submit()
        self.assertEqual(len(self.state.history), 3)
        self.assertEqual(self.state.history[-1], "c5")


class ScrollTests(unittest.TestCase):
    def setUp(self):
        self.state = TuiState()
        for index in range(40):
            self.state.say(f"line {index}")

    def test_new_output_resets_the_scroll_to_the_bottom(self):
        self.state.scroll_by(10, viewport=20)
        self.state.say("fresh")
        self.assertEqual(self.state.scroll, 0)

    def test_scroll_never_goes_negative(self):
        self.state.scroll_by(-100, viewport=20)
        self.assertEqual(self.state.scroll, 0)

    def test_scroll_is_capped_at_the_content_height(self):
        self.state.scroll_by(10_000, viewport=20)
        self.assertLessEqual(self.state.scroll, 40)

    def test_scroll_top_and_bottom(self):
        self.state.scroll_top(viewport=20)
        self.assertGreater(self.state.scroll, 0)
        self.state.scroll_bottom()
        self.assertEqual(self.state.scroll, 0)

    def test_lines_are_bounded(self):
        self.state.max_lines = 5
        for index in range(20):
            self.state.say(f"x{index}")
        self.assertEqual(len(self.state.lines), 5)
        self.assertEqual(self.state.lines[-1].text, "x19")


class FocusAndBindingTests(unittest.TestCase):
    def test_focus_cycles_between_input_and_scrollback(self):
        state = TuiState()
        self.assertIs(state.focus, Panel.INPUT)
        state.cycle_focus()
        self.assertIs(state.focus, Panel.SCROLLBACK)
        state.cycle_focus()
        self.assertIs(state.focus, Panel.INPUT)

    def test_the_same_key_means_different_things_per_focus(self):
        state = TuiState()
        self.assertIs(action_for("j", state), KeyAction.NONE)
        state.focus = Panel.SCROLLBACK
        self.assertIs(action_for("j", state), KeyAction.SCROLL_DOWN)

    def test_control_bindings(self):
        state = TuiState()
        self.assertIs(action_for("\x04", state), KeyAction.QUIT)
        self.assertIs(action_for("\n", state), KeyAction.SUBMIT)
        self.assertIs(action_for("\x0c", state), KeyAction.CLEAR)
        self.assertIs(action_for("\t", state), KeyAction.FOCUS_NEXT)

    def test_unknown_keys_do_nothing(self):
        self.assertIs(action_for("\x99", TuiState()), KeyAction.NONE)

    def test_custom_bindings_override_the_defaults(self):
        state = TuiState()
        custom = {"q": KeyAction.QUIT}
        self.assertIs(action_for("q", state, custom), KeyAction.QUIT)
        self.assertIs(action_for("\x04", state, custom), KeyAction.NONE)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.state = TuiState()

    def test_layout_is_status_then_body_then_input(self):
        self.state.say("content")
        frame = render(self.state, width=60, height=10)
        self.assertEqual(len(frame.rows), 10)
        self.assertEqual(frame.rows[0][1], "status")
        self.assertEqual(frame.rows[-1][1], "input")

    def test_the_input_row_shows_the_prompt_and_buffer(self):
        self.state.insert("hello")
        frame = render(self.state, width=60, height=10)
        self.assertIn("hello", frame.rows[-1][0])
        self.assertTrue(frame.rows[-1][0].startswith(self.state.prompt))

    def test_cursor_column_tracks_the_prompt_plus_cursor(self):
        self.state.insert("abc")
        frame = render(self.state, width=60, height=10)
        self.assertEqual(frame.cursor_col, len(self.state.prompt) + 3)
        self.assertEqual(frame.cursor_row, len(frame.rows) - 1)

    def test_a_busy_state_shows_an_indicator(self):
        self.state.busy = True
        frame = render(self.state, width=60, height=10)
        self.assertIn("…", frame.rows[0][0])

    def test_focus_is_shown_in_the_status_bar(self):
        self.state.focus = Panel.SCROLLBACK
        frame = render(self.state, width=60, height=10)
        self.assertIn("scrollback", frame.rows[0][0])

    def test_a_too_small_terminal_is_reported_not_crashed(self):
        frame = render(self.state, width=10, height=3)
        self.assertEqual(frame.rows[0][0], "terminal too small")

    def test_no_row_exceeds_the_width(self):
        self.state.say("x" * 200)
        frame = render(self.state, width=40, height=12)
        self.assertTrue(all(len(text) <= 40 for text, _ in frame.rows))

    def test_rendering_is_deterministic(self):
        self.state.say("stable")
        first = render(self.state, width=60, height=10).rows
        second = render(self.state, width=60, height=10).rows
        self.assertEqual(first, second)

    def test_scrolling_reveals_earlier_content(self):
        for index in range(30):
            self.state.say(f"row-{index}")
        bottom = render(self.state, width=60, height=10).rows
        self.state.scroll_top(viewport=8)
        top = render(self.state, width=60, height=10).rows
        self.assertIn("row-0", "".join(t for t, _ in top))
        self.assertNotIn("row-0", "".join(t for t, _ in bottom))


class AppTests(unittest.TestCase):
    """The driver's key handling, driven without a terminal."""

    def setUp(self):
        self.state = TuiState()
        self.submitted: list[str] = []
        self.app = TuiApp(state=self.state, on_submit=self.submitted.append)

    def test_printable_characters_reach_the_buffer(self):
        for char in "hello":
            self.app.handle(char)
        self.assertEqual(self.state.buffer, "hello")

    def test_submit_invokes_the_handler_and_echoes(self):
        self.app.handle("h")
        self.app.handle("i")
        self.assertTrue(self.app.handle("\n"))
        self.assertEqual(self.submitted, ["hi"])
        self.assertEqual(self.state.buffer, "")
        self.assertTrue(any("hi" in line.text for line in self.state.lines))

    def test_quitting_returns_false(self):
        self.assertFalse(self.app.handle("\x04"))

    def test_a_raising_handler_is_reported_not_fatal(self):
        def boom(text: str) -> None:
            raise RuntimeError("handler exploded")

        app = TuiApp(state=self.state, on_submit=boom)
        app.handle("x")
        self.assertTrue(app.handle("\n"), "the app must survive a failing handler")
        self.assertFalse(self.state.busy, "busy must be cleared even on failure")
        self.assertTrue(any(line.kind == "error" for line in self.state.lines))

    def test_cancel_clears_the_busy_flag(self):
        self.state.busy = True
        self.app.handle("\x03")
        self.assertFalse(self.state.busy)

    def test_clear_empties_the_scrollback(self):
        self.state.say("gone")
        self.app.handle("\x0c")
        self.assertEqual(self.state.lines, [])

    def test_typing_while_the_scrollback_is_focused_does_not_edit_the_buffer(self):
        self.state.focus = Panel.SCROLLBACK
        self.app.handle("j")
        self.assertEqual(self.state.buffer, "")

    def test_backspace_key_edits_the_buffer(self):
        self.app.handle("a")
        self.app.handle("b")
        self.app.handle("\x7f")
        self.assertEqual(self.state.buffer, "a")

    def test_tab_switches_focus(self):
        self.app.handle("\t")
        self.assertIs(self.state.focus, Panel.SCROLLBACK)


if __name__ == "__main__":
    unittest.main()
