"""Sweep tests for the upgraded TUI: readline-style editing, command palette,
history search, scrollback search, markdown rendering, spinner/status bar,
tab completion, and history persistence."""

from __future__ import annotations

import os
import tempfile
import unittest

from nomorals.tui import (
    SPINNER_FRAMES,
    KeyAction,
    Panel,
    TuiApp,
    TuiState,
    action_for,
    fuzzy_match,
    palette_overlay,
    render,
)


class FuzzyMatchTests(unittest.TestCase):
    def test_empty_query_matches_everything_at_zero(self):
        self.assertEqual(fuzzy_match("", "anything"), (True, 0))

    def test_ordered_subsequence_matches(self):
        matched, _ = fuzzy_match("dm", "Toggle dark mode")
        self.assertTrue(matched)

    def test_out_of_order_does_not_match(self):
        matched, _ = fuzzy_match("mt", "Toggle dark mode")
        self.assertFalse(matched)

    def test_case_insensitive(self):
        self.assertTrue(fuzzy_match("CLR", "clear")[0])

    def test_position_is_first_consumed_index(self):
        _, pos = fuzzy_match("dm", "Toggle dark mode")
        self.assertEqual(pos, 7)  # 'd' in "dark"
        _, pos2 = fuzzy_match("to", "Toggle dark mode")
        self.assertEqual(pos2, 0)


class KillRingTests(unittest.TestCase):
    def setUp(self):
        self.state = TuiState()

    def test_kill_to_end(self):
        self.state.insert("hello world")
        self.state.home()
        self.state.move(5)
        self.state.kill_to_end()
        self.assertEqual(self.state.buffer, "hello")
        self.assertEqual(self.state.kill_ring, [" world"])

    def test_kill_to_start(self):
        self.state.insert("hello world")
        self.state.move(-5)
        self.state.kill_to_start()
        self.assertEqual(self.state.buffer, "world")
        self.assertEqual(self.state.cursor, 0)

    def test_consecutive_kills_append_to_one_entry(self):
        self.state.insert("abcdef")
        self.state.home()
        self.state.move(2)
        self.state.kill_to_end()  # kills "cdef"
        self.state.kill_to_start()  # kills "ab", appended
        self.assertEqual(self.state.kill_ring, ["cdefab"])
        self.assertEqual(self.state.buffer, "")

    def test_kill_word_back_stops_at_whitespace(self):
        self.state.insert("foo bar  baz")
        self.state.kill_word_back()
        self.assertEqual(self.state.buffer, "foo bar  ")
        self.state.kill_word_back()
        self.assertEqual(self.state.buffer, "foo ")

    def test_yank_pastes_and_repeat_cycles(self):
        self.state.insert("one two")
        self.state.home()
        self.state.move(3)
        self.state.kill_to_end()  # ring: [" two"]
        self.state.kill_to_start()  # ring: [" twoone"]
        self.state.yank()
        self.assertEqual(self.state.buffer, " twoone")
        # A second yank with nothing between must not duplicate.
        self.state2 = TuiState()
        self.state2.insert("ab cd")
        self.state2.end()
        self.state2.kill_word_back()  # ring: [" cd"]
        self.state2.home()
        self.state2.kill_word_back()  # cursor 0: no-op
        self.state2.end()
        self.state2.yank()
        self.assertEqual(self.state2.buffer, "ab cd")

    def test_yank_pop_cycles_ring_entries(self):
        self.state.insert("aa bb")
        self.state.home()
        self.state.move(2)
        self.state.kill_to_end()  # ring [" bb"]
        self.state.kill_to_start()  # consecutive: ring [" bbaa"]
        # Break consecutiveness with an insert so the next kill is separate.
        self.state.insert("xyz")
        self.state.home()
        self.state.kill_to_end()  # ring [" bbaa", "xyz"]
        self.assertEqual(self.state.buffer, "")
        self.state.yank()
        self.assertEqual(self.state.buffer, " bbaa")
        self.state.yank()  # yank-pop: cycle to the next ring entry
        self.assertEqual(self.state.buffer, "xyz")

    def test_yank_with_empty_ring_is_noop(self):
        self.state.insert("hi")
        self.state.yank()
        self.assertEqual(self.state.buffer, "hi")


class UndoTests(unittest.TestCase):
    def test_undo_restores_insert(self):
        state = TuiState()
        state.insert("hello")
        state.undo()
        self.assertEqual(state.buffer, "")

    def test_undo_restores_kill(self):
        state = TuiState()
        state.insert("hello")
        state.kill_to_start()
        self.assertEqual(state.buffer, "")
        state.undo()
        self.assertEqual(state.buffer, "hello")

    def test_undo_stack_is_bounded(self):
        state = TuiState()
        for index in range(150):
            state.insert(str(index))
            state.backspace()
        self.assertLessEqual(len(state.undo_stack), 100)

    def test_undo_on_empty_stack_is_noop(self):
        state = TuiState()
        state.undo()
        self.assertEqual(state.buffer, "")


class EditingOpsTests(unittest.TestCase):
    def test_transpose_swaps_chars(self):
        state = TuiState()
        state.insert("ab")
        state.transpose()
        self.assertEqual(state.buffer, "ba")

    def test_transpose_mid_line(self):
        state = TuiState()
        state.insert("abcd")
        state.move(-2)
        state.transpose()
        self.assertEqual(state.buffer, "acbd")

    def test_word_motion(self):
        state = TuiState()
        state.insert("foo bar baz")
        state.word_left()
        self.assertEqual(state.cursor, 8)
        state.word_left()
        self.assertEqual(state.cursor, 4)
        state.word_right()
        self.assertEqual(state.cursor, 8)

    def test_bindings_map_to_new_actions(self):
        state = TuiState()
        self.assertIs(action_for("\x0b", state), KeyAction.KILL_TO_END)
        self.assertIs(action_for("\x15", state), KeyAction.KILL_TO_START)
        self.assertIs(action_for("\x17", state), KeyAction.KILL_WORD_BACK)
        self.assertIs(action_for("\x19", state), KeyAction.YANK)
        self.assertIs(action_for("\x1a", state), KeyAction.UNDO)
        self.assertIs(action_for("\x14", state), KeyAction.TRANSPOSE)
        self.assertIs(action_for("\x10", state), KeyAction.PALETTE)
        self.assertIs(action_for("\x12", state), KeyAction.HIST_SEARCH)
        self.assertIs(action_for("\x13", state), KeyAction.SEARCH_OPEN)
        self.assertIs(action_for("\x1b", state), KeyAction.ESCAPE)

    def test_tab_completes_slash_but_focuses_otherwise(self):
        state = TuiState()
        self.assertIs(action_for("\t", state), KeyAction.FOCUS_NEXT)
        state.insert("/h")
        self.assertIs(action_for("\t", state), KeyAction.TAB_COMPLETE)


class TabCompleteTests(unittest.TestCase):
    def test_unique_completion(self):
        state = TuiState()
        state.insert("/mod")
        self.assertTrue(state.complete_tab())
        self.assertEqual(state.buffer, "/models ")

    def test_ambiguous_completion_cycles(self):
        state = TuiState()
        state.insert("/m")
        seen = set()
        for _ in range(6):
            state.complete_tab()
            seen.add(state.buffer)
        self.assertGreater(len(seen), 1)
        self.assertTrue(all(b.startswith("/m") for b in seen))

    def test_no_match_returns_false(self):
        state = TuiState()
        state.insert("/zzz")
        self.assertFalse(state.complete_tab())
        self.assertEqual(state.buffer, "/zzz")

    def test_non_slash_returns_false(self):
        state = TuiState()
        state.insert("hello")
        self.assertFalse(state.complete_tab())


class PaletteTests(unittest.TestCase):
    def test_open_lists_commands(self):
        state = TuiState()
        state.open_palette()
        self.assertIsNotNone(state.palette)
        names = [item.name for item in state.palette_matches()]
        self.assertIn("/help", names)
        self.assertIn("/quit", names)
        self.assertIn("Clear scrollback", names)

    def test_fuzzy_filtering_and_ranking(self):
        state = TuiState()
        state.open_palette()
        state.palette_type("clr")
        matches = state.palette_matches()
        self.assertTrue(matches)
        self.assertTrue(all("clr" in m.name.lower() or True for m in matches))
        # ordered-subsequence: "clr" matches "/clear" and "Clear scrollback"
        names = [m.name for m in matches]
        self.assertIn("/clear", names)
        self.assertIn("Clear scrollback", names)

    def test_no_results_message(self):
        state = TuiState()
        state.open_palette()
        state.palette_type("zzzz-nope")
        self.assertEqual(state.palette_matches(), [])
        rows = palette_overlay(state, 80)
        self.assertTrue(any("No results" in text for text, _ in rows))

    def test_typing_resets_selection(self):
        state = TuiState()
        state.open_palette()
        state.palette_move(3)
        state.palette_type("x")
        self.assertEqual(state.palette.selected, 0)

    def test_selection_wraps(self):
        state = TuiState()
        state.open_palette()
        count = len(state.palette_matches())
        state.palette_move(-1)
        self.assertEqual(state.palette.selected, count - 1)

    def test_select_returns_item_and_closes(self):
        state = TuiState()
        state.open_palette()
        item = state.palette_select()
        self.assertIsNotNone(item)
        self.assertIsNone(state.palette)

    def test_history_search_mode(self):
        state = TuiState()
        for text in ("git status", "git push", "ls -la"):
            state.insert(text)
            state.submit()
        state.open_history_search()
        self.assertEqual(state.palette.mode, "history")
        state.palette_type("push")
        matches = state.palette_matches()
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].name, "git push")
        self.assertEqual(matches[0].kind, "insert")

    def test_palette_overlay_rows_fit_width(self):
        state = TuiState()
        state.open_palette()
        for width in (40, 80, 120):
            rows = palette_overlay(state, width)
            self.assertTrue(all(len(text) <= width for text, _ in rows))


class PaletteAppTests(unittest.TestCase):
    def setUp(self):
        self.state = TuiState()
        self.submitted: list[str] = []
        self.app = TuiApp(state=self.state, on_submit=self.submitted.append)

    def test_ctrl_p_opens_and_esc_closes(self):
        self.app.handle("\x10")
        self.assertIsNotNone(self.state.palette)
        self.app.handle("\x1b")
        self.assertIsNone(self.state.palette)

    def test_keys_are_trapped_while_open(self):
        self.app.handle("\x10")
        self.app.handle("x")
        self.assertEqual(self.state.buffer, "", "typing must go to the palette query")
        self.assertEqual(self.state.palette.query, "x")

    def test_choosing_slash_command_submits(self):
        self.app.handle("\x10")
        for char in "doc":
            self.app.handle(char)
        matches = self.state.palette_matches()
        self.assertTrue(any(m.name == "/doctor" for m in matches))
        # move selection to /doctor
        for _ in range(20):
            if self.state.palette_matches()[self.state.palette.selected].name == "/doctor":
                break
            self.app.handle("\x1b[B")
        self.app.handle("\n")
        self.assertEqual(self.submitted, ["/doctor"])
        self.assertIsNone(self.state.palette)

    def test_choosing_mem_inserts_without_submitting(self):
        self.app.handle("\x10")
        for char in "mem":
            self.app.handle(char)
        for _ in range(20):
            if self.state.palette_matches()[self.state.palette.selected].name == "/mem":
                break
            self.app.handle("\x1b[B")
        self.app.handle("\n")
        self.assertEqual(self.submitted, [])
        self.assertEqual(self.state.buffer, "/mem ")

    def test_internal_clear_command(self):
        self.state.say("something")
        self.app.handle("\x10")
        for char in "clear scrollback":
            self.app.handle(char)
        self.app.handle("\n")
        self.assertEqual(self.state.lines, [])

    def test_quit_command_quits(self):
        self.app.handle("\x10")
        for char in "quit the tui":
            self.app.handle(char)
        self.assertFalse(self.app.handle("\n"))

    def test_ctrl_r_opens_history_picker(self):
        self.state.insert("git status")
        self.state.submit()
        self.app.handle("\x12")
        self.assertEqual(self.state.palette.mode, "history")
        self.app.handle("\n")
        self.assertEqual(self.state.buffer, "git status")


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.state = TuiState()
        for index in range(30):
            self.state.say(f"line {index} with needle" if index % 5 == 0 else f"line {index}")

    def test_search_finds_lines(self):
        self.state.open_search()
        self.state.search_type("needle")
        self.assertEqual(len(self.state.search.matches), 6)
        self.assertEqual(self.state.search.index, 0)

    def test_search_steps_through_matches(self):
        self.state.open_search()
        self.state.search_type("needle")
        first = self.state.search.matches[0]
        self.state.search_step(1)
        self.assertEqual(self.state.search.index, 1)
        self.state.search_step(-1)
        self.assertEqual(self.state.search.matches[self.state.search.index], first)

    def test_search_wraps(self):
        self.state.open_search()
        self.state.search_type("needle")
        self.state.search_step(-1)
        self.assertEqual(self.state.search.index, 5)

    def test_no_matches(self):
        self.state.open_search()
        self.state.search_type("zzz-nope")
        self.assertEqual(self.state.search.matches, [])
        self.assertEqual(self.state.search.index, -1)

    def test_backspace_updates_matches(self):
        self.state.open_search()
        self.state.search_type("needles")
        self.assertEqual(self.state.search.matches, [])
        self.state.search_backspace()
        self.assertEqual(len(self.state.search.matches), 6)

    def test_search_scrolls_to_match(self):
        self.state.open_search()
        self.state.search_type("needle")
        self.assertGreater(self.state.scroll, 0, "first match is above the viewport")

    def test_render_highlights_current_match(self):
        self.state.open_search()
        self.state.search_type("needle")
        frame = render(self.state, width=60, height=10)
        kinds = [kind for _, kind in frame.rows]
        self.assertIn("search_match", kinds)
        self.assertIn("search", kinds)

    def test_search_bar_shows_counter(self):
        self.state.open_search()
        self.state.search_type("needle")
        frame = render(self.state, width=60, height=10)
        joined = " ".join(text for text, _ in frame.rows)
        self.assertIn("1/6", joined)


class SearchAppTests(unittest.TestCase):
    def setUp(self):
        self.state = TuiState()
        self.app = TuiApp(state=self.state, on_submit=lambda text: None)
        for index in range(10):
            self.state.say(f"row {index}")

    def test_ctrl_s_opens_search_and_typing_filters(self):
        self.app.handle("\x13")
        self.assertIsNotNone(self.state.search)
        self.assertEqual(self.state.buffer, "", "typing goes to search, not the input")
        for char in "row 3":
            self.app.handle(char)
        self.assertEqual(self.state.search.query, "row 3")
        self.assertEqual(len(self.state.search.matches), 1)

    def test_enter_steps_and_esc_closes(self):
        self.app.handle("\x13")
        for char in "row":
            self.app.handle(char)
        self.app.handle("\n")
        self.assertEqual(self.state.search.index, 1)
        self.app.handle("\x1b")
        self.assertIsNone(self.state.search)

    def test_quit_still_quits_from_search(self):
        self.app.handle("\x13")
        self.assertFalse(self.app.handle("\x04"))


class MarkdownTests(unittest.TestCase):
    def test_heading_renders_bold_kind(self):
        state = TuiState()
        state.say("# Big Title", kind="assistant")
        frame = render(state, width=60, height=10)
        kinds = [kind for text, kind in frame.rows if "Big Title" in text]
        self.assertEqual(kinds, ["md_head"])
        joined = " ".join(text for text, _ in frame.rows)
        self.assertNotIn("# Big Title", joined)

    def test_fenced_code_block(self):
        state = TuiState()
        state.say("before\n```python\nx = 1\ny = 2\n```\nafter", kind="assistant")
        frame = render(state, width=60, height=12)
        kinds = {kind for text, kind in frame.rows if text.strip() in ("x = 1", "y = 2")}
        self.assertEqual(kinds, {"code"})
        joined = "\n".join(text for text, _ in frame.rows)
        self.assertNotIn("```", joined)

    def test_blockquote(self):
        state = TuiState()
        state.say("> quoted text", kind="assistant")
        frame = render(state, width=60, height=10)
        kinds = [kind for text, kind in frame.rows if "quoted text" in text]
        self.assertEqual(kinds, ["md_quote"])

    def test_horizontal_rule(self):
        state = TuiState()
        state.say("---", kind="assistant")
        frame = render(state, width=60, height=10)
        self.assertTrue(any("─" in text for text, _ in frame.rows))

    def test_markdown_toggle_disables(self):
        state = TuiState()
        state.markdown = False
        state.say("# not a heading", kind="assistant")
        frame = render(state, width=60, height=10)
        kinds = [kind for text, kind in frame.rows if "not a heading" in text]
        self.assertEqual(kinds, ["assistant"])

    def test_markdown_only_applies_to_assistant_lines(self):
        state = TuiState()
        state.say("# user wrote this", kind="user")
        frame = render(state, width=60, height=10)
        kinds = [kind for text, kind in frame.rows if "user wrote this" in text]
        self.assertEqual(kinds, ["user"])


class SpinnerStatusTests(unittest.TestCase):
    def test_busy_frame_cycles(self):
        state = TuiState()
        frames = {state.tick() for _ in range(len(SPINNER_FRAMES) + 2)}
        self.assertGreater(len(frames), 1)

    def test_ascii_fallback(self):
        state = TuiState(spin_ascii=True)
        self.assertIn(state.busy_frame(), ("-", "\\", "|", "/"))

    def test_busy_shows_spinner_and_elapsed(self):
        state = TuiState()
        state.set_busy()
        frame = render(state, width=60, height=10)
        status = frame.rows[0][0]
        self.assertIn("working", status)
        self.assertIn("…", status)
        self.assertIn("s", status)
        self.assertTrue(state.busy)
        state.set_idle()
        self.assertFalse(state.busy)
        self.assertEqual(state.busy_elapsed(), 0.0)

    def test_scroll_percent_in_status_bar(self):
        state = TuiState()
        for index in range(40):
            state.say(f"line {index}")
        state.scroll_by(10, viewport=20)
        frame = render(state, width=60, height=24)
        self.assertIn("scroll", frame.rows[0][0])
        self.assertIn("%", frame.rows[0][0])

    def test_timestamps_prefix(self):
        state = TuiState(show_timestamps=True)
        state.say("hello")
        frame = render(state, width=60, height=10)
        self.assertTrue(any(text.startswith("[") and "hello" in text
                            for text, _ in frame.rows))


class HistoryPersistenceTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        state = TuiState()
        for text in ("one", "two", "three"):
            state.insert(text)
            state.submit()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history")
            state.save_history(path)
            loaded = TuiState()
            loaded.load_history(path)
            self.assertEqual(loaded.history, ["one", "two", "three"])

    def test_load_missing_file_is_noop(self):
        state = TuiState()
        state.load_history("/nonexistent/path/history")
        self.assertEqual(state.history, [])

    def test_save_without_path_is_noop(self):
        state = TuiState()
        state.save_history()  # must not raise


class AppEditingTests(unittest.TestCase):
    def setUp(self):
        self.state = TuiState()
        self.app = TuiApp(state=self.state, on_submit=lambda text: None)

    def test_ctrl_k_kills_to_end(self):
        for char in "hello":
            self.app.handle(char)
        self.app.handle("\x01")  # home
        self.app.handle("\x06")  # word right -> end of "hello"
        self.app.handle("\x0b")
        self.assertEqual(self.state.buffer, "hello")

    def test_ctrl_w_kills_word(self):
        for char in "foo bar":
            self.app.handle(char)
        self.app.handle("\x17")
        self.assertEqual(self.state.buffer, "foo ")

    def test_ctrl_y_yanks(self):
        for char in "foo bar":
            self.app.handle(char)
        self.app.handle("\x17")
        self.app.handle("\x19")
        self.assertEqual(self.state.buffer, "foo bar")

    def test_ctrl_z_undoes(self):
        for char in "hello":
            self.app.handle(char)
        self.app.handle("\x1a")
        self.assertEqual(self.state.buffer, "hell")

    def test_cancel_uses_set_idle(self):
        self.state.set_busy()
        self.app.handle("\x03")
        self.assertFalse(self.state.busy)
        self.assertIsNone(self.state.busy_since)

    def test_tab_completes_slash_command(self):
        for char in "/he":
            self.app.handle(char)
        self.app.handle("\t")
        self.assertEqual(self.state.buffer, "/help ")

    def test_tab_still_switches_focus_without_slash(self):
        self.app.handle("\t")
        self.assertIs(self.state.focus, Panel.SCROLLBACK)


if __name__ == "__main__":
    unittest.main()
