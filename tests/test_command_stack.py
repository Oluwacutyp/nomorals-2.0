"""Command stacking: one message, one command per line, run in order.

Syntax (the ONE stacking syntax): newline-separated slash commands.
``;`` and ``&&`` are NOT separators — they are live syntax inside
/route, /overview (;;), /exec, /py, URLs, and quoted args, so splitting
on them would corrupt single commands. Newlines + the every-line-must-
parse rule are the only split that cannot misfire on those.

* A message is a stack only when EVERY non-blank line parses as a known
  command (``parse_control`` clean — no error kind). Anything else falls
  back to the existing single-command path, untouched.
* Newlines inside fenced code blocks (``` / ~~~, opened mid-line or at
  line start), double-quoted strings, single-quoted strings (boundary
  heuristic, so don't/it\'s apostrophes don\'t open quotes), and inline
  `code` spans do NOT split.
* ``{prev}`` in a later step expands to the previous step\'s output
  (truncated); ``{prev:N}`` to step N\'s output; ``{{prev}}`` is literal.
  In step 1, ``{prev}`` stays literal (there is no previous output).
* Fail-stop: a step that raises, is denied, or asks for confirmation
  stops the stack with an explicit marker; later steps do not run.
* Pre-validation: gating (per-command owner grant, SMS surface,
  progression lock) and the non-stackable list are checked for EVERY
  step BEFORE anything runs — a denied/unstakable step means the whole
  stack reports it and nothing runs.
* Non-stackable: games (interactive, stateful), confirmable + lifecycle
  commands (quit/forget/restart/deploy/update/pull — confirmation and
  shutdown/redeploy must stay one-message-per-command, never
  auto-confirmed by a stack).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.control import (
    MAX_STACK_DEPTH,
    NON_STACKABLE_COMMANDS,
    expand_prev_placeholders,
    split_command_stack,
    detailed_help,
    help_text,
)
from nomorals.storage.db import Database

OWNER_CHAT = SimpleNamespace(key="telegram:1")
STRANGER_CHAT = SimpleNamespace(key="telegram:999")
CHAT = "telegram:1"


def _bare_runtime():
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(db=Database(":memory:"))
    rt._owner_chats = {"telegram:1"}
    return rt


def _msg(chat, text):
    return SimpleNamespace(chat=chat, text=text, meta={})


class SplitTests(unittest.TestCase):
    def test_single_command_is_not_a_stack(self):
        self.assertIsNone(split_command_stack("/status"))
        self.assertIsNone(split_command_stack("/status extra words here"))

    def test_plain_text_is_not_a_stack(self):
        self.assertIsNone(split_command_stack("hello"))
        self.assertIsNone(split_command_stack("hello\n/status"))
        self.assertIsNone(split_command_stack(""))

    def test_two_line_stack(self):
        self.assertEqual(split_command_stack("/status\n/mood"),
                         ["/status", "/mood"])

    def test_three_line_stack_with_args(self):
        self.assertEqual(
            split_command_stack("/mood happy\n/status\n/providers"),
            ["/mood happy", "/status", "/providers"])

    def test_blank_lines_ignored(self):
        self.assertEqual(
            split_command_stack("/status\n\n/mood\n\n"),
            ["/status", "/mood"])

    def test_crlf(self):
        self.assertEqual(split_command_stack("/status\r\n/mood"),
                         ["/status", "/mood"])

    def test_semicolon_is_not_a_separator(self):
        self.assertIsNone(split_command_stack("/status; /mood"))
        self.assertIsNone(split_command_stack("/say a; b"))

    def test_ampersand_is_not_a_separator(self):
        self.assertIsNone(split_command_stack("/status && /mood"))
        self.assertIsNone(split_command_stack("/exec echo a && echo b"))

    def test_invalid_segment_blocks_stacking(self):
        # /status takes 0 args: the whole message falls back to the
        # single-command path, which reports the usage error.
        self.assertIsNone(split_command_stack("/status extra\n/mood"))

    def test_unknown_command_blocks_stacking(self):
        self.assertIsNone(split_command_stack("/status\n/nope"))

    def test_non_command_line_blocks_stacking(self):
        self.assertIsNone(split_command_stack("/status\nhello there"))

    def test_url_with_semicolons_and_ampersands_kept(self):
        got = split_command_stack(
            "/download https://example.com/a?x=1&y=2;z=3\n/status")
        self.assertEqual(
            got, ["/download https://example.com/a?x=1&y=2;z=3", "/status"])

    def test_fenced_block_swallows_command_like_lines(self):
        text = "/status\n```\n/mood\n```"
        self.assertIsNone(split_command_stack(text))

    def test_midline_fence_opener_swallows(self):
        # /exec with a fenced block: the ``` opens mid-line, the /status
        # line is fence CONTENT, never a second command.
        text = "/exec ```python\n/status\n```"
        self.assertIsNone(split_command_stack(text))

    def test_tilde_fence_swallows(self):
        self.assertIsNone(split_command_stack("/status\n~~~\n/mood\n~~~"))

    def test_double_quoted_multiline_arg_swallows(self):
        # The /status line sits inside the quoted /say argument.
        text = '/say "line one\n/status"\n/mood'
        # Quote closes on line 2, so /mood IS a second command.
        self.assertEqual(split_command_stack(text),
                         ['/say "line one\n/status"', "/mood"])
        text2 = '/say "line one\n/status\nmore"\n/mood'
        self.assertEqual(split_command_stack(text2),
                         ['/say "line one\n/status\nmore"', "/mood"])
        # Unclosed quote: the rest of the message is one quoted arg,
        # so there is no stack at all.
        self.assertIsNone(split_command_stack('/say "line one\n/status'))

    def test_single_quoted_multiline_arg_swallows(self):
        text = "/say 'line one\n/status'\n/mood"
        self.assertEqual(split_command_stack(text),
                         ["/say 'line one\n/status'", "/mood"])

    def test_apostrophe_does_not_open_a_quote(self):
        # don't / it's must not break stacking: the ' is mid-word.
        self.assertEqual(split_command_stack("/say don't stop\n/status"),
                         ["/say don't stop", "/status"])

    def test_inline_code_span_swallows_newline(self):
        text = "/say `line one\n/status`"
        self.assertIsNone(split_command_stack(text))

    def test_too_deep_still_splits(self):
        segs = split_command_stack("\n".join(["/status"] * (MAX_STACK_DEPTH + 1)))
        self.assertEqual(len(segs), MAX_STACK_DEPTH + 1)
        self.assertEqual(MAX_STACK_DEPTH, 5)

    def test_non_stackable_contents(self):
        for kind in ("quit", "forget", "restart", "deploy", "update",
                     "pull", "hangman", "game", "trivia", "predict",
                     "npc", "dm"):
            self.assertIn(kind, NON_STACKABLE_COMMANDS, kind)
        for kind in ("status", "mood", "search", "help", "list", "say",
                     "providers", "book"):
            self.assertNotIn(kind, NON_STACKABLE_COMMANDS, kind)


class PlaceholderTests(unittest.TestCase):
    def test_prev_is_previous_output(self):
        self.assertEqual(
            expand_prev_placeholders("/mood got {prev}", ["STATUS-OK"]),
            "/mood got STATUS-OK")

    def test_prev_numbered(self):
        outs = ["one", "two"]
        self.assertEqual(expand_prev_placeholders("{prev:1}", outs), "one")
        self.assertEqual(expand_prev_placeholders("{prev:2}", outs), "two")

    def test_prev_out_of_range_stays_literal(self):
        self.assertEqual(expand_prev_placeholders("{prev:9}", ["one"]),
                         "{prev:9}")
        self.assertEqual(expand_prev_placeholders("{prev}", []), "{prev}")

    def test_escaped_prev_stays_literal(self):
        self.assertEqual(expand_prev_placeholders("{{prev}}", ["one"]),
                         "{prev}")

    def test_prev_truncated(self):
        out = expand_prev_placeholders("{prev}", ["x" * 5000])
        self.assertLess(len(out), 5000)
        self.assertIn("truncated", out)


class HelpTests(unittest.TestCase):
    def test_help_text_documents_stacking(self):
        text = help_text()
        self.assertIn("stack", text.lower())
        self.assertIn("{prev}", text)

    def test_detailed_help_stack_topic(self):
        page = detailed_help("stack")
        self.assertIn("{prev}", page)
        self.assertIn("one command per line", page)
        self.assertEqual(detailed_help("stacking"), page)


class StackRunTests(unittest.TestCase):
    def setUp(self):
        self.rt = _bare_runtime()
        self.calls = []
        self.rt._control_status = lambda: self.calls.append("status") or "STATUS-OK"
        self.rt._control_mood = lambda tail: self.calls.append(f"mood:{tail}") or f"MOOD:{tail}"
        self.rt._control_search = lambda tail, mode=None, chat_key=None: (
            self.calls.append(f"search:{tail}") or f"SEARCH:{tail}")
        self.rt._control_providers = lambda: self.calls.append("providers") or "PROVIDERS-OK"

    def _owner(self, text, chat_key=CHAT):
        return self.rt.handle_control(text, chat_key, message=_msg(OWNER_CHAT, text))

    def _stranger(self, text, chat_key="telegram:999"):
        return self.rt.handle_control(text, chat_key, message=_msg(STRANGER_CHAT, text))

    def test_runs_in_order_with_headers(self):
        out = self._owner("/status\n/mood happy")
        self.assertIn("── 1/2 /status ──", out)
        self.assertIn("── 2/2 /mood ──", out)
        self.assertIn("STATUS-OK", out)
        self.assertIn("MOOD:happy", out)
        self.assertEqual(self.calls, ["status", "mood:happy"])

    def test_prev_placeholder_feeds_next_command(self):
        out = self._owner("/status\n/mood got {prev}")
        self.assertIn("MOOD:got STATUS-OK", out)

    def test_failure_stops_stack(self):
        def boom():
            self.calls.append("status")
            raise RuntimeError("boom")
        self.rt._control_status = boom
        out = self._owner("/status\n/mood happy")
        self.assertIn("failed", out)
        self.assertIn("did not run", out)
        self.assertNotIn("MOOD:happy", out)
        self.assertEqual(self.calls, ["status"])

    def test_denied_step_blocks_whole_stack_upfront(self):
        out = self._stranger("/search hello\n/status")
        self.assertIn("/status", out)
        self.assertIn("owner", out.lower())
        self.assertIn("Nothing ran", out)
        self.assertEqual(self.calls, [])

    def test_outsider_public_stack_runs(self):
        out = self._stranger("/search hello\n/search world")
        self.assertIn("SEARCH:hello", out)
        self.assertIn("SEARCH:world", out)
        self.assertEqual(self.calls, ["search:hello", "search:world"])

    def test_confirmable_rejected_upfront(self):
        out = self._owner("/status\n/quit")
        self.assertIn("/quit", out)
        self.assertIn("Nothing ran", out)
        self.assertEqual(self.calls, [])
        out2 = self._owner("/status\n/quit confirm")
        self.assertIn("/quit", out2)
        self.assertEqual(self.calls, [])

    def test_game_rejected_upfront(self):
        out = self._owner("/status\n/hangman")
        self.assertIn("/hangman", out)
        self.assertIn("Nothing ran", out)
        self.assertEqual(self.calls, [])

    def test_lifecycle_rejected_upfront(self):
        out = self._owner("/status\n/restart")
        self.assertIn("/restart", out)
        self.assertEqual(self.calls, [])

    def test_too_deep_rejected_upfront(self):
        out = self._owner("\n".join(["/status"] * (MAX_STACK_DEPTH + 1)))
        self.assertIn("too deep", out)
        self.assertEqual(self.calls, [])

    def test_sms_blocked_step_rejects_stack(self):
        out = self.rt.handle_control(
            "/status\n/search hello", "sms:123",
            message=_msg(OWNER_CHAT, "/status\n/search hello"))
        self.assertIn("app", out)
        self.assertIn("Nothing ran", out)
        self.assertEqual(self.calls, [])

    def test_single_command_unaffected(self):
        self.assertEqual(self._owner("/status"), "STATUS-OK")
        self.assertEqual(self._owner("/mood happy"), "MOOD:happy")

    def test_multiline_single_command_unaffected(self):
        # Second line is not a command: the whole message is one /mood.
        out = self._owner("/mood line one\nline two")
        self.assertEqual(out, "MOOD:line one\nline two")
        self.assertNotIn("──", out)

    def test_fenced_multiline_single_command_unaffected(self):
        out = self._owner("/mood ```\n/status\n```")
        self.assertEqual(out, "MOOD:```\n/status\n```")
        self.assertNotIn("──", out)

    def test_semicolon_single_command_is_not_a_command(self):
        # "/status;" is not a registered kind, so the whole message falls
        # through as ordinary text — no stack, nothing ran.
        out = self._owner("/status; /mood")
        self.assertEqual(out, "")
        self.assertEqual(self.calls, [])

    def test_nl_unaffected(self):
        self.assertEqual(self._owner("hello there"), "")


if __name__ == "__main__":
    unittest.main()
