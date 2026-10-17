"""Personal bot (USER-ONLY): BotFather command list + inline buttons.

Covers:
* the BotFather list — parses (``command - description``), ≤100 commands,
  descriptions ≤256 chars, no duplicates, covers the public 52 + the 48
  owner-only commands, and the doc (``docs/PERSONAL_BOT_COMMANDS.md``)
  matches the generator (no drift);
* personal inline keyboards — /help gets category buttons, /power gets
  Power On/Off/Status, button callbacks route to the right handler
  through the adapter (signed callback → synthetic ``/command``);
* the gated bot is unchanged — its replies carry no personal buttons.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from nomorals.social.chat.personal_commands import (
    PERSONAL_OWNER_COMMANDS,
    personal_botfather_commands,
    render_botfather_text,
    validate_botfather_commands,
)
from nomorals.social.chat.tgbot_buttons import (
    buttons_for_text,
    personal_buttons_for,
)

_LINE_RE = re.compile(r"^([a-z0-9_]+) - (.+)$")


def _doc_block() -> str:
    doc = (Path(__file__).resolve().parents[1] / "docs" / "PERSONAL_BOT_COMMANDS.md").read_text()
    start = doc.index("```\n") + len("```\n")
    end = doc.index("\n```", start)
    return doc[start:end]


class BotfatherListTests(unittest.TestCase):
    def test_list_is_100_and_valid(self):
        entries = personal_botfather_commands()
        self.assertEqual(len(entries), 100)
        self.assertEqual(validate_botfather_commands(), [])

    def test_lines_parse_botfather_format(self):
        text = render_botfather_text()
        lines = text.strip().split("\n")
        self.assertEqual(len(lines), 100)
        seen: set[str] = set()
        for line in lines:
            m = _LINE_RE.match(line)
            self.assertIsNotNone(m, f"line does not parse as 'command - description': {line!r}")
            assert m is not None
            cmd, desc = m.group(1), m.group(2)
            self.assertNotIn(cmd, seen, f"duplicate command: {cmd}")
            seen.add(cmd)
            self.assertLessEqual(len(desc), 256, f"description too long for {cmd}")
            self.assertTrue(desc.strip(), f"empty description for {cmd}")

    def test_covers_public_52_and_owner_48(self):
        from nomorals.social.chat.control import PUBLIC_COMMANDS

        kinds = [k for k, _ in personal_botfather_commands()]
        self.assertEqual(set(PUBLIC_COMMANDS), set(kinds[:52]))
        self.assertEqual(list(PERSONAL_OWNER_COMMANDS), kinds[52:])
        self.assertEqual(len(PERSONAL_OWNER_COMMANDS), 48)

    def test_doc_matches_generator(self):
        self.assertEqual(_doc_block().strip(), render_botfather_text().strip())

    def test_gated_public_list_untouched(self):
        # The gated bot's list is PUBLIC_COMMANDS — this change must not
        # have touched it: still exactly the 52.
        from nomorals.social.chat.control import PUBLIC_COMMANDS

        self.assertEqual(len(PUBLIC_COMMANDS), 52)


def _callbacks(keyboard):
    return [data for row in (keyboard or []) for _, data in row]


class PersonalButtonsTests(unittest.TestCase):
    def test_help_gets_category_buttons(self):
        kb = personal_buttons_for("the full catalog …", "/help")
        self.assertIsNotNone(kb)
        assert kb is not None
        cbs = _callbacks(kb)
        for want in ("list search", "list media", "list games", "power status",
                     "list build", "list memory", "list voice", "list wisdom",
                     "list tools", "list control", "list"):
            self.assertIn(want, cbs, f"missing help category button: {want}")

    def test_list_gets_category_buttons(self):
        kb = personal_buttons_for("commands …", "/list")
        assert kb is not None
        self.assertIn("list search", _callbacks(kb))

    def test_power_gets_power_buttons(self):
        kb = personal_buttons_for("power mode: locked", "/power")
        assert kb is not None
        self.assertEqual(
            _callbacks(kb), ["power on", "power off", "power status"])
        # Power On carries NO key — the callback is exactly "power on",
        # nothing after it that could be a key.

    def test_search_gets_next_actions(self):
        kb = personal_buttons_for("results …", "/search quantum computing")
        assert kb is not None
        cbs = _callbacks(kb)
        self.assertIn("search quantum computing", cbs)  # retry
        self.assertIn("searchdeep quantum computing", cbs)  # deepen
        self.assertIn("remember quantum computing", cbs)  # save query
        self.assertIn("list", cbs)

    def test_long_query_buttons_never_truncated(self):
        # A 100-char query cannot fit the 47-byte callback budget: the
        # query-carrying buttons are OMITTED, never sent truncated (a
        # truncated query would run the wrong query).
        kb = personal_buttons_for("results …", "/search " + "x" * 100)
        assert kb is not None
        for data in _callbacks(kb):
            self.assertLessEqual(len(data.encode()), 47)
            self.assertNotIn("x" * 47, data)

    def test_plain_chat_gets_no_keyboard(self):
        self.assertIsNone(personal_buttons_for("hey, how are you?", "hello"))
        self.assertIsNone(personal_buttons_for("hey, how are you?", ""))

    def test_unknown_command_gets_no_keyboard(self):
        self.assertIsNone(personal_buttons_for("done", "/frobnicate"))

    def test_game_result_falls_through_to_shared(self):
        # Personal has no game mapping: None here means the adapter falls
        # through to the shared game-result buttons (Rematch/Next/…).
        self.assertIsNone(personal_buttons_for(
            "The House wins this round. 120 points and coins credited.",
            "/game trivia"))

    def test_all_personal_callbacks_parse_to_real_commands(self):
        from nomorals.social.chat.control import parse_control

        cases = [
            ("/help", "catalog"), ("/list", "catalog"),
            ("/power", "power"), ("/search q", "search"),
            ("/searchdeep q", "search"), ("/task", "task"),
            ("/mission", "mission"), ("/status", "status"),
            ("/providers", "providers"), ("/model", "model"),
            ("/impossible", "impossible"), ("/book", "book"),
        ]
        for origin, _ in cases:
            kb = personal_buttons_for("reply …", origin)
            assert kb is not None, f"no keyboard for {origin}"
            for data in _callbacks(kb):
                cmd = parse_control(f"/{data}")
                self.assertIsNotNone(cmd, f"callback {data!r} does not parse")
                assert cmd is not None
                self.assertNotEqual(cmd.kind, "error",
                                    f"callback {data!r} is an error: {cmd}")

    def test_gated_buttons_unchanged(self):
        # buttons_for_text with no personal flag behaves exactly as
        # before: game results keep the shared buttons, plain chat none.
        kb = buttons_for_text(
            "The House wins this round. 120 points and coins credited.")
        assert kb is not None
        self.assertIn("game rematch", _callbacks(kb))
        self.assertIsNone(buttons_for_text("just chatting"))


class PersonalCallbackRoutingTests(unittest.TestCase):
    """A personal-button tap must route to the same command handler."""

    def _personal_adapter(self):
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
                return _FakeResp()

            def get(self, url, timeout=None):
                self.calls.append((url, {}))
                return _FakeResp()

        session = _FakeSession()
        adapter = TelegramBotAdapter(
            token="test-token-123", session=session,
            adapter_name="telegram-personal", owner_only=True,
            owner_sender_ids={"7"},
        )
        return adapter, session

    def test_power_status_button_routes_to_power_status(self):
        adapter, _ = self._personal_adapter()
        received: list[str] = []
        signed = adapter._sign_callback("power status")
        query = {
            "id": "q1",
            "from": {"id": 7, "is_bot": False, "first_name": "Owner"},
            "message": {"chat": {"id": 7, "type": "private"}},
            "data": signed,
        }
        adapter._handle_callback(query, lambda msg: received.append(msg.text))
        self.assertEqual(received, ["/power status"])

    def test_help_category_button_routes_to_list_group(self):
        adapter, _ = self._personal_adapter()
        received: list[str] = []
        signed = adapter._sign_callback("list games")
        query = {
            "id": "q2",
            "from": {"id": 7, "is_bot": False, "first_name": "Owner"},
            "message": {"chat": {"id": 7, "type": "private"}},
            "data": signed,
        }
        adapter._handle_callback(query, lambda msg: received.append(msg.text))
        self.assertEqual(received, ["/list games"])

    def test_personal_send_attaches_personal_buttons(self):
        from nomorals.social.chat.base import ChatRef

        adapter, session = self._personal_adapter()
        adapter._last_command["7"] = "/power"
        adapter.send(
            ChatRef(platform="telegram-personal", chat_id="7", kind="dm"),
            "power mode: locked",
        )
        sends = [p for url, p in session.calls if url.endswith("/sendMessage")]
        self.assertTrue(sends, "no sendMessage recorded")
        markup = sends[0].get("reply_markup", {})
        rows = markup.get("inline_keyboard", [])
        datas = [
            adapter._verify_callback(btn["callback_data"])
            for row in rows for btn in row
        ]
        self.assertIn("power status", datas)
        self.assertIn("power on", datas)
        self.assertIn("power off", datas)


if __name__ == "__main__":
    unittest.main()
