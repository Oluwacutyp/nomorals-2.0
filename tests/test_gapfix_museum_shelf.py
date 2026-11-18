"""Museum shelf filter — /museum <shelf> shows only that shelf.

The census (`/museum`) lists every card and `/museum send <id>`
delivers one, but there was no way to look at one shelf: a shelf
word was refused as an "unknown museum verb". These tests pin the
filter: a known shelf shows only its cards (same card lines as the
full census, filtered), an unknown shelf is refused in words
naming the shelves that hold cards, and bare `/museum` plus
`/museum send` are byte-identical to before.
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    PUBLIC_COMMANDS,
    caller_may_run,
    list_catalog,
    parse_control,
)
from nomorals.social.chat.telegram_menu import validate_menu
from tests.test_gapfix_museum_command import _msg
from tests.test_gapfix_museum_send import _SendRecordingGateway

_BOOK_TEXT = (
    "Chapter One\n\nThe quick brown fox jumps over the lazy dog, "
    "and the dog watches the fox go by without a sound.\n\n"
    "Chapter Two\n\nPack my box with five dozen liquor jugs, "
    "and pour a cup of tea while the kettle sings its song.\n\n"
    "Chapter Three\n\nHow vexingly quick daft zebras jump, "
    "and the brown fox runs home through the quiet evening.\n"
)


class MuseumShelfRegistrationTest(unittest.TestCase):
    def test_registered_shape(self) -> None:
        # Bare (0) + one-word shelf (1) + send <id> (2): the (0, 2)
        # arity already admits all three forms, no change needed.
        self.assertEqual(CONTROL_COMMANDS["museum"], (0, 2))
        usage = COMMAND_DETAILS["museum"]["usage"]
        self.assertIn("/museum <shelf>", usage)
        self.assertIn("/museum send <id>", usage)
        example = COMMAND_DETAILS["museum"]["example"]
        self.assertIn("/museum library-books", example)
        self.assertIn("/museum send", example)
        self.assertIn("shelf", COMMAND_DETAILS["museum"]["what"])
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertEqual(grouped.count("museum"), 1)
        self.assertIn("/museum", list_catalog())
        ok, problems = validate_menu()
        self.assertTrue(ok, problems)

    def test_owner_only(self) -> None:
        self.assertNotIn("museum", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("museum", False))
        self.assertTrue(caller_may_run("museum", True))
        self.assertEqual(len(PUBLIC_COMMANDS), 279)

    def test_parse_control(self) -> None:
        cmd = parse_control("/museum library-books")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "museum")
        self.assertEqual(cmd.tail, "library-books")
        # Three words still exceed the arity.
        extra = parse_control("/museum a b c")
        self.assertIsNotNone(extra)
        assert extra is not None
        self.assertEqual(extra.kind, "error")


class MuseumShelfCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.home() / "workspace" / "museum-shelf-verify"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-museum-shelf-", dir=str(scratch))
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        # Real tools: the handler must reach the real museum_census
        # tool, not a stub of it.
        self.context = build_context(
            settings, with_executor=False, with_router=False,
            with_tools=True)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        # The send-capable gateway: /museum send must keep working
        # through the same file_send path, so the shelf suite proves
        # it here instead of trusting the send suite alone.
        self.gateway = _SendRecordingGateway()
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str, *, owner: bool = True,
             chat_key: str = "telegram:999") -> str:
        return self.runtime.handle_control(
            text, chat_key, message=_msg(text, chat_key=chat_key,
                                         owner=owner))

    def _tool_census(self, **kwargs: Any) -> Any:
        outcome = self.context.tools.call("museum_census", **kwargs)
        self.assertTrue(outcome.ok, outcome)
        return outcome.value

    def _ingest_book(self, name: str, title: str) -> str:
        book = Path(self.tmp.name) / name
        book.write_text(_BOOK_TEXT, encoding="utf-8")
        ingested = self.context.tools.call(
            "library_ingest", path=str(book), title=title)
        self.assertTrue(ingested.ok, ingested)
        return str(ingested.value["slug"])

    def test_filter_books_only(self) -> None:
        slug_a = self._ingest_book("book-a.txt", "Test Book A")
        slug_b = self._ingest_book("book-b.txt", "Test Book B")
        tool_value = self._tool_census(shelf="library-books")
        reply = self._run("/museum library-books")
        # Same tool, same output: the command is a thin dispatch.
        self.assertEqual(reply, tool_value["card_text"])
        self.assertIn("🏛️ **Museum** — 2 (library-books: 2)", reply)
        self.assertIn(f"library-book:{slug_a}", reply)
        self.assertIn(f"library-book:{slug_b}", reply)
        self.assertNotIn("library-source:", reply)
        self.assertEqual(self.gateway.sent, [])

    def test_filter_sources_only(self) -> None:
        slug_a = self._ingest_book("book-a.txt", "Test Book A")
        self._ingest_book("book-b.txt", "Test Book B")
        tool_value = self._tool_census(shelf="library-sources")
        reply = self._run("/museum library-sources")
        self.assertEqual(reply, tool_value["card_text"])
        self.assertIn("🏛️ **Museum** — 2 (library-sources: 2)", reply)
        self.assertIn("library-source:", reply)
        self.assertNotIn(f"library-book:{slug_a}", reply)

    def test_unknown_shelf_refuses_naming_known(self) -> None:
        self._ingest_book("book-a.txt", "Test Book A")
        self._ingest_book("book-b.txt", "Test Book B")
        reply = self._run("/museum nope")
        self.assertIn("no shelf 'nope'", reply)
        self.assertIn("library-books", reply)
        self.assertIn("library-sources", reply)
        self.assertEqual(self.gateway.sent, [])

    def test_shelf_case_insensitive(self) -> None:
        self._ingest_book("book-a.txt", "Test Book A")
        lower = self._run("/museum library-books")
        upper = self._run("/museum Library-Books")
        self.assertEqual(upper, lower)

    def test_bare_museum_unchanged(self) -> None:
        self._ingest_book("book-a.txt", "Test Book A")
        self._ingest_book("book-b.txt", "Test Book B")
        tool_value = self._tool_census()
        reply = self._run("/museum")
        self.assertEqual(reply, tool_value["card_text"])
        self.assertIn("library-book:", reply)
        self.assertIn("library-source:", reply)
        self.assertEqual(tool_value["by_shelf"], {
            "library-books": 2, "library-sources": 2})

    def test_send_still_works(self) -> None:
        slug = self._ingest_book("book-a.txt", "Test Book A")
        reply = self._run(f"/museum send library-book:{slug}")
        self.assertIn("sent", reply)
        self.assertIn(f"library-book:{slug}", reply)
        self.assertEqual(len(self.gateway.files), 1)

    def test_two_word_non_send_is_usage(self) -> None:
        self._ingest_book("book-a.txt", "Test Book A")
        reply = self._run("/museum library-books extra")
        self.assertIn("usage: /museum <shelf>", reply)
        self.assertEqual(self.gateway.sent, [])

    def test_unknown_shelf_on_empty_museum(self) -> None:
        reply = self._run("/museum nope")
        self.assertIn("no shelf 'nope'", reply)
        self.assertIn("museum is empty", reply)

    def test_tool_filter_shape(self) -> None:
        self._ingest_book("book-a.txt", "Test Book A")
        self._ingest_book("book-b.txt", "Test Book B")
        filtered = self._tool_census(shelf="library-books")
        self.assertEqual(filtered["by_shelf"], {"library-books": 2})
        self.assertEqual(filtered["count"], 2)
        self.assertEqual(filtered["shelf"], "library-books")
        self.assertNotIn("unknown_shelf", filtered)
        self.assertTrue(all(c["shelf"] == "library-books"
                            for c in filtered["artifacts"]))
        refused = self._tool_census(shelf="nope")
        self.assertTrue(refused["unknown_shelf"])
        self.assertEqual(refused["shelf"], "nope")
        self.assertEqual(refused["by_shelf"], {
            "library-books": 2, "library-sources": 2})
        self.assertIn("no shelf 'nope'", refused["card_text"])


if __name__ == "__main__":
    unittest.main()
