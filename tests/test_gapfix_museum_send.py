"""R3-73 remainder: /museum send — deliver one sendable museum card.

The census (`/museum`) lists every card, but there was no way to get
one out: `Museum.find` + the `sendable` flag existed for exactly this
verb and nothing called them. These tests pin the wire: the command
finds the card by id (exact or unique prefix), refuses unknown /
ambiguous / exhibit-only ids in words, and delivers a sendable card
through the same `file_send` tool `/file` and `/artifact send` use.
Bare `/museum` behavior is unchanged.
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
from tests.test_gapfix_museum_command import (
    _msg,
    _RecordingGateway,
)


class _SendRecordingGateway(_RecordingGateway):
    adapters = {"telegram": object()}

    def __init__(self) -> None:
        super().__init__()
        self.files: list = []

    def send_file(self, platform: str, chat_id: str, path: str,
                  caption: str = "", kind: str = "",
                  view_once: bool = False) -> SimpleNamespace:
        self.files.append({
            "platform": platform, "chat_id": chat_id, "path": path,
            "caption": caption, "kind": kind, "view_once": view_once,
        })
        return SimpleNamespace(ok=True, error="", message_id="m-1")


class MuseumSendRegistrationTest(unittest.TestCase):
    def test_send_parses(self) -> None:
        cmd = parse_control("/museum send library-book:foo")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "museum")
        self.assertEqual(cmd.tail, "send library-book:foo")
        bare = parse_control("/museum")
        self.assertIsNotNone(bare)
        assert bare is not None
        self.assertEqual(bare.kind, "museum")
        self.assertEqual(bare.tail, "")
        # Three words still exceed the arity: bare + send <id> only.
        too_long = parse_control("/museum a b c")
        self.assertIsNotNone(too_long)
        assert too_long is not None
        self.assertEqual(too_long.kind, "error")

    def test_registry_unchanged_shape(self) -> None:
        self.assertEqual(CONTROL_COMMANDS["museum"], (0, 2))
        self.assertIn("/museum send", COMMAND_DETAILS["museum"]["usage"])
        self.assertIn("/museum send",
                      COMMAND_DETAILS["museum"]["example"])
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


class MuseumSendCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.home() / "workspace" / "museum-send-verify"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-museum-send-", dir=str(scratch))
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        # Real tools: the handler must reach the real museum_find
        # and file_send tools, not stubs of them.
        self.context = build_context(
            settings, with_executor=False, with_router=False,
            with_tools=True)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
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

    def _tool_census(self) -> Any:
        outcome = self.context.tools.call("museum_census")
        self.assertTrue(outcome.ok, outcome)
        return outcome.value

    def _ingest_book(self) -> str:
        book = Path(self.tmp.name) / "test-book.txt"
        book.write_text(
            "Chapter One\n\nThe quick brown fox jumps over the lazy dog, "
            "and the dog watches the fox go by without a sound.\n\n"
            "Chapter Two\n\nPack my box with five dozen liquor jugs, "
            "and pour a cup of tea while the kettle sings its song.\n\n"
            "Chapter Three\n\nHow vexingly quick daft zebras jump, "
            "and the brown fox runs home through the quiet evening.\n",
            encoding="utf-8")
        ingested = self.context.tools.call(
            "library_ingest", path=str(book), title="Test Book")
        self.assertTrue(ingested.ok, ingested)
        return str(ingested.value["slug"])

    def _cards_by_kind(self) -> dict:
        value = self._tool_census()
        by_kind: dict = {}
        for card in value["artifacts"]:
            by_kind.setdefault(card["kind"], []).append(card)
        return by_kind

    def test_bare_museum_unchanged(self) -> None:
        tool_value = self._tool_census()
        reply = self._run("/museum")
        self.assertEqual(reply, tool_value["card_text"])
        self.assertIn("museum is empty", reply)
        self.assertEqual(self.gateway.files, [])

    def test_send_ingested_book(self) -> None:
        slug = self._ingest_book()
        card_id = f"library-book:{slug}"
        cards = self._cards_by_kind()
        book = cards["library-book"][0]
        self.assertEqual(book["id"], card_id)
        self.assertTrue(book["sendable"])
        self.assertTrue(Path(str(book["path"])).is_file())
        reply = self._run(f"/museum send {card_id}")
        self.assertEqual(len(self.gateway.files), 1)
        sent = self.gateway.files[0]
        self.assertEqual(sent["platform"], "telegram")
        self.assertEqual(sent["chat_id"], "telegram:999")
        self.assertEqual(sent["path"], book["path"])
        self.assertIn("Test Book", sent["caption"])
        self.assertIn("sent", reply)
        self.assertIn(card_id, reply)

    def test_send_unique_prefix(self) -> None:
        slug = self._ingest_book()
        reply = self._run(f"/museum send library-book:{slug[:4]}")
        self.assertEqual(len(self.gateway.files), 1)
        self.assertIn("sent", reply)
        self.assertIn(f"library-book:{slug}", reply)

    def test_send_unknown_id_refuses(self) -> None:
        self._ingest_book()
        reply = self._run("/museum send library-book:nope")
        self.assertIn("no single card matches", reply)
        self.assertIn("library-book:nope", reply)
        self.assertEqual(self.gateway.files, [])

    def test_send_ambiguous_prefix_refuses(self) -> None:
        # After an ingest, "library-" prefixes the book card AND the
        # source card — find must refuse rather than guess one.
        self._ingest_book()
        reply = self._run("/museum send library-")
        self.assertIn("no single card matches", reply)
        self.assertEqual(self.gateway.files, [])

    def test_send_exhibit_only_refuses(self) -> None:
        self._ingest_book()
        cards = self._cards_by_kind()
        source = cards["library-source"][0]
        self.assertFalse(source["sendable"])
        reply = self._run(f"/museum send {source['id']}")
        self.assertIn("exhibit-only", reply)
        self.assertEqual(self.gateway.files, [])

    def test_send_without_id_refuses(self) -> None:
        reply = self._run("/museum send")
        self.assertIn("usage: /museum send <id>", reply)
        self.assertEqual(self.gateway.files, [])

    def test_unknown_shelf_refuses(self) -> None:
        # A shelf word is a filter attempt now, not a verb: on the
        # empty museum it is refused as an unknown shelf, naming no
        # shelves because none hold cards yet.
        reply = self._run("/museum shelf")
        self.assertIn("no shelf 'shelf'", reply)
        self.assertIn("museum is empty", reply)
        self.assertEqual(self.gateway.files, [])

    def test_non_owner_denied(self) -> None:
        slug = self._ingest_book()
        reply = self._run(f"/museum send library-book:{slug}",
                          owner=False)
        self.assertIn("just for the owner", reply)
        self.assertEqual(self.gateway.files, [])

    def test_no_gateway_refuses(self) -> None:
        slug = self._ingest_book()
        saved = self.context.extras.get("gateway")
        self.context.extras["gateway"] = None
        try:
            reply = self._run(f"/museum send library-book:{slug}")
        finally:
            self.context.extras["gateway"] = saved
        self.assertIn("file send failed", reply)
        self.assertEqual(self.gateway.files, [])

    def test_send_without_message_refuses(self) -> None:
        reply = self.runtime._control_museum("send library-book:x",
                                             None)
        self.assertIn("live chat", reply)


if __name__ == "__main__":
    unittest.main()
