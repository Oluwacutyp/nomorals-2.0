"""R3-73: /museum — the Museum census as a direct command.

The census (`nomorals/museum.py`) and its brain tool (`museum_census`
in `books/tools.py`) exist and are live, but typing `/museum` goes
nowhere: no registry entry, no dispatch branch. These tests pin the
wire: the command reaches the same tool the brain uses and returns
its real census output, byte-identical.
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
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    caller_may_run,
    list_catalog,
    parse_control,
)
from nomorals.social.chat.telegram_menu import validate_menu


def _msg(text: str, *, chat_key: str = "telegram:999",
         owner: bool = True) -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "999", kind=ChatKind.DM),
        incoming=True, text=text,
        sender="owner" if owner else "stranger",
        sender_id="5478650254" if owner else "999",
        meta={"is_owner": True} if owner else {},
    )


class _RecordingGateway:
    adapters: dict = {}

    def __init__(self) -> None:
        self.sent: list = []

    def status(self) -> dict:
        return {}

    def set_rate_limit(self, *a: Any, **k: Any) -> None:
        pass

    def typing(self, *a: Any, **k: Any) -> None:
        pass

    def send(self, platform: str, chat: Any, text: str,
             **kwargs: Any) -> SimpleNamespace:
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, error="")


class MuseumRegistrationTest(unittest.TestCase):
    def test_registered_everywhere(self) -> None:
        self.assertIn("museum", CONTROL_COMMANDS)
        self.assertIn("museum", COMMAND_DETAILS)
        self.assertTrue(COMMAND_DETAILS["museum"].get("what"))
        self.assertIn("museum", LIST_ONELINERS)
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertEqual(grouped.count("museum"), 1)
        self.assertIn("/museum", COMMAND_DETAILS["museum"]["usage"])

    def test_owner_only(self) -> None:
        self.assertNotIn("museum", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("museum", False))
        self.assertTrue(caller_may_run("museum", True))

    def test_parse_control(self) -> None:
        cmd = parse_control("/museum")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "museum")
        extra = parse_control("/museum shelf")
        self.assertIsNotNone(extra)
        assert extra is not None
        self.assertEqual(extra.kind, "error")

    def test_listed_and_in_menu(self) -> None:
        self.assertIn("/museum", list_catalog())
        ok, problems = validate_menu()
        self.assertTrue(ok, problems)


class MuseumCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.home() / "workspace" / "gapfix73-verify"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-museum-cmd-", dir=str(scratch))
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
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())

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

    def test_bare_museum_matches_tool_output(self) -> None:
        tool_value = self._tool_census()
        reply = self._run("/museum")
        # Same tool, same output: the command is a thin dispatch.
        self.assertEqual(reply, tool_value["card_text"])
        self.assertIn("museum is empty", reply)
        self.assertEqual(tool_value["count"], 0)

    def test_museum_lists_ingested_book(self) -> None:
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
        slug = ingested.value["slug"]
        reply = self._run("/museum")
        self.assertIn(f"library-book:{slug}", reply)
        self.assertIn("Test Book", reply)
        self.assertIn("library-books", reply)
        tool_value = self._tool_census()
        self.assertEqual(reply, tool_value["card_text"])
        # Ingest files one book card + one source card (the ingest
        # registers its origin as a library source).
        self.assertEqual(tool_value["by_shelf"], {
            "library-books": 1, "library-sources": 1})
        kinds = [c["kind"] for c in tool_value["artifacts"]]
        self.assertEqual(sorted(kinds),
                         ["library-book", "library-source"])

    def test_non_owner_is_denied(self) -> None:
        reply = self._run("/museum", owner=False)
        self.assertIn("just for the owner", reply)
        self.assertNotIn("Museum", reply)
