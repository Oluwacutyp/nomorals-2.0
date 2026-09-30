"""Offline tests for account-history export into the training pipeline.

The network part of the export (telethon client) is replaced by a fake
client, exactly like the chat adapter tests do; the mapping, PII scrub,
dedup, and dataset registration paths run for real against an in-memory
database.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.exporter import export_telegram, history_to_examples, scrub_examples
from nomorals.training.dataset import DatasetRegistry, Example, Turn


class HistoryToExamplesTest(unittest.TestCase):
    def test_dm_alternating_roles(self) -> None:
        dialogs = [("Alice", [(False, "hey"), (True, "hi, how are you"), (False, "good")])]
        examples = history_to_examples(dialogs)
        self.assertEqual(len(examples), 1)
        self.assertEqual([t.role for t in examples[0].turns], ["user", "assistant", "user"])
        self.assertEqual(examples[0].source, "Alice")

    def test_consecutive_same_side_merge_into_one_turn(self) -> None:
        dialogs = [("Alice", [(True, "line one"), (True, "line two"), (False, "ok"), (True, "thanks")])]
        examples = history_to_examples(dialogs)
        self.assertEqual([t.role for t in examples[0].turns], ["assistant", "user", "assistant"])
        self.assertEqual(examples[0].turns[0].content, "line one\nline two")

    def test_group_multi_sender_merges_to_user(self) -> None:
        entries = [(False, "alice: hi"), (False, "bob: hey"), (True, "sup both")]
        examples = history_to_examples([("Group", entries)])
        self.assertEqual([t.role for t in examples[0].turns], ["user", "assistant"])
        self.assertEqual(examples[0].turns[0].content, "alice: hi\nbob: hey")

    def test_pure_owner_dialog_dropped(self) -> None:
        # Saved Messages: only the owner's own notes — nothing to respond to.
        dialogs = [("Me", [(True, "note one"), (True, "note two")])]
        self.assertEqual(history_to_examples(dialogs), [])

    def test_pure_other_dialog_dropped(self) -> None:
        # A channel nobody replied in: no voice material to imitate.
        dialogs = [("Channel", [(False, "post one"), (False, "post two")])]
        self.assertEqual(history_to_examples(dialogs), [])

    def test_short_dialog_dropped(self) -> None:
        self.assertEqual(history_to_examples([("Alice", [(True, "hi")])]), [])

    def test_window_splitting(self) -> None:
        entries = [(i % 2 == 0, f"msg {i}") for i in range(24)]
        examples = history_to_examples([("Big", entries)], window_turns=8)
        self.assertEqual(len(examples), 3)
        for example in examples:
            self.assertLessEqual(len(example.turns), 8)
            self.assertEqual({t.role for t in example.turns}, {"user", "assistant"})

    def test_empty_and_media_only_entries_skipped(self) -> None:
        entries = [(False, ""), (True, "  "), (False, "real text"), (True, "answer")]
        examples = history_to_examples([("Alice", entries)])
        self.assertEqual([t.role for t in examples[0].turns], ["user", "assistant"])


class ScrubExamplesTest(unittest.TestCase):
    def test_pii_redacted_and_near_duplicates_dropped(self) -> None:
        a = Example(
            turns=[Turn("user", "call me at 08034567890 tonight"), Turn("assistant", "will do, love")],
            source="S1",
        )
        b = Example(
            turns=[Turn("user", "call me at 08034567890 tonight"), Turn("assistant", "will do, love")],
            source="S2",
        )
        kept, pii_count, duplicates = scrub_examples([a, b])
        self.assertEqual(len(kept), 1)
        self.assertEqual(pii_count, 2)  # both were redacted before dedup saw them
        self.assertEqual(duplicates, 1)
        self.assertNotIn("08034567890", kept[0].turns[0].content)
        self.assertIn("[REDACTED_PHONE]", kept[0].turns[0].content)

    def test_empty_examples_dropped(self) -> None:
        kept, _, _ = scrub_examples([Example(turns=[Turn("user", "   "), Turn("assistant", "")])])
        self.assertEqual(kept, [])


class _FakeExportClient:
    """Stand-in for telethon.TelegramClient — yields canned dialogs/messages."""

    dialogs: list = []
    messages: dict = {}
    started = False

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def start(self) -> None:
        type(self).started = True

    async def disconnect(self) -> None:
        pass

    def iter_dialogs(self):
        dialogs = type(self).dialogs

        async def _gen():
            for dialog in dialogs:
                yield dialog

        return _gen()

    def iter_messages(self, entity, limit=None, reverse=False):
        messages = type(self).messages.get(entity, [])

        async def _gen():
            for msg in messages:
                yield msg

        return _gen()


def _dm_entries() -> list:
    return [
        SimpleNamespace(raw_text="hey, call me at 08034567890", out=False),
        SimpleNamespace(raw_text="will call you tonight", out=True),
        SimpleNamespace(raw_text=None, out=False),  # media-only message
        SimpleNamespace(raw_text="thanks, see you", out=False),
        SimpleNamespace(raw_text="see you then, love", out=True),
    ]


class ExportTelegramTest(unittest.TestCase):
    def setUp(self) -> None:
        self._fake_mod = types.ModuleType("telethon")
        self._fake_mod.TelegramClient = _FakeExportClient
        self._saved = sys.modules.get("telethon")
        sys.modules["telethon"] = self._fake_mod

        self._tmp = tempfile.TemporaryDirectory(prefix="nm-export-")
        self.context = build_context(
            Settings(home=self._tmp.name),
            with_executor=False,
            with_router=False,
            with_memory=False,
            with_tools=False,
        )
        self.context.__enter__()
        self.settings = self.context.settings
        self.settings.chat.telegram_api_id = "123456"
        self.settings.chat.telegram_api_hash = "abcdef0123456789"

        _FakeExportClient.started = False
        _FakeExportClient.dialogs = [
            SimpleNamespace(id=101, title="Alice", entity="e101"),
            SimpleNamespace(id=5478650254, title="Vrede peace", entity="eMe"),
            SimpleNamespace(id=202, title="Naija Tech News", entity="e202"),
        ]
        _FakeExportClient.messages = {
            "e101": _dm_entries(),
            # Saved Messages: owner-only notes -> must be dropped by the mapper.
            "eMe": [
                SimpleNamespace(raw_text="grocery list: rice, oil, gas", out=True),
                SimpleNamespace(raw_text="book the appointment for the 14th", out=True),
            ],
            # Channel nobody replied in -> must be dropped by the mapper.
            "e202": [
                SimpleNamespace(raw_text="Android 16 released", out=False),
                SimpleNamespace(raw_text="termux tip of the day", out=False),
            ],
        }

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)
        if self._saved is None:
            sys.modules.pop("telethon", None)
        else:
            sys.modules["telethon"] = self._saved
        self._tmp.cleanup()

    def _registry(self) -> DatasetRegistry:
        return DatasetRegistry(self.context.db)

    def test_full_export_registers_scrubbed_dataset(self) -> None:
        result = export_telegram(self.settings, db=self.context.db)

        self.assertTrue(_FakeExportClient.started)
        self.assertEqual(result.stats.dialogs, 3)
        self.assertEqual(result.stats.messages, 8)  # media-only message not counted
        self.assertEqual(result.stats.examples, 1)  # only the DM window qualifies
        self.assertEqual(result.stats.pii_scrubbed, 1)
        self.assertTrue(result.dataset_id)
        self.assertTrue(result.path)

        dataset = self._registry().get(result.dataset_id)
        self.assertEqual(dataset.rows, 1)
        self.assertEqual(dataset.metadata.get("source"), "telegram-export")

        lines = [json.loads(line) for line in Path(result.path).read_text().splitlines()]
        self.assertEqual(len(lines), 1)
        blob = json.dumps(lines)
        self.assertNotIn("08034567890", blob)  # the number never hits disk
        self.assertIn("[REDACTED_PHONE]", blob)
        self.assertEqual(lines[0]["messages"][0]["role"], "user")
        self.assertEqual(lines[0]["messages"][1]["role"], "assistant")
        self.assertEqual(lines[0]["source"], "Alice")

    def test_only_chats_filter(self) -> None:
        result = export_telegram(self.settings, db=self.context.db, only_chats="101")
        self.assertEqual(result.stats.dialogs, 1)
        self.assertEqual(result.stats.messages, 4)
        self.assertEqual(result.stats.examples, 1)
        self.assertTrue(result.dataset_id)

    def test_no_register_writes_file_only(self) -> None:
        result = export_telegram(self.settings, db=self.context.db, register=False)
        self.assertEqual(result.dataset_id, "")
        self.assertTrue(result.path)
        self.assertTrue(Path(result.path).exists())
        self.assertEqual(self._registry().list(), [])

    def test_missing_credentials_raise_clear_error(self) -> None:
        self.settings.chat.telegram_api_id = ""
        with self.assertRaises(RuntimeError) as ctx:
            export_telegram(self.settings, db=self.context.db)
        self.assertIn("telegram_api_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
