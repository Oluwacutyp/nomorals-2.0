"""Gapfix R3-71: `nm train export-telegram` + `nm train --run --dataset <id>`.

The exporter module (`nomorals/exporter.py`) is complete and already
reachable through the brain tool registry (`export_training_data`,
pinned by test_infra_organs_wiring.py::test_aa14). What never existed
is the CLI surface the module docstring promises:
`nm train --run --dataset <id>` — no `--dataset` flag, and `_cmd_train`
never passes a dataset id to `SelfImprovementJob.run` (which accepts
one). These tests pin the wiring:

* the flags parse, and default off/empty;
* `nm train export-telegram` harvests (fake telethon, same pattern as
  tests/test_exporter.py) and registers a PII-scrubbed dataset, rc 0,
  naming the dataset id;
* `--no-register` writes the file only; `--only-chats` restricts;
* missing credentials / missing telethon fail honestly, rc 1;
* `--run --dataset <unknown>` refuses before any mission is created;
* `--run --dataset <real>` passes the id into SelfImprovementJob.run;
* plain `nm train` behaves exactly as before.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.cmdline import _parser
from nomorals.cmdline.commands.train import _cmd_train


def _args(**overrides: object) -> argparse.Namespace:
    ns = argparse.Namespace(
        backends=False, run=False, continual=False, force=False,
        quick=False, backend="", base_model="", json=False,
        export_telegram=False, max_dialogs=50, max_per_chat=300,
        only_chats="", no_register=False, dataset="",
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _context(home: str):
    from nomorals.agents.context import build_context
    from nomorals.core.config import Settings

    ctx = build_context(
        Settings(home=home),
        with_executor=False, with_router=False,
        with_memory=False, with_tools=False,
    )
    ctx.__enter__()
    return ctx


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
        SimpleNamespace(raw_text="thanks, see you", out=False),
        SimpleNamespace(raw_text="see you then, love", out=True),
    ]


class ExportParserTests(unittest.TestCase):
    def test_train_export_telegram_parses(self) -> None:
        args = _parser().parse_args(["train", "--export-telegram"])
        self.assertTrue(args.export_telegram)

    def test_train_export_options_parse(self) -> None:
        args = _parser().parse_args([
            "train", "--export-telegram",
            "--max-dialogs", "7", "--max-per-chat", "40",
            "--only-chats", "101,202", "--no-register",
        ])
        self.assertEqual(args.max_dialogs, 7)
        self.assertEqual(args.max_per_chat, 40)
        self.assertEqual(args.only_chats, "101,202")
        self.assertTrue(args.no_register)

    def test_train_dataset_parses(self) -> None:
        args = _parser().parse_args(["train", "--run", "--dataset", "ds-123"])
        self.assertTrue(args.run)
        self.assertEqual(args.dataset, "ds-123")

    def test_train_defaults_off(self) -> None:
        args = _parser().parse_args(["train"])
        self.assertFalse(args.export_telegram)
        self.assertFalse(args.no_register)
        self.assertEqual(args.max_dialogs, 50)
        self.assertEqual(args.max_per_chat, 300)
        self.assertEqual(args.only_chats, "")
        self.assertEqual(args.dataset, "")

    def test_train_export_shows_in_help(self) -> None:
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as caught, redirect_stdout(buf):
            _parser().parse_args(["train", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--export-telegram", buf.getvalue())
        self.assertIn("--dataset", buf.getvalue())


class ExportCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._fake_mod = types.ModuleType("telethon")
        self._fake_mod.TelegramClient = _FakeExportClient
        self._saved = sys.modules.get("telethon")
        sys.modules["telethon"] = self._fake_mod

        self._tmp = tempfile.TemporaryDirectory(prefix="nm-gapfix71-")
        self.context = _context(self._tmp.name)
        self.settings = self.context.settings
        self.settings.chat.telegram_api_id = "123456"
        self.settings.chat.telegram_api_hash = "abcdef0123456789"

        _FakeExportClient.started = False
        _FakeExportClient.dialogs = [
            SimpleNamespace(id=101, title="Alice", entity="e101"),
            SimpleNamespace(id=202, title="Naija Tech News", entity="e202"),
        ]
        _FakeExportClient.messages = {
            "e101": _dm_entries(),
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

    def _run(self, **overrides: object) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_train(_args(**overrides), self.context)
        return rc, buf.getvalue()

    def test_export_registers_scrubbed_dataset(self) -> None:
        from nomorals.training.dataset import DatasetRegistry

        rc, text = self._run(export_telegram=True)
        self.assertEqual(rc, 0, text)
        self.assertTrue(_FakeExportClient.started)
        # 2 dialogs read (DM + channel), 6 messages read; only the DM
        # window qualifies as an example.
        self.assertIn("2 dialogs", text)
        self.assertIn("6 messages", text)
        self.assertIn("1 example", text)
        self.assertIn("dataset", text)

        registry = DatasetRegistry(self.context.db)
        datasets = registry.list()
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0].metadata.get("source"), "telegram-export")
        self.assertIn(datasets[0].id, text)
        rows = list(datasets[0].examples())
        self.assertEqual(len(rows), 1)
        blob = json.dumps([row.to_dict() for row in rows])
        self.assertNotIn("08034567890", blob)
        self.assertIn("[REDACTED_PHONE]", blob)

    def test_export_no_register_writes_file_only(self) -> None:
        from nomorals.training.dataset import DatasetRegistry

        rc, text = self._run(export_telegram=True, no_register=True)
        self.assertEqual(rc, 0, text)
        self.assertEqual(DatasetRegistry(self.context.db).list(), [])
        self.assertIn(".jsonl", text)

    def test_export_only_chats_restricts(self) -> None:
        rc, text = self._run(export_telegram=True, only_chats="202")
        # The only selected dialog is a channel nobody replied in:
        # nothing to harvest, reported honestly, rc 1.
        self.assertEqual(rc, 1, text)
        self.assertIn("no examples", text.lower())

    def test_export_missing_credentials_fails_honestly(self) -> None:
        self.settings.chat.telegram_api_id = ""
        rc, text = self._run(export_telegram=True)
        self.assertEqual(rc, 1)
        self.assertIn("telegram_api_id", text)

    def test_export_missing_telethon_fails_honestly(self) -> None:
        sys.modules.pop("telethon", None)
        rc, text = self._run(export_telegram=True)
        self.assertEqual(rc, 1)
        self.assertIn("telethon", text)

    def test_plain_train_unchanged(self) -> None:
        rc, text = self._run()
        self.assertEqual(rc, 0)
        self.assertIn("nothing to do", text)


class DatasetFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-gapfix71-ds-")
        self.context = _context(self._tmp.name)

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)
        self._tmp.cleanup()

    def _run(self, **overrides: object) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_train(_args(**overrides), self.context)
        return rc, buf.getvalue()

    def test_run_unknown_dataset_refuses(self) -> None:
        rc, text = self._run(run=True, dataset="no-such-dataset")
        self.assertEqual(rc, 1)
        self.assertIn("no-such-dataset", text)
        self.assertIn("no dataset", text.lower())

    def test_run_dataset_passed_to_job(self) -> None:
        from nomorals.self_improvement import TrainingJobResult
        from nomorals.training.dataset import DatasetRegistry, Example, Turn

        dataset = DatasetRegistry(self.context.db).register_examples(
            "seeded", [Example(turns=[Turn("user", "hi"), Turn("assistant", "hey")],
                               source="seed")],
            str(Path(self._tmp.name) / "data" / "training"),
            kind="chat",
        )
        seen: dict = {}

        class _FakeJob:
            def __init__(self, context) -> None:
                seen["context"] = context

            def run(self, **kwargs):
                seen.update(kwargs)
                return TrainingJobResult(status="done", run_id="r1",
                                         dataset_id=kwargs.get("dataset_id", ""))

        with mock.patch(
            "nomorals.self_improvement.SelfImprovementJob", _FakeJob
        ):
            rc, text = self._run(run=True, dataset=dataset.id)
        self.assertEqual(rc, 0, text)
        self.assertEqual(seen.get("dataset_id"), dataset.id)
        self.assertIn("r1", text)


if __name__ == "__main__":
    unittest.main()
