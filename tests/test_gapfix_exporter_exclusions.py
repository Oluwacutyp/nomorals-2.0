"""Gapfix: wire the Tier-3 exclusion filter into the owner-history export.

R3-71 shipped `nm train export-telegram` on top of `nomorals/exporter.py`,
but the harvest loop in `export_telegram` never applied the Tier-3 guards
that already live in the same module (`is_excluded_dialog` /
`filter_excluded_dialogs`, pinned by
tests/test_remaining_tier3.py::TestExporterExclusions). A dialog titled
e.g. "vault notes" would be harvested, scrubbed, and registered like any
other. These tests pin the consent rule as implemented:

* RULE: a dialog whose title matches a HARD_EXCLUSIONS marker is NEVER
  harvested — no message fetch, no registration, no counting toward the
  harvested totals — no matter what `--only-chats` says. The module's own
  Tier-3 header states it: the guards make sure such material can NEVER
  become training data, "no matter what `only_chats` says". There is no
  per-dialog opt-in override anywhere in the module; none is invented.
* Excluded dialogs do NOT consume the `max_dialogs` budget (a refusal is
  not a harvest).
* The skip is honest, never silent: `ExportStats.excluded` counts it, the
  CLI summary names it, and the JSON payload carries it.
* The PII scrub is a separate guard and still applies to harvested content.
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

from nomorals.cmdline.commands.train import _cmd_train
from nomorals.exporter import export_telegram
from nomorals.training.dataset import DatasetRegistry


class _FakeExportClient:
    """Stand-in for telethon.TelegramClient — yields canned dialogs/messages.

    Records which entities had messages fetched, so a test can prove an
    excluded dialog was never read at all, not merely dropped later.
    """

    dialogs: list = []
    messages: dict = {}
    fetched: list = []
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
        type(self).fetched.append(entity)
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


def _mum_entries() -> list:
    # Deliberately different wording from Alice's DM: identical windows
    # would collapse under the simhash dedup and the test would be
    # asserting against the dedup, not the exclusion filter.
    return [
        SimpleNamespace(raw_text="hey love, call me at 08031112233", out=False),
        SimpleNamespace(raw_text="will ring you after lunch", out=True),
        SimpleNamespace(raw_text="thanks dear, take care", out=False),
        SimpleNamespace(raw_text="talk soon, love you", out=True),
    ]


def _vault_entries() -> list:
    return [
        SimpleNamespace(raw_text="the vault code is hunter2", out=False),
        SimpleNamespace(raw_text="noted, keeping it safe", out=True),
    ]


class ExclusionHarvestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._fake_mod = types.ModuleType("telethon")
        self._fake_mod.TelegramClient = _FakeExportClient
        self._saved = sys.modules.get("telethon")
        sys.modules["telethon"] = self._fake_mod

        self._tmp = tempfile.TemporaryDirectory(prefix="nm-exporter-excl-")
        from nomorals.agents.context import build_context
        from nomorals.core.config import Settings

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
        _FakeExportClient.fetched = []
        _FakeExportClient.dialogs = [
            SimpleNamespace(id=101, title="Alice", entity="e101"),
            SimpleNamespace(id=102, title="vault notes", entity="e102"),
            SimpleNamespace(id=103, title="Partner chat", entity="e103"),
            SimpleNamespace(id=104, title="Mum", entity="e104"),
        ]
        _FakeExportClient.messages = {
            "e101": _dm_entries(),
            "e102": _vault_entries(),
            "e103": _vault_entries(),
            "e104": _mum_entries(),
        }

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)
        if self._saved is None:
            sys.modules.pop("telethon", None)
        else:
            sys.modules["telethon"] = self._saved
        self._tmp.cleanup()

    def test_excluded_dialogs_never_harvested(self) -> None:
        result = export_telegram(self.settings, db=self.context.db)

        self.assertEqual(result.stats.excluded, 2)
        self.assertEqual(result.stats.dialogs, 2)  # Alice + Mum only
        self.assertEqual(result.stats.messages, 8)  # 4 + 4, nothing from excluded
        self.assertEqual(result.stats.examples, 2)
        # The excluded dialogs were never even read: no message fetch.
        self.assertNotIn("e102", _FakeExportClient.fetched)
        self.assertNotIn("e103", _FakeExportClient.fetched)
        self.assertIn("e101", _FakeExportClient.fetched)
        self.assertIn("e104", _FakeExportClient.fetched)

        dataset = DatasetRegistry(self.context.db).get(result.dataset_id)
        rows = list(dataset.examples())
        self.assertEqual(len(rows), 2)
        self.assertEqual(sorted({row.source for row in rows}), ["Alice", "Mum"])
        blob = json.dumps([row.to_dict() for row in rows])
        self.assertNotIn("vault", blob)
        self.assertNotIn("hunter2", blob)
        lines = [json.loads(line) for line in Path(result.path).read_text().splitlines()]
        self.assertEqual(len(lines), 2)

    def test_pii_scrub_still_applies_to_harvested_content(self) -> None:
        result = export_telegram(self.settings, db=self.context.db)
        self.assertEqual(result.stats.pii_scrubbed, 2)  # one per harvested DM
        blob = Path(result.path).read_text()
        self.assertNotIn("08034567890", blob)
        self.assertNotIn("08031112233", blob)
        self.assertIn("[REDACTED_PHONE]", blob)

    def test_excluded_dialogs_do_not_consume_max_dialogs_budget(self) -> None:
        # Two excluded dialogs lead the list; a budget of 2 must still
        # harvest both good dialogs, not stop after the first good one.
        result = export_telegram(
            self.settings, db=self.context.db, max_dialogs=2)
        self.assertEqual(result.stats.excluded, 2)
        self.assertEqual(result.stats.dialogs, 2)
        self.assertEqual(result.stats.examples, 2)

    def test_only_chats_naming_excluded_chat_is_refused(self) -> None:
        # RULE: --only-chats cannot override a hard exclusion.
        result = export_telegram(
            self.settings, db=self.context.db, only_chats="102")
        self.assertEqual(result.stats.excluded, 1)
        self.assertEqual(result.stats.dialogs, 0)
        self.assertEqual(result.stats.messages, 0)
        self.assertEqual(result.stats.examples, 0)
        self.assertEqual(result.dataset_id, "")
        self.assertEqual(DatasetRegistry(self.context.db).list(), [])

    def test_excluded_count_in_stats_dict(self) -> None:
        result = export_telegram(self.settings, db=self.context.db)
        self.assertEqual(result.stats.to_dict()["excluded"], 2)


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


class ExportCommandExclusionTests(ExclusionHarvestTests):
    """Same fake dialogs, driven through the CLI branch."""

    def _run(self, **overrides: object) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_train(_args(**overrides), self.context)
        return rc, buf.getvalue()

    def test_cli_summary_names_excluded_count(self) -> None:
        rc, text = self._run(export_telegram=True)
        self.assertEqual(rc, 0, text)
        self.assertIn("2 dialogs skipped as excluded", text)
        self.assertIn("2 dialogs", text)
        self.assertIn("8 messages", text)
        self.assertIn("2 examples", text)

    def test_cli_only_chats_excluded_reports_skip_honestly(self) -> None:
        rc, text = self._run(export_telegram=True, only_chats="102")
        self.assertEqual(rc, 1, text)
        self.assertIn("1 dialog skipped as excluded", text)
        self.assertIn("no examples", text.lower())

    def test_cli_json_payload_carries_excluded(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_train(_args(export_telegram=True, json=True), self.context)
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["stats"]["excluded"], 2)


class BrainToolExclusionTests(unittest.TestCase):
    """The brain-tool path shares export_telegram, so it inherits the guard."""

    def test_tool_path_skips_excluded_dialogs(self) -> None:
        fake = types.ModuleType("telethon")
        fake.TelegramClient = _FakeExportClient
        saved = sys.modules.get("telethon")
        sys.modules["telethon"] = fake
        tmp = tempfile.TemporaryDirectory(prefix="nm-exporter-excl-tool-")
        try:
            from nomorals.agents.context import build_context
            from nomorals.core.config import Settings
            from nomorals.training.dataset import DatasetRegistry as _Reg

            _FakeExportClient.started = False
            _FakeExportClient.fetched = []
            _FakeExportClient.dialogs = [
                SimpleNamespace(id=101, title="Alice", entity="e101"),
                SimpleNamespace(id=102, title="vault notes", entity="e102"),
            ]
            _FakeExportClient.messages = {
                "e101": _dm_entries(),
                "e102": _vault_entries(),
            }
            with build_context(
                Settings(home=tmp.name),
                with_executor=False, with_router=False,
                with_memory=False, with_tools=True,
            ) as context:
                context.settings.chat.telegram_api_id = "123456"
                context.settings.chat.telegram_api_hash = "abcdef0123456789"
                outcome = context.tools.call("export_training_data")
                self.assertTrue(outcome.ok, outcome.error)
                value = outcome.value
                self.assertTrue(value["ok"], value)
                self.assertEqual(value["stats"]["stats"]["excluded"], 1)
                self.assertNotIn("e102", _FakeExportClient.fetched)
                dataset = _Reg(context.db).get(value["dataset_id"])
                rows = list(dataset.examples())
                self.assertEqual([row.source for row in rows], ["Alice"])
        finally:
            if saved is None:
                sys.modules.pop("telethon", None)
            else:
                sys.modules["telethon"] = saved
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
