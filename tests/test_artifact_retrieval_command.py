"""WS1 P4 — /artifact retrieval surface on chat: search, convert, export.

Pins the chat half of the artifact retrieval surface:

* ``/artifact search <words>`` — full-text search over artifact
  bodies with the index census in the reply (indexed / skipped by
  kind / superseded — nothing silently absent);
* ``/artifact send <ref> as <format>`` — send-time conversion
  through the documents engine: the delivered bytes are converted,
  the reply names the conversion, the stored original is untouched,
  and unsupported conversions refuse in words with the supported
  list, sending nothing;
* ``/artifact export`` — one artifact or a facet-filtered set as a
  bundle over the EXISTING ArtifactStore bundle export (the bundle
  re-imports cleanly through the existing import).
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.documents import to_xlsx
from nomorals.documents.model import Table, new_document
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _msg, _RecordingGateway  # noqa: E402

XLSX_MIME = ("application/vnd.openxmlformats-officedocument"
             ".spreadsheetml.sheet")


def _xlsx_bytes(word: str) -> bytes:
    doc = new_document(format="xlsx", title="sheet")
    doc.tables.append(Table(name="Data", headers=["word"],
                            rows=[[word]]))
    return to_xlsx(doc)


class ArtifactRetrievalCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-artifact-p4-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.tools = _FakeTools()
        self.context.tools = self.tools
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        db = self.context.db
        blob_dir = Path(db.path).parent / "blobs"
        self.store = ArtifactStore(db, BlobStore(db, blob_dir))
        self.report = self.store.put_text(
            "# Q3 report\n\nthe zebrafish revenue grew.", type="report",
            creator="analyst", metadata={"title": "Q3 report"})
        self.sheet = self.store.put(
            _xlsx_bytes("plankton"), type="file", mime=XLSX_MIME,
            creator="datasci", metadata={"filename": "data.xlsx"})
        self.store.set_alias("latest-report", self.report.id)

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str) -> str:
        return self.runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, chat_key="telegram:999"))

    # ── search ──────────────────────────────────────────────────────
    def test_search_finds_text_body_with_census(self) -> None:
        out = self._run("/artifact search zebrafish")
        self.assertIn(self.report.id, out)
        self.assertIn("zebrafish", out)
        self.assertIn("content index:", out)
        self.assertIn("indexed", out)

    def test_search_finds_document_body(self) -> None:
        # Seeded through a plain store (no documents extractor): the
        # chat store's wired extractor backfills the .xlsx body.
        out = self._run("/artifact search plankton")
        self.assertIn(self.sheet.id, out)

    def test_search_no_match_is_honest(self) -> None:
        out = self._run("/artifact search none-such-word")
        self.assertIn("no artifact bodies match", out)
        self.assertIn("content index:", out)

    def test_search_usage(self) -> None:
        out = self._run("/artifact search")
        self.assertIn("usage: /artifact search", out)

    # ── send-time conversion ────────────────────────────────────────
    def test_send_as_pdf_converts_and_names_it(self) -> None:
        before = self.store.read(self.report.id)
        out = self._run(f"/artifact send {self.report.id} as pdf")
        self.assertIn("converted to pdf", out)
        self.assertIn("original is unchanged", out)
        self.assertEqual(len(self.tools.calls), 1)
        name, kwargs = self.tools.calls[0]
        self.assertEqual(name, "file_send")
        self.assertTrue(kwargs["path"].endswith(".pdf"))
        sent = Path(kwargs["path"]).read_bytes()
        self.assertTrue(sent.startswith(b"%PDF"))
        self.assertIn("converted to pdf", kwargs["caption"])
        # The stored artifact is byte-identical afterwards.
        self.assertEqual(self.store.read(self.report.id), before)

    def test_send_as_html_via_alias(self) -> None:
        out = self._run("/artifact send latest-report as html")
        self.assertIn("converted to html", out)
        _, kwargs = self.tools.calls[0]
        sent = Path(kwargs["path"]).read_bytes()
        self.assertIn(b"zebrafish", sent)

    def test_send_as_unsupported_format_refuses_in_words(self) -> None:
        out = self._run(f"/artifact send {self.report.id} as webp")
        self.assertIn("unsupported format 'webp'", out)
        self.assertIn("markdown", out)  # the supported list is named
        self.assertIn("pdf", out)
        self.assertEqual(self.tools.calls, [])

    def test_send_as_csv_without_table_refuses_in_words(self) -> None:
        out = self._run(f"/artifact send {self.report.id} as csv")
        self.assertIn("couldn't convert", out)
        self.assertIn("csv", out)
        self.assertEqual(self.tools.calls, [])

    # ── export ──────────────────────────────────────────────────────
    def test_export_single_reimports_cleanly(self) -> None:
        out = self._run(f"/artifact export {self.report.id}")
        self.assertIn("artifact bundle exported: 1 artifact(s)", out)
        self.assertEqual(len(self.tools.calls), 1)
        _, kwargs = self.tools.calls[0]
        bundle = Path(kwargs["path"])
        self.assertTrue(bundle.name.endswith(".tar.gz"))
        self.assertIn("1 artifact", kwargs["caption"])
        # The bundle rides the existing export/import pair: a fresh
        # store imports it and reads the same bytes back.
        db2 = Database(":memory:")
        db2.migrate()
        self.addCleanup(db2.close)
        store2 = ArtifactStore(
            db2, BlobStore(db2, Path(self.tmp.name) / "blobs2"))
        imported = store2.import_bundle(bundle)
        self.assertEqual(len(imported), 1)
        self.assertEqual(store2.read(imported[0].id),
                         self.store.read(self.report.id))

    def test_export_filtered_set(self) -> None:
        out = self._run("/artifact export type:report")
        self.assertIn("artifact bundle exported: 1 artifact(s)", out)
        self.assertIn("type:report", out)
        _, kwargs = self.tools.calls[0]
        self.assertIn("1 artifact", kwargs["caption"])
        db2 = Database(":memory:")
        db2.migrate()
        self.addCleanup(db2.close)
        store2 = ArtifactStore(
            db2, BlobStore(db2, Path(self.tmp.name) / "blobs3"))
        imported = store2.import_bundle(Path(kwargs["path"]))
        self.assertEqual([a.type for a in imported], ["report"])

    def test_export_no_match_is_honest(self) -> None:
        out = self._run("/artifact export type:video")
        self.assertIn("no artifacts match type:video", out)
        self.assertEqual(self.tools.calls, [])

    def test_export_bare_is_usage(self) -> None:
        out = self._run("/artifact export")
        self.assertIn("usage: /artifact export", out)
        self.assertEqual(self.tools.calls, [])

    def test_export_unknown_ref(self) -> None:
        out = self._run("/artifact export no-such-thing")
        self.assertIn("unknown artifact", out)
        self.assertEqual(self.tools.calls, [])


if __name__ == "__main__":
    unittest.main()
