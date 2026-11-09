"""WS1 P4 — artifact retrieval surface: body search, supersede, stats.

ArtifactStore could always find artifacts by facet; these tests pin
the retrieval surface the census found missing:

* full-text search over artifact BODIES through the repo's shared
  FTS organ (storage/fts.py) — ranked hits with snippets;
* document-format bodies extracted through the documents engine
  (injected as the store's text extractor — the store is L2 and
  cannot import the L4 engine);
* honest accounting: every body is indexed, superseded, or skipped
  BY KIND — nothing is silently absent;
* the atomic supersede verb: the new version is searchable, the old
  version's text no longer matches, lineage keeps the chain;
* delete removes the body from search (and works on a store whose
  alias table was never created — the pre-existing delete defect).
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from nomorals.documents import artifact_body_text, to_xlsx
from nomorals.documents.model import Table, new_document
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database

XLSX_MIME = ("application/vnd.openxmlformats-officedocument"
             ".spreadsheetml.sheet")
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR not really a png"


def make_store(test=None, *, documents: bool = False):
    tmp = tempfile.mkdtemp()
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(":memory:")
    db.migrate()
    blobs = BlobStore(db, Path(tmp) / "blobs")
    kwargs = {"text_extractor": artifact_body_text} if documents else {}
    return ArtifactStore(db, blobs, **kwargs), db


def xlsx_bytes(word: str) -> bytes:
    doc = new_document(format="xlsx", title="sheet")
    doc.tables.append(Table(name="Data", headers=["word"],
                            rows=[[word]]))
    return to_xlsx(doc)


class ContentSearchTest(unittest.TestCase):
    def setUp(self):
        self.store, self.db = make_store(self, documents=True)

    def tearDown(self):
        self.db.close()

    def test_put_text_is_searchable_with_snippet(self):
        art = self.store.put_text("the quarterly zebrafish report",
                                  type="report", creator="analyst",
                                  metadata={"title": "Q3 report"})
        hits = self.store.search_content("zebrafish")
        self.assertEqual([h.artifact.id for h in hits], [art.id])
        self.assertIn("zebrafish", hits[0].snippet)
        self.assertGreater(hits[0].score, 0)

    def test_put_json_is_searchable(self):
        art = self.store.put_json({"finding": "plankton bloom"},
                                  type="json")
        hits = self.store.search_content("plankton")
        self.assertEqual([h.artifact.id for h in hits], [art.id])

    def test_document_body_searchable_via_engine(self):
        # An .xlsx body is a zip of XML: the word lives in no raw
        # text run a plain decoder could read — only the documents
        # engine's parse makes it searchable.
        art = self.store.put(xlsx_bytes("zebrafish"), type="file",
                             mime=XLSX_MIME,
                             metadata={"filename": "data.xlsx"})
        hits = self.store.search_content("zebrafish")
        self.assertEqual([h.artifact.id for h in hits], [art.id])

    def test_html_body_searchable_via_engine(self):
        # The word is split by an HTML entity in the raw bytes; the
        # engine's extracted text carries it whole.
        art = self.store.put(
            b"<html><body><p>zebra&#102;ish melon</p></body></html>",
            type="file", mime="text/html",
            metadata={"filename": "page.html"})
        hits = self.store.search_content("zebrafish")
        self.assertEqual([h.artifact.id for h in hits], [art.id])

    def test_no_match_and_empty_query(self):
        self.store.put_text("something else entirely")
        self.assertEqual(self.store.search_content("zebrafish"), [])
        self.assertEqual(self.store.search_content(""), [])
        self.assertEqual(self.store.search_content("   "), [])

    def test_title_tag_reindexes(self):
        art = self.store.put_text("body without the word",
                                  metadata={"title": "plain title"})
        self.assertEqual(self.store.search_content("quokka"), [])
        self.store.tag(art.id, "title", "the quokka file")
        hits = self.store.search_content("quokka")
        self.assertEqual([h.artifact.id for h in hits], [art.id])


class AccountingTest(unittest.TestCase):
    def setUp(self):
        self.store, self.db = make_store(self, documents=True)

    def tearDown(self):
        self.db.close()

    def test_nontextual_bodies_skipped_by_kind(self):
        self.store.put_text("a searchable text body")
        self.store.put(PNG_BYTES, type="image", mime="image/png")
        stats = self.store.content_index_stats()
        self.assertTrue(stats["available"])
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["indexed"], 1)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["skipped_kinds"], {"image/png": 1})
        # The census identity: nothing unaccounted for.
        self.assertEqual(
            stats["total"],
            stats["indexed"] + stats["skipped"] + stats["superseded"])
        # And the binary body truly does not match text it contains.
        self.assertEqual(self.store.search_content("IHDR"), [])

    def test_default_extractor_skips_document_formats_by_kind(self):
        store, db = make_store(self)  # no documents extractor wired
        self.addCleanup(db.close)
        store.put(xlsx_bytes("zebrafish"), type="file", mime=XLSX_MIME,
                  metadata={"filename": "data.xlsx"})
        store.put_text("plain zebrafish text")
        stats = store.content_index_stats()
        self.assertEqual(stats["indexed"], 1)
        self.assertEqual(stats["skipped_kinds"], {"xlsx": 1})
        hits = store.search_content("zebrafish")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].artifact.type, "text")

    def test_stronger_extractor_backfills_on_search(self):
        # Written through a plain store (document body skipped),
        # found through a documents-wired store over the same db:
        # the index reconciles on search, no re-put needed.
        plain, db = make_store(self)
        self.addCleanup(db.close)
        art = plain.put(xlsx_bytes("zebrafish"), type="file",
                        mime=XLSX_MIME, metadata={"filename": "d.xlsx"})
        self.assertEqual(plain.search_content("zebrafish"), [])
        wired = ArtifactStore(db, plain.blobs,
                              text_extractor=artifact_body_text)
        hits = wired.search_content("zebrafish")
        self.assertEqual([h.artifact.id for h in hits], [art.id])


class SupersedeTest(unittest.TestCase):
    def setUp(self):
        self.store, self.db = make_store(self, documents=True)

    def tearDown(self):
        self.db.close()

    def test_supersede_replaces_search_presence(self):
        old = self.store.put_text("version one mentions zebrafish",
                                  type="report", creator="analyst",
                                  mission_id="m1",
                                  metadata={"title": "v1"})
        new = self.store.supersede(
            old.id, b"version two mentions giraffes",
            metadata={"title": "v2"})
        # New text matches, old text no longer does.
        self.assertEqual(
            [h.artifact.id for h in self.store.search_content("giraffes")],
            [new.id])
        self.assertEqual(self.store.search_content("zebrafish"), [])
        # The old artifact still exists and reads back byte-identical:
        # supersede replaces search presence, not history.
        self.assertEqual(self.store.read_text(old.id),
                         "version one mentions zebrafish")
        # Lineage carries the chain both ways.
        self.assertIn(old.id, new.provenance.supersedes)
        self.assertIn(old.id, new.provenance.derived_from)
        self.assertEqual([a.id for a in self.store.superseded_by(old.id)],
                         [new.id])
        self.assertIn(old.id, [a.id for a in self.store.lineage(new.id)])
        # Defaults inherited from the replaced version.
        self.assertEqual(new.type, "report")
        self.assertEqual(new.creator, "analyst")
        self.assertEqual(new.mission_id, "m1")
        self.assertEqual(new.metadata["title"], "v2")
        # Accounting: one indexed (new), one superseded (old).
        stats = self.store.content_index_stats()
        self.assertEqual((stats["indexed"], stats["superseded"]), (1, 1))
        self.assertEqual(
            stats["total"],
            stats["indexed"] + stats["skipped"] + stats["superseded"])

    def test_supersede_chain_only_latest_matches(self):
        v1 = self.store.put_text("alpha one")
        v2 = self.store.supersede(v1.id, b"bravo two")
        v3 = self.store.supersede(v2.id, b"charlie three")
        self.assertEqual(self.store.search_content("alpha"), [])
        self.assertEqual(self.store.search_content("bravo"), [])
        self.assertEqual(
            [h.artifact.id for h in self.store.search_content("charlie")],
            [v3.id])
        stats = self.store.content_index_stats()
        self.assertEqual((stats["indexed"], stats["superseded"]), (1, 2))

    def test_supersede_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.store.supersede("no-such-id", b"data")

    def test_by_convention_path_still_works(self):
        # derive + a hand-written supersedes link (the pre-verb
        # convention) is honored by search exactly like the verb.
        old = self.store.put_text("convention zebrafish body")
        new = self.store.derive(
            b"convention replacement body", from_ids=[old.id],
            provenance={"supersedes": [old.id]})
        self.assertEqual(self.store.search_content("zebrafish"), [])
        self.assertEqual(
            [h.artifact.id
             for h in self.store.search_content("replacement")],
            [new.id])


class DeleteConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.store, self.db = make_store(self, documents=True)

    def tearDown(self):
        self.db.close()

    def test_delete_removes_body_from_search(self):
        art = self.store.put_text("ephemeral zebrafish body")
        self.assertEqual(len(self.store.search_content("zebrafish")), 1)
        self.assertTrue(self.store.delete(art.id))
        self.assertEqual(self.store.search_content("zebrafish"), [])
        stats = self.store.content_index_stats()
        self.assertEqual(stats["total"], 0)

    def test_delete_on_store_without_alias_table(self):
        # Regression pin: delete() removed aliases without ensuring
        # the lazily-created alias table exists, so deleting from a
        # store that never set an alias raised "no such table".
        art = self.store.put_text("deletable body")
        self.assertTrue(self.store.delete(art.id))
        self.assertIsNone(self.store.get(art.id))


if __name__ == "__main__":
    unittest.main()
