"""Wave-2 burn-down: error-kind classification sweep for the documents
package's remaining unclassified organs — ``index.py``, ``compare.py``,
``summarize.py``.

Every ``DocumentError`` raise in those three modules now carries an
explicit taxonomy ``kind`` (messages byte-identical; kind metadata
only).  This suite proves it two ways:

* an AST closure test — every raise site in the three modules names a
  ``kind=`` from :data:`DOCUMENT_ERROR_KINDS`, so a future unclassified
  raise fails here, not downstream;
* per-site behavioural tests — each classified raise is *triggered*
  and its ``kind`` (and the mirrored ``details["kind"]``) asserted.

The sweep also added one taxonomy kind: ``upstream`` — a backing
engine/service that was present but failed or vanished at runtime
(FTS table/backend loss, facet storage failure, LLM completion
failure), distinct from ``missing_dependency`` (absent).
"""

from __future__ import annotations

import ast
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import importlib

# importlib, not `import nomorals.documents.summarize as ...`: the
# package __init__ rebinds the `summarize` attribute to the exported
# function, shadowing the submodule on attribute access.
compare_mod = importlib.import_module("nomorals.documents.compare")
index_mod = importlib.import_module("nomorals.documents.index")
summarize_mod = importlib.import_module("nomorals.documents.summarize")
from nomorals.core.errors import StorageError
from nomorals.documents import DOCUMENT_ERROR_KINDS, DocumentError
from nomorals.documents.compare import (
    compare_documents, diff_documents, similarity, word_diff)
from nomorals.documents.index import DocumentIndex
from nomorals.documents.model import Document, Section
from nomorals.documents.summarize import (
    bullet_digest, keyphrases, keywords, summarize, summarize_abstractive,
    summarize_query, summarize_sections, summarize_text, tldr)


def _doc(text: str = "", *, doc_id: str = "d1", title: str = "") -> Document:
    doc = Document(id=doc_id, title=title)
    if text:
        doc.sections = [Section(level=1, heading="", text=text)]
    return doc


def _kind_of(case: unittest.TestCase, fn) -> DocumentError:
    with case.assertRaises(DocumentError) as caught:
        fn()
    err = caught.exception
    # the Wave-2 contract, unchanged: classified errors still catch as
    # ValueError, and the kind is mirrored into details
    case.assertIsInstance(err, ValueError)
    case.assertEqual(err.details["kind"], err.kind)
    return err


# ── taxonomy ────────────────────────────────────────────────────────────────

class TestTaxonomyExtension(unittest.TestCase):
    def test_upstream_kind_registered(self):
        self.assertIn("upstream", DOCUMENT_ERROR_KINDS)
        err = DocumentError("backend died", kind="upstream")
        self.assertEqual(err.kind, "upstream")
        self.assertEqual(err.details["kind"], "upstream")

    def test_unknown_kind_still_rejected(self):
        with self.assertRaises(ValueError):
            DocumentError("x", kind="not-a-kind")

    def test_every_taxonomy_kind_constructible(self):
        for kind in DOCUMENT_ERROR_KINDS:
            self.assertEqual(DocumentError("x", kind=kind).kind, kind)


class TestSweepClosure(unittest.TestCase):
    """AST pin: no DocumentError raise in the three swept modules may
    go back to the unclassified default."""

    def test_all_raise_sites_carry_explicit_taxonomy_kind(self):
        for module in (index_mod, compare_mod, summarize_mod):
            source = Path(module.__file__).read_text(encoding="utf-8")
            tree = ast.parse(source)
            sites = 0
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Raise)
                        and isinstance(node.exc, ast.Call)):
                    continue
                func = node.exc.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name != "DocumentError":
                    continue
                sites += 1
                kinds = [kw.value.value for kw in node.exc.keywords
                         if kw.arg == "kind"
                         and isinstance(kw.value, ast.Constant)]
                self.assertEqual(
                    len(kinds), 1,
                    f"{module.__name__}:{node.lineno} has no explicit kind")
                self.assertIn(kinds[0], DOCUMENT_ERROR_KINDS)
            self.assertGreater(sites, 0)


# ── compare.py ──────────────────────────────────────────────────────────────

class TestCompareKinds(unittest.TestCase):
    def test_diff_needs_two_documents(self):
        err = _kind_of(self, lambda: diff_documents("nope", _doc("text")))
        self.assertEqual(err.kind, "invalid_argument")

    def test_diff_no_text_is_empty(self):
        err = _kind_of(self, lambda: diff_documents(_doc(), _doc("text")))
        self.assertEqual(err.kind, "empty")

    def test_word_diff_needs_two_strings(self):
        err = _kind_of(self, lambda: word_diff(42, "text"))
        self.assertEqual(err.kind, "invalid_argument")

    def test_similarity_needs_two_documents(self):
        err = _kind_of(self, lambda: similarity(None, _doc("text")))
        self.assertEqual(err.kind, "invalid_argument")

    def test_compare_needs_two_documents(self):
        err = _kind_of(self, lambda: compare_documents(_doc("text"), "nope"))
        self.assertEqual(err.kind, "invalid_argument")


# ── summarize.py ────────────────────────────────────────────────────────────

_TEXT = ("Gold rallied on Tuesday as the dollar weakened. "
         "Traders cited safe-haven demand for the move. "
         "Silver followed gold higher in late trade.")


class TestSummarizeKinds(unittest.TestCase):
    def test_unknown_method_is_invalid_argument(self):
        err = _kind_of(self, lambda: summarize_text(_TEXT, method="bogus"))
        self.assertEqual(err.kind, "invalid_argument")

    def test_nonpositive_sentences_is_invalid_argument(self):
        err = _kind_of(self, lambda: summarize_text(_TEXT, sentences=0))
        self.assertEqual(err.kind, "invalid_argument")

    def test_empty_text_is_empty(self):
        err = _kind_of(self, lambda: summarize_text("   "))
        self.assertEqual(err.kind, "empty")

    def test_no_summarizable_sentences_is_empty(self):
        with mock.patch.object(summarize_mod, "_sentences_or_fallback",
                               return_value=[]):
            err = _kind_of(self, lambda: summarize_text(_TEXT))
        self.assertEqual(err.kind, "empty")

    def test_summarize_needs_document(self):
        err = _kind_of(self, lambda: summarize("not a document"))
        self.assertEqual(err.kind, "invalid_argument")

    def test_summarize_sections_needs_document(self):
        err = _kind_of(self, lambda: summarize_sections(42))
        self.assertEqual(err.kind, "invalid_argument")

    def test_summarize_sections_nonpositive_sentences(self):
        err = _kind_of(self, lambda: summarize_sections(_doc(_TEXT),
                                                        sentences=0))
        self.assertEqual(err.kind, "invalid_argument")

    def test_summarize_sections_no_sections_is_empty(self):
        err = _kind_of(self, lambda: summarize_sections(_doc()))
        self.assertEqual(err.kind, "empty")

    def test_summarize_query_needs_query(self):
        err = _kind_of(self, lambda: summarize_query(_TEXT, "  "))
        self.assertEqual(err.kind, "invalid_argument")

    def test_tldr_nonpositive_words_is_invalid_argument(self):
        err = _kind_of(self, lambda: tldr(_TEXT, words=0))
        self.assertEqual(err.kind, "invalid_argument")

    def test_tldr_empty_text_is_empty(self):
        err = _kind_of(self, lambda: tldr(""))
        self.assertEqual(err.kind, "empty")

    def test_tldr_no_sentences_is_empty(self):
        with mock.patch.object(summarize_mod, "_sentences_or_fallback",
                               return_value=[]):
            err = _kind_of(self, lambda: tldr(_TEXT))
        self.assertEqual(err.kind, "empty")

    def test_bullet_digest_needs_document(self):
        err = _kind_of(self, lambda: bullet_digest("not a document"))
        self.assertEqual(err.kind, "invalid_argument")

    def test_keywords_nonpositive_top_n(self):
        err = _kind_of(self, lambda: keywords(_TEXT, top_n=0))
        self.assertEqual(err.kind, "invalid_argument")

    def test_keywords_empty_text_is_empty(self):
        err = _kind_of(self, lambda: keywords(""))
        self.assertEqual(err.kind, "empty")

    def test_keywords_no_candidates_is_empty(self):
        err = _kind_of(self, lambda: keywords("the and of the"))
        self.assertEqual(err.kind, "empty")

    def test_keyphrases_nonpositive_top_n(self):
        err = _kind_of(self, lambda: keyphrases(_TEXT, top_n=0))
        self.assertEqual(err.kind, "invalid_argument")

    def test_keyphrases_bad_max_words(self):
        err = _kind_of(self, lambda: keyphrases(_TEXT, max_words=0))
        self.assertEqual(err.kind, "invalid_argument")

    def test_keyphrases_empty_text_is_empty(self):
        err = _kind_of(self, lambda: keyphrases(""))
        self.assertEqual(err.kind, "empty")

    def test_keyphrases_no_candidates_is_empty(self):
        err = _kind_of(self, lambda: keyphrases("the and of the"))
        self.assertEqual(err.kind, "empty")


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


class _FakeBrain:
    def __init__(self, response=None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc

    def complete(self, prompt: str):
        if self._exc is not None:
            raise self._exc
        return self._response


class TestAbstractiveKinds(unittest.TestCase):
    def test_nonpositive_max_words_is_invalid_argument(self):
        err = _kind_of(self, lambda: summarize_abstractive(_TEXT, max_words=0))
        self.assertEqual(err.kind, "invalid_argument")

    def test_empty_text_is_empty(self):
        err = _kind_of(self, lambda: summarize_abstractive("  "))
        self.assertEqual(err.kind, "empty")

    def test_brain_unavailable_is_missing_dependency(self):
        with mock.patch("nomorals.llm.brain.get_brain",
                        side_effect=RuntimeError("no brain configured")):
            err = _kind_of(self, lambda: summarize_abstractive(_TEXT))
        self.assertEqual(err.kind, "missing_dependency")

    def test_brain_failure_is_upstream(self):
        brain = _FakeBrain(exc=RuntimeError("provider down"))
        with mock.patch("nomorals.llm.brain.get_brain", return_value=brain):
            err = _kind_of(self, lambda: summarize_abstractive(_TEXT))
        self.assertEqual(err.kind, "upstream")

    def test_empty_llm_summary_is_empty(self):
        brain = _FakeBrain(response=_FakeResponse(""))
        with mock.patch("nomorals.llm.brain.get_brain", return_value=brain):
            err = _kind_of(self, lambda: summarize_abstractive(_TEXT))
        self.assertEqual(err.kind, "empty")


# ── index.py ────────────────────────────────────────────────────────────────

class _DeadFts:
    """FTSIndex stand-in whose table vanished after the FTS5 probe."""

    def __init__(self, *args, **kwargs) -> None:
        self.available = False


class _StubSourceDb:
    """Database stand-in for _load_sqlite failure injection."""

    def __init__(self, *, table_exists=True, scalar_value="3",
                 scalar_exc: Exception | None = None,
                 query_exc: Exception | None = None) -> None:
        self._table_exists = table_exists
        self._scalar_value = scalar_value
        self._scalar_exc = scalar_exc
        self._query_exc = query_exc

    def table_exists(self, name: str) -> bool:
        if isinstance(self._table_exists, Exception):
            raise self._table_exists
        return self._table_exists

    def scalar(self, sql: str, default=None):
        if self._scalar_exc is not None:
            raise self._scalar_exc
        return self._scalar_value

    def query(self, sql: str, params=()):
        if self._query_exc is not None:
            raise self._query_exc
        return []

    def close(self) -> None:
        pass


class TestIndexConstructionKinds(unittest.TestCase):
    def test_fts5_absent_is_missing_dependency(self):
        with mock.patch.object(index_mod, "_fts5_available",
                               return_value=False):
            err = _kind_of(self, DocumentIndex)
        self.assertEqual(err.kind, "missing_dependency")

    def test_fts_table_vanished_is_upstream(self):
        with mock.patch.object(index_mod, "FTSIndex", _DeadFts):
            err = _kind_of(self, DocumentIndex)
        self.assertEqual(err.kind, "upstream")


class TestIndexAddKinds(unittest.TestCase):
    def test_add_without_id_is_invalid_argument(self):
        index = DocumentIndex()
        doc = _doc("some text")
        doc.id = ""
        err = _kind_of(self, lambda: index.add(doc))
        self.assertEqual(err.kind, "invalid_argument")

    def test_add_without_text_is_empty(self):
        index = DocumentIndex()
        err = _kind_of(self, lambda: index.add(_doc()))
        self.assertEqual(err.kind, "empty")


class TestIndexSearchKinds(unittest.TestCase):
    def setUp(self):
        self.index = DocumentIndex()
        self.index.add(_doc("gold prices rallied sharply today"))

    def test_search_no_terms_is_empty(self):
        err = _kind_of(self, lambda: self.index.search("!!!"))
        self.assertEqual(err.kind, "empty")

    def test_search_unbuildable_match_is_empty(self):
        # "title:x" tokenizes (title) but builds no MATCH expression
        err = _kind_of(self, lambda: self.index.search("title:x"))
        self.assertEqual(err.kind, "empty")

    def test_search_nonpositive_limit_is_invalid_argument(self):
        err = _kind_of(self, lambda: self.index.search("gold", limit=0))
        self.assertEqual(err.kind, "invalid_argument")

    def test_search_bad_operator_is_invalid_argument(self):
        err = _kind_of(self, lambda: self.index.search("gold", operator="XOR"))
        self.assertEqual(err.kind, "invalid_argument")

    def test_search_backend_lost_is_upstream(self):
        self.index._fts.available = False
        err = _kind_of(self, lambda: self.index.search("gold"))
        self.assertEqual(err.kind, "upstream")

    def test_search_bad_query_is_invalid_argument(self):
        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("fts5: syntax error")
        with mock.patch.object(self.index._db, "query", boom):
            err = _kind_of(self, lambda: self.index.search("gold"))
        self.assertEqual(err.kind, "invalid_argument")

    def test_count_no_terms_is_empty(self):
        err = _kind_of(self, lambda: self.index.count("!!!"))
        self.assertEqual(err.kind, "empty")

    def test_count_unbuildable_match_is_empty(self):
        err = _kind_of(self, lambda: self.index.count("title:x"))
        self.assertEqual(err.kind, "empty")

    def test_count_bad_query_is_invalid_argument(self):
        def boom(*args, **kwargs):
            raise StorageError("fts5: syntax error")
        with mock.patch.object(self.index._db, "scalar", boom):
            err = _kind_of(self, lambda: self.index.count("gold"))
        self.assertEqual(err.kind, "invalid_argument")

    def test_suggest_nonpositive_limit_is_invalid_argument(self):
        err = _kind_of(self, lambda: self.index.suggest("go", limit=0))
        self.assertEqual(err.kind, "invalid_argument")

    def test_facet_unknown_field_is_invalid_argument(self):
        err = _kind_of(self, lambda: self.index.facet("title"))
        self.assertEqual(err.kind, "invalid_argument")

    def test_facet_storage_failure_is_upstream(self):
        self.index._db.execute('DROP TABLE "documents_meta"')
        err = _kind_of(self, lambda: self.index.facet())
        self.assertEqual(err.kind, "upstream")


class TestIndexPersistenceKinds(unittest.TestCase):
    def setUp(self):
        self.index = DocumentIndex()
        self.index.add(_doc("gold prices rallied sharply today"))
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_save_to_directory_is_io(self):
        err = _kind_of(self, lambda: self.index.save(self.dir))
        self.assertEqual(err.kind, "io")

    def test_save_unlink_failure_is_io(self):
        target = self.dir / "index.db"
        target.write_bytes(b"old")
        with mock.patch.object(Path, "unlink",
                               side_effect=OSError("file locked")):
            err = _kind_of(self, lambda: self.index.save(target))
        self.assertEqual(err.kind, "io")

    def test_save_open_failure_is_io(self):
        target = self.dir / "index.db"
        with mock.patch.object(index_mod, "Database",
                               side_effect=OSError("cannot create")):
            err = _kind_of(self, lambda: self.index.save(target))
        self.assertEqual(err.kind, "io")

    def test_load_missing_file_is_io(self):
        err = _kind_of(self, lambda: DocumentIndex.load(self.dir / "nope.db"))
        self.assertEqual(err.kind, "io")

    def test_load_unreadable_file_is_io(self):
        target = self.dir / "idx.bin"
        target.write_bytes(b"whatever")
        real_open = open

        def flaky_open(file, *args, **kwargs):
            if str(file).endswith("idx.bin"):
                raise OSError("permission denied")
            return real_open(file, *args, **kwargs)

        with mock.patch("builtins.open", flaky_open):
            err = _kind_of(self, lambda: DocumentIndex.load(target))
        self.assertEqual(err.kind, "io")

    def test_load_corrupt_sqlite_is_io(self):
        target = self.dir / "bad.db"
        with mock.patch.object(
                index_mod, "Database",
                side_effect=sqlite3.DatabaseError("file is not a database")):
            target.write_bytes(index_mod._SQLITE_MAGIC + b"garbage")
            err = _kind_of(self, lambda: DocumentIndex.load(target))
        self.assertEqual(err.kind, "io")

    def test_load_schema_probe_failure_is_io(self):
        target = self.dir / "idx.db"
        target.write_bytes(index_mod._SQLITE_MAGIC + b"x")
        stub = _StubSourceDb(table_exists=StorageError("probe failed"))
        with mock.patch.object(index_mod, "Database", return_value=stub):
            err = _kind_of(self, lambda: DocumentIndex.load(target))
        self.assertEqual(err.kind, "io")

    def test_load_not_an_index_file_is_unsupported_format(self):
        target = self.dir / "plain.db"
        conn = sqlite3.connect(target)
        conn.execute("CREATE TABLE unrelated (x TEXT)")
        conn.commit()
        conn.close()
        err = _kind_of(self, lambda: DocumentIndex.load(target))
        self.assertEqual(err.kind, "unsupported_format")

    def test_load_version_probe_failure_is_io(self):
        target = self.dir / "idx.db"
        target.write_bytes(index_mod._SQLITE_MAGIC + b"x")
        stub = _StubSourceDb(scalar_exc=StorageError("version read failed"))
        with mock.patch.object(index_mod, "Database", return_value=stub):
            err = _kind_of(self, lambda: DocumentIndex.load(target))
        self.assertEqual(err.kind, "io")

    def test_load_unsupported_version_is_unsupported_format(self):
        target = self.dir / "idx.db"
        self.index.save(target)
        conn = sqlite3.connect(target)
        conn.execute('UPDATE "index_meta" SET "value" = ? WHERE "key" = ?',
                     ("99", "version"))
        conn.commit()
        conn.close()
        err = _kind_of(self, lambda: DocumentIndex.load(target))
        self.assertEqual(err.kind, "unsupported_format")

    def test_load_rows_failure_is_io(self):
        target = self.dir / "idx.db"
        target.write_bytes(index_mod._SQLITE_MAGIC + b"x")
        stub = _StubSourceDb(query_exc=StorageError("rows unreadable"))
        with mock.patch.object(index_mod, "Database", return_value=stub):
            err = _kind_of(self, lambda: DocumentIndex.load(target))
        self.assertEqual(err.kind, "io")

    def test_load_legacy_bad_json_is_io(self):
        target = self.dir / "legacy.json"
        target.write_text("{not json", encoding="utf-8")
        err = _kind_of(self, lambda: DocumentIndex.load(target))
        self.assertEqual(err.kind, "io")

    def test_load_legacy_wrong_version_is_unsupported_format(self):
        target = self.dir / "legacy.json"
        target.write_text(json.dumps({"version": 2, "docs": []}),
                          encoding="utf-8")
        err = _kind_of(self, lambda: DocumentIndex.load(target))
        self.assertEqual(err.kind, "unsupported_format")


if __name__ == "__main__":
    unittest.main()
