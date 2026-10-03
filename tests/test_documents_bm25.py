"""BM25 ranking behaviour for nomorals.documents.DocumentIndex.

DocumentIndex ranks with SQLite FTS5 ``bm25()`` (via
:mod:`nomorals.storage.fts`) instead of raw term frequency.  These tests
pin the behaviours raw TF gets wrong:

* IDF weighting — a document matching a *rare* query term outranks a
  document spamming a *common* query term, even when the spam document
  has the higher raw term count.
* Length normalisation — with equal term frequency, the shorter document
  wins instead of tying.

They also pin the preserved search contract: OR semantics, exact-term
matching, title terms indexed, float scores, deterministic tie-breaking,
and fail-fast (never silent empty results) when FTS5 is unavailable.
"""

from __future__ import annotations

import unittest
from unittest import mock

from nomorals.documents import Document, DocumentError, DocumentIndex, Section


def _doc(doc_id: str, text: str, title: str = "") -> Document:
    return Document(id=doc_id, title=title or doc_id,
                    sections=[Section(level=1, heading="", text=text)])


class TestBM25Ranking(unittest.TestCase):
    def test_rare_term_match_beats_common_term_spam(self) -> None:
        # Raw TF scores this: spam = 30 ("nebula" x30), precise = 2
        # ("quasar" x2) -> spam wins.  BM25's IDF must flip it: "quasar" is
        # rare (2 of 7 docs), "nebula" is common (4 of 7), so the precise
        # document ranks first despite the lower raw count.
        index = DocumentIndex()
        spam = _doc("spam", "nebula " * 30)
        precise = _doc("precise", "quasar quasar")
        for doc in (spam, precise,
                    _doc("n1", "nebula alpha"),
                    _doc("n2", "nebula beta"),
                    _doc("n3", "nebula gamma"),
                    _doc("f1", "delta epsilon"),
                    _doc("f2", "eta theta")):
            index.add(doc)
        hits = index.search("quasar nebula")
        self.assertGreater(len(hits), 1)
        self.assertEqual(hits[0]["doc_id"], "precise")
        # The spam document still matches (OR semantics) but ranks below.
        self.assertIn("spam", [h["doc_id"] for h in hits[1:]])

    def test_length_normalisation_equal_tf_shorter_wins(self) -> None:
        # Identical term frequency for "quasar"; raw TF ties (and the old
        # code broke the tie on doc id).  BM25 length normalisation must
        # rank the concise document first regardless of id order.
        index = DocumentIndex()
        index.add(_doc("zzz-long", "quasar " + "filler " * 400))
        index.add(_doc("aaa-short", "quasar " + "filler " * 3))
        hits = index.search("quasar")
        self.assertEqual([h["doc_id"] for h in hits],
                         ["aaa-short", "zzz-long"])

    def test_scores_are_bm25_floats(self) -> None:
        index = DocumentIndex()
        index.add(_doc("d1", "apple apple apple"))
        index.add(_doc("d2", "apple"))
        hits = index.search("apple")
        for hit in hits:
            self.assertIsInstance(hit["score"], float)
        self.assertGreater(hits[0]["score"], hits[1]["score"])

    def test_or_semantics_any_term_matches(self) -> None:
        index = DocumentIndex()
        index.add(_doc("q", "quasar"))
        index.add(_doc("n", "nebula"))
        hits = index.search("quasar nebula")
        self.assertEqual({h["doc_id"] for h in hits}, {"q", "n"})

    def test_exact_terms_no_prefix_expansion(self) -> None:
        # The old index matched whole tokens only; "quantum" must not match
        # a document containing only "quanta".
        index = DocumentIndex()
        index.add(_doc("d", "quanta mechanics overview"))
        self.assertEqual(index.search("quantum"), [])

    def test_title_terms_are_searchable(self) -> None:
        index = DocumentIndex()
        index.add(_doc("d", "body without the word", title="Quasar Report"))
        hits = index.search("quasar")
        self.assertEqual([h["doc_id"] for h in hits], ["d"])

    def test_tie_break_by_doc_id_is_deterministic(self) -> None:
        index = DocumentIndex()
        index.add(_doc("doc-b", "identical text here"))
        index.add(_doc("doc-a", "identical text here"))
        first = [h["doc_id"] for h in index.search("identical")]
        second = [h["doc_id"] for h in index.search("identical")]
        self.assertEqual(first, ["doc-a", "doc-b"])
        self.assertEqual(first, second)

    def test_limit_respected(self) -> None:
        index = DocumentIndex()
        for i in range(5):
            index.add(_doc(f"d{i}", f"common word doc{i}"))
        self.assertEqual(len(index.search("common", limit=3)), 3)

    def test_readd_replaces_entry(self) -> None:
        index = DocumentIndex()
        index.add(_doc("d", "quasar quasar quasar"))
        index.add(_doc("d", "nebula"))
        self.assertEqual(len(index), 1)
        self.assertEqual(index.search("quasar"), [])
        self.assertEqual([h["doc_id"] for h in index.search("nebula")], ["d"])


class TestFTS5Unavailable(unittest.TestCase):
    def test_construction_fails_fast_without_fts5(self) -> None:
        # The index must never silently degrade to empty results: without
        # FTS5, construction raises DocumentError immediately.
        with mock.patch("nomorals.documents.index._fts5_available",
                        return_value=False):
            with self.assertRaises(DocumentError):
                DocumentIndex()


if __name__ == "__main__":
    unittest.main()


class TestDocumentIndexedEvent(unittest.TestCase):
    def test_add_emits_document_indexed(self) -> None:
        from nomorals.core.events import global_bus

        seen: list = []
        sub_id = global_bus.subscribe("document.indexed", seen.append, sync=True)
        try:
            index = DocumentIndex()
            index.add(_doc("d1", "quasar physics", title="Quasar Report"))
        finally:
            global_bus.unsubscribe(sub_id)
        self.assertEqual(len(seen), 1)
        event = seen[0]
        self.assertEqual(event.topic, "document.indexed")
        self.assertEqual(event.data["doc_id"], "d1")
        self.assertEqual(event.data["title"], "Quasar Report")

    def test_add_succeeds_when_bus_is_broken(self) -> None:
        # Fail-open telemetry: a broken bus must never break indexing.
        index = DocumentIndex()
        with mock.patch("nomorals.documents.index.global_bus") as bus:
            bus.publish.side_effect = RuntimeError("bus down")
            index.add(_doc("d1", "quasar physics"))
        self.assertEqual(len(index), 1)
        self.assertEqual([h["doc_id"] for h in index.search("quasar")], ["d1"])
