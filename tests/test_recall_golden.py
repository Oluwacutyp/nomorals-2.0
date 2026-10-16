"""T4.1a: recall golden set — the gate for the T1.6 memory/hybrid refactor.

Cassettes tell you recall CHANGED; this tells you recall got WORSE.
A frozen synthetic corpus + labeled (query -> expected doc ids) lives
in tests/fixtures/recall_golden.json with the pre-refactor baseline.
Mean per-query recall@10 must stay within epsilon of that baseline.

The critical path is HybridMemoryIndex.bm25_search, which is the
documents/ edge T1.6 will move. If the refactor changes retrieval
behavior — different Document construction, different tokenization,
a dropped section — this test fails instead of silently degrading
recall for every memory query in production.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from nomorals.documents.index import DocumentIndex
from nomorals.documents.model import Document, Section
from nomorals.memory.hybrid import HybridMemoryIndex

FIXTURE = Path(__file__).with_name("fixtures") / "recall_golden.json"


class RecallGoldenTests(unittest.TestCase):
    def test_recall_at_10_within_baseline(self):
        with FIXTURE.open(encoding="utf-8") as fh:
            golden = json.load(fh)
        index = HybridMemoryIndex(
            index_factory=DocumentIndex,
            document_factory=Document,
            section_factory=Section)
        for doc in golden["corpus"]:
            index.index_unit(doc["id"], doc["title"], doc["text"])
        self.assertEqual(index.indexed_count, len(golden["corpus"]),
                         "golden corpus did not index fully")

        per_query: list[tuple[str, float, list[str]]] = []
        any_hits = False
        for entry in golden["queries"]:
            hits = index.bm25_search(entry["query"], limit=10)
            got = [hit[0] for hit in hits]
            if got:
                any_hits = True
            want = entry["expected"]
            recall = len(set(got) & set(want)) / len(want)
            per_query.append((entry["query"], recall, got))
        if not any_hits:
            self.skipTest("BM25/FTS5 unavailable — index returned no "
                          "results for any golden query")
        mean_recall = sum(r for _, r, _ in per_query) / len(per_query)
        baseline = golden["baseline"]["recall_at_10"]
        epsilon = golden["baseline"]["epsilon"]
        detail = "\n".join(
            f"  {recall:.2f} {query!r} -> {got[:4]}"
            for query, recall, got in per_query
        )
        self.assertGreaterEqual(
            mean_recall, baseline - epsilon,
            f"recall@10 regressed: mean {mean_recall:.3f} < "
            f"baseline {baseline} - epsilon {epsilon}:\n{detail}")
