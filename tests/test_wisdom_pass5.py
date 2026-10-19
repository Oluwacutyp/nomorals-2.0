"""Pass #5 (WisdomKeeper) failing-first tests.

Bug 1: research-fed ingest queue can never succeed — _ingest_one_queue_row
    builds a ManifestEntry with canon_status="reference", which is not in
    CANON_STATUSES, so validate() raises and every queued row lands 'failed'.
Bug 2: digest_work matches passages by ``slug in p.work``, but _to_provenance
    strips the [slug] prefix so p.work is the bare title — the filter almost
    always misses and digest_work raises ValueError("no passages found").
Bug 3: synthesize() groups by tradition via ``slug in p.work`` (always
    "unknown") and looks up cross-links by title instead of slug (always
    empty).
Bug 4: _control_wisdom_ask / _control_wisdom_compare cut the rendered answer
    at 6000 chars even though _send_long_checked chunks losslessly.
"""
import json
import sqlite3
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.wisdom.autonomy import WisdomOrgan
from nomorals.wisdom.corpus import Answer, ManifestEntry, ProvenanceHit


class _FakeDB:
    def __init__(self):
        self.con = sqlite3.connect(":memory:")
        self.con.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        cur = self.con.execute(sql, params)
        self.con.commit()
        return cur

    def query(self, sql, params=()):
        return [dict(r) for r in self.con.execute(sql, params).fetchall()]

    def query_one(self, sql, params=()):
        row = self.con.execute(sql, params).fetchone()
        return dict(row) if row else None


def _organ():
    ctx = SimpleNamespace()
    ctx.db = _FakeDB()
    ctx.config = SimpleNamespace(get=lambda *a, **k: None)
    with mock.patch("nomorals.wisdom.autonomy._bus") as bus:
        bus.ensure_schema.return_value = None
        bus.drain.return_value = []
        return WisdomOrgan(ctx)


_SNIPPET = (
    "Jesus said: if those who lead you say to you, see, the kingdom is in "
    "heaven, then the birds of heaven will precede you. But the kingdom is "
    "inside of you, and it is outside of you. When you come to know yourselves "
    "you will become known, and you will realize that it is you who are the "
    "sons of the living father.")


def _hit(slug="gospel-of-thomas", work="Gospel of Thomas",
         tradition_section="Saying 3"):
    return ProvenanceHit(
        work=work, translator="", section=tradition_section,
        url="https://example.com/thomas", snippet=_SNIPPET,
        score=1.0, canon_status="gnostic", slug=slug, chapter_number=1)


def _keeper_with(manifest_entries, passages):
    keeper = SimpleNamespace()
    keeper.corpus = SimpleNamespace(_manifest=manifest_entries)
    keeper.ask = lambda *a, **k: Answer(query="kingdom", passages=passages)
    return keeper


def _thomas_entry():
    return ManifestEntry.from_dict({
        "slug": "gospel-of-thomas", "title": "Gospel of Thomas",
        "tradition": "christian-gnostic", "canon_status": "gnostic",
        "source_url": "https://example.com/thomas",
        "ingested_at": time.time(), "sha256": "abc123"})


class QueueIngestTests(unittest.TestCase):
    def test_queue_row_ingest_uses_valid_canon_status(self):
        organ = _organ()
        organ.db.execute(
            "INSERT INTO wisdom_ingest_queue (url, title, tradition, queued_at)"
            " VALUES (?, ?, ?, ?)",
            ("https://example.com/text.txt", "Example Text", "research",
             time.time()))
        row = organ.db.query_one(
            "SELECT id, url, title, tradition FROM wisdom_ingest_queue"
            " WHERE status='queued'")

        seen = {}

        class FakeIngestor:
            def ingest_entry(self, entry):
                entry.validate()  # the real ArchiveIngestor validates first
                seen["canon_status"] = entry.canon_status
                return entry

        slug = organ._ingest_one_queue_row(FakeIngestor(), row)
        self.assertTrue(slug)
        self.assertEqual(seen.get("canon_status"), "secondary")
        st = organ.db.query_one(
            "SELECT status FROM wisdom_ingest_queue WHERE id=?", (row["id"],))
        self.assertEqual(st["status"], "done")


class DigestSlugTests(unittest.TestCase):
    def test_digest_work_matches_passages_by_slug(self):
        organ = _organ()
        keeper = _keeper_with({"gospel-of-thomas": _thomas_entry()},
                              [_hit()])
        with mock.patch.object(WisdomOrgan, "_keeper",
                               return_value=keeper):
            digest = organ.digest_work("gospel-of-thomas")
        self.assertEqual(digest["slug"], "gospel-of-thomas")
        self.assertTrue(digest["passages"])


class SynthesizeTests(unittest.TestCase):
    def test_synthesize_groups_by_tradition_and_links_by_slug(self):
        organ = _organ()
        keeper = _keeper_with({"gospel-of-thomas": _thomas_entry()},
                              [_hit()])
        organ.db.execute(
            "INSERT INTO wisdom_links (slug_a, slug_b, shared_terms,"
            " strength, created_at) VALUES (?, ?, ?, ?, ?)",
            ("gospel-of-thomas", "tao-te-ching",
             json.dumps(["kingdom", "light"]), 0.5, time.time()))
        with mock.patch.object(WisdomOrgan, "_keeper",
                               return_value=keeper):
            out = organ.synthesize("kingdom")
        self.assertIn("christian-gnostic", out["by_tradition"])
        self.assertNotIn("unknown", out["by_tradition"])
        self.assertTrue(out["cross_links"])


def _long_answer(n=10):
    passages = []
    for i in range(n):
        passages.append(ProvenanceHit(
            work=f"Work Number {i}", translator="",
            section=f"Chapter {i}", url="https://example.com",
            snippet=("word " * 600).strip(),  # ~3000 chars per passage
            score=1.0, canon_status="canon", slug=f"work-{i}",
            chapter_number=1))
    return Answer(query="kingdom", passages=passages, synthesis="syn",
                  confidence="high")


class TruncationTests(unittest.TestCase):
    def _rt(self):
        from nomorals.agents.partner.runtime_wisdom import (
            RuntimeWisdomMixin)
        rt = RuntimeWisdomMixin.__new__(RuntimeWisdomMixin)
        captured = {}
        def fake_send(platform, chat, text):
            captured["text"] = text
            return ""
        rt._send_long_checked = fake_send
        return rt, captured

    def test_ask_does_not_truncate_at_6000(self):
        rt, captured = self._rt()
        keeper = SimpleNamespace(
            status=lambda: {"corpus": {"ingested": 10}},
            ask=lambda query, top=5: _long_answer())
        rt._control_wisdom_ask(keeper, "kingdom",
                               SimpleNamespace(platform="telegram"))
        text = captured["text"]
        self.assertGreater(len(text), 6000)
        # the tail (5th passage's footer) must survive — ask shows top 5
        self.assertIn("work-4", text)

    def test_compare_does_not_truncate_at_6000(self):
        rt, captured = self._rt()
        passages = []
        for i in range(10):
            passages.append({
                "work": f"Work Number {i}", "slug": f"work-{i}",
                "section": f"Chapter {i}", "chapter": None,
                "translator": "", "url": "https://example.com",
                "snippet": ("word " * 600).strip(), "score": 1.0,
                "canon_status": "canon"})
        history = SimpleNamespace(
            compare=lambda topic, corpus=None: {
                "topic": topic, "passages": passages,
                "timeline_context": []})
        keeper = SimpleNamespace(
            status=lambda: {"corpus": {"ingested": 10}},
            ask=lambda query, top=5: _long_answer(),
            history=history)
        rt._control_wisdom_compare(keeper, "kingdom vs light",
                                   SimpleNamespace(platform="telegram"))
        text = captured["text"]
        self.assertGreater(len(text), 6000)
        # the tail (5th passage's footer) must survive — compare shows top 5
        self.assertIn("work-4", text)


if __name__ == "__main__":
    unittest.main()
