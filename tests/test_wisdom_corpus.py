"""Phase 1 tests: CanonCorpus, manifest, keeper façade."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.wisdom import (Answer, CanonCorpus, CorpusError, ManifestEntry,
                             WisdomError, WisdomKeeper)
from nomorals.wisdom.corpus import CANON_STATUSES


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-test-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(settings=settings), tmp


def _entry(slug="gospel-of-thomas", **kw):
    d = {
        "slug": slug,
        "title": "Gospel of Thomas",
        "tradition": "christian-gnostic",
        "canon_status": "gnostic",
        "translator": "Patterson & Robinson",
        "source_url": "http://gnosis.org/naghamm/gth_pat_rob.htm",
        "license": "public-domain",
    }
    d.update(kw)
    return ManifestEntry.from_dict(d)


LOREM = (
    "# The Kingdom\n\n"
    + "The kingdom of heaven is within you and all around you. " * 40
    + "\n\n# Sayings\n\n"
    + "Blessed are the seekers of the inner light and the quiet mind. " * 40
)


class ManifestEntryTests(unittest.TestCase):
    def test_valid_entry(self):
        e = _entry()
        e.validate()
        self.assertEqual(e.slug, "gospel-of-thomas")

    def test_bad_canon_status_rejected(self):
        with self.assertRaises(CorpusError):
            _entry(canon_status="bogus")

    def test_bad_url_rejected(self):
        with self.assertRaises(CorpusError):
            _entry(source_url="not-a-url")

    def test_bad_slug_rejected(self):
        with self.assertRaises(CorpusError):
            _entry(slug="bad slug!")

    def test_missing_title_rejected(self):
        with self.assertRaises(CorpusError):
            ManifestEntry.from_dict({
                "slug": "x", "title": "",
                "canon_status": "gnostic",
                "source_url": "https://example.com/x"})

    def test_all_canon_statuses_accepted(self):
        for status in CANON_STATUSES:
            e = _entry(slug=f"s-{status}", canon_status=status)
            e.validate()

    def test_round_trip(self):
        e = _entry()
        e2 = ManifestEntry.from_dict(e.to_dict())
        self.assertEqual(e2.slug, e.slug)
        self.assertEqual(e2.source_url, e.source_url)

    def test_notes_round_trip(self):
        # triage notes (e.g. "mirror unreachable, kept original URL")
        # must survive a manifest save/load cycle.
        e = _entry()
        e.notes = "sacred-texts.com blocked from this network 2026-10-02"
        e2 = ManifestEntry.from_dict(e.to_dict())
        self.assertEqual(e2.notes, e.notes)
        # and absent notes default to empty, not KeyError
        d = e.to_dict()
        del d["notes"]
        self.assertEqual(ManifestEntry.from_dict(d).notes, "")


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = _ctx()
        self.corpus = CanonCorpus(self.ctx)

    def test_register_and_get(self):
        self.corpus.register(_entry())
        got = self.corpus.get("gospel-of-thomas")
        self.assertEqual(got.translator, "Patterson & Robinson")

    def test_duplicate_slug_rejected(self):
        self.corpus.register(_entry())
        with self.assertRaises(CorpusError):
            self.corpus.register(_entry())

    def test_unknown_slug_raises(self):
        with self.assertRaises(CorpusError):
            self.corpus.get("nope")

    def test_manifest_persists(self):
        self.corpus.register(_entry())
        c2 = CanonCorpus(self.ctx)
        self.assertEqual(c2.get("gospel-of-thomas").title,
                         "Gospel of Thomas")

    def test_status_counts(self):
        self.corpus.register(_entry(slug="a", tradition="t1",
                                    canon_status="gnostic"))
        self.corpus.register(_entry(slug="b", tradition="t1",
                                    canon_status="canon"))
        st = self.corpus.status()
        self.assertEqual(st["texts"], 2)
        self.assertEqual(st["pending"], 2)
        self.assertEqual(st["by_tradition"]["t1"], 2)

    def test_ingest_text_and_ask(self):
        self.corpus.register(_entry())
        self.corpus.ingest_text("gospel-of-thomas", LOREM)
        ans = self.corpus.ask("kingdom within")
        self.assertIsInstance(ans, Answer)
        self.assertGreater(len(ans.passages), 0)
        p = ans.passages[0]
        self.assertIn("gnosis.org", p.url)
        self.assertEqual(p.canon_status, "gnostic")
        self.assertTrue(p.snippet)

    def test_ingest_idempotent_on_same_bytes(self):
        self.corpus.register(_entry())
        e1 = self.corpus.ingest_text("gospel-of-thomas", LOREM)
        e2 = self.corpus.ingest_text("gospel-of-thomas", LOREM)
        self.assertEqual(e1.sha256, e2.sha256)

    def test_ingest_too_small_rejected(self):
        self.corpus.register(_entry())
        with self.assertRaises(CorpusError):
            self.corpus.ingest_text("gospel-of-thomas", "tiny")

    def test_ask_empty_query_rejected(self):
        with self.assertRaises(CorpusError):
            self.corpus.ask("  ")

    def test_ask_no_hits_says_so(self):
        self.corpus.register(_entry())
        self.corpus.ingest_text("gospel-of-thomas", LOREM)
        ans = self.corpus.ask("zxqvkw completely absent words")
        self.assertEqual(ans.passages, [])
        self.assertIn("No passages", ans.synthesis)


class KeeperTests(unittest.TestCase):
    def test_keeper_routes_ask(self):
        ctx, _ = _ctx()
        k = WisdomKeeper(ctx)
        k.corpus.register(_entry())
        k.corpus.ingest_text("gospel-of-thomas", LOREM)
        ans = k.ask("kingdom")
        self.assertGreater(len(ans.passages), 0)

    def test_keeper_status(self):
        ctx, _ = _ctx()
        k = WisdomKeeper(ctx)
        st = k.status()
        self.assertIn("corpus", st)
        self.assertEqual(st["corpus"]["texts"], 0)


class SeedTests(unittest.TestCase):
    def test_seed_loads_broad_manifest(self):
        ctx, _ = _ctx()
        c = CanonCorpus(ctx)
        n = c.seed()
        self.assertGreaterEqual(n, 40)
        st = c.status()
        self.assertEqual(st["texts"], n)
        self.assertEqual(st["pending"], n)

    def test_seed_idempotent(self):
        ctx, _ = _ctx()
        c = CanonCorpus(ctx)
        first = c.seed()
        second = c.seed()
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)

    def test_seed_entries_all_valid(self):
        ctx, _ = _ctx()
        c = CanonCorpus(ctx)
        c.seed()
        for e in c.list():
            e.validate()
            self.assertIn(e.canon_status, CANON_STATUSES)


class ErrorHierarchyTests(unittest.TestCase):
    def test_all_wisdom_errors(self):
        for cls in (CorpusError,):
            self.assertTrue(issubclass(cls, WisdomError))
            self.assertTrue(issubclass(cls, Exception))


if __name__ == "__main__":
    unittest.main()
