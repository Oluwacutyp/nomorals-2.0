"""Tests for nomorals.life.archive."""

import unittest

from nomorals.life.archive import LifeArchive

NOW = 1_800_000_000.0


class TestLifeArchive(unittest.TestCase):
    def setUp(self):
        self.archive = LifeArchive()

    def test_record_get_timeline(self):
        eid = self.archive.record_event(
            "journal", "note", "Went for a long walk.",
            ts=NOW, provenance="journal:2026-10-16", owner=True)
        self.assertIsInstance(eid, str)
        fetched = self.archive.get(eid, owner=True)
        assert fetched is not None
        self.assertEqual(fetched["text"], "Went for a long walk.")
        self.assertEqual(fetched["provenance"], "journal:2026-10-16")
        self.assertEqual(fetched["source"], "journal")
        self.assertEqual(fetched["kind"], "note")
        self.assertTrue(fetched["private"])
        timeline = self.archive.timeline(owner=True)
        self.assertEqual([e["id"] for e in timeline], [eid])

    def test_timeline_newest_first_and_kind_filter(self):
        self.archive.record_event("a", "note", "old", ts=NOW, owner=True)
        self.archive.record_event("b", "health", "mid", ts=NOW + 10, owner=True)
        self.archive.record_event("c", "note", "new", ts=NOW + 20, owner=True)
        all_events = self.archive.timeline(owner=True)
        self.assertEqual([e["text"] for e in all_events], ["new", "mid", "old"])
        notes = self.archive.timeline(kind="note", owner=True)
        self.assertEqual([e["text"] for e in notes], ["new", "old"])
        limited = self.archive.timeline(limit=1, owner=True)
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0]["text"], "new")

    def test_search_substring_over_text_and_kind(self):
        self.archive.record_event("journal", "note", "Bought gold coins", ts=NOW, owner=True)
        self.archive.record_event("health", "workout", "Ran five kilometres", ts=NOW + 1, owner=True)
        hits = self.archive.search("gold", owner=True)
        self.assertEqual([h["text"] for h in hits], ["Bought gold coins"])
        # Kind is searchable too; case-insensitive.
        hits = self.archive.search("WORKOUT", owner=True)
        self.assertEqual(len(hits), 1)
        self.assertEqual(self.archive.search("zzz-no-match", owner=True), [])

    def test_provenance_preserved_verbatim(self):
        provenance = "import:health-app:export-2026-10-16#row-42"
        eid = self.archive.record_event(
            "health", "workout", "Ran.", ts=NOW,
            provenance=provenance, owner=True)
        fetched = self.archive.get(eid, owner=True)
        assert fetched is not None
        self.assertEqual(fetched["provenance"], provenance)

    def test_private_flag(self):
        eid = self.archive.record_event(
            "x", "note", "public-ish", ts=NOW, private=False, owner=True)
        fetched = self.archive.get(eid, owner=True)
        assert fetched is not None
        self.assertFalse(fetched["private"])

    def test_non_owner_denied(self):
        for call in (
            lambda: self.archive.record_event("s", "k", "t"),
            lambda: self.archive.record_event("s", "k", "t", owner=False),
            lambda: self.archive.get("x"),
            lambda: self.archive.timeline(),
            lambda: self.archive.search("x"),
        ):
            with self.assertRaises(PermissionError):
                call()

    def test_no_publish_method(self):
        self.assertFalse(hasattr(self.archive, "publish"))
        self.assertFalse(hasattr(self.archive, "publish_event"))

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.archive.record_event("", "k", "t", owner=True)
        with self.assertRaises(ValueError):
            self.archive.record_event("s", "", "t", owner=True)
        with self.assertRaises(ValueError):
            self.archive.record_event("s", "k", "  ", owner=True)
        self.assertIsNone(self.archive.get("missing", owner=True))


if __name__ == "__main__":
    unittest.main()
