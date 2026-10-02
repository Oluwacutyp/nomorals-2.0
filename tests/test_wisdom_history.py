"""Phase 3 tests: HistoryEngine, seed timeline, compare/lineage."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.wisdom import (Answer, HistoryEngine, HistoryError, WisdomError,
                             WisdomKeeper)
from nomorals.wisdom.history import _data_path
from nomorals.wisdom.corpus import ProvenanceHit


def _ctx():
    return SimpleNamespace()


def _write_dataset(entries):
    tmp = tempfile.NamedTemporaryFile(
        prefix="timeline-test-", suffix=".json", delete=False)
    tmp.write(json.dumps(entries).encode("utf-8"))
    tmp.close()
    return Path(tmp.name)


_GOOD_ENTRY = {
    "start": 100, "end": 200, "tradition": "test-trad",
    "region": "Testland", "title": "Test event",
    "summary": "A test event for validation.",
    "sources": ["example.com"],
}


class LoadAndValidationTests(unittest.TestCase):
    def test_loads_seed_dataset(self):
        eng = HistoryEngine(_ctx())
        self.assertGreaterEqual(len(eng.events()), 50)

    def test_every_event_has_all_fields(self):
        for e in HistoryEngine(_ctx()).events():
            for field in ("start", "end", "tradition", "region",
                          "title", "summary", "sources"):
                self.assertIn(field, e, field)
            self.assertIsInstance(e["start"], int)
            self.assertIsInstance(e["end"], int)
            self.assertLessEqual(e["start"], e["end"])

    def test_every_seed_entry_has_sources(self):
        """Validate the seed JSON file itself: no sourceless entries."""
        raw = json.loads(Path(_data_path()).read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(raw), 50)
        for i, e in enumerate(raw):
            sources = e.get("sources")
            self.assertIsInstance(sources, list, f"entry {i}")
            self.assertTrue(sources, f"entry {i} has empty sources")
            for s in sources:
                self.assertIsInstance(s, str, f"entry {i}")
                self.assertTrue(s.strip(), f"entry {i} blank source")

    def test_missing_field_fails_fast_naming_index(self):
        bad = dict(_GOOD_ENTRY)
        del bad["sources"]
        path = _write_dataset([_GOOD_ENTRY, bad, _GOOD_ENTRY])
        with self.assertRaisesRegex(HistoryError, r"entry 1"):
            HistoryEngine(_ctx(), data_path=path)

    def test_bad_types_rejected(self):
        for mutate in (
            lambda e: e.update(start="100"),
            lambda e: e.update(end=50),          # start > end
            lambda e: e.update(tradition=""),
            lambda e: e.update(sources=[]),
            lambda e: e.update(sources=["ok", 42]),
        ):
            bad = dict(_GOOD_ENTRY)
            mutate(bad)
            path = _write_dataset([bad])
            with self.assertRaises(HistoryError):
                HistoryEngine(_ctx(), data_path=path)

    def test_non_list_dataset_rejected(self):
        tmp = tempfile.NamedTemporaryFile(
            prefix="timeline-test-", suffix=".json", delete=False)
        tmp.write(b'{"not": "a list"}')
        tmp.close()
        with self.assertRaises(HistoryError):
            HistoryEngine(_ctx(), data_path=Path(tmp.name))

    def test_history_error_is_wisdom_error(self):
        self.assertTrue(issubclass(HistoryError, WisdomError))


class TimelineQueryTests(unittest.TestCase):
    def setUp(self):
        self.eng = HistoryEngine(_ctx())

    def test_range_query_overlaps_and_sorts(self):
        evs = self.eng.timeline(start_year=-800, end_year=-200)
        self.assertTrue(evs)
        for e in evs:
            self.assertLessEqual(e["start"], -200)
            self.assertGreaterEqual(e["end"], -800)
        starts = [e["start"] for e in evs]
        self.assertEqual(starts, sorted(starts))

    def test_range_query_boundary_overlap(self):
        # An event exactly touching the range edge counts as overlapping.
        evs = self.eng.timeline(start_year=1945, end_year=1945)
        titles = [e["title"] for e in evs]
        self.assertIn("Nag Hammadi library discovered", titles)
        self.assertNotIn("Dead Sea Scrolls discovered", titles)

    def test_tradition_filter(self):
        evs = self.eng.timeline(tradition="taoism")
        self.assertTrue(evs)
        self.assertTrue(all(e["tradition"] == "taoism" for e in evs))
        titles = [e["title"] for e in evs]
        self.assertIn("Tao Te Ching — traditional dating", titles)

    def test_unknown_tradition_raises_with_known_list(self):
        with self.assertRaisesRegex(HistoryError, "taoism") as cm:
            self.eng.timeline(tradition="bogus-trad")
        self.assertIn("known traditions", str(cm.exception))

    def test_reversed_range_raises(self):
        with self.assertRaises(HistoryError):
            self.eng.timeline(start_year=2000, end_year=1000)

    def test_default_range_covers_seed(self):
        evs = self.eng.timeline()
        self.assertEqual(len(evs), len(self.eng.events()))

    def test_traditions_sorted(self):
        trads = self.eng.traditions()
        self.assertEqual(trads, sorted(trads))
        for t in ("buddhism", "christian-gnostic", "hinduism",
                  "judaism", "sufism", "taoism", "theosophy"):
            self.assertIn(t, trads)


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.eng = HistoryEngine(_ctx())

    def test_lineage_traces_across_centuries(self):
        evs = self.eng.lineage("nag hammadi")
        titles = [e["title"] for e in evs]
        # Gospel of Thomas matches too: its summary notes it was
        # preserved in the Nag Hammadi codices.
        self.assertEqual(titles, [
            "Gospel of Thomas composed",
            "Nag Hammadi codices composed",
            "Nag Hammadi codices buried",
            "Nag Hammadi library discovered",
        ])
        starts = [e["start"] for e in evs]
        self.assertEqual(starts, sorted(starts))

    def test_lineage_case_insensitive(self):
        lower = self.eng.lineage("shankara")
        upper = self.eng.lineage("SHANKARA")
        self.assertEqual(lower, upper)
        self.assertTrue(lower)

    def test_lineage_matches_summary(self):
        evs = self.eng.lineage("kundalini")
        self.assertTrue(any("Hatha Yoga Pradipika" in e["title"]
                            for e in evs))

    def test_lineage_no_match_returns_empty(self):
        self.assertEqual(self.eng.lineage("zzz-no-such-figure"), [])

    def test_lineage_empty_query_raises(self):
        with self.assertRaises(HistoryError):
            self.eng.lineage("  ")


def _fake_corpus():
    hits = [
        ProvenanceHit(work="Gospel of Thomas", translator="Patterson",
                      section="Saying 3", url="https://gnosis.org/x",
                      snippet="the kingdom is within you",
                      canon_status="gnostic"),
        ProvenanceHit(work="Tao Te Ching", translator="Legge",
                      section="ch. 1", url="https://sacred-texts.com/y",
                      snippet="the tao that can be told",
                      canon_status="eastern"),
    ]
    entries = [
        SimpleNamespace(title="Gospel of Thomas",
                        tradition="christian-gnostic"),
        SimpleNamespace(title="Tao Te Ching", tradition="taoism"),
    ]
    return SimpleNamespace(
        ask=lambda topic, **kw: Answer(query=topic, passages=list(hits)),
        list=lambda: entries,
    )


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.eng = HistoryEngine(_ctx())

    def test_compare_without_corpus(self):
        out = self.eng.compare("gnostic gospels")
        self.assertEqual(out["topic"], "gnostic gospels")
        self.assertEqual(out["passages"], [])
        self.assertTrue(out["timeline_context"])
        self.assertTrue(all("tradition" in e
                            for e in out["timeline_context"]))

    def test_compare_with_corpus_keeps_traditions_visible(self):
        out = self.eng.compare("kingdom within", corpus=_fake_corpus())
        by_work = {p["work"]: p for p in out["passages"]}
        self.assertEqual(by_work["Gospel of Thomas"]["tradition"],
                         "christian-gnostic")
        self.assertEqual(by_work["Gospel of Thomas"]["canon_status"],
                         "gnostic")
        self.assertEqual(by_work["Tao Te Ching"]["tradition"], "taoism")
        self.assertEqual(by_work["Tao Te Ching"]["canon_status"], "eastern")

    def test_compare_with_keeper_shaped_corpus(self):
        inner = _fake_corpus()
        fake = SimpleNamespace(corpus=inner, ask=inner.ask)
        out = self.eng.compare("kingdom", corpus=fake)
        self.assertEqual(out["passages"][0]["tradition"],
                         "christian-gnostic")

    def test_compare_timeline_context_matches_topic(self):
        out = self.eng.compare("Sufi mysticism")
        self.assertTrue(out["timeline_context"])
        self.assertTrue(any(e["tradition"] == "sufism"
                            for e in out["timeline_context"]))

    def test_compare_empty_topic_raises(self):
        with self.assertRaises(HistoryError):
            self.eng.compare("  ")

    def test_keeper_history_property(self):
        keeper = WisdomKeeper(_ctx())
        self.assertIsInstance(keeper.history, HistoryEngine)
        self.assertIs(keeper.history, keeper.history)  # lazy singleton


if __name__ == "__main__":
    unittest.main()
