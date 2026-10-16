"""Tests for nomorals.people: SQLite identity spine.

Stdlib + sqlite only. No network, no writes outside temp dirs.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from nomorals.people import PersonStore, migrate_from_relationships
from nomorals.social.relationships import RelationshipTracker


class EnsureGetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = PersonStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_ensure_get_roundtrip(self) -> None:
        self.assertIsNone(self.store.get("ada"))
        person = self.store.ensure("ada", "Ada")
        self.assertEqual(person["person_id"], "ada")
        self.assertEqual(person["display_name"], "Ada")
        self.assertEqual(person["closeness"], 0.0)
        self.assertEqual(person["inbound_count"], 0)
        self.assertEqual(person["notes"], [])
        again = self.store.ensure("ada")
        self.assertEqual(again["display_name"], "Ada")
        self.assertEqual(self.store.get("ada"), again)

    def test_ensure_never_clobbers_display_name(self) -> None:
        self.store.ensure("ada", "Ada")
        self.assertEqual(self.store.ensure("ada", "").get("display_name"), "Ada")
        self.assertEqual(
            self.store.ensure("ada", "Someone Else").get("display_name"), "Ada"
        )

    def test_ensure_fills_blank_name(self) -> None:
        self.store.ensure("ada")
        self.assertEqual(self.store.ensure("ada", "Ada")["display_name"], "Ada")


class InteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = PersonStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_inbound_updates_counts_and_closeness(self) -> None:
        before = self.store.ensure("ada", "Ada")
        after = self.store.record_inbound("ada", topic="interview")
        self.assertEqual(after["inbound_count"], before["inbound_count"] + 1)
        self.assertEqual(after["last_topic"], "interview")
        self.assertGreater(after["closeness"], before["closeness"])
        self.assertLessEqual(after["closeness"], 1.0)

    def test_outbound_owner_initiated_weighs_heaviest(self) -> None:
        passive = self.store.record_inbound("passive")
        active = self.store.record_outbound("active", owner_initiated=True)
        self.assertGreater(active["closeness"], passive["closeness"])
        self.assertEqual(active["owner_initiated_count"], 1)
        self.assertEqual(active["outbound_count"], 1)

    def test_closeness_capped_at_one(self) -> None:
        person = None
        for _ in range(200):
            person = self.store.record_outbound("ada", owner_initiated=True)
        assert person is not None
        self.assertLessEqual(person["closeness"], 1.0)
        self.assertEqual(person["closeness"], 1.0)

    def test_warmth_ema(self) -> None:
        warm = self.store.record_inbound("ada", warm=True)
        self.assertGreater(warm["warmth"], 0.5)
        cold = self.store.record_inbound("ada", warm=False)
        self.assertLess(cold["warmth"], warm["warmth"])


class NotesLinksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = PersonStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_notes_preserved_and_append_only(self) -> None:
        self.store.add_note("ada", "Birthday in June")
        self.store.add_note("ada", "Has two kids")
        self.assertEqual(
            self.store.get("ada")["notes"],
            ["Birthday in June", "Has two kids"],
        )
        # Exact duplicate suppressed; a correction appends, never deletes.
        self.store.add_note("ada", "Birthday in June")
        self.store.add_note("ada", "Correction: birthday in July")
        self.assertEqual(
            self.store.get("ada")["notes"],
            ["Birthday in June", "Has two kids", "Correction: birthday in July"],
        )

    def test_add_note_blank_is_noop(self) -> None:
        self.assertIsNone(self.store.add_note("ada", "   "))
        self.assertEqual(self.store.get("ada")["notes"], [])

    def test_add_link_dedupes_and_ensures_endpoints(self) -> None:
        first = self.store.add_link("ada", "grace", "colleague")
        self.assertEqual(self.store.add_link("ada", "grace", "colleague"), first)
        self.assertEqual(
            self.store.add_link("ada", "grace", "friend") != first, True
        )
        self.assertIsNotNone(self.store.get("grace"))
        kinds = {link["kind"] for link in self.store.links_for("ada")}
        self.assertEqual(kinds, {"colleague", "friend"})


class GiftsObligationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = PersonStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_gift_and_obligation_keyed_off_same_person(self) -> None:
        self.store.ensure("ada", "Ada")
        gift_id = self.store.add_gift(
            "ada", "Book", "given", value_minor=2500, currency="NGN",
            date="2026-10-01",
        )
        ob_id = self.store.link_obligation(
            "ada", "c-1", "Send the draft", deadline="2026-10-20"
        )
        gifts = self.store.gifts_for("ada")
        obs = self.store.obligations_for("ada")
        self.assertEqual(len(gifts), 1)
        self.assertEqual(gifts[0]["id"], gift_id)
        self.assertEqual(gifts[0]["direction"], "given")
        self.assertEqual(gifts[0]["value_minor"], 2500)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["id"], ob_id)
        self.assertEqual(obs[0]["commitment_id"], "c-1")
        self.assertEqual(obs[0]["status"], "open")
        # Other people see nothing.
        self.assertEqual(self.store.gifts_for("grace"), [])
        self.assertEqual(self.store.obligations_for("grace"), [])

    def test_gift_direction_validated(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add_gift("ada", "Book", "stolen")  # type: ignore[arg-type]

    def test_link_obligation_idempotent_on_commitment_id(self) -> None:
        first = self.store.link_obligation("ada", "c-1", "Send the draft")
        self.assertEqual(
            self.store.link_obligation("ada", "c-1", "Send the draft"), first
        )
        self.assertEqual(len(self.store.obligations_for("ada")), 1)


class QueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = PersonStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_top_contacts_sorted_by_closeness(self) -> None:
        self.store.record_inbound("quiet")
        for _ in range(3):
            self.store.record_outbound("close", owner_initiated=True)
        top = self.store.top_contacts()
        self.assertEqual(top[0]["person_id"], "close")
        self.assertEqual(
            [p["person_id"] for p in self.store.top_contacts(limit=1)], ["close"]
        )

    def test_needs_reconnect(self) -> None:
        now = time.time()
        # Close contact, last traffic 40 days ago -> due.
        for _ in range(5):
            self.store.record_outbound("old", owner_initiated=True, at=now - 40 * 86400)
        self.assertGreaterEqual(self.store.get("old")["closeness"], 0.4)
        # Close contact, traffic today -> not due.
        for _ in range(5):
            self.store.record_outbound("fresh", owner_initiated=True, at=now)
        # Low-closeness contact, quiet 40 days -> not due (score gate).
        self.store.record_inbound("stranger", at=now - 40 * 86400)
        due = {p["person_id"] for p in self.store.needs_reconnect(days=30)}
        self.assertEqual(due, {"old"})

    def test_search_matches_name_topic_and_notes(self) -> None:
        self.store.ensure("ada", "Ada Lovelace")
        self.store.record_inbound("ada", topic="interview")
        self.store.add_note("grace", "Loves chess")
        self.assertEqual(
            [p["person_id"] for p in self.store.search("lovelace")], ["ada"]
        )
        self.assertEqual(
            [p["person_id"] for p in self.store.search("interview")], ["ada"]
        )
        self.assertEqual(
            [p["person_id"] for p in self.store.search("chess")], ["grace"]
        )
        self.assertEqual(self.store.search("zzz-no-match"), [])


class MigrateTests(unittest.TestCase):
    def tearDown(self) -> None:
        tmp = getattr(self, "_tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def _tracker_with_two_contacts(self) -> RelationshipTracker:
        self._tmp = tempfile.TemporaryDirectory()
        tracker = RelationshipTracker(
            path=str(Path(self._tmp.name) / "relationships.json")
        )
        tracker.note_inbound("ada", display_name="Ada", topic="interview")
        tracker.note_inbound("ada", display_name="Ada")
        tracker.note_outbound("ada", owner_initiated=True)
        tracker.add_note("ada", "Birthday in June")
        tracker.note_inbound("grace", display_name="Grace")
        tracker.add_note("grace", "Loves chess")
        return tracker

    def test_migrate_preserves_counts_notes_and_scores(self) -> None:
        tracker = self._tracker_with_two_contacts()
        store = PersonStore(":memory:")
        try:
            count = migrate_from_relationships(tracker, store)
            self.assertEqual(count, 2)
            ada = store.get("ada")
            assert ada is not None
            self.assertEqual(ada["display_name"], "Ada")
            self.assertEqual(ada["inbound_count"], 2)
            self.assertEqual(ada["outbound_count"], 1)
            self.assertEqual(ada["owner_initiated_count"], 1)
            self.assertEqual(ada["last_topic"], "interview")
            self.assertEqual(ada["notes"], ["Birthday in June"])
            self.assertAlmostEqual(
                ada["closeness"],
                tracker.get("ada").closeness,  # type: ignore[union-attr]
                places=3,
            )
            grace = store.get("grace")
            assert grace is not None
            self.assertEqual(grace["inbound_count"], 1)
            self.assertEqual(grace["notes"], ["Loves chess"])
        finally:
            store.close()

    def test_migrate_is_idempotent(self) -> None:
        tracker = self._tracker_with_two_contacts()
        store = PersonStore(":memory:")
        try:
            self.assertEqual(migrate_from_relationships(tracker, store), 2)
            self.assertEqual(migrate_from_relationships(tracker, store), 0)
            ada = store.get("ada")
            assert ada is not None
            self.assertEqual(ada["inbound_count"], 2)  # not doubled
            self.assertEqual(ada["notes"], ["Birthday in June"])  # not duped
            # Notes added in the store survive a re-migration.
            store.add_note("ada", "Store-side note")
            migrate_from_relationships(tracker, store)
            self.assertEqual(
                store.get("ada")["notes"],
                ["Birthday in June", "Store-side note"],
            )
        finally:
            store.close()

    def test_migrate_from_tracker_with_more_than_ten_contacts(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        tracker = RelationshipTracker(
            path=str(Path(tmp.name) / "relationships.json")
        )
        for i in range(15):
            tracker.note_inbound(f"p{i:02d}", display_name=f"Person {i}")
        store = PersonStore(":memory:")
        try:
            # top_contacts() defaults to 10 — migration must not truncate.
            self.assertEqual(migrate_from_relationships(tracker, store), 15)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
