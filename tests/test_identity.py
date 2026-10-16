"""Tests for nomorals.identity.documents."""

import unittest

from nomorals.identity.documents import (
    IdentityDocument,
    IdentityTracker,
    validate_number_ref,
)

NOW = 1_800_000_000.0  # fixed reference time for expiry math
DAY = 86400.0


def _doc(doc_id="doc-1", expiry=NOW + 30 * DAY, doc_type="passport", owner_id="owner"):
    return IdentityDocument(
        id=doc_id, owner_id=owner_id, doc_type=doc_type,
        issuing_country="NG", number_ref="vault:identity/passport/1",
        expiry_ts=expiry, issued_ts=NOW - 365 * DAY, notes="",
    )


class TestIdentityTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = IdentityTracker()

    def test_add_get_list(self):
        doc = self.tracker.add(_doc(), owner=True)
        self.assertEqual(doc.id, "doc-1")
        fetched = self.tracker.get("doc-1", owner=True)
        assert fetched is not None
        self.assertEqual(fetched.number_ref, "vault:identity/passport/1")
        self.assertEqual([d.id for d in self.tracker.list(owner=True)], ["doc-1"])
        self.assertIsNone(self.tracker.get("missing", owner=True))

    def test_list_sorted_by_expiry(self):
        self.tracker.add(_doc("late", expiry=NOW + 300 * DAY), owner=True)
        self.tracker.add(_doc("soon", expiry=NOW + 10 * DAY), owner=True)
        self.assertEqual([d.id for d in self.tracker.list(owner=True)], ["soon", "late"])

    def test_expiring_within_windows(self):
        self.tracker.add(_doc("soon", expiry=NOW + 10 * DAY), owner=True)
        self.tracker.add(_doc("mid", expiry=NOW + 45 * DAY), owner=True)
        self.tracker.add(_doc("far", expiry=NOW + 400 * DAY), owner=True)
        within_30 = [d.id for d in self.tracker.expiring_within(30, now=NOW, owner=True)]
        self.assertEqual(within_30, ["soon"])
        within_60 = [d.id for d in self.tracker.expiring_within(60, now=NOW, owner=True)]
        self.assertEqual(within_60, ["soon", "mid"])

    def test_overdue_sorted(self):
        self.tracker.add(_doc("old", expiry=NOW - 20 * DAY), owner=True)
        self.tracker.add(_doc("older", expiry=NOW - 40 * DAY), owner=True)
        self.tracker.add(_doc("future", expiry=NOW + 10 * DAY), owner=True)
        self.assertEqual(
            [d.id for d in self.tracker.overdue(now=NOW, owner=True)],
            ["older", "old"])

    def test_renewal_checklist_per_type(self):
        for doc_type in ("passport", "visa", "licence", "id_card", "permit", "other"):
            steps = self.tracker.renewal_checklist(_doc(doc_type=doc_type), owner=True)
            self.assertGreater(len(steps), 0)
            orders = [s["order"] for s in steps]
            self.assertEqual(orders, sorted(orders))
            for step in steps:
                self.assertIn("title", step)
                self.assertIn("detail", step)
        passport = self.tracker.renewal_checklist(_doc(doc_type="passport"), owner=True)
        visa = self.tracker.renewal_checklist(_doc(doc_type="visa"), owner=True)
        self.assertNotEqual(passport, visa)

    def test_renewal_checklist_by_id(self):
        self.tracker.add(_doc(), owner=True)
        steps = self.tracker.renewal_checklist("doc-1", owner=True)
        self.assertGreater(len(steps), 0)
        with self.assertRaises(KeyError):
            self.tracker.renewal_checklist("nope", owner=True)

    def test_non_owner_denied_everywhere(self):
        doc = _doc()
        for call in (
            lambda: self.tracker.add(doc),
            lambda: self.tracker.add(doc, owner=False),
            lambda: self.tracker.get("doc-1"),
            lambda: self.tracker.list(),
            lambda: self.tracker.expiring_within(30, now=NOW),
            lambda: self.tracker.overdue(now=NOW),
            lambda: self.tracker.renewal_checklist(doc),
        ):
            with self.assertRaises(PermissionError):
                call()

    def test_raw_number_rejected(self):
        # A raw-number-looking value has no vault: prefix → rejected.
        with self.assertRaises(ValueError):
            validate_number_ref("A12345678")
        with self.assertRaises(ValueError):
            validate_number_ref("1234 5678 9012")
        with self.assertRaises(ValueError):
            validate_number_ref("")
        with self.assertRaises(ValueError):
            validate_number_ref("vault:")
        with self.assertRaises(ValueError):
            validate_number_ref("vault: ")
        self.assertEqual(
            validate_number_ref("vault:identity/passport/1"),
            "vault:identity/passport/1")

    def test_document_rejects_raw_number(self):
        with self.assertRaises(ValueError):
            IdentityDocument(
                id="x", owner_id="owner", doc_type="passport",
                issuing_country="NG", number_ref="A12345678",
                expiry_ts=NOW + DAY)
        with self.assertRaises(ValueError):
            IdentityDocument(
                id="x", owner_id="owner", doc_type="bogus",
                issuing_country="NG", number_ref="vault:identity/x/1",
                expiry_ts=NOW + DAY)
        with self.assertRaises(ValueError):
            IdentityDocument(
                id="x", owner_id="", doc_type="passport",
                issuing_country="NG", number_ref="vault:identity/x/1",
                expiry_ts=NOW + DAY)

    def test_duplicate_id_rejected(self):
        self.tracker.add(_doc(), owner=True)
        with self.assertRaises(ValueError):
            self.tracker.add(_doc(), owner=True)

    def test_never_stores_raw_number(self):
        # Only the reference column exists; no raw-number field anywhere.
        self.tracker.add(_doc(), owner=True)
        rows = self.tracker._db.execute(
            "SELECT * FROM identity_documents").fetchall()
        self.assertEqual(len(rows), 1)
        columns = [r[1] for r in self.tracker._db.execute(
            "PRAGMA table_info(identity_documents)").fetchall()]
        self.assertIn("number_ref", columns)
        self.assertNotIn("number", columns)
        self.assertNotIn("raw_number", columns)
        self.assertEqual(rows[0]["number_ref"], "vault:identity/passport/1")


if __name__ == "__main__":
    unittest.main()
