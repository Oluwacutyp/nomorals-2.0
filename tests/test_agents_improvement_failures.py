"""Tail-100 pins: nomorals/agents/improvement.py census sites.

The improvement loop persists its cycle rows in SQL (NOT NULL score
columns, sentinel-coerced) and treats the kv details stash as an
auxiliary annotation on top of that durable record. All three
sites are held as exclusions:

137 — ``CycleRecord.from_row``: a corrupt details_json blob ->
details = {}. The row's real facts (scores, action, status) come
from the typed columns beside the blob; a mangled annotation must
not make a genuine cycle record unreadable.

403 — ``_record``: the kv details stash write crashes -> pass.
The cycle row itself was already INSERTed (and is asserted present
by the pin); the stash is a history-display annotation whose
absence reads back as {} via site 479's default. FLAGGED for the
parent as a targeted-fix candidate under the data-loss rule: it is
a save-side swallow whose value is read back later, so if the
details are ever promoted from annotation to evidence, this site
should be converted first. As the code stands the durable record
(the row) is never the swallowed write.

479 — ``history``: the kv details read crashes -> the record keeps
its from_row details ({}) instead of history() raising. Read-side
degrade to the documented default on a display path; the cycle
rows themselves are never hidden by it.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import improvement as I
from nomorals.core.config import ImprovementSettings
from nomorals.storage.db import Database


def _migrated_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(db: Database) -> SimpleNamespace:
    settings = SimpleNamespace(
        improvement=ImprovementSettings(mode="approval", target=0.8))
    return SimpleNamespace(settings=settings, db=db, router=None)


def _row(**over) -> dict:
    row = {
        "id": "improve-x", "dimension": "coding",
        "before_score": 0.5, "after_score": 0.6, "delta": 0.1,
        "proposal_id": "p1", "action": "applied",
        "rationale": "r", "edit_summary": "e",
        "status": "improved", "created_at": 1.0,
        "details_json": '{"tasks": 3}',
    }
    row.update(over)
    return row


class TestFromRowCorruptDetails(unittest.TestCase):
    """137: corrupt details_json -> details {}, record still built."""

    def test_corrupt_details_become_empty_dict(self):
        rec = I.CycleRecord.from_row(_row(details_json="{corrupt"))
        self.assertEqual(rec.details, {})
        self.assertEqual(rec.before_score, 0.5)
        self.assertEqual(rec.status, "improved")

    def test_valid_details_are_parsed(self):
        # Control: a healthy blob comes through untouched.
        rec = I.CycleRecord.from_row(_row())
        self.assertEqual(rec.details, {"tasks": 3})


class TestRecordStashCrash(unittest.TestCase):
    """403: details stash crash -> the SQL cycle row still persists."""

    def test_record_survives_stash_crash(self):
        db = _migrated_db()
        loop = I.ImprovementLoop(_ctx(db))

        class _BoomKV:
            def __init__(self, db_):
                pass

            def set_raw(self, *a, **k):
                raise RuntimeError("kv down")

        with mock.patch.object(I, "KVStore", _BoomKV):
            rec = loop._record("cycle-1", "coding", 0.5, 0.6, 0.1,
                               "p1", "applied", "why", "what",
                               "improved", {"tasks": 3})
        self.assertEqual(rec.id, "cycle-1")
        rows = db.query(
            "SELECT * FROM improvement_runs WHERE id=?", ("cycle-1",))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "improved")
        self.assertEqual(rows[0]["before_score"], 0.5)

    def test_record_with_healthy_stash_keeps_details(self):
        # Control: with the stash alive, history shows the details.
        db = _migrated_db()
        loop = I.ImprovementLoop(_ctx(db))
        loop._record("cycle-2", "coding", 0.5, 0.6, 0.1, "p1",
                     "applied", "why", "what", "improved",
                     {"tasks": 7})
        hist = loop.history(limit=5)
        mine = [r for r in hist if r.id == "cycle-2"]
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0].details, {"tasks": 7})


class TestHistoryReadCrash(unittest.TestCase):
    """479: details read crash -> record returned with {} details."""

    def test_history_survives_kv_read_crash(self):
        db = _migrated_db()
        loop = I.ImprovementLoop(_ctx(db))
        loop._record("cycle-3", "coding", 0.5, 0.6, 0.1, "p1",
                     "applied", "why", "what", "improved",
                     {"tasks": 9})

        class _BoomKV:
            def __init__(self, db_):
                pass

            def get(self, *a, **k):
                raise RuntimeError("kv down")

        with mock.patch.object(I, "KVStore", _BoomKV):
            hist = loop.history(limit=5)
        mine = [r for r in hist if r.id == "cycle-3"]
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0].details, {})
        self.assertEqual(mine[0].status, "improved")


if __name__ == "__main__":
    unittest.main()
