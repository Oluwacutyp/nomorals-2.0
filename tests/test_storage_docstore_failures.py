"""Pass #218 pins: storage/docstore.py — 1 exclusion, 0 converts.

- 665: ``MongoDocumentStore.cas`` DuplicateKeyError control flow. When
  the insert for a fresh (expected=None) CAS raises DuplicateKeyError
  a doc already exists, so the code falls through to the expired-doc
  check: if the existing doc is expired the swap proceeds (replace_one
  against the exact expiry observed); if it is live the CAS honestly
  returns False. The exception is the signal that a doc exists, not a
  failure — the CAS result (True/False) is the honest answer either way.
  A non-duplicate driver error still raises StorageError.

storage/docstore.py DOES record failures elsewhere (the shadow and
resolve seams), so NOT-recording is pinned against the storage ledger:
storage recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use a fake collection whose
insert_one raises the real pymongo DuplicateKeyError, never a swallow.
"""

import time
from types import SimpleNamespace

import pymongo
import pytest

from nomorals.storage import failures as st_fail
from nomorals.storage.docstore import MongoDocumentStore, StorageError


def _snap():
    return list(st_fail.recent_failures(limit=200))


class _FakeColl:
    """A collection whose insert_one raises DuplicateKeyError on demand."""

    def __init__(self, existing=None, dup_error=None):
        self.existing = existing
        self.dup_error = dup_error
        self.inserted = []
        self.replaced = []

    def insert_one(self, doc):
        if self.dup_error is not None:
            raise self.dup_error
        self.inserted.append(doc)
        return SimpleNamespace(inserted_id=doc["_id"])

    def find_one(self, filt):
        return self.existing

    def replace_one(self, filt, doc, upsert=False):
        self.replaced.append((filt, doc))
        return SimpleNamespace(matched_count=1)


def _store(coll, now_fn=None):
    # The client is injected: client[database][collection] is the fake.
    fake_client = {"nomorals": {"things": coll}}
    return MongoDocumentStore(client=fake_client,
                              now_fn=now_fn or time.time)


# ── L665: DuplicateKeyError falls through to the expired-doc check ────


def test_665_fresh_insert_returns_true():
    coll = _FakeColl(existing=None)
    store = _store(coll)
    before = _snap()
    assert store.cas("things", "k1", None, {"v": 1}) is True
    assert len(coll.inserted) == 1
    assert coll.inserted[0]["_id"] == "k1"
    assert _snap() == before


def test_665_duplicate_live_doc_returns_false():
    # The insert raises DuplicateKeyError; the existing doc is live
    # (expires_at None), so the CAS honestly fails — and the
    # DuplicateKeyError never escapes.
    live = {"_id": "k1", "v": 1, "expires_at": None}
    coll = _FakeColl(
        existing=live,
        dup_error=pymongo.errors.DuplicateKeyError("dup"))
    store = _store(coll)
    before = _snap()
    assert store.cas("things", "k1", None, {"v": 2}) is False
    assert coll.replaced == []
    assert _snap() == before


def test_665_duplicate_expired_doc_swaps():
    # The existing doc is expired (expires_at 0 < now), so the swap
    # proceeds against the exact expiry observed.
    expired = {"_id": "k1", "v": 1, "expires_at": 0.0}
    coll = _FakeColl(
        existing=expired,
        dup_error=pymongo.errors.DuplicateKeyError("dup"))
    store = _store(coll, now_fn=lambda: 1000.0)
    before = _snap()
    assert store.cas("things", "k1", None, {"v": 2}) is True
    assert len(coll.replaced) == 1
    assert coll.replaced[0][1]["v"] == 2
    assert _snap() == before


def test_665_duplicate_just_refreshed_doc_returns_false():
    # A racing writer refreshed the doc (expiry in the future), so
    # the observed expiry no longer matches — the CAS honestly fails.
    refreshed = {"_id": "k1", "v": 1, "expires_at": 2000.0}
    coll = _FakeColl(
        existing=refreshed,
        dup_error=pymongo.errors.DuplicateKeyError("dup"))
    store = _store(coll, now_fn=lambda: 1000.0)
    before = _snap()
    assert store.cas("things", "k1", None, {"v": 2}) is False
    assert coll.replaced == []
    assert _snap() == before


def test_665_non_duplicate_driver_error_still_raises():
    # The other except clause is intact: a non-duplicate driver error
    # is a StorageError, not a silent fall-through.
    coll = _FakeColl(
        existing=None,
        dup_error=pymongo.errors.ConnectionFailure("down"))
    store = _store(coll)
    before = _snap()
    with pytest.raises(StorageError):
        store.cas("things", "k1", None, {"v": 1})
    assert _snap() == before
