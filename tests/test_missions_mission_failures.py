"""Tail-79 failure pins — nomorals/missions/mission.py.

Four sites are EXCLUSIONS (children() query tolerance, the
module's uniform corrupt-row skip in children()/list(), and
archive()'s missing-reflections-table tolerance).

Sites 1047 + 1053 are CONVERTS.  Both live in
``MissionStore.live_with_lock`` — the store's one
negative-proof surface: its only job is proving NO live
mission holds a lock key, and its sole production consumer
(``MissionRunner.create_and_run``, runner.py) starts running
a new mission whenever it reads [].  A crashed query (1047)
or an unreadable live holder row (1053) produced exactly
that [] — byte-identical to "lock free" — so the runner
would start a duplicate mission on the same goal, the
duplicate-side-effect outcome the lock exists to prevent.
The store's own idiom is fail-fast (list()/stats()/archive()
queries are unguarded; get() lets from_row errors
propagate), so both guards were removed: the function now
answers only when it can prove the negative, and fails fast
otherwise.  The pins assert the NEW behavior; A/B proof in
nm-tail79-verify/PROGRESS.md.

Probed live first (nm-tail79-verify/probes.py).
"""
import json
import time

import pytest

from nomorals.missions.mission import MissionStore
from nomorals.storage.db import Database


def _store(tmp_path, name="missions.db"):
    db = Database(str(tmp_path / name))
    db.migrate()
    return MissionStore(db), db


def _boom(*a, **k):
    raise RuntimeError("query boom")


def _insert_corrupt(db, mission_id, *, parent_id="", lock_key=None,
                    status="pending"):
    """A row Mission.from_row cannot parse (budget_wall 'abc'
    makes float() raise) that otherwise looks like a normal
    live row — including, optionally, a lock_key."""
    metadata = {"priority": 0, "tags": [], "parent_id": parent_id}
    if lock_key is not None:
        metadata["lock_key"] = lock_key
    now = time.time()
    db.execute(
        "INSERT INTO missions (id, name, goal, status, state,"
        " budget_wall, budget_tokens, spent_wall, spent_tokens,"
        " iterations, success, created_at, updated_at,"
        " finished_at, metadata)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mission_id, "corrupt", "corrupt goal", status, "{}", "abc",
         0, 0, 0, 0, None, now, now, None, json.dumps(metadata)))


# ── 662 (E104) children(): query failure → [] ──────────────────

def test_662_children_query_failure_returns_empty(tmp_path):
    store, db = _store(tmp_path)
    parent = store.create_new("parent goal")
    child = store.create_new("child goal", parent_id=parent.id)
    assert [c.id for c in store.children(parent.id)] == [child.id]
    db.query = _boom
    try:
        # a crashed scan reads as a leaf mission — the same []
        # a genuine leaf returns; descendants() inherits it.
        # No production surface decides anything from this
        # (traversal/display only), and children() maintains
        # no error channel by design.
        assert store.children(parent.id) == []
        assert store.descendants(parent.id) == []
    finally:
        del db.query
    assert [c.id for c in store.children(parent.id)] == [child.id]


# ── 668 (E104) children(): corrupt row skipped ─────────────────

def test_668_children_skips_corrupt_row(tmp_path):
    store, db = _store(tmp_path)
    parent = store.create_new("parent goal")
    child = store.create_new("child goal", parent_id=parent.id)
    _insert_corrupt(db, "corrupt1", parent_id=parent.id)
    # the corrupt child is silently absent; the good child
    # still lists, and the row itself is untouched
    assert [c.id for c in store.children(parent.id)] == [child.id]
    assert db.scalar(
        "SELECT COUNT(*) FROM missions WHERE id='corrupt1'") == 1


# ── 918 (E104) list(): corrupt row skipped ─────────────────────

def test_918_list_skips_corrupt_row(tmp_path):
    store, db = _store(tmp_path)
    good = store.create_new("good goal")
    _insert_corrupt(db, "corrupt1")
    listed = store.list(limit=50)
    # one unreadable row never kills the listing for every
    # other mission (the module's uniform row idiom)
    assert good.id in [m.id for m in listed]
    assert "corrupt1" not in [m.id for m in listed]
    assert db.scalar(
        "SELECT COUNT(*) FROM missions WHERE id='corrupt1'") == 1


# ── 1014 (E103) archive(): missing reflections table ───────────

def test_1014_archive_tolerates_missing_reflections_table(tmp_path):
    store, db = _store(tmp_path)
    old = store.create_new("old done goal")
    old.status = "done"
    old.finished_at = time.time() - 40 * 86400
    store.save(old)
    db.execute(
        "INSERT INTO reflections (id, mission_id, score, summary,"
        " lessons, weights, created_at) VALUES (?,?,?,?,?,?,?)",
        ("rfl1", old.id, 1.0, "s", "[]", "{}", time.time()))
    db.execute("DROP TABLE reflections")
    # the delete against the not-yet-created table is absorbed
    # and the archive proceeds: the mission is deleted and the
    # counts stay literally true (0 reflections were deleted)
    out = store.archive(older_than_seconds=30 * 86400)
    assert out["missions"] == 1
    assert out["reflections"] == 0
    assert db.scalar(
        "SELECT COUNT(*) FROM missions WHERE id=?", (old.id,)) == 0
    # control: with the table present the same archive counts
    # the reflection it deletes
    store2, db2 = _store(tmp_path, "missions2.db")
    old2 = store2.create_new("old done two")
    old2.status = "done"
    old2.finished_at = time.time() - 40 * 86400
    store2.save(old2)
    db2.execute(
        "INSERT INTO reflections (id, mission_id, score, summary,"
        " lessons, weights, created_at) VALUES (?,?,?,?,?,?,?)",
        ("rfl2", old2.id, 1.0, "s", "[]", "{}", time.time()))
    out2 = store2.archive(older_than_seconds=30 * 86400)
    assert out2["missions"] == 1
    assert out2["reflections"] == 1


# ── 1047 (E104) CONVERTED: live_with_lock query failure ────────

def test_1047_lock_query_failure_raises_instead_of_free(tmp_path):
    store, db = _store(tmp_path)
    holder = store.create_new("lock holder",
                              metadata={"lock_key": "k1"})
    # healthy behavior unchanged: holder found, other keys free
    assert [m.id for m in store.live_with_lock("k1")] == [holder.id]
    assert store.live_with_lock("other") == []
    assert store.live_with_lock("") == []
    db.query = _boom
    try:
        # the guard is gone: a crashed lock scan propagates
        # instead of reporting the lock free to the runner
        with pytest.raises(RuntimeError, match="query boom"):
            store.live_with_lock("k1")
    finally:
        del db.query
    assert [m.id for m in store.live_with_lock("k1")] == [holder.id]


# ── 1053 (E104) CONVERTED: live_with_lock corrupt holder row ───

def test_1053_lock_corrupt_holder_row_raises_instead_of_free(tmp_path):
    store, db = _store(tmp_path)
    holder = store.create_new("clean holder",
                              metadata={"lock_key": "k2"})
    assert [m.id for m in store.live_with_lock("k2")] == [holder.id]
    _insert_corrupt(db, "corrupt-holder", lock_key="k9",
                    status="running")
    # an unreadable LIVE row defeats the negative proof, so
    # the parse failure propagates (the same ValueError get()
    # surfaces for the same row) instead of the holder being
    # silently invisible to the lock check
    with pytest.raises(ValueError):
        store.live_with_lock("k9")
