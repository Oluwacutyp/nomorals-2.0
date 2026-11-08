"""Batch B — F02: queue terminal writes are fenced to the current
lease generation.

Failing-first evidence: on base, ``Job`` carries no lease token and
``complete``/``fail``/``requeue`` accept no fencing argument — a
stale worker's completion overwrites the current owner's job.
"""

from __future__ import annotations

import time

import pytest

from nomorals.storage.db import Database
from nomorals.storage.queue import WorkQueue


def _queue(tmp_path) -> tuple[WorkQueue, Database]:
    db = Database(str(tmp_path / "q.db"))
    db.migrate()
    return WorkQueue(db), db


def _expire(db: Database, job_id: str) -> None:
    db.execute("UPDATE work_queue SET lease_until = ? WHERE id = ?",
               (time.time() - 1, job_id))


def test_stale_completion_cannot_commit_another_workers_job(tmp_path):
    q, db = _queue(tmp_path)
    jid = q.enqueue("t", {"n": 1})
    a = q.lease_one("t", worker="a")
    assert a is not None and a.lease_token
    _expire(db, jid)
    b = q.lease_one("t", worker="b")
    assert b is not None and b.lease_token
    assert b.lease_token != a.lease_token
    assert q.complete(jid, {"by": "a"}, lease_token=a.lease_token) is False
    row = q.get(jid)
    assert row is not None and row.status == "leased"
    assert row.lease_owner == "b"
    assert q.complete(jid, {"by": "b"}, lease_token=b.lease_token) is True
    assert q.result_of(jid) == {"by": "b"}


def test_stale_fail_and_requeue_are_refused(tmp_path):
    q, db = _queue(tmp_path)
    jid = q.enqueue("t", {"n": 1})
    a = q.lease_one("t", worker="a")
    assert a is not None
    _expire(db, jid)
    b = q.lease_one("t", worker="b")
    assert b is not None
    assert q.fail(jid, "a died", lease_token=a.lease_token) == "stale"
    row = q.get(jid)
    assert row is not None and row.status == "leased"
    assert row.lease_owner == "b"
    assert q.requeue(jid, lease_token=a.lease_token) is False
    assert q.fail(jid, "b gives up", retry=False,
                  lease_token=b.lease_token) == "dead"
    row = q.get(jid)
    assert row is not None and row.status == "dead"


def test_same_worker_name_reuse_is_a_new_generation(tmp_path):
    q, db = _queue(tmp_path)
    jid = q.enqueue("t", {"n": 1})
    first = q.lease_one("t", worker="w")
    assert first is not None
    _expire(db, jid)
    second = q.lease_one("t", worker="w")
    assert second is not None
    assert second.lease_token != first.lease_token
    assert q.complete(jid, {"gen": 1},
                      lease_token=first.lease_token) is False
    assert q.complete(jid, {"gen": 2},
                      lease_token=second.lease_token) is True
    assert q.result_of(jid) == {"gen": 2}


def test_extend_lease_is_fenced_to_the_current_generation(tmp_path):
    q, db = _queue(tmp_path)
    jid = q.enqueue("t", {"n": 1})
    a = q.lease_one("t", worker="a")
    assert a is not None
    _expire(db, jid)
    b = q.lease_one("t", worker="b")
    assert b is not None
    assert q.extend_lease(jid, "b", 600, lease_token=a.lease_token) is False
    assert q.extend_lease(jid, "b", 600, lease_token=b.lease_token) is True


def test_worker_loop_completion_is_fenced(tmp_path):
    q, db = _queue(tmp_path)
    jid = q.enqueue("t", {"n": 1})

    def handler(job):
        # While the first worker runs, its lease expires and a second
        # worker reclaims the job and completes it.
        _expire(db, jid)
        other = q.lease_one("t", worker="other")
        assert other is not None
        assert q.complete(jid, {"by": "other"},
                          lease_token=other.lease_token) is True
        return {"by": "first"}

    processed = q.run_worker("t", handler, worker="first", max_jobs=1)
    assert processed == 1
    assert q.result_of(jid) == {"by": "other"}, (
        "the stale worker's late completion overwrote the current "
        "owner's result")


def test_unfenced_completion_remains_the_admin_path(tmp_path):
    q, _db = _queue(tmp_path)
    jid = q.enqueue("t", {})
    job = q.lease_one("t", worker="w")
    assert job is not None
    # No token: the deliberate administrative completion API.
    q.complete(jid, {"admin": True})
    assert q.result_of(jid) == {"admin": True}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
