"""Pass tail-118 pins — nomorals/social/chat/single_consumer.py (3 exclusions).

Sites pinned (all E103 teardown/cleanup swallows in the lockfile guard):
- 125: payload write inside ``acquire()`` fails AND the cleanup
  ``os.close(fd)`` raises too -> the swallow changes nothing: acquire
  still returns False. The lock's contention verdict stays fail-closed.
- 139: ``os.close`` raises inside ``release()`` -> release proceeds:
  the fd handle is cleared and the lockfile unlink still runs.
- 148: the holder-check/unlink block in ``release()`` raises -> the
  lockfile may be stranded, but the design self-heals: the stranded
  file names a dead pid once the holder exits, and the next acquirer's
  stale-reclaim removes it and takes the lock (both halves pinned).

The module has no logger at all — non-recording is structural.
"""

from __future__ import annotations

import json
import os

import pytest

from nomorals.social.chat.single_consumer import SingleConsumerLock


def _boom(exc):
    def _raise(*a, **k):
        raise exc
    return _raise


def test_healthy_acquire_holder_release_cycle(tmp_path):
    path = tmp_path / "bot.lock"
    lock = SingleConsumerLock(path)
    assert lock.acquire() is True
    assert (lock.holder() or {}).get("pid") == os.getpid()
    lock.release()
    assert not path.exists()


def test_125_write_failure_and_close_crash_acquire_false(tmp_path, monkeypatch):
    lock = SingleConsumerLock(tmp_path / "bot.lock")
    monkeypatch.setattr(os, "write", _boom(OSError("disk gone")))
    monkeypatch.setattr(os, "close", _boom(OSError("close boom")))
    assert lock.acquire() is False
    assert lock._fd is None


def test_139_close_crash_release_still_releases(tmp_path, monkeypatch):
    path = tmp_path / "bot.lock"
    lock = SingleConsumerLock(path)
    assert lock.acquire() is True
    monkeypatch.setattr(os, "close", _boom(OSError("close boom")))
    lock.release()  # must not raise
    assert lock._fd is None
    assert not path.exists()


def test_148_unlink_crash_strands_then_stale_reclaim(tmp_path, monkeypatch):
    path = tmp_path / "bot.lock"
    lock = SingleConsumerLock(path)
    assert lock.acquire() is True
    monkeypatch.setattr(type(path), "unlink", _boom(OSError("unlink boom")))
    lock.release()  # must not raise; the lockfile is stranded
    assert path.exists()
    monkeypatch.undo()
    # The stranded file names a dead pid -> next acquirer reclaims it.
    path.write_text(json.dumps({"pid": 99999999, "started_at": 0.0}))
    nxt = SingleConsumerLock(path)
    assert nxt.acquire() is True
    nxt.release()
    assert not path.exists()
