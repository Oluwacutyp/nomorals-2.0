"""Pass #218 pins: tools/proxy_sources.py — 1 exclusion, 0 converts.

- 761: ``_ActivationLock.release`` lock-read OSError. Dropping the
  fence reads the lock file to check the token is still ours; when
  the read (or the unlink) raises OSError the code falls through and
  clears ``self.token`` anyway. The release is best-effort: the lock
  file may be gone already (a stolen lock's new holder, a crashed
  peer, a cleaned tmp dir) and that is exactly the state release()
  wants — the holder holds nothing afterwards. The token is cleared
  on every path, so a released lock can never be released twice.

tools/proxy_sources.py calls no failures-ledger record function
anywhere, so NOT-recording is pinned with the agency snapshot method:
agency recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use a deleted lock file
(read_text raises FileNotFoundError), never a swallow.
"""

import json

from nomorals.agents import failures as ag_fail
from nomorals.tools.proxy_sources import _ActivationLock


def _snap():
    return list(ag_fail.recent_failures(limit=200))


def _lock(tmp_path):
    return _ActivationLock(tmp_path / "registry.json")


# ── L761: the unreadable lock file still releases the token ───────────


def test_761_acquire_then_release_removes_the_lock(tmp_path):
    lock = _lock(tmp_path)
    token = lock.acquire()
    assert token
    assert lock.path.exists()
    before = _snap()
    lock.release()
    assert lock.token == ""
    assert not lock.path.exists()
    assert _snap() == before


def test_761_release_with_deleted_lock_file_is_silent(tmp_path):
    # The lock file is gone before release: read_text raises
    # FileNotFoundError (an OSError) — the probe passes, the token
    # is still cleared, and nothing raises.
    lock = _lock(tmp_path)
    token = lock.acquire()
    assert token
    lock.path.unlink()
    before = _snap()
    lock.release()
    assert lock.token == ""
    assert _snap() == before


def test_761_release_with_unreadable_lock_file_is_silent(tmp_path):
    # The lock path is a directory: read_text raises IsADirectoryError
    # (an OSError) — same fall-through, token cleared, nothing raised.
    lock = _lock(tmp_path)
    token = lock.acquire()
    assert token
    lock.path.unlink()
    lock.path.mkdir()
    before = _snap()
    lock.release()
    assert lock.token == ""
    assert lock.path.is_dir()
    assert _snap() == before


def test_761_release_never_unlinks_a_stolen_lock(tmp_path):
    # Another holder owns the fence now (a different token in the
    # file): release clears OUR token but leaves THEIR lock alone.
    lock = _lock(tmp_path)
    token = lock.acquire()
    assert token
    lock.path.write_text(
        json.dumps({"token": "someone-else", "pid": 99999,
                    "at": 0.0}),
        encoding="utf-8")
    before = _snap()
    lock.release()
    assert lock.token == ""
    assert lock.path.exists()
    data = json.loads(lock.path.read_text(encoding="utf-8"))
    assert data["token"] == "someone-else"
    assert _snap() == before


def test_761_release_without_a_token_is_a_noop(tmp_path):
    lock = _lock(tmp_path)
    assert lock.token == ""
    before = _snap()
    lock.release()
    assert lock.token == ""
    assert not lock.path.exists()
    assert _snap() == before
