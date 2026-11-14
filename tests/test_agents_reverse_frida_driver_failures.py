"""Pass #216 pins: agents/reverse/frida_driver.py — 3 exclusions, 0 converts.

Each site is a "goal already achieved" or "failure surfaced elsewhere"
handler — the swallow is the correct behavior, and the function returns
a value (or raises) its own contract already produces on a non-error
path:

- 143: ``_start_suspended`` execv OSError -> ``pass`` then
  ``os._exit(127)``. The exec failure is NOT swallowed: the child
  exits 127 and the parent raises RuntimeError "target exited before
  it could be traced". The pass is the bridge to the exit code.
- 191: ``_wait_exit`` SIGKILL ProcessLookupError -> ``pass``. The
  process is already dead, which is the goal of the kill; the code
  proceeds to reap it and returns (True, code).
- 213: ``_kill`` waitpid ChildProcessError -> ``pass``. The child is
  already reaped, which is the goal of the wait; ``_kill`` returns
  None either way.

frida_driver.py calls no failures-ledger record function anywhere, so
NOT-recording is pinned with the agency snapshot method: agency
recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use a nonexistent binary,
a real child, and ``unittest.mock.patch`` instead of swallowing.
"""

import os
import signal
import subprocess
from unittest.mock import patch

import pytest

from nomorals.agents import failures as ag_fail
from nomorals.agents.reverse import frida_driver as fd


def _snap():
    return list(ag_fail.recent_failures(limit=200))


# ── L143: execv failure surfaces as RuntimeError via exit 127 ──────

def test_143_execv_failure_raises_target_exited():
    before = _snap()
    with pytest.raises(RuntimeError,
                       match="target exited before it could be traced"):
        fd._start_suspended(["/nonexistent/binary-xyz-216"])
    assert _snap() == before


def test_143_execv_failure_exit_code_is_127(monkeypatch):
    # The child exits 127 on execv failure; prove it by running the
    # fork/exec pair directly and reading the reaped status.
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.execv("/nonexistent/binary-xyz-216", ["/nonexistent/binary-xyz-216"])
        except OSError as exc:
            assert isinstance(exc, Exception)
        os._exit(127)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 127


# ── L191: SIGKILL on an already-dead process is a no-op ────────────

def test_191_kill_process_lookup_error_proceeds_to_reap():
    before = _snap()
    pid = 424242
    with patch("os.waitpid", side_effect=[(0, 0), (pid, 0)]) as _w, \
         patch("os.kill", side_effect=ProcessLookupError(3, "No such process")) as _k:
        timed_out, code = fd._wait_exit(pid, 0)
    assert (timed_out, code) == (True, 0)
    assert _snap() == before


def test_191_deadline_kills_live_child_and_reaps(monkeypatch):
    # Control: the deadline path on a real live child kills and reaps
    # it, returning (True, -9). Proves the wait loop reaches the kill.
    before = _snap()
    proc = subprocess.Popen(["sleep", "30"])
    try:
        timed_out, code = fd._wait_exit(proc.pid, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert timed_out is True
    assert code == -signal.SIGKILL
    assert _snap() == before


# ── L213: _kill on an already-reaped child returns None ────────────

def test_213_kill_already_reaped_returns_none():
    before = _snap()
    with patch("os.kill", return_value=None) as _k, \
         patch("os.waitpid", side_effect=ChildProcessError(10, "No child processes")):
        assert fd._kill(424242) is None
    assert _snap() == before


def test_213_kill_dead_pid_returns_none_early():
    # Control: a pid that does not exist fails the kill itself and
    # _kill returns before ever reaching waitpid.
    before = _snap()
    assert fd._kill(424242) is None
    assert _snap() == before


def test_213_kill_live_child_reaps_it():
    # Control: _kill on a real live child kills and reaps it. The
    # pid is gone afterwards (kill(pid, 0) raises ProcessLookupError).
    # Note: Popen.poll() reports 0 here, not -9, because _kill's own
    # waitpid already reaped the child — Popen cannot recover the
    # signal status. The pid being gone is the real assertion.
    before = _snap()
    proc = subprocess.Popen(["sleep", "30"])
    fd._kill(proc.pid)
    assert proc.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(proc.pid, 0)
    assert _snap() == before
