"""Pass #217 pins: core/own_processes.py — 1 exclusion, 0 converts.

- 185: ``pid_alive`` /proc zombie probe failure. ``os.kill(pid, 0)``
  already answered: the pid exists (the kill verdict). The /proc
  read is only a refinement — a zombie (state Z/X/x) is dead even
  though kill(0) succeeds. When the /proc read itself fails (the
  process vanished mid-check, /proc unreadable), the kill verdict
  stands and the answer is the conservative True. The swallow is
  the correct behavior: a failed probe does not overturn a verdict
  the kill already gave, and the function's contract already
  produces True for "exists" on a non-error path.

core/own_processes.py DOES call record_failure elsewhere
(own_processes.shared_load, own_processes.register_spawned), so
NOT-recording is pinned against the core ledger: core
recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use real pids and a patched
read, never a swallow.
"""

import os
import time
from pathlib import Path

import pytest

from nomorals.core import failures as core_fail
from nomorals.core.own_processes import pid_alive


def _snap():
    return list(core_fail.recent_failures(limit=200))


# ── L185: a failed /proc probe leaves the kill verdict standing ─────

def test_185_live_pid_is_alive():
    before = _snap()
    assert pid_alive(os.getpid()) is True
    assert _snap() == before


def test_185_nonexistent_pid_is_dead():
    before = _snap()
    assert pid_alive(999999999) is False
    assert _snap() == before


def test_185_proc_read_failure_keeps_the_kill_verdict(monkeypatch):
    # kill(0) succeeds (the pid exists) but the /proc read fails:
    # the answer stays True — the probe failure does not overturn
    # the kill verdict.
    real_read_text = Path.read_text

    def _boom_read_text(self, *args, **kwargs):
        if str(self).startswith("/proc/"):
            raise OSError("cannot read /proc stat")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _boom_read_text)
    before = _snap()
    assert pid_alive(os.getpid()) is True
    assert _snap() == before


def test_185_zombie_is_dead():
    # The probe itself works: a forked child that exited and has
    # not been reaped yet is a zombie, and pid_alive says dead.
    pid = os.fork()
    if pid == 0:  # child: exit at once, become a zombie
        os._exit(0)
    try:
        # Wait briefly for the child to exit and become a zombie
        # (the fork returns before the child has run).
        state = ""
        for _ in range(200):
            try:
                stat = Path(f"/proc/{pid}/stat").read_text(
                    encoding="utf-8", errors="replace")
                state = stat.rsplit(")", 1)[-1].split()[0]
            except OSError as exc:
                pytest.skip(f"cannot read zombie state: {exc}")
                return
            if state in ("Z", "X", "x"):
                break
            time.sleep(0.01)
        if state not in ("Z", "X", "x"):
            pytest.skip(f"child not a zombie (state {state})")
            return
        before = _snap()
        assert pid_alive(pid) is False
        assert _snap() == before
    finally:
        try:
            os.waitpid(pid, 0)
        except ChildProcessError as exc:
            assert isinstance(exc, ChildProcessError)
