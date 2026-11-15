"""Pass #217 pins: social/bridge_process.py — 1 exclusion, 0 converts.

- 351: ``_linux_process_image`` cmdline read failure. The exe path
  is read first (its own OSError degrades to ""); the cmdline read
  is a second probe — a process that vanished between the two reads,
  or one whose /proc entry the caller cannot read, leaves cmdline
  as "". The function still returns the exe when it has one, and
  returns None only when BOTH are empty. The swallow is the correct
  behavior: an unreadable cmdline is not a failure, and the
  function's contract already produces "" / None for "no image"
  on a non-error path.

social/bridge_process.py calls no failures-ledger record function
anywhere, so NOT-recording is pinned with the agency snapshot
method: agency recent_failures() before/after each trigger must be
unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use real pids, never a swallow.
"""

import os

from nomorals.agents import failures as ag_fail
from nomorals.social.bridge_process import _linux_process_image


def _snap():
    return list(ag_fail.recent_failures(limit=200))


# ── L351: an unreadable cmdline degrades to "" ──────────────────────

def test_351_live_process_returns_exe_and_cmdline():
    before = _snap()
    image = _linux_process_image(os.getpid())
    assert image is not None
    exe, cmdline = image
    # The test runner is a real process with a real image.
    assert exe or cmdline
    assert _snap() == before


def test_351_nonexistent_pid_returns_none():
    # Both probes fail (no /proc entry at all): exe "" and cmdline
    # "" -> None. The cmdline OSError is swallowed, not raised.
    before = _snap()
    assert _linux_process_image(999999999) is None
    assert _snap() == before


def test_351_cmdline_probe_failure_keeps_the_exe(monkeypatch):
    # The exe read succeeds but the cmdline read fails: the exe is
    # still returned with an empty cmdline, not an exception.
    import nomorals.social.bridge_process as bp

    real_open = open

    def _boom_open(path, *args, **kwargs):
        if str(path).endswith("/cmdline"):
            raise OSError("cannot read cmdline")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(bp, "open", _boom_open, raising=False)
    before = _snap()
    exe, cmdline = _linux_process_image(os.getpid())
    assert cmdline == ""
    assert exe  # the exe probe still landed
    assert _snap() == before
