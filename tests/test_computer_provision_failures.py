"""Pass #216 pins: computer/provision.py (process_alive) — 2 exclusions, 0 converts.

Both sites are in ``process_alive``, which answers "is this pid
alive" with a probe ladder. Each excluded handler falls through to
the next probe (or to the conservative answer), so no failure is
hidden:

- 306: waitpid reap ``except (ChildProcessError, OSError): pass``.
  The waitpid is a reap attempt for our own child; on failure the
  ``kill(pid, 0)`` probe decides aliveness. The failure does not
  hide anything — the probe runs next.
- 320: /proc/{pid}/stat zombie check ``except (OSError, IndexError):
  pass``. The zombie check is a refinement; on failure the process
  is treated as alive — the conservative answer, since the kill
  probe already established it exists. A vanished process (race)
  or a host without /proc both fall through to "alive", and the
  caller re-checks.

computer/provision.py records failures to its own ledger
(nomorals.computer.failures), so NOT-recording is pinned by
snapshotting that ledger's recent_failures() before/after each
trigger.

Pins rule (#169): no bare-swallow handler anywhere in this file —
the scanner counts test files too. Triggers use real child
processes and a mocked /proc read, instead of swallowing.
"""

import os
import subprocess
from unittest.mock import patch

from nomorals.computer import failures as cf
from nomorals.computer import provision as pv


def _snap():
    return list(cf.recent_failures(500))


# ── L306: waitpid reap failure falls through to the kill probe ───

def test_306_non_child_live_pid_is_alive():
    # A pid that is not our child: waitpid raises ChildProcessError,
    # the kill probe then decides. Our own parent shell's pid is alive.
    before = _snap()
    assert pv.process_alive(os.getppid()) is True
    assert _snap() == before


def test_306_dead_pid_is_not_alive():
    before = _snap()
    # A pid that never existed: waitpid fails, kill(pid, 0) raises
    # ProcessLookupError -> False.
    assert pv.process_alive(424242) is False
    assert _snap() == before


def test_306_reaped_child_is_not_alive():
    # A real child that we spawn and reap: waitpid returns done, so
    # process_alive is False.
    before = _snap()
    proc = subprocess.Popen(["true"])
    proc.wait()
    assert pv.process_alive(proc.pid) is False
    assert _snap() == before


# ── L320: /proc zombie check failure falls through to "alive" ────

def test_320_proc_read_failure_falls_through_to_alive():
    # A live child whose /proc stat read raises OSError is treated
    # as alive (the kill probe already established it exists).
    before = _snap()
    proc = subprocess.Popen(["sleep", "30"])
    try:
        calls = []

        class _FailingPath:
            def __init__(self, *a, **k):
                pass

            def read_text(self, *a, **k):
                calls.append(True)
                raise OSError(2, "No such file")

        with patch.object(pv, "Path", _FailingPath):
            assert pv.process_alive(proc.pid) is True
        assert calls, "the /proc read was attempted"
    finally:
        proc.kill()
        proc.wait()
    assert _snap() == before


def test_320_proc_stat_zombie_is_not_alive():
    # A /proc stat showing state Z means the process is a zombie
    # (dead for every practical purpose) -> False.
    before = _snap()
    proc = subprocess.Popen(["sleep", "30"])
    try:
        class _ZombiePath:
            def __init__(self, *a, **k):
                pass

            def read_text(self, *a, **k):
                # stat format: pid (comm) state ... — state is the
                # first field after the closing paren.
                return "12345 (sleep) Z 1 2 3"

        with patch.object(pv, "Path", _ZombiePath):
            assert pv.process_alive(proc.pid) is False
    finally:
        proc.kill()
        proc.wait()
    assert _snap() == before


def test_320_live_child_is_alive():
    # Control: a live child is alive (kill probe succeeds, /proc
    # stat shows a non-zombie state).
    before = _snap()
    proc = subprocess.Popen(["sleep", "30"])
    try:
        assert pv.process_alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait()
    assert _snap() == before
