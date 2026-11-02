"""Tail-102 pins: nomorals/core/process_isolation.py census
sites — all inside ``run_isolated``'s fork child or its timeout
kill path, all held as exclusions.

The module's contract: the parent always gets an HONEST verdict
(ok / error / crashed / timeout-killed / not-supported). Every
site guards a step whose failure cannot change that honesty —
the outcome is still reported through the status channel, or
the desired end state already holds. All four triggers were
probed live before pinning.

82 — child ``os.setsid()`` OSError -> pass, the call proceeds.
WEAKEST EXCLUSION of this batch, named plainly: the module's
headline promise is that a timeout verdict means the whole
process TREE is dead, and setsid is what creates the group the
parent's killpg targets — a silent setsid failure would void
exactly that promise (killpg misses, the fallback kills only
the child, grandchildren could survive a "timeout-killed"
verdict). Held as exclusion on reachability evidence: a freshly
forked child is never a process-group leader (its pid is newly
allocated; its pgid is inherited), and setsid fails with EPERM
only for a group leader — the raise is unreachable in the only
context this code runs in. A sketch to surface a setsid failure
in the child's payload (a degraded-isolation flag on the
result) is shelved as a follow-up, not a convert.
91 — pickling the child's error report itself fails (an
exception whose ``__repr__`` raises) -> payload degrades to
(False, "call raised"). The raise is still REPORTED: the parent
returns status "error" carrying that text. Only the message
degrades, never the verdict.
95 — child ``os.write`` of the result payload OSError -> pass
("parent gone; nothing to do"). The child exits regardless; the
parent reads EOF and reports "child exited without a result" —
again an honest error verdict, probe-proven below.
122 — timeout path: ``os.killpg`` raises, fallback ``os.kill``
raises ProcessLookupError -> pass. The kill target is already
gone, which is the end state the kill exists to reach; the
parent still reaps and returns "timeout-killed".

NOT-recording is pinned with the snapshot method: the module
calls no failures-ledger record function anywhere (and has no
logger), so both general ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

import os
import time

import pytest

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.core import process_isolation as pi


def _snapshot():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


pytestmark = pytest.mark.skipif(not pi.fork_supported(),
                                reason="os.fork unavailable on this platform")


def test_82_setsid_failure_child_still_reports(monkeypatch):
    before = _snapshot()

    def _no_session():
        raise OSError("no session")

    monkeypatch.setattr(os, "setsid", _no_session)
    result = pi.run_isolated(lambda: 41 + 1, timeout_s=10)
    assert result.status == "ok"
    assert result.value == 42
    assert _snapshot() == before


def test_91_unpicklable_error_repr_degrades_message_not_verdict():
    before = _snapshot()

    class BadRepr(Exception):
        def __repr__(self):
            raise RuntimeError("repr boom")

    def raiser():
        raise BadRepr("x")

    result = pi.run_isolated(raiser, timeout_s=10)
    assert result.status == "error"
    assert result.error == "call raised"
    assert _snapshot() == before


def test_95_child_write_failure_surfaces_as_honest_error(monkeypatch):
    before = _snapshot()
    main_pid = os.getpid()
    real_write = os.write

    def write_bomb(fd, data):
        if os.getpid() != main_pid:
            raise OSError("parent gone")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", write_bomb)
    result = pi.run_isolated(lambda: 7, timeout_s=10)
    assert result.status == "error"
    assert result.error == "child exited without a result"
    assert _snapshot() == before


def test_122_kill_fallback_target_already_gone(monkeypatch):
    before = _snapshot()

    def _gone(*_args):
        raise ProcessLookupError()

    monkeypatch.setattr(os, "killpg", _gone)
    monkeypatch.setattr(os, "kill", _gone)
    result = pi.run_isolated(lambda: time.sleep(5), timeout_s=0.3)
    assert result.status == "timeout-killed"
    assert _snapshot() == before
