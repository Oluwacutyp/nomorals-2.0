"""Tail-109 pins for nomorals/tools/shell.py — 4 exclusions, 0 flagged-accepted
except site 100 (see below).

Each test pins one census site's fallback behavior plus a control half
proving the except path was really taken. The sites sit in the sandbox
child setup/teardown; none touches a failures ledger (the module's only
recording is _set's own debug log on the setrlimit path, not a census
site), so non-recording on these paths is structural.

Sites:
* 91 E104 — SandboxLimits.preexec(): the ambient thread-count probe
  (/proc) crashes -> ambient = 0 and the CONFIGURED process ceiling is
  kept, exactly as the in-code comment contracts. The ceiling still
  caps runaway forking; only the ambient raise is forgone.
* 98 E103 — preexec(): os.setsid() OSError -> pass ("already a leader",
  pragma: no cover). Same adjudication family as process_isolation's
  setsid guard (tail-102): in the preexec child the raise is the
  documented no-op case and the rlimits have already been applied.
* 100 E103 — preexec(): os.setsid() AttributeError (no setsid on
  Windows) -> pass. FLAGGED-ACCEPTED: this site is in the flagged-scan
  baseline (its comment carries no scanner-honored noqa) and stays
  there; production is untouched.
* 321 E103 — _kill_tree(): killpg/getpgid failed, the fallback
  process.kill() raises OSError ("process already gone") -> pass.
  Teardown of an already-dead process; nothing is claimed by it.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from nomorals.tools import shell

pytestmark = pytest.mark.skipif(shell.resource is None,
                                reason="POSIX resource module required")


def _run_preexec(limits, ambient_fn, setsid_fn):
    """Run preexec with _set recorded and the ambient probe / setsid stubbed."""
    calls = []
    with mock.patch.object(shell, "_set",
                           lambda which, soft, hard: calls.append((which, soft, hard))), \
            mock.patch.object(shell, "_uid_thread_count", ambient_fn), \
            mock.patch.object(os, "setsid", setsid_fn):
        limits.preexec()
    return calls


def _nproc_soft(calls):
    hits = [soft for which, soft, _ in calls
            if which == shell.resource.RLIMIT_NPROC]
    assert len(hits) == 1
    return hits[0]


# ── 91: ambient probe crash keeps the configured ceiling ─────────────────────

def test_91_ambient_probe_crash_keeps_configured_ceiling() -> None:
    limits = shell.SandboxLimits(max_processes=10)
    # Control: ambient 10 threads -> ceiling floored to ambient + 64.
    calls = _run_preexec(limits, lambda: 10, lambda: None)
    assert _nproc_soft(calls) == 74
    # Trigger: probe crash -> ambient 0 -> the configured ceiling (64)
    # applies, never an unbounded or zero ceiling.
    calls = _run_preexec(
        limits,
        lambda: (_ for _ in ()).throw(RuntimeError("no /proc")),
        lambda: None)
    assert _nproc_soft(calls) == 64


# ── 98 / 100: setsid guards ──────────────────────────────────────────────────

def test_98_setsid_oserror_swallowed() -> None:
    limits = shell.SandboxLimits()
    calls = _run_preexec(
        limits, lambda: 0,
        lambda: (_ for _ in ()).throw(OSError("already a leader")))
    # preexec completed: every rlimit was still applied before the guard.
    assert len(calls) == 5


def test_100_setsid_attributeerror_swallowed() -> None:
    limits = shell.SandboxLimits()
    calls = _run_preexec(
        limits, lambda: 0,
        lambda: (_ for _ in ()).throw(AttributeError("no setsid")))
    assert len(calls) == 5


# ── 321: kill fallback on an already-gone process ────────────────────────────

class _FakeProc:
    def __init__(self, kill_exc=None):
        self.pid = 4321
        self.kill_calls = 0
        self.kill_exc = kill_exc

    def kill(self):
        self.kill_calls += 1
        if self.kill_exc is not None:
            raise self.kill_exc


def test_321_kill_tree_kill_oserror_swallowed() -> None:
    # Control: killpg works -> the fallback kill is never attempted.
    proc = _FakeProc()
    with mock.patch.object(os, "killpg", lambda pg, sig: None), \
            mock.patch.object(os, "getpgid", lambda pid: pid):
        shell._kill_tree(proc)
    assert proc.kill_calls == 0
    # Trigger: the group is gone AND the direct kill raises OSError ->
    # swallowed; the process was already dead, which is the goal state.
    proc = _FakeProc(kill_exc=OSError("already gone"))
    with mock.patch.object(
            os, "killpg",
            lambda pg, sig: (_ for _ in ()).throw(ProcessLookupError())), \
            mock.patch.object(os, "getpgid", lambda pid: pid):
        shell._kill_tree(proc)
    assert proc.kill_calls == 1
