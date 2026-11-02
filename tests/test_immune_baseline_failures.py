"""Tail-89 pins: nomorals/immune/baseline.py census sites.

The module's design law is printed in its own header: every
platform backend returns the live process COUNT or None —
"never a fabricated number" — and ``sampling_availability``
reports exactly why no sample exists, so callers can
distinguish "platform cannot sample" from "no sample this
tick". All five sites are rungs of that one fallback ladder
and are held as exclusions:

244 — win32 backend, ``import ctypes`` crash -> None. No
ctypes, no backend; the ladder moves on to psutil.
263 — win32 backend, any ctypes/EnumProcesses failure ->
None. A failed count is not a count; None is the honest
no-sample value, consumed as such by _process_count.
273 — psutil backend, ``import psutil`` crash -> None. The
optional accelerator is absent; the ladder has already tried
the stdlib backend first.
277 — psutil backend, ``pids()`` crash -> None ("a broken
psutil must not lie"). The failure value is the same None
the absent-backend rungs return — never an invented count.
312 — ``_process_count`` dispatch: a backend that RAISES is
treated as a failed backend (count=None) and the next rung
is tried. When every rung fails the aggregate returns
(None, "", reason) with the attempted backends named — the
reason is the surface, via sampling_availability.

NOT-recording is pinned with the snapshot method: baseline.py
calls no failures-ledger record function anywhere, so both
the agency and agents ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

import ctypes
import sys
from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.immune import baseline as B


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


# ── 244: no ctypes -> no win32 backend ──────────────────────────

def test_244_win32_without_ctypes_returns_none():
    before = _snap()
    with mock.patch.dict(sys.modules, {"ctypes": None}):
        assert B._count_processes_win32() is None
    assert _snap() == before


# ── 263: ctypes failure -> no count ─────────────────────────────

def _enum_boom(*args):
    raise RuntimeError("enum boom")


def _enum_ok(byref_buf, size, byref_ret):
    ctypes.cast(byref_ret,
                ctypes.POINTER(ctypes.c_ulong)).contents.value = (
        3 * ctypes.sizeof(ctypes.c_ulong))
    return 1


def _fake_ct(enum):
    return SimpleNamespace(
        c_ulong=ctypes.c_ulong, byref=ctypes.byref,
        sizeof=ctypes.sizeof,
        windll=SimpleNamespace(
            psapi=SimpleNamespace(EnumProcesses=enum)))


def test_263_win32_enum_failure_returns_none():
    before = _snap()
    assert B._count_processes_win32(_fake_ct(_enum_boom)) is None
    # control: a working EnumProcesses yields an int count
    # (all-zero pids -> 0 live processes after the PID-0 cut)
    result = B._count_processes_win32(_fake_ct(_enum_ok))
    assert isinstance(result, int) and result == 0
    assert _snap() == before


# ── 273: no psutil -> no psutil backend ─────────────────────────

def test_273_psutil_missing_returns_none():
    before = _snap()
    with mock.patch.dict(sys.modules, {"psutil": None}):
        assert B._count_processes_psutil() is None
    assert _snap() == before


# ── 277: broken psutil -> no count ──────────────────────────────

def test_277_psutil_failure_returns_none():
    before = _snap()
    assert B._count_processes_psutil(
        SimpleNamespace(pids=_raise)) is None
    # control: a healthy psutil counts its pids
    assert B._count_processes_psutil(
        SimpleNamespace(pids=lambda: [11, 22, 33])) == 3
    assert _snap() == before


# ── 312: a raising backend is a failed backend ──────────────────

def test_312_raising_backend_is_skipped():
    before = _snap()
    # proc raises -> the psutil rung still answers
    assert B._process_count(os_name="posix", proc_count_fn=_raise,
                            psutil_count_fn=lambda: 7) == (7, "psutil", "")
    # every rung fails -> (None, "", reason naming what was tried)
    count, backend, reason = B._process_count(
        os_name="posix", proc_count_fn=_raise, psutil_count_fn=_raise)
    assert count is None and backend == ""
    assert "proc" in reason and "psutil" in reason
    # control: the first healthy rung wins
    assert B._process_count(os_name="posix", proc_count_fn=lambda: 5,
                            psutil_count_fn=_raise) == (5, "proc", "")
    # end to end: no count -> no sample, and availability says why
    assert B.collect_process_sample(
        count_fn=lambda: (None, "", "x")) is None
    avail = B.sampling_availability(
        count_fn=lambda: (None, "", "no backend"))
    assert avail["available"] is False
    assert avail["reason"] == "no backend"
    assert _snap() == before
