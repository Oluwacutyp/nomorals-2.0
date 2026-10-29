"""Tail-48 pins for nomorals/os/resources.py — 0 converts /
8 exclusions, 0 flagged.

8 tests for 8 sites (counts equal). Each test pins one site's
fallback behavior plus a control half proving the guarded path
was actually exercised, asserting via the double's own call log
that the failing operation itself was attempted. The file
contains no exception handlers, so it adds zero census sites.

The module has its own failures ledger
(nomorals.os.resources_failures, from passes #123/#125/#127/
#128/#136); NOT-recording is pinned by snapshotting that
ledger's recent_failures() before/after each trigger.

Design context (resources_failures.py docstring + the #136
split): OSError stays silent BY DESIGN in these handlers —
/proc or sysfs absent on a host is not a failure; the contract
is "None when unavailable" — while malformed content on
present-but-broken input (ValueError/IndexError) records on the
sibling branch. Five of the eight sites are that silent OSError
branch of a split handler; three are fully-excluded sites named
by #123/#127/#136 (_cgroup_mount, _sample_network, sample).

Verdict notes (full grounds in the tail-48 SURVEY.md):
- 436: _cgroup_mount mountinfo OSError -> pass, then the
  /sys/fs/cgroup candidate check decides; None is the genuine
  no-mount value and both cgroup consumers map None to
  "no limit found".
- 571: _read_cpu_stat /proc/stat OSError -> None (no data);
  the delta consumer returns (None, None) on None.
- 591: _read_swap /proc/meminfo OSError -> (None, None);
  _sample_swap maps a falsy total to None (no swap data).
- 611: _read_swap_io /proc/vmstat OSError -> None; the rate
  consumer returns None on None.
- 628: _self_rss_mb /proc/self/status OSError -> None; the
  sample field is honestly None (no measurement).
- 1282: effective_memory_mb meminfo OSError -> host_mb stays
  None and the value is derived from the cgroup limit alone
  (or None = unknown); consumers apply their own `or`
  defaults, never a fabricated host figure.
- 1444: _sample_network sysfs OSError -> pass and fall
  through to a real second probe (socket connect) that
  decides online/offline on its own evidence.
- 1474: sample() detect_environment().to_dict() failure ->
  env None. detect_environment is documented never-raises and
  records its own failures internally, so only an exotic
  to_dict failure reaches this handler; the consult consumer
  renders None as kind "unknown" via (env or {}).
"""

import builtins
import io
import os
import socket

import pytest

import nomorals.os.resources as res
from nomorals.os import resources_failures as rf


@pytest.fixture(autouse=True)
def _clean_ledger():
    rf.clear_failures()
    yield
    rf.clear_failures()


def _snap():
    return list(rf.recent_failures(500))


_real_open = builtins.open


# ---- 436 _cgroup_mount ------------------------------------------------

def test_436_cgroup_mount_mountinfo_oserror_falls_back(monkeypatch):
    before = _snap()
    calls = []

    def raising_open(path, *a, **k):
        calls.append(str(path))
        raise OSError("mountinfo unreadable")

    monkeypatch.setattr(builtins, "open", raising_open)
    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    assert res._cgroup_mount() == "/sys/fs/cgroup"
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    assert res._cgroup_mount() is None
    assert "/proc/self/mountinfo" in calls
    # control: readable mountinfo yields the parsed target token
    monkeypatch.setattr(builtins, "open",
                        lambda path, *a, **k: io.StringIO("x - cgroup2 /cg opts\n"))
    assert res._cgroup_mount() == "/cg"
    assert _snap() == before


# ---- the four /proc readers -------------------------------------------

def _reader_pin(monkeypatch, path, fn, expected_fail, control_text,
                expected_ok):
    before = _snap()
    calls = []

    def fake_open(p, *a, **k):
        calls.append(str(p))
        if str(p) == path:
            raise OSError("no /proc on this host")
        return _real_open(p, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert fn() == expected_fail
    assert path in calls
    monkeypatch.setattr(builtins, "open",
                        lambda p, *a, **k: io.StringIO(control_text))
    assert fn() == expected_ok
    assert _snap() == before


def test_571_read_cpu_stat_oserror_is_none(monkeypatch):
    _reader_pin(monkeypatch, "/proc/stat", res._read_cpu_stat, None,
                "cpu  1 0 1 8 0 0 0 0 0 0\n", (10, 8, 0))


def test_591_read_swap_oserror_is_none_pair(monkeypatch):
    _reader_pin(monkeypatch, "/proc/meminfo", res._read_swap, (None, None),
                "SwapTotal:       1000 kB\nSwapFree:         250 kB\n",
                (1000, 250))


def test_611_read_swap_io_oserror_is_none(monkeypatch):
    _reader_pin(monkeypatch, "/proc/vmstat", res._read_swap_io, None,
                "pswpin 11\npswpout 22\n", (11, 22))


def test_628_self_rss_oserror_is_none(monkeypatch):
    _reader_pin(monkeypatch, "/proc/self/status", res._self_rss_mb, None,
                "VmRSS:       2048 kB\n", 2.0)


# ---- 1282 effective_memory_mb ------------------------------------------

def test_1282_effective_memory_meminfo_oserror_uses_cgroup(monkeypatch):
    mgr = res.ResourceManager()
    before = _snap()
    calls = []

    def fake_open(p, *a, **k):
        calls.append(str(p))
        if str(p) == "/proc/meminfo":
            raise OSError("meminfo unreadable")
        return _real_open(p, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(res, "cgroup_memory_limit_bytes",
                        lambda: 512 * 1024 * 1024)
    assert mgr.effective_memory_mb() == 512.0
    assert "/proc/meminfo" in calls
    # control: readable meminfo, no cgroup limit -> host value
    mgr2 = res.ResourceManager()
    monkeypatch.setattr(builtins, "open",
                        lambda p, *a, **k: io.StringIO(
                            "MemTotal:       1048576 kB\n"))
    monkeypatch.setattr(res, "cgroup_memory_limit_bytes", lambda: None)
    assert mgr2.effective_memory_mb() == 1024.0
    assert _snap() == before


# ---- 1444 _sample_network ----------------------------------------------

class _FakeSock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_1444_sample_network_sysfs_oserror_socket_probe_decides(monkeypatch):
    mgr = res.ResourceManager()
    before = _snap()
    ld_calls = []

    def raising_listdir(p, *a, **k):
        ld_calls.append(str(p))
        raise OSError("no sysfs")

    monkeypatch.setattr(os, "listdir", raising_listdir)
    monkeypatch.setattr(socket, "create_connection",
                        lambda addr, timeout=None: _FakeSock())
    assert mgr._sample_network() is True

    def dead_conn(addr, timeout=None):
        raise OSError("unreachable")

    monkeypatch.setattr(socket, "create_connection", dead_conn)
    assert mgr._sample_network() is False
    assert "/sys/class/net" in ld_calls
    # control: sysfs readable with an up iface -> True, no socket probe
    conn_calls = []
    monkeypatch.setattr(os, "listdir", lambda p: ["lo", "eth0"])
    monkeypatch.setattr(res, "_read_text",
                        lambda p: "up" if "eth0" in str(p) else None)
    monkeypatch.setattr(socket, "create_connection",
                        lambda addr, timeout=None: conn_calls.append(addr) or _FakeSock())
    assert mgr._sample_network() is True
    assert conn_calls == []
    assert _snap() == before


# ---- 1474 sample() environment ------------------------------------------

def test_1474_sample_env_to_dict_failure_is_none(monkeypatch):
    mgr = res.ResourceManager()
    mgr._sample_mem = lambda: (None, None, None)
    mgr._sample_disk = lambda: (None, None)
    mgr._sample_battery = lambda: {}
    mgr._sample_thermal = lambda: (None, None)
    mgr._sample_cpu_deltas = lambda: (None, None)
    mgr._sample_cpu = lambda: None
    mgr._sample_swap = lambda: None
    mgr._sample_network = lambda: False
    mgr.effective_memory_mb = lambda: None
    mgr.effective_cpu_count = lambda: None
    monkeypatch.setattr(res, "sample_psi", lambda: None)
    monkeypatch.setattr(res, "_self_rss_mb", lambda: None)
    de_calls = []

    class _BadEnv:
        def to_dict(self):
            de_calls.append(1)
            raise RuntimeError("to_dict boom")

    monkeypatch.setattr(res, "detect_environment", lambda: _BadEnv())
    before = _snap()
    s = mgr.sample()
    assert s.environment is None
    assert de_calls == [1]
    assert _snap() == before
    # control: healthy environment dict carried through
    good = res.EnvironmentInfo(kind="desktop", virtualized=False,
                               containerized=False, signals={}, detail="ok")
    monkeypatch.setattr(res, "detect_environment", lambda: good)
    s2 = mgr.sample()
    assert s2.environment == good.to_dict()
    assert _snap() == before
