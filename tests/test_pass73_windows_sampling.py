"""Pass #73 — immune process sampling is real on Windows, honest elsewhere.

Before this pass, ``collect_process_sample`` read the process count
from ``/proc`` and returned ``None`` everywhere else. The owner's
production host is Windows Server, so the immune sampler ticked
every minute, sampled nothing, learned nothing — and recorded
``{"ok": True, "sampled": False}`` (a success) while ``/immune
status`` showed the sampler "running". A decorative organ presented
as live.

Pinned here:

* the sample shape is a contract (``SAMPLE_KEYS``): exactly
  ``processes_spawned`` + ``process_growth``, whatever the platform;
* platform backends produce the count behind one seam — ``/proc`` on
  POSIX (unchanged), Win32 ``EnumProcesses`` (ctypes, stdlib) on
  Windows, optional psutil as the cross-platform fallback — faked at
  the seam here, since this build VM is Linux;
* the learner consumes a Windows-shaped sample identically to a
  Linux-shaped one (it cannot tell which OS a sample came from);
* where sampling genuinely cannot run, the organ KNOWS it:
  ``sampling_availability()`` reports it, the sampler tick returns a
  declared-failure (``ok: False``) unavailable result instead of an
  ok-success with "sampled: no" buried in it, the monitor announces
  the condition exactly once (no per-minute spam), and ``/immune
  status`` names sampling UNAVAILABLE in plain words instead of
  presenting a live sampler.
"""

from __future__ import annotations

import types

import pytest

from nomorals.immune.baseline import (
    SAMPLE_KEYS,
    AdaptiveBaseline,
    collect_process_sample,
    sampling_availability,
)
from nomorals.immune.baseline import _count_processes_psutil
from nomorals.immune.baseline import _count_processes_win32
from nomorals.immune.baseline import _process_count
from nomorals.immune.wiring import (
    ImmuneMonitor,
    reset_shared_monitors,
    summarize_tick_event,
)

NORMAL = {"processes_spawned": 100.0, "process_growth": 1.0}

UNAVAILABLE = {
    "available": False,
    "backend": "",
    "platform": "nt",
    "reason": "fake platform: no backend could count processes",
}


@pytest.fixture(autouse=True)
def _clean_monitors():
    reset_shared_monitors()
    yield
    reset_shared_monitors()


# ── the sample-shape contract ─────────────────────────────────────

def test_sample_shape_contract_is_pinned():
    assert tuple(SAMPLE_KEYS) == ("processes_spawned", "process_growth")
    sample = collect_process_sample()
    if sample is None:
        pytest.skip("no live sample on this platform")
    # a real (Linux) sample carries exactly the contracted keys
    assert tuple(sample.keys()) == tuple(SAMPLE_KEYS)
    assert all(isinstance(v, float) for v in sample.values())


def test_collect_with_injected_count_matches_contract():
    base = AdaptiveBaseline()
    sample = collect_process_sample(
        base, count_fn=lambda: (212, "win32", ""))
    assert sample is not None
    assert tuple(sample.keys()) == tuple(SAMPLE_KEYS)
    assert sample["processes_spawned"] == 212.0
    assert sample["process_growth"] == 1.0  # no history yet
    # growth derives from the baseline median, identically per OS
    base.record({"processes_spawned": 100.0, "process_growth": 1.0})
    sample = collect_process_sample(
        base, count_fn=lambda: (250, "win32", ""))
    assert sample["process_growth"] == pytest.approx(2.5)


def test_collect_returns_none_when_no_backend_can_count():
    assert collect_process_sample(
        count_fn=lambda: (None, "", "nothing worked")) is None


# ── the Windows backend, faked at the ctypes seam ─────────────────
#
# The backend uses a narrow ctypes surface: ``ct.windll.psapi`` with
# ``EnumProcesses(byref(buf), sizeof(buf), byref(returned))``,
# ``(ct.c_ulong * n)()`` buffers, ``ct.c_ulong`` scalars and
# ``ct.sizeof(ct.c_ulong())``. The fake implements exactly that.

class _FakePidArray:
    """Stands in for a ``(c_ulong * n)`` ctypes buffer."""

    def __init__(self, n):
        self._buf = [0] * n

    def __len__(self):
        return len(self._buf)

    def __getitem__(self, item):
        return self._buf[item]

    def __setitem__(self, idx, value):
        self._buf[idx] = value


class _FakePsapi:
    def __init__(self, pids, fail=False):
        self._pids = pids
        self._fail = fail
        self.calls = 0

    def EnumProcesses(self, buf, _cb, returned):
        self.calls += 1
        if self._fail:
            return 0
        capacity = len(buf)
        for i, pid in enumerate(self._pids[:capacity]):
            buf[i] = pid
        # cbReturned is a BYTE count, like the real API
        returned.value = min(len(self._pids), capacity) * 4
        return 1


class _FakeDwordMeta(type):
    """Array creation hangs off the metaclass — as in real ctypes,
    where ``c_ulong * n`` is resolved on the type, not instances."""

    def __mul__(cls, n):
        return lambda: _FakePidArray(n)


class _FakeDword(metaclass=_FakeDwordMeta):
    """``c_ulong`` stand-in (scalar; ``* n`` builds buffer arrays)."""

    def __init__(self, value=0):
        self.value = value


def _fake_ctypes(pids, fail=False):
    return types.SimpleNamespace(
        c_ulong=_FakeDword,
        byref=lambda obj: obj,
        sizeof=lambda obj: len(obj) * 4
        if isinstance(obj, _FakePidArray) else 4,
        windll=types.SimpleNamespace(psapi=_FakePsapi(pids, fail=fail)),
    )


def test_win32_backend_counts_processes_and_drops_pid_zero():
    fake = _fake_ctypes([0, 4, 312, 4096])
    assert _count_processes_win32(fake) == 3  # pid 0 (idle) excluded


def test_win32_backend_grows_buffer_until_it_fits():
    pids = [0, 4] + list(range(100, 100 + 3000))  # 3002 processes
    fake = _fake_ctypes(pids)
    assert _count_processes_win32(fake) == 3001  # all but pid 0
    assert fake.windll.psapi.calls >= 2  # the buffer had to grow


def test_win32_backend_failure_is_none_not_fabricated():
    fake = _fake_ctypes([4, 8], fail=True)
    assert _count_processes_win32(fake) is None


def test_win32_backend_without_windll_is_none():
    fake = types.SimpleNamespace()  # no windll at all (non-Windows)
    assert _count_processes_win32(fake) is None


# ── the psutil fallback backend, faked ────────────────────────────

def test_psutil_backend_counts_pids():
    fake = types.SimpleNamespace(pids=lambda: [1, 2, 3, 4, 5])
    assert _count_processes_psutil(fake) == 5


def test_psutil_backend_failure_is_none():
    def _boom():
        raise RuntimeError("psutil exploded")

    fake = types.SimpleNamespace(pids=_boom)
    assert _count_processes_psutil(fake) is None


# ── the dispatcher: one seam, platform order, honest reasons ─────

def test_dispatcher_prefers_win32_on_nt():
    count, backend, _reason = _process_count(
        os_name="nt",
        win32_count_fn=lambda: 137,
        proc_count_fn=lambda: 999,
        psutil_count_fn=lambda: 55)
    assert (count, backend) == (137, "win32")


def test_dispatcher_prefers_proc_on_posix_unchanged():
    count, backend, _reason = _process_count(
        os_name="posix",
        proc_count_fn=lambda: 88,
        win32_count_fn=lambda: 137,
        psutil_count_fn=lambda: 55)
    assert (count, backend) == (88, "proc")


def test_dispatcher_falls_back_to_psutil():
    count, backend, _reason = _process_count(
        os_name="posix",
        proc_count_fn=lambda: None,
        psutil_count_fn=lambda: 55)
    assert (count, backend) == (55, "psutil")


def test_dispatcher_all_backends_down_reports_reasons():
    count, backend, reason = _process_count(
        os_name="posix",
        proc_count_fn=lambda: None,
        psutil_count_fn=lambda: None)
    assert count is None and backend == ""
    assert "proc" in reason and "psutil" in reason


def test_availability_reports_backend_and_platform():
    avail = sampling_availability(count_fn=lambda: (42, "win32", ""))
    assert avail["available"] is True
    assert avail["backend"] == "win32"
    assert avail["platform"]  # the host os name, recorded
    down = sampling_availability(count_fn=lambda: (None, "", "all down"))
    assert down["available"] is False
    assert down["backend"] == ""
    assert "all down" in down["reason"]


# ── the learner cannot tell which OS a sample came from ──────────

def test_learner_consumes_windows_samples_identically():
    linux_base = AdaptiveBaseline()
    win_base = AdaptiveBaseline()
    for i in range(40):
        linux_base.record({"processes_spawned": 100.0 + i,
                           "process_growth": 1.0})
        win = collect_process_sample(
            count_fn=lambda n=100.0 + i: (int(n), "win32", ""))
        win_base.record(win)
    assert win_base.samples() == linux_base.samples()
    lt, wt = linux_base.thresholds(), win_base.thresholds()
    assert wt.status == lt.status == "ready"
    assert wt.spawned_p95 == lt.spawned_p95
    assert wt.growth_p95 == lt.growth_p95


def test_monitor_learns_from_windows_shaped_collect(tmp_path):
    mon = ImmuneMonitor(
        tmp_path / "b.json",
        collect=lambda baseline: collect_process_sample(
            baseline, count_fn=lambda: (150, "win32", "")))
    for _ in range(35):
        out = mon.tick()
        assert out["learned"] is True
    assert mon.baseline.thresholds().status == "ready"
    assert mon.status()["sampling_backend"]


# ── honest unavailability: tick, announce-once, status ───────────

def _unavailable_monitor(tmp_path):
    return ImmuneMonitor(
        tmp_path / "b.json",
        collect=lambda baseline: None,
        availability=lambda: dict(UNAVAILABLE))


def test_unavailable_tick_is_a_declared_failure_not_ok_success(tmp_path):
    mon = _unavailable_monitor(tmp_path)
    out = mon.tick()
    assert out["ok"] is False
    assert out["unavailable"] is True
    assert out["sampled"] is False
    assert out["learned"] is False
    assert "unavailable" in out["error"]
    assert "no backend could count" in out["error"]
    assert len(mon.baseline) == 0
    # the tick still happened (staleness can tell "ticking but
    # cannot sample" from "never ran")
    assert mon.status()["last_tick_at"] is not None
    # under the Pass #71 contract the scheduler records this FAILED
    from nomorals.agents.scheduler import _declared_failure

    assert _declared_failure(out)


def test_unavailable_announced_once_never_spam(tmp_path):
    mon = _unavailable_monitor(tmp_path)
    first = mon.tick()
    event = summarize_tick_event(first, mon.drain_announcements())
    assert event is not None
    assert "unavailable" in (event["title"] + event["body"]).lower()
    # every later tick: no event, no announcement — the condition is
    # standing, and the failed run record is the evidence
    for _ in range(3):
        out = mon.tick()
        assert summarize_tick_event(
            out, mon.drain_announcements()) is None


def test_available_platform_no_sample_keeps_legacy_contract(tmp_path):
    # An injected collector returning None on a platform that CAN
    # sample (the Pass 3 world) keeps the ok no-sample report.
    mon = ImmuneMonitor(
        tmp_path / "b.json",
        collect=lambda baseline: None,
        availability=lambda: {"available": True, "backend": "proc",
                              "platform": "posix", "reason": ""})
    out = mon.tick()
    assert out["ok"] is True
    assert out["sampled"] is False
    assert "no live process sample" in out["reason"]
    assert summarize_tick_event(out, mon.drain_announcements()) is None


def test_status_names_unavailable_sampling(tmp_path):
    from nomorals.immune.chat import _gate_status_lines

    mon = _unavailable_monitor(tmp_path)
    mon.tick()
    st = mon.status()
    assert st["sampling_available"] is False
    assert "no backend could count" in st["sampling_reason"]
    text = "\n".join(_gate_status_lines(mon))
    assert "UNAVAILABLE" in text
    assert "cannot learn" in text or "never been collected" in text


def test_status_on_sampling_platform_never_claims_unavailable(tmp_path):
    from nomorals.immune.chat import _gate_status_lines

    mon = ImmuneMonitor(tmp_path / "b.json")
    text = "\n".join(_gate_status_lines(mon))
    assert "UNAVAILABLE" not in text
    st = mon.status()
    if collect_process_sample() is not None:
        assert st["sampling_available"] is True
        assert st["sampling_backend"] == "proc"


def test_gate_surface_names_unavailable_sampling(tmp_path):
    from nomorals.immune.chat import _ImmuneState, _gate

    mon = _unavailable_monitor(tmp_path)
    out = _gate(_ImmuneState(), mon, "")
    assert "unavailable" in out.lower()
    assert "no backend could count" in out
