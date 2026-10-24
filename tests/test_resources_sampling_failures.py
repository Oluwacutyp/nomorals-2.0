"""Pins for Pass #127: sampling-chain failure ledger in
nomorals/os/resources.py, reusing the #123 resources_failures ledger.

Scope: the ResourceManager sampling sub-cluster — _is_battery_powered,
effective_memory_mb, effective_cpu_count, _sample_cpu_deltas, _sample_swap,
_sample_swap_io_rate, _sample_thermal, sample.safe, sample.alerts
(9 sites). Shared failure story: a sample that silently lies. A failed
platform detection silently runs desktop thresholds on a phone; a failed
effective-memory/CPU computation silently reports "unknown" where the
caller substitutes a host value; a failed swap read silently reports
"no swap configured"; failed delta/thermal samplers silently report
"no data yet"; a sampler bug inside sample() silently substitutes the
unknown fallback; a failing alert dispatch silently drops the alert.
The caller sees a healthy-looking sample throughout.

Deliberate exclusions in this region (all pinned as behaviour):
- effective_memory_mb's inner /proc/meminfo pass: designed-absence probe
  (no /proc on the Windows host); the code falls through to the cgroup
  limit, so a value is still produced — same class as #123's /proc
  reader exclusions.
- _sample_network's sysfs OSError pass: probe fall-through; the socket
  probe still happens and produces the value.
- sample()'s environment fallback: detect_environment's own guard (#123)
  already records the failure at source; sample() must not double-record.
"""

import ast
import os
from types import SimpleNamespace

import pytest

import nomorals.os.resources as res
from nomorals.os.resources_failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(500)]


def _bare_manager(**attrs):
    m = res.ResourceManager.__new__(res.ResourceManager)
    for key, value in attrs.items():
        setattr(m, key, value)
    return m


# ---------------------------------------------------------------------------
# ledger basics (reused #123 ledger, not a new one)
# ---------------------------------------------------------------------------

def test_ledger_records_and_renders():
    assert recent_failures() == []
    assert failure_count() == 0
    assert format_failures() == "no recorded resource failures"
    record_failure("test.site", ValueError("boom"))
    assert failure_count() == 1
    [entry] = recent_failures()
    assert entry.site == "test.site"
    assert entry.exc_type == "ValueError"
    assert "boom" in format_failures()


def test_ledger_bound_and_unprintable():
    for _ in range(250):
        record_failure("test.site", ValueError("x"))
    assert failure_count() == 250
    assert len(recent_failures(500)) == 200

    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    record_failure("test.bad", BadStr())
    assert failure_count() == 251
    assert "unprintable" in format_failures()


def test_direct_import_surface_and_all_unchanged():
    from nomorals.os.resources_failures import format_failures as ff

    assert ff() == "no recorded resource failures"
    assert res.__all__ == [
        "ResourceSample", "ResourceManager", "ResourceBudgets",
        "SubsystemBudget", "DegradationLadder", "DegradationStep",
        "DEGRADATION_LEVELS", "EnvironmentInfo", "PressureStall",
        "default_manager", "reset_default_manager", "advisor_callable",
        "detect_environment", "reset_environment_cache", "sample_psi",
        "cgroup_memory_limit_bytes", "cgroup_cpu_quota_cores",
    ]


# ---------------------------------------------------------------------------
# converted sites — real failures record, benign paths do not
# ---------------------------------------------------------------------------

def test_is_battery_powered_records_on_detection_failure(monkeypatch):
    m = _bare_manager()
    monkeypatch.setattr(res.ResourceManager, "_platform_obj",
                        lambda self: (_ for _ in ()).throw(RuntimeError("plat")))
    assert m._is_battery_powered() is False
    assert "ResourceManager._is_battery_powered" in _sites()


def test_is_battery_powered_no_record_on_healthy_platform():
    m = _bare_manager(_platform=SimpleNamespace(name="linux",
                                                capabilities=None))
    assert m._is_battery_powered() is False
    assert _sites() == []


def test_effective_memory_mb_records_on_cgroup_failure(monkeypatch):
    m = _bare_manager(_eff_cache={})
    monkeypatch.setattr(res, "cgroup_memory_limit_bytes",
                        lambda: (_ for _ in ()).throw(RuntimeError("cgroup")))
    assert m.effective_memory_mb() is None
    assert "ResourceManager.effective_memory_mb" in _sites()


def test_effective_memory_mb_no_record_on_healthy_read():
    m = _bare_manager(_eff_cache={})
    assert isinstance(m.effective_memory_mb(), float)
    assert _sites() == []


def test_effective_cpu_count_records_on_quota_failure(monkeypatch):
    m = _bare_manager(_eff_cache={})
    monkeypatch.setattr(res, "cgroup_cpu_quota_cores",
                        lambda: (_ for _ in ()).throw(RuntimeError("quota")))
    assert m.effective_cpu_count() is None
    assert "ResourceManager.effective_cpu_count" in _sites()


def test_effective_cpu_count_no_record_on_healthy_read():
    m = _bare_manager(_eff_cache={})
    assert isinstance(m.effective_cpu_count(), float)
    assert _sites() == []


def test_sample_cpu_deltas_records_on_stat_failure(monkeypatch):
    m = _bare_manager(_last_cpu_stat=None)
    monkeypatch.setattr(res, "_read_cpu_stat",
                        lambda: (_ for _ in ()).throw(RuntimeError("stat")))
    assert m._sample_cpu_deltas() == (None, None)
    assert "ResourceManager._sample_cpu_deltas" in _sites()


def test_sample_cpu_deltas_no_record_on_designed_absence(monkeypatch):
    m = _bare_manager(_last_cpu_stat=None)
    monkeypatch.setattr(res, "_read_cpu_stat", lambda: None)
    assert m._sample_cpu_deltas() == (None, None)
    assert _sites() == []


def test_sample_swap_records_on_read_failure(monkeypatch):
    m = _bare_manager()
    monkeypatch.setattr(res, "_read_swap",
                        lambda: (_ for _ in ()).throw(RuntimeError("swap")))
    assert m._sample_swap() is None
    assert "ResourceManager._sample_swap" in _sites()


def test_sample_swap_no_record_on_no_swap_configured(monkeypatch):
    m = _bare_manager()
    monkeypatch.setattr(res, "_read_swap", lambda: (0, 0))
    assert m._sample_swap() is None
    assert _sites() == []


def test_sample_swap_io_rate_records_on_read_failure(monkeypatch):
    m = _bare_manager(_last_swap_io=None)
    monkeypatch.setattr(res, "_read_swap_io",
                        lambda: (_ for _ in ()).throw(RuntimeError("vmstat")))
    assert m._sample_swap_io_rate() is None
    assert "ResourceManager._sample_swap_io_rate" in _sites()


def test_sample_swap_io_rate_no_record_on_first_call(monkeypatch):
    m = _bare_manager(_last_swap_io=None)
    monkeypatch.setattr(res, "_read_swap_io", lambda: None)
    assert m._sample_swap_io_rate() is None
    assert _sites() == []


def test_sample_thermal_records_on_probe_failure(monkeypatch):
    m = _bare_manager(thermal_warm_c=55.0, thermal_hot_c=70.0)
    monkeypatch.setattr(res, "_thermal_zone_temps",
                        lambda: (_ for _ in ()).throw(RuntimeError("tz")))
    assert m._sample_thermal() == (None, None)
    assert "ResourceManager._sample_thermal" in _sites()


def test_sample_thermal_no_record_on_empty_zones(monkeypatch):
    m = _bare_manager(thermal_warm_c=55.0, thermal_hot_c=70.0)
    monkeypatch.setattr(res, "_thermal_zone_temps", lambda: [])
    assert m._sample_thermal() == (None, None)
    assert _sites() == []


def test_sample_safe_names_the_failing_sampler(monkeypatch):
    # The site names the failing sampler by __name__; the patch must bear
    # the sampler's name (a lambda would record as "<lambda>" — the
    # dynamic naming working as designed, not a conversion bug).
    def _sample_mem(self):
        raise RuntimeError("mem")

    m = res.ResourceManager()
    monkeypatch.setattr(res.ResourceManager, "_sample_mem", _sample_mem)
    s = m.sample()
    assert s.mem_percent is None  # fallback still applied, sample returned
    assert "ResourceManager.sample.safe._sample_mem" in _sites()


def test_sample_safe_no_record_when_samplers_healthy():
    m = res.ResourceManager()
    m.sample()
    assert [s for s in _sites()
            if s.startswith("ResourceManager.sample.safe.")] == []


def test_sample_alerts_records_on_dispatch_failure(monkeypatch):
    m = res.ResourceManager()
    m.on_alert("cpu", 999.0, lambda info: None)  # subscription present
    monkeypatch.setattr(res.ResourceManager, "pressure",
                        lambda self, s: (_ for _ in ()).throw(
                            RuntimeError("pressure")))
    s = m.sample()  # alerts never break sampling: sample still returned
    assert s is not None
    assert "ResourceManager.sample.alerts" in _sites()


def test_sample_alerts_no_record_without_subscriptions():
    m = res.ResourceManager()
    m.sample()
    assert "ResourceManager.sample.alerts" not in _sites()


# ---------------------------------------------------------------------------
# deliberate exclusions — pinned as behaviour, not comments
# ---------------------------------------------------------------------------

def test_meminfo_absence_falls_through_to_cgroup_without_record(monkeypatch):
    import builtins
    orig_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path).startswith("/proc/meminfo"):
            raise OSError("no meminfo here")
        return orig_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    m = _bare_manager(_eff_cache={})
    m.effective_memory_mb()  # cgroup value or None — either way, no record
    assert "ResourceManager.effective_memory_mb" not in _sites()


def test_network_sysfs_failure_falls_through_to_socket_probe(monkeypatch):
    class DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    m = _bare_manager()
    monkeypatch.setattr(os, "listdir",
                        lambda path: (_ for _ in ()).throw(OSError("no sysfs")))
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: DummyConn())
    assert m._sample_network() is True  # the probe still happened
    assert _sites() == []


def test_environment_failure_recorded_at_source_not_in_sample(monkeypatch):
    # Drive the REAL detect_environment guard (#123): patch only its inner
    # probe to raise, reset the cache so sample() re-detects, and show the
    # failure is recorded once at source while sample() takes the fallback
    # dict without adding a sample-level record.
    monkeypatch.setattr(res, "_detect_environment",
                        lambda: (_ for _ in ()).throw(RuntimeError("env")))
    res.reset_environment_cache()
    try:
        m = res.ResourceManager()
        s = m.sample()
        assert s.environment == {
            "kind": "unknown", "virtualized": False, "containerized": False,
            "signals": {}, "detail": "detection failed",
        }
        assert "detect_environment" in _sites()
        assert "ResourceManager.sample" not in _sites()
    finally:
        res.reset_environment_cache()


# ---------------------------------------------------------------------------
# scan pin — suppression-off remainder across resources.py
# ---------------------------------------------------------------------------

def _enclosing_function(path, line):
    tree = ast.parse(open(path).read())
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= line <= node.end_lineno:
                if best is None or (node.end_lineno - node.lineno) < (
                        best.end_lineno - best.lineno):
                    best = node
    return best.name if best else None


def test_scan_remainder_is_19_probe_sampling_analysis():
    from nomorals.tools import error_scan as es

    original = es._noqa_suppresses
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([os.path.join(os.path.dirname(res.__file__),
                                       "resources.py")])
        # 28 (#125) - 9 converted = 19: 5 excluded probe + 3 excluded
        # sampling (meminfo pass, sysfs pass, env fallback) + 11 analysis.
        assert len(report.findings) == 19
        from collections import Counter
        fns = Counter(_enclosing_function(f.file, f.line)
                      for f in report.findings)
        assert fns == Counter({
            # excluded probe region (#123)
            "_cgroup_mount": 1, "_read_cpu_stat": 1, "_read_swap": 1,
            "_read_swap_io": 1, "_self_rss_mb": 1,
            # excluded sampling sites (this pass)
            "effective_memory_mb": 1, "_sample_network": 1, "sample": 1,
            # analysis region (later pass)
            "_record": 1, "history": 1, "trend": 1, "forecast": 1,
            "battery_forecast": 1, "composite_pressure": 1, "pressure": 1,
            "_degradation_target": 1, "_eval_alerts": 1, "render": 1,
            "summary_line": 1,
        })
    finally:
        es._noqa_suppresses = original


def test_scan_ledger_file_clean():
    from nomorals.tools import error_scan as es

    original = es._noqa_suppresses
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([os.path.join(os.path.dirname(res.__file__),
                                       "resources_failures.py")])
        assert report.findings == []
    finally:
        es._noqa_suppresses = original
