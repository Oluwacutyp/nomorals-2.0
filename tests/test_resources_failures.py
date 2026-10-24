"""Pins for Pass #123: host-probe failure ledger in nomorals/os/resources.py.

Scope: the host-probe sub-cluster (lines 236-719 of resources.py) — the
file's foundation region, whose results every downstream consumer treats
as fact. Converted: 8 sites. Deliberate exclusions, pinned as behaviour:
_cgroup_mount's mountinfo OSError (designed fallback to the isdir check)
and the four /proc readers (OSError = designed absence on non-Linux
hosts; the malformed-content corner is inseparable in the tuple handler
without restructuring and is named as a remainder, not buried).
"""

import ast
import os
import platform
import sys

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


# ---------------------------------------------------------------------------
# ledger basics
# ---------------------------------------------------------------------------

def test_ledger_record_and_render():
    record_failure("probe.site", ValueError("bad value"))
    assert failure_count() == 1
    entries = recent_failures()
    assert len(entries) == 1
    assert entries[0].site == "probe.site"
    assert entries[0].exc_type == "ValueError"
    assert entries[0].message == "bad value"
    assert isinstance(entries[0].ts, float)
    rendered = format_failures()
    assert "probe.site: ValueError: bad value" in rendered


def test_ledger_empty_state():
    assert recent_failures() == []
    assert failure_count() == 0
    assert format_failures() == "no recorded resource failures"


def test_ledger_bound_keeps_newest():
    for i in range(250):
        record_failure(f"site.{i}", RuntimeError(f"err {i}"))
    assert failure_count() == 250
    kept = recent_failures(500)
    assert len(kept) == 200
    assert kept[0].site == "site.249"
    assert kept[-1].site == "site.50"


def test_ledger_unprintable_exc_safe():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    record_failure("probe.badstr", BadStr())
    entry = recent_failures()[0]
    assert entry.exc_type == "BadStr"
    assert "unprintable" in entry.message


def test_direct_import_surface():
    import nomorals.os.resources_failures as lf

    for name in ("record_failure", "recent_failures", "failure_count",
                 "clear_failures", "format_failures"):
        assert callable(getattr(lf, name))


def test_resources_all_unchanged():
    expected = [
        "ResourceSample", "ResourceManager", "ResourceBudgets",
        "SubsystemBudget", "DegradationLadder", "DegradationStep",
        "DEGRADATION_LEVELS", "EnvironmentInfo", "PressureStall",
        "default_manager", "reset_default_manager", "advisor_callable",
        "detect_environment", "reset_environment_cache", "sample_psi",
        "cgroup_memory_limit_bytes", "cgroup_cpu_quota_cores",
    ]
    assert list(res.__all__) == expected


# ---------------------------------------------------------------------------
# converted sites — real drives
# ---------------------------------------------------------------------------

def test_detect_environment_platform_probe(monkeypatch):
    monkeypatch.setenv("PREFIX", "")
    monkeypatch.setattr(platform, "system", lambda: (_ for _ in ()).throw(
        RuntimeError("platform exploded")))
    info = res._detect_environment()
    # The probe's only assignment is kind="android"; with system() raising,
    # kind stays whatever the other markers decided (on this VM the cpuinfo
    # hypervisor flag yields "other-cloud"). The pin is the record, not kind.
    assert info.kind != "android"
    assert "_detect_environment.platform" in _sites()


def test_detect_environment_platform_control(monkeypatch):
    monkeypatch.setenv("PREFIX", "")
    res._detect_environment()
    assert _sites() == []


def test_detect_environment_outer_guard(monkeypatch):
    res.reset_environment_cache()
    try:
        monkeypatch.setattr(res, "_detect_environment", lambda: (_ for _ in ()).throw(
            RuntimeError("detection exploded")))
        info = res.detect_environment()
        assert info.kind == "unknown"
        assert info.detail == "detection failed"
        assert "detect_environment" in _sites()
    finally:
        res.reset_environment_cache()


def test_parse_psi_line_skips_bad_token():
    out = res._parse_psi_line("some avg10=abc avg60=1.0 avg300=2.5")
    assert out == {"avg60": 1.0, "avg300": 2.5}
    assert "_parse_psi_line" in _sites()


def test_parse_psi_line_clean_control():
    out = res._parse_psi_line("some avg10=1.0 avg60=2.0")
    assert out == {"avg10": 1.0, "avg60": 2.0}
    assert _sites() == []


def test_sample_psi_residual_failure(monkeypatch):
    monkeypatch.setattr(res, "_read_text", lambda path: (_ for _ in ()).throw(
        RuntimeError("read exploded")))
    assert res.sample_psi() is None
    assert "sample_psi" in _sites()


def test_sample_psi_absence_is_not_recorded():
    # Absent /proc/pressure is handled by the `continue` path, not the
    # except: no record whatever the return value is on this host.
    res.sample_psi()
    assert "sample_psi" not in _sites()


def test_cgroup_memory_limit_residual_failure(monkeypatch):
    monkeypatch.setattr(res, "_cgroup_mount", lambda: (_ for _ in ()).throw(
        RuntimeError("mount exploded")))
    assert res.cgroup_memory_limit_bytes() is None
    assert "cgroup_memory_limit_bytes" in _sites()


def test_cgroup_memory_limit_control():
    res.cgroup_memory_limit_bytes()
    assert "cgroup_memory_limit_bytes" not in _sites()


def test_cgroup_cpu_quota_malformed_max(monkeypatch):
    monkeypatch.setattr(res, "_cgroup_mount", lambda: "/fake")
    monkeypatch.setattr(res, "_self_cgroup_relpath", lambda: "/")
    monkeypatch.setattr(res, "_walk_cgroup_chain", lambda m, r: ["/fake"])
    monkeypatch.setattr(res, "_read_text",
                        lambda p: "abc def" if p.endswith("cpu.max") else None)
    assert res.cgroup_cpu_quota_cores() is None
    assert "cgroup_cpu_quota_cores.cpu_max" in _sites()
    assert "cgroup_cpu_quota_cores" not in _sites()


def test_cgroup_cpu_quota_outer_failure(monkeypatch):
    monkeypatch.setattr(res, "_cgroup_mount", lambda: "/fake")
    monkeypatch.setattr(res, "_self_cgroup_relpath", lambda: "/")
    monkeypatch.setattr(res, "_walk_cgroup_chain", lambda m, r: (_ for _ in ()).throw(
        RuntimeError("walk exploded")))
    assert res.cgroup_cpu_quota_cores() is None
    assert "cgroup_cpu_quota_cores" in _sites()
    assert "cgroup_cpu_quota_cores.cpu_max" not in _sites()


def test_battery_detail_residual_failure(monkeypatch):
    monkeypatch.setattr(res, "_battery_capacity_paths", lambda: (_ for _ in ()).throw(
        RuntimeError("battery exploded")))
    detail = res._battery_detail()
    assert detail == {"capacity": None, "status": None, "power_w": None,
                      "energy_now_wh": None, "energy_full_wh": None}
    assert "_battery_detail" in _sites()


def test_battery_detail_no_battery_control():
    res._battery_detail()
    assert "_battery_detail" not in _sites()


# ---------------------------------------------------------------------------
# deliberate exclusions — pinned as behaviour, not comments
# ---------------------------------------------------------------------------

def test_cgroup_mount_falls_back_when_mountinfo_unreadable(monkeypatch):
    def _raise_open(*args, **kwargs):
        raise OSError("mountinfo unreadable")

    with monkeypatch.context() as mp:
        mp.setattr("builtins.open", _raise_open)
        mp.setattr(os.path, "isdir", lambda p: True)
        assert res._cgroup_mount() == "/sys/fs/cgroup"
    assert _sites() == []


def test_cgroup_mount_absent_is_none_silently(monkeypatch):
    def _raise_open(*args, **kwargs):
        raise OSError("mountinfo unreadable")

    with monkeypatch.context() as mp:
        mp.setattr("builtins.open", _raise_open)
        mp.setattr(os.path, "isdir", lambda p: False)
        assert res._cgroup_mount() is None
    assert _sites() == []


def test_proc_readers_absence_is_designed_silence(monkeypatch):
    def _raise_open(*args, **kwargs):
        raise OSError("no /proc on this host")

    with monkeypatch.context() as mp:
        mp.setattr("builtins.open", _raise_open)
        assert res._read_cpu_stat() is None
        assert res._read_swap() == (None, None)
        assert res._read_swap_io() is None
        assert res._self_rss_mb() is None
    assert _sites() == []


# ---------------------------------------------------------------------------
# scan pin — suppression-off remainder, by enclosing function
# ---------------------------------------------------------------------------

PROBE_FUNCTIONS = {
    "_detect_environment", "detect_environment", "_parse_psi_line",
    "sample_psi", "_cgroup_mount", "cgroup_memory_limit_bytes",
    "cgroup_cpu_quota_cores", "_read_cpu_stat", "_read_swap",
    "_read_swap_io", "_self_rss_mb", "_battery_detail",
}
CONVERTED_FUNCTIONS = {
    "_detect_environment", "detect_environment", "_parse_psi_line",
    "sample_psi", "cgroup_memory_limit_bytes", "cgroup_cpu_quota_cores",
    "_battery_detail",
}
EXCLUDED_FUNCTIONS = {
    "_cgroup_mount", "_read_cpu_stat", "_read_swap", "_read_swap_io",
    "_self_rss_mb",
}


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


def test_scan_remainder_is_exactly_the_excluded_probe_sites():
    from nomorals.tools import error_scan as es

    original = es._noqa_suppresses
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([os.path.join(os.path.dirname(res.__file__),
                                       "resources.py")])
        # 44 true - 8 probe (#123) - 8 budgets (#125) - 9 sampling (#127)
        assert len(report.findings) == 19
        probe = {}
        for f in report.findings:
            fn = _enclosing_function(f.file, f.line)
            if fn in PROBE_FUNCTIONS:
                probe.setdefault(fn, 0)
                probe[fn] += 1
        assert set(probe) == EXCLUDED_FUNCTIONS
        assert not (set(probe) & CONVERTED_FUNCTIONS)
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
