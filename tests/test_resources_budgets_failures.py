"""Pins for Pass #125: budget-accounting failure ledger in
nomorals/os/resources.py, reusing the #123 resources_failures ledger.

Scope: the budgets sub-cluster of resources.py — DegradationLadder.update
plus ResourceBudgets.define/ensure_defaults/reserve_mb/
machine_headroom_mb/acquire/release/snapshot (8 sites). Shared failure
story: budget accounting that silently lies. A define that fails returns
an untracked budget; a sizing that fails leaves admission control
unguarded (and _sized already True, so it never retries); a headroom
failure silently drops the headroom constraint from admission; a release
that fails silently leaks tracked usage; a snapshot that fails reports
an empty budget table. The caller sees healthy-looking answers throughout.

Deliberate exclusions in this region: none. The honest neighbour,
ResourceBudgets.admission, binds ``exc`` into its reason string and is
outside the estate.
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


def test_ledger_bound_keeps_newest_200():
    for _ in range(250):
        record_failure("test.site", ValueError("x"))
    assert failure_count() == 250
    assert len(recent_failures(500)) == 200


def test_ledger_unprintable_exc_safe():
    class _BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    record_failure("test.site", _BadStr())
    [entry] = recent_failures()
    assert "unprintable" in entry.message


def test_direct_import_surface():
    import nomorals.os.resources_failures as lf

    for name in ("record_failure", "recent_failures", "failure_count",
                 "clear_failures", "format_failures"):
        assert callable(getattr(lf, name))


def test_resources_all_unchanged():
    assert res.__all__ == [
        "ResourceSample",
        "ResourceManager",
        "ResourceBudgets",
        "SubsystemBudget",
        "DegradationLadder",
        "DegradationStep",
        "DEGRADATION_LEVELS",
        "EnvironmentInfo",
        "PressureStall",
        "default_manager",
        "reset_default_manager",
        "advisor_callable",
        "detect_environment",
        "reset_environment_cache",
        "sample_psi",
        "cgroup_memory_limit_bytes",
        "cgroup_cpu_quota_cores",
    ]


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------

def _stub_manager(mem=8192.0, cpu=8.0, sample=None):
    return SimpleNamespace(
        effective_memory_mb=lambda: mem,
        effective_cpu_count=lambda: cpu,
        sample=(lambda: sample) if sample is not None else (lambda: (_ for _ in ()).throw(RuntimeError("no sample"))),
    )


def _sample(eff_mem=8192.0, avail=8192.0):
    return SimpleNamespace(effective_mem_mb=eff_mem, mem_available_mb=avail)


# ---------------------------------------------------------------------------
# DegradationLadder.update
# ---------------------------------------------------------------------------

class _BadInt:
    def __int__(self):
        raise RuntimeError("target unparseable")


def test_ladder_update_failure_holds_level_and_records():
    ladder = res.DegradationLadder()
    assert ladder.update(_BadInt(), now=1000.0) == 0
    assert _sites() == ["DegradationLadder.update"]


def test_ladder_update_success_records_nothing():
    ladder = res.DegradationLadder()
    assert ladder.update(2, critical=True, now=1000.0) == 2
    assert _sites() == []


# ---------------------------------------------------------------------------
# ResourceBudgets.define
# ---------------------------------------------------------------------------

class _RaisingDict(dict):
    def __setitem__(self, key, value):
        raise RuntimeError("store down")


def test_define_failure_returns_untracked_budget_and_records():
    budgets = res.ResourceBudgets(_stub_manager())
    budgets._budgets = _RaisingDict()
    got = budgets.define("x", max_mem_mb=100.0)
    assert got.name == "x"
    assert "x" not in budgets._budgets  # looks defined, is not tracked
    assert _sites() == ["ResourceBudgets.define"]


def test_define_success_stores_and_records_nothing():
    budgets = res.ResourceBudgets(_stub_manager())
    got = budgets.define("x", max_mem_mb=100.0)
    assert budgets._budgets["x"] is got
    assert _sites() == []


# ---------------------------------------------------------------------------
# ResourceBudgets.ensure_defaults
# ---------------------------------------------------------------------------

def test_ensure_defaults_failure_leaves_unguarded_and_records():
    def _raise_mem():
        raise RuntimeError("mem unreadable")

    budgets = res.ResourceBudgets(SimpleNamespace(
        effective_memory_mb=_raise_mem, effective_cpu_count=lambda: 8.0))
    budgets.ensure_defaults()
    assert budgets._budgets == {}
    assert budgets._sized is True  # never retries: admission stays open
    assert _sites() == ["ResourceBudgets.ensure_defaults"]


def test_ensure_defaults_success_populates_and_records_nothing():
    budgets = res.ResourceBudgets(_stub_manager())
    budgets.ensure_defaults()
    assert len(budgets._budgets) == len(res.ResourceBudgets._DEFAULTS)
    assert _sites() == []


# ---------------------------------------------------------------------------
# ResourceBudgets.reserve_mb
# ---------------------------------------------------------------------------

class _FlakyMem:
    """First memory read works (sizing), the next one dies."""

    def __init__(self):
        self.calls = 0

    def effective_memory_mb(self):
        self.calls += 1
        if self.calls == 1:
            return 8192.0
        raise RuntimeError("mem read died")

    def effective_cpu_count(self):
        return 8.0


def test_reserve_mb_failure_returns_floor_and_records():
    budgets = res.ResourceBudgets(_FlakyMem())
    assert budgets.reserve_mb() == budgets.reserve_min_mb
    assert _sites() == ["ResourceBudgets.reserve_mb"]


def test_reserve_mb_absent_memory_returns_floor_without_record():
    budgets = res.ResourceBudgets(_stub_manager(mem=None))
    assert budgets.reserve_mb() == budgets.reserve_min_mb
    assert _sites() == []


# ---------------------------------------------------------------------------
# ResourceBudgets.machine_headroom_mb
# ---------------------------------------------------------------------------

def test_headroom_sample_failure_returns_none_and_records():
    def _raise_sample():
        raise RuntimeError("sample died")

    budgets = res.ResourceBudgets(SimpleNamespace(
        sample=_raise_sample, effective_memory_mb=lambda: 8192.0,
        effective_cpu_count=lambda: 8.0))
    assert budgets.machine_headroom_mb() is None
    assert _sites() == ["ResourceBudgets.machine_headroom_mb"]


def test_headroom_unreadable_memory_returns_none_without_record():
    budgets = res.ResourceBudgets(_stub_manager(
        sample=_sample(eff_mem=None, avail=None)))
    assert budgets.machine_headroom_mb() is None
    assert _sites() == []


def test_headroom_success_records_nothing():
    budgets = res.ResourceBudgets(_stub_manager(sample=_sample()))
    headroom = budgets.machine_headroom_mb()
    assert headroom == pytest.approx(8192.0 - max(8192.0 * 0.06, 256.0))
    assert _sites() == []


# ---------------------------------------------------------------------------
# ResourceBudgets.acquire
# ---------------------------------------------------------------------------

def test_acquire_error_refusal_records(monkeypatch):
    budgets = res.ResourceBudgets(_stub_manager(sample=_sample()))

    def _raise_admission(name, **kwargs):
        raise RuntimeError("admission blew up")

    monkeypatch.setattr(budgets, "admission", _raise_admission)
    assert budgets.acquire("x", mem_mb=10.0) is False
    assert _sites() == ["ResourceBudgets.acquire"]


def test_acquire_over_budget_refusal_records_nothing():
    budgets = res.ResourceBudgets(_stub_manager(sample=_sample()))
    budgets.define("x", max_mem_mb=100.0)
    assert budgets.acquire("x", mem_mb=1000.0) is False
    assert _sites() == []


# ---------------------------------------------------------------------------
# ResourceBudgets.release
# ---------------------------------------------------------------------------

class _PartialBudget:
    """Mem releases, cpu release raises: partial mutation + record."""

    def __init__(self):
        self.used_mem_mb = 500.0
        self.__dict__["used_cpu_pct"] = 50.0
        self.running = 1

    @property
    def used_cpu_pct(self):
        return self.__dict__["used_cpu_pct"]

    @used_cpu_pct.setter
    def used_cpu_pct(self, value):
        raise RuntimeError("cpu unsettable")


def test_release_partial_failure_leaks_and_records():
    budgets = res.ResourceBudgets(_stub_manager())
    partial = _PartialBudget()
    budgets._budgets = {"x": partial}
    budgets.release("x", mem_mb=100.0, cpu_pct=10.0)
    assert partial.used_mem_mb == pytest.approx(400.0)
    assert partial.running == 1  # never reached: usage leaked
    assert _sites() == ["ResourceBudgets.release"]


def test_release_success_reduces_and_records_nothing():
    budgets = res.ResourceBudgets(_stub_manager(sample=_sample()))
    budgets.define("x")
    assert budgets.acquire("x", mem_mb=100.0) is True
    budgets.release("x", mem_mb=100.0)
    assert budgets._budgets["x"].used_mem_mb == pytest.approx(0.0)
    assert budgets._budgets["x"].running == 0
    assert _sites() == []


# ---------------------------------------------------------------------------
# ResourceBudgets.snapshot
# ---------------------------------------------------------------------------

def test_snapshot_failure_reports_empty_and_records(monkeypatch):
    budgets = res.ResourceBudgets(_stub_manager())
    budgets.define("x", max_mem_mb=100.0)

    def _raise_defaults():
        raise RuntimeError("sizing died")

    monkeypatch.setattr(budgets, "ensure_defaults", _raise_defaults)
    assert budgets.snapshot() == {}
    assert _sites() == ["ResourceBudgets.snapshot"]


def test_snapshot_empty_budgets_reports_empty_without_record():
    budgets = res.ResourceBudgets(_stub_manager())
    budgets._sized = True
    budgets._budgets = {}
    assert budgets.snapshot() == {}
    assert _sites() == []


# ---------------------------------------------------------------------------
# scan pin — suppression-off remainder across resources.py
# ---------------------------------------------------------------------------

BUDGET_FUNCTIONS = {
    "update", "define", "ensure_defaults", "reserve_mb",
    "machine_headroom_mb", "acquire", "release", "snapshot",
}
EXCLUDED_PROBE_FUNCTIONS = {
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


SAMPLING_CONVERTED_FUNCTIONS = {
    # Converted by #127; effective_memory_mb and sample remain in the
    # remainder via their excluded sites (meminfo pass, env fallback),
    # so they are NOT in this set.
    "_is_battery_powered", "effective_cpu_count", "_sample_cpu_deltas",
    "_sample_swap", "_sample_swap_io_rate", "_sample_thermal", "safe",
}
SAMPLING_EXCLUDED_FUNCTIONS = {
    # Innermost functions holding #127's excluded sampling sites:
    # effective_memory_mb (meminfo pass), _sample_network (sysfs pass),
    # sample (env fallback).
    "effective_memory_mb", "_sample_network", "sample",
}


def test_scan_remainder_is_19_with_no_budget_or_sampling_sites():
    from nomorals.tools import error_scan as es

    original = es._noqa_suppresses
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([os.path.join(os.path.dirname(res.__file__),
                                       "resources.py")])
        # 36 true - 8 budgets (#125) - 9 sampling (#127) = 19.
        # A count snapshot a conversion is supposed to change, not an
        # architectural invariant: the real invariant — which functions
        # remain — is asserted below and is untouched.
        assert len(report.findings) == 19
        fns = {_enclosing_function(f.file, f.line) for f in report.findings}
        assert not (fns & BUDGET_FUNCTIONS)
        assert not (fns & SAMPLING_CONVERTED_FUNCTIONS)
        assert EXCLUDED_PROBE_FUNCTIONS <= fns
        assert SAMPLING_EXCLUDED_FUNCTIONS <= fns
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
