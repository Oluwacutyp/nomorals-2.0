"""Pins for Pass #128: analysis-chain failure ledger in
nomorals/os/resources.py, reusing the #123 resources_failures ledger.

Scope: the ResourceManager analysis sub-cluster — _record, history,
trend, forecast, battery_forecast, composite_pressure, pressure,
_degradation_target, _eval_alerts, render, summary_line (11 sites).
Shared failure story: an analysis that silently lies. A failed history
append silently drops the sample; a failed history copy silently reports
an empty history; a malformed history row silently reports "insufficient
data" from trend or an empty prediction set from forecast; a broken
pressure computation silently reports "all clear"; a broken degradation
mapping silently reports level 0, not critical; a failing alert callback
silently drops the alert edge; a broken dashboard silently reports
"unavailable" or "n/a". The caller sees a healthy-looking analysis
throughout. This pass closes resources.py: the remainder after it is
exactly the 8 prior exclusions (5 probe + 3 sampling).

No deliberate exclusions in this region: every analysis fallback asserts
something false or unknown about the input (the #124 discriminator) or
loses the cause while naming the fact (the #123 detect_environment
precedent for render/summary_line), and absence is already handled
explicitly in each case, making every except residual-only.
"""

import ast
import os
import time
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


def _sample(**over):
    base = dict(cpu_percent=10.0, mem_percent=20.0, disk_percent=30.0,
                battery_percent=None, thermal_state=None,
                network_online=True, metered=None)
    base.update(over)
    return res.ResourceSample(**base)


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

def test_record_records_on_failed_append():
    class HostileList(list):
        def append(self, _item):
            raise RuntimeError("disk full")

    m = _bare_manager(_history=HostileList(),
                      composite_pressure=lambda s: 0.5)
    m._record(_sample())
    assert _sites() == ["ResourceManager._record"]

    m2 = _bare_manager(_history=[],
                       composite_pressure=lambda s: 0.5)
    m2._record(_sample())
    assert _sites() == ["ResourceManager._record"]
    assert len(m2._history) == 1
    assert m2._history[0]["composite"] == 0.5


def test_history_records_on_failed_copy():
    m = _bare_manager(_history=[None])
    assert m.history() == []
    assert _sites() == ["ResourceManager.history"]

    m2 = _bare_manager(_history=[{"ts": 1.0}])
    assert m2.history() == [{"ts": 1.0}]
    assert _sites() == ["ResourceManager.history"]


def test_trend_records_on_malformed_row():
    m = _bare_manager(_history=["bad-row"])
    assert m.trend("cpu") is None
    assert _sites() == ["ResourceManager.trend"]

    now = time.time()
    m2 = _bare_manager(_history=[{"ts": now - 10, "cpu": 10.0},
                                 {"ts": now, "cpu": 20.0}])
    assert m2.trend("cpu") == pytest.approx(1.0)
    assert _sites() == ["ResourceManager.trend"]


def test_forecast_records_on_malformed_last_row():
    m = _bare_manager(_history=["bad-row"], cpu_critical=0.9,
                      mem_critical=0.9, disk_critical=0.9)
    out = m.forecast()
    assert out["predicted"] == {}
    assert _sites() == ["ResourceManager.forecast"]

    now = time.time()
    rows = [{"ts": now - 10, "cpu": 10.0, "mem": 10.0, "disk": 10.0},
            {"ts": now, "cpu": 20.0, "mem": 20.0, "disk": 20.0}]
    m2 = _bare_manager(_history=rows, cpu_critical=0.9,
                       mem_critical=0.9, disk_critical=0.9)
    out2 = m2.forecast()
    assert out2["predicted"]["cpu"] == pytest.approx(100.0)
    assert _sites() == ["ResourceManager.forecast"]


def test_battery_forecast_records_on_malformed_ts():
    rows = [{"battery": 80.0, "ts": "bad", "battery_status": "Discharging"},
            {"battery": 70.0, "ts": "alsobad", "battery_status": "Discharging"}]
    m = _bare_manager(_history=rows)
    out = m.battery_forecast()
    assert out["drain_pct_per_h"] is None
    assert _sites() == ["ResourceManager.battery_forecast"]

    now = time.time()
    rows2 = [{"battery": 80.0, "ts": now - 3600,
              "battery_status": "Discharging"},
             {"battery": 70.0, "ts": now, "battery_status": "Discharging"}]
    m2 = _bare_manager(_history=rows2)
    out2 = m2.battery_forecast()
    assert out2["drain_pct_per_h"] == pytest.approx(10.0)
    assert out2["time_to_empty_min"] == pytest.approx(420.0)
    assert _sites() == ["ResourceManager.battery_forecast"]


def test_composite_pressure_records_on_malformed_value():
    m = _bare_manager(_sample_swap_io_rate=lambda: None)
    assert m.composite_pressure(_sample(swap_used_percent="bad")) == 0.0
    assert _sites() == ["ResourceManager.composite_pressure"]

    m2 = _bare_manager(_sample_swap_io_rate=lambda: None)
    assert m2.composite_pressure(_sample(mem_percent=50.0)) == pytest.approx(0.075)
    assert _sites() == ["ResourceManager.composite_pressure"]


def test_pressure_records_on_malformed_value():
    m = _bare_manager(_sample_swap_io_rate=lambda: None)
    out = m.pressure(_sample(cpu_percent="bad"))
    assert out == {"cpu": 0.0, "cpu_util": 0.0, "mem": 0.0, "disk": 0.0,
                   "battery": 0.0, "thermal": 0.0, "steal": 0.0,
                   "psi_mem": 0.0, "composite_mem": 0.0, "overall": 0.0}
    assert _sites() == ["ResourceManager.pressure"]

    m2 = _bare_manager(_sample_swap_io_rate=lambda: None)
    out2 = m2.pressure(_sample(mem_percent=50.0))
    assert out2["mem"] == pytest.approx(0.5)
    assert out2["overall"] == pytest.approx(0.5)
    assert _sites() == ["ResourceManager.pressure"]


def test_degradation_target_records_on_missing_pressure_key():
    m = _bare_manager(cpu_throttle=0.99, mem_throttle=0.99,
                      disk_throttle=0.99, battery_throttle=0.99,
                      steal_warn=0.99, psi_mem_warn=0.99,
                      cpu_critical=0.99, mem_critical=0.99,
                      disk_critical=0.99, battery_min=0.99,
                      steal_critical=0.99, psi_mem_full_critical=0.99,
                      max_pressure=1.0, metered=False)
    assert m._degradation_target(_sample(), {}) == (0, False)
    assert _sites() == ["ResourceManager._degradation_target"]

    p = {"cpu": 0.0, "mem": 0.0, "disk": 0.0, "composite_mem": 0.0,
         "overall": 0.0}
    assert m._degradation_target(_sample(), p) == (0, False)
    assert _sites() == ["ResourceManager._degradation_target"]


def test_eval_alerts_records_on_failing_callback():
    def _boom(_event):
        raise RuntimeError("webhook down")

    m = _bare_manager(
        _alerts=[{"id": "t", "key": "cpu", "threshold": 50.0,
                  "above": True, "fn": _boom}],
        _alert_state={})
    m._eval_alerts(_sample(), {"cpu": 60.0})
    assert _sites() == ["ResourceManager._eval_alerts"]
    assert m._alert_state["t"] is True  # edge state set before the callback

    seen = []
    m2 = _bare_manager(
        _alerts=[{"id": "t", "key": "cpu", "threshold": 50.0,
                  "above": True, "fn": seen.append}],
        _alert_state={})
    m2._eval_alerts(_sample(), {"cpu": 60.0})
    assert len(seen) == 1
    assert _sites() == ["ResourceManager._eval_alerts"]


def test_render_records_on_failed_acquisition(monkeypatch):
    m = _bare_manager()
    monkeypatch.setattr(m, "sample",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert m.render() == "resources — unavailable"
    assert _sites() == ["ResourceManager.render"]

    m2 = _bare_manager(
        ladder=SimpleNamespace(step=lambda: SimpleNamespace(name="L0")))
    text = m2.render(_sample(), {"mem": 0.2, "disk": 0.3, "overall": 0.3})
    assert text.startswith("resources\n")
    assert "ladder: L0" in text
    assert _sites() == ["ResourceManager.render"]


def test_summary_line_records_on_unformattable_cpu():
    m = _bare_manager()
    m.sample = lambda: _sample(cpu_percent=None)
    m.pressure = lambda s: {"cpu": 0.0, "mem": 0.2, "disk": 0.3,
                            "overall": 0.3}
    assert m.summary_line() == "resources n/a"
    assert _sites() == ["ResourceManager.summary_line"]

    m2 = _bare_manager()
    m2.sample = lambda: _sample(cpu_percent=10.0)
    m2.pressure = lambda s: {"cpu": 0.1, "mem": 0.2, "disk": 0.3,
                             "overall": 0.3}
    assert m2.summary_line() == "cpu 10% · mem 20% · disk 30% · ok"
    assert _sites() == ["ResourceManager.summary_line"]


# ---------------------------------------------------------------------------
# scan pins — suppression-off remainder across resources.py
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


def test_scan_remainder_is_8_excluded_only():
    from nomorals.tools import error_scan as es

    original = es._noqa_suppresses
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([os.path.join(os.path.dirname(res.__file__),
                                       "resources.py")])
        # 19 (#127) - 11 analysis converted (#128) = 8: 5 excluded
        # probe + 3 excluded sampling. resources.py is now closed: no
        # converted region remains, only the prior exclusions.
        assert len(report.findings) == 8
        from collections import Counter
        fns = Counter(_enclosing_function(f.file, f.line)
                      for f in report.findings)
        assert fns == Counter({
            # excluded probe region (#123)
            "_cgroup_mount": 1, "_read_cpu_stat": 1, "_read_swap": 1,
            "_read_swap_io": 1, "_self_rss_mb": 1,
            # excluded sampling sites (#127)
            "effective_memory_mb": 1, "_sample_network": 1, "sample": 1,
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
        assert len(report.findings) == 0
    finally:
        es._noqa_suppresses = original
