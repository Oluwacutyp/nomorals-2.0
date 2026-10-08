"""Offline tests for nomorals/planning/estimates.py (#91)."""

import os
import tempfile

import pytest

from nomorals.planning import estimates
from nomorals.planning.estimates import EstimateStore, _fmt_band, control_eta


def _store():
    return EstimateStore(db_path=os.path.join(tempfile.mkdtemp(), "e.db"))


def test_band_not_point():
    s = _store()
    est = s.estimate("research", {"prep": 10, "read": 20, "write": 15})
    assert est.low <= est.point <= est.high
    assert est.low < est.high  # a real band, never a point
    assert "range" in est.format()


def test_segment_decomposition():
    s = _store()
    est = s.estimate("delivery", {"prep": 10, "travel": 20, "dropoff": 5})
    assert len(est.segments) == 3
    names = {sg.name for sg in est.segments}
    assert names == {"prep", "travel", "dropoff"}
    total = sum(sg.point for sg in est.segments)
    assert abs(total - est.point) < 1e-6


def test_conservative_deadline_is_max():
    s = _store()
    est = s.estimate("research", {"prep": 10, "write": 20})
    assert est.deadline == est.high  # Meituan: quote the max
    assert est.deadline >= est.point


def test_miss_widens_band():
    s = _store()
    before = s.estimate("research", {"write": 30})
    width_before = before.high - before.low
    # Three big misses.
    for _ in range(3):
        s.record_actual("research", 30, 90, {"write": 90})
    after = s.estimate("research", {"write": 30})
    assert s.miss_count("research") >= 1
    assert (after.high - after.low) > width_before


def test_hit_does_not_widen():
    s = _store()
    before = s.estimate("build", {"compile": 10})
    w0 = before.high - before.low
    s.record_actual("build", 10, 11, {"compile": 11})  # inside the band
    after = s.estimate("build", {"compile": 10})
    assert (after.high - after.low) <= w0 * 1.01


def test_bias_correction_learns_long_tasks():
    s = _store()
    for _ in range(4):
        s.record_actual("deploy", 10, 20, {"ship": 20})
    est = s.estimate("deploy", {"ship": 10})
    # Historical 2x bias shifts the band upward.
    assert est.point > 10


def test_delay_risk_low_early():
    s = _store()
    assert s.delay_risk("research", 5, {"write": 30}) < 0.5


def test_delay_risk_high_when_overdue():
    s = _store()
    est = s.estimate("research", {"write": 30})
    risk = s.delay_risk("research", est.high + 30, {"write": 30})
    assert risk > 0.7


def test_delay_alert_none_when_fine():
    s = _store()
    assert s.delay_alert("research", 5, {"write": 30}) is None


def test_delay_alert_fires_when_late():
    s = _store()
    est = s.estimate("research", {"write": 30})
    alert = s.delay_alert("research", est.high + 60, {"write": 30})
    assert alert is not None
    assert "running late" in alert
    assert "new ETA" in alert


def test_never_raises_on_garbage():
    s = EstimateStore(db_path="/nonexistent-dir-xyz/e.db")
    est = s.estimate(None, "garbage")
    assert est.low <= est.point <= est.high
    assert s.record_actual(None, "x", "y") is False
    assert s.delay_risk(None, "x") >= 0.0
    assert s.delay_alert(None, "x") is None or isinstance(s.delay_alert(None, "x"), str)


def test_fmt_band():
    assert _fmt_band(0.5) == "30s"
    assert _fmt_band(20) == "20 min"
    assert _fmt_band(90) == "1h 30m"
    assert _fmt_band("junk") == "? min"


def test_chat_estimate():
    db = os.path.join(tempfile.mkdtemp(), "e.db")
    out = control_eta("research prep=10 write=20", db_path=db)
    assert "range" in out
    assert "quoted deadline" in out
    assert "bands from past performance" in out


def test_chat_record_and_stats():
    db = os.path.join(tempfile.mkdtemp(), "e.db")
    out = control_eta("record research 30 90", db_path=db)
    assert "logged" in out
    out2 = control_eta("stats research", db_path=db)
    assert "research" in out2
    assert "miss" in out2


def test_chat_risk():
    db = os.path.join(tempfile.mkdtemp(), "e.db")
    out = control_eta("risk research 5", db_path=db)
    assert "miss risk" in out


def test_chat_help_and_garbage():
    db = os.path.join(tempfile.mkdtemp(), "e.db")
    assert "usage" in control_eta("", db_path=db)
    assert "usage" in control_eta("help", db_path=db)
    out = control_eta("\x00\x01", db_path=db)  # garbage never raises
    assert isinstance(out, str) and out


def test_learned_source_after_history():
    s = _store()
    for _ in range(3):
        s.record_actual("compile", 10, 12, {"compile": 12})
    est = s.estimate("compile", {"compile": 0})  # no caller guess → learned avg
    assert est.source == "learned"
    assert est.point > 0


def test_eta_text_format():
    import time as _t
    out = estimates.eta_text("research", {"write": 30})
    assert "range" in out
    out2 = estimates.eta_text("research", {"write": 30}, start_epoch=_t.time() + 3600)
    assert "ETA" in out2
