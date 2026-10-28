"""Pass #215 pins: planning estimates degrade, never crash.

``nomorals/planning/estimates.py`` converts 1 of its 12 true sites
(214, per-item skip — parent adjudication) and excludes 11. The
module contract is the ground for the eleven exclusions:
"Every function never raises: a bad estimate path degrades to a
wide honest band, never a crash and never a fake-precise point."
Each excluded handler returns a value its own function already
returns on a non-error path (or, for the reference-class check,
its own documented error channel, which callers surface):

- malformed items are skipped per-item; later valid segments
  survive (214, CONVERTED — the old code stopped at the first
  malformed item and silently dropped later valid segments);
- failed learned lookups keep the caller point, the +-25%
  half-width, and the no-history stats (233, 246, 264);
- reporting accessors return their empty values: miss count 0,
  mid-scale risk 0.5, no alert (339, 360, 380);
- band builders return the shared generic band (30/15/60,
  confidence 0.3) their sibling ``estimate`` fallback uses
  (431, 464);
- the reference-class check returns its ok:False channel
  (787) and calibration returns its no-data empty list (818);
- the one-line ETA returns its unavailable string (980).

Weakest, ranked: 380 (an error suppresses the proactive alert — but the risk
figure is still reported by the sibling call, and alerting on
unknown risk would be the false signal), 360 (the 0.5 midpoint
could read as a measured 50% — it is the neutral value, and
the alert threshold treats it as no-alert).

The 12 behavior pins below prove each designed output; the
no-write pin proves the failed lookups write nothing to the
learned store; the 4 durability tests mirror the #141+ shape
for the store these pins touch. Every trigger was probed live
before its pin was written. All stores use explicit ``:memory:``
or tmp paths (never the real home, never the repo tree).

Pins rule (#169): no bare-swallow handler anywhere in this
file — the scanner counts test files too. These pins need no
handler at all: triggers use bad inputs, closed connections,
and ``unittest.mock.patch`` instead of swallowing.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from nomorals.planning import estimates as E

GENERIC = (30.0, 15.0, 60.0, 0.3, "heuristic")


def _mem():
    return E.EstimateStore(":memory:")


def _band(est):
    return (est.point, est.low, est.high, est.confidence, est.source)


# ── L214 (CONVERTED by parent adjudication): malformed items are
# skipped per-item; later valid segments survive ─────────────────

def test_norm_segments_skips_malformed_keeps_valid():
    est = _mem().estimate("t", [("a", 5), 42, ("b", 3)])
    assert [(s.name, s.point) for s in est.segments] == [("a", 5.0), ("b", 3.0)]


def test_norm_segments_object_input_bands_total():
    est = _mem().estimate("t", object())
    assert est.segments[0].name == "total"


# ── L233 / L246 / L264: failed learned lookups ────────────────────

def _closed():
    store = _mem()
    store._db.close()
    return store


def test_segment_band_closed_db_keeps_caller_point():
    est = _closed().estimate("t", {"prep": 0})
    assert est.segments[0].point == 0.0
    assert est.source == "heuristic"


def test_segment_band_closed_db_keeps_base_half_width():
    est = _closed().estimate("t", {"prep": 10})
    assert (est.segments[0].low, est.segments[0].high) == (7.5, 12.5)


def test_stats_closed_db_returns_base():
    assert _closed()._stats("t") == {"n": 0, "misses": 0, "ema_ratio": 1.0}


# ── L339 / L360 / L380: reporting accessors ───────────────────────

def test_miss_count_nonstring_returns_zero():
    assert _mem().miss_count(123) == 0
    assert _mem().miss_count("nothing-recorded") == 0


def test_delay_risk_junk_returns_midpoint():
    assert _mem().delay_risk("t", None) == 0.5


def test_delay_alert_failed_risk_returns_none():
    store = _mem()
    with patch.object(E.EstimateStore, "delay_risk",
                      side_effect=RuntimeError("boom")):
        assert store.delay_alert("t", 10) is None
    assert store.delay_alert("t", None) is None


# ── L431 / L464: band builders ────────────────────────────────────

def test_pert_estimate_failed_store_returns_generic_band():
    with patch.object(E, "get_store", side_effect=RuntimeError("boom")):
        assert _band(E.pert_estimate("t", 10, 20, 40)) == GENERIC


def test_aggregate_junk_part_returns_generic_band():
    assert _band(E.aggregate("x", ["junk"])) == GENERIC
    assert _band(E.aggregate("x", [])) == GENERIC


# ── L787 / L818 / L980: check, calibration, one-line ETA ──────────

def test_reference_class_failed_check_returns_error_channel():
    assert E.reference_class_check("t", "junk") == {
        "ok": False, "reason": "check failed"}


class _BadDB:
    def execute(self, *a, **k):
        raise RuntimeError("boom")


class _BadStore:
    _db = _BadDB()


def test_calibration_failed_query_returns_empty():
    with patch.object(E, "get_store", return_value=_BadStore()):
        assert E.calibration() == []


def test_eta_text_failed_store_returns_unavailable():
    with patch.object(E, "get_store", side_effect=RuntimeError("boom")):
        assert E.eta_text("t", {"a": 5}) == "⏱️ estimate unavailable"


# ── NOT-recording: failed lookups write nothing ───────────────────

def test_failed_lookups_write_nothing_to_store(tmp_path):
    path = str(tmp_path / "e.db")
    store = E.EstimateStore(path)
    store._db.close()
    store.estimate("t", {"prep": 0})
    store.estimate("t", {"prep": 10})
    store._stats("t")
    store.miss_count("t")
    store.delay_risk("t", None)
    with patch.object(E.EstimateStore, "delay_risk",
                      side_effect=RuntimeError("boom")):
        store.delay_alert("t", 10)
    reopened = E.EstimateStore(path)
    rows = reopened._db.execute(
        "SELECT COUNT(*) FROM estimate_stats").fetchone()[0]
    seg_rows = reopened._db.execute(
        "SELECT COUNT(*) FROM estimate_segments").fetchone()[0]
    assert (rows, seg_rows) == (0, 0)


# ── Durability: the learned store these pins touch ────────────────

def test_record_persists_across_reopen(tmp_path):
    path = str(tmp_path / "e.db")
    E.EstimateStore(path).record_actual("build", 20.0, 40.0)
    assert E.EstimateStore(path)._stats("build")["n"] == 1


def test_miss_widens_band(tmp_path):
    path = str(tmp_path / "e.db")
    store = E.EstimateStore(path)
    fresh = store.estimate("build", {"total": 20.0})
    store.record_actual("build", 20.0, 40.0)
    after = E.EstimateStore(path).estimate("build", {"total": 20.0})
    assert (after.low, after.high) == (13.75, 26.25)
    assert after.high - after.low > fresh.high - fresh.low


def test_repeated_overruns_shift_band_and_count_misses(tmp_path):
    path = str(tmp_path / "e.db")
    store = E.EstimateStore(path)
    for _ in range(3):
        store.record_actual("build", 10.0, 20.0)
    est = E.EstimateStore(path).estimate("build", {"total": 10.0})
    assert est.source == "learned"
    assert est.point > 10.0
    assert est.miss_count == 3


def test_empty_store_defaults(tmp_path):
    store = E.EstimateStore(str(tmp_path / "e.db"))
    est = store.estimate("newtype", None)
    assert est.source == "heuristic"
    assert est.confidence == 0.4
    assert store.miss_count("newtype") == 0
    assert store._stats("newtype") == {"n": 0, "misses": 0, "ema_ratio": 1.0}
