"""Pass 3 of Devon's trading system: judgment + forecast evidence.

The forecast organ (agents/forecast.py — empirical per-horizon
forecasts to the 2026-11-03 output standard, append-only outcome
attachment, the calibration book and the E3 trust rule) and the
judgment core (agents/judgment.py — market-state estimation, five
competing scenarios, evidence groups, act/abstain in shadow,
journaled beside the Pass 2 lifecycle).

Every test here is a NEW-BEHAVIOR test: on pristine 90d7965 the
organs do not exist and the file fails at import. Preservation of
Pass 1-2 behavior is carried by their own suites + neighborhoods.
"""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nomorals.agents import forecast as fc
from nomorals.agents import judgment as jd
from nomorals.finance import trading_journal
from nomorals.integrations import market_data

BASE_TS = 1762362000.0  # 2025-11-05 17:00 UTC


# ── constructed H1 history ───────────────────────────────────────────

def hist_frame(*, n=900, drift=0.0004, end_ts=BASE_TS):
    """Geometric-drift H1 history with a deterministic wiggle.

    Forward returns vary bar to bar (the wiggle), so the empirical
    distribution has real spread — quantiles are checkable against
    the sample itself, never constants.
    """
    closes = [2000.0 * math.exp(drift * i + 0.01 * math.sin(
        i * 2 * math.pi / 50.0)) for i in range(n)]
    rows, idx = [], []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c * (1 - drift)
        hi = max(o, c) * (1 + 0.0015 + 0.0004 * math.sin(i * 1.7))
        lo = min(o, c) * (1 - 0.0015 - 0.0004 * math.cos(i * 1.3))
        rows.append((o, hi, lo, c, 100.0))
        idx.append(pd.Timestamp(end_ts - (n - 1 - i) * 3600,
                                unit="s", tz="UTC"))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                     "volume"], index=idx)
    df.attrs["source"] = "fake_h1"
    df.attrs["freshness"] = "fresh"
    df.attrs["last_bar_ts"] = end_ts
    return df


def _forecast(tmp_path, frame, *, now=BASE_TS, seed=None):
    journal = tmp_path / "j.jsonl"
    records = fc.make_forecasts("XAUUSD", market="forex",
                                journal_path=journal, now=now,
                                frame=frame, seed_samples=seed)
    return journal, records


# ── forecast: the output standard ────────────────────────────────────

def test_forecast_records_carry_every_field_per_horizon(tmp_path):
    journal, records = _forecast(tmp_path, hist_frame())
    assert [r["horizon"] for r in records] == list(fc.HORIZONS)
    for rec in records:
        assert rec["direction"] in ("UP", "DOWN", "NEUTRAL")
        assert rec["point_price"] is not None
        iv = rec["interval"]
        assert iv["low"] < rec["point_price"] < iv["high"]
        assert iv["coverage"] == fc.INTERVAL_COVERAGE == 0.80
        assert rec["expected_move_pct"] is not None
        assert rec["expected_move_price"] is not None
        assert rec["mean_abs_move_pct"] > 0
        assert rec["fav_excursion"] > 0 and rec["adv_excursion"] > 0
        assert rec["disagreement_pct"] is not None
        assert rec["scenario"] and rec["invalidation"]
        assert rec["invalidation_price"] is not None
        assert rec["created"] == BASE_TS
        assert rec["cutoff"] == BASE_TS  # last bar of the frame
        assert rec["expiry"] - rec["created"] == \
            rec["horizon_bars"] * 3600
        assert rec["version"] == fc.FORECAST_VERSION
        assert rec["method"] == fc.METHOD
        assert rec["n"] >= fc.MIN_SAMPLE_N
        assert rec["forecast_id"].startswith("fc:XAUUSD:")
        assert rec["probabilities"] == {
            "state": "uncalibrated", "n": 0,
            "up": None, "dn": None, "neutral": None}
    events = trading_journal.read_events(limit=0, path=journal,
                                         event="forecast")
    assert len(events) == 5


def test_interval_and_point_are_real_empirical_quantiles(tmp_path):
    frame = hist_frame()
    _journal, records = _forecast(tmp_path, frame)
    by_tag = {r["horizon"]: r for r in records}
    base = float(frame["close"].iloc[-1])
    for tag, bars in (("1d", 24), ("12h", 12)):
        sample = fc.forward_outcomes(frame, bars)["ret_pct"]
        q_lo, q_hi = np.quantile(
            sample, [(1 - 0.80) / 2, 1 - (1 - 0.80) / 2])
        rec = by_tag[tag]
        assert rec["interval"]["low"] == pytest.approx(
            base * (1 + q_lo / 100), abs=0.01)
        assert rec["interval"]["high"] == pytest.approx(
            base * (1 + q_hi / 100), abs=0.01)
        assert rec["point_price"] == pytest.approx(
            base * (1 + float(np.mean(sample)) / 100), abs=0.01)
        assert rec["n"] == len(sample)
    # Not constants: every band has real width and the widths
    # differ across horizons (each is its own sample's quantiles).
    w = {t: by_tag[t]["interval"]["high"]
         - by_tag[t]["interval"]["low"] for t in by_tag}
    assert all(v > 0 for v in w.values())
    assert len({round(v, 2) for v in w.values()}) >= 4


def test_forecast_unavailable_when_sample_thin(tmp_path):
    _journal, records = _forecast(tmp_path, hist_frame(n=100))
    by_tag = {r["horizon"]: r for r in records}
    assert by_tag["15d"]["direction"] is None
    assert "too thin" in by_tag["15d"]["unavailable_reason"]
    assert by_tag["15d"]["point_price"] is None
    assert by_tag["12h"]["direction"] is not None  # 88 windows >= 50


def test_forecast_up_drift_direction_and_invalidation(tmp_path):
    _journal, records = _forecast(tmp_path, hist_frame())
    rec = {r["horizon"]: r for r in records}["1d"]
    assert rec["direction"] == "UP"
    assert rec["expected_move_pct"] > 0
    # UP call: invalidation is the band's adverse (low) edge.
    assert rec["invalidation_price"] == rec["interval"]["low"]


def test_seed_samples_merge_and_shift_the_distribution(tmp_path):
    flat = hist_frame(drift=0.0)
    _j, plain = _forecast(tmp_path, flat)
    plain_1d = {r["horizon"]: r for r in plain}["1d"]
    assert plain_1d["direction"] == "NEUTRAL"
    assert plain_1d["seed_n"] == 0
    seed = {"1d": [{"ret_pct": 3.0, "run_up": 90.0, "run_dn": 10.0}
                   for _ in range(400)]}
    seeded = fc.make_forecasts(
        "XAUUSD", market="forex", journal_path=tmp_path / "j2.jsonl",
        now=BASE_TS, frame=flat, seed_samples=seed)
    seeded_1d = {r["horizon"]: r for r in seeded}["1d"]
    assert seeded_1d["seed_n"] == 400
    assert seeded_1d["direction"] == "UP"
    assert seeded_1d["point_price"] > plain_1d["point_price"]
    assert {r["horizon"]: r for r in seeded}["3d"]["seed_n"] == 0


def test_load_seed_samples_groups_and_skips_bad_rows(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text(
        '{"horizon": "1d", "ret_pct": 1.5, "run_up": 30.0,'
        ' "run_dn": 12.0}\n'
        '{"horizon": 72, "ret_pct": -1.0, "run_up": 8.0,'
        ' "run_dn": 25.0}\n'
        'not json\n'
        '{"horizon": "1d", "ret_pct": 2.5, "run_up": 40.0,'
        ' "run_dn": 9.0}\n', encoding="utf-8")
    out = fc.load_seed_samples(path)
    assert len(out["1d"]) == 2 and len(out["3d"]) == 1
    assert fc.load_seed_samples(tmp_path / "missing.jsonl") == {}


# ── outcomes + calibration ───────────────────────────────────────────

def test_attach_outcome_folds_into_calibration_book(tmp_path):
    journal, records = _forecast(tmp_path, hist_frame())
    rec = {r["horizon"]: r for r in records}["1d"]
    before = trading_journal.read_events(limit=0, path=journal,
                                         event="forecast")
    fc.attach_outcome(rec["forecast_id"], "1d", ret_pct=2.0,
                      run_up=55.0, run_dn=8.0, end_price=2100.0,
                      journal_path=journal, graded_at=BASE_TS + 90000)
    book = fc.calibration_book(symbol="XAUUSD", journal_path=journal)
    entry = book["horizons"]["1d"]
    assert entry["n"] == 1 and entry["hits"] == 1
    assert entry["base_rate"] == 1.0
    assert entry["realized"] == {"up": 1, "dn": 0, "neutral": 0}
    assert entry["by_direction"]["up"]["n"] == 1
    # Append-only: the forecast events are untouched, one outcome
    # event was added beside them.
    after = trading_journal.read_events(limit=0, path=journal,
                                        event="forecast")
    assert after == before
    assert len(trading_journal.read_events(
        limit=0, path=journal, event="forecast_outcome")) == 1
    with pytest.raises(KeyError):
        fc.attach_outcome("fc:NOPE:0:1d", "1d", ret_pct=1.0,
                          run_up=1.0, run_dn=1.0,
                          journal_path=journal)


def test_resolve_due_forecasts_grades_from_market_data(tmp_path):
    journal, _records = _forecast(tmp_path, hist_frame())
    future = hist_frame(n=1000, end_ts=BASE_TS + 100 * 3600)
    made = fc.resolve_due_forecasts(
        "XAUUSD", market="forex", journal_path=journal,
        now=BASE_TS + 13 * 3600, frame=future)
    assert made == 1  # only the 12h horizon has matured
    outcomes = trading_journal.read_events(
        limit=0, path=journal, event="forecast_outcome")
    assert len(outcomes) == 1 and outcomes[0]["horizon"] == "12h"
    base_close = float(future["close"].iloc[899])
    end_close = float(future["close"].iloc[899 + 12])
    assert outcomes[0]["realized"]["ret_pct"] == pytest.approx(
        (end_close / base_close - 1.0) * 100.0, rel=1e-6)
    # Idempotent: a second resolve appends nothing new.
    assert fc.resolve_due_forecasts(
        "XAUUSD", market="forex", journal_path=journal,
        now=BASE_TS + 13 * 3600, frame=future) == 0


def _fabricate(journal, tag, rows, *, direction="UP"):
    """Journal fabricated forecast+outcome pairs for trust tests.

    rows: (conviction_z, realized_direction) pairs.
    """
    for i, (z, realized) in enumerate(rows):
        fid = f"fc:XAUUSD:{1000 + i}:{tag}"
        trading_journal.append_event(
            "forecast", path=journal, forecast_id=fid, horizon=tag,
            symbol="XAUUSD", direction=direction, conviction_z=z,
            deadband_pct=0.0)
        trading_journal.append_event(
            "forecast_outcome", path=journal, forecast_id=fid,
            horizon=tag, symbol="XAUUSD",
            realized_direction=realized,
            realized={"ret_pct": 1.0, "run_up": 1.0, "run_dn": 1.0})


def test_trust_boundary_trusted(tmp_path):
    journal = tmp_path / "j.jsonl"
    rows = [(0.9, "up")] * 18 + [(0.9, "dn")] * 2 \
        + [(0.1, "up")] * 10 + [(0.1, "dn")] * 10
    _fabricate(journal, "1d", rows)
    book = fc.calibration_book(symbol="XAUUSD", journal_path=journal)
    trust = fc.trust_state(book, "1d", 1)
    assert trust["pool_n"] == 20 and trust["n"] == 40
    assert trust["pool_rate"] == pytest.approx(0.9)
    assert trust["base_rate"] == pytest.approx(0.7)
    assert trust["lift"] == pytest.approx(0.9 / 0.7, abs=1e-3)
    assert trust["state"] == "trusted"
    # No dn-side pool exists: the mirror direction stays unproven.
    assert fc.trust_state(book, "1d", -1)["state"] == "unproven"


def test_trust_boundary_pool_too_small_and_mid_lift(tmp_path):
    journal = tmp_path / "j.jsonl"
    rows = [(0.9, "up")] * 19 + [(0.1, "dn")] * 20
    _fabricate(journal, "1d", rows)
    book = fc.calibration_book(symbol="XAUUSD", journal_path=journal)
    trust = fc.trust_state(book, "1d", 1)
    assert trust["pool_n"] == 19 and trust["state"] == "unproven"
    journal2 = tmp_path / "j2.jsonl"
    rows2 = [(0.9, "up")] * 15 + [(0.9, "dn")] * 5 \
        + [(0.1, "up")] * 13 + [(0.1, "dn")] * 7
    _fabricate(journal2, "1d", rows2)
    book2 = fc.calibration_book(symbol="XAUUSD",
                                journal_path=journal2)
    trust2 = fc.trust_state(book2, "1d", 1)
    # pool rate .75 vs base .70 -> lift ~1.07: neither trusted nor
    # anti — unproven, recorded, weightless.
    assert trust2["state"] == "unproven"
    assert trust2["lift"] == pytest.approx(0.75 / 0.70, abs=1e-3)


def test_trust_boundary_anti_calibrated(tmp_path):
    journal = tmp_path / "j.jsonl"
    rows = [(0.9, "up")] * 4 + [(0.9, "dn")] * 16 \
        + [(0.1, "up")] * 16 + [(0.1, "dn")] * 4
    _fabricate(journal, "1d", rows)
    book = fc.calibration_book(symbol="XAUUSD", journal_path=journal)
    trust = fc.trust_state(book, "1d", 1)
    assert trust["state"] == "anti_calibrated"
    assert trust["lift"] == pytest.approx(0.2 / 0.5, abs=1e-3)


def test_probabilities_earned_at_twenty_graded(tmp_path):
    journal = tmp_path / "j.jsonl"
    rows = [(0.3, d) for d in (["up"] * 10 + ["dn"] * 6
                               + ["neutral"] * 4)]
    _fabricate(journal, "3d", rows)
    book = fc.calibration_book(symbol="XAUUSD", journal_path=journal)
    probs = fc.probabilities_from_book(book, "3d")
    assert probs == {"state": "calibrated", "n": 20, "up": 0.5,
                     "dn": 0.3, "neutral": 0.2}
    journal2 = tmp_path / "j2.jsonl"
    _fabricate(journal2, "3d", rows[:19])
    book2 = fc.calibration_book(symbol="XAUUSD",
                                journal_path=journal2)
    assert fc.probabilities_from_book(book2, "3d")["state"] == \
        "uncalibrated"


def test_latest_forecasts_fold_newest_per_horizon(tmp_path):
    journal = tmp_path / "j.jsonl"
    frame = hist_frame()
    fc.make_forecasts("XAUUSD", market="forex", journal_path=journal,
                      now=BASE_TS, frame=frame)
    fc.make_forecasts("XAUUSD", market="forex", journal_path=journal,
                      now=BASE_TS + 3600, frame=frame)
    latest = fc.latest_forecasts("XAUUSD", journal_path=journal)
    assert set(latest) == set(fc.HORIZONS)
    assert all(r["created"] == BASE_TS + 3600
               for r in latest.values())


# ── judgment: read/setup construction ────────────────────────────────

def _block(**kw):
    base = {"status": "ok", "bias": None, "phase": None, "zones": None,
            "fvgs": None, "order_blocks": None,
            "premium_discount": None, "last_break": None}
    base.update(kw)
    return base


def _read(*, swing=1, intraday=1, scalp=1, alignment="aligned_up",
          entry_phase="uptrend", killzone=True, entry_status="ok"):
    def hier(bias):
        return {"timeframe": "1d", "label": "D1", "status": "ok",
                "bias": bias,
                "bias_label": "UP" if bias == 1 else "DOWN"}
    return {
        "kind": "market_read", "symbol": "XAUUSD", "market": "forex",
        "taken_at": BASE_TS,
        "price": {"price": 2000.0},
        "session": {"name": "new_york", "killzone": killzone},
        "hierarchy": {"swing": hier(swing), "intraday": hier(intraday),
                      "scalp": hier(scalp)},
        "alignment": alignment,
        "timeframes": {
            "1d": _block(phase="uptrend"),
            "4h": _block(phase="uptrend"),
            "1h": _block(status=entry_status, phase=entry_phase,
                         premium_discount={
                             "premium": 0.8, "zone": "premium",
                             "range_high": 2100.0,
                             "range_low": 1900.0})},
        "degraded": [], "complete": True}


def _setup(*, direction=1, targets=(106.0, 110.0), evidence=None,
           state="armed"):
    return {
        "setup_id": "fake_h1:XAUUSD:k1", "symbol": "XAUUSD",
        "market": "forex", "playbook": "fake_h1", "route": "fake_h1",
        "family": "trend", "entry_timeframe": "1h",
        "direction": direction, "pattern_key": "k1", "state": state,
        "plan": {"entry_zone": {"top": 101.0, "bottom": 99.0},
                 "entry_ref": 100.0, "invalidation": 98.0,
                 "targets": list(targets)},
        "evidence": evidence or {},
        "read_taken_at": BASE_TS}


# ── judgment: the core ───────────────────────────────────────────────

def test_judgment_acts_with_scenario_why_and_invalidation():
    out = jd.evaluate_setup(_read(), _setup(
        evidence={"committee_stance": "aligned"}))
    assert out["mode"] == "shadow" and out["decision"] == "act"
    assert out["chosen_scenario"] in ("continuation", "expansion")
    assert len(out["scenarios"]) == 5
    assert sum(s["p"] for s in out["scenarios"]) == pytest.approx(
        1.0, abs=0.01)
    assert out["expected_r"] > 0 and out["confidence"] >= 0.15
    assert out["chosen_scenario"] in out["why"]
    assert out["invalidation"]["price"] == 98.0
    assert "kills the fake_h1 premise" in out["invalidation"]["text"]
    groups = out["evidence"]["groups"]
    assert set(groups) == {"htf_structure", "setup_premise",
                           "geometry", "committee", "session",
                           "forecast_ensemble"}
    # No forecasts on record: the group is stated absent and
    # weightless, trust unproven — recorded, never filled.
    fg = groups["forecast_ensemble"]
    assert fg["stance"] == "absent" and fg["weighed"] is False
    assert fg["trust"] == "unproven"
    assert out["subject"]["lifecycle_state"] == "armed"


def test_judgment_abstains_when_reversal_leads():
    read = _read(swing=-1, intraday=-1, scalp=-1,
                 alignment="aligned_down")
    out = jd.evaluate_setup(read, _setup(targets=(102.0,),
                                         evidence={}))
    assert out["decision"] == "abstain"
    assert out["chosen_scenario"] == "reversal"
    assert out["reason_category"] == "direction_opposition"
    assert "reversal wins" in out["why"]


def test_judgment_abstains_in_chop_state():
    read = _read(swing=0, intraday=0, scalp=0, alignment="flat",
                 entry_phase="range", killzone=False)
    out = jd.evaluate_setup(read, _setup(targets=(102.0,),
                                         evidence={}))
    assert out["decision"] == "abstain"
    assert out["chosen_scenario"] == "chop"
    assert out["reason_category"] == "regime_chop"


def test_judgment_invalid_geometry_abstains():
    out = jd.evaluate_setup(_read(), _setup(targets=()))
    assert out["decision"] == "abstain"
    assert out["reason_category"] == "invalid_geometry"
    assert out["scenarios"] == []


def test_judgment_data_gap_abstains():
    out = jd.evaluate_setup(_read(entry_status="missing"), _setup())
    assert out["decision"] == "abstain"
    assert out["reason_category"] == "data_gap"


def test_judgment_is_deterministic():
    read, setup = _read(), _setup(
        evidence={"committee_stance": "aligned"})
    assert jd.evaluate_setup(read, setup) == \
        jd.evaluate_setup(read, setup)


def test_read_level_verdict_is_an_honest_abstention():
    out = jd.evaluate_read(_read())
    assert out["decision"] == "abstain"
    assert out["reason_category"] == "no_setup"
    assert out["market_state"]["alignment"] == "aligned_up"
    assert out["subject"]["type"] == "read"


# ── judgment: the E3 weighing ────────────────────────────────────────

def _forecast_rec(direction="DOWN", z=0.9):
    return {"forecast_id": "fc:X:1:1d", "horizon": "1d",
            "direction": direction, "conviction_z": z}


def _trusted_book():
    return {"horizons": {"1d": {
        "n": 40, "base_rate": 0.7,
        "by_direction": {
            "up": {"n": 20, "hits": 18, "rate": 0.9, "pool_n": 20,
                   "pool_hits": 18, "pool_rate": 0.9},
            "dn": {"n": 20, "hits": 18, "rate": 0.9, "pool_n": 20,
                   "pool_hits": 18, "pool_rate": 0.9}}}}}


def test_forecast_group_weightless_below_trust():
    setup = _setup(evidence={"committee_stance": "aligned"})
    plain = jd.evaluate_setup(_read(), setup)
    contra = jd.evaluate_setup(
        _read(), setup,
        forecasts={"1d": _forecast_rec("DOWN")}, book={"horizons": {}})
    fg = contra["evidence"]["groups"]["forecast_ensemble"]
    assert fg["stance"] == "against" and fg["weighed"] is False
    assert fg["trust"] == "unproven"
    # Weightless means weightless: the support score is untouched.
    assert contra["evidence"]["support_score"] == \
        plain["evidence"]["support_score"]


def test_forecast_group_weighed_above_trust():
    setup = _setup(evidence={"committee_stance": "aligned"})
    plain = jd.evaluate_setup(_read(), setup)
    book = _trusted_book()
    contra = jd.evaluate_setup(
        _read(), setup, forecasts={"1d": _forecast_rec("DOWN")},
        book=book)
    fg = contra["evidence"]["groups"]["forecast_ensemble"]
    assert fg["weighed"] is True and fg["trust"] == "trusted"
    assert contra["evidence"]["support_score"] < \
        plain["evidence"]["support_score"]
    pro = jd.evaluate_setup(
        _read(), setup, forecasts={"1d": _forecast_rec("UP")},
        book=book)
    assert pro["evidence"]["support_score"] > \
        plain["evidence"]["support_score"]


def test_forecast_group_anti_calibrated_recorded_not_weighed():
    setup = _setup()
    book = {"horizons": {"1d": {
        "n": 40, "base_rate": 0.5,
        "by_direction": {"up": {"n": 20, "hits": 4, "rate": 0.2,
                                "pool_n": 20, "pool_hits": 4,
                                "pool_rate": 0.2}}}}}
    out = jd.evaluate_setup(
        _read(), setup, forecasts={"1d": _forecast_rec("UP")},
        book=book)
    fg = out["evidence"]["groups"]["forecast_ensemble"]
    assert fg["trust"] == "anti_calibrated"
    assert fg["weighed"] is False
    assert "ANTI-calibrated" in fg["detail"]


# ── judgment: the pass over the journal ──────────────────────────────

def _journal_armed_setup(journal, *, state="armed"):
    trading_journal.append_event(
        "setup_detected", path=journal, ts=BASE_TS,
        setup_id="fake_h1:XAUUSD:k1", symbol="XAUUSD",
        market="forex", playbook="fake_h1", route="fake_h1",
        family="trend", entry_timeframe="1h", direction=1,
        pattern_key="k1", to_state="detected",
        reason_category="pattern_formed", reason="formed",
        plan={"entry_zone": {"top": 101.0, "bottom": 99.0},
              "entry_ref": 100.0, "invalidation": 98.0,
              "targets": [106.0, 110.0]},
        evidence={"committee_stance": "aligned"},
        detected_bar_ts=BASE_TS, expires_at=BASE_TS + 48 * 3600,
        read_taken_at=BASE_TS)
    if state in ("armed", "triggered"):
        trading_journal.append_event(
            "setup_armed", path=journal, ts=BASE_TS + 60,
            setup_id="fake_h1:XAUUSD:k1", symbol="XAUUSD",
            market="forex", playbook="fake_h1", route="fake_h1",
            family="trend", entry_timeframe="1h", direction=1,
            pattern_key="k1", to_state="armed",
            reason_category="premise_confirmed", reason="armed",
            plan={"entry_zone": {"top": 101.0, "bottom": 99.0},
                  "entry_ref": 100.0, "invalidation": 98.0,
                  "targets": [106.0, 110.0]},
            armed_at=BASE_TS + 60, armed_bar_ts=BASE_TS,
            expires_at=BASE_TS + 48 * 3600, read_taken_at=BASE_TS)
    if state == "triggered":
        trading_journal.append_event(
            "setup_triggered", path=journal, ts=BASE_TS + 120,
            setup_id="fake_h1:XAUUSD:k1", symbol="XAUUSD",
            market="forex", playbook="fake_h1", route="fake_h1",
            family="trend", entry_timeframe="1h", direction=1,
            pattern_key="k1", to_state="triggered",
            reason_category="zone_touched", reason="touched",
            trigger={"bar_ts": BASE_TS + 120, "price": 101.0},
            read_taken_at=BASE_TS)


def test_judgment_pass_journals_beside_the_lifecycle(tmp_path):
    journal = tmp_path / "j.jsonl"
    _journal_armed_setup(journal, state="triggered")
    summary = jd.judgment_pass("XAUUSD", market="forex",
                               journal_path=journal, now=BASE_TS + 300,
                               read=_read())
    assert summary["counts"] == {"judged": 1, "act": 1, "abstain": 0}
    records = trading_journal.read_events(limit=0, path=journal,
                                          event="judgment")
    setup_recs = [r for r in records
                  if (r.get("subject") or {}).get("type") == "setup"]
    read_recs = [r for r in records
                 if (r.get("subject") or {}).get("type") == "read"]
    assert len(setup_recs) == 1 and len(read_recs) == 1
    rec = setup_recs[0]
    # Beside the lifecycle: the record names the setup and the
    # state the lifecycle had put it in; judgment's verdict is its
    # own (shadow: it would have acted, and executed nothing).
    assert rec["subject"]["setup_id"] == "fake_h1:XAUUSD:k1"
    assert rec["subject"]["lifecycle_state"] == "triggered"
    assert rec["decision"] == "act" and rec["mode"] == "shadow"
    assert read_recs[0]["reason_category"] == "no_setup"
    # A second pass does not re-judge the same (setup, state).
    summary2 = jd.judgment_pass("XAUUSD", market="forex",
                                journal_path=journal,
                                now=BASE_TS + 600, read=_read())
    assert summary2["counts"]["judged"] == 0
    found = jd.recent_judgments("XAUUSD", journal_path=journal)
    assert len(found) == 3  # 1 setup + 2 read-level verdicts


def test_judgment_pass_skips_detected_and_terminal(tmp_path):
    journal = tmp_path / "j.jsonl"
    _journal_armed_setup(journal, state="detected")
    summary = jd.judgment_pass("XAUUSD", market="forex",
                               journal_path=journal, now=BASE_TS + 300,
                               read=_read())
    assert summary["counts"]["judged"] == 0


# ── surfaces ─────────────────────────────────────────────────────────

class FakeProviders:
    def ohlcv(self, symbol, market="crypto", timeframe="1h",
              bars=500, source="auto"):
        if timeframe == "1h":
            return hist_frame()
        return hist_frame(n=320, drift=0.002, end_ts=BASE_TS)

    def quote(self, symbol, market="crypto"):
        return {"symbol": symbol, "price": 2677.5,
                "change_pct_24h": 0.8, "currency": "USD",
                "source": "fake_quote", "quote_ts": BASE_TS,
                "quote_age_s": 12.0, "freshness": "fresh",
                "bid": 2677.4, "ask": 2677.6, "spread_bps": 0.7}


@pytest.fixture
def providers(monkeypatch):
    fake = FakeProviders()
    monkeypatch.setattr(market_data, "get_ohlcv", fake.ohlcv)
    monkeypatch.setattr(market_data, "quote", fake.quote)
    return fake


def test_render_surfaces(providers, tmp_path):
    journal = tmp_path / "j.jsonl"
    records = fc.make_forecasts("XAUUSD", market="forex",
                                journal_path=journal, now=BASE_TS)
    text = fc.render_forecasts(records, symbol="XAUUSD")
    assert "XAUUSD forecasts" in text and "point" in text
    assert "12h" in text and "15d" in text
    assert "uncalibrated (n=0)" in text
    verdict = jd.evaluate_setup(_read(), _setup())
    jtext = jd.render_judgments([verdict], symbol="XAUUSD")
    assert "fake_h1" in jtext and "invalidation" in jtext


def test_cmdline_forecast_judge_judgments(providers, tmp_path, capsys):
    from nomorals.cmdline.commands.trade import _cmd_trade
    settings = SimpleNamespace(home_path=tmp_path)
    context = SimpleNamespace(settings=settings)
    args = argparse.Namespace(trade_action="forecast",
                              symbol="XAUUSD", market="forex",
                              json=False)
    assert _cmd_trade(args, context) == 0
    assert "XAUUSD forecasts" in capsys.readouterr().out
    args = argparse.Namespace(trade_action="judge", symbol="XAUUSD",
                              market="forex", json=False)
    assert _cmd_trade(args, context) == 0
    assert "judgment pass XAUUSD" in capsys.readouterr().out
    args = argparse.Namespace(trade_action="judgments",
                              symbol="XAUUSD", market="forex",
                              json=False)
    assert _cmd_trade(args, context) == 0
    assert "no_setup" in capsys.readouterr().out


def test_trading_tool_forecast_judge_judgments(providers, tmp_path):
    from nomorals.tools import trading as trading_tool

    class FakeRegistry:
        def __init__(self, context):
            self.context = context
            self.tools = {}

        def register(self, name, **kwargs):
            def deco(fn):
                self.tools[name] = fn
                return fn
            return deco

    settings = SimpleNamespace(
        home_path=tmp_path,
        trading=SimpleNamespace(enabled=True))
    context = SimpleNamespace(settings=settings)
    reg = FakeRegistry(context)
    trading_tool.register(reg)
    out = reg.tools["trading"](action="forecast", symbol="XAUUSD",
                               market="forex")
    assert len(out["forecasts"]) == 5
    assert "XAUUSD forecasts" in out["rendered"]
    out = reg.tools["trading"](action="judge", symbol="XAUUSD",
                               market="forex")
    assert out["kind"] == "judgment_pass"
    assert "judgment pass" in out["rendered"]
    view = reg.tools["trading"](action="judgments", symbol="XAUUSD",
                                market="forex")
    assert view["judgments"] and "no_setup" in view["rendered"]
