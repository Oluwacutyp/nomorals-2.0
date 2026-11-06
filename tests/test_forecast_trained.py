"""Pass 6 of Devon's trading system: the trained forecast rung.

The estimator under test (agents/forecast_trained.py): a ridge
regressor per horizon over causal multi-timeframe features,
trained on Devon's own H1/H4/D1 bar history with the Pass 5
holdout discipline applied to the FITTING path (the newest 25% of
outcome times is locked out of every fit and fold), graded
walk-forward against the causal form of the empirical rung, and
journaled in the Pass 3 forecast shape so the audit grades both
rungs unchanged.

Every test here is a NEW-BEHAVIOR test: on pristine de02507 the
module does not exist and the file fails at import. Pass 3/5
behavior is carried by their own suites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nomorals.agents import forecast as fc
from nomorals.agents import forecast_trained as ft
from nomorals.agents import trade_audit as audit
from nomorals.finance import trading_journal

BASE = 1762362000.0  # 2025-11-05 17:00 UTC


# ── constructed histories ────────────────────────────────────────────

def _frame(closes, *, step=3600, end=BASE):
    rows, idx = [], []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c * 0.999
        rows.append((o, max(o, c) * 1.001, min(o, c) * 0.999, c, 100.0))
        idx.append(pd.Timestamp(end - (len(closes) - 1 - i) * step,
                                unit="s", tz="UTC"))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                       "volume"], index=idx)


def cycle_frame(*, n=2640, seed=3, end=BASE):
    """A slow persistent cycle: the forward move is (by
    construction) a strong function of the trailing state, so a
    causal regressor must beat the unconditional climatology."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    rets = 0.0025 * np.sin(2 * np.pi * t / 240.0) \
        + rng.normal(0, 0.0004, n)
    return _frame(2000 * np.exp(np.cumsum(rets)), end=end)


def noise_frame(*, n=2640, seed=3, end=BASE):
    """Pure noise: nothing causal predicts the forward move, so
    the trained rung must NOT come out ahead — the comparison is
    pinned to report that honestly."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.002, n)
    return _frame(2000 * np.exp(np.cumsum(rets)), end=end)


def _frames(frame):
    return {"1h": frame, "4h": None, "1d": None}


# ── samples + causality ──────────────────────────────────────────────

def test_samples_pair_features_with_forward_outcomes():
    frame = cycle_frame(n=600)
    samples = ft.build_samples(frame, None, None, 24)
    assert samples["X"].shape[1] == len(ft.FEATURE_NAMES)
    assert len(samples["ret_pct"]) > 400
    # outcome lands exactly horizon bars after the decision bar
    stamps = np.array([ts.timestamp() for ts in frame.index])
    base_i = list(stamps).index(samples["base_ts"][0])
    assert samples["outcome_ts"][0] == stamps[base_i + 24]
    # no warmup rows survive
    assert samples["base_ts"][0] >= stamps[ft.WARMUP_BARS]


def test_features_are_causal_future_bars_cannot_move_a_row():
    frame = cycle_frame(n=600)
    X1, _ = ft.build_features(frame)
    closes = frame["close"].to_numpy().copy()
    closes[-50:] *= 1.20  # violent future the early rows can't see
    X2, _ = ft.build_features(_frame(closes))
    assert np.allclose(X1[:-60], X2[:-60], equal_nan=True)


# ── training: fit, standing, holdout discipline ──────────────────────

def test_train_fits_and_beats_empirical_on_signal_history(tmp_path):
    journal = tmp_path / "j.jsonl"
    rec = ft.train_estimator("XAUUSD", "1d", frames=_frames(
        cycle_frame()), journal_path=journal, now=BASE)
    assert rec["state"] == "fitted", rec
    assert rec["fit_n"] >= ft.MIN_FIT_N
    # the locked holdout was excluded from the fit, by count
    assert rec["holdout_excluded_n"] > 0
    assert rec["fit_n"] + rec["holdout_excluded_n"] == rec["sample_n"]
    assert rec["holdout_cutoff_ts"] is not None
    oos = rec["oos"]
    assert oos["trained"]["median_abs_err_pct"] \
        < oos["empirical"]["median_abs_err_pct"]
    assert oos["trained"]["direction_hit_rate"] \
        > oos["empirical"]["direction_hit_rate"]
    # the model record is journaled and servable
    events = trading_journal.read_events(limit=0, path=journal,
                                         event="forecast_model")
    assert len(events) == 1
    model = ft.latest_model("XAUUSD", "1d", journal_path=journal,
                            now=BASE)
    assert model is not None and model["model"]["coef"]


def test_trained_rung_can_lose_and_the_record_says_so(tmp_path):
    rec = ft.train_estimator("XAUUSD", "1d", frames=_frames(
        noise_frame()), journal_path=tmp_path / "j.jsonl", now=BASE)
    assert rec["state"] == "fitted"
    oos = rec["oos"]
    # on pure noise the regressor earns nothing: it must not beat
    # the climatology by any real margin, and both standings are
    # on the record either way
    assert oos["trained"]["median_abs_err_pct"] \
        > 0.95 * oos["empirical"]["median_abs_err_pct"]
    assert oos["empirical"]["n"] == oos["trained"]["n"] > 0


def test_holdout_outcomes_cannot_move_the_fit(tmp_path):
    frame = cycle_frame()
    samples = ft.build_samples(frame, None, None, 24)
    cutoff, _fit_n, _hold_n = audit._split(
        [float(t) for t in samples["outcome_ts"]])
    assert cutoff is not None
    stamps = np.array([ts.timestamp() for ts in frame.index])
    j0 = int(np.searchsorted(stamps, cutoff))
    closes = frame["close"].to_numpy().copy()
    closes[j0:] *= 1.05  # rewrite history INSIDE the holdout only
    rec_a = ft.train_estimator(
        "XAUUSD", "1d", frames=_frames(frame),
        journal_path=tmp_path / "a.jsonl", now=BASE)
    rec_b = ft.train_estimator(
        "XAUUSD", "1d", frames=_frames(_frame(closes)),
        journal_path=tmp_path / "b.jsonl", now=BASE)
    assert rec_a["state"] == rec_b["state"] == "fitted"
    assert rec_a["sample_n"] == rec_b["sample_n"]
    assert rec_a["model"] == rec_b["model"]  # coefficients unmoved


def test_insufficient_history_journals_honest_refusal(tmp_path):
    journal = tmp_path / "j.jsonl"
    rec = ft.train_estimator("XAUUSD", "1d",
                             frames=_frames(cycle_frame(n=300)),
                             journal_path=journal, now=BASE)
    assert rec["state"] == "insufficient"
    assert rec["model"] is None and rec["reason"]
    assert ft.latest_model("XAUUSD", "1d", journal_path=journal,
                           now=BASE) is None


def test_stale_model_is_not_served(tmp_path):
    journal = tmp_path / "j.jsonl"
    ft.train_estimator("XAUUSD", "1d", frames=_frames(cycle_frame()),
                       journal_path=journal, now=BASE)
    assert ft.latest_model("XAUUSD", "1d", journal_path=journal,
                           now=BASE + ft.MODEL_MAX_AGE_S + 1) is None


# ── the live pass: shape, fallback, journal discipline ───────────────

def test_trained_pass_shape_and_per_horizon_fallback(tmp_path):
    journal = tmp_path / "j.jsonl"
    frame = cycle_frame()
    ft.train_estimator("XAUUSD", "1d", frames=_frames(frame),
                       journal_path=journal, now=BASE)
    records = ft.make_trained_forecasts(
        "XAUUSD", market="forex", frames=_frames(frame),
        journal_path=journal, now=BASE)
    assert [r["horizon"] for r in records] == list(fc.HORIZONS)
    by_tag = {r["horizon"]: r for r in records}
    trained = by_tag["1d"]
    assert trained["method"] == ft.METHOD
    assert trained["forecast_id"].startswith("fct:XAUUSD:")
    assert trained["model"]["fit_n"] >= ft.MIN_FIT_N
    for tag, rec in by_tag.items():
        if tag != "1d":  # no model fitted → empirical answers
            assert rec["method"] == fc.METHOD, tag
        # the full output standard, whichever rung served it
        assert rec["direction"] in ("UP", "DOWN", "NEUTRAL")
        assert rec["point_price"] is not None
        assert rec["interval"]["low"] < rec["point_price"] \
            < rec["interval"]["high"]
        assert rec["fav_excursion"] is not None
        assert rec["scenario"] and rec["invalidation"]
        # probabilities are the book's — an empty book earns none
        assert rec["probabilities"]["state"] == "uncalibrated"
    events = trading_journal.read_events(limit=0, path=journal,
                                         event="forecast")
    assert len(events) == 5
    # and Pass 3's own resolver grades them unchanged, once a
    # later frame covers the matured windows
    later = cycle_frame(end=BASE + 30 * 3600)
    resolved = fc.resolve_due_forecasts(
        "XAUUSD", market="forex", journal_path=journal,
        now=BASE + 30 * 3600, frame=later)
    assert resolved >= 1


# ── the rung comparison, in the audit's terms ────────────────────────

def _pair(journal, fid, *, method, created, graded, point,
          end, direction="UP", realized="up"):
    trading_journal.append_event(
        "forecast", path=journal, ts=created, forecast_id=fid,
        horizon="1d", symbol="XAUUSD", direction=direction,
        method=method, point_price=point,
        interval={"low": point * 0.95, "high": point * 1.05,
                  "coverage": 0.8},
        fav_excursion=30.0, adv_excursion=20.0, conviction_z=0.2,
        deadband_pct=0.1, created=created, base_price=2000.0)
    trading_journal.append_event(
        "forecast_outcome", path=journal, ts=graded,
        forecast_id=fid, horizon="1d", symbol="XAUUSD",
        realized_direction=realized,
        realized={"ret_pct": 1.0, "run_up": 25.0, "run_dn": 10.0,
                  "end_price": end},
        graded_at=graded)


def _comparison_journal(journal, *, trained_better: bool):
    for i in range(30):
        created = BASE + i * 3600
        graded = created + 1800
        if trained_better:
            _pair(journal, f"t{i}", method=ft.METHOD,
                  created=created, graded=graded,
                  point=2010.0, end=2012.0)
            _pair(journal, f"e{i}", method=fc.METHOD,
                  created=created, graded=graded,
                  point=1900.0, end=2012.0)
        else:
            _pair(journal, f"t{i}", method=ft.METHOD,
                  created=created, graded=graded,
                  point=1900.0, end=2012.0)
            _pair(journal, f"e{i}", method=fc.METHOD,
                  created=created, graded=graded,
                  point=2010.0, end=2012.0)


def test_rung_comparison_reports_trained_ahead_when_it_is(tmp_path):
    journal = tmp_path / "j.jsonl"
    _comparison_journal(journal, trained_better=True)
    comp = ft.rung_comparison("XAUUSD", journal_path=journal)
    entry = comp["horizons"]["1d"]
    assert entry["leader_holdout"] == "trained", entry
    assert entry["trained"]["holdout"]["median_abs_point_err_pct"] \
        < entry["empirical"]["holdout"]["median_abs_point_err_pct"]


def test_rung_comparison_reports_empirical_ahead_when_it_is(tmp_path):
    journal = tmp_path / "j.jsonl"
    _comparison_journal(journal, trained_better=False)
    comp = ft.rung_comparison("XAUUSD", journal_path=journal)
    entry = comp["horizons"]["1d"]
    assert entry["leader_holdout"] == "empirical", entry
    rendered = ft.render_comparison(comp)
    assert "empirical" in rendered and "trained" in rendered


def test_rung_comparison_empty_journal(tmp_path):
    comp = ft.rung_comparison("XAUUSD",
                              journal_path=tmp_path / "none.jsonl")
    assert comp["horizons"] == {}
