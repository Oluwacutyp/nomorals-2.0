"""Trained forecast estimator — Devon trading system, Pass 6.

The second rung of the forecast ladder (Pass 3 is the empirical
distribution). A ridge regressor per horizon over causal features
built from Devon's own multi-timeframe bar history (H1 features +
H4/D1 context aligned strictly as-of each H1 bar), fit
chronologically and graded walk-forward. The empirical rung stays
the fallback: a horizon with no fit, fresh model serves the
empirical block unchanged, and the two rungs are compared in the
Pass 5 audit's own terms (:func:`rung_comparison` reuses the
audit's pairing + stats, split fit/holdout the audit's way).

FITTING DISCIPLINE (the P5 contract, applied to training):

* **Chronology only.** Features at bar ``t`` use bars ``<= t``
  (higher-timeframe context is the last bar closed at or before
  ``t``); a sample's target is the forward outcome over
  ``t+1..t+h`` — the same window :func:`forecast.forward_outcomes`
  defines, so both rungs predict the identical quantity.
* **The locked holdout stays locked.** The newest
  ``trade_audit.HOLDOUT_FRACTION`` of sample OUTCOME times is
  excluded from every fit and every walk-forward fold — the
  training path is structurally blind to it, exactly as the
  audit's fit slice is. A test proves holdout outcomes cannot move
  the fitted coefficients.
* **Walk-forward standing, honestly reported.** On the fit slice,
  expanding-window folds produce out-of-sample predictions for the
  trained rung AND for the causal climatology form of the
  empirical rung (the mean of forward outcomes fully realized by
  the decision bar). Both standings are journaled on the model
  record; whichever rung loses, the record says so.

TRAINING PROVENANCE: Devon's own bar history at the market-data
seam. The GoldenSwing seed seam exists for the empirical sample
(:func:`forecast.load_seed_samples`) but its rows carry realized
outcomes only — no feature vectors — so a regressor cannot train
on them; the trained rung is Devon-history only, by construction,
not by omission.

Records: fitted models journal as ``forecast_model`` events (the
coefficients are small; the journal is the store — no new file,
no migration). Forecasts journal in the IDENTICAL ``forecast``
event shape as Pass 3, with ``method`` stamped
``trained-ridge-v1`` and ``fct:`` ids, so outcome resolution, the
calibration book and the Pass 5 audit grade them unchanged.
Probabilities come from the same calibration book — earned only,
per the settled forecast doctrine.

Placement: ``agents/`` beside forecast/trade_audit — the layer
from which the journal and the market-data seam are legal imports.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ..finance import trading_journal
from ..integrations import market_data
from . import forecast as fc
from . import trade_audit as audit_mod
from .playbooks import ts_epoch

_log = logging.getLogger(__name__)

#: Estimator identity, stamped on every trained record.
METHOD = "trained-ridge-v1"
MODEL_VERSION = 1
#: Bars requested per timeframe when training / forecasting.
TRAIN_BARS: dict[str, int] = {"1h": 3000, "4h": 1500, "1d": 750}
#: A horizon's model is served only with at least this many fit
#: samples (post-holdout-exclusion) — below it the empirical rung
#: answers and the training record says why.
MIN_FIT_N = 240
#: A fitted model older than this is stale; the empirical rung
#: answers until the audit cycle refits.
MODEL_MAX_AGE_S = 7 * 24 * 3600
#: Expanding-window walk-forward folds over the fit slice
#: (the P5 audit's window count).
WALK_FOLDS = audit_mod.WALK_WINDOWS
#: Ridge penalty (features are standardized before the solve).
RIDGE_LAMBDA = 1.0
#: Bars of H1 history a feature row needs behind it.
WARMUP_BARS = 80

FEATURE_NAMES: tuple[str, ...] = (
    "ret_1", "ret_4", "ret_12", "ret_24", "ret_72",
    "vol_24", "atr_pct_14", "rsi_14",
    "dist_ma_24", "dist_ma_72",
    "h4_ret_6", "d1_ret_5", "hour_sin", "hour_cos")


# ── causal features ──────────────────────────────────────────────────

def _frame_arrays(frame: Any) -> tuple[np.ndarray, np.ndarray]:
    """(close array, epoch stamps) for one OHLCV frame."""
    close = frame["close"].to_numpy(dtype=float)
    stamps = np.array([ts_epoch(ts) for ts in frame.index],
                      dtype=float)
    return close, stamps


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) >= window:
        csum = np.concatenate([[0.0], np.cumsum(values)])
        out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def _asof_context(stamps: np.ndarray, ctx_close: np.ndarray,
                  ctx_stamps: np.ndarray, lookback: int
                  ) -> np.ndarray:
    """Per-bar % return of the context frame over ``lookback`` of
    ITS bars, using only context bars closed at or before each
    bar's stamp (searchsorted right - 1). NaN where the context
    does not reach back far enough."""
    out = np.full(len(stamps), np.nan)
    if len(ctx_close) == 0:
        return out
    j = np.searchsorted(ctx_stamps, stamps, side="right") - 1
    ok = (j >= lookback) & (j < len(ctx_close))
    if not ok.any():
        return out
    base = ctx_close[j[ok] - lookback]
    now = ctx_close[j[ok]]
    good = np.isfinite(base) & np.isfinite(now) & (base != 0)
    vals = np.full(int(ok.sum()), np.nan)
    vals[good] = (now[good] / base[good] - 1.0) * 100.0
    out[ok] = vals
    return out


def build_features(frame_h1: Any, frame_h4: Any = None,
                   frame_d1: Any = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Causal feature matrix for an H1 frame: (X, stamps).

    Row ``i`` describes bar ``i`` from bars ``<= i`` only. Rows
    that cannot be fully formed (warmup, missing context) carry
    NaN — sample assembly drops them, never fills them.
    """
    close, stamps = _frame_arrays(frame_h1)
    high = frame_h1["high"].to_numpy(dtype=float)
    low = frame_h1["low"].to_numpy(dtype=float)
    n = len(close)
    logret = np.full(n, np.nan)
    logret[1:] = np.log(close[1:] / close[:-1]) * 100.0

    def ret(k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if n > k:
            out[k:] = (close[k:] / close[:-k] - 1.0) * 100.0
        return out

    # realized vol: stdev of 1h log returns over the last 24 bars
    vol = np.full(n, np.nan)
    sq = np.where(np.isfinite(logret), logret, 0.0) ** 2
    mean = _rolling_mean(np.where(np.isfinite(logret), logret, 0.0), 24)
    mean_sq = _rolling_mean(sq, 24)
    var = mean_sq - mean ** 2
    vol = np.sqrt(np.maximum(var, 0.0))
    vol[:24] = np.nan
    # ATR(14) as % of close (true range, simple mean — deterministic)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    atr_pct = _rolling_mean(tr, 14) / close * 100.0
    # RSI(14): simple-mean gains/losses over the last 14 changes
    delta = np.concatenate([[0.0], np.diff(close)])
    gains = _rolling_mean(np.maximum(delta, 0.0), 14)
    losses = _rolling_mean(np.maximum(-delta, 0.0), 14)
    rs = np.where(losses > 0, gains / np.where(losses > 0, losses, 1.0),
                  np.inf)
    rsi = np.where(losses > 0, 100.0 - 100.0 / (1.0 + rs),
                   np.where(gains > 0, 100.0, 50.0))
    rsi[:14] = np.nan
    dist24 = (close / _rolling_mean(close, 24) - 1.0) * 100.0
    dist72 = (close / _rolling_mean(close, 72) - 1.0) * 100.0
    if frame_h4 is not None and len(frame_h4):
        h4_close, h4_stamps = _frame_arrays(frame_h4)
        h4_ret = _asof_context(stamps, h4_close, h4_stamps, 6)
    else:
        h4_ret = np.zeros(n)  # no context supplied: neutral column
    if frame_d1 is not None and len(frame_d1):
        d1_close, d1_stamps = _frame_arrays(frame_d1)
        d1_ret = _asof_context(stamps, d1_close, d1_stamps, 5)
    else:
        d1_ret = np.zeros(n)
    hour = (stamps % 86400.0) // 3600.0
    cols = [ret(1), ret(4), ret(12), ret(24), ret(72), vol, atr_pct,
            rsi, dist24, dist72, h4_ret, d1_ret,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0)]
    return np.column_stack(cols), stamps


# ── samples: features paired with forward outcomes ───────────────────

def build_samples(frame_h1: Any, frame_h4: Any, frame_d1: Any,
                  horizon_bars: int) -> dict[str, Any]:
    """Feature rows paired with their forward outcomes.

    Row ``i`` = bar ``i``'s features; the target is bar ``i``'s
    forward outcome from :func:`forecast.forward_outcomes` (the
    identical window definition the empirical rung samples), and
    ``outcome_ts`` is the stamp of the bar the window ENDS on —
    the time the outcome became knowable, which is what the
    holdout split and the walk-forward folds order by.
    """
    empty = {"X": np.zeros((0, len(FEATURE_NAMES))),
             "ret_pct": np.zeros(0), "run_up": np.zeros(0),
             "run_dn": np.zeros(0), "base_ts": np.zeros(0),
             "outcome_ts": np.zeros(0)}
    if frame_h1 is None or len(frame_h1) <= horizon_bars:
        return empty
    X, stamps = build_features(frame_h1, frame_h4, frame_d1)
    outcomes = fc.forward_outcomes(frame_h1, horizon_bars)
    m = len(outcomes["ret_pct"])
    if m == 0:
        return empty
    X = X[:m]
    base_ts = stamps[:m]
    outcome_ts = stamps[horizon_bars:horizon_bars + m]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(outcomes["ret_pct"])
    ok[:WARMUP_BARS] = False
    return {"X": X[ok], "ret_pct": outcomes["ret_pct"][ok],
            "run_up": outcomes["run_up"][ok],
            "run_dn": outcomes["run_dn"][ok],
            "base_ts": base_ts[ok], "outcome_ts": outcome_ts[ok]}


# ── the ridge fit ────────────────────────────────────────────────────

def _fit_ridge(X: np.ndarray, y: np.ndarray,
               lam: float = RIDGE_LAMBDA) -> dict[str, Any]:
    """Closed-form ridge on standardized features (intercept via
    centering, unpenalized). Pure function of (X, y) — the
    holdout-blindness tests lean on exactly that."""
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std = np.where(x_std > 0, x_std, 1.0)
    y_mean = float(y.mean())
    Xs = (X - x_mean) / x_std
    gram = Xs.T @ Xs + lam * np.eye(X.shape[1])
    coef = np.linalg.solve(gram, Xs.T @ (y - y_mean))
    return {"coef": coef, "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean}


def _predict(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    Xs = (X - model["x_mean"]) / model["x_std"]
    return Xs @ model["coef"] + model["y_mean"]


def _oos_metrics(pred: np.ndarray, y: np.ndarray,
                 deadband: float) -> dict[str, Any]:
    """Walk-forward standing of one rung on identical samples."""
    if len(y) == 0:
        return {"n": 0, "median_abs_err_pct": None,
                "direction_hit_rate": None}
    hits = 0
    for p, realized in zip(pred, y, strict=True):
        call = "up" if p > deadband else "dn" if p < -deadband \
            else "neutral"
        hits += 1 if call == fc.realized_direction(
            float(realized), deadband) else 0
    return {"n": int(len(y)),
            "median_abs_err_pct": round(
                float(np.median(np.abs(pred - y))), 4),
            "direction_hit_rate": round(hits / len(y), 4)}


def _walk_forward(samples: dict[str, Any], fit_mask: np.ndarray,
                  horizon_bars: int) -> dict[str, Any]:
    """Expanding-window OOS predictions over the FIT slice only.

    Returns per-sample out-of-sample predictions for the trained
    rung and for the causal empirical climatology (the mean of
    forward outcomes fully realized by each decision bar), plus
    both standings. Holdout samples never enter: the mask is
    applied before any fold is cut.
    """
    X = samples["X"][fit_mask]
    y = samples["ret_pct"][fit_mask]
    base_ts = samples["base_ts"][fit_mask]
    outcome_ts = samples["outcome_ts"][fit_mask]
    n = len(y)
    out: dict[str, Any] = {"n": 0, "trained": None, "empirical": None,
                           "pred_trained": None, "pred_empirical": None,
                           "y": None}
    if n < WALK_FOLDS + 1:
        return out
    deadband = fc.DEADBAND_SIGMA * float(np.std(y)) if n else 0.0
    chunks = np.array_split(np.arange(n), WALK_FOLDS + 1)
    pred_t = np.full(n, np.nan)
    pred_e = np.full(n, np.nan)
    # Causal climatology via cumulative sums over ALL samples in
    # chronological order: at decision bar i, the realized set is
    # every sample whose outcome landed at or before base_ts[i].
    csum = np.concatenate([[0.0], np.cumsum(y)])
    for fold in range(1, WALK_FOLDS + 1):
        train_idx = np.concatenate(chunks[:fold])
        test_idx = chunks[fold]
        if len(train_idx) < 20 or len(test_idx) == 0:
            continue
        model = _fit_ridge(X[train_idx], y[train_idx])
        pred_t[test_idx] = _predict(model, X[test_idx])
        for i in test_idx:
            realized_n = int(np.searchsorted(
                outcome_ts, base_ts[i], side="right"))
            if realized_n >= fc.MIN_SAMPLE_N:
                pred_e[i] = csum[realized_n] / realized_n
    both = np.isfinite(pred_t) & np.isfinite(pred_e)
    if not both.any():
        return out
    out["n"] = int(both.sum())
    out["trained"] = _oos_metrics(pred_t[both], y[both], deadband)
    out["empirical"] = _oos_metrics(pred_e[both], y[both], deadband)
    out["pred_trained"] = pred_t
    out["pred_empirical"] = pred_e
    out["y"] = y
    return out


# ── training: fit, grade, journal ────────────────────────────────────

def train_estimator(symbol: str, horizon_tag: str, *,
                    frames: dict[str, Any] | None = None,
                    market: str = "crypto",
                    settings: Any = None, journal_path: Any = None,
                    now: float | None = None, journal: bool = True
                    ) -> dict[str, Any]:
    """Fit one horizon's model; journal the ``forecast_model``.

    The record carries the fit window, the holdout cutoff that was
    excluded, both rungs' walk-forward standings, and — when the
    fit is earned — the coefficients the live estimate serves.
    Insufficient data journals an honest ``insufficient`` record
    and serves nothing.
    """
    taken = float(now if now is not None else time.time())
    sym = (symbol or "").strip().upper()
    horizon_bars = fc.HORIZONS[horizon_tag]
    if frames is None:
        frames = _fetch_frames(sym, settings=settings, market=market)
    samples = build_samples(frames.get("1h"), frames.get("4h"),
                            frames.get("1d"), horizon_bars)
    n = int(len(samples["ret_pct"]))
    record: dict[str, Any] = {
        "kind": "forecast_model", "method": METHOD,
        "version": MODEL_VERSION, "symbol": sym,
        "horizon": horizon_tag, "horizon_bars": horizon_bars,
        "trained_at": taken, "sample_n": n, "fit_n": 0,
        "holdout_cutoff_ts": None, "holdout_excluded_n": 0,
        "state": "insufficient", "reason": None, "model": None,
        "oos": None}
    if n == 0:
        record["reason"] = "no usable samples (history too thin)"
        if journal:
            trading_journal.append_event(
                "forecast_model", settings=settings,
                path=journal_path, **record)
        return record
    cutoff, _fit_n, _hold_n = audit_mod._split(
        [float(t) for t in samples["outcome_ts"]])
    fit_mask = np.ones(n, dtype=bool) if cutoff is None \
        else samples["outcome_ts"] < cutoff
    fit_n = int(fit_mask.sum())
    record["fit_n"] = fit_n
    record["holdout_cutoff_ts"] = cutoff
    record["holdout_excluded_n"] = n - fit_n
    if fit_n < MIN_FIT_N:
        record["reason"] = (
            f"fit samples {fit_n} < {MIN_FIT_N} floor "
            f"(holdout excluded {n - fit_n})")
        if journal:
            trading_journal.append_event(
                "forecast_model", settings=settings,
                path=journal_path, **record)
        return record
    X = samples["X"][fit_mask]
    y = samples["ret_pct"][fit_mask]
    oos = _walk_forward(samples, fit_mask, horizon_bars)
    model = _fit_ridge(X, y)
    resid = None
    if oos["pred_trained"] is not None:
        r = oos["y"] - oos["pred_trained"]
        r = r[np.isfinite(r)]
        if len(r):
            resid = r
    if resid is None:
        resid = y - _predict(model, X)
    lo_q = (1.0 - fc.INTERVAL_COVERAGE) / 2.0
    q_lo, q_hi = (float(v) for v in np.quantile(
        resid, [lo_q, 1.0 - lo_q]))
    record["state"] = "fitted"
    record["model"] = {
        "coef": [round(float(c), 6) for c in model["coef"]],
        "x_mean": [round(float(v), 6) for v in model["x_mean"]],
        "x_std": [round(float(v), 6) for v in model["x_std"]],
        "y_mean": round(float(model["y_mean"]), 6),
        "features": list(FEATURE_NAMES),
        "resid_q_lo": round(q_lo, 4), "resid_q_hi": round(q_hi, 4),
        "y_std": round(float(np.std(y)), 4),
        "fit_from_ts": float(samples["base_ts"][fit_mask][0]),
        "fit_to_ts": float(samples["outcome_ts"][fit_mask][-1])}
    record["oos"] = {"n": oos["n"], "trained": oos["trained"],
                     "empirical": oos["empirical"]}
    if journal:
        trading_journal.append_event(
            "forecast_model", settings=settings, path=journal_path,
            **record)
    return record


def _fetch_frames(symbol: str, *, settings: Any = None,
                  market: str = "crypto") -> dict[str, Any]:
    frames: dict[str, Any] = {}
    for tf, bars in TRAIN_BARS.items():
        try:
            frames[tf] = market_data.get_ohlcv(
                symbol, market=market, timeframe=tf, bars=bars)
        except Exception as exc:  # noqa: BLE001 - marked, not fatal
            _log.info("forecast_trained: %s history for %s failed: %s",
                      tf, symbol, exc)
            frames[tf] = None
    return frames


def train_all(symbol: str, market: str = "crypto", *,
              settings: Any = None, journal_path: Any = None,
              now: float | None = None,
              frames: dict[str, Any] | None = None
              ) -> list[dict[str, Any]]:
    """Fit every horizon for one symbol; journal a run summary."""
    taken = float(now if now is not None else time.time())
    sym = (symbol or "").strip().upper()
    if frames is None:
        frames = _fetch_frames(sym, settings=settings, market=market)
    records = [train_estimator(sym, tag, frames=frames,
                               settings=settings,
                               journal_path=journal_path, now=taken)
               for tag in fc.HORIZONS]
    trading_journal.append_event(
        "forecast_training", settings=settings, path=journal_path,
        symbol=sym, trained_at=taken,
        horizons={r["horizon"]: {
            "state": r["state"], "fit_n": r["fit_n"],
            "oos_trained": (r.get("oos") or {}).get("trained"),
            "oos_empirical": (r.get("oos") or {}).get("empirical")}
            for r in records})
    return records


def latest_model(symbol: str, horizon_tag: str, *,
                 settings: Any = None, journal_path: Any = None,
                 now: float | None = None,
                 max_age_s: float = MODEL_MAX_AGE_S
                 ) -> dict[str, Any] | None:
    """The newest FITTED model record for (symbol, horizon), or
    None when none exists or the newest fit is stale — the caller
    then serves the empirical rung."""
    taken = float(now if now is not None else time.time())
    sym = (symbol or "").strip().upper()
    for rec in trading_journal.read_events(
            limit=0, event="forecast_model", settings=settings,
            path=journal_path):
        if rec.get("symbol") != sym \
                or rec.get("horizon") != horizon_tag:
            continue
        if rec.get("state") != "fitted" or not rec.get("model"):
            return None  # newest word on this horizon: not fitted
        if taken - float(rec.get("trained_at") or 0) > max_age_s:
            return None
        return rec
    return None


# ── the live estimate ────────────────────────────────────────────────

def trained_estimate(frame_h1: Any, frame_h4: Any, frame_d1: Any,
                     horizon_tag: str, model_record: dict[str, Any],
                     *, base_price: float, base_ts: float,
                     created: float) -> dict[str, Any]:
    """One horizon's forecast block from a fitted model.

    The EXACT output-standard shape of
    :func:`forecast.estimate_horizon` — direction / point /
    interval / excursions / disagreement / scenario / invalidation
    / timestamps — with ``method`` stamped trained. Excursions are
    the empirical sample's direction-conditioned means (a
    distributional quantity, not a regressed one); disagreement is
    |trained prediction − empirical mean|, the cross-rung spread.
    Probabilities are the caller's to stamp from the book.
    """
    horizon_bars = fc.HORIZONS[horizon_tag]
    model = model_record["model"]
    params = {"coef": np.array(model["coef"], dtype=float),
              "x_mean": np.array(model["x_mean"], dtype=float),
              "x_std": np.array(model["x_std"], dtype=float),
              "y_mean": float(model["y_mean"])}
    X, _stamps = build_features(frame_h1, frame_h4, frame_d1)
    pred = float(_predict(params, X[-1:])[0])
    outcomes = fc.forward_outcomes(frame_h1, horizon_bars)
    ret = outcomes["ret_pct"]
    emp_mean = float(np.mean(ret)) if len(ret) else 0.0
    deadband = fc.DEADBAND_SIGMA * float(model["y_std"])
    if pred > deadband:
        direction = "UP"
    elif pred < -deadband:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"
    eff_up = direction == "UP" or (direction == "NEUTRAL"
                                   and pred >= 0)
    point = base_price * (1.0 + pred / 100.0)
    lo_pct = pred + float(model["resid_q_lo"])
    hi_pct = pred + float(model["resid_q_hi"])
    fav_src = outcomes["run_up"] if eff_up else outcomes["run_dn"]
    adv_src = outcomes["run_dn"] if eff_up else outcomes["run_up"]
    fav = float(np.mean(fav_src)) if len(fav_src) else 0.0
    adv = float(np.mean(adv_src)) if len(adv_src) else 0.0
    inv_pct = lo_pct if eff_up else hi_pct
    inv_price = base_price * (1.0 + inv_pct / 100.0)
    side = "up" if eff_up else "down"
    oos = (model_record.get("oos") or {})
    oos_t = (oos.get("trained") or {}).get("median_abs_err_pct")
    oos_e = (oos.get("empirical") or {}).get("median_abs_err_pct")
    return {
        "horizon": horizon_tag, "horizon_bars": horizon_bars,
        "direction": direction, "point_price": fc._r(point),
        "interval": {
            "low": fc._r(base_price * (1.0 + lo_pct / 100.0)),
            "high": fc._r(base_price * (1.0 + hi_pct / 100.0)),
            "coverage": fc.INTERVAL_COVERAGE},
        "expected_move_pct": fc._r(pred),
        "expected_move_price": fc._r(point - base_price),
        "mean_abs_move_pct": fc._r(
            float(np.mean(np.abs(ret))) if len(ret) else None),
        "fav_excursion": fc._r(fav), "adv_excursion": fc._r(adv),
        "disagreement_pct": fc._r(abs(pred - emp_mean)),
        "conviction_z": fc._r(abs(pred) / float(model["y_std"])
                              if model["y_std"] else 0.0),
        "scenario": (
            f"{horizon_tag} trained path: {direction} drift toward "
            f"{point:,.2f} inside the {fc.INTERVAL_COVERAGE:.0%} "
            f"band {base_price * (1.0 + lo_pct / 100.0):,.2f}–"
            f"{base_price * (1.0 + hi_pct / 100.0):,.2f} "
            f"(ridge over {len(FEATURE_NAMES)} causal features; "
            f"walk-forward median |err| trained {oos_t}% vs "
            f"empirical {oos_e}% on the fit slice); expected "
            f"favourable excursion ~{fav:,.2f}, adverse ~{adv:,.2f}"),
        "invalidation_price": fc._r(inv_price),
        "invalidation": (
            f"a close beyond {inv_price:,.2f} (the band's adverse "
            f"edge) falsifies the {side} trained path"),
        "probabilities": {"state": "uncalibrated", "n": 0,
                          "up": None, "dn": None, "neutral": None},
        "method": METHOD, "version": fc.FORECAST_VERSION,
        "n": int(model_record.get("fit_n") or 0),
        "frame_n": int(len(ret)), "seed_n": 0,
        "n_effective": int((model_record.get("fit_n") or 0)
                           // horizon_bars),
        "deadband_pct": fc._r(deadband), "unavailable_reason": None,
        "created": created, "created_utc": fc._iso_utc(created),
        "cutoff": base_ts, "cutoff_utc": fc._iso_utc(base_ts),
        "expiry": created + horizon_bars * 3600,
        "expiry_utc": fc._iso_utc(created + horizon_bars * 3600),
        "base_price": base_price,
        "model": {"trained_at": model_record.get("trained_at"),
                  "fit_n": model_record.get("fit_n"),
                  "oos_trained": oos.get("trained"),
                  "oos_empirical": oos.get("empirical")}}


def make_trained_forecasts(symbol: str, market: str = "crypto", *,
                           settings: Any = None,
                           journal_path: Any = None,
                           now: float | None = None,
                           frames: dict[str, Any] | None = None,
                           journal: bool = True
                           ) -> list[dict[str, Any]]:
    """One forecast pass on the trained rung.

    Horizons with a fresh fitted model serve the trained block;
    the rest fall back to the empirical estimator inside the same
    pass (method stamped per record). Records journal in the
    Pass 3 ``forecast`` shape, so resolution, calibration and the
    audit treat both rungs identically.
    """
    taken = float(now if now is not None else time.time())
    sym = (symbol or "").strip().upper()
    if frames is None:
        frames = _fetch_frames(sym, settings=settings, market=market)
    frame = frames.get("1h")
    book = fc.calibration_book(symbol=sym, settings=settings,
                               journal_path=journal_path)
    records: list[dict[str, Any]] = []
    if frame is None or len(frame) == 0:
        base_price, base_ts, frame_empty = 0.0, taken, True
    else:
        base_price = float(frame["close"].iloc[-1])
        base_ts = ts_epoch(frame.index[-1])
        frame_empty = False
    for tag in fc.HORIZONS:
        model_rec = None if frame_empty else latest_model(
            sym, tag, settings=settings, journal_path=journal_path,
            now=taken)
        if frame_empty:
            block = fc.estimate_horizon(
                None, tag, base_price=base_price, base_ts=base_ts,
                created=taken)
            block["unavailable_reason"] = "no H1 history available"
        elif model_rec is not None:
            block = trained_estimate(
                frame, frames.get("4h"), frames.get("1d"), tag,
                model_rec, base_price=base_price, base_ts=base_ts,
                created=taken)
        else:
            block = fc.estimate_horizon(
                frame, tag, base_price=base_price, base_ts=base_ts,
                created=taken)
        block["probabilities"] = fc.probabilities_from_book(book, tag)
        trained = block.get("method") == METHOD
        record: dict[str, Any] = {
            "kind": "forecast", "version": fc.FORECAST_VERSION,
            "method": block.get("method") or fc.METHOD,
            "forecast_id": (f"fct:{sym}:{taken:.0f}:{tag}" if trained
                            else f"fc:{sym}:{taken:.0f}:{tag}"),
            "symbol": sym, "market": market,
            "created": taken, "created_utc": fc._iso_utc(taken),
            **block}
        record["horizon"] = tag
        if journal:
            trading_journal.append_event(
                "forecast", settings=settings, path=journal_path,
                **record)
        records.append(record)
    return records


# ── the rung comparison, in the audit's own terms ────────────────────

def rung_comparison(symbol: str | None = None, *,
                    settings: Any = None, journal_path: Any = None
                    ) -> dict[str, Any]:
    """Trained vs empirical, graded exactly as Pass 5 grades.

    Pairs are the audit's own (:func:`trade_audit._forecast_pairs`),
    split fit/holdout by the audit's rule per (horizon, method),
    and statted by the audit's :func:`trade_audit.forecast_stats`.
    The leader per horizon is read off HOLDOUT median absolute
    point error (lower wins; a tie or thin slice reads ``even`` /
    ``insufficient_data``) — reported either way, never rigged.
    """
    sym = symbol.strip().upper() if symbol else None
    records = trading_journal.read_events(
        limit=0, settings=settings, path=journal_path)
    records = list(reversed(records))  # chronological

    def pairs_for(method: str) -> list[dict[str, Any]]:
        # The audit pairing folds outcomes against the forecasts
        # present in the record set — restricting the set to one
        # method's forecasts grades that rung alone, in the
        # audit's identical terms.
        subset = [r for r in records
                  if r.get("event") == "forecast_outcome"
                  or (r.get("event") == "forecast"
                      and (r.get("method") or fc.METHOD) == method)]
        return audit_mod._forecast_pairs(subset, sym)

    by_method = {fc.METHOD: pairs_for(fc.METHOD),
                 METHOD: pairs_for(METHOD)}
    horizons: dict[str, Any] = {}
    all_tags = {p["horizon"] for ps in by_method.values()
                for p in ps}
    for tag in sorted(all_tags):
        entry: dict[str, Any] = {}
        for method, label in ((fc.METHOD, "empirical"),
                              (METHOD, "trained")):
            sub = [p for p in by_method[method]
                   if p["horizon"] == tag]
            cutoff, _f, _h = audit_mod._split(
                [p["outcome_ts"] for p in sub])
            entry[label] = {
                "n": len(sub),
                "fit": audit_mod.forecast_stats(
                    audit_mod._slice(sub, cutoff, holdout=False)),
                "holdout": audit_mod.forecast_stats(
                    audit_mod._slice(sub, cutoff, holdout=True))}
        emp_err = entry["empirical"]["holdout"][
            "median_abs_point_err_pct"]
        tr_err = entry["trained"]["holdout"][
            "median_abs_point_err_pct"]
        if emp_err is None or tr_err is None:
            leader = "insufficient_data"
        elif abs(float(tr_err) - float(emp_err)) < 1e-9:
            leader = "even"
        else:
            leader = "trained" if tr_err < emp_err else "empirical"
        entry["leader_holdout"] = leader
        horizons[tag] = entry
    return {"kind": "forecast_rung_comparison", "symbol": sym,
            "horizons": horizons}


def render_comparison(comp: dict[str, Any]) -> str:
    """Human rendering of one rung comparison."""
    lines = [f"forecast rungs — {comp.get('symbol') or 'all symbols'}"
             " (graded in the audit's terms; holdout leads)"]
    if not comp["horizons"]:
        lines.append("  no graded forecast pairs yet — neither rung "
                     "has a standing")
    for tag, entry in comp["horizons"].items():
        parts = []
        for label in ("empirical", "trained"):
            hold = entry[label]["holdout"]
            err = hold["median_abs_point_err_pct"]
            parts.append(
                f"{label}: holdout n={hold['n']} hit "
                f"{hold['hit_rate']} median |err| "
                f"{err if err is not None else '?'}%")
        lines.append(f"  {tag:<4} " + " · ".join(parts)
                     + f" → {entry['leader_holdout']}")
    return "\n".join(lines)
