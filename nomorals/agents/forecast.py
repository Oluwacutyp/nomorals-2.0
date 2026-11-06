"""Forecast organ — Devon trading system, Pass 3.

Structured per-horizon forecasts to the owner's 2026-11-03 output
standard: percentile ranks alone are NOT a forecast, so every
horizon (12h / 1d / 3d / 7d / 15d) carries a direction
(UP / DOWN / NEUTRAL against a stated deadband), a POINT PRICE, an
interval with its nominal coverage stated, the expected move, the
expected favourable / adverse excursions, model disagreement, a
scenario and its invalidation, and created / cutoff / expiry /
version fields. Probabilities appear ONLY when live calibration
has earned them (empirical frequencies over graded forecasts,
``n`` stated); below that the block reads ``uncalibrated`` with
the count that failed — never a dressed-up number.

Estimation is the F1 shape — empirical distributions, no trained
regressor this pass: for each horizon, the forward outcomes of
every bar in the H1 history (endpoint return, max favourable run,
max adverse run over the next ``horizon`` bars) form the sample;
the point is the sample mean, the interval its central quantiles
at the stated coverage. Two sample sources feed the same
estimator: Devon's own H1 bar history at the market-data seam,
and seed outcome samples in the identical shape imported from
GoldenSwing journal exports (owner decision 8: Devon history +
GS seed — data, not code; the products stay unmerged). Sample
provenance is named on every record.

Outcomes attach later and append-only: a forecast record is never
rewritten. :func:`attach_outcome` / :func:`resolve_due_forecasts`
append ``forecast_outcome`` events keyed by (forecast_id, horizon);
the calibration book folds forecasts + outcomes into per-horizon
hit rates, and the E3 trust rule (GS contract shape: pool of
high-conviction forecasts, n >= 20, lift >= 1.25 trusted,
lift < 0.90 anti-calibrated) decides when judgment may weigh the
forecast group at all. Below trust the group is recorded but
weightless — see agents/judgment.py.

Placement: ``agents/`` beside market_read/playbooks — the layer
from which the TA library, the finance journal and the
market-data integrations are all legal static imports.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..finance import trading_journal
from ..integrations import market_data
from .playbooks import ts_epoch

_log = logging.getLogger(__name__)

#: Horizons: tag -> length in H1 bars (the GS horizon vocabulary).
HORIZONS: dict[str, int] = {"12h": 12, "1d": 24, "3d": 72,
                            "7d": 168, "15d": 360}
#: Candidate entry clock -> the forecast horizon judgment reads
#: (GS E3 HORIZON_BY_TF shape, in Devon's tag vocabulary).
HORIZON_BY_TF: dict[str, str] = {"5m": "12h", "15m": "12h",
                                 "30m": "1d", "1h": "1d",
                                 "4h": "3d", "1d": "3d"}
DEFAULT_HORIZON_TAG = "1d"

#: Estimator identity, stamped on every record.
METHOD = "empirical-h1-v1"
FORECAST_VERSION = 1
#: H1 bars of history requested at the provider seam.
HISTORY_BARS = 3000
#: Nominal coverage of the prediction interval (central quantiles).
INTERVAL_COVERAGE = 0.80
#: Direction deadband: |mean forward return| within this fraction
#: of the sample's return stdev is NEUTRAL, not a call.
DEADBAND_SIGMA = 0.15
#: Floors: below this many sample windows a horizon is honestly
#: unavailable; below this many GRADED forecasts probabilities
#: stay uncalibrated (the E3 trust precedent).
MIN_SAMPLE_N = 50
MIN_PROB_N = 20
#: E3 trust rule (GS forecast_evidence contract): the weighed
#: pool is the high-conviction forecasts (conviction z at or above
#: CONVICTION_Z); trust needs TRUST_MIN_N pooled resolutions at
#: TRUST_LIFT x the horizon's base rate; below ANTI_LIFT the feed
#: is anti-calibrated and judgment records it as such.
TRUST_MIN_N = 20
TRUST_LIFT = 1.25
ANTI_LIFT = 0.90
CONVICTION_Z = 0.5


def horizon_for_tf(timeframe: str | None) -> str:
    return HORIZON_BY_TF.get(str(timeframe or "").strip().lower(),
                             DEFAULT_HORIZON_TAG)


def _iso_utc(taken: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(taken))


def _r(value: float | None, nd: int = 4) -> float | None:
    return None if value is None else round(float(value), nd)


# ── the empirical sample ─────────────────────────────────────────────

def forward_outcomes(frame: Any, horizon_bars: int) -> dict[str, Any]:
    """Realized forward outcomes for a forecast made at each bar.

    For bar ``t`` (full window required): ``ret_pct`` = endpoint
    return % over bars t+1..t+horizon; ``run_up`` = max high in
    the window - close[t]; ``run_dn`` = close[t] - min low in the
    window (both in price units). This is the horizon's empirical
    distribution — the estimator's whole input.
    """
    empty = {"ret_pct": np.zeros(0), "run_up": np.zeros(0),
             "run_dn": np.zeros(0)}
    if frame is None or horizon_bars <= 0 \
            or len(frame) <= horizon_bars:
        return empty
    import pandas as pd

    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    # Rolling extreme at index j covers bars j-h+1..j; base bar t
    # needs the window t+1..t+h, i.e. the extreme at index t+h.
    rmax = pd.Series(high).rolling(horizon_bars).max().to_numpy()
    rmin = pd.Series(low).rolling(horizon_bars).min().to_numpy()
    base = close[:-horizon_bars]
    ret = (close[horizon_bars:] / base - 1.0) * 100.0
    run_up = rmax[horizon_bars:] - base
    run_dn = base - rmin[horizon_bars:]
    ok = np.isfinite(ret) & np.isfinite(run_up) & np.isfinite(run_dn)
    return {"ret_pct": ret[ok], "run_up": run_up[ok],
            "run_dn": run_dn[ok]}


def load_seed_samples(path: str | Path) -> dict[str, list[dict[str, float]]]:
    """GS journal-export seed samples, grouped by horizon tag.

    Rows are JSONL dicts carrying ``horizon`` (a tag like ``"1d"``
    or H1 bars) plus ``ret_pct`` / ``run_up`` / ``run_dn`` — the
    realized-outcome shape GoldenSwing's forecast journal grades.
    Unreadable rows are skipped and logged, never invented.
    """
    out: dict[str, list[dict[str, float]]] = {}
    target = Path(path)
    if not target.exists():
        _log.warning("forecast: seed file missing: %s", target)
        return out
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            tag = row.get("horizon")
            if isinstance(tag, (int, float)):
                tag = next((t for t, b in HORIZONS.items()
                            if b == int(tag)), None)
            sample = {"ret_pct": float(row["ret_pct"]),
                      "run_up": float(row["run_up"]),
                      "run_dn": float(row["run_dn"])}
        except (ValueError, KeyError, TypeError):
            _log.warning("forecast: skipping unparseable seed row")
            continue
        if tag in HORIZONS and all(math.isfinite(v)
                                   for v in sample.values()):
            out.setdefault(str(tag), []).append(sample)
    return out


def _merged_sample(frame: Any, horizon_bars: int,
                   seed: list[dict[str, float]] | None) -> dict[str, Any]:
    own = forward_outcomes(frame, horizon_bars)
    seed = seed or []
    if not seed:
        return {**own, "frame_n": int(len(own["ret_pct"])),
                "seed_n": 0}
    ret = np.concatenate([own["ret_pct"],
                          np.array([s["ret_pct"] for s in seed])])
    run_up = np.concatenate([own["run_up"],
                             np.array([s["run_up"] for s in seed])])
    run_dn = np.concatenate([own["run_dn"],
                             np.array([s["run_dn"] for s in seed])])
    return {"ret_pct": ret, "run_up": run_up, "run_dn": run_dn,
            "frame_n": int(len(own["ret_pct"])), "seed_n": len(seed)}


# ── the estimator ────────────────────────────────────────────────────

def estimate_horizon(frame: Any, horizon_tag: str, *,
                     base_price: float, base_ts: float,
                     created: float,
                     seed: list[dict[str, float]] | None = None
                     ) -> dict[str, Any]:
    """One horizon's forecast block from the empirical sample.

    Every field of the output standard is present; when the
    sample is too thin the block says so (``unavailable_reason``)
    and the value fields are None — no fabricated precision.
    """
    horizon_bars = HORIZONS[horizon_tag]
    sample = _merged_sample(frame, horizon_bars, seed)
    ret = sample["ret_pct"]
    n = int(len(ret))
    block: dict[str, Any] = {
        "horizon": horizon_tag, "horizon_bars": horizon_bars,
        "direction": None, "point_price": None, "interval": None,
        "expected_move_pct": None, "expected_move_price": None,
        "mean_abs_move_pct": None,
        "fav_excursion": None, "adv_excursion": None,
        "disagreement_pct": None, "conviction_z": None,
        "scenario": None, "invalidation_price": None,
        "invalidation": None,
        "probabilities": {"state": "uncalibrated", "n": 0,
                          "up": None, "dn": None, "neutral": None},
        "method": METHOD, "version": FORECAST_VERSION,
        "n": n, "frame_n": sample["frame_n"],
        "seed_n": sample["seed_n"],
        "n_effective": int(n // horizon_bars) if n else 0,
        "deadband_pct": None, "unavailable_reason": None,
        "created": created, "created_utc": _iso_utc(created),
        "cutoff": base_ts, "cutoff_utc": _iso_utc(base_ts),
        "expiry": created + horizon_bars * 3600,
        "expiry_utc": _iso_utc(created + horizon_bars * 3600),
        "base_price": base_price}
    if n < MIN_SAMPLE_N:
        block["unavailable_reason"] = (
            f"empirical sample too thin: {n} forward windows "
            f"< {MIN_SAMPLE_N} floor")
        return block
    sd = float(np.std(ret))
    mean_ret = float(np.mean(ret))
    deadband = DEADBAND_SIGMA * sd
    if mean_ret > deadband:
        direction = "UP"
    elif mean_ret < -deadband:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"
    eff_up = direction == "UP" or (direction == "NEUTRAL"
                                   and mean_ret >= 0)
    lo_q = (1.0 - INTERVAL_COVERAGE) / 2.0
    q_lo, q_hi = (float(v) for v in
                  np.quantile(ret, [lo_q, 1.0 - lo_q]))
    point = base_price * (1.0 + mean_ret / 100.0)
    fav_src = sample["run_up"] if eff_up else sample["run_dn"]
    adv_src = sample["run_dn"] if eff_up else sample["run_up"]
    fav = float(np.mean(fav_src))
    adv = float(np.mean(adv_src))
    # Disagreement: the spread of three honest sub-estimators of
    # the same quantity (full-sample mean, recent-half mean,
    # full-sample median) — where they disagree, the forecast
    # itself is less settled, and the record says by how much.
    half = ret[len(ret) // 2:]
    sub = [mean_ret, float(np.mean(half)), float(np.median(ret))]
    disagreement = float(np.std(sub))
    inv_price = base_price * (1.0 + (q_lo if eff_up else q_hi) / 100.0)
    side = "up" if eff_up else "down"
    block.update({
        "direction": direction,
        "point_price": _r(point),
        "interval": {
            "low": _r(base_price * (1.0 + q_lo / 100.0)),
            "high": _r(base_price * (1.0 + q_hi / 100.0)),
            "coverage": INTERVAL_COVERAGE},
        "expected_move_pct": _r(mean_ret),
        "expected_move_price": _r(point - base_price),
        "mean_abs_move_pct": _r(float(np.mean(np.abs(ret)))),
        "fav_excursion": _r(fav), "adv_excursion": _r(adv),
        "disagreement_pct": _r(disagreement),
        "conviction_z": _r(abs(mean_ret) / sd if sd > 0 else 0.0),
        "deadband_pct": _r(deadband),
        "scenario": (
            f"{horizon_tag} central path: {direction} drift toward "
            f"{point:,.2f} inside the {INTERVAL_COVERAGE:.0%} band "
            f"{base_price * (1.0 + q_lo / 100.0):,.2f}–"
            f"{base_price * (1.0 + q_hi / 100.0):,.2f}; expected "
            f"favourable excursion ~{fav:,.2f}, adverse ~{adv:,.2f}"),
        "invalidation_price": _r(inv_price),
        "invalidation": (
            f"a close beyond {inv_price:,.2f} (the band's adverse "
            f"edge) falsifies the {side} central path")})
    return block


def make_forecasts(symbol: str, market: str = "crypto", *,
                   settings: Any = None, journal_path: Any = None,
                   now: float | None = None, frame: Any = None,
                   seed_samples: dict[str, list[dict[str, float]]
                                      ] | None = None,
                   journal: bool = True) -> list[dict[str, Any]]:
    """One forecast pass: all horizons, journaled, newest data.

    Fetches the H1 history at the market-data seam (unless a frame
    is supplied), estimates every horizon from the empirical
    sample (+ any seed samples), stamps probabilities from the
    calibration book, and appends one ``forecast`` event per
    horizon to the trading journal. Returns the records.
    """
    taken = float(now if now is not None else time.time())
    sym = (symbol or "").strip().upper()
    if frame is None:
        try:
            frame = market_data.get_ohlcv(
                sym, market=market, timeframe="1h", bars=HISTORY_BARS)
        except Exception as exc:  # noqa: BLE001 - stated, not hidden
            _log.info("forecast: H1 history for %s failed: %s",
                      sym, exc)
            frame = None
    book = calibration_book(symbol=sym, settings=settings,
                            journal_path=journal_path)
    records: list[dict[str, Any]] = []
    if frame is None or len(frame) == 0:
        base_price, base_ts = 0.0, taken
        frame_empty = True
    else:
        base_price = float(frame["close"].iloc[-1])
        base_ts = ts_epoch(frame.index[-1])
        frame_empty = False
    for tag in HORIZONS:
        if frame_empty:
            block = estimate_horizon(None, tag, base_price=base_price,
                                     base_ts=base_ts, created=taken)
            block["unavailable_reason"] = "no H1 history available"
        else:
            block = estimate_horizon(
                frame, tag, base_price=base_price, base_ts=base_ts,
                created=taken, seed=(seed_samples or {}).get(tag))
        block["probabilities"] = probabilities_from_book(book, tag)
        record: dict[str, Any] = {
            "kind": "forecast", "version": FORECAST_VERSION,
            "method": METHOD,
            "forecast_id": f"fc:{sym}:{taken:.0f}:{tag}",
            "symbol": sym, "market": market,
            "created": taken, "created_utc": _iso_utc(taken),
            **block}
        # horizon-level timestamps live inside the block already;
        # keep the record's top level unambiguous.
        record["horizon"] = tag
        if journal:
            trading_journal.append_event(
                "forecast", settings=settings, path=journal_path,
                **record)
        records.append(record)
    return records


def latest_forecasts(symbol: str | None = None, *,
                     settings: Any = None, journal_path: Any = None
                     ) -> dict[str, dict[str, Any]]:
    """Newest journaled forecast record per horizon tag."""
    out: dict[str, dict[str, Any]] = {}
    for rec in trading_journal.read_events(
            limit=0, event="forecast", settings=settings,
            path=journal_path):
        if symbol is not None and rec.get("symbol") != \
                symbol.strip().upper():
            continue
        tag = rec.get("horizon")
        if tag and tag not in out:
            out[tag] = rec
    return out


# ── outcomes: append-only grading input ──────────────────────────────

def _find_forecast(forecast_id: str, horizon: str, *,
                   settings: Any, journal_path: Any
                   ) -> dict[str, Any] | None:
    for rec in trading_journal.read_events(
            limit=0, event="forecast", settings=settings,
            path=journal_path):
        if rec.get("forecast_id") == forecast_id \
                and rec.get("horizon") == horizon:
            return rec
    return None


def realized_direction(ret_pct: float, deadband_pct: float | None
                       ) -> str:
    band = deadband_pct or 0.0
    if ret_pct > band:
        return "up"
    if ret_pct < -band:
        return "dn"
    return "neutral"


def attach_outcome(forecast_id: str, horizon: str, *,
                   ret_pct: float, run_up: float, run_dn: float,
                   end_price: float | None = None,
                   settings: Any = None, journal_path: Any = None,
                   graded_at: float | None = None
                   ) -> dict[str, Any]:
    """Append one graded outcome beside its forecast record.

    The forecast record itself is never rewritten — the outcome
    is its own journal event, folded in by the calibration book.
    Raises KeyError when the forecast is unknown (a grade against
    nothing is a fabrication, not a record).
    """
    fc = _find_forecast(forecast_id, horizon, settings=settings,
                        journal_path=journal_path)
    if fc is None:
        raise KeyError(f"unknown forecast {forecast_id} {horizon}")
    realized = {"ret_pct": float(ret_pct), "run_up": float(run_up),
                "run_dn": float(run_dn), "end_price": end_price}
    return trading_journal.append_event(
        "forecast_outcome", settings=settings, path=journal_path,
        forecast_id=forecast_id, horizon=horizon,
        symbol=fc.get("symbol"),
        realized_direction=realized_direction(
            float(ret_pct), fc.get("deadband_pct")),
        realized=realized,
        graded_at=float(graded_at if graded_at is not None
                        else time.time()))


def resolve_due_forecasts(symbol: str, market: str = "crypto", *,
                          settings: Any = None,
                          journal_path: Any = None,
                          now: float | None = None,
                          frame: Any = None) -> int:
    """Grade every matured, ungraded forecast from market data.

    Uses the H1 frame at the market-data seam: for each forecast
    whose expiry has passed and which has no outcome yet, the
    realized window is the ``horizon`` bars after its cutoff bar.
    Forecasts whose window the frame does not cover yet stay
    ungraded. Returns the number of outcomes appended.
    """
    taken = float(now if now is not None else time.time())
    sym = (symbol or "").strip().upper()
    records = trading_journal.read_events(
        limit=0, settings=settings, path=journal_path)
    graded = {(r.get("forecast_id"), r.get("horizon"))
              for r in records if r.get("event") == "forecast_outcome"}
    due = [r for r in records
           if r.get("event") == "forecast"
           and (r.get("symbol") or "") == sym
           and float(r.get("expiry") or 0) <= taken
           and (r.get("forecast_id"), r.get("horizon")) not in graded
           and r.get("direction") is not None]
    if not due:
        return 0
    if frame is None:
        try:
            frame = market_data.get_ohlcv(
                sym, market=market, timeframe="1h", bars=HISTORY_BARS)
        except Exception as exc:  # noqa: BLE001 - grading waits
            _log.info("forecast: resolve frame for %s failed: %s",
                      sym, exc)
            return 0
    if frame is None or len(frame) == 0:
        return 0
    stamps = [ts_epoch(ts) for ts in frame.index]
    made = 0
    for fc in due:
        horizon_bars = int(fc["horizon_bars"])
        cutoff = float(fc["cutoff"])
        base_idx = None
        for i, ts in enumerate(stamps):
            if ts <= cutoff:
                base_idx = i
            else:
                break
        if base_idx is None or base_idx + horizon_bars >= len(frame):
            continue  # window not fully inside the frame yet
        base_close = float(frame["close"].iloc[base_idx])
        window = frame.iloc[base_idx + 1:
                            base_idx + horizon_bars + 1]
        end_close = float(window["close"].iloc[-1])
        attach_outcome(
            fc["forecast_id"], fc["horizon"],
            ret_pct=(end_close / base_close - 1.0) * 100.0,
            run_up=float(window["high"].max()) - base_close,
            run_dn=base_close - float(window["low"].min()),
            end_price=end_close, settings=settings,
            journal_path=journal_path, graded_at=taken)
        made += 1
    return made


# ── the calibration book + E3 trust ──────────────────────────────────

def calibration_book(symbol: str | None = None, *,
                     settings: Any = None, journal_path: Any = None
                     ) -> dict[str, Any]:
    """Fold forecasts + outcomes into per-horizon calibration.

    Per horizon: graded ``n``, overall direction-hit ``base_rate``
    and, per direction (up/dn), the graded count, hit rate and the
    high-conviction pool (conviction_z >= CONVICTION_Z) that the
    E3 trust rule reads. Forecasts with no direction (unavailable
    blocks) never enter the book.
    """
    records = trading_journal.read_events(
        limit=0, settings=settings, path=journal_path)
    forecasts: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        if rec.get("event") == "forecast" \
                and rec.get("direction") is not None \
                and (symbol is None
                     or rec.get("symbol") == symbol.strip().upper()):
            forecasts[(rec["forecast_id"], rec["horizon"])] = rec
    horizons: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec.get("event") != "forecast_outcome":
            continue
        fc = forecasts.get((rec.get("forecast_id"),
                            rec.get("horizon")))
        if fc is None:
            continue
        tag = rec["horizon"]
        side = {"UP": "up", "DOWN": "dn"}.get(fc.get("direction") or "")
        hit = rec.get("realized_direction") == (
            side or "neutral")
        entry = horizons.setdefault(tag, {
            "n": 0, "hits": 0, "realized": {"up": 0, "dn": 0,
                                            "neutral": 0},
            "by_direction": {}})
        entry["n"] += 1
        entry["hits"] += 1 if hit else 0
        rd = rec.get("realized_direction")
        if rd in entry["realized"]:
            entry["realized"][rd] += 1
        if side:
            side_e = entry["by_direction"].setdefault(side, {
                "n": 0, "hits": 0, "pool_n": 0, "pool_hits": 0})
            side_e["n"] += 1
            side_e["hits"] += 1 if hit else 0
            if float(fc.get("conviction_z") or 0) >= CONVICTION_Z:
                side_e["pool_n"] += 1
                side_e["pool_hits"] += 1 if hit else 0
    for entry in horizons.values():
        entry["base_rate"] = (entry["hits"] / entry["n"]
                              if entry["n"] else None)
        for side_e in entry["by_direction"].values():
            side_e["rate"] = (side_e["hits"] / side_e["n"]
                              if side_e["n"] else None)
            side_e["pool_rate"] = (
                side_e["pool_hits"] / side_e["pool_n"]
                if side_e["pool_n"] else None)
    return {"horizons": horizons}


def trust_state(book: dict[str, Any] | None, horizon_tag: str,
                direction: int) -> dict[str, Any]:
    """The E3 trust verdict for one horizon + candidate direction.

    ``trusted`` only when the high-conviction pool at this horizon
    has at least TRUST_MIN_N graded forecasts resolving at
    >= TRUST_LIFT x the horizon base rate; ``anti_calibrated``
    when that pool resolves below ANTI_LIFT x base; otherwise
    ``unproven`` — recorded, never weighed.
    """
    out: dict[str, Any] = {"state": "unproven", "n": 0,
                           "pool_n": 0, "lift": None,
                           "base_rate": None, "pool_rate": None}
    if not book or direction not in (1, -1):
        return out
    entry = (book.get("horizons") or {}).get(horizon_tag) or {}
    side = (entry.get("by_direction") or {}).get(
        "up" if direction == 1 else "dn") or {}
    base = entry.get("base_rate")
    out["n"] = int(entry.get("n") or 0)
    out["pool_n"] = int(side.get("pool_n") or 0)
    out["base_rate"] = base
    out["pool_rate"] = side.get("pool_rate")
    if not base or side.get("pool_rate") is None \
            or out["pool_n"] < TRUST_MIN_N:
        return out
    lift = float(side["pool_rate"]) / float(base)
    out["lift"] = round(lift, 4)
    if lift >= TRUST_LIFT:
        out["state"] = "trusted"
    elif lift < ANTI_LIFT:
        out["state"] = "anti_calibrated"
    return out


def probabilities_from_book(book: dict[str, Any] | None,
                            horizon_tag: str) -> dict[str, Any]:
    """Earned probabilities for one horizon, or uncalibrated.

    Frequencies of realized directions over the horizon's graded
    forecasts — quoted only at MIN_PROB_N graded samples or more,
    with n stated either way.
    """
    entry = ((book or {}).get("horizons") or {}).get(horizon_tag) or {}
    n = int(entry.get("n") or 0)
    if n < MIN_PROB_N:
        return {"state": "uncalibrated", "n": n, "up": None,
                "dn": None, "neutral": None}
    realized = entry.get("realized") or {}
    return {"state": "calibrated", "n": n,
            "up": round(realized.get("up", 0) / n, 4),
            "dn": round(realized.get("dn", 0) / n, 4),
            "neutral": round(realized.get("neutral", 0) / n, 4)}


# ── surfaces ─────────────────────────────────────────────────────────

def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    return f"{float(value):,.2f}"


def render_forecasts(records: list[dict[str, Any]], *,
                     symbol: str | None = None) -> str:
    """Human rendering of one forecast pass — the surface text."""
    if not records:
        return "no forecasts — no H1 history available"
    head = records[0]
    lines = [f"{head.get('symbol')} forecasts — "
             f"{(head.get('created_utc') or '')[:16].replace('T', ' ')}"
             f" UTC · base {_fmt(head.get('base_price'))} · "
             f"{head.get('method')}"]
    for rec in records:
        if rec.get("direction") is None:
            lines.append(f"{rec['horizon']:<4} UNAVAILABLE — "
                         f"{rec.get('unavailable_reason')}")
            continue
        iv = rec.get("interval") or {}
        probs = rec.get("probabilities") or {}
        p_txt = ("uncalibrated (n="
                 f"{probs.get('n', 0)})" if probs.get("state")
                 != "calibrated" else
                 f"P up {probs.get('up'):.0%} / dn {probs.get('dn'):.0%}"
                 f" / flat {probs.get('neutral'):.0%} (n={probs.get('n')})")
        lines.append(
            f"{rec['horizon']:<4} {rec['direction']:<7} "
            f"point {_fmt(rec.get('point_price'))} · "
            f"band {_fmt(iv.get('low'))}–{_fmt(iv.get('high'))} "
            f"({iv.get('coverage', 0):.0%}) · "
            f"move {rec.get('expected_move_pct', 0):+.2f}% · "
            f"fav +{_fmt(rec.get('fav_excursion'))} / "
            f"adv −{_fmt(rec.get('adv_excursion'))} · {p_txt}")
    return "\n".join(lines)
