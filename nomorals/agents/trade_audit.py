"""Self-audit + validation — Devon trading system, Pass 5.

Grades the system's own journaled claims against the outcomes the
same journal later recorded, GoldenSwing-battery style:

- **Forecasts** (Pass 3) pair with their ``forecast_outcome``
  events per horizon: direction hit rate against the naive
  always-the-modal-direction baseline, median point error,
  observed interval coverage against the coverage the record
  STATED, and excursion capture (realized favourable / adverse
  runs against the excursions the record promised).
- **Judgment** (Pass 3, shadow) pairs with what its setups did
  next: a closed Pass 4 position on the setup gives a realized R;
  otherwise the setup's post-judgment lifecycle state is the
  categorical outcome (an abstention the market then triggered
  into is a MISS; an act whose setup died before entry was called
  WRONG). Discrimination is taken-vs-abstained on medians and hit
  rates — never means: one degenerate record must not be able to
  poison the verdict (the GS evidence-pack lesson).
- **Setups / playbooks** (Pass 2) grade through their lifecycle
  funnel (detected -> armed -> triggered / invalidated / expired)
  plus realized R over the playbook's closed positions where an
  outcome exists. A funnel with no executed outcome earns no
  verdict beyond insufficient_data — shadow motion is not proof.

CHRONOLOGY IS THE DISCIPLINE. Every graded pair carries the time
its outcome landed. The newest HOLDOUT_FRACTION of all graded
outcomes is the LOCKED HOLDOUT: verdicts are issued from holdout
grades only. The older fit slice is graded in chronological
walk-forward windows and is the only data the calibration
re-measurement may see — the P3 trust rule (pool n >= 20, lift
>= 1.25 trusted, < 0.90 anti-calibrated) is RE-MEASURED on fit
pairs here, never assumed from the live book; the live book's
states are reported beside it for contrast. The fitting path is
structurally blind to holdout rows: fit statistics are computed
from fit pairs alone, and a test proves holdout content cannot
move them.

VERDICTS are the settled vocabulary — ``insufficient_data``
(holdout paired outcomes < MIN_VERDICT_N), ``not_ready``,
``promotion_ready`` — and they are RECOMMENDATIONS ONLY. Judgment
stays shadow-first; promotion is the owner's separate config
decision, and nothing in this module executes, gates or sizes.

The audit itself is journaled (one ``trade_audit`` event with the
verdict map), so the journal can later grade the grader. Storage
is the Pass 1 journal alone — no new file, no migration.

Placement: ``agents/`` beside forecast/judgment/execution — the
layer from which the finance journal is a legal static import.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from typing import Any

from ..finance import trading_journal
from . import forecast as forecast_mod
from .setups import load_setups

_log = logging.getLogger(__name__)

#: Verdict vocabulary (settled).
VERDICTS: tuple[str, ...] = ("insufficient_data", "not_ready",
                             "promotion_ready")
#: A verdict needs at least this many HOLDOUT paired outcomes.
MIN_VERDICT_N = 20
#: The locked holdout is the newest this fraction of outcomes.
HOLDOUT_FRACTION = 0.25
#: Walk-forward windows across the fit slice.
WALK_WINDOWS = 4
#: Observed interval coverage may drift this far from the stated
#: coverage before the horizon reads not_ready on coverage.
COVERAGE_TOLERANCE = 0.10
#: Hit-rate bar for a promotion_ready judgment route / playbook.
HIT_RATE_BAR = 0.50

_DIR_MAP = {"UP": "up", "DOWN": "dn", "NEUTRAL": "neutral"}


# ── small honest statistics ──────────────────────────────────────────

def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _rate(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _iso(taken: float | None) -> str:
    if taken is None:
        return "?"
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                         time.gmtime(float(taken)))


def _window(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    stamps = [p["outcome_ts"] for p in pairs]
    return {"n": len(pairs),
            "from_ts": min(stamps) if stamps else None,
            "to_ts": max(stamps) if stamps else None}


# ── pair assembly: claims folded against outcomes ────────────────────

def _forecast_pairs(records: list[dict[str, Any]],
                    symbol: str | None) -> list[dict[str, Any]]:
    forecasts: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        if rec.get("event") == "forecast" \
                and rec.get("direction") is not None \
                and (symbol is None or rec.get("symbol") == symbol):
            forecasts[(rec.get("forecast_id"),
                       rec.get("horizon"))] = rec
    pairs: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("event") != "forecast_outcome":
            continue
        fc = forecasts.get((rec.get("forecast_id"),
                            rec.get("horizon")))
        if fc is None:
            continue
        realized = rec.get("realized") or {}
        direction = fc.get("direction")
        end_price = realized.get("end_price")
        point = fc.get("point_price")
        interval = fc.get("interval") or {}
        covered = None
        if end_price is not None and interval.get("low") is not None \
                and interval.get("high") is not None:
            covered = float(interval["low"]) <= float(end_price) \
                <= float(interval["high"])
        fav_ratio = adv_ratio = None
        if direction in ("UP", "DOWN"):
            run_up = realized.get("run_up")
            run_dn = realized.get("run_dn")
            fav_real = run_up if direction == "UP" else run_dn
            adv_real = run_dn if direction == "UP" else run_up
            fav_fc = fc.get("fav_excursion")
            adv_fc = fc.get("adv_excursion")
            if fav_real is not None and fav_fc:
                fav_ratio = float(fav_real) / float(fav_fc)
            if adv_real is not None and adv_fc:
                adv_ratio = float(adv_real) / float(adv_fc)
        pairs.append({
            "horizon": rec.get("horizon"),
            "outcome_ts": float(rec.get("graded_at")
                                or rec.get("ts") or 0.0),
            "created": fc.get("created"),
            "direction": direction,
            "hit": rec.get("realized_direction")
                   == _DIR_MAP.get(direction or ""),
            "realized_direction": rec.get("realized_direction"),
            "conviction_z": fc.get("conviction_z"),
            "point_err_pct": (
                (float(end_price) - float(point)) / float(point)
                * 100.0
                if end_price is not None and point else None),
            "covered": covered,
            "stated_coverage": interval.get("coverage"),
            "fav_ratio": fav_ratio, "adv_ratio": adv_ratio})
    pairs.sort(key=lambda p: p["outcome_ts"])
    return pairs


def _position_outcomes(records: list[dict[str, Any]],
                       symbol: str | None) -> list[dict[str, Any]]:
    """Closed positions folded from the Pass 4 journal events.

    ``position_closed.pnl`` is the position's TOTAL money pnl
    (Pass 4 writes partial-close pnl into it at the final close),
    so realized R = total pnl / the money risked at open
    (|entry - initial stop| x contract value x original volume).
    A position whose risk money cannot be reconstructed carries
    ``realized_r = None`` — counted, never invented.
    """
    opened: dict[str, dict[str, Any]] = {}
    closed: dict[str, dict[str, Any]] = {}
    for rec in records:
        pid = rec.get("position_id")
        if not pid:
            continue
        if rec.get("event") == "position_opened":
            opened[pid] = rec
        elif rec.get("event") == "position_closed":
            closed[pid] = rec
    out: list[dict[str, Any]] = []
    for pid, crec in closed.items():
        orec = opened.get(pid)
        if orec is None:
            continue
        if symbol is not None and orec.get("symbol") != symbol:
            continue
        entry = orec.get("entry")
        stop = orec.get("initial_stop", orec.get("stop_loss"))
        volume = orec.get("original_volume", orec.get("volume"))
        contract = orec.get("contract_value")
        risk_money = None
        if entry is not None and stop is not None and volume \
                and contract:
            risk_money = abs(float(entry) - float(stop)) \
                * float(contract) * float(volume)
        pnl = crec.get("pnl")
        realized_r = None
        if pnl is not None and risk_money:
            realized_r = round(float(pnl) / risk_money, 4)
        out.append({
            "position_id": pid,
            "setup_id": orec.get("setup_id"),
            "playbook": orec.get("playbook"),
            "symbol": orec.get("symbol"),
            "opened_ts": float(orec.get("ts") or 0.0),
            "outcome_ts": float(crec.get("ts") or 0.0),
            "pnl": float(pnl) if pnl is not None else None,
            "realized_r": realized_r,
            "close_reason": crec.get("reason")})
    out.sort(key=lambda p: p["outcome_ts"])
    return out


def _setup_terminals(setups: list[dict[str, Any]]
                     ) -> dict[str, dict[str, Any]]:
    """setup_id -> {state, terminal_ts} for terminal setups."""
    out: dict[str, dict[str, Any]] = {}
    for setup in setups:
        if setup.get("state") in ("triggered", "invalidated",
                                  "expired"):
            out[setup["setup_id"]] = {
                "state": setup["state"],
                "terminal_ts": float(setup.get("updated_at") or 0.0),
                "playbook": setup.get("playbook"),
                "route": setup.get("route")}
    return out


def _judgment_pairs(records: list[dict[str, Any]],
                    terminals: dict[str, dict[str, Any]],
                    positions: list[dict[str, Any]],
                    symbol: str | None) -> list[dict[str, Any]]:
    """Judgments folded against what their setups did next.

    The strongest outcome wins: a closed position opened after the
    judgment on the same setup gives a realized R. Otherwise the
    setup's terminal state — reached strictly AFTER the judgment —
    is the categorical outcome. A judgment on an already-triggered
    setup with no position has no post-judgment outcome at all:
    the pair counts as outcome-less and earns nothing.
    """
    by_setup: dict[str, list[dict[str, Any]]] = {}
    for pos in positions:
        if pos.get("setup_id"):
            by_setup.setdefault(pos["setup_id"], []).append(pos)
    pairs: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("event") != "judgment":
            continue
        subj = rec.get("subject") or {}
        if subj.get("type") != "setup":
            continue
        if symbol is not None and rec.get("symbol") != symbol:
            continue
        judged_ts = float(rec.get("judged_at") or rec.get("ts")
                          or 0.0)
        sid = subj.get("setup_id")
        decision = rec.get("decision")
        realized_r = None
        outcome_ts = None
        category = "no_outcome"
        for pos in by_setup.get(sid or "", []):
            if pos["opened_ts"] >= judged_ts \
                    and pos.get("realized_r") is not None:
                realized_r = pos["realized_r"]
                outcome_ts = pos["outcome_ts"]
                category = "r_outcome"
                break
        if outcome_ts is None:
            term = terminals.get(sid or "")
            if term is not None \
                    and term["terminal_ts"] > judged_ts:
                outcome_ts = term["terminal_ts"]
                state = term["state"]
                if decision == "act":
                    category = "entered" if state == "triggered" \
                        else "called_wrong"
                else:
                    category = "missed" if state == "triggered" \
                        else "stood_aside_right"
        if outcome_ts is None:
            continue  # no outcome exists yet — not a graded pair
        pairs.append({
            "route": subj.get("route") or subj.get("playbook")
                     or "unknown",
            "playbook": subj.get("playbook"),
            "setup_id": sid, "decision": decision,
            "outcome_ts": outcome_ts, "realized_r": realized_r,
            "category": category,
            "expected_r": rec.get("expected_r"),
            "reason_category": rec.get("reason_category")})
    pairs.sort(key=lambda p: p["outcome_ts"])
    return pairs


# ── grading: one stat block per slice of pairs ───────────────────────

def forecast_stats(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-horizon grades over one slice of forecast pairs."""
    n = len(pairs)
    hits = sum(1 for p in pairs if p["hit"])
    mix = {"up": 0, "dn": 0, "neutral": 0}
    for p in pairs:
        if p.get("realized_direction") in mix:
            mix[p["realized_direction"]] += 1
    naive = (max(mix.values()) / n) if n else None
    hit_rate = _rate(hits, n)
    errs = [p["point_err_pct"] for p in pairs
            if p.get("point_err_pct") is not None]
    cov = [p for p in pairs if p.get("covered") is not None]
    covered = sum(1 for p in cov if p["covered"])
    stated = None
    if cov:
        stated_counts: dict[Any, int] = {}
        for p in cov:
            stated_counts[p.get("stated_coverage")] = \
                stated_counts.get(p.get("stated_coverage"), 0) + 1
        stated = max(stated_counts,
                     key=lambda k: stated_counts[k])
    favs = [p["fav_ratio"] for p in pairs
            if p.get("fav_ratio") is not None]
    advs = [p["adv_ratio"] for p in pairs
            if p.get("adv_ratio") is not None]
    # The E3 trust pool, re-measured on THESE pairs only:
    # directional forecasts at conviction >= CONVICTION_Z.
    pool = [p for p in pairs
            if p.get("direction") in ("UP", "DOWN")
            and float(p.get("conviction_z") or 0)
            >= forecast_mod.CONVICTION_Z]
    pool_hits = sum(1 for p in pool if p["hit"])
    pool_rate = _rate(pool_hits, len(pool))
    lift = None
    if pool_rate is not None and hit_rate:
        lift = round(pool_rate / hit_rate, 4)
    trust = "unproven"
    if lift is not None and len(pool) >= forecast_mod.TRUST_MIN_N:
        if lift >= forecast_mod.TRUST_LIFT:
            trust = "trusted"
        elif lift < forecast_mod.ANTI_LIFT:
            trust = "anti_calibrated"
    by_side: dict[str, Any] = {}
    for side, word in (("up", "UP"), ("dn", "DOWN")):
        side_pool = [p for p in pool if p["direction"] == word]
        side_rate = _rate(sum(1 for p in side_pool if p["hit"]),
                          len(side_pool))
        side_lift = round(side_rate / hit_rate, 4) \
            if side_rate is not None and hit_rate else None
        side_trust = "unproven"
        if side_lift is not None \
                and len(side_pool) >= forecast_mod.TRUST_MIN_N:
            if side_lift >= forecast_mod.TRUST_LIFT:
                side_trust = "trusted"
            elif side_lift < forecast_mod.ANTI_LIFT:
                side_trust = "anti_calibrated"
        by_side[side] = {"pool_n": len(side_pool),
                         "pool_rate": side_rate,
                         "lift": side_lift, "trust": side_trust}
    return {"n": n, "hits": hits, "hit_rate": hit_rate,
            "realized_mix": mix, "naive_rate": naive,
            "lift_vs_naive": (round(hit_rate / naive, 4)
                              if hit_rate is not None and naive
                              else None),
            "point_n": len(errs),
            "median_abs_point_err_pct": (
                round(_median([abs(e) for e in errs]), 4)
                if errs else None),
            "median_signed_point_err_pct": (
                round(_median(errs), 4) if errs else None),
            "coverage_n": len(cov), "covered": covered,
            "coverage_observed": _rate(covered, len(cov)),
            "coverage_stated": stated,
            "excursion_n": len(favs),
            "median_fav_capture": (round(_median(favs), 4)
                                   if favs else None),
            "median_adv_capture": (round(_median(advs), 4)
                                   if advs else None),
            "trust_remeasured": {
                "pool_n": len(pool), "pool_hits": pool_hits,
                "pool_rate": pool_rate, "base_rate": hit_rate,
                "lift": lift, "state": trust, "by_side": by_side}}


def _r_stats(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    rs = [p["realized_r"] for p in pairs
          if p.get("realized_r") is not None]
    return {"n": len(pairs), "r_n": len(rs),
            "median_r": (round(_median(rs), 4) if rs else None),
            "hit_rate": (_rate(sum(1 for r in rs if r > 0), len(rs))
                         if rs else None),
            "best_r": max(rs) if rs else None,
            "worst_r": min(rs) if rs else None}


def judgment_stats(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Discrimination grades over one slice of judgment pairs."""
    act = [p for p in pairs if p["decision"] == "act"]
    abstain = [p for p in pairs if p["decision"] != "act"]
    cats: dict[str, int] = {}
    for p in pairs:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    return {"n": len(pairs), "act": _r_stats(act),
            "abstain": _r_stats(abstain), "categories": cats,
            "missed": cats.get("missed", 0),
            "stood_aside_right": cats.get("stood_aside_right", 0),
            "called_wrong": cats.get("called_wrong", 0),
            "entered": cats.get("entered", 0)}


def playbook_stats(setups: list[dict[str, Any]],
                   positions: list[dict[str, Any]]) -> dict[str, Any]:
    """Lifecycle funnel + position R over one slice.

    ``setups`` here are terminal folds (the slice is by terminal
    time); live setups are not outcomes and never enter a slice.
    """
    funnel = {"detected": 0, "armed": 0, "triggered": 0,
              "invalidated": 0, "expired": 0}
    for setup in setups:
        history = setup.get("history") or []
        seen = {h.get("to") for h in history}
        if "detected" in seen or setup.get("detected_at"):
            funnel["detected"] += 1
        if "armed" in seen:
            funnel["armed"] += 1
        state = setup.get("state")
        if state in ("triggered", "invalidated", "expired"):
            funnel[state] += 1
    stats = _r_stats(positions)
    stats["total_pnl"] = round(sum(
        float(p["pnl"]) for p in positions
        if p.get("pnl") is not None), 2)
    return {"funnel": funnel,
            "armed_rate": _rate(funnel["armed"], funnel["detected"]),
            "trigger_rate": _rate(funnel["triggered"],
                                  funnel["armed"]),
            **stats}


# ── the split: locked holdout vs fit, walk-forward windows ──────────

def _split(all_ts: list[float]) -> tuple[float | None, int, int]:
    """(cutoff_ts, fit_n, holdout_n) over every graded outcome.

    The holdout is the newest HOLDOUT_FRACTION of outcome times;
    fit is everything strictly before the cutoff. Fewer than 4
    outcomes cannot hold anything out — all of it is fit and every
    verdict reads insufficient_data on an empty holdout.
    """
    ordered = sorted(all_ts)
    n = len(ordered)
    if n < 4:
        return None, n, 0
    holdout_n = int(math.ceil(n * HOLDOUT_FRACTION))
    cutoff = ordered[n - holdout_n]
    fit_n = sum(1 for t in ordered if t < cutoff)
    return cutoff, fit_n, n - fit_n


def _slice(pairs: list[dict[str, Any]], cutoff: float | None,
           holdout: bool) -> list[dict[str, Any]]:
    if cutoff is None:
        return [] if holdout else list(pairs)
    if holdout:
        return [p for p in pairs if p["outcome_ts"] >= cutoff]
    return [p for p in pairs if p["outcome_ts"] < cutoff]


def _walk_windows(pairs: list[dict[str, Any]], stats_fn: Any
                  ) -> list[dict[str, Any]]:
    """Chronological walk-forward grades across the fit slice."""
    if not pairs:
        return []
    count = min(WALK_WINDOWS, len(pairs))
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    for i, pair in enumerate(pairs):
        chunks[min(count - 1, i * count // len(pairs))].append(pair)
    return [{**_window(chunk), "stats": stats_fn(chunk)}
            for chunk in chunks if chunk]


# ── verdicts: recommendation only, from the holdout alone ────────────

def _forecast_verdict(hold: dict[str, Any], fit: dict[str, Any]
                      ) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if hold["n"] < MIN_VERDICT_N:
        return "insufficient_data", [
            f"holdout paired outcomes {hold['n']} < "
            f"{MIN_VERDICT_N} — no verdict is earned yet"]
    coverage_ok = True
    if hold["coverage_n"] and hold["coverage_stated"] is not None \
            and hold["coverage_observed"] is not None:
        drift = abs(hold["coverage_observed"]
                    - float(hold["coverage_stated"]))
        if drift > COVERAGE_TOLERANCE:
            coverage_ok = False
            reasons.append(
                f"interval coverage {hold['coverage_observed']:.0%} "
                f"vs stated {float(hold['coverage_stated']):.0%} "
                f"(drift {drift:.0%} > {COVERAGE_TOLERANCE:.0%})")
    else:
        coverage_ok = False
        reasons.append("no interval coverage measurable in the "
                       "holdout (outcomes lack end prices)")
    lift_ok = hold["lift_vs_naive"] is not None \
        and hold["lift_vs_naive"] >= forecast_mod.TRUST_LIFT
    if not lift_ok:
        reasons.append(
            f"direction lift vs naive {hold['lift_vs_naive']} < "
            f"{forecast_mod.TRUST_LIFT} (hit rate "
            f"{hold['hit_rate']} vs naive {hold['naive_rate']})")
    anti = fit["trust_remeasured"]["state"] == "anti_calibrated"
    if anti:
        reasons.append("the fit slice re-measures the trust pool "
                       "as ANTI-calibrated — a lucky holdout does "
                       "not promote an anti-calibrated feed")
    if coverage_ok and lift_ok and not anti:
        return "promotion_ready", [
            "holdout bars met: coverage inside tolerance and "
            "direction lift at the trust bar — recommendation "
            "only; promotion is a separate owner decision"]
    return "not_ready", reasons


def _r_verdict(hold: dict[str, Any], *, act: bool = False,
               abstain_hold: dict[str, Any] | None = None
               ) -> tuple[str, list[str]]:
    """Verdict for an R-graded entity (judgment route / playbook)."""
    mine = hold["act"] if act else hold
    reasons: list[str] = []
    if mine["r_n"] < MIN_VERDICT_N:
        return "insufficient_data", [
            f"holdout R-bearing outcomes {mine['r_n']} < "
            f"{MIN_VERDICT_N} — lifecycle motion alone earns no "
            "verdict"]
    if mine["median_r"] is None or mine["median_r"] <= 0:
        reasons.append(f"median holdout R {mine['median_r']} "
                       "is not positive")
    if mine["hit_rate"] is None or mine["hit_rate"] < HIT_RATE_BAR:
        reasons.append(f"holdout hit rate {mine['hit_rate']} < "
                       f"{HIT_RATE_BAR}")
    if act and abstain_hold is not None \
            and abstain_hold["r_n"] >= MIN_VERDICT_N \
            and abstain_hold["median_r"] is not None \
            and mine["median_r"] is not None \
            and mine["median_r"] <= abstain_hold["median_r"]:
        reasons.append(
            "no discrimination: act median R "
            f"{mine['median_r']} does not beat the abstained "
            f"median {abstain_hold['median_r']} where both are "
            "measured")
    if reasons:
        return "not_ready", reasons
    return "promotion_ready", [
        "holdout bars met: positive median R at the hit-rate bar "
        "— recommendation only; promotion is a separate owner "
        "decision"]


# ── the audit ────────────────────────────────────────────────────────

def run_audit(symbol: str | None = None, *,
              settings: Any = None, journal_path: Any = None,
              now: float | None = None, journal: bool = True
              ) -> dict[str, Any]:
    """Grade the trading journal against itself. Pure read + one
    journaled ``trade_audit`` event (the verdict map)."""
    taken = float(now if now is not None else time.time())
    sym = symbol.strip().upper() if symbol else None
    records = trading_journal.read_events(
        limit=0, settings=settings, path=journal_path)
    records = list(reversed(records))  # chronological

    fc_pairs = _forecast_pairs(records, sym)
    positions = _position_outcomes(records, sym)
    setups = load_setups(settings=settings,
                         journal_path=journal_path, symbol=sym)
    terminals = _setup_terminals(setups)
    jd_pairs = _judgment_pairs(records, terminals, positions, sym)

    terminal_setups = [s for s in setups
                       if s.get("state") in ("triggered",
                                             "invalidated",
                                             "expired")]
    all_ts = [p["outcome_ts"] for p in fc_pairs] \
        + [p["outcome_ts"] for p in jd_pairs] \
        + [p["outcome_ts"] for p in positions] \
        + [float(s.get("updated_at") or 0.0)
           for s in terminal_setups]
    cutoff, fit_n, holdout_n = _split(all_ts)

    def entity(pairs: list[dict[str, Any]], stats_fn: Any,
               verdict_fn: Any) -> dict[str, Any]:
        fit_pairs = _slice(pairs, cutoff, holdout=False)
        hold_pairs = _slice(pairs, cutoff, holdout=True)
        fit_stats = stats_fn(fit_pairs)
        hold_stats = stats_fn(hold_pairs)
        verdict, reasons = verdict_fn(hold_stats, fit_stats)
        return {"verdict": verdict, "verdict_reasons": reasons,
                "fit": {**_window(fit_pairs), "stats": fit_stats,
                        "windows": _walk_windows(fit_pairs,
                                                 stats_fn)},
                "holdout": {**_window(hold_pairs),
                            "stats": hold_stats},
                "total_n": len(pairs)}

    forecasts_out: dict[str, Any] = {}
    for tag in sorted({p["horizon"] for p in fc_pairs}):
        forecasts_out[tag] = entity(
            [p for p in fc_pairs if p["horizon"] == tag],
            forecast_stats, _forecast_verdict)

    def _jd_verdict(hold: dict[str, Any], fit: dict[str, Any]
                    ) -> tuple[str, list[str]]:
        return _r_verdict(hold, act=True,
                          abstain_hold=hold["abstain"])

    judgment_out: dict[str, Any] = {"overall": entity(
        jd_pairs, judgment_stats, _jd_verdict), "routes": {}}
    for route in sorted({p["route"] for p in jd_pairs}):
        judgment_out["routes"][route] = entity(
            [p for p in jd_pairs if p["route"] == route],
            judgment_stats, _jd_verdict)

    playbooks_out: dict[str, Any] = {}
    pb_names = sorted({s.get("playbook") for s in terminal_setups}
                      | {p.get("playbook") for p in positions})
    for name in pb_names:
        if not name:
            continue
        pb_setups = [s for s in terminal_setups
                     if s.get("playbook") == name]
        pb_positions = [p for p in positions
                        if p.get("playbook") == name]
        # The playbook slice clock is the outcome time: terminal
        # time for setups, close time for positions — one merged
        # pair list keyed on outcome_ts drives the split.
        merged = [{"outcome_ts": float(s.get("updated_at") or 0.0),
                   "_setup": s} for s in pb_setups] \
            + [{"outcome_ts": p["outcome_ts"], "_pos": p}
               for p in pb_positions]

        def _pb_stats(slice_pairs: list[dict[str, Any]]
                      ) -> dict[str, Any]:
            ss = [p["_setup"] for p in slice_pairs if "_setup" in p]
            pp = [p["_pos"] for p in slice_pairs if "_pos" in p]
            return playbook_stats(ss, pp)

        def _pb_verdict(hold: dict[str, Any],
                        fit: dict[str, Any]
                        ) -> tuple[str, list[str]]:
            return _r_verdict(hold)

        playbooks_out[name] = entity(merged, _pb_stats,
                                     _pb_verdict)

    verdicts: dict[str, dict[str, str]] = {
        "forecasts": {tag: e["verdict"]
                      for tag, e in forecasts_out.items()},
        "judgment_routes": {route: e["verdict"] for route, e
                            in judgment_out["routes"].items()},
        "playbooks": {name: e["verdict"]
                      for name, e in playbooks_out.items()}}
    report: dict[str, Any] = {
        "kind": "trade_audit", "version": 1,
        "symbol": sym, "taken_at": taken,
        "taken_utc": _iso(taken),
        "split": {"outcome_n": len(all_ts), "cutoff_ts": cutoff,
                  "cutoff_utc": _iso(cutoff),
                  "holdout_fraction": HOLDOUT_FRACTION,
                  "fit_n": fit_n, "holdout_n": holdout_n},
        "forecasts": forecasts_out, "judgment": judgment_out,
        "playbooks": playbooks_out, "verdicts": verdicts}
    if journal:
        trading_journal.append_event(
            "trade_audit", settings=settings, path=journal_path,
            symbol=sym, taken_at=taken, outcome_n=len(all_ts),
            cutoff_ts=cutoff, verdicts=verdicts)
    return report


# ── surfaces ─────────────────────────────────────────────────────────

def _pct(value: Any) -> str:
    return "?" if value is None else f"{float(value):.0%}"


def _num(value: Any, nd: int = 2) -> str:
    return "?" if value is None else f"{float(value):+.{nd}f}"


def _plain(value: Any, nd: int = 2) -> str:
    return "?" if value is None else f"{float(value):.{nd}f}"


def _entity_line(name: str, ent: dict[str, Any], detail: str) -> str:
    hold = ent["holdout"]
    return (f"{name:<18} {ent['verdict']:<18} "
            f"holdout n={hold['n']} "
            f"[{_iso(hold['from_ts'])[:10]}.."
            f"{_iso(hold['to_ts'])[:10]}] {detail}")


def render_audit(report: dict[str, Any]) -> str:
    """Human rendering of one audit — the surface text."""
    split = report["split"]
    lines = [
        f"trade audit {report.get('symbol') or 'all symbols'} — "
        f"{report['taken_utc'][:16].replace('T', ' ')} UTC · "
        f"{split['outcome_n']} graded outcomes · holdout = newest "
        f"{split['holdout_fraction']:.0%} ({split['holdout_n']}) "
        f"locked after {_iso(split['cutoff_ts'])[:10]}, "
        f"fit {split['fit_n']} graded walk-forward",
        "verdicts are recommendations only — promotion stays a "
        "separate owner decision; judgment remains shadow"]
    if split["outcome_n"] == 0:
        lines.append("no graded outcomes in the journal yet — "
                     "every section reads insufficient_data")
        return "\n".join(lines)
    lines.append("forecasts (per horizon)")
    if not report["forecasts"]:
        lines.append("  no forecast has an outcome yet")
    for tag, ent in report["forecasts"].items():
        hs = ent["holdout"]["stats"]
        fs = ent["fit"]["stats"]
        detail = (
            f"hit {_pct(hs['hit_rate'])} vs naive "
            f"{_pct(hs['naive_rate'])} (lift "
            f"{_plain(hs['lift_vs_naive'])})"
            f" · coverage {_pct(hs['coverage_observed'])} vs "
            f"stated {_pct(hs['coverage_stated'])} "
            f"(n={hs['coverage_n']}) · median point error "
            f"{_num(hs['median_abs_point_err_pct'])}% · fav capture "
            f"{_plain(hs['median_fav_capture'])} · fit trust "
            f"re-measured: {fs['trust_remeasured']['state']} "
            f"(pool n={fs['trust_remeasured']['pool_n']}, lift "
            f"{_plain(fs['trust_remeasured']['lift'])})")
        lines.append("  " + _entity_line(tag, ent, detail))
        if ent["verdict"] == "insufficient_data":
            lines.append(f"    fit n={ent['fit']['n']} shown for "
                         "context — the verdict waits on the "
                         "holdout")
    lines.append("judgment discrimination (per route)")
    if not report["judgment"]["routes"]:
        lines.append("  no judgment has a graded outcome yet")
    for route, ent in report["judgment"]["routes"].items():
        hs = ent["holdout"]["stats"]
        detail = (
            f"act median R {_num(hs['act']['median_r'])} "
            f"(r_n={hs['act']['r_n']}, hit "
            f"{_pct(hs['act']['hit_rate'])}) · abstain median R "
            f"{_num(hs['abstain']['median_r'])} "
            f"(r_n={hs['abstain']['r_n']}) · missed "
            f"{hs['missed']} · stood-aside-right "
            f"{hs['stood_aside_right']} · called-wrong "
            f"{hs['called_wrong']}")
        lines.append("  " + _entity_line(route, ent, detail))
    lines.append("playbooks (lifecycle + realized R)")
    if not report["playbooks"]:
        lines.append("  no playbook has a terminal setup or a "
                     "closed position yet")
    for name, ent in report["playbooks"].items():
        hs = ent["holdout"]["stats"]
        fn = hs["funnel"]
        detail = (
            f"funnel holdout: armed {fn['armed']} · triggered "
            f"{fn['triggered']} · invalidated {fn['invalidated']} "
            f"· expired {fn['expired']} · median R "
            f"{_num(hs['median_r'])} (r_n={hs['r_n']}, hit "
            f"{_pct(hs['hit_rate'])}) · pnl {_num(hs['total_pnl'])}")
        lines.append("  " + _entity_line(name, ent, detail))
    return "\n".join(lines)
