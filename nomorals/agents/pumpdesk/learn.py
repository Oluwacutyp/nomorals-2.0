"""The learning loop — the desk recalibrates from its own grades.

Design §4.4 rule 6 + §9.5: the score weights recalibrate against
the call journal in the study cycle — but weight changes are
**journaled and reviewable, never silent drift**.  Every study
cycle's recalibration attempt appends a record to
``score_params.jsonl``: an adjustment (with its full evidence
window and before/after), a no-improvement hold, or an
INSUFFICIENT_DATA refusal.  The desk's parameter history is as
auditable as its call history.

The method is deliberately conservative:

* **Evidence = graded calls only**, and only calls that carry
  their score components (P2 calls do; older rows without
  components are excluded and the exclusion is counted in the
  record).  Below ``MIN_GRADED_CALLS`` graded-with-components,
  nothing adjusts — INSUFFICIENT_DATA, journaled.
* **Candidates are single bounded steps**: one weight ±0.03 or
  the call threshold ±0.02, weights re-normalized.  Each
  candidate is backtested ON THE SAME JOURNAL: what hit rate
  would the kept set have had?  A candidate is adopted only if
  its kept set stays >= ``MIN_KEPT`` calls, keeps >= 60% of the
  actual call volume, and improves the kept-set hit rate by
  >= ``MIN_GAIN``.
* **Honest limits, stated in every record**: the journal holds
  only calls the old parameters admitted, so the backtest can
  verify that a step would have FILTERED better; it cannot prove
  what a looser gate would have caught.  Steps therefore tend
  conservative by construction, and the record says
  ``basis: in-sample journal backtest`` — calibration, not proof.
"""

from __future__ import annotations

import time
from typing import Any

from . import score as _score

__all__ = [
    "MIN_GRADED_CALLS", "MIN_KEPT", "MIN_GAIN", "MAX_WEIGHT_STEP",
    "MAX_THRESHOLD_STEP", "default_params", "current_params",
    "recalibrate", "render_learning",
    "WIN_FEATURES", "WIN_MIN_CALLS", "win_features", "win_rows",
    "fit_win_model", "win_probability", "load_win_model",
]

#: Sample floors — below them the loop refuses, in writing.
MIN_GRADED_CALLS = 20
MIN_KEPT = 8
#: A candidate must beat the incumbent by at least this much
#: (kept-set hit rate) to earn adoption.
MIN_GAIN = 0.05
#: Bounded steps: one cycle never moves a parameter further.
MAX_WEIGHT_STEP = 0.03
MAX_THRESHOLD_STEP = 0.02
#: An adopted candidate must keep at least this share of the
#: journal's actual calls — no "improvement" by calling nothing.
MIN_VOLUME_KEPT = 0.6

# ── win model (Phase 2) ─────────────────────────────────────────────────
#: Features for the learned win-probability classifier — the four score
#: components (as safeties, so higher is better), the composite, and
#: the entry size / curve progress. All come from the graded call
#: journal, never from a live guess.
WIN_FEATURES = ("momentum", "smart", "safety_rug", "safety_manip",
                "composite", "mc_log", "curve_pct")
#: Floor for the win model: graded wins/losses with components.
WIN_MIN_CALLS = 30


def _call_get(call: Any, key: str, default: Any = None) -> Any:
    if isinstance(call, dict):
        return call.get(key, default)
    return getattr(call, key, default)


def win_features(call: Any) -> dict[str, float]:
    """The win model's feature vector for one call (a PumpCall or a
    plain dict with the same keys). All 0..1 except mc_log (log10 SOL).
    """
    import math as _math

    comps = _call_get(call, "components") or {}
    momentum = float(comps.get("momentum", 0.0) or 0.0)
    smart = float(comps.get("smart_money", comps.get("smart", 0.0))
                  or 0.0)
    rug = float(comps.get("rug_risk", 0.0) or 0.0)
    manip = float(comps.get("manipulation", 0.0) or 0.0)
    composite = _call_get(call, "composite")
    if composite is None:
        w = _score.DEFAULT_WEIGHTS
        composite = (w["momentum"] * momentum + w["smart"] * smart
                     + w["safety_rug"] * (1.0 - rug)
                     + w["safety_manip"] * (1.0 - manip))
    entry_mc = float(_call_get(call, "entry_mc_sol", 0.0) or 0.0)
    curve_pct = float(_call_get(call, "entry_curve_pct", 0.0) or 0.0)
    return {
        "momentum": max(0.0, min(1.0, momentum)),
        "smart": max(0.0, min(1.0, smart)),
        "safety_rug": max(0.0, min(1.0, 1.0 - rug)),
        "safety_manip": max(0.0, min(1.0, 1.0 - manip)),
        "composite": max(0.0, min(1.0, float(composite))),
        "mc_log": max(0.0, _math.log10(max(entry_mc, 0.0) + 1.0)),
        "curve_pct": max(0.0, min(1.0, curve_pct)),
    }


def win_rows(calls: list[Any]) -> tuple[list[dict[str, Any]], int]:
    """Graded calls as (features, label) rows. "miss-no-data" calls
    are EXCLUDED (no outcome was observed — they are not losses, and
    teaching the model that silence means loss would be a lie).
    Returns (rows, n_excluded)."""
    rows: list[dict[str, Any]] = []
    excluded = 0
    for c in calls or []:
        outcome = str(_call_get(c, "outcome", "") or "").strip().lower()
        if outcome == "miss-no-data":
            excluded += 1
            continue
        scored = _call_get(c, "scored")
        if scored is None:
            if outcome == "win":
                label = 1
            elif outcome == "loss":
                label = 0
            else:
                excluded += 1
                continue
        else:
            label = 1 if scored else 0
        comps = _call_get(c, "components") or {}
        if not comps:
            excluded += 1
            continue
        rows.append({"features": win_features(c), "label": label,
                     "call_id": str(_call_get(c, "call_id", "") or "")})
    return rows, excluded


def fit_win_model(calls: list[Any], *, store: Any = None,
                  backend: str = "logistic") -> Any:
    """Batch-fit the win-probability classifier on the graded call
    journal. Returns the toolkit's :class:`llm.specialists.FitOutcome`
    — "trained" (saved under ``pumpdesk-win``), "held", "insufficient"
    (below ``WIN_MIN_CALLS``), or "unavailable". The code rung is the
    incumbent composite rule (call iff composite >= CALL_THRESHOLD) on
    the same held-out slice — the model must beat the rule that
    admitted the calls, not a straw man. Default backend is logistic:
    the job IS a linear re-weighting of the score components (the
    composite itself is linear), and gradboost underfits that diagonal
    boundary on journal-sized data (0.75 vs logistic 0.95 on the same
    held-out slice)."""
    # NOTE: the backend choice above is measured, not guessed — see the
    # test suite's win-model test, which asserts the trained model
    # beats the composite rule on a held-out slice.
    from ...llm.specialists import LearnerSpec, ModelStore, fit_if_ready

    rows, excluded = win_rows(calls)

    def _code_score(feats: list[dict[str, float]],
                    labels: list[float]) -> float:
        if not labels:
            return 0.0
        correct = 0
        for f, y in zip(feats, labels, strict=True):
            pred = 1 if float(f.get("composite", 0.0) or 0.0) >= \
                _score.CALL_THRESHOLD else 0
            if pred == int(y):
                correct += 1
        return correct / len(labels)

    spec = LearnerSpec(job="pumpdesk-win", features=WIN_FEATURES,
                       kind="classifier", backend=backend,
                       min_examples=WIN_MIN_CALLS, label_name="won")
    outcome = fit_if_ready(rows, spec, code_score_fn=_code_score,
                           store=(store or ModelStore()))
    if outcome.report is not None:
        outcome.report.detail["excluded_no_outcome"] = excluded
    return outcome


def win_probability(model: Any, call: Any) -> float:
    """P(win) for one call under a fitted win model. 0.5 when the
    model is absent or the features are unusable — never a fake
    confident number."""
    if model is None:
        return 0.5
    try:
        return max(0.0, min(1.0, float(
            model.predict_one(win_features(call)))))
    except Exception:  # noqa: BLE001 - a bad model never kills a verdict
        return 0.5


def load_win_model(store: Any = None) -> Any | None:
    """The fitted win model, if one was ever promoted."""
    try:
        from ...llm.specialists import ModelStore

        return (store or ModelStore()).load("pumpdesk-win")
    except Exception:  # noqa: BLE001 - a bad store never kills a verdict
        return None


def default_params() -> dict[str, Any]:
    return {
        "version": 1,
        "weights": dict(_score.DEFAULT_WEIGHTS),
        "call_threshold": _score.CALL_THRESHOLD,
        "momentum_floor": _score.MOMENTUM_FLOOR,
        "updated_at": 0.0,
        "basis": "P1 calibrated constants (design §4.1)",
    }


def current_params(store: Any) -> dict[str, Any]:
    """The desk's live parameters: the latest journaled version,
    or the P1 defaults when the loop has never adjusted."""
    saved = store.score_params()
    if saved and saved.get("weights"):
        return saved
    return default_params()


def _composite(components: dict[str, Any], weights: dict[str, float]) -> float:
    return max(0.0, min(1.0,
                        weights["momentum"] * components.get("momentum", 0.0)
                        + weights["smart"] * components.get("smart_money", 0.0)
                        + weights["safety_rug"] * (1.0 - components.get("rug_risk", 0.0))
                        + weights["safety_manip"] * (1.0 - components.get("manipulation", 0.0))))


def _backtest(rows: list[dict[str, Any]], weights: dict[str, float],
              threshold: float, momentum_floor: float) -> dict[str, Any]:
    kept = [r for r in rows
            if _composite(r["components"], weights) >= threshold
            and r["components"].get("momentum", 0.0) >= momentum_floor]
    wins = sum(1 for r in kept if r["win"])
    return {"kept": len(kept), "wins": wins,
            "hit_rate": round(wins / len(kept), 4) if kept else None}


def _candidates(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Single bounded steps from the incumbent — one parameter at
    a time, weights re-normalized to sum 1."""
    out = []
    w = params["weights"]
    for key in ("momentum", "smart", "safety_rug", "safety_manip"):
        for delta in (MAX_WEIGHT_STEP, -MAX_WEIGHT_STEP):
            cand = dict(w)
            cand[key] = max(0.02, min(0.6, cand[key] + delta))
            total = sum(cand.values())
            cand = {k: round(v / total, 4) for k, v in cand.items()}
            out.append({"weights": cand,
                        "call_threshold": params["call_threshold"],
                        "change": f"weight {key} {delta:+.2f}"})
    for delta in (MAX_THRESHOLD_STEP, -MAX_THRESHOLD_STEP):
        out.append({"weights": dict(w),
                    "call_threshold": round(
                        max(0.4, min(0.85,
                                     params["call_threshold"] + delta)), 4),
                    "change": f"call_threshold {delta:+.2f}"})
    return out


def recalibrate(store: Any, calls: list[Any],
                now: float | None = None) -> dict[str, Any]:
    """One recalibration attempt over the graded call journal.
    Always journals its outcome (adjust / hold / insufficient);
    returns the record."""
    now = time.time() if now is None else now
    params = current_params(store)
    rows: list[dict[str, Any]] = []
    excluded = 0
    for c in calls:
        if c.scored is None:
            continue
        comps = getattr(c, "components", None) or {}
        if not comps:
            excluded += 1
            continue
        rows.append({"components": comps, "win": bool(c.scored),
                     "call_id": c.call_id})
    record: dict[str, Any] = {
        "at": now, "from_version": params["version"],
        "evidence": {"graded_with_components": len(rows),
                     "graded_excluded_no_components": excluded,
                     "basis": "in-sample journal backtest — the "
                              "journal holds only calls the prior "
                              "parameters admitted; steps verify "
                              "filtering quality, not discovery"},
    }
    if len(rows) < MIN_GRADED_CALLS:
        record.update({
            "status": "INSUFFICIENT_DATA",
            "detail": f"{len(rows)}/{MIN_GRADED_CALLS} graded calls "
                      f"carry components — the loop does not tune "
                      f"on thin evidence",
        })
        store.record_param_event(record)
        return record
    incumbent = _backtest(rows, params["weights"],
                          params["call_threshold"],
                          params["momentum_floor"])
    record["evidence"]["incumbent"] = incumbent
    best = None
    for cand in _candidates(params):
        bt = _backtest(rows, cand["weights"], cand["call_threshold"],
                       params["momentum_floor"])
        if bt["kept"] < MIN_KEPT:
            continue
        if bt["kept"] < MIN_VOLUME_KEPT * len(rows):
            continue
        if bt["hit_rate"] is None or incumbent["hit_rate"] is None:
            continue
        gain = bt["hit_rate"] - incumbent["hit_rate"]
        if gain >= MIN_GAIN and (best is None or gain > best["gain"]):
            best = {"gain": round(gain, 4), "backtest": bt, **cand}
    if best is None:
        record.update({
            "status": "HOLD",
            "detail": "no bounded step improves the journaled "
                      "kept-set hit rate by >= "
                      f"{MIN_GAIN:.0%} under the volume floor — "
                      "parameters unchanged",
        })
        store.record_param_event(record)
        return record
    new_params = {
        "version": params["version"] + 1,
        "weights": best["weights"],
        "call_threshold": best["call_threshold"],
        "momentum_floor": params["momentum_floor"],
        "updated_at": now,
        "basis": f"learning loop v{params['version']}→"
                 f"{params['version'] + 1}: {best['change']}",
    }
    store.save_score_params(new_params)
    record.update({
        "status": "ADJUSTED",
        "to_version": new_params["version"],
        "change": best["change"],
        "before": {"weights": params["weights"],
                   "call_threshold": params["call_threshold"]},
        "after": {"weights": new_params["weights"],
                  "call_threshold": new_params["call_threshold"]},
        "backtest": best["backtest"], "gain": best["gain"],
        "detail": f"{best['change']} — journaled kept-set hit rate "
                  f"{incumbent['hit_rate']} → {best['backtest']['hit_rate']} "
                  f"on {best['backtest']['kept']} kept calls",
    })
    store.record_param_event(record)
    return record


def render_learning(store: Any) -> str:
    params = current_params(store)
    events = store.param_events()
    w = params["weights"]
    L = [f"🎓 pump desk learning: parameter version "
         f"{params['version']} (threshold "
         f"{params['call_threshold']:.2f}; weights momentum "
         f"{w['momentum']:.2f} / smart {w['smart']:.2f} / rug-safety "
         f"{w['safety_rug']:.2f} / manip-safety {w['safety_manip']:.2f})"]
    if events:
        last = events[-1]
        L.append(f"  last recalibration: {last.get('status')} — "
                 f"{last.get('detail', '')}")
        adj = sum(1 for e in events if e.get("status") == "ADJUSTED")
        L.append(f"  journal: {len(events)} recalibration record(s), "
                 f"{adj} adjustment(s) — every one reviewable in "
                 f"score_params.jsonl")
    else:
        L.append("  no recalibration has run yet — parameters are "
                 "the P1 calibrated constants")
    return "\n".join(L)
