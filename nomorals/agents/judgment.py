"""Judgment core — Devon trading system, Pass 3 (SHADOW).

Market-state estimation -> five competing scenarios -> evidence
evaluation -> decision or ABSTENTION, as a deterministic pure
function over the Pass 1 whole-market read, one Pass 2 setup and
the Pass 3 forecast records. No LLM anywhere in the path (owner
ruling: LLMs draft explanations elsewhere; they never decide).

SHADOW ONLY: the pass journals what judgment WOULD have done
beside what the setup lifecycle did (an armed setup's plan verdict,
a triggered setup's entry verdict). Judgment has NO order
authority and gates nothing — hard risk invariants live outside
and above it and belong to the execution pass (Pass 4).

Doctrine shapes (settled on the GoldenSwing side, carried here):
- Correlated evidence counts ONCE per correlation group; counter-
  bias evidence is weighed, never a veto (hard veto OFF).
- Absent inputs stay absent: a missing timeframe, a missing
  forecast, an unproven calibration — each is a stated fact in
  the record, never filled in.
- The forecast group is ONE evidence group among several, weighed
  ONLY when live calibration has earned trust (the E3 contract in
  agents/forecast.py: pooled high-conviction resolutions,
  n >= 20, lift >= 1.25). Anti-calibrated forecasts are recorded
  as such and still weigh nothing.
- Abstention is a first-class outcome with a reason category from
  the fixed vocabulary below — never a silent no-op.

Scenarios (five, competing): ``continuation`` (the setup's
direction reaches its first target), ``expansion`` (the move
overshoots — momentum carries beyond the target), ``reversal``
(invalidation is reached first; the other side wins),
``exhaustion`` (the move starts toward the target but stalls and
fades before it), ``chop`` (neither target nor stop — sideways
time decay). Priors and response coefficients are STATED
constants, not fitted values; the shadow journal this pass
accrues is the dataset a later fitted rung must beat.

Placement: ``agents/`` beside market_read/playbooks/forecast —
the layer from which the TA library, the finance journal and the
market-data integrations are all legal static imports.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from ..finance import trading_journal
from . import forecast as forecast_mod
from . import market_read
from .playbooks import TF_LABELS
from .setups import load_setups

_log = logging.getLogger(__name__)

#: Abstention / decision reason vocabulary (fixed).
REASON_CATEGORIES: tuple[str, ...] = (
    "invalid_geometry", "data_gap", "direction_opposition",
    "regime_chop", "exhaustion_fade", "evidence_thin",
    "expected_value", "no_setup", "quality")

#: Scenario priors (stated, not fitted) — they sum to 1.0.
PRIORS: dict[str, float] = {"continuation": 0.34, "expansion": 0.10,
                            "reversal": 0.28, "exhaustion": 0.13,
                            "chop": 0.15}
#: Log-odds response of scenario odds to weighed support.
RESPONSE = 0.45
#: How hard weighed opposition funds the reversal scenario.
OPPOSITION_RESPONSE = 1.2
#: How hard a choppy/exhausted state funds chop + exhaustion.
CHOP_RESPONSE = 1.1
#: One correlation group's vote weight. Hierarchy structure and a
#: trusted forecast are the heaviest evidence; the rest vote less.
GROUP_WEIGHTS: dict[str, float] = {
    "htf_structure": 1.5, "forecast_ensemble": 1.5,
    "setup_premise": 1.25, "geometry": 1.25,
    "committee": 1.0, "session": 0.5}
MAX_TARGET_R = 3.0    # continuation payoff cap (first target)
MIN_TARGET_R = 0.25   # ... and floor (inside the noise = no payoff)
EXPANSION_BONUS_R = 0.5
EXHAUSTION_COST_R = 0.35


@dataclass(frozen=True)
class JudgmentConfig:
    """Abstention thresholds (stated bars, in R)."""

    min_expected_r: float = 0.05
    min_confidence: float = 0.15
    chop_cost_r: float = 0.10  # friction/time cost of a chop exit


# ── market-state estimation ──────────────────────────────────────────

def market_state(read: dict[str, Any],
                 setup: dict[str, Any] | None) -> dict[str, Any]:
    """The estimated state of the market from one read.

    Everything here is read off the Pass 1 snapshot: hierarchy
    biases (structure, never an indicator), the entry timeframe's
    phase and premium/discount seat, session context and the
    read's own degraded list. Absent pieces stay None.
    """
    hier = read.get("hierarchy") or {}
    entry_tf = (setup or {}).get("entry_timeframe")
    block = (read.get("timeframes") or {}).get(entry_tf) or {} \
        if entry_tf else {}
    pd_block = block.get("premium_discount") or {}
    phases = {tf: (read.get("timeframes") or {}).get(tf, {}).get("phase")
              for tf in ("1d", "4h", "1h")}
    entry_phase = block.get("phase")
    if entry_phase in ("range", "compression", "choppy"):
        structure_class = "range"
    elif entry_phase in ("uptrend", "downtrend", "trend", "breakout") \
            or read.get("alignment") in ("aligned_up", "aligned_down"):
        structure_class = "trend"
    elif read.get("alignment") == "flat":
        structure_class = "range"
    else:
        structure_class = "mixed" if entry_phase else "unknown"
    sess = read.get("session") or {}
    return {
        "alignment": read.get("alignment"),
        "hierarchy": {level: (hier.get(level) or {}).get("bias")
                      for level, _tf in market_read.HIERARCHY},
        "phases": phases,
        "entry_timeframe": entry_tf,
        "entry_phase": entry_phase,
        "entry_status": block.get("status"),
        "premium_discount": pd_block.get("zone"),
        "premium": pd_block.get("premium"),
        "session": sess.get("name"),
        "killzone": bool(sess.get("killzone")),
        "structure_class": structure_class,
        "degraded": read.get("degraded") or [],
        "complete": bool(read.get("complete"))}


def _chop_affinity(state: dict[str, Any]) -> float:
    affinity = 0.0
    if state["structure_class"] == "range":
        affinity += 1.0
    elif state["structure_class"] == "trend":
        affinity -= 0.5
    if state.get("alignment") == "flat":
        affinity += 0.5
    if state.get("premium_discount") == "equilibrium":
        affinity += 0.25
    return affinity


# ── evidence groups ──────────────────────────────────────────────────

def _htf_structure(state: dict[str, Any], direction: int) -> dict[str, Any]:
    signs = []
    for level in ("swing", "intraday", "scalp"):
        bias = state["hierarchy"].get(level)
        if bias in (1, -1):
            signs.append(1 if int(bias) == direction else -1)
    if not signs:
        return {"stance": "absent", "strength": 0.0, "weighed": True,
                "detail": "no hierarchy bias recorded on the read"}
    net = sum(signs) / len(signs)
    stance = "support" if net > 0 else "against" if net < 0 else "neutral"
    return {"stance": stance, "strength": round(net, 4), "weighed": True,
            "detail": (f"hierarchy signs vs the setup direction: "
                       f"{signs} (swing/intraday/scalp, absent levels "
                       f"do not vote)")}


def _setup_premise(setup: dict[str, Any]) -> dict[str, Any]:
    ev = setup.get("evidence") or {}
    tags = set(ev.get("tags") or [])
    notes: list[str] = []
    score = 0.0
    if "displacement_break" in tags:
        score += 0.5
        notes.append("displacement break in the premise")
    if "sweep_head" in tags:
        score += 0.25
        notes.append("the head swept a prior extreme first")
    if "counter_swing" in tags:
        score -= 0.5
        notes.append("premise runs counter to the swing bias "
                     "(weighed as evidence, never a veto)")
    pen = ev.get("penetration_atr")
    if isinstance(pen, (int, float)):
        if float(pen) >= 0.5:
            score += 0.25
            notes.append(f"sweep penetration {float(pen):.2f} ATR")
        elif float(pen) > 0:
            notes.append(f"shallow sweep penetration "
                         f"{float(pen):.2f} ATR")
    zone = ev.get("gap") or {}
    if zone:
        notes.append("premise rests on an unmitigated zone")
        score += 0.25
    if not notes:
        return {"stance": "neutral", "strength": 0.0, "weighed": True,
                "detail": "no graded premise evidence recorded"}
    score = max(-1.0, min(1.0, score))
    stance = "support" if score > 0 else "against" if score < 0 \
        else "neutral"
    return {"stance": stance, "strength": round(score, 4),
            "weighed": True, "detail": "; ".join(notes)}


def _geometry_group(target_r: float | None) -> dict[str, Any]:
    if target_r is None:
        return {"stance": "absent", "strength": 0.0, "weighed": True,
                "detail": "no target geometry to grade"}
    if target_r >= 2.0:
        strength, word = 1.0, "pays 2R+ to the first target"
    elif target_r >= 1.5:
        strength, word = 0.5, "pays 1.5R+ to the first target"
    elif target_r >= 1.0:
        strength, word = 0.25, "pays the 1R floor to the first target"
    else:
        strength, word = -0.5, "first target pays under 1R"
    return {"stance": "support" if strength > 0 else "against",
            "strength": strength, "weighed": True,
            "detail": f"{word} ({target_r:.2f}R)"}


def _committee_group(setup: dict[str, Any]) -> dict[str, Any]:
    stance_txt = (setup.get("evidence") or {}).get("committee_stance")
    if stance_txt == "aligned":
        return {"stance": "support", "strength": 0.5, "weighed": True,
                "detail": "committee position aligned with the setup"}
    if stance_txt == "opposed":
        return {"stance": "against", "strength": -0.5, "weighed": True,
                "detail": "committee position opposed to the setup"}
    if stance_txt == "flat":
        return {"stance": "neutral", "strength": 0.0, "weighed": True,
                "detail": "committee position flat"}
    return {"stance": "absent", "strength": 0.0, "weighed": True,
            "detail": "no committee evidence on this setup"}


def _session_group(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("killzone"):
        return {"stance": "support", "strength": 0.5, "weighed": True,
                "detail": f"killzone session ({state.get('session')}) "
                          f"— volatility is expected to deliver"}
    return {"stance": "neutral", "strength": 0.0, "weighed": True,
            "detail": f"session {state.get('session') or 'unknown'}, "
                      f"no killzone"}


def _forecast_group(forecasts: dict[str, dict[str, Any]] | None,
                    setup: dict[str, Any],
                    book: dict[str, Any] | None) -> dict[str, Any]:
    """The E3 block: recorded always, weighed only when trusted."""
    direction = int(setup.get("direction") or 0)
    tag = forecast_mod.horizon_for_tf(setup.get("entry_timeframe"))
    base: dict[str, Any] = {
        "stance": "absent", "strength": 0.0, "weighed": False,
        "horizon": tag, "trust": "unproven", "forecast_id": None,
        "forecast_direction": None, "detail": ""}
    rec = (forecasts or {}).get(tag)
    if rec is None:
        base["detail"] = f"no {tag} forecast on record"
        return base
    base["forecast_id"] = rec.get("forecast_id")
    base["forecast_direction"] = rec.get("direction")
    if rec.get("direction") is None:
        base["detail"] = (f"{tag} forecast unavailable "
                          f"({rec.get('unavailable_reason')})")
        return base
    trust = forecast_mod.trust_state(book, tag, direction)
    base["trust"] = trust["state"]
    base["trust_detail"] = trust
    fdir = {"UP": 1, "DOWN": -1}.get(rec["direction"], 0)
    z = float(rec.get("conviction_z") or 0.0)
    if fdir == 0:
        stance, strength = "neutral", 0.0
        word = "the forecast is NEUTRAL at this horizon"
    elif fdir == direction and z >= 0.25:
        stance, strength = "support", min(1.0, z)
        word = (f"forecast {rec['direction']} with conviction "
                f"z={z:.2f} backs the setup direction")
    elif fdir == direction:
        stance, strength = "neutral", 0.0
        word = (f"forecast {rec['direction']} agrees but weakly "
                f"(z={z:.2f}) — no vote")
    elif z >= 0.25:
        stance, strength = "against", -min(1.0, z)
        word = (f"forecast {rec['direction']} with conviction "
                f"z={z:.2f} contradicts the setup direction")
    else:
        stance, strength = "neutral", 0.0
        word = (f"forecast {rec['direction']} disagrees but weakly "
                f"(z={z:.2f}) — no vote")
    base["stance"] = stance
    base["strength"] = round(strength, 4)
    if trust["state"] == "trusted":
        base["weighed"] = True
        base["detail"] = word + " — calibration has earned trust, " \
                               "so the group weighs"
    elif trust["state"] == "anti_calibrated":
        base["detail"] = word + " — but the feed is ANTI-calibrated " \
                               f"(lift {trust.get('lift')} on pool " \
                               f"n={trust.get('pool_n')}): recorded, " \
                               "no vote until it re-earns one"
    else:
        base["detail"] = word + f" — calibration unproven (pool " \
                               f"n={trust.get('pool_n')}, needs " \
                               f"{forecast_mod.TRUST_MIN_N} at lift " \
                               f">= {forecast_mod.TRUST_LIFT}): " \
                               "recorded, weightless"
    return base


# ── the core ─────────────────────────────────────────────────────────

def _base_record(read: dict[str, Any], state: dict[str, Any],
                 subject: dict[str, Any], *,
                 read_taken_at: float | None) -> dict[str, Any]:
    return {"kind": "judgment", "version": 1, "mode": "shadow",
            "subject": subject, "symbol": read.get("symbol"),
            "market": read.get("market"),
            "read_taken_at": read_taken_at,
            "market_state": state, "scenarios": [],
            "chosen_scenario": None, "decision": "abstain",
            "expected_r": None, "confidence": 0.0,
            "why": "", "invalidation": None, "evidence": {},
            "reason_category": "", "thresholds": {}}


def evaluate_setup(read: dict[str, Any], setup: dict[str, Any], *,
                   forecasts: dict[str, dict[str, Any]] | None = None,
                   book: dict[str, Any] | None = None,
                   config: JudgmentConfig | None = None
                   ) -> dict[str, Any]:
    """One shadow judgment for one setup. Pure and deterministic.

    ``forecasts`` is the latest-per-horizon record map from
    agents/forecast.py; ``book`` its calibration book. Both may be
    None — absence is recorded, never filled.
    """
    cfg = config or JudgmentConfig()
    state = market_state(read, setup)
    subject = {"type": "setup", "setup_id": setup.get("setup_id"),
               "playbook": setup.get("playbook"),
               "route": setup.get("route"),
               "family": setup.get("family"),
               "state": setup.get("state"),
               "lifecycle_state": setup.get("state")}
    out = _base_record(read, state, subject,
                       read_taken_at=read.get("taken_at"))
    out["thresholds"] = {"min_expected_r": cfg.min_expected_r,
                         "min_confidence": cfg.min_confidence}
    plan = setup.get("plan") or {}
    direction = int(setup.get("direction") or 0)
    entry = plan.get("entry_ref")
    stop = plan.get("invalidation")
    targets = plan.get("targets") or []
    tf = setup.get("entry_timeframe") or ""
    if direction in (1, -1) and stop is not None:
        out["invalidation"] = {
            "price": float(stop),
            "text": (f"a {TF_LABELS.get(tf, tf)} close beyond "
                     f"{float(stop):,.2f} kills the "
                     f"{setup.get('playbook')} premise")}
    # A dead data feed on the entry timeframe: nothing to judge.
    if state.get("entry_status") == "missing":
        out["reason_category"] = "data_gap"
        out["why"] = (f"the {TF_LABELS.get(tf, tf)} block of the "
                      f"read is missing — judgment does not grade "
                      f"a setup it cannot see the market for")
        return out
    stop_dist = abs(float(entry) - float(stop)) \
        if entry is not None and stop is not None else None
    if direction not in (1, -1) or entry is None or stop is None \
            or not stop_dist or not targets:
        out["reason_category"] = "invalid_geometry"
        out["why"] = ("invalid geometry: direction/entry/stop "
                      "missing, stop distance zero, or no targets "
                      "— there is no trade to judge")
        return out
    target_r_raw = abs(float(targets[0]) - float(entry)) / stop_dist
    target_r = max(MIN_TARGET_R, min(MAX_TARGET_R, target_r_raw))

    groups: dict[str, dict[str, Any]] = {
        "htf_structure": _htf_structure(state, direction),
        "setup_premise": _setup_premise(setup),
        "geometry": _geometry_group(target_r_raw),
        "committee": _committee_group(setup),
        "session": _session_group(state),
        "forecast_ensemble": _forecast_group(forecasts, setup, book)}
    support = sum(GROUP_WEIGHTS[name] * g["strength"]
                  for name, g in groups.items() if g["weighed"])
    # Opposition: the heavy groups (hierarchy structure, a trusted
    # forecast) voting against — each funds reversal once. An
    # unweighed forecast never funds anything (E3 honesty).
    opposition = sum(
        1.0 for name in ("htf_structure", "forecast_ensemble")
        if groups[name]["stance"] == "against"
        and groups[name]["weighed"])
    chop_aff = _chop_affinity(state)
    displacement = "displacement_break" in set(
        (setup.get("evidence") or {}).get("tags") or [])
    expansion_aff = (0.5 if displacement else 0.0) \
        + (0.5 if state.get("killzone") else 0.0) \
        + (0.5 if state["alignment"] in ("aligned_up", "aligned_down")
           else 0.0)
    weights = {
        "continuation": PRIORS["continuation"] * math.exp(
            RESPONSE * support),
        "expansion": PRIORS["expansion"] * math.exp(
            RESPONSE * support + CHOP_RESPONSE * 0.5 * expansion_aff),
        "reversal": PRIORS["reversal"] * math.exp(
            RESPONSE * (-support)
            + OPPOSITION_RESPONSE * opposition / 2.0),
        "exhaustion": PRIORS["exhaustion"] * math.exp(
            CHOP_RESPONSE * 0.7 * chop_aff
            + RESPONSE * (-abs(support)) * 0.5),
        "chop": PRIORS["chop"] * math.exp(
            CHOP_RESPONSE * chop_aff)}
    total = sum(weights.values())
    payoffs = {"continuation": target_r,
               "expansion": min(target_r + EXPANSION_BONUS_R,
                                MAX_TARGET_R + EXPANSION_BONUS_R),
               "reversal": -1.0, "exhaustion": -EXHAUSTION_COST_R,
               "chop": -cfg.chop_cost_r}
    scenarios = [{"name": name, "p": round(weights[name] / total, 4),
                  "payoff_r": round(payoffs[name], 3)}
                 for name in ("continuation", "expansion", "reversal",
                              "exhaustion", "chop")]
    expected_r = sum(weights[n] / total * payoffs[n] for n in weights)
    mass = sum(abs(GROUP_WEIGHTS[n] * g["strength"])
               for n, g in groups.items() if g["weighed"])
    lead = max(scenarios, key=lambda s: s["p"])
    runner_p = sorted((s["p"] for s in scenarios), reverse=True)[1]
    confidence = (mass / (mass + 2.0)) * (0.3 + 0.7 * lead["p"])
    out["scenarios"] = scenarios
    out["chosen_scenario"] = lead["name"]
    out["expected_r"] = round(expected_r, 4)
    out["confidence"] = round(confidence, 4)
    out["evidence"] = {"groups": groups,
                       "support_score": round(support, 4),
                       "opposition": opposition,
                       "chop_affinity": round(chop_aff, 3)}
    for_txt = [f"{n} {g['stance']}" for n, g in groups.items()
               if g["weighed"] and g["stance"] in ("support",
                                                   "against")]
    side = "LONG" if direction == 1 else "SHORT"
    if lead["name"] in ("continuation", "expansion") \
            and expected_r >= cfg.min_expected_r \
            and confidence >= cfg.min_confidence:
        out["decision"] = "act"
        out["reason_category"] = "quality"
        out["why"] = (
            f"{lead['name']} wins (p={lead['p']:.2f} vs next "
            f"{runner_p:.2f}): the {side} {setup.get('playbook')} "
            f"thesis is the market's most likely path — expected "
            f"{expected_r:+.2f}R at {target_r:.2f}R to the first "
            f"target on weighed evidence [{'; '.join(for_txt)}]")
    else:
        out["decision"] = "abstain"
        if lead["name"] == "reversal":
            out["reason_category"] = "direction_opposition"
            why_lead = ("reversal wins "
                        f"(p={lead['p']:.2f}): the weighed evidence "
                        "funds the other side or the invalidation "
                        "first")
        elif lead["name"] == "chop":
            out["reason_category"] = "regime_chop"
            why_lead = (f"chop wins (p={lead['p']:.2f}) in a "
                        f"{state['structure_class']} state — "
                        "neither target nor stop is the likely "
                        "first outcome")
        elif lead["name"] == "exhaustion":
            out["reason_category"] = "exhaustion_fade"
            why_lead = (f"exhaustion wins (p={lead['p']:.2f}): "
                        "the move likely starts toward the target "
                        "but stalls before it in this state")
        elif confidence < cfg.min_confidence:
            out["reason_category"] = "evidence_thin"
            why_lead = (f"{lead['name']} nominally leads "
                        f"(p={lead['p']:.2f}) but the evidence is "
                        f"too thin to judge: confidence "
                        f"{confidence:.2f} below the "
                        f"{cfg.min_confidence:.2f} bar on "
                        f"{mass:.1f} weighed mass")
        else:
            out["reason_category"] = "expected_value"
            why_lead = (f"{lead['name']} leads (p={lead['p']:.2f}) "
                        f"but expected value {expected_r:+.2f}R is "
                        f"below the {cfg.min_expected_r:+.2f}R bar")
        out["why"] = (why_lead + f" [{'; '.join(for_txt)}]"
                      if for_txt else why_lead)
    return out


def evaluate_read(read: dict[str, Any]) -> dict[str, Any]:
    """The read-level verdict: market state, no candidate.

    A read with no setup against it has nothing to take — the
    record carries the full market-state estimation and an honest
    abstention, so the journal shows judgment looked and stood
    aside, never silence mistaken for consent.
    """
    state = market_state(read, None)
    out = _base_record(read, state,
                       {"type": "read",
                        "read_taken_at": read.get("taken_at")},
                       read_taken_at=read.get("taken_at"))
    out["reason_category"] = "no_setup"
    out["why"] = ("no live setup stands against this read — "
                  "judgment records the market state and stands "
                  "aside; there is no trade to take")
    return out


# ── the shadow pass ──────────────────────────────────────────────────

def _already_judged(settings: Any, journal_path: Any
                    ) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    for rec in trading_journal.read_events(
            limit=0, event="judgment", settings=settings,
            path=journal_path):
        subj = rec.get("subject") or {}
        if subj.get("type") == "setup":
            done.add((subj.get("setup_id"), subj.get("state")))
    return done


def judgment_pass(symbol: str, market: str = "crypto", *,
                  settings: Any = None, journal_path: Any = None,
                  now: float | None = None,
                  read: dict[str, Any] | None = None,
                  forecasts: dict[str, dict[str, Any]] | None = None,
                  config: JudgmentConfig | None = None
                  ) -> dict[str, Any]:
    """One shadow judgment pass over the funnel + the read.

    Takes a fresh read (journaled, unless supplied), reads the
    latest forecasts + calibration book from the journal (unless
    supplied), judges every armed/triggered setup not yet judged
    in its current state, and journals one read-level verdict.
    Every decision lands as a ``judgment`` event beside the
    lifecycle records it judges. Nothing executes; nothing is
    scheduled.
    """
    taken = float(now if now is not None else time.time())
    sym = (symbol or "").strip().upper()
    if read is None:
        read = market_read.whole_market_read(
            sym, market=market, journal=True, settings=settings,
            journal_path=journal_path, now=taken)
    if forecasts is None:
        forecasts = forecast_mod.latest_forecasts(
            sym, settings=settings, journal_path=journal_path)
    book = forecast_mod.calibration_book(
        symbol=sym, settings=settings, journal_path=journal_path)
    done = _already_judged(settings, journal_path)
    judged: list[dict[str, Any]] = []
    for setup in load_setups(settings=settings,
                             journal_path=journal_path, symbol=sym):
        if setup["state"] not in ("armed", "triggered"):
            continue
        if (setup["setup_id"], setup["state"]) in done:
            continue
        verdict = evaluate_setup(read, setup, forecasts=forecasts,
                                 book=book, config=config)
        trading_journal.append_event(
            "judgment", settings=settings, path=journal_path,
            judged_at=taken, **verdict)
        judged.append(verdict)
    read_verdict = evaluate_read(read)
    trading_journal.append_event(
        "judgment", settings=settings, path=journal_path,
        judged_at=taken, **read_verdict)
    acts = sum(1 for v in judged if v["decision"] == "act")
    summary: dict[str, Any] = {
        "kind": "judgment_pass", "symbol": sym, "market": market,
        "taken_at": taken, "read_taken_at": read.get("taken_at"),
        "judged": judged, "read_decision": read_verdict,
        "counts": {"judged": len(judged), "act": acts,
                   "abstain": len(judged) - acts}}
    return summary


def recent_judgments(symbol: str | None = None, *, limit: int = 20,
                     settings: Any = None, journal_path: Any = None
                     ) -> list[dict[str, Any]]:
    """Journaled judgment records, newest first."""
    out: list[dict[str, Any]] = []
    for rec in trading_journal.read_events(
            limit=0, event="judgment", settings=settings,
            path=journal_path):
        if symbol is not None and rec.get("symbol") != \
                symbol.strip().upper():
            continue
        out.append(rec)
        if limit > 0 and len(out) >= limit:
            break
    return out


# ── surfaces ─────────────────────────────────────────────────────────

def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    return f"{float(value):,.2f}"


def render_judgments(records: list[dict[str, Any]], *,
                     symbol: str | None = None) -> str:
    """Human rendering of the judgment view — the surface text."""
    takes = sum(1 for r in records if r.get("decision") == "act")
    lines = [f"{symbol or 'all symbols'} shadow judgments — "
             f"{takes} would-act of {len(records)} shown"]
    if not records:
        lines.append("no judgments — judgment has not run yet")
        return "\n".join(lines)
    for rec in records:
        subj = rec.get("subject") or {}
        if subj.get("type") == "setup":
            head = (f"{subj.get('playbook')} [{subj.get('state')}] "
                    f"{rec.get('decision').upper()} "
                    f"({rec.get('chosen_scenario') or '—'})")
        else:
            head = (f"market read {rec.get('decision').upper()} "
                    f"({rec.get('reason_category')})")
        lines.append(head)
        lines.append(f"  {rec.get('why')}")
        inv = rec.get("invalidation") or {}
        if inv.get("price") is not None:
            lines.append(f"  invalidation {_fmt(inv['price'])}")
    return "\n".join(lines)


def render_pass(summary: dict[str, Any]) -> str:
    """Human rendering of one judgment pass."""
    counts = summary.get("counts") or {}
    lines = [f"judgment pass {summary.get('symbol')} — "
             f"judged {counts.get('judged', 0)}, "
             f"would-act {counts.get('act', 0)}, "
             f"abstain {counts.get('abstain', 0)}"]
    lines.append(render_judgments(
        summary.get("judged") or [], symbol=summary.get("symbol")))
    return "\n".join(lines)
