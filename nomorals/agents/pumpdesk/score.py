"""The de-rug gate — computed scoring, verdicts in writing.

Port of the betdesk lab's discipline to memecoin dossiers
(design §4.1/§4.4):

* components are COMPUTED numerics (rug risk, manipulation,
  momentum, smart-money confluence) — the brain may narrate a
  verdict, it never produces the numbers;
* MIN-sample floors: below them the only honest verdict is
  INSUFFICIENT_DATA, stated with what is missing;
* hard gates (a serial-rugger dev, live mint authority, a block-0
  absorption cluster) are NO_CALL no matter how hot the momentum —
  gates are arithmetic, not vibes;
* manufactured attention (bump/wash activity) DISCOUNTS the score,
  never boosts it (§4.4 rule 3);
* everything is scored in SOL / curve-% — no USD anywhere (§7.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MIN_TRADES", "MIN_UNIQUE_BUYERS", "DEFAULT_WEIGHTS",
    "CALL_THRESHOLD", "MOMENTUM_FLOOR", "ScoreVerdict", "evaluate",
]

#: Sample floors — below these, no verdict but INSUFFICIENT_DATA.
MIN_TRADES = 25
MIN_UNIQUE_BUYERS = 8

#: Composite needed (with all component floors met) for a call.
CALL_THRESHOLD = 0.62
#: Momentum must clear this on its own — a safe-but-dead token is
#: not a call.
MOMENTUM_FLOOR = 0.45

#: Composite weights (P1 calibration).  The learning loop
#: (learn.py) may move these in bounded, journaled steps; these
#: constants remain the version-1 anchor it steps FROM.
DEFAULT_WEIGHTS = {"momentum": 0.30, "smart": 0.15,
                   "safety_rug": 0.30, "safety_manip": 0.25}

#: Hard gates.
RUG_GATE = 0.70            # rug-risk component at/above -> NO_CALL
MANIP_GATE = 0.75          # manipulation component at/above -> NO_CALL
BLOCK0_GATE_SHARE = 0.40   # non-dev block-0 absorption share


@dataclass
class ScoreVerdict:
    mint: str
    verdict: str                    # INSUFFICIENT_DATA | NO_CALL | CALL_CANDIDATE
    composite: float = 0.0
    rug_risk: float = 0.0
    manipulation: float = 0.0
    momentum: float = 0.0
    smart_money: float = 0.0
    reasons: list = field(default_factory=list)
    gates: list = field(default_factory=list)
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint, "verdict": self.verdict,
            "composite": round(self.composite, 4),
            "rug_risk": round(self.rug_risk, 4),
            "manipulation": round(self.manipulation, 4),
            "momentum": round(self.momentum, 4),
            "smart_money": round(self.smart_money, 4),
            "reasons": list(self.reasons), "gates": list(self.gates),
            "evidence": self.evidence,
        }


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def evaluate(dossier: dict[str, Any],
             smart_wallets: set[str] | None = None,
             params: dict[str, Any] | None = None) -> ScoreVerdict:
    """Score one dossier.  Pure decision over the dossier dict —
    publishing lives in calls.py.  ``params`` carries the desk's
    current (possibly recalibrated) weights/threshold; the P1
    constants are the default."""
    weights = (params or {}).get("weights") or DEFAULT_WEIGHTS
    call_threshold = float((params or {}).get("call_threshold",
                                             CALL_THRESHOLD))
    momentum_floor = float((params or {}).get("momentum_floor",
                                             MOMENTUM_FLOOR))
    mint = dossier.get("mint", "")
    v = ScoreVerdict(mint=mint, verdict="NO_CALL")
    trades = dossier.get("trades") or {}
    dev = dossier.get("dev") or {}
    curve = dossier.get("curve") or {}
    auth = dossier.get("authorities") or {}
    holders = dossier.get("holders") or {}

    # ── sample floors ─────────────────────────────────────────────
    missing = []
    if int(trades.get("trades", 0)) < MIN_TRADES:
        missing.append(f"trades {trades.get('trades', 0)}/{MIN_TRADES}")
    if int(trades.get("unique_buyers", 0)) < MIN_UNIQUE_BUYERS:
        missing.append(
            f"unique buyers {trades.get('unique_buyers', 0)}/"
            f"{MIN_UNIQUE_BUYERS}")
    if not curve:
        missing.append("no curve state")
    if missing:
        v.verdict = "INSUFFICIENT_DATA"
        v.reasons.append("sample floors unmet: " + ", ".join(missing))
        v.evidence = "; ".join(v.reasons)
        return v

    # ── rug risk ──────────────────────────────────────────────────
    rug = 0.0
    if dev.get("serial_rugger"):
        rug = max(rug, 0.95)
        v.gates.append("serial-rugger dev "
                       f"({dev.get('launches', 0)} launches, rug ratio "
                       f"{dev.get('rug_ratio', 0):.0%})")
    elif int(dev.get("known_outcomes", 0)) >= 3:
        rug = max(rug, float(dev.get("rug_ratio", 0.0)) * 0.9)
    elif int(dev.get("launches", 0)) == 0:
        rug = max(rug, 0.35)   # an unseen dev is uncertainty, priced
    if auth.get("mint_authority_live"):
        rug = max(rug, 0.9)
        v.gates.append("mint authority still live — supply can inflate")
    if auth.get("freeze_authority_live"):
        rug = max(rug, 0.75)
        v.gates.append("freeze authority still live — balances can freeze")
    if trades.get("dev_sold"):
        rug = max(rug, 0.65)
        v.reasons.append(f"dev already sold {trades.get('dev_sold_sol', 0)} "
                         "SOL into the curve")
    v.rug_risk = _clamp(rug)

    # ── manipulation ──────────────────────────────────────────────
    manip = 0.0
    share = trades.get("block0_absorption_share")
    if share is not None:
        manip = max(manip, min(1.0, share / BLOCK0_GATE_SHARE))
        if share >= BLOCK0_GATE_SHARE:
            v.gates.append(f"block-0 absorption {share:.0%} of sellable "
                           "supply by non-dev wallets in the create slot "
                           "(bundle-shaped)")
    total_share = trades.get("block0_total_share")
    if total_share is not None and total_share >= 0.6:
        manip = max(manip, 0.8)
        v.gates.append(f"block-0 total absorption {total_share:.0%} — "
                       "the float was taken in the creation slot")
    alpha = float(trades.get("bump_alpha_max", 0.0))
    if alpha > 0:
        manip = max(manip, min(0.9, alpha / 6.0))
        if trades.get("bump_wallets"):
            v.reasons.append(f"wash-shaped flow: {trades['bump_wallets']} "
                             f"wallet(s) flipping with ~zero net position "
                             f"(worst alpha {alpha})")
    top10 = float(holders.get("top10_share", 0.0))
    if top10 >= 0.5:
        manip = max(manip, 0.65)
        v.reasons.append(f"holder concentration: top-10 hold {top10:.0%}")
    # virtual/real reserve divergence (P2, from the desk's live
    # finding: a curve showing ~703 SOL of virtual reserves against
    # 0 real SOL).  The virtual advance is fixed at 30 SOL; genuine
    # accumulation leaves real reserves tracking virtual minus
    # that advance.  A wide gap means apparent depth with no net
    # money behind it — churn washing through and out.
    state = curve.get("state") or {}
    vsol = float(state.get("virtual_sol_reserves") or 0) / 1e9
    rsol = float(state.get("real_sol_reserves") or 0) / 1e9
    divergence = curve.get("reserve_divergence")
    if divergence is None and vsol > 0:
        divergence = max(0.0, (vsol - 30.0 - rsol) / vsol)
    if divergence is not None and vsol >= 60.0:
        divergence = float(divergence)
        if divergence >= 0.5:
            manip = max(manip, min(0.9, 0.35 + 0.6 * divergence))
            v.reasons.append(
                f"reserve divergence: {vsol:.0f} SOL virtual vs "
                f"{rsol:.1f} SOL real in the curve — apparent depth "
                f"without net money behind it (divergence "
                f"{divergence:.0%})")
        elif divergence >= 0.25:
            manip = max(manip, 0.35)
            v.reasons.append(
                f"reserve divergence {divergence:.0%} (virtual "
                f"{vsol:.0f} SOL vs real {rsol:.1f} SOL)")
    # attention verdict (P2 verifier): manufactured attention is
    # manipulation evidence; UNKNOWN is a named gap with NO score
    # effect (fail-closed — absence is not evidence).
    att = dossier.get("attention") or {}
    att_verdict = att.get("verdict")
    if att_verdict == "MANUFACTURED":
        manip = max(manip, 0.55)
        pts = "; ".join((att.get("evidence") or {}).get(
            "manufactured_points") or [])
        v.reasons.append(f"manufactured attention: {pts}"
                         if pts else "manufactured attention "
                                     "verdict from the verifier")
    elif att_verdict == "MIXED":
        manip = max(manip, 0.35)
        v.reasons.append("attention mixed — partly manufactured "
                         "signals in the off-chain evidence")
    v.manipulation = _clamp(manip)

    # ── momentum (the graduation-predictor shape, arXiv 2602.14860:
    # SOL accumulated through the FEWEST trades + organic share) ──
    pct = float(curve.get("curve_pct", 0.0))
    vel = float(trades.get("velocity_sol_per_min", 0.0))
    per_trade = float(trades.get("sol_per_trade", 0.0))
    ratio = trades.get("buy_sell_ratio")
    mom = 0.0
    mom += _clamp(pct / 0.85) * 0.40                    # curve progress
    mom += _clamp(vel / 2.0) * 0.25                     # SOL/min velocity
    mom += _clamp(per_trade / 0.35) * 0.20              # conviction size
    if ratio is not None:
        mom += _clamp((ratio - 1.0) / 2.0) * 0.15       # buy pressure
    else:
        mom += 0.15                                     # no sells yet
    # manufactured attention discounts, never boosts (§4.4.3)
    mom *= (1.0 - 0.5 * v.manipulation)
    v.momentum = _clamp(mom)

    # ── smart-money confluence ────────────────────────────────────
    # P2: the graph's track-record confluence (dossier["smart"],
    # computed by the desk from the wallet graph) wins when
    # present — wallets counted by earned weight, cohorts once.
    # The P1 set-intersection count remains the fallback for
    # dossiers the graph has not covered.
    smart = 0.0
    smart_block = dossier.get("smart") or {}
    if isinstance(smart_block, dict) and "weighted" in smart_block:
        smart = _clamp(float(smart_block.get("weighted") or 0.0))
        hits = int(smart_block.get("hits") or 0)
        if hits:
            v.reasons.append(
                f"{hits} graph-graded smart wallet(s) bought "
                f"(track-record confluence {smart:.2f})")
    elif smart_wallets:
        hits = int((dossier.get("smart_hits") or 0))
        smart = _clamp(hits / 3.0)
        if hits:
            v.reasons.append(f"{hits} watched smart wallet(s) bought")
    v.smart_money = smart

    # ── composite + verdict ───────────────────────────────────────
    v.composite = _clamp(
        weights["momentum"] * v.momentum
        + weights["smart"] * v.smart_money
        + weights["safety_rug"] * (1.0 - v.rug_risk)
        + weights["safety_manip"] * (1.0 - v.manipulation))

    if v.gates or v.rug_risk >= RUG_GATE or v.manipulation >= MANIP_GATE:
        v.verdict = "NO_CALL"
        v.reasons.append("gate: " + ("; ".join(v.gates)
                                     if v.gates else "risk component "
                                     "over its gate"))
    elif v.composite >= call_threshold and v.momentum >= momentum_floor:
        v.verdict = "CALL_CANDIDATE"
        v.reasons.append(f"composite {v.composite:.2f} >= "
                         f"{call_threshold}, momentum {v.momentum:.2f} "
                         f">= {momentum_floor}, no gates tripped")
    else:
        v.verdict = "NO_CALL"
        v.reasons.append(f"composite {v.composite:.2f} / momentum "
                         f"{v.momentum:.2f} below call floors "
                         f"({call_threshold} / {momentum_floor})")
    v.evidence = "; ".join(v.reasons)
    return v
