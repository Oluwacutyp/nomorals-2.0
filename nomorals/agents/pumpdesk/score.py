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
    "MIN_TRADES", "MIN_UNIQUE_BUYERS", "ScoreVerdict", "evaluate",
]

#: Sample floors — below these, no verdict but INSUFFICIENT_DATA.
MIN_TRADES = 25
MIN_UNIQUE_BUYERS = 8

#: Composite needed (with all component floors met) for a call.
CALL_THRESHOLD = 0.62
#: Momentum must clear this on its own — a safe-but-dead token is
#: not a call.
MOMENTUM_FLOOR = 0.45

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
             smart_wallets: set[str] | None = None) -> ScoreVerdict:
    """Score one dossier.  Pure decision over the dossier dict —
    publishing lives in calls.py."""
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
    smart = 0.0
    if smart_wallets:
        # buyers are not in the dossier's trade facts by name; the
        # desk passes confluence in via the trades ledger directly.
        hits = int((dossier.get("smart_hits") or 0))
        smart = _clamp(hits / 3.0)
        if hits:
            v.reasons.append(f"{hits} watched smart wallet(s) bought")
    v.smart_money = smart

    # ── composite + verdict ───────────────────────────────────────
    v.composite = _clamp(
        0.30 * v.momentum + 0.15 * v.smart_money
        + 0.30 * (1.0 - v.rug_risk) + 0.25 * (1.0 - v.manipulation))

    if v.gates or v.rug_risk >= RUG_GATE or v.manipulation >= MANIP_GATE:
        v.verdict = "NO_CALL"
        v.reasons.append("gate: " + ("; ".join(v.gates)
                                     if v.gates else "risk component "
                                     "over its gate"))
    elif v.composite >= CALL_THRESHOLD and v.momentum >= MOMENTUM_FLOOR:
        v.verdict = "CALL_CANDIDATE"
        v.reasons.append(f"composite {v.composite:.2f} >= "
                         f"{CALL_THRESHOLD}, momentum {v.momentum:.2f} "
                         f">= {MOMENTUM_FLOOR}, no gates tripped")
    else:
        v.verdict = "NO_CALL"
        v.reasons.append(f"composite {v.composite:.2f} / momentum "
                         f"{v.momentum:.2f} below call floors "
                         f"({CALL_THRESHOLD} / {MOMENTUM_FLOOR})")
    v.evidence = "; ".join(v.reasons)
    return v
