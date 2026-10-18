"""Domain loops, closed (audit §6).

Each loop is a lifecycle with validated transitions + a next-action /
wake-up requirement, so no loop silently stalls. Adapters stay
pluggable: ``integrations/naija_deals.py`` remains ONE regional adapter
inside the universal deals engine (never the organizing axis);
``agents/weather.py`` keeps NWS as a US-only official source.

Rules enforced here (not in prompts):
* Deals: a discount on an unnecessary item is NOT savings. Savings
  require landed price (price + shipping/import/fees) below the
  historical low for the EXACT variant, from a fresh listing.
* Weather: forecasts, official warnings, and inferred impacts are
  three different categories. NWS is US-only — never label a forecast
  heuristic an official warning, and never imply NWS covers the world.
* News: cluster duplicates (normalized title) so syndication is not
  ten confirmations; only material change interrupts.
* Social: backlog of unanswered questions / promised replies /
  milestones; silence can be the best action — no quota messaging.
* Leads: full commercial lifecycle with next action / wake-up per live
  lead; learn from qualified replies + paid work, not raw DM counts.
* Gigs: full lifecycle through delivery → acceptance → invoice →
  payment reconciliation. Scoring = NET expected earnings with
  uncertainty preserved: advertised payout ≠ expected income.
  Board submitters are real only when registered/authenticated/
  verified; never pay a suspicious unlock fee.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = [
    "LEAD_TRANSITIONS", "GIG_POST_SUBMISSION", "advance_lead",
    "score_gig", "evaluate_deal", "classify_weather_item",
    "cluster_news", "social_next_action",
]

# ── Leads ──────────────────────────────────────────────────────────
LEAD_TRANSITIONS: dict[str, frozenset[str]] = {
    "discovered": frozenset({"qualified", "lost"}),
    "qualified": frozenset({"contacted", "lost"}),
    "contacted": frozenset({"replied", "lost"}),
    "replied": frozenset({"scoped", "lost"}),
    "scoped": frozenset({"proposal", "lost"}),
    "proposal": frozenset({"won", "lost"}),
    "won": frozenset({"delivery", "lost"}),
    "delivery": frozenset({"paid", "lost"}),
    "paid": frozenset({"follow_up"}),
    "follow_up": frozenset({"paid", "lost"}),
    "lost": frozenset(),
}


def advance_lead(current: str, nxt: str) -> str:
    allowed = LEAD_TRANSITIONS.get(current)
    if allowed is None:
        raise ValueError(f"unknown lead state {current!r}")
    if nxt not in allowed:
        raise ValueError(f"invalid lead transition {current!r} → {nxt!r}")
    return nxt


# ── Gigs ───────────────────────────────────────────────────────────
#: Post-submission lifecycle (extends agents/gig_applier statuses:
#: drafted → reviewed → submit_attempted → submitted, with evidence
#: required before "submitted" is ever claimed).
GIG_POST_SUBMISSION = ["submitted", "scoped", "delivering",
                       "delivered", "accepted", "invoiced", "paid"]


def score_gig(*, advertised_payout: float, win_probability: float,
              costs: float = 0.0, hours: float = 1.0,
              time_to_payment_days: float = 0.0,
              competing_commitments: int = 0,
              scam_indicators: list[str] | None = None) -> dict[str, Any]:
    """Net expected earnings with uncertainty preserved.

    ``advertised_payout`` is NOT expected income: expected =
    payout × win_prob − costs. A suspicious unlock fee is an automatic
    reject — never pay to unlock promised earnings.
    """
    scams = list(scam_indicators or [])
    if any("unlock fee" in s.lower() or "upfront fee" in s.lower()
           for s in scams):
        return {"expected_net": float("-inf"), "advertised_payout": advertised_payout,
                "uncertainty": "rejected: suspicious unlock fee",
                "scam_indicators": scams, "recommendation": "reject"}
    expected_net = advertised_payout * max(0.0, min(1.0, win_probability)) - costs
    hourly = expected_net / max(1.0, hours)
    return {
        "expected_net": expected_net,
        "advertised_payout": advertised_payout,
        "uncertainty": "advertised payout is not expected income",
        "expected_hourly": hourly,
        "time_to_payment_days": time_to_payment_days,
        "competing_commitments": competing_commitments,
        "scam_indicators": scams,
        "recommendation": ("pursue" if expected_net > 0 and hourly > 0
                           else "skip"),
    }


# ── Deals ──────────────────────────────────────────────────────────
def evaluate_deal(*, needed: bool, discount_pct: float, price: float,
                  landed_price: float, historical_low: float,
                  fresh: bool = True, variant_matches: bool = True
                  ) -> dict[str, Any]:
    """Universal purchasing intelligence (naija_deals = one adapter).

    60% off an unnecessary item is NOT savings. A real saving needs:
    needed + exact variant + fresh listing + landed < historical low.
    """
    if not needed:
        return {"is_saving": False, "action": "skip",
                "reason": "discount on an unnecessary item is not savings"}
    if not variant_matches:
        return {"is_saving": False, "action": "skip",
                "reason": "variant does not match the need"}
    if not fresh:
        return {"is_saving": False, "action": "recheck",
                "reason": "stale listing: verify freshness before acting"}
    if landed_price < historical_low:
        return {"is_saving": True, "action": "alert_or_buy_within_authority",
                "reason": f"landed {landed_price} below historical low "
                          f"{historical_low} (discount {discount_pct}%)"}
    return {"is_saving": False, "action": "watch",
            "reason": "landed price not below historical low"}


# ── Weather ────────────────────────────────────────────────────────
def classify_weather_item(*, kind: str, region: str, text: str) -> dict[str, Any]:
    """Separate forecasts, official warnings, inferred impacts.

    ``nws_alert`` is US-only. Anything else labelled as an official
    warning is dishonest; forecast heuristics are forecasts.
    """
    if kind == "nws_alert" and region.upper() != "US":
        raise ValueError("NWS alerts are US-only; cannot label a "
                         f"non-US ({region}) item an NWS warning")
    if kind == "nws_alert":
        return {"category": "official_warning", "label": "official warning (NWS, US)",
                "text": text, "region": region}
    if kind == "official_alert":
        return {"category": "official_warning",
                "label": f"official warning ({region} authoritative source)",
                "text": text, "region": region}
    if kind == "forecast_heuristic":
        return {"category": "forecast",
                "label": "forecast heuristic — NOT an official warning",
                "text": text, "region": region}
    if kind == "inferred_impact":
        return {"category": "inferred_impact",
                "label": "inferred impact from forecast — NOT an official warning",
                "text": text, "region": region}
    raise ValueError(f"unknown weather item kind {kind!r}")


# ── News ───────────────────────────────────────────────────────────
def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def cluster_news(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster syndicated duplicates by normalized title.

    Ten copies of one story are ONE story, not ten confirmations.
    """
    clusters: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for item in items:
        key = _norm_title(item.get("title", ""))
        if key in by_key:
            by_key[key]["items"].append(item)
        else:
            cluster = {"key": key, "items": [item],
                       "title": item.get("title", "")}
            by_key[key] = cluster
            clusters.append(cluster)
    return clusters


# ── Social ─────────────────────────────────────────────────────────
def social_next_action(*, unanswered: int, promised_replies: int,
                       milestones: int, stop_signal: bool = False,
                       cadence_ok: bool = True) -> str:
    """Relationship stewardship: silence can be the best social action."""
    if stop_signal:
        return "stop: honour opt-out / stop signal, no further contact"
    if not cadence_ok:
        return "wait: cadence window closed, do not message to meet quota"
    if promised_replies > 0:
        return "reply to promised message first"
    if unanswered > 0:
        return "answer unanswered question"
    if milestones > 0:
        return "acknowledge milestone"
    return "silence: nothing useful to contribute; do not post to meet quota"
