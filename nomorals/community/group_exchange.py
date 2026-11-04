"""The currency exchange (DESIGN §25): one purse, two pockets.

A member's games coins and a group's sparks are two pockets of
one money world.  This organ is the bridge between them — an
EXCHANGE, not a window:

* **Floating rate, weekly pool.**  Coins → sparks converts
  through a per-group weekly pool.  The rate lives in a stated
  band around the group's base rate and moves with the pool:
  a full pool pays the band's top, a drained pool its floor.
  The pool is an issuance CAP (converted sparks are issuance —
  the §17 monetary policy sees them), refilled each ISO week.
* **Limits that grow.**  The weekly conversion allowance is
  progression: base × level × mastery × gang-rank multipliers.
  Unused allowance expires at week end.
* **The reverse door is wide and losing.**  Sparks → coins is
  priced off the same live rate minus the spread, so a round
  trip always loses — no arbitrage loop, by construction.
* **Territory touches the exchange.**  While a gang holds
  power in the group (§23 governance capture), a bounded skim
  on conversions routes to that gang's treasury — out of the
  converted amount, never minted.

Every figure here is derived from the estate ledger (kind
``convert`` entries): pool draw, per-member usage, the lot.
Nothing is cached in a side file, so the exchange can be
rebuilt from the journal alone — the same discipline as the
estate itself.  All arithmetic is integer; every division
floors toward the house, so rounding dust can never mint.
"""
from __future__ import annotations

import time
from typing import Any

from .group_economy import GroupEconomy, _period_key

__all__ = [
    "EXCHANGE_DEFAULTS", "GroupExchange",
    "allowance_for", "forward_sparks", "min_coins_for_spark",
    "next_week_start", "rate_for", "reverse_coins",
]

#: Per-group exchange settings (the ``exchange`` settings block;
#: /gsettings exchange … is the admin grammar).  ``base_rate``
#: is sparks per 100 coins at the band's middle.
EXCHANGE_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "pool_sparks": 20_000,      # weekly issuance cap for the
                                # forward door; 0 = exchange
                                # closed that week (stated)
    "base_rate": 10,            # sparks per 100 coins (band mid)
    "spread_bps": 1_500,        # the reverse door's discount:
                                # 15% — the round trip's cost
    "skim_pct": 2,              # territory skim while a gang
                                # holds power (hard-bounded 0–10,
                                # the §23 capture range)
    "allowance_base": 400,      # weekly forward allowance at
                                # standing zero (sparks)
    "reverse_mult": 5,          # the reverse door's allowance
                                # is this × the forward one
}

#: The rate band, in basis points of the base rate: a full pool
#: pays 140%, a drained one 60%.  Stated, never hidden (§25).
BAND_FLOOR_BPS = 6_000
BAND_TOP_BPS = 14_000

#: Gang-rank steps for the allowance: the member's rank inside
#: their gang — plain member / officer / leader.
GANG_RANK_BPS = {0: 0, 1: 2_500, 2: 5_000, 3: 10_000}


def week_key(ts: float) -> str:
    return _period_key("weekly", ts)


def next_week_start(ts: float) -> float:
    """The next ISO-week boundary (Monday 00:00 local) after
    ``ts`` — when pools refill and allowances reset."""
    t = time.localtime(ts)
    days_ahead = (7 - t.tm_wday) % 7
    monday = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0,
                          0, 0, -1)) + days_ahead * 86400
    if monday <= ts:
        monday += 7 * 86400
    return monday


def rate_for(cfg: dict[str, Any], remaining: int,
             capacity: int) -> int:
    """Sparks per 100 coins at the pool's current fill.

    Deterministic in (base rate, remaining, capacity): the fuller
    the pool, the better the rate, inside the stated band."""
    base = max(1, int(cfg.get("base_rate") or 10))
    if capacity <= 0:
        return base
    fill_bps = max(0, min(10_000, int(remaining) * 10_000
                          // int(capacity)))
    mult_bps = BAND_FLOOR_BPS + (
        (BAND_TOP_BPS - BAND_FLOOR_BPS) * fill_bps) // 10_000
    return max(1, base * mult_bps // 10_000)


def forward_sparks(coins: int, rate: int) -> int:
    """Gross sparks a forward conversion pays (before skim)."""
    return int(coins) * int(rate) // 100


def min_coins_for_spark(rate: int) -> int:
    """Smallest coin amount that buys at least one spark."""
    rate = max(1, int(rate))
    return -(-100 // rate)


def reverse_coins(sparks: int, rate: int, cfg: dict[str, Any]) -> int:
    """Coins a reverse conversion pays: the fair amount at the
    SAME live rate, discounted by the spread — so converting
    out and straight back always loses the spread."""
    spread = max(0, min(9_900, int(cfg.get("spread_bps") or 0)))
    return int(sparks) * 100 * (10_000 - spread) // (
        max(1, int(rate)) * 10_000)


def allowance_for(cfg: dict[str, Any], *, level: int = 1,
                  mastery_steps: int = 0, gang_rank: int = 0) -> int:
    """The weekly forward allowance (sparks) for one member's
    standing: base grown by level (+5%/level, capped +100%),
    mastery (+10% per tier step across their games, capped
    +100%) and gang rank (member +25% · officer +50% ·
    leader +100%).  Skill buys a better financial life (§25)."""
    base = max(0, int(cfg.get("allowance_base") or 0))
    mult_bps = 10_000
    mult_bps += min(10_000, max(0, int(level) - 1) * 500)
    mult_bps += min(10_000, max(0, int(mastery_steps)) * 1_000)
    mult_bps += GANG_RANK_BPS.get(int(gang_rank), 0)
    return base * mult_bps // 10_000


class GroupExchange:
    """The exchange over one estate: pool state, quotes, legs.

    Stateless itself — every read derives from the estate
    ledger, every write goes through the estate's convert
    legs, so the journal is the whole truth."""

    def __init__(self, economy: GroupEconomy) -> None:
        self.economy = economy

    # ── derivation: the week's pool + usage from the ledger ──
    def week_state(self, group_id: str, cfg: dict[str, Any], *,
                   member: str = "",
                   now: float | None = None) -> dict[str, Any]:
        """This week's exchange picture from kind=``convert``
        entries: gross sparks issued forward, skim taken,
        sparks returned through the reverse door, and (when
        ``member`` is given) that member's usage both ways.

        Pool accounting: the forward door draws the pool by
        every spark that leaves it (member credits + skim
        legs); the reverse door restores it by the sparks
        paid in minus the skim those conversions routed to
        the territory."""
        now = now if now is not None else time.time()
        wk = week_key(now)
        fwd = skim_fwd = rev_in = skim_rev = 0
        used_fwd = used_rev = 0
        for e in self.economy._ledger(group_id):
            if e.kind != "convert" or week_key(e.ts) != wk:
                continue
            side = (e.ref or "").rsplit(":", 1)[-1]
            if e.member == "" and side == "skim" and e.amount > 0:
                skim_fwd += e.amount
            elif e.member == "" and side == "skimout" and e.amount > 0:
                skim_rev += e.amount
            elif e.amount > 0:
                fwd += e.amount
                if member and e.member == member:
                    used_fwd += e.amount
            elif e.amount < 0 and side == "out":
                rev_in += -e.amount
                if member and e.member == member:
                    used_rev += -e.amount
        drawn = fwd + skim_fwd - max(0, rev_in - skim_rev)
        capacity = max(0, int(cfg.get("pool_sparks") or 0))
        return {
            "week": wk,
            "capacity": capacity,
            "drawn": max(0, drawn),
            "remaining": max(0, capacity - drawn),
            "issued": fwd,
            "skimmed": skim_fwd + skim_rev,
            "returned": rev_in,
            "used_fwd": used_fwd,
            "used_rev": used_rev,
            "resets_at": next_week_start(now),
        }

    def quote(self, group_id: str, cfg: dict[str, Any], *,
              member: str = "", level: int = 1,
              mastery_steps: int = 0, gang_rank: int = 0,
              skim_pct: int = 0,
              now: float | None = None) -> dict[str, Any]:
        """Everything a member sees BEFORE converting: the live
        rate both ways, the pool, their allowances, the skim in
        force.  ``skim_pct`` is the caller's resolved skim (the
        surface decides whether a power gang exists)."""
        state = self.week_state(
            group_id, cfg, member=member, now=now)
        rate = rate_for(cfg, state["remaining"], state["capacity"])
        allow = allowance_for(cfg, level=level,
                              mastery_steps=mastery_steps,
                              gang_rank=gang_rank)
        rev_allow = allow * max(1, int(cfg.get("reverse_mult") or 1))
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "closed": state["capacity"] <= 0,
            "rate": rate,
            "reverse_rate_coins": reverse_coins(100, rate, cfg),
            "spread_bps": int(cfg.get("spread_bps") or 0),
            "pool": state,
            "allowance": allow,
            "allowance_left": max(0, allow - state["used_fwd"]),
            "reverse_allowance": rev_allow,
            "reverse_left": max(0, rev_allow - state["used_rev"]),
            "skim_pct": max(0, min(10, int(skim_pct))),
            "min_coins": min_coins_for_spark(rate),
            "week": state["week"],
            "resets_at": state["resets_at"],
        }
