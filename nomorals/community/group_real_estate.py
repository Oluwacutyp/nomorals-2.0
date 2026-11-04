"""Group real estate: property as an owned asset class inside
the group economy (DESIGN §17 slice 2, expanded to the whole
class by §20; one purse per §25).

The class, whole:

- **Catalog**: tiers (room → tower) across named districts.
  Each district carries its own price level and a **market
  index** that moves prices — the index moves ONLY from
  journaled events (buys, sell-backs, market settlements,
  district events, a slow daily drift back toward par), under
  a stated rule, clamped to a stated band.  No random walk.
- **Ownership + rent**: buying pays the treasury (a sink with
  a stated split — the full price lands in the treasury and
  the fee leg is journaled separately on market trades).
  Rent accrues per property on a 24h clock and is claimed
  with the rent loop; rent pays **gems** (§17).  Sell-back
  returns a depreciated value in the currency the tier is
  priced in — a sink, never a faucet.
- **Gems** (💎): the premium purse.  Its ONLY faucets are
  property rent, season rewards and governance gifts executed
  through the treasury-proposal path — the organ refuses any
  other source tag, there is no sparks→gems conversion
  anywhere, and admins cannot mint outside a proposal.
  Gems buy the premium shelf (gem-priced tiers, cosmetics)
  and nothing else converts them.
- **Member market**: listings at an ask; the treasury takes a
  stated cut.  **Auctions** run through treasury escrow: the
  standing high bid is HELD in the treasury (and counted in
  the surface's reservation accounting), outbids refund in
  full, settlement pays the seller minus the cut, and a bid
  inside the anti-snipe window extends the close.
- **Mortgages**: borrow sparks from the treasury against one
  property's value at a stated LTV and flat per-period rate;
  instalments auto-debit the borrower's BANK vault on
  schedule; missed periods past a stated grace mean
  repossession into the treasury's public stock, the rest of
  the debt written off — journaled, never silent.
- **Insurance + events**: a per-property policy (premium sink)
  covers event repair bills in full; district events (boom,
  bust, repair wave, tenant trouble) fire from the tick with
  bounded effects and are journaled + announced.  Repair
  bills are sparks, priced on the property's sparks-measure
  value (gem tiers at the display weight); unpaid remainders
  become a repair debt the rent offsets first at 1 💎 = 1
  spark of debt — garnished, journaled, never invented away.
- **Public property**: the treasury itself can own buildings
  (bought through the governance proposal flow like any large
  treasury spend); their rent is collected by the tick and
  lands in the treasury as sparks, funding drops and prizes.

Money discipline: every sparks leg rides the group economy's
ledger through its own paired-move machinery (estate kinds —
the economy's ``issuance_state`` counts the member-side estate
spends as sinks, so monetary policy sees this whole class);
gems keep their own append-only ledger in this module with
the same discipline (balances recomputed, never stored).
State lives beside the economy's files, one JSON per group;
an existing-but-unreadable state file RAISES
(:class:`GroupRealEstateError`) rather than reading as empty —
the surface freezes estate writes on a corrupt store, exactly
like the sparks ledger.

Isolation: stdlib + ``nomorals.core`` + the group economy
organ.  Participants are group-local ids; "" is the treasury.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .group_economy import GroupEconomy

_log = get_logger("nomorals.community.group_real_estate")

__all__ = [
    "GroupRealEstate", "GroupRealEstateError",
    "DISTRICTS", "TIERS", "PREMIUM_SHELF", "MANAGER_LEVELS",
    "ESTATE_DEFAULTS", "GEM_SOURCES", "SEASON_GEM_REWARDS",
    "RENT_PERIOD_S", "INDEX_BASE", "PUBLIC_OWNER",
]

#: The treasury's own holdings (public property) sit under the
#: economy ledger's treasury member: "".
PUBLIC_OWNER = ""

#: Rent clock: one period is one day (§17's 24h collection
#: cooldown, as accrual — rent builds up per full period).
RENT_PERIOD_S = 86400.0

# ── the catalog (data, not code paths) ───────────────────────────

#: Districts: id → name + price level.  The same tier is worth
#: more in a hotter district (§20b); the market index moves on
#: top of the level.
DISTRICTS: dict[str, dict[str, Any]] = {
    "old_town": {"name": "Old Town", "level": 0.80},
    "market_mile": {"name": "Market Mile", "level": 1.00},
    "harbour": {"name": "Harbour View", "level": 1.25},
    "heights": {"name": "The Heights", "level": 1.60},
}

#: Tiers: id → name, base price, daily rent (gems), and the
#: currency the tier is PRICED in.  Spark tiers are the working
#: ladder; penthouse + tower are the premium shelf's exclusive
#: tiers — gems only, never buyable with sparks (§17/§25).
TIERS: dict[str, dict[str, Any]] = {
    "room": {"name": "Room", "icon": "🚪", "base": 150,
             "rent": 1, "currency": "sparks"},
    "apartment": {"name": "Apartment", "icon": "🏢", "base": 400,
                  "rent": 3, "currency": "sparks"},
    "shop": {"name": "Shop", "icon": "🏪", "base": 900,
             "rent": 7, "currency": "sparks"},
    "plaza": {"name": "Plaza", "icon": "🏬", "base": 2200,
              "rent": 16, "currency": "sparks"},
    "penthouse": {"name": "Penthouse", "icon": "🌆", "base": 60,
                  "rent": 10, "currency": "gems"},
    "tower": {"name": "Tower", "icon": "🗼", "base": 150,
              "rent": 28, "currency": "gems"},
}

#: The premium shelf: cosmetics bought with gems (status, not
#: power — the shop's house rule, in the premium currency).
PREMIUM_SHELF: dict[str, dict[str, Any]] = {
    "gold_nameplate": {"name": "Golden Nameplate", "cost": 25,
                       "blurb": "Your name rides gold on the "
                                "group boards"},
    "estate_banner": {"name": "Estate Banner", "cost": 40,
                      "blurb": "A banner over your property "
                               "portfolio"},
    "vip_shout": {"name": "VIP Shout-out", "cost": 30,
                  "blurb": "A shout-out with the premium sting"},
}

#: Property management (§20c): a per-member ladder bought with
#: sparks (a sink).  Levels add auto-collect on the tick,
#: yield, a longer accrual cap, and repair cover at the top —
#: and rent keeps collecting through jail at level 1+ (the
#: manager works while the owner can't).
MANAGER_LEVELS: dict[int, dict[str, Any]] = {
    1: {"name": "Caretaker", "cost": 200, "auto": True,
        "yield_pct": 0, "cap_days": 7, "repair_half": False},
    2: {"name": "Property Manager", "cost": 600, "auto": True,
        "yield_pct": 15, "cap_days": 14, "repair_half": False},
    3: {"name": "Estate Agency", "cost": 1500, "auto": True,
        "yield_pct": 30, "cap_days": 30, "repair_half": True},
}
BASE_CAP_DAYS = 7  # rent accrual cap with no manager

#: Group-overridable levers (settings block ``estate``).
ESTATE_DEFAULTS: dict[str, Any] = {
    "market_cut_pct": 5,        # treasury cut on member trades
    "sellback_pct": 70,         # depreciated sell-back rate
    "ownership_cap": 12,        # properties per member
    "ltv_pct": 60,              # mortgage loan-to-value
    "mortgage_rate_pct": 5,     # flat interest per period
}

#: The market index (per-mille of par).  Stated movement rule:
#: a system buy +8, a sell-back −6, a member-market or auction
#: settlement +4, a district boom +60 / bust −60, and a daily
#: drift of 1 toward par in any district that saw no trades.
#: Clamped to [600, 2500] — prices can halve or 2.5×, no more.
INDEX_BASE = 1000
INDEX_MIN = 600
INDEX_MAX = 2500
INDEX_BUY = 8
INDEX_SELL = -6
INDEX_MARKET = 4
INDEX_EVENT = 60
INDEX_DRIFT = 1

#: Mortgages: four weekly instalments; a borrower may miss
#: ``GRACE_MISSED`` periods — the next miss repossesses.
MORTGAGE_PERIOD_S = 7 * 86400.0
MORTGAGE_TERM_PERIODS = 4
GRACE_MISSED = 2

#: Insurance: premium is this % of current value per 30 days;
#: a repair wave bills this % of value per property.
INSURANCE_PREMIUM_PCT = 5
INSURANCE_DAYS = 30
REPAIR_PCT = 8

#: Auctions: a bid landing inside this window extends the
#: close to now + the window (anti-snipe).
ANTI_SNIPE_S = 120.0

#: Public property rent lands in the treasury as sparks at
#: this stated rate per gem of yield (the treasury lets its
#: buildings commercially — journaled issuance, like the
#: group-quest bonus before it).
PUBLIC_RENT_SPARK_PER_GEM = 10

#: Season rewards in gems (§17 premium source): the podium.
SEASON_GEM_REWARDS: dict[int, int] = {1: 30, 2: 20, 3: 10}

#: Gem faucets — the whole list.  ``credit_gems`` refuses any
#: other source tag; there is deliberately no "admin" source.
#: ``drop`` joined the list with §35 (system drops): the loot
#: table rolls 1–3 gems rarely, budget-capped per group per
#: day by the conflict organ's drop budget.
GEM_SOURCES = ("rent", "season", "governance", "drop")

#: Display-only wealth weight of one gem in sparks for the
#: /top board (§25's one wealth picture).  NEVER a conversion
#: rate: nothing anywhere exchanges at this number.
WEALTH_GEM_IN_SPARKS = 50


class GroupRealEstateError(RuntimeError):
    """The group's estate state exists but cannot be read.
    Raised, never swallowed into an empty state: starting over
    would silently drop every property, mortgage and gem in
    the group (the sparks ledger's discipline, applied here).
    """


def _cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(ESTATE_DEFAULTS)
    for key, value in (cfg or {}).items():
        if key in out and isinstance(value, (int, float)):
            out[key] = value
    return out


class GroupRealEstate:
    """The estate organ: catalog + index, ownership + rent,
    gems, the member market + auctions, mortgages, insurance
    + events, public property.  One instance per surface,
    bound to the group's :class:`GroupEconomy`."""

    def __init__(self, economy: GroupEconomy) -> None:
        self.economy = economy

    # ── persistence (the economy's directory + discipline) ──
    def _load_state(self, group_id: str) -> dict[str, Any]:
        path = self.economy._path(group_id, "estate")
        if path.exists():
            import json

            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise GroupRealEstateError(
                    f"estate state for {group_id} is unreadable "
                    f"({type(exc).__name__}: {exc}) — refusing "
                    f"to treat it as empty") from exc
            if not isinstance(raw, dict):
                raise GroupRealEstateError(
                    f"estate state for {group_id} is not an "
                    f"object — refusing to treat it as empty")
            return raw
        return {}

    def _save_state(self, group_id: str,
                    state: dict[str, Any]) -> None:
        self.economy._save_json(group_id, "estate", state)

    def _gems_ledger(self, group_id: str) -> list[dict[str, Any]]:
        path = self.economy._path(group_id, "gems")
        if path.exists():
            import json

            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise GroupRealEstateError(
                    f"gems ledger for {group_id} is unreadable "
                    f"({type(exc).__name__}: {exc}) — refusing "
                    f"to treat it as empty") from exc
            if not isinstance(raw, list):
                raise GroupRealEstateError(
                    f"gems ledger for {group_id} is not a list "
                    f"— refusing to treat it as empty")
            return raw
        return []

    def _save_gems(self, group_id: str,
                   rows: list[dict[str, Any]]) -> None:
        self.economy._save_json(group_id, "gems", rows)

    def health(self, group_id: str) -> dict[str, Any]:
        """Probe: both stores load (raising on corruption), and
        balances recompute.  The surface's freeze probe."""
        state = self._load_state(group_id)
        rows = self._gems_ledger(group_id)
        return {"properties": len(state.get("properties") or {}),
                "gems_entries": len(rows),
                "auctions": len(state.get("auctions") or {}),
                "mortgages": len(state.get("mortgages") or {})}

    @staticmethod
    def _journal(state: dict[str, Any], kind: str, now: float,
                 **fields: Any) -> None:
        events = state.setdefault("events", [])
        events.append({"ts": now, "kind": kind, **fields})
        del events[:-200]

    # ── gems: the premium purse ──
    def gems_balance(self, group_id: str, member: str) -> int:
        return sum(int(r.get("amount") or 0)
                   for r in self._gems_ledger(group_id)
                   if r.get("member") == member)

    def credit_gems(self, group_id: str, member: str, amount: int,
                    *, source: str, reason: str, ref: str = "",
                    now: float | None = None) -> dict[str, Any] | None:
        """Issue gems from a STATED premium source only.

        The source tag is the whole law: rent, season rewards,
        governance gifts and §35 system drops are the faucets
        that exist; any other tag is refused here, in the one
        place gems are minted, so no caller can grow another
        faucet by accident."""
        now = now if now is not None else time.time()
        amount = int(amount)
        if source not in GEM_SOURCES or amount <= 0 or not member:
            return None
        rows = self._gems_ledger(group_id)
        entry = {"id": "gem_" + new_short_id(), "member": member,
                 "kind": source, "amount": amount,
                 "reason": (reason or "")[:300],
                 "ref": (ref or "")[:120], "ts": now}
        rows.append(entry)
        self._save_gems(group_id, rows)
        return entry

    def _spend_gems(self, group_id: str, member: str, amount: int,
                    *, kind: str, reason: str, ref: str = "",
                    now: float | None = None) -> dict[str, Any] | None:
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0 or self.gems_balance(group_id, member) < amount:
            return None
        rows = self._gems_ledger(group_id)
        entry = {"id": "gem_" + new_short_id(), "member": member,
                 "kind": kind, "amount": -amount,
                 "reason": (reason or "")[:300],
                 "ref": (ref or "")[:120], "ts": now}
        rows.append(entry)
        self._save_gems(group_id, rows)
        return entry

    def refund_gems(self, group_id: str, member: str, amount: int,
                    *, reason: str, now: float | None = None
                    ) -> dict[str, Any] | None:
        """Return a member's own gems after a paired purchase
        leg failed (the arsenal's two-currency buys).  NOT a
        faucet: the entry is kind ``refund`` and only ever
        reverses a spend the same operation just made."""
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0 or not member:
            return None
        rows = self._gems_ledger(group_id)
        entry = {"id": "gem_" + new_short_id(), "member": member,
                 "kind": "refund", "amount": amount,
                 "reason": (reason or "")[:300],
                 "ref": "", "ts": now}
        rows.append(entry)
        self._save_gems(group_id, rows)
        return entry

    def grant_gems_governance(self, group_id: str, recipient: str,
                              amount: int, *, proposal_id: str,
                              now: float | None = None
                              ) -> dict[str, Any] | None:
        """A gems gift EXECUTED from an approved treasury
        proposal — the only admin-shaped path gems ever move
        through, journaled with the proposal's id."""
        return self.credit_gems(
            group_id, recipient, amount, source="governance",
            reason="governance gift", ref=proposal_id, now=now)

    # ── the market index ──
    @staticmethod
    def _index_row(state: dict[str, Any],
                   district: str) -> dict[str, Any]:
        index = state.setdefault("index", {})
        row = index.get(district)
        if row is None:
            row = {"value": INDEX_BASE,
                   "history": [{"ts": 0.0, "value": INDEX_BASE,
                                "cause": "open"}]}
            index[district] = row
        return row

    def district_index(self, group_id: str, district: str,
                       *, state: dict[str, Any] | None = None,
                       now: float | None = None) -> dict[str, Any]:
        """The district's index + its 7-day trend, read from the
        journaled history (never from a stored 'trend' claim)."""
        now = now if now is not None else time.time()
        state = state if state is not None \
            else self._load_state(group_id)
        row = self._index_row(state, district)
        value = int(row["value"])
        then = value
        for point in row.get("history") or []:
            if float(point.get("ts") or 0) <= now - 7 * 86400:
                then = int(point.get("value") or value)
            else:
                break
        trend = ((value - then) / then * 100.0) if then else 0.0
        return {"value": value, "trend_pct": round(trend, 1)}

    def _bump_index(self, state: dict[str, Any], district: str,
                    delta: int, cause: str, now: float) -> int:
        row = self._index_row(state, district)
        value = max(INDEX_MIN, min(INDEX_MAX,
                                   int(row["value"]) + int(delta)))
        row["value"] = value
        history = row.setdefault("history", [])
        history.append({"ts": now, "value": value,
                        "cause": cause})
        del history[:-400]
        return value

    # ── pricing ──
    def price(self, group_id: str, district: str, tier: str, *,
              state: dict[str, Any] | None = None) -> int:
        """Current price of a tier in a district, in the tier's
        own currency: base × district level × index."""
        spec = TIERS[tier]
        state = state if state is not None \
            else self._load_state(group_id)
        index = int(self._index_row(state, district)["value"])
        level = float(DISTRICTS[district]["level"])
        return max(1, int(round(
            float(spec["base"]) * level * index / INDEX_BASE)))

    def property_value(self, group_id: str, prop: dict[str, Any],
                       *, state: dict[str, Any] | None = None
                       ) -> int:
        """A property's current market value, in its tier's
        currency (what the system would price it at now)."""
        return self.price(group_id, str(prop["district"]),
                          str(prop["tier"]), state=state)

    def member_property_value(self, group_id: str, member: str, *,
                              state: dict[str, Any] | None = None
                              ) -> int:
        """Property wealth in SPARKS for the boards (§20f):
        spark tiers at market value; gem tiers at the stated
        DISPLAY weight (never an exchange rate)."""
        state = state if state is not None \
            else self._load_state(group_id)
        total = 0
        for prop in (state.get("properties") or {}).values():
            if prop.get("owner") != member:
                continue
            value = self.property_value(group_id, prop, state=state)
            if TIERS[str(prop["tier"])]["currency"] == "gems":
                value *= WEALTH_GEM_IN_SPARKS
            total += value
        return total

    def top_landlord(self, group_id: str, *,
                     state: dict[str, Any] | None = None
                     ) -> tuple[str, int] | None:
        """The member holding the most property wealth (sparks
        measure), or None when nobody owns anything — the
        season rollover crowns them beside the earners (§20f)."""
        state = state if state is not None \
            else self._load_state(group_id)
        owners = {str(p.get("owner")) for p in
                  (state.get("properties") or {}).values()
                  if p.get("owner")}
        best: tuple[str, int] | None = None
        for owner in owners:
            value = self.member_property_value(group_id, owner,
                                               state=state)
            if value > 0 and (best is None or value > best[1]):
                best = (owner, value)
        return best

    def catalog(self, group_id: str, *,
                now: float | None = None) -> list[dict[str, Any]]:
        """The whole shelf: every district with its index +
        trend, every tier priced in it (§20b)."""
        now = now if now is not None else time.time()
        state = self._load_state(group_id)
        out = []
        for district, spec in DISTRICTS.items():
            idx = self.district_index(group_id, district,
                                      state=state, now=now)
            tiers = {tier: self.price(group_id, district, tier,
                                      state=state)
                     for tier in TIERS}
            out.append({"district": district, **spec, **idx,
                        "tiers": tiers})
        return out

    # ── ownership ──
    def _new_property(self, state: dict[str, Any], *, owner: str,
                      district: str, tier: str, price_paid: int,
                      now: float) -> dict[str, Any]:
        pid = "prop-" + new_short_id()
        prop = {"id": pid, "owner": owner, "district": district,
                "tier": tier, "paid": int(price_paid),
                "currency": TIERS[tier]["currency"],
                "anchor": now, "bought_at": now,
                "listed": "", "auction": "", "mortgage": "",
                "insured_until": 0.0, "repair_debt": 0}
        state.setdefault("properties", {})[pid] = prop
        return prop

    def portfolio(self, group_id: str, member: str, *,
                  now: float | None = None) -> list[dict[str, Any]]:
        """A member's properties, enriched with live values and
        the rent standing on each clock."""
        now = now if now is not None else time.time()
        state = self._load_state(group_id)
        out = []
        for prop in (state.get("properties") or {}).values():
            if prop.get("owner") != member:
                continue
            due = self._rent_due(state, prop, now)
            out.append({**prop,
                        "value": self.property_value(
                            group_id, prop, state=state),
                        "rent_periods": due[0],
                        "rent_gems": due[1]})
        return sorted(out, key=lambda p: str(p["bought_at"]))

    def buy(self, group_id: str, member: str, district: str,
            tier: str, *, cfg: dict[str, Any] | None = None,
            now: float | None = None) -> dict[str, Any]:
        """Buy from the system at the index price.  Spark tiers:
        the wallet pays the treasury in full (the sink §17
        names; the split shows on member trades, where a cut
        leg is journaled apart).  Gem tiers: the gems purse
        pays, and gems paid to the system are spent, not held —
        there is no gems treasury."""
        now = now if now is not None else time.time()
        cfg = _cfg(cfg)
        if district not in DISTRICTS or tier not in TIERS:
            return {"ok": False, "reason": "catalog"}
        state = self._load_state(group_id)
        owned = sum(1 for p in (state.get("properties") or {})
                    .values() if p.get("owner") == member)
        if owned >= int(cfg["ownership_cap"]):
            return {"ok": False, "reason": "cap",
                    "cap": int(cfg["ownership_cap"])}
        price = self.price(group_id, district, tier, state=state)
        econ = self.economy
        if TIERS[tier]["currency"] == "gems":
            if self._spend_gems(
                    group_id, member, price, kind="spend",
                    reason=f"bought {TIERS[tier]['name']} in "
                           f"{DISTRICTS[district]['name']}",
                    ref=f"tier:{tier}", now=now) is None:
                return {"ok": False, "reason": "gems",
                        "price": price}
        else:
            if not econ._move(group_id, member, PUBLIC_OWNER,
                              price, "estate_buy",
                              f"bought {TIERS[tier]['name']} in "
                              f"{DISTRICTS[district]['name']}",
                              now):
                return {"ok": False, "reason": "funds",
                        "price": price}
        prop = self._new_property(state, owner=member,
                                  district=district, tier=tier,
                                  price_paid=price, now=now)
        self._bump_index(state, district, INDEX_BUY, "buy", now)
        self._journal(state, "buy", now, member=member,
                      property=prop["id"], district=district,
                      tier=tier, price=price)
        self._save_state(group_id, state)
        return {"ok": True, "property": prop, "price": price,
                "index": self.district_index(
                    group_id, district, state=state, now=now)}

    def buy_public(self, group_id: str, district: str, tier: str,
                   *, by: str = "",
                   now: float | None = None) -> dict[str, Any]:
        """The treasury buys a building into public stock: a
        single journaled burn leg out of the treasury (the
        price leaves circulation to the system, exactly the
        mirror of a quest bonus entering it).  The caller (the
        surface) checks reservation-aware availability and, for
        non-operators, routes the spend through governance —
        this method is the execution, not the permission."""
        now = now if now is not None else time.time()
        if district not in DISTRICTS or tier not in TIERS:
            return {"ok": False, "reason": "catalog"}
        if TIERS[tier]["currency"] == "gems":
            return {"ok": False, "reason": "currency"}
        state = self._load_state(group_id)
        price = self.price(group_id, district, tier, state=state)
        econ = self.economy
        if econ.treasury_balance(group_id) < price:
            return {"ok": False, "reason": "funds",
                    "price": price}
        econ._append(group_id, econ._new_entry(
            group_id, PUBLIC_OWNER, "estate_buy", -price,
            f"public property: {TIERS[tier]['name']} in "
            f"{DISTRICTS[district]['name']} (by {by})",
            ref=f"tier:{tier}", ts=now))
        prop = self._new_property(state, owner=PUBLIC_OWNER,
                                  district=district, tier=tier,
                                  price_paid=price, now=now)
        self._bump_index(state, district, INDEX_BUY,
                         "public_buy", now)
        self._journal(state, "public_buy", now, by=by,
                      property=prop["id"], district=district,
                      tier=tier, price=price)
        self._save_state(group_id, state)
        return {"ok": True, "property": prop, "price": price}

    def sell(self, group_id: str, member: str, pid: str, *,
             cfg: dict[str, Any] | None = None,
             now: float | None = None) -> dict[str, Any]:
        """Sell back to the system at the depreciated rate, in
        the tier's own currency (spark tiers → sparks per §17;
        gem tiers → gems, at the same depreciation — the round
        trip always loses, so neither direction is a faucet)."""
        now = now if now is not None else time.time()
        cfg = _cfg(cfg)
        state = self._load_state(group_id)
        prop = (state.get("properties") or {}).get(pid)
        if prop is None or prop.get("owner") != member:
            return {"ok": False, "reason": "not_found"}
        locked = self._locked_reason(prop)
        if locked:
            return {"ok": False, "reason": locked}
        value = self.property_value(group_id, prop, state=state)
        proceeds = max(1, int(value * int(cfg["sellback_pct"])
                              // 100))
        econ = self.economy
        if prop["currency"] == "gems":
            # A depreciated return of the member's own asset —
            # its own journaled kind, NOT one of the faucets:
            # credit_gems (rent/season/governance) is never the
            # road a sale takes.
            rows = self._gems_ledger(group_id)
            rows.append({"id": "gem_" + new_short_id(),
                         "member": member, "kind": "sellback",
                         "amount": proceeds,
                         "reason": f"sold back "
                                   f"{TIERS[prop['tier']]['name']}",
                         "ref": f"property:{pid}", "ts": now})
            self._save_gems(group_id, rows)
        else:
            if not econ._move(group_id, PUBLIC_OWNER, member,
                              proceeds, "estate_sell",
                              f"sold back "
                              f"{TIERS[prop['tier']]['name']}",
                              now):
                return {"ok": False, "reason": "funds"}
        district = str(prop["district"])
        del state["properties"][pid]
        self._bump_index(state, district, INDEX_SELL, "sell", now)
        self._journal(state, "sell", now, member=member,
                      property=pid, district=district,
                      proceeds=proceeds,
                      currency=prop["currency"])
        self._save_state(group_id, state)
        return {"ok": True, "proceeds": proceeds,
                "currency": prop["currency"], "value": value}

    @staticmethod
    def _locked_reason(prop: dict[str, Any]) -> str | None:
        if prop.get("listed"):
            return "listed"
        if prop.get("auction"):
            return "auction"
        if prop.get("mortgage"):
            return "mortgaged"
        return None

    # ── rent ──
    def manager_level(self, group_id: str, member: str, *,
                      state: dict[str, Any] | None = None) -> int:
        state = state if state is not None \
            else self._load_state(group_id)
        return int((state.get("managers") or {}).get(member) or 0)

    def _rent_due(self, state: dict[str, Any],
                  prop: dict[str, Any],
                  now: float) -> tuple[int, int]:
        """(full periods standing, gems they pay) — accrual is
        capped by the owner's manager level; a district tenant
        event cuts the yield while it runs (insured or not —
        insurance covers bills, not markets)."""
        level = int((state.get("managers") or {})
                    .get(str(prop["owner"])) or 0)
        cap = (MANAGER_LEVELS[level]["cap_days"] if level
               else BASE_CAP_DAYS)
        periods = int((now - float(prop.get("anchor") or now))
                      // RENT_PERIOD_S)
        periods = max(0, min(cap, periods))
        if periods <= 0:
            return 0, 0
        gems = periods * int(TIERS[str(prop["tier"])]["rent"])
        if level:
            gems += gems * int(
                MANAGER_LEVELS[level]["yield_pct"]) // 100
        mod = (state.get("yield_mods") or {}).get(
            str(prop["district"])) or {}
        if float(mod.get("until") or 0) > now:
            gems = gems * (100 + int(mod.get("pct") or 0)) // 100
        return periods, max(0, gems)

    def collect_rent(self, group_id: str, member: str, *,
                     now: float | None = None) -> dict[str, Any]:
        """Collect every standing period across the member's
        portfolio.  Repair debt is offset FIRST (garnished,
        journaled), the rest lands as gems — source ``rent``,
        one of the three faucets that exist."""
        now = now if now is not None else time.time()
        state = self._load_state(group_id)
        total = offset = 0
        lines: list[dict[str, Any]] = []
        for prop in (state.get("properties") or {}).values():
            if prop.get("owner") != member:
                continue
            periods, gems = self._rent_due(state, prop, now)
            if periods <= 0:
                continue
            prop["anchor"] = float(prop["anchor"]) \
                + periods * RENT_PERIOD_S
            debt = int(prop.get("repair_debt") or 0)
            garnished = min(debt, gems)
            if garnished:
                prop["repair_debt"] = debt - garnished
                gems -= garnished
                offset += garnished
                self._journal(state, "repair_offset", now,
                              member=member, property=prop["id"],
                              amount=garnished)
            total += gems
            lines.append({"property": prop["id"],
                          "tier": prop["tier"],
                          "periods": periods, "gems": gems,
                          "garnished": garnished})
        if total > 0:
            self.credit_gems(group_id, member, total,
                             source="rent", reason="property rent",
                             ref="rent", now=now)
        if lines:
            self._journal(state, "rent", now, member=member,
                          gems=total, offset=offset,
                          properties=len(lines))
            self._save_state(group_id, state)
        return {"ok": True, "gems": total, "offset": offset,
                "lines": lines,
                "balance": self.gems_balance(group_id, member)}

    def buy_manager(self, group_id: str, member: str, *,
                    now: float | None = None) -> dict[str, Any]:
        """Climb the management ladder one rung (a sparks sink,
        §20c).  Levels are bought in order — no skipping."""
        now = now if now is not None else time.time()
        state = self._load_state(group_id)
        level = self.manager_level(group_id, member, state=state)
        nxt = MANAGER_LEVELS.get(level + 1)
        if nxt is None:
            return {"ok": False, "reason": "top"}
        if not self.economy._move(
                group_id, member, PUBLIC_OWNER, int(nxt["cost"]),
                "estate_manager",
                f"property management: {nxt['name']}", now):
            return {"ok": False, "reason": "funds",
                    "cost": int(nxt["cost"])}
        state.setdefault("managers", {})[member] = level + 1
        self._journal(state, "manager", now, member=member,
                      level=level + 1)
        self._save_state(group_id, state)
        return {"ok": True, "level": level + 1, **nxt}

    # ── the premium shelf ──
    def buy_premium(self, group_id: str, member: str, item_id: str,
                    *, now: float | None = None
                    ) -> dict[str, Any] | None:
        """Buy a premium-shelf cosmetic with gems.  Mirrors the
        status shop's grant shape; the grant is what the chat
        layer acts on.  Status only — never power."""
        now = now if now is not None else time.time()
        item = PREMIUM_SHELF.get(item_id)
        if item is None:
            return None
        entry = self._spend_gems(
            group_id, member, int(item["cost"]), kind="spend",
            reason=f"premium shelf: {item['name']}",
            ref=f"premium:{item_id}", now=now)
        if entry is None:
            return None
        state = self._load_state(group_id)
        owned = state.setdefault("premium_owned", {}) \
            .setdefault(member, [])
        if item_id not in owned:
            owned.append(item_id)
        self._save_state(group_id, state)
        return {"item": item_id, "name": item["name"],
                "cost": item["cost"], "member": member,
                "ledger_id": entry["id"], "granted_at": now,
                "currency": "gems"}

    # ── the member market ──
    def list_property(self, group_id: str, member: str, pid: str,
                      ask: int, *, now: float | None = None
                      ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        ask = int(ask)
        state = self._load_state(group_id)
        prop = (state.get("properties") or {}).get(pid)
        if prop is None or prop.get("owner") != member:
            return {"ok": False, "reason": "not_found"}
        if prop["currency"] == "gems":
            return {"ok": False, "reason": "currency"}
        locked = self._locked_reason(prop)
        if locked:
            return {"ok": False, "reason": locked}
        if ask <= 0:
            return {"ok": False, "reason": "shape"}
        lid = "lst-" + new_short_id()
        state.setdefault("listings", {})[lid] = {
            "id": lid, "property": pid, "seller": member,
            "ask": ask, "at": now, "open": True}
        prop["listed"] = lid
        self._journal(state, "list", now, member=member,
                      property=pid, ask=ask, listing=lid)
        self._save_state(group_id, state)
        return {"ok": True, "listing": lid, "ask": ask}

    def unlist(self, group_id: str, member: str, pid: str, *,
               now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        state = self._load_state(group_id)
        prop = (state.get("properties") or {}).get(pid)
        if prop is None or prop.get("owner") != member \
                or not prop.get("listed"):
            return {"ok": False, "reason": "not_found"}
        listing = (state.get("listings") or {}).get(
            str(prop["listed"]))
        if listing:
            listing["open"] = False
        prop["listed"] = ""
        self._journal(state, "unlist", now, member=member,
                      property=pid)
        self._save_state(group_id, state)
        return {"ok": True}

    def market(self, group_id: str) -> list[dict[str, Any]]:
        state = self._load_state(group_id)
        out = []
        for listing in (state.get("listings") or {}).values():
            if not listing.get("open"):
                continue
            prop = (state.get("properties") or {}).get(
                str(listing["property"]))
            if prop is None:
                continue
            out.append({**listing, "district": prop["district"],
                        "tier": prop["tier"],
                        "value": self.property_value(
                            group_id, prop, state=state)})
        return sorted(out, key=lambda l: float(l["at"]))

    def buy_listing(self, group_id: str, buyer: str, lid: str, *,
                    cfg: dict[str, Any] | None = None,
                    now: float | None = None) -> dict[str, Any]:
        """Buy a listed property: the buyer pays the ask; the
        treasury's cut leg is journaled apart from the seller's
        net leg (the stated split).  The rent clock follows the
        property — its anchor does not reset on transfer."""
        now = now if now is not None else time.time()
        cfg = _cfg(cfg)
        state = self._load_state(group_id)
        listing = (state.get("listings") or {}).get(lid)
        if not listing or not listing.get("open"):
            return {"ok": False, "reason": "not_found"}
        seller = str(listing["seller"])
        if buyer == seller:
            return {"ok": False, "reason": "own"}
        prop = (state.get("properties") or {}).get(
            str(listing["property"]))
        if prop is None or prop.get("owner") != seller:
            return {"ok": False, "reason": "not_found"}
        owned = sum(1 for p in (state.get("properties") or {})
                    .values() if p.get("owner") == buyer)
        if owned >= int(cfg["ownership_cap"]):
            return {"ok": False, "reason": "cap"}
        ask = int(listing["ask"])
        econ = self.economy
        if econ.balance(group_id, buyer) < ask:
            return {"ok": False, "reason": "funds", "ask": ask}
        cut = ask * int(cfg["market_cut_pct"]) // 100
        net = ask - cut
        if net and not econ._move(
                group_id, buyer, seller, net, "estate_market",
                f"bought {TIERS[prop['tier']]['name']} from "
                f"{seller} (market)", now):
            return {"ok": False, "reason": "funds"}
        if cut and not econ._move(
                group_id, buyer, PUBLIC_OWNER, cut, "estate_fee",
                "market transaction cut", now):
            return {"ok": False, "reason": "funds"}
        prop["owner"] = buyer
        prop["listed"] = ""
        listing["open"] = False
        listing["sold_to"] = buyer
        listing["sold_at"] = now
        district = str(prop["district"])
        self._bump_index(state, district, INDEX_MARKET,
                         "market_sale", now)
        self._journal(state, "market_buy", now, member=buyer,
                      property=prop["id"], listing=lid, ask=ask,
                      cut=cut, seller=seller)
        self._save_state(group_id, state)
        return {"ok": True, "property": prop["id"], "ask": ask,
                "cut": cut, "net": net, "seller": seller}

    # ── auctions (treasury escrow) ──
    def start_auction(self, group_id: str, member: str, pid: str,
                      start: int, hours: float, *,
                      now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        start = int(start)
        state = self._load_state(group_id)
        prop = (state.get("properties") or {}).get(pid)
        if prop is None or prop.get("owner") != member:
            return {"ok": False, "reason": "not_found"}
        if prop["currency"] == "gems":
            return {"ok": False, "reason": "currency"}
        locked = self._locked_reason(prop)
        if locked:
            return {"ok": False, "reason": locked}
        if start <= 0 or hours <= 0:
            return {"ok": False, "reason": "shape"}
        aid = "auc-" + new_short_id()
        state.setdefault("auctions", {})[aid] = {
            "id": aid, "property": pid, "seller": member,
            "start": start, "high": 0, "high_bidder": "",
            "opened": now, "ends": now + float(hours) * 3600,
            "open": True, "settled": None}
        prop["auction"] = aid
        self._journal(state, "auction_open", now, member=member,
                      property=pid, auction=aid, start=start)
        self._save_state(group_id, state)
        return {"ok": True, "auction": aid,
                "ends": now + float(hours) * 3600}

    def auction_reserved(self, group_id: str = "", *,
                         state: dict[str, Any] | None = None
                         ) -> int:
        """Sparks currently HELD as standing high bids — the
        surface's reservation accounting counts these against
        the treasury exactly like open drops."""
        state = state if state is not None \
            else self._load_state(group_id)
        return sum(int(a.get("high") or 0)
                   for a in (state.get("auctions") or {}).values()
                   if a.get("open") and a.get("high_bidder"))

    def place_bid(self, group_id: str, bidder: str, aid: str,
                  amount: int, *, now: float | None = None
                  ) -> dict[str, Any]:
        """Place a bid: the new high bid is HELD in the treasury
        (estate_escrow) and the previous holder is refunded in
        full (estate_refund) — the escrow never keeps a loser's
        sparks.  A bid inside the anti-snipe window extends the
        close."""
        now = now if now is not None else time.time()
        amount = int(amount)
        self.settle_due(group_id, now=now)
        state = self._load_state(group_id)
        auction = (state.get("auctions") or {}).get(aid)
        if not auction or not auction.get("open"):
            return {"ok": False, "reason": "not_found"}
        if bidder == auction.get("seller"):
            return {"ok": False, "reason": "own"}
        if bidder == auction.get("high_bidder"):
            return {"ok": False, "reason": "leading"}
        floor = int(auction["start"]) if not auction["high_bidder"] \
            else int(auction["high"]) + 1
        if amount < floor:
            return {"ok": False, "reason": "low", "floor": floor}
        econ = self.economy
        if econ.balance(group_id, bidder) < amount:
            return {"ok": False, "reason": "funds"}
        prior, prior_high = str(auction.get("high_bidder") or ""), \
            int(auction.get("high") or 0)
        if not econ._move(group_id, bidder, PUBLIC_OWNER, amount,
                          "estate_escrow",
                          f"auction bid hold {aid}", now):
            return {"ok": False, "reason": "funds"}
        if prior and prior_high:
            econ._move(group_id, PUBLIC_OWNER, prior, prior_high,
                       "estate_refund", f"outbid refund {aid}",
                       now)
        auction["high"] = amount
        auction["high_bidder"] = bidder
        extended = False
        if float(auction["ends"]) - now < ANTI_SNIPE_S:
            auction["ends"] = now + ANTI_SNIPE_S
            extended = True
        self._journal(state, "bid", now, member=bidder,
                      auction=aid, amount=amount,
                      extended=extended)
        self._save_state(group_id, state)
        return {"ok": True, "high": amount, "extended": extended,
                "ends": float(auction["ends"]),
                "refunded": prior_high if prior else 0}

    def settle_due(self, group_id: str, *,
                   cfg: dict[str, Any] | None = None,
                   now: float | None = None) -> list[dict[str, Any]]:
        """Close every auction past its end: the winner takes
        the property, the seller is paid the high bid minus the
        treasury cut (the cut simply stays in the treasury —
        the escrow legs already journaled it there), unsold
        auctions hand the property back unlocked."""
        now = now if now is not None else time.time()
        cfg = _cfg(cfg)
        state = self._load_state(group_id)
        settled: list[dict[str, Any]] = []
        changed = False
        for auction in (state.get("auctions") or {}).values():
            if not auction.get("open") \
                    or float(auction.get("ends") or 0) > now:
                continue
            auction["open"] = False
            changed = True
            prop = (state.get("properties") or {}).get(
                str(auction["property"]))
            if prop is not None:
                prop["auction"] = ""
            winner = str(auction.get("high_bidder") or "")
            high = int(auction.get("high") or 0)
            if winner and prop is not None:
                cut = high * int(cfg["market_cut_pct"]) // 100
                net = high - cut
                if net:
                    self.economy._move(
                        group_id, PUBLIC_OWNER,
                        str(auction["seller"]), net,
                        "estate_sale",
                        f"auction settlement {auction['id']}",
                        now)
                prop["owner"] = winner
                auction["settled"] = {"winner": winner,
                                      "price": high, "cut": cut,
                                      "at": now}
                self._bump_index(state, str(prop["district"]),
                                 INDEX_MARKET, "auction", now)
                settled.append({"auction": auction["id"],
                                "winner": winner, "price": high,
                                "cut": cut,
                                "seller": auction["seller"],
                                "property": prop["id"],
                                "tier": prop["tier"]})
                self._journal(state, "auction_settle", now,
                              auction=auction["id"], winner=winner,
                              price=high, cut=cut)
            else:
                auction["settled"] = {"winner": "", "at": now}
                self._journal(state, "auction_unsold", now,
                              auction=auction["id"])
                settled.append({"auction": auction["id"],
                                "winner": "", "price": 0, "cut": 0,
                                "seller": auction["seller"],
                                "property": auction["property"],
                                "tier": prop["tier"] if prop
                                else ""})
        if changed:
            self._save_state(group_id, state)
        return settled

    def open_auctions(self, group_id: str, *,
                      cfg: dict[str, Any] | None = None,
                      now: float | None = None
                      ) -> list[dict[str, Any]]:
        now = now if now is not None else time.time()
        self.settle_due(group_id, cfg=cfg, now=now)
        state = self._load_state(group_id)
        out = []
        for auction in (state.get("auctions") or {}).values():
            if not auction.get("open"):
                continue
            prop = (state.get("properties") or {}).get(
                str(auction["property"]))
            if prop is None:
                continue
            out.append({**auction, "district": prop["district"],
                        "tier": prop["tier"]})
        return sorted(out, key=lambda a: float(a["ends"]))

    # ── mortgages ──
    def borrow(self, group_id: str, member: str, pid: str,
               amount: int, *, cfg: dict[str, Any] | None = None,
               now: float | None = None) -> dict[str, Any]:
        """Borrow sparks from the treasury against ONE
        property: principal ≤ LTV × its current value; flat
        interest per period on the original principal is added
        up front, so the debt is one legible number.  The
        property locks until the debt clears."""
        now = now if now is not None else time.time()
        cfg = _cfg(cfg)
        amount = int(amount)
        state = self._load_state(group_id)
        prop = (state.get("properties") or {}).get(pid)
        if prop is None or prop.get("owner") != member:
            return {"ok": False, "reason": "not_found"}
        if prop["currency"] == "gems":
            return {"ok": False, "reason": "currency"}
        locked = self._locked_reason(prop)
        if locked:
            return {"ok": False, "reason": locked}
        if amount <= 0:
            return {"ok": False, "reason": "shape"}
        value = self.property_value(group_id, prop, state=state)
        max_loan = value * int(cfg["ltv_pct"]) // 100
        if amount > max_loan:
            return {"ok": False, "reason": "ltv", "max": max_loan,
                    "value": value}
        econ = self.economy
        if econ.treasury_balance(group_id) \
                - self.auction_reserved(state=state) < amount:
            return {"ok": False, "reason": "funds"}
        if not econ._move(group_id, PUBLIC_OWNER, member, amount,
                          "estate_loan",
                          f"mortgage on {prop['id']}", now):
            return {"ok": False, "reason": "funds"}
        interest = math.ceil(
            amount * int(cfg["mortgage_rate_pct"]) / 100) \
            * MORTGAGE_TERM_PERIODS
        total = amount + interest
        mid = "mtg-" + new_short_id()
        mortgage = {"id": mid, "property": pid,
                    "borrower": member, "principal": amount,
                    "interest": interest, "remaining": total,
                    "paid": 0,
                    "installment": math.ceil(
                        total / MORTGAGE_TERM_PERIODS),
                    "periods_left": MORTGAGE_TERM_PERIODS,
                    "next_due": now + MORTGAGE_PERIOD_S,
                    "missed": 0, "open": True, "at": now,
                    "defaulted": False}
        state.setdefault("mortgages", {})[mid] = mortgage
        prop["mortgage"] = mid
        self._journal(state, "mortgage_open", now, member=member,
                      property=pid, mortgage=mid, amount=amount,
                      interest=interest)
        self._save_state(group_id, state)
        return {"ok": True, "mortgage": mortgage}

    def repay(self, group_id: str, member: str, mid: str,
              amount: int, *, now: float | None = None
              ) -> dict[str, Any]:
        """Voluntary repayment from the WALLET (the schedule
        debits the bank; a member can always pay early by
        hand).  Clearing the debt unlocks the property."""
        now = now if now is not None else time.time()
        amount = int(amount)
        state = self._load_state(group_id)
        mortgage = (state.get("mortgages") or {}).get(mid)
        if not mortgage or not mortgage.get("open") \
                or mortgage.get("borrower") != member:
            return {"ok": False, "reason": "not_found"}
        pay = min(amount, int(mortgage["remaining"]))
        if pay <= 0:
            return {"ok": False, "reason": "shape"}
        if not self.economy._move(group_id, member, PUBLIC_OWNER,
                                  pay, "estate_repay",
                                  f"mortgage repayment {mid}",
                                  now):
            return {"ok": False, "reason": "funds"}
        mortgage["remaining"] -= pay
        mortgage["paid"] += pay
        closed = mortgage["remaining"] <= 0
        if closed:
            mortgage["open"] = False
            prop = (state.get("properties") or {}).get(
                str(mortgage["property"]))
            if prop is not None:
                prop["mortgage"] = ""
        self._journal(state, "mortgage_repay", now, member=member,
                      mortgage=mid, amount=pay, closed=closed)
        self._save_state(group_id, state)
        return {"ok": True, "paid": pay,
                "remaining": int(mortgage["remaining"]),
                "closed": closed}

    def mortgages(self, group_id: str, member: str = ""
                  ) -> list[dict[str, Any]]:
        state = self._load_state(group_id)
        out = [m for m in (state.get("mortgages") or {}).values()
               if m.get("open")
               and (not member or m.get("borrower") == member)]
        return sorted(out, key=lambda m: float(m["next_due"]))

    def _mortgage_cycle(self, state: dict[str, Any], group_id: str,
                        now: float) -> list[dict[str, Any]]:
        """Due instalments debit the borrower's BANK vault.
        A short bank means a missed period (journaled); after
        the stated grace, the property is repossessed into
        public stock and the remaining debt written off —
        journaled as a write-off, never quietly dropped."""
        notices: list[dict[str, Any]] = []
        econ = self.economy
        for mortgage in (state.get("mortgages") or {}).values():
            if not mortgage.get("open") \
                    or float(mortgage.get("next_due") or 0) > now:
                continue
            borrower = str(mortgage["borrower"])
            due = min(int(mortgage["installment"]),
                      int(mortgage["remaining"]))
            bank = econ.bank_member(borrower)
            if due > 0 and econ.balance(group_id, bank) >= due \
                    and econ._move(group_id, bank, PUBLIC_OWNER,
                                   due, "estate_repay",
                                   f"mortgage instalment "
                                   f"{mortgage['id']}", now):
                mortgage["remaining"] -= due
                mortgage["paid"] += due
                mortgage["periods_left"] = max(
                    0, int(mortgage["periods_left"]) - 1)
                mortgage["next_due"] = float(
                    mortgage["next_due"]) + MORTGAGE_PERIOD_S
                self._journal(state, "mortgage_paid", now,
                              member=borrower,
                              mortgage=mortgage["id"], amount=due)
                if mortgage["remaining"] <= 0:
                    mortgage["open"] = False
                    prop = (state.get("properties") or {}).get(
                        str(mortgage["property"]))
                    if prop is not None:
                        prop["mortgage"] = ""
                    notices.append(
                        {"kind": "mortgage_closed",
                         "member": borrower,
                         "mortgage": mortgage["id"]})
                continue
            mortgage["missed"] = int(mortgage["missed"]) + 1
            mortgage["next_due"] = float(
                mortgage["next_due"]) + MORTGAGE_PERIOD_S
            self._journal(state, "mortgage_missed", now,
                          member=borrower,
                          mortgage=mortgage["id"],
                          missed=mortgage["missed"])
            if int(mortgage["missed"]) > GRACE_MISSED:
                mortgage["open"] = False
                mortgage["defaulted"] = True
                prop = (state.get("properties") or {}).get(
                    str(mortgage["property"]))
                if prop is not None:
                    prop["owner"] = PUBLIC_OWNER
                    prop["mortgage"] = ""
                self._journal(
                    state, "repossession", now, member=borrower,
                    mortgage=mortgage["id"],
                    property=str(mortgage["property"]),
                    written_off=int(mortgage["remaining"]))
                notices.append(
                    {"kind": "repossession", "member": borrower,
                     "mortgage": mortgage["id"],
                     "property": str(mortgage["property"]),
                     "written_off": int(mortgage["remaining"])})
        return notices

    # ── insurance + repair debt ──
    def insure(self, group_id: str, member: str, pid: str, *,
               now: float | None = None) -> dict[str, Any]:
        """Buy/renew a 30-day policy on one property: premium =
        the stated % of current value, paid to the treasury."""
        now = now if now is not None else time.time()
        state = self._load_state(group_id)
        prop = (state.get("properties") or {}).get(pid)
        if prop is None or prop.get("owner") != member:
            return {"ok": False, "reason": "not_found"}
        value = self.property_value(group_id, prop, state=state)
        premium = max(1, int(round(
            value * INSURANCE_PREMIUM_PCT / 100)))
        if prop["currency"] == "gems":
            if self._spend_gems(group_id, member, premium,
                                kind="spend",
                                reason="property insurance",
                                ref=f"property:{pid}",
                                now=now) is None:
                return {"ok": False, "reason": "gems",
                        "premium": premium}
        elif not self.economy._move(
                group_id, member, PUBLIC_OWNER, premium,
                "estate_insure", f"insurance on {pid}", now):
            return {"ok": False, "reason": "funds",
                    "premium": premium}
        prop["insured_until"] = max(
            now, float(prop.get("insured_until") or 0)) \
            + INSURANCE_DAYS * 86400
        self._journal(state, "insure", now, member=member,
                      property=pid, premium=premium,
                      currency=prop["currency"])
        self._save_state(group_id, state)
        return {"ok": True, "premium": premium,
                "currency": prop["currency"],
                "insured_until": float(prop["insured_until"])}

    def pay_repair_debt(self, group_id: str, member: str,
                        pid: str, *, now: float | None = None
                        ) -> dict[str, Any]:
        """Settle a property's repair debt from the wallet
        (rent garnish is the other road; this is the fast one)."""
        now = now if now is not None else time.time()
        state = self._load_state(group_id)
        prop = (state.get("properties") or {}).get(pid)
        if prop is None or prop.get("owner") != member:
            return {"ok": False, "reason": "not_found"}
        debt = int(prop.get("repair_debt") or 0)
        if debt <= 0:
            return {"ok": False, "reason": "no_debt"}
        econ = self.economy
        pay = min(debt, econ.balance(group_id, member))
        if pay <= 0:
            return {"ok": False, "reason": "funds", "debt": debt}
        if not econ._move(group_id, member, PUBLIC_OWNER, pay,
                          "estate_repair",
                          f"repair debt on {pid}", now):
            return {"ok": False, "reason": "funds", "debt": debt}
        prop["repair_debt"] = debt - pay
        self._journal(state, "repair_paid", now, member=member,
                      property=pid, amount=pay)
        self._save_state(group_id, state)
        return {"ok": True, "paid": pay,
                "remaining": debt - pay}

    def assess_repair_bill(self, group_id: str, pid: str,
                           amount: int, *, cause: str,
                           now: float | None = None
                           ) -> dict[str, Any]:
        """Bill ONE property a stated repair amount through
        the same machinery the repair wave uses (the conflict
        organ's sabotage lands here): a live policy covers it
        in full; otherwise bank, then wallet, then repair
        debt — every leg journaled with the cause."""
        now = now if now is not None else time.time()
        state = self._load_state(group_id)
        prop = (state.get("properties") or {}).get(pid)
        if prop is None:
            return {"ok": False, "reason": "not_found"}
        bill = max(1, int(amount))
        owner = str(prop.get("owner") or "")
        if float(prop.get("insured_until") or 0) > now:
            self._journal(state, "insurance_claim", now,
                          member=owner, property=pid,
                          amount=bill, cause=cause)
            self._save_state(group_id, state)
            return {"ok": True, "bill": bill, "covered": bill,
                    "paid": 0, "debt": 0}
        level = self.manager_level(group_id, owner, state=state)
        if level and MANAGER_LEVELS[level]["repair_half"]:
            bill = max(1, bill // 2)
        econ = self.economy
        paid = 0
        bank = econ.bank_member(owner)
        from_bank = min(bill, econ.balance(group_id, bank))
        if from_bank and econ._move(
                group_id, bank, PUBLIC_OWNER, from_bank,
                "estate_repair", f"repair bill {pid} ({cause})",
                now):
            paid += from_bank
        rest = bill - paid
        if rest:
            from_wallet = min(rest, econ.balance(group_id, owner))
            if from_wallet and econ._move(
                    group_id, owner, PUBLIC_OWNER, from_wallet,
                    "estate_repair",
                    f"repair bill {pid} ({cause})", now):
                paid += from_wallet
        shortfall = bill - paid
        if shortfall:
            prop["repair_debt"] = int(
                prop.get("repair_debt") or 0) + shortfall
        self._journal(state, "repair_bill", now, member=owner,
                      property=pid, bill=bill, paid=paid,
                      debt=shortfall, cause=cause)
        self._save_state(group_id, state)
        return {"ok": True, "bill": bill, "covered": 0,
                "paid": paid, "debt": shortfall}

    # ── district events ──
    def _district_event(self, state: dict[str, Any], group_id: str,
                        rng: Any, now: float) -> dict[str, Any]:
        """One bounded district event.  The tick fires at most
        one per group per day; every effect is journaled and
        the notice is returned for the group to see."""
        district = rng.choice(sorted(DISTRICTS))
        kind = rng.choice(
            ["boom", "bust", "repairs", "tenants"])
        notice: dict[str, Any] = {"kind": "district_event",
                                  "event": kind,
                                  "district": district}
        if kind in ("boom", "bust"):
            delta = INDEX_EVENT if kind == "boom" else -INDEX_EVENT
            value = self._bump_index(state, district, delta,
                                     f"event:{kind}", now)
            notice["index"] = value
        elif kind == "tenants":
            state.setdefault("yield_mods", {})[district] = {
                "until": now + 3 * 86400, "pct": -25}
            notice["days"] = 3
        elif kind == "repairs":
            billed = covered = 0
            econ = self.economy
            for prop in (state.get("properties") or {}).values():
                if prop.get("district") != district \
                        or prop.get("owner") == PUBLIC_OWNER:
                    continue
                # Bills are a SPARKS cost of ownership, priced
                # on the property's sparks-measure value (gem
                # tiers at the stated display weight) — one
                # unit for bills, debts and claims alike.
                value = self.property_value(group_id, prop,
                                            state=state)
                if prop["currency"] == "gems":
                    value *= WEALTH_GEM_IN_SPARKS
                bill = max(1, int(round(
                    value * REPAIR_PCT / 100)))
                if float(prop.get("insured_until") or 0) > now:
                    covered += bill
                    self._journal(
                        state, "insurance_claim", now,
                        member=prop["owner"], property=prop["id"],
                        amount=bill, district=district)
                    continue
                level = self.manager_level(
                    group_id, str(prop["owner"]), state=state)
                if level and MANAGER_LEVELS[level]["repair_half"]:
                    bill = max(1, bill // 2)
                owner = str(prop["owner"])
                paid = 0
                bank = econ.bank_member(owner)
                from_bank = min(bill, econ.balance(group_id, bank))
                if from_bank and econ._move(
                        group_id, bank, PUBLIC_OWNER, from_bank,
                        "estate_repair",
                        f"repair bill {prop['id']}", now):
                    paid += from_bank
                rest = bill - paid
                if rest:
                    from_wallet = min(rest, econ.balance(
                        group_id, owner))
                    if from_wallet and econ._move(
                            group_id, owner, PUBLIC_OWNER,
                            from_wallet, "estate_repair",
                            f"repair bill {prop['id']}", now):
                        paid += from_wallet
                shortfall = bill - paid
                if shortfall:
                    prop["repair_debt"] = int(
                        prop.get("repair_debt") or 0) + shortfall
                billed += paid
                self._journal(state, "repair_bill", now,
                              member=owner, property=prop["id"],
                              bill=bill, paid=paid,
                              debt=shortfall, district=district)
            notice["billed"] = billed
            notice["covered"] = covered
        self._journal(state, "district_event", now, district=district,
                      event=kind)
        return notice

    # ── the tick ──
    def tick(self, group_id: str, *, rng: Any,
             cfg: dict[str, Any] | None = None,
             now: float | None = None) -> list[dict[str, Any]]:
        """The rhythm/tick path: settle auctions, run mortgage
        schedules + repossessions, auto-collect rent for managed
        portfolios, collect public rent into the treasury, drift
        quiet districts toward par, and fire the day's district
        event.  Returns notice dicts for the surface to speak."""
        now = now if now is not None else time.time()
        notices: list[dict[str, Any]] = []
        for settled in self.settle_due(group_id, cfg=cfg, now=now):
            notices.append({"kind": "auction_settled", **settled})
        state = self._load_state(group_id)
        notices.extend(self._mortgage_cycle(state, group_id, now))
        # daily drift: a district with no trades today eases 1
        # per mille back toward par (stated, journaled).
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        if state.get("drift_day") != today:
            state["drift_day"] = today
            traded = {str(p.get("district")) for p in
                      (state.get("events") or [])
                      if p.get("kind") in ("buy", "sell",
                                           "market_buy",
                                           "auction_settle",
                                           "public_buy")
                      and time.strftime(
                          "%Y-%m-%d", time.gmtime(
                              float(p.get("ts") or 0))) == today}
            for district in DISTRICTS:
                if district in traded:
                    continue
                row = self._index_row(state, district)
                value = int(row["value"])
                if value != INDEX_BASE:
                    step = INDEX_DRIFT if value < INDEX_BASE \
                        else -INDEX_DRIFT
                    self._bump_index(state, district, step,
                                     "drift", now)
        # manager auto-collect (level 1+: the manager works
        # through jail and absence alike, §20c)
        managers = {m for m, lvl in
                    (state.get("managers") or {}).items()
                    if int(lvl or 0) >= 1}
        self._save_state(group_id, state)
        for member in managers:
            got = self.collect_rent(group_id, member, now=now)
            if got["gems"] > 0:
                notices.append({"kind": "rent_auto",
                                "member": member,
                                "gems": got["gems"]})
        # public rent: the treasury's own buildings pay sparks
        state = self._load_state(group_id)
        public_total = 0
        for prop in (state.get("properties") or {}).values():
            if prop.get("owner") != PUBLIC_OWNER:
                continue
            periods = int((now - float(prop.get("anchor") or now))
                          // RENT_PERIOD_S)
            periods = max(0, min(30, periods))
            if periods <= 0:
                continue
            prop["anchor"] = float(prop["anchor"]) \
                + periods * RENT_PERIOD_S
            rent_sparks = periods * int(
                TIERS[str(prop["tier"])]["rent"]) \
                * PUBLIC_RENT_SPARK_PER_GEM
            public_total += rent_sparks
            self._journal(state, "public_rent", now,
                          property=prop["id"], periods=periods,
                          sparks=rent_sparks)
        if public_total > 0:
            self.economy._append(group_id, self.economy._new_entry(
                group_id, PUBLIC_OWNER, "estate_rent", public_total,
                "public property rent", ref="estate", ts=now))
            notices.append({"kind": "public_rent",
                            "sparks": public_total})
        # The day's district event (one per group per day) —
        # but only in a group that HAS a market: with no
        # property held by anyone, market weather affects
        # nobody, and announcing it would just be noise riding
        # someone else's message (the claimed-once discipline:
        # a tick speaks only about real completions).
        if state.get("event_day") != today:
            state["event_day"] = today
            if state.get("properties"):
                notices.append(self._district_event(
                    state, group_id, rng, now))
        self._save_state(group_id, state)
        return notices

    # ── summaries for surfaces ──
    def summary(self, group_id: str, member: str, *,
                now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        state = self._load_state(group_id)
        props = self.portfolio(group_id, member, now=now)
        rent_ready = sum(p["rent_gems"] for p in props)
        return {
            "gems": self.gems_balance(group_id, member),
            "properties": props,
            "property_count": len(props),
            "property_value": self.member_property_value(
                group_id, member, state=state),
            "rent_ready_gems": rent_ready,
            "manager_level": self.manager_level(
                group_id, member, state=state),
            "premium_owned": list(
                (state.get("premium_owned") or {}).get(
                    member) or []),
            "mortgages": self.mortgages(group_id, member),
        }
