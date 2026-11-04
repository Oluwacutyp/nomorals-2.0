"""Real estate, the whole class (DESIGN §17 slice 2 + §20):
districts with a journaled market index, the rent loop paying
gems, the premium purse's closed faucet set, the member market
with its treasury cut, auctions through treasury escrow,
mortgages with scheduled bank debits and repossession,
insurance + district events, and public property through the
governance flow.  Money assertions read the economy ledger —
the only truth — and gems assertions read the gems ledger.

Rigs follow the house shapes: the organ against a tmp-dir
GroupEconomy, the facade through a real GroupSurface, the
handlers through a RuntimeGroupMixin stub (as the economy /
typography tests).
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.community.group_economy import GroupEconomy
from nomorals.community.group_real_estate import (
    GEM_SOURCES,
    GroupRealEstate,
    GroupRealEstateError,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

G = "telegram-bot:-100777"
T0 = 1_760_000_000.0
DAY = 86400.0


class _Rng:
    """Deterministic rng: choice() returns the queued picks."""

    def __init__(self, *picks):
        self.picks = list(picks)

    def choice(self, seq):
        pick = self.picks.pop(0)
        assert pick in seq, f"{pick} not in {seq}"
        return pick


def _quiet_rng():
    """A tick rng whose daily event lands harmlessly: tenant
    trouble in a district the test doesn't trade in."""
    return _Rng("heights", "tenants")


def _organ(tmp_path: Path):
    econ = GroupEconomy(data_dir=tmp_path / "econ")
    return econ, GroupRealEstate(econ)


def _fund(econ: GroupEconomy, member: str, total: int,
          *, group: str = G) -> None:
    """Real sparks through the real faucet: verified, then
    awards chunked under the earn caps across past weeks
    (the economy test seed pattern)."""
    econ.verify_member(group, member)
    remaining, step = total, 0
    while remaining > 0:
        chunk = min(200, remaining)
        econ.award(group, member, chunk, "seed",
                   now=T0 - 200 * DAY + step * 3 * DAY)
        remaining -= chunk
        step += 1


def _fund_treasury(econ: GroupEconomy, amount: int,
                   *, group: str = G) -> None:
    econ.treasury_bonus(group, amount, "seed treasury")


# ── catalog, index, buy/sell ─────────────────────────────────────


def test_buy_moves_money_to_treasury_and_bumps_index(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 1000)
    out = estate.buy(G, "ada", "old_town", "room", now=T0)
    assert out["ok"]
    # 150 base × 0.80 district level × par index = 120
    assert out["price"] == 120
    assert econ.balance(G, "ada") == 1000 - 120
    assert econ.treasury_balance(G) == 120
    # index +8 per mille → the next room costs 121
    assert estate.district_index(G, "old_town", now=T0)["value"] == 1008
    assert estate.price(G, "old_town", "room") == 121
    prop = out["property"]
    assert estate.portfolio(G, "ada", now=T0)[0]["id"] == prop["id"]


def test_sell_back_is_depreciated_and_dips_index(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 1000)
    out = estate.buy(G, "ada", "old_town", "room", now=T0)
    pid = out["property"]["id"]
    sold = estate.sell(G, "ada", pid, now=T0 + 3600)
    assert sold["ok"]
    # value at index 1008 = 121; 70% sell-back = 84 (a sink:
    # the round trip loses, by construction)
    assert sold["value"] == 121 and sold["proceeds"] == 84
    assert econ.balance(G, "ada") == 1000 - 120 + 84
    assert estate.district_index(G, "old_town", now=T0)["value"] == 1002
    assert estate.portfolio(G, "ada", now=T0) == []


def test_index_is_clamped_and_trend_reads_the_journal(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 100000)
    state = estate._load_state(G)
    estate._bump_index(state, "harbour", 10 ** 6, "test", T0)
    assert estate.district_index(G, "harbour", state=state,
                                 now=T0)["value"] == 2500
    estate._bump_index(state, "harbour", -10 ** 6, "test", T0)
    assert estate.district_index(G, "harbour", state=state,
                                 now=T0)["value"] == 600
    estate._save_state(G, state)
    # trend reads the JOURNAL: a week before the read, the
    # last journaled value was the 600 floor; a late spike to
    # the 2500 cap is a +316.7% week
    state = estate._load_state(G)
    estate._bump_index(state, "harbour", 10 ** 6, "test",
                       T0 + 7.5 * DAY)
    estate._save_state(G, state)
    idx = estate.district_index(G, "harbour", now=T0 + 8 * DAY)
    assert idx["value"] == 2500
    assert idx["trend_pct"] == 316.7


def test_ownership_cap_is_enforced(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 100000)
    cfg = {"ownership_cap": 2}
    estate.buy(G, "ada", "old_town", "room", cfg=cfg, now=T0)
    estate.buy(G, "ada", "old_town", "room", cfg=cfg, now=T0)
    out = estate.buy(G, "ada", "old_town", "room", cfg=cfg,
                     now=T0)
    assert not out["ok"] and out["reason"] == "cap"


# ── rent: the gems faucet ────────────────────────────────────────


def test_rent_accrues_per_day_and_pays_gems(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 1000)
    estate.buy(G, "ada", "old_town", "room", now=T0)  # 1 💎/day
    out = estate.collect_rent(G, "ada", now=T0 + 2 * DAY + 60)
    assert out["gems"] == 2 and out["balance"] == 2
    # the remainder carries: half a day later, nothing new
    out = estate.collect_rent(G, "ada", now=T0 + 2 * DAY + 3600)
    assert out["gems"] == 0 and out["lines"] == []
    # a full day later, exactly one more period
    out = estate.collect_rent(G, "ada", now=T0 + 3 * DAY + 60)
    assert out["gems"] == 1


def test_rent_accrual_is_capped_without_a_manager(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 1000)
    estate.buy(G, "ada", "market_mile", "shop", now=T0)  # 7 💎/day
    out = estate.collect_rent(G, "ada", now=T0 + 30 * DAY)
    assert out["gems"] == 7 * 7  # base cap: 7 days


def test_manager_raises_cap_and_yield_and_autocollects(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 5000)
    estate.buy(G, "ada", "market_mile", "shop", now=T0)
    assert estate.buy_manager(G, "ada", now=T0)["level"] == 1
    out = estate.buy_manager(G, "ada", now=T0)
    assert out["ok"] and out["level"] == 2  # +15%, cap 14
    # 10 days standing → 10 × 7 + 15% = 80 gems
    got = estate.collect_rent(G, "ada", now=T0 + 10 * DAY)
    assert got["gems"] == 70 + 70 * 15 // 100
    # auto-collect on the tick (manager rung 1+)
    notices = estate.tick(G, rng=_quiet_rng(), now=T0 + 12 * DAY)
    auto = [n for n in notices if n["kind"] == "rent_auto"]
    assert auto and auto[0]["gems"] == 2 * 7 + (2 * 7) * 15 // 100


# ── gems: the closed faucet set ──────────────────────────────────


def test_gems_have_no_spark_conversion_path(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 5000)
    # §35 amendment: system drops are a fourth faucet (rare,
    # budget-capped by the group's drops-per-day) — the set
    # stays CLOSED; nothing else may issue gems
    assert GEM_SOURCES == ("rent", "season", "governance",
                           "drop")
    for source in ("admin", "sparks", "exchange", "buy", ""):
        assert estate.credit_gems(
            G, "ada", 50, source=source, reason="sneak") is None
    assert estate.gems_balance(G, "ada") == 0
    # a spark-tier round trip never touches the gems ledger
    out = estate.buy(G, "ada", "old_town", "room", now=T0)
    assert estate.gems_balance(G, "ada") == 0
    estate.sell(G, "ada", out["property"]["id"], now=T0)
    assert estate.gems_balance(G, "ada") == 0
    rows = estate._gems_ledger(G)
    assert rows == []


def test_gem_tiers_cost_gems_and_sell_back_in_gems(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 5000)
    # sparks alone cannot buy the premium shelf's tiers
    out = estate.buy(G, "ada", "old_town", "penthouse", now=T0)
    assert not out["ok"] and out["reason"] == "gems"
    assert out["price"] == 48  # 60 × 0.80 at par
    estate.credit_gems(G, "ada", 100, source="season",
                       reason="seed", now=T0)
    out = estate.buy(G, "ada", "old_town", "penthouse", now=T0)
    assert out["ok"] and estate.gems_balance(G, "ada") == 52
    # penthouse rent: 10 💎/day
    got = estate.collect_rent(G, "ada", now=T0 + DAY)
    assert got["gems"] == 10
    sold = estate.sell(G, "ada", out["property"]["id"],
                       now=T0 + DAY)
    assert sold["ok"] and sold["currency"] == "gems"
    # value at index 1008: round(60 × .8 × 1.008) = 48; 70% = 33
    assert sold["proceeds"] == 33
    assert sold["proceeds"] < 48  # the round trip loses


def test_premium_shelf_spends_gems(tmp_path):
    econ, estate = _organ(tmp_path)
    estate.credit_gems(G, "ada", 30, source="governance",
                       reason="seed", ref="prop-1", now=T0)
    grant = estate.buy_premium(G, "ada", "gold_nameplate",
                               now=T0)
    assert grant and grant["name"] == "Golden Nameplate"
    assert estate.gems_balance(G, "ada") == 5
    assert estate.buy_premium(G, "ada", "estate_banner",
                              now=T0) is None  # 40 > 5


# ── the member market ────────────────────────────────────────────


def test_market_sale_splits_the_cut_to_the_treasury(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 1000)
    _fund(econ, "bola", 1000)
    bought = estate.buy(G, "ada", "old_town", "room", now=T0)
    pid = bought["property"]["id"]
    listed = estate.list_property(G, "ada", pid, 200, now=T0)
    assert listed["ok"]
    # a listed property can't be sold back or re-listed
    assert estate.sell(G, "ada", pid, now=T0)["reason"] == "listed"
    out = estate.buy_listing(G, "bola", listed["listing"],
                             now=T0 + 3600)
    assert out["ok"] and out["cut"] == 10 and out["net"] == 190
    assert econ.balance(G, "ada") == 1000 - 120 + 190
    assert econ.balance(G, "bola") == 1000 - 200
    assert econ.treasury_balance(G) == 120 + 10
    # ownership moved; the rent clock followed the property
    assert estate.portfolio(G, "bola", now=T0)[0]["id"] == pid
    got = estate.collect_rent(G, "bola", now=T0 + DAY)
    assert got["gems"] == 1


# ── auctions: escrow lifecycle ───────────────────────────────────


def test_auction_escrow_hold_refund_settle_antisipe(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 1000)
    _fund(econ, "bola", 1000)
    _fund(econ, "chidi", 1000)
    _fund_treasury(econ, 0)
    bought = estate.buy(G, "ada", "old_town", "room", now=T0)
    pid = bought["property"]["id"]
    started = estate.start_auction(G, "ada", pid, 100, 1,
                                   now=T0)
    aid = started["auction"]
    treasury_before = econ.treasury_balance(G)

    bid = estate.place_bid(G, "bola", aid, 150, now=T0 + 60)
    assert bid["ok"] and not bid["extended"]
    assert econ.balance(G, "bola") == 850  # held, not spent
    assert estate.auction_reserved(G) == 150
    assert econ.treasury_balance(G) == treasury_before + 150

    # too low, own-auction and self-outbid are all refused
    assert estate.place_bid(G, "chidi", aid, 150,
                            now=T0 + 120)["reason"] == "low"
    assert estate.place_bid(G, "ada", aid, 500,
                            now=T0 + 120)["reason"] == "own"
    assert estate.place_bid(G, "bola", aid, 500,
                            now=T0 + 120)["reason"] == "leading"

    outbid = estate.place_bid(G, "chidi", aid, 200,
                              now=T0 + 180)
    assert outbid["ok"] and outbid["refunded"] == 150
    assert econ.balance(G, "bola") == 1000  # refunded in full
    assert estate.auction_reserved(G) == 200

    # anti-snipe: a bid 30s before the close extends it by 120s
    late = estate.place_bid(G, "bola", aid, 250,
                            now=T0 + 3600 - 30)
    assert late["ok"] and late["extended"]
    assert late["ends"] == T0 + 3600 - 30 + 120

    settled = estate.settle_due(G, now=T0 + 3600 + 200)
    assert len(settled) == 1 and settled[0]["winner"] == "bola"
    assert settled[0]["price"] == 250
    assert settled[0]["cut"] == 12  # 5% of 250
    # seller got 238 out of escrow; the cut stayed in treasury
    assert econ.balance(G, "ada") == 1000 - 120 + 238
    assert estate.auction_reserved(G) == 0
    assert estate.portfolio(G, "bola", now=T0)[0]["id"] == pid


def test_auction_unsold_releases_the_property(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 1000)
    bought = estate.buy(G, "ada", "old_town", "room", now=T0)
    estate.start_auction(G, "ada", bought["property"]["id"],
                          100, 1, now=T0)
    settled = estate.settle_due(G, now=T0 + 7200)
    assert settled and settled[0]["winner"] == ""
    out = estate.sell(G, "ada", bought["property"]["id"],
                      now=T0 + 7200)
    assert out["ok"]  # unlocked again


# ── mortgages ────────────────────────────────────────────────────


def _mortgaged(econ, estate, member="ada", fund=5000):
    _fund(econ, member, fund)
    _fund_treasury(econ, 10000)
    bought = estate.buy(G, member, "market_mile", "plaza",
                        now=T0)  # value 2200 at par
    pid = bought["property"]["id"]
    out = estate.borrow(G, member, pid, 1000, now=T0)
    assert out["ok"], out
    return pid, out["mortgage"]


def test_mortgage_ltv_interest_and_scheduled_bank_debit(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 5000)
    _fund_treasury(econ, 10000)
    bought = estate.buy(G, "ada", "market_mile", "plaza",
                        now=T0)
    pid = bought["property"]["id"]
    # the buy bumped the index to 1008, so the building is
    # worth round(2200 × 1.008) = 2218 → LTV 60% caps at 1330
    out = estate.borrow(G, "ada", pid, 1331, now=T0)
    assert not out["ok"] and out["reason"] == "ltv"
    assert out["max"] == 1330
    out = estate.borrow(G, "ada", pid, 1000, now=T0)
    m = out["mortgage"]
    # flat interest: 5% × 4 periods on the principal = 200
    assert m["interest"] == 200 and m["remaining"] == 1200
    assert m["installment"] == 300
    assert econ.balance(G, "ada") == 5000 - 2200 + 1000
    # the property is locked while mortgaged
    assert estate.sell(G, "ada", pid, now=T0)["reason"] \
        == "mortgaged"
    # bank the instalment money; the schedule takes it
    econ.bank_deposit(G, "ada", 400, now=T0)
    estate.tick(G, rng=_quiet_rng(), now=T0 + 7 * DAY + 10)
    after = estate.mortgages(G, "ada")
    assert after and after[0]["remaining"] == 900
    assert after[0]["paid"] == 300
    assert econ.bank_balance(G, "ada") == 100


def test_mortgage_manual_repay_closes_and_unlocks(tmp_path):
    econ, estate = _organ(tmp_path)
    pid, m = _mortgaged(econ, estate)
    out = estate.repay(G, "ada", m["id"], 1200, now=T0 + DAY)
    assert out["ok"] and out["closed"]
    assert estate.mortgages(G, "ada") == []
    assert estate.sell(G, "ada", pid, now=T0 + DAY)["ok"]


def test_mortgage_default_repossesses_after_the_grace(tmp_path):
    econ, estate = _organ(tmp_path)
    pid, m = _mortgaged(econ, estate)
    # bank stays empty: every weekly cycle misses
    now = T0 + 7 * DAY + 10
    for _ in range(2):
        estate.tick(G, rng=_quiet_rng(), now=now)
        now += 7 * DAY
    still = estate.mortgages(G, "ada")
    assert still and still[0]["missed"] == 2  # grace holds
    notices = estate.tick(G, rng=_quiet_rng(), now=now)
    assert estate.mortgages(G, "ada") == []
    repo = [n for n in notices if n["kind"] == "repossession"]
    assert repo and repo[0]["written_off"] == 1200
    # the building is public stock now, and the write-off is
    # on the journal — never silent
    state = estate._load_state(G)
    assert state["properties"][pid]["owner"] == ""
    events = [e for e in state["events"]
              if e["kind"] == "repossession"]
    assert events and events[0]["written_off"] == 1200


# ── insurance + district events ──────────────────────────────────


def test_insurance_covers_repair_wave_uninsured_pays(tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 5000)
    _fund(econ, "bola", 5000)
    a = estate.buy(G, "ada", "market_mile", "shop", now=T0)
    b = estate.buy(G, "bola", "market_mile", "shop", now=T0)
    # two buys moved the index to 1016: ada's shop is worth
    # round(900 × 1.016) = 914 → premium 5% = 46
    cover = estate.insure(G, "ada", a["property"]["id"], now=T0)
    assert cover["ok"] and cover["premium"] == 46
    # repair wave: the tick first drifts the index to 1015,
    # so the bill is 8% of round(900 × 1.015) = 73
    estate.tick(G, rng=_Rng("market_mile", "repairs"),
                now=T0 + DAY)
    state = estate._load_state(G)
    claims = [e for e in state["events"]
              if e["kind"] == "insurance_claim"]
    assert claims and claims[0]["amount"] == 73
    # ada paid only the premium; bola (bought at index 1008
    # for 907) paid the bill
    assert econ.balance(G, "ada") == 5000 - 900 - 46
    assert econ.balance(G, "bola") == 5000 - 907 - 73
    assert state["properties"][b["property"]["id"]][
        "repair_debt"] == 0


def test_repair_shortfall_becomes_debt_and_rent_offsets_it(
        tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 950)  # buys the 900 shop, 50 left
    a = estate.buy(G, "ada", "market_mile", "shop", now=T0)
    estate.tick(G, rng=_Rng("market_mile", "repairs"),
                now=T0 + DAY)
    state = estate._load_state(G)
    prop = state["properties"][a["property"]["id"]]
    # bill = 8% of value 906 (index drifted to 1007) = 72;
    # the wallet covered 50 → 22 of debt
    assert prop["repair_debt"] == 22
    # two days of rent stand (7 💎/day): all of it garnishes
    # the debt first, 1 💎 = 1 spark of debt
    got = estate.collect_rent(G, "ada",
                              now=T0 + 2 * DAY)
    assert got["offset"] == 14 and got["gems"] == 0
    state = estate._load_state(G)
    assert state["properties"][a["property"]["id"]][
        "repair_debt"] == 22 - 14


def test_district_events_move_the_index_and_are_journaled(
        tmp_path):
    econ, estate = _organ(tmp_path)
    _fund(econ, "ada", 1000)
    # with NO property anywhere there is no market, and the
    # tick stays silent — no weather reports for nobody
    silent = estate.tick(G, rng=_Rng("harbour", "boom"),
                         now=T0 - DAY)
    assert not [n for n in silent
                if n["kind"] == "district_event"]
    assert estate.district_index(G, "harbour",
                                 now=T0)["value"] == 1000
    estate.buy(G, "ada", "old_town", "room", now=T0)
    notices = estate.tick(G, rng=_Rng("harbour", "boom"),
                          now=T0)
    events = [n for n in notices
              if n["kind"] == "district_event"]
    assert events and events[0]["event"] == "boom"
    assert estate.district_index(G, "harbour",
                                 now=T0)["value"] == 1060
    # one event per day: a second tick the same day fires none
    notices = estate.tick(G, rng=_Rng("harbour", "bust"),
                          now=T0 + 3600)
    assert not [n for n in notices
                if n["kind"] == "district_event"]
    assert estate.district_index(G, "harbour",
                                 now=T0)["value"] == 1060


# ── public property + governance (facade) ────────────────────────


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def test_public_property_through_governance_and_its_rent(tmp_path):
    surface = _surface(tmp_path)
    econ = surface.economy
    _fund_treasury(econ, 5000, group=G)
    # a second admin executes the proposal; the proposer can't
    proposal = surface.propose_spend(
        G, kind="estate_public",
        payload={"district": "old_town", "tier": "room"},
        by="admin-a")
    out = surface.approve_spend(G, proposal["id"], by="admin-a")
    assert not out["ok"] and out["reason"] == "self_approve"
    out = surface.approve_spend(G, proposal["id"], by="admin-b")
    assert out["ok"], out
    state = surface.estate._load_state(G)
    props = list(state["properties"].values())
    assert props and props[0]["owner"] == ""
    assert econ.treasury_balance(G) == 5000 - 120
    # the governance flow runs at wall-clock; age the building
    # 3 days on its rent clock, then let the tick collect
    now = time.time()
    pid = props[0]["id"]
    state["properties"][pid]["anchor"] = now - 3 * DAY
    surface.estate._save_state(G, state)
    # public rent: room yields 1 💎/day → 10 sparks/day, ×3 days
    notices = surface.estate_cycle(G, rng=_quiet_rng(),
                                   now=now)
    assert any("public property paid" in n for n in notices)
    assert econ.treasury_balance(G) == 5000 - 120 + 30


def test_gems_gift_executes_only_through_a_proposal(tmp_path):
    surface = _surface(tmp_path)
    proposal = surface.propose_spend(
        G, kind="gems_gift",
        payload={"recipient": "ada", "amount": 25,
                 "reason": "top earner"}, by="admin-a")
    out = surface.approve_spend(G, proposal["id"], by="admin-b")
    assert out["ok"], out
    assert surface.estate.gems_balance(G, "ada") == 25


def test_season_rewards_pay_gems_to_the_podium(tmp_path):
    surface = _surface(tmp_path)
    granted = surface.estate_season_rewards(
        G, [("ada", 900), ("bola", 500), ("chidi", 100)],
        now=T0)
    assert granted == [("ada", 30), ("bola", 20),
                       ("chidi", 10)]
    assert surface.estate.gems_balance(G, "bola") == 20


def test_top_holders_count_property_wealth(tmp_path):
    surface = _surface(tmp_path)
    econ = surface.economy
    _fund(econ, "ada", 900, group=G)
    _fund(econ, "bola", 1000, group=G)
    surface.estate_buy(G, "bola", "old_town", "room", now=T0)
    rows = surface.top_holders(G)
    bola = next(r for r in rows if r["member"] == "bola")
    assert bola["property"] == 121  # value at index 1008
    assert bola["worth"] == 1000 - 120 + 121
    # 1001 of worth beats ada's 900 of plain sparks: property
    # alone flips the board
    assert rows[0]["member"] == "bola"


def test_auction_escrow_counts_against_treasury_grants(tmp_path):
    surface = _surface(tmp_path)
    econ = surface.economy
    _fund(econ, "ada", 1000, group=G)
    _fund(econ, "bola", 1000, group=G)
    bought = surface.estate_buy(G, "ada", "old_town", "room",
                                now=T0)
    started = surface.estate_auction_start(
        G, "ada", bought["property"]["id"], 100, 1, now=T0)
    surface.estate_bid(G, "bola", started["auction"], 120,
                       now=T0 + 60)
    # the treasury holds 120 of escrow: grants see only the rest
    assert surface.reserved_treasury(G) == 120
    out = surface.grant(G, "ada", 121, by="admin")
    assert not out["ok"]  # 120 (buy) + 120 (escrow) − 120 held


# ── corrupt store discipline ─────────────────────────────────────


def test_corrupt_estate_state_raises_and_freezes(tmp_path):
    surface = _surface(tmp_path)
    econ = surface.economy
    path = econ._path(G, "estate")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    estate = surface.estate
    with pytest.raises(GroupRealEstateError):
        estate.health(G)
    assert surface.estate_frozen(G)
    out = surface.estate_buy(G, "ada", "old_town", "room",
                             now=T0)
    assert not out["ok"] and out["reason"] == "frozen"


def test_corrupt_gems_ledger_raises(tmp_path):
    econ, estate = _organ(tmp_path)
    path = econ._path(G, "gems")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[broken", encoding="utf-8")
    with pytest.raises(GroupRealEstateError):
        estate.gems_balance(G, "ada")


# ── handlers ─────────────────────────────────────────────────────


def _rig(tmp_path: Path):
    from nomorals.agents.partner.runtime_group import (
        RuntimeGroupMixin)

    class _Adapter:
        def admin_member(self, chat, user_id):
            return {"status": "administrator"}

    class _RT(RuntimeGroupMixin):
        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return _Adapter()

        def _is_operator(self, message):  # type: ignore[no-untyped-def]
            return str(message.sender_id) == "1"

    rt = _RT()
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    return rt, surface


def _msg(sender_id="55", sender="Ada", username="ada", text=""):
    return SimpleNamespace(
        chat=SimpleNamespace(key=G, platform="telegram-bot",
                             kind="group", title="Builders"),
        sender_id=sender_id, sender=sender,
        sender_username=username, text=text, meta={},
        message_id="1")


def _text(out) -> str:
    return getattr(out, "text", out)


def test_handlers_property_map_balance_and_rent(tmp_path):
    rt, surface = _rig(tmp_path)
    econ = surface.economy
    _fund(econ, "55", 2000, group=G)
    surface.handle_message(G, "55", username="ada", name="Ada",
                           text="hello")
    out = _text(rt._control_property("", _msg()))
    assert "Old Town" in out and "The Heights" in out
    assert "Powered by peacethefirst" in out  # the stamp (§27)
    bought = _text(rt._control_buyproperty("old_town room",
                                           _msg()))
    assert "Property bought" in bought
    # rent stands a day later (facade time travel), then the
    # balance card carries the gems row — one purse (§25)
    surface.estate_collect_rent(G, "55", now=time.time()
                                + 2 * DAY)
    balance = _text(rt._control_balance(_msg()))
    assert "Gems" in balance and "💎" in balance
    top = _text(rt._control_top(_msg()))
    assert "property" in top


def test_handlers_auction_buttons_and_bid(tmp_path):
    rt, surface = _rig(tmp_path)
    econ = surface.economy
    _fund(econ, "55", 2000, group=G)
    _fund(econ, "66", 2000, group=G)
    bought = surface.estate_buy(G, "55", "old_town", "room",
                                now=T0)
    # handlers read the wall clock: the auction must open NOW
    started = surface.estate_auction_start(
        G, "55", bought["property"]["id"], 100, 24,
        now=time.time())
    board = rt._control_auctions("", _msg(sender_id="66",
                                          sender="Bola",
                                          username="bola"))
    buttons = getattr(board, "buttons", None) or []
    payloads = [b[1] for row in buttons for b in row]
    assert any(p.startswith("auctions bid ")
               for p in payloads)
    out = _text(rt._control_auctions(
        f"bid {started['auction']} 150",
        _msg(sender_id="66", sender="Bola", username="bola")))
    assert "Bid placed" in out


def test_handlers_gshop_premium_shelf(tmp_path):
    rt, surface = _rig(tmp_path)
    surface.estate.credit_gems(G, "55", 40, source="season",
                               reason="seed", now=T0)
    shop = _text(rt._control_gshop("", _msg()))
    assert "Premium shelf" in shop and "gold_nameplate" in shop
    out = _text(rt._control_gshop("buy gold_nameplate",
                                  _msg()))
    assert "Golden Nameplate" in out
    assert surface.estate.gems_balance(G, "55") == 15
    # and sparks can never substitute for gems at the shelf
    out = _text(rt._control_gshop("buy estate_banner",
                                  _msg()))
    assert "not enough gems" in out
