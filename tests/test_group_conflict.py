"""The conflict layer — gangs, geared crime, heists, wars,
territory, the police and system drops (DESIGN §23/§24/§35).

Pins, in the brief's own order: the arsenal (purchase through
the purse in both currencies, loadouts, counter-pairs in the
heist math), heists across matchups (strong kit vs weak vault
succeeds, naked crew vs bunker busts, the partial band pays
less and jails nobody, every attempt journals its snapshot),
the gang lifecycle (lifestyles are mechanical identity, the
crew→gang→syndicate ladder, strength composition, gang
treasury conservation against the surface's reservations),
wars (rounds, escrow, spoils) and territory income, the
police (ordinary peer play NEVER escalates; farming a weak
newcomer climbs the ladder — fine, vault raid, bounty, jail —
with the evidence riding the action), bounties claimed by a
geared hit and by the citizen hunt, bribes as a risky sink,
governance capture (the power gang drives the EXISTING
governance organ and sets the transfer tax inside its hard
range), inter-group wars through the shared book, and drops
(daily budget, fresh contents, slots 1–5 enforced, expiry,
§33 suppression in an unclaimed room).

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore; a stub rng whose rolls the test chooses (the math
is the system's, the dice are the test's).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.community.group_conflict import (
    ARSENAL,
    LIFESTYLES,
    GroupConflict,
    counter_adjust,
    gang_tier,
    heist_odds,
)
from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupSurface,
)

KEY = "telegram-bot:-1001234567"      # a claimed bot room
KEY2 = "telegram-bot:-1007654321"     # a second claimed room
FOREIGN = "telegram:-100999"          # the userbot in a
                                      # foreign room (§33)


class _Rng:
    """The test's dice: random() returns the chosen roll;
    everything else takes the neutral middle."""

    def __init__(self, roll: float = 0.0) -> None:
        self.roll = roll

    def random(self) -> float:
        return self.roll

    def uniform(self, a: float, b: float) -> float:
        return 1.0

    def randint(self, a: int, b: int) -> int:
        return a

    def choice(self, seq):
        return list(seq)[0]


def _surface(tmp_path: Path, name: str = "g.json") -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / name).load())


def _mode(surface: GroupSurface, key: str, mode: str,
          **over) -> None:
    cfg = dict(surface.store.settings(key)["conflict"])
    cfg["mode"] = mode
    cfg.update(over)
    surface.update_settings(key, conflict=cfg)


def _seed(surface: GroupSurface, key: str, uid: str,
          amount: int, *, name: str | None = None) -> int:
    """Presence (the member ledger + verification) plus real
    sparks through the real faucet, spread across past weeks
    the way the economy rig seeds (earn caps are real)."""
    surface.handle_message(key, uid, username=f"u{uid}",
                           name=name or f"User {uid}",
                           text="hello group")
    base = time.time()
    remaining, step = amount, 0
    while remaining > 0:
        chunk = min(190, remaining)
        surface.game_award(key, uid, chunk, "ttt",
                           now=base - step * 8 * 86400)
        remaining -= chunk
        step += 1
    return surface.economy.balance(key, uid)


def _gems(surface: GroupSurface, key: str, uid: str,
          amount: int) -> None:
    """Fixture gems through a REAL faucet source (season)."""
    out = surface.estate.credit_gems(
        key, uid, amount, source="season", reason="test seed",
        now=time.time())
    assert out is not None


def _gang(surface: GroupSurface, key: str, leader: str,
          members: list[str], *, name: str, lifestyle: str,
          now: float) -> dict:
    out = surface.gang_create(key, leader, name, lifestyle,
                              now=now)
    assert out["ok"], out
    for member in members:
        joined = surface.gang_join(key, member, name, now=now)
        assert joined["ok"], joined
    state = surface.conflict._load_state(key)
    return state["gangs"][out["gang"]["id"]]


def _fund_gang(surface: GroupSurface, key: str, leader: str,
               amount: int, *, now: float) -> None:
    out = surface.conflict.gang_deposit(key, leader, amount,
                                        now=now)
    assert out["ok"], out


# ── the mode gate: nothing in this class exists while off ──────

def test_mode_off_nothing_exists(tmp_path):
    surface = _surface(tmp_path)
    _seed(surface, KEY, "55", 500)
    now = time.time()
    assert surface.gear_buy(KEY, "55", "lockpick_set",
                            now=now)["reason"] == "mode"
    assert surface.gang_create(KEY, "55", "Night Owls",
                               "street", now=now)["reason"] \
        == "mode"
    assert surface.crime_verb(KEY, "55", "77", "shoot",
                              rng=_Rng(), now=now)["reason"] \
        == "mode"
    notices = surface.conflict_cycle(KEY, rng=_Rng(0.0),
                                     now=now)
    assert notices == []
    state = surface.conflict._load_state(KEY)
    assert state["drops"] == {} and state["gangs"] == {}


# ── the arsenal ────────────────────────────────────────────────

def test_arsenal_purchase_sparks_and_gems(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    base = _seed(surface, KEY, "55", 4000)
    now = time.time()
    econ = surface.economy
    out = surface.gear_buy(KEY, "55", "lockpick_set", now=now)
    assert out["ok"] and econ.balance(KEY, "55") == base - 120
    items = surface.conflict.loadout(KEY, "55")
    assert [i["item"] for i in items] == ["lockpick_set"]
    # a two-currency price moves BOTH legs or neither
    _gems(surface, KEY, "55", 5)
    out = surface.gear_buy(KEY, "55", "strike_drone", now=now)
    assert out["ok"], out
    assert econ.balance(KEY, "55") == base - 120 - 1100
    assert surface.estate.gems_balance(KEY, "55") == 0
    # insufficient gems: the sparks leg never moves
    out = surface.gear_buy(KEY, "55", "strike_drone", now=now)
    assert out["reason"] == "gems"
    assert econ.balance(KEY, "55") == base - 120 - 1100
    # the ledger names the sink
    kinds = [e["kind"] for e in econ._load_json(KEY, "ledger",
                                               [])
             if e.get("member") == "55"]
    assert "gear_buy" in kinds


def test_nuke_is_admin_minted_and_unique(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    _seed(surface, KEY, "55", 100)
    _gems(surface, KEY, "55", 1200)
    now = time.time()
    out = surface.gear_buy(KEY, "55", "nuke", now=now)
    assert out["reason"] == "admin_only"
    out = surface.gear_buy(KEY, "55", "nuke", now=now,
                           is_admin=True)
    assert out["ok"], out
    out = surface.gear_buy(KEY, "55", "nuke", now=now,
                           is_admin=True)
    assert out["reason"] == "one_per_group"


def test_counter_pairs_change_the_math():
    bonus, cut, notes = counter_adjust(
        ["emp_jammer"], ["alarm_web"], "heist")
    assert cut == 6 and any("beats" in n for n in notes)
    # the counter-drill mat counters by RATING (it is itself
    # a strong defense), not by cutting the drill's rating
    bonus, cut, notes = counter_adjust(
        ["vault_drill"], ["counter_drill_mat"], "vault")
    assert bonus == 0 and cut == 0
    bonus, cut, notes = counter_adjust(
        ["thermite_charge"], ["vault_plating"], "vault")
    assert cut == int(16 * 0.75)
    # odds are a readable function of the margin + heat
    assert heist_odds(40, 10) > 0.9
    assert heist_odds(0, 40) < 0.1
    assert heist_odds(20, 20, heat_stars=5) < heist_odds(20, 20)


# ── heists ─────────────────────────────────────────────────────

def _heist_rig(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    _seed(surface, KEY, "11", 3000, name="Lead")
    _seed(surface, KEY, "12", 300, name="Two")
    _seed(surface, KEY, "13", 5000, name="Target")
    # the target banks 3000 (the vault the crew comes for)
    # and keeps a working wallet — vault defenses are bought
    # from the wallet like everything else
    surface.deposit(KEY, "13", 3000)
    return surface


def test_heist_strong_kit_vs_weak_vault_succeeds(tmp_path):
    surface = _heist_rig(tmp_path)
    now = time.time()
    surface.gear_buy(KEY, "11", "thermite_charge", now=now)
    surface.gear_buy(KEY, "11", "vault_drill", now=now)
    start = surface.heist_start(KEY, "11", "vault", "13",
                                now=now)
    assert start["ok"], start
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    preview = surface.conflict.heist_preview(
        KEY, start["heist"]["id"], now=now)
    assert preview["odds"] > 0.9, preview
    bank_before = surface.economy.bank_balance(KEY, "13")
    out = surface.heist_go(KEY, "11", rng=_Rng(0.5), now=now)
    assert out["ok"] and out["outcome"] == "success", out
    assert out["take"] == min(int(bank_before * 0.20), 800)
    assert surface.economy.bank_balance(KEY, "13") \
        == bank_before - out["take"]
    # the journal holds the whole attempt: crew, loadout
    # snapshot, defenses, outcome, take
    state = surface.conflict._load_state(KEY)
    jobs = [e for e in state["journal"] if e["kind"] == "heist"]
    assert jobs and jobs[-1]["outcome"] == "success"
    assert "thermite_charge" in jobs[-1]["attack_items"]
    assert jobs[-1]["take"] == out["take"]
    # cooldown: the same crew cannot immediately re-run
    again = surface.heist_start(KEY, "11", "vault", "13",
                                now=now + 10)
    assert again["reason"] == "cooldown"


def test_heist_naked_crew_vs_bunker_busts(tmp_path):
    surface = _heist_rig(tmp_path)
    now = time.time()
    _gems(surface, KEY, "13", 20)
    assert surface.gear_buy(KEY, "13", "bunker", now=now)["ok"]
    out = surface.gear_install(KEY, "13", "bunker",
                               "vault:13", now=now)
    assert out["ok"], out
    assert surface.heist_start(KEY, "11", "vault", "13",
                               now=now)["ok"]
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    bank_before = surface.economy.bank_balance(KEY, "13")
    out = surface.heist_go(KEY, "11", rng=_Rng(0.99), now=now)
    assert out["outcome"] == "bust", out
    assert out["take"] == 0
    assert surface.economy.bank_balance(KEY, "13") == bank_before
    for member in ("11", "12"):
        jail = surface.jail_state(KEY, member, now=now)
        assert jail and jail["until"] > now + 40 * 60


def test_heist_partial_band_pays_less_and_jails_nobody(tmp_path):
    surface = _heist_rig(tmp_path)
    now = time.time()
    assert surface.heist_start(KEY, "11", "vault", "13",
                               now=now)["ok"]
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    bank_before = surface.economy.bank_balance(KEY, "13")
    out = surface.heist_go(KEY, "11", rng=_Rng(0.5), now=now)
    assert out["outcome"] == "partial", out
    expect = int(int(bank_before * 0.20) * 0.4)
    assert out["take"] == expect
    assert surface.jail_state(KEY, "11", now=now) is None
    assert surface.jail_state(KEY, "12", now=now) is None


def test_heist_decoy_vault_pays_nothing(tmp_path):
    surface = _heist_rig(tmp_path)
    now = time.time()
    surface.gear_buy(KEY, "11", "thermite_charge", now=now)
    surface.gear_buy(KEY, "13", "decoy_vault", now=now)
    assert surface.gear_install(KEY, "13", "decoy_vault",
                                "vault:13", now=now)["ok"]
    assert surface.heist_start(KEY, "11", "vault", "13",
                               now=now)["ok"]
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    bank_before = surface.economy.bank_balance(KEY, "13")
    out = surface.heist_go(KEY, "11", rng=_Rng(0.5), now=now)
    assert out["outcome"] == "decoy", out
    assert surface.economy.bank_balance(KEY, "13") == bank_before


# ── crime verbs, hospital, medkit ──────────────────────────────

def test_kill_hospitalizes_and_bail_or_medkit_discharges(
        tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    _seed(surface, KEY, "11", 1500, name="Hit")
    _seed(surface, KEY, "13", 800, name="Mark")
    now = time.time()
    # bare hands cannot kill: the weapon gate is real
    out = surface.crime_verb(KEY, "11", "13", "kill",
                             rng=_Rng(), now=now)
    assert out["reason"] == "weapon"
    surface.gear_buy(KEY, "11", "street_pistol", now=now)
    wallet_before = surface.economy.balance(KEY, "13")
    out = surface.crime_verb(KEY, "11", "13", "kill",
                             rng=_Rng(0.0), now=now)
    assert out["ok"] and out["success"], out
    assert 0 < out["take"] <= 250
    assert surface.economy.balance(KEY, "13") \
        == wallet_before - out["take"]
    ward = surface.hospital_state(KEY, "13", now=now)
    assert ward and ward["bill"] == 40
    # the hospitalized cannot work
    out = surface.work_shift(KEY, "13", now=now)
    assert out["reason"] == "hospital"
    # /bail pays the bill and discharges
    out = surface.bail(KEY, "13", now=now)
    assert out["ok"] and out.get("hospital") and out["cost"] == 40
    assert surface.hospital_state(KEY, "13", now=now) is None
    # a second kill, a medkit discharge this time
    out = surface.crime_verb(KEY, "11", "13", "kill",
                             rng=_Rng(0.0), now=now + 601)
    assert out["success"]
    surface.gear_buy(KEY, "13", "medkit", now=now)
    out = surface.gear_use(KEY, "13", "medkit", now=now + 602)
    assert out["ok"], out
    assert surface.hospital_state(KEY, "13", now=now) is None


def test_failed_verb_jails_the_actor(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    _seed(surface, KEY, "11", 500)
    _seed(surface, KEY, "13", 500)
    now = time.time()
    out = surface.crime_verb(KEY, "11", "13", "shoot",
                             rng=_Rng(0.99), now=now)
    assert out["ok"] and not out["success"]
    jail = surface.jail_state(KEY, "11", now=now)
    assert jail and jail["until"] > now


def test_sabotage_bills_through_the_estate(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    _seed(surface, KEY, "11", 900, name="Wrecker")
    _seed(surface, KEY, "13", 3000, name="Owner")
    now = time.time()
    bought = surface.estate_buy(KEY, "13", "old_town", "shop",
                                now=now)
    assert bought["ok"], bought
    total_before = surface.economy.total_balance(KEY, "13")
    out = surface.crime_sabotage(KEY, "11", "13",
                                 rng=_Rng(0.0), now=now)
    assert out["ok"] and out["success"], out
    assert out["bill"] > 0
    state = surface.estate._load_state(KEY)
    prop = next(iter(state["properties"].values()))
    debt = int(prop.get("repair_debt") or 0)
    paid = total_before - surface.economy.total_balance(
        KEY, "13")
    assert paid + debt == out["bill"]


# ── gangs ──────────────────────────────────────────────────────

def test_gang_lifecycle_lifestyle_and_strength(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    now = time.time()
    members = ["21", "22", "23", "24", "25"]
    for uid in members:
        _seed(surface, KEY, uid, 800)
    gang = _gang(surface, KEY, "21", members[1:], name="Owls",
                 lifestyle="street", now=now)
    from nomorals.community.group_conflict import gang_tier
    assert gang_tier(len(gang["members"])) == "gang"
    assert gang_tier(3) == "crew" and gang_tier(15) == "syndicate"
    # lifestyle identity is mechanical: the street affinity
    # makes a lockpick worth more on a vault job for this
    # gang's members than the same pick is worth plain
    surface.gear_buy(KEY, "22", "lockpick_set", now=now)
    organ = surface.conflict
    from nomorals.community.group_conflict import gear_power
    items = organ.loadout(KEY, "22")
    assert gear_power(items, "vault", now=now) == 8
    assert gear_power(items, "vault", now=now,
                      affinity=["lockpick_set"],
                      affinity_bonus=3) == 11
    # strength composition: every input present and summed
    _fund_gang(surface, KEY, "21", 400, now=now)
    gang = organ._load_state(KEY)["gangs"][gang["id"]]
    strength = organ.gang_strength(KEY, gang, now=now)
    assert strength["total"] == strength["members"] \
        + strength["armory"] + strength["territory"] \
        + strength["treasury"] + strength["wins"]
    assert strength["treasury"] == 4
    # the gang job pays the lifestyle's work, hourly
    job = organ.gang_job(KEY, "23", rng=_Rng(), now=now)
    assert job["ok"] and job["amount"] > 0
    again = organ.gang_job(KEY, "23", rng=_Rng(), now=now)
    assert again["reason"] == "cooldown"
    # ranks: only the leader promotes
    out = organ.gang_set_role(KEY, "22", "23", "officer",
                              now=now)
    assert out["reason"] == "not_leader"
    out = organ.gang_set_role(KEY, "21", "23", "officer",
                              now=now)
    assert out["ok"]
    # leaving dissolves an empty gang and pays its treasury
    # out to the last leader
    for uid in members[1:]:
        surface.gang_leave(KEY, uid, now=now)
    wallet_before = surface.economy.balance(KEY, "21")
    out = surface.gang_leave(KEY, "21", now=now)
    assert out["dissolved"]
    assert surface.economy.balance(KEY, "21") \
        == wallet_before + 400


def test_gang_treasury_is_reserved_from_grants(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    now = time.time()
    _seed(surface, KEY, "21", 1000)
    _seed(surface, KEY, "22", 100)
    _gang(surface, KEY, "21", [], name="Owls",
          lifestyle="mafia", now=now)
    _fund_gang(surface, KEY, "21", 600, now=now)
    treasury = surface.economy.treasury_balance(KEY)
    assert treasury == 600
    assert surface.reserved_treasury(KEY) >= 600
    # the gang's money is NOT the treasury's to grant away
    out = surface.grant(KEY, "22", 100, by="99", now=now)
    assert not out["ok"] and out["reason"] == "funds"
    # but the gang itself can pay it out, by rank
    out = surface.conflict.gang_withdraw(KEY, "22", 100,
                                         now=now)
    assert out["reason"] == "no_gang"
    wallet_before = surface.economy.balance(KEY, "21")
    out = surface.conflict.gang_withdraw(KEY, "21", 250,
                                         now=now)
    assert out["ok"]
    assert surface.economy.balance(KEY, "21") \
        == wallet_before + 250


# ── wars + territory ───────────────────────────────────────────

def _war_rig(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare")
    now = time.time()
    a = ["a1", "a2", "a3", "a4", "a5"]
    b = ["b1", "b2", "b3", "b4", "b5"]
    for uid in a + b:
        _seed(surface, KEY, uid, 1200)
    gang_a = _gang(surface, KEY, "a1", a[1:], name="Owls",
                   lifestyle="street", now=now)
    gang_b = _gang(surface, KEY, "b1", b[1:], name="Wolves",
                   lifestyle="mafia", now=now)
    _fund_gang(surface, KEY, "a1", 600, now=now)
    _fund_gang(surface, KEY, "b1", 600, now=now)
    # armory + rifles for the Owls: their rounds should land
    for uid in a:
        surface.gear_buy(KEY, uid, "hunting_rifle", now=now)
    return surface, now, gang_a, gang_b


def test_war_rounds_escrow_spoils_and_territory(tmp_path):
    surface, now, gang_a, gang_b = _war_rig(tmp_path)
    organ = surface.conflict
    # the Wolves hold Harbour; it is the spoils at stake
    claim = organ.claim_territory(KEY, "b1", "harbour", now=now)
    assert claim["ok"], claim
    declared = surface.war_declare(KEY, "a1", "Wolves", now=now)
    assert declared["ok"], declared
    war_id = declared["war"]["id"]
    # escrow left both treasuries into the pot
    state = organ._load_state(KEY)
    assert state["gangs"][gang_a["id"]]["treasury"] == 400
    answered = surface.war_answer(KEY, "b1", True, now=now)
    assert answered["ok"], answered
    state = organ._load_state(KEY)
    assert state["gangs"][gang_b["id"]]["treasury"] == 300
    assert state["wars"][war_id]["pot"] == 400
    for uid in ("a1", "a2", "b1", "b2"):
        assert surface.war_join(KEY, uid, now=now)["ok"]
    cfg = surface.conflict_cfg(KEY)
    events = []
    for step in range(1, 5):
        for event in organ.cycle_wars(
                KEY, rng=_Rng(), now=now + step * 3601, cfg=cfg):
            events.append(event)
    done = [e for e in events if e.get("finished")]
    assert done, events
    assert done[-1]["winner_gang"]["name"] == "Owls"
    state = organ._load_state(KEY)
    # winner: 400 own + 400 pot; loser keeps its remainder
    assert state["gangs"][gang_a["id"]]["treasury"] == 800
    assert state["gangs"][gang_b["id"]]["treasury"] == 300
    assert state["gangs"][gang_a["id"]]["wins"] == 1
    # spoils: Harbour changes hands
    out = organ.war_claim_spoils(KEY, "a1", "harbour",
                                 now=now + 20000)
    assert out["ok"], out
    assert surface.territory_map(KEY)["harbour"] == "Owls"


def test_territory_income_pays_from_treasury_not_air(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare")
    now = time.time()
    members = ["31", "32", "33", "34", "35"]
    for uid in members:
        _seed(surface, KEY, uid, 900)
    gang = _gang(surface, KEY, "31", members[1:], name="Owls",
                 lifestyle="syndicate", now=now)
    _fund_gang(surface, KEY, "31", 300, now=now)
    organ = surface.conflict
    assert organ.claim_territory(KEY, "31", "old_town",
                                 now=now)["ok"]
    # fixture reset: the seeding messages already ran today's
    # territory accounting (nothing was held then) — reopen
    # the day so THIS call is the day's accounting
    state = organ._load_state(KEY)
    state["territory_day"] = ""
    organ._save_state(KEY, state)
    cfg = surface.conflict_cfg(KEY)
    treasury_before = surface.economy.treasury_balance(KEY)
    events = organ.cycle_territory(KEY, now=now, cfg=cfg,
                                   free_treasury=140)
    income = [e for e in events
              if e["kind"] == "territory_income"]
    assert income and income[0]["amount"] == 20
    state = organ._load_state(KEY)
    # upkeep 10 out, income 20 in — and the treasury only
    # ever held the income's source: gang sub-accounting
    # moved, no ledger mint happened
    assert state["gangs"][gang["id"]]["treasury"] \
        == 300 - 100 - 10 + 20
    assert surface.economy.treasury_balance(KEY) \
        == treasury_before
    # once a day only
    assert organ.cycle_territory(KEY, now=now, cfg=cfg,
                                 free_treasury=140) == []


def test_inter_group_war_through_the_book(tmp_path):
    surface_a = _surface(tmp_path, "ga.json")
    surface_b = _surface(tmp_path, "gb.json")
    now = time.time()
    for surface, key, prefix in ((surface_a, KEY, "a"),
                                 (surface_b, KEY2, "b")):
        _mode(surface, key, "warfare")
        members = [f"{prefix}{i}" for i in range(1, 6)]
        for uid in members:
            _seed(surface, key, uid, 1500)
        _gang(surface, key, members[0], members[1:],
              name="Owls" if prefix == "a" else "Wolves",
              lifestyle="street" if prefix == "a" else "mafia",
              now=now)
        _fund_gang(surface, key, members[0], 600, now=now)
        if prefix == "a":
            for uid in members:
                surface_a.gear_buy(key, uid, "hunting_rifle",
                                   now=now)
    declared = surface_a.war_declare_inter(KEY, "a1", "Wolves",
                                           now=now)
    assert declared["ok"], declared
    answered = surface_b.war_answer_inter(KEY2, "b1", True,
                                          now=now)
    assert answered["ok"], answered
    for uid in ("a1", "a2"):
        assert surface_a.war_join_inter(KEY, uid, now=now)["ok"]
    for uid in ("b1", "b2"):
        assert surface_b.war_join_inter(KEY2, uid, now=now)["ok"]
    # rounds fall due on A's cycle; both sides settle on
    # their own
    for step in range(1, 6):
        surface_a.conflict_cycle(KEY, rng=_Rng(0.99),
                                 now=now + step * 3601)
        surface_b.conflict_cycle(KEY2, rng=_Rng(0.99),
                                 now=now + step * 3601)
    war = surface_a.war_book.wars_for_group(KEY)[0]
    assert war["status"] == "done", war
    assert (war.get("a") or {})["gang_name"] == "Owls"
    assert war["winner"] == "a", war
    assert war["settled"] == {"a": True, "b": True}
    state_a = surface_a.conflict._load_state(KEY)
    state_b = surface_b.conflict._load_state(KEY2)
    gang_a = next(iter(state_a["gangs"].values()))
    gang_b = next(iter(state_b["gangs"].values()))
    # winner's escrow returned to its gang; loser's is gone
    assert gang_a["treasury"] == 600
    assert gang_b["treasury"] == 400
    # and the spoils half landed with the winning leader
    assert surface_a.economy.balance(KEY, "a1") >= 200


# ── the police ─────────────────────────────────────────────────

def _farmer(tmp_path, hits: int = 3):
    """A rich member farming a brand-new weak one: the bully
    index's textbook prey.  Returns (surface, now) after the
    hits have landed."""
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    _seed(surface, KEY, "66", 5000, name="Farmer")
    _seed(surface, KEY, "77", 60, name="Newbie")
    # the farmer banks a vault for the raid to find, and the
    # treasury holds funds for fines to join
    surface.deposit(KEY, "66", 1000)
    now = time.time()
    for step in range(hits):
        out = surface.crime_verb(KEY, "66", "77", "shoot",
                                 rng=_Rng(0.0),
                                 now=now + step * 601)
        assert out["ok"] and out["success"], out
    return surface, now + hits * 601


def test_police_ordinary_play_never_escalates(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    now = time.time()
    _seed(surface, KEY, "66", 5000, name="Peer")
    _seed(surface, KEY, "77", 5000, name="PeerTwo")
    # established members both (aged past the 24h newcomer
    # shield): peers robbing peers is the game — a few
    # successful robs between equals (deep pockets both
    # sides, so nobody is ever punched down) never reach
    # even attention
    entry = surface.store.group(KEY)
    for uid in ("66", "77"):
        entry["members"][uid]["first_seen"] = now - 2 * 86400
        entry["members"][uid]["joined_ts"] = now - 2 * 86400
    for step in range(3):
        out = surface.rob(KEY, "66", "77", rng=_Rng(0.0),
                          now=now + step * 601)
        assert out["ok"], out
    organ = surface.conflict
    stars, index, _inputs = organ.wanted_stars(KEY, "66",
                                               now=now + 2000)
    assert index < 6.0 and stars == 0, (index, stars)
    actions = organ.police_sweep(KEY, now=now + 2000,
                                 free_treasury=1000)
    assert actions == []


def test_police_farming_climbs_the_ladder(tmp_path):
    surface, now = _farmer(tmp_path, hits=3)
    organ = surface.conflict
    stars, index, inputs = organ.wanted_stars(KEY, "66",
                                              now=now)
    assert stars == 2 and index >= 16, (stars, index)
    assert any("newcomer" in i for i in inputs)
    wallet_before = surface.economy.balance(KEY, "66")
    treasury_before = surface.economy.treasury_balance(KEY)
    actions = organ.police_sweep(KEY, now=now,
                                 free_treasury=5000)
    assert len(actions) == 1, actions
    action = actions[0]
    assert action["action"] == "fine" and action["fine"] == 300
    assert surface.economy.balance(KEY, "66") \
        == wallet_before - 300
    assert surface.economy.treasury_balance(KEY) \
        == treasury_before + 300
    # the same rung never acts twice in one episode
    assert organ.police_sweep(KEY, now=now,
                              free_treasury=5000) == []
    # one more farming hit crosses into 3★: the next sweep
    # brings the vault raid + an auto-bounty
    out = surface.crime_verb(KEY, "66", "77", "shoot",
                             rng=_Rng(0.0), now=now)
    assert out["success"]
    bank_before = surface.economy.bank_balance(KEY, "66")
    actions = organ.police_sweep(KEY, now=now + 2000,
                                 free_treasury=5000)
    assert len(actions) == 1, actions
    action = actions[0]
    assert action["stars"] == 3 and action["raid"] \
        == int(bank_before * 0.15), action
    assert action["bounty"] == 150
    bounties = organ.open_bounties(KEY)
    assert len(bounties) == 1 \
        and bounties[0]["target"] == "66"
    # the evidence is IN the action for /glog readers
    assert any("punched down" in i for i in action["inputs"])


def test_police_jail_notice_and_undo_lifts_only_the_jail(
        tmp_path):
    surface, now = _farmer(tmp_path, hits=6)
    organ = surface.conflict
    stars, index, _ = organ.wanted_stars(KEY, "66", now=now)
    assert stars >= 4, (stars, index)
    bank_before = surface.economy.bank_balance(KEY, "66")
    notices = surface.conflict_cycle(KEY, rng=_Rng(0.99),
                                     now=now)
    assert any("🚔" in n for n in notices), notices
    jail = surface.jail_state(KEY, "66", now=now)
    assert jail, "the crackdown rung jails through the surface"
    # the raid already settled: undo must not refund it
    raided = bank_before - surface.economy.bank_balance(
        KEY, "66")
    assert raided > 0
    record = next(r for r in surface.store.group(KEY)
                  .get("notices", {}).values()
                  if r.get("action") == "police_jail")
    assert record["extra"]["reversible"] is True
    assert record["fine"] == 0  # settlements never ride the
    # refundable slot
    assert surface.jail_clear(KEY, "66")
    assert surface.jail_state(KEY, "66", now=now) is None
    assert surface.economy.bank_balance(KEY, "66") \
        == bank_before - raided


def test_bribe_success_forgives_and_failure_adds_heat(tmp_path):
    surface, now = _farmer(tmp_path, hits=3)
    organ = surface.conflict
    _stars, index_before, _ = organ.wanted_stars(KEY, "66",
                                                 now=now)
    assert index_before >= 16
    wallet_before = surface.economy.balance(KEY, "66")
    out = organ.bribe(KEY, "66", rng=_Rng(0.0), now=now)
    assert out["ok"] and out["success"], out
    assert surface.economy.balance(KEY, "66") \
        == wallet_before - out["cost"]
    _s, index_after, _i = organ.wanted_stars(KEY, "66", now=now)
    assert index_after < index_before * 0.5
    # failure: the money still sinks, and the attempt is heat
    surface3 = _surface(tmp_path / "c")
    _mode(surface3, KEY, "crime")
    _seed(surface3, KEY, "66", 5000, name="Farmer")
    _seed(surface3, KEY, "77", 60, name="Newbie")
    t0 = time.time()
    for step in range(3):
        surface3.crime_verb(KEY, "66", "77", "shoot",
                            rng=_Rng(0.0), now=t0 + step * 601)
    organ3 = surface3.conflict
    _s, before, _i = organ3.wanted_stars(KEY, "66",
                                         now=t0 + 2000)
    out = organ3.bribe(KEY, "66", rng=_Rng(0.99),
                       now=t0 + 2000)
    assert out["ok"] and not out["success"]
    _s, after, _i = organ3.wanted_stars(KEY, "66",
                                        now=t0 + 2000)
    assert after > before


# ── bounties ───────────────────────────────────────────────────

def test_bounty_claimed_by_geared_hit_and_by_hunt(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    now = time.time()
    _seed(surface, KEY, "81", 1500, name="Poster")
    _seed(surface, KEY, "82", 900, name="Mark")
    _seed(surface, KEY, "83", 900, name="Hunter")
    out = surface.bounty_post(KEY, "81", "82", 200, now=now)
    assert out["ok"], out
    # the escrow is reserved: grants can't spend it
    assert surface.reserved_treasury(KEY) >= 200
    surface.gear_buy(KEY, "83", "street_pistol", now=now)
    wallet_before = surface.economy.balance(KEY, "83")
    out = surface.crime_verb(KEY, "83", "82", "shoot",
                             rng=_Rng(0.0), now=now)
    assert out["success"] and out["bounty"] == 200, out
    assert surface.economy.balance(KEY, "83") \
        == wallet_before + out["take"] + 200
    assert surface.conflict.open_bounties(KEY) == []
    # the hunt: a wanted mark, brought in by a citizen
    out = surface.bounty_post(KEY, "81", "82", 150,
                              now=now + 700)
    assert out["ok"]
    out = surface.bounty_hunt(KEY, "83", "82", rng=_Rng(0.0),
                              now=now + 700)
    assert out["ok"] and out["success"], out
    assert out["bounty"] == 150
    assert surface.jail_state(KEY, "82", now=now + 700)


# ── governance capture ─────────────────────────────────────────

def test_governance_capture_moves_levers_not_mechanisms(
        tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare")
    now = time.time()
    members = ["41", "42", "43", "44", "45"]
    for uid in members:
        _seed(surface, KEY, uid, 2000)
    gang = _gang(surface, KEY, "41", members[1:], name="Owls",
                 lifestyle="mafia", now=now)
    organ = surface.conflict
    # before capture: the settings tax stands, the gang is
    # nobody at city hall
    assert surface.gang_power_actor(KEY, "41", now=now) is None
    pay = surface.pay(KEY, "42", "43", 1000, now=now)
    assert pay["fee"] == 20  # the default 2%
    _fund_gang(surface, KEY, "41", 500, now=now)
    assert organ.claim_territory(KEY, "41", "old_town",
                                 now=now)["ok"]
    assert organ.claim_territory(KEY, "41", "market_mile",
                                 now=now)["ok"]
    power = surface.gang_power_actor(KEY, "41", now=now)
    assert power and power["name"] == "Owls"
    # a rank-and-file member holds no power
    assert surface.gang_power_actor(KEY, "43", now=now) is None
    # the tax lever moves — inside the hard range only
    out = organ.set_gang_tax(KEY, "41", 9, now=now)
    assert out["ok"]
    out = organ.set_gang_tax(KEY, "41", 11, now=now)
    assert out["reason"] == "range"
    pay = surface.pay(KEY, "42", "43", 1000, now=now)
    assert pay["fee"] == 90  # the ruling gang's 9%
    # proposals still need a second approver: capture changes
    # WHO proposes, never the mechanism
    proposal = surface.propose_spend(
        KEY, kind="grant",
        payload={"recipient": "44", "amount": 50}, by="41",
        now=now)
    out = surface.approve_spend(KEY, proposal["id"], by="41",
                                now=now)
    assert out["reason"] == "self_approve"
    out = surface.approve_spend(KEY, proposal["id"], by="42",
                                now=now)
    assert out["ok"], out


# ── system drops (§35) ─────────────────────────────────────────

def test_drops_budget_contents_slots_and_expiry(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    organ = surface.conflict
    cfg = surface.conflict_cfg(KEY)
    now = time.time()
    for uid in ("51", "52", "53", "54", "55", "56"):
        surface.handle_message(KEY, uid, username=f"u{uid}",
                               name=f"U{uid}", text="hi")
    # fixture reset: a drop the message cycle may have opened
    # while seeding is not this test's subject — start the
    # day clean so the budget below measures only these opens
    state = organ._load_state(KEY)
    state["drops"] = {}
    state["drop_day"] = {}
    organ._save_state(KEY, state)
    # the daily budget is a hard wall
    opened = [organ.open_drop(KEY, rng=_Rng(0.0), now=now,
                              cfg=cfg) for _ in range(3)]
    assert opened[0] and opened[1] and opened[2] is None
    drop = opened[0]
    assert drop["kind"] == "sparks"  # roll 0.0 → first band
    assert 2 <= drop["slots"] <= 5
    # claims fill the slots; the next claim is refused
    claims = 0
    for uid in ("51", "52", "53", "54", "55"):
        out = surface.sysdrop_claim(KEY, uid, drop["id"],
                                    now=now)
        if out["ok"]:
            claims += 1
    assert claims == drop["slots"]
    out = surface.sysdrop_claim(KEY, "56", drop["id"], now=now)
    assert not out["ok"] and out["reason"] in ("gone", "full")
    # one claim per member, ever
    state = organ._load_state(KEY)
    assert state["drops"][drop["id"]]["status"] == "filled"
    # expiry: a late claim is refused and closes the drop…
    drop2 = opened[1]
    out = surface.sysdrop_claim(KEY, "51", drop2["id"],
                                now=now + 901)
    assert out["reason"] == "expired"
    state = organ._load_state(KEY)
    assert state["drops"][drop2["id"]]["status"] == "expired"
    # …and the cycle's own expiry closes a stale open drop
    # nobody ever tried to claim
    stale = organ.roll_drop(rng=_Rng(0.0), now=now)
    stale["id"] = "drop-stale"
    stale["expires"] = now + 10
    state["drops"]["drop-stale"] = stale
    organ._save_state(KEY, state)
    expired = organ.expire_sysdrops(KEY, now=now + 901)
    assert any(d["id"] == "drop-stale" for d in expired)
    state = organ._load_state(KEY)
    assert state["drops"]["drop-stale"]["status"] == "expired"


def test_drop_contents_vary_and_gems_ride_the_faucet(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    organ = surface.conflict
    import random as _random
    kinds = set()
    for seed in range(60):
        drop = organ.roll_drop(rng=_random.Random(seed),
                               now=time.time())
        kinds.add(drop["kind"])
        assert 1 <= drop["slots"] <= 5
    assert len(kinds) >= 4, kinds
    # a gems drop credits through the estate organ's gated
    # faucet (source "drop"), never around it
    surface.handle_message(KEY, "51", username="u51",
                           name="U51", text="hi")
    drop = organ.roll_drop(rng=_Rng(0.90), now=time.time())
    assert drop["kind"] == "gems", drop
    state = organ._load_state(KEY)
    drop["id"] = "drop-test"
    drop["expires"] = time.time() + 900
    state["drops"]["drop-test"] = drop
    organ._save_state(KEY, state)
    out = surface.sysdrop_claim(KEY, "51", "drop-test",
                                now=time.time())
    assert out["ok"], out
    assert surface.estate.gems_balance(KEY, "51") \
        == drop["amount"]


def test_drops_suppressed_in_an_unclaimed_room(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, FOREIGN, "warfare")
    surface.handle_message(FOREIGN, "51", username="u51",
                           name="U51", text="hi")
    notices = surface.conflict_cycle(FOREIGN, rng=_Rng(0.0),
                                     now=time.time())
    assert notices == []
    state = surface.conflict._load_state(FOREIGN)
    assert state["drops"] == {}
    # …while the claimed room's cycle opens one and queues
    # the announcement for the runtime drain (fixture reset
    # first, so only THIS cycle's open is measured)
    _mode(surface, KEY, "crime")
    surface.handle_message(KEY, "51", username="u51",
                           name="U51", text="hi")
    state = surface.conflict._load_state(KEY)
    state["drops"] = {}
    state["drop_day"] = {}
    surface.conflict._save_state(KEY, state)
    surface._extras(KEY)["conflict_feed"] = []
    surface.conflict_cycle(KEY, rng=_Rng(0.0), now=time.time())
    state = surface.conflict._load_state(KEY)
    assert len(state["drops"]) == 1
    feed = surface._extras(KEY).get("conflict_feed") or []
    assert len(feed) == 1 and feed[0]["kind"] == "sysdrop"


# ── runtime wiring ─────────────────────────────────────────────

class _Adapter:
    def __init__(self, admins=()):
        self.admins = set(admins)

    def admin_member(self, chat, user_id):
        return {"status": "administrator"
                if str(user_id) in self.admins else "member"}


def _rig(tmp_path: Path, *, admins=("99",)):
    from nomorals.agents.partner.runtime_group import \
        RuntimeGroupMixin

    class _RT(RuntimeGroupMixin):
        def _group_adapter(self, platform):
            return _Adapter(admins)

        def _is_operator(self, message):
            return False

    rt = _RT()
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    rt.sent = []

    def _send(chat, text, **kw):
        rt.sent.append((str(text), kw.get("buttons")))
        return SimpleNamespace(ok=True, message_id="777")

    rt._send = _send
    return rt, surface


def _msg(*, sender_id="55", sender="Bola", username="bola"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform="telegram-bot",
                             kind="group", title="Builders"),
        sender_id=sender_id, sender=sender,
        sender_username=username, text="", meta={},
        message_id="1")


def _text(out) -> str:
    return str(getattr(out, "text", out))


def test_runtime_conflict_commands_render(tmp_path):
    rt, surface = _rig(tmp_path)
    _mode(surface, KEY, "warfare")
    _seed(surface, KEY, "55", 1500)
    _seed(surface, KEY, "56", 400, name="Ada")
    admin = _msg(sender_id="99", sender="Boss", username="boss")
    out = rt._control_gsettings("conflict warfare", admin)
    assert "warfare" in _text(out)
    out = rt._control_gang("create street Night Owls", _msg())
    assert "Night Owls" in _text(out), out
    out = rt._control_gang("", _msg())
    assert "Night Owls" in _text(out)
    out = rt._control_gshop("", _msg())
    assert "Arsenal" in _text(out), out
    # AA-50: the conflict gear handler carries its own name
    # (_control_conflict_gear); _control_gear is the games handler.
    out = rt._control_conflict_gear("", _msg())
    assert "Street power" in _text(out)
    out = rt._control_police("", _msg())
    assert "precinct" in _text(out)
    # no drop is live in this room (fixture reset — the
    # seeding messages' cycle may have opened one for real)
    state = surface.conflict._load_state(KEY)
    state["drops"] = {}
    surface.conflict._save_state(KEY, state)
    out = rt._control_claimdrop("", _msg())
    assert "no single live drop" in _text(out)
    # the console knows the new section
    assert "conflict" in rt._CONSOLE_SECTIONS
    out = rt._console_apply(surface, KEY, admin, "99",
                            "conflict", ["drops", "3"])
    assert out and "3 per day" in out
    assert surface.conflict_cfg(KEY)["drops_per_day"] == 3


def test_runtime_feed_posts_drop_and_claim_settles(tmp_path):
    rt, surface = _rig(tmp_path)
    _mode(surface, KEY, "crime")
    surface.handle_message(KEY, "55", username="bola",
                           name="Bola", text="hi")
    surface.conflict_cycle(KEY, rng=_Rng(0.0),
                           now=time.time())
    rt._post_conflict_feed(surface, KEY, _msg())
    assert any("Supply drop" in text for text, _b in rt.sent), \
        rt.sent
    state = surface.conflict._load_state(KEY)
    drop_id = next(iter(state["drops"]))
    out = rt._control_claimdrop(drop_id, _msg())
    assert "Drop claimed" in _text(out), out
    state = surface.conflict._load_state(KEY)
    assert state["drops"][drop_id]["claims"] == ["55"]


def test_runtime_police_jail_undo_lifts_jail_only(tmp_path):
    rt, surface = _rig(tmp_path)
    # the same farming street as the organ-level test,
    # rebuilt inside this rig's surface:
    _mode(surface, KEY, "crime")
    _seed(surface, KEY, "66", 5000, name="Farmer")
    _seed(surface, KEY, "77", 60, name="Newbie")
    surface.deposit(KEY, "66", 1000)
    t0 = time.time()
    for step in range(6):
        surface.crime_verb(KEY, "66", "77", "shoot",
                           rng=_Rng(0.0), now=t0 + step * 601)
    surface.conflict_cycle(KEY, rng=_Rng(0.99),
                           now=t0 + 6 * 601)
    assert surface.jail_state(KEY, "66", now=t0 + 6 * 601)
    bank_after_raid = surface.economy.bank_balance(KEY, "66")
    record = next(r for r in surface.store.group(KEY)
                  .get("notices", {}).values()
                  if r.get("action") == "police_jail")
    admin = _msg(sender_id="99", sender="Boss", username="boss")
    out = rt._control_undonotice(record["id"], admin)
    text = _text(out)
    assert "jail time is lifted" in text, text
    assert "stands" in text  # the raid is a settlement
    assert surface.jail_state(KEY, "66",
                              now=t0 + 6 * 601) is None
    assert surface.economy.bank_balance(KEY, "66") \
        == bank_after_raid


def test_runtime_capture_lets_the_power_gang_grant(tmp_path):
    rt, surface = _rig(tmp_path)
    _mode(surface, KEY, "warfare")
    now = time.time()
    members = ["41", "42", "43", "44", "45"]
    for uid in members:
        _seed(surface, KEY, uid, 2000)
    _gang(surface, KEY, "41", members[1:], name="Owls",
          lifestyle="mafia", now=now)
    organ = surface.conflict
    _fund_gang(surface, KEY, "41", 500, now=now)
    organ.claim_territory(KEY, "41", "old_town", now=now)
    organ.claim_territory(KEY, "41", "market_mile", now=now)
    # treasury funds for the grant: a member contributes
    surface.economy.contribute_treasury(KEY, "42", 300,
                                        now=now)
    leader = _msg(sender_id="41", sender="User 41",
                  username="u41")
    plain = _msg(sender_id="43", sender="User 43",
                 username="u43")
    out = rt._control_grant("@u44 100", plain)
    assert "admins" in _text(out), out
    before = surface.economy.balance(KEY, "44")
    out = rt._control_grant("@u44 100", leader)
    assert "granted" in _text(out).lower(), out
    assert surface.economy.balance(KEY, "44") == before + 100


# ── house law: the guardrail file never moves ──────────────────
# (sanctioned moves: the assembly truth-fix AA-64 rewrote
# persona.py's forget accounting so failed forgets stop reporting
# as forgotten; AA-76 made check_contradiction's crashed recall
# leave a trace — log + failure ledger — instead of reading as
# "no contradiction". Digest re-pinned after each; the guard
# stands against any FURTHER unsanctioned move.)

def test_persona_guardrail_untouched():
    path = Path(__file__).resolve().parents[1] \
        / "nomorals" / "memory" / "persona.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest.startswith("ba839924"), digest
