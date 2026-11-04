"""Group conflict: crime, gangs, wars, territory, the police
and system drops (DESIGN §23 + §24 + §23-addendum + §35).

The economy's conflict class, completing the loop (earn →
build → defend → dominate).  Everything here rides the organs
that already exist — never a parallel money path:

- every sparks leg moves on the group economy's ledger through
  its own paired-move machinery, so balances stay
  reconstructible and the §17 issuance throttle sees every
  mint (system-drop sparks are ``award`` earns; heist takes are
  treasury/vault moves; fines and bails are the shipped sinks);
- gems move only through the estate organ's gated credit
  (source ``drop`` — the §35 amendment to its faucet list);
- jail is the surface's existing jail; gangs' money is held IN
  the group treasury with per-gang sub-accounting here, and the
  surface's reservation accounting counts gang treasuries, war
  escrows and bounty escrows as promised treasury, so no other
  treasury path can spend them;
- crime outcomes are gambles a member can READ: the heist and
  verb math is ratings + counter-pairs + police heat, with the
  computed odds returned on every result — never a bare RNG
  flip dressed as fate.

State: one JSON file per group beside the economy's files
(``conflict``), estate discipline — an existing-but-unreadable
file RAISES (:class:`GroupConflictError`) rather than reading
as empty.  Inter-group wars live in ONE global book beside the
group store (the reputation book's precedent): a war names two
(group, gang) sides wherever they live.

Isolation: stdlib + ``nomorals.core`` + the group economy and
estate organs.  Participants are group-local ids; "" is the
treasury.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .group_economy import GroupEconomy
from .group_real_estate import DISTRICTS, GroupRealEstate

_log = get_logger("nomorals.community.group_conflict")

__all__ = [
    "GroupConflict", "GroupConflictError", "WarBook",
    "ARSENAL", "LIFESTYLES", "CONFLICT_DEFAULTS",
    "LOOT_KINDS", "WANTED_RUNGS",
    "HQ_TIERS", "HQ_DEFENSE", "hq_tier_of", "hq_defense_bonus",
    "combat_power_of", "heist_odds",
]


class GroupConflictError(RuntimeError):
    """The group's conflict state exists but cannot be read.
    Raised, never swallowed into empty state: starting over
    would silently drop every gang, loadout and war."""


# ── the arsenal (DESIGN §23 addendum) ────────────────────────────
#: One equipment registry for the group conflict systems AND
#: Devon Arena (which reads the same definitions later — the
#: registry is data, not code paths).  Every item carries:
#: role (breach | defense | support), tier, ratings per job
#: (heist / vault / hit / war), what it counters and what
#: counters it, durability (uses before it breaks; None =
#: indestructible frame) or single-use, a cooldown, and its
#: price in sparks and/or gems — the purse's big sinks (§25).
#: ``drop`` = eligible for the §35 loot table.
ARSENAL: dict[str, dict[str, Any]] = {
    # breach — pulling the job
    "lockpick_set": {
        "name": "Lockpick Set", "icon": "🔓", "role": "breach",
        "tier": 1, "ratings": {"heist": 6, "vault": 8, "hit": 0,
                               "war": 0},
        "counters": [], "countered_by": [],
        "durability": 12, "cooldown_s": 0,
        "price": {"sparks": 120}, "drop": True,
        "blurb": "Quiet hands for quiet doors."},
    "disguise_kit": {
        "name": "Disguise Kit", "icon": "🎭", "role": "breach",
        "tier": 1, "ratings": {"heist": 5, "vault": 4, "hit": 2,
                               "war": 0},
        "counters": ["guard_rotation"], "countered_by": [],
        "durability": 6, "cooldown_s": 0,
        "price": {"sparks": 150}, "drop": True,
        "blurb": "Walk in like you belong there. Lowers the "
                 "heat a job leaves behind."},
    "getaway_bike": {
        "name": "Getaway Bike", "icon": "🏍️", "role": "breach",
        "tier": 1, "ratings": {"heist": 4, "vault": 3, "hit": 0,
                               "war": 2},
        "counters": [], "countered_by": [],
        "durability": 20, "cooldown_s": 0,
        "price": {"sparks": 200}, "drop": True,
        "blurb": "Being gone is half the job."},
    "vault_drill": {
        "name": "Vault Drill", "icon": "🛠️", "role": "breach",
        "tier": 2, "ratings": {"heist": 12, "vault": 16, "hit": 0,
                               "war": 0},
        "counters": [], "countered_by": ["counter_drill_mat"],
        "durability": 10, "cooldown_s": 0,
        "price": {"sparks": 450}, "drop": True,
        "blurb": "Eats vault doors for breakfast."},
    "emp_jammer": {
        "name": "EMP Jammer", "icon": "📡", "role": "breach",
        "tier": 2, "ratings": {"heist": 9, "vault": 9, "hit": 0,
                               "war": 3},
        "counters": ["alarm_web"], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"sparks": 380}, "drop": True,
        "blurb": "One pulse and the alarm web goes dark."},
    "signal_scrambler": {
        "name": "Signal Scrambler", "icon": "📻", "role": "breach",
        "tier": 2, "ratings": {"heist": 8, "vault": 6, "hit": 0,
                               "war": 4},
        "counters": ["counter_drone_net"], "countered_by": [],
        "durability": 8, "cooldown_s": 0,
        "price": {"sparks": 350}, "drop": False,
        "blurb": "Nets and trackers hear only static."},
    "getaway_van": {
        "name": "Getaway Van", "icon": "🚐", "role": "breach",
        "tier": 2, "ratings": {"heist": 8, "vault": 6, "hit": 0,
                               "war": 3},
        "counters": [], "countered_by": [],
        "durability": 25, "cooldown_s": 0,
        "price": {"sparks": 520}, "drop": False,
        "blurb": "Room for the whole crew and the whole take."},
    "scout_drone": {
        "name": "Scout Drone", "icon": "🛸", "role": "breach",
        "tier": 2, "ratings": {"heist": 7, "vault": 7, "hit": 3,
                               "war": 6},
        "counters": ["guard_rotation"], "countered_by": [
            "counter_drone_net"],
        "durability": None, "cooldown_s": 3600,
        "price": {"sparks": 600}, "drop": True,
        "blurb": "Eyes over the wall before anyone climbs it."},
    "thermite_charge": {
        "name": "Thermite Charge", "icon": "🧨", "role": "breach",
        "tier": 3, "ratings": {"heist": 16, "vault": 18, "hit": 0,
                               "war": 5},
        "counters": ["vault_plating"], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"sparks": 900}, "drop": True,
        "blurb": "Beats vault plating — and raises serious "
                 "heat doing it."},
    "strike_drone": {
        "name": "Strike Drone", "icon": "🎯", "role": "breach",
        "tier": 3, "ratings": {"heist": 10, "vault": 8,
                               "hit": 14, "war": 12},
        "counters": [], "countered_by": ["counter_drone_net"],
        "durability": None, "cooldown_s": 7200,
        "price": {"sparks": 1100, "gems": 5}, "drop": True,
        "blurb": "The argument that arrives from the sky."},
    "siege_ram": {
        "name": "Siege Ram", "icon": "🐏", "role": "breach",
        "tier": 3, "ratings": {"heist": 14, "vault": 12, "hit": 4,
                               "war": 14},
        "counters": ["bunker"], "countered_by": [],
        "durability": 15, "cooldown_s": 0,
        "price": {"sparks": 1000}, "drop": False,
        "blurb": "Doors are a suggestion. Bunkers take "
                 "convincing — this convinces."},
    "street_pistol": {
        "name": "Street Pistol", "icon": "🔫", "role": "breach",
        "tier": 1, "ratings": {"heist": 2, "vault": 0, "hit": 10,
                               "war": 4},
        "counters": [], "countered_by": [],
        "durability": 30, "cooldown_s": 0,
        "price": {"sparks": 250}, "drop": True,
        "blurb": "The hitman's first argument."},
    "hunting_rifle": {
        "name": "Hunting Rifle", "icon": "🎯", "role": "breach",
        "tier": 2, "ratings": {"heist": 0, "vault": 0, "hit": 16,
                               "war": 6},
        "counters": [], "countered_by": [],
        "durability": 25, "cooldown_s": 0,
        "price": {"sparks": 650}, "drop": True,
        "blurb": "Distance is a kind of armor — for the "
                 "shooter."},
    "nuke": {
        "name": "The Nuke", "icon": "☢️", "role": "breach",
        "tier": 4, "ratings": {"heist": 40, "vault": 40,
                               "hit": 30, "war": 40},
        "counters": [], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"gems": 500}, "drop": False,
        "admin_mint": True,
        "blurb": "The fantasy ceiling. One exists per economy "
                 "at most; using it is a server event, and the "
                 "police heat is total."},
    # defense — holding what's yours (installed on a target)
    "alarm_web": {
        "name": "Alarm Web", "icon": "🕸️", "role": "defense",
        "tier": 1, "ratings": {"heist": 8, "vault": 8, "hit": 0,
                               "war": 0},
        "counters": [], "countered_by": ["emp_jammer"],
        "durability": 20, "cooldown_s": 0,
        "price": {"sparks": 180}, "drop": True,
        "blurb": "Every door and window, wired."},
    "guard_rotation": {
        "name": "Guard Rotation", "icon": "💂", "role": "defense",
        "tier": 1, "ratings": {"heist": 7, "vault": 6, "hit": 0,
                               "war": 3},
        "counters": [], "countered_by": ["disguise_kit",
                                        "scout_drone"],
        "durability": 30, "cooldown_s": 0,
        "price": {"sparks": 220}, "drop": True,
        "blurb": "Fresh eyes on a schedule nobody outside "
                 "knows."},
    "decoy_vault": {
        "name": "Decoy Vault", "icon": "🎁", "role": "defense",
        "tier": 2, "ratings": {"heist": 10, "vault": 12,
                               "hit": 0, "war": 0},
        "counters": [], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"sparks": 400}, "drop": False,
        "blurb": "They crack the wrong vault and leave with "
                 "confetti — a 'success' that pays nothing."},
    "vault_plating": {
        "name": "Vault Plating", "icon": "🛡️", "role": "defense",
        "tier": 2, "ratings": {"heist": 14, "vault": 16,
                               "hit": 0, "war": 0},
        "counters": [], "countered_by": ["thermite_charge"],
        "durability": 40, "cooldown_s": 0,
        "price": {"sparks": 700}, "drop": True,
        "blurb": "Inches of hardened steel on every surface."},
    "counter_drill_mat": {
        "name": "Counter-Drill Mat", "icon": "🧱", "role": "defense",
        "tier": 2, "ratings": {"heist": 9, "vault": 10, "hit": 0,
                               "war": 0},
        "counters": ["vault_drill"], "countered_by": [],
        "durability": 30, "cooldown_s": 0,
        "price": {"sparks": 420}, "drop": False,
        "blurb": "Drill bits go in; drill bits do not come "
                 "out."},
    "counter_drone_net": {
        "name": "Counter-Drone Net", "icon": "🥅", "role": "defense",
        "tier": 2, "ratings": {"heist": 8, "vault": 8, "hit": 6,
                               "war": 8},
        "counters": ["strike_drone", "scout_drone"],
        "countered_by": ["signal_scrambler"],
        "durability": 25, "cooldown_s": 0,
        "price": {"sparks": 550}, "drop": True,
        "blurb": "The sky over this vault is closed."},
    "bunker": {
        "name": "Bunker", "icon": "🏰", "role": "defense",
        "tier": 3, "ratings": {"heist": 22, "vault": 24,
                               "hit": 0, "war": 10},
        "counters": [], "countered_by": ["siege_ram"],
        "durability": 60, "cooldown_s": 0,
        "price": {"sparks": 1600, "gems": 8}, "drop": False,
        "blurb": "Built to be besieged. Bring a ram or don't "
                 "bother."},
    "shield_ward": {
        "name": "Shield Ward", "icon": "🔮", "role": "defense",
        "tier": 3, "ratings": {"heist": 18, "vault": 18,
                               "hit": 10, "war": 10},
        "counters": [], "countered_by": [],
        "durability": 35, "cooldown_s": 0,
        "price": {"gems": 30}, "drop": True,
        "blurb": "The fantasy side of the fence: a ward that "
                 "turns drills, drones and bad intentions."},
    # personal armor — worn, defends the member (hit / war)
    "street_armor": {
        "name": "Street Armor", "icon": "🦺", "role": "armor",
        "tier": 1, "ratings": {"heist": 0, "vault": 0, "hit": 6,
                               "war": 4},
        "counters": [], "countered_by": [],
        "durability": 20, "cooldown_s": 0,
        "price": {"sparks": 200}, "drop": True,
        "blurb": "Stops the argument street pistols make."},
    "tactical_armor": {
        "name": "Tactical Armor", "icon": "🛡️", "role": "armor",
        "tier": 2, "ratings": {"heist": 0, "vault": 0, "hit": 12,
                               "war": 8},
        "counters": [], "countered_by": [],
        "durability": 30, "cooldown_s": 0,
        "price": {"sparks": 550}, "drop": True,
        "blurb": "Plates where it matters."},
    "powered_armor": {
        "name": "Powered Armor", "icon": "🤖", "role": "armor",
        "tier": 3, "ratings": {"heist": 0, "vault": 0, "hit": 20,
                               "war": 14},
        "counters": [], "countered_by": [],
        "durability": 40, "cooldown_s": 0,
        "price": {"sparks": 1200, "gems": 6}, "drop": False,
        "blurb": "Wear the bunker."},
    # support / consumables
    "medkit": {
        "name": "Medkit", "icon": "🧰", "role": "support",
        "tier": 1, "ratings": {"heist": 0, "vault": 0, "hit": 0,
                               "war": 0},
        "counters": [], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"sparks": 60}, "drop": True,
        "blurb": "Walk out of the hospital now instead of "
                 "waiting it out."},
    "smoke_bomb": {
        "name": "Smoke Bomb", "icon": "💨", "role": "support",
        "tier": 1, "ratings": {"heist": 3, "vault": 3, "hit": 3,
                               "war": 2},
        "counters": [], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"sparks": 70}, "drop": True,
        "blurb": "A small edge, spent in one breath."},
    "stim_pack": {
        "name": "Stim Pack", "icon": "💉", "role": "support",
        "tier": 1, "ratings": {"heist": 2, "vault": 2, "hit": 4,
                               "war": 3},
        "counters": [], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"sparks": 90}, "drop": True,
        "blurb": "Borrowed reflexes, one job only."},
    "fake_plates": {
        "name": "Fake Plates", "icon": "🔢", "role": "support",
        "tier": 1, "ratings": {"heist": 3, "vault": 2, "hit": 0,
                               "war": 0},
        "counters": [], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"sparks": 80}, "drop": True,
        "blurb": "The getaway nobody can trace — less heat "
                 "after the job."},
    "burner_phone": {
        "name": "Burner Phone", "icon": "📱", "role": "support",
        "tier": 1, "ratings": {"heist": 2, "vault": 2, "hit": 0,
                               "war": 0},
        "counters": [], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"sparks": 50}, "drop": True,
        "blurb": "One clean call. Less heat after the job."},
    "bribe_envelope": {
        "name": "Bribe Envelope", "icon": "✉️", "role": "support",
        "tier": 2, "ratings": {"heist": 0, "vault": 0, "hit": 0,
                               "war": 0},
        "counters": [], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"sparks": 300}, "drop": True,
        "blurb": "Makes a /bribe far more likely to land."},
    "repair_kit": {
        "name": "Repair Kit", "icon": "🔧", "role": "support",
        "tier": 1, "ratings": {"heist": 0, "vault": 0, "hit": 0,
                               "war": 0},
        "counters": [], "countered_by": [],
        "durability": None, "single_use": True, "cooldown_s": 0,
        "price": {"sparks": 100}, "drop": True,
        "blurb": "Restores one broken or worn tool to full "
                 "durability."},
}

#: Defense scopes an item can be installed on: the group
#: treasury, a district bank ("bank:<district>"), a member's
#: own vault ("vault:<member>").
DEFENSE_SCOPES = ("treasury", "bank:", "vault:")

#: Base resistance of each heist target before installed gear.
TARGET_BASE: dict[str, int] = {"treasury": 20, "bank": 12,
                               "vault": 8}

#: Heist takes: fraction of the target's funds on a full
#: success, and the hard cap — a heist is a wound, not an
#: execution of the whole economy.
TAKE_FRACTION: dict[str, float] = {"treasury": 0.12,
                                   "bank": 0.15, "vault": 0.20}
TAKE_CAP: dict[str, int] = {"treasury": 1500, "bank": 1200,
                            "vault": 800}

#: Items whose use raises police heat beyond the job's base
#: (the addendum's named example: thermite beats plating but
#: raises heat).
HEAT_ITEMS: dict[str, float] = {"thermite_charge": 3.0,
                                "nuke": 12.0,
                                "strike_drone": 1.5}
#: Items that COOL a job's heat footprint.
COOL_ITEMS: dict[str, float] = {"disguise_kit": 1.0,
                                "fake_plates": 1.0,
                                "burner_phone": 1.0}


# ── gang lifestyles (DESIGN §24) ─────────────────────────────────
#: A lifestyle is identity plus BOUNDED mechanics: an income
#: flavor (its small-job table), gear affinities (rating bonus
#: on named items), a war style multiplier, a heat profile
#: (how fast the police notice it), and its rank names.
LIFESTYLES: dict[str, dict[str, Any]] = {
    "syndicate": {
        "name": "Syndicate", "icon": "🎩",
        "blurb": "Business first: territory and tax income, "
                 "clean suits, quiet ledgers.",
        "ranks": ("Boss", "Underboss", "Associate"),
        "affinity": ["vault_drill", "getaway_van",
                     "signal_scrambler"],
        "affinity_bonus": 3,
        "war_mult": 1.0, "heat_mult": 0.8,
        "income": "territory",
        "jobs": [("ledger work", 40, 70),
                 ("quiet collections", 55, 95),
                 ("boardroom favour", 70, 120)]},
    "cartel": {
        "name": "Cartel", "icon": "🌵",
        "blurb": "Contraband runs: high risk, high pay, and "
                 "the police know the name.",
        "ranks": ("Patrón", "Teniente", "Halcon"),
        "affinity": ["getaway_van", "getaway_bike",
                     "thermite_charge"],
        "affinity_bonus": 3,
        "war_mult": 1.1, "heat_mult": 1.3,
        "income": "runs",
        "jobs": [("border run", 70, 130),
                 ("warehouse move", 90, 160),
                 ("night convoy", 110, 200)]},
    "bikers": {
        "name": "Biker Crew", "icon": "🏍️",
        "blurb": "Loud pipes, louder raids: a duel and war "
                 "edge, straight up.",
        "ranks": ("President", "VP", "Prospect"),
        "affinity": ["getaway_bike", "siege_ram",
                     "street_pistol"],
        "affinity_bonus": 3,
        "war_mult": 1.2, "heat_mult": 1.1,
        "income": "raids",
        "jobs": [("convoy escort", 45, 80),
                 ("turf ride", 60, 100),
                 ("clubhouse collection", 50, 90)]},
    "street": {
        "name": "Street Crew", "icon": "🧢",
        "blurb": "Turf and numbers: recruits fast, hits often, "
                 "lives loud on its own corners.",
        "ranks": ("OG", "Shot Caller", "Youngblood"),
        "affinity": ["lockpick_set", "street_pistol",
                     "smoke_bomb"],
        "affinity_bonus": 3,
        "war_mult": 1.05, "heat_mult": 1.0,
        "income": "turf",
        "jobs": [("corner hustle", 30, 55),
                 ("lookout work", 25, 50),
                 ("block collection", 45, 75)]},
    "mafia": {
        "name": "Mafia", "icon": "🍝",
        "blurb": "Extortion and governance reach: when the "
                 "family holds power, its word moves the "
                 "economy's levers.",
        "ranks": ("Don", "Capo", "Soldato"),
        "affinity": ["disguise_kit", "hunting_rifle",
                     "bribe_envelope"],
        "affinity_bonus": 3,
        "war_mult": 1.0, "heat_mult": 0.9,
        "income": "extortion",
        "jobs": [("protection round", 50, 85),
                 ("a quiet word", 60, 100),
                 ("favours owed", 65, 110)]},
    "vigilantes": {
        "name": "Vigilantes", "icon": "🦸",
        "blurb": "Bounty hunters with a code: the police look "
                 "the other way, and bounties pay them double "
                 "respect.",
        "ranks": ("Marshal", "Deputy", "Watchman"),
        "affinity": ["scout_drone", "hunting_rifle",
                     "tactical_armor"],
        "affinity_bonus": 3,
        "war_mult": 1.1, "heat_mult": 0.6,
        "income": "bounties",
        "jobs": [("patrol the blocks", 35, 60),
                 ("serve a warrant", 55, 90),
                 ("escort a witness", 50, 85)]},
}

#: Crew → gang → syndicate, by member count (§24 small-time
#: path).  Tiers gate the big systems: crews run small jobs
#: and petty vault heists; gangs run full heists and in-group
#: wars; syndicates declare inter-group wars and hold
#: governance capture.
TIER_LADDER: list[tuple[int, str]] = [
    (15, "syndicate"), (5, "gang"), (1, "crew")]

#: Governance capture (§23/§24): the gang IN POWER is the
#: strongest gang holding at least this many districts.  Its
#: leader + officers may drive treasury governance (proposals
#: still need a second approver) and set the transfer tax
#: inside the hard range — capture changes WHO, never the
#: mechanism.
CAPTURE_DISTRICTS = 2
CAPTURE_TAX_RANGE = (0, 10)


def gang_tier(member_count: int) -> str:
    for need, tier in TIER_LADDER:
        if member_count >= need:
            return tier
    return "crew"


# ── the police (DESIGN §24) ──────────────────────────────────────
#: The bully index: points per journaled crime event inside a
#: rolling 7-day window.  Ordinary play (a rob or two against
#: peers) never reaches ATTENTION; farming the weak does.
INDEX_WINDOW_S = 7 * 86400.0
INDEX_POINTS = {
    "base": 1.0,          # any successful geared/ungeared hit
    "punch_down": 2.0,    # target held ≤35% of the actor's total
    "shield_break": 3.0,  # target was inside their first 24h
    "repeat": 1.0,        # 3rd+ hit on the same target
    "heist": 1.0,         # per crew member, per job
    "war_stomp": 1.0,     # war round won vs a <½-strength gang
    "reports": 2.0,       # ≥3 distinct reporters in the window
}
ATTENTION_INDEX = 6.0
#: wanted stars by index: (floor, stars)
WANTED_BANDS: list[tuple[float, int]] = [
    (46.0, 5), (34.0, 4), (24.0, 3), (16.0, 2), (10.0, 1)]
#: What the police DO at each star level (applied once per
#: wanted episode, on the cycle): percentages of the offender's
#: own wealth — police strength scales with the offender's.
WANTED_RUNGS: dict[int, dict[str, Any]] = {
    1: {"action": "attention"},
    2: {"action": "fine", "wallet_pct": 10, "fine_cap": 300},
    3: {"action": "raid", "bank_pct": 15, "bounty": 150},
    4: {"action": "jail", "jail_min": 60, "bank_pct": 20,
        "bounty": 300},
    5: {"action": "crackdown", "jail_min": 120, "bank_pct": 25,
        "wallet_pct": 15, "fine_cap": 600, "bounty": 600},
}

#: Headquarters (§36): a chapter gang dict may carry an
#: ``hq`` block ({"tier": 0..3}) written ONLY by the gang
#: registry — the gang whose HQ sits in this group concentrates
#: its defense here.  The bonus is one number read by every
#: defense surface the organ owns: heist defense against the
#: town's money (and against an HQ member's vault), in-group
#: war side power, and gang strength.  Tier names ride along so
#: no reader hardcodes them.
HQ_TIERS: list[str] = ["none", "hideout", "compound",
                       "fortress"]
HQ_DEFENSE: dict[int, int] = {0: 0, 1: 6, 2: 14, 3: 26}


def hq_tier_of(gang: dict[str, Any] | None) -> int:
    """The HQ tier of a chapter gang dict (0 = no HQ here)."""
    if not gang:
        return 0
    hq = gang.get("hq") or {}
    try:
        return max(0, min(3, int(hq.get("tier") or 0)))
    except (TypeError, ValueError):
        return 0


def hq_defense_bonus(gang: dict[str, Any] | None) -> int:
    return HQ_DEFENSE[hq_tier_of(gang)]


#: System drops (§35): the loot table.  Contents roll fresh
#: every drop; rarity weights the roll; slots scale 1–5 by
#: kind (a single gear piece is winner-take-all; a sparks pool
#: splits across its slots).
LOOT_KINDS: list[tuple[str, int]] = [
    ("sparks", 45), ("gear", 25), ("sparks_big", 15),
    ("gems", 10), ("gear_rare", 5)]

#: Group-overridable conflict levers (settings block
#: ``conflict``).
CONFLICT_DEFAULTS: dict[str, Any] = {
    "mode": "off",              # off | crime | warfare (§23)
    "drops_per_day": 2,         # §35 admin budget (0 disables)
    "drop_ttl_s": 900,          # a system drop's claim window
    "drop_chance": 0.25,        # per cycle pass in budget
    "heist_cooldown_s": 1800,   # per member between heists
    "crime_cooldown_s": 600,    # per member between hit verbs
    "war_escrow": 200,          # sparks per side, from the
                                # gang treasury (in-group wars)
    "war_round_s": 3600,        # one engagement per round
    "war_rounds": 3,            # rounds a war is fought over
    "territory_income_day": 20,  # base sparks per district-day
    "territory_upkeep_day": 10,  # gang treasury per district
    "police": True,             # the system's own faction
}


def _cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(CONFLICT_DEFAULTS)
    for key, value in (cfg or {}).items():
        if key in out:
            out[key] = value
    return out


# ── power math (pure — the readable gamble) ──────────────────────

def _live_instances(instances: list[dict[str, Any]],
                    now: float) -> list[dict[str, Any]]:
    """Instances that can work right now: not broken, off
    cooldown.  An instance's ``durability`` is None for an
    indestructible frame (drones), an int for wear items, and
    1 for single-use items (spent whole on use)."""
    out = []
    for inst in instances or []:
        dur = inst.get("durability")
        if dur is not None and int(dur) <= 0:
            continue
        if float(inst.get("cd_until") or 0) > now:
            continue
        out.append(inst)
    return out


def _new_instance(item_id: str, now: float) -> dict[str, Any]:
    spec = ARSENAL[item_id]
    dur = spec.get("durability")
    if spec.get("single_use"):
        dur = 1
    return {"item": item_id, "durability": dur,
            "cd_until": 0.0, "acquired": now}


def _wear(instances: list[dict[str, Any]], job: str, *,
          now: float) -> list[dict[str, Any]]:
    """Wear a loadout after one job: single-use items that
    contributed are spent, durability items lose one use,
    cooldown items go on cooldown.  Returns the survivors."""
    out = []
    for inst in instances:
        spec = ARSENAL.get(str(inst.get("item") or ""))
        if spec is None:
            continue
        used = int((spec.get("ratings") or {}).get(job) or 0) > 0
        if not used:
            out.append(inst)
            continue
        if spec.get("single_use"):
            continue  # spent whole
        if int(spec.get("cooldown_s") or 0) > 0:
            inst = dict(inst, cd_until=now + int(
                spec["cooldown_s"]))
            out.append(inst)
            continue
        if inst.get("durability") is not None:
            inst = dict(inst, durability=max(
                0, int(inst["durability"]) - 1))
            if int(inst["durability"]) <= 0:
                continue  # broke on the job
        out.append(inst)
    return out


def gear_power(instances: list[dict[str, Any]], job: str, *,
               now: float,
               affinity: list[str] | None = None,
               affinity_bonus: int = 0) -> int:
    """The best-three live items' rating for one job — a
    loadout is a hand, not a warehouse: hauling everything
    you own adds nothing past the third tool."""
    scored = []
    for inst in _live_instances(instances, now):
        item = ARSENAL.get(str(inst.get("item") or ""))
        if item is None:
            continue
        rating = int((item.get("ratings") or {}).get(job) or 0)
        if rating <= 0:
            continue
        if affinity and inst.get("item") in affinity:
            rating += int(affinity_bonus)
        scored.append(rating)
    scored.sort(reverse=True)
    return sum(scored[:3])


def combat_power_of(instances: list[dict[str, Any]], *,
                    now: float, rep: int = 0,
                    affinity: list[str] | None = None,
                    affinity_bonus: int = 0) -> int:
    """A member's street power: base + best hit gear + the
    armor they wear + street rep earned in the conflict
    journal (10 rep = +1)."""
    hit = gear_power(instances, "hit", now=now,
                     affinity=affinity,
                     affinity_bonus=affinity_bonus)
    armor = 0
    for inst in _live_instances(instances, now):
        item = ARSENAL.get(str(inst.get("item") or ""))
        if item is not None and item.get("role") == "armor":
            armor = max(armor, int(
                (item.get("ratings") or {}).get("hit") or 0))
    return 10 + hit + armor + int(rep) // 10


def counter_adjust(attack_items: list[str],
                   defense_items: list[str], job: str
                   ) -> tuple[int, int, list[str]]:
    """Counter-pairs (§23 addendum): returns (offense bonus,
    defense cut, notes).  A breached defense that is countered
    keeps only a quarter of its rating; an attacker whose key
    tool is countered by the defense loses half that tool."""
    notes: list[str] = []
    defense_cut = 0
    offense_bonus = 0
    for ditem in defense_items:
        spec = ARSENAL.get(ditem) or {}
        rating = int((spec.get("ratings") or {}).get(job) or 0)
        if rating <= 0:
            continue
        for counter in spec.get("countered_by") or []:
            if counter in attack_items:
                defense_cut += int(rating * 0.75)
                notes.append(
                    f"{ARSENAL[counter]['name']} beats "
                    f"{spec['name']}")
                break
    for aitem in attack_items:
        spec = ARSENAL.get(aitem) or {}
        if int((spec.get("ratings") or {}).get(job) or 0) <= 0:
            continue
        for counter in spec.get("counters") or []:
            if counter in defense_items:
                offense_bonus += 4
                break
    return offense_bonus, defense_cut, notes


def heist_odds(offense: int, defense: int, *,
               heat_stars: float = 0.0) -> float:
    """The readable gamble: success probability from the
    margin, clamped so no job is a certainty or a fantasy."""
    margin = offense - defense - 4.0 * heat_stars
    p = 0.5 + margin / 60.0
    return max(0.05, min(0.95, p))


# ── the organ ────────────────────────────────────────────────────

def _empty_state() -> dict[str, Any]:
    return {"loadouts": {}, "defenses": {}, "gangs": {},
            "gang_of": {}, "wars": {}, "territory": {},
            "journal": [], "rep": {}, "heists": {},
            "police": {"acted": {}, "forgiven": {},
                       "extra_heat": {}, "bounties": {},
                       "report_credited": {}},
            "drops": {}, "drop_day": {}, "escrow_inter": {},
            "nuke_minted": False, "job_cd": {},
            "territory_day": "", "cycle_war_at": 0.0}


class GroupConflict:
    """Crime, gangs, wars, territory, police and drops for
    one group's economy.  Constructed over the group's
    economy organ; the estate organ rides along for gems and
    property facts."""

    def __init__(self, economy: GroupEconomy) -> None:
        self.economy = economy
        self._estate = GroupRealEstate(economy)

    @property
    def estate(self) -> GroupRealEstate:
        return self._estate

    # ── persistence (estate discipline) ──

    def _path(self, group_id: str) -> Path:
        return self.economy._path(group_id, "conflict")

    def _load_state(self, group_id: str) -> dict[str, Any]:
        path = self._path(group_id)
        if not path.exists():
            return _empty_state()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GroupConflictError(
                f"conflict state for {group_id!r} is unreadable "
                f"({exc}); refusing to read it as empty") from exc
        if not isinstance(data, dict):
            raise GroupConflictError(
                f"conflict state for {group_id!r} is not an "
                "object; refusing to read it as empty")
        state = _empty_state()
        for key, value in data.items():
            state[key] = value
        police = dict(state.get("police") or {})
        for key, value in _empty_state()["police"].items():
            police.setdefault(key, value)
        state["police"] = police
        return state

    def _save_state(self, group_id: str,
                    state: dict[str, Any]) -> None:
        self.economy._save_json(group_id, "conflict", state)

    def health(self, group_id: str) -> dict[str, Any]:
        """Real counts for diagnostics (§11): gangs, wars,
        wanted members, open drops — read from the state
        itself, so a broken organ reports broken."""
        try:
            state = self._load_state(group_id)
        except GroupConflictError as exc:
            return {"ok": False, "error": str(exc)}
        now = time.time()
        wanted = sum(
            1 for member in list(state["rep"]) + list(
                state["police"]["acted"])
            if self.wanted_stars(group_id, member,
                                 state=state, now=now)[0] > 0)
        return {"ok": True,
                "gangs": len(state["gangs"]),
                "wars": len(state["wars"]),
                "loadouts": len(state["loadouts"]),
                "territory": len(state["territory"]),
                "wanted": wanted,
                "open_drops": sum(
                    1 for d in state["drops"].values()
                    if d.get("status") == "open"),
                "open_bounties": sum(
                    1 for b in state["police"]["bounties"].values()
                    if b.get("status") == "open")}

    # ── journal + rep ──

    @staticmethod
    def _journal(state: dict[str, Any], kind: str, now: float,
                 **fields: Any) -> None:
        state["journal"] = (list(state.get("journal") or []) + [
            {"kind": kind, "at": now, **fields}])[-600:]

    def add_rep(self, state: dict[str, Any], member: str,
                delta: int) -> int:
        rep = max(0, int((state.get("rep") or {}).get(
            member, 0)) + int(delta))
        state.setdefault("rep", {})[member] = rep
        return rep

    def reservations(self, group_id: str, *,
                     state: dict[str, Any] | None = None) -> int:
        """Treasury sparks already promised to the conflict
        systems: gang treasuries, in-group war pots, open
        bounties, inter-group escrows.  The surface folds
        this into its reservation accounting."""
        state = state if state is not None \
            else self._load_state(group_id)
        total = sum(int(g.get("treasury") or 0)
                    for g in state["gangs"].values())
        total += sum(int(g.get("hq_vault") or 0)
                     for g in state["gangs"].values())
        total += sum(int(w.get("pot") or 0)
                     for w in state["wars"].values()
                     if w.get("status") in ("declared", "active"))
        total += sum(int(b.get("amount") or 0)
                     for b in state["police"]["bounties"].values()
                     if b.get("status") == "open")
        total += sum(int(v) for v in
                     state["escrow_inter"].values())
        return total

    # ── gear: loadouts, the shop shelf, defenses ──

    def loadout(self, group_id: str, member: str, *,
                state: dict[str, Any] | None = None
                ) -> list[dict[str, Any]]:
        state = state if state is not None \
            else self._load_state(group_id)
        return list((state.get("loadouts") or {}).get(
            member) or [])

    def give_item(self, state: dict[str, Any], member: str,
                  item_id: str, now: float) -> None:
        state.setdefault("loadouts", {}).setdefault(
            member, []).append(_new_instance(item_id, now))

    def member_power(self, group_id: str, member: str, *,
                     state: dict[str, Any] | None = None,
                     now: float) -> int:
        state = state if state is not None \
            else self._load_state(group_id)
        gang = self.gang_of(group_id, member, state=state)
        affinity, bonus = [], 0
        if gang is not None:
            life = LIFESTYLES[gang["lifestyle"]]
            affinity = list(life["affinity"])
            bonus = int(life["affinity_bonus"])
        return combat_power_of(
            self.loadout(group_id, member, state=state),
            now=now, rep=int((state.get("rep") or {}).get(
                member, 0)),
            affinity=affinity, affinity_bonus=bonus)

    def buy_gear(self, group_id: str, member: str, item_id: str,
                 *, now: float, is_admin: bool = False
                 ) -> dict[str, Any]:
        """Buy one arsenal item through the purse: sparks off
        the ledger (kind ``gear_buy`` — a sink), gems through
        the estate organ's gated spend.  Both legs are checked
        before either moves; a failed second leg refunds the
        first."""
        spec = ARSENAL.get(item_id)
        if spec is None:
            return {"ok": False, "reason": "unknown"}
        state = self._load_state(group_id)
        if spec.get("admin_mint"):
            if not is_admin:
                return {"ok": False, "reason": "admin_only"}
            if state.get("nuke_minted"):
                return {"ok": False, "reason": "one_per_group"}
        econ = self.economy
        price = dict(spec.get("price") or {})
        sparks_due = int(price.get("sparks") or 0)
        gems_due = int(price.get("gems") or 0)
        if sparks_due and econ.balance(group_id, member) \
                < sparks_due:
            return {"ok": False, "reason": "sparks",
                    "sparks": sparks_due}
        if gems_due and self._estate.gems_balance(
                group_id, member) < gems_due:
            return {"ok": False, "reason": "gems",
                    "gems": gems_due}
        gems_spent = None
        if gems_due:
            gems_spent = self._estate._spend_gems(
                group_id, member, gems_due, kind="spend",
                reason=f"arsenal: {spec['name']}",
                ref=f"gear:{item_id}", now=now)
            if gems_spent is None:  # pragma: no cover - raced
                return {"ok": False, "reason": "gems",
                        "gems": gems_due}
        if sparks_due and not econ._move(
                group_id, member, "", sparks_due, "gear_buy",
                f"bought {spec['name']}", now):
            if gems_spent is not None:
                self._estate.refund_gems(
                    group_id, member, gems_due,
                    reason=f"arsenal refund: {spec['name']}",
                    now=now)
            return {"ok": False, "reason": "sparks",
                    "sparks": sparks_due}
        if spec.get("admin_mint"):
            state["nuke_minted"] = True
        self.give_item(state, member, item_id, now)
        self._journal(state, "gear_buy", now, member=member,
                      item=item_id)
        self._save_state(group_id, state)
        return {"ok": True, "item": item_id,
                "sparks": sparks_due, "gems": gems_due}

    def use_consumable(self, group_id: str, member: str,
                       item_id: str, *, now: float) -> bool:
        """Spend one held consumable (medkit, bribe envelope,
        repair kit…).  True when one was held and spent."""
        state = self._load_state(group_id)
        items = list((state.get("loadouts") or {}).get(
            member) or [])
        for idx, inst in enumerate(items):
            if inst.get("item") == item_id and float(
                    inst.get("cd_until") or 0) <= now:
                items.pop(idx)
                state["loadouts"][member] = items
                self._save_state(group_id, state)
                return True
        return False

    def repair_item(self, group_id: str, member: str, *,
                    now: float) -> dict[str, Any]:
        """A repair kit restores the member's most-worn tool
        to full durability."""
        state = self._load_state(group_id)
        items = list((state.get("loadouts") or {}).get(
            member) or [])
        target = None
        for inst in items:
            spec = ARSENAL.get(str(inst.get("item") or ""))
            if spec is None or spec.get("durability") is None \
                    or spec.get("single_use"):
                continue
            if inst.get("durability") is not None and int(
                    inst["durability"]) < int(spec["durability"]):
                target = inst
                break
        if target is None:
            return {"ok": False, "reason": "nothing_worn"}
        if not self.use_consumable(group_id, member,
                                   "repair_kit", now=now):
            return {"ok": False, "reason": "no_kit"}
        state = self._load_state(group_id)
        for inst in (state.get("loadouts") or {}).get(
                member) or []:
            if inst is target or (inst.get("item") == target.get(
                    "item") and inst.get("durability") == target.get(
                    "durability")):
                inst["durability"] = ARSENAL[
                    str(inst["item"])]["durability"]
                break
        self._save_state(group_id, state)
        return {"ok": True, "item": target.get("item")}

    def defenses_at(self, group_id: str, scope: str, *,
                    state: dict[str, Any] | None = None
                    ) -> list[dict[str, Any]]:
        state = state if state is not None \
            else self._load_state(group_id)
        return list((state.get("defenses") or {}).get(
            scope) or [])

    def install_defense(self, group_id: str, member: str,
                        item_id: str, scope: str, *,
                        now: float) -> dict[str, Any]:
        """Move one owned defense item from the member's
        loadout onto a target: ``treasury``, ``bank:<district>``
        or ``vault:<member>`` (scope rights are the caller's —
        the surface checks admin/power for shared scopes)."""
        spec = ARSENAL.get(item_id)
        if spec is None or spec.get("role") != "defense":
            return {"ok": False, "reason": "not_defense"}
        if not (scope == "treasury" or scope.startswith("bank:")
                or scope.startswith("vault:")):
            return {"ok": False, "reason": "scope"}
        if scope.startswith("bank:") and scope[5:] not in DISTRICTS:
            return {"ok": False, "reason": "district"}
        state = self._load_state(group_id)
        items = list((state.get("loadouts") or {}).get(
            member) or [])
        for idx, inst in enumerate(items):
            if inst.get("item") == item_id:
                items.pop(idx)
                state["loadouts"][member] = items
                state.setdefault("defenses", {}).setdefault(
                    scope, []).append(inst)
                self._journal(state, "defense_install", now,
                              member=member, item=item_id,
                              scope=scope)
                self._save_state(group_id, state)
                return {"ok": True, "scope": scope}
        return {"ok": False, "reason": "not_owned"}

    # ── gangs ──

    def gang_of(self, group_id: str, member: str, *,
                state: dict[str, Any] | None = None
                ) -> dict[str, Any] | None:
        state = state if state is not None \
            else self._load_state(group_id)
        gid = (state.get("gang_of") or {}).get(member)
        if not gid:
            return None
        return (state.get("gangs") or {}).get(gid)

    def gang_by_name(self, group_id: str, name: str, *,
                     state: dict[str, Any] | None = None
                     ) -> dict[str, Any] | None:
        state = state if state is not None \
            else self._load_state(group_id)
        wanted = str(name or "").strip().lower()
        for gang in (state.get("gangs") or {}).values():
            if str(gang.get("name") or "").lower() == wanted:
                return gang
        return None

    def create_gang(self, group_id: str, member: str, name: str,
                    lifestyle: str, *, now: float
                    ) -> dict[str, Any]:
        name = " ".join(str(name or "").split())[:32]
        if len(name) < 3:
            return {"ok": False, "reason": "name"}
        if lifestyle not in LIFESTYLES:
            return {"ok": False, "reason": "lifestyle"}
        state = self._load_state(group_id)
        if (state.get("gang_of") or {}).get(member):
            return {"ok": False, "reason": "already_in"}
        if self.gang_by_name(group_id, name, state=state):
            return {"ok": False, "reason": "taken"}
        gid = new_short_id("gang")
        gang = {"id": gid, "name": name, "lifestyle": lifestyle,
                "leader": member, "officers": [],
                "members": [member], "created": now,
                "treasury": 0, "armory": [], "tax_pct": None,
                "wins": 0, "losses": 0}
        state["gangs"][gid] = gang
        state["gang_of"][member] = gid
        self._journal(state, "gang_create", now, member=member,
                      gang=gid, name=name, lifestyle=lifestyle)
        self._save_state(group_id, state)
        return {"ok": True, "gang": gang}

    def join_gang(self, group_id: str, member: str, name: str,
                  *, now: float) -> dict[str, Any]:
        state = self._load_state(group_id)
        if (state.get("gang_of") or {}).get(member):
            return {"ok": False, "reason": "already_in"}
        gang = self.gang_by_name(group_id, name, state=state)
        if gang is None:
            return {"ok": False, "reason": "not_found"}
        gang["members"] = list(gang.get("members") or []) + [member]
        state["gang_of"][member] = gang["id"]
        self._journal(state, "gang_join", now, member=member,
                      gang=gang["id"])
        self._save_state(group_id, state)
        return {"ok": True, "gang": gang}

    def leave_gang(self, group_id: str, member: str, *,
                   now: float) -> dict[str, Any]:
        state = self._load_state(group_id)
        gang = self.gang_of(group_id, member, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        gang["members"] = [m for m in gang.get("members") or []
                           if m != member]
        gang["officers"] = [m for m in gang.get("officers") or []
                            if m != member]
        state["gang_of"].pop(member, None)
        dissolved = False
        payout = 0
        if not gang["members"]:
            payout = int(gang.get("treasury") or 0)
            dissolved = True
            for district, hold in list(
                    state["territory"].items()):
                if hold.get("gang") == gang["id"]:
                    state["territory"].pop(district, None)
            state["gangs"].pop(gang["id"], None)
        elif gang.get("leader") == member:
            gang["leader"] = (list(gang.get("officers") or [])
                              + list(gang["members"]))[0]
            gang["officers"] = [m for m in gang.get("officers")
                                 or [] if m != gang["leader"]]
        self._journal(state, "gang_leave", now, member=member,
                      gang=gang["id"], dissolved=dissolved)
        self._save_state(group_id, state)
        if dissolved and payout and gang.get("leader"):
            self.economy.treasury_grant(
                group_id, str(gang["leader"]), payout,
                reason=f"{gang['name']} dissolved — treasury "
                       "to its last leader", now=now)
        return {"ok": True, "dissolved": dissolved,
                "gang": gang}

    def gang_set_role(self, group_id: str, actor: str,
                      target: str, role: str, *, now: float
                      ) -> dict[str, Any]:
        state = self._load_state(group_id)
        gang = self.gang_of(group_id, actor, state=state)
        if gang is None or gang.get("leader") != actor:
            return {"ok": False, "reason": "not_leader"}
        if target not in (gang.get("members") or []):
            return {"ok": False, "reason": "not_member"}
        if target == actor:
            return {"ok": False, "reason": "self"}
        officers = [m for m in gang.get("officers") or []
                    if m != target]
        if role == "officer":
            officers.append(target)
        elif role == "leader":
            gang["officers"] = [m for m in officers
                                 if m != actor] + [actor]
            gang["leader"] = target
            self._journal(state, "gang_role", now, member=target,
                          gang=gang["id"], role="leader")
            self._save_state(group_id, state)
            return {"ok": True, "gang": gang}
        gang["officers"] = officers
        self._journal(state, "gang_role", now, member=target,
                      gang=gang["id"], role=role)
        self._save_state(group_id, state)
        return {"ok": True, "gang": gang}

    def gang_deposit(self, group_id: str, member: str,
                     amount: int, *, now: float
                     ) -> dict[str, Any]:
        """Wallet → group treasury (the economy's own
        contribution leg), credited to the gang's sub-account."""
        state = self._load_state(group_id)
        gang = self.gang_of(group_id, member, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        moved = self.economy.contribute_treasury(
            group_id, member, int(amount), now=now)
        if not moved:
            return {"ok": False, "reason": "funds"}
        gang["treasury"] = int(gang.get("treasury") or 0) \
            + int(amount)
        self._journal(state, "gang_deposit", now, member=member,
                      gang=gang["id"], amount=int(amount))
        self._save_state(group_id, state)
        return {"ok": True, "amount": int(amount), "gang": gang}

    def gang_skim_credit(self, group_id: str, gang_id: str,
                         amount: int, *, now: float
                         ) -> dict[str, Any]:
        """Credit a conversion skim (§25) to a gang's treasury.

        The estate leg (sparks into the group treasury) is the
        exchange's; this is the sub-account half — the gang's
        internal treasury number rises by the skim the exchange
        already moved, journaled here like every gang money
        fact.  Nothing is minted: the skim came out of a
        member's converted amount."""
        amount = int(amount)
        if amount <= 0:
            return {"ok": False, "reason": "amount"}
        state = self._load_state(group_id)
        gang = (state.get("gangs") or {}).get(str(gang_id))
        if gang is None:
            return {"ok": False, "reason": "gone"}
        gang["treasury"] = int(gang.get("treasury") or 0) + amount
        self._journal(state, "gang_skim", now, gang=str(gang_id),
                      amount=amount)
        self._save_state(group_id, state)
        return {"ok": True, "amount": amount, "gang": gang}

    def gang_withdraw(self, group_id: str, actor: str,
                      amount: int, *, now: float,
                      to: str | None = None) -> dict[str, Any]:
        """Gang treasury → a member's wallet, leader/officers
        only, through the economy's treasury grant."""
        state = self._load_state(group_id)
        gang = self.gang_of(group_id, actor, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        if actor != gang.get("leader") and actor not in (
                gang.get("officers") or []):
            return {"ok": False, "reason": "rank"}
        amount = int(amount)
        if amount <= 0 or int(gang.get("treasury") or 0) < amount:
            return {"ok": False, "reason": "funds",
                    "treasury": int(gang.get("treasury") or 0)}
        gang["treasury"] = int(gang["treasury"]) - amount
        self._save_state(group_id, state)
        paid = self.economy.treasury_grant(
            group_id, to or actor, amount,
            reason=f"{gang['name']} treasury payout", now=now)
        if not paid:  # treasury moved under us — restore
            state = self._load_state(group_id)
            gang2 = (state.get("gangs") or {}).get(gang["id"])
            if gang2 is not None:
                gang2["treasury"] = int(
                    gang2.get("treasury") or 0) + amount
                self._save_state(group_id, state)
            return {"ok": False, "reason": "treasury"}
        state = self._load_state(group_id)
        self._journal(state, "gang_withdraw", now, member=actor,
                      gang=gang["id"], amount=amount)
        self._save_state(group_id, state)
        return {"ok": True, "amount": amount}

    def gang_armory_add(self, group_id: str, member: str,
                        item_id: str, *, now: float
                        ) -> dict[str, Any]:
        """Donate one owned item to the gang armory (it fights
        wars and defends the gang's name)."""
        state = self._load_state(group_id)
        gang = self.gang_of(group_id, member, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        items = list((state.get("loadouts") or {}).get(
            member) or [])
        for idx, inst in enumerate(items):
            if inst.get("item") == item_id:
                items.pop(idx)
                state["loadouts"][member] = items
                gang.setdefault("armory", []).append(inst)
                self._save_state(group_id, state)
                return {"ok": True, "gang": gang}
        return {"ok": False, "reason": "not_owned"}

    def gang_strength(self, group_id: str, gang: dict[str, Any],
                      *, state: dict[str, Any] | None = None,
                      now: float) -> dict[str, int]:
        """The living composite (§24): members' power (best
        ten) + armory + territory + treasury depth + war
        record.  Every input is a number the system holds."""
        state = state if state is not None \
            else self._load_state(group_id)
        powers = sorted(
            (self.member_power(group_id, m, state=state, now=now)
             for m in gang.get("members") or []), reverse=True)
        members = sum(powers[:10])
        armory = gear_power(list(gang.get("armory") or []),
                            "war", now=now)
        territory = 25 * sum(
            1 for hold in state["territory"].values()
            if hold.get("gang") == gang.get("id"))
        treasury = min(20, int(gang.get("treasury") or 0) // 100)
        wins = 5 * int(gang.get("wins") or 0)
        parts = {"members": members, "armory": armory,
                 "territory": territory, "treasury": treasury,
                 "wins": wins}
        hq = hq_defense_bonus(gang)
        if hq:
            # the HQ concentration (§36): a gang is stronger
            # where its fortress stands — the key appears only
            # when it is real, so hq-less gangs read as before.
            parts["hq"] = hq
        parts["total"] = sum(parts.values())
        return parts

    def gang_job(self, group_id: str, member: str, *,
                 rng: Any, now: float) -> dict[str, Any]:
        """The gang's small-time work (§24): a lifestyle job
        paid like every other earn in the economy — a journaled
        treasury bonus the issuance throttle sees — on an
        hourly cooldown, with a territory kicker."""
        state = self._load_state(group_id)
        gang = self.gang_of(group_id, member, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        cd = dict(state.get("job_cd") or {})
        if float(cd.get(member) or 0) > now:
            return {"ok": False, "reason": "cooldown",
                    "until": float(cd[member])}
        life = LIFESTYLES[gang["lifestyle"]]
        label, low, high = rng.choice(list(life["jobs"]))
        amount = int(rng.randint(int(low), int(high)))
        holds = any(hold.get("gang") == gang["id"]
                    for hold in state["territory"].values())
        if holds:
            amount = int(amount * 1.1)
        # the wage is a journaled mint (treasury_bonus, the
        # issuance throttle sees it) paid straight through
        # the treasury to the member who worked
        minted = self.economy.treasury_bonus(
            group_id, amount,
            reason=f"{gang['name']} job: {label}", now=now)
        if not minted:
            return {"ok": False, "reason": "economy"}
        self.economy.treasury_grant(
            group_id, member, amount,
            reason=f"{gang['name']} job: {label}", now=now)
        cd[member] = now + 3600
        state["job_cd"] = cd
        self._journal(state, "gang_job", now, member=member,
                      gang=gang["id"], job=label, amount=amount)
        self._save_state(group_id, state)
        return {"ok": True, "job": label, "amount": amount,
                "gang": gang}


    def note_crime(self, group_id: str, *, actor: str,
                   target: str, verb: str, success: bool,
                   take: int = 0, shielded: bool = False,
                   now: float) -> None:
        """Journal a crime committed through ANOTHER organ
        (the shipped /rob lives in the economy) so the bully
        index sees the whole street, not just the new verbs."""
        state = self._load_state(group_id)
        econ = self.economy
        self._journal(
            state, "crime", now, actor=actor, target=target,
            verb=verb, success=bool(success), take=int(take),
            actor_total=econ.total_balance(group_id, actor),
            target_total=econ.total_balance(group_id, target),
            shielded=bool(shielded), heat=1.0)
        self._save_state(group_id, state)

    # ── heists (§23: crew loadout vs installed defenses) ──

    def start_heist(self, group_id: str, leader: str,
                    target_kind: str, target_id: str, *,
                    now: float) -> dict[str, Any]:
        if target_kind not in TARGET_BASE:
            return {"ok": False, "reason": "target"}
        state = self._load_state(group_id)
        for lobby in state["heists"].values():
            if lobby.get("status") == "open" and leader in (
                    lobby.get("crew") or []):
                return {"ok": False, "reason": "already_crew"}
        hid = new_short_id("heist")
        state["heists"][hid] = {
            "id": hid, "leader": leader,
            "target": {"kind": target_kind, "id": target_id},
            "crew": [leader], "created": now, "status": "open"}
        self._save_state(group_id, state)
        return {"ok": True, "heist": state["heists"][hid]}

    def join_heist(self, group_id: str, member: str, hid: str,
                   *, now: float) -> dict[str, Any]:
        state = self._load_state(group_id)
        lobby = state["heists"].get(hid)
        if lobby is None or lobby.get("status") != "open":
            return {"ok": False, "reason": "not_found"}
        if now - float(lobby.get("created") or now) > 600:
            lobby["status"] = "expired"
            self._save_state(group_id, state)
            return {"ok": False, "reason": "expired"}
        if member in (lobby.get("crew") or []):
            return {"ok": False, "reason": "already"}
        if len(lobby.get("crew") or []) >= 5:
            return {"ok": False, "reason": "full"}
        for other in state["heists"].values():
            if other.get("status") == "open" and member in (
                    other.get("crew") or []):
                return {"ok": False, "reason": "already_crew"}
        lobby["crew"] = list(lobby.get("crew") or []) + [member]
        self._save_state(group_id, state)
        return {"ok": True, "heist": lobby}

    def heist_preview(self, group_id: str, hid: str, *,
                      now: float) -> dict[str, Any] | None:
        """The odds BEFORE the job (the readable gamble): the
        same math resolve_heist runs, without the roll."""
        state = self._load_state(group_id)
        lobby = state["heists"].get(hid)
        if lobby is None:
            return None
        math = self._heist_math(group_id, lobby, state=state,
                                now=now)
        return math

    def _heist_math(self, group_id: str, lobby: dict[str, Any],
                    *, state: dict[str, Any], now: float
                    ) -> dict[str, Any]:
        target = dict(lobby.get("target") or {})
        kind = str(target.get("kind") or "vault")
        job = "heist" if kind in ("treasury", "bank") else "vault"
        scope = {"treasury": "treasury",
                 "bank": f"bank:{target.get('id')}",
                 "vault": f"vault:{target.get('id')}"}[kind]
        offense = 0
        attack_items: list[str] = []
        heat_items = 0.0
        cool = 0.0
        for member in lobby.get("crew") or []:
            gang = self.gang_of(group_id, member, state=state)
            affinity, bonus = [], 0
            if gang is not None:
                life = LIFESTYLES[gang["lifestyle"]]
                affinity = list(life["affinity"])
                bonus = int(life["affinity_bonus"])
            items = self.loadout(group_id, member, state=state)
            offense += 2 + gear_power(
                items, job, now=now, affinity=affinity,
                affinity_bonus=bonus)
            for inst in _live_instances(items, now):
                iid = str(inst.get("item") or "")
                spec = ARSENAL.get(iid) or {}
                if int((spec.get("ratings") or {}).get(
                        job) or 0) > 0:
                    attack_items.append(iid)
                    heat_items += float(HEAT_ITEMS.get(iid, 0.0))
                    cool += float(COOL_ITEMS.get(iid, 0.0))
        defenses = self.defenses_at(group_id, scope, state=state)
        defense_items = [str(i.get("item") or "")
                         for i in _live_instances(defenses, now)]
        defense = int(TARGET_BASE[kind]) + sum(
            int((ARSENAL.get(iid, {}).get("ratings") or {}).get(
                job) or 0) for iid in defense_items)
        bonus, cut, notes = counter_adjust(
            sorted(set(attack_items)), defense_items, job)
        offense += bonus
        defense -= cut
        # HQ concentration (§36): a gang's headquarters in this
        # group watches the town's money — treasury and bank
        # scores face the strongest HQ here; a member vault
        # faces its own gang's HQ when it stands here.
        hq_bonus = 0
        if kind in ("treasury", "bank"):
            hq_bonus = max(
                (hq_defense_bonus(g) for g in
                 (state.get("gangs") or {}).values()),
                default=0)
        else:
            owner_gang = self.gang_of(
                group_id, str(target.get("id") or ""),
                state=state)
            hq_bonus = hq_defense_bonus(owner_gang)
        if hq_bonus:
            defense += hq_bonus
            notes = list(notes) + [
                f"an HQ watches this score (+{hq_bonus} "
                "defense)"]
        stars = 0.0
        for member in lobby.get("crew") or []:
            member_stars, _, _ = self.wanted_stars(
                group_id, member, state=state, now=now)
            stars = max(stars, float(member_stars))
        odds = heist_odds(offense, defense, heat_stars=stars)
        return {"job": job, "scope": scope, "offense": offense,
                "defense": defense, "odds": odds,
                "hq_defense": hq_bonus,
                "heat_stars": stars, "notes": notes,
                "attack_items": sorted(set(attack_items)),
                "defense_items": defense_items,
                "heat_items": heat_items, "cool": cool,
                "decoy": "decoy_vault" in defense_items}

    def resolve_heist(self, group_id: str, hid: str, *,
                      rng: Any, now: float,
                      available_treasury: int = 0,
                      shielded: dict[str, bool] | None = None
                      ) -> dict[str, Any]:
        """Run the job.  Outcomes: success (full take) /
        partial (40%, nobody jailed) / bust (the surface
        jails the crew) / decoy (a 'success' that pays
        nothing — the decoy vault did its job) / empty."""
        state = self._load_state(group_id)
        lobby = state["heists"].get(hid)
        if lobby is None or lobby.get("status") != "open":
            return {"ok": False, "reason": "not_found"}
        crew = list(lobby.get("crew") or [])
        if len(crew) < 2:
            return {"ok": False, "reason": "crew_small"}
        target = dict(lobby.get("target") or {})
        kind = str(target.get("kind") or "vault")
        math = self._heist_math(group_id, lobby, state=state,
                                now=now)
        odds = float(math["odds"])
        roll = float(rng.random())
        if roll < odds:
            outcome = "success"
        elif roll < odds + (1.0 - odds) * 0.25:
            outcome = "partial"
        else:
            outcome = "bust"
        econ = self.economy
        take = 0
        funds = 0
        if kind in ("treasury", "bank"):
            funds = max(0, int(available_treasury))
        else:
            funds = econ.balance(group_id, econ.bank_member(
                str(target.get("id") or "")))
        if outcome in ("success", "partial"):
            frac = float(TAKE_FRACTION[kind]) * (
                1.0 if outcome == "success" else 0.4)
            take = min(int(funds * frac), int(TAKE_CAP[kind]))
            if take <= 0:
                outcome, take = "empty", 0
            elif math["decoy"] and outcome == "success":
                outcome = "decoy"
                take = 0
        if take > 0:
            share = take // len(crew)
            paid: dict[str, int] = {}
            for member in crew:
                paid[member] = share
            paid[crew[0]] += take - share * len(crew)
            for member, amount in paid.items():
                if amount <= 0:
                    continue
                if kind in ("treasury", "bank"):
                    econ.treasury_grant(
                        group_id, member, amount,
                        reason=f"heist take ({kind})", now=now)
                else:
                    econ._move(group_id, econ.bank_member(
                        str(target.get("id") or "")), member,
                        amount, "heist",
                        "heist take (member vault)", now)
        # gear wears on the job — both sides
        job = str(math["job"])
        for member in crew:
            state["loadouts"][member] = _wear(
                self.loadout(group_id, member, state=state),
                job, now=now)
        scope = str(math["scope"])
        if scope in (state.get("defenses") or {}):
            state["defenses"][scope] = _wear(
                self.defenses_at(group_id, scope, state=state),
                job, now=now)
            if math["decoy"]:
                state["defenses"][scope] = [
                    i for i in state["defenses"][scope]
                    if i.get("item") != "decoy_vault"]
        lobby["status"] = "done"
        state["heists"][hid] = lobby
        heat = 1.0 + float(math["heat_items"]) - float(math["cool"])
        for member in crew:
            self.add_rep(state, member,
                         5 if outcome == "success"
                         else 2 if outcome == "partial" else -2)
            self._journal(
                state, "crime", now, actor=member,
                target=f"{kind}:{target.get('id') or ''}",
                verb="heist", success=outcome in (
                    "success", "partial"),
                take=take // max(1, len(crew)),
                actor_total=econ.total_balance(group_id, member),
                target_total=funds,
                shielded=bool((shielded or {}).get(member)),
                heat=max(0.5, heat))
        self._journal(state, "heist", now, crew=crew,
                      target=target, outcome=outcome, take=take,
                      odds=odds, offense=math["offense"],
                      defense=math["defense"],
                      attack_items=math["attack_items"],
                      defense_items=math["defense_items"],
                      notes=math["notes"])
        self._save_state(group_id, state)
        return {"ok": True, "outcome": outcome, "take": take,
                "odds": odds, "crew": crew, "target": target,
                "offense": math["offense"],
                "defense": math["defense"],
                "notes": math["notes"],
                "jail_min": 45 if outcome == "bust" else 0}

    # ── geared hit verbs (§23 beside /rob) ──

    def _verb_roll(self, group_id: str, actor: str, target: str,
                   *, state: dict[str, Any], now: float
                   ) -> tuple[float, int, int]:
        power_a = self.member_power(group_id, actor, state=state,
                                    now=now)
        power_d = self.member_power(group_id, target, state=state,
                                    now=now)
        p = 0.45 + (power_a - power_d) / 60.0
        return max(0.05, min(0.90, p)), power_a, power_d

    def _best_hit_rating(self, group_id: str, member: str, *,
                         state: dict[str, Any], now: float
                         ) -> int:
        best = 0
        for inst in _live_instances(
                self.loadout(group_id, member, state=state), now):
            spec = ARSENAL.get(str(inst.get("item") or "")) or {}
            best = max(best, int((spec.get("ratings") or {}).get(
                "hit") or 0))
        return best

    def hit_verb(self, group_id: str, actor: str, target: str,
                 verb: str, *, rng: Any, now: float,
                 shielded: bool = False) -> dict[str, Any]:
        """shoot / kill / assassinate — one math, three
        stakes.  The surface applies jail / hospital from the
        returned sentence; money moves here, on the ledger."""
        table = {
            "shoot": {"need": 0, "lo": 0.05, "hi": 0.10,
                      "cap": 100, "jail_fail": 10, "hospital": 0},
            "kill": {"need": 10, "lo": 0.15, "hi": 0.25,
                     "cap": 250, "jail_fail": 45, "hospital": 20},
            "assassinate": {"need": 14, "lo": 0.18, "hi": 0.28,
                            "cap": 400, "jail_fail": 60,
                            "hospital": 45},
        }
        spec = table[verb]
        state = self._load_state(group_id)
        if self._best_hit_rating(group_id, actor, state=state,
                                 now=now) < int(spec["need"]):
            return {"ok": False, "reason": "weapon",
                    "need": int(spec["need"])}
        odds, power_a, power_d = self._verb_roll(
            group_id, actor, target, state=state, now=now)
        econ = self.economy
        wallet = econ.balance(group_id, target)
        roll = float(rng.random())
        success = roll < odds
        take = 0
        if success:
            frac = float(spec["lo"]) + float(rng.random()) * (
                float(spec["hi"]) - float(spec["lo"]))
            take = min(int(wallet * frac), int(spec["cap"]))
            if take > 0:
                econ._move(group_id, target, actor, take, verb,
                           f"{verb} take", now)
            self.add_rep(state, actor, 3)
        state["loadouts"][actor] = _wear(
            self.loadout(group_id, actor, state=state), "hit",
            now=now)
        self._journal(
            state, "crime", now, actor=actor, target=target,
            verb=verb, success=success, take=take,
            actor_total=econ.total_balance(group_id, actor),
            target_total=econ.total_balance(group_id, target),
            shielded=bool(shielded), heat=1.0)
        self._journal(state, verb, now, actor=actor,
                      target=target, success=success, take=take,
                      odds=odds)
        self._save_state(group_id, state)
        return {"ok": True, "success": success, "take": take,
                "odds": odds, "power_a": power_a,
                "power_d": power_d,
                "jail_min": 0 if success else int(
                    spec["jail_fail"]),
                "hospital_min": int(spec["hospital"])
                if success else 0}

    def sabotage(self, group_id: str, actor: str, target: str,
                 *, rng: Any, now: float,
                 shielded: bool = False) -> dict[str, Any]:
        """Wreck the target's best property: the estate organ
        bills the damage through its own repair machinery
        (insurance answers first, then bank, wallet, debt)."""
        state = self._load_state(group_id)
        estate_state = self._estate._load_state(group_id)
        props = [p for p in (estate_state.get("properties")
                             or {}).values()
                 if str(p.get("owner") or "") == target]
        if not props:
            return {"ok": False, "reason": "no_property"}
        prop = max(props, key=lambda p: self._estate
                   .property_value(group_id, p))
        offense = gear_power(
            self.loadout(group_id, actor, state=state), "heist",
            now=now)
        defenses = self.defenses_at(
            group_id, f"vault:{target}", state=state)
        defense = sum(int((ARSENAL.get(
            str(i.get("item") or ""), {}).get("ratings") or {})
            .get("heist") or 0)
            for i in _live_instances(defenses, now))
        odds = max(0.05, min(0.90, 0.5 + (
            offense - defense) / 60.0))
        success = float(rng.random()) < odds
        bill = 0
        if success:
            value = self._estate.property_value(group_id, prop)
            if prop.get("currency") == "gems":
                from .group_real_estate import WEALTH_GEM_IN_SPARKS
                value *= WEALTH_GEM_IN_SPARKS
            bill = max(5, int(value * 0.08))
            self._estate.assess_repair_bill(
                group_id, str(prop["id"]), bill,
                cause="sabotage", now=now)
            self.add_rep(state, actor, 4)
        state["loadouts"][actor] = _wear(
            self.loadout(group_id, actor, state=state), "heist",
            now=now)
        econ = self.economy
        self._journal(
            state, "crime", now, actor=actor, target=target,
            verb="sabotage", success=success, take=bill,
            actor_total=econ.total_balance(group_id, actor),
            target_total=econ.total_balance(group_id, target),
            shielded=bool(shielded), heat=1.0)
        self._save_state(group_id, state)
        return {"ok": True, "success": success, "bill": bill,
                "odds": odds, "property": prop.get("name"),
                "jail_min": 0 if success else 20}

    # ── territory (§23: districts are the map) ──

    def territory_map(self, group_id: str, *,
                      state: dict[str, Any] | None = None
                      ) -> dict[str, str]:
        state = state if state is not None \
            else self._load_state(group_id)
        out: dict[str, str] = {}
        for district, hold in state["territory"].items():
            gang = (state.get("gangs") or {}).get(
                str(hold.get("gang") or ""))
            if gang is not None:
                out[district] = str(gang.get("name") or "")
        return out

    def claim_territory(self, group_id: str, actor: str,
                        district: str, *, now: float,
                        claim_cost: int = 100) -> dict[str, Any]:
        state = self._load_state(group_id)
        gang = self.gang_of(group_id, actor, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        if district not in DISTRICTS:
            return {"ok": False, "reason": "district"}
        if gang_tier(len(gang.get("members") or [])) == "crew":
            return {"ok": False, "reason": "tier"}
        if actor != gang.get("leader") and actor not in (
                gang.get("officers") or []):
            return {"ok": False, "reason": "rank"}
        if district in state["territory"]:
            return {"ok": False, "reason": "held"}
        if int(gang.get("treasury") or 0) < int(claim_cost):
            return {"ok": False, "reason": "funds",
                    "cost": int(claim_cost)}
        gang["treasury"] = int(gang["treasury"]) - int(claim_cost)
        state["territory"][district] = {
            "gang": gang["id"], "since": now, "arrears": 0}
        self._journal(state, "territory_claim", now,
                      member=actor, gang=gang["id"],
                      district=district)
        self._save_state(group_id, state)
        return {"ok": True, "gang": gang, "district": district}

    def cycle_territory(self, group_id: str, *, now: float,
                        cfg: dict[str, Any],
                        free_treasury: int) -> list[dict[str, Any]]:
        """Daily territory accounting on the cycle: held
        districts pay their gang from the group's free
        treasury (never minted — what the treasury cannot
        cover, it does not pay), and upkeep comes out of the
        gang's own treasury; two unpaid days releases the
        district.  Runs at most once per day per group."""
        state = self._load_state(group_id)
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if state.get("territory_day") == day:
            return []
        state["territory_day"] = day
        events: list[dict[str, Any]] = []
        base = int(cfg.get("territory_income_day") or 0)
        upkeep = int(cfg.get("territory_upkeep_day") or 0)
        props = self._estate._load_state(group_id).get(
            "properties") or {}
        per_district: dict[str, int] = {}
        for prop in props.values():
            per_district[str(prop.get("district"))] = \
                per_district.get(str(prop.get("district")), 0) + 1
        remaining = max(0, int(free_treasury))
        for district, hold in list(state["territory"].items()):
            gang = (state.get("gangs") or {}).get(
                str(hold.get("gang") or ""))
            if gang is None:
                state["territory"].pop(district, None)
                continue
            if int(gang.get("treasury") or 0) >= upkeep:
                gang["treasury"] = int(gang["treasury"]) - upkeep
                hold["arrears"] = 0
            else:
                hold["arrears"] = int(hold.get("arrears") or 0) + 1
            if int(hold.get("arrears") or 0) >= 2:
                state["territory"].pop(district, None)
                events.append({"kind": "territory_lost",
                               "district": district,
                               "gang": gang["name"],
                               "why": "upkeep unpaid"})
                self._journal(state, "territory_lost", now,
                              gang=gang["id"], district=district)
                continue
            income = base + 3 * min(20, per_district.get(
                district, 0))
            pay = min(income, remaining)
            remaining -= pay
            if pay > 0:
                gang["treasury"] = int(
                    gang.get("treasury") or 0) + pay
            events.append({"kind": "territory_income",
                           "district": district,
                           "gang": gang["name"], "amount": pay,
                           "full": pay >= income})
            self._journal(state, "territory_income", now,
                          gang=gang["id"], district=district,
                          amount=pay)
        self._save_state(group_id, state)
        return events

    # ── in-group wars ──

    def declare_war(self, group_id: str, actor: str,
                    target_name: str, *, now: float,
                    cfg: dict[str, Any]) -> dict[str, Any]:
        state = self._load_state(group_id)
        mine = self.gang_of(group_id, actor, state=state)
        if mine is None:
            return {"ok": False, "reason": "no_gang"}
        if actor != mine.get("leader") and actor not in (
                mine.get("officers") or []):
            return {"ok": False, "reason": "rank"}
        theirs = self.gang_by_name(group_id, target_name,
                                   state=state)
        if theirs is None:
            return {"ok": False, "reason": "not_found"}
        if theirs["id"] == mine["id"]:
            return {"ok": False, "reason": "self"}
        for war in state["wars"].values():
            if war.get("status") in ("declared", "active") and (
                    mine["id"] in (war.get("a"), war.get("b"))
                    or theirs["id"] in (war.get("a"),
                                        war.get("b"))):
                return {"ok": False, "reason": "busy"}
        escrow = int(cfg.get("war_escrow") or 0)
        if int(mine.get("treasury") or 0) < escrow:
            return {"ok": False, "reason": "funds",
                    "escrow": escrow}
        mine["treasury"] = int(mine["treasury"]) - escrow
        wid = new_short_id("war")
        state["wars"][wid] = {
            "id": wid, "a": mine["id"], "b": theirs["id"],
            "status": "declared", "pot": escrow,
            "escrow": escrow, "declared_at": now,
            "participants": {"a": [], "b": []},
            "score": {"a": 0, "b": 0}, "rounds_done": 0,
            "next_round_at": 0.0, "spoils_pending": False,
            "log": []}
        self._journal(state, "war_declare", now, member=actor,
                      gang=mine["id"], target=theirs["id"],
                      escrow=escrow)
        self._save_state(group_id, state)
        return {"ok": True, "war": state["wars"][wid],
                "mine": mine, "theirs": theirs}

    def answer_war(self, group_id: str, actor: str, wid: str,
                   accept: bool, *, now: float,
                   cfg: dict[str, Any]) -> dict[str, Any]:
        state = self._load_state(group_id)
        war = state["wars"].get(wid)
        if war is None or war.get("status") != "declared":
            return {"ok": False, "reason": "not_found"}
        theirs = (state.get("gangs") or {}).get(str(war.get("b")))
        if theirs is None:
            return {"ok": False, "reason": "gone"}
        if actor != theirs.get("leader") and actor not in (
                theirs.get("officers") or []):
            return {"ok": False, "reason": "rank"}
        escrow = int(war.get("escrow") or 0)
        if accept:
            if int(theirs.get("treasury") or 0) < escrow:
                return {"ok": False, "reason": "funds",
                        "escrow": escrow}
            theirs["treasury"] = int(theirs["treasury"]) - escrow
            war["pot"] = int(war.get("pot") or 0) + escrow
            war["status"] = "active"
            war["next_round_at"] = now + float(
                cfg.get("war_round_s") or 3600)
        else:
            mine = (state.get("gangs") or {}).get(str(war.get("a")))
            if mine is not None:
                mine["treasury"] = int(
                    mine.get("treasury") or 0) + escrow
            war["status"] = "declined"
        self._journal(state, "war_answer", now, member=actor,
                      war=wid, accept=accept)
        self._save_state(group_id, state)
        return {"ok": True, "war": war}

    def war_for_gang(self, group_id: str, gang_id: str, *,
                     state: dict[str, Any] | None = None
                     ) -> dict[str, Any] | None:
        state = state if state is not None \
            else self._load_state(group_id)
        for war in state["wars"].values():
            if war.get("status") in ("declared", "active") and \
                    gang_id in (war.get("a"), war.get("b")):
                return war
        return None

    def war_join(self, group_id: str, member: str, *,
                 now: float) -> dict[str, Any]:
        state = self._load_state(group_id)
        gang = self.gang_of(group_id, member, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        war = self.war_for_gang(group_id, gang["id"], state=state)
        if war is None or war.get("status") != "active":
            return {"ok": False, "reason": "no_war"}
        side = "a" if war.get("a") == gang["id"] else "b"
        parts = dict(war.get("participants") or {})
        mine = list(parts.get(side) or [])
        if member in mine:
            return {"ok": False, "reason": "already"}
        mine.append(member)
        parts[side] = mine
        war["participants"] = parts
        self._save_state(group_id, state)
        return {"ok": True, "war": war, "side": side}

    def _side_power(self, group_id: str, war: dict[str, Any],
                    side: str, *, state: dict[str, Any],
                    now: float, rng: Any) -> float:
        gang = (state.get("gangs") or {}).get(
            str(war.get(side) or ""))
        if gang is None:
            return 0.0
        life = LIFESTYLES[gang["lifestyle"]]
        parts = list((war.get("participants") or {}).get(
            side) or [])
        if not parts:
            parts = [str(gang.get("leader") or "")]
        power = sum(self.member_power(group_id, m, state=state,
                                      now=now) for m in parts)
        power += gear_power(list(gang.get("armory") or []),
                            "war", now=now)
        # home turf (§36): a gang fighting where its HQ stands
        # fights with the fortress behind it.
        power += hq_defense_bonus(gang)
        power *= float(life["war_mult"])
        return power * float(rng.uniform(0.85, 1.15))

    def war_round(self, group_id: str, wid: str, *,
                  rng: Any, now: float, cfg: dict[str, Any]
                  ) -> dict[str, Any] | None:
        """One engagement: opted-in members + armory +
        lifestyle war style, rolled inside ±15%.  The
        stronger gang stomping a half-strength one feeds the
        winners' bully index (war aggression is journaled
        aggression)."""
        state = self._load_state(group_id)
        war = state["wars"].get(wid)
        if war is None or war.get("status") != "active":
            return None
        gang_a = (state.get("gangs") or {}).get(str(war.get("a")))
        gang_b = (state.get("gangs") or {}).get(str(war.get("b")))
        if gang_a is None or gang_b is None:
            war["status"] = "void"
            self._save_state(group_id, state)
            return {"war": war, "void": True}
        strength_a = self.gang_strength(
            group_id, gang_a, state=state, now=now)["total"]
        strength_b = self.gang_strength(
            group_id, gang_b, state=state, now=now)["total"]
        power_a = self._side_power(group_id, war, "a",
                                   state=state, now=now, rng=rng)
        power_b = self._side_power(group_id, war, "b",
                                   state=state, now=now, rng=rng)
        if strength_a > 0 and strength_b < 0.5 * strength_a:
            stomp = "a"
        elif strength_b > 0 and strength_a < 0.5 * strength_b:
            stomp = "b"
        else:
            stomp = ""
        # the underdog fights above its weight (§24)
        if stomp == "a":
            power_b *= 1.25
        elif stomp == "b":
            power_a *= 1.25
        winner = "a" if power_a >= power_b else "b"
        score = dict(war.get("score") or {"a": 0, "b": 0})
        score[winner] = int(score.get(winner) or 0) + 1
        war["score"] = score
        war["rounds_done"] = int(war.get("rounds_done") or 0) + 1
        war["log"] = (list(war.get("log") or []) + [{
            "round": war["rounds_done"], "winner": winner,
            "power_a": round(power_a, 1),
            "power_b": round(power_b, 1), "at": now}])[-9:]
        if stomp:
            win_gang = gang_a if stomp == "a" else gang_b
            for member in (war.get("participants") or {}).get(
                    stomp) or [win_gang.get("leader")]:
                self._journal(
                    state, "crime", now, actor=str(member),
                    target=f"gang:{(gang_b if stomp == 'a' else gang_a)['name']}",
                    verb="war_stomp", success=True, take=0,
                    actor_total=0, target_total=0,
                    shielded=False, heat=1.0)
        total_rounds = int(cfg.get("war_rounds") or 3)
        needed = total_rounds // 2 + 1
        event: dict[str, Any] = {
            "war": war, "round": war["rounds_done"],
            "winner_gang": (gang_a if winner == "a" else gang_b),
            "loser_gang": (gang_b if winner == "a" else gang_a),
            "power_a": power_a, "power_b": power_b,
            "finished": False}
        if int(score.get(winner) or 0) >= needed or \
                war["rounds_done"] >= total_rounds:
            win_gang = gang_a if int(score.get("a") or 0) > int(
                score.get("b") or 0) else gang_b
            lose_gang = gang_b if win_gang is gang_a else gang_a
            win_gang["treasury"] = int(
                win_gang.get("treasury") or 0) + int(
                war.get("pot") or 0)
            win_gang["wins"] = int(win_gang.get("wins") or 0) + 1
            lose_gang["losses"] = int(
                lose_gang.get("losses") or 0) + 1
            holds_loser = [d for d, h in state["territory"]
                           .items() if h.get("gang")
                           == lose_gang["id"]]
            war["status"] = "done"
            war["winner"] = win_gang["id"]
            war["spoils_pending"] = bool(holds_loser)
            war["spoils_from"] = lose_gang["id"]
            event.update({"finished": True,
                          "winner_gang": win_gang,
                          "loser_gang": lose_gang,
                          "pot": int(war.get("pot") or 0),
                          "spoils_pending": bool(holds_loser)})
            self._journal(state, "war_done", now,
                          gang=win_gang["id"],
                          loser=lose_gang["id"],
                          pot=int(war.get("pot") or 0))
        else:
            war["next_round_at"] = now + float(
                cfg.get("war_round_s") or 3600)
        self._save_state(group_id, state)
        return event

    def war_claim_spoils(self, group_id: str, actor: str,
                         district: str, *, now: float
                         ) -> dict[str, Any]:
        """The in-group war prize: one district held by the
        loser changes hands."""
        state = self._load_state(group_id)
        gang = self.gang_of(group_id, actor, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        if actor != gang.get("leader") and actor not in (
                gang.get("officers") or []):
            return {"ok": False, "reason": "rank"}
        for war in state["wars"].values():
            if war.get("status") == "done" and war.get(
                    "spoils_pending") and war.get(
                    "winner") == gang["id"]:
                hold = state["territory"].get(district)
                if hold is None or hold.get("gang") != war.get(
                        "spoils_from"):
                    return {"ok": False, "reason": "district"}
                state["territory"][district] = {
                    "gang": gang["id"], "since": now,
                    "arrears": 0}
                war["spoils_pending"] = False
                self._journal(state, "territory_spoils", now,
                              gang=gang["id"], district=district)
                self._save_state(group_id, state)
                return {"ok": True, "district": district,
                        "gang": gang}
        return {"ok": False, "reason": "no_spoils"}

    def cycle_wars(self, group_id: str, *, rng: Any, now: float,
                   cfg: dict[str, Any]) -> list[dict[str, Any]]:
        state = self._load_state(group_id)
        events = []
        for wid, war in list(state["wars"].items()):
            if war.get("status") == "declared" and now - float(
                    war.get("declared_at") or now) > 86400:
                mine = (state.get("gangs") or {}).get(
                    str(war.get("a")))
                if mine is not None:
                    mine["treasury"] = int(
                        mine.get("treasury") or 0) + int(
                        war.get("escrow") or 0)
                war["status"] = "expired"
                events.append({"kind": "war_expired",
                               "war": war})
        self._save_state(group_id, state)
        state = self._load_state(group_id)
        for wid, war in list(state["wars"].items()):
            if war.get("status") == "active" and float(
                    war.get("next_round_at") or 0) <= now:
                event = self.war_round(group_id, wid, rng=rng,
                                       now=now, cfg=cfg)
                if event is not None:
                    event["kind"] = "war_round"
                    events.append(event)
        return events


    # ── the police (§24: a system faction that hunts excess) ──

    def bully_index(self, group_id: str, member: str, *,
                    state: dict[str, Any] | None = None,
                    now: float,
                    focus: float = 1.0) -> tuple[float, list[str]]:
        """The bully index, recomputed from the crime journal
        — the system's own evidence, never a vibe.  Returns
        (index, human-readable inputs) so an admin reading
        /glog sees exactly why the police moved.  ``focus`` is
        the owner's enforcement-focus multiplier for the group
        (§36: the aggression setting the thresholds read);
        1.0 — the default everywhere it isn't wired — changes
        nothing."""
        state = state if state is not None \
            else self._load_state(group_id)
        since = now - INDEX_WINDOW_S
        events = [e for e in state.get("journal") or []
                  if e.get("kind") == "crime"
                  and e.get("actor") == member
                  and float(e.get("at") or 0) >= since]
        total = 0.0
        inputs: list[str] = []
        hits_on: dict[str, int] = {}
        life = None
        gang = self.gang_of(group_id, member, state=state)
        if gang is not None:
            life = LIFESTYLES[gang["lifestyle"]]
        for event in events:
            verb = str(event.get("verb") or "")
            if verb == "war_stomp":
                total += INDEX_POINTS["war_stomp"]
                inputs.append("war stomp (gang at <½ strength)")
                continue
            if verb == "heist":
                total += INDEX_POINTS["heist"] * float(
                    event.get("heat") or 1.0)
                inputs.append("heist heat")
                continue
            if not event.get("success"):
                continue
            total += INDEX_POINTS["base"]
            target = str(event.get("target") or "")
            actor_total = float(event.get("actor_total") or 0)
            target_total = float(event.get("target_total") or 0)
            bits = [verb]
            if actor_total > 0 and target_total <= 0.35 * actor_total:
                total += INDEX_POINTS["punch_down"]
                bits.append("punched down")
            if event.get("shielded"):
                total += INDEX_POINTS["shield_break"]
                bits.append("hit a newcomer")
            hits_on[target] = hits_on.get(target, 0) + 1
            if hits_on[target] > 2:
                total += INDEX_POINTS["repeat"]
                bits.append("repeat victim")
            inputs.append(" + ".join(bits))
        police = state.get("police") or {}
        credited = (police.get("report_credited") or {}).get(member)
        week = time.strftime("%Y-%W", time.localtime(now))
        if credited == week:
            total += INDEX_POINTS["reports"]
            inputs.append("community reports (3+ members)")
        for grant in (police.get("extra_heat") or {}).get(
                member) or []:
            if float(grant.get("at") or 0) >= since:
                total += float(grant.get("points") or 0)
                inputs.append("extra heat (failed bribe)")
        forgiven = 0.0
        for grant in (police.get("forgiven") or {}).get(
                member) or []:
            if float(grant.get("at") or 0) >= since:
                forgiven += float(grant.get("points") or 0)
        total = max(0.0, total - forgiven)
        if life is not None:
            total *= float(life["heat_mult"])
        if focus != 1.0:
            total *= float(focus)
            inputs.append(
                f"enforcement focus ×{float(focus):g} (owner)")
        return round(total, 2), inputs

    def owner_mark(self, group_id: str, member: str, *,
                   state: dict[str, Any] | None = None,
                   now: float) -> int | None:
        """The owner's standing wanted mark on a member (§36),
        or None: a mark set inside the index window HOLDS the
        member's stars at the marked level — the ladder reads
        the mark, the journal keeps the evidence.  Written only
        through the registry's override verbs."""
        state = state if state is not None \
            else self._load_state(group_id)
        mark = ((state.get("police") or {}).get(
            "owner_marks") or {}).get(member)
        if not isinstance(mark, dict):
            return None
        if float(mark.get("at") or 0) < now - INDEX_WINDOW_S:
            return None
        try:
            return max(0, min(5, int(mark.get("stars") or 0)))
        except (TypeError, ValueError):
            return None

    def wanted_stars(self, group_id: str, member: str, *,
                     state: dict[str, Any] | None = None,
                     now: float,
                     focus: float = 1.0
                     ) -> tuple[int, float, list[str]]:
        index, inputs = self.bully_index(
            group_id, member, state=state, now=now, focus=focus)
        stars = 0
        for floor, band in WANTED_BANDS:
            if index >= floor:
                stars = band
                break
        mark = self.owner_mark(group_id, member, state=state,
                               now=now)
        if mark is not None:
            stars = mark
            if mark == 0:
                index = 0.0
            else:
                floor = next((f for f, b in WANTED_BANDS
                              if b == mark), 0.0)
                index = max(index, float(floor))
            inputs = list(inputs) + [
                f"owner's mark — held at {mark}★"]
        return stars, index, inputs

    def note_report_attention(self, group_id: str, *,
                              reports: dict[str, int], now: float
                              ) -> None:
        """≥3 distinct members reporting one target inside
        the window is a police input (§24: small gangs and
        victims can REPORT to draw attention).  The facade
        counts distinct reporters from its audit journal and
        passes the tallies in; the credit lands once a week."""
        state = self._load_state(group_id)
        week = time.strftime("%Y-%W", time.localtime(now))
        credited = dict(state["police"].get("report_credited")
                        or {})
        changed = False
        for target, count in (reports or {}).items():
            if int(count) >= 3 and credited.get(target) != week:
                credited[target] = week
                changed = True
                self._journal(state, "police_reports", now,
                              target=target, reporters=int(count))
        if changed:
            state["police"]["report_credited"] = credited
            self._save_state(group_id, state)

    def police_sweep(self, group_id: str, *, now: float,
                     free_treasury: int,
                     focus: float = 1.0) -> list[dict[str, Any]]:
        """One pass of the ladder over every member the
        journal knows.  Each rung acts ONCE per wanted
        episode; when a member's index decays to zero the
        episode closes and the slate is clean.  Fines and
        raids move on the ledger to the treasury; bounties
        post from the treasury's free balance; jail is the
        surface's (returned as an action)."""
        state = self._load_state(group_id)
        econ = self.economy
        acted = dict(state["police"].get("acted") or {})
        members = set((state.get("rep") or {}).keys())
        for event in state.get("journal") or []:
            if event.get("kind") == "crime":
                members.add(str(event.get("actor") or ""))
        members.discard("")
        actions: list[dict[str, Any]] = []
        remaining_free = max(0, int(free_treasury))
        for member in sorted(members):
            stars, index, inputs = self.wanted_stars(
                group_id, member, state=state, now=now,
                focus=focus)
            record = dict(acted.get(member) or {"rung": 0})
            if index <= 0:
                if record.get("rung"):
                    acted.pop(member, None)
                continue
            if stars <= int(record.get("rung") or 0):
                acted[member] = record
                continue
            rung = WANTED_RUNGS[stars]
            action: dict[str, Any] = {
                "member": member, "stars": stars,
                "index": index, "inputs": inputs[-6:],
                "action": rung["action"]}
            if rung["action"] in ("fine", "crackdown") and int(
                    rung.get("wallet_pct") or 0):
                fine = min(int(econ.balance(group_id, member)
                               * int(rung["wallet_pct"]) / 100),
                           int(rung.get("fine_cap") or 0))
                if fine > 0 and econ._move(
                        group_id, member, "", fine, "fine",
                        f"police fine ({stars}★)", now):
                    action["fine"] = fine
            if rung["action"] in ("raid", "jail", "crackdown") \
                    and int(rung.get("bank_pct") or 0):
                bank = econ.bank_member(member)
                raid = int(econ.balance(group_id, bank)
                           * int(rung["bank_pct"]) / 100)
                if raid > 0 and econ._move(
                        group_id, bank, "", raid, "fine",
                        f"police vault raid ({stars}★)", now):
                    action["raid"] = raid
            if int(rung.get("bounty") or 0):
                amount = min(int(rung["bounty"]), remaining_free)
                if amount > 0:
                    remaining_free -= amount
                    bid = self._post_bounty(
                        state, "", member, amount, now,
                        source="police")
                    action["bounty"] = amount
                    action["bounty_id"] = bid
            if int(rung.get("jail_min") or 0):
                action["jail_min"] = int(rung["jail_min"])
            record["rung"] = stars
            acted[member] = record
            actions.append(action)
            self._journal(state, "police_action", now,
                          member=member, stars=stars,
                          index=index, inputs=inputs[-6:],
                          action=rung["action"])
        state["police"]["acted"] = acted
        self._save_state(group_id, state)
        return actions

    def bribe(self, group_id: str, member: str, *, rng: Any,
              now: float, use_envelope: bool = False
              ) -> dict[str, Any]:
        """A sink with real risk (§24): pay the precinct off
        and some of your journaled heat is forgiven — fail,
        and the attempt itself becomes heat (and sometimes
        cuffs).  An envelope from the arsenal tilts the odds."""
        state = self._load_state(group_id)
        stars, index, _ = self.wanted_stars(
            group_id, member, state=state, now=now)
        if stars <= 0:
            return {"ok": False, "reason": "not_wanted"}
        cost = 40 * stars
        econ = self.economy
        envelope = False
        if use_envelope:
            envelope = self.use_consumable(
                group_id, member, "bribe_envelope", now=now)
            state = self._load_state(group_id)
        if not envelope:
            if econ.balance(group_id, member) < cost:
                return {"ok": False, "reason": "funds",
                        "cost": cost}
            if not econ._move(group_id, member, "", cost, "bribe",
                              "a quiet word with the precinct",
                              now):
                return {"ok": False, "reason": "funds",
                        "cost": cost}
        p = 0.60 - 0.05 * stars + (0.25 if envelope else 0.0)
        success = float(rng.random()) < max(0.05, min(0.95, p))
        police = state["police"]
        if success:
            forgiven = list((police.get("forgiven") or {}).get(
                member) or [])
            forgiven.append({"points": round(index * 0.6, 2),
                             "at": now})
            police.setdefault("forgiven", {})[member] = forgiven
            self._journal(state, "bribe", now, member=member,
                          success=True, stars=stars)
            self._save_state(group_id, state)
            return {"ok": True, "success": True, "cost":
                    0 if envelope else cost, "envelope": envelope}
        heat = list((police.get("extra_heat") or {}).get(
            member) or [])
        heat.append({"points": 3.0, "at": now})
        police.setdefault("extra_heat", {})[member] = heat
        jailed = float(rng.random()) < 0.25
        self._journal(state, "bribe", now, member=member,
                      success=False, stars=stars, jailed=jailed)
        self._save_state(group_id, state)
        return {"ok": True, "success": False,
                "cost": 0 if envelope else cost,
                "envelope": envelope, "jail_min": 15 if jailed
                else 0}

    # ── bounties ──

    def _post_bounty(self, state: dict[str, Any], poster: str,
                     target: str, amount: int, now: float, *,
                     source: str) -> str:
        bid = new_short_id("bty")
        state["police"].setdefault("bounties", {})[bid] = {
            "id": bid, "poster": poster, "target": target,
            "amount": int(amount), "source": source,
            "status": "open", "at": now}
        return bid

    def post_bounty(self, group_id: str, poster: str,
                    target: str, amount: int, *, now: float
                    ) -> dict[str, Any]:
        """A member posts their own sparks on a head: the
        money escrows INTO the treasury (reserved while the
        bounty stands) and pays out whole on a claim."""
        amount = int(amount)
        if amount < 10:
            return {"ok": False, "reason": "amount"}
        if not self.economy._move(
                group_id, poster, "", amount, "bounty",
                "bounty posted", now):
            return {"ok": False, "reason": "funds"}
        state = self._load_state(group_id)
        bid = self._post_bounty(state, poster, target, amount,
                                now, source="member")
        self._journal(state, "bounty_post", now, member=poster,
                      target=target, amount=amount)
        self._save_state(group_id, state)
        return {"ok": True, "id": bid, "amount": amount}

    def open_bounties(self, group_id: str, *,
                      state: dict[str, Any] | None = None
                      ) -> list[dict[str, Any]]:
        state = state if state is not None \
            else self._load_state(group_id)
        return [b for b in (state["police"].get("bounties")
                            or {}).values()
                if b.get("status") == "open"]

    def claim_bounties(self, group_id: str, hunter: str,
                       target: str, *, now: float, via: str
                       ) -> dict[str, Any]:
        """Pay every open bounty on a target to the member
        who defeated them (a geared verb, a citizen hunt, a
        war engagement — the caller vouches for the defeat;
        this organ pays and journals)."""
        state = self._load_state(group_id)
        due = [b for b in self.open_bounties(group_id,
                                             state=state)
               if b.get("target") == target]
        if not due:
            return {"ok": False, "reason": "none"}
        total = 0
        for bounty in due:
            paid = self.economy.treasury_grant(
                group_id, hunter, int(bounty.get("amount") or 0),
                reason=f"bounty claimed ({via})", now=now)
            if paid:
                bounty["status"] = "claimed"
                bounty["claimed_by"] = hunter
                bounty["via"] = via
                total += int(bounty.get("amount") or 0)
        if total:
            gang = self.gang_of(group_id, hunter, state=state)
            if gang is not None and gang.get(
                    "lifestyle") == "vigilantes":
                self.add_rep(state, hunter, 10)
            self._journal(state, "bounty_claim", now,
                          member=hunter, target=target,
                          amount=total, via=via)
            self._save_state(group_id, state)
        return {"ok": bool(total), "amount": total}

    def hunt(self, group_id: str, hunter: str, target: str, *,
             rng: Any, now: float) -> dict[str, Any]:
        """The citizen action (§24): confront a wanted member
        with the precinct's backing — your power plus their
        stars against theirs.  Win: they cool off in jail and
        their bounties pay you.  Lose: a fine, or cuffs if
        you can't pay it."""
        state = self._load_state(group_id)
        stars, index, _ = self.wanted_stars(
            group_id, target, state=state, now=now)
        has_bounty = any(b.get("target") == target
                         for b in self.open_bounties(
                             group_id, state=state))
        if stars <= 0 and not has_bounty:
            return {"ok": False, "reason": "not_wanted"}
        power_h = self.member_power(group_id, hunter,
                                    state=state, now=now)
        power_t = self.member_power(group_id, target,
                                    state=state, now=now)
        odds = max(0.10, min(0.90, 0.45 + (
            power_h + 4 * stars - power_t) / 60.0))
        success = float(rng.random()) < odds
        state["loadouts"][hunter] = _wear(
            self.loadout(group_id, hunter, state=state), "hit",
            now=now)
        self._journal(state, "hunt", now, member=hunter,
                      target=target, success=success, odds=odds)
        self._save_state(group_id, state)
        result: dict[str, Any] = {"ok": True, "success": success,
                                  "odds": odds, "stars": stars}
        if success:
            claimed = self.claim_bounties(
                group_id, hunter, target, now=now, via="hunt")
            result["bounty"] = int(claimed.get("amount") or 0)
            result["jail_min"] = 30
        else:
            result["fine_due"] = 25
        return result

    # ── governance capture (§23/§24) ──

    def power_gang(self, group_id: str, *,
                   state: dict[str, Any] | None = None,
                   now: float) -> dict[str, Any] | None:
        """The gang IN POWER: the strongest gang holding at
        least CAPTURE_DISTRICTS districts — or None."""
        state = state if state is not None \
            else self._load_state(group_id)
        best = None
        best_strength = -1
        for gang in (state.get("gangs") or {}).values():
            held = sum(1 for h in state["territory"].values()
                       if h.get("gang") == gang.get("id"))
            if held < CAPTURE_DISTRICTS:
                continue
            strength = self.gang_strength(
                group_id, gang, state=state, now=now)["total"]
            if strength > best_strength:
                best, best_strength = gang, strength
        return best

    def power_actor(self, group_id: str, member: str, *,
                    now: float) -> dict[str, Any] | None:
        """The gang through which this member holds power
        (leader/officer of the power gang), or None."""
        gang = self.power_gang(group_id, now=now)
        if gang is None:
            return None
        if member == gang.get("leader") or member in (
                gang.get("officers") or []):
            return gang
        return None

    def set_gang_tax(self, group_id: str, actor: str, pct: int,
                     *, now: float) -> dict[str, Any]:
        gang = self.power_actor(group_id, actor, now=now)
        if gang is None:
            return {"ok": False, "reason": "not_power"}
        low, high = CAPTURE_TAX_RANGE
        if not (low <= int(pct) <= high):
            return {"ok": False, "reason": "range",
                    "low": low, "high": high}
        state = self._load_state(group_id)
        live = (state.get("gangs") or {}).get(gang["id"])
        if live is None:
            return {"ok": False, "reason": "gone"}
        live["tax_pct"] = int(pct)
        self._journal(state, "gang_tax", now, member=actor,
                      gang=gang["id"], pct=int(pct))
        self._save_state(group_id, state)
        return {"ok": True, "pct": int(pct), "gang": live}

    # ── system drops (§35) ──

    def roll_drop(self, *, rng: Any, now: float) -> dict[str, Any]:
        """Fresh contents, every time: rarity-weighted kind,
        rolled amount, rolled slots (1–5 by kind)."""
        total = sum(w for _, w in LOOT_KINDS)
        pick = float(rng.random()) * total
        kind = "sparks"
        for name, weight in LOOT_KINDS:
            pick -= weight
            if pick <= 0:
                kind = name
                break
        drop: dict[str, Any] = {"kind": kind, "status": "open",
                                "at": now, "claims": []}
        if kind == "sparks":
            drop["amount"] = int(rng.randint(30, 120))
            drop["slots"] = int(rng.randint(2, 5))
        elif kind == "sparks_big":
            drop["amount"] = int(rng.randint(150, 400))
            drop["slots"] = int(rng.randint(1, 2))
        elif kind == "gems":
            drop["amount"] = int(rng.randint(1, 3))
            drop["slots"] = 1
        elif kind in ("gear", "gear_rare"):
            tiers = (3,) if kind == "gear_rare" else (1, 2)
            pool = [iid for iid, spec in ARSENAL.items()
                    if spec.get("drop") and int(
                        spec.get("tier") or 0) in tiers]
            drop["item"] = str(rng.choice(sorted(pool)))
            drop["slots"] = 1
        drop["per_claim"] = max(
            1, int(drop.get("amount") or 0) // max(
                1, int(drop.get("slots") or 1))) \
            if "amount" in drop else 0
        return drop

    def drops_opened_today(self, group_id: str, *,
                           state: dict[str, Any] | None = None,
                           now: float) -> int:
        state = state if state is not None \
            else self._load_state(group_id)
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        record = state.get("drop_day") or {}
        return int(record.get("opened") or 0) \
            if record.get("day") == day else 0

    def open_drop(self, group_id: str, *, rng: Any, now: float,
                  cfg: dict[str, Any]) -> dict[str, Any] | None:
        """Open one system drop inside the admin's daily
        budget; None when the budget is spent."""
        state = self._load_state(group_id)
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        record = dict(state.get("drop_day") or {})
        if record.get("day") != day:
            record = {"day": day, "opened": 0}
        if int(record.get("opened") or 0) >= int(
                cfg.get("drops_per_day") or 0):
            return None
        drop = self.roll_drop(rng=rng, now=now)
        did = new_short_id("drop")
        drop["id"] = did
        drop["expires"] = now + float(cfg.get("drop_ttl_s")
                                       or 900)
        state["drops"][did] = drop
        record["opened"] = int(record.get("opened") or 0) + 1
        state["drop_day"] = record
        self._journal(state, "sysdrop_open", now, drop=did,
                      loot=drop.get("kind"))
        self._save_state(group_id, state)
        return drop

    def claim_sysdrop(self, group_id: str, member: str,
                      did: str, *, now: float) -> dict[str, Any]:
        """First-come claims (§35): sparks pay as an award
        earn (the issuance throttle sees the mint), gems
        through the estate faucet's ``drop`` source, gear
        straight into the loadout.  A claim the economy
        refuses (earn cap) does NOT burn the slot."""
        state = self._load_state(group_id)
        drop = (state.get("drops") or {}).get(did)
        if drop is None or drop.get("status") != "open":
            return {"ok": False, "reason": "gone"}
        if float(drop.get("expires") or 0) <= now:
            drop["status"] = "expired"
            self._save_state(group_id, state)
            return {"ok": False, "reason": "expired"}
        if member in (drop.get("claims") or []):
            return {"ok": False, "reason": "already"}
        if len(drop.get("claims") or []) >= int(
                drop.get("slots") or 1):
            return {"ok": False, "reason": "full"}
        kind = str(drop.get("kind") or "")
        got: dict[str, Any] = {"kind": kind}
        if kind in ("sparks", "sparks_big"):
            paid = self.economy.award(
                group_id, member, int(drop.get("per_claim") or 0),
                "system drop", kind="earn", ref=f"sysdrop:{did}",
                now=now)
            if paid is None:
                return {"ok": False, "reason": "cap"}
            got["amount"] = int(drop.get("per_claim") or 0)
        elif kind == "gems":
            credited = self._estate.credit_gems(
                group_id, member, int(drop.get("amount") or 0),
                source="drop", reason="system drop",
                ref=f"sysdrop:{did}", now=now)
            if credited is None:
                return {"ok": False, "reason": "gems"}
            got["amount"] = int(drop.get("amount") or 0)
        elif kind in ("gear", "gear_rare"):
            self.give_item(state, member, str(drop.get("item")),
                           now)
            got["item"] = drop.get("item")
        drop["claims"] = list(drop.get("claims") or []) + [member]
        closed = len(drop["claims"]) >= int(drop.get("slots")
                                            or 1)
        if closed:
            drop["status"] = "filled"
        self._journal(state, "sysdrop_claim", now, member=member,
                      drop=did, loot=kind)
        self._save_state(group_id, state)
        return {"ok": True, "got": got, "closed": closed,
                "drop": drop}

    def expire_sysdrops(self, group_id: str, *, now: float
                        ) -> list[dict[str, Any]]:
        """Unclaimed drops expire QUIETLY (§35): state closes,
        no announcement — the caller only audits."""
        state = self._load_state(group_id)
        expired = []
        for drop in (state.get("drops") or {}).values():
            if drop.get("status") == "open" and float(
                    drop.get("expires") or 0) <= now:
                drop["status"] = "expired"
                expired.append(drop)
        if expired:
            self._save_state(group_id, state)
        return expired


    # ── inter-group wars: this organ's side of the book ──

    def escrow_for_inter_war(self, group_id: str, actor: str,
                             wid: str, amount: int, *,
                             now: float) -> dict[str, Any]:
        """Move this gang's war escrow out of its treasury
        into the war reservation (the surface's reservation
        accounting holds it until settlement)."""
        state = self._load_state(group_id)
        gang = self.gang_of(group_id, actor, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        if actor != gang.get("leader") and actor not in (
                gang.get("officers") or []):
            return {"ok": False, "reason": "rank"}
        if int(gang.get("treasury") or 0) < int(amount):
            return {"ok": False, "reason": "funds",
                    "escrow": int(amount)}
        gang["treasury"] = int(gang["treasury"]) - int(amount)
        state["escrow_inter"][wid] = int(amount)
        self._journal(state, "war_inter_escrow", now,
                      member=actor, gang=gang["id"], war=wid,
                      amount=int(amount))
        self._save_state(group_id, state)
        return {"ok": True, "gang": gang}

    def settle_inter_war(self, group_id: str, war: dict[str, Any],
                         *, now: float) -> dict[str, Any] | None:
        """This group's half of an inter-group settlement.
        The winner's own escrow returns to its gang treasury
        and the spoils half arrives as a journaled mint to the
        gang leader (kind ``war_spoils`` — cross-economy value
        is created once, in the open, capped by the escrow);
        the loser's escrow stays out of its gang — paid away.
        Each side settles exactly once (the book tracks it)."""
        wid = str(war.get("id") or "")
        state = self._load_state(group_id)
        escrow = int((state.get("escrow_inter") or {}).pop(
            wid, 0))
        if not escrow:
            return None
        side = "a" if (war.get("a") or {}).get(
            "group") == group_id else "b"
        mine = (war.get(side) or {})
        gang = (state.get("gangs") or {}).get(
            str(mine.get("gang") or ""))
        won = war.get("winner") == side
        if gang is not None and won:
            gang["treasury"] = int(gang.get("treasury") or 0) \
                + escrow
            gang["wins"] = int(gang.get("wins") or 0) + 1
            leader = str(gang.get("leader") or "")
            spoils = int(escrow) // 2
            if leader and spoils > 0:
                # the spoils half is a journaled mint through
                # the treasury to the winning leader — each
                # group's sparks are its own closed economy,
                # so the loser's escrow cannot literally cross;
                # both sides journal the settlement
                reason = (f"war spoils vs "
                          f"{(war.get('b' if side == 'a' else 'a') or {}).get('gang_name', 'the enemy')}")
                if self.economy.treasury_bonus(
                        group_id, spoils, reason=reason, now=now):
                    self.economy.treasury_grant(
                        group_id, leader, spoils, reason=reason,
                        now=now)
        elif gang is not None:
            gang["losses"] = int(gang.get("losses") or 0) + 1
        self._journal(state, "war_inter_settle", now, war=wid,
                      won=won, escrow=escrow)
        self._save_state(group_id, state)
        return {"won": won, "escrow": escrow,
                "gang": gang.get("name") if gang else ""}

    def settle_hq_raid(self, group_id: str, war: dict[str, Any],
                       *, now: float) -> dict[str, Any] | None:
        """This group's half of an HQ raid settlement (§36).

        The raid is an inter-group war (kind ``hq_raid`` in
        the book) whose defender side is a gang's HQ chapter.
        Money discipline matches settle_inter_war exactly —
        each group's sparks are its own closed economy: the
        winner's escrow returns to its gang treasury; the
        attacker's spoils arrive as a journaled mint capped
        by the pool the fortress actually held.  The defender's
        loss is REAL and local: the pool burns out of the HQ
        vault first, then the chapter treasury — a fortress's
        reserves are what a raid is FOR."""
        wid = str(war.get("id") or "")
        state = self._load_state(group_id)
        escrow = int((state.get("escrow_inter") or {}).pop(
            wid, 0))
        if not escrow:
            return None
        side = "a" if (war.get("a") or {}).get(
            "group") == group_id else "b"
        mine = (war.get(side) or {})
        gang = (state.get("gangs") or {}).get(
            str(mine.get("gang") or ""))
        won = war.get("winner") == side
        pool = int(war.get("raid_pool") or 0)
        burned = 0
        if gang is not None and won:
            gang["treasury"] = int(gang.get("treasury") or 0) \
                + escrow
            gang["wins"] = int(gang.get("wins") or 0) + 1
            if side == "a" and pool > 0:
                leader = str(gang.get("leader") or "")
                if leader and self.economy.treasury_bonus(
                        group_id, pool,
                        reason=f"HQ raid spoils vs "
                               f"{(war.get('b') or {}).get('gang_name', 'the fortress')}",
                        now=now):
                    self.economy.treasury_grant(
                        group_id, leader, pool,
                        reason="HQ raid spoils", now=now)
        elif gang is not None:
            gang["losses"] = int(gang.get("losses") or 0) + 1
            if side == "b" and pool > 0:
                vault = int(gang.get("hq_vault") or 0)
                from_vault = min(vault, pool)
                gang["hq_vault"] = vault - from_vault
                rest = pool - from_vault
                held = int(gang.get("treasury") or 0)
                from_treasury = min(rest, held)
                gang["treasury"] = held - from_treasury
                burned = from_vault + from_treasury
        self._journal(state, "hq_raid_settle", now, war=wid,
                      won=won, escrow=escrow, side=side,
                      pool=pool, burned=burned)
        self._save_state(group_id, state)
        return {"won": won, "escrow": escrow, "side": side,
                "pool": pool, "burned": burned,
                "gang": gang.get("name") if gang else ""}


# ── the inter-group war book ─────────────────────────────────────

class WarBook:
    """One global book of inter-group wars + a gang index,
    beside the group store (the reputation book's shape).
    A war record carries SNAPSHOTS (participant powers,
    armory, lifestyle) so rounds resolve from the book alone;
    money never moves through the book — each side's own
    organ escrows and settles in its own economy."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"wars": {}, "gangs": {}}
        try:
            data = json.loads(self.path.read_text(
                encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GroupConflictError(
                f"the war book at {self.path} is unreadable "
                f"({exc}); refusing to read it as empty") from exc
        if not isinstance(data, dict):
            raise GroupConflictError(
                f"the war book at {self.path} is not an object")
        data.setdefault("wars", {})
        data.setdefault("gangs", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1,
                                  sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.path)

    def register_gang(self, group_key: str, store_path: str,
                      gang: dict[str, Any]) -> None:
        data = self._load()
        entry = dict(data["gangs"].get(group_key) or {})
        gangs = dict(entry.get("gangs") or {})
        gangs[str(gang.get("name") or "").lower()] = str(
            gang.get("id") or "")
        entry["gangs"] = gangs
        entry["store"] = str(store_path)
        data["gangs"][group_key] = entry
        self._save(data)

    def unregister_gang(self, group_key: str,
                        gang_id: str) -> None:
        data = self._load()
        entry = dict(data["gangs"].get(group_key) or {})
        gangs = {k: v for k, v in (entry.get("gangs") or {})
                 .items() if v != gang_id}
        entry["gangs"] = gangs
        data["gangs"][group_key] = entry
        self._save(data)

    def find_gangs(self, name: str) -> list[dict[str, Any]]:
        data = self._load()
        wanted = str(name or "").strip().lower()
        out = []
        for group_key, entry in (data.get("gangs") or {}).items():
            gid = (entry.get("gangs") or {}).get(wanted)
            if gid:
                out.append({"group": group_key,
                            "store": entry.get("store"),
                            "gang": gid, "name": name})
        return out

    def create_war(self, a: dict[str, Any], b: dict[str, Any],
                   *, escrow: int, now: float,
                   cfg: dict[str, Any]) -> dict[str, Any]:
        data = self._load()
        wid = new_short_id("xwar")
        war = {"id": wid, "a": a, "b": b, "status": "declared",
               "escrow": int(escrow), "declared_at": now,
               "participants": {"a": {}, "b": {}},
               "score": {"a": 0, "b": 0}, "rounds_done": 0,
               "next_round_at": 0.0,
               "settled": {"a": False, "b": False}, "log": []}
        data["wars"][wid] = war
        self._save(data)
        return war

    def get(self, wid: str) -> dict[str, Any] | None:
        return (self._load().get("wars") or {}).get(wid)

    def update(self, war: dict[str, Any]) -> None:
        data = self._load()
        data["wars"][str(war.get("id"))] = war
        self._save(data)

    def wars_for_group(self, group_key: str
                       ) -> list[dict[str, Any]]:
        return [w for w in (self._load().get("wars") or {})
                .values()
                if (w.get("a") or {}).get("group") == group_key
                or (w.get("b") or {}).get("group") == group_key]


def inter_war_round(war: dict[str, Any], *, rng: Any,
                    now: float, cfg: dict[str, Any]
                    ) -> dict[str, Any]:
    """One inter-group engagement from the book's snapshots:
    opted-in members' powers + the armory snapshot, rolled
    ±15%, best of the configured rounds."""
    def side_power(side: str) -> float:
        record = dict(war.get(side) or {})
        parts = dict((war.get("participants") or {}).get(
            side) or {})
        if parts:
            power = sum(float(p) for p in parts.values())
        else:
            power = float(record.get("leader_power") or 10)
        power += float(record.get("armory_power") or 0)
        life = LIFESTYLES.get(str(record.get("lifestyle")
                                  or "street")) or {}
        power *= float(life.get("war_mult") or 1.0)
        return power * float(rng.uniform(0.85, 1.15))

    power_a = side_power("a")
    power_b = side_power("b")
    strength_a = float((war.get("a") or {}).get("strength") or 0)
    strength_b = float((war.get("b") or {}).get("strength") or 0)
    if strength_a > 0 and strength_b < 0.5 * strength_a:
        power_b *= 1.25
    elif strength_b > 0 and strength_a < 0.5 * strength_b:
        power_a *= 1.25
    winner = "a" if power_a >= power_b else "b"
    score = dict(war.get("score") or {"a": 0, "b": 0})
    score[winner] = int(score.get(winner) or 0) + 1
    war["score"] = score
    war["rounds_done"] = int(war.get("rounds_done") or 0) + 1
    war["log"] = (list(war.get("log") or []) + [{
        "round": war["rounds_done"], "winner": winner,
        "power_a": round(power_a, 1),
        "power_b": round(power_b, 1), "at": now}])[-9:]
    total_rounds = int(cfg.get("war_rounds") or 3)
    needed = total_rounds // 2 + 1
    finished = int(score.get(winner) or 0) >= needed or \
        war["rounds_done"] >= total_rounds
    if finished:
        war["status"] = "done"
        war["winner"] = "a" if int(score.get("a") or 0) > int(
            score.get("b") or 0) else "b"
    else:
        war["next_round_at"] = now + float(
            cfg.get("war_round_s") or 3600)
    return {"war": war, "round": war["rounds_done"],
            "winner_side": winner, "power_a": power_a,
            "power_b": power_b, "finished": finished}
