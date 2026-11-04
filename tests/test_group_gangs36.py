"""Cross-group gangs (DESIGN §36): the global registry,
chapters, the Peacethefirst flagship, headquarters, loyalty,
claim codes, and the owner's police override.

Pins, in the brief's order: the flagship auto-appears in a
freshly claimed group with owner leadership (and never in an
unclaimed room); the capacity table is enforced per tier and
is settings data (refusal payload + success at slots +
uncapped at the top tier, honestly None); HQ designation and
tier purchase move NO mint (the chapter treasury decreases,
the group treasury total does not change), the vault and the
shared armory round-trip, HQ defense reads into a heist
quote, an HQ raid pays and blunts per tier, relocation costs
and drops a tier; the loyalty matrix (allowed shapes, the
cross/local refusals, officer exclusivity, war declaration /
sit-out); the police override (stars adjust, pardon lifts a
jail, focus changes a bully-index outcome, call-off stops an
enforcement) with police-journal entries; claim-code
cross-platform joining binds the right identity; and a
pre-existing local gang registers without losing its
treasury or members.

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore, a stub rng, crimes seeded as the organ's own
journal evidence.
"""
from __future__ import annotations

import time
from pathlib import Path

from nomorals.community.group_conflict import (
    LIFESTYLES,
    GroupConflict,
)
from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupSurface,
)
from nomorals.social.chat.owner_system import OwnerRegistry

KEY = "telegram-bot:-1001234567"      # a claimed bot room
KEY2 = "telegram-bot:-1007654321"     # a second claimed room
KEY3 = "telegram-bot:-1005555555"     # a third
KEY4 = "telegram-bot:-1004444444"     # a fourth
WA = "whatsapp:120363@g.us"           # a WhatsApp group
FOREIGN = "telegram:-100999"          # unclaimed (§33)

OWNER = "999"
OWNER_PERSON = "telegram:999"


class _Rng:
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


def _surface(tmp_path: Path, name: str = "g.json", *,
             owner: bool = False) -> GroupSurface:
    reg = None
    if owner:
        reg = OwnerRegistry(tmp_path / "owner.json").load()
        reg.data["identities"] = {"telegram": [OWNER]}
    return GroupSurface(GroupStore(tmp_path / name).load(),
                        owner_registry=reg)


def _mode(surface: GroupSurface, key: str, mode: str,
          **over) -> None:
    cfg = dict(surface.store.settings(key)["conflict"])
    cfg["mode"] = mode
    cfg.update(over)
    surface.update_settings(key, conflict=cfg)


def _seed(surface: GroupSurface, key: str, uid: str,
          amount: int) -> int:
    surface.handle_message(key, uid, username=f"u{uid}",
                           name=f"User {uid}", text="hello")
    base = time.time()
    remaining, step = amount, 0
    while remaining > 0:
        chunk = min(190, remaining)
        surface.game_award(key, uid, chunk, "ttt",
                           now=base - step * 8 * 86400)
        remaining -= chunk
        step += 1
    return surface.economy.balance(key, uid)


def _gang(surface: GroupSurface, key: str, leader: str,
          members: list[str], *, name: str,
          lifestyle: str = "street", now: float) -> dict:
    out = surface.gang_create(key, leader, name, lifestyle,
                              now=now)
    assert out["ok"], out
    for member in members:
        joined = surface.gang_join(key, member, name, now=now)
        assert joined["ok"], joined
    state = surface.conflict._load_state(key)
    return state["gangs"][out["gang"]["id"]]


def _fund(surface: GroupSurface, key: str, leader: str,
          amount: int, *, now: float) -> None:
    out = surface.conflict.gang_deposit(key, leader, amount,
                                        now=now)
    assert out["ok"], out


def _crimes(surface: GroupSurface, key: str, member: str,
            count: int, *, now: float) -> None:
    """Journal evidence in the organ's own format: farming a
    weak victim (punch-down) — 3 points a hit, +1 a hit once
    a victim repeats."""
    organ = surface.conflict
    state = organ._load_state(key)
    for i in range(count):
        state["journal"].append({
            "kind": "crime", "at": now - i, "actor": member,
            "verb": "rob", "success": True,
            "target": f"victim{i % 2}",
            "actor_total": 1000, "target_total": 10,
            "shielded": False, "heat": 1.0})
    organ._save_state(key, state)


def _journal_kinds(surface: GroupSurface, key: str,
                   kind: str) -> list[dict]:
    state = surface.conflict._load_state(key)
    return [e for e in state.get("journal") or []
            if e.get("kind") == kind]


NOW = time.time()


# ── the flagship ─────────────────────────────────────────────

def test_flagship_seeds_lazily_in_claimed_group_only(tmp_path):
    surface = _surface(tmp_path, owner=True)
    _mode(surface, KEY, "off")  # even with the street asleep
    surface.conflict_cycle(KEY, rng=_Rng(), now=NOW)
    state = surface.conflict._load_state(KEY)
    gangs = list(state["gangs"].values())
    assert len(gangs) == 1
    chapter = gangs[0]
    assert chapter["name"] == "Peacethefirst"
    assert chapter["leader"] == OWNER
    assert chapter.get("flagship") is True
    record = surface.gangs.gang(str(chapter["global"]))
    assert record is not None and record["flagship"] is True
    assert record["roster"][OWNER_PERSON]["role"] == "leader"
    assert surface.gangs.class_of(record) == "cross"
    # an unclaimed room accrues nothing (§33)
    surface.gangs_touch(FOREIGN, now=NOW)
    foreign = surface.conflict._load_state(FOREIGN)
    assert not (foreign.get("gangs") or {})
    assert FOREIGN not in (
        record.get("chapters") or {})


def test_flagship_leadership_cannot_be_granted_away(tmp_path):
    surface = _surface(tmp_path, owner=True)
    _mode(surface, KEY, "warfare")
    surface.gangs_touch(KEY, now=NOW)
    joined = surface.gang_join(KEY, "41", "Peacethefirst",
                               now=NOW)
    assert joined["ok"], joined
    record = surface.gangs.flagship()
    out = surface.gangs.set_role(
        str(record["id"]), by_person=OWNER_PERSON,
        person="telegram:41", role="leader", now=NOW)
    assert out["reason"] == "flagship_leader"
    # and the name itself is held: another gang wearing it
    # can never register over the flagship
    _mode(surface, KEY2, "warfare")
    made = surface.conflict.create_gang(
        KEY2, "77", "Peacethefirst", "street", now=NOW)
    assert made["ok"]
    reg = surface.gangs.register_local_gang(
        KEY2, str(made["gang"]["id"]), now=NOW)
    assert reg["reason"] == "name_registered"


# ── expansion gating ─────────────────────────────────────────

def test_capacity_table_enforced_per_tier(tmp_path):
    surface = _surface(tmp_path)
    for key in (KEY, KEY2, KEY3, KEY4):
        _mode(surface, key, "warfare")
    _gang(surface, KEY, "55", [], name="Night Owls", now=NOW)
    record = surface.gangs.gang_by_name("Night Owls")
    gid = str(record["id"])
    # crew tier: home chapter only — the refusal names the
    # tier and the slots, plainly
    out = surface.gang_found_chapter(KEY2, "55", "Night Owls",
                                     now=NOW)
    assert out["reason"] == "capacity"
    assert out["tier"] == "crew" and out["slots"] == 1 \
        and out["used"] == 1
    # five worldwide: gang tier — three slots
    for member in ("56", "57", "58", "59"):
        assert surface.gang_join(KEY, member, "Night Owls",
                                 now=NOW)["ok"]
    assert surface.gangs.tier_name(
        surface.gangs.gang(gid)) == "gang"
    assert surface.gang_found_chapter(
        KEY2, "55", "Night Owls", now=NOW)["ok"]
    assert surface.gang_found_chapter(
        KEY3, "55", "Night Owls", now=NOW)["ok"]
    out = surface.gang_found_chapter(KEY4, "55", "Night Owls",
                                     now=NOW)
    assert out["reason"] == "capacity" and out["slots"] == 3
    # fifteen worldwide: syndicate — uncapped, honestly None
    for i in range(10):
        assert surface.gang_join(KEY, f"6{i}", "Night Owls",
                                 now=NOW)["ok"]
    record = surface.gangs.gang(gid)
    assert surface.gangs.tier_name(record) == "syndicate"
    assert surface.gangs.capacity(record) is None
    view = surface.gangs.global_view(record)
    assert view["capacity_text"] == "uncapped"
    assert surface.gang_found_chapter(
        KEY4, "55", "Night Owls", now=NOW)["ok"]


def test_capacity_table_is_settings_data(tmp_path):
    surface = _surface(tmp_path)
    for key in (KEY, KEY2, KEY3):
        _mode(surface, key, "warfare")
    _gang(surface, KEY, "55", ["56", "57", "58", "59"],
          name="Iron Fen", now=NOW)
    assert surface.gang_found_chapter(
        KEY2, "55", "Iron Fen", now=NOW)["ok"]
    # KEY3's own settings narrow the gang tier to 2 slots —
    # the table is data, read where the action happens
    block = dict(surface.store.settings(KEY3)["gangs"])
    block["chapter_slots"] = {"crew": 1, "gang": 2,
                              "syndicate": None}
    surface.update_settings(KEY3, gangs=block)
    out = surface.gang_found_chapter(KEY3, "55", "Iron Fen",
                                     now=NOW)
    assert out["reason"] == "capacity" and out["slots"] == 2


# ── headquarters ─────────────────────────────────────────────

def _hq_gang(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare")
    _mode(surface, KEY2, "warfare")
    _seed(surface, KEY, "55", 9000)
    gang = _gang(surface, KEY, "55", ["56", "57", "58", "59"],
                 name="Harbour Kings", lifestyle="mafia",
                 now=NOW)
    _fund(surface, KEY, "55", 8000, now=NOW)
    return surface, gang


def test_hq_designate_buy_vault_no_mint(tmp_path):
    surface, gang = _hq_gang(tmp_path)
    organ = surface.conflict
    assert surface.gang_hq_designate(KEY, "55", now=NOW)["ok"]
    treasury_before = surface.economy.treasury_balance(KEY)
    reserved_before = organ.reservations(KEY)
    out = surface.gang_hq_buy(KEY, "55", now=NOW)
    assert out["ok"] and out["tier_name"] == "hideout" \
        and out["cost"] == 500
    state = organ._load_state(KEY)
    live = state["gangs"][gang["id"]]
    assert live["treasury"] == 7500
    # no mint: the group's treasury total is untouched; the
    # sparks simply stop being reserved to the gang
    assert surface.economy.treasury_balance(KEY) \
        == treasury_before
    assert organ.reservations(KEY) == reserved_before - 500
    assert surface.gang_hq_buy(KEY, "55", now=NOW)["tier_name"] \
        == "compound"
    assert surface.gang_hq_buy(KEY, "55", now=NOW)["tier_name"] \
        == "fortress"
    out = surface.gang_hq_buy(KEY, "55", now=NOW)
    assert out["reason"] == "max_tier"
    # strength reads the fortress where it stands
    state = organ._load_state(KEY)
    live = state["gangs"][gang["id"]]
    strength = organ.gang_strength(KEY, live, state=state,
                                   now=NOW)
    assert strength["hq"] == 26
    # the vault: a partition, counted in reservations either
    # way — the total reserved does not move on deposit
    reserved = organ.reservations(KEY)
    out = surface.gang_hq_vault(KEY, "55", "deposit", 300,
                                now=NOW)
    assert out["ok"] and out["vault"] == 300
    assert organ.reservations(KEY) == reserved
    state = organ._load_state(KEY)
    live = state["gangs"][gang["id"]]
    assert live["treasury"] == 1700 \
        and live["hq_vault"] == 300
    # members may feed it; spending it is governance
    out = surface.gang_hq_vault(KEY, "56", "withdraw", 100,
                                now=NOW)
    assert out["reason"] == "rank"
    out = surface.gang_hq_vault(KEY, "55", "withdraw", 100,
                                now=NOW)
    assert out["ok"] and out["vault"] == 200
    # and the vault is worked standing at the fortress
    assert surface.gang_found_chapter(
        KEY2, "55", "Harbour Kings", now=NOW)["ok"]
    out = surface.gang_hq_vault(KEY2, "55", "deposit", 10,
                                now=NOW)
    assert out["reason"] == "hq_elsewhere"


def test_hq_defense_reads_into_heist_quote(tmp_path):
    surface, gang = _hq_gang(tmp_path)
    assert surface.gang_hq_designate(KEY, "55", now=NOW)["ok"]
    for _ in range(3):
        assert surface.gang_hq_buy(KEY, "55", now=NOW)["ok"]
    organ = surface.conflict
    started = organ.start_heist(KEY, "66", "treasury", "",
                                now=NOW)
    assert started["ok"]
    hid = started["heist"]["id"]
    assert organ.join_heist(KEY, "77", hid, now=NOW)["ok"]
    quote = organ.heist_preview(KEY, hid, now=NOW)
    assert quote["hq_defense"] == 26
    # the same crew, the same target, in a group where the
    # fortress does not stand: exactly the HQ bonus less
    _mode(surface, KEY3, "warfare")
    started = organ.start_heist(KEY3, "66", "treasury", "",
                                now=NOW)
    hid = started["heist"]["id"]
    assert organ.join_heist(KEY3, "77", hid, now=NOW)["ok"]
    plain = organ.heist_preview(KEY3, hid, now=NOW)
    assert plain["hq_defense"] == 0
    assert quote["defense"] - plain["defense"] == 26


def test_hq_armory_round_trip_across_chapters(tmp_path):
    surface, gang = _hq_gang(tmp_path)
    assert surface.gang_hq_designate(KEY, "55", now=NOW)["ok"]
    assert surface.gang_found_chapter(
        KEY2, "55", "Harbour Kings", now=NOW)["ok"]
    # a chapter seat is local: 56 takes theirs in KEY2, then
    # shelves a piece from there into the shared armory
    assert surface.gang_join(KEY2, "56", "Harbour Kings",
                             now=NOW)["ok"]
    organ = surface.conflict
    state = organ._load_state(KEY2)
    organ.give_item(state, "56", "getaway_van", NOW)
    organ._save_state(KEY2, state)
    out = surface.gang_hq_armory(KEY2, "56", "store",
                                 "getaway_van", now=NOW)
    assert out["ok"], out
    assert organ.loadout(KEY2, "56") == []
    # the leader, standing at HQ, draws it out
    out = surface.gang_hq_armory(KEY, "55", "draw",
                                 "getaway_van", now=NOW)
    assert out["ok"], out
    assert [i["item"] for i in organ.loadout(KEY, "55")] \
        == ["getaway_van"]
    listed = surface.gang_hq_armory(KEY, "55", "list",
                                    now=NOW)
    assert listed["ok"] and listed["armory"] == []
    out = surface.gang_hq_armory(KEY, "55", "draw",
                                 "getaway_van", now=NOW)
    assert out["reason"] == "not_stocked"


def test_hq_relocation_costs_and_drops_a_tier(tmp_path):
    surface, gang = _hq_gang(tmp_path)
    assert surface.gang_hq_designate(KEY, "55", now=NOW)["ok"]
    assert surface.gang_hq_buy(KEY, "55", now=NOW)["ok"]
    assert surface.gang_hq_buy(KEY, "55", now=NOW)["ok"]
    assert surface.gang_hq_vault(KEY, "55", "deposit", 200,
                                 now=NOW)["ok"]
    assert surface.gang_found_chapter(
        KEY2, "55", "Harbour Kings", now=NOW)["ok"]
    _seed(surface, KEY2, "55", 2000)
    _fund(surface, KEY2, "55", 1000, now=NOW)
    organ = surface.conflict
    before = organ._load_state(KEY)["gangs"][gang["id"]]
    assert before["treasury"] == 5800  # 8000 −2000 tiers −200
    out = surface.gang_hq_move(KEY2, "55", now=NOW)
    assert out["ok"], out
    assert out["cost"] == 750 and out["tier_name"] == "hideout"
    assert out["released"] == 200
    after = organ._load_state(KEY)["gangs"][gang["id"]]
    assert "hq" not in after
    assert after["treasury"] == 6000  # the vault released home
    assert after["hq_vault"] == 0
    status = surface.gangs.hq_status(str(gang["global"]))
    assert status["hq"]["group"] == KEY2
    assert status["hq"]["tier_name"] == "hideout"
    key2 = organ._load_state(KEY2)
    chapter = [g for g in key2["gangs"].values()
               if g.get("global") == gang["global"]][0]
    assert chapter["treasury"] == 250


# ── the HQ raid ──────────────────────────────────────────────

def test_hq_raid_pays_and_blunts_per_tier(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare", war_round_s=60)
    _mode(surface, KEY2, "warfare", war_round_s=60)
    _seed(surface, KEY, "55", 5000)
    _gang(surface, KEY, "55", ["56", "57", "58", "59"],
          name="Sea Wolves", now=NOW)
    _fund(surface, KEY, "55", 2000, now=NOW)
    _seed(surface, KEY2, "88", 5000)
    court = _gang(surface, KEY2, "88", ["89", "90", "91", "92"],
                  name="Stone Court", now=NOW)
    _fund(surface, KEY2, "88", 3000, now=NOW)
    assert surface.gang_hq_designate(KEY2, "88", now=NOW)["ok"]
    assert surface.gang_hq_buy(KEY2, "88", now=NOW)["ok"]
    assert surface.gang_hq_vault(KEY2, "88", "deposit", 1000,
                                 now=NOW)["ok"]
    out = surface.war_raid_hq(KEY, "55", "Stone Court",
                              now=NOW)
    assert out["ok"], out
    war = out["war"]
    assert war["kind"] == "hq_raid"
    assert war["escrow"] == 400          # double war escrow
    assert out["pool"] == 850            # 1000 × (1 − 15%)
    assert war["b"]["hq_tier"] == 1
    # the fortress blunts the snapshot: +10% at hideout
    base = surface.conflict.gang_strength(
        KEY2, surface.conflict._load_state(KEY2)["gangs"]
        [court["id"]], now=NOW)["total"]
    assert war["b"]["strength"] == int(base * 1.1)
    assert surface.war_answer_inter(KEY2, "88", True,
                                    now=NOW)["ok"]
    assert surface.war_join_inter(KEY, "55", now=NOW)["ok"]
    assert surface.war_join_inter(KEY2, "88", now=NOW)["ok"]
    war = surface.war_book.wars_for_group(KEY)[0]
    war["participants"] = {"a": {"55": 5000}, "b": {"88": 1}}
    surface.war_book.update(war)
    wolves = surface.gangs.gang_by_name("Sea Wolves")
    prestige_before = int(wolves["prestige"])
    leader_before = surface.economy.balance(KEY, "55")
    now = NOW
    for _ in range(6):
        now += 1000
        surface.conflict_cycle(KEY, rng=_Rng(), now=now)
        war = surface.war_book.wars_for_group(KEY)[0]
        if war.get("status") == "done":
            break
    assert war["status"] == "done" and war["winner"] == "a"
    surface.conflict_cycle(KEY2, rng=_Rng(), now=now + 1000)
    # the vault burned by exactly the blunted pool
    state = surface.conflict._load_state(KEY2)
    live = state["gangs"][court["id"]]
    assert live["hq_vault"] == 150
    # the attacker's leader collects the pool as spoils
    assert surface.economy.balance(KEY, "55") \
        == leader_before + 850
    wolves = surface.gangs.gang_by_name("Sea Wolves")
    assert int(wolves["prestige"]) == prestige_before + 40
    state = surface.conflict._load_state(KEY)
    mine = [g for g in state["gangs"].values()
            if g["name"] == "Sea Wolves"][0]
    assert mine["wins"] == 1


# ── loyalty ──────────────────────────────────────────────────

def test_loyalty_matrix(tmp_path):
    surface = _surface(tmp_path, owner=True)
    for key in (KEY, KEY2, KEY3, KEY4):
        _mode(surface, key, "warfare")
    surface.gangs_touch(KEY, now=NOW)
    # 41 rides the flagship (cross slot) + a local crew —
    # the one allowed dual shape
    assert surface.gang_join(KEY, "41", "Peacethefirst",
                             now=NOW)["ok"]
    _gang(surface, KEY2, "42", [], name="Dock Rats", now=NOW)
    assert surface.gang_join(KEY2, "41", "Dock Rats",
                             now=NOW)["ok"]
    # a second cross-group gang refuses, naming the holder
    _gang(surface, KEY3, "43", ["a1", "a2", "a3", "a4"],
          name="Iron Fen", now=NOW)
    out = surface.gang_join(KEY3, "41", "Iron Fen", now=NOW)
    assert out["reason"] == "cross_taken"
    assert out["holder"] == "Peacethefirst"
    # a second local gang refuses too
    _gang(surface, KEY4, "44", [], name="Alley Cats", now=NOW)
    out = surface.gang_join(KEY4, "41", "Alley Cats", now=NOW)
    assert out["reason"] == "local_taken"
    assert out["holder"] == "Dock Rats"
    view = surface.gangs.person_view("telegram:41")
    slots = {m["gang"]: m["slot"] for m in view["memberships"]}
    assert slots == {"Peacethefirst": "cross",
                     "Dock Rats": "local"}


def test_officer_exclusivity_reads_chapters_live(tmp_path):
    surface = _surface(tmp_path, owner=True)
    _mode(surface, KEY, "warfare")
    _mode(surface, KEY2, "warfare")
    surface.gangs_touch(KEY, now=NOW)
    _gang(surface, KEY2, "42", ["45"], name="Dock Rats",
          now=NOW)
    # 45 is a chapter officer of Dock Rats (set in the
    # chapter, the way the street does it) and a flagship
    # member — promoting them in the flagship must refuse:
    # officer in one, never both.
    out = surface.conflict.gang_set_role(
        KEY2, "42", "45", "officer", now=NOW)
    assert out["ok"]
    assert surface.gang_join(KEY, "45", "Peacethefirst",
                             now=NOW)["ok"]
    record = surface.gangs.flagship()
    out = surface.gangs.set_role(
        str(record["id"]), by_person=OWNER_PERSON,
        person="telegram:45", role="officer", now=NOW)
    assert out["reason"] == "officer_elsewhere"
    assert out["holder"] == "Dock Rats"


def test_war_declaration_and_sit_out(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY2, "warfare")
    _mode(surface, KEY3, "warfare")
    _seed(surface, KEY3, "43", 3000)
    _gang(surface, KEY3, "43", ["a1", "a2", "a3", "a4"],
          name="Iron Fen", now=NOW)
    _fund(surface, KEY3, "43", 1000, now=NOW)
    _seed(surface, KEY2, "42", 3000)
    _gang(surface, KEY2, "42", ["46"], name="Dock Rats",
          now=NOW)
    _fund(surface, KEY2, "42", 1000, now=NOW)
    # 46 holds Iron Fen (cross) + Dock Rats (local, a crew
    # when joined)...
    assert surface.gang_join(KEY3, "46", "Iron Fen",
                             now=NOW)["ok"]
    # ...then Dock Rats grows into gang tier: the membership
    # stands (grandfathered), and the war rules take over
    for member in ("47", "48", "49"):
        assert surface.gang_join(KEY2, member, "Dock Rats",
                                 now=NOW)["ok"]
    out = surface.war_declare_inter(KEY3, "43", "Dock Rats",
                                    now=NOW)
    assert out["ok"], out
    assert surface.war_answer_inter(KEY2, "42", True,
                                    now=NOW)["ok"]
    # undeclared, the dual holder sits the war out
    out = surface.war_join_inter(KEY3, "46", now=NOW)
    assert out["reason"] == "undeclared"
    # declare for Dock Rats — then Iron Fen gets nothing
    out = surface.gang_war_side(KEY3, "46", "Dock Rats",
                                now=NOW)
    assert out["ok"], out
    out = surface.war_join_inter(KEY3, "46", now=NOW)
    assert out["reason"] == "declared_other"
    out = surface.war_join_inter(KEY2, "46", now=NOW)
    assert out["ok"], out


# ── chapter perks (the benefits loop) ────────────────────────

def test_chapter_pays_the_gangs_lifestyle_perks(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare")
    _mode(surface, KEY2, "warfare")
    _gang(surface, KEY, "55", ["56", "57", "58", "59"],
          name="Cartel Norte", lifestyle="cartel", now=NOW)
    assert surface.gang_found_chapter(
        KEY2, "55", "Cartel Norte", now=NOW)["ok"]
    assert surface.gang_join(KEY2, "56", "Cartel Norte",
                             now=NOW)["ok"]
    # the chapter IS a cartel gang locally: its job pays from
    # the cartel table, in the chapter's own group
    labels = {job[0] for job in LIFESTYLES["cartel"]["jobs"]}
    out = surface.conflict.gang_job(KEY2, "56", rng=_Rng(),
                                    now=NOW)
    assert out["ok"] and out["job"] in labels


# ── the owner's police override ──────────────────────────────

def test_police_override_stars_pardon_call_off(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare")
    organ = surface.conflict
    _crimes(surface, KEY, "66", 14, now=NOW)   # ≈46 → 5★
    view = surface.police_wanted_view(KEY, "66", now=NOW)
    assert view["stars"] == 5
    # set stars: the mark holds the ladder where the owner
    # puts it, and the police journal carries the verb
    out = surface.police_owner_set_stars(
        KEY, "66", 1, by_person=OWNER_PERSON, now=NOW)
    assert out["ok"] and out["stars"] == 1
    entries = [e for e in _journal_kinds(
        surface, KEY, "police_override")
        if e.get("verb") == "set_stars"]
    assert entries and entries[-1]["stars"] == 1
    # pardon: slate forgiven, jail lifts as a release,
    # police bounties cancelled
    _crimes(surface, KEY, "70", 14, now=NOW)
    surface.jail_member(KEY, "70", 60, reason="test",
                        now=NOW)
    assert surface.jail_state(KEY, "70", now=NOW) is not None
    actions = organ.police_sweep(KEY, now=NOW,
                                 free_treasury=10000)
    assert any(a["member"] == "70" for a in actions)
    assert any(b.get("target") == "70"
               and b.get("source") == "police"
               for b in organ.open_bounties(KEY))
    out = surface.police_owner_pardon(
        KEY, "70", by_person=OWNER_PERSON, now=NOW)
    assert out["ok"] and out["released"] is True
    assert out["bounties_cancelled"] >= 1
    assert surface.jail_state(KEY, "70", now=NOW) is None
    view = surface.police_wanted_view(KEY, "70", now=NOW)
    assert view["stars"] == 0
    assert not any(b.get("target") == "70"
                   and b.get("source") == "police"
                   for b in organ.open_bounties(KEY))
    entries = [e for e in _journal_kinds(
        surface, KEY, "police_override")
        if e.get("verb") == "pardon"]
    assert entries
    # call-off: the enforcement stands down at its rung —
    # the next sweep does not re-fire it, bounties cancel,
    # but the heat itself stays on the books
    _crimes(surface, KEY, "71", 7, now=NOW)    # ≈26 → 3★
    actions = organ.police_sweep(KEY, now=NOW,
                                 free_treasury=10000)
    assert any(a["member"] == "71" for a in actions)
    out = surface.police_owner_call_off(
        KEY, "71", by_person=OWNER_PERSON, now=NOW)
    assert out["ok"] and out["bounties_cancelled"] >= 1
    actions = organ.police_sweep(KEY, now=NOW + 10,
                                 free_treasury=10000)
    assert not any(a["member"] == "71" for a in actions)
    view = surface.police_wanted_view(KEY, "71", now=NOW + 10)
    assert view["stars"] == 3  # heat kept — stood down, not
    # forgiven
    entries = [e for e in _journal_kinds(
        surface, KEY, "police_override")
        if e.get("verb") == "call_off"]
    assert entries


def test_police_focus_changes_the_sweep_outcome(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare")
    organ = surface.conflict
    _crimes(surface, KEY, "68", 4, now=NOW)    # index 14 → 1★
    _crimes(surface, KEY, "69", 4, now=NOW)
    actions = organ.police_sweep(KEY, now=NOW,
                                 free_treasury=10000, focus=1.0)
    mine = {a["member"]: a for a in actions}
    assert mine["68"]["stars"] == 1
    assert "jail_min" not in mine["68"]
    # the owner's focus triples the pressure: the same
    # evidence reads 4★ and the ladder jails
    out = surface.police_owner_focus(
        KEY, 3.0, by_person=OWNER_PERSON, now=NOW)
    assert out["ok"] and out["aggression"] == 3.0
    assert surface.gangs.police_focus(KEY) == 3.0
    actions = organ.police_sweep(
        KEY, now=NOW, free_treasury=10000,
        focus=surface.gangs.police_focus(KEY))
    mine = {a["member"]: a for a in actions}
    assert mine["69"]["stars"] == 4
    assert mine["69"]["jail_min"] == 60


# ── claim codes ──────────────────────────────────────────────

def test_claim_code_binds_identity_across_platforms(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare")
    _mode(surface, WA, "warfare")
    _gang(surface, KEY, "55", ["56", "57", "58", "59"],
          name="Night Owls", now=NOW)
    # the leader binds their WhatsApp identity first — no
    # chapter there yet, so the seat waits but the BINDING
    # lands (and person resolution follows it)
    code = surface.gang_claim_code(KEY, "55", now=NOW)
    assert code["ok"]
    out = surface.gang_redeem_code(WA, "2348001", code["code"],
                                   now=NOW)
    assert out["reason"] == "no_chapter" and out["bound"]
    assert surface.gangs.person_for(WA, "2348001") \
        == "telegram:55"
    # now the leader can found the WhatsApp chapter — the
    # registry knows who is standing there
    out = surface.gang_found_chapter(WA, "2348001",
                                     "Night Owls", now=NOW)
    assert out["ok"], out
    # a member's code seats their other-platform identity
    code = surface.gang_claim_code(KEY, "56", now=NOW)
    out = surface.gang_redeem_code(WA, "2348002", code["code"],
                                   now=NOW)
    assert out["ok"] and out["joined"]
    chapter = surface.conflict.gang_of(WA, "2348002")
    assert chapter is not None \
        and chapter["name"] == "Night Owls"
    record = surface.gangs.gang_by_name("Night Owls")
    entry = record["roster"]["telegram:56"]
    assert entry["platforms"]["whatsapp"] == "2348002"
    # codes are one-shot
    out = surface.gang_redeem_code(WA, "2348003", code["code"],
                                   now=NOW)
    assert out["reason"] == "bad_code"


# ── registration of a pre-existing gang ──────────────────────

def test_preexisting_gang_registers_with_history_intact(
        tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare")
    organ = surface.conflict
    _seed(surface, KEY, "55", 2000)
    # built entirely through the old organ, as a live gang
    # would have been before §36 existed
    made = organ.create_gang(KEY, "55", "Old Hands", "street",
                             now=NOW)
    assert made["ok"]
    assert organ.join_gang(KEY, "56", "Old Hands",
                           now=NOW)["ok"]
    assert organ.join_gang(KEY, "57", "Old Hands",
                           now=NOW)["ok"]
    assert organ.gang_deposit(KEY, "55", 700, now=NOW)["ok"]
    assert "global" not in made["gang"]
    out = surface.gang_register_here(KEY, "55", now=NOW)
    assert out["ok"], out
    record = out["gang"]
    assert len(record["roster"]) == 3
    assert record["chapters"].keys() == {KEY}
    state = organ._load_state(KEY)
    live = state["gangs"][made["gang"]["id"]]
    assert live["treasury"] == 700
    assert live["members"] == ["55", "56", "57"]
    assert live["global"] == record["id"]
    card = surface.gang_card(KEY, "55", now=NOW)
    assert card["global"]["tier"] == "crew"
    assert card["global"]["chapter_count"] == 1


# ── the gang board (§14) ─────────────────────────────────────

def test_gangs_board_ranks_by_prestige(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "warfare")
    _gang(surface, KEY, "55", [], name="Alpha Crew", now=NOW)
    _gang(surface, KEY, "66", [], name="Beta Crew", now=NOW)
    alpha = surface.gangs.gang_by_name("Alpha Crew")
    beta = surface.gangs.gang_by_name("Beta Crew")
    surface.gangs.note_inter_war_win("war1", str(alpha["id"]),
                                     now=NOW)
    surface.gangs.note_inter_war_win("war2", str(alpha["id"]),
                                     now=NOW)
    surface.gangs.note_inter_war_win("war3", str(beta["id"]),
                                     now=NOW)
    rows = surface.board_rows("gangs:prestige")
    assert [r["label"] for r in rows] == ["Alpha Crew",
                                          "Beta Crew"]
    assert rows[0]["value"] == 50 and rows[1]["value"] == 25
    text = surface.render_board("gangs:prestige",
                                group_key=KEY)
    assert "gang prestige" in text and "Alpha Crew" in text
    assert surface.board_rows("gangs:bogus") is None
