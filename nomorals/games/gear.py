"""Arena gear: durable equipment with types, grades, sets, and persistence.

Shop-bought gear used to be one-shot consumables keyed by count
(``gear_sword`` etc.) — and the arena couldn't even find them because it
looked up ``sword``/``armor`` instead of the real slugs.  This module
replaces that with real equipment:

* **Types**: katana, broadsword, rapier, warhammer (weapons); leather,
  chainmail, plate, dragonscale (armor).
* **Grades**: common ×1.0, rare ×1.25, epic ×1.5, legendary ×2.0 — they
  multiply base attack/defense and scale durability and price.
* **Durability**: every piece wears with use and breaks at 0.  Broken
  gear stays in your inventory and can be repaired for coins — it never
  silently disappears.
* **Sets**: matching weapon + armor of one set (storm, shadow) grant a
  set bonus — extra stats plus a combo attack in the arena.
* **Persistence**: every piece is a row in ``game_gear`` keyed by player,
  so gear survives restarts.  Legacy ``gear_sword``/``gear_armor``
  counts are migrated into real pieces on first access.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

__all__ = [
    "GRADES", "GRADE_MULT", "GRADE_DURABILITY", "SLOTS",
    "GearDef", "GearInstance", "GEAR_CATALOG", "SET_BONUSES",
    "LEGACY_GEAR_MAP", "GearStore", "RAID_EXCLUSIVE_GEAR",
    "effective_stats", "detect_set_bonus", "durability_bar",
    "durability_display",
    # Diablo-style affix loot
    "Affix", "PREFIXES", "SUFFIXES", "LootItem",
    "forge_loot", "describe_loot", "grant_loot",
    "instance_stats", "instance_describe",
]

_log = get_logger(__name__)

#: Grade → stat multiplier.
GRADE_MULT: dict[str, float] = {
    "common": 1.0,
    "rare": 1.25,
    "epic": 1.5,
    "legendary": 2.0,
    "myth": 3.0,
}
GRADES = tuple(GRADE_MULT)

#: Grade → max durability (hits before the piece breaks).
GRADE_DURABILITY: dict[str, int] = {
    "common": 25,
    "rare": 40,
    "epic": 60,
    "legendary": 100,
    "myth": 999,
}

#: Equipment slots. One piece per slot equipped at a time.
SLOTS = ("weapon", "armor", "trinket")

#: Grade → price multiplier over the kind's base cost.
_GRADE_COST_MULT: dict[str, float] = {
    "common": 1.0, "rare": 2.0, "epic": 4.0, "legendary": 7.0,
    "myth": 12.0,
}

#: Old shop slugs → the gear they become on migration.
LEGACY_GEAR_MAP: dict[str, str] = {
    "gear_sword": "broadsword_common",
    "gear_armor": "chainmail_common",
}


@dataclass(frozen=True)
class GearDef:
    """The blueprint for one piece of gear sold in the shop."""

    slug: str
    name: str
    cost: int
    slot: str            # weapon | armor | trinket
    kind: str            # katana | broadsword | ... (flavor + base stats)
    grade: str           # common | rare | epic | legendary | myth
    set_name: str = ""   # "" = no set; else e.g. "storm"
    base_atk: int = 0
    base_def: int = 0
    unbreakable: bool = False  # never wears, never breaks (legacy pieces)
    raid_only: bool = False  # never sold — drops from raid bosses only
    # ── RPG attribute bonuses: folded into the fighter at battle setup ──
    stat_strength: int = 0
    stat_stamina: int = 0
    stat_mana: int = 0
    stat_intelligence: int = 0

    @property
    def max_durability(self) -> int:
        return GRADE_DURABILITY[self.grade]

    def to_dict(self) -> dict[str, Any]:
        atk, df = effective_stats(self)
        return {"slug": self.slug, "name": self.name, "cost": self.cost,
                "slot": self.slot, "kind": self.kind, "grade": self.grade,
                "set": self.set_name, "atk": atk, "def": df,
                "durability": self.max_durability,
                "raid_only": self.raid_only,
                "stat_strength": self.stat_strength,
                "stat_stamina": self.stat_stamina,
                "stat_mana": self.stat_mana,
                "stat_intelligence": self.stat_intelligence}


def effective_stats(defn: GearDef) -> tuple[int, int]:
    """(attack, defense) after the grade multiplier."""
    mult = GRADE_MULT[defn.grade]
    return (int(round(defn.base_atk * mult)),
            int(round(defn.base_def * mult)))


def _build_catalog() -> dict[str, GearDef]:
    defn: dict[str, GearDef] = {}

    def add(slug: str, name: str, base_cost: int, slot: str, kind: str,
            grade: str, base_atk: int = 0, base_def: int = 0,
            set_name: str = "", unbreakable: bool = False,
            raid_only: bool = False, stat_strength: int = 0,
            stat_stamina: int = 0, stat_mana: int = 0,
            stat_intelligence: int = 0) -> None:
        cost = int(base_cost * _GRADE_COST_MULT[grade])
        defn[slug] = GearDef(slug=slug, name=f"{name} [{grade}]",
                             cost=cost, slot=slot, kind=kind, grade=grade,
                             set_name=set_name, base_atk=base_atk,
                             base_def=base_def, unbreakable=unbreakable,
                             raid_only=raid_only,
                             stat_strength=stat_strength,
                             stat_stamina=stat_stamina,
                             stat_mana=stat_mana,
                             stat_intelligence=stat_intelligence)

    weapons = (("katana", "Katana", 300, 12),
               ("broadsword", "Broadsword", 350, 15),
               ("rapier", "Rapier", 200, 8),
               ("warhammer", "Warhammer", 450, 20),
               # ── new blood: heavier steel for higher ranks ──
               ("odachi", "Odachi", 550, 24),
               ("spear", "Spear", 400, 17),
               ("twin_daggers", "Twin Daggers", 380, 16),
               ("battle_axe", "Battle Axe", 500, 22))
    # myth is reserved for the named legacy set — ordinary kinds stop at
    # legendary.
    shop_grades = ("common", "rare", "epic", "legendary")
    for kind, label, base_cost, base_atk in weapons:
        for grade in shop_grades:
            add(f"{kind}_{grade}", label, base_cost, "weapon", kind,
                grade, base_atk=base_atk)

    armors = (("leather", "Leather Armor", 150, 8),
              ("chainmail", "Chainmail", 300, 15),
              ("plate", "Plate Armor", 450, 22),
              ("dragonscale", "Dragonscale Mail", 600, 30),
              # ── new blood ──
              ("samurai", "Samurai Armor", 520, 26),
              ("mithril", "Mithril Mail", 700, 34))
    for kind, label, base_cost, base_def in armors:
        for grade in shop_grades:
            add(f"{kind}_{grade}", label, base_cost, "armor", kind,
                grade, base_def=base_def)

    # ── trinkets: the third slot, finally stocked ──
    # Trinkets grant RPG attribute bonuses instead of raw stats —
    # the warlord's belt makes you hit harder, the sage's amulet
    # sharpens your techniques.
    trinkets = (
        # kind, label, base_cost, stat bonuses
        ("warlord_belt", "Warlord's Belt", 400,
         {"stat_strength": 4}),
        ("titan_heart", "Titan's Heart", 400,
         {"stat_stamina": 4}),
        ("mana_crystal", "Mana Crystal", 400,
         {"stat_mana": 4}),
        ("sage_amulet", "Sage's Amulet", 450,
         {"stat_intelligence": 4}),
    )
    trinket_grades = ("rare", "epic", "legendary")
    _trinket_scale = {"rare": 1, "epic": 2, "legendary": 3}
    for kind, label, base_cost, bonuses in trinkets:
        for grade in trinket_grades:
            scale = _trinket_scale[grade]
            scaled = {k: v * scale for k, v in bonuses.items()}
            add(f"{kind}_{grade}", label, base_cost, "trinket", kind,
                grade, **scaled)

    # Named sets — matching weapon + armor unlock the set bonus.
    # NOTE: base_cost is pre-grade; the grade multiplier applies on top.
    # Set pieces carry ~2x the plain-epic price as the set-bonus premium.
    add("storm_katana", "Storm Katana", 600, "weapon", "katana", "epic",
        base_atk=14, set_name="storm")
    add("storm_plate", "Storm Plate", 900, "armor", "plate", "epic",
        base_def=24, set_name="storm")
    add("shadow_rapier", "Shadow Rapier", 400, "weapon", "rapier", "epic",
        base_atk=10, set_name="shadow")
    add("shadow_mail", "Shadow Mail", 600, "armor", "chainmail", "epic",
        base_def=17, set_name="shadow")
    # Dragon set — for those who hunt the great wyrms. Carries a
    # strength bonus on top of the set stats.
    add("dragon_fang", "Dragon Fang Blade", 700, "weapon", "odachi", "epic",
        base_atk=26, set_name="dragon", stat_strength=3)
    add("dragon_scale", "Dragon Scale Armor", 1000, "armor", "dragonscale",
        "epic", base_def=32, set_name="dragon", stat_stamina=3)

    # The Cutyp legacy set — myth-tier, unbreakable, endgame priced.
    add("cutyp_steel_katana", "Cutyp Steel Katana", 800, "weapon",
        "katana", "myth", base_atk=35, set_name="cutyp",
        unbreakable=True)
    add("cutyp_robe", "Cutyp Robe", 650, "armor", "robe", "myth",
        base_def=30, set_name="cutyp", unbreakable=True)

    # ── raid-exclusive gear ─────────────────────────────────────────
    # Never sold in the shop — the only way to earn these is to bring
    # down a raid boss and get lucky.  The bossbane set is the best
    # non-myth kit in the game, which is exactly why raiders chase it.
    add("bossbane_cleaver", "Bossbane Cleaver", 0, "weapon", "cleaver",
        "legendary", base_atk=24, set_name="bossbane", raid_only=True)
    add("bossbane_plate", "Bossbane Plate", 0, "armor", "bossplate",
        "legendary", base_def=34, set_name="bossbane", raid_only=True)
    add("rustfang_dagger", "Rustfang Dagger", 0, "weapon", "dagger",
        "epic", base_atk=16, raid_only=True)
    add("hollow_greaves", "Hollow Greaves", 0, "armor", "greaves",
        "epic", base_def=26, raid_only=True)
    return defn


GEAR_CATALOG: dict[str, GearDef] = _build_catalog()

#: raid-exclusive gear slugs — the only pieces with raid_only=True.
#: Roll a drop from these; the shop never lists them.
RAID_EXCLUSIVE_GEAR: tuple[str, ...] = tuple(
    slug for slug, defn in GEAR_CATALOG.items() if defn.raid_only)


@dataclass(frozen=True)
class SetBonus:
    """What a complete set grants."""

    set_name: str
    needs: tuple[str, ...]      # slots that must match
    atk_pct: float              # bonus attack, fraction of base
    def_pct: float              # bonus defense, fraction of base
    combo_name: str             # the combo attack it unlocks
    combo_every: int            # every Nth attack strikes twice


SET_BONUSES: dict[str, SetBonus] = {
    "storm": SetBonus("storm", ("weapon", "armor"), atk_pct=0.25,
                      def_pct=0.25, combo_name="twin lightning",
                      combo_every=3),
    "shadow": SetBonus("shadow", ("weapon", "armor"), atk_pct=0.35,
                       def_pct=0.10, combo_name="umbral flurry",
                       combo_every=4),
    # The Cutyp legacy — the strongest set in the game. Unbreakable
    # pieces, so the combo never stops firing.
    "cutyp": SetBonus("cutyp", ("weapon", "armor"), atk_pct=0.50,
                      def_pct=0.50, combo_name="cutyp's fury",
                      combo_every=2),
    # Bossbane — forged from raid bosses, the best non-myth kit.
    # Only earnable as a raid drop, never bought.
    "bossbane": SetBonus("bossbane", ("weapon", "armor"), atk_pct=0.30,
                         def_pct=0.30, combo_name="bossbane rend",
                         combo_every=3),
    # Dragon — the wyrm hunter's pride. Hits hard, endures harder.
    "dragon": SetBonus("dragon", ("weapon", "armor"), atk_pct=0.35,
                       def_pct=0.35, combo_name="dragon's wrath",
                       combo_every=3),
}


def detect_set_bonus(equipped: dict[str, Any]) -> SetBonus | None:
    """Return the set bonus if the equipped pieces complete a set.

    ``equipped`` maps slot → GearInstance *or* gear slug.
    """
    by_set: dict[str, set[str]] = {}
    for slot, inst in (equipped or {}).items():
        if isinstance(inst, str):
            defn = GEAR_CATALOG.get(inst)
        else:
            defn = GEAR_CATALOG.get(inst.slug)
        if defn and defn.set_name:
            by_set.setdefault(defn.set_name, set()).add(slot)
    for set_name, slots in by_set.items():
        bonus = SET_BONUSES.get(set_name)
        if bonus and all(s in slots for s in bonus.needs):
            return bonus
    return None


def durability_bar(durability: int, max_durability: int,
                   width: int = 10) -> str:
    """Text durability meter, e.g. ``██████░░░░ 12/20``."""
    if max_durability <= 0:
        return "—"
    filled = int(round(width * max(0, durability) / max_durability))
    return ("█" * filled + "░" * (width - filled)
            + f" {max(0, durability)}/{max_durability}")


def durability_display(defn: GearDef | None = None, durability: int = 0,
                       max_durability: int = 0) -> str:
    """Human durability readout — unbreakable pieces show ``∞ unbreakable``."""
    if defn is not None and defn.unbreakable:
        return "∞ unbreakable"
    return durability_bar(durability, max_durability)


@dataclass
class GearInstance:
    """One owned piece of gear — durable, repairable, equippable."""

    id: str
    player_key: str
    slug: str
    durability: int
    max_durability: int
    equipped: bool = False
    created_at: float = field(default_factory=time.time)
    # Diablo-style rolled affixes: ((affix name, {stat: value}), ...).
    # () for plain catalog gear.
    affixes: tuple = ()

    @property
    def broken(self) -> bool:
        return self.durability <= 0

    def is_loot(self) -> bool:
        return self.slug.startswith("loot:") or bool(self.affixes)

    def base_slug(self) -> str:
        if self.slug.startswith("loot:"):
            return self.slug.split(":")[1]
        return self.slug

    def display_name(self) -> str:
        if self.is_loot():
            base = GEAR_CATALOG.get(self.base_slug())
            base_label = (base.kind.replace("_", " ").title()
                          if base else self.base_slug())
            pre = [n for n, _ in self.affixes
                   if not n.startswith("the ") and n != "Annihilation"]
            suf = [n for n, _ in self.affixes if n not in pre]
            name = (((" ".join(pre) + " ") if pre else "") + base_label
                    + ((" of " + ", ".join(s[3:] if s.startswith("of ")
                                           else s for s in suf))
                       if suf else ""))
            return " ".join(name.split())
        defn = GEAR_CATALOG.get(self.slug)
        return defn.name if defn else self.slug


class GearStore:
    """Persistent per-player gear. Backed by ``game_gear``.

    All mutations are transactional; a piece is never silently lost —
    wear only ever decreases durability, and breakage keeps the row so
    it can be repaired.
    """

    def __init__(self, db: Any) -> None:
        self.db = db
        self._migrated: set[str] = set()

    # ── schema ─────────────────────────────────────────────────────────────
    def _ensure(self) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS game_gear ("
                "id TEXT PRIMARY KEY, player_key TEXT NOT NULL, "
                "slug TEXT NOT NULL, durability INTEGER NOT NULL, "
                "max_durability INTEGER NOT NULL, "
                "equipped INTEGER NOT NULL DEFAULT 0, "
                "created_at REAL NOT NULL DEFAULT 0)")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_game_gear_player "
                "ON game_gear(player_key)")
            # affixes column for Diablo-style loot (added post-launch;
            # older tables get it via ALTER).
            try:
                self.db.execute(
                    "ALTER TABLE game_gear ADD COLUMN affixes TEXT "
                    "NOT NULL DEFAULT '[]'")
            except Exception:  # noqa: BLE001 - already there
                pass
        except Exception:  # noqa: BLE001 - table may already exist
            _log.debug("game_gear ensure failed", exc_info=True)

    # ── reads ──────────────────────────────────────────────────────────────
    def _row_to_instance(self, row: dict[str, Any]) -> GearInstance:
        inst = GearInstance(
            id=row["id"], player_key=row["player_key"], slug=row["slug"],
            durability=int(row["durability"]),
            max_durability=int(row["max_durability"]),
            equipped=bool(row["equipped"]),
            created_at=float(row.get("created_at") or 0.0),
        )
        try:
            import json as _json
            raw = row.get("affixes") or "[]"
            inst.affixes = tuple(
                (a[0], dict(a[1])) for a in _json.loads(raw))
        except Exception:  # noqa: BLE001
            inst.affixes = ()
        return inst

    def grant_loot(self, player_key: str, loot: LootItem) -> GearInstance:
        """Persist a forged loot piece. Loot keeps its rolled affixes in
        the ``affixes`` column; stats resolve through
        :func:`instance_stats`."""
        import json as _json
        self._ensure()
        base = GEAR_CATALOG.get(loot.base_slug)
        max_dur = base.max_durability if base else 40
        inst_id = new_id("gear")
        affixes_json = _json.dumps([[n, m] for n, m in loot.affixes])
        self.db.execute(
            "INSERT INTO game_gear (id, player_key, slug, durability, "
            "max_durability, equipped, created_at, affixes) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
            (inst_id, player_key, loot.slug, max_dur, max_dur,
             time.time(), affixes_json))
        inst = GearInstance(
            id=inst_id, player_key=player_key, slug=loot.slug,
            durability=max_dur, max_durability=max_dur,
            equipped=False, created_at=time.time())
        inst.affixes = loot.affixes
        return inst

    def list(self, player_key: str) -> list[GearInstance]:
        """Every piece the player owns, equipped first then newest."""
        self._ensure()
        self._migrate_legacy(player_key)
        if self.db is None:
            return []
        try:
            rows = self.db.query(
                "SELECT * FROM game_gear WHERE player_key = ? "
                "ORDER BY equipped DESC, created_at DESC", (player_key,))
        except Exception:  # noqa: BLE001
            _log.debug("game_gear list failed", exc_info=True)
            return []
        return [self._row_to_instance(r) for r in rows]

    def get(self, instance_id: str) -> GearInstance | None:
        self._ensure()
        if self.db is None:
            return None
        try:
            row = self.db.query_one(
                "SELECT * FROM game_gear WHERE id = ?", (instance_id,))
        except Exception:  # noqa: BLE001
            return None
        return self._row_to_instance(row) if row else None

    def equipped(self, player_key: str) -> dict[str, GearInstance]:
        """slot → instance for what's currently worn."""
        out: dict[str, GearInstance] = {}
        for inst in self.list(player_key):
            if not inst.equipped or inst.broken:
                continue
            defn = GEAR_CATALOG.get(inst.slug)
            if defn:
                out.setdefault(defn.slot, inst)
        return out

    def find(self, player_key: str, ref: str) -> GearInstance | None:
        """Resolve a piece by id prefix, slug, or (fuzzy) name."""
        ref = (ref or "").strip().lower()
        if not ref:
            return None
        pieces = self.list(player_key)
        for inst in pieces:
            if inst.id.lower().startswith(ref):
                return inst
        for inst in pieces:
            if inst.slug.lower() == ref:
                return inst
        for inst in pieces:
            if ref in inst.slug.lower() or ref in inst.display_name().lower():
                return inst
        return None

    # ── mutations ──────────────────────────────────────────────────────────
    def transfer(self, instance_id: str,
                 new_player_key: str) -> tuple[bool, str]:
        """Move a gear piece to another player (gifting).

        The piece is unequipped first — the recipient equips it
        themselves with /equip. Atomic single UPDATE.
        """
        self._ensure()
        if self.db is None:
            return False, "gear storage is unavailable."
        try:
            with self.db.transaction():
                row = self.db.query_one(
                    "SELECT id, slug FROM game_gear WHERE id = ?",
                    (instance_id,))
                if row is None:
                    return False, "that gear piece no longer exists."
                self.db.execute(
                    "UPDATE game_gear SET player_key = ?, equipped = 0 "
                    "WHERE id = ?",
                    (new_player_key, instance_id))
        except Exception as exc:  # noqa: BLE001
            return False, f"transfer failed: {exc}"
        defn = GEAR_CATALOG.get(
            row["slug"] if hasattr(row, "get") else row[1])
        name = defn.name if defn else "gear"
        return True, name

    def grant(self, player_key: str, slug: str) -> GearInstance:
        """Give the player a fresh piece. Raises KeyError on bad slug."""
        defn = GEAR_CATALOG.get(slug)
        if defn is None:
            raise KeyError(f"no such gear {slug!r}")
        self._ensure()
        inst = GearInstance(id=new_id("gear"), player_key=player_key,
                            slug=slug, durability=defn.max_durability,
                            max_durability=defn.max_durability)
        if self.db is not None:
            try:
                with self.db.transaction():
                    self.db.execute(
                        "INSERT INTO game_gear (id, player_key, slug, "
                        "durability, max_durability, equipped, created_at) "
                        "VALUES (?, ?, ?, ?, ?, 0, ?)",
                        (inst.id, player_key, slug, inst.durability,
                         inst.max_durability, inst.created_at))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"could not save gear: {exc}") from exc
        return inst

    def equip(self, player_key: str, ref: str) -> tuple[bool, str]:
        """Wear a piece (unequips whatever was in its slot)."""
        inst = self.find(player_key, ref)
        if inst is None:
            return False, f"you don't own {ref!r} — /game shop to buy gear."
        if inst.broken:
            return False, (f"{inst.display_name()} is broken — "
                           f"/repair {inst.slug} first.")
        defn = GEAR_CATALOG.get(inst.slug)
        if defn is None:
            return False, f"{inst.slug!r} isn't wearable gear."
        if self.db is None:
            return False, "gear storage is unavailable."
        try:
            with self.db.transaction():
                # unequip the whole slot, then wear this piece
                for other in self.list(player_key):
                    other_def = GEAR_CATALOG.get(other.slug)
                    if (other_def and other_def.slot == defn.slot
                            and other.equipped and other.id != inst.id):
                        self.db.execute(
                            "UPDATE game_gear SET equipped = 0 WHERE id = ?",
                            (other.id,))
                self.db.execute(
                    "UPDATE game_gear SET equipped = 1 WHERE id = ?",
                    (inst.id,))
        except Exception as exc:  # noqa: BLE001
            return False, f"equip failed: {exc}"
        atk, df = effective_stats(defn)
        bonus = detect_set_bonus(self.equipped(player_key))
        msg = (f"equipped {inst.display_name()} (+{atk} atk, +{df} def, "
               f"{durability_display(defn, inst.durability, inst.max_durability)})")
        if bonus:
            msg += (f"\n✨ {bonus.set_name} set complete — "
                    f"{bonus.combo_name}! (+{int(bonus.atk_pct*100)}% atk, "
                    f"+{int(bonus.def_pct*100)}% def, combo every "
                    f"{bonus.combo_every} attacks)")
        return True, msg

    def unequip(self, player_key: str,
                slot: str = "") -> tuple[bool, str]:
        """Take gear off. No slot = everything."""
        slot = (slot or "").strip().lower()
        if slot and slot not in SLOTS:
            return False, f"no such slot {slot!r} — {', '.join(SLOTS)}."
        worn = self.equipped(player_key)
        targets = [i for s, i in worn.items() if not slot or s == slot]
        if not targets:
            return False, "nothing equipped" + (f" in {slot}" if slot else "")
        if self.db is None:
            return False, "gear storage is unavailable."
        try:
            with self.db.transaction():
                for inst in targets:
                    self.db.execute(
                        "UPDATE game_gear SET equipped = 0 WHERE id = ?",
                        (inst.id,))
        except Exception as exc:  # noqa: BLE001
            return False, f"unequip failed: {exc}"
        names = ", ".join(i.display_name() for i in targets)
        return True, f"unequipped {names}."

    def wear(self, instance_id: str, amount: int = 1) -> tuple[bool, bool]:
        """Apply wear. Returns (ok, broke_now). Never deletes the row.

        Unbreakable pieces (the Cutyp legacy set) ignore wear entirely —
        they can never lose durability or break.

        Single atomic UPDATE — no read-modify-write, so two concurrent
        battles wearing the same piece can't lose a decrement.
        """
        if self.db is None or amount <= 0:
            return False, False
        self._ensure()
        try:
            with self.db.transaction():
                row = self.db.query_one(
                    "SELECT durability, slug FROM game_gear WHERE id = ?",
                    (instance_id,))
                if row is None:
                    return False, False
                defn = GEAR_CATALOG.get(str(row.get("slug") or ""))
                if defn is not None and defn.unbreakable:
                    return True, False
                before = int(row["durability"])
                if before <= 0:
                    return True, False
                cur = self.db.execute(
                    "UPDATE game_gear SET durability = max(0, durability - ?), "
                    "equipped = CASE WHEN durability - ? <= 0 THEN 0 "
                    "ELSE equipped END WHERE id = ? AND durability > 0",
                    (amount, amount, instance_id))
                if cur.rowcount == 0:
                    # another writer already wore it to 0
                    return True, False
                after_row = self.db.query_one(
                    "SELECT durability FROM game_gear WHERE id = ?",
                    (instance_id,))
                after = int(after_row["durability"]) if after_row else 0
                return True, before > 0 and after == 0
        except Exception:  # noqa: BLE001
            _log.debug("gear wear failed", exc_info=True)
            return False, False

    def repair(self, player_key: str, ref: str) -> tuple[bool, str]:
        """Restore a piece to full durability for coins.

        Returns (ok, message); the caller deducts coins on ok.  Cost is
        60% of the piece's shop price, scaled by missing durability.
        """
        inst = self.find(player_key, ref)
        if inst is None:
            return False, f"you don't own {ref!r}."
        if inst.durability >= inst.max_durability:
            return False, f"{inst.display_name()} is already at full."
        defn = GEAR_CATALOG.get(inst.slug)
        if defn is None:
            return False, f"{inst.slug!r} isn't repairable gear."
        missing = 1.0 - inst.durability / inst.max_durability
        cost = max(10, math.ceil(defn.cost * 0.6 * missing))
        return True, f"{inst.display_name()}|{cost}"

    def apply_repair(self, instance_id: str) -> bool:
        """Set durability back to max (after the caller took coins).

        Conditional: only repairs when durability is actually below max.
        Returns False when another concurrent repair already fixed it —
        the caller must refund in that case instead of double-charging.
        """
        if self.db is None:
            return False
        self._ensure()
        try:
            with self.db.transaction():
                cur = self.db.execute(
                    "UPDATE game_gear SET durability = max_durability "
                    "WHERE id = ? AND durability < max_durability",
                    (instance_id,))
                return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            _log.debug("gear repair failed", exc_info=True)
            return False

    def sync_equipped(self, player_key: str,
                      slot_to_id: dict[str, str]) -> None:
        """Make the DB match a battle's final loadout.

        Used on room close: whatever the arena had equipped (and not
        broken) stays equipped; everything else is unequipped.  Never
        deletes rows.
        """
        if self.db is None:
            return
        self._ensure()
        try:
            with self.db.transaction():
                self.db.execute(
                    "UPDATE game_gear SET equipped = 0 WHERE player_key = ?",
                    (player_key,))
                for instance_id in slot_to_id.values():
                    self.db.execute(
                        "UPDATE game_gear SET equipped = 1 "
                        "WHERE id = ? AND player_key = ? AND durability > 0",
                        (instance_id, player_key))
        except Exception:  # noqa: BLE001
            _log.debug("gear sync_equipped failed", exc_info=True)

    # ── legacy migration ───────────────────────────────────────────────────
    def _migrate_legacy(self, player_key: str) -> int:
        """Turn old count-based ``gear_sword``/``gear_armor`` into pieces.

        Runs once per player per process.  Players who paid for the old
        one-shot items get real durable gear of equivalent value instead
        of losing them.
        """
        if player_key in self._migrated or self.db is None:
            return 0
        self._migrated.add(player_key)
        try:
            from .players import PlayerStore
            store = PlayerStore(self.db)
            # Hold the (process-wide) store lock for the whole
            # read-modify-upsert so a concurrent game finish can't
            # slip a profile write between our read and our write.
            with store._lock:
                prof = store.get(player_key)
                moved = 0
                changed = False
                for legacy_slug, gear_slug in LEGACY_GEAR_MAP.items():
                    count = int(prof.items.get(legacy_slug) or 0)
                    for _ in range(max(0, count)):
                        try:
                            self.grant(player_key, gear_slug)
                            moved += 1
                        except Exception:  # noqa: BLE001
                            break
                    if count > 0:
                        prof.items.pop(legacy_slug, None)
                        changed = True
                if changed:
                    try:
                        # _upsert re-takes the (reentrant) lock; the write
                        # itself is one transaction.
                        store._write(prof)
                    except Exception:  # noqa: BLE001
                        _log.debug("legacy gear migration save failed",
                                   exc_info=True)
        except Exception:  # noqa: BLE001
            return 0
        if moved:
            _log.info("migrated %d legacy gear items for %s", moved,
                      player_key)
        return moved


# ── affixes: Diablo-style randomized loot ───────────────────────────────────
#
# Shop gear is fixed; the chase loop needs *drops*. Affixes are rolled
# Diablo-II-style:
#   * magic items: 1–2 affixes (25% prefix+suffix, 25% prefix only,
#     50% suffix only)
#   * rare items: 3–6 affixes, max 3 prefixes / 3 suffixes
#   * one affix per GROUP max — "Bronze" (+atk) and "Iron" (+atk) never
#     stack on the same piece, exactly like D2's affix groups
#   * affix value tiers are gated by item level (ilvl): deeper content
#     drops bigger numbers
# The affix names build the item name: "Bronze Broadsword of the Whale".

from dataclasses import dataclass as _dc2  # local alias, same dataclass


@_dc2(frozen=True)
class Affix:
    """One rollable modifier. ``mods`` maps stat → (lo, hi) per ilvl tier;
    the rolled value scales with the item's level."""
    name: str
    group: str            # one affix per group per item (D2 rule)
    kind: str             # "prefix" | "suffix"
    mods: dict[str, tuple[int, int]]
    min_ilvl: int = 1
    slots: tuple[str, ...] = ("weapon", "armor")


#: (ilvl threshold, value multiplier) — higher ilvl, bigger rolls.
_ILVL_TIERS: tuple[tuple[int, float], ...] = (
    (1, 0.6), (15, 0.8), (30, 1.0), (50, 1.3), (70, 1.6),
)

PREFIXES: tuple[Affix, ...] = (
    Affix("Bronze", "atk_flat", "prefix", {"atk": (1, 4)}, 1),
    Affix("Iron", "atk_flat", "prefix", {"atk": (3, 7)}, 12),
    Affix("Steel", "atk_flat", "prefix", {"atk": (6, 12)}, 28),
    Affix("Mithril", "atk_flat", "prefix", {"atk": (10, 18)}, 48),
    Affix("Cruel", "atk_pct", "prefix", {"atk": (4, 8)}, 20),
    Affix("Merciless", "atk_pct", "prefix", {"atk": (8, 15)}, 45),
    Affix("Sturdy", "def_flat", "prefix", {"def": (2, 6)}, 1),
    Affix("Stone", "def_flat", "prefix", {"def": (5, 11)}, 25),
    Affix("Diamond", "def_flat", "prefix", {"def": (9, 16)}, 50),
    Affix("Warlord's", "str", "prefix", {"strength": (2, 5)}, 15),
    Affix("Titan's", "stam", "prefix", {"stamina": (2, 5)}, 15),
    Affix("Sage's", "int", "prefix", {"intelligence": (2, 5)}, 20),
    Affix("Archmage's", "mana", "prefix", {"mana": (3, 8)}, 20),
    Affix("Swift", "speed", "prefix", {"atk": (1, 3), "def": (1, 3)}, 30),
)

SUFFIXES: tuple[Affix, ...] = (
    Affix("the Whale", "hp", "suffix", {"stamina": (2, 6)}, 1),
    Affix("the Bear", "str2", "suffix", {"strength": (1, 4)}, 8),
    Affix("the Owl", "int2", "suffix", {"intelligence": (1, 4)}, 8),
    Affix("the Leech", "leech", "suffix", {"atk": (2, 5)}, 35),
    Affix("the Titan", "big", "suffix",
          {"strength": (3, 6), "stamina": (3, 6)}, 40),
    Affix("the Storm", "storm", "suffix", {"atk": (3, 7), "mana": (2, 5)},
          45),
    Affix("the Fortress", "fort", "suffix", {"def": (4, 9)}, 30),
    Affix("Annihilation", "anni", "suffix",
          {"atk": (6, 12), "strength": (2, 5)}, 60),
    Affix("the Gods", "gods", "suffix",
          {"strength": (4, 8), "stamina": (4, 8),
           "intelligence": (4, 8)}, 70),
)

#: loot rarity → (magic-style affix budget, grade label)
_LOOT_RARITY = ("magic", "rare", "unique")


def _ilvl_mult(ilvl: int) -> float:
    mult = 0.6
    for threshold, m in _ILVL_TIERS:
        if ilvl >= threshold:
            mult = m
    return mult


def _roll_affix_count(rarity: str, rng: Any) -> tuple[int, int]:
    """(prefixes, suffixes) by D2 distribution."""
    if rarity == "magic":
        r = rng.random()
        if r < 0.25:
            return 1, 1
        if r < 0.50:
            return 1, 0
        return 0, 1
    if rarity == "rare":
        total = rng.randint(3, 6)
        pre = min(3, rng.randint(1, total - 1))
        return pre, min(3, total - pre)
    return 2, 2  # unique: fixed-feel, always both


def _pick_affixes(pool: tuple[Affix, ...], count: int, ilvl: int,
                  slot: str, used_groups: set[str], rng: Any
                  ) -> list[tuple[Affix, dict[str, int]]]:
    cands = [a for a in pool
             if a.min_ilvl <= ilvl and a.group not in used_groups
             and slot in a.slots]
    rng.shuffle(cands)
    out: list[tuple[Affix, dict[str, int]]] = []
    mult = _ilvl_mult(ilvl)
    for a in cands:
        if len(out) >= count:
            break
        used_groups.add(a.group)
        rolled = {stat: max(1, int(round(rng.randint(lo, hi) * mult)))
                  for stat, (lo, hi) in a.mods.items()}
        out.append((a, rolled))
    return out


@_dc2(frozen=True)
class LootItem:
    """A generated loot piece: base + rolled affixes."""
    slug: str
    name: str
    base_slug: str
    rarity: str            # magic | rare | unique
    ilvl: int
    affixes: tuple[tuple[str, dict[str, int]], ...]  # (affix name, rolled mods)

    def stat_bonuses(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for _name, mods in self.affixes:
            for stat, val in mods.items():
                total[stat] = total.get(stat, 0) + val
        return total


def forge_loot(base_slug: str, ilvl: int,
               rng: Any = None) -> LootItem | None:
    """Roll a Diablo-style loot piece on a catalog base item.

    Returns None for unknown/raid-only bases. The slug is unique
    (``loot:<base>:<hex>``) and the instance is grantable via
    :func:`grant_loot`.
    """
    import random as _random
    rng = rng or _random.Random()
    base = GEAR_CATALOG.get(base_slug)
    if base is None or base.raid_only:
        return None
    roll = rng.random()
    rarity = "magic" if roll < 0.60 else ("rare" if roll < 0.90 else "unique")
    n_pre, n_suf = _roll_affix_count(rarity, rng)
    used: set[str] = set()
    affixes = _pick_affixes(PREFIXES, n_pre, ilvl, base.slot, used, rng)
    affixes += _pick_affixes(SUFFIXES, n_suf, ilvl, base.slot, used, rng)
    pre_names = [a.name for a, _ in affixes if a.kind == "prefix"]
    suf_names = [a.name for a, _ in affixes if a.kind == "suffix"]
    base_label = base.name.split(" [")[0]
    if rarity == "rare":
        # D2 rares get evocative random names, not affix lists
        syllables = ("Dread", "Grim", "Storm", "Night", "Blood", "Iron",
                     "Doom", "Fang", "Skull", "Rune")
        name = (f"{rng.choice(syllables)} {rng.choice(syllables)}"
                .replace("  ", " "))
    else:
        name = ((" ".join(pre_names) + " " if pre_names else "")
                + base_label
                + (" of " + ", ".join(
                    s[3:] if s.startswith("of ") else s
                    for s in suf_names)
                   if suf_names else ""))
        name = " ".join(name.split())
    slug = f"loot:{base_slug}:{rng.getrandbits(32):08x}"
    return LootItem(slug=slug, name=name, base_slug=base_slug,
                    rarity=rarity, ilvl=ilvl,
                    affixes=tuple((a.name, mods) for a, mods in affixes))


def describe_loot(loot: LootItem) -> str:
    """``🌟 Dread Fang [rare ilvl 42] — Katana base · +9 atk, +4 str``."""
    base = GEAR_CATALOG.get(loot.base_slug)
    base_label = base.kind if base else loot.base_slug
    bonuses = loot.stat_bonuses()
    stat_words = {"atk": "atk", "def": "def", "strength": "str",
                  "stamina": "sta", "mana": "mana",
                  "intelligence": "int"}
    parts = [f"+{v} {stat_words.get(k, k)}"
             for k, v in sorted(bonuses.items())]
    stars = {"magic": "✨", "rare": "🌟", "unique": "💫"}[loot.rarity]
    return (f"{stars} {loot.name} [{loot.rarity} ilvl {loot.ilvl}] — "
            f"{base_label} base" + (f" · {', '.join(parts)}" if parts else ""))


def grant_loot(store: "GearStore", player_key: str,
               loot: LootItem) -> GearInstance | None:
    """Persist a forged loot piece into the player's inventory."""
    try:
        return store.grant_loot(player_key, loot)
    except Exception:  # noqa: BLE001
        _log.debug("grant_loot failed", exc_info=True)
        return None


def instance_stats(inst: GearInstance) -> dict[str, int]:
    """Full stat picture for an owned piece: base (attack, defense) plus
    rolled affix bonuses. Keys: atk, def, strength, stamina, mana,
    intelligence."""
    base = GEAR_CATALOG.get(inst.base_slug())
    atk, df = effective_stats(base) if base else (0, 0)
    out: dict[str, int] = {"atk": atk, "def": df}
    for _name, mods in inst.affixes:
        for stat, val in mods.items():
            out[stat] = out.get(stat, 0) + val
    return out


def instance_describe(inst: GearInstance) -> str:
    """``🌟 Bronze Broadsword of the Whale — 24 atk · +4 str``."""
    stats = instance_stats(inst)
    parts = []
    if stats.get("atk"):
        parts.append(f"{stats['atk']} atk")
    if stats.get("def"):
        parts.append(f"{stats['def']} def")
    for k in ("strength", "stamina", "mana", "intelligence"):
        if stats.get(k):
            parts.append(f"+{stats[k]} {k[:3]}")
    star = "🌟" if inst.is_loot() else "⚔️"
    dur = f"{inst.durability}/{inst.max_durability} dur"
    return (f"{star} {inst.display_name()} — {', '.join(parts)} · {dur}")
    """Persist a forged loot piece into the player's inventory."""
    try:
        return store.grant_loot(player_key, loot)
    except Exception:  # noqa: BLE001
        _log.debug("grant_loot failed", exc_info=True)
        return None
