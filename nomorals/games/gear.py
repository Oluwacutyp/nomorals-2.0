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
    "LEGACY_GEAR_MAP", "GearStore",
    "effective_stats", "detect_set_bonus", "durability_bar",
]

_log = get_logger(__name__)

#: Grade → stat multiplier.
GRADE_MULT: dict[str, float] = {
    "common": 1.0,
    "rare": 1.25,
    "epic": 1.5,
    "legendary": 2.0,
}
GRADES = tuple(GRADE_MULT)

#: Grade → max durability (hits before the piece breaks).
GRADE_DURABILITY: dict[str, int] = {
    "common": 25,
    "rare": 40,
    "epic": 60,
    "legendary": 100,
}

#: Equipment slots. One piece per slot equipped at a time.
SLOTS = ("weapon", "armor", "trinket")

#: Grade → price multiplier over the kind's base cost.
_GRADE_COST_MULT: dict[str, float] = {
    "common": 1.0, "rare": 2.0, "epic": 4.0, "legendary": 7.0,
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
    grade: str           # common | rare | epic | legendary
    set_name: str = ""   # "" = no set; else e.g. "storm"
    base_atk: int = 0
    base_def: int = 0

    @property
    def max_durability(self) -> int:
        return GRADE_DURABILITY[self.grade]

    def to_dict(self) -> dict[str, Any]:
        atk, df = effective_stats(self)
        return {"slug": self.slug, "name": self.name, "cost": self.cost,
                "slot": self.slot, "kind": self.kind, "grade": self.grade,
                "set": self.set_name, "atk": atk, "def": df,
                "durability": self.max_durability}


def effective_stats(defn: GearDef) -> tuple[int, int]:
    """(attack, defense) after the grade multiplier."""
    mult = GRADE_MULT[defn.grade]
    return (int(round(defn.base_atk * mult)),
            int(round(defn.base_def * mult)))


def _build_catalog() -> dict[str, GearDef]:
    defn: dict[str, GearDef] = {}

    def add(slug: str, name: str, base_cost: int, slot: str, kind: str,
            grade: str, base_atk: int = 0, base_def: int = 0,
            set_name: str = "") -> None:
        cost = int(base_cost * _GRADE_COST_MULT[grade])
        defn[slug] = GearDef(slug=slug, name=f"{name} [{grade}]",
                             cost=cost, slot=slot, kind=kind, grade=grade,
                             set_name=set_name, base_atk=base_atk,
                             base_def=base_def)

    weapons = (("katana", "Katana", 300, 12),
               ("broadsword", "Broadsword", 350, 15),
               ("rapier", "Rapier", 200, 8),
               ("warhammer", "Warhammer", 450, 20))
    for kind, label, base_cost, base_atk in weapons:
        for grade in GRADES:
            add(f"{kind}_{grade}", label, base_cost, "weapon", kind,
                grade, base_atk=base_atk)

    armors = (("leather", "Leather Armor", 150, 8),
              ("chainmail", "Chainmail", 300, 15),
              ("plate", "Plate Armor", 450, 22),
              ("dragonscale", "Dragonscale Mail", 600, 30))
    for kind, label, base_cost, base_def in armors:
        for grade in GRADES:
            add(f"{kind}_{grade}", label, base_cost, "armor", kind,
                grade, base_def=base_def)

    # Named sets — matching weapon + armor unlock the set bonus.
    add("storm_katana", "Storm Katana", 1400, "weapon", "katana", "epic",
        base_atk=14, set_name="storm")
    add("storm_plate", "Storm Plate", 1600, "armor", "plate", "epic",
        base_def=24, set_name="storm")
    add("shadow_rapier", "Shadow Rapier", 1400, "weapon", "rapier", "epic",
        base_atk=10, set_name="shadow")
    add("shadow_mail", "Shadow Mail", 1600, "armor", "chainmail", "epic",
        base_def=17, set_name="shadow")
    return defn


GEAR_CATALOG: dict[str, GearDef] = _build_catalog()


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

    @property
    def broken(self) -> bool:
        return self.durability <= 0

    def display_name(self) -> str:
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
        except Exception:  # noqa: BLE001 - table may already exist
            _log.debug("game_gear ensure failed", exc_info=True)

    # ── reads ──────────────────────────────────────────────────────────────
    def _row_to_instance(self, row: dict[str, Any]) -> GearInstance:
        return GearInstance(
            id=row["id"], player_key=row["player_key"], slug=row["slug"],
            durability=int(row["durability"]),
            max_durability=int(row["max_durability"]),
            equipped=bool(row["equipped"]),
            created_at=float(row.get("created_at") or 0.0),
        )

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
               f"{durability_bar(inst.durability, inst.max_durability)})")
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
        """Apply wear. Returns (ok, broke_now). Never deletes the row."""
        if self.db is None or amount <= 0:
            return False, False
        self._ensure()
        try:
            with self.db.transaction():
                row = self.db.query_one(
                    "SELECT durability FROM game_gear WHERE id = ?",
                    (instance_id,))
                if row is None:
                    return False, False
                before = int(row["durability"])
                after = max(0, before - amount)
                self.db.execute(
                    "UPDATE game_gear SET durability = ?, equipped = "
                    "CASE WHEN ? <= 0 THEN 0 ELSE equipped END "
                    "WHERE id = ?", (after, after, instance_id))
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
        """Set durability back to max (after the caller took coins)."""
        inst = self.get(instance_id)
        if inst is None or self.db is None:
            return False
        try:
            with self.db.transaction():
                self.db.execute(
                    "UPDATE game_gear SET durability = max_durability "
                    "WHERE id = ?", (instance_id,))
            return True
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
            prof = store.get(player_key)
        except Exception:  # noqa: BLE001
            return 0
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
                store._upsert(prof)
            except Exception:  # noqa: BLE001
                _log.debug("legacy gear migration save failed",
                           exc_info=True)
        if moved:
            _log.info("migrated %d legacy gear items for %s", moved,
                      player_key)
        return moved
