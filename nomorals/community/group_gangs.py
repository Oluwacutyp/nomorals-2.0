"""Cross-group gangs (§36): the registry above the groups.

§23/§24 built gangs INSIDE one group — a gang dict in that
group's conflict state with its treasury slice, armory,
territory and wars.  §36 keeps that machinery exactly as it is
and gives a gang an identity ABOVE groups:

- a **global record**: name (unique across the deployment),
  founder, lifestyle, prestige, the global roster (members as
  resolved identities — platform-bound person keys, §29), and
  **chapters**: one per group, each chapter binding the
  existing per-group gang dict in that group's conflict state
  (the gang dict carries ``"global": <record id>``).
  A single-group gang is a record with one chapter; nothing
  about its local life changes.  Registration binds a
  pre-existing local gang WITHOUT rewriting its history — the
  roster is read off the chapter, not re-founded.
- **the flagship**: PEACETHEFIRST, the owner's gang, seeded
  lazily into every claimed group (§33) with the owner as
  leader.  Its name and leadership are system constants:
  nobody else can register the name, hold its leadership, or
  dissolve the record.
- **expansion gating**: chapter capacity is a function of the
  gang's tier (the §24 ladder read over the GLOBAL roster),
  from the ``gangs`` settings block — data, not constants:
  below the gate tier a gang holds its home chapter only;
  the gate tier buys a few slots; the top tier is uncapped
  (``None`` in the table, never a big number).  The gate tier
  is ``gang`` — the same rung where inter-group war opens.
- **headquarters**: one chapter is the HQ.  Its tier
  (hideout → compound → fortress) is bought from the chapter
  treasury through the gang sub-account (no minting), its
  vault is a reserved partition of that treasury (counted in
  the conflict organ's reservations), its armory is the gang's
  shared arsenal — stored at the HQ chapter, drawable at any
  chapter — and its defense concentration is read by the
  conflict organ's heist/war/strength math.  Relocating the
  HQ costs the new chapter treasury and drops a tier: a
  fortress is built, not carried; the old vault releases back
  into the old chapter's spendable treasury (money never
  crosses economies).
- **loyalty**: one person holds at most ONE cross-group
  membership + ONE local membership; officership is exclusive
  across both; when their two gangs meet in an inter-group
  war the member declares a side for that war or sits it out
  (the surface's war-join reads the declaration).  A gang's
  class is read live: the flagship is always cross; any other
  gang is cross once its tier reaches the gate.
- **the owner's police override**: the §24 police stay a
  separate system gang; the owner alone can set a member's
  wanted stars (an owner mark the ladder reads), pardon
  (forgive the slate, close the episode, cancel police
  bounties), call off an in-flight enforcement (episode
  marked handled at its current rung, bounties cancelled,
  heat kept), and set a group's enforcement focus (the
  multiplier the bully index thresholds read).  Every
  override is journaled in the group's police journal —
  a held record, never a member-facing narration.

Storage discipline matches the war book / reputation book:
one JSON beside the group store, atomic saves, and an
unreadable file raises (frozen) rather than reading as an
empty registry.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from .group_conflict import (
    HQ_TIERS,
    LIFESTYLES,
    TIER_LADDER,
    GroupConflict,
    gang_tier,
    hq_tier_of,
)

__all__ = [
    "GANGS36_DEFAULTS", "GangRegistry", "GangRegistryError",
    "canon_platform", "person_key", "group_platform",
    "tier_level",
]

#: The ``gangs`` settings block (§36): the capacity table, the
#: flagship's identity, HQ prices and the raid/prestige tuning.
#: Settings data per the economy pattern — a group admin's
#: stored block deep-merges over these defaults.
GANGS36_DEFAULTS: dict[str, Any] = {
    "gate_tier": "gang",
    # tier name → chapter slots; None = uncapped (the max
    # tier's honest representation, never a huge number).
    "chapter_slots": {"crew": 1, "gang": 3, "syndicate": None},
    "flagship_name": "Peacethefirst",
    "flagship_lifestyle": "syndicate",
    "hq_costs": {"hideout": 500, "compound": 1500,
                 "fortress": 4000},
    "hq_relocate_cost": 750,
    "hq_raid": {"escrow_mult": 2, "pool_cap": 2000,
                "strength_per_tier_pct": 10,
                "pool_per_tier_pct": 15,
                "prestige_win": 40, "prestige_hold": 30},
    "prestige": {"chapter": 5, "war_win": 25, "hq_tier": 10},
    "claim_code_minutes": 30,
}

_TIER_LEVELS: dict[str, int] = {
    name: len(TIER_LADDER) - i for i, (_n, name) in
    enumerate(TIER_LADDER)}


def tier_level(tier_name: str) -> int:
    return _TIER_LEVELS.get(str(tier_name or ""), 1)


def canon_platform(platform: str) -> str:
    """One identity namespace per platform family: the
    telegram bot and the telegram userbot see the SAME user
    ids, so both spellings canon to ``telegram``."""
    plat = str(platform or "").strip().lower()
    if plat in ("telegram", "telegram-bot", "telegram-personal",
                "telegram-userbot"):
        return "telegram"
    return plat


def group_platform(group_key: str) -> str:
    return canon_platform(str(group_key or "").split(":", 1)[0])


def person_key(platform: str, user_id: str) -> str:
    """A resolved identity (§29): canonical platform + the
    platform's own id.  Persons are bound across platforms
    ONLY by claim codes — never by fuzzy matching."""
    return f"{canon_platform(platform)}:{user_id}"


def _name_key(name: str) -> str:
    return " ".join(str(name or "").split()).lower()


class GangRegistryError(RuntimeError):
    """The registry file is unreadable — frozen, never empty."""


class GangRegistry:
    """The §36 registry: global gang records over the
    per-group conflict organ."""

    def __init__(self, conflict: GroupConflict,
                 path: str | Path) -> None:
        self.conflict = conflict
        self.path = Path(path)

    # ── persistence ──────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"gangs": {}, "by_name": {},
                    "flagship_id": None, "claims": {},
                    "war_decls": {}, "war_awards": {},
                    "police_focus": {}, "journal": []}
        try:
            data = json.loads(self.path.read_text(
                encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GangRegistryError(
                f"the gang registry at {self.path} is "
                f"unreadable ({exc}); refusing to read it as "
                "empty") from exc
        if not isinstance(data, dict):
            raise GangRegistryError(
                f"the gang registry at {self.path} is not an "
                "object; refusing to read it as empty")
        for key, empty in (("gangs", {}), ("by_name", {}),
                           ("claims", {}), ("war_decls", {}),
                           ("war_awards", {}),
                           ("police_focus", {}), ("journal", [])):
            data.setdefault(key, empty)
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1,
                                  sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _journal(data: dict[str, Any], kind: str, now: float,
                 **fields: Any) -> None:
        data["journal"] = (list(data.get("journal") or []) + [
            {"kind": kind, "at": now, **fields}])[-800:]

    def journal(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._load().get("journal") or [])[-limit:]

    # ── config ───────────────────────────────────────────

    @staticmethod
    def _cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
        out = copy.deepcopy(GANGS36_DEFAULTS)
        for key, value in (cfg or {}).items():
            if key in out:
                out[key] = value
        return out

    # ── record reads ─────────────────────────────────────

    def gang(self, global_id: str) -> dict[str, Any] | None:
        return self._load()["gangs"].get(str(global_id or ""))

    def gang_by_name(self, name: str) -> dict[str, Any] | None:
        data = self._load()
        gid = (data.get("by_name") or {}).get(_name_key(name))
        return (data["gangs"].get(gid) if gid else None)

    def all_gangs(self) -> list[dict[str, Any]]:
        data = self._load()
        return [data["gangs"][k] for k in sorted(data["gangs"])]

    def flagship(self) -> dict[str, Any] | None:
        data = self._load()
        fid = data.get("flagship_id")
        if fid and fid in data["gangs"]:
            return data["gangs"][fid]
        for record in data["gangs"].values():
            if record.get("flagship"):
                return record
        return None

    def tier_name(self, record: dict[str, Any]) -> str:
        return gang_tier(len(record.get("roster") or {}))

    def capacity(self, record: dict[str, Any], *,
                 cfg: dict[str, Any] | None = None) -> int | None:
        """Chapter slots for the gang's CURRENT tier — None
        means uncapped, honestly (never a huge number)."""
        table = self._cfg(cfg).get("chapter_slots") or {}
        value = table.get(self.tier_name(record), 1)
        return None if value is None else int(value)

    def class_of(self, record: dict[str, Any], *,
                 cfg: dict[str, Any] | None = None) -> str:
        """"cross" when the gang counts against a member's
        cross-group slot: the flagship always; any gang at or
        above the gate tier.  Everything else is "local"."""
        if record.get("flagship"):
            return "cross"
        gate = str(self._cfg(cfg).get("gate_tier") or "gang")
        if tier_level(self.tier_name(record)) >= tier_level(gate):
            return "cross"
        return "local"

    def memberships(self, person: str) -> list[dict[str, Any]]:
        data = self._load()
        return [r for r in data["gangs"].values()
                if person in (r.get("roster") or {})]

    def person_for(self, group_key: str, local_id: str) -> str:
        """The canonical person key for a local identity: if
        any roster entry has this platform id bound (claim
        codes bind them), the entry's key — so a person bound
        on two platforms is ONE person everywhere.  Unbound
        identities key by their own platform, as minted."""
        plat = group_platform(group_key)
        wanted = str(local_id or "")
        data = self._load()
        for record in data["gangs"].values():
            for person, entry in (
                    record.get("roster") or {}).items():
                if (entry.get("platforms") or {}).get(
                        plat) == wanted:
                    return person
        return person_key(plat, wanted)

    def chapter_gang(self, record: dict[str, Any], group_key: str
                     ) -> dict[str, Any] | None:
        """The live chapter gang dict for a record in a group,
        or None when the binding is stale (the local gang is
        gone or no longer bound to this record)."""
        chapter = (record.get("chapters") or {}).get(group_key)
        if not chapter:
            return None
        state = self.conflict._load_state(group_key)
        gang = (state.get("gangs") or {}).get(
            str(chapter.get("gang") or ""))
        if gang is None or gang.get("global") != record["id"]:
            return None
        return gang

    def global_view(self, record: dict[str, Any], *,
                    cfg: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
        """The standing card data: global tier, prestige,
        chapters, HQ — everything /gang shows above the
        chapter you're standing in."""
        slots = self.capacity(record, cfg=cfg)
        hq = None
        hq_group = record.get("hq_group")
        if hq_group:
            gang = self.chapter_gang(record, str(hq_group))
            if gang is not None:
                tier = hq_tier_of(gang)
                hq = {"group": str(hq_group), "tier": tier,
                      "tier_name": HQ_TIERS[tier],
                      "vault": int(gang.get("hq_vault") or 0)}
        return {
            "id": record["id"], "name": record["name"],
            "lifestyle": record.get("lifestyle") or "street",
            "flagship": bool(record.get("flagship")),
            "founder": record.get("founder") or "",
            "tier": self.tier_name(record),
            "prestige": int(record.get("prestige") or 0),
            "roster_size": len(record.get("roster") or {}),
            "chapters": sorted(
                (record.get("chapters") or {}).keys()),
            "chapter_count": len(record.get("chapters") or {}),
            "capacity": slots,
            "capacity_text": ("uncapped" if slots is None
                              else f"{slots} chapter slots"),
            "class": self.class_of(record, cfg=cfg),
            "hq": hq}

    # ── registration (local gang → global record) ────────

    def register_local_gang(self, group_key: str,
                            local_gang_id: str, *,
                            now: float) -> dict[str, Any]:
        """Bind an EXISTING per-group gang into the registry.
        The gang's history is untouched: the roster is read
        off the chapter as it stands, the treasury stays in
        the chapter, and the only write to the chapter is its
        ``global`` binding."""
        state = self.conflict._load_state(group_key)
        gang = (state.get("gangs") or {}).get(
            str(local_gang_id or ""))
        if gang is None:
            return {"ok": False, "reason": "not_found"}
        data = self._load()
        bound = gang.get("global")
        if bound and bound in data["gangs"]:
            return {"ok": True, "gang": data["gangs"][bound],
                    "already": True}
        key = _name_key(str(gang.get("name") or ""))
        held = (data.get("by_name") or {}).get(key)
        if held and held in data["gangs"]:
            return {"ok": False, "reason": "name_registered",
                    "holder": data["gangs"][held]["name"]}
        gid = new_short_id("ggang")
        plat = group_platform(group_key)
        roster: dict[str, Any] = {}
        for member in gang.get("members") or []:
            role = ("leader" if member == gang.get("leader")
                    else "officer" if member in (
                        gang.get("officers") or [])
                    else "member")
            roster[person_key(plat, str(member))] = {
                "role": role, "since": float(
                    gang.get("created") or now),
                "platforms": {plat: str(member)},
                "name": "", "username": ""}
        record = {
            "id": gid, "name": str(gang.get("name") or ""),
            "name_key": key,
            "lifestyle": str(gang.get("lifestyle") or "street"),
            "founder": person_key(plat, str(
                gang.get("leader") or "")),
            "created": float(gang.get("created") or now),
            "flagship": False, "prestige": 0,
            "home": group_key,
            "chapters": {group_key: {
                "gang": str(gang.get("id") or ""),
                "founded": float(gang.get("created") or now)}},
            "hq_group": None, "roster": roster}
        data["gangs"][gid] = record
        data["by_name"][key] = gid
        gang["global"] = gid
        self.conflict._save_state(group_key, state)
        self._journal(data, "register", now, gang=gid,
                      name=record["name"], group=group_key)
        self._save(data)
        return {"ok": True, "gang": record}

    def ensure_registered(self, group_key: str,
                          gang: dict[str, Any], *,
                          now: float) -> dict[str, Any] | None:
        """The record for a chapter gang dict — registering it
        on first touch when it has none.  None when the name
        is held by another gang (the local gang then simply
        runs local-only; its street life is never blocked)."""
        data = self._load()
        bound = gang.get("global")
        if bound and bound in data["gangs"]:
            return data["gangs"][bound]
        out = self.register_local_gang(
            group_key, str(gang.get("id") or ""), now=now)
        return out.get("gang") if out.get("ok") else None

    # ── the flagship (Peacethefirst) ─────────────────────

    def ensure_flagship(self, group_key: str, *,
                        owner_person: str | None,
                        owner_local_id: str | None,
                        now: float,
                        cfg: dict[str, Any] | None = None
                        ) -> dict[str, Any]:
        """Seed the flagship chapter into a claimed group —
        lazily, on first contact, no daemon.  The flagship's
        identity is a system constant: its record is created
        once, its leader is the owner, and no admin action in
        any group can dissolve the record or take the name.
        When the owner's identity isn't known for this group's
        platform yet, seeding defers (owner_unknown) and the
        next contact retries."""
        conf = self._cfg(cfg)
        if not owner_person or not owner_local_id:
            return {"ok": False, "reason": "owner_unknown"}
        name = str(conf.get("flagship_name") or "Peacethefirst")
        lifestyle = str(conf.get("flagship_lifestyle")
                        or "syndicate")
        if lifestyle not in LIFESTYLES:
            lifestyle = "syndicate"
        data = self._load()
        record = None
        fid = data.get("flagship_id")
        if fid and fid in data["gangs"]:
            record = data["gangs"][fid]
        if record is None:
            held = (data.get("by_name") or {}).get(
                _name_key(name))
            if held and held in data["gangs"]:
                return {"ok": False,
                        "reason": "flagship_name_held",
                        "holder": data["gangs"][held]["name"]}
            gid = new_short_id("ggang")
            record = {
                "id": gid, "name": name,
                "name_key": _name_key(name),
                "lifestyle": lifestyle,
                "founder": owner_person, "created": now,
                "flagship": True, "prestige": 0,
                "home": group_key, "chapters": {},
                "hq_group": None,
                "roster": {owner_person: {
                    "role": "leader", "since": now,
                    "platforms": {
                        group_platform(group_key):
                            str(owner_local_id)},
                    "name": "", "username": ""}}}
            data["gangs"][gid] = record
            data["by_name"][_name_key(name)] = gid
            data["flagship_id"] = gid
            self._journal(data, "flagship_founded", now,
                          gang=gid, name=name)
        # the owner's leadership is the constant: heal any
        # drift (a restyle may change the name, never this).
        entry = (record.get("roster") or {}).get(owner_person)
        if entry is None:
            record.setdefault("roster", {})[owner_person] = {
                "role": "leader", "since": now,
                "platforms": {group_platform(group_key):
                              str(owner_local_id)},
                "name": "", "username": ""}
        else:
            entry["role"] = "leader"
            (entry.setdefault("platforms", {}))[
                group_platform(group_key)] = str(owner_local_id)
        record["founder"] = owner_person
        self._save(data)
        # the chapter itself
        gang = self.chapter_gang(record, group_key)
        if gang is not None:
            return {"ok": True, "gang": record, "chapter": gang,
                    "created": False}
        # a stale binding (chapter dissolved) re-seeds cleanly
        data = self._load()
        record = data["gangs"][record["id"]]
        record.get("chapters", {}).pop(group_key, None)
        self._save(data)
        out = self.conflict.create_gang(
            group_key, str(owner_local_id), record["name"],
            str(record["lifestyle"]), now=now)
        if out.get("ok"):
            gang = out["gang"]
        elif out.get("reason") == "taken":
            state = self.conflict._load_state(group_key)
            gang = self.conflict.gang_by_name(
                group_key, record["name"], state=state)
            if gang is None or (
                    gang.get("leader") != str(owner_local_id)
                    and gang.get("global") != record["id"]):
                return {"ok": False, "reason": "name_taken"}
        elif out.get("reason") == "already_in":
            return {"ok": False, "reason": "owner_elsewhere"}
        else:
            return {"ok": False,
                    "reason": str(out.get("reason") or "create")}
        state = self.conflict._load_state(group_key)
        live = (state.get("gangs") or {}).get(
            str(gang.get("id") or ""))
        if live is None:
            return {"ok": False, "reason": "gone"}
        live["global"] = record["id"]
        live["flagship"] = True
        self.conflict._save_state(group_key, state)
        data = self._load()
        record = data["gangs"][record["id"]]
        record.setdefault("chapters", {})[group_key] = {
            "gang": str(live.get("id") or ""), "founded": now}
        self._journal(data, "flagship_chapter", now,
                      gang=record["id"], group=group_key)
        self._save(data)
        return {"ok": True, "gang": record, "chapter": live,
                "created": True}

    def restyle_flagship(self, *, name: str | None = None,
                         lifestyle: str | None = None,
                         now: float) -> dict[str, Any]:
        """The owner's one edit on the flagship constant: its
        presentation.  Name changes re-key the registry (the
        old name frees); leadership never moves here."""
        record = self.flagship()
        if record is None:
            return {"ok": False, "reason": "no_flagship"}
        data = self._load()
        record = data["gangs"][record["id"]]
        if name:
            new_name = " ".join(str(name).split())[:32]
            if len(new_name) < 3:
                return {"ok": False, "reason": "name"}
            held = (data.get("by_name") or {}).get(
                _name_key(new_name))
            if held and held != record["id"]:
                return {"ok": False, "reason": "name_taken"}
            data["by_name"].pop(record["name_key"], None)
            record["name"] = new_name
            record["name_key"] = _name_key(new_name)
            data["by_name"][record["name_key"]] = record["id"]
        if lifestyle and lifestyle in LIFESTYLES:
            record["lifestyle"] = lifestyle
        self._journal(data, "flagship_restyle", now,
                      gang=record["id"], name=record["name"],
                      lifestyle=record["lifestyle"])
        self._save(data)
        return {"ok": True, "gang": record}

    # ── chapters: founding, joining, leaving ─────────────

    def _drop_chapter(self, data: dict[str, Any],
                      record: dict[str, Any], group_key: str
                      ) -> None:
        record.get("chapters", {}).pop(group_key, None)
        if record.get("hq_group") == group_key:
            record["hq_group"] = None
        if record.get("home") == group_key:
            rest = sorted((record.get("chapters") or {}).keys())
            record["home"] = rest[0] if rest else None

    def found_chapter(self, global_id: str, group_key: str, *,
                      by_person: str, by_local_id: str,
                      now: float,
                      cfg: dict[str, Any] | None = None
                      ) -> dict[str, Any]:
        """A leader/officer present in a new group founds the
        gang's chapter there — capacity permitting.  The
        chapter is a real §24 gang in that group, founded by
        them (they lead it locally), bound to the record."""
        data = self._load()
        record = data["gangs"].get(str(global_id or ""))
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        entry = (record.get("roster") or {}).get(by_person)
        if entry is None or entry.get("role") not in (
                "leader", "officer"):
            return {"ok": False, "reason": "rank"}
        if group_key in (record.get("chapters") or {}):
            if self.chapter_gang(record, group_key) is not None:
                return {"ok": False, "reason": "already_chapter"}
            self._drop_chapter(data, record, group_key)
            self._save(data)
            data = self._load()
            record = data["gangs"][record["id"]]
        slots = self.capacity(record, cfg=cfg)
        used = len(record.get("chapters") or {})
        if slots is not None and used >= slots:
            return {"ok": False, "reason": "capacity",
                    "tier": self.tier_name(record),
                    "slots": slots, "used": used,
                    "gang": record["name"]}
        out = self.conflict.create_gang(
            group_key, str(by_local_id), record["name"],
            str(record["lifestyle"]), now=now)
        if not out.get("ok"):
            reason = str(out.get("reason") or "")
            return {"ok": False,
                    "reason": "name_taken" if reason == "taken"
                    else reason}
        gang = out["gang"]
        state = self.conflict._load_state(group_key)
        live = (state.get("gangs") or {}).get(
            str(gang.get("id") or ""))
        if live is not None:
            live["global"] = record["id"]
            self.conflict._save_state(group_key, state)
        entry.setdefault("platforms", {})[
            group_platform(group_key)] = str(by_local_id)
        record.setdefault("chapters", {})[group_key] = {
            "gang": str(gang.get("id") or ""), "founded": now}
        record["prestige"] = int(record.get("prestige") or 0) \
            + int(self._cfg(cfg).get("prestige", {}).get(
                "chapter") or 0)
        self._journal(data, "chapter_founded", now,
                      gang=record["id"], group=group_key,
                      by=by_person)
        self._save(data)
        return {"ok": True, "gang": record, "chapter": gang}

    def join_chapter(self, global_id: str, group_key: str, *,
                     person: str, local_id: str,
                     name: str = "", username: str = "",
                     now: float,
                     cfg: dict[str, Any] | None = None
                     ) -> dict[str, Any]:
        """Join a gang through the chapter in this group —
        the loyalty rules (§36) read BEFORE the local join:
        one cross-group + one local membership, ever."""
        data = self._load()
        record = data["gangs"].get(str(global_id or ""))
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        gang = self.chapter_gang(record, group_key)
        if gang is None:
            return {"ok": False, "reason": "no_chapter"}
        roster = record.setdefault("roster", {})
        if person in roster:
            # already theirs globally — the local seat is all
            # that's missing (they hold the slot already).
            if str(local_id) in (gang.get("members") or []):
                return {"ok": False, "reason": "already"}
            out = self.conflict.join_gang(
                group_key, str(local_id), str(gang["name"]),
                now=now)
            if not out.get("ok"):
                return out
            roster[person].setdefault("platforms", {})[
                group_platform(group_key)] = str(local_id)
            self._save(data)
            return {"ok": True, "gang": record,
                    "chapter": out["gang"]}
        mine = self.class_of(record, cfg=cfg)
        for other in data["gangs"].values():
            if other["id"] == record["id"]:
                continue
            if person not in (other.get("roster") or {}):
                continue
            if self.class_of(other, cfg=cfg) == mine:
                return {"ok": False,
                        "reason": f"{mine}_taken",
                        "holder": other["name"]}
        out = self.conflict.join_gang(
            group_key, str(local_id), str(gang["name"]),
            now=now)
        if not out.get("ok"):
            return out
        roster[person] = {
            "role": "member", "since": now,
            "platforms": {group_platform(group_key):
                          str(local_id)},
            "name": str(name or ""), "username": str(
                username or "")}
        self._journal(data, "chapter_join", now,
                      gang=record["id"], group=group_key,
                      person=person)
        self._save(data)
        return {"ok": True, "gang": record,
                "chapter": out["gang"]}

    def leave_chapter(self, global_id: str, group_key: str, *,
                      person: str, local_id: str,
                      now: float) -> dict[str, Any]:
        """Leave the chapter in this group.  The global seat
        ends when the member holds no chapter anywhere —
        except the record's leader, whose seat IS the gang
        (leadership is global; a leader between chapters
        still leads)."""
        data = self._load()
        record = data["gangs"].get(str(global_id or ""))
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        out = self.conflict.leave_gang(group_key, str(local_id),
                                       now=now)
        if not out.get("ok"):
            return out
        if out.get("dissolved"):
            self._drop_chapter(data, record, group_key)
        entry = (record.get("roster") or {}).get(person)
        if entry is not None and entry.get("role") != "leader":
            still = False
            for other_group in list(
                    (record.get("chapters") or {}).keys()):
                chapter = self.chapter_gang(record, other_group)
                if chapter is None:
                    continue
                local = (entry.get("platforms") or {}).get(
                    group_platform(other_group))
                if local and local in (
                        chapter.get("members") or []):
                    still = True
                    break
            if not still:
                record["roster"].pop(person, None)
        self._journal(data, "chapter_leave", now,
                      gang=record["id"], group=group_key,
                      person=person,
                      dissolved=bool(out.get("dissolved")))
        self._save(data)
        return {"ok": True, "gang": record,
                "dissolved": bool(out.get("dissolved"))}

    # ── roles + officership exclusivity ──────────────────

    def officer_holds(self, person: str) -> list[str]:
        """Global ids where this person holds office — read
        from the roster AND live from the chapters (a chapter
        promotion lands in the chapter first)."""
        data = self._load()
        holds: list[str] = []
        for record in data["gangs"].values():
            entry = (record.get("roster") or {}).get(person)
            if entry is None:
                continue
            if entry.get("role") in ("leader", "officer"):
                holds.append(record["id"])
                continue
            for group_key in (record.get("chapters") or {}):
                chapter = self.chapter_gang(record, group_key)
                if chapter is None:
                    continue
                local = (entry.get("platforms") or {}).get(
                    group_platform(group_key))
                if local and (local == chapter.get("leader")
                              or local in (
                                  chapter.get("officers")
                                  or [])):
                    holds.append(record["id"])
                    break
        return holds

    def set_role(self, global_id: str, *, by_person: str,
                 person: str, role: str,
                 now: float) -> dict[str, Any]:
        """Grant a GLOBAL rank (leader/officer/member).  The
        record's leader grants; officership is exclusive
        across a person's two gangs — officer in one, never
        both.  The flagship's leadership is the owner's and
        cannot be granted away."""
        if role not in ("leader", "officer", "member"):
            return {"ok": False, "reason": "role"}
        data = self._load()
        record = data["gangs"].get(str(global_id or ""))
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        roster = record.setdefault("roster", {})
        by = roster.get(by_person)
        if by is None or by.get("role") != "leader":
            return {"ok": False, "reason": "not_leader"}
        target = roster.get(person)
        if target is None:
            return {"ok": False, "reason": "not_member"}
        if record.get("flagship") and role == "leader" \
                and person != record.get("founder"):
            return {"ok": False, "reason": "flagship_leader"}
        if role in ("leader", "officer"):
            elsewhere = [g for g in self.officer_holds(person)
                         if g != record["id"]]
            if elsewhere:
                holder = data["gangs"].get(elsewhere[0]) or {}
                return {"ok": False,
                        "reason": "officer_elsewhere",
                        "holder": holder.get("name") or ""}
        if role == "leader":
            for other_person, other in roster.items():
                if other.get("role") == "leader" \
                        and other_person != person:
                    other["role"] = "officer"
        target["role"] = role
        self._journal(data, "role", now, gang=record["id"],
                      person=person, role=role, by=by_person)
        self._save(data)
        return {"ok": True, "gang": record}

    # ── headquarters ─────────────────────────────────────

    def _hq_state(self, record: dict[str, Any]
                  ) -> tuple[str, dict[str, Any],
                             dict[str, Any]] | None:
        """(group_key, conflict state, chapter gang) for the
        record's HQ chapter — None when no live HQ stands."""
        hq_group = record.get("hq_group")
        if not hq_group:
            return None
        chapter = (record.get("chapters") or {}).get(
            str(hq_group))
        if not chapter:
            return None
        state = self.conflict._load_state(str(hq_group))
        gang = (state.get("gangs") or {}).get(
            str(chapter.get("gang") or ""))
        if gang is None or gang.get("global") != record["id"]:
            return None
        return str(hq_group), state, gang

    def hq_status(self, global_id: str) -> dict[str, Any]:
        record = self.gang(global_id)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        found = self._hq_state(record)
        if found is None:
            return {"ok": True, "gang": record, "hq": None}
        group_key, _state, gang = found
        tier = hq_tier_of(gang)
        return {"ok": True, "gang": record,
                "hq": {"group": group_key, "tier": tier,
                       "tier_name": HQ_TIERS[tier],
                       "vault": int(gang.get("hq_vault") or 0),
                       "treasury": int(gang.get("treasury")
                                       or 0),
                       "armory": list(gang.get("armory")
                                      or [])}}

    def hq_designate(self, global_id: str, group_key: str, *,
                     by_person: str,
                     now: float) -> dict[str, Any]:
        """Name this chapter the gang's HQ (global leader's
        call, made standing in that group).  One HQ per gang;
        moving an existing HQ is relocation, not this."""
        data = self._load()
        record = data["gangs"].get(str(global_id or ""))
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        entry = (record.get("roster") or {}).get(by_person)
        if entry is None or entry.get("role") != "leader":
            return {"ok": False, "reason": "not_leader"}
        gang = self.chapter_gang(record, group_key)
        if gang is None:
            return {"ok": False, "reason": "no_chapter"}
        if record.get("hq_group") == group_key:
            return {"ok": True, "gang": record, "already": True}
        if record.get("hq_group"):
            return {"ok": False, "reason": "hq_exists",
                    "hq_group": record["hq_group"]}
        state = self.conflict._load_state(group_key)
        live = (state.get("gangs") or {}).get(
            str(gang.get("id") or ""))
        if live is None:
            return {"ok": False, "reason": "gone"}
        live["hq"] = {"tier": 0}
        live.setdefault("hq_vault", 0)
        self.conflict._journal(state, "hq_designate", now,
                               gang=live["id"],
                               **{"global": record["id"]})
        self.conflict._save_state(group_key, state)
        record["hq_group"] = group_key
        self._journal(data, "hq_designate", now,
                      gang=record["id"], group=group_key,
                      by=by_person)
        self._save(data)
        return {"ok": True, "gang": record}

    def hq_buy_tier(self, global_id: str, *, by_local_id: str,
                    now: float,
                    cfg: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
        """Buy the next HQ tier from the HQ chapter's gang
        treasury — the sub-account decreases, nothing mints:
        hideout → compound → fortress."""
        conf = self._cfg(cfg)
        record = self.gang(global_id)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        found = self._hq_state(record)
        if found is None:
            return {"ok": False, "reason": "no_hq"}
        group_key, state, gang = found
        if str(by_local_id) != gang.get("leader") \
                and str(by_local_id) not in (
                    gang.get("officers") or []):
            return {"ok": False, "reason": "rank"}
        tier = hq_tier_of(gang)
        if tier >= len(HQ_TIERS) - 1:
            return {"ok": False, "reason": "max_tier"}
        next_name = HQ_TIERS[tier + 1]
        cost = int((conf.get("hq_costs") or {}).get(
            next_name) or 0)
        if int(gang.get("treasury") or 0) < cost:
            return {"ok": False, "reason": "funds",
                    "cost": cost,
                    "treasury": int(gang.get("treasury") or 0)}
        gang["treasury"] = int(gang.get("treasury") or 0) - cost
        gang.setdefault("hq", {})["tier"] = tier + 1
        self.conflict._journal(state, "hq_tier", now,
                               gang=gang["id"], tier=next_name,
                               cost=cost)
        self.conflict._save_state(group_key, state)
        data = self._load()
        live = data["gangs"][record["id"]]
        live["prestige"] = int(live.get("prestige") or 0) \
            + int(conf.get("prestige", {}).get("hq_tier") or 0)
        self._journal(data, "hq_tier", now, gang=record["id"],
                      group=group_key, tier=next_name,
                      cost=cost)
        self._save(data)
        return {"ok": True, "gang": live, "tier": tier + 1,
                "tier_name": next_name, "cost": cost}

    def hq_relocate(self, global_id: str, to_group: str, *,
                    by_person: str, now: float,
                    cfg: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
        """Move the HQ to another chapter: the NEW chapter's
        treasury pays the relocation cost, the tier drops by
        one (a fortress is built, not carried), and the old
        vault releases into the old chapter's spendable
        treasury — money never crosses economies."""
        conf = self._cfg(cfg)
        data = self._load()
        record = data["gangs"].get(str(global_id or ""))
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        entry = (record.get("roster") or {}).get(by_person)
        if entry is None or entry.get("role") != "leader":
            return {"ok": False, "reason": "not_leader"}
        found = self._hq_state(record)
        if found is None:
            return {"ok": False, "reason": "no_hq"}
        old_group, old_state, old_gang = found
        if to_group == old_group:
            return {"ok": False, "reason": "already_hq"}
        target = self.chapter_gang(record, to_group)
        if target is None:
            return {"ok": False, "reason": "no_chapter"}
        cost = int(conf.get("hq_relocate_cost") or 0)
        if int(target.get("treasury") or 0) < cost:
            return {"ok": False, "reason": "funds",
                    "cost": cost,
                    "treasury": int(target.get("treasury") or 0)}
        tier = hq_tier_of(old_gang)
        released = int(old_gang.get("hq_vault") or 0)
        old_gang["treasury"] = int(
            old_gang.get("treasury") or 0) + released
        old_gang["hq_vault"] = 0
        old_gang.pop("hq", None)
        self.conflict._journal(old_state, "hq_relocated_from",
                               now, gang=old_gang["id"],
                               released=released)
        self.conflict._save_state(old_group, old_state)
        state = self.conflict._load_state(to_group)
        live = (state.get("gangs") or {}).get(
            str(target.get("id") or ""))
        if live is None:
            return {"ok": False, "reason": "gone"}
        live["treasury"] = int(live.get("treasury") or 0) - cost
        live["hq"] = {"tier": max(0, tier - 1)}
        live["hq_vault"] = 0
        self.conflict._journal(state, "hq_relocated_to", now,
                               gang=live["id"], cost=cost,
                               tier=HQ_TIERS[max(0, tier - 1)])
        self.conflict._save_state(to_group, state)
        record["hq_group"] = to_group
        self._journal(data, "hq_relocate", now,
                      gang=record["id"], **{"from": old_group},
                      to=to_group, cost=cost,
                      tier=HQ_TIERS[max(0, tier - 1)],
                      by=by_person)
        self._save(data)
        return {"ok": True, "gang": record, "cost": cost,
                "tier": max(0, tier - 1),
                "tier_name": HQ_TIERS[max(0, tier - 1)],
                "released": released}

    def hq_vault_deposit(self, global_id: str, *,
                         local_id: str, amount: int,
                         now: float) -> dict[str, Any]:
        """Chapter treasury → the HQ vault partition.  Any HQ
        chapter member may feed the reserve; the partition is
        what raids come for and what reservations count."""
        record = self.gang(global_id)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        found = self._hq_state(record)
        if found is None:
            return {"ok": False, "reason": "no_hq"}
        group_key, state, gang = found
        if str(local_id) not in (gang.get("members") or []):
            return {"ok": False, "reason": "not_member"}
        amount = int(amount)
        if amount <= 0:
            return {"ok": False, "reason": "amount"}
        if int(gang.get("treasury") or 0) < amount:
            return {"ok": False, "reason": "funds",
                    "treasury": int(gang.get("treasury") or 0)}
        gang["treasury"] = int(gang.get("treasury") or 0) \
            - amount
        gang["hq_vault"] = int(gang.get("hq_vault") or 0) \
            + amount
        self.conflict._journal(state, "hq_vault_deposit", now,
                               gang=gang["id"], member=str(
                                   local_id), amount=amount)
        self.conflict._save_state(group_key, state)
        return {"ok": True, "amount": amount,
                "vault": int(gang["hq_vault"])}

    def hq_vault_withdraw(self, global_id: str, *,
                          local_id: str, amount: int,
                          now: float) -> dict[str, Any]:
        """The vault → chapter treasury, HQ chapter leader /
        officers only (spending the reserve is governance)."""
        record = self.gang(global_id)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        found = self._hq_state(record)
        if found is None:
            return {"ok": False, "reason": "no_hq"}
        group_key, state, gang = found
        if str(local_id) != gang.get("leader") \
                and str(local_id) not in (
                    gang.get("officers") or []):
            return {"ok": False, "reason": "rank"}
        amount = int(amount)
        if amount <= 0:
            return {"ok": False, "reason": "amount"}
        if int(gang.get("hq_vault") or 0) < amount:
            return {"ok": False, "reason": "funds",
                    "vault": int(gang.get("hq_vault") or 0)}
        gang["hq_vault"] = int(gang.get("hq_vault") or 0) \
            - amount
        gang["treasury"] = int(gang.get("treasury") or 0) \
            + amount
        self.conflict._journal(state, "hq_vault_withdraw", now,
                               gang=gang["id"], member=str(
                                   local_id), amount=amount)
        self.conflict._save_state(group_key, state)
        return {"ok": True, "amount": amount,
                "vault": int(gang["hq_vault"])}

    def hq_armory_store(self, global_id: str, from_group: str,
                        *, person: str, local_id: str,
                        item_id: str,
                        now: float) -> dict[str, Any]:
        """Store one owned item into the gang's shared armory:
        the instance leaves the member's loadout in the group
        they're standing in and lands in the HQ chapter's
        armory — gear crosses groups because the ARMORY is
        gang property, not chapter property."""
        record = self.gang(global_id)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        if person not in (record.get("roster") or {}):
            return {"ok": False, "reason": "not_member"}
        found = self._hq_state(record)
        if found is None:
            return {"ok": False, "reason": "no_hq"}
        chapter = self.chapter_gang(record, from_group)
        if chapter is None or str(local_id) not in (
                chapter.get("members") or []):
            return {"ok": False, "reason": "no_chapter"}
        state = self.conflict._load_state(from_group)
        items = list((state.get("loadouts") or {}).get(
            str(local_id)) or [])
        taken = None
        for idx, inst in enumerate(items):
            if inst.get("item") == item_id:
                taken = items.pop(idx)
                break
        if taken is None:
            return {"ok": False, "reason": "not_owned"}
        state["loadouts"][str(local_id)] = items
        self.conflict._journal(state, "hq_armory_store", now,
                               gang=str(chapter.get("id") or ""),
                               member=str(local_id),
                               item=item_id)
        self.conflict._save_state(from_group, state)
        hq_group, hq_state, hq_gang = self._hq_state(record)
        hq_gang.setdefault("armory", []).append(taken)
        self.conflict._journal(hq_state, "hq_armory_in", now,
                               gang=hq_gang["id"], item=item_id,
                               **{"from": from_group})
        self.conflict._save_state(hq_group, hq_state)
        return {"ok": True, "item": item_id}

    def hq_armory_draw(self, global_id: str, to_group: str, *,
                       person: str, local_id: str,
                       item_id: str,
                       now: float) -> dict[str, Any]:
        """Draw one item from the shared armory into the
        member's loadout in the group they're standing in.
        Any roster member seated in a chapter may draw — the
        armory exists so the gang fights as one gang."""
        record = self.gang(global_id)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        if person not in (record.get("roster") or {}):
            return {"ok": False, "reason": "not_member"}
        found = self._hq_state(record)
        if found is None:
            return {"ok": False, "reason": "no_hq"}
        chapter = self.chapter_gang(record, to_group)
        if chapter is None or str(local_id) not in (
                chapter.get("members") or []):
            return {"ok": False, "reason": "no_chapter"}
        hq_group, hq_state, hq_gang = found
        armory = list(hq_gang.get("armory") or [])
        taken = None
        for idx, inst in enumerate(armory):
            if inst.get("item") == item_id:
                taken = armory.pop(idx)
                break
        if taken is None:
            return {"ok": False, "reason": "not_stocked"}
        hq_gang["armory"] = armory
        self.conflict._journal(hq_state, "hq_armory_out", now,
                               gang=hq_gang["id"], item=item_id,
                               to=to_group)
        self.conflict._save_state(hq_group, hq_state)
        state = self.conflict._load_state(to_group)
        loadout = list((state.get("loadouts") or {}).get(
            str(local_id)) or [])
        loadout.append(taken)
        state.setdefault("loadouts", {})[str(local_id)] = loadout
        self.conflict._journal(state, "hq_armory_draw", now,
                               gang=str(chapter.get("id") or ""),
                               member=str(local_id),
                               item=item_id)
        self.conflict._save_state(to_group, state)
        return {"ok": True, "item": item_id}

    # ── war declarations + prestige outcomes ─────────────

    def note_war(self, war_id: str, gid_a: str, gid_b: str, *,
                 now: float) -> bool:
        """Pair two registered gangs under a war-book id so
        dual holders can declare a side for THIS war."""
        if not war_id or not gid_a or not gid_b \
                or gid_a == gid_b:
            return False
        data = self._load()
        if war_id in (data.get("war_decls") or {}):
            return True
        data.setdefault("war_decls", {})[war_id] = {
            "a": gid_a, "b": gid_b, "declared": {}}
        self._save(data)
        return True

    def _dual(self, record_a: dict[str, Any],
              record_b: dict[str, Any], person: str) -> bool:
        return person in (record_a.get("roster") or {}) \
            and person in (record_b.get("roster") or {})

    def declare_war_side(self, *, person: str, war_id: str,
                         global_id: str,
                         now: float) -> dict[str, Any]:
        """A dual holder picks their side for one war — the
        declaration is recorded per war and the war-join reads
        it.  Single-gang members need none (their side is
        their gang) and are refused plainly."""
        data = self._load()
        decl = (data.get("war_decls") or {}).get(war_id)
        if decl is None:
            return {"ok": False, "reason": "no_war"}
        if global_id not in (decl.get("a"), decl.get("b")):
            return {"ok": False, "reason": "not_side"}
        rec_a = data["gangs"].get(str(decl.get("a") or ""))
        rec_b = data["gangs"].get(str(decl.get("b") or ""))
        if rec_a is None or rec_b is None \
                or not self._dual(rec_a, rec_b, person):
            return {"ok": False, "reason": "not_dual"}
        decl.setdefault("declared", {})[person] = global_id
        self._journal(data, "war_declare_side", now,
                      war=war_id, person=person, side=global_id)
        self._save(data)
        return {"ok": True, "side": global_id}

    def war_join_check(self, *, person: str, war_id: str,
                       global_id: str) -> dict[str, Any]:
        """The loyalty gate on war participation: a dual
        holder fights ONLY for the side they declared for
        this war; undeclared, they sit it out."""
        data = self._load()
        decl = (data.get("war_decls") or {}).get(war_id)
        if decl is None:
            return {"ok": True}
        rec_a = data["gangs"].get(str(decl.get("a") or ""))
        rec_b = data["gangs"].get(str(decl.get("b") or ""))
        if rec_a is None or rec_b is None \
                or not self._dual(rec_a, rec_b, person):
            return {"ok": True}
        declared = (decl.get("declared") or {}).get(person)
        if declared == global_id:
            return {"ok": True}
        if declared:
            return {"ok": False, "reason": "declared_other"}
        return {"ok": False, "reason": "undeclared"}

    def note_inter_war_win(self, war_id: str, winner_gid: str,
                           *, now: float,
                           cfg: dict[str, Any] | None = None
                           ) -> bool:
        """Prestige for an inter-group war win — once per war,
        whichever side's cycle settles first."""
        data = self._load()
        awards = data.setdefault("war_awards", {})
        key = f"{war_id}:win"
        if key in awards or winner_gid not in data["gangs"]:
            return False
        awards[key] = now
        record = data["gangs"][winner_gid]
        record["prestige"] = int(record.get("prestige") or 0) \
            + int(self._cfg(cfg).get("prestige", {}).get(
                "war_win") or 0)
        self._journal(data, "prestige", now, gang=winner_gid,
                      why="war_win", war=war_id)
        self._save(data)
        return True

    def note_hq_raid_outcome(self, war_id: str, *,
                             attacker_gid: str,
                             defender_gid: str,
                             attacker_won: bool, now: float,
                             cfg: dict[str, Any] | None = None
                             ) -> bool:
        """Raid prestige — once per raid: the crown jewel
        pays the attacker most; a fortress that HOLDS earns
        its own."""
        conf = self._cfg(cfg)
        raid = conf.get("hq_raid") or {}
        data = self._load()
        awards = data.setdefault("war_awards", {})
        key = f"{war_id}:raid"
        if key in awards:
            return False
        winner = attacker_gid if attacker_won else defender_gid
        if winner not in data["gangs"]:
            return False
        awards[key] = now
        amount = int(raid.get("prestige_win") or 0) \
            if attacker_won else int(
                raid.get("prestige_hold") or 0)
        record = data["gangs"][winner]
        record["prestige"] = int(record.get("prestige") or 0) \
            + amount
        self._journal(data, "prestige", now, gang=winner,
                      why="hq_raid_win" if attacker_won
                      else "hq_raid_hold", war=war_id,
                      amount=amount)
        self._save(data)
        return True

    # ── claim codes (cross-platform chapter joining) ─────

    def mint_claim_code(self, global_id: str, *, person: str,
                        now: float,
                        cfg: dict[str, Any] | None = None
                        ) -> dict[str, Any]:
        """A roster member mints a one-shot code (the games
        join-code machinery): whoever redeems it from another
        platform binds THAT identity into this roster and
        takes the chapter seat there.  Codes expire; identity
        binds by proof of the code, never by name matching."""
        data = self._load()
        record = data["gangs"].get(str(global_id or ""))
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        if person not in (record.get("roster") or {}):
            return {"ok": False, "reason": "not_member"}
        from ..games.join_codes import new_join_code
        claims = data.setdefault("claims", {})
        code = new_join_code(set(claims.keys()))
        minutes = int(self._cfg(cfg).get(
            "claim_code_minutes") or 30)
        claims[code] = {"gang": record["id"], "person": person,
                        "at": now, "expires": now + minutes * 60}
        self._journal(data, "claim_minted", now,
                      gang=record["id"], person=person)
        self._save(data)
        return {"ok": True, "code": code,
                "expires": now + minutes * 60}

    def redeem_claim_code(self, code: str, group_key: str, *,
                          local_id: str, name: str = "",
                          username: str = "",
                          now: float) -> dict[str, Any]:
        """Redeem a claim code standing in a chapter's group
        on ANOTHER platform: the redeemer's local identity
        binds into the minter's roster entry, and the chapter
        seat is taken under that identity."""
        data = self._load()
        claims = data.setdefault("claims", {})
        entry = claims.pop(str(code or "").strip().upper(),
                           None)
        if entry is None:
            return {"ok": False, "reason": "bad_code"}
        if float(entry.get("expires") or 0) < now:
            self._save(data)
            return {"ok": False, "reason": "expired"}
        record = data["gangs"].get(str(entry.get("gang") or ""))
        if record is None:
            self._save(data)
            return {"ok": False, "reason": "no_gang"}
        person = str(entry.get("person") or "")
        roster_entry = (record.get("roster") or {}).get(person)
        if roster_entry is None:
            self._save(data)
            return {"ok": False, "reason": "gone"}
        plat = group_platform(group_key)
        roster_entry.setdefault("platforms", {})[plat] = str(
            local_id)
        if name and not roster_entry.get("name"):
            roster_entry["name"] = str(name)
        if username and not roster_entry.get("username"):
            roster_entry["username"] = str(username)
        self._journal(data, "claim_redeemed", now,
                      gang=record["id"], person=person,
                      group=group_key, platform=plat)
        self._save(data)
        gang = self.chapter_gang(record, group_key)
        if gang is None:
            return {"ok": False, "reason": "no_chapter",
                    "bound": True, "person": person,
                    "gang": record}
        if str(local_id) in (gang.get("members") or []):
            return {"ok": True, "bound": True, "joined": True,
                    "person": person, "gang": record,
                    "chapter": gang}
        out = self.conflict.join_gang(
            group_key, str(local_id), str(gang["name"]),
            now=now)
        if not out.get("ok"):
            return {"ok": False,
                    "reason": str(out.get("reason") or ""),
                    "bound": True, "person": person,
                    "gang": record}
        return {"ok": True, "bound": True, "joined": True,
                "person": person, "gang": record,
                "chapter": out["gang"]}

    # ── the owner's police override (§36) ────────────────

    def police_focus(self, group_key: str) -> float:
        """The group's enforcement-focus multiplier (owner-set;
        1.0 until set).  The bully index thresholds read it
        through the sweep."""
        data = self._load()
        focus = (data.get("police_focus") or {}).get(group_key)
        if not isinstance(focus, dict):
            return 1.0
        try:
            return float(focus.get("aggression") or 1.0)
        except (TypeError, ValueError):
            return 1.0

    def police_set_focus(self, group_key: str, aggression: float,
                         *, by_person: str,
                         now: float) -> dict[str, Any]:
        """Direct enforcement focus for a group: aggression
        scales the index the ladder reads (0.25×–3×)."""
        value = max(0.25, min(3.0, float(aggression)))
        data = self._load()
        data.setdefault("police_focus", {})[group_key] = {
            "aggression": value, "by": by_person, "at": now}
        self._journal(data, "police_override", now,
                      verb="focus", group=group_key,
                      aggression=value, by=by_person)
        self._save(data)
        state = self.conflict._load_state(group_key)
        self.conflict._journal(state, "police_override", now,
                               verb="focus", aggression=value,
                               by=by_person)
        self.conflict._save_state(group_key, state)
        return {"ok": True, "aggression": value}

    def police_view(self, group_key: str, member: str, *,
                    now: float) -> dict[str, Any]:
        """A member's wanted state as the override surface
        reads it: stars under the group's focus, the episode
        rung already served, open police bounties, the mark."""
        organ = self.conflict
        state = organ._load_state(group_key)
        focus = self.police_focus(group_key)
        stars, index, inputs = organ.wanted_stars(
            group_key, member, state=state, now=now,
            focus=focus)
        acted = dict((state.get("police") or {}).get(
            "acted") or {})
        bounties = [b for b in organ.open_bounties(
            group_key, state=state)
            if b.get("target") == member
            and b.get("source") == "police"]
        return {"member": member, "stars": stars,
                "index": index, "inputs": inputs,
                "focus": focus,
                "episode_rung": int(
                    (acted.get(member) or {}).get("rung") or 0),
                "mark": organ.owner_mark(
                    group_key, member, state=state, now=now),
                "police_bounties": len(bounties)}

    def police_set_stars(self, group_key: str, member: str,
                         stars: int, *, by_person: str,
                         now: float) -> dict[str, Any]:
        """Set a member's wanted level: an owner mark the
        ladder reads for the index window.  Journaled in the
        police journal — a held record of the owner's hand."""
        try:
            stars = int(stars)
        except (TypeError, ValueError):
            return {"ok": False, "reason": "range"}
        if stars < 0 or stars > 5:
            return {"ok": False, "reason": "range"}
        organ = self.conflict
        state = organ._load_state(group_key)
        police = state.setdefault("police", {})
        marks = police.setdefault("owner_marks", {})
        marks[str(member)] = {"stars": stars, "at": now,
                              "by": by_person}
        organ._journal(state, "police_override", now,
                       verb="set_stars", member=str(member),
                       stars=stars, by=by_person)
        organ._save_state(group_key, state)
        data = self._load()
        self._journal(data, "police_override", now,
                      verb="set_stars", group=group_key,
                      member=str(member), stars=stars,
                      by=by_person)
        self._save(data)
        return {"ok": True, **self.police_view(
            group_key, str(member), now=now)}

    @staticmethod
    def _cancel_police_bounties(state: dict[str, Any],
                                member: str, status: str) -> int:
        bounties = (state.get("police") or {}).get(
            "bounties") or {}
        count = 0
        for bounty in bounties.values():
            if bounty.get("target") == member \
                    and bounty.get("source") == "police" \
                    and bounty.get("status") == "open":
                bounty["status"] = status
                count += 1
        return count

    def police_pardon(self, group_key: str, member: str, *,
                      by_person: str,
                      now: float) -> dict[str, Any]:
        """Pardon: the slate is forgiven, the wanted episode
        closes, police bounties on the member are cancelled,
        and a standing mark holds them at 0★ for the window.
        The surface lifts any active jail — the one part of
        an override a member sees, as a release."""
        organ = self.conflict
        state = organ._load_state(group_key)
        focus = self.police_focus(group_key)
        _stars, index, _inputs = organ.wanted_stars(
            group_key, member, state=state, now=now,
            focus=focus)
        police = state.setdefault("police", {})
        acted = dict(police.get("acted") or {})
        was_rung = int((acted.pop(member, None) or {}).get(
            "rung") or 0)
        police["acted"] = acted
        forgiven = list((police.get("forgiven") or {}).get(
            member) or [])
        if index > 0:
            forgiven.append({"points": round(index, 2),
                             "at": now})
            police.setdefault("forgiven", {})[member] = forgiven
        police.setdefault("owner_marks", {})[str(member)] = {
            "stars": 0, "at": now, "by": by_person}
        cancelled = self._cancel_police_bounties(
            state, member, "pardoned")
        organ._journal(state, "police_override", now,
                       verb="pardon", member=str(member),
                       was_rung=was_rung, index=index,
                       bounties=cancelled, by=by_person)
        organ._save_state(group_key, state)
        data = self._load()
        self._journal(data, "police_override", now,
                      verb="pardon", group=group_key,
                      member=str(member), by=by_person)
        self._save(data)
        return {"ok": True, "stars": 0, "was_rung": was_rung,
                "bounties_cancelled": cancelled}

    def police_call_off(self, group_key: str, member: str, *,
                        by_person: str,
                        now: float) -> dict[str, Any]:
        """Call off an in-flight enforcement: the current
        episode is marked handled at its present rung (the
        ladder will not re-fire it), police bounties are
        cancelled — but the heat itself stays on the books.
        If the member climbs to a higher rung, the precinct
        moves again."""
        organ = self.conflict
        state = organ._load_state(group_key)
        focus = self.police_focus(group_key)
        stars, index, _inputs = organ.wanted_stars(
            group_key, member, state=state, now=now,
            focus=focus)
        police = state.setdefault("police", {})
        acted = dict(police.get("acted") or {})
        acted[str(member)] = {"rung": stars, "at": now,
                              "called_off": True}
        police["acted"] = acted
        cancelled = self._cancel_police_bounties(
            state, member, "called_off")
        organ._journal(state, "police_override", now,
                       verb="call_off", member=str(member),
                       stars=stars, index=index,
                       bounties=cancelled, by=by_person)
        organ._save_state(group_key, state)
        data = self._load()
        self._journal(data, "police_override", now,
                      verb="call_off", group=group_key,
                      member=str(member), by=by_person)
        self._save(data)
        return {"ok": True, "stars": stars, "index": index,
                "bounties_cancelled": cancelled}

    # ── people ───────────────────────────────────────────

    def person_view(self, person: str, *,
                    cfg: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
        """A person's gang standing: their memberships with
        roles and which loyalty slot each one fills."""
        out = []
        for record in self.memberships(person):
            entry = (record.get("roster") or {}).get(
                person) or {}
            out.append({
                "gang": record["name"], "id": record["id"],
                "role": entry.get("role") or "member",
                "slot": self.class_of(record, cfg=cfg),
                "flagship": bool(record.get("flagship")),
                "chapters": sorted(
                    (record.get("chapters") or {}).keys())})
        return {"person": person, "memberships": out,
                "officer_of": self.officer_holds(person)}
