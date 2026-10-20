"""Per-person family store: one directory per person, no cross-person reads.

Layout under the family root (default ``~/.devon/family/``)::

    <root>/members.json          household roster (members.py owns it)
    <root>/audit.jsonl           household audit log (members.py owns it)
    <root>/<person_id>/profile.json
    <root>/<person_id>/items.json

Every method takes an explicit ``person_id`` and touches ONLY that
person's directory. There is no "list everything for everyone" path:
``get_item`` / ``list_items`` enforce shared-vs-private visibility in
code — private items are returned only to their owner or to a member
with the ``read_private_other`` permission (the household owner), and
a revoked member's items are owner-only regardless of visibility.

This store imports stdlib only. It cannot see owner memory, accounts,
vaults, or connectors — there is no path to them.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import roles as roles_mod
from .safety import is_kid_safe, scan_text

__all__ = ["FamilyItem", "PerPersonStore", "default_family_root"]


def default_family_root() -> Path:
    """The family state root. Per-person state lives under this, always."""
    return Path.home() / ".devon" / "family"


_PERSON_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: Household-structural filenames that live AT the family root. A person
#: id equal to one of these would make the person's directory collide
#: with the roster / audit log / shared-layer DB file itself.
_RESERVED_PERSON_IDS = frozenset({
    "members.json",      # members.py roster file
    "audit.jsonl",       # members.py audit log
    "household_shared.db",  # household shared-layer SQLite file
})


def sanitize_person_id(person_id: str) -> str:
    """Reject anything that could escape the person's directory."""
    if not person_id or not isinstance(person_id, str):
        raise ValueError("person_id must be a non-empty string")
    if person_id in (".", "..") or "/" in person_id or "\\" in person_id:
        raise ValueError(f"invalid person_id {person_id!r}")
    if person_id.lower() in _RESERVED_PERSON_IDS:
        raise ValueError(
            f"person_id {person_id!r} collides with a household file name")
    if not _PERSON_RE.match(person_id):
        raise ValueError(f"invalid person_id {person_id!r}")
    return person_id


_ROOT_LOCKS: dict[str, threading.RLock] = {}
_ROOT_LOCKS_GUARD = threading.Lock()


def root_lock(root: Path | str) -> threading.RLock:
    """The one lock per family root, shared by every store on that root.

    Item and roster files are whole-file JSON: a load-modify-save from
    two store/household instances over the same root must serialise,
    or the later save silently drops the earlier one's changes.
    """
    key = str(Path(root).resolve())
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ROOT_LOCKS[key] = lock
        return lock


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + ``os.replace``).

    A crash mid-write must never leave a truncated JSON file behind:
    the previous complete file stays in place until the new one is
    fully written, so a reader can never observe a torn state file.
    """
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class FamilyItem:
    """One stored family item. ``visibility`` is the sharing boundary."""

    id: str
    owner_id: str
    kind: str
    title: str
    body: str = ""
    visibility: str = "private"  # "shared" | "private"
    kid_safe: bool = True
    locale: str = "ng"
    created_ts: float = 0.0
    updated_ts: float = 0.0
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FamilyItem":
        return cls(
            id=str(data["id"]),
            owner_id=str(data["owner_id"]),
            kind=str(data.get("kind", "")),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            visibility=str(data.get("visibility", "private")),
            kid_safe=bool(data.get("kid_safe", True)),
            locale=str(data.get("locale", "ng")),
            created_ts=float(data.get("created_ts", 0.0)),
            updated_ts=float(data.get("updated_ts", 0.0)),
            flags=list(data.get("flags", [])),
        )


class PerPersonStore:
    """Per-person item storage with visibility enforced on every read."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_family_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = root_lock(self.root)

    # ── paths (per-person only) ──────────────────────────────────────
    def person_dir(self, person_id: str) -> Path:
        return self.root / sanitize_person_id(person_id)

    def _items_path(self, person_id: str) -> Path:
        return self.person_dir(person_id) / "items.json"

    def _profile_path(self, person_id: str) -> Path:
        return self.person_dir(person_id) / "profile.json"

    # ── profiles ─────────────────────────────────────────────────────
    def put_profile(self, person_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            d = self.person_dir(person_id)
            d.mkdir(parents=True, exist_ok=True)
            data = dict(profile)
            data["person_id"] = sanitize_person_id(person_id)
            data["updated_ts"] = time.time()
            atomic_write_text(
                self._profile_path(person_id),
                json.dumps(data, indent=2, sort_keys=True))
            return data

    def get_profile(self, person_id: str) -> dict[str, Any] | None:
        p = self._profile_path(person_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    # ── items ────────────────────────────────────────────────────────
    def _load_items(self, person_id: str) -> list[FamilyItem]:
        p = self._items_path(person_id)
        if not p.exists():
            return []
        raw = json.loads(p.read_text(encoding="utf-8"))
        items = []
        for row in raw:
            # Defence in depth: a row stored under person A must SAY A.
            if str(row.get("owner_id")) != sanitize_person_id(person_id):
                raise ValueError(
                    f"item {row.get('id')!r} owner mismatch in {person_id!r} store")
            items.append(FamilyItem.from_dict(row))
        return items

    def _save_items(self, person_id: str, items: list[FamilyItem]) -> None:
        d = self.person_dir(person_id)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self._items_path(person_id),
            json.dumps([it.to_dict() for it in items], indent=2, sort_keys=True))

    def put_item(self, item: FamilyItem, *, requester_role: str) -> FamilyItem:
        """Store an item. Kid-safe rules enforced for teen/child writers.

        The writer must be the item's owner (or the household owner
        acting on their behalf is handled by the registry layer, which
        passes the owner's own role). Child writers: any flagged
        content is refused. Teen writers: flagged content is masked
        by the caller (safety.filter_for_role) before it gets here;
        a teen item that still scans dirty is refused.
        """
        if item.visibility not in ("shared", "private"):
            raise ValueError(f"bad visibility {item.visibility!r}")
        role = roles_mod.validate_role(requester_role)
        text = f"{item.title}\n{item.body}"
        if role == roles_mod.CHILD and not is_kid_safe(text, item.locale):
            raise ValueError("content refused: not kid-safe for a child member")
        if role == roles_mod.TEEN and not is_kid_safe(text, item.locale):
            raise ValueError("content refused: teen content must be filtered first")
        flags = scan_text(text, item.locale)
        item.flags = flags
        item.kid_safe = not flags
        now = time.time()
        if not item.created_ts:
            item.created_ts = now
        item.updated_ts = now
        # Load-modify-save under the per-root lock: two store instances
        # over one root must not interleave, or one save drops the
        # other's items.
        with self._lock:
            items = self._load_items(item.owner_id)
            for i, existing in enumerate(items):
                if existing.id == item.id:
                    items[i] = item
                    self._save_items(item.owner_id, items)
                    return item
            items.append(item)
            self._save_items(item.owner_id, items)
            return item

    def get_item(self, person_id: str, item_id: str, *,
                 requester_id: str, requester_role: str,
                 requester_revoked: bool = False,
                 owner_revoked: bool = False) -> FamilyItem:
        """Read one item with visibility enforced.

        * Revoked requesters read nothing.
        * A revoked owner's items are readable only by the household
          owner (audit) — never by peers.
        * Private items: owner, or ``read_private_other`` holders.
        * Shared items: any non-revoked household member.
        """
        role = roles_mod.validate_role(requester_role)
        if requester_revoked:
            raise PermissionError("revoked members cannot read family items")
        if owner_revoked and role != roles_mod.OWNER:
            raise PermissionError("items of a revoked member are owner-only")
        for it in self._load_items(person_id):
            if it.id == item_id:
                if it.visibility == "shared":
                    return it
                if requester_id == person_id:
                    return it
                if roles_mod.can(role, "read_private_other"):
                    return it
                raise PermissionError("private item: not the owner")
        raise KeyError(f"no item {item_id!r} for person {person_id!r}")

    def list_items(self, person_id: str, *,
                   requester_id: str, requester_role: str,
                   requester_revoked: bool = False,
                   owner_revoked: bool = False) -> list[FamilyItem]:
        """List what ``requester`` may see of ``person_id``'s items."""
        role = roles_mod.validate_role(requester_role)
        if requester_revoked:
            raise PermissionError("revoked members cannot read family items")
        if owner_revoked and role != roles_mod.OWNER:
            return []
        out: list[FamilyItem] = []
        for it in self._load_items(person_id):
            if it.visibility == "shared":
                out.append(it)
            elif requester_id == person_id:
                out.append(it)
            elif roles_mod.can(role, "read_private_other"):
                out.append(it)
        return out

    def delete_item(self, person_id: str, item_id: str, *,
                    requester_id: str, requester_role: str) -> bool:
        """Delete your own item (owners may delete any household item)."""
        role = roles_mod.validate_role(requester_role)
        with self._lock:
            items = self._load_items(person_id)
            for i, it in enumerate(items):
                if it.id == item_id:
                    if requester_id != person_id and role != roles_mod.OWNER:
                        raise PermissionError("cannot delete another member's item")
                    del items[i]
                    self._save_items(person_id, items)
                    return True
            return False
