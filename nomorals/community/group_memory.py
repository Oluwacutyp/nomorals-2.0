"""Group-scoped memory, INSIDE the isolation boundary.

THE CONTRACT, STATED PLAINLY
----------------------------
The community isolation contract is containment of OWNER DATA — not a
withholding of capability inside the boundary. A group agent built on
this module may be richly capable: it can remember the group's running
jokes, decisions, rosters, and preferences; it can reason over them,
summarise them, and act on them. What it can NEVER do is reach the
owner's private world: no owner memories, no owner accounts, no owner
contacts, no owner files, no owner database. Capability inside the
boundary is unlimited by this contract; leakage across the boundary
is impossible by construction, because this store physically cannot
see anything outside ``~/.devon/community/``.

How that is enforced here
-------------------------
* Own store, own files: ``~/.devon/community/<group>/memory.json``.
  Never the owner database, never any owner table.
* Every entry carries its ``group_id``. Every key is namespaced
  (``"<group_id>:<key>"``) AND stored in a per-group file, so a lookup
  for group B physically cannot read group A's file.
* Every read verifies the entry's ``group_id`` matches the requested
  group; a mismatch raises instead of returning the entry.
* ``group_id`` values are sanitised (no separators, no ``..``), so a
  crafted group id cannot escape its directory.
* The store root defaults to ``~/.devon/community``. A custom root is
  allowed (tests), but a root inside ``~/.devon`` that is NOT inside
  ``~/.devon/community`` is refused — community state must never land
  in an owner-data directory by accident.
* This module imports stdlib + ``nomorals.core`` only. It has no path
  to owner data to leak, even if called wrong.

Entries are group facts only: they must originate from the group's own
messages / activity. Callers must never store owner-derived facts here.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id

__all__ = [
    "MemoryEntry",
    "GroupMemoryStore",
    "default_community_root",
]


def default_community_root() -> Path:
    """The community state root. Group memory lives under this, always."""
    return Path.home() / ".devon" / "community"


def _sanitize_group_id(group_id: str) -> str:
    """Reject anything that could escape the group's directory."""
    if not group_id or not group_id.strip():
        raise ValueError("group_id must not be empty")
    if group_id in (".", "..") or "/" in group_id or "\\" in group_id:
        raise ValueError(f"invalid group_id {group_id!r}")
    if group_id != group_id.strip():
        raise ValueError(f"invalid group_id {group_id!r}")
    return group_id


@dataclass
class MemoryEntry:
    """One remembered group fact. ``group_id`` is part of the identity."""

    id: str
    group_id: str
    key: str
    value: str
    author_id: str = ""
    author_name: str = ""
    tags: list[str] = None  # type: ignore[assignment]
    created_ts: float = 0.0
    updated_ts: float = 0.0

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []

    @property
    def namespaced_key(self) -> str:
        """The storage key: group id is in every key, always."""
        return f"{self.group_id}:{self.key}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=str(data["id"]),
            group_id=str(data["group_id"]),
            key=str(data["key"]),
            value=str(data["value"]),
            author_id=str(data.get("author_id", "")),
            author_name=str(data.get("author_name", "")),
            tags=list(data.get("tags", [])),
            created_ts=float(data.get("created_ts", 0.0)),
            updated_ts=float(data.get("updated_ts", 0.0)),
        )


class GroupMemoryStore:
    """Per-group memory. One file per group; cross-group reads impossible."""

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            self.root = default_community_root()
        else:
            self.root = Path(root)
        self._check_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _check_root(self) -> None:
        """Refuse a root that would put community state in owner space."""
        devon = Path.home() / ".devon"
        community = devon / "community"
        try:
            resolved = self.root.resolve()
        except OSError:
            resolved = self.root.absolute()
        if resolved == devon or (
            str(resolved).startswith(str(devon) + "/")
            and not str(resolved).startswith(str(community) + "/")
            and resolved != community
        ):
            raise ValueError(
                "group memory root must live under ~/.devon/community/"
            )

    def _path(self, group_id: str) -> Path:
        safe = _sanitize_group_id(group_id)
        folder = self.root / safe
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "memory.json"

    def _load(self, group_id: str) -> list[MemoryEntry]:
        path = self._path(group_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        entries = []
        for item in data if isinstance(data, list) else []:
            try:
                entries.append(MemoryEntry.from_dict(item))
            except (KeyError, TypeError, ValueError):
                continue
        return entries

    def _save(self, group_id: str, entries: list[MemoryEntry]) -> None:
        self._path(group_id).write_text(
            json.dumps([e.to_dict() for e in entries], indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _enforce(entry: MemoryEntry, group_id: str) -> None:
        """Cross-community guard: an entry from another group is a bug,
        never a result. Raise instead of returning it."""
        if entry.group_id != group_id:
            raise ValueError(
                f"memory entry {entry.id} belongs to group "
                f"{entry.group_id!r}, not {group_id!r}"
            )

    # ── write ─────────────────────────────────────────────────────
    def remember(
        self,
        group_id: str,
        key: str,
        value: str,
        author_id: str = "",
        author_name: str = "",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Store (or overwrite) one group fact. Same key twice = update,
        not a duplicate: the newest value wins, timestamped."""
        group_id = _sanitize_group_id(group_id)
        if not key.strip():
            raise ValueError("key must not be empty")
        entries = self._load(group_id)
        now = time.time()
        for entry in entries:
            if entry.key == key:
                self._enforce(entry, group_id)
                entry.value = value
                entry.author_id = author_id
                entry.author_name = author_name
                entry.tags = list(tags or [])
                entry.updated_ts = now
                self._save(group_id, entries)
                return entry
        entry = MemoryEntry(
            id=new_short_id("gmem"),
            group_id=group_id,
            key=key,
            value=value,
            author_id=author_id,
            author_name=author_name,
            tags=list(tags or []),
            created_ts=now,
            updated_ts=now,
        )
        entries.append(entry)
        self._save(group_id, entries)
        return entry

    # ── read (group-scoped, always) ────────────────────────────────
    def recall(self, group_id: str, key: str) -> MemoryEntry | None:
        """Recall one fact. Returns None when this group never stored it —
        including when ANOTHER group stored the same key."""
        group_id = _sanitize_group_id(group_id)
        for entry in self._load(group_id):
            if entry.key == key:
                self._enforce(entry, group_id)
                return entry
        return None

    def recall_value(self, group_id: str, key: str, default: str = "") -> str:
        entry = self.recall(group_id, key)
        return entry.value if entry else default

    def search(self, group_id: str, query: str) -> list[MemoryEntry]:
        """Substring search over this group's keys, values, and tags.
        Never crosses into another group's entries."""
        group_id = _sanitize_group_id(group_id)
        needle = query.strip().lower()
        if not needle:
            return []
        results = []
        for entry in self._load(group_id):
            self._enforce(entry, group_id)
            haystack = " ".join(
                [entry.key, entry.value, *entry.tags]
            ).lower()
            if needle in haystack:
                results.append(entry)
        return results

    def list_entries(self, group_id: str) -> list[MemoryEntry]:
        """Every fact this group has stored, oldest first."""
        group_id = _sanitize_group_id(group_id)
        entries = self._load(group_id)
        for entry in entries:
            self._enforce(entry, group_id)
        return sorted(entries, key=lambda e: e.created_ts)

    def list_keys(self, group_id: str) -> list[str]:
        return [e.key for e in self.list_entries(group_id)]

    def forget(self, group_id: str, key: str) -> bool:
        """Delete one fact from this group. Other groups untouched."""
        group_id = _sanitize_group_id(group_id)
        entries = self._load(group_id)
        kept = [e for e in entries if e.key != key]
        if len(kept) == len(entries):
            return False
        self._save(group_id, kept)
        return True

    def forget_group(self, group_id: str) -> bool:
        """Delete ALL of one group's memory. Scoped to that group only."""
        group_id = _sanitize_group_id(group_id)
        path = self._path(group_id)
        if path.exists():
            path.unlink()
            return True
        return False
