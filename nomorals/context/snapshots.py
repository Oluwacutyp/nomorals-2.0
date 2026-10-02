"""Named snapshots of assembled context: save, rewind, restore.

A mission that is about to try a risky plan can snapshot its context first
and rewind to it afterwards.  Snapshots are plain JSON on disk — inspectable,
diffable, portable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .engine import BuiltContext
from .sections import Section

__all__ = ["SnapshotStore", "snapshot_to_dict", "snapshot_from_dict"]

_SCHEMA_VERSION = 1


def snapshot_to_dict(built: BuiltContext, name: str) -> dict:
    return {
        "schema": _SCHEMA_VERSION,
        "name": name,
        "saved_at": time.time(),
        "text": built.text,
        "total_tokens": built.total_tokens,
        "over_budget": built.over_budget,
        "budget_total": built.budget_total,
        "dropped": built.dropped,
        "truncated": built.truncated,
        "sections": [
            {
                "name": s.name,
                "content": s.content,
                "priority": s.priority,
                "load_bearing": s.load_bearing,
                "keep": list(s.keep),
                "truncated": s.truncated,
                "dropped": s.dropped,
                "meta": s.meta,
            }
            for s in built.sections
        ],
    }


def snapshot_from_dict(data: dict) -> BuiltContext:
    sections = [
        Section(
            name=s["name"],
            content=s.get("content", ""),
            priority=s.get("priority", 50.0),
            load_bearing=bool(s.get("load_bearing", False)),
            keep=tuple(s.get("keep") or ()),
            truncated=bool(s.get("truncated", False)),
            dropped=bool(s.get("dropped", False)),
            meta=dict(s.get("meta") or {}),
        )
        for s in data.get("sections", [])
    ]
    return BuiltContext(
        text=data.get("text", ""),
        sections=sections,
        total_tokens=int(data.get("total_tokens", 0)),
        over_budget=bool(data.get("over_budget", False)),
        budget_total=int(data.get("budget_total", 0)),
        dropped=list(data.get("dropped", [])),
        truncated=list(data.get("truncated", [])),
    )


class SnapshotStore:
    """Save/restore named :class:`BuiltContext` snapshots as JSON files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, name: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
        if not safe:
            raise ValueError("snapshot name required")
        return self.path / f"{safe}.json"

    def save(self, name: str, built: BuiltContext) -> Path:
        """Persist a built context under ``name``.  Overwrites atomically."""
        target = self._file(name)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot_to_dict(built, name), indent=2),
                       encoding="utf-8")
        tmp.replace(target)
        return target

    def load(self, name: str) -> BuiltContext:
        """Restore a snapshot.  Raises FileNotFoundError when unknown."""
        target = self._file(name)
        if not target.exists():
            raise FileNotFoundError(f"no context snapshot named {name!r}")
        return snapshot_from_dict(
            json.loads(target.read_text(encoding="utf-8")))

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.path.glob("*.json"))

    def delete(self, name: str) -> bool:
        target = self._file(name)
        if target.exists():
            target.unlink()
            return True
        return False
