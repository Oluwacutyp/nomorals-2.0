"""Named snapshots of assembled context: save, rewind, restore.

A mission that is about to try a risky plan can snapshot its context first
and rewind to it afterwards.  Snapshots are plain JSON on disk — inspectable,
diffable, portable.

:func:`snapshot_diff` compares two snapshots (or two :class:`BuiltContext`
objects) section by section: which sections appeared, vanished, flipped
truncated, and where the token deltas are.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .engine import BuiltContext
from .sections import Section

__all__ = [
    "SnapshotStore",
    "snapshot_to_dict",
    "snapshot_from_dict",
    "snapshot_diff",
    "format_snapshot_diff",
]

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
                "volatile": s.volatile,
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
            volatile=bool(s.get("volatile", False)),
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


def snapshot_diff(
    older: BuiltContext, newer: BuiltContext
) -> dict[str, Any]:
    """Section-level diff between two built contexts.

    Returns ``token_delta`` (newer − older), plus per-section rows with
    ``status`` in ``added``/``removed``/``changed``/``same`` and flags for
    sections that flipped ``truncated`` either way.
    """
    old_map = {s.name: s for s in older.sections}
    new_map = {s.name: s for s in newer.sections}
    rows: list[dict[str, Any]] = []
    for name in dict.fromkeys(list(old_map) + list(new_map)):
        old_s = old_map.get(name)
        new_s = new_map.get(name)
        if old_s is None:
            rows.append({
                "name": name, "status": "added",
                "old_tokens": 0, "new_tokens": new_s.tokens,
                "token_delta": new_s.tokens,
                "truncated_flip": bool(new_s.truncated),
            })
        elif new_s is None:
            rows.append({
                "name": name, "status": "removed",
                "old_tokens": old_s.tokens, "new_tokens": 0,
                "token_delta": -old_s.tokens,
                "truncated_flip": False,
            })
        else:
            changed = old_s.content != new_s.content
            rows.append({
                "name": name,
                "status": "changed" if changed else "same",
                "old_tokens": old_s.tokens,
                "new_tokens": new_s.tokens,
                "token_delta": new_s.tokens - old_s.tokens,
                "truncated_flip": old_s.truncated != new_s.truncated,
            })
    rows.sort(key=lambda r: abs(r["token_delta"]), reverse=True)
    return {
        "old_tokens": older.total_tokens,
        "new_tokens": newer.total_tokens,
        "token_delta": newer.total_tokens - older.total_tokens,
        "sections": rows,
    }


def format_snapshot_diff(diff: dict[str, Any], *, style: str = "plain") -> str:
    """Render a :func:`snapshot_diff` result as readable text."""
    fancy = style == "fancy"
    delta = diff["token_delta"]
    sign = "+" if delta >= 0 else ""
    lines = [
        f"Snapshot diff: ~{diff['old_tokens']} → ~{diff['new_tokens']} "
        f"tokens ({sign}{delta})"
    ]
    for row in diff["sections"]:
        if row["status"] == "same" and not row["truncated_flip"]:
            continue
        marker = {
            "added": "+" if not fancy else "✚",
            "removed": "-" if not fancy else "✖",
            "changed": "~" if not fancy else "≈",
        }[row["status"]]
        flip = " [truncation flipped]" if row["truncated_flip"] else ""
        d = row["token_delta"]
        dsign = "+" if d >= 0 else ""
        lines.append(
            f"  {marker} {row['name']:<14} {dsign}{d} tokens{flip}"
        )
    if len(lines) == 1:
        lines.append("  (no section changes)")
    return "\n".join(lines)


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

    def describe(self, name: str) -> dict[str, Any]:
        """Snapshot metadata without loading the full text content."""
        target = self._file(name)
        if not target.exists():
            raise FileNotFoundError(f"no context snapshot named {name!r}")
        data = json.loads(target.read_text(encoding="utf-8"))
        sections = data.get("sections", [])
        return {
            "name": data.get("name", name),
            "saved_at": data.get("saved_at"),
            "total_tokens": data.get("total_tokens", 0),
            "over_budget": bool(data.get("over_budget", False)),
            "budget_total": data.get("budget_total", 0),
            "section_count": len(sections),
            "sections": [s.get("name") for s in sections],
            "dropped": list(data.get("dropped", [])),
            "truncated": list(data.get("truncated", [])),
        }

    def prune(self, keep_n: int = 10) -> list[str]:
        """Keep the ``keep_n`` most recently saved snapshots; delete the rest.

        Returns the names that were deleted.
        """
        if keep_n < 0:
            raise ValueError("keep_n must be non-negative")
        files = sorted(
            self.path.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        doomed = files[keep_n:]
        deleted = [p.stem for p in doomed]
        for path in doomed:
            path.unlink()
        return deleted

    def delete(self, name: str) -> bool:
        target = self._file(name)
        if target.exists():
            target.unlink()
            return True
        return False
