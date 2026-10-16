"""Preparation usage — did the owner actually use what we prepared?

Autonomy prepares things ahead of time (morning briefings, pre-fetched
research, window prep). Preparation is only worth its cost if the
owner uses it. This module measures that:

* :meth:`UsageTracker.record_prep` — log one prepared item (a dict
  with at least a title, or a plain title string). Returns its id.
* :meth:`UsageTracker.mark_used` / :meth:`UsageTracker.mark_unused` —
  the verdict. A second verdict on an already-decided item is refused
  (returns ``False``) so a flip-flop cannot rewrite history.
* :meth:`UsageTracker.usage_rate` — ``used / decided`` over items with
  a verdict, plus raw counts. When nothing has been decided yet the
  rate is ``None`` — "no data", not a fabricated 0% or 100%.

Optional JSON persistence (atomic write-ahead, same pattern as the
unified scheduler queue) so usage survives restarts when a path is
given; without a path the tracker is in-memory.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .unified import atomic_write_json
import json

__all__ = ["UsageTracker"]


def _new_id() -> str:
    return f"prep-{uuid.uuid4().hex[:12]}"


class UsageTracker:
    """Measure whether prepared items were used."""

    def __init__(self, path: str | Path | None = None, *,
                 clock: Callable[[], float] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock or time.time
        self._items: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            items = raw.get("items", []) if isinstance(raw, dict) else raw
            for item in items or []:
                if isinstance(item, dict) and item.get("id"):
                    self._items[str(item["id"])] = dict(item)
        except (json.JSONDecodeError, OSError, ValueError):
            self._items = {}

    def _persist(self) -> None:
        if self.path is None:
            return
        atomic_write_json(self.path,
                          {"version": 1, "items": list(self._items.values())})

    def record_prep(self, item: Any, *,
                    kind: str = "", detail: str = "") -> str:
        """Log one prepared item. ``item`` may be a title string or a
        dict (``title`` / ``kind`` / ``detail`` keys honoured)."""
        if isinstance(item, dict):
            data = dict(item)
            title = str(data.pop("title", "") or data.get("name") or "prep")
            kind = str(data.pop("kind", "") or kind)
            detail = str(data.pop("detail", "") or detail)
        else:
            data = {}
            title = str(item or "prep")
        record = {
            "id": _new_id(),
            "title": title,
            "kind": kind,
            "detail": detail,
            "extra": data,
            "status": "pending",  # pending | used | unused
            "created_ts": self._clock(),
            "decided_ts": None,
            "note": "",
        }
        self._items[record["id"]] = record
        self._persist()
        return record["id"]

    def _decide(self, item_id: str, status: str, note: str = "") -> bool:
        record = self._items.get(str(item_id))
        if record is None or record["status"] != "pending":
            return False
        record["status"] = status
        record["decided_ts"] = self._clock()
        record["note"] = str(note or "")
        self._persist()
        return True

    def mark_used(self, item_id: str, note: str = "") -> bool:
        """Mark a prep item used. False when unknown / already decided."""
        return self._decide(item_id, "used", note)

    def mark_unused(self, item_id: str, note: str = "") -> bool:
        """Mark a prep item unused. False when unknown / already decided."""
        return self._decide(item_id, "unused", note)

    def get(self, item_id: str) -> dict[str, Any] | None:
        record = self._items.get(str(item_id))
        return dict(record) if record is not None else None

    def pending(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._items.values()
                if r["status"] == "pending"]

    def all(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._items.values()]

    def usage_rate(self) -> dict[str, Any]:
        """``{"total", "decided", "used", "unused", "pending", "rate"}``.

        ``rate`` is ``used / decided`` — ``None`` when nothing has been
        decided yet (no data, not zero)."""
        total = len(self._items)
        used = sum(1 for r in self._items.values() if r["status"] == "used")
        unused = sum(1 for r in self._items.values()
                     if r["status"] == "unused")
        decided = used + unused
        return {
            "total": total,
            "decided": decided,
            "used": used,
            "unused": unused,
            "pending": total - decided,
            "rate": (used / decided) if decided else None,
        }
