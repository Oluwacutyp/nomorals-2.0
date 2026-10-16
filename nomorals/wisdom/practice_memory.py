"""Personal practice memory + spaced repetition, grounded in corpus.

Wisdom gap fix: practice sessions evaporated — no memory of what the
owner practiced, what stuck, or what to review. This module is the
memory: recall items (each *must* cite a corpus source — a session
without a source is rejected, so practice stays grounded), SM-2
scheduling, due lists, and session → item links.

SM-2 (SuperMemo 2, the public algorithm): quality 0–5 per review;
q ≥ 3 grows the interval (1 → 6 → interval × easiness), q < 3 resets
to a 1-day re-learn with a lapse count. Easiness is floored at 1.3.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RecallItem", "PracticeMemory", "MissingSourceError"]


class MissingSourceError(ValueError):
    """A recall item without a corpus source is rejected — practice
    must cite its source text/tradition, or it is not storable."""


@dataclass
class RecallItem:
    """One thing to remember: prompt + answer pointer + source."""

    id: str
    prompt: str
    answer_ref: str  # pointer to the answer (passage id / quote ref)
    source: dict[str, Any]  # {tradition, work, section, url} — REQUIRED
    easiness: float = 2.5
    interval_days: int = 0
    repetitions: int = 0
    due_at: float = 0.0
    lapses: int = 0
    reviews: int = 0
    created_at: float = 0.0
    session_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecallItem":
        return cls(**{k: d.get(k, v) for k, v in {
            "id": "", "prompt": "", "answer_ref": "", "source": {},
            "easiness": 2.5, "interval_days": 0, "repetitions": 0,
            "due_at": 0.0, "lapses": 0, "reviews": 0, "created_at": 0.0,
            "session_ids": []}.items()})


def _validate_source(source: dict[str, Any]) -> None:
    if not isinstance(source, dict):
        raise MissingSourceError("recall item needs a source dict")
    missing = [k for k in ("tradition", "work") if not source.get(k)]
    if missing:
        raise MissingSourceError(
            f"recall item source is missing {missing} — every practice "
            "item must cite its source text/tradition")


class PracticeMemory:
    """Recall items + SM-2 scheduling, persisted as JSON."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._items: dict[str, RecallItem] = {}
        if self._path and self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                for d in raw.get("items", []):
                    item = RecallItem.from_dict(d)
                    self._items[item.id] = item
            except (OSError, ValueError):
                self._items = {}

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"items": [i.to_dict() for i in self._items.values()]},
            indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    def add_item(self, prompt: str, answer_ref: str, *,
                 source: dict[str, Any], session_id: str = "",
                 now: float | None = None) -> RecallItem:
        """Add a recall item. Source (tradition + work) is REQUIRED."""
        if not (prompt or "").strip():
            raise ValueError("add_item needs a prompt")
        _validate_source(source)
        item = RecallItem(
            id=uuid.uuid4().hex[:12], prompt=prompt.strip(),
            answer_ref=answer_ref, source=dict(source),
            due_at=now if now is not None else time.time(),
            created_at=now if now is not None else time.time(),
            session_ids=[session_id] if session_id else [])
        self._items[item.id] = item
        self._save()
        return item

    def get(self, item_id: str) -> RecallItem:
        try:
            return self._items[item_id]
        except KeyError:
            raise ValueError(f"unknown recall item {item_id!r}") from None

    def link_session(self, item_id: str, session_id: str) -> RecallItem:
        item = self.get(item_id)
        if session_id and session_id not in item.session_ids:
            item.session_ids.append(session_id)
            self._save()
        return item

    def review(self, item_id: str, quality: int, *,
               now: float | None = None) -> RecallItem:
        """SM-2 review. ``quality`` 0–5 (5 = perfect recall)."""
        if not 0 <= quality <= 5:
            raise ValueError(f"quality must be 0..5, got {quality!r}")
        item = self.get(item_id)
        now = now if now is not None else time.time()
        item.reviews += 1
        if quality < 3:
            item.repetitions = 0
            item.interval_days = 1
            item.lapses += 1
        else:
            if item.repetitions == 0:
                item.interval_days = 1
            elif item.repetitions == 1:
                item.interval_days = 6
            else:
                item.interval_days = max(
                    1, round(item.interval_days * item.easiness))
            item.repetitions += 1
        item.easiness = max(
            1.3, item.easiness + 0.1 - (5 - quality)
            * (0.08 + (5 - quality) * 0.02))
        item.due_at = now + item.interval_days * 86400.0
        self._save()
        return item

    def due_items(self, *, now: float | None = None) -> list[RecallItem]:
        now = now if now is not None else time.time()
        return sorted((i for i in self._items.values() if i.due_at <= now),
                      key=lambda i: (i.due_at, i.id))

    def stats(self, *, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        items = list(self._items.values())
        return {"items": len(items),
                "due": len(self.due_items(now=now)),
                "reviewed": sum(i.reviews for i in items),
                "lapses": sum(i.lapses for i in items),
                "avg_easiness": round(
                    sum(i.easiness for i in items) / len(items), 3)
                if items else 0.0}
