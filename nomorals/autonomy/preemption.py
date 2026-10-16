"""Coordinator priority ladder + append-only preemption log.

When two firings want the same slot (the same organ, the same budget,
the owner's attention), the ladder decides:

    safety > owner-direct > scheduled > ambient > background

Lower rank number = higher priority. A firing may preempt a running /
queued firing only when it is *strictly* higher on the ladder — equal
priority never preempts (first-come wins, no thrash), and a lower
priority firing never displaces a higher one.

:class:`PreemptionLog` records every preemption append-only: entries
are appended as JSONL (or kept in-memory when no path is given) and
the log is never truncated or rewritten by this module. The log is
the audit trail for "why did my background job get killed" — it must
survive the preemption it records.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

__all__ = ["PRIORITY_LADDER", "PRIORITY_RANK", "PriorityLadder",
           "PreemptionLog", "should_preempt", "normalize_priority"]

PRIORITY_LADDER = ("safety", "owner-direct", "scheduled", "ambient",
                   "background")
PRIORITY_RANK = {name: rank for rank, name in enumerate(PRIORITY_LADDER)}

_ALIASES = {"owner_direct": "owner-direct", "owner": "owner-direct",
            "user": "owner-direct", "urgent": "safety", "critical": "safety",
            "normal": "scheduled", "low": "background"}


def normalize_priority(priority: str) -> str:
    """Canonical ladder name for a priority string. Unknown priorities
    fall back to ``"scheduled"`` rather than crashing the caller."""
    name = str(priority or "scheduled").lower().strip()
    name = _ALIASES.get(name, name)
    return name if name in PRIORITY_RANK else "scheduled"


def should_preempt(current: str, incoming: str) -> bool:
    """True when ``incoming`` may preempt ``current``: strictly higher
    on the ladder. Equal priority never preempts."""
    return PRIORITY_RANK[normalize_priority(incoming)] < PRIORITY_RANK[normalize_priority(current)]


class PriorityLadder:
    """The ladder as an object: rank, compare, sort."""

    ladder = PRIORITY_LADDER

    @staticmethod
    def rank(priority: str) -> int:
        return PRIORITY_RANK[normalize_priority(priority)]

    @staticmethod
    def higher(a: str, b: str) -> bool:
        """True when ``a`` outranks ``b`` (strictly)."""
        return PRIORITY_RANK[normalize_priority(a)] < PRIORITY_RANK[normalize_priority(b)]

    @staticmethod
    def should_preempt(current: str, incoming: str) -> bool:
        return should_preempt(current, incoming)

    @staticmethod
    def sort(items: list[Any], *, key: Callable[[Any], str] | None = None) -> list[Any]:
        """Highest-priority first, stable (equal priority keeps order)."""
        get = key or (lambda item: str(item))
        return sorted(items, key=lambda item: PRIORITY_RANK[normalize_priority(get(item))])


class PreemptionLog:
    """Append-only log of preemptions. Never truncated, never rewritten."""

    def __init__(self, path: str | Path | None = None, *,
                 clock: Callable[[], float] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock or time.time
        self._entries: list[dict[str, Any]] = []
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn trailing line: skipped, never fatal
                if isinstance(record, dict) and record.get("id"):
                    self._entries.append(record)
        except OSError:
            self._entries = []

    def log_preempted(self, victim: str, preempted_by: str, *,
                      reason: str = "",
                      victim_priority: str = "",
                      incoming_priority: str = "",
                      ts: float | None = None) -> dict[str, Any]:
        """Append one preemption record. Returns the record.

        ``victim`` is what got displaced; ``preempted_by`` is what
        displaced it. The caller decides whether the preemption is
        legitimate (see :func:`should_preempt`); the log records what
        happened, it does not re-judge it.
        """
        record = {
            "id": f"preempt-{uuid.uuid4().hex[:12]}",
            "ts": float(ts if ts is not None else self._clock()),
            "victim": str(victim),
            "preempted_by": str(preempted_by),
            "reason": str(reason or "higher-priority work displaced lower-priority work"),
            "victim_priority": normalize_priority(victim_priority) if victim_priority else "",
            "incoming_priority": normalize_priority(incoming_priority) if incoming_priority else "",
        }
        self._entries.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return dict(record)

    #: Alias — ``log(...)`` reads naturally at call sites.
    def log(self, victim: str, preempted_by: str, **kwargs: Any) -> dict[str, Any]:
        return self.log_preempted(victim, preempted_by, **kwargs)

    def entries(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self._entries]

    def for_victim(self, victim: str) -> list[dict[str, Any]]:
        return [dict(e) for e in self._entries if e["victim"] == str(victim)]

    def __len__(self) -> int:
        return len(self._entries)
