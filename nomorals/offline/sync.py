"""Sync engine: merge queued offline actions on reconnect.

The host calls ``sync()`` when it knows connectivity is back — this
engine never polls, never retries on its own, and never assumes it
is online.

Merge semantics, in FIFO order:

* Action with ``base_version is None`` → apply unconditionally
  (the action does not depend on prior state).
* Action with ``base_version == current_versions[entity]`` → apply,
  then bump the entity's version to ``base_version + 1``.
* Action with ``base_version != current`` → CONFLICT: the world
  moved while offline. The action is NOT applied, NOT dropped — it
  is recorded in ``conflicts.json`` with status ``conflict`` so a
  human (or a merge policy) can resolve it. Sync continues with the
  remaining actions.
* An applier that raises → action marked ``failed`` with the error
  recorded; sync continues. Nothing is silently dropped.

The applier is an injected callable ``applier(action) -> Any`` and
the version map is owned by the caller (or this engine, starting
empty). The engine never touches the network or any real store
itself — it is the merge discipline, not the transport.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .queue import OfflineAction, OfflineQueue

__all__ = ["Conflict", "SyncEngine", "SyncResult"]


@dataclass
class Conflict:
    """A queued action that lost the race with newer state."""

    action_id: str
    kind: str
    entity: str
    base_version: int | None
    current_version: int | None
    detected_ts: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action_id": self.action_id, "kind": self.kind,
                "entity": self.entity, "base_version": self.base_version,
                "current_version": self.current_version,
                "detected_ts": self.detected_ts, "payload": self.payload}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conflict":
        return cls(action_id=str(data["action_id"]), kind=str(data["kind"]),
                   entity=str(data.get("entity", "")),
                   base_version=data.get("base_version"),
                   current_version=data.get("current_version"),
                   detected_ts=float(data.get("detected_ts", 0.0)),
                   payload=dict(data.get("payload", {})))


@dataclass
class SyncResult:
    """Outcome of one ``sync()`` run. Always complete, never partial-hidden."""

    applied: list[str] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.conflicts and not self.failed


class SyncEngine:
    """Drain an OfflineQueue through an injected applier + version map."""

    def __init__(self, queue: OfflineQueue,
                 applier: Callable[[OfflineAction], Any],
                 versions: dict[str, int] | None = None) -> None:
        self.queue = queue
        self.applier = applier
        self.versions: dict[str, int] = dict(versions or {})

    def _conflicts_path(self) -> Path:
        return self.queue.root / "conflicts.json"

    def _load_conflicts(self) -> list[Conflict]:
        p = self._conflicts_path()
        if not p.exists():
            return []
        return [Conflict.from_dict(row)
                for row in json.loads(p.read_text(encoding="utf-8"))]

    def _save_conflicts(self, conflicts: list[Conflict]) -> None:
        self._conflicts_path().write_text(
            json.dumps([c.to_dict() for c in conflicts], indent=2, sort_keys=True),
            encoding="utf-8")

    def conflicts(self) -> list[Conflict]:
        """All recorded conflicts, oldest first."""
        return self._load_conflicts()

    def sync(self) -> SyncResult:
        """Merge every pending action, in FIFO order. See module docstring."""
        result = SyncResult()
        stored_conflicts = self._load_conflicts()
        for action in self.queue.pending():
            action.attempts += 1
            current = self.versions.get(action.entity) if action.entity else None
            if (action.entity and action.base_version is not None
                    and current is not None and current != action.base_version):
                conflict = Conflict(
                    action_id=action.id, kind=action.kind,
                    entity=action.entity, base_version=action.base_version,
                    current_version=current, payload=dict(action.payload))
                stored_conflicts.append(conflict)
                action.status = "conflict"
                action.error = (
                    f"version moved: base {action.base_version} "
                    f"!= current {current}")
                self.queue.update(action)
                result.conflicts.append(conflict)
                continue
            try:
                self.applier(action)
            except Exception as exc:  # noqa: BLE001 — recorded, not hidden
                action.status = "failed"
                action.error = f"{type(exc).__name__}: {exc}"
                self.queue.update(action)
                result.failed.append({"action_id": action.id,
                                      "error": action.error})
                continue
            action.status = "synced"
            action.error = ""
            self.queue.update(action)
            if action.entity:
                self.versions[action.entity] = (action.base_version or 0) + 1 \
                    if action.base_version is not None \
                    else self.versions.get(action.entity, 0) + 1
            result.applied.append(action.id)
        self._save_conflicts(stored_conflicts)
        return result
