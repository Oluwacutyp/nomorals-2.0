"""Offline action queue: persisted, FIFO, survives restarts.

Actions taken while offline are appended to ``queue.json`` under
``~/.devon/offline/`` on EVERY mutation — a crash or restart loses
nothing. Each action carries an entity + base version so the sync
engine can detect conflicts instead of clobbering newer state.

Statuses: ``pending`` → ``synced`` | ``conflict`` | ``failed``.
``failed`` actions stay in the queue with an error recorded, so a
failed sync is visible, never silently dropped.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["OfflineAction", "OfflineQueue", "default_offline_root"]


def default_offline_root() -> Path:
    """Offline state root. Queue + conflicts live under this, always."""
    return Path.home() / ".devon" / "offline"


@dataclass
class OfflineAction:
    """One action recorded while offline, awaiting sync."""

    id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    entity: str = ""
    base_version: int | None = None
    created_ts: float = field(default_factory=time.time)
    status: str = "pending"  # pending | synced | conflict | failed
    attempts: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OfflineAction":
        return cls(
            id=str(data["id"]), kind=str(data["kind"]),
            payload=dict(data.get("payload", {})),
            entity=str(data.get("entity", "")),
            base_version=data.get("base_version"),
            created_ts=float(data.get("created_ts", 0.0)),
            status=str(data.get("status", "pending")),
            attempts=int(data.get("attempts", 0)),
            error=str(data.get("error", "")),
        )


class OfflineQueue:
    """File-backed FIFO queue. Every mutation persists immediately."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_offline_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.root / "queue.json"

    def _load(self) -> list[OfflineAction]:
        p = self._path()
        if not p.exists():
            return []
        return [OfflineAction.from_dict(row)
                for row in json.loads(p.read_text(encoding="utf-8"))]

    def _save(self, actions: list[OfflineAction]) -> None:
        self._path().write_text(
            json.dumps([a.to_dict() for a in actions], indent=2, sort_keys=True),
            encoding="utf-8")

    def enqueue(self, kind: str, payload: dict[str, Any] | None = None, *,
                entity: str = "", base_version: int | None = None) -> OfflineAction:
        """Append an action. Returns the stored action (with id)."""
        actions = self._load()
        action = OfflineAction(
            id=f"q_{uuid.uuid4().hex[:12]}", kind=kind,
            payload=dict(payload or {}), entity=entity,
            base_version=base_version)
        actions.append(action)
        self._save(actions)
        return action

    def pending(self) -> list[OfflineAction]:
        """Pending actions in FIFO (enqueue) order."""
        return [a for a in self._load() if a.status == "pending"]

    def all(self) -> list[OfflineAction]:
        """Every action ever queued, in enqueue order."""
        return self._load()

    def get(self, action_id: str) -> OfflineAction | None:
        for a in self._load():
            if a.id == action_id:
                return a
        return None

    def update(self, action: OfflineAction) -> OfflineAction:
        """Persist a mutated action (status/attempts/error)."""
        actions = self._load()
        for i, existing in enumerate(actions):
            if existing.id == action.id:
                actions[i] = action
                self._save(actions)
                return action
        raise KeyError(f"no queued action {action.id!r}")

    def count_pending(self) -> int:
        return len(self.pending())

    def clear_synced(self) -> int:
        """Drop synced actions from the file. Returns how many dropped."""
        actions = self._load()
        kept = [a for a in actions if a.status != "synced"]
        self._save(kept)
        return len(actions) - len(kept)
