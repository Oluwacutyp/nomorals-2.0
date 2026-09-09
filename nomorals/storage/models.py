"""Typed dataclass mirrors of the core tables.

Rows come back from SQLite as mappings with string keys and no type information,
so a misspelled column is a silent ``None`` rather than an error. These dataclasses
give the hot paths real types and a single place where a column's decoding is
decided.

Deliberately *not* an ORM. There is no query builder, no session, no unit of work
— ``Repository`` and plain SQL still do the work. These classes only translate
rows in and out, which keeps the storage layer's behaviour exactly as tested
while making call sites readable.

Every ``from_row`` tolerates a row that is missing columns, because a migration
can add a column that older rows never had.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AgentRecord",
    "TaskRecord",
    "Reflection",
    "ModelRow",
    "decode_json",
    "encode_json",
]


def encode_json(value: Any) -> Any:
    """Encode a JSON column for storage.

    ``Repository`` encodes on the way in, but a raw ``db.insert()`` does not, and
    sqlite3 refuses to bind a dict. Encoding here means ``to_row()`` produces
    something both paths accept, instead of only working through one of them.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def decode_json(raw: Any, default: Any) -> Any:
    """Decode a JSON column, tolerating one that is already decoded.

    Rows arrive as JSON text from SQLite but as real objects when they have been
    through ``Repository``'s ``json_columns`` handling. Both must work.
    """
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def _num(row: dict[str, Any], key: str, cast: Any = float, default: Any = 0) -> Any:
    value = row.get(key)
    if value is None:
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


@dataclass
class AgentRecord:
    """A row from ``agents``: one agent instance, its grant, and its lifecycle.

    The columns are ``name``, ``capabilities`` and ``spawned_at`` — not the
    ``input``/``output``/``tokens`` a first pass assumed. Agents keep their work
    product in the blackboard and the task rows, not here.
    """

    id: str
    role: str = ""
    name: str = ""
    status: str = "pending"
    mission_id: str = ""
    parent_id: str = ""
    capabilities: list[str] = field(default_factory=list)
    spawned_at: float = 0.0
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if not self.spawned_at or not self.finished_at:
            return 0.0
        return max(0.0, self.finished_at - self.spawned_at)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"done", "failed", "cancelled"}

    @property
    def depth(self) -> int:
        """Nesting depth, reconstructed by counting lineage separators."""
        return self.metadata.get("depth", 0)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "AgentRecord":
        return cls(
            id=row["id"],
            role=row.get("role") or "",
            name=row.get("name") or "",
            status=row.get("status") or "pending",
            mission_id=row.get("mission_id") or "",
            parent_id=row.get("parent_id") or "",
            capabilities=decode_json(row.get("capabilities"), []),
            spawned_at=_num(row, "spawned_at"),
            finished_at=_num(row, "finished_at", float, None),
            metadata=decode_json(row.get("metadata"), {}),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "role": self.role, "name": self.name, "status": self.status,
            "mission_id": self.mission_id, "parent_id": self.parent_id,
            "capabilities": encode_json(self.capabilities),
            "spawned_at": self.spawned_at, "finished_at": self.finished_at,
            "metadata": encode_json(self.metadata),
        }


@dataclass
class TaskRecord:
    """A row from ``tasks``: one node of a persisted task DAG."""

    id: str
    name: str
    kind: str = "io"
    status: str = "pending"
    mission_id: str = ""
    parent_id: str = ""
    agent_role: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    deps: list[str] = field(default_factory=list)
    priority: int = 0
    attempts: int = 0
    max_attempts: int = 3
    result: str = ""
    error: str = ""
    tokens: int = 0
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exhausted(self) -> bool:
        """True when no retry budget remains."""
        return self.attempts >= self.max_attempts

    @property
    def is_terminal(self) -> bool:
        return self.status in {"done", "failed", "cancelled"}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TaskRecord":
        return cls(
            id=row["id"],
            name=row.get("name") or "",
            kind=row.get("kind") or "io",
            status=row.get("status") or "pending",
            mission_id=row.get("mission_id") or "",
            parent_id=row.get("parent_id") or "",
            agent_role=row.get("agent_role") or "",
            payload=decode_json(row.get("payload"), {}),
            deps=decode_json(row.get("deps"), []),
            priority=_num(row, "priority", int),
            attempts=_num(row, "attempts", int),
            max_attempts=_num(row, "max_attempts", int, 3),
            result=row.get("result") or "",
            error=row.get("error") or "",
            tokens=_num(row, "tokens", int),
            created_at=_num(row, "created_at"),
            started_at=_num(row, "started_at", float, None),
            finished_at=_num(row, "finished_at", float, None),
            metadata=decode_json(row.get("metadata"), {}),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "status": self.status,
            "mission_id": self.mission_id, "parent_id": self.parent_id,
            "agent_role": self.agent_role, "payload": encode_json(self.payload),
            "deps": encode_json(self.deps),
            "priority": self.priority, "attempts": self.attempts,
            "max_attempts": self.max_attempts, "result": self.result, "error": self.error,
            "tokens": self.tokens, "created_at": self.created_at,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "metadata": encode_json(self.metadata),
        }


@dataclass
class Reflection:
    """A row from ``reflections``: what a mission learned."""

    id: str
    score: float = 0.0
    summary: str = ""
    mission_id: str = ""
    lessons: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Reflection":
        return cls(
            id=row["id"],
            score=_num(row, "score"),
            summary=row.get("summary") or "",
            mission_id=row.get("mission_id") or "",
            lessons=decode_json(row.get("lessons"), []),
            weights=decode_json(row.get("weights"), {}),
            created_at=_num(row, "created_at"),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "score": self.score, "summary": self.summary,
            "mission_id": self.mission_id, "lessons": encode_json(self.lessons),
            "weights": encode_json(self.weights), "created_at": self.created_at,
        }


@dataclass
class ModelRow:
    """A row from ``models`` — the storage-level view of a registered model.

    Distinct from ``llm.registry.ModelRecord``, which adds catalog metadata and
    the promotion logic. This one is just the row.
    """

    id: str
    name: str
    kind: str = "foundation"
    source: str = ""
    path: str = ""
    active: bool = False
    base_model: str = ""
    params: int = 0
    context_length: int = 0
    quantization: str = ""
    license: str = ""
    sha256: str = ""
    size_bytes: int = 0
    revision: str = ""
    eval_scores: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return bool(self.path)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ModelRow":
        return cls(
            id=row["id"],
            name=row.get("name") or "",
            kind=row.get("kind") or "foundation",
            source=row.get("source") or "",
            path=row.get("path") or "",
            active=bool(row.get("active")),
            base_model=row.get("base_model") or "",
            params=_num(row, "params", int),
            context_length=_num(row, "context_length", int),
            quantization=row.get("quantization") or "",
            license=row.get("license") or "",
            sha256=row.get("sha256") or "",
            size_bytes=_num(row, "size_bytes", int),
            revision=row.get("revision") or "",
            eval_scores=decode_json(row.get("eval_scores"), {}),
            created_at=_num(row, "created_at"),
            metadata=decode_json(row.get("metadata"), {}),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "source": self.source,
            "path": self.path, "active": int(self.active), "base_model": self.base_model,
            "params": self.params, "context_length": self.context_length,
            "quantization": self.quantization, "license": self.license,
            "sha256": self.sha256, "size_bytes": self.size_bytes, "revision": self.revision,
            "eval_scores": encode_json(self.eval_scores), "created_at": self.created_at,
            "metadata": encode_json(self.metadata),
        }
