"""First-class OS projects.

Relationship to :mod:`nomorals.agents.projects`
------------------------------------------------
:mod:`agents.projects` owns the *behavioral* project system: ``Project``
with plans/steps/budgets and a ``ProjectManager`` that autonomously runs
multi-step work.  This module's ``os.Project`` is the *scoping shell* for
the control plane: an id, a name, a description, the mission ids that
belong to it, and a lifecycle state.  It deliberately duplicates none of
``ProjectManager``'s planning/execution logic — when behavior overlaps
(plan, run, report), reuse ``agents.projects.ProjectManager``; use
``os.ProjectStore`` when you need a durable, frontend-visible project
container that sessions and artifacts scope against.

:func:`artifact_scope` composes the *existing*
:meth:`nomorals.storage.artifacts.ArtifactStore.for_mission` per mission in
the project.  It takes the store as a parameter on purpose: no methods are
added to ``ArtifactStore`` (that file belongs to a sibling H2 worker), and
no import of it is needed here — duck typing keeps this module decoupled.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..storage.db import Database

_log = logging.getLogger(__name__)

__all__ = ["Project", "ProjectStore", "artifact_scope"]

_OS_PROJECTS_DDL = """
CREATE TABLE IF NOT EXISTS os_projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    mission_ids TEXT NOT NULL DEFAULT '[]',
    state       TEXT NOT NULL DEFAULT 'active',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_os_projects_state ON os_projects(state);
"""


@dataclass
class Project:
    """A durable scoping shell: name + the missions that belong to it."""

    id: str
    name: str
    description: str = ""
    mission_ids: list[str] = field(default_factory=list)
    state: str = "active"  # active | paused | archived
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "mission_ids": list(self.mission_ids),
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            mission_ids=list(data.get("mission_ids", []) or []),
            state=str(data.get("state", "active")),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )

    @classmethod
    def _from_row(cls, row: dict[str, Any]) -> "Project":
        try:
            mission_ids = list(json.loads(row.get("mission_ids") or "[]") or [])
        except (TypeError, ValueError):
            mission_ids = []
        return cls(
            id=row["id"],
            name=row.get("name", "") or "",
            description=row.get("description", "") or "",
            mission_ids=mission_ids,
            state=row.get("state", "active") or "active",
            created_at=float(row.get("created_at", time.time())),
            updated_at=float(row.get("updated_at", time.time())),
        )


class ProjectStore:
    """Create / fetch / list / scope OS projects, persisted in SQLite."""

    def __init__(self, db: Database | str | Path | None = None) -> None:
        if isinstance(db, Database):
            self.db = db
        else:
            self.db = Database(":memory:" if db is None else str(db))
        self.db.executescript(_OS_PROJECTS_DDL)
        self._lock = threading.RLock()

    def create(self, name: str, *, description: str = "",
               mission_ids: list[str] | None = None,
               state: str = "active") -> Project:
        project = Project(
            id=new_short_id("proj_"),
            name=name,
            description=description or "",
            mission_ids=list(mission_ids or []),
            state=state or "active",
        )
        with self._lock:
            self.db.execute(
                "INSERT INTO os_projects (id, name, description, mission_ids,"
                " state, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (project.id, project.name, project.description,
                 json.dumps(project.mission_ids), project.state,
                 project.created_at, project.updated_at),
            )
        _log.info("os project %s created (%r)", project.id, project.name)
        return project

    def get(self, project_id: str) -> Project | None:
        rows = self.db.query("SELECT * FROM os_projects WHERE id = ?",
                             (project_id,))
        return Project._from_row(rows[0]) if rows else None

    def list(self, *, state: str | None = None) -> list[Project]:
        if state:
            rows = self.db.query(
                "SELECT * FROM os_projects WHERE state = ? ORDER BY created_at",
                (state,))
        else:
            rows = self.db.query(
                "SELECT * FROM os_projects ORDER BY created_at")
        return [Project._from_row(r) for r in rows]

    def _save(self, project: Project) -> None:
        project.updated_at = time.time()
        with self._lock:
            self.db.execute(
                "UPDATE os_projects SET name = ?, description = ?,"
                " mission_ids = ?, state = ?, updated_at = ? WHERE id = ?",
                (project.name, project.description,
                 json.dumps(project.mission_ids), project.state,
                 project.updated_at, project.id),
            )

    def add_mission(self, project_id: str, mission_id: str) -> bool:
        """Attach a mission to the project (idempotent)."""
        project = self.get(project_id)
        if project is None:
            return False
        if mission_id not in project.mission_ids:
            project.mission_ids.append(mission_id)
            self._save(project)
        return True

    def remove_mission(self, project_id: str, mission_id: str) -> bool:
        """Detach a mission from the project. False when project unknown."""
        project = self.get(project_id)
        if project is None:
            return False
        if mission_id in project.mission_ids:
            project.mission_ids.remove(mission_id)
            self._save(project)
        return True

    def set_state(self, project_id: str, state: str) -> bool:
        project = self.get(project_id)
        if project is None:
            return False
        project.state = state
        self._save(project)
        return True


def artifact_scope(artifact_store: Any, project: Project) -> list[Any]:
    """All artifacts across the project's missions, deduplicated by id.

    ``artifact_store`` is the existing
    :class:`nomorals.storage.artifacts.ArtifactStore` (or anything with a
    ``for_mission(mission_id)`` method) — passed in, never imported, so this
    helper adds no methods to ``ArtifactStore`` and no coupling to it.
    """
    artifacts: list[Any] = []
    seen: set[str] = set()
    for_mission = getattr(artifact_store, "for_mission", None)
    if not callable(for_mission):
        _log.warning("artifact_scope: store has no for_mission(); returning []")
        return []
    for mission_id in project.mission_ids or []:
        try:
            found = for_mission(mission_id) or []
        except Exception:  # noqa: BLE001 — one bad mission must not kill the scope
            _log.exception("artifact_scope: for_mission(%r) failed", mission_id)
            continue
        for artifact in found:
            key = str(getattr(artifact, "id", None) or id(artifact))
            if key not in seen:
                seen.add(key)
                artifacts.append(artifact)
    return artifacts
