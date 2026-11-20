"""The project registry — loaded projects, by name.

Mirrors the show registry (:mod:`nomorals.media.show.registry`):
one process-wide table of open projects. A project is loaded
from its ``.dvproject`` file or its directory; the name comes
from the manifest, never from the caller. Dropping a project
only forgets it — the directory on disk is untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ProjectError
from .project import Project

__all__ = [
    "drop_project",
    "get_project",
    "list_projects",
    "load_project",
    "reset_projects",
]

_PROJECTS: dict[str, Project] = {}


def load_project(path: str | Path) -> Project:
    """Open a project and register it under its manifest name.
    Re-loading a project replaces the loaded one."""
    project = Project.open(path)
    _PROJECTS[project.name] = project
    return project


def get_project(name: str = "default") -> Project:
    """The named loaded project. An unknown name is refused by
    name, listing the loaded ones."""
    name = (name or "").strip() or "default"
    project = _PROJECTS.get(name)
    if project is None:
        raise ProjectError(
            f"no loaded project {name!r} — loaded: "
            f"{', '.join(sorted(_PROJECTS)) or 'none'}",
            kind="unknown_entry")
    return project


def list_projects() -> list[dict[str, Any]]:
    """Every loaded project, as status dicts."""
    return [p.status() for p in _PROJECTS.values()]


def drop_project(name: str) -> dict[str, Any]:
    """Forget a loaded project. The directory on disk is
    untouched. An unknown name is refused by name."""
    name = (name or "").strip()
    if not name:
        raise ProjectError(
            "drop needs a project name", kind="bad_name")
    project = _PROJECTS.pop(name, None)
    if project is None:
        raise ProjectError(
            f"no loaded project {name!r}", kind="unknown_entry")
    return {"dropped": name, "root": str(project.root)}


def reset_projects() -> None:
    """Forget every loaded project (tests)."""
    _PROJECTS.clear()
