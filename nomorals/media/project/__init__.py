"""Devon's project — one directory tying scenes, takes, rigs and
assets together (StreamTwo's UE project shape, pattern only, no
upstream code).

The study (storytold/StreamTwo): a UE 4.25 project — a
``.uproject`` manifest (FileVersion, EngineAssociation, one
Runtime module, a plugin list), ``Source/`` module rules and
targets (``.Build.cs``, ``.Target.cs``, ``Editor.Target.cs``),
``Config/`` (DefaultEngine/Game/Editor/Input.ini) and ``Content/``
maps and assets. The patterns taken are the shape itself: a
sealed manifest naming modules / plugins / targets, a config
directory for runtime settings, a content tree for the assets,
and a source skeleton.

Devon's project is the same shape in Python, over the organs
that already hold its content: Scene3D documents, BVH/JSON
takes, rig definitions, and the show that runs them. A project
is a directory: ``<name>.dvproject`` (sealed manifest),
``Config/DefaultProject.ini``, ``Content/Scenes|Takes|Rigs|
Assets/``, ``Source/``. One project is one directory, direct —
no hub, no shared queue.
"""

from __future__ import annotations

from .config import CONFIG_FILENAME, ProjectConfig
from .errors import ProjectError
from .manifest import (
    KNOWN_MODULES,
    KNOWN_PLUGINS,
    ProjectManifest,
    default_modules,
    default_plugins,
    default_targets,
    import_uproject,
)
from .project import ASSET_KINDS, CONTENT_DIRS, Project
from .registry import (
    drop_project,
    get_project,
    list_projects,
    load_project,
    reset_projects,
)

__all__ = [
    "ASSET_KINDS",
    "CONFIG_FILENAME",
    "CONTENT_DIRS",
    "KNOWN_MODULES",
    "KNOWN_PLUGINS",
    "Project",
    "ProjectConfig",
    "ProjectError",
    "ProjectManifest",
    "default_modules",
    "default_plugins",
    "default_targets",
    "drop_project",
    "get_project",
    "import_uproject",
    "list_projects",
    "load_project",
    "reset_projects",
]
