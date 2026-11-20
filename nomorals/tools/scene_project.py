"""Scene project: one directory tying scenes, takes, rigs and
assets together (StreamTwo's UE project shape, pattern only, no
upstream code).

A thin surface over :mod:`nomorals.media.project` — the organ
owns the manifest, the config, the content index, validation and
the show seam; the tool only routes arguments and shapes the
reply.

One project is one directory, direct, no hub: it holds its own
scenes, takes, rigs and assets, and a show bound to it runs the
project's content. Writes project files, so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.project import (
    Project,
    drop_project,
    get_project,
    list_projects,
    load_project,
)

__all__ = ["register"]

_ACTIONS = ("create", "list", "status", "open", "validate",
            "add_scene", "add_take", "add_rig", "add_asset",
            "remove", "set_default", "open_scene", "take_path",
            "export", "bind_show", "import_uproject", "drop")


def _params(params: dict | None) -> dict:
    return dict(params or {})


def _project(name: str) -> Project:
    return get_project(name)


def register(registry: Any) -> None:
    """Attach the scene-project family to a registry."""
    @registry.register(
        "scene_project",
        description=(
            "A Devon scene project — one directory tying scenes, "
            "takes, rigs and assets together (StreamTwo UE project "
            "shape, pattern only). A project is a sealed manifest "
            "(<name>.dvproject: modules, plugins, targets, the "
            "content index) plus Config/DefaultProject.ini plus "
            "Content/Scenes|Takes|Rigs|Assets. Actions: "
            "create=<scaffold a project, node=<name>, "
            "params=<{root, description, category}>>; "
            "list=<loaded projects>; status=<one project, "
            "node=<name>>; open=<open a .dvproject file or "
            "directory, sources=[path]>; validate=<every problem "
            "with the project on disk, node=<name>>; "
            "add_scene=<validate and add a Scene3D file, "
            "node=<name>, sources=[path], params=<{copy}>>; "
            "add_take=<validate and add a BVH/JSON take, "
            "node=<name>, sources=[path], params=<{copy}>>; "
            "add_rig=<validate and add a rig definition, "
            "node=<name>, sources=[path], params=<{copy}>>; "
            "add_asset=<record a file as an asset, node=<name>, "
            "sources=[path], params=<{kind, copy}>>; "
            "remove=<drop an entry, node=<name>, params=<{kind, "
            "entry, delete_file}>>; set_default=<name the default "
            "scene, node=<name>, params=<{scene}>>; open_scene=<"
            "load a scene entry as a Scene3D, node=<name>, "
            "params=<{scene}>>; take_path=<a take entry's path, "
            "node=<name>, params=<{take}>>; export=<copy the whole "
            "project as a sealed bundle, node=<name>, output=<dir>>; "
            "bind_show=<load a show over the project's default "
            "take, node=<name>, params=<{show, take}>>; "
            "import_uproject=<import a real UE .uproject, "
            "sources=[path.uproject], params=<{root}>>; drop=<"
            "forget a loaded project, node=<name>>. action=<"
            "create|list|status|open|validate|add_scene|add_take|"
            "add_rig|add_asset|remove|set_default|open_scene|"
            "take_path|export|bind_show|import_uproject|drop>; "
            "sources=<list of paths>; output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def scene_project_tool(action: str = "status",
                           sources: list[str] | None = None,
                           output: str = "", kind: str = "",
                           node: str = "", value: Any = None,
                           command: str = "",
                           params: dict | None = None,
                           title: str = "",
                           ) -> dict[str, Any]:
        action = (action or "status").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown scene_project action "
                             f"{action!r}; use one of "
                             f"{', '.join(_ACTIONS)}")
        args = _params(params)
        sources = list(sources or [])
        name = (node or "").strip() or "default"
        if action == "list":
            return {"projects": list_projects()}
        if action == "create":
            root = str(args.get("root") or ".")
            project = Project.create(
                name, root,
                description=str(args.get("description") or ""),
                category=str(args.get("category") or ""))
            load_project(project.manifest_path)
            return dict(project.status(), created=True)
        if action == "open":
            if not sources:
                raise ValueError(
                    "open needs sources=[path to .dvproject or dir]")
            return load_project(sources[0]).status()
        if action == "import_uproject":
            if not sources:
                raise ValueError(
                    "import_uproject needs sources=[path.uproject]")
            root = str(args.get("root") or ".")
            project, receipt = Project.from_uproject(
                sources[0], root)
            load_project(project.manifest_path)
            return dict(project.status(), import_receipt=receipt)
        project = _project(name)
        if action == "status":
            return project.status()
        if action == "validate":
            problems = project.validate()
            return {"project": name, "valid": not problems,
                    "problems": problems}
        if action in ("add_scene", "add_take", "add_rig",
                      "add_asset"):
            if not sources:
                raise ValueError(
                    f"{action} needs sources=[path]")
            copy = bool(args.get("copy", True))
            if action == "add_scene":
                receipt = project.add_scene(
                    str(args.get("entry") or
                        Path(sources[0]).stem),
                    sources[0], copy=copy)
            elif action == "add_take":
                receipt = project.add_take(
                    str(args.get("entry") or
                        Path(sources[0]).stem),
                    sources[0], copy=copy)
            elif action == "add_rig":
                receipt = project.add_rig(
                    str(args.get("entry") or
                        Path(sources[0]).stem),
                    sources[0], copy=copy)
            else:
                receipt = project.add_asset(
                    str(args.get("entry") or
                        Path(sources[0]).stem),
                    sources[0],
                    kind=str(args.get("kind") or ""), copy=copy)
            project.save()
            return dict(receipt, project=name)
        if action == "remove":
            entry_kind = str(args.get("kind") or "")
            entry = str(args.get("entry") or "")
            if not entry_kind or not entry:
                raise ValueError(
                    "remove needs params={kind, entry}")
            receipt = project.remove(
                entry_kind, entry,
                delete_file=bool(args.get("delete_file", False)))
            project.save()
            return dict(receipt, project=name)
        if action == "set_default":
            scene = str(args.get("scene") or "")
            if not scene:
                raise ValueError(
                    "set_default needs params={scene}")
            receipt = project.set_default_scene(scene)
            project.save()
            return dict(receipt, project=name)
        if action == "open_scene":
            scene = str(args.get("scene") or "")
            if not scene:
                raise ValueError(
                    "open_scene needs params={scene}")
            loaded = project.open_scene(scene)
            return {"project": name, "scene": scene,
                    "title": loaded.title,
                    "nodes": len(loaded.nodes),
                    "tracks": len(loaded.tracks)}
        if action == "take_path":
            take = str(args.get("take") or "")
            if not take:
                raise ValueError(
                    "take_path needs params={take}")
            return {"project": name, "take": take,
                    "path": str(project.take_path(take))}
        if action == "export":
            if not output:
                raise ValueError("export needs output=<dir>")
            return project.export_bundle(output)
        if action == "bind_show":
            from .mocap import _STORE

            show = str(args.get("show") or name)
            take = str(args.get("take") or "")
            return project.bind_show(show, _STORE, take=take)
        if action == "drop":
            return drop_project(name)
        raise ValueError(f"unhandled scene_project action "
                         f"{action!r}")
