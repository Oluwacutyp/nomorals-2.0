"""``nm sceneproject`` — one directory tying scenes, takes, rigs
and assets together, live from the terminal.

Thin CLI over the project organ: the same projects the
``scene_project`` brain tool drives, from the terminal. A
project is a sealed manifest (``<name>.dvproject``) plus
``Config/DefaultProject.ini`` plus ``Content/Scenes|Takes|Rigs|
Assets/`` — no hub, no shared queue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ...media.project import Project
from ...media.project.errors import ProjectError


def _out(args: argparse.Namespace, obj: Any) -> int:
    if getattr(args, "json", False):
        print(json.dumps(obj, indent=2, default=str))
        return 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{key}: {value}")
    else:
        print(obj)
    return 0


def _fail(message: str) -> int:
    print(f"sceneproject: {message}", file=sys.stderr)
    return 2


def _cmd_sceneproject(args: argparse.Namespace, context: Any) -> int:
    """Route `nm sceneproject` subcommands."""
    from ...media.project import (
        drop_project,
        get_project,
        list_projects,
        load_project,
    )

    action = args.sceneproject_action
    name = (getattr(args, "name", "") or "").strip() or "default"
    try:
        if action == "list":
            return _out(args, {"projects": list_projects()})
        if action == "create":
            project = Project.create(
                args.project_name, args.root,
                description=(args.description or ""),
                category=(args.category or ""))
            load_project(project.manifest_path)
            return _out(args, dict(project.status(),
                                   created=True))
        if action == "open":
            return _out(args, load_project(args.path).status())
        if action == "import-uproject":
            project, receipt = Project.from_uproject(
                args.path, args.root)
            load_project(project.manifest_path)
            return _out(args, dict(project.status(),
                                   import_receipt=receipt))
        project = get_project(name)
        if action == "status":
            return _out(args, project.status())
        if action == "validate":
            problems = project.validate()
            return _out(args, {"project": name,
                               "valid": not problems,
                               "problems": problems})
        if action in ("add-scene", "add-take", "add-rig",
                      "add-asset"):
            entry = (getattr(args, "entry", "") or "").strip()
            if not entry:
                entry = Path(args.path).stem
            copy = not getattr(args, "link", False)
            if action == "add-scene":
                receipt = project.add_scene(
                    entry, args.path, copy=copy)
            elif action == "add-take":
                receipt = project.add_take(
                    entry, args.path, copy=copy)
            elif action == "add-rig":
                receipt = project.add_rig(
                    entry, args.path, copy=copy)
            else:
                receipt = project.add_asset(
                    entry, args.path,
                    kind=(getattr(args, "kind", "") or ""),
                    copy=copy)
            project.save()
            return _out(args, dict(receipt, project=name))
        if action == "remove":
            receipt = project.remove(
                args.kind, args.entry,
                delete_file=bool(
                    getattr(args, "delete_file", False)))
            project.save()
            return _out(args, dict(receipt, project=name))
        if action == "set-default":
            receipt = project.set_default_scene(args.scene)
            project.save()
            return _out(args, dict(receipt, project=name))
        if action == "open-scene":
            scene = project.open_scene(args.scene)
            return _out(args, {"project": name,
                               "scene": args.scene,
                               "title": scene.title,
                               "nodes": len(scene.nodes),
                               "tracks": len(scene.tracks)})
        if action == "take-path":
            return _out(args, {"project": name,
                               "take": args.take,
                               "path": str(
                                   project.take_path(args.take))})
        if action == "export":
            return _out(args, project.export_bundle(args.out))
        if action == "bind-show":
            from ...tools.mocap import _STORE

            return _out(args, project.bind_show(
                args.show or name, _STORE,
                take=(getattr(args, "take", "") or "")))
        if action == "drop":
            return _out(args, drop_project(name))
    except ProjectError as exc:
        return _fail(str(exc))
    return _fail(f"unknown sceneproject action {action!r}")
