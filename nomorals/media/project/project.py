"""A Devon project — one directory that ties scenes, takes, rigs
and assets together.

The shape is StreamTwo's UE project shape, pattern only, over
Devon's own organs:

* ``<name>.dvproject`` — the manifest (:mod:`manifest`), the
  ``.uproject`` analogue: modules, plugins, targets, the content
  index, sealed with a sha256.
* ``Config/DefaultProject.ini`` — runtime settings (:mod:`config`),
  the ``Config/*.ini`` analogue.
* ``Content/Scenes/`` — Scene3D documents (``media.edit_engine``).
* ``Content/Takes/`` — BVH/JSON takes (``media.rig`` /
  ``media.show``).
* ``Content/Rigs/`` — rig definitions (``media.rig``).
* ``Content/Assets/`` — anything else, recorded with a kind.
* ``Source/`` — the module skeleton, the ``Source/`` analogue
  (the organs themselves live in the repo; the directory records
  the shape).

One project is one directory, direct — no hub, no shared queue.
Entries record copy-vs-link: a copy lives inside the project
(path relative to the root); a link points at an outside file
(absolute path). A scene entry is validated by actually loading
it as a Scene3D; a take by actually parsing it; a rig by
actually building its RigDefinition. Nothing is recorded on
trust.

A project can hand its default scene to the caller as a loaded
Scene3D, and can bind a show (:mod:`media.show`) to its default
take — the show runs the project's content, the project does not
own the show.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ...version import __version__ as _DEVON_VERSION
from ..edit_engine.scene3d import Scene3D, Scene3DError
from ..rig import RigDefinition
from ..rig.errors import RigError
from ..rig.session import load_take_bvh, load_take_json
from .config import CONFIG_FILENAME, ProjectConfig, read_config, write_config
from .errors import ProjectError
from .manifest import (
    ProjectManifest,
    import_uproject,
    validate_entry_name,
)

__all__ = [
    "ASSET_KINDS",
    "CONTENT_DIRS",
    "Project",
]

#: Content subdirectories, the ``Content/`` analogue.
CONTENT_DIRS: dict[str, str] = {
    "scenes": "Content/Scenes",
    "takes": "Content/Takes",
    "rigs": "Content/Rigs",
    "assets": "Content/Assets",
}

#: Asset kinds, by file suffix. Anything else is kind ``file`` —
#: recorded, not inspected, and said so.
ASSET_KINDS: dict[str, str] = {
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".webp": "image", ".gif": "image",
    ".mp3": "audio", ".wav": "audio", ".ogg": "audio",
    ".mp4": "video", ".mov": "video",
    ".json": "data", ".bvh": "take", ".txt": "text",
}


def _asset_kind(path: Path) -> str:
    return ASSET_KINDS.get(path.suffix.lower(), "file")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Project:
    """One Devon project on disk."""

    def __init__(self, root: str | Path,
                 manifest: ProjectManifest,
                 config: ProjectConfig | None = None):
        self.root = Path(root).expanduser().resolve()
        self.manifest = manifest
        self.config = config or ProjectConfig()
        self._manifest_path = self.root / f"{manifest.name}.dvproject"

    # -- identity ----------------------------------------------------

    @property
    def name(self) -> str:
        """The project name."""
        return self.manifest.name

    @property
    def manifest_path(self) -> Path:
        """The ``.dvproject`` file."""
        return self._manifest_path

    @property
    def entries(self) -> dict[str, dict[str, dict[str, Any]]]:
        """The content index, by kind then entry name."""
        return self.manifest.entries

    # -- creation / opening ------------------------------------------

    @classmethod
    def create(cls, name: str, root: str | Path, *,
               description: str = "", category: str = "",
               modules: list[dict[str, Any]] | None = None,
               plugins: list[dict[str, Any]] | None = None,
               ) -> Project:
        """Scaffold a project: the directory, the manifest, the
        config, the Content tree and the Source skeleton. A root
        that already holds a project of this name is refused."""
        name = validate_entry_name(name, "project")
        root = Path(root).expanduser().resolve()
        target = root / name
        manifest_path = target / f"{name}.dvproject"
        if manifest_path.exists():
            raise ProjectError(
                f"a project named {name!r} already exists at "
                f"{target}",
                kind="already_exists")
        manifest = ProjectManifest(
            name=name, description=description, category=category,
            devon_version=_DEVON_VERSION,
            modules=modules, plugins=plugins)
        project = cls(target, manifest)
        for sub in CONTENT_DIRS.values():
            (target / sub).mkdir(parents=True, exist_ok=True)
        (target / "Source").mkdir(parents=True, exist_ok=True)
        project.save()
        return project

    @classmethod
    def open(cls, path: str | Path) -> Project:
        """Open a project from its ``.dvproject`` file or its
        directory. A bad seal, a missing config or a missing entry
        file is refused by name."""
        target = Path(path).expanduser().resolve()
        if target.is_dir():
            found = sorted(target.glob("*.dvproject"))
            if not found:
                raise ProjectError(
                    f"{target} holds no .dvproject file",
                    kind="not_a_project")
            if len(found) > 1:
                raise ProjectError(
                    f"{target} holds more than one .dvproject "
                    f"file: {', '.join(p.name for p in found)}",
                    kind="not_a_project")
            target = found[0]
        if not target.exists():
            raise ProjectError(
                f"no project file at {target}",
                kind="not_a_project")
        if target.suffix != ".dvproject":
            raise ProjectError(
                f"{target} is not a .dvproject file",
                kind="not_a_project")
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ProjectError(
                f"project manifest at {target} is unreadable: "
                f"{exc}",
                kind="bad_manifest") from exc
        manifest = ProjectManifest.from_dict(data)
        config_path = target.parent / "Config" / CONFIG_FILENAME
        config = read_config(config_path)
        project = cls(target.parent, manifest, config=config)
        problems = project.validate()
        if problems:
            raise ProjectError(
                "project " + manifest.name + " failed validation: "
                + "; ".join(problems),
                kind="bad_manifest")
        return project

    @classmethod
    def from_uproject(cls, path: str | Path, root: str | Path,
                      ) -> tuple[Project, dict[str, Any]]:
        """Import a real UE ``.uproject`` as a Devon project: the
        manifest is mapped (:func:`manifest.import_uproject`), the
        directory is scaffolded, and the receipt names the mapped
        modules and plugins and the unmapped ones. The UE file
        itself is read, never modified."""
        target = Path(path).expanduser().resolve()
        if not target.exists():
            raise ProjectError(
                f"no .uproject file at {target}",
                kind="missing_file")
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ProjectError(
                f".uproject at {target} is unreadable: {exc}",
                kind="bad_manifest") from exc
        manifest, receipt = import_uproject(data)
        root = Path(root).expanduser().resolve()
        dest = root / manifest.name
        manifest_path = dest / f"{manifest.name}.dvproject"
        if manifest_path.exists():
            raise ProjectError(
                f"a project named {manifest.name!r} already "
                f"exists at {dest}",
                kind="already_exists")
        manifest.devon_version = _DEVON_VERSION
        project = cls(dest, manifest)
        for sub in CONTENT_DIRS.values():
            (dest / sub).mkdir(parents=True, exist_ok=True)
        (dest / "Source").mkdir(parents=True, exist_ok=True)
        project.save()
        return project, receipt

    # -- persistence ---------------------------------------------------

    def save(self) -> str:
        """Write the manifest (re-sealed) and the config, atomically.
        Returns the manifest path."""
        self.root.mkdir(parents=True, exist_ok=True)
        data = self.manifest.sealed_dict()
        tmp = self._manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2),
                       encoding="utf-8")
        tmp.replace(self._manifest_path)
        write_config(self.root / "Config" / CONFIG_FILENAME,
                     self.config)
        return str(self._manifest_path)

    # -- entries ---------------------------------------------------------

    def _entry_path(self, kind: str, name: str) -> Path:
        try:
            rec = self.manifest.entries[kind][name]
        except KeyError:
            raise ProjectError(
                f"project {self.name!r} has no {kind[:-1]} "
                f"entry {name!r}",
                kind="unknown_entry") from None
        path = Path(rec["path"])
        if rec["kind"] == "link":
            return path
        return self.root / path

    def _add(self, kind: str, name: str, path: str | Path, *,
             copy: bool) -> dict[str, Any]:
        name = validate_entry_name(name, f"{kind[:-1]} entry")
        if name in self.manifest.entries[kind]:
            raise ProjectError(
                f"project {self.name!r} already has a "
                f"{kind[:-1]} entry {name!r}",
                kind="duplicate_entry")
        source = Path(path).expanduser()
        if not source.exists():
            raise ProjectError(
                f"no file at {source} to add as {kind[:-1]} "
                f"{name!r}",
                kind="missing_file")
        if copy:
            dest_dir = self.root / CONTENT_DIRS[kind]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / source.name
            if dest.exists() and dest.resolve() != source.resolve():
                raise ProjectError(
                    f"Content/{CONTENT_DIRS[kind].split('/')[-1]}/"
                    f"{source.name} already holds a different "
                    f"file — rename the source first",
                    kind="duplicate_entry")
            if dest.resolve() != source.resolve():
                shutil.copyfile(source, dest)
            record: dict[str, Any] = {
                "path": f"{CONTENT_DIRS[kind]}/{source.name}",
                "kind": "copy",
            }
        else:
            record = {"path": str(source.resolve()), "kind": "link"}
        if kind == "assets":
            record["asset_kind"] = _asset_kind(source)
        self.manifest.entries[kind][name] = record
        return {"entry": name, "kind": kind[:-1],
                "path": record["path"],
                "stored": record["kind"]}

    def add_scene(self, name: str, path: str | Path, *,
                  copy: bool = True) -> dict[str, Any]:
        """Add a scene entry. The file is actually loaded as a
        Scene3D — a file that does not parse as a scene is refused
        naming the scene error."""
        source = Path(path).expanduser()
        try:
            Scene3D.load(source)
        except Scene3DError as exc:
            raise ProjectError(
                f"{source} is not a loadable scene3d: {exc}",
                kind="bad_scene") from exc
        return self._add("scenes", name, source, copy=copy)

    def add_take(self, name: str, path: str | Path, *,
                 copy: bool = True) -> dict[str, Any]:
        """Add a take entry. A ``.bvh`` is actually parsed as a
        BVH take; a ``.json`` is actually parsed as a JSON take.
        Anything else, or a file that does not parse, is refused
        by name."""
        source = Path(path).expanduser()
        suffix = source.suffix.lower()
        try:
            if suffix == ".bvh":
                load_take_bvh(source)
            elif suffix == ".json":
                load_take_json(source)
            else:
                raise ProjectError(
                    f"take {source} must be .bvh or .json, got "
                    f"{suffix!r}",
                    kind="bad_take")
        except ProjectError:
            raise
        except RigError as exc:
            raise ProjectError(
                f"take {source} does not parse: {exc}",
                kind="bad_take") from exc
        except Exception as exc:  # noqa: BLE001 - parse errors vary
            raise ProjectError(
                f"take {source} does not parse: {exc}",
                kind="bad_take") from exc
        return self._add("takes", name, source, copy=copy)

    def add_rig(self, name: str, path: str | Path, *,
                copy: bool = True) -> dict[str, Any]:
        """Add a rig entry. The file is actually parsed as a rig
        definition — a file without a name or bones is refused
        naming the rig error."""
        source = Path(path).expanduser()
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ProjectError(
                f"rig {source} is not readable JSON: {exc}",
                kind="bad_rig") from exc
        try:
            RigDefinition.from_dict(data)
        except RigError as exc:
            raise ProjectError(
                f"rig {source} is not a valid rig definition: "
                f"{exc}",
                kind="bad_rig") from exc
        return self._add("rigs", name, source, copy=copy)

    def add_asset(self, name: str, path: str | Path, *,
                  kind: str = "", copy: bool = True) -> dict[str, Any]:
        """Add an asset entry. The file must exist; its kind comes
        from the suffix unless named. An asset is recorded, not
        inspected — and said so."""
        receipt = self._add("assets", name, path, copy=copy)
        if kind:
            self.manifest.entries["assets"][name]["asset_kind"] = kind
            receipt["asset_kind"] = kind
        else:
            receipt["asset_kind"] = self.manifest.entries[
                "assets"][name]["asset_kind"]
        return receipt

    def remove(self, kind: str, name: str, *,
               delete_file: bool = False) -> dict[str, Any]:
        """Remove an entry. The file is left on disk unless
        ``delete_file`` — and only a copied file is ever deleted,
        never a link target."""
        kind = (kind or "").strip().lower()
        if kind not in CONTENT_DIRS:
            raise ProjectError(
                f"unknown entry kind {kind!r} — scenes, takes, "
                "rigs, assets",
                kind="bad_entry")
        name = validate_entry_name(name, f"{kind[:-1]} entry")
        try:
            rec = self.manifest.entries[kind].pop(name)
        except KeyError:
            raise ProjectError(
                f"project {self.name!r} has no {kind[:-1]} "
                f"entry {name!r}",
                kind="unknown_entry") from None
        deleted: str | None = None
        if delete_file and rec["kind"] == "copy":
            target = self.root / rec["path"]
            if target.exists():
                target.unlink()
                deleted = str(target)
        if self.manifest.default_scene == name and kind == "scenes":
            self.manifest.default_scene = ""
        return {"removed": name, "kind": kind[:-1],
                "deleted_file": deleted}

    def set_default_scene(self, name: str) -> dict[str, Any]:
        """Name the default scene — the startup map, the
        ``GameDefaultMap`` analogue. An unknown scene is refused
        by name."""
        name = validate_entry_name(name, "default scene")
        if name not in self.manifest.entries["scenes"]:
            raise ProjectError(
                f"default scene {name!r} is not one of the "
                f"project's scenes: "
                f"{', '.join(sorted(self.manifest.entries['scenes'])) or 'none'}",
                kind="unknown_entry")
        self.manifest.default_scene = name
        return {"default_scene": name,
                "path": self.manifest.entries["scenes"][name]["path"]}

    # -- reads -------------------------------------------------------------

    def open_scene(self, name: str) -> Scene3D:
        """The named scene, loaded as a Scene3D."""
        name = validate_entry_name(name, "scene entry")
        return Scene3D.load(self._entry_path("scenes", name))

    def take_path(self, name: str) -> Path:
        """The filesystem path of a take entry."""
        name = validate_entry_name(name, "take entry")
        return self._entry_path("takes", name)

    def rig_path(self, name: str) -> Path:
        """The filesystem path of a rig entry."""
        name = validate_entry_name(name, "rig entry")
        return self._entry_path("rigs", name)

    def asset_path(self, name: str) -> Path:
        """The filesystem path of an asset entry."""
        name = validate_entry_name(name, "asset entry")
        return self._entry_path("assets", name)

    def status(self) -> dict[str, Any]:
        """The project as a status dict: identity, modules,
        plugins, targets, entry counts, default scene, seal."""
        return {
            "name": self.manifest.name,
            "project_id": self.manifest.project_id,
            "devon_version": self.manifest.devon_version,
            "source_engine": self.manifest.source_engine,
            "description": self.manifest.description,
            "category": self.manifest.category,
            "root": str(self.root),
            "modules": [m["name"] for m in self.manifest.modules],
            "plugins": [p["name"] for p in self.manifest.plugins
                        if p["enabled"]],
            "plugins_disabled": [p["name"]
                                 for p in self.manifest.plugins
                                 if not p["enabled"]],
            "unmapped_plugins": list(
                self.manifest.unmapped_plugins),
            "targets": [t["name"] for t in self.manifest.targets],
            "scenes": sorted(self.manifest.entries["scenes"]),
            "takes": sorted(self.manifest.entries["takes"]),
            "rigs": sorted(self.manifest.entries["rigs"]),
            "assets": sorted(self.manifest.entries["assets"]),
            "default_scene": self.manifest.default_scene,
            "render": dict(self.config.render),
            "seal": self.manifest.seal(),
        }

    # -- validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Every problem with the project on disk, named. An empty
        list means the project is whole: seal matches, entries
        exist, modules/plugins/targets are known, the default
        scene is one of the scenes."""
        problems: list[str] = []
        if not (self.root / "Config" / CONFIG_FILENAME).exists():
            problems.append(
                f"missing config {CONFIG_FILENAME}")
        if not (self.root / "Source").is_dir():
            problems.append("missing Source/ directory")
        for sub in CONTENT_DIRS.values():
            if not (self.root / sub).is_dir():
                problems.append(f"missing {sub}/ directory")
        if self.manifest.default_scene and \
                self.manifest.default_scene not in \
                self.manifest.entries["scenes"]:
            problems.append(
                f"default scene "
                f"{self.manifest.default_scene!r} is not one of "
                "the project's scenes")
        for kind in ("scenes", "takes", "rigs", "assets"):
            for name, rec in sorted(
                    self.manifest.entries[kind].items()):
                path = (self.root / rec["path"]
                        if rec["kind"] == "copy"
                        else Path(rec["path"]))
                if not path.exists():
                    problems.append(
                        f"{kind[:-1]} entry {name!r} file missing: "
                        f"{rec['path']}")
        return problems

    # -- show seam -------------------------------------------------------------

    def bind_show(self, show_name: str, store: Any, *,
                  take: str = "") -> dict[str, Any]:
        """Bind a live show to this project: load (or replace) the
        named show over the shared mocap store, load the project's
        default take into it (or the named take), and report the
        show's status with the project's default scene. The show
        is the show organ's — this project only supplies its
        content."""
        from ..show import load_show

        take_name = (take or "").strip()
        if take_name:
            take_path = self.take_path(take_name)
        else:
            takes = sorted(self.manifest.entries["takes"])
            if not takes:
                raise ProjectError(
                    f"project {self.name!r} has no takes to bind "
                    "a show to",
                    kind="unknown_entry")
            take_name = takes[0]
            take_path = self.take_path(take_name)
        show = load_show(show_name, store)
        receipt = show.load_take(take_path)
        status = show.status()
        status["project"] = self.name
        status["project_take"] = take_name
        status["project_take_path"] = str(take_path)
        status["project_default_scene"] = \
            self.manifest.default_scene
        status["take_receipt"] = receipt
        return status

    # -- export ------------------------------------------------------------------

    def export_bundle(self, out_dir: str | Path) -> dict[str, Any]:
        """Export the whole project as a sealed bundle: the tree
        copied to ``out_dir`` plus a bundle manifest with a sha256
        per file. A project that fails validation is refused
        first, naming the problems."""
        problems = self.validate()
        if problems:
            raise ProjectError(
                "project " + self.name + " failed validation: "
                + "; ".join(problems),
                kind="bad_manifest")
        out = Path(out_dir).expanduser().resolve()
        if out.exists():
            raise ProjectError(
                f"export target {out} already exists",
                kind="already_exists")
        shutil.copytree(self.root, out)
        files: dict[str, str] = {}
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.name != "bundle.json":
                files[str(path.relative_to(out))] = _sha256(path)
        bundle = {
            "format": "devon-project-bundle",
            "format_version": 1,
            "project": self.name,
            "project_id": self.manifest.project_id,
            "files": files,
        }
        (out / "bundle.json").write_text(
            json.dumps(bundle, indent=2), encoding="utf-8")
        return {"project": self.name, "out": str(out),
                "files": len(files), "bundle": "bundle.json"}
