"""The project manifest — the ``.uproject`` analogue.

StreamTwo's ``StreamTwo.uproject`` is the shape: a manifest JSON
with a file version, an engine association, a category and
description, a ``Modules`` list (name / type / loading phase) and
a ``Plugins`` list (name / enabled / marketplace URL). Devon's
manifest is the same shape over Devon's own organs:

* ``format_version`` — the manifest format, 1.
* ``devon_version`` — the Devon runtime the project was created
  against (the engine-association analogue; from
  :mod:`nomorals.version`).
* ``project_id`` — a 32-hex id, the ``ProjectID`` analogue from
  StreamTwo's ``DefaultGame.ini``.
* ``modules`` — the organs the project loads: scene, rig, show,
  mocap, cloudcam, spawn. A module declares its type
  (``runtime`` / ``editor``), its loading phase (``default`` /
  ``pre_default`` / ``post_default``) and its ``depends_on`` —
  other modules it needs. Unknown modules, unknown dependencies
  and dependency cycles are refused by name.
* ``plugins`` — the integrations the project enables: the Devon
  analogues of StreamTwo's plugin set (vocodes, websocket,
  audio_import, files_downloader, media_player). Unknown plugins
  are refused by name.
* ``targets`` — the build targets, the ``.Target.cs`` analogue:
  ``game`` (runtime) and ``editor`` (authoring), each naming the
  modules it loads.
* ``entries`` — the project's content index: scenes, takes, rigs
  and assets, each an entry name → {path, kind} record. A copy
  lives inside the project (path relative to the project root);
  a link points at an outside file (absolute path).
* ``unmapped_plugins`` — plugin names from a ``.uproject`` import
  with no Devon analogue (``EventCenterPlugin``). Recorded
  disabled and named, never silently dropped.
* ``seal`` — sha256 over the canonical manifest (everything but
  the seal itself), the Scene3D seal idiom. A manifest whose seal
  does not match is refused, never silently trusted.

Nothing here compiles a ``.uproject`` — the manifest is Devon's
own format. :func:`import_uproject` reads a real UE ``.uproject``
and maps it onto this shape, naming what it cannot map.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from .errors import ProjectError

__all__ = [
    "KNOWN_MODULES",
    "KNOWN_PLUGINS",
    "LOADING_PHASES",
    "MODULE_TYPES",
    "TARGET_TYPES",
    "ProjectManifest",
    "default_modules",
    "default_plugins",
    "default_targets",
    "import_uproject",
    "validate_entry_name",
]

#: The organs a project may load. ``spawn`` is the B.19 actor
#: spawner; the rest are the organs this programme built around it.
KNOWN_MODULES: frozenset[str] = frozenset({
    "scene", "rig", "show", "mocap", "cloudcam", "spawn",
})

#: The integrations a project may enable — the Devon analogues of
#: StreamTwo's plugin set: VocodesPlugin → vocodes,
#: BlueprintWebSocket → websocket, RuntimeAudioImporter →
#: audio_import, RuntimeFilesDownloader → files_downloader,
#: MediaPlayerPlugin → media_player.
KNOWN_PLUGINS: frozenset[str] = frozenset({
    "vocodes", "websocket", "audio_import", "files_downloader",
    "media_player",
})

MODULE_TYPES: frozenset[str] = frozenset({"runtime", "editor"})
LOADING_PHASES: frozenset[str] = frozenset({
    "default", "pre_default", "post_default",
})
TARGET_TYPES: frozenset[str] = frozenset({"runtime", "editor"})

#: What each module needs loaded before it. ``show`` drives the
#: rig and the cloudcam through the mocap store; ``spawn`` acts
#: on a scene; the rest stand alone.
_MODULE_DEPS: dict[str, tuple[str, ...]] = {
    "scene": (),
    "rig": ("mocap",),
    "show": ("rig", "cloudcam", "mocap"),
    "mocap": (),
    "cloudcam": ("mocap",),
    "spawn": ("scene",),
}

#: A UE ``.uproject`` plugin name → the Devon plugin it maps to.
#: ``EventCenterPlugin`` has no Devon analogue — the project
#: shape is per-organ and direct by the owner's law, so it is
#: recorded as unmapped, never silently dropped.
UPROJECT_PLUGIN_MAP: dict[str, str] = {
    "VocodesPlugin": "vocodes",
    "BlueprintWebSocket": "websocket",
    "RuntimeAudioImporter": "audio_import",
    "RuntimeFilesDownloader": "files_downloader",
    "MediaPlayerPlugin": "media_player",
}

#: UE ``.uproject`` plugin names with no Devon analogue. An
#: imported project records them as unmapped plugins, disabled,
#: named in the import receipt.
UPROJECT_UNMAPPED_PLUGINS: frozenset[str] = frozenset({
    "EventCenterPlugin",
})

_ENTRY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_entry_name(name: str, what: str = "entry") -> str:
    """An entry name: alphanumerics, ``_`` and ``-``, max 64 chars,
    starting alphanumerical. Anything else is refused by name."""
    name = (name or "").strip()
    if not _ENTRY_NAME_RE.match(name):
        raise ProjectError(
            f"{what} name {name!r} is not a valid entry name "
            "(letters, digits, _ and -, max 64 chars, starting "
            "alphanumerical)",
            kind="bad_name")
    return name


def default_modules() -> list[dict[str, Any]]:
    """The six organs, each with its real dependencies."""
    return [{
        "name": name,
        "type": "runtime",
        "loading_phase": "default",
        "depends_on": list(_MODULE_DEPS[name]),
    } for name in ("scene", "rig", "show", "mocap", "cloudcam",
                   "spawn")]


def default_plugins() -> list[dict[str, Any]]:
    """StreamTwo's plugin set, Devon-side — media_player disabled,
    mirroring StreamTwo's ``MediaPlayerPlugin`` disabled."""
    return [
        {"name": "vocodes", "enabled": True, "source": "builtin"},
        {"name": "websocket", "enabled": True, "source": "builtin"},
        {"name": "audio_import", "enabled": True,
         "source": "builtin"},
        {"name": "files_downloader", "enabled": True,
         "source": "builtin"},
        {"name": "media_player", "enabled": False,
         "source": "builtin"},
    ]


def default_targets() -> list[dict[str, Any]]:
    """``game`` (runtime) and ``editor`` (authoring) — the
    ``.Target.cs`` analogue. Both load every module."""
    names = [m["name"] for m in default_modules()]
    return [
        {"name": "game", "type": "runtime", "modules": list(names)},
        {"name": "editor", "type": "editor", "modules": list(names)},
    ]


def _check_modules(modules: Any) -> list[dict[str, Any]]:
    if not isinstance(modules, list) or not modules:
        raise ProjectError(
            "a project needs a non-empty 'modules' list",
            kind="bad_manifest")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for raw in modules:
        if not isinstance(raw, dict):
            raise ProjectError(
                f"module {raw!r} is not an object",
                kind="bad_manifest")
        name = raw.get("name")
        if name not in KNOWN_MODULES:
            raise ProjectError(
                f"unknown module {name!r} — known modules: "
                f"{', '.join(sorted(KNOWN_MODULES))}",
                kind="unknown_module")
        if name in seen:
            raise ProjectError(
                f"duplicate module {name!r}", kind="bad_manifest")
        seen.add(name)
        mtype = raw.get("type", "runtime")
        if mtype not in MODULE_TYPES:
            raise ProjectError(
                f"module {name!r} type must be one of "
                f"{', '.join(sorted(MODULE_TYPES))}, got {mtype!r}",
                kind="bad_manifest")
        phase = raw.get("loading_phase", "default")
        if phase not in LOADING_PHASES:
            raise ProjectError(
                f"module {name!r} loading_phase must be one of "
                f"{', '.join(sorted(LOADING_PHASES))}, got {phase!r}",
                kind="bad_manifest")
        deps = raw.get("depends_on", [])
        if not isinstance(deps, list) or any(
                not isinstance(d, str) for d in deps):
            raise ProjectError(
                f"module {name!r} depends_on must be a list of "
                "module names",
                kind="bad_manifest")
        for dep in deps:
            if dep not in KNOWN_MODULES:
                raise ProjectError(
                    f"module {name!r} depends on unknown module "
                    f"{dep!r}",
                    kind="unknown_module")
            if dep == name:
                raise ProjectError(
                    f"module {name!r} depends on itself",
                    kind="bad_manifest")
        out.append({"name": name, "type": mtype,
                    "loading_phase": phase,
                    "depends_on": list(deps)})
    for mod in out:
        for dep in mod["depends_on"]:
            if dep not in seen:
                raise ProjectError(
                    f"module {mod['name']!r} depends on {dep!r}, "
                    "which this project does not load",
                    kind="unknown_module")
    order = {m["name"]: i for i, m in enumerate(out)}
    visiting: set[str] = set()
    done: set[str] = set()

    def _visit(name: str, trail: list[str]) -> None:
        if name in done:
            return
        if name in visiting:
            raise ProjectError(
                "module dependency cycle: "
                + " -> ".join(trail + [name]),
                kind="bad_manifest")
        visiting.add(name)
        for dep in out[order[name]]["depends_on"]:
            _visit(dep, trail + [name])
        visiting.discard(name)
        done.add(name)

    for mod in out:
        _visit(mod["name"], [])
    return out


def _check_plugins(plugins: Any) -> list[dict[str, Any]]:
    if not isinstance(plugins, list):
        raise ProjectError(
            "'plugins' must be a list", kind="bad_manifest")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for raw in plugins:
        if not isinstance(raw, dict):
            raise ProjectError(
                f"plugin {raw!r} is not an object",
                kind="bad_manifest")
        name = raw.get("name")
        if name not in KNOWN_PLUGINS:
            raise ProjectError(
                f"unknown plugin {name!r} — known plugins: "
                f"{', '.join(sorted(KNOWN_PLUGINS))}",
                kind="unknown_plugin")
        if name in seen:
            raise ProjectError(
                f"duplicate plugin {name!r}", kind="bad_manifest")
        seen.add(name)
        out.append({"name": name,
                    "enabled": bool(raw.get("enabled", True)),
                    "source": str(raw.get("source") or "builtin")})
    return out


def _check_targets(targets: Any,
                   modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(targets, list) or not targets:
        raise ProjectError(
            "a project needs a non-empty 'targets' list",
            kind="bad_manifest")
    names = {m["name"] for m in modules}
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for raw in targets:
        if not isinstance(raw, dict):
            raise ProjectError(
                f"target {raw!r} is not an object",
                kind="bad_manifest")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProjectError(
                "a target needs a non-empty 'name'",
                kind="bad_manifest")
        if name in seen:
            raise ProjectError(
                f"duplicate target {name!r}", kind="bad_manifest")
        seen.add(name)
        ttype = raw.get("type", "runtime")
        if ttype not in TARGET_TYPES:
            raise ProjectError(
                f"target {name!r} type must be one of "
                f"{', '.join(sorted(TARGET_TYPES))}, got {ttype!r}",
                kind="bad_target")
        tmods = raw.get("modules", [])
        if not isinstance(tmods, list) or any(
                not isinstance(m, str) for m in tmods):
            raise ProjectError(
                f"target {name!r} modules must be a list of "
                "module names",
                kind="bad_target")
        for mod in tmods:
            if mod not in names:
                raise ProjectError(
                    f"target {name!r} loads unknown module {mod!r}",
                    kind="unknown_module")
        out.append({"name": name, "type": ttype,
                    "modules": list(tmods)})
    return out


def _check_entries(entries: Any) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(entries, dict):
        raise ProjectError(
            "'entries' must be an object", kind="bad_manifest")
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in ("scenes", "takes", "rigs", "assets"):
        raw = entries.get(kind, {})
        if not isinstance(raw, dict):
            raise ProjectError(
                f"entries.{kind} must be an object",
                kind="bad_manifest")
        group: dict[str, dict[str, Any]] = {}
        for name, rec in raw.items():
            validate_entry_name(name, f"{kind[:-1]} entry")
            if not isinstance(rec, dict):
                raise ProjectError(
                    f"entry {name!r} in {kind} is not an object",
                    kind="bad_entry")
            path = rec.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ProjectError(
                    f"entry {name!r} in {kind} needs a path",
                    kind="bad_entry")
            ekind = rec.get("kind", "copy")
            if ekind not in ("copy", "link"):
                raise ProjectError(
                    f"entry {name!r} in {kind} kind must be "
                    f"'copy' or 'link', got {ekind!r}",
                    kind="bad_entry")
            entry: dict[str, Any] = {"path": path, "kind": ekind}
            if kind == "assets":
                entry["asset_kind"] = str(
                    rec.get("asset_kind") or "file")
            group[name] = entry
        out[kind] = group
    return out


@dataclass
class ProjectManifest:
    """A Devon project manifest — the ``.uproject`` analogue."""

    name: str
    description: str = ""
    category: str = ""
    project_id: str = ""
    devon_version: str = ""
    source_engine: str = ""
    modules: list[dict[str, Any]] = field(default_factory=list)
    plugins: list[dict[str, Any]] = field(default_factory=list)
    unmapped_plugins: list[str] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    default_scene: str = ""
    entries: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict)
    format_version: int = 1

    def __post_init__(self) -> None:
        self.name = validate_entry_name(self.name, "project")
        if not self.project_id:
            self.project_id = uuid.uuid4().hex
        if not re.fullmatch(r"[0-9a-f]{32}", self.project_id):
            raise ProjectError(
                f"project_id {self.project_id!r} must be 32 hex "
                "chars",
                kind="bad_manifest")
        self.modules = _check_modules(
            self.modules or default_modules())
        self.plugins = _check_plugins(
            self.plugins if self.plugins is not None
            else default_plugins())
        if not isinstance(self.unmapped_plugins, list) or any(
                not isinstance(p, str) for p in self.unmapped_plugins):
            raise ProjectError(
                "'unmapped_plugins' must be a list of names",
                kind="bad_manifest")
        self.unmapped_plugins = sorted(set(self.unmapped_plugins))
        self.targets = _check_targets(
            self.targets or default_targets(), self.modules)
        self.entries = _check_entries(self.entries or {})
        if self.default_scene:
            validate_entry_name(self.default_scene, "default scene")
            if self.default_scene not in self.entries["scenes"]:
                raise ProjectError(
                    f"default scene {self.default_scene!r} is not "
                    "one of the project's scenes",
                    kind="unknown_entry")

    def to_dict(self) -> dict[str, Any]:
        """The manifest as plain JSON-able data, without the seal."""
        return {
            "format": "devon-project",
            "format_version": int(self.format_version),
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "project_id": self.project_id,
            "devon_version": self.devon_version,
            "source_engine": self.source_engine,
            "modules": [dict(m) for m in self.modules],
            "plugins": [dict(p) for p in self.plugins],
            "unmapped_plugins": list(self.unmapped_plugins),
            "targets": [dict(t) for t in self.targets],
            "default_scene": self.default_scene,
            "entries": {k: {n: dict(r) for n, r in v.items()}
                        for k, v in self.entries.items()},
        }

    def seal(self) -> str:
        """sha256 over the canonical manifest — the Scene3D seal
        idiom. A manifest whose seal does not match is refused."""
        blob = json.dumps(self.to_dict(), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def sealed_dict(self) -> dict[str, Any]:
        """The manifest plus its seal, ready to write."""
        data = self.to_dict()
        data["seal"] = self.seal()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectManifest:
        """Build a manifest from stored JSON. A bad seal, a wrong
        format or a bad field is refused by name."""
        if not isinstance(data, dict):
            raise ProjectError(
                "manifest must be a JSON object",
                kind="bad_manifest")
        if data.get("format") != "devon-project":
            raise ProjectError(
                f"not a devon-project manifest (format="
                f"{data.get('format')!r})",
                kind="bad_manifest")
        if data.get("format_version") != 1:
            raise ProjectError(
                "unsupported manifest format_version "
                f"{data.get('format_version')!r}",
                kind="bad_manifest")
        seal = data.get("seal")
        manifest = cls(
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            category=str(data.get("category") or ""),
            project_id=str(data.get("project_id") or ""),
            devon_version=str(data.get("devon_version") or ""),
            source_engine=str(data.get("source_engine") or ""),
            modules=data.get("modules") or [],
            plugins=data.get("plugins") or [],
            unmapped_plugins=data.get("unmapped_plugins") or [],
            targets=data.get("targets") or [],
            default_scene=str(data.get("default_scene") or ""),
            entries=data.get("entries") or {},
            format_version=1)
        if seal is not None and seal != manifest.seal():
            raise ProjectError(
                "manifest seal mismatch — the file was edited or "
                "corrupted after it was written",
                kind="bad_seal")
        return manifest


def import_uproject(data: dict[str, Any]) -> tuple[ProjectManifest,
                                                   dict[str, Any]]:
    """Map a real UE ``.uproject`` onto a Devon manifest.

    The primary module (named after the project itself) becomes
    the ``scene`` module — the runtime module that runs the
    game. Every other module must be a known Devon organ or the
    import is refused. Plugins map through
    :data:`UPROJECT_PLUGIN_MAP`; unmapped plugins
    (``EventCenterPlugin``) are recorded disabled and named in
    the receipt, never silently dropped. The UE engine
    association is kept as ``source_engine``.

    Returns the manifest and a receipt: the mapped modules and
    plugins, the unmapped plugins, and the source engine.
    """
    if not isinstance(data, dict):
        raise ProjectError(
            "a .uproject must be a JSON object",
            kind="bad_manifest")
    raw_modules = data.get("Modules")
    if not isinstance(raw_modules, list) or not raw_modules:
        raise ProjectError(
            "a .uproject needs a non-empty 'Modules' list",
            kind="bad_manifest")
    project_name = ""
    for raw in raw_modules:
        if isinstance(raw, dict) and raw.get("Name"):
            project_name = str(raw["Name"])
            break
    if not project_name:
        raise ProjectError(
            "a .uproject module needs a 'Name'",
            kind="bad_manifest")
    modules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_modules:
        if not isinstance(raw, dict):
            raise ProjectError(
                f".uproject module {raw!r} is not an object",
                kind="bad_manifest")
        raw_name = str(raw.get("Name") or "")
        if raw_name == project_name:
            mapped = "scene"
        elif raw_name in KNOWN_MODULES:
            mapped = raw_name
        else:
            raise ProjectError(
                f".uproject module {raw_name!r} has no Devon "
                "organ — known modules: "
                f"{', '.join(sorted(KNOWN_MODULES))}",
                kind="unknown_module")
        if mapped in seen:
            raise ProjectError(
                f".uproject maps two modules onto {mapped!r}",
                kind="bad_manifest")
        seen.add(mapped)
        mtype = str(raw.get("Type") or "Runtime").lower()
        if mtype not in MODULE_TYPES:
            raise ProjectError(
                f".uproject module {raw_name!r} type {mtype!r} "
                "is not runtime/editor",
                kind="bad_manifest")
        phase = str(raw.get("LoadingPhase") or "Default").lower()
        if phase not in LOADING_PHASES:
            raise ProjectError(
                f".uproject module {raw_name!r} loading phase "
                f"{phase!r} is not default/pre_default/"
                "post_default",
                kind="bad_manifest")
        modules.append({"name": mapped, "type": mtype,
                        "loading_phase": phase,
                        "depends_on": list(_MODULE_DEPS[mapped])})
    for mod in modules:
        mod["depends_on"] = [d for d in mod["depends_on"]
                             if d in seen]
    plugins: list[dict[str, Any]] = []
    unmapped: list[str] = []
    for raw in data.get("Plugins") or []:
        if not isinstance(raw, dict):
            raise ProjectError(
                f".uproject plugin {raw!r} is not an object",
                kind="bad_manifest")
        raw_name = str(raw.get("Name") or "")
        if raw_name in UPROJECT_PLUGIN_MAP:
            plugins.append({
                "name": UPROJECT_PLUGIN_MAP[raw_name],
                "enabled": bool(raw.get("Enabled", True)),
                "source": "uproject"})
        elif raw_name in UPROJECT_UNMAPPED_PLUGINS:
            unmapped.append(raw_name)
        else:
            raise ProjectError(
                f".uproject plugin {raw_name!r} has no Devon "
                "analogue — known plugins: "
                f"{', '.join(sorted(KNOWN_PLUGINS))}",
                kind="unknown_plugin")
    # Unmapped plugins stay out of the plugin list — they are
    # recorded on the manifest, disabled, named in the receipt.
    manifest = ProjectManifest(
        name=project_name,
        description=str(data.get("Description") or ""),
        category=str(data.get("Category") or ""),
        source_engine=str(data.get("EngineAssociation") or ""),
        modules=modules,
        plugins=plugins,
        unmapped_plugins=sorted(set(unmapped)),
        targets=[
            {"name": "game", "type": "runtime",
             "modules": [m["name"] for m in modules]},
            {"name": "editor", "type": "editor",
             "modules": [m["name"] for m in modules]},
        ])
    receipt = {
        "project": project_name,
        "source_engine": manifest.source_engine,
        "modules": [m["name"] for m in modules],
        "plugins": [p["name"] for p in plugins
                    if p["source"] == "uproject"],
        "unmapped_plugins": sorted(set(unmapped)),
    }
    return manifest, receipt
