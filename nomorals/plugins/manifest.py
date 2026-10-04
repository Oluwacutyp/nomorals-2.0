"""Plugin manifests: name, version, entry points, permissions.

A plugin is a directory containing a ``plugin.json`` manifest plus Python
modules. The manifest declares everything the loader needs::

    {
      "name": "my_plugin",
      "version": "1.0.0",
      "description": "Does useful things.",
      "author": "owner",
      "entry_points": {"main": "my_plugin:run", "hooks": "my_plugin:hooks"},
      "permissions": ["artifacts.write", "network.fetch"]
    }

Permission names are dotted capability strings. The loader grants exactly
the permissions listed at install time; anything else the plugin tries
to use raises :exc:`PermissionDenied`. Known permissions:

* ``artifacts.read`` / ``artifacts.write`` — read/write Devon's artifacts
* ``network.fetch`` — outbound HTTP(S) fetches
* ``storage.kv`` — a private key-value table for the plugin
* ``llm.chat`` — call the model broker for chat completions

Unknown permission strings are rejected at install — fail fast, no
silent grants.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ManifestError

__all__ = [
    "KNOWN_PERMISSIONS",
    "MANIFEST_FILENAME",
    "PluginManifest",
    "load_manifest",
]

MANIFEST_FILENAME = "plugin.json"

KNOWN_PERMISSIONS = frozenset({
    "artifacts.read",
    "artifacts.write",
    "network.fetch",
    "storage.kv",
    "llm.chat",
})

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_VERSION_RE = re.compile(
    r"^\d+\.\d+\.\d+(([ab]\d+)|([-+][0-9A-Za-z.-]+))?$")


@dataclass
class PluginManifest:
    name: str
    version: str
    description: str = ""
    author: str = ""
    entry_points: dict[str, str] = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "entry_points": dict(self.entry_points),
            "permissions": list(self.permissions),
        }


def load_manifest(data: dict[str, Any]) -> PluginManifest:
    """Validate a manifest dict. Raises :exc:`ManifestError` on any problem."""
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")
    name = data.get("name", "")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ManifestError(
            f"bad plugin name {name!r}: 2-64 chars, lowercase alnum/-/_")
    version = data.get("version", "")
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise ManifestError(
            f"bad version {version!r}: expected semver like 1.0.0 "
            f"(prerelease/build suffixes like 1.0.0-beta or 1.0.0+build "
            f"are allowed)")
    entry_points = data.get("entry_points", {})
    if not isinstance(entry_points, dict) or not entry_points:
        raise ManifestError("manifest needs a non-empty entry_points map")
    for key, target in entry_points.items():
        if not isinstance(target, str) or ":" not in target:
            raise ManifestError(
                f"entry point {key!r} must look like 'module:attr', "
                f"got {target!r}")
    permissions = data.get("permissions", [])
    if not isinstance(permissions, list):
        raise ManifestError("permissions must be a list")
    unknown = [p for p in permissions if p not in KNOWN_PERMISSIONS]
    if unknown:
        raise ManifestError(
            f"unknown permissions {unknown}; known: {sorted(KNOWN_PERMISSIONS)}")
    return PluginManifest(
        name=name,
        version=str(version),
        description=str(data.get("description", "")),
        author=str(data.get("author", "")),
        entry_points={str(k): str(v) for k, v in entry_points.items()},
        permissions=[str(p) for p in permissions],
    )


def load_manifest_file(path: str | Path) -> PluginManifest:
    """Read and validate ``plugin.json`` from a plugin directory."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_FILENAME
    if not p.is_file():
        raise ManifestError(f"no manifest at {p} (expected plugin.json)")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"could not read manifest {p}: {exc}") from exc
    return load_manifest(data)
