"""Plugin shipping path — a plugin is a data artifact with a manifest.

The existing loader/registry path (``loader.py`` / ``registry.py``)
ships *code* plugins: Python entry points, lifecycle hooks, capability
grants. This module ships the other kind of plugin Devon needs: a
**data artifact** — a directory (or zip) whose manifest declares data
payloads rather than code:

* ``sources`` — source packs ( corpora, scraped datasets, feeds )
* ``culture_packs`` — culture / locale packs
* ``playbooks`` — procedure / runbook packs
* ``g2p`` — grapheme-to-phoneme / pronunciation packs
* ``tradition_packs`` — tradition / ritual / lore packs
* ``skill_packs`` — skill definition packs

**The plugin-declares-permissions invariant.** Every shipped plugin
must declare its permissions in its manifest — there is no implicit
grant. A manifest with no ``permissions`` key is rejected. Shipping
permissions use the small, auditable vocabulary ``net`` / ``fs`` /
``spawn`` (plus the dotted ``KNOWN_PERMISSIONS`` names from
``manifest.py`` for code-adjacent data plugins). The mapping is:

* ``net`` → ``network.fetch`` (outbound HTTP fetches)
* ``fs`` → ``artifacts.read`` + ``artifacts.write`` + ``storage.kv``
* ``spawn`` → subprocess / worker spawn (no dotted equivalent in
  ``KNOWN_PERMISSIONS`` — it is recorded verbatim and enforced by the
  host, never silently widened)

``install_plugin(path)`` validates the manifest *before* anything is
copied: bad manifest, unknown permission, missing artifact file, or a
path-traversal artifact entry (``..`` / absolute path) all fail fast
with :exc:`ManifestError` and install nothing. Installs land in the
plugins dir as ``<name>-<version>/`` with an ``installed.json``
receipt; :func:`list_plugins` reads those receipts;
:func:`uninstall` removes the directory (and only that directory).
"""

from __future__ import annotations

import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .errors import AlreadyInstalled, ManifestError, NotInstalled
from .manifest import KNOWN_PERMISSIONS

_log = get_logger(__name__)

__all__ = [
    "ARTIFACT_KINDS",
    "PERMISSION_MAP",
    "SHIPPING_PERMISSIONS",
    "ShippingManifest",
    "declared_permissions",
    "install_plugin",
    "list_plugins",
    "uninstall",
    "validate_shipping_manifest",
]

#: Data-artifact sections a shipped plugin may carry.
ARTIFACT_KINDS = frozenset({
    "sources",
    "culture_packs",
    "playbooks",
    "g2p",
    "tradition_packs",
    "skill_packs",
})

#: Shipping permission vocabulary. ``net``/``fs``/``spawn`` are the
#: short forms; dotted names are the existing KNOWN_PERMISSIONS.
SHIPPING_PERMISSIONS = frozenset({
    "net", "fs", "spawn",
    *KNOWN_PERMISSIONS,
})

#: Short-form → dotted capability expansion. ``spawn`` has no dotted
#: equivalent — it stays verbatim so the host can enforce it exactly.
PERMISSION_MAP: dict[str, tuple[str, ...]] = {
    "net": ("network.fetch",),
    "fs": ("artifacts.read", "artifacts.write", "storage.kv"),
    "spawn": ("spawn",),
}

#: Manifest file names accepted at the plugin root (in order).
MANIFEST_CANDIDATES = ("plugin.json", "manifest.json")

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_VERSION_RE = re.compile(
    r"^\d+\.\d+\.\d+(([ab]\d+)|([-+][0-9A-Za-z.-]+))?$")
# Max unpacked size for a shipped plugin — refuse zip bombs, don't fill disk.
MAX_PLUGIN_BYTES = 100 * 1024 * 1024


def declared_permissions(manifest: dict[str, Any]) -> list[str]:
    """Permissions a manifest declares, expanded to dotted capabilities.

    The invariant: the manifest *must* declare ``permissions`` —
    missing/empty is an error, not an empty grant. Short forms are
    expanded via :data:`PERMISSION_MAP`; dotted names pass through.
    Unknown names raise :exc:`ManifestError`. Duplicates are removed,
    order preserved.
    """
    raw = manifest.get("permissions", None)
    if raw is None:
        raise ManifestError(
            "plugin must declare permissions: manifest has no "
            "'permissions' key (no implicit grant)")
    if not isinstance(raw, list) or not raw:
        raise ManifestError(
            "plugin must declare permissions: 'permissions' must be a "
            "non-empty list")
    unknown = [p for p in raw if p not in SHIPPING_PERMISSIONS]
    if unknown:
        raise ManifestError(
            f"unknown permissions {unknown}; known shipping permissions: "
            f"{sorted(SHIPPING_PERMISSIONS)}")
    out: list[str] = []
    for perm in raw:
        for expanded in PERMISSION_MAP.get(str(perm), (str(perm),)):
            if expanded not in out:
                out.append(expanded)
    return out


def validate_shipping_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate a shipping manifest → normalized copy. Raises ManifestError.

    Checks: ``name`` (``[a-z0-9_-]``), ``version`` (semver),
    ``permissions`` declared (see :func:`declared_permissions`),
    ``artifacts`` (if present) mapping known kinds → lists of entries
    with a ``path`` each. Artifact ``path`` values are checked for
    traversal later against the real plugin root (see
    :func:`_check_artifact_paths`) because the manifest alone cannot
    prove a path stays inside the directory.
    """
    if not isinstance(manifest, dict):
        raise ManifestError("plugin manifest must be a JSON object")
    name = manifest.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ManifestError(f"bad plugin name: {name!r}")
    version = manifest.get("version")
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise ManifestError(f"bad plugin version: {version!r}")
    # The invariant: permissions must be declared. Validates + expands.
    perms = declared_permissions(manifest)
    artifacts = manifest.get("artifacts", {})
    if artifacts is None:
        artifacts = {}
    if not isinstance(artifacts, dict):
        raise ManifestError("'artifacts' must be an object keyed by kind")
    unknown_kinds = [k for k in artifacts if k not in ARTIFACT_KINDS]
    if unknown_kinds:
        raise ManifestError(
            f"unknown artifact kinds {unknown_kinds}; known: "
            f"{sorted(ARTIFACT_KINDS)}")
    for kind, entries in artifacts.items():
        if not isinstance(entries, list):
            raise ManifestError(
                f"artifacts[{kind!r}] must be a list of entries")
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("path"):
                raise ManifestError(
                    f"artifacts[{kind!r}] entries need a 'path'")
            if not isinstance(entry["path"], str):
                raise ManifestError(
                    f"artifacts[{kind!r}] entry 'path' must be a string")
    normalized = dict(manifest)
    normalized["artifacts"] = artifacts
    normalized["_declared_permissions"] = perms
    return normalized


def _check_artifact_paths(root: Path, artifacts: dict[str, Any]) -> None:
    """Every artifact file must exist and stay inside ``root``.

    Absolute paths and ``..`` escapes are rejected (path traversal).
    Symlinks that resolve outside ``root`` are rejected too.
    """
    root_resolved = root.resolve()
    for kind, entries in (artifacts or {}).items():
        for entry in entries:
            rel = Path(str(entry["path"]))
            if rel.is_absolute() or ".." in rel.parts:
                raise ManifestError(
                    f"artifacts[{kind!r}] path escapes the plugin dir: "
                    f"{entry['path']!r}")
            target = (root_resolved / rel).resolve()
            if not str(target).startswith(str(root_resolved) + os_sep()):
                raise ManifestError(
                    f"artifacts[{kind!r}] path escapes the plugin dir: "
                    f"{entry['path']!r}")
            if not target.is_file():
                raise ManifestError(
                    f"artifacts[{kind!r}] file missing: {entry['path']!r}")


def os_sep() -> str:
    import os as _os
    return _os.sep


def _load_manifest_from_dir(root: Path) -> dict[str, Any]:
    for candidate in MANIFEST_CANDIDATES:
        path = root / candidate
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ManifestError(
                    f"unparsable manifest {candidate}: {exc}") from exc
            return validate_shipping_manifest(data)
    raise ManifestError(
        f"no manifest found in {root} "
        f"(looked for {', '.join(MANIFEST_CANDIDATES)})")


def _extract_zip(path: Path, dest: Path) -> Path:
    """Extract a plugin zip to ``dest``; returns the plugin root dir.

    Refuses zip bombs (total unpacked size cap) and traversal entries.
    A zip may contain the plugin at its root, or one top-level dir.
    """
    total = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            total += info.file_size
            if total > MAX_PLUGIN_BYTES:
                raise ManifestError(
                    f"plugin zip exceeds {MAX_PLUGIN_BYTES} bytes")
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ManifestError(
                    f"zip entry escapes the plugin dir: {info.filename!r}")
        zf.extractall(dest)
    # One top-level dir → that is the root; else dest itself.
    children = [p for p in dest.iterdir() if p.is_dir()]
    files = [p for p in dest.iterdir() if p.is_file()]
    if len(children) == 1 and not files:
        return children[0]
    return dest


class ShippingManifest:
    """A validated shipping manifest + its artifact inventory."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.raw = validate_shipping_manifest(manifest)
        self.name: str = self.raw["name"]
        self.version: str = self.raw["version"]
        self.permissions: list[str] = self.raw["_declared_permissions"]
        self.artifacts: dict[str, Any] = self.raw.get("artifacts", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "permissions": list(self.permissions),
            "artifacts": self.artifacts,
        }


def install_plugin(path: str | Path, *,
                    plugins_dir: str | Path | None = None,
                    overwrite: bool = False) -> dict[str, Any]:
    """Install a shipped (data-artifact) plugin.

    ``path`` is a plugin directory or a ``.zip`` containing one.
    Validates the manifest + permissions + artifact paths *before*
    copying anything; on any failure nothing is installed.
    Installs to ``<plugins_dir>/<name>-<version>/`` and writes an
    ``installed.json`` receipt (manifest + permissions + installed_at).
    Returns the receipt dict. Re-installing the same name+version
    raises :exc:`AlreadyInstalled` unless ``overwrite=True``.
    """
    import tempfile

    src = Path(path)
    if not src.exists():
        raise ManifestError(f"plugin path does not exist: {path!r}")
    dest_base = Path(plugins_dir) if plugins_dir else (
        Path.cwd() / "plugins")
    dest_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="devon-plugin-") as tmp:
        tmp_path = Path(tmp)
        if src.is_file() and src.suffix.lower() == ".zip":
            root = _extract_zip(src, tmp_path / "extracted")
        elif src.is_dir():
            root = src
        else:
            raise ManifestError(
                f"plugin path must be a directory or .zip: {path!r}")
        manifest = _load_manifest_from_dir(root)
        shipping = ShippingManifest(manifest)
        _check_artifact_paths(root, shipping.artifacts)
        dest = dest_base / f"{shipping.name}-{shipping.version}"
        if dest.exists():
            if not overwrite:
                raise AlreadyInstalled(
                    f"{shipping.name} {shipping.version} already installed")
            shutil.rmtree(dest)
        # Copy from the validated root (for a zip, the extracted copy).
        if root == src:
            shutil.copytree(src, dest)
        else:
            shutil.copytree(root, dest)
        receipt = {
            "name": shipping.name,
            "version": shipping.version,
            "permissions": list(shipping.permissions),
            "artifacts": shipping.artifacts,
            "path": str(dest),
            "installed_at": time.time(),
        }
        (dest / "installed.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True),
            encoding="utf-8")
    _log.info("installed shipped plugin %s %s (perms=%s)",
              shipping.name, shipping.version, shipping.permissions)
    return receipt


def list_plugins(plugins_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Installed shipped plugins: one receipt per ``<name>-<version>/``.

    Directories without an ``installed.json`` receipt are skipped
    (they are not shipping installs — e.g. code plugins from the
    registry path, or stray directories).
    """
    base = Path(plugins_dir) if plugins_dir else (Path.cwd() / "plugins")
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        receipt_path = child / "installed.json"
        if not receipt_path.is_file():
            continue
        try:
            out.append(json.loads(receipt_path.read_text(encoding="utf-8")))
        except ValueError:
            continue
    return out


def uninstall(name: str, *,
              plugins_dir: str | Path | None = None,
              version: str | None = None,
              purge: bool = True) -> bool:
    """Uninstall a shipped plugin by name (and optional version).

    Removes exactly the ``<name>-<version>/`` directory recorded in
    the receipt — never anything outside the plugins dir. Raises
    :exc:`NotInstalled` when no matching receipt exists. Returns True.
    """
    base = Path(plugins_dir) if plugins_dir else (Path.cwd() / "plugins")
    candidates = [p for p in list_plugins(base)
                  if p.get("name") == name
                  and (version is None or p.get("version") == version)]
    if not candidates:
        raise NotInstalled(f"no shipped plugin installed: {name!r}")
    if len(candidates) > 1 and version is None:
        raise ManifestError(
            f"multiple versions of {name!r} installed — pass version="
            f"{sorted(c['version'] for c in candidates)}")
    target = Path(candidates[0]["path"]).resolve()
    base_resolved = base.resolve()
    if not str(target).startswith(str(base_resolved) + os_sep()):
        raise ManifestError(
            f"refusing to remove path outside plugins dir: {target}")
    if purge:
        shutil.rmtree(target)
    return True
