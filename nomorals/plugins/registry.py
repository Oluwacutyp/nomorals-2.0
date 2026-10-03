"""Persisted registry of installed plugin packages.

Follows the ``nomorals/skills/registry.py`` pattern: a ``Database`` plus
``CREATE TABLE IF NOT EXISTS`` DDL executed eagerly at construction.

Multiple versions of one plugin coexist (``PRIMARY KEY (name,
version)``); ``enable``/``disable`` flips the per-version switch.
Installing copies the plugin directory into the registry's plugin home
so the source can go away afterwards.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .errors import AlreadyInstalled, ManifestError, NotInstalled
from .manifest import MANIFEST_FILENAME, PluginManifest, load_manifest_file

__all__ = ["PluginRegistry", "PLUGIN_PACKAGES_DDL", "InstalledPlugin"]

PLUGIN_PACKAGES_DDL = """
CREATE TABLE IF NOT EXISTS plugin_packages (
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    manifest    TEXT NOT NULL,
    path        TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (name, version)
);
CREATE INDEX IF NOT EXISTS idx_plugin_packages_name
    ON plugin_packages(name);
"""

_log = get_logger(__name__)


@dataclass
class InstalledPlugin:
    name: str
    version: str
    manifest: PluginManifest
    path: str
    enabled: bool
    created_at: float
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.manifest.description,
            "author": self.manifest.author,
            "entry_points": dict(self.manifest.entry_points),
            "permissions": list(self.manifest.permissions),
            "path": self.path,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class PluginRegistry:
    """Install, list, enable/disable, remove plugin packages."""

    def __init__(self, db: Any, plugin_home: str | Path) -> None:
        self.db = db
        self.home = Path(plugin_home)
        self.home.mkdir(parents=True, exist_ok=True)
        db.executescript(PLUGIN_PACKAGES_DDL)

    # ── install ─────────────────────────────────────────────────────
    def install(self, source: str | Path, *,
                enabled: bool = True) -> InstalledPlugin:
        """Install a plugin from a directory path, zip file, or URL.

        Returns the installed record. Raises :exc:`AlreadyInstalled`,
        :exc:`ManifestError`, or :exc:`PluginError` on download failure.
        """
        src = str(source)
        if src.startswith(("http://", "https://")):
            staged = self._download(src)
        else:
            staged = Path(src)
            if not staged.exists():
                raise ManifestError(f"plugin source not found: {src}")
        manifest = load_manifest_file(staged)
        if self._get(manifest.name, manifest.version) is not None:
            raise AlreadyInstalled(
                f"{manifest.name} {manifest.version} already installed")
        dest = self.home / f"{manifest.name}-{manifest.version}"
        if dest.exists():
            shutil.rmtree(dest)
        if staged.is_dir():
            shutil.copytree(staged, dest)
        else:
            # zip file: unpack
            dest.mkdir(parents=True)
            with zipfile.ZipFile(staged) as zf:
                zf.extractall(dest)
            # tolerate a single top-level directory wrapper
            inner = [p for p in dest.iterdir() if p.is_dir()]
            if len(inner) == 1 and not (dest / MANIFEST_FILENAME).exists():
                for child in inner[0].iterdir():
                    shutil.move(str(child), dest)
                inner[0].rmdir()
            # re-validate after unpack (manifest must be at top level)
            load_manifest_file(dest)
        now = time.time()
        self.db.execute(
            "INSERT INTO plugin_packages "
            "(name, version, manifest, path, enabled, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (manifest.name, manifest.version,
             json.dumps(manifest.to_dict()), str(dest),
             int(enabled), now, now),
        )
        _log.info("installed plugin %s %s", manifest.name, manifest.version)
        return self._get(manifest.name, manifest.version)  # type: ignore[return-value]

    def _download(self, url: str) -> Path:
        from .errors import PluginError
        tmp = self.home / "_downloads"
        tmp.mkdir(parents=True, exist_ok=True)
        dest = tmp / f"dl_{int(time.time() * 1000)}.zip"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "DevonPluginRegistry/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, \
                    open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        except Exception as exc:
            raise PluginError(f"download failed for {url}: {exc}") from exc
        return dest

    # ── read ────────────────────────────────────────────────────────
    def _row_to_plugin(self, row: Any) -> InstalledPlugin:
        from .manifest import load_manifest
        manifest = load_manifest(json.loads(row["manifest"]))
        return InstalledPlugin(
            name=row["name"], version=row["version"], manifest=manifest,
            path=row["path"], enabled=bool(row["enabled"]),
            created_at=row["created_at"], updated_at=row["updated_at"])

    def _get(self, name: str, version: str) -> InstalledPlugin | None:
        row = self.db.execute(
            "SELECT * FROM plugin_packages WHERE name=? AND version=?",
            (name, version)).fetchone()
        return self._row_to_plugin(row) if row else None

    def get(self, name: str, version: str = "") -> InstalledPlugin:
        """Get a plugin; latest version when ``version`` is empty."""
        if version:
            plugin = self._get(name, version)
        else:
            row = self.db.execute(
                "SELECT * FROM plugin_packages WHERE name=? "
                "ORDER BY version DESC LIMIT 1", (name,)).fetchone()
            plugin = self._row_to_plugin(row) if row else None
        if plugin is None:
            raise NotInstalled(f"plugin {name!r} is not installed")
        return plugin

    def list(self, *, enabled_only: bool = False) -> list[InstalledPlugin]:
        q = "SELECT * FROM plugin_packages"
        if enabled_only:
            q += " WHERE enabled=1"
        q += " ORDER BY name, version"
        return [self._row_to_plugin(r) for r in self.db.execute(q).fetchall()]

    # ── enable / disable / remove ───────────────────────────────────
    def enable(self, name: str, version: str = "") -> InstalledPlugin:
        plugin = self.get(name, version)
        self.db.execute(
            "UPDATE plugin_packages SET enabled=1, updated_at=? "
            "WHERE name=? AND version=?",
            (time.time(), plugin.name, plugin.version))
        plugin.enabled = True
        return plugin

    def disable(self, name: str, version: str = "") -> InstalledPlugin:
        plugin = self.get(name, version)
        self.db.execute(
            "UPDATE plugin_packages SET enabled=0, updated_at=? "
            "WHERE name=? AND version=?",
            (time.time(), plugin.name, plugin.version))
        plugin.enabled = False
        return plugin

    def remove(self, name: str, version: str = "") -> None:
        plugin = self.get(name, version)
        self.db.execute(
            "DELETE FROM plugin_packages WHERE name=? AND version=?",
            (plugin.name, plugin.version))
        path = Path(plugin.path)
        if path.exists() and path.is_relative_to(self.home):
            shutil.rmtree(path, ignore_errors=True)
        _log.info("removed plugin %s %s", plugin.name, plugin.version)
