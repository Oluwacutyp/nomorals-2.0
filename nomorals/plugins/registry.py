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
import re
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .errors import AlreadyInstalled, ManifestError, NotInstalled, PluginError
from .manifest import MANIFEST_FILENAME, PluginManifest, load_manifest_file

_log = get_logger(__name__)

# Plugins are small by nature — refuse anything bigger rather than
# filling the disk with a corrupt/huge download or zip bomb.
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_PLUGIN_BYTES = 100 * 1024 * 1024


def _version_key(version: str) -> tuple:
    """Ordering key so 1.10.0 > 1.9.0 and 1.0.0 > 1.0.0-beta.

    Compares the numeric core first; a release (no suffix) sorts above
    the same core with a prerelease suffix, per semver.
    """
    parts = re.split(r"[.\-+]", str(version))
    nums: list[int] = []
    suffix: list[str] = []
    for p in parts:
        if p.isdigit() and not suffix:
            nums.append(int(p))
        else:
            suffix.append(p)
    return (tuple(nums), (1,) if not suffix else (0, tuple(suffix)))


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break plugin management (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

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
            staged = self._download(src)  # always a zip file
        else:
            staged = Path(src)
            if not staged.exists():
                raise ManifestError(f"plugin source not found: {src}")
        if staged.is_dir():
            manifest = load_manifest_file(staged)
        elif zipfile.is_zipfile(staged):
            manifest = self._manifest_from_zip(staged)
        else:
            raise ManifestError(
                f"plugin source {src!r} is neither a directory nor a "
                f"zip file")
        if self._get(manifest.name, manifest.version) is not None:
            raise AlreadyInstalled(
                f"{manifest.name} {manifest.version} already installed")
        dest = self.home / f"{manifest.name}-{manifest.version}"
        if dest.exists():
            shutil.rmtree(dest)
        if staged.is_dir():
            shutil.copytree(staged, dest)
        else:
            dest.mkdir(parents=True)
            self._safe_extract(staged, dest)
            # tolerate a single top-level directory wrapper
            inner = [p for p in dest.iterdir() if p.is_dir()]
            if len(inner) == 1 and not (dest / MANIFEST_FILENAME).exists():
                for child in inner[0].iterdir():
                    shutil.move(str(child), dest)
                inner[0].rmdir()
        # re-validate after staging (manifest must be at top level)
        manifest = load_manifest_file(dest)
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
        _emit("plugin.installed", {
            "name": manifest.name,
            "version": manifest.version,
            "enabled": bool(enabled),
            "path": str(dest),
        })
        return self._get(manifest.name, manifest.version)  # type: ignore[return-value]

    def _manifest_from_zip(self, zip_path: Path) -> PluginManifest:
        """Read + validate the manifest inside a zip without extracting.

        Tolerates the manifest at the top level or inside one wrapper
        directory (same rule as the extractor below).
        """
        from .manifest import load_manifest
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            candidate = None
            for n in names:
                if n == MANIFEST_FILENAME or n.endswith(
                        "/" + MANIFEST_FILENAME):
                    candidate = n
                    break
            if candidate is None:
                raise ManifestError(
                    f"no {MANIFEST_FILENAME} in plugin zip {zip_path}")
            try:
                data = json.loads(zf.read(candidate).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ManifestError(
                    f"could not read manifest in {zip_path}: {exc}") from exc
        return load_manifest(data)

    def _safe_extract(self, zip_path: Path, dest: Path) -> None:
        """Extract a plugin zip, guarding against zip-slip and zip bombs."""
        resolved_dest = dest.resolve()
        total = 0
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                total += info.file_size
                if total > MAX_PLUGIN_BYTES:
                    raise ManifestError(
                        f"plugin zip {zip_path} unpacks to more than "
                        f"{MAX_PLUGIN_BYTES} bytes — refusing")
                p = Path(info.filename)
                if p.is_absolute() or ".." in p.parts:
                    raise ManifestError(
                        f"unsafe path in plugin zip: {info.filename!r}")
                target = (dest / p).resolve()
                if not target.is_relative_to(resolved_dest):
                    raise ManifestError(
                        f"unsafe path in plugin zip: {info.filename!r}")
            zf.extractall(dest)

    def _download(self, url: str) -> Path:
        tmp = self.home / "_downloads"
        tmp.mkdir(parents=True, exist_ok=True)
        dest = tmp / f"dl_{int(time.time() * 1000)}.zip"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "DevonPluginRegistry/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, \
                    open(dest, "wb") as fh:
                total = 0
                for chunk in iter(lambda: resp.read(65536), b""):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise PluginError(
                            f"download exceeds {MAX_DOWNLOAD_BYTES} bytes: "
                            f"{url}")
                    fh.write(chunk)
        except PluginError:
            dest.unlink(missing_ok=True)
            raise
        except Exception as exc:
            dest.unlink(missing_ok=True)
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
            # Numeric version ordering — lexicographic ORDER BY would rank
            # "1.9.0" above "1.10.0".
            rows = self.db.execute(
                "SELECT * FROM plugin_packages WHERE name=?",
                (name,)).fetchall()
            best = None
            for row in rows:
                if best is None or _version_key(
                        row["version"]) > _version_key(best["version"]):
                    best = row
            plugin = self._row_to_plugin(best) if best else None
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
        _emit("plugin.enabled", {"name": plugin.name,
                                 "version": plugin.version})
        return plugin

    def disable(self, name: str, version: str = "") -> InstalledPlugin:
        plugin = self.get(name, version)
        self.db.execute(
            "UPDATE plugin_packages SET enabled=0, updated_at=? "
            "WHERE name=? AND version=?",
            (time.time(), plugin.name, plugin.version))
        plugin.enabled = False
        _emit("plugin.disabled", {"name": plugin.name,
                                  "version": plugin.version})
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
        _emit("plugin.removed", {"name": plugin.name,
                                 "version": plugin.version})
