"""Persisted registry of installed plugin packages.

Follows the ``nomorals/skills/registry.py`` pattern: a ``Database`` plus
``CREATE TABLE IF NOT EXISTS`` DDL executed eagerly at construction.

Multiple versions of one plugin coexist (``PRIMARY KEY (name,
version)``); ``enable``/``disable`` flips the per-version switch.
Installing copies the plugin directory into the registry's plugin home
so the source can go away afterwards.

Install/upgrade enforce the manifest contract before anything lands:

* ``requires_devon`` — the running engine must satisfy the spec, else
  :exc:`IncompatibleEngine`.
* ``dependencies`` — every ``name[spec]`` must be installed at a
  satisfying version, else :exc:`DependencyError` naming exactly what is
  missing.
* ``requirements`` (pip packages) can't be installed by the registry —
  they're surfaced as a ``plugin.requirements_noted`` event so the host
  can decide.

Lifecycle (WordPress discipline): ``on_install``/``on_upgrade`` run
fail-fast — a failing install hook rolls the install back.
``on_enable``/``on_disable``/``on_uninstall`` run fail-open: a broken
hook is captured, emitted as ``plugin.lifecycle_failed``, and never
traps the plugin in its current state. Deactivation keeps data;
``remove()`` only purges the plugin's kv rows when the manifest sets
``purge_data_on_remove: true``.
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

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .errors import (
    AlreadyInstalled,
    DependencyError,
    IncompatibleEngine,
    LoadError,
    ManifestError,
    NotInstalled,
    PluginError,
)
from .manifest import (
    MANIFEST_FILENAME,
    PluginManifest,
    load_manifest_file,
    satisfies_version,
)
from .manifest import version_key as _version_key

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

# Plugins are small by nature — refuse anything bigger rather than
# filling the disk with a corrupt/huge download or zip bomb.
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_PLUGIN_BYTES = 100 * 1024 * 1024


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break plugin management (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)


class _DbShim:
    """Minimal agent-context stand-in for capability wiring.

    Lifecycle hooks need real backends (kv, artifacts, …); all
    :func:`wiring.wire_capabilities` needs from a context is ``.db``
    (plus optional ``.settings`` / ``.router``).
    """

    def __init__(self, db: Any) -> None:
        self.db = db


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
            "display_name": self.manifest.display_name,
            "description": self.manifest.description,
            "author": self.manifest.author,
            "license": self.manifest.license,
            "homepage": self.manifest.homepage,
            "tags": list(self.manifest.tags),
            "requires_devon": self.manifest.requires_devon,
            "dependencies": [
                {"name": d.name, "spec": d.spec}
                for d in self.manifest.dependencies
            ],
            "entry_points": dict(self.manifest.entry_points),
            "lifecycle": dict(self.manifest.lifecycle),
            "contributes": self.manifest.contribution_summary(),
            "permissions": list(self.manifest.permissions),
            "limits": dict(self.manifest.limits),
            "path": self.path,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class PluginRegistry:
    """Install, list, enable/disable, remove, upgrade plugin packages."""

    def __init__(self, db: Any, plugin_home: str | Path) -> None:
        self.db = db
        self.home = Path(plugin_home)
        self.home.mkdir(parents=True, exist_ok=True)
        db.executescript(PLUGIN_PACKAGES_DDL)

    # ── install ─────────────────────────────────────────────────────
    def install(self, source: str | Path, *,
                enabled: bool = True) -> InstalledPlugin:
        """Install a plugin from a directory path, zip file, or URL.

        Enforces ``requires_devon`` and ``dependencies`` before anything
        lands; runs the ``on_install`` lifecycle hook fail-fast (a
        failing hook rolls the install back).

        Returns the installed record. Raises :exc:`AlreadyInstalled`,
        :exc:`ManifestError`, :exc:`DependencyError`,
        :exc:`IncompatibleEngine`, or :exc:`PluginError` on download
        failure.
        """
        staged, manifest = self._stage(source)
        if self._get(manifest.name, manifest.version) is not None:
            raise AlreadyInstalled(
                f"{manifest.name} {manifest.version} already installed")
        self._check_engine(manifest)
        self._check_dependencies(manifest)
        plugin = self._record_install(staged, manifest, enabled=enabled)
        try:
            self._run_lifecycle(plugin, "on_install", fail_fast=True)
        except Exception:
            # Fail-fast install: a broken on_install leaves no trace.
            self._rollback_install(plugin)
            raise
        _emit("plugin.installed", {
            "name": plugin.name,
            "version": plugin.version,
            "enabled": bool(enabled),
            "path": str(plugin.path),
        })
        if manifest.requirements:
            _emit("plugin.requirements_noted", {
                "name": plugin.name,
                "version": plugin.version,
                "requirements": list(manifest.requirements),
            })
            _log.warning(
                "plugin %s lists pip requirements %s — install them "
                "yourself; the registry never pip-installs",
                plugin.name, manifest.requirements)
        return plugin

    def upgrade(self, source: str | Path, *,
                enabled: bool = True) -> InstalledPlugin:
        """Install a *newer* version of an installed plugin.

        The new version must sort above the latest installed one, else
        :exc:`PluginError`. Runs ``on_upgrade`` fail-fast (failure rolls
        the new version back; the old version is untouched). The
        plugin's kv data carries over automatically — kv is namespaced
        per plugin *name*, not version. When nothing of that name is
        installed, behaves like :meth:`install`.
        """
        staged, manifest = self._stage(source)
        try:
            current = self.get(manifest.name)
        except NotInstalled:
            return self.install(source, enabled=enabled)
        if _version_key(manifest.version) <= _version_key(current.version):
            raise PluginError(
                f"upgrade refused: {manifest.version} is not newer than "
                f"installed {current.version}")
        if self._get(manifest.name, manifest.version) is not None:
            raise AlreadyInstalled(
                f"{manifest.name} {manifest.version} already installed")
        self._check_engine(manifest)
        self._check_dependencies(manifest)
        plugin = self._record_install(staged, manifest, enabled=enabled)
        try:
            self._run_lifecycle(plugin, "on_upgrade", fail_fast=True)
        except Exception:
            self._rollback_install(plugin)
            raise
        _log.info("upgraded plugin %s %s -> %s",
                  manifest.name, current.version, manifest.version)
        _emit("plugin.upgraded", {
            "name": plugin.name,
            "from_version": current.version,
            "to_version": plugin.version,
        })
        return plugin

    def _stage(self, source: str | Path) -> tuple[Path, PluginManifest]:
        """Resolve a source to a staged path + validated manifest."""
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
        return staged, manifest

    def _record_install(self, staged: Path, manifest: PluginManifest,
                        *, enabled: bool) -> InstalledPlugin:
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
        installed = self._get(manifest.name, manifest.version)
        assert installed is not None  # just inserted
        return installed

    def _rollback_install(self, plugin: InstalledPlugin) -> None:
        """Undo a failed install/upgrade: drop the record + files."""
        self.db.execute(
            "DELETE FROM plugin_packages WHERE name=? AND version=?",
            (plugin.name, plugin.version))
        path = Path(plugin.path)
        if path.exists() and path.is_relative_to(self.home):
            shutil.rmtree(path, ignore_errors=True)
        _log.warning("rolled back plugin %s %s after lifecycle failure",
                     plugin.name, plugin.version)

    # ── contract checks ─────────────────────────────────────────────
    def _check_engine(self, manifest: PluginManifest) -> None:
        spec = manifest.requires_devon
        if not spec:
            return
        from ..version import __version__
        if not satisfies_version(__version__, spec):
            raise IncompatibleEngine(
                f"plugin {manifest.name!r} requires Devon {spec}, but "
                f"this engine is {__version__} — upgrade Devon first")

    def _check_dependencies(self, manifest: PluginManifest) -> None:
        problems: list[str] = []
        for dep in manifest.dependencies:
            rows = self.db.execute(
                "SELECT version FROM plugin_packages WHERE name=?",
                (dep.name,)).fetchall()
            satisfied = [r["version"] for r in rows
                         if dep.satisfied_by(r["version"])]
            if not satisfied:
                have = ", ".join(r["version"] for r in rows) or "not installed"
                want = dep.spec or "any version"
                problems.append(
                    f"{dep.name} (want {want}; have {have})")
        if problems:
            raise DependencyError(
                f"plugin {manifest.name!r} has unsatisfied dependencies: "
                + "; ".join(problems))

    def _run_lifecycle(self, plugin: InstalledPlugin, hook: str, *,
                       fail_fast: bool) -> Any:
        """Run one lifecycle entry point for an installed plugin.

        ``fail_fast=True`` (install/upgrade): exceptions propagate as
        :exc:`PluginError`. ``False`` (enable/disable/uninstall): the
        failure is captured, emitted, and swallowed — a broken hook
        must never trap the plugin in its current state.
        """
        from .loader import load_plugin, unload_plugin
        from .wiring import wire_capabilities

        target = plugin.manifest.lifecycle.get(hook)
        if not target:
            return None
        loaded = load_plugin(plugin.manifest, plugin.path)
        try:
            caps = wire_capabilities(
                _DbShim(self.db), plugin,
                workspace_base=self.home.parent)
            return loaded.lifecycle(hook, caps)
        except Exception as exc:
            _log.warning("plugin %s lifecycle %s failed: %s",
                         plugin.name, hook, exc)
            _emit("plugin.lifecycle_failed", {
                "name": plugin.name, "version": plugin.version,
                "hook": hook, "error": f"{type(exc).__name__}: {exc}",
            })
            if fail_fast:
                raise PluginError(
                    f"plugin {plugin.name!r} lifecycle {hook!r} failed: "
                    f"{exc}") from exc
            return None
        finally:
            unload_plugin(loaded)

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

    def search(self, query: str) -> list[InstalledPlugin]:
        """Case-insensitive search over name, display name, description,
        author, and tags."""
        q = str(query or "").strip().lower()
        if not q:
            return []
        hits: list[InstalledPlugin] = []
        for plugin in self.list():
            m = plugin.manifest
            haystack = " ".join([
                plugin.name, m.display_name, m.description, m.author,
                " ".join(m.tags),
            ]).lower()
            if q in haystack:
                hits.append(plugin)
        return hits

    def health(self, name: str, version: str = "") -> dict[str, Any]:
        """No-side-effects check: can this plugin load, do its entry
        points resolve, are deps + engine satisfied?

        Loads the plugin's modules in isolation, resolves every
        declared target, then unloads — nothing is *called*. Returns a
        plain dict the host (or CLI) can render.
        """
        from .loader import load_plugin, unload_plugin
        from ..version import __version__

        plugin = self.get(name, version)
        manifest = plugin.manifest
        report: dict[str, Any] = {
            "name": plugin.name,
            "version": plugin.version,
            "enabled": plugin.enabled,
            "manifest_ok": True,
            "loadable": False,
            "load_error": None,
            "entry_points": {},
            "lifecycle": {},
            "hooks": [h.name for h in manifest.hooks],
            "contributes": manifest.contribution_summary(),
            "dependencies": [],
            "engine": {
                "requires": manifest.requires_devon or "any",
                "running": __version__,
                "ok": satisfies_version(__version__,
                                       manifest.requires_devon),
            },
        }
        for dep in manifest.dependencies:
            rows = self.db.execute(
                "SELECT version FROM plugin_packages WHERE name=?",
                (dep.name,)).fetchall()
            installed = [r["version"] for r in rows]
            report["dependencies"].append({
                "name": dep.name,
                "spec": dep.spec or "any",
                "installed": installed,
                "satisfied": any(dep.satisfied_by(v) for v in installed),
            })
        loaded = None
        try:
            loaded = load_plugin(manifest, plugin.path)
            report["loadable"] = True
            for key, target in manifest.entry_points.items():
                try:
                    loaded.resolve(target)
                    report["entry_points"][key] = {"ok": True, "target": target}
                except LoadError as exc:
                    report["entry_points"][key] = {
                        "ok": False, "target": target, "error": str(exc)}
            for key, target in manifest.lifecycle.items():
                try:
                    loaded.resolve(target)
                    report["lifecycle"][key] = {"ok": True, "target": target}
                except LoadError as exc:
                    report["lifecycle"][key] = {
                        "ok": False, "target": target, "error": str(exc)}
            for hook in manifest.hooks:
                try:
                    loaded.resolve(hook.entry)
                except LoadError as exc:
                    report.setdefault("hook_errors", {})[hook.name] = str(exc)
        except LoadError as exc:
            report["load_error"] = str(exc)
        finally:
            if loaded is not None:
                unload_plugin(loaded)
        return report

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
        self._run_lifecycle(plugin, "on_enable", fail_fast=False)
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
        self._run_lifecycle(plugin, "on_disable", fail_fast=False)
        return plugin

    def remove(self, name: str, version: str = "") -> None:
        plugin = self.get(name, version)
        # Uninstall hook runs while the files are still on disk; a
        # broken hook is captured, never blocks removal.
        self._run_lifecycle(plugin, "on_uninstall", fail_fast=False)
        self.db.execute(
            "DELETE FROM plugin_packages WHERE name=? AND version=?",
            (plugin.name, plugin.version))
        path = Path(plugin.path)
        if path.exists() and path.is_relative_to(self.home):
            shutil.rmtree(path, ignore_errors=True)
        if plugin.manifest.purge_data_on_remove:
            from .wiring import purge_plugin_data
            purged = purge_plugin_data(self.db, plugin.name)
            _log.info("purged %d kv rows for removed plugin %s",
                      purged, plugin.name)
        _log.info("removed plugin %s %s", plugin.name, plugin.version)
        _emit("plugin.removed", {"name": plugin.name,
                                 "version": plugin.version})
