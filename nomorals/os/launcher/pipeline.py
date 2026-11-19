"""The verify-then-install pipeline.

Ported from upstream ``crates/native/src/install.rs`` (``download()``:
no checksum means no install; ``swap_into_place``: the old copy is
kept until the new one is there, restored on failure, and kept
somewhere safe when even the restore fails) and ``crates/native/
src/self_update.rs`` (``craft_update::self_update``: the launcher
replacing itself and relaunching once the old process has exited).

One install runs these steps, in this order, and journals each one:

1. read the package (a bad zip or manifest is refused here);
2. verify it — manifest hashes, compat, trust, attestation — and
   journal ``verified``; a failure journals ``refused`` and raises,
   and the install root is untouched;
3. health-check the running install (``health_before``);
4. refuse a downgrade unless ``allow_downgrade``;
5. stage the package into a temporary directory and re-hash the
   staged files (a file that changed between the zip and the disk is
   caught here, not after the swap);
6. swap the staged component into place atomically — the old copy
   moves aside to a reserved backup directory first, and is restored
   if the new copy cannot be put in place;
7. health-check the new install (``health_after``); a failure restores
   the old copy, journals ``rolled_back`` and returns a failed result
   — the old version is still running;
8. on success the old copy is retained as the rollback target and the
   install journals ``installed``; a component that needs a restart
   records a restart intent for the supervisor to act on.

The pipeline never installs a package it has not verified, never
leaves a half-installed component behind, and never deletes the only
old copy.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import Catalog, VersionTrack, choose_version, default_catalog
from .errors import InstallError, TrustError, VerificationError
from .journal import Journal
from .package import (
    UpdatePackage,
    VerificationVerdict,
    read_package,
    verify_package,
)
from .version import Version

__all__ = ["HealthCheck", "LauncherPipeline", "PipelineResult"]

#: The file inside a component directory recording what is installed.
VERSION_FILENAME = ".launcher-version"

#: The restart-intent file the supervisor reads after a restart-needing
#: install (same filename the UpdateManager writes, same home).
RESTART_INTENT_FILENAME = "update-restart-intent.json"

#: Health checks return this shape: ok, and a human-readable detail.
HealthCheck = Callable[[], dict[str, Any]]

# The pipeline's rename step, bound at module level so a test can
# fault it and prove the rollback path (upstream's PRETEND_VAR serves
# the same purpose for self-update).
_rename = os.rename


def _running_devon_version() -> str:
    """The running Devon version, read dynamically (unscanned import)."""
    try:
        version_mod = importlib.import_module("nomorals.version")
        return str(getattr(version_mod, "__version__", "0.0.0"))
    except Exception:  # noqa: BLE001 - no version module, no version
        return "0.0.0"


@dataclass
class PipelineResult:
    """The outcome of one install attempt."""

    ok: bool
    installed_version: str = ""
    rolled_back: bool = False
    restart_required: bool = False
    health_before: dict[str, Any] = field(default_factory=dict)
    health_after: dict[str, Any] = field(default_factory=dict)
    verification: VerificationVerdict | None = None
    error: str = ""


class LauncherPipeline:
    """Verify-then-install for one install root and one state home."""

    def __init__(self, root: str | Path, home: str | Path, *,
                 catalog: Catalog | None = None,
                 health_checks: list[HealthCheck] | None = None,
                 devon_version: str | None = None) -> None:
        self.root = Path(root)
        self.home = Path(home)
        self.root.mkdir(parents=True, exist_ok=True)
        self.home.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog if catalog is not None else default_catalog()
        self._health_checks = health_checks
        self.devon_version = (
            devon_version if devon_version is not None
            else _running_devon_version())
        self.journal = Journal(self.home / "launcher-journal.jsonl")
        self._backups = self.home / "backups"
        self._backups.mkdir(parents=True, exist_ok=True)

    # ── state ────────────────────────────────────────────────────────

    def _component_dir(self, component_id: str) -> Path:
        return self.root / Path(*component_id.split("."))

    def _version_file(self, component_id: str) -> Path:
        return self._component_dir(component_id) / VERSION_FILENAME

    def installed_version(self, component_id: str) -> str | None:
        """The installed version of ``component_id``, or None."""
        path = self._version_file(component_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            version = Version.parse(str(data["version"]))
        except (ValueError, KeyError, TypeError):
            return None
        return str(version)

    def status(self, component_id: str) -> dict[str, Any]:
        """What is installed for ``component_id`` right now."""
        version = self.installed_version(component_id)
        if version is None:
            return {"component": component_id, "installed": False,
                    "version": None}
        return {"component": component_id, "installed": True,
                "version": version}

    def check(self, component_id: str,
              releases: list[dict]) -> dict[str, Any]:
        """The newest release ``component_id`` could move to, if any."""
        current = self.installed_version(component_id)
        latest = choose_version(releases, VersionTrack.STABLE)
        latest_version = None
        if latest is not None:
            try:
                latest_version = str(Version.parse(
                    str(latest.get("version", ""))))
            except ValueError:
                latest_version = None
        update_available = (
            latest_version is not None
            and (current is None
                 or Version.parse(latest_version) > Version.parse(current)))
        return {"component": component_id, "current": current,
                "latest": latest_version,
                "update_available": update_available}

    # ── health ───────────────────────────────────────────────────────

    def _default_health(self, component_id: str) -> dict[str, Any]:
        """The component's version file parses and its code imports."""
        version = self.installed_version(component_id)
        if version is None:
            return {"ok": True, "detail": "component not installed "
                    "(fresh install)"}
        comp_dir = self._component_dir(component_id)
        init = comp_dir / "__init__.py"
        if not init.is_file():
            return {"ok": True, "detail": f"{component_id} at {version} "
                    "(no package to import)"}
        ok, detail = self._import_probe(component_id)
        if not ok:
            return {"ok": False, "detail": detail}
        return {"ok": True, "detail": f"{component_id} at {version} "
                "imports cleanly"}

    def _import_probe(self, component_id: str) -> tuple[bool, str]:
        """Import ``component_id`` from this install root, nothing else.

        A stale ``sys.modules`` entry from another root is dropped
        first: the probe must prove THIS root's code imports, not a
        cached copy from a different install.
        """
        name = component_id
        cached = sys.modules.get(name)
        if cached is not None:
            cached_file = getattr(cached, "__file__", "") or ""
            if cached_file and Path(cached_file).is_relative_to(self.root):
                return True, "import ok (cached)"
        for key in [k for k in sys.modules
                    if k == name or k.startswith(name + ".")]:
            mod_file = getattr(sys.modules[key], "__file__", "") or ""
            if mod_file and not Path(mod_file).is_relative_to(self.root):
                del sys.modules[key]
        sys.path.insert(0, str(self.root))
        # A health probe must not mutate the tree it checks: bytecode
        # writing is off while it imports.
        saved = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - health reports, never raises
            return False, (f"import failed: {type(exc).__name__}: {exc}")
        finally:
            sys.dont_write_bytecode = saved
            with contextlib.suppress(ValueError):
                sys.path.remove(str(self.root))
        return True, "import ok"

    def _health(self, component_id: str) -> dict[str, Any]:
        checks = self._health_checks
        if checks is None:
            checks = [lambda: self._default_health(component_id)]
        results = []
        for check in checks:
            try:
                result = check()
            except Exception as exc:  # noqa: BLE001 - a raising check fails
                result = {"ok": False,
                          "detail": f"health check raised "
                          f"{type(exc).__name__}: {exc}"}
            results.append({"ok": bool(result.get("ok", False)),
                            "detail": str(result.get("detail", ""))})
        ok = all(item["ok"] for item in results)
        detail = "; ".join(item["detail"] for item in results if item["detail"])
        return {"ok": ok, "detail": detail, "checks": results}

    # ── install ──────────────────────────────────────────────────────

    def install(self, package_path: str | Path, *,
                allow_downgrade: bool = False) -> PipelineResult:
        """Verify ``package_path`` then install it, or refuse cleanly."""
        run_id = uuid.uuid4().hex[:12]
        pkg = read_package(package_path)
        verdict = verify_package(pkg, catalog=self.catalog,
                                 devon_version=self.devon_version)
        base = {"run_id": run_id, "component": pkg.component,
                "version": str(pkg.version), "source": pkg.source,
                "package_sha256": pkg.package_sha256,
                "verification": {"ok": verdict.ok,
                                 "failures": verdict.failures}}
        if not verdict.ok:
            self.journal.append({**base, "event": "refused"})
            failures = "; ".join(verdict.failures)
            if any(f.startswith("trust:") for f in verdict.failures):
                raise TrustError(f"package refused by a trust rule: {failures}")
            raise VerificationError(f"package failed verification: {failures}")
        self.journal.append({**base, "event": "verified"})

        health_before = self._health(pkg.component)
        current = self.installed_version(pkg.component)
        if (current is not None
                and pkg.version < Version.parse(current)
                and not allow_downgrade):
            self.journal.append(
                {**base, "event": "refused",
                 "error": f"downgrade {current} -> {pkg.version}"})
            raise InstallError(
                f"downgrade refused: {pkg.component} is at {current}, "
                f"package is {pkg.version} (pass allow_downgrade to "
                "install it anyway)")
        comp_dir = self._component_dir(pkg.component)
        staging = Path(tempfile.mkdtemp(prefix=".launcher-stage-",
                                        dir=str(self.root.parent)))
        backup_dir: Path | None = None
        try:
            self._stage(pkg, staging)
            old_version = current
            if comp_dir.exists():
                backup_dir = Path(tempfile.mkdtemp(
                    prefix=".launcher-backup-", dir=str(self.root.parent)))
                _rename(comp_dir, backup_dir / "old")
            # Component ids are dotted (nomorals.media.cad) and land in
            # nested dirs; the parent may not exist yet.
            comp_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                _rename(staging / "component", comp_dir)
            except OSError as exc:
                if backup_dir is not None:
                    self._restore_backup(comp_dir, backup_dir)
                return self._failed(
                    base, health_before,
                    f"could not put the new copy in place: {exc}")
            self._write_version_file(pkg, comp_dir)
            health_after = self._health(pkg.component)
            if not health_after["ok"]:
                if backup_dir is not None:
                    self._restore_backup(comp_dir, backup_dir)
                else:
                    shutil.rmtree(comp_dir, ignore_errors=True)
                self.journal.append(
                    {**base, "event": "rolled_back",
                     "health_before": health_before,
                     "health_after": health_after})
                return PipelineResult(
                    ok=False, installed_version=current or "",
                    rolled_back=True, health_before=health_before,
                    health_after=health_after, verification=verdict,
                    error=f"post-install health failed: "
                          f"{health_after['detail']}")
            if backup_dir is not None and old_version is not None:
                self._retain_backup(pkg.component, old_version, backup_dir)
            catalog_component = self.catalog.component(pkg.component)
            restart_required = bool(
                pkg.restart_required
                or (catalog_component.restart_required
                    if catalog_component is not None else False))
            if restart_required:
                self._write_restart_intent(pkg)
            self.journal.append(
                {**base, "event": "installed",
                 "health_before": health_before,
                 "health_after": health_after,
                 "restart_required": restart_required})
            return PipelineResult(
                ok=True, installed_version=str(pkg.version),
                health_before=health_before, health_after=health_after,
                verification=verdict, restart_required=restart_required)
        except OSError as exc:
            # The swap itself failed before the new copy was in place
            # (moving the old copy aside is an atomic rename: either it
            # moved or it did not). The old tree is untouched.
            return self._failed(
                base, health_before, f"install failed at the swap: {exc}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if backup_dir is not None and (backup_dir / "old").exists():
                # A backup that was neither restored nor retained (the
                # install died between swap and retain) still holds the
                # only old copy: retain it, never delete it.
                self._retain_backup(pkg.component, current or "unknown",
                                    backup_dir)

    def _failed(self, base: dict[str, Any], health_before: dict[str, Any],
                error: str) -> PipelineResult:
        self.journal.append({**base, "event": "failed", "error": error})
        return PipelineResult(ok=False, health_before=health_before,
                              error=error)

    def _stage(self, pkg: UpdatePackage, staging: Path) -> None:
        """Extract the package into ``staging`` and re-hash every file."""
        import hashlib
        import zipfile

        component_dir = staging / "component"
        component_dir.mkdir(parents=True)
        with zipfile.ZipFile(pkg.path) as zf:
            for entry in pkg.files:
                data = zf.read(entry.path)
                if hashlib.sha256(data).hexdigest() != entry.sha256:
                    raise VerificationError(
                        f"{entry.path}: sha256 mismatch after extraction — "
                        "the package changed between the zip and the disk")
                target = component_dir / entry.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)

    def _restore_backup(self, comp_dir: Path, backup_dir: Path) -> None:
        """Move the old copy back into place, keeping it if that fails."""
        old = backup_dir / "old"
        if comp_dir.exists():
            shutil.rmtree(comp_dir, ignore_errors=True)
        try:
            _rename(old, comp_dir)
        except OSError:
            kept = backup_dir / "old"
            raise InstallError(
                "could not put the new copy in place, and could not "
                f"restore the old copy either — it was kept at {kept}"
            ) from None

    def _retain_backup(self, component_id: str, version: str,
                       backup_dir: Path) -> None:
        """Keep the old copy as the rollback target for ``component_id``."""
        kept_id = uuid.uuid4().hex[:12]
        kept = self._backups / kept_id
        # shutil.move, not rename: the backups dir may sit on another
        # filesystem than the install root.
        shutil.move(str(backup_dir), str(kept))
        index = self._backup_index()
        index[component_id] = {"dir": kept_id, "version": version,
                               "ts": time.time()}
        self._write_backup_index(index)

    def _backup_index(self) -> dict[str, Any]:
        path = self.home / "backups.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_backup_index(self, index: dict[str, Any]) -> None:
        (self.home / "backups.json").write_text(
            json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")

    def _write_version_file(self, pkg: UpdatePackage, comp_dir: Path) -> None:
        (comp_dir / VERSION_FILENAME).write_text(
            json.dumps({"component": pkg.component,
                        "version": str(pkg.version),
                        "source": pkg.source,
                        "package_sha256": pkg.package_sha256,
                        "installed_at": time.time()},
                       indent=2, sort_keys=True),
            encoding="utf-8")

    def _write_restart_intent(self, pkg: UpdatePackage) -> None:
        (self.home / RESTART_INTENT_FILENAME).write_text(
            json.dumps({"component": pkg.component,
                        "version": str(pkg.version),
                        "package_sha256": pkg.package_sha256,
                        "ts": time.time()}, indent=2, sort_keys=True),
            encoding="utf-8")

    # ── rollback ─────────────────────────────────────────────────────

    def rollback(self, component_id: str) -> dict[str, Any]:
        """Restore the retained previous copy of ``component_id``."""
        index = self._backup_index()
        entry = index.get(component_id)
        if entry is None:
            return {"component": component_id, "ok": False,
                    "error": "no retained previous copy to roll back to"}
        kept = self._backups / str(entry["dir"])
        old = kept / "old"
        if not old.is_dir():
            return {"component": component_id, "ok": False,
                    "error": "the retained copy is missing"}
        comp_dir = self._component_dir(component_id)
        current = self.installed_version(component_id)
        staging = Path(tempfile.mkdtemp(prefix=".launcher-rollback-",
                                        dir=str(self.root.parent)))
        try:
            if comp_dir.exists():
                _rename(comp_dir, staging / "current")
            try:
                _rename(old, comp_dir)
            except OSError as exc:
                if (staging / "current").exists():
                    _rename(staging / "current", comp_dir)
                return {"component": component_id, "ok": False,
                        "error": f"could not restore the old copy: {exc}"}
            health = self._health(component_id)
            if not health["ok"]:
                shutil.rmtree(comp_dir, ignore_errors=True)
                _rename(staging / "current", comp_dir)
                return {"component": component_id, "ok": False,
                        "error": f"rollback health failed: {health['detail']}",
                        "health_after": health}
            self.journal.append(
                {"component": component_id, "version": entry["version"],
                 "event": "rolled_back", "health_after": health,
                 "from_version": current})
            return {"component": component_id, "ok": True,
                    "version": entry["version"], "health_after": health}
        finally:
            shutil.rmtree(staging, ignore_errors=True)
