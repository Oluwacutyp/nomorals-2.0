"""Prune the stocked font pack down to the families renderers
actually ask for (gapfix 164).

The pack in Devon's data estate holds ~2,000 families (~2.5 GB)
plus a build-dir duplicate, while the renderers resolve from a
handful of policy families and owner pins. The pruner removes
only what nothing asks for:

* the referenced set = every family named in
  ``CONTEXT_POLICIES`` + every owner-pinned family (all scopes)
  + DejaVu Sans, the layout organ's no-context measurement
  fallback;
* scope = the stocked PACK tier only (``<NM home>/fonts/library``)
  — never the vendored core subset, never an owner-local
  ``NM_FONT_DIRS`` dir, never ``C:\\Windows\\Fonts`` or any
  system dir, never a path outside the pack root (every deletion
  is checked to sit inside it);
* the build dir (``library.build``) goes only when the stocked
  pack is whole (``kind == "full"``) — while the pack is partial
  it holds resumable build progress and is kept;
* the manifest is rewritten to match (counts, bytes, the
  per-family issue lists filtered to the kept set) and the pack
  is re-verified on a sample afterwards.

Nothing here raises: a missing manifest, an unreadable pin
store or a failed deletion is reported in the returned dict,
never thrown — the update's self-clear and the chat verb both
treat the report as data.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any

from ...core.style_pins import get_pin_store
from .fontlib import MANIFEST_NAME, estate_library_dir, normalize_family
from .fontpack import estate_build_dir, verify_pack
from .policy import CONTEXT_POLICIES

_log = logging.getLogger(__name__)

#: The layout organ measures with DejaVu Sans when no context
#: is given (pinned by the craftfonts lane), so it is referenced
#: even though no policy row names it.
_MEASUREMENT_FALLBACK = "DejaVu Sans"

__all__ = ["prune_library", "prune_report", "referenced_families"]


def referenced_families() -> set[str]:
    """Families some renderer asks for by name: every policy
    family, every owner-pinned family (all scopes, global
    included), and the measurement fallback."""
    refs = {_MEASUREMENT_FALLBACK}
    for policy in CONTEXT_POLICIES.values():
        refs.update(policy.families)
    try:
        store = get_pin_store()
        scopes = list(store.scopes())
        if "" not in scopes:
            scopes.append("")
        for scope in scopes:
            pin = store.get(scope)
            if pin.font_family:
                refs.add(pin.font_family)
    except Exception:  # noqa: BLE001 - pins are best-effort here
        _log.debug("pinned families could not be read",
                   exc_info=True)
    return refs


def _read_pack_manifest(lib_dir: Path) -> dict[str, Any] | None:
    path = lib_dir / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _family_files(family: dict[str, Any]) -> list[str]:
    """The pack-relative paths one manifest family owns: its
    faces plus its licence file."""
    rels = [str(face.get("file") or "")
            for face in family.get("faces") or []]
    licence = str(family.get("license_file") or "")
    if licence:
        rels.append(licence)
    return [rel for rel in rels if rel]


def _dir_size(root: Path) -> int:
    total = 0
    if not root.is_dir():
        return 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _inside(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def prune_report(home: Any = None) -> dict[str, Any]:
    """The honest accounting, without deleting anything: which
    pack families are prunable, how many files and bytes that
    is, and whether the build dir is a leftover."""
    lib_dir = estate_library_dir(home)
    build_dir = estate_build_dir(home)
    report: dict[str, Any] = {
        "pack_dir": str(lib_dir), "pack_present": False,
        "pack_kind": "", "pack_bytes": 0,
        "build_dir": str(build_dir), "build_bytes": 0,
        "build_pending": False,
        "referenced_families": sorted(referenced_families()),
        "prunable_families": [], "prunable_files": 0,
        "prunable_bytes": 0, "kept_families": 0,
        "dry_run": True, "removed_families": [],
        "removed_files": 0, "bytes_freed": 0,
        "build_dir_removed": False, "build_bytes_freed": 0,
        "verified": False, "error": "",
    }
    manifest = _read_pack_manifest(lib_dir)
    if manifest is None:
        report["error"] = (
            f"no stocked pack manifest at {lib_dir / MANIFEST_NAME}"
            " — nothing to prune")
        return report
    report["pack_present"] = True
    report["pack_kind"] = str(manifest.get("kind") or "")
    report["build_bytes"] = _dir_size(build_dir)
    # A build is pending while the stocked pack is not whole:
    # the build dir then holds resumable progress, never waste.
    report["build_pending"] = report["pack_kind"] != "full"
    refs = {normalize_family(name)
            for name in report["referenced_families"]}
    kept = 0
    prunable: list[str] = []
    prunable_files = 0
    prunable_bytes = 0
    pack_bytes = 0
    for family in manifest.get("families") or []:
        name = str(family.get("name") or "")
        faces = family.get("faces") or []
        pack_bytes += sum(int(face.get("bytes") or 0)
                          for face in faces)
        if normalize_family(name) in refs:
            kept += 1
            continue
        prunable.append(name)
        for rel in _family_files(family):
            prunable_files += 1
            try:
                prunable_bytes += (lib_dir / rel).stat().st_size
            except OSError:
                continue
    report["prunable_families"] = sorted(prunable)
    report["prunable_files"] = prunable_files
    report["prunable_bytes"] = prunable_bytes
    report["kept_families"] = kept
    report["pack_bytes"] = pack_bytes
    report["removed_families"] = sorted(prunable)
    report["removed_files"] = prunable_files
    return report


def prune_library(*, home: Any = None, dry_run: bool = False
                  ) -> dict[str, Any]:
    """Prune unreferenced families from the stocked pack.

    Dry-run lists exactly what would go and removes nothing
    (``bytes_freed`` stays 0; the would-be bytes are in
    ``prunable_bytes``). A real run deletes the prunable
    families' files, removes their now-empty dirs, rewrites the
    manifest to match, removes the build dir when the pack is
    whole, and re-verifies the pack on a sample. Never raises:
    failures land in ``error`` / ``verified``.
    """
    report = prune_report(home)
    report["dry_run"] = bool(dry_run)
    if report["error"]:
        return report
    lib_dir = Path(report["pack_dir"])
    build_dir = Path(report["build_dir"])
    manifest = _read_pack_manifest(lib_dir)
    if manifest is None:
        report["error"] = "pack manifest vanished mid-prune"
        return report
    refs = {normalize_family(name)
            for name in report["referenced_families"]}
    removed: list[str] = []
    removed_files = 0
    freed = 0
    kept_families: list[dict[str, Any]] = []
    for family in manifest.get("families") or []:
        name = str(family.get("name") or "")
        if normalize_family(name) in refs:
            kept_families.append(family)
            continue
        removed.append(name)
        for rel in _family_files(family):
            target = lib_dir / rel
            if not _inside(lib_dir, target):
                _log.warning("prune refuses a path outside the "
                             "pack: %s", rel)
                continue
            if dry_run:
                removed_files += 1
                with suppress(OSError):
                    freed += target.stat().st_size
                continue
            try:
                freed += target.stat().st_size
                target.unlink()
                removed_files += 1
            except FileNotFoundError:
                continue
            except OSError as exc:
                _log.warning("could not prune %s: %s", target, exc)
        if not dry_run:
            fam_dir = str(family.get("dir") or "")
            if fam_dir:
                # Non-empty: something else lives there.
                with suppress(OSError):
                    os.rmdir(lib_dir / fam_dir)
    if dry_run:
        report["removed_families"] = sorted(removed)
        report["removed_files"] = removed_files
        report["bytes_freed"] = 0
        return report
    report["removed_families"] = sorted(removed)
    report["removed_files"] = removed_files
    report["bytes_freed"] = freed
    if removed:
        kept_keys = {normalize_family(
            str(fam.get("name") or "")) for fam in kept_families}
        new_manifest = dict(manifest)
        new_manifest["families"] = kept_families
        new_manifest["family_count"] = len(kept_families)
        new_manifest["file_count"] = sum(
            len(fam.get("faces") or []) for fam in kept_families)
        new_manifest["total_bytes"] = sum(
            int(face.get("bytes") or 0)
            for fam in kept_families
            for face in fam.get("faces") or [])
        for key in ("families_without_license_file",
                    "metadata_less_families"):
            values = new_manifest.get(key)
            if isinstance(values, list):
                new_manifest[key] = sorted(
                    name for name in values
                    if normalize_family(str(name)) in kept_keys)
        unlisted = new_manifest.get("unlisted_faces")
        if isinstance(unlisted, dict):
            new_manifest["unlisted_faces"] = {
                key: value for key, value in unlisted.items()
                if normalize_family(str(key)) in kept_keys}
        report["pack_bytes"] = int(new_manifest["total_bytes"])
        tmp = lib_dir / (MANIFEST_NAME + ".part")
        try:
            tmp.write_text(json.dumps(new_manifest, indent=1) + "\n",
                           encoding="utf-8")
            os.replace(tmp, lib_dir / MANIFEST_NAME)
        except OSError as exc:
            report["error"] = f"manifest rewrite failed: {exc}"
            _log.warning("prune manifest rewrite failed: %s", exc)
    if not report["build_pending"] and report["build_bytes"] > 0:
        # The pack is whole, so the build dir is a leftover
        # duplicate of it — never live build progress.
        try:
            shutil.rmtree(build_dir)
            report["build_dir_removed"] = True
            report["build_bytes_freed"] = int(report["build_bytes"])
        except OSError as exc:
            _log.warning("could not remove the build dir %s: %s",
                         build_dir, exc)
    try:
        check = verify_pack(lib_dir, sample=5)
        report["verified"] = bool(check.get("ok"))
        if not report["verified"]:
            report["error"] = (
                "pack failed re-verification after prune: "
                + "; ".join(str(bad) for bad in check.get("bad") or []))
    except Exception as exc:  # noqa: BLE001 - report, never raise
        report["verified"] = False
        report["error"] = f"pack re-verification failed: {exc}"
    _log.info("font pack prune: removed %d families (%d files, "
              "%d bytes), build dir %s, verified=%s",
              len(report["removed_families"]),
              report["removed_files"], report["bytes_freed"],
              "removed" if report["build_dir_removed"] else "kept",
              report["verified"])
    return report
