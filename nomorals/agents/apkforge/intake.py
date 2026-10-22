"""ApkForge intake — land an APK or bundle, record exactly how it
arrived, never mutate the original.

Sources: an owner-uploaded file (a path the chat layer hands over),
a URL (``core/http.download``, resumable), or a local path.  Bundle
containers (.apks/.apkm/.xapk/.aab) are detected by *content*, their
parts extracted zip-slip-guarded into ``bundle/`` and inventoried —
in A1 they are studied, not yet merged/converted (that is A2's
bundletool work, and the inventory says so honestly).
"""

from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from .dissect import light_identity
from .project import Project, create_project, list_projects, sha256_file

__all__ = ["IntakeError", "detect_container", "land_file", "land_url",
           "MAX_BYTES"]

# Resource control, not a product limit: overridable per host.
MAX_BYTES = int(os.environ.get("NOMORALS_APKFORGE_MAX_MB", "2048")) * 1024 * 1024


class IntakeError(RuntimeError):
    """A designed intake failure — the message names the exact cause."""


def detect_container(path: str | Path) -> dict[str, Any]:
    """Sniff a zip-family artifact by content.

    Returns {kind, entries:[{name,size}], apk_entries:[names]} where
    kind is apk | bundle-apks | bundle-apkm | bundle-xapk | aab."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise IntakeError(f"{path.name} is not a ZIP/APK file")
    with zipfile.ZipFile(path) as zf:
        names = [i.filename for i in zf.infolist()]
        entries = [{"name": i.filename, "size": i.file_size}
                   for i in zf.infolist()]
    lower = [n.lower() for n in names]
    apk_entries = [n for n in names if n.lower().endswith(".apk")]
    kind = "apk"
    if "bundleconfig.pb" in lower and any(
            n.startswith("base/manifest/") for n in lower):
        kind = "aab"
    elif path.suffix.lower() == ".apks" or (
            len(apk_entries) > 1 and any(
                n.lower().startswith("splits/") for n in lower)):
        kind = "bundle-apks"
    elif "info.json" in lower and apk_entries:
        kind = "bundle-apkm"
    elif any(n.lower().endswith(".obb") for n in lower) or (
            "manifest.json" in lower and apk_entries):
        kind = "bundle-xapk"
    elif len(apk_entries) > 1:
        kind = "bundle-apks"
    return {"kind": kind, "entries": entries, "apk_entries": apk_entries}


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> list[str]:
    """Extract with zip-slip confinement: any entry that would land
    outside ``dest`` refuses the whole intake, by name."""
    dest = dest.resolve()
    written: list[str] = []
    for info in zf.infolist():
        name = info.filename
        if name.endswith("/"):
            continue
        target = (dest / name).resolve()
        if target != dest and dest not in target.parents:
            raise IntakeError(
                f"bundle entry {name!r} escapes the project dir — "
                "refusing the whole intake (zip-slip guard)")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        written.append(name)
    return written


def land_file(root: Path, source_path: str | Path, *,
              source_type: str = "upload",
              ref: str = "") -> tuple[Project, bool]:
    """Land a local artifact.  Returns (project, created).

    Dedupe is by content: landing the same bytes twice returns the
    existing project untouched (the original is immutable)."""
    src = Path(source_path).expanduser()
    if not src.exists():
        raise IntakeError(f"no such file: {src}")
    size = src.stat().st_size
    if size > MAX_BYTES:
        raise IntakeError(
            f"{src.name} is {size:,} bytes — over this host's intake "
            f"cap of {MAX_BYTES:,} (NOMORALS_APKFORGE_MAX_MB)")
    sha = sha256_file(src)
    for existing in list_projects(root):
        if existing.sha256 == sha:
            return existing, False
    sniff = detect_container(src)
    kind = sniff["kind"]
    ext = src.suffix.lower() or ".apk"
    original_name = "original" + ext if kind != "apk" else "original.apk"
    provenance = {
        "type": source_type,
        "ref": ref or str(src),
        "sha256": sha,
        "size": size,
        "fetched_at": time.time(),
        "original_filename": src.name,
    }
    proj = create_project(root, source_sha256=sha, kind=kind,
                          original_name=original_name,
                          provenance=provenance)
    shutil.copyfile(src, proj.original_path)

    inventory: dict[str, Any] = {"entries": sniff["entries"],
                                 "apk_entries": sniff["apk_entries"]}
    if kind != "apk":
        bundle_dir = proj.dir / "bundle"
        bundle_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(proj.original_path) as zf:
                _safe_extract(zf, bundle_dir)
        except IntakeError:
            # a refused bundle leaves nothing behind — the intake
            # either lands whole or not at all
            import shutil as _shutil

            _shutil.rmtree(proj.dir, ignore_errors=True)
            raise
        parts: list[dict[str, Any]] = []
        for name in sniff["apk_entries"]:
            part_path = bundle_dir / name
            ident = light_identity(part_path) if part_path.exists() else {}
            parts.append({"name": name,
                          "size": part_path.stat().st_size
                          if part_path.exists() else 0,
                          "sha256": sha256_file(part_path)
                          if part_path.exists() else "",
                          "package": ident.get("package", ""),
                          "version_name": ident.get("version_name", ""),
                          "version_code": ident.get("version_code")})
        inventory["parts"] = parts
        inventory["note"] = (
            "bundle container — parts inventoried; merging / "
            "device sets / AAB conversion run through the A2 "
            "splits ops (bundletool / APKEditor)")
        proj.record["package"] = ""
        proj.record["title"] = src.name
    else:
        ident = light_identity(proj.original_path)
        proj.record["package"] = ident.get("package", "")
        proj.record["title"] = (ident.get("app_name")
                                or ident.get("package") or src.name)
        proj.record["version_name"] = ident.get("version_name", "")
        proj.record["version_code"] = ident.get("version_code")
        proj.record["cert_sha256"] = ident.get("cert_sha256", "")
    proj.record["inventory"] = inventory
    proj.save()
    proj.journal("intake",
                 f"{source_type}: {provenance['ref']} "
                 f"({size:,} bytes, sha256 {sha[:16]}…)",
                 kind=kind)
    return proj, True


def land_url(root: Path, url: str) -> tuple[Project, bool]:
    """Download through the shared HTTP organ, then land the file."""
    from ...core.http import download

    tmp_dir = root / "inbox"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = download(url, tmp_dir)
    try:
        return land_file(root, dest, source_type="url", ref=url)
    finally:
        try:
            Path(dest).unlink(missing_ok=True)
        except OSError:
            pass


def inventory_json(project: Project) -> str:
    return json.dumps(project.record.get("inventory") or {}, indent=1)
