"""Update packages: build, read and verify them.

A package is a zip holding ``manifest.json`` plus the component's
files. Verification — the "verify" half of verify-then-install,
ported from upstream ``crates/native/src/install.rs`` ``download()``
(a release with no published checksum "can't be verified and wasn't
installed") and ``published_checksum`` (a checksum list that exists
but can't be fetched is an error, never a downgrade of the check) —
runs in this order:

1. the zip must open and hold a valid manifest (format
   ``devon-package/1`` with component, version, source and a file list);
2. every listed file must be present in the zip with matching SHA-256
   and size — a mismatch names the file;
3. the zip must hold no file the manifest does not list, and no path
   may escape the component directory;
4. the running Devon version must sit inside the manifest's
   ``devon_min``/``devon_max`` bounds;
5. the catalog's trust rules must pass for (component, source, hash);
6. a manifest attestation, when present, must match the catalog's
   pinned attestation for that component and source.

A package that fails any step is refused with the failing step named.
Nothing here installs anything — that is the pipeline's job.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import Catalog
from .errors import VerificationError
from .version import Version

__all__ = [
    "ManifestFile",
    "UpdatePackage",
    "VerificationVerdict",
    "build_package",
    "read_package",
    "verify_package",
]

MANIFEST_NAME = "manifest.json"
PACKAGE_FORMAT = "devon-package/1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ManifestFile:
    """One file a package carries: its path, hash and size."""

    path: str
    sha256: str
    size: int


@dataclass
class UpdatePackage:
    """A parsed update package: manifest plus the zip it came from."""

    path: Path
    component: str
    version: Version
    source: str
    devon_min: str
    devon_max: str
    files: list[ManifestFile]
    attestation: str = ""
    restart_required: bool = False
    package_sha256: str = ""

    @property
    def component_dir(self) -> Path:
        """The component's directory relative to an install root."""
        return Path(*self.component.split("."))


def _check_file_path(path: str) -> None:
    if not path or path.startswith("/") or path.startswith("\\"):
        raise VerificationError(
            f"file path outside the component directory: {path!r}")
    parts = Path(path).parts
    if ".." in parts or any(part in ("", ".") for part in parts):
        raise VerificationError(
            f"file path outside the component directory: {path!r}")


def build_package(path: str | Path, *, component: str, version: str,
                  source: str, files: dict[str, bytes],
                  devon_min: str = "0.1.0", devon_max: str = "",
                  attestation: str = "",
                  restart_required: bool = False) -> Path:
    """Write a package zip for ``component`` and return its path.

    ``files`` maps a path inside the component directory to its bytes.
    A path that escapes the component directory is refused here, at
    build time, not at install time.
    """
    for file_path in files:
        _check_file_path(file_path)
    Version.parse(version)
    if devon_min:
        Version.parse(devon_min)
    if devon_max:
        Version.parse(devon_max)
    manifest_files = [
        {"path": file_path, "sha256": _sha(data), "size": len(data)}
        for file_path, data in sorted(files.items())
    ]
    manifest = {
        "format": PACKAGE_FORMAT,
        "component": component,
        "version": version,
        "source": source,
        "devon_min": devon_min,
        "devon_max": devon_max,
        "files": manifest_files,
        "attestation": attestation,
        "restart_required": bool(restart_required),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME,
                    json.dumps(manifest, indent=2, sort_keys=True))
        for file_path, data in sorted(files.items()):
            zf.writestr(file_path, data)
    return target


def read_package(path: str | Path) -> UpdatePackage:
    """Parse a package zip or raise VerificationError naming the fault."""
    target = Path(path)
    if not target.is_file():
        raise VerificationError(f"package not found: {target}")
    try:
        with zipfile.ZipFile(target) as zf:
            names = zf.namelist()
            if MANIFEST_NAME not in names:
                raise VerificationError(
                    "package has no manifest.json — it is not an update "
                    "package")
            try:
                manifest = json.loads(
                    zf.read(MANIFEST_NAME).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise VerificationError(
                    f"package manifest is not valid JSON: {exc}") from exc
    except zipfile.BadZipFile as exc:
        raise VerificationError(
            f"package is not a valid zip: {target}") from exc
    if not isinstance(manifest, dict):
        raise VerificationError("package manifest is not a JSON object")
    if manifest.get("format") != PACKAGE_FORMAT:
        raise VerificationError(
            "package manifest has an unsupported format: "
            f"{manifest.get('format')!r} (want {PACKAGE_FORMAT!r})")
    for key in ("component", "version", "source", "files"):
        if key not in manifest:
            raise VerificationError(
                f"package manifest is missing {key!r}")
    if not isinstance(manifest["files"], list):
        raise VerificationError("package manifest 'files' is not a list")
    files = []
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not all(
                k in entry for k in ("path", "sha256", "size")):
            raise VerificationError(
                "package manifest has a malformed file entry")
        _check_file_path(str(entry["path"]))
        files.append(ManifestFile(path=str(entry["path"]),
                                  sha256=str(entry["sha256"]),
                                  size=int(entry["size"])))
    try:
        version = Version.parse(str(manifest["version"]))
    except ValueError as exc:
        raise VerificationError(
            f"package manifest has an invalid version: {exc}") from exc
    return UpdatePackage(
        path=target,
        component=str(manifest["component"]),
        version=version,
        source=str(manifest["source"]),
        devon_min=str(manifest.get("devon_min") or ""),
        devon_max=str(manifest.get("devon_max") or ""),
        files=files,
        attestation=str(manifest.get("attestation") or ""),
        restart_required=bool(manifest.get("restart_required", False)),
        package_sha256=_sha(target.read_bytes()),
    )


@dataclass
class VerificationVerdict:
    """The answer to one verification: ok, and every failing check."""

    ok: bool
    failures: list[str] = field(default_factory=list)
    package_sha256: str = ""


def verify_package(pkg: UpdatePackage, *, catalog: Catalog,
                   devon_version: str) -> VerificationVerdict:
    """Verify ``pkg`` against its manifest, the running Devon and trust."""
    failures: list[str] = []
    try:
        running = Version.parse(devon_version)
    except ValueError as exc:
        return VerificationVerdict(
            False, [f"running Devon version is not a version: {exc}"],
            pkg.package_sha256)
    with zipfile.ZipFile(pkg.path) as zf:
        names = set(zf.namelist()) - {MANIFEST_NAME}
        listed = {entry.path for entry in pkg.files}
        for extra in sorted(names - listed):
            failures.append(
                f"unexpected file in package (not in the manifest): {extra}")
        for entry in pkg.files:
            if entry.path not in names:
                failures.append(
                    f"manifest lists {entry.path} but the package does "
                    "not hold it")
                continue
            data = zf.read(entry.path)
            if len(data) != entry.size:
                failures.append(
                    f"{entry.path}: size {len(data)} != manifest "
                    f"{entry.size}")
            if _sha(data) != entry.sha256:
                failures.append(
                    f"{entry.path}: sha256 mismatch — the file was "
                    "modified after the package was built")
    if pkg.devon_min:
        try:
            if running < Version.parse(pkg.devon_min):
                failures.append(
                    f"package needs Devon >= {pkg.devon_min}, running "
                    f"{running}")
        except ValueError as exc:
            failures.append(f"package devon_min is not a version: {exc}")
    if pkg.devon_max:
        try:
            if running > Version.parse(pkg.devon_max):
                failures.append(
                    f"package needs Devon <= {pkg.devon_max}, running "
                    f"{running}")
        except ValueError as exc:
            failures.append(f"package devon_max is not a version: {exc}")
    trust = catalog.check_trust(pkg.component, pkg.source,
                                pkg.package_sha256)
    if not trust.ok:
        failures.append(f"trust: {trust.reason}")
    if pkg.attestation:
        pinned = catalog.attestation_pins.get((pkg.component, pkg.source))
        if pinned is None:
            failures.append(
                "attestation: the package carries an attestation but the "
                f"catalog pins none for {pkg.component} from {pkg.source}")
        elif pinned != pkg.attestation:
            failures.append(
                "attestation: the package's attestation does not match "
                "the catalog pin")
    return VerificationVerdict(not failures, failures, pkg.package_sha256)
