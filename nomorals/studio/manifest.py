"""The package manifest, ``craft-extension.json``, read from an
untrusted archive.

A port of the rules in ``craft-extensions`` (``crates/ext-core/src/
manifest.rs``, dual MIT/Apache): the size is capped before anything
is parsed; unknown fields are ignored (so a newer publisher's extra
data never breaks an older host) but a package can never state its
own trust tier, because the type has no such field; permissions must
be empty in version 1; every file the package carries is listed with
its size and SHA-256, and the list must match what the kind needs.

Devon's addition, kept separate from the upstream shape: an
extension declares which Studio extension points it fills
(``extension_points``), and the host refuses a point it does not
have or one the package's kind cannot fill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .ids import AppId, ExtensionId, IdError
from .paths import validate_all
from .text import Lines, validate_text
from .version import Version, VersionError

__all__ = [
    "EXTENSION_POINTS",
    "MANIFEST_NAME",
    "MAX_APPS",
    "MAX_MANIFEST_BYTES",
    "PLUGIN_FILE",
    "SCHEMA",
    "THEME_FILE",
    "Compat",
    "FileEntry",
    "Kind",
    "Manifest",
    "ManifestError",
    "Provenance",
    "Publisher",
]

MANIFEST_NAME = "craft-extension.json"
MAX_MANIFEST_BYTES = 64 * 1024
SCHEMA = 1
THEME_FILE = "theme.json"
PLUGIN_FILE = "panel.json"
MAX_APPS = 32

#: The extension points Devon's Studio offers, by the kind that
#: may fill each one. A theme wears the Studio's look; a plug-in
#: adds a panel to a Studio page; a resource pack is data the
#: Studio can list (serving its files is a later lane).
EXTENSION_POINTS: dict[str, str] = {
    "studio.theme": "theme",
    "photo.panel": "plugin",
    "studio.resource": "resource",
}


class ManifestError(ValueError):
    """A manifest was refused, with the reason as the message."""


class Kind(str):
    """What a package is: theme, plug-in or resource pack."""

    THEME = "theme"
    PLUGIN = "plugin"
    RESOURCE = "resource"

    @classmethod
    def parse(cls, value: object) -> Kind:
        if value in ("theme", "plugin", "resource"):
            return cls(str(value))
        raise ManifestError(f"the package kind is not valid: {value!r}")

    def limits(self) -> tuple[int, int, int]:
        """(max files, max bytes per file, max bytes total)."""
        if self == "theme":
            return (1, 64 * 1024, 64 * 1024)
        if self == "plugin":
            return (1, 32 * 1024 * 1024, 32 * 1024 * 1024)
        return (512, 64 * 1024 * 1024, 256 * 1024 * 1024)


@dataclass(frozen=True)
class Publisher:
    """Who the package says made it. A claim, never proof."""

    name: str


@dataclass(frozen=True)
class Compat:
    """Oldest app version the package works with, and — for
    plug-ins — the interface version it was built for."""

    min_app: Version | None = None
    interface: int | None = None


@dataclass(frozen=True)
class FileEntry:
    """One file the package carries, with its SHA-256 and size."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Provenance:
    """Where the code came from: a source repo and exact commit."""

    source: str
    commit: str


@dataclass(frozen=True)
class Manifest:
    """A validated package manifest."""

    schema: int
    id: ExtensionId
    name: str
    version: Version
    kind: Kind
    publisher: Publisher
    description: str
    compat: dict
    files: tuple
    permissions: tuple
    provenance: Provenance | None
    extension_points: tuple = ()

    @classmethod
    def parse(cls, data: bytes) -> Manifest:
        """Parse and validate a manifest, or raise ManifestError."""
        if len(data) > MAX_MANIFEST_BYTES:
            raise ManifestError("the package description is too large")
        try:
            raw = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ManifestError(
                "the package description cannot be read") from None
        if not isinstance(raw, dict):
            raise ManifestError("the package description cannot be read")
        # Read the schema alone first, so a newer package is refused
        # for being newer, not for a field this host never heard of.
        schema = raw.get("schema")
        if schema != SCHEMA:
            raise ManifestError(
                "the package uses a format version "
                f"({schema}) this app does not know")
        try:
            ext_id = ExtensionId.new(str(raw.get("id", "")))
        except IdError as exc:
            raise ManifestError(
                f"the package id is not valid: {exc}") from None
        try:
            version = Version.parse(str(raw.get("version", "")))
        except VersionError as exc:
            raise ManifestError(str(exc)) from None
        kind = Kind.parse(raw.get("kind", ""))
        name = raw.get("name", "")
        description = raw.get("description", "") or ""
        publisher = raw.get("publisher") or {}
        compat = _parse_compat(raw.get("compat"), kind)
        files = _parse_files(raw.get("files"))
        permissions = raw.get("permissions") or []
        if not isinstance(permissions, list):
            raise ManifestError(
                "the package asks for permissions, which are not "
                "available yet")
        provenance = _parse_provenance(raw.get("provenance"))
        points = raw.get("extension_points") or []
        if not isinstance(points, list):
            raise ManifestError(
                "the package fills an extension point this Studio "
                "does not have")
        manifest = cls(
            schema=SCHEMA,
            id=ext_id,
            name=name if isinstance(name, str) else "",
            version=version,
            kind=kind,
            publisher=Publisher(
                name=publisher.get("name", "")
                if isinstance(publisher, dict) else ""),
            description=description if isinstance(description, str) else "",
            compat=compat,
            files=files,
            permissions=tuple(str(p) for p in permissions),
            provenance=provenance,
            extension_points=tuple(str(p) for p in points),
        )
        manifest._validate()
        return manifest

    def _validate(self) -> None:
        err = validate_text(self.name, 64, Lines.ONE)
        if err is not None:
            raise ManifestError(
                f"the package name is not valid: {err.value}")
        if self.description:
            err = validate_text(self.description, 500, Lines.MANY)
            if err is not None:
                raise ManifestError(
                    "the package description text is not valid: "
                    f"{err.value}")
        err = validate_text(self.publisher.name, 64, Lines.ONE)
        if err is not None:
            raise ManifestError(
                f"the publisher name is not valid: {err.value}")
        if self.permissions:
            raise ManifestError(
                "the package asks for permissions, which are not "
                "available yet")
        self._validate_files()
        self._validate_points()
        if self.provenance is not None:
            _validate_provenance(self.provenance)

    def _validate_files(self) -> None:
        max_files, max_file, max_total = self.kind.limits()
        if not self.files:
            raise ManifestError("the package is missing a file it needs")
        if len(self.files) > max_files:
            raise ManifestError("the package holds too many files")
        total = 0
        for entry in self.files:
            if entry.size > max_file:
                raise ManifestError("a file in the package is too big")
            total += entry.size
        if total > max_total:
            raise ManifestError("the package is too big")
        err = validate_all([e.path for e in self.files])
        if err is not None:
            raise ManifestError(str(err.value))
        required = {"theme": THEME_FILE, "plugin": PLUGIN_FILE}.get(
            str(self.kind))
        if required is not None and not any(
                e.path == required for e in self.files):
            raise ManifestError("the package is missing a file it needs")

    def _validate_points(self) -> None:
        for point in self.extension_points:
            want = EXTENSION_POINTS.get(point)
            if want is None:
                raise ManifestError(
                    "the package fills an extension point this "
                    f"Studio does not have: {point}")
            if want != str(self.kind):
                raise ManifestError(
                    f"a {self.kind} cannot fill the {point} "
                    "extension point")

    def file(self, path: str) -> FileEntry | None:
        """The manifest entry for one path, or None."""
        for entry in self.files:
            if entry.path == path:
                return entry
        return None


def _parse_compat(raw: object, kind: Kind) -> dict:
    if not isinstance(raw, dict) or not raw:
        raise ManifestError(
            "the package does not say which apps it is for")
    if len(raw) > MAX_APPS:
        raise ManifestError("the package lists too many apps")
    compat: dict = {}
    for app, entry in raw.items():
        if AppId.try_new(str(app)) is None:
            raise ManifestError(
                "the package's app requirements are not valid for "
                "this kind of package")
        if not isinstance(entry, dict):
            raise ManifestError(
                "the package's app requirements are not valid for "
                "this kind of package")
        interface = entry.get("interface")
        if kind == "plugin":
            if interface is None or isinstance(interface, bool) \
                    or not isinstance(interface, int) or interface < 1:
                raise ManifestError(
                    "the package's app requirements are not valid "
                    "for this kind of package")
        elif interface is not None:
            raise ManifestError(
                "the package's app requirements are not valid for "
                "this kind of package")
        min_app = entry.get("min_app")
        parsed_min: Version | None = None
        if min_app is not None:
            try:
                parsed_min = Version.parse(str(min_app))
            except VersionError:
                raise ManifestError(
                    "the package's app requirements are not valid "
                    "for this kind of package") from None
        compat[str(app)] = Compat(min_app=parsed_min,
                                  interface=interface)
    return compat


def _parse_files(raw: object) -> tuple:
    if not isinstance(raw, list):
        raise ManifestError("the package is missing a file it needs")
    entries: list[FileEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ManifestError("the package is missing a file it needs")
        path = item.get("path", "")
        sha = item.get("sha256", "")
        size = item.get("size", -1)
        if (not isinstance(sha, str) or len(sha) != 64
                or not all(c in "0123456789abcdef" for c in sha)):
            raise ManifestError("a file's checksum is not valid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ManifestError("a file's checksum is not valid")
        entries.append(FileEntry(path=str(path), sha256=sha, size=size))
    return tuple(entries)


def _parse_provenance(raw: object) -> Provenance | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ManifestError("the package's source details are not valid")
    return Provenance(source=str(raw.get("source", "")),
                      commit=str(raw.get("commit", "")))


def _validate_provenance(provenance: Provenance) -> None:
    commit_ok = (len(provenance.commit) == 40 and all(
        c in "0123456789abcdef" for c in provenance.commit))
    if not _valid_https_url(provenance.source, 200) or not commit_ok:
        raise ManifestError("the package's source details are not valid")


def _valid_https_url(url: str, max_len: int) -> bool:
    """A plain ``https://host/path`` address: printable ASCII, no
    spaces, no user name or password, a host present, bounded."""
    if not url.startswith("https://"):
        return False
    rest = url[len("https://"):]
    authority = rest.split("/")[0] if rest else ""
    return (len(url) <= max_len
            and all(33 <= ord(c) <= 126 for c in url)
            and bool(authority) and "@" not in authority)
