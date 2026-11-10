"""The map-pack manifest contract (Devon Atlas, doc 21 §3 as adopted).

Doc 21 superseded doc 19's per-source TileStore URL registry (S2):
a basemap is a Devon-built *release* — a directory of member files
plus one ``manifest.json`` that pins every member by SHA-256 and
carries a pack-level digest over the sorted member list.  The
construction follows the shipped manifest idiom (``core/longdoc.py``):
content hashes, never copies; the envelope digest is a hash over the
canonical JSON of the member identities, so it transitively covers
every content byte in the pack.

The manifest also carries the release's **attribution** as data
(OSM/ODbL credit today; surfaces render it, code never hardcodes it)
and an optional **signature**.  M1 does not sign — the release
registry that signs is M2's — so the signature seam is explicit
about state: an unsigned pack reads ``"unsigned"`` forever, never
``"verified"``; a signed pack checked without a verifier reads
``"unverifiable"``.  Verification of a pack directory is exact:
every member present and hash-matching, tampered or missing members
named, and files the manifest does not know named as extras.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .coords import MAX_ZOOM, BBox

__all__ = [
    "FORMAT_VERSION", "MANIFEST_NAME", "PackError", "PackManifest",
    "PackMember", "PackVerification", "SignatureVerifier",
    "build_manifest", "hash_bytes", "load_manifest", "tile_relpath",
    "verify_pack", "verify_signature", "write_manifest",
]

#: Manifest format version; bump when the schema changes shape.
FORMAT_VERSION = 1

#: The manifest's own filename inside a pack directory.  It is the
#: one file a pack may contain that is not itself a member.
MANIFEST_NAME = "manifest.json"

#: A signature verifier: manifest + signature text → genuine or not.
#: M2's release registry supplies the real one; M1 ships the seam.
SignatureVerifier = Callable[["PackManifest", str], bool]


class PackError(ValueError):
    """A pack or manifest defect, named in the message."""


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex of ``data`` — the one hash primitive packs use."""
    return hashlib.sha256(data).hexdigest()


def _envelope_digest(members: tuple[PackMember, ...]) -> str:
    """Pack digest over the sorted member identities (longdoc idiom)."""
    payload = [[m.path, m.sha256, m.size]
               for m in sorted(members, key=lambda m: m.path)]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tile_relpath(z: int, x: int, y: int, tile_format: str = "png") -> str:
    """A tile member's pack-relative path: ``tiles/{z}/{x}/{y}.png``."""
    return "tiles/" + str(int(z)) + "/" + str(int(x)) + "/" \
        + str(int(y)) + "." + tile_format


@dataclass(frozen=True)
class PackMember:
    """One pinned file: pack-relative POSIX path, hash, size."""

    path: str
    sha256: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, data: dict) -> PackMember:
        return cls(path=str(data["path"]), sha256=str(data["sha256"]),
                   size=int(data["size"]))


@dataclass(frozen=True)
class PackManifest:
    """A pack release: identity, extent, and every member pinned."""

    format: int
    pack_id: str
    name: str
    bbox: BBox
    zoom_min: int
    zoom_max: int
    built_at: str
    attribution: str
    tile_format: str = "png"
    members: tuple[PackMember, ...] = ()
    digest: str = ""
    signature: str = ""

    def __post_init__(self) -> None:
        if int(self.format) != FORMAT_VERSION:
            raise PackError(
                "unsupported manifest format: " + repr(self.format))
        if not self.pack_id or "/" in self.pack_id or "\\" in self.pack_id:
            raise PackError(
                "pack_id must be a non-empty single name: "
                + repr(self.pack_id))
        if not (0 <= int(self.zoom_min) <= int(self.zoom_max) <= MAX_ZOOM):
            raise PackError(
                "zoom range out of bounds: " + repr(self.zoom_min)
                + ".." + repr(self.zoom_max))
        paths = [m.path for m in self.members]
        if len(paths) != len(set(paths)):
            raise PackError("manifest lists a member path twice")

    @property
    def signature_state(self) -> str:
        """``"unsigned"`` or ``"signed"`` — never ``"verified"`` here."""
        return "signed" if self.signature else "unsigned"

    def compute_digest(self) -> str:
        """The digest this manifest's member list *should* carry."""
        return _envelope_digest(self.members)

    def with_digest(self) -> PackManifest:
        """A copy whose digest is recomputed over the member list."""
        return replace(self, digest=self.compute_digest())

    def check_digest(self) -> None:
        """Refuse a manifest whose stored digest is stale or forged."""
        if self.digest != self.compute_digest():
            raise PackError(
                "manifest digest mismatch for pack " + repr(self.pack_id)
                + ": stored " + repr(self.digest) + ", computed "
                + repr(self.compute_digest()))

    def member(self, path: str) -> PackMember | None:
        for entry in self.members:
            if entry.path == path:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": int(self.format), "pack_id": self.pack_id,
            "name": self.name, "bbox": self.bbox.as_list(),
            "zoom_min": int(self.zoom_min), "zoom_max": int(self.zoom_max),
            "built_at": self.built_at, "attribution": self.attribution,
            "tile_format": self.tile_format,
            "members": [m.to_dict() for m in self.members],
            "digest": self.digest, "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PackManifest:
        manifest = cls(
            format=int(data["format"]), pack_id=str(data["pack_id"]),
            name=str(data["name"]), bbox=BBox.from_list(list(data["bbox"])),
            zoom_min=int(data["zoom_min"]), zoom_max=int(data["zoom_max"]),
            built_at=str(data["built_at"]),
            attribution=str(data["attribution"]),
            tile_format=str(data.get("tile_format", "png")),
            members=tuple(PackMember.from_dict(m)
                          for m in data.get("members", ())),
            digest=str(data.get("digest", "")),
            signature=str(data.get("signature", "")))
        manifest.check_digest()
        return manifest


@dataclass(frozen=True)
class PackVerification:
    """The verdict on a pack directory against its manifest.

    ``ok`` is content integrity: every member present and matching.
    Extra files do not corrupt pinned content, so they are *named*
    in ``extra`` rather than failing the verdict — a release that
    must be exact can require ``extra`` empty on top.
    """

    ok: bool
    pack_id: str
    digest: str
    attribution: str
    signature_state: str
    missing: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "pack_id": self.pack_id,
                "digest": self.digest, "attribution": self.attribution,
                "signature_state": self.signature_state,
                "missing": list(self.missing),
                "mismatched": list(self.mismatched),
                "extra": list(self.extra)}

    def summary(self) -> str:
        """One honest line: state, signature posture, named defects."""
        parts = ["pack " + self.pack_id + ": "
                 + ("ok" if self.ok else "FAILED"),
                 "signature " + self.signature_state]
        if self.missing:
            parts.append("missing: " + ", ".join(self.missing))
        if self.mismatched:
            parts.append("mismatched: " + ", ".join(self.mismatched))
        if self.extra:
            parts.append("extra: " + ", ".join(self.extra))
        return " · ".join(parts)


def _disk_members(pack_dir: Path) -> dict[str, Path]:
    """Every file in the pack except the manifest, by POSIX relpath."""
    found: dict[str, Path] = {}
    for path in sorted(pack_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(pack_dir).as_posix()
            if rel != MANIFEST_NAME:
                found[rel] = path
    return found


def build_manifest(pack_dir: Path | str, *, pack_id: str, name: str,
                   bbox: BBox, zoom_min: int, zoom_max: int,
                   attribution: str, built_at: str = "",
                   tile_format: str = "png",
                   signature: str = "") -> PackManifest:
    """Pin every file currently in ``pack_dir`` into a new manifest."""
    root = Path(pack_dir)
    if not root.is_dir():
        raise PackError("pack directory does not exist: " + str(root))
    members = tuple(
        PackMember(path=rel,
                   sha256=hash_bytes(path.read_bytes()),
                   size=path.stat().st_size)
        for rel, path in sorted(_disk_members(root).items()))
    manifest = PackManifest(
        format=FORMAT_VERSION, pack_id=pack_id, name=name, bbox=bbox,
        zoom_min=int(zoom_min), zoom_max=int(zoom_max),
        built_at=built_at or datetime.now(UTC).isoformat(),
        attribution=attribution, tile_format=tile_format,
        members=members, signature=signature)
    return manifest.with_digest()


def write_manifest(pack_dir: Path | str,
                   manifest: PackManifest) -> Path:
    """Write the manifest (digest recomputed) into the pack directory."""
    root = Path(pack_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_NAME
    path.write_text(
        json.dumps(manifest.with_digest().to_dict(), indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8")
    return path


def load_manifest(pack_dir: Path | str) -> PackManifest:
    """Read + fully validate a pack's manifest (digest checked)."""
    path = Path(pack_dir) / MANIFEST_NAME
    if not path.is_file():
        raise PackError("no manifest in pack directory: " + str(pack_dir))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackError("manifest unreadable: " + str(exc)) from exc
    if not isinstance(data, dict):
        raise PackError("manifest is not a JSON object")
    return PackManifest.from_dict(data)


def verify_pack(pack_dir: Path | str,
                manifest: PackManifest | None = None) -> PackVerification:
    """Verify a pack directory against its manifest, naming defects."""
    root = Path(pack_dir)
    manifest = manifest if manifest is not None else load_manifest(root)
    on_disk = _disk_members(root)
    missing: list[str] = []
    mismatched: list[str] = []
    for entry in manifest.members:
        path = on_disk.pop(entry.path, None)
        if path is None:
            missing.append(entry.path)
        elif path.stat().st_size != entry.size \
                or hash_bytes(path.read_bytes()) != entry.sha256:
            mismatched.append(entry.path)
    return PackVerification(
        ok=not missing and not mismatched,
        pack_id=manifest.pack_id, digest=manifest.digest,
        attribution=manifest.attribution,
        signature_state=manifest.signature_state,
        missing=tuple(missing), mismatched=tuple(mismatched),
        extra=tuple(sorted(on_disk)))


def verify_signature(manifest: PackManifest,
                     verifier: SignatureVerifier | None = None) -> str:
    """The signature verdict, honestly graded.

    ``"unsigned"`` — no signature is carried.  ``"unverifiable"`` —
    a signature is carried but no verifier was supplied (M2's
    registry supplies one).  ``"verified"`` / ``"invalid"`` — the
    supplied verifier's answer.  Nothing else ever reads verified.
    """
    if not manifest.signature:
        return "unsigned"
    if verifier is None:
        return "unverifiable"
    return "verified" if verifier(manifest, manifest.signature) else "invalid"
