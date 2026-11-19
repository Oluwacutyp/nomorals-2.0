"""The catalog: which components Devon can update, from which sources,
and under which trust rules.

Ported from upstream ``crates/core/src/catalog.rs`` (the suite list:
official vs community apps, per-kind install rules), ``crates/native/
src/source.rs`` (where release information comes from) and ``crates/
native/src/pins.rs`` (version tracks synced to a shared pins file),
with the release chooser from ``crates/core/src/release.rs``.

The trust rules, in the order they are checked:
1. the source must exist in the catalog — an unknown source is refused;
2. the component must exist in the catalog — an unknown component is refused;
3. a community source may install only its own listed components;
4. a revoked package hash is refused, whoever published it;
5. a community package must carry a hash the catalog pins — the pin IS
   the trust for a community source, because Devon has no signature
   infrastructure to check anything stronger, and the catalog says so
   honestly instead of pretending.

An official source (the owner's own repos) needs no hash pin: the
package is verified by its manifest hashes and the source match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .version import Version, is_stable

__all__ = [
    "Catalog",
    "Component",
    "Source",
    "TrustVerdict",
    "VersionTrack",
    "choose_version",
    "default_catalog",
]

# Devon's real updatable components: the eight craft organs the
# launcher knows how to update, the core runtime and the OS kernel /
# supervisor. The ids are the real dotted paths in this repo.
_SEED_COMPONENTS: tuple[tuple[str, str, bool], ...] = (
    ("nomorals.media.cad", "CADCraft", False),
    ("nomorals.media.material", "MaterialCraft", False),
    ("nomorals.media.textures", "TextureCraft", False),
    ("nomorals.media.animation", "AniCraft", False),
    ("nomorals.media.design", "DesignCraft", False),
    ("nomorals.media.vector", "VectorCraft", False),
    ("nomorals.media.sound", "SoundCraft", False),
    ("nomorals.media.layout", "LayoutCraft", False),
    ("nomorals.core", "Devon core runtime", True),
    ("nomorals.os.kernel", "OS kernel / supervisor", True),
)


@dataclass(frozen=True)
class Component:
    """One updatable component of the Devon install."""

    id: str
    name: str
    source: str
    community: bool = False
    restart_required: bool = False


@dataclass(frozen=True)
class Source:
    """One place update packages may come from."""

    id: str
    kind: str  # "official" or "community"
    repo: str
    allowed_components: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrustVerdict:
    """The answer to one trust check: ok, and the reason when not."""

    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class VersionTrack:
    """Which releases a component follows (upstream pins.rs)."""

    kind: str  # "stable" | "nightly" | "pinned"
    tag: str = ""

    @classmethod
    def pinned(cls, tag: str) -> VersionTrack:
        """Follow exactly the release tagged ``tag``."""
        return cls("pinned", tag)


# The two tracks every component follows by default. (Plain module
# constants, not dataclass fields — a @dataclass would treat an
# annotated class attribute as a third field.)
VersionTrack.STABLE = VersionTrack("stable")  # type: ignore[attr-defined]
VersionTrack.NIGHTLY = VersionTrack("nightly")  # type: ignore[attr-defined]


@dataclass
class Catalog:
    """The components, sources, pins and revocations the launcher trusts."""

    components: list[Component] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    pins: dict[str, set[str]] = field(default_factory=dict)
    attestation_pins: dict[tuple[str, str], str] = field(default_factory=dict)
    revoked: dict[str, str] = field(default_factory=dict)

    def component(self, component_id: str) -> Component | None:
        """The component with ``component_id``, or None."""
        for component in self.components:
            if component.id == component_id:
                return component
        return None

    def source(self, source_id: str) -> Source | None:
        """The source with ``source_id``, or None."""
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def pin(self, component_id: str, package_sha256: str) -> None:
        """Pin one package hash as trusted for ``component_id``."""
        self.pins.setdefault(component_id, set()).add(package_sha256)

    def pin_attestation(self, component_id: str, source_id: str,
                        value: str) -> None:
        """Pin the attestation value a component's packages must carry."""
        self.attestation_pins[(component_id, source_id)] = value

    def revoke(self, package_sha256: str, reason: str) -> None:
        """Revoke one package hash, with the reason it was revoked."""
        self.revoked[package_sha256] = reason

    def check_trust(self, component_id: str, source_id: str,
                    package_sha256: str) -> TrustVerdict:
        """Apply the trust rules to one (component, source, hash)."""
        source = self.source(source_id)
        if source is None:
            return TrustVerdict(False, f"unknown source: {source_id}")
        component = self.component(component_id)
        if component is None:
            return TrustVerdict(False, f"unknown component: {component_id}")
        if source.kind == "community":
            if component_id not in source.allowed_components:
                return TrustVerdict(
                    False,
                    f"community source {source_id} may not install "
                    f"{component_id}")
            if package_sha256 not in self.pins.get(component_id, set()):
                return TrustVerdict(
                    False,
                    f"community source {source_id} requires a pinned hash "
                    f"for {component_id}")
        if package_sha256 in self.revoked:
            return TrustVerdict(
                False,
                f"revoked package hash {package_sha256[:16]}…: "
                f"{self.revoked[package_sha256]}")
        return TrustVerdict(True)


def default_catalog() -> Catalog:
    """The catalog seeded with Devon's real components and sources."""
    components = [
        Component(id=cid, name=name, source="devon-official",
                  restart_required=restart)
        for cid, name, restart in _SEED_COMPONENTS
    ]
    sources = [
        Source(id="devon-official", kind="official",
               repo="Oluwacutyp/nomorals-2.0"),
    ]
    return Catalog(components=components, sources=sources)


def _has_published_checksum(release: dict) -> bool:
    return any(bool(asset.get("sha256"))
               for asset in release.get("assets", []))


def choose_version(releases: list[dict],
                   track: VersionTrack) -> dict | None:
    """Pick the release ``track`` follows (upstream release.rs).

    Stable: the highest stable version. Nightly: the newest nightly
    that publishes a checksum (a nightly without one cannot be
    verified, so it is never chosen). Pinned: the exact tag, or None.
    """
    if track.kind == "pinned":
        for release in releases:
            if release.get("tag") == track.tag:
                return release
        return None
    if track.kind == "nightly":
        nightlies = [r for r in releases
                     if r.get("prerelease") and _has_published_checksum(r)]
        if not nightlies:
            return None
        return max(nightlies, key=lambda r: str(r.get("published_at", "")))
    stable = []
    for release in releases:
        if release.get("prerelease"):
            continue
        try:
            version = Version.parse(str(release.get("version", "")))
        except ValueError:
            continue
        if is_stable(version):
            stable.append((version, release))
    if not stable:
        return None
    return max(stable, key=lambda pair: pair[0])[1]
