"""The FontLibrary — Devon's stocked font library, as a registry.

The font census (2026-11-11) found Devon shipped ZERO font
files: every renderer slid down a private ladder of host paths
to Pillow's bitmap default, and a font the host happened to
lack simply vanished.  The owner's ruling was the opposite of
a fetch-on-demand service: the library is STOCKED — delivered
whole, hers, present in her data estate before any render
needs it, working fully offline.  Rendering NEVER touches the
network; stocking happens exactly once, through
:mod:`fontpack`.

Three tiers, searched in this order:

* **pack** — the full stocked library in her data estate
  (``<NM home>/fonts/library/``), built from the google/fonts
  catalogue by :mod:`fontpack` and described by its
  ``MANIFEST.json`` (≈1,900 families, licences carried inside
  the pack).
* **core** — the vendored core subset inside this package
  (``nomorals/media/typography/fonts/`` + its own manifest):
  the bare-clone floor, tens of MB of core families so a fresh
  clone renders properly before the pack is stocked.  The
  moment the pack is stocked it supersedes the subset.
* **local** — owner-local directories named by ``NM_FONT_DIRS``
  (the 1001fonts route: fonts the owner personally drops on
  his own machine, under his own licence responsibility, light
  up here with zero code change).  Never shipped stock.

A family the library does not hold is a NAMED outcome:
:meth:`FontLibrary.resolve` raises :class:`FontNotStocked`
naming the family and every tier searched — never a silent
slide to some other face.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "CORE_TIER",
    "LOCAL_TIER",
    "MANIFEST_NAME",
    "PACK_TIER",
    "FaceStock",
    "FamilyStock",
    "FontLibrary",
    "FontNotStocked",
    "estate_library_dir",
    "normalize_family",
]

MANIFEST_NAME = "MANIFEST.json"
PACK_TIER = "pack"
CORE_TIER = "core"
LOCAL_TIER = "local"
TIER_ORDER = (PACK_TIER, CORE_TIER, LOCAL_TIER)

_FONT_EXTS = {".ttf", ".otf"}


class FontNotStocked(LookupError):
    """The stocked library does not hold the requested family.

    Carries the family name and a human description of every
    tier that was searched, so the refusal can be quoted
    verbatim — "not in the stocked library" is an answer, not
    a shrug.
    """

    def __init__(self, family: str, searched: tuple[str, ...]):
        self.family = str(family)
        self.searched = tuple(searched)
        where = "; ".join(self.searched) if self.searched else \
            "no font tiers exist on this host"
        super().__init__(
            f"font family {self.family!r} is not in the stocked "
            f"library (searched: {where})")


def normalize_family(name: Any) -> str:
    """The lookup key for a family name: casefolded, whitespace
    collapsed.  ``"Open Sans"`` and ``"open  sans"`` are one
    family."""
    return " ".join(str(name or "").split()).casefold()


def _squash(key: str) -> str:
    return key.replace(" ", "").replace("-", "").replace("_", "")


def estate_library_dir(home: Any = None) -> Path:
    """Where the stocked pack lives in Devon's data estate."""
    if home:
        base = Path(home).expanduser()
    else:
        env = os.environ.get("NM_HOME", "")
        base = Path(env).expanduser() if env else \
            Path.home() / ".nomorals"
    return base / "fonts" / "library"


def core_fonts_dir() -> Path:
    """The vendored core subset shipped inside this package."""
    return Path(__file__).resolve().parent / "fonts"


def local_font_dirs() -> list[Path]:
    """Owner-local font directories from ``NM_FONT_DIRS``."""
    env = os.environ.get("NM_FONT_DIRS", "")
    out: list[Path] = []
    for part in env.split(os.pathsep):
        if part.strip():
            out.append(Path(part.strip()).expanduser())
    return out


@dataclass(frozen=True)
class FaceStock:
    """One stocked face: a file inside its tier root."""

    file: str                       # path relative to the tier root
    weight: int = 400
    weight_range: tuple[int, int] | None = None
    style: str = "normal"           # "normal" | "italic"
    variable: bool = False
    sha256: str = ""
    size: int = 0
    face_index: int = 0

    @property
    def italic(self) -> bool:
        return self.style == "italic"

    def weight_distance(self, want: int) -> int:
        if self.variable and self.weight_range:
            lo, hi = self.weight_range
            if lo <= want <= hi:
                return 0
            return min(abs(want - lo), abs(want - hi))
        return abs(self.weight - want)


@dataclass
class FamilyStock:
    """One stocked family, and the tier that stocks it."""

    name: str
    tier: str
    category: str = ""
    license: str = ""
    license_file: str = ""
    reserved_font_name: bool = False
    faces: tuple[FaceStock, ...] = ()

    @property
    def key(self) -> str:
        return normalize_family(self.name)


@dataclass
class _Tier:
    name: str
    root: Path
    present: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    families: dict[str, FamilyStock] = field(default_factory=dict)
    _scanned: bool = False

    def describe(self) -> str:
        if self.name == PACK_TIER:
            label = f"pack {self.root}"
        elif self.name == CORE_TIER:
            label = f"core subset {self.root}"
        else:
            label = f"local dir {self.root}"
        if not self.present:
            return f"{label} (not stocked)"
        count = self.info.get("family_count", len(self.families))
        return f"{label} ({count} families)"


def _face_from_manifest(raw: dict[str, Any]) -> FaceStock:
    wr = raw.get("weight_range")
    return FaceStock(
        file=str(raw.get("file", "")),
        weight=int(raw.get("weight", 400) or 400),
        weight_range=(int(wr[0]), int(wr[1])) if wr else None,
        style=str(raw.get("style", "normal") or "normal"),
        variable=bool(raw.get("variable", False)),
        sha256=str(raw.get("sha256", "")),
        size=int(raw.get("bytes", 0) or 0),
        face_index=int(raw.get("face_index", 0) or 0))


def _family_from_manifest(raw: dict[str, Any], tier: str) -> FamilyStock:
    return FamilyStock(
        name=str(raw.get("name", "")),
        tier=tier,
        category=str(raw.get("category", "")),
        license=str(raw.get("license", "")),
        license_file=str(raw.get("license_file", "")),
        reserved_font_name=bool(raw.get("reserved_font_name", False)),
        faces=tuple(_face_from_manifest(f)
                    for f in raw.get("faces", [])
                    if isinstance(f, dict)))


class FontLibrary:
    """The registry over the stocked tiers.

    Constructed cheaply: manifests are read eagerly (one JSON
    per stocked tier); owner-local dirs are scanned lazily on
    first use (fontparse over whatever the owner dropped in).
    """

    def __init__(self, *, pack_dir: Any = None, core_dir: Any = None,
                 local_dirs: Any = ()):
        self._tiers: list[_Tier] = []
        self._add_manifest_tier(PACK_TIER, pack_dir)
        self._add_manifest_tier(CORE_TIER, core_dir)
        for item in local_dirs or ():
            root = Path(item).expanduser()
            self._tiers.append(_Tier(
                name=LOCAL_TIER, root=root, present=root.is_dir()))

    # construction ────────────────────────────────────────────
    @classmethod
    def discover(cls, *, home: Any = None) -> FontLibrary:
        """The library as this host actually holds it."""
        return cls(pack_dir=estate_library_dir(home),
                   core_dir=core_fonts_dir(),
                   local_dirs=local_font_dirs())

    def _add_manifest_tier(self, name: str, root: Any) -> None:
        if root is None:
            return
        root = Path(root).expanduser()
        tier = _Tier(name=name, root=root)
        manifest_path = root / MANIFEST_NAME
        if manifest_path.is_file():
            try:
                data = json.loads(
                    manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                _log.warning("font library: manifest %s unreadable: "
                             "%s", manifest_path, exc)
                data = None
            if isinstance(data, dict):
                tier.present = True
                tier.info = {
                    "kind": str(data.get("kind", "")),
                    "source_commit": str(
                        (data.get("source") or {}).get("commit", "")),
                    "family_count": int(
                        data.get("family_count", 0) or 0),
                    "file_count": int(data.get("file_count", 0) or 0),
                    "total_bytes": int(
                        data.get("total_bytes", 0) or 0),
                }
                for raw in data.get("families", []):
                    if not isinstance(raw, dict):
                        continue
                    fam = _family_from_manifest(raw, name)
                    if fam.name:
                        tier.families.setdefault(fam.key, fam)
                tier._scanned = True
        self._tiers.append(tier)

    def _scan_local_tier(self, tier: _Tier) -> None:
        """Index an owner-local dir with fontparse (lazy)."""
        if tier._scanned or not tier.present:
            return
        tier._scanned = True
        from .fontparse import parse_font_file

        grouped: dict[str, list[FaceStock]] = {}
        names: dict[str, str] = {}
        for dirpath, _dirs, filenames in os.walk(tier.root):
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if path.suffix.lower() not in _FONT_EXTS:
                    continue
                try:
                    faces = parse_font_file(path)
                except Exception as exc:  # noqa: BLE001 - one bad
                    _log.debug("font library: local file %s did "
                               "not parse: %s", path, exc)
                    continue
                rel = str(path.relative_to(tier.root))
                for face in faces:
                    axes = {a.tag: a for a in face.axes}
                    wr = None
                    if "wght" in axes:
                        wr = (int(axes["wght"].minimum),
                              int(axes["wght"].maximum))
                    key = normalize_family(face.family)
                    names.setdefault(key, face.family)
                    grouped.setdefault(key, []).append(FaceStock(
                        file=rel, weight=int(face.weight),
                        weight_range=wr,
                        style="italic" if face.italic else "normal",
                        variable=bool(face.axes),
                        face_index=int(face.face_index)))
        for key, faces in grouped.items():
            tier.families.setdefault(key, FamilyStock(
                name=names[key], tier=LOCAL_TIER,
                faces=tuple(sorted(
                    faces, key=lambda f: (f.file, f.face_index)))))
        tier.info = {"kind": "local",
                     "family_count": len(tier.families),
                     "file_count": sum(len(f.faces)
                                        for f in tier.families.values()),
                     "total_bytes": 0}

    def _ensure_scanned(self) -> None:
        for tier in self._tiers:
            if tier.name == LOCAL_TIER:
                self._scan_local_tier(tier)

    # views ───────────────────────────────────────────────────
    def tiers(self) -> list[_Tier]:
        return list(self._tiers)

    def roots(self) -> list[Path]:
        """Tier roots that exist on disk, in search order — the
        catalogue roots the typography service puts first."""
        return [t.root for t in self._tiers if t.root.is_dir()]

    def searched(self) -> tuple[str, ...]:
        return tuple(t.describe() for t in self._tiers)

    def families(self) -> list[FamilyStock]:
        """Every stocked family, merged across tiers: the
        highest-precedence tier that stocks a family speaks
        for it."""
        self._ensure_scanned()
        merged: dict[str, FamilyStock] = {}
        for tier in self._tiers:
            for key, fam in tier.families.items():
                merged.setdefault(key, fam)
        return [merged[k] for k in sorted(merged)]

    def family(self, name: Any) -> FamilyStock | None:
        """The merged record for one family, or None."""
        self._ensure_scanned()
        key = normalize_family(name)
        for tier in self._tiers:
            fam = tier.families.get(key) or \
                self._squashed_get(tier, key)
            if fam is not None:
                return fam
        return None

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for fam in self.families():
            counts[fam.category or "Uncategorised"] = \
                counts.get(fam.category or "Uncategorised", 0) + 1
        return dict(sorted(counts.items()))

    def search(self, query: Any) -> list[FamilyStock]:
        """Deterministic family search: exact, prefix, then
        substring over family names."""
        q = normalize_family(query)
        if not q:
            return []
        ranked: list[tuple[int, FamilyStock]] = []
        for fam in self.families():
            key = fam.key
            if key == q:
                rank = 0
            elif key.startswith(q):
                rank = 1
            elif q in key:
                rank = 2
            elif all(tok in key for tok in q.split()):
                rank = 3
            else:
                continue
            ranked.append((rank, fam))
        ranked.sort(key=lambda rf: (rf[0], rf[1].key))
        return [fam for _rank, fam in ranked]

    def stats(self) -> dict[str, Any]:
        """Per-tier stocking facts + merged totals."""
        self._ensure_scanned()
        tiers = []
        for tier in self._tiers:
            tiers.append({
                "tier": tier.name,
                "root": str(tier.root),
                "stocked": tier.present,
                "kind": tier.info.get("kind", ""),
                "source_commit": tier.info.get("source_commit", ""),
                "families": tier.info.get(
                    "family_count", len(tier.families)),
                "files": tier.info.get("file_count", 0),
                "bytes": tier.info.get("total_bytes", 0),
            })
        merged = self.families()
        return {
            "tiers": tiers,
            "families": len(merged),
            "faces": sum(len(f.faces) for f in merged),
            "categories": self.categories(),
        }

    # resolution ──────────────────────────────────────────────
    @staticmethod
    def _squashed_get(tier: _Tier, key: str) -> FamilyStock | None:
        want = _squash(key)
        for fam_key, fam in tier.families.items():
            if _squash(fam_key) == want:
                return fam
        return None

    @staticmethod
    def _pick_face(fam: FamilyStock, weight: int | None,
                   italic: bool) -> FaceStock | None:
        if not fam.faces:
            return None
        want = int(weight) if weight else 400
        styled = [f for f in fam.faces if f.italic == bool(italic)]
        pool = styled or list(fam.faces)
        return min(pool, key=lambda f: (f.weight_distance(want),
                                        f.variable, f.file))

    def resolve(self, family: Any, *, weight: int | None = None,
                italic: bool = False) -> Path:
        """The stocked file for ``family`` — the face closest
        to the requested weight/style.  Raises
        :class:`FontNotStocked` when no tier holds the family:
        a named outcome, never a silent slide."""
        self._ensure_scanned()
        key = normalize_family(family)
        for tier in self._tiers:
            if not tier.present:
                continue
            fam = tier.families.get(key) or \
                self._squashed_get(tier, key)
            if fam is None:
                continue
            face = self._pick_face(fam, weight, italic)
            if face is not None:
                return tier.root / face.file
        raise FontNotStocked(str(family), self.searched())

    def stocked(self, family: Any) -> bool:
        return self.family(family) is not None
