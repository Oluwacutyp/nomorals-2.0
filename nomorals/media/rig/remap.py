"""Bone remap — the tracking skeleton into the rig skeleton.

The HyperJail study's one real artifact is SkeletonRemap.txt: a
named table from Xsens bone names to Blender DEF-bone names (full
body plus every finger segment). Devon's remap is the same shape in
data: a dialect (``bvh`` for BVH joint names, ``xsens`` for Xsens
MVN names) mapping source bone names to rig bone names. A frame's
joints are renamed through the table; joints with no entry are
reported as unmapped, never silently dropped; a table pointing at a
bone the rig does not have is refused at load. When two source
bones map to one rig bone, the one later in frame order wins, and
the merge is named in the result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..mocap.bvh import SkeletonFrame
from .errors import RigError
from .rig import RigDefinition

__all__ = [
    "BoneRemap",
    "RemapResult",
    "default_remap",
    "default_remap_path",
]


def default_remap_path() -> Path:
    """The bundled default remap table (this organ's data)."""
    return Path(__file__).resolve().parent / "default_remap.json"


@dataclass(frozen=True)
class RemapResult:
    """A remapped frame: rig-bone → channel values, plus the
    joints the table did not cover and any merges that happened."""

    values: dict[str, tuple[float, ...]]
    unmapped: tuple[str, ...]
    merged: tuple[str, ...]


@dataclass(frozen=True)
class BoneRemap:
    """One dialect's source-bone → rig-bone table."""

    dialect: str
    mapping: dict[str, str]

    def __post_init__(self) -> None:
        if not (self.dialect or "").strip():
            raise RigError("remap dialect must not be empty",
                           kind="bad_remap")
        if not self.mapping:
            raise RigError(
                f"remap dialect {self.dialect!r} has no entries",
                kind="bad_remap")
        for source, target in self.mapping.items():
            if not source or not target:
                raise RigError(
                    f"remap dialect {self.dialect!r} has an empty "
                    f"source or target", kind="bad_remap")

    def check_against(self, rig: RigDefinition) -> None:
        """Refuse a table whose targets the rig does not have —
        naming the first bad target, not failing at drive time."""
        for source, target in self.mapping.items():
            if target not in {bone.name for bone in rig.bones}:
                raise RigError(
                    f"remap dialect {self.dialect!r} maps "
                    f"{source!r} to rig bone {target!r}, which rig "
                    f"{rig.name!r} does not have", kind="bad_remap")

    def apply(self, frame: SkeletonFrame) -> RemapResult:
        """Rename a frame's joints through the table, in frame
        order. Two source bones landing on one rig bone merge —
        the later one wins and is named."""
        values: dict[str, tuple[float, ...]] = {}
        unmapped: list[str] = []
        merged: list[str] = []
        for source, channels in frame.values.items():
            target = self.mapping.get(source)
            if target is None:
                unmapped.append(source)
                continue
            if target in values:
                merged.append(f"{source}->{target}")
            values[target] = tuple(channels)
        return RemapResult(values=values, unmapped=tuple(unmapped),
                           merged=tuple(merged))

    def to_dict(self) -> dict[str, Any]:
        """The remap as plain JSON-able data."""
        return {"dialect": self.dialect,
                "mapping": dict(self.mapping)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoneRemap:
        """Build a remap from :meth:`to_dict` output."""
        if not isinstance(data, dict):
            raise RigError("remap data must be a JSON object",
                           kind="bad_remap")
        dialect = data.get("dialect")
        mapping = data.get("mapping")
        if not isinstance(dialect, str) or not dialect.strip():
            raise RigError("remap data needs a 'dialect' name",
                           kind="bad_remap")
        if not isinstance(mapping, dict) or not mapping:
            raise RigError(
                f"remap dialect {dialect!r} needs a non-empty "
                f"'mapping'", kind="bad_remap")
        if not all(isinstance(k, str) and isinstance(v, str)
                   for k, v in mapping.items()):
            raise RigError(
                f"remap dialect {dialect!r} mapping must be "
                f"string -> string", kind="bad_remap")
        return cls(dialect=dialect, mapping=dict(mapping))

    @classmethod
    def from_json(cls, path: str | Path, *,
                  dialect: str | None = None) -> BoneRemap:
        """Load a remap from a JSON file holding one dialect or a
        ``{dialect: mapping}`` table. A missing file, an unknown
        dialect, or a malformed table is refused by name."""
        target = Path(path).expanduser()
        if not target.exists():
            raise RigError(f"no remap file at {target}",
                           kind="missing_remap")
        try:
            blob = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise RigError(
                f"remap file at {target} is unreadable: {exc}",
                kind="bad_remap") from exc
        if not isinstance(blob, dict) or not blob:
            raise RigError(
                f"remap file at {target} holds no dialects",
                kind="bad_remap")
        if "mapping" in blob:
            return cls.from_dict(blob)
        if dialect is None:
            if len(blob) == 1:
                dialect = next(iter(blob))
            else:
                raise RigError(
                    f"remap file at {target} holds dialects "
                    f"{sorted(blob)} — pass dialect=<name>",
                    kind="bad_remap")
        mapping = blob.get(dialect)
        if not isinstance(mapping, dict):
            raise RigError(
                f"remap file at {target} has no dialect "
                f"{dialect!r}", kind="bad_remap")
        return cls(dialect=dialect, mapping=dict(mapping))


def default_remap(dialect: str = "bvh") -> BoneRemap:
    """The bundled default remap table for a dialect."""
    return BoneRemap.from_json(default_remap_path(), dialect=dialect)
