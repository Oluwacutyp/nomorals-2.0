"""Rig definition — the avatar skeleton, as data.

The HyperJail study (storytold/hyperjail, pattern only) is a UE5
project whose game mode is an empty shell: the rig IS the
configuration — a skeleton the tracking inputs drive, a face
blendshape set, and a viseme set. Devon's rig is the same shape in
data: a bone tree (name, parent, rest offset, rest rotation), the
52 ARKit face blendshapes, and the viseme set the directed lip-sync
organ already defines. Nothing here is hardcoded to one model — a
rig is a JSON file, loaded and validated, and a bad file is refused
naming the bone or value that is wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import RigError

__all__ = [
    "ARKIT_BLENDSHAPES",
    "Bone",
    "RigDefinition",
    "default_rig",
    "default_rig_path",
]

#: The 52 ARKit face blendshapes — the face-capture vocabulary the
#: HyperJail rig takes from AppleARKitFaceSupport. A rig's face is
#: weights over these names; an unknown name is refused at drive
#: time, never silently ignored.
ARKIT_BLENDSHAPES: tuple[str, ...] = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft",
    "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
)


def default_rig_path() -> Path:
    """The bundled default rig definition (this organ's data)."""
    return Path(__file__).resolve().parent / "default_rig.json"


@dataclass(frozen=True)
class Bone:
    """One bone: name, parent bone (None for the root), rest offset
    in metres from the parent (Y-up), rest rotation in degrees
    (Z, X, Y — the BVH channel order)."""

    name: str
    parent: str | None
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rest: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RigDefinition:
    """A rig: a bone tree, a face blendshape set, a viseme set."""

    name: str
    bones: tuple[Bone, ...]
    blendshapes: tuple[str, ...] = ARKIT_BLENDSHAPES
    visemes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.name or "").strip():
            raise RigError("rig name must not be empty",
                           kind="bad_rig")
        if not self.bones:
            raise RigError(f"rig {self.name!r} has no bones",
                           kind="bad_rig")
        names = [bone.name for bone in self.bones]
        if len(set(names)) != len(names):
            seen: set[str] = set()
            for bone_name in names:
                if bone_name in seen:
                    raise RigError(
                        f"rig {self.name!r} has duplicate bone "
                        f"{bone_name!r}", kind="bad_rig")
                seen.add(bone_name)
        roots = [bone for bone in self.bones if bone.parent is None]
        if len(roots) != 1:
            raise RigError(
                f"rig {self.name!r} needs exactly one root bone, "
                f"found {len(roots)}", kind="bad_rig")
        by_name = {bone.name: bone for bone in self.bones}
        for bone in self.bones:
            if not bone.name:
                raise RigError("rig bone name must not be empty",
                               kind="bad_rig")
            if len(bone.offset) != 3 or len(bone.rest) != 3:
                raise RigError(
                    f"bone {bone.name!r} offset and rest must each "
                    f"have 3 values", kind="bad_rig")
            for value in (*bone.offset, *bone.rest):
                if value != value or value in (float("inf"),
                                               float("-inf")):
                    raise RigError(
                        f"bone {bone.name!r} has a non-finite "
                        f"offset or rest value", kind="bad_rig")
            if bone.parent is not None and bone.parent not in by_name:
                raise RigError(
                    f"bone {bone.name!r} has unknown parent "
                    f"{bone.parent!r}", kind="bad_rig")
        # Cycle check: walk every bone to the root; a repeat is a loop.
        for bone in self.bones:
            seen_names: set[str] = set()
            current: Bone | None = bone
            while current is not None:
                if current.name in seen_names:
                    raise RigError(
                        f"rig {self.name!r} bone cycle at "
                        f"{current.name!r}", kind="bad_rig")
                seen_names.add(current.name)
                current = (by_name[current.parent]
                           if current.parent is not None else None)

    def bone(self, name: str) -> Bone:
        """The bone by name — refused by name when it is not there."""
        for bone in self.bones:
            if bone.name == name:
                return bone
        raise RigError(f"rig {self.name!r} has no bone {name!r}",
                       kind="unknown_bone")

    def children(self, name: str) -> tuple[Bone, ...]:
        """A bone's children, in definition order."""
        return tuple(bone for bone in self.bones
                     if bone.parent == name)

    def hierarchy_order(self) -> tuple[Bone, ...]:
        """Bones parent-before-children (definition order already
        is, when the file lists parents first — verified here
        instead of assumed)."""
        ordered: list[Bone] = []
        placed: set[str] = set()

        def place(bone: Bone) -> None:
            if bone.name in placed:
                return
            if bone.parent is not None:
                place(self.bone(bone.parent))
            ordered.append(bone)
            placed.add(bone.name)

        for bone in self.bones:
            place(bone)
        return tuple(ordered)

    def to_dict(self) -> dict[str, Any]:
        """The rig as plain JSON-able data."""
        return {
            "name": self.name,
            "bones": [{
                "name": bone.name,
                "parent": bone.parent,
                "offset": list(bone.offset),
                "rest": list(bone.rest),
            } for bone in self.bones],
            "blendshapes": list(self.blendshapes),
            "visemes": list(self.visemes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RigDefinition:
        """Build a rig from :meth:`to_dict` output. Malformed data
        is refused naming the bone or value."""
        if not isinstance(data, dict):
            raise RigError("rig data must be a JSON object",
                           kind="bad_rig")
        name = data.get("name")
        raw_bones = data.get("bones")
        if not isinstance(name, str) or not name.strip():
            raise RigError("rig data needs a non-empty 'name'",
                           kind="bad_rig")
        if not isinstance(raw_bones, list) or not raw_bones:
            raise RigError(f"rig {name!r} needs a non-empty 'bones' list",
                           kind="bad_rig")
        bones: list[Bone] = []
        for index, node in enumerate(raw_bones):
            if not isinstance(node, dict):
                raise RigError(
                    f"rig {name!r} bone {index} is not an object",
                    kind="bad_rig")
            bone_name = node.get("name")
            parent = node.get("parent")
            if not isinstance(bone_name, str) or not bone_name:
                raise RigError(
                    f"rig {name!r} bone {index} needs a name",
                    kind="bad_rig")
            if parent is not None and not isinstance(parent, str):
                raise RigError(
                    f"rig {name!r} bone {bone_name!r} parent must be "
                    f"a name or null", kind="bad_rig")
            try:
                offset = tuple(float(v)
                               for v in node.get("offset", (0, 0, 0)))
                rest = tuple(float(v)
                             for v in node.get("rest", (0, 0, 0)))
            except (TypeError, ValueError) as exc:
                raise RigError(
                    f"rig {name!r} bone {bone_name!r} offset/rest "
                    f"must be numbers: {exc}", kind="bad_rig") from exc
            bones.append(Bone(name=bone_name, parent=parent,
                              offset=offset, rest=rest))
        blendshapes = data.get("blendshapes")
        if blendshapes is None:
            blendshapes = ARKIT_BLENDSHAPES
        if not isinstance(blendshapes, list) or not all(
                isinstance(b, str) for b in blendshapes):
            raise RigError(
                f"rig {name!r} blendshapes must be a list of names",
                kind="bad_rig")
        visemes = data.get("visemes")
        if visemes is None:
            visemes = _directed_visemes()
        if not isinstance(visemes, list) or not all(
                isinstance(v, str) for v in visemes):
            raise RigError(
                f"rig {name!r} visemes must be a list of names",
                kind="bad_rig")
        return cls(name=name, bones=tuple(bones),
                   blendshapes=tuple(blendshapes),
                   visemes=tuple(visemes))

    @classmethod
    def from_json(cls, path: str | Path) -> RigDefinition:
        """Load a rig from a JSON file. A missing or malformed file
        is refused by name."""
        target = Path(path).expanduser()
        if not target.exists():
            raise RigError(f"no rig file at {target}",
                           kind="missing_rig")
        try:
            blob = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise RigError(
                f"rig file at {target} is unreadable: {exc}",
                kind="bad_rig") from exc
        return cls.from_dict(blob)


def _directed_visemes() -> list[str]:
    """The viseme names the directed lip-sync organ defines — the
    rig's mouth vocabulary is the organ's, not a second copy."""
    from ..directed.lipsync import VISEME_SHAPES
    return list(VISEME_SHAPES)


def default_rig() -> RigDefinition:
    """The bundled default humanoid rig."""
    return RigDefinition.from_json(default_rig_path())
