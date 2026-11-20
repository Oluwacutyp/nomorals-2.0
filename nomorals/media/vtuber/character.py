"""A VTuber character — a named performer over a rig.

A character is the identity a show performs: the rig session it
drives, an expression library (named presets over the rig's 52
ARKit blendshapes) and a gesture library (named pose presets over
the rig's 60 bones). Presets are data, validated against the rig
at load — an unknown blendshape or bone is refused by name, never
carried silently into a performance.

Built-ins cover the expressions and gestures a VTuber show reaches
for first; a character file adds its own on top (a file preset with
the same name as a built-in replaces it, by name, and the file says
so in the load receipt — no silent shadowing).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..rig.rig import RigDefinition
from .errors import VtuberError

__all__ = [
    "BUILTIN_EXPRESSIONS",
    "BUILTIN_GESTURES",
    "Character",
]

#: Named expressions over the ARKit blendshape set. Weights are
#: 0..1 (clamped at drive time, like every blendshape the driver
#: takes); a weight outside 0..1 at load is refused by name.
BUILTIN_EXPRESSIONS: dict[str, dict[str, float]] = {
    "neutral": {},
    "smile": {
        "mouthSmileLeft": 0.8, "mouthSmileRight": 0.8,
        "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4,
    },
    "laugh": {
        "mouthSmileLeft": 1.0, "mouthSmileRight": 1.0,
        "jawOpen": 0.6, "eyeSquintLeft": 0.5, "eyeSquintRight": 0.5,
    },
    "surprised": {
        "jawOpen": 0.5, "eyeWideLeft": 0.9, "eyeWideRight": 0.9,
        "browInnerUp": 0.8,
    },
    "wink": {
        "eyeBlinkLeft": 1.0, "mouthSmileRight": 0.5,
    },
    "angry": {
        "browDownLeft": 0.9, "browDownRight": 0.9,
        "mouthPressLeft": 0.6, "mouthPressRight": 0.6,
        "eyeSquintLeft": 0.4, "eyeSquintRight": 0.4,
    },
    "sad": {
        "browInnerUp": 0.8, "mouthFrownLeft": 0.7,
        "mouthFrownRight": 0.7,
    },
    "thinking": {
        "browOuterUpLeft": 0.6, "mouthPucker": 0.4,
        "eyeLookUpLeft": 0.5, "eyeLookUpRight": 0.5,
    },
}

#: Named gestures as bone rotations (Z, X, Y degrees, rest
#: included) over the rig's bones. Only the bones a gesture names
#: move — the rest hold their driven value.
BUILTIN_GESTURES: dict[str, dict[str, tuple[float, float, float]]] = {
    "wave": {
        "shoulder_R": (0.0, 0.0, -120.0),
        "upper_arm_R": (0.0, 0.0, -30.0),
        "forearm_R": (0.0, 0.0, -20.0),
    },
    "bow": {
        "spine": (25.0, 0.0, 0.0),
        "spine1": (15.0, 0.0, 0.0),
        "neck": (-10.0, 0.0, 0.0),
    },
    "point": {
        "shoulder_R": (0.0, 0.0, -90.0),
        "upper_arm_R": (0.0, 0.0, 0.0),
        "forearm_R": (0.0, 0.0, 0.0),
    },
    "nod": {
        "neck": (18.0, 0.0, 0.0),
        "head": (10.0, 0.0, 0.0),
    },
    "shrug": {
        "shoulder_L": (0.0, 0.0, 25.0),
        "shoulder_R": (0.0, 0.0, -25.0),
    },
    "hands-up": {
        "shoulder_L": (0.0, 0.0, 120.0),
        "shoulder_R": (0.0, 0.0, -120.0),
        "upper_arm_L": (0.0, 0.0, 20.0),
        "upper_arm_R": (0.0, 0.0, -20.0),
    },
}


def _check_expression(name: str, weights: Any,
                      blendshapes: set[str]) -> dict[str, float]:
    """Validate one expression preset against the rig."""
    if not isinstance(weights, dict):
        raise VtuberError(
            f"expression {name!r} must be a {{blendshape: weight}} "
            f"mapping, got {type(weights).__name__}",
            kind="bad_expression")
    checked: dict[str, float] = {}
    for blend, value in weights.items():
        if blend not in blendshapes:
            raise VtuberError(
                f"expression {name!r} names blendshape {blend!r}, "
                f"which is not in rig blendshapes",
                kind="bad_blendshape")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise VtuberError(
                f"expression {name!r} blendshape {blend!r} weight "
                f"must be a number: {exc}",
                kind="bad_expression") from exc
        if not 0.0 <= number <= 1.0:
            raise VtuberError(
                f"expression {name!r} blendshape {blend!r} weight "
                f"must be 0..1, got {number}",
                kind="bad_expression")
        checked[str(blend)] = number
    return checked


def _check_gesture(name: str, rotations: Any,
                   bones: set[str]) -> dict[str, tuple[float, float, float]]:
    """Validate one gesture preset against the rig."""
    if not isinstance(rotations, dict):
        raise VtuberError(
            f"gesture {name!r} must be a {{bone: [z, x, y]}} "
            f"mapping, got {type(rotations).__name__}",
            kind="bad_gesture")
    checked: dict[str, tuple[float, float, float]] = {}
    for bone, values in rotations.items():
        if bone not in bones:
            raise VtuberError(
                f"gesture {name!r} names bone {bone!r}, which is "
                f"not in rig bones",
                kind="bad_bone")
        if (not isinstance(values, (list, tuple))
                or len(values) != 3):
            raise VtuberError(
                f"gesture {name!r} bone {bone!r} needs 3 rotation "
                f"values, got {values!r}",
                kind="bad_gesture")
        try:
            checked[str(bone)] = tuple(float(v) for v in values)
        except (TypeError, ValueError) as exc:
            raise VtuberError(
                f"gesture {name!r} bone {bone!r} rotations must "
                f"be numbers: {exc}",
                kind="bad_gesture") from exc
    return checked


@dataclass
class Character:
    """A named performer: rig session + expression + gesture
    libraries."""

    name: str
    rig: str = "default"
    expressions: dict[str, dict[str, float]] = field(
        default_factory=dict)
    gestures: dict[str, dict[str, tuple[float, float, float]]] = (
        field(default_factory=dict))

    def __post_init__(self) -> None:
        self.name = (self.name or "").strip()
        if not self.name:
            raise VtuberError("character name must not be empty",
                              kind="bad_character")
        self.rig = (self.rig or "").strip() or "default"

    # -- lookup ----------------------------------------------------

    def expression(self, name: str) -> dict[str, float]:
        """One expression preset, by name — refused by name."""
        try:
            return dict(self.expressions[name])
        except KeyError as exc:
            known = sorted(self.expressions)
            raise VtuberError(
                f"character {self.name!r} has no expression "
                f"{name!r} — expressions are "
                f"{', '.join(known) or '(none)'}",
                kind="unknown_expression") from exc

    def gesture(self, name: str) -> dict[str, tuple[float, float, float]]:
        """One gesture preset, by name — refused by name."""
        try:
            return dict(self.gestures[name])
        except KeyError as exc:
            known = sorted(self.gestures)
            raise VtuberError(
                f"character {self.name!r} has no gesture "
                f"{name!r} — gestures are "
                f"{', '.join(known) or '(none)'}",
                kind="unknown_gesture") from exc

    # -- serialisation ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The character as plain JSON-able data."""
        return {
            "name": self.name,
            "rig": self.rig,
            "expressions": {
                name: dict(weights)
                for name, weights in sorted(self.expressions.items())},
            "gestures": {
                name: {bone: list(values)
                       for bone, values in sorted(rot.items())}
                for name, rot in sorted(self.gestures.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any],
                  rig: RigDefinition) -> Character:
        """Build a character from data, validated against the
        rig. File presets replace built-ins of the same name."""
        if not isinstance(data, dict):
            raise VtuberError(
                f"character must be a mapping, got "
                f"{type(data).__name__}",
                kind="bad_character")
        blendshapes = set(rig.blendshapes)
        bones = {bone.name for bone in rig.bones}
        expressions: dict[str, dict[str, float]] = {
            name: dict(weights)
            for name, weights in BUILTIN_EXPRESSIONS.items()}
        gestures: dict[str, dict[str, tuple[float, float, float]]] = {
            name: dict(rot)
            for name, rot in BUILTIN_GESTURES.items()}
        file_expr = data.get("expressions") or {}
        if not isinstance(file_expr, dict):
            raise VtuberError(
                "character 'expressions' must be a mapping",
                kind="bad_character")
        for name, weights in file_expr.items():
            expressions[str(name)] = _check_expression(
                str(name), weights, blendshapes)
        file_gest = data.get("gestures") or {}
        if not isinstance(file_gest, dict):
            raise VtuberError(
                "character 'gestures' must be a mapping",
                kind="bad_character")
        for name, rotations in file_gest.items():
            gestures[str(name)] = _check_gesture(
                str(name), rotations, bones)
        return cls(
            name=str(data.get("name") or ""),
            rig=str(data.get("rig") or "default"),
            expressions=expressions,
            gestures=gestures)

    @classmethod
    def from_file(cls, path: str | Path,
                  rig: RigDefinition) -> Character:
        """Load a character from a JSON file."""
        target = Path(path).expanduser()
        if not target.exists():
            raise VtuberError(f"no character file at {target}",
                              kind="missing_character")
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise VtuberError(
                f"character file at {target} is unreadable: {exc}",
                kind="bad_character") from exc
        return cls.from_dict(data, rig)
