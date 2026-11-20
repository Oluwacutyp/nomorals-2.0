"""The mapping between a LiveLink CSV and Devon's rig.

The bevy-facial-mocap pattern (storytold/bevy-mocap, pattern
only): a mapping is a JSON list of ``[mapping, csv_name]``
pairs. A mapping is a morph name — the rig blendshape the CSV
column drives, reordered to the rig's set — or a bone object
``{path, axis, offset?, scale?}`` — the CSV column drives a
rotation on a bone: ``value * 180° * scale + offset`` (offset
in degrees), applied about ``axis`` (``X``/``Y``/``Z`` or an
``[x, y, z]`` vector), composed per bone in mapping order.

Two honest seams, named where they differ from the clone:

* The clone emits Bevy quaternions. Devon's rig speaks Euler
  degrees — ``(Z, X, Y)``, the BVH channel order
  (``media/rig/rig.py``: ``Bone.rest``). Bone rotations are
  composed as quaternions exactly like the clone, then
  converted to Euler under a documented convention:
  intrinsic rotations applied in order Z, then X, then Y
  (``R = Rz(z) · Rx(x) · Ry(y)``). Principal-axis cases are
  exact (a pure head yaw is a pure Z angle); combined
  rotations follow the stated convention, and a round-trip
  test pins the conversion.
* Morphs with no mapping entry are reported as unmapped,
  never silently dropped; a mapped morph whose CSV column is
  missing reads 0.0 for every frame (the clone's rule).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .csv import CSVAnimation
from .errors import FaceMocapError

__all__ = [
    "BoneMapping",
    "FaceMapping",
    "MorphMapping",
    "default_arkit_mapping",
    "default_mapping_path",
]

#: A quaternion as (w, x, y, z).
_Quat = tuple[float, float, float, float]


def default_mapping_path() -> Path:
    """The bundled ARKit rename table (this organ's data)."""
    return Path(__file__).resolve().parent / "default_mapping.json"


# -- quaternions -------------------------------------------------
# The clone composes bone rotations as quaternions; Devon keeps
# that composition and converts at the end, so a mapped bone
# turns the same way in both systems.


def _axis_angle(axis: tuple[float, float, float],
                angle_rad: float) -> _Quat:
    length = math.sqrt(sum(v * v for v in axis))
    if length == 0:
        raise FaceMocapError(
            f"bone axis {axis} has zero length",
            kind="bad_mapping")
    x, y, z = (v / length for v in axis)
    half = angle_rad / 2.0
    s = math.sin(half)
    return (math.cos(half), x * s, y * s, z * s)


def _quat_mul(a: _Quat, b: _Quat) -> _Quat:
    """Quaternion multiplication: ``a`` then ``b`` — the
    product applies ``b`` first, then ``a``. Bevy's
    ``Transform::rotate_axis`` pre-multiplies (``rotation =
    step * rotation``), so the clone walks the mappings in
    file order with each new step on the LEFT: the later
    mapping is the outermost rotation."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_to_euler_zxy(quat: _Quat) -> tuple[float, float, float]:
    """A quaternion as Euler degrees ``(z, x, y)`` — intrinsic
    rotations applied in order Z, then X, then Y
    (``R = Rz(z) · Rx(x) · Ry(y)``), the rig's BVH channel
    order. The pole (|x| = 90°) is handled by clamping the
    arcsine argument, never by crashing."""
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise FaceMocapError(
            "bone rotation quaternion has zero length",
            kind="bad_mapping")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    # R = Rz(z) Rx(x) Ry(y); reading the matrix entries back:
    #   R[2][1] = sin(x)              -> x = asin(R[2][1])
    #   R[0][1] = -sin(z)cos(x),
    #   R[1][1] =  cos(z)cos(x)       -> z = atan2(-R[0][1], R[1][1])
    #   R[2][0] = -cos(x)sin(y),
    #   R[2][2] =  cos(x)cos(y)       -> y = atan2(-R[2][0], R[2][2])
    # (The first draft of this function extracted the ZYX order
    # instead — X and Y came out swapped. The matrix above is
    # the check; the round-trip test pins it.)
    m00 = 1 - 2 * (y * y + z * z)
    m01 = 2 * (x * y - w * z)
    m10 = 2 * (x * y + w * z)
    m11 = 1 - 2 * (x * x + z * z)
    m20 = 2 * (x * z - w * y)
    m21 = 2 * (y * z + w * x)
    m22 = 1 - 2 * (x * x + y * y)
    x_angle = math.asin(max(-1.0, min(1.0, m21)))
    cos_x = math.cos(x_angle)
    if abs(cos_x) < 1e-9:
        # At the pole the Z and Y angles share one degree of
        # freedom; Z takes it all, Y reads 0 — named, not hidden.
        z_angle = math.atan2(m10, m00)
        y_angle = 0.0
    else:
        z_angle = math.atan2(-m01, m11)
        y_angle = math.atan2(-m20, m22)
    return (math.degrees(z_angle), math.degrees(x_angle),
            math.degrees(y_angle))


def _euler_zxy_to_quat(
        angles: tuple[float, float, float]) -> _Quat:
    """The inverse of :func:`_quat_to_euler_zxy` — Euler degrees
    ``(z, x, y)`` back to a quaternion (the round-trip check)."""
    z, x, y = (math.radians(a) for a in angles)
    qz = _axis_angle((0.0, 0.0, 1.0), z)
    qx = _axis_angle((1.0, 0.0, 0.0), x)
    qy = _axis_angle((0.0, 1.0, 0.0), y)
    return _quat_mul(_quat_mul(qz, qx), qy)


# -- mapping entries ----------------------------------------------

@dataclass(frozen=True)
class MorphMapping:
    """A CSV column drives one rig blendshape, by name."""

    blendshape: str


@dataclass(frozen=True)
class BoneMapping:
    """A CSV column drives a rotation on a bone: ``value *
    180° * scale + offset`` about ``axis``. ``path`` names the
    bone from the root; the last element is the rig bone the
    rotation lands on."""

    path: tuple[str, ...]
    axis: tuple[float, float, float]
    offset: float = 0.0
    scale: float = 1.0

    @property
    def bone(self) -> str:
        """The rig bone this mapping drives."""
        return self.path[-1]


@dataclass
class FaceMapping:
    """A mapping file: ``[mapping, csv_name]`` pairs, in file
    order. Morph entries reorder CSV columns to the rig's
    blendshape set; bone entries rotate bones."""

    entries: list[tuple[MorphMapping | BoneMapping, str]]

    @classmethod
    def from_json(cls, blob: Any, *, where: str = "mapping"
                  ) -> FaceMapping:
        """Parse a mapping blob. A pair that is not a 2-list, a
        morph that is not a string, or a bone object missing its
        path/axis is refused by name."""
        if not isinstance(blob, list):
            raise FaceMocapError(
                f"{where} must be a JSON list of "
                f"[mapping, csv_name] pairs",
                kind="bad_mapping")
        entries: list[tuple[MorphMapping | BoneMapping, str]] = []
        for index, item in enumerate(blob):
            if (not isinstance(item, list)) or len(item) != 2:
                raise FaceMocapError(
                    f"{where} entry {index} must be a "
                    f"[mapping, csv_name] pair",
                    kind="bad_mapping")
            raw, csv_name = item
            if not isinstance(csv_name, str) or not csv_name:
                raise FaceMocapError(
                    f"{where} entry {index} needs a non-empty "
                    f"csv column name",
                    kind="bad_mapping")
            if isinstance(raw, str):
                if not raw.strip():
                    raise FaceMocapError(
                        f"{where} entry {index} has an empty "
                        f"morph name",
                        kind="bad_mapping")
                entries.append((MorphMapping(raw.strip()), csv_name))
                continue
            if isinstance(raw, dict):
                entries.append(
                    (_bone_mapping(raw, where, index), csv_name))
                continue
            raise FaceMocapError(
                f"{where} entry {index}: a mapping is a morph "
                f"name or a bone object, got "
                f"{type(raw).__name__}",
                kind="bad_mapping")
        return cls(entries=entries)

    @classmethod
    def from_file(cls, path: str | Path) -> FaceMapping:
        """Load a mapping JSON file. A missing or malformed file
        is refused by name."""
        target = Path(path).expanduser()
        if not target.exists():
            raise FaceMocapError(
                f"no mapping file at {target}",
                kind="missing_mapping")
        try:
            blob = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise FaceMocapError(
                f"mapping file at {target} is unreadable: {exc}",
                kind="bad_mapping") from exc
        return cls.from_json(blob, where=f"mapping file {target}")

    # -- applying -------------------------------------------------

    def apply(self, animation: CSVAnimation, *,
              rig_blendshapes: tuple[str, ...] | None = None
              ) -> dict[str, Any]:
        """Turn a parsed animation into rig frames: per-frame
        blendshape weights (rig order) and per-frame bone
        rotations in ``(z, x, y)`` degrees. Returns
        ``{"blendshapes": [...], "bones": [...], "unmapped":
        [...]}`` — one entry per frame, plus the CSV morphs no
        entry named (reported, never silently dropped). A morph
        entry naming a blendshape the rig does not have is
        refused by name when ``rig_blendshapes`` is given."""
        order = {name: index for index, name in
                 enumerate(animation.morph_names)}
        morph_entries = [(m, csv_name) for m, csv_name in
                         self.entries
                         if isinstance(m, MorphMapping)]
        bone_entries = [(m, csv_name) for m, csv_name in
                        self.entries
                        if isinstance(m, BoneMapping)]
        mapped_csv = {csv_name for _, csv_name in self.entries}
        unmapped = tuple(name for name in animation.morph_names
                         if name not in mapped_csv)
        if rig_blendshapes is not None:
            known = set(rig_blendshapes)
            for mapping, _ in morph_entries:
                if mapping.blendshape not in known:
                    raise FaceMocapError(
                        f"mapping names blendshape "
                        f"{mapping.blendshape!r}, which this rig "
                        f"does not have",
                        kind="bad_mapping")
        blend_frames: list[dict[str, float]] = []
        bone_frames: list[dict[str, tuple[float, float, float]]] = []
        for frame in animation.frames:
            blend: dict[str, float] = {}
            for mapping, csv_name in morph_entries:
                index = order.get(csv_name)
                blend[mapping.blendshape] = (
                    frame.morphs[index] if index is not None
                    else 0.0)
            # Bone rotations compose as quaternions, per bone,
            # in mapping order — the clone's composition.
            quats: dict[str, _Quat] = {}
            for mapping, csv_name in bone_entries:
                index = order.get(csv_name)
                value = (frame.morphs[index]
                         if index is not None else 0.0)
                angle = (value * math.radians(180.0)
                         * mapping.scale
                         + math.radians(mapping.offset))
                step = _axis_angle(mapping.axis, angle)
                bone = mapping.bone
                # Bevy pre-multiplies each step (rotate_axis is
                # rotation = step * rotation), so the later
                # mapping is the outermost rotation.
                quats[bone] = (_quat_mul(step, quats[bone])
                               if bone in quats else step)
            bones = {bone: _quat_to_euler_zxy(quat)
                     for bone, quat in quats.items()}
            blend_frames.append(blend)
            bone_frames.append(bones)
        return {"blendshapes": blend_frames,
                "bones": bone_frames,
                "unmapped": list(unmapped)}


def _bone_mapping(raw: dict[str, Any], where: str,
                  index: int) -> BoneMapping:
    """One bone object: ``{path, axis, offset?, scale?}``."""
    path = raw.get("path")
    if (not isinstance(path, list) or not path
            or not all(isinstance(p, str) and p for p in path)):
        raise FaceMocapError(
            f"{where} entry {index}: a bone mapping needs a "
            f"non-empty 'path' list of bone names",
            kind="bad_mapping")
    axis = raw.get("axis")
    if isinstance(axis, str):
        axes = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0),
                "Z": (0.0, 0.0, 1.0)}
        vector = axes.get(axis.upper())
        if vector is None:
            raise FaceMocapError(
                f"{where} entry {index}: bone axis {axis!r} "
                f"must be X, Y, Z, or [x, y, z]",
                kind="bad_mapping")
    elif (isinstance(axis, list) and len(axis) == 3
            and all(isinstance(v, (int, float)) for v in axis)):
        vector = (float(axis[0]), float(axis[1]),
                  float(axis[2]))
    else:
        raise FaceMocapError(
            f"{where} entry {index}: bone axis must be X, Y, Z, "
            f"or [x, y, z]",
            kind="bad_mapping")
    offset = raw.get("offset", 0.0)
    scale = raw.get("scale", 1.0)
    if not isinstance(offset, (int, float)):
        raise FaceMocapError(
            f"{where} entry {index}: bone offset must be a "
            f"number (degrees)",
            kind="bad_mapping")
    if not isinstance(scale, (int, float)):
        raise FaceMocapError(
            f"{where} entry {index}: bone scale must be a number",
            kind="bad_mapping")
    return BoneMapping(path=tuple(path), axis=vector,
                       offset=float(offset), scale=float(scale))


def default_arkit_mapping() -> FaceMapping:
    """The bundled ARKit rename table: LiveLink CSVs carry
    PascalCase ARKit names (``EyeBlinkLeft``); Devon's rig
    carries camelCase (``eyeBlinkLeft``). The table maps all 52
    — the rename the mapping file exists to do, bundled so a
    real recording loads without hand-writing it."""
    return FaceMapping.from_file(default_mapping_path())
