"""Scene3D — the 3D rung's shared spine (plan §1.5e).

The owner's decision (2026-11-05) closed the §1.5e either/or: **both**
engine rungs get built — a native Rust renderer and a Blender-class
external engine, driven as a substrate — and the profile runtime picks
per device (standing max-capability doctrine; phone constraints do not
shape the design). What neither engine may do is invent its own scene
format: this module is the one spine both plug into.

Three pieces, in the repo's own idioms:

* :class:`Scene3D` — the document, in the D1 idiom of
  :class:`~.document.Document` and ``MotionProject``: a scene graph
  (nodes with :class:`Transform` s, parented by name), cameras (the
  ``camera3d`` property group made real — position, rotation or
  look-at, fov, near/far), lights and materials as data, meshes as
  data (named primitives + explicit vertex/face shapes), and
  animation as :class:`AnimTrack` keyframes on integer frames of the
  scene's :class:`~.frametime.FrameGrid`. JSON-pure
  ``to_dict``/``from_dict`` round-trip, ``compute_hash``/``seal()``
  identity, save/load like ``EditStudio`` projects.

* :func:`scene_state_at` — the deterministic frame function: the
  scene's full evaluated state at one frame (world matrices, the
  active camera's view/projection, resolved lights and materials,
  animated values applied) as comparable data. A pure function of
  (document, frame) — no wall clock, no RNG, no host state. Engines
  rasterize this state; the state is what the golden harness pins,
  so two engines that agree on state differ only in pixels.

* The renderer registry — the §1.1e contract's honesty rules,
  3D-shaped. A rung registers a :class:`SceneRenderer`: capabilities
  as comparable data (:class:`RendererCapabilities` — surfaces,
  node/mesh/material/light kinds, animated props, a declared max
  canvas, a determinism flag), a :meth:`~SceneRenderer.probe`
  (available + the reason when not), and
  :meth:`~SceneRenderer.render_frames` producing a frame sequence
  into the pipeline's existing image-sequence shape (an
  ``f_%04d.png``-style pattern + framerate, the ``genedit`` shape).
  :func:`resolve_renderer` is capability-checked composition: the
  first rung (registration order) that probes available *and* covers
  the scene's requirements serves; every other rung's refusal is
  recorded with its reason.

The legacy seams resolve through this registry: ``MotionProject``'s
``camera3d/*`` compile gate, the format registry's ``camera3d``
renderer, and the effects registry's deferred ``camera3d`` all ask
:func:`resolve_renderer` (per surface) before refusing. **While no
rung is registered, every one of those refusals fires exactly as it
did before this module existed** — the spine changes no user-visible
behavior until an engine lands.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ...media_edit.videos import MediaEditError
from ..directed.motion_score import EASE_FUNCS, ease_value
from .frametime import FrameGrid

__all__ = [
    "Scene3DError",
    "Vec3", "Quat", "Mat4",
    "Rotation", "Transform",
    "Camera", "Light", "Material", "Mesh", "SceneNode",
    "AnimKey", "AnimTrack", "ANIMATED_PROPS",
    "Scene3D", "FrameState", "scene_state_at", "scene_state_dict",
    "NODE_KINDS", "MESH_KINDS", "PRIMITIVES", "MATERIAL_KINDS",
    "LIGHT_KINDS", "RENDERER_SURFACES",
    "RendererCapabilities", "RendererProbe", "SceneRenderer",
    "register_renderer", "unregister_renderer", "reset_renderers",
    "registered_renderers", "get_renderer", "renderer_status",
    "scene_requirements", "capability_gaps", "resolve_renderer",
    "render_scene",
]

_DOC_KIND = "devon-scene3d"
_DOC_VERSION = 1


class Scene3DError(MediaEditError):
    """A Scene3D document, evaluation, or resolution failure."""


# ---------------------------------------------------------------------------
# math — pure stdlib, tuples, right-handed, camera looks down -Z
# ---------------------------------------------------------------------------

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]        # (x, y, z, w)
Mat4 = tuple[float, ...]                        # 16 floats, row-major

_IDENTITY: Mat4 = (1.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 0.0, 1.0)


def _vec3(value: Any, what: str = "vector") -> Vec3:
    try:
        x, y, z = value
        return (float(x), float(y), float(z))
    except Exception:
        raise Scene3DError(
            f"{what} must be a 3-component vector, got {value!r}") \
            from None


def _quat(value: Any, what: str = "quaternion") -> Quat:
    try:
        x, y, z, w = value
        return _quat_normalize((float(x), float(y), float(z), float(w)))
    except Scene3DError:
        raise
    except Exception:
        raise Scene3DError(
            f"{what} must be a 4-component quaternion (x, y, z, w), "
            f"got {value!r}") from None


def _quat_normalize(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q))
    if n <= 0.0:
        raise Scene3DError("zero-length quaternion has no rotation")
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def _euler_matrix(x_deg: float, y_deg: float, z_deg: float) -> Mat4:
    """Rotation matrix for euler degrees, XYZ order: R = Rz·Ry·Rx
    (the X rotation applies first to a column vector)."""
    ax, ay, az = (math.radians(v) for v in (x_deg, y_deg, z_deg))
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    # Ry·Rx, then Rz·(Ry·Rx) — expanded once, no drift-prone chaining.
    r00, r01, r02 = cy, sy * sx, sy * cx
    r10, r11, r12 = 0.0, cx, -sx
    r20, r21, r22 = -sy, cy * sx, cy * cx
    return (
        cz * r00 - sz * r10, cz * r01 - sz * r11, cz * r02 - sz * r12, 0.0,
        sz * r00 + cz * r10, sz * r01 + cz * r11, sz * r02 + cz * r12, 0.0,
        r20, r21, r22, 0.0,
        0.0, 0.0, 0.0, 1.0)


def _quat_matrix(q: Quat) -> Mat4:
    x, y, z, w = q
    return (
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0,
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0,
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0,
        0.0, 0.0, 0.0, 1.0)


def _matrix_quat(m: Mat4) -> Quat:
    """Rotation matrix → quaternion (largest-component branch)."""
    r00, r01, r02 = m[0], m[1], m[2]
    r10, r11, r12 = m[4], m[5], m[6]
    r20, r21, r22 = m[8], m[9], m[10]
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return _quat_normalize(((r21 - r12) / s, (r02 - r20) / s,
                                (r10 - r01) / s, 0.25 * s))
    if r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        return _quat_normalize((0.25 * s, (r01 + r10) / s,
                                (r02 + r20) / s, (r21 - r12) / s))
    if r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        return _quat_normalize(((r01 + r10) / s, 0.25 * s,
                                (r12 + r21) / s, (r02 - r20) / s))
    s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
    return _quat_normalize(((r02 + r20) / s, (r12 + r21) / s,
                            0.25 * s, (r10 - r01) / s))


def _mat_mul(a: Mat4, b: Mat4) -> Mat4:
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            out[r * 4 + c] = sum(a[r * 4 + k] * b[k * 4 + c]
                                 for k in range(4))
    return tuple(out)


def _mat_point(m: Mat4, p: Vec3) -> Vec3:
    x, y, z = p
    return (m[0] * x + m[1] * y + m[2] * z + m[3],
            m[4] * x + m[5] * y + m[6] * z + m[7],
            m[8] * x + m[9] * y + m[10] * z + m[11])


def _mat_dir(m: Mat4, p: Vec3) -> Vec3:
    x, y, z = p
    return (m[0] * x + m[1] * y + m[2] * z,
            m[4] * x + m[5] * y + m[6] * z,
            m[8] * x + m[9] * y + m[10] * z)


def _normalize(v: Vec3, what: str) -> Vec3:
    n = math.sqrt(sum(c * c for c in v))
    if n <= 0.0:
        raise Scene3DError(f"cannot normalize a zero {what}")
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _lookat_quat(eye: Vec3, target: Vec3, up: Vec3) -> Quat:
    """Quaternion orienting -Z from ``eye`` toward ``target``."""
    if eye == target:
        raise Scene3DError("look-at target equals the eye position")
    zaxis = _normalize(_sub(eye, target), "view direction")
    xaxis = _cross(up, zaxis)
    if sum(c * c for c in xaxis) <= 1e-12:
        raise Scene3DError("look-at up vector is parallel to the "
                           "view direction")
    xaxis = _normalize(xaxis, "look-at side vector")
    yaxis = _cross(zaxis, xaxis)
    m: Mat4 = (xaxis[0], yaxis[0], zaxis[0], 0.0,
               xaxis[1], yaxis[1], zaxis[1], 0.0,
               xaxis[2], yaxis[2], zaxis[2], 0.0,
               0.0, 0.0, 0.0, 1.0)
    return _matrix_quat(m)


def _perspective(fov_y_deg: float, aspect: float, near: float,
                 far: float) -> Mat4:
    """OpenGL-style perspective projection (NDC z in [-1, 1])."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    return (f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, (far + near) / (near - far),
            (2.0 * far * near) / (near - far),
            0.0, 0.0, -1.0, 0.0)


def _view_matrix(world: Mat4) -> Mat4:
    """Inverse of a rigid world matrix (rotation + translation)."""
    r = world
    tx, ty, tz = r[3], r[7], r[11]
    return (r[0], r[4], r[8], -(r[0] * tx + r[4] * ty + r[8] * tz),
            r[1], r[5], r[9], -(r[1] * tx + r[5] * ty + r[9] * tz),
            r[2], r[6], r[10], -(r[2] * tx + r[6] * ty + r[10] * tz),
            0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# transforms, cameras, lights, materials, meshes — the document's data
# ---------------------------------------------------------------------------

ROTATION_KINDS = ("euler", "quaternion", "lookat")


@dataclass(frozen=True)
class Rotation:
    """An orientation, one of three authored shapes.

    ``euler`` — degrees, XYZ order (see :func:`_euler_matrix`).
    ``quaternion`` — (x, y, z, w), normalized at construction.
    ``lookat`` — aim -Z at ``target`` (a point in the *parent's*
    space, i.e. the same space the owning transform's position
    lives in) with ``up`` as the roll reference.
    """

    kind: str = "euler"
    euler: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    target: Vec3 = (0.0, 0.0, -1.0)
    up: Vec3 = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        if self.kind not in ROTATION_KINDS:
            raise Scene3DError(
                f"rotation kind must be one of {ROTATION_KINDS}, "
                f"got {self.kind!r}")
        object.__setattr__(self, "euler", _vec3(self.euler, "euler"))
        object.__setattr__(self, "quat", _quat(self.quat))
        object.__setattr__(self, "target", _vec3(self.target, "target"))
        object.__setattr__(self, "up", _vec3(self.up, "up"))

    @classmethod
    def from_euler(cls, x: float, y: float, z: float) -> Rotation:
        return cls(kind="euler", euler=(float(x), float(y), float(z)))

    @classmethod
    def from_quaternion(cls, q: Sequence[float]) -> Rotation:
        return cls(kind="quaternion", quat=_quat(q))

    @classmethod
    def look_at(cls, target: Sequence[float],
                up: Sequence[float] = (0.0, 1.0, 0.0)) -> Rotation:
        return cls(kind="lookat", target=_vec3(target, "target"),
                   up=_vec3(up, "up"))

    def to_quat(self, position: Vec3) -> Quat:
        """This rotation as a quaternion, resolved at ``position``
        (the look-at shape needs the eye point)."""
        if self.kind == "quaternion":
            return self.quat
        if self.kind == "lookat":
            return _lookat_quat(position, self.target, self.up)
        return _matrix_quat(_euler_matrix(*self.euler))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "euler": list(self.euler),
                "quat": list(self.quat), "target": list(self.target),
                "up": list(self.up)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Rotation:
        return cls(kind=str(d.get("kind", "euler")),
                   euler=_vec3(d.get("euler", (0, 0, 0)), "euler"),
                   quat=_quat(d.get("quat", (0, 0, 0, 1))),
                   target=_vec3(d.get("target", (0, 0, -1)), "target"),
                   up=_vec3(d.get("up", (0, 1, 0)), "up"))


@dataclass(frozen=True)
class Transform:
    """Position + rotation + scale, in the parent's space."""

    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Rotation = field(default_factory=Rotation)
    scale: Vec3 = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position",
                           _vec3(self.position, "position"))
        object.__setattr__(self, "scale", _vec3(self.scale, "scale"))
        if not isinstance(self.rotation, Rotation):
            object.__setattr__(
                self, "rotation",
                Rotation.from_dict(dict(self.rotation)))

    def matrix(self) -> Mat4:
        """The local TRS matrix: T · R · S."""
        rot = _quat_matrix(self.rotation.to_quat(self.position))
        sx, sy, sz = self.scale
        scaled: Mat4 = (rot[0] * sx, rot[1] * sy, rot[2] * sz, 0.0,
                        rot[4] * sx, rot[5] * sy, rot[6] * sz, 0.0,
                        rot[8] * sx, rot[9] * sy, rot[10] * sz, 0.0,
                        0.0, 0.0, 0.0, 1.0)
        px, py, pz = self.position
        return (scaled[0], scaled[1], scaled[2], px,
                scaled[4], scaled[5], scaled[6], py,
                scaled[8], scaled[9], scaled[10], pz,
                0.0, 0.0, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {"position": list(self.position),
                "rotation": self.rotation.to_dict(),
                "scale": list(self.scale)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transform:
        return cls(position=_vec3(d.get("position", (0, 0, 0)),
                                  "position"),
                   rotation=Rotation.from_dict(
                       dict(d.get("rotation") or {})),
                   scale=_vec3(d.get("scale", (1, 1, 1)), "scale"))


@dataclass
class Camera:
    """The ``camera3d`` group made real: a transform (position +
    rotation-or-lookat), vertical fov in degrees, near/far planes.
    ``aspect`` is width/height; ``None`` resolves to the scene's
    canvas at evaluation."""

    transform: Transform = field(default_factory=Transform)
    fov: float = 50.0
    near: float = 0.1
    far: float = 1000.0
    aspect: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transform, Transform):
            self.transform = Transform.from_dict(dict(self.transform))
        self.fov = float(self.fov)
        if not 0.0 < self.fov < 180.0:
            raise Scene3DError(
                f"camera fov must be in (0, 180), got {self.fov}")
        self.near = float(self.near)
        self.far = float(self.far)
        if not 0.0 < self.near < self.far:
            raise Scene3DError(
                f"camera planes need 0 < near < far, got "
                f"near={self.near} far={self.far}")
        if self.aspect is not None:
            self.aspect = float(self.aspect)
            if self.aspect <= 0:
                raise Scene3DError("camera aspect must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {"transform": self.transform.to_dict(),
                "fov": self.fov, "near": self.near, "far": self.far,
                "aspect": self.aspect}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Camera:
        return cls(transform=Transform.from_dict(
                       dict(d.get("transform") or {})),
                   fov=float(d.get("fov", 50.0)),
                   near=float(d.get("near", 0.1)),
                   far=float(d.get("far", 1000.0)),
                   aspect=d.get("aspect"))


LIGHT_KINDS = ("ambient", "directional", "point", "spot")


@dataclass
class Light:
    """A light payload on a node. Directional/spot lights shine
    along their node's world -Z; point/spot falloff is the engine's
    shading concern — the state carries the evaluated facts."""

    kind: str = "point"
    color: Vec3 = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    spot_angle: float = 45.0        # degrees, spot only
    spot_penumbra: float = 0.5      # 0..1, spot only

    def __post_init__(self) -> None:
        if self.kind not in LIGHT_KINDS:
            raise Scene3DError(
                f"light kind must be one of {LIGHT_KINDS}, "
                f"got {self.kind!r}")
        self.color = _vec3(self.color, "light color")
        if any(c < 0.0 for c in self.color):
            raise Scene3DError("light color components must be >= 0")
        self.intensity = float(self.intensity)
        if self.intensity < 0.0:
            raise Scene3DError("light intensity must be >= 0")
        self.spot_angle = float(self.spot_angle)
        if not 0.0 < self.spot_angle <= 180.0:
            raise Scene3DError("spot_angle must be in (0, 180]")
        self.spot_penumbra = float(self.spot_penumbra)
        if not 0.0 <= self.spot_penumbra <= 1.0:
            raise Scene3DError("spot_penumbra must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "color": list(self.color),
                "intensity": self.intensity,
                "spot_angle": self.spot_angle,
                "spot_penumbra": self.spot_penumbra}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Light:
        return cls(kind=str(d.get("kind", "point")),
                   color=_vec3(d.get("color", (1, 1, 1)), "light color"),
                   intensity=float(d.get("intensity", 1.0)),
                   spot_angle=float(d.get("spot_angle", 45.0)),
                   spot_penumbra=float(d.get("spot_penumbra", 0.5)))


MATERIAL_KINDS = ("flat", "lambert", "phong")


@dataclass
class Material:
    """A shading model as data. ``flat`` = unlit color; ``lambert``
    = diffuse; ``phong`` adds the specular ``shininess`` lobe.
    ``emissive`` is light the surface adds on its own."""

    kind: str = "lambert"
    color: Vec3 = (1.0, 1.0, 1.0)
    opacity: float = 1.0
    shininess: float = 32.0
    emissive: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.kind not in MATERIAL_KINDS:
            raise Scene3DError(
                f"material kind must be one of {MATERIAL_KINDS}, "
                f"got {self.kind!r}")
        self.color = _vec3(self.color, "material color")
        self.emissive = _vec3(self.emissive, "material emissive")
        if any(c < 0.0 for c in self.color + self.emissive):
            raise Scene3DError("material colors must be >= 0")
        self.opacity = float(self.opacity)
        if not 0.0 <= self.opacity <= 1.0:
            raise Scene3DError("material opacity must be in [0, 1]")
        self.shininess = float(self.shininess)
        if self.shininess < 0.0:
            raise Scene3DError("material shininess must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "color": list(self.color),
                "opacity": self.opacity, "shininess": self.shininess,
                "emissive": list(self.emissive)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Material:
        return cls(kind=str(d.get("kind", "lambert")),
                   color=_vec3(d.get("color", (1, 1, 1)),
                               "material color"),
                   opacity=float(d.get("opacity", 1.0)),
                   shininess=float(d.get("shininess", 32.0)),
                   emissive=_vec3(d.get("emissive", (0, 0, 0)),
                                  "material emissive"))


MESH_KINDS = ("primitive", "explicit")
PRIMITIVES = ("cube", "sphere", "plane", "cylinder", "cone")


@dataclass
class Mesh:
    """Geometry as data: a named primitive (``params`` are numeric
    shape parameters — ``size``, ``radius``, ``height``,
    ``segments``, … — the engine interprets them) or explicit
    vertices/faces (faces are index tuples, 3+ corners each;
    ``normals`` optional per-vertex)."""

    kind: str = "primitive"
    primitive: str = "cube"
    params: dict[str, float] = field(default_factory=dict)
    vertices: list[Vec3] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    normals: list[Vec3] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in MESH_KINDS:
            raise Scene3DError(
                f"mesh kind must be one of {MESH_KINDS}, "
                f"got {self.kind!r}")
        if self.kind == "primitive":
            if self.primitive not in PRIMITIVES:
                raise Scene3DError(
                    f"primitive must be one of {PRIMITIVES}, "
                    f"got {self.primitive!r}")
            clean: dict[str, float] = {}
            for k, v in (self.params or {}).items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise Scene3DError(
                        f"primitive param {k!r} must be a number, "
                        f"got {v!r}")
                clean[str(k)] = float(v)
            self.params = clean
            return
        self.vertices = [_vec3(v, "vertex") for v in self.vertices]
        if not self.vertices:
            raise Scene3DError("explicit mesh needs vertices")
        faces: list[tuple[int, ...]] = []
        for f in self.faces:
            face = tuple(int(i) for i in f)
            if len(face) < 3:
                raise Scene3DError(
                    f"mesh face needs 3+ corners, got {face!r}")
            for i in face:
                if not 0 <= i < len(self.vertices):
                    raise Scene3DError(
                        f"mesh face index {i} out of range "
                        f"({len(self.vertices)} vertices)")
            faces.append(face)
        if not faces:
            raise Scene3DError("explicit mesh needs faces")
        self.faces = faces
        if self.normals:
            self.normals = [_vec3(n, "normal") for n in self.normals]
            if len(self.normals) != len(self.vertices):
                raise Scene3DError(
                    "mesh normals must match the vertex count "
                    f"({len(self.normals)} != {len(self.vertices)})")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "primitive": self.primitive,
                "params": dict(self.params),
                "vertices": [list(v) for v in self.vertices],
                "faces": [list(f) for f in self.faces],
                "normals": [list(n) for n in self.normals]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Mesh:
        return cls(kind=str(d.get("kind", "primitive")),
                   primitive=str(d.get("primitive", "cube")),
                   params=dict(d.get("params") or {}),
                   vertices=[_vec3(v, "vertex")
                             for v in d.get("vertices") or []],
                   faces=[tuple(int(i) for i in f)
                          for f in d.get("faces") or []],
                   normals=[_vec3(n, "normal")
                            for n in d.get("normals") or []])


#: node kinds, derived from payloads: a node with a mesh is a
#: "mesh" node, with a light a "light" node; a bare node groups.
NODE_KINDS = ("mesh", "light", "group")


@dataclass
class SceneNode:
    """One scene-graph node. ``parent`` names another node (or
    ``None`` for a root). ``material`` names a scene material.
    A node carries at most one of ``mesh`` / ``light`` as its
    payload; a node with neither is a group."""

    name: str
    transform: Transform = field(default_factory=Transform)
    parent: str | None = None
    mesh: Mesh | None = None
    material: str | None = None
    light: Light | None = None
    visible: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise Scene3DError("scene node needs a name")
        self.name = str(self.name)
        if not isinstance(self.transform, Transform):
            self.transform = Transform.from_dict(dict(self.transform))
        if self.mesh is not None and not isinstance(self.mesh, Mesh):
            self.mesh = Mesh.from_dict(dict(self.mesh))
        if self.light is not None and not isinstance(self.light, Light):
            self.light = Light.from_dict(dict(self.light))
        if self.mesh is not None and self.light is not None:
            raise Scene3DError(
                f"node {self.name!r} carries both a mesh and a "
                f"light — one payload per node")
        if self.material is not None and self.mesh is None:
            raise Scene3DError(
                f"node {self.name!r} names material "
                f"{self.material!r} but carries no mesh")
        self.visible = bool(self.visible)
        self.extras = copy.deepcopy(self.extras or {})

    @property
    def kind(self) -> str:
        if self.mesh is not None:
            return "mesh"
        if self.light is not None:
            return "light"
        return "group"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name,
                "transform": self.transform.to_dict(),
                "parent": self.parent,
                "mesh": self.mesh.to_dict() if self.mesh else None,
                "material": self.material,
                "light": self.light.to_dict() if self.light else None,
                "visible": self.visible,
                "extras": copy.deepcopy(self.extras)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SceneNode:
        return cls(name=str(d.get("name", "")),
                   transform=Transform.from_dict(
                       dict(d.get("transform") or {})),
                   parent=d.get("parent"),
                   mesh=Mesh.from_dict(dict(d["mesh"]))
                   if d.get("mesh") else None,
                   material=d.get("material"),
                   light=Light.from_dict(dict(d["light"]))
                   if d.get("light") else None,
                   visible=bool(d.get("visible", True)),
                   extras=copy.deepcopy(d.get("extras") or {}))


# ---------------------------------------------------------------------------
# animation — keyframed property paths on the frame grid
# ---------------------------------------------------------------------------

#: animatable property paths, by target scope. The target grammar is
#: ``<scope>/<name>/<prop>`` with scope ``nodes`` | ``cameras`` |
#: ``materials``; the value is the sampled tuple's arity.
ANIMATED_PROPS: dict[str, dict[str, int]] = {
    "nodes": {"position": 3, "rotation": 3, "scale": 3,
              "intensity": 1},
    "cameras": {"position": 3, "fov": 1, "near": 1, "far": 1},
    "materials": {"color": 3, "opacity": 1},
}


def _parse_target(target: str) -> tuple[str, str, str]:
    parts = str(target).split("/")
    if len(parts) != 3 or parts[0] not in ANIMATED_PROPS:
        raise Scene3DError(
            f"animation target {target!r} must be "
            f"<nodes|cameras|materials>/<name>/<prop>")
    scope, name, prop = parts
    if prop not in ANIMATED_PROPS[scope]:
        raise Scene3DError(
            f"property {prop!r} is not animatable on {scope} — "
            f"animatable: "
            f"{', '.join(sorted(ANIMATED_PROPS[scope]))}")
    return scope, name, prop


@dataclass
class AnimKey:
    """One keyframe: at integer ``frame``, the value tuple.
    ``easing`` names the curve INTO this key from the previous one
    (the PropKey convention), drawn from the expression layer's
    easing library (``directed/motion_score``)."""

    frame: int
    values: tuple[float, ...]
    easing: str = "smooth"

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or int(self.frame) != self.frame:
            raise Scene3DError(
                f"animation key frame must be an integer frame "
                f"count, got {self.frame!r}")
        self.frame = int(self.frame)
        if self.frame < 0:
            raise Scene3DError("animation key frame must be >= 0")
        self.values = tuple(float(v) for v in self.values)
        if not self.values:
            raise Scene3DError("animation key needs at least one value")
        self.easing = str(self.easing or "smooth").lower()
        if self.easing not in EASE_FUNCS:
            raise Scene3DError(
                f"unknown easing {self.easing!r} — the expression "
                f"layer speaks: {', '.join(sorted(EASE_FUNCS))}")

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "values": list(self.values),
                "easing": self.easing}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnimKey:
        return cls(frame=d["frame"],
                   values=tuple(float(v) for v in d["values"]),
                   easing=str(d.get("easing", "smooth")))


@dataclass
class AnimTrack:
    """Keyframes for one property path (``target``), in frames.

    Sampling mirrors ``PropertyTrack``: between keys, progress is
    eased by the destination key's easing, component-wise; outside
    the key range the end values hold."""

    target: str
    keys: list[AnimKey] = field(default_factory=list)

    def __post_init__(self) -> None:
        scope, _name, prop = _parse_target(self.target)
        arity = ANIMATED_PROPS[scope][prop]
        self.keys = [k if isinstance(k, AnimKey) else AnimKey.from_dict(k)
                     for k in self.keys]
        if not self.keys:
            raise Scene3DError(
                f"animation track {self.target!r} has no keyframes")
        for k in self.keys:
            if len(k.values) != arity:
                raise Scene3DError(
                    f"animation track {self.target!r}: keys carry "
                    f"{arity} values, got {len(k.values)}")
        self.keys.sort(key=lambda k: k.frame)
        for a, b in pairwise(self.keys):
            if b.frame <= a.frame:
                raise Scene3DError(
                    f"animation track {self.target!r}: key frames "
                    f"must be strictly increasing (frame {b.frame} "
                    f"repeats)")

    def sample(self, frame: float) -> tuple[float, ...]:
        ks = self.keys
        f = float(frame)
        if f <= ks[0].frame:
            return ks[0].values
        if f >= ks[-1].frame:
            return ks[-1].values
        for prev, nxt in pairwise(ks):
            if prev.frame <= f <= nxt.frame:
                span = nxt.frame - prev.frame
                u = 0.0 if span <= 0 else (f - prev.frame) / span
                e = ease_value(nxt.easing, u)
                return tuple(a + (b - a) * e
                             for a, b in zip(prev.values, nxt.values, strict=True))
        return ks[-1].values

    @property
    def last_frame(self) -> int:
        return self.keys[-1].frame

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target,
                "keys": [k.to_dict() for k in self.keys]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnimTrack:
        return cls(target=str(d.get("target", "")),
                   keys=[AnimKey.from_dict(k)
                         for k in d.get("keys") or []])


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------

def _canonical(value: Any) -> Any:
    """Document._canonical's discipline, mirrored (sorted keys,
    JSON primitives, to_dict-able values) for the scene hash."""
    if isinstance(value, dict):
        return {str(k): _canonical(value[k])
                for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    raise Scene3DError(
        f"not JSON-serializable in Scene3D: {type(value).__name__}")


@dataclass
class Scene3D:
    """A 3D scene as a document — the spine both engine rungs
    render. Pure-JSON round-trip, ``seal()`` snapshot identity,
    save/load like ``EditStudio`` projects (the D1 idiom).

    ``output`` carries the canvas (``width``/``height``/``fps``) —
    a render hint the resolver checks against a rung's declared
    max, not scene identity. ``duration_frames`` is derived from
    the animation tracks (0 for a still scene)."""

    title: str = ""
    nodes: list[SceneNode] = field(default_factory=list)
    cameras: dict[str, Camera] = field(default_factory=dict)
    active_camera: str = ""
    materials: dict[str, Material] = field(default_factory=dict)
    tracks: list[AnimTrack] = field(default_factory=list)
    background: Vec3 = (0.0, 0.0, 0.0)
    output: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    doc_hash: str = ""

    def __post_init__(self) -> None:
        self.nodes = [n if isinstance(n, SceneNode)
                      else SceneNode.from_dict(dict(n))
                      for n in self.nodes]
        self.cameras = {str(k): (v if isinstance(v, Camera)
                                 else Camera.from_dict(dict(v)))
                        for k, v in (self.cameras or {}).items()}
        self.materials = {str(k): (v if isinstance(v, Material)
                                   else Material.from_dict(dict(v)))
                          for k, v in (self.materials or {}).items()}
        self.tracks = [t if isinstance(t, AnimTrack)
                       else AnimTrack.from_dict(dict(t))
                       for t in self.tracks]
        self.background = _vec3(self.background, "background")
        self.output = dict(self.output or {})
        self.output.setdefault("width", 1920)
        self.output.setdefault("height", 1080)
        self.output.setdefault("fps", 30)
        self.version = int(self.version)
        self._validate()

    def _validate(self) -> None:
        names: set[str] = set()
        for node in self.nodes:
            if node.name in names:
                raise Scene3DError(
                    f"duplicate scene node name {node.name!r}")
            names.add(node.name)
        for node in self.nodes:
            if node.parent is not None and node.parent not in names:
                raise Scene3DError(
                    f"node {node.name!r} parents to unknown node "
                    f"{node.parent!r}")
            if node.material is not None \
                    and node.material not in self.materials:
                raise Scene3DError(
                    f"node {node.name!r} names unknown material "
                    f"{node.material!r}")
        by_name = {n.name: n for n in self.nodes}
        for node in self.nodes:
            seen: set[str] = set()
            cur: SceneNode | None = node
            while cur is not None and cur.parent is not None:
                if cur.parent in seen:
                    raise Scene3DError(
                        f"scene graph cycle through {cur.parent!r}")
                seen.add(cur.parent)
                cur = by_name[cur.parent]
        if self.active_camera \
                and self.active_camera not in self.cameras:
            raise Scene3DError(
                f"active camera {self.active_camera!r} is not a "
                f"scene camera ({', '.join(sorted(self.cameras))})")
        seen_targets: set[str] = set()
        for track in self.tracks:
            scope, name, prop = _parse_target(track.target)
            if track.target in seen_targets:
                raise Scene3DError(
                    f"duplicate animation track for {track.target!r}")
            seen_targets.add(track.target)
            if scope == "nodes":
                node = by_name.get(name)
                if node is None:
                    raise Scene3DError(
                        f"animation track {track.target!r}: no node "
                        f"{name!r}")
                if prop == "intensity" and node.light is None:
                    raise Scene3DError(
                        f"animation track {track.target!r}: node "
                        f"{name!r} carries no light to animate")
                if prop == "rotation" \
                        and node.transform.rotation.kind != "euler":
                    raise Scene3DError(
                        f"animation track {track.target!r}: rotation "
                        f"tracks animate euler degrees, but node "
                        f"{name!r} uses a "
                        f"{node.transform.rotation.kind} rotation")
            elif scope == "cameras":
                if name not in self.cameras:
                    raise Scene3DError(
                        f"animation track {track.target!r}: no "
                        f"camera {name!r}")
            elif name not in self.materials:
                raise Scene3DError(
                    f"animation track {track.target!r}: no material "
                    f"{name!r}")

    # -- identity ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"kind": _DOC_KIND, "version": _DOC_VERSION,
                "title": self.title,
                "nodes": [n.to_dict() for n in self.nodes],
                "cameras": {k: self.cameras[k].to_dict()
                            for k in sorted(self.cameras)},
                "active_camera": self.active_camera,
                "materials": {k: self.materials[k].to_dict()
                              for k in sorted(self.materials)},
                "tracks": [t.to_dict() for t in self.tracks],
                "background": list(self.background),
                "output": dict(self.output),
                "doc_version": self.version,
                "doc_hash": self.doc_hash}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene3D:
        if not isinstance(data, dict):
            raise Scene3DError("scene3d document must be a dict")
        if data.get("kind") not in (None, _DOC_KIND):
            raise Scene3DError(
                f"not a scene3d document (kind={data.get('kind')!r})")
        if int(data.get("version", _DOC_VERSION)) != _DOC_VERSION:
            raise Scene3DError(
                f"unsupported scene3d version {data.get('version')}")
        return cls(title=str(data.get("title", "")),
                   nodes=[SceneNode.from_dict(n)
                          for n in data.get("nodes") or []],
                   cameras={str(k): Camera.from_dict(dict(v))
                            for k, v in
                            (data.get("cameras") or {}).items()},
                   active_camera=str(data.get("active_camera", "")),
                   materials={str(k): Material.from_dict(dict(v))
                              for k, v in
                              (data.get("materials") or {}).items()},
                   tracks=[AnimTrack.from_dict(t)
                           for t in data.get("tracks") or []],
                   background=_vec3(data.get("background", (0, 0, 0)),
                                    "background"),
                   output=dict(data.get("output") or {}),
                   version=int(data.get("doc_version", 1)),
                   doc_hash=str(data.get("doc_hash", "")))

    def compute_hash(self) -> str:
        """SHA-256 of the canonical document (hash field excluded)."""
        raw = self.to_dict()
        raw.pop("doc_hash", None)
        blob = json.dumps(_canonical(raw), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def seal(self) -> Scene3D:
        """Bump version + stamp hash. Call after any edit."""
        self.version = int(self.version) + 1
        self.doc_hash = self.compute_hash()
        return self

    def is_dirty(self) -> bool:
        if not self.doc_hash:
            return False
        return self.compute_hash() != self.doc_hash

    def copy(self) -> Scene3D:
        return Scene3D.from_dict(self.to_dict())

    def save(self, path: str | os.PathLike) -> str:
        """Save like EditStudio projects: one JSON file, sealed."""
        self.seal()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2),
                     encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str | os.PathLike) -> Scene3D:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise Scene3DError(
                f"could not load scene3d {path}: {exc}") from exc
        return cls.from_dict(data)

    # -- lookups ---------------------------------------------------------
    def node(self, name: str) -> SceneNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise Scene3DError(
            f"no scene node {name!r} — scene has: "
            f"{', '.join(n.name for n in self.nodes)}")

    def camera(self, name: str | None = None) -> Camera:
        key = name or self.active_camera
        if key:
            try:
                return self.cameras[key]
            except KeyError:
                raise Scene3DError(
                    f"no camera {key!r} — scene has: "
                    f"{', '.join(sorted(self.cameras))}") from None
        if not self.cameras:
            raise Scene3DError("scene has no camera")
        return self.cameras[sorted(self.cameras)[0]]

    def camera_name(self) -> str:
        """The active camera's name (explicit, else first sorted)."""
        if self.active_camera:
            return self.active_camera
        if not self.cameras:
            raise Scene3DError("scene has no camera")
        return sorted(self.cameras)[0]

    @property
    def grid(self) -> FrameGrid:
        return FrameGrid(self.output.get("fps", 30))

    @property
    def duration_frames(self) -> int:
        return max((t.last_frame for t in self.tracks), default=0)


# ---------------------------------------------------------------------------
# the deterministic frame function
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameState:
    """The evaluated scene at one frame — comparable data, the
    golden harness's unit. Matrices are row-major 16-tuples;
    ``seconds`` is the exact grid time as a float at the API edge
    (the frame count is the identity)."""

    frame: int
    seconds: float
    background: Vec3
    camera: dict[str, Any]
    nodes: dict[str, dict[str, Any]]
    materials: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "seconds": self.seconds,
                "background": list(self.background),
                "camera": copy.deepcopy(self.camera),
                "nodes": copy.deepcopy(self.nodes),
                "materials": copy.deepcopy(self.materials)}


def scene_state_at(scene: Scene3D, frame: int) -> FrameState:
    """The scene's state at ``frame`` — a pure function of the
    document and the frame count. Double evaluation is identical;
    the document is not mutated."""
    if isinstance(frame, bool) or int(frame) != frame:
        raise Scene3DError(
            f"frame state is addressed in integer frames, "
            f"got {frame!r}")
    frame = int(frame)
    if frame < 0:
        raise Scene3DError("frame must be >= 0")
    grid = scene.grid
    f = float(frame)

    # animated values, sampled once per track
    sampled: dict[str, tuple[float, ...]] = {
        t.target: t.sample(f) for t in scene.tracks}

    def node_transform(node: SceneNode) -> Transform:
        t = node.transform
        pos, rot, scale = t.position, t.rotation, t.scale
        base = f"nodes/{node.name}"
        if f"{base}/position" in sampled:
            pos = _vec3(sampled[f"{base}/position"], "position")
        if f"{base}/scale" in sampled:
            scale = _vec3(sampled[f"{base}/scale"], "scale")
        if f"{base}/rotation" in sampled:
            rot = Rotation.from_euler(*_vec3(
                sampled[f"{base}/rotation"], "rotation"))
        return Transform(position=pos, rotation=rot, scale=scale)

    # world matrices, parents before children (graph is acyclic —
    # the document validated it)
    world: dict[str, Mat4] = {}
    visible: dict[str, bool] = {}

    def resolve(node: SceneNode) -> Mat4:
        if node.name in world:
            return world[node.name]
        local = node_transform(node).matrix()
        if node.parent is None:
            world[node.name] = local
            visible[node.name] = node.visible
        else:
            parent = scene.node(node.parent)
            world[node.name] = _mat_mul(resolve(parent), local)
            visible[node.name] = bool(visible[node.parent]
                                      and node.visible)
        return world[node.name]

    materials_state: dict[str, dict[str, Any]] = {}
    for name in sorted(scene.materials):
        mat = scene.materials[name]
        d = mat.to_dict()
        base = f"materials/{name}"
        if f"{base}/color" in sampled:
            d["color"] = list(_vec3(sampled[f"{base}/color"],
                                    "material color"))
        if f"{base}/opacity" in sampled:
            d["opacity"] = float(sampled[f"{base}/opacity"][0])
        materials_state[name] = d

    nodes_state: dict[str, dict[str, Any]] = {}
    for node in scene.nodes:
        w = resolve(node)
        entry: dict[str, Any] = {
            "kind": node.kind,
            "parent": node.parent,
            "world_matrix": [float(v) for v in w],
            "world_position": [float(v) for v in _mat_point(w, (0, 0, 0))],
            "visible": visible[node.name],
            "mesh": node.mesh.to_dict() if node.mesh else None,
            "material": (materials_state[node.material]
                         if node.material else None),
            "light": None,
        }
        if node.light is not None:
            light = node.light.to_dict()
            base = f"nodes/{node.name}"
            if f"{base}/intensity" in sampled:
                light["intensity"] = float(
                    sampled[f"{base}/intensity"][0])
            if node.light.kind in ("directional", "spot"):
                direction = _mat_dir(w, (0.0, 0.0, -1.0))
                light["direction"] = [float(v) for v in _normalize(
                    direction, "light direction")]
            else:
                light["direction"] = None
            light["position"] = entry["world_position"]
            entry["light"] = light
        nodes_state[node.name] = entry

    cam_name = scene.camera_name()
    cam = scene.cameras[cam_name]
    cpos = cam.transform.position
    crot = cam.transform.rotation
    cfov, cnear, cfar = cam.fov, cam.near, cam.far
    base = f"cameras/{cam_name}"
    if f"{base}/position" in sampled:
        cpos = _vec3(sampled[f"{base}/position"], "camera position")
    if f"{base}/fov" in sampled:
        cfov = float(sampled[f"{base}/fov"][0])
    if f"{base}/near" in sampled:
        cnear = float(sampled[f"{base}/near"][0])
    if f"{base}/far" in sampled:
        cfar = float(sampled[f"{base}/far"][0])
    if not 0.0 < cnear < cfar:
        raise Scene3DError(
            f"camera {cam_name!r} planes invalid at frame {frame}: "
            f"near={cnear} far={cfar} (animated values must keep "
            f"0 < near < far)")
    if not 0.0 < cfov < 180.0:
        raise Scene3DError(
            f"camera {cam_name!r} fov invalid at frame {frame}: "
            f"{cfov} (animated values must keep 0 < fov < 180)")
    cam_world = Transform(position=cpos, rotation=crot,
                          scale=(1.0, 1.0, 1.0)).matrix()
    aspect = cam.aspect
    if aspect is None:
        aspect = (float(scene.output.get("width", 1920))
                  / float(scene.output.get("height", 1080)))
    camera_state = {
        "name": cam_name,
        "position": [float(v) for v in _mat_point(cam_world, (0, 0, 0))],
        "world_matrix": [float(v) for v in cam_world],
        "view_matrix": [float(v) for v in _view_matrix(cam_world)],
        "projection_matrix": [float(v) for v in _perspective(
            cfov, aspect, cnear, cfar)],
        "fov": cfov, "near": cnear, "far": cfar, "aspect": aspect,
    }
    return FrameState(
        frame=frame, seconds=float(grid.seconds(frame)),
        background=scene.background, camera=camera_state,
        nodes=nodes_state, materials=materials_state)


def scene_state_dict(scene: Scene3D, frame: int) -> dict[str, Any]:
    """Frame state as a plain dict (the golden/comparison shape)."""
    return scene_state_at(scene, frame).to_dict()


# ---------------------------------------------------------------------------
# the renderer contract + registry (§1.1e honesty rules, 3D-shaped)
# ---------------------------------------------------------------------------

#: surfaces a rung can declare it serves: ``scene3d`` (Scene3D
#: documents — the core), ``motion_camera`` (MotionProject
#: ``camera3d/*`` groups), ``format3d`` (the format registry's
#: ``renderer="camera3d"``).
RENDERER_SURFACES = ("scene3d", "motion_camera", "format3d")


@dataclass(frozen=True)
class RendererCapabilities:
    """What one rung can render — comparable data, not prose.

    ``max_width``/``max_height`` of 0 mean *undeclared*: resolution
    treats an undeclared ceiling as no ceiling but records the fact
    in the resolution record's notes (the §1.1e rule — a real number
    or a named unknown, never a fabricated one)."""

    surfaces: tuple[str, ...] = ("scene3d",)
    node_kinds: tuple[str, ...] = ()
    mesh_kinds: tuple[str, ...] = ()
    primitives: tuple[str, ...] = ()
    material_kinds: tuple[str, ...] = ()
    light_kinds: tuple[str, ...] = ()
    animated_props: tuple[str, ...] = ()
    max_width: int = 0
    max_height: int = 0
    deterministic: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        bad = [s for s in self.surfaces if s not in RENDERER_SURFACES]
        if bad:
            raise Scene3DError(
                f"renderer surfaces must be a subset of "
                f"{RENDERER_SURFACES}, got {bad}")
        for name in ("node_kinds", "mesh_kinds", "primitives",
                     "material_kinds", "light_kinds", "animated_props",
                     "surfaces"):
            object.__setattr__(self, name,
                               tuple(str(v) for v in getattr(self, name)))
        if int(self.max_width) < 0 or int(self.max_height) < 0:
            raise Scene3DError("renderer max canvas must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {"surfaces": list(self.surfaces),
                "node_kinds": list(self.node_kinds),
                "mesh_kinds": list(self.mesh_kinds),
                "primitives": list(self.primitives),
                "material_kinds": list(self.material_kinds),
                "light_kinds": list(self.light_kinds),
                "animated_props": list(self.animated_props),
                "max_width": int(self.max_width),
                "max_height": int(self.max_height),
                "deterministic": bool(self.deterministic),
                "notes": self.notes}


@dataclass(frozen=True)
class RendererProbe:
    """One rung on this host, right now: available, or the reason."""

    available: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"available": bool(self.available),
                "reason": self.reason}


@runtime_checkable
class SceneRenderer(Protocol):
    """One 3D engine rung behind the registry.

    ``render_frames`` renders ``frames`` (integer frame counts) of
    ``scene`` into ``workdir`` and returns the pipeline record:
    ``{"frames_dir", "pattern", "framerate", "frames", "width",
    "height", ...}`` — an image sequence in the shape the existing
    pipeline already consumes (``genedit``'s ``f_%04d.png`` +
    ``-framerate`` assembly). The record may carry more; those keys
    are the contract."""

    name: str

    def capabilities(self) -> RendererCapabilities:
        ...

    def probe(self) -> RendererProbe:
        ...

    def render_frames(self, scene: Scene3D, *,
                      frames: Sequence[int],
                      workdir: str | os.PathLike) -> dict[str, Any]:
        ...


_RENDERERS: dict[str, SceneRenderer] = {}


def register_renderer(renderer: SceneRenderer) -> SceneRenderer:
    """Register one engine rung. Duplicate names refuse — a rung
    replacing another is an explicit unregister first."""
    name = getattr(renderer, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise Scene3DError("a scene renderer needs a name")
    for verb in ("capabilities", "probe", "render_frames"):
        if not callable(getattr(renderer, verb, None)):
            raise Scene3DError(
                f"scene renderer {name!r} does not implement "
                f"{verb}() — the contract is capabilities/probe/"
                f"render_frames")
    renderer.capabilities()  # the declaration must be real data
    if name in _RENDERERS:
        raise Scene3DError(
            f"scene renderer {name!r} is already registered")
    _RENDERERS[name] = renderer
    return renderer


def unregister_renderer(name: str) -> bool:
    return _RENDERERS.pop(name, None) is not None


def reset_renderers() -> None:
    """Drop every registration (test support; engines register at
    their own import/wiring time)."""
    _RENDERERS.clear()


def registered_renderers() -> list[str]:
    """Registered rung names, in registration order."""
    return list(_RENDERERS)


def get_renderer(name: str) -> SceneRenderer:
    try:
        return _RENDERERS[name]
    except KeyError:
        raise Scene3DError(
            f"no scene renderer {name!r} — registered: "
            f"{', '.join(registered_renderers()) or '(none)'}"
        ) from None


def renderer_status() -> list[dict[str, Any]]:
    """Every registered rung as comparable data: capabilities +
    probe outcome — the registry's honesty surface."""
    out: list[dict[str, Any]] = []
    for name, renderer in _RENDERERS.items():
        entry: dict[str, Any] = {"name": name}
        try:
            entry["capabilities"] = renderer.capabilities().to_dict()
        except Exception as exc:  # noqa: BLE001 — status never raises
            entry["capabilities"] = None
            entry["capabilities_error"] = str(exc)
        try:
            entry["probe"] = renderer.probe().to_dict()
        except Exception as exc:  # noqa: BLE001
            entry["probe"] = {"available": False,
                               "reason": f"probe raised: {exc}"}
        out.append(entry)
    return out


def scene_requirements(scene: Scene3D) -> dict[str, Any]:
    """What a scene demands of a rung, as data — the input to
    capability-checked composition."""
    node_kinds: set[str] = set()
    mesh_kinds: set[str] = set()
    primitives: set[str] = set()
    material_kinds: set[str] = set()
    light_kinds: set[str] = set()
    for node in scene.nodes:
        node_kinds.add(node.kind)
        if node.mesh is not None:
            mesh_kinds.add(node.mesh.kind)
            if node.mesh.kind == "primitive":
                primitives.add(node.mesh.primitive)
        if node.material is not None:
            material_kinds.add(scene.materials[node.material].kind)
        if node.light is not None:
            light_kinds.add(node.light.kind)
    animated: set[str] = set()
    for track in scene.tracks:
        _scope, _name, prop = _parse_target(track.target)
        animated.add(prop)
    return {"surfaces": ["scene3d"],
            "node_kinds": sorted(node_kinds),
            "mesh_kinds": sorted(mesh_kinds),
            "primitives": sorted(primitives),
            "material_kinds": sorted(material_kinds),
            "light_kinds": sorted(light_kinds),
            "animated_props": sorted(animated),
            "width": int(scene.output.get("width", 1920)),
            "height": int(scene.output.get("height", 1080))}


def capability_gaps(caps: RendererCapabilities,
                    requirements: dict[str, Any]) -> list[str]:
    """The unmet requirements, as named strings ([] = covered)."""
    gaps: list[str] = []
    for surface in requirements.get("surfaces") or ["scene3d"]:
        if surface not in caps.surfaces:
            gaps.append(f"surface {surface!r} not served")
    for key, declared in (("node_kinds", caps.node_kinds),
                          ("mesh_kinds", caps.mesh_kinds),
                          ("primitives", caps.primitives),
                          ("material_kinds", caps.material_kinds),
                          ("light_kinds", caps.light_kinds),
                          ("animated_props", caps.animated_props)):
        missing = [v for v in requirements.get(key) or []
                   if v not in declared]
        if missing:
            gaps.append(f"{key}: needs {', '.join(missing)}")
    width = int(requirements.get("width") or 0)
    height = int(requirements.get("height") or 0)
    if caps.max_width and width > caps.max_width:
        gaps.append(f"width {width} exceeds declared max "
                    f"{caps.max_width}")
    if caps.max_height and height > caps.max_height:
        gaps.append(f"height {height} exceeds declared max "
                    f"{caps.max_height}")
    return gaps


def resolve_renderer(scene: Scene3D | None = None, *,
                     surface: str = "scene3d",
                     requirements: dict[str, Any] | None = None,
                     name: str | None = None
                     ) -> tuple[SceneRenderer, dict[str, Any]]:
    """Capability-checked composition: the serving rung + the
    resolution record.

    Tries rungs in registration order (or the one named): a rung
    serves when it probes available *and* its declared capabilities
    cover the requirements. Every refusal is recorded in the raised
    error — and, on success, in the record's ``refused`` list — so
    a resolution is never a silent pick. With no rung registered the
    error is the §1.5e no-rung refusal; the legacy seams catch the
    empty-registry case and fire their own historical wording.
    """
    if surface not in RENDERER_SURFACES:
        raise Scene3DError(
            f"unknown renderer surface {surface!r} — surfaces: "
            f"{', '.join(RENDERER_SURFACES)}")
    req: dict[str, Any] = dict(requirements or {})
    if scene is not None:
        merged = scene_requirements(scene)
        for key, value in merged.items():
            req.setdefault(key, value)
    req["surfaces"] = sorted(set(req.get("surfaces") or []) | {surface})
    candidates: list[SceneRenderer] = (
        [get_renderer(name)] if name is not None
        else list(_RENDERERS.values()))
    refused: list[dict[str, Any]] = []
    for renderer in candidates:
        entry: dict[str, Any] = {"renderer": renderer.name}
        try:
            probe = renderer.probe()
        except Exception as exc:  # noqa: BLE001 — a rung that cannot
            # probe is a rung that cannot serve; the reason is data.
            entry["reason"] = f"probe raised: {exc}"
            refused.append(entry)
            continue
        entry["probe"] = probe.to_dict()
        if not probe.available:
            entry["reason"] = probe.reason or "probe unavailable"
            refused.append(entry)
            continue
        caps = renderer.capabilities()
        gaps = capability_gaps(caps, req)
        if gaps:
            entry["reason"] = "; ".join(gaps)
            entry["gaps"] = gaps
            refused.append(entry)
            continue
        notes: list[str] = []
        if not caps.max_width or not caps.max_height:
            notes.append("rung declares no max canvas — canvas "
                         "accepted on its word")
        if req.get("surfaces") and not caps.deterministic:
            notes.append("rung does not declare deterministic "
                         "rendering — frame states stay the "
                         "contract's determinism surface")
        record = {"renderer": renderer.name, "surface": surface,
                  "requirements": req,
                  "capabilities": caps.to_dict(),
                  "probe": probe.to_dict(),
                  "refused": refused, "notes": notes}
        return renderer, record
    if not _RENDERERS:
        raise Scene3DError(
            "no 3D renderer is registered — the §1.5e rung has no "
            "engine behind the registry yet (owner decision "
            "2026-11-05: both rungs — the native Rust renderer and "
            "the Blender-class external driver — plug into this "
            "registry; neither has landed)")
    detail = "; ".join(
        f"{e['renderer']}: {e.get('reason', 'cannot serve')}"
        for e in refused)
    raise Scene3DError(
        f"no registered 3D renderer can serve surface {surface!r}: "
        f"{detail}")


def _state_hash(state: FrameState) -> str:
    blob = json.dumps(_canonical(state.to_dict()), sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def render_scene(scene: Scene3D, *,
                 frames: Iterable[int] | None = None,
                 workdir: str | os.PathLike | None = None,
                 name: str | None = None) -> dict[str, Any]:
    """Resolve a rung and render ``frames`` of ``scene`` through it.

    The returned record is the rung's own record enriched with the
    resolution record and the per-frame **state hashes** — the
    rendered frames are thereby tied to the deterministic frame
    function an auditor can recompute from the document alone.
    With no serving rung this raises :class:`Scene3DError` (the
    registry refusal) — it never falls back to a stub render.
    """
    renderer, record = resolve_renderer(scene, name=name)
    frame_list = (sorted({int(f) for f in frames})
                  if frames is not None
                  else list(range(scene.duration_frames + 1)))
    if not frame_list:
        raise Scene3DError("render_scene needs at least one frame")
    out_dir = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="scene3d-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = renderer.render_frames(scene, frames=frame_list,
                                    workdir=out_dir)
    if not isinstance(result, dict) or "frames" not in result:
        raise Scene3DError(
            f"renderer {renderer.name!r} returned a record without "
            f"'frames' — the contract is an image-sequence record "
            f"(frames_dir/pattern/framerate/frames)")
    result = dict(result)
    result["renderer"] = renderer.name
    result["resolution"] = record
    result["state_hashes"] = {
        str(f): _state_hash(scene_state_at(scene, f))
        for f in frame_list}
    result["scene_hash"] = scene.doc_hash or scene.compute_hash()
    return result
