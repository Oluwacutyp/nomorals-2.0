"""3D camera — EffectCraft's 3D camera + light surface, honest rung.

EffectCraft composes in 3D: compositions have cameras and lights
and layers sit in space. This module is two things:

* :class:`Camera3D` + :func:`apply_camera` — a perspective camera
  that warps a flat plate (one image) the way a 3D camera sees it.
  Rotation/position/fov are real; the warp is a homography solved
  from the projected plate corners, so the perspective is exact,
  not a fake skew.

* :class:`PerspectiveCameraRenderer` — the scene3d rung that
  completes the motion_studio ``camera3d`` group, which refuses
  with no rung registered. Its declared capability is honest and
  narrow: 2.5D plate camera — ``scene3d`` scenes whose mesh nodes
  are ``plane`` primitives carrying a plate image path in
  ``node.extras["plate"]``, viewed through the scene camera (camera
  position + fov are animatable). No lighting, no 3D geometry —
  the capability declaration says so, and ``resolve_renderer``
  refuses anything else by name.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..edit_engine.scene3d import (
    RendererCapabilities,
    RendererProbe,
    Scene3D,
    register_renderer,
    scene_state_at,
)

__all__ = [
    "Camera3D",
    "PerspectiveCameraRenderer",
    "apply_camera",
    "register_camera_rung",
]

RUNG_NAME = "effects-perspective-camera"


@dataclass
class Camera3D:
    """A perspective camera: position, euler rotation (degrees, XYZ
    order — the scene3d convention), vertical field of view."""

    position: tuple[float, float, float] = (0.0, 0.0, 500.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov_y: float = 40.0
    near: float = 0.1
    far: float = 1000.0

    def __post_init__(self) -> None:
        if len(self.position) != 3 or len(self.rotation) != 3:
            raise ValueError("camera position/rotation need 3 components")
        self.position = tuple(float(v) for v in self.position)
        self.rotation = tuple(float(v) for v in self.rotation)
        self.fov_y = float(self.fov_y)
        if not 0.0 < self.fov_y < 180.0:
            raise ValueError(f"camera fov_y must be in (0, 180), got {self.fov_y}")
        self.near = float(self.near)
        self.far = float(self.far)
        if not 0.0 < self.near < self.far:
            raise ValueError("camera needs 0 < near < far")

    def to_dict(self) -> dict[str, Any]:
        return {"position": list(self.position),
                "rotation": list(self.rotation),
                "fov_y": self.fov_y, "near": self.near, "far": self.far}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Camera3D:
        return cls(position=tuple(data.get("position", (0, 0, 500))),
                   rotation=tuple(data.get("rotation", (0, 0, 0))),
                   fov_y=float(data.get("fov_y", 40.0)),
                   near=float(data.get("near", 0.1)),
                   far=float(data.get("far", 1000.0)))


def _euler_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    ax, ay, az = (math.radians(v) for v in (x_deg, y_deg, z_deg))
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _project_points(points: np.ndarray, cam: Camera3D,
                    width: int, height: int) -> np.ndarray | None:
    """World points (N, 3) -> pixel coords (N, 2); None if any point
    sits behind the camera."""
    rot = _euler_matrix(*cam.rotation)
    eye = np.asarray(cam.position, dtype=np.float64)
    rel = points - eye
    # camera looks down -Z: view = R^T (p - eye); as a row vector
    # that is rel @ R.
    view = rel @ rot
    if np.any(view[:, 2] >= -cam.near):
        return None
    f = 1.0 / math.tan(math.radians(cam.fov_y) / 2.0)
    aspect = width / height
    sx = (f / aspect) * view[:, 0] / (-view[:, 2])
    sy = f * view[:, 1] / (-view[:, 2])
    px = (sx + 1.0) / 2.0 * width
    py = (1.0 - (sy + 1.0) / 2.0) * height
    return np.stack([px, py], axis=1)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping src corners to dst corners."""
    a: list[list[float]] = []
    b: list[float] = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h, *_ = np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=None)
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]])


def apply_camera(img: Image.Image, cam: Camera3D,
                 background: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """View flat plate ``img`` through ``cam`` (plate centred at the
    world origin, facing the camera). Identity rotation + a camera
    on the axis returns the plate unchanged; anything else warps
    it in true perspective. A plate behind the camera comes back
    unwarped — it cannot be projected, and the function says so by
    returning the input."""
    w, h = img.size
    corners = np.array([[-w / 2, -h / 2, 0.0], [w / 2, -h / 2, 0.0],
                        [w / 2, h / 2, 0.0], [-w / 2, h / 2, 0.0]])
    projected = _project_points(corners, cam, w, h)
    if projected is None:
        return img.copy()
    src = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    if np.allclose(projected, src, atol=0.5):
        return img.copy()
    h_mat = _homography(src, projected)
    inv = np.linalg.inv(h_mat)
    coeffs = list(inv.flatten()[:8])
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         Image.BICUBIC, fillcolor=background)


# ---------------------------------------------------------------------------
# the scene3d rung
# ---------------------------------------------------------------------------

@dataclass
class PerspectiveCameraRenderer:
    """2.5D plate-camera rung for the scene3d registry.

    Renders scenes whose mesh nodes are ``plane`` primitives with a
    plate image path in ``node.extras["plate"]``: each plate is
    warped through the scene camera (position + fov animated per
    frame) and composited back-to-front by camera-space depth.
    """

    name: str = RUNG_NAME
    background: tuple[int, int, int] = (0, 0, 0)

    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(
            surfaces=("scene3d",),
            node_kinds=("mesh", "group"),
            mesh_kinds=("primitive",),
            primitives=("plane",),
            material_kinds=("flat",),
            light_kinds=(),
            animated_props=("position", "fov"),
            max_width=3840,
            max_height=2160,
            deterministic=True,
            notes=("2.5D plate camera: plane primitives with a plate "
                   "image in node extras; no lighting, no 3D geometry"))

    def probe(self) -> RendererProbe:
        return RendererProbe(available=True, reason="numpy+PIL plate camera")

    def render_frames(self, scene: Scene3D, *,
                      frames: Sequence[int],
                      workdir: str | Path) -> dict[str, Any]:
        from ..edit_engine.scene3d import Scene3DError

        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        width = int(scene.output.get("width", 1920))
        height = int(scene.output.get("height", 1080))
        fps = int(scene.output.get("fps", 30))
        rendered: list[int] = []
        for frame in frames:
            state = scene_state_at(scene, int(frame))
            canvas = Image.new("RGB", (width, height), self.background)
            cam_state = state.camera
            scene_cam = scene.camera()
            cam_rot = (0.0, 0.0, 0.0)
            if scene_cam.transform.rotation.kind == "euler":
                cam_rot = tuple(scene_cam.transform.rotation.euler)
            cam = Camera3D(
                position=tuple(cam_state["position"]),
                rotation=cam_rot,
                fov_y=float(cam_state["fov"]),
                near=float(cam_state["near"]),
                far=float(cam_state["far"]))
            plates: list[tuple[float, Image.Image]] = []
            for node in scene.nodes:
                entry = state.nodes[node.name]
                if not entry["visible"] or entry["mesh"] is None:
                    continue
                plate_path = (node.extras or {}).get("plate")
                if not plate_path:
                    raise Scene3DError(
                        f"node {node.name!r} has no plate image — "
                        f"the plate-camera rung renders plane "
                        f"primitives with extras['plate'] set")
                plate = Image.open(str(plate_path)).convert("RGB")
                world = np.asarray(entry["world_matrix"]).reshape(4, 4)
                # plane corners in local space: unit plane scaled by
                # the node's world scale, centred on its world origin
                sx = math.sqrt(sum(world[0, i] ** 2 for i in range(3)))
                sy = math.sqrt(sum(world[1, i] ** 2 for i in range(3)))
                origin = world[:3, 3]
                half_w = plate.size[0] * sx / 2.0
                half_h = plate.size[1] * sy / 2.0
                corners = np.array([
                    origin + np.array([-half_w, -half_h, 0.0]),
                    origin + np.array([half_w, -half_h, 0.0]),
                    origin + np.array([half_w, half_h, 0.0]),
                    origin + np.array([-half_w, half_h, 0.0]),
                ])
                projected = _project_points(corners, cam, width, height)
                if projected is None:
                    continue
                src = np.array([[0.0, 0.0], [plate.size[0], 0.0],
                                [plate.size[0], plate.size[1]],
                                [0.0, plate.size[1]]])
                h_mat = _homography(src, projected)
                inv_coeffs = list(np.linalg.inv(h_mat).flatten()[:8])
                warped = plate.transform(
                    (width, height), Image.PERSPECTIVE, inv_coeffs,
                    Image.BICUBIC, fillcolor=self.background)
                # the warped quad's mask: the plate is opaque only
                # inside its projected corners
                mask_src = Image.new("L", plate.size, 255)
                mask = mask_src.transform(
                    (width, height), Image.PERSPECTIVE, inv_coeffs,
                    Image.BICUBIC, fillcolor=0)
                depth = float(-((origin - np.asarray(cam.position))
                                @ _euler_matrix(*cam.rotation))[2])
                plates.append((depth, warped, mask))
            plates.sort(key=lambda item: item[0], reverse=True)
            for _depth, warped, mask in plates:
                canvas.paste(warped, (0, 0), mask)
            path = work / f"f_{int(frame):04d}.png"
            canvas.save(path)
            rendered.append(int(frame))
        return {"frames_dir": str(work), "pattern": "f_%04d.png",
                "framerate": fps, "frames": rendered,
                "width": width, "height": height,
                "renderer": RUNG_NAME}


def register_camera_rung() -> PerspectiveCameraRenderer:
    """Register the plate-camera rung (idempotent for tests)."""
    from ..edit_engine.scene3d import registered_renderers, unregister_renderer

    if RUNG_NAME in registered_renderers():
        unregister_renderer(RUNG_NAME)
    rung = PerspectiveCameraRenderer()
    register_renderer(rung)
    return rung
