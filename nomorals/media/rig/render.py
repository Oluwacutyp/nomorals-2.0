"""Rig renderer — a posed rig as RGB video frames.

The HyperJail rig renders in Unreal; Devon has no 3D renderer on
this box, so the rig renders as a 2D puppet: forward kinematics
projects the bone tree onto the X/Y plane (Z rotations turn the
limbs in-plane), and the face is drawn from the blendshapes and
the viseme mouth shape — eyes blink, brows lift, the mouth opens,
widens, and rounds with the viseme. It is honestly a 2D puppet,
not a 3D avatar — and it is exactly the RGB24 shape a LiveKit
video frame source takes, so the same frames that render to a PNG
here publish into a room there.
"""

from __future__ import annotations

import math
from typing import Any

from .driver import PosedFrame
from .errors import RigError
from .rig import RigDefinition

__all__ = ["render_pose"]

#: Frame background, limb, joint, head, and face colours.
_BG = (10, 10, 14)
_SPINE = (235, 235, 240)
_LIMB_L = (120, 180, 255)
_LIMB_R = (120, 255, 170)
_JOINT = (255, 200, 80)
_HEAD = (240, 220, 200)
_FACE = (30, 30, 40)
_TEXT = (160, 160, 170)


def _require_pil() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RigError(
            "rig rendering needs Pillow — it is not installed on "
            "this machine (pip install Pillow)",
            kind="missing_pil") from exc
    return Image, ImageDraw


def _fk_positions(rig: RigDefinition, pose: PosedFrame,
                  ) -> dict[str, tuple[float, float]]:
    """World X/Y positions per bone (2D projection: Z rotation
    turns the offset in-plane; X/Y rotations tilt out of plane and
    are not drawn — this is a 2D puppet)."""
    positions: dict[str, tuple[float, float]] = {}
    angles: dict[str, float] = {}

    def place(bone_name: str) -> None:
        if bone_name in positions:
            return
        bone = rig.bone(bone_name)
        z_rot = pose.rotations.get(bone.name, bone.rest)[0]
        if bone.parent is None:
            positions[bone.name] = (bone.offset[0], bone.offset[1])
            angles[bone.name] = math.radians(z_rot)
            return
        place(bone.parent)
        parent_pos = positions[bone.parent]
        parent_angle = angles[bone.parent]
        angle = parent_angle + math.radians(z_rot)
        ox, oy = bone.offset[0], bone.offset[1]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        positions[bone.name] = (
            parent_pos[0] + ox * cos_a - oy * sin_a,
            parent_pos[1] + ox * sin_a + oy * cos_a,
        )
        angles[bone.name] = angle

    for bone in rig.hierarchy_order():
        place(bone.name)
    return positions


def render_pose(rig: RigDefinition, pose: PosedFrame, *,
                width: int = 640, height: int = 360) -> bytes:
    """Render one posed frame as RGB24 bytes (``width * height *
    3``). A bad size is refused, naming the size."""
    if width < 64 or height < 64:
        raise RigError(
            f"rig frame size must be at least 64x64, got "
            f"{width}x{height}", kind="bad_size")
    Image, ImageDraw = _require_pil()
    positions = _fk_positions(rig, pose)
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    span_x = max(xs) - min(xs) or 1.0
    span_y = max(ys) - min(ys) or 1.0
    margin = 24
    scale = min((width - 2 * margin) / span_x,
                (height - 2 * margin) / span_y)
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2

    def screen(x: float, y: float) -> tuple[float, float]:
        return (width / 2 + (x - cx) * scale,
                height / 2 - (y - cy) * scale)

    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)
    head_r = max(10.0, 0.09 * scale)
    root_name = next(b.name for b in rig.bones if b.parent is None)

    def bone_colour(name: str) -> tuple[int, int, int]:
        if name.endswith("_L"):
            return _LIMB_L
        if name.endswith("_R"):
            return _LIMB_R
        return _SPINE

    for bone in rig.bones:
        if bone.parent is None:
            continue
        p0 = screen(*positions[bone.parent])
        p1 = screen(*positions[bone.name])
        draw.line([p0, p1], fill=bone_colour(bone.name), width=5)
    for bone in rig.bones:
        p = screen(*positions[bone.name])
        r = 4 if bone.name != root_name else 6
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                     fill=_JOINT)
    # -- head + face -------------------------------------------------
    if "head" in positions:
        hx, hy = screen(*positions["head"])
        draw.ellipse([hx - head_r, hy - head_r, hx + head_r,
                      hy + head_r], outline=_HEAD, width=3)
        blend = pose.blendshapes
        blink_l = blend.get("eyeBlinkLeft", 0.0)
        blink_r = blend.get("eyeBlinkRight", 0.0)
        brow = blend.get("browInnerUp", 0.0)
        eye_dx = head_r * 0.38
        eye_dy = -head_r * 0.15
        eye_r = max(1.0, head_r * 0.10)
        for side, blink in ((-1, blink_l), (1, blink_r)):
            ex = hx + side * eye_dx
            ey = hy + eye_dy
            open_h = max(0.5, eye_r * 2 * (1.0 - blink))
            draw.ellipse([ex - eye_r, ey - open_h / 2,
                          ex + eye_r, ey + open_h / 2], fill=_FACE)
            brow_y = ey - head_r * 0.28 - brow * head_r * 0.25
            draw.line([(ex - eye_r, brow_y), (ex + eye_r, brow_y)],
                      fill=_FACE, width=2)
        jaw_open, lip_width, lip_round = pose.mouth
        smile = (blend.get("mouthSmileLeft", 0.0)
                 + blend.get("mouthSmileRight", 0.0)) / 2
        smile += pose.laughter * 0.6
        mouth_w = max(2.0, head_r * 0.55 * lip_width)
        mouth_h = max(1.5, head_r * 0.55 * jaw_open)
        mx, my = hx, hy + head_r * 0.45
        if mouth_h <= 2.0:
            draw.line([(mx - mouth_w, my), (mx + mouth_w, my)],
                      fill=_FACE, width=2)
        else:
            draw.ellipse([mx - mouth_w, my - mouth_h,
                          mx + mouth_w, my + mouth_h],
                         outline=_FACE, width=2)
        if smile > 0.05:
            lift = smile * head_r * 0.15
            draw.line([(mx - mouth_w, my - lift),
                       (mx - mouth_w * 0.5, my)],
                      fill=_FACE, width=2)
            draw.line([(mx + mouth_w, my - lift),
                       (mx + mouth_w * 0.5, my)],
                      fill=_FACE, width=2)
    draw.text((8, height - 18),
              f"rig:{rig.name} f:{pose.seq} {pose.viseme}",
              fill=_TEXT)
    return img.tobytes()
