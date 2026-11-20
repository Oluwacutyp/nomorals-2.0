"""Show plugins — the organs, as Bevy plugins.

A plugin is any object with ``build(show)``. The four built-in
plugins are the organs the show coordinates: the mocap take
player, the rig driver + renderer, the cloudcam stager, and
the face-take player. Each registers its systems into the
schedule's phases; a plugin that needs a resource another
provides refuses by name at build time, never mid-show.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..mocap.errors import MocapError
from ..rig.driver import FaceInput
from ..rig.session import take_from_store
from .errors import ShowError

__all__ = [
    "CloudCamPlugin",
    "FaceMocapPlugin",
    "MocapPlugin",
    "RigPlugin",
    "ShowPlugin",
]


class ShowPlugin:
    """A plugin registers systems and resources onto a show."""

    def build(self, show: Any) -> None:
        raise NotImplementedError


class MocapPlugin(ShowPlugin):
    """The take player, in the capture phase: one frame per tick
    into the store's ``skeleton`` channel."""

    def build(self, show: Any) -> None:
        show.system("capture", _play_take)


def _play_take(show: Any) -> None:
    take = show.take
    if take is None:
        return
    frame = take.next()
    if frame is None:
        if not show._take_finished_noted:
            show._take_finished_noted = True
            show._record("take_finished",
                         {"take": take.name, "frames": take.total})
        return
    show.store.write("skeleton", show.engine,
                     take.payload(frame), kind="snapshot")


class RigPlugin(ShowPlugin):
    """The rig, in the drive + render phases: drive the newest
    skeleton frame, hold the last pose when nothing new arrived,
    render the posed frame."""

    def build(self, show: Any) -> None:
        show.system("drive", _drive)
        show.system("render", _render)


def _drive(show: Any) -> None:
    rig = show.resource("rig")
    face, face_bones, face_stale = _read_face(show)
    try:
        take = take_from_store(show.store, channel="skeleton")
    except MocapError:
        take = None
    if (take is None or not take["frames"]
            or take["seq"] == show._last_skeleton_seq):
        # No new skeleton frame — hold the last pose. A dropped
        # mocap frame never snaps the rig to rest, and the hold is
        # named once, not spammed every tick.
        pose = rig.drive_frame(None, face,
                               stale=face_stale)
        if not show._held:
            show._record("drive_held", {
                "reason": "no new skeleton frame — holding the "
                          "last pose"})
        show._held = True
    else:
        # take_from_store already parsed the channel into a
        # skeleton + SkeletonFrames — drive the newest one.
        frame = take["frames"][-1]
        pose = rig.drive_frame(
            frame, face,
            stale=bool(take["stale"]) or face_stale)
        show._last_skeleton_seq = take["seq"]
        if show._held:
            show._record("drive_resumed", {"seq": take["seq"]})
        show._held = False
    pose = _overlay_face_bones(show, rig, pose, face_bones)
    if pose.stale and not show._was_stale:
        show._record("frame_stale", {"pose_seq": pose.seq})
    show._was_stale = pose.stale
    show._last_posed = pose
    show._last_pose = pose.to_dict()
    show._posed.append(pose)
    while len(show._posed) > show.max_posed:
        show._posed.pop(0)
        show._posed_dropped += 1


def _read_face(show: Any) -> tuple[Any, dict[str, Any], bool]:
    """The latest face frame off the store's ``face`` channel:
    (FaceInput | None, bone rotations, stale). No writes yet
    means no face — a show without a face take drives exactly
    as before. A malformed payload is refused by name (the
    tick's system-error handling records it, it never kills
    the show)."""
    try:
        view = show.store.read("face")
    except MocapError:
        return None, {}, False
    payload = view.get("payload")
    if not isinstance(payload, dict):
        raise ShowError(
            f"face channel holds {type(payload).__name__}, "
            f"not a face frame",
            kind="bad_face")
    blend = payload.get("blendshapes")
    bones = payload.get("bones")
    if blend is None and bones is None:
        return None, {}, bool(view.get("stale", False))
    face = None
    if isinstance(blend, dict):
        face = FaceInput(blendshapes={
            str(name): float(value)
            for name, value in blend.items()})
    face_bones: dict[str, tuple[float, float, float]] = {}
    if isinstance(bones, dict):
        for name, values in bones.items():
            if (not isinstance(values, (list, tuple))
                    or len(values) != 3):
                raise ShowError(
                    f"face bone {name!r} needs 3 rotation "
                    f"values, got {values!r}",
                    kind="bad_face")
            face_bones[str(name)] = tuple(
                float(v) for v in values)
    return face, face_bones, bool(view.get("stale", False))


def _overlay_face_bones(show: Any, rig: Any, pose: Any,
                        face_bones: dict[str, Any]) -> Any:
    """The face take's bones over the driven pose: the live
    head signal wins for the bones it names, the body take
    keeps the rest. Only bones the rig actually has are
    overlaid — anything else is recorded once, never
    silently carried into the render or the export. The
    overlay happens after the driver's smoothing, so a live
    head track re-applies every tick and the body value
    resumes (smoothing back) when the face take ends."""
    if not face_bones:
        return pose
    rig_bones = {bone.name for bone in rig.rig.bones}
    known = {name: values for name, values in face_bones.items()
             if name in rig_bones}
    unknown = sorted(set(face_bones) - rig_bones)
    if unknown and not show._face_bones_unknown_noted:
        show._face_bones_unknown_noted = True
        show._record("face_bone_unknown", {"bones": unknown})
    if not known:
        return pose
    return replace(pose, rotations={**pose.rotations, **known})


def _render(show: Any) -> None:
    rig = show.resource("rig")
    posed = show._last_posed
    if posed is None:
        return
    rgb = rig.render_frame(posed)
    show._last_frame = rgb
    show._rendered_bytes = len(rgb)
    if show.render_dir is None:
        return
    target = Path(show.render_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"frame-{show.tick_count:04d}.png"
    try:
        from PIL import Image
    except ImportError as exc:
        raise ShowError(
            "rendering a show frame needs Pillow — it is not "
            "installed on this machine (pip install Pillow)",
            kind="missing_pil") from exc
    Image.frombytes("RGB", (rig.width, rig.height), rgb).save(path)
    show._record("frame_written", {"path": str(path)})


class CloudCamPlugin(ShowPlugin):
    """The cloudcam, in the publish phase: stage the latest
    point-cloud frame (or the honest placeholder when nothing
    has arrived yet)."""

    def build(self, show: Any) -> None:
        show.system("publish", _stage_cloudcam)


def _stage_cloudcam(show: Any) -> None:
    try:
        cam = show.resource("cloudcam")
    except ShowError:
        show._last_cloudcam = {"attached": False}
        return
    receipt = cam.tick()
    show._last_cloudcam = {
        "attached": True,
        "placeholder": receipt["placeholder"],
        "point_count": receipt["point_count"],
        "frame_seq": receipt["frame_seq"],
        "packed": receipt["packed"],
        "dropped": receipt["dropped"],
    }


class FaceMocapPlugin(ShowPlugin):
    """The face take, in the capture phase: one face frame per
    tick into the store's ``face`` channel, so the rig plugin
    drives the blendshapes and the head in the drive phase. A
    show with no face take attached is untouched — the system
    no-ops and the show's receipt says so."""

    def build(self, show: Any) -> None:
        show.system("capture", _play_face)


def _play_face(show: Any) -> None:
    try:
        player = show.resource("face_take")
    except ShowError:
        show._last_face = {"attached": False}
        return
    frame = player.next()
    if frame is None:
        if not show._face_finished_noted:
            show._face_finished_noted = True
            show._record("face_finished",
                         {"take": player.name,
                          "frames": player.total})
        show._last_face = {"attached": True, "exhausted": True}
        return
    show.store.write("face", show.engine,
                     player.payload(frame), kind="snapshot")
    show._last_face = {
        "attached": True,
        "frame": player.index - 1,
        "total": player.total,
        "exhausted": False,
    }
