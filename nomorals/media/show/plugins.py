"""Show plugins — the organs, as Bevy plugins.

A plugin is any object with ``build(show)``. The three built-in
plugins are the organs the show coordinates: the mocap take
player, the rig driver + renderer, and the cloudcam stager. Each
registers its systems into the schedule's phases; a plugin that
needs a resource another provides refuses by name at build time,
never mid-show.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..mocap.errors import MocapError
from ..rig.session import take_from_store
from .errors import ShowError

__all__ = [
    "CloudCamPlugin",
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
    try:
        take = take_from_store(show.store, channel="skeleton")
    except MocapError:
        take = None
    if (take is None or not take["frames"]
            or take["seq"] == show._last_skeleton_seq):
        # No new skeleton frame — hold the last pose. A dropped
        # mocap frame never snaps the rig to rest, and the hold is
        # named once, not spammed every tick.
        pose = rig.drive_frame(None)
        if not show._held:
            show._record("drive_held", {
                "reason": "no new skeleton frame — holding the "
                          "last pose"})
        show._held = True
    else:
        # take_from_store already parsed the channel into a
        # skeleton + SkeletonFrames — drive the newest one.
        frame = take["frames"][-1]
        pose = rig.drive_frame(frame, stale=bool(take["stale"]))
        show._last_skeleton_seq = take["seq"]
        if show._held:
            show._record("drive_resumed", {"seq": take["seq"]})
        show._held = False
    if pose.stale and not show._was_stale:
        show._record("frame_stale", {"pose_seq": pose.seq})
    show._was_stale = pose.stale
    show._last_posed = pose
    show._last_pose = pose.to_dict()
    show._posed.append(pose)
    while len(show._posed) > show.max_posed:
        show._posed.pop(0)
        show._posed_dropped += 1


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
