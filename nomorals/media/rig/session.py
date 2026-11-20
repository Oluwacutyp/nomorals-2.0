"""Rig session — drive a take, render it, publish it, export it.

One session holds a rig, its remap, and a driver. It drives whole
takes (a BVH take, or the ``skeleton`` channel of the mocap store),
renders the posed frames, and hands them to the sinks the repo
already has: a ``FrameSource`` closure (elapsed seconds in, one
RGB24 frame out) for the LiveKit video session, PNG files on disk
for an offline take, and BVH out through the mocap exporter so
game engines can read the driven rig back.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..mocap.bvh import (
    Joint,
    Skeleton,
    SkeletonFrame,
    hierarchy_joints,
    parse_bvh,
    skeleton_from_dict,
    write_bvh,
)
from ..mocap.errors import MocapError
from ..mocap.state import MocapStateStore
from .driver import FaceInput, PosedFrame, RigDriver
from .errors import RigError
from .remap import BoneRemap
from .render import render_pose
from .rig import RigDefinition

__all__ = [
    "RigSession",
    "take_from_store",
]

#: A LiveKit frame source: elapsed seconds -> RGB24 frame or None.
FrameSource = Callable[[float], tuple[bytes, int, int] | None]


def take_from_store(store: MocapStateStore, *,
                    channel: str = "skeleton") -> dict[str, Any]:
    """Read a take off a mocap store channel: skeleton + frames.
    A channel with no take is refused by name."""
    view = store.read(channel)
    payload = view["payload"]
    if not isinstance(payload, dict) or "skeleton" not in payload:
        raise MocapError(
            f"channel {channel!r} holds no skeleton take",
            kind="no_take")
    skeleton = skeleton_from_dict(payload["skeleton"])
    frames = [
        SkeletonFrame(values={
            name: tuple(float(v) for v in values)
            for name, values in raw.items()})
        for raw in payload.get("frames", [])]
    return {"skeleton": skeleton, "frames": frames,
            "stale": bool(view.get("stale", False)),
            "seq": view.get("seq", 0)}


def _rig_skeleton(rig: RigDefinition) -> Skeleton:
    """The rig as a mocap skeleton (for BVH export)."""

    def joint_for(bone_name: str) -> Joint:
        bone = rig.bone(bone_name)
        children = [joint_for(child.name)
                    for child in rig.children(bone_name)]
        if bone.parent is None:
            channels = ("Xposition", "Yposition", "Zposition",
                        "Zrotation", "Xrotation", "Yrotation")
        else:
            channels = ("Zrotation", "Xrotation", "Yrotation")
        return Joint(name=bone.name, offset=tuple(bone.offset),
                     channels=channels, children=children)

    root = next(bone for bone in rig.bones if bone.parent is None)
    return Skeleton(root=joint_for(root.name))


class RigSession:
    """One driven rig: load it, drive takes through it, render
    the posed frames, publish them."""

    def __init__(self, rig: RigDefinition, remap: BoneRemap, *,
                 width: int = 640, height: int = 360,
                 fps: float = 15.0, smooth: float = 0.35) -> None:
        if width < 64 or height < 64:
            raise RigError(
                f"rig frame size must be at least 64x64, got "
                f"{width}x{height}", kind="bad_size")
        if fps <= 0:
            raise RigError(f"fps must be positive, got {fps}",
                           kind="bad_size")
        self.rig = rig
        self.remap = remap
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.driver = RigDriver(rig, remap, smooth=smooth, fps=fps)

    # -- driving ---------------------------------------------------

    def drive_frame(self, frame: SkeletonFrame | None,
                    face: FaceInput | None = None, *,
                    stale: bool = False) -> PosedFrame:
        """Drive one frame through the rig."""
        return self.driver.drive(frame, face, stale=stale)

    def drive_take(self, skeleton: Skeleton,
                   frames: list[SkeletonFrame], *,
                   text: str | None = None,
                   face: FaceInput | None = None) -> list[PosedFrame]:
        """Drive a whole take. With ``text``, the visemes follow
        the text on uniform timing (the lip-sync organ's honest
        rule); with ``face``, every frame takes that face input.
        Text and a face input together are refused — one face
        driver per take."""
        if text and face is not None:
            raise RigError(
                "drive_take takes text or face, not both",
                kind="bad_face")
        if not frames:
            raise RigError("drive_take needs at least one frame",
                           kind="bad_take")
        phonemes: list[str] = []
        visemes: list[str] = []
        if text:
            from ..directed.lipsync import (
                phonemes_to_visemes,
                text_to_phonemes,
            )
            phonemes = text_to_phonemes(text)
            visemes = phonemes_to_visemes(phonemes)
        self.driver.reset()
        posed: list[PosedFrame] = []
        for index, frame in enumerate(frames):
            frame_face = face
            if visemes:
                at = min(len(visemes) - 1,
                         int(index * len(visemes) / len(frames)))
                frame_face = FaceInput(viseme=visemes[at])
            posed.append(self.driver.drive(frame, frame_face))
        return posed

    def drive_from_store(self, store: MocapStateStore, *,
                         channel: str = "skeleton",
                         require_fresh: bool = False,
                         face: FaceInput | None = None) -> PosedFrame:
        """Drive the latest frame off a store channel. A stale
        channel is flagged on the pose; with ``require_fresh`` it
        is refused by name instead."""
        view = store.read(channel, require_fresh=require_fresh)
        payload = view["payload"]
        if not isinstance(payload, dict) or "skeleton" not in payload:
            raise MocapError(
                f"channel {channel!r} holds no skeleton take",
                kind="no_take")
        raw_frames = payload.get("frames", [])
        if not raw_frames:
            raise MocapError(
                f"channel {channel!r} holds no frames",
                kind="no_take")
        last = raw_frames[-1]
        frame = SkeletonFrame(values={
            name: tuple(float(v) for v in values)
            for name, values in last.items()})
        return self.driver.drive(frame, face,
                                 stale=bool(view.get("stale", False)))

    # -- rendering -------------------------------------------------

    def render_frame(self, pose: PosedFrame) -> bytes:
        """One posed frame as RGB24 bytes."""
        return render_pose(self.rig, pose, width=self.width,
                           height=self.height)

    def render_take(self, posed: list[PosedFrame], *,
                    out_dir: str | Path | None = None) -> list[Any]:
        """Render a posed take. Without ``out_dir`` the RGB frames
        come back in memory; with one, PNG files land there (one
        per frame, ``frame-0000.png`` …) and the paths come back."""
        if not posed:
            raise RigError("render_take needs at least one pose",
                           kind="bad_take")
        frames = [self.render_frame(pose) for pose in posed]
        if out_dir is None:
            return frames
        target = Path(out_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
        except ImportError as exc:
            raise RigError(
                "rig rendering needs Pillow — it is not installed "
                "on this machine (pip install Pillow)",
                kind="missing_pil") from exc
        paths: list[str] = []
        for index, rgb in enumerate(frames):
            path = target / f"frame-{index:04d}.png"
            Image.frombytes("RGB", (self.width, self.height),
                            rgb).save(path)
            paths.append(str(path))
        return paths

    def frame_source(self, skeleton: Skeleton,
                     frames: list[SkeletonFrame], *,
                     text: str | None = None,
                     face: FaceInput | None = None,
                     loop: bool = False) -> FrameSource:
        """A LiveKit frame source over a driven take: elapsed
        seconds in, one RGB24 frame out, None when the take ends
        (or looping, when ``loop``). The take is driven once, up
        front — the source itself only renders."""
        posed = self.drive_take(skeleton, frames, text=text,
                                face=face)
        rendered = [self.render_frame(pose) for pose in posed]
        frame_time = skeleton.frame_time or (1.0 / self.fps)

        def _source(t: float) -> tuple[bytes, int, int] | None:
            index = int(t / frame_time)
            if index >= len(rendered):
                if not loop or not rendered:
                    return None
                index %= len(rendered)
            return rendered[index], self.width, self.height

        return _source

    # -- export ----------------------------------------------------

    def export_bvh(self, posed: list[PosedFrame],
                   path: str | Path, *,
                   store: MocapStateStore | None = None) -> dict[str, Any]:
        """Write posed frames out as BVH (the rig skeleton +
        driven rotations) — the format game engines read. The
        absolute path and frame count come back, and the export is
        recorded on the store's ``bvh`` channel when one is given."""
        if not posed:
            raise RigError("export_bvh needs at least one pose",
                           kind="bad_take")
        skeleton = _rig_skeleton(self.rig)
        frames: list[SkeletonFrame] = []
        for pose in posed:
            values: dict[str, tuple[float, ...]] = {}
            for joint in hierarchy_joints(skeleton):
                rot = pose.rotations.get(joint.name, (0.0, 0.0, 0.0))
                if joint is skeleton.root:
                    values[joint.name] = (0.0, 0.0, 0.0, *rot)
                else:
                    values[joint.name] = tuple(rot)
            frames.append(SkeletonFrame(values=values))
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(write_bvh(skeleton, frames),
                          encoding="utf-8")
        result: dict[str, Any] = {
            "path": str(target),
            "frames": len(frames),
            "joints": [joint.name for joint in
                       hierarchy_joints(skeleton)],
            "rig": self.rig.name,
        }
        if store is not None:
            # The ``bvh`` channel has ONE writer — the BVH exporter
            # (engine "bvh"). The rig records through that writer,
            # never as a second engine: a second writer would claim
            # the channel and refuse every later BVH export in the
            # process (and vice versa). The rig name rides in the
            # payload, so the provenance survives.
            from ..mocap.kinect_bridge import _ensure_engine
            _ensure_engine(store, "bvh", produces=("bvh",),
                           note="BVH export — the rig records its "
                                "export through the BVH writer")
            store.write("bvh", "bvh",
                        {"path": str(target), "frames": len(frames),
                         "rig": self.rig.name}, kind="snapshot")
        return result

    def to_dict(self) -> dict[str, Any]:
        """The session's rig + remap + frame size, as data."""
        return {
            "rig": self.rig.name,
            "bones": len(self.rig.bones),
            "blendshapes": len(self.rig.blendshapes),
            "visemes": len(self.rig.visemes),
            "remap_dialect": self.remap.dialect,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


def load_take_bvh(path: str | Path) -> dict[str, Any]:
    """Parse a BVH file into a skeleton + frames for driving. A
    missing or malformed file is refused by name."""
    target = Path(path).expanduser()
    if not target.exists():
        raise RigError(f"no BVH file at {target}",
                       kind="missing_bvh")
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise RigError(
            f"BVH file at {target} is unreadable: {exc}",
            kind="bad_bvh") from exc
    try:
        skeleton, frames = parse_bvh(text)
    except MocapError as exc:
        raise RigError(f"BVH file at {target} is malformed: {exc}",
                       kind="bad_bvh") from exc
    return {"skeleton": skeleton, "frames": frames}


def load_take_json(path: str | Path) -> dict[str, Any]:
    """Load a take from a JSON file shaped like the ``skeleton``
    channel payload."""
    target = Path(path).expanduser()
    if not target.exists():
        raise RigError(f"no JSON take at {target}",
                       kind="missing_take")
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RigError(
            f"JSON take at {target} is unreadable: {exc}",
            kind="bad_take") from exc
    if not isinstance(blob, dict) or "skeleton" not in blob:
        raise RigError(
            f"JSON take at {target} has no 'skeleton' key",
            kind="bad_take")
    skeleton = skeleton_from_dict(blob["skeleton"])
    frames = [
        SkeletonFrame(values={
            name: tuple(float(v) for v in values)
            for name, values in raw.items()})
        for raw in blob.get("frames", [])]
    return {"skeleton": skeleton, "frames": frames}
