"""The face take — a mapped LiveLink recording, ready to play.

A :class:`FaceTake` is the clone's ``AnimationClip`` in Devon's
shape: one frame per CSV row, each frame carrying the rig
blendshape weights and the bone rotations the mapping produced.
A :class:`FaceTakePlayer` steps it one frame per tick — the
same shape as the show's :class:`TakePlayer`, over face frames
instead of skeleton frames — and its payload is the store's
``face`` channel shape, so the show's face plugin can write it
straight into the mocap store.

The take also converts both ways across the organs that
already exist: its frames become :class:`FaceInput`s (so a
rig session drives it frame by frame), and its bones become
:class:`SkeletonFrame`s (the skeleton-frame shape — the
bones are rig-named, so a take played through a
source-dialect remap needs the face overlay, not the raw
frame: :func:`drive_take` and the show's face plugin apply
it; the driven poses export through the rig's BVH export).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..mocap.bvh import SkeletonFrame
from ..rig.driver import FaceInput
from .errors import FaceMocapError

__all__ = [
    "FaceFrame",
    "FaceTake",
    "FaceTakePlayer",
    "drive_take",
    "render_frame",
]


@dataclass(frozen=True)
class FaceFrame:
    """One mapped frame: seconds from the take's start, rig
    blendshape weights, and bone rotations in ``(z, x, y)``
    degrees."""

    time_s: float
    blendshapes: dict[str, float]
    bones: dict[str, tuple[float, float, float]]

    def to_dict(self) -> dict[str, Any]:
        """The frame as plain JSON-able data."""
        return {
            "time_s": self.time_s,
            "blendshapes": dict(self.blendshapes),
            "bones": {name: list(values)
                      for name, values in self.bones.items()},
        }


@dataclass
class FaceTake:
    """A mapped face recording: frames, fps, the morphs it
    carries, and the CSV morphs no mapping entry named."""

    name: str
    root_name: str
    fps: float
    morph_names: list[str]
    frames: list[FaceFrame]
    unmapped: tuple[str, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        if not self.frames:
            raise FaceMocapError(
                "a face take needs at least one frame",
                kind="bad_take")
        if self.fps <= 0:
            raise FaceMocapError(
                f"face take fps must be positive, got {self.fps}",
                kind="bad_take")

    @property
    def total(self) -> int:
        """Frames in the take."""
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        """Seconds from the first frame to the last."""
        return (len(self.frames) - 1) / self.fps if self.fps else 0.0

    def player(self) -> FaceTakePlayer:
        """A player over this take."""
        return FaceTakePlayer(self)

    def to_face_inputs(self) -> list[FaceInput]:
        """Every frame as rig face input."""
        return [FaceInput(blendshapes=dict(frame.blendshapes))
                for frame in self.frames]

    def to_skeleton_frames(self) -> list[SkeletonFrame]:
        """Every frame's bones as a skeleton frame — the shape
        the show's take player and the rig's BVH export read.
        A frame names only the bones its mapping drove; the
        driver holds the rest at rest."""
        return [SkeletonFrame(values={
            name: tuple(values)
            for name, values in frame.bones.items()})
            for frame in self.frames]

    def to_dict(self) -> dict[str, Any]:
        """The take's summary, as data."""
        return {
            "name": self.name,
            "root": self.root_name,
            "fps": self.fps,
            "frames": len(self.frames),
            "duration_s": round(self.duration_s, 3),
            "morphs": list(self.morph_names),
            "unmapped": list(self.unmapped),
            "source": self.source,
        }

    def to_json(self) -> dict[str, Any]:
        """The whole take, as JSON-able data — the shape the
        loader reads back, so a take persists across restarts."""
        return {
            "format": "face-take-v1",
            "name": self.name,
            "root": self.root_name,
            "fps": self.fps,
            "morphs": list(self.morph_names),
            "unmapped": list(self.unmapped),
            "source": self.source,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_json(cls, blob: Any, *, where: str = "take"
                  ) -> FaceTake:
        """Rebuild a take from :meth:`to_json` output. A blob
        with the wrong format, no frames, or a malformed frame
        is refused by name."""
        if not isinstance(blob, dict):
            raise FaceMocapError(
                f"{where} must be a JSON object",
                kind="bad_take")
        if blob.get("format") != "face-take-v1":
            raise FaceMocapError(
                f"{where} is not a face take "
                f"(format={blob.get('format')!r})",
                kind="bad_take")
        fps = blob.get("fps")
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise FaceMocapError(
                f"{where} needs a positive 'fps'",
                kind="bad_take")
        raw_frames = blob.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise FaceMocapError(
                f"{where} needs at least one frame",
                kind="bad_take")
        frames: list[FaceFrame] = []
        for index, raw in enumerate(raw_frames):
            if not isinstance(raw, dict):
                raise FaceMocapError(
                    f"{where} frame {index} must be an object",
                    kind="bad_take")
            blend = raw.get("blendshapes")
            bones = raw.get("bones")
            if not isinstance(blend, dict) or not isinstance(
                    bones, dict):
                raise FaceMocapError(
                    f"{where} frame {index} needs 'blendshapes' "
                    f"and 'bones' objects",
                    kind="bad_take")
            try:
                frames.append(FaceFrame(
                    time_s=float(raw.get("time_s", index / fps)),
                    blendshapes={str(k): float(v)
                                 for k, v in blend.items()},
                    bones={str(k): tuple(float(v) for v in vals)
                           for k, vals in bones.items()}))
            except (TypeError, ValueError) as exc:
                raise FaceMocapError(
                    f"{where} frame {index} has a non-numeric "
                    f"value: {exc}",
                    kind="bad_take") from exc
            for name, values in frames[-1].bones.items():
                if len(values) != 3:
                    raise FaceMocapError(
                        f"{where} frame {index} bone {name!r} "
                        f"needs 3 rotation values",
                        kind="bad_take")
        morphs = blob.get("morphs")
        if not isinstance(morphs, list):
            raise FaceMocapError(
                f"{where} needs a 'morphs' list",
                kind="bad_take")
        return cls(
            name=str(blob.get("name") or "from-json"),
            root_name=str(blob.get("root") or ""),
            fps=float(fps),
            morph_names=[str(m) for m in morphs],
            frames=frames,
            unmapped=tuple(
                str(u) for u in (blob.get("unmapped") or [])),
            source=str(blob.get("source") or ""))


class FaceTakePlayer:
    """One face take, played frame by frame."""

    def __init__(self, take: FaceTake) -> None:
        self.take = take
        self.name = take.name
        self._index = 0

    @property
    def index(self) -> int:
        """Frames played so far."""
        return self._index

    @property
    def total(self) -> int:
        """Frames in the take."""
        return self.take.total

    @property
    def exhausted(self) -> bool:
        """True once every frame has played."""
        return self._index >= len(self.take.frames)

    def next(self) -> FaceFrame | None:
        """The next frame, or None when the take is spent."""
        if self.exhausted:
            return None
        frame = self.take.frames[self._index]
        self._index += 1
        return frame

    def reset(self) -> None:
        """Play from the first frame again."""
        self._index = 0

    def payload(self, frame: FaceFrame) -> dict[str, Any]:
        """One frame as a ``face``-channel payload — the shape
        the rig plugin reads back: blendshapes, bone rotations,
        the frame number and its time."""
        return {
            "blendshapes": dict(frame.blendshapes),
            "bones": {name: list(values)
                      for name, values in frame.bones.items()},
            "frame": self._index - 1,
            "time_s": frame.time_s,
        }

    def to_dict(self) -> dict[str, Any]:
        """The player's state, as data."""
        return {
            "name": self.name,
            "index": self._index,
            "total": self.total,
            "exhausted": self.exhausted,
        }


def drive_take(session: Any, take: FaceTake) -> list[Any]:
    """Drive a whole face take through a rig session, one
    frame at a time, and return the posed frames.

    The take's bones are already rig-named — the mapping
    names the rig bone the rotation lands on — so they must
    NOT go through the driver's source-dialect remap (it
    drops rig names as unmapped: the BVH table maps
    ``Head`` → ``head`` and knows no ``head``). Each frame's
    blendshapes drive through the session (the body holds at
    rest — a face take has no body), then the frame's bones
    overlay the posed rotations, the same rule the show's
    face plugin uses: bones the rig has win, anything else
    is named on the pose's ``unmapped`` tuple, never silently
    carried or dropped. The driver's smoothing state resets
    first, so a second drive of the same take plays the same
    poses."""
    if take.total == 0:
        raise FaceMocapError("drive_take needs at least one frame",
                             kind="bad_take")
    session.driver.reset()
    rig_bones = {bone.name for bone in session.rig.bones}
    faces = take.to_face_inputs()
    posed: list[Any] = []
    for frame, face in zip(take.frames, faces, strict=True):
        pose = session.drive_frame(None, face)
        known = {name: values
                 for name, values in frame.bones.items()
                 if name in rig_bones}
        unknown = tuple(sorted(set(frame.bones) - rig_bones))
        if not known and not unknown:
            posed.append(pose)
            continue
        posed.append(replace(
            pose,
            rotations={**pose.rotations, **known},
            unmapped=tuple(pose.unmapped) + unknown))
    return posed


def render_frame(session: Any, take: FaceTake, index: int,
                 output: str | Path) -> dict[str, Any]:
    """Render one frame of a take as a PNG: the take is driven
    to that frame (smoothing state matters), then the pose
    renders to exactly ``output``. An out-of-range frame is
    refused by name."""
    if index < 0 or index >= take.total:
        raise FaceMocapError(
            f"take {take.name!r} has frames 0..{take.total - 1}, "
            f"got {index}",
            kind="bad_frame")
    posed = drive_take(session, take)
    rgb = session.render_frame(posed[index])
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError as exc:
        raise FaceMocapError(
            "rendering a face frame needs Pillow — it is not "
            "installed on this machine (pip install Pillow)",
            kind="missing_pil") from exc
    Image.frombytes("RGB", (session.width, session.height),
                    rgb).save(target)
    return {"take": take.name, "frame": index,
            "path": str(target), "rig": session.rig.name,
            "width": session.width, "height": session.height}
