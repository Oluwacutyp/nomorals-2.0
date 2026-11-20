"""The take player — a BVH take, stepped one frame per tick.

The studio-bvh pattern, in Python: the clone's Bevy crate plays
BVH motion capture through Bevy's animation system; Devon's
player steps a take's frames into the mocap store's ``skeleton``
channel, one per show tick, so the rig drives the live frame
instead of the whole take at once. A take that runs out ends
with a named event and the rig holds its last pose — a finished
take never snaps the avatar to rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..mocap.bvh import Skeleton, SkeletonFrame, skeleton_to_dict
from ..rig.session import load_take_bvh, load_take_json
from .errors import ShowError

__all__ = ["TakePlayer"]


class TakePlayer:
    """One BVH take, played frame by frame."""

    def __init__(self, skeleton: Skeleton,
                 frames: list[SkeletonFrame], *,
                 name: str = "in-memory") -> None:
        if not frames:
            raise ShowError("a take player needs at least one frame",
                            kind="bad_take")
        self.skeleton = skeleton
        self.frames = list(frames)
        self.name = name
        self._index = 0
        self._skeleton_dict = skeleton_to_dict(skeleton)

    # -- loading ---------------------------------------------------

    @classmethod
    def from_bvh(cls, path: str | Path) -> TakePlayer:
        """Parse a BVH file into a player. A missing or malformed
        file is refused by name."""
        target = Path(path).expanduser()
        if not target.exists():
            raise ShowError(f"no BVH take at {target}",
                            kind="missing_take")
        try:
            take = load_take_bvh(target)
        except Exception as exc:
            raise ShowError(
                f"BVH take at {target} is malformed: {exc}",
                kind="bad_take") from exc
        return cls(take["skeleton"], take["frames"], name=str(target))

    @classmethod
    def from_json(cls, path: str | Path) -> TakePlayer:
        """Load a take from a JSON file shaped like the
        ``skeleton`` channel payload."""
        target = Path(path).expanduser()
        if not target.exists():
            raise ShowError(f"no JSON take at {target}",
                            kind="missing_take")
        try:
            take = load_take_json(target)
        except Exception as exc:
            raise ShowError(
                f"JSON take at {target} is malformed: {exc}",
                kind="bad_take") from exc
        return cls(take["skeleton"], take["frames"], name=str(target))

    # -- playing ---------------------------------------------------

    @property
    def index(self) -> int:
        """Frames played so far."""
        return self._index

    @property
    def total(self) -> int:
        """Frames in the take."""
        return len(self.frames)

    @property
    def exhausted(self) -> bool:
        """True once every frame has played."""
        return self._index >= len(self.frames)

    def next(self) -> SkeletonFrame | None:
        """The next frame, or None when the take is spent."""
        if self.exhausted:
            return None
        frame = self.frames[self._index]
        self._index += 1
        return frame

    def reset(self) -> None:
        """Play from the first frame again."""
        self._index = 0

    def payload(self, frame: SkeletonFrame) -> dict[str, Any]:
        """One frame as a ``skeleton``-channel payload — the shape
        the rig's ``drive_from_store`` reads back."""
        return {
            "skeleton": self._skeleton_dict,
            "frames": [{name: list(values)
                        for name, values in frame.values.items()}],
        }

    def to_dict(self) -> dict[str, Any]:
        """The player's state, as data."""
        return {
            "name": self.name,
            "index": self._index,
            "total": len(self.frames),
            "exhausted": self.exhausted,
        }
