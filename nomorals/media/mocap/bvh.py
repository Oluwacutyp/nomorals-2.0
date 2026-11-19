"""BVH read/write — the mocap interchange format for game engines.

Upstream's ``MocapNet`` job turns video into "BVH mocap animation
data for game engines"; this module is Devon's own BVH reader and
writer for that interchange shape (see ``MOCAP_STATE_SURVEY.md`` —
pattern only, no upstream code). BVH is a plain-text format: a
``HIERARCHY`` of joints with offsets and channels, then a ``MOTION``
section of frame values. The writer emits it; the parser reads it
back; the round trip is exact to within float formatting.

A skeleton is a joint tree. The root joint carries 6 channels
(position + rotation); every other joint carries 3 (rotation only)
unless stated otherwise. ``End Site`` joints carry no channels —
they are the tips of the hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import MocapError

__all__ = [
    "Joint",
    "Skeleton",
    "SkeletonFrame",
    "hierarchy_joints",
    "parse_bvh",
    "skeleton_to_dict",
    "skeleton_from_dict",
    "write_bvh",
]

ROOT_CHANNELS: tuple[str, ...] = (
    "Xposition", "Yposition", "Zposition",
    "Zrotation", "Xrotation", "Yrotation",
)
JOINT_CHANNELS: tuple[str, ...] = ("Zrotation", "Xrotation", "Yrotation")


@dataclass
class Joint:
    """One joint in the hierarchy."""

    name: str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    channels: tuple[str, ...] = JOINT_CHANNELS
    children: list[Joint] = field(default_factory=list)
    is_end_site: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise MocapError("joint name must not be empty",
                             kind="bad_skeleton")
        if len(self.offset) != 3:
            raise MocapError(
                f"joint {self.name!r} offset must have 3 values",
                kind="bad_skeleton")
        if self.is_end_site and self.channels:
            raise MocapError(
                f"End Site joint {self.name!r} must carry no channels",
                kind="bad_skeleton")


@dataclass
class Skeleton:
    """A joint hierarchy plus the frame time in seconds."""

    root: Joint
    frame_time: float = 1.0 / 30.0

    def __post_init__(self) -> None:
        if self.frame_time <= 0:
            raise MocapError("frame_time must be positive",
                             kind="bad_skeleton")


@dataclass
class SkeletonFrame:
    """One motion frame: joint name → channel values, in the joint's
    own channel order."""

    values: dict[str, tuple[float, ...]]


def hierarchy_joints(skeleton: Skeleton) -> list[Joint]:
    """Every joint in hierarchy order (parent before children)."""
    ordered: list[Joint] = []

    def walk(joint: Joint) -> None:
        ordered.append(joint)
        for child in joint.children:
            walk(child)

    walk(skeleton.root)
    return ordered


def skeleton_to_dict(skeleton: Skeleton) -> dict:
    """A skeleton as plain JSON-able data (for channel payloads)."""

    def joint_dict(joint: Joint) -> dict:
        return {
            "name": joint.name,
            "offset": list(joint.offset),
            "channels": list(joint.channels),
            "is_end_site": joint.is_end_site,
            "children": [joint_dict(child) for child in joint.children],
        }

    return {"root": joint_dict(skeleton.root),
            "frame_time": skeleton.frame_time}


def skeleton_from_dict(data: dict) -> Skeleton:
    """Rebuild a skeleton from :func:`skeleton_to_dict` output."""

    def joint_from(node: dict) -> Joint:
        return Joint(
            name=node["name"],
            offset=tuple(float(v) for v in node["offset"]),
            channels=tuple(node["channels"]),
            children=[joint_from(child)
                      for child in node.get("children", [])],
            is_end_site=bool(node.get("is_end_site", False)),
        )

    try:
        return Skeleton(root=joint_from(data["root"]),
                        frame_time=float(data["frame_time"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise MocapError(
            f"skeleton data is malformed: {exc}",
            kind="bad_skeleton") from exc


def _check_frame(skeleton: Skeleton, frame: SkeletonFrame,
                 index: int) -> None:
    for joint in hierarchy_joints(skeleton):
        if not joint.channels:
            continue
        values = frame.values.get(joint.name)
        if values is None:
            raise MocapError(
                f"frame {index} has no values for joint "
                f"{joint.name!r}", kind="bad_frame")
        if len(values) != len(joint.channels):
            raise MocapError(
                f"frame {index} joint {joint.name!r} needs "
                f"{len(joint.channels)} values, got {len(values)}",
                kind="bad_frame")


def write_bvh(skeleton: Skeleton,
              frames: list[SkeletonFrame]) -> str:
    """Serialise a skeleton and its frames as BVH text."""
    for index, frame in enumerate(frames):
        _check_frame(skeleton, frame, index)
    lines: list[str] = ["HIERARCHY"]

    def emit(joint: Joint, depth: int) -> None:
        pad = "  " * depth
        if joint.is_end_site:
            lines.append(f"{pad}End Site")
        else:
            kind = "ROOT" if joint is skeleton.root else "JOINT"
            lines.append(f"{pad}{kind} {joint.name}")
        lines.append(f"{pad}{{")
        lines.append(
            f"{pad}  OFFSET "
            f"{joint.offset[0]:.6f} {joint.offset[1]:.6f} "
            f"{joint.offset[2]:.6f}")
        if joint.channels:
            lines.append(
                f"{pad}  CHANNELS {len(joint.channels)} "
                + " ".join(joint.channels))
        for child in joint.children:
            emit(child, depth + 1)
        lines.append(f"{pad}}}")

    emit(skeleton.root, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(frames)}")
    lines.append(f"Frame Time: {skeleton.frame_time:.6f}")
    ordered = [joint for joint in hierarchy_joints(skeleton)
               if joint.channels]
    for frame in frames:
        values: list[str] = []
        for joint in ordered:
            values.extend(f"{v:.6f}"
                          for v in frame.values[joint.name])
        lines.append(" ".join(values))
    return "\n".join(lines) + "\n"


def _tokenise(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for char in text:
        if char in "{}":
            flush()
            tokens.append(char)
        elif char.isspace():
            flush()
        else:
            current.append(char)
    flush()
    return tokens


class _Parser:
    """Recursive-descent BVH parser over the token stream."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._pos = 0

    def peek(self) -> str:
        if self._pos >= len(self._tokens):
            raise MocapError("unexpected end of BVH text",
                             kind="bad_bvh")
        return self._tokens[self._pos]

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if expected is not None and token != expected:
            raise MocapError(
                f"expected {expected!r} in BVH text, got {token!r}",
                kind="bad_bvh")
        self._pos += 1
        return token

    def number(self) -> float:
        token = self.take()
        try:
            return float(token)
        except ValueError as exc:
            raise MocapError(
                f"expected a number in BVH text, got {token!r}",
                kind="bad_bvh") from exc

    def joint(self, kind: str) -> Joint:
        # An End Site carries no name in BVH text — "End Site {"
        # opens the block directly.
        name = "End Site" if kind == "End Site" else self.take()
        self.take("{")
        self.take("OFFSET")
        offset = (self.number(), self.number(), self.number())
        channels: tuple[str, ...] = ()
        children: list[Joint] = []
        if kind != "End Site" and self.peek() == "CHANNELS":
            self.take()
            count = int(self.number())
            channels = tuple(self.take() for _ in range(count))
        while self.peek() != "}":
            token = self.peek()
            if token == "JOINT":
                self.take()
                children.append(self.joint("JOINT"))
            elif token == "End":
                self.take()
                self.take("Site")
                children.append(self.joint("End Site"))
            else:
                raise MocapError(
                    f"unexpected {token!r} inside joint {name!r}",
                    kind="bad_bvh")
        self.take("}")
        return Joint(name=name, offset=offset, channels=channels,
                     children=children,
                     is_end_site=(kind == "End Site"))

    def skeleton(self) -> Skeleton:
        self.take("HIERARCHY")
        kind = self.take()
        if kind != "ROOT":
            raise MocapError(
                f"BVH hierarchy must start with ROOT, got {kind!r}",
                kind="bad_bvh")
        root = self.joint("ROOT")
        return Skeleton(root=root)

    def motion(self, skeleton: Skeleton) -> list[SkeletonFrame]:
        self.take("MOTION")
        self.take("Frames:")
        count = int(self.number())
        self.take("Frame")
        self.take("Time:")
        frame_time = self.number()
        skeleton.frame_time = frame_time
        ordered = [joint for joint in hierarchy_joints(skeleton)
                   if joint.channels]
        frames: list[SkeletonFrame] = []
        for _ in range(count):
            values: dict[str, tuple[float, ...]] = {}
            for joint in ordered:
                values[joint.name] = tuple(
                    self.number() for _ in joint.channels)
            frames.append(SkeletonFrame(values=values))
        if self._pos != len(self._tokens):
            raise MocapError(
                f"trailing data after BVH motion: "
                f"{self._tokens[self._pos]!r}", kind="bad_bvh")
        return frames


def parse_bvh(text: str) -> tuple[Skeleton, list[SkeletonFrame]]:
    """Parse BVH text into a skeleton and its frames. Malformed
    input is refused with the position named, never half-parsed."""
    if not text or not text.strip():
        raise MocapError("BVH text is empty", kind="bad_bvh")
    parser = _Parser(_tokenise(text))
    skeleton = parser.skeleton()
    frames = parser.motion(skeleton)
    return skeleton, frames
