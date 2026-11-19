"""Bridges between Devon's engines and the mocap state store.

Producers (they write channels):

* :class:`KinectProducer` — the Kinect capture path. From a live
  :class:`Capture` it publishes the ``depth`` and ``pointcloud``
  channels; from the vendored fixture (no device needed) it
  publishes the same shape from the recorded depth frame. The point
  cloud is downsampled by stride — a full 640x576 cloud is 368,640
  points and does not belong in a shared-state payload — and the
  stats always cover the full frame.
* :class:`BvhPlaybackProducer` — a BVH take played into the
  ``skeleton`` channel: the snapshot holds the skeleton plus the
  frames so far, bounded so a long take drops its oldest frames and
  says so.

Consumers (they read channels):

* :class:`BvhExporter` — reads the ``skeleton`` channel and writes
  a ``.bvh`` file. The absolute output path is recorded in the
  export result (and on the ``bvh`` channel) at export time — the
  upstream rule that completion locations are absolute paths
  recorded when the work completes, so they stay correct even if
  the download directory changes later.

No live Kinect body tracking exists in Devon (lane 122's port has
no body-tracking API — it needs a separate library), so the Kinect
producer carries depth/point-cloud channels and ``skeleton`` comes
from BVH playback. That is stated here, not faked.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...hardware.kinect import (
    Calibration,
    Capture,
    KinectError,
    parse_raw_calibration,
    point_cloud,
)
from ...hardware.kinect.frames import (
    decode_depth16,
    depth_stats,
)
from .bvh import (
    Skeleton,
    SkeletonFrame,
    hierarchy_joints,
    parse_bvh,
    skeleton_from_dict,
    skeleton_to_dict,
    write_bvh,
)
from .errors import MocapError
from .state import MocapStateStore

__all__ = [
    "BvhExporter",
    "BvhPlaybackProducer",
    "KinectProducer",
]

DEFAULT_MAX_FRAMES = 600
DEFAULT_STRIDE = 16


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class KinectProducer:
    """Publishes Kinect captures into the store as ``depth`` and
    ``pointcloud`` channels."""

    def __init__(self, store: MocapStateStore, *,
                 engine: str = "kinect",
                 max_age_s: float = 5.0) -> None:
        self._store = store
        self._engine = engine
        self._max_age_s = max_age_s
        _ensure_engine(
            store, engine, produces=("depth", "pointcloud"),
            note="Kinect capture producer — depth + point cloud")

    def publish_capture(self, capture: Capture,
                        calibration: Calibration, *,
                        stride: int = DEFAULT_STRIDE) -> dict[str, Any]:
        """Publish one capture's depth frame. The depth channel
        carries the stats; the pointcloud channel carries a
        stride-sampled cloud plus the same stats."""
        if not capture.has_depth or capture.depth is None:
            raise MocapError(
                "capture has no depth frame to publish",
                kind="no_depth")
        depth = capture.depth
        rows = depth.depth_values()
        stats = depth_stats(rows)
        self._store.write(
            "depth", self._engine,
            {"width": depth.width, "height": depth.height,
             "valid_pixels": stats.valid_pixels,
             "zero_pixels": stats.zero_pixels,
             "min_mm": stats.min_mm, "max_mm": stats.max_mm,
             "mean_mm": stats.mean_mm,
             "temperature_c": capture.temperature_c},
            kind="snapshot", max_age_s=self._max_age_s)
        cloud = point_cloud(calibration, rows)
        sampled = [list(cloud[y * depth.width + x])
                   for y in range(0, depth.height, stride)
                   for x in range(0, depth.width, stride)]
        receipt = self._store.write(
            "pointcloud", self._engine,
            {"width": depth.width, "height": depth.height,
             "stride": stride, "count": len(cloud),
             "sampled": len(sampled),
             "valid_pixels": stats.valid_pixels,
             "min_mm": stats.min_mm, "max_mm": stats.max_mm,
             "mean_mm": stats.mean_mm, "points": sampled},
            kind="snapshot", max_age_s=self._max_age_s)
        receipt["points_sampled"] = len(sampled)
        receipt["points_total"] = len(cloud)
        return receipt

    def publish_fixture(self, fixtures_dir: str | Path, *,
                        stride: int = DEFAULT_STRIDE) -> dict[str, Any]:
        """Publish the vendored Kinect fixture as if it were a live
        capture — the no-device path. The fixture is the SDK's own
        unit-test calibration plus a recorded NFOV depth frame."""
        root = Path(fixtures_dir).expanduser()
        depth_path = root / "depth_nfov_640x576.raw"
        calib_path = root / "calibration.json"
        if not depth_path.exists():
            raise MocapError(
                f"no Kinect depth fixture at {depth_path}",
                kind="missing_fixture")
        if not calib_path.exists():
            raise MocapError(
                f"no Kinect calibration fixture at {calib_path}",
                kind="missing_fixture")
        try:
            calibration = parse_raw_calibration(
                calib_path.read_text(encoding="utf-8"))
        except (ValueError, KinectError) as exc:
            raise MocapError(
                f"fixture calibration is unreadable: {exc}",
                kind="bad_fixture") from exc
        raw = depth_path.read_bytes()
        width, height = 640, 576
        rows = decode_depth16(raw, width, height, width * 2)
        stats = depth_stats(rows)
        self._store.write(
            "depth", self._engine,
            {"width": width, "height": height,
             "valid_pixels": stats.valid_pixels,
             "zero_pixels": stats.zero_pixels,
             "min_mm": stats.min_mm, "max_mm": stats.max_mm,
             "mean_mm": stats.mean_mm, "source": "fixture"},
            kind="snapshot", max_age_s=self._max_age_s)
        cloud = point_cloud(calibration, rows)
        sampled = [list(cloud[y * width + x])
                   for y in range(0, height, stride)
                   for x in range(0, width, stride)]
        receipt = self._store.write(
            "pointcloud", self._engine,
            {"width": width, "height": height, "stride": stride,
             "count": len(cloud), "sampled": len(sampled),
             "valid_pixels": stats.valid_pixels,
             "min_mm": stats.min_mm, "max_mm": stats.max_mm,
             "mean_mm": stats.mean_mm, "points": sampled,
             "source": "fixture"},
            kind="snapshot", max_age_s=self._max_age_s)
        receipt["points_sampled"] = len(sampled)
        receipt["points_total"] = len(cloud)
        return receipt


class BvhPlaybackProducer:
    """Plays a BVH take into the ``skeleton`` channel. The snapshot
    holds the skeleton plus the frames so far, bounded at
    ``max_frames`` — past that the oldest frames drop and the
    dropped count is carried in the payload, never hidden."""

    def __init__(self, store: MocapStateStore, *,
                 engine: str = "bvh-playback",
                 max_frames: int = DEFAULT_MAX_FRAMES) -> None:
        self._store = store
        self._engine = engine
        self._max_frames = max(1, int(max_frames))
        self._skeleton: Skeleton | None = None
        self._frames: list[SkeletonFrame] = []
        self._dropped = 0
        _ensure_engine(
            store, engine, produces=("skeleton",),
            note="BVH take playback — skeleton channel producer")

    def load_take(self, skeleton: Skeleton,
                  frames: list[SkeletonFrame]) -> dict[str, Any]:
        """Load a whole take at once (e.g. parsed from a BVH file)
        and publish it."""
        self._skeleton = skeleton
        self._frames = list(frames)
        self._dropped = max(0, len(self._frames) - self._max_frames)
        if self._dropped:
            self._frames = self._frames[-self._max_frames:]
        return self.publish()

    def load_bvh_text(self, text: str) -> dict[str, Any]:
        """Parse BVH text and publish the take it holds."""
        skeleton, frames = parse_bvh(text)
        return self.load_take(skeleton, frames)

    def add_frame(self, frame: SkeletonFrame) -> dict[str, Any]:
        """Append one frame to the current take and publish."""
        if self._skeleton is None:
            raise MocapError(
                "no take loaded — load_take first",
                kind="no_take")
        self._frames.append(frame)
        if len(self._frames) > self._max_frames:
            self._frames.pop(0)
            self._dropped += 1
        return self.publish()

    def publish(self) -> dict[str, Any]:
        """Publish the current take as the ``skeleton`` snapshot."""
        if self._skeleton is None:
            raise MocapError(
                "no take loaded — load_take first",
                kind="no_take")
        payload = {
            "skeleton": skeleton_to_dict(self._skeleton),
            "frames": [{name: list(values)
                        for name, values in frame.values.items()}
                       for frame in self._frames],
            "frame_count": len(self._frames),
            "dropped_frames": self._dropped,
            "frame_time": self._skeleton.frame_time,
        }
        return self._store.write("skeleton", self._engine, payload,
                                 kind="snapshot")

    @property
    def frame_count(self) -> int:
        """Frames held in the current take."""
        return len(self._frames)

    @property
    def dropped_frames(self) -> int:
        """Frames dropped from the take by the bound."""
        return self._dropped


class BvhExporter:
    """Consumes the ``skeleton`` channel and writes a ``.bvh`` file
    — the mocap output game engines read."""

    def export_from_store(self, store: MocapStateStore, path: str | Path,
                          *, channel: str = "skeleton") -> dict[str, Any]:
        """Read the take off the channel and write it out. The
        absolute output path is recorded on the ``bvh`` channel at
        export time, and returned."""
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
        return self.export_frames(skeleton, frames, path, store=store)

    def export_frames(self, skeleton: Skeleton,
                      frames: list[SkeletonFrame], path: str | Path,
                      *, store: MocapStateStore | None = None
                      ) -> dict[str, Any]:
        """Write frames out as BVH. Returns the absolute path, the
        frame count, and the joint names — and, when a store is
        given, records the same on the ``bvh`` channel."""
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        text = write_bvh(skeleton, frames)
        target.write_text(text, encoding="utf-8")
        result = {
            "path": str(target),
            "frames": len(frames),
            "joints": [joint.name for joint in
                       hierarchy_joints(skeleton)],
            "exported_at": _iso_now(),
        }
        if store is not None:
            _ensure_engine(
                store, "bvh", produces=("bvh",),
                note="local BVH reader/writer")
            store.write("bvh", "bvh",
                        {"path": str(target), "frames": len(frames),
                         "exported_at": result["exported_at"]},
                        kind="snapshot")
        return result

    @staticmethod
    def read_bvh(path: str | Path) -> dict[str, Any]:
        """Parse a BVH file: joints, frame count, frame time, and
        the first frame's values. A missing or malformed file is
        refused by name."""
        target = Path(path).expanduser()
        if not target.exists():
            raise MocapError(f"no BVH file at {target}",
                             kind="missing_bvh")
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise MocapError(
                f"BVH file at {target} is unreadable: {exc}",
                kind="bad_bvh") from exc
        skeleton, frames = parse_bvh(text)
        first = frames[0].values if frames else {}
        return {
            "path": str(target.resolve()),
            "joints": [joint.name for joint in
                       hierarchy_joints(skeleton)],
            "frames": len(frames),
            "frame_time": skeleton.frame_time,
            "first_frame": {name: list(values)
                            for name, values in first.items()},
        }


def _ensure_engine(store: MocapStateStore, name: str, *,
                   produces: tuple[str, ...] = (),
                   note: str = "") -> None:
    """Register an engine only when it is not already on the
    roster — repeat exports must not fail on a duplicate."""
    if name not in {engine["name"] for engine in store.engines()}:
        store.register_engine(name, produces=produces, configured=True,
                              note=note)


def load_bvh_json(path: str | Path) -> dict[str, Any]:
    """Load a skeleton + frames from a JSON file shaped like the
    ``skeleton`` channel payload (``{"skeleton": …, "frames": …}``).
    A missing or malformed file is refused by name."""
    target = Path(path).expanduser()
    if not target.exists():
        raise MocapError(f"no JSON take at {target}",
                         kind="missing_take")
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise MocapError(
            f"JSON take at {target} is unreadable: {exc}",
            kind="bad_take") from exc
    if not isinstance(blob, dict) or "skeleton" not in blob:
        raise MocapError(
            f"JSON take at {target} has no 'skeleton' key",
            kind="bad_take")
    skeleton = skeleton_from_dict(blob["skeleton"])
    frames = [
        SkeletonFrame(values={
            name: tuple(float(v) for v in values)
            for name, values in raw.items()})
        for raw in blob.get("frames", [])]
    return {"skeleton": skeleton, "frames": frames}
