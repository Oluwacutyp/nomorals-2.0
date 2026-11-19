"""Resolution buckets and the bucket sampler — a port of upstream ``dataset.py``'s resizing half.

Upstream ``ImageOrVideoDatasetWithResizing`` and
``ImageOrVideoDatasetWithResizeAndRectangleCrop`` (finetrainers/dataset.py)
pick a (frames, height, width) bucket per video and group training
batches by bucket so every batch shares one shape. This module ports
that logic as pure shape math (no torch, no decord): the callers that
need pixels are the encoder callables in
:mod:`nomorals.training.conditioning.precompute`.

Three upstream bugs are fixed here, each recorded in the survey and
pinned by a parity case — the port does not reproduce them:

* ``ImageOrVideoDatasetWithResizeAndRectangleCrop._preprocess_video``
  compares ``bucket <= video_num_frames`` where ``bucket`` is a tuple
  (``TypeError`` on Python 3); the port compares ``bucket[0]``.
* The same class's ``_find_nearest_resolution`` reads
  ``self.resolutions``, which does not exist (``AttributeError``); the
  port reads the resolution buckets.
* ``BucketSampler.__iter__`` only yields the leftover partial buckets
  inside ``if self.shuffle:`` — with ``shuffle=False`` the tail of the
  dataset is silently dropped. The port yields the leftovers either way.
* ``ImageOrVideoDataset._preprocess_video`` divides by
  ``video_num_frames // max_num_frames`` as a ``range`` step, which is
  ``0`` (``ValueError``) for any clip shorter than the frame cap; the
  port floors the step at 1.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...core.logging_setup import get_logger
from .errors import ConditioningError

__all__ = [
    "BucketSampler",
    "VideoItem",
    "find_nearest_resolution",
    "frame_bucket_for",
    "rectangle_crop_geometry",
    "sample_frame_indices",
]

_log = get_logger(__name__)

#: A (frames, height, width) bucket, as upstream's
#: ``--video_resolution_buckets`` entries.
ResolutionBucket = tuple[int, int, int]


@dataclass(frozen=True)
class VideoItem:
    """One captioned video, reduced to what bucketing needs.

    ``frames`` / ``height`` / ``width`` are the clip's real shape
    (upstream reads them off the decoded tensor); ``payload`` is
    whatever the caller wants back when the item is yielded — usually
    the (prompt, video path) pair.
    """

    frames: int
    height: int
    width: int
    payload: Any = None

    @property
    def shape(self) -> ResolutionBucket:
        return (self.frames, self.height, self.width)


def find_nearest_resolution(
    buckets: Sequence[ResolutionBucket], height: int, width: int
) -> tuple[int, int]:
    """The bucket resolution closest to ``(height, width)``.

    Verbatim port of ``ImageOrVideoDatasetWithResizing.
    _find_nearest_resolution``: minimise ``|bucket_h - height| +
    |bucket_w - width|`` and return the bucket's ``(height, width)``.
    """
    if not buckets:
        raise ConditioningError("no resolution buckets to choose from")
    nearest = min(buckets, key=lambda b: abs(b[1] - height) + abs(b[2] - width))
    return nearest[1], nearest[2]


def frame_bucket_for(
    num_frames: int,
    buckets: Sequence[ResolutionBucket],
    max_num_frames: int,
) -> int:
    """The frame count to sample for a clip of ``num_frames`` frames.

    Verbatim port of the resizing subclass's bucket pick: among buckets
    whose frame count fits the clip, take the one closest to
    ``min(num_frames, max_num_frames)``. Upstream passes ``default=1``
    to ``min`` and then indexes ``[0]`` — a clip shorter than every
    bucket would crash with ``TypeError``; the port returns ``1`` for
    that case (a single-frame sample) instead.
    """
    candidates = [b for b in buckets if b[0] <= num_frames]
    if not candidates:
        return 1
    nearest = min(
        candidates,
        key=lambda b: abs(b[0] - min(num_frames, max_num_frames)),
    )
    return nearest[0]


def sample_frame_indices(num_frames: int, max_num_frames: int) -> list[int]:
    """Evenly-spaced frame indices, capped at ``max_num_frames``.

    Port of upstream's ``range(0, num_frames, num_frames //
    max_num_frames)`` slice: upstream's step is ``0`` for clips shorter
    than the cap (``ValueError`` from ``range``); the port floors the
    step at 1, which returns every frame for a short clip.
    """
    if num_frames <= 0:
        raise ConditioningError(
            f"cannot sample frames from a clip with {num_frames} frames"
        )
    step = max(1, num_frames // max_num_frames)
    return list(range(0, num_frames, step))[:max_num_frames]


def rectangle_crop_geometry(
    arr_height: int,
    arr_width: int,
    image_size: tuple[int, int],
    mode: str = "center",
    *,
    rng: random.Random | None = None,
) -> dict[str, int]:
    """Resize + crop geometry for the rectangle-crop dataset variant.

    Port of ``ImageOrVideoDatasetWithResizeAndRectangleCrop.
    _resize_for_rectangle_crop`` as pure shape math: returns the resized
    shape and the crop origin ``(top, left)`` for the target
    ``image_size``. ``mode`` is ``"center"`` (default), ``"random"``, or
    ``"none"`` (upstream treats ``"none"`` exactly like ``"random"``);
    anything else raises instead of upstream's bare ``NotImplementedError``.
    """
    target_h, target_w = image_size
    if arr_width / arr_height > target_w / target_h:
        resized_h = target_h
        resized_w = int(arr_width * target_h / arr_height)
    else:
        resized_h = int(arr_height * target_w / arr_width)
        resized_w = target_w
    delta_h = resized_h - target_h
    delta_w = resized_w - target_w
    if mode in ("random", "none"):
        rand = rng or random
        top = rand.randint(0, delta_h + 1) if delta_h >= 0 else 0
        left = rand.randint(0, delta_w + 1) if delta_w >= 0 else 0
    elif mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise ConditioningError(
            f"unknown video_reshape_mode {mode!r} (expected 'center', "
            f"'random', or 'none')"
        )
    return {
        "resized_height": resized_h,
        "resized_width": resized_w,
        "top": top,
        "left": left,
        "height": target_h,
        "width": target_w,
    }


@dataclass
class BucketSampler:
    """Group items into batches that share one (frames, height, width).

    Port of upstream ``BucketSampler`` (finetrainers/dataset.py) without
    the torch ``DataLoader``: iterate the items in order, collect them
    into per-shape buckets, and yield a bucket as soon as it reaches
    ``batch_size``; at the end, yield the partial buckets too unless
    ``drop_last``. Upstream only yields those leftovers when shuffling
    — with ``shuffle=False`` they vanish; this port yields them either
    way (see the module docstring).
    """

    items: Sequence[VideoItem]
    batch_size: int = 8
    shuffle: bool = True
    drop_last: bool = False
    _warned_drop_last: bool = field(default=False, init=False, repr=False)

    def __len__(self) -> int:
        if self.drop_last and not self._warned_drop_last:
            object.__setattr__(self, "_warned_drop_last", True)
            _log.warning(
                "Calculating the length for bucket sampler is not possible "
                "when `drop_last` is set to True. This may cause problems "
                "when setting the number of epochs used for training."
            )
        return (len(self.items) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[VideoItem]]:
        buckets: dict[ResolutionBucket, list[VideoItem]] = {}
        for item in self.items:
            bucket = buckets.setdefault(item.shape, [])
            bucket.append(item)
            if len(bucket) == self.batch_size:
                if self.shuffle:
                    random.shuffle(bucket)
                yield bucket
                buckets[item.shape] = []
        if self.drop_last:
            return
        for shape, bucket in list(buckets.items()):
            if not bucket:
                continue
            if self.shuffle:
                random.shuffle(bucket)
            yield bucket
            buckets[shape] = []

    def batches(self) -> list[list[VideoItem]]:
        """All batches, eagerly — the convenient form for tests and tools."""
        return list(self)
