"""Landmark frames to a face take — the facemesh side of the
take format the facemcap organ built.

A video gives one landmark set (and one head matrix) per
frame; this module turns them into a :class:`FaceTake` in
the exact shape the rig, the show, and the ``facemcap``
tool already drive — ``morph_names`` is the rig's 52 ARKit
names, the bones carry the ``head`` rotation, the source is
the video the landmarks came from. Nothing here duplicates
the take format: it builds the same object the CSV loader
builds, from a camera instead of a CSV.
"""

from __future__ import annotations

from pathlib import Path

from ..facemcap.take import FaceFrame, FaceTake
from ..rig.rig import ARKIT_BLENDSHAPES
from .errors import FaceMeshError
from .landmarks import FaceRest, estimate_rest, landmarks_to_blendshapes, landmarks_to_head_bones

__all__ = ["landmarks_to_take"]

#: A landmark set is a list of (x, y, z) triples.
Landmarks = list[tuple[float, float, float]]


def landmarks_to_take(
        name: str,
        frames: list[tuple[float, Landmarks]],
        matrices: list[object | None] | None = None, *,
        fps: float,
        rest: FaceRest | None = None,
        root_name: str | None = None,
        source: str = "") -> FaceTake:
    """Frames of (time_s, landmarks) to a FaceTake.

    The rest baseline is estimated from the video's own
    frames when not given — the same frames that become the
    take, so a face that never opens its eyes in the video
    still gets its widest frame as its open eye. A frame
    with no head matrix (IMAGE mode) holds the head at rest.
    ``tongueOut`` reads 0 on every frame — no tongue
    landmark exists, and the mapping says so rather than
    faking one."""
    name = (name or "").strip()
    if not name:
        raise FaceMeshError("a face take needs a name",
                            kind="bad_landmarks")
    if not frames:
        raise FaceMeshError(
            "no landmark frames — a face take needs at least "
            "one frame with a face",
            kind="no_face")
    if fps <= 0:
        raise FaceMeshError(f"face take fps must be positive, "
                            f"got {fps}",
                            kind="bad_landmarks")
    use_rest = rest or estimate_rest([lm for _, lm in frames])
    use_matrices = matrices or [None] * len(frames)
    if len(use_matrices) != len(frames):
        raise FaceMeshError(
            f"{len(frames)} landmark frames but "
            f"{len(use_matrices)} head matrices",
            kind="bad_landmarks")
    take_frames = []
    for (time_s, lm), matrix in zip(frames, use_matrices,
                                    strict=True):
        take_frames.append(FaceFrame(
            time_s=float(time_s),
            blendshapes=landmarks_to_blendshapes(lm,
                                                 rest=use_rest),
            bones=landmarks_to_head_bones(matrix)))
    return FaceTake(
        name=name,
        root_name=root_name or Path(source).stem or name,
        fps=float(fps),
        morph_names=list(ARKIT_BLENDSHAPES),
        frames=take_frames,
        unmapped=(),
        source=source)
