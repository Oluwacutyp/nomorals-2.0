"""Rig driver — mocap frames + face input in, a posed rig out.

The HyperJail rig is driven by three live streams: body mocap
(LiveLink), face capture (ARKit blendshapes), and voice (OVR
LipSync visemes). Devon's driver takes the same three shapes from
the organs that already produce them: a :class:`SkeletonFrame`
from the mocap store or a BVH take, blendshape weights over the
rig's ARKit set, and a viseme — explicit, from text through the
directed lip-sync organ's G2P, or from audio features through its
acoustic heuristic (labeled as such, never as phoneme recognition).

Poses are smoothed: every driven value lerps toward its target, so
a jittery tracker does not snap the rig. The first frame snaps.
Stale input is carried on the pose, never hidden; a caller that
needs fresh data refuses it before driving.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..directed.lipsync import (
    acoustic_viseme,
    mouth_shape_for_viseme,
    text_to_phonemes,
)
from ..mocap.bvh import SkeletonFrame
from .errors import RigError
from .remap import BoneRemap
from .rig import RigDefinition

__all__ = [
    "FaceInput",
    "PosedFrame",
    "RigDriver",
]


@dataclass
class FaceInput:
    """One frame of face input. Blendshape weights are ARKit names
    in 0..1; ``viseme`` is a name from the rig's viseme set; ``text``
    is spoken text (G2P, uniform timing — honest, not forced
    alignment); ``openness``/``centroid`` are the audio features
    the acoustic heuristic reads when there is no text. ``laughter``
    is 0..1 (the OVR frame's laughter score)."""

    blendshapes: dict[str, float] | None = None
    viseme: str | None = None
    text: str | None = None
    openness: float | None = None
    centroid: float | None = None
    laughter: float = 0.0


@dataclass(frozen=True)
class PosedFrame:
    """One posed rig frame: bone rotations (Z, X, Y degrees, rest
    included), blendshape weights, the viseme and its mouth shape,
    laughter, and the provenance flags."""

    seq: int
    rotations: dict[str, tuple[float, float, float]]
    blendshapes: dict[str, float]
    viseme: str
    mouth: tuple[float, float, float]
    laughter: float
    stale: bool = False
    unmapped: tuple[str, ...] = ()
    merged: tuple[str, ...] = ()
    face_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """The pose as plain JSON-able data."""
        return {
            "seq": self.seq,
            "rotations": {name: list(values)
                          for name, values in self.rotations.items()},
            "blendshapes": dict(self.blendshapes),
            "viseme": self.viseme,
            "mouth": list(self.mouth),
            "laughter": self.laughter,
            "stale": self.stale,
            "unmapped": list(self.unmapped),
            "merged": list(self.merged),
            "face_notes": list(self.face_notes),
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + alpha * (b - a)


class RigDriver:
    """Stateful driver: remap + smooth + face, one frame at a time."""

    def __init__(self, rig: RigDefinition, remap: BoneRemap, *,
                 smooth: float = 0.35, fps: float = 30.0,
                 phoneme_frames: int = 4) -> None:
        if not 0.0 <= smooth <= 1.0:
            raise RigError(
                f"smooth must be 0..1, got {smooth}",
                kind="bad_driver")
        if fps <= 0:
            raise RigError(f"fps must be positive, got {fps}",
                           kind="bad_driver")
        if phoneme_frames < 1:
            raise RigError("phoneme_frames must be >= 1",
                           kind="bad_driver")
        remap.check_against(rig)
        self.rig = rig
        self.remap = remap
        self.smooth = float(smooth)
        self.fps = float(fps)
        self.phoneme_frames = int(phoneme_frames)
        self._seq = 0
        self._rot: dict[str, tuple[float, float, float]] = {}
        self._mouth: tuple[float, float, float] | None = None
        self._blend: dict[str, float] = {}
        self._phonemes: list[str] = []
        self._phoneme_text: str | None = None

    # -- face ------------------------------------------------------

    def _face(self, face: FaceInput | None) -> tuple[
            dict[str, float], str, tuple[float, float, float],
            float, tuple[str, ...]]:
        """Resolve face input to (blendshapes, viseme, mouth,
        laughter, notes)."""
        notes: list[str] = []
        blend: dict[str, float] = {}
        if face is not None and face.blendshapes:
            for name, value in face.blendshapes.items():
                if name not in self.rig.blendshapes:
                    raise RigError(
                        f"blendshape {name!r} is not in rig "
                        f"{self.rig.name!r}", kind="bad_face")
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise RigError(
                        f"blendshape {name!r} value must be a "
                        f"number: {exc}", kind="bad_face") from exc
                clamped = _clamp01(number)
                if clamped != number:
                    notes.append(
                        f"blendshape {name} clamped "
                        f"{number} -> {clamped}")
                blend[name] = clamped
        viseme = "sil"
        if face is not None and (face.viseme or "").strip():
            viseme = (face.viseme or "").strip()
            if viseme not in self.rig.visemes:
                raise RigError(
                    f"viseme {viseme!r} is not in rig "
                    f"{self.rig.name!r}", kind="bad_face")
        elif face is not None and (face.text or "").strip():
            text = (face.text or "").strip()
            if text != self._phoneme_text:
                self._phonemes = text_to_phonemes(text)
                self._phoneme_text = text
            index = min(len(self._phonemes) - 1,
                        self._seq // self.phoneme_frames) \
                if self._phonemes else -1
            if index >= 0:
                from ..directed.lipsync import phoneme_to_viseme
                viseme = phoneme_to_viseme(self._phonemes[index])
        elif face is not None and face.openness is not None:
            centroid = face.centroid if face.centroid is not None \
                else 0.5
            viseme = acoustic_viseme(
                _clamp01(face.openness), _clamp01(centroid))
            notes.append("viseme is an acoustic heuristic "
                         "(energy + spectral centroid), not phoneme "
                         "recognition")
        mouth = mouth_shape_for_viseme(viseme)
        laughter = 0.0
        if face is not None:
            try:
                laughter = _clamp01(face.laughter)
            except (TypeError, ValueError) as exc:
                raise RigError(
                    f"laughter must be a number: {exc}",
                    kind="bad_face") from exc
            if laughter != float(face.laughter):
                notes.append(
                    f"laughter clamped {face.laughter} -> {laughter}")
        return blend, viseme, mouth, laughter, tuple(notes)

    # -- drive -----------------------------------------------------

    def drive(self, frame: SkeletonFrame | None,
              face: FaceInput | None = None, *,
              stale: bool = False) -> PosedFrame:
        """Drive one frame. ``frame=None`` holds the last pose (a
        dropped mocap frame never snaps the rig to rest)."""
        seq = self._seq
        self._seq += 1
        unmapped: tuple[str, ...] = ()
        merged: tuple[str, ...] = ()
        target: dict[str, tuple[float, float, float]] = {}
        if frame is not None:
            remapped = self.remap.apply(frame)
            unmapped = remapped.unmapped
            merged = remapped.merged
            for bone in self.rig.bones:
                rest = bone.rest
                values = remapped.values.get(bone.name)
                if values is None:
                    target[bone.name] = tuple(rest)
                    continue
                if len(values) == 6:
                    z, x, y = values[3], values[4], values[5]
                elif len(values) == 3:
                    z, x, y = values
                else:
                    raise RigError(
                        f"bone {bone.name!r} frame values must have "
                        f"3 or 6 channels, got {len(values)}",
                        kind="bad_frame")
                target[bone.name] = (
                    rest[0] + float(z),
                    rest[1] + float(x),
                    rest[2] + float(y),
                )
        else:
            target = dict(self._rot)
        rotations: dict[str, tuple[float, float, float]] = {}
        for bone in self.rig.bones:
            wanted = target.get(bone.name, tuple(bone.rest))
            previous = self._rot.get(bone.name)
            if previous is None or self.smooth >= 1.0:
                rotations[bone.name] = tuple(wanted)
            elif self.smooth <= 0.0:
                rotations[bone.name] = tuple(previous)
            else:
                rotations[bone.name] = tuple(
                    _lerp(p, w, self.smooth)
                    for p, w in zip(previous, wanted, strict=True))
        self._rot = rotations
        blend, viseme, mouth, laughter, notes = self._face(face)
        if self._mouth is None or self.smooth >= 1.0:
            smoothed_mouth = tuple(mouth)
        elif self.smooth <= 0.0:
            smoothed_mouth = tuple(self._mouth)
        else:
            smoothed_mouth = tuple(
                _lerp(p, w, self.smooth)
                for p, w in zip(self._mouth, mouth, strict=True))
        self._mouth = smoothed_mouth
        if blend:
            self._blend = dict(blend)
        return PosedFrame(
            seq=seq, rotations=dict(rotations),
            blendshapes=dict(self._blend), viseme=viseme,
            mouth=tuple(smoothed_mouth), laughter=laughter,
            stale=bool(stale), unmapped=unmapped, merged=merged,
            face_notes=notes)

    def reset(self) -> None:
        """Forget smoothing state — the next frame snaps."""
        self._seq = 0
        self._rot = {}
        self._mouth = None
        self._blend = {}
        self._phonemes = []
        self._phoneme_text = None
