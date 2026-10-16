"""Generative video as EditSpec ops.

Image-to-video and video-to-video are not a separate pipeline: they
are ops on an :class:`EditSpec`, compiled and rendered by the same
engine as everything else.

* ``{"op": "motion_still", "image": ..., "duration": 5.0, "move": "zoom_in"}``
  — animate a still (Ken Burns camera move at compile time; the neural
  i2v backend at execution time when one is available).
* ``{"op": "restyle", "source": ..., "grade": "teal_orange"}``
  — restyle a clip (a ``color_grade`` effect at compile time; the same
  effect chain at execution time).

Compile (:func:`genop_to_spec`, :func:`apply_genop`) is pure — no
model, no ffmpeg, just spec math. Execute (:func:`execute_genop`) is
the real thing: i2v goes through the videogen front door
(neural when the hardware is there, motion studio otherwise, and the
result always says which), v2v restyle goes through the engine's
:func:`apply_effect`.

Invariant: compiling the same op twice yields equal spec dicts, and
every compiled op is renderable by the engine (its effects are all in
the engine's effect registry).
"""

from __future__ import annotations

from typing import Any

from ...media_edit.videos import MediaEditError
from .effects import list_color_grades
from .spec import EditSpec
from .timeline import Clip, Effect

__all__ = ["GEN_OPS", "validate_genop", "genop_to_spec", "apply_genop",
           "execute_genop"]

GEN_OPS = ("motion_still", "restyle")


def _known_moves() -> set[str]:
    from ..motion_studio.kenburns import list_moves  # lazy: same layer
    return {m["name"] for m in list_moves()} | {"auto", "random"}


def validate_genop(op: Any) -> dict[str, Any]:
    """Validate + normalize a gen op dict. Raises on anything malformed."""
    if not isinstance(op, dict):
        raise MediaEditError(f"gen op must be a dict, got {op!r}")
    kind = str(op.get("op", "")).strip()
    if kind not in GEN_OPS:
        raise MediaEditError(
            f"unknown gen op {kind!r} — use: {list(GEN_OPS)}")
    if kind == "motion_still":
        image = str(op.get("image", "") or "").strip()
        if not image:
            raise MediaEditError("motion_still needs 'image'")
        duration = float(op.get("duration", 5.0))
        if duration <= 0:
            raise MediaEditError("motion_still needs duration > 0")
        move = str(op.get("move", "zoom_in") or "zoom_in").strip().lower()
        if move not in _known_moves():
            raise MediaEditError(
                f"unknown motion_still move {move!r} — use: "
                f"{sorted(_known_moves())}")
        return {"op": "motion_still", "image": image, "duration": duration,
                "move": move, "seed": int(op.get("seed", 0))}
    # restyle
    source = str(op.get("source", "") or "").strip()
    if not source:
        raise MediaEditError("restyle needs 'source'")
    grade = str(op.get("grade", "") or "").strip().lower()
    if grade not in list_color_grades():
        raise MediaEditError(
            f"unknown restyle grade {grade!r}; use: {list_color_grades()}")
    duration = op.get("duration")
    if duration is not None:
        duration = float(duration)
        if duration <= 0:
            raise MediaEditError("restyle needs duration > 0")
    target = op.get("clip_idx")
    if target is not None:
        target = int(target)
        if target < 0:
            raise MediaEditError("restyle clip_idx must be >= 0")
    return {"op": "restyle", "source": source, "grade": grade,
            "duration": duration, "clip_idx": target,
            "seed": int(op.get("seed", 0))}


def _motion_clip(op: dict[str, Any], *, fps: int, width: int,
                 height: int) -> Clip:
    """A motion_still op → one Clip with a Ken Burns effect.

    The engine's ``ken_burns`` effect is the compile-time form of the
    camera move (``zoom_in`` → direction ``in``, the rest map to pans
    via ``pan_x``/``pan_y`` — the effect builder owns that vocabulary).
    """
    move = op["move"]
    params: dict[str, Any] = {"direction": "out" if "out" in move else "in"}
    if move in ("pan_left", "left"):
        params["pan_x"] = -12.0
    elif move in ("pan_right", "right"):
        params["pan_x"] = 12.0
    elif move in ("pan_up", "up"):
        params["pan_y"] = -12.0
    elif move in ("pan_down", "down"):
        params["pan_y"] = 12.0
    return Clip(path=op["image"], duration=float(op["duration"]),
                effects=(Effect(name="ken_burns", params=params),))


def genop_to_spec(op: Any, *, fps: int = 30, width: int = 1080,
                  height: int = 1920, seed: int = 0) -> EditSpec:
    """Pure: one gen op → a standalone :class:`EditSpec`."""
    norm = validate_genop(op)
    if norm["op"] == "motion_still":
        clip = _motion_clip(norm, fps=fps, width=width, height=height)
        return EditSpec(clips=[clip], segments=[(0, norm["duration"])],
                        fps=fps, width=width, height=height,
                        seed=int(norm.get("seed", seed)))
    clip = Clip(path=norm["source"],
                duration=norm["duration"],
                effects=(Effect(name="color_grade",
                                params={"grade": norm["grade"]}),))
    dur = norm["duration"]
    if dur is None:
        raise MediaEditError(
            "restyle as a standalone spec needs 'duration' (the source "
            "clip's play length) — pass it explicitly")
    return EditSpec(clips=[clip], segments=[(0, float(dur))],
                    fps=fps, width=width, height=height,
                    seed=int(norm.get("seed", seed)))


def apply_genop(spec: EditSpec, op: Any) -> EditSpec:
    """Pure: apply a gen op to an existing spec → a new spec.

    ``motion_still`` appends its clip (+ segment) to the timeline.
    ``restyle`` appends the ``color_grade`` effect to the target clip
    (``clip_idx``) or to every clip when no target is given — and when
    the op carries a ``source`` that is not already in the spec, it
    appends it as a new restyled clip instead of guessing.
    """
    norm = validate_genop(op)
    clips = list(spec.clips)
    segments = list(spec.segments) if spec.segments else None
    if norm["op"] == "motion_still":
        clips.append(_motion_clip(norm, fps=spec.fps, width=spec.width,
                                  height=spec.height))
        if segments is not None:
            segments.append((len(clips) - 1, float(norm["duration"])))
        return EditSpec(clips=clips, segments=segments,
                        transition=spec.transition,
                        text_layers=list(spec.text_layers),
                        audio=spec.audio, fps=spec.fps,
                        width=spec.width, height=spec.height,
                        seed=spec.seed, crf=spec.crf, preset=spec.preset,
                        profile=spec.profile, out=spec.out,
                        keep_clip_audio=spec.keep_clip_audio)
    # restyle
    effect = Effect(name="color_grade", params={"grade": norm["grade"]})
    target = norm.get("clip_idx")
    if target is not None:
        if target >= len(clips):
            raise MediaEditError(
                f"restyle clip_idx {target} out of range "
                f"({len(clips)} clip(s))")
        old = clips[target]
        clips[target] = Clip(path=old.path, start=old.start, end=old.end,
                             at=old.at, duration=old.duration,
                             speed=old.speed,
                             effects=tuple(old.effects) + (effect,),
                             gain=old.gain)
    else:
        hit = [i for i, c in enumerate(clips) if c.path == norm["source"]]
        if hit:
            for i in hit:
                old = clips[i]
                clips[i] = Clip(path=old.path, start=old.start,
                                 end=old.end, at=old.at,
                                 duration=old.duration, speed=old.speed,
                                 effects=tuple(old.effects) + (effect,),
                                 gain=old.gain)
        else:
            if norm.get("duration") is None:
                raise MediaEditError(
                    "restyle of a source not in the spec needs 'duration'")
            clips.append(Clip(path=norm["source"],
                              duration=float(norm["duration"]),
                              effects=(effect,)))
            if segments is not None:
                segments.append((len(clips) - 1, float(norm["duration"])))
    return EditSpec(clips=clips, segments=segments,
                    transition=spec.transition,
                    text_layers=list(spec.text_layers),
                    audio=spec.audio, fps=spec.fps,
                    width=spec.width, height=spec.height,
                    seed=spec.seed, crf=spec.crf, preset=spec.preset,
                    profile=spec.profile, out=spec.out,
                    keep_clip_audio=spec.keep_clip_audio)


def execute_genop(op: Any, *, out: str | None = None) -> dict[str, Any]:
    """Execute a gen op for real (not a compile — this renders/generates).

    ``motion_still`` → :func:`videogen.generate` in ``i2v`` mode (neural
    when available, motion studio otherwise — the result says which).
    ``restyle`` → the engine's :func:`apply_effect` with the
    ``color_grade`` effect. Returns a dict with at least ``output``.
    """
    norm = validate_genop(op)
    if norm["op"] == "motion_still":
        from ..videogen.pipeline import generate  # lazy: same layer
        result = generate("", mode="i2v", image=norm["image"],
                          duration_s=norm["duration"], seed=norm["seed"],
                          out=out)
        return {"output": result.path, "engine": result.engine,
                "routed": result.routed, "note": result.note,
                "op": "motion_still"}
    from .render import apply_effect  # lazy: same layer
    res = apply_effect(norm["source"], "color_grade", out=out,
                       grade=norm["grade"])
    return {**res, "op": "restyle"}
