"""Shot QA — frames + the Seer + a brain judge, against design intent.

The loop, honestly scoped: frames are sampled from the generated
clip with the existing cv_video extractor; the **Seer** describes
them (it is a sensor — it never decides); the **brain** judges the
descriptions against the shot's design intent and returns component
flags (subject present / lighting matches / composition sane). The
verdict is recomputed from the flags here — a shot passes only when
all three hold.

Anything that cannot run (no frames, no vision path, no judge)
yields ``qa_unavailable`` with the reason — never a pass, never a
silent skip. This is still-image QA over sampled frames; the full
temporal WatchVideo organ (motion, continuity, performance over
time) is a later build and this module does not pretend to be it.
"""

from __future__ import annotations

from typing import Any, Callable

from ...core.logging_setup import get_logger
from .breakdown import Shot

_log = get_logger(__name__)

__all__ = ["qa_shot", "QA_FRAME_COUNT"]

QA_FRAME_COUNT = 3

_SEER_QUESTION = (
    "Describe this film frame factually and precisely: the main "
    "subject and what they are doing, the direction and quality of "
    "the light, and the framing/composition (what fills the frame, "
    "where the subject sits). Do not judge or suggest anything.")


def _default_extract(clip_path: str) -> list[str]:
    from ...media_edit.cv_video import extract_frames
    res = extract_frames(clip_path, count=QA_FRAME_COUNT)
    return [str(f) for f in res.get("frames", [])]


def qa_shot(shot: Shot, clip_path: str, *, seer: Any = None,
            brain: Any = None,
            extract_frames_fn: Callable[[str], list[str]] | None = None
            ) -> dict[str, Any]:
    """Run the QA loop for one generated shot. Returns the QA record
    (also stored on ``shot.qa`` by the caller):

    ``{status: pass|fail|unavailable, subject_present, lighting_match,
    composition_ok, notes, frames, descriptions}``
    """
    from ...llm.base import Message

    record: dict[str, Any] = {"status": "unavailable", "frames": [],
                              "descriptions": [], "notes": ""}
    extract = extract_frames_fn or _default_extract
    try:
        frames = extract(clip_path)
    except Exception as exc:  # noqa: BLE001 - QA reports, never crashes
        record["notes"] = f"frame extraction failed: {exc}"
        return record
    if not frames:
        record["notes"] = "frame extraction produced no frames"
        return record
    record["frames"] = frames[:2]

    if seer is None:
        record["notes"] = "no vision path configured (Seer unavailable)"
        return record
    descriptions: list[str] = []
    try:
        for frame in record["frames"]:
            text = seer.see(frame, _SEER_QUESTION)
            descriptions.append(str(text))
    except Exception as exc:  # noqa: BLE001 - VisionUnavailable lands here
        record["notes"] = f"vision unavailable: {exc}"
        return record
    record["descriptions"] = descriptions

    if brain is None:
        record["notes"] = ("frames described but no brain configured "
                            "to judge them")
        return record
    design = shot.design
    intent = "; ".join(design.prompt_fragments()) if design else ""
    prompt = (
        "You are the QA judge on a film set. A shot was designed with "
        "this intent:\n"
        f"  shot: {shot.description}\n"
        f"  expected subject: {shot.subject}\n"
        f"  design: {intent}\n"
        "A vision sensor described sampled frames of the generated "
        "clip:\n"
        + "\n---\n".join(descriptions)
        + "\nJudge ONLY from those descriptions. Reply with JSON: "
        '{"subject_present": bool, "lighting_match": bool, '
        '"composition_ok": bool, "notes": str}. lighting_match = the '
        "described light's direction/quality plausibly matches the "
        "design. composition_ok = the described framing is coherent "
        "for the shot (subject placed sensibly, frame not empty or "
        "garbled).")
    try:
        data, resp = brain.chat_json([Message.user(prompt)],
                                     task_kind="qa")
    except Exception as exc:  # noqa: BLE001
        record["notes"] = f"QA judge failed: {exc}"
        return record
    if not isinstance(data, dict):
        err = getattr(resp, "error", "") or "no usable reply"
        record["notes"] = f"QA judge returned no verdict ({err})"
        return record
    subject = bool(data.get("subject_present"))
    lighting = bool(data.get("lighting_match"))
    composition = bool(data.get("composition_ok"))
    record.update({
        "subject_present": subject, "lighting_match": lighting,
        "composition_ok": composition,
        "notes": str(data.get("notes", "")),
        "status": "pass" if (subject and lighting and composition)
                  else "fail",
    })
    return record
