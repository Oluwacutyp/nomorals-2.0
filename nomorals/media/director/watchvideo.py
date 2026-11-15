"""WatchVideo — temporal QA: watch a rendered video over time.

The director's shot QA (:mod:`nomorals.media.director.qa`) judges
still frames against design intent. This organ is the other half:
it watches a finished render *as a video* and verifies the edit
actually landed — cuts on the expected grid, segments at the planned
durations, motion present where an effect should be moving the frame.

How it measures (no trust in the renderer's report): the video is
sampled at its native frame rate through the existing ffmpeg frame
extractor (:func:`nomorals.media_edit.videos.extract_frames`), and
consecutive frames are differenced (mean absolute grayscale
difference, 0..1, via PIL — no cv2 needed; this box has none). The
decision is *relative*: a cut must change the picture clearly more
than the video's own typical frame-to-frame change. The thresholds
come from measurement, not tuning: on a real TrendOps cut from
moving footage the cuts measure 2.23x–4.92x the video's median frame
change, while the same footage rendered with no cuts at all peaks at
1.41x — so the bar sits between the two classes, at 1.8x, and every
verdict carries the measured numbers so a marginal call is visible
as marginal.

What it does NOT do: it cannot verify cuts in footage that carries
no measurable frame variation at all (a solid-colour clip — every
frame identical, effects included, since a spatial transform of a
uniform field is still uniform). That is reported honestly as
``qa_unavailable`` with the reason — unmeasurable, never a fake
pass and never a fail for a render that may be perfectly fine.
Likewise an unreadable file, a missing ffmpeg, or any sampling
failure yields ``qa_unavailable`` with the reason, never a crash.

Verdict shape (also what lands in a TrendOps review's ``qa`` leg)::

    {"status": "pass" | "fail" | "qa_unavailable",
     "notes": "<one-line human summary>",
     "checks": {"cuts": {...}, "pacing": {...}, "motion": {...}},
     "duration": float, "fps": float, "frames": int,
     "backend": "ffmpeg"}

Each check carries its own status (``pass`` / ``fail`` /
``unavailable``) plus the evidence: the cuts leg lists found and
missing cuts with the measured diff and ratio for each; pacing lists
planned vs measured segment durations; motion lists each segment's
median frame change.
"""

from __future__ import annotations

import statistics
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from ...media_edit.videos import MediaEditError, extract_frames, video_probe

_log = get_logger(__name__)

__all__ = ["watch_video", "CUT_RATIO", "CUT_TOL_S", "PACE_TOL_S"]

#: A cut's frame change must be at least this multiple of the video's
#: median frame-to-frame change. Measured: real cuts 2.23x–4.92x, the
#: same footage with no cuts at all peaks at 1.41x — 1.8x sits
#: between the two classes.
CUT_RATIO = 1.8

#: Absolute floor on a cut's frame change — below this the picture
#: barely moved at all, whatever the ratio says.
CUT_FLOOR = 0.02

#: Seconds either side of an expected cut to look for the change.
CUT_TOL_S = 0.15

#: Seconds a measured segment duration may differ from the plan.
PACE_TOL_S = 0.15

#: Below this median frame change a segment counts as frozen — and
#: below this *maximum* change the whole video counts as featureless
#: (unmeasurable, not failed).
MOTION_FLOOR = 0.001

#: Sampling cap (frames/sec) and the width frames are downscaled to
#: for differencing. Native fps below the cap is used as-is.
SAMPLE_FPS_CAP = 30
SAMPLE_WIDTH = 160


def watch_video(path: str | Path, *, cut_points: list[float] | None = None,
                segments: list[float] | None = None,
                sample_fn: Callable[..., Any] | None = None
                ) -> dict[str, Any]:
    """Watch ``path`` over time and judge the edit against the plan.

    ``cut_points`` = expected cut times in output seconds (the
    render's cut grid); ``segments`` = planned play durations in
    output seconds. Either may be empty — a render with no cuts
    still gets its motion and pacing checked. ``sample_fn`` is the
    frame-sampling seam (tests inject a synthetic diff series);
    it must return ``(pairs, fps, duration)`` where ``pairs`` is a
    list of ``(t0, t1, diff)`` for consecutive sampled frames.

    Never raises for a bad video: an unreadable file, a failed probe
    or a failed sample yields ``qa_unavailable`` with the reason.
    """
    cuts = [float(c) for c in (cut_points or [])]
    planned = [float(d) for d in (segments or [])]
    sample = sample_fn or _sample_diffs
    try:
        pairs, fps, duration = sample(str(path))
    except Exception as exc:  # noqa: BLE001 - QA reports, never crashes
        return _unavailable(f"could not watch the video: {exc}")
    if duration <= 0 or not pairs:
        return _unavailable("no frames to watch — empty or unreadable video")
    return _decide(pairs, cuts, planned, duration, fps)


def _unavailable(notes: str) -> dict[str, Any]:
    return {"status": "qa_unavailable", "notes": notes,
            "checks": {"cuts": {"status": "unavailable", "notes": notes},
                       "pacing": {"status": "unavailable", "notes": notes},
                       "motion": {"status": "unavailable", "notes": notes}},
            "duration": 0.0, "fps": 0.0, "frames": 0,
            "backend": "ffmpeg"}


def _sample_diffs(path: str) -> tuple[list[tuple[float, float, float]],
                                      float, float]:
    """Sample the video at its native frame rate and diff consecutive
    frames. Returns ``(pairs, fps, duration)`` where each pair is
    ``(t0, t1, mean_abs_diff)`` with times in output seconds."""
    info = video_probe(path)
    duration = float(info.get("duration") or 0.0)
    fps = float(info.get("fps") or 0.0)
    if duration <= 0 or fps <= 0:
        raise MediaEditError(
            f"cannot watch {path}: no duration/fps to sample at")
    rate = min(fps, float(SAMPLE_FPS_CAP))
    with tempfile.TemporaryDirectory(prefix="watchvideo-") as tmp:
        res = extract_frames(path, interval=1.0 / rate, out_dir=tmp,
                             width=SAMPLE_WIDTH)
        files = sorted(res.get("frames") or [],
                       key=_frame_index)
        if len(files) < 2:
            raise MediaEditError(
                f"only {len(files)} frame(s) sampled from {path}")
        pairs: list[tuple[float, float, float]] = []
        for j in range(len(files) - 1):
            pairs.append((j / rate, (j + 1) / rate,
                          _mean_diff(files[j], files[j + 1])))
    return pairs, fps, duration


def _frame_index(path: str) -> int:
    """Numeric order of an extracted frame file (``frame-042.jpg``)."""
    try:
        return int(Path(path).stem.split("-")[1])
    except (IndexError, ValueError):
        return 0


def _mean_diff(path_a: str, path_b: str) -> float:
    """Mean absolute grayscale difference between two frame files,
    0..1. PIL only — no cv2, no numpy."""
    from PIL import Image, ImageChops

    with Image.open(path_a) as ima, Image.open(path_b) as imb:
        diff = ImageChops.difference(ima.convert("L"), imb.convert("L"))
        hist = diff.histogram()
        total = sum(hist)
        if total <= 0:
            return 0.0
        return sum(i * n for i, n in enumerate(hist)) / (total * 255.0)


def _decide(pairs: list[tuple[float, float, float]],
            cuts: list[float], planned: list[float],
            duration: float, fps: float) -> dict[str, Any]:
    """Judge a sampled diff series against the plan. Pure function
    of the series — no video access, so unit tests feed it synthetic
    series directly."""
    diffs = [d for _, _, d in pairs]
    median = statistics.median(diffs)
    peak = max(diffs)
    base: dict[str, Any] = {
        "duration": round(duration, 3), "fps": fps,
        "frames": len(pairs) + 1, "backend": "ffmpeg",
    }
    if peak < MOTION_FLOOR:
        notes = ("no measurable frame variation — every frame is "
                 "effectively identical, so cuts cannot be verified "
                 "from the pixels")
        verdict = _unavailable(notes)
        verdict.update(base)
        return verdict

    # -- cuts: a visible change at each expected cut time -----------
    found: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for cut in cuts:
        window = [(t0, t1, d) for (t0, t1, d) in pairs
                  if abs((t0 + t1) / 2.0 - cut) <= CUT_TOL_S]
        if not window:
            missing.append({"at": cut, "diff": 0.0, "ratio": 0.0,
                            "baseline": round(median, 4),
                            "reason": "no sampled frames near the cut"})
            continue
        t0, t1, best = max(window, key=lambda p: p[2])
        ratio = (best / median) if median > 0 else float("inf")
        entry = {"at": cut, "measured_at": round((t0 + t1) / 2.0, 3),
                 "diff": round(best, 4), "ratio": round(ratio, 2),
                 "baseline": round(median, 4)}
        if best >= max(CUT_RATIO * median, CUT_FLOOR):
            found.append(entry)
        else:
            entry["reason"] = (
                f"max change {best:.3f} is only {ratio:.2f}x the "
                f"video's typical frame change ({median:.3f})")
            missing.append(entry)
    cuts_check: dict[str, Any] = {
        "status": ("unavailable" if not cuts else
                   "pass" if not missing else "fail"),
        "expected": cuts, "found": found, "missing": missing,
        "threshold_ratio": CUT_RATIO, "baseline": round(median, 4),
    }
    if not cuts:
        cuts_check["notes"] = "no cuts in the plan to check"

    # -- pacing: measured segment durations vs the plan --------------
    pacing_check: dict[str, Any] = {"status": "unavailable",
                                    "segments": []}
    if planned:
        if missing:
            pacing_check["notes"] = (
                "cannot measure pacing — cut(s) missing: "
                + ", ".join(f"{m['at']}s" for m in missing))
        else:
            bounds = [0.0] + [f["measured_at"] for f in found] + [duration]
            segs: list[dict[str, Any]] = []
            bad = False
            for i, plan in enumerate(planned):
                measured = bounds[i + 1] - bounds[i]
                delta = measured - plan
                segs.append({"planned": round(plan, 3),
                             "measured": round(measured, 3),
                             "delta": round(delta, 3)})
                if abs(delta) > PACE_TOL_S:
                    bad = True
            pacing_check["segments"] = segs
            pacing_check["status"] = "fail" if bad else "pass"
            if bad:
                pacing_check["notes"] = "; ".join(
                    f"segment {i} is {s['measured']}s, plan "
                    f"{s['planned']}s"
                    for i, s in enumerate(segs)
                    if abs(s["delta"]) > PACE_TOL_S)
    else:
        pacing_check["notes"] = "no segment plan to check against"

    # -- motion: every planned segment must actually move ------------
    bounds = [0.0] + list(cuts) + [duration]
    motion_segs: list[dict[str, Any]] = []
    frozen: list[int] = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        inside = [d for (t0, t1, d) in pairs if t0 >= lo and t1 <= hi]
        if not inside:
            motion_segs.append({"segment": i, "median": None,
                                "notes": "too short to sample"})
            continue
        med = statistics.median(inside)
        motion_segs.append({"segment": i, "median": round(med, 4)})
        if med < MOTION_FLOOR:
            frozen.append(i)
    motion_check: dict[str, Any] = {
        "status": "fail" if frozen else "pass",
        "segments": motion_segs,
    }
    if frozen:
        motion_check["notes"] = (
            "frozen segment(s) — no frame change where a motion "
            "effect should be moving the picture: "
            + ", ".join(str(i) for i in frozen))

    # -- overall ------------------------------------------------------
    statuses = [cuts_check["status"], pacing_check["status"],
                motion_check["status"]]
    if "fail" in statuses:
        status = "fail"
    elif "pass" in statuses:
        status = "pass"
    else:
        status = "qa_unavailable"
    if status == "pass":
        notes = (f"{len(found)}/{len(cuts)} cuts on beats · pacing ok · "
                 f"motion ok" if cuts else
                 "motion ok · pacing ok (no cuts in the plan)")
    elif status == "fail":
        reasons: list[str] = []
        if missing:
            reasons.append(
                "cut(s) not found: "
                + ", ".join(f"{m['at']}s" for m in missing))
        if pacing_check["status"] == "fail":
            reasons.append(pacing_check.get("notes", "pacing off"))
        if motion_check["status"] == "fail":
            reasons.append(motion_check.get("notes", "frozen segment"))
        notes = "; ".join(reasons)
    else:
        notes = "temporal QA could not run a real check"
    return {"status": status, "notes": notes,
            "checks": {"cuts": cuts_check, "pacing": pacing_check,
                       "motion": motion_check},
            **base}
