"""Render-critique loop — score a FINISHED render, feed it back.

"dream-loop" (X post) is unverified as a port — no repo exists to port —
but the shape it names is real and was missing: nothing in the directed
organ scored a finished render and fed the score back into the next
render. ``motion_score.generate_track`` compiles NL into a track and
``validate`` checks the score/track STRUCTURALLY; ``blender_live.iterate``
loops run -> screenshot -> score, but its default scorer counts goal
words in the scene-summary TEXT and never looks at the render, and its
adjustment is a free-text note for a brain-supplied ``code_fn``.

This module closes the loop on the existing headless render path, on
``motion_score`` itself:

    render (pose video, PIL + ffmpeg) -> critique the PIXELS ->
    concrete adjustments to the motion score -> re-render

until the render scores at or above the threshold or the iteration
budget is spent. Every critique is journaled (same never-raises ring
pattern as ``directed/failures.py``, namespace
``media.directed.render_critique``).

How the critique measures a render (no track values are trusted — the
pixels are read back): the pose renderer draws every keypoint as a
white dot (radius 3) on a black background, so each joint's rendered
position is recovered by finding the white-dot centroid near where the
track says it should be. A joint with no dot in frame is clipped, not
"close enough". Four components, each 0..1:

- achievement — for every joint move in the score, how far the RENDERED
  joint travelled toward its target by the phase's end frame;
- framing — fraction of (frame, joint) pairs actually visible in the
  rendered frame (root scale / travel pushing joints off-canvas shows
  up here, never in the structural validator);
- plant — planted ankles must not drift in the RENDER (the track-level
  check in ``validate_track`` cannot see a renderer-level drift);
- smoothness — worst frame-to-frame joint step in the render.

Adjustments are mechanical, derived from the measurement, never guessed:

- a compiler clamp shortfall (``compile_score`` notes
  ``clamped {joint} travel …``) splits the phase into
  ``ceil(travel / limit)`` sub-phases, each within the travel limit, so
  the full travel is achieved on the next render;
- a framing loss pulls every root scale toward 1.0, halves root travel,
  and clamps joint targets inside the frame;
- a planted-foot drift in the render halves root travel in the drifting
  phase.

Renderers: the loop renders with the headless pose renderer (PIL +
ffmpeg) — the only render whose pixels this critique can measure. A
missing Pillow is an honest refusal naming Pillow; a missing ffmpeg
means the final mp4 is skipped (named), the PNG frames are still the
render. ``renderer="mimicmotion"`` is refused by name: MimicMotion
frames carry no skeleton overlay, so the pixel critique cannot measure
them. Blender headless rendering of a motion track goes through
``edit_engine/render3d_blender.py`` + a Scene3D document, which a
PoseTrack is not; the live-Blender loop is ``blender_live.iterate``.

Entry point (brain-called directly, the ``blender_live.iterate``
precedent — no new brain tool): :func:`critique_loop`.
"""

from __future__ import annotations

import copy
import math
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nomorals.core import failure_store as _failure_store

from .motion_score import (
    _BODY_IDX,
    compile_score,
    generate_motion_score,
    resolve_action,
)
from .pose_rig import (
    BODY_NAMES,
    N_BODY,
    PoseTrack,
    render_pose_frame,
    render_pose_video,
)
from .validate import validate_score

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

NUMPY_AVAILABLE = np is not None


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "nomorals.media.directed.render_critique is unavailable: "
            "numpy is not installed (pip install numpy)")

__all__ = [
    "RenderCritique",
    "render_pose_frames",
    "critique_render",
    "adjust_score",
    "critique_loop",
    "record_critique",
    "recent_critiques",
    "critique_count",
    "clear_critiques",
    "format_critiques",
]

#: score weights: achievement dominates, framing next, plant/smoothness guard
_W_ACHIEVE = 0.50
_W_FRAMING = 0.25
_W_PLANT = 0.15
_W_SMOOTH = 0.10

#: planted-ankle drift limit in the render (normalized units) — mirrors
#: validate_track's 0.02, measured on pixels instead of the track arrays
_PLANT_DRIFT_LIMIT = 0.02
#: frame-to-frame joint step limit in the render (normalized units)
_STEP_LIMIT = 0.25
#: white-dot search radius (px) around the expected joint position.
#: Dots are radius 3; ±1px integer rounding + antialiasing needs slack.
_DOT_RADIUS = 6
#: achievement floor: moves shorter than this (px) count as achieved when
#: the rendered end lands within it — pixel quantization, not a defect
_ACHIEVE_FLOOR_PX = 8.0
#: joint targets are pulled inside this margin when framing is lost
_FRAME_MARGIN = 0.03

_CLAMP_NOTE_RE = re.compile(
    r"clamped (\w+) travel ([\d.]+)->([\d.]+) in '([^']+)'")


# ── critique journal ─────────────────────────────────────────────────
# Same never-raises ring pattern as directed/failures.py: entries are
# recorded, never re-raised, and readable from the module. The store
# persists across restarts under NM_HOME (or ~/.nomorals).

_MAX_CRITIQUES = 200


@dataclass
class _CritiqueEntry:
    ts: float
    action: str
    pass_no: int
    score: float
    issues: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)
    met: bool = False


_journal: deque[_CritiqueEntry] = deque(maxlen=_MAX_CRITIQUES)
_journal_lock = threading.Lock()
_journal_total = 0
_journal_store = _failure_store.Store("media.directed.render_critique")
_journal_loaded = False


def _journal_ensure_loaded() -> None:
    """Load persisted entries once per process. Caller holds the lock."""
    global _journal_loaded, _journal_total
    if _journal_loaded:
        return
    _journal_loaded = True
    entries, total = _journal_store.load()
    for row in entries:
        try:
            _journal.append(_CritiqueEntry(
                ts=float(row["ts"]), action=str(row["action"]),
                pass_no=int(row["pass_no"]), score=float(row["score"]),
                issues=list(row.get("issues") or []),
                adjustments=list(row.get("adjustments") or []),
                met=bool(row.get("met", False))))
        except (KeyError, TypeError, ValueError):
            continue
    _journal_total = max(_journal_total, total, len(_journal))


def _journal_to_row(e: _CritiqueEntry) -> dict[str, Any]:
    return {"ts": e.ts, "action": e.action, "pass_no": e.pass_no,
            "score": e.score, "issues": list(e.issues),
            "adjustments": list(e.adjustments), "met": e.met}


def record_critique(action: str, pass_no: int, score: float,
                    issues: list[str], adjustments: list[str],
                    met: bool) -> None:
    """Append one pass's critique to the journal. Never raises."""
    try:
        entry = _CritiqueEntry(ts=time.time(), action=action,
                               pass_no=int(pass_no), score=float(score),
                               issues=list(issues),
                               adjustments=list(adjustments), met=bool(met))
    except Exception:  # noqa: BLE001 - journaling must never break a render
        return
    global _journal_total
    with _journal_lock:
        try:
            _journal_ensure_loaded()
            _journal.append(entry)
            _journal_total += 1
            _journal_store.save([_journal_to_row(e) for e in _journal],
                                _journal_total)
        except Exception:  # noqa: BLE001 - best-effort persistence
            return


def recent_critiques(limit: int = 20) -> list[dict[str, Any]]:
    """Newest-first slice of the critique journal."""
    with _journal_lock:
        _journal_ensure_loaded()
        items = list(_journal)
    return [_journal_to_row(e) for e in items[::-1][:max(0, limit)]]


def critique_count() -> int:
    """Total critiques ever recorded (journal length is capped)."""
    with _journal_lock:
        _journal_ensure_loaded()
        return _journal_total


def clear_critiques() -> None:
    """Empty the journal and reset the counter. Tests only."""
    global _journal_total, _journal_loaded
    with _journal_lock:
        _journal.clear()
        _journal_total = 0
        _journal_loaded = True
        _journal_store.clear()


def format_critiques(limit: int = 20) -> str:
    """One human-readable block for reading the journal back."""
    items = recent_critiques(limit)
    if not items:
        return "no render critiques recorded."
    lines = [f"🎬 render critiques ({critique_count()} total, "
             f"showing {len(items)} newest):"]
    for e in items:
        ts = time.strftime("%m-%d %H:%M", time.localtime(e["ts"]))
        lines.append(
            f"• {ts} {e['action']} pass {e['pass_no']}: "
            f"score {e['score']:.2f} {'MET' if e['met'] else 'missed'}"
            + (f" — {'; '.join(e['issues'][:2])}" if e["issues"] else ""))
    return "\n".join(lines)


# ── rendering ────────────────────────────────────────────────────────

def render_pose_frames(track: PoseTrack, out_dir: str) -> list[str]:
    """Render every frame of ``track`` to PNGs via the shared renderer.

    Uses :func:`pose_rig.render_pose_frame` — the same draw call
    ``render_pose_video`` uses — so the critique measures the real
    render, not a second drawing of it. Raises RuntimeError naming
    Pillow when the renderer is unreachable.
    """
    _require_numpy()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, kp in enumerate(track.frames):
        p = out / f"f_{i:04d}.png"
        render_pose_frame(kp, track.width, track.height).save(p)
        paths.append(str(p))
    return paths


# ── measuring the finished render ────────────────────────────────────

def _recover_joints(frame_paths: list[str], track: PoseTrack,
                    ) -> list[dict[int, tuple[float, float] | None]]:
    """Recover body-joint pixel positions from the rendered frames.

    For each frame and each body joint, the white-dot centroid inside a
    small window around the track's expected position — or None when no
    white dot is there (joint clipped off-frame, or never drawn).
    Returns per-frame {joint_idx: (x_px, y_px) | None}.
    """
    from PIL import Image

    _require_numpy()
    W, H = track.width, track.height
    n = len(frame_paths)
    recovered: list[dict[int, tuple[float, float] | None]] = []
    for i in range(n):
        arr = np.asarray(Image.open(frame_paths[i]).convert("RGB"))
        white = ((arr[:, :, 0] == 255) & (arr[:, :, 1] == 255)
                 & (arr[:, :, 2] == 255))
        frame_meas: dict[int, tuple[float, float] | None] = {}
        for j in range(N_BODY):
            ex, ey = track.frames[i, j] * np.array([W, H])
            xi, yi = int(round(ex)), int(round(ey))
            if not (0 <= xi < W and 0 <= yi < H):
                frame_meas[j] = None  # expected off-canvas: clipped
                continue
            x0, x1 = max(0, xi - _DOT_RADIUS), min(W, xi + _DOT_RADIUS + 1)
            y0, y1 = max(0, yi - _DOT_RADIUS), min(H, yi + _DOT_RADIUS + 1)
            ys, xs = np.nonzero(white[y0:y1, x0:x1])
            if len(xs) == 0:
                frame_meas[j] = None
            else:
                frame_meas[j] = (float(x0 + xs.mean()),
                                 float(y0 + ys.mean()))
        recovered.append(frame_meas)
    return recovered


def _frame_at(t: float, n_frames: int) -> int:
    return max(0, min(n_frames - 1, int(round(t * (n_frames - 1)))))


def _phase_frames(t0: float, t1: float, n_frames: int) -> tuple[int, int]:
    """(start, end) frame indices for measuring one phase in the render.

    The compiler commits a phase's effects one frame LATE (after the
    first frame with t >= t1 renders), so the frame exactly ON a phase
    boundary shows neither phase's completed state: the old phase is
    no longer active and its commit has not landed yet, while the new
    phase has barely started from the uncommitted state. Measuring on
    that frame would score a phantom defect. So: the end is the last
    frame BEFORE the boundary (phase active, easing ~1), the start is
    the first frame AFTER the previous boundary (previous phase
    committed, this phase barely started). The final phase ends on the
    last frame; the first phase starts on frame 0.
    """
    i0 = _frame_at(t0, n_frames) + (1 if t0 > 0 else 0)
    i1 = _frame_at(t1, n_frames) - (1 if t1 < 1.0 else 0)
    i0 = max(0, min(n_frames - 1, i0))
    i1 = max(i0, min(n_frames - 1, i1))
    i0 = min(i0, i1)
    return i0, i1


@dataclass
class RenderCritique:
    """The measurement of one finished render against its score."""

    score: float = 0.0
    achievement: float = 0.0
    framing: float = 0.0
    plant: float = 0.0
    smoothness: float = 0.0
    issues: list[str] = field(default_factory=list)
    measured: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues


def critique_render(frame_paths: list[str], track: PoseTrack,
                    score: dict[str, Any]) -> RenderCritique:
    """Score a FINISHED render (PNG frames) against its motion score.

    Reads pixels only — the track is used to know WHERE to look, never
    as the measurement itself (prove.py blacks out a frame and the
    score drops, which is the check that this reads the render).
    """
    _require_numpy()
    n = len(frame_paths)
    if n == 0:
        return RenderCritique(issues=["render produced no frames"])
    if track.frames.shape[0] != n:
        return RenderCritique(
            issues=[f"render has {n} frames but the track has "
                    f"{track.frames.shape[0]} — refusing to score a "
                    f"mismatched render"])
    W, H = track.width, track.height
    meas = _recover_joints(frame_paths, track)
    crit = RenderCritique()
    issues: list[str] = []

    # framing: every (frame, body joint) must show its white dot
    found = sum(1 for fm in meas for v in fm.values() if v is not None)
    total = n * N_BODY
    crit.framing = found / total if total else 0.0
    if crit.framing < 1.0:
        missing = total - found
        issues.append(
            f"framing: {missing}/{total} joint-dots missing from the "
            f"render (clipped off-frame)")

    # achievement: per joint move, rendered travel toward its target
    achs: list[float] = []
    ach_detail: dict[str, float] = {}
    for ph in score.get("phases", []) or []:
        t = ph.get("t", [0.0, 1.0])
        try:
            t0, t1 = float(t[0]), float(t[1])
        except (TypeError, ValueError, IndexError):
            continue
        i0, i1 = _phase_frames(t0, t1, n)
        for m in ph.get("moves", []) or []:
            if not isinstance(m, dict) or "joint" not in m:
                continue
            jn = m["joint"]
            if jn not in _BODY_IDX:
                continue
            to = m.get("to")
            if (not isinstance(to, (list, tuple)) or len(to) != 2
                    or not all(isinstance(v, (int, float)) for v in to)):
                continue
            j = _BODY_IDX[jn]
            start = meas[i0].get(j)
            end = meas[i1].get(j)
            target = (float(to[0]) * W, float(to[1]) * H)
            if start is None or end is None:
                achs.append(0.0)
                ach_detail[f"{ph.get('name', '?')}:{jn}"] = 0.0
                issues.append(
                    f"achievement: {jn} in phase '{ph.get('name', '?')}' "
                    f"is missing from the render at the phase "
                    f"{'start' if start is None else 'end'} — cannot "
                    f"measure its travel")
                continue
            intended = math.hypot(target[0] - start[0],
                                  target[1] - start[1])
            remaining = math.hypot(target[0] - end[0],
                                   target[1] - end[1])
            ach = max(0.0, min(1.0, 1.0 - remaining
                               / max(intended, _ACHIEVE_FLOOR_PX)))
            achs.append(ach)
            ach_detail[f"{ph.get('name', '?')}:{jn}"] = round(ach, 3)
            if ach < 0.95:
                issues.append(
                    f"achievement: {jn} in phase '{ph.get('name', '?')}' "
                    f"reached only {ach:.2f} of its target "
                    f"(rendered end ({end[0] / W:.2f},{end[1] / H:.2f}) "
                    f"vs target ({to[0]:.2f},{to[1]:.2f}))")
    crit.achievement = (sum(achs) / len(achs)) if achs else 1.0
    crit.measured["achievement_by_move"] = ach_detail

    # plant: planted ankles must not drift in the render
    plant_scores: list[float] = []
    plant_drift: dict[str, float] = {}
    ankle = {"left": BODY_NAMES.index("l_ankle"),
             "right": BODY_NAMES.index("r_ankle")}
    for ph in score.get("phases", []) or []:
        plant = ph.get("plant")
        if plant not in ("left", "right", "both"):
            continue
        t = ph.get("t", [0.0, 1.0])
        try:
            t0, t1 = float(t[0]), float(t[1])
        except (TypeError, ValueError, IndexError):
            continue
        i0, i1 = _phase_frames(t0, t1, n)
        if i1 <= i0:
            continue
        sides = ("left", "right") if plant == "both" else (plant,)
        for side in sides:
            j = ankle[side]
            pts = [(meas[i][j][0] / W, meas[i][j][1] / H)
                   for i in range(i0, i1 + 1)
                   if meas[i].get(j) is not None]
            if len(pts) < 2:
                continue
            drift = max(math.hypot(px - pts[0][0], py - pts[0][1])
                        for px, py in pts)
            plant_drift[f"{ph.get('name', '?')}:{side}"] = round(drift, 4)
            plant_scores.append(max(
                0.0, 1.0 - drift / _PLANT_DRIFT_LIMIT))
            if drift > _PLANT_DRIFT_LIMIT:
                issues.append(
                    f"plant: {side} ankle drifted {drift:.3f} in the "
                    f"render during planted phase "
                    f"'{ph.get('name', '?')}' (limit "
                    f"{_PLANT_DRIFT_LIMIT})")
    crit.plant = (sum(plant_scores) / len(plant_scores)) if plant_scores else 1.0
    crit.measured["plant_drift"] = plant_drift

    # smoothness: worst frame-to-frame joint step in the render
    worst = 0.0
    for i in range(1, n):
        for j in range(N_BODY):
            a, b = meas[i - 1].get(j), meas[i].get(j)
            if a is None or b is None:
                continue
            step = math.hypot((b[0] - a[0]) / W, (b[1] - a[1]) / H)
            worst = max(worst, step)
    crit.smoothness = max(0.0, 1.0 - worst / _STEP_LIMIT)
    crit.measured["worst_frame_step"] = round(worst, 4)
    if worst > _STEP_LIMIT:
        issues.append(
            f"smoothness: max rendered joint step {worst:.2f} between "
            f"frames (limit {_STEP_LIMIT})")

    crit.score = (_W_ACHIEVE * crit.achievement
                  + _W_FRAMING * crit.framing
                  + _W_PLANT * crit.plant
                  + _W_SMOOTH * crit.smoothness)
    crit.issues = issues
    crit.measured["frames"] = n
    return crit


# ── concrete adjustments ─────────────────────────────────────────────

def _split_clamped_phases(score: dict[str, Any], track: PoseTrack,
                          notes: list[str],
                          underachieved: set[str]) -> tuple[dict[str, Any], list[str]]:
    """Split phases whose moves the compiler clamped, so the full travel
    is achieved: a clamped move is re-timed across ceil(travel/limit)
    sub-phases, each within the limit. Returns (new_score, adjustments).

    Only clamps that ALSO shortfall in the render (``underachieved``,
    ``"phase:joint"`` keys from the critique) are split: the compiler
    commits a phase one frame late, so a clamp note can fire on the
    first frame of the NEXT phase even when the render already reaches
    the target — splitting on the note alone would split forever.
    """
    clamps: dict[str, list[tuple[str, float, float]]] = {}
    for m in _CLAMP_NOTE_RE.finditer("\n".join(notes)):
        joint, travel, limit, phase = (m.group(1), float(m.group(2)),
                                       float(m.group(3)), m.group(4))
        if f"{phase}:{joint}" not in underachieved:
            continue  # note fired, but the render still reached the target
        clamps.setdefault(phase, []).append((joint, travel, limit))
    if not clamps:
        return score, []
    new = copy.deepcopy(score)
    adjustments: list[str] = []
    phases = new.get("phases", []) or []
    out: list[dict[str, Any]] = []
    n_frames = track.frames.shape[0]
    for ph in phases:
        name = str(ph.get("name", "phase"))
        hit = clamps.get(name)
        if not hit:
            out.append(ph)
            continue
        if int(ph.get("repeat", 1) or 1) != 1:
            adjustments.append(
                f"phase '{name}': clamped moves cannot be split "
                f"(repeat={ph.get('repeat')}) — left as-is")
            out.append(ph)
            continue
        t = ph.get("t", [0.0, 1.0])
        t0, t1 = float(t[0]), float(t[1])
        i0 = _frame_at(t0, n_frames)
        n_split = 1
        waypoints: dict[str, list[tuple[float, float]]] = {}
        for joint, travel, limit in hit:
            k = max(1, math.ceil(travel / max(limit, 1e-6)))
            n_split = max(n_split, k)
            j = _BODY_IDX[joint]
            start = (float(track.frames[i0, j, 0]),
                     float(track.frames[i0, j, 1]))
            target = None
            for mv in ph.get("moves", []) or []:
                if isinstance(mv, dict) and mv.get("joint") == joint:
                    target = (float(mv["to"][0]), float(mv["to"][1]))
            if target is None:
                continue
            waypoints[joint] = [
                (start[0] + (target[0] - start[0]) * s / n_split,
                 start[1] + (target[1] - start[1]) * s / n_split)
                for s in range(1, n_split + 1)]
        if not waypoints:
            out.append(ph)
            continue
        span = (t1 - t0) / n_split
        clamped_names = ", ".join(sorted(waypoints))
        for s in range(n_split):
            sub = copy.deepcopy(ph)
            sub["t"] = [round(t0 + s * span, 4),
                        round(t0 + (s + 1) * span, 4)]
            sub["name"] = f"{name} (part {s + 1}/{n_split})"
            if s == 0:
                kept = []
                for mv in sub.get("moves", []) or []:
                    if (isinstance(mv, dict) and mv.get("joint") in waypoints):
                        jn = mv["joint"]
                        wp = waypoints[jn][0]
                        kept.append({"joint": jn, "to": [wp[0], wp[1]]})
                    elif isinstance(mv, dict) and (
                            "root" in mv
                            or mv.get("hand") in ("right", "left")
                            or "joint" in mv):
                        kept.append(mv)
                sub["moves"] = kept
            else:
                sub["moves"] = [
                    {"joint": jn, "to": [wp[s][0], wp[s][1]]}
                    for jn, wp in waypoints.items()]
            out.append(sub)
        adjustments.append(
            f"split phase '{name}' into {n_split} sub-phases so "
            f"{clamped_names} travel fits the per-phase limit "
            f"(was clamped: {', '.join(f'{j} {tr:.2f}' for j, tr, _ in hit)})")
    new["phases"] = out
    return new, adjustments


def _fix_framing(score: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Pull root scale toward 1.0, halve root travel, clamp joint
    targets inside the frame. Returns (new_score, adjustments).

    Scale keeps only a quarter of its deviation from 1.0 per pass: a
    framing loss means joints are INVISIBLE in the render, so getting
    the figure back in frame outranks preserving the dolly — the
    residual quarter keeps some of the dolly intent.
    """
    new = copy.deepcopy(score)
    adjustments: list[str] = []
    changed = False
    for ph in new.get("phases", []) or []:
        for mv in ph.get("moves", []) or []:
            if not isinstance(mv, dict):
                continue
            if "root" in mv and isinstance(mv["root"], dict):
                rm = mv["root"]
                try:
                    sc = float(rm.get("scale", 1.0))
                except (TypeError, ValueError):
                    sc = 1.0
                if abs(sc - 1.0) > 1e-9:
                    rm["scale"] = round(1.0 + (sc - 1.0) * 0.25, 4)
                    changed = True
                to = rm.get("to", [0.0, 0.0])
                if (isinstance(to, (list, tuple)) and len(to) == 2
                        and any(abs(float(v)) > 1e-9 for v in to)):
                    rm["to"] = [round(float(to[0]) * 0.5, 4),
                                round(float(to[1]) * 0.5, 4)]
                    changed = True
            if "joint" in mv and isinstance(mv.get("to"), (list, tuple)):
                to = mv["to"]
                if len(to) == 2:
                    cx = max(_FRAME_MARGIN,
                             min(1.0 - _FRAME_MARGIN, float(to[0])))
                    cy = max(_FRAME_MARGIN,
                             min(1.0 - _FRAME_MARGIN, float(to[1])))
                    if (cx, cy) != (float(to[0]), float(to[1])):
                        mv["to"] = [round(cx, 4), round(cy, 4)]
                        changed = True
    if changed:
        adjustments.append(
            "framing fix: root scale pulled three-quarters of the way "
            "toward 1.0, root travel halved, joint targets clamped "
            f"inside the frame (margin {_FRAME_MARGIN})")
    return new, adjustments


def _fix_plant_drift(score: dict[str, Any],
                     drift: dict[str, float]) -> tuple[dict[str, Any], list[str]]:
    """Halve root travel in phases whose planted foot drifted in the
    render. Returns (new_score, adjustments)."""
    if not drift:
        return score, []
    drifting = {key.split(":", 1)[0] for key, d in drift.items()
                if d > _PLANT_DRIFT_LIMIT}
    if not drifting:
        return score, []
    new = copy.deepcopy(score)
    changed = False
    for ph in new.get("phases", []) or []:
        if str(ph.get("name", "")) not in drifting:
            continue
        for mv in ph.get("moves", []) or []:
            if (isinstance(mv, dict) and "root" in mv
                    and isinstance(mv["root"], dict)):
                to = mv["root"].get("to", [0.0, 0.0])
                if isinstance(to, (list, tuple)) and len(to) == 2:
                    mv["root"]["to"] = [round(float(to[0]) * 0.5, 4),
                                        round(float(to[1]) * 0.5, 4)]
                    changed = True
    if not changed:
        return score, []
    return new, [f"plant fix: root travel halved in {sorted(drifting)} "
                 f"(planted foot drifted past {_PLANT_DRIFT_LIMIT} in "
                 f"the render)"]


def adjust_score(score: dict[str, Any], critique: RenderCritique,
                 track: PoseTrack, notes: list[str] | None = None,
                 ) -> tuple[dict[str, Any], list[str]]:
    """Derive concrete adjustments from a render critique.

    Returns (new_score, adjustments). When nothing mechanical applies,
    the score comes back unchanged and ``adjustments`` says so — the
    loop then stops instead of burning budget on identical re-renders.
    """
    new = copy.deepcopy(score)
    adjustments: list[str] = []
    if critique.plant < 1.0:
        new, adj = _fix_plant_drift(
            new, critique.measured.get("plant_drift", {}))
        adjustments.extend(adj)
    if critique.framing < 1.0:
        new, adj = _fix_framing(new)
        adjustments.extend(adj)
    if critique.achievement < 0.95:
        under = {key for key, ach
                 in critique.measured.get("achievement_by_move", {}).items()
                 if ach < 0.95}
        new, adj = _split_clamped_phases(new, track, notes or [], under)
        adjustments.extend(adj)
    if new == score and not adjustments:
        adjustments = ["no mechanical adjustment applies to this "
                       "critique — the render already matches what the "
                       "score asks for within measurement noise"]
    return new, adjustments


# ── the loop ─────────────────────────────────────────────────────────

def critique_loop(*, score: dict[str, Any] | None = None,
                  description: str | None = None,
                  suggest: Any | None = None,
                  track: PoseTrack | None = None,
                  threshold: float = 0.8,
                  max_iterations: int = 3,
                  n_frames: int = 24,
                  fps: float = 10.0,
                  width: int = 512,
                  height: int = 512,
                  workdir: str | None = None,
                  renderer: str = "pose") -> dict[str, Any]:
    """Render -> critique -> adjust -> re-render, until the render
    passes or the budget is spent.

    Give a motion ``score`` dict directly, or a ``description`` plus a
    ``suggest`` model fn (the LLM motion-score path). Preset actions
    have no score to adjust, so a preset ``description`` is refused by
    name. ``track`` may carry the first pass's track; later passes
    always recompile the adjusted score. Returns a result dict with
    ``ok`` / ``met`` / ``score`` / ``passes`` / ``video`` /
    ``final_score``. Never raises for a render/critique failure — the
    failure is the returned ``{"ok": False, "error": ...}``.
    """
    if renderer != "pose":
        if renderer == "mimicmotion":
            return {"ok": False,
                    "error": ("renderer 'mimicmotion' refused: MimicMotion "
                              "frames carry no skeleton overlay, so the "
                              "pixel critique cannot measure them — use "
                              "renderer='pose' (headless PIL+ffmpeg "
                              "skeleton render)")}
        return {"ok": False,
                "error": (f"unknown renderer {renderer!r} — the "
                          "render-critique loop renders with 'pose' "
                          "(headless PIL+ffmpeg skeleton render)")}
    if score is None and description is None:
        return {"ok": False,
                "error": "critique_loop needs a motion score dict or a "
                         "description (with a model to direct it)"}
    if score is None:
        desc = (description or "").strip()
        if not desc:
            return {"ok": False, "error": "empty action description"}
        if resolve_action(desc) is not None:
            return {"ok": False,
                    "error": (f"preset action {desc!r} has no motion score "
                              "to critique — pass a score dict, or a "
                              "non-preset description with a model")}
        if suggest is None:
            return {"ok": False,
                    "error": (f"no model is connected to direct {desc!r} "
                              "into a motion score — connect a model or "
                              "pass a score dict")}
        try:
            score = generate_motion_score(desc, suggest)
        except (ValueError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc)}
    action = str(score.get("action", description or ""))
    wd = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="render_critique_"))
    wd.mkdir(parents=True, exist_ok=True)
    threshold = max(0.0, min(1.0, float(threshold)))
    limit = max(1, int(max_iterations))
    passes: list[dict[str, Any]] = []
    final_track = track
    final_score_dict = copy.deepcopy(score)
    final_score = 0.0
    met = False
    notes_out: list[str] = []
    try:
        for number in range(1, limit + 1):
            rep = validate_score(final_score_dict)
            if not rep.ok:
                return {"ok": False,
                        "error": ("motion score failed structural "
                                  "validation: " + "; ".join(rep.errors)),
                        "passes": passes}
            if final_track is None or number > 1:
                try:
                    final_track, cnotes = compile_score(
                        final_score_dict, n_frames=n_frames, fps=fps)
                except (ValueError, RuntimeError) as exc:
                    return {"ok": False, "error": str(exc),
                            "passes": passes}
                final_track.width, final_track.height = width, height
                notes_out = list(cnotes)
            else:
                notes_out = []
            pass_dir = wd / f"pass{number}"
            try:
                frame_paths = render_pose_frames(final_track, str(pass_dir))
            except RuntimeError as exc:
                return {"ok": False, "error": str(exc), "passes": passes}
            crit = critique_render(frame_paths, final_track,
                                   final_score_dict)
            final_score = crit.score
            new_score, adjustments = adjust_score(
                final_score_dict, crit, final_track, notes_out)
            met = crit.score >= threshold
            record_critique(action, number, crit.score, crit.issues,
                            adjustments, met)
            passes.append({
                "pass": number,
                "score": round(crit.score, 4),
                "components": {
                    "achievement": round(crit.achievement, 4),
                    "framing": round(crit.framing, 4),
                    "plant": round(crit.plant, 4),
                    "smoothness": round(crit.smoothness, 4),
                },
                "issues": list(crit.issues),
                "adjustments": list(adjustments),
                "frames_dir": str(pass_dir),
                "n_frames": len(frame_paths),
            })
            if met:
                break
            if new_score == final_score_dict:
                break  # no mechanical adjustment: re-rendering is identical
            final_score_dict = new_score
            final_track = None  # recompile the adjusted score next pass
    except Exception as exc:  # noqa: BLE001 - the loop reports, never raises
        return {"ok": False, "error": f"render-critique loop failed: {exc}",
                "passes": passes}
    video: str | None = None
    if final_track is not None and shutil.which("ffmpeg"):
        try:
            video = render_pose_video(final_track, str(wd / "final.mp4"))
        except Exception as exc:  # noqa: BLE001 - video is packaging, not the render
            notes_out.append(f"final video not encoded: {exc}")
    elif final_track is not None:
        notes_out.append(
            "ffmpeg not found — final video not encoded; the PNG frames "
            f"in {passes[-1]['frames_dir'] if passes else wd} are the render")
    return {
        "ok": True,
        "goal": action,
        "met": met,
        "score": round(final_score, 4),
        "passes": passes,
        "video": video,
        "final_score": final_score_dict,
        "notes": notes_out,
    }
