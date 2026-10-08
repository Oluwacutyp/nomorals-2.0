"""Seer as form coach — video movement analysis (build-map #86).

"Devon's vision does this natively."

- :func:`analyze_form` — a video of a lift → Seer frame descriptions →
  per-checkpoint PASS/FLAG → specific corrections.
  "Knees caving in on rep 3 — push them out over your toes."
- 5 foundational movements: squat, deadlift, push-up, plank, lunge.
- :func:`analyze_gait` — Ochy pattern: running gait from phone video →
  risk flags (overstriding, knee collapse, asymmetry...).
- #85 integration: :func:`augment_today_workout` prepends mobility work
  for flagged checkpoints — "your squat depth is limited, adding
  mobility work to today's session."

Phase 1 is qualitative (structured Seer prompts, honest bands).
Phase 2 hooks (:data:`ANGLE_HOOKS`) are where quantitative joint-angle
estimation plugs in when a pose model is available.

The vision seam is injectable — ``vision(image_path, question) -> str``.
Default lazily uses :class:`~nomorals.vision.seer.Seer`. No Seer, no
ffmpeg, no frames → an honest "can't analyze this" — never fabricated
feedback, never raises.

CRITICAL POSITIONING: coaching, NOT medical advice. Every public
string passes :func:`guard_coaching` (the #53 diagnostic-phrase ban)
and carries the "stop if it hurts" line. Owner-scoped only.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .coach import guard_coaching
from .training import (
    Exercise,
    TrainingCoach,
    Workout,
    TRAINING_DISCLAIMER,
)

_log = logging.getLogger(__name__)

__all__ = [
    "MOVEMENTS",
    "CHECKPOINTS",
    "FIXES",
    "GAIT_RISKS",
    "FormIssue",
    "FormAnalysis",
    "GaitIssue",
    "GaitAnalysis",
    "FormStore",
    "analyze_form",
    "analyze_gait",
    "mobility_for",
    "augment_today_workout",
    "control_form",
    "FORM_DISCLAIMER",
]

FORM_DISCLAIMER = (
    "form coaching, not medical advice: this is movement feedback from "
    "video only — it can't assess health or injuries. " + TRAINING_DISCLAIMER)

#: Frames extracted per video (cheap + enough for qualitative form).
_MAX_FRAMES = 3

#: Image suffixes usable directly as single "frames".
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

#: Phase-2 hook points for quantitative pose estimation (models TBD).
ANGLE_HOOKS = ("knee_angle", "hip_angle", "spine_angle", "ankle_angle")


# ── movement references ──────────────────────────────────────────────────
# checkpoint id → (short label, what good looks like)

CHECKPOINTS: dict[str, dict[str, tuple[str, str]]] = {
    "squat": {
        "knees": ("knees", "knees track over the toes, no inward caving"),
        "depth": ("depth", "hips reach at least parallel with the knees"),
        "spine": ("spine", "back stays neutral, no lower-back rounding"),
        "feet": ("feet", "whole foot planted, weight through mid-foot"),
    },
    "deadlift": {
        "spine": ("spine", "neutral spine head to tailbone, no rounding"),
        "hips": ("hips", "hips hinge back, bar stays close to the shins"),
        "lockout": ("lockout", "full hip extension at the top, no lean-back"),
        "setup": ("setup", "shoulders over the bar at the start"),
    },
    "push-up": {
        "body": ("body", "straight line head to heels, no sag or pike"),
        "elbows": ("elbows", "elbows around 45 degrees, not flared wide"),
        "depth": ("depth", "chest comes close to the floor each rep"),
        "head": ("head", "neck neutral, gaze slightly forward"),
    },
    "plank": {
        "body": ("body", "straight line from head to heels"),
        "hips": ("hips", "hips level — neither sagging nor raised"),
        "core": ("core", "glutes and abs visibly braced"),
    },
    "lunge": {
        "front_knee": ("front knee", "front knee tracks over the ankle"),
        "torso": ("torso", "torso upright, core braced"),
        "stride": ("stride", "long enough stride to stay balanced"),
        "back_knee": ("back knee", "back knee drops toward the floor"),
    },
}

MOVEMENTS = tuple(CHECKPOINTS)

# checkpoint id → concrete fix. Keys: (exercise, checkpoint).
FIXES: dict[tuple[str, str], str] = {
    ("squat", "knees"): ("push your knees out over your toes — think 'spread "
                         "the floor apart'. A light band around the knees "
                         "gives instant feedback."),
    ("squat", "depth"): ("work ankle and hip mobility; squat to a box at "
                         "parallel until depth feels comfortable."),
    ("squat", "spine"): ("brace before each rep — big breath, ribs down. "
                         "Drop the weight until you can keep a neutral "
                         "spine."),
    ("squat", "feet"): ("keep the whole foot planted; if heels lift, "
                        "elevate them slightly and work ankle mobility."),
    ("deadlift", "spine"): ("stop the rep range where your back stays flat. "
                            "Think 'chest up, lats tight' before you pull."),
    ("deadlift", "hips"): ("push your hips back like closing a car door "
                           "with your backside; drag the bar up your shins."),
    ("deadlift", "lockout"): ("squeeze the glutes to stand tall — don't lean "
                              "back past vertical."),
    ("deadlift", "setup"): ("set the bar over mid-foot and get your "
                            "shoulders just in front of it before pulling."),
    ("push-up", "body"): ("squeeze glutes and brace abs like a moving "
                          "plank; elevate hands until the line holds."),
    ("push-up", "elbows"): ("tuck elbows to about 45 degrees — 'armpits to "
                            "hips', not flared wide."),
    ("push-up", "depth"): ("go chest-to-floor; elevate the surface until "
                           "full depth is clean."),
    ("push-up", "head"): ("tuck the chin slightly — make a double chin. "
                          "Eyes a fist-length ahead of your hands."),
    ("plank", "body"): ("squeeze glutes, press the floor away, long spine "
                        "head to heels."),
    ("plank", "hips"): ("tuck the pelvis slightly; drop to knees if the "
                        "hips sag before the timer ends."),
    ("plank", "core"): ("breathe steadily behind a braced belly — if you "
                        "can't breathe, the brace is too much."),
    ("lunge", "front_knee"): ("step longer and think 'knee over ankle' — "
                              "shorten the stride until the knee tracks."),
    ("lunge", "torso"): ("stack ribs over hips; if you lean forward, "
                         "shorten the stride."),
    ("lunge", "stride"): ("take a longer step — the front shin should be "
                          "near vertical at the bottom."),
    ("lunge", "back_knee"): ("drop straight down and lightly tap the back "
                             "knee toward the floor each rep."),
}

# gait risks (Ochy pattern): risk id → (label, what it looks like, fix).
GAIT_RISKS: dict[str, tuple[str, str, str]] = {
    "overstride": (
        "overstriding",
        "foot lands well ahead of the hips",
        "shorten the stride — aim to land with the foot under the hips, "
        "cadence near 170-180 steps/min."),
    "heel_strike": (
        "heavy heel strike",
        "hard heel landing with a locked knee",
        "land softer with a slightly bent knee; think 'quiet feet'."),
    "knee_collapse": (
        "knee collapse",
        "knee caves inward during stance",
        "strengthen glutes and hip abductors; check shoe support."),
    "bounce": (
        "excessive bounce",
        "lots of vertical movement each stride",
        "run 'quieter' — less up-and-down, more forward glide."),
    "asymmetry": (
        "asymmetry",
        "left and right sides look visibly different",
        "note which side differs and for how long — persistent asymmetry "
        "deserves a physio's eyes."),
}

_GENERIC_MOBILITY = Exercise(
    key="mobility_flow", name="Mobility flow", sets=1, reps="5 min",
    rest_secs=0, intensity="easy",
    notes="easy joint circles head to ankles — form-coach addition.")

#: checkpoint → mobility exercise. (exercise, checkpoint) keys; fallback:
#: (_GENERIC_MOBILITY) for anything unmapped.
MOBILITY_MAP: dict[tuple[str, str], Exercise] = {
    ("squat", "depth"): Exercise(
        key="deep_squat_hold", name="Deep squat hold", sets=2, reps="30s",
        rest_secs=30, intensity="easy",
        notes="hold the bottom of a squat, elbows pushing knees out."),
    ("squat", "knees"): Exercise(
        key="banded_lateral_walk", name="Banded lateral walk", sets=2,
        reps="10/side", rest_secs=30, intensity="easy",
        notes="mini-band above knees, stay low, don't let knees cave."),
    ("squat", "spine"): Exercise(
        key="cat_cow", name="Cat-cow", sets=1, reps="10",
        rest_secs=0, intensity="easy",
        notes="slow spinal waves, breathe with each segment."),
    ("squat", "feet"): Exercise(
        key="calf_stretch", name="Wall calf stretch", sets=2, reps="30s/side",
        rest_secs=15, intensity="easy",
        notes="knee over toes, heel down — ankle mobility."),
    ("deadlift", "spine"): Exercise(
        key="hip_hinge_drill", name="Hip hinge drill", sets=2, reps="10",
        rest_secs=30, intensity="easy",
        notes="dowel along spine touching head, back, tailbone."),
    ("deadlift", "hips"): Exercise(
        key="glute_bridge", name="Glute bridge", sets=2, reps="12",
        rest_secs=30, intensity="easy",
        notes="squeeze glutes hard at the top."),
    ("push-up", "body"): Exercise(
        key="plank_hold", name="Plank hold", sets=2, reps="30s",
        rest_secs=30, intensity="easy",
        notes="straight line head to heels."),
    ("push-up", "elbows"): Exercise(
        key="scap_pushup", name="Scapular push-up", sets=2, reps="10",
        rest_secs=30, intensity="easy",
        notes="straight arms, pinch and spread shoulder blades."),
    ("plank", "hips"): Exercise(
        key="dead_bug", name="Dead bug", sets=2, reps="8/side",
        rest_secs=30, intensity="easy",
        notes="low back pressed to floor the whole time."),
    ("lunge", "front_knee"): Exercise(
        key="split_squat_iso", name="Split squat hold", sets=2,
        reps="20s/side", rest_secs=30, intensity="easy",
        notes="front knee over ankle, upright torso."),
    ("lunge", "torso"): Exercise(
        key="wall_slide", name="Wall slide", sets=2, reps="10",
        rest_secs=30, intensity="easy",
        notes="back to wall, ribs down, arms overhead."),
}


# ── data ─────────────────────────────────────────────────────────────────


@dataclass
class FormIssue:
    """One flagged checkpoint: what was seen + the fix."""
    checkpoint: str          # "knees"
    label: str               # "knees"
    observed: str            # Seer's words
    fix: str                 # from FIXES


@dataclass
class FormAnalysis:
    """Qualitative form read of a video. Never raises."""
    id: str
    exercise: str
    available: bool          # False → vision couldn't run; read .note
    score: float             # passed / total checkpoints (0 when unavailable)
    band: str                # solid | good | needs work | major issues | n/a
    issues: list[FormIssue] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)   # checkpoint labels
    frames_used: int = 0
    note: str = ""           # honest limitation / failure reason
    created_at: float = 0.0

    def format(self) -> str:
        if not self.available:
            return guard_coaching(
                f"🎥 {self.exercise} — couldn't analyze this video.\n"
                f"{self.note}\n\n{FORM_DISCLAIMER}")
        emoji = {"solid": "✅", "good": "👍", "needs work": "🛠️",
                 "major issues": "🚧"}.get(self.band, "🎥")
        lines = [f"{emoji} {self.exercise} — {self.band} "
                 f"({len(self.passes)}/{len(self.passes) + len(self.issues)} "
                 f"checkpoints)"]
        if self.passes:
            lines.append("passing: " + ", ".join(self.passes))
        for i in self.issues:
            lines.append(f"\n⚠️ {i.label}: {i.observed}\n   fix: {i.fix}")
        if self.note:
            lines.append(f"\nnote: {self.note}")
        lines.append(f"\n{FORM_DISCLAIMER}")
        return guard_coaching("\n".join(lines))


@dataclass
class GaitIssue:
    risk: str        # "overstride"
    label: str       # "overstriding"
    observed: str
    fix: str


@dataclass
class GaitAnalysis:
    """Running-gait risk flags from a phone video. Never raises."""
    id: str
    available: bool
    issues: list[GaitIssue] = field(default_factory=list)
    frames_used: int = 0
    note: str = ""
    created_at: float = 0.0

    def format(self) -> str:
        if not self.available:
            return guard_coaching(
                f"🏃 gait — couldn't analyze this video.\n{self.note}\n\n"
                f"{FORM_DISCLAIMER}")
        lines = ["🏃 gait analysis"]
        if not self.issues:
            lines.append("no risk flags spotted — smooth and symmetrical.")
        for i in self.issues:
            lines.append(f"\n⚠️ {i.label}: {i.observed}\n   try: {i.fix}")
        lines.append(f"\nnote: {self.note}"
                     if self.note else "\nnote: side/rear phone video, "
                     "steady pace, gives the best read.")
        lines.append(f"\n{FORM_DISCLAIMER}")
        return guard_coaching("\n".join(lines))


# ── vision plumbing ──────────────────────────────────────────────────────

VisionFn = Callable[[str, str], str]


def _default_vision() -> VisionFn:
    """Lazy Seer — heavy vision deps never import at module load."""
    from ..vision.seer import get_seer
    seer = get_seer()
    return seer.see


def _frames_for(video_path: str, max_frames: int = _MAX_FRAMES
                ) -> tuple[list[str] | None, str]:
    """Video → up-to-N frame image paths. Never raises.

    Returns (frames, "") or (None, honest-reason).
    """
    try:
        p = Path(video_path or "")
        if not p.is_file():
            return None, f"no video found at '{video_path}'."
        if p.suffix.lower() in _IMAGE_SUFFIXES:
            return [str(p)], ""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return (None, "can't pull frames from video — ffmpeg isn't "
                          "available here. Send a photo/frame instead.")
        tmpdir = tempfile.mkdtemp(prefix="form_frames_")
        pattern = os.path.join(tmpdir, "frame_%02d.jpg")
        try:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(p),
                 "-vf", "fps=1", "-frames:v", str(max_frames), pattern],
                timeout=60, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            return None, "frame extraction failed for this video."
        frames = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
        if not frames:
            return None, "couldn't extract any frames from this video."
        return frames, ""
    except Exception as exc:  # noqa: BLE001
        _log.warning("frame extraction failed: %s", exc)
        return None, "frame extraction failed for this video."


def _checkpoint_prompt(exercise: str) -> str:
    cps = CHECKPOINTS[exercise]
    lines = [
        f"You are describing video frames of someone doing a {exercise}.",
        "Be a precise visual sensor: describe body positions factually.",
        "For EACH checkpoint below, reply with exactly one line:",
        "<name>: PASS — <short reason>   OR   <name>: FLAG — <what you see>",
        "Checkpoints:"]
    for cid, (label, desc) in cps.items():
        lines.append(f"- {cid} ({label}): {desc}")
    lines.append("Reply with one line per checkpoint, nothing else.")
    return "\n".join(lines)


def _parse_checkpoint_lines(text: str, exercise: str
                            ) -> tuple[list[str], list[tuple[str, str]]]:
    """Seer text → (passed checkpoint ids, [(id, observed)] flagged)."""
    cps = CHECKPOINTS[exercise]
    passes: list[str] = []
    flags: list[tuple[str, str]] = []
    try:
        for raw in (text or "").splitlines():
            line = raw.strip()
            if ":" not in line:
                continue
            name, _, rest = line.partition(":")
            cid = name.strip().lower().replace(" ", "_")
            if cid not in cps:
                continue
            rest_u = rest.strip().upper()
            if rest_u.startswith("FLAG"):
                observed = rest.strip()[4:].lstrip("—- ").strip() or "flagged"
                flags.append((cid, observed))
            elif rest_u.startswith("PASS"):
                passes.append(cid)
        return passes, flags
    except Exception:  # noqa: BLE001
        return [], []


def _score_band(score: float) -> str:
    if score >= 0.99:
        return "solid"
    if score >= 0.7:
        return "good"
    if score >= 0.4:
        return "needs work"
    return "major issues"


# ── public API ───────────────────────────────────────────────────────────


def analyze_form(video_path: str, exercise: str, *,
                 vision: VisionFn | None = None,
                 store: "FormStore | None" = None) -> FormAnalysis:
    """Analyze a lift video → qualitative form read. Never raises.

    ``vision(image_path, question) -> str`` is the injectable Seer seam.
    """
    aid = "form_" + uuid.uuid4().hex[:8]
    try:
        exercise = (exercise or "").lower().strip().replace(" ", "_")
        if exercise not in CHECKPOINTS:
            return FormAnalysis(
                id=aid, exercise=exercise or "?", available=False, score=0.0,
                band="n/a",
                note=("unknown movement — I coach: "
                      f"{', '.join(MOVEMENTS)}."),
                created_at=time.time())
        vision = vision or _default_vision()
        frames, reason = _frames_for(video_path)
        if frames is None:
            return FormAnalysis(
                id=aid, exercise=exercise, available=False, score=0.0,
                band="n/a", note=reason, created_at=time.time())

        passes: list[str] = []
        flagged: dict[str, str] = {}
        for frame in frames:
            try:
                text = vision(frame, _checkpoint_prompt(exercise))
            except Exception as exc:  # noqa: BLE001
                _log.warning("vision failed on frame: %s", exc)
                continue
            fp, ff = _parse_checkpoint_lines(text, exercise)
            for cid in fp:
                if cid not in flagged and cid not in passes:
                    passes.append(cid)
            for cid, observed in ff:
                if cid not in flagged:
                    flagged[cid] = observed
                if cid in passes:
                    passes.remove(cid)

        if not passes and not flagged:
            return FormAnalysis(
                id=aid, exercise=exercise, available=False, score=0.0,
                band="n/a", frames_used=len(frames),
                note=("the vision model didn't return usable checkpoint "
                      "reads — try a clearer, well-lit, side-on video."),
                created_at=time.time())

        total = len(CHECKPOINTS[exercise])
        issues = [
            FormIssue(checkpoint=cid,
                      label=CHECKPOINTS[exercise][cid][0],
                      observed=observed,
                      fix=FIXES.get((exercise, cid), "slow down and "
                                    "re-check this position."))
            for cid, observed in flagged.items()]
        score = len(passes) / total if total else 0.0
        analysis = FormAnalysis(
            id=aid, exercise=exercise, available=True, score=score,
            band=_score_band(score), issues=issues,
            passes=[CHECKPOINTS[exercise][c][0] for c in passes],
            frames_used=len(frames),
            note=("qualitative read from video frames — not measured "
                  "joint angles."),
            created_at=time.time())
        if store is not None:
            try:
                store.save(analysis)
            except Exception:  # noqa: BLE001
                _log.warning("form store save failed", exc_info=True)
        return analysis
    except Exception as exc:  # noqa: BLE001
        _log.warning("analyze_form failed: %s", exc)
        return FormAnalysis(
            id=aid, exercise=exercise or "?", available=False, score=0.0,
            band="n/a", note="analysis failed unexpectedly.",
            created_at=time.time())


def _gait_prompt() -> str:
    lines = [
        "You are describing video frames of someone running.",
        "Be a precise visual sensor: describe body positions factually.",
        "For EACH risk below, reply with exactly one line:",
        "<name>: PASS — <short reason>   OR   <name>: FLAG — <what you see>",
        "Risks:"]
    for rid, (label, desc, _fix) in GAIT_RISKS.items():
        lines.append(f"- {rid} ({label}): {desc}")
    lines.append("Reply with one line per risk, nothing else.")
    return "\n".join(lines)


def analyze_gait(video_path: str, *,
                 vision: VisionFn | None = None) -> GaitAnalysis:
    """Running-gait risk flags from a phone video. Never raises."""
    gid = "gait_" + uuid.uuid4().hex[:8]
    try:
        vision = vision or _default_vision()
        frames, reason = _frames_for(video_path)
        if frames is None:
            return GaitAnalysis(id=gid, available=False, note=reason,
                                created_at=time.time())
        flagged: dict[str, str] = {}
        for frame in frames:
            try:
                text = vision(frame, _gait_prompt())
            except Exception as exc:  # noqa: BLE001
                _log.warning("vision failed on gait frame: %s", exc)
                continue
            for raw in (text or "").splitlines():
                line = raw.strip()
                if ":" not in line:
                    continue
                name, _, rest = line.partition(":")
                rid = name.strip().lower().replace(" ", "_")
                if rid not in GAIT_RISKS:
                    continue
                if rest.strip().upper().startswith("FLAG") and rid not in flagged:
                    observed = rest.strip()[4:].lstrip("—- ").strip() or "flagged"
                    flagged[rid] = observed
        issues = [
            GaitIssue(risk=rid, label=GAIT_RISKS[rid][0],
                      observed=observed, fix=GAIT_RISKS[rid][2])
            for rid, observed in flagged.items()]
        return GaitAnalysis(
            id=gid, available=True, issues=issues, frames_used=len(frames),
            note=("qualitative read — persistent issues deserve a physio's "
                  "eyes, not just an app's."),
            created_at=time.time())
    except Exception as exc:  # noqa: BLE001
        _log.warning("analyze_gait failed: %s", exc)
        return GaitAnalysis(id=gid, available=False,
                            note="gait analysis failed unexpectedly.",
                            created_at=time.time())


def mobility_for(analysis: FormAnalysis) -> list[Exercise]:
    """Flagged checkpoints → mobility exercises. Pure; never raises."""
    try:
        out: list[Exercise] = []
        seen: set[str] = set()
        for issue in analysis.issues or []:
            ex = MOBILITY_MAP.get((analysis.exercise, issue.checkpoint))
            if ex is None:
                ex = _GENERIC_MOBILITY
            if ex.key not in seen:
                seen.add(ex.key)
                out.append(ex)
        return out
    except Exception:  # noqa: BLE001
        return []


def augment_today_workout(coach: TrainingCoach,
                          analysis: FormAnalysis) -> Workout:
    """#85 integration: prepend mobility work for form issues.

    "Your squat depth is limited — adding mobility work to today's
    session." Returns today's (readiness-gated) workout, augmented.
    Never raises.
    """
    try:
        workout = coach.today(None)
    except Exception:  # noqa: BLE001
        workout = None
    if workout is None:
        workout = Workout(date="", kind="general", exercises=[])
    try:
        mob = mobility_for(analysis) if analysis.available else []
        if mob and workout is not None:
            workout.exercises = list(mob) + list(workout.exercises)
            detail = ", ".join(i.checkpoint for i in analysis.issues[:3])
            extra = (f"form coach: added {len(mob)} mobility movement(s) "
                     f"for your {analysis.exercise} ({detail}).")
            workout.gate_note = ((workout.gate_note + " ") if workout.gate_note
                                 else "") + extra
            workout.gated = True
        return workout
    except Exception:  # noqa: BLE001
        _log.warning("augment_today_workout failed", exc_info=True)
        try:
            return coach.today(None)
        except Exception:  # noqa: BLE001
            return Workout(date="", kind="general", exercises=[])


# ── storage ──────────────────────────────────────────────────────────────


def _store_path(path: str = "") -> str:
    if path:
        return path
    base = os.environ.get("NOMORALS_HOME", os.path.expanduser("~/.nomorals"))
    return os.path.join(base, "health", "form.db")


class FormStore:
    """SQLite store for form analyses. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            p = _store_path(db_path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            self._db = sqlite3.connect(p)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS form_analyses (
                       id TEXT PRIMARY KEY, exercise TEXT,
                       analysis_json TEXT, created_at REAL)""")
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.warning("form store unavailable, running empty",
                         exc_info=True)
            self._db = None

    def save(self, analysis: FormAnalysis) -> bool:
        try:
            if self._db is None:
                return False
            self._db.execute(
                "INSERT OR REPLACE INTO form_analyses VALUES (?,?,?,?)",
                (analysis.id, analysis.exercise,
                 json.dumps(asdict(analysis)),
                 analysis.created_at))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def get(self, analysis_id: str) -> FormAnalysis | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT analysis_json FROM form_analyses WHERE id = ?",
                (analysis_id,)).fetchone()
            if row is None:
                return None
            return _analysis_from_dict(json.loads(
                row["analysis_json"]))
        except Exception:  # noqa: BLE001
            return None

    def latest(self, exercise: str = "") -> FormAnalysis | None:
        try:
            if self._db is None:
                return None
            q = ("SELECT analysis_json FROM form_analyses ORDER BY "
                 "created_at DESC LIMIT 1")
            args: tuple = ()
            if exercise:
                q = ("SELECT analysis_json FROM form_analyses WHERE "
                     "exercise = ? ORDER BY created_at DESC LIMIT 1")
                args = (exercise,)
            row = self._db.execute(q, args).fetchone()
            if row is None:
                return None
            return _analysis_from_dict(json.loads(
                row["analysis_json"]))
        except Exception:  # noqa: BLE001
            return None


def _analysis_from_dict(d: dict) -> FormAnalysis:
    issues = [FormIssue(**i) for i in (d.get("issues") or [])]
    return FormAnalysis(
        id=str(d.get("id", "")), exercise=str(d.get("exercise", "?")),
        available=bool(d.get("available")), score=float(d.get("score", 0.0)),
        band=str(d.get("band", "n/a")), issues=issues,
        passes=list(d.get("passes") or []),
        frames_used=int(d.get("frames_used", 0)),
        note=str(d.get("note", "")),
        created_at=float(d.get("created_at", 0.0)))


# ── chat ─────────────────────────────────────────────────────────────────

_USAGE = (
    "🎥 form coach — Seer-powered movement analysis.\n"
    "/form analyze <video-or-photo> <exercise> — form read + fixes\n"
    "/form gait <video> — running gait risk flags\n"
    "/form movements — the 5 coached movements\n"
    "/form apply <analysis-id> — add mobility work to today's session\n"
    "coached: squat, deadlift, push-up, plank, lunge")


def control_form(tail: str, *,
                 coach: "TrainingCoach | None" = None,
                 vision: VisionFn | None = None,
                 store: "FormStore | None" = None) -> str:
    """Chat entry: /form. Owner-only at the dispatch layer. Never raises."""
    try:
        raw = (tail or "").strip()
        if not raw or raw.split()[0] in ("help", "?"):
            return _USAGE
        verb, _, rest = raw.partition(" ")
        verb = verb.lower()
        store = store or FormStore()

        if verb == "movements":
            return ("coached movements: " + ", ".join(MOVEMENTS) +
                    "\nplus /form gait <video> for running gait.")

        if verb == "analyze":
            parts = rest.split()
            if len(parts) < 2:
                return ("usage: /form analyze <video-or-photo> <exercise>\n"
                        f"exercises: {', '.join(MOVEMENTS)}")
            path, exercise = parts[0], " ".join(parts[1:])
            analysis = analyze_form(path, exercise, vision=vision,
                                    store=store)
            head = f"analysis id: {analysis.id}\n"
            return head + analysis.format()

        if verb == "gait":
            path = rest.strip()
            if not path:
                return "usage: /form gait <video>"
            return analyze_gait(path, vision=vision).format()

        if verb == "apply":
            analysis_id = rest.strip()
            if not analysis_id:
                return "usage: /form apply <analysis-id>"
            analysis = store.get(analysis_id)
            if analysis is None:
                return f"no analysis '{analysis_id}' saved."
            if not analysis.available or not analysis.issues:
                return ("form looks solid — nothing to add to today's "
                        "session.")
            tc = coach or TrainingCoach()
            workout = augment_today_workout(tc, analysis)
            names = [e.name for e in mobility_for(analysis)]
            lines = [
                f"✅ applied to today's session ({workout.kind}):",
                "added: " + ", ".join(names),
                "",
                FORM_DISCLAIMER]
            return guard_coaching("\n".join(lines))

        return _USAGE
    except Exception as exc:  # noqa: BLE001
        _log.warning("control_form failed: %s", exc)
        return "form coach hit a snag — try again."
