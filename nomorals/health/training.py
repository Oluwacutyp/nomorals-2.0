"""Recovery-gated training plans + voice coaching (build-map #85).

"Biometrics gate the plan; voice drives retention."

- Text-to-structured-plan: "12-week strength block" → a periodized
  :class:`Plan` (base → build → peak → deload), generated in seconds.
- Readiness gating: the workout you get *today* depends on recovery
  signals (HRV trend, sleep, resting HR, recent strain), not just the
  calendar. Score < 40 → recovery day; 40–70 → the planned day,
  moderated; > 70 → full send. **Never a hard session on a
  low-recovery day.**
- Voice coaching: :func:`coaching_cues` turns a workout into ordered
  spoken cues ("next: 10 squats", "rest 90 seconds"). A TTS seam speaks
  them — injected, never imported at module level.

CRITICAL POSITIONING: coaching, NOT medical advice. This never
prescribes around pain or injury — every hard output carries the
"stop if it hurts, see a professional" line, and every public string
passes :func:`guard_coaching` (the #53 diagnostic-phrase ban).

Readiness source: :class:`HealthCoach.readiness` when health data is
synced; an explicit :class:`BioMetrics` (used by tests and manual
entry); otherwise an honest "unknown" that caps intensity at moderate.

Privacy: owner-scoped ONLY. ``TrainingCoach`` refuses to initialize in
a community context, mirroring :class:`HealthTimeline`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from .coach import Readiness as _CoachReadiness, guard_coaching

_log = logging.getLogger(__name__)

__all__ = [
    "BioMetrics",
    "ReadinessInfo",
    "Exercise",
    "Workout",
    "Plan",
    "Cue",
    "TrainingCoach",
    "compute_readiness",
    "generate_plan",
    "coaching_cues",
    "control_train",
    "GOALS",
    "EQUIPMENT",
    "TRAINING_DISCLAIMER",
    "MUSCLE_GROUPS",
    "VOLUME_LANDMARKS",
    "PersonalRecord",
    "muscle_groups_for",
]

TRAINING_DISCLAIMER = (
    "coaching, not medical advice: stop if anything hurts and talk to a "
    "qualified professional about pain, injury, or health conditions. "
    "Devon is not a doctor.")

#: Stop-training red flags — any mention routes to rest + the disclaimer.
_PAIN_WORDS = ("pain", "hurt", "injured", "injury", "dizzy", "chest",
               "faint", "bleeding")

GOALS = ("strength", "hypertrophy", "fat-loss", "conditioning", "general")
EQUIPMENT = ("full", "dumbbells", "home", "none")

# readiness bands (build-map spec)
_RECOVERY_MAX = 40.0
_FULL_SEND_MIN = 70.0


# ── biometrics → readiness ───────────────────────────────────────────────


@dataclass
class BioMetrics:
    """Explicit recovery inputs. ``None`` = unknown (honest, not zero).

    ``hrv_ratio``: 7-day HRV average ÷ 30-day baseline (1.0 = steady).
    ``rhr_delta_bpm``: resting HR vs personal baseline (+ = elevated).
    ``hard_sessions_7d``: hard workouts in the last 7 days.
    """
    sleep_hours: float | None = None
    hrv_ratio: float | None = None
    rhr_delta_bpm: float | None = None
    hard_sessions_7d: int = 0


@dataclass
class ReadinessInfo:
    """A 0–100 recovery score with reasons. ``has_data=False`` → unknown."""
    score: float
    level: str  # recovery | moderate | full | unknown
    reasons: list[str] = field(default_factory=list)
    has_data: bool = True

    def format(self) -> str:
        if not self.has_data:
            return ("no recovery data yet — sync health data or log "
                    "biometrics and I'll gate your training on real "
                    "numbers. Until then, sessions stay moderate.")
        emoji = {"recovery": "🔴", "moderate": "🟡",
                 "full": "🟢"}.get(self.level, "⚪")
        lines = [f"{emoji} readiness: **{self.level}** "
                 f"({self.score:.0f}/100)"]
        lines.extend(f"• {r}" for r in self.reasons)
        if self.level == "recovery":
            lines.append("\ntoday is a recovery day — mobility + walk, "
                         "no hard session.")
        return guard_coaching("\n".join(lines))


def compute_readiness(bio: BioMetrics) -> ReadinessInfo:
    """Pure readiness from explicit biometrics. Never raises.

    Weights: sleep 40%, HRV trend 30%, resting-HR 15%, recent strain 15%.
    Unknown inputs are skipped (their weight redistributes), not zeroed.
    """
    try:
        parts: list[float] = []
        weights: list[float] = []
        reasons: list[str] = []

        if bio.sleep_hours is not None:
            h = max(0.0, float(bio.sleep_hours))
            if 7.0 <= h <= 9.0:
                s = 100.0
            elif h < 7.0:
                s = max(0.0, 100.0 - (7.0 - h) * 30.0)
            else:
                s = max(0.0, 100.0 - (h - 9.0) * 20.0)
            parts.append(s); weights.append(0.40)
            reasons.append(f"sleep {h:.1f}h last night (target 7–9h)")

        if bio.hrv_ratio is not None:
            r = float(bio.hrv_ratio)
            if r >= 1.0:
                h_score, trend = 100.0, "steady or up"
            elif r >= 0.95:
                h_score, trend = 85.0, "steady"
            elif r >= 0.90:
                h_score, trend = 65.0, "dipping"
            elif r >= 0.85:
                h_score, trend = 45.0, "down"
            else:
                h_score, trend = 25.0, "down a lot"
            parts.append(h_score); weights.append(0.30)
            reasons.append(f"HRV {r:.2f}× baseline — {trend}")

        if bio.rhr_delta_bpm is not None:
            d = float(bio.rhr_delta_bpm)
            r_score = max(0.0, 100.0 - max(0.0, d) * 12.0)
            parts.append(r_score); weights.append(0.15)
            reasons.append(
                f"resting HR {'+' if d >= 0 else ''}{d:.0f} bpm vs baseline")

        n = max(0, int(bio.hard_sessions_7d or 0))
        has_signal = (bio.sleep_hours is not None
                      or bio.hrv_ratio is not None
                      or bio.rhr_delta_bpm is not None or n > 0)
        if has_signal:
            strain = max(0.0, 100.0 - min(60.0, 15.0 * n))
            parts.append(strain); weights.append(0.15)
            reasons.append(f"{n} hard session(s) in the last 7 days")

        if not parts:
            return ReadinessInfo(score=0.0, level="unknown", has_data=False,
                                 reasons=["no recovery data yet"])

        score = sum(p * w for p, w in zip(parts, weights)) / sum(weights)
        if score < _RECOVERY_MAX:
            level = "recovery"
        elif score < _FULL_SEND_MIN:
            level = "moderate"
        else:
            level = "full"
        return ReadinessInfo(score=round(score, 1), level=level,
                             reasons=reasons, has_data=True)
    except Exception:  # noqa: BLE001 — readiness never breaks training
        _log.debug("compute_readiness failed", exc_info=True)
        return ReadinessInfo(score=0.0, level="unknown", has_data=False,
                             reasons=["readiness unavailable"])


# ── exercise science: the plan templates ─────────────────────────────────

#: Muscle groups tracked for volume landmarks.
MUSCLE_GROUPS = ("chest", "back", "shoulders", "legs", "arms", "core")

#: Weekly hard-set landmarks per muscle group (RP-style MEV/MRV bands,
#: simplified estimates): (minimum effective, maximum recoverable).
#: Below MEV → "add volume"; above MRV → "pull back".
VOLUME_LANDMARKS: dict[str, tuple[int, int]] = {
    "chest": (6, 20),
    "back": (6, 22),
    "shoulders": (4, 18),
    "legs": (6, 20),
    "arms": (4, 16),
    "core": (2, 10),
}

#: Exercise key → muscle groups it trains.
_MUSCLE_MAP: dict[str, tuple[str, ...]] = {
    "bench": ("chest", "shoulders", "arms"),
    "ohp": ("shoulders", "chest", "arms"),
    "dips": ("chest", "arms", "shoulders"),
    "lateral": ("shoulders",),
    "triceps": ("arms",),
    "row": ("back", "arms"),
    "pullup": ("back", "arms"),
    "facepull": ("back", "shoulders"),
    "curl": ("arms",),
    "squat": ("legs",),
    "rdl": ("legs", "back"),
    "lunge": ("legs",),
    "deadlift": ("back", "legs"),
    "hipthrust": ("legs",),
    "calf": ("legs",),
    "core": ("core",),
    "sprint": ("legs",),
    "bike": ("legs",),
    "burpee": ("chest", "legs", "core"),
    "hipflow": ("core",),
    "thoracic": ("back",),
    "hamstring": ("legs",),
    "shoulder": ("shoulders",),
    "breath": (),
    "walk": ("legs",),
    "glute_bridge": ("legs",),
    "plank_hold": ("core",),
    "dead_bug": ("core",),
    "split_squat_iso": ("legs",),
    "wall_slide": ("shoulders", "core"),
    "deep_squat_hold": ("legs",),
    "banded_lateral_walk": ("legs",),
    "cat_cow": ("back", "core"),
    "calf_stretch": ("legs",),
    "hip_hinge_drill": ("back", "legs"),
    "scap_pushup": ("shoulders", "chest"),
}


def muscle_groups_for(key: str) -> tuple[str, ...]:
    """Muscle groups an exercise key trains. Pure; never raises."""
    return _MUSCLE_MAP.get((key or "").lower(), ())


@dataclass
class PersonalRecord:
    """An automatic PR: heaviest load (or best bodyweight volume)."""
    exercise_key: str
    exercise_name: str
    load: str
    date: str
    kind: str = "load"  # load | volume

@dataclass
class Exercise:
    """One movement. ``load`` is kg or \"bodyweight\"."""
    key: str
    name: str
    sets: int
    reps: str           # "8" | "8-12" | "30s" | "60s/side"
    rest_secs: int
    intensity: str      # easy | moderate | hard
    load: str = "bodyweight"
    notes: str = ""
    rpe_target: int | None = None  # 1-10; auto-regulated from history


# equipment variants: key → {equipment: display name}; default is the key name
_VARIANTS: dict[str, dict[str, str]] = {
    "bench": {"none": "Push-ups", "dumbbells": "Dumbbell bench press",
              "home": "Dumbbell bench press"},
    "ohp": {"none": "Pike push-ups", "dumbbells": "Dumbbell overhead press",
            "home": "Dumbbell overhead press"},
    "dips": {"none": "Chair dips", "dumbbells": "Chair dips",
             "home": "Chair dips"},
    "row": {"none": "Table rows", "dumbbells": "One-arm dumbbell row",
            "home": "One-arm dumbbell row"},
    "pullup": {"none": "Doorframe hangs + rows", "dumbbells": "Bent rows",
               "home": "Bent rows"},
    "squat": {"none": "Bodyweight squats",
              "dumbbells": "Goblet squats", "home": "Goblet squats"},
    "rdl": {"none": "Single-leg hip hinge",
            "dumbbells": "Dumbbell Romanian deadlift",
            "home": "Dumbbell Romanian deadlift"},
    "lunge": {"none": "Walking lunges", "dumbbells": "Dumbbell lunges",
              "home": "Dumbbell lunges"},
    "deadlift": {"none": "Glute bridges", "dumbbells": "Dumbbell deadlift",
                 "home": "Dumbbell deadlift"},
    "hipthrust": {"none": "Glute bridges", "dumbbells": "Dumbbell hip thrust",
                  "home": "Dumbbell hip thrust"},
}

# (key, base display name, sets, reps, rest)
_DAY_LIBRARY: dict[str, list[tuple[str, str, int, str, int]]] = {
    "push": [
        ("bench", "Bench press", 4, "6-8", 120),
        ("ohp", "Overhead press", 3, "8-10", 90),
        ("dips", "Dips", 3, "8-12", 90),
        ("lateral", "Lateral raises", 3, "12-15", 60),
        ("triceps", "Triceps pushdown", 3, "10-12", 60),
    ],
    "pull": [
        ("deadlift", "Deadlift", 4, "5", 180),
        ("pullup", "Pull-ups", 3, "6-10", 120),
        ("row", "Barbell row", 3, "8-10", 90),
        ("facepull", "Face pulls", 3, "12-15", 60),
        ("curl", "Barbell curl", 3, "10-12", 60),
    ],
    "legs": [
        ("squat", "Back squat", 4, "6-8", 150),
        ("rdl", "Romanian deadlift", 3, "8-10", 120),
        ("lunge", "Lunges", 3, "10/leg", 90),
        ("calf", "Calf raises", 3, "15-20", 45),
        ("core", "Hanging leg raises", 3, "10-12", 60),
    ],
    "upper": [
        ("bench", "Bench press", 3, "8", 120),
        ("row", "Barbell row", 3, "8", 90),
        ("ohp", "Overhead press", 3, "8", 90),
        ("pullup", "Pull-ups", 3, "max", 120),
        ("curl", "Curls", 2, "12", 60),
    ],
    "lower": [
        ("squat", "Back squat", 4, "6", 150),
        ("deadlift", "Deadlift", 3, "5", 180),
        ("lunge", "Lunges", 3, "10/leg", 90),
        ("calf", "Calf raises", 3, "15", 45),
    ],
    "full_body": [
        ("squat", "Goblet squat", 3, "10", 90),
        ("bench", "Push-ups", 3, "12", 60),
        ("row", "One-arm row", 3, "10/side", 60),
        ("rdl", "Romanian deadlift", 3, "10", 90),
        ("core", "Plank", 3, "45s", 45),
    ],
    "conditioning": [
        ("sprint", "Sprint intervals", 8, "30s", 90),
        ("bike", "Bike / brisk walk", 1, "20 min", 0),
        ("burpee", "Burpees", 4, "45s", 60),
        ("core", "Mountain climbers", 3, "40s", 45),
    ],
    "mobility": [
        ("hipflow", "Hip flow", 2, "60s/side", 30),
        ("thoracic", "Thoracic openers", 2, "10", 30),
        ("hamstring", "Hamstring stretch", 2, "45s/side", 30),
        ("shoulder", "Shoulder circles + band", 2, "12", 30),
        ("breath", "Box breathing", 3, "4-4-4-4", 30),
    ],
}

_GOAL_WEEKS: dict[str, list[str]] = {
    # 7-day templates; periodization modulates volume/intensity by phase
    "strength": ["push", "legs", "rest", "pull", "legs", "conditioning",
                 "rest"],
    "hypertrophy": ["push", "pull", "legs", "rest", "upper", "lower",
                    "rest"],
    "fat-loss": ["full_body", "conditioning", "full_body", "rest",
                 "full_body", "conditioning", "rest"],
    "conditioning": ["conditioning", "full_body", "conditioning", "rest",
                     "conditioning", "mobility", "rest"],
    "general": ["full_body", "rest", "full_body", "rest", "conditioning",
                "mobility", "rest"],
}

_GOAL_BLURBS = {
    "strength": "12-week strength block — heavy compounds, progressive "
                "overload, one conditioning day.",
    "hypertrophy": "12-week hypertrophy block — push/pull/legs plus an "
                   "upper/lower finish.",
    "fat-loss": "12-week fat-loss block — full-body circuits + "
                "conditioning, short rests.",
    "conditioning": "12-week conditioning block — engine work with two "
                    "strength anchors.",
    "general": "12-week general fitness — strength, engine, and mobility "
               "in balance.",
}


def _phase(week: int, weeks: int) -> str:
    """Periodization phase for a 1-based week."""
    frac = week / max(1, weeks)
    if frac <= 0.25:
        return "base"
    if frac <= 0.55:
        return "build"
    if frac <= 0.85:
        return "peak"
    if frac <= 0.92:
        return "deload"
    return "test"


_PHASE_MODS = {
    # (volume multiplier, intensity cap, note)
    "base": (1.15, "moderate", "base — volume up, weights moderate"),
    "build": (1.0, "hard", "build — heavier, standard volume"),
    "peak": (0.9, "hard", "peak — low reps, heaviest loads"),
    "deload": (0.5, "easy", "deload — half volume, move well"),
    "test": (0.8, "hard", "test — show what the block built"),
}


@dataclass
class Workout:
    """One day's session, already readiness-gated."""
    date: str            # YYYY-MM-DD
    kind: str            # push | pull | legs | ... | recovery | rest
    exercises: list[Exercise] = field(default_factory=list)
    duration_min: int = 45
    intensity: str = "moderate"   # easy | moderate | hard
    gated: bool = False           # True when readiness changed the plan
    gate_note: str = ""
    readiness_score: float = 0.0
    plan_id: str = ""


@dataclass
class Plan:
    id: str
    goal: str
    equipment: str
    weeks: int
    schedule: dict[str, str]      # "w3d1" -> day-kind
    blurb: str
    created_at: float = 0.0


def _resolve_name(key: str, base: str, equipment: str) -> str:
    return _VARIANTS.get(key, {}).get(equipment, base)


def _build_day(kind: str, equipment: str, phase: str,
               last_loads: dict[str, str] | None = None) -> list[Exercise]:
    """Build a day's exercises with phase modulation + progression."""
    vol_mult, intensity_cap, _ = _PHASE_MODS[phase]
    out: list[Exercise] = []
    for key, base, sets, reps, rest in _DAY_LIBRARY.get(kind, []):
        n_sets = max(1, round(sets * vol_mult))
        load = (last_loads or {}).get(key, "bodyweight")
        out.append(Exercise(
            key=key, name=_resolve_name(key, base, equipment),
            sets=n_sets, reps=reps, rest_secs=rest,
            intensity=intensity_cap, load=load))
    return out


def generate_plan(goal: str, equipment: str = "full",
                  weeks: int = 12) -> Plan:
    """Text-to-structured-plan. Never raises; unknown goal → general."""
    goal = (goal or "general").lower().strip().replace(" ", "-")
    if goal not in _GOAL_WEEKS:
        goal = "general"
    equipment = (equipment or "full").lower().strip()
    if equipment not in EQUIPMENT:
        equipment = "full"
    try:
        weeks = int(weeks)
    except (TypeError, ValueError):
        weeks = 12
    weeks = max(2, min(24, weeks))
    sched: dict[str, str] = {}
    for w in range(1, weeks + 1):
        for d, kind in enumerate(_GOAL_WEEKS[goal], start=1):
            sched[f"w{w}d{d}"] = kind
    return Plan(id="plan_" + uuid.uuid4().hex[:8], goal=goal,
                equipment=equipment, weeks=weeks, schedule=sched,
                blurb=_GOAL_BLURBS[goal], created_at=time.time())


_RECOVERY_WORKOUT = [
    ("walk", "Easy walk", 1, "20 min", 0),
    ("hipflow", "Hip flow", 2, "60s/side", 30),
    ("thoracic", "Thoracic openers", 2, "10", 30),
    ("breath", "Box breathing", 3, "4-4-4-4", 30),
]


def _moderate(workout: Workout) -> Workout:
    """Downgrade a planned workout: cap intensity, cut volume 25%."""
    capped = {"hard": "moderate"}.get(workout.intensity, workout.intensity)
    exercises = []
    for ex in workout.exercises:
        exercises.append(Exercise(
            key=ex.key, name=ex.name, sets=max(1, round(ex.sets * 0.75)),
            reps=ex.reps, rest_secs=ex.rest_secs,
            intensity={"hard": "moderate"}.get(ex.intensity, ex.intensity),
            load=ex.load, notes=ex.notes))
    workout.exercises = exercises
    workout.intensity = capped
    workout.duration_min = max(20, int(workout.duration_min * 0.8))
    return workout


# ── voice coaching ───────────────────────────────────────────────────────


@dataclass
class Cue:
    """One spoken coaching cue. ``pause_after`` seconds of quiet follow."""
    text: str
    pause_after: int = 0


def coaching_cues(workout: Workout) -> list[Cue]:
    """Workout → ordered voice-coaching cues. Pure; never raises."""
    try:
        cues: list[Cue] = []
        n = len(workout.exercises)
        cues.append(Cue(
            f"Alright — {workout.kind.replace('_', ' ')} day. "
            f"{n} movements, about {workout.duration_min} minutes. "
            "Let's work.", pause_after=3))
        for i, ex in enumerate(workout.exercises, start=1):
            cues.append(Cue(
                f"Movement {i} of {n}: {ex.name}. {ex.sets} sets of "
                f"{ex.reps}.", pause_after=5))
            if ex.rest_secs:
                cues.append(Cue(
                    f"Rest {ex.rest_secs} seconds. Breathe — nose in, "
                    "mouth out.", pause_after=ex.rest_secs))
        cues.append(Cue(
            "Done. Good work — log it when you're ready, and hydrate."))
        return [c for c in cues if c.text]
    except Exception:  # noqa: BLE001
        _log.debug("coaching_cues failed", exc_info=True)
        return []


# ── the coach ────────────────────────────────────────────────────────────


def _store_path(path: str) -> str:
    return path or os.path.expanduser("~/.nomorals/health/training.db")


class TrainingCoach:
    """Owner-scoped training plans, readiness gating, workout logging.

    ``bio`` injects explicit biometrics (tests / manual entry).
    ``health_coach`` injects a :class:`HealthCoach`; without either,
    readiness is honestly "unknown" and sessions cap at moderate.
    ``tts`` is a callable ``(text) -> audio_path | None`` — the voice
    seam; heavy TTS deps are never imported here.
    """

    def __init__(self, store_path: str = "",
                 bio: BioMetrics | None = None,
                 health_coach: Any | None = None,
                 tts: Callable[[str], str | None] | None = None,
                 community: bool = False) -> None:
        if community:
            raise PermissionError(
                "training plans are owner-scoped — not available in "
                "community spaces.")
        self._bio = bio
        self._coach = health_coach
        self._tts = tts
        self._db: sqlite3.Connection | None = None
        try:
            p = _store_path(store_path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            self._db = sqlite3.connect(p)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS plans (
                       id TEXT PRIMARY KEY, goal TEXT, equipment TEXT,
                       weeks INTEGER, schedule_json TEXT, blurb TEXT,
                       created_at REAL)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS workout_log (
                       id TEXT PRIMARY KEY, date TEXT, plan_id TEXT,
                       kind TEXT, intensity TEXT, completed INTEGER,
                       rpe INTEGER, loads_json TEXT, notes TEXT,
                       created_at REAL)""")
            self._db.execute(
                """CREATE INDEX IF NOT EXISTS wl_date
                   ON workout_log(date)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS set_feedback (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       date TEXT NOT NULL,
                       exercise_key TEXT NOT NULL,
                       rpe INTEGER NOT NULL,
                       load TEXT NOT NULL DEFAULT '',
                       created_at REAL NOT NULL)""")
            self._db.execute(
                """CREATE INDEX IF NOT EXISTS sf_key_date
                   ON set_feedback(exercise_key, date)""")
            self._db.commit()
        except Exception:  # noqa: BLE001 — bad disk → in-memory behavior
            _log.warning("training: store unavailable, running ephemeral",
                         exc_info=True)
            self._db = None

    # -- readiness --------------------------------------------------------

    def readiness(self) -> ReadinessInfo:
        """Current recovery score. Explicit bio first, HealthCoach next,
        honest unknown last."""
        if self._bio is not None:
            return compute_readiness(self._bio)
        coach = self._coach
        if coach is None:
            try:
                from .coach import HealthCoach, HealthDataSource
                coach = HealthCoach(source=HealthDataSource())
            except Exception:  # noqa: BLE001
                coach = None
        if coach is not None:
            try:
                r: _CoachReadiness = coach.readiness(log=False)
                if r.has_data:
                    level = {"high": "full", "moderate": "moderate",
                             "low": "recovery"}.get(r.level, "unknown")
                    return ReadinessInfo(score=r.score, level=level,
                                         reasons=list(r.reasons),
                                         has_data=True)
            except Exception:  # noqa: BLE001 — coach trouble → unknown
                _log.debug("training: health-coach readiness failed",
                           exc_info=True)
        return ReadinessInfo(score=0.0, level="unknown", has_data=False,
                             reasons=["no recovery data yet"])

    # -- plans ------------------------------------------------------------

    def new_plan(self, goal: str, equipment: str = "full",
                 weeks: int = 12) -> Plan:
        """Generate + persist a periodized plan."""
        plan = generate_plan(goal, equipment, weeks)
        try:
            if self._db is not None:
                self._db.execute(
                    "INSERT OR REPLACE INTO plans VALUES (?,?,?,?,?,?,?)",
                    (plan.id, plan.goal, plan.equipment, plan.weeks,
                     json.dumps(plan.schedule), plan.blurb,
                     plan.created_at))
                self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("training: plan persist failed", exc_info=True)
        return plan

    def get_plan(self, plan_id: str) -> Plan | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
            if row is None:
                return None
            return Plan(id=row["id"], goal=row["goal"],
                        equipment=row["equipment"], weeks=row["weeks"],
                        schedule=json.loads(row["schedule_json"] or "{}"),
                        blurb=row["blurb"] or "",
                        created_at=row["created_at"] or 0.0)
        except Exception:  # noqa: BLE001
            return None

    def latest_plan(self) -> Plan | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM plans ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return self.get_plan(row["id"]) if row else None
        except Exception:  # noqa: BLE001
            return None

    # -- the daily workout (readiness-gated) ------------------------------

    def _last_loads(self, kind: str) -> dict[str, str]:
        """Per-exercise loads from the last completed same-kind session."""
        try:
            if self._db is None:
                return {}
            row = self._db.execute(
                """SELECT loads_json FROM workout_log
                   WHERE kind = ? AND completed = 1
                   ORDER BY date DESC LIMIT 1""", (kind,)).fetchone()
            return json.loads(row["loads_json"] or "{}") if row else {}
        except Exception:  # noqa: BLE001
            return {}

    def today(self, plan: Plan | None = None,
              day: date | None = None) -> Workout:
        """Today's workout, gated by readiness.

        Never a hard session on a low-recovery day; unknown biometrics
        cap the session at moderate (honest, not reckless).
        """
        day = day or date.today()
        plan = plan or self.latest_plan()
        info = self.readiness()
        day_key = day.isoformat()

        # recovery day — the plan waits
        if info.level == "recovery":
            w = Workout(
                date=day_key, kind="recovery",
                exercises=[Exercise(
                    key=k, name=n, sets=s, reps=r, rest_secs=rs,
                    intensity="easy") for k, n, s, r, rs in
                    _RECOVERY_WORKOUT],
                duration_min=25, intensity="easy", gated=True,
                gate_note=("readiness "
                          f"{info.score:.0f}/100 — recovery day, no hard "
                          "session."),
                readiness_score=info.score,
                plan_id=plan.id if plan else "")
            return w

        if plan is None:
            # no plan yet → a sensible general day, gated like everything
            kinds = _GOAL_WEEKS["general"]
            kind = kinds[day.weekday() % 7]
            equipment = "full"
        else:
            elapsed = (day - date.fromtimestamp(plan.created_at)).days
            week = max(1, min(plan.weeks, elapsed // 7 + 1))
            kind = plan.schedule.get(f"w{week}d{day.isoweekday()}", "rest")
            equipment = plan.equipment

        if kind == "rest":
            return Workout(date=day_key, kind="rest", exercises=[],
                           duration_min=0, intensity="easy", gated=False,
                           readiness_score=info.score,
                           plan_id=plan.id if plan else "")

        week_no = 1
        if plan is not None:
            elapsed = (day - date.fromtimestamp(plan.created_at)).days
            week_no = max(1, min(plan.weeks, elapsed // 7 + 1))
        phase = _phase(week_no, plan.weeks if plan else 12)
        exercises = _build_day(kind, equipment, phase,
                               self._last_loads(kind))
        w = Workout(date=day_key, kind=kind, exercises=exercises,
                    duration_min=20 + 6 * len(exercises),
                    intensity=_PHASE_MODS[phase][1],
                    gated=False, readiness_score=info.score,
                    plan_id=plan.id if plan else "")

        if info.level == "moderate":
            _moderate(w)
            w.gated = True
            w.gate_note = (f"readiness {info.score:.0f}/100 — moderated: "
                           "intensity capped, volume −25%.")
        elif not info.has_data:
            _moderate(w)
            w.gated = True
            w.gate_note = ("no recovery data — capped at moderate until "
                           "health data syncs or you log biometrics.")
        return w

    # -- logging + progression --------------------------------------------

    def log_workout(self, workout: Workout, completed: bool,
                    rpe: int | None = None, notes: str = "",
                    loads: dict[str, str] | None = None,
                    set_rpes: dict[str, int] | None = None) -> bool:
        """Log a session. Progression bumps loads when RPE ≤ 7.

        ``set_rpes``: per-exercise RPE ({"squat": 8}) — feeds the
        auto-regulation loop (Juggernaut pattern): RPE ≥9 twice running
        → hold progression; RPE ≤6 → double bump.

        Never prescribes around pain: pain words → forced rest guidance.
        """
        try:
            lowered = (notes or "").lower()
            if any(w in lowered for w in _PAIN_WORDS):
                notes = (notes + " [pain reported — rest; see a "
                         "professional if it persists]").strip()
            final_loads: dict[str, str] = dict(loads or {})
            auto = self._autoregulation(workout, set_rpes or {})
            if completed and (rpe is None or rpe <= 7):
                for ex in workout.exercises:
                    base = (loads or {}).get(ex.key, ex.load)
                    final_loads[ex.key] = _bump_load(base,
                                                    factor=auto.get(
                                                        ex.key, 1.0))
            elif not final_loads:
                final_loads = {ex.key: ex.load for ex in workout.exercises}
            if self._db is not None:
                self._db.execute(
                    """INSERT INTO workout_log VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    ("wl_" + uuid.uuid4().hex[:8], workout.date,
                     workout.plan_id, workout.kind, workout.intensity,
                     1 if completed else 0,
                     int(rpe) if rpe is not None else None,
                     json.dumps(final_loads), notes, time.time()))
                for key, erpe in (set_rpes or {}).items():
                    try:
                        self._db.execute(
                            "INSERT INTO set_feedback "
                            "(date, exercise_key, rpe, load, created_at) "
                            "VALUES (?,?,?,?,?)",
                            (workout.date, key, int(erpe),
                             final_loads.get(key, ""), time.time()))
                    except Exception:  # noqa: BLE001
                        pass
                self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            _log.debug("training: log failed", exc_info=True)
            return False

    def _autoregulation(self, workout: Workout,
                        set_rpes: dict[str, int]) -> dict[str, float]:
        """Per-exercise progression factor from RPE history.

        Last two logged RPEs for the key: ≥9 twice → 0.0 (hold, no
        bump); ≤6 twice → 2.0 (double bump); else 1.0. Pure-ish;
        never raises.
        """
        factors: dict[str, float] = {}
        try:
            if self._db is None:
                return factors
            for ex in workout.exercises:
                rows = self._db.execute(
                    "SELECT rpe FROM set_feedback "
                    "WHERE exercise_key = ? ORDER BY date DESC LIMIT 2",
                    (ex.key,)).fetchall()
                # include today's just-logged values
                hist = [int(set_rpes.get(ex.key, 0)) or None]
                hist = [r for r in hist if r] + [int(r[0]) for r in rows]
                hist = hist[:2]
                if len(hist) == 2 and all(r >= 9 for r in hist):
                    factors[ex.key] = 0.0
                elif len(hist) == 2 and all(r <= 6 for r in hist):
                    factors[ex.key] = 2.0
                else:
                    factors[ex.key] = 1.0
        except Exception:  # noqa: BLE001
            _log.debug("autoregulation failed", exc_info=True)
        return factors

    def progression_advice(self, exercise_key: str) -> str:
        """Why the next bump is what it is — the auto-regulation loop
        explained. Never raises."""
        try:
            if self._db is None:
                return "no history yet — log RPEs and I'll auto-regulate."
            rows = self._db.execute(
                "SELECT rpe, date FROM set_feedback "
                "WHERE exercise_key = ? ORDER BY date DESC LIMIT 3",
                (exercise_key,)).fetchall()
            if not rows:
                return (f"no RPE history for {exercise_key} yet — log "
                        f"per-exercise RPEs and I'll tune progression.")
            rpes = [int(r[0]) for r in rows]
            avg = sum(rpes) / len(rpes)
            if len(rpes) >= 2 and all(r >= 9 for r in rpes[:2]):
                return (f"{exercise_key}: last RPEs {rpes[:2]} — holding "
                        f"the load steady until it feels like ≤8.")
            if len(rpes) >= 2 and all(r <= 6 for r in rpes[:2]):
                return (f"{exercise_key}: last RPEs {rpes[:2]} — that was "
                        f"easy, bumping the load extra next time.")
            return (f"{exercise_key}: recent RPEs {rpes} (avg {avg:.1f}) "
                    f"— standard +2.5% progression.")
        except Exception:  # noqa: BLE001
            return "couldn't read progression history."

    def history(self, days: int = 14) -> list[dict[str, Any]]:
        try:
            if self._db is None:
                return []
            since = (date.today() - timedelta(days=days)).isoformat()
            rows = self._db.execute(
                """SELECT date, kind, intensity, completed, rpe, notes
                   FROM workout_log WHERE date >= ? ORDER BY date DESC""",
                (since,)).fetchall()
            return [dict(r) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    # -- volume landmarks (RP pattern) -------------------------------------

    def weekly_sets(self, weeks_back: int = 0) -> dict[str, int]:
        """Hard sets per muscle group for a calendar week.

        Derived from completed workout logs via the day library
        (kind → exercises → sets → muscle groups). Pure-ish; never
        raises. ``weeks_back=0`` = this week.
        """
        out: dict[str, int] = {m: 0 for m in MUSCLE_GROUPS}
        try:
            if self._db is None:
                return out
            today = date.today()
            monday = today - timedelta(days=today.weekday()
                                       + 7 * weeks_back)
            sunday = monday + timedelta(days=6)
            rows = self._db.execute(
                """SELECT kind, loads_json FROM workout_log
                   WHERE date >= ? AND date <= ? AND completed = 1""",
                (monday.isoformat(), sunday.isoformat())).fetchall()
            for row in rows:
                kind = row["kind"] or ""
                try:
                    keys = set(json.loads(row["loads_json"] or "{}"))
                except Exception:  # noqa: BLE001
                    keys = set()
                lib = {k: s for k, _n, s, _r, _rest
                       in _DAY_LIBRARY.get(kind, [])}
                for key in keys or lib:
                    sets = lib.get(key, 3)
                    for muscle in muscle_groups_for(key):
                        out[muscle] = out.get(muscle, 0) + sets
            return out
        except Exception:  # noqa: BLE001
            _log.debug("weekly_sets failed", exc_info=True)
            return out

    def volume_report(self) -> str:
        """This week's sets vs landmarks — MEV/MRV bands. Never raises."""
        try:
            sets = self.weekly_sets()
            total = sum(sets.values())
            lines = ["📊 **weekly volume** — hard sets per muscle group "
                     "(MEV–MRV bands):"]
            for muscle in MUSCLE_GROUPS:
                lo, hi = VOLUME_LANDMARKS[muscle]
                n = sets.get(muscle, 0)
                if n < lo:
                    flag = "🔵 add volume"
                elif n > hi:
                    flag = "🔴 pull back"
                else:
                    flag = "🟢 sweet spot"
                bar = "█" * min(10, n) + "░" * max(0, 10 - min(10, n))
                lines.append(f"• {muscle:9s} {bar} {n:2d} "
                             f"(band {lo}–{hi}) {flag}")
            lines.append(f"\ntotal: {total} hard sets this week.")
            return guard_coaching("\n".join(lines))
        except Exception:  # noqa: BLE001
            return "couldn't build the volume report."

    # -- automatic PR tracking (Strava pattern) ----------------------------

    @staticmethod
    def _kg(load: str) -> float | None:
        import re
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(kg)?\s*", load or "")
        return float(m.group(1)) if m else None

    def personal_records(self) -> list[PersonalRecord]:
        """Heaviest logged load per exercise — automatic, from history."""
        out: list[PersonalRecord] = []
        try:
            if self._db is None:
                return out
            rows = self._db.execute(
                """SELECT date, loads_json FROM workout_log
                   WHERE completed = 1 ORDER BY date DESC
                   LIMIT 500""").fetchall()
            best: dict[str, tuple[float, str, str]] = {}
            names: dict[str, str] = {}
            for row in rows:
                try:
                    loads = json.loads(row["loads_json"] or "{}")
                except Exception:  # noqa: BLE001
                    continue
                for key, load in loads.items():
                    kg = self._kg(str(load))
                    if kg is None:
                        continue
                    if key not in best or kg > best[key][0]:
                        best[key] = (kg, str(load), row["date"])
            # resolve display names from the day library
            for kind, days in _DAY_LIBRARY.items():
                for key, name, _s, _r, _rest in days:
                    names.setdefault(key, name)
            for key, (kg, load, d) in sorted(
                    best.items(), key=lambda kv: -kv[1][0]):
                out.append(PersonalRecord(
                    exercise_key=key,
                    exercise_name=names.get(key, key),
                    load=load, date=d))
            return out
        except Exception:  # noqa: BLE001
            _log.debug("personal_records failed", exc_info=True)
            return out

    def format_prs(self, limit: int = 10) -> str:
        try:
            prs = self.personal_records()[:max(1, limit)]
            if not prs:
                return "no PRs yet — log workouts with loads and I'll " \
                    "track them automatically. 🏋️"
            lines = ["🏆 **personal records** (auto-tracked):"]
            medals = ["🥇", "🥈", "🥉"]
            for i, pr in enumerate(prs):
                medal = medals[i] if i < 3 else "•"
                lines.append(f"{medal} {pr.exercise_name}: **{pr.load}** "
                             f"({pr.date})")
            return guard_coaching("\n".join(lines))
        except Exception:  # noqa: BLE001
            return "couldn't read PRs right now."

    # -- deload advisory (RP pattern) --------------------------------------

    def deload_advisory(self) -> str:
        """3 weeks of rising volume + sagging readiness → suggest deload.

        Never raises; returns "" when no deload is indicated.
        """
        try:
            w0 = sum(self.weekly_sets(0).values())
            w1 = sum(self.weekly_sets(1).values())
            w2 = sum(self.weekly_sets(2).values())
            if w2 == 0 and w1 == 0:
                return ""
            rising = w0 >= w1 >= w2 and w0 > 0 and (w0 - w2) >= 6
            readiness_low = False
            try:
                info = self.readiness()
                readiness_low = info.has_data and info.score < 55
            except Exception:  # noqa: BLE001
                pass
            if rising and (readiness_low or w0 - w2 >= 15):
                text = (f"📉 deload advisory: volume climbed {w2} → "
                        f"{w1} → {w0} hard sets over 3 weeks"
                        + (" and readiness is sagging"
                           if readiness_low else "")
                        + ". Next week: half volume, same movements, "
                          "move well — then come back fresh.")
                return guard_coaching(text)
            return ""
        except Exception:  # noqa: BLE001
            return ""

    # -- weak-point notes (form → programming) -----------------------------

    def weak_point_notes(self, analyses: list[Any] | None = None) -> str:
        """Map form-analysis issues to programming notes.

        "Upper-back flagged twice → add face pulls." Never raises.
        """
        try:
            from collections import Counter
            counts: Counter[str] = Counter()
            for a in analyses or []:
                for issue in getattr(a, "issues", None) or []:
                    counts[getattr(issue, "checkpoint", "?")] += 1
            if not counts:
                return ""
            suggestions = {
                "knees": "banded lateral walks + goblet squats",
                "depth": "deep squat holds + ankle mobility",
                "spine": "hip hinge drills + cat-cow",
                "elbows": "scapular push-ups",
                "body": "plank holds",
                "hips": "glute bridges",
                "front_knee": "split squat holds",
            }
            lines = ["🎯 **weak-point notes** (from form checks):"]
            for cp, n in counts.most_common(3):
                fix = suggestions.get(cp, "targeted accessory work")
                lines.append(f"• {cp} flagged {n}× → add {fix}")
            return guard_coaching("\n".join(lines))
        except Exception:  # noqa: BLE001
            return ""

    # -- voice ------------------------------------------------------------

    def speak_cues(self, workout: Workout) -> list[str]:
        """Speak coaching cues via the injected TTS seam.

        Returns audio paths (or the cue text when no TTS is wired —
        honest, never a fabricated path).
        """
        cues = coaching_cues(workout)
        if self._tts is None:
            return [c.text for c in cues]
        out: list[str] = []
        for cue in cues:
            try:
                path = self._tts(cue.text)
                out.append(path or cue.text)
            except Exception:  # noqa: BLE001
                out.append(cue.text)
        return out

    # -- formatting -------------------------------------------------------

    def format_workout(self, workout: Workout) -> str:
        try:
            lines = [f"🏋️ **{workout.kind.replace('_', ' ').title()}** — "
                     f"{workout.date} ({workout.intensity}, "
                     f"~{workout.duration_min} min)"]
            if workout.gated and workout.gate_note:
                lines.append(f"_{workout.gate_note}_")
            for i, ex in enumerate(workout.exercises, start=1):
                load = f" @ {ex.load}" if ex.load != "bodyweight" else ""
                rpe = f" (RPE {ex.rpe_target})" \
                    if ex.rpe_target else ""
                lines.append(f"{i}. {ex.name} — {ex.sets}×{ex.reps}{load}"
                             f"{rpe} (rest {ex.rest_secs}s)")
            lines.append(f"\n_{TRAINING_DISCLAIMER}_")
            return guard_coaching("\n".join(lines))
        except Exception:  # noqa: BLE001
            return "workout unavailable."

    def format_plan(self, plan: Plan) -> str:
        try:
            lines = [f"📋 **{plan.goal.title()} plan** — {plan.weeks} weeks "
                     f"({plan.equipment})",
                     f"_{plan.blurb}_", ""]
            for w in range(1, min(plan.weeks, 4) + 1):
                days = [plan.schedule.get(f"w{w}d{d}", "rest")
                        for d in range(1, 8)]
                lines.append(f"week {w} [{_phase(w, plan.weeks)}]: "
                             + " · ".join(k[:4] for k in days))
            if plan.weeks > 4:
                lines.append(f"… ({plan.weeks - 4} more weeks, same "
                             "rotation, periodized)")
            lines.append(f"\n_{TRAINING_DISCLAIMER}_")
            return guard_coaching("\n".join(lines))
        except Exception:  # noqa: BLE001
            return "plan unavailable."


_USAGE = (
    "/train plan <goal> [equipment] [weeks] — build a periodized plan "
    "(goals: strength, hypertrophy, fat-loss, conditioning, general; "
    "equipment: full, dumbbells, home, none)\n"
    "/train today — today's workout, gated by your recovery\n"
    "/train readiness — recovery score from sleep + HRV + strain\n"
    "/train log [completed|skipped] [rpe 1-10] [notes] — log today's session\n"
    "/train volume — this week's sets per muscle vs MEV/MRV bands\n"
    "/train prs — your auto-tracked personal records\n"
    "/train deload — check whether a deload week is due\n"
    "/train list — your plans")


def control_train(tail: str, *,
                  coach: "TrainingCoach | None" = None) -> str:
    """Chat entry: /train. Owner-only at the dispatch layer. Never raises."""
    try:
        raw = (tail or "").strip()
        tc = coach or TrainingCoach()
        if not raw or raw.split()[0] in ("help", "?"):
            return _USAGE
        verb, _, rest = raw.partition(" ")
        verb = verb.lower()

        if verb == "plan":
            parts = rest.split()
            goal = parts[0] if parts else "general"
            equipment = parts[1] if len(parts) > 1 else "full"
            weeks = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() \
                else 12
            plan = tc.new_plan(goal, equipment, weeks)
            return tc.format_plan(plan)

        if verb == "today":
            plan = tc.latest_plan()
            if plan is None:
                return ("no plans yet — /train plan strength, for example.")
            w = tc.today(plan)
            return tc.format_workout(w)

        if verb == "readiness":
            return tc.readiness().format()

        if verb == "volume":
            return tc.volume_report()

        if verb == "prs":
            return tc.format_prs()

        if verb == "deload":
            advice = tc.deload_advisory()
            return advice or ("no deload indicated right now — volume "
                              "and recovery look sustainable. 💪")

        if verb == "log":
            plan = tc.latest_plan()
            if plan is None:
                return ("no plans yet — /train plan strength, for example.")
            w = tc.today(plan)
            parts = rest.split()
            completed = True
            rpe = None
            notes_parts: list[str] = []
            for p in parts:
                pl = p.lower()
                if pl in ("completed", "done", "yes"):
                    completed = True
                elif pl in ("skipped", "no", "missed"):
                    completed = False
                elif pl.isdigit() and 1 <= int(pl) <= 10:
                    rpe = int(pl)
                else:
                    notes_parts.append(p)
            ok = tc.log_workout(w, completed, rpe=rpe,
                                notes=" ".join(notes_parts).strip())
            if not ok:
                return "couldn't save that — try again."
            if completed:
                return ("logged. " + ("RPE noted — loads bumped where it "
                        "was easy." if rpe is not None and rpe <= 7 else
                        "solid work."))
            return "logged as skipped — the plan adjusts, no guilt."

        if verb == "list":
            plan = tc.latest_plan()
            if plan is None:
                return ("no plans yet — /train plan strength, for example.")
            return tc.format_plan(plan)

        return _USAGE
    except Exception:  # noqa: BLE001 — chat never breaks
        _log.debug("control_train failed", exc_info=True)
        return "training hiccup — try /train help."


def _bump_load(load: str, factor: float = 1.0) -> str:
    """Small linear progression: +2.5% on numeric kg loads, scaled by
    the auto-regulation factor (0.0 = hold, 2.0 = double bump)."""
    try:
        import re
        if factor <= 0:
            return load
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(kg)?\s*", load or "")
        if not m:
            return load
        kg = float(m.group(1)) * (1 + 0.025 * factor)
        num = f"{kg:.1f}".rstrip("0").rstrip(".")
        return num + (m.group(2) or "")
    except Exception:  # noqa: BLE001
        return load
