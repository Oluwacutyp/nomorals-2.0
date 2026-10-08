"""Tests for nomorals/health/training.py (build-map #85).

Recovery-gated programming + voice coaching. All offline. Biometrics
are injected via BioMetrics — no health-cli, no network, no real TTS.
"""

import os
import tempfile
from datetime import date

import pytest

from nomorals.health.training import (
    BioMetrics,
    TrainingCoach,
    _bump_load,
    _phase,
    coaching_cues,
    compute_readiness,
    control_train,
    generate_plan,
    guard_coaching,
)


def _tmp() -> str:
    return os.path.join(tempfile.mkdtemp(), "training.db")


def _full_bio() -> BioMetrics:
    return BioMetrics(sleep_hours=8.0, hrv_ratio=1.05, rhr_delta_bpm=0.0,
                      hard_sessions_7d=0)


def _mod_bio() -> BioMetrics:
    return BioMetrics(sleep_hours=5.5, hrv_ratio=0.87, rhr_delta_bpm=4.0,
                      hard_sessions_7d=3)


def _low_bio() -> BioMetrics:
    return BioMetrics(sleep_hours=4.0, hrv_ratio=0.80, rhr_delta_bpm=8.0,
                      hard_sessions_7d=5)


MON = date(2026, 10, 5)  # a Monday → w1d1 = push (strength)


# ── readiness scoring ──────────────────────────────────────────────────


def test_readiness_full():
    r = compute_readiness(_full_bio())
    assert r.level == "full" and r.score > 70 and r.has_data


def test_readiness_moderate():
    r = compute_readiness(_mod_bio())
    assert r.level == "moderate" and 40 <= r.score < 70


def test_readiness_recovery():
    r = compute_readiness(_low_bio())
    assert r.level == "recovery" and r.score < 40


def test_readiness_unknown_on_empty_bio():
    r = compute_readiness(BioMetrics())
    assert r.level == "unknown" and not r.has_data


def test_readiness_never_raises():
    r = compute_readiness(BioMetrics(sleep_hours=float("nan")))
    assert isinstance(r.score, float)


# ── plan generation ────────────────────────────────────────────────────


def test_generate_plan_structure():
    p = generate_plan("strength", "full", 12)
    assert p.goal == "strength" and p.weeks == 12
    assert p.schedule["w1d1"] == "push"
    assert p.schedule["w12d7"] == "rest"
    assert len(p.schedule) == 12 * 7


def test_generate_plan_unknown_goal_falls_back():
    assert generate_plan("skydiving").goal == "general"


def test_generate_plan_weeks_clamped():
    assert generate_plan("strength", weeks=99).weeks == 24
    assert generate_plan("strength", weeks=0).weeks == 2


def test_phase_rotation():
    assert _phase(1, 12) == "base"
    assert _phase(6, 12) == "build"
    assert _phase(9, 12) == "peak"
    assert _phase(11, 12) == "deload"
    assert _phase(12, 12) == "test"


def test_equipment_variants_bodyweight():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    w = tc.today(generate_plan("strength", "none", 4), day=MON)
    names = " ".join(e.name for e in w.exercises)
    assert "Push-ups" in names  # bench → bodyweight variant


# ── readiness gating ───────────────────────────────────────────────────


def test_low_recovery_never_hard():
    tc = TrainingCoach(store_path=_tmp(), bio=_low_bio())
    plan = tc.new_plan("strength")
    w = tc.today(plan, day=MON)
    assert w.kind == "recovery"
    assert w.gated
    assert w.intensity != "hard"
    assert all(e.intensity != "hard" for e in w.exercises)
    assert "recovery day" in w.gate_note


def test_moderate_recovery_moderates_plan():
    tc = TrainingCoach(store_path=_tmp(), bio=_mod_bio())
    plan = tc.new_plan("strength")
    w = tc.today(plan, day=MON)
    assert w.kind == "push" and w.gated
    assert w.intensity != "hard"
    assert all(e.intensity != "hard" for e in w.exercises)


def test_full_recovery_full_send():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    plan = tc.new_plan("strength")
    w = tc.today(plan, day=MON)
    assert w.kind == "push" and not w.gated
    assert w.exercises  # real work prescribed


def test_unknown_readiness_caps_at_moderate():
    tc = TrainingCoach(store_path=_tmp())  # no bio, no coach data
    plan = tc.new_plan("strength")
    w = tc.today(plan, day=MON)
    assert w.kind == "push" and w.gated
    assert w.intensity != "hard"
    assert "no recovery data" in w.gate_note


def test_rest_day_stays_rest():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    plan = tc.new_plan("strength")
    w = tc.today(plan, day=date(2026, 10, 7))  # Wednesday → rest
    assert w.kind == "rest" and not w.exercises


def test_recovery_overrides_even_hard_day():
    tc = TrainingCoach(store_path=_tmp(), bio=_low_bio())
    plan = tc.new_plan("hypertrophy")
    w = tc.today(plan, day=MON)
    assert w.kind == "recovery"  # plan waits


# ── logging + progression ──────────────────────────────────────────────


def test_log_progression_bumps_load():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    plan = tc.new_plan("strength")
    w = tc.today(plan, day=MON)
    assert tc.log_workout(w, True, rpe=6, loads={"bench": "60kg"})
    w2 = tc.today(plan, day=MON)
    bench = next(e for e in w2.exercises if e.key == "bench")
    assert bench.load == "61.5kg"


def test_hard_rpe_no_bump():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    plan = tc.new_plan("strength")
    w = tc.today(plan, day=MON)
    tc.log_workout(w, True, rpe=9, loads={"bench": "60kg"})
    w2 = tc.today(plan, day=MON)
    bench = next(e for e in w2.exercises if e.key == "bench")
    assert bench.load == "60kg"


def test_bump_load_math():
    assert _bump_load("60kg") == "61.5kg"
    assert _bump_load("bodyweight") == "bodyweight"
    assert _bump_load("") == ""


def test_pain_note_forces_rest_guidance():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    plan = tc.new_plan("strength")
    w = tc.today(plan)  # today, so history(days=1) sees it
    assert tc.log_workout(w, False, notes="knee pain during squats")
    hist = tc.history(days=1)
    assert hist and "pain" in hist[0]["notes"] and "rest" in hist[0]["notes"]


def test_history_roundtrip():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    plan = tc.new_plan("general")
    tc.log_workout(tc.today(plan), True, rpe=7)
    assert len(tc.history(days=1)) == 1


# ── voice coaching ─────────────────────────────────────────────────────


def test_coaching_cues_structure():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    w = tc.today(tc.new_plan("strength"), day=MON)
    cues = coaching_cues(w)
    assert len(cues) >= 3
    assert "push" in cues[0].text.lower()
    assert any("rest" in c.text.lower() for c in cues[1:])
    assert all(c.text for c in cues)


def test_speak_cues_no_tts_honest():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    w = tc.today(tc.new_plan("strength"), day=MON)
    out = tc.speak_cues(w)
    assert out and all(isinstance(x, str) for x in out)
    assert "push" in out[0].lower()  # cue text, not a fake path


def test_speak_cues_with_tts():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio(),
                       tts=lambda text: "/tmp/cue.wav")
    w = tc.today(tc.new_plan("strength"), day=MON)
    assert tc.speak_cues(w) == ["/tmp/cue.wav"] * len(coaching_cues(w))


def test_speak_cues_tts_failure_falls_back():
    def boom(text):
        raise RuntimeError("nope")
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio(), tts=boom)
    w = tc.today(tc.new_plan("strength"), day=MON)
    assert tc.speak_cues(w)  # text fallback, no raise


# ── chat ───────────────────────────────────────────────────────────────


def test_chat_plan():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    out = control_train("plan strength full 12", coach=tc)
    assert "strength" in out.lower() and "week 1" in out.lower()


def test_chat_today_readiness_log():
    tc = TrainingCoach(store_path=_tmp(), bio=_mod_bio())
    tc.new_plan("strength")
    today = control_train("today", coach=tc)
    assert "moderated" in today.lower() or "push" in today.lower()
    assert "logged" in control_train("log completed 6", coach=tc).lower()
    assert "readiness" in control_train("readiness", coach=tc).lower()


def test_chat_usage_and_garbage():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    assert "/train plan" in control_train("", coach=tc)
    assert "/train plan" in control_train("frobnicate", coach=tc)
    assert "no plans yet" in control_train("today", coach=tc).lower()


def test_chat_low_readiness_today():
    tc = TrainingCoach(store_path=_tmp(), bio=_low_bio())
    tc.new_plan("strength")
    assert "recovery day" in control_train("today", coach=tc).lower()


# ── positioning guards ─────────────────────────────────────────────────


def test_no_diagnostic_phrases_in_outputs():
    tc = TrainingCoach(store_path=_tmp(), bio=_low_bio())
    plan = tc.new_plan("strength")
    texts = [tc.format_plan(plan), tc.format_workout(tc.today(plan, day=MON)),
             tc.readiness().format(),
             control_train("plan strength", coach=tc)]
    for t in texts:
        guard_coaching(t)  # raises on banned phrases


def test_disclaimer_present():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    w = tc.today(tc.new_plan("general"), day=MON)
    assert "not medical advice" in tc.format_workout(w)


def test_community_refused():
    with pytest.raises(PermissionError):
        TrainingCoach(community=True)


def test_never_raises_on_garbage():
    tc = TrainingCoach(store_path=_tmp(), bio=_full_bio())
    control_train(None, coach=tc)  # type: ignore[arg-type]
    tc.today(None, day=None)
    coaching_cues(None)  # type: ignore[arg-type]
