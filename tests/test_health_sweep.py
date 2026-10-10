"""Sweep tests: mined-then-built upgrades across nomorals/health.

Covers the new behavior added in the health sweep (dynamic sleep need,
readiness contributors/history, chronic drift window, illness watch,
volume landmarks, PR tracking, auto-regulation, streak freeze, monthly
challenges, quantitative form hooks, clarifying triage questions,
handover reports, macro tracking, daily totals, water, mood stats/map,
lag patterns, vitals trends, symptom stats, doctor reports).

All offline. Nothing here shells out or needs mediapipe.
"""
import time
from datetime import date, timedelta

import pytest

from nomorals.health.timeline import (
    HealthTimeline,
    parse_health_note,
    parse_measurement,
    sparkline,
)
from nomorals.health.coach import (
    HealthCoach,
    HealthDataSource,
    guard_coaching,
    progress_bar,
    readiness_band_advice,
    sleep_need,
    sleep_score_vs_need,
)
from nomorals.health.drift import (
    DriftMonitor,
    illness_watch,
    projected_recovery,
    recovery_plan,
)
from nomorals.health.training import (
    Exercise,
    TrainingCoach,
    Workout,
    control_train,
    muscle_groups_for,
)
from nomorals.health.challenges import (
    ChallengeStore,
    control_challenge,
)
from nomorals.health.form import (
    ANGLE_HOOKS,
    ANGLE_RULES,
    FormStore,
    analyze_form,
    control_form,
    count_reps,
    form_trend,
    joint_angle,
    mediapipe_available,
)
from nomorals.health.previsit import (
    answer_clarifications,
    clarify_questions,
    format_route,
    handover_report,
    route_with_context,
    triage_route,
)
from nomorals.health.nutrition import (
    NIGERIAN_FOODS,
    DayCard,
    MealDraft,
    MealItem,
    NutritionStore,
    complete_meal,
    daily_totals,
    log_water,
    repeat_meal,
)
from nomorals.health.patterns import (
    best_worst_days,
    detect_lag_patterns,
    detect_patterns,
    format_patterns,
    mood_map,
    mood_stats,
)


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def tl(tmp_path):
    t = HealthTimeline(db_path=str(tmp_path / "health.db"))
    yield t
    t.close()


@pytest.fixture()
def dm(tmp_path, tl):
    m = DriftMonitor(tl, db_path=str(tmp_path / "drift.db"))
    yield m
    m.close()


@pytest.fixture()
def tc(tmp_path):
    return TrainingCoach(store_path=str(tmp_path / "training.db"))


@pytest.fixture()
def cs(tmp_path):
    return ChallengeStore(db_path=str(tmp_path / "challenges.db"))


@pytest.fixture()
def ns(tmp_path):
    return NutritionStore(db_path=str(tmp_path / "nutrition.db"))


@pytest.fixture()
def fs(tmp_path):
    return FormStore(db_path=str(tmp_path / "form.db"))


class FakeSource(HealthDataSource):
    def __init__(self, *, metrics=None, sleep=None, workouts=None):
        super().__init__(binary="/nonexistent/health-cli")
        self._metrics = metrics or []
        self._sleep = sleep or []
        self._workouts = workouts or []

    def _run(self, *args):
        raise AssertionError("no subprocesses in tests")

    def status(self, provider):
        n = 120 if (self._metrics or self._sleep) else 0
        return {"categories": [{"name": "x", "record_count": n}]}

    def metrics(self, start, end):
        return self._metrics

    def sleep_sessions(self, start, end):
        return self._sleep

    def workouts(self, start, end):
        return self._workouts


def _metric(hrv):
    return {"heart_rate_variability_ms": hrv, "step_count": 8000}


def _sleep(hours):
    return {"sleep_in_bed_duration_sec": hours * 3600,
            "sleep_awake_duration_sec": 0.3 * 3600,
            "end_datetime": "2026-10-10T06:30:00-04:00",
            "start_datetime": "2026-10-09T22:30:00-04:00"}


def _coach(source, tmp_path):
    from nomorals.health.timeline import HealthTimeline as HT
    tl = HT(db_path=str(tmp_path / "coach-tl.db"))
    c = HealthCoach(source=source, timeline=tl)
    # keep readiness history out of the real home dir
    hist_db = str(tmp_path / "coach.db")

    def _hdb():
        import sqlite3, os
        os.makedirs(os.path.dirname(hist_db), exist_ok=True)
        db = sqlite3.connect(hist_db)
        db.execute(
            """CREATE TABLE IF NOT EXISTS readiness_log (
                   ts REAL PRIMARY KEY, level TEXT, score REAL,
                   numbers_json TEXT)""")
        db.commit()
        return db
    c._history_db = _hdb
    return c


# ── timeline: vitals, symptoms, search, report ────────────────────────────

def test_parse_measurement_bp():
    assert parse_measurement("120/80", "mmHg") == [
        ("blood_pressure_systolic", 120.0, "mmHg"),
        ("blood_pressure_diastolic", 80.0, "mmHg")]


def test_parse_measurement_weight():
    assert parse_measurement("72kg")[0] == ("weight", 72.0, "kg")
    assert parse_measurement("160 lbs")[0][1] == pytest.approx(72.6, abs=0.1)


def test_parse_measurement_unknown():
    assert parse_measurement("feeling fine") == []
    assert parse_measurement("") == []


def test_sparkline_shape():
    s = sparkline([1, 2, 3, 4, 5])
    assert len(s) == 5 and s[0] != s[-1]
    assert sparkline([5, 5, 5]) == "▄▄▄"
    assert sparkline([]) == ""


def test_vitals_trend(tl):
    now = time.time()
    tl.log("measurement", "morning BP", value="120/80", unit="mmHg",
           ts=now - 86400)
    tl.log("measurement", "morning BP", value="130/85", unit="mmHg",
           ts=now)
    trends = {t.metric: t for t in tl.vitals_trend()}
    assert trends["blood_pressure_systolic"].latest == 130.0
    assert trends["blood_pressure_systolic"].average == 125.0
    assert len(trends["blood_pressure_systolic"].points) == 2


def test_symptom_stats_trend(tl):
    now = time.time()
    tl.log("symptom", "headache", severity=2, ts=now - 86400 * 5)
    tl.log("symptom", "headache", severity=2, ts=now - 86400 * 4)
    tl.log("symptom", "headache", severity=4, ts=now - 86400 * 1)
    tl.log("symptom", "headache", severity=5, ts=now)
    stats = tl.symptom_stats()
    assert len(stats) == 1
    assert stats[0].count == 4
    assert stats[0].trend == "↑"  # severity rising


def test_search(tl):
    tl.log("note", "ate jollof rice for lunch")
    tl.log("note", "walked the dog")
    hits = tl.search("jollof")
    assert len(hits) == 1 and "jollof" in hits[0].text
    assert tl.search("nope-nothing") == []


def test_logging_streak(tl):
    now = time.time()
    for i in range(3):
        tl.log("note", f"day {i}", ts=now - 86400 * i)
    assert tl.logging_streak() >= 3


def test_export_report_sections(tl):
    now = time.time()
    tl.log("symptom", "headache", severity=3, ts=now - 86400)
    tl.log("measurement", "BP", value="120/80", unit="mmHg", ts=now)
    tl.log("medication", "took vitamin D", ts=now)
    rep = tl.export_report(days=7)
    assert "## Symptoms" in rep
    assert "## Measurements" in rep
    assert "## Medications" in rep
    assert "## Full log" in rep


# ── coach: dynamic sleep need, contributors, history ──────────────────────

def test_sleep_need_dynamic():
    assert sleep_need() == 8.0
    assert sleep_need(recent_avg_hours=6.0) > 8.0  # debt repayment
    assert sleep_need(workouts_48h=2) > 8.0  # strain premium
    assert 7.0 <= sleep_need(recent_avg_hours=3.0,
                             workouts_48h=5) <= 10.0  # capped


def test_sleep_score_vs_need():
    assert sleep_score_vs_need(8.2, 8.0) == 100.0
    mid = sleep_score_vs_need(5.3, 9.0)
    assert 0 < mid < 100
    # proportional: worse sleep → worse score
    assert sleep_score_vs_need(4.0, 9.0) < mid


def test_readiness_contributors_and_explain(tmp_path):
    src = FakeSource(metrics=[_metric(52.0) for _ in range(30)],
                     sleep=[_sleep(8.2) for _ in range(7)],
                     workouts=[])
    r = _coach(src, tmp_path).readiness(log=False)
    assert r.has_data and r.level == "high"
    assert set(r.contributors) == {"hrv", "sleep", "strain"}
    text = r.explain()
    guard_coaching(text)
    assert "hrv" in text and "weight" in text


def test_readiness_dynamic_need_in_reasons(tmp_path):
    src = FakeSource(metrics=[_metric(52.0) for _ in range(30)],
                     sleep=[_sleep(8.2) for _ in range(7)],
                     workouts=[])
    r = _coach(src, tmp_path).readiness(log=False)
    assert any("your need" in reason for reason in r.reasons)


def test_readiness_history_and_trend(tmp_path):
    src = FakeSource(metrics=[_metric(52.0) for _ in range(30)],
                     sleep=[_sleep(8.2) for _ in range(7)],
                     workouts=[])
    c = _coach(src, tmp_path)
    c.readiness(log=True)
    time.sleep(0.02)
    c.readiness(log=True)
    hist = c.readiness_history(days=7)
    assert len(hist) >= 1
    assert all(h.level == "high" for h in hist)
    trend = c.readiness_trend(days=7)
    guard_coaching(trend)


def test_progress_bar():
    assert progress_bar(100).count("█") == 12
    assert progress_bar(0).count("░") == 12
    assert "█" in progress_bar(62) and "░" in progress_bar(62)


def test_readiness_band_advice():
    assert "green light" in readiness_band_advice("high")
    assert "light day" in readiness_band_advice("low")


# ── drift: chronic window, illness watch, escalation ───────────────────────

def test_chronic_sleep_debt_signal(tl, dm):
    now = time.time()
    for i in range(14):
        tl.log("sleep", "slept 5h", ts=now - 86400 * (14 - i))
    report = dm.check(now=now)
    assert report is not None
    kinds = {s.kind for s in report.signals}
    assert "chronic_sleep_debt" in kinds
    assert "sleep_debt" in kinds  # acute still fires


def test_chronic_needs_minimum_data(tl, dm):
    now = time.time()
    tl.log("sleep", "slept 5h", ts=now - 86400)
    tl.log("sleep", "slept 5h", ts=now - 86400 * 2)
    report = dm.check(now=now)
    kinds = {s.kind for s in report.signals} if report else set()
    assert "chronic_sleep_debt" not in kinds  # too thin


def test_illness_watch_needs_two_signals():
    assert illness_watch(hrv_ratio=0.80, rhr_delta_bpm=6,
                         sleep_hours=5.0).active
    # single signal → quiet
    assert not illness_watch(hrv_ratio=0.80).active
    assert not illness_watch(rhr_delta_bpm=8).active
    w = illness_watch(hrv_ratio=0.80, rhr_delta_bpm=6, sleep_hours=5.0)
    guard_coaching(w.detail)


def test_projected_recovery():
    assert "back at baseline" in projected_recovery(
        [7.5, 8.0, 8.2, 8.3], target=8.0)
    assert "wrong way" in projected_recovery([8.0, 7.0, 6.0, 5.0],
                                             target=8.0)
    improving = projected_recovery([5.0, 5.5, 6.0, 6.5], target=8.0)
    assert "baseline" in improving


def test_consecutive_act_escalation(tl, dm):
    now = time.time()
    for i in range(3):
        tl.log("sleep", "slept 4h", ts=now - 86400 * (3 - i))
        tl.log("mood", "rough", severity=1, ts=now - 86400 * (3 - i))
    for day_offset in range(3):
        dm.check(now=now - 86400 * (2 - day_offset))
    assert dm.consecutive_act_days(now=now) >= 2
    note = dm.escalation_note(now=now)
    # only fires at 3+
    assert isinstance(note, str)


def test_recovery_plan_chronic_action(tl, dm):
    now = time.time()
    for i in range(14):
        tl.log("sleep", "slept 5h", ts=now - 86400 * (14 - i))
    report = dm.check(now=now)
    kinds = {a.kind for a in recovery_plan(report)}
    assert "reset_week" in kinds  # structural, not a bad weekend


# ── training: volume, PRs, auto-regulation, deload ─────────────────────────

def _push_workout(day, load="60kg"):
    return Workout(
        date=day.isoformat(), kind="push",
        exercises=[Exercise(key="bench", name="Bench press", sets=4,
                            reps="8", rest_secs=120, intensity="hard",
                            load=load),
                   Exercise(key="ohp", name="Overhead press", sets=3,
                            reps="8", rest_secs=90, intensity="hard",
                            load="40kg")],
        duration_min=45, intensity="hard")


def test_weekly_sets_and_volume_bands(tc):
    tc.log_workout(_push_workout(date.today()), completed=True, rpe=7)
    sets = tc.weekly_sets()
    assert sets["chest"] >= 4  # bench 4 sets → chest
    assert sets["shoulders"] >= 3
    report = tc.volume_report()
    guard_coaching(report)
    assert "sweet spot" in report or "add volume" in report


def test_personal_records_autotracked(tc):
    tc.log_workout(_push_workout(date.today(), load="60kg"),
                   completed=True, rpe=8)  # rpe 8 → no auto-bump
    prs = tc.personal_records()
    bench = [p for p in prs if p.exercise_key == "bench"]
    assert bench and bench[0].load == "60kg"
    text = tc.format_prs()
    guard_coaching(text)
    assert "Bench press" in text


def test_autoregulation_double_bump(tc):
    w = _push_workout(date.today())
    tc.log_workout(w, completed=True, rpe=6, set_rpes={"bench": 6})
    tc.log_workout(_push_workout(date.today()), completed=True, rpe=6,
                   set_rpes={"bench": 6})
    # easy twice → double bump: 60 → 63
    prs = {p.exercise_key: p for p in tc.personal_records()}
    assert prs["bench"].load == "63kg"
    assert "extra" in tc.progression_advice("bench")


def test_autoregulation_hold(tc):
    w = _push_workout(date.today())
    tc.log_workout(w, completed=True, rpe=9, set_rpes={"bench": 9})
    tc.log_workout(_push_workout(date.today()), completed=True, rpe=9,
                   set_rpes={"bench": 9})
    prs = {p.exercise_key: p for p in tc.personal_records()}
    assert prs["bench"].load == "60kg"  # held, no bump
    assert "holding" in tc.progression_advice("bench")


def test_muscle_groups_for():
    assert "chest" in muscle_groups_for("bench")
    assert "legs" in muscle_groups_for("squat")
    assert muscle_groups_for("nope") == ()


def test_weak_point_notes(tc):
    class FakeIssue:
        checkpoint = "knees"

    class FakeAnalysis:
        issues = [FakeIssue(), FakeIssue()]
    text = tc.weak_point_notes([FakeAnalysis()])
    guard_coaching(text)
    assert "knees" in text and "banded lateral walks" in text


def test_train_chat_new_verbs(tc):
    assert "sweet spot" in control_train("volume", coach=tc) or \
        "add volume" in control_train("volume", coach=tc)
    tc.log_workout(_push_workout(date.today(), load="60kg"),
                   completed=True, rpe=7)
    assert "Bench press" in control_train("prs", coach=tc)
    assert "deload" in control_train("deload", coach=tc).lower() or \
        "sustainable" in control_train("deload", coach=tc)


# ── challenges: freeze, monthly, PBs, sync ─────────────────────────────────

def test_streak_freeze_earns_and_spends(cs):
    t0 = time.time()
    cs.checkin("amy", now=t0 - 86400 * 8)
    for i in range(7, 0, -1):
        cs.checkin("amy", now=t0 - 86400 * i)
    assert cs.streak("amy") == 8
    assert cs.freezes("amy") == 1  # earned at 7
    # miss a day → freeze consumed, streak survives
    cs.checkin("amy", now=t0 + 86400)
    assert cs.streak("amy") == 9
    assert cs.freezes("amy") == 0


def test_streak_resets_without_freeze(cs):
    t0 = time.time()
    cs.checkin("bob", now=t0 - 86400 * 3)
    cs.checkin("bob", now=t0 - 86400 * 2)
    cs.checkin("bob", now=t0)  # 2-day gap, no freeze
    assert cs.streak("bob") == 1


def test_monthly_challenge(cs):
    c = cs.monthly_challenge("distance", created_by="amy")
    assert c is not None
    assert "Distance Challenge" in c.name
    assert c.badge.startswith("monthly:")
    assert c.proof_method == "honor"


def test_personal_best_tracked(cs):
    c = cs.create_challenge("squats", "workout-count", 7, 5)
    cs.join_challenge(c.id, "amy")
    cs.log_workout(c.id, "amy")
    cs.log_workout(c.id, "amy")
    assert cs.personal_best(c.id, "amy") == 2


def test_board_progress_bar(cs):
    c = cs.create_challenge("pushups", "workout-count", 7, 4)
    cs.join_challenge(c.id, "amy")
    cs.log_workout(c.id, "amy")
    ctx = type("C", (), {"challenge_store": cs})()
    out = control_challenge(f"board {c.id}", context=ctx, sender="amy")
    assert "█" in out and "1/4" in out


def test_challenge_chat_new_verbs(cs):
    out = control_challenge("monthly strength", sender="amy")
    assert "Strength Challenge" in out
    out = control_challenge("streak", context=type(
        "C", (), {"challenge_store": cs})(), sender="zed")
    assert "freeze" in out or "streak" in out


def test_sync_from_training(cs, tc):
    c = cs.create_challenge("lift", "workout-count", 30, 5)
    cs.join_challenge(c.id, "owner")
    tc.log_workout(_push_workout(date.today()), completed=True, rpe=7)
    synced = cs.sync_from_training(coach=tc, member="owner")
    assert c.id in synced
    assert cs.progress(c.id, "owner")[0] >= 1


# ── form: quantitative hooks (honest without mediapipe) ────────────────────

def test_joint_angle_math():
    # right angle at B
    assert joint_angle((0, 0, 1), (0, 1, 1), (1, 1, 1)) == pytest.approx(
        90.0, abs=0.5)
    # straight line
    assert joint_angle((0, 0, 1), (1, 0, 1), (2, 0, 1)) == pytest.approx(
        180.0, abs=0.5)
    assert joint_angle((0, 0, 1), (0, 0, 1), (1, 1, 1)) is None


def test_angle_hooks_and_rules():
    assert set(ANGLE_HOOKS) == {"knee_angle", "hip_angle", "spine_angle",
                                "ankle_angle"}
    for movement in ("squat", "deadlift", "push-up", "plank", "lunge"):
        assert movement in ANGLE_RULES
        for hook, triple, lo, hi, cp, desc in ANGLE_RULES[movement]:
            assert hook in ANGLE_HOOKS and lo < hi and cp


def test_mediapipe_absent_is_honest():
    # mediapipe is not installed in this env → quantitative path off,
    # qualitative seam still honest
    assert mediapipe_available() is False
    assert count_reps([], "squat") == 0


def test_analyze_form_without_vision_is_honest(fs):
    a = analyze_form("/nonexistent/video.mp4", "squat", vision=lambda p,
                     q: (_ for _ in ()).throw(RuntimeError("nope")),
                     store=fs)
    assert not a.available
    assert "couldn't analyze" in a.format() or "no video" in a.note


def test_form_trend_needs_two(fs):
    assert form_trend(fs, "squat") == ""
    a = analyze_form("/nonexistent/x.mp4", "squat",
                     vision=lambda p, q: "knees: PASS — fine")
    fs.save(a)
    assert form_trend(fs, "squat") == ""  # still only one


def test_form_trend_two_points(fs):
    import json as _json
    from nomorals.health.form import FormAnalysis
    for score in (0.5, 0.75):
        fs.save(FormAnalysis(id=f"f{score}", exercise="squat",
                             available=True, score=score, band="good"))
    text = form_trend(fs, "squat")
    assert "↑" in text and "50%" in text and "75%" in text


def test_control_form_trend_verb(fs):
    out = control_form("trend squat", store=fs)
    assert "not enough" in out or "trend" in out


# ── previsit: clarifications, context routing, handover ────────────────────

def test_clarify_questions():
    qs = clarify_questions(["headache"])
    assert len(qs) <= 4
    assert any("how long" in q for q in qs)
    assert any("1–10" in q for q in qs)
    assert any("headache" in q or "vision" in q for q in qs)
    # general probe always present
    assert any("chest pain" in q for q in qs)


def test_answer_clarifications_merges_red_flags():
    qs = clarify_questions(["headache"])
    probe = next(q for q in qs if "vision" in q)
    merged = answer_clarifications(["headache"], {probe: "yes"})
    assert any("reported:" in s for s in merged)
    # yes carries into triage → emergency
    r = triage_route(merged)
    assert r.level == "emergency"


def test_answer_clarifications_severity_duration():
    qs = clarify_questions(["cough"])
    dur_q = next(q for q in qs if "how long" in q)
    sev_q = next(q for q in qs if "1–10" in q)
    merged = answer_clarifications(
        ["cough"], {dur_q: "3 days", sev_q: "8"})
    assert any("3 days" in s for s in merged)
    assert any("8/10" in s for s in merged)


def test_route_with_context_recurrence_bump(tl):
    now = time.time()
    for i in range(3):
        tl.log("symptom", "headache, mild", severity=2,
               ts=now - 86400 * (10 - i))
    base = triage_route(["mild headache"])
    assert base.level == "self_care"
    routed = route_with_context(["mild headache"], timeline=tl)
    assert routed.level == "routine_care"  # recurrence routes up
    assert any("logged this 3" in r for r in routed.reasons)


def test_route_confidence_on_format():
    r = route_with_context(["chest pain"])
    assert r.level == "emergency" and r.confidence == "high"
    text = format_route(r)
    assert "confidence" in text
    assert "112" in text  # crisis resources always visible


def test_handover_report(tl):
    now = time.time()
    tl.log("symptom", "headache", severity=3, ts=now - 86400)
    tl.log("measurement", "BP", value="120/80", unit="mmHg", ts=now)
    tl.log("medication", "took vitamin D", ts=now)
    rep = handover_report(["headache"], timeline=tl)
    assert "# Care handover" in rep
    assert "## Symptom history" in rep
    assert "## Recent measurements" in rep
    assert "## Medications" in rep
    assert "## Questions I want to ask" in rep


# ── nutrition: macros, store, day card, water, repeat ──────────────────────

def test_nigerian_foods_have_macros():
    e = NIGERIAN_FOODS["jollof rice"]
    assert e.protein_g[0] > 0 and e.carbs_g[0] > 0 and e.fat_g[0] > 0
    assert e.protein_g[0] <= e.protein_g[1]
    # every entry has macro ranges
    for name, entry in NIGERIAN_FOODS.items():
        assert entry.protein_g[0] <= entry.protein_g[1], name
        assert entry.fat_g[0] <= entry.fat_g[1], name


def test_meal_macro_totals():
    from nomorals.health.nutrition import MealLog
    items = [MealItem(name="jollof rice", cal_low=450, cal_high=700,
                      protein_g=(12, 18), carbs_g=(70, 95),
                      fat_g=(15, 25))]
    log = MealLog(items=items, answers={}, cal_low=450, cal_high=700)
    mt = log.macro_totals
    assert mt["protein"] == (12, 18)
    assert "protein" in log.summary()


def test_oil_answer_bumps_fat(tl, ns):
    d = MealDraft(photo_path="x")
    d.items.append(MealItem(name="jollof rice", cal_low=450, cal_high=700,
                            known=True, oily=True,
                            protein_g=(12, 18), carbs_g=(70, 95),
                            fat_g=(15, 25)))
    log = complete_meal(d, {"oil": "yes", "portion": "medium"},
                        store=ns, timeline=tl)
    assert log.macro_totals["fat"][0] > 15  # invisible fats added


def test_nutrition_store_roundtrip(ns):
    d = MealDraft(photo_path="x")
    d.items.append(MealItem(name="suya", cal_low=250, cal_high=450,
                            protein_g=(25, 35), carbs_g=(3, 6),
                            fat_g=(15, 25)))
    log = complete_meal(d, {"portion": "medium"}, store=ns,
                        timeline=None)
    day = log_water(0, store=ns)  # ensure store path exists
    meals = ns.meals_on(daily_totals(ns).day)
    assert len(meals) == 1
    assert meals[0]["cal_low"] == 250


def test_daily_totals_and_water(ns):
    day = daily_totals(ns).day
    ns.log_water(500, day=day)
    ns.log_water(250, day=day)
    card = daily_totals(ns, day=day)
    assert card.water_ml == 750
    text = card.format()
    assert "750/2500ml" in text
    assert "tracking only" in text


def test_log_water_message(ns):
    msg = log_water(500, store=ns)
    assert "+500ml" in msg and "2500ml" in msg


def test_repeat_meal(ns):
    assert not repeat_meal(ns).ok  # nothing yet
    d = MealDraft(photo_path="x")
    d.items.append(MealItem(name="suya", cal_low=250, cal_high=450,
                            protein_g=(25, 35), carbs_g=(3, 6),
                            fat_g=(15, 25)))
    complete_meal(d, {"portion": "medium"}, store=ns, timeline=None)
    again = repeat_meal(ns)
    assert again.ok and again.items[0].name == "suya"


def test_daycard_format_empty(ns):
    card = DayCard(day="2026-01-01")
    assert "0 meal(s)" in card.format()


# ── patterns: stats, map, best/worst, lag ──────────────────────────────────

def _mood_timeline(tl, now):
    sevs = [2, 2, 3, 4, 2, 4, 5]
    for i, sev in enumerate(sevs):
        tl.log("mood", f"day {i}", severity=sev, ts=now - 86400 * (7 - i))


def test_mood_stats(tl):
    now = time.time()
    _mood_timeline(tl, now)
    s = mood_stats(tl, days=10)
    assert s.entries == 7
    assert s.low_days == 3 and s.good_days == 3
    assert s.best_day and s.worst_day
    text = s.format()
    assert "average" in text and "best" in text


def test_mood_map(tl):
    now = time.time()
    _mood_timeline(tl, now)
    text = mood_map(tl, weeks=4)
    assert "mood map" in text
    assert "🟥" in text or "🟩" in text  # pixels rendered


def test_best_worst_days(tl):
    now = time.time()
    _mood_timeline(tl, now)
    tl.log("sleep", "slept 8h", ts=now - 86400)
    text = best_worst_days(tl, days=10)
    assert "best days" in text and "toughest days" in text


def test_detect_lag_patterns_conservative(tl):
    # thin data → no claims
    assert detect_lag_patterns(tl, days=30) == []


def test_detect_lag_patterns_alcohol(tl):
    now = time.time()
    # 4 low-mood days, 3 preceded by alcohol mentions
    for i, sev in enumerate([2, 2, 2, 2, 4]):
        tl.log("mood", f"d{i}", severity=sev, ts=now - 86400 * (5 - i))
    for off in (5, 4, 3):
        tl.log("note", "had beer with friends", ts=now - 86400 * off)
    pats = detect_lag_patterns(tl, days=10)
    assert any("alcohol" in p.description for p in pats)


def test_detect_patterns_factor_library(tl):
    now = time.time()
    for i in range(4):
        tl.log("mood", "low", severity=2, ts=now - 86400 * (5 - i))
        tl.log("note", "deadline overtime stressful day",
               ts=now - 86400 * (5 - i))
    pats = detect_patterns(tl, days=10)
    assert any("stressful" in p.description for p in pats)
    text = format_patterns(pats)
    assert "💚" in text  # crisis resources ride along with low mood


def test_format_patterns_empty():
    assert "no strong patterns" in format_patterns([])
