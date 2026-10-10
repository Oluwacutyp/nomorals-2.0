"""Sweep tests for nomorals/learn — BKT/HLR mastery, EMT dialogue, FSRS
notebook, Ekiti dialect upgrades, curriculum exam simulation.

All offline. No LLM, no network, no microphone.
"""

import json
import subprocess
import sys
import time

import pytest

from nomorals.learn import (
    BKT_DEFAULTS,
    EXAM_FORMATS,
    MISCONCEPTION_BANK,
    Course,
    CourseScope,
    EkitiTutor,
    ExamSim,
    Expectation,
    FSRS,
    FSRSCardState,
    Lesson,
    MasteryModel,
    Misconception,
    MistakeNotebook,
    QuestionScript,
    SocraticEngine,
    StudyPlan,
    TutorSession,
    answer_scope,
    build_course,
    clear_pending_scope,
    course_to_anki_tsv,
    detect_dialect_detail,
    grade_for_verdict,
    mid_tone_diagnosis,
    misconceptions_for,
    pending_scope,
    set_pending_scope,
    study_plan,
    suggest_ekiti_fix,
    syllabus_coverage,
    tonal_note_for,
    tone_pattern,
    tutor_from_files,
    waec_grade,
)
from nomorals.learn import flashcards as fc_mod
from nomorals.learn import tutor as tutor_mod


# ── BKT mastery ──────────────────────────────────────────────────────────────

def test_bkt_defaults_match_pybkt_readme():
    assert BKT_DEFAULTS["p_learn"] == 0.30
    assert BKT_DEFAULTS["p_guess"] == 0.10
    assert BKT_DEFAULTS["p_slip"] == 0.03
    assert BKT_DEFAULTS["p_init"] == 0.10


def test_bkt_correct_raises_p_knows():
    m = MasteryModel(["algebra"])
    before = m.bkt("algebra")
    after = m.bkt_update("algebra", correct=True)
    assert after > before


def test_bkt_wrong_lowers_p_knows():
    m = MasteryModel(["algebra"])
    m.bkt_update("algebra", correct=True)
    m.bkt_update("algebra", correct=True)
    before = m.bkt("algebra")
    after = m.bkt_update("algebra", correct=False)
    assert after < before


def test_bkt_guess_slip_aware():
    # A correct answer with high guess prob moves P(knows) less than with
    # low guess prob — the model discounts lucky guesses.
    m1, m2 = MasteryModel(["s"]), MasteryModel(["s"])
    m1.set_bkt_params("s", p_guess=0.5)
    m2.set_bkt_params("s", p_guess=0.01)
    assert m1.bkt_update("s", correct=True) < m2.bkt_update("s", correct=True)


def test_predict_correct_bounded():
    m = MasteryModel(["s"])
    m.bkt_update("s", correct=True)
    p = m.predict_correct("s")
    assert 0.0 <= p <= 1.0


def test_bkt_persists_roundtrip(tmp_path):
    m = MasteryModel(["a", "b"])
    m.bkt_update("a", correct=True)
    m.set_bkt_params("b", p_learn=0.5)
    d = m.to_dict()
    m2 = MasteryModel.from_dict(d)
    assert m2.bkt("a") == m.bkt("a")
    assert m2._params("b")["p_learn"] == 0.5


def test_linear_update_backcompat():
    m = MasteryModel(["x"])
    assert m.update("x", correct=True) > 0.5
    assert m.update("x", correct=False) < 0.5
    assert m.weakest() == "x"
    assert "x" in m.snapshot()


# ── HLR forgetting ───────────────────────────────────────────────────────────

def test_hlr_recall_decays_with_time():
    m = MasteryModel(["s"])
    m.halflife_update("s", correct=True, now=1000.0)
    assert m.recall_prob("s", now=1000.0) == pytest.approx(1.0)
    # one half-life later -> 0.5
    h = m._halflife["s"]
    assert m.recall_prob("s", now=1000.0 + h * 86400) == pytest.approx(0.5)


def test_hlr_doubles_on_success_halves_on_failure():
    m = MasteryModel(["s"])
    h1 = m.halflife_update("s", correct=True, now=1.0)
    h2 = m.halflife_update("s", correct=True, now=2.0)
    h3 = m.halflife_update("s", correct=False, now=3.0)
    assert h2 == pytest.approx(h1 * 2)
    assert h3 == pytest.approx(h2 * 0.5)


# ── socratic guard + sanitizer ───────────────────────────────────────────────

def test_guard_redacts_verbatim_and_variants():
    e = SocraticEngine()
    out = e.guard("The answer is 42 m/s, yes 42 m/s!", "42 m/s")
    assert "42 m/s" not in out
    assert "[the answer]" in out
    # quote-wrapped variant
    out2 = e.guard('Say "photosynthesis" now', "photosynthesis")
    assert "photosynthesis" not in out2.lower()


def test_sanitize_feedback_masks_expected_tokens():
    e = SocraticEngine()
    fb = e.sanitize_feedback(
        "Partly there — you're missing: wavelength, frequency.",
        "the wavelength and frequency of the wave")
    assert "wavelength" not in fb.lower()
    assert "frequency" not in fb.lower()


def test_fallback_diagnose_no_longer_leaks_keywords():
    verdict, specifics = SocraticEngine._fallback_diagnose(
        "the wavelength of the wave", "the wavelength and frequency of the wave")
    assert verdict == "partial"
    assert "frequency" not in specifics.lower()


def test_frustration_detected():
    assert SocraticEngine.detect_frustration("I don't get this, ugh")
    assert not SocraticEngine.detect_frustration("the mitochondria is...")


# ── EMT cycle ────────────────────────────────────────────────────────────────

def _script():
    return QuestionScript(
        question="Why does ice float on water?",
        expected_answer="ice is less dense than liquid water",
        skill="density",
        expectations=[
            Expectation("ice is less dense than liquid water"),
            Expectation("density determines whether objects float or sink"),
        ],
        misconceptions=[
            Misconception(pattern="cold things sink",
                          correction="temperature alone doesn't decide "
                                     "floating — density does",
                          probe="Does everything cold sink? What about an "
                                "iceberg vs a cold stone?",
                          source="test"),
        ],
    )


def test_emt_ladder_pump_hint_prompt_assert():
    e = SocraticEngine()
    e.set_script(_script())
    moves = []
    for _ in range(4):
        turn = e.emt_turn("ice is cold")
        moves.append(turn["move"])
    assert moves == ["pump", "hint", "prompt", "assert"], moves


def test_emt_misconception_match_targets_probe():
    e = SocraticEngine()
    e.set_script(_script())
    turn = e.emt_turn("cold things always sink")
    assert turn["misconceptions"] == ["cold things sink"]
    assert "iceberg" in turn["prompt"]


def test_emt_done_when_all_covered():
    e = SocraticEngine()
    e.set_script(_script())
    turn = e.emt_turn("ice is less dense than liquid water and density "
                      "determines whether objects float or sink")
    assert turn["move"] == "done"
    assert turn["missing"] == []


def test_misconception_fuzzy_match():
    m = Misconception(pattern="cold things sink", correction="x")
    assert m.matches("I think cold things sink in water")
    assert not m.matches("density is mass over volume")


# ── session: notebook wiring, report, persistence ────────────────────────────

def test_wrong_answer_recorded_in_notebook():
    nb = MistakeNotebook()
    s = TutorSession("density", llm_fn=None, notebook=nb)
    s.set_question("Why does ice float?", "ice is less dense than water")
    s.respond("because it is cold and heavy")
    assert nb.count() == 1
    card = nb.notebook()[0]
    assert card["correct"] == "ice is less dense than water"


def test_correct_answer_not_recorded():
    nb = MistakeNotebook()
    s = TutorSession("density", notebook=nb)
    s.set_question("Why does ice float?", "ice is less dense than water")
    s.respond("ice is less dense than liquid water")
    assert nb.count() == 0


def test_report_contains_metacognition_sections():
    s = TutorSession("waves", sub_skills=["waves", "optics"])
    s.start()
    s.set_question("What is wavelength?", "distance between crests")
    s.respond("distance between crests")
    rep = s.report()
    assert "Session report" in rep
    assert "weakest skill" in rep
    assert "BKT" in rep


def test_frustrated_student_gets_encouragement():
    s = TutorSession("waves")
    s.set_question("What is wavelength?", "distance between crests")
    turn = s.respond("ugh i don't get this at all")
    assert "that's normal" in turn.feedback.lower()


def test_session_save_load_roundtrip(tmp_path):
    s = TutorSession("waves", sub_skills=["waves"])
    s.start()
    s.set_question("Q?", "A")
    s.respond("wrong answer here")
    path = s.save(tmp_path / "sess.json")
    s2 = TutorSession.load(path)
    assert s2.topic == "waves"
    assert s2.turns == s.turns
    assert s2.expected_answer == "A"
    assert s2.mastery.bkt("waves") == s.mastery.bkt("waves")


def test_next_item_interleaves_due_review():
    nb = MistakeNotebook()
    card = nb.record("Q", "wrong", "right", topic="t")
    # Lapsed 3 days ago -> interval ~1 day -> due now.
    nb.review(card.id, 1, now=time.time() - 3 * 86400)
    s = TutorSession("t", notebook=nb)
    item = s.next_item()
    assert item["kind"] == "review"
    assert item["card_id"] == card.id


def test_plan_states_objective():
    s = TutorSession("photosynthesis")
    assert "photosynthesis" in s.plan()


# ── FSRS-lite ────────────────────────────────────────────────────────────────

def test_retrievability_is_09_at_stability():
    f = FSRS()
    assert f.retrievability(10.0, 10.0) == pytest.approx(0.9, abs=1e-9)


def test_next_interval_grows_with_stability():
    f = FSRS()
    assert f.next_interval(20.0) > f.next_interval(5.0) > 0


def test_review_success_grows_stability_lapse_shrinks():
    f = FSRS()
    st = FSRSCardState()
    f.review(st, 3, now=1000.0)
    s_good = st.stability
    assert s_good > 0
    st2 = FSRSCardState()
    f.review(st2, 3, now=1000.0)
    f.review(st2, 1, now=2000.0)  # lapse
    assert st2.lapses == 1
    assert st2.stability < s_good


def test_due_for_new_and_overdue_cards():
    f = FSRS()
    assert f.due(FSRSCardState()) is True  # new cards are due
    st = FSRSCardState()
    f.review(st, 3, now=1000.0)
    assert f.due(st, now=1000.0) is False
    assert f.due(st, now=1000.0 + 400 * 86400) is True


def test_grades_clamped():
    f = FSRS()
    st = FSRSCardState()
    f.review(st, 99, now=1.0)
    assert st.reps == 1
    assert 1.0 <= st.difficulty <= 10.0


def test_grade_for_verdict():
    assert grade_for_verdict("wrong") == 1
    assert grade_for_verdict("partial") == 2
    assert grade_for_verdict("correct") == 3
    assert grade_for_verdict("correct", easy=True) == 4


# ── notebook upgrades ────────────────────────────────────────────────────────

def test_due_cards_most_overdue_first():
    nb = MistakeNotebook()
    c1 = nb.record("Q1", "a", "b")
    c2 = nb.record("Q2", "a", "b")
    nb.review(c1.id, 3, now=1000.0)
    nb.review(c2.id, 1, now=1000.0)  # lapse -> tiny stability -> more overdue
    due = nb.due_cards(now=1000.0 + 30 * 86400)
    ids = [c.id for c in due]
    assert ids[0] == c2.id  # lapsed card is more overdue


def test_stats_breakdown():
    nb = MistakeNotebook()
    c = nb.record("Q", "a", "b")
    stats = nb.stats()
    assert stats["total"] == 1
    assert stats["states"]["new"] == 1
    nb.review(c.id, 3, now=time.time())
    stats = nb.stats()
    assert stats["states"]["young"] == 1


def test_anki_tsv_format():
    nb = MistakeNotebook()
    nb.record("What is 2+2?", "5", "4", why="addition", topic="maths")
    tsv = nb.to_anki_tsv(deck="Devon::Test")
    lines = tsv.splitlines()
    assert lines[0] == "#separator:tab"
    assert lines[1] == "#html:true"
    assert lines[2] == "#notetype:Basic"
    assert lines[3] == "#deck:Devon::Test"
    body = [ln for ln in lines if not ln.startswith("#")]
    assert len(body) == 1
    front, back = body[0].split("\t")
    assert "2+2" in front and "You said: 5" in front
    assert "Correct: 4" in back and "<br>" in back


def test_notebook_save_load_roundtrip(tmp_path):
    nb = MistakeNotebook()
    c = nb.record("Q", "a", "b", topic="t", tags=["x"])
    nb.review(c.id, 3, now=1000.0)
    path = nb.save(tmp_path / "nb.json")
    nb2 = MistakeNotebook.load(path)
    assert nb2.count() == 1
    card = nb2.get(c.id)
    assert card.topic == "t" and card.tags == ["x"]
    assert card.fsrs.reps == 1


# ── dialect: tone science ────────────────────────────────────────────────────

def test_tone_pattern_extraction():
    assert tone_pattern("igbá") == "MH"
    assert tone_pattern("ìgbá") == "LH"
    assert tone_pattern("igba") == "MM"
    assert tone_pattern("ìgbà") == "LL"
    assert tone_pattern("igbà") == "ML"


def test_igba_quintuple_corrected():
    from nomorals.learn.dialect import YORUBA_MINIMAL_PAIRS
    table = {form: gloss for form, gloss, _ in YORUBA_MINIMAL_PAIRS}
    assert table["igbá"] == "calabash"      # was wrongly 'garden egg'
    assert table["ìgbá"] == "garden egg"
    assert table["igba"] == "two hundred"
    assert table["ìgbà"] == "time / season"
    assert table["igbà"] == "climbing rope"


def test_nasal_pairs_present():
    from nomorals.learn.dialect import NASAL_MINIMAL_PAIRS
    table = {form: gloss for form, gloss, _ in NASAL_MINIMAL_PAIRS}
    assert table["àdá"] == "cutlass"
    assert table["àdán"] == "bat (animal)"


def test_mid_tone_diagnosis_flags_flattening():
    # Expected MM, heard MH: the final mid sung as high — the Orie 2006b
    # English-speaker pattern.
    note = mid_tone_diagnosis("igbá", "igba")
    assert "mid-tone flattening" in note
    assert mid_tone_diagnosis("igbá", "igbá") == ""


def test_tonal_note_for_nasal():
    note = tonal_note_for("àdán")
    assert "nasal" in note.lower()


def test_pronunciation_reports_tonal_accuracy():
    t = EkitiTutor()
    # Heard HH for expected MM: mid tones flattened to high (Orie 2006b).
    rep = t.assess_pronunciation("ígbá owó", "igba owó")
    assert rep.tonal_accuracy < 1.0  # first word's tones wrong
    assert rep.words[0].tone_miss is True
    assert "mid-tone flattening" in rep.words[0].tonal_note


def test_drill_pair_cycle():
    t = EkitiTutor()
    drill = t.drill_pair("tone")
    assert drill["answer"] in drill["options"]
    assert len(drill["options"]) == 2
    result = t.grade_drill(drill["answer"])
    assert result["correct"] is True
    result2 = t.drill_pair("nasal")
    bad = t.grade_drill("not the answer")
    assert bad["correct"] is False


def test_dialect_detection_and_fix():
    d = detect_dialect_detail("mi lọ sí ilé")
    assert d["label"] == "ekiti"
    fix = suggest_ekiti_fix("mo lọ sí ilé")
    assert fix is not None and "mi lọ" in fix
    assert suggest_ekiti_fix("owo mi") is None  # possessive, not corrected


def test_learn_imports_without_finance_first():
    """Regression: nomorals.learn must import standalone (the lazy
    voice.money import fixed the circular-import crash)."""
    code = ("import sys; sys.path.insert(0, '.');"
            "import nomorals.learn as L;"
            "print(L.EkitiTutor, L.ExamSim, L.FSRS)")
    r = subprocess.run([sys.executable, "-c", code], cwd="/home/hatch/workspace/devon",
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


# ── curriculum: exams, misconceptions, plans ─────────────────────────────────

def test_waec_physics_format_verified():
    fmt = EXAM_FORMATS["WAEC Physics"]
    p1 = fmt["papers"][0]
    assert (p1["questions"], p1["minutes"], p1["marks"]) == (50, 75, 50)
    p3 = fmt["papers"][3]
    assert (p3["questions"], p3["to_answer"], p3["minutes"]) == (3, 2, 165)


def test_jamb_format_verified():
    fmt = EXAM_FORMATS["JAMB"]
    total_q = sum(p["questions"] * p.get("repeat", 1) for p in fmt["papers"])
    assert total_q == 180
    assert fmt["minutes"] == 120
    assert fmt["negative_marking"] is False
    assert fmt["options"] == ("A", "B", "C", "D")


def test_waec_grades():
    assert waec_grade(80) == "A1"
    assert waec_grade(52) == "C6"
    assert waec_grade(20) == "F9"


def test_misconception_bank_sourced():
    assert len(MISCONCEPTION_BANK) >= 10
    assert all(m.source for m in MISCONCEPTION_BANK)
    assert any("Spearman" in m.correction for m in MISCONCEPTION_BANK)


def test_misconceptions_for_query():
    hits = misconceptions_for("spearman rank correlation")
    assert hits and all("Spearman" in h.correction or "rank" in h.pattern
                        for h in hits)


def _mcq_questions():
    return [
        {"q": "2+2?", "options": ["3", "4", "5", "6"], "answer": "4",
         "code": "WAEC Maths 1.1"},
        {"q": "Capital of Nigeria?", "options": ["Lagos", "Abuja", "Kano"],
         "answer": "Abuja", "code": "WAEC English 3.1"},
    ]


def test_exam_sim_mcq_flow():
    sim = ExamSim(exam="JAMB", subject="Use of English",
                  questions=_mcq_questions(), minutes=10)
    info = sim.start()
    assert info["questions"] == 2
    r1 = sim.answer(0, "B")
    assert r1["verdict"] == "correct"
    r2 = sim.answer(1, "A")
    assert r2["verdict"] == "wrong"
    result = sim.finish()
    assert result["score"] == 1.0
    assert result["total"] == 2
    assert "pace" in result
    text = sim.report_text(result)
    assert "1.0/2" in text and "50.0%" in text


def test_exam_sim_theory_partial_credit():
    sim = ExamSim(exam="WAEC Physics", subject="Paper 2B",
                  questions=[{"q": "Define density",
                              "answer": "mass per unit volume of a substance",
                              "code": "WAEC Physics 2.2"}],
                  minutes=10)
    sim.start()
    r = sim.answer(0, "the mass per unit volume of a substance")
    assert r["verdict"] == "correct"
    result = sim.finish()
    assert result["grade_band"] == "A1"


def test_exam_sim_misconception_hits():
    sim = ExamSim(exam="WAEC", subject="Maths",
                  questions=[{"q": "Spearman?", "options": ["a", "b"],
                              "answer": "a", "code": "WAEC Maths 6.2"}],
                  minutes=5)
    sim.start()
    sim.answer(0, "b")
    result = sim.finish()
    assert result["misconception_hits"]


def test_study_plan_structure():
    scope = CourseScope(topic="waves", subject="Physics", level="both",
                        depth="quick", weeks=2)
    course = build_course(scope)  # no LLM -> honest skeleton
    assert len(course.lessons) > 0
    plan = study_plan(course, "2026-12-01", start_date="2026-11-20")
    assert plan.sessions
    dates = [s["date"] for s in plan.sessions]
    assert dates == sorted(dates)
    kinds = [s["kind"] for s in plan.sessions]
    assert kinds[-1] == "mock"
    assert "review" in kinds or len(plan.sessions) < 4
    text = plan.to_text()
    assert "Study plan" in text


def test_study_plan_bad_date():
    scope = CourseScope(topic="waves", subject="Physics")
    course = build_course(scope)
    with pytest.raises(ValueError):
        study_plan(course, "not-a-date")


def test_syllabus_coverage():
    scope = CourseScope(topic="Physics", subject="Physics", depth="quick")
    course = build_course(scope)
    cov = syllabus_coverage(course)
    assert cov["subject"] == "Physics"
    assert cov["total_topics"] > 0
    assert 0 < cov["pct"] <= 100


def test_course_to_anki_tsv():
    scope = CourseScope(topic="waves", subject="Physics", depth="quick")
    course = build_course(scope)
    tsv = course_to_anki_tsv(course)
    assert tsv.startswith("#separator:tab\n#html:true")
    body = [ln for ln in tsv.splitlines() if not ln.startswith("#")]
    assert body and all("\t" in ln for ln in body)


def test_answer_scope_validation():
    set_pending_scope("chat1", CourseScope(topic="waves"))
    msg = answer_scope("chat1", "level", "jamb")
    assert "level = jamb" in msg
    assert pending_scope("chat1").level == "jamb"
    bad = answer_scope("chat1", "level", "nonsense")
    assert "isn't one of" in bad
    unknown = answer_scope("chat1", "bogus", "x")
    assert "unknown field" in unknown
    clear_pending_scope("chat1")
    assert pending_scope("chat1") is None
    assert "no course being scoped" in answer_scope("chat1", "level", "waec")


def test_tutor_from_files_still_works(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Photosynthesis converts light energy to chemical energy "
                 "in chlorophyll.")
    s = tutor_from_files([str(p)], "photosynthesis")
    assert s.citations
    assert s.topic == "photosynthesis"


def test_storyboard_grounded_no_llm():
    from nomorals.learn import storyboard
    boards = storyboard(CourseScope(topic="waves", subject="Physics"))
    assert boards
    assert all(b["syllabus_code"].startswith("WAEC Physics") for b in boards)
    assert storyboard(CourseScope(topic="zzz-no-such-topic")) == []


def test_public_api_surface():
    # Everything the chat layer needs is importable from the package root.
    import nomorals.learn as L
    for name in ["ExamSim", "StudyPlan", "Misconception", "FSRS",
                 "QuestionScript", "Expectation", "tone_pattern",
                 "misconceptions_for", "study_plan", "syllabus_coverage",
                 "course_to_anki_tsv", "persist_session", "restore_session",
                 "EKITI_DIALECT_NOTES", "MINIMAL_PAIR_SETS",
                 "grade_for_verdict", "waec_grade"]:
        assert hasattr(L, name), name
