"""WAEC/JAMB curriculum alignment + course generation (build-map #48)."""
import pytest

from nomorals.learn.curriculum import (
    Course,
    CourseScope,
    build_course,
    curriculum_order,
    find_topic,
    scoping_prompt,
    storyboard,
    subject_topics,
    syllabus_code,
    SYLLABI,
    SUBJECTS,
)
from nomorals.learn.tutor import TutorSession


# ── syllabus data ────────────────────────────────────────────────────────────

def test_six_subjects_present():
    assert set(SUBJECTS) == {"Physics", "Chemistry", "Biology",
                             "Mathematics", "English", "Economics"}
    for subject in SUBJECTS:
        assert len(SYLLABI[subject]) >= 10, subject


def test_waves_lookup():
    hit = find_topic("waves")
    assert hit is not None
    assert hit.subject == "Physics"
    assert hit.code.startswith("WAEC Physics 4.")


def test_fuzzy_matching():
    assert find_topic("quadratic").code == "WAEC Maths 2.2"
    assert find_topic("photosynthesis").code == "WAEC Biology 2.1"
    assert find_topic("balance of payments").code == "WAEC Economics 10.2"
    assert find_topic("zzzznothing") is None
    assert find_topic("") is None


def test_fuzzy_subject_scoped():
    hit = find_topic("cells", subject="Biology")
    assert hit is not None and hit.subject == "Biology"


def test_syllabus_code_format():
    hit = find_topic("wave motion")
    assert hit is not None
    code = syllabus_code(hit)
    assert code.startswith("WAEC Physics 4.1")
    assert "Wave Motion" in code


def test_jamb_tags():
    # JAMB-heavy topics are tagged.
    vectors = find_topic("vectors", subject="Mathematics")
    assert vectors is not None and vectors.jamb
    code = syllabus_code(vectors)
    assert "JAMB" in code
    # Plain WAEC topics are not tagged.
    motion = find_topic("motion", subject="Physics")
    assert motion is not None and not motion.jamb


def test_curriculum_order_is_syllabus_order():
    topics = curriculum_order("Physics")
    codes = [t.code for t in topics]
    assert codes == sorted(codes, key=lambda c: [int(x) for x in
                                                 c.split()[-1].split(".")])
    assert subject_topics("nope") == []


# ── scoping (never from a bare prompt) ───────────────────────────────────────

def test_scoping_questions_asked():
    scope = CourseScope(topic="quadratic equations")
    prompt = scoping_prompt(scope)
    assert "3 quick questions" in prompt
    assert "/course set" in prompt
    assert "/course build" in prompt


def test_scope_validation():
    with pytest.raises(ValueError):
        CourseScope(topic="x", level="igcse")
    with pytest.raises(ValueError):
        CourseScope(topic="x", depth="phd")


def test_storyboard_needs_scope_not_bare_prompt():
    # storyboard() requires a CourseScope — there is no bare-prompt path.
    import inspect
    sig = inspect.signature(storyboard)
    assert "scope" in sig.parameters
    assert "prompt" not in sig.parameters


def test_storyboard_structure():
    scope = CourseScope(topic="quadratic equations", subject="Mathematics")
    boards = storyboard(scope)
    assert len(boards) >= 3
    first = boards[0]
    assert {"n", "title", "syllabus_code", "objectives"} <= set(first)
    assert first["syllabus_code"].startswith("WAEC Maths")
    assert len(first["objectives"]) >= 1


def test_storyboard_unknown_topic_honest():
    scope = CourseScope(topic="zzzznothing")
    assert storyboard(scope) == []
    with pytest.raises(ValueError, match="no syllabus topics matched"):
        build_course(scope)


def test_storyboard_quick_depth_caps():
    scope = CourseScope(topic="physics", subject="Physics", depth="quick")
    assert len(storyboard(scope)) <= 6


def test_storyboard_jamb_level_prioritizes():
    scope = CourseScope(topic="maths", subject="Mathematics", level="jamb")
    boards = storyboard(scope)
    assert boards, "jamb maths should storyboard"
    # JAMB-emphasis topics (vectors/matrices/calculus) lead.
    assert "8." in boards[0]["syllabus_code"] or \
           "9." in boards[0]["syllabus_code"]


def test_storyboard_llm_path():
    def fake_llm(system, user):
        return ('[{"title": "Quadratics I", "code": "WAEC Maths 2.2", '
                '"objectives": ["factorise", "use the formula"]}]')
    scope = CourseScope(topic="quadratic equations", subject="Mathematics")
    boards = storyboard(scope, llm_fn=fake_llm)
    assert boards[0]["title"] == "Quadratics I"
    assert boards[0]["syllabus_code"] == "WAEC Maths 2.2"


def test_storyboard_llm_garbage_falls_back():
    scope = CourseScope(topic="quadratic equations", subject="Mathematics")
    boards = storyboard(scope, llm_fn=lambda s, u: "not json at all")
    assert boards and boards[0]["syllabus_code"].startswith("WAEC Maths")


# ── course build ─────────────────────────────────────────────────────────────

def test_build_course_skeleton_honest():
    scope = CourseScope(topic="quadratic equations", subject="Mathematics")
    course = build_course(scope)
    assert course.lessons
    assert all(ls.explainer for ls in course.lessons)
    assert all(ls.quiz for ls in course.lessons)
    # Honest stub marker, not fake prose.
    assert "needs a language model" in course.lessons[0].explainer


def test_build_course_with_llm():
    def fake_llm(system, user):
        if "Design a" in user or "curriculum designer" in system:
            return ('[{"title": "Quadratics I", "code": "WAEC Maths 2.2", '
                    '"objectives": ["factorise"]}]')
        return ('{"explainer": "Quadratics are polynomial equations of degree '
                'two.", "worked_examples": ["x^2-5x+6=0 -> x=2 or 3"], '
                '"quiz": [{"q": "Solve x^2-5x+6=0", "answer": "x=2 or x=3"}]}')
    scope = CourseScope(topic="quadratic equations", subject="Mathematics")
    course = build_course(scope, llm_fn=fake_llm)
    assert "polynomial equations of degree two" in course.lessons[0].explainer
    assert course.lessons[0].quiz[0]["q"] == "Solve x^2-5x+6=0"


def test_course_save_load_roundtrip(tmp_path, monkeypatch):
    import nomorals.learn.curriculum as cur
    monkeypatch.setattr(cur, "courses_dir", lambda: tmp_path)
    scope = CourseScope(topic="quadratic equations", subject="Mathematics")
    course = build_course(scope)
    path = course.save()
    loaded = Course.load(course.id)
    assert loaded is not None
    assert loaded.title == course.title
    assert len(loaded.lessons) == len(course.lessons)
    assert loaded.lessons[0].quiz[0]["q"] == course.lessons[0].quiz[0]["q"]
    assert Course.load("nope") is None


# ── course tutor (grounded) ──────────────────────────────────────────────────

def test_course_tutor_grounded():
    from nomorals.learn.curriculum import course_tutor
    scope = CourseScope(topic="quadratic equations", subject="Mathematics")
    course = build_course(scope)
    session = course_tutor(course)
    assert isinstance(session, TutorSession)
    assert session.source == f"course:{course.id}"
    assert session.citations, "grounded tutor must carry citations"
    assert "WAEC Maths 2.2" in session.topic


def test_course_tutor_empty_raises():
    from nomorals.learn.curriculum import course_tutor
    from nomorals.learn.tutor import TutorError
    course = Course(id="x", title="Empty",
                    scope=CourseScope(topic="zzzznothing"), lessons=[])
    with pytest.raises(TutorError):
        course_tutor(course)


# ── tutor syllabus integration (additive) ────────────────────────────────────

def test_tutor_without_syllabus_unchanged():
    s = TutorSession(topic="fractions", mode="direct")
    assert s.syllabus is None
    turn = s.start()
    assert "📖" not in (turn.feedback + turn.prompt)


def test_tutor_syllabus_string_resolves():
    s = TutorSession(topic="waves", mode="direct", syllabus="waves")
    turn = s.start()
    assert "📖 WAEC Physics 4." in (turn.feedback + turn.prompt)


def test_tutor_syllabus_topic_object():
    hit = find_topic("quadratic equations")
    s = TutorSession(topic="quadratics", mode="direct", syllabus=hit)
    turn = s.start()
    assert "WAEC Maths 2.2" in (turn.feedback + turn.prompt)


def test_tutor_syllabus_unresolvable_keeps_string():
    s = TutorSession(topic="x", mode="direct", syllabus="zzzznothing")
    turn = s.start()  # must not crash
    assert turn is not None
