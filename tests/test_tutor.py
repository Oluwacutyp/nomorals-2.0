"""Socratic tutoring + photo input + mistake notebook (build-map #46)."""
import pytest

from nomorals.learn import (
    MasteryModel,
    MistakeNotebook,
    SocraticEngine,
    TutorError,
    TutorSession,
    card_from_mistake,
    tutor_from_files,
    tutor_from_photo,
)


# ── answer guard: socratic NEVER reveals ──────────────────────────────────────

def test_guard_redacts_verbatim_leak():
    out = SocraticEngine.guard("The answer is photosynthesis, well done!",
                               "photosynthesis")
    assert "photosynthesis" not in out.lower()
    assert "[the answer]" in out


def test_guard_skips_short_answers():
    # Redacting 1-2 char answers would mangle ordinary words.
    assert SocraticEngine.guard("x marks the spot", "x") == "x marks the spot"


def test_guard_case_insensitive():
    out = SocraticEngine.guard("PHOTOSYNTHESIS drives it", "photosynthesis")
    assert "photosynthesis" not in out.lower()


def test_socratic_never_reveals_even_when_llm_slips():
    def leaky_llm(system, user):
        return "The answer is photosynthesis — remember that!"
    sess = TutorSession("plants", mode="socratic",
                        llm_fn=leaky_llm, expected_answer="photosynthesis")
    turn = sess.hint()
    assert "photosynthesis" not in turn.prompt.lower()
    assert turn.revealed is False


def test_socratic_respond_never_reveals():
    sess = TutorSession("2+2", mode="socratic", expected_answer="4")
    # "4" is < 3 chars so guard skips it — use a longer answer instead.
    sess.set_question("capital of France?", "Paris")
    turn = sess.respond("London")
    assert "paris" not in (turn.feedback + turn.prompt).lower()
    assert turn.revealed is False


# ── diagnosis ─────────────────────────────────────────────────────────────────

def test_diagnose_correct_fallback():
    eng = SocraticEngine()
    verdict, _ = eng.diagnose("photosynthesis", "photosynthesis")
    assert verdict == "correct"


def test_diagnose_wrong_fallback():
    eng = SocraticEngine()
    verdict, specifics = eng.diagnose("respiration", "photosynthesis")
    assert verdict == "wrong"
    assert specifics  # says something useful


def test_diagnose_partial_fallback():
    eng = SocraticEngine()
    verdict, specifics = eng.diagnose(
        "photosynthesis uses sunlight", "photosynthesis uses sunlight and water")
    assert verdict == "partial"
    assert "water" in specifics


def test_diagnose_numeric():
    eng = SocraticEngine()
    assert eng.diagnose("42", "42")[0] == "correct"
    assert eng.diagnose("43", "42")[0] == "wrong"


def test_diagnose_empty_student():
    eng = SocraticEngine()
    verdict, _ = eng.diagnose("", "photosynthesis")
    assert verdict == "wrong"


# ── dialogue flow ─────────────────────────────────────────────────────────────

def test_backtrack_on_repeated_struggle():
    sess = TutorSession("fractions", mode="socratic", expected_answer="one half")
    sess.set_question("what is 1/2 + 1/4?", "three quarters")
    t1 = sess.respond("one third")
    t2 = sess.respond("one fifth")
    # Second struggle -> backtrack to prerequisites
    assert "step back" in t2.prompt.lower()


def test_probe_deeper_on_success():
    sess = TutorSession("fractions", mode="socratic")
    sess.set_question("what is 1/2?", "one half")
    turn = sess.respond("one half")
    assert turn.diagnosis == "correct"
    assert "step further" in turn.prompt.lower() or "why" in turn.prompt.lower()


def test_hint_progression():
    sess = TutorSession("fractions", mode="socratic",
                        expected_answer="three quarters")
    h1 = sess.hint()
    h2 = sess.hint()
    h3 = sess.hint()
    assert h1.prompt != h2.prompt != h3.prompt
    for h in (h1, h2, h3):
        assert "three quarters" not in h.prompt.lower()
        assert h.revealed is False


def test_start_socratic_vs_direct():
    soc = TutorSession("fractions", mode="socratic").start()
    assert soc.revealed is False
    d = TutorSession("fractions", mode="direct",
                     expected_answer="a/b").start()
    assert d.revealed is True


def test_direct_mode_gives_answers():
    sess = TutorSession("fractions", mode="direct", expected_answer="a/b")
    turn = sess.respond("c/d")
    assert turn.revealed is True
    assert "a/b" in turn.prompt


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        TutorSession("x", mode="lecture")


# ── mastery ───────────────────────────────────────────────────────────────────

def test_mastery_updates():
    m = MasteryModel(["fractions", "decimals"])
    up = m.update("fractions", correct=True)
    assert up > 0.5
    down = m.update("decimals", correct=False)
    assert down < 0.5
    assert m.weakest() == "decimals"
    assert m.strongest() == "fractions"


def test_mastery_clamped():
    m = MasteryModel(["x"])
    for _ in range(20):
        m.update("x", correct=True)
    assert m.skills["x"] <= 1.0
    for _ in range(20):
        m.update("x", correct=False)
    assert m.skills["x"] >= 0.0


def test_mastery_difficulty_weighting():
    easy = MasteryModel(["a"])
    hard = MasteryModel(["a"])
    easy.update("a", correct=True, difficulty=0.1)
    hard.update("a", correct=True, difficulty=0.9)
    assert hard.skills["a"] > easy.skills["a"]


# ── flashcards / mistake notebook ─────────────────────────────────────────────

def test_card_from_mistake():
    card = card_from_mistake("what is 1/2 + 1/4?", "one third",
                             "three quarters", why="added numerators",
                             topic="fractions")
    assert "one third" in card.front
    assert "three quarters" in card.back
    assert "added numerators" in card.back


def test_notebook_records_and_reviews():
    nb = MistakeNotebook(scheduler=None)
    card = nb.record("q?", "wrong answer", "right answer", topic="math")
    assert nb.count() == 1
    assert nb.review(card.id, 3) is True
    assert nb.review("nope", 3) is False
    notes = nb.notebook()
    assert notes[0]["last_grade"] == 3
    assert notes[0]["question"] == "q?"


def test_notebook_schedules_with_broken_scheduler():
    class Broken:
        def schedule(self, *a): raise RuntimeError("boom")
    nb = MistakeNotebook(scheduler=Broken())
    card = nb.record("q?", "a", "b")  # must not raise
    assert nb.count() == 1


# ── photo input ───────────────────────────────────────────────────────────────

def test_tutor_from_photo_mocked_seer():
    sess = tutor_from_photo("/tmp/prob.png", mode="socratic",
                            seer_fn=lambda p, q: "Solve: 3x + 5 = 20")
    assert isinstance(sess, TutorSession)
    assert "3x + 5 = 20" in sess.topic
    assert sess.source.startswith("photo:")


def test_tutor_from_photo_no_problem():
    with pytest.raises(TutorError):
        tutor_from_photo("/tmp/blank.png", seer_fn=lambda p, q: "")


def test_tutor_from_photo_seer_failure():
    def boom(p, q):
        raise RuntimeError("camera dead")
    with pytest.raises(TutorError):
        tutor_from_photo("/tmp/x.png", seer_fn=boom)


# ── grounded tutoring ─────────────────────────────────────────────────────────

def test_tutor_from_files(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Fractions\n\nA fraction a/b means a parts of b equal parts. "
                 "To add fractions, find a common denominator first.")
    sess = tutor_from_files([str(f)], "fractions", mode="direct")
    assert isinstance(sess, TutorSession)
    assert sess.citations  # cites the file
    assert str(f) in sess.source


def test_tutor_from_files_missing():
    with pytest.raises(TutorError):
        tutor_from_files(["/tmp/does-not-exist-xyz.md"], "x")


def test_tutor_from_files_empty():
    with pytest.raises(TutorError):
        tutor_from_files([], "x")
