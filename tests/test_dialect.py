"""Ekiti/Ilawe Ekiti dialect tutoring (build-map #47). All offline."""
import pytest

from nomorals.learn.dialect import (
    DialectUnsupported,
    EkitiTutor,
    PhonemeFeedback,
    arm_check,
    consume_check,
    detect_dialect_detail,
    get_tutor,
    pending_check,
    suggest_ekiti_fix,
    TONE_GUIDE,
    YORUBA_MINIMAL_PAIRS,
)


def test_detect_ekiti_markers():
    d = detect_dialect_detail("mi lọ sí ọjà")
    assert d["label"] == "ekiti"
    assert d["ekiti_score"] > 0


def test_detect_standard_yoruba():
    d = detect_dialect_detail("mo lọ sí ọjà")
    assert d["label"] == "yoruba"
    assert d["std_score"] > 0


def test_detect_other():
    d = detect_dialect_detail("hello how are you")
    assert d["label"] == "other"


def test_detect_empty_never_raises():
    d = detect_dialect_detail("")
    assert d["label"] == "other"


def test_suggest_ekiti_fix():
    fix = suggest_ekiti_fix("mo lọ sí ibẹ̀")
    assert fix is not None
    assert "mi lọ" in fix
    assert "mo lọ" in fix


def test_no_fix_for_possessive_mi():
    # "owo mi" (my money) is possessive — must NOT be "corrected"
    assert suggest_ekiti_fix("owo mi pọ̀") is None


def test_no_fix_for_ekiti_already():
    assert suggest_ekiti_fix("mi lọ sí ọjà") is None


def test_converse_corrects_gently():
    tutor = EkitiTutor()
    turn = tutor.converse("mo fẹ́ lọ sí ọjà")
    assert turn.correction
    assert "mi" in turn.correction
    assert turn.dialect == "yoruba"


def test_converse_rewards_ekiti():
    tutor = EkitiTutor()
    turn = tutor.converse("mi lọ sí ọjà")
    assert turn.correction == ""
    assert turn.dialect == "ekiti"


def test_converse_tracks_mastery():
    tutor = EkitiTutor()
    before = tutor.mastery.snapshot()["particles (mi vs mo)"]
    tutor.converse("mi lọ")
    after = tutor.mastery.snapshot()["particles (mi vs mo)"]
    assert after >= before


def test_debate_takes_opposing_side_and_escalates():
    tutor = EkitiTutor()
    r1 = tutor.debate("school")
    r2 = tutor.debate("school")
    assert r1.debate_round == 1
    assert r2.debate_round == 2
    assert r1.response != r2.response


def test_debate_needs_topic():
    with pytest.raises(Exception):
        EkitiTutor().debate("")


def test_assess_pronunciation_accuracy():
    tutor = EkitiTutor()
    report = tutor.assess_pronunciation("mi lọ sí ọjà", "mi lọ sí ọjà")
    assert report.accuracy == 1.0
    report2 = tutor.assess_pronunciation("mi lọ sí ilé", "mi lọ sí ọjà")
    assert report2.accuracy == 0.75
    assert any(not w.ok for w in report2.words)


def test_assess_pronunciation_tonal_note():
    tutor = EkitiTutor()
    # tonal near-miss: same letters, wrong tone marks
    report = tutor.assess_pronunciation("igba", "igbá")
    missed = [w for w in report.words if not w.ok]
    assert missed and missed[0].tonal_note


def test_assess_pronunciation_never_raises():
    tutor = EkitiTutor()
    report = tutor.assess_pronunciation("", "")
    assert report.accuracy == 0.0


def test_phoneme_feedback_honest():
    tutor = EkitiTutor()
    fb = tutor.phoneme_feedback("mi lọ", "mi lọ sí")
    assert isinstance(fb, PhonemeFeedback)
    assert fb.precision == "word"  # NOT faked as phoneme-level
    assert fb.phonemes == []
    assert "research-grade" in fb.note


def test_reference_audio_uses_private_xtts():
    calls = {}

    class FakeTTS:
        def speak(self, text, voice_name=None, out_path="", audience=None):
            calls.update(text=text, audience=audience)
            return {"path": "/tmp/ref.wav", "backend": "xtts"}

    tutor = EkitiTutor(tts=FakeTTS())
    result = tutor.reference_audio("mi lọ sí ọjà")
    assert calls["audience"] == "private"  # XTTS allowed: owner's own use
    assert result["dialect"] == "ekiti"
    assert result["path"] == "/tmp/ref.wav"


def test_reference_audio_needs_text():
    with pytest.raises(Exception):
        EkitiTutor().reference_audio("")


def test_non_ekiti_dialect_honest():
    with pytest.raises(DialectUnsupported, match="[Ee]kiti first"):
        EkitiTutor(dialect="hausa")
    with pytest.raises(DialectUnsupported, match="[Ee]kiti first"):
        EkitiTutor(dialect="igbo")


def test_unknown_dialect_rejected():
    with pytest.raises(DialectUnsupported):
        EkitiTutor(dialect="klingon")


def test_arm_and_consume_check():
    msg = arm_check("chat-1", "mi lọ sí ọjà")
    assert "voice note" in msg
    assert pending_check("chat-1") is not None
    scored = consume_check("chat-1", "mi lọ sí ọjà")
    assert scored is not None
    assert "100%" in scored
    # consumed — second call finds nothing
    assert consume_check("chat-1", "mi lọ") is None


def test_arm_check_needs_text():
    with pytest.raises(Exception):
        arm_check("chat-1", "")


def test_consume_check_scores_misses():
    arm_check("chat-2", "mi lọ sí ọjà")
    scored = consume_check("chat-2", "mi lọ sí ilé")
    assert "75%" in scored
    assert "ọjà" in scored


def test_get_tutor_per_chat():
    t1 = get_tutor("chat-a")
    t2 = get_tutor("chat-a")
    t3 = get_tutor("chat-b")
    assert t1 is t2
    assert t1 is not t3


def test_tone_guide_real_content():
    assert "HIGH" in TONE_GUIDE
    assert "MID" in TONE_GUIDE
    assert "LOW" in TONE_GUIDE
    assert len(YORUBA_MINIMAL_PAIRS) >= 3
    forms = [f for f, _, _ in YORUBA_MINIMAL_PAIRS]
    assert "igbá" in forms and "igba" in forms and "ìgba" in forms
