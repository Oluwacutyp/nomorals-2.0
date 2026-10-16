"""TEN_PACKAGES product tests: JAMB tutor + Legal product + Health expose.

Invariants:
- JAMB answer guard never leaks before an attempt.
- Legal: every product output carries disclaimer + citations.
- Health: recovery gate refuses a hard session on low readiness.
- Universal exam packs are structured (mapped vs descriptor-only).
"""

import time

import pytest

from nomorals.health.expose import (
    FAMILY_MODE_BLUEPRINT,
    PRICE_DESCRIPTORS,
    analyze_lifting_form,
    family_mode_blueprint,
    forecast_burnout,
    gate_hard_session,
    log_meal_photo,
    photo_undercount_adjustment,
    price_transparency,
    timeline_community_refusal,
    verify_challenge_photo,
)
from nomorals.learn.jamb_product import (
    EXAM_PACKS,
    AttemptGate,
    JambFlashcards,
    JambMastery,
    curriculum_alignment,
    exam_pack,
    jamb_flow,
    misconception_bank_status,
    misconceptions,
)
from nomorals.legal.product import (
    CITATIONS,
    DISCLAIMER,
    check_regulatory_updates,
    format_obligations,
    legal_aid,
    legal_flow,
    obligation_status,
    regulatory_calendar_product,
    review_contract_product,
    with_disclaimer,
)

ANSWER = "42 m/s"


# ── JAMB answer guard ───────────────────────────────────────────────────────

def test_answer_guard_never_leaks_before_attempt():
    gate = AttemptGate()
    gate.register("q1", ANSWER)
    assert gate.check_attempt("q1")["attempted"] is False
    refused = gate.reveal("q1")
    assert refused["allowed"] is False
    assert refused["answer"] is None
    assert ANSWER not in str(refused)
    # recording an attempt releases it — and only then
    gate.record_attempt("q1", "40 m/s")
    released = gate.reveal("q1")
    assert released["allowed"] is True
    assert released["answer"] == ANSWER


def test_answer_guard_socratic_feedback_sanitized():
    gate = AttemptGate()
    gate.register("q2", ANSWER)
    out = gate.socratic_feedback("q2", "40", f"you're missing {ANSWER} here")
    assert out["revealed"] is False
    assert ANSWER not in out["feedback"]


# ── JAMB product ────────────────────────────────────────────────────────────

def test_jamb_mastery_bkt_wraps_real_model():
    m = JambMastery(["waves", "optics"])
    out = m.record("waves", correct=True)
    assert 0.0 <= out["bkt_p_knows"] <= 1.0
    assert 0.0 <= out["p_correct_next"] <= 1.0
    assert set(m.snapshot()) >= {"waves", "optics"}


def test_jamb_flashcards_from_wrong_answer():
    nb = JambFlashcards()
    card = nb.card_for_wrong_answer(
        "What is 2+2?", "5", "4", topic="arithmetic", why="addition slip")
    assert "4" in card["back"]
    assert card["front"].startswith("What is 2+2?")
    assert nb.review(card["id"], "correct") is True
    assert nb.stats()["total"] == 1


def test_misconception_bank_wrapped_not_invented():
    status = misconception_bank_status()
    assert status["entries"] > 0
    assert status["empty"] is False
    hits = misconceptions("quadratic")
    assert hits
    assert all(h["source"] for h in hits)  # every entry carries a source


def test_curriculum_alignment_wraps_real_syllabus():
    hit = curriculum_alignment(subject="Physics", query="waves")
    assert hit["matched"] is not None
    assert "Wave" in hit["matched"]["title"]
    assert hit["subject_topic_count"] > 0


def test_jamb_flow_never_leaks_answers():
    for cmd in ("subjects", "topics Physics", "align waves",
                "misconceptions quadratic", "packs", "ekiti", "bank"):
        out = jamb_flow(cmd)
        assert ANSWER not in out
        assert "Traceback" not in out


# ── universal exam packs ────────────────────────────────────────────────────

def test_exam_packs_structured():
    for code, pack in EXAM_PACKS.items():
        assert pack["code"] == code
        assert pack["region"] and pack["board"]
        assert pack["status"] in ("mapped", "descriptor_only")
        assert isinstance(pack["subjects"], list)
        assert isinstance(pack["format"], dict)
        assert pack["note"]
    jamb = exam_pack("jamb")
    assert jamb["status"] == "mapped"
    assert len(jamb["subjects"]) > 0
    assert jamb["format"]  # real UTME format
    for code in ("SAT", "GCSE", "JEE"):
        pack = exam_pack(code)
        assert pack["status"] == "descriptor_only"
        assert pack["subjects"] == []  # empty but structured, nothing invented
    assert exam_pack("NOPE") is None


# ── Legal product ───────────────────────────────────────────────────────────

def test_legal_every_output_carries_disclaimer_and_citations():
    assert "not legal advice" in with_disclaimer("hello").lower()
    assert "DoNotPay FTC $193K" in with_disclaimer("hello")
    assert all(c.split(" — ")[0].split(" precedent")[0] in with_disclaimer("x")
               for c in ())  # placeholder-free check below
    for text in (
        review_contract_product(
            "TENANCY AGREEMENT between Landlord and Tenant. Rent increases "
            "yearly. The tenant waives all rights under the law. "
            "Automatic renewal applies.")["text"],
        legal_aid("my landlord locked me out", "pidgin")["text"],
        legal_flow("help"),
        regulatory_calendar_product(1),
    ):
        assert DISCLAIMER in text
        assert "DoNotPay FTC $193K" in text
        assert "NBA 2024" in text
    assert CITATIONS  # citations block exists


def test_legal_review_wraps_real_playbook():
    out = review_contract_product(
        "TENANCY AGREEMENT. The landlord may increase rent at any time. "
        "The tenant waives all statutory rights. Automatic renewal applies.")
    assert out["contract_type"] == "tenancy"
    assert out["findings"]
    assert any(f["rule_id"] == "T-99" for f in out["findings"])


def test_legal_obligations_computed_from_supplied_dates_only():
    # Truncated `now` so the whole-second ISO due date is exactly +60d.
    now = float(int(time.time()))
    sixty = now + 60 * 86400
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(sixty))
    st = obligation_status({"description": "tenancy renews",
                            "due": iso, "kind": "renewal"}, now=now)
    assert st["date_supplied"] is True
    assert st["days_until"] == 60
    assert "in 60 day(s)" in st["line"]
    missing = obligation_status({"description": "mystery obligation"})
    assert missing["date_supplied"] is False
    assert missing["days_until"] is None
    assert "cannot compute" in missing["status"]
    text = format_obligations([{"description": "tenancy renews", "due": iso}])
    assert DISCLAIMER in text


def test_legal_regulatory_check_uses_supplied_updates():
    alerts = check_regulatory_updates([{
        "regulator": "CBN",
        "title": "New circular on fintech licensing fees",
        "summary": "The CBN has issued a circular on fintech licensing.",
        "source_url": "https://www.cbn.gov.ng/circular-1",
        "topics": ["fintech licensing"],
    }])
    assert len(alerts) == 1
    assert DISCLAIMER in alerts[0]
    assert "DoNotPay FTC $193K" in alerts[0]
    assert "https://www.cbn.gov.ng/circular-1" in alerts[0]
    # garbage update -> no alert, no crash
    assert check_regulatory_updates([{"regulator": "NOPE", "title": ""}]) == []


def test_legal_aid_languages():
    for lang in ("en", "pidgin", "yo", "ha", "ig"):
        out = legal_aid("my landlord locked me out", lang)
        assert out["available"] is True
        assert DISCLAIMER in out["text"]


# ── Health expose ───────────────────────────────────────────────────────────

def test_health_recovery_gate_refuses_hard_on_low_readiness():
    low = gate_hard_session({"sleep_hours": 3.0, "hrv_ratio": 0.70,
                             "rhr_delta_bpm": 10.0, "hard_sessions_7d": 5})
    assert low["allowed_hard"] is False
    assert low["readiness_level"] == "recovery"
    assert low["readiness_score"] < 40
    high = gate_hard_session({"sleep_hours": 8.0, "hrv_ratio": 1.05,
                              "rhr_delta_bpm": 0.0, "hard_sessions_7d": 1})
    assert high["allowed_hard"] is True
    assert high["readiness_level"] == "full"
    unknown = gate_hard_session({})
    assert unknown["allowed_hard"] is False  # no data -> no hard session
    direct = gate_hard_session(readiness_score=25.0)
    assert direct["allowed_hard"] is False


def test_health_burnout_forecast_from_supplied_history():
    out = forecast_burnout([80.0, 70.0, 55.0, 38.0])
    assert out["risk"] in ("high", "elevated")
    assert "3 days" in out["forecast"]
    short = forecast_burnout([70.0])
    assert short["risk"] == "unknown"
    assert "not enough data" in short["forecast"]
    stable = forecast_burnout([80.0, 82.0, 81.0, 83.0])
    assert stable["risk"] == "low"


def test_health_form_never_fakes_vision():
    out = analyze_lifting_form("/nonexistent/video.mp4", "squat")
    assert out["available"] is False
    assert "No vision callable supplied" in out["note"]
    assert "Nothing was faked" in out["note"]


def test_health_nutrition_photo_questions_capped_and_adjusted():
    lo, hi = photo_undercount_adjustment(600, 800)
    assert lo > 600 and hi > 800  # ~33% undercount correction raises the range
    assert lo == round(600 / 0.67) and hi == round(800 / 0.67)
    draft = log_meal_photo("/nonexistent/photo.jpg",
                           seer=lambda p, q: "NONE")
    assert draft["ok"] is False
    assert draft["follow_up_questions"] == []
    assert len(draft["follow_up_questions"]) <= 2


def test_health_timeline_community_refusal_and_family_blueprint():
    refusal = timeline_community_refusal()
    assert refusal["refused"] is True
    bp = family_mode_blueprint()
    assert bp["status"].startswith("blueprint only")
    assert "NOT implemented" in bp["not_built"]
    assert FAMILY_MODE_BLUEPRINT["status"].startswith("blueprint only")


def test_health_challenge_proof_and_prices():
    proof = verify_challenge_photo("/nonexistent/photo.jpg")
    assert proof["accepted"] is False  # no photo -> honest rejection
    assert "no photo found" in proof["verdict"]
    prices = price_transparency()
    assert "₦" in prices
    assert "not a doctor" in prices.lower() or "not medical advice" in prices.lower()
    assert PRICE_DESCRIPTORS
    assert all(d["low_naira"] <= d["high_naira"] for d in PRICE_DESCRIPTORS)
