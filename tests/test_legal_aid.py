"""Tests for #77 — consumer legal aid (plain language, Nigerian languages).

All offline. Covers: landlord lockout, Pidgin/Yoruba/Hausa/Igbo answers,
escalation, disclaimer on every output, grounded citations, billing,
no-advice enforcement, chat control, never-raises.
"""

import os
import tempfile

import pytest

from nomorals.legal.aid import (
    LANGUAGES,
    Answer,
    answer_legal_question,
    bill_answer,
    control_legal,
    corpus_search,
    format_answer,
)
from nomorals.legal.contracts import DISCLAIMER, information_only_check


# ── helpers ────────────────────────────────────────────────────────────

def _render(question, language="en"):
    return format_answer(answer_legal_question(question, language))


# ── scenario: landlord lockout ─────────────────────────────────────────

def test_landlord_lockout_english():
    out = _render("my landlord locked me out, what are my rights?")
    assert "cannot lock you out" in out
    assert "court order" in out
    assert DISCLAIMER in out


def test_landlord_lockout_pidgin():
    out = _render("my landlord don lock me out", "pcm")
    assert "no get right" in out
    assert DISCLAIMER in out


def test_landlord_lockout_yoruba():
    out = _render("my landlord locked me out, what are my rights?", "yo")
    # core answer is translated; disclaimer is shared
    assert "Onílé" in out or "onílé" in out.lower()
    assert DISCLAIMER in out


def test_landlord_lockout_hausa():
    out = _render("my landlord locked me out, what are my rights?", "ha")
    assert "Mai gidan" in out or "mai gidan" in out.lower()
    assert DISCLAIMER in out


def test_landlord_lockout_igbo():
    out = _render("my landlord locked me out, what are my rights?", "ig")
    assert "Onye nwe" in out or "onye nwe" in out.lower()
    assert DISCLAIMER in out


def test_eviction_variant_matches():
    out = _render("landlord wants to evict me without notice")
    assert "proper written notice" in out or "court order" in out


# ── other scenarios ────────────────────────────────────────────────────

def test_wrongful_termination():
    out = _render("my boss sacked me without notice and refused to pay my salary")
    assert "Labour Act" in out
    assert "National Industrial Court" in out
    assert DISCLAIMER in out


def test_defective_goods():
    out = _render("I bought a phone and it is defective, the seller refused to refund me")
    assert "FCCPC" in out
    assert DISCLAIMER in out


def test_arrest():
    out = _render("police arrested my brother yesterday")
    assert "Constitution" in out
    assert "24 hours" in out
    assert DISCLAIMER in out


# ── escalation ─────────────────────────────────────────────────────────

def test_lockout_escalates():
    ans = answer_legal_question("landlord locked me out")
    assert ans.escalate is True
    assert "lawyer" in ans.render().lower()


def test_defective_goods_no_escalation():
    ans = answer_legal_question("seller refused refund for defective item")
    assert ans.escalate is False


def test_unknown_question_escalates():
    ans = answer_legal_question("what is the capital gains tax rate on crypto")
    assert ans.escalate is True
    assert DISCLAIMER in ans.render()


# ── disclaimer + no-advice ─────────────────────────────────────────────

def test_disclaimer_on_every_output():
    for lang in LANGUAGES:
        out = _render("landlord locked me out", lang)
        assert DISCLAIMER in out, f"missing disclaimer for {lang}"


def test_no_advice_phrases():
    for lang in LANGUAGES:
        out = _render("my landlord locked me out", lang)
        hits = information_only_check(out)
        # the escalation line says "talk to a lawyer" — that's a referral, not advice
        assert not hits, f"advice phrases in {lang}: {hits}"


def test_never_claims_lawyer():
    out = _render("are you a lawyer?")
    assert "not a lawyer" in out.lower() or DISCLAIMER in out


# ── grounded citations ─────────────────────────────────────────────────

def test_citations_present():
    ans = answer_legal_question("landlord locked me out")
    assert len(ans.citations) >= 1
    assert any("tenancy" in c.lower() for c in ans.citations)


def test_corpus_search_works():
    hits = corpus_search("notice to quit", limit=2)
    assert len(hits) >= 1
    assert "title" in hits[0]


def test_corpus_search_never_raises():
    assert corpus_search("") == corpus_search("")


# ── language handling ──────────────────────────────────────────────────

def test_language_aliases():
    assert answer_legal_question("locked out", "pidgin").language == "pcm"
    assert answer_legal_question("locked out", "yoruba").language == "yo"
    assert answer_legal_question("locked out", "hausa").language == "ha"
    assert answer_legal_question("locked out", "igbo").language == "ig"


def test_unknown_language_defaults_english():
    assert answer_legal_question("locked out", "klingon").language == "en"


def test_empty_question():
    out = _render("")
    assert DISCLAIMER in out


# ── billing (#68) ──────────────────────────────────────────────────────

def test_bill_answer_service_rate():
    db = tempfile.mktemp(suffix=".db")
    cost = bill_answer("+2348012345678", db_path=db)
    assert cost == 1400  # ₦14 service message


def test_bill_answer_never_raises():
    # a directory as db_path makes sqlite connect fail → billed 0, no raise
    assert bill_answer("", db_path=tempfile.mkdtemp()) == 0


def test_bill_answer_bad_category_raises():
    with pytest.raises(ValueError):
        bill_answer("+2348012345678", category="bogus",
                    db_path=tempfile.mktemp(suffix=".db"))


# ── chat control ───────────────────────────────────────────────────────

def test_control_help():
    assert "/legal" in control_legal("help")


def test_control_empty():
    assert "/legal" in control_legal("")


def test_control_question():
    out = control_legal("my landlord locked me out")
    assert "cannot lock you out" in out
    assert DISCLAIMER in out


def test_control_language_prefix():
    out = control_legal("pcm my landlord don lock me out")
    assert "no get right" in out


def test_control_never_raises():
    assert control_legal(None)  # type: ignore[arg-type]
    assert control_legal("x" * 10000)


def test_control_too_short():
    out = control_legal("hi")
    assert "/legal" in out
