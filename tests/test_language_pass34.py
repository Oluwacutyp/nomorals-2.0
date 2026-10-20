"""Pass #34 — language + translation surfaces.

Defects pinned here (all reproduced on pristine HEAD):

1. Unicode normalization: language/dialect detection, voice-money
   parsing, and pronunciation assessment matched precomposed (NFC)
   bytes only. The same text in decomposed (NFD) form — which STT
   engines and some keyboards emit — was misclassified (Ekiti read as
   Standard Yoruba, Yoruba read as English), a transfer command
   degraded to "unknown", and a perfectly pronounced sentence scored
   20% with the student told to work on words they said correctly.

2. tonal G2P: word-final nasal vowels (àdán, rùn, yẹn…) never
   nasalized — the "next char" probe is "" at word end and
   ``"" in "aeiou"`` is True, so the branch read it as vowel-followed.

3. /ekiti chat surface never wired the runtime brain into the tutor:
   converse/debate always returned the canned scaffolding replies,
   byte-identical for every input, in production.
"""
import unicodedata

import pytest

from nomorals.learn.dialect import (
    EkitiTutor,
    detect_dialect_detail,
    suggest_ekiti_fix,
)
from nomorals.media.tonal_g2p import detect_languages, g2p
from nomorals.voice.money import detect_language, parse_voice_money


def nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


# ── 1. detection must not depend on unicode encoding form ──────────────────

def test_detect_language_nfd_ekiti():
    text = "Mi fẹ́ rán owó sí Mama"
    assert detect_language(text)[0] == "ekiti"
    assert detect_language(nfd(text))[0] == "ekiti"


def test_detect_language_nfd_standard_yoruba_not_english():
    text = "Mo ná ẹgbẹ̀rún márùn-ún fún data"
    assert detect_language(text)[0] == "yoruba"
    lang, conf = detect_language(nfd(text))
    assert lang == "yoruba"
    assert conf > 0.0


def test_parse_voice_money_nfd_transfer_survives():
    text = "Ránṣẹ́ 5k sí Mama"
    nfc_intent = parse_voice_money(text)
    assert nfc_intent.kind == "transfer"
    nfd_intent = parse_voice_money(nfd(text))
    assert nfd_intent.kind == "transfer"
    assert nfd_intent.amount_kobo == 500000
    assert nfd_intent.recipient == "Mama"


def test_parse_voice_money_nfd_ekiti_expense():
    text = "Mi ná 5k fún data"
    intent = parse_voice_money(nfd(text))
    assert intent.kind == "log_expense"
    assert intent.language == "ekiti"
    assert intent.amount_kobo == 500000


def test_detect_dialect_detail_nfd_ekiti():
    d = detect_dialect_detail(nfd("mi lọ sí ọjà"))
    assert d["label"] == "ekiti"
    assert d["ekiti_score"] > 0


def test_suggest_ekiti_fix_nfd():
    fix = suggest_ekiti_fix(nfd("mo lọ sí ibẹ̀"))
    assert fix is not None
    assert "mi lọ" in fix


def test_assess_pronunciation_encoding_invariant():
    tutor = EkitiTutor()
    expected = "Mi fẹ́ lọ sí ilé"
    # Heard in NFD (decomposed) — same sounds, different bytes.
    report = tutor.assess_pronunciation(nfd(expected), expected)
    assert report.accuracy == 1.0
    # Expected stored in NFD, heard in NFC.
    report2 = tutor.assess_pronunciation(expected, nfd(expected))
    assert report2.accuracy == 1.0


def test_detect_languages_nfd_keeps_diacritic_evidence():
    # No marker-lexicon words here: the ONLY Yoruba evidence is the
    # dot-below diacritics + tone marks.
    line = "Ọ̀nà yìí gùn gan"
    nfc_langs = detect_languages(line)[0]["languages"]
    nfd_langs = detect_languages(nfd(line))[0]["languages"]
    assert nfc_langs and nfc_langs[0]["language"] == "yo"
    assert nfd_langs and nfd_langs[0]["language"] == "yo"
    assert nfd_langs[0]["score"] >= 3.0


# ── 2. G2P word-final nasal vowels ──────────────────────────────────────────

def test_g2p_word_final_nasal_vowel():
    assert g2p("rùn") == ["r", "u\u0303"]
    assert g2p("àdán") == ["a", "d", "a\u0303"]
    assert g2p("yẹn") == ["y", "ẹ\u0303"]


def test_g2p_midword_nasal_and_plain_vowels_unchanged():
    # Mid-word nasal (vowel + n before a consonant) already worked.
    assert g2p("ránṣẹ́") == ["r", "a\u0303", "ṣ", "ẹ"]
    # No trailing n -> no nasalization invented.
    assert g2p("ilé") == ["i", "l", "e"]


# ── 3. /ekiti surface uses the brain when the runtime has one ──────────────

def test_ekiti_control_converse_uses_llm(monkeypatch):
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin

    calls = []

    def fake_llm(self, system, prompt):
        calls.append((system, prompt))
        return "LLM-EKITI-REPLY"

    monkeypatch.setattr(RuntimeMemoryMixin, "_ekiti_llm", fake_llm,
                        raising=False)
    m = RuntimeMemoryMixin()
    out = m._control_ekiti("converse mi lọ sí ọjà", chat_key="p34-a")
    assert "LLM-EKITI-REPLY" in out
    assert calls  # the tutor really called through


def test_ekiti_control_debate_uses_llm(monkeypatch):
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin

    def fake_llm(self, system, prompt):
        return "LLM-DEBATE-REPLY"

    monkeypatch.setattr(RuntimeMemoryMixin, "_ekiti_llm", fake_llm,
                        raising=False)
    m = RuntimeMemoryMixin()
    out = m._control_ekiti("debate education", chat_key="p34-b")
    assert "LLM-DEBATE-REPLY" in out


def test_ekiti_control_falls_back_when_llm_fails(monkeypatch):
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin

    def broken_llm(self, system, prompt):
        raise RuntimeError("brain down")

    monkeypatch.setattr(RuntimeMemoryMixin, "_ekiti_llm", broken_llm,
                        raising=False)
    m = RuntimeMemoryMixin()
    # No crash, and the documented scaffolding reply still answers.
    out = m._control_ekiti("converse mi lọ sí ọjà", chat_key="p34-c")
    assert out
    assert "LLM" not in out
