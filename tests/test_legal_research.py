"""Build-map #78 — RAG + confidence + citations anti-hallucination stack.

All offline. The corpus is the seed under nomorals/legal/corpus/.
"""

import os

import pytest

from nomorals.legal.contracts import DISCLAIMER, information_only_check
from nomorals.legal.research import (
    CONFIDENTIALITY,
    LegalResearch,
    _calibrate,
    control_research,
    format_research,
    research_citations,
)
from nomorals.legal.aid import answer_legal_question


@pytest.fixture()
def researcher():
    r = LegalResearch()
    yield r
    r.close()


# ── grounded answers ─────────────────────────────────────────────────

def test_grounded_answer(researcher):
    result = researcher.research("my landlord locked me out of my flat")
    assert result.answered is True
    assert result.confidence > 0
    assert result.citations
    # Every claim in the text comes from corpus snippets — citations are
    # traceable to document + section.
    labels = [c.label() for c in result.citations]
    assert any("Lagos Tenancy Law" in label for label in labels)
    assert any(" — " in label for label in labels)


def test_confidence_in_range(researcher):
    for q in (
        "landlord locked me out",
        "minimum notice before termination",
        "defective goods refund FCCPC",
        "arrested by police",
    ):
        r = researcher.research(q)
        assert 0.0 <= r.confidence <= 1.0, q


def test_calibrate_monotone():
    # More evidence → higher confidence. Pure function, easy to check.
    assert _calibrate(10.0, 3) > _calibrate(1.5, 1)
    assert _calibrate(5.0, 3) > _calibrate(5.0, 1)
    assert _calibrate(5.0, 3) > _calibrate(0.5, 3)
    assert 0.0 <= _calibrate(0.0, 0) <= 1.0


def test_citation_traceability(researcher):
    """Snippets must appear verbatim in the corpus files — no fabrication."""
    from nomorals.legal import research as research_mod
    cdir = os.path.join(os.path.dirname(research_mod.__file__), "corpus")
    result = researcher.research("minimum notice before termination")
    assert result.answered
    for ev in result.evidence:
        path = os.path.join(cdir, ev.doc_id.split("#")[0].replace("legal-", "") + ".md")
        assert os.path.isfile(path), path
        with open(path, encoding="utf-8") as fh:
            corpus_text = fh.read().replace("\n", " ")
        # Section heading must exist verbatim in the file…
        assert ev.section in corpus_text, ev.section
        # …and a contiguous 20-char chunk of the snippet must too (snippets
        # may straddle the title/section boundary, so check chunks).
        words = ev.snippet.replace("…", " ").split()
        found = any(
            " ".join(words[i:i + 5]) in corpus_text
            for i in range(max(1, len(words) - 4))
        )
        assert found, ev.snippet[:60]


def test_snippet_comes_from_corpus_section(researcher):
    result = researcher.research("rent increase notice")
    assert result.answered
    for c in result.citations:
        assert c.title  # document
        # section is non-empty for all seeded sections
        assert c.section


# ── honest unknowns ──────────────────────────────────────────────────

def test_unknown_query_not_answered(researcher):
    result = researcher.research("what is the capital gains tax rate on crypto trading")
    assert result.answered is False
    assert result.confidence == 0.0
    assert result.citations == []
    assert "won't guess" in result.text or "don't know" in result.text.lower()


def test_gibberish_not_answered(researcher):
    result = researcher.research("zxqv wobble quark 9xyz")
    assert result.answered is False
    assert result.confidence == 0.0


def test_never_raises(researcher):
    for q in ("", "   ", None, "a" * 4000):
        r = researcher.research(q)  # noqa: SLF001 — None handled gracefully
        assert r.confidence == 0.0 or isinstance(r.confidence, float)


# ── confidentiality ──────────────────────────────────────────────────

def test_confidentiality_constant():
    assert CONFIDENTIALITY
    assert "training" in CONFIDENTIALITY.lower()


def test_no_files_written(researcher):
    """Research writes nothing to disk — queries can't leak into files."""
    from nomorals.legal import research as research_mod
    cdir = os.path.join(os.path.dirname(research_mod.__file__), "corpus")
    before = set(os.listdir(cdir))
    researcher.research("landlord locked me out of my flat query-xyzzy-123")
    assert set(os.listdir(cdir)) == before


def test_no_network_imports():
    from nomorals.legal import research as research_mod
    src = open(research_mod.__file__, encoding="utf-8").read()
    for bad in ("import requests", "import urllib", "import httpx",
                "socket.", "urlopen"):
        assert bad not in src, f"network code found: {bad}"


# ── shared stack: #78 serves #77 ─────────────────────────────────────

def test_aid_integration_uses_research_citations(researcher):
    ans = answer_legal_question("my landlord locked me out", "en",
                                researcher=researcher)
    # Research citations ("Title — Section") are merged up front.
    assert any(" — " in c for c in ans.citations)
    # Scenario core content is intact (backward compatible).
    assert "lock" in ans.text.lower()


def test_aid_without_researcher_unchanged(researcher):
    # Same answer as before the wiring — default None keeps behavior.
    ans = answer_legal_question("my landlord locked me out", "en")
    assert ans.citations
    assert "lock" in ans.text.lower()


def test_aid_never_raises_with_broken_researcher():
    class Broken:
        def research(self, q):
            raise RuntimeError("boom")

    ans = answer_legal_question("my landlord locked me out", "en",
                                researcher=Broken())
    assert "lock" in ans.text.lower()  # falls back cleanly


# ── information-only enforcement ─────────────────────────────────────

def test_rendered_text_is_information_only(researcher):
    for q in ("landlord locked me out", "minimum notice before termination",
              "defective goods"):
        r = researcher.research(q)
        assert information_only_check(r.text) == [], q


def test_disclaimer_on_render(researcher):
    r = researcher.research("landlord locked me out")
    assert DISCLAIMER in r.render()
    r2 = researcher.research("zxqv wobble quark 9xyz")
    assert DISCLAIMER in r2.render()


# ── chat control ─────────────────────────────────────────────────────

def test_control_research_answer():
    out = control_research("can my landlord increase the rent without notice?")
    assert "Confidence" in out
    assert DISCLAIMER in out


def test_control_research_unknown():
    out = control_research("capital gains tax on crypto trading rates")
    assert "won't guess" in out or "couldn't find" in out
    assert DISCLAIMER in out


def test_control_research_usage():
    out = control_research("")
    assert "/research" in out


def test_control_research_never_raises():
    out = control_research(None)
    assert isinstance(out, str) and out


def test_format_research_never_raises():
    from nomorals.legal.research import ResearchResult
    assert format_research(ResearchResult(query="x", answered=False, text=""))


def test_research_citations_dedupe():
    from nomorals.legal.research import Citation, ResearchResult
    r = ResearchResult(
        query="x", answered=True, text="t",
        citations=[Citation("a", "Doc", "Sec", "s"),
                   Citation("b", "Doc", "Sec", "s")])
    assert research_citations(r) == ["Doc — Sec"]
