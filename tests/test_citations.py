"""Build-map #20 — claim-level evidence + citation infrastructure.

All offline: no LLM, no network.
"""

from __future__ import annotations

import hashlib

import pytest

from nomorals.research.citations import (
    CitationManager,
    Claim,
    extract_claims,
    sentence_supported,
    verify_claims,
)


# ── CitationManager: register / dedupe / hash ─────────────────────────────

def _mgr():
    return CitationManager(clock=lambda: 1728393600.0)  # 2024-10-08 UTC


def test_register_returns_sequential_ids_and_dedupes():
    m = _mgr()
    s1 = m.register_source("https://a.example/x", "A", "alpha text")
    s2 = m.register_source("https://b.example/y", "B", "beta text")
    assert (s1, s2) == ("S1", "S2")
    # same URL twice → same id (dedupe, first wins)
    assert m.register_source("https://a.example/x", "A-changed", "other") == "S1"
    assert m.get("S1").text == "alpha text"


def test_register_stores_hash_and_timestamp():
    m = _mgr()
    m.register_source("https://a.example/x", "A", "alpha text")
    rec = m.get("S1")
    assert rec.sha256 == hashlib.sha256(b"alpha text").hexdigest()
    assert rec.accessed_ts == 1728393600.0


def test_cite_exact_quote_returns_marker():
    m = _mgr()
    m.register_source("https://a.example/x", "A",
                      "Lagos has 16 million people.")
    assert m.cite("S1", "Lagos has 16 million people.") == "[S1]"
    # case-insensitive match is fine
    assert m.cite("S1", "lagos HAS 16 million people.") == "[S1]"


def test_cite_invented_quote_raises():
    m = _mgr()
    m.register_source("https://a.example/x", "A", "Lagos has 16 million people.")
    with pytest.raises(ValueError):
        m.cite("S1", "Lagos has 99 million people.")


def test_cite_unknown_source_raises():
    m = _mgr()
    with pytest.raises(ValueError):
        m.cite("S9", "anything")


# ── resolve: deterministic remap + Works Cited ────────────────────────────

def test_resolve_remaps_in_first_appearance_order():
    m = _mgr()
    m.register_source("https://a.example/x", "Alpha", "aaa")
    m.register_source("https://b.example/y", "Beta", "bbb")
    text, works = m.resolve("First [S2] then [S1] then again [S2].")
    assert text == "First [1] then [2] then again [1]."
    assert works[0].startswith("[1] Beta — https://b.example/y")
    assert works[1].startswith("[2] Alpha — https://a.example/x")


def test_resolve_drops_unknown_markers():
    m = _mgr()
    m.register_source("https://a.example/x", "Alpha", "aaa")
    text, works = m.resolve("Known [S1] and unknown [S9] here.")
    assert text == "Known [1] and unknown  here."
    assert len(works) == 1


def test_resolve_works_cited_format():
    m = _mgr()
    m.register_source("https://a.example/x", "Alpha", "aaa")
    _, works = m.resolve("See [S1].")
    assert works == [
        "[1] Alpha — https://a.example/x "
        "(accessed 2024-10-08, sha256:" +
        hashlib.sha256(b"aaa").hexdigest()[:12] + "…)"
    ]


def test_audit_trail_carries_hashes_timestamps_quotes():
    m = _mgr()
    m.register_source("https://a.example/x", "Alpha", "alpha text here")
    m.register_source("https://b.example/y", "Beta", "beta text here")
    m.cite("S1", "alpha text here")
    trail = m.audit_trail()
    by_id = {t["source_id"]: t for t in trail}
    assert set(by_id) == {"S1", "S2"}
    s1 = by_id["S1"]
    assert s1["sha256"] == hashlib.sha256(b"alpha text here").hexdigest()
    assert s1["accessed_ts"] == 1728393600.0
    assert s1["quotes"] == ["alpha text here"]
    assert by_id["S2"]["quotes"] == []  # registered, never cited


# ── extract_claims / verify_claims ────────────────────────────────────────

SOURCE = (
    "Lagos has an estimated population of 16 million people as of 2024. "
    "It is the largest city in Nigeria. "
    "What is the capital of Nigeria? "
    "The city keeps growing every single year without pause."
)


def test_extract_claims_finds_factual_sentences_with_quotes():
    claims = extract_claims(SOURCE, source_id="S1")
    texts = [c.text for c in claims]
    assert any("16 million" in t for t in texts)
    assert any("largest city" in t for t in texts)
    # questions are not claims
    assert not any("capital" in t for t in texts)
    # every claim paired with its exact source sentence
    for c in claims:
        assert c.quote in SOURCE
        assert c.source_id == "S1"
        assert c.confidence in ("high", "medium", "low")


def test_extract_claims_confidence_signal_strength():
    claims = extract_claims(SOURCE, source_id="S1")
    pop = next(c for c in claims if "16 million" in c.text)
    assert pop.confidence == "high"  # number + named entity + predication
    vague = extract_claims("Things are generally quite nice nowadays.",
                           source_id="S1")
    assert vague and vague[0].confidence == "low"


def test_extract_claims_rerank_hook_optional():
    claims = extract_claims(SOURCE, source_id="S1",
                            rerank_fn=lambda cs: list(reversed(cs)))
    assert claims[0].text != extract_claims(SOURCE, source_id="S1")[0].text
    # a failing reranker falls back to the baseline, never raises
    claims2 = extract_claims(SOURCE, source_id="S1",
                             rerank_fn=lambda cs: 1 / 0)
    assert claims2


def test_verify_claims_drops_unsupported():
    good = Claim(text="Lagos has 16 million people.",
                 quote="Lagos has 16 million people.",
                 confidence="high", source_id="S1")
    bad = Claim(text="Lagos has 99 million people.",
                quote="Lagos has 99 million people.",
                confidence="high", source_id="S1")
    out = verify_claims([good, bad], "Lagos has 16 million people.")
    assert [c.text for c in out] == ["Lagos has 16 million people."]
    assert out[0].verified is True


def test_verify_claims_never_raises():
    assert verify_claims(None, "text") == []
    weird = Claim(text="x", quote=None, confidence="low", source_id="S1")
    assert verify_claims([weird], "text") == []


# ── sentence_supported ────────────────────────────────────────────────────

def test_sentence_supported_true():
    assert sentence_supported(
        "Lagos has an estimated population of 16 million people as of 2024.",
        "Lagos has an estimated population of 16 million people as of 2024. "
        "Extra context here.") is True


def test_sentence_supported_rejects_invented_number():
    assert sentence_supported(
        "Lagos covers 99999 square kilometers of land.",
        "Lagos has an estimated population of 16 million people.") is False


def test_sentence_supported_rejects_unrelated():
    assert sentence_supported(
        "The moon is made of green cheese.",
        "Lagos has an estimated population of 16 million people.") is False


# ── pipeline integration: synthesize strips unverified sentences ──────────

def _finding(title, url, snippet, detail=""):
    from nomorals.research.pipeline import ResearchFinding
    return ResearchFinding(job_id="j", title=title, url=url,
                           snippet=snippet, detail=detail)


def test_synthesize_strips_sentence_with_unverified_citation():
    from nomorals.research.pipeline import synthesize
    findings = [_finding(
        "Lagos Population Study", "https://example.com/lagos",
        "Lagos has an estimated population of 16 million people as of 2024.")]
    llm = lambda prompt: (  # noqa: E731
        "Lagos has an estimated population of 16 million people as of 2024 [S1]. "
        "Lagos covers 99999 square kilometers of land [S1].")
    out = synthesize("How big is Lagos?", findings, llm_fn=llm)
    assert "16 million" in out
    assert "99999" not in out  # smuggled figure stripped
    assert "[1]" in out  # surviving citation remapped


def test_synthesize_keeps_fully_supported_answer():
    from nomorals.research.pipeline import synthesize
    findings = [_finding(
        "Lagos Population Study", "https://example.com/lagos",
        "Lagos has an estimated population of 16 million people as of 2024.")]
    llm = lambda prompt: (  # noqa: E731
        "Lagos has an estimated population of 16 million people as of 2024 [S1].")
    out = synthesize("How big is Lagos?", findings, llm_fn=llm)
    assert "16 million" in out and "[1]" in out


def test_synthesize_still_strips_invented_labels():
    from nomorals.research.pipeline import synthesize
    findings = [_finding("Real", "https://example.com/r", "Real facts here.")]
    llm = lambda prompt: "Stuff happened [S7]."  # noqa: E731
    out = synthesize("What?", findings, llm_fn=llm)
    assert "[S7]" not in out


# ── DeepReport carries the audit trail ────────────────────────────────────

def test_research_deep_report_carries_citations():
    from types import SimpleNamespace

    from nomorals.core.result import Ok
    from nomorals.research.pipeline import (
        ResearchContext, research_deep,
    )
    from nomorals.storage.db import Database

    class _Reg:
        def call(self, name, **kw):
            if name == "web_search":
                return Ok({"results": [{
                    "title": "Lagos Population Study",
                    "url": "https://example.com/lagos",
                    "snippet": ("Lagos has an estimated population of "
                                "16 million people as of 2024."),
                }]})
            if name == "web_fetch":
                return Ok({"url": kw.get("url"), "text": "fetched"})
            raise AssertionError(name)

    def llm(prompt):
        if "scoping a research question" in prompt:
            return "CLEAR"
        if "Break this research question" in prompt:
            return "lagos population 2024"
        assert "FINDINGS" in prompt
        return ("Lagos has an estimated population of 16 million people "
                "as of 2024 [S1].")

    rctx = ResearchContext(db=Database(":memory:"), registry=_Reg())
    report = research_deep("How big is Lagos?", rctx, llm_fn=llm)
    assert "16 million" in report.synthesis
    assert report.citations, "DeepReport must carry the evidence trail"
    c0 = report.citations[0]
    assert c0["url"] == "https://example.com/lagos"
    assert len(c0["sha256"]) == 64
    assert c0["accessed_ts"] > 0
