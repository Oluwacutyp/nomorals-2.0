"""Phase 7: Research/OSINT god-tier upgrade tests."""

import sys
sys.path.insert(0, '.')

from nomorals.research.pipeline import _reasoning_pass, _MAX_REFINEMENTS
from nomorals.research.grounded import extract_claims, Source
from nomorals.search.osint import (
    OSINT_SPECS, PhoneIntelAdapter, GitHubReconAdapter,
    WaybackAdapter, GravatarAdapter, LeakCheckAdapter,
    _looks_like_phone, _confidence,
)


def test_reasoning_pass_done():
    """Returns [] when LLM says DONE."""
    findings = [type('F', (), {'title': 'T', 'snippet': 'S'})()]
    result = _reasoning_pass("q", findings, llm_fn=lambda p: "DONE")
    assert result == [], f"expected [], got {result}"
    print("✓ reasoning_pass DONE")


def test_reasoning_pass_followups():
    """Returns follow-up queries when LLM provides them."""
    findings = [type('F', (), {'title': 'T', 'snippet': 'S'})()]
    result = _reasoning_pass("q", findings,
                             llm_fn=lambda p: "What is X?\nHow does Y work?")
    assert len(result) == 2, f"expected 2, got {result}"
    print("✓ reasoning_pass follow-ups")


def test_reasoning_pass_no_llm():
    """Returns [] without LLM (graceful degradation)."""
    assert _reasoning_pass("q", [], llm_fn=None) == []
    print("✓ reasoning_pass no-LLM")


def test_max_refinements():
    assert _MAX_REFINEMENTS == 3
    print("✓ max refinements = 3")


def test_claim_extraction():
    sources = [Source(doc_id='1', title='Python Guide',
                      snippet='Python is a programming language created by Guido van Rossum in 1991')]
    claims = extract_claims(
        'Python is a programming language created by Guido van Rossum. '
        'It was released in 1991.', sources)
    assert len(claims) >= 1
    verified = [c for c in claims if c.verified]
    assert len(verified) >= 1, "expected at least one verified claim"
    print(f"✓ claim extraction: {len(claims)} claims, {len(verified)} verified")


def test_claim_confidence():
    s1 = Source(doc_id='1', title='A', snippet='Python is great for data science work')
    s2 = Source(doc_id='2', title='B', snippet='Python excels at data science tasks')
    claims = extract_claims('Python is great for data science work.', [s1, s2])
    assert claims[0].confidence == "high", f"got {claims[0].confidence}"
    print("✓ claim confidence high with 2 sources")


def test_osint_specs_count():
    assert len(OSINT_SPECS) == 9, f"expected 9, got {len(OSINT_SPECS)}"
    names = [s[0] for s in OSINT_SPECS]
    for expected in ["osint_phone", "osint_github", "osint_wayback",
                     "osint_gravatar", "osint_leakcheck"]:
        assert expected in names, f"missing {expected}"
    print("✓ 9 OSINT adapters registered")


def test_phone_detection():
    assert _looks_like_phone("+1234567890") == "+1234567890"
    assert _looks_like_phone("not a phone") is None
    print("✓ phone detection")


def test_confidence_ratings():
    assert _confidence(3) == "high"
    assert _confidence(1, True) == "medium"
    assert _confidence(1) == "low"
    assert _confidence(0) == "low"
    print("✓ confidence ratings")


def test_adapter_instantiation():
    for cls in [PhoneIntelAdapter, GitHubReconAdapter, WaybackAdapter,
                GravatarAdapter, LeakCheckAdapter]:
        adapter = cls()
        assert adapter.name.startswith("osint_")
        assert adapter.description
        print(f"✓ {cls.__name__}")


if __name__ == "__main__":
    test_reasoning_pass_done()
    test_reasoning_pass_followups()
    test_reasoning_pass_no_llm()
    test_max_refinements()
    test_claim_extraction()
    test_claim_confidence()
    test_osint_specs_count()
    test_phone_detection()
    test_confidence_ratings()
    test_adapter_instantiation()
    print("\nAll Phase 7 tests passed!")
