"""Pass tail-3 pins: legal/research failures stay silent, ledger intact.

Target: nomorals/legal/research.py — 1 convert / 10 exclusions
(parent-adjudicated: site 323 converted — see its test).
Every site below maps a failure to a genuine degraded value (the
documented miss/unknown value of that function) and the stack proceeds
honestly — it never fabricates an answer from outside the corpus. None
of them records, because none of them hides a user-visible outcome.
Ledger REUSED: nomorals/legal/failures.py (same package — read-only
for pins; no record call, no import added to the target, which
contains no record_failure call at all).

Each test pins the degraded behavior AND the NOT-recording property
(ledger empty after the trigger). No bare ``except: pass`` anywhere
in this file — the scanner counts test files too; the tests below
need no except blocks at all (stubs raise, the code under test
catches).
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

from nomorals.legal import failures as lf
from nomorals.legal.contracts import DISCLAIMER
from nomorals.legal.research import (
    Citation,
    Evidence,
    LegalResearch,
    ResearchResult,
    expand_query,
    format_research,
    groundedness,
    verify_citations,
)


def _silent():
    assert lf.recent_failures() == []


class _BadStr(str):
    def strip(self, *a, **k):
        raise RuntimeError("strip boom")


def test_290_expand_query_failure_returns_original_query():
    bad = _BadStr("landlord sacked me")
    assert expand_query(bad) == [bad]
    _silent()


def test_323_verify_citations_failure_clears_unverified_evidence():
    # Tail-3 convert (parent-adjudicated): a failed verification must
    # NOT return the unfiltered evidence list — research() composes
    # claims from result.evidence under a verified-corpus header
    # without checking flags, so unfiltered evidence out is a
    # fabricated claim. Clearing both lists makes research() fall
    # through to _unknown, as when verification filters everything.
    res = ResearchResult(query="q", answered=True, text="t",
                         evidence=[Evidence(doc_id="d1", title="T",
                                            section="S", snippet=123,
                                            score=1.0)],
                         citations=[Citation(doc_id="d1", title="T",
                                             section="S", snippet="123")])
    out = verify_citations(res, {"d1": ("T", "S", "some body text")})
    assert out is res
    assert out.evidence == []
    assert out.citations == []
    _silent()


class _BoomVerified:
    @property
    def verified(self):
        raise RuntimeError("verified boom")


def test_334_groundedness_failure_is_zero():
    res = ResearchResult(query="q", answered=True, evidence=[_BoomVerified()])
    assert groundedness(res) == 0.0
    _silent()


def test_366_import_failure_gives_no_index_and_unknown_answer(tmp_path):
    with mock.patch.dict(sys.modules, {"nomorals.documents.index": None}):
        r = LegalResearch(corpus_dir=str(tmp_path))
    assert r._index is None
    assert r.research("landlord locked me out").answered is False
    _silent()


def test_396_index_build_failure_gives_no_index(tmp_path):
    with mock.patch("os.listdir", side_effect=OSError("listdir boom")):
        r = LegalResearch(corpus_dir=str(tmp_path))
    assert r._index is None
    _silent()


def test_400_one_bad_doc_skipped_rest_indexed(tmp_path):
    from nomorals.documents.index import DocumentIndex as RealIndex

    class FlakyIndex(RealIndex):
        def __init__(self):
            super().__init__()
            self._n = 0

        def add(self, doc):
            self._n += 1
            if self._n == 1:
                raise RuntimeError("add boom")
            return super().add(doc)

    for stem in ("a", "b"):
        (tmp_path / f"{stem}.md").write_text(
            f"# Doc {stem}\n\n## Section {stem}\n\nBody text {stem}.\n")
    with mock.patch("nomorals.documents.index.DocumentIndex", FlakyIndex):
        r = LegalResearch(corpus_dir=str(tmp_path))
    assert r._index is not None
    assert len(r._index) == 1
    assert r._sections and all(k.startswith("legal-b#")
                               for k in r._sections)
    r.close()
    _silent()


def test_407_failing_close_still_marks_closed():
    fake = SimpleNamespace(close=mock.Mock(side_effect=RuntimeError("close boom")))
    r = LegalResearch(index=fake)
    r.close()
    assert r._closed is True
    assert r._search("x") == []
    _silent()


def test_416_failing_search_is_empty():
    fake = SimpleNamespace(search=mock.Mock(side_effect=RuntimeError("search boom")))
    r = LegalResearch(index=fake)
    assert r._search("landlord") == []
    _silent()


class _BadDict(dict):
    def items(self):
        raise RuntimeError("items boom")


def test_527_describe_failure_is_disclaimer_only():
    r = LegalResearch(index=SimpleNamespace())
    r._sections = _BadDict({"legal-a#s": ("T", "H", "body")})
    assert r.describe_corpus() == DISCLAIMER
    _silent()


def test_546_related_failure_is_empty():
    r = LegalResearch(index=SimpleNamespace())
    r._sections = _BadDict({"legal-a#s": ("T", "H", "body")})
    assert r.related_sections(ResearchResult(query="q", answered=True)) == []
    _silent()


def test_557_format_failure_is_disclaimer():
    bad = SimpleNamespace(render=mock.Mock(side_effect=RuntimeError("render boom")))
    assert format_research(bad) == DISCLAIMER
    _silent()


def test_research_with_broken_index_answers_unknown_never_raises():
    fake = SimpleNamespace(search=mock.Mock(side_effect=RuntimeError("search boom")))
    r = LegalResearch(index=fake)
    out = r.research("my landlord locked me out of my flat")
    assert out.answered is False
    assert "won't guess" in out.text
    _silent()
