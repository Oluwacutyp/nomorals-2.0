"""Pass #129 pins: PoliteFetch failures stop vanishing.

PoliteFetch is the per-host politeness gate (backoff, rate budget,
robots.txt) behind a never-raises contract — and every one of its
fallbacks leaned toward ALLOWING the fetch: a crashed gate returned
"allowed", a crashed robots check returned "not disallowed", dropped
bookkeeping silently un-enforced the budget and the backoff. A fetch
that should have been gated looked identical to a fetch that was
checked and allowed. The 4 sites now record into the shared
source-registry ledger (reused from #121) while returning exactly
the same fallback values.
"""
import ast
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media.source_registry import PoliteFetch
from nomorals.media import source_registry as sr_mod
from nomorals.media.source_registry_failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _pf():
    # PoliteFetch only touches _last_hit/_backoff_until/_robots in the
    # converted methods; a dummy registry keeps DEFAULT_REGISTRY out.
    return PoliteFetch(registry=object())


def _desc(**over):
    d = {"domain": "example.com", "rate_limit_per_min": 10,
         "robots_respect": False}
    d.update(over)
    return SimpleNamespace(**d)


def _sites():
    return [f.site for f in recent_failures()]


# ── ledger basics ──────────────────────────────────────────────────────


def test_ledger_basics_and_format():
    assert recent_failures() == []
    assert format_failures() == "no recorded source registry failures"
    record_failure("PoliteFetch.allowed", ValueError("boom"))
    assert failure_count() == 1
    assert _sites() == ["PoliteFetch.allowed"]
    assert "PoliteFetch.allowed: ValueError: boom" in format_failures()


def test_ledger_bound_and_unprintable():
    class _BadStr(Exception):
        def __str__(self):
            raise RuntimeError("nope")

    for _ in range(250):
        record_failure("PoliteFetch.allowed", _BadStr())
    assert failure_count() == 250
    assert len(recent_failures(limit=1000)) == 200
    assert "<unprintable RuntimeError>" in format_failures()


def test_direct_import_surface_and_all_unchanged():
    import nomorals.media.source_registry_failures as led

    assert led.__all__ == ["Failure", "record_failure", "recent_failures",
                           "failure_count", "clear_failures", "format_failures"]
    # No widening: PoliteFetch was already exported; the pass adds nothing.
    assert sr_mod.__all__ == [
        "TrustTier", "Licence", "SourceDescriptor", "Provenance",
        "tag_chunk", "SourceRegistry", "LearnedRule", "DEFAULT_REGISTRY",
        "render_status_text", "EkitiLexicon", "EKITI_LEXICON",
        "PoliteFetch", "ContentAddressedCache", "as_data_block",
        "contains_instruction_attempt", "policy_allows_tool_call",
        "requires_owner_approval", "redact_identity",
        "BOOK_SOURCE_DOMAINS", "WEB_BACKEND_DOMAINS",
        "WISDOM_SOURCE_DOMAINS",
    ]


# ── converted site: allowed ────────────────────────────────────────────


def test_allowed_failure_records_and_still_returns_allowed():
    # The fail-open return is UNCHANGED — the record is what makes it
    # distinguishable from a genuine "checked and allowed".
    assert _pf().allowed(None) == (True, 0.0, "")
    assert _sites() == ["PoliteFetch.allowed"]


def test_allowed_benign_paths_record_nothing():
    pf = _pf()
    # Genuine allow: same return shape, no record.
    assert pf.allowed(_desc(), now=1000.0) == (True, 0.0, "")
    # Genuine backoff refusal: same gate, no record.
    pf._backoff_until["example.com"] = 1060.0
    ok, wait, reason = pf.allowed(_desc(), now=1000.0)
    assert ok is False and wait == 60.0 and reason == "backoff after 429/5xx"
    # Genuine rate refusal: no record either.
    pf2 = _pf()
    pf2._last_hit["example.com"] = 999.0
    ok, wait, reason = pf2.allowed(_desc(), now=1000.0)
    assert ok is False and reason == "rate budget"
    assert recent_failures() == []


# ── converted site: note_request ───────────────────────────────────────


def test_note_request_failure_records_and_does_not_raise():
    _pf().note_request(None)
    assert _sites() == ["PoliteFetch.note_request"]


def test_note_request_benign_path_sets_last_hit_without_record():
    pf = _pf()
    pf.note_request(_desc(), now=1234.0)
    assert pf._last_hit["example.com"] == 1234.0
    assert recent_failures() == []


# ── converted site: note_response ──────────────────────────────────────


def test_note_response_failure_records_and_does_not_raise():
    _pf().note_response(None, 429)
    assert _sites() == ["PoliteFetch.note_response"]


def test_note_response_benign_paths_update_backoff_without_record():
    pf = _pf()
    pf.note_response(_desc(), 429, now=1000.0)
    assert pf._backoff_until["example.com"] == 1060.0
    pf.note_response(_desc(), 200, now=1000.0)
    assert "example.com" not in pf._backoff_until
    assert recent_failures() == []


# ── converted site: _disallowed ────────────────────────────────────────


def test_disallowed_failure_records_and_returns_not_disallowed():
    pf = _pf()
    with mock.patch.object(PoliteFetch, "_fetch_robots",
                           side_effect=RuntimeError("robots down")):
        assert pf._disallowed("example.com", "https://example.com/x", 1000.0) is False
    assert _sites() == ["PoliteFetch._disallowed"]


def test_disallowed_benign_paths_record_nothing():
    pf = _pf()
    with mock.patch.object(PoliteFetch, "_fetch_robots",
                           return_value=["/private"]):
        assert pf._disallowed("example.com",
                              "https://example.com/private/x", 1000.0) is True
        # Cached on the second call: same verdict, still no record.
        assert pf._disallowed("example.com",
                              "https://example.com/open/x", 1000.0) is False
    assert recent_failures() == []


# ── honest neighbour: _fetch_robots ────────────────────────────────────


def test_fetch_robots_failure_is_logged_and_not_ledgered():
    # Documented fail-open ("[] on any failure (fail-open, logged)")
    # with the exception bound into the log line — honest, outside
    # the estate; the ledger stays empty.
    with mock.patch("nomorals.media.source_registry.urllib.request.urlopen",
                    side_effect=OSError("net down")):
        assert PoliteFetch._fetch_robots("example.com") == []
    assert recent_failures() == []


# ── scan pin: PoliteFetch remainder is empty, file remainder is 4 ──────
# (Pass #131 update: 16 -> 8 is a count snapshot a conversion is
# supposed to change, not an architectural invariant. The real
# invariant — no PoliteFetch site remains — is untouched, and the
# pin additionally asserts no converted safety-helper site remains.)


def test_scan_politefetch_remainder_is_empty_and_file_is_4():
    from nomorals.tools import error_scan as es

    path = sr_mod.__file__
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([path])
    finally:
        import importlib
        importlib.reload(es)
    tree = ast.parse(open(path).read())
    # Enclosing class + method per handler line.
    owner_of = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, ast.FunctionDef):
                    for t in ast.walk(m):
                        if isinstance(t, ast.Try):
                            for h in t.handlers:
                                owner_of[h.lineno] = (node.name, m.name)
    owners = [owner_of.get(f.line) for f in report.findings]
    # No PoliteFetch site remains: all 4 converted.
    assert not [o for o in owners if o and o[0] == "PoliteFetch"]
    # No converted safety-helper site remains either (Pass #130): the
    # module-level safety functions are not class methods, so they do
    # not appear in owner_of — assert via the count arithmetic instead.
    # File remainder: 26 (Pass #121 inventory, re-verified at #129 base)
    # minus exactly the 4 PoliteFetch conversions (#129) minus exactly
    # the 6 safety-helper conversions (#130) minus exactly the 8
    # tail-region conversions (#131).
    # #140: minus exactly the 4 debug-honest sites the scanner now
    # recognizes via exc_info=True.
    assert len(report.findings) == 4
