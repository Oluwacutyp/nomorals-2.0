"""Tests for build-map #97 — AEO/GEO visibility tracking. All offline."""

import os
import tempfile

from nomorals.marketing.aeo import (
    AEOTracker,
    AEO_WEEKLY_ACTION,
    ENGINE_NAMES,
    Mention,
    VisibilityReport,
    cross_check,
    default_prompts,
    ensure_weekly,
    estimate_cost,
    find_mentions,
    parse_citations,
    control_aeo,
)


def _tmpdb():
    return tempfile.mktemp(suffix=".db")


# ── parsing ────────────────────────────────────────────────────────────────

def test_parse_citations_urls_and_domains():
    text = ("Check https://example.com/pricing for details, or visit "
            "example.org. Also [our docs](https://docs.example.com/x).")
    cites = parse_citations(text)
    assert any("example.com/pricing" in c for c in cites)
    assert any("example.org" == c for c in cites)
    assert any("docs.example.com/x" in c for c in cites)


def test_parse_citations_never_raises_on_garbage():
    assert parse_citations(None) == []
    assert parse_citations("") == []
    assert isinstance(parse_citations(object()), list)


def test_find_mentions_sentence_level():
    text = ("Acme is a great widget maker. Nobody knows about other brands. "
            "I love Acme products, see https://acme.com.")
    ms = find_mentions(text, "Acme", "chatgpt", "What is Acme?")
    assert len(ms) == 2
    assert all(m.engine == "chatgpt" for m in ms)
    assert ms[0].brand == "Acme"
    # second mention carries the citation
    assert any("acme.com" in c for m in ms for c in m.citations)


def test_find_mentions_word_boundary():
    text = "AcmeCorp is unrelated. Acme is great."
    ms = find_mentions(text, "Acme", "claude", "p")
    assert len(ms) == 1
    assert "Acme is great" in ms[0].context


def test_find_mentions_empty_brand():
    assert find_mentions("Acme is great.", "", "chatgpt", "p") == []


# ── cross-check ────────────────────────────────────────────────────────────

def test_cross_check_yes_confirms():
    m = Mention(engine="chatgpt", prompt="p", brand="Acme",
                context="Acme is a great widget maker.")
    ok = cross_check(m, "claude", lambda e, p: "YES\nQuote: 'Acme is a great widget maker.'")
    assert ok is True
    assert m.confirmed is True
    assert m.confirming_engine == "claude"


def test_cross_check_no_rejects():
    m = Mention(engine="chatgpt", prompt="p", brand="Acme",
                context="Acme is a great widget maker.")
    ok = cross_check(m, "claude", lambda e, p: "NO\nNo such brand named.")
    assert ok is False
    assert m.confirmed is False


def test_cross_check_ambiguous_counts_as_unconfirmed():
    m = Mention(engine="chatgpt", prompt="p", brand="Acme", context="Acme is great.")
    ok = cross_check(m, "claude", lambda e, p: "MAYBE, hard to say")
    assert ok is False
    assert m.confirmed is False


def test_cross_check_never_raises():
    m = Mention(engine="chatgpt", prompt="p", brand="Acme", context="x")
    assert cross_check(m, "claude", None) is False
    assert cross_check(m, "claude", lambda e, p: 1 / 0) is False
    assert cross_check(None, "claude", lambda e, p: "YES") is False


# ── tracking ───────────────────────────────────────────────────────────────

def _mock_caller_factory(responses, verify="YES"):
    def caller(engine, prompt):
        if prompt.startswith("Answer with exactly YES or NO"):
            return verify
        return responses.get((engine, prompt), "")
    return caller


def test_track_visibility_share_of_answer():
    responses = {
        ("chatgpt", "What is Acme?"): "Acme is a great widget maker, widely praised.",
        ("claude", "What is Acme?"): "I don't know about that company specifically.",
    }
    t = AEOTracker(db_path=_tmpdb())
    rep = t.track_visibility(
        "Acme", ["What is Acme?"], ["chatgpt", "claude"],
        engine_caller=_mock_caller_factory(responses))
    assert rep.total_pairs == 2
    assert len(rep.mentions) == 1
    # the one mention confirmed by the other engine
    assert len(rep.confirmed) == 1
    assert rep.share == 0.5
    assert rep.gaps == []


def test_track_visibility_filters_hallucinated_mentions():
    responses = {
        ("chatgpt", "What is Acme?"): "Acme is a great widget maker.",
        ("claude", "What is Acme?"): "Acme is a great widget maker.",
    }
    t = AEOTracker(db_path=_tmpdb())
    rep = t.track_visibility(
        "Acme", ["What is Acme?"], ["chatgpt", "claude"],
        engine_caller=_mock_caller_factory(responses, verify="NO"))
    # mentions found but none confirmed → share 0, prompt is a gap
    assert len(rep.mentions) == 2
    assert len(rep.confirmed) == 0
    assert rep.share == 0.0
    assert rep.gaps == ["What is Acme?"]


def test_track_visibility_unavailable_engines_honest():
    t = AEOTracker(db_path=_tmpdb())
    rep = t.track_visibility("Acme", ["What is Acme?"], ["chatgpt", "claude"],
                             engine_caller=lambda e, p: "")
    assert rep.total_pairs == 2
    assert rep.confirmed == []
    assert rep.share == 0.0
    assert all(not r.ok for r in rep.responses)


def test_track_visibility_never_raises():
    t = AEOTracker(db_path=_tmpdb())
    rep = t.track_visibility("", None, None, engine_caller=lambda e, p: 1 / 0)
    assert isinstance(rep, VisibilityReport)
    rep2 = t.track_visibility("Acme", ["p"], ["bogus-engine"],
                              engine_caller=lambda e, p: "x")
    assert isinstance(rep2, VisibilityReport)


def test_share_property():
    rep = VisibilityReport(report_id="r", brand="B", prompts=["a", "b"],
                           engines=["e1", "e2"])
    rep.mentions = [Mention(engine="e1", prompt="a", confirmed=True)]
    assert rep.total_pairs == 4
    assert rep.share == 0.25


def test_report_format_contains_share_and_context():
    rep = VisibilityReport(report_id="r", brand="Acme", prompts=["What is Acme?"],
                           engines=["chatgpt", "claude"])
    rep.mentions = [Mention(engine="chatgpt", prompt="What is Acme?",
                            brand="Acme", context="Acme is a great widget maker.",
                            confirmed=True, confirming_engine="claude")]
    rep.gaps = []
    out = rep.format()
    assert "Acme" in out
    assert "50%" in out
    assert "1/2" in out
    assert "Acme is a great widget maker" in out


def test_default_prompts():
    ps = default_prompts("Acme")
    assert len(ps) >= 4
    assert all("Acme" in p for p in ps)


def test_estimate_cost_range():
    lo, hi = estimate_cost(5, 4)
    assert 0.15 <= lo <= 0.5
    assert 0.5 <= hi <= 1.5
    assert lo <= hi
    assert estimate_cost(0, 0) == (0.0, 0.0)


# ── history & briefs ───────────────────────────────────────────────────────

def test_history_and_trend():
    t = AEOTracker(db_path=_tmpdb())
    responses = {("chatgpt", "What is Acme?"): "Acme is great."}
    t.track_visibility("Acme", ["What is Acme?"], ["chatgpt"],
                       engine_caller=_mock_caller_factory(responses))
    t.track_visibility("Acme", ["What is Acme?"], ["chatgpt"],
                       engine_caller=_mock_caller_factory(responses, verify="NO"))
    hist = t.history("Acme")
    assert len(hist) == 2
    assert hist[0]["brand"] == "Acme"
    assert hist[0]["share"] == 0.0   # latest: unconfirmed
    assert hist[1]["share"] == 1.0   # earlier: confirmed


def test_latest_rebuilds_gaps():
    t = AEOTracker(db_path=_tmpdb())
    responses = {("chatgpt", "What is Acme?"): "Nothing about them."}
    t.track_visibility("Acme", ["What is Acme?"], ["chatgpt"],
                       engine_caller=_mock_caller_factory(responses))
    latest = t.latest("Acme")
    assert latest is not None
    assert latest.brand == "Acme"
    assert latest.gaps == ["What is Acme?"]


def test_content_briefs_from_gaps():
    t = AEOTracker(db_path=_tmpdb())
    rep = VisibilityReport(report_id="r", brand="Acme",
                           prompts=["What is Acme?"], engines=["chatgpt"],
                           gaps=["What is Acme?", "Acme pricing?"])
    briefs = t.content_briefs(rep)
    assert len(briefs) == 2
    assert all("title" in b and "angle" in b and "gap_prompt" in b for b in briefs)
    assert "Acme" in briefs[0]["why"]


def test_content_briefs_empty_when_no_gaps():
    t = AEOTracker(db_path=_tmpdb())
    rep = VisibilityReport(report_id="r", brand="Acme", prompts=["p"], engines=["e"])
    assert t.content_briefs(rep) == []


# ── chat ───────────────────────────────────────────────────────────────────

def test_control_aeo_usage():
    out = control_aeo("")
    assert "usage" in out.lower()
    assert "/aeo track" in out


def test_control_aeo_report_none_yet():
    # point at a fresh DB via monkeypatched tracker
    import nomorals.marketing.aeo as aeo_mod
    orig = aeo_mod._get_tracker
    aeo_mod._get_tracker = lambda: AEOTracker(db_path=_tmpdb())
    try:
        out = control_aeo("report")
        assert "no AEO report yet" in out
    finally:
        aeo_mod._get_tracker = orig


def test_control_aeo_full_flow():
    import nomorals.marketing.aeo as aeo_mod
    orig_tracker = aeo_mod._get_tracker
    orig_caller = aeo_mod._default_engine_caller
    db = _tmpdb()
    responses = {("chatgpt", "What is Acme?"): "Acme is a great widget maker."}

    def fake_caller(engine, prompt):
        if prompt.startswith("Answer with exactly YES or NO"):
            return "YES"
        return responses.get((engine, prompt), "")

    aeo_mod._get_tracker = lambda: AEOTracker(db_path=db)
    aeo_mod._default_engine_caller = fake_caller
    try:
        out = control_aeo("track Acme; What is Acme?")
        assert "share of answer" in out
        assert "Acme" in out
        rep_out = control_aeo("report Acme")
        assert "share of answer" in rep_out
        br_out = control_aeo("briefs Acme")
        # one prompt confirmed → no gaps → congrats message
        assert "no content gaps" in br_out
    finally:
        aeo_mod._get_tracker = orig_tracker
        aeo_mod._default_engine_caller = orig_caller


def test_control_aeo_briefs_with_gap():
    import nomorals.marketing.aeo as aeo_mod
    orig_tracker = aeo_mod._get_tracker
    orig_caller = aeo_mod._default_engine_caller
    db = _tmpdb()
    aeo_mod._get_tracker = lambda: AEOTracker(db_path=db)
    aeo_mod._default_engine_caller = lambda e, p: "I know nothing."
    try:
        control_aeo("track Acme")
        out = control_aeo("briefs Acme")
        assert "content briefs" in out.lower()
    finally:
        aeo_mod._get_tracker = orig_tracker
        aeo_mod._default_engine_caller = orig_caller


def test_control_aeo_never_raises():
    assert isinstance(control_aeo(None), str)
    assert isinstance(control_aeo("track"), str)
    assert isinstance(control_aeo("report \x00"), str)


def test_share_bounded_when_multiple_mentions_per_pair():
    # regression: share must never exceed 100% even when one response
    # contains several mentions of the brand.
    responses = {
        ("chatgpt", "What is Acme?"): "Acme is great. Acme is the best. Acme wins.",
    }
    t = AEOTracker(db_path=_tmpdb())
    rep = t.track_visibility(
        "Acme", ["What is Acme?"], ["chatgpt"],
        engine_caller=_mock_caller_factory(responses))
    assert len(rep.mentions) == 3
    assert rep.share == 1.0
    assert len(rep.confirmed_pairs) == 1


def test_ensure_weekly_registers_and_is_idempotent():
    import asyncio
    from types import SimpleNamespace

    calls = []

    class FakeSched:
        def list_jobs(self):
            return []

        async def schedule_cron(self, **kw):
            calls.append(kw)
            return True

    s = FakeSched()
    assert ensure_weekly(s) is True
    assert calls[0]["cron_expr"] == "0 9 * * MON"
    assert calls[0]["action"] == AEO_WEEKLY_ACTION

    s2 = FakeSched()
    s2.list_jobs = lambda: [SimpleNamespace(action=AEO_WEEKLY_ACTION)]
    assert ensure_weekly(s2) is True
    assert len(calls) == 1  # no duplicate registration


def test_ensure_weekly_never_raises():
    assert ensure_weekly(None) is False
    assert ensure_weekly(object()) is False
