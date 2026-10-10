"""Sweep tests for the research module upgrade (2026-10-10).

Covers the mined-then-built changes:
- pipeline: _router_llm_fn crash fix, budget wall-clock deadline,
  learnings/refinement pass, working summary, synthesize_with_stats,
  finding styles, deliver_digest, digest-mode execute_job, domain rerank,
  depth param on research_deep
- citations: find_conflicts, corroboration_map
- grounded: relevance gate, confidence badge, hierarchical chunks,
  entity-mismatch flagging, _complete crash fix
- autonomy: gap priority, fuzzy dedup, attempts cap, gap_stats
- grounded_store: log_qa / export_thread / session_stats
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nomorals.research import pipeline as P
from nomorals.research.autonomy import ResearchOrgan
from nomorals.research.citations import (
    Conflict,
    corroboration_map,
    find_conflicts,
    sentence_supported,
)
from nomorals.research.grounded import (
    GroundedAnswer,
    GroundedSession,
    Source,
    _answer_confidence,
    _source_relevant,
)
from nomorals.research.grounded_store import GroundedSessionStore
from nomorals.storage.db import Database


# ── fakes ────────────────────────────────────────────────────────────────

class Ok:
    def __init__(self, value: Any):
        self.ok = True
        self.value = value
        self.error = ""


class Fail:
    ok = False
    value = None
    error = "boom"


class FakeRegistry:
    """Registry fake: search_results / fetch_text drive web_search/web_fetch."""

    def __init__(self, search_results: list[dict] | None = None,
                 fetch_text: str = ""):
        self.search_results = search_results or []
        self.fetch_text = fetch_text
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, actor: str = "system", **kwargs: Any):
        self.calls.append((name, kwargs))
        if name == "web_search":
            return Ok({"results": self.search_results})
        if name == "web_fetch":
            return Ok({"text": self.fetch_text})
        return Fail()


class FakeGateway:
    def __init__(self):
        self.owner_chats = ["telegram:123"]
        self.sent: list[tuple[str, str, str]] = []

    def send(self, platform: str, chat_key: str, text: str):
        self.sent.append((platform, chat_key, text))
        return Ok({"id": "m1"})


def make_rctx(db=None, registry=None, gateway=None, **kw) -> P.ResearchContext:
    base = dict(
        db=db or Database(":memory:"),
        registry=registry or FakeRegistry(),
        memory=None,
        gateway=gateway or FakeGateway(),
        goal_keywords={"money": ["outlier", "gig", "hiring", "$"]},
        worth_threshold=0.65,
        daily_delivery_cap=10,
    )
    base.update(kw)
    return P.ResearchContext(**base)


def worthy_finding(**kw) -> P.ResearchFinding:
    snippet = ("Outlier AI training gig hiring remote Nigerians paying $ per "
               "hour freelance new release apply today deadline soon")
    return P.ResearchFinding(
        job_id="j1",
        title=kw.get("title", "Outlier opens new AI training roles for Nigeria"),
        url=kw.get("url", "https://example.com/outlier-nigeria"),
        snippet=snippet,
        detail=kw.get("detail", ""),
    )


# ── pipeline: _router_llm_fn crash fix ─────────────────────────────────

def test_router_llm_fn_uses_router_not_self():
    """Regression: _router_llm_fn used brain_for(self.context) in a module
    function → NameError. It must call router.complete(prompt)."""
    class Resp:
        text = "hello"
    calls = {}

    class Router:
        def complete(self, prompt):
            calls["prompt"] = prompt
            return Resp()

    fn = P._router_llm_fn(Router())
    assert fn("ping?") == "hello"
    assert calls["prompt"] == "ping?"
    assert fn.last_response is not None
    assert fn.last_response.text == "hello"
    assert P._router_llm_fn(None) is None


# ── pipeline: budget wall-clock deadline ────────────────────────────────

def test_budget_deadline():
    b = P.ResearchBudget(10.0, deadline_s=3600)
    assert b.time_exceeded() is False
    assert b.time_left_s is not None and b.time_left_s > 0
    b2 = P.ResearchBudget(10.0, deadline_s=0)
    assert b2.time_exceeded() is True
    assert b2.exhausted is True
    b3 = P.ResearchBudget(10.0)
    assert b3.time_exceeded() is False
    assert b3.time_left_s is None


# ── pipeline: refinement pass (learnings) ────────────────────────────────

def test_refinement_pass_offline():
    f = worthy_finding()
    r = P._refinement_pass("q", [f], llm_fn=None)
    assert isinstance(r, P.Refinement)
    assert r.done is True
    assert r.follow_ups == []
    assert r.learnings and f.title in r.learnings[0]


def test_refinement_pass_llm_sections():
    def llm(prompt: str) -> str:
        return ("LEARNINGS:\n- Outlier pays $30/hr [S1]\n- Roles are remote [S1]\n"
                "FOLLOW-UPS:\n- Outlier Nigeria payout methods\n- NONE extra")
    r = P._refinement_pass("q", [worthy_finding()], llm_fn=llm)
    assert len(r.learnings) == 2
    assert r.follow_ups == ["Outlier Nigeria payout methods"]
    assert r.done is False


def test_refinement_pass_llm_done():
    def llm(prompt: str) -> str:
        return "LEARNINGS:\n- Enough [S1]\nFOLLOW-UPS:\nNONE"
    r = P._refinement_pass("q", [worthy_finding()], llm_fn=llm)
    assert r.done is True and r.follow_ups == []


def test_working_summary_template_path():
    s = P._update_working_summary("", ["alpha [S1]", "beta [S2]"], llm_fn=None)
    assert "alpha" in s and "beta" in s
    big = "x" * 5000
    s2 = P._update_working_summary(big, ["tail learning"], llm_fn=None)
    assert len(s2) <= 4000 and "tail learning" in s2


# ── pipeline: synthesize_with_stats ─────────────────────────────────────

def test_synthesize_with_stats_llm():
    f = worthy_finding(detail="Outlier pays $30 per hour for AI training gigs.")
    def llm(prompt: str) -> str:
        return "Outlier pays $30 per hour for AI training gigs. [S1]"
    text, stats = P.synthesize_with_stats("What does Outlier pay?", [f], llm)
    assert "[1]" in text
    assert stats["sentences_checked"] >= 1
    assert stats["sentences_stripped"] == 0
    assert stats["sources_cited"] == 1
    # back-compat wrapper still returns plain str
    assert isinstance(P.synthesize("q", [f], llm), str)


def test_synthesize_with_stats_strips_unverifiable():
    f = worthy_finding(detail="Outlier pays $30 per hour.")
    def llm(prompt: str) -> str:
        return ("Outlier pays $30 per hour. [S1] "
                "Outlier also pays $999 per hour. [S1]")
    text, stats = P.synthesize_with_stats("q", [f], llm)
    assert "$999" not in text
    assert stats["sentences_stripped"] >= 1


def test_verify_cited_sentences_backcompat():
    f = worthy_finding(detail="Outlier pays $30 per hour.")
    out = P._verify_cited_sentences("Outlier pays $30 per hour. [S1]", [f])
    assert isinstance(out, str)
    text, checked, stripped = P._verify_cited_sentences_stats(
        "Outlier pays $30 per hour. [S1]", [f])
    assert (checked, stripped) == (1, 0)
    assert text == out


# ── pipeline: finding styles + digest ───────────────────────────────────

def test_format_finding_styles():
    f = worthy_finding()
    a = P.Assessment(True, 0.85, ["relevant to money", "fresh", "score 0.85"])
    classic = P.format_finding(f, a)
    card = P.format_finding(f, a, style="card")
    brief = P.format_finding(f, a, style="brief")
    verbose = P.format_finding(f, a, style="verbose")
    assert classic.startswith("🔍 Outlier")
    assert "▰" in card and card != classic
    assert len(brief) < len(card)
    assert "Signals:" in verbose
    with pytest.raises(ValueError):
        P.format_finding(f, a, style="nope")


def test_deliver_digest_batches_one_message():
    db = Database(":memory:")
    P.ensure_schema(db)
    gw = FakeGateway()
    rctx = make_rctx(db=db, gateway=gw)
    pairs = [(worthy_finding(url=f"https://example.com/{i}"),
              P.Assessment(True, 0.8, ["relevant to money", "score 0.80"]))
             for i in range(3)]
    sent = P.deliver_digest(pairs, rctx, title="🔍 test digest")
    assert sent == ["telegram:123"]
    assert len(gw.sent) == 1  # one message, not three
    assert "test digest" in gw.sent[0][2]
    n = db.scalar("SELECT COUNT(*) FROM research_deliveries")
    assert n == 3  # each finding still recorded (never re-sent)


def test_deliver_digest_rejects_unworthy_and_cap():
    db = Database(":memory:")
    P.ensure_schema(db)
    rctx = make_rctx(db=db, daily_delivery_cap=1)
    good = (worthy_finding(), P.Assessment(True, 0.8, ["ok"]))
    with pytest.raises(ValueError):
        P.deliver_digest(
            [(worthy_finding(), P.Assessment(False, 0.1, ["nope"]))], rctx)
    with pytest.raises(RuntimeError):
        P.deliver_digest([good, good], rctx)  # 2 findings > cap 1


def test_execute_job_digest_mode():
    db = Database(":memory:")
    P.ensure_schema(db)
    results = [
        {"title": f"Outlier AI training gig {i} Nigeria",
         "url": f"https://example.com/gig{i}",
         "snippet": ("Outlier AI training gig hiring remote Nigerians paying "
                     "$ per hour freelance new release apply today deadline")}
        for i in range(2)
    ]
    gw = FakeGateway()
    rctx = make_rctx(db=db, registry=FakeRegistry(search_results=results),
                     gateway=gw)
    job = P.ResearchJob(id="j1", topic="gigs", queries=["q"], digest=True,
                        goals=["money"])
    report = P.execute_job(job, rctx)
    assert report.delivered == 2
    assert len(gw.sent) == 1  # digest: single message


def test_domain_rerank_demotes_weak_domains():
    db = Database(":memory:")
    db.execute("CREATE TABLE IF NOT EXISTS source_quality (domain TEXT PRIMARY KEY,"
               " runs INTEGER NOT NULL DEFAULT 0, worth_runs INTEGER NOT NULL DEFAULT 0,"
               " last_ts REAL NOT NULL DEFAULT 0)")
    db.execute("INSERT INTO source_quality VALUES ('noisy.example', 5, 0, 1)")
    good = worthy_finding(url="https://good.example/a")
    bad = worthy_finding(url="https://noisy.example/b")
    ranked = P._rerank_by_learned_quality([bad, good], db)
    assert ranked[0].url == good.url
    assert ranked[1].url == bad.url
    # unseen domains are never penalized
    assert P._learned_domain_score(db, "https://new.example/x") is None


def test_assess_worth_press_discount():
    db = Database(":memory:")
    P.ensure_schema(db)
    rctx = make_rctx(db=db)
    f = worthy_finding(
        title="Outlier press release: new AI training roles for Nigeria")
    f.snippet += " sponsored content"
    a = P.assess_worth(f, rctx)
    assert "placed/press content — independence discount" in a.reasons


# ── pipeline: research_deep depth + learnings + verification ────────────

def _deep_rctx() -> P.ResearchContext:
    db = Database(":memory:")
    P.ensure_schema(db)
    results = [
        {"title": "Outlier AI training gigs Nigeria 2026",
         "url": "https://example.com/a",
         "snippet": ("Outlier AI training gig hiring remote Nigerians paying "
                     "$ per hour freelance new release")},
        {"title": "Mindrift hiring AI trainers remote",
         "url": "https://other.example/b",
         "snippet": ("Mindrift hiring AI trainers remote freelance $ payout "
                     "new release")},
    ]
    return make_rctx(db=db,
                     registry=FakeRegistry(search_results=results,
                                           fetch_text="details here"))


def test_research_deep_depth1_offline():
    rctx = _deep_rctx()
    report = P.research_deep(
        "What are the best AI training gig platforms for Nigerians in 2026?",
        rctx, depth=1)
    assert report.needs_clarification is False
    assert report.depth == 1
    assert len(report.findings) == 2
    assert report.learnings  # offline learnings from titles
    assert report.running_summary
    assert report.verification["sources_cited"] == 2
    assert isinstance(report.conflicts, list)
    md = report.to_markdown()
    assert md.startswith("# Research:") and "## TL;DR" in md
    assert "## Evidence check" in md


def test_research_deep_clamps_depth():
    rctx = _deep_rctx()
    report = P.research_deep(
        "What are the best AI training gig platforms for Nigerians in 2026?",
        rctx, depth=99)
    assert report.depth == P._MAX_REFINEMENTS


def test_research_deep_time_budget():
    rctx = _deep_rctx()
    report = P.research_deep(
        "What are the best AI training gig platforms for Nigerians in 2026?",
        rctx, budget_usd=1.0, time_budget_s=0, depth=3)
    assert report.budget_exhausted is True


# ── citations: conflicts + corroboration ─────────────────────────────────

def _cf(title, url, snippet):
    return SimpleNamespace(title=title, url=url, snippet=snippet, detail="")


def test_find_conflicts():
    a = _cf("Outlier Nigeria payout", "https://a.example/x",
            "Outlier pays trainers $30 per hour for AI training gigs")
    b = _cf("Outlier Nigeria rates", "https://b.example/y",
            "Outlier pays trainers $45 per hour for AI training gigs")
    conflicts = find_conflicts([a, b])
    assert len(conflicts) == 1
    c = conflicts[0]
    assert isinstance(c, Conflict)
    assert "30" in c.value_a and "45" in c.value_b


def test_find_conflicts_same_domain_ignored():
    a = _cf("Outlier payout", "https://a.example/x",
            "Outlier pays trainers $30 per hour for AI training gigs")
    b = _cf("Outlier rates", "https://a.example/y",
            "Outlier pays trainers $45 per hour for AI training gigs")
    assert find_conflicts([a, b]) == []


def test_corroboration_map():
    a = _cf("Outlier Nigeria payout", "https://a.example/x",
            "Outlier pays trainers $30 per hour for AI training gigs")
    b = _cf("Outlier Nigeria rates", "https://b.example/y",
            "Outlier pays trainers $30 per hour for AI training gigs")
    c = _cf("Unrelated cooking", "https://c.example/z",
            "How to bake sourdough bread at home today")
    m = corroboration_map([a, b, c])
    assert m[0] == [1] and m[1] == [0]
    assert m[2] == []


# ── grounded: relevance gate, confidence, chunks, entity match ───────────

def test_source_relevant_gate():
    s = Source(doc_id="d", title="Banana guide",
               snippet="quantum bananas are yellow fruits")
    assert _source_relevant(s, "What are quantum bananas?") is True
    assert _source_relevant(s, "How do I repair a car engine?") is False


def test_ask_irrelevant_retrieval_falls_back_labeled(monkeypatch):
    sess = GroundedSession()
    sess.add_text("Quantum bananas are yellow fruits that grow on Mars.",
                  title="Bananas")
    bad = [Source(doc_id="d#c0", title="Bananas",
                  snippet="quantum bananas are yellow fruits")]
    monkeypatch.setattr(sess, "_retrieve", lambda q, k: bad)
    # Default: clearly-labeled ungrounded answer, never a grounded-looking
    # one over irrelevant context.
    ans = sess.ask("How do I repair a car engine transmission?",
                   llm_fn=lambda p: "general engine knowledge here.")
    assert ans.refused is False
    assert "Not in your documents" in ans.text
    # strict=True: hard refusal instead.
    ans2 = sess.ask("How do I repair a car engine transmission?",
                    llm_fn=lambda p: "never reached", strict=True)
    assert ans2.refused is True
    assert "couldn't find anything relevant" in ans2.text


def test_ask_answers_when_relevant():
    sess = GroundedSession()
    sess.add_text("Quantum bananas are yellow fruits that grow on Mars.",
                  title="Bananas")
    ans = sess.ask("What color are quantum bananas?",
                   llm_fn=lambda p: ("Quantum bananas are yellow fruits "
                                     "according to the guide. [S1]"))
    assert ans.refused is False
    assert ans.confidence == "high"
    rendered = ans.render()
    assert "confidence:" in rendered and "[1]" in rendered


def test_answer_confidence_levels():
    s = [Source(doc_id="d", title="t", snippet="s")]
    high = ("The first important claim is definitely true. [1] "
            "The second important claim is also true. [2]")
    assert _answer_confidence(high, s) == "high"
    assert _answer_confidence(
        "This long sentence has no citations at all in it.", s) == "low"


def test_section_chunks_carry_hierarchy():
    sess = GroundedSession()
    doc = SimpleNamespace(sections=[
        SimpleNamespace(heading="Intro", text="word " * 400),
        SimpleNamespace(heading="Details", text="thing " * 400),
    ])
    chunks = sess._section_chunks(doc, "MyDoc", "b")
    assert chunks
    assert all(c.startswith("Document: MyDoc\n") for c in chunks)
    assert any("Section: Intro" in c for c in chunks)
    assert any("Section: Details" in c for c in chunks)


def test_flag_entity_mismatches():
    s1 = Source(doc_id="a", title="Tesla revenue", snippet="Tesla made money")
    s2 = Source(doc_id="b", title="Apple earnings", snippet="Apple made money")
    sess = GroundedSession()
    sess._flag_entity_mismatches("What is Tesla revenue?", [s1, s2])
    assert s1.entity_match is True
    assert s2.entity_match is False


def test_complete_uses_context_not_self():
    """Regression: GroundedSession._complete used brain_for(self.context) —
    GroundedSession has no .context → AttributeError. Must use the passed
    context's router via the minimal complete(prompt) interface."""
    sess = GroundedSession()

    class Resp:
        text = "grounded!"

    class Router:
        def complete(self, prompt):
            assert "q?" in prompt
            return Resp()

    ctx = SimpleNamespace(router=Router())
    assert sess._complete("q?", None, ctx) == "grounded!"
    with pytest.raises(Exception):
        sess._complete("q?", None, None)
    with pytest.raises(Exception):
        sess._complete("q?", None, SimpleNamespace())


# ── autonomy: gaps ───────────────────────────────────────────────────────

def _organ():
    db = Database(":memory:")
    ctx = SimpleNamespace(db=db, tools=FakeRegistry(), memory=None,
                          gateway=None, router=None)
    return ResearchOrgan(ctx), db


def test_gap_priority_and_fuzzy_dedup():
    organ, _ = _organ()
    hi = organ._gap_priority(
        "What are the best AI training gigs hiring Nigerians this week?",
        "user wants paid work")
    lo = organ._gap_priority("stuff?", "")
    assert hi > lo
    gid = organ.note_gap("How do I find remote AI training jobs hiring now?")
    gid2 = organ.note_gap("How can I find remote AI training jobs that hire now?")
    assert gid == gid2  # fuzzy dedup
    gaps = organ.open_gaps()
    assert any(g["id"] == gid for g in gaps)
    assert all("priority" in g for g in gaps)


def test_gap_stats():
    organ, _ = _organ()
    organ.note_gap("What is the capital of France?")
    stats = organ.gap_stats()
    assert stats["open"] == 1
    assert stats["top_open"][0]["question"].startswith("What is the capital")


def test_weak_sources_report():
    organ, db = _organ()
    for _ in range(4):
        organ.record_source_quality("https://noisy.example/a", worth=False)
    organ.record_source_quality("https://good.example/a", worth=True)
    weak = organ.weak_sources()
    assert any(w["domain"] == "noisy.example" for w in weak)
    assert all(w["domain"] != "good.example" for w in weak)


# ── grounded_store: qa log + export + stats ─────────────────────────────

def test_store_qa_log_export_stats(tmp_path):
    store = GroundedSessionStore(tmp_path / "g")
    store.bind("chat:1")
    store.log_qa("chat:1", "What is this?", "It is a test. [1]")
    md = store.export_thread("chat:1")
    assert "What is this?" in md and "It is a test." in md
    stats = store.session_stats("chat:1")
    assert stats["active"] is True and stats["qa_logged"] == 1
    assert store.session_stats("chat:nope")["active"] is False
    assert store.export_thread("chat:nope") == ""
