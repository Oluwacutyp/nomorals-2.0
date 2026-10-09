"""Tests for the autonomous research + wisdom organs.

Covers: organ event bus, persistent watches, knowledge gaps, learned
source quality, the research tick (network mocked), wisdom digest /
cross-link / queue (corpus faked), and spine-tool registration.
"""

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _fake_db():
    """Minimal in-memory stand-in for nomorals.storage.db.Database."""
    import sqlite3

    class FakeDB:
        def __init__(self):
            self._c = sqlite3.connect(":memory:")
            self._c.row_factory = sqlite3.Row

        def execute(self, sql, params=()):
            cur = self._c.execute(sql, params)
            self._c.commit()
            return cur

        def executescript(self, script):
            self._c.executescript(script)
            self._c.commit()

        def query(self, sql, params=()):
            return [dict(r) for r in self._c.execute(sql, params).fetchall()]

        def query_one(self, sql, params=()):
            row = self._c.execute(sql, params).fetchone()
            return dict(row) if row is not None else None

    return FakeDB()


def _ctx(db):
    ctx = types.SimpleNamespace(db=db, tools=None, memory=None,
                                gateway=None, router=None)
    return ctx


# ── event bus ─────────────────────────────────────────────────────────

def test_bus_emit_drain_consumes():
    from nomorals import organs
    db = _fake_db()
    organs.emit(db, "research", "wisdom", "finding.esoteric",
                {"title": "t"})
    organs.emit(db, "research", "wisdom", "finding.esoteric",
                {"title": "t2"})
    got = organs.drain(db, "wisdom")
    assert len(got) == 2
    assert got[0]["payload"]["title"] == "t"
    assert got[0]["src"] == "research"
    # Second drain: nothing left (consumed, not deleted).
    assert organs.drain(db, "wisdom") == []
    assert organs.pending_count(db, "wisdom") == 0


def test_bus_kind_filter():
    from nomorals import organs
    db = _fake_db()
    organs.emit(db, "a", "b", "k1", {})
    organs.emit(db, "a", "b", "k2", {})
    got = organs.drain(db, "b", kinds=["k1"])
    assert [e["kind"] for e in got] == ["k1"]
    assert organs.pending_count(db, "b") == 1


# ── research organ ────────────────────────────────────────────────────

def test_watch_persists_and_lists():
    from nomorals.research import autonomy
    db = _fake_db()
    organ = autonomy.ResearchOrgan(_ctx(db))
    wid = organ.add_watch("AI gigs", ["outlier hiring"], 12.0,
                          ["money"], created_by="test")
    watches = organ.list_watches()
    assert len(watches) == 1
    assert watches[0]["id"] == wid
    assert watches[0]["queries"] == ["outlier hiring"]
    assert watches[0]["cadence_hours"] == 12.0
    # Re-adding the same watch is idempotent, not a duplicate.
    wid2 = organ.add_watch("AI gigs", ["outlier hiring"], 12.0)
    assert wid2 == wid
    assert len(organ.list_watches()) == 1
    assert organ.remove_watch(wid) is True
    assert organ.list_watches()[0]["enabled"] is False


def test_gap_dedupe():
    from nomorals.research import autonomy
    db = _fake_db()
    organ = autonomy.ResearchOrgan(_ctx(db))
    g1 = organ.note_gap("what is qlora?", "training chat")
    g2 = organ.note_gap("what is qlora?", "training chat")
    assert g1 == g2  # no duplicate open gap
    assert len(organ.open_gaps()) == 1
    g3 = organ.note_gap("what is dpo?")
    assert g3 != g1
    assert len(organ.open_gaps()) == 2


def test_source_quality_learned():
    from nomorals.research import autonomy
    db = _fake_db()
    organ = autonomy.ResearchOrgan(_ctx(db))
    for _ in range(4):
        organ.record_source_quality("https://spam.example/x", False)
    organ.record_source_quality("https://good.example/y", True)
    organ.record_source_quality("https://good.example/z", True)
    assert organ.domain_score("spam.example") == 0.0
    assert organ.domain_score("good.example") == 1.0
    assert organ.domain_score("never.seen") is None
    weak = organ.weak_sources(min_runs=3)
    assert [w["domain"] for w in weak] == ["spam.example"]


def test_tick_runs_due_watch_without_network(monkeypatch):
    from nomorals.research import autonomy
    from nomorals.research import pipeline as pl
    db = _fake_db()
    organ = autonomy.ResearchOrgan(_ctx(db))
    organ.add_watch("t", ["q"], cadence_hours=0.001)  # due immediately

    finding = pl.ResearchFinding(job_id="w_x", title="Apocrypha text found",
                                 url="https://good.example/t",
                                 snippet="gnostic gospel fragment")
    monkeypatch.setattr(pl, "run_job", lambda job, rctx, **kw: [finding])

    captured = {}

    def fake_assess(f, rctx):
        return pl.Assessment(worth=True, score=0.9, reasons=["test"])
    monkeypatch.setattr(pl, "assess_worth", fake_assess)

    def fake_deliver(f, assessment, rctx):
        captured["delivered"] = True
        return ["chat:1"]
    monkeypatch.setattr(pl, "deliver", fake_deliver)

    report = organ.tick()
    assert report.watches_run == 1
    assert report.findings == 1
    assert report.delivered == 1
    assert captured.get("delivered") is True
    # Source quality was learned from the assessment.
    assert organ.domain_score("good.example") == 1.0
    # Esoteric finding forwarded to wisdom as an event.
    from nomorals import organs
    assert organs.pending_count(db, "wisdom") == 1
    # Second tick: not due yet, nothing re-runs.
    report2 = organ.tick()
    assert report2.watches_run == 0


def test_tick_resolves_gap(monkeypatch):
    from nomorals.research import autonomy
    from nomorals.research import pipeline as pl
    db = _fake_db()
    organ = autonomy.ResearchOrgan(_ctx(db))
    organ.note_gap("what is qlora?")

    class FakeDeep:
        synthesis = "QLoRA is quantized LoRA."
        findings = []

    monkeypatch.setattr(pl, "research_deep",
                        lambda q, rctx, llm_fn=None, **kw: FakeDeep())
    monkeypatch.setattr(pl, "_router_llm_fn", lambda router: None)
    report = organ.tick()
    assert report.gaps_resolved == 1
    assert organ.open_gaps() == []
    from nomorals import organs
    events = organs.drain(db, "brain")
    assert any(e["kind"] == "gap.answered" for e in events)


# ── wisdom organ ─────────────────────────────────────────────────────

class _FakePassage:
    def __init__(self, work, section, snippet, url=""):
        self.work = work
        self.section = section
        self.snippet = snippet
        self.url = url


class _FakeAnswer:
    def __init__(self, passages):
        self.passages = passages
        self.synthesis = ""


class _FakeCorpus:
    def __init__(self, entries, passages):
        self._manifest = entries
        self._passages = passages

    def ask(self, query, top=5, mode="keyword"):
        return _FakeAnswer(self._passages[:top])


class _FakeKeeper:
    def __init__(self, entries, passages):
        self.corpus = _FakeCorpus(entries, passages)

    def ask(self, query, top=5, mode="keyword"):
        return self.corpus.ask(query, top=top, mode=mode)

    def status(self):
        return {"corpus": {"ingested": 1}}


def _entry(slug, title, tradition, ingested=True):
    e = types.SimpleNamespace(
        slug=slug, title=title, tradition=tradition,
        canon_status="canon", source_url="http://x", license="pd",
        translator="", sha256="abc", ingested_at=1.0 if ingested else 0.0,
        notes="")
    return e


def _wisdom_organ(db, entries, passages):
    from nomorals.wisdom import autonomy as wa
    organ = wa.WisdomOrgan.__new__(wa.WisdomOrgan)
    organ.context = _ctx(db)
    organ.db = db
    wa.ensure_schema(db)
    keeper = _FakeKeeper(entries, passages)
    organ._keeper = lambda: keeper  # noqa: SLF001 - test seam
    return organ


def test_wisdom_digest_extractive():
    from nomorals.wisdom import autonomy as wa
    db = _fake_db()
    passages = [
        _FakePassage("[bible-kjv] King James Bible", "Gen 1",
                     "In the beginning God created the heaven and the earth. "
                     "And the earth was without form, and void."),
        _FakePassage("[bible-kjv] King James Bible", "John 1",
                     "In the beginning was the Word, and the Word was with "
                     "God, and the Word was God."),
    ]
    entries = {"bible-kjv": _entry("bible-kjv", "King James Bible",
                                   "christianity")}
    organ = _wisdom_organ(db, entries, passages)
    digest = organ.digest_work("bible-kjv", top_passages=2)
    assert digest["slug"] == "bible-kjv"
    assert len(digest["passages"]) == 2
    assert digest["key_terms"]  # salient terms extracted
    # Persisted and retrievable.
    assert organ.get_digest("bible-kjv")["title"] == "King James Bible"


def test_wisdom_cross_link():
    from nomorals.wisdom import autonomy as wa
    db = _fake_db()
    db2 = db
    organ = _wisdom_organ(db2, {}, [])
    # Two digests in different traditions sharing salient terms.
    for slug, tradition, terms in (
        ("work-a", "christianity",
         ["beginning", "word", "light", "spirit", "creation", "heaven"]),
        ("work-b", "sufism",
         ["beginning", "word", "light", "spirit", "creation", "divine"]),
    ):
        db.execute(
            "INSERT INTO wisdom_digests"
            " (slug, title, tradition, key_terms, passages, digested_at)"
            " VALUES (?, ?, ?, ?, '[]', 1)",
            (slug, slug, tradition, json.dumps(terms)))
    made = organ._cross_link(deadline=9999999999.0)  # noqa: SLF001
    assert made == 1
    links = organ.links_for("work-a")
    assert len(links) == 1
    assert links[0]["work"] == "work-b"
    assert "light" in links[0]["shared_terms"]
    # Same-tradition pairs are NOT linked.
    db.execute(
        "INSERT INTO wisdom_digests"
        " (slug, title, tradition, key_terms, passages, digested_at)"
        " VALUES ('work-c', 'c', 'christianity', ?, '[]', 1)",
        (json.dumps(["beginning", "word", "light", "spirit", "creation"]),))
    made2 = organ._cross_link(deadline=9999999999.0)  # noqa: SLF001
    # work-b (sufism) ↔ work-c (christianity) is a valid new cross-tradition
    # link; work-a ↔ work-c share a tradition and stay unlinked.
    assert made2 == 2
    assert organ.links_for("work-c") != []
    assert all(l["work"] != "work-a" for l in organ.links_for("work-c"))


def test_wisdom_queue_ingest_dedupe():
    from nomorals.wisdom import autonomy as wa
    db = _fake_db()
    organ = _wisdom_organ(db, {}, [])
    q1 = organ.queue_ingest("http://x/y", "Some Text", "sufism")
    q2 = organ.queue_ingest("http://x/y", "Some Text", "sufism")
    assert q1 == q2


def test_wisdom_drain_event_queues_ingest():
    from nomorals.wisdom import autonomy as wa
    from nomorals import organs
    db = _fake_db()
    organ = _wisdom_organ(db, {}, [])
    organs.emit(db, "research", "wisdom", "finding.esoteric",
                {"title": "Gospel of Thomas", "url": "http://x/thomas"})
    n = organ._drain_events()  # noqa: SLF001
    assert n == 1
    row = db.query_one("SELECT url FROM wisdom_ingest_queue")
    assert row["url"] == "http://x/thomas"


# ── spine tools ───────────────────────────────────────────────────────

def test_spine_tools_register():
    from nomorals.tools import registry as reg_mod
    from nomorals.tools import research as research_tools
    from nomorals.tools import wisdom as wisdom_tools
    reg = reg_mod.ToolRegistry(context=_ctx(_fake_db()))
    research_tools.register(reg)
    wisdom_tools.register(reg)
    names = set(reg._tools.keys())
    for expected in ("research_schedule", "research_jobs", "research_run_now",
                     "research_remove_watch", "note_knowledge_gap",
                     "research_weak_sources", "research_organ_tick",
                     "wisdom_ingest", "wisdom_digest", "wisdom_synthesize",
                     "wisdom_brief", "wisdom_organ_tick",
                     "wisdom_ask", "wisdom_status"):
        assert expected in names, f"missing spine tool {expected}"


def test_research_schedule_tool_end_to_end():
    from nomorals.tools import registry as reg_mod
    from nomorals.tools import research as research_tools
    db = _fake_db()
    ctx = _ctx(db)
    reg = reg_mod.ToolRegistry(context=ctx)
    research_tools.register(reg)
    out = reg._tools["research_schedule"].fn(
        topic="AI gigs", queries=["outlier"], cadence_hours=24)
    assert out["ok"] is True
    jobs = reg._tools["research_jobs"].fn()
    assert len(jobs["watches"]) == 1
    gap = reg._tools["note_knowledge_gap"].fn(question="what is qlora?")
    assert gap["ok"] is True
