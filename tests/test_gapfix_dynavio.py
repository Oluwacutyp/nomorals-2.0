"""Dynavio REA (52-websites #51): requirements extraction, analysis,
graph representation — the published Mokammel/Coatanéa/Nenchev method,
Devon-native on the Document model.

The fixture spec below is built to the organ's spec rules: every
planted requirement carries a modal verb or an explicit requirement
ID, every planted smell fires exactly one lexicon rule, and the two
non-requirements carry neither. Recall/precision are measured against
that plant, per rung, and recorded honestly.
"""

from __future__ import annotations

import json
from typing import Any

from nomorals.documents.model import Document, Section, full_text
from nomorals.documents.requirements import (
    RequirementAnalysis,
    RequirementGraph,
    analyze_corpus,
    analyze_document,
    extract_requirements,
    format_requirements_reply,
    open_requirements_store,
    render_report_markdown,
    score_links,
    verify_spans,
    write_report,
)

# ── the fixture spec ─────────────────────────────────────────────────────
# 14 planted requirements (S1–S12, S15–S16) + 2 non-requirements.
FIXTURE_SPEC = """\
REQ-1: The system shall authenticate users with multi-factor authentication.
The system shall respond appropriately to invalid input.
The dashboard shall load quickly.
REQ-4: The service must encrypt all stored data at rest.
The system shall support reporting as required.
REQ-6: TBD.
The system shall encrypt data at rest and shall rotate encryption keys every 90 days and shall audit all access.
REQ-8: The API shall return search results within 200 milliseconds.
The system must never store passwords in plain text.
REQ-10: The system shall send a confirmation email after a user registers.
REQ-11: The system shall send a confirmation email after registration, as defined in REQ-10.
The system shall send confirmation emails to newly registered users.
This document describes the new onboarding flow.
The team met on Tuesday to discuss the launch.
REQ-15: The system shall retain all user data indefinitely.
REQ-16: The system shall delete all user data after 30 days.
"""

#: requirement ids the code rung must find, in document order.
PLANTED_IDS = ["REQ-1", "REQ-2", "REQ-3", "REQ-4", "REQ-5", "REQ-6",
               "REQ-7", "REQ-8", "REQ-9", "REQ-10", "REQ-11", "REQ-12",
               "REQ-15", "REQ-16"]

#: (requirement index in PLANTED_IDS order, flag kind, evidence word)
PLANTED_FLAGS = [
    (1, "ambiguous", "appropriately"),
    (2, "ambiguous", "quickly"),
    (2, "unverifiable", "quickly"),
    (4, "weak_verb", "support"),
    (4, "ambiguous", "as required"),
    (5, "incomplete", "TBD"),
    (6, "compound", "shall"),
]


def _fixture_doc() -> Document:
    return Document(
        id="doc-fixture",
        format="text",
        title="Onboarding spec (fixture)",
        sections=[Section(level=1, heading="Requirements",
                          text=FIXTURE_SPEC)],
    )


def _fake_llm_contradiction(prompt: str) -> str:
    """A deterministic brain stub: names the planted contradiction pair
    with span citations, and nothing else."""
    return json.dumps({
        "reclassify": [],
        "contradictions": [
            {"a": "REQ-15", "b": "REQ-16",
             "reason": "REQ-15 retains all user data indefinitely while "
                       "REQ-16 deletes all user data after 30 days — the "
                       "two cannot both hold.",
             "confidence": 0.9},
        ],
        "missed": [],
    })


def _fake_llm_bogus_span(prompt: str) -> str:
    """A brain stub whose 'missed' requirement quotes text that is NOT
    in the document — the organ must drop it, never invent the span."""
    return json.dumps({
        "reclassify": [],
        "contradictions": [],
        "missed": [{"text": "The system shall teleport users to Mars.",
                    "kind": "functional",
                    "reason": "invented by a careless brain"}],
    })


def _fake_llm_garbage(prompt: str) -> str:
    return "this is not json at all"


# ── extraction: recall + precision against the plant ─────────────────────

class TestExtraction:
    def test_code_rung_finds_all_planted_requirements(self):
        reqs = extract_requirements(_fixture_doc())
        found = [r.id for r in reqs]
        assert found == PLANTED_IDS, f"found {found}"

    def test_code_rung_ignores_prose(self):
        reqs = extract_requirements(_fixture_doc())
        texts = [r.text for r in reqs]
        assert not any("describes the new onboarding flow" in t
                       for t in texts)
        assert not any("met on Tuesday" in t for t in texts)

    def test_every_requirement_span_is_verbatim(self):
        doc = _fixture_doc()
        text = full_text(doc)
        for req in extract_requirements(doc):
            assert req.text == text[req.span.start:req.span.end], \
                f"span mismatch on {req.id}"
            assert req.span.text == req.text
        assert verify_spans(
            analyze_document(document=doc), doc) == []

    def test_explicit_ids_survive(self):
        reqs = extract_requirements(_fixture_doc())
        by_id = {r.id: r for r in reqs}
        assert by_id["REQ-1"].text.startswith("REQ-1:")
        assert by_id["REQ-6"].kind == "unknown"
        assert by_id["REQ-6"].text == "REQ-6: TBD."

    def test_kinds_follow_the_cue_rules(self):
        reqs = {r.id: r for r in extract_requirements(_fixture_doc())}
        assert reqs["REQ-1"].kind == "quality"   # authentication
        assert reqs["REQ-4"].kind == "quality"   # encrypt
        assert reqs["REQ-8"].kind == "quality"   # latency + number
        assert reqs["REQ-9"].kind == "constraint"  # never
        assert reqs["REQ-10"].kind == "functional"
        assert reqs["REQ-2"].kind == "functional"

    def test_empty_document_finds_nothing(self):
        doc = Document(id="doc-empty", format="text", title="empty",
                       sections=[Section(level=1, heading="Nothing",
                                         text="")])
        assert extract_requirements(doc) == []

    def test_text_intake_and_path_intake(self, tmp_path):
        analysis = analyze_document(text=FIXTURE_SPEC,
                                    doc_ref="pasted-fixture")
        assert analysis.ok and len(analysis.requirements) == 14
        spec_file = tmp_path / "spec.txt"
        spec_file.write_text(FIXTURE_SPEC, encoding="utf-8")
        analysis2 = analyze_document(path=str(spec_file))
        assert analysis2.ok and len(analysis2.requirements) == 14
        assert analysis2.doc_ref == str(spec_file)

    def test_missing_path_refuses_honestly(self):
        analysis = analyze_document(path="/no/such/spec.txt")
        assert not analysis.ok
        assert "no such" in (analysis.error or "").lower() \
            or "not found" in (analysis.error or "").lower()

    def test_no_intake_refuses_honestly(self):
        analysis = analyze_document()
        assert not analysis.ok
        assert analysis.error


# ── quality flags: precision against the plant ───────────────────────────

class TestQualityFlags:
    def test_planted_smells_flagged_with_evidence(self):
        reqs = extract_requirements(_fixture_doc())
        for index, kind, evidence in PLANTED_FLAGS:
            req = reqs[index]
            kinds = [(f.kind, f.evidence) for f in req.quality_flags]
            assert (kind, evidence) in kinds, \
                f"{req.id} missing flag {(kind, evidence)}; has {kinds}"

    def test_flag_precision_no_false_smells(self):
        reqs = extract_requirements(_fixture_doc())
        flagged = {(r.id, f.kind, f.evidence)
                   for r in reqs for f in r.quality_flags}
        planted = {(PLANTED_IDS[i], k, e) for i, k, e in PLANTED_FLAGS}
        assert flagged == planted, f"extra flags: {flagged - planted}"

    def test_clean_requirement_has_no_flags(self):
        reqs = {r.id: r for r in extract_requirements(_fixture_doc())}
        assert reqs["REQ-1"].quality_flags == []
        assert reqs["REQ-8"].quality_flags == []


# ── links: similarity, dependency, never contradiction from code ─────────

class TestLinks:
    def test_near_duplicates_link_similar(self):
        reqs = extract_requirements(_fixture_doc())
        score_links(reqs)
        by_id = {r.id: r for r in reqs}
        sims = [ln for ln in by_id["REQ-12"].links
                if ln.kind == "similar"]
        assert sims, "REQ-12 should link similar to REQ-10/REQ-11"
        assert all(ln.score >= 0.5 for ln in sims)
        assert all(ln.method in ("tfidf_cosine", "lsa_cosine")
                   for ln in sims)

    def test_unrelated_requirements_do_not_link(self):
        reqs = extract_requirements(_fixture_doc())
        score_links(reqs)
        by_id = {r.id: r for r in reqs}
        assert by_id["REQ-8"].links == [] or all(
            ln.to != "REQ-1" for ln in by_id["REQ-8"].links)

    def test_explicit_id_reference_links_depends(self):
        reqs = extract_requirements(_fixture_doc())
        score_links(reqs)
        by_id = {r.id: r for r in reqs}
        deps = [ln for ln in by_id["REQ-11"].links
                if ln.kind == "depends" and ln.to == "REQ-10"]
        assert len(deps) == 1
        assert deps[0].score == 0.95
        assert deps[0].method == "keyword"

    def test_code_rung_never_emits_contradicts(self):
        reqs = extract_requirements(_fixture_doc())
        score_links(reqs)
        for req in reqs:
            assert not any(ln.kind == "contradicts"
                           for ln in req.links), req.id

    def test_similarity_is_symmetric_and_bounded(self):
        reqs = extract_requirements(_fixture_doc())
        score_links(reqs)
        by_id = {r.id: r for r in reqs}
        for req in reqs:
            for ln in req.links:
                assert 0.0 <= ln.score <= 1.0
                if ln.kind == "similar":
                    back = [b for b in by_id[ln.to].links
                            if b.to == req.id and b.kind == "similar"]
                    assert back, f"{req.id} -> {ln.to} not symmetric"

    def test_graph_clusters_and_summary(self):
        analysis = analyze_document(document=_fixture_doc())
        graph = RequirementGraph(analysis.requirements)
        assert graph.summary()["requirements"] == 14
        assert graph.summary()["similar_links"] >= 1
        assert graph.summary()["depends_links"] == 1
        assert graph.summary()["contradictions"] == 0
        clusters = graph.clusters()
        assert any({"REQ-10", "REQ-11", "REQ-12"} <= set(c)
                   for c in clusters)
        assert graph.contradictions() == []
        assert graph.neighbors("REQ-10")


# ── brain rung: judgments with span citations, or nothing ────────────────

class TestBrainRung:
    def test_brain_contradiction_is_a_labelled_judgment(self):
        analysis = analyze_document(document=_fixture_doc(),
                                    llm_fn=_fake_llm_contradiction)
        graph = RequirementGraph(analysis.requirements)
        assert graph.summary()["contradictions"] == 1
        contra = graph.contradictions()[0]
        assert {contra["a"], contra["b"]} == {"REQ-15", "REQ-16"}
        assert contra["judgment"] is True
        assert contra["method"] == "brain"
        assert "REQ-15" in contra["reason"]

    def test_brain_rung_without_a_brain_refuses_honestly(self):
        analysis = analyze_document(document=_fixture_doc())
        assert analysis.brain_rung == "skipped: no brain available"
        assert RequirementGraph(analysis.requirements).contradictions() == []

    def test_brain_output_with_bogus_span_is_dropped(self):
        analysis = analyze_document(document=_fixture_doc(),
                                    llm_fn=_fake_llm_bogus_span)
        texts = [r.text for r in analysis.requirements]
        assert not any("teleport" in t for t in texts)
        assert len(analysis.requirements) == 14

    def test_brain_garbage_is_a_note_not_a_crash(self):
        analysis = analyze_document(document=_fixture_doc(),
                                    llm_fn=_fake_llm_garbage)
        assert analysis.ok
        assert "brain" in (analysis.brain_rung or "").lower()


# ── corpus: the lens on her own documents ────────────────────────────────

class TestCorpus:
    def test_two_directives_analyzed_together(self):
        doc_a = Document(id="doc-a", format="text", title="directive A",
                         sections=[Section(
                             level=1, heading="Rules",
                             text="REQ-A1: The system shall retain all "
                                  "user data indefinitely.")])
        doc_b = Document(id="doc-b", format="text", title="directive B",
                         sections=[Section(
                             level=1, heading="Rules",
                             text="REQ-B1: The system shall delete all "
                                  "user data after 30 days.")])
        analysis = analyze_corpus([doc_a, doc_b],
                                  llm_fn=_fake_llm_contradiction)
        # the stub names REQ-15/REQ-16, which are not in this corpus —
        # a judgment naming ids that do not exist must be dropped.
        assert len(analysis.requirements) == 2
        assert {r.doc_ref for r in analysis.requirements} == {
            "doc-a", "doc-b"}
        assert RequirementGraph(analysis.requirements).contradictions() == []

    def test_corpus_keeps_doc_attribution(self):
        analysis = analyze_corpus([_fixture_doc(), _fixture_doc()])
        assert len(analysis.requirements) == 28
        assert analysis.summary["documents"] == 2


# ── store, report, reply ─────────────────────────────────────────────────

class TestStoreAndReport:
    def test_store_round_trip(self, tmp_path):
        store = open_requirements_store(str(tmp_path))
        analysis = analyze_document(document=_fixture_doc())
        store.save(analysis)
        got = store.get(analysis.analysis_id)
        assert got is not None
        assert len(got.requirements) == 14
        assert got.to_dict() == analysis.to_dict()
        listed = store.list()
        assert len(listed) == 1
        assert listed[0]["analysis_id"] == analysis.analysis_id
        assert listed[0]["requirements"] == 14

    def test_shapes_round_trip(self):
        analysis = analyze_document(document=_fixture_doc())
        clone = RequirementAnalysis.from_dict(analysis.to_dict())
        assert clone.to_dict() == analysis.to_dict()

    def test_report_names_flags_evidence_and_graph(self):
        analysis = analyze_document(
            document=_fixture_doc(), llm_fn=_fake_llm_contradiction)
        md = render_report_markdown(analysis)
        assert "REQ-2" in md and "appropriately" in md
        assert "REQ-15" in md and "REQ-16" in md
        assert "contradiction" in md.lower()
        assert "clusters" in md.lower()

    def test_write_report_lands_a_file(self, tmp_path):
        analysis = analyze_document(document=_fixture_doc())
        path = write_report(analysis, str(tmp_path))
        assert path.exists()
        assert "REQ-1" in path.read_text(encoding="utf-8")

    def test_reply_is_plain_and_complete(self):
        analysis = analyze_document(document=_fixture_doc())
        reply = format_requirements_reply(analysis)
        assert "14 requirements" in reply
        assert "REQ-2" in reply
        assert "appropriately" in reply


# ── tool surface ─────────────────────────────────────────────────────────

class _FakeSettings:
    def __init__(self, home: str):
        self.home_path = home
        self.home = home


class _FakeCtx:
    """Minimal context: settings with a home_path, nothing else."""

    def __init__(self, home: str):
        self.settings = _FakeSettings(home)
        self.router = None
        self.tools = None


class _FakeRegistry:
    """Minimal stand-in for the tool registry: records the tool."""

    def __init__(self, home: str = ""):
        self.context = _FakeCtx(home)
        self.tools: dict[str, Any] = {}

    def register(self, name, fn=None, **kw):
        def deco(f):
            self.tools[name] = f
            return f
        if fn is not None:
            return deco(fn)
        return deco


class TestTool:
    def test_tool_registers_and_analyzes_text(self, tmp_path):
        from nomorals.tools.requirements import register

        registry = _FakeRegistry(str(tmp_path))
        register(registry)
        assert "analyze_requirements" in registry.tools
        fn = registry.tools["analyze_requirements"]
        out = fn(text=FIXTURE_SPEC)
        assert out["ok"] is True
        assert len(out["requirements"]) == 14
        assert "reply" in out
        assert out["analysis_id"]

    def test_tool_list_and_show(self, tmp_path):
        from nomorals.tools.requirements import register

        registry = _FakeRegistry(str(tmp_path))
        register(registry)
        fn = registry.tools["analyze_requirements"]
        made = fn(text=FIXTURE_SPEC)
        listed = fn(action="list")
        assert len(listed["analyses"]) == 1
        shown = fn(action="show", analysis_id=made["analysis_id"])
        assert shown["ok"] is True
        assert len(shown["requirements"]) == 14

    def test_tool_report_action(self, tmp_path):
        from nomorals.tools.requirements import register

        registry = _FakeRegistry(str(tmp_path))
        register(registry)
        fn = registry.tools["analyze_requirements"]
        made = fn(text=FIXTURE_SPEC)
        report = fn(action="report", analysis_id=made["analysis_id"])
        assert report["ok"] is True
        assert report["report_path"].endswith(".md")
        assert "REQ-1" in report["report"]

    def test_tool_needs_input(self, tmp_path):
        from nomorals.tools.requirements import register

        registry = _FakeRegistry(str(tmp_path))
        register(registry)
        fn = registry.tools["analyze_requirements"]
        out = fn()
        assert out["ok"] is False
        assert out["error"]

    def test_tool_wired_into_the_real_registry(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register_builtins()
        assert "analyze_requirements" in registry.names()
        assert registry.get("analyze_requirements") is not None
