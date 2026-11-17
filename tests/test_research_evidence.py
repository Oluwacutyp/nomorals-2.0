"""Elicit organ tests: evidence tables + systematic-review workflow.

The organ is nomorals/research/extraction.py + screening.py — the
Elicit core on Devon's own research spine: ask a research question,
get a paper set from open scholarly sources, screen it against
inclusion criteria, extract a cited cell matrix (claim / method /
result / stance per paper), and synthesise what agrees, what
disagrees and what is still unanswered.

The one structural rule, from the report and from grounded.py: a
cell without a quote is not allowed to exist. Every cell carries a
verbatim quote from the paper it came from, and the table drops any
cell whose quote does not string-match its source paper.

Fixture papers are hand-written abstracts (no live network in
tests); retrieval parsing is tested with injected fetch callables
returning fixture payloads.
"""

import pytest

from nomorals.research.extraction import (
    Cell,
    EvidenceReviewStore,
    EvidenceTable,
    ExtractionColumn,
    ExtractionSchema,
    Paper,
    PaperSet,
    answer_question,
    default_schema,
    export_csv,
    export_ris,
    extract_cells,
    format_evidence_reply,
    paper_id_for,
    retrieve_papers,
    synthesize,
)
from nomorals.research.screening import (
    InclusionCriteria,
    screen_papers,
)

QUESTION = "Does vitamin D supplementation reduce respiratory infections?"

ABSTRACT_A = (
    "We conducted a randomized controlled trial of 240 adults with "
    "vitamin D deficiency. Participants received daily vitamin D3 or "
    "placebo for 12 weeks. Vitamin D supplementation reduced "
    "respiratory infections by 22 percent compared with placebo. "
    "No serious adverse events were reported."
)
ABSTRACT_B = (
    "In a prospective cohort study of 1150 adults followed for two "
    "years, we found no significant difference in respiratory "
    "infection rates between participants with sufficient vitamin D "
    "and those with deficiency. The findings do not support routine "
    "supplementation for infection prevention."
)


def paper_a() -> Paper:
    return Paper(
        title="Vitamin D supplementation and respiratory infection: "
              "a randomized controlled trial",
        authors=["A. Smith", "B. Jones"], year=2023,
        doi="10.1000/vitd-rct", url="https://doi.org/10.1000/vitd-rct",
        source="OpenAlex", abstract=ABSTRACT_A, citations=42)


def paper_b() -> Paper:
    return Paper(
        title="Vitamin D and respiratory infections: "
              "a prospective cohort study",
        authors=["C. Lee"], year=2022,
        doi="10.1000/vitd-cohort",
        url="https://doi.org/10.1000/vitd-cohort",
        source="Semantic Scholar", abstract=ABSTRACT_B, citations=17)


def paper_no_abstract() -> Paper:
    return Paper(
        title="Vitamin D: an overview", authors=["D. Ray"], year=2024,
        doi="10.1000/vitd-overview",
        url="https://doi.org/10.1000/vitd-overview",
        source="Crossref", abstract="", citations=3)


def paper_old() -> Paper:
    return Paper(
        title="Early observations on vitamin D and colds",
        authors=["E. Old"], year=2010, doi="10.1000/vitd-old",
        url="https://doi.org/10.1000/vitd-old",
        source="OpenAlex",
        abstract="A small case series noted fewer colds among "
                 "patients given vitamin D.",
        citations=5)


def paper_no_year() -> Paper:
    return Paper(
        title="Vitamin D and immunity: a review", authors=["F. New"],
        year=None, doi="10.1000/vitd-review",
        url="https://doi.org/10.1000/vitd-review",
        source="arXiv",
        abstract="This review surveys the evidence linking vitamin D "
                 "status to immune function.",
        citations=None)


def fixture_set() -> PaperSet:
    return PaperSet(
        id="test-set", query=QUESTION,
        papers=[paper_a(), paper_b(), paper_no_abstract(), paper_old(),
                paper_no_year()])


# ── shapes ────────────────────────────────────────────────────────────────

class TestPaperShape:
    def test_id_prefers_doi(self):
        assert paper_a().id == "10.1000/vitd-rct"
        assert paper_id_for("Some Title", "", "") == paper_id_for(
            "Some Title", "", "")

    def test_text_is_title_plus_abstract(self):
        p = paper_a()
        assert p.text.startswith(p.title)
        assert ABSTRACT_A in p.text

    def test_round_trip(self):
        p = paper_a()
        assert Paper.from_dict(p.to_dict()) == p


class TestCellRule:
    def test_cell_without_quote_cannot_exist(self):
        with pytest.raises(ValueError):
            Cell(paper_id="x", column="claim", value="something",
                 quote="", location="abstract")

    def test_forged_quote_dropped_by_table(self):
        forged = Cell(paper_id=paper_a().id, column="claim",
                      value="vitamin D cures everything",
                      quote="vitamin D cures everything, trust me",
                      location="abstract")
        table = EvidenceTable(paper_set=fixture_set(),
                              schema=default_schema(QUESTION),
                              cells=[forged])
        assert table.dropped_unverified == 1
        assert table.verified_cells == []

    def test_verbatim_quote_survives(self):
        quote = ("Vitamin D supplementation reduced respiratory "
                 "infections by 22 percent compared with placebo.")
        cell = Cell(paper_id=paper_a().id, column="result",
                    value=quote, quote=quote, location="abstract")
        table = EvidenceTable(paper_set=fixture_set(),
                              schema=default_schema(QUESTION),
                              cells=[cell])
        assert table.dropped_unverified == 0
        assert len(table.verified_cells) == 1


# ── extraction ────────────────────────────────────────────────────────────

class TestExtractCells:
    def test_claim_method_result_found_verbatim(self):
        cells = extract_cells(paper_a(), default_schema(QUESTION))
        by_col = {c.column: c for c in cells}
        assert "claim" in by_col and "method" in by_col
        assert "result" in by_col
        for c in cells:
            assert c.quote in paper_a().text
        assert "reduced respiratory infections by 22 percent" in \
            by_col["result"].quote
        assert "randomized controlled trial" in by_col["method"].quote

    def test_number_column_parses_a_number(self):
        schema = ExtractionSchema(columns=[
            ExtractionColumn(name="sample size",
                             question="How many participants were studied?",
                             value_type="number",
                             keywords=["participants", "adults",
                                       "enrolled"])])
        cells = extract_cells(paper_a(), schema)
        assert len(cells) == 1
        assert cells[0].value == "240"

    def test_number_column_with_no_number_leaves_no_cell(self):
        schema = ExtractionSchema(columns=[
            ExtractionColumn(name="sample size",
                             question="How many participants were studied?",
                             value_type="number",
                             keywords=["participants", "adults",
                                       "enrolled"])])
        assert extract_cells(paper_no_year(), schema) == []

    def test_stance_yes_and_no(self):
        schema = ExtractionSchema(columns=[
            ExtractionColumn(name="stance",
                             question="Does this paper support vitamin D "
                                      "reducing infections?",
                             value_type="yes_no")])
        a = {c.paper_id: c for c in extract_cells(paper_a(), schema)}
        b = {c.paper_id: c for c in extract_cells(paper_b(), schema)}
        assert a[paper_a().id].value == "yes"
        assert b[paper_b().id].value == "no"

    def test_empty_abstract_extracts_nothing(self):
        assert extract_cells(paper_no_abstract(),
                             default_schema(QUESTION)) == []


# ── screening ─────────────────────────────────────────────────────────────

class TestScreening:
    def test_include_names_its_criterion(self):
        decisions = screen_papers(
            [paper_a(), paper_b()], InclusionCriteria())
        assert all(d.decision == "include" for d in decisions)
        assert all(d.criterion for d in decisions)
        assert all(d.reason for d in decisions)

    def test_no_abstract_excluded_with_reason(self):
        decisions = screen_papers(
            [paper_no_abstract()], InclusionCriteria())
        assert decisions[0].decision == "exclude"
        assert "abstract" in decisions[0].criterion
        assert "abstract" in decisions[0].reason

    def test_old_paper_excluded_by_year(self):
        decisions = screen_papers(
            [paper_old()], InclusionCriteria(min_year=2015))
        assert decisions[0].decision == "exclude"
        assert decisions[0].criterion == "year"
        assert "2010" in decisions[0].reason

    def test_unknown_year_is_borderline_not_silent(self):
        decisions = screen_papers(
            [paper_no_year()], InclusionCriteria(min_year=2015))
        assert decisions[0].decision == "borderline"
        assert "year" in decisions[0].criterion

    def test_excluded_keyword_refuses(self):
        decisions = screen_papers(
            [paper_a()],
            InclusionCriteria(keywords_exclude=["placebo"]))
        assert decisions[0].decision == "exclude"
        assert decisions[0].criterion == "keywords"

    def test_missing_required_keyword_excluded(self):
        decisions = screen_papers(
            [paper_b()],
            InclusionCriteria(keywords_include=["randomized"]))
        assert decisions[0].decision == "exclude"

    def test_screening_is_reproducible(self):
        crit = InclusionCriteria(min_year=2015)
        first = screen_papers(fixture_set().papers, crit)
        second = screen_papers(fixture_set().papers, crit)
        assert first == second


# ── synthesis ─────────────────────────────────────────────────────────────

def _full_table() -> EvidenceTable:
    pset = fixture_set()
    included = [paper_a(), paper_b()]
    pset = PaperSet(id="test-set", query=QUESTION, papers=included)
    cells = []
    for p in included:
        cells.extend(extract_cells(p, default_schema(QUESTION)))
    return EvidenceTable(paper_set=pset,
                         schema=default_schema(QUESTION), cells=cells)


class TestSynthesis:
    def test_disagreement_named_with_both_sides(self):
        out = synthesize(_full_table(), QUESTION)
        assert out["disagreements"], "expected a stance disagreement"
        stance = [d for d in out["disagreements"]
                  if d["column"] == "stance"]
        assert stance
        assert set(stance[0]["values"]) == {"yes", "no"}
        assert "unanswered" in out
        assert isinstance(out["summary"], str) and out["summary"]

    def test_agreement_when_all_say_yes(self):
        pset = PaperSet(id="t", query=QUESTION, papers=[paper_a()])
        cells = extract_cells(paper_a(), default_schema(QUESTION))
        out = synthesize(EvidenceTable(paper_set=pset,
                                       schema=default_schema(QUESTION),
                                       cells=cells), QUESTION)
        assert any(a["column"] == "stance"
                   for a in out["agreements"])
        assert out["disagreements"] == []

    def test_unanswered_column_named(self):
        schema = ExtractionSchema(columns=[
            ExtractionColumn(name="claim", question="claim?",
                             value_type="text"),
            ExtractionColumn(name="funding",
                             question="Who funded this study?",
                             value_type="text")])
        pset = PaperSet(id="t", query=QUESTION, papers=[paper_a()])
        cells = extract_cells(paper_a(), schema)
        out = synthesize(EvidenceTable(paper_set=pset, schema=schema,
                                       cells=cells), QUESTION)
        assert "funding" in out["unanswered"]


# ── retrieval parsing (injected fetchers, no network) ─────────────────────

def _inverted_index(text: str) -> dict[str, list[int]]:
    """Build an OpenAlex-style abstract inverted index from full text."""
    index: dict[str, list[int]] = {}
    for pos, word in enumerate(text.split()):
        index.setdefault(word, []).append(pos)
    return index


OPENALEX_FIXTURE = {
    "results": [{
        "doi": "https://doi.org/10.1000/vitd-rct",
        "title": "Vitamin D supplementation and respiratory infection: "
                 "a randomized controlled trial",
        "publication_year": 2023,
        "cited_by_count": 42,
        "abstract_inverted_index": _inverted_index(ABSTRACT_A),
    }]
}

S2_FIXTURE = {
    "data": [{
        "title": "Vitamin D and respiratory infections: "
                 "a prospective cohort study",
        "year": 2022, "citationCount": 17,
        "externalIds": {"DOI": "10.1000/vitd-cohort"},
        "url": "https://www.semanticscholar.org/paper/x",
        "abstract": ABSTRACT_B,
    }]
}

ARXIV_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Vitamin D and immunity: a review</title>
<id>http://arxiv.org/abs/2401.00001v1</id>
<summary>This review surveys the evidence linking vitamin D status
to immune function.</summary>
<published>2024-01-02T00:00:00Z</published></entry>
</feed>"""


class TestRetrievalParsing:
    def test_openalex_reconstructs_full_abstract(self):
        def fake_json(url, timeout=15):
            return OPENALEX_FIXTURE

        papers, notes = retrieve_papers(
            "vitamin D", limit=5, fetch_json=fake_json,
            sources=("openalex",))
        assert len(papers) == 1
        assert papers[0].abstract == ABSTRACT_A
        assert papers[0].doi == "10.1000/vitd-rct"
        assert papers[0].year == 2023
        assert papers[0].citations == 42

    def test_semantic_scholar_full_abstract(self):
        def fake_json(url, timeout=15):
            return S2_FIXTURE

        papers, _ = retrieve_papers(
            "vitamin D", limit=5, fetch_json=fake_json,
            sources=("semantic_scholar",))
        assert len(papers) == 1
        assert papers[0].abstract == ABSTRACT_B
        assert papers[0].doi == "10.1000/vitd-cohort"

    def test_arxiv_atom_parsing(self):
        def fake_text(url, timeout=15):
            return ARXIV_FIXTURE

        papers, _ = retrieve_papers(
            "vitamin D", limit=5, fetch_text=fake_text,
            sources=("arxiv",))
        assert len(papers) == 1
        assert papers[0].year == 2024
        assert "review surveys the evidence" in papers[0].abstract

    def test_failed_source_named_never_invented(self):
        def fake_json(url, timeout=15):
            raise OSError("network down")

        papers, notes = retrieve_papers(
            "vitamin D", limit=5, fetch_json=fake_json,
            sources=("openalex",))
        assert papers == []
        assert notes and "OpenAlex" in notes[0]
        assert "unavailable" in notes[0].lower()

    def test_dedup_by_doi_across_sources(self):
        def fake_json(url, timeout=15):
            if "openalex" in url:
                return OPENALEX_FIXTURE
            return S2_FIXTURE

        papers, _ = retrieve_papers(
            "vitamin D", limit=5, fetch_json=fake_json,
            sources=("openalex", "semantic_scholar"))
        assert len(papers) == 2  # distinct DOIs, both kept


# ── end to end ────────────────────────────────────────────────────────────

def _fake_fetchers():
    def fake_json(url, timeout=15):
        if "openalex" in url:
            return OPENALEX_FIXTURE
        return S2_FIXTURE

    def fake_text(url, timeout=15):
        return ARXIV_FIXTURE

    return fake_json, fake_text


class TestAnswerQuestion:
    def test_full_flow_with_evidence_and_synthesis(self, tmp_path):
        store = EvidenceReviewStore(str(tmp_path / "reviews.json"))
        fake_json, fake_text = _fake_fetchers()
        ans = answer_question(
            str(tmp_path), QUESTION, store=store, limit=5,
            fetch_json=fake_json, fetch_text=fake_text)
        assert ans["ok"] is True
        assert ans["papers_included"] >= 2
        assert ans["review_id"]
        assert "disagreement" in ans["reply"].lower() or \
            "disagree" in ans["reply"].lower()
        assert "22 percent" in ans["reply"]
        assert ans["csv_path"]
        saved = store.get(ans["review_id"])
        assert saved is not None
        assert len(saved["table"]["cells"]) > 0

    def test_unknown_question_refuses_honestly(self, tmp_path):
        store = EvidenceReviewStore(str(tmp_path / "reviews.json"))

        def empty_json(url, timeout=15):
            return {"results": [], "data": []}

        ans = answer_question(
            str(tmp_path), "xqzvk blorp frobnication",
            store=store, limit=5, fetch_json=empty_json,
            fetch_text=lambda url, timeout=15: "<feed/>")
        assert ans["ok"] is False
        assert "no papers" in ans["error"].lower()
        assert ans["papers_included"] == 0
        assert "no papers" in ans["reply"].lower()

    def test_reply_names_corpus_boundary(self, tmp_path):
        store = EvidenceReviewStore(str(tmp_path / "reviews.json"))
        fake_json, fake_text = _fake_fetchers()
        ans = answer_question(
            str(tmp_path), QUESTION, store=store, limit=5,
            fetch_json=fake_json, fetch_text=fake_text)
        assert "OpenAlex" in ans["reply"] or "Semantic Scholar" in ans["reply"]


class TestStoreAndExport:
    def test_store_round_trip(self, tmp_path):
        store = EvidenceReviewStore(str(tmp_path / "reviews.json"))
        table = _full_table()
        review = {"id": "r1", "question": QUESTION,
                  "table": table.to_dict()}
        store.save(review)
        assert store.get("r1")["question"] == QUESTION
        assert len(store.list()) == 1

    def test_csv_export_carries_quotes(self, tmp_path):
        table = _full_table()
        path = export_csv(table, str(tmp_path / "table.csv"))
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        assert "Vitamin D supplementation" in text
        assert "reduced respiratory infections by 22 percent" in text
        assert "paper" in text.splitlines()[0].lower()

    def test_ris_export_names_papers(self, tmp_path):
        path = export_ris([paper_a(), paper_b()],
                          str(tmp_path / "papers.ris"))
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        assert "TY  - JOUR" in text
        assert "10.1000/vitd-rct" in text
        assert "10.1000/vitd-cohort" in text

    def test_format_reply_is_plain_text(self):
        table = _full_table()
        reply = format_evidence_reply({
            "ok": True, "question": QUESTION, "review_id": "r1",
            "papers_screened": 2, "papers_included": 2,
            "table": table.to_dict(),
            "synthesis": synthesize(table, QUESTION),
            "sources": ["OpenAlex: 1 paper", "Semantic Scholar: 1 paper"],
            "csv_path": "/tmp/table.csv"})
        assert "Vitamin D" in reply
        assert "22 percent" in reply
        assert "disagree" in reply.lower()


# ── chat + CLI + registration ─────────────────────────────────────────────

class TestChatRegistration:
    def test_evidence_command_registered(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            PUBLIC_COMMANDS,
            caller_may_run,
            list_catalog,
        )

        assert "evidence" in CONTROL_COMMANDS
        assert CONTROL_COMMANDS["evidence"] == (1, None)
        assert "evidence" in PUBLIC_COMMANDS
        assert len(PUBLIC_COMMANDS) == 274
        assert "evidence" in COMMAND_DETAILS
        assert "/evidence" in COMMAND_DETAILS["evidence"]["usage"]
        assert "evidence" in LIST_ONELINERS
        assert "evidence" in [c for _, kinds in LIST_GROUPS for c in kinds]
        assert [c for _, kinds in LIST_GROUPS for c in kinds].count("evidence") == 1
        assert "evidence" in list_catalog()
        assert caller_may_run("evidence", False) is True

    def test_evidence_tool_registered(self):
        from nomorals.tools import evidence as evidence_tool
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        evidence_tool.register(reg)
        assert "evidence" in reg._tools

    def test_evidence_handler_answers_and_never_raises(
            self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        def fake_retrieve(query, limit=10, fetch_json=None,
                          fetch_text=None, sources=None):
            if "xqzvk" in query:
                return [], ["OpenAlex: no papers"]
            return [paper_a(), paper_b()], ["OpenAlex: 2 papers"]

        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers", fake_retrieve)
        runtime = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        runtime.settings = SimpleNamespace(home_path=str(tmp_path))
        handler = runtime._control_evidence
        assert "22 percent" in handler(QUESTION)
        assert "usage" in handler("").lower()
        assert "no papers" in handler("xqzvk blorp").lower()


class TestCli:
    @staticmethod
    def _context(tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
            settings=SimpleNamespace(home_path=str(tmp_path)))

    def test_nm_evidence_answers(self, capsys, tmp_path, monkeypatch):
        import argparse

        from nomorals.cmdline.commands.evidence import _cmd_evidence

        def fake_retrieve(query, limit=10, fetch_json=None,
                          fetch_text=None, sources=None):
            return [paper_a(), paper_b()], ["OpenAlex: 2 papers"]

        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers", fake_retrieve)
        args = argparse.Namespace(evidence_query=[QUESTION], json=False,
                                  limit=5)
        assert _cmd_evidence(args, self._context(tmp_path)) == 0
        assert "22 percent" in capsys.readouterr().out

    def test_nm_evidence_unknown_exits_1(self, capsys, tmp_path,
                                         monkeypatch):
        import argparse

        from nomorals.cmdline.commands.evidence import _cmd_evidence

        def fake_retrieve(query, limit=10, fetch_json=None,
                          fetch_text=None, sources=None):
            return [], ["OpenAlex: no papers"]

        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers", fake_retrieve)
        args = argparse.Namespace(evidence_query=["xqzvk blorp"],
                                  json=False, limit=5)
        assert _cmd_evidence(args, self._context(tmp_path)) == 1
        # Refusals go to stderr by design (like the other nm commands).
        assert "no papers" in capsys.readouterr().err

    def test_nm_evidence_list(self, capsys, tmp_path):
        import argparse

        from nomorals.cmdline.commands.evidence import _cmd_evidence

        args = argparse.Namespace(evidence_query=["list"], json=False,
                                  limit=5)
        assert _cmd_evidence(args, self._context(tmp_path)) == 0
