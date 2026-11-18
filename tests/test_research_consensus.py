"""Consensus organ tests: stance meter over the evidence organ.

The organ is nomorals/research/consensus.py — the Consensus core
(52-websites #35) on Devon's own research spine: given a research
question and the evidence organ's paper set, classify each paper's
finding as yes / no / possibly / mixed, weigh the papers by quality
(citations, recency, study design — never count alone), and report
the distribution as a meter with the dissent named by name.

Fixture papers are hand-written abstracts (no live network in
tests); retrieval is injected the same way the evidence tests do
it. The one structural rule carries over from extraction: every
verdict quotes its paper verbatim, and a verdict whose quote does
not string-match its source paper cannot exist.
"""

import pytest

from nomorals.research.consensus import (
    ConsensusStore,
    StanceVerdict,
    answer_consensus,
    build_meter,
    classify_stance,
    detect_study_design,
    format_consensus_reply,
    is_binary_question,
    paper_weight,
    what_would_change_it,
)
from nomorals.research.extraction import (
    Paper,
    default_schema,
    extract_cells,
)

QUESTION = "Does vitamin D supplementation reduce respiratory infections?"
NON_BINARY = "What does the evidence say about vitamin D and immunity?"

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
ABSTRACT_META = (
    "A meta-analysis of 14 randomized trials found vitamin D "
    "supplementation significantly reduced respiratory infections "
    "by 18 percent across 3200 participants. The effect was "
    "consistent across age groups."
)
ABSTRACT_COHORT_YES = (
    "In a cohort study of 640 adults, vitamin D supplementation "
    "improved respiratory outcomes and reduced infection episodes "
    "over one winter season."
)
ABSTRACT_MIXED = (
    "In the primary analysis, vitamin D supplementation reduced "
    "respiratory infections by 15 percent. However, in the subgroup "
    "with severe deficiency, we found no significant difference in "
    "infection rates."
)
ABSTRACT_REVIEW = (
    "This review surveys the evidence linking vitamin D status to "
    "immune function and respiratory health."
)
ABSTRACT_SLEEP = (
    "Sleep duration predicts next-day alertness in shift workers. "
    "Participants sleeping under six hours showed slower reaction "
    "times on every test."
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


def paper_meta() -> Paper:
    return Paper(
        title="Vitamin D for respiratory infections: a meta-analysis",
        authors=["G. Meta"], year=2024,
        doi="10.1000/vitd-meta",
        url="https://doi.org/10.1000/vitd-meta",
        source="OpenAlex", abstract=ABSTRACT_META, citations=120)


def paper_cohort_yes() -> Paper:
    return Paper(
        title="Vitamin D supplementation in a winter cohort",
        authors=["H. Cohort"], year=2021,
        doi="10.1000/vitd-cohort-yes",
        url="https://doi.org/10.1000/vitd-cohort-yes",
        source="Semantic Scholar", abstract=ABSTRACT_COHORT_YES,
        citations=8)


def paper_mixed() -> Paper:
    return Paper(
        title="Vitamin D and respiratory infections: mixed findings",
        authors=["I. Mixed"], year=2023,
        doi="10.1000/vitd-mixed",
        url="https://doi.org/10.1000/vitd-mixed",
        source="OpenAlex", abstract=ABSTRACT_MIXED, citations=5)


def paper_review() -> Paper:
    return Paper(
        title="Vitamin D and immunity: a review", authors=["F. New"],
        year=2024, doi="10.1000/vitd-review",
        url="https://doi.org/10.1000/vitd-review",
        source="arXiv", abstract=ABSTRACT_REVIEW, citations=3)


def paper_sleep() -> Paper:
    return Paper(
        title="Sleep duration and alertness in shift workers",
        authors=["J. Sleep"], year=2023,
        doi="10.1000/sleep-alert",
        url="https://doi.org/10.1000/sleep-alert",
        source="OpenAlex", abstract=ABSTRACT_SLEEP, citations=9)


def _verdict(paper: Paper, stance: str, weight: float = 1.0) -> StanceVerdict:
    return StanceVerdict(
        paper_id=paper.id, stance=stance,
        finding_quote=(paper.abstract.split(". ")[0] + "."),
        study_design=detect_study_design(paper),
        citations_count=paper.citations, year=paper.year,
        weight=weight)


# ── binary detection ──────────────────────────────────────────────────────

class TestIsBinary:
    def test_yes_no_openers_are_binary(self):
        for q in ("Does vitamin D reduce infections?",
                  "Is vitamin D effective for colds?",
                  "Can sleep improve memory?",
                  "Should adults supplement vitamin D?",
                  "Do probiotics help gut health?"):
            assert is_binary_question(q) is True, q

    def test_open_questions_are_not_binary(self):
        for q in ("What does the evidence say about vitamin D?",
                  "How does vitamin D work?",
                  "Why do infections rise in winter?",
                  "Which treatment works best for colds?"):
            assert is_binary_question(q) is False, q


# ── stance classification ─────────────────────────────────────────────────

class TestClassifyStance:
    def test_yes_and_no_from_the_fixture_pair(self):
        assert classify_stance(paper_a(), QUESTION).stance == "yes"
        assert classify_stance(paper_b(), QUESTION).stance == "no"

    def test_mixed_when_both_directions_decide(self):
        verdict = classify_stance(paper_mixed(), QUESTION)
        assert verdict.stance == "mixed"
        assert verdict.counter_quote, "mixed needs the opposing quote"
        assert "no significant difference" in verdict.counter_quote
        assert "reduced" in verdict.finding_quote

    def test_unclear_becomes_possibly(self):
        verdict = classify_stance(paper_review(), QUESTION)
        assert verdict.stance == "possibly"

    def test_unrelated_paper_has_no_verdict(self):
        assert classify_stance(paper_sleep(), QUESTION) is None

    def test_every_verdict_quote_is_verbatim(self):
        for paper in (paper_a(), paper_b(), paper_meta(),
                      paper_cohort_yes(), paper_mixed(), paper_review()):
            verdict = classify_stance(paper, QUESTION)
            assert verdict is not None
            assert verdict.finding_quote in paper.text
            if verdict.counter_quote:
                assert verdict.counter_quote in paper.text

    def test_stance_agrees_with_the_extraction_cell(self):
        schema = default_schema(QUESTION)
        for paper, expected in ((paper_a(), "yes"), (paper_b(), "no")):
            cells = extract_cells(paper, schema)
            stance_cell = next(c for c in cells if c.column == "stance")
            assert stance_cell.value == expected
            assert classify_stance(paper, QUESTION).stance == expected


# ── study design + weight ─────────────────────────────────────────────────

class TestDesignAndWeight:
    def test_design_detection(self):
        assert detect_study_design(paper_a()) == \
            "randomized controlled trial"
        assert detect_study_design(paper_meta()) == "meta-analysis"
        assert detect_study_design(paper_b()) == "cohort study"
        assert detect_study_design(paper_review()) == "review"
        assert detect_study_design(paper_sleep()) == "not stated"

    def test_weight_rewards_quality_not_just_count(self):
        # RCT, 42 citations, 2023: 1.0 + 0.5 design + 0.25 cites + 0.5 recent
        assert paper_weight(paper_a(), "randomized controlled trial",
                            now=2026) == 2.25
        # Meta-analysis, 120 citations, 2024: capped at 2.5
        assert paper_weight(paper_meta(), "meta-analysis",
                            now=2026) == 2.5
        # Cohort, 8 citations, 2021: 1.0 + 0.25 design + 0.5 recent
        assert paper_weight(paper_cohort_yes(), "cohort study",
                            now=2026) == 1.75

    def test_weight_penalises_no_abstract_and_old_papers(self):
        old = Paper(title="Old vitamin D notes", year=2001,
                    abstract="Vitamin D was given to patients.",
                    citations=1)
        assert paper_weight(old, "not stated", now=2026) == 1.0
        no_abs = Paper(title="Vitamin D overview", year=2024,
                       abstract="", citations=60)
        assert paper_weight(no_abs, "not stated", now=2026) == 1.75

    def test_weight_never_exceeds_cap_or_floor(self):
        rich = Paper(title="x", year=2026, abstract="Vitamin D reduced "
                     "infections.", citations=10000)
        assert paper_weight(rich, "meta-analysis", now=2026) == 2.5
        bare = Paper(title="x", year=1990, abstract="", citations=0)
        assert paper_weight(bare, "not stated", now=2026) == 0.75


# ── the meter ─────────────────────────────────────────────────────────────

def _meter_papers() -> list[StanceVerdict]:
    return [_verdict(paper_a(), "yes", 2.25),
            _verdict(paper_meta(), "yes", 2.5),
            _verdict(paper_cohort_yes(), "yes", 1.75),
            _verdict(paper_b(), "no", 2.0)]


class TestMeter:
    def test_moderate_consensus_from_the_fixture_set(self):
        meter = build_meter(_meter_papers())
        assert meter["n"] == 4
        assert meter["counts"] == {"yes": 3, "no": 1,
                                   "possibly": 0, "mixed": 0}
        assert meter["leading"] == "yes"
        assert meter["strength"] == "moderate"
        assert abs(meter["shares"]["yes"] - 6.5 / 8.5) < 0.001
        assert meter["dissent"] == ["no"]

    def test_strong_when_the_field_agrees(self):
        meter = build_meter([_verdict(paper_a(), "yes", 2.0),
                             _verdict(paper_meta(), "yes", 2.0),
                             _verdict(paper_cohort_yes(), "yes", 2.0)])
        assert meter["strength"] == "strong"
        assert meter["dissent"] == []

    def test_split_when_the_field_is_even(self):
        meter = build_meter([_verdict(paper_a(), "yes", 2.0),
                             _verdict(paper_meta(), "yes", 2.0),
                             _verdict(paper_b(), "no", 2.0),
                             _verdict(paper_mixed(), "no", 2.0)])
        assert meter["strength"] == "split"
        assert meter["leading"] in ("yes", "no")

    def test_too_thin_below_three_classified(self):
        meter = build_meter([_verdict(paper_a(), "yes", 2.0),
                             _verdict(paper_b(), "no", 2.0)])
        assert meter["strength"] == "too thin"
        assert meter["n"] == 2

    def test_empty_set_is_too_thin_not_a_consensus(self):
        meter = build_meter([])
        assert meter["strength"] == "too thin"
        assert meter["leading"] == ""


class TestWhatWouldChangeIt:
    def test_thin_meter_names_the_shortfall(self):
        meter = build_meter([_verdict(paper_a(), "yes", 2.0)])
        text = what_would_change_it(meter)
        assert "more papers" in text
        assert "3" in text

    def test_split_meter_names_the_flip_gap(self):
        meter = build_meter([_verdict(paper_a(), "yes", 2.0),
                             _verdict(paper_meta(), "yes", 2.0),
                             _verdict(paper_b(), "no", 2.0),
                             _verdict(paper_mixed(), "mixed", 1.0)])
        text = what_would_change_it(meter)
        assert "flip" in text

    def test_unanimous_meter_names_the_overturn_bar(self):
        meter = build_meter([_verdict(paper_a(), "yes", 2.0),
                             _verdict(paper_meta(), "yes", 2.0),
                             _verdict(paper_cohort_yes(), "yes", 2.0)])
        text = what_would_change_it(meter)
        assert "opposite" in text


# ── end to end ────────────────────────────────────────────────────────────

def _fake_retrieve(papers):
    def fake(query, limit=10, fetch_json=None, fetch_text=None,
             sources=None):
        return list(papers), [f"OpenAlex: {len(papers)} paper(s)"]
    return fake


class TestAnswerConsensus:
    def test_meter_matches_the_hand_labelled_set(
            self, tmp_path, monkeypatch):
        store = ConsensusStore(str(tmp_path / "consensus.json"))
        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers",
            _fake_retrieve([paper_a(), paper_meta(), paper_cohort_yes(),
                            paper_b()]))
        ans = answer_consensus(str(tmp_path), QUESTION, store=store,
                               now=2026)
        assert ans["ok"] is True
        meter = ans["meter"]
        assert meter["counts"] == {"yes": 3, "no": 1,
                                   "possibly": 0, "mixed": 0}
        assert meter["leading"] == "yes"
        assert meter["strength"] == "moderate"
        assert ans["evidence_review_id"]
        assert "MODERATE" in ans["reply"]
        assert "does vitamin d" in ans["reply"].lower()
        # Dissent is named by name, with its quote.
        assert "prospective cohort study" in ans["reply"]
        assert "no significant difference" in ans["reply"]
        # What would change it is stated.
        assert "What would change it" in ans["reply"]
        # The n is always shown — a meter over 4 never wears 60's authority.
        assert "4 papers" in ans["reply"]
        assert "OpenAlex" in ans["reply"]
        saved = store.get(ans["review_id"])
        assert saved is not None
        assert saved["meter"]["leading"] == "yes"

    def test_mixed_and_possibly_papers_land_in_the_meter(
            self, tmp_path, monkeypatch):
        store = ConsensusStore(str(tmp_path / "consensus.json"))
        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers",
            _fake_retrieve([paper_a(), paper_meta(), paper_cohort_yes(),
                            paper_b(), paper_mixed(), paper_review()]))
        ans = answer_consensus(str(tmp_path), QUESTION, store=store,
                               now=2026)
        assert ans["ok"] is True
        assert ans["meter"]["counts"]["mixed"] == 1
        assert ans["meter"]["counts"]["possibly"] == 1
        assert ans["meter"]["n"] == 6

    def test_non_binary_question_gets_prose_not_a_meter(
            self, tmp_path, monkeypatch):
        store = ConsensusStore(str(tmp_path / "consensus.json"))
        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers",
            _fake_retrieve([paper_a(), paper_b()]))
        ans = answer_consensus(str(tmp_path), NON_BINARY, store=store,
                               now=2026)
        assert ans["ok"] is True
        assert ans["meter"] is None
        assert "no meter" in ans["reply"].lower()
        assert "disagree" in ans["reply"].lower()

    def test_unknown_question_refuses_honestly(
            self, tmp_path, monkeypatch):
        store = ConsensusStore(str(tmp_path / "consensus.json"))
        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers",
            _fake_retrieve([]))
        ans = answer_consensus(str(tmp_path), "xqzvk blorp frobnication",
                               store=store, now=2026)
        assert ans["ok"] is False
        assert "no papers" in ans["error"].lower()
        assert ans["meter"] is None

    def test_deep_asks_for_more_papers(self, tmp_path, monkeypatch):
        store = ConsensusStore(str(tmp_path / "consensus.json"))
        seen: dict[str, int] = {}

        def fake(query, limit=10, fetch_json=None, fetch_text=None,
                 sources=None):
            seen["limit"] = limit
            return [paper_a(), paper_b()], ["OpenAlex: 2 papers"]

        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers", fake)
        ans = answer_consensus(str(tmp_path), QUESTION, store=store,
                               depth="deep", now=2026)
        assert ans["ok"] is True
        assert seen["limit"] == 25
        assert ans["depth"] == "deep"

    def test_bad_depth_is_refused(self, tmp_path):
        store = ConsensusStore(str(tmp_path / "consensus.json"))
        with pytest.raises(ValueError):
            answer_consensus(str(tmp_path), QUESTION, store=store,
                             depth="pro")

    def test_empty_question_is_refused(self, tmp_path):
        store = ConsensusStore(str(tmp_path / "consensus.json"))
        ans = answer_consensus(str(tmp_path), "   ", store=store)
        assert ans["ok"] is False


class TestStore:
    def test_store_round_trip(self, tmp_path):
        store = ConsensusStore(str(tmp_path / "consensus.json"))
        review = {"id": "cs1", "question": QUESTION, "meter": None,
                  "created_at": 1.0}
        store.save(review)
        assert store.get("cs1")["question"] == QUESTION
        assert len(store.list()) == 1
        assert store.latest()["id"] == "cs1"

    def test_reply_traces_every_verdict_to_its_paper(self):
        verdicts = [classify_stance(paper_a(), QUESTION),
                    classify_stance(paper_b(), QUESTION)]
        meter = build_meter([v for v in verdicts if v])
        reply = format_consensus_reply({
            "ok": True, "question": QUESTION, "review_id": "cs1",
            "depth": "quick", "binary": True,
            "evidence_review_id": "ev1",
            "papers_screened": 2, "papers_included": 2,
            "meter": meter, "verdicts": [v.to_dict() for v in verdicts
                                         if v],
            "papers": [paper_a().to_dict(), paper_b().to_dict()],
            "synthesis": {"summary": "disagree on stance: yes vs no",
                          "disagreements": [{"column": "stance",
                                             "values": ["yes", "no"]}],
                          "agreements": [], "unanswered": []},
            "sources": ["OpenAlex: 2 paper(s)"]})
        assert "22 percent" in reply
        assert "no significant difference" in reply
        assert "Consensus" in reply


# ── chat + CLI + registration ─────────────────────────────────────────────

class TestChatRegistration:
    def test_consensus_command_registered(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            PUBLIC_COMMANDS,
            caller_may_run,
            list_catalog,
        )

        assert "consensus" in CONTROL_COMMANDS
        assert CONTROL_COMMANDS["consensus"] == (1, None)
        assert "consensus" in PUBLIC_COMMANDS
        assert len(PUBLIC_COMMANDS) == 277
        assert "consensus" in COMMAND_DETAILS
        assert "/consensus" in COMMAND_DETAILS["consensus"]["usage"]
        assert "consensus" in LIST_ONELINERS
        assert "consensus" in [c for _, kinds in LIST_GROUPS
                               for c in kinds]
        assert [c for _, kinds in LIST_GROUPS for c in kinds].count(
            "consensus") == 1
        assert "consensus" in list_catalog()
        assert caller_may_run("consensus", False) is True

    def test_consensus_tool_registered(self):
        from nomorals.tools import consensus as consensus_tool
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        consensus_tool.register(reg)
        assert "consensus" in reg._tools

    def test_consensus_handler_answers_and_never_raises(
            self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        def fake_retrieve(query, limit=10, fetch_json=None,
                          fetch_text=None, sources=None):
            if "xqzvk" in query:
                return [], ["OpenAlex: no papers"]
            return [paper_a(), paper_b()], ["OpenAlex: 2 papers"]

        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers",
            fake_retrieve)
        runtime = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        runtime.settings = SimpleNamespace(home_path=str(tmp_path))
        handler = runtime._control_consensus
        assert "no consensus reviews" in handler("list").lower()
        assert "Consensus" in handler(QUESTION)
        assert "usage" in handler("").lower()
        assert "no papers" in handler("xqzvk blorp").lower()


class TestCli:
    @staticmethod
    def _context(tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
            settings=SimpleNamespace(home_path=str(tmp_path)))

    def test_nm_consensus_answers(self, capsys, tmp_path, monkeypatch):
        import argparse

        from nomorals.cmdline.commands.consensus import _cmd_consensus

        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers",
            _fake_retrieve([paper_a(), paper_meta(), paper_cohort_yes(),
                            paper_b()]))
        args = argparse.Namespace(consensus_query=[QUESTION], json=False,
                                  limit=None, deep=False)
        assert _cmd_consensus(args, self._context(tmp_path)) == 0
        assert "Consensus" in capsys.readouterr().out

    def test_nm_consensus_unknown_exits_1(self, capsys, tmp_path,
                                          monkeypatch):
        import argparse

        from nomorals.cmdline.commands.consensus import _cmd_consensus

        monkeypatch.setattr(
            "nomorals.research.extraction.retrieve_papers",
            _fake_retrieve([]))
        args = argparse.Namespace(consensus_query=["xqzvk blorp"],
                                  json=False, limit=None, deep=False)
        assert _cmd_consensus(args, self._context(tmp_path)) == 1
        assert "no papers" in capsys.readouterr().err

    def test_nm_consensus_list(self, capsys, tmp_path):
        import argparse

        from nomorals.cmdline.commands.consensus import _cmd_consensus

        args = argparse.Namespace(consensus_query=["list"], json=False,
                                  limit=None, deep=False)
        assert _cmd_consensus(args, self._context(tmp_path)) == 0
