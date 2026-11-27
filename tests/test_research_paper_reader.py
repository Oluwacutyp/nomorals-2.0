"""Paper reader — SciSpace (52-websites #38): read a paper, explain it.

The fixture paper is SYNTHETIC (an invented study about morning
light and sleep onset), written for these tests — no claim about any
real paper is made here. The organ must never invent a finding, a
quote, or an answer a paper does not carry; every finding and every
explanation carries the verbatim quote it came from.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from nomorals.research.paper_reader import (
    Explanation,
    PaperReadingStore,
    PaperText,
    acquire_paper,
    ask_paper,
    explain_paper,
    explain_span,
    format_citation,
    format_reading_reply,
    open_reading_store,
    paraphrase_span,
    read_paper,
    structure_text,
)

FIXTURE_TEXT = """Morning light and sleep onset in shift workers

Abstract
We ask whether bright morning light shortens sleep onset in night-shift
workers. Forty-two shift workers were assigned to a bright-light group
or a dim-light control group for two weeks. The bright-light group fell
asleep 23 minutes faster on average than the control group (p = 0.02).
Morning light is a cheap, practical lever for shift-worker sleep.

Introduction
Night-shift workers often struggle to fall asleep during the day. We
investigate whether bright morning light can shorten sleep onset.
Previous work shows light shifts the circadian clock, but few studies
test morning light directly in working shift staff.

Methods
We recruited 42 night-shift nurses from two hospitals. Participants
were assigned at random to bright light (10,000 lux for 30 minutes
each morning) or dim light (under 100 lux). Sleep onset was measured
with wrist actigraphy every night for 14 nights.

Results
The bright-light group fell asleep 23 minutes faster on average than
the control group (p = 0.02). We found that sleep onset improved by
18 percent in the bright-light group. The effect was largest in the
first week, when onset fell by 31 minutes. No adverse effects were
reported in either group.

The onset time follows a simple relation:

$$ T = T_0 + k \\log(L / L_0) $$

where T is sleep onset time in minutes, T_0 is the baseline onset,
L is morning light exposure in lux, and k is a fitted constant.

Table 1: Sleep onset by light group

| Group | n | Mean onset (min) | SD |
|-------|---|------------------|----|
| Bright light | 21 | 34 | 8 |
| Dim light | 21 | 57 | 11 |

Discussion
Our results show that bright morning light shortens sleep onset in
shift workers. The 23-minute gain matters because even small gains
compound across a working week. A limitation is that all participants
were nurses; other shift occupations may respond differently.

Conclusion
Bright morning light is a practical way to help shift workers fall
asleep faster. Future work should test the effect over months rather
than weeks.
"""


def _fixture_record():
    from nomorals.research.paper_records import PaperRecord

    return PaperRecord(
        paper_id="fixture001",
        title="Morning light and sleep onset in shift workers",
        authors=["Ada Lovelace", "Grace Hopper"],
        venue="Journal of Sleep Research",
        year=2024,
        doi="10.1111/jsr.12345",
        url="https://doi.org/10.1111/jsr.12345",
        abstract=("We ask whether bright morning light shortens sleep "
                  "onset in night-shift workers."),
        citation_count=12,
        reference_count=31,
        influential_citation_count=2,
        fields_of_study=["Medicine"],
        open_access_pdf="https://example.org/papers/jsr12345.pdf",
    )


def _fixture_paper() -> PaperText:
    return PaperText(
        paper_id="fixture001",
        title="Morning light and sleep onset in shift workers",
        authors=["Ada Lovelace", "Grace Hopper"],
        year=2024,
        doi="10.1111/jsr.12345",
        venue="Journal of Sleep Research",
        source="open-access PDF",
        text=FIXTURE_TEXT,
    )


class AcquireTests(unittest.TestCase):
    def test_acquire_from_doi_with_open_access_pdf(self):
        def fetch_record(query):
            self.assertEqual(query, "10.1111/jsr.12345")
            return _fixture_record(), "Semantic Scholar: full record"

        def fetch_text(url):
            self.assertIn("jsr12345.pdf", url)
            return FIXTURE_TEXT

        paper, note = acquire_paper(
            "10.1111/jsr.12345",
            fetch_record=fetch_record, fetch_text=fetch_text)
        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.title,
                         "Morning light and sleep onset in shift workers")
        self.assertEqual(paper.source, "open-access PDF")
        self.assertIn("23 minutes faster", paper.text)
        self.assertIn("open-access", note)

    def test_acquire_falls_back_to_abstract_when_no_pdf(self):
        record = _fixture_record()
        record.open_access_pdf = None

        def fetch_record(query):
            return record, "Semantic Scholar: full record"

        paper, note = acquire_paper(
            "10.1111/jsr.12345", fetch_record=fetch_record)
        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.source, "abstract only")
        self.assertIn("shortens sleep onset", paper.text)
        self.assertIn("abstract", note)

    def test_acquire_unknown_paper_is_honest(self):
        def fetch_record(query):
            return None, "Semantic Scholar: no paper for 'no such paper'."

        paper, note = acquire_paper("no such paper",
                                    fetch_record=fetch_record)
        self.assertIsNone(paper)
        self.assertIn("no paper", note)

    def test_acquire_dead_source_never_raises(self):
        def fetch_record(query):
            raise RuntimeError("network down")

        paper, note = acquire_paper("10.1111/jsr.12345",
                                    fetch_record=fetch_record)
        self.assertIsNone(paper)
        self.assertIn("unavailable", note)

    def test_acquire_pasted_text(self):
        paper, note = acquire_paper(FIXTURE_TEXT)
        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.source, "pasted text")
        self.assertEqual(paper.title,
                         "Morning light and sleep onset in shift workers")

    def test_acquire_local_file(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "paper.txt"
            path.write_text(FIXTURE_TEXT, encoding="utf-8")
            paper, note = acquire_paper(str(path))
            self.assertIsNotNone(paper)
            assert paper is not None
            self.assertEqual(paper.source, "local file")
            self.assertIn("23 minutes faster", paper.text)


class StructureTests(unittest.TestCase):
    def test_sections_detected_in_order(self):
        sections = structure_text(FIXTURE_TEXT)
        headings = [s.heading.lower() for s in sections]
        for wanted in ("abstract", "introduction", "methods",
                       "results", "discussion", "conclusion"):
            self.assertIn(wanted, headings)
        self.assertLess(headings.index("abstract"),
                        headings.index("results"))
        results = next(s for s in sections
                       if s.heading.lower() == "results")
        self.assertIn("23 minutes faster", results.text)

    def test_hard_parts_found(self):
        paper = _fixture_paper()
        explanation = explain_paper(paper)
        kinds = [h.kind for h in explanation.hard_parts]
        self.assertIn("equation", kinds)
        self.assertIn("table", kinds)
        equation = next(h for h in explanation.hard_parts
                        if h.kind == "equation")
        self.assertIn("equation 1", equation.span_ref)
        table = next(h for h in explanation.hard_parts
                     if h.kind == "table")
        self.assertIn("table 1", table.span_ref)


class ExplainTests(unittest.TestCase):
    def test_explain_end_to_end(self):
        paper = _fixture_paper()
        explanation = explain_paper(paper)
        self.assertTrue(explanation.overview.strip())
        self.assertIn("23 minutes", explanation.overview)
        self.assertTrue(explanation.why_it_matters.strip())
        self.assertGreaterEqual(len(explanation.findings), 2)
        for finding in explanation.findings:
            # every finding carries its verbatim quote from the paper
            self.assertIn(finding.quote, FIXTURE_TEXT)
            self.assertTrue(finding.section)
        self.assertTrue(any("23 minutes faster" in f.statement
                            for f in explanation.findings))

    def test_explain_hard_parts_carry_grounded_quotes(self):
        paper = _fixture_paper()
        explanation = explain_paper(paper)
        self.assertTrue(explanation.hard_parts)
        for part in explanation.hard_parts:
            self.assertTrue(part.plain_text.strip())
            self.assertTrue(part.grounded_quotes)
            for quote in part.grounded_quotes:
                self.assertIn(quote, FIXTURE_TEXT)
        equation = next(h for h in explanation.hard_parts
                        if h.kind == "equation")
        # the unpacking names a symbol the paper itself defines
        self.assertIn("T_0", equation.plain_text)

    def test_explain_span_by_ref(self):
        paper = _fixture_paper()
        result = explain_span(paper, "equation 1")
        self.assertTrue(result["ok"])
        part = result["explanation"]
        self.assertIsInstance(part, Explanation)
        self.assertEqual(part.kind, "equation")
        self.assertIn("sleep onset", part.plain_text)

    def test_explain_span_unknown_ref_is_honest(self):
        paper = _fixture_paper()
        result = explain_span(paper, "equation 9")
        self.assertFalse(result["ok"])
        self.assertIn("no equation 9", result["error"])

    def test_llm_overview_that_drops_a_number_is_discarded(self):
        paper = _fixture_paper()

        def llm_fn(prompt):
            # fluent but lossy: drops the 23-minute number
            return ("This paper is about light and sleep. It found that "
                    "morning light helps shift workers fall asleep.")

        explanation = explain_paper(paper, llm_fn=llm_fn)
        # the extractive overview survives; the lossy rewrite does not
        self.assertIn("23 minutes", explanation.overview)
        self.assertEqual(explanation.overview_source, "extractive")


class AskTests(unittest.TestCase):
    def test_ask_answered_from_the_paper(self):
        paper = _fixture_paper()
        result = ask_paper(paper, "How many participants were studied?")
        self.assertTrue(result["ok"])
        self.assertIn("42", result["answer"])
        self.assertTrue(result["quotes"])
        for quote in result["quotes"]:
            self.assertIn(quote, FIXTURE_TEXT)

    def test_ask_uncovered_question_refused_honestly(self):
        paper = _fixture_paper()
        result = ask_paper(paper, "What did they find about caffeine?")
        self.assertFalse(result["ok"])
        self.assertIn("does not cover", result["error"])
        self.assertEqual(result["quotes"], [])


class CitationTests(unittest.TestCase):
    def test_apa(self):
        record = _fixture_record()
        self.assertEqual(
            format_citation(record, "apa"),
            "Lovelace, A., & Hopper, G. (2024). Morning light and sleep "
            "onset in shift workers. Journal of Sleep Research. "
            "https://doi.org/10.1111/jsr.12345")

    def test_mla(self):
        record = _fixture_record()
        self.assertEqual(
            format_citation(record, "mla"),
            'Lovelace, Ada, and Grace Hopper. "Morning light and sleep '
            'onset in shift workers." Journal of Sleep Research, 2024, '
            'doi:10.1111/jsr.12345.')

    def test_chicago(self):
        record = _fixture_record()
        self.assertEqual(
            format_citation(record, "chicago"),
            'Lovelace, Ada, and Grace Hopper. "Morning light and sleep '
            'onset in shift workers." Journal of Sleep Research (2024). '
            'https://doi.org/10.1111/jsr.12345.')

    def test_bibtex(self):
        record = _fixture_record()
        self.assertEqual(
            format_citation(record, "bibtex"),
            "@article{lovelace2024morning,\n"
            "  author = {Lovelace, Ada and Hopper, Grace},\n"
            "  title = {Morning light and sleep onset in shift workers},\n"
            "  journal = {Journal of Sleep Research},\n"
            "  year = {2024},\n"
            "  doi = {10.1111/jsr.12345},\n"
            "  url = {https://doi.org/10.1111/jsr.12345}\n"
            "}")

    def test_unknown_style_is_honest(self):
        record = _fixture_record()
        with self.assertRaises(ValueError):
            format_citation(record, "vancouver")


class ParaphraseTests(unittest.TestCase):
    def test_paraphrase_keeps_numbers_and_entities(self):
        span = ("The bright-light group fell asleep 23 minutes faster on "
                "average than the control group (p = 0.02).")
        result = paraphrase_span(span)
        self.assertTrue(result["ok"])
        self.assertIn("23", result["paraphrase"])
        self.assertIn("0.02", result["paraphrase"])
        self.assertTrue(result["preserved"])

    def test_paraphrase_simplifies_jargon(self):
        span = ("We utilized a randomized methodology in order to "
                "demonstrate the effect.")
        result = paraphrase_span(span)
        self.assertTrue(result["ok"])
        self.assertIn("used", result["paraphrase"].lower())
        self.assertNotIn("utilized", result["paraphrase"].lower())

    def test_llm_paraphrase_that_drops_a_number_falls_back(self):
        span = ("Sleep onset improved by 18 percent in the bright-light "
                "group of 42 nurses.")

        def llm_fn(prompt):
            return "Sleep got better in the bright-light group."

        result = paraphrase_span(span, llm_fn=llm_fn)
        self.assertTrue(result["ok"])
        # the lossy rewrite is refused; the original span is quoted
        self.assertEqual(result["method"], "quoted")
        self.assertIn("18 percent", result["paraphrase"])
        self.assertIn("42", result["paraphrase"])


class StoreTests(unittest.TestCase):
    def test_store_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            store = PaperReadingStore(Path(d) / "paper_readings.json")
            paper = _fixture_paper()
            explanation = explain_paper(paper)
            store.save(paper, explanation)
            self.assertEqual(len(store.list()), 1)
            again = PaperReadingStore(Path(d) / "paper_readings.json")
            row = again.get("fixture001")
            self.assertIsNotNone(row)
            self.assertEqual(row["paper"]["title"], paper.title)
            self.assertTrue(row["explanation"]["overview"])

    def test_open_reading_store_default_path(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            store = open_reading_store(d)
            self.assertTrue(str(store.path).endswith("paper_readings.json"))


class ReadPaperTests(unittest.TestCase):
    def test_read_paper_end_to_end(self):
        import tempfile

        def fetch_record(query):
            return _fixture_record(), "Semantic Scholar: full record"

        def fetch_text(url):
            return FIXTURE_TEXT

        with tempfile.TemporaryDirectory() as d:
            result = read_paper(
                "10.1111/jsr.12345", home=d,
                fetch_record=fetch_record, fetch_text=fetch_text)
            self.assertTrue(result["ok"])
            self.assertIn("23 minutes", result["reply"])
            self.assertIn("Key findings", result["reply"])
            # the reading is stored for later questions
            store = open_reading_store(d)
            self.assertEqual(len(store.list()), 1)

    def test_read_paper_unknown_is_honest(self):
        def fetch_record(query):
            return None, "Semantic Scholar: no paper for 'no such paper'."

        result = read_paper("no such paper", fetch_record=fetch_record)
        self.assertFalse(result["ok"])
        self.assertIn("no paper", result["error"])
        self.assertEqual(format_reading_reply(result),
                         "paper read failed: Semantic Scholar: no paper "
                         "for 'no such paper'.")

    def test_read_paper_reply_names_its_source(self):
        import tempfile

        record = _fixture_record()
        record.open_access_pdf = None

        def fetch_record(query):
            return record, "Semantic Scholar: full record"

        with tempfile.TemporaryDirectory() as d:
            result = read_paper("10.1111/jsr.12345", home=d,
                                fetch_record=fetch_record)
            self.assertTrue(result["ok"])
            self.assertIn("abstract only", result["reply"])


class WiringTests(unittest.TestCase):
    def test_readpaper_registered_everywhere_public(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            PUBLIC_COMMANDS,
            caller_may_run,
            list_catalog,
            parse_control,
        )

        self.assertIn("readpaper", PUBLIC_COMMANDS)
        self.assertEqual(287, len(PUBLIC_COMMANDS))
        self.assertIn("readpaper", CONTROL_COMMANDS)
        self.assertIn("readpaper", COMMAND_DETAILS)
        self.assertTrue(COMMAND_DETAILS["readpaper"].get("what"))
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertIn("readpaper", grouped)
        self.assertEqual(grouped.count("readpaper"), 1)
        self.assertIn("/readpaper <DOI or title>",
                      COMMAND_DETAILS["readpaper"]["usage"])
        self.assertEqual(parse_control("/readpaper foo").kind, "readpaper")
        self.assertTrue(caller_may_run("readpaper", is_owner=False))
        catalog = list_catalog()
        self.assertIn("/readpaper", catalog)

    def test_paper_reader_tool_registered(self):
        from nomorals.tools.paper_reader import register
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register(registry)
        self.assertIn("paper_reader", registry.names())

    def test_cli_subparser_exists(self):
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        parsed = parser.parse_args(["readpaper", "list"])
        self.assertEqual(parsed.command, "readpaper")
        self.assertEqual(parsed.readpaper_query, ["list"])


class HandlerTests(unittest.TestCase):
    def _runtime(self, home):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        runtime = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        runtime.settings = SimpleNamespace(home_path=str(home))
        return runtime

    def test_handler_reads_lists_asks_and_cites(self):
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as d:
            patches = (
                mock.patch(
                    "nomorals.research.paper_records.fetch_paper_record",
                    return_value=(_fixture_record(),
                                  "Semantic Scholar: full record")),
                mock.patch(
                    "nomorals.research.paper_reader._default_fetch_text",
                    return_value=FIXTURE_TEXT),
            )
            with patches[0], patches[1]:
                runtime = self._runtime(d)
                handler = runtime._control_readpaper
                self.assertIn("usage", handler("").lower())
                reply = handler("10.1111/jsr.12345")
                self.assertIn("23 minutes", reply)
                self.assertIn("Key findings", reply)
                self.assertIn("morning light", handler("list").lower())
                asked = handler(
                    "ask fixture001 How many participants were studied?")
                self.assertIn("42", asked)
                cited = handler("cite fixture001 apa")
                self.assertIn("Lovelace", cited)
                explained = handler("explain fixture001 equation 1")
                self.assertIn("sleep onset", explained)

    def test_cli_reads_and_refuses_unknown(self):
        import argparse
        import tempfile
        from types import SimpleNamespace
        from unittest import mock

        from nomorals.cmdline.commands.paper_reader import _cmd_readpaper

        with tempfile.TemporaryDirectory() as d:
            context = SimpleNamespace(
                settings=SimpleNamespace(home_path=d))
            with mock.patch(
                    "nomorals.research.paper_records.fetch_paper_record",
                    return_value=(_fixture_record(),
                                  "Semantic Scholar: full record")), \
                    mock.patch(
                        "nomorals.research.paper_reader._default_fetch_text",
                        return_value=FIXTURE_TEXT):
                args = argparse.Namespace(
                    readpaper_query=["10.1111/jsr.12345"], json=False)
                self.assertEqual(_cmd_readpaper(args, context), 0)
            with mock.patch(
                    "nomorals.research.paper_records.fetch_paper_record",
                    return_value=(None, "Semantic Scholar: no paper for "
                                        "'no such paper'.")):
                args = argparse.Namespace(
                    readpaper_query=["no", "such", "paper"], json=False)
                self.assertEqual(_cmd_readpaper(args, context), 1)


if __name__ == "__main__":
    unittest.main()
