"""Tests for the Semantic Scholar paper-record organ (52-websites #37).

The organ extends the research family (extraction / consensus /
paper_graph) with the full paper record itself: authors, venue, year,
citation counts, influential citations named, a one-line TLDR, fields
of study, and the open-access PDF when one exists — plus topic feeds
that surface new papers. Fixtures are modeled on the real Semantic
Scholar Graph API payloads probed live 2026-11-17 (DOI
10.1038/nature12373): field names, TLDR shape, openAccessPdf shape
and the citations endpoint shape are the live ones. Nothing is ever
invented: an unknown paper, a dead source, or a record with no TLDR
and no PDF is refused or named honestly.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

# The organ does not exist yet on the pristine base — this import is
# the failing-first proof (collection error, RC=2).
from nomorals.research.paper_records import (
    CitationEdge,
    PaperRecord,
    check_feeds,
    classify_citation_intent,
    ensure_tldr,
    fetch_influential_citations,
    fetch_paper_record,
    format_record_reply,
    local_tldr,
    open_record_store,
    search_records,
)

S2_ID = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
DOI = "10.1038/nature12373"

#: The live record payload for the probe paper (field values real,
#: abstract shortened to two sentences for the fixture).
RECORD_PAYLOAD = {
    "paperId": S2_ID,
    "title": "Nanometer scale thermometry in a living cell",
    "abstract": (
        "The local temperature of a living cell can change in response "
        "to physiological processes. We demonstrate nanoscale thermometry "
        "using coherent manipulation of the electronic spin associated "
        "with nitrogen-vacancy colour centres in diamond."
    ),
    "venue": "Nature",
    "year": 2013,
    "authors": [
        {"authorId": "1", "name": "G. Kucsko"},
        {"authorId": "2", "name": "P. Maurer"},
        {"authorId": "3", "name": "N. Yao"},
    ],
    "citationCount": 1842,
    "referenceCount": 41,
    "influentialCitationCount": 29,
    "externalIds": {"DOI": DOI, "ArXiv": "1304.1068"},
    "url": "https://www.semanticscholar.org/paper/" + S2_ID,
    "openAccessPdf": {
        "url": "https://www.nature.com/articles/nature12373.pdf",
        "status": "BRONZE",
    },
    "fieldsOfStudy": ["Medicine", "Physics", "Materials Science"],
    "tldr": {
        "model": "tldr@v2.0.0",
        "text": (
            "A new approach to nanoscale thermometry is demonstrated "
            "that uses coherent manipulation of the electronic spin "
            "associated with nitrogen-vacancy colour centres."
        ),
    },
}

#: A record with no TLDR and no open-access PDF — the honest thin path.
THIN_PAYLOAD = {
    "paperId": "thin000000000000000000000000000000000001",
    "title": "A note on unremarkable thermometry",
    "abstract": "",
    "venue": "Journal of Small Things",
    "year": 2020,
    "authors": [{"authorId": "9", "name": "A. Quiet"}],
    "citationCount": 2,
    "referenceCount": 3,
    "influentialCitationCount": 0,
    "externalIds": {"DOI": "10.9999/thin.1"},
    "url": "https://www.semanticscholar.org/paper/thin1",
    "openAccessPdf": None,
    "fieldsOfStudy": ["Physics"],
    "tldr": None,
}

CITATIONS_PAYLOAD = {
    "data": [
        {
            "citingPaper": {
                "paperId": "c1",
                "title": "Coherence-Enhanced Spatial Quantum Thermometry",
                "authors": [{"name": "L. Chen"}],
                "year": 2026,
                "citationCount": 4,
            },
            "contexts": ["We build on the nanoscale thermometry of Kucsko et al."],
            "intents": ["background"],
            "isInfluential": True,
        },
        {
            "citingPaper": {
                "paperId": "c2",
                "title": "Quantum biosensing illuminates infected cells",
                "authors": [{"name": "M. Osei"}],
                "year": 2026,
                "citationCount": 11,
            },
            "contexts": ["Using the method of Kucsko et al., we measured ..."],
            "intents": [],
            "isInfluential": True,
        },
        {
            "citingPaper": {
                "paperId": "c3",
                "title": "A passing mention in a review",
                "authors": [{"name": "R. Review"}],
                "year": 2019,
                "citationCount": 300,
            },
            "contexts": ["Thermometry has been reviewed elsewhere."],
            "intents": ["background"],
            "isInfluential": False,
        },
    ],
    "next": None,
    "offset": 0,
}

SEARCH_PAYLOAD = {
    "total": 2,
    "data": [
        {
            "paperId": "new000000000000000000000000000000000001",
            "title": "Fresh results on quantum thermometry",
            "year": 2026,
            "authors": [{"name": "L. Chen"}],
        },
        {
            "paperId": "old000000000000000000000000000000000001",
            "title": "An older thermometry survey",
            "year": 2021,
            "authors": [{"name": "R. Review"}],
        },
    ],
}


def _fake_fetch(mapping: dict[str, object]):
    """A fetcher that answers from a {url-substring: payload} map."""

    def fetch(url: str, timeout: int = 20):
        for needle, payload in mapping.items():
            if needle in url:
                return payload
        return None

    return fetch


def _record_fetcher():
    return _fake_fetch({
        "citations": CITATIONS_PAYLOAD,
        "paper/DOI%3A10.1038%2Fnature12373": RECORD_PAYLOAD,
        "paper/" + S2_ID: RECORD_PAYLOAD,
        "paper/search": {"total": 1, "data": [RECORD_PAYLOAD]},
        "paper/thin": THIN_PAYLOAD,
    })


class RecordParseTests(unittest.TestCase):
    def test_full_record_parses_every_field(self):
        record = PaperRecord.from_s2(RECORD_PAYLOAD)
        self.assertEqual(record.paper_id, S2_ID)
        self.assertEqual(record.title, "Nanometer scale thermometry in a living cell")
        self.assertEqual(record.authors, ["G. Kucsko", "P. Maurer", "N. Yao"])
        self.assertEqual(record.venue, "Nature")
        self.assertEqual(record.year, 2013)
        self.assertEqual(record.doi, DOI)
        self.assertEqual(record.arxiv_id, "1304.1068")
        self.assertEqual(record.citation_count, 1842)
        self.assertEqual(record.reference_count, 41)
        self.assertEqual(record.influential_citation_count, 29)
        self.assertEqual(record.fields_of_study,
                         ["Medicine", "Physics", "Materials Science"])
        self.assertEqual(record.tldr_model, "tldr@v2.0.0")
        self.assertTrue(record.tldr.startswith("A new approach"))
        self.assertEqual(record.tldr_source, "semantic-scholar")
        self.assertEqual(record.open_access_pdf,
                         "https://www.nature.com/articles/nature12373.pdf")

    def test_thin_record_parses_with_empty_tldr_and_no_pdf(self):
        record = PaperRecord.from_s2(THIN_PAYLOAD)
        self.assertEqual(record.tldr, "")
        self.assertEqual(record.tldr_source, "none")
        self.assertIsNone(record.open_access_pdf)
        self.assertEqual(record.citation_count, 2)

    def test_record_round_trips_through_dict(self):
        record = PaperRecord.from_s2(RECORD_PAYLOAD)
        again = PaperRecord.from_dict(record.to_dict())
        self.assertEqual(again, record)

    def test_missing_title_is_not_a_record(self):
        with self.assertRaises(ValueError):
            PaperRecord.from_s2({"paperId": "x"})


class FetchTests(unittest.TestCase):
    def test_fetch_by_doi(self):
        record, note = fetch_paper_record(DOI, fetch_json=_record_fetcher())
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.doi, DOI)
        self.assertIn("Semantic Scholar", note)

    def test_fetch_by_title_searches_then_fetches(self):
        record, note = fetch_paper_record(
            "Nanometer scale thermometry in a living cell",
            fetch_json=_record_fetcher())
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.paper_id, S2_ID)

    def test_unknown_paper_is_refused_honestly(self):
        record, note = fetch_paper_record(
            "a paper that does not exist anywhere at all",
            fetch_json=_fake_fetch(
                {"paper/search": {"total": 0, "data": []}}))
        self.assertIsNone(record)
        self.assertIn("no paper", note.lower())

    def test_dead_source_never_raises(self):
        record, note = fetch_paper_record(DOI, fetch_json=lambda u, timeout=20: None)
        self.assertIsNone(record)
        self.assertIn("unavailable", note.lower())

    def test_search_records_returns_thin_rows(self):
        rows, note = search_records(
            "quantum thermometry", fetch_json=_fake_fetch(
                {"paper/search": SEARCH_PAYLOAD}))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "Fresh results on quantum thermometry")
        self.assertIn("2", note)


class InfluentialTests(unittest.TestCase):
    def test_influential_citations_named_and_sorted(self):
        edges = fetch_influential_citations(S2_ID, fetch_json=_record_fetcher())
        self.assertEqual(len(edges), 2)
        titles = [e.citing_title for e in edges]
        self.assertIn("Coherence-Enhanced Spatial Quantum Thermometry", titles)
        self.assertIn("Quantum biosensing illuminates infected cells", titles)
        # Most-cited citer first: the 11-citation paper outranks the 4.
        self.assertEqual(edges[0].citing_title,
                         "Quantum biosensing illuminates infected cells")
        self.assertTrue(all(e.influential for e in edges))
        self.assertNotIn("A passing mention in a review", titles)

    def test_s2_intents_kept_when_present(self):
        edges = fetch_influential_citations(S2_ID, fetch_json=_record_fetcher())
        by_title = {e.citing_title: e for e in edges}
        self.assertEqual(
            by_title["Coherence-Enhanced Spatial Quantum Thermometry"].intents,
            ["background"])

    def test_citation_edge_round_trips(self):
        edge = CitationEdge.from_s2(CITATIONS_PAYLOAD["data"][0])
        self.assertEqual(CitationEdge.from_dict(edge.to_dict()), edge)


class IntentTests(unittest.TestCase):
    def test_methodology_cue(self):
        self.assertEqual(
            classify_citation_intent(["Using the method of Smith et al., we measured"]),
            "methodology")

    def test_background_cue(self):
        self.assertEqual(
            classify_citation_intent(["Thermometry was introduced by Smith et al."]),
            "background")

    def test_result_cue(self):
        self.assertEqual(
            classify_citation_intent(["Our results confirm the findings of Smith et al."]),
            "result")

    def test_no_cue_is_other(self):
        self.assertEqual(classify_citation_intent(["See Smith et al."]), "other")
        self.assertEqual(classify_citation_intent([]), "other")


class TldrTests(unittest.TestCase):
    def test_s2_tldr_preferred(self):
        record = PaperRecord.from_s2(RECORD_PAYLOAD)
        out = ensure_tldr(record)
        self.assertEqual(out.tldr_source, "semantic-scholar")
        self.assertTrue(out.tldr.startswith("A new approach"))

    def test_local_tldr_generated_when_s2_has_none(self):
        record = PaperRecord.from_s2({
            **THIN_PAYLOAD,
            "abstract": (
                "We measured the temperature of single cells with a new "
                "diamond sensor. The sensor resolves changes of a tenth "
                "of a degree across the cell."
            ),
        })
        out = ensure_tldr(record)
        self.assertEqual(out.tldr_source, "local")
        self.assertTrue(out.tldr)
        self.assertIn("temperature", out.tldr.lower())

    def test_no_abstract_means_no_tldr(self):
        record = PaperRecord.from_s2(THIN_PAYLOAD)
        out = ensure_tldr(record)
        self.assertEqual(out.tldr, "")
        self.assertEqual(out.tldr_source, "none")

    def test_local_tldr_is_one_line(self):
        text = local_tldr(
            "First sentence here. Second sentence follows. Third one too.")
        self.assertTrue(text)
        self.assertNotIn("\n", text)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.store = open_record_store(str(self.home))

    def tearDown(self):
        self.tmp.cleanup()

    def test_store_round_trip(self):
        record = PaperRecord.from_s2(RECORD_PAYLOAD)
        self.store.save(record)
        got = self.store.get(S2_ID)
        self.assertEqual(got, record)
        self.assertEqual([r["paper_id"] for r in self.store.list()], [S2_ID])

    def test_tldr_generated_once_and_reused(self):
        record = ensure_tldr(PaperRecord.from_s2({
            **THIN_PAYLOAD,
            "abstract": "We measured cell temperatures with a diamond sensor.",
        }))
        self.store.save(record)
        again = self.store.get(record.paper_id)
        assert again is not None
        self.assertEqual(again.tldr, record.tldr)
        self.assertEqual(again.tldr_source, "local")

    def test_watch_and_unwatch(self):
        self.store.watch("quantum thermometry")
        self.assertIn("quantum thermometry", self.store.feeds())
        self.store.unwatch("quantum thermometry")
        self.assertNotIn("quantum thermometry", self.store.feeds())

    def test_check_feeds_reports_only_new_papers(self):
        self.store.watch("quantum thermometry")
        first = check_feeds(self.store, fetch_json=_fake_fetch(
            {"paper/search": SEARCH_PAYLOAD}))
        self.assertEqual(len(first), 1)
        topic, new = first[0]
        self.assertEqual(topic, "quantum thermometry")
        self.assertEqual(len(new), 2)
        second = check_feeds(self.store, fetch_json=_fake_fetch(
            {"paper/search": SEARCH_PAYLOAD}))
        self.assertEqual(second, [])


class ReplyTests(unittest.TestCase):
    def test_reply_names_record_and_influential(self):
        record = PaperRecord.from_s2(RECORD_PAYLOAD)
        record.influential = fetch_influential_citations(
            S2_ID, fetch_json=_record_fetcher())
        reply = format_record_reply({"ok": True, "record": record})
        self.assertIn("Nanometer scale thermometry", reply)
        self.assertIn("Nature", reply)
        self.assertIn("1,842", reply)
        self.assertIn("A new approach", reply)
        self.assertIn("Coherence-Enhanced Spatial Quantum Thermometry", reply)
        self.assertIn("nature.com/articles/nature12373.pdf", reply)

    def test_reply_names_missing_pdf_honestly(self):
        record = PaperRecord.from_s2(THIN_PAYLOAD)
        reply = format_record_reply({"ok": True, "record": record})
        self.assertIn("no open-access PDF", reply)
        self.assertIn("no TLDR", reply)

    def test_failed_answer_renders_the_reason(self):
        reply = format_record_reply(
            {"ok": False, "error": "no paper for that title"})
        self.assertIn("no paper for that title", reply)


class ChatRegistrationTests(unittest.TestCase):
    def test_registered_everywhere_public(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            PUBLIC_COMMANDS,
            caller_may_run,
            list_catalog,
            parse_control,
        )

        self.assertIn("scholar", PUBLIC_COMMANDS)
        self.assertEqual(275, len(PUBLIC_COMMANDS))
        self.assertIn("scholar", CONTROL_COMMANDS)
        self.assertIn("scholar", COMMAND_DETAILS)
        self.assertTrue(COMMAND_DETAILS["scholar"].get("what"))
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertIn("scholar", grouped)
        self.assertEqual(grouped.count("scholar"), 1)
        self.assertIn("/scholar <title or DOI>",
                      COMMAND_DETAILS["scholar"]["usage"])
        self.assertEqual(parse_control("/scholar foo").kind, "scholar")
        self.assertTrue(caller_may_run("scholar", is_owner=False))
        catalog = list_catalog()
        self.assertIn("/scholar", catalog)


class ToolRegistrationTests(unittest.TestCase):
    def test_paper_record_tool_registered(self):
        from nomorals.tools.paper_record import register
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register(registry)
        self.assertIn("paper_record", registry.names())


class CliTests(unittest.TestCase):
    def test_cli_record_and_list_round_trip(self):
        import argparse

        from nomorals.cmdline.commands.scholar import _cmd_scholar

        home = Path(tempfile.mkdtemp())
        args = argparse.Namespace(
            scholar_query=["Nanometer scale thermometry in a living cell"],
            json=False)
        context = mock.Mock()
        context.settings = mock.Mock()
        context.settings.home_path = str(home)
        context.settings.home = None
        with mock.patch(
                "nomorals.research.paper_records._default_fetch_json",
                _record_fetcher()):
            rc = _cmd_scholar(args, context)
        self.assertEqual(rc, 0)
        store = open_record_store(str(home))
        self.assertEqual(len(store.list()), 1)
        args = argparse.Namespace(scholar_query=["list"], json=False)
        self.assertEqual(_cmd_scholar(args, context), 0)

    def test_cli_unknown_paper_exits_1(self):
        import argparse

        from nomorals.cmdline.commands.scholar import _cmd_scholar

        home = Path(tempfile.mkdtemp())
        context = mock.Mock()
        context.settings = mock.Mock()
        context.settings.home_path = str(home)
        context.settings.home = None
        args = argparse.Namespace(
            scholar_query=["a paper that does not exist anywhere at all"],
            json=False)
        with mock.patch(
                "nomorals.research.paper_records._default_fetch_json",
                _fake_fetch({})):
            rc = _cmd_scholar(args, context)
        self.assertEqual(rc, 1)

    def test_cli_no_query_exits_2(self):
        import argparse

        from nomorals.cmdline.commands.scholar import _cmd_scholar

        context = mock.Mock()
        rc = _cmd_scholar(argparse.Namespace(scholar_query=[], json=False),
                          context)
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
