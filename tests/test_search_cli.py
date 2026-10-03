"""``nm search`` CLI: parser wiring, dispatch, output shapes, error paths."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.cmdline.commands.search import _cmd_search
from nomorals.cmdline.dispatch import _canonical_command
from nomorals.cmdline.parser import CLI_ALIASES, _parser
from nomorals.search.model import SearchResponse, SearchResult


def _args(argv):
    return _parser().parse_args(argv)


def _ctx():
    tmp = tempfile.mkdtemp(prefix="search-cli-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings)


def _run(argv, ctx):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _cmd_search(_args(argv), ctx)
    return rc, buf.getvalue()


def _resp():
    hit = SearchResult(query="dragons", title="Dragon Notes", snippet="fire",
                       source="memory", type="memory", score=1.0,
                       raw_score=0.8, timestamp=1700000000.0,
                       source_id="memory:m1")
    return SearchResponse(query="dragons", hits=[hit],
                          sources_searched=["memory", "books"],
                          sources_skipped={"code": "no code indexed yet"},
                          deduped=1)


class SearchParserTests(unittest.TestCase):
    def test_alias_registered(self):
        self.assertIn("s", CLI_ALIASES["search"])

    def test_canonical(self):
        self.assertEqual(_canonical_command("search"), "search")
        self.assertEqual(_canonical_command("s"), "search")

    def test_flags(self):
        args = _args(["search", "fire", "breathing", "dragons",
                      "--source", "memory", "--source", "books",
                      "--type", "memory",
                      "--since", "2026-01-01", "--before", "2026-12-31",
                      "--limit", "5", "--dir", "/tmp/docs", "--json"])
        self.assertEqual(args.command, "search")
        self.assertEqual(args.task, ["fire", "breathing", "dragons"])
        self.assertEqual(args.source, ["memory", "books"])
        self.assertEqual(args.type, ["memory"])
        self.assertEqual(args.since, "2026-01-01")
        self.assertEqual(args.before, "2026-12-31")
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.dir, "/tmp/docs")
        self.assertTrue(args.json)

    def test_alias_parses(self):
        args = _args(["s", "dragons"])
        self.assertEqual(args.task, ["dragons"])


class SearchCommandTests(unittest.TestCase):
    def test_no_query_is_usage_error(self):
        rc, _ = _run(["search"], _ctx())
        self.assertEqual(rc, 2)

    def test_unknown_source_fails_fast(self):
        rc, _ = _run(["search", "dragons", "--source", "nope"], _ctx())
        self.assertEqual(rc, 2)

    def test_human_output(self):
        with patch("nomorals.search.federated_search", return_value=_resp()):
            rc, out = _run(["search", "dragons"], _ctx())
        self.assertEqual(rc, 0)
        self.assertIn("Dragon Notes", out)
        self.assertIn("[memory/memory 1.00]", out)
        self.assertIn("[code skipped: no code indexed yet]", out)
        self.assertIn("1 duplicate(s) merged", out)

    def test_json_output(self):
        with patch("nomorals.search.federated_search", return_value=_resp()):
            rc, out = _run(["search", "dragons", "--json"], _ctx())
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["hits"][0]["title"], "Dragon Notes")
        self.assertEqual(payload["sources_skipped"],
                         {"code": "no code indexed yet"})

    def test_zero_hits_names_searched_sources(self):
        empty = SearchResponse(query="q", hits=[],
                               sources_searched=["memory", "books"])
        with patch("nomorals.search.federated_search", return_value=empty):
            rc, out = _run(["search", "nothing-matches-this"], _ctx())
        self.assertEqual(rc, 0)
        self.assertIn("no matches (searched: memory, books)", out)


if __name__ == "__main__":
    unittest.main()
