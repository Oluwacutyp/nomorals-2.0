"""``nm wisdom`` CLI: dispatch, verbs, aliases."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.cmdline.commands.wisdom import _cmd_wisdom
from nomorals.cmdline.dispatch import _canonical_command
from nomorals.cmdline.parser import CLI_ALIASES, _parser


def _args(argv):
    return _parser().parse_args(argv)


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-cli-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, extras={}, settings=settings), tmp


def _run(argv, ctx):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _cmd_wisdom(_args(argv), ctx)
    return rc, buf.getvalue()


LOREM = ("# The Kingdom\n\n" +
         "The kingdom of heaven is within you and all around you. " * 40)


class WisdomAliasTests(unittest.TestCase):
    def test_alias_registered(self):
        self.assertIn("wis", CLI_ALIASES["wisdom"])

    def test_canonical(self):
        self.assertEqual(_canonical_command("wisdom"), "wisdom")
        self.assertEqual(_canonical_command("wis"), "wisdom")

    def test_parse(self):
        args = _args(["wisdom", "ask", "kingdom"])
        self.assertEqual(args.command, "wisdom")
        self.assertEqual(args.task, ["ask", "kingdom"])

    def test_no_verb_usage(self):
        ctx, _ = _ctx()
        rc, _ = _run(["wisdom"], ctx)
        self.assertEqual(rc, 2)

    def test_unknown_verb(self):
        ctx, _ = _ctx()
        rc, _ = _run(["wisdom", "frobnicate"], ctx)
        self.assertEqual(rc, 2)


class WisdomStatusTests(unittest.TestCase):
    def test_status_empty(self):
        ctx, _ = _ctx()
        rc, out = _run(["wisdom", "status"], ctx)
        self.assertEqual(rc, 0)
        self.assertIn("0/0", out)

    def test_status_json(self):
        ctx, _ = _ctx()
        rc, out = _run(["wisdom", "status", "--json"], ctx)
        self.assertEqual(rc, 0)
        doc = json.loads(out)
        self.assertIn("corpus", doc)


class WisdomAskTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = _ctx()
        from nomorals.wisdom import ManifestEntry, WisdomKeeper
        k = WisdomKeeper(self.ctx)
        k.corpus.register(ManifestEntry.from_dict({
            "slug": "gth", "title": "Gospel of Thomas",
            "tradition": "christian-gnostic", "canon_status": "gnostic",
            "source_url": "https://example.com/gth"}))
        k.corpus.ingest_text("gth", LOREM)

    def test_ask(self):
        rc, out = _run(["wisdom", "ask", "kingdom"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("Gospel of Thomas", out)

    def test_ask_json(self):
        rc, out = _run(["wisdom", "ask", "kingdom", "--json"], self.ctx)
        self.assertEqual(rc, 0)
        doc = json.loads(out)
        self.assertGreater(len(doc["passages"]), 0)
        self.assertIn("url", doc["passages"][0])

    def test_ask_needs_query(self):
        rc, _ = _run(["wisdom", "ask"], self.ctx)
        self.assertEqual(rc, 2)


class WisdomHistoryTests(unittest.TestCase):
    def test_timeline(self):
        ctx, _ = _ctx()
        rc, out = _run(["wisdom", "timeline", "--tradition", "buddhism",
                        "--start", "-600", "--end", "-300"], ctx)
        self.assertEqual(rc, 0)
        self.assertIn("buddhism", out.lower())

    def test_timeline_json(self):
        ctx, _ = _ctx()
        rc, out = _run(["wisdom", "timeline", "--json"], ctx)
        self.assertEqual(rc, 0)
        doc = json.loads(out)
        self.assertGreater(len(doc), 40)

    def test_timeline_bad_tradition_fails_fast(self):
        ctx, _ = _ctx()
        rc, _ = _run(["wisdom", "timeline", "--tradition", "nope"], ctx)
        self.assertEqual(rc, 1)

    def test_compare(self):
        ctx, _ = _ctx()
        rc, out = _run(["wisdom", "compare", "kingdom"], ctx)
        self.assertEqual(rc, 0)
        self.assertIn("topic: kingdom", out)

    def test_compare_needs_topic(self):
        ctx, _ = _ctx()
        rc, _ = _run(["wisdom", "compare"], ctx)
        self.assertEqual(rc, 2)


class WisdomPracticeTests(unittest.TestCase):
    def test_practice_list(self):
        ctx, _ = _ctx()
        rc, out = _run(["wisdom", "practice", "list"], ctx)
        self.assertEqual(rc, 0)
        self.assertIn("four-seven-eight", out)

    def test_practice_list_json(self):
        ctx, _ = _ctx()
        rc, out = _run(["wisdom", "practice", "list", "--json"], ctx)
        self.assertEqual(rc, 0)
        doc = json.loads(out)
        self.assertGreater(len(doc), 4)


class WisdomSeedTests(unittest.TestCase):
    def test_seed(self):
        ctx, _ = _ctx()
        rc, out = _run(["wisdom", "seed"], ctx)
        self.assertEqual(rc, 0)
        self.assertIn("seeded", out)

    def test_seed_idempotent(self):
        ctx, _ = _ctx()
        _run(["wisdom", "seed"], ctx)
        rc, out = _run(["wisdom", "seed"], ctx)
        self.assertEqual(rc, 0)
        self.assertIn("seeded 0", out)


if __name__ == "__main__":
    unittest.main()
