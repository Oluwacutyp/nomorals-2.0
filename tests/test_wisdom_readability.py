"""WisdomKeeper readability: read full texts, sentence-complete ask,
timeline arg parsing, case-insensitive traditions, per-work receipts,
named seeds. Failing-first for the owner's screenshot complaints."""
from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.cmdline.commands.wisdom import _cmd_wisdom
from nomorals.cmdline.parser import _parser
from nomorals.wisdom import ManifestEntry, WisdomKeeper
from nomorals.wisdom.history import HistoryEngine
from nomorals.wisdom.reading import (
    normalize_tradition,
    parse_read_args,
    parse_timeline_args,
)


def _args(argv):
    return _parser().parse_args(argv)


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-read-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, extras={}, settings=settings), tmp


def _run(argv, ctx):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _cmd_wisdom(_args(argv), ctx)
    return rc, buf.getvalue()


def _entry(slug, title, tradition, **kw):
    d = {
        "slug": slug, "title": title, "tradition": tradition,
        "canon_status": "gnostic", "translator": "Tester",
        "source_url": f"https://example.com/{slug}",
    }
    d.update(kw)
    return ManifestEntry.from_dict(d)


# Long multi-sentence text: >3500 chars so pagination has >= 2 pages,
# split across two chapters so chapter reads are testable. Every
# sentence is numbered/unique so page-boundary overlap checks are exact.
CH1 = "# Sayings\n\n" + " ".join(
    f"Saying {i}: the kingdom of heaven is within you and all around you."
    for i in range(1, 121))
CH2 = "# Parables\n\n" + " ".join(
    f"Parable {i}: a sower went out to sow his seed upon the good earth "
    f"and it bore fruit a hundredfold."
    for i in range(1, 121))
LONG_TEXT = CH1 + "\n\n" + CH2

SENT_END = re.compile(r"[.!?\u2026][\"'\u201d)\]]?\s*$")


class ReadingHelpersTests(unittest.TestCase):
    def test_normalize_case_hyphen_space(self):
        self.assertEqual(normalize_tradition("Christian-gnostic"),
                         "christian-gnostic")
        self.assertEqual(normalize_tradition("CHRISTIAN GNOSTIC"),
                         "christian-gnostic")
        self.assertEqual(normalize_tradition("christian_gnostic"),
                         "christian-gnostic")

    def test_parse_timeline_count_arg(self):
        trad, n = parse_timeline_args(["christian-gnostic", "50"])
        self.assertEqual(trad, "christian-gnostic")
        self.assertEqual(n, 50)
        trad, n = parse_timeline_args(["Christian-gnostic", "2"])
        self.assertEqual(trad, "Christian-gnostic")
        self.assertEqual(n, 2)
        trad, n = parse_timeline_args(["50"])
        self.assertEqual(trad, "")
        self.assertEqual(n, 50)
        trad, n = parse_timeline_args([])
        self.assertEqual((trad, n), ("", None))

    def test_parse_read_args(self):
        self.assertEqual(parse_read_args([]), (None, 1))
        self.assertEqual(parse_read_args(["3"]), (None, 3))
        self.assertEqual(parse_read_args(["2", "3"]), (2, 3))
        self.assertEqual(parse_read_args(["chapter", "2"]), (2, 1))
        self.assertEqual(parse_read_args(["page", "3"]), (None, 3))
        self.assertEqual(parse_read_args(["Sayings"]), ("Sayings", 1))


class ReadWorkTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = _ctx()
        self.k = WisdomKeeper(self.ctx)
        self.k.corpus.register(_entry("gth", "Gospel of Thomas",
                                      "christian-gnostic"))
        self.k.corpus.ingest_text("gth", LONG_TEXT)

    def test_read_page1_and_page2_continue(self):
        p1 = self.k.corpus.read_work("gth", page=1)
        p2 = self.k.corpus.read_work("gth", page=2)
        self.assertEqual(p1["page"], 1)
        self.assertGreater(p1["total_pages"], 1)
        self.assertEqual(p2["page"], 2)
        # Page 2 must continue page 1 with no gap and no overlap:
        # joining all pages reproduces the full cleaned text.
        pages = [self.k.corpus.read_work("gth", page=i)["text"]
                 for i in range(1, p1["total_pages"] + 1)]
        joined = "".join(pages)
        self.assertIn("kingdom of heaven is within you", joined)
        self.assertIn("bore fruit a hundredfold", joined)
        # No page ends mid-sentence (except possibly the very last).
        for pg in pages[:-1]:
            self.assertRegex(pg.strip(), SENT_END,
                             f"page ends mid-sentence: …{pg[-80:]!r}")
        # Consecutive pages share no duplicated tail/head sentence.
        self.assertNotIn(pages[0][-60:].strip(), pages[1])

    def test_read_by_title_case_insensitive(self):
        got = self.k.corpus.read_work("GOSPEL OF THOMAS", page=1)
        self.assertEqual(got["slug"], "gth")

    def test_read_chapter(self):
        ch = self.k.corpus.read_work("gth", chapter=2, page=1)
        self.assertIn("sower went out", ch["text"])
        self.assertNotIn("kingdom of heaven", ch["text"])
        ch_by_name = self.k.corpus.read_work("gth", chapter="Parables")
        self.assertIn("sower went out", ch_by_name["text"])

    def test_read_page_out_of_range(self):
        from nomorals.wisdom import CorpusError
        with self.assertRaises(CorpusError):
            self.k.corpus.read_work("gth", page=9999)

    def test_read_uningested_gives_ingest_hint(self):
        self.k.corpus.register(_entry("ps", "Pistis Sophia",
                                      "christian-gnostic"))
        from nomorals.wisdom import CorpusError
        with self.assertRaisesRegex(CorpusError, "ingest"):
            self.k.corpus.read_work("ps")

    def test_read_unknown_work_lists_candidates(self):
        from nomorals.wisdom import CorpusError
        with self.assertRaises(CorpusError):
            self.k.corpus.read_work("no-such-work")


class AskSentenceTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = _ctx()
        self.k = WisdomKeeper(self.ctx)
        self.k.corpus.register(_entry("gth", "Gospel of Thomas",
                                      "christian-gnostic"))
        # OCR-style hyphenation break + a mid-text sentence to catch.
        text = ("# Sayings\n\n" +
                "The kingdom of heaven is within you. " * 5 +
                "The scribes hid the sacred liter- \n ature from the people. " +
                "Seek and you shall find the hidden light. " * 20)
        self.k.corpus.ingest_text("gth", text)

    def test_ask_passages_end_at_sentence_boundaries(self):
        ans = self.k.ask("kingdom heaven", top=3)
        self.assertTrue(ans.passages)
        for p in ans.passages:
            self.assertRegex(p.snippet.strip(), SENT_END,
                             f"snippet ends mid-sentence: {p.snippet[-80:]!r}")
            self.assertNotIn("-\n", p.snippet)
            # No dangling fragment: snippet must not start mid-sentence
            # with a lowercase continuation of a cut word.
            self.assertNotIn("…go", p.snippet)

    def test_ask_passage_carries_read_hint(self):
        ans = self.k.ask("kingdom heaven", top=1)
        p = ans.passages[0]
        self.assertEqual(p.slug, "gth")
        card = ans.render("chat")
        self.assertIn("/wisdom read gth", card)

    def test_ask_dehyphenates(self):
        ans = self.k.ask("literature scribes", top=3)
        joined = " ".join(p.snippet for p in ans.passages)
        self.assertIn("literature", joined)
        self.assertNotIn("liter-", joined)


class TimelineParsingTests(unittest.TestCase):
    def test_history_case_insensitive(self):
        eng = HistoryEngine(SimpleNamespace())
        lower = eng.timeline(tradition="christian-gnostic")
        upper = eng.timeline(tradition="Christian-gnostic")
        spaced = eng.timeline(tradition="Christian Gnostic")
        self.assertEqual(lower, upper)
        self.assertEqual(lower, spaced)
        self.assertTrue(lower)

    def test_history_limit(self):
        eng = HistoryEngine(SimpleNamespace())
        evs = eng.timeline(tradition="christian-gnostic", limit=2)
        self.assertEqual(len(evs), 2)

    def test_cli_timeline_count_and_case(self):
        ctx, _ = _ctx()
        rc, out = _run(["wisdom", "timeline", "Christian-gnostic", "2"], ctx)
        self.assertEqual(rc, 0)
        # Exactly 2 event lines (plus header).
        lines = [ln for ln in out.splitlines() if ln.strip().startswith(("1", "2", "3", "-", "4", "5", "6", "7", "8", "9", "0"))]
        self.assertEqual(len(lines), 2)
        # Full summaries, not one-line stubs.
        self.assertGreater(len(out), 200)

    def test_cli_timeline_count_only(self):
        ctx, _ = _ctx()
        rc, out = _run(["wisdom", "timeline", "3"], ctx)
        self.assertEqual(rc, 0)


class TraditionsAndSeedsTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = _ctx()
        self.k = WisdomKeeper(self.ctx)
        self.k.corpus.register(_entry("gth", "Gospel of Thomas",
                                      "christian-gnostic"))
        self.k.corpus.ingest_text("gth", LONG_TEXT)

    def test_bare_tradition_lists_works_cli(self):
        rc, out = _run(["wisdom", "christian-gnostic"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("Gospel of Thomas", out)
        self.assertIn("/wisdom read gth", out)

    def test_bare_tradition_case_insensitive_cli(self):
        rc, out = _run(["wisdom", "Christian-gnostic"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("Gospel of Thomas", out)

    def test_traditions_command_explains_categories(self):
        rc, out = _run(["wisdom", "traditions"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("Gospel of Thomas", out)
        # A "tradition" must be explained, not just named.
        self.assertRegex(out, r'(?i)tradition" is')
        self.assertIn("christian-gnostic", out.lower())

    def test_status_lists_every_work_with_receipts(self):
        rc, out = _run(["wisdom", "status", "--json"], self.ctx)
        self.assertEqual(rc, 0)
        doc = json.loads(out)
        works = doc["corpus"]["works"]
        self.assertEqual(len(works), 1)
        w = works[0]
        for key in ("slug", "title", "tradition", "source_url", "seed",
                    "ingested", "chars", "pages", "read_command"):
            self.assertIn(key, w, key)
        self.assertTrue(w["ingested"])
        self.assertGreater(w["chars"], 1000)
        self.assertGreater(w["pages"], 1)

    def test_every_status_work_is_openable(self):
        rc, out = _run(["wisdom", "status", "--json"], self.ctx)
        works = json.loads(out)["corpus"]["works"]
        for w in works:
            if w["ingested"]:
                got = self.k.corpus.read_work(w["slug"], page=1)
                self.assertTrue(got["text"])

    def test_seeds_command_names_sources(self):
        rc, out = _run(["wisdom", "seeds"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("packaged-seed", out)
        # The test-registered work is recorded under its seed (manual),
        # by slug — every seed lists the works that came from it.
        self.assertIn("gth", out)
        self.assertIn("manual", out)
        # How to add a seed must be documented.
        self.assertRegex(out, r"(?i)add")

    def test_read_cli_paginates(self):
        rc, out = _run(["wisdom", "read", "gth", "2"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("page 2", out)
        self.assertIn("/wisdom read gth 3", out)

    def test_read_cli_chapter(self):
        rc, out = _run(["wisdom", "read", "gth", "2", "1"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("sower went out", out)


if __name__ == "__main__":
    unittest.main()
