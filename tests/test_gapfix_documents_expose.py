"""Gapfix R3-72: the ``nm doc`` CLI must route through the documents
expose facade (``nomorals/documents/expose.py``) instead of calling
the raw engine exports directly.

The facade is the single public entry point for read / convert /
summarize / compare (plus the format + capability cards). Before the
wire the CLI re-implemented the convert dispatch table itself and
called ``parse_path`` / ``compare_documents`` / ``summarize`` directly
in five places, so the facade had zero production callers.

Spy tests (fail before the wire, pass after): patching the facade
functions with wraps and running each CLI verb asserts the facade is
actually called. Control tests (pass before and after): CLI output is
byte-identical to the direct-engine output, and error shapes are
unchanged.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from nomorals.cmdline.commands.doc import _cmd_doc
from nomorals.cmdline.parser import _parser
from nomorals.documents import expose

_TMP_ROOT = os.environ.get("NM_TEST_TMPDIR")


def _args(argv):
    return _parser().parse_args(argv)


_CTX = SimpleNamespace(db=None, extras={})


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _cmd_doc(_args(argv), _CTX)
    return rc, buf.getvalue()


class TestDocCliRoutesThroughExpose(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=_TMP_ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.a = os.path.join(self.tmp.name, "a.md")
        self.b = os.path.join(self.tmp.name, "b.md")
        with open(self.a, "w", encoding="utf-8") as fh:
            fh.write("# Doc\n\nHello world. This is a test document.\n")
        with open(self.b, "w", encoding="utf-8") as fh:
            fh.write("# Doc\n\nHello brave world. This is a test document.\n")

    # ── spy tests: the CLI must reach the facade ──────────────────────

    def test_parse_calls_read_document(self) -> None:
        with mock.patch.object(expose, "read_document",
                               wraps=expose.read_document) as spy:
            rc, out = _run(["doc", "parse", self.a])
        self.assertEqual(rc, 0)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.kwargs.get("path"), self.a)
        self.assertIn("Hello world", out)

    def test_convert_calls_read_and_convert_document(self) -> None:
        with mock.patch.object(expose, "read_document",
                               wraps=expose.read_document) as read_spy, \
                mock.patch.object(expose, "convert_document",
                                  wraps=expose.convert_document) as conv_spy:
            rc, out = _run(["doc", "convert", self.a, "--to", "md"])
        self.assertEqual(rc, 0)
        self.assertEqual(read_spy.call_count, 1)
        self.assertEqual(conv_spy.call_count, 1)
        # The CLI's "md" maps to the facade's "markdown" target.
        self.assertEqual(conv_spy.call_args.args[1], "markdown")
        self.assertIn("Hello world", out)

    def test_show_calls_read_document(self) -> None:
        with mock.patch.object(expose, "read_document",
                               wraps=expose.read_document) as spy:
            rc, out = _run(["doc", "show", self.a])
        self.assertEqual(rc, 0)
        self.assertEqual(spy.call_count, 1)
        self.assertIn("Hello world", out)

    def test_diff_calls_read_and_compare_document_pair(self) -> None:
        with mock.patch.object(expose, "read_document",
                               wraps=expose.read_document) as read_spy, \
                mock.patch.object(expose, "compare_document_pair",
                                  wraps=expose.compare_document_pair) as cmp_spy:
            rc, out = _run(["doc", "diff", self.a, self.b])
        self.assertEqual(rc, 0)
        self.assertEqual(read_spy.call_count, 2)
        self.assertEqual(cmp_spy.call_count, 1)
        self.assertIn("changed", out)

    def test_summarize_calls_read_and_summarize_document(self) -> None:
        with mock.patch.object(expose, "read_document",
                               wraps=expose.read_document) as read_spy, \
                mock.patch.object(expose, "summarize_document",
                                  wraps=expose.summarize_document) as sum_spy:
            rc, out = _run(["doc", "summarize", self.a, "--sentences", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(read_spy.call_count, 1)
        self.assertEqual(sum_spy.call_count, 1)
        self.assertEqual(sum_spy.call_args.kwargs.get("sentences"), 1)
        # With --sentences 1 the tf ranker keeps the second sentence.
        self.assertIn("This is a test document.", out)

    # ── control tests: output identical to the direct engine ──────────

    def test_convert_md_matches_direct_engine(self) -> None:
        from nomorals.documents import parse_path, to_markdown

        rc, out = _run(["doc", "convert", self.a, "--to", "md"])
        self.assertEqual(rc, 0)
        # The CLI prints the converted text with print(), so the output
        # is the converter result plus one trailing newline.
        self.assertEqual(out, to_markdown(parse_path(self.a)) + "\n")

    def test_convert_txt_matches_direct_engine(self) -> None:
        from nomorals.documents import parse_path, to_text

        rc, out = _run(["doc", "convert", self.a, "--to", "txt"])
        self.assertEqual(rc, 0)
        # The CLI prints the converted text with print(), so the output
        # is the converter result plus one trailing newline.
        self.assertEqual(out, to_text(parse_path(self.a)) + "\n")

    def test_parse_json_shape(self) -> None:
        rc, out = _run(["doc", "parse", self.a, "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["format"], "markdown")
        self.assertEqual(payload["title"], "Doc")

    def test_diff_identical_docs_says_no_changes(self) -> None:
        rc, out = _run(["doc", "diff", self.a, self.a])
        self.assertEqual(rc, 0)
        self.assertIn("no changes", out)

    def test_convert_binary_target_writes_bytes_via_out(self) -> None:
        out_path = os.path.join(self.tmp.name, "a.pdf")
        rc, out = _run(["doc", "convert", self.a, "--to", "pdf",
                        "--out", out_path])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(out_path))
        with open(out_path, "rb") as fh:
            head = fh.read(5)
        self.assertEqual(head, b"%PDF-")
        self.assertIn("wrote", out)

    def test_unknown_target_still_rc2(self) -> None:
        rc, out = _run(["doc", "convert", self.a, "--to", "bogus"])
        self.assertEqual(rc, 2)

    def test_missing_file_still_rc1(self) -> None:
        import io as _io
        from contextlib import redirect_stderr

        buf = _io.StringIO()
        with redirect_stderr(buf):
            rc, _ = _run(["doc", "parse",
                          os.path.join(self.tmp.name, "nope.md")])
        self.assertEqual(rc, 1)
        self.assertIn("error:", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
