"""Wave K CLI: ``nm doc`` / ``nm browse`` / ``nm repo`` dispatch and behavior."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.cmdline.commands.browse import _cmd_browse
from nomorals.cmdline.commands.doc import _cmd_doc
from nomorals.cmdline.commands.repo import _cmd_repo
from nomorals.cmdline.dispatch import _canonical_command
from nomorals.cmdline.parser import _parser


def _args(argv):
    return _parser().parse_args(argv)


_CTX = SimpleNamespace(db=None, extras={})


def _run(fn, argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fn(_args(argv), _CTX)
    return rc, buf.getvalue()


class TestWaveKAliases(unittest.TestCase):
    def test_canonical(self):
        self.assertEqual(_canonical_command("doc"), "doc")
        self.assertEqual(_canonical_command("docs"), "doc")
        self.assertEqual(_canonical_command("browse"), "browse")
        self.assertEqual(_canonical_command("brw"), "browse")
        self.assertEqual(_canonical_command("repo"), "repo")
        self.assertEqual(_canonical_command("rp"), "repo")

    def test_parse(self):
        args = _args(["doc", "parse", "x.md"])
        self.assertEqual(args.command, "doc")
        self.assertEqual(args.task, ["parse", "x.md"])


class TestDocCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.md = os.path.join(self.tmp.name, "note.md")
        with open(self.md, "w", encoding="utf-8") as fh:
            fh.write("# Shopping\n\nBuy milk and eggs.\n\n## Budget\n\n$40 max.\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse(self):
        rc, out = _run(_cmd_doc, ["doc", "parse", self.md])
        self.assertEqual(rc, 0)
        self.assertIn("Shopping", out)
        self.assertIn("sections: 2", out)

    def test_parse_json(self):
        rc, out = _run(_cmd_doc, ["doc", "parse", self.md, "--json"])
        self.assertEqual(rc, 0)
        self.assertIn('"heading": "Shopping"', out)

    def test_parse_missing_file_fails_fast(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            rc, _ = _run(_cmd_doc, ["doc", "parse", "/no/such/file.md"])
        self.assertEqual(rc, 1)

    def test_convert_md_to_html(self):
        out_path = os.path.join(self.tmp.name, "note.html")
        rc, _ = _run(_cmd_doc, ["doc", "convert", self.md, "--to", "html",
                                "--out", out_path])
        self.assertEqual(rc, 0)
        html = open(out_path, encoding="utf-8").read()
        self.assertIn("<h1>", html)
        self.assertIn("Shopping", html)

    def test_convert_unknown_target(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            rc, _ = _run(_cmd_doc, ["doc", "convert", self.md, "--to", "exe"])
        self.assertEqual(rc, 2)

    def test_search(self):
        rc, out = _run(_cmd_doc, ["doc", "search", "milk", "--dir", self.tmp.name])
        self.assertEqual(rc, 0)
        self.assertIn("Shopping", out)

    def test_show(self):
        rc, out = _run(_cmd_doc, ["doc", "show", self.md])
        self.assertEqual(rc, 0)
        self.assertIn("Buy milk", out)

    def test_no_verb_usage(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            rc, _ = _run(_cmd_doc, ["doc"])
        self.assertEqual(rc, 2)


class TestRepoCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        subprocess.run(["git", "init", "-q", self.root], check=True)
        subprocess.run(["git", "-C", self.root, "config", "user.email", "t@t"],
                       check=True)
        subprocess.run(["git", "-C", self.root, "config", "user.name", "t"],
                       check=True)
        with open(os.path.join(self.root, "a.txt"), "w") as fh:
            fh.write("hello\n")
        subprocess.run(["git", "-C", self.root, "add", "."], check=True)
        subprocess.run(["git", "-C", self.root, "commit", "-qm", "init"],
                       check=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _repo(self, *words):
        return _run(_cmd_repo, ["repo", *words, "--root", self.root])

    def test_status_clean(self):
        rc, out = self._repo("status")
        self.assertEqual(rc, 0)
        self.assertIn("clean", out)

    def test_status_dirty(self):
        with open(os.path.join(self.root, "b.txt"), "w") as fh:
            fh.write("new\n")
        rc, out = self._repo("status")
        self.assertEqual(rc, 0)
        self.assertIn("b.txt", out)

    def test_branches_and_log(self):
        rc, out = self._repo("branches")
        self.assertEqual(rc, 0)
        rc, out = self._repo("log", "3")
        self.assertEqual(rc, 0)
        self.assertIn("init", out)

    def test_diff(self):
        with open(os.path.join(self.root, "a.txt"), "a") as fh:
            fh.write("more\n")
        rc, out = self._repo("diff")
        self.assertEqual(rc, 0)
        self.assertIn("+more", out)

    def test_patch_review_and_dry_run(self):
        diff = ("--- a/a.txt\n+++ b/a.txt\n@@ -1 +1,2 @@\n hello\n+world\n")
        dfile = os.path.join(self.root, "c.diff")
        with open(dfile, "w") as fh:
            fh.write(diff)
        rc, out = self._repo("patch", "review", dfile)
        self.assertEqual(rc, 0)
        self.assertIn("a.txt", out)
        rc, out = self._repo("patch", "apply", dfile)
        self.assertEqual(rc, 0)
        # dry-run: file untouched
        self.assertEqual(open(os.path.join(self.root, "a.txt")).read(), "hello\n")

    def test_not_a_repo_fails_fast(self):
        other = tempfile.mkdtemp()
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            rc, _ = _run(_cmd_repo, ["repo", "status", "--root", other])
        self.assertEqual(rc, 1)


class TestBrowseCLI(unittest.TestCase):
    def test_sessions_lists(self):
        rc, out = _run(_cmd_browse, ["browse", "sessions"])
        self.assertEqual(rc, 0)

    def test_no_verb_usage(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            rc, _ = _run(_cmd_browse, ["browse"])
        self.assertEqual(rc, 2)

    def test_unknown_verb(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            rc, _ = _run(_cmd_browse, ["browse", "frobnicate"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
