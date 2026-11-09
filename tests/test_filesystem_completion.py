"""WS1 Pass 3 — filesystem family completion.

The fs_* family had no move/rename, no content search, and no way to
read part of a file too big for fs_read's cap.  These tests pin the
new verbs (fs_move, fs_search, fs_read_range, fs_mkdir, fs_hash)
through the production registry — the surface the brain actually
calls — including the refusal rules: no silent clobber, no traversal
on any verb, caps stated when hit.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.core.policy import CapabilitySet

ALL = CapabilitySet.all()


class FilesystemCompletionTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-fs3-")
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.tools = self.context.tools.register_builtins()
        self.root = Path(self.context.settings.workspace_dir)

    def tearDown(self):
        self.context.__exit__(None, None, None)
        shutil.rmtree(self.home, ignore_errors=True)

    # ------------------------------------------------------------ helpers
    def call(self, name, **kwargs):
        return self.tools.call(name, actor="t", capabilities=ALL, **kwargs)

    def write(self, rel: str, content: str) -> Path:
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    # ------------------------------------------------------------ wiring
    def test_new_verbs_are_registered(self):
        names = set(self.tools.names())
        for expected in ("fs_move", "fs_search", "fs_read_range",
                         "fs_mkdir", "fs_hash"):
            self.assertIn(expected, names)

    # ------------------------------------------------------------ fs_move
    def test_move_renames_a_file(self):
        self.write("old.txt", "payload")
        outcome = self.call("fs_move", source="old.txt", destination="new.txt")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertTrue(outcome.value["moved"])
        self.assertFalse(outcome.value["overwrote"])
        self.assertFalse((self.root / "old.txt").exists())
        self.assertEqual((self.root / "new.txt").read_text(encoding="utf-8"),
                         "payload")

    def test_move_moves_a_directory_tree(self):
        self.write("tree/sub/leaf.txt", "leaf")
        self.write("tree/top.txt", "top")
        outcome = self.call("fs_move", source="tree", destination="moved")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertTrue(outcome.value["is_dir"])
        self.assertFalse((self.root / "tree").exists())
        self.assertEqual(
            (self.root / "moved/sub/leaf.txt").read_text(encoding="utf-8"),
            "leaf")

    def test_move_refuses_to_clobber_without_overwrite(self):
        self.write("a.txt", "a")
        self.write("b.txt", "b")
        outcome = self.call("fs_move", source="a.txt", destination="b.txt")
        self.assertFalse(outcome.ok)
        # Neither file was touched by the refusal.
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "a")
        self.assertEqual((self.root / "b.txt").read_text(encoding="utf-8"), "b")

    def test_move_overwrite_replaces_and_says_so(self):
        self.write("a.txt", "a")
        self.write("b.txt", "b")
        outcome = self.call("fs_move", source="a.txt", destination="b.txt",
                            overwrite=True)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertTrue(outcome.value["overwrote"])
        self.assertEqual((self.root / "b.txt").read_text(encoding="utf-8"), "a")
        self.assertFalse((self.root / "a.txt").exists())

    def test_move_into_an_existing_directory(self):
        self.write("loose.txt", "x")
        (self.root / "folder").mkdir()
        outcome = self.call("fs_move", source="loose.txt", destination="folder")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertTrue((self.root / "folder/loose.txt").exists())

    def test_move_directory_into_itself_is_refused(self):
        self.write("tree/leaf.txt", "leaf")
        outcome = self.call("fs_move", source="tree",
                            destination="tree/inner")
        self.assertFalse(outcome.ok)
        self.assertTrue((self.root / "tree/leaf.txt").exists())

    def test_move_same_path_is_a_named_noop(self):
        self.write("a.txt", "a")
        outcome = self.call("fs_move", source="a.txt", destination="a.txt")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertFalse(outcome.value["moved"])
        self.assertIn("same path", outcome.value["note"])

    # ------------------------------------------------------------ fs_search
    def _corpus(self):
        self.write("one.txt", "alpha\nbeta needle\ngamma\n")
        self.write("sub/two.txt", "needle here\nand NEEDLE there\n")
        self.write("three.md", "nothing relevant\n")

    def test_search_finds_hits_with_line_numbers(self):
        self._corpus()
        outcome = self.call("fs_search", pattern="needle")
        self.assertTrue(outcome.ok, outcome.error)
        value = outcome.value
        self.assertEqual(value["count"], 2)
        self.assertFalse(value["truncated"])
        hits = {(m["file"], m["line"]) for m in value["matches"]}
        self.assertEqual(hits, {("one.txt", 2), ("sub/two.txt", 1)})
        self.assertEqual(value["files_searched"], 3)

    def test_search_case_insensitive_and_regex(self):
        self._corpus()
        outcome = self.call("fs_search", pattern="needle",
                            case_sensitive=False)
        self.assertEqual(outcome.value["count"], 3)
        outcome = self.call("fs_search", pattern=r"NE+DLE", regex=True)
        self.assertEqual(outcome.value["count"], 1)

    def test_search_skips_binary_files_and_counts_them(self):
        self.write("text.txt", "needle\n")
        blob = self.root / "blob.bin"
        blob.write_bytes(b"\x00\x01needle\x00\x02")
        outcome = self.call("fs_search", pattern="needle")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.value["count"], 1)
        self.assertEqual(outcome.value["files_skipped_binary"], 1)

    def test_search_cap_is_stated_when_hit(self):
        for i in range(5):
            self.write(f"f{i}.txt", "needle\n")
        outcome = self.call("fs_search", pattern="needle", max_matches=2)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.value["count"], 2)
        self.assertTrue(outcome.value["truncated"])
        self.assertEqual(outcome.value["cap"], 2)

    def test_search_invalid_regex_is_refused(self):
        outcome = self.call("fs_search", pattern="(", regex=True)
        self.assertFalse(outcome.ok)

    # ------------------------------------------------------------ fs_read_range
    def test_read_range_byte_windows(self):
        self.write("digits.txt", "0123456789" * 10)
        head = self.call("fs_read_range", path="digits.txt",
                         offset=0, length=10)
        self.assertTrue(head.ok, head.error)
        self.assertEqual(head.value["content"], "0123456789")
        self.assertEqual(head.value["total_bytes"], 100)
        self.assertFalse(head.value["eof"])
        self.assertEqual(head.value["next_offset"], 10)
        tail = self.call("fs_read_range", path="digits.txt",
                         offset=90, length=64)
        self.assertEqual(tail.value["content"], "0123456789")
        self.assertTrue(tail.value["eof"])
        self.assertIsNone(tail.value["next_offset"])

    def test_read_range_pages_a_whole_file(self):
        payload = "".join(str(i % 10) for i in range(250))
        self.write("big.txt", payload)
        seen = ""
        offset = 0
        while offset is not None:
            outcome = self.call("fs_read_range", path="big.txt",
                                offset=offset, length=64)
            self.assertTrue(outcome.ok, outcome.error)
            seen += outcome.value["content"]
            offset = outcome.value["next_offset"]
        self.assertEqual(seen, payload)

    def test_read_range_beyond_end_is_honest(self):
        self.write("small.txt", "abc")
        outcome = self.call("fs_read_range", path="small.txt", offset=50)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.value["content"], "")
        self.assertTrue(outcome.value["eof"])
        self.assertEqual(outcome.value["total_bytes"], 3)
        self.assertIn("beyond the end", outcome.value["note"])

    def test_read_range_line_windows(self):
        self.write("lines.txt", "\n".join(f"line {i}" for i in range(1, 11)))
        outcome = self.call("fs_read_range", path="lines.txt",
                            start_line=3, line_count=2)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.value["content"], "line 3\nline 4")
        self.assertEqual(outcome.value["total_lines"], 10)
        self.assertFalse(outcome.value["eof"])
        beyond = self.call("fs_read_range", path="lines.txt", start_line=99)
        self.assertEqual(beyond.value["content"], "")
        self.assertIn("beyond the end", beyond.value["note"])

    def test_read_range_reaches_what_fs_read_refuses(self):
        self.write("capped.txt", "x" * 500)
        refused = self.call("fs_read", path="capped.txt", max_bytes=100)
        self.assertFalse(refused.ok)
        outcome = self.call("fs_read_range", path="capped.txt",
                            offset=0, length=100)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.value["returned_bytes"], 100)

    # ------------------------------------------------------------ fs_mkdir / fs_hash
    def test_mkdir_creates_and_reports_existing(self):
        outcome = self.call("fs_mkdir", path="a/b/c")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertTrue(outcome.value["created"])
        self.assertTrue((self.root / "a/b/c").is_dir())
        again = self.call("fs_mkdir", path="a/b/c")
        self.assertTrue(again.ok, again.error)
        self.assertFalse(again.value["created"])

    def test_mkdir_over_a_file_is_refused(self):
        self.write("file.txt", "x")
        outcome = self.call("fs_mkdir", path="file.txt")
        self.assertFalse(outcome.ok)

    def test_hash_matches_hashlib(self):
        self.write("h.txt", "payload")
        outcome = self.call("fs_hash", path="h.txt")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.value["hexdigest"],
                         hashlib.sha256(b"payload").hexdigest())
        self.assertEqual(outcome.value["bytes"], 7)

    def test_hash_rejects_unknown_algorithm(self):
        self.write("h.txt", "payload")
        outcome = self.call("fs_hash", path="h.txt", algorithm="crc32")
        self.assertFalse(outcome.ok)

    # ------------------------------------------------------------ traversal
    def test_traversal_is_refused_on_every_new_verb(self):
        self.write("inside.txt", "needle")
        escapes = [
            ("fs_move", dict(source="../escape.txt", destination="x.txt")),
            ("fs_move", dict(source="inside.txt", destination="../x.txt")),
            ("fs_search", dict(pattern="needle", path="../")),
            ("fs_read_range", dict(path="../secret.txt")),
            ("fs_mkdir", dict(path="../evil")),
            ("fs_hash", dict(path="../../etc/passwd")),
        ]
        for name, kwargs in escapes:
            with self.subTest(tool=name, kwargs=kwargs):
                outcome = self.call(name, **kwargs)
                self.assertFalse(outcome.ok)


if __name__ == "__main__":
    unittest.main()
