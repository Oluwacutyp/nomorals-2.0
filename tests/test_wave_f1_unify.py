"""Wave F1 Stream 1B: unification parity tests.

Proves the unified helpers behave identically to the copies they replace:

1. ``nomorals.agents.skill_evolution.apply_unified_diff`` is now a thin
   adapter over the canonical ``nomorals.core.diff`` engine.  The
   pre-unification naive implementation is embedded below verbatim (from
   git HEAD, renamed) as the oracle: on the agreed corpus both must
   agree exactly, and both must raise ``SkillEvolutionError`` on the
   failure corpus.
2. ``nomorals.agents.patch.apply_unified_diff`` was already an adapter;
   its dict contract is pinned here.
3. Inline short-id copies (``uuid.uuid4().hex[:N]``,
   ``secrets.token_hex(6)``) now route through
   ``nomorals.core.ids.new_short_id`` — shape and uniqueness pinned.
4. Inline char-truncation copies now route through
   ``nomorals.core.text.truncate`` — boundary semantics pinned.
"""

from __future__ import annotations

import re
import unittest

from nomorals.agents.skill_evolution import (
    SkillEvolutionError,
    apply_unified_diff as skill_apply,
)
from nomorals.agents.patch import apply_unified_diff as mission_apply
from nomorals.core.diff import apply_unified_diff as engine_apply
from nomorals.core.ids import new_short_id
from nomorals.core.text import truncate


# ── oracle: the pre-unification naive implementation, verbatim ─────────────
# (from git HEAD nomorals/agents/skill_evolution.py, renamed so the new
# adapter can be compared against it).

_OLD_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _old_parse_hunks(diff_text):
    hunks = []
    current = None
    for raw in (diff_text or "").splitlines():
        if raw.startswith("@@"):
            m = _OLD_HUNK_RE.match(raw)
            if not m:
                raise SkillEvolutionError(f"bad hunk header: {raw[:60]}")
            current = {"old_start": int(m.group(1)), "lines": []}
            hunks.append(current)
        elif current is not None and raw[:1] in (" ", "+", "-"):
            current["lines"].append((raw[0], raw[1:]))
    return hunks


def _old_apply_unified_diff(original, diff_text):
    orig_lines = original.splitlines()
    out = []
    pos = 0
    hunks = _old_parse_hunks(diff_text)
    if not hunks:
        raise SkillEvolutionError("no hunks found in diff")
    for hunk in hunks:
        old_start = hunk["old_start"] - 1
        if old_start < pos:
            raise SkillEvolutionError("overlapping hunks in diff")
        out.extend(orig_lines[pos:old_start])
        cursor = old_start
        for kind, text in hunk["lines"]:
            if kind == " ":
                if cursor >= len(orig_lines) or orig_lines[cursor] != text:
                    raise SkillEvolutionError(
                        f"context mismatch at original line {cursor + 1}")
                out.append(orig_lines[cursor])
                cursor += 1
            elif kind == "-":
                if cursor >= len(orig_lines) or orig_lines[cursor] != text:
                    raise SkillEvolutionError(
                        f"removal mismatch at original line {cursor + 1}")
                cursor += 1
            elif kind == "+":
                out.append(text)
            else:
                raise SkillEvolutionError(f"bad diff line kind {kind!r}")
        pos = cursor
    out.extend(orig_lines[pos:])
    trailing = "\n" if original.endswith("\n") else ""
    return "\n".join(out) + trailing


# ── corpus both engines must agree on ──────────────────────────────────────

def _hdr(name="skill"):
    return f"--- a/{name}\n+++ b/{name}\n"


_SINGLE = _hdr() + (
    "@@ -1,3 +1,4 @@\n"
    " line one\n"
    "-old line\n"
    "+new line\n"
    "+added line\n"
    " line three\n"
)

_MULTI = _hdr() + (
    "@@ -1,2 +1,2 @@\n"
    " a\n"
    "-b\n"
    "+B\n"
    "@@ -4,2 +4,2 @@\n"
    " d\n"
    "-e\n"
    "+E\n"
)

_INSERT = _hdr() + (
    "@@ -2,1 +2,3 @@\n"
    " two\n"
    "+two-point-five\n"
    "+two-point-six\n"
)

_DELETE = _hdr() + (
    "@@ -1,4 +1,2 @@\n"
    " keep\n"
    "-drop one\n"
    "-drop two\n"
    " tail\n"
)

_BARE = (
    "@@ -1,3 +1,4 @@\n"
    " line one\n"
    "-old line\n"
    "+new line\n"
    "+added line\n"
    " line three\n"
)

_PARITY_CASES = [
    ("line one\nold line\nline three\n", _SINGLE),
    ("a\nb\nc\nd\ne\n", _MULTI),
    ("one\ntwo\nthree\n", _INSERT),
    ("keep\ndrop one\ndrop two\ntail\n", _DELETE),
    # original without trailing newline; hunk does not touch the last line
    ("one\ntwo\nthree", _hdr() + "@@ -1,2 +1,3 @@\n one\n-two\n+TWO\n+EXTRA\n"),
    # bare @@ hunks, no file headers (old parser accepted these)
    ("line one\nold line\nline three\n", _BARE),
    # hunk headers are validated even though the path is ignored
    ("line one\nold line\nline three\n",
     _hdr("other/path.py") + _SINGLE.split("\n", 2)[2]),
]


class SkillAdapterParityTestCase(unittest.TestCase):
    def test_parity_with_old_engine(self):
        for original, diff in _PARITY_CASES:
            with self.subTest(diff=diff[:40]):
                self.assertEqual(skill_apply(original, diff),
                                 _old_apply_unified_diff(original, diff))

    def test_no_hunks_message_preserved(self):
        with self.assertRaises(SkillEvolutionError) as ctx:
            skill_apply("x\n", "no diff here")
        self.assertEqual(str(ctx.exception), "no hunks found in diff")
        with self.assertRaises(SkillEvolutionError) as ctx:
            skill_apply("x\n", "--- a/f\n+++ b/f\n")
        self.assertEqual(str(ctx.exception), "no hunks found in diff")

    def test_context_mismatch_shape(self):
        with self.assertRaises(SkillEvolutionError):
            skill_apply("totally different\ntext\n", _SINGLE)
        with self.assertRaises(SkillEvolutionError):
            _old_apply_unified_diff("totally different\ntext\n", _SINGLE)

    def test_malformed_hunk_header_shape(self):
        bad = _hdr() + "@@ -1 @@\n line\n"
        with self.assertRaises(SkillEvolutionError):
            skill_apply("line\n", bad)

    def test_never_leaks_diff_apply_error(self):
        from nomorals.core.diff import DiffApplyError
        for original, diff in [("a\n", _hdr() + "@@ -9,1 +9,1 @@\n-a\n+b\n"),
                              ("x\n", "garbage @@ -1 @@\n")]:
            with self.subTest(diff=diff[:30]):
                leaked = False
                try:
                    skill_apply(original, diff)
                except DiffApplyError:
                    leaked = True
                except SkillEvolutionError as exc:
                    self.assertTrue(str(exc))
                self.assertFalse(leaked, "DiffApplyError leaked")

    def test_no_newline_marker_now_authoritative(self):
        # Deliberate, documented improvement: the old engine ignored the
        # "\ No newline" marker; the canonical engine honors it (GNU patch
        # semantics).
        diff = (_hdr() + "@@ -1,2 +1,2 @@\n a\n-b\n+B\n"
                "\\ No newline at end of file\n")
        out = skill_apply("a\nb\n", diff)
        self.assertEqual(out, "a\nB")
        self.assertEqual(_old_apply_unified_diff("a\nb\n", diff), "a\nB\n")

    def test_offset_tolerance_now_applies(self):
        # Deliberate, documented improvement: a hunk whose stated line
        # numbers drifted (lines inserted above since the diff was made)
        # now applies at the forward match (patch-style offset tolerance)
        # instead of raising "context mismatch".
        diff = _hdr() + "@@ -1,2 +1,2 @@\n a\n-b\n+B\n"
        self.assertEqual(skill_apply("x\na\nb\n", diff), "x\na\nB\n")
        with self.assertRaises(SkillEvolutionError):
            _old_apply_unified_diff("x\na\nb\n", diff)


class MissionAdapterContractTestCase(unittest.TestCase):
    """agents.patch.apply_unified_diff keeps its dict contract."""

    def test_dict_contract(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "f.txt").write_text("hello\n", encoding="utf-8")
            patch = ("--- a/f.txt\n+++ b/f.txt\n"
                     "@@ -1 +1 @@\n-hello\n+bye\n")
            result = mission_apply(patch, root)
            self.assertEqual(result["applied_files"], ["f.txt"])
            self.assertEqual(result["failed_hunks"], [])
            self.assertEqual((root / "f.txt").read_text(), "bye\n")

    def test_failure_shape(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            result = mission_apply(
                "--- a/nope.txt\n+++ b/nope.txt\n@@ -1 +1 @@\n-a\n+b\n",
                Path(tmp))
            self.assertEqual(result["applied_files"], [])
            self.assertEqual(len(result["failed_hunks"]), 1)
            fh = result["failed_hunks"][0]
            self.assertEqual(set(fh), {"file", "hunk", "reason"})
            self.assertEqual(fh["file"], "nope.txt")

    def test_engine_still_canonical(self):
        out = engine_apply("--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n",
                           {"f": "a\n"})
        self.assertEqual(out, {"f": "b\n"})


class ShortIdUnifyTestCase(unittest.TestCase):
    def test_new_short_id_shapes(self):
        for length in (8, 12):
            got = new_short_id(length=length)
            self.assertEqual(len(got), length)
            self.assertRegex(got, r"^[0-9a-f]+$")
        self.assertEqual(len({new_short_id(length=12) for _ in range(200)}),
                         200)

    def test_rewired_call_sites(self):
        from nomorals.agents.arena.core import _new_id as arena_new_id
        from nomorals.agents.trial.flow import new_id as trial_new_id
        for fn in (arena_new_id, trial_new_id):
            got = fn()
            self.assertEqual(len(got), 12)
            self.assertRegex(got, r"^[0-9a-f]{12}$")


class TruncateUnifyTestCase(unittest.TestCase):
    def test_boundary_semantics(self):
        self.assertEqual(truncate("abc", 3), "abc")
        self.assertEqual(truncate("abcd", 3), "abc…")
        self.assertEqual(truncate("", 0), "")
        self.assertEqual(truncate("abc", 10), "abc")
        self.assertEqual(truncate("abcdef", 4, suffix="..."), "abcd...")

    def test_alias_parity(self):
        from nomorals.agents.roles.orchestrator_helpers import (
            truncate as legacy_truncate)
        for text, limit in [("x" * 5000, 4000), ("short", 4000),
                            ("exact" * 800, 4000), ("", 4000)]:
            self.assertEqual(legacy_truncate(text, limit),
                             truncate(text, limit))
        self.assertEqual(legacy_truncate("y" * 10), truncate("y" * 10, 4000))


if __name__ == "__main__":
    unittest.main()
