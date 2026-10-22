"""Pass #64: context packs for edit calls (W6).

The defect, from the 2026-10-20 coding-agent survey:

- W6: ``_draft_change`` pastes the WHOLE target file into every edit
  round — no imports context, no neighboring definitions, no covering
  tests — and the ranked-context machinery (repo_context.rank_context,
  gap G8: CLOSED for CodeExecutor, PARTIAL for the main loop) is never
  called by the drafting loop.  Prompt cost scaled with file size,
  and the model edited blind to the symbols the file depends on and
  to the tests that would judge the change.

These tests pin the new contract:

- a pack carries the target's relevant REGION (omitted definitions
  named in markers + metadata), the import surface (referenced
  neighbor definitions + signatures), and the covering tests the
  verify round would run (#62 selection: names + assertions);
- the pack is steered by the plan step (#60): different instructions
  select different regions;
- the budget is stated, deterministic, and every trim is recorded;
- fallbacks are honest: small / unparseable files and a broken import
  graph degrade to the whole-file prompt (or stem test selection)
  with the reason in the pack's notes — never a silently emptier
  prompt, never a crash;
- the loop records pack metadata on ``CodingResult.context_packs``
  and the edit outcome on a large file is identical to the
  whole-file outcome at a fraction of the prompt chars.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomorals.tools.context_pack import build_context_pack


# ── fixture repo ──────────────────────────────────────────────────────

HELPERS = '''\
"""Helper utilities."""


def format_money(amount):
    """Format an amount as a dollar string."""
    return f"${amount:,.2f}"


def clamp(value, lo, hi):
    """Clamp value into [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def unused_helper():
    return "unused"
'''


def _filler(name: str, lines: int = 30) -> str:
    marker = f"FILLER-{name}"
    body = [f"def {name}(payload):",
            f'    """{marker} — unrelated bulk."""',
            "    total = 0"]
    for i in range(lines):
        body.append(f"    total += (len(str(payload)) + {i})"
                    f"  # {marker} padding {i}")
    body.append("    return total")
    return "\n".join(body)


def _big_module() -> str:
    parts = [
        '"""Billing: invoice computation and rendering."""',
        "",
        "from pkg.helpers import clamp",
        "import pkg.helpers as helpers",
        "",
        "",
        "def compute_invoice_total(items, discount=0.0):",
        '    """Total an invoice, clamped, keeping cents."""',
        "    subtotal = sum(items)",
        "    total = subtotal * (1.0 - discount)",
        "    total = clamp(total, 0.0, 100000.0)",
        "    return int(total)",
        "",
        "",
        "def render_invoice(items):",
        '    """Render a full invoice: header, one money line per',
        '    item, and a totals footer."""',
        "    lines = []",
        '    lines.append("INVOICE")',
        '    lines.append("-------")',
        "    for idx, it in enumerate(items):",
        '        label = f"item {idx + 1}"',
        "        lines.append(f\"{label}: "
        "{helpers.format_money(it)}\")",
        '    lines.append("-------")',
        "    total = 0.0",
        "    for it in items:",
        "        total += it",
        '    lines.append(f"TOTAL: {helpers.format_money(total)}")',
        '    lines.append("")',
        '    lines.append("Thank you for your business.")',
        '    lines.append("Terms: net 30.")',
        '    lines.append("Late payments accrue 1.5% monthly.")',
        '    lines.append("Questions: billing@example.com")',
        '    lines.append("Remit to: Example Co, 1 Main St.")',
        '    lines.append("Tax ID: 00-0000000")',
        '    lines.append("All amounts in USD.")',
        "    return chr(10).join(lines)",
        "",
        "",
    ]
    names = ["filler_alpha", "filler_beta", "filler_gamma", "filler_delta",
             "filler_epsilon", "filler_zeta", "filler_eta", "filler_theta",
             "filler_iota", "filler_kappa", "filler_lambda", "filler_mu",
             "filler_nu", "filler_xi"]
    for name in names:
        parts.append(_filler(name))
        parts.append("")
        parts.append("")
    return "\n".join(parts)


REPORT = '''\
"""Reporting over billing."""

from . import big


def invoice_summary(items):
    return f"total={big.compute_invoice_total(items)}"
'''

TEST_BIG = '''\
from pkg.big import compute_invoice_total, render_invoice


def test_compute_invoice_total_keeps_cents():
    assert compute_invoice_total([10.5, 0.25]) == 10.75


def test_render_invoice_formats_money():
    out = render_invoice([1.5])
    assert "item 1: $1.50" in out
    assert "TOTAL: $1.50" in out
'''

TEST_REPORT = '''\
from pkg.report import invoice_summary


def test_invoice_summary_uses_total():
    assert invoice_summary([10.5, 0.25]) == "total=10.75"
'''


def _write_repo(root: Path) -> Path:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "helpers.py").write_text(HELPERS)
    (root / "pkg" / "big.py").write_text(_big_module())
    (root / "pkg" / "report.py").write_text(REPORT)
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "test_big.py").write_text(TEST_BIG)
    (root / "tests" / "test_report.py").write_text(TEST_REPORT)
    return root


class PackFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="pass64_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        _write_repo(self.root)
        from nomorals.tools import import_graph

        import_graph.clear_cache()
        self.current = (self.root / "pkg" / "big.py").read_text()
        self.step = {
            "id": "s1", "path": "pkg/big.py",
            "instructions": "Fix compute_invoice_total so it keeps "
                            "cents instead of truncating",
            "acceptance": "test_big passes",
        }

    def pack(self, **kw):
        kw.setdefault("task", "Fix the invoice total")
        kw.setdefault("step", self.step)
        return build_context_pack(self.root, "pkg/big.py", self.current,
                                  **kw)


class RegionTests(PackFixture):
    def test_pack_mode_with_focused_region(self):
        pack = self.pack()
        self.assertEqual(pack.mode, "pack")
        # the focused definition is shown, in full
        self.assertIn("def compute_invoice_total", pack.text)
        self.assertIn("return int(total)", pack.text)
        # an unrelated filler is NOT shown — but it is NAMED, in the
        # omission markers and in the metadata (never silent)
        self.assertNotIn("FILLER-filler_alpha padding 3", pack.text)
        self.assertIn("filler_alpha", pack.text)
        meta = pack.metadata()
        self.assertIn("filler_alpha", meta["omitted_defs"])
        self.assertTrue(meta["region_ranges"])
        self.assertLess(meta["region_ranges"][0][1],
                        len(self.current.split("\n")))

    def test_prompt_size_reduction(self):
        pack = self.pack()
        self.assertLess(len(pack.text), len(self.current) * 0.6,
                        f"pack {len(pack.text)} vs whole "
                        f"{len(self.current)}")
        self.assertLessEqual(len(pack.text), pack.budget_chars)


class ImportSurfaceTests(PackFixture):
    def test_referenced_neighbor_definitions_included(self):
        pack = self.pack()
        # clamp is from-imported AND referenced: its real body rides
        self.assertIn("def clamp(value, lo, hi):", pack.text)
        self.assertIn("if value < lo:", pack.text)
        # format_money is referenced through the module alias in
        # render_invoice: its definition is present too
        self.assertIn("def format_money(amount):", pack.text)
        kinds = {s["kind"] for s in pack.metadata()["sections"]}
        self.assertIn("neighbor-def", kinds)

    def test_import_signatures_present(self):
        pack = self.pack()
        self.assertIn("import surface", pack.text)


class CoveringTestsTests(PackFixture):
    def test_covering_tests_named_with_assertions(self):
        pack = self.pack()
        meta = pack.metadata()
        self.assertIn("tests/test_big.py", meta["covering_tests"])
        self.assertIn("tests/test_report.py", meta["covering_tests"],
                      "transitive coverage (via pkg.report) must ride")
        self.assertIn("test_compute_invoice_total_keeps_cents", pack.text)
        self.assertIn("assert compute_invoice_total([10.5, 0.25]) == 10.75",
                      pack.text)
        label = next(s["label"] for s in meta["sections"]
                     if s["label"].startswith("tests/test_big.py"))
        self.assertIn("graph-direct", label)


class SteeringTests(PackFixture):
    def test_step_instructions_steer_the_region(self):
        step_b = {"id": "s1", "path": "pkg/big.py",
                  "instructions": "Prettify render_invoice output "
                                  "lines for the receipt printer",
                  "acceptance": "render test passes"}
        pack_b = self.pack(step=step_b)
        self.assertIn("def render_invoice", pack_b.text)
        # Region choice follows the step: under a tight budget only
        # the steered definition survives — A's step keeps the
        # invoice-total body and drops render_invoice; B's step does
        # the reverse.  (At the default budget both small real defs
        # legitimately ride: the module is about invoices.  Steering
        # decides what survives scarcity, which is the point.)
        tight_a = self.pack(budget_chars=1450)
        tight_b = self.pack(step=step_b, budget_chars=1450)
        self.assertNotEqual(tight_a.text, tight_b.text)
        self.assertIn("return int(total)", tight_a.text)
        self.assertNotIn("def render_invoice", tight_a.text)
        self.assertIn("helpers.format_money(it)", tight_b.text)
        self.assertNotIn("return int(total)", tight_b.text)


class BudgetTests(PackFixture):
    def test_budget_deterministic_and_trims_visible(self):
        p1 = self.pack(budget_chars=3500)
        p2 = self.pack(budget_chars=3500)
        self.assertEqual(p1.text, p2.text, "same inputs, same pack")
        self.assertLessEqual(len(p1.text), 3500)
        meta = p1.metadata()
        self.assertTrue(meta["trimmed"],
                        "a tight budget must record what it cut")
        for item in meta["trimmed"]:
            self.assertIn("reason", item)
            self.assertIn("label", item)
        self.assertEqual(meta["used_chars"], len(p1.text))
        self.assertEqual(meta["budget_chars"], 3500)
        self.assertGreater(meta["whole_file_chars"], 3500)


class FallbackTests(PackFixture):
    def test_small_file_whole_file_mode(self):
        small = (self.root / "pkg" / "helpers.py").read_text()
        pack = build_context_pack(self.root, "pkg/helpers.py", small,
                                  task="tweak clamp")
        self.assertEqual(pack.mode, "whole-file")
        self.assertEqual(pack.text, small)
        self.assertTrue(any("small enough" in n
                            for n in pack.metadata()["notes"]))

    def test_unparseable_file_whole_file_mode(self):
        broken = "def broken(:\n" + ("    x = (\n" * 3000)
        pack = build_context_pack(self.root, "pkg/big.py", broken,
                                  task="fix it", step=self.step)
        self.assertEqual(pack.mode, "whole-file")
        self.assertEqual(pack.text, broken)
        self.assertTrue(any("does not parse" in n
                            for n in pack.metadata()["notes"]))

    def test_broken_import_graph_degrades_with_notes(self):
        from nomorals.tools import import_graph

        with mock.patch.object(import_graph, "build_import_graph",
                               side_effect=RuntimeError("graph exploded")):
            pack = self.pack()
        self.assertEqual(pack.mode, "pack",
                         "a broken graph degrades the pack, "
                         "it must not kill it")
        self.assertIn("def compute_invoice_total", pack.text)
        notes = " ".join(pack.metadata()["notes"])
        self.assertIn("import graph unavailable", notes)
        self.assertIn("stem-only", notes)

    def test_broken_selection_falls_back_to_stem(self):
        from nomorals.tools import pytest_runner

        with mock.patch.object(pytest_runner,
                               "select_changed_tests_detailed",
                               side_effect=RuntimeError("no selection")):
            pack = self.pack()
        self.assertEqual(pack.mode, "pack")
        notes = " ".join(pack.metadata()["notes"])
        self.assertIn("covering-test selection unavailable", notes)
        self.assertIn("tests/test_big.py",
                      pack.metadata()["covering_tests"])


# ── the loop ──────────────────────────────────────────────────────────


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


class RuleRouter:
    """Content-routed fake LLM (the pass #60 pattern): answers by
    what the call asks, so call ORDER never matters."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, str]] = []

    def chat(self, messages, params=None):
        texts = [getattr(m, "content", "") or "" for m in messages]
        system = texts[0] if texts else ""
        user = texts[-1] if texts else ""
        self.calls.append((system, user))
        text = self.handler(system, user)
        return SimpleNamespace(ok=bool(text), text=text,
                               error="" if text else "empty")

    def edits_for(self, filename):
        return [c for c in self.calls
                if "surgical code editor" in c[0]
                and f"File: {filename}" in c[1]]


class StubDB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE coding_log (id TEXT, task TEXT, filename TEXT,"
            " attempt INT, exit_code INT, timed_out INT, stdout TEXT,"
            " stderr TEXT, code TEXT, created_at REAL)")

    def execute(self, sql, params=()):
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


class StubContext:
    def __init__(self, router):
        self.router = router
        self.db = StubDB()
        self.settings = SimpleNamespace(reasoning_mode="off")


def _steps_json(steps):
    return "```json\n" + json.dumps({"steps": steps}) + "\n```"


def _edits(edits):
    return "```json\n" + json.dumps({"edits": edits}) + "\n```"


@unittest.skipUnless(shutil.which("git"), "git not available")
class LoopPackTests(unittest.TestCase):
    def _repo(self):
        root = Path(tempfile.mkdtemp(prefix="pass64loop_"))
        self.addCleanup(shutil.rmtree, root, True)
        _write_repo(root)
        _git("init", "-q", cwd=root)
        _git("config", "user.email", "t@example.com", cwd=root)
        _git("config", "user.name", "T", cwd=root)
        _git("add", ".", cwd=root)
        _git("commit", "-qm", "init", cwd=root)
        from nomorals.tools import import_graph

        import_graph.clear_cache()
        return root

    def test_result_defaults_are_additive(self):
        from nomorals.agents.coding import CodingResult

        bare = CodingResult(ok=True, iterations=0)
        self.assertEqual(bare.context_packs, [])
        self.assertEqual(bare.to_dict()["context_packs"], [])

    def test_large_file_edit_runs_on_pack_identical_outcome(self):
        from nomorals.agents.coding import CodingAgent

        root = self._repo()
        before = (root / "pkg" / "big.py").read_text()
        steps = [{
            "path": "pkg/big.py", "why": "billing", "new_file": False,
            "instructions": "Fix compute_invoice_total so it keeps "
                            "cents: return the total rounded to 2 "
                            "decimals instead of int() truncation",
            "acceptance": "tests/test_big.py passes",
        }]

        def handler(system, user):
            if "build planner" in system:
                return _steps_json(steps)
            if "surgical code editor" in system:
                return _edits([{"old_text": "    return int(total)",
                                "new_text": "    return round(total, 2)"}])
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(root),
                            pack_budget_chars=9000)
        result = agent.run("Fix the invoice total rounding",
                           max_iterations=5, timeout=60,
                           plan_mode="never")
        self.assertTrue(result.ok, result.error)
        # identical outcome to the whole-file run: exactly the one
        # scripted edit landed, nothing else moved
        after = (root / "pkg" / "big.py").read_text()
        self.assertEqual(after, before.replace(
            "    return int(total)", "    return round(total, 2)"))
        # the edit call was fed a pack, at a fraction of the file
        edit_calls = router.edits_for("pkg/big.py")
        self.assertEqual(len(edit_calls), 1)
        _system, user = edit_calls[0]
        self.assertIn("Context pack:", user)
        self.assertIn("def compute_invoice_total", user)
        self.assertNotIn("FILLER-filler_alpha padding 3", user)
        self.assertLess(len(user), len(before) * 0.5,
                        f"edit prompt {len(user)} vs whole file "
                        f"{len(before)}")
        # and the pack is on the result, budget + trims visible
        self.assertEqual(len(result.context_packs), 1)
        meta = result.context_packs[0]
        self.assertEqual(meta["mode"], "pack")
        self.assertEqual(meta["rel"], "pkg/big.py")
        self.assertLessEqual(meta["used_chars"], 9000)
        self.assertEqual(meta["whole_file_chars"], len(before))
        self.assertTrue(meta["trimmed"])
        self.assertIn("tests/test_big.py", meta["covering_tests"])
        self.assertEqual(result.to_dict()["context_packs"],
                         result.context_packs)
        by_path = {s["path"]: s for s in result.steps}
        self.assertEqual(by_path["pkg/big.py"]["status"], "done")

    def test_small_file_run_records_whole_file_fallback(self):
        from nomorals.agents.coding import CodingAgent

        root = Path(tempfile.mkdtemp(prefix="pass64small_"))
        self.addCleanup(shutil.rmtree, root, True)
        (root / "tests").mkdir()
        (root / "tests" / "__init__.py").write_text("")
        (root / "tests" / "test_main.py").write_text(
            "import main\n\n\ndef test_value():\n"
            "    assert main.VALUE == 2\n")
        (root / "main.py").write_text("VALUE = 1\n")
        _git("init", "-q", cwd=root)
        _git("config", "user.email", "t@example.com", cwd=root)
        _git("config", "user.name", "T", cwd=root)
        _git("add", ".", cwd=root)
        _git("commit", "-qm", "init", cwd=root)
        steps = [{"path": "main.py", "why": "the value",
                  "new_file": False,
                  "instructions": "set VALUE to 2",
                  "acceptance": "test_main passes"}]

        def handler(system, user):
            if "build planner" in system:
                return _steps_json(steps)
            if "surgical code editor" in system:
                return _edits([{"old_text": "VALUE = 1",
                                "new_text": "VALUE = 2"}])
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(root))
        result = agent.run("Fix the value", max_iterations=5,
                           timeout=60, plan_mode="never")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.context_packs), 1)
        meta = result.context_packs[0]
        self.assertEqual(meta["mode"], "whole-file")
        self.assertTrue(any("small enough" in n for n in meta["notes"]))
        # the prompt itself is the pre-#64 prompt, byte for byte
        _system, user = router.edits_for("main.py")[0]
        self.assertIn("Current content:\n```\nVALUE = 1\n\n```", user)


if __name__ == "__main__":
    unittest.main()
