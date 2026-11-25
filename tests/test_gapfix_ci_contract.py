"""Tests for the CI thin-caller contract (census §3B.4).

The contract (nomorals/tools/ci_contract.py) is the pattern mined from
storytold/craft-common-actions: caller workflows hold no job bodies,
third-party actions are SHA-pinned, secrets are listed by name and never
appear in run text, inputs flow through declared workflow_call inputs.
"""
from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import yaml

from nomorals.tools import ci_contract
from nomorals.tools.ci_contract import (
    Violation,
    check_directory,
    check_workflow,
    load_workflow,
    resolve_pin,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
ACTIONS_DIR = REPO_ROOT / ".github" / "actions"


def _codes(violations: list[Violation]) -> list[str]:
    return [v.code for v in violations]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


SHARED = """\
name: shared
on:
  workflow_call:
    inputs:
      level:
        type: string
        required: false
        default: "info"
      mode:
        type: string
        required: true
permissions:
  contents: read
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4
        with:
          fetch-depth: 0
"""

CALLER = """\
name: ci
on:
  push:
    branches: ["main"]
permissions:
  contents: read
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
jobs:
  ci:
    uses: ./.github/workflows/shared.yml
    with:
      level: "debug"
      mode: "full"
"""


class FixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        (self.root / ".github" / "workflows").mkdir(parents=True)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _check(self, caller_text: str, shared_text: str = SHARED) -> list[Violation]:
        _write(self.root / ".github/workflows/shared.yml", shared_text)
        path = _write(self.root / ".github/workflows/ci.yml", caller_text)
        return check_workflow(load_workflow(path), str(path), self.root)

    def test_thin_caller_passes(self):
        self.assertEqual(self._check(CALLER), [])

    def test_fat_caller_refused(self):
        fat = CALLER.replace(
            "    uses: ./.github/workflows/shared.yml\n"
            "    with:\n      level: \"debug\"\n      mode: \"full\"\n",
            "    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: echo hi\n")
        codes = _codes(self._check(fat))
        self.assertIn("caller_has_steps", codes)

    def test_uses_plus_own_steps_refused(self):
        both = CALLER.replace(
            "      mode: \"full\"\n",
            "      mode: \"full\"\n    steps:\n      - run: echo hi\n")
        codes = _codes(self._check(both))
        self.assertIn("caller_has_steps", codes)

    def test_missing_permissions_refused(self):
        codes = _codes(self._check(CALLER.replace(
            "permissions:\n  contents: read\n", "")))
        self.assertIn("caller_missing_permissions", codes)

    def test_missing_concurrency_refused(self):
        codes = _codes(self._check(CALLER.replace(
            "concurrency:\n  group: ci-${{ github.ref }}\n"
            "  cancel-in-progress: true\n", "")))
        self.assertIn("caller_missing_concurrency", codes)

    def test_uses_without_ref_refused(self):
        codes = _codes(self._check(CALLER.replace(
            "uses: ./.github/workflows/shared.yml",
            "uses: storytold/craft-common-actions/.github/workflows/gate.yml")))
        self.assertIn("uses_missing_ref", codes)

    def test_tag_pinned_action_refused_in_shared_workflow(self):
        tagged = SHARED.replace(
            "actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4",
            "actions/checkout@v4")
        path = _write(self.root / ".github/workflows/shared.yml", tagged)
        codes = _codes(check_workflow(load_workflow(path), str(path), self.root))
        self.assertIn("action_not_sha_pinned", codes)

    def test_secrets_inherit_refused(self):
        inherit = CALLER.replace(
            "      mode: \"full\"\n",
            "      mode: \"full\"\n    secrets: inherit\n")
        codes = _codes(self._check(inherit))
        self.assertIn("secrets_inherit", codes)

    def test_unknown_input_refused(self):
        codes = _codes(self._check(CALLER.replace(
            '      level: "debug"\n', '      level: "debug"\n      bogus: "x"\n')))
        self.assertIn("unknown_input", codes)

    def test_missing_required_input_refused(self):
        codes = _codes(self._check(CALLER.replace('      mode: "full"\n', "")))
        self.assertIn("missing_required_input", codes)

    def test_missing_called_workflow_refused(self):
        codes = _codes(self._check(CALLER.replace(
            "shared.yml", "no-such.yml")))
        self.assertIn("missing_called_workflow", codes)


class SecretInRunTests(unittest.TestCase):
    def test_secret_in_run_step_refused(self):
        doc = yaml.safe_load("""\
name: shared
on:
  workflow_call:
    inputs: {}
permissions:
  contents: read
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - name: Leak
        run: echo "${{ secrets.MY_TOKEN }}"
""")
        codes = _codes(check_workflow(doc, "shared.yml", REPO_ROOT))
        self.assertIn("secret_in_run", codes)

    def test_secret_via_env_allowed(self):
        doc = yaml.safe_load("""\
name: shared
on:
  workflow_call:
    inputs: {}
permissions:
  contents: read
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - name: Use
        env:
          TOKEN: ${{ secrets.MY_TOKEN }}
        run: ./use-token.sh
""")
        self.assertEqual(check_workflow(doc, "shared.yml", REPO_ROOT), [])


class ResolvePinTests(unittest.TestCase):
    def test_local_path_resolves_to_head_sha(self):
        pin = resolve_pin("./.github/workflows/shared-ci.yml", REPO_ROOT)
        self.assertEqual(pin["kind"], "local")
        self.assertTrue(pin["pinned"])
        self.assertRegex(pin["resolved"] or "", r"^[0-9a-f]{40}$")

    def test_sha_action_pinned(self):
        pin = resolve_pin(
            "actions/checkout@11d5960a326750d5838078e36cf38b85af677262")
        self.assertEqual(pin["kind"], "action")
        self.assertTrue(pin["pinned"])
        self.assertEqual(pin["resolved"],
                         "11d5960a326750d5838078e36cf38b85af677262")

    def test_tag_action_not_pinned(self):
        pin = resolve_pin("actions/checkout@v4")
        self.assertFalse(pin["pinned"])
        self.assertIsNone(pin["resolved"])

    def test_reusable_at_named_ref_is_tracking(self):
        pin = resolve_pin(
            "storytold/craft-common-actions/.github/workflows/gate.yml@main")
        self.assertEqual(pin["kind"], "reusable")
        self.assertTrue(pin["pinned"])
        self.assertIsNone(pin["resolved"])

    def test_bare_uses_has_no_ref(self):
        pin = resolve_pin("actions/checkout")
        self.assertEqual(pin["ref"], "")
        self.assertFalse(pin["pinned"])


class RealRepoTests(unittest.TestCase):
    def test_real_workflows_pass_the_contract(self):
        results = check_directory(WORKFLOWS, REPO_ROOT)
        self.assertEqual(set(results), {f"{WORKFLOWS}/ci.yml",
                                       f"{WORKFLOWS}/shared-ci.yml"})
        for path, violations in results.items():
            self.assertEqual(violations, [], path)

    def test_caller_is_thin_and_calls_the_shared_workflow(self):
        doc = load_workflow(WORKFLOWS / "ci.yml")
        jobs = doc["jobs"]
        self.assertEqual(list(jobs), ["ci"])
        job = jobs["ci"]
        self.assertEqual(job["uses"], "./.github/workflows/shared-ci.yml")
        self.assertNotIn("runs-on", job)
        self.assertNotIn("steps", job)
        self.assertIn("permissions", doc)
        self.assertIn("concurrency", doc)
        shared = load_workflow(WORKFLOWS / "shared-ci.yml")
        on_block = shared.get(True) or shared.get("on") or {}
        declared = (on_block.get("workflow_call") or {}).get("inputs") or {}
        for key in job["with"]:
            self.assertIn(key, declared)

    def test_shared_workflow_matrix_comes_from_inputs(self):
        doc = load_workflow(WORKFLOWS / "shared-ci.yml")
        matrix = doc["jobs"]["ci"]["strategy"]["matrix"]
        self.assertIn("fromJSON(inputs.python-versions)",
                      str(matrix["python-version"]))
        self.assertIn("fromJSON(inputs.shards)", str(matrix["shard"]))

    def test_composite_action_parses_and_pins_setup_python(self):
        doc = load_workflow(ACTIONS_DIR / "ci-setup" / "action.yml")
        self.assertEqual(doc["runs"]["using"], "composite")
        uses = [s["uses"] for s in doc["runs"]["steps"] if "uses" in s]
        self.assertEqual(len(uses), 1)
        pin = resolve_pin(uses[0], REPO_ROOT)
        self.assertTrue(pin["pinned"])

    def test_no_job_body_was_lost_in_the_split(self):
        """Every run: command the old fat ci.yml executed still executes
        somewhere in the new split (shared workflow or composite action)."""
        try:
            old = subprocess.check_output(
                ["git", "show", "8589d5d2:.github/workflows/ci.yml"],
                cwd=REPO_ROOT, text=True, timeout=30)
        except (subprocess.SubprocessError, OSError) as exc:
            self.skipTest(f"cannot read old ci.yml from git: {exc}")
        old_doc = yaml.safe_load(old)

        def run_strings(doc: dict) -> set[str]:
            found: set[str] = set()
            for job in (doc.get("jobs") or {}).values():
                for step in (job or {}).get("steps") or []:
                    if isinstance(step, dict) and isinstance(step.get("run"), str):
                        found.add(step["run"])
            return found

        old_runs = run_strings(old_doc)
        new_runs = run_strings(load_workflow(WORKFLOWS / "shared-ci.yml"))
        action_doc = load_workflow(ACTIONS_DIR / "ci-setup" / "action.yml")
        for step in action_doc["runs"]["steps"]:
            if isinstance(step.get("run"), str):
                new_runs.add(step["run"])
        missing = old_runs - new_runs
        self.assertEqual(missing, set())

    # 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library.
    def test_tool_count_still_778(self):
        from types import SimpleNamespace

        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        # 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library.
        self.assertEqual(len(registry._tools), 787)


class CliTests(unittest.TestCase):
    def test_cli_check_real_dir_returns_zero(self):
        self.assertEqual(ci_contract.main(
            ["check", str(WORKFLOWS)]), 0)

    def test_cli_check_json(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ci_contract.main(["check", "--json", str(WORKFLOWS)])
        self.assertEqual(rc, 0)
        parsed = json.loads(buf.getvalue())
        self.assertEqual(parsed, {f"{WORKFLOWS}/ci.yml": [],
                                 f"{WORKFLOWS}/shared-ci.yml": []})

    def test_cli_usage_returns_two(self):
        self.assertEqual(ci_contract.main([]), 2)
        self.assertEqual(ci_contract.main(["bogus"]), 2)


if __name__ == "__main__":
    unittest.main()
