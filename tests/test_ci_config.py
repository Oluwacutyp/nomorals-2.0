"""CI workflow config contract.

Tier: unit. Parses ``.github/workflows/ci.yml`` and asserts it runs the four
required stages (repo linter, layering contract, error scan, offline unit
suite), keeps the secret scan advisory-only, and does not fake a
Termux/aarch64 job.
"""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a CI/dev dependency
    yaml = None

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


@unittest.skipIf(yaml is None, "PyYAML not installed")
class CiConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(WORKFLOW.is_file(), f"missing {WORKFLOW}")
        self.doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        self.jobs = self.doc.get("jobs", {})

    def _job_text(self, job: dict) -> str:
        return "\n".join(
            str(step.get("name", "")) + "\n" + str(step.get("run", ""))
            for step in job.get("steps", [])
        )

    def test_workflow_parses_with_required_top_level_keys(self) -> None:
        self.assertIn("name", self.doc)
        # YAML 1.1 parsers read the bare key `on:` as boolean True.
        triggers = self.doc.get("on", self.doc.get(True))
        self.assertTrue(triggers, "workflow must declare triggers")
        self.assertIn("jobs", self.doc)

    def test_python_matrix_matches_requires_python(self) -> None:
        ci = self.jobs.get("ci", {})
        matrix = ci.get("strategy", {}).get("matrix", {}).get("python-version", [])
        # pyproject.toml sets requires-python >= 3.11, so the matrix must not
        # include 3.10 (pip refuses to install the package there).
        self.assertEqual(sorted(matrix), ["3.11", "3.12"])

    def test_required_stages_present(self) -> None:
        text = self._job_text(self.jobs.get("ci", {}))
        # (a) the repo's linter: ruff, per [tool.ruff] in pyproject.toml
        self.assertIn("ruff", text, "lint stage must use the repo's ruff config")
        # (b) the layering contract
        self.assertIn("test_layering", text)
        # (c) the error scanner over the package
        self.assertIn("error_scan", text)
        # (d) the offline unit suite, NM_RUN_INTEGRATION unset
        self.assertIn("unittest discover", text)
        self.assertIn("NM_RUN_INTEGRATION", text)

    def test_secret_scan_is_advisory_only(self) -> None:
        scan = self.jobs.get("secret-scan", {})
        self.assertTrue(scan, "expected a secret-scan job")
        self.assertTrue(
            scan.get("continue-on-error") is True,
            "secret-scan must be continue-on-error so it can never block a push",
        )
        text = self._job_text(scan)
        self.assertIn("advisory", scan.get("name", "").lower())
        self.assertIn("exit 0", text)

    def test_no_faked_termux_or_aarch64_job(self) -> None:
        for name in self.jobs:
            lowered = name.lower()
            self.assertNotIn("termux", lowered)
            self.assertNotIn("aarch64", lowered)
            self.assertNotIn("arm64", lowered)

    def test_tmpdir_set_for_offline_suite(self) -> None:
        ci = self.jobs.get("ci", {})
        env = ci.get("env", {}) or {}
        steps_text = self._job_text(ci)
        self.assertTrue(
            "TMPDIR" in env or "TMPDIR" in steps_text,
            "TMPDIR must be set so temp files land on the runner temp dir",
        )


if __name__ == "__main__":
    unittest.main()
