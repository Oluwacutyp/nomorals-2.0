"""Tests for nomorals.builders: scaffold, run/serve, smoke, install, export.

All offline except localhost HTTP.  Never pip-installs anything: the
install tests exercise the nothing-to-install and policy-denial paths
with the capability check mocked, and assert ``subprocess.run`` is never
reached on denial.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nomorals.builders import (
    KINDS,
    build_and_verify,
    export_project,
    install_deps,
    parse_requirements,
    run_config,
    scaffold,
    serve,
    smoke_test,
    verify_export,
)
from nomorals.builders.run import ServeError
from nomorals.core.errors import ToolError
from nomorals.core.policy import Capability, CapabilitySet, Policy, PolicyDecision


class BuildersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="builders-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    # -- scaffold -----------------------------------------------------------
    def test_scaffold_all_kinds(self) -> None:
        for kind in KINDS:
            with self.subTest(kind=kind):
                result = scaffold(kind, f"proj_{kind}", self.root)
                self.assertTrue(result.project_dir.is_dir())
                self.assertTrue((result.project_dir / "run.py").is_file())
                self.assertTrue((result.project_dir / "README.md").is_file())
                self.assertTrue((result.project_dir / "requirements.txt").is_file())
                self.assertGreater(len(result.files), 3)
                manifest = json.loads(
                    (result.project_dir / ".builders.json").read_text())
                self.assertEqual(manifest["kind"], kind)
                self.assertEqual(manifest["name"], f"proj_{kind}")
                # no unsubstituted placeholders left in text files
                for rel in result.files:
                    path = result.project_dir / rel
                    try:
                        text = path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        continue
                    self.assertNotIn("$PROJECT_NAME", text, rel)
                    if text.strip():  # empty marker files carry no name
                        self.assertIn(f"proj_{kind}", text, rel)

    def test_scaffold_unknown_kind_raises(self) -> None:
        with self.assertRaises(ToolError):
            scaffold("spaceship", "x", self.root)

    def test_scaffold_empty_name_raises(self) -> None:
        with self.assertRaises(ToolError):
            scaffold("webapp", "  ", self.root)

    def test_scaffold_refuses_nonempty_dest(self) -> None:
        scaffold("webapp", "dup", self.root)
        with self.assertRaises(ToolError):
            scaffold("webapp", "dup", self.root)

    def test_rendered_projects_pass_own_tests(self) -> None:
        for kind in KINDS:
            with self.subTest(kind=kind):
                result = scaffold(kind, f"t_{kind}", self.root)
                proc = subprocess.run(
                    result.test_cmd, cwd=str(result.project_dir),
                    capture_output=True, text=True, timeout=120)
                self.assertEqual(
                    proc.returncode, 0,
                    f"{kind} template tests failed:\n{proc.stdout}\n{proc.stderr}")

    # -- run_config ----------------------------------------------------------
    def test_run_config_detects_webapp(self) -> None:
        result = scaffold("webapp", "cfgapp", self.root)
        config = run_config(result.project_dir)
        self.assertEqual(config.command, [sys.executable, "run.py"])
        self.assertEqual(config.kind, "http")
        self.assertEqual(config.entrypoint, "run.py")

    def test_run_config_detects_cli_tool(self) -> None:
        result = scaffold("cli_tool", "cfgcli", self.root)
        config = run_config(result.project_dir)
        self.assertEqual(config.kind, "cli")

    def test_run_config_no_entrypoint_raises(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ToolError):
            run_config(empty)

    # -- serve + smoke --------------------------------------------------------
    def test_serve_and_smoke_webapp(self) -> None:
        result = scaffold("webapp", "serveapp", self.root)
        with serve(result.project_dir, port=0, startup_timeout=10) as handle:
            self.assertTrue(handle.is_running())
            self.assertTrue(handle.url.startswith("http://127.0.0.1:"))
            smoke = smoke_test(handle, timeout=10)
            self.assertTrue(smoke.ok, smoke.to_dict())
            self.assertTrue(all(c.ok for c in smoke.checks))
            self.assertGreater(smoke.elapsed, 0)
        self.assertFalse(handle.is_running())

    def test_serve_fails_fast_with_stderr(self) -> None:
        proj = self.root / "crashapp"
        proj.mkdir()
        (proj / ".builders.json").write_text(
            json.dumps({"kind": "webapp", "name": "crashapp"}))
        (proj / "run.py").write_text(
            "import sys\n"
            "print('boom: missing thing', file=sys.stderr)\n"
            "sys.exit(3)\n")
        with self.assertRaises(ServeError) as ctx:
            serve(proj, port=0, startup_timeout=10)
        self.assertIn("boom: missing thing", str(ctx.exception))
        self.assertIn("boom: missing thing", ctx.exception.stderr)

    def test_serve_refuses_non_http(self) -> None:
        result = scaffold("cli_tool", "noserve", self.root)
        with self.assertRaises(ToolError):
            serve(result.project_dir, port=0, startup_timeout=5)

    def test_smoke_cli_project(self) -> None:
        result = scaffold("cli_tool", "smokecli", self.root)
        smoke = smoke_test(result.project_dir, timeout=30)
        self.assertTrue(smoke.ok, smoke.to_dict())

    def test_smoke_bot_project(self) -> None:
        result = scaffold("bot", "smokebot", self.root)
        smoke = smoke_test(result.project_dir, timeout=30)
        self.assertTrue(smoke.ok, smoke.to_dict())

    def test_smoke_will_not_autoserve_http_dir(self) -> None:
        result = scaffold("webapp", "noauto", self.root)
        with self.assertRaises(ToolError):
            smoke_test(result.project_dir, timeout=5)

    # -- install_deps ----------------------------------------------------------
    def test_install_deps_empty_requirements(self) -> None:
        result = scaffold("webapp", "nodeps", self.root)
        outcome = install_deps(result.project_dir)
        self.assertEqual(outcome.status, "nothing-to-install")
        self.assertTrue(outcome.ok)

    def test_install_deps_missing_requirements_file(self) -> None:
        proj = self.root / "noreq"
        proj.mkdir()
        outcome = install_deps(proj)
        self.assertEqual(outcome.status, "nothing-to-install")

    def test_install_deps_denied_by_default_policy(self) -> None:
        result = scaffold("webapp", "denyapp", self.root)
        (result.project_dir / "requirements.txt").write_text("requests==2.31.0\n")
        with patch("subprocess.run") as mock_run:
            outcome = install_deps(result.project_dir)
        self.assertFalse(mock_run.called, "pip must not spawn on policy denial")
        self.assertEqual(outcome.status, "denied")
        self.assertIn("exec.install", outcome.detail)
        self.assertFalse(outcome.ok)

    def test_install_deps_denied_by_mocked_check(self) -> None:
        result = scaffold("webapp", "mockdeny", self.root)
        (result.project_dir / "requirements.txt").write_text("requests==2.31.0\n")
        policy = MagicMock()
        policy.check.return_value = PolicyDecision(
            allowed=False, reason="mock says no",
            capability=Capability.EXEC_INSTALL, actor="tester")
        with patch("subprocess.run") as mock_run:
            outcome = install_deps(result.project_dir, policy=policy, actor="tester")
        self.assertFalse(mock_run.called)
        self.assertEqual(outcome.status, "denied")
        self.assertIn("exec.install", outcome.detail)
        self.assertIn("mock says no", outcome.detail)
        policy.check.assert_called()

    def test_install_deps_denied_names_net_download(self) -> None:
        # exec.install granted+confirmed but net.download missing -> names net.download
        result = scaffold("webapp", "nonet", self.root)
        (result.project_dir / "requirements.txt").write_text("requests==2.31.0\n")
        policy = Policy(default_grant=CapabilitySet.of(Capability.EXEC_INSTALL))
        token = policy.issue_confirmation(Capability.EXEC_INSTALL)
        policy.deny(Capability.NET_DOWNLOAD, note="no network for you")
        outcome = install_deps(result.project_dir, policy=policy, confirmation=token)
        self.assertEqual(outcome.status, "denied")
        self.assertIn("net.download", outcome.detail)

    def test_parse_requirements(self) -> None:
        proj = self.root / "reqparse"
        proj.mkdir()
        (proj / "requirements.txt").write_text(
            "# a comment\n\nrequests==2.31.0  # inline\n"
            "-r other.txt\n--index-url https://x\nflask\n")
        self.assertEqual(parse_requirements(proj), ["requests==2.31.0", "flask"])

    # -- export ---------------------------------------------------------------
    def test_export_round_trip(self) -> None:
        result = scaffold("webapp", "exportapp", self.root)
        # junk that must be excluded
        (result.project_dir / ".git").mkdir()
        (result.project_dir / ".git" / "HEAD").write_text("ref: x")
        pycache = result.project_dir / "__pycache__"
        pycache.mkdir()
        (pycache / "run.cpython-312.pyc").write_bytes(b"\x00" * 16)
        exported = export_project(result.project_dir, dest=self.root)
        self.assertTrue(exported.archive.is_file())
        self.assertGreater(exported.bytes, 0)
        self.assertFalse(any(".git" in f for f in exported.files))
        self.assertFalse(any("__pycache__" in f for f in exported.files))
        verified = verify_export(exported.archive)
        self.assertTrue(verified.ok, verified.problems)
        self.assertEqual(verified.files_checked, len(exported.files))

    def test_verify_export_detects_tamper(self) -> None:
        result = scaffold("cli_tool", "tamperapp", self.root)
        exported = export_project(result.project_dir, dest=self.root)
        # repack with one file modified but the manifest untouched
        tampered = self.root / "tampered.tar.gz"
        with tarfile.open(exported.archive, "r:gz") as src:
            members = src.getmembers()
            contents = {m.name: src.extractfile(m).read()
                        for m in members if m.isfile()}
        victim = next(n for n in contents if n.endswith("tool.py"))
        contents[victim] = b"# evil\n" + contents[victim]
        with tarfile.open(tampered, "w:gz") as dst:
            for name, data in contents.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                dst.addfile(info, io.BytesIO(data))
        verified = verify_export(tampered)
        self.assertFalse(verified.ok)
        self.assertTrue(any("sha256 mismatch" in p for p in verified.problems),
                        verified.problems)

    def test_verify_export_rejects_garbage(self) -> None:
        junk = self.root / "junk.tar.gz"
        junk.write_bytes(b"not a tarball")
        verified = verify_export(junk)
        self.assertFalse(verified.ok)

    def test_verify_export_missing_manifest(self) -> None:
        bare = self.root / "bare.tar.gz"
        with tarfile.open(bare, "w:gz") as tar:
            info = tarfile.TarInfo("x/y.txt")
            data = b"hi"
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        verified = verify_export(bare)
        self.assertFalse(verified.ok)
        self.assertTrue(any("MANIFEST" in p for p in verified.problems))

    # -- build_and_verify -------------------------------------------------------
    def test_build_and_verify_cli_tool(self) -> None:
        report = build_and_verify("cli_tool", "bv_cli", self.root)
        self.assertTrue(report.ok, report.summary())
        self.assertEqual(
            [s.name for s in report.steps],
            ["scaffold", "install_deps", "tests", "smoke", "export"])
        self.assertTrue(all(s.ok for s in report.steps))

    def test_build_and_verify_webapp(self) -> None:
        report = build_and_verify("webapp", "bv_web", self.root,
                                  startup_timeout=10)
        self.assertTrue(report.ok, report.summary())
        names = [s.name for s in report.steps]
        self.assertIn("serve+smoke", names)

    def test_build_and_verify_captures_failure(self) -> None:
        report = build_and_verify("webapp", "dup_name", self.root)
        # second build into the same dest must fail at scaffold, not raise
        report2 = build_and_verify("webapp", "dup_name", self.root)
        self.assertFalse(report2.ok)
        self.assertEqual(report2.steps[0].name, "scaffold")
        self.assertFalse(report2.steps[0].ok)
        self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
