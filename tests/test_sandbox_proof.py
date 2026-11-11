"""Sandbox live proof: hostile probes against the real sandbox backend.

Each probe runs through the real ``run_sandboxed`` (never a mock of
the sandbox itself) with an outside-sandbox control that must
succeed, so a pass means enforcement rather than a broken probe.
The assertions are backend-relative on purpose: a namespace backend
must actually block egress, jail the filesystem and hide host PIDs;
the rlimit floor provides none of those, so on it the proof asserts
the flags honestly read False and the rlimit backstops still hold.
A flag that disagrees with its observation is a lie and fails.

Skipped entirely off POSIX: the sandboxed probes need ``/bin/sh``,
which a Windows host does not provide.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.tools.sandbox_proof import (
    _TOKEN,
    backend_evidence,
    probe_backend_selection,
    probe_cpu_ceiling,
    probe_env_scrub,
    probe_file_size_cap,
    probe_fs_escape,
    probe_network_egress,
    probe_pid_namespace,
    render_proof,
    run_proof,
    verdict_for,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="sandboxed probes need a POSIX /bin/sh host",
)


class _ProofCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="sandbox-proof-test-")
        self.addCleanup(tmp.cleanup)
        self.workdir = Path(tmp.name) / "work"
        self.workdir.mkdir()
        self.outside = Path(tmp.name) / "outside"
        self.outside.mkdir()

    def _secret(self) -> Path:
        path = self.outside / "secret.txt"
        path.write_text(_TOKEN + "\n", encoding="utf-8")
        return path


class BackendSelectionTests(_ProofCase):
    def test_selection_is_the_strongest_working_backend(self):
        evidence = backend_evidence()
        self.assertEqual(evidence["selected"], evidence["expected"])
        # The reason names both candidates, so a reader can see why
        # the tier below the top was (not) chosen.
        self.assertIn("bwrap", evidence["reason"])
        self.assertIn("unshare", evidence["reason"])
        probe = probe_backend_selection()
        self.assertEqual(probe["status"], "pass", probe["evidence"])
        # Backend-relative on purpose: on a host where unshare works
        # this selects unshare, on this container (which refuses the
        # /proc mount unshare needs) it selects the rlimit floor.
        # Either way the selected backend is the strongest working one
        # and the reason names why the tiers above it were not chosen.

    def test_explicit_backend_request_is_honoured(self):
        probe = probe_backend_selection("rlimit")
        self.assertEqual(probe["status"], "pass", probe["evidence"])


class NetworkEgressTests(_ProofCase):
    def test_flag_matches_the_observed_egress(self):
        probe = probe_network_egress(self.workdir)
        self.assertEqual(probe["status"], "pass", probe["evidence"])
        # The flag must agree with what the child actually observed:
        # enforced ⟺ blocked. On the rlimit floor egress is open and
        # the flag must honestly read False.
        self.assertEqual(
            probe["flag"], probe["observed"] == "blocked", probe["evidence"],
        )


class FsEscapeTests(_ProofCase):
    def test_flag_matches_the_observed_read(self):
        secret = self._secret()
        probe = probe_fs_escape(self.workdir, secret, _TOKEN)
        self.assertEqual(probe["status"], "pass", probe["evidence"])
        self.assertEqual(
            probe["flag"], probe["observed"] == "blocked", probe["evidence"],
        )


class PidNamespaceTests(_ProofCase):
    def test_observation_matches_the_backend_that_ran(self):
        probe = probe_pid_namespace(self.workdir)
        self.assertEqual(probe["status"], "pass", probe["evidence"])


class CpuCeilingTests(_ProofCase):
    def test_spinner_dies_at_the_cpu_ceiling(self):
        probe = probe_cpu_ceiling(self.workdir)
        self.assertEqual(probe["status"], "pass", probe["evidence"])


class FileSizeCapTests(_ProofCase):
    def test_write_stops_at_the_cap(self):
        probe = probe_file_size_cap(self.workdir)
        self.assertEqual(probe["status"], "pass", probe["evidence"])


class EnvScrubTests(_ProofCase):
    def test_parent_secret_does_not_reach_the_child(self):
        probe = probe_env_scrub(self.workdir)
        self.assertEqual(probe["status"], "pass", probe["evidence"])
        # The probe restores the parent environment it borrowed.
        self.assertNotIn("NM_SANDBOX_PROOF_SECRET", os.environ)


class VerdictTests(unittest.TestCase):
    def test_all_pass_is_pass(self):
        probes = [
            {"name": "a", "status": "pass", "evidence": "ok"},
            {"name": "b", "status": "pass", "evidence": "ok"},
        ]
        verdict, failures = verdict_for(probes)
        self.assertEqual(verdict, "pass")
        self.assertEqual(failures, [])

    def test_one_failure_is_fail_and_named(self):
        probes = [
            {"name": "a", "status": "pass", "evidence": "ok"},
            {"name": "b", "status": "fail", "evidence": "the flag lied"},
        ]
        verdict, failures = verdict_for(probes)
        self.assertEqual(verdict, "fail")
        self.assertEqual(failures, ["b: the flag lied"])


class FullProofTests(_ProofCase):
    def test_full_proof_passes_on_this_host(self):
        report = run_proof()
        self.assertEqual(report["verdict"], "pass", report["failures"])
        self.assertIn(report["backend"], ("bwrap", "unshare", "rlimit"))
        # The reported flags are exactly the backend's claims.
        self.assertEqual(
            report["network_enforced"],
            report["backend"] in ("bwrap", "unshare"),
        )
        self.assertEqual(
            report["fs_confined"], report["backend"] == "bwrap",
        )
        self.assertEqual(
            [p["name"] for p in report["probes"]],
            ["backend-selection", "network-egress", "fs-escape",
             "pid-namespace", "cpu-ceiling", "file-size-cap", "env-scrub"],
        )
        rendered = render_proof(report)
        self.assertIn("VERDICT: PASS", rendered)
        self.assertIn("backend: " + report["backend"], rendered)
        self.assertIn(
            "network_enforced=" + str(report["network_enforced"]), rendered,
        )
        self.assertIn(
            "fs_confined=" + str(report["fs_confined"]), rendered,
        )
        for probe in report["probes"]:
            self.assertIn(probe["name"], rendered)

    def test_cli_proof_runs_and_exits_zero(self):
        from nomorals.cmdline.commands.sandbox import _cmd_sandbox

        args = SimpleNamespace(sandbox_action="proof")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_sandbox(args, None)
        self.assertEqual(rc, 0)
        self.assertIn("VERDICT: PASS", buf.getvalue())

    def test_cli_unknown_action_exits_two(self):
        from nomorals.cmdline.commands.sandbox import _cmd_sandbox

        args = SimpleNamespace(sandbox_action="nope")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_sandbox(args, None)
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
