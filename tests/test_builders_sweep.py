"""Sweep tests for the builders module upgrade (2026-10-10).

Covers what's new: style themes, template catalog + scaffold options
(variables/dry_run/overwrite/git_init), k8s-style probes + serve
supervision (restart policies, watch, logs, stop_all), smoke
expectations, installer selection + lockfiles, reproducible exports,
zip splitting + delivery retries, selectable verify steps + report
persistence, and the AppBuilder additions (themes, readyz, dockerize,
ci, patch, duplicate, remove).

All offline except localhost HTTP.  Never pip-installs anything.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.builders import (
    KINDS,
    AppBuilder,
    BuildReport,
    HttpExpectation,
    HttpProbe,
    build_and_verify,
    build_zip_and_deliver,
    describe,
    install_deps,
    list_templates,
    resolve_installer,
    resolve_theme,
    run_config,
    run_probe,
    scaffold,
    serve,
    smoke_test,
    status_glyph,
    stop_all,
    tcp_probe,
    theme_names,
    validate_sources,
    verify_lock,
    verify_reproducible,
    zip_project,
    banner,
    render_steps,
)
from nomorals.builders.deliver import _render_caption, _send_zip
from nomorals.builders.export import export_project, verify_export
from nomorals.builders.install import (
    ensure_venv,
    generate_lock,
)
from nomorals.builders.run import TcpProbe
from nomorals.builders.scaffold import _builtin_variables
from nomorals.builders.style import render_kv
from nomorals.core.errors import ToolError


class _Settings:
    def __init__(self, ws: str) -> None:
        self.workspace_dir = ws


class _Context:
    def __init__(self, ws: str) -> None:
        self.settings = _Settings(ws)


class _FakeGateway:
    """Duck-typed chat gateway: fails `fail_times` sends, then succeeds."""

    def __init__(self, fail_times: int = 0, message_id: str = "mid-1"):
        self.fail_times = fail_times
        self.message_id = message_id
        self.calls: list[dict] = []

    def send_file(self, platform, chat, path, caption="", max_send_mb=0.0):
        self.calls.append({"platform": platform, "chat": chat,
                           "path": path, "caption": caption})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("transient send failure")
        return SimpleNamespace(ok=True, message_id=self.message_id,
                               error="")


# ── style ──────────────────────────────────────────────────────────────

class StyleTests(unittest.TestCase):
    def test_theme_names(self):
        names = theme_names()
        self.assertIn("neon", names)
        self.assertIn("plain", names)

    def test_plain_theme_is_noop(self):
        th = resolve_theme("plain")
        self.assertEqual(th.paint("hi", "\033[31m"), "hi")
        self.assertEqual(status_glyph(True, th), "✓")

    def test_no_color_env_forces_plain(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertEqual(resolve_theme().name, "plain")

    def test_banner_and_steps_render(self):
        th = resolve_theme("plain")
        out = banner("hello", "world", theme=th)
        self.assertIn("hello", out)
        self.assertIn("world", out)
        steps = [{"name": "scaffold", "ok": True, "elapsed": 1.2,
                  "detail": "5 files"},
                 {"name": "tests", "ok": False, "elapsed": 0.3,
                  "detail": "boom"}]
        rendered = render_steps(steps, theme=th)
        self.assertIn("✓", rendered)
        self.assertIn("✗", rendered)
        self.assertIn("scaffold", rendered)
        kv = render_kv([("a", 1), ("bb", 2)], theme=th)
        self.assertIn("a", kv)


# ── scaffold ───────────────────────────────────────────────────────────

class ScaffoldSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="scaffold-sweep-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_list_templates_catalog(self):
        infos = list_templates()
        self.assertEqual({i.kind for i in infos}, set(KINDS))
        for info in infos:
            d = info.to_dict()
            self.assertTrue(d["title"])
            self.assertTrue(d["file_count"] > 0)

    def test_describe(self):
        info = describe("webapp")
        self.assertEqual(info.kind, "webapp")
        self.assertIn("PROJECT_NAME", info.variables)
        with self.assertRaises(ToolError):
            describe("nope")

    def test_builtin_variables(self):
        varmap = _builtin_variables("My App", {"author": "Zed"})
        self.assertEqual(varmap["PROJECT_NAME"], "My App")
        self.assertEqual(varmap["AUTHOR"], "Zed")  # user wins
        self.assertIn("YEAR", varmap)
        self.assertIn("PROJECT_SLUG", varmap)

    def test_scaffold_custom_variables(self):
        result = scaffold("webapp", "varapp", self.root,
                          variables={"description": "custom desc here"})
        self.assertEqual(result.variables["DESCRIPTION"], "custom desc here")
        run_py = result.project_dir / "run.py"
        self.assertIn("varapp", run_py.read_text())

    def test_scaffold_dry_run_touches_nothing(self):
        result = scaffold("bot", "drybot", self.root, dry_run=True)
        self.assertTrue(result.dry_run)
        self.assertTrue(result.files)
        self.assertFalse((self.root / "drybot").exists())

    def test_scaffold_overwrite(self):
        scaffold("bot", "owbot", self.root)
        with self.assertRaises(ToolError):
            scaffold("bot", "owbot", self.root)
        result = scaffold("bot", "owbot", self.root, overwrite=True)
        self.assertTrue((result.project_dir / "bot.py").exists())

    def test_scaffold_git_init(self):
        result = scaffold("cli_tool", "gitapp", self.root, git_init=True)
        self.assertTrue(result.git_initialized, result.git_note)
        self.assertTrue((result.project_dir / ".git").is_dir())

    def test_scaffold_still_rejects_bad_name(self):
        with self.assertRaises(ToolError):
            scaffold("webapp", "a/b", self.root)


# ── run: probes + supervision ──────────────────────────────────────────

class RunSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="run-sweep-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proj = scaffold("webapp", "probeapp", self.root).project_dir

    def tearDown(self):
        stop_all()

    def test_health_path_detected(self):
        config = run_config(self.proj)
        self.assertEqual(config.kind, "http")
        self.assertEqual(config.health_path, "/api/health")
        self.assertIsNotNone(config.probe)
        self.assertEqual(config.probe.path, "/api/health")

    def test_probe_healthy_status_ranges(self):
        probe = HttpProbe()
        self.assertTrue(probe.healthy_status(200))
        self.assertTrue(probe.healthy_status(301))
        self.assertFalse(probe.healthy_status(500))
        pinned = HttpProbe(statuses=(200,))
        self.assertFalse(pinned.healthy_status(201))

    def test_run_probe_against_live_server(self):
        with serve(self.proj, startup_timeout=15) as handle:
            port = int(handle.url.rsplit(":", 1)[1])
            result = run_probe(HttpProbe(path="/api/health"), "127.0.0.1",
                               port)
            self.assertTrue(result.ok, result.detail)
            self.assertEqual(result.status, 200)
            bad = run_probe(HttpProbe(path="/nope", failure_threshold=2,
                                      period=0.2), "127.0.0.1", port)
            self.assertFalse(bad.ok)
            self.assertEqual(bad.attempts, 2)

    def test_tcp_probe(self):
        with serve(self.proj, startup_timeout=15) as handle:
            port = int(handle.url.rsplit(":", 1)[1])
            check = tcp_probe("127.0.0.1", port)
            self.assertTrue(check.ok, check.detail)
            dead = tcp_probe("127.0.0.1", 1)
            self.assertFalse(dead.ok)

    def test_serve_runs_startup_probe(self):
        # webapp serves /api/health; serve() must pass its own probe
        with serve(self.proj, startup_timeout=15) as handle:
            self.assertTrue(handle.is_running())
            self.assertGreater(handle.uptime, 0)
            logs = handle.logs(5)
            self.assertIsInstance(logs, str)

    def test_restart_policy_on_failure(self):
        with serve(self.proj, startup_timeout=15,
                   restart_policy="on-failure", max_restarts=2) as handle:
            pid_before = handle.pid
            os.kill(pid_before, signal.SIGKILL)
            deadline = time.time() + 15
            while handle.restarts == 0 and time.time() < deadline:
                time.sleep(0.3)
            self.assertGreaterEqual(handle.restarts, 1)
            self.assertNotEqual(handle.pid, pid_before)
            self.assertIn("exit=", handle.last_restart_reason)

    def test_manual_restart(self):
        with serve(self.proj, startup_timeout=15) as handle:
            pid_before = handle.pid
            handle.restart()
            self.assertEqual(handle.restarts, 1)
            self.assertNotEqual(handle.pid, pid_before)
            self.assertTrue(handle.is_running())

    def test_watch_restarts_on_file_change(self):
        with serve(self.proj, startup_timeout=15, watch=True) as handle:
            run_py = self.proj / "run.py"
            run_py.write_text(run_py.read_text() + "\n# touch\n")
            deadline = time.time() + 15
            while handle.restarts == 0 and time.time() < deadline:
                time.sleep(0.3)
            self.assertGreaterEqual(handle.restarts, 1)
            self.assertIn("watch", handle.last_restart_reason)

    def test_stop_all(self):
        handle = serve(self.proj, startup_timeout=15)
        self.assertTrue(handle.is_running())
        out = stop_all()
        self.assertGreaterEqual(out["stopped"], 1)
        self.assertFalse(handle.is_running())

    def test_bad_restart_policy_rejected(self):
        with self.assertRaises(ToolError):
            serve(self.proj, restart_policy="sometimes")


# ── smoke ──────────────────────────────────────────────────────────────

class SmokeSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="smoke-sweep-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proj = scaffold("webapp", "smokeapp", self.root).project_dir

    def tearDown(self):
        stop_all()

    def test_expectation_label_and_health(self):
        exp = HttpExpectation(path="/api/health", body_contains="ok")
        self.assertIn("/api/health", exp.label())
        self.assertTrue(exp.healthy(200))
        self.assertTrue(exp.healthy(302))
        self.assertFalse(exp.healthy(404))

    def test_smoke_with_expectations(self):
        with serve(self.proj, startup_timeout=15) as handle:
            result = smoke_test(
                handle,
                expectations=[
                    HttpExpectation(path="/api/health",
                                    body_contains='"status": "ok"'),
                    HttpExpectation(path="/", statuses=(200,)),
                ])
            self.assertTrue(result.ok, result.to_dict())
            self.assertEqual(len(result.checks), 2)

    def test_smoke_failing_expectation(self):
        with serve(self.proj, startup_timeout=15) as handle:
            result = smoke_test(
                handle,
                expectations=[HttpExpectation(path="/missing-thing")])
            self.assertFalse(result.ok)
            self.assertIn("404", result.checks[0].detail)

    def test_smoke_cli_still_works(self):
        cli = scaffold("cli_tool", "smokecli", self.root).project_dir
        result = smoke_test(cli)
        self.assertTrue(result.ok, result.to_dict())


# ── install ────────────────────────────────────────────────────────────

class InstallSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="install-sweep-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proj = self.root / "proj"
        self.proj.mkdir()
        (self.proj / "requirements.txt").write_text("requests>=2.0\n")

    def test_resolve_installer(self):
        # uv is not installed in this environment -> pip fallback
        self.assertEqual(resolve_installer("auto"), "pip")
        self.assertEqual(resolve_installer("pip"), "pip")
        with self.assertRaises(ToolError):
            resolve_installer("uv")
        with self.assertRaises(ToolError):
            resolve_installer("brew")

    def test_install_denied_by_default(self):
        result = install_deps(self.proj)
        self.assertEqual(result.status, "denied")
        self.assertFalse(result.ok)

    def test_install_cmd_construction_uv(self):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with patch("shutil.which", return_value="/usr/bin/uv"):
            with patch("nomorals.builders.install.subprocess.run",
                       side_effect=fake_run):
                from nomorals.core.policy import (
                    Capability, CapabilitySet, Policy)
                policy = Policy(default_grant=CapabilitySet.all())
                token = policy.issue_confirmation(Capability.EXEC_INSTALL)
                result = install_deps(self.proj, policy=policy,
                                      confirmation=token,
                                      installer="auto", dry_run=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.installer_used, "uv")
        self.assertTrue(result.dry_run)
        self.assertIn("uv", calls[0])
        self.assertIn("--dry-run", calls[0])

    def test_install_cmd_construction_pip_venv(self):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        venv = self.proj / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        with patch("nomorals.builders.install.subprocess.run",
                   side_effect=fake_run):
            from nomorals.core.policy import (
                Capability, CapabilitySet, Policy)
            policy = Policy(default_grant=CapabilitySet.all())
            token = policy.issue_confirmation(Capability.EXEC_INSTALL)
            result = install_deps(self.proj, policy=policy,
                                  confirmation=token, installer="pip",
                                  venv=".venv", use_lock=False)
        self.assertTrue(result.ok)
        self.assertEqual(result.installer_used, "pip")
        self.assertTrue(result.venv.endswith(".venv"))
        self.assertIn(str(venv / "bin" / "python"), calls[0][0])

    def test_generate_lock_denied_by_default(self):
        result = generate_lock(self.proj)
        self.assertEqual(result.status, "denied")

    def test_verify_lock_missing(self):
        out = verify_lock(self.proj)
        self.assertFalse(out["ok"])
        self.assertIn("generate_lock", out["reason"])

    def test_verify_lock_fresh_and_stale(self):
        lock = self.proj / "requirements.lock.txt"
        lock.write_text("requests==2.31.0 \\\n")
        out = verify_lock(self.proj)
        self.assertTrue(out["ok"], out)
        # touch requirements.txt newer than the lock -> stale
        future = time.time() + 50
        os.utime(self.proj / "requirements.txt", (future, future))
        out = verify_lock(self.proj)
        self.assertFalse(out["ok"])
        self.assertIn("newer", out["reason"])

    def test_ensure_venv(self):
        venv = ensure_venv(self.proj)
        self.assertTrue((venv / "bin" / "python").is_file()
                        or (venv / "Scripts" / "python.exe").is_file())
        # second call reuses it
        self.assertEqual(ensure_venv(self.proj), venv)


# ── export ─────────────────────────────────────────────────────────────

class ExportSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="export-sweep-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proj = scaffold("cli_tool", "expapp", self.root).project_dir

    def test_export_result_carries_sha256(self):
        dest = self.root / "a"
        dest.mkdir(exist_ok=True)
        result = export_project(self.proj, dest=dest)
        self.assertEqual(len(result.sha256), 64)
        self.assertEqual(result.file_count, len(result.files))
        self.assertFalse(result.reproducible)

    def test_reproducible_exports_are_byte_identical(self):
        out = verify_reproducible(self.proj, epoch=1700000000)
        self.assertTrue(out["ok"], out["detail"])
        self.assertEqual(len(out["sha256"]), 64)

    def test_reproducible_flag_and_manifest(self):
        import tarfile
        dest = self.root / "r"
        dest.mkdir()
        result = export_project(self.proj, dest=dest, reproducible=True,
                                epoch=1700000000)
        self.assertTrue(result.reproducible)
        with tarfile.open(result.archive, "r:gz") as tar:
            manifest_name = next(n for n in tar.getnames()
                                 if n.endswith("MANIFEST.json"))
            manifest = json.loads(
                tar.extractfile(manifest_name).read().decode())
        self.assertTrue(manifest["reproducible"])
        self.assertIn("toolchain", manifest)
        self.assertIn("python", manifest["toolchain"])
        verified = verify_export(result.archive)
        self.assertTrue(verified.ok, verified.problems)

    def test_source_date_epoch_env(self):
        with patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "1700000000"}):
            from nomorals.builders.export import source_date_epoch
            self.assertEqual(source_date_epoch(), 1700000000)
        with patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "bogus"}):
            from nomorals.builders.export import source_date_epoch
            self.assertIsNone(source_date_epoch())


# ── deliver ────────────────────────────────────────────────────────────

class DeliverSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="deliver-sweep-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proj = scaffold("cli_tool", "delapp", self.root).project_dir
        # pad the project so splitting actually splits
        (self.proj / "big.bin").write_bytes(os.urandom(300_000))

    def test_zip_split_parts(self):
        dest = self.root / "z"
        dest.mkdir()
        result = zip_project(self.proj, dest=dest, split_mb=0.1)
        self.assertGreater(len(result.parts), 1)
        total = sum(p.stat().st_size for p in result.parts)
        self.assertEqual(total, result.bytes)
        self.assertTrue(all(p.name.startswith(result.archive.name + ".part")
                            for p in result.parts))
        self.assertEqual(len(result.sha256), 64)

    def test_zip_no_split_single_part(self):
        dest = self.root / "z2"
        dest.mkdir()
        result = zip_project(self.proj, dest=dest)
        self.assertEqual(result.parts, [result.archive])

    def test_caption_templating(self):
        out = _render_caption("{name} v1 {mb}MB {parts}p {bytes}b {archive}",
                              name="n", kind="k", archive=Path("a.zip"),
                              size=1048576, parts=3)
        self.assertEqual(out, "n v1 1.0MB 3p 1048576b a.zip")

    def test_send_retries_then_succeeds(self):
        dest = self.root / "z3"
        dest.mkdir()
        zipped = zip_project(self.proj, dest=dest)
        gw = _FakeGateway(fail_times=2)
        result = _send_zip(gw, zipped, platform="telegram", chat="@owner",
                           caption="hi", max_send_mb=0.0,
                           retries=3, backoff=0.01)
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(gw.calls), 3)

    def test_send_gives_up_after_retries(self):
        dest = self.root / "z4"
        dest.mkdir()
        zipped = zip_project(self.proj, dest=dest)
        gw = _FakeGateway(fail_times=99)
        result = _send_zip(gw, zipped, platform="telegram", chat="@owner",
                           caption="hi", max_send_mb=0.0,
                           retries=1, backoff=0.01)
        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.problems)

    def test_deliver_report_fancy_summary(self):
        report = build_zip_and_deliver(
            "cli_tool", "fancyapp", self.root,
            platform="telegram", chat="@owner",
            gateway=_FakeGateway(), startup_timeout=10)
        self.assertTrue(report.ok, report.summary())
        fancy = report.fancy_summary(theme="plain")
        self.assertIn("fancyapp", fancy)
        self.assertIn("✓", fancy)


# ── verify ─────────────────────────────────────────────────────────────

class VerifySweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="verify-sweep-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_step_selection(self):
        report = build_and_verify("cli_tool", "selapp", self.root,
                                  steps=["scaffold", "validate"])
        self.assertTrue(report.ok, report.summary())
        self.assertEqual([s.name for s in report.steps],
                         ["scaffold", "validate"])

    def test_skip_export(self):
        report = build_and_verify("cli_tool", "skipapp", self.root,
                                  skip=["export", "install_deps"])
        names = [s.name for s in report.steps]
        self.assertNotIn("export", names)
        self.assertNotIn("install_deps", names)
        self.assertIn("tests", names)

    def test_fail_fast(self):
        report = build_and_verify("cli_tool", "ffapp", self.root,
                                  fail_fast=True)
        self.assertTrue(report.ok)

    def test_no_steps_selected_raises(self):
        with self.assertRaises(ValueError):
            build_and_verify("cli_tool", "nosteps", self.root, steps=[])

    def test_validate_sources(self):
        proj = scaffold("cli_tool", "valapp", self.root).project_dir
        step = validate_sources(proj)
        self.assertEqual(step.name, "validate")
        self.assertTrue(step.ok, step.detail)
        self.assertIn("compile ok", step.detail)
        # break a file -> validation fails honestly
        (proj / "broken.py").write_text("def f(:\n")
        step = validate_sources(proj)
        self.assertFalse(step.ok)

    def test_report_save_load_roundtrip(self):
        report = build_and_verify("cli_tool", "persistapp", self.root,
                                  steps=["scaffold", "tests"])
        path = self.root / "report.json"
        saved = report.save(path)
        self.assertTrue(saved.is_file())
        loaded = BuildReport.load(path)
        self.assertEqual(loaded.name, "persistapp")
        self.assertEqual([s.name for s in loaded.steps],
                         [s.name for s in report.steps])
        self.assertEqual(loaded.ok, report.ok)

    def test_report_markdown(self):
        report = build_and_verify("cli_tool", "mdapp", self.root,
                                  steps=["scaffold"])
        md = report.as_markdown()
        self.assertIn("# build_and_verify", md)
        self.assertIn("scaffold", md)

    def test_report_fancy_summary(self):
        report = build_and_verify("cli_tool", "fapp", self.root,
                                  steps=["scaffold", "tests"])
        fancy = report.fancy_summary(theme="plain")
        self.assertIn("fapp", fancy)
        self.assertIn("✓", fancy)

    def test_scaffold_kwargs_forwarded(self):
        report = build_and_verify(
            "cli_tool", "gitfwd", self.root, steps=["scaffold"],
            scaffold_kwargs={"git_init": True})
        self.assertTrue(report.ok)
        self.assertTrue((self.root / "gitfwd" / ".git").is_dir())


# ── app_builder additions ──────────────────────────────────────────────

class AppBuilderSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="appbuilder-sweep-")
        self.addCleanup(self.tmp.cleanup)
        self.builder = AppBuilder(_Context(self.tmp.name))

    def test_theme_neon_css(self):
        result = self.builder.build({"name": "neonapp", "stack": "static",
                                     "theme": "neon"}, verify=False)
        css = Path(result["dir"]) / "style.css"
        self.assertIn("#00f0ff", css.read_text())
        man = self.builder.info("neonapp")
        self.assertEqual(man["theme"], "neon")

    def test_theme_light_css(self):
        result = self.builder.build({"name": "lightapp", "stack": "static",
                                     "theme": "light"}, verify=False)
        css = Path(result["dir"]) / "style.css"
        self.assertIn("#0969da", css.read_text())

    def test_bad_theme_rejected(self):
        with self.assertRaises(ToolError):
            self.builder.build({"name": "badtheme", "stack": "static",
                                "theme": "holographic"}, verify=False)

    def test_readyz_endpoints(self):
        flask = self.builder.build({"name": "rzflask", "stack": "flask"},
                                   verify=False)
        app_py = Path(flask["dir"]) / "app.py"
        self.assertIn("/readyz", app_py.read_text())
        fastapi = self.builder.build({"name": "rzfapi", "stack": "fastapi"},
                                     verify=False)
        self.assertIn("/readyz", (Path(fastapi["dir"]) / "main.py").read_text())
        express = self.builder.build({"name": "rzexp", "stack": "express"},
                                     verify=False)
        self.assertIn("/readyz",
                      (Path(express["dir"]) / "server.js").read_text())
        nxt = self.builder.build({"name": "rznxt", "stack": "nextjs"},
                                 verify=False)
        self.assertTrue(
            (Path(nxt["dir"]) / "app" / "api" / "readyz" / "route.ts").is_file())

    def test_stacks_metadata(self):
        stacks = self.builder.stacks()
        by_name = {s["stack"]: s for s in stacks}
        self.assertFalse(by_name["flask"]["server"] is False)
        self.assertTrue(by_name["flask"]["server"])
        self.assertFalse(by_name["cli-python"]["server"])
        self.assertIn("python", by_name["fastapi"]["language"])
        self.assertEqual(by_name["static"]["themes"], ["dark", "light", "neon"])
        self.assertEqual(by_name["flask"]["themes"], [])
        self.assertEqual(by_name["flask"]["health_path"], "/api/health")

    def test_dockerize(self):
        self.builder.build({"name": "dockapp", "stack": "flask"},
                           verify=False)
        out = self.builder.dockerize("dockapp")
        self.assertIn("Dockerfile", out["dockerfile"])
        self.assertTrue(Path(out["dockerfile"]).is_file())
        self.assertTrue((Path(self.builder._apps_dir) / "dockapp"
                         / ".dockerignore").is_file())
        man = self.builder.info("dockapp")
        self.assertTrue(man["dockerfile"])
        # second call skips honestly
        out2 = self.builder.dockerize("dockapp")
        self.assertTrue(out2["skipped"])

    def test_dockerize_bot_skipped(self):
        self.builder.build({"name": "dockbot", "stack": "bot-telegram"},
                           verify=False)
        out = self.builder.dockerize("dockbot")
        self.assertTrue(out["skipped"])  # ships its own Dockerfile

    def test_ci(self):
        self.builder.build({"name": "ciapp", "stack": "flask"}, verify=False)
        out = self.builder.ci("ciapp")
        wf = Path(out["workflow"])
        self.assertTrue(wf.is_file())
        text = wf.read_text()
        self.assertIn("curl", text)
        self.assertIn("/api/health", text)
        man = self.builder.info("ciapp")
        self.assertTrue(man["ci"])

    def test_patch(self):
        built = self.builder.build({"name": "patchapp", "stack": "static"},
                                   verify=False)
        out = self.builder.patch("patchapp", {"extra.txt": "hello",
                                              "sub/note.md": "# note"})
        self.assertEqual(sorted(out["patched"]), ["extra.txt", "sub/note.md"])
        self.assertTrue(out["validation"]["ok"])
        self.assertTrue((Path(built["dir"]) / "extra.txt").is_file())
        with self.assertRaises(ToolError):
            self.builder.patch("patchapp", {"../evil.txt": "x"})

    def test_duplicate_and_remove(self):
        self.builder.build({"name": "origapp", "stack": "static"},
                           verify=False)
        dup = self.builder.duplicate("origapp", "copyapp")
        self.assertEqual(dup["app"], "copyapp")
        self.assertTrue((Path(dup["dir"]) / "index.html").is_file())
        man = self.builder.info("copyapp")
        self.assertEqual(man["name"], "copyapp")
        removed = self.builder.remove("copyapp")
        self.assertTrue(removed["removed"])
        self.assertFalse(Path(dup["dir"]).exists())
        with self.assertRaises(ToolError):
            self.builder.remove("copyapp")


if __name__ == "__main__":
    unittest.main()
