"""Wave F2 builder tests: new templates + build -> zip -> deliver.

* Every template kind (old + new) scaffolds into a temp dir, passes its
  own test suite unmodified, and smoke-passes (serve+smoke for HTTP
  kinds, --help smoke for console/CLI kinds).
* zip_project round-trips and excludes junk dirs.
* deliver_project zips and sends through a fake gateway's send_file
  path; failures are captured, never raised.
* build_zip_and_deliver runs the whole pipeline end to end per kind.

All offline except localhost HTTP.  No pip installs, no real network.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.builders import (
    KINDS,
    build_zip_and_deliver,
    deliver_project,
    run_config,
    scaffold,
    serve,
    smoke_test,
    template_dir,
    zip_project,
)
from nomorals.core.errors import ToolError

NEW_KINDS = ("rest_api", "telegram_bot", "dashboard")
ALL_KINDS = tuple(KINDS)

HTTP_KINDS = {"webapp", "rest_api", "dashboard"}


class _FakeGateway:
    """Duck-typed stand-in for ChatGateway: records send_file calls."""

    def __init__(self, ok: bool = True, message_id: str = "mid-1",
                 error: str = "", explode: bool = False) -> None:
        self._ok = ok
        self._message_id = message_id
        self._error = error
        self._explode = explode
        self.calls: list[dict] = []

    def send_file(self, platform, chat, path, *, caption="",
                  max_send_mb=0.0):
        self.calls.append({"platform": platform, "chat": chat, "path": path,
                           "caption": caption, "max_send_mb": max_send_mb})
        if self._explode:
            raise RuntimeError("boom: transport down")
        return SimpleNamespace(ok=self._ok, message_id=self._message_id,
                               error=self._error)


class _FakeContext:
    """Agent-context stand-in: gateway lives in extras, like the runtime."""

    def __init__(self, gateway) -> None:
        self.extras = {"gateway": gateway}


class F2NewKindsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="builders-f2-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    # -- registry ---------------------------------------------------------
    def test_new_kinds_registered(self) -> None:
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                self.assertIn(kind, KINDS)
                self.assertTrue(template_dir(kind).is_dir())

    def test_unknown_kind_still_raises(self) -> None:
        with self.assertRaises(ToolError):
            scaffold("spaceship", "x", self.root)

    # -- scaffold invariants (mirror test_builders.py's contract) ---------
    def test_new_templates_have_no_leftover_placeholders(self) -> None:
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                result = scaffold(kind, f"inv_{kind}", self.root)
                for rel in result.files:
                    path = result.project_dir / rel
                    try:
                        text = path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        continue
                    self.assertNotIn("$PROJECT_NAME", text, rel)
                    if text.strip():
                        self.assertIn(f"inv_{kind}", text, rel)

    # -- rendered projects pass their own suites ---------------------------
    def test_rendered_projects_pass_own_tests(self) -> None:
        for kind in ALL_KINDS:
            with self.subTest(kind=kind):
                result = scaffold(kind, f"t_{kind}", self.root)
                proc = subprocess.run(
                    result.test_cmd, cwd=str(result.project_dir),
                    capture_output=True, text=True, timeout=120)
                self.assertEqual(
                    proc.returncode, 0,
                    f"{kind} template tests failed:\n{proc.stdout}\n{proc.stderr}")

    # -- run_config kinds ---------------------------------------------------
    def test_run_config_kinds(self) -> None:
        expected = {"rest_api": "http", "dashboard": "http",
                    "telegram_bot": "console"}
        for kind, want in expected.items():
            with self.subTest(kind=kind):
                result = scaffold(kind, f"cfg_{kind}", self.root)
                config = run_config(result.project_dir)
                self.assertEqual(config.kind, want)
                self.assertEqual(config.command, [sys.executable, "run.py"])

    # -- smoke --------------------------------------------------------------
    def test_smoke_all_kinds(self) -> None:
        for kind in ALL_KINDS:
            with self.subTest(kind=kind):
                result = scaffold(kind, f"smk_{kind}", self.root)
                config = run_config(result.project_dir)
                if config.kind == "http":
                    with serve(result.project_dir, port=0,
                               startup_timeout=10) as handle:
                        smoke = smoke_test(handle, timeout=10)
                else:
                    smoke = smoke_test(result.project_dir, timeout=30)
                self.assertTrue(smoke.ok, f"{kind}: {smoke.to_dict()}")

    # -- telegram_bot specifics ----------------------------------------------
    def test_telegram_bot_missing_token_fails_fast(self) -> None:
        result = scaffold("telegram_bot", "tgbot_fail", self.root)
        run_py = result.project_dir / "run.py"
        env = {k: v for k, v in os.environ.items() if k != "BOT_TOKEN"}
        proc = subprocess.run(
            [sys.executable, str(run_py), "--once"],
            env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30,
            cwd=str(result.project_dir))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("BOT_TOKEN", proc.stderr)

    def test_telegram_bot_self_test_offline(self) -> None:
        result = scaffold("telegram_bot", "tgbot_self", self.root)
        run_py = result.project_dir / "run.py"
        env = {k: v for k, v in os.environ.items() if k != "BOT_TOKEN"}
        proc = subprocess.run(
            [sys.executable, str(run_py), "--self-test"],
            env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30,
            cwd=str(result.project_dir))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OK", proc.stdout)


class F2ZipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="builders-f2zip-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_zip_round_trip(self) -> None:
        result = scaffold("rest_api", "zipapp", self.root)
        # junk that must be excluded
        (result.project_dir / ".git").mkdir()
        (result.project_dir / ".git" / "HEAD").write_text("ref: x")
        pycache = result.project_dir / "__pycache__"
        pycache.mkdir()
        (pycache / "run.cpython-312.pyc").write_bytes(b"\x00" * 16)

        zipped = zip_project(result.project_dir, dest=self.root)
        self.assertTrue(zipped.archive.is_file())
        self.assertEqual(zipped.archive.suffix, ".zip")
        self.assertGreater(zipped.bytes, 0)

        with zipfile.ZipFile(zipped.archive) as zf:
            names = zf.namelist()
        self.assertFalse(any(".git" in n for n in names))
        self.assertFalse(any("__pycache__" in n for n in names))
        self.assertTrue(all(n.startswith("zipapp/") for n in names))
        self.assertIn("zipapp/run.py", names)
        self.assertIn("zipapp/README.md", names)
        # every non-excluded project file landed in the archive
        expected = {f for f in zipped.files}
        archived = {n[len("zipapp/"):] for n in names}
        self.assertEqual(expected, archived)

    def test_zip_is_deflated(self) -> None:
        result = scaffold("dashboard", "zipdash", self.root)
        zipped = zip_project(result.project_dir, dest=self.root)
        with zipfile.ZipFile(zipped.archive) as zf:
            methods = {i.compress_type for i in zf.infolist()}
        self.assertEqual(methods, {zipfile.ZIP_DEFLATED})

    def test_zip_refuses_existing_archive(self) -> None:
        result = scaffold("bot", "zipdup", self.root)
        zip_project(result.project_dir, dest=self.root)
        with self.assertRaises(ToolError):
            zip_project(result.project_dir, dest=self.root)

    def test_zip_rejects_bad_project_dir(self) -> None:
        with self.assertRaises(ToolError):
            zip_project(self.root / "nope", dest=self.root)
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ToolError):
            zip_project(empty, dest=self.root)


class F2DeliverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="builders-f2del-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.result = scaffold("cli_tool", "deliverapp", self.root)

    def test_deliver_sends_zip_through_gateway(self) -> None:
        gw = _FakeGateway()
        delivered = deliver_project(
            self.result.project_dir, platform="telegram", chat="@owner",
            gateway=gw, dest=self.root)
        self.assertTrue(delivered.ok, delivered.to_dict())
        self.assertEqual(delivered.message_id, "mid-1")
        self.assertTrue(delivered.zip.archive.is_file())
        self.assertEqual(len(gw.calls), 1)
        call = gw.calls[0]
        self.assertEqual(call["platform"], "telegram")
        self.assertEqual(call["chat"], "@owner")
        self.assertEqual(call["path"], str(delivered.zip.archive))
        self.assertIn("deliverapp", call["caption"])

    def test_deliver_resolves_gateway_from_context_extras(self) -> None:
        gw = _FakeGateway(message_id="ctx-mid")
        delivered = deliver_project(
            self.result.project_dir, platform="telegram", chat="@owner",
            context=_FakeContext(gw), dest=self.root)
        self.assertTrue(delivered.ok)
        self.assertEqual(delivered.message_id, "ctx-mid")
        self.assertEqual(len(gw.calls), 1)

    def test_deliver_prefers_explicit_gateway(self) -> None:
        gw_explicit = _FakeGateway(message_id="explicit")
        gw_ctx = _FakeGateway(message_id="ctx")
        delivered = deliver_project(
            self.result.project_dir, platform="telegram", chat="@owner",
            gateway=gw_explicit, context=_FakeContext(gw_ctx),
            dest=self.root)
        self.assertTrue(delivered.ok)
        self.assertEqual(delivered.message_id, "explicit")

    def test_deliver_requires_a_gateway(self) -> None:
        with self.assertRaises(ToolError):
            deliver_project(self.result.project_dir, platform="telegram",
                            chat="@owner", dest=self.root)

    def test_deliver_rejects_gateway_without_send_file(self) -> None:
        with self.assertRaises(ToolError):
            deliver_project(self.result.project_dir, platform="telegram",
                            chat="@owner", gateway=object(), dest=self.root)

    def test_deliver_captures_send_failure(self) -> None:
        gw = _FakeGateway(ok=False, error="chat not found")
        delivered = deliver_project(
            self.result.project_dir, platform="telegram", chat="@ghost",
            gateway=gw, dest=self.root)
        self.assertFalse(delivered.ok)
        self.assertTrue(any("chat not found" in p for p in delivered.problems),
                        delivered.problems)
        # the zip was still produced even though the send failed
        self.assertTrue(delivered.zip.archive.is_file())

    def test_deliver_captures_send_exception(self) -> None:
        gw = _FakeGateway(explode=True)
        delivered = deliver_project(
            self.result.project_dir, platform="telegram", chat="@owner",
            gateway=gw, dest=self.root)
        self.assertFalse(delivered.ok)
        self.assertTrue(any("RuntimeError" in p for p in delivered.problems),
                        delivered.problems)


class F2EndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="builders-f2e2e-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_build_zip_and_deliver_all_kinds(self) -> None:
        for kind in ALL_KINDS:
            with self.subTest(kind=kind):
                gw = _FakeGateway(message_id=f"mid-{kind}")
                report = build_zip_and_deliver(
                    kind, f"e2e_{kind}", self.root,
                    platform="telegram", chat="@owner", gateway=gw,
                    startup_timeout=10)
                self.assertTrue(report.ok, report.summary())
                self.assertEqual(
                    [s.name for s in report.steps],
                    ["scaffold", "tests",
                     "serve+smoke" if kind in HTTP_KINDS else "smoke",
                     "zip", "deliver"])
                self.assertTrue(all(s.ok for s in report.steps))
                self.assertTrue(report.zip_path.is_file())
                self.assertEqual(report.message_id, f"mid-{kind}")
                self.assertEqual(len(gw.calls), 1)
                self.assertEqual(gw.calls[0]["path"], str(report.zip_path))

    def test_end_to_end_captures_failed_send(self) -> None:
        gw = _FakeGateway(ok=False, error="flood control")
        report = build_zip_and_deliver(
            "webapp", "e2e_fail", self.root,
            platform="telegram", chat="@owner", gateway=gw,
            startup_timeout=10)
        self.assertFalse(report.ok)
        self.assertEqual(report.broken, ["deliver"])
        # everything before the send still passed
        for step in report.steps[:-1]:
            self.assertTrue(step.ok, step.name)
        self.assertTrue(report.zip_path.is_file())

    def test_end_to_end_captures_scaffold_failure(self) -> None:
        gw = _FakeGateway()
        first = build_zip_and_deliver(
            "bot", "e2e_dup", self.root,
            platform="telegram", chat="@owner", gateway=gw,
            startup_timeout=10)
        self.assertTrue(first.ok)
        second = build_zip_and_deliver(
            "bot", "e2e_dup", self.root,
            platform="telegram", chat="@owner", gateway=gw,
            startup_timeout=10)
        self.assertFalse(second.ok)
        self.assertEqual(second.steps[0].name, "scaffold")
        self.assertFalse(second.steps[0].ok)
        self.assertEqual(len(gw.calls), 1)  # failed build sent nothing


if __name__ == "__main__":
    unittest.main()
