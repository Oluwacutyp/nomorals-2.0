"""Worker-C builders tests: verify.py wired into the pipelines.

* ``build_and_verify`` is the canonical post-build verification
  primitive: ``build_zip_and_deliver`` delegates its scaffold/install/
  tests/smoke/export steps to it (no duplicated pipeline logic), and
  forwards ``policy``/``confirmation`` to the install step.
* ``AppBuilder.build(verify=True)`` runtime-verifies the built app
  (serve + health for server stacks, --help smoke for cli-python,
  real go build for go-cli) and records honest ``skipped`` entries
  for stacks that cannot be verified offline.
* ``nm build`` (kinds | verify | deliver) works end to end, and
  ``nm apps build`` passes --verify/--no-verify through to the
  build_app tool.

All offline except localhost HTTP.  No pip installs, no real network.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from nomorals.builders import (
    KINDS,
    AppBuilder,
    build_and_verify,
    build_zip_and_deliver,
)
from nomorals.cmdline.commands.builders import _cmd_build


class _FakeGateway:
    """Duck-typed ChatGateway stand-in: records send_file calls."""

    def __init__(self, ok: bool = True, message_id: str = "mid-1",
                 error: str = "") -> None:
        self._ok = ok
        self._message_id = message_id
        self._error = error
        self.calls: list[dict] = []

    def send_file(self, platform, chat, path, *, caption="",
                  max_send_mb=0.0):
        self.calls.append({"platform": platform, "chat": chat,
                           "path": path, "caption": caption,
                           "max_send_mb": max_send_mb})
        result = SimpleNamespace(ok=self._ok, message_id=self._message_id,
                                 error=self._error)
        return result


def _ns(**kwargs) -> argparse.Namespace:
    base = {"json": False, "build_action": "kinds", "kind": "",
            "name": "", "dest": "", "export_dir": "",
            "startup_timeout": 10.0, "to": "", "platform": "",
            "caption": ""}
    base.update(kwargs)
    return argparse.Namespace(**base)


class VerifyWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="nm-bv-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    # -- build_zip_and_deliver delegates to build_and_verify ----------------
    def test_deliver_pipeline_uses_verify_steps(self) -> None:
        gw = _FakeGateway(message_id="mid-c1")
        report = build_zip_and_deliver(
            "cli_tool", "del_cli", self.root,
            platform="telegram", chat="@owner", gateway=gw)
        self.assertTrue(report.ok, report.summary())
        self.assertEqual(
            [s.name for s in report.steps],
            ["scaffold", "install_deps", "tests", "smoke",
             "export", "zip", "deliver"])
        self.assertTrue(all(s.ok for s in report.steps))
        # verify's tamper-evident tar.gz export exists alongside the zip
        export_step = next(s for s in report.steps if s.name == "export")
        self.assertIn(".tar.gz", export_step.detail)
        self.assertTrue(report.zip_path.is_file())
        self.assertEqual(report.message_id, "mid-c1")

    def test_deliver_pipeline_skips_zip_and_send_when_verify_fails(self) -> None:
        gw = _FakeGateway()
        first = build_zip_and_deliver(
            "cli_tool", "del_dup", self.root,
            platform="telegram", chat="@owner", gateway=gw)
        self.assertTrue(first.ok, first.summary())
        second = build_zip_and_deliver(
            "cli_tool", "del_dup", self.root,
            platform="telegram", chat="@owner", gateway=gw)
        self.assertFalse(second.ok)
        self.assertEqual(second.steps[0].name, "scaffold")
        self.assertFalse(second.steps[0].ok)
        # a broken build is never zipped or delivered
        self.assertEqual([s.name for s in second.steps], ["scaffold"])
        self.assertIsNone(second.zip_path)
        self.assertEqual(second.message_id, "")
        self.assertEqual(len(gw.calls), 1)  # only the first (good) run sent

    def test_build_and_verify_forwards_confirmation(self) -> None:
        import nomorals.builders.verify as verify_mod

        calls: dict = {}
        real = verify_mod.install_deps

        def fake(project_dir, *, policy=None, confirmation=None, **kw):
            calls["confirmation"] = confirmation
            return real(project_dir, policy=policy,
                        confirmation=confirmation, **kw)

        verify_mod.install_deps = fake  # type: ignore[method-assign]
        try:
            report = build_and_verify("cli_tool", "bv_conf", self.root,
                                      confirmation="tok-123")
        finally:
            verify_mod.install_deps = real  # type: ignore[method-assign]
        self.assertTrue(report.ok, report.summary())
        self.assertEqual(calls.get("confirmation"), "tok-123")


class AppBuilderRuntimeVerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp(prefix="nm-abv-"))
        self.addCleanup(shutil.rmtree, self.home, True)
        ctx = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=str(self.home)),
            db=None,
        )
        self.builder = AppBuilder(ctx)

    def _build(self, name: str, stack: str,
               verify: bool = True) -> dict:
        return self.builder.build(
            {"name": name, "stack": stack, "title": name, "features": []},
            verify=verify)

    def test_static_build_serves_and_health_checks(self) -> None:
        out = self._build("vstatic", "static")
        v = out["validation"]
        self.assertTrue(v["ok"], v)
        runtime = v.get("runtime") or {}
        self.assertTrue(runtime.get("ok"), runtime)
        self.assertFalse(runtime.get("skipped"))
        self.assertIn("200", runtime.get("detail", ""))
        # no server left behind in the served registry
        self.assertEqual(self.builder.served()["count"], 0)

    def test_cli_python_build_help_smoke(self) -> None:
        out = self._build("vcli", "cli-python")
        v = out["validation"]
        self.assertTrue(v["ok"], v)
        runtime = v.get("runtime") or {}
        self.assertTrue(runtime.get("ok"), runtime)
        self.assertIn("--help", runtime.get("detail", ""))

    def test_flask_build_skips_honestly_without_deps(self) -> None:
        out = self._build("vflask", "flask")
        v = out["validation"]
        self.assertTrue(v["ok"], v)  # static validation passed
        runtime = v.get("runtime") or {}
        self.assertTrue(runtime.get("skipped"))
        self.assertIn("flask", runtime.get("detail", ""))

    def test_bot_telegram_build_skips_honestly(self) -> None:
        out = self._build("vbot", "bot-telegram")
        v = out["validation"]
        self.assertTrue(v["ok"], v)
        runtime = v.get("runtime") or {}
        self.assertTrue(runtime.get("skipped"))
        self.assertIn("BOT_TOKEN", runtime.get("detail", ""))

    def test_go_cli_build_skips_without_toolchain(self) -> None:
        if shutil.which("go"):
            self.skipTest("go toolchain present; skip the skip-path test")
        out = self._build("vgo", "go-cli")
        runtime = (out["validation"].get("runtime") or {})
        self.assertTrue(runtime.get("skipped"))

    def test_verify_false_skips_runtime(self) -> None:
        out = self._build("vnrt", "static", verify=False)
        v = out["validation"]
        self.assertTrue(v["ok"], v)
        self.assertNotIn("runtime", v)


class NmBuildCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from nomorals.agents.context import build_context
        from nomorals.core.config import Settings

        cls.home = tempfile.mkdtemp(prefix="nm-bcli-")
        cls._ctx_mgr = build_context(Settings(home=cls.home))
        cls.ctx = cls._ctx_mgr.__enter__()
        cls.addClassCleanup(cls._ctx_mgr.__exit__, None, None, None)
        cls.addClassCleanup(shutil.rmtree, cls.home, True)

    def _run(self, **kwargs):
        args = _ns(**kwargs)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_build(args, self.ctx)
        return rc, buf.getvalue()

    def test_kinds(self) -> None:
        rc, out = self._run(build_action="kinds")
        self.assertEqual(rc, 0)
        for kind in KINDS:
            self.assertIn(kind, out)

    def test_verify_cli_tool_end_to_end(self) -> None:
        dest = tempfile.mkdtemp(prefix="nm-bvcli-")
        self.addCleanup(shutil.rmtree, dest, True)
        rc, out = self._run(build_action="verify", kind="cli_tool",
                            name="cb_cli", dest=dest)
        self.assertEqual(rc, 0, out)
        self.assertIn("OK", out)
        self.assertIn("smoke", out)

    def test_verify_broken_build_exits_1(self) -> None:
        dest = tempfile.mkdtemp(prefix="nm-bvbrk-")
        self.addCleanup(shutil.rmtree, dest, True)
        (Path(dest) / "cb_dup").mkdir()  # force a scaffold failure
        (Path(dest) / "cb_dup" / "x.txt").write_text("x")
        rc, out = self._run(build_action="verify", kind="cli_tool",
                            name="cb_dup", dest=dest)
        self.assertEqual(rc, 1)
        self.assertIn("BROKEN", out)

    def test_deliver_end_to_end_with_gateway(self) -> None:
        gw = _FakeGateway(message_id="mid-cli")
        self.ctx.extras["gateway"] = gw
        try:
            dest = tempfile.mkdtemp(prefix="nm-bdlv-")
            self.addCleanup(shutil.rmtree, dest, True)
            rc, out = self._run(build_action="deliver", kind="cli_tool",
                                name="cb_del", dest=dest,
                                to="telegram:123456")
        finally:
            self.ctx.extras.pop("gateway", None)
        self.assertEqual(rc, 0, out)
        self.assertEqual(gw.calls[0]["platform"], "telegram")
        self.assertEqual(gw.calls[0]["chat"], "123456")

    def test_deliver_without_gateway_fails_honestly(self) -> None:
        self.ctx.extras.pop("gateway", None)
        dest = tempfile.mkdtemp(prefix="nm-bnog-")
        self.addCleanup(shutil.rmtree, dest, True)
        rc, out = self._run(build_action="deliver", kind="cli_tool",
                            name="cb_nogw", dest=dest,
                            to="telegram:123456")
        self.assertEqual(rc, 1)
        self.assertIn("deliver", out)
        self.assertIn("archive kept at:", out)

    def test_apps_build_tool_runtime_verifies(self) -> None:
        out = self.ctx.tools.call(
            "build_app", action="build", name="nmcli_static",
            stack="static", title="cli-built", verify=True)
        self.assertTrue(out.ok, getattr(out, "error", ""))
        v = (out.value or {}).get("validation") or {}
        self.assertTrue(v.get("ok"), v)
        runtime = v.get("runtime") or {}
        self.assertTrue(runtime.get("ok"), runtime)

    def test_apps_build_tool_verify_false(self) -> None:
        out = self.ctx.tools.call(
            "build_app", action="build", name="nmcli_noverify",
            stack="static", title="no verify", verify=False)
        self.assertTrue(out.ok, getattr(out, "error", ""))
        v = (out.value or {}).get("validation") or {}
        self.assertNotIn("runtime", v)


class ParserRegistrationTests(unittest.TestCase):
    def test_build_command_parses(self) -> None:
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        ns = parser.parse_args(["build", "kinds"])
        self.assertEqual(ns.command, "build")
        self.assertEqual(ns.build_action, "kinds")
        ns = parser.parse_args(
            ["build", "verify", "cli_tool", "myapp", "--dest", "/tmp/x"])
        self.assertEqual(ns.kind, "cli_tool")
        self.assertEqual(ns.name, "myapp")
        self.assertEqual(ns.dest, "/tmp/x")
        ns = parser.parse_args(
            ["build", "deliver", "webapp", "w1", "--to", "telegram:1"])
        self.assertEqual(ns.build_action, "deliver")
        self.assertEqual(ns.to, "telegram:1")
        ns = parser.parse_args(["apps", "build", "a1", "--no-verify"])
        self.assertFalse(ns.verify)


if __name__ == "__main__":
    unittest.main()
