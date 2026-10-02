"""Wave G3 — deliver_report failure hardening.

Covers the failure paths ``nomorals/tools/deliver_report.py`` must get
right:

1. BAD CHAT ID: the destination is validated BEFORE anything is
   generated — a malformed chat id raises with a clear, specific error
   and no PDF/HTML is rendered (the renderer is mocked and asserted
   not called).
2. SEND FAILURE: when the send edge dies after the artifact is built,
   the error reports the bytes built, the exact local zip path, and the
   resend command — nothing built is silently lost.
3. EMPTY TOPIC: empty/whitespace-only topics are rejected before
   generation.
4. CAPS: oversized topics/sections are truncated with a clear warning,
   never silently.
5. RESEND: ``resend_report`` / ``nm deliver send <zip> --to platform:chat``
   — the recovery path — validates the destination first and refuses
   missing/non-zip files.
"""

from __future__ import annotations

import argparse
import io
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from nomorals.core.errors import ToolError
from nomorals.tools import deliver_report as DR


def _context(**over: Any) -> SimpleNamespace:
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()),
        extras={},
        router=None,
    )
    for key, value in over.items():
        setattr(ctx, key, value)
    return ctx


class FakeGateway:
    """Gateway whose live adapters are exactly telegram+whatsapp."""

    def __init__(self) -> None:
        self.adapters = {"telegram": object(), "whatsapp": object()}
        self.sent: list[dict[str, Any]] = []

    def send_file(self, platform: str, chat_id: str, path: str,
                  caption: str = "") -> SimpleNamespace:
        self.sent.append({"platform": platform, "chat": chat_id,
                          "path": path, "caption": caption})
        return SimpleNamespace(ok=True, message_id="m-resend-1")


# ── 1. destination validation ─────────────────────────────────────────────


class ValidateDestinationTests(unittest.TestCase):
    def test_valid_destinations_pass(self) -> None:
        self.assertEqual(DR.validate_destination("telegram", "123456"),
                         ("telegram", "123456"))
        self.assertEqual(DR.validate_destination("telegram-bot", "-987654"),
                         ("telegram-bot", "-987654"))
        self.assertEqual(DR.validate_destination("telegram", "@mychannel"),
                         ("telegram", "@mychannel"))
        self.assertEqual(DR.validate_destination("discord", "112233445566"),
                         ("discord", "112233445566"))
        self.assertEqual(DR.validate_destination("whatsapp", "+2348012345678"),
                         ("whatsapp", "+2348012345678"))
        self.assertEqual(DR.validate_destination("whatsapp", "2348012345678@c.us"),
                         ("whatsapp", "2348012345678@c.us"))
        self.assertEqual(DR.validate_destination("local", "owner-dm"),
                         ("local", "owner-dm"))

    def test_platform_is_case_insensitive(self) -> None:
        self.assertEqual(DR.validate_destination("Telegram", "123"),
                         ("telegram", "123"))

    def test_empty_platform_rejected(self) -> None:
        with self.assertRaisesRegex(ToolError, "platform"):
            DR.validate_destination("", "123")
        with self.assertRaisesRegex(ToolError, "platform"):
            DR.validate_destination("   ", "123")

    def test_unsupported_platform_rejected_with_supported_list(self) -> None:
        with self.assertRaisesRegex(ToolError, "unsupported platform.*gmail"):
            DR.validate_destination("gmail", "123")
        with self.assertRaisesRegex(ToolError, "telegram"):
            DR.validate_destination("sms", "123")

    def test_empty_chat_id_rejected(self) -> None:
        with self.assertRaisesRegex(ToolError, "chat id"):
            DR.validate_destination("telegram", "")
        with self.assertRaisesRegex(ToolError, "chat id"):
            DR.validate_destination("telegram", "   ")

    def test_non_numeric_telegram_id_rejected(self) -> None:
        with self.assertRaisesRegex(ToolError, "bad chat id.*telegram.*abc"):
            DR.validate_destination("telegram", "abc")

    def test_non_numeric_discord_id_rejected(self) -> None:
        with self.assertRaisesRegex(ToolError, "bad chat id.*discord"):
            DR.validate_destination("discord", "general")

    def test_non_numeric_whatsapp_id_rejected(self) -> None:
        with self.assertRaisesRegex(ToolError, "bad chat id.*whatsapp"):
            DR.validate_destination("whatsapp", "mom")

    def test_chat_id_with_colon_rejected(self) -> None:
        with self.assertRaisesRegex(ToolError, "only once"):
            DR.validate_destination("telegram", "123:456")

    def test_chat_id_with_whitespace_rejected(self) -> None:
        with self.assertRaisesRegex(ToolError, "whitespace"):
            DR.validate_destination("telegram", "12 34")

    def test_live_adapter_checked_before_generation(self) -> None:
        # whatsapp is supported but NOT live on this gateway → fail fast
        gw = SimpleNamespace(adapters={"telegram": object()})
        ctx = _context(extras={"gateway": gw})
        with self.assertRaisesRegex(ToolError, "no live adapter.*whatsapp"):
            DR.validate_destination("whatsapp", "+2348012345678", ctx)
        # telegram IS live → passes
        self.assertEqual(
            DR.validate_destination("telegram", "123", ctx), ("telegram", "123"))


class BadChatIdPreGenerationTests(unittest.TestCase):
    def test_bad_chat_id_raises_before_generate_is_called(self) -> None:
        ctx = _context()
        spy = Mock(side_effect=AssertionError(
            "generate_report must not run with a bad chat id"))
        with patch.object(DR, "generate_report", spy):
            with self.assertRaisesRegex(ToolError, "bad chat id.*telegram"):
                DR.deliver_report(ctx, "T", [("A", "body")], "telegram",
                                  "not-a-number")
        spy.assert_not_called()

    def test_unsupported_platform_raises_before_generate_is_called(self) -> None:
        ctx = _context()
        spy = Mock(side_effect=AssertionError(
            "generate_report must not run with a bad platform"))
        with patch.object(DR, "generate_report", spy):
            with self.assertRaisesRegex(ToolError, "unsupported platform"):
                DR.deliver_report(ctx, "T", [("A", "body")], "gmail", "123")
        spy.assert_not_called()

    def test_empty_topic_rejected_before_any_render(self) -> None:
        ctx = _context()
        renderer = Mock()
        with patch.object(DR, "render_report_html", renderer):
            for topic in ("", "   "):
                with self.assertRaisesRegex(ToolError, "topic"):
                    DR.deliver_report(ctx, topic, [("A", "body")],
                                      "telegram", "123")
        renderer.assert_not_called()
        # and no report directory was created either
        self.assertFalse(
            any(Path(ctx.settings.workspace_dir).iterdir()),
            "empty topic must not create anything on disk")

    def test_empty_sections_rejected_before_any_render(self) -> None:
        ctx = _context()
        renderer = Mock()
        with patch.object(DR, "render_report_html", renderer):
            with self.assertRaisesRegex(ToolError, "section"):
                DR.deliver_report(ctx, "T", [], "telegram", "123")
        renderer.assert_not_called()


# ── 2. send failure keeps the artifact + names the resend path ────────────


def _fake_raising(*a: Any, **k: Any) -> dict[str, Any]:
    raise RuntimeError("gateway down")


class SendFailureTests(unittest.TestCase):
    def test_send_exception_error_names_bytes_path_and_resend(self) -> None:
        ctx = _context()
        with self.assertRaises(ToolError) as cm:
            DR.deliver_report(ctx, "T", [("A", "body")], "telegram", "123",
                              sender=_fake_raising)
        msg = str(cm.exception)
        self.assertIn("send failed", msg)
        self.assertIn(".zip", msg)
        self.assertIn("B", msg)  # byte count
        self.assertIn("nm deliver send", msg)
        self.assertIn("--to telegram:123", msg)

    def test_send_reported_failure_error_names_resend_path(self) -> None:
        ctx = _context()
        with self.assertRaises(ToolError) as cm:
            DR.deliver_report(ctx, "T", [("A", "body")], "telegram", "123",
                              sender=lambda *a, **k: {"sent": False})
        msg = str(cm.exception)
        self.assertIn("nm deliver send", msg)
        self.assertIn(".zip", msg)

    def test_built_zip_survives_the_failed_send(self) -> None:
        ctx = _context()
        try:
            DR.deliver_report(ctx, "T", [("A", "body")], "telegram", "123",
                              sender=_fake_raising)
        except ToolError as exc:
            msg = str(exc)
        # the error names a path that still exists on disk
        zip_path = next(
            tok for tok in msg.replace("—", " ").split()
            if tok.endswith(".zip"))
        self.assertTrue(Path(zip_path).is_file(),
                        f"built zip must survive: {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            self.assertTrue(any(n.endswith("report.html") for n in zf.namelist()))


# ── 4. caps ────────────────────────────────────────────────────────────────


class CapsTests(unittest.TestCase):
    def test_oversized_topic_capped_with_warning(self) -> None:
        ctx = _context()
        bundle = DR.generate_report(ctx, "T" * 500, [("A", "body")])
        self.assertLessEqual(len(bundle.topic), DR.MAX_TOPIC_CHARS)
        self.assertTrue(any("topic capped" in w for w in bundle.warnings),
                        bundle.warnings)
        self.assertTrue(bundle.warnings, "cap must carry a clear message")

    def test_oversized_section_body_capped_with_warning_and_marker(self) -> None:
        ctx = _context()
        body = "x" * (DR.MAX_SECTION_BODY_CHARS + 500)
        bundle = DR.generate_report(ctx, "T", [("A", body)])
        self.assertTrue(any("body capped" in w for w in bundle.warnings),
                        bundle.warnings)
        html = Path(bundle.html_path).read_text(encoding="utf-8")
        self.assertIn("truncated at", html)

    def test_oversized_section_title_capped_with_warning(self) -> None:
        ctx = _context()
        bundle = DR.generate_report(ctx, "T", [("T" * 500, "body")])
        self.assertLessEqual(len(bundle.sections[0]), DR.MAX_SECTION_TITLE_CHARS)
        self.assertTrue(any("title" in w and "capped" in w
                            for w in bundle.warnings), bundle.warnings)

    def test_normal_sizes_produce_no_warnings(self) -> None:
        ctx = _context()
        bundle = DR.generate_report(ctx, "T", [("A", "body")])
        self.assertEqual(bundle.warnings, [])


# ── 5. resend ─────────────────────────────────────────────────────────────


class ResendTests(unittest.TestCase):
    def _built_zip(self, ctx: SimpleNamespace) -> str:
        bundle = DR.generate_report(ctx, "T", [("A", "body")])
        return bundle.zip_path

    def test_resend_delivers_the_built_zip(self) -> None:
        ctx = _context()
        zip_path = self._built_zip(ctx)
        seen: dict[str, Any] = {}

        def fake_sender(context: Any, platform: str, chat_id: str,
                        path: str, caption: str = "") -> dict[str, Any]:
            seen.update(platform=platform, chat=chat_id, path=path,
                        caption=caption, data=Path(path).read_bytes())
            return {"sent": True, "message_id": "m-resent-1"}

        out = DR.resend_report(ctx, zip_path, "telegram", "123",
                               sender=fake_sender)
        self.assertTrue(out["ok"])
        self.assertTrue(out["resent"])
        self.assertEqual(out["platform"], "telegram")
        self.assertEqual(out["chat_id"], "123")
        self.assertEqual(out["message_id"], "m-resent-1")
        self.assertEqual(seen["data"], Path(zip_path).read_bytes())
        self.assertGreater(out["zip_bytes"], 0)

    def test_resend_validates_destination_first(self) -> None:
        ctx = _context()
        zip_path = self._built_zip(ctx)
        spy = Mock()
        with self.assertRaisesRegex(ToolError, "bad chat id"):
            DR.resend_report(ctx, zip_path, "telegram", "not-a-number",
                              sender=spy)
        spy.assert_not_called()

    def test_resend_rejects_missing_zip(self) -> None:
        ctx = _context()
        with self.assertRaisesRegex(ToolError, "no such report zip"):
            DR.resend_report(ctx, "/tmp/does-not-exist-r.zip", "telegram",
                             "123", sender=lambda *a, **k: {})

    def test_resend_rejects_non_zip(self) -> None:
        ctx = _context()
        not_zip = Path(ctx.settings.workspace_dir) / "note.txt"
        not_zip.write_text("hello", encoding="utf-8")
        with self.assertRaisesRegex(ToolError, "not a zip archive"):
            DR.resend_report(ctx, str(not_zip), "telegram", "123",
                             sender=lambda *a, **k: {})

    def test_resend_failure_names_the_intact_zip(self) -> None:
        ctx = _context()
        zip_path = self._built_zip(ctx)
        with self.assertRaises(ToolError) as cm:
            DR.resend_report(ctx, zip_path, "telegram", "123",
                             sender=_fake_raising)
        msg = str(cm.exception)
        self.assertIn(zip_path, msg)
        self.assertIn("intact", msg)
        self.assertTrue(Path(zip_path).is_file())

    def test_resend_registered_in_builtins(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()),
            extras={}, router=None))
        reg.register_builtins()
        self.assertIsNotNone(reg.get("resend_report"))


# ── CLI: nm deliver send ───────────────────────────────────────────────────


class CliSendTests(unittest.TestCase):
    def test_parser_accepts_deliver_send(self) -> None:
        from nomorals.cli import _parser

        args = _parser().parse_args(
            ["deliver", "send", "/tmp/r.zip", "--to", "telegram:123"])
        self.assertEqual(args.command, "deliver")
        self.assertEqual(args.deliver_action, "send")
        self.assertEqual(args.path, "/tmp/r.zip")
        self.assertEqual(args.to, "telegram:123")

    def test_cmd_deliver_send_resends(self) -> None:
        from nomorals.cli import _cmd_deliver

        ctx = _context()
        captured: dict[str, Any] = {}

        def fake_resend(context: Any, path: str, platform: str, chat: str,
                        **kw: Any) -> dict[str, Any]:
            captured.update(path=path, platform=platform, chat=chat)
            return {"ok": True, "resent": True, "platform": platform,
                    "chat_id": chat, "zip_path": path, "zip_bytes": 42,
                    "message_id": "m9"}

        args = argparse.Namespace(
            deliver_action="send", path="/tmp/r.zip", to="telegram:123",
            platform="", caption="", json=False)
        with patch("nomorals.tools.deliver_report.resend_report",
                   side_effect=fake_resend):
            # _cmd_deliver imports resend_report at call time, so this patch lands
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_deliver(args, ctx)
        self.assertEqual(rc, 0)
        self.assertEqual(captured["platform"], "telegram")
        self.assertEqual(captured["chat"], "123")
        self.assertIn("resent", buf.getvalue())

    def test_cmd_deliver_send_needs_destination(self) -> None:
        from nomorals.cli import _cmd_deliver

        ctx = _context()
        args = argparse.Namespace(
            deliver_action="send", path="/tmp/r.zip", to="", platform="",
            caption="", json=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_deliver(args, ctx)
        self.assertEqual(rc, 2)

    def test_cmd_deliver_send_bad_chat_id_fails_with_clear_error(self) -> None:
        from nomorals.cli import _cmd_deliver

        ctx = _context()
        args = argparse.Namespace(
            deliver_action="send", path="/tmp/r.zip", to="telegram:abc",
            platform="", caption="", json=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_deliver(args, ctx)
        self.assertEqual(rc, 1)
        self.assertIn("bad chat id", buf.getvalue())

    def test_cmd_deliver_report_bad_destination_fails_before_build(self) -> None:
        from nomorals.cli import _cmd_deliver

        ctx = _context()
        args = argparse.Namespace(
            deliver_action="report", topic="T", title="",
            section=["A::body"], to="gmail:123", platform="",
            no_pdf=True, json=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_deliver(args, ctx)
        self.assertEqual(rc, 1)
        self.assertIn("unsupported platform", buf.getvalue())
        # nothing generated: no reports dir under the workspace
        self.assertFalse(
            (Path(ctx.settings.workspace_dir) / DR.REPORTS_DIR).exists(),
            "unsupported platform must not trigger generation")

    def test_help_deliver_lists_send(self) -> None:
        from nomorals.cli import _cli_subparsers

        deliver_p = _cli_subparsers().choices["deliver"]
        sub_action = next(
            a for a in deliver_p._actions  # noqa: SLF001 - argparse internals
            if isinstance(a, argparse._SubParsersAction))
        self.assertIn("send", sub_action.choices)


if __name__ == "__main__":
    unittest.main()
