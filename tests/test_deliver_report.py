"""Wave F3 — deliver: report generate → zip → send end-to-end.

Covers ``nomorals/tools/deliver_report.py``:

1. section normalization (dataclass / tuple / dict; honest failures);
2. styled HTML generation (title, sections, TOC anchors, escaped chrome);
3. generate_report (HTML + real pure-Python PDF → zip via archive.zip_create);
4. deliver_report with an injected sender — the zip bytes that would be
   sent are verified (zip contains the report; HTML carries the
   title/sections);
5. CLI parsing + ``_cmd_deliver`` (build-only, errors, ``nm help deliver``);
6. chat wiring (parse_control routes /deliver; _control_deliver runs the
   real pipeline against a fake gateway).
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
from unittest.mock import patch

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


SECTIONS = [
    ("Overview", "The quarter was **volatile** — see the *numbers*."),
    ("Risks", "- rate hikes\n- liquidity\n\n> watch the curve"),
]


class FakeGateway:
    """Minimal stand-in for the live chat gateway's file-send path."""

    def __init__(self) -> None:
        self.adapters = {"telegram": object(), "whatsapp": object()}
        self.sent: list[dict[str, Any]] = []

    def send_file(self, platform: str, chat_id: str, path: str,
                  caption: str = "") -> SimpleNamespace:
        self.sent.append({"platform": platform, "chat": chat_id,
                          "path": path, "caption": caption})
        return SimpleNamespace(ok=True, message_id="m-deliver-1")


class NormalizeSectionsTests(unittest.TestCase):
    def test_accepts_all_three_shapes(self) -> None:
        secs = DR.normalize_sections([
            DR.ReportSection(title="A", body="a-body"),
            ("B", "b-body"),
            {"title": "C", "body": "c-body"},
        ])
        self.assertEqual([s.title for s in secs], ["A", "B", "C"])
        self.assertEqual(secs[2].body, "c-body")

    def test_rejects_empty_list(self) -> None:
        with self.assertRaises(ToolError):
            DR.normalize_sections([])

    def test_rejects_all_empty_bodies(self) -> None:
        with self.assertRaises(ToolError):
            DR.normalize_sections([("A", ""), ("B", "   ")])

    def test_rejects_bad_shape(self) -> None:
        with self.assertRaises(ToolError):
            DR.normalize_sections([{"nope": 1}])  # type: ignore[list-item]

    def test_rejects_blank_title(self) -> None:
        with self.assertRaises(ToolError):
            DR.normalize_sections([("  ", "body")])


class ComposeMarkdownTests(unittest.TestCase):
    def test_markdown_has_title_and_sections(self) -> None:
        md = DR.compose_markdown("Q3 markets", SECTIONS)
        self.assertIn("# Q3 markets", md)
        self.assertIn("## Overview", md)
        self.assertIn("## Risks", md)
        self.assertIn("**volatile**", md)

    def test_empty_topic_fails(self) -> None:
        with self.assertRaises(ToolError):
            DR.compose_markdown("", SECTIONS)


class RenderHtmlTests(unittest.TestCase):
    def test_html_carries_title_sections_and_toc(self) -> None:
        out = DR.render_report_html("Q3 markets", SECTIONS,
                                    generated="2026-10-01 12:00")
        self.assertIn("<title>Q3 markets</title>", out)
        self.assertIn("Q3 markets", out)
        self.assertIn("Overview", out)
        self.assertIn("<strong>volatile</strong>", out)
        # TOC with working anchors
        self.assertIn('class="report-toc"', out)
        self.assertIn('href="#overview"', out)
        self.assertIn('id="overview"', out)
        self.assertIn('href="#risks"', out)
        # styled chrome, not the plain renderer
        self.assertIn("report-cover", out)
        self.assertIn("report-footer", out)
        self.assertTrue(out.startswith("<!doctype html>"))

    def test_html_escapes_section_titles(self) -> None:
        out = DR.render_report_html("T", [("<script>alert(1)</script>", "x")])
        self.assertNotIn("<script>alert(1)</script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_empty_topic_fails(self) -> None:
        with self.assertRaises(ToolError):
            DR.render_report_html("", SECTIONS)


class GenerateReportTests(unittest.TestCase):
    def test_generate_writes_html_pdf_and_zip(self) -> None:
        ctx = _context()
        bundle = DR.generate_report(ctx, "Q3 markets", SECTIONS,
                                    generated="2026-10-01 12:00")
        self.assertEqual(bundle.topic, "Q3 markets")
        self.assertEqual(bundle.sections, ["Overview", "Risks"])
        html_path = Path(bundle.html_path)
        pdf_path = Path(bundle.pdf_path)
        self.assertTrue(html_path.is_file())
        self.assertTrue(pdf_path.is_file())
        self.assertGreater(bundle.html_bytes, 500)
        self.assertGreater(bundle.pdf_bytes, 500)

        # the zip holds both artifacts, under reports/<slug>/
        with zipfile.ZipFile(bundle.zip_path) as zf:
            names = zf.namelist()
        self.assertEqual(len(names), 2)
        self.assertTrue(any(n.endswith("report.html") for n in names))
        self.assertTrue(any(n.endswith("report.pdf") for n in names))
        self.assertEqual(sorted(bundle.zip_files), sorted(names))

        # PDF is a real PDF and carries the report text
        data = pdf_path.read_bytes()
        self.assertTrue(data.startswith(b"%PDF-"))
        from nomorals.core.pdf import read_pdf_text
        text = read_pdf_text(data)
        self.assertIn("Q3 markets", text)
        self.assertIn("Overview", text)
        self.assertIn("Risks", text)

        # HTML inside the zip has the title + sections
        with zipfile.ZipFile(bundle.zip_path) as zf:
            html_name = next(n for n in names if n.endswith("report.html"))
            html_text = zf.read(html_name).decode("utf-8")
        self.assertIn("Q3 markets", html_text)
        self.assertIn("Overview", html_text)

    def test_generate_without_pdf(self) -> None:
        ctx = _context()
        bundle = DR.generate_report(ctx, "T", [("A", "body")],
                                    include_pdf=False)
        self.assertEqual(bundle.pdf_path, "")
        self.assertEqual(bundle.pdf_bytes, 0)
        with zipfile.ZipFile(bundle.zip_path) as zf:
            names = zf.namelist()
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].endswith("report.html"))

    def test_repeat_generation_does_not_collide(self) -> None:
        ctx = _context()
        b1 = DR.generate_report(ctx, "Same topic", [("A", "x")])
        b2 = DR.generate_report(ctx, "Same topic", [("A", "x")])
        self.assertNotEqual(b1.zip_path, b2.zip_path)

    def test_empty_topic_and_sections_fail(self) -> None:
        ctx = _context()
        with self.assertRaises(ToolError):
            DR.generate_report(ctx, "", [("A", "x")])
        with self.assertRaises(ToolError):
            DR.generate_report(ctx, "T", [])


class DeliverReportTests(unittest.TestCase):
    def test_mocked_send_verifies_the_zip_bytes(self) -> None:
        ctx = _context()
        seen: dict[str, Any] = {}

        def fake_sender(context: Any, platform: str, chat_id: str, path: str,
                        caption: str = "") -> dict[str, Any]:
            data = Path(path).read_bytes()
            seen.update(platform=platform, chat=chat_id, path=path,
                        caption=caption, zip_bytes=data)
            return {"sent": True, "message_id": "m-mock-1"}

        out = DR.deliver_report(ctx, "Q3 markets", SECTIONS, "telegram",
                                "123", sender=fake_sender)
        self.assertTrue(out["ok"])
        self.assertEqual(out["platform"], "telegram")
        self.assertEqual(out["chat_id"], "123")
        self.assertEqual(out["message_id"], "m-mock-1")

        # the bytes the fake saw are the zip on disk — inspect them
        zip_bytes = seen["zip_bytes"]
        self.assertEqual(zip_bytes, Path(out["zip_path"]).read_bytes())
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            html_name = next(n for n in names if n.endswith("report.html"))
            html_text = zf.read(html_name).decode("utf-8")
        self.assertIn("Q3 markets", html_text)
        self.assertIn("Overview", html_text)
        self.assertIn("Risks", html_text)
        self.assertIn('class="report-toc"', html_text)

    def test_sender_with_small_signature_still_works(self) -> None:
        ctx = _context()
        calls: list[tuple] = []

        def tiny_sender(context: Any, platform: str, chat_id: str,
                        path: str) -> dict[str, Any]:
            calls.append((platform, chat_id, path))
            return {"sent": True, "message_id": "m-tiny"}

        out = DR.deliver_report(ctx, "T", [("A", "body")], "whatsapp",
                                "555", sender=tiny_sender)
        self.assertTrue(out["ok"])
        self.assertEqual(calls[0][:2], ("whatsapp", "555"))

    def test_send_failure_raises(self) -> None:
        ctx = _context()

        def bad_sender(*a: Any, **k: Any) -> dict[str, Any]:
            raise RuntimeError("gateway down")

        with self.assertRaises(ToolError):
            DR.deliver_report(ctx, "T", [("A", "x")], "telegram", "1",
                              sender=bad_sender)

    def test_send_reported_failure_raises(self) -> None:
        ctx = _context()
        with self.assertRaises(ToolError):
            DR.deliver_report(ctx, "T", [("A", "x")], "telegram", "1",
                              sender=lambda *a, **k: {"sent": False})

    def test_missing_platform_or_chat_fails(self) -> None:
        ctx = _context()
        with self.assertRaises(ToolError):
            DR.deliver_report(ctx, "T", [("A", "x")], "", "1",
                              sender=lambda *a, **k: {})
        with self.assertRaises(ToolError):
            DR.deliver_report(ctx, "T", [("A", "x")], "telegram", "",
                              sender=lambda *a, **k: {})

    def test_real_send_edge_with_fake_gateway(self) -> None:
        from nomorals.tools.filesend import send_file

        ctx = _context()
        ctx.extras = {"gateway": FakeGateway()}
        out = DR.deliver_report(ctx, "Q3 markets", SECTIONS, "telegram",
                                "123")
        self.assertTrue(out["ok"])
        gw = ctx.extras["gateway"]
        self.assertEqual(len(gw.sent), 1)
        self.assertEqual(gw.sent[0]["platform"], "telegram")
        self.assertEqual(gw.sent[0]["chat"], "telegram:123")
        self.assertEqual(gw.sent[0]["path"], out["zip_path"])


class CliTests(unittest.TestCase):
    def test_parser_accepts_deliver_report(self) -> None:
        from nomorals.cli import _parser

        args = _parser().parse_args([
            "deliver", "report", "Q3 markets",
            "--section", "Overview::The quarter was **volatile**",
            "--section", "Risks::- rate hikes",
            "--to", "telegram:123",
        ])
        self.assertEqual(args.command, "deliver")
        self.assertEqual(args.deliver_action, "report")
        self.assertEqual(args.topic, "Q3 markets")
        self.assertEqual(len(args.section), 2)
        self.assertEqual(args.to, "telegram:123")

    def test_alias_parses(self) -> None:
        from nomorals.cli import _parser

        args = _parser().parse_args(
            ["dlv", "report", "T", "--section", "A::b"])
        self.assertEqual(args.command, "dlv")

    def test_cmd_deliver_build_only(self) -> None:
        from nomorals.cli import _cmd_deliver

        ctx = _context()
        args = argparse.Namespace(
            deliver_action="report", topic="Q3 markets", title="",
            section=["Overview::body here"], to="", platform="",
            no_pdf=True, json=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_deliver(args, ctx)
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("report built", text)
        self.assertIn("--to", text)

    def test_cmd_deliver_needs_sections(self) -> None:
        from nomorals.cli import _cmd_deliver

        ctx = _context()
        args = argparse.Namespace(
            deliver_action="report", topic="T", title="",
            section=[], to="", platform="", no_pdf=False, json=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_deliver(args, ctx)
        self.assertEqual(rc, 2)
        self.assertIn("--section", buf.getvalue())

    def test_cmd_deliver_rejects_bad_section(self) -> None:
        from nomorals.cli import _cmd_deliver

        ctx = _context()
        args = argparse.Namespace(
            deliver_action="report", topic="T", title="",
            section=["no-separator-here"], to="", platform="",
            no_pdf=False, json=False)
        with redirect_stdout(io.StringIO()):
            rc = _cmd_deliver(args, ctx)
        self.assertEqual(rc, 2)

    def test_cmd_deliver_send_path_uses_mocked_send(self) -> None:
        from nomorals.cli import _cmd_deliver

        ctx = _context()
        captured: dict[str, Any] = {}

        def fake_deliver(context: Any, topic: str, sections: Any,
                         platform: str, chat: str, **kw: Any) -> dict:
            captured.update(topic=topic, platform=platform, chat=chat,
                            sections=list(sections))
            return {"ok": True, "zip_path": "/tmp/r.zip", "zip_bytes": 42,
                    "message_id": "m9", "sections": ["A"]}

        args = argparse.Namespace(
            deliver_action="report", topic="Q3 markets", title="",
            section=["Overview::body"], to="telegram:123", platform="",
            no_pdf=False, json=False)
        with patch("nomorals.tools.deliver_report.deliver_report",
                   side_effect=fake_deliver):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_deliver(args, ctx)
        self.assertEqual(rc, 0)
        self.assertEqual(captured["platform"], "telegram")
        self.assertEqual(captured["chat"], "123")
        self.assertIn("delivered", buf.getvalue())

    def test_help_deliver_lists_report(self) -> None:
        from nomorals.cli import _cli_command_help, _cli_subparsers

        page = _cli_command_help("deliver")
        self.assertIsNotNone(page)
        assert page is not None
        self.assertIn("report", page)
        # the report subcommand's own help carries the send flags
        deliver_p = _cli_subparsers().choices["deliver"]
        sub_action = next(
            a for a in deliver_p._actions  # noqa: SLF001 - argparse internals
            if isinstance(a, argparse._SubParsersAction))
        report_help = sub_action.choices["report"].format_help()
        self.assertIn("--to", report_help)
        self.assertIn("--section", report_help)

    def test_help_cli_overview_lists_deliver(self) -> None:
        from nomorals.cli import _cli_overview

        self.assertIn("deliver", _cli_overview())


class ChatWiringTests(unittest.TestCase):
    def test_parse_control_routes_deliver(self) -> None:
        from nomorals.social.chat.control import parse_control

        cmd = parse_control('/deliver report "Q3 markets"')
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "deliver")

    def test_control_details_and_catalog_cover_deliver(self) -> None:
        from nomorals.social.chat.control import (
            COMMAND_DETAILS, LIST_GROUPS, help_text, list_catalog)

        self.assertIn("deliver", COMMAND_DETAILS)
        self.assertTrue(any("deliver" in cmds for _, cmds in LIST_GROUPS))
        self.assertIn("/deliver", help_text())
        self.assertIn("/deliver", list_catalog())

    def test_control_deliver_runs_real_pipeline(self) -> None:
        from nomorals.agents.partner_runtime import PartnerRuntime

        ctx = _context()
        ctx.extras = {"gateway": FakeGateway()}
        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = ctx
        reply = rt._control_deliver(
            'report "Q3 markets" --section "Overview::The quarter was hot"',
            "telegram:123")
        self.assertIn("delivered", reply)
        self.assertIn("telegram:123", reply)
        gw = ctx.extras["gateway"]
        self.assertEqual(len(gw.sent), 1)
        sent_path = Path(gw.sent[0]["path"])
        self.assertTrue(sent_path.suffix == ".zip")
        with zipfile.ZipFile(sent_path) as zf:
            names = zf.namelist()
            html_text = zf.read(
                next(n for n in names if n.endswith(".html"))).decode()
        self.assertIn("Q3 markets", html_text)
        self.assertIn("Overview", html_text)

    def test_control_deliver_usage_errors(self) -> None:
        from nomorals.agents.partner_runtime import PartnerRuntime

        ctx = _context()
        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = ctx
        self.assertIn("usage", rt._control_deliver("", "telegram:123"))
        self.assertIn("usage", rt._control_deliver("bogus", "telegram:123"))
        self.assertIn("section", rt._control_deliver(
            'report "T"', "telegram:123").lower())


class RegistryTests(unittest.TestCase):
    def test_tool_registered_in_builtins(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()),
            extras={}, router=None))
        reg.register_builtins()
        tool = reg.get("deliver_report")
        self.assertIsNotNone(tool)


if __name__ == "__main__":
    unittest.main()
