"""L4 — tools: registry, capability enforcement, sandbox, parsers, vision.

The security assertions matter more than the happy paths: a tool layer that can be
tricked out of its workspace, or that runs without a capability grant, is worse
than no tool layer at all.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
import zipfile
from io import BytesIO
from types import SimpleNamespace
from pathlib import Path

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.core.errors import CapabilityDenied, NotFound, ToolError, ValidationError
from nomorals.core.policy import CapabilitySet
from nomorals.tools.filesystem import safe_path
from nomorals.tools.parsers import detect_kind, parse, parse_pdf
from nomorals.tools.registry import ToolRegistry
from nomorals.tools.shell import detect_backend, run_sandboxed
from nomorals.tools.vision import image_metadata


class SafePathTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="nm-fs-"))
        self.context = SimpleNamespace(settings=SimpleNamespace(workspace_dir=str(self.root)))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_normal_path_resolves_inside_the_workspace(self):
        target = safe_path(self.context, "notes/a.txt")
        self.assertTrue(str(target).startswith(str(self.root)))

    def test_parent_traversal_is_refused_not_normalized(self):
        with self.assertRaises(ValidationError):
            safe_path(self.context, "../../etc/passwd")

    def test_absolute_path_outside_the_workspace_is_refused(self):
        with self.assertRaises(ValidationError):
            safe_path(self.context, "/etc/passwd")

    def test_symlink_escape_is_refused(self):
        outside = self.root.parent / "outside-target"
        outside.write_text("secret", encoding="utf-8")
        try:
            link = self.root / "link"
            link.symlink_to(outside)
            with self.assertRaises(ValidationError):
                safe_path(self.context, "link")
        finally:
            outside.unlink(missing_ok=True)

    def test_missing_file_raises_when_required(self):
        with self.assertRaises(NotFound):
            safe_path(self.context, "nope.txt", must_exist=True)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

        @self.registry.register("add", description="Add two numbers.", capability=None)
        def add(a: int, b: int = 0) -> dict:
            return {"sum": a + b}

        @self.registry.register("boom", description="Always fails.", capability=None)
        def boom() -> dict:
            raise ToolError("intentional failure")

    def test_registration_and_schema_inference(self):
        schemas = {s["name"]: s for s in self.registry.schemas()}
        self.assertIn("add", schemas)
        self.assertIn("a", schemas["add"]["parameters"])
        self.assertIn("b", schemas["add"]["parameters"])
        self.assertEqual(schemas["add"]["description"], "Add two numbers.")

    def test_call_returns_the_value(self):
        outcome = self.registry.call("add", a=2, b=3)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.value["sum"], 5)

    def test_unknown_tool_is_an_error_result_not_an_exception(self):
        outcome = self.registry.call("nope")
        self.assertFalse(outcome.ok)

    def test_tool_exceptions_become_error_results(self):
        outcome = self.registry.call("boom")
        self.assertFalse(outcome.ok)
        self.assertIsNotNone(outcome.error)

    def test_reregistering_a_name_replaces_the_implementation(self):
        @self.registry.register("add", description="replacement")
        def again(a: int = 5) -> dict:
            return {"sum": a * 10}

        outcome = self.registry.call("add", a=5)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.value["sum"], 50)
        descriptions = {s["name"]: s["description"] for s in self.registry.schemas()}
        self.assertEqual(descriptions["add"], "replacement")

    def test_capability_is_enforced_against_the_grant(self):
        registry = ToolRegistry(enforce=True)

        @registry.register("danger", description="needs exec", capability="exec.shell")
        def danger() -> dict:
            return {"ran": True}

        denied = registry.call("danger", capabilities=CapabilitySet.of("fs.read"))
        self.assertFalse(denied.ok)
        self.assertIsInstance(denied.error, CapabilityDenied)
        allowed = registry.call("danger", capabilities=CapabilitySet.of("exec.shell"))
        self.assertTrue(allowed.ok)

    def test_stats_count_calls_and_denials(self):
        self.registry.call("add", a=1)
        self.registry.call("boom")
        self.assertGreaterEqual(self.registry.stats["calls"], 2)
        self.assertGreaterEqual(self.registry.stats["errors"], 1)


class ShellSandboxTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="nm-sh-")

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_command_runs_in_the_given_directory(self):
        result = run_sandboxed("pwd", cwd=self.workdir, timeout=20.0)
        self.assertEqual(result["exit_code"], 0)
        self.assertIn(Path(self.workdir).name, result["stdout"])

    def test_nonzero_exit_is_reported_not_raised(self):
        result = run_sandboxed("exit 3", cwd=self.workdir, timeout=20.0)
        self.assertEqual(result["exit_code"], 3)
        self.assertFalse(result["timed_out"])

    def test_stderr_is_captured_separately(self):
        result = run_sandboxed("echo out; echo err 1>&2", cwd=self.workdir, timeout=20.0)
        self.assertIn("out", result["stdout"])
        self.assertIn("err", result["stderr"])

    def test_timeout_kills_the_whole_process_group(self):
        result = run_sandboxed("sleep 30 & sleep 30", cwd=self.workdir, timeout=2.0)
        self.assertTrue(result["timed_out"])
        self.assertLess(result["seconds"], 10.0)

    def test_backend_detection_never_returns_unavailable(self):
        for preferred in ("auto", "bwrap", "unshare", "rlimit", "none"):
            self.assertIn(detect_backend(preferred), {"bwrap", "unshare", "rlimit", "none"})


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="nm-parse-"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_detect_kind_by_magic_bytes_beats_extension(self):
        pdf = self.dir / "misleading.txt"
        pdf.write_bytes(b"%PDF-1.7\n")
        self.assertEqual(detect_kind(pdf, pdf.read_bytes()[:16]), "pdf")

    def test_detect_kind_for_images(self):
        self.assertEqual(detect_kind("x.png", b"\x89PNG\r\n\x1a\n"), "image")
        self.assertEqual(detect_kind("x.jpg", b"\xff\xd8\xff\xe0"), "image")
        self.assertEqual(detect_kind("x.gif", b"GIF89a"), "image")

    def test_plain_text_round_trips(self):
        target = self.dir / "a.txt"
        target.write_text("hello there", encoding="utf-8")
        result = parse(target)
        self.assertEqual(result["kind"], "text")
        self.assertIn("hello there", result["text"])

    def test_json_is_pretty_printed(self):
        target = self.dir / "a.json"
        target.write_text('{"b":1,"a":2}', encoding="utf-8")
        result = parse(target)
        self.assertEqual(result["kind"], "json")
        self.assertIn('"a"', result["text"])

    def test_csv_reports_row_and_column_counts(self):
        target = self.dir / "a.csv"
        target.write_text("name,age\nann,30\nbob,40\n", encoding="utf-8")
        result = parse(target)
        self.assertEqual(result["meta"]["rows"], 3)
        self.assertEqual(result["meta"]["columns"], 2)
        self.assertIn("ann", result["text"])

    def test_html_is_stripped_to_text(self):
        target = self.dir / "a.html"
        target.write_text(
            "<html><head><title>T</title><script>evil()</script></head>"
            "<body><p>Visible &amp; readable</p></body></html>",
            encoding="utf-8",
        )
        result = parse(target)
        self.assertIn("Visible & readable", result["text"])
        self.assertNotIn("evil()", result["text"])

    def test_docx_extracts_paragraphs_from_the_zip(self):
        target = self.dir / "a.docx"
        document = (
            '<?xml version="1.0"?><w:document '
            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Extracted paragraph</w:t></w:r></w:p></w:body></w:document>"
        )
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("word/document.xml", document)
        result = parse(target)
        self.assertEqual(result["kind"], "docx")
        self.assertIn("Extracted paragraph", result["text"])

    def test_xlsx_resolves_shared_strings(self):
        target = self.dir / "a.xlsx"
        shared = (
            '<?xml version="1.0"?><sst '
            'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<si><t>Revenue</t></si><si><t>42</t></si></sst>"
        )
        sheet = (
            '<?xml version="1.0"?><worksheet '
            'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row></sheetData>'
            "</worksheet>"
        )
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("xl/sharedStrings.xml", shared)
            archive.writestr("xl/worksheets/sheet1.xml", sheet)
        result = parse(target)
        self.assertIn("Revenue", result["text"])
        self.assertIn("42", result["text"])

    def test_pdf_extracts_text_from_a_flate_stream(self):
        import zlib

        stream = zlib.compress(b"BT (Hello PDF world) Tj ET")
        blob = (
            b"%PDF-1.4\n1 0 obj<</Title (Test Doc)>>endobj\n"
            b"2 0 obj<</Filter/FlateDecode/Length " + str(len(stream)).encode() + b">>stream\n"
            + stream
            + b"\nendstream\nendobj\ntrailer<</Size 3>>\n%%EOF"
        )
        target = self.dir / "a.pdf"
        target.write_bytes(blob)
        result = parse(target)
        self.assertEqual(result["kind"], "pdf")
        self.assertIn("Hello PDF world", result["text"])

    def test_pdf_rejects_a_non_pdf(self):
        with self.assertRaises(Exception):
            parse_pdf(b"not a pdf at all")

    def test_corrupt_zip_is_a_parse_error_not_a_crash(self):
        target = self.dir / "a.docx"
        target.write_bytes(b"PK\x03\x04garbage")
        with self.assertRaises(Exception):
            parse(target)

    def test_missing_file_is_a_parse_error(self):
        with self.assertRaises(Exception):
            parse(self.dir / "absent.txt")


class VisionTests(unittest.TestCase):
    def test_png_dimensions(self):
        data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            + (640).to_bytes(4, "big")
            + (480).to_bytes(4, "big")
            + b"\x08\x02"
        )
        meta = image_metadata(data)
        self.assertEqual(meta["format"], "png")
        self.assertEqual(meta["width"], 640)
        self.assertEqual(meta["height"], 480)

    def test_jpeg_dimensions(self):
        # SOI, a well-formed APP0 whose length the scanner must step over, then SOF0.
        app0_payload = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        app0 = b"\xff\xe0" + (len(app0_payload) + 2).to_bytes(2, "big") + app0_payload
        sof0_payload = b"\x08" + (300).to_bytes(2, "big") + (200).to_bytes(2, "big")
        sof0 = b"\xff\xc0" + (len(sof0_payload) + 2).to_bytes(2, "big") + sof0_payload
        data = b"\xff\xd8" + app0 + sof0
        meta = image_metadata(data)
        self.assertEqual(meta["format"], "jpeg")
        self.assertEqual(meta["width"], 200)
        self.assertEqual(meta["height"], 300)

    def test_gif_dimensions(self):
        data = b"GIF89a" + (100).to_bytes(2, "little") + (50).to_bytes(2, "little")
        meta = image_metadata(data)
        self.assertEqual(meta["format"], "gif")
        self.assertEqual(meta["width"], 100)
        self.assertEqual(meta["height"], 50)

    def test_unknown_format_still_reports_a_hash(self):
        meta = image_metadata(b"arbitrary bytes")
        self.assertEqual(meta["format"], "unknown")
        self.assertEqual(len(meta["sha256"]), 64)


class IntegrationTests(unittest.TestCase):
    """The registry wired to a real context, with policy enforcement live."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-tools-")
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.tools = self.context.tools.register_builtins()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_builtins_all_register(self):
        names = set(self.tools.names())
        for expected in ("fs_read", "fs_write", "shell_run", "web_fetch", "parse_file"):
            self.assertIn(expected, names)

    def test_write_then_read_round_trip(self):
        self.tools.call("fs_write", actor="t", capabilities=CapabilitySet.all(),
                        path="a.txt", content="payload")
        outcome = self.tools.call("fs_read", actor="t", capabilities=CapabilitySet.all(), path="a.txt")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.value["content"], "payload")

    def test_traversal_is_blocked_through_the_registry(self):
        outcome = self.tools.call("fs_read", actor="t", capabilities=CapabilitySet.all(),
                                  path="../../etc/passwd")
        self.assertFalse(outcome.ok)

    def test_delete_requires_a_confirmation_token(self):
        self.tools.call("fs_write", actor="t", capabilities=CapabilitySet.all(),
                        path="d.txt", content="x")
        denied = self.tools.call("fs_delete", actor="t", capabilities=CapabilitySet.all(),
                                 path="d.txt")
        self.assertFalse(denied.ok)
        token = self.context.policy.issue_confirmation("fs.delete", ttl=30.0)
        allowed = self.tools.call("fs_delete", actor="t", capabilities=CapabilitySet.all(),
                                  confirmation=token, path="d.txt")
        self.assertTrue(allowed.ok, allowed.error)

    def test_every_call_is_audited(self):
        before = int(self.context.db.scalar("SELECT COUNT(*) FROM tool_calls") or 0)
        self.tools.call("fs_list", actor="auditor", capabilities=CapabilitySet.all(), path=".")
        after = int(self.context.db.scalar("SELECT COUNT(*) FROM tool_calls") or 0)
        self.assertEqual(after, before + 1)

    def test_denials_are_audited_too(self):
        self.tools.call("shell_run", actor="weak", capabilities=CapabilitySet.of("fs.read"),
                        command="id")
        rows = self.context.db.query(
            "SELECT decision FROM tool_calls WHERE tool='shell_run' AND actor='weak'"
        )
        self.assertTrue(rows)
        self.assertEqual(dict(rows[0])["decision"], "deny")


if __name__ == "__main__":
    unittest.main()
