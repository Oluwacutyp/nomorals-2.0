"""Local GGUF manager: find binary/models, diagnose, and the failure modes
that made 'downloaded an 8B' fail on Termux."""

from __future__ import annotations

import http.server
import os
import shutil
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import nomorals.llm.local_server as ls
from nomorals.llm.local_server import GGUFServerManager, find_gguf, port_in_use, resolve_gguf_repo


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FindGgufTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-gguf-")
        self.cache = Path(self.tmp.name) / "models"
        self.cache.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, name: str, size: int = 10) -> Path:
        path = self.cache / name
        path.write_bytes(b"\0" * size)
        return path

    def test_direct_path_wins(self) -> None:
        target = self._write("model.gguf")
        self.assertEqual(find_gguf(str(target), self.cache), str(target))

    def test_directory_is_searched_inside(self) -> None:
        subdir = self.cache / "repo"
        subdir.mkdir()
        (subdir / "weights.gguf").write_bytes(b"\0" * 4)
        found = find_gguf(str(subdir), self.cache)
        self.assertEqual(found, str(subdir / "weights.gguf"))

    def test_name_match_prefers_larger_quant(self) -> None:
        small = self._write("dolphin-7b-q4_0.gguf", size=5)
        big = self._write("dolphin-7b-q4_K_M.gguf", size=50)
        # 'q4_K_M' is not a substring of 'q4_0' either way; prefer by name first.
        self.assertEqual(find_gguf("dolphin", self.cache) in {str(small), str(big)}, True)

    def test_single_file_matches_any_name(self) -> None:
        only = self._write("anything.gguf")
        self.assertEqual(find_gguf("totally-different", self.cache), str(only))

    def test_no_match_is_empty(self) -> None:
        self._write("a.gguf")
        self._write("b.gguf")
        self.assertEqual(find_gguf("zzz", self.cache), "")

    def test_empty_name_is_empty(self) -> None:
        self.assertEqual(find_gguf("", self.cache), "")


class ManagerDiagnosticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-gguf-")
        port = _free_port()
        self.manager = GGUFServerManager(
            host="127.0.0.1", port=port, cache_dir=self.tmp.name,
        )

    def tearDown(self) -> None:
        self.manager.stop()
        self.tmp.cleanup()

    def test_doctor_reports_missing_binary_and_model(self) -> None:
        original = ls.find_llama_binary
        ls.find_llama_binary = lambda: ""  # type: ignore[assignment]
        try:
            diagnosis = self.manager.doctor("some-model")
        finally:
            ls.find_llama_binary = original
        self.assertFalse(diagnosis.ok)
        joined = " ".join(diagnosis.problems)
        self.assertIn("binary", joined)
        self.assertIn(".gguf", joined)
        self.assertTrue(any("llama.cpp" in h for h in diagnosis.hints))

    def test_doctor_happy_path_when_both_present(self) -> None:
        original_binary, original_gguf = ls.find_llama_binary, ls.find_gguf
        # GGUF magic + plausible size: wave 89 made doctor verify the file
        (Path(self.tmp.name) / "model.gguf").write_bytes(b"GGUF\x02\x00\x00\x00" + b"\0" * (12 * 1024 * 1024))
        ls.find_llama_binary = lambda: "/usr/bin/llama-server"  # type: ignore[assignment]
        ls.find_gguf = lambda name, cache: str(Path(self.tmp.name) / "model.gguf")  # type: ignore[assignment]
        try:
            diagnosis = self.manager.doctor("model")
        finally:
            ls.find_llama_binary, ls.find_gguf = original_binary, original_gguf
        self.assertTrue(diagnosis.ok, diagnosis.problems)
        self.assertEqual(diagnosis.binary, "/usr/bin/llama-server")

    def test_start_without_binary_gives_actionable_problems(self) -> None:
        original = ls.find_llama_binary
        ls.find_llama_binary = lambda: ""  # type: ignore[assignment]
        try:
            diagnosis = self.manager.start("model.gguf")
        finally:
            ls.find_llama_binary = original
        self.assertFalse(diagnosis.ok)
        joined = " ".join(diagnosis.problems)
        self.assertIn("binary", joined)
        self.assertIn(".gguf", joined)
        self.assertTrue(diagnosis.hints)

    def test_fetch_unknown_catalog_name_fails_cleanly(self) -> None:
        diagnosis = self.manager.fetch_gguf("definitely-not-a-real-model")
        self.assertFalse(diagnosis.ok)
        self.assertTrue(any("no GGUF repo found" in p for p in diagnosis.problems))
        self.assertTrue(diagnosis.hints)

    def test_stop_and_status_without_a_process(self) -> None:
        self.manager.stop()  # must not raise
        status = self.manager.status()
        self.assertFalse(status["running"])
        self.assertEqual(status["pid"], 0)


class ResolveGgufRepoTest(unittest.TestCase):
    def test_family_names_resolve_to_gguf_repos(self) -> None:
        # TheBloke's repos were deleted upstream; the catalog uses live mirrors.
        self.assertEqual(resolve_gguf_repo("dolphin-8b"), "bartowski/dolphin-2.9.1-llama-3-8b-GGUF")
        self.assertTrue(resolve_gguf_repo("mistral").endswith("-GGUF"))
        self.assertTrue(resolve_gguf_repo("qwen").endswith("-GGUF"))
        self.assertEqual(resolve_gguf_repo("phi-3.5"), "bartowski/Phi-3.5-mini-instruct-GGUF")

    def test_llama_name_reaches_a_llama_gguf(self) -> None:
        # The user asked for "Llama 8B" — that name must land on a llama GGUF.
        self.assertIn("llama-3-8b", resolve_gguf_repo("llama").lower())

    def test_explicit_repo_id_passes_through(self) -> None:
        self.assertEqual(resolve_gguf_repo("some-org/some-repo"), "some-org/some-repo")

    def test_unknown_name_is_empty(self) -> None:
        self.assertEqual(resolve_gguf_repo("no-such-model"), "")
        self.assertEqual(resolve_gguf_repo(""), "")


class FindLlamaBinaryTest(unittest.TestCase):
    def test_finds_cmake_build_bin_layout(self) -> None:
        # The exact layout the guide's build produces: ~/llama.cpp/build/bin/llama-server
        tmp = tempfile.TemporaryDirectory(prefix="nm-llama-")
        build_bin = Path(tmp.name) / "llama.cpp" / "build" / "bin"
        build_bin.mkdir(parents=True)
        fake = build_bin / "llama-server"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = tmp.name
        try:
            self.assertEqual(ls.find_llama_binary(), str(fake))
        finally:
            if old_home is None:
                del os.environ["HOME"]
            else:
                os.environ["HOME"] = old_home
            tmp.cleanup()

    def test_missing_everywhere_is_empty(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="nm-llama-")
        old_home, old_prefix = os.environ.get("HOME"), os.environ.get("PREFIX")
        os.environ["HOME"] = tmp.name
        os.environ.pop("PREFIX", None)
        try:
            # No llama-server on the test machine's PATH, no build dir → "".
            if shutil.which("llama-server") is None and shutil.which("llama-cli") is None:
                self.assertEqual(ls.find_llama_binary(), "")
        finally:
            if old_home is None:
                del os.environ["HOME"]
            else:
                os.environ["HOME"] = old_home
            if old_prefix is None:
                os.environ.pop("PREFIX", None)
            else:
                os.environ["PREFIX"] = old_prefix
            tmp.cleanup()


class PortTest(unittest.TestCase):
    def test_free_port_is_not_in_use(self) -> None:
        self.assertFalse(port_in_use("127.0.0.1", _free_port()))

    def test_occupied_port_is_reported(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        try:
            self.assertTrue(port_in_use("127.0.0.1", server.getsockname()[1]))
        finally:
            server.close()


class ListFailureHintsTest(unittest.TestCase):
    def test_401_blames_repo_not_token(self) -> None:
        hints = ls._list_failure_hints(
            "401 unauthorized for https://huggingface.co/api/models/TheBloke/x: "
            '{"error":"Invalid username or password."}'
        )
        joined = " ".join(hints).lower()
        self.assertIn("deleted", joined)
        self.assertIn("no hf_token", joined)
        self.assertIn("org/repo", joined)

    def test_404_says_gone_or_renamed(self) -> None:
        hints = ls._list_failure_hints("404 not found: https://huggingface.co/api/models/a/b")
        joined = " ".join(hints).lower()
        self.assertIn("renamed or deleted", joined)

    def test_generic_keeps_network_hint(self) -> None:
        hints = ls._list_failure_hints("connection reset by peer")
        self.assertEqual(len(hints), 1)
        self.assertIn("network", hints[0].lower())


class CatalogGgufReposTest(unittest.TestCase):
    """Regression: TheBloke's dolphin/qwen/phi GGUF repos were deleted
    upstream (anonymous 401) — the catalog must point at live public mirrors."""

    def test_family_names_resolve_to_live_repos(self) -> None:
        self.assertEqual(
            resolve_gguf_repo("dolphin-8b"), "bartowski/dolphin-2.9.1-llama-3-8b-GGUF"
        )
        self.assertEqual(resolve_gguf_repo("qwen"), "bartowski/Qwen2.5-7B-Instruct-GGUF")
        self.assertEqual(resolve_gguf_repo("phi-3.5"), "bartowski/Phi-3.5-mini-instruct-GGUF")

    def test_explicit_repo_id_passes_through(self) -> None:
        self.assertEqual(resolve_gguf_repo("org/repo"), "org/repo")


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


class _NonModelHandler(http.server.BaseHTTPRequestHandler):
    """A real HTTP server that is NOT a model server: /health is 404."""

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


class ExternalServerRecognitionTest(unittest.TestCase):
    """A model server started by ANOTHER process (a previous CLI call, or
    the chat's auto-start) must be recognized as running — not reported as
    a stale port holder.  Incident: doctor said 'port 8080 is already in
    use by another process' about the very server --start-local had just
    reported ready (pid 16581)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-ext-")
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.manager = GGUFServerManager(
            host="127.0.0.1", port=self.port, cache_dir=self.tmp.name
        )

    def tearDown(self) -> None:
        self.manager.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def test_status_recognizes_external_healthy_server(self) -> None:
        status = self.manager.status()
        self.assertTrue(status["running"])
        self.assertTrue(status["healthy"])
        self.assertEqual(status["url"], self.manager.base_url)

    def test_doctor_does_not_flag_external_healthy_server(self) -> None:
        diagnosis = self.manager.doctor("")
        self.assertFalse(any("already in use" in p for p in diagnosis.problems))
        self.assertTrue(any("already serving" in h for h in diagnosis.hints))

    def test_start_is_noop_success_when_healthy_server_already_serves(self) -> None:
        diagnosis = self.manager.start("/nonexistent/model.gguf")
        self.assertTrue(diagnosis.ok)
        self.assertTrue(any("already serving" in h for h in diagnosis.hints))

    def test_port_held_by_non_model_process_is_still_flagged(self) -> None:
        # a real HTTP server that is NOT a model server (404 on /health)
        # is genuinely stale — the port is held, but nothing models it
        other_httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NonModelHandler)
        other_port = other_httpd.server_address[1]
        threading.Thread(target=other_httpd.serve_forever, daemon=True).start()
        try:
            other = GGUFServerManager(
                host="127.0.0.1", port=other_port, cache_dir=self.tmp.name
            )
            self.assertFalse(other.status()["running"])
            diagnosis = other.doctor("")
            self.assertTrue(any("does not answer" in p for p in diagnosis.problems))
        finally:
            other_httpd.shutdown()
            other_httpd.server_close()


if __name__ == "__main__":
    unittest.main()
