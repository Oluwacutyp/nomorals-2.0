"""Tests for the $PROJECT_NAME web app.  Run from the project root:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.request

import run


class WebAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = run.make_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _get(self, path: str) -> tuple[int, str]:
        with urllib.request.urlopen(self.base + path, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")

    def test_index_returns_html(self) -> None:
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("$PROJECT_NAME", body)
        self.assertIn("<html", body.lower())

    def test_health_returns_json(self) -> None:
        status, body = self._get("/api/health")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["project"], "$PROJECT_NAME")

    def test_echo_round_trip(self) -> None:
        data = json.dumps({"hello": "world"}).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/api/echo", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(payload["echo"], {"hello": "world"})

    def test_echo_rejects_bad_json(self) -> None:
        req = urllib.request.Request(
            self.base + "/api/echo", data=b"not json{{{",
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
        else:
            self.fail("expected HTTP 400 for invalid JSON")

    def test_unknown_path_404(self) -> None:
        try:
            self._get("/nope")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)
        else:
            self.fail("expected HTTP 404")


if __name__ == "__main__":
    unittest.main()
