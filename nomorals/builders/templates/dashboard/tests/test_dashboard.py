"""Tests for the $PROJECT_NAME dashboard.  Run from the project root:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

import run


class DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = run.make_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
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

    def test_index_is_dashboard_html(self) -> None:
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("$PROJECT_NAME", body)
        self.assertIn("<canvas", body)
        self.assertIn("/api/data", body)

    def test_health(self) -> None:
        status, body = self._get("/api/health")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["project"], "$PROJECT_NAME")

    def test_data_endpoint(self) -> None:
        status, body = self._get("/api/data")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        rows = payload["rows"]
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)
        for row in rows:
            for key in ("t", "requests", "errors", "latency_ms"):
                self.assertIn(key, row)

    def test_summary_stats_are_consistent(self) -> None:
        _, body = self._get("/api/data")
        payload = json.loads(body)
        rows, summary = payload["rows"], payload["summary"]
        self.assertEqual(summary["points"], len(rows))
        self.assertEqual(summary["total_requests"],
                         sum(r["requests"] for r in rows))
        self.assertEqual(summary["total_errors"],
                         sum(r["errors"] for r in rows))
        latencies = [r["latency_ms"] for r in rows]
        self.assertAlmostEqual(summary["avg_latency_ms"],
                               sum(latencies) / len(latencies), places=2)
        self.assertEqual(summary["max_latency_ms"], max(latencies))
        expected_rate = summary["total_errors"] / summary["total_requests"]
        self.assertAlmostEqual(summary["error_rate"], expected_rate, places=4)

    def test_summarize_empty_rows(self) -> None:
        summary = run.summarize([])
        self.assertEqual(summary["points"], 0)
        self.assertEqual(summary["total_requests"], 0)

    def test_unknown_path_404_json(self) -> None:
        try:
            self._get("/nope")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)
            payload = json.loads(exc.read().decode("utf-8"))
            self.assertIn("error", payload)
        else:
            self.fail("expected HTTP 404")


if __name__ == "__main__":
    unittest.main()
