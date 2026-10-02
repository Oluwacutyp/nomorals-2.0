"""Tests for the $PROJECT_NAME REST API.  Run from the project root:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

import run


class RestApiTest(unittest.TestCase):
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

    # -- helpers ---------------------------------------------------------
    def _request(self, method: str, path: str,
                 payload: object | None = None) -> tuple[int, dict]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data,
                                     headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            try:
                return exc.code, json.loads(body)
            except json.JSONDecodeError:
                return exc.code, {"raw": body}

    # -- tests -----------------------------------------------------------
    def test_root_describes_api(self) -> None:
        status, body = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(body["project"], "$PROJECT_NAME")
        self.assertIn("GET /api/health", body["endpoints"])

    def test_health(self) -> None:
        status, body = self._request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["project"], "$PROJECT_NAME")

    def test_crud_lifecycle(self) -> None:
        # create
        status, created = self._request("POST", "/api/items",
                                       {"name": "first"})
        self.assertEqual(status, 201)
        self.assertEqual(created["name"], "first")
        self.assertFalse(created["done"])
        item_id = created["id"]

        # fetch one
        status, fetched = self._request("GET", f"/api/items/{item_id}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, created)

        # list contains it
        status, listed = self._request("GET", "/api/items")
        self.assertEqual(status, 200)
        self.assertTrue(any(i["id"] == item_id for i in listed["items"]))

        # partial update
        status, updated = self._request("PATCH", f"/api/items/{item_id}",
                                       {"done": True})
        self.assertEqual(status, 200)
        self.assertTrue(updated["done"])
        self.assertEqual(updated["name"], "first")

        # delete
        status, deleted = self._request("DELETE", f"/api/items/{item_id}")
        self.assertEqual(status, 200)
        self.assertEqual(deleted["deleted"], item_id)

        # gone now
        status, body = self._request("GET", f"/api/items/{item_id}")
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    def test_create_requires_name(self) -> None:
        for bad in ({}, {"name": ""}, {"name": "   "}):
            status, body = self._request("POST", "/api/items", bad)
            self.assertEqual(status, 400, bad)
            self.assertIn("error", body)

    def test_create_rejects_bad_json(self) -> None:
        req = urllib.request.Request(
            self.base + "/api/items", data=b"not json{{{",
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            body = json.loads(exc.read().decode("utf-8"))
            self.assertIn("error", body)
        else:
            self.fail("expected HTTP 400 for invalid JSON")

    def test_missing_item_404s(self) -> None:
        for method in ("GET", "PATCH", "DELETE"):
            status, body = self._request(method, "/api/items/424242",
                                         {} if method == "PATCH" else None)
            self.assertEqual(status, 404, method)
            self.assertIn("error", body)

    def test_wrong_method_405_with_allow(self) -> None:
        req = urllib.request.Request(self.base + "/api/items", method="PUT")
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 405)
            self.assertIn("Allow", exc.headers)
        else:
            self.fail("expected HTTP 405 for PUT /api/items")

    def test_unknown_path_404_json(self) -> None:
        status, body = self._request("GET", "/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
