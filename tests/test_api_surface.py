"""Surface track — new API routes: SSE /stream, module endpoints, /docs.

Hermetic: dispatch-level tests for status mapping (CapabilityDenied→403,
NoMoralsError→400, else 200) plus a real ThreadingHTTPServer on loopback
for the endpoints that need live HTTP (/stream SSE, auth paths).
"""

from __future__ import annotations

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

from nomorals.api.server import APIServer, Principal, _make_handler
from nomorals.core.events import Event
from nomorals.core.policy import Capability, CapabilitySet
from nomorals.os.session_bridge import SessionBridge
from nomorals.os.timeline import Timeline
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry

OWNER_TOKEN = "owner-secret-surface-test"
RO_TOKEN = "read-only-surface-test"
EMPTY_TOKEN = "empty-grant-surface-test"


def _context(tmpdir: str, db_path: str | None = None) -> Any:
    db = Database(db_path) if db_path else Database(":memory:")
    settings = SimpleNamespace(
        api=SimpleNamespace(token="", max_body_mb=0),
        backup_dir=tmpdir,
        workspace_dir=tmpdir,
    )
    return SimpleNamespace(
        db=db, settings=settings, tools=ToolRegistry(), bus=None, memory=None
    )


def _server(context: Any, **kw: Any) -> APIServer:
    kw.setdefault("max_body_bytes", 64 * 1024)
    return APIServer(
        context,
        token=OWNER_TOKEN,
        principals={
            RO_TOKEN: Principal(
                "ro", CapabilitySet.of(Capability.DB_READ)
            ),
            EMPTY_TOKEN: Principal("empty", CapabilitySet.none()),
        },
        **kw,
    )


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class _LiveServer:
    """Real HTTP server on loopback; http.client for full header control."""

    def __init__(self, api: APIServer) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(api))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body=body, headers=headers or {})
        return conn.getresponse()


# ── GET /docs ────────────────────────────────────────────────────────────


class DocsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = _server(_context("/tmp"))
        self.owner = self.api.resolve_principal(f"Bearer {OWNER_TOKEN}")

    def test_docs_lists_every_route_with_description(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/docs", {}, {}, principal=self.owner
        )
        self.assertEqual(status, 200)
        routes = {r["path"]: r for r in payload["routes"]}
        for path in (
            "/health", "/models", "/tools", "/tools/call", "/chat",
            "/memory/remember", "/memory/recall", "/memory/stats",
            "/agents/run", "/backup", "/events",
            "/stream", "/wisdom/ask", "/search", "/connectors",
            "/sessions", "/triggers", "/timeline", "/docs",
            "/triggers/webhook",
        ):
            self.assertIn(path, routes, f"{path} missing from /docs")
        for route in payload["routes"]:
            self.assertIn("method", route)
            self.assertIn("path", route)
            self.assertIn("description", route)
            self.assertTrue(route["description"], f"{route['path']} has no description")
        methods = {(r["method"], r["path"]) for r in payload["routes"]}
        self.assertIn(("POST", "/triggers"), methods)
        self.assertIn(("GET", "/triggers"), methods)
        self.assertIn(("GET", "/stream"), methods)

    def test_docs_reachable_without_owner_grant(self) -> None:
        # The local bounded principal may read the API's own documentation.
        status, payload = self.api.dispatch("GET", "/docs", {}, {})
        self.assertEqual(status, 200)
        self.assertTrue(payload["routes"])


# ── POST /wisdom/ask ─────────────────────────────────────────────────────


class WisdomAskTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.tmpdir = tempfile.mkdtemp(prefix="api-wisdom-")
        self.api = _server(_context(self.tmpdir))
        self.owner = self.api.resolve_principal(f"Bearer {OWNER_TOKEN}")
        self.ro = self.api.resolve_principal(f"Bearer {RO_TOKEN}")

    def test_ask_returns_provenance_shape(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/wisdom/ask", {"query": "void state"}, {}, principal=self.owner
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["query"], "void state")
        self.assertIn("passages", payload)
        self.assertIn("synthesis", payload)
        self.assertIsInstance(payload["passages"], list)

    def test_ask_rejects_empty_query(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/wisdom/ask", {"query": "  "}, {}, principal=self.owner
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_ask_rejects_bad_top(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/wisdom/ask", {"query": "x", "top": "many"}, {},
            principal=self.owner,
        )
        self.assertEqual(status, 400)

    def test_ask_denied_without_mem_read(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/wisdom/ask", {"query": "x"}, {}, principal=self.ro
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["kind"], "CapabilityDenied")


# ── POST /search ─────────────────────────────────────────────────────────


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = _server(_context("/tmp"))
        self.owner = self.api.resolve_principal(f"Bearer {OWNER_TOKEN}")
        self.ro = self.api.resolve_principal(f"Bearer {RO_TOKEN}")

    def test_search_returns_federated_shape(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/search", {"query": "test query"}, {}, principal=self.owner
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["query"], "test query")
        self.assertIn("hits", payload)
        self.assertIn("sources_searched", payload)
        self.assertIn("sources_skipped", payload)

    def test_search_rejects_empty_query(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/search", {"query": ""}, {}, principal=self.owner
        )
        self.assertEqual(status, 400)

    def test_search_rejects_unknown_source(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/search",
            {"query": "x", "sources": ["no-such-source"]}, {},
            principal=self.owner,
        )
        self.assertEqual(status, 400)
        self.assertIn("unknown search source", payload["error"])

    def test_search_rejects_bad_limit(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/search", {"query": "x", "limit": "lots"}, {},
            principal=self.owner,
        )
        self.assertEqual(status, 400)

    def test_search_denied_without_net_out(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/search", {"query": "x"}, {}, principal=self.ro
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["kind"], "CapabilityDenied")


# ── GET /connectors ──────────────────────────────────────────────────────


class ConnectorsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = _server(_context("/tmp"))
        self.owner = self.api.resolve_principal(f"Bearer {OWNER_TOKEN}")
        self.empty = self.api.resolve_principal(f"Bearer {EMPTY_TOKEN}")

    def test_list_connectors(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/connectors", {}, {}, principal=self.owner
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["connectors"])
        entry = payload["connectors"][0]
        for key in ("id", "name", "description", "auth_methods"):
            self.assertIn(key, entry)
        ids = [c["id"] for c in payload["connectors"]]
        self.assertIn("github", ids)

    def test_list_connectors_denied_without_db_read(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/connectors", {}, {}, principal=self.empty
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["kind"], "CapabilityDenied")


# ── GET /sessions ────────────────────────────────────────────────────────


class SessionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Database(":memory:")
        self.api = _server(_context("/tmp"))
        # Point the route at this db: the context owns the db.
        self.api.context.db = self.db
        self.owner = self.api.resolve_principal(f"Bearer {OWNER_TOKEN}")
        self.empty = self.api.resolve_principal(f"Bearer {EMPTY_TOKEN}")

    def test_list_sessions_empty(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/sessions", {}, {}, principal=self.owner
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["sessions"], [])

    def test_list_sessions_shows_active(self) -> None:
        bridge = SessionBridge(self.db)
        session = bridge.store.create(
            frontend="api", principal="owner", conversation_id="api:test"
        )
        status, payload = self.api.dispatch(
            "GET", "/sessions", {}, {}, principal=self.owner
        )
        self.assertEqual(status, 200)
        ids = [s["id"] for s in payload["sessions"]]
        self.assertIn(session.id, ids)

    def test_list_sessions_denied_without_db_read(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/sessions", {}, {}, principal=self.empty
        )
        self.assertEqual(status, 403)


# ── GET/POST /triggers ───────────────────────────────────────────────────


class TriggersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Database(":memory:")
        ctx = _context("/tmp")
        ctx.db = self.db
        self.api = _server(ctx)
        self.owner = self.api.resolve_principal(f"Bearer {OWNER_TOKEN}")
        # The bounded local principal: DB_READ yes, DB_WRITE no.
        self.local = self.api.resolve_principal("")

    def _valid_body(self) -> dict[str, Any]:
        return {
            "name": "hourly ping",
            "source": "schedule",
            "condition": {"interval": "30m"},
            "action": "notify",
            "action_params": {"title": "ping"},
        }

    def test_list_empty(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/triggers", {}, {}, principal=self.owner
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["triggers"], [])

    def test_create_and_list(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/triggers", self._valid_body(), {}, principal=self.owner
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        trigger = payload["trigger"]
        self.assertTrue(trigger["id"].startswith("trg_"))
        self.assertEqual(trigger["name"], "hourly ping")
        self.assertEqual(trigger["source"], "schedule")
        self.assertEqual(trigger["action"], "notify")

        status, payload = self.api.dispatch(
            "GET", "/triggers", {}, {}, principal=self.owner
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["triggers"]), 1)

    def test_create_rejects_unknown_source(self) -> None:
        body = self._valid_body()
        body["source"] = "telepathy"
        status, payload = self.api.dispatch(
            "POST", "/triggers", body, {}, principal=self.owner
        )
        self.assertEqual(status, 400)
        self.assertIn("unknown source", payload["error"])

    def test_create_rejects_missing_name(self) -> None:
        body = self._valid_body()
        body["name"] = " "
        status, payload = self.api.dispatch(
            "POST", "/triggers", body, {}, principal=self.owner
        )
        self.assertEqual(status, 400)

    def test_create_denied_without_db_write(self) -> None:
        status, payload = self.api.dispatch(
            "POST", "/triggers", self._valid_body(), {}, principal=self.local
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["kind"], "CapabilityDenied")

    def test_list_denied_without_db_read(self) -> None:
        empty = self.api.resolve_principal(f"Bearer {EMPTY_TOKEN}")
        status, _ = self.api.dispatch("GET", "/triggers", {}, {}, principal=empty)
        self.assertEqual(status, 403)


# ── GET /timeline ────────────────────────────────────────────────────────


class TimelineTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        import os

        self.tmpdir = tempfile.mkdtemp(prefix="api-timeline-")
        self.db_path = os.path.join(self.tmpdir, "nomorals.db")
        self.ctx = _context(self.tmpdir, db_path=self.db_path)
        self.api = _server(self.ctx)
        self.owner = self.api.resolve_principal(f"Bearer {OWNER_TOKEN}")
        self.empty = self.api.resolve_principal(f"Bearer {EMPTY_TOKEN}")
        tl = Timeline(self.db_path)
        try:
            tl.record(Event(topic="test.ping", data={"n": 1}, source="api-test"))
        finally:
            tl.close()

    def test_query_returns_events(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/timeline", {}, {"topic": "test.*"}, principal=self.owner
        )
        self.assertEqual(status, 200)
        topics = [e["topic"] for e in payload["events"]]
        self.assertIn("test.ping", topics)

    def test_query_bad_limit_is_400(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/timeline", {}, {"limit": "many"}, principal=self.owner
        )
        self.assertEqual(status, 400)

    def test_query_bad_since_is_400_not_500(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/timeline", {}, {"since": "not-a-date"}, principal=self.owner
        )
        self.assertEqual(status, 400)

    def test_query_denied_without_db_read(self) -> None:
        status, _ = self.api.dispatch(
            "GET", "/timeline", {}, {}, principal=self.empty
        )
        self.assertEqual(status, 403)


# ── GET /stream (live SSE) ───────────────────────────────────────────────


class StreamTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        import os

        self.tmpdir = tempfile.mkdtemp(prefix="api-stream-")
        self.db_path = os.path.join(self.tmpdir, "nomorals.db")
        self.ctx = _context(self.tmpdir, db_path=self.db_path)
        self.api = _server(self.ctx)
        self.live = _LiveServer(self.api)
        tl = Timeline(self.db_path)
        try:
            tl.record(Event(topic="stream.probe", data={"v": 7}, source="api-test"))
        finally:
            tl.close()

    def tearDown(self) -> None:
        self.live.close()

    def test_stream_emits_timeline_events_as_sse(self) -> None:
        resp = self.live.request(
            "GET", "/stream?since=0", headers=_auth(OWNER_TOKEN)
        )
        try:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/event-stream", resp.getheader("Content-Type"))
            lines = [
                resp.fp.readline().decode("utf-8", "replace").rstrip("\n")
                for _ in range(4)
            ]
            self.assertEqual(lines[0], "event: timeline")
            self.assertTrue(lines[1].startswith("id: "))
            self.assertTrue(lines[2].startswith("data: "))
            data = json.loads(lines[2][len("data: "):])
            self.assertEqual(data["topic"], "stream.probe")
            self.assertEqual(lines[3], "")
        finally:
            resp.close()

    def test_stream_bad_since_is_400(self) -> None:
        resp = self.live.request(
            "GET", "/stream?since=nope", headers=_auth(OWNER_TOKEN)
        )
        try:
            self.assertEqual(resp.status, 400)
            payload = json.loads(resp.read().decode("utf-8"))
            self.assertIn("bad since", payload["error"])
        finally:
            resp.close()

    def test_stream_requires_auth(self) -> None:
        resp = self.live.request("GET", "/stream")
        try:
            self.assertEqual(resp.status, 401)
        finally:
            resp.close()

    def test_stream_denied_without_db_read(self) -> None:
        resp = self.live.request("GET", "/stream", headers=_auth(EMPTY_TOKEN))
        try:
            self.assertEqual(resp.status, 403)
            payload = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(payload["kind"], "CapabilityDenied")
        finally:
            resp.close()

    def test_docs_advertises_stream_over_http(self) -> None:
        resp = self.live.request("GET", "/docs", headers=_auth(OWNER_TOKEN))
        try:
            self.assertEqual(resp.status, 200)
            payload = json.loads(resp.read().decode("utf-8"))
            methods = {(r["method"], r["path"]) for r in payload["routes"]}
            self.assertIn(("GET", "/stream"), methods)
        finally:
            resp.close()


if __name__ == "__main__":
    unittest.main()
