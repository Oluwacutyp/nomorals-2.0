"""Wave H3 — explicit API principals and request body limits. All hermetic:
a real ThreadingHTTPServer on loopback with a fake tool registry."""

from __future__ import annotations

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

from nomorals.api.server import (
    APIServer,
    DEFAULT_GRANT,
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_PRINCIPAL,
    OWNER_PRINCIPAL,
    Principal,
    _json_limit_error,
    _make_handler,
)
from nomorals.core.policy import Capability, CapabilitySet
from nomorals.tools.registry import ToolRegistry

OWNER_TOKEN = "owner-secret-token"
ANALYST_TOKEN = "analyst-token-abc"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "note_read",
        lambda: "read-ok",
        capability=Capability.MEM_READ,
        description="read-only tool",
    )
    reg.register(
        "shellish",
        lambda: "did-it",
        capability=Capability.EXEC_SHELL,
        description="privileged tool",
    )
    reg.register(
        "open_tool",
        lambda: "open-ok",
        description="tool with no capability requirement",
    )
    return reg


def _context(reg: ToolRegistry) -> Any:
    return SimpleNamespace(tools=reg)


def _server(reg: ToolRegistry, **kw: Any) -> APIServer:
    kw.setdefault("max_body_bytes", 64 * 1024)
    return APIServer(
        _context(reg),
        token=OWNER_TOKEN,
        principals={ANALYST_TOKEN: ("analyst", CapabilitySet.of(Capability.MEM_READ))},
        **kw,
    )


def _auth_header(token: str) -> str:
    return f"Bearer {token}"


# ── principal resolution ───────────────────────────────────────────────────


class PrincipalResolutionTests(unittest.TestCase):
    def test_owner_token_gets_full_power_explicitly(self) -> None:
        server = _server(_registry())
        principal = server.resolve_principal(_auth_header(OWNER_TOKEN))
        self.assertIs(principal, OWNER_PRINCIPAL)
        self.assertEqual(principal.name, "owner")
        self.assertTrue(principal.grant.grants(Capability.EXEC_SHELL))
        self.assertTrue(principal.grant.grants("anything.at.all"))

    def test_named_principal_gets_exactly_its_grant(self) -> None:
        server = _server(_registry())
        principal = server.resolve_principal(_auth_header(ANALYST_TOKEN))
        assert principal is not None
        self.assertEqual(principal.name, "analyst")
        self.assertTrue(principal.grant.grants(Capability.MEM_READ))
        self.assertFalse(principal.grant.grants(Capability.EXEC_SHELL))

    def test_missing_or_wrong_token_is_rejected_when_token_configured(self) -> None:
        server = _server(_registry())
        self.assertIsNone(server.resolve_principal(""))
        self.assertIsNone(server.resolve_principal("Bearer wrong-token"))
        self.assertIsNone(server.resolve_principal("Token " + OWNER_TOKEN))

    def test_no_token_configured_means_bounded_default_never_all(self) -> None:
        server = APIServer(_context(_registry()), max_body_bytes=64 * 1024)
        for header in ("", "Bearer anything-at-all"):
            principal = server.resolve_principal(header)
            self.assertIsNotNone(principal)
            assert principal is not None
            self.assertEqual(principal.name, "local")
            # The bounded default grant object — shared intentionally, and
            # crucially never CapabilitySet.all().
            self.assertIs(principal.grant, DEFAULT_GRANT)
            self.assertNotIn("*", principal.grant.patterns)
            # Bounded: some things allowed, destructive things not.
            self.assertTrue(principal.grant.grants(Capability.MEM_READ))
            self.assertFalse(principal.grant.grants(Capability.EXEC_SHELL))
            self.assertFalse(principal.grant.grants(Capability.FS_WRITE))
            self.assertFalse(principal.grant.grants("anything.at.all"))

    def test_principals_accept_principal_objects(self) -> None:
        custom = Principal(name="bot", grant=CapabilitySet.of(Capability.DB_READ))
        server = APIServer(
            _context(_registry()),
            token=OWNER_TOKEN,
            principals={"tok": custom},
            max_body_bytes=1024,
        )
        self.assertIs(server.resolve_principal("Bearer tok"), custom)


# ── /tools/call honors the principal ───────────────────────────────────────


class ToolsCallPrincipalTests(unittest.TestCase):
    def test_owner_can_invoke_privileged_tool(self) -> None:
        reg = _registry()
        server = _server(reg)
        owner = server.resolve_principal(_auth_header(OWNER_TOKEN))
        status, payload = server.dispatch(
            "POST", "/tools/call", {"name": "shellish"}, {}, principal=owner
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"ok": True, "result": "did-it"})

    def test_lesser_principal_allowed_tool_succeeds(self) -> None:
        reg = _registry()
        server = _server(reg)
        analyst = server.resolve_principal(_auth_header(ANALYST_TOKEN))
        status, payload = server.dispatch(
            "POST", "/tools/call", {"name": "note_read"}, {}, principal=analyst
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"ok": True, "result": "read-ok"})

    def test_lesser_principal_denied_tool_is_non_200(self) -> None:
        reg = _registry()
        server = _server(reg)
        analyst = server.resolve_principal(_auth_header(ANALYST_TOKEN))
        status, payload = server.dispatch(
            "POST", "/tools/call", {"name": "shellish"}, {}, principal=analyst
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["kind"], "CapabilityDenied")
        self.assertIn("error", payload)

    def test_tools_call_uses_principal_grant_not_unconditional_all(self) -> None:
        reg = _registry()
        seen: dict[str, Any] = {}
        original_call = reg.call

        def spy(name: str, /, **kw: Any):  # type: ignore[no-untyped-def]
            seen["capabilities"] = kw.get("capabilities")
            seen["actor"] = kw.get("actor")
            return original_call(name, **kw)

        reg.call = spy  # type: ignore[method-assign]
        try:
            server = _server(reg)
            analyst = server.resolve_principal(_auth_header(ANALYST_TOKEN))
            assert analyst is not None
            server.dispatch(
                "POST", "/tools/call", {"name": "note_read"}, {}, principal=analyst
            )
            # Exactly the principal's grant object — not CapabilitySet.all().
            self.assertIs(seen["capabilities"], analyst.grant)
            self.assertFalse(seen["capabilities"].grants(Capability.EXEC_SHELL))
            # Actor comes from the principal identity, not the request body.
            self.assertEqual(seen["actor"], "api:analyst")
        finally:
            reg.call = original_call  # type: ignore[method-assign]

    def test_body_actor_cannot_spoof_identity(self) -> None:
        reg = _registry()
        seen: dict[str, Any] = {}
        original_call = reg.call

        def spy(name: str, /, **kw: Any):  # type: ignore[no-untyped-def]
            seen["actor"] = kw.get("actor")
            return original_call(name, **kw)

        reg.call = spy  # type: ignore[method-assign]
        try:
            server = _server(reg)
            analyst = server.resolve_principal(_auth_header(ANALYST_TOKEN))
            server.dispatch(
                "POST",
                "/tools/call",
                {"name": "open_tool", "actor": "api:owner"},
                {},
                principal=analyst,
            )
            self.assertEqual(seen["actor"], "api:analyst")
        finally:
            reg.call = original_call  # type: ignore[method-assign]

    def test_default_principal_gets_bounded_grant_on_tools_call(self) -> None:
        reg = _registry()
        server = APIServer(_context(reg), max_body_bytes=64 * 1024)
        status, payload = server.dispatch(
            "POST", "/tools/call", {"name": "note_read"}, {}
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        status, payload = server.dispatch(
            "POST", "/tools/call", {"name": "shellish"}, {}
        )
        self.assertEqual(status, 403)

    def test_unknown_tool_is_non_200(self) -> None:
        server = _server(_registry())
        owner = server.resolve_principal(_auth_header(OWNER_TOKEN))
        status, payload = server.dispatch(
            "POST", "/tools/call", {"name": "nope"}, {}, principal=owner
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)


# ── JSON guard unit tests ──────────────────────────────────────────────────


class JsonLimitTests(unittest.TestCase):
    def _limits(self, **kw: Any) -> dict[str, Any]:
        base = {
            "max_depth": 64,
            "max_string_len": 100,
            "max_total_string_bytes": 10_000,
            "max_elements": 50,
        }
        base.update(kw)
        return base

    def test_normal_body_passes(self) -> None:
        body = {"name": "note_read", "arguments": {"a": [1, 2, {"b": "x"}]}}
        self.assertIsNone(_json_limit_error(body, **self._limits()))

    def test_deep_nesting_rejected(self) -> None:
        body: Any = {}
        node = body
        for _ in range(80):
            node["n"] = {}
            node = node["n"]
        message, status = _json_limit_error(body, **self._limits())  # type: ignore[misc]
        self.assertEqual(status, 400)
        self.assertIn("depth", message)

    def test_huge_string_rejected(self) -> None:
        message, status = _json_limit_error(
            {"s": "x" * 101}, **self._limits()
        )  # type: ignore[misc]
        self.assertEqual(status, 413)
        self.assertIn("string", message)

    def test_huge_array_rejected(self) -> None:
        message, status = _json_limit_error(
            {"a": list(range(51))}, **self._limits()
        )  # type: ignore[misc]
        self.assertEqual(status, 413)
        self.assertIn("array", message)

    def test_huge_object_rejected(self) -> None:
        message, status = _json_limit_error(
            {f"k{i}": i for i in range(51)}, **self._limits()
        )  # type: ignore[misc]
        self.assertEqual(status, 413)
        self.assertIn("object", message)

    def test_total_string_bytes_rejected(self) -> None:
        body = {f"key{i}": "x" * 40 for i in range(40)}  # ~2k bytes of strings
        message, status = _json_limit_error(
            body, **self._limits(max_total_string_bytes=1000)
        )  # type: ignore[misc]
        self.assertEqual(status, 413)
        self.assertIn("total", message)


# ── live HTTP limit tests ──────────────────────────────────────────────────


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

    def raw(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def raw_headers_only(
        self, path: str, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        """Send headers with no body at all (for garbage/missing length)."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.putrequest("POST", path)
            for key, value in headers.items():
                conn.putheader(key, value)
            conn.endheaders()
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()


class HttpLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = _registry()
        self.api = _server(self.reg, max_body_bytes=4096)
        self.live = _LiveServer(self.api)

    def tearDown(self) -> None:
        self.live.close()

    def _authed(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Authorization": _auth_header(OWNER_TOKEN),
                   "Content-Type": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def test_normal_request_passes(self) -> None:
        body = json.dumps({"name": "note_read"}).encode()
        status, raw = self.live.raw("POST", "/tools/call",
                                    self._authed(), body)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), {"ok": True, "result": "read-ok"})

    def test_oversized_content_length_rejected_before_reading(self) -> None:
        # Declared 1 MiB, cap is 4 KiB — server must 413 without waiting
        # for (or reading) the body.
        status, raw = self.live.raw_headers_only(
            "/tools/call",
            {"Authorization": _auth_header(OWNER_TOKEN),
             "Content-Type": "application/json",
             "Content-Length": str(1024 * 1024)},
        )
        self.assertEqual(status, 413)
        self.assertIn("exceeds", json.loads(raw)["error"])

    def test_garbage_content_length_is_400(self) -> None:
        status, raw = self.live.raw_headers_only(
            "/tools/call",
            {"Authorization": _auth_header(OWNER_TOKEN),
             "Content-Length": "banana"},
        )
        self.assertEqual(status, 400)
        self.assertIn("Content-Length", json.loads(raw)["error"])

    def test_missing_content_length_is_400(self) -> None:
        status, raw = self.live.raw_headers_only(
            "/tools/call",
            {"Authorization": _auth_header(OWNER_TOKEN)},
        )
        self.assertEqual(status, 400)
        self.assertIn("Content-Length", json.loads(raw)["error"])

    def test_negative_content_length_is_400(self) -> None:
        status, raw = self.live.raw_headers_only(
            "/tools/call",
            {"Authorization": _auth_header(OWNER_TOKEN),
             "Content-Length": "-5"},
        )
        self.assertEqual(status, 400)

    def test_deeply_nested_json_rejected(self) -> None:
        node: Any = {}
        top = node
        for _ in range(100):
            node["n"] = {}
            node = node["n"]
        body = json.dumps({"name": "open_tool", "arguments": top}).encode()
        status, raw = self.live.raw("POST", "/tools/call",
                                    self._authed(), body)
        self.assertEqual(status, 400)
        self.assertIn("depth", json.loads(raw)["error"])

    def test_huge_string_rejected(self) -> None:
        body = json.dumps({"name": "x", "pad": "y" * 5000}).encode()
        status, raw = self.live.raw("POST", "/tools/call",
                                    self._authed(), body)
        self.assertEqual(status, 413)

    def test_huge_array_rejected(self) -> None:
        body = json.dumps({"a": list(range(20_000))}).encode()
        status, raw = self.live.raw("POST", "/tools/call",
                                    self._authed(), body)
        self.assertEqual(status, 413)

    def test_live_probe_needs_no_token(self) -> None:
        # Dumb supervisors (systemd, Docker HEALTHCHECK) cannot mint a
        # bearer token: GET /live answers 200 with a static payload,
        # with no token, a wrong token, or a valid one.
        for headers in ({}, {"Authorization": "Bearer nope"},
                        self._authed()):
            status, raw = self.live.raw("GET", "/live", headers)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(raw), {"ok": True})

    def test_health_still_requires_token(self) -> None:
        # The informative /health stays behind bearer auth.
        status, _ = self.live.raw("GET", "/health", {})
        self.assertEqual(status, 401)
        status, raw = self.live.raw("GET", "/health", self._authed())
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw)["ok"])

    def test_unauthorized_without_token(self) -> None:
        body = json.dumps({"name": "note_read"}).encode()
        status, _ = self.live.raw(
            "POST", "/tools/call", {"Content-Type": "application/json"}, body)
        self.assertEqual(status, 401)
        status, _ = self.live.raw(
            "POST", "/tools/call",
            {"Authorization": "Bearer wrong", "Content-Type": "application/json"},
            body)
        self.assertEqual(status, 401)

    def test_lesser_principal_denied_over_http(self) -> None:
        body = json.dumps({"name": "shellish"}).encode()
        status, raw = self.live.raw(
            "POST", "/tools/call",
            {"Authorization": _auth_header(ANALYST_TOKEN),
             "Content-Type": "application/json"},
            body)
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(raw)["kind"], "CapabilityDenied")


class SettingsFallbackTests(unittest.TestCase):
    def test_max_body_bytes_honors_settings_max_body_mb(self) -> None:
        settings = SimpleNamespace(api=SimpleNamespace(max_body_mb=2))
        server = APIServer(SimpleNamespace(tools=_registry(), settings=settings))
        self.assertEqual(server.max_body_bytes, 2 * 1_048_576)

    def test_max_body_bytes_defaults_to_1mib_without_settings(self) -> None:
        server = APIServer(SimpleNamespace(tools=_registry()))
        self.assertEqual(server.max_body_bytes, DEFAULT_MAX_BODY_BYTES)

    def test_invalid_max_body_bytes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            APIServer(SimpleNamespace(tools=_registry()), max_body_bytes=0)


if __name__ == "__main__":
    unittest.main()
