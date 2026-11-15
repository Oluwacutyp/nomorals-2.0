"""CORS wiring pins: settings.api.cors_origins -> serve() -> APIServer.

The CORS setting exists (api.cors_origins, config default "*") and the
server honours it (APIServer.cors_origins, echoed in _respond and the
OPTIONS preflight), but nothing wired it through: serve() never passed
cors_origins to APIServer, and ``nm serve`` never passed it to serve().
On a stock ``nm serve`` run CORS was OFF and the config field was dead.
These pins prove the wiring end to end over live HTTP: a request with
the configured Origin gets Access-Control-Allow-Origin, and with
cors_origins unset no CORS header is served.
"""
from __future__ import annotations

import http.client
import socket
import time
from types import SimpleNamespace

from nomorals.api.server import serve
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry

TEST_ORIGIN = "https://devon-vercel-example.vercel.app"
OTHER_ORIGIN = "https://not-allowed.example"


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _context(cors_origins: str | None) -> SimpleNamespace:
    settings = SimpleNamespace(
        api=SimpleNamespace(
            token="",
            max_body_mb=0,
            cors_origins=cors_origins,
        ),
        backup_dir="/tmp",
        workspace_dir="/tmp",
    )
    return SimpleNamespace(
        db=Database(":memory:"),
        settings=settings,
        tools=ToolRegistry(),
        bus=None,
        memory=None,
    )


def _wait_up(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/live")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"server did not come up on port {port}")


def _get(port: int, path: str, origin: str | None) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers: dict[str, str] = {}
    if origin is not None:
        headers["Origin"] = origin
    conn.request("GET", path, headers=headers)
    return conn.getresponse()


def test_serve_wires_cors_origins_from_settings() -> None:
    """serve() with settings.api.cors_origins set serves the header."""
    port = _free_port()
    serve(
        _context(TEST_ORIGIN),
        host="127.0.0.1",
        port=port,
        background=True,
        protocols=False,
    )
    _wait_up(port)
    resp = _get(port, "/live", TEST_ORIGIN)
    try:
        body = resp.read()
        assert resp.status == 200, body
        assert resp.getheader("Access-Control-Allow-Origin") == TEST_ORIGIN
    finally:
        resp.close()


def test_serve_wildcard_cors_origins_echoes_star() -> None:
    """The config default '*' is passed through as allow-all."""
    port = _free_port()
    serve(
        _context("*"),
        host="127.0.0.1",
        port=port,
        background=True,
        protocols=False,
    )
    _wait_up(port)
    resp = _get(port, "/live", OTHER_ORIGIN)
    try:
        body = resp.read()
        assert resp.status == 200, body
        assert resp.getheader("Access-Control-Allow-Origin") == "*"
    finally:
        resp.close()


def test_serve_unset_cors_origins_serves_no_header() -> None:
    """With cors_origins unset, no CORS header is served."""
    port = _free_port()
    serve(
        _context(None),
        host="127.0.0.1",
        port=port,
        background=True,
        protocols=False,
    )
    _wait_up(port)
    resp = _get(port, "/live", TEST_ORIGIN)
    try:
        body = resp.read()
        assert resp.status == 200, body
        assert resp.getheader("Access-Control-Allow-Origin") is None
    finally:
        resp.close()


def test_serve_empty_cors_origins_serves_no_header() -> None:
    """An empty cors_origins string means CORS off, not allow-all."""
    port = _free_port()
    serve(
        _context(""),
        host="127.0.0.1",
        port=port,
        background=True,
        protocols=False,
    )
    _wait_up(port)
    resp = _get(port, "/live", TEST_ORIGIN)
    try:
        body = resp.read()
        assert resp.status == 200, body
        assert resp.getheader("Access-Control-Allow-Origin") is None
    finally:
        resp.close()


def test_serve_comma_separated_cors_origins() -> None:
    """A comma-separated list allows each listed origin, no other."""
    port = _free_port()
    serve(
        _context(TEST_ORIGIN + ", " + OTHER_ORIGIN),
        host="127.0.0.1",
        port=port,
        background=True,
        protocols=False,
    )
    _wait_up(port)
    resp = _get(port, "/live", OTHER_ORIGIN)
    try:
        body = resp.read()
        assert resp.status == 200, body
        assert resp.getheader("Access-Control-Allow-Origin") == OTHER_ORIGIN
    finally:
        resp.close()
    resp = _get(port, "/live", "https://third.example")
    try:
        body = resp.read()
        assert resp.status == 200, body
        assert resp.getheader("Access-Control-Allow-Origin") is None
    finally:
        resp.close()


def test_serve_preflight_options_returns_cors_headers() -> None:
    """OPTIONS preflight answers with the CORS headers for a listed origin."""
    port = _free_port()
    serve(
        _context(TEST_ORIGIN),
        host="127.0.0.1",
        port=port,
        background=True,
        protocols=False,
    )
    _wait_up(port)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(
        "OPTIONS",
        "/live",
        headers={
            "Origin": TEST_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )
    resp = conn.getresponse()
    try:
        body = resp.read()
        assert resp.status == 204, body
        assert resp.getheader("Access-Control-Allow-Origin") == TEST_ORIGIN
    finally:
        resp.close()
        conn.close()


def test_serve_explicit_cors_origins_argument() -> None:
    """serve() also accepts cors_origins directly (overrides settings)."""
    port = _free_port()
    serve(
        _context(None),
        host="127.0.0.1",
        port=port,
        background=True,
        protocols=False,
        cors_origins=[TEST_ORIGIN],
    )
    _wait_up(port)
    resp = _get(port, "/live", TEST_ORIGIN)
    try:
        body = resp.read()
        assert resp.status == 200, body
        assert resp.getheader("Access-Control-Allow-Origin") == TEST_ORIGIN
    finally:
        resp.close()
