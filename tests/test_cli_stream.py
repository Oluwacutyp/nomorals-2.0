"""``nm stream`` (SSE timeline stream) and ``nm serve --port/--host``.

Covers the surface-1.0 CLI additions:

* ``nm stream`` command: parser, alias, dispatch routing, serve/status
  behavior.
* ``nm serve --port/--host`` overrides reaching ``nomorals.api.server.serve``.
* ``_cmd_stream`` exported through the ``nomorals.cli`` façade.
"""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nomorals.cli as cli
from nomorals.cmdline.commands.serve import _cmd_serve
from nomorals.cmdline.commands.stream import (
    _cmd_stream,
    _cmd_stream_serve,
    _cmd_stream_status,
)
from nomorals.cmdline.dispatch import _canonical_command
from nomorals.cmdline.parser import _parser


def _args(argv):
    return _parser().parse_args(argv)


def _ctx(**kw):
    base = {"db": None, "extras": {}}
    base.update(kw)
    return SimpleNamespace(**base)


class TestStreamParser(unittest.TestCase):
    def test_command_exists_with_defaults(self):
        args = _args(["stream"])
        self.assertEqual(args.command, "stream")
        self.assertEqual(args.action, "serve")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8899)

    def test_status_action_and_overrides(self):
        args = _args(["stream", "status", "--host", "0.0.0.0", "--port", "9000"])
        self.assertEqual(args.action, "status")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)

    def test_alias(self):
        args = _args(["strm", "status"])
        self.assertEqual(_canonical_command(args.command), "stream")
        self.assertEqual(_canonical_command("strm"), "stream")

    def test_serve_port_host_flags(self):
        args = _args(["serve", "--port", "8123", "--host", "0.0.0.0"])
        self.assertEqual(args.command, "serve")
        self.assertEqual(args.port, 8123)
        self.assertEqual(args.host, "0.0.0.0")
        args = _args(["serve"])
        self.assertIsNone(args.port)
        self.assertIsNone(args.host)


class TestServeOverrides(unittest.TestCase):
    def _settings_ctx(self):
        api = SimpleNamespace(host="127.0.0.1", port=8787)
        return SimpleNamespace(settings=SimpleNamespace(api=api), db=None, extras={})

    def test_port_and_host_override_reach_api_serve(self):
        args = _args(["serve", "--port", "8123", "--host", "0.0.0.0"])
        with mock.patch("nomorals.api.server.serve", return_value=0) as m:
            rc = _cmd_serve(args, self._settings_ctx())
        self.assertEqual(rc, 0)
        m.assert_called_once()
        _, kwargs = m.call_args
        self.assertEqual(kwargs["port"], 8123)
        self.assertEqual(kwargs["host"], "0.0.0.0")

    def test_defaults_come_from_settings(self):
        args = _args(["serve"])
        with mock.patch("nomorals.api.server.serve", return_value=0) as m:
            rc = _cmd_serve(args, self._settings_ctx())
        self.assertEqual(rc, 0)
        _, kwargs = m.call_args
        self.assertEqual(kwargs["port"], 8787)
        self.assertEqual(kwargs["host"], "127.0.0.1")


class TestStreamStatus(unittest.TestCase):
    def test_unreachable_server_fails_fast(self):
        args = _args(["stream", "status", "--port", "1"])
        err = io.StringIO()
        with redirect_stderr(err):
            rc = _cmd_stream_status(args, _ctx())
        self.assertEqual(rc, 1)
        self.assertIn("no server", err.getvalue())

    def test_live_server_reports_up(self):
        from nomorals.stream import StreamServer

        class FakeTimeline:
            def query(self, **kw):
                return [{"ts": 1.0, "topic": "mission.x"}]

            def close(self):
                pass

        srv = StreamServer(lambda: FakeTimeline(), host="127.0.0.1", port=0)
        srv.start(background=True)
        try:
            args = _args(["stream", "status", "--port", str(srv.port)])
            out = io.StringIO()
            with redirect_stdout(out):
                rc = _cmd_stream_status(args, _ctx())
            self.assertEqual(rc, 0)
            self.assertIn("up", out.getvalue())
            self.assertIn(str(srv.port), out.getvalue())
        finally:
            srv.stop()

    def test_live_server_json(self):
        import json

        from nomorals.stream import StreamServer

        class FakeTimeline:
            def query(self, **kw):
                return []

            def close(self):
                pass

        srv = StreamServer(lambda: FakeTimeline(), host="127.0.0.1", port=0)
        srv.start(background=True)
        try:
            args = _args(["stream", "status", "--port", str(srv.port), "--json"])
            out = io.StringIO()
            with redirect_stdout(out):
                rc = _cmd_stream_status(args, _ctx())
            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["running"])
            self.assertEqual(payload["port"], srv.port)
        finally:
            srv.stop()


class TestStreamServe(unittest.TestCase):
    def test_serve_needs_db_path(self):
        args = _args(["stream"])
        err = io.StringIO()
        with redirect_stderr(err):
            rc = _cmd_stream_serve(args, _ctx(db=None))
        self.assertEqual(rc, 2)
        self.assertIn("no database path", err.getvalue())

    def test_serve_wires_host_port_and_timeline_factory(self):
        args = _args(["stream", "--port", "8898"])
        db = SimpleNamespace(path="/tmp/fake-nomorals.db")
        seen = {}

        class FakeServer:
            def __init__(self, factory, host, port):
                seen["host"] = host
                seen["port"] = port
                seen["factory"] = factory
                self.url = f"http://{host}:{port}"

            def start(self, background):
                seen["background"] = background

            def stop(self):
                seen["stopped"] = True

        out = io.StringIO()
        with (
            mock.patch("nomorals.stream.StreamServer", FakeServer),
            redirect_stdout(out),
        ):
            rc = _cmd_stream(args, _ctx(db=db))
        self.assertEqual(rc, 0)
        self.assertEqual(seen["host"], "127.0.0.1")
        self.assertEqual(seen["port"], 8898)
        self.assertFalse(seen["background"])  # foreground: blocks until Ctrl-C
        self.assertTrue(seen["stopped"])
        self.assertIsNotNone(seen["factory"])
        self.assertIn("/stream", out.getvalue())

    def test_unknown_action_fails_fast(self):
        args = _args(["stream"])
        args.action = "bogus"
        err = io.StringIO()
        with redirect_stderr(err):
            rc = _cmd_stream(args, _ctx())
        self.assertEqual(rc, 2)
        self.assertIn("unknown stream action", err.getvalue())


class TestStreamDispatchAndFacade(unittest.TestCase):
    def test_dispatch_routes_stream(self):
        src = open(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "nomorals", "cmdline", "dispatch.py")
            )
        ).read()
        self.assertIn('_cmd_stream', src)
        self.assertIn('args.command == "stream"', src)

    def test_facade_exports_cmd_stream(self):
        self.assertTrue(hasattr(cli, "_cmd_stream"))
        from nomorals.cmdline import _cmd_stream as via_cmdline
        from nomorals.cmdline.commands import _cmd_stream as via_commands

        self.assertIs(cli._cmd_stream, via_cmdline)
        self.assertIs(cli._cmd_stream, via_commands)


if __name__ == "__main__":
    unittest.main()
