"""Tests for the SSE stream server."""
from __future__ import annotations

import json
import socket
import threading
import time
import unittest
import urllib.request
import urllib.error

from nomorals.os.timeline import Timeline
from nomorals.stream import StreamServer


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class StreamServerTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = self._tmp.name
        self.timeline = Timeline(self.db_path)
        self.port = _free_port()
        self.server = StreamServer(
            lambda: Timeline(self.db_path), host="127.0.0.1", port=self.port
        )
        self.server.start(background=True)
        # Wait for the socket to accept.
        for _ in range(50):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=1
                ).read()
                break
            except OSError:
                time.sleep(0.05)

    def tearDown(self):
        self.server.stop()
        self.timeline.close()
        import os
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_health(self):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/health", timeout=5
        ) as r:
            body = json.loads(r.read())
        self.assertTrue(body["ok"])
        self.assertIn("version", body)

    def test_404(self):
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/nope", timeout=5
            )
            self.fail("expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_bad_since_400(self):
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/stream?since=abc", timeout=5
            )
            self.fail("expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_stream_receives_events(self):
        import socket
        # Record an event. Timeline expects an object with attributes.
        from types import SimpleNamespace
        self.timeline.record(SimpleNamespace(
            topic="test.event",
            session_id="s1",
            data={"hello": "world"},
        ))
        # Connect with a raw socket and read the SSE stream.
        s = socket.socket()
        s.settimeout(8)
        s.connect(("127.0.0.1", self.port))
        s.sendall(b"GET /stream?since=0 HTTP/1.1\r\nHost: localhost\r\n\r\n")
        buf = b""
        # Read until the recorded event arrives or we time out. (The
        # stream now opens with `retry:` + an `event: ready` handshake
        # before data frames, so wait for the event itself.)
        deadline = time.time() + 8
        try:
            while b"test.event" not in buf and time.time() < deadline:
                try:
                    chunk = s.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                buf += chunk
        finally:
            s.close()
        self.assertIn(b"test.event", buf, "expected SSE event in stream")

    def test_double_start_raises(self):
        with self.assertRaises(RuntimeError):
            self.server.start(background=True)


if __name__ == "__main__":
    unittest.main()
