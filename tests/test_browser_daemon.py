"""Browser daemon (nomorals/browser/daemon.py): lifecycle, persistence across
CLI-equivalent invocations, timeline-event parity with the default flow."""

from __future__ import annotations

import http.server
import io
import json
import os
import shutil
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from nomorals.browser.daemon import (
    DaemonClient,
    DaemonControl,
    DaemonError,
    _pid_alive,
    republish_events,
)
from nomorals.cmdline.commands.browse import _cmd_browse
from nomorals.core.events import global_bus
from nomorals.os.timeline import Timeline

_BINARY = b"\x00\x01\x02binary-payload" * 64


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102 - silence test server noise
        pass

    def do_GET(self):  # noqa: D102
        if self.path == "/":
            body = (b"<html><head><title>Home</title></head><body>"
                    b"<h1>Home Heading</h1></body></html>")
            self._send(200, body, "text/html; charset=utf-8")
        elif self.path == "/file.bin":
            self._send(200, _BINARY, "application/octet-stream")
        else:
            self._send(404, b"nope", "text/plain")

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DaemonTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old_env = os.environ.get("NOMORALS_BROWSER_DIR")
        os.environ["NOMORALS_BROWSER_DIR"] = os.path.join(self.tmp, "browser")
        # LIFO: stop the daemon first (needs the env var), then drop the var.
        self.addCleanup(self._restore_env)
        self.addCleanup(self._stop_daemon_quiet)
        self.ctl = DaemonControl()

    def _restore_env(self):
        if self._old_env is None:
            os.environ.pop("NOMORALS_BROWSER_DIR", None)
        else:
            os.environ["NOMORALS_BROWSER_DIR"] = self._old_env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stop_daemon_quiet(self):
        try:
            DaemonControl().stop()
        except Exception:  # noqa: BLE001 - cleanup must not fail the test
            pass

    # -- helpers -----------------------------------------------------------
    def _args(self, *task, session="", json=False, out=""):
        return SimpleNamespace(task=list(task), session=session,
                               json=json, out=out)

    def _ctx(self):
        return SimpleNamespace(db=None)

    def _browse(self, *task, **kwargs):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_browse(self._args(*task, **kwargs), self._ctx())
        return rc, buf.getvalue()

    # -- lifecycle -----------------------------------------------------------
    def test_start_stop_status_lifecycle(self):
        self.assertFalse(self.ctl.running())
        info = self.ctl.start()
        self.assertTrue(info["started"])
        pid = info["pid"]
        self.assertTrue(_pid_alive(pid))
        self.assertTrue(self.ctl.running())

        st = self.ctl.status()
        self.assertTrue(st["running"])
        self.assertTrue(st.get("responsive", False))
        self.assertEqual(st["pid"], pid)

        # Second start is idempotent: same daemon, no second process.
        again = self.ctl.start()
        self.assertTrue(again.get("already"))
        self.assertEqual(again["pid"], pid)

        stopped = self.ctl.stop()
        self.assertTrue(stopped["stopped"])
        self.assertEqual(stopped["pid"], pid)

        st = self.ctl.status()
        self.assertFalse(st["running"])
        self.assertFalse(_pid_alive(pid), "daemon process must be gone after stop")
        self.assertFalse(self.ctl.pid_path.exists(), "pid file must be removed")
        self.assertFalse(self.ctl.sock_path.exists(), "socket file must be removed")

    def test_stop_when_not_running_is_idempotent(self):
        self.assertFalse(self.ctl.running())
        info = self.ctl.stop()
        self.assertFalse(info["stopped"])
        self.assertEqual(info["reason"], "not running")

    def test_daemon_survives_a_bad_op(self):
        self.ctl.start()
        client = self.ctl.client()
        with self.assertRaises(DaemonError):
            client.call("nope-not-an-op", {})
        # Still alive and answering.
        self.assertTrue(self.ctl.running())
        pong = client.ping()
        self.assertIn("pid", pong)

    def test_client_errors_without_daemon(self):
        with self.assertRaises(DaemonError):
            DaemonClient(timeout=2.0).call("ping", {}, timeout=2.0)
        with self.assertRaises(DaemonError):
            self.ctl.client()

    # -- persistence across invocations --------------------------------------
    def test_session_persists_across_clients_without_restore(self):
        self.ctl.start()
        first = self.ctl.client()
        result, _ = first.call("open", {"session": "persist",
                                       "url": f"{self.base}/"})
        tab_id = result["tab"]["tab_id"]
        self.assertEqual(result["tab"]["url"], f"{self.base}/")
        del first  # drop the client entirely, like a CLI exit

        second = self.ctl.client()
        result, _ = second.call("tabs", {"session": "persist"})
        tab_ids = [t["tab_id"] for t in result["tabs"]]
        self.assertIn(tab_id, tab_ids)

        # The live tab still serves the page — no re-navigation needed.
        result, _ = second.call("read", {"session": "persist", "kind": "text"})
        self.assertIn("Home Heading", result["data"].get("text", ""))

    # -- timeline parity -------------------------------------------------------
    def test_events_republished_to_timeline(self):
        timeline = Timeline()  # in-memory
        timeline.attach(global_bus, sync=True)
        self.addCleanup(timeline.detach, global_bus)
        self.addCleanup(timeline.close)

        self.ctl.start()
        client = self.ctl.client()

        def events_for(topic):
            return timeline.query(topic=topic, limit=50)

        _, events = client.call("open", {"session": "evts",
                                        "url": f"{self.base}/"})
        republished = republish_events(events)
        self.assertGreater(republished, 0)

        opened = events_for("browser.session.opened")
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["data"]["session"], "evts")
        navigated = events_for("browser.tab.navigated")
        self.assertEqual(len(navigated), 1)
        self.assertEqual(navigated[0]["data"]["url"], f"{self.base}/")
        self.assertEqual(navigated[0]["data"]["title"], "Home")

        _, events = client.call("download", {"session": "evts",
                                            "url": f"{self.base}/file.bin"})
        republish_events(events)
        done = events_for("browser.download.completed")
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["data"]["size"], len(_BINARY))

        _, events = client.call("close", {"session": "evts"})
        republish_events(events)
        closed = events_for("browser.session.closed")
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["data"]["session"], "evts")

    # -- CLI routing -----------------------------------------------------------
    def test_cli_falls_back_to_local_flow_without_daemon(self):
        self.assertFalse(self.ctl.running())
        rc, out = self._browse("sessions", json=True)
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), [])

        rc, out = self._browse("daemon", "status", json=True)
        self.assertEqual(rc, 0)
        self.assertFalse(json.loads(out)["running"])

    def test_cli_daemon_verbs(self):
        rc, out = self._browse("daemon", "start")
        self.assertEqual(rc, 0)
        self.assertIn("started", out)

        rc, out = self._browse("daemon", "status")
        self.assertEqual(rc, 0)
        self.assertIn("running", out)

        rc, out = self._browse("daemon", "stop")
        self.assertEqual(rc, 0)
        self.assertIn("stopped", out)

    def test_cli_end_to_end_through_daemon(self):
        self._browse("daemon", "start")
        rc, out = self._browse("open", f"{self.base}/", session="e2e", json=True)
        self.assertEqual(rc, 0)
        tab = json.loads(out)
        self.assertEqual(tab["url"], f"{self.base}/")

        rc, out = self._browse("tabs", session="e2e", json=True)
        self.assertEqual(rc, 0)
        self.assertIn(tab["tab_id"], [t["tab_id"] for t in json.loads(out)])

        rc, out = self._browse("text", session="e2e")
        self.assertEqual(rc, 0)
        self.assertIn("Home Heading", out)

        rc, out = self._browse("history", session="e2e", json=True)
        self.assertEqual(rc, 0)
        self.assertTrue(any(e["url"] == f"{self.base}/" for e in json.loads(out)))

        rc, out = self._browse("close", session="e2e")
        self.assertEqual(rc, 0)
        self.assertIn("closed", out)

    def test_cli_daemon_mode_records_timeline(self):
        timeline = Timeline()
        timeline.attach(global_bus, sync=True)
        self.addCleanup(timeline.detach, global_bus)
        self.addCleanup(timeline.close)

        self._browse("daemon", "start")
        rc, _ = self._browse("open", f"{self.base}/", session="tl")
        self.assertEqual(rc, 0)
        rows = timeline.query(topic="browser.tab.navigated", limit=50)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["data"]["session"], "tl")


if __name__ == "__main__":
    unittest.main()
