"""Timeline wiring for the browser organ (Wave K).

Attaches a :class:`~nomorals.os.timeline.Timeline` to the process bus and
asserts the browser service's ``browser.session.*`` / ``browser.tab.*`` /
``browser.download.*`` events land in it.
"""

from __future__ import annotations

import http.server
import os
import shutil
import tempfile
import threading
import unittest

from nomorals.browser import BrowserError, BrowserService
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


class BrowserTimelineTestCase(unittest.TestCase):
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
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.svc = BrowserService(data_dir=os.path.join(self.tmp, "data"))
        self.timeline = Timeline()  # in-memory
        self.addCleanup(self.timeline.close)
        self.timeline.attach(global_bus, sync=True)
        self.addCleanup(self.timeline.detach, global_bus)

    def _events(self, topic):
        return self.timeline.query(topic=topic, limit=50)

    def test_session_open_and_close(self):
        self.svc.open_session("s1")
        opened = self._events("browser.session.opened")
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["data"]["session"], "s1")
        self.assertEqual(opened[0]["source"], "nomorals.browser.service")

        self.svc.close_session("s1")
        closed = self._events("browser.session.closed")
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["data"]["session"], "s1")

    def test_tab_navigated_records_url_and_title(self):
        handle = self.svc.open_session("nav")
        tab = handle.open_tab(f"{self.base}/")
        rows = self._events("browser.tab.navigated")
        self.assertEqual(len(rows), 1)
        data = rows[0]["data"]
        self.assertEqual(data["session"], "nav")
        self.assertEqual(data["tab_id"], tab.tab_id)
        self.assertEqual(data["url"], f"{self.base}/")
        self.assertEqual(data["title"], "Home")

    def test_download_completed(self):
        result = self.svc.download(f"{self.base}/file.bin")
        rows = self._events("browser.download.completed")
        self.assertEqual(len(rows), 1)
        data = rows[0]["data"]
        self.assertEqual(data["url"], f"{self.base}/file.bin")
        self.assertEqual(data["path"], result.path)
        self.assertEqual(data["size"], len(_BINARY))
        self.assertEqual(data["mime"], "application/octet-stream")

    def test_failed_operation_emits_nothing(self):
        with self.assertRaises(BrowserError):
            self.svc.close_session("nope")
        self.assertEqual(self._events("browser.session.closed"), [])

    def test_no_timeline_works_fine(self):
        # Fail-open telemetry: browsing must work with no subscriber.
        self.timeline.detach(global_bus)
        self.svc.open_session("quiet")
        self.svc.close_session("quiet")


if __name__ == "__main__":
    unittest.main()
