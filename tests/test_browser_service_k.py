"""Browser service (nomorals/browser): sessions, tabs, history, downloads,
persistence round-trip, screenshot fail-fast."""

import http.server
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from nomorals.browser import (
    BrowserError,
    BrowserService,
    DownloadResult,
    ScreenshotResult,
    SessionHandle,
    Tab,
)
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database

BINARY_PAYLOAD = b"\x00\x01\x02B1N4RY-payload" * 64


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102 - silence test server noise
        pass

    def do_GET(self):  # noqa: D102
        if self.path == "/":
            body = (
                b"<html><head><title>Home</title></head><body>"
                b"<h1>Home Heading</h1>"
                b"<a href='/page2'>go to page two</a> "
                b"<a href='/file.bin'>binary file</a>"
                b"</body></html>"
            )
            self._send(200, body, "text/html; charset=utf-8")
        elif self.path == "/page2":
            self._send(200, b"<html><body><h1>Page Two</h1></body></html>",
                       "text/html; charset=utf-8")
        elif self.path == "/file.bin":
            self._send(200, BINARY_PAYLOAD, "application/octet-stream")
        else:
            self._send(404, b"nope", "text/plain")

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _make_store(tmpdir):
    db = Database(":memory:")
    db.migrate()
    blobs = BlobStore(db, Path(tmpdir) / "blobs")
    return ArtifactStore(db, blobs), db


class TestBrowserService(unittest.TestCase):
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

    # -- sessions & tabs ------------------------------------------------------
    def test_open_session_open_tab_navigate(self):
        handle = self.svc.open_session("s1")
        self.assertIsInstance(handle, SessionHandle)
        tab = handle.open_tab(f"{self.base}/")
        self.assertIsInstance(tab, Tab)
        self.assertEqual(handle.active_tab.tab_id, tab.tab_id)
        self.assertEqual(tab.url, f"{self.base}/")
        self.assertEqual(len(tab.history), 1)

    def test_open_session_duplicate_raises(self):
        self.svc.open_session("dup")
        with self.assertRaises(BrowserError):
            self.svc.open_session("dup")

    def test_get_session_unknown_raises(self):
        with self.assertRaises(BrowserError):
            self.svc.get_session("nope")

    def test_two_tabs_switch_and_list(self):
        handle = self.svc.open_session("tabs")
        t1 = handle.open_tab(f"{self.base}/")
        t2 = handle.open_tab(f"{self.base}/page2")
        self.assertEqual(handle.active_tab.tab_id, t2.tab_id)
        got = handle.switch_tab(t1.tab_id)
        self.assertEqual(got.tab_id, t1.tab_id)
        self.assertEqual(handle.active_tab.tab_id, t1.tab_id)
        listed = handle.list_tabs()
        self.assertEqual({t["tab_id"] for t in listed}, {t1.tab_id, t2.tab_id})
        self.assertEqual(len(listed), 2)

    def test_switch_unknown_tab_raises(self):
        handle = self.svc.open_session("sw")
        handle.open_tab(f"{self.base}/")
        with self.assertRaises(BrowserError):
            handle.switch_tab("bogus-tab-id")

    def test_close_tab_unknown_raises(self):
        handle = self.svc.open_session("ct")
        handle.open_tab(f"{self.base}/")
        with self.assertRaises(BrowserError):
            handle.close_tab("bogus-tab-id")

    def test_close_tab_moves_active(self):
        handle = self.svc.open_session("ca")
        t1 = handle.open_tab(f"{self.base}/")
        t2 = handle.open_tab(f"{self.base}/page2")
        handle.close_tab(t2.tab_id)
        self.assertEqual(handle.active_tab.tab_id, t1.tab_id)
        self.assertEqual(len(handle.list_tabs()), 1)

    def test_navigate_no_tabs_raises(self):
        handle = self.svc.open_session("empty")
        with self.assertRaises(BrowserError):
            handle.navigate(f"{self.base}/")

    def test_navigate_404_fails_fast(self):
        handle = self.svc.open_session("nf")
        tab = handle.open_tab()
        with self.assertRaises(BrowserError):
            tab.navigate(f"{self.base}/missing")

    # -- page work --------------------------------------------------------------
    def test_text_and_markdown(self):
        handle = self.svc.open_session("pg")
        tab = handle.open_tab(f"{self.base}/")
        text = tab.text()["text"]
        self.assertIn("Home Heading", text)
        md = tab.markdown()["markdown"]
        self.assertIn("# Home Heading", md)

    def test_links(self):
        handle = self.svc.open_session("lk")
        tab = handle.open_tab(f"{self.base}/")
        links = tab.links()["links"]
        urls = [link["url"] for link in links]
        self.assertIn(f"{self.base}/page2", urls)
        self.assertIn(f"{self.base}/file.bin", urls)

    def test_history_merged_across_tabs_sorted(self):
        handle = self.svc.open_session("hist")
        t1 = handle.open_tab(f"{self.base}/")
        t1.navigate(f"{self.base}/page2")
        t2 = handle.open_tab(f"{self.base}/")
        merged = handle.history()
        urls = [entry["url"] for entry in merged]
        self.assertEqual(
            urls, [f"{self.base}/", f"{self.base}/page2", f"{self.base}/"])
        self.assertEqual(merged[0]["tab_id"], t1.tab_id)
        self.assertEqual(merged[2]["tab_id"], t2.tab_id)
        ts = [entry["ts"] for entry in merged]
        self.assertEqual(ts, sorted(ts))

    def test_back(self):
        handle = self.svc.open_session("bk")
        tab = handle.open_tab(f"{self.base}/")
        tab.navigate(f"{self.base}/page2")
        self.assertEqual(tab.url, f"{self.base}/page2")
        tab.back()
        self.assertEqual(tab.url, f"{self.base}/")
        self.assertEqual(len(tab.history), 1)

    def test_back_without_history_raises(self):
        handle = self.svc.open_session("bk2")
        tab = handle.open_tab(f"{self.base}/")
        with self.assertRaises(BrowserError):
            tab.back()

    # -- downloads ---------------------------------------------------------------
    def test_download_byte_identical_and_artifact(self):
        store, db = _make_store(self.tmp)
        self.addCleanup(db.close)
        svc = BrowserService(data_dir=os.path.join(self.tmp, "dl"),
                             artifact_store=store, mission_id="m1")
        handle = svc.open_session("dl-sess")
        tab = handle.open_tab(f"{self.base}/")
        result = svc.download(tab, f"{self.base}/file.bin")
        self.assertIsInstance(result, DownloadResult)
        data = Path(result.path).read_bytes()
        self.assertEqual(data, BINARY_PAYLOAD)
        self.assertEqual(result.size, len(BINARY_PAYLOAD))
        self.assertEqual(result.mime, "application/octet-stream")
        self.assertIsNotNone(result.artifact_uri)
        art = store.resolve(result.artifact_uri)
        self.assertIsNotNone(art)
        self.assertEqual(art.type, "download")
        self.assertEqual(art.mission_id, "m1")
        self.assertEqual(art.provenance.source_type, "browser")
        self.assertEqual(art.provenance.source_id, f"{self.base}/file.bin")
        self.assertEqual(store.read(art.id), BINARY_PAYLOAD)
        self.assertEqual(result.to_dict()["artifact_uri"], result.artifact_uri)

    def test_download_url_only_no_tab(self):
        result = self.svc.download(f"{self.base}/file.bin")
        self.assertEqual(Path(result.path).read_bytes(), BINARY_PAYLOAD)
        self.assertIsNone(result.artifact_uri)

    def test_download_404_raises(self):
        handle = self.svc.open_session("d404")
        tab = handle.open_tab(f"{self.base}/")
        with self.assertRaises(BrowserError):
            self.svc.download(tab, f"{self.base}/missing")

    def test_download_bad_scheme_raises(self):
        with self.assertRaises(BrowserError):
            self.svc.download("ftp://example.com/file.bin")

    # -- persistence --------------------------------------------------------------
    def test_save_restore_round_trip(self):
        data_dir = os.path.join(self.tmp, "persist")
        svc1 = BrowserService(data_dir=data_dir)
        handle = svc1.open_session("work")
        t1 = handle.open_tab(f"{self.base}/")
        t1.navigate(f"{self.base}/page2")
        handle.open_tab(f"{self.base}/")
        handle.switch_tab(t1.tab_id)
        path = svc1.save()
        self.assertTrue(Path(path).is_file())
        self.assertIn("sessions.json", path)

        svc2 = BrowserService(data_dir=data_dir)
        restored = svc2.restore()
        self.assertEqual(restored, 1)
        self.assertEqual(svc2.list_sessions(), ["work"])
        h2 = svc2.get_session("work")
        self.assertEqual(len(h2.list_tabs()), 2)
        self.assertEqual(h2.active_tab.tab_id, t1.tab_id)
        self.assertEqual(h2.active_tab.url, f"{self.base}/page2")
        self.assertEqual(len(h2.active_tab.history), 2)
        for tab in (h2.active_tab,):
            self.assertEqual(tab.error, "")
        # cookie dirs were re-wired for the restored session
        self.assertTrue(
            Path(data_dir, "cookies", "work").is_dir())
        self.assertEqual([e["url"] for e in h2.history()],
                         [f"{self.base}/", f"{self.base}/page2", f"{self.base}/"])

    def test_restore_without_save_is_noop(self):
        svc = BrowserService(data_dir=os.path.join(self.tmp, "nosave"))
        self.assertEqual(svc.restore(), 0)

    def test_close_session_saves_and_removes(self):
        svc = BrowserService(data_dir=os.path.join(self.tmp, "close"))
        handle = svc.open_session("gone")
        handle.open_tab(f"{self.base}/")
        svc.close_session("gone")
        self.assertEqual(svc.list_sessions(), [])
        self.assertTrue(Path(self.tmp, "close", "sessions.json").is_file())
        with self.assertRaises(BrowserError):
            svc.get_session("gone")

    def test_close_session_unknown_raises(self):
        with self.assertRaises(BrowserError):
            self.svc.close_session("ghost")

    # -- attach / mission ---------------------------------------------------------
    def test_attach_store_and_set_mission(self):
        store, db = _make_store(self.tmp)
        self.addCleanup(db.close)
        self.svc.attach_store(store)
        self.svc.set_mission("mission-9")
        result = self.svc.download(f"{self.base}/file.bin")
        art = store.resolve(result.artifact_uri)
        self.assertEqual(art.mission_id, "mission-9")
        self.assertEqual(art.creator, "browser-service")

    # -- screenshots ----------------------------------------------------------------
    def test_screenshot_fail_fast_without_playwright(self):
        import importlib.util

        handle = self.svc.open_session("shot")
        tab = handle.open_tab(f"{self.base}/")
        if importlib.util.find_spec("playwright") is None:
            with self.assertRaises(BrowserError) as ctx:
                self.svc.screenshot(tab)
            self.assertIn("playwright", str(ctx.exception))
        else:
            result = self.svc.screenshot(tab)
            self.assertIsInstance(result, ScreenshotResult)
            self.assertTrue(
                Path(result.path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_screenshot_no_url_raises(self):
        handle = self.svc.open_session("shot2")
        tab = handle.open_tab()
        with self.assertRaises(BrowserError):
            self.svc.screenshot(tab)

    def test_screenshot_unknown_tab_raises(self):
        self.svc.open_session("shot3")
        with self.assertRaises(BrowserError):
            self.svc.screenshot("no-such-tab")


if __name__ == "__main__":
    unittest.main()
