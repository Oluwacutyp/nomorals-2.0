"""Replay harness: ONE record/replay mechanism for LLM + L2 network.

* :class:`GenericCassetteStore` — first-write-wins, canonical keys.
* LLM cassettes delegate to the generic store (same file, same format).
* Network cassettes: record a live fetch, replay it byte-identically
  with the server gone (zero network I/O); a replay miss is a denial,
  never a live fetch.
"""

from __future__ import annotations

import http.server
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from nomorals.core.cassette import GenericCassetteStore, canonical_key
from nomorals.core.egress import (
    EgressClient,
    EgressController,
    EgressDenied,
    EgressToken,
)
from nomorals.llm.base import Message
from nomorals.llm.cassette import CassetteStore as LLMCassetteStore
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.router import LLMRouter


class _TestServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, body, headers=None):
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        self.server.hits[self.path] = self.server.hits.get(self.path, 0) + 1
        if self.path == "/hello":
            self._send(200, b"hello-l2", {"X-Test": "yes"})
        else:
            self._send(404, b"not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.hits["POST " + self.path] = \
            self.server.hits.get("POST " + self.path, 0) + 1
        self._send(200, b"echo:" + body)


class ReplayHarnessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-replay-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self._server = _TestServer(("127.0.0.1", 0), _Handler)
        self._server.hits = {}
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
            daemon=True)
        self._thread.start()
        self.port = self._server.server_address[1]
        self.addCleanup(self._server.shutdown)
        self.addCleanup(self._server.server_close)

    def _token(self, **kw):
        kw.setdefault("domains", ("localhost",))
        kw.setdefault("allow_private_ips", True)
        return EgressToken.mint("task-replay", **kw)

    def _client(self, mode, store):
        return EgressClient(controller=EgressController(), cassette=store,
                             cassette_mode=mode)

    # ── generic store ──
    def test_canonical_key_deterministic_and_kind_separated(self):
        a = canonical_key("net", {"method": "GET", "url": "http://x/"})
        b = canonical_key("net", {"url": "http://x/", "method": "GET"})
        c = canonical_key("llm", {"method": "GET", "url": "http://x/"})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_first_write_wins(self):
        store = GenericCassetteStore(self.tmp / "g")
        self.assertTrue(store.record("k", {"v": 1}))
        self.assertFalse(store.record("k", {"v": 2}))
        self.assertEqual(store.lookup("k")["v"], 1)
        self.assertEqual(len(store), 1)

    # ── LLM cassettes ride the generic store ──
    def test_llm_cassette_is_the_generic_store(self):
        router = LLMRouter(cassette_dir=str(self.tmp / "llm"))
        provider = MockProvider(model="mock-7b")
        router.add(provider, name="mock", primary=True)
        first = router.chat([Message.user("hello replay")])
        self.assertTrue(first.ok)
        # The same file the router wrote is readable as a generic store,
        # and the LLM record is in it — one mechanism, not two.
        generic = GenericCassetteStore(self.tmp / "llm")
        self.assertEqual(len(generic), 1)
        record = next(generic.iter_records())
        self.assertEqual(record["provider"], "mock")
        self.assertEqual(record["response"]["text"], first.text)
        # And the LLM store itself can read what the generic store wrote.
        llm_store = LLMCassetteStore(self.tmp / "llm")
        self.assertIsNotNone(llm_store.lookup(record["key"]))

    # ── network cassettes: record → replay byte-identical, zero network ──
    def test_network_record_then_replay_byte_identical(self):
        store = GenericCassetteStore(self.tmp / "net",
                                     filename="network_cassettes.jsonl")
        url = f"http://localhost:{self.port}/hello"
        token = self._token()
        recorded = self._client("record", store).get(token, url)
        self.assertEqual(recorded.status, 200)
        self.assertEqual(recorded.body, b"hello-l2")
        self.assertEqual(recorded.headers.get("x-test"), "yes")
        self.assertFalse(recorded.replayed)
        self.assertEqual(self._server.hits.get("/hello"), 1)

        replay = self._client("replay", store)
        again = replay.get(self._token(), url)
        self.assertTrue(again.replayed)
        self.assertEqual(again.status, recorded.status)
        self.assertEqual(again.headers, recorded.headers)
        self.assertEqual(again.body, recorded.body)
        self.assertEqual(again.url, recorded.url)
        # No second hit: replay performed zero network I/O.
        self.assertEqual(self._server.hits.get("/hello"), 1)

    def test_network_replay_after_server_gone(self):
        store = GenericCassetteStore(self.tmp / "net2",
                                     filename="network_cassettes.jsonl")
        url = f"http://localhost:{self.port}/hello"
        self._client("record", store).get(self._token(), url)
        self._server.shutdown()
        self._server.server_close()
        replay = self._client("replay", store)
        again = replay.get(self._token(), url)
        self.assertTrue(again.replayed)
        self.assertEqual(again.body, b"hello-l2")

    def test_network_replay_miss_is_denial_not_live_fetch(self):
        store = GenericCassetteStore(self.tmp / "net3",
                                     filename="network_cassettes.jsonl")
        url = f"http://localhost:{self.port}/hello"
        with self.assertRaises(EgressDenied):
            self._client("replay", store).get(self._token(), url)
        # Nothing was recorded, nothing was fetched.
        self.assertEqual(len(store), 0)
        self.assertEqual(self._server.hits.get("/hello", 0), 0)

    def test_network_key_separates_post_bodies(self):
        store = GenericCassetteStore(self.tmp / "net4",
                                     filename="network_cassettes.jsonl")
        url = f"http://localhost:{self.port}/echo"
        recorder = self._client("record", store)
        r1 = recorder.fetch(self._token(), "POST", url, data=b"alpha")
        r2 = recorder.fetch(self._token(), "POST", url, data=b"beta")
        self.assertEqual(r1.body, b"echo:alpha")
        self.assertEqual(r2.body, b"echo:beta")
        self.assertEqual(len(store), 2)
        replay = self._client("replay", store)
        self.assertEqual(
            replay.fetch(self._token(), "POST", url, data=b"alpha").body,
            b"echo:alpha")
        self.assertEqual(
            replay.fetch(self._token(), "POST", url, data=b"beta").body,
            b"echo:beta")

    def test_network_cassette_never_records_credentials(self):
        store = GenericCassetteStore(self.tmp / "net5",
                                     filename="network_cassettes.jsonl")
        url = f"http://localhost:{self.port}/hello"
        self._client("record", store).get(
            self._token(), url,
            headers={"Authorization": "Bearer SECRET",
                     "X-Api-Key": "SECRET2", "X-Ok": "kept"})
        record = next(store.iter_records())
        self.assertNotIn("authorization", record["request_headers"])
        self.assertNotIn("x-api-key", record["request_headers"])
        self.assertEqual(record["request_headers"].get("x-ok"), "kept")
        raw = (self.tmp / "net5" / "network_cassettes.jsonl").read_text()
        self.assertNotIn("SECRET", raw)


if __name__ == "__main__":
    unittest.main()
