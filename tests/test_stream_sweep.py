"""Sweep tests for nomorals.stream: SSE framing, EventHub broadcast,
emit_sse, and the StreamServer upgrades (hub, Last-Event-ID resume,
retry handshake, CORS, enriched health, backpressure)."""
from __future__ import annotations

import fnmatch
import http.client
import io
import json
import queue
import socket
import time
import unittest

from nomorals.stream import (
    EventHub,
    ServerSentEvent,
    StreamClosed,
    StreamError,
    StreamServer,
    SubscriberLimitExceeded,
    Subscription,
    emit_sse,
    format_sse,
    serve,
)
from nomorals.stream.errors import StreamClosed as SC2  # noqa: F401


# ── fakes ──────────────────────────────────────────────────────────────

class FakeTimeline:
    """Newest-first Timeline double over a canned/appendable event list."""

    def __init__(self, events=()):
        self._events = list(events)
        self.closed = 0

    def append(self, event):
        self._events.append(event)

    def query(self, *, since=None, topic=None, limit=200, until=None, **kw):
        rows = [e for e in self._events
                if (since is None or e["ts"] >= since)]
        if until is not None:
            rows = [r for r in rows if r["ts"] <= until]
        if topic:
            rows = [r for r in rows
                    if fnmatch.fnmatchcase(r["topic"], topic)]
        rows.sort(key=lambda e: (e["ts"], e["event_id"]), reverse=True)
        return [dict(r) for r in rows[:limit]]

    def close(self):
        self.closed += 1


def _ev(topic, ts, eid=None, **kw):
    d = {"event_id": eid or f"{topic}-{ts}", "ts": ts, "topic": topic}
    d.update(kw)
    return d


def _parse_frames(buf: bytes):
    """Parse an SSE byte stream into frame dicts."""
    frames = []
    for raw in buf.split(b"\n\n"):
        if not raw.strip():
            continue
        frame = {"event": None, "id": None, "retry": None,
                 "data": [], "comment": []}
        for line in raw.split(b"\n"):
            line = line.rstrip(b"\r")
            if line.startswith(b":"):
                frame["comment"].append(line[1:].strip().decode())
            elif line.startswith(b"event:"):
                frame["event"] = line[6:].strip().decode()
            elif line.startswith(b"id:"):
                frame["id"] = line[3:].strip().decode()
            elif line.startswith(b"retry:"):
                frame["retry"] = int(line[6:].strip())
            elif line.startswith(b"data:"):
                frame["data"].append(line[5:].strip().decode())
        frames.append(frame)
    return frames


class ExplodingBytesIO(io.BytesIO):
    """Raises BrokenPipeError after N writes (simulates disconnect)."""

    def __init__(self, explode_after):
        super().__init__()
        self._explode_after = explode_after
        self._writes = 0

    def write(self, b):
        self._writes += 1
        if self._writes > self._explode_after:
            raise BrokenPipeError("client went away")
        return super().write(b)


class FakeSocket:
    def __init__(self):
        self.timeout = None
        self.set_calls: list = []

    def settimeout(self, t):
        self.timeout = t
        self.set_calls.append(t)

    def gettimeout(self):
        return self.timeout


class FakeHandler:
    """Minimal BaseHTTPRequestHandler double for emit_sse."""

    def __init__(self, query, headers=None, explode_after=10**9):
        self._query = query
        self.headers = headers or {}
        self.wfile = ExplodingBytesIO(explode_after)
        self.connection = FakeSocket()
        self.status = None
        self.sent_headers = {}

    def send_response(self, code):
        self.status = code

    def send_header(self, key, value):
        self.sent_headers[key] = value

    def end_headers(self):
        pass


def _drain(sub: Subscription, timeout=5.0):
    """Collect everything currently queued for a subscription."""
    out = []
    end = time.time() + timeout
    while time.time() < end:
        try:
            out.append(sub.get(timeout=0.05))
        except queue.Empty:
            if out:
                break
    return out


# ── SSE framing ──────────────────────────────────────────────────────────

class TestServerSentEvent(unittest.TestCase):
    def test_basic_frame(self):
        raw = ServerSentEvent("hi", event="greet", id="1",
                              retry=3000).encode()
        self.assertEqual(raw, b"event: greet\nid: 1\ndata: hi\nretry: 3000\n\n")

    def test_multiline_data_splits(self):
        raw = ServerSentEvent("a\nb\r\nc\rd", event="x").encode()
        self.assertEqual(raw, b"event: x\ndata: a\ndata: b\ndata: c\ndata: d\n\n")

    def test_multiline_comment_splits(self):
        raw = ServerSentEvent(comment="one\ntwo").encode()
        self.assertEqual(raw, b": one\n: two\n\n")

    def test_id_and_event_newlines_stripped(self):
        raw = ServerSentEvent("d", event="a\nb", id="1\n2").encode()
        self.assertEqual(raw, b"event: ab\nid: 12\ndata: d\n\n")

    def test_retry_must_be_int(self):
        with self.assertRaises(TypeError):
            ServerSentEvent("d", retry="3000")

    def test_bad_sep_rejected(self):
        with self.assertRaises(ValueError):
            ServerSentEvent("d", sep=";")

    def test_custom_sep(self):
        raw = ServerSentEvent("d", event="e", sep="\r\n").encode()
        self.assertEqual(raw, b"event: e\r\ndata: d\r\n\r\n")

    def test_format_sse_shortcut(self):
        self.assertEqual(format_sse("d", event="e"),
                         ServerSentEvent("d", event="e").encode())

    def test_non_string_data_stringified(self):
        raw = ServerSentEvent({"a": 1}, event="e").encode()
        self.assertIn(b"data: {'a': 1}\n\n", raw)


# ── EventHub ─────────────────────────────────────────────────────────────

class TestEventHub(unittest.TestCase):
    def setUp(self):
        self.timeline = FakeTimeline()
        self.hub = EventHub(lambda: self.timeline, poll_interval=0.05)

    def tearDown(self):
        self.hub.stop()

    def test_inject_reaches_subscriber_with_monotonic_seq(self):
        self.hub.start()
        sub = self.hub.subscribe()
        try:
            self.hub.inject("a.b", {"v": 1})
            self.hub.inject("a.b", {"v": 2})
            got = _drain(sub)
            self.assertEqual(len(got), 2)
            (s1, e1), (s2, e2) = got
            self.assertLess(s1, s2)
            self.assertEqual(e1["v"], 1)
            self.assertEqual(e2["v"], 2)
            self.assertEqual(e1["topic"], "a.b")
            self.assertIn("event_id", e1)
            self.assertIn("ts", e1)
        finally:
            sub.close()

    def test_inject_returns_event(self):
        ev = self.hub.inject("t.x", {"k": "v"})
        self.assertEqual(ev["topic"], "t.x")
        self.assertEqual(ev["k"], "v")
        self.assertTrue(ev["event_id"])
        self.assertGreater(ev["ts"], 0)

    def test_topic_glob_filtering(self):
        self.hub.start()
        sub = self.hub.subscribe(topic="mission.*")
        try:
            self.hub.inject("mission.start", {})
            self.hub.inject("chat.msg", {})
            got = _drain(sub)
            self.assertEqual(len(got), 1)
            self.assertEqual(got[0][1]["topic"], "mission.start")
        finally:
            sub.close()

    def test_multi_topic_csv(self):
        self.hub.start()
        sub = self.hub.subscribe(topic="mission.*,chat.*")
        try:
            self.hub.inject("mission.x", {})
            self.hub.inject("chat.y", {})
            self.hub.inject("other.z", {})
            got = _drain(sub)
            topics = sorted(e["topic"] for _, e in got)
            self.assertEqual(topics, ["chat.y", "mission.x"])
        finally:
            sub.close()

    def test_backfill_from_timeline_oldest_first(self):
        self.timeline.append(_ev("t.a", 10.0))
        self.timeline.append(_ev("t.b", 20.0))
        self.timeline.append(_ev("t.c", 30.0))
        sub = self.hub.subscribe(since=0.0)
        try:
            got = _drain(sub)
            self.assertEqual([e["topic"] for _, e in got],
                             ["t.a", "t.b", "t.c"])
        finally:
            sub.close()

    def test_backfill_exactly_once_same_timestamp(self):
        for i in range(5):
            self.timeline.append(_ev(f"dup.{i}", 100.0, eid=f"e{i}"))
        sub = self.hub.subscribe(since=0.0)
        try:
            got = _drain(sub)
            ids = [e["event_id"] for _, e in got]
            self.assertEqual(len(ids), 5)
            self.assertEqual(len(set(ids)), 5)
        finally:
            sub.close()

    def test_backfill_respects_since(self):
        self.timeline.append(_ev("old", 10.0))
        self.timeline.append(_ev("new", 20.0))
        sub = self.hub.subscribe(since=15.0)
        try:
            got = _drain(sub)
            self.assertEqual([e["topic"] for _, e in got], ["new"])
        finally:
            sub.close()

    def test_last_event_id_replay(self):
        self.hub.start()
        s1 = self.hub.subscribe()
        try:
            self.hub.inject("t.a", {"n": 1})
            self.hub.inject("t.a", {"n": 2})
            self.hub.inject("t.a", {"n": 3})
            got = _drain(s1)
            seqs = [s for s, _ in got]
            self.assertEqual(len(seqs), 3)
        finally:
            s1.close()
        # Reconnect after the 2nd event: only the 3rd is replayed.
        s2 = self.hub.subscribe(last_event_id=seqs[1])
        try:
            got2 = _drain(s2)
            self.assertEqual(len(got2), 1)
            self.assertEqual(got2[0][1]["n"], 3)
            self.assertEqual(got2[0][0], seqs[2])
        finally:
            s2.close()

    def test_last_event_id_fully_caught_up_no_dupes(self):
        self.hub.start()
        s1 = self.hub.subscribe()
        try:
            self.hub.inject("t.a", {"n": 1})
            got = _drain(s1)
            last = got[-1][0]
        finally:
            s1.close()
        # Nothing new since: the timeline backfill must not re-deliver
        # the event the client already saw.
        self.timeline.append(_ev("t.a", 5.0, eid="persisted-1"))
        s2 = self.hub.subscribe(last_event_id=last)
        try:
            got2 = _drain(s2, timeout=1.0)
            # persisted-1 predates the ring's newest event; the client is
            # caught up, so nothing is re-delivered.
            self.assertEqual(got2, [])
        finally:
            s2.close()

    def test_backpressure_drop_oldest(self):
        self.hub.start()
        sub = self.hub.subscribe(queue_size=2)
        try:
            for i in range(5):
                self.hub.inject("t.a", {"n": i})
            got = _drain(sub)
            self.assertEqual(sub.dropped, 3)
            self.assertEqual([e["n"] for _, e in got], [3, 4])
        finally:
            sub.close()

    def test_max_subscribers(self):
        hub = EventHub(lambda: self.timeline, max_subscribers=1)
        hub.start()
        try:
            s1 = hub.subscribe()
            with self.assertRaises(SubscriberLimitExceeded):
                hub.subscribe()
            s1.close()
            # Slot freed — subscribing works again.
            s2 = hub.subscribe()
            s2.close()
        finally:
            hub.stop()

    def test_stop_closes_subscriptions(self):
        self.hub.start()
        sub = self.hub.subscribe()
        self.hub.stop()
        with self.assertRaises(StreamClosed):
            sub.get(timeout=1.0)
        with self.assertRaises(StreamClosed):
            self.hub.subscribe()

    def test_unsubscribe(self):
        self.hub.start()
        sub = self.hub.subscribe()
        self.assertEqual(self.hub.subscriber_count, 1)
        sub.close()
        self.assertEqual(self.hub.subscriber_count, 0)

    def test_live_poll_routing(self):
        self.hub.start()
        sub = self.hub.subscribe(since=time.time())
        try:
            self.timeline.append(_ev("live.ev", time.time() + 0.01))
            got = []
            end = time.time() + 5.0
            while time.time() < end and not got:
                got = _drain(sub, timeout=0.2)
            self.assertEqual(len(got), 1)
            self.assertEqual(got[0][1]["topic"], "live.ev")
        finally:
            sub.close()

    def test_stats(self):
        self.hub.start()
        stats = self.hub.stats()
        for key in ("running", "subscribers", "max_subscribers",
                    "events_emitted", "events_dropped", "ring_size",
                    "ring_capacity", "poll_interval", "uptime_s", "epoch"):
            self.assertIn(key, stats)
        self.assertTrue(stats["running"])

    def test_errors_share_base(self):
        self.assertTrue(issubclass(SubscriberLimitExceeded, StreamError))
        self.assertTrue(issubclass(StreamClosed, StreamError))

    def test_double_start_raises(self):
        self.hub.start()
        with self.assertRaises(RuntimeError):
            self.hub.start()


# ── emit_sse ─────────────────────────────────────────────────────────────

class TestEmitSse(unittest.TestCase):
    def setUp(self):
        self.timeline = FakeTimeline([
            _ev("t.a", 10.0),
            _ev("t.b", 20.0),
        ])

    def _run(self, query, headers=None, explode_after=10, hub=None):
        handler = FakeHandler(query, headers, explode_after)
        # Short heartbeat so the exploding writer trips promptly once
        # the queued frames are delivered (instead of idling 60 s).
        emit_sse(handler, lambda: self.timeline, query, hub=hub,
                 heartbeat_interval=0.05)
        return handler

    def test_retry_and_ready_lead(self):
        handler = self._run({"since": ["0"]}, explode_after=4)
        frames = _parse_frames(handler.wfile.getvalue())
        self.assertEqual(frames[0]["retry"], 3000)
        self.assertIsNone(frames[0]["event"])
        self.assertEqual(frames[1]["event"], "ready")
        ready = json.loads(frames[1]["data"][0])
        self.assertIn("hub_epoch", ready)
        self.assertIn("server", ready)

    def test_timeline_frames_follow(self):
        handler = self._run({"since": ["0"]}, explode_after=4)
        frames = _parse_frames(handler.wfile.getvalue())
        data_frames = [f for f in frames if f["event"] == "timeline"]
        self.assertEqual(len(data_frames), 2)
        # Oldest first, each with a hub seq id.
        self.assertEqual(json.loads(data_frames[0]["data"][0])["topic"], "t.a")
        self.assertEqual(json.loads(data_frames[1]["data"][0])["topic"], "t.b")
        self.assertTrue(data_frames[0]["id"])
        self.assertLess(int(data_frames[0]["id"]), int(data_frames[1]["id"]))

    def test_heartbeat_comment_on_idle(self):
        tl = FakeTimeline()
        handler = FakeHandler({"since": ["0"]}, explode_after=3)
        emit_sse(handler, lambda: tl, {"since": ["0"]},
                 heartbeat_interval=0.05)
        frames = _parse_frames(handler.wfile.getvalue())
        comments = [c for f in frames for c in f["comment"]]
        self.assertIn("ping", comments)

    def test_bad_since_400(self):
        handler = self._run({"since": ["abc"]})
        self.assertEqual(handler.status, 400)
        body = json.loads(handler.wfile.getvalue())
        self.assertFalse(body["ok"])

    def test_hub_full_503(self):
        hub = EventHub(lambda: self.timeline, max_subscribers=1)
        hub.start()
        try:
            holder = hub.subscribe()  # occupy the only slot
            handler = FakeHandler({"since": ["0"]})
            emit_sse(handler, lambda: self.timeline, {"since": ["0"]},
                     hub=hub)
            self.assertEqual(handler.status, 503)
            self.assertEqual(handler.sent_headers.get("Retry-After"), "5")
            holder.close()
        finally:
            hub.stop()

    def test_last_event_id_header_resume(self):
        hub = EventHub(lambda: FakeTimeline())
        hub.start()
        try:
            hub.inject("t.a", {"n": 1})
            hub.inject("t.a", {"n": 2})
            probe = hub.subscribe()
            seqs = [s for s, _ in _drain(probe)]
            probe.close()
            handler = FakeHandler(
                {"since": ["0"]},
                headers={"Last-Event-ID": str(seqs[0])},
                explode_after=4,
            )
            emit_sse(handler, lambda: FakeTimeline(),
                     {"since": ["0"]}, hub=hub, heartbeat_interval=0.05)
            frames = _parse_frames(handler.wfile.getvalue())
            data_frames = [f for f in frames if f["event"] == "timeline"]
            self.assertEqual(len(data_frames), 1)
            self.assertEqual(json.loads(data_frames[0]["data"][0])["n"], 2)
        finally:
            hub.stop()

    def test_cors_header_sent(self):
        handler = self._run({"since": ["0"]}, explode_after=3)
        self.assertEqual(
            handler.sent_headers.get("Access-Control-Allow-Origin"), "*")

    def test_send_timeout_applied(self):
        handler = self._run({"since": ["0"]}, explode_after=3)
        # Applied during the stream...
        self.assertIn(10.0, handler.connection.set_calls)
        # ...and restored afterwards (the socket may be reused).
        self.assertIsNone(handler.connection.timeout)


# ── StreamServer over real HTTP ──────────────────────────────────────────

class TestStreamServerHttp(unittest.TestCase):
    def setUp(self):
        self.timeline = FakeTimeline()
        self.server = StreamServer(
            lambda: self.timeline, host="127.0.0.1", port=0,
            poll_interval=0.05, heartbeat_interval=60.0)
        self.server.start(background=True)
        self.base = f"http://127.0.0.1:{self.server.port}"
        # Wait for accept.
        import urllib.request
        for _ in range(100):
            try:
                urllib.request.urlopen(self.base + "/health",
                                       timeout=1).read()
                break
            except OSError:
                time.sleep(0.05)

    def tearDown(self):
        self.server.stop()

    def _open_stream(self, path, headers=None):
        # Raw socket: http.client detaches the socket after getresponse()
        # in Python 3.12, and we want timeout-controlled reads anyway.
        sock = socket.create_connection(("127.0.0.1", self.server.port),
                                        timeout=10)
        sock.settimeout(0.5)
        lines = [f"GET {path} HTTP/1.1", "Host: 127.0.0.1",
                 "Accept: text/event-stream"]
        for key, value in (headers or {}).items():
            lines.append(f"{key}: {value}")
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise AssertionError("stream: no response headers")
            buf += chunk
        head, rest = buf.split(b"\r\n\r\n", 1)
        status = head.split(b"\r\n", 1)[0]
        self.assertIn(b"200", status, status)
        return sock, rest

    def _collect(self, sock, rest, duration):
        buf = bytearray(rest)
        end = time.time() + duration
        while time.time() < end:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            buf += chunk
        return bytes(buf)

    def test_health_enriched(self):
        import urllib.request
        with urllib.request.urlopen(self.base + "/health",
                                     timeout=5) as r:
            body = json.loads(r.read())
        for key in ("ok", "version", "uptime_s", "subscribers",
                    "max_subscribers", "events_emitted", "events_dropped",
                    "ring_size"):
            self.assertIn(key, body)
        self.assertTrue(body["ok"])

    def test_options_preflight(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port,
                                          timeout=5)
        conn.request("OPTIONS", "/stream")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 204)
        self.assertEqual(resp.getheader("Access-Control-Allow-Origin"), "*")
        conn.close()

    def test_live_inject_over_http(self):
        sock, rest = self._open_stream("/stream")
        try:
            self.server.hub.inject("live.test", {"hello": "world"})
            buf = self._collect(sock, rest, 3.0)
        finally:
            sock.close()
        frames = _parse_frames(buf)
        kinds = [f["event"] for f in frames]
        self.assertIsNotNone(
            next((f for f in frames if f["retry"] == 3000), None))
        self.assertIn("ready", kinds)
        data_frames = [f for f in frames if f["event"] == "timeline"]
        self.assertTrue(data_frames)
        payload = json.loads(data_frames[0]["data"][0])
        self.assertEqual(payload["topic"], "live.test")
        self.assertEqual(payload["hello"], "world")
        self.assertTrue(data_frames[0]["id"])

    def test_resume_over_http(self):
        # Seed two events, read the first connection's ids, reconnect
        # with Last-Event-ID and get only what was missed.
        self.server.hub.inject("t.a", {"n": 1})
        sock, rest = self._open_stream("/stream")
        try:
            buf = self._collect(sock, rest, 3.0)
        finally:
            sock.close()
        frames = _parse_frames(buf)
        first_id = next(
            f["id"] for f in frames if f["event"] == "timeline")
        self.server.hub.inject("t.a", {"n": 2})
        self.server.hub.inject("t.a", {"n": 3})
        sock2, rest2 = self._open_stream(
            "/stream", headers={"Last-Event-ID": first_id})
        try:
            buf2 = self._collect(sock2, rest2, 3.0)
        finally:
            sock2.close()
        frames2 = _parse_frames(buf2)
        data_frames = [f for f in frames2 if f["event"] == "timeline"]
        nums = sorted(json.loads(f["data"][0])["n"] for f in data_frames)
        self.assertEqual(nums, [2, 3])

    def test_topic_filter_over_http(self):
        sock, rest = self._open_stream("/stream?topic=want.*")
        try:
            self.server.hub.inject("drop.this", {"n": 0})
            self.server.hub.inject("want.this", {"n": 1})
            buf = self._collect(sock, rest, 3.0)
        finally:
            sock.close()
        frames = _parse_frames(buf)
        data_frames = [f for f in frames if f["event"] == "timeline"]
        self.assertEqual(len(data_frames), 1)
        self.assertEqual(json.loads(data_frames[0]["data"][0])["n"], 1)

    def test_subscriber_count_tracks(self):
        self.assertEqual(self.server.subscriber_count, 0)
        sock, rest = self._open_stream("/stream")
        try:
            end = time.time() + 5
            while self.server.subscriber_count == 0 and time.time() < end:
                time.sleep(0.05)
            self.assertEqual(self.server.subscriber_count, 1)
        finally:
            sock.close()

    def test_context_manager(self):
        with StreamServer(lambda: self.timeline, host="127.0.0.1",
                          port=0) as srv:
            import urllib.request
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{srv.port}/health",
                    timeout=5) as r:
                self.assertTrue(json.loads(r.read())["ok"])
        # After exit the port is closed.
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", srv.port), timeout=2)

    def test_serve_convenience(self):
        srv = serve(lambda: self.timeline, host="127.0.0.1", port=0,
                    poll_interval=0.05, max_subscribers=7)
        try:
            self.assertEqual(srv.hub.max_subscribers, 7)
            self.assertTrue(srv.url.startswith("http://127.0.0.1:"))
        finally:
            srv.stop()


if __name__ == "__main__":
    unittest.main()
