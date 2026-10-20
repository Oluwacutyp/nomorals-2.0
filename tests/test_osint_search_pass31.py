"""Pass #31 — OSINT + search tool surfaces: verified regression pins.

Each test was written failing-first against HEAD d7c514fc:

1. ``dns_query(..., "SPF")`` queried TXT at ``spf.<domain>`` — the SPF
   policy lives as a TXT record at the domain itself (RFC 7208), so SPF
   lookups silently came back empty for nearly every real domain.
2. ``osint_url`` never followed redirects: the no-redirect opener makes
   urllib raise HTTPError for 3xx, and the handler treated that as a
   terminal error — the "redirect chain" was one entry mislabeled as an
   error, with no final response.
3. ``_normalize_url`` dropped the whole query string, so distinct pages
   (``watch?v=AAA`` vs ``watch?v=BBB``) deduped into one result.
4. The crt.sh / certspotter / rapiddns subdomain helpers swallowed every
   failure and returned empty data, so a dead source reported as a
   successful sweep source with zero subdomains.
5. ``_run_engines`` never looked at the HTTP status: an engine answering
   403/429/5xx was silently "no results" instead of a failed engine.
6. ``email_investigate`` read ``nameservers`` from a key osint_domain
   never returns (always None), and ``osint_people`` overwrote an
   email's spf/dmarc section with its breach note.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.core.errors import ToolError
from nomorals.tools import network, osint, osint_people, web

CTX = SimpleNamespace(settings=None)


# ── 1. SPF lookup ────────────────────────────────────────────────────────

def _wire_name(name: str) -> bytes:
    out = b""
    for label in name.rstrip(".").split("."):
        out += bytes([len(label)]) + label.encode()
    return out + b"\x00"


class _FakeUdpSocket:
    """Answers TXT for example.com with its SPF record; nothing else."""

    queried: list[str] = []

    def __init__(self, *args, **kwargs):
        self._query = b""
        self._qname = ""

    def settimeout(self, _t):
        pass

    def close(self):
        pass

    def sendto(self, data, _addr):
        self._query = data
        pos = 12
        labels = []
        while data[pos] != 0:
            ln = data[pos]
            labels.append(data[pos + 1:pos + 1 + ln].decode())
            pos += 1 + ln
        self._qname = ".".join(labels)
        _FakeUdpSocket.queried.append(self._qname)

    def recvfrom(self, _bufsize):
        tid = self._query[:2]
        question = self._query[12:]
        answer = b""
        an = 0
        if self._qname == "example.com":
            txt = b"v=spf1 -all"
            rdata = bytes([len(txt)]) + txt
            answer = (b"\xc0\x0c" + struct.pack(">HHIH", 16, 1, 300, len(rdata))
                      + rdata)
            an = 1
        header = tid + struct.pack(">HHHHH", 0x8180, 1, an, 0, 0)
        return header + question + answer, ("127.0.0.1", 53)


class SpfLookupTests(unittest.TestCase):
    def test_spf_queried_at_the_domain_itself(self):
        _FakeUdpSocket.queried = []
        with mock.patch.object(socket, "socket", _FakeUdpSocket):
            got = network.dns_query("example.com", "SPF",
                                    resolvers=("127.0.0.1",), timeout=1.0)
        self.assertEqual(_FakeUdpSocket.queried, ["example.com"])
        self.assertEqual(got, ["v=spf1 -all"])


# ── 2. osint_url redirect chain ──────────────────────────────────────────

class _RedirectHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib handler API
        if self.path == "/start":
            self.send_response(302)
            self.send_header("Location", "/final")
            self.end_headers()
        elif self.path == "/hop":
            self.send_response(301)
            self.send_header("Location", "/start")
            self.end_headers()
        else:
            body = b"final-body-here"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):
        pass


class OsintUrlRedirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_redirect_chain_is_followed_to_final(self):
        out = osint.osint_url(CTX, f"http://127.0.0.1:{self.port}/hop")
        self.assertIn("final", out)
        self.assertEqual(out["final"]["status"], 200)
        self.assertIn("final-body-here", out["final"]["body_start"])
        statuses = [entry["status"] for entry in out["chain"]]
        self.assertEqual(statuses, [301, 302, 200])
        self.assertNotIn("error", out["chain"][0])
        self.assertTrue(out["chain"][0]["location"].endswith("/start"))


# ── 3. dedupe keeps identity-bearing query params ────────────────────────

class DedupeQueryTests(unittest.TestCase):
    def test_distinct_query_pages_are_not_duplicates(self):
        items = [
            {"url": "https://youtube.com/watch?v=AAA", "title": "a", "snippet": ""},
            {"url": "https://youtube.com/watch?v=BBB", "title": "b", "snippet": ""},
        ]
        self.assertEqual(len(web._dedupe_results(items)), 2)

    def test_tracking_params_still_canonicalize(self):
        items = [
            {"url": "https://example.com/beta", "title": "a", "snippet": ""},
            {"url": "https://example.com/beta/", "title": "b", "snippet": ""},
            {"url": "https://example.com/beta?utm_source=x", "title": "c", "snippet": ""},
        ]
        self.assertEqual(len(web._dedupe_results(items)), 1)

    def test_param_order_is_canonicalized(self):
        items = [
            {"url": "https://example.com/p?a=1&b=2", "title": "a", "snippet": ""},
            {"url": "https://example.com/p?b=2&a=1", "title": "b", "snippet": ""},
        ]
        self.assertEqual(len(web._dedupe_results(items)), 1)


# ── 4. subdomain feeds fail honestly ─────────────────────────────────────

class _BoomHttp:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, url, **kwargs):
        raise ConnectionError("source unreachable")


class _StatusHttp:
    status = 502

    def __init__(self, *args, **kwargs):
        pass

    def get(self, url, **kwargs):
        return SimpleNamespace(ok=False, status=self.status, text="")


class SubdomainFeedHonestyTests(unittest.TestCase):
    def test_crtsh_transport_failure_raises(self):
        with mock.patch.object(osint, "HttpClient", _BoomHttp):
            with self.assertRaises(ToolError):
                osint._crtsh_subdomains("example.com", 2.0)

    def test_crtsh_http_error_raises(self):
        with mock.patch.object(osint, "HttpClient", _StatusHttp):
            with self.assertRaises(ToolError):
                osint._crtsh_subdomains("example.com", 2.0)

    def test_certspotter_failure_raises(self):
        with mock.patch.object(osint, "HttpClient", _BoomHttp):
            with self.assertRaises(ToolError):
                osint._certspotter_subdomains("example.com", 2.0)

    def test_rapiddns_failure_raises(self):
        with mock.patch.object(osint, "HttpClient", _BoomHttp):
            with self.assertRaises(ToolError):
                osint._rapiddns_subdomains("example.com", 2.0)

    def test_sweep_section_marks_dead_source_failed(self):
        with mock.patch.object(osint, "HttpClient", _BoomHttp):
            section = osint._run_source(
                "crtsh", lambda: osint._crtsh_subdomains("example.com", 2.0))
        self.assertFalse(section["ok"])
        self.assertIsNotNone(section["error"])

    def test_genuine_empty_is_still_ok(self):
        class EmptyHttp:
            def __init__(self, *args, **kwargs):
                pass

            def get(self, url, **kwargs):
                return SimpleNamespace(
                    ok=True, status=200, text="[]",
                    json=lambda: [])

        with mock.patch.object(osint, "HttpClient", EmptyHttp):
            out = osint._crtsh_subdomains("example.com", 2.0)
        self.assertEqual(out["count"], 0)
        self.assertEqual(out["subdomains"], [])


# ── 5. engine HTTP errors count as failures ──────────────────────────────

_DDG_HTML = ('<a class="result__a" href="https://example.com/alpha">'
             "Example Alpha Result</a>")


class _MixedClient:
    def get(self, url, **kwargs):
        if "bing.com" in url:
            return SimpleNamespace(ok=False, status=403, text="<html>blocked</html>")
        if "html.duckduckgo.com" in url:
            return SimpleNamespace(ok=True, status=200, text=_DDG_HTML)
        return SimpleNamespace(ok=True, status=200, text="<html></html>")


class EngineStatusTests(unittest.TestCase):
    def test_http_error_status_counts_as_failed(self):
        _results, used, failed = web._run_engines(_MixedClient(), "x")
        self.assertIn("duckduckgo", used)
        self.assertIn("bing", failed)

    def test_ok_with_no_results_is_not_a_failure(self):
        class EmptyClient:
            def get(self, url, **kwargs):
                return SimpleNamespace(ok=True, status=200, text="<html></html>")

        _results, used, failed = web._run_engines(EmptyClient(), "x")
        self.assertEqual(failed, [])

    def test_legacy_fake_without_ok_attr_still_works(self):
        class BareClient:
            def get(self, url, **kwargs):
                return SimpleNamespace(text="<html></html>")

        _results, _used, failed = web._run_engines(BareClient(), "x")
        self.assertEqual(failed, [])


# ── 6. people-surface wiring ─────────────────────────────────────────────

class EmailInvestigateWiringTests(unittest.TestCase):
    def test_nameservers_come_from_the_domain_dns(self):
        with mock.patch.object(
                osint_people, "dns_query",
                lambda domain, record="A", **kw:
                    ["10 mail.example.com"] if record == "MX" else []), \
             mock.patch.object(
                osint_people, "osint_domain",
                lambda c, d: {"registration": {"registrar": "Reg"},
                              "dns": {"NS": ["ns1.example.com"]}}), \
             mock.patch.object(
                osint_people, "breach_check",
                lambda c, t: {"checked": False, "reason": "no key"}):
            out = osint_people.email_investigate(CTX, "ada@example.com")
        self.assertEqual(out["domain_nameservers"], ["ns1.example.com"])

    def test_people_section_keeps_spf_and_breach_note(self):
        canned = {
            "domain": "example.com", "mx": ["10 mail.example.com"],
            "spf": "v=spf1 -all", "dmarc": "v=DMARC1; p=none",
            "deliverable": "MX present",
            "breaches": {"checked": True, "pwned": False, "breaches": []},
        }
        with mock.patch.object(osint_people, "email_investigate",
                               lambda c, e: canned):
            out = osint_people.osint_people(
                CTX, json.dumps({"emails": ["ada@example.com"]}))
        section = out["sections"]["emails"]["ada@example.com"]
        self.assertEqual(section["spf"], "v=spf1 -all")
        self.assertEqual(section["breach"], "clean (HIBP)")


if __name__ == "__main__":
    unittest.main()
