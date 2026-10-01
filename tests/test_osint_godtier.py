"""OSINT god-tier upgrade tests.

Covers the new keyless sources and the unified sweep:
- DNS-over-HTTPS parsing (Cloudflare primary, Google fallback, both-fail)
- urlscan.io / urlhaus / ThreatFox / hackertarget / ip-api.com parsing
- osint_sweep fan-out: auto-classification + per-source isolation when one
  source raises
- osint_report sweep enrichment
- register() exposes the new tools

No live network: the HTTP layer (nomorals.tools.osint.HttpClient), the UDP
dns_query helper, rdap_domain, and time.sleep are all mocked.
"""

from __future__ import annotations

import re
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from nomorals.core.errors import ToolError
from nomorals.tools import osint as O
from nomorals.tools.registry import ToolRegistry


# ── fakes ────────────────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None, text: str = "") -> None:
        self.status = status
        self._payload = payload
        self._text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload

    @property
    def text(self) -> str:
        return self._text


class FakeHttp:
    """Routes URLs to canned responses. Route values may be FakeResponse,
    Exception (raised), or a callable taking the URL and returning one."""

    routes: dict[str, Any] = {}
    calls: list[tuple[str, str]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _route(self, url: str) -> FakeResponse:
        for needle, resp in FakeHttp.routes.items():
            if needle in url:
                if callable(resp):
                    return resp(url)
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise AssertionError(f"unexpected URL in test: {url}")

    def get(self, url: str, **kw: Any) -> FakeResponse:
        FakeHttp.calls.append(("GET", url))
        return self._route(url)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        FakeHttp.calls.append(("POST-FORM", url))
        return self._route(url)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        FakeHttp.calls.append(("POST-JSON", url))
        return self._route(url)


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(settings=SimpleNamespace(osint=SimpleNamespace(
        request_timeout=7.0, abuseipdb_key="", shodan_key="", crtsh_days=3650)))


DOH_DATA = {
    "A": ["93.184.216.34"],
    "AAAA": ["2606:2800:220:1:248:1893:25c8:1946"],
    "MX": ["10 mail.example.com."],
    "TXT": ['"v=spf1 -all"'],
    "NS": ["a.iana-servers.net.", "b.iana-servers.net."],
    "CAA": ['0 issue "letsencrypt.org"'],
    "SOA": ["ns.example.com. hostmaster.example.com. 2024010101 7200 3600 1209600 3600"],
}


def doh_answer(url: str) -> FakeResponse:
    match = re.search(r"[?&]type=([A-Z]+)", url)
    rtype = match.group(1) if match else "A"
    answers = [{"name": "example.com.", "type": 1, "TTL": 300, "data": d}
               for d in DOH_DATA.get(rtype, [])]
    return FakeResponse(payload={"Status": 0, "Answer": answers})


CRT_PAYLOAD = [{
    "common_name": "example.com",
    "name_value": "example.com\nwww.example.com",
    "issuer_name": "C=US, O=Let's Encrypt",
    "not_before": "2024-01-01T00:00:00",
}]

URLSCAN_PAYLOAD = {
    "total": 12,
    "results": [{
        "page": {"url": "http://example.com/", "domain": "example.com",
                 "ip": "93.184.216.34", "country": "US"},
        "task": {"time": "2024-02-01T10:00:00.000Z"},
        "verdicts": {"overall": {"malicious": True, "score": 85}},
        "_id": "abc-123",
    }],
}

URLHAUS_HOST_PAYLOAD = {
    "query_status": "ok",
    "host": "example.com",
    "firstseen": "2024-01-15 10:00:00",
    "lastseen": "2024-03-01 10:00:00",
    "url_count": 3,
    "urls": [{"url": "http://example.com/evil.exe", "threat": "malware_download",
              "url_status": "online", "tags": ["exe", "emotet"]}],
}

URLHAUS_URL_PAYLOAD = {
    "query_status": "ok",
    "urlhaus_reference": "https://urlhaus.abuse.ch/url/12345/",
    "threat": "malware_download",
    "tags": ["exe"],
    "url_status": "online",
    "firstseen": "2024-01-15 10:00:00",
}

THREATFOX_PAYLOAD = {
    "query_status": "ok",
    "data": [{
        "ioc": "example.com",
        "ioc_type_desc": "domain",
        "threat_type_desc": "botnet_cc",
        "malware": "emotet",
        "confidence_level": 90,
        "first_seen": "2024-01-15 10:00:00 UTC",
        "last_seen": "2024-03-01 10:00:00 UTC",
        "tags": ["emotet"],
    }],
}

IPAPI_PAYLOAD = {
    "status": "success",
    "country": "United States", "regionName": "California", "city": "Los Angeles",
    "lat": 34.05, "lon": -118.24, "isp": "Edgecast",
    "org": "Edgecast", "as": "AS15133 Edgecast", "asn": "AS15133",
    "reverse": "example.com", "query": "93.184.216.34",
}


def _base_routes() -> dict[str, Any]:
    return {
        "crt.sh": FakeResponse(payload=CRT_PAYLOAD),
        "cloudflare-dns.com/dns-query": doh_answer,
        "dns.google/resolve": doh_answer,
        "urlscan.io": FakeResponse(payload=URLSCAN_PAYLOAD),
        "urlhaus-api.abuse.ch/v1/host/": FakeResponse(payload=URLHAUS_HOST_PAYLOAD),
        "urlhaus-api.abuse.ch/v1/url/": FakeResponse(payload=URLHAUS_URL_PAYLOAD),
        "threatfox-api.abuse.ch": FakeResponse(payload=THREATFOX_PAYLOAD),
        "api.hackertarget.com/dnslookup/": FakeResponse(text="A : 93.184.216.34\n"),
        "api.hackertarget.com/hostsearch/": FakeResponse(
            text="93.184.216.34,www.example.com\n93.184.216.34,mail.example.com\n"),
        "api.hackertarget.com/httpheaders/": FakeResponse(
            text="HTTP/1.1 200 OK\nServer: nginx\n"),
        "api.hackertarget.com/pagelinks/": FakeResponse(
            text="http://example.com/a\nhttp://example.com/b\n"),
        "api.hackertarget.com/reverseiplookup/": FakeResponse(text="example.com\n"),
        "ip-api.com": FakeResponse(payload=IPAPI_PAYLOAD),
        "rdap.org/ip/": FakeResponse(payload={"handle": "NET-93-184-216-0-1",
                                               "entities": []}),
        "ipwho.is": FakeResponse(payload={"country": "United States",
                                           "city": "Los Angeles",
                                           "connection": {"isp": "Edgecast",
                                                          "org": "Edgecast"}}),
    }


class GodtierBase(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _ctx()
        FakeHttp.routes = _base_routes()
        FakeHttp.calls = []
        self._http = patch("nomorals.tools.osint.HttpClient", FakeHttp).start()
        self._dns = patch("nomorals.tools.osint.dns_query",
                          return_value=["93.184.216.34"]).start()
        self._rdap = patch("nomorals.tools.osint.rdap_domain",
                           return_value={"domain": "example.com"}).start()
        self._sleep = patch("time.sleep", lambda s: None).start()
        self.addCleanup(patch.stopall)


# ── classification ───────────────────────────────────────────────────────────


class ClassificationTests(GodtierBase):
    def test_classify_all_kinds(self) -> None:
        self.assertEqual(O._classify("example.com"), "domain")
        self.assertEqual(O._classify("93.184.216.34"), "ip")
        self.assertEqual(O._classify("2001:db8::1"), "ip")
        self.assertEqual(O._classify("https://example.com/x"), "url")
        self.assertEqual(O._classify("http://example.com/x"), "url")
        self.assertEqual(O._classify("user@example.com"), "email")

    def test_classify_rejects_garbage(self) -> None:
        for bad in ("not a target!", "@@@", "://bad", "foo bar"):
            with self.assertRaises(ToolError, msg=bad):
                O._classify(bad)

    def test_sweep_domain_derivation(self) -> None:
        self.assertEqual(O._sweep_domain("domain", "Example.COM."), "example.com")
        self.assertEqual(O._sweep_domain("email", "User@Example.com"), "example.com")
        self.assertEqual(O._sweep_domain("url", "https://Sub.Example.com/x"),
                         "sub.example.com")
        self.assertEqual(O._sweep_domain("ip", "1.2.3.4"), "")


# ── DNS-over-HTTPS ───────────────────────────────────────────────────────────


class DohTests(GodtierBase):
    def test_doh_records_parsed(self) -> None:
        out = O.osint_dns(self.ctx, "example.com")
        self.assertEqual(out["records"]["A"], ["93.184.216.34"])
        self.assertEqual(out["records"]["AAAA"],
                         ["2606:2800:220:1:248:1893:25c8:1946"])
        self.assertEqual(out["records"]["MX"], ["10 mail.example.com."])
        # TXT quotes stripped
        self.assertEqual(out["records"]["TXT"], ["v=spf1 -all"])
        self.assertEqual(len(out["records"]["NS"]), 2)
        self.assertIn("cloudflare-dns.com", out["providers"])
        self.assertNotIn("errors", out)

    def test_doh_falls_back_to_google(self) -> None:
        FakeHttp.routes["cloudflare-dns.com/dns-query"] = FakeResponse(status=503)
        out = O.osint_dns(self.ctx, "example.com")
        self.assertEqual(out["records"]["A"], ["93.184.216.34"])
        self.assertIn("dns.google", out["providers"])
        self.assertNotIn("cloudflare-dns.com", out["providers"])

    def test_doh_both_fail_is_fail_soft(self) -> None:
        FakeHttp.routes["cloudflare-dns.com/dns-query"] = RuntimeError("down")
        FakeHttp.routes["dns.google/resolve"] = RuntimeError("down")
        out = O.osint_dns(self.ctx, "example.com")
        self.assertEqual(out["records"]["A"], [])
        self.assertIn("A", out["errors"])
        self.assertEqual(out["providers"], [])

    def test_doh_bad_domain(self) -> None:
        with self.assertRaises(ToolError):
            O.osint_dns(self.ctx, "not a domain")


# ── threat feeds ─────────────────────────────────────────────────────────────


class ThreatFeedTests(GodtierBase):
    def test_urlscan_parsing(self) -> None:
        out = O._urlscan_search("example.com", "domain", 7.0)
        self.assertEqual(out["total"], 12)
        self.assertEqual(len(out["scans"]), 1)
        scan = out["scans"][0]
        self.assertEqual(scan["url"], "http://example.com/")
        self.assertTrue(scan["malicious"])
        self.assertEqual(scan["score"], 85)
        self.assertEqual(scan["uuid"], "abc-123")
        # query shape is domain-scoped
        self.assertTrue(any("q=domain%3Aexample.com" in url or "domain:example.com"
                            in url for _, url in FakeHttp.calls))

    def test_urlscan_url_query_shape(self) -> None:
        O._urlscan_search("https://example.com/x", "url", 7.0)
        self.assertTrue(any("page.url" in url for _, url in FakeHttp.calls))

    def test_urlhaus_host_parsing(self) -> None:
        out = O._urlhaus_host("example.com", 7.0)
        self.assertEqual(out["query_status"], "ok")
        self.assertEqual(out["url_count"], 3)
        self.assertEqual(out["sample_urls"][0]["threat"], "malware_download")
        self.assertEqual(out["tags"], ["emotet", "exe"])

    def test_urlhaus_no_results(self) -> None:
        FakeHttp.routes["urlhaus-api.abuse.ch/v1/host/"] = FakeResponse(
            payload={"query_status": "no_results"})
        out = O._urlhaus_host("clean.example", 7.0)
        self.assertEqual(out["query_status"], "no_results")

    def test_urlhaus_url_parsing(self) -> None:
        out = O._urlhaus_url("https://example.com/evil.exe", 7.0)
        self.assertEqual(out["threat"], "malware_download")
        self.assertEqual(out["urlhaus_reference"],
                         "https://urlhaus.abuse.ch/url/12345/")

    def test_threatfox_parsing(self) -> None:
        out = O._threatfox_search("example.com", 7.0)
        self.assertEqual(out["query_status"], "ok")
        self.assertEqual(out["count"], 1)
        ioc = out["iocs"][0]
        self.assertEqual(ioc["malware"], "emotet")
        self.assertEqual(ioc["confidence"], 90)
        self.assertEqual(ioc["threat_type"], "botnet_cc")

    def test_hackertarget_bundle_domain(self) -> None:
        out = O._hackertarget_bundle("domain", "example.com", 7.0)
        self.assertIn("93.184.216.34", out["dnslookup"])
        self.assertEqual(len(out["hostsearch"]), 2)
        self.assertEqual(out["hostsearch"][0],
                         {"ip": "93.184.216.34", "host": "www.example.com"})
        # politeness delay between the two calls
        methods = [m for m, u in FakeHttp.calls if "hackertarget" in u]
        self.assertEqual(len(methods), 2)

    def test_hackertarget_rate_limit_is_fail_soft(self) -> None:
        FakeHttp.routes["api.hackertarget.com/dnslookup/"] = FakeResponse(
            text="error you exceeded the daily limit")
        out = O._hackertarget_bundle("domain", "example.com", 7.0)
        self.assertTrue(out["dnslookup"].startswith("unavailable:"))
        self.assertEqual(len(out["hostsearch"]), 2)

    def test_ipapi_parsing(self) -> None:
        out = O._ipapi("93.184.216.34", 7.0)
        self.assertEqual(out["country"], "United States")
        self.assertEqual(out["asn"], "AS15133")
        self.assertEqual(out["isp"], "Edgecast")
        self.assertNotIn("status", out)

    def test_osint_threat_bundle(self) -> None:
        out = O.osint_threat(self.ctx, "example.com")
        self.assertEqual(out["kind"], "domain")
        self.assertEqual(out["urlscan"]["total"], 12)
        self.assertEqual(out["urlhaus"]["query_status"], "ok")
        self.assertEqual(out["threatfox"]["iocs"][0]["malware"], "emotet")

    def test_osint_threat_feed_failure_isolated(self) -> None:
        FakeHttp.routes["urlscan.io"] = RuntimeError("urlscan down")
        out = O.osint_threat(self.ctx, "example.com")
        self.assertTrue(out["urlscan"].startswith("unavailable:"))
        self.assertEqual(out["urlhaus"]["query_status"], "ok")
        self.assertIn("threatfox", out)


# ── sweep ────────────────────────────────────────────────────────────────────


class SweepTests(GodtierBase):
    def test_sweep_domain_fanout(self) -> None:
        out = O.osint_sweep(self.ctx, "example.com")
        self.assertEqual(out["kind"], "domain")
        self.assertEqual(out["target"], "example.com")
        self.assertEqual(out["base"]["kind"], "domain")
        for name in ("dns_doh", "urlscan", "urlhaus", "threatfox", "hackertarget"):
            self.assertIn(name, out["sources"])
            section = out["sources"][name]
            self.assertTrue(section["ok"], name)
            self.assertIsNone(section["error"])
            self.assertGreaterEqual(section["seconds"], 0.0)
            self.assertIsNotNone(section["data"])
        self.assertEqual(out["sources_ok"], 5)
        self.assertEqual(out["sources_failed"], 0)
        # spot-check merged data
        self.assertEqual(out["sources"]["dns_doh"]["data"]["records"]["A"],
                         ["93.184.216.34"])
        self.assertEqual(out["sources"]["urlscan"]["data"]["total"], 12)

    def test_sweep_source_failure_is_isolated(self) -> None:
        FakeHttp.routes["api.hackertarget.com/dnslookup/"] = RuntimeError("boom")
        FakeHttp.routes["api.hackertarget.com/hostsearch/"] = RuntimeError("boom")
        out = O.osint_sweep(self.ctx, "example.com")
        ht = out["sources"]["hackertarget"]
        self.assertFalse(ht["ok"])
        self.assertIsNone(ht["data"])
        self.assertIn("RuntimeError", ht["error"])
        self.assertTrue(out["sources"]["urlscan"]["ok"])
        self.assertTrue(out["sources"]["threatfox"]["ok"])
        self.assertEqual(out["sources_ok"], 4)
        self.assertEqual(out["sources_failed"], 1)

    def test_sweep_ip_uses_ip_sources(self) -> None:
        out = O.osint_sweep(self.ctx, "93.184.216.34")
        self.assertEqual(out["kind"], "ip")
        self.assertEqual(out["base"]["kind"], "ip")
        self.assertIn("ipapi", out["sources"])
        self.assertIn("hackertarget", out["sources"])
        self.assertNotIn("dns_doh", out["sources"])
        self.assertEqual(out["sources"]["ipapi"]["data"]["country"], "United States")
        self.assertIn("example.com",
                      out["sources"]["hackertarget"]["data"]["reverseiplookup"])

    def test_sweep_email_uses_domain_part(self) -> None:
        out = O.osint_sweep(self.ctx, "user@example.com")
        self.assertEqual(out["kind"], "email")
        self.assertEqual(out["base"]["kind"], "email")
        self.assertIn("dns_doh", out["sources"])
        self.assertEqual(out["sources"]["dns_doh"]["data"]["target"], "example.com")

    def test_sweep_rejects_empty_and_garbage(self) -> None:
        with self.assertRaises(ToolError):
            O.osint_sweep(self.ctx, "")
        with self.assertRaises(ToolError):
            O.osint_sweep(self.ctx, "not a target!")


# ── report enrichment ────────────────────────────────────────────────────────


class ReportTests(GodtierBase):
    def test_report_includes_intel_sections(self) -> None:
        out = O.osint_report(self.ctx, "example.com")
        # base keys untouched
        self.assertEqual(out["kind"], "domain")
        self.assertIn("dns", out)
        self.assertIn("cert_transparency", out)
        # sweep enrichment merged in
        self.assertIn("intel", out)
        for name in ("dns_doh", "urlscan", "urlhaus", "threatfox", "hackertarget"):
            self.assertIn(name, out["intel"])
        self.assertNotIn("intel_errors", out)
        self.assertEqual(set(out["intel_seconds"]),
                         set(out["intel"]))
        self.assertEqual(out["intel"]["urlscan"]["total"], 12)

    def test_report_records_intel_errors(self) -> None:
        FakeHttp.routes["threatfox-api.abuse.ch"] = RuntimeError("threatfox down")
        out = O.osint_report(self.ctx, "example.com")
        self.assertIn("threatfox", out["intel_errors"])
        self.assertNotIn("threatfox", out["intel"])
        self.assertIn("urlscan", out["intel"])


# ── registration ─────────────────────────────────────────────────────────────


class RegisterTests(GodtierBase):
    def test_new_tools_registered(self) -> None:
        registry = ToolRegistry(self.ctx)
        O.register(registry)
        names = set(registry.names())
        for name in ("osint_report", "osint_domain", "osint_ip", "osint_url",
                     "osint_email", "osint_dns", "osint_threat", "osint_sweep"):
            self.assertIn(name, names, name)
        for name in ("osint_dns", "osint_threat", "osint_sweep"):
            self.assertTrue(registry.get(name).description, name)


if __name__ == "__main__":
    unittest.main()
