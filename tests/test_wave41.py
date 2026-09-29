"""Wave 41 — the high-power tool layer (hermetic: no external network).

Covers the six new modules and their wiring:

- tools/network.py   DNS codec, scope guard, real local port scan, listeners
- tools/proxy.py     URL parsing, ProxyManager persistence + egress probes
- tools/scriptgen.py all 9 script kinds, validation, path safety
- tools/osint.py     domain/ip/url/email bundles + report auto-detect
- tools/macros.py    recorder lifecycle, store, live tool registration, replay
- wiring             registry dispatch (positional-only name), settings env
                     binding, /dns /scan /proxy /gen /osint /record /macro
                     command table, devon catalog + recorder hook
"""
from __future__ import annotations

import ipaddress
import json
import socket
import struct
import threading
import time
import unittest
from unittest import mock

from tests.test_partner_runtime import _make_context

from nomorals.core.config import load_settings
from nomorals.core.errors import ToolError
from nomorals.tools import macros, network, osint as osint_mod, proxy as proxy_mod, scriptgen
from nomorals.tools.registry import ToolRegistry


def _registry_for(context) -> ToolRegistry:
    reg = ToolRegistry()
    reg.context = context
    reg.register_builtins()
    context.tools = reg
    return reg


class _LocalListener:
    """A real TCP listener on 127.0.0.1 for the port scanner to find."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            try:
                # real services (SSH/SMTP/HTTP) greet the client on connect
                conn.sendall(b"NMTEST\r\n")
            except OSError:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2)


# ── DNS codec ────────────────────────────────────────────────────────────────


def _dns_response(answers: list[tuple[str, int, bytes]], *, rcode: int = 0,
                  qname: str = "example.com") -> bytes:
    """Hand-build a DNS response (id 0x1234, RD flag, question for qname)."""
    # header: id, flags, qdcount, ANCOUNT, nscount, arcount
    header = struct.pack(">HHHHHH", 0x1234, 0x8000 | rcode, 1, len(answers), 0, 0)
    question = network._encode_name(qname) + struct.pack(">HH", 1, 1)
    body = b""
    for name, rtype, rdata in answers:
        body += struct.pack(">H", 0xC00C)  # compression pointer to qname
        body += struct.pack(">HHIH", rtype, 1, 60, len(rdata))
        body += rdata
    return header + question + body


def _mx_rdata(pref: int, exchange: str) -> bytes:
    return struct.pack(">H", pref) + network._encode_name(exchange)


class DnsCodecTest(unittest.TestCase):
    """The parser must survive real-world byte layouts, including name
    compression, multi-string TXT, and SOA with two domain names."""

    def test_a_record(self) -> None:
        data = _dns_response([("example.com", 1, bytes([93, 184, 216, 34]))])
        answers, rcode = network._parse_response(data)
        self.assertEqual(rcode, 0)
        self.assertEqual(answers, ["93.184.216.34"])

    def test_multiple_a_records(self) -> None:
        data = _dns_response([
            ("example.com", 1, bytes([1, 1, 1, 1])),
            ("example.com", 1, bytes([2, 2, 2, 2])),
        ])
        answers, _ = network._parse_response(data)
        self.assertEqual(answers, ["1.1.1.1", "2.2.2.2"])

    def test_aaaa_record(self) -> None:
        ip = ipaddress.ip_address("2606:4700:4700::1111")
        data = _dns_response([("example.com", 28, ip.packed)])
        answers, _ = network._parse_response(data)
        self.assertEqual(answers, ["2606:4700:4700::1111"])

    def test_mx_records_with_full_names(self) -> None:
        data = _dns_response([
            ("example.com", 15, _mx_rdata(10, "mail1.example.com")),
            ("example.com", 15, _mx_rdata(20, "mail2.example.com")),
        ])
        answers, _ = network._parse_response(data)
        self.assertEqual(answers, ["10 mail1.example.com", "20 mail2.example.com"])

    def test_soa_with_two_uncompressed_names(self) -> None:
        mname = network._encode_name("ns1.example.com")
        rname = network._encode_name("hostmaster.example.com")
        rdata = mname + rname + struct.pack(">IIIII", 2025091101, 7200, 3600,
                                            1209600, 300)
        data = _dns_response([("example.com", 6, rdata)])
        answers, _ = network._parse_response(data)
        self.assertEqual(
            answers,
            ["ns1.example.com hostmaster.example.com 2025091101 7200 3600 "
             "1209600 300"],
        )

    def test_txt_multi_string(self) -> None:
        rdata = bytes([6]) + b"v=spf1" + bytes([5]) + b" -all"
        data = _dns_response([("example.com", 16, rdata)])
        answers, _ = network._parse_response(data)
        self.assertEqual(answers, ["v=spf1 -all"])

    def test_txt_stops_at_rdata_boundary(self) -> None:
        # a TXT followed by another record: the walker must not eat the
        # following record's bytes as more TXT strings
        txt = bytes([3]) + b"abc"
        following = bytes([9, 9, 9, 9])  # looks like garbage if mis-walked
        data = _dns_response([
            ("example.com", 16, txt),
            ("example.com", 1, following),
        ])
        answers, _ = network._parse_response(data)
        self.assertEqual(answers, ["abc", "9.9.9.9"])

    def test_cname_record(self) -> None:
        data = _dns_response([("www.example.com", 5,
                               network._encode_name("cdn.example.net"))])
        answers, _ = network._parse_response(data)
        self.assertEqual(answers, ["cdn.example.net"])

    def test_nxdomain_is_definitive_empty(self) -> None:
        data = _dns_response([], rcode=3)
        answers, rcode = network._parse_response(data)
        self.assertEqual((answers, rcode), ([], 3))

    def test_empty_answer_section(self) -> None:
        data = _dns_response([], rcode=0)
        answers, rcode = network._parse_response(data)
        self.assertEqual((answers, rcode), ([], 0))

    def test_short_garbage_raises(self) -> None:
        with self.assertRaises(ToolError):
            network._parse_response(b"\x12\x34")

    def test_name_compression_pointer_walk(self) -> None:
        # answer NAME is a pointer to the question name: question is
        # 13 bytes ("example.com") at offset 12, then 4 bytes of type/class,
        # so the answer (and its pointer) starts at 29
        data = _dns_response([("example.com", 1, bytes([1, 2, 3, 4]))])
        name, offset = network._parse_name(data, 29)
        self.assertEqual(name, "example.com")
        self.assertEqual(offset, 31)  # pointer consumed: +2 bytes

    def test_encode_name_roundtrip(self) -> None:
        encoded = network._encode_name("a.b.example.com")
        self.assertEqual(encoded, b"\x01a\x01b\x07example\x03com\x00")


# ── scope guard ──────────────────────────────────────────────────────────────


class ScopeGuardTest(unittest.TestCase):
    def test_own_ranges_always_pass(self) -> None:
        for target in ("127.0.0.1", "10.0.0.5", "192.168.1.2", "172.16.0.1",
                       "169.254.10.10", "100.64.0.1"):
            self.assertTrue(network.is_scannable(target), target)

    def test_public_needs_allowlist(self) -> None:
        self.assertFalse(network.is_scannable("8.8.8.8"))
        self.assertTrue(network.is_scannable("8.8.8.8", allowed_targets="8.8.8.8"))
        self.assertTrue(network.is_scannable("8.8.8.8", allowed_targets="1.2.3.4, 8.8.8.8"))

    def test_hostname_allowlist_case_insensitive(self) -> None:
        self.assertFalse(network.is_scannable("MyPhone.Local"))
        self.assertTrue(network.is_scannable("MyPhone.Local",
                                             allowed_targets="myphone.local"))

    def test_scan_refuses_out_of_scope(self) -> None:
        context, tmp = _make_context()
        try:
            with self.assertRaises(ToolError) as ctx:
                network._assert_scannable("8.8.8.8", "")
            self.assertIn("outside the allowed scope", str(ctx.exception))
        finally:
            context.close()
            tmp.cleanup()


# ── port scan (real local listener) ──────────────────────────────────────────


class PortScanLiveTest(unittest.TestCase):
    def test_finds_open_and_closed_ports(self) -> None:
        listener = _LocalListener()
        closed = listener.port + 1
        try:
            self.assertTrue(network.tcp_probe("127.0.0.1", listener.port, 2.0).startswith("open"))
            self.assertEqual(network.tcp_probe("127.0.0.1", closed, 1.0), "closed")
        finally:
            listener.close()

    def test_banner_grab_on_own_port(self) -> None:
        listener = _LocalListener()
        try:
            self.assertEqual(network.tcp_probe("127.0.0.1", listener.port, 2.0,
                                               banner=True), "open:NMTEST")
        finally:
            listener.close()

    def test_local_listeners_see_the_real_socket(self) -> None:
        listener = _LocalListener()
        try:
            rows = network.local_listeners()
            self.assertGreaterEqual(len(rows), 1, "expected at least one listener")
            ports = {r["port"] for r in rows}
            self.assertIn(listener.port, ports, ports)
        finally:
            listener.close()

    def test_expand_ports(self) -> None:
        self.assertEqual(network._expand_ports("80,443"), [80, 443])
        self.assertEqual(network._expand_ports("80-82"), [80, 81, 82])
        self.assertEqual(network._expand_ports("22, 80-81, 443"), [22, 80, 81, 443])
        # garbage is dropped silently; the tool itself refuses an empty result
        self.assertEqual(network._expand_ports("bogus"), [])
        self.assertEqual(network._expand_ports("99999"), [])

    def test_scan_with_no_usable_ports_refused(self) -> None:
        context, tmp = _make_context()
        try:
            reg = _registry_for(context)
            r = reg.call("port_scan", target="127.0.0.1", ports="bogus")
            self.assertFalse(r.ok)
            self.assertIn("no ports", str(r.error))
        finally:
            context.close()
            tmp.cleanup()


# ── whois / rdap (mocked client) ─────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status = status
        self.ok = 200 <= status < 300

    def json(self):
        return self._payload


class _FakeClient:
    """Stands in for HttpClient: get(url) -> canned JSON response."""

    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def get(self, url, **kwargs):  # noqa: ANN001
        return _FakeResponse(self.payload, self.status)


class WhoisTest(unittest.TestCase):
    def test_rdap_domain_extracts_registrar_and_dates(self) -> None:
        payload = {
            "handle": "EXAMPLE-COM",
            "entities": [
                {"roles": ["registrar"],
                 "vcardArray": ["vcard", [["fn", {}, "text", "Fake Registrar"]]]},
                {"roles": ["registrant"],
                 "vcardArray": ["vcard", [["fn", {}, "text", "Some Org"]]]},
            ],
            "events": [
                {"eventAction": "registration", "eventDate": "1995-09-15T04:00:00Z"},
                {"eventAction": "expiration", "eventDate": "2027-09-14T04:00:00Z"},
                {"eventAction": "last changed", "eventDate": "2026-01-01T00:00:00Z"},
            ],
            "nameservers": [{"ldhName": "NS1.FAKE.COM"}, {"ldhName": "NS2.FAKE.COM"}],
            "status": ["clientTransferProhibited"],
        }
        out = network.rdap_domain("example.com", client=_FakeClient(payload))
        self.assertEqual(out["registrar"], "Fake Registrar")
        self.assertEqual(out["registration"], "1995-09-15")
        self.assertEqual(out["expiration"], "2027-09-14")
        self.assertEqual(out["nameservers"], ["ns1.fake.com", "ns2.fake.com"])

    def test_rdap_failure_raises_clean_toolerror(self) -> None:
        with self.assertRaises(ToolError):
            network.rdap_domain("example.com",
                                client=_FakeClient({}, status=404))

    def test_whois_lookup_failure_is_a_clean_toolerror(self) -> None:
        context, tmp = _make_context()
        try:
            reg = _registry_for(context)
            with mock.patch.object(network, "rdap_domain",
                                   side_effect=ToolError("RDAP lookup failed: net down")):
                result = reg.call("whois_lookup", domain="example.com")
            self.assertFalse(result.ok)
            self.assertIn("RDAP", str(result.error))
        finally:
            context.close()
            tmp.cleanup()


# ── http_craft (mocked opener) ───────────────────────────────────────────────


class _FakeOpener:
    def __init__(self, status: int = 200, headers: dict | None = None,
                 body: bytes = b"hello") -> None:
        self.status = status
        self.headers = headers or {"Server": "NM/1", "Location": "/next"}
        self.body = body
        self.captured = None

    class _Resp:
        def __init__(self, owner: "_FakeOpener") -> None:
            self._o = owner
            self.status = owner.status
            self.headers = owner.headers
            self._body = owner.body

        def read(self, n: int = -1) -> bytes:
            return self._body if n < 0 else self._body[:n]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def open(self, request, timeout=None):  # noqa: ANN001
        self.captured = request
        return self._Resp(self)


class HttpCraftTest(unittest.TestCase):
    def test_headers_body_method_are_sent_and_reported(self) -> None:
        opener = _FakeOpener(status=201, body=b"created",
                             headers={"Server": "NM/1", "Content-Type": "text/plain"})
        with mock.patch("urllib.request.build_opener", return_value=opener):
            out = network.http_craft(
                "post", "http://127.0.0.1/api",
                headers="X-Token: abc\nX-Custom: y",
                body='{"a": 1}',
            )
        self.assertEqual(out["method"], "POST")
        self.assertEqual(out["status"], 201)
        self.assertTrue(out["ok"])
        self.assertEqual(out["body"], "created")
        self.assertEqual(out["request_headers"]["X-Token"], "abc")
        self.assertEqual(out["request_headers"]["X-Custom"], "y")
        self.assertEqual(opener.captured.data, b'{"a": 1}')
        self.assertEqual(opener.captured.get_method(), "POST")

    def test_3xx_is_not_followed_by_default(self) -> None:
        opener = _FakeOpener(status=302, headers={"Location": "http://x/next"})
        with mock.patch("urllib.request.build_opener", return_value=opener):
            out = network.http_craft("get", "http://127.0.0.1/old")
        self.assertEqual(out["status"], 302)
        self.assertFalse(out["ok"])
        self.assertEqual(out["location"], "http://x/next")

    def test_404_is_an_answer_not_a_failure(self) -> None:
        import urllib.error

        opener = mock.Mock()

        def _open(request, timeout=None):  # noqa: ANN001
            raise urllib.error.HTTPError(
                request.full_url, 404, "Not Found", {"Server": "NM"},
                __import__("io").BytesIO(b"nope"))
        opener.open.side_effect = _open
        opener.open.return_value = None
        with mock.patch("urllib.request.build_opener", return_value=opener):
            out = network.http_craft("get", "http://127.0.0.1/missing")
        self.assertEqual(out["status"], 404)
        self.assertEqual(out["body"], "nope")

    def test_bad_method_and_scheme_rejected(self) -> None:
        with self.assertRaises(ToolError):
            network.http_craft("FROB", "http://127.0.0.1/")
        with self.assertRaises(ToolError):
            network.http_craft("get", "ftp://127.0.0.1/")


# ── proxy ────────────────────────────────────────────────────────────────────


class ProxyUrlTest(unittest.TestCase):
    def test_bare_host_port_becomes_http(self) -> None:
        self.assertEqual(proxy_mod.parse_proxy_url("10.0.0.5:8080"),
                         "http://10.0.0.5:8080")
        self.assertEqual(proxy_mod.parse_proxy_url("host:3128"), "http://host:3128")

    def test_schemes_kept(self) -> None:
        self.assertEqual(proxy_mod.parse_proxy_url("socks5://h:9050"),
                         "socks5://h:9050")
        self.assertEqual(proxy_mod.parse_proxy_url("http://h:1"), "http://h:1")
        self.assertEqual(proxy_mod.parse_proxy_url("https://h:1"), "https://h:1")

    def test_empty_is_rejected(self) -> None:
        with self.assertRaises(ToolError):
            proxy_mod.parse_proxy_url("")


class ProxyManagerTest(unittest.TestCase):
    def test_set_active_persists_and_applies_process_wide(self) -> None:
        from nomorals.core.http import get_default_proxy, set_default_proxy

        context, tmp = _make_context()
        try:
            set_default_proxy("")
            manager = proxy_mod.ProxyManager(context)
            out = manager.set_active("http://10.0.0.9:8080")
            self.assertEqual(out["active"], "http://10.0.0.9:8080")
            self.assertTrue(out["persisted"])
            self.assertEqual(get_default_proxy(), "http://10.0.0.9:8080")
            self.assertEqual(manager.status()["active"], "http://10.0.0.9:8080")
            # a FRESH manager reads the same choice from kv_store
            again = proxy_mod.ProxyManager(context)
            self.assertEqual(again.status()["active"], "http://10.0.0.9:8080")
            out = manager.clear()
            self.assertEqual(get_default_proxy(), "")
            self.assertEqual(proxy_mod.ProxyManager(context).status()["active"],
                             "direct (no proxy)")
        finally:
            set_default_proxy("")
            context.close()
            tmp.cleanup()

    def test_known_merges_settings_env_and_active(self) -> None:
        context, tmp = _make_context()
        try:
            manager = proxy_mod.ProxyManager(context)
            with mock.patch.object(type(manager), "_env_proxy",
                                   return_value="socks5://env:9050"), \
                 mock.patch.object(manager, "_known",
                                   "http://cfg:1, http://cfg:2"):
                self.assertEqual(manager.known(),
                                 ["http://cfg:1", "http://cfg:2", "socks5://env:9050"])
        finally:
            context.close()
            tmp.cleanup()

    def test_test_one_dead_proxy_is_a_result_not_a_crash(self) -> None:
        context, tmp = _make_context()
        try:
            manager = proxy_mod.ProxyManager(context)
            out = manager._test_one("http://127.0.0.1:1", 0.4)  # nothing there
            self.assertFalse(out["ok"])
            self.assertIn(out["proxy"], "http://127.0.0.1:1")
        finally:
            context.close()
            tmp.cleanup()

    def test_tools_registered_and_status_works(self) -> None:
        context, tmp = _make_context()
        try:
            reg = _registry_for(context)
            r = reg.call("proxy_status")
            self.assertTrue(r.ok)
            self.assertEqual(r.value["active"], "direct (no proxy)")
            self.assertTrue(r.value["routes_all_clients"])
            r = reg.call("proxy_list")
            self.assertTrue(r.ok)
            self.assertIsInstance(r.value["proxies"], list)
        finally:
            context.close()
            tmp.cleanup()


# ── global routing (SOCKS patch + urllib proxy handler) ─────────────────────


def _pysocks_installed():
    import importlib.util

    return importlib.util.find_spec("socks")


class GlobalRoutingTest(unittest.TestCase):
    def test_socks_apply_without_pysocks_is_a_clean_noop(self) -> None:
        from nomorals.core.http import apply_socks_proxy, reset_socks_proxy

        try:
            import socks  # noqa: F401

            have = True
        except ImportError:
            have = False
        if have:
            self.skipTest("PySocks installed — covered by the live path")
        self.assertFalse(apply_socks_proxy("socks5://10.0.0.1:1080"))
        reset_socks_proxy()  # idempotent either way

    @unittest.skipIf(
        _pysocks_installed() is None,
        "PySocks not installed",
    )
    def test_socks_apply_patches_and_reset_restores(self) -> None:
        import socket

        from nomorals.core.http import apply_socks_proxy, reset_socks_proxy

        original = socket.socket
        try:
            self.assertTrue(apply_socks_proxy("socks5://10.0.0.1:1080"))
            self.assertIsNot(socket.socket, original)
            self.assertTrue(apply_socks_proxy("socks5://10.0.0.1:1080"))  # idempotent
        finally:
            reset_socks_proxy()
        self.assertIs(socket.socket, original)

    def test_default_proxy_handler_for_http_proxies(self) -> None:
        import urllib.request

        from nomorals.core.http import get_default_proxy, set_default_proxy

        previous = get_default_proxy()
        try:
            set_default_proxy("http://10.0.0.2:3128")
            handler = default_proxy_handler_under_test()
            self.assertIsInstance(handler, urllib.request.ProxyHandler)
            set_default_proxy("socks5://10.0.0.3:1080")
            self.assertIsNone(default_proxy_handler_under_test())
            set_default_proxy("")
            self.assertIsNone(default_proxy_handler_under_test())
        finally:
            set_default_proxy(previous)

    def test_http_craft_routes_through_active_http_proxy(self) -> None:
        from nomorals.core.http import get_default_proxy, set_default_proxy

        previous = get_default_proxy()
        set_default_proxy("http://10.0.0.4:3128")
        try:
            opener = _FakeOpener()
            captured = {}

            def _build(*handlers):  # noqa: ANN001
                captured["handlers"] = handlers
                return opener
            with mock.patch("urllib.request.build_opener", side_effect=_build):
                out = network.http_craft("get", "http://127.0.0.1/x")
            self.assertEqual(out["status"], 200)
            import urllib.request as _ur
            self.assertTrue(any(isinstance(h, _ur.ProxyHandler)
                                for h in captured["handlers"]),
                            captured["handlers"])
        finally:
            set_default_proxy(previous)


def default_proxy_handler_under_test():
    from nomorals.core.http import default_proxy_handler

    return default_proxy_handler()


# ── script generator ─────────────────────────────────────────────────────────


class ScriptGenTest(unittest.TestCase):
    def _gen(self, context, kind: str, name: str, config: dict) -> dict:
        return scriptgen.generate(context, kind, name, config)

    def test_all_kinds_generate_validated_files(self) -> None:
        context, tmp = _make_context()
        try:
            cases = {
                "backup": {"source": "/data/a", "dest": "/backup", "keep": 5},
                "webhook_notify": {"service": "ntfy", "target": "myhost/topic"},
                "jsonl_to_csv": {"src": "in.jsonl", "dst": "out.csv",
                                 "columns": ["a", "b"], "dedupe": True},
                "dedupe_lines": {"src": "f.txt"},
                "log_rotate": {"log": "/tmp/a.log", "keep": 3},
                "termux_service": {"command": "python app.py"},
                "git_autopush": {"repo": "/repo", "branch": "main"},
                "hf_download": {"repo": "owner/model", "dest": "./m"},
                "cron_sh": {"schedule": "0 6 * * *", "command": "echo hi",
                            "name": "job"},
            }
            for kind, cfg in cases.items():
                with self.subTest(kind=kind):
                    out = self._gen(context, kind, kind.replace("_", "-"), cfg)
                    self.assertTrue(out["validated"])
                    self.assertGreater(out["bytes"], 40)
                    self.assertTrue(out["path"].endswith(f".{out['ext']}"))
                    import os
                    self.assertTrue(os.path.isfile(out["path"]))
        finally:
            context.close()
            tmp.cleanup()

    def test_unknown_kind_rejected(self) -> None:
        context, tmp = _make_context()
        try:
            with self.assertRaises(ToolError):
                self._gen(context, "rm_rf", "x", {})
        finally:
            context.close()
            tmp.cleanup()

    def test_bad_name_rejected(self) -> None:
        context, tmp = _make_context()
        try:
            with self.assertRaises(ToolError):
                self._gen(context, "backup", "../escape", {"source": "a", "dest": "b"})
            with self.assertRaises(ToolError):
                self._gen(context, "backup", "", {"source": "a", "dest": "b"})
        finally:
            context.close()
            tmp.cleanup()

    def test_generated_shell_is_really_executable_syntax(self) -> None:
        context, tmp = _make_context()
        try:
            out = self._gen(context, "backup", "bk",
                            {"source": "/a", "dest": "/b", "keep": 2})
            import subprocess
            proc = subprocess.run(["bash", "-n", out["path"]], capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        finally:
            context.close()
            tmp.cleanup()

    def test_tool_roundtrip_with_json_config_string(self) -> None:
        context, tmp = _make_context()
        try:
            reg = _registry_for(context)
            r = reg.call("script_kinds")
            self.assertTrue(r.ok)
            self.assertIn("backup", r.value["kinds"])
            r = reg.call("script_gen", kind="jsonl_to_csv", name="conv",
                         config=json.dumps({"src": "a.jsonl", "dst": "a.csv"}))
            self.assertTrue(r.ok, r.error)
            self.assertTrue(r.value["path"].endswith(".py"))
            r = reg.call("script_gen", kind="backup", name="bad",
                         config="not json")
            self.assertFalse(r.ok)
        finally:
            context.close()
            tmp.cleanup()


# ── osint ────────────────────────────────────────────────────────────────────


class OsintDomainTest(unittest.TestCase):
    def _context(self):
        context, tmp = _make_context()
        self.addCleanup(context.close)
        self.addCleanup(tmp.cleanup)
        return context

    def test_domain_bundle_assembles_dns_registrar_ct(self) -> None:
        context = self._context()

        def fake_dns(domain, record, **kw):
            table = {
                "A": ["1.2.3.4"], "AAAA": [], "NS": ["ns1.example.com"],
                "MX": ["10 mx1.example.com"], "TXT": ["v=spf1 -all"],
                "SPF": ["v=spf1 -all"], "CAA": ["0 issue example-ca"],
            }
            return table.get(record, [])

        ct = [{"common_name": "example.com", "issuer_name": "Fake CA",
               "not_before": "2026-01-01T00:00:00Z", "name_value":
               "example.com\nwww.example.com"}]
        with mock.patch.object(osint_mod, "dns_query", side_effect=fake_dns), \
             mock.patch.object(osint_mod, "rdap_domain", return_value={
                 "registrar": "Fake Registrar", "registration": "1995-09-15"}), \
             mock.patch.object(osint_mod, "HttpClient",
                               return_value=_FakeClient(ct)):
            out = osint_mod.osint_domain(context, "Example.COM.")
        self.assertEqual(out["kind"], "domain")
        self.assertEqual(out["dns"]["A"], ["1.2.3.4"])
        self.assertEqual(out["spf"], "v=spf1 -all")
        self.assertEqual(out["registration"]["registrar"], "Fake Registrar")
        self.assertEqual(out["cert_transparency"]["count"], 1)
        self.assertIn("www.example.com",
                      out["cert_transparency"]["distinct_names"])

    def test_domain_bundle_survives_total_network_failure(self) -> None:
        context = self._context()
        with mock.patch.object(osint_mod, "dns_query",
                               side_effect=ToolError("down")), \
             mock.patch.object(osint_mod, "rdap_domain",
                               side_effect=ToolError("down")), \
             mock.patch.object(osint_mod, "HttpClient",
                               side_effect=ToolError("down")):
            out = osint_mod.osint_domain(context, "example.com")
        self.assertEqual(out["dns"]["A"], [])
        self.assertIn("unavailable", out["registration"])
        self.assertIn("unavailable", out["cert_transparency"])


class OsintIpTest(unittest.TestCase):
    def test_ip_bundle_with_geo_and_abuse_score(self) -> None:
        context, tmp = _make_context()
        self.addCleanup(context.close)
        self.addCleanup(tmp.cleanup)
        context.settings.osint.abuseipdb_key = "K"

        calls: list[str] = []

        def fake_http(context_, url, timeout):  # noqa: ANN001
            calls.append(url)
            if "rdap.org" in url:
                return _FakeResponse({"handle": "H", "entities": []})
            if "ipwho.is" in url:
                return _FakeResponse({"country": "US", "city": "Ashburn",
                                      "connection": {"isp": "Cloud Inc"}})
            return _FakeResponse({"data": {"abuseScore": 42, "totalReports": 7}})

        with mock.patch.object(osint_mod, "dns_query", return_value=["ptr.example.com"]), \
             mock.patch.object(osint_mod, "_http", side_effect=fake_http), \
             mock.patch.object(osint_mod, "HttpClient",
                               return_value=_FakeClient(
                                   {"data": {"abuseScore": 42,
                                             "totalReports": 7}})):
            out = osint_mod.osint_ip(context, "1.2.3.4")
        self.assertEqual(out["kind"], "ip")
        self.assertFalse(out["private"])
        self.assertEqual(out["reverse_dns"], "ptr.example.com")
        self.assertEqual(out["geo"]["city"], "Ashburn")
        self.assertEqual(out["abuseipdb"]["score"], 42)
        self.assertTrue(any("rdap.org" in c for c in calls))

    def test_non_ip_rejected(self) -> None:
        context, tmp = _make_context()
        try:
            with self.assertRaises(ToolError):
                osint_mod.osint_ip(context, "not-an-ip")
        finally:
            context.close()
            tmp.cleanup()


class OsintUrlTest(unittest.TestCase):
    def test_bad_scheme_rejected(self) -> None:
        context, tmp = _make_context()
        try:
            with self.assertRaises(ToolError):
                osint_mod.osint_url(context, "ftp://example.com/x")
        finally:
            context.close()
            tmp.cleanup()

    def test_redirect_chain_followed_without_ssl(self) -> None:
        context, tmp = _make_context()
        self.addCleanup(context.close)
        self.addCleanup(tmp.cleanup)

        opener = mock.Mock()
        opener.build_opener = mock.Mock()

        class Resp:
            def __init__(self, status, headers):
                self.status = status
                self.headers = headers

            def read(self, n=-1):
                return b"body"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        sequence = iter([
            Resp(301, {"Location": "https://final.example/x"}),
            Resp(200, {"Content-Type": "text/html", "Server": "NM/1",
                       "Content-Length": "4"}),
        ])
        with mock.patch.object(osint_mod.urllib.request, "build_opener",
                               return_value=opener), \
             mock.patch.object(opener, "open",
                               side_effect=lambda *a, **kw: next(sequence)):
            out = osint_mod.osint_url(context, "http://start.example/")
        self.assertEqual(len(out["chain"]), 2)
        self.assertEqual(out["chain"][0]["status"], 301)
        self.assertEqual(out["final"]["status"], 200)
        self.assertEqual(out["final"]["body_start"], "body")

    def test_http_error_is_reported_not_raised(self) -> None:
        import urllib.error

        context, tmp = _make_context()
        self.addCleanup(context.close)
        self.addCleanup(tmp.cleanup)
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.HTTPError(
            "http://x/", 403, "Forbidden", {}, None)
        with mock.patch.object(osint_mod.urllib.request, "build_opener",
                               return_value=opener):
            out = osint_mod.osint_url(context, "http://x/")
        self.assertEqual(out["chain"][0]["status"], 403)


class OsintEmailTest(unittest.TestCase):
    def test_email_bundle_passive(self) -> None:
        context, tmp = _make_context()
        try:
            def fake_dns(domain, record, **kw):
                if record == "MX":
                    return ["10 mx.example.com"] if domain == "example.com" else []
                if record == "SPF" and domain == "example.com":
                    return ["v=spf1 -all"]
                if record == "TXT" and domain == "_dmarc.example.com":
                    return ["v=DMARC1; p=reject"]
                return []
            with mock.patch.object(osint_mod, "dns_query", side_effect=fake_dns):
                out = osint_mod.osint_email(context, "User@Example.COM")
        finally:
            context.close()
            tmp.cleanup()
        self.assertEqual(out["kind"], "email")
        self.assertEqual(out["mx"], ["10 mx.example.com"])
        self.assertEqual(out["spf"], "v=spf1 -all")
        self.assertEqual(out["dmarc"], "v=DMARC1; p=reject")
        self.assertIn("passive", out["note"])

    def test_bad_email_rejected(self) -> None:
        context, tmp = _make_context()
        try:
            with self.assertRaises(ToolError):
                osint_mod.osint_email(context, "no-at-sign")
        finally:
            context.close()
            tmp.cleanup()


class OsintReportTest(unittest.TestCase):
    def test_auto_detect(self) -> None:
        context, tmp = _make_context()
        self.addCleanup(context.close)
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(osint_mod, "osint_url") as url, \
             mock.patch.object(osint_mod, "osint_email") as email, \
             mock.patch.object(osint_mod, "osint_ip") as ip, \
             mock.patch.object(osint_mod, "osint_domain") as domain:
            self.assertIs(osint_mod.osint_report(context, "https://a.b/c"), url.return_value)
            url.assert_called_once()
            self.assertIs(osint_mod.osint_report(context, "u@a.b"), email.return_value)
            self.assertIs(osint_mod.osint_report(context, "1.2.3.4"), ip.return_value)
            self.assertIs(osint_mod.osint_report(context, "a.b.co"), domain.return_value)
        with self.assertRaises(ToolError):
            osint_mod.osint_report(context, "###")
        with self.assertRaises(ToolError):
            osint_mod.osint_report(context, "")


# ── macros ──────────────────────────────────────────────────────────────────


class _MacroBase(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.addCleanup(self.context.close)
        self.addCleanup(self.tmp.cleanup)
        self.reg = _registry_for(self.context)
        # never let a test leave a half-open recording for the next one
        self.addCleanup(self._reset_recorder)

    def _reset_recorder(self) -> None:
        with macros._recorder_lock:
            macros._active.pop("current", None)


class MacroRecorderTest(_MacroBase):
    def test_full_lifecycle_creates_live_tool(self) -> None:
        self.assertTrue(self.reg.call("record_start", name="lifecycle").ok)
        r = self.reg.call("record_status")
        self.assertTrue(r.value["recording"])
        r = self.reg.call("record_step", tool="local_services", args="{}")
        self.assertTrue(r.ok)
        self.assertEqual(r.value["steps"], 1)
        r = self.reg.call("record_stop")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.value["tool"], "macro_lifecycle")
        # the macro is now a REAL registered tool
        self.assertIn("macro_lifecycle", self.reg.names())
        r = self.reg.call("macro_lifecycle")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.value["summary"], "1 steps ok")

    def test_explicit_and_devon_steps_mixed(self) -> None:
        self.reg.call("record_start", name="mixed")
        self.reg.call("record_step", tool="local_services")
        # what devon's _execute hook does while a recording is active:
        macros.record_step("dns_lookup", {"domain": "example.com"})
        r = self.reg.call("record_stop")
        self.assertTrue(r.ok)
        self.assertEqual(r.value["steps"], 2)
        r = self.reg.call("macro_show", name="mixed")
        self.assertTrue(r.ok)
        tools = [s["tool"] for s in r.value["steps"]]
        self.assertEqual(tools, ["local_services", "dns_lookup"])

    def test_bad_names_rejected(self) -> None:
        r = self.reg.call("record_start", name="../evil")
        self.assertFalse(r.ok)
        r = self.reg.call("record_start", name="x" * 60)
        self.assertFalse(r.ok)

    def test_stop_without_start_fails(self) -> None:
        r = self.reg.call("record_stop")
        self.assertFalse(r.ok)
        self.assertIn("no recording", str(r.error))

    def test_empty_recording_fails(self) -> None:
        self.reg.call("record_start", name="empty")
        r = self.reg.call("record_stop")
        self.assertFalse(r.ok)

    def test_unknown_tool_rejected_at_save(self) -> None:
        self.reg.call("record_start", name="bad")
        with mock.patch.object(macros, "record_step",
                               side_effect=macros.record_step):
            macros.record_step("definitely_not_a_tool", {})
        r = self.reg.call("record_stop")
        self.assertFalse(r.ok)
        self.assertIn("unknown tool", str(r.error))

    def test_step_without_recording_is_noop(self) -> None:
        out = macros.record_step("local_services", {})
        self.assertFalse(out["recording"])


class MacroReplayTest(_MacroBase):
    def _save(self, name: str, steps: list[dict]) -> None:
        macros.save_macro(self.context, self.reg, name, "test", steps)

    def test_replay_reports_each_step(self) -> None:
        self._save("replay", [{"tool": "local_services", "args": {}}])
        r = self.reg.call("macro_run", name="replay")
        self.assertTrue(r.ok, r.error)
        self.assertTrue(r.value["ok"])
        self.assertEqual(len(r.value["steps"]), 1)
        # run counter persisted
        r = self.reg.call("macro_show", name="replay")
        self.assertGreaterEqual(r.value["runs"], 1)

    def test_replay_stops_at_first_failure(self) -> None:
        # step 1 fails the scope guard (no network involved, deterministic)
        self._save("fail", [
            {"tool": "port_scan", "args": {"target": "8.8.8.8", "ports": "22"}},
            {"tool": "local_services", "args": {}},
        ])
        r = self.reg.call("macro_run", name="fail")
        self.assertTrue(r.ok, r.error)  # the macro ran; its STEP failed
        self.assertFalse(r.value["ok"])
        self.assertEqual(len(r.value["steps"]), 1, "must stop at first failure")

    def test_overrides_replace_step_args(self) -> None:
        # stored step is out of scope (would fail); the override swaps it for a
        # hermetic own-machine scan — proving overrides actually replace args
        self._save("ovr", [{"tool": "port_scan",
                            "args": {"target": "8.8.8.8", "ports": "22"}}])
        r = self.reg.call("macro_run", name="ovr",
                          overrides=json.dumps(
                              {"0": {"target": "127.0.0.1", "ports": "22"}}))
        self.assertTrue(r.ok, r.error)
        self.assertTrue(r.value["ok"], r.value)
        self.assertIn("127.0.0.1", r.value["steps"][0]["result"])

    def test_overrides_bad_json_rejected(self) -> None:
        self._save("ovr2", [{"tool": "local_services", "args": {}}])
        r = self.reg.call("macro_run", name="ovr2", overrides="not json")
        self.assertFalse(r.ok)
        r = self.reg.call("macro_run", name="ovr2", overrides='{"0": "x"}')
        self.assertFalse(r.ok)

    def test_missing_macro_rejected(self) -> None:
        r = self.reg.call("macro_run", name="ghost")
        self.assertFalse(r.ok)
        self.assertIn("no macro", str(r.error))

    def test_list_show_delete(self) -> None:
        self._save("one", [{"tool": "local_services", "args": {}}])
        self._save("two", [{"tool": "local_services", "args": {}}])
        r = self.reg.call("macro_list")
        names = {m["name"] for m in r.value["macros"]}
        self.assertGreaterEqual(names, {"one", "two"})
        r = self.reg.call("macro_delete", name="one")
        self.assertTrue(r.ok)
        r = self.reg.call("macro_list")
        names = {m["name"] for m in r.value["macros"]}
        self.assertNotIn("one", names)
        self.assertIn("two", names)
        # the live tool is gone too
        self.assertNotIn("macro_one", self.reg.names())

    def test_rerun_replaces_and_keeps_single_tool(self) -> None:
        self._save("dup", [{"tool": "local_services", "args": {}}])
        before = self.reg.names().count("macro_dup")
        macros.save_macro(self.context, self.reg, "dup", "v2",
                          [{"tool": "local_services", "args": {}}])
        self.assertEqual(self.reg.names().count("macro_dup"), before)
        r = self.reg.call("macro_dup")
        self.assertTrue(r.ok)


class MacroBootTest(unittest.TestCase):
    def test_stored_macros_are_registered_at_boot(self) -> None:
        context, tmp = _make_context()
        try:
            macros.save_macro(context, None, "booted", "desc",
                              [{"tool": "local_services", "args": {}}])
            # simulate a process restart: fresh registry, register() re-runs
            reg = ToolRegistry()
            reg.context = context
            reg.register_builtins()
            self.assertIn("macro_booted", reg.names())
        finally:
            context.close()
            tmp.cleanup()


# ── registry dispatch ────────────────────────────────────────────────────────


class RegistryDispatchTest(_MacroBase):
    def test_positional_only_name_avoids_kwarg_collision(self) -> None:
        # tools that have their own `name` / `tool` parameters
        r = self.reg.call("record_start", name="kwname")
        self.assertTrue(r.ok)
        r = self.reg.call("record_step", tool="local_services", args="{}")
        self.assertTrue(r.ok)
        self.reg.call("record_stop")

    def test_extra_positional_args_rejected(self) -> None:
        r = self.reg.call("local_services", "stray")
        self.assertFalse(r.ok)
        self.assertIn("keywords", str(r.error))

    def test_unknown_tool_lists_available(self) -> None:
        r = self.reg.call("definitely_missing")
        self.assertFalse(r.ok)
        self.assertIn("available", str(r.error))


# ── settings / env wiring ────────────────────────────────────────────────────


class PowerSettingsEnvTest(unittest.TestCase):
    def test_all_new_env_vars_reach_settings(self) -> None:
        s = load_settings(use_env_file=False, env={
            "NM_NET_PROBE_TIMEOUT": "2.5",
            "NM_NET_ALLOWED_TARGETS": "10.0.0.5, myphone.local",
            "NM_NET_DEFAULT_PORTS": "22, 443",
            "NM_NET_MAX_PROBE_PORTS": "64",
            "NM_NET_BANNER": "true",
            "NM_PROXY_ACTIVE": "socks5://127.0.0.1:9050",
            "NM_PROXY_KNOWN": "http://p1:8080, socks5://p2:9050",
            "NM_PROXY_URL": "http://p3:9090",
            "NM_OSINT_ABUSEIPDB_KEY": "k",
            "NM_OSINT_SHODAN_KEY": "s",
            "NM_OSINT_REQUEST_TIMEOUT": "9",
            "NM_OSINT_CRTSH_DAYS": "100",
        })
        self.assertEqual(s.net.probe_timeout, 2.5)
        self.assertEqual(s.net.allowed_targets, "10.0.0.5, myphone.local")
        self.assertEqual(s.net.default_ports, "22, 443")
        self.assertEqual(s.net.max_probe_ports, 64)
        self.assertTrue(s.net.banner)
        self.assertEqual(s.proxy.active, "socks5://127.0.0.1:9050")
        self.assertEqual(s.proxy.known, "http://p1:8080, socks5://p2:9050")
        self.assertEqual(s.proxy.url, "http://p3:9090")
        self.assertEqual(s.osint.abuseipdb_key, "k")
        self.assertEqual(s.osint.shodan_key, "s")
        self.assertEqual(s.osint.request_timeout, 9.0)
        self.assertEqual(s.osint.crtsh_days, 100)

    def test_defaults_are_conservative(self) -> None:
        s = load_settings(use_env_file=False, env={})
        self.assertEqual(s.net.allowed_targets, "")
        self.assertFalse(s.net.banner)
        self.assertEqual(s.proxy.active, "")
        self.assertEqual(s.proxy.url, "")
        self.assertEqual(s.osint.abuseipdb_key, "")


class ControlTableTest(unittest.TestCase):
    def test_power_commands_registered_with_limits(self) -> None:
        from nomorals.social.chat.control import CONTROL_COMMANDS, help_text

        for kind in ("dns", "scan", "whois", "ports", "proxy", "gen",
                     "osint", "record", "macro"):
            self.assertIn(kind, CONTROL_COMMANDS, kind)
        self.assertEqual(CONTROL_COMMANDS["ports"], (0, 1))
        self.assertEqual(CONTROL_COMMANDS["scan"], (1, 6))
        self.assertEqual(CONTROL_COMMANDS["whois"], (1, 3))
        text = help_text()
        for token in ("/dns", "/scan", "/whois", "/ports", "/proxy",
                      "/gen", "/osint", "/record", "/macro"):
            self.assertIn(token, text, token)

    def test_parse_new_commands(self) -> None:
        from nomorals.social.chat.control import parse_control

        c = parse_control("/dns example.com MX")
        self.assertEqual(c.kind, "dns")
        self.assertEqual(c.tail, "example.com MX")
        c = parse_control("/scan 127.0.0.1 22-25 banner")
        self.assertEqual(c.kind, "scan")
        c = parse_control("/proxy set socks5://h:9050")
        self.assertEqual(c.kind, "proxy")
        c = parse_control("/record stop")
        self.assertEqual(c.kind, "record")


class DevonWiringTest(unittest.TestCase):
    def test_catalog_and_tool_methods_exist(self) -> None:
        from nomorals.agents.devon import TOOL_CATALOG, DevonAgent

        names = {name for name, _ in TOOL_CATALOG}
        for needed in ("dns_lookup", "whois_lookup", "local_services",
                       "osint_report", "proxy_status", "proxy_set",
                       "script_gen", "macro_run"):
            self.assertIn(needed, names)
        context, tmp = _make_context()
        try:
            reg = _registry_for(context)
            agent = DevonAgent(context)
            for needed in ("dns_lookup", "whois_lookup", "local_services",
                           "osint_report", "proxy_status", "proxy_set",
                           "script_gen", "macro_run"):
                self.assertIn(needed, agent._tools)
        finally:
            context.close()
            tmp.cleanup()

    def test_devon_execute_records_while_recording(self) -> None:
        context, tmp = _make_context()
        self.addCleanup(context.close)
        self.addCleanup(self._reset)
        self.addCleanup(tmp.cleanup)
        reg = _registry_for(context)
        from nomorals.agents.devon import DevonAgent

        agent = DevonAgent(context)
        self.assertTrue(reg.call("record_start", name="devrec").ok)
        outcome = agent._execute("local_services", {})
        self.assertTrue(outcome.ok)
        r = reg.call("record_status")
        self.assertTrue(r.value["recording"])
        self.assertEqual(r.value["steps"], 1)
        self.assertEqual(r.value["last"], "local_services")
        reg.call("record_stop")

    def _reset(self) -> None:
        with macros._recorder_lock:
            macros._active.pop("current", None)


class ControlCommandE2ETest(unittest.TestCase):
    """The /dns /scan /ports /proxy /gen /osint /record /macro chat commands,
    driven through PartnerRuntime exactly the way the dispatch invokes them."""

    def setUp(self) -> None:
        from tests.test_partner_runtime import FakeAdapter, FakeRouter
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.social.chat import ChatGateway, ChatKind, ChatRef

        self.context, self.tmp = _make_context()
        self.addCleanup(self.context.close)
        self.addCleanup(self.tmp.cleanup)
        self.context.router = FakeRouter()
        self.adapter = FakeAdapter("local")
        self.gateway = ChatGateway({"local": self.adapter}, db=self.context.db)
        self.chat = ChatRef(platform="local", chat_id="console",
                            kind=ChatKind.DM, peer="you")
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.reg = _registry_for(self.context)
        self.key = "local:console"

    def _sent(self) -> list[str]:
        return [t for t in self.adapter.sent if t]

    def test_dns_command(self) -> None:
        with mock.patch.object(network, "dns_query",
                               return_value=["93.184.216.34"]):
            reply = self.runtime._control_dns("example.com A", chat_key=self.key)
        self.assertIn("example.com A:", reply)
        self.assertIn("93.184.216.34", reply)

    def test_dns_command_failure_is_a_message(self) -> None:
        with mock.patch.object(network, "dns_query",
                               side_effect=ToolError("resolver down")):
            reply = self.runtime._control_dns("example.com A", chat_key=self.key)
        self.assertIn("dns failed", reply)

    def test_scan_command_against_own_machine(self) -> None:
        listener = _LocalListener()
        self.addCleanup(listener.close)
        reply = self.runtime._control_scan(f"127.0.0.1 {listener.port}",
                                           chat_key=self.key)
        self.assertIn("scan 127.0.0.1", reply)
        self.assertIn(str(listener.port), reply)
        self.assertIn("own-infrastructure", reply)

    def test_scan_command_refuses_foreign_target(self) -> None:
        reply = self.runtime._control_scan("8.8.8.8 22", chat_key=self.key)
        self.assertIn("outside the allowed scope", reply)

    def test_ports_command(self) -> None:
        reply = self.runtime._control_ports("", chat_key=self.key)
        self.assertIn("listening on this machine", reply)

    def test_proxy_command_cycle(self) -> None:
        from nomorals.core.http import get_default_proxy, set_default_proxy

        self.addCleanup(lambda: set_default_proxy(""))
        status = self.runtime._control_proxy("status", chat_key=self.key)
        self.assertIn("direct (no proxy)", status)
        set = self.runtime._control_proxy("set http://10.9.9.9:8080",
                                          chat_key=self.key)
        self.assertEqual(get_default_proxy(), "http://10.9.9.9:8080")
        # the long reply goes through the gateway; the returned string is ""
        self.assertEqual(set, "")
        self.assertTrue(any("http://10.9.9.9:8080" in t for t in self._sent()),
                        self._sent())
        clear = self.runtime._control_proxy("clear", chat_key=self.key)
        self.assertEqual(clear, "")
        self.assertEqual(get_default_proxy(), "")
        status = self.runtime._control_proxy("status", chat_key=self.key)
        self.assertIn("direct (no proxy)", status)

    def test_gen_command_lists_kinds_and_generates(self) -> None:
        listing = self.runtime._control_gen("", chat_key=self.key)
        for kind in ("backup", "cron_sh", "termux_service", "webhook_notify",
                     "jsonl_to_csv", "log_rotate", "dedupe_lines",
                     "git_autopush", "hf_download"):
            self.assertIn(kind, listing)
        reply = self.runtime._control_gen(
            "backup nightly {\"source\": \"/a\", \"dest\": \"/b\", \"keep\": 3}",
            chat_key=self.key)
        self.assertIn("generated", reply)
        self.assertIn("validated", reply)
        bad = self.runtime._control_gen("nope x", chat_key=self.key)
        self.assertIn("gen failed", bad)

    def test_osint_command_streams_report(self) -> None:
        fake = {"target": "example.com", "kind": "domain",
                "dns": {"A": ["1.2.3.4"]}, "spf": "v=spf1 -all"}
        with mock.patch.object(osint_mod, "osint_report", return_value=fake):
            reply = self.runtime._control_osint("example.com", chat_key=self.key)
        self.assertEqual(reply, "")
        streamed = "\n".join(self._sent())
        self.assertIn("osint report: example.com", streamed)
        self.assertIn("1.2.3.4", streamed)
        self.assertIn("v=spf1 -all", streamed)

    def test_record_macro_cycle_through_chat(self) -> None:
        start = self.runtime._control_record("start chatdemo", chat_key=self.key)
        self.assertIn("recording chatdemo", start)
        step = self.runtime._control_record("step local_services {}",
                                            chat_key=self.key)
        self.assertIn("step 1 added", step)
        status = self.runtime._control_record("status", chat_key=self.key)
        self.assertIn("1 steps", status)
        stop = self.runtime._control_record("stop", chat_key=self.key)
        self.assertIn("saved macro", stop)
        listing = self.runtime._control_macro("list", chat_key=self.key)
        self.assertIn("chatdemo", listing)
        # replay streams per-step results through the gateway
        self.runtime._control_macro("chatdemo", chat_key=self.key)
        streamed = "\n".join(self._sent())
        self.assertIn("chatdemo: 1 steps ok", streamed)
        self.assertIn("local_services", streamed)

    def test_devon_power_tool_via_execute(self) -> None:
        # the model-facing surface: devon's planner can now call dns_lookup
        from nomorals.agents.devon import DevonAgent

        agent = DevonAgent(self.context)
        with mock.patch.object(network, "dns_query",
                               return_value=["1.1.1.1"]):
            outcome = agent._execute("dns_lookup", {"domain": "example.com"})
        self.assertTrue(outcome.ok, outcome.error)
        self.assertIn("1.1.1.1", outcome.observation)


class ProxyBootRestoreTest(unittest.TestCase):
    def test_build_context_restores_persisted_proxy(self) -> None:
        from nomorals.agents.context import build_context
        from nomorals.core.http import get_default_proxy, set_default_proxy

        set_default_proxy("")
        self.addCleanup(set_default_proxy)

        context, tmp = _make_context()
        try:
            context.db.execute(
                "INSERT OR REPLACE INTO kv_store (key, value, kind, updated_at) "
                "VALUES (?, ?, 'json', ?)",
                ("proxy.active", json.dumps({"proxy": "http://10.1.1.1:8080"}),
                 time.time()),
            )
            fresh = build_context(context.settings, with_executor=False,
                                  with_tools=False)
            self.assertEqual(get_default_proxy(), "http://10.1.1.1:8080")
            fresh.close()
        finally:
            set_default_proxy("")
            context.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
