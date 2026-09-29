"""Wave 71 — Universal Decoder + MonitorAgent + KG curation + CLI.

Hermetic: no network. URL fetches are monkeypatched; the binary
fixtures are generated in-process; CLI checks run as subprocesses
against a throwaway NM_HOME.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest import mock

from tests.test_deep_research import _make_context
from tests.test_search_and_model import _FakeRouter

from nomorals.agents.context import build_context
from nomorals.core.config import load_settings
from nomorals.core.decoder import (DECODERS, analyze, decode_any,
                                   identify_hash, identify_magic,
                                   known_hash_lookup, shannon_entropy)
from nomorals.core.policy import CapabilitySet


def _settings(tmp: str) -> Any:
    return load_settings(overrides={"home": tmp, "partner.platforms": "local",
                                    "chat.local_enabled": "true"})


def _ctx(tmp: str) -> Any:
    context = build_context(_settings(tmp), with_executor=False,
                            with_tools=True, with_router=False)
    context.router = _FakeRouter()
    return context


def _as_bytes(out: Any) -> bytes:
    """Decoder outputs are str when printable, bytes otherwise."""
    return out if isinstance(out, (bytes, bytearray)) else str(out).encode()


# ────────────────────────────── core decoders ────────────────────────────────

class CoreDecoderTests(unittest.TestCase):
    def test_base64_text(self) -> None:
        report = analyze(base64.b64encode(b"hello code beast").decode())
        self.assertIsNotNone(report.best)
        self.assertEqual(report.best["output"], "hello code beast")
        self.assertEqual(report.best["chain"][0], "base64")

    def test_base64url(self) -> None:
        # url-safe alphabet, padding stripped
        b64url = base64.urlsafe_b64encode(
            b"hello code beast").decode().rstrip("=")
        report = analyze(b64url)
        self.assertIsNotNone(report.best)
        self.assertEqual(report.best["output"], "hello code beast")
        self.assertEqual(report.best["chain"][0], "base64")

    def test_hex_forms(self) -> None:
        raw = b"code beast"
        for enc in (raw.hex(), "0x" + raw.hex(),
                    "\\x" + "\\x".join(f"{b:02x}" for b in raw)):
            report = analyze(enc)
            self.assertIsNotNone(report.best, f"hex form {enc!r}")
            self.assertEqual(_as_bytes(report.best["output"]), raw)

    def test_hexdump(self) -> None:
        raw = b"hello code beast!"
        lines = []
        for off in range(0, len(raw), 16):
            chunk = raw[off:off + 16]
            lines.append(f"{off:08x}: " +
                         " ".join(f"{b:02x}" for b in chunk))
        report = analyze("\n".join(lines))
        self.assertIsNotNone(report.best)
        self.assertEqual(_as_bytes(report.best["output"]), raw)

    def test_base32(self) -> None:
        raw = b"code beast 123"
        encoded = base64.b32encode(raw).decode().rstrip("=")
        report = analyze(encoded)
        self.assertIsNotNone(report.best)
        self.assertEqual(_as_bytes(report.best["output"]), raw)

    def test_binary_bits(self) -> None:
        raw = b"hi"
        bits = "".join(f"{b:08b}" for b in raw)
        report = analyze(bits)
        self.assertIsNotNone(report.best)
        self.assertEqual(_as_bytes(report.best["output"]), raw)

    def test_rot13_wordy(self) -> None:
        cipher = "pbyq jryr onex"  # "obef wher lbha" rot13 → not wordy
        plain = "gur dhvpx"  # rot13 → "the shift"
        report = analyze(plain)
        self.assertIsNotNone(report.best)
        self.assertIn("rot13", report.best["chain"])

    def test_reverse(self) -> None:
        report = analyze("tsaeb edoc olleh")
        self.assertIsNotNone(report.best)
        self.assertEqual(report.best["output"], "hello code beast")

    def test_url_encoding(self) -> None:
        report = analyze("hello%20code%20beast")
        self.assertIsNotNone(report.best)
        self.assertEqual(report.best["output"], "hello code beast")

    def test_html_entities(self) -> None:
        report = analyze("code&amp;beast&nbsp;rules")
        self.assertIsNotNone(report.best)
        self.assertIn("&", str(report.best["output"]))

    def test_leet(self) -> None:
        report = analyze("p@ssw0rd")
        self.assertIsNotNone(report.best)
        self.assertEqual(report.best["output"], "password")

    def test_leet_repeat_run_rejected(self) -> None:
        # "card4111111111111111ok" must NOT leet-decode into garbage
        report = analyze("card4111111111111111ok")
        if report.best:
            self.assertNotIn("leet", report.best["chain"])

    def test_nested_chain_base64_rot13(self) -> None:
        import codecs
        plain = "the quick brown fox jumps"
        rot = codecs.decode(plain, "rot13")
        enc = base64.b64encode(rot.encode()).decode()
        report = analyze(enc, max_depth=3)
        self.assertIsNotNone(report.best)
        self.assertEqual(report.best["output"], plain)
        self.assertGreaterEqual(len(report.best["chain"]), 2)

    def test_gzip_binary(self) -> None:
        payload = b"gunzipped code beast payload"
        blob = gzip.compress(payload)
        report = analyze(blob)
        self.assertIsNotNone(report.best)
        self.assertEqual(_as_bytes(report.best["output"]), payload)

    def test_fp_gate_random_alnum(self) -> None:
        # random alphanumeric ids should not confidently "decode" to text
        noise = "aB3kQ9zXw7Lm2pRt5Yv8Nj4sFh1Gd6cUo0Ie"
        report = analyze(noise)
        if report.best is not None:
            self.assertLess(report.best["confidence"], 0.85)

    def test_luhn_card(self) -> None:
        report = analyze("card 4111 1111 1111 1111 ok")
        self.assertIsNotNone(report.best)
        self.assertEqual(report.best["chain"][0], "luhn")

    def test_identify_magic(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        self.assertEqual(identify_magic(png), "png")
        self.assertEqual(identify_magic(b"PK\x03\x04" + b"\x00" * 8), "zip")
        self.assertEqual(identify_magic(b"%PDF-1.7"), "pdf")
        gz = gzip.compress(b"abc")
        self.assertEqual(identify_magic(gz), "gzip")
        self.assertEqual(identify_magic(b"\x00" * 32), "")

    def test_identify_hash_candidates(self) -> None:
        md5 = hashlib.md5(b"x").hexdigest()
        sha1 = hashlib.sha1(b"x").hexdigest()
        sha256 = hashlib.sha256(b"x").hexdigest()
        self.assertIn("md5", identify_hash(md5))
        self.assertIn("sha1", identify_hash(sha1))
        self.assertIn("sha256", identify_hash(sha256))
        self.assertEqual(identify_hash("not-a-digest"), [])

    def test_known_hash_lookup(self) -> None:
        md5 = hashlib.md5(b"password").hexdigest()
        known = known_hash_lookup(md5)
        self.assertIsNotNone(known)
        self.assertEqual(known["plaintext"], "password")
        self.assertEqual(known["algorithm"], "md5")
        self.assertIsNone(known_hash_lookup(
            hashlib.md5(b"zzz-unknown").hexdigest()))

    def test_report_hash_field(self) -> None:
        md5 = hashlib.md5(b"password").hexdigest()
        report = analyze(md5)
        self.assertIsNotNone(report.hash)
        self.assertEqual(report.hash["known"]["plaintext"], "password")

    def test_cookie_parse(self) -> None:
        line = "Set-Cookie: session=abc123; Path=/; HttpOnly; Max-Age=3600"
        report = analyze(line)
        self.assertTrue(report.cookies)
        self.assertEqual(report.cookies[0]["name"], "session")
        self.assertEqual(report.cookies[0]["value"], "abc123")
        flags = [c.get("flag") for c in report.cookies]
        self.assertIn("HttpOnly", flags)

    def test_cookie_fp_gate(self) -> None:
        # a single key=value with no cookie flags is not a cookie
        report = analyze("user=bob")
        self.assertFalse(report.cookies)

    def test_jwt_decode(self) -> None:
        header = base64.urlsafe_b64encode(
            b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(
            b'{"sub":"user-1","role":"admin","exp":1999999999}').rstrip(b"=").decode()
        token = f"{header}.{payload}.c2lnbmF0dXJl"
        report = analyze(token)
        self.assertIsNotNone(report.jwt)
        self.assertEqual(report.jwt["payload"]["role"], "admin")
        self.assertEqual(report.jwt["header"]["alg"], "HS256")

    def test_token_patterns_masked(self) -> None:
        key = "sk-" + "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        report = analyze(f"using key {key} for calls")
        self.assertTrue(report.tokens)
        self.assertTrue(any("OpenAI" in t["label"] for t in report.tokens))
        samples = " ".join(t["sample"] for t in report.tokens)
        self.assertNotIn(key[4:], samples)  # masked

    def test_forensics_fields(self) -> None:
        report = analyze("hello world " * 4)
        f = report.forensics
        for key in ("chars", "bytes", "entropy", "printable_ratio",
                    "charset", "magic", "top_chars", "looks_like_text"):
            self.assertIn(key, f)
        self.assertGreater(f["entropy"], 0)
        self.assertEqual(f["magic"], "")

    def test_entropy_bounds(self) -> None:
        self.assertLess(shannon_entropy("aaaa"),
                        shannon_entropy("abcde fghij"))
        self.assertGreaterEqual(shannon_entropy("a"), 0.0)

    def test_decoders_registry(self) -> None:
        names = {d.name for d in DECODERS}
        for required in ("hex", "base64", "base32", "rot13", "leet",
                         "jwt", "cookies", "gzip", "morse", "uuid",
                         "key-value", "ini", "csv", "pem", "mime"):
            self.assertIn(required, names)
        self.assertGreaterEqual(len(DECODERS), 25)

    def test_structured_json_yaml_kv(self) -> None:
        report = analyze('{"name": "code beast", "n": 3}')
        self.assertIsNotNone(report.best)
        self.assertIsInstance(report.best["output"], dict)
        self.assertEqual(report.best["output"]["name"], "code beast")

        rep2 = analyze("[server]\nhost = 10.0.0.1\nport = 2222\n")
        self.assertIsNotNone(rep2.best)
        self.assertIn("ini", rep2.best["chain"])

        rep3 = analyze("host=10.0.0.1\nport=2222")
        self.assertIsNotNone(rep3.best)
        self.assertIn("key-value", rep3.best["chain"])
        self.assertEqual(rep3.best["output"]["host"], "10.0.0.1")

    def test_morse(self) -> None:
        report = analyze(".... . .-.. .-.. ---")  # "hello"
        self.assertIsNotNone(report.best)
        self.assertEqual(report.best["output"], "hello")

    def test_uuid_and_luhn_in_analyze(self) -> None:
        report = analyze("id 123e4567-e89b-12d3-a456-426614174000 end")
        self.assertIsNotNone(report.best)
        self.assertIn("uuid", report.best["chain"])


# ────────────────────────────── tool layer ───────────────────────────────────

class DecoderToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="w71-tool-")
        self.ctx = _ctx(self.tmp.name)
        from nomorals.tools.registry import ToolRegistry

        self.reg = ToolRegistry(self.ctx).register_builtins()
        self.caps = CapabilitySet.all()

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_decoder_tool_text(self) -> None:
        out = self.reg.call("decoder", data=base64.b64encode(b"password").decode(),
                            capabilities=self.caps)
        self.assertTrue(out.ok, out.error)
        v = out.unwrap()
        self.assertEqual(v["report"]["best"]["output"], "password")
        self.assertIn("base64", v["summary"])

    def test_decoder_tool_hash(self) -> None:
        md5 = hashlib.md5(b"password").hexdigest()
        out = self.reg.call("decoder", data=md5, capabilities=self.caps)
        self.assertTrue(out.ok)
        v = out.unwrap()
        self.assertEqual(v["report"]["hash"]["known"]["plaintext"],
                         "password")

    def test_decoder_tool_path(self) -> None:
        ws = os.path.join(self.tmp.name, "workspace")
        os.makedirs(ws, exist_ok=True)
        with open(os.path.join(ws, "sample.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(base64.b64encode(b"from file").decode())
        out = self.reg.call("decoder", path="sample.txt",
                            capabilities=self.caps)
        self.assertTrue(out.ok, out.error)
        self.assertEqual(out.unwrap()["report"]["best"]["output"],
                         "from file")

    def test_decoder_tool_decoders_mode(self) -> None:
        out = self.reg.call("decoder", mode="decoders", capabilities=self.caps)
        self.assertTrue(out.ok)
        v = out.unwrap()
        self.assertGreaterEqual(v["count"], 25)

    def test_decoder_tool_requires_input(self) -> None:
        out = self.reg.call("decoder", capabilities=self.caps)
        self.assertFalse(out.ok)

    def test_identify_hash_tool(self) -> None:
        sha = hashlib.sha256(b"codebeast").hexdigest()
        out = self.reg.call("identify_hash", digest=sha, capabilities=self.caps)
        self.assertTrue(out.ok, out.error)
        v = out.unwrap()
        self.assertIn("sha256", v["candidates"])
        self.assertEqual(v["known"]["plaintext"], "codebeast")

    def test_decoder_registered_in_catalog(self) -> None:
        names = self.reg.names()
        self.assertIn("decoder", names)
        self.assertIn("decoder_agent", names)
        self.assertIn("identify_hash", names)
        self.assertIn("monitor", names)


# ────────────────────────────── agent layer ──────────────────────────────────

class DecoderAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="w71-agen-")
        self.ctx = _ctx(self.tmp.name)
        from nomorals.agents.decoder import DecoderAgent

        self.agent = DecoderAgent(context=self.ctx, name="t")

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_text_run(self) -> None:
        r = self.agent.run({"data": base64.b64encode(b"hello code beast").decode()})
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.output["report"]["best"]["output"], "hello code beast")
        self.assertTrue(r.output["explanation"])

    def test_binary_save_with_magic_ext(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
        enc = base64.b64encode(png).decode()
        r = self.agent.run({"data": enc, "save": True})
        self.assertTrue(r.ok)
        saved = r.output["saved_to"]
        self.assertTrue(saved)
        self.assertTrue(saved.endswith(".png"), saved)
        with open(saved, "rb") as fh:
            self.assertEqual(fh.read(), png)

    def test_file_task(self) -> None:
        ws = os.path.join(self.tmp.name, "workspace")
        os.makedirs(ws, exist_ok=True)
        payload = gzip.compress(b"file payload code")
        with open(os.path.join(ws, "blob.bin"), "wb") as fh:
            fh.write(payload)
        r = self.agent.run("file:blob.bin")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(_as_bytes(r.output["report"]["best"]["output"]),
                         b"file payload code")

    def test_action_hash_and_decoders(self) -> None:
        out = reg_call(self.ctx, "decoder_agent", action="hash",
                       data=hashlib.md5(b"password").hexdigest())
        self.assertEqual(out.unwrap()["known"]["plaintext"], "password")
        out = reg_call(self.ctx, "decoder_agent", action="decoders")
        self.assertGreaterEqual(out.unwrap()["count"], 25)

    def test_explain_rule_based(self) -> None:
        r = self.agent.run({"data": base64.b64encode(b"secret").decode(),
                            "explain": True})
        self.assertIn("base64", r.output["explanation"])

    def test_kg_curation_side_effect(self) -> None:
        from nomorals.agents.kg import KnowledgeGraph

        md5 = hashlib.md5(b"password").hexdigest()
        r = self.agent.run(
            {"data": f"log line {md5} host https://example.com/a x@y.io",
             "explain": False})
        self.assertTrue(r.ok)
        g = KnowledgeGraph(self.ctx.db)
        labels = {n["label"] for n in self.ctx.db.query(
            "SELECT label FROM kg_nodes")}
        self.assertIn(md5, labels)
        self.assertIn("secret:password", labels)
        self.assertIn("https://example.com/a", labels)
        rels = {e["relation"] for e in self.ctx.db.query(
            "SELECT relation FROM kg_edges")}
        self.assertIn("decodes_to", rels)

    def test_agent_requires_input(self) -> None:
        r = self.agent.run({})
        self.assertFalse(r.ok)


def reg_call(ctx: Any, name: str, **kw: Any) -> Any:
    from nomorals.tools.registry import ToolRegistry

    if not hasattr(ctx, "_w71_reg"):
        ctx._w71_reg = ToolRegistry(ctx).register_builtins()
    return ctx._w71_reg.call(name, capabilities=CapabilitySet.all(), **kw)


# ────────────────────────────── monitor agent ────────────────────────────────

class _StubNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def publish(self, kind: str, title: str, body: str = "",
                **kw: Any) -> Any:
        self.sent.append((kind, title, body))
        return True


class MonitorAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="w71-mon-")
        self.ctx = _ctx(self.tmp.name)
        self.notifier = _StubNotifier()
        from nomorals.agents.monitor import MonitorAgent

        self.agent = MonitorAgent(self.ctx, notifier=self.notifier)
        self.ws = os.path.join(self.tmp.name, "workspace")
        os.makedirs(self.ws, exist_ok=True)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def _file(self, name: str, text: str) -> str:
        p = os.path.join(self.ws, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        return name

    def test_add_auto_kind(self) -> None:
        info = self.agent.add("https://example.com/x")
        self.assertEqual(info["kind"], "url")
        name = self._file("a.txt", "one")
        info = self.agent.add(name)
        self.assertEqual(info["kind"], "file")

    def test_add_dedupes_by_target(self) -> None:
        a = self.agent.add("https://example.com/x")
        b = self.agent.add("https://example.com/x", interval=999)
        self.assertFalse(b["created"])
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(self.agent.list()[0]["interval_s"], 999.0)

    def test_interval_floor(self) -> None:
        info = self.agent.add("https://example.com/x", interval=2)
        self.assertGreaterEqual(info["interval_s"], 30.0)

    def test_file_change_detected_with_diff(self) -> None:
        name = self._file("watch.txt", "alpha\nbeta\n")
        self.agent.add(name, interval=30)
        first = self.agent.tick(now=1000.0)
        self.assertEqual(first["checked"], 1)
        self.assertEqual(first["changed"], [])
        # not yet due
        second = self.agent.tick(now=1010.0)
        self.assertEqual(second["checked"], 0)
        # due + changed
        with open(os.path.join(self.ws, name), "w", encoding="utf-8") as fh:
            fh.write("alpha\ngamma\n")
        third = self.agent.tick(now=2000.0)
        self.assertEqual(len(third["changed"]), 1)
        entry = third["changed"][0]
        self.assertIn("-beta", entry["diff"])
        self.assertIn("+gamma", entry["diff"])
        self.assertEqual(self.notifier.sent[-1][1], f"change: {name}")

    def test_file_no_change_no_alert(self) -> None:
        name = self._file("steady.txt", "same\n")
        self.agent.add(name, interval=30)
        self.agent.tick(now=1000.0)
        before = len(self.notifier.sent)
        self.agent.tick(now=5000.0)
        self.assertEqual(len(self.notifier.sent), before)

    def test_remove_and_find_by_ref(self) -> None:
        name = self._file("rm.txt", "x")
        self.agent.add(name)
        self.assertTrue(self.agent.remove(name))
        self.assertFalse(self.agent.remove(name))
        self.assertEqual(self.agent.list(), [])

    def test_enable_disable(self) -> None:
        name = self._file("onoff.txt", "x")
        self.agent.add(name)
        self.agent.set_enabled(name, False)
        self.assertFalse(self.agent.list()[0]["enabled"])
        self.assertEqual(self.agent.tick(now=1000.0)["checked"], 0)
        self.agent.set_enabled(name, True)
        self.assertTrue(self.agent.list()[0]["enabled"])

    def test_status(self) -> None:
        self.agent.add("https://a.example/")
        self.agent.add("https://b.example/")
        st = self.agent.status()
        self.assertEqual(st["total"], 2)
        self.assertEqual(st["enabled"], 2)

    def test_url_change_via_patched_fetch(self) -> None:
        from nomorals.core import http as http_mod

        self.agent.add("https://example.com/page", interval=30)

        calls = {"n": 0}

        def fake_get(self_: Any, url: str, **kw: Any) -> Any:
            calls["n"] += 1
            body = b"v1" if calls["n"] == 1 else b"v2-changed"
            return http_mod.HttpResponse(status=200, body=body, url=url)

        with mock.patch.object(http_mod.HttpClient, "get", fake_get):
            self.agent.tick(now=100.0)
            first = self.agent.tick(now=200.0)
        self.assertEqual(len(first["changed"]), 1)
        self.assertEqual(first["changed"][0]["new_size"], len(b"v2-changed"))

    def test_url_error_streak_alert(self) -> None:
        from nomorals.core import http as http_mod

        self.agent.add("https://dead.example/x", interval=30)

        def dead(self_: Any, url: str, **kw: Any) -> Any:
            raise ConnectionError("no egress in sandbox")

        with mock.patch.object(http_mod.HttpClient, "get", dead):
            self.agent.tick(now=100.0)
            self.agent.tick(now=200.0)
            self.assertEqual(len(self.notifier.sent), 0)  # streak 2, quiet
            self.agent.tick(now=300.0)                     # streak 3 → alert
        self.assertEqual(len(self.notifier.sent), 1)
        self.assertIn("watch down", self.notifier.sent[0][1])
        row = self.ctx.db.query_one("SELECT error_streak FROM monitors")
        self.assertEqual(row["error_streak"], 3)


# ────────────────────────────── KG curation ──────────────────────────────────

class KgCurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="w71-kg-")
        self.ctx = _ctx(self.tmp.name)
        from nomorals.agents.kg import KnowledgeGraph

        self.g = KnowledgeGraph(self.ctx.db)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_extracts_known_entities(self) -> None:
        md5 = hashlib.md5(b"password").hexdigest()
        text = (f"digest {md5} seen, url https://example.com/a, "
                "mail admin@example.com, ip 10.1.2.3")
        out = self.g.curate_from_text(text, source="src-test")
        self.assertGreaterEqual(out["added_nodes"], 4)
        labels = {n["label"] for n in self.ctx.db.query("SELECT label FROM kg_nodes")}
        for want in (md5, "https://example.com/a", "admin@example.com",
                     "10.1.2.3", "secret:password"):
            self.assertIn(want, labels)

    def test_decodes_to_edge(self) -> None:
        sha256 = hashlib.sha256(b"codebeast").hexdigest()
        self.g.curate_from_text(sha256, source="src-2")
        edge = self.ctx.db.query_one(
            "SELECT * FROM kg_edges WHERE relation='decodes_to'")
        self.assertIsNotNone(edge)

    def test_empty_text_noop(self) -> None:
        out = self.g.curate_from_text("   ", source="x")
        self.assertEqual(out["added_nodes"], 0)

    def test_kg_tool_curate_action(self) -> None:
        out = reg_call(self.ctx, "kg", action="curate",
                       query="see https://example.com/b and x@y.io",
                       label="src-tool")
        self.assertTrue(out.ok)
        self.assertGreaterEqual(out.unwrap()["curated"]["added_nodes"], 2)

    def test_idempotent_upserts(self) -> None:
        text = "mail a@b.io"
        self.g.curate_from_text(text, source="idem")
        n1 = self.g.stats()["nodes"]
        self.g.curate_from_text(text, source="idem")
        n2 = self.g.stats()["nodes"]
        self.assertEqual(n1, n2)


# ────────────────────────────── migrations ───────────────────────────────────

class MigrationTests(unittest.TestCase):
    def test_v23_applied(self) -> None:
        # wave 71 introduced v23; later waves (72: media_queue) keep
        # stacking on top, so pin "at least 23" and check the v23 table.
        from nomorals.storage.migrations import latest_version

        self.assertGreaterEqual(latest_version(), 23)
        with tempfile.TemporaryDirectory(prefix="w71-mig-") as tmp:
            ctx = _ctx(tmp)
            try:
                row = ctx.db.query_one(
                    "SELECT name FROM sqlite_master WHERE name='monitors'")
                self.assertIsNotNone(row)
            finally:
                ctx.close()


# ────────────────────────────── cipher core ──────────────────────────────────

class CipherCoreTests(unittest.TestCase):
    def test_fips197_block_vectors(self) -> None:
        from nomorals.core.cipher import (_decrypt_block, _encrypt_block,
                                          _expand_key)

        key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
        pt = bytes.fromhex("00112233445566778899aabbccddeeff")
        rk = _expand_key(key)
        self.assertEqual(_encrypt_block(pt, rk),
                         bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a"))
        self.assertEqual(_decrypt_block(_encrypt_block(pt, rk), rk), pt)

    def test_authenticated_round_trips(self) -> None:
        from nomorals.core.cipher import aes_decrypt, aes_encrypt

        for mode in ("ctr", "cbc"):
            for sz in (1, 2, 15, 16, 17, 100, 4096):
                blob = aes_encrypt(b"x" * sz, passphrase="pw" + "a" * 10,
                                   mode=mode)
                self.assertEqual(aes_decrypt(blob, passphrase="pw" + "a" * 10),
                                 b"x" * sz, (mode, sz))

    def test_wrong_passphrase_rejected(self) -> None:
        from nomorals.core.cipher import CipherError, aes_decrypt, aes_encrypt

        for mode in ("ctr", "cbc"):
            blob = aes_encrypt(b"the secret", passphrase="right", mode=mode)
            with self.assertRaises(CipherError):
                aes_decrypt(blob, passphrase="wrong")

    def test_tamper_detected(self) -> None:
        from nomorals.core.cipher import CipherError, aes_decrypt, aes_encrypt

        blob = aes_encrypt(b"tamper me", passphrase="p")
        parts = blob.split(":")
        ct = bytearray(bytes.fromhex(parts[6]))
        ct[0] ^= 0xff
        parts[6] = bytes(ct).hex()
        with self.assertRaises(CipherError):
            aes_decrypt(":".join(parts), passphrase="p")

    def test_raw_key_and_legacy(self) -> None:
        from nomorals.core.cipher import aes_decrypt, aes_encrypt

        raw = bytes(range(32))
        blob = aes_encrypt(b"raw key payload", key=raw)
        self.assertEqual(aes_decrypt(blob, key=raw), b"raw key payload")
        # strip the tag → legacy unauthenticated blob still decrypts
        legacy = ":".join(blob.split(":")[:7])
        self.assertEqual(aes_decrypt(legacy, key=raw), b"raw key payload")

    def test_unicode(self) -> None:
        from nomorals.core.cipher import aes_decrypt, aes_encrypt

        u = "héllo wörld ✓ 42"
        self.assertEqual(aes_decrypt(aes_encrypt(u, passphrase="p"),
                                     passphrase="p"), u.encode("utf-8"))


class CipherToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="w71-cip-")
        self.ctx = _ctx(self.tmp.name)
        from nomorals.tools.registry import ToolRegistry

        self.reg = ToolRegistry(self.ctx).register_builtins()
        self.caps = CapabilitySet.all()

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_encrypt_decrypt_tool(self) -> None:
        enc = self.reg.call("cipher", action="encrypt",
                            data="postgres://u:p@db:5432/prod",
                            passphrase="hunter2", capabilities=self.caps)
        self.assertTrue(enc.ok, enc.error)
        blob = enc.unwrap()["blob"]
        self.assertTrue(blob.startswith("nmc1:v1:"))
        dec = self.reg.call("cipher", action="decrypt", blob=blob,
                            passphrase="hunter2", capabilities=self.caps)
        self.assertEqual(dec.unwrap()["data"], "postgres://u:p@db:5432/prod")

    def test_wrong_passphrase_tool(self) -> None:
        enc = self.reg.call("cipher", action="encrypt", data="secret",
                            passphrase="right", capabilities=self.caps)
        dec = self.reg.call("cipher", action="decrypt",
                            blob=enc.unwrap()["blob"], passphrase="wrong",
                            capabilities=self.caps)
        self.assertFalse(dec.ok)
        self.assertIn("integrity", str(dec.error))

    def test_classic_ciphers(self) -> None:
        r = self.reg.call("cipher", action="classic", data="abc",
                          algorithm="caesar", shift=1, capabilities=self.caps)
        self.assertEqual(r.unwrap()["data"], "bcd")
        r = self.reg.call("cipher", action="classic", data="BCD",
                          algorithm="caesar", shift=1, decrypt=True,
                          capabilities=self.caps)
        self.assertEqual(r.unwrap()["data"], "ABC")
        r = self.reg.call("cipher", action="classic", data="abc",
                          algorithm="vigenere", keyword="b",
                          capabilities=self.caps)
        self.assertEqual(r.unwrap()["data"], "bcd")
        r = self.reg.call("cipher", action="classic", data="CodeBeast",
                          algorithm="atbash", capabilities=self.caps)
        self.assertEqual(r.unwrap()["data"], "XlwvYvzhg")
        r = self.reg.call("cipher", action="classic", data="aGVsbG8=",
                          algorithm="b64", decrypt=True,
                          capabilities=self.caps)
        self.assertEqual(r.unwrap()["data"], "hello")

    def test_hmac_and_formats(self) -> None:
        r = self.reg.call("cipher", action="hmac", key="s3cr3t",
                          data="payload", capabilities=self.caps)
        self.assertEqual(len(r.unwrap()["digest"]), 64)
        r = self.reg.call("cipher", action="formats", capabilities=self.caps)
        self.assertIn("aes-ctr", r.unwrap()["algorithms"])

    def test_registered(self) -> None:
        self.assertIn("cipher", self.reg.names())


# ────────────────────────────── CLI ──────────────────────────────────────────

class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="w71-cli-")
        self.env = dict(os.environ,
                        NM_HOME=self.tmp.name,
                        PYTHONPATH="/home/user/No-morals-ai")
        ws = os.path.join(self.tmp.name, "workspace")
        os.makedirs(ws, exist_ok=True)
        with open(os.path.join(ws, "probe.txt"), "w", encoding="utf-8") as fh:
            fh.write("v1\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _nm(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "nomorals", *args],
            capture_output=True, text=True, env=self.env, timeout=90)

    def test_decode_text(self) -> None:
        r = self._nm("decode", base64.b64encode(b"password").decode())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("base64", r.stdout)
        self.assertIn("password", r.stdout)

    def test_decode_hash(self) -> None:
        digest = hashlib.sha256(b"codebeast").hexdigest()
        r = self._nm("decode", "--hash", digest)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("codebeast", r.stdout)

    def test_decode_decoders(self) -> None:
        r = self._nm("decode", "--mode", "decoders")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("base64", r.stdout)

    def test_decode_json(self) -> None:
        import json

        r = self._nm("decode", "--json",
                     base64.b64encode(b"abc").decode())
        self.assertEqual(r.returncode, 0, r.stderr)
        payload = json.loads(r.stdout)
        self.assertIn("explanation", payload)
        self.assertIn("report", payload)

    def test_cipher_cli_round_trip(self) -> None:
        r = self._nm("cipher", "encrypt", "a secret note",
                     "--passphrase", "hunter2")
        self.assertEqual(r.returncode, 0, r.stderr)
        blob = r.stdout.strip()
        self.assertTrue(blob.startswith("nmc1:v1:"), blob)
        r2 = self._nm("cipher", "decrypt", "--blob", blob,
                      "--passphrase", "hunter2")
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertEqual(r2.stdout.strip(), "a secret note")

    def test_cipher_cli_wrong_pw(self) -> None:
        r = self._nm("cipher", "encrypt", "x" * 40, "--passphrase", "right")
        blob = r.stdout.strip()
        r2 = self._nm("cipher", "decrypt", "--blob", blob,
                      "--passphrase", "wrong")
        self.assertNotEqual(r2.returncode, 0)

    def test_cipher_cli_classic(self) -> None:
        r = self._nm("cipher", "classic", "abc", "--algorithm", "caesar",
                     "--shift", "1")
        self.assertEqual(r.stdout.strip(), "bcd")

    def test_monitor_add_tick_change(self) -> None:
        r = self._nm("monitor", "add", "probe.txt", "--interval", "30")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("watching probe.txt", r.stdout)
        self._nm("monitor", "tick")
        with open(os.path.join(self.tmp.name, "workspace", "probe.txt"),
                  "w", encoding="utf-8") as fh:
            fh.write("v2\n")
        # force-due by waiting past the 30s interval is too slow for a test;
        # instead verify via a second home where the interval floor (30s)
        # already elapsed — so we check list + status + remove here and
        # rely on the agent-level tick tests for change detection.
        r = self._nm("monitor", "list")
        self.assertIn("probe.txt", r.stdout)
        r = self._nm("monitor", "status")
        self.assertIn("1/1", r.stdout)
        r = self._nm("monitor", "remove", "probe.txt")
        self.assertIn("removed", r.stdout)
        r = self._nm("monitor", "list")
        self.assertIn("no monitors", r.stdout)


if __name__ == "__main__":
    unittest.main()
