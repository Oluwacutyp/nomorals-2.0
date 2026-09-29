"""Wave 74 systems — hermetic verification.

1. Known-hash chain: digests cracked once are known forever
   (learn_hash → learned_hash_lookup → known_hash_lookup_chained,
   hash_crack short-circuit, decoder auto-attack learning, KG enrichment).
2. Monitor webhooks + per-monitor alert throttling
   (real local HTTP receiver; change/alert/webhook/throttle lifecycle).
3. Cipher vault: named secrets, AES-256 sealed, passphrase-gated.
4. OSINT agent consuming decoder findings
   (cookies/JWTs → person + domain entities, tool + CLI + chat).
"""
from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from nomorals.agents.context import build_context
from nomorals.core.config import Settings


def _ctx(tmp: str):
    return build_context(Settings(home=tmp), with_executor=False,
                         with_tools=True, with_router=False,
                         with_memory=False)


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _jwt(payload: dict) -> str:
    header = _b64u(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    return ".".join([header, _b64u(json.dumps(payload).encode()), "sig"])


# ── 1. known-hash chain ─────────────────────────────────────────────────────

class KnownHashChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w74-chain-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def test_migration25_tables_and_columns(self) -> None:
        from nomorals.core.decoder import _KNOWN_HASHES_DDL

        self.ctx.db.execute(_KNOWN_HASHES_DDL)  # idempotent
        mon = {r["name"] for r in self.ctx.db.query(
            "PRAGMA table_info(monitors)")}
        for col in ("webhook_url", "min_alert_gap_s", "last_alert_ts"):
            self.assertIn(col, mon)
        tables = {r["name"] for r in self.ctx.db.query(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("cipher_vault", tables)
        self.assertIn("known_hashes", tables)

    def test_learn_and_lookup_chained(self) -> None:
        from nomorals.core.decoder import (KNOWN_HASHES, learn_hash,
                                           learned_hash_lookup,
                                           known_hash_lookup,
                                           known_hash_lookup_chained)

        plain = "s3cr3t-w74"
        d = hashlib.md5(plain.encode()).hexdigest()
        self.assertFalse(learned_hash_lookup(self.ctx.db, d))
        self.assertTrue(learn_hash(self.ctx.db, d, plain, "md5", "test"))
        # idempotent
        self.assertTrue(learn_hash(self.ctx.db, d, plain, "md5", "test"))
        hit = learned_hash_lookup(self.ctx.db, d)
        self.assertEqual(hit, {"plaintext": plain, "algorithm": "md5"})
        # built-in table still first: a built-in digest resolves without the db
        builtin_digest = next(iter(KNOWN_HASHES))
        builtin_plain = KNOWN_HASHES[builtin_digest][0]
        self.assertEqual(known_hash_lookup(builtin_digest)["plaintext"],
                         builtin_plain)
        chained = known_hash_lookup_chained(self.ctx.db, builtin_digest)
        self.assertEqual(chained["plaintext"], builtin_plain)
        self.assertEqual(known_hash_lookup_chained(self.ctx.db, d),
                         {"plaintext": plain, "algorithm": "md5"})
        # fail-soft: bad digests never stored
        self.assertFalse(learn_hash(self.ctx.db, "", plain))
        self.assertFalse(learn_hash(self.ctx.db, d, ""))
        self.assertFalse(learn_hash(None, d, plain))

    def test_hashcrack_short_circuits_on_chain(self) -> None:
        from nomorals.tools.hashcrack import crack_hash

        plain = "s3cr3t-w74"
        d = hashlib.md5(plain.encode()).hexdigest()
        from nomorals.core.decoder import learn_hash

        learn_hash(self.ctx.db, d, plain, "md5", "test")
        res = crack_hash(d, db=self.ctx.db)
        self.assertTrue(res.found)
        self.assertEqual(res.backend, "known-hash")
        self.assertEqual(res.tested, 0)
        self.assertEqual(res.found[d], plain)
        # without db the engine runs (no chain) — still cracks via digits?
        # (md5 of a fixed secret won't; we only assert no crash)
        res2 = crack_hash(d, max_len=4, max_candidates=100)
        self.assertFalse(res2.found)

    def test_tool_live_crack_leaves_learning(self) -> None:
        d = hashlib.md5(b"9876").hexdigest()
        r = self.ctx.tools.call(
            "hash_crack", target=d, charset="0123456789", max_len=6)
        self.assertTrue(r.ok, r.error)
        v = r.unwrap()
        self.assertTrue(v["found"])
        self.assertEqual(v["found"][d], "9876")
        self.assertTrue(v.get("learned"))
        # second sighting: instant known-hash
        r2 = self.ctx.tools.call("hash_crack", target=d)
        v2 = r2.unwrap()
        self.assertEqual(v2["backend"], "known-hash")
        self.assertEqual(v2["tested"], 0)
        self.assertFalse(v2.get("learned"))

    def test_decoder_crack_mode_learns(self) -> None:
        d = hashlib.md5(b"13579").hexdigest()
        r = self.ctx.tools.call("decoder", data=d, mode="crack",
                                charset="0123456789", max_len=6)
        self.assertTrue(r.ok, r.error)
        v = r.unwrap()
        self.assertTrue(v["cracked"], v)
        self.assertTrue(v.get("learned"))
        self.assertIn("live-crack", v["via"])
        r2 = self.ctx.tools.call("decoder", data=d, mode="crack")
        v2 = r2.unwrap()
        self.assertTrue(v2["cracked"])
        self.assertIn("known-secrets", v2["via"])
        self.assertEqual(v2.get("known_source"), "learned")

    def test_decoder_crack_builtin_known_source(self) -> None:
        from nomorals.core.decoder import KNOWN_HASHES

        digest, (plain, algo) = next(iter(KNOWN_HASHES.items()))
        r = self.ctx.tools.call("decoder", data=digest, mode="crack")
        self.assertTrue(r.ok, r.error)
        v = r.unwrap()
        self.assertTrue(v["cracked"])
        self.assertEqual(v["plaintext"], plain)
        self.assertIn("known-secrets", v["via"])
        self.assertEqual(v.get("known_source"), "built-in")

    def test_kg_enrichment_records_known_source(self) -> None:
        from nomorals.agents.kg import KnowledgeGraph
        from nomorals.core.decoder import learn_hash

        plain = "kg-secret-74"
        d = hashlib.md5(plain.encode()).hexdigest()
        learn_hash(self.ctx.db, d, plain, "md5", "test")
        kg = KnowledgeGraph(self.ctx.db)
        kg.curate_from_text(f"digest found: {d} cracked",
                            source="w74-test")
        rows = self.ctx.db.query(
            "SELECT label, properties FROM kg_nodes")
        props = {r["label"]: json.loads(r["properties"] or "{}")
                 for r in rows}
        self.assertIn(f"secret:{plain}", props)
        dig_node = next((v for k, v in props.items() if d in k), None)
        self.assertIsNotNone(dig_node, props.keys())
        self.assertEqual(dig_node.get("known_source"), "learned")


# ── 2. monitor webhooks + throttling ────────────────────────────────────────

class _Hook(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        _Hook.received.append(json.loads(self.rfile.read(n)))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a) -> None:  # silence
        pass


class MonitorWebhookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w74-mon-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        _Hook.received = []
        self.srv = HTTPServer(("127.0.0.1", 0), _Hook)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/hook"
        self.thread = threading.Thread(target=self.srv.serve_forever,
                                       daemon=True)
        self.thread.start()
        ws = Path(self.ctx.settings.workspace_dir)
        ws.mkdir(parents=True, exist_ok=True)
        self.f = ws / "watched.txt"
        self.f.write_text("v1")

    def tearDown(self) -> None:
        self.srv.shutdown()
        self.ctx.__exit__(None, None, None)

    def test_webhook_and_throttle_lifecycle(self) -> None:
        from nomorals.agents.monitor import MonitorAgent

        agent = MonitorAgent(self.ctx)
        info = agent.add(str(self.f), interval=30, webhook=self.url,
                         min_gap=60)
        self.assertEqual(info["webhook_url"], self.url)
        self.assertEqual(info["min_alert_gap_s"], 60)

        t0 = 1_000_000.0
        agent.tick(now=t0)                       # baseline
        self.f.write_text("v2 — changed")
        c = agent.tick(now=t0 + 31)["changed"][0]
        self.assertFalse(c["throttled"])
        self.assertTrue(c["webhook"]["ok"])
        p = _Hook.received[0]
        self.assertEqual(p["event"], "change")
        self.assertEqual(p["target"], str(self.f))
        self.assertEqual(p["watch_kind"], "file")
        self.assertEqual(p["source"], "nomorals-monitor")
        self.assertIn("v2", p["diff"])

        # within the gap: recorded, but no webhook / no notifier
        self.f.write_text("v3 — again")
        c3 = agent.tick(now=t0 + 62)["changed"][0]
        self.assertTrue(c3["throttled"])
        self.assertNotIn("webhook", c3)
        self.assertEqual(len(_Hook.received), 1)

        # after the gap: fires again
        self.f.write_text("v4")
        c4 = agent.tick(now=t0 + 93)["changed"][0]
        self.assertFalse(c4["throttled"])
        self.assertTrue(c4["webhook"]["ok"])
        self.assertEqual(len(_Hook.received), 2)

        # gap=0 disables throttling
        row = agent.set_alerting(info["id"], min_gap=0)
        self.assertEqual(row["min_alert_gap_s"], 0)
        self.assertEqual(row["webhook_url"], self.url)
        self.f.write_text("v5")
        c5 = agent.tick(now=t0 + 124)["changed"][0]
        self.assertFalse(c5["throttled"])
        self.assertTrue(c5["webhook"]["ok"])     # gap=0 → not throttled
        self.assertEqual(len(_Hook.received), 3)

        # webhook can be turned off
        row = agent.set_alerting(info["id"], webhook="")
        self.assertEqual(row["webhook_url"], "")
        self.f.write_text("v6")
        c6 = agent.tick(now=t0 + 155)["changed"][0]
        self.assertFalse(c6["throttled"])
        self.assertFalse(c6["webhook"]["fired"])  # webhook turned off
        self.assertEqual(len(_Hook.received), 3)

    def test_bad_webhook_never_breaks_tick(self) -> None:
        from nomorals.agents.monitor import MonitorAgent

        agent = MonitorAgent(self.ctx)
        # port 1 is not listening
        info = agent.add(str(self.f), interval=30,
                         webhook="http://127.0.0.1:1/dead", min_gap=0)
        t0 = 2_000_000.0
        agent.tick(now=t0)
        self.f.write_text("boom")
        res = agent.tick(now=t0 + 31)
        c = res["changed"][0]
        self.assertFalse(c["throttled"])
        self.assertFalse(c["webhook"]["ok"])
        self.assertTrue(c["webhook"].get("error"))

    def test_add_rejects_bad_webhook(self) -> None:
        from nomorals.agents.monitor import MonitorAgent

        agent = MonitorAgent(self.ctx)
        with self.assertRaises(ValueError):
            agent.add(str(self.f), webhook="ftp://nope")

    def test_tool_and_cli(self) -> None:
        r = self.ctx.tools.call("monitor", action="add", target=str(self.f),
                                webhook=self.url, min_gap=120)
        self.assertTrue(r.ok, r.error)
        d = r.unwrap()
        self.assertEqual(d["webhook_url"], self.url)
        self.assertEqual(d["min_alert_gap_s"], 120)
        r = self.ctx.tools.call("monitor", action="alert", ref=d["id"],
                                min_gap=300)
        self.assertTrue(r.ok, r.error)
        d2 = r.unwrap()
        self.assertEqual(d2["monitor"]["min_alert_gap_s"], 300)
        self.assertEqual(d2["monitor"]["webhook_url"], self.url)

        from nomorals import cli

        parser = cli._parser()
        args = parser.parse_args(
            ["monitor", "add", str(self.f), "--webhook", self.url,
             "--min-gap", "45"])
        self.assertEqual(args.command, "monitor")
        self.assertEqual(args.min_gap, 45)
        rc = cli._cmd_monitor(args, self.ctx)
        self.assertEqual(rc, 0)
        rows = self.ctx.db.query("SELECT * FROM monitors")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["webhook_url"], self.url)
        self.assertEqual(rows[0]["min_alert_gap_s"], 45)

        args2 = parser.parse_args(
            ["monitor", "alert", rows[0]["id"], "--min-gap", "0"])
        self.assertEqual(cli._cmd_monitor(args2, self.ctx), 0)
        row = self.ctx.db.query_one(
            "SELECT * FROM monitors WHERE id=?", (rows[0]["id"],))
        self.assertEqual(row["min_alert_gap_s"], 0)


# ── 3. cipher vault ─────────────────────────────────────────────────────────

class CipherVaultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w74-vault-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def _agent(self):
        from nomorals.agents.cipher import CipherAgent

        return CipherAgent(context=self.ctx, name="test-vault")

    def test_put_get_list_rm_roundtrip(self) -> None:
        a = self._agent()
        r = a.run({"action": "vault_put", "name": "db",
                   "data": "hunter2-s3cret", "passphrase": "key1"})
        self.assertTrue(r.ok, r.error)
        self.assertTrue(r.output["stored"])

        r = a.run({"action": "vault_get", "name": "db",
                   "passphrase": "key1"})
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.output["data"], "hunter2-s3cret")

        r = a.run({"action": "vault_put", "name": "api",
                   "data": "sk-abc123", "passphrase": "key1"})
        self.assertTrue(r.ok)
        r = a.run({"action": "vault_list"})
        self.assertTrue(r.ok)
        self.assertEqual(r.output["names"], ["api", "db"])
        self.assertEqual(r.output["count"], 2)

        r = a.run({"action": "vault_rm", "name": "api",
                   "passphrase": "key1"})
        self.assertTrue(r.ok)
        self.assertTrue(r.output["removed"])
        r = a.run({"action": "vault_list"})
        self.assertEqual(r.output["names"], ["db"])

    def test_wrong_passphrase_rejected(self) -> None:
        a = self._agent()
        a.run({"action": "vault_put", "name": "db", "data": "top",
               "passphrase": "key1"})
        r = a.run({"action": "vault_get", "name": "db", "passphrase": "nope"})
        self.assertFalse(r.ok)
        self.assertIn("wrong passphrase", r.error)

    def test_requires_passphrase_and_name(self) -> None:
        a = self._agent()
        r = a.run({"action": "vault_put", "name": "x", "data": "y"})
        self.assertFalse(r.ok)
        r = a.run({"action": "vault_put", "data": "y", "passphrase": "k"})
        self.assertFalse(r.ok)
        r = a.run({"action": "vault_get", "passphrase": "k"})
        self.assertFalse(r.ok)

    def test_missing_entry(self) -> None:
        a = self._agent()
        r = a.run({"action": "vault_get", "name": "ghost",
                   "passphrase": "k"})
        self.assertFalse(r.ok)
        self.assertIn("no vault entry", r.error)

    def test_blob_is_sealed_not_plaintext(self) -> None:
        a = self._agent()
        a.run({"action": "vault_put", "name": "db",
               "data": "PLAINTEXT-MARKER-12345", "passphrase": "key1"})
        row = self.ctx.db.query_one(
            "SELECT blob FROM cipher_vault WHERE name='db'")
        self.assertNotIn("PLAINTEXT-MARKER-12345", row["blob"])
        self.assertTrue(row["blob"])

    def test_persists_across_agent_instances(self) -> None:
        a = self._agent()
        a.run({"action": "vault_put", "name": "persist",
               "data": "again", "passphrase": "k"})
        b = self._agent()
        r = b.run({"action": "vault_get", "name": "persist", "passphrase": "k"})
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.output["data"], "again")

    def test_tool_vault_actions(self) -> None:
        r = self.ctx.tools.call("cipher", action="vault_put", name="t",
                                data="secret-t", passphrase="p")
        self.assertTrue(r.ok, r.error)
        r = self.ctx.tools.call("cipher", action="vault_get", name="t",
                                passphrase="p")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.unwrap()["data"], "secret-t")
        r = self.ctx.tools.call("cipher", action="vault_list")
        self.assertIn("t", r.unwrap()["names"])

    def test_cli_parser_accepts_vault(self) -> None:
        from nomorals import cli

        parser = cli._parser()
        args = parser.parse_args(
            ["cipher", "vault_put", "db", "hunter2", "--passphrase", "k"])
        self.assertEqual(args.action, "vault_put")
        self.assertEqual(args.data, "db")
        self.assertEqual(args.secret, "hunter2")
        rc = cli._cmd_cipher(args, self.ctx)
        self.assertEqual(rc, 0)
        args = parser.parse_args(
            ["cipher", "vault_get", "db", "--passphrase", "k"])
        self.assertEqual(cli._cmd_cipher(args, self.ctx), 0)


# ── 4. osint consumes decoder findings ──────────────────────────────────────

class OsintDecoderBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w74-osint-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        self.claims = {"sub": "user-123",
                       "email": "Ada.Lovelace@Example.COM",
                       "name": "Ada Lovelace",
                       "preferred_username": "ada",
                       "iss": "https://api.example.com/v1"}
        self.jwt = _jwt(self.claims)
        self.cookie_doc = (f"api_token={self.jwt}; Domain=app.example.com; "
                           "Path=/\n"
                           "session_id=xyz789; Domain=app.example.com")

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def test_ingest_decoder_findings_dict_report(self) -> None:
        from nomorals.agents.osint_graph import IdentityGraph

        report = {
            "best": {"chain": ["jwt"], "confidence": 0.95,
                     "output": {"header": {}, "payload": self.claims,
                                "signed": True, "warnings": []},
                     "note": "decoded"},
            "hits": [], "forensics": {}, "hash": None,
            "tokens": [{"token": "www.example.com"}],
            "cookies": [
                {"name": "api_token", "value": self.jwt,
                 "jwt": {"payload": self.claims}},
                {"flag": "Domain=app.example.com"},
            ],
            "jwt": {"payload": self.claims},
        }
        g = IdentityGraph(self.ctx)
        out = g.ingest_decoder_findings(report, source="unit")
        self.assertTrue(any("ada.lovelace@example.com" in p
                            for p in out["persons"]), out)
        self.assertTrue(any("username:ada" in p for p in out["persons"]),
                        out)
        self.assertTrue(any("api.example.com" in d for d in out["domains"]),
                        out)
        self.assertTrue(any("app.example.com" in d for d in out["domains"]),
                        out)
        # person↔domain edges created
        st = g.stats()
        self.assertGreater(st["edges"], 0)
        n = g.node("email:ada.lovelace@example.com")
        linked = {l["kind"] for l in n["links"]}
        self.assertIn("domain", linked)
        # idempotent re-ingest: no duplicate nodes
        before2 = g.stats()["nodes"]
        g.ingest_decoder_findings(report, source="unit")
        self.assertEqual(g.stats()["nodes"], before2)

    def test_ingest_decoder_findings_real_report(self) -> None:
        from nomorals.agents.osint_graph import IdentityGraph
        from nomorals.core.decoder import analyze

        rep = analyze(f"{self.jwt}\n{self.cookie_doc}")
        g = IdentityGraph(self.ctx)
        out = g.ingest_decoder_findings(rep, source="decoder:test")
        self.assertTrue(any("ada.lovelace@example.com" in p
                            for p in out["persons"]), out)
        self.assertTrue(any("app.example.com" in d for d in out["domains"]),
                        out)
        self.assertTrue(any("api.example.com" in d for d in out["domains"]),
                        out)

    def test_decoder_tool_auto_hook(self) -> None:
        r = self.ctx.tools.call("decoder",
                                data=f"{self.jwt}\n{self.cookie_doc}",
                                mode="analyze")
        self.assertTrue(r.ok, r.error)
        v = r.unwrap()
        self.assertIn("osint", v)
        self.assertTrue(v["osint"]["persons_found"] >= 3, v["osint"])
        self.assertTrue(any("app.example.com" in d
                            for d in v["osint"]["domains"]), v["osint"])
        g = self._graph()
        self.assertGreaterEqual(g.stats()["nodes"], 4)

    def test_tool_ingest_decoder_action(self) -> None:
        from nomorals.core.decoder import analyze

        rep = analyze(f"{self.jwt}\n{self.cookie_doc}")
        r = self.ctx.tools.call("osint_graph", action="ingest_decoder",
                                report=json.dumps(rep.to_dict()))
        self.assertTrue(r.ok, r.error)
        v = r.unwrap()
        self.assertGreaterEqual(v["ingest_decoder"]["persons_found"], 3)
        self.assertGreaterEqual(v["stats"]["nodes"], 4)
        r = self.ctx.tools.call("osint_graph", action="ingest_decoder")
        self.assertFalse(r.ok)

    def test_cli_decoder_ingest(self) -> None:
        from nomorals.core.decoder import analyze
        from nomorals import cli

        rep = analyze(f"{self.jwt}\n{self.cookie_doc}")
        path = Path(self.tmp) / "report.json"
        path.write_text(json.dumps(rep.to_dict()))
        parser = cli._parser()
        args = parser.parse_args(["osint", "decoder", str(path),
                                  "--source", "cli-test"])
        self.assertEqual(cli._cmd_osint(args, self.ctx), 0)
        args = parser.parse_args(["osint", "stats"])
        self.assertEqual(cli._cmd_osint(args, self.ctx), 0)
        args = parser.parse_args(["osint", "clusters"])
        self.assertEqual(cli._cmd_osint(args, self.ctx), 0)
        args = parser.parse_args(["osint", "node",
                                  "email:ada.lovelace@example.com"])
        self.assertEqual(cli._cmd_osint(args, self.ctx), 0)

    def test_chat_handlers(self) -> None:
        from nomorals.agents.partner_runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = self.ctx

        # /cipher vault …
        out = rt._control_cipher("vault put mydb hunter2 with vaultkey")
        self.assertIn("stored mydb", out)
        out = rt._control_cipher("vault get mydb with vaultkey")
        self.assertIn("hunter2", out)
        out = rt._control_cipher("vault list")
        self.assertIn("mydb", out)
        out = rt._control_cipher("vault rm mydb with vaultkey")
        self.assertIn("removed mydb", out)
        out = rt._control_cipher("vault get mydb with vaultkey")
        self.assertIn("no vault entry", out)

        # /monitor add … --webhook … --min-gap … + alert
        ws = Path(self.ctx.settings.workspace_dir)
        ws.mkdir(parents=True, exist_ok=True)
        f = ws / "m.txt"
        f.write_text("1")
        out = rt._control_monitor(
            f"add {f} every 60s --webhook http://127.0.0.1:9/h "
            "--min-gap 120")
        self.assertIn("watching", out)
        self.assertIn("webhook", out)
        out = rt._control_monitor("list")
        self.assertIn("webhook", out)
        out = rt._control_monitor(f"alert {f} --min-gap 0")
        self.assertIn("gap=0", out)

        # /osint graph decoder <inline json>
        from nomorals.core.decoder import analyze

        rep = analyze(f"{self.jwt}\n{self.cookie_doc}")
        payload = json.dumps(rep.to_dict())
        out = rt._control_osint(f"graph decoder {payload}",
                                chat_key="local:test")
        self.assertIn("ingested", out)
        self.assertIn("ada", out.lower())

    def _graph(self):
        from nomorals.agents.osint_graph import IdentityGraph

        return IdentityGraph(self.ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
