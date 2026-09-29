"""Wave 45 — god-tier OSINT correlation + system-wide de-restriction.

- agents/osint_graph.py: entity extraction (emails, NG phones, IPs,
  domains, @handles, names, NG locations), persistent identity
  correlation graph (kv_store) with alias/associated edges, union-find
  clustering with confidence, neighborhood queries, timelines, and the
  CampaignRunner — BFS over the discovered web with injectable sources
- tools/metadata.py: email header forensics (routing chain, SPF/DKIM/
  DMARC verdicts, Message-ID vs From-domain spoofing signal)
- tools/database.py: _MAX_LIMIT 200 -> 5000 (owner's own state DB)
- tools/sandbox_code.py: power-aware timeout ceiling (300s normal /
  1800s power mode)

Fully hermetic: the campaign runs against fake source functions; the
graph persists in the test context's real sqlite kv_store; EML
forensics runs on a synthetic .eml written inside the context home.
No external network.
"""
from __future__ import annotations

import inspect
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from tests.test_partner_runtime import _make_context

from nomorals.core.errors import ToolError
from nomorals.agents.osint_graph import (
    CampaignRunner,
    EntityExtractor,
    IdentityGraph,
    name_variants,
    _norm_phone,
    _parse_seeds,
)
from nomorals.tools import database as db_tool
from nomorals.tools import metadata as md
from nomorals.tools import sandbox_code as sc


# ── fixtures ─────────────────────────────────────────────────────────────────

SPOOFED_EML = b"""From: "Ade Bankole" <info@adebankole.ng>
To: victim@example.com
Subject: Urgent - your account
Date: Mon, 11 Sep 2026 09:30:00 +0100
Message-ID: <20260911093000.abc123@mail.ghostrelay.io>
Reply-To: ade-claims@ghostrelay.io
X-Originating-IP: 45.227.191.12
Authentication-Results: mx.example.com;
 dkim=none header.d=adebankole.ng;
 spf=fail smtp.mailfrom=ghostrelay.io;
 dmarc=fail header.from=adebankole.ng
Received: by mx.example.com (Postfix) with ESMTP id A1B2C3
 for <victim@example.com>; Mon, 11 Sep 2026 09:30:00 +0100
Received: from mail.ghostrelay.io (mail.ghostrelay.io [103.149.108.77])
 by edge.example.com (Postfix) with ESMTPS id XYZ9
 for <victim@example.com>; Mon, 11 Sep 2026 09:29:58 +0100

The money must leave today.
"""

LEAK_BLOB = """\
Leaked intake form, Lagos office:
  Name: Oluwaseun Adebayo
  Email: Oluwaseun.adebayo@gmail.com
  Phone: 0803 123 4567
  Also known as @seunade on Twitter
  Office: Lekki, Lagos
  Server backup host: 197.210.144.55
  Intranet: intranet.adebankole.ng
  Photo: staff-photo.jpg
"""


class _FakeTool:
    """Records calls; returns canned JSON per tool name."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.responses = responses or {}

    def call(self, name: str, **params):
        self.calls.append((name, dict(params)))
        value = json.loads(self.responses.get(name, "{}"))
        return type("O", (), {"ok": True, "value": value})()


# ── entity extraction ────────────────────────────────────────────────────────


class EntityExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = EntityExtractor()

    def test_extracts_all_kinds_from_a_leak(self) -> None:
        found = {(e.kind, e.value) for e in self.x.extract(LEAK_BLOB)}
        self.assertIn(("email", "oluwaseun.adebayo@gmail.com"), found)
        self.assertIn(("phone", "2348031234567"), found)   # NG local 0xx
        self.assertIn(("ip", "197.210.144.55"), found)
        self.assertIn(("domain", "intranet.adebankole.ng"), found)
        self.assertIn(("username", "seunade"), found)
        self.assertIn(("name", "oluwaseun adebayo"), found)
        self.assertIn(("location", "lagos"), found)

    def test_single_label_cc_domains(self) -> None:
        found = {e.value for e in
                 self.x.extract("office at corp.ng, mail at mail2.corp.ng")}
        self.assertIn("corp.ng", found)
        self.assertIn("mail2.corp.ng", found)

    def test_file_extensions_are_not_domains(self) -> None:
        found = {e.value for e in self.x.extract(
            "see report.pdf, staff-photo.jpg and data.csv on wiki.corp.ng")}
        self.assertNotIn("report.pdf", found)
        self.assertNotIn("staff-photo.jpg", found)
        self.assertNotIn("data.csv", found)
        self.assertIn("wiki.corp.ng", found)

    def test_ips_are_not_domains(self) -> None:
        found = {(e.kind, e.value) for e in
                 self.x.extract("host 203.116.44.9 is up")}
        self.assertIn(("ip", "203.116.44.9"), found)
        self.assertNotIn(("domain", "203.116.44.9"), found)

    def test_nigerian_phone_normalization(self) -> None:
        self.assertEqual(_norm_phone("0803 123 4567"), "2348031234567")
        self.assertEqual(_norm_phone("+234 803 123 4567"), "2348031234567")
        self.assertEqual(_norm_phone("8031234567"), "2348031234567")

    def test_name_variants_are_deterministic(self) -> None:
        vs = name_variants("Oluwaseun Adebayo")
        self.assertIn("oluwaseun adebayo", vs)
        self.assertIn("oluwaseunadebayo", vs)
        self.assertIn("adebayo oluwaseun", vs)
        self.assertIn("o a.", vs)
        self.assertEqual(name_variants("Oluwaseun Adebayo"),
                         name_variants("OLUWASEUN ADEBAYO"))

    def test_dedup_and_line_numbers(self) -> None:
        out = self.x.extract("x@y.com\nx@y.com\nz@y.com")
        emails = [e for e in out if e.kind == "email"]
        self.assertEqual([e.value for e in emails], ["x@y.com", "z@y.com"])
        self.assertEqual(emails[1].line, 3)


# ── identity graph ───────────────────────────────────────────────────────────


class IdentityGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _make_context()
        self.graph = IdentityGraph(self.ctx)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_ingest_persists_and_roundtrips(self) -> None:
        self.graph.ingest(LEAK_BLOB, source="leak.pdf", ts=1000.0)
        stats = self.graph.stats()
        self.assertGreaterEqual(stats["nodes"], 5)
        # a brand-new graph object sees the persisted state
        again = IdentityGraph(self.ctx)
        self.assertEqual(again.stats()["nodes"], stats["nodes"])
        node = again.node("oluwaseun.adebayo@gmail.com")
        self.assertEqual(node["kind"], "email")
        self.assertIn("leak.pdf", node["sources"])

    def test_cooccurrence_edges_and_clusters(self) -> None:
        self.graph.ingest(LEAK_BLOB, source="s1", ts=1000.0)
        self.graph.ingest(
            "Oluwaseun.adebayo@gmail.com again at 197.210.144.55",
            source="s2", ts=2000.0)
        clusters = self.graph.clusters()
        self.assertTrue(clusters)
        top = clusters[0]
        values = {e["value"] for e in top["entities"]}
        self.assertIn("oluwaseun.adebayo@gmail.com", values)
        self.assertIn("197.210.144.55", values)
        self.assertGreaterEqual(top["confidence"], 0.5)
        self.assertLessEqual(top["confidence"], 1.0)
        # evidence from two sources
        self.assertGreaterEqual(top["evidence_sources"], 2)
        # edge weight grew with the repeat, capped at 1.0
        edge = next(e for e in self.graph.data["edges"]
                    if e["class"] == "associated" and
                    {e["a"], e["b"]} ==
                    {"email:oluwaseun.adebayo@gmail.com",
                     "ip:197.210.144.55"})
        self.assertGreater(edge["weight"], 0.5)
        self.assertLessEqual(edge["weight"], 1.0)

    def test_merge_collapses_and_rehomes_edges(self) -> None:
        self.graph.ingest(LEAK_BLOB, source="s1", ts=1000.0)
        self.graph.ingest("ade_seun@proton.me owns seunade", source="s2",
                          ts=1100.0)
        before = self.graph.stats()["nodes"]
        res = self.graph.merge("oluwaseun.adebayo@gmail.com",
                               "oluwaseun adebayo", source="manual")
        self.assertTrue(res["merged"])
        self.assertEqual(res["representative"], "oluwaseun.adebayo@gmail.com")
        self.assertEqual(res["alias"], "oluwaseun adebayo")
        # name node gone, email node keeps it as an alias
        nodes = self.graph.data["nodes"]
        self.assertNotIn("name:oluwaseun adebayo", nodes)
        self.assertIn("name:oluwaseun adebayo",
                      nodes["email:oluwaseun.adebayo@gmail.com"]["aliases"])
        self.assertEqual(self.graph.stats()["nodes"], before - 1)
        # associated edges that pointed at the name now point at the email
        for edge in self.graph.data["edges"]:
            if edge["class"] == "associated":
                self.assertNotIn("oluwaseun adebayo", (edge["a"], edge["b"]))
        # alias tombstone edge recorded for provenance
        self.assertTrue(any(
            e["class"] == "alias" and
            e["a"] == "email:oluwaseun.adebayo@gmail.com" and
            e["b"] == "name:oluwaseun adebayo"
            for e in self.graph.data["edges"]))
        # clusters still work and still connect the neighborhood
        top = self.graph.clusters()[0]
        values = {e["value"] for e in top["entities"]}
        self.assertIn("seunade", values)
        self.assertIn("197.210.144.55", values)

    def test_merge_representative_prefers_stronger_kinds(self) -> None:
        self.graph.ingest("Oluwaseun Adebayo, ade_seun@proton.me",
                          source="s1", ts=1000.0)
        res = self.graph.merge("oluwaseun adebayo", "ade_seun@proton.me")
        self.assertEqual(res["representative"], "ade_seun@proton.me")
        self.assertEqual(res["alias"], "oluwaseun adebayo")

    def test_merge_missing_entity_raises(self) -> None:
        self.graph.ingest("a@b.com", source="s")
        with self.assertRaises(ToolError):
            self.graph.merge("a@b.com", "ghost@nowhere.com")

    def test_node_neighborhood_and_lookup_forms(self) -> None:
        self.graph.ingest(LEAK_BLOB, source="s1", ts=1000.0)
        n = self.graph.node("email:oluwaseun.adebayo@gmail.com")
        linked = {l["value"] for l in n["links"]}
        self.assertIn("197.210.144.55", linked)
        self.assertIn("seunade", linked)
        # bare lookup works too
        n2 = self.graph.node("197.210.144.55")
        self.assertEqual(n2["kind"], "ip")
        with self.assertRaises(ToolError):
            self.graph.node("no-such-entity")

    def test_timeline_and_stats_and_clear(self) -> None:
        self.graph.ingest("a@b.com", source="s1", ts=100.0)
        self.graph.ingest("a@b.com and 10.0.0.1", source="s2", ts=200.0)
        tl = self.graph.timeline()
        self.assertEqual([e["kind"] for e in tl], ["ingest:s1", "ingest:s2"])
        stats = self.graph.stats()
        self.assertEqual(stats["by_kind"].get("email"), 1)
        self.assertEqual(stats["by_kind"].get("ip"), 1)
        self.assertGreaterEqual(stats["edges"], 1)
        self.graph.clear()
        self.assertEqual(self.graph.stats()["nodes"], 0)
        self.assertEqual(IdentityGraph(self.ctx).stats()["nodes"], 0)

    def test_node_cap_does_not_crash(self) -> None:
        for i in range(40):
            self.graph.ingest(f"user{i}@example{i}.com",
                              source=f"s{i}", ts=float(i))
        self.assertLessEqual(self.graph.stats()["nodes"], 5000)


# ── campaign runner ──────────────────────────────────────────────────────────


class CampaignRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _make_context()
        self.graph = IdentityGraph(self.ctx)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def _sources(self) -> dict:
        """Fake OSINT tools that discover a two-hop web."""
        data = {
            "phone": "Contact: Oluwaseun Adebayo <seun.ade@corp.ng> "
                     "at 203.116.44.9; handle @seunade, office corp.ng",
            "email": "email seun.ade@corp.ng belongs to oluwade "
                     "host mx.corp.ng",
            "domain": "corp.ng resolves 203.116.44.9 registrar NG-REG",
        }

        def make(kind):
            def fn(value: str) -> str:
                return data.get(kind, f"nothing about {value}")
            return fn

        return {k: make(k) for k in data}

    def test_bfs_walks_the_whole_discovered_web(self) -> None:
        runner = CampaignRunner(
            self.graph, self._sources(),
            max_steps=40, max_entities=100, time_budget=30.0)
        dossier = runner.run([("phone", "2348031234567")],
                             source_label="t")
        self.assertEqual(dossier["stopped"], "exhausted")
        investigated = {(f["kind"], f["value"]) for f in dossier["findings"]}
        self.assertIn(("phone", "2348031234567"), investigated)
        self.assertIn(("email", "seun.ade@corp.ng"), investigated)
        self.assertIn(("domain", "corp.ng"), investigated)
        # everything found is in the graph
        stats = self.graph.stats()
        self.assertGreaterEqual(stats["nodes"], 4)
        # dossier carries ranked clusters + timeline
        self.assertTrue(dossier["clusters"])
        self.assertTrue(dossier["timeline"])
        self.assertEqual(dossier["graph"]["nodes"], stats["nodes"])

    def test_error_isolation_one_dead_source_keeps_running(self) -> None:
        sources = self._sources()

        def dead_email(value: str) -> str:
            raise RuntimeError("HIBP key missing")

        sources["email"] = dead_email
        runner = CampaignRunner(self.graph, sources, max_steps=40,
                                time_budget=30.0)
        dossier = runner.run([("phone", "2348031234567")])
        failed = [f for f in dossier["findings"] if f.get("error")]
        self.assertEqual(len(failed), 1)
        self.assertIn("HIBP key missing", failed[0]["error"])
        # the walk continued past the failure
        ok = {(f["kind"], f["value"]) for f in dossier["findings"]
              if not f.get("error")}
        self.assertIn(("phone", "2348031234567"), ok)
        self.assertEqual(dossier["stopped"], "exhausted")

    def test_step_budget_stops_the_walk(self) -> None:
        r1 = CampaignRunner(self.graph, self._sources(), max_steps=1,
                            time_budget=30.0)
        d1 = r1.run([("phone", "2348031234567")])
        self.assertEqual(d1["steps"], 1)
        self.assertEqual(d1["stopped"], "budget")
        self.assertGreater(d1["worklist_left"], 0)

    def test_time_budget_stops_an_endless_web(self) -> None:
        # a source that always reveals one more email -> infinite web;
        # the time budget must end it
        counter = {"n": 0}

        def endless(value: str) -> str:
            counter["n"] += 1
            return f"next contact e{counter['n']}@x.com"

        runner = CampaignRunner(IdentityGraph(self.ctx),
                                {"email": endless},
                                max_steps=1000, time_budget=0.8)
        d = runner.run([("email", "e0@x.com")])
        self.assertEqual(d["stopped"], "budget")
        self.assertGreater(d["steps"], 1)
        self.assertLess(d["steps"], 1000)

    def test_no_revisits_and_no_self_loop(self) -> None:
        # a source that 'discovers' itself must not loop forever
        sources = {"phone": lambda v: f"same number {v} again",
                   "email": lambda v: "nothing here"}
        runner = CampaignRunner(self.graph, sources, max_steps=100,
                                time_budget=30.0)
        d = runner.run([("phone", "2348031234567")])
        self.assertEqual(d["entities_investigated"], 1)
        self.assertEqual(d["stopped"], "exhausted")

    def test_dossier_clusters_have_confidence(self) -> None:
        runner = CampaignRunner(self.graph, self._sources(),
                                max_steps=40, time_budget=30.0)
        d = runner.run([("phone", "2348031234567")])
        top = d["clusters"][0]
        self.assertGreater(top["confidence"], 0.5)
        kinds = {e["kind"] for e in top["entities"]}
        self.assertIn("email", kinds)


class SeedParsingTest(unittest.TestCase):
    def test_explicit_kinds(self) -> None:
        seeds = _parse_seeds("phone:0803 123 4567, email: X@Y.com ")
        self.assertIn(("phone", "2348031234567"), seeds)
        self.assertIn(("email", "x@y.com"), seeds)

    def test_auto_detection(self) -> None:
        seeds = _parse_seeds("seun.ade@corp.ng, 203.116.44.9, @seunade")
        self.assertIn(("email", "seun.ade@corp.ng"), seeds)
        self.assertIn(("ip", "203.116.44.9"), seeds)
        self.assertIn(("username", "seunade"), seeds)

    def test_dedup_and_empty(self) -> None:
        self.assertEqual(_parse_seeds(""), [])
        seeds = _parse_seeds("a@b.com, A@B.com")
        self.assertEqual(len(seeds), 1)


# ── email forensics ──────────────────────────────────────────────────────────


class EmlForensicsTest(unittest.TestCase):
    def test_detects_spoofed_message_id_domain(self) -> None:
        meta = md._eml_meta(SPOOFED_EML)
        self.assertEqual(meta["format"], "EMAIL")
        self.assertEqual(meta["message_id_domain"], "mail.ghostrelay.io")
        self.assertEqual(meta["from_domain"], "adebankole.ng")
        self.assertFalse(meta["domain_match"])

    def test_parses_headers_and_verdicts(self) -> None:
        meta = md._eml_meta(SPOOFED_EML)
        self.assertIn("Ade Bankole", meta["from"])
        self.assertEqual(meta["subject"], "Urgent - your account")
        self.assertEqual(meta["x_originating_ip"], "45.227.191.12")
        self.assertEqual(meta["authentication"],
                         {"dkim": "none", "spf": "fail", "dmarc": "fail"})

    def test_received_chain_oldest_first(self) -> None:
        meta = md._eml_meta(SPOOFED_EML)
        self.assertEqual(meta["received_hops"], 2)
        # oldest hop first: the external ghostrelay box, then the MX
        self.assertEqual(meta["first_hop_from"], "mail.ghostrelay.io")
        self.assertEqual(meta["last_hop_via"], "mx.example.com")
        self.assertEqual(meta["hop_chain"][0]["from"],
                         "mail.ghostrelay.io")

    def test_dispatch_email_vs_jpeg_regression(self) -> None:
        ctx, tmpctx = _make_context()
        try:
            home = Path(ctx.settings.home) / "workspace"
            home.mkdir(parents=True, exist_ok=True)
            eml = home / "msg.eml"
            eml.write_bytes(SPOOFED_EML)
            out = md.extract_metadata(ctx, str(eml))
            self.assertEqual(out["format"], "EMAIL")
            self.assertFalse(out["domain_match"])
            # a real JPEG still goes down the EXIF path
            jpg = home / "x.jpg"
            jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"JFIF\x00\x01\x00\x01"
                            + b"\x00" * 24 + b"\xff\xd9")
            out2 = md.extract_metadata(ctx, str(jpg))
            self.assertEqual(out2["format"], "JPEG")
        finally:
            ctx.close()
            tmpctx.cleanup()

    def test_hostile_eml_does_not_raise(self) -> None:
        meta = md._eml_meta(b"From: \x00\x01\x02 garbage\r\n\r\nbody")
        self.assertEqual(meta["format"], "EMAIL")  # partial, never raises


# ── de-restriction: caps ─────────────────────────────────────────────────────


class DeRestrictionCapsTest(unittest.TestCase):
    def test_database_max_limit_raised(self) -> None:
        self.assertEqual(db_tool._MAX_LIMIT, 5000)

    def test_sandbox_timeout_clamp_is_configurable(self) -> None:
        self.assertEqual(sc.MAX_TIMEOUT, 300.0)
        self.assertEqual(sc.MAX_TIMEOUT_POWER, 1800.0)
        root = tempfile.mkdtemp(prefix="sb-")
        try:
            interp = sc.CodeInterpreter(root=root)
            self.assertEqual(interp.max_timeout, sc.MAX_TIMEOUT)
            interp.max_timeout = sc.MAX_TIMEOUT_POWER
            self.assertEqual(interp.max_timeout, 1800.0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_sandbox_register_uses_power_gate(self) -> None:
        src = inspect.getsource(sc)
        self.assertIn("power_mode_for", src)
        self.assertIn("MAX_TIMEOUT_POWER", src)


# ── wiring ───────────────────────────────────────────────────────────────────


class WiringTest(unittest.TestCase):
    def test_registry_registers_osint_tools(self) -> None:
        from nomorals.tools.registry import ToolRegistry
        ctx, tmp = _make_context()
        try:
            reg = ToolRegistry(ctx)
            reg.register_builtins()
            names = reg.names()
            self.assertIn("osint_graph", names)
            self.assertIn("osint_campaign", names)
        finally:
            ctx.close()
            tmp.cleanup()

    def test_devon_catalog_and_handlers(self) -> None:
        from nomorals.agents import devon
        names = {n for n, _ in devon.TOOL_CATALOG}
        self.assertIn("osint_graph", names)
        self.assertIn("osint_campaign", names)
        self.assertTrue(hasattr(devon.DevonAgent, "_tool_osint_graph"))
        self.assertTrue(hasattr(devon.DevonAgent, "_tool_osint_campaign"))

    def test_control_help_lists_campaign_and_graph(self) -> None:
        from nomorals.social.chat import control
        # wave 65: free-text commands are uncapped (seeds lists can be long)
        self.assertIsNone(control.CONTROL_COMMANDS["osint"][1])
        src = inspect.getsource(control)
        self.assertIn("/osint campaign", src)
        self.assertIn("/osint graph", src)

    def test_campaign_tool_end_to_end_with_fake_registry(self) -> None:
        # production path: _production_sources -> context.tools.call —
        # verified with a fake tool namespace (no network)
        from nomorals.agents.osint_graph import _production_sources
        fake = _FakeTool({
            "phone_investigate": json.dumps({
                "e164": "+2348031234567",
                "contact": "seun.ade@corp.ng @seunade corp.ng"}),
            "email_investigate": json.dumps({"mx": "mx.corp.ng"}),
            "username_check": json.dumps({"found": ["github"]}),
            "osint_domain": json.dumps({"ip": "203.116.44.9"}),
            "osint_ip": json.dumps({"asn": "AS123"}),
        })
        ctx, tmp = _make_context()
        try:
            ctx.tools = fake
            sources = _production_sources(ctx)
            runner = CampaignRunner(IdentityGraph(ctx), sources,
                                    max_steps=40, time_budget=30.0)
            dossier = runner.run([("phone", "2348031234567")])
            tools_used = {name for name, _ in fake.calls}
            self.assertIn("phone_investigate", tools_used)
            self.assertIn("email_investigate", tools_used)
            self.assertIn("username_check", tools_used)
            self.assertIn("osint_domain", tools_used)
            self.assertGreaterEqual(dossier["steps"], 4)
            # the graph absorbed the campaign
            self.assertGreaterEqual(IdentityGraph(ctx).stats()["nodes"], 3)
        finally:
            ctx.close()
            tmp.cleanup()

    def test_graph_tool_ingest_via_registry(self) -> None:
        from nomorals.tools.registry import ToolRegistry
        ctx, tmp = _make_context()
        try:
            reg = ToolRegistry(ctx)
            reg.register_builtins()
            out = reg.call("osint_graph", action="ingest",
                           text="seun.ade@corp.ng and 203.116.44.9",
                           source="unit")
            self.assertTrue(out.ok)
            self.assertGreaterEqual(out.value["ingested"], 2)
            out2 = reg.call("osint_graph", action="clusters")
            self.assertTrue(out2.ok)
            self.assertTrue(out2.value["clusters"])
            out3 = reg.call("osint_graph", action="clear")
            self.assertTrue(out3.ok)
            self.assertTrue(out3.value["cleared"])
        finally:
            ctx.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
