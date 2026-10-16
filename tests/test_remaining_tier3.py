"""Remaining Tier 3 depth: invariants for the new modules + root additions.

* sync never silently discards a conflict (map fork + list fork both recorded)
* list merge never loses an append; counter merge is a CRDT sum; merge idempotent
* egress audit logs every decision + blocks private/link-local by default
* archives traversal-safe (names refused; hostile archive refused whole)
* exporter excludes partner/vault/community; contribution report flags leaks
* bench honest metric flags a claimed-vs-measured mismatch
* compat detection works (known + unknown names)
* remote VCPUs route by capability with local fallback + visible routing
* travel<->community: share + RSVP + exact cost split
* planning->commerce logistics: fallback estimate + injected quote + attach
* builders_proxy: routing + TLS verify + deploy story
"""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from nomorals.archives import (
    detect_file_magic,
    detect_magic_bytes,
    extract_for_downloader,
    is_safe_member_name,
    validate_archive_members,
)
from nomorals.bench import honest_speedup_report, verify_speedup
from nomorals.builders_proxy import (
    build_proxy_command,
    deploy_story,
    route_request,
    verify_tls_cert,
)
from nomorals.compat import detect_optional_dep
from nomorals.exporter import (
    contribution_report,
    filter_excluded_dialogs,
    is_excluded_dialog,
)
from nomorals.planning.commerce_logistics import (
    LogisticsDescriptor,
    attach_logistics_step,
    estimate_cost,
    estimate_eta,
    plan_logistics_step,
)
from nomorals.security.egress_audit import EgressAuditor, is_blocked_network
from nomorals.sync.namespaces import Namespace, NamespaceSet
from nomorals.travel.community_impl import CommunityGroupTripBridge
from nomorals.travel.groups import GroupTripStore
from nomorals.workspace.remote_vcpu import RemoteFarm, RemoteWorker


class TestSyncNamespaces(unittest.TestCase):
    def test_map_fork_never_silently_discarded(self):
        a = Namespace("prefs", "map", device_id="phone")
        b = Namespace("prefs", "map", device_id="laptop")
        a.put("theme", "dark", ts=100.0, device="phone")
        b.put("theme", "light", ts=100.0, device="laptop")
        new = a.merge(b)
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0].values, ["dark", "light"])
        # Tiebreak is (ts, device): "phone" > "laptop", so "dark" wins.
        self.assertEqual(new[0].resolved_to, "dark")
        self.assertEqual(a.value()["theme"], "dark")
        self.assertEqual(len(a.conflicts()), 1)

    def test_map_lww_no_conflict_when_timestamps_differ(self):
        a = Namespace("prefs", "map", device_id="phone")
        b = Namespace("prefs", "map", device_id="laptop")
        a.put("theme", "dark", ts=100.0, device="phone")
        b.put("theme", "light", ts=200.0, device="laptop")
        self.assertEqual(a.merge(b), [])
        self.assertEqual(a.value()["theme"], "light")
        self.assertEqual(a.conflicts(), [])

    def test_list_merge_never_loses_appends(self):
        a = Namespace("todo", "list", device_id="phone")
        b = Namespace("todo", "list", device_id="laptop")
        a.append("buy milk", item_id="i1", ts=1.0, device="phone")
        b.append("buy eggs", item_id="i2", ts=1.0, device="laptop")
        b.append("buy milk", item_id="i1", ts=1.0, device="phone")
        a.merge(b)
        self.assertEqual(sorted(a.value()), ["buy eggs", "buy milk"])

    def test_list_same_id_fork_keeps_both_and_flags_conflict(self):
        a = Namespace("todo", "list", device_id="phone")
        b = Namespace("todo", "list", device_id="laptop")
        a.append("v1", item_id="i1", ts=1.0, device="phone")
        b.append("v2", item_id="i1", ts=1.0, device="laptop")
        new = a.merge(b)
        self.assertEqual(len(new), 1)
        self.assertEqual(sorted(a.value()), ["v1", "v2"])
        flagged = [e for e in a.entries() if e["conflicted"]]
        self.assertEqual(len(flagged), 2)

    def test_counter_crdt_merge_sums_devices(self):
        a = Namespace("clicks", "counter", device_id="phone")
        b = Namespace("clicks", "counter", device_id="laptop")
        a.increment(3, device="phone")
        a.increment(2, device="phone")
        b.increment(4, device="laptop")
        b.increment(1, device="phone")  # stale shard, below a's 5
        a.merge(b)
        self.assertEqual(a.value(), 9)  # 5 phone + 4 laptop
        self.assertEqual(a.conflicts(), [])

    def test_merge_idempotent(self):
        a = Namespace("prefs", "map", device_id="phone")
        b = Namespace("prefs", "map", device_id="laptop")
        a.put("theme", "dark", ts=100.0, device="phone")
        b.put("theme", "light", ts=100.0, device="laptop")
        a.merge(b)
        a.merge(b)
        self.assertEqual(len(a.conflicts()), 1)
        lst_a = Namespace("todo", "list", device_id="phone")
        lst_b = Namespace("todo", "list", device_id="laptop")
        lst_a.append("x", item_id="i1", ts=1.0, device="phone")
        lst_b.append("y", item_id="i2", ts=1.0, device="laptop")
        lst_a.merge(lst_b)
        lst_a.merge(lst_b)
        self.assertEqual(lst_a.value(), ["x", "y"])

    def test_namespace_set_conflicts_visible(self):
        mine = NamespaceSet(device_id="phone")
        theirs = NamespaceSet(device_id="laptop")
        mine.get_or_create("prefs", "map").put("k", "1", ts=5.0,
                                               device="phone")
        theirs.get_or_create("prefs", "map").put("k", "2", ts=5.0,
                                                 device="laptop")
        mine.merge(theirs)
        self.assertEqual(len(mine.conflicts()), 1)

    def test_roundtrip(self):
        ns = Namespace("prefs", "map", device_id="phone")
        ns.put("k", "v", ts=1.0, device="phone")
        restored = Namespace.from_dict(ns.to_dict())
        self.assertEqual(restored.value(), {"k": "v"})


class TestEgressAudit(unittest.TestCase):
    def test_blocks_private_link_local_loopback(self):
        auditor = EgressAuditor()
        for host in ("10.1.2.3", "192.168.1.1", "172.16.0.5", "127.0.0.1",
                     "169.254.169.254", "fe80::1", "::1", "0.0.0.0"):
            decision = auditor.audit(host, 443)
            self.assertFalse(decision.allowed, host)
            self.assertIn("blocked", decision.reason, host)
        self.assertEqual(len(auditor.entries()), 8)

    def test_allows_public_ip(self):
        auditor = EgressAuditor()
        decision = auditor.audit("8.8.8.8", 53)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "public IP literal")

    def test_invalid_port_blocked(self):
        auditor = EgressAuditor()
        self.assertFalse(auditor.audit("8.8.8.8", 0).allowed)
        self.assertFalse(auditor.audit("8.8.8.8", 70000).allowed)

    def test_allowlist_bypasses(self):
        auditor = EgressAuditor(allowlist=("api.example.com",
                                           ".internal.example.com"))
        self.assertTrue(auditor.audit("api.example.com", 443).allowed)
        self.assertTrue(auditor.audit("svc.internal.example.com", 443).allowed)

    def test_audit_report_counts_everything(self):
        auditor = EgressAuditor()
        auditor.audit("8.8.8.8", 53)
        auditor.audit("10.0.0.1", 80)
        auditor.audit("192.168.0.1", 80)
        report = auditor.audit_report()
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["allowed"], 1)
        self.assertEqual(report["blocked"], 2)
        self.assertEqual(len(report["blocked_entries"]), 2)

    def test_is_blocked_network(self):
        blocked, _ = is_blocked_network("169.254.169.254")
        self.assertTrue(blocked)
        blocked, _ = is_blocked_network("93.184.216.34")
        self.assertFalse(blocked)


class TestArchivesSafety(unittest.TestCase):
    def test_detect_magic_bytes(self):
        self.assertEqual(detect_magic_bytes(b"PK\x03\x04" + b"\x00" * 10),
                         "zip")
        self.assertEqual(detect_magic_bytes(b"\x1f\x8b" + b"\x00" * 10),
                         "gzip")
        self.assertEqual(detect_magic_bytes(b"7z\xbc\xaf\x27\x1c"), "7z")
        self.assertEqual(detect_magic_bytes(b"hello world"), "unknown")
        tar_head = b"\x00" * 257 + b"ustar" + b"\x00" * 10
        self.assertEqual(detect_magic_bytes(tar_head), "tar")

    def test_detect_file_magic_tarball_hint(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bundle.tar.gz"
            import gzip
            p.write_bytes(gzip.compress(b"fake tar"))
            self.assertEqual(detect_file_magic(p), "tar.gz")

    def test_member_names(self):
        self.assertTrue(is_safe_member_name("a/b.txt")[0])
        for bad in ("../evil.txt", "a/../../evil.txt", "/etc/passwd",
                    "C:\\Windows\\x", "", "a\x00b"):
            ok, reason = is_safe_member_name(bad)
            self.assertFalse(ok, bad)
            self.assertTrue(reason, bad)

    def test_validate_archive_members(self):
        verdict = validate_archive_members(["ok.txt", "../evil.txt"])
        self.assertEqual(verdict["safe"], ["ok.txt"])
        self.assertEqual(len(verdict["unsafe"]), 1)

    def test_extract_for_downloader_refuses_hostile_whole(self):
        class StubArchivist:
            def __init__(self):
                self.extracted = False

            def list(self, path):
                return {"entries": [{"name": "ok.txt"},
                                     {"name": "../evil.txt"}]}

            def extract(self, path, dest=""):
                self.extracted = True
                return {"extracted": 1}

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hostile.zip"
            with zipfile.ZipFile(p, "w") as zf:
                zf.writestr("../evil.txt", "x")
                zf.writestr("ok.txt", "y")
            stub = StubArchivist()
            result = extract_for_downloader(stub, str(p))
            self.assertTrue(result["blocked"])
            self.assertEqual(result["extracted"], 0)
            self.assertFalse(stub.extracted)

    def test_extract_for_downloader_passes_clean(self):
        class StubArchivist:
            def list(self, path):
                return {"entries": [{"name": "ok.txt"}]}

            def extract(self, path, dest=""):
                return {"extracted": 1, "files": ["ok.txt"]}

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "clean.zip"
            with zipfile.ZipFile(p, "w") as zf:
                zf.writestr("ok.txt", "y")
            result = extract_for_downloader(StubArchivist(), str(p))
            self.assertFalse(result["blocked"])
            self.assertEqual(result["extracted"], 1)


class TestExporterExclusions(unittest.TestCase):
    def test_partner_vault_community_excluded(self):
        for title in ("Partner chat", "vault: secrets", "Community Lagos",
                      "trial account", "my private diary", "SECRET notes"):
            blocked, reason = is_excluded_dialog(title)
            self.assertTrue(blocked, title)
            self.assertTrue(reason, title)
        self.assertFalse(is_excluded_dialog("Ada — Lagos Life")[0])
        self.assertFalse(is_excluded_dialog("Mum")[0])

    def test_filter_excluded_dialogs(self):
        dialogs = [("Partner chat", [(True, "hi")]),
                   ("Mum", [(True, "hi")]),
                   ("community: devs", [(True, "hi")])]
        kept, excluded = filter_excluded_dialogs(dialogs)
        self.assertEqual([t for t, _ in kept], ["Mum"])
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(e["reason"] for e in excluded))

    def test_contribution_report_flags_leak(self):
        from nomorals.training.dataset import Example, Turn

        examples = [
            Example(turns=[Turn(role="user", content="hey"),
                           Turn(role="assistant", content="yo")],
                    source="Mum"),
            Example(turns=[Turn(role="assistant", content="secret")],
                    source="vault: secrets"),
        ]
        report = contribution_report(examples)
        self.assertEqual(report["total_examples"], 2)
        self.assertEqual(report["owner_turns"], 2)
        self.assertEqual(report["other_turns"], 1)
        self.assertFalse(report["clean"])
        self.assertIn("vault: secrets", report["excluded_sources_present"])


class TestBenchHonesty(unittest.TestCase):
    def test_matching_claim_passes(self):
        check = verify_speedup(claimed=2.0, python_ms=200.0, native_ms=100.0)
        self.assertFalse(check["mismatch"])
        self.assertAlmostEqual(check["measured"], 2.0)

    def test_mismatch_flagged(self):
        check = verify_speedup(claimed=12.0, python_ms=120.0,
                               native_ms=100.0)
        self.assertTrue(check["mismatch"])
        self.assertAlmostEqual(check["measured"], 1.2)

    def test_zero_timings_flagged(self):
        self.assertTrue(verify_speedup(claimed=5.0, python_ms=0.0,
                                       native_ms=100.0)["mismatch"])

    def test_report_honesty(self):
        report = honest_speedup_report({
            "good": {"python_ms": 200.0, "native_ms": 100.0,
                     "speedup": 2.0},
            "liar": {"python_ms": 120.0, "native_ms": 100.0,
                     "speedup": 12.0},
            "no_timings": {"backend": "pure-python"},
        })
        self.assertFalse(report["honest"])
        self.assertEqual(len(report["mismatches"]), 1)
        self.assertEqual(report["mismatches"][0]["name"], "liar")
        self.assertEqual(report["checked"], 2)


class TestCompatDetection(unittest.TestCase):
    def test_known_feature(self):
        result = detect_optional_dep("numpy")
        self.assertEqual(result["name"], "numpy")
        self.assertIsInstance(result["available"], bool)
        self.assertTrue(result["install"])
        self.assertTrue(result["known"])

    def test_unknown_module_not_available(self):
        result = detect_optional_dep("definitely_not_a_real_module_xyz")
        self.assertFalse(result["available"])
        self.assertFalse(result["known"])

    def test_stdlib_module_available(self):
        self.assertTrue(detect_optional_dep("json")["available"])


class TestRemoteVcpu(unittest.TestCase):
    def test_routes_to_best_worker_by_capability(self):
        farm = RemoteFarm()
        farm.register(RemoteWorker("laptop", host="laptop",
                                   capabilities=frozenset({"cpu"})))
        farm.register(RemoteWorker("kaggle", host="kaggle-gpu",
                                   capabilities=frozenset({"cpu", "gpu",
                                                           "train"}),
                                   gpu=True, latency_ms=120.0))
        result = farm.submit(lambda: "trained",
                             require={"gpu", "train"}).result(timeout=10)
        self.assertEqual(result.worker, "kaggle")
        self.assertEqual(result.route, "remote")
        self.assertEqual(result.result, "trained")
        self.assertFalse(result.fallback)
        farm.shutdown()

    def test_fallback_local_on_remote_failure(self):
        farm = RemoteFarm()
        def _boom(fn, args, kwargs):
            raise RuntimeError("gpu host down")
        farm.register(RemoteWorker("kaggle", host="kaggle-gpu",
                                   capabilities=frozenset({"cpu", "gpu"}),
                                   gpu=True), runner=_boom)
        result = farm.submit(lambda: 42, require={"gpu"}).result(timeout=10)
        self.assertEqual(result.route, "local-fallback")
        self.assertTrue(result.fallback)
        self.assertEqual(result.result, 42)
        self.assertIn("gpu host down", result.error)
        self.assertEqual(farm.stats["local_fallback"], 1)
        farm.shutdown()

    def test_no_capable_worker_goes_local(self):
        farm = RemoteFarm()
        farm.register(RemoteWorker("laptop", host="laptop",
                                   capabilities=frozenset({"cpu"})))
        result = farm.submit(lambda: "x",
                             require={"gpu"}).result(timeout=10)
        self.assertEqual(result.route, "local")
        self.assertEqual(result.worker, "local")
        farm.shutdown()

    def test_pick_prefers_idle_over_busy(self):
        farm = RemoteFarm()
        farm.register(RemoteWorker("a", host="a",
                                   capabilities=frozenset({"cpu"})))
        farm.register(RemoteWorker("b", host="b",
                                   capabilities=frozenset({"cpu"})))
        with farm._lock:
            farm._inflight["a"] = 5
        self.assertEqual(farm.pick_worker({"cpu"}).name, "b")
        farm.shutdown()


class TestTravelCommunity(unittest.TestCase):
    def _bridge(self):
        tmp = tempfile.mkdtemp(prefix="tier3-trip-")
        store = GroupTripStore("test-group",
                               db_path=str(Path(tmp) / "trips.db"))
        posted: list[dict] = []

        def post(payload):
            posted.append(payload)
            return "msg-1"

        return CommunityGroupTripBridge(store, post=post), posted

    def test_share_rsvp_cost_split(self):
        bridge, posted = self._bridge()
        trip = bridge.create_group_trip(
            "Lagos weekend", ["Ada", "Chidi", "Mama"],
            destination="Lekki", start_date="2026-12-01",
            end_date="2026-12-03", created_by="Ada")
        bridge.add_cost(trip.id, "Ada", 900_000, "villa deposit")
        share = bridge.share_itinerary(trip.id, "community:lagos-weekend")
        self.assertEqual(share.destination, "Lekki")
        self.assertEqual(share.post_ref, "msg-1")
        self.assertEqual(len(posted), 1)
        self.assertIn("Lekki", share.format_post())
        # cost split: exact, sums to the total
        split = bridge.cost_split(trip.id, 100_001)
        self.assertEqual(sum(split.values()), 100_001)
        self.assertEqual(set(split), {"Ada", "Chidi", "Mama"})
        # settle-up from real expenses: Chidi + Mama owe Ada
        transfers = bridge.settle(trip.id)
        self.assertEqual(sum(a for _, _, a in transfers), 600_000)
        # RSVP flow
        bridge.rsvp(trip.id, "Ada", "going")
        bridge.rsvp(trip.id, "Chidi", "maybe")
        bridge.rsvp(trip.id, "Mama", "declined")
        summary = bridge.rsvp_summary(trip.id)
        self.assertEqual(summary["counts"], {"going": 1, "maybe": 1,
                                             "declined": 1, "no_reply": 0})
        with self.assertRaises(ValueError):
            bridge.rsvp(trip.id, "Ada", "perhaps")
        with self.assertRaises(ValueError):
            bridge.rsvp("no-such-trip", "Ada", "going")

    def test_no_post_hook_share_stays_local(self):
        tmp = tempfile.mkdtemp(prefix="tier3-trip-")
        store = GroupTripStore("test-group",
                               db_path=str(Path(tmp) / "trips.db"))
        bridge = CommunityGroupTripBridge(store, post=None)
        trip = bridge.create_group_trip("Trip", ["Ada"], destination="Abuja")
        share = bridge.share_itinerary(trip.id, "community:x")
        self.assertEqual(share.post_ref, "")


class TestPlanningLogistics(unittest.TestCase):
    def test_fallback_estimate_is_labelled(self):
        desc = LogisticsDescriptor(carrier="gig", origin="Lagos",
                                   destination="Abuja", weight_kg=2.0,
                                   distance_km=700.0)
        cost = estimate_cost(desc)
        self.assertTrue(cost["estimated"])
        self.assertIn("fallback", cost["source"])
        self.assertGreater(cost["fee_minor"], 0)
        eta = estimate_eta(desc)
        self.assertTrue(eta["estimated"])
        self.assertEqual(eta["eta"], "2–4 days")

    def test_injected_quote_wins_verbatim(self):
        desc = LogisticsDescriptor(carrier="kwik", origin="Lekki",
                                   destination="Ikeja", weight_kg=1.0,
                                   distance_km=15.0)

        def quote_fn(d):
            return {"carrier": "Kwik", "fee_minor": 250_000,
                    "currency": "NGN", "estimated": False,
                    "source": "live"}

        def eta_fn(d):
            return {"carrier": "Kwik", "eta": "2–6 hours",
                    "estimated": False, "source": "live"}

        step = plan_logistics_step("deliver order", desc,
                                   quote_fn=quote_fn, eta_fn=eta_fn)
        self.assertEqual(step.cost_minor, 250_000)
        self.assertEqual(step.eta, "2–6 hours")
        self.assertFalse(step.estimated)
        self.assertIn("deliver order", step.format())

    def test_attach_does_not_mutate(self):
        desc = LogisticsDescriptor()
        step = plan_logistics_step("ship", desc)
        steps = [{"kind": "book"}]
        out = attach_logistics_step(steps, step)
        self.assertEqual(len(steps), 1)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1]["kind"], "logistics")

    def test_bad_weight_rejected(self):
        with self.assertRaises(ValueError):
            estimate_cost(LogisticsDescriptor(weight_kg=0))


class TestBuildersProxyAdditions(unittest.TestCase):
    def test_route_request_strips_prefix(self):
        routed = route_request("/myapp/api/v1", prefix="/myapp",
                               domain="example.com",
                               backend_host="127.0.0.1", backend_port=8002)
        self.assertEqual(routed["target_path"], "/api/v1")
        self.assertEqual(routed["host_header"], "example.com")
        self.assertEqual(routed["backend_url"],
                         "http://127.0.0.1:8002/api/v1")

    def test_verify_missing_cert_not_ok(self):
        result = verify_tls_cert("/nonexistent/cert.pem")
        self.assertFalse(result["ok"])
        self.assertFalse(result["exists"])
        self.assertTrue(result["error"])

    def test_deploy_story(self):
        story = deploy_story(backend="127.0.0.1:8002",
                             listen="0.0.0.0:9000", prefix="/app",
                             domain="example.com", tls=True)
        self.assertIn("--tls", story["command"])
        self.assertTrue(story["public_url"].startswith("https://"))
        self.assertTrue(any("self-signed" in n for n in story["notes"]))
        cmd = build_proxy_command(backend="127.0.0.1:8002",
                                  listen="0.0.0.0:9000")
        self.assertIn("-m", cmd)
        self.assertIn("nomorals.builders_proxy", cmd)


if __name__ == "__main__":
    unittest.main()
