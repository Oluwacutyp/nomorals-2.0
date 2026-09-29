"""The owner's 2026-09-10 build-out: per-chat rate limits (owner never
dropped), inbound media gates, single-account trial vault + delivery,
feature flags, and the self-improvement arena (research → knowledge →
builds → owner review → export for training).

Fully offline: stubbed search, canned pages, fake gateways, no network.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
import unittest
from typing import Any

from nomorals.agents.arena import Arena
from nomorals.agents.arena.topics import TOPIC_BANK, sample_topic
from nomorals.agents.context import build_context
from nomorals.agents.features import FEATURES, FeatureRegistry, feature_enabled
from nomorals.agents.search.engine import SearchEngine
from nomorals.agents.trial import TrialFlow, TrialVault, active_delivery_platforms
from nomorals.core.config import env_var_path, load_settings
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.social.chat.base import ChatAdapter, ChatKind, ChatMessage, ChatRef, SendResult
from nomorals.social.chat.control import parse_control, help_text
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.telegram import media_download_allowed
from nomorals.storage.db import Database


def _settings(tmp: str, **overrides: str):
    base = {"home": tmp, "partner.platforms": "local", "chat.local_enabled": "true"}
    base.update(overrides)
    return load_settings(overrides=base)


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str = "a") -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[tuple[str, str]] = []
        self._handler = None
        self.started_flag = False

    def run(self, handler) -> None:
        self._handler = handler
        self.started_flag = True
        while not self.stopped:
            time.sleep(0.01)

    def push(self, chat: ChatRef, text: str, *, sender: str = "someone") -> None:
        assert self._handler is not None
        self._deliver(self._handler, ChatMessage(chat=chat, incoming=True, text=text, sender=sender))

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        self.sent.append((chat.key, text))
        return SendResult(ok=True, platform=self.name, message_id=f"m{len(self.sent)}")

    def wait_started(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.started_flag:
                return True
            time.sleep(0.01)
        return False


def _dm(platform: str, chat_id: str) -> ChatRef:
    return ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM, peer="someone")


# ── 1. per-chat rate limits: owner never dropped, chats isolated ─────────────


class PerChatRateLimitTest(unittest.TestCase):
    def _gateway(self, **kw) -> ChatGateway:
        db = Database(":memory:")
        db.migrate()
        return ChatGateway({"a": FakeAdapter("a")}, db=db, **kw)

    def test_owner_is_never_rate_limited(self) -> None:
        gw = self._gateway(max_per_hour=3, owner_chats={"a:owner"})
        received: dict[str, int] = {"a:owner": 0, "a:spammer": 0}
        gw.start(lambda m: received.__setitem__(m.chat.key, received.get(m.chat.key, 0) + 1))
        adapter = gw.adapters["a"]
        self.assertTrue(adapter.wait_started())
        for i in range(6):
            adapter.push(_dm("a", "spammer"), f"spam {i}")
        for i in range(6):
            adapter.push(_dm("a", "owner"), f"owner cmd {i}")
        time.sleep(0.1)
        # spammer: 3 through, 3 dropped
        self.assertEqual(received["a:spammer"], 3)
        self.assertEqual(gw.stats["dropped_rate_limited"], 3)
        # owner: every single one delivered — the original bug
        self.assertEqual(received["a:owner"], 6)
        gw.stop()

    def test_chat_flood_does_not_starve_other_chats(self) -> None:
        gw = self._gateway(max_per_hour=3)
        received: dict[str, int] = {}
        gw.start(lambda m: received.__setitem__(m.chat.key, received.get(m.chat.key, 0) + 1))
        adapter = gw.adapters["a"]
        self.assertTrue(adapter.wait_started())
        for i in range(6):
            adapter.push(_dm("a", "flood"), "noise")
        for i in range(3):
            adapter.push(_dm("a", "calm"), "quiet please")
        time.sleep(0.1)
        self.assertEqual(received["a:flood"], 3)
        self.assertEqual(received["a:calm"], 3)  # own window, untouched by the flood
        gw.stop()

    def test_set_rate_limit_zero_unlimits_live(self) -> None:
        gw = self._gateway(max_per_hour=2)
        received: list[str] = []
        gw.start(lambda m: received.append(m.chat.key))
        adapter = gw.adapters["a"]
        self.assertTrue(adapter.wait_started())
        adapter.push(_dm("a", "flood"), "1")
        adapter.push(_dm("a", "flood"), "2")
        adapter.push(_dm("a", "flood"), "3")  # dropped: over the 2/h cap
        time.sleep(0.05)
        self.assertEqual(gw.stats["dropped_rate_limited"], 1)
        gw.set_rate_limit(0)  # power mode
        adapter.push(_dm("a", "flood"), "4")
        adapter.push(_dm("a", "flood"), "5")
        time.sleep(0.05)
        self.assertEqual(gw.stats["dropped_rate_limited"], 1)  # no new drops
        self.assertEqual(received[-2:], ["a:flood", "a:flood"])
        gw.set_rate_limit(2)  # back to base
        gw.stop()


# ── 2. inbound media download gate ───────────────────────────────────────────


class MediaGateTest(unittest.TestCase):
    MB = 1024 * 1024

    def test_group_media_off_by_default(self) -> None:
        self.assertFalse(media_download_allowed(ChatKind.GROUP, 1 * self.MB, False, 20.0))

    def test_group_media_opt_in(self) -> None:
        self.assertTrue(media_download_allowed(ChatKind.GROUP, 1 * self.MB, True, 20.0))

    def test_dm_media_always(self) -> None:
        self.assertTrue(media_download_allowed(ChatKind.DM, 5 * self.MB, False, 20.0))

    def test_oversized_file_skipped(self) -> None:
        self.assertFalse(media_download_allowed(ChatKind.DM, 25 * self.MB, False, 20.0))
        self.assertTrue(media_download_allowed(ChatKind.DM, 19 * self.MB, False, 20.0))

    def test_zero_cap_means_no_cap(self) -> None:
        self.assertTrue(media_download_allowed(ChatKind.DM, 500 * self.MB, False, 0.0))


# ── 3. trial vault (encrypted single-account credentials) ────────────────────


class VaultTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-vault-")
        self.vault = TrialVault(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_roundtrip(self) -> None:
        self.vault.store("Clickworker", "owner@mail", "S3cret!pass", note="trial")
        entry = self.vault.get("clickworker")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["login"], "owner@mail")
        self.assertEqual(entry["secret"], "S3cret!pass")
        self.assertEqual(entry["note"], "trial")

    def test_stored_secret_is_not_plaintext_on_disk(self) -> None:
        self.vault.store("Prolific", "owner@mail", "S3cret!pass")
        raw = (self.vault.home / "trial_accounts.json").read_text("utf-8")
        self.assertNotIn("S3cret!pass", raw)

    def test_tamper_detected(self) -> None:
        self.vault.store("X", "l", "password-123")
        path = self.vault.data_path
        raw = json.loads(path.read_text("utf-8"))
        blob = bytearray(bytes.fromhex(raw["x"]["secret"]))
        blob[-1] ^= 0xFF  # flip a ciphertext byte
        raw["x"]["secret"] = bytes(blob).hex()
        path.write_text(json.dumps(raw), "utf-8")
        entry = self.vault.get("x")
        self.assertTrue(entry["unreadable"])
        self.assertIsNone(entry["secret"])

    def test_list_and_delete(self) -> None:
        self.vault.store("A", "a@x", "p1")
        self.vault.store("B", "b@x", "p2")
        names = [row["platform"] for row in self.vault.list()]
        self.assertEqual(names, ["a", "b"])
        self.assertTrue(self.vault.delete("a"))
        self.assertFalse(self.vault.delete("a"))
        self.assertEqual([row["platform"] for row in self.vault.list()], ["b"])

    def test_mask(self) -> None:
        self.assertEqual(TrialVault.mask("abcd"), "••••")
        self.assertEqual(TrialVault.mask("hunter2"), "h…2 (7 chars)")


class _FakeGateway:
    def __init__(self, running: set[str]) -> None:
        self._running = set(running)
        self.sent: list[tuple[str, str, str]] = []

    def status(self) -> dict:
        return {p: {"running_in_session": True} for p in self._running} | {"_stats": {}}

    def send(self, platform: str, chat: ChatRef, text: str, **kw: Any) -> SendResult:
        self.sent.append((platform, chat.key, text))
        return SendResult(ok=True, platform=platform, message_id="m")


class TrialFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-trial-")
        self.settings = _settings(
            self.tmp.name,
            **{"partner.owner_chats": "whatsapp:99, telegram:1"},
        )
        self.context = build_context(self.settings, with_executor=False, with_tools=False)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_delivery_order_and_both_platforms(self) -> None:
        gw = _FakeGateway({"telegram", "whatsapp", "local"})
        self.assertEqual(active_delivery_platforms(gw), ["whatsapp", "telegram"])
        flow = TrialFlow(self.context)
        flow.save("Prolific", "owner@mail", "pw-123")
        reply = flow.deliver("prolific", gateway=gw, via="telegram:1")
        self.assertEqual(reply, "sent on: whatsapp, telegram")
        self.assertEqual(len(gw.sent), 2)
        self.assertEqual([s[0] for s in gw.sent], ["whatsapp", "telegram"])
        self.assertEqual(gw.sent[0][1], "whatsapp:99")  # owner chat from settings
        self.assertEqual(gw.sent[1][1], "telegram:1")
        for _plat, _key, body in gw.sent:
            self.assertIn("owner@mail", body)
            self.assertIn("pw-123", body)

    def test_only_active_platform_gets_it(self) -> None:
        gw = _FakeGateway({"telegram"})
        flow = TrialFlow(self.context)
        flow.save("P", "l", "s")
        reply = flow.deliver("p", gateway=gw, via="telegram:1")
        self.assertEqual(reply, "sent on: telegram")
        self.assertEqual(len(gw.sent), 1)

    def test_no_live_channel_returns_body_inline(self) -> None:
        flow = TrialFlow(self.context)
        flow.save("P", "l", "s")
        reply = flow.deliver("p", gateway=_FakeGateway(set()), via="local:console")
        self.assertTrue(reply.startswith("(no live"))
        self.assertIn("l", reply)

    def test_save_list_remove(self) -> None:
        flow = TrialFlow(self.context)
        flow.save("Platform", "login1", "pass1")
        self.assertIn("platform: login1", flow.list())
        self.assertIn("deleted", flow.remove("platform"))
        self.assertEqual(flow.list(), "no stored trial accounts yet.")

    def test_start_survives_missing_tools(self) -> None:
        flow = TrialFlow(self.context)  # built without tools — research degrades
        text = flow.start("some platform")
        self.assertIn("some platform", text)
        self.assertIn("/trial save some platform", text)


# ── 4. feature flags ─────────────────────────────────────────────────────────


class FeatureRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Database(":memory:")
        self.db.migrate()
        self.reg = FeatureRegistry(self.db)

    def test_defaults(self) -> None:
        self.assertFalse(self.reg.get("arena"))  # new heavy subsystem: off by default
        self.assertTrue(self.reg.get("vision"))
        self.assertTrue(self.reg.get("search"))
        self.assertTrue(self.reg.get("group_posts"))
        self.assertTrue(self.reg.get("proactive_dm"))

    def test_set_get_roundtrip_persists(self) -> None:
        self.assertTrue(self.reg.set("arena", True))
        self.assertTrue(self.reg.get("arena"))
        self.assertTrue(self.reg.set("arena", False))
        self.assertFalse(self.reg.get("arena"))
        row = self.db.query_one("SELECT value FROM kv_store WHERE key = 'feature.arena'")
        self.assertEqual(row["value"], "off")

    def test_unknown_feature_rejected(self) -> None:
        self.assertFalse(self.reg.set("banana", True))
        self.assertFalse(self.reg.get("banana"))

    def test_feature_enabled_without_db_uses_default(self) -> None:
        self.assertFalse(feature_enabled(object(), "arena"))
        self.assertTrue(feature_enabled(object(), "vision"))

    def test_list_shape(self) -> None:
        rows = self.reg.list()
        self.assertEqual(len(rows), len(FEATURES))
        for row in rows:
            self.assertIn("name", row)
            self.assertIn("on", row)
            self.assertIn("description", row)


# ── 5. config wiring ─────────────────────────────────────────────────────────


class ArenaConfigTest(unittest.TestCase):
    def test_env_paths(self) -> None:
        self.assertEqual(env_var_path("NM_CHAT_MEDIA_IN_GROUPS"), "chat.media_in_groups")
        self.assertEqual(env_var_path("NM_CHAT_MEDIA_MAX_MB"), "chat.media_max_mb")
        self.assertEqual(env_var_path("NM_ARENA_BUILD"), "arena.build")
        self.assertEqual(env_var_path("NM_ARENA_ENABLED"), "arena.enabled")
        self.assertEqual(env_var_path("NM_ARENA_INTERVAL_HOURS"), "arena.interval_hours")

    def test_defaults(self) -> None:
        s = load_settings()
        self.assertFalse(s.chat.media_in_groups)
        self.assertEqual(s.chat.media_max_mb, 20.0)
        self.assertFalse(s.arena.enabled)
        self.assertFalse(s.arena.build)
        self.assertEqual(s.arena.research_pages, 3)

    def test_overrides(self) -> None:
        s = load_settings(overrides={"arena.build": "1", "chat.media_in_groups": "true"})
        self.assertTrue(s.arena.build)
        self.assertTrue(s.chat.media_in_groups)


# ── 6. new control-command parsing ───────────────────────────────────────────


class NewCommandParseTest(unittest.TestCase):
    def test_search_variants(self) -> None:
        self.assertEqual(parse_control("/search who created python").tail, "who created python")
        self.assertEqual(parse_control("/searchdeep quantum computing").kind, "searchdeep")
        self.assertEqual(parse_control("/searchleads").kind, "searchleads")
        self.assertEqual(parse_control("/searchhist 7").arg, "7")

    def test_features(self) -> None:
        self.assertEqual(parse_control("/features").kind, "features")
        cmd = parse_control("/features arena on")
        self.assertEqual((cmd.kind, cmd.arg), ("features", "arena"))

    def test_arena(self) -> None:
        self.assertEqual(parse_control("/arena").kind, "arena")
        self.assertEqual(parse_control("/arena run eBPF internals").tail, "run eBPF internals")
        cmd = parse_control("/arena approve abc123")
        self.assertEqual(cmd.arg, "approve")
        self.assertIn("abc123", cmd.tail)

    def test_trial(self) -> None:
        cmd = parse_control("/trial save Prolific me@mail pw")
        self.assertEqual(cmd.kind, "trial")
        self.assertEqual(cmd.tail, "save Prolific me@mail pw")
        # "save" has 2 args (min is 0 for bare /trial); the dispatch layer
        # answers with the precise usage line (covered in CliSmokeTest).
        short = parse_control("/trial save onlyone")
        self.assertIsNotNone(short)
        self.assertEqual(short.kind, "trial")
        self.assertIsNotNone(parse_control("/trial list"))

    def test_help_mentions_every_new_command(self) -> None:
        text = help_text()
        for kind in ("search", "searchdeep", "searchleads", "searchhist", "features", "arena", "trial"):
            self.assertIn(f"/{kind}", text)


# ── 7. arena: research → knowledge → stream → builds → review → export ──────


CANNED_REPORT = {
    "id": "r1", "query": "eBPF rootkits", "mode": "quick", "sub_queries": [],
    "model_summary": True, "pages_read": 2, "seconds": 1.5,
    "summary": "eBPF lets programs run in the kernel; attackers abuse this for hidden hooks.",
    "results": [{"url": "https://kernel.org/docs", "title": "eBPF docs"}],
}

BUILD_JSON = json.dumps({
    "name": "hexutil",
    "purpose": "tiny hex encoding helpers",
    "files": [{"path": "hexutil.py", "content": '"""Hex helpers."""\n\n\ndef to_hex(b: bytes) -> str:\n    return b.hex()\n'}],
})


class _BuildRouter:
    def chat(self, messages, params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        return LLMResponse(text=f"here you go:\n```json\n{BUILD_JSON}\n```", model="fake-builder")


class ArenaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-arena-")
        self.settings = _settings(self.tmp.name)
        self.context = build_context(self.settings, with_executor=False, with_tools=False)
        self.arena = Arena(self.context)
        self.arena._research = lambda topic: dict(CANNED_REPORT, query=topic)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_research_cycle_stores_knowledge_and_stream(self) -> None:
        result = self.arena.run_cycle(topic="eBPF rootkits", category="security")
        self.assertTrue(result["ok"])
        self.assertIn("eBPF", result["digest"])
        row = self.context.db.query_one("SELECT * FROM arena_knowledge WHERE topic = ?", ("eBPF rootkits",))
        self.assertIsNotNone(row)
        self.assertEqual(row["category"], "security")
        events = self.arena.stream(10)
        kinds = [e["kind"] for e in events]
        self.assertIn("topic", kinds)
        self.assertIn("digest", kinds)

    def test_run_cycle_with_real_engine_pages_read_list(self) -> None:
        # SearchEngine reports pages_read as a LIST of urls (the int in
        # CANNED_REPORT is the fixture's simplification). The cycle must
        # count the list, not int() it — that crash hit live on the phone.
        self.arena._research = lambda topic: dict(
            CANNED_REPORT,
            query=topic,
            pages_read=["https://kernel.org/docs", "https://example.com/b"],
        )
        result = self.arena.run_cycle(topic="arena crash regression", category="debug")
        self.assertTrue(result["ok"], f"cycle crashed: {result.get('error')}")
        self.assertEqual(result["pages_read"], ["https://kernel.org/docs", "https://example.com/b"])
        row = self.context.db.query_one(
            "SELECT * FROM arena_knowledge WHERE topic = ?", ("arena crash regression",)
        )
        self.assertIsNotNone(row)

    def test_run_cycle_refuses_empty_pages_list(self) -> None:
        self.arena._research = lambda topic: dict(CANNED_REPORT, query=topic, pages_read=[])
        result = self.arena.run_cycle(topic="no pages", category="debug")
        self.assertFalse(result["ok"])
        self.assertIn("nothing readable", result["error"])

    def test_no_build_without_power(self) -> None:
        self.settings.arena.build = True  # config says yes…
        result = self.arena.run_cycle(topic="t1")
        self.assertNotIn("build", result)  # …but power mode gates it

    def test_stats_counts_cycles_and_knowledge(self) -> None:
        self.arena.run_cycle(topic="stats topic", category="debug")
        s = self.arena.stats()
        self.assertEqual(s["cycles"], 1)
        self.assertEqual(s["knowledge"], 1)
        self.assertIsNotNone(s["last_cycle"])

    def test_digests_returns_last(self) -> None:
        self.arena.run_cycle(topic="digest topic", category="debug")
        rows = self.arena.digests(1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["topic"], "digest topic")
        self.assertIn("eBPF", rows[0]["digest"])  # the canned digest

    def test_custom_topics_take_priority_over_bank(self) -> None:
        self.arena.add_topic("my custom topic")
        result = self.arena.run_cycle()  # no topic → custom bank first
        self.assertTrue(result["ok"], f"cycle failed: {result.get('error')}")
        self.assertEqual(result["topic"], "my custom topic")
        self.assertEqual(result["category"], "custom")

    def test_custom_topic_not_repicked_until_digested(self) -> None:
        self.arena.add_topic("once topic")
        r1 = self.arena.run_cycle()
        self.assertEqual(r1["topic"], "once topic")
        # Digested now — a second run falls through to the built-in bank.
        r2 = self.arena.run_cycle()
        self.assertTrue(r2["ok"])
        self.assertNotEqual(r2["topic"], "once topic")

    def test_interval_roundtrip(self) -> None:
        self.assertIsNone(self.arena.interval_hours())
        self.assertTrue(self.arena.set_interval(2.5))
        self.assertEqual(self.arena.interval_hours(), 2.5)
        # Bounds are clamped, never zero.
        self.arena.set_interval(0.001)
        self.assertGreaterEqual(self.arena.interval_hours(), 0.05)

    def test_powered_build_proposes_and_approve_stages(self) -> None:
        self.settings.arena.build = True
        self.settings.partner.owner_key = "testkey"
        from nomorals.agents.power import power_mode_for

        power_mode_for(self.context).unlock("testkey")
        self.context.router = _BuildRouter()
        packets: list[str] = []
        result = self.arena.run_cycle(topic="hex history", category="history", notify=packets.append)
        build = result.get("build")
        self.assertIsNotNone(build, "a powered builder should propose something")
        self.assertEqual(build["syntax"], "ok")
        file = self.arena.build_root / build["id"] / "hexutil" / "hexutil.py"
        self.assertTrue(file.exists())
        self.assertEqual(len(packets), 1)
        self.assertIn(f"/arena approve {build['id']}", packets[0])
        # owner approves → staged outside the repo, DB updated
        reply = self.arena.approve(build["id"])
        self.assertIn("staged at", reply)
        staged = self.arena.approve_root / "hexutil" / "hexutil.py"
        self.assertTrue(staged.exists())
        row = self.context.db.query_one("SELECT status FROM arena_builds WHERE id = ?", (build["id"],))
        self.assertEqual(row["status"], "approved")
        decisions = [e for e in self.arena.stream(10) if e["kind"] == "decision"]
        self.assertTrue(any(e["payload"].get("decision") == "approved" for e in decisions))

    def test_deny_marks_rejected_but_keeps_stream(self) -> None:
        self.settings.arena.build = True
        self.settings.partner.owner_key = "k"
        from nomorals.agents.power import power_mode_for

        power_mode_for(self.context).unlock("k")
        self.context.router = _BuildRouter()
        result = self.arena.run_cycle(topic="t")
        build = result["build"]
        reply = self.arena.deny(build["id"])
        self.assertIn("denied", reply)
        row = self.context.db.query_one("SELECT status FROM arena_builds WHERE id = ?", (build["id"],))
        self.assertEqual(row["status"], "denied")

    def test_build_rejects_path_escape(self) -> None:
        evil = json.dumps({
            "name": "evil", "purpose": "escape",
            "files": [{"path": "../../etc/passwd.py", "content": "x = 1\n"}],
        })

        class _EvilRouter(_BuildRouter):
            def chat(self, messages, params=None, **kw):
                return LLMResponse(text=evil, model="fake")

        self.settings.arena.build = True
        self.settings.partner.owner_key = "k"
        from nomorals.agents.power import power_mode_for

        power_mode_for(self.context).unlock("k")
        self.context.router = _EvilRouter()
        result = self.arena.run_cycle(topic="escape attempt")
        self.assertIsNone(result.get("build"))

    def test_export_is_training_jsonl(self) -> None:
        self.arena.run_cycle(topic="export me", category="web")
        text = self.arena.export(10)
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["source"], "arena")
        self.assertIn("user", rec["messages"][0]["role"])
        self.assertIn("eBPF", rec["messages"][1]["content"])

    def test_sample_topic_avoids_already_digestd(self) -> None:
        db = self.context.db
        bank = TOPIC_BANK["web"]
        for i, topic in enumerate(bank[:-1]):  # digest everything but the last
            db.execute(
                "INSERT INTO arena_knowledge (id, topic, category, digest, sources, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"k{i}", topic, "web", "d", "[]", time.time()),
            )
        rng = random.Random(7)
        picked = {sample_topic(db, "web", rng) for _ in range(100)}
        self.assertEqual(picked, {( "web", bank[-1])})

    def test_research_failure_is_reported_not_raised(self) -> None:
        def _boom(topic):
            raise RuntimeError("no network")

        self.arena._research = _boom
        result = self.arena.run_cycle(topic="anything")
        self.assertFalse(result["ok"])
        self.assertIn("research failed", result["error"])

    def test_zero_page_report_is_not_stored_as_knowledge(self) -> None:
        self.arena._research = lambda topic: dict(
            CANNED_REPORT, pages_read=0,
            summary="extractive summary (no live model — top sentences, cited):\n  • (no readable text on the fetched pages)")
        result = self.arena.run_cycle(topic="unreachable topic")
        self.assertFalse(result["ok"])
        row = self.context.db.query_one("SELECT * FROM arena_knowledge WHERE topic = ?", ("unreachable topic",))
        self.assertIsNone(row, "an empty digest must not enter the training stream")


# ── 8. runtime dispatch: /search, /features, /arena, /trial from a chat ─────


class _StubTools:
    def call(self, name, **kw):
        from nomorals.core.result import Ok

        assert name == "web_search", name
        return Ok({"query": kw.get("query", ""), "count": 1,
                   "results": [{"url": "https://py.org/history", "title": "Python history",
                                "snippet": "created by Guido"}]})


class RuntimeDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.agents.partner_runtime import PartnerRuntime

        self.tmp = tempfile.TemporaryDirectory(prefix="nm-dispatch-")
        settings = _settings(self.tmp.name)
        self.context = build_context(settings, with_executor=False, with_tools=False)
        self.context.tools = _StubTools()
        self.context.router = object.__new__(object)  # replaced per test as needed
        self.adapter = FakeAdapter("local")
        self.gateway = ChatGateway(
            {"local": self.adapter}, db=self.context.db,
            owner_chats={"local:console"},
        )
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_search_delivers_report_in_chunks(self) -> None:
        orig = SearchEngine.run

        def _stub_run(self_, query, mode="quick", pages=3, crawl=False):
            return dict(CANNED_REPORT, query=query, mode=mode,
                        summary="Python was created by Guido van Rossum in 1991.")

        SearchEngine.run = _stub_run  # type: ignore[method-assign]
        try:
            reply = self.runtime.handle_control("/search who created python", "local:console")
        finally:
            SearchEngine.run = orig  # type: ignore[method-assign]
        self.assertEqual(reply, "")  # report already sent in chunks
        joined = "\n".join(t for _k, t in self.adapter.sent)
        self.assertIn("⏳ quick research", joined)
        self.assertIn("Guido van Rossum", joined)
        self.assertIn("kernel.org", joined)

    def test_searchdeep_requires_power(self) -> None:
        reply = self.runtime.handle_control("/searchdeep anything", "local:console")
        self.assertIn("power", reply)

    def test_search_feature_gate(self) -> None:
        FeatureRegistry(self.context.db).set("search", False)
        reply = self.runtime.handle_control("/search hello", "local:console")
        self.assertIn("search is off", reply)

    def test_features_list_and_toggle(self) -> None:
        listing = self.runtime.handle_control("/features", "local:console")
        self.assertIn("arena", listing)
        self.assertIn("vision", listing)
        reply = self.runtime.handle_control("/features arena on", "local:console")
        self.assertIn("arena: on", reply)
        reply = self.runtime.handle_control("/features banana on", "local:console")
        self.assertIn("unknown feature", reply)

    def test_arena_run_end_to_end(self) -> None:
        orig = SearchEngine.run
        SearchEngine.run = lambda self_, query, mode="quick", pages=3, crawl=False: dict(
            CANNED_REPORT, query=query)  # type: ignore[method-assign]
        try:
            reply = self.runtime.handle_control("/arena run", "local:console")
            self.assertIn("arena is off", reply)  # flag off by default
            self.runtime.handle_control("/features arena on", "local:console")
            reply = self.runtime.handle_control("/arena run eBPF kernel tricks", "local:console")
            self.assertIn("digested into knowledge", reply)
            self.assertIn("eBPF kernel tricks", reply)
            status = self.runtime.handle_control("/arena status", "local:console")
            self.assertIn("knowledge rows: 1", status)
            stream = self.runtime.handle_control("/arena stream 5", "local:console")
            self.assertIn("digest", stream)
        finally:
            SearchEngine.run = orig  # type: ignore[method-assign]

    def test_arena_export_writes_file(self) -> None:
        orig = SearchEngine.run
        SearchEngine.run = lambda self_, query, mode="quick", pages=3, crawl=False: dict(
            CANNED_REPORT, query=query)  # type: ignore[method-assign]
        try:
            self.runtime.handle_control("/features arena on", "local:console")
            self.runtime.handle_control("/arena run export topic", "local:console")
            reply = self.runtime.handle_control("/arena export 10", "local:console")
            self.assertIn("exported 1 training rows", reply)
            self.assertIn(".jsonl", reply)
        finally:
            SearchEngine.run = orig  # type: ignore[method-assign]

    def test_trial_save_stores_and_reports_no_live_channel(self) -> None:
        reply = self.runtime.handle_control("/trial save Prolific me@mail pw-42", "local:console")
        self.assertIn("stored the prolific trial account", reply)
        self.assertIn("me@mail", reply)
        self.assertIn("pw-42", reply)  # owner chat: the credential is the point
        listing = self.runtime.handle_control("/trial list", "local:console")
        self.assertIn("prolific: me@mail", listing)
        # and it's encrypted at rest
        raw = (self.context.settings.home_path / "trial_accounts.json").read_text("utf-8")
        self.assertNotIn("pw-42", raw)

    def test_vision_gate_strips_media(self) -> None:
        from nomorals.social.chat.base import MediaRef

        class _RecordingRouter:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def chat(self, messages, params=None, **kw):
                self.calls.append([m.content for m in messages])
                return LLMResponse(text="ok", model="fake")

        FeatureRegistry(self.context.db).set("vision", False)
        rec = _RecordingRouter()
        self.context.router = rec
        self.runtime.brain.responder.router = rec
        chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM, peer="you")
        msg = ChatMessage(chat=chat, incoming=True, text="look at this",
                          media=[MediaRef(path="/tmp/x-marked.jpg", mime="image/jpeg")])
        self.runtime._process(msg)
        joined = " ".join(c for call in rec.calls for c in call)
        self.assertNotIn("x-marked.jpg", joined, "vision is off: the brain must not see the media")


# ── 8b. arena background loop wiring in the runtime ─────────────────────────


class ArenaLoopWiringTest(unittest.TestCase):
    def test_loop_starts_when_enabled_and_stops_cleanly(self) -> None:
        from nomorals.agents.partner_runtime import PartnerRuntime

        tmp = tempfile.TemporaryDirectory(prefix="nm-aloop-")
        try:
            settings = _settings(tmp.name, **{"arena.enabled": "1"})
            context = build_context(settings, with_executor=False, with_tools=False)
            adapter = FakeAdapter("local")
            gateway = ChatGateway({"local": adapter}, db=context.db,
                                  owner_chats={"local:console"})
            runtime = PartnerRuntime(context, gateway=gateway)
            self.assertIsNone(getattr(runtime, "_arena", None))
            runtime.start()
            try:
                self.assertIsNotNone(runtime._arena)
                self.assertTrue(runtime._arena.loop_running(),
                                "NM_ARENA_ENABLED=1 must start the background loop")
            finally:
                runtime.stop()
            # the daemon thread is signalled to stop; give it a beat to exit
            end = time.time() + 3.0
            while runtime._arena.loop_running() and time.time() < end:
                time.sleep(0.02)
            self.assertFalse(runtime._arena.loop_running())
            context.close()
        finally:
            tmp.cleanup()

    def test_loop_absent_when_disabled(self) -> None:
        from nomorals.agents.partner_runtime import PartnerRuntime

        tmp = tempfile.TemporaryDirectory(prefix="nm-anloop-")
        try:
            settings = _settings(tmp.name)  # arena.enabled defaults to False
            context = build_context(settings, with_executor=False, with_tools=False)
            adapter = FakeAdapter("local")
            gateway = ChatGateway({"local": adapter}, db=context.db,
                                  owner_chats={"local:console"})
            runtime = PartnerRuntime(context, gateway=gateway)
            runtime.start()
            try:
                self.assertIsNone(runtime._arena)
            finally:
                runtime.stop()
            context.close()
        finally:
            tmp.cleanup()


# ── 9. CLI: nm trial / nm arena smoke ────────────────────────────────────────


class CliSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-cli-")
        self._old_home = os.environ.get("NM_HOME")
        os.environ["NM_HOME"] = self.tmp.name

    def tearDown(self) -> None:
        if self._old_home is None:
            os.environ.pop("NM_HOME", None)
        else:
            os.environ["NM_HOME"] = self._old_home
        self.tmp.cleanup()

    def test_trial_list_empty(self) -> None:
        from nomorals.cli import main

        rc = main(["trial", "list"])
        self.assertEqual(rc, 0)

    def test_arena_status_and_usage_errors(self) -> None:
        from nomorals.cli import main

        self.assertEqual(main(["arena", "status"]), 0)
        self.assertEqual(main(["arena", "approve"]), 2)
        self.assertEqual(main(["trial", "save"]), 2)


if __name__ == "__main__":
    unittest.main()
