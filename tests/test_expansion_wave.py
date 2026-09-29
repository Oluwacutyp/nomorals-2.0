"""The expansion wave: native hash layer, chat games, news, always-on
research, notifier, directives (direct instructions), compressor, image
lookup / reverse search, and the new control-command dispatch.

Everything runs offline: HTTP is stubbed, the LLM router is a scripted fake,
and the native layer is skipped (not failed) when no C compiler exists.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from tests.test_partner_runtime import FakeRouter, _make_context, _wait


# ── native layer availability (builds + verifies nmhash.so on first use) ──

def _native_available() -> bool:
    try:
        from nomorals.tools.native import loader

        return loader.available()
    except Exception:  # noqa: BLE001
        return False


HAS_NATIVE = _native_available()


class NativeVectorsTest(unittest.TestCase):
    """The C hashes must match the Python oracles bit-for-bit, including the
    padding boundaries (63/64/65 bytes) that break most hand-rolled MD code."""

    @classmethod
    def setUpClass(cls):
        if not HAS_NATIVE:
            raise unittest.SkipTest("no C compiler — native layer unavailable")
        from nomorals.tools.native import loader

        cls.lib = loader._lib

    def _hx(self, fn: str, out_len: int, data: bytes) -> str:
        buf = ctypes.create_string_buffer(out_len)
        getattr(self.lib, fn)(ctypes.c_char_p(data), ctypes.c_int(len(data)), buf)
        return buf.value.decode("ascii")

    def test_all_four_match_python_oracle(self) -> None:
        from nomorals.tools.hashcrack import md4

        vectors = [b"", b"a", b"abc", b"hello", b"x" * 63, b"y" * 64,
                   b"z" * 65, b"w" * 127, b"v" * 128, b"u" * 129,
                   b"q" * 1000, os.urandom(300)]
        for data in vectors:
            self.assertEqual(self._hx("nm_md5_hex", 33, data), hashlib.md5(data).hexdigest())
            self.assertEqual(self._hx("nm_sha1_hex", 41, data), hashlib.sha1(data).hexdigest())
            self.assertEqual(self._hx("nm_sha256_hex", 65, data), hashlib.sha256(data).hexdigest())
            self.assertEqual(self._hx("nm_md4_hex", 33, data), md4(data))

    def test_pinned_vectors(self) -> None:
        self.assertEqual(self._hx("nm_md5_hex", 33, b""), "d41d8cd98f00b204e9800998ecf8427e")
        self.assertEqual(self._hx("nm_md5_hex", 33, b"hunter2"), "2ab96390c7dbe3439de74d0c9b0b1767")
        self.assertEqual(self._hx("nm_sha1_hex", 41, b"abc"),
                         "a9993e364706816aba3e25717850c26c9cd0d89d")
        self.assertEqual(self._hx("nm_sha256_hex", 65, b"abc"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(self._hx("nm_md4_hex", 33, b""), "31d6cfe0d16ae931b73c59d7e0c089c0")
        self.assertEqual(self._hx("nm_md4_hex", 33, b"abc"), "a448017aaf21d8525fc10ae87aa6729d")
        # the well-known NTLM digest of "password"
        self.assertEqual(self._hx("nm_md4_hex", 33, "password".encode("utf-16-le")),
                         "8846f7eaee8fb117ad06bdd830b7586c")

    def test_fast_path_only_where_c_wins(self) -> None:
        from nomorals.tools.native import fast_hash_fn

        # hashlib is already C under Python's hood; ctypes would be ~2x slower.
        self.assertIsNone(fast_hash_fn("md5"))
        self.assertIsNone(fast_hash_fn("sha1"))
        ntlm_fn = fast_hash_fn("ntlm")
        self.assertIsNotNone(ntlm_fn)
        self.assertEqual(ntlm_fn("password"), "8846f7eaee8fb117ad06bdd830b7586c")

    def test_build_report(self) -> None:
        from nomorals.tools.native import build_report

        report = build_report()
        self.assertEqual(report["native"], "c")
        self.assertTrue(report["verified"])
        self.assertIn("ntlm", report["fast_algos"])


class EngineBackendTest(unittest.TestCase):
    """The engine takes the C fast path only where it genuinely wins."""

    def _wordlist(self, words: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(words)
        return path

    def test_md5_stays_on_hashlib(self) -> None:
        from nomorals.tools.hashcrack import Engine

        # md5("a") — the very first brute-force candidate, so the run is instant
        engine = Engine(["0cc175b9c0f1b6a831c399e269772661"], algo="md5")
        self.assertEqual(engine.hash_backend, "python")
        result = engine.run()
        self.assertEqual(result.found, {"0cc175b9c0f1b6a831c399e269772661": "a"})
        self.assertEqual(result.as_dict()["backend"], "python")

    def test_ntlm_uses_c_when_available(self) -> None:
        from nomorals.tools.hashcrack import Engine

        wl = self._wordlist("password\nabc\nletmein\n")
        engine = Engine(["8846f7eaee8fb117ad06bdd830b7586c"], algo="ntlm",
                        mode="wordlist", wordlist=wl, quiet=True)
        engine.run()
        os.unlink(wl)
        expected = "c" if HAS_NATIVE else "python"
        self.assertEqual(engine.hash_backend, expected)
        self.assertEqual(engine.found, {"8846f7eaee8fb117ad06bdd830b7586c": "password"})


# ── games ────────────────────────────────────────────────────────────────────


class GamesTest(unittest.TestCase):
    def setUp(self):
        self.context, self.tmp = _make_context()
        self.context.router = FakeRouter()

    def tearDown(self):
        self.context.close()
        self.tmp.cleanup()

    def _games(self):
        from nomorals.agents.games import GamesAgent

        return GamesAgent(self.context)

    def test_rps_full_flow(self) -> None:
        games = self._games()
        self.assertIn("rps", games.list_games())
        begin = games.begin("local:console", "rps")
        self.assertIn("rock paper scissors", begin)
        active = games.active("local:console")
        self.assertIsNotNone(active)
        self.assertEqual(active["game"], "rps")
        reply = games.play("local:console", "rock")
        self.assertIsInstance(reply, str)
        self.assertIn("menu", games.play("local:console", "spock").lower())
        self.assertIn("game over", games.quit("local:console"))
        self.assertIsNone(games.active("local:console"))

    def test_unknown_game_lists_options(self) -> None:
        games = self._games()
        self.assertIn("rps", games.begin("local:console", "chess"))

    def test_games_table_persisted(self) -> None:
        games = self._games()
        games.begin("local:console", "rps")
        row = self.context.db.query_one("SELECT * FROM game_sessions WHERE chat_key = 'local:console'")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "active")


# ── news ─────────────────────────────────────────────────────────────────────

_RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Example</title>
  <item><title>Kernel 7 ships</title><link>https://example.com/kernel-7</link>
        <description>The new kernel landed.</description></item>
  <item><title>GPUs get cheaper</title><link>https://example.com/gpus</link>
        <description>Prices dropped across the board.</description></item>
</channel></rss>"""


class NewsTest(unittest.TestCase):
    def setUp(self):
        self.context, self.tmp = _make_context()
        self.context.router = FakeRouter()

    def tearDown(self):
        self.context.close()
        self.tmp.cleanup()

    def test_parse_feed(self) -> None:
        from nomorals.agents.news import parse_feed

        items = parse_feed(_RSS_FIXTURE)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["title"], "Kernel 7 ships")
        self.assertEqual(items[0]["url"], "https://example.com/kernel-7")

    def test_run_stores_and_digests(self) -> None:
        from nomorals.agents.news import NewsAgent

        self.context.settings.news.feeds = "https://example.com/feed.xml"  # one feed

        calls: list[str] = []

        class _Notif:
            def publish(self, kind, title, body, **kw):
                calls.append(title)

        class _Client:
            def get(self, url, **kw):
                return SimpleNamespace(text=_RSS_FIXTURE, status=200,
                                        body=_RSS_FIXTURE.encode())

        agent = NewsAgent(self.context, notifier=_Notif())
        agent._client = _Client()
        report = agent.run(cap=5)
        self.assertTrue(report["ok"])
        self.assertEqual(report["items"], 2)
        self.assertIn("Kernel 7 ships", report["digest"])
        self.assertIn("GPU", report["digest"])
        self.assertEqual(len(calls), 1)
        rows = agent.recent(5)
        self.assertEqual(len(rows), 2)
        # _feeds() names a source by the URL's host
        self.assertEqual({r["source"] for r in rows}, {"example.com"})

    def test_run_dedupes_on_second_pass(self) -> None:
        from nomorals.agents.news import NewsAgent

        self.context.settings.news.feeds = "https://example.com/feed.xml"

        class _Client:
            def get(self, url, **kw):
                return SimpleNamespace(text=_RSS_FIXTURE, status=200,
                                        body=_RSS_FIXTURE.encode())

        agent = NewsAgent(self.context)
        agent._client = _Client()
        first = agent.run(cap=5)
        second = agent.run(cap=5)
        self.assertEqual(first["fresh"], 2)
        self.assertEqual(second["fresh"], 0)


# ── always-on research ───────────────────────────────────────────────────────


class ResearchTest(unittest.TestCase):
    def setUp(self):
        self.context, self.tmp = _make_context()
        self.context.router = FakeRouter(replies=["try the new queue API this week"])

    def tearDown(self):
        self.context.close()
        # under full-suite load a writer can briefly outlive close();
        # retry the rmtree instead of erroring on an empty-after-drain dir
        for _ in range(10):
            try:
                self.tmp.cleanup()
                break
            except OSError:
                time.sleep(0.15)

    def _agent(self, notifier=None):
        from nomorals.agents.researcher import ResearchAgent

        return ResearchAgent(self.context, notifier=notifier)

    def test_domain_bank(self) -> None:
        from nomorals.agents.researcher import DOMAINS

        self.assertEqual(set(DOMAINS), {"lifestyle", "tech", "cyber"})
        total = sum(len(v) for v in DOMAINS.values())
        self.assertGreaterEqual(total, 16)

    def test_run_cycle_stores_and_notifies(self) -> None:
        from unittest import mock

        from nomorals.agents import researcher as mod

        calls: list[str] = []

        class _Notif:
            def publish(self, kind, title, body, **kw):
                calls.append(title)

        agent = self._agent(notifier=_Notif())
        # A genuinely valuable report: substantive digest, three independent
        # sources, a concrete actionable suggestion — clears the quality gate.
        strong_digest = (
            "Operators cutting cold-start latency in 2026 report a 42% drop after "
            "prewarming edge caches and pinning the resolver. Teams at three "
            "companies describe the same 24h retry window as the change that paid "
            "back fastest, and two independent post-mortems flag stale DNS as the "
            "common thread across the 2025 incident data."
        )
        fake_report = {
            "summary": strong_digest,
            "pages_read": 3,
            "results": [
                {"title": "t1", "url": "https://a.example.com/post", "date": "2026"},
                {"title": "t2", "url": "https://b.example.org/writeup", "date": "2026"},
                {"title": "t3", "url": "https://c.example.net/notes", "date": "2025"},
            ],
        }
        with mock.patch.object(mod.SearchEngine, "run", return_value=fake_report), \
                mock.patch.object(agent, "_model_suggestion",
                                  return_value="Add a 24h retry window to the nightly fetcher to cut wasted cold calls."):
            result = agent.run_cycle("tech")
        self.assertTrue(result["ok"])
        self.assertEqual(result["domain"], "tech")
        self.assertIn("suggestion", result)
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(result["score"], 0.55)
        self.assertEqual(result["status"], "notified")
        rows = self.context.db.query("SELECT * FROM research_log")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["delivered"], 1)
        self.assertEqual(rows[0]["status"], "notified")
        self.assertGreater(float(rows[0]["score"]), 0.0)

    def test_low_value_idea_is_logged_but_never_pushed(self) -> None:
        from unittest import mock

        from nomorals.agents import researcher as mod

        calls: list[str] = []

        class _Notif:
            def publish(self, kind, title, body, **kw):
                calls.append(title)

        agent = self._agent(notifier=_Notif())
        # Weak report: thin digest, one source, vague suggestion → below bar.
        fake_report = {
            "summary": "A digest of the topic with a usable takeaway.",
            "pages_read": 2,
            "results": [{"title": "t", "url": "https://example.com"}],
        }
        with mock.patch.object(mod.SearchEngine, "run", return_value=fake_report), \
                mock.patch.object(agent, "_model_suggestion",
                                  return_value="Worth reading and keeping an eye on this space."):
            result = agent.run_cycle("tech")
        self.assertTrue(result["ok"])
        self.assertEqual(len(calls), 0)
        self.assertEqual(result["status"], "skipped")
        rows = self.context.db.query("SELECT * FROM research_log")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["delivered"], 0)
        self.assertEqual(rows[0]["status"], "skipped")

    def test_daily_cap_blocks_delivery(self) -> None:
        agent = self._agent()
        now = time.time()
        for _ in range(3):
            self.context.db.execute(
                "INSERT INTO research_log (id, domain, topic, digest, suggestion, sources, delivered, created_at) "
                "VALUES (?, 'tech', 't', 'd', 's', '[]', 1, ?)",
                (f"r{time.time_ns()}", now),
            )
        self.assertFalse(agent._under_cap())

    def test_background_loop_start_stop(self) -> None:
        agent = self._agent()
        self.assertTrue(agent.start_loop())
        self.assertTrue(agent.loop_running())
        agent.stop_loop()
        self.assertTrue(_wait(lambda: not agent.loop_running(), 5.0))
        # restart is clean; a SECOND start while running is the only refusal
        self.assertTrue(agent.start_loop())
        self.assertTrue(agent.loop_running())
        self.assertFalse(agent.start_loop())
        agent.stop_loop()
        self.assertTrue(_wait(lambda: not agent.loop_running(), 5.0))


# ── notifier ─────────────────────────────────────────────────────────────────


class NotifierTest(unittest.TestCase):
    def setUp(self):
        self.context, self.tmp = _make_context()
        self.context.router = FakeRouter()

    def tearDown(self):
        self.context.close()
        self.tmp.cleanup()

    def test_publish_delivers_to_live_channels(self) -> None:
        from nomorals.agents.notifier import Notifier

        sent: list[str] = []

        class _Gateway:
            def status(self):
                return {"local": {"running_in_session": True}}

            def send(self, plat, chat, text, **kw):
                sent.append(text)
                return SimpleNamespace(ok=True)

        self.context.settings.partner.owner_chats = "local:console"
        notifier = Notifier(self.context, gateway=_Gateway())
        out = notifier.publish("test", "hello owner", "body text")
        self.assertTrue(out["delivered"])
        self.assertEqual(len(sent), 1)
        self.assertIn("hello owner", sent[0])
        rows = notifier.recent(5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "test")
        self.assertTrue(rows[0]["delivered"])

    def test_publish_without_gateway_stores_undelivered(self) -> None:
        from nomorals.agents.notifier import Notifier

        notifier = Notifier(self.context, gateway=None)
        out = notifier.publish("test", "quiet", "")
        self.assertFalse(out["delivered"])
        self.assertEqual(notifier.recent(5)[0]["delivered"], 0)

    def test_feature_off_stores_but_silences(self) -> None:
        from nomorals.agents.features import FeatureRegistry
        from nomorals.agents.notifier import Notifier

        FeatureRegistry(self.context.db).set("notifier", False)
        notifier = Notifier(self.context, gateway=None)
        out = notifier.publish("test", "muted", "")
        self.assertFalse(out["delivered"])
        self.assertTrue(notifier.recent(5))


# ── directives: direct instructions to the core ─────────────────────────────


class DirectivesTest(unittest.TestCase):
    def setUp(self):
        self.context, self.tmp = _make_context()
        self.context.router = FakeRouter(replies=["done: 42"])

    def tearDown(self):
        self.context.close()
        self.tmp.cleanup()

    def _agent(self, notifier=None):
        from nomorals.agents.directives import DirectivesAgent

        return DirectivesAgent(self.context, notifier=notifier)

    def test_add_list_run(self) -> None:
        calls: list[str] = []

        class _Notif:
            def publish(self, kind, title, body, **kw):
                calls.append(title)

        agent = self._agent(notifier=_Notif())
        # plain instruction text — deliberately no download/research/code
        # keywords, so it takes the model path (scripted FakeRouter, offline)
        added = agent.add("do the thing")
        self.assertTrue(added["ok"])
        listed = agent.list(10)
        self.assertEqual(len(listed), 1)
        self.assertIn("thing", agent.format_list(listed))
        result = agent.run(added["id"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], "done: 42")
        self.assertEqual(calls[0].startswith("task done"), True)
        rows = agent.list(10)
        self.assertEqual(rows[0]["status"], "done")

    def test_run_with_nothing_pending(self) -> None:
        agent = self._agent()
        result = agent.run("")
        self.assertFalse(result["ok"])
        self.assertIn("no pending", result["error"])


# ── compressor ───────────────────────────────────────────────────────────────


class CompressTest(unittest.TestCase):
    def test_zips_a_loose_file(self) -> None:
        from nomorals.tools.compress import compress_file

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "repeat.txt"
            src.write_text("same line\n" * 2000)
            report = compress_file(src)
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["method"], "zip")
            self.assertTrue(zipfile.is_zipfile(report["path"]))
            self.assertLess(report["new_bytes"], report["original_bytes"])
            Path(report["path"]).unlink(missing_ok=True)

    def test_missing_file(self) -> None:
        from nomorals.tools.compress import compress_file

        report = compress_file("/nonexistent/file.bin")
        self.assertFalse(report["ok"])
        self.assertIn("no such file", report["error"])

    def test_already_compressed_is_honest(self) -> None:
        from nomorals.tools.compress import compress_file

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "photo.jpg"
            src.write_bytes(b"\xff\xd8\xff\xe0fakejpeg" * 100)
            report = compress_file(src)
            self.assertFalse(report["ok"])
            self.assertEqual(report["method"], "none")
            self.assertIn("already compressed", report["reason"])


# ── image lookup + reverse search ────────────────────────────────────────────


def _png(width: int, height: int) -> bytes:
    """A valid-enough PNG header for format sniffing + dimension parsing."""
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
            + struct.pack(">II", width, height) + b"\x00\x08\x06\x00\x00\x00"
            + b"\x00" * 32)


class ImagedbTest(unittest.TestCase):
    def test_primitives(self) -> None:
        from nomorals.tools import imagedb

        self.assertEqual(imagedb.sniff_format(b"\x89PNG\r\n\x1a\nxxxx"), "png")
        self.assertEqual(imagedb.sniff_format(b"\xff\xd8\xff\xe0xxxx"), "jpeg")
        self.assertEqual(imagedb.sniff_format(b"GIF89a" + b"xxxx"), "gif")
        self.assertEqual(imagedb.sniff_format(b"RIFF" + b"\x00" * 4 + b"WEBP"), "webp")
        self.assertEqual(imagedb.sniff_format(b"????????"), "unknown")
        self.assertEqual(imagedb.png_dimensions(_png(320, 240)), (320, 240))
        # no PIL in this environment -> dhash fails open
        self.assertIsNone(imagedb.dhash("anything"))

    def test_lookup_and_seen_before(self) -> None:
        from nomorals.agents.context import build_context
        from nomorals.core.config import load_settings

        # a context WITH the tool registry so image_lookup is registered
        tmp2 = tempfile.TemporaryDirectory(prefix="nm-img-")
        settings = load_settings(overrides={"home": tmp2.name})
        ctx = build_context(settings, with_executor=False, with_tools=True)
        try:
            path = Path(tmp2.name) / "pic.png"
            path.write_bytes(_png(64, 48))
            out = ctx.tools.call("image_lookup", path=str(path))
            self.assertTrue(out.ok)
            data = out.value
            self.assertTrue(data["ok"])
            self.assertEqual(data["format"], "png")
            self.assertEqual((data["width"], data["height"]), (64, 48))
            self.assertFalse(data["seen_before"])
            out2 = ctx.tools.call("image_lookup", path=str(path))
            self.assertTrue(out2.value["seen_before"])
        finally:
            ctx.close()
            tmp2.cleanup()

    def test_reverse_search_local_file(self) -> None:
        from nomorals.agents.context import build_context
        from nomorals.core.config import load_settings

        tmp2 = tempfile.TemporaryDirectory(prefix="nm-rev-")
        settings = load_settings(overrides={"home": tmp2.name})
        ctx = build_context(settings, with_executor=False, with_tools=True)
        try:
            path = Path(tmp2.name) / "pic.png"
            path.write_bytes(_png(10, 10))
            out = ctx.tools.call("reverse_image_search", path=str(path))
            self.assertTrue(out.ok)
            data = out.value
            self.assertTrue(data["ok"])
            self.assertIn("lookup_links", data)
            self.assertIn("note", data)  # local file: no public URL
        finally:
            ctx.close()
            tmp2.cleanup()


# ── control parsing + runtime dispatch ───────────────────────────────────────


class ControlParseTest(unittest.TestCase):
    def test_new_commands_parse(self) -> None:
        from nomorals.social.chat.control import parse_control

        cases = {
            "/game rps": "game",
            "/game quit": "game",
            "/news run": "news",
            "/news status": "news",
            "/research run tech": "research",
            "/research status": "research",
            "/code write a tiny script": "code",
            "/task add buy milk": "task",
            "/task run": "task",
            "/notify 5": "notify",
            "/image /tmp/pic.png": "image",
            "/lens https://example.com/pic.png": "lens",
        }
        for text, kind in cases.items():
            cmd = parse_control(text)
            self.assertIsNotNone(cmd, text)
            self.assertEqual(cmd.kind, kind, text)

    def test_arities(self) -> None:
        from nomorals.social.chat.control import parse_control

        self.assertIsNone(parse_control("hello"))
        self.assertEqual(parse_control("/image").kind, "error")       # needs 1
        self.assertEqual(parse_control("/notify 1 2 3").kind, "error")  # max 1
        # unknown slash → not a command at all (she answers it as conversation)
        self.assertIsNone(parse_control("/notacommand"))


class RuntimeDispatchTest(unittest.TestCase):
    def setUp(self):
        from tests.test_partner_runtime import FakeAdapter
        from nomorals.social.chat.gateway import ChatGateway

        self.context, self.tmp = _make_context()
        self.context.router = FakeRouter(replies=["done: 42"])
        self.adapter = FakeAdapter("local")
        self.gateway = ChatGateway({"local": self.adapter}, db=self.context.db)
        from nomorals.agents.partner_runtime import PartnerRuntime

        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)

    def tearDown(self):
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_notify_empty(self) -> None:
        self.assertIn("no notifications", self.runtime.handle_control("/notify", "local:console"))

    def test_news_off_by_default(self) -> None:
        self.assertIn("news is off", self.runtime.handle_control("/news run", "local:console"))

    def test_research_off_by_default(self) -> None:
        self.assertIn("research is off", self.runtime.handle_control("/research status", "local:console"))

    def test_game_command_starts_rps(self) -> None:
        reply = self.runtime.handle_control("/game rps", "local:console")
        self.assertIn("rock paper scissors", reply)
        reply2 = self.runtime.handle_control("/game", "local:console")
        self.assertIn("live game", reply2)
        reply3 = self.runtime.handle_control("/game quit", "local:console")
        self.assertIn("game over", reply3)

    def test_image_needs_a_path(self) -> None:
        # arity is enforced by the parser before the handler even sees it
        self.assertIn("needs 1 argument", self.runtime.handle_control("/image", "local:console"))
        self.assertIn("needs 1 argument", self.runtime.handle_control("/lens", "local:console"))

    def test_task_add_list_run(self) -> None:
        # "do the thing" = no download/research/code keywords → model path
        added = self.runtime.handle_control("/task add do the thing", "local:console")
        self.assertIn("queued", added)
        listed = self.runtime.handle_control("/task list", "local:console")
        self.assertIn("thing", listed)
        done = self.runtime.handle_control("/task run", "local:console")
        self.assertEqual(done, "")  # long output went to the gateway

    def test_game_moves_routed_not_slashed(self) -> None:
        self.runtime.handle_control("/game rps", "local:console")
        move_reply = self.runtime._route_game_move("local:console", "rock")
        self.assertIsInstance(move_reply, str)
        self.assertIsNone(self.runtime._route_game_move("local:console", "/status"))

    def test_arena_bare_topic_shorthand_runs_cycle(self) -> None:
        # /arena hacking == /arena run hacking — the phone typed it this
        # way and got a "usage:" reply.
        from nomorals.agents.features import FeatureRegistry

        FeatureRegistry(self.context.db).set("arena", True)
        reply = self.runtime.handle_control("/arena hacking", "local:console")
        self.assertNotIn("usage:", reply)
        # No tool registry in this context → the cycle fails HONESTLY
        # (research needs tools), it must not crash or print usage.
        self.assertIn("arena cycle failed", reply)

    def test_arena_new_verbs(self) -> None:
        # stats / schedule / topic / loop / digest / builds — the expansion
        # the owner asked for.
        self.assertIn("arena stats", self.runtime.handle_control("/arena stats", "local:console"))
        self.assertIn("loop interval set to 2h",
                      self.runtime.handle_control("/arena schedule 2", "local:console"))
        self.assertIn("2h", self.runtime.handle_control("/arena schedule", "local:console"))
        self.assertIn("topic bank now has 1",
                      self.runtime.handle_control("/arena topic add on-device inference",
                                                  "local:console"))
        self.assertIn("on-device inference",
                      self.runtime.handle_control("/arena topic", "local:console"))
        self.assertIn("loop: idle",
                      self.runtime.handle_control("/arena loop status", "local:console"))
        # Loop gates: feature first, then power mode — honest messages either way.
        self.assertIn("feature off",
                      self.runtime.handle_control("/arena loop on", "local:console"))
        from nomorals.agents.features import FeatureRegistry

        FeatureRegistry(self.context.db).set("arena", True)
        self.assertIn("power mode is locked",
                      self.runtime.handle_control("/arena loop on", "local:console"))
        self.assertIn("no digests yet",
                      self.runtime.handle_control("/arena digest", "local:console"))
        self.assertIn("no arena builds",
                      self.runtime.handle_control("/arena builds", "local:console"))
        # stream with a kind filter
        self.assertIn("arena stream is empty",
                      self.runtime.handle_control("/arena stream topic", "local:console"))


if __name__ == "__main__":
    unittest.main()
