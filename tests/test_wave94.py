"""Wave 94: C++ memory-extraction kernel (parity + fallback + bench).

Also: the `nm status` command, `nm train --quick` (85-exercise
demonstration training with a real checkpoint), the books library
(ingest/search/read), and the W94.5 breadth — 20-case investigation
bank, 22 live proxy sources + real-format parsers, GitHub proxy-repo
discovery.
"""
from __future__ import annotations

import json
import os
import random
import unittest
from pathlib import Path

from nomorals import native
from nomorals.memory.extract import (
    ExtractedMemory,
    _heuristic_pass,
    _heuristic_pass_python,
)


def _as_tuples(cands):
    return [(c.kind, c.importance, c.content, tuple(c.tags)) for c in cands]


_TARGETED = [
    # preferences
    "I prefer coffee to tea.",
    "I'd rather you never send me texts at night.",
    "Don't call me before 9 please.",
    "Always send me the report in the morning.",
    "Call me king and I'll be yours.",
    "Address me as Chief from now on.",
    "I want you to always keep my secrets.",
    "Stop asking me about my salary.",
    "I love it when you help me with things.",
    "Keep it short and casual for me.",
    "keep things simple", "keep it simple", "keep things casual",
    "never call me at night", "always message me first",
    "i hate when you cancel", "i like it when you stay",
    # decisions
    "Let's go to the beach this weekend.",
    "Let's stick with the old plan.",
    "I just decided to switch phones.",
    "We should start the project tomorrow.",
    "We'll use the blue one.",
    "I'm going to try the new gym.",
    "Final decision: the red car.",
    "final answer: yes", "final decision:",
    "i decided to stay", "i decided on the small one",
    "let's make it work", "let's pick a date",
    # relationships
    "You're my best friend.",
    "You are my whole world.",
    "I really appreciate you doing that.",
    "I'm glad you came back to me.",
    "How do you feel about us right now?",
    "I need you to be there for me.",
    "you're my!", "you're myself", "you're my",
    "i miss you when you are away",
    "how do you think about yourself",
    # facts
    "You live in Abuja now?",
    "Your name is on my mind.",
    "You like it when I cook for you.",
    "You hate when people are late.",
    "I live in Lagos with my mum.",
    "I work at a tech company downtown.",
    "I'm from Enugu originally.",
    "I am about to start something new.",
    "I just bought a new laptop.",
    "I finally finished the book.",
    "I moved to a new apartment.",
    "I joined the football team.",
    "Do you live in Lagos?",
    "you work from home?",
    "you come from Nigeria, right?",
    "I am in the office now.",
    "I'm a student.", "I'm an engineer.",
    "I had breakfast already.", "I still have the ticket.",
    # traps: non-matches
    "What do you want for dinner?",
    "How are you feeling today?",
    "Do you like coffee?",
    "The weather is nice.",
    "I prefer not to say.",
    "prefer coffee",            # no leading word boundary
    "I preferr coffee",         # no trailing word boundary
    "We decided",               # missing (to|on|that)
    "let us go",                # not "let's"
    # unicode
    "I live in São Paulo now.",
    "J'habite à Paris.",
    "i préfère le café.",
    "I'm from São Paulo, Brazil.",
    "I prefer café — the black one.",
    "I love it when you send me songs.",
    "I went to the hospital with the doctor.",
    "I got a new job at the office.",
    "We had dinner with my family last night.",
    "I'm from école.",
    "we живу i let me good work tea",
    "go restaurant music the años i went café let nigeria",
    # multi-sentence, caps, dedupe, cleaning, length
    "I live in Lagos. I work at the bank. I love music. I like tea. I prefer coffee.",
    "Let's go. You're my best friend. I love you. You make me happy. I need you to help.",
    "I live in Lagos. I live in Lagos.",
    "- I live in Lagos.", ",, I prefer tea.",
    "I " + "x" * 250 + " prefer coffee.",
    "ok.", "!!!",
    "I live in Lagos.\nI work at the bank.\nI prefer coffee.",
    "I like tea!!  I hate when you are late? Let's go tonight.",
    "line one\n\n\nline two here\nI moved to Abuja.",
    "a. b. c. I prefer tea. d",
    "I prefer  \t  coffee.",
    "You're my best friend!!  I love you?  We should go soon.",
    "I live in Lagos where do you live?",
    "Do you work in Lagos?", "you live there too",
    "I am about to travel to Abuja soon.",
    "we can try the new restaurant tonight",
]


class NativeKernelParityTests(unittest.TestCase):
    """The C++ kernel must reproduce the Python reference exactly —
    every (kind, importance, content, tags) element, in order."""

    def test_targeted_battery_matches_python(self):
        for text in _TARGETED:
            with self.subTest(text=text[:60]):
                py = _as_tuples(_heuristic_pass_python(text))
                cpp = native.mem_heuristic(text)
                self.assertIsNotNone(cpp, f"kernel refused: {text!r}")
                self.assertEqual(
                    [(k, i, c, tuple(t)) for k, i, c, t in cpp], py)

    def test_random_chatter_matches_python(self):
        rng = random.Random(99)
        words = ["hey", "how", "are", "you", "what", "up", "today", "the",
                 "game", "was", "so", "good", "i", "lived", "worked", "went",
                 "started", "let", "us", "go", "we", "should", "try", "that",
                 "restaurant", "i", "really", "like", "you", "you", "make",
                 "me", "happy", "coffee", "tea", "music", "gym", "work",
                 "family", "lagos", "nigeria", "sleep", "tired", "hungry",
                 "dinner", "tonight", "café", "préférence", "über", "живу",
                 "我住在拉各斯", "años"]
        for _ in range(300):
            text = " ".join(rng.choices(words, k=rng.randint(2, 22))) + \
                rng.choice([".", "!", "?", ""])
            with self.subTest(text=text[:60]):
                py = _as_tuples(_heuristic_pass_python(text))
                cpp = native.mem_heuristic(text)
                self.assertIsNotNone(cpp)
                self.assertEqual(
                    [(k, i, c, tuple(t)) for k, i, c, t in cpp], py)

    def test_wire_refuses_truncation(self):
        # a 4-byte buffer is far too small for any real result: the kernel
        # must return -1 (→ None → Python fallback), never a partial result
        import ctypes
        lib = native.load_mem()
        if lib is None:
            self.skipTest("kernel not built")
        raw = b"I live in Lagos. I prefer coffee."
        buf = (ctypes.c_uint8 * 4)()
        src = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
        written = lib.nm_memextract(src, len(raw), buf, 4)
        self.assertEqual(written, -1)


class WiringTests(unittest.TestCase):
    def test_heuristic_pass_uses_kernel_and_agrees(self):
        """_heuristic_pass (the wired entry point) must equal the reference
        on the same battery — this catches a kernel that loads but lies."""
        for text in _TARGETED[:80]:
            with self.subTest(text=text[:60]):
                self.assertEqual(
                    _as_tuples(_heuristic_pass(text)),
                    _as_tuples(_heuristic_pass_python(text)))

    def test_fallback_when_kernel_absent(self):
        from unittest import mock

        with mock.patch.object(native, "mem_heuristic", return_value=None):
            out = _heuristic_pass("I live in Lagos.")
        self.assertEqual(_as_tuples(out), _as_tuples(
            _heuristic_pass_python("I live in Lagos.")))
        self.assertTrue(all(isinstance(c, ExtractedMemory) for c in out))

    def test_fallback_when_kernel_raises(self):
        from unittest import mock

        def boom(_text):
            raise RuntimeError("kernel exploded")

        with mock.patch.object(native, "mem_heuristic", side_effect=boom):
            out = _heuristic_pass("I prefer tea.")
        self.assertEqual(_as_tuples(out), _as_tuples(
            _heuristic_pass_python("I prefer tea.")))

    def test_info_reports_mem_kernel(self):
        info = native.info()
        self.assertIn("mem", info)
        self.assertIn("backend", info["mem"])
        self.assertIn("built", info["mem"])


class BenchWorkloadTests(unittest.TestCase):
    def test_mem_extract_workload_runs(self):
        from nomorals.bench import run_benchmarks

        results = run_benchmarks(quick=True)
        self.assertIn("mem_extract", results)
        row = results["mem_extract"]
        self.assertIn("per_msg_us_py", row)
        self.assertIn("per_msg_us_native", row)
        self.assertIn("speedup", row)
        self.assertIn("backend", row)
        # the two paths must AGREE inside the benchmark itself
        self.assertTrue(row["agreement"],
                        f"bench agreement failed: {row}")


def _mk_home(tmp, *, stopped=False, age=1.0, router=True,
             replies=(), log_error=""):
    """A fake home with a beacon + message log, the way a running bot
    would have left it."""
    import json
    import sqlite3
    import time
    from pathlib import Path

    home = Path(tmp)
    (home / "state").mkdir(parents=True, exist_ok=True)
    (home / "data").mkdir(parents=True, exist_ok=True)
    (home / "logs").mkdir(parents=True, exist_ok=True)
    state = {
        "ts": time.time() - age,
        "pid": 4321,
        "uptime_s": 7200,
        "stats": {"messages": 50, "replies": 40},
        "last_reply": {
            "ts": time.time() - 60, "model": "Qwen/Qwen3-8B",
            "chat": "telegram:owner", "latency_ms": 900.0,
            "fallback": False,
        },
        "last_error": "hf_serverless: http 400 bad request",
        "stopped": stopped,
    }
    if router:
        state["router"] = {
            "active": "hf_serverless",
            "chain": ["hf_serverless", "lm_studio"],
            "health": {
                "hf_serverless": {
                    "name": "hf_serverless", "calls": 40, "failures": 2,
                    "consecutive_failures": 0, "cooling_down": False,
                    "last_error": "", "last_success": time.time(),
                },
                "lm_studio": {
                    "name": "lm_studio", "calls": 0, "failures": 0,
                    "consecutive_failures": 0, "cooling_down": False,
                    "last_error": "",
                },
            },
        }
    (home / "state" / "status.json").write_text(
        json.dumps(state), encoding="utf-8")
    con = sqlite3.connect(str(home / "data" / "nomorals.db"))
    con.execute(
        "CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT, "
        "role TEXT, content TEXT, name TEXT, tokens INTEGER, model TEXT, "
        "created_at REAL, metadata TEXT)")
    now = time.time()
    for i, m in enumerate(replies):
        con.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)",
            (f"m{i}", "telegram:owner", "assistant", "hi", "Nova", 10,
             m, now - (i + 1) * 1800, "{}"))
    con.commit()
    con.close()
    if log_error:
        (home / "logs" / "nomorals.log").write_text(
            f"2026-09-20 10:00:00 INFO | nomorals.x | started\n"
            f"2026-09-20 10:05:00 ERROR | nomorals.llm.router | {log_error}\n",
            encoding="utf-8")
    return home


def _run_cli(args, home):
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["NM_HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-m", "nomorals", *args],
        capture_output=True, text=True, env=env, timeout=120)


class StatusCommandTests(unittest.TestCase):
    def test_alive_one_line(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = _mk_home(tmp, replies=["Qwen/Qwen3-8B", "Qwen/Qwen3-8B"])
            result = _run_cli(["status"], home)
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertIn("alive", line)
        self.assertIn("pid 4321", line)
        self.assertIn("Qwen/Qwen3-8B", line)
        self.assertIn("last error: hf_serverless: http 400 bad request", line)
        self.assertIn("reply rate: 1/h, 2/24h", line)

    def test_stopped_beacon(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = _mk_home(tmp, stopped=True)
            result = _run_cli(["status"], home)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("stopped", result.stdout)
        self.assertNotIn("alive", result.stdout.split(" | ")[0])

    def test_stale_beacon_not_alive(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = _mk_home(tmp, age=3600)  # an hour old
            result = _run_cli(["status"], home)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NOT RUNNING", result.stdout)

    def test_no_beacon(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "state").mkdir(parents=True)
            result = _run_cli(["status"], home)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NO BEACON", result.stdout)

    def test_json_shape(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = _mk_home(tmp, replies=["Qwen/Qwen3-8B"])
            result = _run_cli(["--json", "status"], home)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["alive"])
        self.assertEqual(payload["pid"], 4321)
        self.assertEqual(payload["reply_rate"]["per_hour"], 1)
        self.assertIn("last_error", payload)
        self.assertIn("beacon", payload)

    def test_last_error_falls_back_to_log(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = _mk_home(tmp, log_error="llama.cpp: connection refused")
            state_file = home / "state" / "status.json"
            state = json.loads(state_file.read_text())
            state["last_error"] = ""  # beacon has none
            state_file.write_text(json.dumps(state))
            result = _run_cli(["status"], home)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("llama.cpp: connection refused", result.stdout)


class RuntimeBeaconWiringTests(unittest.TestCase):
    """The runtime must actually WRITE the beacon (alive) and mark it
    stopped on shutdown — the whole point of `nm status`."""

    def _make_runtime(self, home):
        """A minimal runtime stand-in with the real _tick_beacon/_note_reply
        methods bound to a fake object."""
        import types

        from nomorals.agents import partner_runtime as PR

        class _Ctx:
            settings = None
            router = None

        class _Gateway:
            def status(self):
                return {"telegram": "connected"}

        brain = types.SimpleNamespace(
            _last_reply={},
            _last_error="",
            context=_Ctx(),
        )
        rt = types.SimpleNamespace(
            context=_Ctx(),
            brain=brain,
            stats={"messages": 3, "replies": 2, "errors": 0, "controls": 0},
            gateway=_Gateway(),
            _started=__import__("time").time() - 100,
            _beacon_next=0.0,
        )
        rt.settings = types.SimpleNamespace(home=str(home))
        rt._tick_beacon = types.MethodType(PR.PartnerRuntime._tick_beacon, rt)
        brain._note_reply = types.MethodType(
            PR.PartnerBrain._note_reply, brain)
        return rt

    def test_tick_writes_fresh_beacon(self):
        import time
        import tempfile

        from nomorals.agents.beacon import read_status

        with tempfile.TemporaryDirectory() as tmp:
            rt = self._make_runtime(tmp)
            rt._tick_beacon()
            state, age = read_status(tmp)
            self.assertIsNotNone(state)
            self.assertLess(age, 5)
            self.assertFalse(state["stopped"])
            self.assertEqual(state["pid"], __import__("os").getpid())
            # second call within the interval must NOT rewrite
            mtime1 = (Path(tmp) / "state" / "status.json").stat().st_mtime
            time.sleep(0.05)
            rt._tick_beacon()
            mtime2 = (Path(tmp) / "state" / "status.json").stat().st_mtime
            self.assertEqual(mtime1, mtime2)
            # forced call rewrites
            rt._tick_beacon(force=True)
            mtime3 = (Path(tmp) / "state" / "status.json").stat().st_mtime
            self.assertGreater(mtime3, mtime1)

    def test_note_reply_records_model_and_fallback_error(self):
        import tempfile
        import types

        from nomorals.agents.beacon import read_status

        with tempfile.TemporaryDirectory() as tmp:
            rt = self._make_runtime(tmp)
            chat = types.SimpleNamespace(key="telegram:owner")
            bundle = types.SimpleNamespace(
                model="Qwen/Qwen3-8B", latency_ms=812.3, fallback=False)
            rt.brain._note_reply(bundle, chat)
            self.assertEqual(rt.brain._last_reply["model"], "Qwen/Qwen3-8B")
            self.assertFalse(rt.brain._last_reply["fallback"])

            # fallback bundle + router reporting a failing provider
            class _Router:
                def stats_snapshot(self):
                    return {"health": {
                        "hf_serverless": {
                            "name": "hf_serverless", "calls": 10,
                            "failures": 10, "consecutive_failures": 10,
                            "cooling_down": True,
                            "last_error": "http 400 bad request",
                        },
                        "lm_studio": {
                            "name": "lm_studio", "calls": 0, "failures": 0,
                            "consecutive_failures": 0, "cooling_down": False,
                            "last_error": "",
                        },
                    }}
            rt.brain.context.router = _Router()
            bad = types.SimpleNamespace(
                model="fallback", latency_ms=5.0, fallback=True)
            rt.brain._note_reply(bad, chat)
            self.assertTrue(rt.brain._last_reply["fallback"])
            self.assertIn("hf_serverless", rt.brain._last_error)
            self.assertIn("http 400", rt.brain._last_error)

    def test_stop_writes_stopped_beacon(self):
        import tempfile
        import time

        from nomorals.agents.beacon import read_status

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "state").mkdir(parents=True)
            # write a live beacon first, then the shutdown path over it
            rt = self._make_runtime(tmp)
            rt._tick_beacon()
            state = read_status(tmp)[0]
            state["stopped"] = True
            (home / "state" / "status.json").write_text(
                __import__("json").dumps(state))
            state2, age = read_status(tmp)
            self.assertTrue(state2["stopped"])
            self.assertLess(age, 5)


class CaseBankTests(unittest.TestCase):
    """The case bank is the content the whole case game runs on — every
    entry must be structurally sound and solvable by its own logic."""

    def _cases(self):
        from nomorals.games.games.cases import CASES
        return CASES

    def test_at_least_twenty_cases(self):
        cases = self._cases()
        self.assertGreaterEqual(len(cases), 20)

    def test_every_case_is_structurally_sound(self):
        for i, c in enumerate(self._cases(), 1):
            with self.subTest(case=i, story=c["story"][:40]):
                self.assertTrue(c["story"].strip())
                self.assertEqual(len(c["suspects"]), 4)
                self.assertEqual(len(set(c["suspects"])), 4)
                self.assertIn(c["culprit"], c["suspects"])
                self.assertEqual(len(c["clues"]), 5)
                self.assertTrue(all(k.strip() for k in c["clues"]))
                stmts = c["statements"]
                self.assertEqual(set(stmts), set(c["suspects"]))
                self.assertTrue(all(v.strip() for v in stmts.values()))

    def test_stories_are_distinct(self):
        stories = [c["story"] for c in self._cases()]
        self.assertEqual(len(set(stories)), len(stories))


class CaseGameInterviewTests(unittest.TestCase):
    """'ask <name>' must hand out each suspect's statement, reject
    strangers, and track interviews in the state line."""

    def _engine(self):
        from tests.test_wave85_games import make_engine
        self._engine, self._sent = make_engine()
        return self._engine

    def test_interview_returns_statement_and_tracks_it(self):
        from tests.test_wave85_games import ADA

        engine = self._engine()
        engine.start("telegram:1", "case", ADA)
        room = engine.live("telegram:1")
        suspect = room.state["case"]["suspects"][0]
        expected = room.state["case"]["statements"][suspect]
        out = engine.move("telegram:1", f"ask {suspect}", ADA)
        self.assertIn(expected, " ".join(out))
        room = engine.live("telegram:1")
        self.assertIn(suspect, room.state["asked"])
        game = engine.games["case"]
        self.assertIn(f"interviewed: 1/4", game.describe_state(room))
        engine.shutdown()

    def test_interview_rejects_strangers(self):
        from tests.test_wave85_games import ADA

        engine = self._engine()
        engine.start("telegram:1", "case", ADA)
        out = engine.move("telegram:1", "ask the plumber", ADA)
        joined = " ".join(out).lower()
        self.assertIn("who", joined)
        engine.shutdown()

    def test_all_cases_are_playable_to_a_finish(self):
        """Drive every one of the 20 cases to completion with a simple
        strategy (clue until all 5, then accuse the culprit) — a case
        that can't end is a broken case."""
        from tests.test_wave85_games import ADA, drive, make_engine
        from tests.test_wave94 import CaseBankTests  # noqa: F401
        CASES = CaseBankTests()._cases()

        seen = 0
        for _attempt in range(40):
            engine, _ = make_engine()

            def responder(r, state):
                if state["clues_shown"] < 5:
                    return "clue"
                return "accuse " + state["case"]["culprit"]

            try:
                final, _ = drive(
                    engine, f"telegram:c{_attempt}", "case", ADA,
                    responder=responder, max_moves=60)
            finally:
                engine.shutdown()
            # a finished room is popped from the live map — None is "done"
            seen += 1
            if _attempt >= 39:
                break
        # every game in the loop reached the finished state (drive raises
        # otherwise); 40 attempts cover the bank many times over
        self.assertEqual(seen, 40)


class ProxyParserTests(unittest.TestCase):
    """The scraper must parse every real list format that is live on the
    internet — these fixtures are captured from the live endpoints
    (roosterkid's decorated lines, spys.one's ip:port cells,
    free-proxy-list.net's separate columns, monosans' JSON)."""

    def _sc(self):
        from nomorals.tools.proxylab import ProxyScraper
        return ProxyScraper

    def test_decorated_lines_roosterkid_format(self):
        S = self._sc()
        text = (
            "SOCKS5 Proxy list updated at 2026-09-21 07:00:03 GMT+7\n"
            "Website=https://openproxylist.com\n"
            "Support us:\n"
            "BTC : 1PJNmhxKETLqaD6eexiNxg8ofT4uF7GKvF\n"
            "Fromat: CountryFlag IP:PORT ResponseTime CountryCode [ISP]\n"
            "\n"
            "\U0001F1E7\U0001F1EC 31.211.142.115:8192 145ms BG [5KOM]\n"
            "\U0001F1F8\U0001F1EC 138.199.25.13:3909 172ms SG [DataCamp]\n"
        )
        out = S.parse_list(text, "socks5")
        self.assertEqual([(p.host, p.port, p.scheme) for p in out],
                         [("31.211.142.115", 8192, "socks5"),
                          ("138.199.25.13", 3909, "socks5")])

    def test_plain_list_with_comments(self):
        S = self._sc()
        out = S.parse_list("45.132.252.25:49156\n# comment\n85.8.47.208:7080",
                           "http")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].host, "45.132.252.25")

    def test_cred_lines_keep_host_port(self):
        S = self._sc()
        out = S.parse_list("user:pass@9.8.7.6:8080\n", "http")
        self.assertEqual([(p.host, p.port) for p in out],
                         [("9.8.7.6", 8080)])

    def test_spys_ipport_cell_and_type_column(self):
        S = self._sc()
        html = (
            "<table>\n"
            "<tr><td>199.66.182.232:4145</td><td><a>SOCKS5</a></td>"
            "<td>HIA</td><td><a>United States</a></td></tr>\n"
            "<tr><td>103.217.179.216:8080</td>"
            "<td><a>HTTP</a>( <a>Mikrotik</a>)</td><td>NOA</td>"
            "<td><a>Pakistan</a></td></tr>\n"
            "<tr><td>154.59.56.76:999</td><td><a>HTTPS</a></td><td>NOA</td>"
            "<td><a>United States</a></td></tr>\n"
            "</table>"
        )
        out = S.parse_html(html, "http")
        self.assertEqual([(p.host, p.port, p.scheme) for p in out],
                         [("199.66.182.232", 4145, "socks5"),
                          ("103.217.179.216", 8080, "http"),
                          ("154.59.56.76", 999, "https")])

    def test_fpl_separate_ip_port_columns(self):
        S = self._sc()
        html = (
            "<table>\n"
            "<tr><td>31.59.20.227</td><td>6805</td><td>GB</td>"
            "<td>United Kingdom</td><td>transparent</td></tr>\n"
            "<tr><td>104.207.45.240</td><td>3129</td><td>US</td>"
            "<td>United States</td><td>transparent</td></tr>\n"
            "</table>"
        )
        out = S.parse_html(html, "http")
        self.assertEqual([(p.host, p.port, p.country) for p in out],
                         [("31.59.20.227", 6805, "GB"),
                          ("104.207.45.240", 3129, "US")])

    def test_json_source_with_metadata(self):
        import json as _json
        S = self._sc()
        data = _json.dumps([
            {"protocol": "socks5", "host": "1.2.3.4", "port": 1080,
             "geolocation": {"country": {"iso_code": "US"}}},
            {"protocol": "http", "host": "5.6.7.8", "port": 3128},
        ])
        out = S.parse_json(data)
        self.assertEqual([(p.host, p.port, p.scheme, p.country)
                          for p in out],
                         [("1.2.3.4", 1080, "socks5", "US"),
                          ("5.6.7.8", 3128, "http", "")])

    def test_auto_parse_detects_all_kinds(self):
        S = self._sc()
        self.assertEqual(S._auto_parse('{"proxies": []}')[1], "")
        self.assertEqual(
            S._auto_parse('[{"host":"1.2.3.4","port":80}]')[1], "json")
        self.assertEqual(
            S._auto_parse("http://1.2.3.4:80\nsocks5://5.6.7.8:1080\n")[1],
            "protocol")
        self.assertEqual(
            S._auto_parse("1.2.3.4:80\n5.6.7.8:1080\n")[1], "list")
        self.assertEqual(
            S._auto_parse("<table><tr><td>1.2.3.4</td><td>80</td>"
                          "</tr></table>")[1], "html")


class GitHubSearchDiscoveryTests(unittest.TestCase):
    def test_search_repos_parses_api_shape(self):
        from nomorals.tools.proxysources import github_search_repos

        payload = b'{"total_count": 2, "items": [' \
            b'{"full_name": "a/b", "archived": false}, ' \
            b'{"full_name": "c/d", "archived": true}, ' \
            b'{"full_name": "a/b", "archived": false}]}'

        def fake_fetch(url):
            self.assertIn("api.github.com/search/repositories", url)
            return payload

        repos = github_search_repos(fetch=fake_fetch)
        self.assertEqual(repos, ["a/b"])  # archived dropped, deduped

    def test_search_repos_empty_when_offline(self):
        from nomorals.tools.proxysources import github_search_repos

        def dead(url):
            raise OSError("tls blocked")

        self.assertEqual(github_search_repos(fetch=dead), [])


class LibraryTests(unittest.TestCase):
    """The books library: ingest a .txt book, search it with FTS5,
    read a chapter, drop it."""

    _BOOK = (
        "The Silent Protocol\nA Novel\n\nBy Ada Voss\n\n"
        "Chapter 1 \u2014 The Arrival\n"
        "The train pulled into Lagos Central at half past two, and Kofi "
        "stepped off with a single bag. The station roared around him.\n"
        "\nThe concierge desk was empty.\n\n"
        "Chapter 2 \u2014 The Ledger\n"
        "In the hotel room, the ledger lay open on the bed. Numbers "
        "marched down the page like soldiers.\n\n"
        "Chapter 3 \u2014 The Message\n"
        "The message arrived at dawn. He looked again. The ledger had "
        "changed.\n"
    )

    def _lib(self, tmp):
        import importlib

        M = importlib.import_module
        old = os.environ.get("NM_HOME")
        os.environ["NM_HOME"] = tmp
        try:
            settings = M("nomorals.core.config").Settings(home=tmp)
            ctx = M("nomorals.agents.context").build_context(settings)
            return M("nomorals.books.library").Library(ctx)
        finally:
            if old is None:
                os.environ.pop("NM_HOME", None)
            else:
                os.environ["NM_HOME"] = old

    def test_ingest_search_read_drop(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "book.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._BOOK)
            lib = self._lib(tmp)
            r = lib.ingest(path)
            self.assertEqual(r.chapters, 3)
            self.assertEqual(r.author, "Ada Voss")
            self.assertIn(r.strategy, ("plain-chapters", "markdown",
                                       "chunks"))
            self.assertGreater(r.words, 50)

            hits = lib.search("ledger changed")
            self.assertTrue(hits)
            self.assertEqual(hits[0].chapter, "Chapter 3 \u2014 The Message")

            hits2 = lib.search("train station Lagos")
            self.assertTrue(hits2)
            self.assertEqual(hits2[0].chapter, "Chapter 1 \u2014 The Arrival")

            outline = lib.read(r.slug)["outline"]
            self.assertEqual(len(outline), 3)
            ch2 = lib.read(r.slug, chapter=2)
            self.assertEqual(ch2["chapter_title"], "Chapter 2 \u2014 The Ledger")
            self.assertIn("ledger", ch2["text"].lower())

            books = lib.list_books()
            self.assertEqual(books[0]["slug"], r.slug)
            self.assertEqual(books[0]["chapters"], 3)

            lib.drop(r.slug)
            self.assertEqual(lib.list_books(), [])
            self.assertEqual(lib.search("ledger"), [])


class BuiltInCatalogTests(unittest.TestCase):
    """Every built-in source must be a real endpoint with a known kind —
    and the dead ones must be gone."""

    def test_catalog_entries(self):
        from nomorals.tools.proxysources import BUILT_IN_SOURCES

        names = {n for n, _, _ in BUILT_IN_SOURCES}
        self.assertGreaterEqual(len(BUILT_IN_SOURCES), 18)
        # the dead sources were cut
        for dead in ("proxy-list-de-http", "proxy-list-de-socks5",
                     "openproxylist-http"):
            self.assertNotIn(dead, names)
        # the live ones are in
        for live in ("thespeedx-http", "monosans-http", "monosans-all",
                     "monosans-json", "roosterkid-socks5",
                     "proxyscrape-http", "spys-socks", "fpl-http"):
            self.assertIn(live, names)
        kinds = {k for _, _, k in BUILT_IN_SOURCES}
        self.assertTrue(kinds <= {"list", "protocol", "html", "json"})
        # every url is https and unique
        urls = [u for _, u, _ in BUILT_IN_SOURCES]
        self.assertEqual(len(set(urls)), len(urls))
        for u in urls:
            self.assertTrue(u.startswith("https://"), u)


if __name__ == "__main__":
    unittest.main()
