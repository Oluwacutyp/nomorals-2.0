"""Wave 75 systems — hermetic verification.

1.  Podcast transcripts auto-sent to the newest live chat (MediaHub.auto_send_target,
    tri-state send_transcript, forced target, offline handling).
2.  run_until_green on whole files & projects (test suite as the green
    criterion; honest red without an LLM backend; command selection).
3.  Digest a whole DIRECTORY into the KG (Archivist.digest_directory,
    archive tool + CLI).
4.  Apps deploy — a real reverse proxy behind a domain (builders_proxy,
    AppBuilder.deploy / stop_deploy / deployed, prefix + Host rewrite).
5.  Investigate agent — one pass over any artifact (classify → decode →
    crack → OSINT → KG + archived report).
6.  Decoder report archive — every DecodeReport persisted (save_report /
    decode_history / get_report, nm decode --history/--show).
7.  Hash corpus expansion — bundled wordlist + 15 mutation rules,
    parallel multi-digest attack (mixed algorithms in one stream).
8.  Vault key hierarchy — NM_VAULT_KEY master entries + encrypted
    export/import (re-sealed under the file key).
9.  Webhook hardening — HMAC signing over the exact body, retry/backoff,
    nm monitor webhook-test.
10. Monitor → decoder → OSINT pipeline — URL monitors auto-decode changed
    responses and feed the identity graph.
"""
from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import os
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


class _Hook(BaseHTTPRequestHandler):
    """Local webhook receiver: keeps raw body + signature header."""

    received = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        _Hook.received.append(
            {"raw": raw, "body": json.loads(raw),
             "sig": self.headers.get("X-NoMorals-Signature", "")})
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


class _FakeGateway:
    def __init__(self, online: tuple[str, ...] = ("telegram",)):
        self.online = online
        self.sent: list[tuple] = []

    def status(self):
        out = {n: {"connected": n in self.online} for n in
               ("telegram", "whatsapp")}
        out["_stats"] = {}
        return out

    def send_file(self, platform, chat, path, caption=""):
        self.sent.append((platform, chat, path, caption))

        class R:
            ok = True
            error = ""

        return R()


# ── 1. podcast transcript auto-send ─────────────────────────────────────────


class PodcastAutoSendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w75-pod-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        self.ws = Path(self.ctx.settings.workspace_dir)
        self.ws.mkdir(parents=True, exist_ok=True)
        self.afile = self.ws / "pod.mp3"
        self.afile.write_bytes(b"fakeaudio")
        self.ctx.gateway = _FakeGateway()

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)
        for k in ("NM_VAULT_KEY",):
            os.environ.pop(k, None)

    def _seed_chats(self, *rows):
        for row in rows:
            self.ctx.db.execute(
                "INSERT OR REPLACE INTO chats (id, platform, chat_id, kind, "
                "title, peer, is_owner, in_us, last_active) "
                "VALUES (?,?,?,?,?,?,?,?,?)", row)

    def test_auto_target_owner_first_then_newest(self) -> None:
        from nomorals.media import MediaHub

        hub = MediaHub(self.ctx)
        self._seed_chats(
            ("telegram:111", "telegram", "111", "dm", "A", "A", 1, 1, 100.0),
            ("telegram:222", "telegram", "222", "dm", "B", "B", 0, 1, 500.0))
        self.assertEqual(hub.auto_send_target(), ("telegram", "111"))
        self.ctx.db.execute("UPDATE chats SET is_owner=0 WHERE chat_id='111'")
        self.assertEqual(hub.auto_send_target(), ("telegram", "222"))
        self.ctx.gateway = _FakeGateway(online=())
        self.assertEqual(hub.auto_send_target(), ("", ""))

    def test_podcast_transcribe_auto_send(self) -> None:
        import nomorals.tools.audio as audio_mod
        from nomorals.media import MediaHub

        real = audio_mod.stt
        audio_mod.stt = lambda *a, **k: {
            "text": "hello from the test podcast " * 20, "provider": "fake"}
        try:
            self._seed_chats(
                ("telegram:222", "telegram", "222", "dm", "B", "B", 0, 1,
                 500.0))
            hub = MediaHub(self.ctx)
            out = hub._podcast_transcribe(str(self.afile), "Show",
                                          send_transcript=None)
            self.assertTrue(out.get("transcript_path"))
            self.assertEqual(self.ctx.gateway.sent[0][:2],
                             ("telegram", "222"))
            self.assertIn("Show", self.ctx.gateway.sent[0][3])

            self.ctx.gateway.sent.clear()
            hub._podcast_transcribe(str(self.afile), "Show",
                                    send_transcript=False)
            self.assertEqual(self.ctx.gateway.sent, [])

            self.ctx.gateway = _FakeGateway(online=())
            out3 = hub._podcast_transcribe(str(self.afile), "Show",
                                           send_transcript=True)
            self.assertIn("no send target", out3.get("send", ""))
            out4 = hub._podcast_transcribe(str(self.afile), "Show",
                                           send_transcript=None)
            self.assertIn("no platform online", out4.get("send", ""))

            gw = _FakeGateway()
            self.ctx.gateway = gw
            hub._podcast_transcribe(str(self.afile), "Show",
                                    send_transcript=True,
                                    send_to=("telegram", "999"))
            self.assertEqual(gw.sent[0][:2], ("telegram", "999"))
        finally:
            audio_mod.stt = real


# ── 2. run_until_green on files & projects ──────────────────────────────────


class ProjectUntilGreenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w75-ug-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        self.ws = Path(self.ctx.settings.workspace_dir)
        self.ws.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def _make_project(self, name: str, body: str) -> Path:
        proj = self.ws / name
        (proj / "tests").mkdir(parents=True)
        (proj / "calc.py").write_text(body)
        (proj / "tests" / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n"
            "class T(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n"
            "if __name__ == '__main__':\n    unittest.main()\n")
        return proj

    def test_green_project_round_one(self) -> None:
        from nomorals.execbox import CodeRunner

        out = CodeRunner(self.ctx).run_project_until_green(
            str(self._make_project("green", "def add(a, b):\n    return a + b\n")))
        self.assertTrue(out["green"])
        self.assertEqual(out["rounds"], 1)
        self.assertIn("unittest discover", out["criterion"])

    def test_red_project_without_llm_is_honest(self) -> None:
        from nomorals.execbox import CodeRunner

        out = CodeRunner(self.ctx).run_project_until_green(
            str(self._make_project("red", "def add(a, b):\n    return a - b\n")),
            max_rounds=2)
        self.assertFalse(out["green"])
        self.assertIn("no LLM backend", out["history"][0]["fix_note"])
        self.assertIn("failures", out["final_run"]["stderr"].lower())

    def test_command_selection(self) -> None:
        from nomorals.execbox import CodeRunner

        box = CodeRunner(self.ctx)
        f3 = self.ws / "solo_test.py"
        f3.write_text("import unittest\n"
                      "class T(unittest.TestCase):\n"
                      "    def test_x(self):\n        self.assertTrue(True)\n"
                      "if __name__ == '__main__':\n    unittest.main()\n")
        self.assertTrue(box._test_command(f3)[1].endswith("(tests)"))
        f4 = self.ws / "script.py"
        f4.write_text("print('hi')\n")
        self.assertEqual(box._test_command(f4)[1], "python script.py")
        out = box.run_project_until_green(str(f4))
        self.assertTrue(out["green"])

        proj4 = self.ws / "mainproj"
        proj4.mkdir()
        (proj4 / "main.py").write_text("print('ok')\n")
        out4 = box.run_project_until_green(str(proj4))
        self.assertTrue(out4["green"])
        self.assertEqual(out4["criterion"], "python main.py")

        proj5 = self.ws / "emptyproj"
        proj5.mkdir()
        (proj5 / "readme.txt").write_text("nothing")
        with self.assertRaises(Exception):
            box.run_project_until_green(str(proj5))


# ── 3. digest a whole directory ─────────────────────────────────────────────


class DirectoryDigestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w75-dd-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        self.ws = Path(self.ctx.settings.workspace_dir)
        self.ws.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def test_digest_directory(self) -> None:
        from nomorals.archives import Archivist

        d = self.ws / "docs"
        (d / "sub").mkdir(parents=True)
        (d / "a.md").write_text(
            "# Alpha\nThe quick brown fox jumps over the identity graph.")
        (d / "sub" / "b.txt").write_text("Second document about tools.")
        (d / "c.py").write_text("X = 1\n")
        out = Archivist(self.ctx).digest_directory(str(d))
        self.assertEqual(out["format"], "directory")
        self.assertEqual(out["text_files"], 3)
        self.assertGreater(out["kg"]["added_nodes"], 0)

    def test_tool_and_cli_take_a_directory(self) -> None:
        from nomorals import cli
        from nomorals.archives import Archivist

        d = self.ws / "dd"
        d.mkdir()
        (d / "n.md").write_text("# Note\nAbout the graph and people.")
        a = Archivist(self.ctx)
        self.assertTrue(d.is_dir())
        out = a.digest_directory(str(d))
        self.assertEqual(out["format"], "directory")

        parser = cli._parser()
        args = parser.parse_args(["zip", "digest", str(d)])
        self.assertEqual(cli._cmd_zip(args, self.ctx), 0)


# ── 4. apps deploy: a real reverse proxy behind a domain ────────────────────


class DeployTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w75-dep-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        from nomorals.builders import AppBuilder

        self.b = AppBuilder(self.ctx)
        out = self.b.build({"name": "demo", "stack": "static",
                            "title": "Demo", "features": ["homepage"]})
        self.assertTrue(out["validation"]["ok"])

    def tearDown(self) -> None:
        for name in ("demo",):
            try:
                self.b.stop_deploy(name)
            except Exception:
                pass
            try:
                self.b.stop(name)
            except Exception:
                pass
        self.ctx.__exit__(None, None, None)

    def test_plain_deploy_end_to_end(self) -> None:
        import urllib.request

        dep = self.b.deploy("demo")
        self.assertTrue(dep["health"]["ok"], dep)
        direct = urllib.request.urlopen(
            f"http://127.0.0.1:{dep['backend_port']}/", timeout=10
        ).read().decode()
        proxied = urllib.request.urlopen(
            dep["url"].replace("localhost", "127.0.0.1"), timeout=10
        ).read().decode()
        self.assertEqual(direct, proxied)
        self.assertIn("Demo", proxied)

    def test_domain_and_prefix_deploy(self) -> None:
        import urllib.request

        dep = self.b.deploy("demo", domain="demo.nomorals.local",
                            path="/demo")
        self.assertEqual(dep["url"], "http://demo.nomorals.local/demo")
        self.assertTrue(dep["health"]["ok"], dep)
        body = urllib.request.urlopen(
            f"http://127.0.0.1:{dep['port']}/demo/", timeout=10
        ).read().decode()
        self.assertIn("Demo", body)
        again = self.b.deploy("demo", domain="demo.nomorals.local",
                              path="/demo")
        self.assertEqual(again.get("note"), "already deployed")
        self.b.stop_deploy("demo")
        self.assertEqual(self.b.deployed()["count"], 0)

    def test_redirect_rewrite(self) -> None:
        from nomorals.builders_proxy import ReverseProxyHandler

        rw = ReverseProxyHandler._rewrite_location
        self.assertEqual(rw("/x", "http://d.x/demo"), "http://d.x/demo/x")
        self.assertEqual(rw("http://127.0.0.1:9/x?y=1", "http://d.x/demo"),
                         "http://d.x/demo/x?y=1")
        self.assertEqual(rw("/x", ""), "/x")


# ── 5. investigate: one pass over any artifact ──────────────────────────────


class InvestigateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w75-inv-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        self.ws = Path(self.ctx.settings.workspace_dir)
        self.ws.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def test_classify(self) -> None:
        from nomorals.agents.investigate import classify_artifact

        self.assertEqual(
            classify_artifact("5f4dcc3b5aa765d61d8327deb882cf99"), "digest")
        self.assertEqual(classify_artifact("https://example.com/a"), "url")
        self.assertEqual(classify_artifact("foo bar"), "text")

    def test_jwt_one_pass(self) -> None:
        from nomorals.agents.investigate import InvestigateAgent

        rep = InvestigateAgent(self.ctx).run(_jwt(
            {"email": "ada@example.com", "name": "Ada Lovelace"}))
        self.assertTrue(rep["ok"])
        self.assertGreaterEqual(rep["osint"]["persons_found"], 1)
        self.assertGreaterEqual(rep["osint"]["domains_found"], 1)
        self.assertTrue(rep["report_id"])
        self.assertIn("jwt", " ".join(rep["steps"]) or "jwt")

    def test_digest_one_pass_cracks(self) -> None:
        from nomorals.agents.investigate import InvestigateAgent

        d = hashlib.md5(b"hunter21").hexdigest()
        rep = InvestigateAgent(self.ctx).run(d, max_len=5)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["cracked"].get(d), "hunter21")

    def test_cookie_one_pass_osint(self) -> None:
        from nomorals.agents.investigate import InvestigateAgent

        rep = InvestigateAgent(self.ctx).run(
            "session=abc123; Domain=shop.example.com; Path=/; "
            "email=bob@shop.example.com")
        self.assertTrue(rep["ok"])
        self.assertGreaterEqual(rep["osint"]["persons_found"], 1)
        self.assertGreaterEqual(rep["osint"]["domains_found"], 1)


# ── 6. decoder report archive ───────────────────────────────────────────────


class ReportArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w75-rep-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def test_save_history_get(self) -> None:
        from nomorals.core.decoder import (analyze, decode_history, get_report,
                                           save_report)

        rep = analyze("aGVsbG8gd29ybGQ=")
        rid = save_report(self.ctx.db, rep, source="t", kind="blob",
                          input_text="aGVsbG8gd29ybGQ=")
        self.assertTrue(rid)
        hist = decode_history(self.ctx.db, limit=5)
        self.assertEqual(hist[0]["id"], rid)
        row = get_report(self.ctx.db, rid)
        self.assertIn("report_json", row)
        json.loads(row["report_json"])

    def test_analyze_tool_persists(self) -> None:
        from nomorals.core.decoder import decode_history

        r = self.ctx.tools.call("decoder", data="aGVsbG8=",
                                mode="analyze", auto_crack=False)
        self.assertTrue(r.ok, r.error)
        self.assertTrue(r.value.get("report_id"))
        hist = decode_history(self.ctx.db, limit=5)
        self.assertEqual(hist[0]["id"], r.value["report_id"])

    def test_cli_history_and_show(self) -> None:
        from nomorals import cli
        from nomorals.core.decoder import analyze, save_report

        rid = save_report(self.ctx.db, analyze("aGVsbG8="), source="cli",
                          kind="blob", input_text="aGVsbG8=")
        parser = cli._parser()
        self.assertEqual(cli._cmd_decode(
            parser.parse_args(["decode", "--history"]), self.ctx), 0)
        self.assertEqual(cli._cmd_decode(
            parser.parse_args(["decode", "--show", rid]), self.ctx), 0)


# ── 7. hash corpus + parallel multi-digest attack ───────────────────────────


class CorpusCrackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w75-cor-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        self.ws = Path(self.ctx.settings.workspace_dir)
        self.ws.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def test_corpus_shape(self) -> None:
        from nomorals.core import corpus

        st = corpus.corpus_stats()
        self.assertGreaterEqual(st["base_words"], 300)
        self.assertEqual(st["rules"], 15)
        self.assertGreater(st["expanded_unique"], 40000)
        variants = corpus.apply_rules("hunter2")
        for v in ("hunter2", "Hunter2", "hunter21", "HUNTER2"):
            self.assertIn(v, variants)
        n = sum(1 for _ in corpus.rule_stream(["hunter2"]))
        self.assertGreaterEqual(n, 20)

    def test_export_wordlist(self) -> None:
        from nomorals.core import corpus

        out = self.ws / "wl.txt"
        n = corpus.export_wordlist(str(out))
        self.assertEqual(n, len(out.read_text().splitlines()))
        self.assertGreater(n, 40000)

    def test_live_crack_bundled_corpus(self) -> None:
        from nomorals.tools.hashcrack import crack_hash

        d = hashlib.md5(b"hunter21").hexdigest()
        res = crack_hash(d, db=self.ctx.db, max_candidates=400000)
        self.assertEqual(res.found.get(d), "hunter21")
        d2 = hashlib.sha1(b"Password123").hexdigest()
        res2 = crack_hash(d2, db=self.ctx.db, max_candidates=400000)
        self.assertEqual(res2.found.get(d2), "Password123")

    def test_mixed_algorithm_batch(self) -> None:
        from nomorals.tools.hashcrack import crack_hash

        d_md5 = hashlib.md5(b"p@ssw0rd").hexdigest()
        d_sha = hashlib.sha1(b"Password").hexdigest()
        d_256 = hashlib.sha256(b"hunter21234").hexdigest()
        res = crack_hash(f"{d_md5},{d_sha},{d_256}", db=self.ctx.db,
                         max_candidates=800000)
        self.assertEqual(res.found.get(d_md5), "p@ssw0rd")
        self.assertEqual(res.found.get(d_sha), "Password")
        self.assertEqual(res.found.get(d_256), "hunter21234")
        self.assertEqual(res.remaining, [])

    def test_known_hash_chain_short_circuits_cli(self) -> None:
        from nomorals import cli
        from nomorals.core.decoder import learn_hash

        d = hashlib.md5(b"13579").hexdigest()
        learn_hash(self.ctx.db, d, "13579", algorithm="md5",
                   source="w75-test")
        parser = cli._parser()
        rc = cli._cmd_crack(parser.parse_args(
            ["crack", "--hash", d]), self.ctx)
        self.assertEqual(rc, 0)


# ── 8. vault key hierarchy ──────────────────────────────────────────────────


class VaultHierarchyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w75-vlt-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        self.ws = Path(self.ctx.settings.workspace_dir)
        self.ws.mkdir(parents=True, exist_ok=True)
        self._old_key = os.environ.get("NM_VAULT_KEY")
        self.ag = None

    def tearDown(self) -> None:
        if self._old_key is None:
            os.environ.pop("NM_VAULT_KEY", None)
        else:
            os.environ["NM_VAULT_KEY"] = self._old_key
        self.ctx.__exit__(None, None, None)

    def _agent(self, ctx=None):
        from nomorals.agents.cipher import CipherAgent

        return CipherAgent(context=ctx or self.ctx, name="t")

    def test_master_key_entries(self) -> None:
        os.environ["NM_VAULT_KEY"] = "mk-123"
        ag = self._agent()
        r = ag.run({"action": "vault_put", "name": "m1", "data": "s3cr3t"})
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.output["key_scheme"], "master")
        self.assertEqual(ag.run(
            {"action": "vault_get", "name": "m1"}).output["data"], "s3cr3t")
        os.environ["NM_VAULT_KEY"] = "nope"
        bad = ag.run({"action": "vault_get", "name": "m1"})
        self.assertFalse(bad.ok)
        self.assertIn("wrong key", bad.error)
        os.environ["NM_VAULT_KEY"] = "mk-123"

    def test_pass_entries_and_rm_without_pass(self) -> None:
        ag = self._agent()
        ag.run({"action": "vault_put", "name": "p1", "data": "v",
                "passphrase": "pp"})
        self.assertEqual(ag.run(
            {"action": "vault_get", "name": "p1", "passphrase": "pp"}
        ).output["data"], "v")
        bad = ag.run({"action": "vault_get", "name": "p1",
                      "passphrase": "x"})
        self.assertIn("wrong passphrase", bad.error)
        self.assertTrue(ag.run(
            {"action": "vault_rm", "name": "p1"}).output["removed"])

    def test_list_shows_schemes(self) -> None:
        os.environ["NM_VAULT_KEY"] = "mk-123"
        ag = self._agent()
        ag.run({"action": "vault_put", "name": "m", "data": "a"})
        ag.run({"action": "vault_put", "name": "p", "data": "b",
                "passphrase": "pp"})
        lst = ag.run({"action": "vault_list"}).output
        schemes = {e["name"]: e["key_scheme"] for e in lst["entries"]}
        self.assertEqual(schemes, {"m": "master", "p": "pass"})

    def test_export_import_round_trip(self) -> None:
        os.environ["NM_VAULT_KEY"] = "mk-123"
        ag = self._agent()
        ag.run({"action": "vault_put", "name": "m", "data": "A"})
        ag.run({"action": "vault_put", "name": "p", "data": "B",
                "passphrase": "pp"})
        bundle = self.ws / "vault.json"
        r = ag.run({"action": "vault_export", "path": str(bundle),
                    "passphrase": "filekey", "entry_pass": "pp"})
        self.assertTrue(r.ok, r.error)
        self.assertEqual(json.loads(bundle.read_text())["format"],
                         "nomorals-vault")
        tmp2 = tempfile.mkdtemp(prefix="w75-vlt2-")
        ctx2 = _ctx(tmp2)
        ctx2.__enter__()
        try:
            os.environ.pop("NM_VAULT_KEY", None)
            ag2 = self._agent(ctx2)
            r2 = ag2.run({"action": "vault_import", "path": str(bundle),
                          "passphrase": "filekey"})
            self.assertTrue(r2.ok, r2.error)
            self.assertEqual(r2.output["count"], 2)
            self.assertEqual(ag2.run(
                {"action": "vault_get", "name": "m",
                 "passphrase": "filekey"}).output["data"], "A")
            self.assertEqual(ag2.run(
                {"action": "vault_get", "name": "p",
                 "passphrase": "filekey"}).output["data"], "B")
        finally:
            ctx2.__exit__(None, None, None)

    def test_cli_export_import(self) -> None:
        from nomorals import cli

        os.environ["NM_VAULT_KEY"] = "mk-123"
        parser = cli._parser()
        self.assertEqual(cli._cmd_cipher(parser.parse_args(
            ["cipher", "vault_put", "cli1", "val"]), self.ctx), 0)
        f = self.ws / "v.json"
        self.assertEqual(cli._cmd_cipher(parser.parse_args(
            ["cipher", "vault_export", str(f), "--passphrase", "ekey"]),
            self.ctx), 0)
        self.assertEqual(json.loads(f.read_text())["format"],
                         "nomorals-vault")
        self.assertEqual(cli._cmd_cipher(parser.parse_args(
            ["cipher", "vault_import", str(f), "--passphrase", "ekey"]),
            self.ctx), 0)


# ── 9. webhook hardening ────────────────────────────────────────────────────


class WebhookHardenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w75-wh-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        self.ws = Path(self.ctx.settings.workspace_dir)
        self.ws.mkdir(parents=True, exist_ok=True)
        _Hook.received = []
        self.srv = HTTPServer(("127.0.0.1", 0), _Hook)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/hook"
        self.thread = threading.Thread(target=self.srv.serve_forever,
                                       daemon=True)
        self.thread.start()
        from nomorals.agents.monitor import MonitorAgent

        self.agent = MonitorAgent(self.ctx)

    def tearDown(self) -> None:
        self.srv.shutdown()
        self.ctx.__exit__(None, None, None)

    def test_hmac_signature_over_raw_body(self) -> None:
        f = self.ws / "w.txt"
        f.write_text("v1")
        self.agent.add(str(f), interval=30, webhook=self.url,
                       secret="s3cret", min_gap=0)
        t0 = 1_000_000.0
        self.agent.tick(now=t0)
        f.write_text("v2 changed")
        c = self.agent.tick(now=t0 + 31)["changed"][0]
        self.assertTrue(c["webhook"]["ok"])
        rec = _Hook.received[0]
        self.assertTrue(rec["sig"].startswith("sha256="))
        expect = hmac_mod.new(b"s3cret", rec["raw"],
                              hashlib.sha256).hexdigest()
        self.assertEqual(rec["sig"], "sha256=" + expect)
        self.assertEqual(rec["body"]["event"], "change")

    def test_retry_on_dead_endpoint_never_raises(self) -> None:
        f = self.ws / "d.txt"
        f.write_text("a")
        self.agent.add(str(f), interval=30, webhook="http://127.0.0.1:1/x",
                       min_gap=0)
        t0 = 2_000_000.0
        self.agent.tick(now=t0)
        f.write_text("b")
        c = self.agent.tick(now=t0 + 31)["changed"][0]
        self.assertFalse(c["webhook"]["ok"])
        self.assertGreaterEqual(c["webhook"].get("attempts", 0), 3)

    def test_webhook_test_cli(self) -> None:
        from nomorals import cli

        f = self.ws / "wt.txt"
        f.write_text("x")
        parser = cli._parser()
        self.assertEqual(cli._cmd_monitor(parser.parse_args(
            ["monitor", "add", str(f), "--interval", "30",
             "--webhook", self.url, "--secret", "s"]), self.ctx), 0)
        self.assertEqual(cli._cmd_monitor(parser.parse_args(
            ["monitor", "webhook-test", str(f)]), self.ctx), 0)
        rec = _Hook.received[-1]
        self.assertEqual(rec["body"]["event"], "test")
        expect = hmac_mod.new(b"s", rec["raw"], hashlib.sha256).hexdigest()
        self.assertEqual(rec["sig"], "sha256=" + expect)


# ── 10. monitor → decoder → OSINT pipeline ──────────────────────────────────


class MonitorDecodePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="w75-pipe-")
        self.ctx = _ctx(self.tmp)
        self.ctx.__enter__()
        self.ws = Path(self.ctx.settings.workspace_dir)
        self.ws.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def test_url_monitor_change_feeds_decoder_reports(self) -> None:
        from nomorals.agents.monitor import MonitorAgent
        from nomorals.core.decoder import decode_history

        class Site(BaseHTTPRequestHandler):
            content = "aGVsbG8K"

            def do_GET(self):
                body = self.content.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), Site)
        url = f"http://127.0.0.1:{srv.server_address[1]}/page"
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            agent = MonitorAgent(self.ctx)
            before = len(decode_history(self.ctx.db, limit=200))
            agent.add(url, interval=30, min_gap=0)
            t0 = 3_000_000.0
            agent.tick(now=t0)
            Site.content = "ZXlKbWMyOXBaQzVpYVc1akJqWT0="
            agent.tick(now=t0 + 31)
            hist = decode_history(self.ctx.db, limit=200)
            mon = [h for h in hist if h["source"].startswith("monitor:")]
            self.assertTrue(mon)
            self.assertGreater(len(hist), before)
            self.assertTrue(any(m["best_name"] for m in mon))
        finally:
            srv.shutdown()

    def test_file_monitor_decode_row(self) -> None:
        from nomorals.agents.monitor import MonitorAgent
        from nomorals.core.decoder import decode_history

        f = self.ws / "mon2.txt"
        f.write_text("v1")
        agent = MonitorAgent(self.ctx)
        agent.add(str(f), interval=30, min_gap=0)
        t0 = 4_000_000.0
        agent.tick(now=t0)
        f.write_text(base64.b64encode(b"plain secret text").decode())
        agent.tick(now=t0 + 31)
        mon = [h for h in decode_history(self.ctx.db, limit=200)
               if h["source"].startswith("monitor:")]
        self.assertTrue(mon)


if __name__ == "__main__":
    unittest.main()
