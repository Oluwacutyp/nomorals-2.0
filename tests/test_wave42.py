"""Wave 42 — the god-tier build (hermetic: no external network).

Covers the five new systems and their wiring:

- core/pdf.py               pure-Python PDF writer (render_pdf)
- tools/filesend.py         create_file + send_file (any live platform)
- training/collect.py       ConversationMiner → ShareGPT/Alpaca/ChatML bundles
- tools/osint_people.py     people-side OSINT (username/email/phone/breach/graph)
- tools/metadata.py         file forensics (EXIF, tEXt, ID3, docProps, …)
- training/free_datasets.py free dataset catalog + HF fetch/normalize
- agents/evolution.py       self-improvement agent with the mandatory test gate
- wiring                    registry tools, devon catalog, control table, env
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import zlib
from unittest import mock

from tests.test_partner_runtime import FakeAdapter, _make_context

from nomorals.core.errors import ToolError
from nomorals.core.pdf import render_pdf
from nomorals.social.chat.gateway import ChatGateway, SendResult
from nomorals.tools import metadata as meta_mod
from nomorals.tools import osint_people
from nomorals.tools.registry import ToolRegistry


def _registry_for(context) -> ToolRegistry:
    reg = ToolRegistry()
    reg.context = context
    reg.register_builtins()
    context.tools = reg
    return reg


class _MediaFake(FakeAdapter):
    """A FakeAdapter that also implements send_media and captures it."""

    def __init__(self, name: str = "media") -> None:
        super().__init__(name)
        self.media_sent: list[tuple[str, str, str]] = []

    def send_media(self, chat, media, *, caption: str = "") -> SendResult:
        self.media_sent.append((chat.key, media.path, caption))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"med{len(self.media_sent)}")


# ── PDF writer ───────────────────────────────────────────────────────────────


class PdfWriterTest(unittest.TestCase):
    def test_render_pdf_basic_structure(self) -> None:
        data = render_pdf("Hello PDF world", title="Wave 42")
        self.assertTrue(data.startswith(b"%PDF-1."))
        self.assertIn(b"%%EOF", data)
        self.assertIn(b"xref", data)
        self.assertGreater(len(data), 400)

    def test_render_pdf_multi_paragraph_and_escaping(self) -> None:
        data = render_pdf("Line one.\n\nLine two with (parens) and a backslash.",
                          title="Escape test")
        raw = data.decode("latin-1")
        self.assertIn("/FlateDecode", raw)
        # decode the content stream and assert on the drawn text
        stream = raw.split("stream\n", 1)[1].split("\nendstream", 1)[0]
        import zlib as _zlib
        drawn = _zlib.decompress(stream.encode("latin-1")).decode("latin-1")
        self.assertIn("Line one.", drawn)
        # parens are escaped inside content streams
        self.assertIn(r"\(", drawn)

    def test_render_pdf_unicode_falls_back_cleanly(self) -> None:
        data = render_pdf("naïve café — 42 ✓")
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertIn(b"%%EOF", data)


# ── research file creator + social sender ────────────────────────────────────


class FileToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _make_context()
        self.reg = _registry_for(self.ctx)
        self.ws = self.ctx.settings.workspace_dir
        self.ws.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_create_file_formats(self) -> None:
        for fmt, body in (("md", "# Title\n\ntext"), ("json", '{"a": 1}'),
                          ("txt", "plain text"), ("pdf", "pdf body")):
            r = self.reg.call("file_create", format=fmt, content=body,
                              title="T " + fmt, name="w42")
            self.assertTrue(r.ok, f"{fmt}: {getattr(r.error, 'message', r.error)}")
            v = r.value
            self.assertTrue(os.path.exists(v["path"]))
            self.assertEqual(v["format"], fmt)
            self.assertGreater(v["bytes"], 0)
        # pdf round-trips
        pdf = self.reg.call("file_create", format="pdf", content="x y z",
                            name="w42pdf")
        with open(pdf.value["path"], "rb") as fh:
            self.assertTrue(fh.read(5).startswith(b"%PDF-"))

    def test_send_file_with_media_adapter(self) -> None:
        adapter = _MediaFake("media")
        self.ctx.extras["gateway"] = ChatGateway({"media": adapter}, db=self.ctx.db)
        made = self.reg.call("file_create", format="txt", content="hello",
                             name="sendme")
        r = self.reg.call("file_send", platform="media", chat_id="999",
                          path=made.value["path"], caption="there")
        self.assertTrue(r.ok, getattr(r.error, "message", r.error))
        self.assertEqual(r.value["sent"], True)
        self.assertEqual(adapter.media_sent[0][0], "media:999")
        self.assertEqual(adapter.media_sent[0][2], "there")
        self.assertTrue(adapter.media_sent[0][1].endswith(".txt"))

    def test_send_file_adapter_without_media(self) -> None:
        plain = FakeAdapter("plain")
        self.ctx.extras["gateway"] = ChatGateway({"plain": plain}, db=self.ctx.db)
        made = self.reg.call("file_create", format="txt", content="x", name="x1")
        r = self.reg.call("file_send", platform="plain", chat_id="1",
                          path=made.value["path"])
        self.assertFalse(r.ok)
        self.assertIn("media", str(getattr(r.error, "message", r.error)).lower())

    def test_send_file_unknown_platform(self) -> None:
        self.ctx.extras["gateway"] = ChatGateway(
            {"media": _MediaFake()}, db=self.ctx.db)
        r = self.reg.call("file_send", platform="nope", chat_id="1", path="a.txt")
        self.assertFalse(r.ok)
        self.assertIn("no live adapter", str(getattr(r.error, "message", r.error)))


# ── conversation → training agent ────────────────────────────────────────────


class ConversationMinerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _make_context()
        self.reg = _registry_for(self.ctx)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def _seed_conversation(self, conv_id: str, turns: list[tuple[str, str]]) -> None:
        now = time.time()
        self.ctx.db.execute(
            "INSERT INTO conversations (id, title, agent, channel, summary, "
            "tokens, created_at, updated_at, metadata) VALUES (?, ?, 'partner', "
            "'test', '', 0, ?, ?, '{}')", (conv_id, conv_id, now, now))
        for i, (role, content) in enumerate(turns):
            self.ctx.db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, name, "
                "tokens, model, created_at, metadata) VALUES (?, ?, ?, ?, '', 0, "
                "'test-model', ?, '{}')",
                (f"{conv_id}-m{i}", conv_id, role, content, now + i))

    def test_mine_produces_full_bundle(self) -> None:
        self._seed_conversation("good", [
            ("user", "What's the best way to structure a python package for "
                     "a long-running bot so imports stay fast?"),
            ("assistant", "Use lazy imports for heavy third-party libraries, "
                          "keep the core package import-light, and split the "
                          "social layer into its own module so the CLI can "
                          "boot without it."),
            ("user", "And where should the state live?"),
            ("assistant", "SQLite with WAL for everything structured, and a "
                          "blob store for files. Never put secrets in the db — "
                          "env or a keychain."),
        ])
        self._seed_conversation("noise", [
            ("user", "/status"),
            ("assistant", "ok"),
        ])
        r = self.reg.call("train_mine", name="w42bundle")
        self.assertTrue(r.ok, getattr(r.error, "message", r.error))
        v = r.value
        self.assertGreaterEqual(v["examples"], 2)
        bundle_dir = v["dir"]
        base = v["name"]
        for suffix in ("alpaca.json", "sharegpt.json", "chatml.jsonl",
                       "manifest.json"):
            self.assertTrue(os.path.exists(os.path.join(bundle_dir, base + "." + suffix)),
                            f"missing {suffix}")
        with open(os.path.join(bundle_dir, base + ".alpaca.json")) as fh:
            alpaca = json.loads(fh.read())
        self.assertTrue(alpaca)
        self.assertIn("instruction", alpaca[0])
        # noise conversation produced nothing
        self.assertNotIn("/status", json.dumps(alpaca))
        self.assertTrue(v.get("dataset_id"))  # registered for nm train

    def test_empty_history_is_graceful(self) -> None:
        r = self.reg.call("train_mine", name="empty")
        self.assertTrue(r.ok)
        self.assertEqual(r.value["examples"], 0)


# ── OSINT: people-side ───────────────────────────────────────────────────────


class OsintPeopleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _make_context()
        self.reg = _registry_for(self.ctx)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_phone_investigate_offline(self) -> None:
        r = self.reg.call("phone_investigate", phone="+2348031234567")
        self.assertTrue(r.ok)
        self.assertEqual(r.value["e164"], "+2348031234567")
        self.assertEqual(r.value["country"], "Nigeria")
        r2 = self.reg.call("phone_investigate", phone="+442079460958")
        self.assertEqual(r2.value["country"], "United Kingdom")

    def test_username_check_mocked(self) -> None:
        calls = []

        class Resp:
            def __init__(self, code=200):
                self.status = code
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return b"ok"

        def fake_urlopen(request, timeout=None):
            url = request.full_url
            calls.append(url)
            if "github.com" in url or "gitlab.com" in url:
                return Resp(200)
            if "npmjs.com" in url or "reddit.com" in url or "t.me" in url:
                return Resp(404)
            return Resp(403)  # sandbox-style cut → unknown

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            r = self.reg.call("username_check", handle="cutyp")
        self.assertTrue(r.ok)
        v = r.value
        found = {r["site"] for r in v["found"]}
        self.assertIn("github", found)
        self.assertIn("gitlab", found)
        self.assertIn("npm", v["absent"])
        self.assertTrue(v["unknown"])  # cuts are never guessed

    def test_breach_check_without_key(self) -> None:
        os.environ.pop("NM_OSINT_HIBP_KEY", None)
        r = self.reg.call("breach_check", target="owner@example.com")
        self.assertTrue(r.ok)
        self.assertFalse(r.value["checked"])
        self.assertIn("NM_OSINT_HIBP_KEY", r.value["reason"])

    def test_breach_check_with_key_mocked(self) -> None:
        import io

        class Resp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return json.dumps([
                    {"Name": "BreachA", "BreachDate": "2024-01-01", "Title": "A"},
                    {"Name": "BreachB", "BreachDate": "2023-05-05", "Title": "B"},
                ]).encode()

        self.ctx.settings.osint.hibp_key = "testkey"
        with mock.patch("urllib.request.urlopen", side_effect=lambda *a, **k: Resp()):
            r = self.reg.call("breach_check", target="owner@example.com")
        self.assertTrue(r.ok)
        self.assertTrue(r.value["checked"])
        self.assertTrue(r.value["pwned"])
        self.assertEqual(len(r.value["breaches"]), 2)

    def test_email_investigate_mocked(self) -> None:
        import io

        class Resp:
            def __init__(self, payload):
                self._p = payload
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return json.dumps([
                    {"Name": "Leak", "BreachDate": "2024-02-02", "Title": "L"}
                ]).encode()

        def fake_dns(domain, record="A", **kw):
            if record == "MX":
                return ["10 mail1.example.com", "5 mail2.example.com"]
            if record == "TXT" and domain == "_dmarc.example.com":
                return ["v=DMARC1; p=reject"]
            if record in {"TXT", "SPF"}:
                return ["v=spf1 include:_spf.example.com ~all"]
            return []

        self.ctx.settings.osint.hibp_key = "testkey"
        with mock.patch.object(osint_people, "dns_query", side_effect=fake_dns):
            with mock.patch.object(
                    osint_people, "osint_domain",
                    return_value={"domain": "example.com",
                                  "registration": {"registrar": "FakeReg"}}):
                with mock.patch("urllib.request.urlopen",
                                side_effect=lambda *a, **k: Resp(None)):
                    r = self.reg.call("email_investigate",
                                      email="owner@example.com")
        self.assertTrue(r.ok)
        v = r.value
        self.assertIn("mail1.example.com", str(v["mx"]))
        self.assertIn("spf1", str(v["spf"]))
        self.assertIn("DMARC1", str(v["dmarc"]))
        self.assertEqual(v["domain_registration"]["registrar"], "FakeReg")
        self.assertTrue(v["breaches"]["pwned"])


# ── metadata forensics ───────────────────────────────────────────────────────


class MetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _make_context()
        self.reg = _registry_for(self.ctx)
        self.ws = self.ctx.settings.workspace_dir
        self.ws.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def _png(self) -> bytes:
        import struct as st
        def chunk(tag, data):
            c = tag + data
            return st.pack(">I", len(data)) + c + st.pack(">I", zlib.crc32(c) & 0xffffffff)
        ihdr = st.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0)
        t1 = b"Title" + b"\x00" + b"My Image"
        t2 = b"Software" + b"\x00" + b"TestGen"
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                + chunk(b"tEXt", t1) + chunk(b"tEXt", t2)
                + chunk(b"IEND", b""))

    def test_png(self) -> None:
        (self.ws / "t.png").write_bytes(self._png())
        r = self.reg.call("metadata_extract", source="t.png")
        self.assertTrue(r.ok)
        m = r.value
        self.assertEqual(m["dimensions"], "640x480")
        self.assertEqual(m["Title"], "My Image")
        self.assertEqual(m["Software"], "TestGen")
        self.assertEqual(len(m["sha256"]), 64)

    def test_pdf(self) -> None:
        (self.ws / "t.pdf").write_bytes(render_pdf("page one", title="T"))
        r = self.reg.call("metadata_extract", source="t.pdf")
        self.assertTrue(r.ok)
        self.assertEqual(r.value["format"], "PDF")
        self.assertGreaterEqual(r.value["pages"], 1)

    def test_docx(self) -> None:
        import zipfile
        core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                "<cp:coreProperties xmlns:cp='http://schemas.openxmlformats.org/"
                "package/2006/metadata/core-properties'>"
                "<dc:title xmlns:dc='http://purl.org/dc/elements/1.1/'>Quarterly"
                "</dc:title>"
                "<dc:creator xmlns:dc='http://purl.org/dc/elements/1.1/'>Cutyp"
                "</dc:creator></cp:coreProperties>")
        app = ("<?xml version='1.0'?><Properties "
               "xmlns='http://schemas.openxmlformats.org/officeDocument/2006/"
               "extensions'><Application>LibreOffice</Application></Properties>")
        with zipfile.ZipFile(self.ws / "t.docx", "w") as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types></Types>')
            zf.writestr("word/document.xml", '<?xml version="1.0"?><w:document></w:document>')
            zf.writestr("docProps/core.xml", core)
            zf.writestr("docProps/app.xml", app)
        r = self.reg.call("metadata_extract", source="t.docx")
        self.assertTrue(r.ok)
        self.assertEqual(r.value["title"], "Quarterly")
        self.assertEqual(r.value["creator"], "Cutyp")
        self.assertEqual(r.value["application"], "LibreOffice")

    def test_gif_comment(self) -> None:
        data = (b"GIF89a" + struct.pack("<HH", 10, 10) + b"\x00" + b"\x00\x00"
                + b"\x21\xfe\x05hello\x00" + b"\x3b")
        (self.ws / "t.gif").write_bytes(data)
        r = self.reg.call("metadata_extract", source="t.gif")
        self.assertTrue(r.ok)
        self.assertEqual(r.value["comment"], "hello")

    def test_mp3_id3v2_and_v1(self) -> None:
        def frame(fid, text):
            data = b"\x03" + text.encode()
            return fid + struct.pack(">I", len(data)) + b"\x00\x00" + data
        frames = frame(b"TIT2", "Song A") + frame(b"TPE1", "Artist B")
        hdr = (b"ID3" + b"\x04\x00" + b"\x00"
               + bytes([(len(frames) >> 21) & 0x7F, (len(frames) >> 14) & 0x7F,
                        (len(frames) >> 7) & 0x7F, len(frames) & 0x7F]))
        tag1 = (b"TAG" + b"Song A".ljust(30, b"\x00") + b"Artist B".ljust(30, b"\x00")
                + b"Album".ljust(30, b"\x00") + b"2026" + b"\x00" * 31)
        (self.ws / "t.mp3").write_bytes(hdr + frames + b"\xff\xfb" + b"\x00" * 200 + tag1)
        r = self.reg.call("metadata_extract", source="t.mp3")
        self.assertTrue(r.ok)
        m = r.value
        self.assertEqual(m["title"], "Song A")
        self.assertEqual(m["artist"], "Artist B")
        self.assertEqual(m["year_v1"], "2026")

    def test_wav(self) -> None:
        data = (b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00"
                b"\x44\xac\x00\x00\x88\x58\x01\x00\x04\x00\x10\x00" + b"\x00" * 20)
        (self.ws / "t.wav").write_bytes(data)
        r = self.reg.call("metadata_extract", source="t.wav")
        self.assertTrue(r.ok)
        self.assertEqual(r.value["sample_rate"], 44100)

    def test_junk_never_crashes(self) -> None:
        (self.ws / "junk.bin").write_bytes(b"\x00\x01\x02 junk")
        r = self.reg.call("metadata_extract", source="junk.bin")
        self.assertTrue(r.ok)
        import hashlib
        self.assertEqual(r.value["sha256"],
                         hashlib.sha256(b"\x00\x01\x02 junk").hexdigest())


# ── free datasets + fetch ────────────────────────────────────────────────────


class DatasetToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _make_context()
        self.reg = _registry_for(self.ctx)
        self.ws = self.ctx.settings.workspace_dir
        self.ws.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_free_catalog_in_train_datasets(self) -> None:
        r = self.reg.call("train_datasets", limit="10")
        self.assertTrue(r.ok)
        names = {e["name"] for e in r.value["free_catalog"]}
        for expected in ("oasst1", "hh-rlhf", "alpaca", "swe-bench-verified"):
            self.assertIn(expected, names)
        self.assertIn("dataset_fetch", r.value.get("fetch_hint", ""))

    def test_dataset_fetch_normalizes(self) -> None:
        import io
        import urllib.parse as up

        def server_rows(offset, length):
            rows, feats = [], [{"name": "instruction"}, {"name": "input"},
                               {"name": "output"}]
            for i in range(offset, min(offset + length, 40)):
                rows.append({"row": [f"instr {i}", f"in {i}", f"out {i}"]})
            return {"rows": rows, "num_rows_total": 40, "features": feats}

        class Resp(io.BytesIO):
            def __init__(self, payload):
                super().__init__(json.dumps(payload).encode())
            def __enter__(self):
                return self
            def __exit__(self, *a):
                self.close()

        def fake_urlopen(request, timeout=None):
            url = request.full_url
            if "/api/datasets/" in url:
                return Resp({"id": "x", "gated": False, "private": False,
                             "disabled": False, "siblings": []})
            if url.startswith("https://datasets-server.huggingface.co/info"):
                return Resp({"configs": [{"config_name": "default",
                                          "data": {"train": {}}}]})
            q = up.parse_qs(up.urlparse(url).query)
            return Resp(server_rows(int(q["offset"][0]), int(q["length"][0])))

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            r = self.reg.call("dataset_fetch", ref="alpaca", max_rows="25")
        self.assertTrue(r.ok, getattr(r.error, "message", r.error))
        self.assertEqual(r.value["rows"], 25)
        with open(r.value["path"]) as fh:
            first = json.loads(fh.readline())
        self.assertEqual(first["instruction"], "instr 0")
        # registered
        r2 = self.reg.call("train_datasets", limit="10")
        self.assertTrue(any("alpaca" in d["name"] for d in r2.value["datasets"]))

    def test_dataset_fetch_gated_refusal(self) -> None:
        import io
        import urllib.error as ue

        def fake_urlopen(request, timeout=None):
            raise ue.HTTPError(request.full_url, 401, "gated", {}, io.BytesIO())

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            r = self.reg.call("dataset_fetch", ref="oasst1")
        self.assertFalse(r.ok)
        self.assertIn("gated", str(getattr(r.error, "message", r.error)))


# ── the self-improvement agent (the test gate) ──────────────────────────────


def _temp_repo(tmp: str) -> None:
    os.makedirs(os.path.join(tmp, "tests"), exist_ok=True)
    open(os.path.join(tmp, "tests", "__init__.py"), "w").close()
    with open(os.path.join(tmp, "mod.py"), "w") as fh:
        fh.write("def answer():\n    return 41\n")
    with open(os.path.join(tmp, "tests", "test_mod.py"), "w") as fh:
        fh.write(
            "import unittest\nfrom mod import answer\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_answer(self):\n"
            "        self.assertEqual(answer(), 41)\n")
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"], ["git", "add", "-A"],
                ["git", "commit", "-q", "-m", "baseline"]):
        subprocess.run(cmd, cwd=tmp, check=True, capture_output=True)


class _ScriptedRouter:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages, params=None, **kw):
        from nomorals.llm.base import LLMResponse

        return LLMResponse(text=self.reply, model="fake")


class EvolutionTest(unittest.TestCase):
    """The god-tier guarantee: an unverified change never stays."""

    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="evo-w42-")
        _temp_repo(self.tmp)
        from nomorals.agents.evolution import EvolutionAgent

        self.agent = EvolutionAgent(self.ctx, repo_root=self.tmp)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmpctx.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _commit(self) -> None:
        subprocess.run(["git", "add", "-A"], cwd=self.tmp, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "step"], cwd=self.tmp,
                       capture_output=True)

    def test_good_change_applies(self) -> None:
        self.ctx.router = _ScriptedRouter(json.dumps({
            "rationale": "docstring",
            "edits": [{"path": "mod.py",
                       "old": "def answer():\n    return 41",
                       "new": "def answer():\n    \"\"\"The answer.\"\"\"\n    return 41"}]}))
        p = self.agent.plan("add a docstring to answer()")
        out = self.agent.apply(p.id)
        self.assertTrue(out["applied"])
        self.assertEqual(out["status"], "applied")
        with open(os.path.join(self.tmp, "mod.py")) as fh:
            self.assertIn('"""The answer."""', fh.read())

    def test_bad_change_reverts_exactly(self) -> None:
        self.ctx.router = _ScriptedRouter(json.dumps({
            "rationale": "break",
            "edits": [{"path": "mod.py", "old": "return 41", "new": "return 42"}]}))
        p = self.agent.plan("change the answer to 42")
        out = self.agent.apply(p.id)
        self.assertFalse(out["applied"])
        self.assertEqual(out["status"], "reverted")
        with open(os.path.join(self.tmp, "mod.py")) as fh:
            content = fh.read()
        self.assertNotIn("return 42", content)
        self.assertIn("return 41", content)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=self.tmp,
                                capture_output=True, text=True).stdout.strip()
        self.assertEqual(status, "")

    def test_gate_cannot_be_skipped_without_power(self) -> None:
        self.ctx.router = _ScriptedRouter(json.dumps({
            "rationale": "x",
            "edits": [{"path": "mod.py", "old": "return 41", "new": "return 41"}]}))
        p = self.agent.plan("noop marker edit for the gate test")
        with self.assertRaises(ToolError):
            self.agent.apply(p.id, verify=False)


# ── wiring ──────────────────────────────────────────────────────────────────


class WiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _make_context()
        self.reg = _registry_for(self.ctx)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_registry_has_wave42_tools(self) -> None:
        for name in ("file_create", "file_send", "report_publish",
                     "train_mine", "train_datasets",
                     "dataset_fetch", "username_check", "email_investigate",
                     "phone_investigate", "breach_check", "osint_people",
                     "metadata_extract", "evolve_plan", "evolve_apply",
                     "evolve_list"):
            self.assertIn(name, self.reg._tools, f"missing tool {name}")

    def test_devon_catalog_has_wave42(self) -> None:
        from nomorals.agents import devon

        names = {n for n, _ in devon.TOOL_CATALOG}
        for expected in ("username_check", "email_investigate",
                         "phone_investigate", "breach_check", "osint_people",
                         "metadata_extract", "file_create", "file_send",
                         "train_mine", "train_datasets", "dataset_fetch",
                         "evolve_plan", "evolve_apply", "evolve_list"):
            self.assertIn(expected, names, f"devon catalog missing {expected}")

    def test_control_table_has_wave42_commands(self) -> None:
        from nomorals.social.chat.control import CONTROL_COMMANDS

        for cmd in ("file", "publish", "data", "evolve"):
            self.assertIn(cmd, CONTROL_COMMANDS)

    def test_hibp_env_binds_to_settings(self) -> None:
        from nomorals.core.config import load_settings

        with mock.patch.dict(os.environ, {"NM_OSINT_HIBP_KEY": "abc123"}):
            settings = load_settings()
        self.assertEqual(settings.osint.hibp_key, "abc123")


if __name__ == "__main__":
    unittest.main()
