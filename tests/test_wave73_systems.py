"""Wave 73 — decoder auto-attack, real MediaHub.run, archive digest →
knowledge graph, app serving, and the code-runner CI loop, end to end,
hermetically.

* **Decoder auto-attack** — `nomorals/tools/decoder.py auto_attack`:
  identify → known-secrets → bounded multi-pass live crack
  (digits / lowercase / hybrid), honest not-found.
* **MediaHub.run** — `nomorals/media/__init__.py`: song → real MIDI;
  podcast → download → transcribe (verified against a *fake*
  OpenAI-compatible STT HTTP server) → summary → chapters → saved
  transcript; honest no-backend note.
* **Archivist.digest** — extract any archive and curate every text
  document into the knowledge graph + a memory episode.
* **AppBuilder.serve/stop/served** — actually spawn a server, health-check
  it, track it, kill it.
* **CodeRunner.run_until_green** — the CI loop: run → fail → model
  rewrite → re-run until exit 0, honestly red when no model can fix it.
* **Wiring** — tools in the registry, chat control commands, CLI
  subcommands, the NM_AUDIO_STT_* env map.

No network (the fake STT server is 127.0.0.1), no real mpv, temp homes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
import wave
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from nomorals.core.config import Settings
from nomorals.core.errors import ToolError
from nomorals.llm.base import LLMResponse


def _ctx(tmp: str, *, with_memory: bool = True,
         with_router: bool = False):
    from nomorals.agents.context import build_context

    ctx = build_context(Settings(home=tmp), with_executor=False,
                        with_tools=True, with_router=with_router,
                        with_memory=with_memory)
    ctx.__enter__()
    ws = Path(ctx.settings.workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    return ctx


class FakeRouter:
    """A router that always replies with one canned string."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def chat(self, messages, params=None, **kw: Any) -> LLMResponse:  # noqa: ANN001
        self.calls += 1
        return LLMResponse(text=self.reply)


class _FakeSTT:
    """OpenAI-compatible /audio/transcriptions server on 127.0.0.1."""

    def __init__(self, text: str) -> None:
        self.text = text

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = json.dumps({"text": text}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):  # silence
                pass

        self._srv = HTTPServer(("127.0.0.1", 0), H)
        self.port = self._srv.server_address[1]
        self._t = threading.Thread(target=self._srv.serve_forever,
                                   daemon=True)
        self._t.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()


def _wav(path: Path, seconds: float = 1.0) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * int(22050 * seconds))


# ═════════════════════════════════════════════════════════════════════════
# 1 · decoder auto-attack
# ═════════════════════════════════════════════════════════════════════════
class TestDecoderAutoAttack(unittest.TestCase):
    def test_known_secrets_table(self) -> None:
        from nomorals.tools.decoder import auto_attack

        d = hashlib.md5(b"hunter").hexdigest()
        out = auto_attack(d)
        self.assertTrue(out["cracked"])
        self.assertEqual(out["plaintext"], "hunter")
        self.assertEqual(out["via"], "known-secrets")

    def test_live_crack_digits_pass(self) -> None:
        from nomorals.tools.decoder import auto_attack

        d = hashlib.md5(b"9876").hexdigest()
        out = auto_attack(d, max_candidates=200_000)
        self.assertTrue(out["cracked"], out)
        self.assertEqual(out["plaintext"], "9876")
        self.assertEqual(out["via"], "live-crack:digits")
        self.assertTrue(out["passes"])
        self.assertGreater(out["tested"], 0)

    def test_custom_charset_single_pass(self) -> None:
        from nomorals.tools.decoder import auto_attack

        d = hashlib.md5(b"ab12").hexdigest()
        out = auto_attack(d, charset="abcdef0123456789", min_len=4,
                          max_len=4, max_candidates=300_000)
        self.assertTrue(out["cracked"], out)
        self.assertEqual(out["via"], "live-crack:custom")
        self.assertEqual(out["passes"][0]["pass"], "custom")

    def test_honest_not_found(self) -> None:
        from nomorals.tools.decoder import auto_attack

        d = hashlib.md5(b"fin7x9").hexdigest()  # 4-char mixed — out of range
        out = auto_attack(d, max_candidates=200_000)
        self.assertFalse(out["cracked"])
        self.assertEqual(out["via"], "not-found")
        self.assertIn("hashcrack", out["note"])
        self.assertGreaterEqual(len(out["passes"]), 2)

    def test_tool_crack_mode(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-dec-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            d = hashlib.md5(b"9876").hexdigest()
            r = ctx.tools.call("decoder", data=d, mode="crack")
            self.assertTrue(r.ok, r.error)
            out = r.unwrap()
            self.assertTrue(out["cracked"])
            self.assertEqual(out["plaintext"], "9876")
        finally:
            ctx.__exit__(None, None, None)


# ═════════════════════════════════════════════════════════════════════════
# 2 · MediaHub.run
# ═════════════════════════════════════════════════════════════════════════
class TestMediaHubRun(unittest.TestCase):
    def test_run_song_makes_real_midi(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-hub-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            r = ctx.tools.call("media_hub", action="run", mode="song",
                               topic="rain on a tin roof", style="lofi",
                               play=False)
            self.assertTrue(r.ok, r.error)
            out = r.unwrap()
            song = out["song"]
            self.assertTrue(song.get("midi_path"))
            midi = Path(song["midi_path"])
            self.assertTrue(midi.exists())
            self.assertEqual(midi.read_bytes()[:4], b"MThd")
        finally:
            ctx.__exit__(None, None, None)

    def test_podcast_transcribe_via_fake_stt(self) -> None:
        transcript = ("Welcome to the keynote. We are shipping the new "
                      "compiler with a faster optimizer. Build times drop "
                      "by forty percent on large codebases. The release "
                      "ships this Friday with migration guides. Thanks.")
        stt = _FakeSTT(transcript)
        os.environ["NM_AUDIO_STT_BASE_URL"] = stt.url
        os.environ["NM_AUDIO_STT_API_KEY"] = "fake-key"
        tmp = tempfile.mkdtemp(prefix="w73-hub2-")
        try:
            ctx = _ctx(tmp)
            ws = Path(ctx.settings.workspace_dir)
            wav = ws / "ep.wav"
            _wav(wav)

            from nomorals.media import MediaHub

            hub = MediaHub(ctx)
            t = hub._podcast_transcribe(str(wav), "Developer Keynote")
            self.assertEqual(t.get("transcript"), transcript)
            self.assertTrue(t.get("stt_provider"))
            self.assertTrue(t.get("chapters"))
            self.assertTrue(t.get("summary"))
            tp = Path(t["transcript_path"])
            self.assertTrue(tp.exists())
            blob = tp.read_text()
            self.assertIn("Podcast: Developer Keynote", blob)
            self.assertIn("chapters:", blob)
            self.assertIn("transcript:", blob)
        finally:
            os.environ.pop("NM_AUDIO_STT_BASE_URL", None)
            os.environ.pop("NM_AUDIO_STT_API_KEY", None)
            stt.close()
            ctx.__exit__(None, None, None)

    def test_podcast_honest_without_backend(self) -> None:
        os.environ.pop("NM_AUDIO_STT_BASE_URL", None)
        os.environ.pop("NM_AUDIO_STT_API_KEY", None)
        tmp = tempfile.mkdtemp(prefix="w73-hub3-")
        try:
            ctx = _ctx(tmp)
            wav = Path(ctx.settings.workspace_dir) / "ep.wav"
            _wav(wav)
            from nomorals.media import MediaHub

            t = MediaHub(ctx)._podcast_transcribe(str(wav), "No Backend")
            self.assertIn("stt_error", t)
            self.assertIn("note", t)
            self.assertIn("STT", t["note"])
        finally:
            ctx.__exit__(None, None, None)

    def test_hub_status_and_tool_registered(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-hub4-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            r = ctx.tools.call("media_hub", action="status")
            self.assertTrue(r.ok, r.error)
            out = r.unwrap()
            self.assertIn("backend", out)
            self.assertIn("playing", out)
        finally:
            ctx.__exit__(None, None, None)


# ═════════════════════════════════════════════════════════════════════════
# 3 · Archivist.digest → knowledge graph + memory
# ═════════════════════════════════════════════════════════════════════════
class TestArchiveDigest(unittest.TestCase):
    def _zip(self, ws: Path) -> Path:
        (ws / "doc1.txt").write_text(
            "Contact admin@example.com about the API. Server at 10.0.0.5, "
            "repo at https://github.com/acme/widget.")
        (ws / "notes.md").write_text(
            "# plan\n- ship parser · example.com primary · 2026-10-01")
        (ws / "page.html").write_text(
            "<html><body><p>support@acme.io</p></body></html>")
        (ws / "binary.bin").write_bytes(b"\x00\x01" * 50)
        z = ws / "docs.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for f in ("doc1.txt", "notes.md", "page.html", "binary.bin"):
                zf.write(ws / f, f)
        return z

    def test_zip_digest_curates_into_kg_and_memory(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-dig-")
        try:
            ctx = _ctx(tmp)
            ws = Path(ctx.settings.workspace_dir)
            z = self._zip(ws)
            r = ctx.tools.call("archive", action="digest", path="docs.zip")
            self.assertTrue(r.ok, r.error)
            d = r.unwrap()
            self.assertEqual(d["format"], "zip")
            self.assertEqual(d["text_files"], 3)  # html yes, binary no
            self.assertGreater(d["kg"]["added_nodes"], 0)
            self.assertTrue(d["memory_stored"])
            self.assertTrue(Path(d["extracted_to"]).is_dir())
            row = ctx.db.query_one(
                "SELECT 1 FROM kg_nodes WHERE label LIKE "
                "'%admin@example.com%'")
            self.assertIsNotNone(row)
            n = ctx.db.query_one(
                "SELECT COUNT(*) AS n FROM memories "
                "WHERE source='archive-digest'")
            self.assertEqual(n["n"], 1)
        finally:
            ctx.__exit__(None, None, None)

    def test_single_gz_digest(self) -> None:
        import gzip

        tmp = tempfile.mkdtemp(prefix="w73-dig2-")
        try:
            ctx = _ctx(tmp)
            ws = Path(ctx.settings.workspace_dir)
            with gzip.open(ws / "solo.txt.gz", "wb") as g:
                g.write(b"contact ops@ops.org daily")
            d = ctx.tools.call("archive", action="digest",
                               path="solo.txt.gz").unwrap()
            self.assertEqual(d["text_files"], 1)
            self.assertGreater(d["kg"]["added_nodes"], 0)
            self.assertIsNotNone(ctx.db.query_one(
                "SELECT 1 FROM kg_nodes WHERE label LIKE '%ops@ops.org%'"))
        finally:
            ctx.__exit__(None, None, None)


# ═════════════════════════════════════════════════════════════════════════
# 4 · AppBuilder serve / stop / served
# ═════════════════════════════════════════════════════════════════════════
class TestAppServing(unittest.TestCase):
    def _build(self, ctx: Any, name: str, stack: str = "static") -> None:
        from nomorals.builders import AppBuilder

        b = AppBuilder(ctx)
        out = b.build({"name": name, "stack": stack, "title": name})
        self.assertTrue(out["validation"]["ok"], out)

    def test_serve_served_stop_cycle(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-app-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            self._build(ctx, "demopage")
            from nomorals.builders import AppBuilder

            b = AppBuilder(ctx)
            out = b.serve("demopage")
            try:
                self.assertIn("http://localhost:", out["url"])
                self.assertTrue(out["pid"] > 0)
                self.assertTrue(out["health"].get("ok"), out)

                # the server actually answers
                from urllib.request import urlopen

                with urlopen(out["url"], timeout=5) as resp:
                    self.assertEqual(resp.status, 200)

                # re-serve: already running
                again = b.serve("demopage")
                self.assertEqual(again.get("note"), "already running")

                served = b.served()
                self.assertEqual(served["count"], 1)
                self.assertTrue(served["served"][0]["alive"])

                stopped = b.stop("demopage")
                self.assertTrue(stopped["stopped"])
            finally:
                if b.served().get("count"):
                    b.stop("demopage")
            self.assertEqual(b.served()["count"], 0)
        finally:
            ctx.__exit__(None, None, None)

    def test_cli_app_cannot_be_served(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-app2-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            self._build(ctx, "mycli", stack="cli-python")
            from nomorals.builders import AppBuilder

            with self.assertRaises(ToolError):
                AppBuilder(ctx).serve("mycli")
        finally:
            ctx.__exit__(None, None, None)


# ═════════════════════════════════════════════════════════════════════════
# 5 · CodeRunner CI loop
# ═════════════════════════════════════════════════════════════════════════
class TestRunUntilGreen(unittest.TestCase):
    BROKEN = "def f(x):\n    return x + 1\nprint(f('a'))\n"
    GOOD = "def f(x):\n    return x + 1\nprint(f(41))\n"

    def test_green_first_round(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-ci-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            from nomorals.execbox import CodeRunner

            r = CodeRunner(ctx).run_until_green(
                'print("hello green")', lang="python", expected="hello green")
            self.assertTrue(r["green"])
            self.assertEqual(r["rounds"], 1)
        finally:
            ctx.__exit__(None, None, None)

    def test_no_router_honest_red(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-ci2-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            from nomorals.execbox import CodeRunner

            r = CodeRunner(ctx).run_until_green(self.BROKEN, lang="python",
                                                max_rounds=4)
            self.assertFalse(r["green"])
            self.assertEqual(r["rounds"], 1)
            self.assertIn("no LLM backend", r["history"][0]["fix_note"])
            self.assertEqual(r["history"][0]["exit_code"], 1)
        finally:
            ctx.__exit__(None, None, None)

    def test_model_fixes_it(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-ci3-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            router = FakeRouter("```python\n" + self.GOOD + "```\n")
            ctx.router = router
            from nomorals.execbox import CodeRunner

            r = CodeRunner(ctx).run_until_green(self.BROKEN, lang="python",
                                                expected="42", max_rounds=4)
            self.assertTrue(r["green"])
            self.assertEqual(r["rounds"], 2)
            self.assertEqual(router.calls, 1)
            self.assertIn("42", r["final_run"]["stdout"])
        finally:
            ctx.__exit__(None, None, None)

    def test_unfixable_honest_red_max_rounds(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-ci4-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            ctx.router = FakeRouter("```python\n" + self.GOOD.replace(
                "41", "40") + "```\n")
            from nomorals.execbox import CodeRunner

            r = CodeRunner(ctx).run_until_green(self.BROKEN, lang="python",
                                                expected="42", max_rounds=3)
            self.assertFalse(r["green"])
            self.assertEqual(r["rounds"], 3)
            self.assertIn("still red", r["note"])
        finally:
            ctx.__exit__(None, None, None)

    def test_chatty_model_reply_rejected(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-ci5-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            ctx.router = FakeRouter("Acknowledged. Working on the fix now.")
            from nomorals.execbox import CodeRunner

            r = CodeRunner(ctx).run_until_green(self.BROKEN, lang="python",
                                                max_rounds=3)
            self.assertFalse(r["green"])
            self.assertEqual(r["rounds"], 1)
            self.assertIn("no usable code", r["history"][0]["fix_note"])
        finally:
            ctx.__exit__(None, None, None)

    def test_tool_until_green(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-ci6-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            ctx.router = FakeRouter("```python\n" + self.GOOD + "```\n")
            r = ctx.tools.call("run_code", action="until_green",
                               code=self.BROKEN, lang="python",
                               max_rounds=4, expected="42")
            self.assertTrue(r.ok, r.error)
            d = r.unwrap()
            self.assertTrue(d["green"])
            self.assertEqual(d["rounds"], 2)
        finally:
            ctx.__exit__(None, None, None)


# ═════════════════════════════════════════════════════════════════════════
# 6 · wiring: registry, chat commands, CLI, env map
# ═════════════════════════════════════════════════════════════════════════
class TestWave73Wiring(unittest.TestCase):
    def test_media_hub_tool_in_registry(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-w1-")
        try:
            ctx = _ctx(tmp, with_memory=False)
            names = ctx.tools.names()
            for tool in ("media_hub", "decoder", "archive", "build_app",
                         "run_code"):
                self.assertIn(tool, names)
        finally:
            ctx.__exit__(None, None, None)

    def test_control_commands_registered(self) -> None:
        from nomorals.social.chat.control import (CONTROL_COMMANDS,
                                                  parse_control)

        for k in ("hub", "podcast", "fix"):
            self.assertIn(k, CONTROL_COMMANDS)
        for msg in ("/hub song rain lofi", "/podcast dev keynote",
                    "/fix print(1) python --rounds 2", "/hub status"):
            c = parse_control(msg)
            self.assertIsNotNone(c, msg)

    def test_chat_handlers(self) -> None:
        tmp = tempfile.mkdtemp(prefix="w73-w2-")
        try:
            ctx = _ctx(tmp)
            ws = Path(ctx.settings.workspace_dir)
            from nomorals.agents.partner_runtime import PartnerRuntime

            rt = PartnerRuntime.__new__(PartnerRuntime)
            rt.context = ctx

            out = rt._control_hub("song rain on the window lofi")
            self.assertIn("🎵", out)
            self.assertIn(".mid", out)

            out = rt._control_fix('print("ok green")')
            self.assertIn("🟢 GREEN", out)

            out = rt._control_fix("print(total) python")
            self.assertIn("🔴 RED", out)

            with zipfile.ZipFile(ws / "docs.zip", "w") as zf:
                zf.writestr("a.txt", "contact boss@corp.io about release 9")
            out = rt._control_zip("digest docs.zip")
            self.assertIn("knowledge graph", out)
            self.assertIn("memory: episode stored", out)

            out = rt._control_apps("served")
            self.assertIn("no apps", out)
        finally:
            ctx.__exit__(None, None, None)

    def test_cli_has_hub_and_new_actions(self) -> None:
        from nomorals import cli

        parser = cli._parser()
        args = parser.parse_args(["hub", "song", "test", "--style", "lofi"])
        self.assertEqual(args.command, "hub")
        self.assertEqual(args.style, "lofi")

        args = parser.parse_args(["exec", "until_green", "print(1)",
                                  "--rounds", "2", "--expect", "1"])
        self.assertEqual(args.command, "exec")
        self.assertEqual(args.rounds, 2)

        args = parser.parse_args(["apps", "serve", "myapp", "--port", "8010"])
        self.assertEqual(args.command, "apps")
        self.assertEqual(args.port, 8010)

        args = parser.parse_args(["zip", "digest", "x.zip"])
        self.assertEqual(args.command, "zip")

    def test_audio_stt_env_map(self) -> None:
        from nomorals.core.config import load_settings

        tmp = tempfile.mkdtemp(prefix="w73-w3-")
        os.environ["NM_HOME"] = tmp
        os.environ["NM_AUDIO_STT_BASE_URL"] = "http://127.0.0.1:9/v1"
        os.environ["NM_AUDIO_STT_API_KEY"] = "k"
        try:
            s = load_settings(use_env_file=False)
            self.assertEqual(s.audio.stt_base_url, "http://127.0.0.1:9/v1")
            self.assertEqual(s.audio.stt_api_key, "k")
        finally:
            os.environ.pop("NM_HOME", None)
            os.environ.pop("NM_AUDIO_STT_BASE_URL", None)
            os.environ.pop("NM_AUDIO_STT_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
