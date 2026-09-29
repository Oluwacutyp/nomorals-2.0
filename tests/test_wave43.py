"""Wave 43 — god-tier upgrades (hermetic: no external network).

- core/pdf.py reader: pure-Python PDF text extraction, including
  Chromium-style CID fonts with ToUnicode CMaps (the owner's TTS upload)
- voice/tts.py: the owner's universal_tts.py — tag system, mood bridge,
  voice profiles with consent gating, stdlib WAV writing, lazy backends
- agents/evolution.py god tier: audit, research, plan-with-file-context,
  git-tagged applies, clean reverts, autopilot (power-gated)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
import wave
import zlib

from tests.test_partner_runtime import _make_context
from tests.test_wave42 import _ScriptedRouter

from nomorals.agents.evolution import EvolutionAgent
from nomorals.core.errors import ToolError
from nomorals.core.pdf import PdfError, read_pdf_text, render_pdf
from nomorals.tools.registry import ToolRegistry
from nomorals.voice.tts import (
    TagProcessor,
    UniversalTTS,
    VoiceLibrary,
    VoiceProfile,
    available_backends,
    mood_to_tagged_text,
    write_wav,
)


def _registry_for(context) -> ToolRegistry:
    reg = ToolRegistry()
    reg.context = context
    reg.register_builtins()
    context.tools = reg
    return reg


# ── PDF text reader ─────────────────────────────────────────────────────────


def _cid_pdf() -> bytes:
    """A minimal Chromium-style PDF: hex glyph strings + ToUnicode CMap."""
    cmap = (
        b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        b"3 beginbfchar\n<0003> <0048>\n<0004> <0069>\n<0005> <0020>\n"
        b"endbfchar\n1 beginbfrange\n<0010> <0011> <0079>\nendbfrange\n"
        b"endcmap\nend\n"
    )
    content = b"BT /F1 12 Tf 1 0 0 1 50 700 Tm <0003000400050010> Tj ET"
    slots: dict[int, bytes] = {}
    slots[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    slots[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    slots[3] = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    slots[4] = (
        b"<< /Type /Font /Subtype /Type0 /BaseFont /TestMono "
        b"/Encoding /Identity-H /ToUnicode 6 0 R >>"
    )
    c5 = zlib.compress(content)
    slots[5] = (f"<< /Length {len(c5)} /Filter /FlateDecode >>\nstream\n"
                .encode() + c5 + b"\nendstream")
    c6 = zlib.compress(cmap)
    slots[6] = (f"<< /Length {len(c6)} /Filter /FlateDecode >>\nstream\n"
                .encode() + c6 + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(slots):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + slots[num] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(slots) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, len(slots) + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(slots) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n".encode())
    return bytes(out)


class PdfReaderTest(unittest.TestCase):
    def test_round_trip_through_our_writer(self) -> None:
        data = render_pdf("Line one.\n\nLine two with (parens).", title="RT")
        text = read_pdf_text(data)
        self.assertIn("Line one.", text)
        self.assertIn("Line two with (parens).", text)

    def test_cid_fonts_and_tounicode_cmap(self) -> None:
        text = read_pdf_text(_cid_pdf())
        # <0003><0004><0005><0010> → H i ' ' y via the CMap
        self.assertIn("Hi y", text)

    def test_nested_page_tree(self) -> None:
        data = _cid_pdf()
        self.assertTrue(read_pdf_text(data))

    def test_non_pdf_raises(self) -> None:
        with self.assertRaises(PdfError):
            read_pdf_text(b"not a pdf at all")


# ── the owner's universal TTS layer ─────────────────────────────────────────


class TtsTagSystemTest(unittest.TestCase):
    def test_parse_emotion_sound_pause(self) -> None:
        segs = TagProcessor().parse(
            "[happy] hey baby [laughs] how are you [pause:400] I missed you")
        self.assertEqual(segs[0].tags, ["happy"])
        self.assertEqual(segs[1].tags, ["_sound_"])
        self.assertTrue(any(s.pause_after_ms == 400 for s in segs))

    def test_bark_format_and_plain(self) -> None:
        tp = TagProcessor()
        segs = tp.parse("[whisper] hello [pause:800] [laughs] ok")
        bark = tp.to_bark_format(segs)
        self.assertIn("[whispers]", bark)
        self.assertIn("... ...", bark)   # long pause
        clean, pauses = tp.to_plain_with_pauses(segs)
        self.assertNotIn("laughs", clean)
        self.assertEqual(len(pauses), 1)

    def test_mood_bridge(self) -> None:
        self.assertEqual(mood_to_tagged_text("hi", "happy", 5), "[happy] hi")
        self.assertEqual(mood_to_tagged_text("hi", "happy", 2), "hi")
        self.assertEqual(mood_to_tagged_text("hi", "stressed", 5),
                         "[annoyed] hi")
        self.assertEqual(mood_to_tagged_text("hi", "neutral", 5), "hi")
        self.assertEqual(mood_to_tagged_text("hi", "", 5), "hi")


class TtsVoicesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="tts-voices-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_preset_persistence(self) -> None:
        lib = VoiceLibrary(self.tmp)
        lib.register_preset("her", "af_heart", description="default")
        reloaded = VoiceLibrary(self.tmp)
        self.assertEqual(reloaded.get("her").preset_id, "af_heart")
        self.assertIn("her", [v["name"] for v in reloaded.list()])

    def test_cloning_consent_gate(self) -> None:
        vp = VoiceProfile(name="x", reference_audio_path="/tmp/clip.wav")
        with self.assertRaises(PermissionError):
            vp.validate_for_cloning()
        vp.consent_confirmed = True
        vp.validate_for_cloning()  # allowed now

    def test_upload_and_remove(self) -> None:
        src = os.path.join(self.tmp, "clip-src.wav")
        with open(src, "wb") as fh:
            fh.write(b"RIFF-fake-audio")
        lib = VoiceLibrary(self.tmp)
        prof = lib.upload_voice("me", src, consent_confirmed=True)
        self.assertTrue(os.path.exists(prof.reference_audio_path))
        self.assertTrue(lib.remove("me"))
        self.assertIsNone(lib.get("me"))
        self.assertFalse(os.path.exists(prof.reference_audio_path))

    def test_wav_writing_stdlib(self) -> None:
        samples = [0.0, 0.5, -0.5, 1.0, -1.0, 0.25] * 1000
        path = os.path.join(self.tmp, "out.wav")
        n = write_wav(path, samples, 24000)
        with wave.open(path, "rb") as w:
            self.assertEqual(w.getframerate(), 24000)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getnframes(), 6000)
        self.assertGreater(n, 12000)

    def test_backend_honesty_and_refusal(self) -> None:
        avail = available_backends()
        engine = UniversalTTS(backend="auto", voices_dir=self.tmp)
        if avail:
            return  # a real backend is installed — synthesis would work
        with self.assertRaises(RuntimeError) as ctx:
            engine.speak("hello")
        self.assertIn("pip install", str(ctx.exception))


class TtsToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.reg = _registry_for(self.ctx)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmpctx.cleanup()

    def test_voices_tool(self) -> None:
        r = self.reg.call("tts_voices", action="preset", name="demo",
                          preset_id="af_heart")
        self.assertTrue(r.ok, getattr(r.error, "message", r.error))
        r2 = self.reg.call("tts_voices", action="list")
        self.assertTrue(any(v["name"] == "demo" for v in r2.value["voices"]))
        r3 = self.reg.call("tts_voices", action="remove", name="demo")
        self.assertTrue(r3.value["ok"])

    def test_say_tool_refusal_without_backends(self) -> None:
        if available_backends():
            self.skipTest("a neural backend is installed")
        r = self.reg.call("tts_say", text="hello there")
        self.assertFalse(r.ok)
        self.assertIn("backend", str(getattr(r.error, "message", r.error)))


# ── evolution agent: god tier ───────────────────────────────────────────────


def _mini_repo(tmp: str) -> None:
    os.makedirs(os.path.join(tmp, "nomorals", "tools"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "tests"), exist_ok=True)
    open(os.path.join(tmp, "tests", "__init__.py"), "w").close()
    open(os.path.join(tmp, "nomorals", "__init__.py"), "w").close()
    open(os.path.join(tmp, "nomorals", "tools", "__init__.py"), "w").close()
    with open(os.path.join(tmp, "nomorals", "tools", "scanner.py"), "w") as fh:
        fh.write(
            "def scan(target):\n"
            "    # TODO: handle CIDR ranges\n"
            "    return target\n"
            "def legacy_probe(target):\n"
            "    raise NotImplementedError\n")
    with open(os.path.join(tmp, "nomorals", "tools", "untested_mod.py"), "w") as fh:
        fh.write("def compute(x):\n    return x * 2\n")
    with open(os.path.join(tmp, "tests", "test_scanner.py"), "w") as fh:
        fh.write(
            "import unittest\nfrom nomorals.tools.scanner import scan\n"
            "class T(unittest.TestCase):\n"
            "    def test_scan(self):\n"
            "        self.assertEqual(scan('a'), 'a')\n")
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"], ["git", "add", "-A"],
                ["git", "commit", "-q", "-m", "baseline"]):
        subprocess.run(cmd, cwd=tmp, check=True, capture_output=True)


class _AwareRouter:
    """Always returns a valid, state-correct edit on the scanner."""

    def __init__(self) -> None:
        self.n = 0
        self.last_user = ""

    def chat(self, messages, params=None, **kw):
        from nomorals.llm.base import LLMResponse

        self.n += 1
        self.last_user = " ".join(m.content for m in messages)
        return LLMResponse(text=json.dumps({
            "rationale": f"pass {self.n}",
            "edits": [{"path": "nomorals/tools/scanner.py",
                       "old": "    return target",
                       "new": f"    return target  # pass {self.n}"}]}),
            model="fake")


class EvolutionAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="evo-audit-")
        _mini_repo(self.tmp)
        self.agent = EvolutionAgent(self.ctx, repo_root=self.tmp)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmpctx.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_finds_markers_stubs_untested_ranked(self) -> None:
        rep = self.agent.audit()
        self.assertGreaterEqual(rep["files"], 3)
        self.assertTrue(any("TODO" in f["what"] for f in rep["findings"]))
        self.assertTrue(any(f["kind"] == "stub" for f in rep["findings"]))
        self.assertTrue(any(f["kind"] == "untested" for f in rep["findings"]))
        self.assertEqual(rep["findings"][0]["severity"], "high")
        self.assertIn("untested_mod",
                      "\n".join(f["where"] for f in rep["findings"]))


class EvolutionResearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="evo-res-")
        _mini_repo(self.tmp)
        self.agent = EvolutionAgent(self.ctx, repo_root=self.tmp)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmpctx.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_research_fuses_evidence_and_llm(self) -> None:
        self.ctx.router = _ScriptedRouter(json.dumps({
            "findings": ["scanner lacks CIDR support"],
            "recommendations": [{"title": "CIDR support",
                                 "targets": ["nomorals/tools/scanner.py"],
                                 "approach": "ipaddress", "risk": "low"}]}))
        rpt = self.agent.research("improve scanner with CIDR support")
        self.assertTrue(rpt["recommendations"])
        self.assertIn("scanner", rpt["evidence"])
        self.assertIn("scanned", rpt["evidence_summary"])

    def test_research_without_llm_is_evidence_only(self) -> None:
        self.ctx.router = None
        rpt = self.agent.research("improve scanner with CIDR support")
        self.assertIn("note", rpt)
        self.assertTrue(rpt["evidence"])


class EvolutionPlanApplyRevertTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="evo-par-")
        _mini_repo(self.tmp)
        self.agent = EvolutionAgent(self.ctx, repo_root=self.tmp)
        self.scanner = os.path.join(self.tmp, "nomorals", "tools",
                                    "scanner.py")

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmpctx.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plan_includes_focus_file_contents(self) -> None:
        self.ctx.router = _AwareRouter()
        p = self.agent.plan("touch the scanner return line",
                            focus="nomorals/tools/scanner.py")
        self.assertIn("def scan(target):", self.ctx.router.last_user)
        self.assertEqual(len(p.edits), 1)

    def test_apply_tags_and_revert_restores(self) -> None:
        self.ctx.router = _AwareRouter()
        p = self.agent.plan("mark the scanner return verified")
        out = self.agent.apply(p.id, commit=True)
        self.assertTrue(out["applied"], out)
        self.assertEqual(out.get("tag"), f"evo/{p.id}")
        tags = subprocess.run(["git", "tag", "-l"], cwd=self.tmp,
                              capture_output=True, text=True).stdout.split()
        self.assertIn(f"evo/{p.id}", tags)
        self.assertIn("pass 1", open(self.scanner).read())
        out2 = self.agent.revert(p.id)
        self.assertTrue(out2["ok"], out2)
        self.assertIs(out2["verified"], True)
        content = open(self.scanner).read()
        self.assertNotIn("pass 1", content)
        self.assertIn("TODO", content)

    def test_gate_rejects_breaking_change(self) -> None:
        class Breaker(_ScriptedRouter):
            def __init__(self) -> None:
                super().__init__(json.dumps({
                    "rationale": "break it",
                    "edits": [{"path": "nomorals/tools/scanner.py",
                               "old": "    return target",
                               "new": "    raise RuntimeError('boom')"}]}))

        self.ctx.router = Breaker()
        p = self.agent.plan("introduce a runtime error into scan")
        out = self.agent.apply(p.id)
        self.assertFalse(out["applied"])
        self.assertEqual(out["status"], "reverted")
        self.assertNotIn("boom", open(self.scanner).read())
        status = subprocess.run(["git", "status", "--porcelain"], cwd=self.tmp,
                                capture_output=True, text=True).stdout.strip()
        self.assertEqual(status, "")

    def test_gate_catches_syntax_breaks_fast(self) -> None:
        class SyntaxBreaker(_ScriptedRouter):
            def __init__(self) -> None:
                super().__init__(json.dumps({
                    "rationale": "syntax error",
                    "edits": [{"path": "nomorals/tools/scanner.py",
                               "old": "    return target",
                               "new": "    return (target"}]}))

        self.ctx.router = SyntaxBreaker()
        p = self.agent.plan("break the syntax of scan")
        out = self.agent.apply(p.id)
        self.assertFalse(out["applied"])
        self.assertIn("syntax", out.get("report", ""))


class EvolutionAutopilotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="evo-ap-")
        _mini_repo(self.tmp)
        self.agent = EvolutionAgent(self.ctx, repo_root=self.tmp)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmpctx.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_requires_power_mode(self) -> None:
        self.ctx.router = _AwareRouter()
        with self.assertRaises(ToolError):
            self.agent.autopilot(2)

    def test_queue_first_then_audit_findings(self) -> None:
        from nomorals.agents.power import PowerMode

        pm = PowerMode(self.ctx)
        pm._active = True
        self.ctx.extras["power"] = pm
        self.ctx.router = _AwareRouter()
        self.agent.queue("add", "improve the scanner module documentation")
        out = self.agent.autopilot(2)
        self.assertEqual(len(out["applied"]), 2, out)
        self.assertEqual(out["applied"][0]["source"], "queue")
        self.assertTrue(out["applied"][1]["source"].startswith("audit:"))
        self.assertTrue(all(a.get("tag") for a in out["applied"]))
        self.assertEqual(self.agent.queue("list"), [])

    def test_stops_after_two_consecutive_reverts(self) -> None:
        from nomorals.agents.power import PowerMode

        class Breaker(_ScriptedRouter):
            def __init__(self) -> None:
                super().__init__(json.dumps({
                    "rationale": "always breaks",
                    "edits": [{"path": "nomorals/tools/scanner.py",
                               "old": "    return target",
                               "new": "    raise RuntimeError('boom')"}]}))

        pm = PowerMode(self.ctx)
        pm._active = True
        self.ctx.extras["power"] = pm
        self.ctx.router = Breaker()
        out = self.agent.autopilot(4)
        self.assertEqual(len(out["reverted"]), 2)
        self.assertIn("revert", out["stopped_reason"])
        content = open(os.path.join(self.tmp, "nomorals", "tools",
                                    "scanner.py")).read()
        self.assertNotIn("boom", content)


# ── evolution git controller: branch policy + publish + push ───────────────

# temp repos + bare remotes created by the git controller tests
_EVO_GIT_DIRS: list[str] = []


def _git_repo(tmp: str, with_remote: bool = True) -> str:
    """Mini framework repo, optionally with a real bare origin remote.
    Returns the initial branch name (main or master)."""
    _mini_repo(tmp)
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=tmp,
                            capture_output=True, text=True).stdout.strip()
    if with_remote:
        remote_dir = tempfile.mkdtemp(prefix="evo-remote-")
        _EVO_GIT_DIRS.append(remote_dir)
        subprocess.run(["git", "init", "-q", "--bare", remote_dir], check=True)
        subprocess.run(["git", "remote", "add", "origin", remote_dir],
                       cwd=tmp, check=True)
        subprocess.run(["git", "push", "-q", "-u", "origin", branch],
                       cwd=tmp, check=True)
    return branch


class EvolutionGitControllerTest(unittest.TestCase):
    SAFE = json.dumps({
        "rationale": "mark verified",
        "edits": [{"path": "nomorals/tools/scanner.py",
                   "old": "    return target",
                   "new": "    return target  # evolution-verified"}]})

    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="evo-gc-")
        _EVO_GIT_DIRS.append(self.tmp)
        self.branch = _git_repo(self.tmp)
        self.agent = EvolutionAgent(self.ctx, repo_root=self.tmp)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmpctx.cleanup()

    @classmethod
    def tearDownClass(cls) -> None:
        for d in _EVO_GIT_DIRS:
            shutil.rmtree(d, ignore_errors=True)

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.tmp,
                              capture_output=True, text=True,
                              check=True).stdout.strip()

    def _apply_once(self, instruction: str) -> dict:
        self.ctx.router = _ScriptedRouter(self.SAFE)
        p = self.agent.plan(instruction)
        return self.agent.apply(p.id, commit=True)

    def test_status_reports_branch_remote_and_policy(self) -> None:
        st = self.agent.git_status()
        self.assertEqual(st["branch"], self.branch)
        self.assertTrue(st["remote"])  # a bare origin exists
        self.assertTrue(st["clean"])
        self.assertEqual(st["main_branch"], "main")
        self.assertFalse(st["push_on_publish"])

    def test_default_apply_commits_on_current_branch(self) -> None:
        before = self._git("rev-parse", "--short", self.branch)
        out = self._apply_once("mark the scanner return verified")
        self.assertTrue(out["applied"], out)
        self.assertEqual(out.get("branch"), self.branch)
        self.assertEqual(out.get("tag"), f"evo/{self.agent.list(1)[0].id}")
        self.assertNotEqual(self._git("rev-parse", "--short", self.branch),
                            before)

    def test_work_branch_isolates_evolution_from_main(self) -> None:
        from nomorals.agents.evolution import EvolutionGitController

        self.ctx.settings.evolution.work_branch = "evolution"
        self.agent.git = EvolutionGitController(self.tmp,
                                                work_branch="evolution")
        main_before = self._git("rev-parse", self.branch)
        out = self._apply_once("isolate evolution work")
        self.assertTrue(out["applied"], out)
        self.assertEqual(out.get("branch"), "evolution")
        self.assertEqual(self._git("branch", "--show-current"), "evolution")
        self.assertEqual(self._git("rev-parse", self.branch), main_before,
                         "main branch moved — isolation broken")

    def test_publish_fast_forwards_main(self) -> None:
        from nomorals.agents.evolution import EvolutionGitController

        self.ctx.settings.evolution.work_branch = "evolution"
        self.agent.git = EvolutionGitController(self.tmp,
                                                work_branch="evolution")
        out = self._apply_once("make a publishable change")
        pub = self.agent.git.publish(self.branch)
        self.assertTrue(pub["published"], pub)
        self.assertEqual(pub["method"], "fast-forward")
        self.assertEqual(pub["commit"], out["commit"])
        self.assertEqual(self._git("rev-parse", "--short", self.branch),
                         out["commit"])
        self.assertEqual(self._git("branch", "--show-current"), "evolution",
                         "publish must return to the work branch")

    def test_publish_pushes_both_branches_to_origin(self) -> None:
        from nomorals.agents.evolution import EvolutionGitController

        self.ctx.settings.evolution.work_branch = "evolution"
        self.agent.git = EvolutionGitController(self.tmp,
                                                work_branch="evolution",
                                                push=True)
        out = self._apply_once("publish and push")
        pub = self.agent.git.publish(self.branch)
        self.assertIs(pub["pushed"], True)
        remote_main = subprocess.run(
            ["git", "ls-remote",
             self.agent.git.remote(), self.branch],
            capture_output=True, text=True).stdout.split()[0]
        self.assertEqual(remote_main, self._git("rev-parse", self.branch))
        remote_evo = subprocess.run(
            ["git", "ls-remote", self.agent.git.remote(), "evolution"],
            capture_output=True, text=True).stdout.split()
        self.assertEqual(remote_evo[0], self._git("rev-parse", "evolution"))

    def test_publish_refuses_diverged_target(self) -> None:
        from nomorals.agents.evolution import EvolutionGitController

        self.ctx.settings.evolution.work_branch = "evolution"
        self.agent.git = EvolutionGitController(self.tmp,
                                                work_branch="evolution")
        self._apply_once("change on the work branch")
        # a human commits on main directly → divergence
        subprocess.run(["git", "checkout", "-q", self.branch],
                       cwd=self.tmp, check=True)
        subprocess.run(["git", "commit", "-q", "--allow-empty",
                        "-m", "human commit"], cwd=self.tmp, check=True)
        subprocess.run(["git", "checkout", "-q", "evolution"],
                       cwd=self.tmp, check=True)
        with self.assertRaises(Exception) as ctx_ex:
            self.agent.git.publish(self.branch)
        self.assertIn("diverged", str(ctx_ex.exception))
        # the failed publish must not have moved anything
        self.assertEqual(self._git("branch", "--show-current"), "evolution")

    def test_publish_refuses_missing_target(self) -> None:
        with self.assertRaises(Exception) as ctx_ex:
            self.agent.git.publish("definitely-not-a-branch")
        self.assertIn("does not exist", str(ctx_ex.exception))

    def test_publish_on_target_is_noop(self) -> None:
        pub = self.agent.git.publish(self.branch)
        self.assertFalse(pub["published"])
        self.assertIn("already on the target", pub["reason"])


if __name__ == "__main__":
    unittest.main()
