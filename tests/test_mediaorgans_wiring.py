"""Dormant media/social organs wiring (assembly audit AA-06/08/13/17/
20/29/32/36): every organ exercised through its PRODUCTION path —
the durable scheduler + tool registry, the /dj and /music chat
surfaces, the voice-call CLI, the chat gateway's inbound funnel, and
the /voice + /audio control surfaces — never the organ alone.
"""

from __future__ import annotations

import json
import math
import struct
import time
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SR = 22050


# ── shared helpers ───────────────────────────────────────────────────

def _click_track(bpm: float, seconds: float = 30.0,
                 freq: float = 180.0) -> array:
    n = int(SR * seconds)
    out = array("d", [0.0]) * n
    beat_n = int(SR * 60.0 / bpm)
    for start in range(0, n, beat_n):
        for i in range(min(2205, n - start)):
            t = i / SR
            out[start + i] += math.exp(-t * 60.0) * math.sin(
                2 * math.pi * freq * t)
    return out


def _write_wav(path: str, samples: array, amp: int = 20000) -> str:
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(struct.pack(f"<{len(samples)}h",
                                  *(int(s * amp) for s in samples)))
    return path


def _registry(tmp_path, *, db=None):
    from nomorals.tools.registry import ToolRegistry

    ctx = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(workspace_dir=str(tmp_path),
                                 home_path=tmp_path),
        extras={}, router=None, tools=None)
    registry = ToolRegistry(ctx)
    ctx.tools = registry
    return ctx, registry


# ── AA-13 / AA-36: the media tool loader registers both organs ──────

class TestMediaToolLoader:
    def test_musicgen_chain_and_global_rhythm_registered(self, tmp_path):
        _ctx, registry = _registry(tmp_path)
        registry.register_builtins()
        assert registry.get("musicgen_chain") is not None
        assert registry.get("global_rhythm") is not None
        failed = {f["module"] for f in registry._failed_modules}
        assert "nomorals.media.musicgen_chain" not in failed
        assert "nomorals.media.global_rhythm" not in failed

    def test_global_rhythm_tool_generates_drums(self, tmp_path):
        _ctx, registry = _registry(tmp_path)
        registry.register_builtins()
        out = registry.call("global_rhythm", action="drums",
                            genre="afrobeats", bars=2)
        assert out.ok, out
        assert out.value["ok"] is True
        assert out.value["genre"] == "afrobeats"
        assert len(out.value["events"]) > 0
        voices = {e["voice"] for e in out.value["events"]}
        assert "kick" in voices

    def test_musicgen_chain_tool_runs_the_chain(self, tmp_path,
                                                 monkeypatch):
        monkeypatch.chdir(tmp_path)
        _ctx, registry = _registry(tmp_path)
        registry.register_builtins()
        out = registry.call("musicgen_chain", title="test-chain",
                            lyrics="Mo dé, ayọ dé", language="yo",
                            workdir=str(tmp_path / "chain_out"))
        assert out.ok, out
        value = out.value
        stages = {s["name"]: s["status"] for s in value["stages"]}
        # The draft stage is pure-local and always real; model stages
        # fail closed (skipped) with no backends provisioned — never
        # faked.
        assert stages.get("draft") == "real"
        assert Path(value["midi_path"]).exists()
        assert Path(value["spec_path"]).exists()
        assert Path(value["manifest_path"]).exists()


# ── AA-06: weekly content pipeline on the durable scheduler ─────────

def _sched_ctx(tmp_path):
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    return _registry(tmp_path, db=db)


class _SchedNotifier:
    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body})
        return {"delivered": True, "channel": "test"}


def _patch_pipeline_env(monkeypatch, tmp_path):
    """Offline pipeline env: no research hits, drafts in a tmp queue."""
    monkeypatch.setattr(
        "nomorals.research.pipeline.run_job",
        lambda job, ctx, budget=None: [])
    monkeypatch.setattr(
        "nomorals.social.drafts._db_path",
        lambda explicit=None: Path(explicit) if explicit
        else tmp_path / "drafts.db")


class TestWeeklyContentJob:
    def test_ensure_registers_the_job_idempotently(self, tmp_path):
        from nomorals.agents.scheduler import Scheduler
        from nomorals.social.jobs import ensure_weekly_content_jobs

        ctx, _registry_obj = _sched_ctx(tmp_path)
        sched = Scheduler(ctx)
        sched.notifier = _SchedNotifier()
        out = ensure_weekly_content_jobs(sched)
        assert out == [{"name": "weekly content", "scheduled": True,
                        "job_id": out[0]["job_id"]}]
        rows = [j for j in sched.list_jobs()
                if j.get("name") == "weekly content"]
        assert len(rows) == 1
        payload = json.loads(ctx.db.query_one(
            "SELECT payload FROM schedule_jobs WHERE name = ?",
            ("weekly content",))["payload"])
        assert payload == {"tool": "weekly_content",
                           "args": {"niche": ""}}
        routine = ctx.db.query_one(
            "SELECT routine FROM schedule_jobs WHERE name = ?",
            ("weekly content",))["routine"]
        assert not routine  # the batch report is the point — not routine
        again = ensure_weekly_content_jobs(sched)
        assert again == [{"name": "weekly content",
                          "already_scheduled": True}]
        assert len([j for j in sched.list_jobs()
                    if j.get("name") == "weekly content"]) == 1

    def test_job_runs_pipeline_end_to_end(self, tmp_path, monkeypatch):
        from nomorals.agents.scheduler import Scheduler
        from nomorals.social.drafts import DraftQueue
        from nomorals.social.jobs import ensure_weekly_content_jobs

        _patch_pipeline_env(monkeypatch, tmp_path)
        ctx, _registry_obj = _sched_ctx(tmp_path)
        _registry_obj.register_builtins()
        sched = Scheduler(ctx)
        sched.notifier = _SchedNotifier()
        ensure_weekly_content_jobs(sched)
        row = [j for j in sched.list_jobs()
               if j.get("name") == "weekly content"][0]
        out = sched.run_now(row["id"])
        assert out["ok"] is True, out
        runs = ctx.db.query(
            "SELECT ok FROM schedule_runs WHERE job_id = ?",
            (row["id"],))
        assert runs and all(r["ok"] for r in runs)
        # Drafts landed in the review queue — proposed, never posted.
        queue = DraftQueue()
        pending = queue.pending()
        assert len(pending) >= 5
        assert all(d.status in ("pending_review", "scheduled")
                   for d in pending)
        assert all(d.metadata.get("pipeline") == "weekly_content"
                   for d in pending)

    def test_tool_direct_call(self, tmp_path, monkeypatch):
        _patch_pipeline_env(monkeypatch, tmp_path)
        _ctx, registry = _sched_ctx(tmp_path)
        registry.register_builtins()
        out = registry.call("weekly_content", niche="afrobeats")
        assert out.ok, out
        assert out.value["ok"] is True
        assert out.value["niche"] == "afrobeats"
        assert out.value["drafts"] >= 5
        assert "drafts ready for review" in out.value["text"]


# ── AA-08: /dj long — the long-form mix on the DJ surface ───────────

def _fake_fetches(monkeypatch, tmp_path):
    from nomorals.media import dj_mixdrop as md
    from nomorals.media.dj_mixdrop import MixFetch

    p1 = _write_wav(str(tmp_path / "t1.wav"), _click_track(120.0))
    p2 = _write_wav(str(tmp_path / "t2.wav"), _click_track(124.0))

    def fake_fetch(q, wd, ctx=None):
        return {"song a": MixFetch(query=q, path=p1, title="Song A",
                                   artist="A", source="test"),
                "song b": MixFetch(query=q, path=p2, title="Song B",
                                   artist="B", source="test")}[q]

    monkeypatch.setattr(md, "fetch_track_audio", fake_fetch)
    return md


class TestDjLong:
    def test_control_dj_long_renders_real_segments(self, tmp_path,
                                                   monkeypatch):
        from nomorals.media.dj_long import control_dj_long

        _fake_fetches(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        out = control_dj_long("song a, song b", deliver_text=lambda t: None,
                              context=None)
        assert "long mix" in out
        assert "rendered as real audio" in out
        mix_dirs = list((tmp_path / "dj_mixes").glob("long_*"))
        assert len(mix_dirs) == 1
        chunks = list(mix_dirs[0].glob("chunk_*.json"))
        segments = list(mix_dirs[0].glob("segment_*.wav"))
        assert chunks and segments
        manifest = json.loads(
            (mix_dirs[0] / "mix_checkpoint.json").read_text())
        assert manifest["complete"] is True
        chunk0 = json.loads(chunks[0].read_text())
        assert chunk0["audio_rendered"] is True

    def test_control_dj_long_minutes_and_genre(self, tmp_path,
                                               monkeypatch):
        from nomorals.media import dj_mixdrop as md
        from nomorals.media.dj_long import control_dj_long

        _fake_fetches(monkeypatch, tmp_path)
        seen = {}

        def fake_trending(genre, n, context=None):
            seen["genre"] = genre
            return ["song a", "song b"]

        monkeypatch.setattr(md, "_trending_queries", fake_trending)
        monkeypatch.chdir(tmp_path)
        out = control_dj_long("30 afrobeats", context=None)
        assert seen["genre"] == "afrobeats"
        assert "long mix" in out

    def test_control_dj_long_needs_two_tracks(self, tmp_path,
                                              monkeypatch):
        from nomorals.media import dj_mixdrop as md
        from nomorals.media.dj_long import control_dj_long
        from nomorals.media.dj_mixdrop import MixFetch

        monkeypatch.setattr(
            md, "fetch_track_audio",
            lambda q, wd, ctx=None: MixFetch(query=q, error="offline"))
        monkeypatch.chdir(tmp_path)
        out = control_dj_long("song a, song b", context=None)
        assert "need at least 2" in out

    def test_runtime_dispatches_long(self, tmp_path, monkeypatch):
        from nomorals.agents.partner import runtime_media as rm

        _fake_fetches(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        mixin = MagicMock()
        mixin.context = MagicMock()
        out = rm.RuntimeMediaMixin._control_dj(
            mixin, "long song a, song b", chat_key="telegram:1")
        assert "long mix" in out


# ── AA-13: /music chain — the lyrics-first surface ──────────────────

class TestMusicChainSurface:
    def test_control_music_chain_produces_artifacts(self, tmp_path,
                                                    monkeypatch):
        from nomorals.agents.partner import runtime_media as rm

        monkeypatch.chdir(tmp_path)
        mixin = MagicMock()
        mixin.context = MagicMock()
        out = rm.RuntimeMediaMixin._control_music_chain(
            mixin, "Mo dé, ayọ dé, ìfẹ́ ni mo ní")
        assert "chain" in out
        assert "draft" in out
        midis = list((tmp_path / "musicgen_out").glob("*/*.mid"))
        manifests = list((tmp_path / "musicgen_out").glob("*/manifest*.json")) \
            or list((tmp_path / "musicgen_out").glob("*/*.json"))
        assert midis and manifests

    def test_control_music_chain_language_prefix(self, tmp_path,
                                                 monkeypatch):
        from nomorals.agents.partner import runtime_media as rm

        monkeypatch.chdir(tmp_path)
        mixin = MagicMock()
        mixin.context = MagicMock()
        out = rm.RuntimeMediaMixin._control_music_chain(
            mixin, "en: we rise together tonight")
        assert "chain" in out

    def test_control_music_chain_usage(self):
        from nomorals.agents.partner import runtime_media as rm

        mixin = MagicMock()
        out = rm.RuntimeMediaMixin._control_music_chain(mixin, "")
        assert "usage: /music chain" in out

    def test_music_dispatch_routes_chain(self, monkeypatch):
        from nomorals.agents.partner import runtime_media as rm

        seen = {}
        monkeypatch.setattr(
            rm.RuntimeMediaMixin, "_control_music_chain",
            lambda self, tail, chat_key="": seen.update(tail=tail)
            or "chain-ok")
        stub = object.__new__(rm.RuntimeMediaMixin)
        stub.context = None
        out = rm.RuntimeMediaMixin._control_music(stub, "chain some lyrics")
        assert out == "chain-ok"
        assert seen["tail"] == "some lyrics"


# ── AA-17: VoiceBridge as the live session's STT backend ────────────

class _FakeSTTEngine:
    def transcribe(self, path, language="en"):
        return {"text": "hello from the bridge", "language": language,
                "segments": []}


class TestVoiceBridgeWiring:
    def test_default_bridge_builds_on_universal_stt(self):
        from nomorals.voice.bridge import VoiceBridge, default_bridge
        from nomorals.voice.stt import UniversalSTT

        bridge = default_bridge()
        assert isinstance(bridge, VoiceBridge)
        assert isinstance(bridge.voice, UniversalSTT)

    def test_make_default_bridge_stt_transcribes(self, monkeypatch):
        from nomorals.voice import session as session_mod

        monkeypatch.setattr("nomorals.voice.stt.UniversalSTT",
                            _FakeSTTEngine)
        stt_fn = session_mod.make_default_bridge_stt()
        assert stt_fn.supports_partial is False
        assert stt_fn("note.wav") == "hello from the bridge"

    def test_parser_stt_flag(self):
        from nomorals.cmdline import _parser

        args = _parser().parse_args(["voice", "call", "--stt", "bridge"])
        assert args.stt == "bridge"
        args = _parser().parse_args(["voice", "call"])
        assert args.stt == "tools"
        with pytest.raises(SystemExit):
            _parser().parse_args(["voice", "call", "--stt", "bogus"])

    def test_voice_call_selects_bridge_stt(self, monkeypatch, tmp_path):
        from nomorals.cmdline.commands import voice as voice_cmd
        from nomorals.voice import session as session_mod

        sentinel = object()
        monkeypatch.setattr(session_mod, "make_default_bridge_stt",
                            lambda **kw: sentinel)
        monkeypatch.setattr(voice_cmd, "_voice_think_runtime",
                            lambda ctx, dev: (lambda text: ""))
        captured = {}

        class _FakeSession:
            def __init__(self, **kw):
                captured.update(kw)

            def run(self, **kw):
                return SimpleNamespace(session_id="s", turns=0,
                                       barge_ins=0, end_reason="done",
                                       error="")

        monkeypatch.setattr(session_mod, "VoiceSession", _FakeSession)
        settings = SimpleNamespace(audio=SimpleNamespace(
            voice_data_dir=str(tmp_path), tts_engine="",
            voice_silence_ms=500, voice_speech_cap_secs=30))
        context = SimpleNamespace(settings=settings, db=None)
        args = SimpleNamespace(
            stt="bridge", no_tts=True, device="default", profile="",
            keep_audio=False, audio_key_file="", turns=1, json=False)
        rc = voice_cmd._cmd_voice_call(args, context)
        assert rc == 0
        assert captured["stt"] is sentinel

    def test_voice_call_default_keeps_tools_stt(self, monkeypatch,
                                                tmp_path):
        from nomorals.cmdline.commands import voice as voice_cmd
        from nomorals.voice import session as session_mod

        sentinel = object()
        monkeypatch.setattr(voice_cmd, "_voice_stt_from_tools",
                            lambda ctx: sentinel)
        monkeypatch.setattr(voice_cmd, "_voice_think_runtime",
                            lambda ctx, dev: (lambda text: ""))
        captured = {}

        class _FakeSession:
            def __init__(self, **kw):
                captured.update(kw)

            def run(self, **kw):
                return SimpleNamespace(session_id="s", turns=0,
                                       barge_ins=0, end_reason="done",
                                       error="")

        monkeypatch.setattr(session_mod, "VoiceSession", _FakeSession)
        settings = SimpleNamespace(audio=SimpleNamespace(
            voice_data_dir=str(tmp_path), tts_engine="",
            voice_silence_ms=500, voice_speech_cap_secs=30))
        context = SimpleNamespace(settings=settings, db=None)
        args = SimpleNamespace(
            no_tts=True, device="default", profile="", keep_audio=False,
            audio_key_file="", turns=1, json=False)  # no stt attr at all
        rc = voice_cmd._cmd_voice_call(args, context)
        assert rc == 0
        assert captured["stt"] is sentinel


# ── AA-20: identity registration in the gateway inbound funnel ──────

def _gateway_with_store(tmp_path, **kw):
    from nomorals.social.chat.base import ChatAdapter, SendResult
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.identity import IdentityStore

    class _Adapter(ChatAdapter):
        name = "fake"

        def run(self, handler):
            pass

        def send(self, chat, text, *, reply_to=""):
            return SendResult(ok=True, platform=self.name)

    store = IdentityStore(str(tmp_path / "ids.json"))
    gw = ChatGateway({"fake": _Adapter()}, identity_store=store, **kw)
    return gw, store


def _inbound(platform, chat_id, sender_id, sender="Mama",
             username="mama_vee"):
    from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM),
        incoming=True, text="hi", sender=sender, sender_id=sender_id,
        sender_username=username, media=[], reply_to="", mentioned=False,
        ts=time.time(), message_id="m1")


class TestGatewayIdentity:
    def test_sender_registered_and_person_id_stamped(self, tmp_path):
        gw, store = _gateway_with_store(tmp_path)
        seen = []
        gw._inbound = seen.append
        gw._on_inbound(_inbound(
            "whatsapp", "2348012345678@s.whatsapp.net",
            "2348012345678@s.whatsapp.net"))
        assert len(seen) == 1
        person_id = seen[0].meta.get("person_id")
        assert person_id
        person = store.find("whatsapp", "2348012345678@s.whatsapp.net")
        assert person is not None and person.person_id == person_id
        # The WhatsApp JID IS the phone number — stored as the link key.
        assert person.identities[0].phone == "2348012345678"
        assert person.display_name == "Mama"
        # Persisted: a fresh store on the same path sees the person.
        from nomorals.social.identity import IdentityStore

        store2 = IdentityStore(str(tmp_path / "ids.json"))
        assert store2.find(
            "whatsapp", "2348012345678@s.whatsapp.net") is not None

    def test_same_sender_same_person_new_sender_new_person(self, tmp_path):
        gw, store = _gateway_with_store(tmp_path)
        seen = []
        gw._inbound = seen.append
        gw._on_inbound(_inbound("whatsapp", "2348012345678@s.whatsapp.net",
                                "2348012345678@s.whatsapp.net"))
        gw._on_inbound(_inbound("whatsapp", "2348012345678@s.whatsapp.net",
                                "2348012345678@s.whatsapp.net"))
        gw._on_inbound(_inbound("telegram", "555", "555",
                                sender="Other", username="other"))
        assert seen[0].meta["person_id"] == seen[1].meta["person_id"]
        assert seen[2].meta["person_id"] != seen[0].meta["person_id"]

    def test_owner_sender_flagged_owner(self, tmp_path):
        jid = "2348012345678@s.whatsapp.net"
        gw, store = _gateway_with_store(
            tmp_path, owner_chats={f"whatsapp:{jid}"})
        seen = []
        gw._inbound = seen.append
        gw._on_inbound(_inbound("whatsapp", jid, jid))
        person = store.find("whatsapp", jid)
        assert person is not None and person.is_owner is True

    def test_anonymous_sender_not_registered(self, tmp_path):
        gw, store = _gateway_with_store(tmp_path)
        seen = []
        gw._inbound = seen.append
        gw._on_inbound(_inbound("telegram", "999", ""))
        assert "person_id" not in seen[0].meta

    def test_no_store_no_stamp(self, tmp_path):
        from nomorals.social.chat.base import ChatAdapter, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class _Adapter(ChatAdapter):
            name = "fake"

            def run(self, handler):
                pass

            def send(self, chat, text, *, reply_to=""):
                return SendResult(ok=True, platform=self.name)

        gw = ChatGateway({"fake": _Adapter()})
        seen = []
        gw._inbound = seen.append
        gw._on_inbound(_inbound("telegram", "555", "555"))
        assert "person_id" not in seen[0].meta


# ── AA-29: /voice speakers — the diarized transcript surface ────────

class _FakeUniversalSTT:
    def transcribe(self, path, language="en"):
        return {"text": "hello there hi back", "language": language,
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "hello there"},
                    {"start": 2.0, "end": 4.0, "text": "hi back"}]}


def _patch_diarize(monkeypatch, diarizer):
    monkeypatch.setattr("nomorals.voice.stt.UniversalSTT",
                        _FakeUniversalSTT)
    monkeypatch.setattr("nomorals.voice.stt.diarize", diarizer)


class TestVoiceSpeakers:
    def test_speakers_produces_labeled_transcript_and_artifact(
            self, tmp_path, monkeypatch):
        from nomorals.audio.pipeline import control_voice

        _patch_diarize(monkeypatch, lambda path, speaker_names=None: {
            "segments": [
                {"start": 0.0, "end": 2.0, "speaker": "Ada"},
                {"start": 2.0, "end": 4.0, "speaker": "Bola"}]})
        audio = _write_wav(str(tmp_path / "meet.wav"),
                           _click_track(120.0, seconds=4.0))
        out = control_voice(f"speakers {audio}")
        assert "Ada: hello there" in out
        assert "Bola: hi back" in out
        artifact = Path(audio + ".diarized.json")
        assert artifact.exists()
        data = json.loads(artifact.read_text(encoding="utf-8"))
        assert data["speakers"] == ["Ada", "Bola"]
        assert data["segments"][0]["speaker"] == "Ada"

    def test_speakers_honest_unknown_without_diarizer(
            self, tmp_path, monkeypatch):
        from nomorals.audio.pipeline import control_voice

        def _no_pyannote(path, speaker_names=None):
            raise RuntimeError("diarize needs: pip install pyannote.audio")

        _patch_diarize(monkeypatch, _no_pyannote)
        audio = _write_wav(str(tmp_path / "meet.wav"),
                           _click_track(120.0, seconds=4.0))
        out = control_voice(f"speakers {audio}")
        assert "unknown: hello there" in out

    def test_speakers_rename_edits_the_artifact(self, tmp_path,
                                                monkeypatch):
        from nomorals.audio.pipeline import control_voice

        _patch_diarize(monkeypatch, lambda path, speaker_names=None: {
            "segments": [
                {"start": 0.0, "end": 2.0, "speaker": "Ada"},
                {"start": 2.0, "end": 4.0, "speaker": "Bola"}]})
        audio = _write_wav(str(tmp_path / "meet.wav"),
                           _click_track(120.0, seconds=4.0))
        control_voice(f"speakers {audio}")
        artifact = audio + ".diarized.json"
        out = control_voice(f"speakers rename {artifact} Ada Mama")
        assert "renamed Ada → Mama" in out
        data = json.loads(Path(artifact).read_text(encoding="utf-8"))
        assert data["segments"][0]["speaker"] == "Mama"
        out = control_voice(f"speakers rename {artifact} Nobody X")
        assert "no segments by" in out


# ── AA-32: /audio understand — the understanding layer on /audio ────

class TestAudioUnderstand:
    def _loud_quiet_wav(self, tmp_path) -> str:
        loud = array("d", [0.8 * math.sin(2 * math.pi * 440 * i / SR)
                           for i in range(SR * 3)])
        quiet = array("d", [0.02 * math.sin(2 * math.pi * 440 * i / SR)
                            for i in range(SR * 3)])
        return _write_wav(str(tmp_path / "clip.wav"), loud + quiet,
                          amp=32767)

    def test_understand_describes_sections_and_mood(self, tmp_path):
        from nomorals.audio.edit import control_audio

        path = self._loud_quiet_wav(tmp_path)
        out = control_audio(f"understand {path}")
        assert "mood:" in out
        assert "sections:" in out
        assert "main" in out

    def test_understand_missing_file_is_honest(self, tmp_path):
        from nomorals.audio.edit import control_audio

        out = control_audio(f"understand {tmp_path}/nope.wav")
        assert "couldn't listen" in out

    def test_usage_lists_understand(self):
        from nomorals.audio.edit import control_audio

        assert "/audio understand" in control_audio("")
