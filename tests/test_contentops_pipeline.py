"""Tests for nomorals.media.contentops.pipeline.

ALL heavy stages are mocked: fake brain / fake TTS / fake imggen / fake
edit contract returning tiny fixtures.  The pipeline orchestration itself
(stage order, artifact layout, resume, clean errors, calendar cadence, job
persistence, estimator) is what's under test.

The fake edit renderer shells out to the real ffmpeg (testsrc) so the
downstream burn-in / mix / mux stages exercise real media handling on
valid files; the whole file is skipped when ffmpeg is absent.
"""

import json
import math
import os
import shutil
import struct
import sys
import wave
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.media.contentops.pipeline import (  # noqa: E402
    STAGES,
    ContentCalendar,
    Job,
    ShortPipeline,
    StageError,
    parse_script,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

CALLS: list[str] = []

SCRIPT_FIXTURE = """HOOK: Your brain deletes most of what you see.
[SCENE 1]
SAY: Every second your eyes take in ten million bits of information.
SHOW: a giant human eye made of glowing data streams, dark background
[SCENE 2]
SAY: Your brain throws away almost all of it and shows you a highlight reel.
SHOW: a tiny brain filtering a flood of light into a single beam
[SCENE 3]
SAY: That is why two people can watch the same thing and remember it differently.
SHOW: two silhouettes watching different movies on the same screen
TITLE: Your Brain Is Lying To You
DESCRIPTION: Perception is edited in real time. Here is the 60 second version.
"""


# ── fakes ────────────────────────────────────────────────────────────


class FakeBrain:
    def __init__(self, text: str = SCRIPT_FIXTURE, fail: bool = False):
        self.text = text
        self.fail = fail

    def complete(self, prompt, **kw):
        CALLS.append("brain.complete")
        assert prompt and prompt.strip(), "empty script prompt"
        if self.fail:
            return SimpleNamespace(ok=False, text="",
                                   error="all providers down (simulated)",
                                   failed_providers=["groq", "hf"])
        return SimpleNamespace(ok=True, text=self.text, error="",
                               failed_providers=[])


class FakeTTS:
    def __init__(self, fail_key: bool = False):
        self.fail_key = fail_key

    def speak_as(self, text, voice_name, *, out_path=""):
        CALLS.append("tts.speak_as")
        if self.fail_key:
            raise KeyError(voice_name)
        sr, secs = 22050, 2.0
        n = int(sr * secs)
        frames = struct.pack(
            "<" + "h" * n,
            *(int(12000 * math.sin(2 * math.pi * 440 * i / sr))
              for i in range(n)))
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with wave.open(out_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(frames)
        return {"path": out_path, "sample_rate": sr, "backend": "fake"}


class FakeStudio:
    def __init__(self, fail_on: int = -1):
        self.fail_on = fail_on
        self.count = 0

    def generate(self, prompt, **kwargs):
        CALLS.append("studio.generate")
        assert prompt.strip(), "empty visual prompt"
        if self.count == self.fail_on:
            raise RuntimeError("simulated imggen outage")
        self.count += 1
        from PIL import Image
        dest = kwargs.get("save_to")
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        Image.new("RGB", (64, 113), (20, 30, 60)).save(dest)
        return [dest]


@dataclass
class FakeEditSpec:
    scenes: list = field(default_factory=list)
    audio: str = ""
    captions: str = ""
    music: str = ""
    beat_times: list = field(default_factory=list)
    output: str = ""
    width: int = 320
    height: int = 568
    fps: int = 15
    style: str = ""


def _fake_detect_beats(audio_path):
    CALLS.append("edit.detect_beats")
    return [0.5, 1.0, 1.5]


def _fake_render(spec):
    CALLS.append("edit.render")
    total = sum(float(s.get("duration", 1.0)) for s in spec.scenes) or 1.0
    import subprocess
    cmd = [shutil.which("ffmpeg"), "-y", "-v", "error",
           "-f", "lavfi", "-i",
           f"testsrc=size=320x568:rate=15:duration={total:.2f}"]
    if spec.audio and os.path.exists(spec.audio):
        cmd += ["-i", spec.audio, "-map", "0:v", "-map", "1:a", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", spec.output]
    subprocess.run(cmd, check=True, timeout=120)
    return spec.output


def _fake_build_captions(cues, out_path):
    CALLS.append("edit.build_captions")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (text, a, b) in enumerate(cues, 1):
            f.write(f"{i}\n00:00:{a:06.3f} --> 00:00:{b:06.3f}\n"
                    f"{text}\n\n".replace(".", ",", 1))
    return out_path


def _fake_music_bed(duration_s, out_path, seed=7):
    CALLS.append("edit.make_music_bed")
    sr = 22050
    n = max(1, int(duration_s * sr))
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)
    return out_path


def make_fake_edit():
    return SimpleNamespace(
        EditSpec=FakeEditSpec,
        render=_fake_render,
        detect_beats=_fake_detect_beats,
        build_captions=_fake_build_captions,
        make_music_bed=_fake_music_bed,
    )


class FakeNiche:
    name = "history"
    voice_style = "narrator"
    title_template = "{topic} — explained in 60 seconds"
    description_template = "{hook} {hashtags}"
    hashtags = ["#history", "#shorts"]
    cadence = {"posts_per_day": 1}

    def script_prompt(self, topic):
        CALLS.append("niche.script_prompt")
        return f"write a 60-second short-form video script about {topic}"

    def visual_strategy(self, script):
        CALLS.append("niche.visual_strategy")
        return [f"cinematic still for: {line[:40]}"
                for line in script.splitlines() if line.strip()][:6]


def make_pipeline(tmp_path, **kw):
    kw.setdefault("brain", FakeBrain())
    kw.setdefault("tts", FakeTTS())
    kw.setdefault("studio", FakeStudio())
    kw.setdefault("edit", make_fake_edit())
    kw.setdefault("get_niche", lambda name: FakeNiche())
    return ShortPipeline(runs_root=str(tmp_path / "runs"), **kw)


# ── tests ────────────────────────────────────────────────────────────


@needs_ffmpeg
def test_full_run_stage_order_and_artifacts(tmp_path):
    CALLS.clear()
    pipe = make_pipeline(tmp_path)
    job = pipe.plan("history", "why the library of alexandria burned",
                    platforms=["tiktok"])
    result = pipe.run(job)

    assert result.ok, result.error
    assert result.final_path and os.path.exists(result.final_path)
    assert result.final_path.endswith("final.mp4")

    # stage order: first occurrence of each stage's heavy call
    order = []
    for c in CALLS:
        if c not in order:
            order.append(c)
    expected = ["niche.script_prompt", "brain.complete", "tts.speak_as",
                "niche.visual_strategy", "studio.generate",
                "edit.detect_beats", "edit.render", "edit.build_captions",
                "edit.make_music_bed"]
    idx = [order.index(c) for c in expected]
    assert idx == sorted(idx), f"stage order violated: {order}"

    run_dir = tmp_path / "runs" / result.run_id
    for rel in ("script.txt", "script.json", "post_copy.json",
                "voiceover.wav", "voiceover_meta.json",
                "frames/scene_000.png", "frames.json",
                "beats.json", "edit_spec.json", "draft.mp4",
                "captions.srt", "captions_meta.json", "captioned.mp4",
                "music_bed.wav", "mixed_audio.wav", "music_meta.json",
                "final.mp4", "manifest.json"):
        assert (run_dir / rel).exists(), f"missing artifact {rel}"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert [s for s in STAGES] == list(manifest["stages"])
    assert all(v["status"] == "done" for v in manifest["stages"].values())
    assert all(v["ms"] >= 0 for v in manifest["stages"].values())

    # job persisted through to ready
    assert pipe.jobs.load(job.id).status == "ready"

    # post copy rendered from templates
    copy = json.loads((run_dir / "post_copy.json").read_text())
    assert "alexandria" in copy["title"].lower()
    assert "#history" in copy["hashtags"]


@needs_ffmpeg
def test_resume_from_failed_stage_skips_done_stages(tmp_path):
    CALLS.clear()
    studio = FakeStudio(fail_on=1)  # second image blows up
    pipe = make_pipeline(tmp_path, studio=studio)
    job = pipe.plan("history", "topic with failing visuals")

    result = pipe.run(job)
    assert not result.ok
    assert isinstance(result.error, StageError)
    assert result.error.stage == "visuals"
    assert result.error.code == "IMGGEN_FAILED"
    assert "Traceback" not in str(result.error)
    assert (tmp_path / "runs" / result.run_id / "error.json").exists()
    assert pipe.jobs.load(job.id).status == "failed"

    brain_calls = CALLS.count("brain.complete")
    tts_calls = CALLS.count("tts.speak_as")
    assert brain_calls == 1 and tts_calls == 1

    # fix the outage, resume — script/voiceover must NOT re-run
    studio.fail_on = -1
    resumed = pipe.resume(result.run_id)
    assert resumed.ok, resumed.error
    assert resumed.run_id == result.run_id
    assert CALLS.count("brain.complete") == brain_calls
    assert CALLS.count("tts.speak_as") == tts_calls
    assert os.path.exists(resumed.final_path)
    assert pipe.jobs.load(job.id).status == "ready"


@needs_ffmpeg
def test_brain_failure_is_clean_error(tmp_path):
    pipe = make_pipeline(tmp_path, brain=FakeBrain(fail=True))
    job = pipe.plan("history", "doomed topic")
    result = pipe.run(job)
    assert not result.ok
    err = result.error
    assert isinstance(err, StageError)
    assert err.stage == "script" and err.code == "BRAIN_UNAVAILABLE"
    assert "groq" in err.message  # names what was tried
    assert err.hint
    d = err.to_dict()
    assert set(d) == {"stage", "code", "message", "hint", "retriable"}


@needs_ffmpeg
def test_unknown_voice_is_clean_error(tmp_path):
    pipe = make_pipeline(tmp_path, tts=FakeTTS(fail_key=True))
    job = pipe.plan("history", "voiceless topic")
    result = pipe.run(job)
    assert not result.ok
    assert result.error.stage == "voiceover"
    assert result.error.code == "TTS_UNKNOWN_VOICE"


def test_resume_unknown_run_id(tmp_path):
    pipe = make_pipeline(tmp_path)
    with pytest.raises(FileNotFoundError):
        pipe.resume("run_nope_000000_abcdef")


def test_parse_script_fallback_no_markers():
    parsed = parse_script("Just a paragraph.\n\nAnd another one.")
    assert len(parsed["scenes"]) == 2
    assert parsed["scenes"][0]["say"].startswith("Just a paragraph")


def test_parse_script_markers():
    parsed = parse_script(SCRIPT_FIXTURE)
    assert parsed["hook"].startswith("Your brain deletes")
    assert len(parsed["scenes"]) == 3
    assert parsed["scenes"][0]["show"].startswith("a giant human eye")
    assert "Lying" in parsed["title"]


def test_job_persistence_round_trip(tmp_path):
    pipe = make_pipeline(tmp_path)
    job = pipe.plan("history", "round trip", platforms=["yt-shorts"])
    loaded = pipe.jobs.load(job.id)
    assert loaded.to_dict() == job.to_dict()
    loaded.status = "posted"
    pipe.jobs.save(loaded)
    assert pipe.jobs.load(job.id).status == "posted"
    assert [j.id for j in pipe.jobs.list("posted")] == [job.id]
    with pytest.raises(FileNotFoundError):
        pipe.jobs.load("job_nope")


def test_calendar_cadence_enforcement(tmp_path):
    pipe = make_pipeline(tmp_path)
    cal = pipe.calendar
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    p1 = cal.add("history", "topic one", scheduled_for=past)
    p2 = cal.add("history", "topic two", scheduled_for=past)
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    p3 = cal.add("history", "topic three", scheduled_for=future)

    due = cal.due()
    assert [p.id for p in due] == [p1.id]  # posts_per_day=1, future not due
    assert p2.id not in [p.id for p in due]
    assert p3.id not in [p.id for p in due]

    cal.mark(p1.id, "posted")
    assert cal.due() == []  # cadence exhausted for today

    # a fresh niche gets its own allowance
    p4 = cal.add("science", "topic four", scheduled_for=past)
    assert [p.id for p in cal.due()] == [p4.id]

    with pytest.raises(KeyError):
        cal.mark("post_nope", "posted")


@needs_ffmpeg
def test_real_sibling_niche_end_to_end(tmp_path):
    """The pipeline drives a REAL sibling niche plugin (not a fake).

    Uses the real ``niches.get_niche('motivation')`` — VisualPlan,
    VoiceSpec candidates, tags_for, cadence_spec — with only the heavy
    engines (brain/tts/imggen/edit) mocked.
    """
    from nomorals.media.contentops.niches import get_niche
    CALLS.clear()
    pipe = make_pipeline(tmp_path, get_niche=get_niche)
    job = pipe.plan("motivation", "discipline beats talent",
                    platforms=["tiktok"])
    result = pipe.run(job)
    assert result.ok, result.error
    assert os.path.exists(result.final_path)

    run_dir = tmp_path / "runs" / result.run_id
    copy = json.loads((run_dir / "post_copy.json").read_text())
    assert "#motivation" in copy["hashtags"]  # real tags_for()
    assert "discipline" in copy["title"].lower()  # real title_for()

    vo_meta = json.loads((run_dir / "voiceover_meta.json").read_text())
    # VoiceSpec candidates tried in order; FakeTTS accepts the first
    assert vo_meta["voice"] in ("devon-deep", "default", "narrator")

    frames = json.loads((run_dir / "frames.json").read_text())
    assert len(frames) >= 1
    # real VisualPlan motions flowed into the edit spec
    spec = json.loads((run_dir / "edit_spec.json").read_text())
    assert all(s["effect"] in ("kenburns", "static")
               for s in spec["scenes"])


def test_estimator_shape_and_honesty(tmp_path):
    pipe = make_pipeline(tmp_path)
    est = pipe.estimate("history", "some topic", scenes=5)
    assert [r["stage"] for r in est["stages"]] == list(STAGES)
    for row in est["stages"]:
        assert row["time_low_s"] <= row["time_high_s"]
        assert row["time_low_s"] >= 0
        assert row["cost_usd_low"] == 0.0 and row["cost_usd_high"] == 0.0
        assert row["basis"]  # every row says where the number came from
    assert est["total_time_low_s"] == round(
        sum(r["time_low_s"] for r in est["stages"]), 1)
    assert "guess" in est["stages"][2]["basis"]  # no past runs → guesses
    assert any("$0" in n for n in est["notes"])
