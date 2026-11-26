"""Gapfix 163 — TrendClip port into Devon.

The port: discovery (YouTube/yt-dlp, views-per-hour), ingest, Faster
Whisper transcription with word timestamps, moment scoring with BOTH
rungs (deterministic formula + brain), vertical render (1080x1920,
-14 LUFS), validation, export package, museum shelf, /clip + /trend,
and the shorts-publisher sweep using the pre-rendered clip.

Failing-first: on a pristine at 93afabe3 this file collection-errors —
the organ does not exist there. Every heavy dependency (yt-dlp, STT,
OpenCV, FFmpeg) is faked or injected here; the real-video acceptance
run lives in ~/workspace/trendclip-verify/acceptance.py, not in this
file.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomorals.media.trendclip import (  # noqa: E402
    TrendclipPipeline,
    assemble_drafts,
    brain_pick_moments,
    build_pick_prompt,
    discover_youtube_trends,
    inspect_youtube_url,
    load_topic,
    normalize_words,
    prerendered_clip_for_job,
    rank_feed_items,
    score_feed_item,
    select_moments,
    topic_from_dict,
    topic_from_query,
    transcribe_audio,
    trendclip_exhibitor,
    trendclip_post_copy,
    validate_picks,
    validate_rights,
    write_ass_captions,
    zscore,
)
from nomorals.media.trendclip.export_package import (  # noqa: E402
    export_package,
    write_srt,
)
from nomorals.media.trendclip.jobs import JobStore, new_job_id  # noqa: E402
from nomorals.media.trendclip.render import visual_signals  # noqa: E402
from nomorals.media.trendclip.topics import TopicConfig  # noqa: E402
from nomorals.media.trendclip.validate import (  # noqa: E402
    probe_video,
    validate_clip,
    write_validation,
)
from nomorals.social.chat.control import (  # noqa: E402
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
)
from nomorals.tools.registry import ToolRegistry  # noqa: E402

# ── topics ─────────────────────────────────────────────────────────────

def test_topic_from_query_builds_keywords_and_slug():
    topic = topic_from_query("GTA 6 gameplay moments", strategy="gaming")
    assert topic.slug == "gta-6-gameplay-moments"
    assert topic.strategy == "gaming"
    assert "GTA" in topic.keywords and "gameplay" in topic.keywords
    assert topic.validate() == []


def test_topic_from_query_drops_stopwords_and_unknown_strategy():
    topic = topic_from_query("show me the best clips", strategy="nope")
    assert topic.strategy == "conversation"
    assert "the" not in [k.lower() for k in topic.keywords]
    assert topic.validate() == []


def test_topic_from_dict_uses_gta6_shape_and_ignores_scenes():
    # The source's gta6.json shape: slug/topic_name/keywords/exclusions
    # plus a graphical "scenes" block Devon does not port (makers own it).
    topic = topic_from_dict({
        "slug": "gta6",
        "topic_name": "GTA 6",
        "language": "en",
        "keywords": ["GTA 6", "gameplay", "trailer"],
        "exclusions": ["leak", "fake"],
        "strategy": "gaming",
        "min_clip_seconds": 15,
        "max_clip_seconds": 120,
        "target_clip_seconds": 60,
        "channels": ["Rockstar Games"],
        "scenes": [{"kind": "title", "text": "GTA 6"}],
        "metadata": {"game": "GTA 6"},
    })
    assert topic.slug == "gta6"
    assert topic.channels == ["Rockstar Games"]
    assert topic.validate() == []


def test_topic_validate_names_problems(tmp_path):
    bad = TopicConfig(slug="x", topic_name="X", keywords=[],
                      strategy="nope", min_clip_seconds=0)
    problems = bad.validate()
    assert any("keywords is empty" in p for p in problems)
    assert any("strategy" in p for p in problems)
    path = tmp_path / "topic.json"
    path.write_text(json.dumps({"slug": "x", "topic_name": "X",
                                "keywords": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        load_topic(path)


# ── rights ─────────────────────────────────────────────────────────────

def test_rights_third_party_is_a_review_draft_not_creator_owned():
    result = validate_rights({"category": "third_party_review_draft"},
                             source_url="https://www.youtube.com/watch?v=x")
    assert result["ok"] is True
    assert "review draft" in result["label"]
    assert "creator-owned" not in result["label"]


def test_rights_unknown_and_research_only_block_render():
    assert validate_rights(None)["ok"] is False
    assert validate_rights({"category": "research_only"})["ok"] is False
    assert validate_rights({"category": "bogus"})["ok"] is False


# ── jobs ───────────────────────────────────────────────────────────────

def test_jobs_create_save_get_stage_list(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(topic="gta6", source_url="https://example.test/v")
    assert job.job_id.startswith("tc_")
    assert job.stage == "queued"
    store.set_stage(job, "analyzing")
    again = store.get(job.job_id)
    assert again.stage == "analyzing"
    assert again.source_url == "https://example.test/v"
    assert [j.job_id for j in store.list(stage="analyzing")] == [job.job_id]
    assert store.list(stage="ready") == []
    with pytest.raises(KeyError):
        store.get("tc_nope123")
    with pytest.raises(ValueError):
        store.set_stage(job, "nope")
    assert new_job_id().startswith("tc_")


# ── scoring (the deterministic rung) ───────────────────────────────────

def test_zscore_flat_series_is_zero():
    assert zscore([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]
    assert zscore([]) == []


def test_select_moments_finds_the_peak_and_does_not_overlap():
    loud = [-30.0] * 120
    loud[40] = -12.0
    loud[41] = -14.0
    loud[90] = -13.0
    words = [{"w": "and", "s": 40.0, "e": 40.3},
             {"w": "here", "s": 40.4, "e": 40.7},
             {"w": "we", "s": 41.0, "e": 41.2},
             {"w": "go", "s": 41.3, "e": 41.6},
             {"w": "now", "s": 90.0, "e": 90.3}]
    moments = select_moments(loudness=loud, words=words, strategy="gaming",
                             target_seconds=30, min_seconds=15,
                             max_seconds=60, max_moments=5)
    assert moments
    # Every picked window contains a real peak second (40, 41 or 90).
    for m in moments:
        assert any(m["start"] <= peak < m["end"]
                   for peak in (40.0, 41.0, 90.0)), m
    for a, b in zip(moments, moments[1:], strict=False):
        assert a["end"] <= b["start"] or b["end"] <= a["start"]
    for moment in moments:
        assert 0.0 <= moment["quality_score"] <= 10.0
        assert set(moment["components"]) == {
            "clean_start", "clean_end", "hook", "payoff", "gaming_event"}


def test_select_moments_empty_loudness_returns_nothing():
    assert select_moments(loudness=[], words=[]) == []


def test_assemble_drafts_tiers_by_quality():
    drafts = assemble_drafts([
        {"n": 1, "quality_score": 8.5, "start": 0, "end": 60,
         "why": [], "warnings": []},
        {"n": 2, "quality_score": 7.0, "start": 60, "end": 120,
         "why": [], "warnings": []},
        {"n": 3, "quality_score": 6.0, "start": 120, "end": 180,
         "why": [], "warnings": []},
        {"n": 4, "quality_score": 2.0, "start": 180, "end": 240,
         "why": [], "warnings": ["too quiet"]},
    ])
    assert [d["tier"] for d in drafts] == [
        "top_pick", "high_energy", "quick_hit", "rejected"]
    assert [d["verdict"] for d in drafts] == [
        "Top pick", "High energy", "Quick hit", "Rejected"]


def test_score_feed_item_bounds():
    assert score_feed_item(relevance=1, freshness=1, reliability=1) == 1.0
    assert score_feed_item(relevance=0, freshness=0, reliability=0) == 0.0
    assert score_feed_item(relevance=2, freshness=-1, reliability=0.5) == 0.6


# ── discovery ──────────────────────────────────────────────────────────

def _topic(**overrides):
    data = {"slug": "gta6", "topic_name": "GTA 6",
            "keywords": ["GTA", "gameplay"], "exclusions": ["leak"],
            "strategy": "gaming"}
    data.update(overrides)
    return TopicConfig(**data)


def test_rank_feed_items_views_per_hour_first():
    items = rank_feed_items([
        {"title": "GTA gameplay", "url": "https://www.youtube.com/watch?v=a",
         "channel": "Rockstar", "published_at": "", "views_per_hour": 10.0},
        {"title": "GTA gameplay best", "url": "https://www.youtube.com/watch?v=b",
         "channel": "Other", "published_at": "", "views_per_hour": 900.0},
    ], _topic())
    assert items[0]["url"].endswith("?v=b")
    assert items[0]["feed_score"] >= 0


class _Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_discover_youtube_trends_fake_runner_ranks_and_excludes():
    payload = {"entries": [
        {"id": "aaa", "title": "GTA gameplay best moments",
         "channel": "Rockstar", "view_count": 1000,
         "upload_date": "20261125"},
        {"id": "bbb", "title": "GTA leak gameplay",
         "channel": "Other", "view_count": 999999,
         "upload_date": "20261125"},
        {"id": "ccc", "title": "cooking show",
         "channel": "Other", "view_count": 5,
         "upload_date": "20261125"},
    ]}
    seen = {}
    def runner(cmd, **kw):
        seen["cmd"] = cmd
        return _Proc(stdout=json.dumps(payload))
    items = discover_youtube_trends(_topic(), limit=10, runner=runner,
                                    ytdlp="yt-dlp")
    assert "ytsearch10:" in " ".join(seen["cmd"])
    urls = [i["url"] for i in items]
    assert not any("bbb" in u for u in urls)  # exclusion filtered
    assert any("aaa" in u for u in urls)
    assert items[0]["views_per_hour"] >= items[-1]["views_per_hour"]


def test_discover_youtube_trends_reports_yt_dlp_failure():
    items = discover_youtube_trends(
        _topic(), runner=lambda *a, **k: _Proc(stderr="boom", returncode=1))
    assert items and "error" in items[0]


def test_inspect_youtube_url_fake_runner_and_disallowed_host():
    def runner(cmd, **kw):
        return _Proc(stdout=json.dumps({
            "title": "GTA gameplay", "uploader": "Rockstar",
            "duration": 120, "view_count": 42,
            "webpage_url": "https://www.youtube.com/watch?v=aaa"}))
    info = inspect_youtube_url("https://www.youtube.com/watch?v=aaa",
                               runner=runner)
    assert info["ok"] is True
    assert info["duration"] == 120.0
    assert info["view_count"] == 42
    refused = inspect_youtube_url("https://evil.example/v")
    assert refused["ok"] is False


# ── brain rung (injected pick_fn, no model needed) ─────────────────────

def test_brain_pick_prompt_digests_transcript_and_candidates():
    prompt = build_pick_prompt(
        topic="GTA 6",
        transcript="0s: and here we go\n8s: the city wakes up",
        candidates=[{"n": 1, "start": 0, "end": 60, "quality_score": 8.2,
                     "why": ["loudness peak 41s"]}])
    assert "GTA 6" in prompt
    assert "the city wakes up" in prompt
    assert "#1" in prompt
    assert "ONLY JSON" in prompt


def test_validate_picks_keeps_only_real_numbered_reasons():
    candidates = [{"n": 1, "start": 0, "end": 60},
                  {"n": 2, "start": 60, "end": 120}]
    picks = validate_picks({"picks": [
        {"n": 1, "reason": "the hook lands"},
        {"n": 1, "reason": "duplicate"},
        {"n": 99, "reason": "not a candidate"},
        {"n": 2, "reason": ""},
        {"n": "x", "reason": "not a number"},
    ]}, candidates)
    assert [p["n"] for p in picks] == [1]
    assert picks[0]["reason"] == "the hook lands"
    assert picks[0]["source"] == "brain"
    assert validate_picks({"picks": []}, candidates) == []
    assert validate_picks("nope", candidates) == []


def test_brain_pick_moments_fake_pick_and_no_brain():
    candidates = [{"n": 1, "start": 0, "end": 60, "quality_score": 8.0,
                   "why": []}]
    got = brain_pick_moments(
        lambda prompt: {"picks": [{"n": 1, "reason": "the hook lands"}]},
        topic="GTA 6", transcript="0s: hello", candidates=candidates)
    assert got["error"] is None
    assert [p["n"] for p in got["picks"]] == [1]
    assert got["picks"][0]["candidate"]["n"] == 1
    missing = brain_pick_moments(
        lambda prompt: None, topic="GTA 6", transcript="0s: hello",
        candidates=candidates)
    assert missing["picks"] == []
    assert "no brain" in (missing["error"] or "")
    bad = brain_pick_moments(
        lambda prompt: "{not json", topic="GTA 6", transcript="0s: hello",
        candidates=candidates)
    assert bad["picks"] == []
    assert brain_pick_moments(
        lambda prompt: {"picks": [{"n": 1, "reason": "x"}]},
        topic="GTA 6", transcript="", candidates=[])["picks"] == []


# ── transcribe ─────────────────────────────────────────────────────────

def test_normalize_words_flattens_segments():
    result = {"text": "hello world", "segments": [
        {"words": [{"word": " hello", "start": 0.1, "end": 0.4},
                   {"word": "world", "start": 0.5, "end": 0.9}]},
        {"words": [{"word": " ", "start": 1.0, "end": 1.1}]},
    ]}
    words = normalize_words(result)
    assert words == [{"w": "hello", "s": 0.1, "e": 0.4},
                     {"w": "world", "s": 0.5, "e": 0.9}]


def test_transcribe_audio_honors_model_via_backend_seam(monkeypatch):
    """The requested model must reach the backend, not the transcribe
    call: UniversalSTT takes no model argument and the faster-whisper
    backend reads FASTER_WHISPER_MODEL at construction (its default is
    large-v3-turbo); passing model= to transcribe() would TypeError.
    UniversalSTT constructs its backend LAZILY, on the first
    transcribe() call — so the env var must be set around that call,
    not around the UniversalSTT construction (the first version of
    this fix scoped it to the constructor and silently loaded the
    default). The env var is restored after."""
    import os

    seen: dict = {}

    class _FakeSTT:
        def __init__(self, backend="auto"):
            # Lazy like the real one: nothing is built here.
            seen["env_at_construction"] = os.environ.get(
                "FASTER_WHISPER_MODEL")
            seen["backend"] = backend

        def transcribe(self, audio_path, language="en", **kwargs):
            # The backend is built here, in the real UniversalSTT.
            seen["env_at_transcribe"] = os.environ.get(
                "FASTER_WHISPER_MODEL")
            seen["kwargs"] = kwargs
            return {"text": "hi", "language": language,
                    "backend": "faster-whisper",
                    "segments": [{"words": [{"word": "hi", "start": 0.0,
                                             "end": 0.3}]}]}

    monkeypatch.setattr(
        "nomorals.voice.stt.UniversalSTT", _FakeSTT)
    monkeypatch.delenv("FASTER_WHISPER_MODEL", raising=False)
    out = transcribe_audio("/nonexistent/x.wav", language="en",
                           model="tiny")
    assert seen["env_at_construction"] is None
    assert seen["env_at_transcribe"] == "tiny"
    assert seen["backend"] == "faster-whisper"
    assert "model" not in seen["kwargs"]
    assert seen["kwargs"]["word_timestamps"] is True
    assert out["model"] == "tiny"
    assert out["words"] == [{"w": "hi", "s": 0.0, "e": 0.3}]
    assert "FASTER_WHISPER_MODEL" not in os.environ


def test_render_samples_by_time_not_source_cadence(monkeypatch, tmp_path):
    """A 24fps source must still render a real-time 30fps clip: the
    output frame count is duration * 30 (time-based sampling), not
    duration * source-fps. The first version read frames sequentially
    while declaring them 30fps, so a 60s Big Buck Bunny moment (24fps)
    came out as a 48s clip running 25% fast with the audio truncated
    — caught by the acceptance run, not by a test."""
    import numpy as np

    import nomorals.media.trendclip.render as render_mod

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    seeks: list = []

    class _FakeCap:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == "fps":
                return 24.0
            if prop == "width":
                return 1280.0
            if prop == "height":
                return 720.0
            return 0.0

        def set(self, prop, value):
            if prop == "msec":
                seeks.append(value)
            return True

        def read(self):
            return True, frame

        def release(self):
            return None

    class _FakeStdin:
        def __init__(self):
            self.writes = 0

        def write(self, data):
            self.writes += 1

        def close(self):
            return None

    class _FakeProc:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.returncode = 0
            self.cmd = None

        def wait(self, timeout=None):
            return None

    proc = _FakeProc()

    def _fake_popen(cmd, **kwargs):
        proc.cmd = cmd
        return proc

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS="fps", CAP_PROP_FRAME_WIDTH="width",
        CAP_PROP_FRAME_HEIGHT="height", CAP_PROP_POS_FRAMES="frames",
        CAP_PROP_POS_MSEC="msec",
        COLOR_BGR2GRAY="gray", THRESH_BINARY=0, RETR_EXTERNAL=0,
        CHAIN_APPROX_SIMPLE=0, FONT_HERSHEY_SIMPLEX=0, LINE_AA=0,
        VideoCapture=lambda path: _FakeCap(),
        resize=lambda f, size: np.zeros((size[1], size[0], 3),
                                        dtype=np.uint8),
        GaussianBlur=lambda f, k, s: f,
        addWeighted=lambda a, x, b, y, z: a,
        cvtColor=lambda f, code: np.zeros((90, 160), dtype=np.uint8),
        threshold=lambda f, t, m, c: (0, f),
        findContours=lambda *a, **k: ([], None),
        contourArea=lambda c: 0.0,
        boundingRect=lambda c: (0, 0, 0, 0),
        getTextSize=lambda *a, **k: ((100, 20), 0),
        putText=lambda *a, **k: None,
    )
    monkeypatch.setattr(render_mod, "_require_cv2",
                        lambda: fake_cv2)
    out = render_mod.render_vertical_clip(
        video_path="/fake/proxy.mp4", audio_path="/fake/section.wav",
        words=[], start=5.0, end=7.0,
        output_path=tmp_path / "clip.mp4", layout="balanced",
        hook_text="", runner=_fake_popen)
    assert out["ok"] is True
    assert out["frames"] == 60  # 2.0s * 30fps, NOT 2.0s * 24fps = 48
    assert out["duration"] == 2.0
    assert proc.stdin.writes == 60
    assert proc.cmd[proc.cmd.index("-r") + 1] == "30"
    # Time-based seeks: first at the section start, stepping 1/30s.
    assert len(seeks) == 60
    assert abs(seeks[0] - 5000.0) < 0.01
    assert abs(seeks[1] - (5000.0 + 1000.0 / 30.0)) < 0.01
    assert abs(seeks[-1] - (5000.0 + 59 * 1000.0 / 30.0)) < 0.01


def test_provision_pins_pyav_below_14(monkeypatch):
    """faster-whisper 1.2.1 calls av.open(..., metadata_errors=...)
    and PyAV 14+ dropped that parameter — an unbounded install
    resolves av 19 and every transcription dies with TypeError.
    Both the self-provisioner and pyproject's media extra must carry
    the av>=11,<14 bound."""
    import importlib.util
    import tomllib

    import nomorals.media.trendclip.provision as provision

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: None if name == "faster_whisper"
        else real_find_spec(name))
    monkeypatch.setattr(provision, "_under_pytest", lambda: False)
    seen: dict = {}

    def _fake_run(argv, **kwargs):
        seen["argv"] = argv
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)
    assert provision.ensure_faster_whisper() is False
    argv = seen["argv"]
    assert "faster-whisper>=1.1" in argv
    assert "av>=11,<14" in argv
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml")
        .read_text(encoding="utf-8"))
    media = pyproject["project"]["optional-dependencies"]["media"]
    assert "av>=11,<14" in media
    assert "faster-whisper>=1.1" in media


# ── render / validate / export (pure parts) ────────────────────────────

def _write_wav(path: Path, *, seconds: int, amplitude: int,
               rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = struct.pack(f"<{seconds * rate}h",
                             *([amplitude] * (seconds * rate)))
        wav.writeframes(frames)


def test_loudness_series_full_scale_is_zero_db_and_silence_is_floor(tmp_path):
    loud_path = tmp_path / "loud.wav"
    _write_wav(loud_path, seconds=2, amplitude=32767)
    series = TrendclipPipeline.loudness_series(loud_path)
    assert len(series) == 2
    assert all(abs(v) < 0.5 for v in series)
    quiet_path = tmp_path / "quiet.wav"
    _write_wav(quiet_path, seconds=1, amplitude=0)
    assert TrendclipPipeline.loudness_series(quiet_path) == [-60.0]


def test_visual_signals_without_opencv_is_zeros():
    assert visual_signals("/nonexistent/video.mp4", 5) == [0.0] * 5
    assert visual_signals("/nonexistent/video.mp4", 0) == []


def test_write_ass_captions_word_timed(tmp_path):
    out = write_ass_captions(
        [{"w": "hello", "s": 1.0, "e": 1.4},
         {"w": "world", "s": 1.5, "e": 1.9}],
        tmp_path / "caps.ass", start=1.0)
    text = out.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in text
    assert "0:00:00.00" in text and "hello" in text
    assert "0:00:00.90" in text and "world" in text


def _ffprobe_good():
    return _Proc(stdout=json.dumps({
        "streams": [
            {"codec_type": "video", "codec_name": "h264",
             "width": 1080, "height": 1920, "pix_fmt": "yuv420p",
             "avg_frame_rate": "30/1"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "12.0"},
    }))


def test_validate_clip_passes_on_clean_fake_probe():
    def runner(cmd, **kw):
        if cmd[0] == "ffprobe":
            return _ffprobe_good()
        return _Proc(stderr="")  # detect filters: no hits
    report = validate_clip("/tmp/fake.mp4", runner=runner)
    assert report["passed"] is True
    assert report["problems"] == []
    assert report["probe"]["width"] == 1080


def test_validate_clip_fails_on_black_frames_and_wrong_size():
    def runner(cmd, **kw):
        if cmd[0] == "ffprobe":
            bad = _ffprobe_good()
            data = json.loads(bad.stdout)
            data["streams"][0]["width"] = 720
            return _Proc(stdout=json.dumps(data))
        if "blackdetect" in " ".join(cmd):
            return _Proc(stderr="black_start:0 black_end:1\nblack_start:2")
        return _Proc(stderr="")
    report = validate_clip("/tmp/fake.mp4", runner=runner)
    assert report["passed"] is False
    assert any("720" in p for p in report["problems"])
    assert any("black" in p for p in report["problems"])


def test_probe_video_reports_unparseable_ffprobe():
    report = probe_video("/tmp/fake.mp4",
                         runner=lambda *a, **k: _Proc(stdout="{nope"))
    assert report["ok"] is False


def test_write_validation_round_trips(tmp_path):
    report = {"passed": True, "problems": []}
    out = write_validation(report, tmp_path / "v.json")
    assert json.loads(out.read_text(encoding="utf-8")) == report


def test_write_srt_groups_words_into_lines(tmp_path):
    words = [{"w": f"w{i}", "s": float(i), "e": float(i) + 0.4}
             for i in range(12)]
    out = write_srt(words, tmp_path / "caps.srt")
    text = out.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:03,400" in text
    assert text.count("-->") >= 2


def test_export_package_manifest_checksums_and_provenance(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-clip-bytes")
    words = [{"w": "hello", "s": 0.0, "e": 0.4}]
    package = export_package(
        clip_path=clip, words=words, job_id="tc_test1", topic="gta6",
        source_url="https://www.youtube.com/watch?v=aaa",
        rights_label="review draft — third-party media",
        moment={"n": 1, "start": 0.0, "end": 30.0, "quality_score": 8.2,
                "strategy": "gaming", "brain_pick": True},
        destinations=["youtube", "tiktok"],
        copy={"youtube": {"title": "GTA 6 — clip", "description": "watch",
                          "hashtags": ["gta6"]}},
        out_dir=tmp_path / "exports" / "tc_test1")
    manifest = json.loads(Path(package["manifest"]).read_text(
        encoding="utf-8"))
    assert manifest["files"]["clip"]["bytes"] == len(b"fake-clip-bytes")
    assert len(manifest["files"]["clip"]["sha256"]) == 64
    provenance = json.loads(Path(package["provenance"]).read_text(
        encoding="utf-8"))
    assert provenance["job_id"] == "tc_test1"
    assert provenance["moment"]["brain_pick"] is True
    assert provenance["rights"].startswith("review draft")
    assert package["platforms"]["youtube"]["title"] == "GTA 6 — clip"
    assert Path(package["captions"]).is_file()


# ── pipeline with fakes (offline end-to-end) ───────────────────────────

def _fake_transcriber(audio_path, *, language="en", model="small"):
    words = []
    for i in range(120):
        words.append({"w": f"word{i}", "s": float(i), "e": float(i) + 0.4})
    return {"text": " ".join(w["w"] for w in words), "words": words,
            "language": language, "backend": "faster-whisper",
            "model": model}


def _fake_yt_dlp_runner(job_dir: Path):
    def runner(cmd, **kw):
        if "-J" in cmd:
            if "watch?v=" in " ".join(cmd):
                return _Proc(stdout=json.dumps({
                    "title": "GTA gameplay best moments",
                    "uploader": "Rockstar", "duration": 120,
                    "view_count": 5000,
                    "webpage_url": "https://www.youtube.com/watch?v=aaa"}))
            return _Proc(stdout=json.dumps({"entries": []}))
        # a download: create the file yt-dlp would have written
        out = None
        for i, token in enumerate(cmd):
            if token == "-o" and i + 1 < len(cmd):
                out = cmd[i + 1]
        if out:
            target = Path(str(out).replace("%(ext)s", "mp4"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake-media")
        return _Proc(stdout="", stderr="")
    return runner


def _fake_ffmpeg(monkeypatch):
    """ffmpeg stub: creates the output file, returns rc 0. A .wav output
    is a real wav (the loudness reader parses it)."""
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if isinstance(cmd, list) and cmd and cmd[0] == "ffmpeg":
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.suffix == ".wav":
                _write_wav(out, seconds=120, amplitude=20000)
            else:
                out.write_bytes(b"fake-output")
            return _Proc(stdout="", stderr="", returncode=0)
        return real_run(cmd, **kw)
    monkeypatch.setattr(subprocess, "run", fake_run)


def test_pipeline_analyze_both_rungs_with_fakes(tmp_path, monkeypatch):
    root = tmp_path / "trendclip"
    pipe = TrendclipPipeline(
        root, runner=_fake_yt_dlp_runner(root),
        transcriber=_fake_transcriber,
        pick_fn=lambda prompt: {"picks": [
            {"n": 1, "reason": "the hook lands in the first seconds"}]})
    _fake_ffmpeg(monkeypatch)
    job = pipe.analyze("https://www.youtube.com/watch?v=aaa",
                       _topic(), model="tiny")
    assert job.stage == "ready", job.error
    receipt = job.receipt
    assert receipt["source"]["title"] == "GTA gameplay best moments"
    assert receipt["rights"]["label"].startswith("review draft")
    assert receipt["transcript"]["words"] == 120
    assert receipt["moments"], "no moments scored"
    assert receipt["brain"]["ran"] is True
    assert [p["n"] for p in receipt["brain"]["picks"]] == [1]
    assert receipt["moments"][0].get("brain_pick") is True
    assert receipt["editorial_summary"]
    assert (root / "jobs" / (job.job_id + ".json")).is_file()


def test_pipeline_analyze_without_brain_records_the_note(tmp_path, monkeypatch):
    root = tmp_path / "trendclip"
    pipe = TrendclipPipeline(root, runner=_fake_yt_dlp_runner(root),
                             transcriber=_fake_transcriber)
    _fake_ffmpeg(monkeypatch)
    job = pipe.analyze("https://www.youtube.com/watch?v=aaa", _topic(),
                       brain=True)
    assert job.stage == "ready", job.error
    assert job.receipt["brain"]["ran"] is False
    assert "no brain consult" in (job.receipt["brain"]["error"] or "")


def test_pipeline_render_full_fake_path_and_queue(tmp_path, monkeypatch):
    root = tmp_path / "trendclip"
    shorts_root = tmp_path / "shorts_runs"
    pipe = TrendclipPipeline(
        root, runner=_fake_yt_dlp_runner(root),
        transcriber=_fake_transcriber,
        pick_fn=lambda prompt: {"picks": [
            {"n": 1, "reason": "the hook lands"}]},
        copy_fn=lambda prompt: {"youtube": {
            "title": "GTA 6 — the hook lands",
            "description": "watch to the end", "hashtags": ["gta6"]}},
        shorts_runs_root=shorts_root)
    _fake_ffmpeg(monkeypatch)
    job = pipe.analyze("https://www.youtube.com/watch?v=aaa", _topic())
    assert job.stage == "ready", job.error
    pick = job.receipt["moments"][0]["n"]

    def fake_render(*, video_path, audio_path, words, start, end,
                    output_path, layout, hook_text, runner=None):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-clip")
        return {"ok": True, "path": str(out), "duration": end - start,
                "frames": 900, "layout": layout, "crop": [0, 0, 100, 100]}

    def fake_validate(path, *, runner=None):
        return {"ok": True, "passed": True, "problems": [],
                "probe": {"width": 1080, "height": 1920},
                "quality": {}}
    monkeypatch.setattr(
        "nomorals.media.trendclip.pipeline._render.render_vertical_clip",
        fake_render)
    monkeypatch.setattr(
        "nomorals.media.trendclip.pipeline._validate.validate_clip",
        fake_validate)
    job = pipe.render(job.job_id, pick, layout="punch",
                      destinations=["youtube"])
    assert job.stage == "complete", job.error
    render_info = job.receipt["render"]
    assert Path(render_info["clip_path"]).is_file()
    assert render_info["validation"]["passed"] is True
    assert Path(render_info["package"]["manifest"]).is_file()
    assert render_info["package"]["platforms"]["youtube"]["title"] == \
        "GTA 6 — the hook lands"

    # The museum shelf sees the rendered clip.
    cards = trendclip_exhibitor(root)
    assert len(cards) == 1
    assert cards[0]["shelf"] == "trendclip"
    assert cards[0]["kind"] == "clip"
    assert cards[0]["provenance"]["job_id"] == job.job_id

    # Queue it: the sweep can use the pre-rendered clip.
    queued = pipe.queue(job.job_id, platforms=["youtube"],
                        scheduled_for="")
    assert queued["post_id"]
    assert prerendered_clip_for_job(root, job.job_id) == \
        render_info["clip_path"]
    copy = trendclip_post_copy(root, job.job_id)
    assert copy is not None
    assert copy["title"] == "GTA 6 — the hook lands"
    assert "word0" in copy["script"]

    # Status + list.
    status = pipe.status(job.job_id)
    assert status["stage"] == "complete"
    assert status["validation_passed"] is True
    assert len(pipe.list_jobs()) == 1
    assert pipe.list_jobs(stage="ready") == []


def test_pipeline_render_refuses_blocked_rights_and_unknown_pick(
        tmp_path, monkeypatch):
    root = tmp_path / "trendclip"
    pipe = TrendclipPipeline(root, runner=_fake_yt_dlp_runner(root),
                             transcriber=_fake_transcriber)
    _fake_ffmpeg(monkeypatch)
    job = pipe.analyze("https://www.youtube.com/watch?v=aaa", _topic(),
                       rights_category="research_only")
    assert job.stage == "ready", job.error
    with pytest.raises(RuntimeError, match="render blocked"):
        pipe.render(job.job_id, job.receipt["moments"][0]["n"])
    with pytest.raises(RuntimeError, match="no moment"):
        pipe.render(job.job_id, 999)
    with pytest.raises(KeyError):
        pipe.status("tc_nope123")


def test_prerendered_clip_only_for_complete_validated_jobs(tmp_path):
    root = tmp_path / "trendclip"
    store = JobStore(root)
    job = store.create(topic="gta6")
    assert prerendered_clip_for_job(root, job.job_id) == ""
    assert trendclip_post_copy(root, job.job_id) is None
    job.receipt = {"render": {"clip_path": str(root / "clips" / "x.mp4")},
                   "validation": {"passed": False}}
    store.set_stage(job, "complete")
    assert prerendered_clip_for_job(root, job.job_id) == ""


def test_exhibitor_missing_root_exhibits_nothing(tmp_path):
    assert trendclip_exhibitor(tmp_path / "nope") == []


# ── tools ──────────────────────────────────────────────────────────────

def _ctx(home):
    return SimpleNamespace(settings=SimpleNamespace(home=str(home)))


def test_registry_count_is_800_and_trendclip_tools_register(tmp_path):
    # 794, not 789: the trendclip port added 5 tools (trendclip_discover,
    # trendclip_analyze, trendclip_render, trendclip_status,
    # trendclip_queue).
    reg = ToolRegistry(context=_ctx(tmp_path))
    reg.register_builtins()
    assert len(reg._tools) == 805
    for name in ("trendclip_discover", "trendclip_analyze",
                 "trendclip_render", "trendclip_status", "trendclip_queue"):
        assert name in reg.names(), name


def test_trendclip_status_tool_lists_jobs(tmp_path):
    reg = ToolRegistry(context=_ctx(tmp_path))
    reg.register_builtins()
    out = reg.call("trendclip_status")
    assert out.ok, out
    assert out.value["jobs"] == []
    missing = reg.call("trendclip_status", job_id="tc_nope123")
    assert missing.ok
    assert missing.value["ok"] is False


# ── commands ───────────────────────────────────────────────────────────

def test_clip_and_trend_commands_registered():
    assert CONTROL_COMMANDS["clip"] == (0, None)
    assert CONTROL_COMMANDS["trend"] == (1, None)
    for kind in ("clip", "trend"):
        assert kind in COMMAND_DETAILS
        assert COMMAND_DETAILS[kind]["usage"].startswith(f"/{kind}")
        assert kind in LIST_ONELINERS
    groups = [name for name, members in LIST_GROUPS
              if "clip" in members or "trend" in members]
    assert groups == ["media system — music · playback · video · podcast"]
    runtime_src = Path(__file__).resolve().parents[1] / \
        "nomorals" / "agents" / "partner" / "runtime.py"
    text = runtime_src.read_text(encoding="utf-8")
    assert 'if kind == "clip":' in text
    assert 'if kind == "trend":' in text


# ── publisher sweep uses the pre-rendered clip ─────────────────────────

class _FakeCalendarPipeline:
    """The sweep's pipeline seam: a real ContentCalendar, plan/run that
    must never be called for a trendclip post."""

    def __init__(self, tmp_path):
        from nomorals.media.contentops.pipeline import ContentCalendar
        self.runs_root = tmp_path
        self.calendar = ContentCalendar(
            tmp_path, get_niche=lambda name: {"posts_per_day": 5})

    def plan(self, niche, topic, *, platforms=None, style=""):
        raise AssertionError("plan() must not run for a trendclip post")

    def run(self, job):
        raise AssertionError("run() must not run for a trendclip post")


class _FakeLedger:
    def __init__(self):
        self.entries = []

    def record(self, *, platform, file, title, status, notes="",
               platform_video_id="", **kwargs):
        entry = {"id": f"led{len(self.entries)}", "platform": platform,
                 "file": file, "title": title, "status": status,
                 "notes": notes}
        self.entries.append(entry)
        return entry


class _FakeYouTube:
    def __init__(self):
        self.seen_video = None

    def publish(self, video_path, title, *, description="", tags=None,
                publish_at=None, privacy="public", confirmed=True,
                ledger=None):
        self.seen_video = video_path
        entry = ledger.record(platform="youtube", file=video_path,
                              title=title, platform_video_id="vid123",
                              status="uploaded")
        return {"video_id": "vid123", "ledger_id": entry["id"]}


def _an_hour_ago():
    return (datetime.now(UTC) - timedelta(hours=1)).isoformat()


def test_sweep_uses_prerendered_trendclip_clip(tmp_path, monkeypatch):
    from nomorals.agents.shorts_publisher import publish_due_shorts

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-clip-bytes")
    monkeypatch.setattr(
        "nomorals.media.trendclip.pipeline.prerendered_clip_for_job",
        lambda root, job_id: str(clip))
    monkeypatch.setattr(
        "nomorals.media.trendclip.pipeline.trendclip_post_copy",
        lambda root, job_id: {"title": "A great moment",
                              "description": "watch to the end",
                              "hashtags": ["gta6"],
                              "script": "0s: hello world"})
    pipe = _FakeCalendarPipeline(tmp_path)
    post = pipe.calendar.add("trendclip", "GTA 6 — the hook lands",
                             platforms=["youtube"],
                             scheduled_for=_an_hour_ago())
    pipe.calendar.mark(post.id, "queued", job_id="tc_test123")
    publisher = _FakeYouTube()
    outcomes = publish_due_shorts(
        pipeline=pipe, db=None, ledger=_FakeLedger(),
        publisher_for=lambda platform: publisher)
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "posted"
    assert publisher.seen_video == str(clip)
    assert pipe.calendar.get(post.id).status == "posted"


def test_sweep_still_renders_a_normal_post(tmp_path):
    from nomorals.agents.shorts_publisher import publish_due_shorts

    class _RenderingPipeline(_FakeCalendarPipeline):
        def plan(self, niche, topic, *, platforms=None, style=""):
            return SimpleNamespace(id="job_fake1", run_id="run_fake1",
                                   topic=topic, niche=niche)

        def run(self, job):
            run_dir = self.runs_root / job.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            video = run_dir / "final.mp4"
            video.write_bytes(b"fake-mp4-bytes")
            (run_dir / "post_copy.json").write_text(json.dumps({
                "title": "Three hooks that land",
                "description": "watch to the end",
                "hashtags": ["hooks"]}), encoding="utf-8")
            return SimpleNamespace(ok=True, final_path=str(video),
                                   error=None, job_id=job.id,
                                   run_id=job.run_id)

    pipe = _RenderingPipeline(tmp_path)
    post = pipe.calendar.add("motivation", "morning hooks that land",
                             platforms=["youtube"],
                             scheduled_for=_an_hour_ago())
    outcomes = publish_due_shorts(
        pipeline=pipe, db=None, ledger=_FakeLedger(),
        publisher_for=lambda platform: _FakeYouTube())
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "posted"
    assert pipe.calendar.get(post.id).status == "posted"
