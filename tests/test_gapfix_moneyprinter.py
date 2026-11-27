"""Tests for the MoneyPrinterTurbo port (gapfix moneyprinter).

Stock footage (Pexels/Pixabay), Edge TTS voiceover, stock visuals,
the make_short factory, the make_short tool, and the /shorts make
flags. Network is faked throughout; renders use the fake edit
contract from test_contentops_pipeline (real ffmpeg testsrc only).
"""

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from test_contentops_pipeline import (  # noqa: E402
    SCRIPT_FIXTURE,
    FakeBrain,
    FakeNiche,
    FakeStudio,
    FakeTTS,
    make_fake_edit,
    make_pipeline,
)

from nomorals.agents.partner.runtime_media import RuntimeMediaMixin  # noqa: E402
from nomorals.agents.shorts_publisher import publish_video  # noqa: E402
from nomorals.media.contentops.factory import (  # noqa: E402
    factory_status,
    make_short,
    retime_captions_with_whisper,
)
from nomorals.media.contentops.pipeline import (  # noqa: E402
    ShortPipeline,
    parse_script,
)
from nomorals.media.contentops.stock import (  # noqa: E402
    extract_poster,
    fetch_stock_clip,
    stock_status,
)
from nomorals.tools.registry import ToolRegistry  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


# ── stock fakes ────────────────────────────────────────────────────

PEXELS_PAYLOAD = {"videos": [
    {"id": 11, "duration": 9, "url": "https://www.pexels.com/video/11",
     "image": "https://images.pexels.com/11.jpg",
     "video_files": [
         {"link": "https://videos.pexels.com/11-landscape.mp4",
          "width": 1920, "height": 1080, "file_type": "video/mp4"},
         {"link": "https://videos.pexels.com/11-portrait.mp4",
          "width": 1080, "height": 1920, "file_type": "video/mp4"},
     ]},
]}

PIXABAY_PAYLOAD = {"hits": [
    {"id": 22, "duration": 7, "pageURL": "https://pixabay.com/videos/22",
     "videos": {
         "large": {"url": "https://cdn.pixabay.com/22.mp4",
                   "width": 720, "height": 1280},
         "medium": {"url": "https://cdn.pixabay.com/22-m.mp4",
                    "width": 480, "height": 854}}},
]}


_USE_DEFAULT = object()


class FakeStockClient:
    """Canned Pexels/Pixabay search + a byte-writing download."""

    def __init__(self, pexels=_USE_DEFAULT, pixabay=_USE_DEFAULT):
        self.pexels = PEXELS_PAYLOAD if pexels is _USE_DEFAULT else pexels
        self.pixabay = PIXABAY_PAYLOAD if pixabay is _USE_DEFAULT else pixabay
        self.searches: list[str] = []
        self.downloads: list[str] = []

    def get_json(self, url, *, headers=None, params=None):
        if "pexels" in url:
            self.searches.append("pexels")
            if self.pexels is None:
                raise RuntimeError("pexels down (simulated)")
            return self.pexels
        self.searches.append("pixabay")
        if self.pixabay is None:
            raise RuntimeError("pixabay down (simulated)")
        return self.pixabay

    def download(self, url, dest):
        self.downloads.append(url)
        Path(dest).write_bytes(b"FAKE-MP4" * 100)
        return Path(dest)


def _set_stock_keys(monkeypatch, pexels=True, pixabay=True):
    if pexels:
        monkeypatch.setenv("NM_PEXELS_API_KEY", "fake-pexels-key")
    else:
        monkeypatch.delenv("NM_PEXELS_API_KEY", raising=False)
    if pixabay:
        monkeypatch.setenv("NM_PIXABAY_API_KEY", "fake-pixabay-key")
    else:
        monkeypatch.delenv("NM_PIXABAY_API_KEY", raising=False)


# ── stock ──────────────────────────────────────────────────────────


def test_stock_unconfigured_is_honest_failure(monkeypatch, tmp_path):
    _set_stock_keys(monkeypatch, pexels=False, pixabay=False)
    client = FakeStockClient()
    result = fetch_stock_clip("ocean waves", tmp_path, client=client)
    assert result["ok"] is False
    assert "no key" in result["reason"]
    assert client.searches == []  # no network touched at all


def test_stock_status_names_keys(monkeypatch):
    _set_stock_keys(monkeypatch)
    status = stock_status()
    assert status["configured"] == ["pexels", "pixabay"]
    _set_stock_keys(monkeypatch, pixabay=False)
    assert stock_status()["configured"] == ["pexels"]


def test_stock_pexels_picks_portrait_file(monkeypatch, tmp_path):
    _set_stock_keys(monkeypatch)
    client = FakeStockClient()
    result = fetch_stock_clip("ocean waves", tmp_path, client=client)
    assert result["ok"] is True
    assert result["provider"] == "pexels"
    assert result["source_url"] == "https://www.pexels.com/video/11"
    assert client.downloads == [
        "https://videos.pexels.com/11-portrait.mp4"]
    assert Path(result["path"]).exists()


def test_stock_pixabay_fallback_when_pexels_empty(monkeypatch, tmp_path):
    _set_stock_keys(monkeypatch)
    client = FakeStockClient(pexels={"videos": []})
    result = fetch_stock_clip("ocean waves", tmp_path, client=client)
    assert result["ok"] is True
    assert result["provider"] == "pixabay"
    assert client.searches == ["pexels", "pixabay"]
    assert client.downloads == ["https://cdn.pixabay.com/22.mp4"]


def test_stock_prefers_pixabay_when_asked(monkeypatch, tmp_path):
    _set_stock_keys(monkeypatch)
    client = FakeStockClient()
    result = fetch_stock_clip("ocean waves", tmp_path, prefer="pixabay",
                              client=client)
    assert result["ok"] is True
    assert result["provider"] == "pixabay"
    assert client.searches[0] == "pixabay"


def test_stock_provider_error_falls_through(monkeypatch, tmp_path):
    _set_stock_keys(monkeypatch)
    client = FakeStockClient(pexels=None, pixabay=None)
    result = fetch_stock_clip("ocean waves", tmp_path, client=client)
    assert result["ok"] is False
    assert "pexels down" in result["reason"]
    assert "pixabay down" in result["reason"]


def test_stock_oversize_clip_rejected(monkeypatch, tmp_path):
    _set_stock_keys(monkeypatch)
    client = FakeStockClient()
    result = fetch_stock_clip("ocean waves", tmp_path, max_bytes=10,
                              client=client)
    assert result["ok"] is False
    assert "over 10 bytes" in result["reason"]
    assert not list(tmp_path.glob("*.mp4"))


@needs_ffmpeg
def test_extract_poster_midpoint_frame(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-y", "-v", "error",
                    "-f", "lavfi", "-i",
                    "testsrc=size=320x568:rate=15:duration=2.0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
                   check=True, timeout=60)
    out = tmp_path / "poster.png"
    result = extract_poster(clip, out)
    assert result["ok"] is True
    assert out.exists() and out.stat().st_size > 0


# ── edge TTS + stock visuals (full fake runs) ──────────────────────


def _write_wav(out_path: str, secs: float = 2.0) -> dict:
    sr = 22050
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
    return {"path": out_path}


def _fake_edge(text, out_path):
    return _write_wav(out_path)


def _make_clip(tmp_path: Path, name: str = "stock.mp4") -> str:
    clip = tmp_path / name
    subprocess.run([shutil.which("ffmpeg"), "-y", "-v", "error",
                    "-f", "lavfi", "-i",
                    "testsrc=size=320x568:rate=15:duration=2.0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
                   check=True, timeout=60)
    return str(clip)


@needs_ffmpeg
def test_pipeline_edge_voiceover_first(tmp_path):
    pipe = make_pipeline(tmp_path, edge_speak=_fake_edge)
    job = pipe.plan("did_you_know", "the brain edits reality",
                    voice="edge")
    result = pipe.run(job)
    assert result.ok, result.error
    meta = json.loads((tmp_path / "runs" / result.run_id
                       / "voiceover_meta.json").read_text())
    assert meta["voice"] == "edge-tts"


@needs_ffmpeg
def test_pipeline_edge_failure_falls_back_to_catalogue(tmp_path):
    def _boom(text, out_path):
        raise RuntimeError("edge-tts down (simulated)")

    pipe = make_pipeline(tmp_path, edge_speak=_boom)
    job = pipe.plan("did_you_know", "the brain edits reality",
                    voice="edge")
    result = pipe.run(job)
    assert result.ok, result.error
    meta = json.loads((tmp_path / "runs" / result.run_id
                       / "voiceover_meta.json").read_text())
    assert meta["voice"] == "narrator"


@needs_ffmpeg
def test_pipeline_stock_visuals_record_source(tmp_path):
    clip = _make_clip(tmp_path)

    def _stock(query, out_dir):
        return {"ok": True, "path": clip, "provider": "pexels",
                "source_url": "https://www.pexels.com/video/11",
                "clip_id": "11", "width": 1080, "height": 1920,
                "duration": 2.0, "query": query}

    pipe = make_pipeline(tmp_path, stock_fetch=_stock)
    job = pipe.plan("did_you_know", "the brain edits reality",
                    footage="stock")
    result = pipe.run(job)
    assert result.ok, result.error
    frames = json.loads((tmp_path / "runs" / result.run_id
                         / "frames.json").read_text())
    assert len(frames) == 3
    assert all(f["source"] == "stock" for f in frames)
    assert all(f["provider"] == "pexels" for f in frames)
    assert all(f["source_url"].startswith("https://www.pexels.com")
               for f in frames)
    assert all(Path(f["path"]).exists() for f in frames)
    assert all(f["clip"] == clip for f in frames)


@needs_ffmpeg
def test_pipeline_stock_miss_falls_back_to_stills(tmp_path):
    def _miss(query, out_dir):
        return {"ok": False, "reason": "no stock footage: no key"}

    pipe = make_pipeline(tmp_path, stock_fetch=_miss)
    job = pipe.plan("did_you_know", "the brain edits reality",
                    footage="stock")
    result = pipe.run(job)
    assert result.ok, result.error
    frames = json.loads((tmp_path / "runs" / result.run_id
                         / "frames.json").read_text())
    assert all(f["source"] == "stills" for f in frames)


def test_job_fields_and_plan_passthrough(tmp_path):
    pipe = ShortPipeline(runs_root=str(tmp_path / "runs"),
                         brain=FakeBrain(), tts=FakeTTS(),
                         studio=FakeStudio(), edit=make_fake_edit(),
                         get_niche=lambda name: FakeNiche())
    job = pipe.plan("did_you_know", "topic", footage="stock",
                    voice="edge")
    assert job.footage == "stock"
    assert job.voice == "edge"
    plain = pipe.plan("did_you_know", "topic")
    assert plain.footage == "" and plain.voice == ""


# ── factory ────────────────────────────────────────────────────────


def _fake_transcriber(script_text=SCRIPT_FIXTURE, total_s: float = 2.0):
    """Perfect transcription of a script's narration, uniform timing.

    The words must be speakable inside ``total_s`` at a realistic
    rate — the caption engine floors words at 50 ms, so a 47-word
    fixture crammed into a 2 s fake voiceover (23 words/sec) fails
    validation honestly. The factory test uses SHORT_SCRIPT for this
    reason; the fixture's full script stays for the pipeline tests.
    """
    parsed = parse_script(script_text)
    texts = [parsed.get("hook", "")] + [s.get("say", "")
                                        for s in parsed["scenes"]]
    words = [w for t in texts for w in t.split()]

    def fake(audio, *, language="en", model=""):
        per = total_s / max(len(words), 1)
        return [{"text": w, "start": round(i * per, 3),
                 "end": round((i + 1) * per, 3)}
                for i, w in enumerate(words)]

    return fake


SHORT_SCRIPT = """HOOK: Short hook.
[SCENE 1]
SAY: Cats sleep all day long.
SHOW: a sleeping cat
TITLE: Cats
DESCRIPTION: d
"""


def _stock_miss(query, out_dir):
    return {"ok": False, "reason": "no stock footage: no key"}


@needs_ffmpeg
def test_factory_full_run_names_every_rung(tmp_path):
    pipe = make_pipeline(tmp_path, brain=FakeBrain(SHORT_SCRIPT),
                         edge_speak=_fake_edge,
                         stock_fetch=_stock_miss)
    out = make_short("cats sleep all day", pipeline=pipe,
                     transcriber=_fake_transcriber(SHORT_SCRIPT))
    assert out["ok"] is True, out
    assert Path(out["path"]).exists()
    assert out["rungs"]["voiceover"] == "edge-tts"
    assert out["rungs"]["visuals"] == "stills"
    assert out["rungs"]["captions"] == "whisper"
    # Publishing is the tool's rung, not the factory's — the factory
    # returns no publish rung at all.
    assert "publish" not in out["rungs"]
    assert any("stock footage unavailable" in n for n in out["notes"])
    assert any("Whisper" in n for n in out["notes"])
    captions = json.loads((tmp_path / "runs" / out["run_id"]
                           / "captions_meta.json").read_text())
    assert captions["timings_source"] == "whisper"


@needs_ffmpeg
def test_factory_without_whisper_keeps_pipeline_captions(tmp_path):
    pipe = make_pipeline(tmp_path, edge_speak=_fake_edge,
                         stock_fetch=_stock_miss)
    out = make_short("the brain edits reality", pipeline=pipe)
    assert out["ok"] is True, out
    # faster-whisper is not installed in this env — the pipeline's
    # estimated timings stand and the note says so.
    assert out["rungs"]["captions"] in ("estimated", "tts")
    assert any("captions kept" in n for n in out["notes"])


@needs_ffmpeg
def test_factory_run_failure_is_clean(tmp_path):
    pipe = make_pipeline(tmp_path, brain=FakeBrain(fail=True))
    out = make_short("the brain edits reality", pipeline=pipe)
    assert out["ok"] is False
    assert out["error"]["stage"] == "script"


def test_factory_empty_topic(tmp_path):
    out = make_short("")
    assert out["ok"] is False
    assert "topic is required" in out["error"]


def test_publish_video_rejects_unknown_platform(tmp_path):
    result = publish_video(tmp_path / "final.mp4", "myspace")
    assert result["ok"] is False
    assert "unknown platform" in result["reason"]


def test_publish_video_needs_file(tmp_path):
    result = publish_video(tmp_path / "final.mp4", "tiktok")
    assert result["ok"] is False
    assert "video file missing" in result["reason"]


def test_factory_status_shape():
    status = factory_status()
    assert set(status) >= {"edge_tts", "whisper", "stock",
                           "publish_platforms", "notes"}
    assert status["publish_platforms"] == ["tiktok", "youtube", "x"]


@needs_ffmpeg
def test_retime_without_run_artifacts(tmp_path):
    result = retime_captions_with_whisper(tmp_path)
    assert result["ok"] is False
    assert "missing" in result["reason"]


# ── tool + chat ────────────────────────────────────────────────────


def test_make_short_tool_registered():
    registry = ToolRegistry()
    registry.register_builtins()
    assert len(registry._tools) == 811
    assert "make_short" in registry._tools
    spec = registry._tools["make_short"]
    assert "Edge TTS" in spec.description
    assert registry.failed_modules() == []


def _fake_tool_factory(tmp_path):
    fake_final = tmp_path / "final.mp4"
    fake_final.write_bytes(b"fake")

    def _fake(topic, **kw):
        return {"ok": True, "path": str(fake_final), "run_id": "run_x",
                "job_id": "job_x",
                "rungs": {"script": "brain", "voiceover": "edge-tts",
                          "visuals": "stills", "captions": "estimated",
                          "music": "composed"}, "notes": []}

    return _fake


def test_tool_publish_rung_reports_unknown_platform(monkeypatch, tmp_path):
    import nomorals.tools.shorts as shorts_tool

    monkeypatch.setattr(shorts_tool, "_make_short",
                        _fake_tool_factory(tmp_path))
    out = shorts_tool.make_short("cats sleep", publish="myspace")
    assert out["ok"] is True
    assert out["rungs"]["publish"].startswith("not posted: unknown platform")


def test_tool_publish_rung_posts_through_publisher(monkeypatch, tmp_path):
    import nomorals.tools.shorts as shorts_tool

    monkeypatch.setattr(shorts_tool, "_make_short",
                        _fake_tool_factory(tmp_path))

    def _fake_publish(video_path, platform, **kw):
        return {"ok": True, "platform": platform, "video_id": "vid1"}

    monkeypatch.setattr(
        "nomorals.agents.shorts_publisher.publish_video", _fake_publish)
    out = shorts_tool.make_short("cats sleep", publish="tiktok")
    assert out["rungs"]["publish"] == "posted"
    assert "posted to tiktok" in out["notes"]


def test_tool_publish_rung_skipped_without_platform(monkeypatch, tmp_path):
    import nomorals.tools.shorts as shorts_tool

    monkeypatch.setattr(shorts_tool, "_make_short",
                        _fake_tool_factory(tmp_path))
    out = shorts_tool.make_short("cats sleep")
    assert out["rungs"]["publish"] == "skipped (no platform asked)"


def test_shorts_make_flags_parse(monkeypatch):
    seen: dict = {}

    class FakePipe:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self, niche, topic, *, style="", footage="",
                 voice=""):
            seen.update(niche=niche, topic=topic, style=style,
                        footage=footage, voice=voice)
            return SimpleNamespace(id="job_test1")

    monkeypatch.setattr(
        "nomorals.media.contentops.pipeline.ShortPipeline", FakePipe)
    self = SimpleNamespace()
    text = RuntimeMediaMixin._control_shorts(
        self, 'make did_you_know "the brain edits reality" '
              '--footage stock --voice edge')
    assert "queued job_test1" in text
    assert seen["footage"] == "stock"
    assert seen["voice"] == "edge"
    assert seen["topic"] == "the brain edits reality"


def test_shorts_make_flags_default_to_classic(monkeypatch):
    seen: dict = {}

    class FakePipe:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self, niche, topic, *, style="", footage="",
                 voice=""):
            seen.update(footage=footage, voice=voice)
            return SimpleNamespace(id="job_test2")

    monkeypatch.setattr(
        "nomorals.media.contentops.pipeline.ShortPipeline", FakePipe)
    text = RuntimeMediaMixin._control_shorts(
        SimpleNamespace(), 'make did_you_know "the brain edits reality"')
    assert "queued job_test2" in text
    assert seen["footage"] == ""
    assert seen["voice"] == ""
