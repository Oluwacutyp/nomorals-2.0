"""Tests for Animato Studio series templates (port step 3).

The four genres as pipelines over contentops: each template is a
brief-shape + scene-shape + voice/music preset that turns a seed into
queued topics in a StudioProject's topic_queue, shaped so the step-2
auto-post mission picks them up and renders them through the existing
spark → build → share flow. The render and publish steps run against
fakes at the organ boundaries (same shapes as test_animato_studio.py /
test_animato_autopost.py).
"""

from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.media.contentops import _shims  # noqa: E402
from nomorals.media.contentops.autopost import (  # noqa: E402
    AutopostStore,
    run_autopost,
)
from nomorals.media.contentops.niches import get as get_niche  # noqa: E402
from nomorals.media.contentops.niches import validate  # noqa: E402
from nomorals.media.contentops.pipeline import Job, ShortPipeline  # noqa: E402
from nomorals.media.contentops.studio import StudioRunner, StudioStore  # noqa: E402
from nomorals.media.contentops.templates import (  # noqa: E402
    MAX_TOPICS_PER_APPLY,
    TEMPLATES,
    apply_template,
    get_template,
    list_templates,
)

# ── fakes (same shapes as test_animato_studio.py) ────────────────────


class FakeJob:
    def __init__(self, job_id: str, topic: str):
        self.id = job_id
        self.topic = topic
        self.run_id = ""


class FakeResult:
    def __init__(self, ok: bool, run_id: str, job_id: str,
                 final_path: str | None, error: str = ""):
        self.ok = ok
        self.run_id = run_id
        self.job_id = job_id
        self.final_path = final_path
        self.error = (SimpleNamespace(to_dict=lambda: {"message": error})
                      if error else None)


class FakePipeline:
    """ShortPipeline-shaped fake: plan → preflight → run, all in-memory."""

    def __init__(self, root: Path, *, preflight_ok: bool = True,
                 render_ok: bool = True):
        self.runs_root = root
        self.calendar = None
        self.preflight_ok = preflight_ok
        self.render_ok = render_ok
        self.planned: list[dict] = []
        self._n = 0

    def plan(self, niche, topic, *, style="", footage="", voice="",
             music=""):
        self._n += 1
        job = FakeJob(f"job_fake{self._n}", topic)
        self.planned.append({"niche": niche, "topic": topic, "style": style,
                             "footage": footage, "voice": voice,
                             "music": music})
        return job

    def preflight(self, job):
        return {"ok": True, "checks": [
            {"name": "ffmpeg", "ok": True, "detail": "/usr/bin/ffmpeg"}]}

    def run(self, job):
        job.run_id = f"run_fake{self._n}"
        run_dir = Path(self.runs_root) / job.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        final = run_dir / "final.mp4"
        if not self.render_ok:
            return FakeResult(False, job.run_id, job.id, None, "render exploded")
        final.write_bytes(b"fake-mp4")
        (run_dir / "post_copy.json").write_text(json.dumps(
            {"title": f"TITLE: {job.topic}",
             "description": "a test episode",
             "hashtags": ["#test"]}), encoding="utf-8")
        return FakeResult(True, job.run_id, job.id, str(final))


class FakePublisher:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.published: list[dict] = []

    def publish(self, path, title, *, description="", tags=None,
                privacy="public", confirmed=False, **kw):
        if self.fail:
            raise RuntimeError("YouTube credentials missing")
        self.published.append({"path": path, "title": title,
                               "privacy": privacy})
        return {"video_id": "yt_fake123",
                "url": "https://youtube.com/watch?v=yt_fake123"}


class FakeEdit:
    """Edit contract with a recording music-bed maker."""

    EditSpec = None
    render = None
    detect_beats = None
    build_captions = None

    def __init__(self, *, reject_roots: bool = False):
        self.reject_roots = reject_roots
        self.bed_calls: list[dict] = []

    def make_music_bed(self, duration_s, out_path, seed=7, roots=None):
        if self.reject_roots and roots is not None:
            raise TypeError("make_music_bed() got an unexpected "
                            "keyword argument 'roots'")
        self.bed_calls.append({"duration_s": duration_s,
                               "out_path": out_path, "seed": seed,
                               "roots": roots})
        Path(out_path).write_bytes(b"fake-bed")
        return out_path


def _runner(tmp_path, **kw):
    return StudioRunner(tmp_path / "studio",
                        pipeline=kw.pop("pipeline", None),
                        publisher_factory=kw.pop("publisher_factory", None))


def _cli_args(action, **kw):
    args = SimpleNamespace(animato_action=action, json=False,
                           project="", name="", idea="", topic="",
                           episode="", niche="did_you_know", style="",
                           voice="edge", music="", channel="",
                           schedule="", publish="", private=False,
                           genre="", seed="", count=1,
                           autopost_action="")
    args.__dict__.update(kw)
    return args


# ── the registry ─────────────────────────────────────────────────────


def test_four_templates_in_genre_order():
    templates = list_templates()
    assert [t.id for t in templates] == ["stories", "news", "music", "cooking"]
    assert set(TEMPLATES) == {"stories", "news", "music", "cooking"}
    for template in templates:
        assert template.name.strip()
        assert template.description.strip()
        assert template.niche.strip()
        assert template.style.strip()
        assert template.voice.strip()
        assert template.music.strip()
        assert template.to_dict()["id"] == template.id


def test_template_presets_are_real_render_knobs():
    expected = {
        "stories": ("stories", "minimal", "cinematic"),
        "news": ("news", "documentary", "ambient"),
        "music": ("music_videos", "phonk", "phonk"),
        "cooking": ("cooking", "vlog", "upbeat"),
    }
    for template_id, (niche, style, music) in expected.items():
        template = get_template(template_id)
        assert (template.niche, template.style, template.music) == (
            niche, style, music)
        # The niche resolves and validates: the genre's brief-shape
        # and scene-shape are real, not a name pointing nowhere.
        plugin = get_niche(niche)
        assert validate(plugin) == []
        assert plugin.default_style == style


def test_get_template_unknown_names_available():
    with pytest.raises(ValueError, match="unknown template 'dance'"):
        get_template("dance")
    with pytest.raises(ValueError, match="stories, news, music, cooking"):
        get_template("nope")
    assert get_template("  Stories ").id == "stories"


# ── seed shaping ─────────────────────────────────────────────────────


def test_single_seed_shapes_one_topic_naming_the_seed():
    cases = {
        "stories": "a lighthouse keeper who hears the sea talking",
        "news": "gold hit a record high overnight",
        "music": "a midnight drive out of a city that never loved you",
        "cooking": "jollof rice in one pan",
    }
    for template_id, seed in cases.items():
        topics = get_template(template_id).topics(seed)
        assert topics == [seed], template_id


def test_single_seed_count_yields_a_part_arc():
    stories = get_template("stories").topics(
        "a lighthouse keeper", count=3)
    assert stories == ["a lighthouse keeper",
                       "a lighthouse keeper (part 2: the twist)",
                       "a lighthouse keeper (part 3: the ending)"]
    news = get_template("news").topics("gold hit a record high", count=3)
    assert news == ["gold hit a record high",
                    "gold hit a record high — why it matters",
                    "gold hit a record high — what happens next"]
    music = get_template("music").topics("midnight drive", count=3)
    assert music == ["midnight drive",
                     "midnight drive (the chorus)",
                     "midnight drive (the bridge)"]
    cooking = get_template("cooking").topics("jollof rice", count=3)
    assert cooking == ["jollof rice",
                       "jollof rice — the secret technique",
                       "jollof rice — the twist"]
    # Every part names the seed verbatim — nothing invented beyond it.
    for topics in (stories, news, music, cooking):
        assert all("a lighthouse keeper" in t or "gold hit" in t
                   or "midnight drive" in t or "jollof rice" in t
                   for t in topics)


def test_multi_item_seed_queues_one_episode_per_item():
    topics = get_template("cooking").topics(
        "jollof rice, fried rice; ofada stew\npepper soup", count=10)
    assert topics == ["jollof rice", "fried rice", "ofada stew",
                      "pepper soup"]
    # Count caps a long list; empty items are dropped, not queued.
    assert get_template("news").topics("a, b, c", count=2) == ["a", "b"]
    assert get_template("news").topics("a,, ,b", count=5) == ["a", "b"]


def test_seed_shaping_fails_closed():
    template = get_template("stories")
    with pytest.raises(ValueError, match="give a seed"):
        template.topics("")
    with pytest.raises(ValueError, match="give a seed"):
        template.topics("  , ; \n ")
    with pytest.raises(ValueError, match="count must be 1"):
        template.topics("a lighthouse keeper", count=0)
    with pytest.raises(ValueError, match="count must be 1"):
        template.topics("a lighthouse keeper",
                        count=MAX_TOPICS_PER_APPLY + 1)
    with pytest.raises(ValueError, match="count must be a whole number"):
        template.topics("a lighthouse keeper", count="three")


# ── apply ────────────────────────────────────────────────────────────


def _project(tmp_path, name="Stories", **fields):
    store = StudioStore(tmp_path / "studio")
    return store, store.create(name, **fields)


def test_apply_sets_genre_presets_and_queues_topics(tmp_path):
    store, _ = _project(tmp_path)
    result = apply_template(store, "stories", "stories",
                            "a lighthouse keeper", count=3)
    assert result["ok"] is True
    assert result["template"] == "stories"
    assert result["topics"] == [
        "a lighthouse keeper",
        "a lighthouse keeper (part 2: the twist)",
        "a lighthouse keeper (part 3: the ending)"]
    assert result["queued"] == 3
    assert result["changes"]["niche"] == "did_you_know → stories"
    assert result["changes"]["style"] == "(none) → minimal"
    assert result["changes"]["music"] == "(none) → cinematic"
    # Persisted: a reload shows the genre and the queue.
    project = store.load("stories")
    assert project.niche == "stories"
    assert project.style == "minimal"
    assert project.voice == "edge"
    assert project.music == "cinematic"
    assert project.topic_queue == result["topics"]


def test_apply_reapplying_the_same_genre_reports_no_preset_moves(tmp_path):
    store, _ = _project(tmp_path)
    apply_template(store, "news", "stories", "gold hit a record high")
    second = apply_template(store, "news", "stories", "markets rallied")
    assert second["changes"] == {}
    assert second["topics"] == ["markets rallied"]
    assert second["queued"] == 2


def test_apply_fails_closed(tmp_path):
    store, _ = _project(tmp_path)
    with pytest.raises(ValueError, match="unknown template 'dance'"):
        apply_template(store, "dance", "stories", "a seed")
    with pytest.raises(ValueError, match="give a seed"):
        apply_template(store, "stories", "stories", "")
    with pytest.raises(FileNotFoundError, match="unknown studio project"):
        apply_template(store, "stories", "ghost", "a seed")


# ── music-bed presets ────────────────────────────────────────────────


def test_bed_roots_known_presets_distinct_unknown_falls_back():
    presets = ("ambient", "cinematic", "lofi", "phonk", "upbeat",
               "tense", "gentle")
    roots = {p: _shims.bed_roots(p) for p in presets}
    assert len({tuple(v) for v in roots.values()}) == len(presets)
    assert _shims.bed_roots("no-such-preset") == _shims.bed_roots("ambient")
    assert _shims.bed_roots("") == _shims.bed_roots("ambient")
    assert _shims.bed_roots("  PHONK ") == _shims.bed_roots("phonk")


def test_make_music_bed_preset_changes_the_audio(tmp_path):
    out_a = str(tmp_path / "a.wav")
    out_b = str(tmp_path / "b.wav")
    _shims.make_music_bed(1.0, out_a, seed=7,
                          roots=_shims.bed_roots("phonk"))
    _shims.make_music_bed(1.0, out_b, seed=7,
                          roots=_shims.bed_roots("upbeat"))
    assert Path(out_a).stat().st_size > 0
    assert Path(out_a).read_bytes() != Path(out_b).read_bytes()
    # Same preset + same seed is deterministic.
    out_c = str(tmp_path / "c.wav")
    _shims.make_music_bed(1.0, out_c, seed=7,
                          roots=_shims.bed_roots("phonk"))
    assert Path(out_a).read_bytes() == Path(out_c).read_bytes()


def test_job_music_round_trip():
    job = Job(id="job1", niche="stories", topic="t", music="cinematic")
    restored = Job.from_dict(job.to_dict())
    assert restored.music == "cinematic"
    assert Job.from_dict({"id": "job2", "niche": "news",
                          "topic": "t"}).music == ""


def test_stage_music_uses_job_preset_and_records_it(tmp_path, monkeypatch):
    import nomorals.media.contentops.pipeline as pipeline_mod

    run_dir = tmp_path / "runs" / "run1"
    run_dir.mkdir(parents=True)
    vo_path = run_dir / "voiceover.wav"
    with wave.open(str(vo_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 22050 * 2)
    (run_dir / "voiceover_meta.json").write_text(json.dumps(
        {"path": str(vo_path), "duration_s": 2.0}), encoding="utf-8")
    edit = FakeEdit()
    pipe = ShortPipeline(tmp_path / "runs", edit=edit)
    ffmpeg_calls: list[tuple] = []
    monkeypatch.setattr(pipeline_mod, "_ffmpeg",
                        lambda *args, **kw: ffmpeg_calls.append(args))
    job = Job(id="job1", niche="music", topic="midnight drive",
              music="phonk")
    pipe._stage_music({"run_dir": run_dir, "manifest": {"run_id": "run1"},
                       "job": job})
    assert len(edit.bed_calls) == 1
    assert edit.bed_calls[0]["roots"] == _shims.bed_roots("phonk")
    meta = json.loads((run_dir / "music_meta.json").read_text(
        encoding="utf-8"))
    assert meta["preset"] == "phonk"
    assert ffmpeg_calls  # the bed was mixed under the voiceover


def test_stage_music_falls_back_for_a_bed_maker_without_roots(
        tmp_path, monkeypatch):
    import nomorals.media.contentops.pipeline as pipeline_mod

    run_dir = tmp_path / "runs" / "run1"
    run_dir.mkdir(parents=True)
    vo_path = run_dir / "voiceover.wav"
    with wave.open(str(vo_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 22050 * 2)
    (run_dir / "voiceover_meta.json").write_text(json.dumps(
        {"path": str(vo_path), "duration_s": 2.0}), encoding="utf-8")
    edit = FakeEdit(reject_roots=True)
    pipe = ShortPipeline(tmp_path / "runs", edit=edit)
    monkeypatch.setattr(pipeline_mod, "_ffmpeg", lambda *args, **kw: None)
    job = Job(id="job1", niche="stories", topic="t", music="cinematic")
    out = pipe._stage_music({"run_dir": run_dir,
                             "manifest": {"run_id": "run1"}, "job": job})
    assert out["music_bed.wav"] == "music_bed.wav"
    assert len(edit.bed_calls) == 1
    assert edit.bed_calls[0]["roots"] is None  # retried without roots
    meta = json.loads((run_dir / "music_meta.json").read_text(
        encoding="utf-8"))
    assert meta["preset"] == "cinematic"


# ── mission consumes a templated topic end to end ────────────────────


def test_mission_renders_a_templated_topic_with_genre_knobs(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    publisher = FakePublisher()
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: publisher)
    store = AutopostStore(tmp_path / "studio")
    runner.store.create("Stories", schedule="daily")
    apply_template(runner.store, "stories", "stories",
                   "a lighthouse keeper who hears the sea talking")
    result = run_autopost(runner=runner, store=store, force=True)
    assert result["posted"] == 1
    assert result["outcomes"][0]["status"] == "posted"
    assert result["outcomes"][0]["topic"] == (
        "a lighthouse keeper who hears the sea talking")
    assert len(pipeline.planned) == 1
    planned = pipeline.planned[0]
    assert planned["niche"] == "stories"
    assert planned["style"] == "minimal"
    assert planned["voice"] == "edge"
    assert planned["music"] == "cinematic"
    # The queue was consumed; the episode is on the manifest.
    assert runner.store.load("stories").topic_queue == []
    assert len(publisher.published) == 1


# ── CLI ──────────────────────────────────────────────────────────────


def test_cli_templates_list_and_template_apply(tmp_path, monkeypatch, capsys):
    import nomorals.media.contentops.studio as studio_mod
    from nomorals.cmdline.commands.animato import cmd_animato

    monkeypatch.setattr(studio_mod, "DEFAULT_STUDIO_ROOT",
                        tmp_path / "studio")
    assert cmd_animato(_cli_args("templates"), None) == 0
    out = capsys.readouterr().out
    for template_id in ("stories", "news", "music", "cooking"):
        assert template_id in out
    assert cmd_animato(_cli_args("templates", json=True), None) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [t["id"] for t in listed] == ["stories", "news", "music",
                                         "cooking"]
    assert cmd_animato(_cli_args("new", name="Kitchen"), None) == 0
    assert cmd_animato(_cli_args(
        "template", genre="cooking", name="kitchen",
        seed="jollof rice", count=2), None) == 0
    out = capsys.readouterr().out
    assert "applied cooking to kitchen" in out
    assert "jollof rice" in out
    assert "the secret technique" in out
    reloaded = StudioStore(tmp_path / "studio").load("kitchen")
    assert reloaded.niche == "cooking"
    assert reloaded.music == "upbeat"
    assert reloaded.topic_queue == ["jollof rice",
                                    "jollof rice — the secret technique"]


def test_cli_template_failures_are_plain_text(tmp_path, monkeypatch, capsys):
    import nomorals.media.contentops.studio as studio_mod
    from nomorals.cmdline.commands.animato import cmd_animato

    monkeypatch.setattr(studio_mod, "DEFAULT_STUDIO_ROOT",
                        tmp_path / "studio")
    assert cmd_animato(_cli_args("new", name="Kitchen"), None) == 0
    assert cmd_animato(_cli_args(
        "template", genre="dance", name="kitchen",
        seed="jollof rice"), None) == 1
    assert "unknown template" in capsys.readouterr().err
    assert cmd_animato(_cli_args(
        "template", genre="cooking", name="ghost",
        seed="jollof rice"), None) == 1
    assert "unknown studio project" in capsys.readouterr().err
    assert cmd_animato(_cli_args(
        "template", genre="cooking", name="kitchen",
        seed="   "), None) == 1
    assert "give a seed" in capsys.readouterr().err
